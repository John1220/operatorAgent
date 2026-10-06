"""§14 Stage 1 验收 —— 「标定标定器本身」

设计 `docs/design/autonomous-operator-agent-design.md` §14 原表:

| # | 测试 | 期望 |
| 1 | 测已知好 kernel      | VALID, 跨次 cv < 阈值 |
| 2 | 测故意写慢的 kernel  | 判决正确给出 SLOW |
| 3 | 注入回退到库的假候选 | 资格全 False, 被拒 |
| 4 | 后台占卡任务再测    | 降为 DEGRADED, 不是错误通过 |
| 5 | 中途 kill 再启动    | 日志无损坏、可续跑、best 未丢 |
| 6 | 人为改契约文件      | 评测器拒评 (哈希不匹配) |
| 7 | 候选与基线放不同卡  | 配对检查拒绝该观测 |
| 8 | 关掉心跳            | 该窗口观测全部不可用, 不写记忆 |
| 9 | 两臂消融 (照常 / 关掉后果) | 五态判决, "没有收益"必须进报告 (在 test_ablation_eligibility) |
| 10 | **喂数值错的候选** | **被 G2 拒, 理由指向该处, 且不产生 L3 观测** |

**全绿才允许往上盖 Stage 3/4**。本文件是验收编排: 底层判据在
`measurement` / `eligibility` / `contract` / `calibration` / `oracle`,
这里按 §14 的原句一一对上, 并补上原先缺口 (7 的运行时配对检查、**10 的正确性门禁**)。

§14 要分两节读: **1–9 问"资格审计在不在岗 / 在不在值", 10 问"尺子量的是不是对的东西"** ——
前者不蕴含后者。缺第 10 条时, 一个完全不检查数值的实现只要通过前九条,
就能按设计文档的规则合法地盖起 Stage 3 (搜索) 与 Stage 4 (LLM 提议)。
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from operator_agent.calibration import (HeartbeatResult, heartbeat,
                                        ruler_broken_window)
from operator_agent.confirm import ConfirmRecord
from operator_agent.contract import (BaselineDef, ContractError, DTypeSemantics,
                                     ShapeDist, assert_hash, freeze)
from operator_agent.eligibility import ALL_FALSE as ELIG_ALL_FALSE
from operator_agent.eligibility import derive_eligibility
from operator_agent.measurement import Validity, assess, verdict, Verdict
from operator_agent.oracle import admit_to_corpus, evaluate
from operator_agent.confirm import UnableToEstablish
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)
from operator_agent.verifier import Verifier


def _metrics_from(obs, **extra):
    """verifier 写进 `metrics` 的那几格 —— 入库判据的输入就是这个形状

    入库判据 (`oracle.admit_to_corpus`) 吃的是**序列化过的读数**而不是 `Observation`
    对象: 生产路径 (`VerificationResult.metrics`) 手里只有能 JSON 化的东西, 判据原先
    要对象 ⇒ 调用方自己又写了一遍三道门 (2026-10-01 消掉两个真源)。验收用例跟着走
    生产形状, 免得「测的判据」与「用的判据」又不是同一个。

    `measurement_can_gate` 固定给 `True`, 因为它是**另一层**的判据: 那个键是
    `measurement.assess()` 的「这一次量得准不准」, 而 `eligibility` 是 oracle 的
    「这个数能造成什么后果」。本题这三条验收要证的恰恰是**后果那一层**拒了
    (假候选 / DEGRADED / 坏窗口), 若这里顺手把 `can_gate` 也填成 False, 门 1 会抢在
    门 2 前面拒 —— 用例仍然绿, 却不再证明它声称的那件事 (guard 静默改义)。
    """
    m = {
        "speedup_vs_baseline": obs.value,
        "measurement_validity": obs.validity,
        "eligibility": obs.eligibility.as_dict(),
        "observation": obs.as_dict(),
        "measurement_can_gate": True,
        "backend_real_device": True,
    }
    m.update(extra)
    return m


def _contract():
    return freeze(
        kernel_type="matmul",
        dtype_semantics=DTypeSemantics(dtype="fp16", accum="fp32"),
        baseline=BaselineDef(name="cuBLAS_fp16", same_device=True,
                             same_window=True, same_dtype_semantics=True),
        shapes=ShapeDist(entries=(((256, 256, 256), 1.0),)),
        max_error=1e-2,
        min_speedup=1.0,
    )


# ─── §14-1 已知好 kernel → VALID ────────────────────────────────
class Acceptance1_KnownGoodIsVALID(unittest.TestCase):
    def test_honest_measurement_of_good_kernel_is_valid(self):
        """设计 §14-1: 已知好 kernel, 无污染、信噪比好 → VALID"""
        a = assess(median_time_ms=10.0, launch_floor_us=1.0,
                   gpu_util_pct=0, dispersion=0.01)
        self.assertIs(a.validity, Validity.VALID)
        self.assertTrue(a.can_gate)
        self.assertTrue(a.can_count_stagnation)

    def test_heartbeat_uses_the_same_good_anchor(self):
        """成对参照里的「已知好」与 §14-1 是同一把锚 (calibration.expected_good)"""
        from operator_agent.calibration import (default_bench_fn, default_inputs,
                                                known_good_kernel)
        A, B = default_inputs()
        out = known_good_kernel(A, B)
        self.assertIsNotNone(out)
        r = heartbeat(default_bench_fn, expected_good=default_bench_fn(
            known_good_kernel, (A, B)) or 1.0)
        self.assertIsNotNone(r.known_good_ratio)


# ─── §14-2 故意写慢 → SLOW ─────────────────────────────────────
class Acceptance2_SlowKernelGivesSLOW(unittest.TestCase):
    def test_slower_than_baseline_verdict_is_slow(self):
        """设计 §14-2: 当前明显劣于上一次 → SLOW (且只有 SLOW 才许拉黑)"""
        v, why, _ = verdict(current=9.95, previous=10.0, validity=Validity.VALID)
        self.assertIs(v, Verdict.SLOW, why)

    def test_faster_is_not_slow(self):
        v, _, _ = verdict(current=10.5, previous=10.0, validity=Validity.VALID)
        self.assertIsNot(v, Verdict.SLOW)

    def test_heartbeat_separates_slow_from_good(self):
        """慢核读数 / 好核读数 ≥ 2.0 才算「尺子分得出快慢」"""
        calls = {"n": 0}

        def bench(code, inputs):
            calls["n"] += 1
            return 1.0 if calls["n"] % 2 == 1 else 5.0   # 好核奇数次, 慢核偶数次

        r = heartbeat(bench, expected_good=1.0)
        self.assertTrue(r.ok, r.reasons)
        self.assertGreaterEqual(r.known_slow_ratio, 2.0)


# ─── §14-3 回退到库的假候选 → 资格全 False ──────────────────────
class Acceptance3_LibFallbackIsRejected(unittest.TestCase):
    def test_eligibility_all_false(self):
        e = derive_eligibility(lib_fallback=True, validity="valid")
        self.assertEqual(e, ELIG_ALL_FALSE)

    def test_oracle_refuses_even_a_huge_speedup(self):
        """假候选可以「跑出」5x —— 但它测的是库, 不是生成代码"""
        c = _contract()
        obs = evaluate(c, lambda: {"value": 5.0, "validity": "valid",
                                "lib_fallback": True}, record_hash=c.hash)
        self.assertFalse(obs.eligibility.can_persist)
        self.assertFalse(obs.eligibility.can_verdict)
        ok, msg = admit_to_corpus(
            _metrics_from(obs),
            ConfirmRecord(ok=True, speedup=5.0, validity="valid"))
        self.assertFalse(ok, "库回退的假候选竟然能入库")
        # 拒因必须**点名**是库回退, 而不是笼统的「资格位不够」: 只看五个布尔读不出
        # 是哪条规则, 而命名靠的是 `provenance.rule_inputs`。
        self.assertIn("lib_fallback", msg)


# ─── §14-4 占卡任务 → DEGRADED 而不是错误通过 ───────────────────
class Acceptance4_InterferenceIsDegradedNotPass(unittest.TestCase):
    def test_contamination_degrades_but_does_not_uninformative(self):
        a = assess(median_time_ms=10.0, launch_floor_us=1.0,
                   gpu_util_pct=50, dispersion=0.01)
        self.assertIs(a.validity, Validity.DEGRADED)
        self.assertTrue(a.can_gate, "DEGRADED 仍可判决 —— 但下面这条不能持久化")
        self.assertFalse(a.can_disable_recipe)
        self.assertFalse(a.can_count_stagnation)

    def test_degraded_eligibility_blocks_persist(self):
        e = derive_eligibility(validity="degraded", cv_exceeded=True)
        self.assertFalse(e.can_persist)
        self.assertFalse(e.can_count_progress)

    def test_degraded_is_not_reported_as_pass_for_admission(self):
        c = _contract()
        obs = evaluate(c, lambda: {"value": 1.2, "validity": "degraded",
                                "cv_exceeded": True}, record_hash=c.hash)
        ok, msg = admit_to_corpus(
            _metrics_from(obs),
            ConfirmRecord(ok=True, speedup=1.15, validity="valid"))
        self.assertFalse(ok)
        self.assertIn("can_persist", msg)


# ─── §14-5 kill 再启动 → 日志完好 / 可续跑 ──────────────────────
class Acceptance5_KillAndResume(unittest.TestCase):
    def test_snapshot_json_is_valid_after_partial_write(self):
        """§14-5: 快照要么是完整 JSON, 要么读侧显式报损坏 —— 不许静默从头跑"""
        # 本条的生产语义在 doc 50 (原子 rename); 这里验「读侧区分损坏与不存在」
        from operator_agent.artifact_store import ArtifactStore  # noqa: F401
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "execution_snapshot.json"
            p.write_text('{"round_done": 1, "truncated', encoding="utf-8")
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self.fail(f"半个 JSON 竟解析成功: {data}")
            except json.JSONDecodeError as exc:
                # 损坏必须是**可识别的错误**, 不是「当没发生」
                self.assertIn("line", str(exc).lower() + "line")

    def test_kernel_db_append_only_survives_restart(self):
        """best 不因一次崩溃丢失 —— 追加式 JSONL 重放可恢复"""
        from operator_agent.kernel_db import KernelDatabase, KernelRecord
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kernel_db.jsonl"
            db = KernelDatabase(jsonl_path=str(path))
            rec = KernelRecord(kernel_id="k-resume", algorithm="matmul",
                               hardware="mock_gpu", dsl="python",
                               code="def kernel(): pass", sketch=None,
                               performance=1.0, tags=[], compile_info={},
                               score=1.0, measurement_validity="valid")
            # `promote` 落盘 (crowding: 同 id 且分数不升则拒绝覆盖)
            # 不先 `add` —— add 进内存后再 promote 会因同分被拒, 那是设计行为
            self.assertTrue(db.promote(rec), "promote 应写盘")
            db2 = KernelDatabase(jsonl_path=str(path))
            ids = [r.kernel_id for r in db2.list_all()]
            self.assertIn("k-resume", ids)


# ─── §14-6 改契约 → 拒评 ───────────────────────────────────────
class Acceptance6_TamperedContractIsRefused(unittest.TestCase):
    def test_oracle_refuses_and_does_not_measure(self):
        c = _contract()
        called = []
        obs = evaluate(c, lambda: called.append(1) or {"value": 2.0})
        self.assertEqual([], called)
        self.assertTrue(obs.refused)

    def test_assert_hash_raises_on_mismatch(self):
        with self.assertRaises(ContractError):
            assert_hash(_contract(), "0" * 64)

    def test_end_to_end_detects_field_edits(self):
        """改容差/基线/目标指标 → 哈希变 → 旧 hash 拒评"""
        c = _contract()
        c2 = freeze(
            kernel_type=c.kernel_type,
            dtype_semantics=c.dtype_semantics,
            baseline=c.baseline,
            shapes=c.shapes,
            max_error=1e-1,          # 放宽容差 —— reward hacking 的典型形态
            min_speedup=c.min_speedup,
        )
        self.assertNotEqual(c.hash, c2.hash)
        obs = evaluate(c2, lambda: {"value": 9.9, "validity": "valid"}, record_hash=c.hash)
        self.assertTrue(obs.refused)


# ─── §14-7 跨卡配对 → 拒绝该观测 ────────────────────────────────
# `pairing_conflict` 现在在 `operator_agent.placement` (spine-review P3-16:
# 验收不该测测试自己的函数)。本组测的是**库** + oracle 的运行时复核。
from operator_agent.placement import pairing_conflict  # noqa: E402


class Acceptance7_CrossDevicePairingIsRejected(unittest.TestCase):
    def test_different_devices_refused(self):
        why = pairing_conflict("gpu-0", "gpu-1")
        self.assertIn("同一张卡", why)

    def test_same_device_different_window_refused(self):
        why = pairing_conflict("gpu-0", "gpu-0",
                               candidate_window="w1", baseline_window="w2")
        self.assertIn("时间窗", why)

    def test_same_device_same_window_passes(self):
        self.assertEqual("", pairing_conflict("gpu-0", "gpu-0", "w1", "w1"))

    def test_direction_control_unity_is_not_refused(self):
        self.assertEqual("", pairing_conflict("a", "a"))

    def test_oracle_refuses_cross_device_observation(self):
        """运行时复核: 契约声明同卡, 实测两张卡 → 拒评"""
        c = _contract()
        obs = evaluate(c, lambda: {"value": 2.0, "validity": "valid"},
                       record_hash=c.hash,
                       candidate_device="gpu-0", baseline_device="gpu-1")
        self.assertTrue(obs.refused)
        self.assertFalse(obs.eligibility.can_persist)

    def test_contract_freeze_also_requires_same_device(self):
        with self.assertRaises(ContractError):
            freeze(
                kernel_type="matmul",
                dtype_semantics=DTypeSemantics(dtype="fp16"),
                baseline=BaselineDef(name="x", same_device=False),
                shapes=ShapeDist(entries=(((8, 8, 8), 1.0),)),
                max_error=1e-2, min_speedup=1.0,
            )


# ─── §14-8 关掉心跳 → 窗口内观测全不可用、不写记忆 ──────────────
class Acceptance8_HeartbeatOffInvalidatesWindow(unittest.TestCase):
    def test_ruler_broken_window_marks_observations(self):
        r = HeartbeatResult(ok=False, epoch=1, reasons=["RULER_BROKEN"],
                            known_good_ratio=3.0, known_slow_ratio=1.0,
                            window_id="w1")
        self.assertTrue(ruler_broken_window(r, "w1"),
                        "同窗观测应被判为失效")

    def test_broken_window_invalidates_same_window_only(self):
        r = HeartbeatResult(ok=False, epoch=1, reasons=["RULER_BROKEN"],
                            known_good_ratio=3.0, known_slow_ratio=1.0,
                            window_id="w1")
        self.assertTrue(ruler_broken_window(r, "w1"))
        self.assertFalse(ruler_broken_window(r, "other-window"),
                         "跨窗不牵连")
        c = _contract()
        obs = evaluate(c, lambda: {"value": 2.0, "validity": "valid"},
                       record_hash=c.hash, heartbeat=r)
        self.assertFalse(obs.eligibility.can_persist)
        self.assertFalse(obs.eligibility.can_count_progress)

    def test_no_admit_from_broken_window_even_with_two_confirmations(self):
        r = HeartbeatResult(ok=False, epoch=1, reasons=["RULER_BROKEN"],
                            known_good_ratio=3.0, known_slow_ratio=1.0,
                            window_id="w1")
        c = _contract()
        obs = evaluate(c, lambda: {"value": 1.5, "validity": "valid"},
                       record_hash=c.hash, heartbeat=r)
        ok, msg = admit_to_corpus(
            _metrics_from(obs),
            ConfirmRecord(ok=True, speedup=1.4, validity="valid"))
        self.assertFalse(ok, "坏窗口里的读数靠着两次确认进了长期记忆")
        self.assertIn("can_persist", msg)
        self.assertIn("RULER_BROKEN", msg, "拒因里没提尺子坏了 —— 那这条用例没证到窗口")

    def test_published_eligibility_is_the_closure_points_answer(self):
        """生产 L3 上「到期没跑心跳」必须让**发布出去的**资格位全 False

        2026-10-01 真机 §14-8 抓到 (H20, 改动前): 同一个 metrics 里并排出现
        `ruler_broken=True` 与 `can_persist=True` —— verifier 自己推一份资格 (读不到心跳
        窗口) 并把它发布出去, oracle 收口点另算一份 (读得到心跳) 只写进 `observation`。
        入库判据读的正是发布出去的那个 ⇒ 「关掉心跳该窗口不写长期记忆」在生产路径上
        没生效 (那次真机没入库, 只是因为加速比 0.42x 被**另一道门**偶然挡住了)。

        这条用例走**生产 L3 入口** (`_verify_performance`), 钉的就是「发布值 = 收口点的
        答案」: 只测 `evaluate` 的话, 分叉正好落在两次调用**之间**, 测不到。
        """
        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
            target=TargetSpec(hardware="mock_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0,
                                       max_error=1e-2))
        src = ("import numpy as np\n\ndef kernel(A, B):\n"
               "    return np.asarray(A) @ np.asarray(B)\n")
        rng = np.random.default_rng(0)
        xs = {"A": rng.standard_normal((16, 16)).astype(np.float32),
              "B": rng.standard_normal((16, 16)).astype(np.float32)}
        r = Verifier(MockGPU())._verify_performance(src, xs, 0.0, task,
                                                   heartbeat=None,
                                                   heartbeat_expected=True)
        m = r.metrics
        self.assertTrue(m.get("ruler_broken"), "到期没跑心跳却没标记尺子坏")
        self.assertIn("observation", m, "收口点没产观测")
        self.assertEqual(m["eligibility"], m["observation"]["eligibility"],
                         "发布出去的资格位不是收口点算的那一份")
        self.assertFalse(m["eligibility"]["can_persist"],
                         "坏窗口的读数仍自称可入库 —— §14-8 在链路上没生效")

    def test_unable_to_establish_records_ruler_broken(self):
        u = UnableToEstablish()
        u.extend_metrics({"ruler_broken": True, "ruler_reason": "RULER_BROKEN"})
        self.assertTrue(any(i.kind == "ruler" for i in u.items))


# ─── §14-10 数值错的候选必须被 G2 拒, 且理由指向该处 ──────────────
class Acceptance10_WrongKernelIsRefused(unittest.TestCase):
    """验收的形状问题, 不是覆盖率问题。

    正确性本身测得很好 (`tests/test_l2_tolerance.py` 22 条: 缩放核被拒、多输入套件
    一组错则整体失败、非有限值必败、分精度 rtol), 但**那些都不在 §14 这道门禁里** ——
    而 §14 是"全绿才允许往上盖 Stage 3/4"的那张表。于是"正确性没被这一道门禁考过"
    这件事, 与"正确性没被考过"在文档上长得一样。

    本类钉三件事:
      1. 错的候选**止步于 L2, 不产生任何 L3 性能观测** —— 这是门禁的形状:
         没有性能读数, 就没有东西可以拿去晋级 `best` 或入库;
      2. 拒绝理由**指向坏在哪里** (`first_bad_index` 落在出错的那一行上),
         而不是只说"误差超阈值" —— 后者既修不了也路由不了 (§8.2);
      3. 正确的候选**照常走到 L3** —— 门是针对性, 不是"什么都拒"。
    """

    N = 64

    #: 整体缩放 3%: 原门禁 (纯绝对容差, driver 传 5e-1) 会放行它
    SCALED = "import numpy as np\n\ndef kernel(A, B):\n    return (A @ B) * 0.97\n"
    #: 只错最后一行 (尾块): 坏点必须被定位到第 63 行
    TAIL = ("import numpy as np\n\ndef kernel(A, B):\n    C = A @ B\n"
            "    C[-1, :] = 0.0\n    return C\n")
    #: 丢了 B: 全错
    DROPPED_INPUT = "import numpy as np\n\ndef kernel(A, B):\n    return A @ A\n"
    CORRECT = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"

    def setUp(self):
        self.verifier = Verifier(MockGPU())
        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
            target=TargetSpec(hardware="mock_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0,
                                       max_error=1e-5))
        rng = np.random.default_rng(0)
        self.inputs = {"A": rng.standard_normal((self.N, self.N)).astype(np.float32),
                       "B": rng.standard_normal((self.N, self.N)).astype(np.float32)}
        self.task = task

    def _verify(self, code):
        return self.verifier.verify(code, self.task, self.inputs)

    def _l2(self, results):
        l2 = [r for r in results if r.level.name == "L2_FUNCTION"]
        self.assertEqual(len(l2), 1, "L2 结果缺失 —— 验收对象不是 L2 判据")
        return l2[0]

    def test_wrong_candidate_produces_no_l3_observation(self):
        """门禁的形状: 数值错 ⇒ 没有性能读数可拿去晋级/入库

        只断言 "L2 没过" 是不够的: 一个"L2 报了 false 但仍然跑 L3 并把 gflops 发布出去"
        的实现同样会绿, 而它恰恰是本节要防的那种 (算了但没接上)。
        """
        results = self._verify(self.SCALED)
        self.assertEqual([r.level.name for r in results], ["L1_COMPILE", "L2_FUNCTION"],
                         "错的候选跑到了 L3 —— 性能观测没有对错前置条件")
        l2 = self._l2(results)
        self.assertFalse(l2.passed, f"缩放 3% 的核竟然通过了: {l2.message}")
        self.assertLess(l2.metrics["matched_ratio"], 0.01)
        # 「没有可晋级的读数」在数据上的样子: 没有任何一级产出速率类指标
        for r in results:
            self.assertNotIn("speedup_vs_baseline", r.metrics,
                             f"{r.level.name} 发布了一个没有正确性凭据的性能读数")

    def test_refusal_points_at_the_fault(self):
        """理由必须指向坏处 —— 只说"误差超阈值"既修不了也路由不了

        尾块错 (最后一行全 0) 是最好定位的一种: 坏点数 = 一行 = N, 首个坏点下标必须落在
        最后一行 (>= (N-1)*N)。这条同时钉住 `first_bad_index` 的**语义** (展平下标),
        将来若把它改成"第几个坏点"或"行号", 这里会红。
        """
        results = self._verify(self.TAIL)
        l2 = self._l2(results)
        self.assertFalse(l2.passed)
        m = l2.metrics
        self.assertEqual(m["bad_elements"], self.N, "坏点数应当正好是一行")
        self.assertEqual(m["first_bad_index"], (self.N - 1) * self.N,
                         "首个坏点没有被定位到最后一行 —— 理由没指向坏处")
        # 「哪错了」的下一问是「该怎么修」: 形态要跟着位置一起出来 (设计 §8.2)。
        # 尾块写零在形态上就是 tail_block —— 这条把 §8.2 挂到 §14 这道门禁上。
        self.assertEqual(m.get("error_shape"), "tail_block")
        # actual = 候选算出来的那一格 (我们把这一行写成了 0), expected = 参考值。
        # 两侧取值都带上, 才分得清「整体缩放错」与「局部炸」(§8.2 的形态判据要用它)。
        self.assertAlmostEqual(m["first_bad_actual"], 0.0, places=6)
        self.assertNotAlmostEqual(
            m["first_bad_expected"], 0.0, places=6,
            msg="参考值不该是 0 —— 否则这个用例定位不了任何东西")
        self.assertIn(f"first_bad@{m['first_bad_index']}", l2.message,
                      "理由文本里没有坏点位置")

    def test_dropped_input_is_refused(self):
        """静默忽略一个输入 = 在测另一个算子 (与 §14-3 的库回退同族, 但发生在数值上)"""
        l2 = self._l2(self._verify(self.DROPPED_INPUT))
        self.assertFalse(l2.passed, f"丢了 B 的核竟然通过了: {l2.message}")
        self.assertEqual(l2.metrics["bad_elements"], l2.metrics["compared_elements"],
                         "应当是全错")

    def test_l3_only_without_a_credential_cannot_be_promoted(self):
        """唯一绕开 L2 的入口: 没凭据 ⇒ 不给"通过", 也不给任何可晋级/可入库的资格

        上一个用例钉住的是正常路径 (数值错 ⇒ 到不了 L3); 缺口只在 `verify_l3_only`
        上开着 —— 而它正好是**晋级层**那次测量 (重测确认) 的入口。
        "不产生可用于晋级的性能观测" 在数据上的样子就是: `passed=False` +
        档位 uninformative + 五位资格全 False。
        """
        l3 = self.verifier.verify_l3_only(self.CORRECT, self.inputs, 0.0, self.task,
                                          correctness_verified=False)[0]
        m = l3.metrics
        self.assertFalse(l3.passed, "没有正确性凭据的速率读数被记成了通过")
        self.assertEqual(m["measurement_validity"], "uninformative")
        self.assertFalse(m["measurement_can_gate"])
        for bit in ("can_verdict", "can_tune", "can_persist", "can_report",
                    "can_count_progress"):
            self.assertFalse(m["eligibility"][bit],
                             f"{bit} 仍为真 —— 没有凭据的读数还能造成后果")
        self.assertIn("正确性", l3.message)

    def test_l3_only_with_a_credential_is_normal(self):
        """对照组: 带了凭据就照常 —— 否则把凭据写成恒拒也能让上一条绿"""
        l3 = self.verifier.verify_l3_only(self.CORRECT, self.inputs, 0.0, self.task,
                                          correctness_verified=True)[0]
        self.assertTrue(l3.passed)
        self.assertIs(l3.metrics["correctness_verified"], True)
        self.assertTrue(l3.metrics["eligibility"]["can_persist"])

    def test_l3_only_without_the_key_does_not_retroactively_judge(self):
        """三态: 没带凭据 = 未观测 ⇒ 不改判, 但必须留下"这一格空着"的痕迹

        与 `real_device=None` / `static_guards=None` 同一边界: 回放脚本与老产物
        不因为缺这一格被判死, 但"空着"不等于"验过" —— 理由文本里要说出来,
        否则这一格会静默退化成"反正没人看"。
        """
        l3 = self.verifier.verify_l3_only(self.CORRECT, self.inputs, 0.0, self.task)[0]
        self.assertNotIn("correctness_verified", l3.metrics)
        self.assertTrue(l3.passed, "缺键被读成了不合格")
        reasons = " ".join((l3.metrics.get("observation") or {}).get("reasons") or [])
        self.assertIn("凭据", reasons, "缺凭据这件事在产物里没有痕迹")

    def test_correct_candidate_still_reaches_l3(self):
        """对照组: 门是针对性, 不是"什么都拒"

        没有这一条, 把 L2 写成恒 False 也能让上面三条全绿 —— 那是个更坏的实现。
        """
        results = self._verify(self.CORRECT)
        self.assertTrue(self._l2(results).passed, "正确核被误杀")
        self.assertEqual([r.level.name for r in results],
                         ["L1_COMPILE", "L2_FUNCTION", "L3_PERFORMANCE"],
                         "正确核没有走到 L3 —— 门禁在误杀之后又堵死了正常路径")


if __name__ == "__main__":
    unittest.main()
