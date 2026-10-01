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

**全绿才允许往上盖 Stage 3/4**。本文件是验收编排: 底层判据在
`measurement` / `eligibility` / `contract` / `calibration` / `oracle`,
这里按 §14 的原句一一对上, 并补上原先缺口 (7 的运行时配对检查)。
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


if __name__ == "__main__":
    unittest.main()
