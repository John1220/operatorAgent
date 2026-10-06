"""脊柱接入 Conductor 的离线标定 (设计 §10.5-4)

入库门 = persist_decision  ∧  资格位 can_persist  ∧  **两次独立重测**。
第三道门是设计 §3.5 的**不变量**, 不是开关: 没有重测就**拒绝入库** ——
不是「没有就当过了」, 也不是「今天没要求就当过了」(CHECKLOG I79)。

消融 (§14-9 的「关掉这道门会怎样」那一臂) **只能在 harness 侧 patch**:
见 `TestTheDoubleConfirmationIsAnInvariant` 里那条把生产者换成空实现的用例 ——
§3.5 规则 3: 一个能一键关安全门的环境变量就是那把尺子。
"""
import contextlib
import dataclasses
import io
import re as _RE
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from operator_agent.confirm import ConfirmRecord
from operator_agent.conductor import Conductor
from operator_agent.kernel_db import KernelDatabase
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (ConstraintSpec, KernelArtifact, Precision,
                                    SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
                                    TaskType)
from tests.backend_fakes import fake_backend
from tests.test_measurement_validity import LaunchDominatedGPU, NET_MS_MATMUL_4096


def _task():
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul"),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=1.0, max_error=1e-4),
    )


def _artifact(metrics, code="def kernel(A, B):\n    return A @ B\n"):
    return KernelArtifact(
        kernel_id="k-test",
        code=code,
        verification_results=[],
        benchmark_data=metrics,
    )


def _good_metrics(**extra):
    m = {
        "speedup_vs_baseline": 1.25,
        "measurement_validity": "valid",
        "measurement_can_gate": True,
        "backend_real_device": True,
        "gflops": 10.0,
    }
    m.update(extra)
    return m


class TestSaveToDbGates(unittest.TestCase):
    def setUp(self):
        self.cond = Conductor(backend=MockGPU(), kernel_db=KernelDatabase())

    def test_a_good_run_without_a_remeasure_does_not_persist(self):
        """**默认配置下这条路走不通** (§3.5 规则 2)

        ⚠️ 这条用例以前断言的是反面 (`test_good_run_still_persists_without_double_confirm_flag`
        ——「默认不强制两次确认, 不打断现有链路」)。它把「默认关的安全装置等于不存在」
        这件事**钉成了正确行为**: 一个默认 `False` 的构造参数让单次读数就进了语料,
        而设计 §3.5 把这道门列在「不允许配置」里。守卫钉住错的行为比没有守卫更贵 (I79)。
        """
        ok = self.cond._save_to_db(_artifact(_good_metrics()), _task())
        self.assertFalse(ok, "没有独立重测却入库了 —— 两次确认是空的")

    def test_degraded_eligibility_blocks_persist(self):
        m = _good_metrics(measurement_validity="degraded")
        m["eligibility"] = {"can_verdict": True, "can_tune": True,
                            "can_persist": False, "can_report": True,
                            "can_count_progress": False}
        ok = self.cond._save_to_db(_artifact(m), _task())
        self.assertFalse(ok, "can_persist=False 仍入库了")

    def test_it_refuses_without_a_remeasure(self):
        ok = self.cond._save_to_db(_artifact(_good_metrics()), _task())
        self.assertFalse(ok, "没有重测却入库 —— 两次确认是空的")

    def test_it_accepts_with_an_independent_remeasure(self):
        art = _artifact(_good_metrics())
        art.remeasure = ConfirmRecord(ok=True, speedup=1.2, validity="valid",
                                      independent=True)
        ok = self.cond._save_to_db(art, _task())
        self.assertTrue(ok)

    def test_remeasure_dropping_below_gate_refuses(self):
        art = _artifact(_good_metrics())
        art.remeasure = ConfirmRecord(ok=True, speedup=0.8, validity="valid")
        ok = self.cond._save_to_db(art, _task())
        self.assertFalse(ok)

    def test_direction_slower_is_stricter_than_faster(self):
        fast = _artifact(_good_metrics())
        fast.remeasure = ConfirmRecord(ok=True, speedup=1.5, validity="valid")
        slow = _artifact(_good_metrics())
        slow.remeasure = ConfirmRecord(ok=True, speedup=0.9, validity="valid")
        self.assertTrue(self.cond._save_to_db(fast, _task()))
        self.assertFalse(self.cond._save_to_db(slow, _task()))


class TestTheGateHasExactlyOneImplementation(unittest.TestCase):
    """入库判据只许有一个实现 (2026-10-01, spine-review ①)

    WHY: `oracle.admit_to_corpus` 一直是「入库门」的规范实现, 但 `_save_to_db` **自己又写了
    一遍**那三道门, 而且两份口径不同 (前者永远要求重测, 后者默认不要求)。断言库函数的行为
    证明不了生产路径的行为 —— 这正是「判据不在链路上」那一族的形状 (P0-1)。

    所以这里不测判据的语义 (那在 `test_oracle.py`), 只钉**接线**:
      1. 生产路径必须调它, 且把 `keep` 当成自己的结论 (两个方向都验, 免得只测「调了」);
      2. 传进去的 `requires_remeasure` 必须**恒为 True** —— 它是设计 §3.5 的不变量
         (不允许配置那一栏), 所以接线处**就该**写死; 能关掉它的东西一个都不许有。
    """

    def setUp(self):
        self.cond = Conductor(backend=MockGPU(), kernel_db=KernelDatabase())

    def _spy(self, verdict, seen):
        def spy(metrics, remeasure, **kw):
            seen.append({"metrics": metrics, "remeasure": remeasure, "kw": kw})
            return verdict
        return spy

    def test_a_refusing_gate_actually_blocks_persistence(self):
        """判据说拒, 生产路径就必须不入库 —— 不许它自己另判一遍"""
        seen = []
        with mock.patch("operator_agent.oracle.admit_to_corpus",
                        side_effect=self._spy((False, "哨兵: 判据拒了"), seen)):
            ok = self.cond._save_to_db(_artifact(_good_metrics()), _task())
        self.assertEqual(1, len(seen), "入库门没有走 oracle.admit_to_corpus")
        self.assertFalse(ok, "判据说拒, 生产路径却入库了 —— 它有自己的第二套判据")

    def test_an_allowing_gate_actually_allows_persistence(self):
        """反方向也要: 判据说行, 生产路径不许自己加门"""
        seen = []
        with mock.patch("operator_agent.oracle.admit_to_corpus",
                        side_effect=self._spy((True, "哨兵: 判据放行"), seen)):
            ok = self.cond._save_to_db(_artifact(_good_metrics()), _task())
        self.assertEqual(1, len(seen))
        self.assertTrue(ok, "判据说行, 生产路径却入不了库 —— 门在外面又加了一道")

    def test_the_double_confirmation_is_passed_as_an_invariant(self):
        """接线处写死 `True` —— 判据的那道门在**每一次**入库尝试上都要求着"""
        seen = []
        with mock.patch("operator_agent.oracle.admit_to_corpus",
                        side_effect=self._spy((True, "哨兵"), seen)):
            self.cond._save_to_db(_artifact(_good_metrics()), _task())
            self.cond._save_to_db(_artifact(_good_metrics()), _task())
        self.assertEqual([True, True],
                         [c["kw"].get("requires_remeasure") for c in seen],
                         "requires_remeasure 不是恒开的 —— 有人给安全语义留了开关")
        self.assertEqual(float(_task().constraints.min_speedup or 1.0),
                         seen[0]["kw"].get("min_speedup"))

    def test_the_gate_gets_the_metrics_the_artifact_carries(self):
        """判据的输入就是产物携带的那份读数 (不是重新拼一份)"""
        seen = []
        m = _good_metrics()
        with mock.patch("operator_agent.oracle.admit_to_corpus",
                        side_effect=self._spy((True, "哨兵"), seen)):
            self.cond._save_to_db(_artifact(m), _task())
        self.assertIs(seen[0]["metrics"], m, "判据拿到的不是产物里那份 metrics")


class TestRemeasureField(unittest.TestCase):
    """spine-review P2-10: `KernelArtifact.remeasure` 必须是声明字段

    WHY: `conductor._save_to_db` 读 `getattr(artifact, "remeasure", None)`,
    但字段从未声明 → 生产路径无人写入 → 要求两次确认时入库率 100% 归零。
    此前测试靠 monkeypatch 属性赋值绕过, 测不出这个缺口。
    """

    def test_remeasure_is_declared_on_kernel_artifact(self):
        names = {f.name for f in dataclasses.fields(KernelArtifact)}
        self.assertIn("remeasure", names,
                      "KernelArtifact 缺 remeasure 字段 —— 要求两次确认即全拒")

    def test_constructor_with_remeasure_persists(self):
        cond = Conductor(backend=MockGPU(), kernel_db=KernelDatabase())
        art = KernelArtifact(
            kernel_id="k-remeasure",
            code="def kernel(A, B):\n    return A @ B\n",
            verification_results=[],
            benchmark_data=_good_metrics(),
            remeasure=ConfirmRecord(ok=True, speedup=1.2, validity="valid",
                                    independent=True),
        )
        self.assertTrue(cond._save_to_db(art, _task()))

    def test_constructor_without_remeasure_refuses(self):
        cond = Conductor(backend=MockGPU(), kernel_db=KernelDatabase())
        art = KernelArtifact(
            kernel_id="k-no-remeasure",
            code="def kernel(A, B):\n    return A @ B\n",
            verification_results=[],
            benchmark_data=_good_metrics(),
        )
        self.assertIsNone(art.remeasure)
        self.assertFalse(cond._save_to_db(art, _task()),
                         "没有重测却入库 —— 两次确认是空的")


class TestHeartbeatProducer(unittest.TestCase):
    """标定心跳的**生产端** (spine-review P1-5)

    判据早就齐了 (`calibration` 的判定 + oracle 的规则 2 与「到期没跑」), 缺的是
    「谁计数、谁到期真的去量参照对」—— 以及 `ruler_broken`/`ruler_reason` 那两个死读键
    的写者。这里钉住生产端: 计数、触发点、失败不静默、以及结论真的进了产物。
    """

    def _cond(self, backend=None, interval=2):
        return Conductor(backend=backend or MockGPU(), kernel_db=KernelDatabase(),
                         heartbeat_interval=interval)

    def test_interval_must_be_positive(self):
        """非法 interval 在构造时就响, 不在真机循环深处才炸"""
        for bad in (0, -1, True, 1.5, "3"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self._cond(interval=bad)

    def test_not_due_before_the_interval(self):
        cond = self._cond(interval=3)
        for n in (0, 1):
            cond._true_measurements = n
            self.assertEqual((None, False), cond._heartbeat_for_next_measurement())

    def test_due_on_the_interval_and_only_once_per_measurement(self):
        """计数不前进的轮次 (L1/L2 挂了) 不该每轮重跑一次心跳"""
        cond = self._cond(interval=1)
        first, expected = cond._heartbeat_for_next_measurement()
        self.assertTrue(expected)
        self.assertIsNotNone(first)
        again, expected2 = cond._heartbeat_for_next_measurement()
        self.assertTrue(expected2)
        self.assertIs(first, again, "同一次测量重跑了心跳 (白烧机时)")
        self.assertEqual(1, cond._heartbeat_epoch)

    def test_an_honest_backend_passes_the_ruler(self):
        cond = self._cond(interval=1)
        heartbeat, expected = cond._heartbeat_for_next_measurement()
        self.assertTrue(expected)
        self.assertTrue(heartbeat.ok, heartbeat.reasons)
        self.assertGreaterEqual(heartbeat.known_slow_ratio, 2.0, heartbeat.reasons)

    def test_an_indistinguishable_pair_is_reported_broken(self):
        """两臂同速 = 尺子分不出快慢 → RULER_BROKEN, 不许静默放行"""
        backend = fake_backend(
            benchmark=lambda code, inputs, warmup, repeat: {"median_time_ms": 1.0})
        cond = self._cond(backend=backend, interval=1)
        heartbeat, expected = cond._heartbeat_for_next_measurement()
        self.assertTrue(expected)
        self.assertIsNotNone(heartbeat)
        self.assertFalse(heartbeat.ok, "两臂同速却报 ok")

    def test_a_backend_that_cannot_be_measured_fails_closed(self):
        """后端量不出参照物 (带 `success: False` 的传输失败形态) → RULER_BROKEN"""
        backend = fake_backend(
            benchmark=lambda code, inputs, warmup, repeat: {"success": False,
                                                            "error": "SSH 没回答"})
        cond = self._cond(backend=backend, interval=1)
        heartbeat, expected = cond._heartbeat_for_next_measurement()
        self.assertTrue(expected)
        self.assertIsNotNone(heartbeat)
        self.assertFalse(heartbeat.ok)


CODE = """import numpy as np

def kernel(A, B):
    C = A
    for _ in range(20):
        C = A @ B
    return C
"""

SKETCH = {"algorithm": "matmul", "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
          "memory_plan": {}, "pipeline": [], "optimization_hints": []}


class _ScriptedLLM:
    def chat(self, messages, **kw):
        return CODE

    def chat_json(self, messages, **kw):
        return dict(SKETCH)


class _PinnedTimerGPU(MockGPU):
    """计时通道**坏了**的假件: 快核慢核报同一个数 —— 尺子分不出快慢。"""

    def benchmark(self, kernel_code, inputs, warmup=3, repeat=10):
        result = super().benchmark(kernel_code, inputs, warmup, repeat)
        for key in ("min_time_ms", "median_time_ms", "avg_time_ms", "max_time_ms"):
            result[key] = 1.0
        return result


class TestHeartbeatIsInTheLoop(unittest.TestCase):
    """端到端: 心跳真的被插进测量流, 且结论进了产物 (不只是库里有判据)"""

    def _run(self, backend, interval=1):
        import io as _io
        import contextlib as _ctx
        import numpy as _np
        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(framework="pytorch", kernel_type="matmul", fused_ops=[],
                              complexity=TaskComplexity.MEDIUM),
            target=TargetSpec(hardware="mock_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=1.0,
                                       max_error=1e-4),
        )
        rng = _np.random.default_rng(0)
        inputs = {"A": rng.standard_normal((64, 64)).astype(_np.float32),
                  "B": rng.standard_normal((64, 64)).astype(_np.float32)}
        cond = Conductor(backend=backend, kernel_db=KernelDatabase(), llm=_ScriptedLLM(),
                         max_optimize_rounds=1, heartbeat_interval=interval)
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            artifact = cond.execute(task, inputs, target_gflops=0.0)
        return cond, artifact, buf.getvalue()

    def test_a_healthy_ruler_is_due_and_recorded_in_the_artifact(self):
        cond, artifact, out = self._run(MockGPU())
        self.assertGreaterEqual(cond._heartbeat_epoch, 1, f"心跳没被触发:\n{out}")
        self.assertGreaterEqual(cond._true_measurements, 1, "真值测量计数没前进")
        metrics = artifact.benchmark_data or {}
        self.assertIs(False, metrics.get("ruler_broken"),
                      "有心跳且通过就该写 False (「检查过、是好的」那种写法)")
        self.assertIs(True, (metrics.get("heartbeat") or {}).get("heartbeat_checked"))

    def test_a_broken_timer_is_caught_by_the_heartbeat(self):
        """后端对快核慢核报同一个数 → 心跳宣布 RULER_BROKEN, 且产物里看得见"""
        _, artifact, out = self._run(_PinnedTimerGPU())
        self.assertIn("RULER_BROKEN", out, f"心跳没抓到坏计时器:\n{out}")
        metrics = artifact.benchmark_data or {}
        self.assertIs(True, metrics.get("ruler_broken"))
        self.assertTrue(metrics.get("ruler_reason"))
        self.assertFalse((metrics.get("observation") or {})
                         .get("eligibility", {}).get("can_verdict", True),
                         "尺子坏了这条读数仍然可判决")


class TestRemeasureProducer(unittest.TestCase):
    """spine-review P2-10 的**生产端**: 「一律要求两次确认」不再等于入库率归零

    消费者 (`_save_to_db` → `confirm_persist`) 早就在读 `artifact.remeasure`, 而全仓没有
    写者 —— 于是「要求两次确认」只有「一律拒」一档。这里钉住写者: 谁在什么时候补这次
    重测、记录里带什么、以及它真的能让入库门放行。要求恒开 (I79) 之后这一条更吃重:
    没有这个写者, 每一份产物都进不了语料。
    """

    def _cond(self, backend=None, **kw):
        return Conductor(backend=backend or MockGPU(), kernel_db=KernelDatabase(), **kw)

    def _inputs(self):
        import numpy as _np
        rng = _np.random.default_rng(0)
        return {"A": rng.standard_normal((64, 64)).astype(_np.float32),
                "B": rng.standard_normal((64, 64)).astype(_np.float32)}

    def _produced(self):
        cond = self._cond()
        art = _artifact(_good_metrics())
        cond._produce_remeasure(art, _task(), self._inputs(), 0.0)
        return cond, art

    def _stubbed(self, art):
        """把机时那部分钉成过关值 —— 见 `test_the_produced_record_...` 的说明"""
        art.remeasure = dataclasses.replace(art.remeasure, ok=True, speedup=1.5,
                                            validity="valid")
        return art

    def test_there_is_no_switch_to_turn_it_off(self):
        """**没有开关** —— 不许是可配置面 (=「默认关的安全装置等于不存在」的反面)

        两条一起判, 因为「删掉构造参数」与「删掉实现」是两件事:
          * API 层: 传 `require_double_confirm=` 必须失败 (TypeError);
          * 对象层: 实例上不存在一个能改的布尔属性 (改 `cond.require_double_confirm = False`
            以前会静默生效, 因为 `_save_to_db` 读的就是它)。
        """
        with self.assertRaises(TypeError):
            self._cond(require_double_confirm=False)
        c = self._cond()
        for name in ("require_double_confirm", "double_confirm", "require_remeasure"):
            self.assertFalse(hasattr(c, name), f"{name} 还在 —— 它就是一个可关的开关")

    def test_the_producer_fills_in_a_usable_record(self):
        cond, art = self._produced()
        self.assertIsNotNone(art.remeasure,
                             "打开两次确认却没有生产者 —— 入库率归零的老问题")
        rm = art.remeasure
        self.assertTrue(rm.independent)
        self.assertTrue(rm.contract_hash,
                        "重测记录没带契约戳 → 「同一条判据」那条判据无从生效")
        self.assertIsInstance(rm.speedup, float, rm.reason)
        self.assertTrue(rm.validity, "重测的档位没带上")

    def test_a_remeasure_in_a_broken_window_cannot_confirm(self):
        """尺子坏了的窗口里的读数不能当第二次确认 (与 `admit_to_corpus` 同一条道理)

        重测也是一条 L3 读数, 所以它同样过尺子检查; 而「决定永久记忆的那一次测量」尤其
        不能是唯一没被检查过的那一次。
        """
        cond = self._cond(backend=_PinnedTimerGPU(), heartbeat_interval=1)
        art = _artifact(_good_metrics())
        cond._produce_remeasure(art, _task(), self._inputs(), 0.0)
        self.assertFalse(art.remeasure.ok, "坏窗口里的重测被当成了有效确认")
        self.assertIn("ruler_broken", art.remeasure.reason,
                      f"拒的理由没说是尺子的问题: {art.remeasure.reason}")

    def test_the_produced_record_can_actually_open_the_door(self):
        """端到端: 生产者写、消费者读 —— 两边接上, 入库门才放行

        速度比在这条里**钉成过关值**: 64×64 上候选与基线都落在 ~2µs, 比值基本就是
        噪声 (实测在 0.96x~1.04x 之间跳), 拿它当断言等于让用例随机红。这条要验的是
        「生产者写下的记录消费者认, 且判据同一性判得准」, 机时不是它的题目 ——
        `_produce_remeasure` 真的跑了一次 L3 这件事由上面那条用例覆盖。
        """
        cond, art = self._produced()
        self._stubbed(art)
        art.benchmark_data["contract_hash"] = art.remeasure.contract_hash
        self.assertTrue(cond._save_to_db(art, _task()),
                        "生产者写了重测, 入库门仍然拒 —— 两边没接上")

    def test_a_remeasure_from_another_contract_is_refused(self):
        """首次与重测之间判据被改写 → 不是「同一条断言量了两遍」

        断言**理由**而不只是 False: 把机时钉成过关值之后, 若这条判据被删掉, 用例仍会
        因为别的门而拒 —— 那就成了空转。这里要求拒的理由正是契约那一条。
        """
        import contextlib as _ctx
        import io as _io
        cond, art = self._produced()
        self._stubbed(art)
        art.benchmark_data["contract_hash"] = "f" * 64
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            keep = cond._save_to_db(art, _task())
        self.assertFalse(keep, "判据在两次测量之间变了, 却仍然入库")
        self.assertIn("另一条契约", buf.getvalue(),
                      f"拒的理由不是判据同一性 —— 这条用例在空转: {buf.getvalue()}")


class TestCorrectnessCredentialWiring(unittest.TestCase):
    """`verify_l3_only` 是**唯一能绕开 L2** 产出 L3 观测的入口 (设计 §14-10 / §20.4)

    它同时是**晋级层**那一次测量 (重测确认) 的入口 —— 所以"这个核算得对"必须由调用方
    按它手里那份 L2 结果给出。漏传的失效方式是**静默的**: 三态里的 `None` = 没观测 =
    不改判, 于是"忘了接"与"验过了"在产物里长得一样。结构守卫在这里比行为用例合适 ——
    行为用例只能证明"传了会怎样", 证明不了"每个调用点都传了"。
    """

    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.src = (root / "operator_agent" / "conductor.py").read_text(encoding="utf-8")

    @staticmethod
    def _call_body(src, open_paren):
        """从 '(' 起取到**与它配对的那个** ')' —— 不用写死窗口宽度

        写死窗口 (`src[i:i+800]`) 的失效方式是静默的: 调用点前面的注释一长,
        守卫就在守卫一个已经不含目标的行区间 —— 那时它是绿的, 但它什么都没查。
        """
        depth = 0
        for j in range(open_paren, len(src)):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    return src[open_paren:j + 1]
        return src[open_paren:]

    def test_both_production_call_sites_pass_the_credential(self):
        opens = [m.end() - 1 for m in _RE.finditer(r"verify_l3_only\(", self.src)]
        self.assertEqual(len(opens), 2,
                         f"生产调用点数量变了 ({len(opens)}) —— 本守卫需要跟着复核")
        for i in opens:
            self.assertIn("correctness_verified=", self._call_body(self.src, i),
                          "有个 verify_l3_only 调用点没带正确性凭据")

    def test_the_credential_is_derived_not_asserted(self):
        """凭据必须从**那一份候选自己的** L2 结果里读, 不许写死成 True

        写死的话它就不再是凭据, 而是"我们相信调用方总是对的"这句注释 ——
        而 §14-10 要防的正是"以为验过了"。
        """
        self.assertIn("_correctness_credential(", self.src)
        self.assertNotIn("correctness_verified=True", self.src)


class _SketchLlm:
    """只答 sketch 的最小替身 (设计器要求 LLM; 生成侧用 `use_coder_pipeline=False`)"""

    def chat(self, messages, **kw):
        return "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"

    def chat_json(self, messages, **kw):
        return {"algorithm": "matmul",
                "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                "memory_plan": {}, "pipeline": [], "optimization_hints": []}


class TestTheDoubleConfirmationIsAnInvariant(unittest.TestCase):
    """设计 §3.5: 「长期记忆的独立确认」**不允许配置** —— 今天它无条件生效

    这一组用例的存在理由 (CHECKLOG I79): 它此前是一个默认 `False` 的构造参数, 于是
    「默认配置下单次读数即可入库」。翻转之后要钉住三件事, 缺一条这个不变量就会退回去:

      1. 默认配置下真的走这条路 (不是"文档说走");
      2. 没有开关: API 上给不出关闭值, 对象上也没有一个能改的布尔;
      3. 消融 (§14-9 的「关掉这道门会怎样」那一臂) **只能在 harness 侧 patch** ——
         把生产者换成空实现, 结果是**一份都进不了语料**, 而不是静默入库。
    """

    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.src = (root / "operator_agent" / "conductor.py").read_text(encoding="utf-8")

    def _task(self, min_speedup=0.001):
        return TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul"),
            target=TargetSpec(hardware="mock_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32,
                                       min_speedup=min_speedup, max_error=1e-5),
        )

    def _run(self, cond):
        rng = np.random.default_rng(0)
        inputs = {"A": rng.standard_normal((64, 64)).astype(np.float32),
                  "B": rng.standard_normal((64, 64)).astype(np.float32)}
        with contextlib.redirect_stdout(io.StringIO()):
            art = cond.execute(self._task(), inputs, None, target_gflops=0.0)
        return art

    def _cond(self, **kw):
        return Conductor(backend=LaunchDominatedGPU(NET_MS_MATMUL_4096),
                         kernel_db=KernelDatabase(jsonl_path=None),
                         llm=_SketchLlm(), use_coder_pipeline=False,
                         max_optimize_rounds=2, heartbeat_interval=1000, **kw)

    def test_the_producer_runs_on_the_normal_path(self):
        """① 默认配置下: 有结论要进语料时, 产物里**一定**带着那条独立重测记录

        ⚠️ 这里断言的是「重测**做了**」而不是「入库**成了**」: 在控制机上没有 CUDA,
        `MockGPU` 只会声明 `EMULATED`/`SIMULATED_PERF` ⇒ 读数按设计被判 uninformative,
        入库在**更前面**那道门上就该被拒 (「不能把假信号固化成语料」)。在这台机器上
        断言 `kept=True` 等于制造一条假证据 —— 想验「入库成功」得有一台真卡,
        这条正向端到端对照今天全仓都缺 (记在 CHECKLOG I80)。
        """
        art = self._run(self._cond())
        self.assertTrue(art.all_passed, "前提: 这一跑要让结论有资格进长期记忆")
        self.assertIsNotNone(art.remeasure,
                             "默认配置下没补重测 —— 不变量没生效 (I79 的洞回来了)")
        self.assertTrue(art.remeasure.independent)
        self.assertTrue(art.remeasure.validity, "重测记录没带档位")

    @staticmethod
    def _method_body(src: str, name: str) -> str:
        """取 `def <name>(...)` 到下一个同级 `    def ` 之间的源码

        用正则找**成对的** def 边界, 不写死窗口宽度: 写死窗口的失效方式是静默的
        (注释一长, 守卫就在守一个已经不含目标的行区间 —— 那时它是绿的, 但什么都没查)。
        """
        m = _RE.search(rf"^    def {_RE.escape(name)}\(", src, _RE.M)
        if m is None:
            raise AssertionError(f"没找到 {name} —— 本守卫需要跟着复核")
        nxt = _RE.search(r"^    def ", src[m.end():], _RE.M)
        return src[m.start(): m.end() + (nxt.start() if nxt else len(src))]

    @staticmethod
    def _signature(src: str, name: str) -> str:
        """只取形参表 (到与它配对的那个右括号) —— 参数名与参数名之间的注释都在里面"""
        m = _RE.search(rf"^    def {_RE.escape(name)}\(", src, _RE.M)
        if m is None:
            raise AssertionError(f"没找到 {name} —— 本守卫需要跟着复核")
        depth = 0
        for j in range(m.end() - 1, len(src)):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    return src[m.start():j + 1]
        raise AssertionError(f"{name} 的形参表没有配对右括号")

    @staticmethod
    def _without_comments(text: str) -> str:
        """去掉 `#` 到行尾 —— 守卫判的是**代码**, 注释里出现这些词是正当的

        ⚠️ 这个剥离必须是保守的: 只去注释, 不去字符串 (藏在字符串里的
        `"os.environ"` 仍然会被判成读环境变量 —— 宁可误报也不许漏报)。
        """
        return _RE.sub(r"#[^\n]*", "", text)

    def test_no_environment_variable_or_cli_flag_can_turn_it_off(self):
        """§3.5 的**形状**: 安全语义不许有环境变量 / 配置文件 / 命令行开关

        源码级守卫 —— 它的反面在行为上不可观察: 今天没人会去设那个变量, 所以加上一行
        `os.environ.get("OPERATOR_AGENT_DOUBLE_CONFIRM", "0")` 之后**所有行为用例都还是
        绿的**, 而不变量已经变回开关。同一个理由也适用于构造参数: 名字像开关的布尔一律
        不许再出现在 `Conductor.__init__` 上。
        """
        for method in ("_save_to_db", "_produce_remeasure"):
            body = self._without_comments(self._method_body(self.src, method))
            for needle in ("environ", "getenv", "argparse"):
                self.assertNotIn(needle, body,
                                 f"{method} 里能读环境变量/命令行 —— 那就是一个开关")
        # 形参表单独判 (注释里提到那个**已删除**的参数名是正当的, 见 `__init__` 的说明)
        sig = self._without_comments(self._signature(self.src, "__init__"))
        self.assertNotRegex(sig, r"require_\w*confirm|double_confirm|skip_confirm",
                            "构造参数里又出现了「可关掉两次确认」的开关")

    def test_removing_the_producer_leaves_the_gate_without_a_second_reading(self):
        """③ 消融臂 (§14-9 的「关掉两次确认会怎样」): 生产者 patch 成空实现

        这是 §3.5 规则 3 要的形状 —— 消融**不是**产品开关, 而是 harness 侧的一处 patch。
        两件事一起钉:
          * 生产者没了 ⇒ 判据手里那次重测就是 `None` (少了第二份读数, 不是少了别的);
          * 判据那一侧**还是**要求着 (`requires_remeasure=True`) —— 消融消的是生产者,
            不是判据; 否则消融就变成了「把开关关掉」, 正是规则 3 不许的那种做法。
        """
        seen = []

        def spy(metrics, remeasure, **kw):
            seen.append({"remeasure": remeasure, "kw": kw})
            return True, "哨兵: 判据放行"

        normal = self._run(self._cond())
        with mock.patch("operator_agent.oracle.admit_to_corpus", side_effect=spy):
            normal = self._run(self._cond())
        with mock.patch.object(Conductor, "_produce_remeasure", lambda *a, **k: None), \
                mock.patch("operator_agent.oracle.admit_to_corpus", side_effect=spy):
            ablated = self._run(self._cond())

        self.assertIsNotNone(normal.remeasure, "前提: 正常臂里有那条重测记录")
        self.assertIsNone(ablated.remeasure, "生产者被 patch 掉了, 记录却还在?")
        self.assertTrue(seen[0]["remeasure"] is not None
                        and seen[1]["remeasure"] is None,
                        f"判据收到的重测记录不对: {[c['remeasure'] for c in seen]}")
        self.assertEqual([True, True],
                         [c["kw"].get("requires_remeasure") for c in seen],
                         "消融臂把判据那道门也关了 —— §3.5 规则 3 不许这么做")


if __name__ == "__main__":
    unittest.main()
