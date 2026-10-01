"""脊柱接入 Conductor 的离线标定 (设计 §10.5-4)

入库门 = persist_decision  ∧  资格位 can_persist  ∧  (可选) 两次独立重测。
`require_double_confirm` 打开后, 没有重测就**拒绝入库** —— 不是「没有就当过了」。
"""
import dataclasses
import unittest

from operator_agent.confirm import ConfirmRecord
from operator_agent.conductor import Conductor
from operator_agent.kernel_db import KernelDatabase
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (ConstraintSpec, KernelArtifact, Precision,
                                    SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
                                    TaskType)
from tests.backend_fakes import fake_backend


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

    def test_good_run_still_persists_without_double_confirm_flag(self):
        """默认不强制两次确认 —— 不打断现有链路"""
        ok = self.cond._save_to_db(_artifact(_good_metrics()), _task())
        self.assertTrue(ok)

    def test_degraded_eligibility_blocks_persist(self):
        m = _good_metrics(measurement_validity="degraded")
        m["eligibility"] = {"can_verdict": True, "can_tune": True,
                            "can_persist": False, "can_report": True,
                            "can_count_progress": False}
        ok = self.cond._save_to_db(_artifact(m), _task())
        self.assertFalse(ok, "can_persist=False 仍入库了")

    def test_require_double_confirm_refuses_without_remeasure(self):
        self.cond.require_double_confirm = True
        ok = self.cond._save_to_db(_artifact(_good_metrics()), _task())
        self.assertFalse(ok, "没有重测却入库 —— 两次确认是空的")

    def test_require_double_confirm_accepts_with_independent_remeasure(self):
        self.cond.require_double_confirm = True
        art = _artifact(_good_metrics())
        art.remeasure = ConfirmRecord(ok=True, speedup=1.2, validity="valid",
                                      independent=True)
        ok = self.cond._save_to_db(art, _task())
        self.assertTrue(ok)

    def test_remeasure_dropping_below_gate_refuses_even_with_flag(self):
        self.cond.require_double_confirm = True
        art = _artifact(_good_metrics())
        art.remeasure = ConfirmRecord(ok=True, speedup=0.8, validity="valid")
        ok = self.cond._save_to_db(art, _task())
        self.assertFalse(ok)

    def test_direction_slower_is_stricter_than_faster(self):
        self.cond.require_double_confirm = True
        fast = _artifact(_good_metrics())
        fast.remeasure = ConfirmRecord(ok=True, speedup=1.5, validity="valid")
        slow = _artifact(_good_metrics())
        slow.remeasure = ConfirmRecord(ok=True, speedup=0.9, validity="valid")
        self.assertTrue(self.cond._save_to_db(fast, _task()))
        self.assertFalse(self.cond._save_to_db(slow, _task()))


class TestRemeasureField(unittest.TestCase):
    """spine-review P2-10: `KernelArtifact.remeasure` 必须是声明字段

    WHY: `conductor._save_to_db` 读 `getattr(artifact, "remeasure", None)`,
    但字段从未声明 → 生产路径无人写入 → `require_double_confirm=True` 时
    入库率 100% 归零。此前测试靠 monkeypatch 属性赋值绕过, 测不出这个缺口。
    """

    def test_remeasure_is_declared_on_kernel_artifact(self):
        names = {f.name for f in dataclasses.fields(KernelArtifact)}
        self.assertIn("remeasure", names,
                      "KernelArtifact 缺 remeasure 字段 —— require_double_confirm 打开即全拒")

    def test_constructor_with_remeasure_persists_under_double_confirm(self):
        cond = Conductor(backend=MockGPU(), kernel_db=KernelDatabase())
        cond.require_double_confirm = True
        art = KernelArtifact(
            kernel_id="k-remeasure",
            code="def kernel(A, B):\n    return A @ B\n",
            verification_results=[],
            benchmark_data=_good_metrics(),
            remeasure=ConfirmRecord(ok=True, speedup=1.2, validity="valid",
                                    independent=True),
        )
        self.assertTrue(cond._save_to_db(art, _task()))

    def test_constructor_without_remeasure_refuses_under_double_confirm(self):
        cond = Conductor(backend=MockGPU(), kernel_db=KernelDatabase())
        cond.require_double_confirm = True
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
    """spine-review P2-10 的**生产端**: 打开 `require_double_confirm` 不再等于入库率归零

    消费者 (`_save_to_db` → `confirm_persist`) 早就在读 `artifact.remeasure`, 而全仓没有
    写者 —— 于是那个开关只有「一律拒」一档。这里钉住写者: 谁在什么时候补这次重测、
    记录里带什么、以及它真的能让入库门放行。
    """

    def _cond(self, backend=None, **kw):
        return Conductor(backend=backend or MockGPU(), kernel_db=KernelDatabase(), **kw)

    def _inputs(self):
        import numpy as _np
        rng = _np.random.default_rng(0)
        return {"A": rng.standard_normal((64, 64)).astype(_np.float32),
                "B": rng.standard_normal((64, 64)).astype(_np.float32)}

    def _produced(self):
        cond = self._cond(require_double_confirm=True)
        art = _artifact(_good_metrics())
        cond._produce_remeasure(art, _task(), self._inputs(), 0.0)
        return cond, art

    def _stubbed(self, art):
        """把机时那部分钉成过关值 —— 见 `test_the_produced_record_...` 的说明"""
        art.remeasure = dataclasses.replace(art.remeasure, ok=True, speedup=1.5,
                                            validity="valid")
        return art

    def test_double_confirm_defaults_off(self):
        """默认关: 它是「每一份要记住的结论都量两遍」的机时承诺, 不该悄悄打开"""
        self.assertFalse(self._cond().require_double_confirm)

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
        cond = self._cond(backend=_PinnedTimerGPU(), require_double_confirm=True,
                          heartbeat_interval=1)
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


if __name__ == "__main__":
    unittest.main()
