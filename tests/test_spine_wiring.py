"""脊柱接入 Conductor 的离线标定 (设计 §10.5-4)

入库门 = persist_decision  ∧  资格位 can_persist  ∧  (可选) 两次独立重测。
`require_double_confirm` 打开后, 没有重测就**拒绝入库** —— 不是「没有就当过了」。
"""
import unittest

from operator_agent.confirm import ConfirmRecord
from operator_agent.conductor import Conductor
from operator_agent.kernel_db import KernelDatabase
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (ConstraintSpec, KernelArtifact, Precision,
                                    SourceSpec, TargetSpec, TaskSchema, TaskType)


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


if __name__ == "__main__":
    unittest.main()
