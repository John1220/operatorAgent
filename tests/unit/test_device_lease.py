"""§11-4 的观测租约在生产链路上真的落了产物 (CHECKLOG I59)

设计原话: 「租不到独占卡时, 记录 `external_load` 并在 `evidence` 中显式标注; 观测降为
DEGRADED, 而不是失败」。这条链路此前**只有定义、没有生产者**: `DeviceLease` 全仓零
构造点, 于是 §11-4 的「记录」那一半是纸面承诺。

本文件钉两件事 (判据本身的三态在 `tests/unit/test_placement.py`):
  1. 一次真实的 L3 调用之后, 产物里带着**这张卡**的租约, 字段来自**同一份载荷**
     (`gpu_util_pct`), 且 `exclusive` 恒 `None` (不租卡 ⇒ 不许声称独占);
  2. 载荷里没有 util 时, 租约说「未观测」而不是「干净」—— 缺键不是 0。

运行: python3 -m unittest tests.test_device_lease -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from operator_agent.conductor import Conductor
from operator_agent.kernel_db import KernelDatabase
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)
from operator_agent.placement import LEASE_CONTENDED, LEASE_UNOBSERVED
from tests.unit.test_conductor_stagnation import (SLOW_MATMUL, ScriptedGflopsGPU,
                                             ScriptedLLM)


class _UtilReportingGPU(ScriptedGflopsGPU):
    """在 L3 载荷里报出测量前的卡占用 (远端 runner 报的就是这几个键)"""

    def __init__(self, util_pct):
        super().__init__([10.0])
        self._util = util_pct

    def benchmark(self, kernel_code, inputs, warmup=3, repeat=10):
        result = super().benchmark(kernel_code, inputs, warmup, repeat)
        result["gpu_util_pct"] = self._util
        result["selected_gpu"] = 1
        result["device"] = "cuda:1"
        return result


def _task():
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(framework="pytorch", kernel_type="matmul",
                          fused_ops=[], complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="mock_gpu", dsl="python",
                          compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32,
                                   min_speedup=5.0, max_error=1e-5),
    )


def _inputs():
    rng = np.random.default_rng(0)
    return {"A": rng.standard_normal((64, 64)).astype(np.float32),
            "B": rng.standard_normal((64, 64)).astype(np.float32)}


def _run(backend):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        conductor = Conductor(backend=backend, kernel_db=KernelDatabase(),
                              llm=ScriptedLLM(SLOW_MATMUL), max_optimize_rounds=1)
        artifact = conductor.execute(_task(), _inputs(), target_gflops=0.0)
    return artifact, buf.getvalue()


def _last_l3_metrics(artifact):
    for result in reversed(artifact.verification_results):
        if "device_lease" in (result.metrics or {}):
            return result.metrics
    raise AssertionError("没有任何一轮 L3 的产物里带 device_lease —— 喂给者没接上")


class TestTheLeaseReachesTheArtifact(unittest.TestCase):

    def test_the_card_this_round_used_gets_a_lease(self):
        artifact, _ = _run(_UtilReportingGPU(util_pct=37.0))
        metrics = _last_l3_metrics(artifact)
        lease = metrics["device_lease"]
        self.assertEqual(LEASE_CONTENDED, lease["competition"])
        self.assertAlmostEqual(0.37, lease["external_load"])
        self.assertIsNone(lease["exclusive"],
                          "harness 不租卡 ⇒ 没有任何观测能支撑「独占」这个声明")
        self.assertEqual(1, lease["device"]["index"])
        self.assertEqual("cuda:1", lease["device"]["node"])
        self.assertEqual("", lease["device"]["uuid"],
                         "§11: uuid 才是最稳的锚; 只有索引时必须如实记空 uuid")
        self.assertIn("external_load=0.37", metrics["device_lease_text"])
        self.assertIn("exclusive 未观测", metrics["device_lease_text"])
        # 判决仍然只有测量侧那一处: util=37% 会污染档位, 而租约只是**记录**了它
        self.assertTrue(metrics["benchmark_contaminated"])
        self.assertIn("measurement_validity", metrics)

    def test_a_payload_without_util_reports_unobserved_not_clean(self):
        """本地 MockGPU 不报 util ⇒ 租约说「未观测」(不是「干净」)"""
        artifact, _ = _run(ScriptedGflopsGPU([10.0]))
        metrics = _last_l3_metrics(artifact)
        lease = metrics["device_lease"]
        self.assertIsNone(lease["external_load"])
        self.assertEqual(LEASE_UNOBSERVED, lease["competition"])
        self.assertIn("external_load 未观测", metrics["device_lease_text"])

    def test_the_lease_does_not_manufacture_a_device_identity(self):
        """载荷没有 selected_gpu 时 index 记 -1 (读失败), 不许编一个 0 出来"""
        artifact, _ = _run(ScriptedGflopsGPU([10.0]))
        lease = _last_l3_metrics(artifact)["device_lease"]
        self.assertEqual(-1, lease["device"]["index"])
        self.assertIn("index=-1", lease["device_label"])


if __name__ == "__main__":
    unittest.main(verbosity=2)