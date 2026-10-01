"""§10.4 的「有希望的区域已穷尽」—— 停滞出口的另一半（CHECKLOG I64 / Stage 3 第四片）

设计 §10.4 一行两条件：

```
| **可靠停滞** | K 轮 valid 无改善，且**有希望的区域已穷尽** | 必须"穷尽"才算, 不能只因轮数 |
```

前半句早有实现（`max_stagnant_rounds`），后半句在代码里**不存在** ⇒ 循环在「轮数够了」那
一刻就宣布停滞。这个文件守 `operator_agent/exhaustion.py` 的三态判据与它的两种接法：

* **可枚举**（模板型：剪枝后的参数点）⇒ `exhausted` / `unexplored`，并附上「试过几个／
  没试的是哪几个」；
* **不可枚举**（代码型：候选由 LLM/规则按需产生）⇒ `unknown` = **无法确立**，于是停滞条件
  **不成立**（运行由轮次上限与 §10.2 的预算收口），而「达到阈值却没停」这件事进
  `coverage.declined` —— 没有它, 「因为没穷尽所以继续」与「判据根本没人读」长得一样。

真停滞仍能终止的保护在 `TestExhaustedRegionStillTerminates`（可判的那条路径）。

运行: python3 -m unittest tests.test_exhaustion -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.bandit import Bandit
from operator_agent.exhaustion import (EXHAUSTED, FROM_LLM_ON_DEMAND,
                                       FROM_TEMPLATE_SPACE, UNEXPLORED, UNKNOWN,
                                       Coverage, CoverageLedger,
                                       assess_coverage, undeclared)
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                    TargetSpec, TaskComplexity, TaskSchema,
                                    TaskType)


class TestTheThreeStates(unittest.TestCase):

    def setUp(self):
        self.bandit = Bandit()
        self.cands = ["a", "b", "c"]

    def test_all_observed_is_exhausted(self):
        for c in self.cands:
            self.bandit.update(c, 0.1, eligible=True)
        cov = assess_coverage(self.cands, self.bandit, lambda x: x)
        self.assertEqual(EXHAUSTED, cov.state)
        self.assertIs(True, cov.exhausted)
        self.assertEqual(3, cov.total)
        self.assertEqual(3, cov.tried)
        self.assertEqual((), cov.untried)
        self.assertIn("都试过", cov.note())

    def test_an_unobserved_candidate_is_unexplored_and_named(self):
        self.bandit.update("a", 0.1, eligible=True)
        cov = assess_coverage(self.cands, self.bandit, lambda x: x)
        self.assertEqual(UNEXPLORED, cov.state)
        self.assertIs(False, cov.exhausted)
        self.assertEqual(("b", "c"), cov.untried, "没试过的是谁必须能读出来")
        self.assertIn("不许宣布停滞", cov.note())

    def test_an_empty_universe_is_not_exhausted(self):
        """空集**不许**算已穷尽：那会让空队列的每一轮都自证「可以停」"""
        cov = assess_coverage([], self.bandit, lambda x: x)
        self.assertEqual(UNEXPLORED, cov.state)
        self.assertEqual(0, cov.total)
        self.assertIn("空集", cov.reason)
        self.assertIs(False, cov.exhausted)

    def test_none_universe_is_not_a_silent_pass(self):
        cov = assess_coverage(None, self.bandit, lambda x: x)
        self.assertEqual(UNEXPLORED, cov.state)
        self.assertIn("空集", cov.reason)

    def test_unknown_is_neither_true_nor_false(self):
        """无法确立: 三态里的第三个值, **不许**被读成 True/False 的任何一边"""
        cov = undeclared(FROM_LLM_ON_DEMAND, "候选按需产生")
        self.assertEqual(UNKNOWN, cov.state)
        self.assertIsNone(cov.exhausted)
        self.assertIn("无法确立", cov.note())

    def test_the_three_states_have_three_distinct_notes(self):
        notes = {assess_coverage(self.cands, self.bandit, lambda x: x).note(),
                 undeclared(FROM_LLM_ON_DEMAND, "x").note()}
        for c in self.cands:
            self.bandit.update(c, 0.1, eligible=True)
        notes.add(assess_coverage(self.cands, self.bandit, lambda x: x).note())
        self.assertEqual(3, len(notes), "三种状态的说明不许长得一样")

    def test_the_record_carries_the_universe_source(self):
        """「穷尽」是对**哪个集合**说的必须一起记 —— 换个集合, 同一句话含义就变了"""
        cov = assess_coverage(self.cands, self.bandit, lambda x: x,
                              universe=FROM_TEMPLATE_SPACE)
        self.assertEqual(FROM_TEMPLATE_SPACE, cov.as_dict()["universe"])

    def test_labels_are_used_for_the_untried_list(self):
        cov = assess_coverage([1, 2], self.bandit, lambda x: x, label=lambda x: f"点{x}")
        self.assertEqual(("点1", "点2"), cov.untried)


class TestTheLedger(unittest.TestCase):

    def test_a_declined_stop_is_counted_by_state(self):
        """达到阈值却没停 ⇒ 按覆盖状态分开计数 (这是「我们没有宣布停滞」的证据)"""
        led = CoverageLedger()
        led.record(undeclared(FROM_LLM_ON_DEMAND, "x"), threshold_reached=True)
        led.record(undeclared(FROM_LLM_ON_DEMAND, "x"), threshold_reached=True)
        led.record(Coverage(state=EXHAUSTED, universe=FROM_TEMPLATE_SPACE,
                            total=1, tried=1), threshold_reached=False)
        d = led.as_dict()
        self.assertEqual(3, d["rounds"])
        self.assertEqual({"unknown": 2}, d["declined"],
                         "没到阈值的轮次不许计进 declined")
        self.assertEqual(EXHAUSTED, d["last"]["state"])

    def test_without_a_record_the_last_is_explicitly_absent(self):
        d = CoverageLedger().as_dict()
        self.assertIsNone(d["last"], "一轮都没跑 ≠ 有一个状态 (缺 ≠ 0 同一族)")


# ── 接进主循环 ────────────────────────────────────────────────────────────

SLOW_MATMUL = """import numpy as np

def kernel(A, B):
    for _ in range(20):
        C = A @ B
    return C
"""


def _flat_template_backend():
    """模板型 + 平读数: 参数空间可枚举 ⇒ 覆盖可判 (停滞这才算「成立」)

    直接继承用例库里那个假模板后端 (`param_space(contract=None)` /
    `instantiate(template_id, params, contract=None)` 的签名都由它守), 只把读数改成平的。
    """
    from tests.test_template_queue import _TemplateBackend

    class _FlatTemplateBackend(_TemplateBackend):
        def execute(self, kernel_code, inputs):
            result = super().execute(kernel_code, inputs)
            result.gflops = 10.0        # 平: 每轮都不改善
            return result

    return _FlatTemplateBackend()


class _FlatGflopsGPU(MockGPU):
    """代码型 + 平读数: 候选按需产生 ⇒ 覆盖无法确立"""

    def execute(self, kernel_code, inputs):
        result = super().execute(kernel_code, inputs)
        result.gflops = 10.0
        return result


class _DesignerOnlyLLM:
    def chat(self, messages, **kw):
        return f"```python\n{SLOW_MATMUL}```"

    def chat_json(self, messages, **kw):
        return {"algorithm": "matmul",
                "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                "memory_plan": {}, "pipeline": [], "optimization_hints": []}


class ExhaustionCase(unittest.TestCase):

    maxDiff = 600

    def _run(self, backend, rounds=4):
        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(framework="pytorch", kernel_type="matmul",
                              fused_ops=[], complexity=TaskComplexity.MEDIUM),
            target=TargetSpec(hardware="mock_gpu", dsl="python",
                              compute_units=64, warp_size=32),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=99.0,
                                       max_error=1e-5),
        )
        rng = np.random.default_rng(0)
        inputs = {"A": rng.standard_normal((64, 64)).astype(np.float32),
                  "B": rng.standard_normal((64, 64)).astype(np.float32)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            conductor = Conductor(backend=backend, kernel_db=KernelDatabase(),
                                  llm=_DesignerOnlyLLM(),
                                  max_optimize_rounds=rounds)
            artifact = conductor.execute(task, inputs, target_gflops=0.0)
        return artifact, buf.getvalue()


class TestExhaustedRegionStillTerminates(ExhaustionCase):
    """可判路径上的保护: 停滞**两个条件都成立**时照样提前终止"""

    def test_flat_measurements_plus_exhausted_region_stops_early(self):
        artifact, out = self._run(_flat_template_backend(), rounds=4)
        self.assertIn("性能未改善", out)
        self.assertIn("提前终止", out, "候选穷尽 + 连续无改善 ⇒ 停滞可以成立")
        self.assertIn("候选区域已穷尽", out)
        self.assertEqual("stagnant", artifact.metadata["stop"]["reason"])
        self.assertEqual("exhausted", artifact.metadata["coverage"]["last"]["state"])
        self.assertEqual(1, artifact.metadata["coverage"]["last"]["total"],
                         "这台设备的 smem 上界只留得下一个点 (剪枝的已知行为)")

    def test_the_exhausted_stop_names_the_coverage(self):
        artifact, out = self._run(_flat_template_backend(), rounds=4)
        detail = artifact.metadata["stop"]["detail"]
        self.assertIn("穷尽", detail, f"停机理由要说明是**哪一半**成立: {detail}")


class TestUnexhaustedRegionIsNotAStagnation(ExhaustionCase):
    """模板型但**还有没试过的点** ⇒ 不许宣布停滞 (只因轮数 = 设计点名的错读法)"""

    def test_a_second_point_keeps_the_loop_alive(self):
        # 把上界放大到能容下多个点: 剪枝不再只剩一个 ⇒ 第 2、3 轮仍有未观测候选
        import operator_agent.resource_check as rc
        from operator_agent.conductor import Conductor
        from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                            TargetSpec, TaskComplexity, TaskSchema,
                                            TaskType)
        backend = _flat_template_backend()
        real = rc.caps_from_metrics
        rc.caps_from_metrics = lambda src: rc.MemCaps(1024)      # 上界放宽 ⇒ 全部点留下
        try:
            import operator_agent.conductor as cond
            cond.caps_from_metrics = rc.caps_from_metrics
            artifact, out = self._run(backend, rounds=4)
        finally:
            rc.caps_from_metrics = real
            import operator_agent.conductor as cond
            cond.caps_from_metrics = real
        cov = artifact.metadata["coverage"]
        self.assertIn(cov["last"]["state"], ("exhausted", "unexplored"))
        rounds_with_untried = [h["coverage"] for h in artifact.iteration_history
                               if h["coverage"].get("state") == "unexplored"
                               and h["coverage"].get("threshold_reached")]
        self.assertTrue(rounds_with_untried,
                        "阈值到了但还有没试过的点 ⇒ 必须留下「没停」的记录")
        self.assertNotIn("提前终止", out, "还有候选没试过就不许因为轮数停")


class TestTheCodePathDeclinesHonestly(ExhaustionCase):

    def test_unknown_coverage_is_recorded_per_round(self):
        artifact, out = self._run(_FlatGflopsGPU(), rounds=4)
        states = {h["coverage"]["state"] for h in artifact.iteration_history
                  if h["coverage"]}
        self.assertEqual({"unknown"}, states)
        self.assertIn("无法确立", out)

    def test_the_run_still_ends(self):
        """「不停」不等于「无限跑」: 轮次上限与 §10.2 的预算才是收口那两条"""
        artifact, out = self._run(_FlatGflopsGPU(), rounds=3)
        self.assertEqual("rounds_exhausted", artifact.metadata["stop"]["reason"])
        self.assertEqual(3, artifact.metadata["coverage"]["rounds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)