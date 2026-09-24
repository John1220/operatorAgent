"""Conductor L3 停滞判定的回归测试

钉住的是 `best_gflops` 的记账 bug: 初值 0.0, 而修复前唯一的赋值点被
`best_gflops > 0` 守卫 —— 自锁死锁, 从 0.0 出发永远变不成非零。后果是

  1. 首轮明明拿到了可用的 gflops, 却掉进"性能指标不完整"分支, 被报成缺失;
  2. 该分支无条件禁用本轮配方 (首轮 plan 是在 roofline=None 下按算子类型选的,
     无从判断好坏), 拿噪声做永久决策;
  3. `max_stagnant_rounds=2` 下, 一次正在起效的优化在第 2 轮就被提前终止。

所以这里不只测"首轮建立基线", 还要测"改善趋势能跑满预算"与"真停滞仍能提前
终止"两侧 —— 只测前者会把 bug 修成"干脆不做停滞判定"。

gflops 用脚本化后端注入 (MockGPU.benchmark 是真计时, 逐轮改善无法稳定复现);
kernel 代码保持真执行, L3 因 min_speedup=5.0 每轮都 FAILED, 确保进入停滞分支。

运行: python3 -m unittest tests.test_conductor_stagnation -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity,
    TaskSchema, TaskType,
)


SLOW_MATMUL = """import numpy as np

def kernel(A, B):
    for _ in range(20):
        C = A @ B
    return C
"""

SKETCH = {
    "algorithm": "matmul",
    "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
    "memory_plan": {}, "pipeline": [], "optimization_hints": [],
}


class ScriptedLLM:
    def chat(self, messages, **kw):
        return SLOW_MATMUL

    def chat_json(self, messages, **kw):
        return dict(SKETCH)


class ScriptedGflopsGPU(MockGPU):
    """按轮次注入 gflops, 其余行为 (compile/run/计时) 全部沿用 MockGPU

    注入点是 `execute()` 而不是 `benchmark()`: 非远程后端下 Verifier 的 L3 从
    `backend.execute(...).gflops` 取值 (verifier.py:307), `benchmark()` 返回的
    gflops 只用于远程分支。
    每轮的调用序是 L2 execute → L3 benchmark → L3 execute, 所以用 benchmark
    的调用次数当轮次指针: 只有 L3 那次 execute 会读到本轮的脚本值 (L2 不消费
    gflops, 读到上一轮的无妨)。
    """

    def __init__(self, gflops_seq):
        super().__init__()
        self._seq = list(gflops_seq)
        self._round = 0
        self.observed = []

    def benchmark(self, kernel_code, inputs, warmup=3, repeat=10):
        result = super().benchmark(kernel_code, inputs, warmup, repeat)
        self._round = len(self.observed)
        self.observed.append(self._gflops_for_round())
        return result

    def execute(self, kernel_code, inputs):
        result = super().execute(kernel_code, inputs)
        result.gflops = self._gflops_for_round()
        return result

    def _gflops_for_round(self):
        idx = min(self._round, len(self._seq) - 1)
        return self._seq[idx]


class StagnationCase(unittest.TestCase):

    maxDiff = 600   # Conductor 的 stdout 很长, 失败时不要刷屏

    def _run(self, gflops_seq, max_rounds):
        backend = ScriptedGflopsGPU(gflops_seq)
        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(framework="pytorch", kernel_type="matmul",
                              fused_ops=[], complexity=TaskComplexity.MEDIUM),
            target=TargetSpec(hardware="mock_gpu", dsl="python",
                              compute_units=64, warp_size=32),
            constraints=ConstraintSpec(precision=Precision.FP32,
                                       min_speedup=5.0, max_error=1e-5),
        )
        rng = np.random.default_rng(0)
        inputs = {"A": rng.standard_normal((64, 64)).astype(np.float32),
                  "B": rng.standard_normal((64, 64)).astype(np.float32)}

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            artifact = Conductor(backend=backend, kernel_db=KernelDatabase(),
                                 llm=ScriptedLLM(),
                                 max_optimize_rounds=max_rounds,
                                 ).execute(task, inputs, target_gflops=0.0)
        return artifact, buf.getvalue(), backend

    def _rounds_run(self, out, max_rounds):
        return sum(1 for r in range(1, max_rounds + 1) if f"轮次 {r}/{max_rounds}" in out)


class TestFirstObservationEstablishesBaseline(StagnationCase):
    """首个有效观测必须被认作基线, 而不是被报成「指标不完整」"""

    def test_round_one_reports_baseline_not_incomplete(self):
        _, out, _ = self._run([10.0, 20.0, 40.0], max_rounds=3)
        self.assertIn("建立性能基线", out)
        self.assertNotIn("性能指标不完整", out,
                         "gflops 可用时不该走「指标不完整」分支 (修复前的死锁症状)")

    def test_round_one_does_not_disable_recipes(self):
        """首轮无「之前」可比, 禁用配方等于拿噪声做永久决策

        断言的是后果而不是日志: 修复前首轮走 else 分支, 那里 `disabled_recipes.add()`
        是静默的 (没有 print), 只看 stdout 会漏判。配方被禁用的可观测后果是后续
        轮次 Planner 无方可选。
        """
        _, out, _ = self._run([10.0, 20.0, 40.0], max_rounds=3)
        self.assertNotIn("本轮无新增优化配方", out)
        self.assertEqual(out.count("[Planner] 选中"), 3, "三轮都该拿到配方")

    def test_gflops_reaches_artifact(self):
        """best_gflops 能离开 0.0, 才有资格写进产物/知识库"""
        artifact, _, backend = self._run([10.0, 20.0, 40.0], max_rounds=3)
        self.assertEqual(backend.observed, [10.0, 20.0, 40.0])
        self.assertGreater(artifact.benchmark_data.get("gflops", 0), 0)


class TestImprovingTrendRunsFullBudget(StagnationCase):
    """正在起效的优化不得被提前终止 —— 这是死锁最直接的损失"""

    def test_all_rounds_run_when_gflops_improves(self):
        _, out, _ = self._run([10.0, 20.0, 40.0], max_rounds=3)
        self.assertEqual(self._rounds_run(out, 3), 3)
        self.assertNotIn("提前终止", out)

    def test_improvement_resets_stagnation_counter(self):
        _, out, _ = self._run([10.0, 20.0, 40.0], max_rounds=3)
        self.assertNotIn("性能未改善", out)


class TestRealStagnationStillTerminates(StagnationCase):
    """反向保护: 修死锁不能顺手把停滞判定废掉"""

    def test_flat_gflops_terminates_early(self):
        _, out, _ = self._run([10.0, 10.0, 10.0, 10.0], max_rounds=4)
        self.assertIn("性能未改善", out)
        self.assertIn("提前终止", out)
        self.assertEqual(self._rounds_run(out, 4), 3,
                         "max_stagnant_rounds=2: 基线轮 + 2 轮无改善后应停")

    def test_flat_gflops_disables_recipes_after_baseline(self):
        _, out, _ = self._run([10.0, 10.0, 10.0, 10.0], max_rounds=4)
        after_first = out.split("轮次 2/4", 1)[-1]
        self.assertIn("禁用配方", after_first)


class TestMissingGflopsFirstRound(StagnationCase):
    """gflops 不可用 (=0) 时首轮同样不该禁用配方"""

    def test_zero_gflops_round_one_keeps_recipes(self):
        _, out, _ = self._run([0.0, 0.0, 0.0], max_rounds=3)
        first_round = out.split("轮次 2/3")[0]
        self.assertIn("性能指标不完整", first_round)
        # 同上: else 分支的禁用是静默的, 只能看 Planner 后续还有没有方可选
        self.assertNotIn("本轮无新增优化配方", out)
        self.assertGreaterEqual(out.count("[Planner] 选中"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
