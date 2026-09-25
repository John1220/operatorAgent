"""L2 数值门 (绝对 + 相对, AND 形式) 的回归测试

钉住的是 docs/research/12 实测出的缺陷: 原门禁是 `max_abs_error <= max_error`,
纯绝对容差, 而三个 driver 都传 max_error=5e-1 —— 在 4096³ fp16 matmul 上相当
于允许把结果整体缩小 14.93% 仍判「功能验证通过」。

这里同时钉两侧:
  - 整体缩放 3% 的错核必须被拦下 (原门禁会放行)
  - 合法但精度差的核 (fp16 累加, 近零参考处 rel_err 可达 2.1e4) 必须放行
    —— 这正是「直接把 max_rel_error <= rtol 加进门禁」这个直觉修法会打死的情况,
    也是必须用 AND 形式而不是 OR 的原因。

纯本地测试, 不需要 GPU / LLM / 远端。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent import MockGPU, Verifier
from operator_agent import tolerance
from operator_agent.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity,
    TaskSchema, TaskType,
)


SCALED_KERNEL = """import numpy as np

def kernel(A, B):
    return (A @ B) * 0.97
"""

CORRECT_KERNEL = """import numpy as np

def kernel(A, B):
    return A @ B
"""

# 近零参考处相对误差爆炸的核: 绝对误差只有 1e-6, 但在 ref 恰好为 0 的元素上
# rel_err = 1e-6/(0+EPS) = 100。AND 形式必须放过它 —— 只看相对项的话它会变成
# 一个巨大的假阳性。
NEAR_ZERO_REF_KERNEL = """import numpy as np

def kernel(A, B):
    return (A @ B) + 1e-6
"""


def _task(precision=Precision.FP32, max_error=1e-5):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=precision, min_speedup=0.0,
                                   max_error=max_error),
    )


def _inputs(n=64, scale=1.0, seed=0, zero_row=False):
    """默认 scale=1.0: 让 |ref| 的峰值到 ~29, 3% 的偏差远超 atol"""
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((n, n)) * scale).astype(np.float32)
    B = (rng.standard_normal((n, n)) * scale).astype(np.float32)
    if zero_row:
        A[0, :] = 0.0        # 让 ref 的第 0 行恰好全为 0, 制造 rel_err 无界点
    return {"A": A, "B": B}


class TestToleranceTable(unittest.TestCase):

    def test_per_precision_values(self):
        self.assertEqual(tolerance.tolerance_for("fp32"), (1e-4, 1e-4, 1.0))
        self.assertEqual(tolerance.tolerance_for("fp16"), (1e-2, 1e-2, 1.0))
        self.assertEqual(tolerance.tolerance_for("bf16"), (1e-2, 1e-2, 1.0))

    def test_unknown_precision_falls_back_to_strictest(self):
        """缺字段时兜底到最严的 fp32 档, 而不是放行"""
        self.assertEqual(tolerance.tolerance_for(""), tolerance.TOLERANCE_BY_PRECISION["fp32"])
        self.assertEqual(tolerance.tolerance_for("fp64"), tolerance.TOLERANCE_BY_PRECISION["fp32"])

    def test_task_max_error_can_only_tighten(self):
        """driver 传的宽松值变 no-op, 严格值仍然生效

        三个 driver 历史上都传 5e-1。那不是疏忽: 在只有绝对项的门禁下, 输出幅值到
        3.35 时确实需要很大的 atol 才能让正确核通过。加上相对项之后它就不再需要,
        而 5e-1 恰好等于「允许整体缩小 14.93%」。
        """
        self.assertEqual(tolerance.effective_atol("fp16", 5e-1)[0], 1e-2,
                         "宽松的任务阈值不得覆盖容差表")
        self.assertEqual(tolerance.effective_atol("fp16", 1e-5)[0], 1e-5,
                         "更严的任务阈值应当生效")

    def test_non_numeric_max_error_ignored(self):
        self.assertEqual(tolerance.effective_atol("fp16", None)[0], 1e-2)
        self.assertEqual(tolerance.effective_atol("fp16", True)[0], 1e-2)


class TestEvaluate(unittest.TestCase):

    def test_nonfinite_always_fails(self):
        """NaN/Inf 不进容差比较: NaN 与任何值比较都是 False, 会让坏点统计静默漏掉它"""
        passed, ratio = tolerance.evaluate(n_bad=0, total=100, has_nonfinite=True, rho=1.0)
        self.assertFalse(passed)
        self.assertEqual(ratio, 0.0)

    def test_empty_output_fails(self):
        passed, _ = tolerance.evaluate(0, 0, False, 1.0)
        self.assertFalse(passed)

    def test_matched_ratio_gate(self):
        self.assertTrue(tolerance.evaluate(0, 1000, False, 1.0)[0])
        self.assertFalse(tolerance.evaluate(1, 1000, False, 1.0)[0])
        self.assertTrue(tolerance.evaluate(1, 1000, False, 0.999)[0])

    def test_ratio_clamped(self):
        _, ratio = tolerance.evaluate(2000, 1000, False, 1.0)
        self.assertEqual(ratio, 0.0)


class TestVerifierL2Gate(unittest.TestCase):
    """端到端: 走 Verifier 的真实 L2 路径 (含反作弊静态审计)

    一律用 FP32 档。原因见 test_mockgpu_fp16_is_not_a_calibration_proxy:
    MockGPU 在 FP16 档下把输入降成 float16 再用 numpy 的 fp16 累加算, 连「正确」
    核都会产生 1.3e-2 的误差与 1.9% 的坏点, 而真机 Triton 的 tl.dot 用 fp32
    累加器。用它标定 fp16 容差会得到一个只对 MockGPU 成立的数字。
    """

    def setUp(self):
        self.verifier = Verifier(MockGPU())

    def _l2(self, code, inputs, precision=Precision.FP32, max_error=1e-5):
        return self.verifier._verify_function(code, inputs, None,
                                              _task(precision, max_error))

    def test_scaled_kernel_rejected(self):
        """整体缩放 3% 的错核必须被拦下 —— 原门禁 (max_error=5e-1) 会放行"""
        r = self._l2(SCALED_KERNEL, _inputs(), max_error=5e-1)
        self.assertFalse(r.passed, f"缩放 3% 的核竟然通过了: {r.message}")
        self.assertLess(r.metrics["matched_ratio"], 0.01)
        self.assertGreater(r.metrics["max_rel_error"], 0.029)

    def test_correct_kernel_accepted(self):
        r = self._l2(CORRECT_KERNEL, _inputs())
        self.assertTrue(r.passed, f"正确核被误杀: {r.message}")
        self.assertEqual(r.metrics["matched_ratio"], 1.0)
        self.assertEqual(r.metrics["bad_elements"], 0)

    def test_tiny_absolute_error_with_huge_relative_error_accepted(self):
        """AND 形式的核心价值: 近零参考处 rel_err 无界, 但 abs_err 在 atol 内就不算坏点

        改成 OR、或者直接用 max_rel_error <= rtol 的话, 这个用例会失败 —— 而那会
        打死每一个输出含近零元素的正确核 (docs/research/12 实测合法 fp16 累加核
        的 max_rel 高达 2.1e4)。
        """
        r = self._l2(NEAR_ZERO_REF_KERNEL, _inputs(zero_row=True))
        self.assertTrue(r.passed, f"绝对误差 1e-6 的核被误杀: {r.message}")
        self.assertGreater(r.metrics["max_rel_error"], 1.0,
                           "构造前提: 这个核在 ref=0 处的相对误差确实很大")
        self.assertEqual(r.metrics["bad_elements"], 0)

    def test_new_metrics_surfaced(self):
        """控制端要能分清「差一点」和「整体缩放错了」, 光有 max_error 不够"""
        r = self._l2(CORRECT_KERNEL, _inputs())
        for key in ("matched_ratio", "bad_elements", "atol", "rtol",
                    "required_matched_ratio", "max_rel_error"):
            self.assertIn(key, r.metrics)

    def test_rtol_is_per_precision(self):
        """rtol 按精度分档 —— 这是原来完全缺失的那一半判据"""
        r32 = self._l2(CORRECT_KERNEL, _inputs(), precision=Precision.FP32)
        r16 = self._l2(CORRECT_KERNEL, _inputs(), precision=Precision.FP16)
        self.assertLess(r32.metrics["rtol"], r16.metrics["rtol"])
        self.assertEqual(r32.metrics["rtol"], 1e-4)
        self.assertEqual(r16.metrics["rtol"], 1e-2)
        self.assertTrue(r32.passed)

    def test_schema_default_max_error_dominates_atol(self):
        """钉住一个容易看漏的事实: ConstraintSpec.max_error 的默认值 1e-5 比容差表
        里每一档的 atol 都严, 所以 effective_atol 的 min() 实际总是取 1e-5,
        容差表的 atol 列只在任务显式传了更宽的值时才起作用。

        这是有意的: 缺了相对项时 atol 必须放得很宽 (driver 传 5e-1) 才能让正确核
        通过; 补上 rtol 之后 atol 收紧不再造成误杀, 于是让更严的那个赢。
        """
        r = self._l2(CORRECT_KERNEL, _inputs(), precision=Precision.FP16)
        self.assertEqual(r.metrics["atol"], 1e-5, "schema 默认 max_error 应当压过表的 1e-2")
        self.assertEqual(r.metrics["rtol"], 1e-2, "rtol 不受 max_error 影响")

    def test_table_atol_applies_when_task_is_looser(self):
        """任务传了宽松值时, 容差表的 atol 才现身, 且按精度分档"""
        r32 = self._l2(CORRECT_KERNEL, _inputs(), precision=Precision.FP32, max_error=5e-1)
        r16 = self._l2(CORRECT_KERNEL, _inputs(), precision=Precision.FP16, max_error=5e-1)
        self.assertEqual(r32.metrics["atol"], 1e-4)
        self.assertEqual(r16.metrics["atol"], 1e-2)

    def test_mockgpu_fp16_is_not_a_calibration_proxy(self):
        """钉住一个实测事实, 免得将来拿 MockGPU 的 fp16 数字去标容差表

        同一个逐元素相等的核, FP16 档下 MockGPU 报出非零误差 —— 因为它把输入
        降成 float16 后用 numpy 的 fp16 累加, 而参考是 fp32。真机 tl.dot 用
        fp32 累加器, 不是这个量级。
        """
        r = self._l2(CORRECT_KERNEL, _inputs(), precision=Precision.FP16)
        self.assertGreater(r.metrics["max_error"], 0.0,
                           "MockGPU 的 fp16 路径应当表现出累加误差; 若这条不再成立, "
                           "说明 MockGPU 的实现变了, 本用例的前提需要重新核实")


class TestMultiInputSetAggregation(unittest.TestCase):
    """多组输入时新指标的聚合语义

    这些指标原先在 _verify_function 的聚合步骤里被整个丢掉 (只保留 max_error /
    compared_elements / coverage), 控制端因此看不到 matched_ratio。
    """

    def setUp(self):
        self.verifier = Verifier(MockGPU())

    def test_correct_over_three_sets(self):
        sets = [_inputs(seed=s) for s in (0, 1, 2)]
        r = self.verifier._verify_function(CORRECT_KERNEL, sets, None, _task())
        self.assertTrue(r.passed, r.message)
        self.assertEqual(r.metrics["num_input_sets"], 3)
        self.assertEqual(r.metrics["matched_ratio"], 1.0)
        self.assertEqual(r.metrics["bad_elements"], 0)
        self.assertEqual(r.metrics["compared_elements"], 3 * 64 * 64)

    def test_one_bad_set_fails_whole_verification(self):
        """任一组不过就整体不过, 且报出是第几组"""
        sets = [_inputs(seed=0), _inputs(seed=1)]
        r = self.verifier._verify_function(SCALED_KERNEL, sets, None, _task())
        self.assertFalse(r.passed)
        self.assertIn("[输入集 1/2]", r.message)


class TestRunnerUsesSharedRule(unittest.TestCase):
    """真机 runner 必须用同一份容差表, 不能各写一套

    runner 是独立部署到远端的单文件, 无法被本地 import (依赖 torch), 所以这里做
    结构断言: 它必须 import 同级的 tolerance 模块, 且不得再出现旧的纯绝对容差
    判定式。
    """

    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.runner = (root / "remote_scripts" / "run_kernel.py").read_text(encoding="utf-8")

    def test_runner_imports_tolerance(self):
        self.assertIn("import tolerance", self.runner)
        self.assertIn("tolerance.effective_atol", self.runner)
        self.assertIn("tolerance.evaluate", self.runner)

    def test_runner_no_longer_gates_on_absolute_error_alone(self):
        self.assertNotIn('"success": max_abs_error <= max_error', self.runner,
                         "纯绝对容差门禁又回来了 (docs/research/12 第一节)")

    def test_runner_uses_and_form(self):
        self.assertIn("(abs_error > atol) & (rel_error > rtol)", self.runner,
                      "必须是 AND 形式; 换成 OR 会误杀近零参考处的正确核")

    def test_runner_checks_nonfinite(self):
        self.assertIn("isfinite", self.runner)


if __name__ == "__main__":
    unittest.main(verbosity=2)
