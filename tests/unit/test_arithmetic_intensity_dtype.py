"""算数强度的 Byte 轴要用**声明精度**, 不是调用方数组的 dtype

起因是 doc 48 §6.5: 真机那一批 matmul 打出 `AI=682.67 FLOP/Byte`, 而 doc 42 同一个
shape 是 `1365.33` —— 正好差 2 倍。查下来是 `_compute_arithmetic_intensity()` 用
`a.itemsize`(调用方传进来的 numpy 数组), 而 runner 会按声明精度把输入转成 fp16
(`remote_scripts/run_kernel.py`: `precision == "fp16"` 时 `.to(torch.float16)`)。
于是 AI 是在一个**核根本见不到的 dtype** 上算的。

达成率与计时对比都不受影响 (前者与 dtype 无关, 后者候选与基线都被转成 fp16, 公平)。
受影响的是 AI ⇒ 脊点两侧的分类 ⇒ 收敛出口与 SearchAgent 的 bound 感知走哪条。
所以这一组用例的重点不是「数字对不对」, 而是**分类会不会翻**。
"""
import unittest

import numpy as np

from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)
from operator_agent.verifier import Verifier

# doc 42 实测的 H20 dense 规格: fp16 148 TFLOPS / 显存带宽 4.0 TB/s ⇒ 脊点 37 FLOP/Byte
H20_PEAK = {"name": "NVIDIA H20", "device": "NVIDIA H20", "compute_capability": "9.0",
            "fp32_tflops": 44.0, "fp16_tflops": 148.0, "memory_bandwidth_gbs": 4000.0}


def _task(precision=Precision.FP16, kernel_type="matmul"):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
        constraints=ConstraintSpec(precision=precision, min_speedup=1.0, max_error=1e-2),
    )


def _gemm(m, n, k, dtype=np.float32):
    # 只有 shape 与 dtype 进 AI, 值不进 —— 用 empty 免得一例分配 4096² 还去填它
    return {"A": np.empty((m, k), dtype=dtype), "B": np.empty((k, n), dtype=dtype)}


def _ai(task, inputs):
    return Verifier._compute_arithmetic_intensity(task, inputs)


class TestAbsoluteCalibration(unittest.TestCase):
    """先标定再信读数: 钉住 doc 42 那个已知数, 而不是只钉「两倍关系」"""

    def test_matmul_4096_fp16_matches_the_published_number(self):
        ai = _ai(_task(Precision.FP16), _gemm(4096, 4096, 4096))
        self.assertAlmostEqual(1365.3333, ai, places=3,
                               msg="这不是 doc 42 里那个 1365.33 —— Byte 轴又用错 dtype 了")

    def test_the_defective_value_was_exactly_half(self):
        """682.67 是缺陷值: 声明 FP16 却按 float32 数组算 Byte"""
        ai = _ai(_task(Precision.FP16), _gemm(4096, 4096, 4096))
        self.assertNotAlmostEqual(682.6667, ai, places=2)
        self.assertAlmostEqual(682.6667, ai / 2, places=2)


class TestDirectionAndIndependence(unittest.TestCase):
    def test_halving_the_element_width_doubles_the_intensity(self):
        inputs = _gemm(1024, 1024, 1024)
        fp32 = _ai(_task(Precision.FP32), inputs)
        fp16 = _ai(_task(Precision.FP16), inputs)
        bf16 = _ai(_task(Precision.BF16), inputs)
        int8 = _ai(_task(Precision.INT8), inputs)
        self.assertAlmostEqual(2 * fp32, fp16, places=6)
        self.assertAlmostEqual(fp16, bf16, places=6)
        self.assertAlmostEqual(2 * fp16, int8, places=6)

    def test_the_caller_array_dtype_no_longer_matters(self):
        """缺陷的本体: 同一份声明精度下, float32 与 float64 的输入必须给出同一个 AI"""
        task = _task(Precision.FP16)
        self.assertAlmostEqual(_ai(task, _gemm(512, 512, 512, np.float32)),
                               _ai(task, _gemm(512, 512, 512, np.float64)), places=9)

    def test_layernorm_uses_the_declared_precision_too(self):
        """四个分支都要改到; 只改 matmul 那条是最容易漏的一半"""
        inputs = {"X": np.ones((4096, 1024), dtype=np.float32)}
        self.assertAlmostEqual(7.0 / (2 * 2), _ai(_task(Precision.FP16, "layernorm"), inputs),
                               places=9)
        self.assertAlmostEqual(7.0 / (2 * 4), _ai(_task(Precision.FP32, "layernorm"), inputs),
                               places=9)

    def test_every_kernel_branch_scales_with_the_declared_precision(self):
        """`_compute_arithmetic_intensity` 有四个分支, 上一版用例只覆盖了两个

        变异标定的现场: 把 softmax 与 elementwise 两个分支单独改回 `x.itemsize`,
        7 条用例**全绿** —— 也就是说这两条分支上的同一个缺陷可以悄悄回来。
        所以这里按分支逐个断言「FP16 的 AI 正好是 FP32 的两倍」, 而不是再挑一个好写的 shape。
        """
        x = {"X": np.ones((256, 256), dtype=np.float32)}
        cases = {"matmul": _gemm(256, 256, 256), "softmax": x, "relu": x, "layernorm": x}
        for kt, inputs in cases.items():
            with self.subTest(kernel_type=kt):
                fp32 = _ai(_task(Precision.FP32, kt), inputs)
                fp16 = _ai(_task(Precision.FP16, kt), inputs)
                self.assertGreater(fp32, 0.0, f"{kt} 分支根本没算出 AI, 这条断言会空转")
                self.assertAlmostEqual(2 * fp32, fp16, places=9,
                                       msg=f"{kt} 分支的 Byte 轴没有跟随声明精度")


class TestRidgePointClassification(unittest.TestCase):
    def test_a_shape_straddling_the_ridge_is_classified_by_the_declared_precision(self):
        """128×128×4096 的 GEMM: 正确 dtype 给 63.0 (compute), 缺陷 dtype 给 31.5 (memory)

        这个 shape 是**挑出来的**: 它横跨脊点 37, 所以「用错 dtype」在这里不是小数点问题
        而是分类翻转 —— 而分类决定 doc 38 的收敛出口与 SearchAgent 的建议走哪条。
        """
        m, n, k = 128, 128, 4096
        v = Verifier(MockGPU())
        v._theoretical_peak = dict(H20_PEAK)
        r = v._roofline_analysis(_task(Precision.FP16), _gemm(m, n, k),
                                 actual_gflops=1000.0, actual_bandwidth_gbs=100.0)
        self.assertAlmostEqual(37.0, r["ridge_point"], places=3)
        ai_wrong = 2.0 * m * n * k / ((m * k + k * n + m * n) * 4)   # 旧口径
        self.assertLess(ai_wrong, r["ridge_point"], "这个 shape 不再横跨脊点, 用例失去判别力")
        self.assertLess(r["ridge_point"], r["arithmetic_intensity"])
        self.assertEqual("compute", r["roofline_bound"])
        self.assertEqual("compute", r["bound_type"])

    def test_mockgpu_without_a_credible_peak_still_refuses_to_classify(self):
        """反向检查: 修 dtype 不该顺手让「没有可信峰值」的机器也开始下结论"""
        v = Verifier(MockGPU())
        r = v._roofline_analysis(_task(Precision.FP16), _gemm(256, 256, 256),
                                 actual_gflops=1.0, actual_bandwidth_gbs=1.0)
        self.assertEqual("unknown", r["roofline_bound"])


if __name__ == "__main__":
    unittest.main()
