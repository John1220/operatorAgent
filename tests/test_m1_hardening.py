"""M1 正确性加固的单元测试 (P0-2 / P0-4 / P0-5)

覆盖:
- P0-5: CoderAgent._clean_code 不再做 float32->float64 语义改写,
        纯 Python kernel 不再被 Triton 专用管道误删为空串
- P0-4: anticheat.static_audit 静态审计 + shield_torch 运行时屏蔽,
        验证端拒绝回退调用原生算子的"假 kernel"
- P0-2: TestInputGenerator 种子可控 + 多组输入套件,
        Verifier L2 多组输入聚合 / 远程模式全量比对 + 形状检查

运行: python3 tests/test_m1_hardening.py  或  python3 -m unittest tests.test_m1_hardening -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent.agents import CoderAgent
from operator_agent.anticheat import (
    static_audit, shield_torch, CheatDetectedError,
)
from operator_agent.verifier import Verifier
from operator_agent.mock_gpu import MockGPU
from operator_agent.intent_parser import TestInputGenerator
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision, VerificationLevel,
)
from tests.backend_fakes import fake_backend

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


PYTHON_MATMUL = """import numpy as np

def kernel(A, B):
    return A @ B
"""

TRITON_CLEAN = """import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, BLOCK_M: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
    acc += tl.dot(a_ptr, b_ptr)
    tl.store(c_ptr, acc)

def kernel(A, B):
    M, K = A.shape
    C = torch.empty((M, M), device=A.device, dtype=torch.float32)
    matmul_kernel[(1,)](A, B, C, BLOCK_M=64)
    return C
"""

# 作弊 kernel: wrapper 里直接调用 torch.matmul 回退
TRITON_CHEAT = """import triton
import triton.language as tl
import torch

@triton.jit
def dummy_kernel(x_ptr):
    pass

def kernel(A, B):
    return torch.matmul(A, B)
"""


def _make_task(dsl: str = "python") -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl=dsl,
                          compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32,
                                   min_speedup=0.5, max_error=1e-5),
    )


def _make_inputs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"A": (rng.standard_normal((8, 4)) * 0.1).astype(np.float32),
            "B": (rng.standard_normal((4, 8)) * 0.1).astype(np.float32)}


# ─── P0-5: _clean_code 语义安全 ──────────────────────────────

class TestCleanCodeSafety(unittest.TestCase):

    def test_no_float64_rewrite(self):
        """float32 累加器不得被静默改写为 float64"""
        out = CoderAgent._clean_code(TRITON_CLEAN)
        self.assertIn("dtype=tl.float32", out)
        self.assertNotIn("float64", out)
        self.assertNotIn(".to(tl.float64)", out)

    def test_python_kernel_not_emptied(self):
        """纯 Python kernel 经清理后必须保留 (旧实现返回空串)"""
        out = CoderAgent._clean_code(PYTHON_MATMUL)
        self.assertIn("def kernel", out)
        self.assertIn("import numpy", out)

    def test_fenced_python_extracted(self):
        out = CoderAgent._clean_code(f"```python\n{PYTHON_MATMUL}```")
        self.assertIn("def kernel", out)

    def test_triton_kernel_still_cleaned(self):
        out = CoderAgent._clean_code(f"```python\n{TRITON_CLEAN}```")
        self.assertIn("@triton.jit", out)
        self.assertIn("import triton", out)


# ─── P0-4: 静态审计 ──────────────────────────────────────────

class TestStaticAudit(unittest.TestCase):
    def test_detects_torch_matmul_in_triton(self):
        violations = static_audit(TRITON_CHEAT, "triton").violations
        self.assertTrue(any("torch" in v for v in violations))

    def test_clean_triton_passes(self):
        self.assertEqual(static_audit(TRITON_CLEAN, "triton").violations, [])

    def test_python_dsl_allows_numpy(self):
        self.assertEqual(static_audit(PYTHON_MATMUL, "python").violations, [])

    def test_python_dsl_blocks_torch(self):
        code = "import torch\ndef kernel(A, B):\n    return torch.matmul(A, B)"
        self.assertTrue(static_audit(code, "python").violations)

    def test_detects_functional_and_numpy_fallback(self):
        code = "import torch.nn.functional as F\nimport numpy as np\ndef kernel(X):\n    return F.softmax(X, dim=-1)"
        self.assertTrue(static_audit(code, "triton").violations)
        np_code = "import numpy as np\ndef kernel(A, B):\n    return np.matmul(A, B)"
        self.assertTrue(static_audit(np_code, "triton").violations)


# ─── P0-4: 运行时屏蔽 ────────────────────────────────────────

class TestShieldTorch(unittest.TestCase):

    @unittest.skipUnless(HAS_TORCH, "torch 未安装")
    def test_blocks_matmul_inside_shield(self):
        a = torch.randn(4, 4)
        with shield_torch():
            with self.assertRaises(CheatDetectedError):
                torch.matmul(a, a)

    @unittest.skipUnless(HAS_TORCH, "torch 未安装")
    def test_restores_after_exit(self):
        a = torch.randn(4, 4)
        with shield_torch():
            pass
        out = torch.matmul(a, a)  # 退出屏蔽后必须恢复
        self.assertEqual(out.shape, (4, 4))

    @unittest.skipUnless(HAS_TORCH, "torch 未安装")
    def test_runtime_shield_catches_evasive_cheat(self):
        """静态正则 evade (getattr 拼接) 也能被运行时屏蔽拦截"""
        evasive = (
            "import torch\n"
            "def kernel(A, B):\n"
            "    fn = getattr(torch, 'mat' + 'mul')\n"
            "    return fn(A, B)\n"
        )
        verifier = Verifier(MockGPU())
        results = verifier.verify(evasive, _make_task("python"),
                                  _make_inputs(), target_gflops=0.0)
        l2 = results[-1]
        self.assertEqual(l2.level, VerificationLevel.L2_FUNCTION)
        self.assertFalse(l2.passed)


# ─── P0-4 + P0-2: 验证端集成 ─────────────────────────────────

class TestVerifierHardening(unittest.TestCase):

    def test_cheating_triton_kernel_rejected(self):
        """Triton 代码回退 torch.matmul -> L2 反作弊拦截"""
        verifier = Verifier(MockGPU())
        results = verifier.verify(TRITON_CHEAT, _make_task("triton"),
                                  _make_inputs(), target_gflops=0.0)
        self.assertTrue(results[0].passed)   # L1 语法上可通过
        l2 = results[1]
        self.assertEqual(l2.level, VerificationLevel.L2_FUNCTION)
        self.assertFalse(l2.passed)
        self.assertIn("反作弊", l2.message)

    def test_multi_input_suite_aggregates(self):
        """3 组输入套件全部通过, 聚合指标正确"""
        suite = [_make_inputs(seed=s) for s in range(3)]
        verifier = Verifier(MockGPU())
        results = verifier.verify(PYTHON_MATMUL, _make_task(), suite,
                                  target_gflops=0.0)
        l2 = results[1]
        self.assertTrue(l2.passed)
        self.assertEqual(l2.metrics["num_input_sets"], 3)
        self.assertEqual(l2.metrics["compared_elements"], 3 * 64)
        self.assertEqual(l2.metrics["coverage"], 1.0)

    def test_multi_input_suite_fails_on_wrong_kernel(self):
        wrong = "import numpy as np\ndef kernel(A, B):\n    return (A @ B) * 2"
        suite = [_make_inputs(seed=s) for s in range(3)]
        verifier = Verifier(MockGPU())
        results = verifier.verify(wrong, _make_task(), suite, target_gflops=0.0)
        l2 = results[1]
        self.assertFalse(l2.passed)
        self.assertIn("输入集 1/3", l2.message)


class TestL1LocalPrecheck(unittest.TestCase):
    """L1 本地语法预检: SyntaxError 不发去后端/远程编译"""

    def test_syntax_error_blocked_before_backend(self):
        class CountingMockGPU(MockGPU):
            def __init__(self):
                super().__init__()
                self.compile_calls = 0

            def compile(self, code):
                self.compile_calls += 1
                return super().compile(code)

        backend = CountingMockGPU()
        verifier = Verifier(backend)
        broken = "import numpy as np\ndef kernel(A, B:\n    return A @ B"  # 语法错误
        results = verifier.verify(broken, _make_task(), _make_inputs(), target_gflops=0.0)
        self.assertFalse(results[0].passed)
        self.assertIn("本地语法预检", results[0].message)
        self.assertEqual(backend.compile_calls, 0)  # 未触达后端编译

    def test_valid_syntax_proceeds_to_backend(self):
        class CountingMockGPU(MockGPU):
            def __init__(self):
                super().__init__()
                self.compile_calls = 0

            def compile(self, code):
                self.compile_calls += 1
                return super().compile(code)

        backend = CountingMockGPU()
        verifier = Verifier(backend)
        verifier.verify(PYTHON_MATMUL, _make_task(), _make_inputs(), target_gflops=0.0)
        self.assertEqual(backend.compile_calls, 1)


class TestRemoteFullComparison(unittest.TestCase):
    """在控制机上验证「执行侧只回样本」那条 L2 比对路径 (能力空集 ⇒ 无远端协议)"""

    def _verifier(self) -> Verifier:
        return Verifier(fake_backend())

    def test_remote_full_coverage_compare(self):
        results = self._verifier().verify(
            PYTHON_MATMUL, _make_task(), _make_inputs(), target_gflops=0.0,
        )
        l2 = results[1]
        self.assertTrue(l2.passed, l2.message)
        self.assertEqual(l2.metrics["coverage"], 1.0)
        self.assertEqual(l2.metrics["compared_elements"], 64)

    def test_remote_shape_mismatch_fails(self):
        wrong_shape = "import numpy as np\ndef kernel(A, B):\n    return A"
        results = self._verifier().verify(
            wrong_shape, _make_task(), _make_inputs(), target_gflops=0.0,
        )
        l2 = results[1]
        self.assertFalse(l2.passed)
        self.assertIn("形状不匹配", l2.message)

    def test_remote_wrong_values_fail(self):
        wrong = "import numpy as np\ndef kernel(A, B):\n    return (A @ B) * 2"
        results = self._verifier().verify(
            wrong, _make_task(), _make_inputs(), target_gflops=0.0,
        )
        l2 = results[1]
        self.assertFalse(l2.passed)
        self.assertIn("误差", l2.message)

    def test_compact_remote_verification_does_not_need_output_payload(self):
        """新协议只消费远端摘要，不回传 output_sample。

        假件**声明** `Cap.REMOTE_VERIFY` (由「替换了 verify_function」自动推出):
        在新模型下光把方法挂上去是不够的 —— 消费方读的是声明, 不是 hasattr。
        """
        summary = {
            "success": True, "max_error": 0.0, "max_rel_error": 0.0,
            "compared_elements": 64, "coverage": 1.0,
            "reference_type": "pytorch_eager", "device": "cuda:0",
        }
        verifier = Verifier(fake_backend(verify_function=lambda *a, **kw: summary))
        l2 = verifier._verify_function(PYTHON_MATMUL, _make_inputs(), None, _make_task())
        self.assertTrue(l2.passed, l2.message)
        self.assertEqual(l2.metrics["compared_elements"], 64)
        self.assertEqual(l2.metrics["device"], "cuda:0")


# ─── P0-2: 测试输入生成 ──────────────────────────────────────

class TestInputGeneratorSeeding(unittest.TestCase):

    def test_same_seed_reproducible(self):
        a = TestInputGenerator.generate(_make_task(), (8, 8, 4), seed=7)
        b = TestInputGenerator.generate(_make_task(), (8, 8, 4), seed=7)
        np.testing.assert_array_equal(a["A"], b["A"])

    def test_different_seeds_differ(self):
        a = TestInputGenerator.generate(_make_task(), (8, 8, 4), seed=1)
        b = TestInputGenerator.generate(_make_task(), (8, 8, 4), seed=2)
        self.assertFalse(np.array_equal(a["A"], b["A"]))

    def test_default_seed_unpredictable(self):
        a = TestInputGenerator.generate(_make_task())
        b = TestInputGenerator.generate(_make_task())
        self.assertFalse(np.array_equal(a["A"], b["A"]))

    def test_generate_suite(self):
        suite = TestInputGenerator.generate_suite(_make_task(), (8, 8, 4),
                                                  num_sets=5, base_seed=100)
        self.assertEqual(len(suite), 5)
        self.assertFalse(np.array_equal(suite[0]["A"], suite[1]["A"]))
        # base_seed 固定 -> 套件整体可复现
        suite2 = TestInputGenerator.generate_suite(_make_task(), (8, 8, 4),
                                                   num_sets=5, base_seed=100)
        np.testing.assert_array_equal(suite[0]["A"], suite2[0]["A"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
