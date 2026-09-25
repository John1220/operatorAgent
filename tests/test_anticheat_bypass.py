"""反作弊静态审计回归测试

对应 docs/research/03-反作弊与正确性防线实测.md 第二节的 8 个实测用例，
以及 docs/research/12-正确性验证与reward-hacking防护.md 的吸纳决策。

历史: 这 8 个用例最初有 5 个是漏过的，当时按「单个向量 × 单个 DSL」的粒度逐个
挂 `@unittest.expectedFailure` —— 粒度必须这么细，因为 expectedFailure 只在方法
内**全部** subTest 通过时才翻成 unexpected success，把 dict 塞进一个 subTest
循环会让部分修复被完全掩盖。static_audit 改用 AST 后 10 个标记全部翻成
unexpected success，于是摘掉标记，本文件从「缺陷清单」转为「回归防线」。

仍未修的一项是运行时屏蔽没进真机 runner (TestRuntimeShieldCoverage)，标记保留。

纯本地测试, 不需要 GPU / LLM / 远端。
"""

import unittest

from operator_agent.anticheat import static_audit

# ─── 两个 DSL 下都必须拦下 ───────────────────────────────────
# python DSL 允许 NumPy (它是实现介质)，但下列写法语义明确指向 torch，
# 与 NumPy 无关，所以两个 DSL 下都算回退。
# 这些都不是刻意构造的对抗样本: `A @ B` 与 `A.matmul(B)` 是 PyTorch 里写矩阵乘
# 最自然的两种形态, LLM 在 Triton 路径遇挫后「退一步用 torch 兜底」时最可能写出。

CAUGHT_BOTH_DSL = {
    "torch.matmul 直呼": "def kernel(A,B):\n    import torch\n    return torch.matmul(A,B)\n",
    "torch.einsum": "def kernel(A,B):\n    import torch\n    return torch.einsum('ij,jk->ik',A,B)\n",
    "换别名 import functional": (
        "def kernel(X):\n    import torch.nn.functional as Fn\n    return Fn.softmax(X,-1)\n"
    ),
    "方法形式 .matmul()": "def kernel(A,B):\n    return A.matmul(B)\n",
    "from torch import matmul": (
        "def kernel(A,B):\n    from torch import matmul\n    return matmul(A,B)\n"
    ),
    "getattr + 字符串拼接": (
        'def kernel(A,B):\n    import torch\n    return getattr(torch,"mat"+"mul")(A,B)\n'
    ),
    "torch.nn 模块实例化": (
        "def kernel(X,w,b):\n    import torch\n"
        "    return torch.nn.LayerNorm(X.shape[-1]).to(X.device)(X)\n"
    ),
}

# ─── 仅 GPU DSL 下必须拦下 ───────────────────────────────────
# `A @ B` 是歧义写法: 对 numpy 数组它是合法矩阵乘，对 torch 张量它是回退。
# python DSL 下 NumPy 是实现介质 (anticheat.py 模块 docstring 明说)，本仓自己的
# tests/test_p0_3.py::PYTHON_MATMUL 与 tests/test_m1_hardening.py::PYTHON_MATMUL
# 正是这一形态，属正样本。详见 docs/research/03 的更正记录。

CAUGHT_GPU_DSL_ONLY = {
    "@ 运算符 (__matmul__)": "def kernel(A,B):\n    return A @ B\n",
}

# ─── 合法 kernel: 不得误报 ───────────────────────────────────
# 这两条是 AST 实现的主要误报风险点。`tl.dot` 的方法名 `.dot` 恰好也在 NumPy
# 禁用名单里，只能靠「接收者解析到 triton 模块就走路径规则、不退化成按方法名判」
# 来放过 —— 少了这一步，每一个正经的 Triton matmul kernel 都会被判作弊。

LEGIT_TRITON = """import triton
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
    return C
"""

LEGIT_PYTHON_NUMPY = """import numpy as np

def kernel(A, B):
    return A @ B
"""


class TestStaticAuditCatchesBothDSL(unittest.TestCase):
    """两个 DSL 下都必须拦下的写法，不得回归"""

    def test_triton(self):
        for name, code in CAUGHT_BOTH_DSL.items():
            with self.subTest(case=name):
                self.assertTrue(static_audit(code, "triton"),
                                f"{name} 本该被拦下, 现在漏过了 (回归)")

    def test_python(self):
        for name, code in CAUGHT_BOTH_DSL.items():
            with self.subTest(case=name):
                self.assertTrue(static_audit(code, "python"),
                                f"{name} 本该被拦下, 现在漏过了 (回归)")


class TestStaticAuditCatchesGpuDSL(unittest.TestCase):
    """歧义写法只在 GPU DSL 下断言"""

    def test_triton(self):
        for name, code in CAUGHT_GPU_DSL_ONLY.items():
            with self.subTest(case=name):
                self.assertTrue(static_audit(code, "triton"), f"{name} 漏过 (回归)")

    def test_cuda_c(self):
        for name, code in CAUGHT_GPU_DSL_ONLY.items():
            with self.subTest(case=name):
                self.assertTrue(static_audit(code, "cuda_c"), f"{name} 漏过 (回归)")


class TestStaticAuditNoFalsePositive(unittest.TestCase):
    """反向保护: 收紧规则时最容易打死的就是这几条"""

    def test_legit_triton_kernel_passes(self):
        """`tl.dot` 与 `torch.empty` 都是合法用法，不得被误拦"""
        self.assertEqual(static_audit(LEGIT_TRITON, "triton"), [])

    def test_python_dsl_allows_numpy_at_operator(self):
        """python DSL 下 numpy 的 `@` 是实现介质

        若将来为拦 torch 的 `A @ B` 而把 @ 运算符一律禁掉, 这个用例会失败 ——
        那是有意的。
        """
        self.assertEqual(static_audit(LEGIT_PYTHON_NUMPY, "python"), [])

    def test_python_dsl_allows_numpy_dot_method(self):
        """`.dot()` 是 numpy 数组的合法方法, python DSL 下不得拦"""
        code = "import numpy as np\n\ndef kernel(A,B):\n    return A.dot(B)\n"
        self.assertEqual(static_audit(code, "python"), [])

    def test_numpy_fallback_still_blocked_under_gpu_dsl(self):
        """同一份代码换到 GPU DSL 下就该拦 —— 证明按 DSL 分流真的在起作用"""
        code = "import numpy as np\n\ndef kernel(A,B):\n    return np.matmul(A,B)\n"
        self.assertTrue(static_audit(code, "triton"))

    def test_docstring_mention_not_flagged(self):
        """docstring 里提到 torch.matmul 不算调用

        正则版必须靠 _strip_non_kernel_regions 预删 module docstring 才不误报；
        AST 版里它只是一个 Constant 节点，天然不匹配。
        """
        code = ('"""本 kernel 用 tl.dot 实现，参考结果用 torch.matmul 验证。"""\n'
                "import triton.language as tl\n\n"
                "def kernel(A,B):\n    return tl.dot(A,B)\n")
        self.assertEqual(static_audit(code, "triton"), [])

    def test_main_block_not_flagged(self):
        """`if __name__ == "__main__":` 里的自测代码不算 kernel 实现"""
        code = ("def kernel(A,B):\n    return A*B\n\n"
                'if __name__ == "__main__":\n'
                "    import torch\n"
                "    ref = torch.matmul(A, B)\n")
        self.assertEqual(static_audit(code, "triton"), [])


class TestHardwareWhitelist(unittest.TestCase):
    """自研加速卡白名单: torch.* 是唯一能激活厂商 Tensor Core 的路径

    重构 static_audit 时这一段分支被改写过 (原先有两条等价的 `return []` 与一处
    被后一行覆盖的 rules 赋值)，这里钉住行为，避免改回去。
    """

    def test_ascend_python_dsl_allows_torch_matmul(self):
        code = "import torch\n\ndef kernel(A,B):\n    return torch.matmul(A,B)\n"
        self.assertEqual(static_audit(code, "python", hardware="Ascend910"), [])

    def test_ascend_triton_dsl_allows_torch_but_blocks_numpy(self):
        torch_code = "import torch\n\ndef kernel(A,B):\n    return torch.matmul(A,B)\n"
        self.assertEqual(static_audit(torch_code, "triton", hardware="ascend"), [])
        np_code = "import numpy as np\n\ndef kernel(A,B):\n    return np.matmul(A,B)\n"
        self.assertTrue(static_audit(np_code, "triton", hardware="ascend"),
                        "白名单只放开 torch, 用 NumPy 在 host 侧算出结果仍然是作弊")

    def test_nvidia_not_whitelisted(self):
        code = "import torch\n\ndef kernel(A,B):\n    return torch.matmul(A,B)\n"
        self.assertTrue(static_audit(code, "python", hardware="nvidia_gpu"))


class TestSyntaxErrorFailClosed(unittest.TestCase):
    """解析不出时代入正则兜底，不得因为 AST 抛错就放行"""

    def test_broken_code_still_audited_by_regex(self):
        # 括号未闭合 → ast.parse 抛 SyntaxError；正则仍应抓到 torch.matmul
        code = "def kernel(A,B:\n    import torch\n    return torch.matmul(A,B\n"
        self.assertTrue(static_audit(code, "triton"),
                        "AST 解析失败时必须退回正则, 不能直接返回空列表")


class TestRuntimeShieldCoverage(unittest.TestCase):
    """洞 1: 真机 runner 无运行时反作弊

    这不是行为测试而是结构测试 —— 直接断言 remote_scripts/run_kernel.py 里
    出现了 shield 引用。当前会失败, 修好 (把 shield_torch 移进 runner) 后转为
    unexpected success。
    """

    @unittest.expectedFailure
    def test_runner_applies_shield(self):
        from pathlib import Path
        runner = Path(__file__).resolve().parent.parent / "remote_scripts" / "run_kernel.py"
        src = runner.read_text(encoding="utf-8")
        self.assertTrue(
            "shield_torch" in src or "anticheat" in src,
            "remote_scripts/run_kernel.py 未引用任何运行时反作弊; "
            "shield_torch 目前只在 LocalExecutor 与 MockGPU 生效, "
            "而唯一产出真机 L3 数字的是 runner 这条路径",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
