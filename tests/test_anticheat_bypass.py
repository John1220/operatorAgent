"""反作弊绕过回归测试

对应 docs/research/03-反作弊与正确性防线实测.md 第二节的 8 个实测用例。

标记说明:
  被 @unittest.expectedFailure 标记的用例代表"当前会漏过、但应该被拦下"的写法。
  今天它们是 expected failure (套件保持绿色); 一旦 static_audit 修好, 它们会变成
  "unexpected success" 并被报为失败 —— 那是有意的, 逼着改动者摘掉标记, 从而让
  这个文件从"缺陷清单"转成"回归防线"。

  未被标记的用例代表当前已能拦下, 属纯回归保护。

纯本地测试, 不需要 GPU / LLM / 远端。
"""

import unittest

from operator_agent.anticheat import static_audit

# ─── 当前已能拦下 (纯回归保护) ───────────────────────────────

CAUGHT = {
    "torch.matmul 直呼": "def kernel(A,B):\n    import torch\n    return torch.matmul(A,B)\n",
    "torch.einsum": "def kernel(A,B):\n    import torch\n    return torch.einsum('ij,jk->ik',A,B)\n",
    "换别名 import functional": (
        "def kernel(X):\n    import torch.nn.functional as Fn\n    return Fn.softmax(X,-1)\n"
    ),
}

# ─── 当前会漏过, 但应该被拦下 ───────────────────────────────
# 这些都不是刻意构造的对抗样本: A @ B 与 A.matmul(B) 是 PyTorch 里写矩阵乘
# 最自然的两种形态, LLM 在 Triton 路径遇挫后"退一步用 torch 兜底"时最可能写出。

BYPASSED = {
    "@ 运算符 (__matmul__)": "def kernel(A,B):\n    return A @ B\n",
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


class TestStaticAuditCatches(unittest.TestCase):
    """已能拦下的写法不得回归"""

    def test_caught_triton(self):
        for name, code in CAUGHT.items():
            with self.subTest(case=name):
                self.assertTrue(static_audit(code, "triton"),
                                f"{name} 本该被拦下, 现在漏过了 (回归)")

    def test_caught_python(self):
        for name, code in CAUGHT.items():
            with self.subTest(case=name):
                self.assertTrue(static_audit(code, "python"),
                                f"{name} 本该被拦下, 现在漏过了 (回归)")


class TestStaticAuditBypasses(unittest.TestCase):
    """已知漏过的写法: 修好后这些会转为 unexpected success 并报失败, 届时摘掉标记"""

    def _assert_blocked(self, code, dsl):
        self.assertTrue(static_audit(code, dsl), "static_audit 未拦下")

    def test_bypass_triton(self):
        for name, code in BYPASSED.items():
            with self.subTest(case=name):
                self._assert_blocked(code, "triton")

    def test_bypass_python(self):
        for name, code in BYPASSED.items():
            with self.subTest(case=name):
                self._assert_blocked(code, "python")


# 逐个方法挂 expectedFailure, 使 subTest 里任一用例修好都能被察觉
for _m in ("test_bypass_triton", "test_bypass_python"):
    setattr(TestStaticAuditBypasses, _m,
            unittest.expectedFailure(getattr(TestStaticAuditBypasses, _m)))


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
