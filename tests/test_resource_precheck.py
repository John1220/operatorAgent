"""LeetCUDA 借鉴收尾: ① 优化阶梯进配方库 ② 资源可行性预检 (smem/形状)

出处:
  · 阶梯 —— LeetCUDA `kernels/hgemm/{naive,wmma,mma/...}` 的渐进实现 +
    `sgemm.cuh:10-15` 的 Level 1→5 注释
  · 预检 —— `bench/utils.cu:27-33` smem 预检、`book/tests/common_test.h:31-37`
    `book_require_sm()` 的「资源不满足 → SKIP 而不是 FAIL」
"""
import unittest

from operator_agent.hardware_caps import HardwareCaps
from operator_agent.resource_check import (check_resource_feasibility,
                                           check_shape_divisibility,
                                           estimate_matmul_smem_bytes)
from operator_agent.verifier import Verifier
from tests.backend_fakes import DeclaredFakeBackend

MATMUL_BIG = """
import triton
import triton.language as tl

@triton.jit
def kernel(A, B, BLOCK_M: tl.constexpr = 128, BLOCK_N: tl.constexpr = 128,
           BLOCK_K: tl.constexpr = 128):
    pid = tl.program_id(0)
    # tl.dot ...
"""

MATMUL_SMALL = """
import triton
import triton.language as tl

@triton.jit
def kernel(A, B, BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 16,
           BLOCK_K: tl.constexpr = 16):
    pass
"""

AUTOTUNE_HUGE = """
import triton
@triton.autotune(configs=[
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}),
], key=['M', 'N', 'K'])
@triton.jit
def kernel(A, B):
    pass
"""


def _caps(smem_kb):
    return HardwareCaps(vendor="nvidia", device_name="fake",
                        shared_mem_per_sm_kb=smem_kb)


class TestSmemPrecheckCatchesTheOversizedTile(unittest.TestCase):
    def test_estimate_matmul_is_three_tiles_times_elem(self):
        # (64*32 + 32*64 + 64*64) * 2 = 16384
        est = estimate_matmul_smem_bytes(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, 2)
        self.assertEqual((64 * 32 + 32 * 64 + 64 * 64) * 2, est)

    def test_oversized_tile_is_refused_with_block_shapes_in_the_message(self):
        rep = check_resource_feasibility(MATMUL_BIG, _caps(64), "matmul")
        self.assertFalse(rep.feasible, "128³ tile 在 64 KiB 上居然放行了")
        msg = rep.as_message()
        self.assertIn("BLOCK", msg)
        self.assertIn("资源预检", msg)
        self.assertIn("未动 GPU", msg)

    def test_small_tile_passes(self):
        rep = check_resource_feasibility(MATMUL_SMALL, _caps(64), "matmul")
        self.assertTrue(rep.feasible, rep.as_message())

    def test_autotune_takes_the_worst_config(self):
        """autotune 里最大那组 BLOCK 决定预检结论 —— 取最小会漏拦"""
        rep = check_resource_feasibility(AUTOTUNE_HUGE, _caps(96), "matmul")
        self.assertFalse(rep.feasible, "按最坏配置估, 256×256×128 必须被拦")
        self.assertIn("256", rep.as_message())

    def test_no_block_literals_skips_without_refusing(self):
        rep = check_resource_feasibility("def kernel(X):\n    return X\n",
                                         _caps(64), "matmul")
        self.assertTrue(rep.feasible)
        self.assertTrue(rep.notes, "没解析到 BLOCK 时要在 notes 里说明跳过")

    def test_unknown_smem_limit_skips_rather_than_guessing(self):
        rep = check_resource_feasibility(MATMUL_BIG, _caps(0), "matmul")
        self.assertTrue(rep.feasible, "caps 没给 smem 时不许猜一个上限")

    def test_slack_boundary_is_not_a_hard_wall(self):
        """超限 20% 以内放行 (宁可漏拦) —— 边界附近交给真启动"""
        # 估 (16*16+16*16+16*16)*2 = 1536 B; 上限 1300*1.2=1560 → 放行
        # 上限 1200*1.2=1440 → 拒
        ok = check_resource_feasibility(MATMUL_SMALL, _caps(1300 / 1024), "matmul")
        self.assertTrue(ok.feasible)
        bad = check_resource_feasibility(MATMUL_SMALL, _caps(1200 / 1024), "matmul")
        self.assertFalse(bad.feasible)

    def test_direction_control_smaller_limit_is_stricter(self):
        loose = check_resource_feasibility(MATMUL_BIG, _caps(228), "matmul")
        tight = check_resource_feasibility(MATMUL_BIG, _caps(48), "matmul")
        self.assertTrue(loose.feasible, "228 KiB 应该放得下 128³ tile (估算约 96 KiB)")
        self.assertFalse(tight.feasible)
        self.assertLess(tight.estimated_smem_bytes, loose.estimated_smem_bytes * 10)


class TestShapeDivisibilityOnlyWhenHardRequired(unittest.TestCase):
    def test_plain_triton_is_not_refused_for_nondivisible_shapes(self):
        """Triton 自动 mask —— 普通写法下 M % BM != 0 是性能问题不是错误"""
        import numpy as np
        rep = check_shape_divisibility(
            "def kernel(A, B, BLOCK_M: tl.constexpr = 64): pass",
            {"A": np.zeros((100, 8)), "B": np.zeros((8, 8))})
        self.assertTrue(rep.feasible)

    def test_hard_require_marker_is_honoured(self):
        import numpy as np
        code = ("# hard-require: BLOCK_M | M\n"
                "def kernel(A, B, BLOCK_M: tl.constexpr = 64, BLOCK_N: tl.constexpr = 64): pass")
        rep = check_shape_divisibility(code, {"A": np.zeros((100, 8)),
                                              "B": np.zeros((8, 8))})
        self.assertFalse(rep.feasible)
        self.assertIn("BLOCK_M", rep.reason)


class TestLadderSectionIsInRecipeLibrary(unittest.TestCase):
    def test_curriculum_section_lists_the_gem_m_ladder(self):
        from pathlib import Path
        text = Path("operator_agent/skills/optimization_recipes.md").read_text(encoding="utf-8")
        self.assertIn("## 优化阶梯 (Curriculum)", text)
        self.assertIn("GEMM / Matmul 阶梯", text)
        # 阶梯必须按级引用真实配方 ID
        for rid in ("R-MATMUL-04", "R-MATMUL-09", "R-SOFTMAX-01", "R-LN-01"):
            self.assertIn(rid, text, f"阶梯没引用 {rid}")

    def test_ladder_order_is_monotone_in_level(self):
        """L0→L7 顺序出现, 不许乱序 —— 阶梯的价值就是顺序"""
        from pathlib import Path
        import re
        text = Path("operator_agent/skills/optimization_recipes.md").read_text(encoding="utf-8")
        gemm = text.split("### GEMM / Matmul 阶梯")[1].split("### Softmax")[0]
        levels = [int(m) for m in re.findall(r"\| L(\d) \|", gemm)]
        self.assertEqual(list(range(len(levels))), levels,
                         f"GEMM 阶梯级号不是 0..n 递增: {levels}")


class TestVerifierRunsTheResourcePrecheck(unittest.TestCase):
    def test_l1_rejects_oversized_tile_before_ssh(self):
        v = Verifier(DeclaredFakeBackend())
        v._l1_caps = _caps(48)  # 钉死 caps, 不去探测
        r = v._verify_compile(MATMUL_BIG, dsl="triton", input_key_sets=[{"A", "B"}])
        self.assertFalse(r.passed)
        self.assertEqual("resource_infeasible", r.metrics.get("compile_status"))
        self.assertIn("资源预检", r.message)

    def test_l1_still_passes_a_small_tile(self):
        v = Verifier(DeclaredFakeBackend())
        v._l1_caps = _caps(228)
        r = v._verify_compile(MATMUL_SMALL, dsl="triton", input_key_sets=[{"A", "B"}])
        # 假后端 compile 可能过也可能不过; 这里只断言**不是**资源拒绝
        self.assertNotEqual("resource_infeasible", r.metrics.get("compile_status"))


if __name__ == "__main__":
    unittest.main()
