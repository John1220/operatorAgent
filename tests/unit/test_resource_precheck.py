"""LeetCUDA 借鉴收尾: ① 优化阶梯进配方库 ② 资源可行性预检 (smem/形状)

出处:
  · 阶梯 —— LeetCUDA `kernels/hgemm/{naive,wmma,mma/...}` 的渐进实现 +
    `sgemm.cuh:10-15` 的 Level 1→5 注释
  · 预检 —— `bench/utils.cu:27-33` smem 预检、`book/tests/common_test.h:31-37`
    `book_require_sm()` 的「资源不满足 → SKIP 而不是 FAIL」
"""
import unittest
from unittest.mock import Mock, patch

from operator_agent.core.hardware_caps import HardwareCaps
from operator_agent.decide.resource_check import (check_resource_feasibility,
                                           check_shape_divisibility,
                                           estimate_matmul_smem_bytes,
                                           SharedBuffer, SharedMemoryEvidence,
                                           source_resource_binding)
from operator_agent.decide.verifier import Verifier
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

# This synthetic source has one explicit binary/configuration scope. Only tests
# that supply a separate harness storage contract use it; BLOCK-only fixtures
# above continue to exercise unknown physical resources.
MATMUL_SINGLE_LAUNCH = """
import triton
import triton.language as tl

@triton.jit
def _matmul_helper(A, B, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)

def kernel(A, B):
    _matmul_helper[(1,)](A, B, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
                         num_stages=1, num_warps=4)
    return A
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


def _layout(code, *buffers):
    """Synthetic harness storage contract, never inferred from the source tile."""
    return SharedMemoryEvidence(binding=source_resource_binding(code),
                                kind="trusted_shared_layout",
                                producer="resource-precheck-test-harness-v1",
                                buffers=tuple(buffers))


def _three_half_staging_tiles(code, bm=128, bn=128, bk=128):
    return _layout(code, SharedBuffer("A_staging", (bm, bk), "float16"),
                   SharedBuffer("B_staging", (bk, bn), "float16"),
                   SharedBuffer("C_staging", (bm, bn), "float16"))


class TestSmemPrecheckRequiresPhysicalStorageEvidence(unittest.TestCase):
    def test_legacy_logical_tile_proxy_keeps_its_arithmetic(self):
        # Compatibility arithmetic only; this does not prove a shared layout.
        # (64*32 + 32*64 + 64*64) * 2 = 16384
        est = estimate_matmul_smem_bytes(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, 2)
        self.assertEqual((64 * 32 + 32 * 64 + 64 * 64) * 2, est)

    def test_large_block_literals_alone_are_unknown_instead_of_refused(self):
        rep = check_resource_feasibility(MATMUL_BIG, _caps(64), "matmul")
        self.assertTrue(rep.feasible, "BLOCK literals must reach native resource checking")
        self.assertEqual("unknown", rep.assessment)
        self.assertIsNone(rep.estimated_smem_bytes)

    def test_oversized_declared_storage_is_refused_with_source_metadata(self):
        rep = check_resource_feasibility(
            MATMUL_SINGLE_LAUNCH, _caps(64), "matmul",
            resource_evidence=_three_half_staging_tiles(MATMUL_SINGLE_LAUNCH))
        self.assertFalse(rep.feasible, "已声明 96 KiB shared storage 必须在 64 KiB 上被拒")
        self.assertEqual("over_limit", rep.assessment)
        self.assertEqual(96 * 1024, rep.estimated_smem_bytes)
        msg = rep.as_message()
        self.assertIn("BLOCK", msg)
        self.assertIn("资源预检", msg)
        self.assertIn("未动 GPU", msg)

    def test_small_block_literals_continue_without_claiming_fit(self):
        rep = check_resource_feasibility(MATMUL_SMALL, _caps(64), "matmul")
        self.assertTrue(rep.feasible, rep.as_message())
        self.assertEqual("unknown", rep.assessment)
        self.assertIsNone(rep.estimated_smem_bytes)

    def test_autotune_literals_are_metadata_until_physical_storage_is_known(self):
        """配置需逐组留存；逻辑 tile 的最大值不能确定物理 shared 字节。"""
        rep = check_resource_feasibility(AUTOTUNE_HUGE, _caps(96), "matmul")
        self.assertTrue(rep.feasible)
        self.assertEqual("unknown", rep.assessment)
        self.assertIsNone(rep.estimated_smem_bytes)
        self.assertEqual([{"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
                          {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128}],
                         [item["params"] for item in rep.configurations
                          if item["source"] == "triton.Config"])

    def test_no_block_literals_skips_without_refusing(self):
        rep = check_resource_feasibility("def kernel(X):\n    return X\n",
                                         _caps(64), "matmul")
        self.assertTrue(rep.feasible)
        self.assertEqual("unknown", rep.assessment)
        self.assertIsNone(rep.estimated_smem_bytes)
        self.assertTrue(rep.notes, "没解析到 BLOCK 时要在 notes 里说明跳过")

    def test_unknown_smem_limit_skips_rather_than_guessing(self):
        rep = check_resource_feasibility(MATMUL_BIG, _caps(0), "matmul")
        self.assertTrue(rep.feasible, "caps 没给 smem 时不许猜一个上限")
        self.assertEqual("unknown", rep.assessment)
        self.assertIsNone(rep.limit_bytes)
        self.assertIsNone(rep.estimated_smem_bytes)

    def test_known_storage_with_unknown_limit_keeps_bytes_without_claiming_fit(self):
        rep = check_resource_feasibility(
            MATMUL_SINGLE_LAUNCH, _caps(0), "matmul",
            resource_evidence=_three_half_staging_tiles(MATMUL_SINGLE_LAUNCH))
        self.assertTrue(rep.feasible)
        self.assertEqual("unknown", rep.assessment)
        self.assertEqual(96 * 1024, rep.estimated_smem_bytes)
        self.assertIsNone(rep.limit_bytes)

    def test_slack_boundary_is_not_a_hard_wall(self):
        """超限 20% 以内放行 (宁可漏拦) —— 边界附近交给真启动"""
        # Harness 声明 768 个 half = 1536 B; 上限 1300*1.2=1560 → 放行
        # 上限 1200*1.2=1440 → 拒
        evidence = _layout(MATMUL_SINGLE_LAUNCH, SharedBuffer("half_staging", (768,), "float16"))
        ok = check_resource_feasibility(MATMUL_SINGLE_LAUNCH, _caps(1300 / 1024), "matmul",
                                        resource_evidence=evidence)
        self.assertTrue(ok.feasible)
        self.assertEqual("within_limit", ok.assessment)
        self.assertEqual(1536, ok.estimated_smem_bytes)
        self.assertEqual(1300, ok.limit_bytes)
        bad = check_resource_feasibility(MATMUL_SINGLE_LAUNCH, _caps(1200 / 1024), "matmul",
                                         resource_evidence=evidence)
        self.assertFalse(bad.feasible)
        self.assertEqual("over_limit", bad.assessment)
        self.assertEqual(1200, bad.limit_bytes)

    def test_direction_control_smaller_limit_is_stricter(self):
        evidence = _three_half_staging_tiles(MATMUL_SINGLE_LAUNCH)
        loose = check_resource_feasibility(MATMUL_SINGLE_LAUNCH, _caps(228), "matmul",
                                           resource_evidence=evidence)
        tight = check_resource_feasibility(MATMUL_SINGLE_LAUNCH, _caps(48), "matmul",
                                           resource_evidence=evidence)
        self.assertTrue(loose.feasible, "96 KiB 声明 storage 没有超过 228 KiB 宽松拒绝阈值")
        self.assertEqual("within_limit", loose.assessment)
        self.assertFalse(tight.feasible)
        self.assertEqual("over_limit", tight.assessment)
        self.assertEqual(loose.estimated_smem_bytes, tight.estimated_smem_bytes)
        self.assertLess(tight.limit_bytes, loose.limit_bytes)


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
    def test_l1_rejects_established_oversized_storage_before_backend_compile(self):
        compile_backend = Mock(return_value=True)
        v = Verifier(DeclaredFakeBackend(compile=compile_backend))
        v._l1_caps = _caps(48)  # 钉死 caps, 不去探测
        report = check_resource_feasibility(
            MATMUL_SINGLE_LAUNCH, v._l1_caps, "matmul",
            resource_evidence=_three_half_staging_tiles(MATMUL_SINGLE_LAUNCH))
        self.assertEqual("over_limit", report.assessment)
        with patch("operator_agent.decide.resource_check.check_resource_feasibility", return_value=report):
            r = v._verify_compile(MATMUL_SINGLE_LAUNCH, dsl="triton", input_key_sets=[{"A", "B"}])
        self.assertFalse(r.passed)
        self.assertEqual("resource_infeasible", r.metrics.get("compile_status"))
        self.assertEqual("over_limit", r.metrics["resource_precheck"]["assessment"])
        self.assertEqual("trusted_shared_layout", r.metrics["resource_precheck"]["evidence_kind"])
        self.assertIn("资源预检", r.message)
        compile_backend.assert_not_called()

    def test_unknown_large_tile_reaches_backend_compile_and_its_failure_is_preserved(self):
        compile_backend = Mock(side_effect=RuntimeError("synthetic backend resource failure"))
        v = Verifier(DeclaredFakeBackend(compile=compile_backend))
        v._l1_caps = _caps(1)
        r = v._verify_compile(MATMUL_BIG, dsl="triton", input_key_sets=[{"A", "B"}])
        compile_backend.assert_called_once_with(MATMUL_BIG)
        self.assertFalse(r.passed, "资源 unknown 不得变成 L1 PASSED")
        self.assertNotEqual("resource_infeasible", r.metrics.get("compile_status"))
        self.assertEqual("unknown", r.metrics["resource_precheck"]["assessment"])
        self.assertIsNone(r.metrics["resource_precheck"]["estimated_smem_bytes"])
        self.assertIn("synthetic backend resource failure", r.error_log)

    def test_small_tile_l1_pass_comes_from_an_observed_backend_compile(self):
        compile_backend = Mock(return_value=True)
        v = Verifier(DeclaredFakeBackend(compile=compile_backend))
        v._l1_caps = _caps(228)
        r = v._verify_compile(MATMUL_SMALL, dsl="triton", input_key_sets=[{"A", "B"}])
        compile_backend.assert_called_once_with(MATMUL_SMALL)
        self.assertTrue(r.passed)
        self.assertEqual("success", r.metrics.get("compile_status"))
        self.assertEqual("unknown", r.metrics["resource_precheck"]["assessment"])


if __name__ == "__main__":
    unittest.main()


class TestProseIsNotAConfiguration(unittest.TestCase):
    """2026-10-19 真机：Fixer 写下的一行**注释**把预检骗了。

    `# 修复: 移除 BLOCK_M=256 的超大 tile …` —— 预检把注释里的 256 读成真实配置，
    于是正确的修复被同一个错误再拒一次。判据只许读代码。
    """

    def test_a_comment_mentioning_a_big_block_does_not_poison_the_check(self):
        code = "\n".join([
            "import triton",
            "@triton.jit",
            "def kernel(A, B, BLOCK_M: tl.constexpr = 128, BLOCK_N: tl.constexpr = 128,",
            "           BLOCK_K: tl.constexpr = 64):",
            "    # 修复: 移除 BLOCK_M=256 的超大 tile 配置 (smem 384KiB > 228KiB 上限)",
            "    pass",
        ])
        caps = HardwareCaps(vendor="nvidia", device_name="H20", shared_mem_per_sm_kb=228)
        rep = check_resource_feasibility(code, caps, "matmul")
        self.assertTrue(rep.feasible, rep.as_message())
        self.assertEqual("unknown", rep.assessment)
        self.assertIsNone(rep.estimated_smem_bytes)
        shapes = dict(item.split("=") for item in rep.block_shapes)
        self.assertEqual(shapes.get("BLOCK_M"), "128")

    def test_a_docstring_mentioning_a_big_block_is_ignored_too(self):
        code = "\n".join([
            "def kernel(A, B, BLOCK_M: tl.constexpr = 64, BLOCK_N: tl.constexpr = 64,",
            "           BLOCK_K: tl.constexpr = 32):",
            '    """优化说明：早期版本用过 BLOCK_M=512，会 OOM。"""',
            "    pass",
        ])
        caps = HardwareCaps(vendor="nvidia", device_name="H20", shared_mem_per_sm_kb=228)
        rep = check_resource_feasibility(code, caps, "matmul")
        self.assertTrue(rep.feasible, rep.as_message())
        self.assertEqual("unknown", rep.assessment)
        self.assertIsNone(rep.estimated_smem_bytes)

    def test_autotune_dict_keys_survive_the_stripping(self):
        """剥字符串时不能把 autotune 的配置键一起剥掉；它们是源码元数据。"""
        code = "\n".join([
            "@triton.autotune(configs=[",
            "    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),",
            "    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}),",
            "], key=['M'])",
            "def kernel(): pass",
        ])
        caps = HardwareCaps(vendor="nvidia", device_name="H20", shared_mem_per_sm_kb=228)
        rep = check_resource_feasibility(code, caps, "matmul")
        # 这里只问配置字面量是否留存。旧代理按 2B/4B 得到 192/384 KiB，
        # 两个数字都不能证明 Triton 实际分配的 shared storage。
        self.assertEqual("unknown", rep.assessment)
        self.assertIsNone(rep.estimated_smem_bytes)
        self.assertIn("BLOCK_M=256", rep.block_shapes)
        self.assertIn("BLOCK_N=256", rep.block_shapes)

    def test_unparseable_source_still_strips_line_comments(self):
        """语法坏掉的源码走兜底路径，但注释仍然要被去掉。"""
        code = "def kernel(:\n    # BLOCK_M=1024\n    pass\n"
        from operator_agent.decide.resource_check import _extract_block_sizes
        self.assertEqual(_extract_block_sizes(code), [])
