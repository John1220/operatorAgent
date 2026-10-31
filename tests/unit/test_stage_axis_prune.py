"""S3: stages remains configuration metadata, not a count of physical copies.

The legacy logical tile formula keeps its compatibility arithmetic. Pruning needs
an independently supplied harness layout or native receipt. In these synthetic
layouts A/B copies are explicit, while the scratch buffer has its own dtype and
copy count. Missing stages and missing physical resources are separate unknowns.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from operator_agent.decide.resource_check import (MemCaps, SharedBuffer,  # noqa: E402
                                           SharedMemoryEvidence)
from operator_agent.core.candidates import (ParamSpace, prune_by_smem,  # noqa: E402
                                       stages_of, template_resource_binding, tile_smem_bytes)

# Compatibility logical numbers only; actual layouts below are declared separately.
BM, BN, BK, ELEM = 64, 64, 32, 4
ONE = tile_smem_bytes(BM, BN, BK, ELEM)
FOUR = tile_smem_bytes(BM, BN, BK, ELEM, stages=4)


def _space(stages_values=()):
    axes = {"BLOCK_M": (BM,), "BLOCK_N": (BN,), "BLOCK_K": (BK,)}
    if stages_values:
        axes["stages"] = tuple(stages_values)
    return ParamSpace(template_id="t", axes=axes)


def _layout_receipts(*point_copies):
    """The synthetic harness establishes copies explicitly; stages does not."""
    receipts = {}
    for point, copies in point_copies:
        binding = template_resource_binding(point)
        receipts[binding] = SharedMemoryEvidence(
            binding=binding, kind="trusted_shared_layout", producer="stage-layout-test-harness-v1",
            buffers=(SharedBuffer("A_staging", (64, 32), "float32", copies),
                     SharedBuffer("B_staging", (32, 64), "float32", copies),
                     SharedBuffer("scratch", (64, 64), "float32", 1)))
    return receipts


class TestTheEstimateSaysWhatItAssumes(unittest.TestCase):
    def test_the_multiplier_applies_to_the_ab_tiles_only(self):
        """保留旧逻辑代理公式；这个数不证明物理 shared 占用。"""
        ab = (BM * BK + BK * BN) * ELEM
        for stages in (1, 2, 4, 8):
            with self.subTest(stages=stages):
                self.assertEqual(ONE + ab * (stages - 1),
                                 tile_smem_bytes(BM, BN, BK, ELEM, stages=stages))

    def test_stages_one_is_exactly_the_old_number(self):
        self.assertEqual(tile_smem_bytes(BM, BN, BK, ELEM),
                         tile_smem_bytes(BM, BN, BK, ELEM, stages=1))

    def test_an_unusable_value_is_unobserved_not_one(self):
        for bogus in ("4", 0, -2, True, 4.0, None):
            with self.subTest(value=bogus):
                stages, why = stages_of({"stages": bogus})
                self.assertIsNone(stages, f"{bogus!r} 不是可用的 stages 配置字面量")
                self.assertTrue(why, "「改不动」这件事要说出来, 不能静默当 1")

    def test_the_alias_is_one_word_list(self):
        self.assertEqual((3, ""), stages_of({"num_stages": 3}))
        self.assertEqual((3, ""), stages_of({"STAGES": 3}))
        self.assertEqual((None, ""), stages_of({"num_stages_x": 3}),
                         "只认词表里的名字 (认不出 = 未观测)")


class TestThePruneActuallyUsesIt(unittest.TestCase):
    def test_explicit_layout_copies_distinguish_32_kib_from_80_kib(self):
        self.assertLess(ONE, 48 * 1024)
        self.assertGreater(FOUR, 48 * 1024)
        points = _space((1, 4)).points()
        res = prune_by_smem(points, MemCaps(48), resource_evidence=_layout_receipts(
            (points[0], 1), (points[1], 4)))
        self.assertEqual([1], [c.params["stages"] for c in res.kept],
                         "harness 声明的 80KiB shared 超过 48KiB 宽松阈值 ⇒ 必须摘掉")
        self.assertEqual([4], [c.params["stages"] for c, _ in res.dropped])
        self.assertIn("80.0 KiB", res.dropped[0][1])
        self.assertEqual([4, 4, 1], [buffer["copies"]
                                    for buffer in res.resource_reports[1]["buffers"]])
        self.assertEqual(80 * 1024, res.resource_reports[1]["estimated_smem_bytes"])
        self.assertIn("stages={1,4}", res.as_note(), "日志里要能看出乘过哪几档")
        # Stages remains observable metadata; copies is recorded independently above.
        self.assertEqual((1, 4), res.stages_seen)

    def test_a_fully_observed_axis_keeps_the_note_empty(self):
        """资源布局已确立且没摘点 ⇒ 不应恒发资源未知提示。"""
        points = _space((1,)).points()
        res = prune_by_smem(points, MemCaps(48), resource_evidence=_layout_receipts((points[0], 1)))
        self.assertEqual("", res.as_note())
        self.assertEqual((1,), res.stages_seen)
        self.assertEqual(0, res.stages_unobserved)
        self.assertEqual(0, res.resource_unknown)

    def test_an_unobserved_axis_and_physical_resources_are_separate_unknowns(self):
        res = prune_by_smem(_space().points(), MemCaps(48))
        self.assertEqual(1, res.stages_unobserved)
        self.assertEqual((), res.stages_seen)
        self.assertEqual(1, res.resource_unknown)
        self.assertIsNone(res.resource_reports[0]["estimated_smem_bytes"])
        self.assertIn("未知", res.as_note())
        self.assertNotIn("下界", res.as_note())
        self.assertIn("没摘不等于放得下", res.as_note())

    def test_a_broken_stage_value_does_not_manufacture_one_physical_copy(self):
        res = prune_by_smem(_space(("4",)).points(), MemCaps(48))
        self.assertEqual(1, len(res.kept), "漏拦比误拦好: 认不出的值照旧保留待远端判定")
        self.assertEqual(1, res.stages_unobserved)
        self.assertEqual((), res.stages_seen)
        self.assertEqual(1, res.resource_unknown)
        self.assertIsNone(res.resource_reports[0]["estimated_smem_bytes"])

    def test_the_starving_case_still_keeps_everything(self):
        """全摘光 ⇒ 一个都不摘。加了一轴之后这条不许失效"""
        points = _space((8,)).points()
        res = prune_by_smem(points, MemCaps(8), resource_evidence=_layout_receipts((points[0], 8)))
        self.assertEqual(1, len(res.kept))
        self.assertTrue(res.kept_all_to_avoid_starvation)
        self.assertEqual(1, len(res.dropped), "理由仍留在 dropped 里")

    def test_no_limit_still_means_no_pruning(self):
        res = prune_by_smem(_space((4,)).points(), MemCaps(0))
        self.assertTrue(res.no_limit)
        self.assertEqual(1, len(res.kept))
        self.assertEqual(0, res.stages_unobserved, "缺上界早退时不应声称已检查 stages 元数据")

    def test_observed_stages_without_a_layout_cannot_prune_a_point(self):
        points = _space((1, 4)).points()
        res = prune_by_smem(points, MemCaps(1))
        self.assertEqual(points, res.kept)
        self.assertEqual([], res.dropped)
        self.assertEqual((1, 4), res.stages_seen)
        self.assertEqual(2, res.resource_unknown)
        self.assertFalse(res.kept_all_to_avoid_starvation)

    def test_stage_four_does_not_multiply_a_declared_single_copy_layout(self):
        points = _space((1, 4)).points()
        res = prune_by_smem(points, MemCaps(48), resource_evidence=_layout_receipts(
            (points[0], 1), (points[1], 1)))
        self.assertEqual(points, res.kept)
        self.assertEqual([], res.dropped)
        self.assertEqual([32 * 1024, 32 * 1024],
                         [report["estimated_smem_bytes"] for report in res.resource_reports])
        self.assertEqual(0, res.resource_unknown)


if __name__ == "__main__":
    unittest.main(verbosity=2)
