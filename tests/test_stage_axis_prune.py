"""`stages` 那一轴必须进 smem 剪枝（CHECKLOG I66）

剪枝是**代理**（省一次冤枉测量），而它原来只读 tiling 三维：一份声明了
`stages=4`（软件流水 4 份 A/B 缓冲）的空间，估算**还是 1 份** ⇒ 明显放不下的点被放行。
这是 fail-open：剪枝的全部价值就是「别为必然失败的点花机时」。

口径必须写死并测出来（估算器是**代理**，不是测量）：
    smem ≈ BM*BN*elem + (BM*BK + BK*BN)*elem*stages
累加器/输出那块 (BM*BN) **不随流水翻倍** —— 只乘 A/B 那两块。

三态在这里是「未观测 ≠ 1」：没有 `stages` 这一轴、或值不可用（`"4"`/0/`True`）⇒ 按
1 份算，于是那个数是**下界**；下界超上界仍可放心摘点，但**没摘不等于放得下** ——
所以逐步都记下来 (`stages_unobserved`/`stages_seen`)，而不是让读者以为算准了。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from operator_agent.resource_check import MemCaps  # noqa: E402
from operator_agent.candidates import (ParamSpace, prune_by_smem,  # noqa: E402
                                       stages_of, tile_smem_bytes)

#: 1 份缓冲时 ≈32KiB; 4 份时 ≈80KiB ⇒ 48KiB 的卡上「1 份放得下、4 份放不下」
BM, BN, BK, ELEM = 64, 64, 32, 4
ONE = tile_smem_bytes(BM, BN, BK, ELEM)
FOUR = tile_smem_bytes(BM, BN, BK, ELEM, stages=4)


def _space(stages_values=()):
    axes = {"BLOCK_M": (BM,), "BLOCK_N": (BN,), "BLOCK_K": (BK,)}
    if stages_values:
        axes["stages"] = tuple(stages_values)
    return ParamSpace(template_id="t", axes=axes)


class TestTheEstimateSaysWhatItAssumes(unittest.TestCase):
    def test_the_multiplier_applies_to_the_ab_tiles_only(self):
        """累加器那块 (BM*BN) 不随流水翻倍 —— 公式写死在这里, 换口径必须改这条"""
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
                self.assertIsNone(stages, f"{bogus!r} 不是可用的缓冲份数")
                self.assertTrue(why, "「改不动」这件事要说出来, 不能静默当 1")

    def test_the_alias_is_one_word_list(self):
        self.assertEqual((3, ""), stages_of({"num_stages": 3}))
        self.assertEqual((3, ""), stages_of({"STAGES": 3}))
        self.assertEqual((None, ""), stages_of({"num_stages_x": 3}),
                         "只认词表里的名字 (认不出 = 未观测)")


class TestThePruneActuallyUsesIt(unittest.TestCase):
    def test_a_space_that_only_fits_at_one_stage_drops_the_deep_pipeline(self):
        self.assertLess(ONE, 48 * 1024)
        self.assertGreater(FOUR, 48 * 1024)
        res = prune_by_smem(_space((1, 4)).points(), MemCaps(48))
        self.assertEqual([1], [c.params["stages"] for c in res.kept],
                         "4 份缓冲 ≈80KiB 放不进 48KiB 的卡 ⇒ 必须被摘掉")
        self.assertEqual([4], [c.params["stages"] for c, _ in res.dropped])
        self.assertIn("×4 缓冲", res.dropped[0][1], "理由要能看出是乘了几倍")
        self.assertIn("stages={1,4}", res.as_note(), "日志里要能看出乘过哪几档")
        #: 见到过的**全部**取值 (含 1): 产物要能复算出每个点乘了几倍
        self.assertEqual((1, 4), res.stages_seen)

    def test_a_fully_observed_axis_keeps_the_note_empty(self):
        """反向: 乘数齐全且没摘点 ⇒ 仍然没有可报的事 (提示不许恒真)"""
        res = prune_by_smem(_space((1,)).points(), MemCaps(48))
        self.assertEqual("", res.as_note())
        self.assertEqual((1,), res.stages_seen)
        self.assertEqual(0, res.stages_unobserved)

    def test_an_unobserved_axis_is_recorded_as_a_lower_bound(self):
        """没有 `stages` 那一轴 ⇒ 估算是下界。产物里必须说, 否则读者以为算准了"""
        res = prune_by_smem(_space().points(), MemCaps(48))
        self.assertEqual(1, res.stages_unobserved)
        self.assertEqual((), res.stages_seen)
        self.assertIn("下界", res.as_note())
        self.assertIn("没摘不等于放得下", res.as_note())

    def test_a_broken_value_falls_back_to_the_lower_bound_instead_of_guessing(self):
        res = prune_by_smem(_space(("4",)).points(), MemCaps(48))
        self.assertEqual(1, len(res.kept), "漏拦比误拦好: 认不出的值照旧保留待远端判定")
        self.assertEqual(1, res.stages_unobserved, "而且它是下界这一件事照样要记")
        self.assertEqual((), res.stages_seen)

    def test_the_starving_case_still_keeps_everything(self):
        """全摘光 ⇒ 一个都不摘。加了一轴之后这条不许失效"""
        res = prune_by_smem(_space((8,)).points(), MemCaps(8))
        self.assertEqual(1, len(res.kept))
        self.assertTrue(res.kept_all_to_avoid_starvation)
        self.assertEqual(1, len(res.dropped), "理由仍留在 dropped 里")

    def test_no_limit_still_means_no_pruning(self):
        res = prune_by_smem(_space((4,)).points(), MemCaps(0))
        self.assertTrue(res.no_limit)
        self.assertEqual(1, len(res.kept))
        self.assertEqual(0, res.stages_unobserved, "没剪的时候不该编出下界计数")


if __name__ == "__main__":
    unittest.main(verbosity=2)