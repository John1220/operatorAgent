"""`benchmarks/inventory_intent_resend.py` 的读者 —— 清点存量产物时「取不到」不许当「没有」

这一片是 #102 的第 3 片 (Limbo `2609.29095v1` §3/§8 那一族: 远端/落盘的写副作用到底
幂等不幂等)。doc 60 §四 给的形状是「先量出同一 intent 被重发的频率，再谈要不要 pin key」，
而**这片真正的产物是两处归因都被读数推翻了**：

  ① **doc 60 §三** 把 12,608 个目录归因成「没有 per-intent key ⇒ 重复留在磁盘上」。拆开数
     之后完整产物那一侧的重发只有 **43 次 / 2,982 份 = 1.4%**，其余是 staging 快照目录。
  ② **本片自己第一版** 在 `render()` 里断言那些多出来的快照目录是「一次运行里的不同轮」
     —— 实测 `extra_by_new_round = 0`、`extra_by_same_round = 674`：一个都不是换轮，
     全部是同一个 task_id 在**同一个轮次**上又开了一次目录（= 重新发过）。

②这一条决定了这套用例的形状：光测数值不够，**解释也要有证人**（I40 那一族：一份报表可以
同时做到每个数字都对、解释是错的，而后者不改变任何数值断言）。所以 `render()` 里那句
「每一轮各占一个目录」被两条用例钉住：不许作为对 `extra_dirs` 的解释出现，而「这两个数不能
相加」那句必须在。

用例分四组：
  · **退出码**（`--selftest` / 根目录不存在 / 一个目录都没数到）—— 后两条是「测不到」与
    「数是 0」的分别（#53/#64 那一族）；
  · **三格分类**（完整产物 / staging / 两种都不是）+ 两条额外的分母外格子
    （读不出的 `result.json`、有 `result.json` 却带子目录的盲区）；
  · **那条恒等式**：`extra_by_new_round + extra_by_same_round == extra_dirs`，两个方向各一棵
    树（全同轮、全新轮）—— 真机存量上「全新轮」那一支今天**一个实例都没有**，所以它必须由
    合成树来覆盖，这条不对称写在下面而不是藏起来；
  · **两个 id 集合的三格**（两处都有 / 只有产物 / 只有快照）与 `render()` 的两处措辞。

本机没有 `output/` 时那一组自动 skip —— 那是**读数条件**，不是「重发率为 0」（doc 49 §7.7
那一族）。

变异标定（表在 `benchmarks/mut_specs/inventory_resend_split.py`。2026-09-30 实跑：基线
**28 passed**，行为分母 **12/12 拦住**、结构分母 0/0）:
    C1 `classify` 把「两种都不是」并进完整产物（第三格消失 ⇒ 分母看着是全覆盖的）
    C2 `_read_json` 失败返回 `{}`（「读不出」那一格从账上蒸发，而不是进 intent 分母）
    C3 盲区计数不看 `result.json` 在不在（把中间层也数进去 ⇒ 印出一个不存在的「少算了 N 份」）
    C4 两个来源颠倒（恒等式**照样成立**，所以「两边相加正好等于它」不是这条的证人）
    C5 两个来源填同一个总数（回到 doc 60 §三 那个「都是重复」的归因）
    C6 `round_hist` 取不到的那一档被当成第 1 轮
    C7 `id_sets` 的 `staging_only` 算成 `len(staging_ids)`（不减交集 ⇒ 与 `intents` 同数，
       读起来像「快照目录都对应的运行都没写完」）
    C8 `render()` 把两个分母相加成一个「总重复数」
    C9 `render()` 里那句被推翻的解释写回来（「一次运行里的每一轮各占一个目录」）—— 数值两版
       全同，只有解释不同 ⇒ 没有任何数值断言杀得掉它
    C10 `main()` 数到 0 个叶子目录时返回 0（空输入读成「没有重发」）
    C11 `staging_no_id` 那 10 个目录并进 `dirs`（「取不到 id」当「是一个新 intent」）
    C12 staging 的判据从「只有快照」放宽成「带快照」（半截写入并进那个分母）

运行: PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest tests/unit/test_inventory_intent_resend.py -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.inventory_intent_resend import (KIND_FULL, KIND_OTHER,  # noqa: E402
                                                KIND_STAGING, classify,
                                                inventory, main, render,
                                                selftest)

RESULT = "result.json"
SNAP = "execution_snapshot.json"
CORPUS = Path(__file__).resolve().parents[2] / "output"


def mk(base: str, name: str, files: dict) -> str:
    p = os.path.join(base, name)
    os.makedirs(p, exist_ok=True)
    for fn, body in files.items():
        full = os.path.join(p, fn)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(body)
    return p


def snap_body(task_id, round_done=None) -> str:
    d = {}
    if task_id is not None:
        d["task_id"] = task_id
    if round_done is not None:
        d["round_done"] = round_done
    return json.dumps(d)


def result_body(task_id, kernel_id="u-0a1b") -> str:
    return json.dumps({"task_id": task_id, "kernel_id": kernel_id})


class TestExitCodes(unittest.TestCase):
    def test_selftest_of_the_script_itself_passes(self):
        """脚本自带的对拍必须为 0 —— 它一红，下面所有用例读的那份实现就不成立了"""
        self.assertEqual(selftest(), 0)

    def test_missing_root_is_a_two_not_a_zero(self):
        """根目录不存在是「测不到」，返回 0 就会被 cron 读成「没有重发」"""
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(main(["--root", os.path.join(td, "nope")]), 2)

    def test_existing_but_empty_root_is_also_a_two(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(main(["--root", td]), 2)

    def test_main_returns_zero_when_it_counted_something(self):
        with tempfile.TemporaryDirectory() as td:
            mk(td, "op/gpu/20260101_101010_uuid", {RESULT: result_body("case-A")})
            self.assertEqual(main(["--root", td]), 0)


class TestThreeBuckets(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.base = self.td.name
        mk(self.base, "op/gpu/r1", {RESULT: result_body("case-A", "k1")})
        mk(self.base, "op/gpu/r2", {RESULT: result_body("case-A", "k2")})
        mk(self.base, "snap_20260101_101010_case-B", {SNAP: snap_body("case-B", 1)})
        mk(self.base, "halfway", {"code.py": "x = 1"})
        self.addCleanup(self.td.cleanup)

    def test_every_leaf_dir_lands_in_exactly_one_bucket(self):
        rep = inventory(self.base)
        d = rep["dirs"]
        self.assertEqual(d["total"], d[KIND_FULL] + d[KIND_STAGING] + d[KIND_OTHER])
        self.assertEqual((d[KIND_FULL], d[KIND_STAGING], d[KIND_OTHER]), (2, 1, 1))

    def test_a_dir_that_is_neither_is_kept_visible_instead_of_dropped(self):
        """C1: 把第三格并进完整产物之后 `total` 还是那个数，所以必须逐格断言"""
        self.assertEqual(classify(mk(self.base, "only_code", {"code.py": ""})), KIND_OTHER)
        self.assertEqual(inventory(self.base)["dirs"][KIND_OTHER], 2)

    def test_result_json_wins_over_a_coexisting_snapshot(self):
        """一轮走完后快照与结果同目录 ⇒ 它是完整产物，不能再当 staging"""
        p = mk(self.base, "both_here", {RESULT: result_body("case-C"), SNAP: "{}"})
        self.assertEqual(classify(p), KIND_FULL)

    def test_snapshot_plus_loose_code_is_a_halfway_write_not_a_snapshot_dir(self):
        """C12: staging 的判据是「只有快照」; 放宽成「带快照」就把半截写入并进那个分母"""
        p = mk(self.base, "halfway2", {SNAP: snap_body("case-E", 1), "code.py": "x = 1"})
        self.assertEqual(classify(p), KIND_OTHER)

    def test_unreadable_result_stays_out_of_the_intent_denominator(self):
        """C2: 「读不出」既不是「没有」也不是「一个空 intent」

        这棵树里两份完整产物同属 `case-A`（那格 intent 只有 1），所以 intent 分母必须是 1
        而不是 2 —— 多出来的那 1 就是「把读不出当成一个空 intent」的读法。
        """
        mk(self.base, "broken", {RESULT: "{不是 JSON"})
        rep = inventory(self.base)
        self.assertEqual(rep["unreadable_result"], 1)
        self.assertEqual(rep["dirs"][KIND_FULL], 3)      # 分类那一格它算完整产物
        self.assertEqual(rep["intents"], 1)              # intent 分母里没有它
        self.assertEqual(rep["resend_full"]["dirs"], 2)
        self.assertEqual(rep["resend_full"]["extra_writes"], 1)


class TestBlindSpotOfLeafOnlyWalk(unittest.TestCase):
    def test_result_in_a_nonleaf_dir_is_counted_as_a_blind_spot(self):
        """C3: 「只看叶子」是这条分母的前提，前提自己必须被数一遍"""
        with tempfile.TemporaryDirectory() as td:
            mk(td, "op/gpu/run/variants", {"v1.py": ""})
            mk(td, "op/gpu/run", {RESULT: result_body("case-D")})
            rep = inventory(td)
            self.assertEqual(rep["nonleaf_with_result"], 1)
            self.assertEqual(rep["intents"], 0)          # 它确实没进分母

    def test_zero_blind_spot_is_reported_as_measured_not_assumed(self):
        with tempfile.TemporaryDirectory() as td:
            mk(td, "op/gpu/r1", {RESULT: result_body("case-A")})
            text = render(inventory(td))
            self.assertIn("盲区", text)
            self.assertIn("逐目录数出来的", text)
            self.assertNotIn("少算了", text)             # 0 的那一支不许印成警告

    def test_a_nonzero_blind_spot_says_the_denominator_is_only_a_floor(self):
        """两个分支各一条: 只测 0 那一支的话, 「永远印警告」也能绿"""
        with tempfile.TemporaryDirectory() as td:
            mk(td, "op/gpu/run/variants", {"v1.py": ""})
            mk(td, "op/gpu/run", {RESULT: result_body("case-D")})
            text = render(inventory(td))
            self.assertIn("少算了 1 份", text)
            self.assertIn("下限", text)


class TestRoundSplitIdentity(unittest.TestCase):
    """`extra_by_new_round + extra_by_same_round == extra_dirs` —— 两个方向各一棵树

    真机存量上 `extra_by_new_round` 今天**等于 0**（没有一个 task_id 的快照目录跨两个轮次），
    所以下面那棵「全新轮」的树是这个分支唯一的证人；这条不对称记在这里，别让下一轮的人
    以为它是从存量上学来的。
    """

    def _split(self, bodies):
        with tempfile.TemporaryDirectory() as td:
            for i, body in enumerate(bodies):
                mk(td, f"snap_20260101_1010{i}_case-A", {SNAP: body})
            return inventory(td)["staging_per_intent"]

    def test_same_round_repeats_are_attributed_to_resend_not_to_rounds(self):
        """这就是推翻本片第一版的那个数: 全同轮 ⇒ 换轮那一侧必须是 0"""
        s = self._split([snap_body("case-A", 1), snap_body("case-A", 1), snap_body("case-A", 1)])
        self.assertEqual((s["extra_by_new_round"], s["extra_by_same_round"]), (0, 2))
        self.assertEqual(s["extra_dirs"], 2)

    def test_a_new_round_per_dir_shows_up_on_the_other_side(self):
        s = self._split([snap_body("case-A", 1), snap_body("case-A", 2), snap_body("case-A", 3)])
        self.assertEqual((s["extra_by_new_round"], s["extra_by_same_round"]), (2, 0))

    def test_the_two_sides_always_sum_to_extra_dirs(self):
        """C4/C5: 挪一侧的数或合并成一个数都在这三条输入下被抓"""
        for bodies in ([snap_body("case-A", 1)] * 4,
                       [snap_body("case-A", r) for r in (1, 2, 3)],
                       [snap_body("case-A", 1), snap_body("case-A", 1), snap_body("case-A", 5)]):
            s = self._split(bodies)
            self.assertEqual(s["extra_by_new_round"] + s["extra_by_same_round"],
                             s["extra_dirs"], f"恒等式在 {bodies} 上破了")

    def test_snapshot_without_round_done_gets_its_own_bucket(self):
        """C6: 取不到轮次不许被当成第 0 轮或某一轮"""
        s = self._split([snap_body("case-A", 1), snap_body("case-A")])
        self.assertEqual(s["round_hist"].get("None"), 1)

    def test_snapshot_without_task_id_is_not_merged_into_the_denominators(self):
        """C11: `no_task_id` 是一格独立的账"""
        with tempfile.TemporaryDirectory() as td:
            mk(td, "snap_a", {SNAP: snap_body("case-A", 1)})
            mk(td, "snap_b", {SNAP: snap_body(None, 2)})
            rep = inventory(td)
            s = rep["staging_per_intent"]
            self.assertEqual((s["dirs"], s["intents"], s["no_task_id"]), (1, 1, 1))
            self.assertEqual(rep["dirs"][KIND_STAGING], 2)   # 分类那一格两个都在


class TestTwoIdSets(unittest.TestCase):
    def test_the_three_set_buckets_are_kept_apart(self):
        """C7: `staging_only` 必须是差集，不是 `len(staging_ids)`"""
        with tempfile.TemporaryDirectory() as td:
            mk(td, "op/gpu/r1", {RESULT: result_body("has-both")})
            mk(td, "snap_1_a", {SNAP: snap_body("has-both", 1)})
            mk(td, "snap_2_b", {SNAP: snap_body("only-snap", 1)})
            rep = inventory(td)
            self.assertEqual(rep["id_sets"],
                             {"both": 1, "full_only": 0, "staging_only": 1})

    def test_full_side_buckets_add_up_to_the_intent_denominator(self):
        with tempfile.TemporaryDirectory() as td:
            mk(td, "op/gpu/r1", {RESULT: result_body("a")})
            mk(td, "op/gpu/r2", {RESULT: result_body("b")})
            mk(td, "snap_1", {SNAP: snap_body("b", 1)})
            mk(td, "snap_2", {SNAP: snap_body("c", 1)})
            st = inventory(td)["id_sets"]
            self.assertEqual(st["both"] + st["full_only"], 2)     # == intents
            self.assertEqual(st["staging_only"], 1)


class TestRenderSentences(unittest.TestCase):
    """I40 那一族: 「为什么这样解释这组数」不改变任何数值断言，所以必须单独钉"""

    def _render(self, bodies):
        with tempfile.TemporaryDirectory() as td:
            for i, b in enumerate(bodies):
                mk(td, f"snap_2026010{i}_case-A", {SNAP: b})
            mk(td, "op/gpu/r1", {RESULT: result_body("case-A", "kkk")})
            return render(inventory(td))

    def test_the_refuted_explanation_must_not_come_back(self):
        """C9: 那句「一次运行里的每一轮各占一个目录」在 2026-09-30 被读数推翻"""
        text = self._render([snap_body("case-A", 1), snap_body("case-A", 1)])
        self.assertNotIn("每一轮各占一个目录", text)
        self.assertIn("同一个轮次再来一次", text)

    def test_the_two_denominators_keep_their_own_labels(self):
        """C8: 把两个数相加的那条变异必然改掉「不能相加」这句, 或改掉两个标签之一"""
        text = self._render([snap_body("case-A", 1)] * 3)
        self.assertIn("不能相加", text)
        self.assertIn("同一个 intent 被重发 (只数完整产物)", text)
        self.assertIn("另一件事 (staging 快照)", text)

    def test_the_printed_counts_come_from_the_report_not_from_the_sentence(self):
        """数字必须是从 `rep` 读的: 改一个读数, 句子必须跟着改（render 里不许写死计数）"""
        with tempfile.TemporaryDirectory() as td:
            mk(td, "op/gpu/r1", {RESULT: result_body("a", "k1")})
            mk(td, "op/gpu/r2", {RESULT: result_body("a", "k2")})
            rep = inventory(td)
            self.assertIn("多余写入 1 次", render(rep))
            rep["resend_full"]["extra_writes"] = 999
            self.assertIn("多余写入 999 次", render(rep))

    def test_intent_invisibility_at_directory_level_is_stated_with_its_numbers(self):
        with tempfile.TemporaryDirectory() as td:
            mk(td, "op/gpu/r1", {RESULT: result_body("a", "k1")})
            text = render(inventory(td))
            self.assertIn("kernel_id ≠ task_id", text)
            self.assertIn("相等的 0 份", text)


@unittest.skipUnless(CORPUS.is_dir(),
                     "本机没有未跟踪的 output/ 语料 —— 这是读数条件, 不是「重发率为 0」")
class TestAgainstTheLocalCorpus(unittest.TestCase):
    """存量语料上只断言**不变量**（不断言条数: 并发会话还在往 output/ 里写新产物）"""

    def setUp(self):
        self.rep = inventory(str(CORPUS))

    def test_the_three_buckets_still_partition_every_leaf_dir(self):
        d = self.rep["dirs"]
        self.assertEqual(d["total"], d[KIND_FULL] + d[KIND_STAGING] + d[KIND_OTHER])
        self.assertGreater(d[KIND_FULL], 0, "完整产物一格为 0 说明走错了目录, 不是「没有产物」")

    def test_the_round_split_identity_holds_on_the_real_corpus(self):
        s = self.rep["staging_per_intent"]
        self.assertEqual(s["extra_by_new_round"] + s["extra_by_same_round"], s["extra_dirs"])

    def test_the_id_set_partition_is_disjoint_and_covers_the_full_side(self):
        st = self.rep["id_sets"]
        self.assertEqual(st["both"] + st["full_only"], self.rep["intents"])
        self.assertGreater(st["staging_only"], 0,
                           "staging 比完整产物多得多却一个「只有快照」都没有，说明两个集合是同一个")

    def test_directory_names_do_not_expose_the_intent(self):
        """doc 60 §四 那句「两种 key 约定」在存量上的实测形状: 目录名用的那个 id 每次都是新值"""
        self.assertEqual(self.rep["kernel_id_equals_task_id"], 0)
        self.assertGreater(self.rep["kernel_id_differs_from_task_id"], 0)


if __name__ == "__main__":
    unittest.main()
