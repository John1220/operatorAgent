"""§5 bandit 的判据 (确定性 UCB1 + 资格门)

这一层的失效方式有三种, 都得有守卫:
  * **把没资格的回报算进均值** —— §10.5-3「毒不入环」在搜索侧的唯一实现就在这里;
    算进去就等于让一个被污染的读数永久压住/抬高一个臂 (与 §16「让噪声大的数永久拉黑
    配置」同一族)。
  * **把「没观测」当成 0** —— `None` 不是「回报为 0」, 它是「没有回报」。写成 0 会把
    「没数」读成「不变」, 与仓里那条「缺键 ≠ 0」是同一条纪律。
  * **引入随机源** —— 同一份血缘重跑两次给出不同顺序, 可复算性就没了 (血缘里刚记上
    资格位与永久后果, 再塞一个 RNG 就把这件事作废)。

运行: python3 -m unittest tests.test_bandit -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.bandit import Bandit


class TestColdStartKeepsThePrior(unittest.TestCase):

    def test_untried_arms_keep_the_prior_order(self):
        """冷启动 = 先验顺序原样: LLM/规则先验是**先验**不是采纳 (P1-4), 加 bandit 不改这一点"""
        b = Bandit()
        prior = ["c", "a", "b"]
        self.assertEqual(prior, b.order(prior))
        self.assertIsNone(b.score("a"))
        self.assertIsNone(b.mean("a"))

    def test_an_untried_arm_ranks_before_a_scored_one(self):
        """UCB 的无限乐观初始化: 没观测的排前面 (探索), 但各自保持先验顺序"""
        b = Bandit()
        b.update("done", 0.5, eligible=True)
        self.assertEqual(["new1", "new2", "done"], b.order(["new1", "done", "new2"]))


class TestEligibilityGatesTheReward(unittest.TestCase):

    def test_an_ineligible_update_is_counted_but_does_not_scoring(self):
        """核心: 没资格的那一轮只记「试过」, 回报不进均值 (变异点)"""
        b = Bandit()
        b.update("a", 9.9, eligible=False)
        self.assertEqual(1, b.stats["a"]["pulls"], "「试过」要记 —— 否则会重复烧机时")
        self.assertEqual(0, b.stats["a"]["scored"], "没资格的回报不许进统计")
        self.assertIsNone(b.mean("a"), "没有采纳过任何回报 ⇒ 没有均值可言 (不是 0)")
        self.assertEqual(1, b.unscored_updates)

    def test_a_none_reward_is_none_not_zero(self):
        """算不出相对改进 (reward=None) 与「回报为 0」不是一回事"""
        b = Bandit()
        b.update("a", None, eligible=True)
        self.assertIsNone(b.mean("a"))
        b.update("a", 0.0, eligible=True)
        self.assertEqual(0.0, b.mean("a"), "0.0 是**有**回报 (没变), 必须与 None 分得开")

    def test_bool_and_non_numeric_rewards_are_not_counted(self):
        """`isinstance(True, int)` 是真的 —— speedup 那一处踩过同一个坑, 这里显式排除"""
        b = Bandit()
        for bad in (True, False, "0.5", [1], {}):
            b.update("a", bad, eligible=True)
        self.assertIsNone(b.mean("a"))
        self.assertEqual(5, b.unscored_updates)


class TestOrdering(unittest.TestCase):

    def test_a_better_arm_ranks_first(self):
        b = Bandit()
        b.update("good", 0.30, eligible=True)
        b.update("bad", -0.10, eligible=True)
        self.assertEqual(["good", "bad"], b.order(["bad", "good"]))

    def test_ties_keep_the_prior_order(self):
        b = Bandit()
        for arm in ("x", "y", "z"):
            b.update(arm, 0.10, eligible=True)
        self.assertEqual(["x", "y", "z"], b.order(["x", "y", "z"]))
        self.assertEqual(["z", "x", "y"], b.order(["z", "x", "y"]),
                         "同分必须保持**给的**顺序 (稳定排序), 不许改序")

    def test_the_exploration_term_favours_a_less_sampled_arm(self):
        """同均值、样本更少 ⇒ 分数更高 (UCB 的探索项真的在算)"""
        b = Bandit()
        for _ in range(5):
            b.update("many", 0.10, eligible=True)
        b.update("few", 0.10, eligible=True)
        self.assertGreater(b.score("few"), b.score("many"))
        self.assertEqual(["few", "many"], b.order(["many", "few"]))

    def test_the_order_can_be_taken_on_objects(self):
        """调用方给的是 sketch 对象: `key=` 抽出臂名, 返回值仍是原对象 (顺序变了)"""
        b = Bandit()
        b.update("a", 0.5, eligible=True)

        class S:
            def __init__(self, arm):
                self.tiling = type("T", (), {"block_m": arm, "block_n": 1, "block_k": 1})()

            def __repr__(self):
                return f"S({self.tiling.block_m})"

        objs = [S("b"), S("a")]
        out = b.order(objs, key=lambda s: f"{s.tiling.block_m}x1x1")
        # b 没观测过 ⇒ 排前面 (乐观初始化); a 有观测 ⇒ 排后面
        self.assertEqual(["b", "a"], [x.tiling.block_m for x in out])
        self.assertIs(out[0], objs[0], "返回的必须是原来那些对象, 不是键")

    def test_it_is_deterministic(self):
        """同样的历史 ⇒ 同样的顺序 (无 RNG); 源码里不许出现随机源"""
        def build():
            b = Bandit()
            b.update("a", 0.2, eligible=True)
            b.update("b", -0.2, eligible=False)
            b.update("c", 0.05, eligible=True)
            return b
        self.assertEqual(build().order(["a", "b", "c", "d"]),
                         build().order(["a", "b", "c", "d"]))
        src = (Path(__file__).resolve().parents[1] / "operator_agent"
               / "bandit.py").read_text(encoding="utf-8")
        for forbidden in ("import random", "random.", "shuffle", "np.random"):
            self.assertNotIn(forbidden, src, f"{forbidden}: 排序不许有随机源")


class TestAuditability(unittest.TestCase):

    def test_the_state_carries_what_was_not_admitted(self):
        """产物里必须能看见「有几轮的数没进统计」, 否则排序是个没有出处的黑箱"""
        b = Bandit()
        b.update("a", 0.1, eligible=True)
        b.update("a", 0.9, eligible=False)
        d = b.as_dict()
        self.assertEqual(1, d["total_scored"])
        self.assertEqual(1, d["unscored_updates"])
        self.assertEqual({"pulls": 2, "scored": 1, "mean": 0.1, "score": d["arms"]["a"]["score"]},
                         d["arms"]["a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)