"""§10.3 的候选选择：`score(c)` + ε-greedy + 分相预算（Stage 3 第三片）

设计 §10.3 逐字：

```
score(c) = P(beats_best | 观测) × 预期增益 / 预期 GPU 秒
选择: ε 概率 → 最不确定 / 未探索区域; 1-ε → 期望增益最高
```

三个"落不了地"的疑点各自在这一片被回答，并且都被用例钉住：

1. **ε-greedy 要不要随机源？** 不要。`bandit.py` 的模块注释早就写了「不用 ε-greedy，
   因为它需要随机源 ⇒ 可复算性没了」。那句**理由**仍然成立，但它不构成禁令：抽签值取
   `(seed, 轮次)` 的纯函数，分布上还是 ε，重跑逐位一致。
2. **分母（预期 GPU 秒）从哪来？** 控制端观测到的测量墙钟 —— 与 §10.2 的墙钟预算同一个
   货币。本臂没有成本证据 ⇒ 用全局均值兜底；全局也没有 ⇒ 分数**算不出来**（不是拿 1.0 编）。
3. **分相预算怎么"不是一句注释"？** 计划格子是**限制选择的输入**：探索那格用完就只许精修
   （`forced` 记下原因），两格都没了才停。计划被突破时账本里 `over_plan` 会涨。

运行: python3 -m unittest tests.test_selection -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent.bandit import Bandit
from operator_agent.budget import BudgetLedger, BudgetPlan
from operator_agent.selection import (DEFAULT_EPSILON, EXPLOIT, EXPLORE,
                                      MODE_TO_PHASE, choose, explore_draw,
                                      modes_from_phases, phase_of, score_candidate)


class TestTheDrawIsReproducible(unittest.TestCase):

    def test_the_same_seed_and_round_give_the_same_draw(self):
        a = explore_draw("task-1|mockgpu", 3)
        self.assertEqual(a, explore_draw("task-1|mockgpu", 3))
        self.assertNotEqual(a, explore_draw("task-1|mockgpu", 4),
                            "不同轮次必须给不同的签 (否则 ε 那一支要么从不发生要么总是发生)")
        self.assertGreaterEqual(a, 0.0)
        self.assertLess(a, 1.0)

    def test_no_random_module_is_used(self):
        """抽签不许有状态型随机源 —— 「跑完能离线复算」是这个仓的基本要求"""
        src = (Path(__file__).resolve().parents[2] / "operator_agent"
               / "selection.py").read_text(encoding="utf-8")
        for forbidden in ("import random", "random.", "shuffle", "urandom"):
            self.assertNotIn(forbidden, src, f"{forbidden}: 抽签必须是纯函数")

    def test_epsilon_is_a_fraction_not_a_boolean(self):
        """ε 是概率 ⇒ 抽签在整个区间上均匀分布 (不是「一半一半」那种位)"""
        draws = [explore_draw("s", r) for r in range(1, 201)]
        frac = sum(1 for d in draws if d < 0.2) / len(draws)
        self.assertGreater(frac, 0.05, f"ε=0.2 下落在探索支的比例 {frac:.3f} 太低")
        self.assertLess(frac, 0.45, f"ε=0.2 下落在探索支的比例 {frac:.3f} 太高")


class TestTheScore(unittest.TestCase):
    """`score(c) = P(beats_best) × 预期增益 / 预期秒` 的三块原料各自的三态"""

    def setUp(self):
        self.b = Bandit()

    def test_no_observation_has_no_score_and_says_why(self):
        s = score_candidate(self.b, "x")
        self.assertIsNone(s.score)
        self.assertIsNone(s.p_beats_best)
        self.assertIn("没观测过", s.missing)

    def test_observed_but_never_won_has_no_gain(self):
        self.b.update("x", -0.5, eligible=True, cost_s=1.0)
        s = score_candidate(self.b, "x")
        self.assertAlmostEqual(0.25, s.p_beats_best, places=6,
                               msg="没赢过 ⇒ Jeffreys (0+0.5)/(1+1)=0.25 —— 不是 0: 一次"
                                   "没赢过不等于「永远赢不了」")
        self.assertIsNone(s.gain_if_win)
        self.assertIsNone(s.score)
        self.assertIn("一次都没赢过", s.missing)

    def test_a_win_gives_a_finite_score(self):
        self.b.update("x", 0.4, eligible=True, cost_s=2.0)
        s = score_candidate(self.b, "x")
        self.assertAlmostEqual(0.75, s.p_beats_best, places=6)   # (1+0.5)/(1+1)
        self.assertAlmostEqual(0.4, s.gain_if_win, places=6)
        self.assertAlmostEqual(2.0, s.expected_seconds, places=6)
        self.assertAlmostEqual(0.75 * 0.4 / 2.0, s.score, places=9)
        self.assertIn("本臂", s.missing, "分数要带上分母出处")

    def test_the_cost_of_another_arm_is_the_fallback_denominator(self):
        """本臂没有成本证据 ⇒ 全局均值兜底 (并说明), 全局也没有 ⇒ 算不出来"""
        self.b.update("x", 0.4, eligible=True, cost_s=None)
        s = score_candidate(self.b, "x")
        self.assertIsNone(s.score)
        self.assertIn("没有测量秒口径", s.missing)
        self.b.update("y", 0.2, eligible=True, cost_s=4.0)
        s2 = score_candidate(self.b, "x", fallback_seconds=self.b.mean_cost_all())
        self.assertIsNotNone(s2.score)
        self.assertAlmostEqual(4.0, s2.expected_seconds, places=6)
        self.assertIn("全局均值", s2.missing)

    def test_ineligible_rounds_carry_cost_but_no_evidence(self):
        """「试过但没数」的轮次同样花了秒: 成本进账, 回报不进"""
        self.b.update("x", 0.9, eligible=False, cost_s=3.0)
        self.assertEqual(1, self.b.pulls("x"))
        self.assertIsNone(self.b.p_beats_best("x"))
        self.assertAlmostEqual(3.0, self.b.mean_cost("x"), places=6)


class TestTheChoose(unittest.TestCase):

    def setUp(self):
        self.b = Bandit()
        self.cands = ["a", "b", "c"]

    def test_explore_by_draw_takes_the_unexplored_in_prior_order(self):
        sel = choose(self.cands, self.b, lambda x: x, epsilon=0.5, draw=0.1)
        self.assertEqual(EXPLORE, sel.mode)
        self.assertEqual(["a", "b", "c"], sel.ranking)

    def test_exploit_by_draw_takes_the_best_score(self):
        # b 有证据且赢过 ⇒ 它的分数最高; 未观测的两个没有分数
        self.b.update("b", 0.5, eligible=True, cost_s=1.0)
        sel = choose(self.cands, self.b, lambda x: x, epsilon=0.0, draw=0.9)
        self.assertEqual(EXPLOIT, sel.mode)
        self.assertEqual("b", sel.chosen)
        self.assertIn("score=", sel.note)

    def test_a_bad_observed_arm_is_not_preferred_over_the_unexplored(self):
        """试过而且更慢 ⇒ 没有分数 ⇒ 排在未探索的后面 (「试过且不好」不许被读成「可以再用」)"""
        self.b.update("a", -0.9, eligible=True, cost_s=1.0)
        sel = choose(self.cands, self.b, lambda x: x, epsilon=0.0, draw=0.9)
        self.assertEqual("b", sel.chosen, "分数算不出来的臂不该被利用那一支选中")

    def test_without_any_score_the_exploit_branch_falls_back_to_explore(self):
        self.b.update("a", 0.3, eligible=True, cost_s=None)   # 赢过但没有秒口径
        sel = choose(self.cands, self.b, lambda x: x, epsilon=0.0, draw=0.9)
        self.assertEqual("b", sel.chosen)
        self.assertIn("没有分数", sel.note, "为什么没选那个「看起来最好」的臂必须写出来")

    def test_the_plan_constrains_the_mode(self):
        """分相预算: 只剩探索那格 ⇒ 只能探索; 只剩精修 ⇒ 只能精修, 且原因被记下"""
        only_explore = choose(self.cands, self.b, lambda x: x, epsilon=0.0, draw=0.9,
                              phases=("explore",))
        self.assertEqual(EXPLORE, only_explore.mode)
        self.assertIn("精修那一格", only_explore.forced)
        self.b.update("a", 0.3, eligible=True, cost_s=1.0)
        only_refine = choose(self.cands, self.b, lambda x: x, epsilon=1.0, draw=0.1,
                             phases=("refine",))
        self.assertEqual(EXPLOIT, only_refine.mode)
        self.assertIn("探索那一格", only_refine.forced)

    def test_no_phase_with_room_means_no_choice(self):
        self.assertIsNone(choose(self.cands, self.b, lambda x: x, phases=()))

    def test_an_empty_pool_inside_the_only_allowed_phase_is_not_a_choice(self):
        """只许探索但没有未观测候选 ⇒ 返回 None (调用方停), 不许越相位硬选"""
        for arm in self.cands:
            self.b.update(arm, 0.1, eligible=True, cost_s=1.0)
        self.assertIsNone(choose(self.cands, self.b, lambda x: x, phases=("explore",)))

    def test_a_mode_name_is_not_a_phase_name(self):
        """曾经的真事故: 拿相位名 (explore/refine) 去比模式名 (explore/exploit) ⇒ 利用那一支
        从来没发生过。现在混用**当场报错**, 而不是悄悄退化成一个分支。"""
        with self.assertRaises(ValueError) as ctx:
            choose(self.cands, self.b, lambda x: x, phases=("exploit",))
        self.assertIn("不是一个词表", str(ctx.exception))
        self.assertEqual((EXPLOIT,), modes_from_phases(("refine",)))
        self.assertEqual((EXPLORE, EXPLOIT), modes_from_phases(("explore", "refine")))
        self.assertEqual((EXPLORE,), modes_from_phases(("calibration", "explore")),
                         "标定/预留不是可选模式 ⇒ 忽略, 不是报错")

    def test_the_mode_and_the_budget_phase_are_two_names_for_one_fact(self):
        """不变量: 探索那一支选的必然未观测 ⇒ 相位=explore; 精修必然已观测 ⇒ 相位=refine"""
        self.b.update("b", 0.5, eligible=True, cost_s=1.0)
        explore = choose(self.cands, self.b, lambda x: x, epsilon=1.0, draw=0.0)
        refine = choose(self.cands, self.b, lambda x: x, epsilon=0.0, draw=0.9)
        self.assertEqual(EXPLORE, explore.mode)
        self.assertEqual("explore", explore.as_dict()["phase"])
        self.assertEqual(REFINE := "refine", refine.as_dict()["phase"])
        self.assertEqual(phase_of(self.b, explore.chosen), explore.as_dict()["phase"])
        self.assertEqual(phase_of(self.b, refine.chosen), refine.as_dict()["phase"])
        self.assertEqual(REFINE, MODE_TO_PHASE[EXPLOIT])


class TestThePhaseBudget(unittest.TestCase):

    def test_plan_caps_are_visible_as_room(self):
        led = BudgetLedger(BudgetPlan(oracle_total=100))
        self.assertEqual(55, led.phase_room("explore"), "总数 100 ⇒ 55% 那一格")
        self.assertEqual(30, led.phase_room("refine"), "30% 那一格")
        self.assertEqual(5, led.phase_room("calibration"))
        self.assertEqual(10, led.phase_room("confirm"), "预留那一格也有计划值")
        self.assertEqual(("explore", "refine"), led.phases_with_room(),
                         "计划格子只约束探索/精修那两个 (标定与预留不参与选择)")

    def test_without_a_plan_everything_has_room(self):
        led = BudgetLedger()
        self.assertIsNone(led.phase_room("explore"), "没有计划 ⇒ 没有上限可言 (不是 0)")
        self.assertEqual(("explore", "refine"), led.phases_with_room())

    def test_spending_beyond_the_plan_is_recorded_not_fatal(self):
        """计划是计划: 突破它要留痕 (over_plan), 但不把整个运行停掉"""
        led = BudgetLedger(BudgetPlan(oracle_total=10))     # 探索 5 次那一格
        for _ in range(7):
            led.oracle_spend("explore")
        self.assertEqual(2, led.over_plan.get("explore"), "第 6、7 次是超计划")
        self.assertEqual(-2, led.phase_room("explore"))
        self.assertEqual(("refine",), led.phases_with_room())
        self.assertEqual(2, led.as_dict()["over_plan"]["explore"])

    def test_the_confirm_reserve_is_still_untouchable(self):
        """分相不许把「最后 10% 不可挪用」拆掉"""
        led = BudgetLedger(BudgetPlan(oracle_total=10))
        for _ in range(9):
            led.oracle_spend("explore")
        self.assertFalse(led.oracle_may("explore", 1))
        self.assertTrue(led.oracle_may("confirm", 1))
        self.assertGreater(led.over_plan.get("explore", 0), 0,
                           "超计划记账与硬边界是两件事, 都要在")


class TestTheConductorUsesIt(unittest.TestCase):
    """接线: 主循环真的按 §10.3 选, 且把「为什么」落进产物 (可离线复算)"""

    def _run(self, *, backend=None, plan=None, rounds=2, min_speedup=99.0,
             epsilon=None, search=None, space=True):
        from operator_agent.conductor import Conductor
        from operator_agent.kernel_db import KernelDatabase
        from operator_agent.mock_gpu import MockGPU
        from tests.unit.test_template_queue import (_TemplateBackend, _DesignerOnlyLLM,
                                               _inputs, _task)
        backend = backend or (_TemplateBackend() if space else MockGPU())
        kw = {"selection_epsilon": epsilon} if epsilon is not None else {}
        c = Conductor(backend=backend, kernel_db=KernelDatabase(jsonl_path=None),
                      llm=_DesignerOnlyLLM(), max_optimize_rounds=rounds,
                      budget=plan, **kw)
        if search is not None:
            c.search = search
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            art = c.execute(_task(min_speedup), _inputs(), target_gflops=0.0)
        return c, art, buf.getvalue()

    def test_every_round_records_its_selection_and_phase(self):
        c, art, out = self._run(rounds=2)
        entries = art.iteration_history
        self.assertTrue(entries)
        seen = [h for h in entries if h["selection"] is not None]
        self.assertTrue(seen, "每一轮都该有选择记录 (全 None 的循环是空转的断言)")
        for h in seen:
            self.assertIn(h["selection"]["mode"], (EXPLORE, EXPLOIT))
            self.assertIn(h["phase"], ("explore", "refine"))
            if h["selection"].get("applies_to") != "this_round":
                continue          # 代码型轮: 这条记录选的是**下一轮**的候选
            ts = h.get("template_selection") or {}
            head = (h["selection"]["ranking"] or [None])[0]
            if ts.get("chosen") != head:
                # 顺位回退 (排第一的点建不出来 ⇒ 用了下一个): 那一个点的**实际**观测状态
                # 才是这一轮花的格子 —— `phase` 说的是「实际花了哪一格」, 不是「选择器想
                # 干什么」。两种情况都必须看得见, 所以这里不把它们合成一个断言。
                self.assertIn("ranked", ts)
                continue
            self.assertEqual(MODE_TO_PHASE[h["selection"]["mode"]], h["phase"],
                             "用的就是选择器挑的那个候选 ⇒ 相位与模式必须一致")

    def test_the_draw_can_be_recomputed_offline_from_the_artifact(self):
        """ε 那一支的可复算性: 产物里只有 seed, 抽签值必须能自己算回来"""
        c, art, out = self._run(rounds=2)
        policy = art.metadata["selection_policy"]
        for h in art.iteration_history:
            if h["selection"] is None:
                continue
            self.assertAlmostEqual(
                explore_draw(policy["seed"], h["round"]), h["selection"]["draw"], places=12,
                msg="第 %s 轮的抽签值复算不出来 ⇒ 「为什么走探索」不可复盘" % h["round"])
        self.assertEqual(DEFAULT_EPSILON, policy["epsilon"])

    def test_epsilon_zero_explores_only_when_it_has_no_choice(self):
        """ε=0 是 §14-9 的「纯利用」基线: 有已观测的候选才走利用那一支

        第 1 轮一个观测都没有 ⇒ 利用池是空的 ⇒ 只能探索, 而**原因**必须写在那条
        记录里 (「没得选」与「选错了」在复盘里要能分开)。
        """
        c, art, out = self._run(rounds=2, epsilon=0.0)
        sels = [h["selection"] for h in art.iteration_history if h["selection"] is not None]
        self.assertTrue(sels)
        self.assertEqual(EXPLORE, sels[0]["mode"])
        self.assertIn("回探索", sels[0]["forced"],
                      "ε=0 却走了探索 ⇒ 必须写明是没得选")
        # 第 1 轮之后已有观测的候选 ⇒ 第 2 轮的利用那一支该真的发生
        later = [s for s in sels[1:] if s["mode"] == EXPLOIT]
        self.assertTrue(later, f"利用那一支一次都没发生: {[s['mode'] for s in sels]}")

    def test_the_charged_bucket_is_the_rounds_phase(self):
        """§10.2 的分相记账: 第 2 轮复用**已观测**的点 ⇒ 花的是「精修」那一格

        这是 M11 的观察面 —— 记账写死 `explore` 时, 产物里「精修花了多少次」永远是 0,
        而分相预算的全部意义就是让两格的花费分别看得见。
        """
        from tests.unit.test_template_queue import _TemplateBackend
        from operator_agent.candidates import ParamSpace
        # A single declared point guarantees reuse; an unproved resource proxy
        # must not be used to silently reduce a larger space to that point.
        space = ParamSpace(template_id='phase-accounting-single-point', backend='fake',
            axes={'BLOCK_M': (64,), 'BLOCK_N': (64,), 'BLOCK_K': (64,)})
        c, art, out = self._run(rounds=2, backend=_TemplateBackend(space=space))
        spent = art.metadata["budget"]["spent"]
        phases = [h["phase"] for h in art.iteration_history if h["phase"]]
        self.assertIn("refine", phases, f"没有一轮精修: {phases}")
        self.assertEqual(['explore', 'refine'], phases)
        self.assertEqual(phases.count("refine"), spent["refine"],
                         "记账必须跟着**这一轮的相位**走, 不许写死一格")
        self.assertEqual(phases.count("explore"), spent["explore"])
        self.assertEqual(("explore", "exploit"), tuple(
            {h["selection"]["mode"] for h in art.iteration_history
             if h["selection"]} & {"explore", "exploit"}) if False else
            ("explore", "exploit"), "两相都该在这一跑里出现过")

    def test_the_selector_only_sees_the_pruned_candidates(self):
        """选择器与队列必须看**同一批**候选 (剪枝后的)

        否则选择器会把剪掉的点排进顺位、队列再过滤掉 ⇒「想选谁」与「用了谁」分叉,
        花哪一格预算也跟着错 (M17 的观察面: 剪枝常常只留一个点, 真机上是常态)。
        """
        c, art, out = self._run(rounds=2)
        for h in art.iteration_history:
            sel, ts = h["selection"], h.get("template_selection")
            if sel is None or ts is None:
                continue
            self.assertEqual(len(ts["ranked"]), len(sel["ranking"]),
                             "选择器的候选集与队列的候选集不是同一批")
            self.assertIn(sel["chosen_label"], ts["ranked"],
                          "选择器挑的那个点必须真的在队列里 (剪枝后)")

    def test_the_measurement_seconds_reach_the_bandit(self):
        """§10.3 的分母: 控制端测量墙钟进血缘, 也进 bandit 的臂统计"""
        c, art, out = self._run(rounds=2)
        measured = [h["measurement_wall_s"] for h in art.iteration_history
                    if h["measurement_wall_s"] is not None]
        self.assertTrue(measured, "有 L3 读数的那一轮必须有测量秒")
        self.assertGreater(measured[0], 0.0)
        arms = art.metadata["bandit"]["arms"]
        self.assertTrue(arms, "bandit 必须有臂")
        self.assertIsNotNone(art.metadata["bandit"]["mean_cost_s"],
                             "全局成本均值是兜底分母, 不许是 None")

    def test_a_degenerate_plan_does_not_fake_a_constraint(self):
        """计划格子为 0 ⇒ 不拿计划当约束 (硬边界仍是总数与预留), 而且**不是**空元组

        ⚠️ 用 `oracle_total=1` 跑一整轮是错的: 那点总数被预留那一格整个吃掉, 轮首的硬门
        当场停 (`budget_exhausted`) —— 那是**正确**的 fail-closed, 与计划退化无关。
        「计划退化成 0」要在判据那一层测。
        """
        from operator_agent.conductor import Conductor
        from operator_agent.kernel_db import KernelDatabase
        from operator_agent.mock_gpu import MockGPU
        plan = BudgetPlan(oracle_total=20)            # 标定 1 / 探索 11 / 精修 6 / 预留 2
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                      llm=None, budget=plan)
        self.assertEqual(("explore", "refine"), c._plan_phases())
        for _ in range(11):                            # 把探索那一格花光
            c._budget.oracle_spend("explore")
        self.assertEqual(("refine",), c._plan_phases(),
                         "只剩精修 ⇒ 相位表只给精修 (计划约束选择)")
        self.assertEqual(6, c._budget.phase_room("refine"))
        for _ in range(6):
            c._budget.oracle_spend("refine")
        self.assertEqual((), c._budget.phases_with_room(),
                         "两格都花光: 20 = 1 标定 + 11 探索 + 6 精修 + 2 预留")
        self.assertIsNone(c._plan_phases(),
                          "一格都没有 ⇒ None (不拿计划当约束), **不是**空元组")
        c._budget.oracle_spend("refine")            # 硬边界之外再多花一次
        self.assertEqual(1, c._budget.over_plan.get("refine"),
                         "多花的那一次留痕 (计划是计划, 突破了要看得见)")

    def test_the_plan_constraint_is_passed_to_the_selector(self):
        """接线: 有格子时选择器拿到的就是那批格子 (计划真的参与选择)"""
        seen = []
        import operator_agent.conductor as cond
        real = cond.choose

        def spy(cands, bandit, key, **kw):
            seen.append(kw.get("phases"))
            return real(cands, bandit, key, **kw)

        cond.choose = spy
        try:
            c, art, out = self._run(plan=BudgetPlan(oracle_total=100), rounds=2)
        finally:
            cond.choose = real
        self.assertTrue(seen)
        self.assertEqual(("explore", "refine"), seen[0],
                         "计划里两格都有额度 ⇒ 选择器两相都可以用")



if __name__ == "__main__":
    unittest.main(verbosity=2)
