"""§10.2 三种预算的账本 (`operator_agent/budget.py`)

这一层的失效方式有三种, 都得有守卫:
  * **预留被挪用** —— 「最后 10% 的最终确认预算不可挪用」是这条预算存在的唯一理由;
    没这条, 预算就只是「总数」, 而设计明说它挡的是「一次侥幸读数变成永久真理」。
  * **把没设上限读成「拦过了」** —— 全 `None` 时 `oracle_may` 必须为真**且不假装记账**;
    默认生效的机时承诺会改变每一次真机运行, 那是调用方该拍的板。
  * **相位名写错被静默算成别的相** —— 四格是闭集, 写错当场炸 (fail-closed)。

运行: python3 -m unittest tests.test_budget -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.budget import (
    DEFAULT_SPLIT, MIN_TOTAL_FOR_RESERVE, PHASES, BudgetLedger, BudgetPlan,
)


class TestPlanValidation(unittest.TestCase):

    def test_bad_limits_are_rejected_at_construction(self):
        """非法上限在构造时就响 —— 与其在真机循环深处按 -1 秒判定, 不如当场拒绝"""
        for bad in (0, -1, True, 1.5, "5"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                BudgetPlan(oracle_total=bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                BudgetPlan(llm_tokens=bad)
        for bad in (0, -0.5, True, "10"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                BudgetPlan(wall_clock_s=bad)

    def test_all_off_by_default(self):
        plan = BudgetPlan()
        self.assertFalse(plan.any_set)
        self.assertIsNone(plan.phase_caps())

    def test_caps_sum_to_the_total_exactly(self):
        """账目恒等式: 余数归 confirm ⇒ 四格之和恒等于总数 (1..300 全覆盖)"""
        for total in range(1, 301):
            caps = BudgetPlan(oracle_total=total).phase_caps()
            self.assertEqual(total, sum(caps.values()), f"total={total}: {caps}")

    def test_the_reserve_exists_from_the_documented_threshold(self):
        """阈值写进用例, 不许悄悄放大: 从 `MIN_TOTAL_FOR_RESERVE` 起预留恒 ≥ 1"""
        for total in range(1, MIN_TOTAL_FOR_RESERVE):
            caps = BudgetPlan(oracle_total=total).phase_caps()
            self.assertEqual(1, caps["confirm"], f"total={total} 该留 1 次: {caps}")
            self.assertEqual(total, sum(caps.values()), f"total={total}: {caps}")
        # total < 3 放不下三格 ⇒ 先收标定的保底, 再收 explore; 只剩 1 次时全留给预留
        self.assertEqual({"calibration": 0, "explore": 1, "refine": 0, "confirm": 1},
                         BudgetPlan(oracle_total=2).phase_caps())
        self.assertEqual({"calibration": 0, "explore": 0, "refine": 0, "confirm": 1},
                         BudgetPlan(oracle_total=1).phase_caps())
        for total in range(MIN_TOTAL_FOR_RESERVE, 60):
            caps = BudgetPlan(oracle_total=total).phase_caps()
            self.assertGreaterEqual(caps["confirm"], 1, f"total={total}: {caps}")
            self.assertGreaterEqual(caps["calibration"], 1, f"total={total}: {caps}")

    def test_the_split_follows_the_design_ratios(self):
        """大预算下四格必须贴着 5/55/30/10 —— 份额只有一个出处 (`DEFAULT_SPLIT`)"""
        caps = BudgetPlan(oracle_total=100).phase_caps()
        self.assertEqual({"calibration": 5, "explore": 55, "refine": 30, "confirm": 10},
                         caps)
        self.assertAlmostEqual(1.0, sum(DEFAULT_SPLIT.values()), places=12,
                               msg="份额之和必须是 1")


class TestTheReserveIsNotFungible(unittest.TestCase):
    """本模块存在的唯一理由"""

    def setUp(self):
        # total=10 ⇒ 标定 1 / 探索 5 / 精修 3 / 确认 1 (余数归 confirm)
        self.led = BudgetLedger(BudgetPlan(oracle_total=10))

    def test_non_confirm_spending_stops_at_the_reserve(self):
        for _ in range(9):                     # 10 - 1(预留) = 9 次
            self.assertTrue(self.led.oracle_may("explore"))
            self.led.oracle_spend("explore")
        self.assertFalse(self.led.oracle_may("explore"),
                         "非 confirm 的开销碰到了预留")
        self.assertEqual(9, self.led.spent_all)

    def test_confirm_can_still_spend_after_explore_is_refused(self):
        for _ in range(9):
            self.led.oracle_spend("explore")
        self.assertTrue(self.led.oracle_may("confirm"), "预留被前面的开销吃掉了")
        self.led.oracle_spend("confirm")
        self.assertFalse(self.led.oracle_may("confirm"), "预留那一格只有一次")
        self.assertFalse(self.led.oracle_may("explore"), "花了预留也不能让总数变多")

    def test_the_reserve_is_measured_against_the_total_not_the_phase(self):
        """预算是「总数 - 预留」而不是「explore 那一格」: 分不出 explore/refine 时如实只花一格"""
        self.assertEqual({"calibration": 1, "explore": 5, "refine": 3, "confirm": 1},
                         self.led.caps)
        for _ in range(5):                     # 已超过 explore 那一格 (5)
            self.led.oracle_spend("explore")
        self.assertTrue(self.led.oracle_may("explore"),
                        "非 confirm 的开销今天只被总数与预留两条拦 (相位是计划, 不是闸门)")

    def test_unknown_phase_raises(self):
        for bad in ("Explore", "confirm ", "", None, "wall_clock"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.led.oracle_may(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.led.oracle_spend(bad)
        self.assertEqual(tuple(PHASES), PHASES)


class TestTheRulerIsNotStarved(unittest.TestCase):
    """标定只记账、不被拦 (让预算饿死标定 = 把「尺子有没有坏」变成缺钱就能跳过的事)"""

    def test_calibration_is_recorded_and_overrun_is_visible(self):
        led = BudgetLedger(BudgetPlan(oracle_total=100))   # 标定份额 5
        for _ in range(7):
            led.note_calibration_run()
        self.assertEqual(7, led.spent["calibration"])
        self.assertEqual(2, led.calibration_over_plan, "超支必须留痕")
        # 超支也不影响它继续跑: 这里没有门可关, 只有账。
        led.note_calibration_run()
        self.assertEqual(8, led.spent["calibration"])

    def test_without_a_plan_nothing_is_counted_as_overrun(self):
        led = BudgetLedger()
        led.note_calibration_run()
        self.assertEqual(1, led.spent["calibration"])
        self.assertEqual(0, led.calibration_over_plan)


class TestWallClockAndTokens(unittest.TestCase):

    def test_wall_clock_is_strict_and_records_the_last_check(self):
        led = BudgetLedger(BudgetPlan(wall_clock_s=10))
        self.assertTrue(led.within_wall_clock(9.999))
        self.assertEqual(9.999, led.wall_clock_checked_s)
        self.assertFalse(led.within_wall_clock(10), "到点即停 (不许把 == 读成还没到)")

    def test_wall_clock_is_off_by_default(self):
        led = BudgetLedger()
        self.assertTrue(led.within_wall_clock(10 ** 9))

    def test_tokens_left_and_exhaustion(self):
        led = BudgetLedger(BudgetPlan(llm_tokens=100))
        self.assertEqual(40, led.tokens_left(60))
        self.assertFalse(led.tokens_exhausted(99))
        self.assertTrue(led.tokens_exhausted(100), "刚好花完就是花完")
        self.assertTrue(led.tokens_exhausted(101))
        self.assertIsNone(BudgetLedger().tokens_left(10 ** 9))

    def test_degraded_rounds_are_counted_only_when_over_budget(self):
        led = BudgetLedger(BudgetPlan(llm_tokens=100))
        self.assertFalse(led.note_llm_usage(10))
        self.assertEqual(0, led.llm_degraded_rounds)
        self.assertTrue(led.note_llm_usage(150))
        self.assertEqual(1, led.llm_degraded_rounds)
        self.assertEqual(150, led.llm_tokens_used)


class TestAuditSnapshot(unittest.TestCase):

    def test_the_snapshot_answers_plan_spent_and_who_was_blocked(self):
        led = BudgetLedger(BudgetPlan(oracle_total=10, wall_clock_s=60, llm_tokens=1000))
        led.oracle_spend("explore", 2)
        led.oracle_refuse("explore")
        led.note_calibration_run()
        led.note_llm_usage(1500)
        d = led.as_dict()
        self.assertEqual(10, d["plan"]["oracle_total"])
        self.assertEqual({"calibration": 1, "explore": 5, "refine": 3, "confirm": 1},
                         d["plan"]["phase_caps"])
        self.assertEqual({"calibration": 1, "explore": 2, "refine": 0, "confirm": 0},
                         d["spent"])
        self.assertEqual(3, d["spent_all"])
        self.assertEqual({"explore": 1}, d["refusals"])
        self.assertEqual(1, d["llm_degraded_rounds"])
        self.assertEqual(1500, d["llm_tokens_used"])


if __name__ == "__main__":
    unittest.main(verbosity=2)