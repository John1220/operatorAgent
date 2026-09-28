"""成功率量具的聚合逻辑离线标定

这一组不碰 GPU、不碰网络, 只用假记录验 `aggregate()` / `format_report()`。
理由是本项目反复踩的那一条 (doc 46): **判据方向写反时, 量具明明看见了信号却报「看不见」**,
而一份坏掉的报表会打印出很像样的百分数。真机跑一轮很贵, 所以聚合逻辑必须能离线验。

每条用例都尽量做成**方向检查**: 断言「两种相反的输入给出两种相反的输出」,
而不是只断言某一个具体数字 —— 后者在判据整体失效时也可能碰巧成立。
"""
import unittest

from benchmarks.harness_pass_rate import MIN_DENOM, aggregate, format_report

VALID, DEGRADED, UNINFO = "valid", "degraded", "uninformative"


def _round(n, l1=True, l2=None, l3=None, validity=None, speedup=None, action="optimize"):
    """造一轮血缘记录。l2/l3 为 None 表示「没走到这一级」(键不存在, 不是 False)"""
    lv = {"L1": l1}
    if l2 is not None:
        lv["L2"] = l2
    if l3 is not None:
        lv["L3"] = l3
    return {"round": n, "action": action, "all_passed": all(lv.values()),
            "levels_passed": lv, "measurement_validity": validity,
            "gflops": None, "speedup": speedup}


def _run(rounds, final=None, admitted=False, reason="", tokens=100, usd=0.01,
         wall=10.0, ok=True, error=None):
    if final is None:
        final = bool(rounds) and any(r["all_passed"] for r in rounds)
    rtp = next((r["round"] for r in rounds if r["all_passed"]), None) if final else None
    return {"case": "c", "ok": ok, "error": error, "rounds": rounds,
            "final_all_passed": final, "rounds_to_pass": rtp,
            "admitted": admitted, "admit_reason": reason,
            "tokens": tokens, "cost_usd": usd, "wall_s": wall}


def _n_passing(n):
    """n 次全部一轮通过、读数全 valid、加速比 1.5x、且都入库"""
    return [_run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.5)],
                 admitted=True) for _ in range(n)]


class TestDirectionIsRight(unittest.TestCase):
    """最要命的一类错: 报表对「全过」和「全挂」给出同一句话"""

    def test_all_pass_and_all_fail_report_opposite_rates(self):
        good = aggregate(_n_passing(MIN_DENOM))
        bad = aggregate([_run([_round(1, l1=False)], final=False) for _ in range(MIN_DENOM)])
        self.assertEqual(100.0, good["task_success_pct"])
        self.assertEqual(0.0, bad["task_success_pct"])
        self.assertNotEqual(good["task_success_pct"], bad["task_success_pct"],
                            "两种相反的输入给出同一个成功率 —— 量具是瞎的")
        self.assertEqual(100.0, good["level_pcts"]["L1"])
        self.assertEqual(0.0, bad["level_pcts"]["L1"])

    def test_report_text_differs_between_pass_and_fail(self):
        good = format_report(aggregate(_n_passing(MIN_DENOM)))
        bad = format_report(aggregate([_run([_round(1, l1=False)], final=False)
                                       for _ in range(MIN_DENOM)]))
        self.assertIn("100.0%", good)
        self.assertIn("0.0%", bad)
        self.assertNotEqual(good, bad)


class TestLevelRateDenominator(unittest.TestCase):
    """分母必须是「走到该级的轮数」, 否则 L1 的失败率会污染 L2/L3"""

    def test_rounds_that_died_at_l1_do_not_count_against_l2(self):
        rounds = [_round(1, l1=False),                      # 死在 L1
                  _round(1, l2=True, l3=True, validity=VALID),
                  _round(1, l2=False),                      # 死在 L2
                  _round(1, l2=True, l3=True, validity=VALID)]
        agg = aggregate([_run([r]) for r in rounds])
        self.assertEqual((3, 4), agg["level_rates"]["L1"], "4 轮都走到 L1, 其中 3 轮过")
        self.assertEqual((2, 3), agg["level_rates"]["L2"],
                         "L2 的分母该是「走到 L2 的 3 轮」, 不是 4 轮")
        self.assertEqual((2, 2), agg["level_rates"]["L3"],
                         "L3 的分母该是「走到 L3 的 2 轮」")

    def test_missing_level_key_is_not_read_as_failure(self):
        """键不存在 ≠ 值为 False。混在一起会把「没走到」算成「没通过」"""
        runs = [_run([_round(1, l1=False)]) for _ in range(MIN_DENOM)]
        agg = aggregate(runs)
        self.assertEqual(0, agg["level_rates"]["L2"][1], "L2 分母应为 0 (一轮都没走到)")
        self.assertIsNone(agg["level_pcts"]["L2"], "分母为 0 时不该编一个百分数出来")


class TestSmallDenominatorRefusesPercentage(unittest.TestCase):
    def test_below_min_denom_does_not_print_a_rate(self):
        agg = aggregate(_n_passing(MIN_DENOM - 1))
        self.assertFalse(agg["denominator_ok"])
        text = format_report(agg)
        self.assertIn("分母不足", text)
        self.assertNotIn("任务级成功】0/", text)
        self.assertNotRegex(text, r"任务级成功】\d+/\d+ = \d")

    def test_at_min_denom_does_print(self):
        agg = aggregate(_n_passing(MIN_DENOM))
        self.assertTrue(agg["denominator_ok"])
        self.assertIn("任务级成功", format_report(agg))


class TestDecisionGradeIsSeparateFromPassRate(unittest.TestCase):
    """通过率与「读数可不可信」必须分开报 —— 这是这份量具存在的理由之一"""

    def test_tiers_are_counted_and_valid_only_is_decision_grade(self):
        runs = [
            _run([_round(1, l2=True, l3=True, validity=VALID, speedup=2.0)]),
            _run([_round(1, l2=True, l3=True, validity=DEGRADED, speedup=1.2)]),
            _run([_round(1, l2=True, l3=True, validity=UNINFO, speedup=0.4)]),
            _run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.8)]),
        ]
        agg = aggregate(runs)
        self.assertEqual({VALID: 2, DEGRADED: 1, UNINFO: 1}, agg["validity_tiers"])
        self.assertEqual((2, 4), agg["decision_grade"])
        self.assertEqual(50.0, agg["decision_grade_pct"])
        # 全部 4 轮都「通过」了, 但只有一半的读数有资格驱动闭环
        self.assertEqual(100.0, agg["task_success_pct"])
        self.assertIn("一起读", format_report(agg),
                      "报表必须提醒这两个数不能分开引用")

    def test_speedup_is_split_by_grade_not_merged(self):
        runs = [
            _run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.1)]),
            _run([_round(1, l2=True, l3=True, validity=UNINFO, speedup=9.9)]),
            _run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.3)]),
            _run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.2)]),
        ]
        agg = aggregate(runs)
        self.assertEqual(4, agg["speedup_all"]["n"])
        self.assertEqual(3, agg["speedup_decision_grade"]["n"],
                         "uninformative 那轮的 9.9x 不该混进成绩口径")
        self.assertLess(agg["speedup_decision_grade"]["median"], 2.0)
        self.assertGreater(agg["speedup_all"]["max"], 9.0)


class TestCrashesAndVacuousTruth(unittest.TestCase):
    def test_crashed_runs_are_counted_but_not_in_the_success_denominator(self):
        runs = _n_passing(2) + [_run([], ok=False, error="Timeout: x")] * 2
        agg = aggregate(runs)
        self.assertEqual(4, agg["runs"])
        self.assertEqual(2, agg["runs_ok"])
        self.assertEqual(2, agg["runs_crashed"])
        self.assertEqual((2, 2), agg["task_success"],
                         "崩溃样本不该混进成功率的分母, 但要单独计数")
        self.assertIn("Timeout: x", agg["crash_reasons"])

    def test_a_run_with_no_rounds_is_not_a_success(self):
        """`all([])` 是 True: 一份没有验证结果的产物会被 all_passed 判成「全过」。
        量具自己也得守住这一条, 不能只依赖 Conductor 的早退路径填了合成失败。"""
        agg = aggregate([_run([], final=False) for _ in range(MIN_DENOM)])
        self.assertEqual(0.0, agg["task_success_pct"])
        self.assertEqual(0, agg["rounds"])


class TestAdmissionAndCost(unittest.TestCase):
    def test_admission_rate_is_over_passing_runs_only(self):
        runs = [_run([_round(1, l2=True, l3=True, validity=VALID)], admitted=True),
                _run([_round(1, l2=True, l3=True, validity=UNINFO)], admitted=False,
                     reason="测量不可信 (can_gate=False)"),
                _run([_round(1, l2=True, l3=True, validity=VALID)], admitted=True),
                _run([_round(1, l2=True, l3=True, validity=VALID)], admitted=True)]
        agg = aggregate(runs)
        self.assertEqual((3, 4), agg["admitted"])
        self.assertEqual(75.0, agg["admitted_pct"])
        self.assertIn("测量不可信", agg["admit_rejections"][0])

    def test_cost_is_summed(self):
        agg = aggregate(_n_passing(MIN_DENOM))
        self.assertEqual(100 * MIN_DENOM, agg["cost"]["tokens"])
        self.assertEqual(round(0.01 * MIN_DENOM, 4), agg["cost"]["usd"])


if __name__ == "__main__":
    unittest.main()
