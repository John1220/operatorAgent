"""双记分口径：同一批运行同时报「含真机 / 摘掉 L3」两个数 —— #77

外部一手 (arXiv:2609.26777v1 行 99-101, 逐字):
    "45.9% under the verifier versus 69.4% with E2E tests excluded from scoring"

那一行对我方的用处不是那两个数, 是**形状**: 同一批运行、同一个分母, 只差最后那道端到端门,
于是「本地做不到」与「做到了但过不了性能门」第一次被分开报。我方原来只有
`task_success` (最终 `all_passed`, 含 L3), 那一个数把两件事折在一起 —— 而缺口 4/#61/#65 那一整条线
要的正是「真机那道门到底拦掉了多少本来能用的核」。

判据的形状跟着 #85 那条规则走: 「本地过」= 最后一轮的 **L1 与 L2 都到场且都通过**
(`measurement.correctness_verified_levels`)。两处必须读同一个集合, 所以这一组测试里有一条
**成对**断言: 对象形态 (Conductor 用的) 与 dict 形态 (报表用的) 对同一批输入必须给同样答案。

⚠️ 这一支不碰 GPU、不碰网络、不跑真任务 —— `aggregate` 是纯函数, 喂合成运行即可。
真机那侧的读数由 #77 落地后**下一次**真机报表自己给出 (本会话零机时)。

变异标定 (2026-09-30 实跑, **9 条全拦**; 每条改一处、跑完立刻逐字节复原并断言。
表还没进 `benchmarks/mut_specs/` —— 迁一次要重跑全套, 等它真被重跑时再迁):
    M1 「曾经有哪一轮过」替代「最后一轮」            → 1 红 (test_only_the_last_round_decides)
    M2 空 rounds 不挡, 直接 `rounds[-1]`             → 1 红 (test_an_empty_round_list_...)
    M3 本地口径的分母换成全部运行 (含崩溃)            → 2 红 (分母同源 + 报表那一栏)
    M4 名单取自全部运行而不是本地过的那批              → 1 红 (test_the_blockers_are_named_...)
    M5 口径矛盾不再单独喊 (静默折进百分数)            → 1 红 (test_conflicting_run_...)
    M6 差值取绝对值 (丢掉符号)                        → 1 红 (与 M5 **同一条**用例, 见下)
    M7 集合里混进 L3 (把性能门当正确性)                → 4 红
    M8 谓词 `all` 改成 `any`                          → 6 红
    M9 Conductor 不走单点谓词、自己另写一套            → 1 红 (两种形态成对断言那条)
    ⚠️ M5 与 M6 红名单逐条相同 (1 条) —— 归属复核过: 两行改的是「一次口径互相矛盾」的两种
       失效形态 (根本不喊 / 喊了但把负号抹平成看起来正常的差值), 读者本来就是同一条同时
       断言那两件事的用例。保留两行是因为缺陷各在一层, 不是为了凑数。
"""
from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.harness_pass_rate import (aggregate, format_report,  # noqa: E402
                                          last_round_correctness_passed)
from operator_agent.conductor import correctness_verified  # noqa: E402
from operator_agent.measurement import (CORRECTNESS_LEVELS,  # noqa: E402
                                        correctness_verified_levels)
from operator_agent.schemas import VerificationLevel, VerificationResult  # noqa: E402


def _run(case, levels_last, final_passed, ok=True, extra_rounds=None):
    """一份最小运行记录: 只放 `aggregate` 真会读的键"""
    rounds = list(extra_rounds or []) + [{"round": len(extra_rounds or []) + 1,
                                          "levels_passed": dict(levels_last),
                                          "all_passed": final_passed}]
    return {"case": case, "ok": ok, "rounds": rounds,
            "final_all_passed": final_passed, "rounds_to_pass": None,
            "passed_via": None, "admitted": False, "admit_reason": ""}


L1 = VerificationLevel.L1_COMPILE
L2 = VerificationLevel.L2_FUNCTION
L3 = VerificationLevel.L3_PERFORMANCE

PASS_ALL = {L1.value: True, L2.value: True, L3.value: True}
LOCAL_ONLY = {L1.value: True, L2.value: True, L3.value: False}
L2_FAILED = {L1.value: True, L2.value: False}
L1_ONLY = {L1.value: True}          # fail-fast: L2 那一格根本没出现


class TestTheTwoColumnsShareOneDenominator(unittest.TestCase):
    def setUp(self):
        # 4 次完成的运行不是随手挑的: `format_report` 在分母不足 (`MIN_DENOM`) 时**拒绝**印
        # 百分数, 所以「报表印不印」这件事本身也在这组的覆盖面里 (见下面那条守卫)。
        self.runs = [
            _run("a-passes", PASS_ALL, True),
            _run("b-l3-blocks", LOCAL_ONLY, False),
            _run("c-l2-fails", L2_FAILED, False),
            _run("d-l3-blocks", LOCAL_ONLY, False),
            {"case": "e-crashed", "ok": False, "error": "RuntimeError: ssh"},
        ]
        self.agg = aggregate(self.runs)

    def test_both_rates_are_reported_with_the_same_denominator(self):
        self.assertEqual((1, 4), self.agg["task_success"])
        self.assertEqual((3, 4), self.agg["task_success_local"])
        self.assertEqual(self.agg["task_success"][1], self.agg["task_success_local"][1],
                         "两套口径必须共用分母, 否则差值里混着「样本不同」而不是「门不同」")
        self.assertEqual(4, self.agg["runs_ok"], "崩溃那次不许进任何一侧的分母")

    def test_the_blockers_are_named_not_just_counted(self):
        self.assertEqual(["b-l3-blocks", "d-l3-blocks"], self.agg["l3_only_blockers"])
        self.assertEqual([], self.agg["dual_scoring_conflicts"])

    def test_the_report_prints_both_numbers_and_the_gap(self):
        text = format_report(self.agg)
        self.assertIn("【任务级成功】1/4", text)
        self.assertIn("【同一批, 摘掉真机 L3】3/4", text)
        self.assertIn("差 2 次", text)
        self.assertIn("b-l3-blocks", text)

    def test_the_new_column_is_subject_to_the_same_denominator_gate(self):
        # 分母不足时两栏都不许印 —— 新加的口径不能成为绕过那道门的后门
        text = format_report(aggregate(self.runs[:2] + [self.runs[4]]))
        self.assertIn("分母不足", text)
        self.assertNotIn("摘掉真机 L3】", text)


class TestWhichRoundCounts(unittest.TestCase):
    def test_only_the_last_round_decides(self):
        """「第 1 轮本地过、第 2 轮改坏了」不算本地通过 —— 两个口径必须落在同一份代码上"""
        run = _run("drifted", L1_ONLY, False,
                   extra_rounds=[{"round": 1, "levels_passed": dict(LOCAL_ONLY),
                                  "all_passed": False}])
        self.assertFalse(last_round_correctness_passed(run))
        agg = aggregate([run])
        self.assertEqual((0, 1), agg["task_success_local"])

    def test_a_missing_l2_row_is_not_a_pass(self):
        """L1 挂时 Verifier fail-fast, L2 那一格根本不出现 ⇒ 缺证据不是通过"""
        self.assertFalse(last_round_correctness_passed(_run("x", L1_ONLY, False)))
        self.assertFalse(last_round_correctness_passed(_run("y", {}, False)))

    def test_an_empty_round_list_is_not_a_pass(self):
        # `all([])` 是 True —— 那正是 `KernelArtifact.all_passed` 被点名过的形状
        self.assertFalse(last_round_correctness_passed(
            {"case": "z", "ok": True, "rounds": [], "final_all_passed": True}))

    def test_conflicting_run_is_surfaced_rather_than_folded_in(self):
        """`final_all_passed` 为真而最后一轮的 L1/L2 不支持 ⇒ 血缘与判决不同源

        这种运行不能安静地变成一个「差 -1」: 那说明两份读数不是同一份代码的, 报表必须先喊出来。
        """
        agg = aggregate([_run("impossible", L2_FAILED, True)]
                        + [_run(f"p{i}", PASS_ALL, True) for i in range(3)])
        self.assertEqual(["impossible"], agg["dual_scoring_conflicts"])
        text = format_report(agg)
        self.assertIn("❌ 两套口径互相矛盾", text)
        self.assertIn("差 -1 次", text,
                      "负差值就是矛盾的信号, 它必须可见而不是被 max(0, …) 抹平")


class TestThePredicateIsASinglePoint(unittest.TestCase):
    """#85 的起点退回与 #77 的这一栏读的是同一个集合 —— 两处各写一遍就是两套口径"""

    def test_object_form_and_dict_form_agree_on_every_shape(self):
        outcomes = (True, False, None)
        for l1, l2, l3 in itertools.product(outcomes, outcomes, outcomes):
            levels = {}
            for lvl, val in ((L1, l1), (L2, l2), (L3, l3)):
                if val is not None:
                    levels[lvl.value] = val
            results = [VerificationResult(level=lvl, passed=val, message="",
                                          error_log=None, metrics={})
                       for lvl, val in ((L1, l1), (L2, l2), (L3, l3)) if val is not None]
            self.assertEqual(correctness_verified_levels(levels),
                             correctness_verified(results),
                             f"两种形态对同一批读数给了不同答案: {levels}")

    def test_the_set_is_still_L1_and_L2_derived_from_the_enum(self):
        """marker 表防腐: 枚举成员改名的那天, 这一条会红而不是让谓词悄悄变空集"""
        self.assertEqual({L1.value, L2.value}, set(CORRECTNESS_LEVELS))
        self.assertNotIn(L3.value, CORRECTNESS_LEVELS,
                         "L3 进这个集合 = 「正确但不够快」的核不再是被验证过的起点")


if __name__ == "__main__":
    unittest.main()
