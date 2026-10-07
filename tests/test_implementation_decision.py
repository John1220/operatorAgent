"""§17.1 实现裁决 (ImplementationDecision) —— 离线标定

设计原文（§17.1 末段）:

    'USE_CUSTOM' 要求 'ClaimLevel ≥ VALIDATED' 且所有 'required_cases' 正确;
    'USE_LIBRARY' 至少要求 'can_report'; 其余一律 'INCONCLUSIVE'。

这一格为什么要有自己的守卫: 它与「入库」是同一族的**永久后果** —— 设计把失效模式写在同一段
里（「因为一次噪声读数就建议你别上自定义核」与「因为一次噪声读数就永久拉黑一个配方」是同一
失效模式的两个方向）。所以本文件钉的不是措辞, 而是五件事:

1. **阶梯是闭的**: 旁举 4x7x3x3 组证据, 断言 'use_library' 当且仅当（'can_report' 且**确立**
   了回退越限）、'use_custom' 当且仅当三条证据全中 —— 没有第四条出路。
2. **三态不补齐**: 'required_cases_ok' / 'regression_over_limit' 的 'None'（未确立）挡住
   'use_custom', 但**不**变成 'use_library' 的依据。「没量过」不是「不好」。
3. **fail-closed**: 掩码不齐 / 等级形状读不懂 ⇒ 一档都够不上, 而且理由里说出来。
4. **有人读**: 裁决进 best-known-good（交付物第 1 件）; 缺键**不印**。
5. **推导只有一处**: 生产路径一处写入（Conductor._save_to_db）, 报告只读不算。

纯本地: 不 import measurement / 不需要 GPU / LLM。
"""

import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.eligibility import (  # noqa: E402
    ALL_FALSE, ALL_TRUE, CLAIM_LEVELS, CLAIM_OBSERVED, CLAIM_PERSISTED,
    CLAIM_PROMOTABLE, CLAIM_PROVISIONAL, CLAIM_UNOBSERVED, CLAIM_VALIDATED,
    DEGRADED, DECISION_INCONCLUSIVE, DECISION_USE_CUSTOM, DECISION_USE_LIBRARY,
    IMPLEMENTATION_DECISIONS, DecisionRecord, at_least, implementation_decision,
)

_VALIDATED = object()


def _claim(level=CLAIM_VALIDATED):
    """裁决的输入是**已经算好**的那一格（阶梯本身由 tests/test_claim_level.py 钉住）"""
    return {"level": level, "ceiling": CLAIM_PERSISTED, "coverage_mode": "ast",
            "reasons": []}


def _decide(eligibility=ALL_TRUE, claim=_VALIDATED, **evidence):
    return implementation_decision(
        eligibility=eligibility,
        claim_record=_claim() if claim is _VALIDATED else claim,
        **{"required_cases_ok": None, "regression_over_limit": None, **evidence})


class TestTheVocabularyIsTheDesignsThree(unittest.TestCase):

    def test_the_three_are_the_designs_three_and_in_order(self):
        self.assertEqual(["use_custom", "use_library", "inconclusive"],
                         list(IMPLEMENTATION_DECISIONS))

    def test_an_out_of_vocabulary_decision_refuses(self):
        """词表外的裁决不许被造出来（读不懂的结论既不是敢用也不是判不了）"""
        with self.assertRaises(ValueError):
            DecisionRecord(decision="maybe", claim_level="validated",
                           claim_ceiling="persisted", can_report=True,
                           required_cases_ok=True, regression_over_limit=False)


class TestTheLadderIsClosed(unittest.TestCase):
    """旁举全部组合: 只有两条路能走到 use_custom / use_library, 其余一律 inconclusive"""

    def test_the_decision_is_a_function_of_the_three_evidence(self):
        checked = {DECISION_USE_CUSTOM: 0, DECISION_USE_LIBRARY: 0,
                   DECISION_INCONCLUSIVE: 0}
        for elig, level, rco, rol in itertools.product(
                (None, ALL_FALSE, DEGRADED, ALL_TRUE), (None,) + CLAIM_LEVELS,
                (None, True, False), (None, True, False)):
            with self.subTest(elig=elig, level=level, rco=rco, rol=rol):
                d = _decide(eligibility=elig,
                            claim=None if level is None else _claim(level),
                            required_cases_ok=rco, regression_over_limit=rol)
                can_report = elig is not None and elig.can_report
                if not can_report:
                    expect = DECISION_INCONCLUSIVE
                elif rol is True:
                    expect = DECISION_USE_LIBRARY
                elif at_least(level, CLAIM_VALIDATED) and rco is True and rol is False:
                    expect = DECISION_USE_CUSTOM
                else:
                    expect = DECISION_INCONCLUSIVE
                self.assertEqual(expect, d.decision, d.reasons)
                checked[d.decision] += 1
        for decision, n in checked.items():
            self.assertGreater(n, 0, f"{decision} 在全部证据组合里一次都没被走到")


class TestGateZeroNeedsCanReport(unittest.TestCase):
    """门 0: 连「这个读数能对外报」都没确立 ⇒ 没有可裁决的对象"""

    def test_no_eligibility_is_inconclusive(self):
        d = _decide(eligibility=None, required_cases_ok=True,
                    regression_over_limit=False)
        self.assertEqual(DECISION_INCONCLUSIVE, d.decision)
        self.assertIn("没有资格位", " ".join(d.reasons))

    def test_a_closed_can_report_is_inconclusive(self):
        d = _decide(eligibility=ALL_FALSE)
        self.assertEqual(DECISION_INCONCLUSIVE, d.decision)
        self.assertIn("can_report=False", " ".join(d.reasons))

    def test_a_short_mask_is_not_filled_with_defaults(self):
        """掩码缺一位 ⇒ 按「没有资格位」处理, 不许补齐（三态纪律）"""
        d = _decide(eligibility={"can_verdict": True, "can_tune": True,
                                 "can_persist": True, "can_count_progress": True})
        self.assertEqual(DECISION_INCONCLUSIVE, d.decision)
        self.assertIn("没有资格位", " ".join(d.reasons))

    def test_a_degraded_mask_can_still_report(self):
        """DEGRADED 的 can_report 仍为 True ⇒ 门 0 放行（低资格不是这条的理由）"""
        d = _decide(eligibility=DEGRADED, required_cases_ok=True,
                    regression_over_limit=False)
        self.assertNotIn("门 0", " ".join(d.reasons))


class TestUseCustomNeedsAllThree(unittest.TestCase):
    """门 1: 声明等级 >= validated 且 required_cases 全对 且 没有回退越限"""

    def test_the_happy_path(self):
        d = _decide(required_cases_ok=True, regression_over_limit=False)
        self.assertEqual(DECISION_USE_CUSTOM, d.decision)
        self.assertIn("三条证据全中", " ".join(d.reasons))
        self.assertEqual(
            {"decision": "use_custom", "claim_level": "validated",
             "claim_ceiling": "persisted", "can_report": True,
             "required_cases_ok": True, "regression_over_limit": False},
            {k: v for k, v in d.as_dict().items() if k != "reasons"})

    def test_promotable_and_persisted_are_also_enough(self):
        for level in (CLAIM_PROMOTABLE, CLAIM_PERSISTED):
            d = _decide(claim=_claim(level), required_cases_ok=True,
                        regression_over_limit=False)
            self.assertEqual(DECISION_USE_CUSTOM, d.decision, level)

    def test_a_lower_level_blocks_it(self):
        for level in (CLAIM_UNOBSERVED, CLAIM_OBSERVED, CLAIM_PROVISIONAL):
            d = _decide(claim=_claim(level), required_cases_ok=True,
                        regression_over_limit=False)
            self.assertNotEqual(DECISION_USE_CUSTOM, d.decision, level)
            self.assertIn("声明等级", " ".join(d.reasons), level)

    def test_an_unestablished_required_cases_blocks_it(self):
        d = _decide(required_cases_ok=None, regression_over_limit=False)
        self.assertNotEqual(DECISION_USE_CUSTOM, d.decision)
        self.assertIn("未确立", " ".join(d.reasons))

    def test_a_wrong_required_case_blocks_it(self):
        d = _decide(required_cases_ok=False, regression_over_limit=False)
        self.assertNotEqual(DECISION_USE_CUSTOM, d.decision)
        self.assertIn("有必判 case 不正确", " ".join(d.reasons))

    def test_an_unestablished_regression_blocks_it(self):
        d = _decide(required_cases_ok=True, regression_over_limit=None)
        self.assertNotEqual(DECISION_USE_CUSTOM, d.decision)
        self.assertIn("max_regression", " ".join(d.reasons))

    def test_a_measured_regression_recommends_the_library(self):
        """门 2: §6.2 条款 3「回退越限 ⇒ 候选不得晋级」—— 设计点名的那条用库理由"""
        d = _decide(required_cases_ok=True, regression_over_limit=True)
        self.assertEqual(DECISION_USE_LIBRARY, d.decision)
        self.assertIn("不得晋级", " ".join(d.reasons))

    def test_a_regression_recommends_the_library_even_with_a_low_level(self):
        d = _decide(claim=_claim(CLAIM_PROVISIONAL), regression_over_limit=True)
        self.assertEqual(DECISION_USE_LIBRARY, d.decision)


class TestTheFallbackIsNotASilentLibraryRecommendation(unittest.TestCase):
    """门 3: 没有一条证据被确立 ⇒ inconclusive（**不**是「建议用库」）"""

    def test_missing_evidence_is_not_a_library_recommendation(self):
        d = _decide()
        self.assertEqual(DECISION_INCONCLUSIVE, d.decision)
        reasons = " ".join(d.reasons)
        self.assertIn("required_cases", reasons)
        self.assertIn("回退越限", reasons)
        self.assertIn("没有「建议用库」的依据", reasons)

    def test_every_blocker_gets_its_own_reason(self):
        d = _decide(claim=_claim(CLAIM_OBSERVED), required_cases_ok=False,
                    regression_over_limit=None)
        reasons = " ".join(d.reasons)
        for piece in ("声明等级 observed < validated", "有必判 case 不正确",
                      "未确立（今天没有 max_regression 的生产者）"):
            self.assertIn(piece, reasons, piece)

    def test_a_wrong_required_case_is_not_a_library_recommendation(self):
        """读法登记（CHECKLOG I91）: 设计给 USE_LIBRARY 的三条理由都预设自定义实现正确"""
        d = _decide(required_cases_ok=False, regression_over_limit=False)
        self.assertEqual(DECISION_INCONCLUSIVE, d.decision)


class TestTheShapesThatCannotBeReadAreFailClosed(unittest.TestCase):

    def test_a_missing_or_unreadable_record_is_not_custom(self):
        for bad in (None, {}, "validated", 3, []):
            d = _decide(claim=bad, required_cases_ok=True,
                        regression_over_limit=False)
            self.assertNotEqual(DECISION_USE_CUSTOM, d.decision, bad)
            self.assertIn("缺失", " ".join(d.reasons), bad)

    def test_a_missing_record_is_different_from_a_low_level(self):
        """「没有 §4.4 的记录」与「等级很低」都要落 inconclusive, 但理由必须分得开"""
        missing = " ".join(_decide(claim=None, required_cases_ok=True,
                                   regression_over_limit=False).reasons)
        low = " ".join(_decide(claim=_claim(CLAIM_OBSERVED),
                               required_cases_ok=True,
                               regression_over_limit=False).reasons)
        self.assertIn("缺失", missing)
        self.assertNotIn("缺失", low)
        self.assertIn("observed < validated", low)

    def test_the_record_carries_the_evidence_shape(self):
        """产物里三态原样记（None 不许被写成 False —— 那会让「没量过」读成「量过、没有」）"""
        d = _decide().as_dict()
        self.assertIsNone(d["required_cases_ok"])
        self.assertIsNone(d["regression_over_limit"])
        self.assertIs(d["can_report"], True)
        self.assertIsInstance(d["reasons"], list)


class TestTheReportPrintsIt(unittest.TestCase):
    """裁决必须有人读: 进 best-known-good（交付物第 1 件）"""

    def _artifact(self, **metrics):
        from operator_agent.schemas import KernelArtifact
        m = {"measurement_validity": "valid", "gflops": 1.0}
        m.update(metrics)
        return KernelArtifact(kernel_id="kw", task_id="t", code="# x",
                              language="triton", benchmark_data=m)

    def test_the_decision_and_its_reasons_are_printed(self):
        from operator_agent.report import build_report
        text = build_report(artifact=self._artifact(
            implementation_decision=_decide().as_dict()))
        self.assertIn("实现裁决 (§17.1)", text)
        self.assertIn("inconclusive", text)
        self.assertIn("门 3", text)
        self.assertLess(text.index("## 1. best-known-good"), text.index("实现裁决"))
        self.assertLess(text.index("实现裁决"), text.index("## 2."))

    def test_a_use_custom_decision_is_printed_too(self):
        from operator_agent.report import build_report
        text = build_report(artifact=self._artifact(
            implementation_decision=_decide(required_cases_ok=True,
                                            regression_over_limit=False).as_dict()))
        self.assertIn("use_custom", text)
        self.assertIn("三条证据全中", text)

    def test_a_missing_key_means_no_line(self):
        """缺键不许印一个默认裁决（那会让「没算」读成「算出来是它」）"""
        from operator_agent.report import build_report
        self.assertNotIn("实现裁决", build_report(artifact=self._artifact()))

    def test_a_malformed_record_prints_nothing(self):
        from operator_agent.report import build_report
        text = build_report(artifact=self._artifact(
            implementation_decision={"reasons": ["x"]}))
        self.assertNotIn("实现裁决", text)


class TestTheOnlyProducer(unittest.TestCase):
    """结构: 推导一处、写入一处、报告只读; 且生产路径不许替两条证据编默认值"""

    def _src(self, name):
        return (Path(__file__).resolve().parents[1] / "operator_agent"
                / name).read_text(encoding="utf-8")

    def test_the_derivation_lives_in_exactly_one_module(self):
        self.assertEqual(1, self._src("eligibility.py").count(
            "def implementation_decision("))
        self.assertEqual(1, self._src("conductor.py").count(
            "implementation_decision("))
        self.assertEqual(0, self._src("report.py").count(
            "implementation_decision("), "报告只许读, 不许自己再判一次")

    def test_the_conductor_writes_it_after_the_final_claim_level(self):
        src = self._src("conductor.py")
        self.assertLess(src.index('metrics["claim_level"] = final'),
                        src.index('metrics["implementation_decision"] ='),
                        "裁决记的必须是**入库之后**那个等级（终态）")

    def test_the_production_call_site_does_not_invent_the_two_evidence(self):
        """今天两条证据没有生产者 —— 不许塞 False 冒充「量过、没有回退」(fail-open)"""
        src = self._src("conductor.py")
        self.assertNotIn("required_cases_ok=", src)
        self.assertNotIn("regression_over_limit=", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
