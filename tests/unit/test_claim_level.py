"""§4.4 声明等级 (ClaimLevel) + §22.3 覆盖上限 —— 离线标定

钉住的不是措辞, 而是四个方向:

1. **阶梯只能逐级上升**: 每一级都要有对应证据, 缺证据不许升（`None` 不补齐）。
2. **覆盖不足只压级**: §22.3 的 `mode` 决定等级上限；`can_*` 那几个位**一个字都不改**
   （「只能影响 ClaimLevel 与 promotion, 不能替代执行沙箱」）。
3. **上限对入库门是判据**: `admit_to_corpus` 读的就是这一格；三态 —— 缺键（老产物）
   不追溯重判, 有键而读不懂则**拒**（fail-closed）。
4. **同一个数**: 入库门拿来判的 `mode` 与 §21.4 发布在血缘里的 `AuditCoverage.mode`
   必须是同一个函数（`source_lang.classify`）在同一个来源上的结论。

纯本地, 不 import measurement / 不需要 GPU / LLM。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent.generate import agent_runtime as ar  # noqa: E402
from operator_agent.generate.agent_runtime import precompile_checks  # noqa: E402
from operator_agent.core.confirm import ConfirmRecord  # noqa: E402
from operator_agent.decide.eligibility import (  # noqa: E402
    ALL_FALSE, ALL_TRUE, CLAIM_LEVELS, CLAIM_OBSERVED, CLAIM_PERSISTED,
    CLAIM_PROMOTABLE, CLAIM_PROVISIONAL, CLAIM_UNOBSERVED, CLAIM_VALIDATED,
    DEGRADED, Eligibility, at_least, claim_ceiling, claim_level, claim_rank,
)
from operator_agent.decide.oracle import admit_to_corpus  # noqa: E402
from operator_agent.core.source_lang import COVERAGE_AST, COVERAGE_NONE, COVERAGE_REGEX  # noqa: E402
from operator_agent.core.source_lang import classify  # noqa: E402
from tests.unit.test_agent_runtime import _action, _ctx  # noqa: E402


def _metrics(**extra):
    m = {"speedup_vs_baseline": 1.3, "measurement_validity": "valid",
         "measurement_can_gate": True, "backend_real_device": True}
    m.update(extra)
    return m


def _claim(**kw):
    return claim_level(eligibility=ALL_TRUE, validity="valid",
                       correctness_verified=True, coverage_mode=COVERAGE_AST, **kw)


def _patch_with(source):
    return ar.Patch(path="candidate/code.py", new_source=source,
                    parent_source="def kernel(A, B):\n    return A + B\n")


class TestTheLadderIsClosedAndMonotone(unittest.TestCase):

    def test_the_six_levels_are_the_designs_six(self):
        self.assertEqual(
            ["unobserved", "observed", "provisional", "validated",
             "promotable", "persisted"], list(CLAIM_LEVELS))

    def test_rank_is_the_position(self):
        self.assertEqual([0, 1, 2, 3, 4, 5], [claim_rank(x) for x in CLAIM_LEVELS])

    def test_an_unknown_level_is_not_higher_than_anything(self):
        """读不懂的等级 ⇒ -1（fail-closed）。写反了会变成「未知 = 最高级」"""
        self.assertEqual(-1, claim_rank("validated "))       # 尾空格也算读不懂
        self.assertEqual(-1, claim_rank(None))
        self.assertFalse(at_least("nope", CLAIM_PROMOTABLE))
        self.assertFalse(at_least(CLAIM_PERSISTED, "nope"))
        self.assertTrue(at_least(CLAIM_PERSISTED, CLAIM_PROMOTABLE))
        self.assertFalse(at_least(CLAIM_VALIDATED, CLAIM_PROMOTABLE))


class TestEachRungNeedsItsOwnEvidence(unittest.TestCase):

    def test_no_measurement_is_unobserved(self):
        self.assertEqual(CLAIM_UNOBSERVED, claim_level().level)

    def test_all_false_eligibility_is_unobserved(self):
        """整组拒绝 = 这个数没有指称对象 ⇒ 不是「有观测但资格不足」"""
        rec = claim_level(eligibility=ALL_FALSE, validity="uninformative",
                          coverage_mode=COVERAGE_AST)
        self.assertEqual(CLAIM_UNOBSERVED, rec.level)

    def test_an_observation_without_eligibility_is_only_observed(self):
        rec = claim_level(validity="valid", coverage_mode=COVERAGE_AST)
        self.assertEqual(CLAIM_OBSERVED, rec.level)

    def test_can_tune_without_valid_or_correctness_is_provisional(self):
        for kw in ({"validity": "degraded", "correctness_verified": True},
                   {"validity": "valid", "correctness_verified": None}):
            rec = claim_level(eligibility=ALL_TRUE, coverage_mode=COVERAGE_AST, **kw)
            self.assertEqual(CLAIM_PROVISIONAL, rec.level, kw)

    def test_degraded_eligibility_with_a_valid_tier_is_provisional(self):
        """DEGRADED 的位型里 `can_tune=True`, 但档位不是 valid ⇒ 停在 provisional"""
        rec = claim_level(eligibility=DEGRADED, validity="degraded",
                          correctness_verified=True, coverage_mode=COVERAGE_AST)
        self.assertEqual(CLAIM_PROVISIONAL, rec.level)

    def test_a_can_verdict_but_not_can_tune_mask_is_observed(self):
        """阶梯里 `can_tune=False` 那一档今天没有生产掩码 ⇒ 用合成掩码证明它不是死枝

        （`derive_eligibility` 今天只产三种位型, 三种的 `can_verdict` 与 `can_tune`
        都同进同出 —— 但阶梯是**定义**, 不是这三种位型的枚举。）
        """
        mask = Eligibility(True, False, False, True, False)
        self.assertEqual(CLAIM_OBSERVED,
                         claim_level(eligibility=mask, validity="valid",
                                     correctness_verified=True).level)

    def test_valid_tier_and_correctness_reach_validated(self):
        self.assertEqual(CLAIM_VALIDATED, _claim().level)

    def test_independent_confirmation_reaches_promotable(self):
        self.assertEqual(CLAIM_PROMOTABLE, _claim(independent_confirmation=True).level)

    def test_persisted_wins_over_everything_below(self):
        rec = _claim(persisted=True)
        self.assertEqual(CLAIM_PERSISTED, rec.level)
        self.assertEqual([], [r for r in rec.reasons if "未达" in r])

    def test_the_rungs_are_not_skipped_upward(self):
        """没有正确性凭据时, 即使有独立确认也不许跳到 promotable（逐级上升）"""
        rec = claim_level(eligibility=ALL_TRUE, validity="degraded",
                          correctness_verified=None, coverage_mode=COVERAGE_AST,
                          independent_confirmation=False)
        self.assertEqual(CLAIM_PROVISIONAL, rec.level)


class TestTheCoverageCeiling(unittest.TestCase):

    def test_ast_does_not_cap(self):
        self.assertEqual(CLAIM_PERSISTED, claim_ceiling(COVERAGE_AST))

    def test_regex_only_caps_at_validated(self):
        self.assertEqual(CLAIM_VALIDATED, claim_ceiling(COVERAGE_REGEX))

    def test_none_and_not_judged_cap_at_provisional(self):
        """三态: 「判不了」与「没判过」同档 —— 缺证据不是有效证据"""
        for mode in (COVERAGE_NONE, None, "", "   "):
            self.assertEqual(CLAIM_PROVISIONAL, claim_ceiling(mode), repr(mode))

    def test_an_unknown_mode_is_capped_conservatively(self):
        """设计词表里的 typed_ast 今天没有生产者（source_lang 只产三种）——
        将来加它的人要同时改表, 否则会被保守默认压到 provisional（刻意 fail-closed）。"""
        self.assertEqual(CLAIM_PROVISIONAL, claim_ceiling("typed_ast"))

    def test_a_capped_source_cannot_reach_promotable(self):
        rec = claim_level(eligibility=ALL_TRUE, validity="valid",
                          correctness_verified=True, coverage_mode=COVERAGE_REGEX)
        self.assertEqual(CLAIM_VALIDATED, rec.level)
        self.assertEqual(CLAIM_VALIDATED, rec.ceiling)
        self.assertTrue(any("§22.3" in r for r in rec.reasons),
                        "上限压住了等级, 理由里却没有它")

    def test_the_cap_never_touches_the_eligibility_bits(self):
        """「只能影响 ClaimLevel 与 promotion, 不能替代执行沙箱」"""
        rec = claim_level(eligibility=ALL_TRUE, validity="valid",
                          correctness_verified=True, coverage_mode=COVERAGE_NONE)
        self.assertEqual(CLAIM_PROVISIONAL, rec.level)
        self.assertTrue(ALL_TRUE.can_persist, "等级推导动到了资格位")
        self.assertEqual(ALL_TRUE.as_dict(), ALL_TRUE.as_dict())

    def test_a_cap_reason_is_printed_even_when_it_did_not_bind(self):
        """上限没压住当前这一级时也要说出来（否则「为什么进不了语料」只剩门里那一句）"""
        rec = claim_level(eligibility=ALL_TRUE, validity="degraded",
                          correctness_verified=True, coverage_mode=COVERAGE_REGEX)
        self.assertEqual(CLAIM_PROVISIONAL, rec.level)      # 本来就是 provisional
        self.assertTrue(any("regex_only" in r for r in rec.reasons))


class TestTheMaskIsReadWithoutFilling(unittest.TestCase):

    def test_a_full_mask_round_trips(self):
        self.assertEqual(ALL_TRUE, Eligibility.from_mapping(ALL_TRUE.as_dict()))
        self.assertEqual(DEGRADED, Eligibility.from_mapping(DEGRADED.as_dict()))

    def test_a_short_mask_is_not_filled_with_defaults(self):
        for bad in ({"can_verdict": True}, {}, {"can_tune": False}):
            self.assertIsNone(Eligibility.from_mapping(bad), bad)

    def test_a_non_mapping_is_none(self):
        for bad in (None, "can_persist", 3):
            self.assertIsNone(Eligibility.from_mapping(bad), bad)

    def test_a_false_bit_survives_the_round_trip(self):
        m = DEGRADED.as_dict()
        got = Eligibility.from_mapping(m)
        self.assertFalse(got.can_persist)
        self.assertFalse(got.can_count_progress)

    def test_claim_level_accepts_a_mapping_without_filling_it(self):
        """字典缺一位 ⇒ 按「没有资格位」⇒ 最多 observed（而**不是**补齐成 VALID）"""
        rec = claim_level(eligibility={"can_verdict": True},
                          validity="valid", correctness_verified=True,
                          coverage_mode=COVERAGE_AST)
        self.assertEqual(CLAIM_OBSERVED, rec.level)


class TestTheCoverageIsTheSameNumberAsTheRuntimePublishes(unittest.TestCase):
    """入库门算的 mode 与 §21.4 发布在血缘里的 `AuditCoverage.mode` 必须同一个来源

    入库那条线走 `source_lang.classify(artifact.code, task.target.dsl)`; 血缘那条线走
    `precompile_checks(...).coverage.mode`, 而它内部也是 `classify(patch.new_source,
    ctx.language)`。两条线读同一份源码时必须给同一个读数 —— 否则「上游留痕的覆盖」与
    「下游拿来判的覆盖」是两句话。
    """

    CASES = (("python", "def kernel(A, B):\n    return A @ B\n"),
             ("triton", "def kernel(A, B):\n    return A @ B\n"),
             ("cuda_c", "// C++ 源码\n__global__ void k() {}\n"),
             ("verilog", "module m; endmodule\n"))

    def test_the_two_readings_agree(self):
        for declared, source in self.CASES:
            with self.subTest(declared=declared):
                published = precompile_checks(
                    _action(patch=_patch_with(source)), _ctx(language=declared)
                ).coverage.mode
                self.assertEqual(classify(source, declared).coverage, published,
                                 "上游留痕的覆盖与下游拿来判的覆盖不是同一个读数")


class TestTheAdmissionGateReadsTheCeiling(unittest.TestCase):
    """§22.3 的「与 promotion」那一半: 上限是**判据**, 不只是留痕"""

    def test_a_legacy_reading_without_a_claim_level_is_not_re_judged(self):
        ok, _ = admit_to_corpus(_metrics(), None, requires_remeasure=False)
        self.assertTrue(ok, "老产物被新门追溯性拒了")

    def test_a_persisted_ceiling_passes(self):
        ok, _ = admit_to_corpus(
            _metrics(claim_level={"level": CLAIM_VALIDATED,
                                  "ceiling": CLAIM_PERSISTED,
                                  "coverage_mode": COVERAGE_AST}),
            None, requires_remeasure=False)
        self.assertTrue(ok)

    def test_a_validated_ceiling_refuses(self):
        ok, msg = admit_to_corpus(
            _metrics(claim_level={"level": CLAIM_VALIDATED,
                                  "ceiling": CLAIM_VALIDATED,
                                  "coverage_mode": COVERAGE_REGEX}),
            None, requires_remeasure=False)
        self.assertFalse(ok, "覆盖不足的源码进了长期记忆")
        self.assertIn("§22.3", msg)
        self.assertIn("regex_only", msg)

    def test_the_gate_fires_before_the_remeasure_gate(self):
        """覆盖是**静态**事实 ⇒ 排在花机时的两次确认之前（与 §21.4 零 GPU 时间同一条纪律）"""
        ok, msg = admit_to_corpus(
            _metrics(claim_level={"level": CLAIM_PROVISIONAL,
                                  "ceiling": CLAIM_PROVISIONAL,
                                  "coverage_mode": COVERAGE_NONE}),
            None)   # 没有重测 ⇒ 两道门都会拒; 该看到的是覆盖那一句
        self.assertFalse(ok)
        self.assertIn("§22.3", msg)
        self.assertNotIn("两次确认", msg)

    def test_a_capped_source_is_refused_even_with_a_good_remeasure(self):
        """两次确认过了也不许把「覆盖不足」买过去"""
        ok, msg = admit_to_corpus(
            _metrics(claim_level={"level": CLAIM_PROVISIONAL,
                                  "ceiling": CLAIM_PROVISIONAL,
                                  "coverage_mode": COVERAGE_NONE}),
            ConfirmRecord(ok=True, speedup=1.4, validity="valid", independent=True))
        self.assertFalse(ok)
        self.assertIn("§22.3", msg)

    def test_a_legacy_gate_order_still_holds(self):
        """门 1 拒了就不该走到覆盖那一步（顺序即短路顺序）"""
        ok, msg = admit_to_corpus(
            _metrics(measurement_can_gate=False,
                     claim_level={"ceiling": CLAIM_PROVISIONAL,
                                  "coverage_mode": COVERAGE_NONE}),
            None, requires_remeasure=False)
        self.assertFalse(ok)
        self.assertIn("can_gate=False", msg)
        self.assertNotIn("§22.3", msg)

    def test_a_bare_string_claim_level_is_refused(self):
        """有键而形状不对 ⇒ 拒（读不懂的等级不许当通过）"""
        ok, msg = admit_to_corpus(_metrics(claim_level=CLAIM_VALIDATED),
                                  None, requires_remeasure=False)
        self.assertFalse(ok)
        self.assertIn("形状", msg)

    def test_an_unknown_ceiling_is_refused(self):
        ok, msg = admit_to_corpus(
            _metrics(claim_level={"ceiling": "very-good", "coverage_mode": "?"}),
            None, requires_remeasure=False)
        self.assertFalse(ok)
        self.assertIn("very-good", msg)

    def test_a_claim_level_produced_by_the_ladder_is_acceptable(self):
        """端到端形状: `claim_level(...).as_dict()` 直接喂给门（生产者与消费者同形）"""
        good = claim_level(eligibility=ALL_TRUE, validity="valid",
                           correctness_verified=True,
                           coverage_mode=COVERAGE_AST).as_dict()
        ok, _ = admit_to_corpus(_metrics(claim_level=good), None,
                                requires_remeasure=False)
        self.assertTrue(ok)
        capped = claim_level(eligibility=ALL_TRUE, validity="valid",
                             correctness_verified=True,
                             coverage_mode=COVERAGE_REGEX).as_dict()
        ok, msg = admit_to_corpus(_metrics(claim_level=capped), None,
                                  requires_remeasure=False)
        self.assertFalse(ok)
        self.assertIn("§22.3", msg)


class TestTheReportAndTheUnableSectionReadIt(unittest.TestCase):
    """等级必须有**人**读: 一行进报告, 受限时进「我无法确立的事」"""

    def _artifact(self, metrics):
        from operator_agent.core.schemas import KernelArtifact
        return KernelArtifact(kernel_id="kw", task_id="t", code="# x",
                              language="triton", benchmark_data=metrics)

    def _capped(self):
        return claim_level(eligibility=ALL_TRUE, validity="valid",
                           correctness_verified=True,
                           coverage_mode=COVERAGE_REGEX).as_dict()

    def test_the_report_prints_the_level_and_its_cap(self):
        from operator_agent.core.report import build_report
        text = build_report(artifact=self._artifact(
            {"measurement_validity": "valid", "gflops": 1.0,
             "claim_level": self._capped()}))
        self.assertIn("声明等级 (§4.4)", text)
        self.assertIn("regex_only", text)

    def test_no_claim_key_means_no_line(self):
        """缺键不许印一个默认等级出来（那会让「没算」读成「算出来是它」）"""
        from operator_agent.core.report import build_report
        text = build_report(artifact=self._artifact({"measurement_validity": "valid"}))
        self.assertNotIn("声明等级", text)

    def test_a_capped_level_shows_up_in_the_unable_section(self):
        from operator_agent.core.report import build_report
        text = build_report(artifact=self._artifact(
            {"measurement_validity": "valid", "gflops": 1.0,
             "claim_level": self._capped()}))
        self.assertIn("声明等级受限", text)

    def test_an_uncapped_level_adds_no_unable_item(self):
        from operator_agent.core.report import build_report
        good = claim_level(eligibility=ALL_TRUE, validity="valid",
                           correctness_verified=True,
                           coverage_mode=COVERAGE_AST).as_dict()
        text = build_report(artifact=self._artifact(
            {"measurement_validity": "valid", "gflops": 1.0, "claim_level": good}))
        self.assertNotIn("声明等级受限", text)


class TestTheOnlyProducerFeedsTheOnlyGate(unittest.TestCase):
    """结构守卫: 门读的那一格必须由生产路径在**调门之前**写进去"""

    def test_the_conductor_writes_the_claim_level_before_calling_the_gate(self):
        src = (Path(__file__).resolve().parents[2] / "operator_agent"
               / "app/conductor.py").read_text(encoding="utf-8")
        written = src.index('metrics["claim_level"] =')
        called = src.index("admit_to_corpus(")
        self.assertLess(written, called,
                        "入库门读了 claim_level, 但生产路径没在调它之前写进去")
        self.assertEqual(1, src.count("admit_to_corpus("),
                         "入库判据的调用点不止一处 —— 判据只有一处是这条线的全部意义")

