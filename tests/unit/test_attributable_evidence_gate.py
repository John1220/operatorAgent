"""可归因证据门: 聚合数涨没涨 **与** 哪条 case 变坏了, 算在同一份存量证据上 (#103 第 1 片)

一手依据 (doc 60 §六, HarnessLens `2608.27311v1`)：

    行 98-99:  "...allowing **aggregate scores to obscure specific regressions**."
    行 878-880: "A verification stage advances only if it contains **attributable positive
               evidence and no attributable regression**. ... **preservation alone is insufficient**."
    行 1050:   "**Nothing is re-executed**: both rules are applied to the same stored evidence,
               so the comparison is exact for these runs."

**最该抄的是行 1050 那个做法**：两条规则读同一批已落盘的记录 ⇒ 分歧是精确的, 不需要机时、
不需要 LLM。我方今天两份判据都只看聚合数（`compare_runs` 印的是任务级成功率的 Fisher 与 Δ 的
符号），从不印「哪一条 case 变坏了」。

在 2026-09-30 的存量两份报表上量到的形状**比论文那句还糟一档**（出处：
`output/harness/20260928_174223` 与 `.../20260929_142830`，`git_head` 分别是 `0a4f0fd` 与
`4e80b13`，各 6 次真机运行、同一份 case 清单）：聚合那层印的是
**「⚠️ 判不出 (样本量不够) —— 不是『没变化』」**，而逐 case 那一层其实是
**1 条修好 + 3 条变坏**。也就是说我方那道统计门不是「被聚合数掩盖了回归」，而是**整句都不说**：
它把这份证据读成「不知道」。缺的东西不是更严的阈，是**点名**这件事。

三条判据形状，每条对着一种本仓已经犯过的错：
  · **三态**：一条都没配上时 `accepted_*` 与 `disagreement` 都是 `None`。写成「零回归 ⇒ 接受」
    就是 I25 那一族（输入是空气时判据永真）。
  · **缺席不当反证**：崩溃的那次 / 这一侧没有这条 case / 同名出现两次，都记成「没有可用读数」，
    既不算变坏也不算通过（#53 那一族，两侧都不许折）。
  · **只观测不判决**：这一栏不参与任何一次决定，聚合门（Fisher）仍是唯一的判决书。
    所以有一条用例专门钉住「加了这一栏之后 `verdict` 与判决文案一字未变」。

⚠️ 一处**与原文的口径偏差**（写在 `measurement.attributable_evidence` 里，不留白）：那句给了
两种正证据，其中 "a **stable success** with an increased pass count" 在「按 case 配对」的口径下
**蕴含在**第一种里（两侧都过的 case 不动任何一侧的计数 ⇒ `pass_b > pass_a` 必然意味着至少一条
从不过到过）。他们的通过数是在固定任务集上算的，「同一批里换了 case」也会让它涨 —— 那在我们
看来恰好不是正证据。下面 `test_increased_pass_count_is_implied_by_a_recovery` 钉的就是这个蕴含
关系；留着那个分支只会变成走不到的代码 + 一张杀不掉的变异行。

⚠️ 抄 I39 那条学费：`format_report` 有自己的分母门，但 `format_comparison` 没有 —— 这一栏的
用例断言的 needle 全部落在**本栏独有的标题**上（「【逐 case 的得与失】」），而不是那条与别栏共用
的「判决」前缀，否则整栏删掉也照样绿（I32 那一族）。

变异标定（2026-09-30 实跑，基线 **20 passed**，表在 `benchmarks/mut_specs/attributable_evidence_gate.py`，
**12/12 全部拦住、无逃逸**；括号里是实测红数）：

    A1 preservation 也算正证据                 → 1 红
    A2 可归因门不看回归数                       → 3 红   ← 整篇对着的那句病
    A3 缺席读数折成「这一侧没过」                → 2 红
    A4 metric-only 改成 Δ>=0 就算好             → 1 红
    A5 没有配对样本时不再早退（空输入折成布尔）    → 2 红
    P1 配对层把崩溃读成「没过」                  → 1 红   ← 与 A3 同错，长在搬运层
    P2 同名两次静默取最后一次                    → 1 红
    P3 「为什么没有结论」不填                    → 1 红
    R1 变坏的 case 不点名                        → 1 红   ← 本片的全部价值在这一行
    R2 没有配对样本时不印「测不到」               → 1 红
    R3 「打架」那句注解无条件印                  → 1 红   ← **第一次跑是逃逸的**，见下
    R4 compare_runs 不再带这一格                 → 5 红

R3 的逃逸与修法值得单独记：那条变异改的是**注解文案**（「← 聚合数与具体回归打架的那一次」），
而我最初只断言了数值串「分歧 有/无」——两份夹具的数值串都没变，所以两条用例都绿。补上
`assertNotIn("聚合数与具体回归打架", …)` 才杀掉。也就是说本仓那句「反向对照」**第一次用到解释性
文案上**，而不只是数字上：数字对、解释错，读者照样会被带走。

运行: PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest tests/unit/test_attributable_evidence_gate.py -q
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.drivers.harness_pass_rate import (compare_runs, format_comparison,  # noqa: E402
                                          pair_cases)
from operator_agent.measurement import (ATTRIBUTABLE_OUTCOMES,  # noqa: E402
                                        attributable_evidence)

A_ROOT = Path("output/harness/20260928_174223/pass_rate.json")
B_ROOT = Path("output/harness/20260929_142830/pass_rate.json")


def _ev(cases):
    return attributable_evidence(cases)


def _run(name, ok=True, passed=False):
    return {"case": name, "ok": ok, "final_all_passed": passed, "rounds": []}


class TestTheRule(unittest.TestCase):
    def test_nothing_matched_is_not_no_regression(self):
        """I25 那一族的正面: 空输入时三条结论都是 `None`, 不是「接受」也不是「零回归」"""
        out = _ev([])
        self.assertEqual(0, out["matched"])
        for k in ("accepted_metric_only", "accepted_gate", "disagreement"):
            self.assertIsNone(out[k], f"{k} 折成了布尔值: {out[k]}")
        self.assertIn("没有一条配上的 case", out["note"])

    def test_absent_readings_are_none_too(self):
        out = _ev([("x", None, None), ("y", True, None), ("z", None, False)])
        self.assertEqual(0, out["matched"])
        self.assertEqual(["x", "y", "z"], sorted(out["unmatched_cases"]))
        self.assertIsNone(out["accepted_gate"])

    def test_a_recovery_with_no_regression_is_accepted_by_both(self):
        out = _ev([("a", False, True), ("b", True, True)])
        self.assertEqual("recovery", out["positive_evidence"])
        self.assertIs(True, out["accepted_metric_only"])
        self.assertIs(True, out["accepted_gate"])
        self.assertIs(False, out["disagreement"])

    def test_aggregate_up_while_a_case_regresses_is_the_disagreement(self):
        """论文点名的那个形状（行 99）在本栏里的样子: 聚合涨了两条规则却不同意

        两条从不过到过、一条从过到不过 ⇒ 配对口径 Δ=+1: metric-only 接受, 可归因门不接受
        (它有回归)。这正是「聚合分数掩盖具体回归」的可执行判据。
        """
        out = _ev([("fixed-1", False, True), ("fixed-2", False, True),
                   ("broken", True, False)])
        self.assertEqual((1, 2), (out["pass_a"], out["pass_b"]))
        self.assertIs(True, out["accepted_metric_only"])
        self.assertIs(False, out["accepted_gate"])
        self.assertIs(True, out["disagreement"])
        self.assertEqual(["broken"], out["regressed_cases"])

    def test_preservation_alone_is_not_positive_evidence(self):
        """行 880 那句 "preservation alone is insufficient" 的直接对应"""
        out = _ev([("a", True, True), ("b", True, True), ("c", False, False)])
        self.assertEqual(2, out["counts"]["stable_pass"])
        self.assertIsNone(out["positive_evidence"], "两侧都过不该被读成正证据")
        self.assertIs(False, out["accepted_metric_only"], "Δ=0 时 metric-only 也不该接受")
        self.assertIs(False, out["accepted_gate"])

    def test_unchanged_failures_are_evidence_of_nothing(self):
        out = _ev([("a", False, False), ("b", False, False)])
        self.assertEqual(0, out["counts"]["stable_pass"] + out["counts"]["recovered"])
        self.assertIsNone(out["positive_evidence"])
        self.assertEqual([], out["regressed_cases"])

    def test_pass_counts_are_only_over_matched_cases(self):
        """换过 case 清单的两批: 某一侧没有的那条**不许**动通过数, 只进无结论

        否则「B 侧删掉一条一直失败的 case」会被读成通过数上涨 —— 那是换考卷不是变好。
        """
        out = _ev([("easy-new", None, True), ("hard-gone", True, None),
                   ("flipped", False, True)])
        self.assertEqual(1, out["matched"])
        self.assertEqual((0, 1), (out["pass_a"], out["pass_b"]))
        self.assertEqual(2, len(out["unmatched_cases"]))

    def test_every_matched_case_lands_in_exactly_one_class(self):
        cases = [("w", True, True), ("x", True, False), ("y", False, True), ("z", False, False)]
        out = _ev(cases)
        self.assertEqual(out["matched"], sum(out["counts"][k] for k in ATTRIBUTABLE_OUTCOMES))
        self.assertEqual(4, out["matched"])

    def test_increased_pass_count_is_implied_by_a_recovery(self):
        """那句原文的第二种正证据在配对口径下**不可达** —— 这条是把那句偏差钉住的证人

        枚举三个 case 的全部 2×2 组合: 凡是 `pass_b > pass_a` 的, `recovered_cases` 都非空。
        没有这条, 「口径其实与原文不同」这件事只活在我的注释里 (I28: 声明可核验就要真核一次)。
        """
        import itertools
        triples = list(itertools.product([True, False], repeat=2))
        for n in (1, 2, 3):
            for combo in itertools.product(triples, repeat=n):
                cases = [(f"c{i}", a, b) for i, (a, b) in enumerate(combo)]
                out = _ev(cases)
                if out["pass_b"] > out["pass_a"]:
                    self.assertTrue(out["recovered_cases"],
                                    f"Δ>0 却没有一条修好: {combo}")
                    self.assertEqual("recovery", out["positive_evidence"])

    def test_the_three_conclusions_are_three_state(self):
        """`accepted_*` / `disagreement` 只能是 `True` / `False` / `None`, 不许是字符串或数字"""
        for cases in ([], [("a", None, True)], [("a", True, True)], [("a", False, True),
                                                                     ("b", True, False)]):
            out = _ev(cases)
            for key in ("accepted_metric_only", "accepted_gate", "disagreement"):
                self.assertIn(out[key], (True, False, None), (key, out[key]))


class TestPairing(unittest.TestCase):
    def test_a_crashed_run_is_no_reading_not_a_failure(self):
        """`ok=False` 那次**没有**任务级结果 —— 折成 False 就是拿缺证据当反证 (#53 那一族)"""
        cases, reasons = pair_cases([_run("a", ok=False)], [_run("a", passed=True)])
        self.assertEqual([("a", None, True)], cases)
        self.assertIn("A 侧那次运行崩溃了", reasons["a"])

    def test_a_case_only_on_one_side_is_absent_with_a_reason(self):
        cases, reasons = pair_cases([_run("a", passed=True)], [_run("b", passed=True)])
        self.assertEqual([("a", True, None), ("b", None, True)], cases)
        self.assertIn("B 侧没有这条 case", reasons["a"])
        self.assertIn("A 侧没有这条 case", reasons["b"])

    def test_a_duplicate_name_is_flagged_instead_of_silently_taking_one(self):
        """同一批里同名两次 ⇒ 不猜该用哪一次。

        真机那侧 `case` 名带 `#1/#2/#3` 后缀就是为了让它唯一; 出现重名说明记录方式与这个
        约定打架, 那件事必须印出来而不是一次被另一次覆盖。
        """
        cases, reasons = pair_cases([_run("a", passed=True), _run("a", passed=False)],
                                    [_run("a", passed=True)])
        self.assertEqual([("a", None, True)], cases)
        self.assertIn("A 侧同名出现多次", reasons["a"])

    def test_pairing_aligns_by_name_not_by_position(self):
        cases, _ = pair_cases([_run("m", passed=True), _run("l", passed=False)],
                              [_run("l", passed=False), _run("m", passed=False)])
        self.assertEqual([("l", False, False), ("m", True, False)], cases)
        self.assertEqual(["m"], _ev(cases)["regressed_cases"])

    def test_compare_runs_carries_the_column(self):
        runs_a = [_run("c1", passed=True), _run("c2", passed=False), _run("c3", passed=False)]
        runs_b = [_run("c1", passed=False), _run("c2", passed=True), _run("c3", passed=True)]
        att = compare_runs(runs_a, runs_b)["attributable"]
        self.assertEqual(["c1"], att["regressed_cases"])
        self.assertEqual(["c2", "c3"], att["recovered_cases"])
        self.assertIs(True, att["accepted_metric_only"])
        self.assertIs(False, att["accepted_gate"])


class TestTheColumnHasAReader(unittest.TestCase):
    @staticmethod
    def _cmp(runs_a, runs_b):
        return compare_runs(runs_a, runs_b, meta_a={"git_head": "aaaaaaa"},
                            meta_b={"git_head": "bbbbbbb"})

    def test_the_column_names_the_regressed_cases(self):
        """先断言**本栏独有的标题**再断数值（I32/I39 那一族: needle 落在别处就是假绿）"""
        # 两条修好、一条变坏 ⇒ 配对口径 Δ=+1: metric-only 接受, 可归因门因那条回归不接受
        text = format_comparison(self._cmp(
            [_run("matmul#1", passed=True), _run("matmul#2", passed=False),
             _run("layernorm#3", passed=False)],
            [_run("matmul#1", passed=False), _run("matmul#2", passed=True),
             _run("layernorm#3", passed=True)]))
        self.assertIn("【逐 case 的得与失】", text)
        self.assertIn("配对口径的通过数: 1 → 2 (Δ=+1", text)
        self.assertIn("❌ 变坏: matmul#1", text)
        self.assertIn("✅ 修好: layernorm#3, matmul#2", text)
        self.assertIn("分歧 有", text)
        self.assertIn("聚合数与具体回归打架", text)

    def test_a_batch_with_nothing_matched_says_measured_nothing(self):
        att = {"matched": 0, "counts": {k: 0 for k in ATTRIBUTABLE_OUTCOMES},
               "unmatched_cases": ["x"], "unmatched_reasons": {"x": ["A 侧没有这条 case"]},
               "recovered_cases": [], "regressed_cases": [],
               "accepted_metric_only": None, "accepted_gate": None, "disagreement": None,
               "positive_evidence": None}
        text = format_comparison({"verdict": "underpowered", "reason": "r", "alpha": 0.05,
                                  "target_delta": 0.3, "meta": {}, "attributable": att,
                                  "task_success": {"a": (0, 0), "b": (0, 0), "rate_a": None,
                                                   "rate_b": None, "delta": None,
                                                   "p_value": None},
                                  "l3": {"a": (0, 0), "b": (0, 0), "p_value": None, "note": ""},
                                  "power": {"n_a": 0, "n_b": 0, "n_min": 0,
                                            "detectable_delta": None, "required_per_arm": None,
                                            "target_delta": 0.3},
                                  "speedup": {"a": None, "b": None}})
        self.assertIn("【逐 case 的得与失】", text)
        self.assertIn("测不到: 一条都没配上", text)
        self.assertIn("这不是「没有回归」", text)
        self.assertNotIn("两条规则: metric-only", text, "没有原料时不该印一条像结论的两规则读数")

    def test_agreeing_rules_do_not_claim_a_disagreement(self):
        """**反向对照**: 「分歧 无」必须真的印成无 —— 只测「有分歧时报出来」会放过条件反转"""
        text = format_comparison(self._cmp(
            [_run("a", passed=False), _run("b", passed=False)],
            [_run("a", passed=True), _run("b", passed=True)]))
        self.assertIn("【逐 case 的得与失】", text)
        self.assertIn("分歧 无", text)
        self.assertNotIn("分歧 有", text)
        # R3 的证人: 那句「打架」的注解是无条件印的话, 上面两条断言都还绿 —— 它印的是一句
        # 对这份证据不成立的解释。与 #87 的 M10 同族: 「该报的时候报」测到了, 「不该报时别报」
        # 才是这条断言的职责。
        self.assertNotIn("聚合数与具体回归打架", text)
        self.assertNotIn("❌ 变坏", text)

    def test_the_column_does_not_move_the_statistical_verdict(self):
        """本片的存在理由那条承诺: 只观测。同一份证据上判决文案与这一栏无关"""
        runs_a = [_run("c", passed=True)]
        runs_b = [_run("c", passed=False)]
        cmp_obj = self._cmp(runs_a, runs_b)
        self.assertEqual("underpowered", cmp_obj["verdict"], cmp_obj["reason"])
        self.assertIs(False, cmp_obj["attributable"]["accepted_gate"])
        text = format_comparison(cmp_obj)
        self.assertIn("判决: ⚠️  判不出", text)
        self.assertIn("只观测不改判决", text)
        self.assertLess(text.index("判决: ⚠️"), text.index("【逐 case 的得与失】"),
                        "这一栏排在判决之后 —— 它是补名字, 不是第二张判决书")

    @unittest.skipUnless(A_ROOT.exists() and B_ROOT.exists(),
                         "本机没有那两份存量报表 (未跟踪产物, 净树上会跳过)")
    def test_the_stored_real_pair(self):
        """存量真机证据上的读数（`0a4f0fd` 5/6 → `4e80b13` 3/6）: 本片的数字都从这儿来

        断言写成**互推**而不是硬编码名字: 同一份 case 清单 ⇒ 配上的条数应等于两侧完成的
        运行数; 而「聚合门说不知道」与「逐 case 说变坏了 3 条」必须同时成立, 那句话才是本片的
        立论。名字列表也核一遍, 因为它是要印出来的东西。
        """
        ja = json.loads(A_ROOT.read_text(encoding="utf-8"))
        jb = json.loads(B_ROOT.read_text(encoding="utf-8"))
        att = compare_runs(ja["runs"], jb["runs"])["attributable"]
        self.assertEqual(len(ja["runs"]), att["matched"])
        self.assertEqual((5, 3), (att["pass_a"], att["pass_b"]))
        self.assertEqual(1, att["counts"]["recovered"])
        self.assertEqual(3, att["counts"]["regressed"])
        self.assertIs(False, att["accepted_metric_only"])
        self.assertEqual(["layernorm/fp32/16384x6144#3",
                          "matmul/fp16/4096x4096x4096#1",
                          "matmul/fp16/4096x4096x4096#3"],
                         sorted(att["regressed_cases"]))
        self.assertEqual(["matmul/fp16/4096x4096x4096#2"], att["recovered_cases"])
        self.assertIsNone(att["unmatched_cases"] or None, "这两批的 case 清单应当完全一致")


if __name__ == "__main__":
    unittest.main()
