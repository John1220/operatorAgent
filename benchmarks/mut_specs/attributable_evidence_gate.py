"""`tests/unit/test_attributable_evidence_gate.py` 那张变异表的可执行形态 (#103 第 1 片)

这一片把 HarnessLens 行 1050 那句做法搬过来 —— **两条规则算在同一份存量证据上, 什么都不重新
执行**：metric-only（配对口径 Δ>0 就接受）与可归因门（要有正证据且零回归），再点名变坏的 case。
判据在 `measurement.attributable_evidence`，配对与报表在 `benchmarks/drivers/harness_pass_rate.py`。

跑: python3 -B -m benchmarks.mutation_calibrate \
      --spec benchmarks/mut_specs/attributable_evidence_gate.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "ms": "operator_agent/measurement.py",
    "hp": "benchmarks/drivers/harness_pass_rate.py",
}

TARGETS = ["tests/unit/test_attributable_evidence_gate.py"]

ROWS = [
    Row("A1 preservation 也算正证据（行 880 那句被反写）", "ms",
        '    if recovered:\n        out["positive_evidence"] = "recovery"',
        '    if recovered or counts[ATTRIBUTABLE_STABLE_PASS]:\n'
        '        out["positive_evidence"] = "recovery"',
        "⇒ `test_preservation_alone_is_not_positive_evidence` 红。原文那句 "
        "\"preservation alone is insufficient\" 是这条门与「什么都没变所以通过」之间唯一的区别; "
        "把它写成或, 一栏不动的批也会被判「有正证据」"),

    Row("A2 可归因门不看回归数（只看有没有正证据）", "ms",
        '    out["accepted_gate"] = bool(out["positive_evidence"]) and not regressed',
        '    out["accepted_gate"] = bool(out["positive_evidence"])',
        "⇒ 3 条红 (`test_aggregate_up_while_a_case_regresses_is_the_disagreement` / "
        "`test_compare_runs_carries_the_column` / `test_the_column_names_the_regressed_cases`)。整篇文档对着的病就是 \"aggregate scores to obscure "
        "specific regressions\" —— 门不看回归就等于没这道门"),

    Row("A3 缺席读数折成「这一侧没过」（缺证据当反证）", "ms",
        "        if a is None or b is None:\n            unmatched.append(str(name))\n            continue",
        "        if a is None and b is None:\n            unmatched.append(str(name))\n            continue",
        "⇒ 2 条红 (`test_absent_readings_are_none_too`: (True, None) 被算成「两侧都不过」、"
        "(None, False) 被算成回归; `test_pass_counts_are_only_over_matched_cases`: 换考卷被算成涨分)。#53 那一族的反面 —— 崩溃的那次没有读数, 不是读数说了「不」"),

    Row("A4 metric-only 改成「聚合没跌就算好」（Δ>=0）", "ms",
        '    out["accepted_metric_only"] = pass_b > pass_a',
        '    out["accepted_metric_only"] = pass_b >= pass_a',
        "⇒ `test_preservation_alone_is_not_positive_evidence` 红（它断 Δ=0 时 metric-only 也"
        "不接受）。这条也是 A1 的对照面: 两条一起改就看不出谁在管什么"),

    Row("A5 一条都没配上时不再早退（空输入把三条结论折成布尔）", "ms",
        "    if not matched:\n        out[\"note\"]",
        "    if False:\n        out[\"note\"]",
        "⇒ `test_nothing_matched_is_not_no_regression` 与 `test_absent_readings_are_none_too` 红: "
        "空证据会印出「metric-only 不接受 · 门不接受 · 分歧 无」这种**像结论的三件套**, 而它底下"
        "没有一条配对样本 (I25 那一族: 输入是空气时判据永真/永假的两侧)"),

    Row("P1 崩溃的那次读成「没过」而不是「没有读数」", "hp",
        '            elif not r.get("ok"):\n                seen[name] = None',
        '            elif not r.get("ok"):\n                seen[name] = False',
        "⇒ `test_a_crashed_run_is_no_reading_not_a_failure` 红。这一行与 A3 是同一个错误长在"
        "**搬运层**上: 规则层不许折, 配对层先折了规则层也拦不住"),

    Row("P2 同名两次静默取最后一次（约定的打架不印出来）", "hp",
        '            if name in seen:\n                seen[name] = "duplicate"',
        '            if name in seen:\n                seen[name] = bool(r.get("final_all_passed"))',
        "⇒ `test_a_duplicate_name_is_flagged_instead_of_silently_taking_one` 红。"
        "`case` 名带 #1/#2/#3 后缀是为了唯一; 真出现重名说明记录方式与约定打架, "
        "静默取一次就等于这件事永远不会被看见"),

    Row("P3 「为什么没有结论」不填（无结论只剩一个名字）", "hp",
        '            elif val is None:\n                reasons.setdefault(name, []).append(f"{side} 侧那次运行崩溃了")',
        "            elif val is None:\n                pass",
        "⇒ `test_a_crashed_run_is_no_reading_not_a_failure` 红。报表那一行会退化成"
        "「⚪ 无结论: x —— 这一侧没有可用读数」, 读者分不清是崩溃、没这条 case 还是重名"),

    Row("R1 变坏的 case 不点名（回到只有聚合数）", "hp",
        '        if att["regressed_cases"]:\n            lines.append("  ❌ 变坏: "',
        '        if False:\n            lines.append("  ❌ 变坏: "',
        "⇒ `test_the_column_names_the_regressed_cases` 红 (1 条; 存量真机那条断的是**数据**而不是打印, "
        "所以它对这条变异是绿的 —— 名字与数值是两个读者)。这一行就是本片的全部价值 —— 名字不印出来, 这一栏与聚合数没有区别"),

    Row("R2 没有配对样本时不印「测不到」（缺原料读成零回归）", "hp",
        '    if not att.get("matched"):\n        lines.append("  测不到: 一条都没配上',
        '    if False:\n        lines.append("  测不到: 一条都没配上',
        "⇒ `test_a_batch_with_nothing_matched_says_measured_nothing` 红: 那时印的是"
        "「配上的 case 0 · 变坏 0」, 读者会当成「查过了, 没有任何回归」"),

    Row("R3 「分歧 有」那句变成无条件印（不区分有没有分歧），反向对照失效", "hp",
        "                     + (\" ← 聚合数与具体回归打架的那一次\" if _dis is True else \"\"))",
        '                     + (" ← 聚合数与具体回归打架的那一次"))',
        "⇒ 1 条红 (`test_agreeing_rules_do_not_claim_a_disagreement`; 打印数值那条对它绿, 因为那份夹具"
        "本来就有分歧, 注解印在那儿是对的)。⚠️ **第一次跑这条是"
  "逃逸的**: 那时用例只断言了「分歧 有/无」那一小串, 而这个变异动的是**后面那句注解**, "
  "两条断言都还绿 —— 补上 `assertNotIn(聚合数与具体回归打架)` 才杀掉。这是 #87 那条 M10 的同一族, "
  "也是本仓第一次把「反向对照」用到**注解文案**而不只是数值上"),

    Row("R4 compare_runs 不再带这一格（产它的与读它的断链）", "hp",
        '        "verdict": verdict,\n        "attributable": attributable,',
        '        "verdict": verdict,',
        "⇒ 5 条红（走 `compare_runs` 的那些 + 打印那组, KeyError）。这一行也是「算了没人读」的反面: "
        "规则算出来了但没接到任何消费者上, 报表那一栏就永远是空的"),
]
