# -*- coding: utf-8 -*-
"""#87 第 4 片 (改善判据那条带宽跨轮是不是同一把) 的变异表。

跑法:
    python3 -B -m benchmarks.calibration.mutation_calibrate \\
        --spec benchmarks/mut_specs/improvement_band_drift.py

本表 12 行**全在行为分母**、结构分母 0 行 —— 这不是省事，是这片的每条变异今天都能被一个
具体输入观察到：它改的是归因分支与报表文案，而文案就是它的输出。想加而没加的那条是
「`_band_of_tier` 被抄第二份」：那份抄件对同一份输入给同一个数 ⇒ 今天没有行为差异可测，
按 #83 它得带隔离理由才能进表，而为一个还没出现的第二个实现写 AST 守卫不在本片的账上。

一行了得特别解释: B12 的证人**不在本片的测试文件里** —— 把带宽的键改成 `"l3_ruler"` 时，
本片的用例全都照过 (它们一律用常量读)，杀掉它的是邻居那张 `test_perf_ruler_across_rounds.py`
里真跑那组 (同一轮记录上性能尺那一格被带宽的浮点覆盖)。这条留在表里是因为它正是
「跨表覆盖」的证据：共用一个键名这件事，只有另一张表会响。
"""

from benchmarks.calibration.mutation_calibrate import Row

FILES = {
    "ms": "operator_agent/decide/measurement.py",
    "cond": "operator_agent/app/conductor.py",
    "hpr": "benchmarks/drivers/harness_pass_rate.py",
}

TARGETS = [
    "tests/unit/test_improvement_band_drift.py",
    "tests/unit/test_measurement_validity.py",
    "tests/unit/test_perf_ruler_across_rounds.py",
]

ROWS = [
    Row("B1 归因不查表：档位不同就一律算「降级换宽」",
        "ms",
        "        e1, e2 = _band_of_tier(t1), _band_of_tier(t2)\n"
        "        if e1 is None or e2 is None or e1 != b1 or e2 != b2:\n"
        "            explainable = False   # 连表都对不上 (或那一格根本不是个档位) —— 那就不只是降级\n"
        "            continue",
        "        drifted_by_tier = True\n"
        "        continue",
        "存在具体输入可观察: 轮 1 档位 valid/带宽 0.001、轮 2 档位 degraded/带宽 0.009 —— "
        "0.009 对不上 degraded 那条 0.004 ⇒ 本该 `unexplained`，改后报成 `tier_explained`，"
        "于是「表被换过」这件事被印成「测量变差了，另一格早就记着」⇒ 指错修法", True),

    Row("B2 档位不同也算规格漂移（两种归因折成一种）",
        "ms",
        "        if t1 == t2:\n"
        "            out[\"band_state\"] = BAND_SPEC_DRIFT\n"
        "            out[\"drift_rounds\"].append((r1, r2))\n"
        "            continue",
        "        out[\"band_state\"] = BAND_SPEC_DRIFT\n"
        "        out[\"drift_rounds\"].append((r1, r2))\n"
        "        continue",
        "存在具体输入可观察: valid 0.001 → degraded 0.004 那对 (档位换了、值对得上表) 本该是"
        " `tier_explained`，改后一律 `spec_drift` ⇒ 本片立片的那句「别把降级报成漂移」正面对撞", True),

    Row("B3 不足 2 轮时兜成 `same`（没有输入读成一致）",
        "ms", "    if len(used) < 2:", "    if len(used) < 1:",
        "存在具体输入可观察: 只有一轮用过带宽 ⇒ 改后 `distinct` 恰好一个值 ⇒ 判成 `BAND_SAME`，"
        "报表于是把「这一场没凑出比较」印成「各轮同一条带宽」(三态里缺证据被当成通过)", True),

    Row("B4 「走到判决但没用带宽」的轮并进 rounds_with_band",
        "ms",
        "        if band is None:\n            no_band += 1\n            continue",
        "        if False:\n            no_band += 1\n            continue",
        "存在具体输入可观察: [0.001, None, 缺席, 0.001] 那组里 `None` 那一轮 (AMBIGUOUS 没 consult "
        "表) 被当成一个读数 ⇒ `rounds_with_band` 从 2 变 3、`rounds_no_band` 从 1 变 0，三格被折成两格", True),

    Row("B5 `noise_band_for(UNINFORMATIVE)` 回表里的 0.02（记一个没用过的数）",
        "ms",
        "    if validity is Validity.UNINFORMATIVE:\n        return None\n"
        "    return NOISE_BAND[validity]",
        "    return NOISE_BAND[validity]",
        "存在具体输入可观察: `noise_band_for(UNINFORMATIVE)` 与 `verdict(…, UNINFORMATIVE)` 的第三"
        "元素都从 `None` 变成 0.02 —— 那一支判决在查表之前就返回 AMBIGUOUS，把从没参与决定的数"
        "记进血缘就是给产物造证据 (#81 那一族的反面)", True),

    Row("B6 conductor 忘了在判决现场写这一格（产它的与读它的断链）",
        "cond", "                    _round_rec[LINEAGE_BAND_KEY] = used_band", "",
        "存在具体条件可观察: 真跑一场三轮任务后血缘里没有这一项 ⇒ `TestThroughARealRun` 那三条"
        "全红 (键缺席 ≠ 各轮同带宽)，报表那栏退成「测不到」。P7 那一课: 纯规则用例对这条无感，"
        "只有从真实循环里查这一格才杀得掉", True),

    Row("B7 报表把 tier_explained 与 spec_drift 合成一个数",
        "hpr",
        "        \"band_tier_explained_cases\": _cases(BAND_TIER_EXPLAINED),\n"
        "        \"band_spec_drift_cases\": _cases(BAND_SPEC_DRIFT),",
        "        \"band_tier_explained_cases\": [],\n"
        "        \"band_spec_drift_cases\": (_cases(BAND_TIER_EXPLAINED)\n"
        "                                    + _cases(BAND_SPEC_DRIFT)),",
        "存在具体输入可观察: 五步夹具里 b (降级换宽) 与 c (同档位换值) 各一个 ⇒ 改后 "
        "`band_tier_explained_cases` 数 0、`band_spec_drift_cases` 数 2，且 b 的名字出现在"
        "「规格漂移:」那一行下 ⇒ 两条计数断言 + 那条反向对照一起红", True),

    Row("B8 报表「测不到」那一支反过来（没原料时印一个像结论的 0）",
        "hpr", "    if agg[\"band_runs_with_band\"] == 0:", "    if agg[\"band_runs_with_band\"] > 0:",
        "存在具体输入可观察: 四场全是旧产物 (没有这一项) 时改后走 else 支 ⇒ 印"
        "「可比的运行 0 个 · 同一条带宽 0 …」，也就是把「无从可查」报成一个像结论的 0，"
        "而 `测不到` 与键名都不再出现", True),

    Row("B9 报表自己拼句子，不走 `band_note`（同一件事两处两种说法）",
        "hpr",
        "        return sorted(f\"{c} → {band_note(a)}\" for c, a in per_run.items()",
        "        return sorted(f\"{c} → 带宽状态 {a.get('band_state')}\" for c, a in per_run.items()",
        "存在具体输入可观察: 报表正文里的归因句子与 conductor 现场打印那句不再是同一实现产出"
        "的同一句话 ⇒ 那条测**出处**的用例 (逐字比较 `band_note(agg)` 是否原样出现在报表里) 红。"
        "只改措辞不改数, 但两处两种说法以后就会长成两件事 (I22 那一族)", True),

    Row("B10 conductor 不用判决返回的带宽、自己再查一次表",
        "cond",
        "                    _round_rec[LINEAGE_BAND_KEY] = used_band",
        "                    from .measurement import noise_band_for as _nb\n"
        "                    _round_rec[LINEAGE_BAND_KEY] = (\n"
        "                        None if v is Verdict.AMBIGUOUS else _nb(_validity))",
        "存在具体条件可观察: 这就是本片第一版的真红法 —— 真跑一场、在第 1 次与第 2 次判决之间"
        "改一次表，落盘的是 `[0.1234, 0.1234]` 而判决实际用过 `[0.001, 0.1234]` ⇒ "
        "`test_a_table_changed_between_two_verdicts_shows_up_as_spec_drift` 红 (改成从判决返回之后"
        "这条才绿)。记下的数与用过的数不是同一个, 跨轮那栏就整栏失去意义", True),

    Row("B11 判决没用过带宽的两支也返回表里的值",
        "ms",
        "    if current is None or previous is None or current <= 0 or previous <= 0:\n"
        "        return Verdict.AMBIGUOUS, \"无可比的性能值 (缺 current 或 previous)\", None\n"
        "    if validity is Validity.UNINFORMATIVE:\n"
        "        return Verdict.AMBIGUOUS, \"测量档位 uninformative: 判决带宽比信号本身宽\", None",
        "    if current is None or previous is None or current <= 0 or previous <= 0:\n"
        "        return Verdict.AMBIGUOUS, \"无可比的性能值 (缺 current 或 previous)\", "
        "NOISE_BAND[validity]\n"
        "    if validity is Validity.UNINFORMATIVE:\n"
        "        return Verdict.AMBIGUOUS, \"测量档位 uninformative: 判决带宽比信号本身宽\", "
        "NOISE_BAND[validity]",
        "存在具体输入可观察: `verdict(None, 1.0, VALID)[2]` 与 `verdict(1.2, 1.0, UNINFORMATIVE)[2]` "
        "都从 `None` 变成一个数 ⇒ 那两轮从「走到判决但没用带宽」被移到「用过带宽」，`rounds_no_band` "
        "归零、跨轮比较的分母悄悄变大。与 B5 同族, 但从产它的那一侧改", True),

    Row("B12 带宽那一格与性能尺共用同一个键名",
        "ms", 'LINEAGE_BAND_KEY = "l3_improvement_band"', 'LINEAGE_BAND_KEY = "l3_ruler"',
        "存在具体条件可观察, 但证人不在本片的文件里: 本片的用例一律通过常量读写这一格 ⇒ 全部照过; "
        "真跑那组里同一轮记录上的性能尺那一格 (`LINEAGE_PERF_RULER_KEY = \"l3_ruler\"`) 被判决现场"
        "后写的浮点覆盖 ⇒ `tests/unit/test_perf_ruler_across_rounds.py` 那批真跑用例红。留这一行是因为"
        "它证明跨表覆盖真实存在: 两个读者共用一个键名时, 只有另一个读者会响 (与 P6 同一条, 方向反过来)", True),
]
