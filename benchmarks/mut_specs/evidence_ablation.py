"""变异表: §21.3 对照实验的接线与结论边界 (靶子 = `tests/test_evidence_ablation.py`)

⚠️ 这张表由**主 agent 串行标定** (`python3 -B -m benchmarks.mutation_calibrate --spec ...`)。
`--check-only` 只核锚点唯一性与 witness 字段, 不动盘。

每一行都在问同一个问题: 「这个实验的接线**坏了**的时候, 有没有一条守卫会红?」
坏法都是这个实验特有的那种 —— 不是算法算错, 而是**对照不再是对照**:
两臂多了第二处差异 / 两臂都开 / 留痕说了谎 / 结论越过 N / 坏码被改成好码。

`behavioral=False` 的那一行 (`M7`) 今天在行为层不可区分: 两臂顺序怎么排, 两边都会跑完,
`pair_up` 与结论只按 (场景, 臂) 配对, 顺序不进任何读数 —— 只有读源码的那条守卫杀得掉它。
"""
from benchmarks.mutation_calibrate import Row

FILES = {
    "ab": "benchmarks/evidence_ablation_h20.py",
    "te": "tests/test_evidence_ablation.py",
}

TARGETS = ["tests/test_evidence_ablation.py"]

ROWS = [
    Row("M1 两臂多出第二处差异 (轮数随开关变)", "ab",
        '    return {"structured_evidence": bool(structured_evidence),\n'
        '            "max_optimize_rounds": ROUNDS,',
        '    return {"structured_evidence": bool(structured_evidence),\n'
        '            "max_optimize_rounds": ROUNDS if structured_evidence else ROUNDS + 1,',
        "**具体输入**: 调 `arm_diff()` ⇒ 键差集变成 "
        "`{'structured_evidence', 'max_optimize_rounds'}` ⇒ "
        "`test_the_key_diff_is_exactly_the_switch` 红。这一行钉的是「对照失去对照」的第一种"
        "坏法: 两臂不止差一个开关时, 后面所有差值都无法归因到那一段证据"),

    Row("M2 两臂都开 (关臂名存实亡)", "ab",
        '    return {"structured_evidence": bool(structured_evidence),',
        '    return {"structured_evidence": True,',
        "**具体输入**: `arm_diff()` 变空集 (两臂同值) ⇒ "
        "`test_the_key_diff_is_exactly_the_switch` / `test_the_switch_actually_flips` 红; "
        "并且离线两臂跑一遍时关臂也会注入 ⇒ `test_the_off_arm_records_no_evidence` 红。"
        "这是「对照实验」最可能的自欺形态: 开关还在, 但两臂跑的是同一条路"),

    Row("M3 留痕恒说「注入过」(血缘说谎)", "ab",
        '        "evidence_injected": [bool(e.get("injected")) for e in evidence],',
        '        "evidence_injected": [True for e in evidence],',
        "**具体输入**: 离线跑关臂 ⇒ `evidence_injected` 里出现 `True` ⇒ "
        "`test_the_off_arm_records_no_evidence` 红。这一行是「把留痕断言去掉」那条要求的"
        "可标定形态: 断言在, 血缘说谎就会被抓到; 断言被删掉, 这一行就逃逸 (ESCAPED 也是读数)"),

    Row("M4 结论里写上「有收益」", "ab",
        '        f"  · 结果 (分辨不出的一侧): 通过 {on_pass}/{n} (开) vs {off_pass}/{n} (关), "',
        '        f"  · 结果: 证据段**有收益** —— 通过 {on_pass}/{n} (开) vs {off_pass}/{n} (关), "',
        "**具体输入**: N=3、两臂各 3 格 (离线替身跑得出来) ⇒ 结论句里出现「有收益」这类"
        "越界断言 ⇒ `test_the_conclusion_has_n_the_range_and_no_claim` 红。这一行守的是本仓"
        "最贵的那类错误: 把 N=3 的方向读成结论"),

    Row("M5 结论里不再写量程数字", "ab",
        '        f"  · 量程: N={n} 时通过率量具最多只能分辨 ≈{detectable} pp 的差 "',
        '        f"  · 量程: N 小时分辨不出差别 "',
        "**具体输入**: 同上那一跑 ⇒ 结论句里找不到 "
        "`min_detectable_delta(n)` 算出来的那个数 ⇒ "
        "`test_the_conclusion_has_n_the_range_and_no_claim` 红。「不许声称收益」这句话"
        "只有在量程也在场时才有内容, 少了它读者无从判断 N 到底不够到哪一档"),

    Row("M6 定点改动不再动源码 (坏码==好码)", "ab",
        "    return good.replace(old, new)",
        "    return good",
        "**具体输入**: 构造三个场景 ⇒ 每一格的 `broken == good` ⇒ "
        "`test_every_broken_source_differs_from_the_good_one` 红; 离线跑一遍时第一轮直接通过、"
        "没有 Fixer ⇒ `test_every_scenario_reaches_the_fixer` 也红。这是**静默失效**那一类: "
        "改动没命中时脚本照样跑完, 只是对照价值在无声中归零"),

    Row("M7 两臂顺序钉死 (不再轮换)", "ab",
        "            arms = [True, False] if (si + rep) % 2 else [False, True]",
        "            arms = [True, False]",
        "今天**行为不可区分**: 两种顺序都会把两臂跑完, `pair_up` 只按 (场景, 臂) 配对, "
        "顺序不进任何读数 —— 它防的是端点漂移被读成开关效应, 那需要跨天的重复跑才看得见。"
        "所以只有源码守卫 `test_the_arm_order_alternates_between_scenarios` 杀得掉它 "
        "(behavioral=False; 2026-10-06 标定时发现这个关键字当初只写在证人字符串里、没真的传, "
        "于是它在分母里被算成了行为行 —— 已补上)",
        behavioral=False),

    Row("M8 `error_excerpt` 不丢掉标题那一行", "ab",
        "    lines = [ln for ln in body.splitlines()[1:] if ln.strip()]   # 丢掉标题那一行",
        "    lines = [ln for ln in body.splitlines() if ln.strip()]",
        "**具体输入**: 一段带 `## 错误信息` 的 prompt (用例里手写的那份) ⇒ 取回来的首行变成标题 "
        "`## 错误信息` 而不是错误原话。"
        "「场景自检」印的就是这一行的内容, 取错读者看到的是样板而不是死因。"
        "⚠️ 2026-10-06 首次标定这一行**逃逸**了: 用例当时只断言「样板句不在」，标题行漏掉了 —— "
        "补了 `assertNotIn('## 错误信息')` + 首行逐字相等两条后才杀掉。"
        "(所以这一行的名字也改了: 它动的从来不是样板句, 是标题行)"),
]