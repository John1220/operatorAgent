# -*- coding: utf-8 -*-
"""一期 T3.7 (人在环节上的出现记录) 的变异表。

跑法（命令里的模块路径是 Phase 3 分目录之后的真实路径；旧写法 `-m benchmarks.calibration.mutation_calibrate`
已不存在，别照抄别的 spec 头部那行）:
    python3 -B -m benchmarks.calibration.mutation_calibrate \\
        --spec benchmarks/mut_specs/human_ledger.py
    python3 -B -m benchmarks.calibration.mutation_calibrate --spec … --check-only   # 只核锚点与证人

八行全在行为分母：这个量具的每条不变量都对应一个**能打印出来的错表**，所以没有结构行。
它们共同守的是同一类错误 —— 把「没问过」显示成「问过」，或把两种单位折成一个数。

M8 是这一族里最阴的一条：`bool` 在 Python 里是 `int` 的子类，所以只写
`not isinstance(wall, (int, float))` 的校验会让 `True` 变成 1 秒安然通过耗时列 ——
这一条不是假想病，是本地语言层的既成事实，因此单独占一行并要求活的用例。
"""

from benchmarks.calibration.mutation_calibrate import Row

FILES = {
    "hl": "operator_agent/core/human_ledger.py",
}

TARGETS = [
    "tests/unit/test_human_ledger.py",
]

ROWS = [
    Row("M1 新表的 `involved` 默认从 unknown 改成 no（没问被当成没人做）", "hl",
        '"involved": UNKNOWN, "action": None',
        '"involved": "no", "action": None',
        "存在具体输入可观察: `new_ledger('run-x')` 之后直接 `counts()` —— 原实现报 "
        "answered=0 / no=0 / unknown=6，改后报 answered=6 / no=6 / unknown=0。报表上"
        "「一张没问过的空表」与「六个环节都没人介入」就长成一个样，正是 §6.3 要防的那类假象",
        True),

    Row("M2 完整度的分母从「全部环节」改成「已答环节」（vacuous truth）", "hl",
        '    return {"total": len(STAGES), "answered": answered,',
        '    return {"total": answered, "answered": answered,',
        "存在具体输入可观察: 空表调 `counts()` —— 原实现 total=6 / answered=0，改后 total=0 / "
        "answered=0，任何下游把 answered/total 当完整度就得到 100%「已答完」。未答项从分母里消失",
        True),

    Row("M3 量级档校验关掉，接受任意数值（档被当成秒）", "hl",
        "    if minutes_band is not None and minutes_band not in MINUTE_BANDS:",
        '    if minutes_band is not None and minutes_band in ("__never__",):',
        "存在具体输入可观察: `record(ledger, 'prepare', involved='yes', minutes_band=25)` —— "
        "原实现抛 LedgerError 且消息里列出合法档位，改后 25 直接进表；随后 `as_report_rows` 的"
        "人自报列混进数值，与机器秒在同一张表里可被相加", True),

    Row("M4 填档不再要求 involved=yes（没问被伪装成问过且很快）", "hl",
        '    if minutes_band is not None and entry_asked != "yes":',
        '    if minutes_band is not None and entry_asked == "no":',
        "存在具体输入可观察: `record(ledger, 'analyze', minutes_band='5-15')`（完全不传 involved）"
        "—— 原实现拒绝并说明该格当前是 unknown，改后静默收下；报表于是显示「人只花了 5–15 分钟」，"
        "而事实是这一格从没被问过", True),

    Row("M5 环节名单校验关掉（拼错的环节名静默新增一格）", "hl",
        "    if stage not in STAGES:",
        '    if stage not in ("__never__",):',
        "存在具体输入可观察: `record(ledger, 'integrate')` —— 原实现抛 LedgerError 并把六个合法"
        "环节名回给调用方，改后不报错。后果不是崩，是**表里多出一个谁都不认识的环节**，六格分母"
        "与实际记录从此错位；而且它把「代码里根本没有接入阶段」这个事实洗成了「接入记过了」", True),

    Row("M6 报表把人自报的档位写进机器秒那一列（两栏折成一栏）", "hl",
        '                     "machine_wall_s": wall})',
        '                     "machine_wall_s": entry["minutes_band"]})',
        "存在具体输入可观察: `record(..., involved='yes', minutes_band='15-60')` 后 "
        "`as_report_rows(ledger, {'check_run': 241.2})` —— 原实现该列是 241.2，改后是字符串 "
        "'15-60'；用例既断言数值也断言「量级档必须是字符串」，两栏合并立刻红。这正是 §6.3 "
        "「模型墙钟不能代替人工投入」的反向形态：反过来用人的档位冒充墙钟", True),

    Row("M7 读回时不再要求六个环节齐全（少一格算干净）", "hl",
        '    if set(ledger.get("entries", {})) != set(STAGES):',
        "    if False:",
        "存在具体输入可观察: 写一张删掉 `compare` 的表再 `read_ledger` —— 原实现拒绝并回显缺的"
        "环节名，改后照单全收。后果是「unknown 太多太难看」可以被悄悄删成简洁表，"
        "完整度分母同时缩水，报表看起来更绿", True),

    Row("M8 只挡非数值、不挡 bool（True 当成 1 秒进耗时列）", "hl",
        "        if isinstance(wall, bool) or (wall is not None and not isinstance(wall, (int, float))):",
        "        if wall is not None and not isinstance(wall, (int, float)):",
        "存在具体输入可观察: `as_report_rows(new_ledger('run-x'), {'prepare': True})` —— 原实现"
        "拒绝，改后放行（Python 里 `isinstance(True, int)` 为真），耗时列里出现一个 1「秒」。"
        "这一行的价值在于它是语言层既成事实，不是设想出来的输入", True),
]
