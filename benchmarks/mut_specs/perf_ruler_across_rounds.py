# -*- coding: utf-8 -*-
"""#87 第 3 片 (性能那一半的尺) 的变异表。

跑法:
    python3 -m benchmarks.calibration.mutation_calibrate --spec benchmarks/mut_specs/perf_ruler_across_rounds.py

`behavioral=False` 的两行都写了「为什么今天行为层不可区分」—— 它们只有结构守卫杀得掉,
按 #83 那条不进行为分母。
"""

from benchmarks.calibration.mutation_calibrate import Row

FILES = {
    "ms": "operator_agent/decide/measurement.py",
    "ver": "operator_agent/decide/verifier.py",
    "cond": "operator_agent/app/conductor.py",
    "hpr": "benchmarks/drivers/harness_pass_rate.py",
}

TARGETS = [
    "tests/unit/test_perf_ruler_across_rounds.py",
    "tests/unit/test_cross_round_ruler.py",
    "tests/unit/test_bound_axis_gate.py",
]

ROWS = [
    Row("P1 `unknown` 被当成一个读数 (撤掉 not_a_value)",
        "ms",
        'return _ruler_reading(l3_metrics, PERF_GATE_PARAM_KEYS,\n                          not_a_value={BOUND_AXIS_KEY: (BOUND_AXIS_UNKNOWN,)})',
        'return _ruler_reading(l3_metrics, PERF_GATE_PARAM_KEYS)',
        "两轮都推不出轴时 `bound_type` 都是 unknown ⇒ 逐键判据读成「这一键各轮同值」, "
        "于是「谁都不知道是哪条轴」被印成「已确认同一把尺」", True),

    Row("P2 键集合里去掉判据 (只剩数值门槛)",
        "ms", 'PERF_GATE_PARAM_KEYS: tuple = ("min_speedup_required", "perf_criterion", BOUND_AXIS_KEY)',
        'PERF_GATE_PARAM_KEYS: tuple = ("min_speedup_required", BOUND_AXIS_KEY)',
        "「从比加速比改成比 GFLOPS」这件事不再有证人: 两条路 min_speedup 同为 1.0 时"
        "跨轮读成一致, 而它们判的根本不是同一个量", True),

    Row("P3 legacy 那条不声明判据",
        "ver",
        '"min_speedup_required": min_required,\n                        "perf_criterion": PERF_CRITERION_ABSOLUTE_GFLOPS})',
        '"min_speedup_required": min_required})',
        "第三条比对路退回本片之前: 它其实判过, 而在跨轮读数里是「这一轮没报尺」⇒ "
        "分母悄悄少一轮, 报警的那一行永远说 0", True),

    Row("P4 三条路共用同一个判据字符串",
        "ver",
        '"perf_criterion": PERF_CRITERION_ABSOLUTE_GFLOPS})',
        '"perf_criterion": PERF_CRITERION_SPEEDUP})',
        "换量纲的漂移被抹平: 两条路的尺读数变成逐键相同 ⇒ 「两路可分」那条用例与跨轮"
        "的 `ruler_changed` 一起红", True),

    Row("P5 跨轮判据的默认键集合换成性能那一半",
        "ms",
        "def cross_round_ruler(iteration_history: list, *,\n"
        "                      keys: tuple = GATE_PARAM_KEYS,\n"
        "                      lineage_key: str = LINEAGE_RULER_KEY) -> dict:",
        "def cross_round_ruler(iteration_history: list, *,\n"
        "                      keys: tuple = PERF_GATE_PARAM_KEYS,\n"
        "                      lineage_key: str = LINEAGE_RULER_KEY) -> dict:",
        "既有读者 (conductor 每轮那句 + 报表 L2 那一栏) 全都读不到 atol/rtol/ρ ⇒ "
        "「默认参数就是 correctness 那半」那条用例红 —— 收成一份实现不等于可以改默认", True),

    Row("P6 血缘里两半共用一个键名",
        "ms", 'LINEAGE_PERF_RULER_KEY = "l3_ruler"',
        'LINEAGE_PERF_RULER_KEY = "l2_ruler"',
        "性能那半的读数盖掉 correctness 那半 (后写的赢) ⇒ L2 那一栏从今天起说的是性能的事, "
        "而它的标题还写着 atol/rtol", True),

    Row("P7 conductor 不写性能那一格",
        "cond",
        "                LINEAGE_PERF_RULER_KEY: _round_perf_ruler(verify_results),",
        "",
        "血缘里没有原料 ⇒ 报表那一栏退成「测不到」而不是「查过了没漂移」; 这是 #40/#61 "
        "那一族 (写了没人读) 的反面: 算了却没落盘", True),

    Row("P8 报表只印 L2 那一栏",
        "hpr",
        '            ("各轮 L3 报的 min_speedup/判的是哪个量/绑定轴 是不是同一把", "perf_ruler",',
        '            ("各轮 L3 报的 min_speedup/判的是哪个量/绑定轴 是不是同一把", "___removed___",',
        "第 3 片新写的那些格子没有读者 ⇒ 与本片的立片理由正面对撞 ("
        "「算了没人读」清单那一族)", True),

    Row("P9 `ruler_note` 忽略标签",
        "ms", '    return (f"{label}: " if label else "") + head + "; " + counts',
        '    return head + "; " + counts',
        "日志里出现两行逐字相同的「各轮的尺…」而读者分不清哪行说的是哪一把 "
        "(本仓「同一句话有多个副本」的反面: 两个副本各说不同的事却长得一样)", True),

    Row("P10 让判决去读新加的名字 (本片明说没做的事)",
        "ver",
        "        passed = target_gflops <= 0 or actual_gflops >= target_gflops * min_required",
        '        passed = (target_gflops <= 0 or actual_gflops >= target_gflops * min_required)\n'
        '        passed = passed and metrics.get("perf_criterion") != ""',
        "隔离理由: 上面两行刚把 `perf_criterion` 填成非空常量, 所以这个 `and` 恒真 ⇒ "
        "今天没有输入能让判决改变, 行为层与原文不可区分。杀它的是「判决位置不许出现 "
        "perf_criterion」那条 AST 守卫 —— 它钉的是**这片给的名字只能被观测读**, 换判决"
        "那一步要带真机前后对照 (与 #80 第 2 片的 S10 同一条姿势)", False),

    Row("P11 性能那半自己抄一份选择逻辑, 不走共享实现",
        "ms",
        "    return _ruler_reading(l3_metrics, PERF_GATE_PARAM_KEYS,\n"
        "                          not_a_value={BOUND_AXIS_KEY: (BOUND_AXIS_UNKNOWN,)})",
        "    m = l3_metrics if isinstance(l3_metrics, dict) else {}\n"
        "    vals = {k: m[k] for k in PERF_GATE_PARAM_KEYS\n"
        "            if k in m and m[k] != BOUND_AXIS_UNKNOWN}\n"
        "    return {\"state\": RULER_REPORTED, **vals} if vals else {\"state\": RULER_UNREPORTED}",
        "隔离理由: 这份抄件对同一份输入给出**同一个 dict** (连 unknown 那条都照抄了) ⇒ "
        "无输入可观察差异。拦住它的是那条「两个壳都必须调用 `_ruler_reading`、自己不许长出 "
        "isinstance 挑选」的委托守卫 —— 本片的立片理由就是「判据只有一份」, 而一份判据"
        "被抄成两份时, 行为可以相同、口径已经开始漂 (I22/H4 那一族)", False),

    Row("P12 判据字符串就地拼在 verifier 里 (不用常量)",
        "ver",
        '"perf_criterion": PERF_CRITERION_ABSOLUTE_GFLOPS})',
        '"perf_criterion": "gflops>=target_gflops*min_speedup"})',
        "隔离理由: 值逐字相同 ⇒ 行为与产物今天都不可区分。杀它的是那条 AST 守卫"
        "(每个判据都必须是 `PERF_CRITERION_*` 常量) 与那条「verifier 里不许出现自己拼的"
        "判据字符串」的计数断言 —— 名字一旦被抄第二遍, 两处就会开始漂", False),
]
