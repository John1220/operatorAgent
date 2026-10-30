"""「§6.2 的目标函数与逐 case 账本」那张表的可执行形态

靶子: `tests/unit/test_case_ledger.py`（契约权重/口径 + 聚合三态 + 账本 + 真跑一遍 + 结构）。

为什么单独一张表: 这一片让**目标函数**第一次成为一个可查的数，而它最容易写歪的地方是
**默认值**与**分母**: 没声明权重被读成「等权」、没声明口径被读成「加权几何平均」、
没量到的 case 被读成 1.0x、非正加速比被放过、权重零和却照样给数 —— 五条入口都会**替人
写**一个目标函数（M1~M5）; M6 是最差那条方向反了（总体好看掩盖最差）; M7/M8 是「权重/
口径没进 hash」（判据变成可改的常量）; M9 是声明渠道断了; M10 是主 case 不在账本里（它
会被读成「没量到」）; M11 是绕过共用的那对调用（协议一漂就比出两种数，今天行为不可区分）。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/case_ledger.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {
    "ct": "operator_agent/contract.py",
    "el": "operator_agent/eligibility.py",
    "ve": "operator_agent/verifier.py",
}

TARGETS = ["tests/unit/test_case_ledger.py"]

ROWS = [
    Row("M1 没声明权重被读成「等权」", "el",
        '        return None, ("契约没有声明 case_weights ⇒ 目标函数没有权重 ⇒ 未确立",)',
        '        return 1.0, ("契约没有声明 case_weights ⇒ 目标函数没有权重 ⇒ 未确立",)',
        "**具体输入**: `aggregate_speedup((), {case: 4.0}, \"mean\")` ⇒ 变异后给出一个数"
        " ⇒ `test_no_weights_is_not_equal_weights` 红。没声明就是没判据: 替人补一个等权"
        " 等于替人写了目标函数"),
    Row("M2 没声明口径被读成「加权几何平均」", "el",
        '        return None, ("契约没有声明 aggregate（聚合口径）⇒ 未确立",)',
        '        return 1.0, ("契约没有声明 aggregate（聚合口径）⇒ 未确立",)',
        "**具体输入**: `aggregate_speedup(W, S, \"\")` ⇒ 变异后给出一个数 ⇒"
        " `test_no_aggregate_is_not_the_default` 红"),
    Row("M3 没量到的带权 case 被当成 1.0x", "el",
        '    for case in weights:\n'
        '        try:\n'
        '            value = speedups.get(case)\n'
        '            measured[case] = None if value is None else float(value)',
        '    for case in weights:\n'
        '        try:\n'
        '            value = speedups.get(case)\n'
        '            measured[case] = 1.0 if value is None else float(value)',
        "**具体输入**: 两条 case 只量到一条 ⇒ 变异后分母「补齐」成 1.0x 并给出一个数 ⇒"
        " `test_a_weighted_case_that_was_not_measured_is_unestablished` 红。少算了最差的"
        " 那条却照样报一个好看的数, 正是目标函数最坏的失效"),
    Row("M4 非正加速比被放过（几何平均给出 0）", "el",
        '        return None, (f"几何平均要求每条加速比 > 0, 但 {bad} 不是 ⇒ 未确立",)',
        '        return 0.0, (f"几何平均要求每条加速比 > 0, 但 {bad} 不是 ⇒ 未确立",)',
        "**具体输入**: `{case: 0.0}` 进几何平均 ⇒ 变异后返回 0.0 ⇒"
        " `test_a_non_positive_speedup_blocks_only_the_geomean` 红（同一变异顺带证明"
        " mean/worst_case 那两条不受影响）"),
    Row("M5 权重零和却照样给数", "el",
        '        return None, (f"case_weights 的权重之和是 {total:g} ⇒ 聚合没有分母 ⇒ 未确立",)',
        '        return 0.0, (f"case_weights 的权重之和是 {total:g} ⇒ 聚合没有分母 ⇒ 未确立",)',
        "**具体输入**: `((case, 0.0),)` ⇒ 变异后返回 0.0（一个没有分母的数）⇒"
        " `test_a_zero_total_weight_is_unestablished` 红 —— 契约层挡了一手, 纯函数自己"
        " 也要失败关闭"),
    Row("M6 worst_case 取了最好的一条", "el",
        '        worst = min(measured[c] for c in weights)',
        '        worst = max(measured[c] for c in weights)',
        "**具体输入**: `{A: 9.0, B: 0.5}` ⇒ 变异后报 9.0 ⇒"
        " `test_the_worst_case_reads_the_worst_one` 红。设计里 `max_regret` 要挡的正是"
        " 「总体好看掩盖最差那条」, 方向一反这个口径就变成它的反面"),
    Row("M7 权重没有进契约 hash（条款 4 反例）", "ct",
        '        if self.case_weights:',
        '        if self.case_weights and False:',
        "**具体输入**: 一份声明了权重的契约 ⇒ 变异后 hash 与没声明的那份**相同** ⇒"
        " `test_a_weight_changes_the_hash` 红（同族还有顺序规范化那条）。权重能改 ="
        " 目标函数能改"),
    Row("M8 聚合口径没有进契约 hash", "ct",
        '        if self.aggregate:',
        '        if self.aggregate and False:',
        "**具体输入**: 同一套权重、两个口径 ⇒ 变异后两份契约同 hash ⇒"
        " `test_the_aggregate_changes_the_hash` 红"),
    Row("M9 任务侧权重没透传进契约", "ct",
        '        case_weights=_weight_pairs(getattr(cons, "case_weights", None)),',
        '        case_weights=(),',
        "**具体输入**: 声明了权重的任务走 `freeze_from_task` ⇒ 变异后契约里是空"
        " ⇒ `test_a_declared_task_reaches_the_frozen_contract` 红。「声明渠道断了」的样子"),
    Row("M10 主 case 不在账本里（它会被读成「没量到」）", "ve",
        '        if first:',
        '        if first and False:',
        "**具体输入**: 传两组输入 ⇒ 变异后账本只剩另一条 case, 主 case 那条读数无处可查"
        " ⇒ `test_the_primary_case_is_not_re_measured` /"
        " `test_multiple_input_sets_publish_one_entry_per_case` 红。判据面会把「评过的」"
        " 读成「没评过」"),
    Row("M11 逐 case 测量绕过共用的那对调用", "ve",
        '            bench, remote = self._bench_case(kernel_code, inputs, task)',
        '            bench, remote = self.backend.benchmark(kernel_code, inputs, warmup=2, repeat=5), None',
        "**具体输入**: 今天**行为不可区分**（MockGPU 不读 warmup/repeat, 两条路给同一个数）"
        " ⇒ 只有结构守卫杀得掉 —— `test_the_ledger_has_one_producer_and_two_declared_exits`"
        " 钉住 `self._bench_case(` 恰好两处（主 case 与账本共用一对调用）",
        behavioral=False),
]
