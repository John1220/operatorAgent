"""「§16 收口: 目标函数进门槛」那张表的可执行形态（切片 J / CHECKLOG I95）

靶子: `tests/unit/test_performance_verdict.py`（纯判据 / 真过门 / 三态与来源 / 报告那一行 / 结构）。

为什么单独一张表: 这一片把「性能面的成功标准」从**一行比较**变成一个判决，最容易写歪的正是
判决与门之间的那一层 —— 声明了目标函数却把「未确立」放行（M1）、把「没声明」也当成声明
（M2，等于把所有老调用方一律打死）、退回单形状却不登记（M3）、目标函数成立还回头要求主 case
（M4，§16 的反模式回来）、条款 3 被聚合盖过（M5）、判决不影响门（M6）、算了却不发布（M7）、
账本还没发布就判（M8 目标函数必然未确立）、契约冻不住就静默当「没声明」（M9 把 fail-closed
换成放水）、判据读任务原始字段而不是冻结契约（M10）、报告把单形状那条印成目标函数（M11）、
老产物没有那一格也照印（M12）、「量过但没得出」被当成 0（M13）。

**一条结构行**: 本机那条出口把「主 case 达标」又接回来（S1）—— 今天的测试全走远端那条出口
（确定性载荷），所以它在**行为上不可区分**，只有结构守卫 `test_the_primary_speedup_is_no_longer_
the_success_criterion` 杀得掉（§16 的那一行不许回来）。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/performance_verdict.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {
    "el": "operator_agent/eligibility.py",
    "ve": "operator_agent/verifier.py",
    "rp": "operator_agent/report.py",
}

TARGETS = ["tests/unit/test_performance_verdict.py"]

ROWS = [
    Row("M1 声明了目标函数, 未确立也放行", "el",
        '    if value is None:\n        return (False, PERF_BASIS_OBJECTIVE,',
        '    if value is None:\n        return (True, PERF_BASIS_OBJECTIVE,',
        "**具体输入**: 只声明权重、或权重指向的 case 没量到 ⇒ 变异后**通过** ⇒"
        " `test_weights_alone_are_a_declaration` / `test_an_unestablished_objective_fails_closed`"
        " / 门那几条全红。「声明了就是承诺按它评」是这一片的核心一条"),
    Row("M2 「没声明」被当成「声明了目标函数」", "el",
        '    declared = bool(case_weights) or bool(aggregate)',
        '    declared = True',
        "**具体输入**: 一份**没声明**任何目标函数的契约 ⇒ 变异后走目标函数那条路、而目标"
        " 函数必然未确立 ⇒ **一律不通过** = 把所有老调用方一次性打死（自锁）⇒"
        " `test_no_declaration_is_the_primary_criterion` 与门那条红"),
    Row("M3 退回单形状时不登记欠账", "el",
        '                   f"—— ⚠️ §16 未收口: 这一次的成功标准是**单形状**的")',
        '                   f"")',
        "**具体输入**: 一份没声明目标函数的契约 ⇒ 变异后产物与日志都不再说「这是单形状标准」"
        " ⇒ `test_no_declaration_is_the_primary_criterion`（`§16 未收口`）与门那条"
        " （`assertIn(\"§16 未收口\", out)`）红"),
    Row("M4 目标函数成立也回头要求主 case 达标", "el",
        '    passed = value >= float(min_speedup)',
        '    passed = (value >= float(min_speedup)\n              and float(primary_speedup or 0.0) >= float(min_speedup))',
        "**具体输入**: 聚合 3.0 够门槛、主 case 只有 0.5x ⇒ 变异后**不通过** ⇒"
        " `test_the_objective_does_not_also_require_the_primary_case` 红 —— §16 点的正是"
        " 「拿单形状当成功标准」，聚合够还得主 case 也够就是把那一条接了回来"),
    Row("M5 条款 3 被聚合盖过", "el",
        '    if regression_over_limit_value is True:',
        '    if False:',
        "**具体输入**: 聚合成立、但有一条 case 回退越限 ⇒ 变异后通过 ⇒"
        " `test_a_regression_over_limit_beats_the_aggregate`（纯函数）与门那条"
        " `test_a_regression_over_the_declared_limit_fails_the_gate` 红 —— §6.2 条款 3 说"
        " 逐 shape 硬回退**优先于**聚合"),
    Row("M6 判决不影响门", "ve",
        '        return bool(prior) and bool(passed)',
        '        return bool(prior)',
        "**具体输入**: 主 case 2.0x 够门槛、权重压在 0.5x 那条上（聚合 0.53）⇒ 变异后 **L3"
        " 仍然通过** ⇒ `test_the_objective_is_the_gate_when_declared` 与"
        " `test_an_objective_that_cannot_be_established_fails_the_gate` 红 —— 判决算了但没接上"),
    Row("M7 判决算了却不发布", "ve",
        '        metrics["performance_verdict"] = {',
        '        metrics["performance_verdict_off"] = {',
        "**具体输入**: 任何一次 L3 ⇒ 变异后产物里没有那一格 ⇒ 门那几条与报告那几条"
        " （读 `metrics[\"performance_verdict\"]`）全红。「算了但不写进产物」与「没算」在"
        " 产物上分不开"),
    Row("M8 账本还没发布就判", "ve",
        '                _primary = self._publish_case_ledger(metrics, kernel_code, test_inputs, task)\n                passed = self._apply_performance_verdict(passed, metrics, task, speedup,\n                                                         primary_key=_primary)',
        '                passed = self._apply_performance_verdict(passed, metrics, task, speedup)\n                self._publish_case_ledger(metrics, kernel_code, test_inputs, task)',
        "**具体输入**: 两组输入、声明了目标函数 ⇒ 变异后判决读到的是**空账** ⇒ 目标函数必然"
        " 未确立 ⇒ `test_a_distribution_beats_the_slow_case_when_the_weights_say_so` 与"
        " `test_the_objective_is_the_gate_when_declared`（值必须等于手算值）红"),
    Row("M9 契约冻不住就静默退回「没声明」", "ve",
        '            cons = getattr(task, "constraints", None)\n            weights = tuple(getattr(cons, "case_weights", ()) or ())\n            aggregate = str(getattr(cons, "aggregate", "") or "")\n            limits = tuple(getattr(cons, "max_regression", ()) or ())',
        '            weights, aggregate, limits = (), "", ()',
        "**具体输入**: 权重指向一条没在 `cases` 里声明的 case（I94 的 lint 拒绝冻结）、任务"
        " 里明说了目标函数 ⇒ 变异后判据变成「没声明」⇒ 退回单形状 ⇒"
        " `test_an_unfreezable_contract_does_not_silently_weaken_the_criterion` 红。"
        " 冻不住时把「声明过」读成「没声明」= 把 fail-closed 换成放水"),
    Row("M10 判据读任务原始字段而不是冻结契约", "ve",
        '        if contract is not None:\n            weights, aggregate = tuple(contract.case_weights), str(contract.aggregate or "")',
        '        if False:\n            weights, aggregate = tuple(contract.case_weights), str(contract.aggregate or "")',
        "**具体输入**: 正常一份契约 ⇒ 变异后 `contract_frozen` 变成 False（判据面其实读的是"
        " 任务字段）⇒ `test_the_verdict_names_its_criteria_source` 红。「判据是谁的」这件事"
        " 必须由产物说出来"),
    Row("M11 报告把单形状那条印成目标函数", "rp",
        '    if v.get("basis") == "objective":',
        '    if True:',
        "**具体输入**: `basis=primary_case` 的那一格 ⇒ 变异后报告印成「**目标函数**」⇒"
        " `test_the_primary_case_basis_prints_the_open_debt` 红。产物里最容易被读者信任的"
        " 就是这一行 —— 印反了等于把「单形状标准」藏起来"),
    Row("M12 老产物没有那一格也照印", "rp",
        '    if not isinstance(v, Mapping):\n        return []',
        '    if not isinstance(v, Mapping):\n        v = {}',
        "**具体输入**: 一份没有 `performance_verdict` 的老产物 ⇒ 变异后印出「**主 case**"
        " 加速比 ⇒ 未确立」⇒ `test_an_old_payload_without_the_key_prints_nothing` 红。"
        " 「没这一格」被读成「按主 case 判的」正是三态纪律要挡的"),
    Row("M13 「量过但没得出」被当成 0", "el",
        '            out[str(case)] = None if value is None else value',
        '            out[str(case)] = 0.0 if value is None else value',
        "**具体输入**: 账本里一条 `speedup_vs_baseline: None`（量了但没得出数）⇒ 变异后变成"
        " 0.0 ⇒ `test_a_ledger_entry_without_a_number_stays_none` 红。0 是**一个数**，"
        " 会让判据面把它当成「量到了、就是 0x」"),
    Row("S1 本机那条出口把「主 case 达标」又接回来", "ve",
        '            passed = True\n            base_ms = baseline["baseline_time_ms"]',
        '            passed = speedup >= min_required\n            base_ms = baseline["baseline_time_ms"]',
        "**具体输入**: 今天**行为不可区分**（测试全走远端那条出口: 确定性载荷经"
        " `benchmark_with_baselines`），只有结构守卫"
        " `test_the_primary_speedup_is_no_longer_the_success_criterion` 杀得掉 ——"
        " 它钉住的正是 §16 那一行不许回来",
        behavioral=False),
]
