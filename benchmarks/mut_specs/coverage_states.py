"""「§21.4 的三态不许互相借名: 「没有量具」与「没有对象」必须分开记、且覆盖缺口有人读」那张表的**可执行**形态

靶子: `tests/unit/test_agent_runtime.py`（capability 源码半边的状态 / 没有对象不进覆盖表）、
`tests/unit/test_agent_runtime_wiring.py`（真跑一轮之后血缘里的覆盖表）、
`tests/unit/test_report_deliverable.py`（覆盖缺口的**人类可读下游**）。

为什么单独一张表: 这一片修的是「同一份数据被记成两件事」——
`not_applicable`（这一份候选没有这个对象）与 `unknown`（有对象、没量具）此前并进同一个
`uncovered_rules`，于是「这一轮不适用」被读成「这一层查不了」；而源码级「幻觉 API」的
真实状态（有对象、没量具）又被记成「没有对象」。两类失效方式不同：前者是**集合并错**，
后者是**状态写错**，第三类是**没人读**（数据算出来没有下游）。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/coverage_states.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "ar": "operator_agent/agent_runtime.py",
    "rp": "operator_agent/report.py",
}

TARGETS = ["tests/unit/test_agent_runtime.py", "tests/unit/test_agent_runtime_wiring.py",
           "tests/unit/test_report_deliverable.py"]

ROWS = [
    Row("M1 源码级「幻觉 API」又被记成「没有对象」", "ar",
        '    return CheckOutcome(\n'
        '        "capability", STATE_UNKNOWN,\n'
        '        "源码级「幻觉 API」没有判据: 仓里没有任何语言的 API 面清单 "\n'
        '        "(没有 Triton 白名单/符号表) ⇒ 这一半判不了 (不是「没有幻觉 API」)")',
        '    return CheckOutcome(\n'
        '        "capability", STATE_NOT_APPLICABLE,\n'
        '        "源码级「幻觉 API」没有判据: 仓里没有任何语言的 API 面清单 "\n'
        '        "(没有 Triton 白名单/符号表) ⇒ 这一半判不了 (不是「没有幻觉 API」)")',
        "**具体输入**: 一份带 patch 的 `propose_source_patch` 走一次 `precompile_checks` "
        "⇒ 状态回到 `not_applicable` ⇒ "
        "`test_the_source_level_hallucinated_api_half_is_reported_uncovered` 红。"
        "对象**在**（源码里有 API 调用），判不了的是量具 —— 记成「没有对象」等于把"
        "「尺子没有」说成「没有可量的东西」"),
    Row("M2 「没有对象」又并回覆盖表", "ar",
        "        uncovered = {o.rule for o in self.outcomes if o.state == STATE_UNKNOWN}",
        "        uncovered = {o.rule for o in self.outcomes "
        "if o.state in (STATE_NOT_APPLICABLE, STATE_UNKNOWN)}",
        "**具体输入**: 首轮候选（没有父代 ⇒ `new_dependency` / `duplicate_of_parent` "
        "没有对象）⇒ 两条又出现在 `uncovered_rules` 里 ⇒ "
        "`test_a_rule_without_an_object_is_not_a_coverage_gap` 与真跑那一条"
        "（`test_every_round_records_the_action_the_verdict_and_the_context`，"
        "断言 `uncovered_rules == ['capability']`）红。这正是本片之前的形状"),
    Row("M3 覆盖缺口整个丢掉 (unknown 不进 uncovered)", "ar",
        "        uncovered = {o.rule for o in self.outcomes if o.state == STATE_UNKNOWN}",
        "        uncovered = set()",
        "**具体输入**: 同 M1 的一次 `precompile_checks` ⇒ `uncovered_rules` 空 ⇒ "
        "`test_the_source_level_hallucinated_api_half_is_reported_uncovered` 红, 且报告那"
        "两条（`test_an_uncovered_rule_becomes_an_item_with_its_detail` / 真跑那一条）红。"
        "「这一层有一条判据永远查不了」从产物里消失 = fail-open"),
    Row("M4 报告不再读覆盖缺口", "rp",
        "    gaps = _uncovered_rules_from_lineage(artifact)",
        "    gaps = []",
        "**具体输入**: 带 `agent_action.precompile.coverage.uncovered_rules=['capability']` "
        "的产物渲染一次报告 ⇒ 「我无法确立的事」少一条 ⇒ "
        "`test_an_uncovered_rule_becomes_an_item_with_its_detail` 与真跑那一条红。"
        "覆盖数据在血缘里留了痕却没人读 —— 就是「有生产者、零消费者」那一族 (I82/I89)"),
    Row("M5 读到的不是最后一轮裁决", "rp",
        "    for h in reversed(_rounds(artifact)):",
        "    for h in _rounds(artifact):",
        "**具体输入**: 两轮血缘（第 1 轮有缺口、第 2 轮没有）⇒ 从头读 ⇒ 把**已经补上**的"
        "缺口又印一条 ⇒ `test_only_the_last_adjudicated_round_is_read` 红。"
        "这一节一旦开始复读就会被跳过不读"),
    Row("M6 没判过被读成「有一条缺口」", "rp",
        "        if not isinstance(cov, dict):\n"
        "            continue",
        "        if not isinstance(cov, dict):\n"
        '            return [("capability", "")]',
        "**具体输入**: 没有 `agent_action` 的老产物（= 没判过）⇒ 报告里凭空多一条"
        "`capability` 缺口 ⇒ `test_a_run_without_a_verdict_says_nothing` 红。"
        "三态的反面：缺证据被当成了证据（「没判过」与「没有缺口」在产物里长得一样）"),
]
