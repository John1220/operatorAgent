"""「§17.1 实现裁决: 三条门槛只许由**确立的**证据打开」那张表的可执行形态

靶子: `tests/test_implementation_decision.py`（阶梯闭性旁举 + 逐门）、
`tests/test_kernel_db_persistence.py`（生产路径必须发布这一格）、
`tests/test_report_deliverable.py`（真跑一遍以后报告里要有它）。

为什么单独一张表: 这一格的后果是**部署决定**, 而它的三条证据今天都还没有生产者
（契约里没有 `required_cases` / `max_regression`）—— 于是这一片最大的风险不是「算错」,
而是「把**未确立**当成**好**」：`None` 一旦被读成 True/False, 一个没有量具的字段就会
自己长出结论来。M1/M2/M3/M7 就是这一类（fail-open 的四个入口）, M4 是读法反面
（没判成就建议用库 —— 设计点名的失效模式）, M5 是门槛抬错, M6 是「算出来没人用」。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/implementation_decision.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {
    "el": "operator_agent/eligibility.py",
    "cu": "operator_agent/conductor.py",
    "rp": "operator_agent/report.py",
}

TARGETS = ["tests/test_implementation_decision.py",
           "tests/test_kernel_db_persistence.py",
           "tests/test_report_deliverable.py"]

ROWS = [
    Row("M1 未确立的 required_cases 被当成「全对」", "el",
        '    if required_cases_ok is not True:',
        '    if required_cases_ok is False:',
        "**具体输入**: 一份 validated 的 claim + `required_cases_ok=None`（未确立）+ 没有回退越限"
        " ⇒ 变异后落 `use_custom` ⇒ `test_an_unestablished_required_cases_blocks_it` 与旁举那条"
        "（4x7x3x3 组）红。这正是「缺证据不是证据」的反面：`None` 被读成「全对」"),
    Row("M2 未确立的回退被当成「没有回退」", "el",
        '    if regression_over_limit is not False:',
        '    if regression_over_limit is True:',
        "**具体输入**: validated + `required_cases_ok=True` + `regression_over_limit=None`"
        " ⇒ 变异后落 `use_custom` ⇒ `test_an_unestablished_regression_blocks_it` 红。"
        "「没有量具去量回退」与「量过、没有回退」是两件事（§6.2 条款 3 的判据面）"),
    Row("M3 没有资格位被当成「能报」", "el",
        '    if can_report is not True:',
        '    if can_report is False:',
        "**具体输入**: `eligibility=None`（没经过收口点）, 或掩码缺一位被 `from_mapping` 归成"
        " `None` ⇒ `None is False` 为假 ⇒ 门 0 放行 ⇒ `test_no_eligibility_is_inconclusive` /"
        " `test_a_short_mask_is_not_filled_with_defaults` / 旁举那条红（§17.1 门 0 的意义）"),
    Row("M4 没判成时给出「建议用库」（读法反面）", "el",
        '    return _record(DECISION_INCONCLUSIVE,\n'
        '                   ["「自定义」的三条证据没有一条被否定、也没有一条被确立 ⇒ "\n'
        '                    "既上不了 use_custom, 也没有「建议用库」的依据 (门 3)"])',
        '    return _record(DECISION_USE_LIBRARY,\n'
        '                   ["「自定义」的三条证据没有一条被否定、也没有一条被确立 ⇒ "\n'
        '                    "既上不了 use_custom, 也没有「建议用库」的依据 (门 3)"])',
        "**具体输入**: validated + 两条证据都是 `None`（今天生产路径的真实情形）"
        " ⇒ 变异后把「契约里没有 `required_cases`」读成「建议你别上自定义核」⇒"
        " `test_missing_evidence_is_not_a_library_recommendation` 与旁举那条红。"
        "设计 §17.1 那一句两侧都在防这件事（「因为一次噪声读数就建议你别上自定义核」）"),
    Row("M5 等级门槛抬到 promotable", "el",
        '    if not at_least(level, CLAIM_VALIDATED):',
        '    if not at_least(level, CLAIM_PROMOTABLE):',
        "**具体输入**: validated + 两条证据全中 ⇒ 变异后被当成「等级不够」⇒"
        " `test_the_happy_path`（与 promotable/persisted 那一条）红 —— §17.1 的原句是"
        " `≥ VALIDATED`, 不是 `≥ PROMOTABLE`"),
    Row("M6 裁决没有发布进产物", "cu",
        '        metrics["implementation_decision"] = implementation_decision(\n'
        '            eligibility=metrics.get("eligibility"), claim_record=claim).as_dict()',
        "        pass",
        "**具体输入**: 真调一次 `Conductor._save_to_db`（可信读数 + 独立重测）"
        " ⇒ `metrics` 里没有这一格 ⇒ `test_the_implementation_decision_is_published_next_to_the_claim`"
        " 与报告那条端到端红。「推导写好了但生产路径没接线」是本仓反复出现的那一族（I45/I48）"),
    Row("M7 缺键也印一个默认裁决", "rp",
        '    d = _get(m, "implementation_decision", None)\n'
        '    if not isinstance(d, dict) or not d.get("decision"):\n'
        "        return []",
        '    d = _get(m, "implementation_decision", None)\n'
        '    if not isinstance(d, dict) or not d.get("decision"):\n'
        '        return ["- 实现裁决 (§17.1): inconclusive (变异: 缺键也印默认)"]',
        "**具体输入**: 一份没有 `implementation_decision` 键的产物（老产物 / 没走 `_save_to_db`）"
        " ⇒ 报告里出现一条默认裁决 ⇒ `test_a_missing_key_means_no_line` 与"
        " `test_a_malformed_record_prints_nothing` 红。三态纪律：印一个默认值会让「没算」"
        "读成「算出来是它」"),
]
