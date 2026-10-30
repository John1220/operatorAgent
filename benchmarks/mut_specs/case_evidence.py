"""「§6.2 条款 2/3 的证据生产者」那张表的可执行形态

靶子: `tests/unit/test_case_evidence.py`（case 身份 + 声明进契约 + 两条判据的三态 + 真跑一遍）、
`tests/unit/test_implementation_decision.py`（调用点不许自己写证据值）、
`tests/unit/test_report_deliverable.py`（报告里那两格的三态）。

为什么单独一张表: 这一片的后果是**部署建议**（`use_custom` 第一次可达），而它的两条证据
是「本次评过哪条 case」与「这条 case 上的正确性/回退」。风险全在**三态被抹平**上:
没声明被读成「都通过」（vacuous truth）、没评过被读成「通过」、没量到被读成「没越限」、
读不到的形状被读成 0 —— M1/M2/M3/M4/M5 就是这五个入口; M6 是判据方向; M7/M8 是
「声明没进 hash / 没进契约」（那会让判据变成可改的常量）; M9 是「算出来了但没发布」。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/case_evidence.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "ct": "operator_agent/contract.py",
    "el": "operator_agent/eligibility.py",
    "cu": "operator_agent/conductor.py",
}

TARGETS = ["tests/unit/test_case_evidence.py",
           "tests/unit/test_implementation_decision.py",
           "tests/unit/test_report_deliverable.py"]

ROWS = [
    Row("M1 读不到形状被写成 0（把不确定当确定）", "ct",
        'if shape else "?"',
        'if shape else "0"',
        "**具体输入**: `case_key({'A': [1, 2]})`（没有 `.shape`）⇒ 变异后得到 `A=0`"
        " ⇒ `test_a_shape_that_cannot_be_read_is_a_question_mark` 红（它还断言了"
        " `A=?` 与 `A=0` **不是**同一个 key）。这一位一旦被填成数字, 判据面就把"
        "「没读到」读成了「读到 0」"),
    Row("M2 没声明 required_cases 被当成「都通过」（vacuous truth）", "el",
        '        return None, ("契约没有声明 required_cases（老契约 / 任务没声明）"',
        '        return True, ("契约没有声明 required_cases（老契约 / 任务没声明）"',
        "**具体输入**: `required_cases_ok((), {case: True})` ⇒ 变异后落 `True` ⇒"
        " `test_no_declaration_is_not_vacuous_truth` 与 E 组"
        " `test_an_undeclared_task_leaves_both_unestablished` 红。这一条一旦反转,"
        "「没声明」就比「声明了且都过」更容易拿到 `use_custom`"),
    Row("M3 契约要求的 case 没评过被当成通过", "el",
        '        return None, (f"契约要求的 case {sorted(missing)} 本次没有评过"',
        '        return True, (f"契约要求的 case {sorted(missing)} 本次没有评过"',
        '**具体输入**: `required_cases_ok((case,), {"B=1x1": True})` ⇒ 变异后落 `True`'
        " ⇒ `test_a_declared_case_that_was_never_evaluated_is_unestablished` 红。"
        "「没评过」被读成「通过」正是 §6.2 条款 2 要挡的那件事的镜像"),
    Row("M4 没量到的 case 被当成「没越限」", "el",
        '        return None, (f"契约给了 case {unmeasured} 的回退上限, 但本次没有它们的加速比"',
        '        return False, (f"契约给了 case {unmeasured} 的回退上限, 但本次没有它们的加速比"',
        '**具体输入**: `regression_over_limit(((case, 0.05),), {"B=1x1": 0.9})` ⇒ 变异后落'
        " `False`（= 量过、没越限）⇒ `test_an_unmeasured_case_is_not_a_pass` 红"),
    Row("M5 没声明上限被当成「都没越限」", "el",
        '        return None, ("契约没有声明 max_regression',
        '        return False, ("契约没有声明 max_regression',
        "**具体输入**: `regression_over_limit((), {case: 0.5})` ⇒ 变异后落 `False` ⇒"
        " `test_no_limit_is_not_no_regression` 与 E 组的未声明那一条红"),
    Row("M6 回退判据方向反了", "el",
        '            if (1.0 - measured[c]) > limits[c] + REGRESSION_SLACK]',
        '            if (1.0 - measured[c]) < limits[c] + REGRESSION_SLACK]',
        "**具体输入**: `speedup=0.8`（回退 0.2）、上限 0.05 ⇒ 变异后**不**越限;"
        " 而 `speedup=1.3` 反而被判越限 ⇒ `test_a_regression_over_the_limit_is_true` /"
        " `test_a_regression_at_the_limit_is_not_over_it` / 负上限那一条红。方向一错,"
        "「越限 ⇒ 不得晋级」（§6.2 条款 3）就变成了「越限 ⇒ 晋级」"),
    Row("M7 声明没有进契约 hash（条款 4 反例）", "ct",
        '        if self.required_cases:',
        '        if self.required_cases and False:',
        "**具体输入**: 声明了 `required_cases` 的契约 ⇒ 变异后 hash 与没声明的那份**相同**"
        " ⇒ `test_a_required_case_changes_the_hash` 红（同族还有 max_regression 那条）。"
        "契约哈希的第一条要求是「改一个字必须变哈希」—— 判据成了可改的常量"),
    Row("M8 任务侧声明没有透传进契约", "ct",
        '        required_cases=tuple(getattr(cons, "required_cases", ()) or ()),',
        '        required_cases=(),',
        "**具体输入**: 声明了必判 case 的任务走 `freeze_from_task` ⇒ 变异后契约里是空"
        " ⇒ `test_a_declared_task_reaches_the_frozen_contract` 与 E 组的三条红。"
        "这正是「声明渠道断了」的样子: 调用方声明了, 判据面看不见"),
    Row("M9 评测侧不发布「本次评了哪条 case」", "cu",
        '                artifact.benchmark_data["case_key"] = case_key(test_inputs)',
        '                pass',
        "**具体输入**: 真跑一次 `Conductor.execute`（任务声明了必判 case）⇒ 变异后"
        " `metrics` 里没有这一格 ⇒ E 组 `test_the_run_publishes_the_case_it_evaluated`"
        " 与 `test_a_declared_case_reaches_use_custom_and_stops_at_corpus_admission` 红"
        "（两条证据退回恒 `None`）"),
    Row("M10 证据值不走 L2 判定（调用点自己写 True）", "cu",
        '            evaluated = {case: self._l2_verdict(artifact)} if case else {}',
        '            evaluated = {case: True} if case else {}',
        "**具体输入**: 今天**行为不可区分**: 裁决只在 `all_passed` 那一支写（L2 必然通过）"
        " ⇒ 只有结构守卫杀得掉 —— `tests/unit/test_case_evidence.py::TestOneProducerPerFact`"
        " 钉住「这条 case 的正确性必须由 `_l2_verdict` 这一处读」（§4.4 的"
        " `correctness_verified` 与条款 2 读的是同一份 L2 结果）",
        behavioral=False),
]
