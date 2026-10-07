"""「§4.4 的声明等级 + §22.3 的覆盖上限真的会影响 ClaimLevel 与 promotion」那张表的**可执行**形态

靶子: `tests/test_claim_level.py`（阶梯 / 上限 / 三态 / 同一个数 / 报告）与
`tests/test_kernel_db_persistence.py::TestGateWiring`（生产路径那一根线:
`Conductor._save_to_db` 写进去的上限真的决定了落盘）。

为什么单独一张表: 这一片修的是「**覆盖数据有生产者、零消费者**」——
判据算了、血缘里留了痕, 却没有一处后果。所以这张表里最要紧的不是阶梯算错,
而是**接线断了**: 上限不再压级 / 门不再读上限 / 生产路径不再写入。
前两类是判据本身错, 第三类是「判据与调用方脱钩」——两者失效方式不同。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/claim_level.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {
    "el": "operator_agent/eligibility.py",
    "or": "operator_agent/oracle.py",
    "cu": "operator_agent/conductor.py",
    "rp": "operator_agent/report.py",
    "cf": "operator_agent/confirm.py",
}

TARGETS = ["tests/test_claim_level.py", "tests/test_kernel_db_persistence.py"]

ROWS = [
    Row("M1 regex_only 不再压级 (文本规则也当看得穿源码)", "el",
        "    COVERAGE_REGEX: CLAIM_VALIDATED,      # 真实但更弱：运行内可比, 不足以永久",
        "    COVERAGE_REGEX: CLAIM_PERSISTED,      # 真实但更弱：运行内可比, 不足以永久",
        "**具体输入**: 一份 `dsl=cuda_c` 的产物 (源码是 C++, `classify` ⇒ regex_only) + "
        "一份通过的重测 ⇒ 上限从 validated 抬到 persisted ⇒ "
        "`test_a_capped_source_cannot_reach_promotable` 与 "
        "`test_a_regex_only_source_is_refused_even_with_a_confirmation` 红。"
        "这正是 §22.3 那句话的反面: 覆盖不足**必须**能拦住 promotion"),
    Row("M2 「判不了」当通过", "el",
        "    COVERAGE_NONE: CLAIM_PROVISIONAL,     # 判不了（无法确立 ≠ 通过）",
        "    COVERAGE_NONE: CLAIM_PERSISTED,       # 判不了（无法确立 ≠ 通过）",
        "**具体输入**: `coverage_mode=none` 的 metrics ⇒ 上限变成 persisted ⇒ "
        "`test_none_and_not_judged_cap_at_provisional` 红, 且 "
        "`test_the_gate_fires_before_the_remeasure_gate` 看到的是门 4 的理由而不是 §22.3。"
        "「无法确立」被读成「通过」是本仓记过最多次的那一类"),
    Row("M3 缺正确性凭据也能升 validated (三态破坏)", "el",
        '    elif tier != "valid" or correctness_verified is not True:',
        '    elif tier != "valid":',
        "**具体输入**: `eligibility=ALL_TRUE` + `validity=valid` + "
        "`correctness_verified=None`（没带 L2 凭据）⇒ 升到 validated ⇒ "
        "`test_can_tune_without_valid_or_correctness_is_provisional` 红。"
        "阶梯的每一级都要有对应证据 —— 缺证据不是有效证据"),
    Row("M4 入库门不再读覆盖上限", "or",
        "        if not at_least(ceiling, CLAIM_PROMOTABLE):",
        "        if False:",
        "**具体输入**: ceiling=validated 的 metrics + 一份通过的重测 ⇒ 入库 ⇒ "
        "`test_a_validated_ceiling_refuses` / "
        "`test_a_capped_source_is_refused_even_with_a_good_remeasure` 红, 而 "
        "`test_a_regex_only_source_is_refused_even_with_a_confirmation` 会看到盘上多了一条 "
        "—— 上限从判据退回成留痕。这一条就是「有生产者、零消费者」的原形 (I82)"),
    Row("M5 读不懂的等级比任何一级都高", "el",
        "    except ValueError:\n        return -1",
        "    except ValueError:\n        return len(CLAIM_LEVELS)",
        "**具体输入**: `ceiling=very-good` 的 metrics ⇒ 词表外的值反而排在最高 ⇒ "
        "门放行 ⇒ `test_an_unknown_ceiling_is_refused` 与 "
        "`test_an_unknown_level_is_not_higher_than_anything` 红。"
        "fail-closed 的反面是「读不懂 = 通行证」"),
    Row("M6 生产路径不再把等级写给判据", "cu",
        '        metrics["claim_level"] = claim.as_dict()',
        "        pass  # 判据那一格没人写",
        "**具体输入**: 同 M1 的 cuda_c 产物 ⇒ 入库门看不到 claim_level（按老产物不改判）⇒ "
        "盘上多出一条 regex_only 的永久结论 ⇒ "
        "`test_a_regex_only_source_is_refused_even_with_a_confirmation` 红; 结构守卫 "
        "`test_the_conductor_writes_the_claim_level_before_calling_the_gate` 也红（写入点"
        "落到了调用点之后）。判据算对了但没人喂它 = I82 那一族"),
    Row("M7 报告不再印声明等级", "rp",
        "    lines += _claim_level_lines(m)",
        "    pass  # 等级没人读",
        "**具体输入**: 带 `claim_level` 的 metrics 渲染一次报告 ⇒ 那两行不见了 ⇒ "
        "`test_the_report_prints_the_level_and_its_cap` 红。"
        "等级是**给人看**的后果之一: 产物里读不到就等于只在内存里存在"),
    Row("M8 「无法确立的事」不再收覆盖受限", "cf",
        '        if isinstance(_cl, dict) and _cl.get("ceiling") and not at_least(\n'
        '                _cl.get("ceiling"), CLAIM_PERSISTED):',
        '        if isinstance(_cl, dict) and False and not at_least(\n'
        '                _cl.get("ceiling"), CLAIM_PERSISTED):',
        "**具体输入**: ceiling=validated 的 metrics 渲染报告 ⇒ 「我无法确立的事」少一条 ⇒ "
        "`test_a_capped_level_shows_up_in_the_unable_section` 红。§17 那一节是这套东西的"
        "出口: 一条被压级的结论必须在那句「我确立不了」里出现, 否则读者以为它只是没跑完"),
]
