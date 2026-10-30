"""「§3.4 契约拥有输入 + case 身份一个空间」那张表的可执行形态

靶子: `tests/unit/test_bench_cases.py`（声明与身份 / 布局表 / 契约字段与收紧 lint / 任务渠道 /
真跑一遍 / 结构）。

为什么单独一张表: 这一片把「评的是什么输入」放进契约，最容易写歪的正是**声明**与**真输入**
之间的那一层 —— 身份各写一份（M1）、没输入也造一条 case（M2）、形状读不到写成 0（M3）、张量
语义认不出就默认 fp32（M4）、造出来的输入是假的（M5/M6）、未知算子的布局靠猜（M7/M8）、具体
输入不进 hash（M9）、key 指向不存在的 case 也放行（M10）、声明了形状却不派生（M11）、两处声明
对不上也放行（M12）、契约驱动那条路其实没走契约（M13）、没输入也开工（M14）、对账恒报「对上」
（M15）、审计名/声明顺序污染判据（M16）。

**没有结构行**: 这一片每条故障都能被一条**行为**断言看见（M16 也是 —— 名字或顺序一变，契约
hash 就变，`test_the_audit_name_is_not_part_of_the_judgement` 会红）。F 组那几条结构守卫照旧
钉住「一处生产」，但今天没有一条**行为不可区分**的故障可以拿它当唯一杀手 —— 硬凑一行假的
结构行比缺一行更坏。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/bench_cases.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {
    "bc": "operator_agent/bench_cases.py",
    "ct": "operator_agent/contract.py",
    "co": "operator_agent/conductor.py",
}

TARGETS = ["tests/unit/test_bench_cases.py"]

ROWS = [
    Row("M1 身份改成人读的名字", "bc",
        '        return case_key({t.name: t for t in self.tensors if isinstance(t, TensorSpec)})',
        '        return str(self.name)',
        "**具体输入**: 契约里声明的 case 与本次评的那组（`case_key(数组)`）对不上 ⇒"
        " `test_the_identity_is_one_function` / `test_the_key_of_a_case_is_the_key_of_its_arrays`"
        " 等多条红。身份一旦变成「谁起的名字」，两个进程就再也认不出同一条 case"),
    Row("M2 没有输入被当成一条空 case", "bc",
        '    if not isinstance(inputs, Mapping):\n        return ""',
        '    if not isinstance(inputs, Mapping):\n        return "?"',
        "**具体输入**: `case_key({})` / `case_key(None)` ⇒ 变异后返回一个非空串 ⇒"
        " `test_no_inputs_is_no_case` 红。「没有 case」被读成「有一条 case」会让"
        " `required_cases` 凭空配上号"),
    Row("M3 形状读不到写成 0", "bc",
        '        parts.append(f"{name}=" + ("x".join(str(d) for d in shape) if shape else "?"))',
        '        parts.append(f"{name}=" + ("x".join(str(d) for d in shape) if shape else "0"))',
        "**具体输入**: 输入值没有 `.shape` ⇒ 变异后写成 `A=0`（一个看起来像真形状的假身份）⇒"
        " `test_a_value_without_shape_is_a_question_mark` 红"),
    Row("M4 认不出的张量语义被当成 fp32", "bc",
        '    mapped = _NUMPY_TO_CASE_DTYPE.get(plain) or (plain if plain in CASE_DTYPES else None)',
        '    mapped = _NUMPY_TO_CASE_DTYPE.get(plain) or "fp32"',
        "**具体输入**: 一组 `bool_` 输入 ⇒ 变异后声明成 fp32（读数与声明不符，而没人看得出来）"
        " ⇒ `test_case_from_inputs_refuses_unreadable` 红。不降级是这一条的要点"),
    Row("M5 造出来的输入是全 1", "bc",
        '                out[t.name] = rng.standard_normal(t.shape).astype(dt)',
        '                out[t.name] = np.ones(t.shape, dtype=dt)',
        "**具体输入**: `materialize()` ⇒ 变异后每组张量都是 1 ⇒"
        " `test_materialize_is_not_all_ones` 红。全 1 会把「索引写错」掩盖成正确（这正是"
        " 这一片要喂给 L2 的那类错误）"),
    Row("M6 造出来的输入不确定", "bc",
        '        rng = np.random.default_rng(int(self.seed))',
        '        rng = np.random.default_rng()',
        "**具体输入**: 同一个声明造两次 ⇒ 变异后两组数组不同 ⇒"
        " `test_materialize_is_deterministic` 红。「契约里的输入」必须可复算"),
    Row("M7 未知算子的布局靠猜", "bc",
        '    if layout is None:\n        return None',
        '    if layout is None:\n        layout = (2, _rows_case)',
        "**具体输入**: `cases_from_shapes(\"attention\", ((8, 64, 64), 1.0))` ⇒ 变异后造出一条"
        " 形状完全不对的 case ⇒ `test_an_unknown_kernel_type_is_not_guessed` 红。"
        " 猜错的布局会让「评的是什么」静默漂移"),
    Row("M8 matmul 的 A 用 (m,n)（与输入工厂漂开）", "bc",
        '        tensors=(TensorSpec("A", (m, k), dtype), TensorSpec("B", (k, n), dtype)),',
        '        tensors=(TensorSpec("A", (m, n), dtype), TensorSpec("B", (k, n), dtype)),',
        "**具体输入**: `cases_from_shapes(\"matmul\", ((32, 16, 8), 1.0))` 与真机驱动的"
        " `inputs_for` 比 ⇒ 变异后 key 不同 ⇒ `test_the_matmul_layout_matches_the_input_factory`"
        " 红。两处名字/形状漂了，L2 的参考实现就找不到输入"),
    Row("M9 具体输入不进 hash", "ct",
        '        if self.cases:\n            d["cases"] = [c.content_dict() for c in self.cases]',
        '        if False:\n            d["cases"] = [c.content_dict() for c in self.cases]',
        "**具体输入**: 两份契约只差声明的输入 ⇒ 变异后 hash 相同 ⇒"
        " `test_a_declared_case_changes_the_hash` 红。「评的是哪组输入」被换掉而判据没变是最"
        " 难发现的一类漂移"),
    Row("M10 key 指向不存在的 case 也放行", "ct",
        '    for key in required_cases or ():\n        if isinstance(key, str) and key and key not in declared:',
        '    for key in required_cases or ():\n        if False:',
        "**具体输入**: 契约声明了 8x8 那条 case，`required_cases` 却写着 9x9 ⇒ 变异后照样冻结"
        " ⇒ `test_a_key_outside_the_declared_cases_is_refused` 红。指向不存在的 case 的判据"
        " 是一条永远判不了的判据"),
    Row("M11 声明了 shapes 却不派生 cases", "ct",
        '    if declared_shapes:\n        derived = cases_from_shapes(kt, shape_entries, dtype_s)',
        '    if declared_shapes and False:\n        derived = cases_from_shapes(kt, shape_entries, dtype_s)',
        "**具体输入**: 任务只声明 `shapes=((32,32,32),(16,16,16))` ⇒ 变异后契约里没有任何具体"
        " 输入、`required_cases` 也空 ⇒ `test_declared_shapes_become_cases_and_all_of_them_are_required`"
        " 与真跑那组全红（「契约拥有输入」名存实亡）"),
    Row("M12 两处声明对不上也放行", "ct",
        '        if cases and {c.key() for c in cases} != {c.key() for c in derived}:\n            raise ContractError(',
        '        if False:\n            raise ContractError(',
        "**具体输入**: 任务同时声明 `shapes=((32,32,32),)` 与 `cases=(16x16 那条,)` ⇒ 变异后"
        " 32x32 那条派生结果被**丢掉**（两处只有一个真相来源）⇒ `test_two_sources_must_agree` 红"),
    Row("M13 契约驱动那条路其实没走契约", "co",
        '        return [c.materialize() for c in contract.cases], "contract", declared_cases',
        '        return [c.materialize() for c in contract.cases], "caller", declared_cases',
        "**具体输入**: 不传输入、契约声明了 case 的那次 run ⇒ 变异后产物说「输入是调用方给的」"
        " ⇒ `test_the_inputs_come_from_the_contract` / `test_the_report_says_where_the_inputs_came_from`"
        " 红。来源那一格是这一片唯一的可观测差别（值可以一样，来源不能混）"),
    Row("M14 没有输入也开工", "co",
        '        if not declared_cases:\n            raise ValueError(\n                "没有输入，契约也没声明 cases ⇒ 没有 case 可评（不拿别的输入凑）: "\n                "要么显式传 test_inputs，要么在任务里声明 shapes/cases")\n        return [c.materialize() for c in contract.cases], "contract", declared_cases',
        '        if not declared_cases:\n            return [{}], "", ()\n        return [c.materialize() for c in contract.cases], "contract", declared_cases',
        "**具体输入**: 既没传输入、契约也没声明 case ⇒ 变异后拿一组空输入开工（崩在更深处）⇒"
        " `test_no_inputs_and_no_declared_cases_is_refused` 红。一句说得清的拒绝胜过深处的"
        " `TypeError`"),
    Row("M15 对账恒报「对上了」", "co",
        '                    artifact.benchmark_data["case_match"] = bool(\n                        evaluated and evaluated <= set(declared_cases))',
        '                    artifact.benchmark_data["case_match"] = True',
        "**具体输入**: 契约声明了 16/32 两条，调用方却给了 8x8 那组 ⇒ 变异后报 True ⇒"
        " `test_caller_inputs_are_reconciled_with_the_declaration` 与报告那条红。"
        " 「评的不是契约说的那条」必须看得见"),
    Row("M16 审计名与声明顺序污染判据", "bc",
        '        return {"seed": int(self.seed),\n                "tensors": [t.to_dict() for t in\n                            sorted(self.tensors, key=lambda t: t.name)]}',
        '        return {"name": self.name, "seed": int(self.seed),\n                "tensors": [t.to_dict() for t in self.tensors]}',
        "**具体输入**: 同一组输入换一个审计名 / 换个声明顺序 ⇒ 变异后契约 hash 变了（判据变了"
        " 一个字？没有）⇒ `test_the_audit_name_is_not_part_of_the_judgement` 与"
        " `test_the_declaration_is_canonical` 红"),
]
