"""§3.4 的 `BenchmarkCase`：契约拥有输入，且 case 身份只有**一个**空间（切片 I / CHECKLOG I94）

本片之前有两个互不相干的空间：契约里的 `ShapeDist`（kernel 级 `(M,N,K)`，**零消费者**）与
评测侧的 case 身份（`contract.case_key` = 输入形状签名，I92）。于是「契约声明要评什么」与
「评测真的评了什么」谁也管不着谁 —— `required_cases` 能不能被确立，全看调用方有没有恰好
传对那几组输入。这一片把设计 §3.4 的 `BenchmarkCase` 立起来（形状 + 数值语义 + seed），
并让三条路收口到**同一处生产**：身份（`case_key`）、输入（`materialize`）、形状分布到具体
输入的布局表（`cases_from_shapes`）。

六组, 各挡一类失效:

A. case 声明与身份：数组那条路与声明这条路**同一个函数**（否则「契约说的」永远匹配不上
   「本次评的」）; 形状读不到写 `?`、没有输入 = **没有 case**（不是空签名的 case）; 造出来的
   输入是确定性的、且**不是全 1**（全 1 会把「索引写错」掩盖成正确）; 认不出的 dtype 拒绝。
B. 形状分布 → case：布局表只有一张，且它的键与真机驱动的输入工厂一致（两处名字漂了，L2 的
   参考实现就找不到输入）; 未知 kernel_type ⇒ **None**（不猜布局）; 元数不对 ⇒ 拒绝。
C. 契约拥有输入：`cases` 进 hash（改一个 dtype 就是另一份判据）; 空声明与本片之前同 hash;
   顺序无语义; 身份分不开 ⇒ 拒绝; `required_cases`/`max_regression`/`case_weights` 里出现
   一个不在 `cases` 里的 key ⇒ **拒绝冻结**（一条永远判不了的判据）。
D. 任务渠道：声明 `shapes` ⇒ 派生 `cases` 且「全部形状」进 `required_cases`（§3.4 规范 2 的
   原话）; 显式 `cases` 透传; 两处声明对不上 ⇒ 拒绝; 未知布局 ⇒ 拒绝; 没声明 ⇒ 一个字不动。
E. 真跑一遍（MockGPU + 假 LLM）：**不传输入**也能评（输入由契约 `materialize()` 造）——「契约
   拥有输入」在生产路径上真的会走那条路; 调用方给了输入时，`case_match` 说得清对不对得上;
   既没输入又没声明 ⇒ 一句说得清的拒绝（不是深处一个 `TypeError`）。
F. 结构：一个事实一处生产（身份 / 布局表 / 输入解析 / 报告只读）。

运行: python3 -m unittest tests.test_bench_cases -v
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.bench_cases import (BenchCaseError, BenchmarkCase, TensorSpec, case_key,
                                        case_from_inputs, cases_from_shapes, known_layouts)
from operator_agent.contract import (ContractError, DTypeSemantics, BaselineDef, ShapeDist,
                                     freeze, freeze_from_task, load_dict)
from operator_agent.report import REPORT_NAME
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)

CODE = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
SKETCH = {"algorithm": "matmul",
          "tiling": {"block_m": 16, "block_n": 16, "block_k": 16},
          "memory_plan": {}, "pipeline": [], "optimization_hints": []}


def _case(shape=(32, 32), name="c", dtype="fp32"):
    return BenchmarkCase(name=name,
                         tensors=(TensorSpec("A", shape, dtype),
                                  TensorSpec("B", shape, dtype)))


def _ok_contract(**over):
    kw = dict(kernel_type="matmul",
              dtype_semantics=DTypeSemantics(dtype="fp32"),
              baseline=BaselineDef(name="b", same_device=True, same_window=True,
                                   same_dtype_semantics=True),
              shapes=ShapeDist(entries=(((64, 64, 64), 1.0),)),
              max_error=1e-3, min_speedup=1.0)
    kw.update(over)
    return freeze(**kw)


def _task(**over):
    kw = dict(task_id="bench-cases-e2e",
              task_type=TaskType.OPERATOR_GENERATION,
              source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
              target=TargetSpec(hardware="mock_gpu", dsl="python"),
              constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0,
                                         max_error=1e-3))
    kw.update(over)
    return TaskSchema(**kw)


def _constraints(**over):
    kw = dict(precision=Precision.FP32, min_speedup=0.0, max_error=1e-3)
    kw.update(over)
    return ConstraintSpec(**kw)


def _inputs(shape: int = 32) -> dict:
    return {"A": np.ones((shape, shape), dtype=np.float32),
            "B": np.ones((shape, shape), dtype=np.float32)}


class _Llm:
    def chat(self, messages, **kw):
        return CODE

    def chat_json(self, messages, **kw):
        return dict(SKETCH)


class TestTheCaseIsDeclared(unittest.TestCase):
    """A 组: case 声明与身份 —— 两条路同一个函数、确定性的输入、认不出就拒绝"""

    def test_the_identity_is_one_function(self):
        """数组那条路与声明这条路必须给出**同一个**字符串（本片的核心要求）"""
        c = _case()
        self.assertEqual(case_key(c.materialize()), c.key())

    def test_the_signature_ignores_order(self):
        a = case_key({"A": np.ones((2, 3)), "B": np.ones((3, 4))})
        b = case_key({"B": np.ones((3, 4)), "A": np.ones((2, 3))})
        self.assertEqual(a, b)
        self.assertEqual("A=2x3|B=3x4", a)

    def test_a_value_without_shape_is_a_question_mark(self):
        self.assertEqual("A=?", case_key({"A": 1.0}))

    def test_no_inputs_is_no_case(self):
        for empty in ({}, None, 42, []):
            self.assertEqual("", case_key(empty), repr(empty))

    def test_materialize_is_deterministic(self):
        first, second = _case().materialize(), _case().materialize()
        self.assertTrue(np.array_equal(first["A"], second["A"]))

    def test_materialize_is_not_all_ones(self):
        """全 1 的输入会把「索引写错」这类错误掩盖成正确 —— 造出来的必须是真数据"""
        arr = _case().materialize()["A"]
        self.assertFalse(np.allclose(arr, 1.0))
        self.assertGreater(float(arr.std()), 0.0)

    def test_an_unknown_dtype_is_refused(self):
        with self.assertRaises(BenchCaseError) as ctx:
            BenchmarkCase(tensors=(TensorSpec("A", (4, 4), "bf16"),)).materialize()
        self.assertIn("bf16", str(ctx.exception))

    def test_the_roundtrip_keeps_the_key(self):
        c = _case(shape=(8, 16), name="rt")
        back = BenchmarkCase.from_dict(json.loads(json.dumps(c.to_dict())))
        self.assertEqual(c.key(), back.key())
        self.assertEqual(8, back.tensors[0].shape[0])

    def test_the_declaration_is_canonical(self):
        """声明顺序不进 hash —— 同一组输入谁怎么写都是同一份判据"""
        a = BenchmarkCase(tensors=(TensorSpec("A", (2, 2)), TensorSpec("B", (2, 2))))
        b = BenchmarkCase(tensors=(TensorSpec("B", (2, 2)), TensorSpec("A", (2, 2))))
        self.assertEqual(a.content_dict(), b.content_dict())
        self.assertEqual(_ok_contract(cases=(a,)).hash, _ok_contract(cases=(b,)).hash)

    def test_the_audit_name_is_not_part_of_the_judgement(self):
        """`name` 是给人读的（同契约的 `notes`）: 名字变了，判据一个字没变"""
        a = BenchmarkCase(name="alice", tensors=(TensorSpec("A", (2, 2)),))
        b = BenchmarkCase(name="bob", tensors=(TensorSpec("A", (2, 2)),))
        self.assertEqual(_ok_contract(cases=(a,)).hash, _ok_contract(cases=(b,)).hash)

    def test_case_from_inputs_reads_shape_and_dtype(self):
        inputs = _inputs(32)
        declared = case_from_inputs(inputs, name="mm")
        self.assertEqual(case_key(inputs), declared.key())
        self.assertEqual("fp32", declared.tensors[0].dtype)

    def test_case_from_inputs_refuses_unreadable(self):
        with self.assertRaises(BenchCaseError):
            case_from_inputs({"A": "not-an-array"})
        with self.assertRaises(BenchCaseError):
            case_from_inputs({"A": np.ones((2, 2), dtype=np.bool_)})
        with self.assertRaises(BenchCaseError):
            case_from_inputs({})


class TestTheShapeDistributionBuildsCases(unittest.TestCase):
    """B 组: 布局表 —— 与真机驱动的输入工厂同一套键; 未知算子不猜"""

    def test_the_matmul_layout_matches_the_input_factory(self):
        from benchmarks.drivers.harness_pass_rate import inputs_for
        shape = (32, 16, 8)
        mine = cases_from_shapes("matmul", ((shape, 1.0),))[0]
        theirs = inputs_for("matmul", shape)
        self.assertEqual(case_key(theirs), mine.key(),
                         "两处名字漂了, L2 的参考实现就找不到输入")

    def test_the_rows_layout_matches_the_input_factory(self):
        from benchmarks.drivers.harness_pass_rate import inputs_for
        shape = (64, 32)
        mine = cases_from_shapes("layernorm", ((shape, 1.0),))[0]
        self.assertEqual(case_key(inputs_for("layernorm", shape)), mine.key())

    def test_an_unknown_kernel_type_is_not_guessed(self):
        self.assertIsNone(cases_from_shapes("attention", (((8, 64, 64), 1.0),)))
        self.assertIsNone(cases_from_shapes("conv2d", (((8, 64, 64), 1.0),)))
        self.assertNotIn("attention", known_layouts())

    def test_a_wrong_arity_is_refused(self):
        with self.assertRaises(BenchCaseError):
            cases_from_shapes("matmul", (((32, 32), 1.0),))

    def test_the_key_of_a_case_is_the_key_of_its_arrays(self):
        for c in cases_from_shapes("gemm", (((8, 8, 8), 1.0), ((4, 4, 4), 1.0))):
            self.assertEqual(case_key(c.materialize()), c.key())

    def test_the_weight_is_not_carried_into_the_case(self):
        """权重是目标函数那一半（`case_weights`）—— 两处都记一遍 = 两份会漂的权重"""
        c = cases_from_shapes("matmul", (((8, 8, 8), 7.5),))[0]
        self.assertEqual("A=8x8|B=8x8", c.key())


class TestTheContractOwnsItsInputs(unittest.TestCase):
    """C 组: `cases` 进 hash + lint 的两条收紧（身份分不开 / key 指向不存在的 case）"""

    def test_a_declared_case_changes_the_hash(self):
        plain = _ok_contract()
        with_case = _ok_contract(cases=(_case(),))
        self.assertNotEqual(plain.hash, with_case.hash, "评的是哪组输入也是判据")

    def test_a_changed_dtype_changes_the_hash_but_not_the_key(self):
        """⚠️ 身份 = 形状签名 ⇒ 同形状不同 dtype 的 key 相同（见 C 组最后一条的拒绝）"""
        a = _case(dtype="fp32")
        b = _case(dtype="fp16")
        self.assertEqual(a.key(), b.key())
        self.assertNotEqual(_ok_contract(cases=(a,)).hash, _ok_contract(cases=(b,)).hash)

    def test_no_declaration_is_hash_neutral(self):
        self.assertEqual(_ok_contract().hash, _ok_contract(cases=()).hash)

    def test_the_case_order_is_normalised(self):
        a, b = _case(shape=(8, 8), name="a"), _case(shape=(16, 16), name="b")
        self.assertEqual(_ok_contract(cases=(a, b)).hash,
                         _ok_contract(cases=(b, a)).hash)

    def test_the_roundtrip_keeps_the_inputs(self):
        c = _ok_contract(cases=(_case(shape=(8, 16)),), required_cases=("A=8x16|B=8x16",))
        back = load_dict(c.as_dict())
        self.assertEqual(c.hash, back.hash)
        self.assertEqual(c.cases[0].key(), back.cases[0].key())
        self.assertEqual(c.cases[0].tensors[0].shape, back.cases[0].tensors[0].shape)

    def test_the_reader_is_three_state(self):
        c = _ok_contract(cases=(_case(shape=(8, 8)),))
        self.assertIsNotNone(c.case("A=8x8|B=8x8"))
        self.assertIsNone(c.case("A=9x9|B=9x9"), "不是这条 case ⇒ 不造一个空壳出来")
        self.assertIsNone(_ok_contract().case("A=8x8|B=8x8"), "契约没声明输入")

    def test_duplicate_identities_are_refused(self):
        """身份 = 形状签名 ⇒ 同形状不同 dtype 的两条 case 在账本里分不开 ⇒ 拒绝"""
        with self.assertRaises(ContractError) as ctx:
            _ok_contract(cases=(_case(dtype="fp32"), _case(dtype="fp16")))
        self.assertIn("身份相同", str(ctx.exception))

    def test_a_key_outside_the_declared_cases_is_refused(self):
        for bad in (dict(required_cases=("A=9x9|B=9x9",)),
                    dict(max_regression=(("A=9x9|B=9x9", 0.05),)),
                    dict(case_weights=(("A=9x9|B=9x9", 1.0),))):
            kw = dict(cases=(_case(shape=(8, 8)),));
            kw.update(bad)
            with self.assertRaises(ContractError, msg=repr(bad)) as ctx:
                _ok_contract(**kw)
            self.assertIn("不在契约声明的 cases 里", str(ctx.exception))

    def test_a_key_inside_the_declared_cases_is_fine(self):
        c = _ok_contract(cases=(_case(shape=(8, 8)),), required_cases=("A=8x8|B=8x8",),
                         max_regression=(("A=8x8|B=8x8", 0.05),),
                         case_weights=(("A=8x8|B=8x8", 1.0),))
        self.assertEqual(1, len(c.cases))

    def test_a_bad_tensor_is_refused(self):
        for bad in (BenchmarkCase(tensors=()),
                    BenchmarkCase(tensors=(TensorSpec("", (4, 4)),)),
                    BenchmarkCase(tensors=(TensorSpec("A", (-1, 4)),)),
                    BenchmarkCase(tensors=("A",))):
            with self.assertRaises(ContractError, msg=repr(bad)):
                _ok_contract(cases=(bad,))


class TestTheTaskChannel(unittest.TestCase):
    """D 组: 任务声明 shapes / cases ⇒ 契约拥有输入（没声明 ⇒ 一个字不动）"""

    def test_declared_shapes_become_cases_and_all_of_them_are_required(self):
        c = freeze_from_task(_task(constraints=_constraints(
            shapes=((32, 32, 32), (16, 16, 16)))))
        self.assertEqual(["A=16x16|B=16x16", "A=32x32|B=32x32"],
                         [x.key() for x in c.cases])
        self.assertEqual(("A=16x16|B=16x16", "A=32x32|B=32x32"), c.required_cases,
                         "§3.4 规范 2: 候选须在**全部**形状上通过")
        self.assertEqual({((32, 32, 32), 1.0), ((16, 16, 16), 1.0)}, set(c.shapes.entries))

    def test_an_explicit_required_cases_wins(self):
        c = freeze_from_task(_task(constraints=_constraints(
            shapes=((32, 32, 32), (16, 16, 16)), required_cases=("A=32x32|B=32x32",))))
        self.assertEqual(("A=32x32|B=32x32",), c.required_cases)
        self.assertEqual(2, len(c.cases), "声明过的输入仍然全都进契约")

    def test_explicit_cases_go_through_both_shapes(self):
        obj = case_from_inputs(_inputs(16), name="mm")
        for declared in (obj, obj.to_dict()):
            c = freeze_from_task(_task(constraints=_constraints(cases=(declared,))))
            self.assertEqual(obj.key(), c.cases[0].key(), repr(type(declared)))
            self.assertEqual((obj.key(),), c.required_cases)

    def test_two_sources_must_agree(self):
        with self.assertRaises(ContractError) as ctx:
            freeze_from_task(_task(constraints=_constraints(
                shapes=((32, 32, 32),), cases=(case_from_inputs(_inputs(16)),))))
        self.assertIn("对不上", str(ctx.exception))

    def test_an_unknown_layout_is_refused_when_shapes_are_declared(self):
        task = _task(source=SourceSpec(kernel_type="attention",
                                       complexity=TaskComplexity.LOW),
                     constraints=_constraints(shapes=((8, 64, 64),)))
        with self.assertRaises(ContractError) as ctx:
            freeze_from_task(task)
        self.assertIn("没有输入布局表", str(ctx.exception))

    def test_an_undeclared_task_stays_empty(self):
        c = freeze_from_task(_task())
        self.assertEqual((), c.cases, "没声明 ⇒ 不派生（输入由调用方给）")
        self.assertEqual((), c.required_cases)
        self.assertEqual((((64, 64, 64), 1.0),), c.shapes.entries, "老的代表形状不变")

    def test_to_dict_is_additive(self):
        plain = dict(_task().to_dict()["constraints"])
        declared = dict(_task(constraints=_constraints(
            shapes=((32, 32, 32),), cases=(case_from_inputs(_inputs(32)),))).to_dict()["constraints"])
        for payload in (plain, declared):
            payload.pop("shapes", None)
            payload.pop("cases", None)
        self.assertEqual(plain, declared, "没声明与声明了只差那两个键")

    def test_a_bad_case_declaration_is_reported_by_the_schema(self):
        errs = _constraints(cases=({"name": "x", "tensors": [{"name": "A", "shape": [-1]}]},)).validate()
        self.assertTrue(any("读不出来" in e for e in errs), errs)

class TestThroughARealRun(unittest.TestCase):
    """E 组: 真跑一遍 —— **不传输入**也能评（输入由契约造）/ 调用方给了输入时对账"""

    def _run(self, task, inputs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                      llm=_Llm(), max_optimize_rounds=1)
        c.store.output_dir = Path(tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            art = c.execute(task, inputs, target_gflops=0.0)
        return art, (art.benchmark_data or {})

    def _contract_task(self, **over):
        cons = _constraints(shapes=((32, 32, 32), (16, 16, 16)), aggregate="weighted_geomean",
                            case_weights={"A=32x32|B=32x32": 3.0, "A=16x16|B=16x16": 1.0},
                            max_regression={"A=32x32|B=32x32": 5.0, "A=16x16|B=16x16": 5.0})
        for key, value in over.items():
            setattr(cons, key, value)
        return _task(constraints=cons)

    def test_the_inputs_come_from_the_contract(self):
        art, m = self._run(self._contract_task(), None)
        self.assertTrue(art.all_passed)
        self.assertEqual("contract", m.get("case_source"))
        self.assertIs(True, m.get("case_match"))
        self.assertEqual(2, m.get("case_declared"))

    def test_the_ledger_covers_every_declared_case(self):
        _, m = self._run(self._contract_task(), None)
        self.assertEqual({"A=16x16|B=16x16", "A=32x32|B=32x32"}, set(m.get("cases") or {}))
        d = m["implementation_decision"]
        self.assertIs(True, d["required_cases_ok"])
        # This legacy Mock fixture runs with task min_speedup=0 to exercise
        # bookkeeping, while FrozenContract normalizes that sentinel to 1.
        # It is not a matching workload deployment credential. Real matching
        # objective/use_custom is covered by test_workload_evidence.
        self.assertEqual("inconclusive", d["decision"])
        self.assertIn("criterion differs from frozen contract", " ".join(d["reasons"]))

    def test_the_objective_is_recomputable_from_the_ledger(self):
        import math
        _, m = self._run(self._contract_task(), None)
        led = m["cases"]
        manual = math.exp((3.0 * math.log(led["A=32x32|B=32x32"]["speedup_vs_baseline"])
                           + 1.0 * math.log(led["A=16x16|B=16x16"]["speedup_vs_baseline"])) / 4.0)
        self.assertAlmostEqual(manual, m["objective"]["speedup"], places=3)

    def test_the_report_says_where_the_inputs_came_from(self):
        art, _ = self._run(self._contract_task(), None)
        txt = (Path(art.saved_dir) / REPORT_NAME).read_text(encoding="utf-8")
        self.assertIn("输入来源 (§3.4)", txt)
        self.assertIn("契约的 `cases`", txt)
        self.assertIn("契约与本次对得上: 是", txt)

    def test_caller_inputs_are_reconciled_with_the_declaration(self):
        """调用方给的那几组**不在**契约声明里 ⇒ `case_match=False`（数据，不是门）"""
        art, m = self._run(self._contract_task(), [_inputs(8)])
        self.assertEqual("caller", m.get("case_source"))
        self.assertIs(False, m.get("case_match"))
        self.assertEqual(2, m.get("case_declared"))

    def test_the_report_says_caller_when_the_caller_gave_the_inputs(self):
        art, _ = self._run(self._contract_task(), [_inputs(8)])
        txt = (Path(art.saved_dir) / REPORT_NAME).read_text(encoding="utf-8")
        self.assertIn("调用方给的数组", txt, "报告不许把「调用方给的」印成「契约给的」")
        self.assertIn("契约与本次对得上: 否", txt)

    def test_caller_inputs_that_match_are_reconciled_true(self):
        _, m = self._run(self._contract_task(), None)  # 契约那条路必然对得上
        self.assertIs(True, m.get("case_match"))

    def test_no_inputs_and_no_declared_cases_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(_task(), None)
        self.assertIn("没有 case 可评", str(ctx.exception))


class TestOneProducerPerFact(unittest.TestCase):
    """F 组: 结构 —— 身份 / 布局表 / 输入解析各一处，报告只读"""

    def _src(self, rel):
        return (Path(__file__).resolve().parents[2] / rel).read_text(encoding="utf-8")

    def test_the_identity_has_one_definition(self):
        total = (self._src("operator_agent/bench_cases.py").count("def case_key(")
                 + self._src("operator_agent/contract.py").count("def case_key("))
        self.assertEqual(1, total)

    def test_the_layout_table_has_one_home(self):
        self.assertEqual(1, self._src("operator_agent/bench_cases.py").count("_CASE_LAYOUTS ="));
        for rel in ("operator_agent/verifier.py", "operator_agent/conductor.py",
                    "operator_agent/contract.py", "benchmarks/drivers/harness_pass_rate.py"):
            self.assertNotIn("_CASE_LAYOUTS", self._src(rel), rel)

    def test_the_conductor_resolves_inputs_in_one_place(self):
        cu = self._src("operator_agent/conductor.py")
        self.assertEqual(1, cu.count("def _resolve_input_sets("))
        self.assertEqual(1, cu.count("self._resolve_input_sets("))
        self.assertNotIn("input_sets = list(test_inputs) if isinstance(test_inputs, list)", cu,
                         "输入解析只许在 `_resolve_input_sets` 里做（两处各写一遍就会漂）")

    def test_the_input_source_is_read_from_the_published_key(self):
        """报告只许**读**那一格（`case_source`），不许拿着别的格自己推来源"""
        rp = self._src("operator_agent/report.py")
        self.assertEqual(1, rp.count('_get(m, "case_source"'))

    def test_the_reporter_derives_nothing(self):
        rp = self._src("operator_agent/report.py")
        for forbidden in ("case_key(", "case_from_inputs(", ".materialize(", "cases_from_shapes("):
            self.assertNotIn(forbidden, rp, forbidden)

    def test_the_harness_declares_the_case_from_the_inputs_it_passes(self):
        hp = self._src("benchmarks/drivers/harness_pass_rate.py")
        self.assertIn("_declared = case_from_inputs(inputs", hp)
        self.assertIn("cases=(_declared,)", hp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
