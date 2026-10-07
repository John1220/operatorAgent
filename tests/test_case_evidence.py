"""§6.2 条款 2/3 的**证据生产者**: case 身份 + 必判 case + 逐 case 硬回退上限

设计原文 (§6.2):

    规则 2: 候选必须在**全部** required_cases 上通过 —— 这是唯一能挡住「针对测试 shape
            特化」的机制。
    规则 3: 关键 shape 的硬回退约束优先于总体聚合 —— 一个候选不能因为在某个 shape 上快
            很多, 就掩盖关键 shape 上的严重回退; 回退越限 ⇒ §17.1 的 USE_CUSTOM 不成立。
    规则 4: required_cases / 权重 / aggregate 全部进契约 hash。

这一片之前: 契约里没有这两个字段, 评测侧也不记「本次评了哪一条 case」 ⇒ §17.1 的两条
证据恒 `None`（未确立, 见 I91）。这一片把它们变成**逐 run 可判**的三态证据 —— 于是
`use_custom` 的**证据**条件在生产路径上第一次真的能被满足（能不能拿到那一档还卡在声明
等级与真机读数, E 组把这条边界也钉住）。

四组, 各挡一类失效:

A. case 身份 (`contract.case_key`): 同一组输入任何时候同一个 key（顺序无关、可离线
   复算）; 形状读不到 ⇒ `?` 而不是 0; 没有输入 ⇒ 空串（**没有 case**, 不许拿去匹配
   `required_cases`）。
B. 声明进契约 (lint + hash + 往返): 读不出来的声明**拒绝冻结**（失败关闭）; 改一个声明
   必须改 hash（条款 4）; 空声明与本片之前的契约同 hash（旧载荷还能复盘）。
C. 两条判据的三态: 没声明 / 没评过 / 没量到 ⇒ `None`（未确立 —— 不许 vacuous truth,
   也不许把「没量到」读成「没越限」）; 只有真判过的才给 True/False。
D. 真跑一遍 `Conductor.execute`（MockGPU + 假 LLM）: 任务**声明了** ⇒ 两条证据被确立;
   任务**没声明** ⇒ 两条都未确立、理由说得出为什么。防的是「模块写好了但生产路径上没人
   调」（I45/I48 那一族）。

运行: python3 -m unittest tests.test_case_evidence -v
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.contract import (BaselineDef, ContractError, DTypeSemantics,
                                     ShapeDist, case_key, freeze,
                                     freeze_from_task, load_dict)
from operator_agent.eligibility import regression_over_limit, required_cases_ok
from operator_agent.report import REPORT_NAME
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                    TargetSpec, TaskComplexity, TaskSchema,
                                    TaskType)

CASE_A = "A=2x3|B=3x4"


def _ok_contract(**over):
    kw = dict(kernel_type="matmul",
              dtype_semantics=DTypeSemantics(dtype="fp32"),
              baseline=BaselineDef(name="b", same_device=True, same_window=True,
                                   same_dtype_semantics=True),
              shapes=ShapeDist(entries=(((64, 64, 64), 1.0),)),
              max_error=1e-2, min_speedup=1.0)
    kw.update(over)
    return freeze(**kw)


def _task(*, required_cases=(), max_regression=None):
    # 与 `tests/test_report_deliverable.py` 的端到端任务同形（min_speedup=0 / mock_gpu /
    # python）: 这一组要的是「跑到 `_save_to_db`」, 不是「L3 真的够快」——MockGPU 的读数
    # 在真实门槛下必然不过, 那样裁决根本不会被写出来（E 组的边界断言另有其条）。
    return TaskSchema(
        task_id="case-evidence-e2e",
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0,
                                   max_error=1e-3,
                                   required_cases=tuple(required_cases),
                                   max_regression=dict(max_regression or {})),
    )


def _inputs():
    rng = np.random.default_rng(0)
    return {"A": rng.standard_normal((32, 32)).astype(np.float32),
            "B": rng.standard_normal((32, 32)).astype(np.float32)}


class TestTheCaseIdentity(unittest.TestCase):
    """A 组: 「本次评的是哪一条 case」只有输入可查, 所以身份必须从输入派生"""

    def test_the_same_inputs_always_get_the_same_key(self):
        self.assertEqual(case_key(_inputs()), case_key(_inputs()))

    def test_the_name_order_does_not_matter(self):
        one = {"A": np.zeros((2, 3)), "B": np.zeros((3, 4))}
        two = {"B": np.zeros((3, 4)), "A": np.zeros((2, 3))}
        self.assertEqual(case_key(one), case_key(two))
        self.assertEqual(CASE_A, case_key(one))

    def test_different_shapes_are_different_cases(self):
        small = {"A": np.zeros((2, 3)), "B": np.zeros((3, 4))}
        big = {"A": np.zeros((2, 3)), "B": np.zeros((3, 8))}
        self.assertNotEqual(case_key(small), case_key(big))

    def test_a_shape_that_cannot_be_read_is_a_question_mark(self):
        """读不到形状 ⇒ `?`（**不是** 0, 也不是省略那一位）

        0 会让「这一位是 0」与「这一位读不到」长得一样, 省略会让「读不到」与「没有这位」
        长得一样 —— 两种混同都会让判据面把不确定当成确定。
        """
        self.assertEqual("A=?", case_key({"A": [1, 2]}))
        self.assertEqual("A=?", case_key({"A": 3}))
        self.assertEqual("A=0", case_key({"A": np.zeros(0)}))
        self.assertNotEqual(case_key({"A": [1, 2]}), case_key({"A": np.zeros(0)}))

    def test_no_inputs_is_no_case(self):
        """没有输入 ⇒ 空串 = **没有 case**: 不许拿它去匹配 required_cases（会假通过）"""
        for nothing in (None, {}, [], "", 3):
            self.assertEqual("", case_key(nothing), nothing)

    def test_the_key_is_offline_recomputable(self):
        """身份是**纯函数**: 同一组形状在任何进程、任何 dtype 下都是同一个 key"""
        self.assertEqual(case_key({"A": np.zeros((8, 8))}),
                         case_key({"A": np.zeros((8, 8)).astype(np.float64)}))


class TestTheDeclarationEntersTheContract(unittest.TestCase):
    """B 组: 条款 4 —— 声明进 hash; 读不出来的声明拒绝冻结; 空声明不改变 hash"""

    def test_a_required_case_changes_the_hash(self):
        self.assertNotEqual(_ok_contract().hash,
                            _ok_contract(required_cases=(CASE_A,)).hash,
                            "条款 4: 改判据必须改 hash")

    def test_a_regression_limit_changes_the_hash(self):
        a = _ok_contract(max_regression=((CASE_A, 0.05),))
        b = _ok_contract(max_regression=((CASE_A, 0.06),))
        self.assertNotEqual(a.hash, b.hash)

    def test_an_empty_declaration_is_no_declaration(self):
        """空声明 = 没声明（判据内容相同）⇒ 与本片之前冻结的契约同 hash, 旧载荷还能复盘"""
        plain = _ok_contract()
        self.assertEqual(plain.hash, _ok_contract(required_cases=(),
                                                  max_regression=()).hash)
        back = load_dict(plain.as_dict())
        self.assertEqual((), back.required_cases)
        self.assertEqual((), back.max_regression)

    def test_the_order_is_normalised_and_duplicates_collapse(self):
        a = _ok_contract(required_cases=("b", "a", "a"))
        b = _ok_contract(required_cases=("a", "b"))
        self.assertEqual(a.hash, b.hash, "顺序无语义 ⇒ 同一份判据不许有两种 hash")
        self.assertEqual(("a", "b"), a.required_cases)

    def test_the_reader_separates_undeclared_from_zero(self):
        c = _ok_contract(max_regression=((CASE_A, 0.0),))
        self.assertEqual(0.0, c.regression_limit(CASE_A))
        self.assertIsNone(c.regression_limit("B=1x1"), "没声明 ≠ 声明了 0")

    def test_unreadable_declarations_refuse_to_freeze(self):
        for bad in (dict(required_cases=("",)),
                    dict(required_cases=(3,)),
                    dict(required_cases=("a",), max_regression=(("a", "x"),)),
                    dict(required_cases=("a",), max_regression=(("a", -1.0),)),
                    dict(required_cases=("a",), max_regression=(("a", True),)),
                    dict(required_cases=("a",), max_regression=(("a",),))):
            with self.subTest(bad=bad):
                with self.assertRaises(ContractError):
                    _ok_contract(**bad)

    def test_the_roundtrip_keeps_both_declarations(self):
        c = _ok_contract(required_cases=(CASE_A,), max_regression=((CASE_A, 0.05),))
        back = load_dict(c.as_dict())
        self.assertEqual(c.hash, back.hash)
        self.assertEqual((CASE_A,), back.required_cases)
        self.assertEqual(0.05, back.regression_limit(CASE_A))

class TestTheTaskIsTheDeclarationChannel(unittest.TestCase):
    """C 组: 声明从**任务**进契约 —— 任务侧是唯一「契约与输入同时在手」的地方"""

    def test_a_declared_task_reaches_the_frozen_contract(self):
        c = freeze_from_task(_task(required_cases=(CASE_A,),
                                    max_regression={CASE_A: 0.05}))
        self.assertEqual((CASE_A,), c.required_cases)
        self.assertEqual(0.05, c.regression_limit(CASE_A))

    def test_an_undeclared_task_declares_nothing(self):
        c = freeze_from_task(_task())
        self.assertEqual((), c.required_cases)
        self.assertEqual((), c.max_regression)
        self.assertIsNone(c.regression_limit(CASE_A))

    def test_the_declaration_is_part_of_the_contract_identity(self):
        """条款 4 在「任务 → 契约」这条链上也成立: 声明不同 ⇒ 契约 hash 不同"""
        declared = freeze_from_task(_task(required_cases=(CASE_A,)))
        plain = freeze_from_task(_task())
        self.assertNotEqual(declared.hash, plain.hash)

    def test_the_task_validation_refuses_unreadable_declarations(self):
        for spec in (ConstraintSpec(required_cases=("",)),
                     ConstraintSpec(required_cases=(3,)),
                     ConstraintSpec(max_regression={"": 0.05}),
                     ConstraintSpec(max_regression={CASE_A: "x"}),
                     ConstraintSpec(max_regression={CASE_A: -1.0}),
                     ConstraintSpec(max_regression=[(CASE_A, 0.05)])):
            with self.subTest(spec=spec):
                self.assertTrue(spec.validate(), "读不出来的声明必须在任务校验就被拒")

    def test_the_task_dict_writes_the_declaration_only_when_declared(self):
        """只在非空时写（同 target.required_caps）：恒为 [] 的键会被读成「声明过、要求为空」"""
        for key in ("required_cases", "max_regression"):
            self.assertNotIn(key, _task().to_dict()["constraints"], key)
        d = _task(required_cases=(CASE_A,), max_regression={CASE_A: 0.05}).to_dict()
        self.assertEqual([CASE_A], d["constraints"]["required_cases"])
        self.assertEqual({CASE_A: 0.05}, d["constraints"]["max_regression"])

    def test_the_declaration_is_additive(self):
        """声明是**加法**: 打开它不许动任务里别的字段（防止顺手改了判据的别处）"""
        plain = dict(_task().to_dict()["constraints"])
        declared = dict(_task(required_cases=(CASE_A,)).to_dict()["constraints"])
        for payload in (plain, declared):
            payload.pop("required_cases", None)
            payload.pop("max_regression", None)
        self.assertEqual(plain, declared)


class TestTheTwoJudgingFaces(unittest.TestCase):
    """D 组: 三态 —— 没声明 / 没评过 / 没量到都是 `None`（未确立）, 不是 False"""

    def test_no_declaration_is_not_vacuous_truth(self):
        value, why = required_cases_ok((), {CASE_A: True})
        self.assertIsNone(value, "没声明 ≠ 都通过了（否则不声明比声明更容易上台）")
        self.assertIn("没有声明", " ".join(why))

    def test_a_declared_case_that_was_never_evaluated_is_unestablished(self):
        value, why = required_cases_ok((CASE_A,), {"B=1x1": True})
        self.assertIsNone(value)
        self.assertIn(CASE_A, " ".join(why))
        self.assertIn("没有评过", " ".join(why))
        self.assertIn("B=1x1", " ".join(why), "理由要说清本次评过什么")

    def test_a_case_evaluated_without_a_verdict_is_unestablished(self):
        value, why = required_cases_ok((CASE_A,), {CASE_A: None})
        self.assertIsNone(value)
        self.assertIn("没有正确性结论", " ".join(why))

    def test_only_a_judged_case_gives_true_or_false(self):
        self.assertIs(False, required_cases_ok((CASE_A,), {CASE_A: False})[0])
        self.assertIs(True, required_cases_ok((CASE_A,), {CASE_A: True})[0])

    def test_extra_evaluated_cases_do_not_change_the_verdict(self):
        """别的 case 评过什么不影响「必判的那几条对不对」（那是**加**的账, 不是替代）"""
        self.assertIs(True, required_cases_ok((CASE_A,),
                                             {CASE_A: True, "X=1": False})[0])

    def test_no_limit_is_not_no_regression(self):
        value, why = regression_over_limit((), {CASE_A: 0.5})
        self.assertIsNone(value)
        self.assertIn("没有声明", " ".join(why))

    def test_an_unmeasured_case_is_not_a_pass(self):
        value, why = regression_over_limit(((CASE_A, 0.05),), {"B=1x1": 0.9})
        self.assertIsNone(value, "「没量到」不许读成「没越限」")
        self.assertIn("没有它们的加速比", " ".join(why))

    def test_a_regression_over_the_limit_is_true(self):
        value, why = regression_over_limit(((CASE_A, 0.05),), {CASE_A: 0.8})
        self.assertIs(True, value)
        self.assertIn("条款 3", " ".join(why))
        self.assertIn("0.2", " ".join(why), "理由要带量, 不能只说「越限」")

    def test_a_regression_at_the_limit_is_not_over_it(self):
        """边界: 1 - speedup == limit ⇒ **不算**越限（判据是 >, 不是 >=）"""
        self.assertIs(False, regression_over_limit(((CASE_A, 0.05),),
                                                   {CASE_A: 0.95})[0])

    def test_a_negative_limit_demands_faster_than_the_baseline(self):
        """值可以是负的: -0.1 = 要求比基线快 10%（值域的下界因此是 -1）"""
        self.assertIs(True, regression_over_limit(((CASE_A, -0.1),),
                                                  {CASE_A: 1.05})[0])
        self.assertIs(False, regression_over_limit(((CASE_A, -0.1),),
                                                   {CASE_A: 1.2})[0])

    def test_another_target_metric_is_unestablished(self):
        value, why = regression_over_limit(((CASE_A, 0.05),), {CASE_A: 0.8},
                                           target_metric="latency")
        self.assertIsNone(value, "今天只有 speedup_vs_baseline 的口径 ⇒ 不许猜")
        self.assertIn("latency", " ".join(why))

    def test_unreadable_limits_are_unestablished_not_crashes(self):
        for bad in (((CASE_A, "x"),), (("a",),), "nope", 7):
            with self.subTest(bad=bad):
                self.assertIsNone(regression_over_limit(bad, {CASE_A: 0.8})[0])

    def test_uncertainty_is_not_a_soft_false(self):
        """两个判据的 `None` 不许与 False 混同（下游读的是 `is False` / `is True`）"""
        for value, _ in (required_cases_ok((), {}), regression_over_limit((), {})):
            self.assertIsNone(value)
            self.assertIsNot(False, value)

    def test_a_case_key_of_empty_string_is_not_a_match(self):
        """空串（没有 case）不许匹配到任何一条 required_cases —— 否则「没评过」会假通过"""
        value, why = required_cases_ok(("",), {"": True})
        self.assertIs(True, value, "这是**显式**声明过的空 case（判据面照常判它）")
        missing, _ = required_cases_ok((CASE_A,), {case_key(None): False})
        self.assertIsNone(missing, "评测侧没给输入 ⇒ 空 case 不顶上必判 case")

class TestThroughARealRun(unittest.TestCase):
    """E 组: 模块写好了 ≠ 接上了（I45/I48 那一族的守卫）"""

    CODE = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
    SKETCH = {"algorithm": "matmul",
              "tiling": {"block_m": 16, "block_n": 16, "block_k": 16},
              "memory_plan": {}, "pipeline": [], "optimization_hints": []}

    class _Llm:
        def chat(self, messages, **kw):
            return TestThroughARealRun.CODE

        def chat_json(self, messages, **kw):
            return dict(TestThroughARealRun.SKETCH)

    def _run(self, *, declare: bool):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                      llm=self._Llm(), max_optimize_rounds=1)
        c.store.output_dir = Path(tmp.name)
        inputs = _inputs()
        case = case_key(inputs)
        task = _task(required_cases=(case,) if declare else (),
                     # 上限故意宽到不可能越界（同 `tests/test_case_ledger.py` 的 LOOSE_LIMIT）:
                     # MockGPU 的读数是墙钟量级的，紧上限会让这条断言在机器忙时随机变红 ——
                     # 这一组钉的是「两条证据被**确立**」，不是「这条 case 真的没退」。
                     max_regression={case: 5.0} if declare else None)
        with contextlib.redirect_stdout(io.StringIO()):
            art = c.execute(task, inputs, None, target_gflops=0.0)
        return art, case

    def test_the_run_publishes_the_case_it_evaluated(self):
        art, case = self._run(declare=True)
        self.assertEqual(case, art.benchmark_data.get("case_key"),
                         "「本次评了哪条 case」必须与 L3 读数同处发布")

    def test_a_declared_case_makes_both_evidence_established(self):
        """声明了 + 评过 + L2 通过 ⇒ 这一条第一次不是 `None`（I91 时它恒未确立）"""
        art, _ = self._run(declare=True)
        d = art.benchmark_data["implementation_decision"]
        self.assertIs(True, d["required_cases_ok"])
        self.assertIsNotNone(d["regression_over_limit"], "声明了上限 ⇒ 该有 True/False")
        self.assertIn("都评过且都正确", " ".join(d["reasons"]))

    def test_an_undeclared_task_leaves_both_unestablished(self):
        art, _ = self._run(declare=False)
        d = art.benchmark_data["implementation_decision"]
        self.assertIsNone(d["required_cases_ok"])
        self.assertIsNone(d["regression_over_limit"])
        text = " ".join(d["reasons"])
        self.assertIn("没有声明 required_cases", text)
        self.assertIn("没有声明 max_regression", text)
        self.assertEqual("inconclusive", d["decision"],
                         "证据未确立 ⇒ 门 1 挡住: 提不出「用自定义」的建议")

    def test_the_case_key_comes_from_the_inputs_not_from_the_contract(self):
        """两处空间今天不是同一个（§3.4 的欠账）: case 身份是**输入签名**

        拿契约里那个 kernel 级 shape（`freeze_from_task` 的 64x64x64 单点代表）冒充
        「本次评过」就是假证据 —— 这一条同时钉住「键从哪来」与「契约 shape 不参与」。
        """
        art, case = self._run(declare=True)
        self.assertEqual("A=32x32|B=32x32", case)
        self.assertEqual(case, art.benchmark_data.get("case_key"))
        self.assertNotIn("64x64x64", str(art.benchmark_data.get("case_key")))

    def test_a_declared_case_reaches_use_custom_and_stops_at_corpus_admission(self):
        """本片的可见后果: 三条证据**真的成立** ⇒ §17.1 阶梯的顶部第一次够得着

        两个方向都要钉住:

          * `use_custom` —— I91 时它**不可达**（两条证据恒 `None`）;
          * 它跨过的**不是**永久后果那条线: 同一次运行的 `corpus_admission.kept` 仍是
            False（执行侧没声明 REAL_DEVICE ⇒ 数字不能当语料）。裁决只对「怎么实现」给
            建议, 不改入库判定 —— 两条判据各读各的证据, 这一条防的是把它们并成一个开关。
        """
        art, _ = self._run(declare=True)
        d = art.benchmark_data["implementation_decision"]
        self.assertEqual("use_custom", d["decision"])
        self.assertIs(True, d["required_cases_ok"])
        self.assertIs(False, d["regression_over_limit"])
        self.assertIn("三条证据全中", " ".join(d["reasons"]))
        self.assertIs(False, (art.metadata or {})["corpus_admission"]["kept"],
                      "「建议用自定义核」不许顺手把这条读数变成语料")

    def test_the_report_carries_the_evidence_states(self):
        art, _ = self._run(declare=True)
        txt = (Path(art.saved_dir) / REPORT_NAME).read_text(encoding="utf-8")
        self.assertIn("实现裁决 (§17.1)", txt)
        self.assertIn("必判 case 全对 `是`", txt)


class TestOneProducerPerFact(unittest.TestCase):
    """结构: 身份一处派生、两条判据一处推导、评测侧一处发布; 报告只读"""

    def _src(self, rel):
        return (Path(__file__).resolve().parents[1] / rel).read_text(encoding="utf-8")

    def test_the_case_identity_has_one_definition(self):
        self.assertEqual(1, self._src("operator_agent/contract.py").count("def case_key("))

    def test_the_two_faces_have_one_derivation_each(self):
        el = self._src("operator_agent/eligibility.py")
        self.assertEqual(1, el.count("def required_cases_ok("))
        self.assertEqual(1, el.count("def regression_over_limit("))

    def test_the_conductor_publishes_the_case_once_and_derives_both(self):
        cu = self._src("operator_agent/conductor.py")
        self.assertEqual(1, cu.count("case_key(test_inputs)"))
        self.assertEqual(1, cu.count("required_cases_ok(contract.required_cases"))
        self.assertEqual(1, cu.count("regression_over_limit(contract.max_regression"))
        self.assertIn("artifact.benchmark_data[\"case_key\"]", cu)

    def test_the_reporter_does_not_derive_the_evidence(self):
        rp = self._src("operator_agent/report.py")
        for forbidden in ("required_cases_ok(", "regression_over_limit(", "case_key("):
            self.assertNotIn(forbidden, rp, forbidden)

    def test_the_case_verdict_is_read_through_the_shared_l2_helper(self):
        """§4.4 的 `correctness_verified` 与 §6.2 条款 2 读的是**同一份 L2 结果**

        今天这条守卫是**结构**的: 裁决只在 `all_passed` 那一支写（L2 必然通过）⇒ 行为上
        区分不出「调用点自己写 True」与「读 L2 判定」，所以按结构钉 —— 调用点必须走
        `_l2_verdict`，而它只有一处定义、两个读者（两处各挑一次就会漂）。
        """
        cu = self._src("operator_agent/conductor.py")
        self.assertEqual(1, cu.count("def _l2_verdict("))
        self.assertEqual(2, cu.count("self._l2_verdict(artifact)"),
                         "两个读者: correctness_verified 与条款 2 的 case 正确性")
        self.assertIn("l2_verdict = self._l2_verdict(artifact)", cu,
                      "证据侧必须**通过这个助手**读 L2, 而不是自己再挑一次")
        self.assertIn("evaluated = {c: l2_verdict for c in speedups}", cu)
    def test_the_harness_declares_the_case_it_evaluates(self):
        """真实通过率驱动: 声明与输入必须来自**同一次**构造（同一个 case_key）"""
        hp = self._src("benchmarks/harness_pass_rate.py")
        self.assertIn("_case = case_key(inputs)", hp)
        self.assertIn("required_cases=(_case,)", hp)
        self.assertIn("max_regression={_case: MAX_REGRESSION}", hp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
