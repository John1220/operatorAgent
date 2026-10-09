"""§6.2 的**目标函数**与**逐 case 账本**（切片 H / CHECKLOG I93）

设计原文 (§6.2 / §3.4 / §16):

    §6.2 规则 4: required_cases / **权重** / aggregate 全部进契约 hash（本片补上后两格）。
    §3.4: 目标函数是「加权几何平均」而不是单形状加速比 —— 而这正是 §16 反模式里那条
          「用单形状的加速比作为成功标准」。

I92 把**必判 case** 与**回退上限**变成了逐 run 可判的证据，但「本次评了几条 case」仍然
只有一条（L3 明说「多组输入套件时，性能基准使用第一组输入」）⇒ 聚合无从谈起。本片补的
是两件事: ① **逐 case 账本**（多组输入时 L3 逐 case 量加速比）; ② 契约里的**权重 + 聚合
口径**，于是目标函数第一次是一个可查的数 —— ⚠️ 但它今天**不进门槛**（准入仍看主 case 的
加速比），E 组与 D 组把这条边界一起钉住。

五组, 各挡一类失效:

A. 权重/口径进契约: 改一个权重必须改 hash（条款 4）; 空声明与本片之前的契约同 hash;
   读不出来的声明**拒绝冻结**（失败关闭）; `case_weight` 分得开「没声明」与「声明 0」。
B. `aggregate_speedup` 的三态: 没声明权重 / 没声明口径 / 有 case 没量到 / 加速比 ≤ 0 进
   不了几何平均 / 指标不是 speedup ⇒ `None`（**不替人补默认**，也不给一个少算了最差那条
   的数）。
C. verifier 的账本: 只传一组 ⇒ **不发布**（与 I92 之前一字不差）; 多组 ⇒ 逐 case 一条;
   主 case **不重复测**; 同形状的两组是**同一条 case**（身份 = 形状签名）。
D. 真跑一遍 `Conductor.execute`（MockGPU + 假 LLM, 两组输入）: 账本 ⇒ 两条判据 ⇒ 目标
   函数 ⇒ 报告，一条链上全都在。
E. 结构: 一个事实一处生产（账本一个生产者、聚合一处推导、报告只读）。

运行: python3 -m unittest tests.test_case_ledger -v
"""

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.contract import (CASE_AGGREGATE_MEAN, CASE_AGGREGATE_WORST_CASE,
                                     CASE_AGGREGATES, BaselineDef, ContractError,
                                     DTypeSemantics, ShapeDist, case_key, freeze,
                                     freeze_from_task, load_dict)
from operator_agent.eligibility import aggregate_speedup
from operator_agent.report import REPORT_NAME
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                    TargetSpec, TaskComplexity, TaskSchema,
                                    TaskType)
from operator_agent.verifier import Verifier

CASE_A = "A=2x3|B=3x4"
CASE_B = "A=4x5|B=5x6"

#: 端到端那几组声明用的回退上限。**故意宽到不可能越界**（要越界得跑出 speedup < -4，
#: 而加速比恒为正）: MockGPU 的读数是墙钟量级的，紧上限（如 0.05）会让「判据被确立」
#: 这件事在机器忙的时候随机变红 —— 那几组要钉的是**三态被确立**，不是「这条 case 真的
#: 没退」（后者由 `TestTheAggregateIsThreeState` 的纯函数用例逐值钉住）。
LOOSE_LIMIT = 5.0
CODE = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
SKETCH = {"algorithm": "matmul",
          "tiling": {"block_m": 16, "block_n": 16, "block_k": 16},
          "memory_plan": {}, "pipeline": [], "optimization_hints": []}


def _ok_contract(**over):
    """一份合法契约（与 `tests/test_case_evidence.py` 同形）+ 本片的两个字段。"""
    kw = dict(kernel_type="matmul",
              dtype_semantics=DTypeSemantics(dtype="fp32"),
              baseline=BaselineDef(name="b", same_device=True, same_window=True,
                                   same_dtype_semantics=True),
              shapes=ShapeDist(entries=(((64, 64, 64), 1.0),)),
              max_error=1e-3, min_speedup=1.0)
    kw.update(over)
    return freeze(**kw)


def _task(**over):
    """端到端任务: mock_gpu / python / min_speedup=0（要让流水线走到 `_save_to_db`）。"""
    kw = dict(task_id="case-ledger-e2e",
              task_type=TaskType.OPERATOR_GENERATION,
              source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
              target=TargetSpec(hardware="mock_gpu", dsl="python"),
              constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0,
                                         max_error=1e-3))
    kw.update(over)
    return TaskSchema(**kw)


def _task_with(**constraint_over):
    return _task(constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0,
                                             max_error=1e-3, **constraint_over))


def _inputs(shape: int = 32) -> dict:
    return {"A": np.ones((shape, shape)), "B": np.ones((shape, shape))}


class _Llm:
    def chat(self, messages, **kw):
        return CODE

    def chat_json(self, messages, **kw):
        return dict(SKETCH)


class TestWeightsAndAggregateEnterTheContract(unittest.TestCase):
    """A 组: §6.2 条款 4 —— 权重与聚合口径进契约 hash；读不出来的拒绝冻结"""

    def test_a_weight_changes_the_hash(self):
        plain = _ok_contract()
        with_w = _ok_contract(case_weights=((CASE_A, 1.0),))
        self.assertNotEqual(plain.hash, with_w.hash, "权重能改 = 目标函数能改")

    def test_the_aggregate_changes_the_hash(self):
        a = _ok_contract(case_weights=((CASE_A, 1.0),), aggregate=CASE_AGGREGATE_MEAN)
        b = _ok_contract(case_weights=((CASE_A, 1.0),), aggregate=CASE_AGGREGATE_WORST_CASE)
        self.assertNotEqual(a.hash, b.hash)

    def test_an_empty_declaration_is_hash_neutral(self):
        """空 = 没声明 ⇒ 与本片之前的契约**同一份判据内容** ⇒ 同 hash（旧载荷还能复盘）"""
        self.assertEqual(_ok_contract().hash,
                         _ok_contract(case_weights=(), aggregate="").hash)

    def test_the_weight_order_is_normalised(self):
        a = _ok_contract(case_weights=((CASE_B, 2.0), (CASE_A, 1.0)))
        b = _ok_contract(case_weights=((CASE_A, 1.0), (CASE_B, 2.0)))
        self.assertEqual(a.hash, b.hash, "顺序无语义 ⇒ 同内容必须同 hash")

    def test_the_roundtrip_keeps_both(self):
        c = _ok_contract(required_cases=(CASE_A,), case_weights=((CASE_A, 3.0),),
                         aggregate=CASE_AGGREGATE_MEAN)
        back = load_dict(c.as_dict())
        self.assertEqual(c.hash, back.hash)
        self.assertEqual(c.case_weights, back.case_weights)
        self.assertEqual(c.aggregate, back.aggregate)

    def test_the_weight_reader_tells_not_declared_from_declared_zero(self):
        """「没声明权重」与「声明了 0 权重（这条不进聚合）」必须是两件事"""
        c = _ok_contract(case_weights=((CASE_A, 0.0), (CASE_B, 2.0)))
        self.assertEqual(0.0, c.case_weight(CASE_A))
        self.assertEqual(2.0, c.case_weight(CASE_B))
        self.assertIsNone(c.case_weight("A=9x9|B=9x9"))

    def test_unreadable_weight_declarations_are_refused(self):
        for bad in (dict(case_weights=(("", 1.0),)),
                    dict(case_weights=((CASE_A, -1.0),)),
                    dict(case_weights=((CASE_A, 0.0),)),
                    dict(case_weights=((CASE_A, "x"),)),
                    dict(case_weights=(CASE_A,))):
            with self.assertRaises(ContractError, msg=repr(bad)):
                _ok_contract(**bad)

    def test_an_unknown_aggregate_is_refused(self):
        with self.assertRaises(ContractError) as ctx:
            _ok_contract(case_weights=((CASE_A, 1.0),), aggregate="geomean-ish")
        self.assertIn("aggregate", str(ctx.exception))

    def test_the_vocabulary_has_one_source(self):
        """词表就是那三个（设计里的 `geomean` 不单列: 它是权重全 1 的加权几何平均）"""
        self.assertEqual(("weighted_geomean", "mean", "worst_case"), CASE_AGGREGATES)

    def test_a_declared_task_reaches_the_frozen_contract(self):
        task = _task_with(required_cases=(CASE_A,), max_regression={CASE_A: 0.05},
                          case_weights={CASE_A: 2.0, CASE_B: 1.0},
                          aggregate=CASE_AGGREGATE_MEAN)
        c = freeze_from_task(task)
        self.assertEqual(((CASE_A, 2.0), (CASE_B, 1.0)), c.case_weights)
        self.assertEqual(CASE_AGGREGATE_MEAN, c.aggregate)

    def test_the_task_channel_is_additive(self):
        plain = dict(_task().to_dict()["constraints"])
        declared = dict(_task_with(case_weights={CASE_A: 1.0},
                                  aggregate=CASE_AGGREGATE_MEAN).to_dict()["constraints"])
        for payload in (plain, declared):
            payload.pop("case_weights", None)
            payload.pop("aggregate", None)
        self.assertEqual(plain, declared, "没声明与声明了只差那两个键")


class TestTheAggregateIsThreeState(unittest.TestCase):
    """B 组: 目标函数的判据面 —— 六条未确立入口 + 三个口径的算式"""

    W = ((CASE_A, 3.0), (CASE_B, 1.0))
    S = {CASE_A: 4.0, CASE_B: 1.0}

    def test_the_weighted_geomean_matches_the_closed_form(self):
        value, why = aggregate_speedup(self.W, self.S, "weighted_geomean")
        self.assertAlmostEqual(4.0 ** 0.75, value, places=9)
        self.assertIn("加权几何平均", why[0])

    def test_equal_weights_is_the_plain_geomean(self):
        value, _ = aggregate_speedup(((CASE_A, 1.0), (CASE_B, 1.0)),
                                     {CASE_A: 4.0, CASE_B: 9.0}, "weighted_geomean")
        self.assertAlmostEqual(6.0, value, places=9)

    def test_the_mean_is_weighted(self):
        value, _ = aggregate_speedup(self.W, self.S, "mean")
        self.assertAlmostEqual(3.25, value, places=9)   # (3*4 + 1*1) / 4

    def test_the_worst_case_reads_the_worst_one(self):
        """设计里 `max_regret` 要挡的方向: 总体好看不能掩盖最差的那一条"""
        value, why = aggregate_speedup(self.W, {CASE_A: 9.0, CASE_B: 0.5}, "worst_case")
        self.assertAlmostEqual(0.5, value, places=9)
        self.assertIn("最差", why[0])

    def test_no_weights_is_not_equal_weights(self):
        value, why = aggregate_speedup((), self.S, "weighted_geomean")
        self.assertIsNone(value, "没声明权重 ≠ 等权")
        self.assertIn("case_weights", why[0])

    def test_no_aggregate_is_not_the_default(self):
        value, why = aggregate_speedup(self.W, self.S, "")
        self.assertIsNone(value, "没声明口径 ≠ 加权几何平均")
        self.assertIn("aggregate", why[0])

    def test_an_unknown_aggregate_is_unestablished(self):
        self.assertIsNone(aggregate_speedup(self.W, self.S, "geomean-ish")[0])

    def test_a_weighted_case_that_was_not_measured_is_unestablished(self):
        value, why = aggregate_speedup(self.W, {CASE_A: 4.0}, "mean")
        self.assertIsNone(value, "分母不全 ⇒ 这个数没有指称对象")
        self.assertIn(CASE_B, why[0])

    def test_a_non_positive_speedup_blocks_only_the_geomean(self):
        s = {CASE_A: 0.0, CASE_B: 1.0}
        value, why = aggregate_speedup(self.W, s, "weighted_geomean")
        self.assertIsNone(value, "ln 的定义域")
        self.assertIn(CASE_A, why[0])
        self.assertEqual(0.0, aggregate_speedup(self.W, s, "worst_case")[0])
        self.assertAlmostEqual(0.25, aggregate_speedup(self.W, s, "mean")[0], places=9)

    def test_another_target_metric_is_unestablished(self):
        self.assertIsNone(aggregate_speedup(self.W, self.S, "mean",
                                            target_metric="throughput")[0])

    def test_unreadable_shapes_are_unestablished(self):
        self.assertIsNone(aggregate_speedup("nope", self.S, "mean")[0])
        self.assertIsNone(aggregate_speedup(((CASE_A, 1.0, 2.0),), self.S, "mean")[0])
        self.assertIsNone(aggregate_speedup(((CASE_A, "x"),), self.S, "mean")[0])

    def test_a_zero_total_weight_is_unestablished(self):
        """契约层已经挡了这一手；纯函数自己也要失败关闭（它可能被别处直接调）"""
        value, why = aggregate_speedup(((CASE_A, 0.0),), self.S, "mean")
        self.assertIsNone(value)
        self.assertIn("分母", why[0])

class TestTheCaseLedgerInTheVerifier(unittest.TestCase):
    """C 组: 只传一组不发布（旧行为不变）；多组逐 case 一条；同形状是同一条 case"""

    def _verify(self, sets):
        v = Verifier(MockGPU())
        with contextlib.redirect_stdout(io.StringIO()):
            res = v.verify(CODE, _task(), sets, None, target_gflops=0.0)
        return res[-1].metrics or {}

    def test_a_single_input_set_publishes_no_ledger(self):
        m = self._verify(_inputs(32))
        self.assertNotIn("cases", m, "只传一组 = 与 I92 之前一字不差")

    def test_multiple_input_sets_publish_one_entry_per_case(self):
        m = self._verify([_inputs(32), _inputs(16)])
        self.assertEqual({case_key(_inputs(32)), case_key(_inputs(16))}, set(m["cases"]))

    def test_the_primary_case_is_not_re_measured(self):
        m = self._verify([_inputs(32), _inputs(16)])
        first = m["cases"][case_key(_inputs(32))]
        self.assertEqual(m.get("speedup_vs_baseline"), first.get("speedup_vs_baseline"))
        self.assertIn("basis", first, "主 case 的来历要写出来（它就是 L3 那个读数）")

    def test_an_extra_case_carries_its_own_baseline(self):
        m = self._verify([_inputs(32), _inputs(16)])
        extra = m["cases"][case_key(_inputs(16))]
        self.assertIsNotNone(extra.get("speedup_vs_baseline"))
        self.assertIn("baseline_time_ms", extra, "逐 case 的加速比要有自己的分母")

    def test_two_sets_with_the_same_shape_are_one_case(self):
        m = self._verify([_inputs(32), _inputs(16), _inputs(16)])
        self.assertEqual(2, len(m["cases"]), "身份 = 形状签名 ⇒ 同形状不重复记")


class TestThroughARealRun(unittest.TestCase):
    """D 组: 多组输入 ⇒ 账本 ⇒ 两条判据 ⇒ 目标函数 ⇒ 报告（一条链全在）"""

    def _run(self, sets, *, required=(), limits=None, weights=None,
             aggregate="weighted_geomean"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                      llm=_Llm(), max_optimize_rounds=1)
        c.store.output_dir = Path(tmp.name)
        task = _task_with(required_cases=tuple(required),
                          max_regression=dict(limits or {}),
                          case_weights=dict(weights or {}), aggregate=aggregate)
        with contextlib.redirect_stdout(io.StringIO()):
            art = c.execute(task, sets, None, target_gflops=0.0)
        return art, (art.benchmark_data or {})

    def test_the_ledger_covers_every_case(self):
        a, b = case_key(_inputs(32)), case_key(_inputs(16))
        _, m = self._run([_inputs(32), _inputs(16)], required=(a, b),
                         limits={a: LOOSE_LIMIT, b: LOOSE_LIMIT}, weights={a: 3.0, b: 1.0})
        self.assertEqual({a, b}, set(m["cases"]))

    def test_both_required_cases_are_established(self):
        a, b = case_key(_inputs(32)), case_key(_inputs(16))
        _, m = self._run([_inputs(32), _inputs(16)], required=(a, b),
                         limits={a: LOOSE_LIMIT, b: LOOSE_LIMIT}, weights={a: 3.0, b: 1.0})
        d = m["implementation_decision"]
        self.assertIs(True, d["required_cases_ok"])
        self.assertIs(False, d["regression_over_limit"])
        self.assertIn("2 条必判 case", " ".join(d["reasons"]))

    def test_the_objective_is_the_declared_aggregate_of_the_ledger(self):
        """产物里的那个数要能从账本**手工复算**（不是自己编的第二个数）"""
        a, b = case_key(_inputs(32)), case_key(_inputs(16))
        _, m = self._run([_inputs(32), _inputs(16)], required=(a, b),
                         limits={a: LOOSE_LIMIT, b: LOOSE_LIMIT}, weights={a: 3.0, b: 1.0})
        led = m["cases"]
        manual = math.exp(sum(w * math.log(led[k]["speedup_vs_baseline"])
                              for k, w in ((a, 3.0), (b, 1.0))) / 4.0)
        obj = m["objective"]
        self.assertEqual("weighted_geomean", obj["aggregate"])
        self.assertEqual(2, obj["evaluated_cases"])
        self.assertEqual(2, obj["weighted_cases"])
        self.assertAlmostEqual(manual, obj["speedup"], places=3)

    def test_a_declared_case_that_was_not_evaluated_is_unestablished(self):
        a, b = case_key(_inputs(32)), case_key(_inputs(16))
        _, m = self._run([_inputs(32)], required=(a, b), limits={a: LOOSE_LIMIT},
                         weights={a: 1.0})
        d = m["implementation_decision"]
        self.assertIsNone(d["required_cases_ok"], "声明了但没评过 ⇒ 未确立")
        self.assertIn(b, " ".join(d["reasons"]))
        self.assertEqual("inconclusive", d["decision"])

    def test_an_undeclared_extra_case_does_not_block(self):
        """只声明了 A, 却传了两组: A 的判据照常成立, B 只在账本里（不许自锁）"""
        a, b = case_key(_inputs(32)), case_key(_inputs(16))
        _, m = self._run([_inputs(32), _inputs(16)], required=(a,),
                         limits={a: LOOSE_LIMIT}, weights={a: 1.0})
        self.assertEqual({a, b}, set(m["cases"]))
        self.assertIs(True, m["implementation_decision"]["required_cases_ok"])
        self.assertIs(False, m["implementation_decision"]["regression_over_limit"])

    def test_the_report_carries_the_objective_and_its_boundary(self):
        a, b = case_key(_inputs(32)), case_key(_inputs(16))
        art, _ = self._run([_inputs(32), _inputs(16)], required=(a, b),
                           limits={a: LOOSE_LIMIT, b: LOOSE_LIMIT}, weights={a: 3.0, b: 1.0})
        txt = (Path(art.saved_dir) / REPORT_NAME).read_text(encoding="utf-8")
        self.assertIn("目标函数 (§6.2)", txt)
        self.assertIn("weighted_geomean", txt)
        self.assertIn("独立确认与入库 score", txt, "声明的目标函数贯穿性能门、确认和入库")


class TestOneProducerPerFact(unittest.TestCase):
    """E 组: 结构 —— 一个事实一处生产（账本、聚合、读取各一处）"""

    def _src(self, rel):
        return (Path(__file__).resolve().parents[1] / rel).read_text(encoding="utf-8")

    def test_the_ledger_has_one_producer_and_two_declared_exits(self):
        v = self._src("operator_agent/verifier.py")
        self.assertEqual(1, v.count("def _publish_case_ledger("))
        self.assertEqual(2, v.count("self._publish_case_ledger("),
                         "两条有主 case 读数的出口: 远端 / 本机有参考实现")
        self.assertEqual(1, v.count("def _bench_case("),
                         "主 case 与账本必须共用同一对调用（协议漂了就会比出两种数）")
        self.assertEqual(1, v.count("self._bench_case("),
                         "所有 case 都进入同一 L3 方法，不能另写简化测量路")

    def test_the_aggregate_has_one_derivation(self):
        self.assertEqual(1, self._src("operator_agent/eligibility.py")
                         .count("def aggregate_speedup("))

    def test_the_ledger_reader_has_one_definition_and_no_second_copy(self):
        """收口在判据层（I95 搬家: 评测侧的门也要读这份账, 而门里不 import 调用方）"""
        el = self._src("operator_agent/eligibility.py")
        self.assertEqual(1, el.count("def case_speedups("))
        self.assertEqual(1, el.count("def performance_verdict("))
        for rel in ("operator_agent/conductor.py", "operator_agent/verifier.py"):
            self.assertNotIn("def case_speedups(", self._src(rel), rel)
        self.assertIn("case_speedups(metrics)", self._src("operator_agent/conductor.py"))

    def test_the_reporter_derives_nothing(self):
        rp = self._src("operator_agent/report.py")
        for forbidden in ("aggregate_speedup(", "case_key(", "regression_over_limit("):
            self.assertNotIn(forbidden, rp, forbidden)

    def test_the_harness_declares_the_weight_of_the_case_it_evaluates(self):
        hp = self._src("benchmarks/harness_pass_rate.py")
        self.assertIn("case_weights={_case: SINGLE_CASE_WEIGHT}", hp)
        self.assertIn("aggregate=CASE_AGGREGATE", hp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
