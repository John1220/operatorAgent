# -*- coding: utf-8 -*-
"""#87 第 3 片: 不可变的评测规格 𝒮 覆盖到**性能**那一半。

外部一手 (arXiv:2609.35140v1 行 150, 逐字):
    "𝒮 is the **immutable** evaluation specification ... It keeps correctness **and
     performance** criteria fixed across attempts."

第 1/2 片只问了 correctness 那一半 (`atol`/`rtol`/`ρ` 跨组、跨轮是不是同一把)。这一片把
同一个问句问到 L3: `min_speedup_required` / 判的是哪个量 / roofline 绑定轴。

四层结构, 每一层都在回答一个不同的问题:

1. **分类器带着这一类** (`TestThePerfKeysAreSingleSourced`) —— 键集合、判据身份、
   `unknown` 那个占位值都只在 `measurement` 里定义一次; 抄进 verifier 或抄进报表就红。
2. **三条比对路各自声明了自己** (`TestEveryL3PathDeclaresItsCriterion`) —— 两半是**真跑**
   `_verify_performance` 拿到的 metrics (本机那条与 legacy 绝对目标那条), 远端同机基线那条
   今天在这台控制机上没有输入, 所以由 AST 守卫钉住: 凡报 `min_speedup_required` 的那个
   字面量都必须同时报 `perf_criterion`, 且值必须是 `PERF_CRITERION_*` 常量。
   **这条不对称是本片的实话**: 两条实测 + 三处结构。
3. **跨轮的判据只有一份实现** (`TestTheSameRuleServesBothHalves`) —— `cross_round_ruler`
   默认参数下的输出与显式传 L2 那组键**逐键相同**; 换判据必须两侧都改。
4. **它有读者** (`TestBothHalvesHaveReaders`) —— conductor 每轮往血缘里写 `l3_ruler`,
   尺变了才打印; 端到端报表两栏都在, 且各带自己的分母。

实跑读数 (python3.9 + numpy, MockGPU, 零 LLM / 零机时):
    18 例全绿; 变异表 `benchmarks/mut_specs/perf_ruler_across_rounds.py`
    行为分母 9/9 + 结构分母 3/3。

本片**不改任何判决**: L3 的 `passed` 今天与昨天逐字相同, 多出来的只有观测。这一点由
`test_the_verdict_of_this_slice_is_observation_only` 钉住 (与 #80 第 1 片同一条纪律)。

这一片也顺手撞出一处**量具自己的失明** (不是本片要修的东西, 但被本片撞出来了):
`tests/unit/test_bound_axis_gate.py` 那张「读者清单」用 tokenize 剔 docstring, 判据是
「STRING 后面跟着 NEWLINE」—— 于是 `BOUND_AXIS_KEY = "bound_type"` 这种**把键名收成常量**
的写法被整行当成 docstring 丢掉, 清单对这一类读者永久失明。修法与双向证人都在那个文件里,
登记 CHECKLOG I35。
"""

import ast
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from operator_agent import measurement as ms  # noqa: E402
from operator_agent.measurement import (  # noqa: E402
    BOUND_AXIS_KEY, BOUND_AXIS_UNKNOWN, LINEAGE_PERF_RULER_KEY, LINEAGE_RULER_KEY,
    PERF_CRITERION_ABSOLUTE_GFLOPS, PERF_CRITERION_SPEEDUP, PERF_GATE_PARAM_KEYS,
    RULER_REPORTED, RULER_UNREPORTED, cross_round_ruler, l3_ruler_reading, ruler_note)
from operator_agent.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType, VerificationLevel)
from operator_agent.verifier import Verifier  # noqa: E402
from operator_agent.mock_gpu import MockGPU  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VERIFIER_SRC = open(os.path.join(_ROOT, "operator_agent", "verifier.py"),
                     encoding="utf-8").read()

GOOD = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"


def _task(min_speedup=1.0, kernel_type="matmul"):
    return TaskSchema(
        task_id="t-perf-ruler",
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, max_error=1e-5,
                                   min_speedup=min_speedup),
    )


def _inputs():
    rng = np.random.default_rng(0)
    return {"A": rng.standard_normal((32, 32)).astype(np.float32),
            "B": rng.standard_normal((32, 32)).astype(np.float32)}


def _l3_local(**kw):
    """本机那条比对路 (`_measure_baseline` 用本机 reference 计时)。"""
    with contextlib.redirect_stdout(io.StringIO()):
        return Verifier(MockGPU())._verify_performance(GOOD, _inputs(), 0.0, _task(**kw))


def _l3_legacy(**kw):
    """legacy 那条: 没有可用 reference ⇒ 按绝对 GFLOPS 目标判定。

    门是 `_measure_baseline(...) is None`, 这里直接把它收成 None 而不是去找一个真没有
    参考实现的算子 —— 因为**这条路的形状**才是本片的对象, 换一个算子会连带换 roofline
    轴的来源, 那条已经由另一条用例单独钉。
    """
    v = Verifier(MockGPU())
    v._measure_baseline = lambda *a, **k: None   # 不碰真实测量, 只选分支
    with contextlib.redirect_stdout(io.StringIO()):
        return v._verify_performance(GOOD, _inputs(), 100.0, _task(**kw))


def _reading(**over):
    r = {"min_speedup_required": 1.0, "perf_criterion": PERF_CRITERION_SPEEDUP,
         BOUND_AXIS_KEY: "compute"}
    r.update(over)
    return {k: v for k, v in r.items() if v is not None}


def _rounds(*readings):
    return [{"round": i + 1, LINEAGE_PERF_RULER_KEY: r} for i, r in enumerate(readings)]


class TestThePerfKeysAreSingleSourced(unittest.TestCase):
    """第 1 层: 这一类是分类器带的, 不是调用点各自写的"""

    def test_the_key_set_is_exactly_the_three_things_l3_reads(self):
        self.assertEqual(("min_speedup_required", "perf_criterion", BOUND_AXIS_KEY),
                         PERF_GATE_PARAM_KEYS,
                         "这一组键就是「L3 的尺」的定义; 增减都要连同报表那一栏一起改")

    def test_the_criterion_identities_live_in_measurement_not_at_the_call_sites(self):
        # 抄在 verifier 里就地写字符串 = 两处口径; 这两条常量是唯一的出处
        self.assertNotEqual(PERF_CRITERION_SPEEDUP, PERF_CRITERION_ABSOLUTE_GFLOPS)
        self.assertEqual(0, _VERIFIER_SRC.count('"speedup_vs_baseline>='),
                         "verifier 里不许再出现自己拼的判据字符串")

    def test_unknown_axis_is_not_a_reading(self):
        """两轮都 `unknown` 会被逐键判据读成「同值」⇒ 那是把「谁都不知道」印成「已确认」"""
        got = l3_ruler_reading(_reading(**{BOUND_AXIS_KEY: BOUND_AXIS_UNKNOWN}))
        self.assertNotIn(BOUND_AXIS_KEY, got,
                         "占位值必须进不了读数, 否则它会在跨轮那一层被算成一次一致")
        self.assertIn(BOUND_AXIS_KEY, l3_ruler_reading(_reading()),
                      "反向证人: 真的轴照样要收进来, 否则这条规则是恒假")

    def test_a_round_that_reports_neither_key_reads_unreported_not_agrees(self):
        got = l3_ruler_reading({})
        self.assertEqual(RULER_UNREPORTED, got["state"])
        self.assertNotIn("min_speedup_required", got)


class TestEveryL3PathDeclaresItsCriterion(unittest.TestCase):
    """第 2 层: 三条比对路都得说自己判的是哪个量 —— 两条实测, 第三条只能结构地钉"""

    def test_the_local_baseline_path_publishes_all_three(self):
        r = _l3_local(min_speedup=1.25)
        m = r.metrics
        self.assertEqual(1.25, m["min_speedup_required"], "阈值要如实搬, 不许倒推")
        self.assertEqual(PERF_CRITERION_SPEEDUP, m["perf_criterion"])
        self.assertEqual(RULER_REPORTED, l3_ruler_reading(m)["state"])

    def test_the_legacy_absolute_path_publishes_its_own_threshold(self):
        """本片之前这一条**一个尺键都不报** ⇒ 它在跨轮读数里永远是「没报」, 而它其实判过"""
        m = _l3_legacy(min_speedup=1.5).metrics
        self.assertEqual(PERF_CRITERION_ABSOLUTE_GFLOPS, m["perf_criterion"])
        self.assertEqual(1.5, m["min_speedup_required"],
                         "这个旋钮在这条路上确实被用了 (乘在绝对目标上), 不报就是撒谎")

    def test_the_two_paths_are_not_tellable_apart_by_the_number_alone(self):
        """`min_speedup=1.0` 在两条路上数值相同而含义不同 —— 全靠 `perf_criterion` 分开"""
        a = _l3_local(min_speedup=1.0).metrics
        b = _l3_legacy(min_speedup=1.0).metrics
        self.assertEqual(a["min_speedup_required"], b["min_speedup_required"])
        self.assertNotEqual(a["perf_criterion"], b["perf_criterion"])
        agg = cross_round_ruler(_rounds(l3_ruler_reading(a), l3_ruler_reading(b)),
                                keys=PERF_GATE_PARAM_KEYS,
                                lineage_key=LINEAGE_PERF_RULER_KEY)
        self.assertIs(False, agg["ruler_same"], "换了判据必须读成「尺变了」")
        self.assertIn("perf_criterion", agg["ruler_changed"])

    def test_every_threshold_publication_also_declares_its_criterion(self):
        """AST 守卫: 凡报 `min_speedup_required` 的那个字面量, 必须同时报 `perf_criterion`。

        为什么查这一层而不是查 `return`: 三条比对路都是先 `metrics.update({...})` 再
        `return VerificationResult(..., metrics=metrics)`, 名字出现在 update 里而不是 return
        里 —— 一开始按 return 那一段查, 五个 L3 return 一个都不命中, 那条断言就成了恒假
        (I32 那一族: 判据的 needle 长错了地方)。
        为什么只有这一条 covering 远端那一路: 它要 `SAME_DEVICE_BASELINE` 载荷, 今天这台
        控制机上没有它的输入。**这条不对称是本片的实话**: 两条实测 + 一条结构。
        """
        tree = ast.parse(_VERIFIER_SRC)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_verify_performance")
        sites = []
        for d in ast.walk(fn):
            if not isinstance(d, ast.Dict):
                continue
            keys = [k.value for k in d.keys if isinstance(k, ast.Constant)]
            if "min_speedup_required" not in keys:
                continue
            sites.append(d.lineno)
            self.assertIn("perf_criterion", keys,
                          f"第 {d.lineno} 行报了门槛却没说判的是哪个量")
            val = d.values[keys.index("perf_criterion")]
            self.assertIsInstance(val, ast.Name,
                                  "判据必须是 `measurement` 里的常量, 就地拼字符串就是第二套口径")
            self.assertTrue(val.id.startswith("PERF_CRITERION_"), val.id)
        self.assertEqual(3, len(sites),
                         f"报门槛的地方应该是三条比对路各一处, 实数 {sites}")


class TestTheSameRuleServesBothHalves(unittest.TestCase):
    """第 3 层: 判据只有一份, 两半各喂一组键"""

    def test_the_default_call_is_still_the_correctness_half(self):
        hist = [{LINEAGE_RULER_KEY: {"state": RULER_REPORTED, "atol": 1e-4}},
                {LINEAGE_RULER_KEY: {"state": RULER_REPORTED, "atol": 1e-3}}]
        self.assertEqual(cross_round_ruler(hist),
                         cross_round_ruler(hist, keys=ms.GATE_PARAM_KEYS,
                                           lineage_key=LINEAGE_RULER_KEY),
                         "默认参数就是 L2 那半 —— 既有读者读的正是它")
        self.assertIs(False, cross_round_ruler(hist)["ruler_same"])

    def test_the_two_halves_do_not_leak_into_each_other(self):
        """L2 报了尺、L3 没报 ⇒ 两栏的分母必须各算各的 (H4: 跨口径不能折成一个率)"""
        hist = [{LINEAGE_RULER_KEY: {"state": RULER_REPORTED, "atol": 1e-4},
                 LINEAGE_PERF_RULER_KEY: {"state": RULER_UNREPORTED}},
                {LINEAGE_RULER_KEY: {"state": RULER_REPORTED, "atol": 1e-4},
                 LINEAGE_PERF_RULER_KEY: {"state": RULER_UNREPORTED}}]
        l2 = cross_round_ruler(hist)
        l3 = cross_round_ruler(hist, keys=PERF_GATE_PARAM_KEYS,
                               lineage_key=LINEAGE_PERF_RULER_KEY)
        self.assertIs(True, l2["ruler_same"])
        self.assertIsNone(l3["ruler_same"], "没报就是没报, 不许折成「一致」")
        self.assertEqual(0, l3["ruler_rounds_reported"])

    def test_a_bound_axis_flip_alone_is_enough_to_call_it_a_different_ruler(self):
        agg = cross_round_ruler(
            _rounds(l3_ruler_reading(_reading()), l3_ruler_reading(
                _reading(**{BOUND_AXIS_KEY: "memory"}))),
            keys=PERF_GATE_PARAM_KEYS, lineage_key=LINEAGE_PERF_RULER_KEY)
        self.assertIs(False, agg["ruler_same"])
        self.assertEqual({1: "compute", 2: "memory"},
                         dict(agg["ruler_changed"][BOUND_AXIS_KEY]),
                         "证人要指着「第 1 轮 compute、第 2 轮 memory」这一对")

    def test_both_readings_delegate_to_the_one_shared_implementation(self):
        """P11 的证人: 两个壳都必须把活交给 `_ruler_reading`, 自己不再长出一份选择逻辑。

        这一条只能结构地查: 把选择逻辑**照抄一份且连规则都抄对**时, 对任何输入两种写法
        给出同一个 dict ⇒ 行为层永远撞不到它。而「一份判据被抄成两份」正是本仓反复登记的
        失效起点 (I22/H4), 它的症状通常在几周之后才出现 —— 到那时已经没人记得哪份是权威。
        """
        src = open(os.path.join(_ROOT, "operator_agent", "measurement.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        checked = 0
        for name in ("l2_ruler_reading", "l3_ruler_reading"):
            fn = next((n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == name), None)
            self.assertIsNotNone(fn, f"找不到 {name} —— 匹配器要跟着被测代码一起改")
            checked += 1
            called = [n.func.id for n in ast.walk(fn)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertIn("_ruler_reading", called, f"{name} 没走共享实现")
            seg = ast.get_source_segment(src, fn) or ""
            self.assertNotIn("isinstance(", seg,
                             f"{name} 里又长出一份选择逻辑 ⇒ 两份口径开始漂")
        self.assertEqual(2, checked, "两条壳都该被查到, 一条都不许漏")

    def test_the_note_names_which_ruler_it_is_talking_about(self):
        """两行挨着印, 不带标签就没人知道说的是哪一把 (同一句话有多个副本的反面)"""
        agg = cross_round_ruler(_rounds(l3_ruler_reading(_reading()),
                                        l3_ruler_reading(_reading(min_speedup_required=2.0))),
                                keys=PERF_GATE_PARAM_KEYS,
                                lineage_key=LINEAGE_PERF_RULER_KEY)
        labelled = ruler_note(agg, ms.RULER_LABEL_PERFORMANCE)
        self.assertTrue(labelled.startswith(ms.RULER_LABEL_PERFORMANCE + ": "),
                        f"标签必须在最前面: {labelled[:40]}")
        self.assertIn("min_speedup_required", labelled, "证人要跟着那一行走")

    def test_the_verdict_of_this_slice_is_observation_only(self):
        """本片只加观测: 新引入的名字**不许出现在任何判定位置**。

        查法不是「跑一次看 passed 对不对」(那只能证明这一次没坏), 而是结构地查:
        `_verify_performance` 里每一个 `If` / `Compare` / `BoolOp` / 条件表达式里都不许
        出现 `perf_criterion` 或 `PERF_CRITERION_`。谁哪天让判决去读它, 这条立刻红 ——
        那一步属于「换判决」, 而换判决要带真机前后对照 (与 #80 第 2 片同一条纪律)。
        """
        tree = ast.parse(_VERIFIER_SRC)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_verify_performance")
        # 只查**条件那一侧**: `ast.get_source_segment(If)` 连 body 一起返回, 拿它查会把
        # 「if 的是别的事、body 里顺手 update 了 metrics」误判成读到了判定 (第一版就这么红过)。
        tests = [n.test for n in ast.walk(fn)
                 if isinstance(n, (ast.If, ast.IfExp)) and n.test is not None]
        tests += [n for n in ast.walk(fn) if isinstance(n, (ast.Compare, ast.BoolOp))]
        self.assertTrue(tests, "匹配器自己不许恒空转: 这个函数里本来就该有判定")
        offenders = []
        for node in tests:
            seg = ast.get_source_segment(_VERIFIER_SRC, node) or ""
            if "perf_criterion" in seg or "PERF_CRITERION_" in seg:
                offenders.append((node.lineno, seg.splitlines()[0][:60]))
        self.assertEqual([], offenders,
                         f"判决位置读到了本片新加的名字: {offenders}")


class TestBothHalvesHaveReaders(unittest.TestCase):
    """第 4 层: 写了就要有人读 (#40/#61 那一族的规矩)"""

    def test_the_conductor_writes_the_perf_reading_into_lineage(self):
        from operator_agent.conductor import _round_perf_ruler
        res = [type("R", (), {"level": VerificationLevel.L3_PERFORMANCE,
                              "metrics": _l3_legacy().metrics})()]
        got = _round_perf_ruler(res)
        self.assertEqual(RULER_REPORTED, got["state"])
        self.assertEqual(PERF_CRITERION_ABSOLUTE_GFLOPS, got["perf_criterion"])

    def test_a_round_without_l3_writes_unreported_not_a_copy_of_the_previous(self):
        from operator_agent.conductor import _round_perf_ruler
        self.assertEqual(RULER_UNREPORTED, _round_perf_ruler([])["state"],
                         "L1/L2 就死了的那轮没有性能尺读数 —— 而它绝不能被读成「与上轮同尺」")

    def test_the_report_prints_both_columns(self):
        sys.path.insert(0, os.path.join(_ROOT, "benchmarks", "drivers"))
        import importlib
        hpr = importlib.import_module("harness_pass_rate")
        # 报表有一道「分母不足就不出百分数」的门 (下限 4 次运行), 所以夹具至少四个 case;
        # 形状照抄 tests/unit/test_cross_round_ruler.py 的那份 (`ok`/`rounds`/`levels_passed`),
        # 自己另发明一份就会测到这个测试文件而不是报表。
        def _run(case, pairs):
            rounds = []
            for i, (l2, l3) in enumerate(pairs, 1):
                rec = {"round": i, "all_passed": False,
                       "levels_passed": {VerificationLevel.L1_COMPILE.value: True,
                                         VerificationLevel.L2_FUNCTION.value: True}}
                if l2 is not None:
                    rec[LINEAGE_RULER_KEY] = l2
                if l3 is not None:
                    rec[LINEAGE_PERF_RULER_KEY] = l3
                rounds.append(rec)
            return {"case": case, "ok": True, "rounds": rounds,
                    "final_all_passed": False, "rounds_to_pass": None,
                    "passed_via": None, "admitted": False, "admit_reason": ""}

        same2 = {"state": RULER_REPORTED, "atol": 1e-4}
        runs = [
            _run("a", [(same2, l3_ruler_reading(_reading(min_speedup_required=1.0))),
                       (same2, l3_ruler_reading(_reading(min_speedup_required=2.0)))]),
            _run("b", [({"state": RULER_UNREPORTED}, {"state": RULER_UNREPORTED})] * 2),
            _run("c", [(None, None), (None, None)]),
            _run("d", [(same2, l3_ruler_reading(_reading())),
                       (same2, l3_ruler_reading(_reading()))]),
        ]
        agg = hpr.aggregate(runs)
        text = hpr.format_report(agg)
        self.assertIn("【跨轮的尺 · 各轮 L2", text)
        self.assertIn("【跨轮的尺 · 各轮 L3", text)
        # 两半的分母各算各的 (H4: 跨口径不许折成一个率):
        #   a/d 的 L2 同 atol ⇒ 可比 2、变过 0; 而 a 的性能尺 1.0→2.0 ⇒ 可比 2、变过 1。
        #   b 两半都报了「没报」⇒ 既不记一致也不记变过, 单独进 uncomparable。
        #   c 的血缘里根本没有这两项 ⇒ 连 uncomparable 都不算, 它落在 measurable 的分母外。
        self.assertEqual((0, 2), agg["ruler_drift_runs"])
        self.assertEqual((1, 2), agg["perf_ruler_drift_runs"])
        self.assertEqual(1, agg["perf_ruler_uncomparable_runs"])
        # 上面这条我第一版写成 0, 那是**猜**的不是读的: b 那两轮两半都写了 `unreported`
        # 读数 —— 「写了但没值」在两半里是同一件事, 所以 L2 那一栏也必须各记 1 次。
        self.assertEqual(1, agg["ruler_uncomparable_runs"],
                         "两半对「报了读数但没有可比值」的处置必须同形 —— 不同形就是两套口径")
        self.assertTrue(any(c.startswith("a →") for c in agg["perf_ruler_drift_cases"]),
                        "明细那一行要能指回是哪个 case 换了尺")

    def test_the_lineage_key_name_is_not_hardcoded_in_the_report(self):
        """报表那句「测不到」要说的是**这一半**的键名; 写死 `l2_ruler` 就是两栏共用一个假出处"""
        self.assertNotIn('血缘里没有 {LINEAGE_RULER_KEY}',
                         open(os.path.join(_ROOT, "benchmarks", "drivers", "harness_pass_rate.py"),
                              encoding="utf-8").read())
        self.assertIn("_measurable_lineage_key",
                      open(os.path.join(_ROOT, "benchmarks", "drivers", "harness_pass_rate.py"),
                           encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
