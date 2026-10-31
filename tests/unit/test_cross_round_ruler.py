"""跨轮的尺是不是同一把 —— #87 第 2 片 (arXiv:2609.35140v1 行 150 那条 𝒮 immutable)

外部一手（行 150，逐字）：
    "𝒮 is the **immutable** evaluation specification ... It keeps correctness and
     performance criteria fixed across attempts."

第 1 片（commit 2b6a390）管的是「一场之内、几组输入之间」；本片管「一次运行之内、几轮之间」。
两片的规则与键名**住在同一个地方**（`measurement.GATE_PARAM_KEYS` / `single_ruler`），本片把
第 1 片原本长在 `verifier` 里的那份上收到 `measurement` —— 跨轮那半必须用同一个集合，两处各写
一遍就是两套口径（CHECKLOG I22/H4）。

现场是**部署漂移**：远端那路的载荷按 `_REMOTE_L2_METRIC_KEYS` 逐键过滤透传，`atol/rtol/rho` 是
**远端脚本自己算完报上来的**（`remote_scripts/run_kernel.py:633-635`）。所以中途换过部署版本 ⇒
第 1 轮报 1e-4、第 5 轮报 1e-3，而控制端除了这份自报之外**没有别的通道**能知道。这也是为什么
本片要真跑一次 Conductor（见 `TestThroughARealRun`）而不是只喂合成 dict：合成 dict 证不了
「那条路真的把远端的自报搬到了血缘里」。

三条判据形状，每条对应一种本仓已经犯过的错：
  · **三态**：`ruler_same` 是 `True / False / None`。把 None 折成 True 是「拿缺证据当通过」，
    折成 False 是「拿缺证据当反证」（#53 那一族：`speedup=None` 被印成「没有加速」）。
  · **逐键**：atol 变了而 rtol 一致时，rtol 的结论照说（第 1 片那条 M6 的跨轮版本）。
  · **只观测不判定**：跨轮换尺今天**没有阈可判**，处置与环境漂移同一条 —— 报出来，不据此
    判核失败。所以下面有一条用例专门钉住「打了这一行之后 L2 的判决没变」。

变异标定（2026-09-30 实跑，基线 29 passed，**12/12 全部拦住、无逃逸**；每条只改一处、跑完
立刻逐字节复原并断言与改前一致。按 #83 那条规矩，这里每一行的红名单里都有「能指出哪句判据
被打歪了」的证人，不是只有结构性断言）：
    M1  可比阈值 `len(v) >= 2` → `>= 1`                    → 4 红
    M2  把「无从判定」折成 True（缺证据当通过）                 → 5 红
    M3  把「无从判定」折成 False（缺证据当反证）                → 6 红
    M4  半覆盖不再可见（`ruler_keys_partial` 那半句写死）        → 2 红
    M5  血缘抄走整个 metrics（把 `max_error` 这类聚合量当尺）    → 4 红
    M6  `self_conflict` 记成 `unreported`（两个桶合并）       → 3 红
    M7  `self_conflict` 记成 `reported`（内部打架与漂移混格）    → 3 红
    M8  conductor 血缘里换个键名（产它的与读它的不同源）          → 4 红  ← 只有真跑那组杀得掉
    M9  报表「测不到」那一支反过来（没原料时印一个像结论的 0）      → 2 红
    M10 报警行条件反转（一致时印、不一致时沉默）                 → 1 红
    M11 `record_rounds` 退回白名单（这项丢在落盘之前）          → 1 红
    M12 打架那一轮连已发布的一致键一起丢（整轮沉默）              → 1 红
⚠️ M10 只有**正向对照**（`test_two_rounds_of_the_same_deployed_table_...` 那条
`assertNotIn("[Ruler]")`）杀得掉：把条件写成 `is True or not self_conflict` 之后，漂移那一轮
照样会打印（`not 0` 为真），所以「报警的那条用例」对它是绿的。这与第 1 片那条 M8 是同一句话
—— 只测「该响的时候响」的判据，「不该响的时候别响」那一侧没人管。
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.drivers.harness_pass_rate import (aggregate, format_report,  # noqa: E402
                                          record_rounds)
from operator_agent.core import tolerance  # noqa: E402
from operator_agent.app.conductor import Conductor, _round_ruler  # noqa: E402
from operator_agent.core.kernel_db import KernelDatabase  # noqa: E402
from operator_agent.decide.measurement import (  # noqa: E402
    GATE_PARAM_KEYS, LINEAGE_RULER_KEY, RULER_CONFLICT, RULER_REPORTED,
    RULER_UNREPORTED, cross_round_ruler, l2_ruler_reading, ruler_note)
from operator_agent.execute.mock_gpu import MockGPU  # noqa: E402
from operator_agent.core.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity,
    TaskSchema, TaskType, VerificationLevel, VerificationResult)

ATOL = "atol"
RTOL = "rtol"
RHO = "required_matched_ratio"


def _reading(state=RULER_REPORTED, **values) -> dict:
    """血缘里一轮那一项的形状"""
    return {LINEAGE_RULER_KEY: dict({"state": state}, **values)}


def _round(label, reading=None, levels=None) -> dict:
    rec = {"round": label, "levels_passed": dict(levels or {})}
    if reading is not None:
        rec.update(reading)
    return rec


class TestTheRule(unittest.TestCase):
    """纯函数：什么才配说「各轮同一把尺」"""

    def test_two_rounds_with_identical_values_are_comparable_and_agree(self):
        hist = [_round(1, _reading(**{ATOL: 1e-4, RTOL: 1e-4, RHO: 1.0})),
                _round(2, _reading(**{ATOL: 1e-4, RTOL: 1e-4, RHO: 1.0}))]
        agg = cross_round_ruler(hist)
        self.assertIs(True, agg["ruler_same"])
        self.assertEqual({}, agg["ruler_changed"])
        self.assertEqual(2, agg["ruler_rounds_reported"])
        self.assertEqual(2, agg["ruler_rounds_total"])

    def test_a_changed_value_is_flagged_with_both_readings_as_witness(self):
        """现场：远端脚本中途换了版本 ⇒ 第 2 轮自报的 atol 不一样"""
        hist = [_round(1, _reading(**{ATOL: 1e-4, RTOL: 1e-4, RHO: 1.0})),
                _round(2, _reading(**{ATOL: 1e-3, RTOL: 1e-4, RHO: 1.0}))]
        agg = cross_round_ruler(hist)
        self.assertIs(False, agg["ruler_same"])
        self.assertEqual({1: 1e-4, 2: 1e-3}, agg["ruler_changed"][ATOL],
                         "报警要带「哪一轮报了多少」, 读的人不必重跑")
        self.assertEqual({"atol"}, set(agg["ruler_changed"]))

    def test_the_comparison_is_per_key_not_per_round(self):
        """M4/M12：rtol 全程一致 ⇒ 就算 atol 变了, rtol 那句也还得说

        「一个键打架就让整轮沉默」是第 1 片那条 M6（全有或全无）的跨轮版本。
        """
        hist = [_round(1, _reading(**{ATOL: 1e-4, RTOL: 1e-4})),
                _round(2, _reading(**{ATOL: 1e-3, RTOL: 1e-4}))]
        agg = cross_round_ruler(hist)
        self.assertIs(False, agg["ruler_same"], "变了就是变了, 不许因为别的键一致而抹平")
        self.assertNotIn(RTOL, agg["ruler_changed"])
        self.assertEqual((), agg["ruler_keys_partial"])

    def test_one_round_only_is_neither_agreeing_nor_drifting(self):
        """M2/M3 的那一侧：只有一轮报了尺 ⇒ 无从判定, 不是「一致」也不是「变过」"""
        for folded in ({ATOL: 1e-4}, {}):
            hist = [_round(1, _reading(**folded))] if folded else [_round(1, _reading())]
            agg = cross_round_ruler(hist)
            self.assertIsNone(agg["ruler_same"],
                              f"没有可比对时不许给布尔结论: {agg}")

    def test_a_round_that_died_at_l1_is_unreported_not_agreeing(self):
        """`verify()` 是 fail-fast：L1 挂时 L2 那一格根本不存在

        把它当「这一轮的尺与上一轮相同」就是拿缺证据当通过；把它当「换了尺」是拿缺证据当
        反证。两个方向都不许, 所以它进 `ruler_rounds_unreported` 这个单独的计数。
        """
        hist = [_round(1, _reading(**{ATOL: 1e-4}), levels={VerificationLevel.L1_COMPILE.value: True,
                                                            VerificationLevel.L2_FUNCTION.value: True}),
                _round(2, _reading(RULER_UNREPORTED), levels={VerificationLevel.L1_COMPILE.value: False}),
                _round(3, _reading(**{ATOL: 1e-4}), levels={VerificationLevel.L1_COMPILE.value: True,
                                                            VerificationLevel.L2_FUNCTION.value: True})]
        agg = cross_round_ruler(hist)
        self.assertEqual(1, agg["ruler_rounds_unreported"])
        self.assertEqual(2, agg["ruler_rounds_reported"])
        self.assertIs(True, agg["ruler_same"], "第 2 轮没报不该被读成漂移")

    def test_recorded_counts_add_up_to_the_rounds_that_carry_the_field(self):
        """三态必须互斥且穷尽 —— 分母对不上时后面每个百分数都是错的"""
        hist = [_round(1, _reading(**{ATOL: 1e-4})),
                _round(2, _reading(RULER_CONFLICT, conflicted_keys=[ATOL])),
                _round(3, _reading(RULER_UNREPORTED)),
                _round(4)]
        agg = cross_round_ruler(hist)
        self.assertEqual(4, agg["ruler_rounds_total"])
        self.assertEqual(3, agg["ruler_rounds_recorded"], "第 4 轮压根没有这项")
        self.assertEqual(agg["ruler_rounds_recorded"],
                         agg["ruler_rounds_reported"] + agg["ruler_rounds_self_conflict"]
                         + agg["ruler_rounds_unreported"])

    def test_self_conflict_is_counted_separately_from_cross_round_drift(self):
        """M6：「这一轮自己说不清用了哪把尺」与「这一轮换了尺」是两件不同的事

        前者是第 1 片那道守卫的产物（`metrics.gate_params_conflict`），本片给它第一个程序侧读者；
        后者才是跨轮漂移。修法也不同：前者查多组输入的口径，后者查部署。
        """
        hist = [_round(1, _reading(**{ATOL: 1e-4, RTOL: 1e-4})),
                _round(2, _reading(RULER_CONFLICT, conflicted_keys=[ATOL], **{RTOL: 1e-4}))]
        agg = cross_round_ruler(hist)
        self.assertEqual(1, agg["ruler_rounds_self_conflict"])
        self.assertEqual(1, agg["ruler_rounds_reported"], "self_conflict 不许混进 reported")
        self.assertEqual(0, agg["ruler_rounds_unreported"], "说不清 ≠ 没报, 两个桶不许合并")
        # atol 只有第 1 轮报了 (第 2 轮自己就说不清), rtol 两轮都报了且一致 ⇒ 谓词按**逐键**
        # 给结论: 可比的键一致就是 True, 而半覆盖的 atol 必须出现在 partial 里
        self.assertIs(True, agg["ruler_same"])
        self.assertEqual((ATOL,), agg["ruler_keys_partial"],
                         "atol 只被一轮报过这件事必须可见, 否则「一致」会被读成覆盖全了")
        self.assertIn(ATOL, ruler_note(agg), "报警那句得把半覆盖念出来")

    def test_keys_reported_by_part_of_the_rounds_are_flagged_partial(self):
        """远端逐键过滤那一侧的跨轮版本：某一轮少报一个键时, 那个键的结论只有半覆盖"""
        hist = [_round(1, _reading(**{ATOL: 1e-4, RTOL: 1e-4, RHO: 1.0})),
                _round(2, _reading(**{ATOL: 1e-4})),
                _round(3, _reading(**{ATOL: 1e-4, RTOL: 1e-4, RHO: 1.0}))]
        agg = cross_round_ruler(hist)
        self.assertIs(True, agg["ruler_same"])
        # 谓词自己按 `GATE_PARAM_KEYS` 的顺序排, 不在用例里再 sorted 一遍 (那会把顺序这件事
        # 从断言里抹掉, 而顺序正是「报表印出来的名单稳不稳定」的一部分)
        self.assertEqual((RTOL, RHO), agg["ruler_keys_partial"])
        self.assertEqual(3, agg["ruler_key_coverage"])

    def test_non_hashable_and_odd_shapes_do_not_raise(self):
        """观测性判据没有权利炸掉主流程（第 1 片同一条：值可能不是标量）"""
        hist = [_round(1, _reading(**{ATOL: (1e-4, 1e-5)})),
                _round(2, _reading(**{ATOL: (1e-4, 1e-5)})),
                _round(3), {"round": 4}, None]
        agg = cross_round_ruler(hist)
        self.assertIs(True, agg["ruler_same"])
        self.assertEqual(5, agg["ruler_rounds_total"], "畸形血缘也只计入分母, 不抛")

    def test_empty_lineage_is_reported_as_unmeasurable_not_agreeing(self):
        agg = cross_round_ruler([])
        self.assertIsNone(agg["ruler_same"])
        self.assertEqual(0, agg["ruler_rounds_recorded"])
        self.assertEqual("", ruler_note(agg), "没原料时不许印一句像结论的话")


class TestReadingFromVerifierMetrics(unittest.TestCase):
    """`l2_ruler_reading`：一轮的 L2 metrics 收成血缘那一项

    M10 的落点在这一层：读了不该读的东西（把跨组聚合量当尺）就会把 `max_error` 也搬进血缘,
    于是「尺变过」会跟着**候选代码**一起变 —— 那是本仓那一族「分支判别键选错」的镜像。
    """

    def test_only_the_three_ruler_keys_are_copied(self):
        m = {ATOL: 1e-4, RTOL: 1e-4, RHO: 1.0,
             "max_error": 3e-8, "matched_ratio": 1.0, "bad_elements": 0,
             "num_input_sets": 2, "oracle_rho_max": 1.7}
        got = l2_ruler_reading(m)
        self.assertEqual({"state", ATOL, RTOL, RHO}, set(got),
                         f"血缘里混进了不属于尺的键: {set(got) - {'state'}}")

    def test_a_round_that_published_a_conflict_keeps_both_the_flag_and_its_values(self):
        m = {RTOL: 1e-4, "gate_params_conflict": {ATOL: {0: 1e-4, 1: 5e-2}}}
        got = l2_ruler_reading(m)
        self.assertEqual(RULER_CONFLICT, got["state"])
        self.assertEqual([ATOL], got["conflicted_keys"])
        self.assertEqual(1e-4, got[RTOL], "rtol 说得清这件事得留着, 跨轮那侧才用得上的")

    def test_missing_or_non_dict_metrics_are_unreported(self):
        for bad in (None, {}, [], {"max_error": 1e-3}):
            self.assertEqual({"state": RULER_UNREPORTED}, l2_ruler_reading(bad),
                             f"{bad!r} 时造一个假读数就是把缺证据当证据")

    def test_the_three_states_are_exactly_the_three_the_predicate_counts(self):
        """成对：状态字符串不是私有方言, 谓词数的那三个桶就是这里产的那三个"""
        states = {l2_ruler_reading({ATOL: 1e-4})["state"],
                  l2_ruler_reading({"gate_params_conflict": {ATOL: {}}})["state"],
                  l2_ruler_reading(None)["state"]}
        self.assertEqual({RULER_REPORTED, RULER_CONFLICT, RULER_UNREPORTED}, states)
        agg = cross_round_ruler([_round(n, _reading(s))
                                 for n, s in enumerate(sorted(states), 1)])
        self.assertEqual(0, agg["ruler_rounds_total"] - agg["ruler_rounds_recorded"])


class TestNoteIsNotAJudgement(unittest.TestCase):
    """`ruler_note`：给人读的那一行, 三个分支都要单独被测出来（I19 的两侧取样）"""

    def test_agreeing_rounds_do_not_produce_a_warning(self):
        agg = cross_round_ruler([_round(1, _reading(**{ATOL: 1e-4})),
                                 _round(2, _reading(**{ATOL: 1e-4}))])
        note = ruler_note(agg)
        self.assertIn("各轮的尺一致", note)
        self.assertNotIn("⚠️", note)
        self.assertIn("报了尺的轮 2", note, "分母要跟着一起出来")

    def test_drift_produces_a_warning_naming_the_key_and_the_values(self):
        agg = cross_round_ruler([_round(1, _reading(**{ATOL: 1e-4})),
                                 _round(2, _reading(**{ATOL: 1e-3}))])
        note = ruler_note(agg)
        self.assertIn("⚠️", note)
        self.assertIn("不一致", note)
        self.assertIn(ATOL, note)
        self.assertIn("不据此判核失败", note, "这一句必须自我声明不影响判决")

    def test_uncomparable_says_so_instead_of_claiming_agreement(self):
        agg = cross_round_ruler([_round(1, _reading(**{ATOL: 1e-4}))])
        note = ruler_note(agg)
        self.assertIn("无从比对", note)
        self.assertNotIn("各轮的尺一致", note)
        self.assertNotIn("⚠️", note, "没证据时连报警都不该有, 否则信号会被读成缺陷")


class TestThroughARealRun(unittest.TestCase):
    """真跑 Conductor + MockGPU：合成 dict 证不了「远端自报的那三个键真的过得了落盘边界」"""

    PYTHON_MATMUL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
    SKETCH = {"algorithm": "matmul",
              "tiling": {"block_m": 32, "block_n": 32, "block_k": 16},
              "memory_plan": {}, "pipeline": [], "optimization_hints": []}

    class _Llm:
        def chat(self, messages, **kw):
            return TestThroughARealRun.PYTHON_MATMUL

        def chat_json(self, messages, **kw):
            return dict(TestThroughARealRun.SKETCH)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._table = dict(tolerance.TOLERANCE_BY_PRECISION)
        self.addCleanup(lambda: tolerance.TOLERANCE_BY_PRECISION.update(self._table))

    def _conductor(self, rounds=2):
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                      llm=TestThroughARealRun._Llm(), max_optimize_rounds=rounds)
        c.store.output_dir = Path(self.tmp.name)
        return c

    @staticmethod
    def _task():
        return TaskSchema(
            task_id="ruler-case",
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
            target=TargetSpec(hardware="mock_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=5.0,
                                       max_error=1e-3))

    def _run(self, drift_after_first_round=False) -> tuple:
        c = self._conductor()
        calls = []
        orig = c.verifier._verify_function

        def wrapped(*a, **kw):
            calls.append(1)
            # 「第 2 轮起部署的表换了」—— 现场是远端脚本换了版本, 这里换的是同一份**自报来源**
            if drift_after_first_round and len(calls) == 2:
                tolerance.TOLERANCE_BY_PRECISION["fp32"] = (1e-3, 1e-4, 1.0)
            return orig(*a, **kw)

        c.verifier._verify_function = wrapped
        inputs = {"A": np.ones((24, 16), np.float32), "B": np.ones((16, 24), np.float32)}
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            art = c.execute(self._task(), inputs, None, target_gflops=1.0)
        saved = Path(art.saved_dir)
        result = json.loads((saved / "result.json").read_text(encoding="utf-8"))
        return result["iteration_history"], buf.getvalue(), saved

    def test_the_real_pipeline_writes_the_ruler_into_lineage_with_real_values(self):
        """绝对锚点：1e-4/1e-4/1.0 是**独立算出来**的（fp32 表 + min(表, max_error=1e-3)）,
        不是从代码里读的。跑完再核一次这条链, 才知道血缘里那个数确实来自数值门。"""
        history, _, _ = self._run()
        self.assertGreaterEqual(len(history), 2, "构造前提: 这确实是跨轮的两次验证")
        first = history[0][LINEAGE_RULER_KEY]
        self.assertEqual(RULER_REPORTED, first["state"], first)
        self.assertEqual({ATOL: 1e-4, RTOL: 1e-4, RHO: 1.0},
                         {k: first[k] for k in GATE_PARAM_KEYS},
                         "血缘里的尺必须是这一轮真的用来判 L2 的那一把")

    def test_the_same_round_also_writes_the_performance_reading(self):
        """#87 第 3 片: 同一条落盘路径必须把**两半**都写进血缘 —— 这一条是 P7 的证人。

        为什么单独一条: 上一那条只查 L2 那一格, 于是「conductor 忘了写 `l3_ruler`」这件事
        在整张测试网里是**不可见的** (变异 P7 第一版 0 红就是它)。而这一格一丢, 报表那一栏
        只会退成「测不到」—— 一个听起来已经尽职的词, 所以它必须有一条从**产它的那一侧**
        出发的用例, 而不是从读它的那一侧。
        """
        from operator_agent.decide.measurement import (LINEAGE_PERF_RULER_KEY,
                                                PERF_CRITERION_SPEEDUP,
                                                PERF_GATE_PARAM_KEYS)
        history, _, _ = self._run()
        self.assertTrue(all(LINEAGE_PERF_RULER_KEY in r for r in history),
                        "每一轮都要有性能那一半的格子, 缺一个就不是「每轮都量过」")
        states = [r[LINEAGE_PERF_RULER_KEY]["state"] for r in history]
        self.assertTrue(set(states) <= {RULER_REPORTED, RULER_UNREPORTED},
                        f"性能那半结构上没有 self_conflict (L3 只拿第一组输入): {states}")
        reported = [r[LINEAGE_PERF_RULER_KEY] for r in history
                    if r[LINEAGE_PERF_RULER_KEY]["state"] == RULER_REPORTED]
        self.assertTrue(reported, "这几次运行应当至少有一轮真的走到过 L3, 否则这一条是空转")
        for r in reported:
            self.assertIn(r["perf_criterion"], (PERF_CRITERION_SPEEDUP,),
                          "mock_gpu 走的是加速比那条 (没有远端载荷, 也不该掉进 legacy 那条)")
            self.assertEqual(set(PERF_GATE_PARAM_KEYS) - {  # 三个键都该在场才叫"报了尺"
                k for k in PERF_GATE_PARAM_KEYS if k in r}, set(), r)

    def test_two_rounds_of_the_same_deployed_table_are_reported_as_agreeing(self):
        """正向对照（不发放）：守卫不许把今天的主路径判成漂移

        M10 的另一半 —— 它也是唯一能给「报警行」派活的用例：把打印条件反过来就红。
        """
        history, log, _ = self._run()
        agg = cross_round_ruler(history)
        self.assertIs(True, agg["ruler_same"], agg)
        self.assertEqual(0, agg["ruler_rounds_self_conflict"])
        self.assertNotIn("[Ruler]", log, "一致的时候不该占日志")

    def test_a_round_that_changed_its_ruler_is_flagged_without_changing_the_verdict(self):
        """本片的存在理由：换部署版本要被看见, 而**看见之后不改判决**（观测先于判定）"""
        history, log, _ = self._run(drift_after_first_round=True)
        agg = cross_round_ruler(history)
        self.assertIs(False, agg["ruler_same"], agg)
        self.assertEqual({1: 1e-4, 2: 1e-3}, agg["ruler_changed"][ATOL])
        self.assertNotIn(RTOL, agg["ruler_changed"], "rtol 没换, 不许陪着报警")
        self.assertIn("[Ruler]", log)
        self.assertIn("不据此判核失败", log)
        # 判决没被这一片动过: 两轮该过的还是过了
        for rec in history[:2]:
            self.assertIs(True, rec["levels_passed"][VerificationLevel.L2_FUNCTION.value],
                          rec)

    def test_the_snapshot_carries_the_ruler_so_resume_does_not_lose_it(self):
        """续跑那一路的原料也在血缘里: 快照与 result.json 必须是同一份"""
        history, _, saved = self._run()
        snap = json.loads((saved / "execution_snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(history, snap["iteration_history"],
                         "续跑读的是快照那份血缘 —— 两边不是同一份的话, 恢复之后就丢了尺读数")
        self.assertTrue(all(LINEAGE_RULER_KEY in r for r in snap["iteration_history"]),
                        "每一轮都要带这项, 否则续跑后第一次判定就没有原料")


class TestTheReportHasAReader(unittest.TestCase):
    """第二个读者：端到端报表那一栏（第 1 个是 conductor 每轮那句 `[Ruler]`）"""

    @staticmethod
    def _run(case, readings, extra_levels=None):
        """readings: 每轮那份**由生产者算出的**读数 (`l2_ruler_reading` 的返回);
        None 表示这一轮的血缘里根本没有这项 (旧产物)。

        这里坚持走生产者而不是手写 dict: 报表读的是生产者的输出形状, 两侧一起造就只是
        在测这个测试文件自己。
        """
        rounds = []
        for i, reading in enumerate(readings, 1):
            rec = {"round": i, "all_passed": False,
                   "levels_passed": dict(extra_levels or {
                       VerificationLevel.L1_COMPILE.value: True,
                       VerificationLevel.L2_FUNCTION.value: True})}
            if reading is not None:
                rec[LINEAGE_RULER_KEY] = reading
            rounds.append(rec)
        return {"case": case, "ok": True, "rounds": rounds,
                "final_all_passed": False, "rounds_to_pass": None,
                "passed_via": None, "admitted": False, "admit_reason": ""}

    def setUp(self):
        self.runs = [
            self._run("a-drift", [l2_ruler_reading({ATOL: 1e-4}),
                                  l2_ruler_reading({ATOL: 1e-3})]),
            self._run("b-agrees", [l2_ruler_reading({ATOL: 1e-4, RTOL: 1e-4}),
                                   l2_ruler_reading({ATOL: 1e-4, RTOL: 1e-4})]),
            self._run("c-one-round-only", [l2_ruler_reading({ATOL: 1e-4})]),
            self._run("d-old-lineage", [None, None]),
        ]
        self.agg = aggregate(self.runs)

    def test_only_runs_that_can_be_compared_enter_the_denominator(self):
        self.assertEqual((1, 2), self.agg["ruler_drift_runs"],
                         "可比运行 = 谓词给了布尔结论的那些; a 变了 / b 一致 / c 无从判定 / d 测不到")
        self.assertEqual(["a-drift → {'atol': {1: 0.0001, 2: 0.001}}"],
                         self.agg["ruler_drift_cases"],
                         "名单里要带着「哪个键、哪几轮各报了多少」, 读的人不必再去翻 result.json")
        self.assertEqual(1, self.agg["ruler_uncomparable_runs"])

    def test_conflict_rounds_are_counted_program_side(self):
        """`gate_params_conflict` 的程序侧读者（#87 第 1 片留的那笔债）"""
        self.assertEqual(0, self.agg["ruler_conflict_rounds"])
        agg = aggregate(self.runs + [self._run(
            "e-conflict", [l2_ruler_reading({"gate_params_conflict": {ATOL: {0: 1.0, 1: 2.0}}}),
                           l2_ruler_reading({ATOL: 1e-4})])])
        self.assertEqual(1, agg["ruler_conflict_rounds"])

    def test_the_report_prints_the_column_with_its_denominator(self):
        text = format_report(self.agg)
        # 第 3 片之后报表里有**两栏**「跨轮的尺」(correctness / performance 各一)。
        # 针必须落在**本半那半的标题**上, 而不是那个两栏共用的前缀 —— 只查共用的前缀时,
        # L2 那一整栏被删掉也照样绿 (I32 那一族: 词在哪都会印, 分辨不出来自哪一栏)。
        self.assertIn("【跨轮的尺 · 各轮 L2 报的 atol", text)
        self.assertIn("可比运行 (≥2 轮报了尺) 2 · 尺变过 1", text)
        self.assertIn("a-drift", text)
        self.assertIn("凑不出可比对", text)

    def test_old_lineage_is_reported_as_unmeasurable_not_as_zero_drift(self):
        """M8：「测不到」与「查过了没漂移」不是一回事 —— 后者会让读者以为这道门在起作用"""
        agg = aggregate([self._run("d-old-lineage", [None, None]),
                         self._run("d2", [None]), self._run("d3", [None]),
                         self._run("d4", [None, None])])
        self.assertFalse(agg["ruler_measurable"])
        text = format_report(agg)
        self.assertIn("测不到", text)
        self.assertNotIn("尺变过", text, "没有原料时不许印一个像结论的 0")
        self.assertIn(LINEAGE_RULER_KEY, text)

    def test_record_rounds_does_not_drop_the_field(self):
        """M9：落盘那一步是白名单犯过老病的地方（doc 48 那批真机血缘永久没了）"""
        rec = _round(1, _reading(**{ATOL: 1e-4}))
        out = record_rounds([rec])
        self.assertEqual(rec[LINEAGE_RULER_KEY], out[0][LINEAGE_RULER_KEY])


class TestTheLineageProducer(unittest.TestCase):
    """`_round_ruler`：血缘那一项的唯一生产者"""

    @staticmethod
    def _l2(metrics, passed=True):
        return VerificationResult(level=VerificationLevel.L2_FUNCTION,
                                  passed=passed, message="x", metrics=metrics)

    def test_it_picks_l2_out_of_a_three_level_result_list(self):
        l1 = VerificationResult(level=VerificationLevel.L1_COMPILE, passed=True, message="ok")
        out = _round_ruler([l1, self._l2({ATOL: 1e-4, RTOL: 1e-4, "max_error": 1e-9}),
                            VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                                               passed=False, message="慢", metrics={})])
        self.assertEqual({"state": RULER_REPORTED, ATOL: 1e-4, RTOL: 1e-4}, out)

    def test_a_round_that_never_got_an_l2_is_unreported(self):
        """fail-fast 那一侧：L1 挂时没有 L2 结果, 这不许被读成「尺与上一轮相同」"""
        l1 = VerificationResult(level=VerificationLevel.L1_COMPILE, passed=False, message="编不过")
        self.assertEqual({"state": RULER_UNREPORTED}, _round_ruler([l1]))
        self.assertEqual({"state": RULER_UNREPORTED}, _round_ruler([]))

    def test_a_failed_l2_still_reports_the_ruler_it_used(self):
        """漂移常常就出现在「换了表的那一轮正好判死」时 —— 失败侧的读数同样是证据"""
        out = _round_ruler([self._l2({ATOL: 1e-3, RTOL: 1e-4, "max_error": 3.0}, passed=False)])
        self.assertEqual({"state": RULER_REPORTED, ATOL: 1e-3, RTOL: 1e-4}, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
