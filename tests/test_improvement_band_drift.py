"""改善判据那条带宽跨轮是不是同一把 —— #87 第 4 片 (arXiv:2609.35140v1 行 150 的 performance 那一半)

外部一手 (行 150，逐字):

    "𝒮 is the **immutable** evaluation specification ... It keeps correctness and
     performance criteria fixed across attempts."

第 2 片管了 correctness 那半 (`atol/rtol/ρ`)，第 3 片管了 performance 那半的**门槛**
(`min_speedup` / 判的是哪个量 / 绑定轴)。这一片管 performance 那半里**最后一条还没被问过的尺**:
`NOISE_BAND[档位]` —— 四态判决 (FAST/SLOW/SAME/AMBIGUOUS) 真正用过的那个带宽。
今天它只活在 `why` 那句文本里 (`"+0.4% > 带宽 0.10%"`)，于是「跨轮是不是同一条带宽」这个问句
在产物里无从可查。

**这片当初被挂起来的原因**（任务 #87 里写着的）: 「它的漂移与测量质量混在一起 —— 档位降级本来就
让带宽变宽」。现在有了答案，而且是**两种归因分开数**:

| 观察 | 归因 | 处置 |
|---|---|---|
| 带宽换了、档位也换了，且值对得上表 | `tier_explained` | 那是**测量变差**，另一格 `measurement_validity` 早就记着；别再报成规格漂移 |
| 带宽换了、档位**没换** | `spec_drift` | 那是**表被换过**（现场: 续跑跨过一次代码改动，前一半轮用的旧表）|
| 带宽换了、档位缺失或对不上表 | `unexplained` | 两边都不许说（缺证据既不当通过也不当反证）|
| 用过带宽的轮不足 2 | `unmeasurable` | **不是**「各轮同一条带宽」 |

把它们折成一句「尺变了」就是第一行的现场每次都被误报成第二行；反过来只认「档位一致才算同尺」
则永远看不见表被换过。这两种误读都会**指向不同的修法**（前者该去看测量质量，后者该去看代码），
所以归因必须分开 —— 这也是这片不直接复用 `cross_round_ruler` 的原因: 那条门逐键比的是「值有没有
变」，而这里的「变」有两个来源。

判决一格未动: 四态判决照旧用**当轮**的档位查表，`verdict()` 多返回一个「这次用过的那条带宽」
(第三元素)，判决结果与文案一字未改；这一片只把那个数落盘并算给两个读者看。

⚠️ 为什么这个数必须**从判决里出来**而不是调用方再查一次表: 分两次查同一张表，中间表被换过
就会记下一个「判决没用过的数」—— 用例 `test_a_table_changed_between_two_verdicts_shows_up_as_spec_drift`
第一版就是这个红法（记成 `[0.1234, 0.1234]` 而不是 `[0.001, 0.1234]`）。「有没有用过带宽」还是
判决自己的分支结构（缺值与 uninformative 两支都没用过），调用方复刻它等于第二套判据 (I22 那一族)。

⚠️ 一处刻意的「不填」: `noise_band_for(UNINFORMATIVE)` 返回 `None` 而**不是**表里的 0.02 ——
`verdict()` 在查表之前就返回 AMBIGUOUS，那一轮没有任何判决用过这条带宽。把没用过的值记进血缘
是给产物造证据（#81 那一族，方向反过来）。

变异标定（2026-09-30 实跑，基线 **23 passed**，表在 `benchmarks/mut_specs/improvement_band_drift.py`，
**行为分母 12/12 拦住、结构分母 0/0**，每条的红数与具体输入写在表头的 witness 里）:

    B1 归因不查表（档位不同就一律 tier_explained）        ← 把「表被换过」报成「测量变差」  2 红
    B2 档位不同也算规格漂移（两种归因折成一种）                                              6 红
    B3 不足 2 轮时兜成 `same`（空输入读成一致）                                              3 红
    B4 缺席的轮并进 `rounds_with_band`（「没走到判决」当「用过」）                            1 红
    B5 `noise_band_for(UNINFORMATIVE)` 回表里的 0.02（记一个没用过的数）                      1 红
    B6 conductor 忘了在判决现场写这一格（产它的与读它的断链）—— 只有真跑那组杀得掉           3 红
    B7 报表把 tier_explained 与 spec_drift 合成一个数                                         3 红
    B8 报表「测不到」那一支反过来（没原料时印一个像结论的 0）                                  4 红
    B9 报表自己拼句子，不走 `band_note`（同一件事两处两种说法）                                1 红
    B10 conductor 不用判决返回的那条带宽、自己再查一次表（记下的不是用过的）                  1 红
    B11 判决没用过带宽的两支也返回表里的值（与 B5 同族, 从产它那一侧改）                      1 红
    B12 带宽那一格与性能尺共用 `"l3_ruler"` —— **证人不在本文件里**, 是邻居那张真跑那组       6 红

B10 那一行不是假想的坏写法, 是**本片第一版的真红法**（当时落盘 `[0.1234, 0.1234]` 而判决用过
`[0.001, 0.1234]`）; 它把「查表两次之间表可以被换掉」这件事从推测变成了实测, 登记 CHECKLOG I41。

运行: PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest tests/test_improvement_band_drift.py -q
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.harness_pass_rate import (aggregate, format_report,  # noqa: E402
                                          record_rounds)
from operator_agent import measurement as ms  # noqa: E402
from operator_agent.conductor import Conductor  # noqa: E402
from operator_agent.kernel_db import KernelDatabase  # noqa: E402
from operator_agent.measurement import (BAND_SAME, BAND_SPEC_DRIFT, BAND_TIER_EXPLAINED,
                                        BAND_UNEXPLAINED, BAND_UNMEASURABLE,
                                        LINEAGE_BAND_KEY, NOISE_BAND, Validity, Verdict,
                                        band_note, cross_round_band, noise_band_for,
                                        verdict)
from operator_agent.mock_gpu import MockGPU  # noqa: E402
from operator_agent.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType, VerificationLevel)

VALID = "valid"
DEGRADED = "degraded"
UNINFORMATIVE = "uninformative"


def _rec(round_no, band, tier=VALID):
    """一轮的血缘记录: `band` 传字符串 "ABSENT" 表示这一项根本没写 (旧产物/没到判决)"""
    rec = {"round": round_no, "measurement_validity": tier}
    if band != "ABSENT":
        rec[LINEAGE_BAND_KEY] = band
    return rec


def _state(recs):
    return cross_round_band(recs)["band_state"]


class TestAttribution(unittest.TestCase):
    """这片的全部价值在「两种归因分开」那一句"""

    def test_two_rounds_of_the_same_band_are_the_same_ruler(self):
        agg = cross_round_band([_rec(1, 0.001), _rec(2, 0.001)])
        self.assertEqual(BAND_SAME, agg["band_state"])
        self.assertEqual(2, agg["rounds_with_band"])

    def test_a_wider_band_from_a_downgraded_tier_is_not_spec_drift(self):
        """存在理由那条的第一半: 档位降级本来就按表把带宽换宽 —— 报成漂移就是误归因"""
        agg = cross_round_band([_rec(1, NOISE_BAND[Validity.VALID], VALID),
                                _rec(2, NOISE_BAND[Validity.DEGRADED], DEGRADED)])
        self.assertEqual(BAND_TIER_EXPLAINED, agg["band_state"])
        self.assertIn("测量档位降级", band_note(agg))
        self.assertNotIn("⇒ 那是规格漂移", band_note(agg),
                         "「不是规格漂移」这半句本来就该出现 —— 针要落在只有漂移那一支才有的措辞上")

    def test_a_different_band_under_the_same_tier_is_spec_drift(self):
        """存在理由那条的第二半: 档位在、值对不上表 ⇒ 是表被换过, 不是测量变差"""
        agg = cross_round_band([_rec(1, 0.001, VALID), _rec(2, 0.004, VALID)])
        self.assertEqual(BAND_SPEC_DRIFT, agg["band_state"])
        self.assertEqual([(1, 2)], agg["drift_rounds"])
        self.assertIn("同一档位", band_note(agg))

    def test_missing_tier_is_attributed_to_neither(self):
        """第三态: 档位那一格缺失时, 「只是降级」与「表被换过」两句都不许说"""
        self.assertEqual(BAND_UNEXPLAINED, _state([_rec(1, 0.001, VALID),
                                                   _rec(2, 0.004, None)]))

    def test_bands_that_do_not_match_the_table_are_unexplained(self):
        """档位变了、值却对不上表 ⇒ 不是「按表换宽」那件事, 别替它归因"""
        self.assertEqual(BAND_UNEXPLAINED, _state([_rec(1, 0.001, VALID),
                                                   _rec(2, 0.009, DEGRADED)]))

    def test_an_unknown_tier_string_does_not_crash_the_review(self):
        """离线复盘会读到旧产物/别的写法; `Validity("…")` 直接抛会让一次统计变成崩溃"""
        self.assertEqual(BAND_UNEXPLAINED, _state([_rec(1, 0.001, "valid-ish"),
                                                   _rec(2, 0.004, "degraded")]))

    def test_fewer_than_two_band_rounds_is_not_agreement(self):
        """B3 的证人: 不足 2 轮是「没有输入」, 不是「各轮同一条带宽」"""
        for recs in ([], None, [_rec(1, 0.001)], [_rec(1, "ABSENT"), _rec(2, "ABSENT")]):
            agg = cross_round_band(recs)
            self.assertEqual(BAND_UNMEASURABLE, agg["band_state"], recs)
            self.assertIn("不是", agg["note"])

    def test_an_unknown_state_string_says_so(self):
        """判据的枚举与读者的分支必须同源: 加一态忘了配文案时, 这句要能指出来 (I22 那一族)"""
        text = band_note({"band_state": "brand_new_state"})
        self.assertIn("brand_new_state", text)
        self.assertIn("判据与读者不同源", text)


class TestNoBandRoundsAreNotMerged(unittest.TestCase):
    """「用过带宽」「走到判决但没用」「没到判决」是三格, 折起来就又少一个分母"""

    def test_ambiguous_rounds_and_absent_rounds_are_counted_apart(self):
        agg = cross_round_band([_rec(1, 0.001), _rec(2, None), _rec(3, "ABSENT"),
                                _rec(4, 0.001)])
        self.assertEqual(BAND_SAME, agg["band_state"], "比较只用那两轮用过带宽的")
        self.assertEqual(2, agg["rounds_with_band"])
        self.assertEqual(1, agg["rounds_no_band"])
        self.assertEqual(1, agg["rounds_no_key"])

    def test_uninformative_tier_never_used_a_band(self):
        """`noise_band_for(UNINFORMATIVE)` 必须是 `None` 而不是表里的 0.02

        `verdict()` 在查表**之前**就返回 AMBIGUOUS, 那一轮没有任何判决用过这条带宽 ——
        记下一个没用过的值就是给产物造证据 (#81 那一族的反面)。
        """
        self.assertIsNone(noise_band_for(Validity.UNINFORMATIVE))
        self.assertEqual(NOISE_BAND[Validity.UNINFORMATIVE], 0.02, "表里那个值仍然在")

    def test_the_other_tiers_return_the_table_value(self):
        """正向对照 (不发放): 守卫不许把三档都读成 None, 那样整栏永远「测不到」"""
        for v in (Validity.VALID, Validity.DEGRADED):
            self.assertEqual(NOISE_BAND[v], noise_band_for(v))

    def test_verdict_behavior_is_untouched_by_this_slice(self):
        """**这片不改判决**: 四态的边界与文案形状逐一核一遍 (B5/B1 也要让它红)

        绝对锚点是表里的数 (0.001/0.004), 不是从实现里读的 —— 从代码读回来的断言只证明
        代码等于它自己。
        """
        self.assertEqual((Verdict.FAST, "带宽"), (
            verdict(1.002, 1.0, Validity.VALID)[0], "带宽"))
        self.assertEqual(Verdict.SAME, verdict(1.0005, 1.0, Validity.VALID)[0])
        self.assertEqual(Verdict.SLOW, verdict(0.99, 1.0, Validity.DEGRADED)[0])
        self.assertEqual(Verdict.AMBIGUOUS, verdict(1.2, 1.0, Validity.UNINFORMATIVE)[0])
        self.assertEqual(Verdict.AMBIGUOUS, verdict(None, 1.0, Validity.VALID)[0])
        self.assertIn("带宽", verdict(1.002, 1.0, Validity.VALID)[1])

    def test_verdict_returns_the_band_it_actually_used(self):
        """血缘里那一格的**出处**: 由判决自己给出, 不由调用方再查一次表 (B10 的证人)

        两条 `None` 不是缺值: `verdict` 在 consult 表之前就返回 AMBIGUOUS, 那一支没有任何
        带宽参与过决定。把它们与 `0.001/0.004` 一起钉住, 才使「再查一次表」这种写法可杀 ——
        查表两次之间表被换过时, 记下的数与用过的数就不是同一个 (见文件头那条红法)。
        """
        self.assertEqual(0.001, verdict(1.002, 1.0, Validity.VALID)[2])
        self.assertEqual(0.001, verdict(1.0005, 1.0, Validity.VALID)[2])
        self.assertEqual(0.004, verdict(0.99, 1.0, Validity.DEGRADED)[2])
        self.assertIsNone(verdict(1.2, 1.0, Validity.UNINFORMATIVE)[2])
        self.assertIsNone(verdict(None, 1.0, Validity.VALID)[2])


class TestThroughARealRun(unittest.TestCase):
    """真跑 Conductor + MockGPU: 这一格必须由**判决现场**写下, 而不是由读它的一侧推导

    P7 那一课 (上一片写下又用过): 「用例都直接调 helper, 没有一条从真实循环里查『血缘有没有
    这一格』」⇒ 纯规则测试全绿而 conductor 忘了写, 报表只会退成「测不到」这个听起来尽职的词。
    """

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
        # `NOISE_BAND` 在判决现场才被查 (这正是这片要量的事), 所以测试中途改这张表
        # 就是「换过一次代码」那个现场的**同一条机制**。逐键复原。
        self._table = dict(NOISE_BAND)
        self.addCleanup(lambda: (NOISE_BAND.clear(), NOISE_BAND.update(self._table)))

    def _run(self, widen_after_verdict_no=None):
        """跑一场真任务; `widen_after_verdict_no` = 第 N 次四态判决**之后**换表 (0 = 开场就换)。

        钩子挂在 `conductor.verdict` 这个名字上而不是 verify 的调用次数上: 一轮里 verify 可能
        按输入组跑好几次, 而四态判决每轮最多一次 —— 「两次判决之间换过一次表」要量的正是后者,
        挂在前者上会让用例的构造前提依赖一个与本题无关的计数。
        """
        import operator_agent.conductor as cond
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                      llm=TestThroughARealRun._Llm(), max_optimize_rounds=3)
        c.store.output_dir = Path(self.tmp.name)
        seen = []
        orig_verdict = cond.verdict

        def spy(*a, **kw):
            if widen_after_verdict_no == 0 and not seen:
                NOISE_BAND[Validity.VALID] = 0.1234     # 第一次判决之前就已经换过
            out = orig_verdict(*a, **kw)
            seen.append(1)
            if widen_after_verdict_no == len(seen):
                NOISE_BAND[Validity.VALID] = 0.1234     # 两次判决之间换过
            return out

        cond.verdict = spy
        self.addCleanup(setattr, cond, "verdict", orig_verdict)
        task = TaskSchema(
            task_id="band-case", task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
            target=TargetSpec(hardware="mock_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=5.0,
                                       max_error=1e-3))
        inputs = {"A": np.ones((24, 16), np.float32), "B": np.ones((16, 24), np.float32)}
        with contextlib.redirect_stdout(io.StringIO()):
            art = c.execute(task, inputs, None, target_gflops=1.0)
        self.assertTrue(seen, "构造前提: 这场运行至少调用过一次四态判决")
        saved = Path(art.saved_dir)
        return (json.loads((saved / "result.json").read_text(encoding="utf-8"))
                ["iteration_history"])

    def test_the_baseline_round_has_no_key_while_later_verdicts_do(self):
        """三格在**一场真运行里**就齐了: 第 1 轮立基线 (没调用判决 ⇒ 键缺席),
        第 2、3 轮调用了判决 ⇒ 记下当轮真正用过的那条带宽。

        值断言取的是表里的数 (`NOISE_BAND[Validity.VALID]`) 而不是从实现里读回的字符串 ——
        这条同时也是「conductor 有没有在判决现场写这一格」的证人 (B6)。
        """
        history = self._run()
        self.assertGreaterEqual(len(history), 3, "构造前提: 这确实是三轮")
        self.assertNotIn(LINEAGE_BAND_KEY, history[0], history[0])
        bands = [r[LINEAGE_BAND_KEY] for r in history[1:]]
        self.assertEqual([NOISE_BAND[Validity.VALID]] * len(bands), bands,
                         "mock 这台机的档位是 valid ⇒ 每轮判决用过的带宽都是表里那条 0.10%")
        self.assertEqual(BAND_SAME, cross_round_band(history)["band_state"])

    def test_a_table_changed_between_two_verdicts_shows_up_as_spec_drift(self):
        """最强的一条证人: 第 1 次判决之后换过一次表 ⇒ 落盘血缘里是「同档位、换了值」

        这一条从**产它的那一侧**出发。`noise_band_for` 若改成在 append 时预先算好 (而不是
        在判决现场记), 或者 conductor 忘了写, 这里都会红 —— 而纯规则用例对两者全都无感。
        """
        history = self._run(widen_after_verdict_no=1)
        agg = cross_round_band(history)
        self.assertEqual([0.001, 0.1234], [b["band"] for b in agg["bands"]], agg["bands"])
        self.assertEqual(BAND_SPEC_DRIFT, agg["band_state"], agg)
        self.assertEqual([(2, 3)], [tuple(x) for x in agg["drift_rounds"]],
                         "变的那一对是轮 2 与轮 3 (轮 1 没调用判决, 没有带宽这一项)")

    def test_a_wider_band_before_the_first_verdict_is_not_drift(self):
        """**反向对照**: 第 0 次判决之前就换表 ⇒ 两轮用的是同一条新带宽, 没有漂移

        少了这条, 「只要表被改过就报漂移」这种写法也能绿 —— 而它报的是一件从没发生的事。
        """
        history = self._run(widen_after_verdict_no=0)
        agg = cross_round_band(history)
        self.assertEqual([0.1234, 0.1234], [b["band"] for b in agg["bands"]], agg["bands"])
        self.assertEqual(BAND_SAME, agg["band_state"], agg)

    def test_the_run_still_completes_with_the_observation_added(self):
        """**这片不改判决**: 跑同一个任务, 每轮的 `all_passed` 与分级结果逐格不变"""
        plain = self._run()
        hooked = self._run(widen_after_verdict_no=99)     # 永远不触发换表
        self.assertEqual([[r.get("all_passed"), sorted((r.get("levels_passed") or {}).items())]
                          for r in plain],
                         [[r.get("all_passed"), sorted((r.get("levels_passed") or {}).items())]
                          for r in hooked])


class TestTheReportHasAReader(unittest.TestCase):
    """第二个读者: 端到端报表那一栏（第 1 个是 conductor 的 `[Band]`，它只在有话说时印）"""

    @staticmethod
    def _run(case, cells, tiers=None):
        rounds = []
        tiers = tiers or [VALID] * len(cells)
        for i, (cell, tier) in enumerate(zip(cells, tiers), 1):
            rec = {"round": i, "all_passed": False,
                   "levels_passed": {VerificationLevel.L1_COMPILE.value: True,
                                     VerificationLevel.L2_FUNCTION.value: True,
                                     VerificationLevel.L3_PERFORMANCE.value: True},
                   "measurement_validity": tier}
            if cell != "ABSENT":
                rec[LINEAGE_BAND_KEY] = cell
            rounds.append(rec)
        return {"case": case, "ok": True, "rounds": rounds, "final_all_passed": False,
                "rounds_to_pass": None, "passed_via": None, "admitted": False,
                "admit_reason": ""}

    def test_the_two_attributions_are_counted_separately(self):
        agg = aggregate([
            self._run("a-same", [0.001, 0.001]),
            self._run("b-downgrade", [0.001, 0.004], [VALID, DEGRADED]),
            self._run("c-drift", [0.001, 0.004]),
            self._run("d-unexplained", [0.001, 0.004], [VALID, None]),
            self._run("e-one-round", [0.001]),
        ])
        self.assertEqual(4, agg["band_comparable_runs"], "e 凑不出比较, 不进分母")
        self.assertEqual(1, agg["band_unmeasurable_runs"])
        self.assertEqual(1, agg["band_same_runs"])
        self.assertEqual(1, len(agg["band_tier_explained_cases"]))
        self.assertEqual(1, len(agg["band_spec_drift_cases"]))
        self.assertEqual(1, len(agg["band_unexplained_cases"]))

    def test_the_column_names_each_attribution_on_its_own_line(self):
        # ⚠️ `format_report` 有自己的分母门 (运行数或轮数 < 4 ⇒ 印「分母不足」并 return),
        # 所以这里给 5 次运行 —— 见 I39: 那种红是「门没放行」而不是「这栏算错了」。
        text = format_report(aggregate([
            self._run("a-same", [0.001, 0.001]),
            self._run("b-downgrade", [0.001, 0.004], [VALID, DEGRADED]),
            self._run("c-drift", [0.001, 0.004]),
            self._run("d-unexplained", [0.001, 0.004], [VALID, None]),
            self._run("e-one-round", [0.001]),
        ]))
        self.assertIn("【改善判据那条带宽】", text)
        self.assertIn("可比的运行 4 个", text)
        self.assertIn("同一条带宽 1", text)
        self.assertIn("规格漂移) 1", text)
        self.assertIn("c-drift", text)
        self.assertIn("降级换宽: b-downgrade", text)
        self.assertIn("只观测不改判决", text)

    def test_a_downgrade_is_never_listed_under_spec_drift(self):
        """**反向对照**: 两种归因印在两条不同的行上, 折起来就指错修法"""
        text = format_report(aggregate([
            self._run("a", [0.001, 0.001]),
            self._run("b", [0.001, 0.004], [VALID, DEGRADED]),
            self._run("c", [0.001, 0.004]),
            self._run("d", [0.001, 0.004], [VALID, None]),
            self._run("e", [0.001]),
        ]))
        drift_lines = [ln for ln in text.splitlines() if ln.strip().startswith("规格漂移:")]
        self.assertEqual(1, len(drift_lines), drift_lines)
        self.assertIn("c", drift_lines[0])
        self.assertNotIn("b-downgrade", drift_lines[0])

    def test_old_lineage_is_reported_as_unmeasurable_not_as_zero_drift(self):
        """B8 的证人: 没有原料时那一栏要说「测不到」并念出键名, 而不是印一个像结论的 0"""
        agg = aggregate([self._run("old-1", ["ABSENT", "ABSENT"]),
                         self._run("old-2", ["ABSENT"]),
                         self._run("old-3", ["ABSENT"]),
                         self._run("old-4", ["ABSENT"])])
        self.assertEqual(0, agg["band_runs_with_band"])
        self.assertEqual(0, agg["band_comparable_runs"])
        text = format_report(agg)
        self.assertIn("【改善判据那条带宽】测不到", text)
        self.assertIn(LINEAGE_BAND_KEY, text)
        self.assertNotIn("同一条带宽 0", text)

    def test_record_rounds_does_not_drop_the_cell(self):
        out = record_rounds([_rec(1, 0.001)])
        self.assertEqual(0.001, out[0][LINEAGE_BAND_KEY])

    def test_the_column_is_not_built_from_a_second_implementation(self):
        """报表里的句子必须出自 `band_note` 那一个实现 (与 conductor 现场同一措辞)

        这条不是测文本, 是测**出处**: 把 `_band_stats` 里的 `band_note(a)` 换成自己拼句子
        (变异 B9) 时, 同一件事在两个地方会长成两种说法 —— 本仓那一族的第 N 次。
        """
        agg = cross_round_band([_rec(1, 0.001, VALID), _rec(2, 0.004, VALID)])
        one_line = band_note(agg)
        text = format_report(aggregate([self._run("c-drift", [0.001, 0.004]),
                                        self._run("a", [0.001, 0.001]),
                                        self._run("b", [0.001, 0.001]),
                                        self._run("e", [0.001])]))
        self.assertIn(one_line, text, "报表里那句归因应当与现场打印那句逐字相同")


if __name__ == "__main__":
    unittest.main()
