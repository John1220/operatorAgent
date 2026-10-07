"""报告产物 (设计 §17 第 3 条) 的守卫

设计的四件交付物里, 前两件早就有产物 (code/result/benchmark JSON), 第四件半有 (指纹),
**第三件此前完全没有**: `confirm.UnableToEstablish` 这个收集器写好了, 但全仓只有两个
benchmark 脚本调它 —— 产品链路跑完一次真实验, 没有任何文件能回答「这一轮哪些事**没能**
确立」。而设计把这一件写成「自动跑系统的信誉所在」:

    一个从不说「我测不准」的自动系统, 它的所有数字都不可信。

两组, 各挡一类失效:

A. 纯函数 (`build_report` / `collect_unable`): 逐条信号成对写 —— 坏的时候必须进
   「我无法确立的事」, 好的时候必须**不**进, 且空清单必须带依据。防的是这一节变成
   复读机 (信号一多就被跳过不读) 或者变成静默通道 (空 = 通过, 而没收集也是空)。

B. 真跑一遍 `Conductor.execute` (MockGPU + 假 LLM): 断言产物目录里**真有** `report.md`,
   且三个小节都在。防的是「模块写好了但没接上」—— 这正是本仓反复出现的那一族
   (CHECKLOG I45/I46/I48): 函数存在、单元测试绿、生产路径上没人调。变异: 把
   `conductor._execute_inner` 结尾那次 `write_report(...)` 删掉, B 组全红。

运行: python3 -m unittest tests.test_report_deliverable -v
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.confirm import UnableToEstablish
from operator_agent.report import (REPORT_NAME, build_report, collect_unable,
                                   write_report)
from operator_agent.schemas import (ConstraintSpec, KernelArtifact, Precision,
                                    SourceSpec, TargetSpec, TaskComplexity,
                                    TaskSchema, TaskType)

#: 一份**站得住**的最后一轮读数: 8 项前置全过。峰值那一位是 True —— 今天真机发布的
#: 是 False (probe_peak 未落地), 这里造的是「探测已落地之后」的形状, 用来钉「为空」的
#: 那条路径; 「没实测 ⇒ 有条目」由下面单独的用例负责。
HEALTHY_METRICS = {
    "measurement_can_gate": True,
    "measurement_validity": "valid",
    "speedup_vs_baseline": 1.25,
    "gflops": 123.0,
    "ruler_broken": False,
    "benchmark_contaminated": False,
    "roofline_analysis": {"reference_lower_bound": True, "peak_trusted": True,
                          "peak_source": "实测探测 (probe_peak)"},
    "eligibility": {"can_verdict": True, "can_tune": True, "can_persist": True,
                    "can_report": True, "can_count_progress": True},
}


def _artifact(metrics=None, **overrides):
    a = KernelArtifact(kernel_id="k1", code="def kernel(A, B):\n    return A @ B\n")
    a.benchmark_data = dict(HEALTHY_METRICS if metrics is None else metrics)
    a.environment_drift = {"comparable": True, "drifted": False, "changed": [],
                           "previous_from": "out/prev"}
    a.iteration_history = [
        {"round": 1, "action": "generate", "levels_passed": {"L1": True, "L2": True, "L3": True},
         "gflops": 123.0, "speedup": 1.25, "measurement_validity": "valid",
         "recipes_applied": []},
    ]
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _unable_text(artifact, **kw):
    return collect_unable(artifact=artifact, **kw).render()


class TestUnestablishedSection(unittest.TestCase):
    """A 组: 「我无法确立的事」这一节的形状"""

    def test_the_report_carries_the_three_deliverable_sections(self):
        txt = build_report(artifact=_artifact())
        for section in ("best-known-good", "证据链", "我无法确立的事"):
            self.assertIn(section, txt, section)

    def test_an_empty_section_prints_its_basis(self):
        """空 = 一个断言, 它必须带依据 (逐项核过的那 8 条)"""
        u = collect_unable(artifact=_artifact())
        self.assertEqual([], u.items)
        self.assertIsNotNone(u.absence_basis, "空了却没写依据 —— render 会点名")
        text = u.render()
        self.assertIn("(无)", text)
        self.assertIn("依据", text)
        self.assertIn("峰值分母有参考下界", text, "依据要逐项列出, 不是一句「一切正常」")

    def test_an_empty_section_without_a_basis_is_itself_named(self):
        """反向: 手工造一个空且无依据的收集器 ⇒ 不许印成「无」就过去"""
        text = UnableToEstablish().render()
        self.assertIn("无法确立", text)
        self.assertIn("没有任何依据", text)

    def test_each_bad_precondition_becomes_an_item(self):
        """八项前置逐条成对: 每一项坏掉都要有自己那条, 而且理由要指名道姓"""
        cases = [
            ("峰值未实测", {"roofline_analysis": {"reference_lower_bound": False,
                                                  "peak_source": "规格表"}},
             "天花板"),
            ("心跳没记录", {"ruler_broken": None}, "心跳"),
            ("心跳失败", {"ruler_broken": True, "ruler_reason": "参照偏出"},
             "标定心跳失败"),
            ("窗口被抢占", {"benchmark_contaminated": True}, "抢占"),
            ("档位降级", {"measurement_validity": "degraded"}, "valid 档"),
            ("不能判事", {"measurement_can_gate": False}, "不足以支撑任何结论"),
            ("契约被改", {"contract_hash_mismatch": True}, "契约被改写"),
            ("收口点没跑", {"eligibility": None}, "收口点"),
        ]
        for name, patch, expect in cases:
            with self.subTest(case=name):
                m = dict(HEALTHY_METRICS)
                m.update(patch)
                u = collect_unable(artifact=_artifact(metrics=m))
                self.assertTrue(u.items, f"{name}: 坏了却一条都没收")
                self.assertIn(expect, u.render(), f"{name}: 收了但没指名道姓")
        # 反向: 全好的读数上一条都不许有 (否则这一节会被读成复读机)。
        # 断言 `items` 而不是整段文字 —— 成立的依据串里也会出现「标定心跳」这类词。
        self.assertEqual([], collect_unable(artifact=_artifact()).items)

    def test_drift_that_was_never_compared_is_not_a_pass(self):
        """「无从比较」与「比较了没漂」必须分开 —— 前者是缺证据, 不是证据"""
        a = _artifact()
        a.environment_drift = None
        self.assertIn("无法比较环境漂移", _unable_text(a))
        a.environment_drift = {"comparable": False, "drifted": False, "changed": []}
        self.assertIn("不可比", _unable_text(a))
        a.environment_drift = {"comparable": True, "drifted": True,
                               "changed": ["driver"], "previous_from": "p"}
        self.assertIn("已漂移", _unable_text(a))

    def test_a_missing_speedup_key_is_not_read_as_no_speedup(self):
        """三态: 缺键 ≠ 没加速。这一条以前没有任何读者"""
        m = {k: v for k, v in HEALTHY_METRICS.items() if k != "speedup_vs_baseline"}
        self.assertIn("缺 speedup_vs_baseline", _unable_text(_artifact(metrics=m)))

    def test_the_ingest_refusal_is_carried_into_the_report(self):
        """「这条结论为什么没进语料」是报告里最要紧的一句之一 (此前只有一个 print)"""
        a = _artifact()
        a.metadata = {"corpus_admission": {"kept": False,
                                          "reason": "相对同机基线没有加速 (speedup=0.8)"}}
        text = _unable_text(a)
        self.assertIn("没有进入长期记忆", text)
        self.assertIn("speedup=0.8", text, "拒因原文要一起带出来")

    def test_the_same_judgement_is_not_said_twice_by_two_sources(self):
        """同一条判据在收集器与前置项里都发话 —— 说法必须只有一种

        这条是真跑 demo 时抓到的: 峰值未实测会在 `confirm.extend_metrics` 与
        `report._preconditions` 各产生一条, 措辞差一个词就去重不掉, 于是同一件事
        被数了两遍。这一节一旦复读就会被跳过不读。
        """
        m = dict(HEALTHY_METRICS, roofline_analysis={"reference_lower_bound": False,
                                                     "peak_source": "规格表推导"})
        items = collect_unable(artifact=_artifact(metrics=m)).items
        # 去重键是 (claim, reason) **文本本身** ⇒ 这句话一改名, 两个生产者与这里的
        # 定位必须同时改 (设计 §13.5 那次改名就是一次演练: 措辞一变, 这条用例当场红)。
        hits = [i for i in items if "天花板" in i.claim]
        self.assertEqual(1, len(hits), [i.claim for i in items])

    def test_no_duplicate_claims(self):
        """同一条不许数两遍: 这一节一旦复读就会被跳过不读"""
        m = dict(HEALTHY_METRICS, ruler_broken=True, ruler_reason="参照偏出")
        u = collect_unable(artifact=_artifact(metrics=m),
                           observations=[m, m])   # 同一份读数传两次
        keys = [(i.claim, i.reason) for i in u.items]
        self.assertEqual(len(keys), len(set(keys)), keys)


class TestBestKnownGood(unittest.TestCase):
    def test_it_says_when_the_number_may_not_be_remembered(self):
        """can_persist=False 是**后果**上的差别, 必须在正文里明写"""
        m = dict(HEALTHY_METRICS,
                 eligibility=dict(HEALTHY_METRICS["eligibility"], can_persist=False))
        txt = build_report(artifact=_artifact(metrics=m))
        self.assertIn("这条没有进长期记忆", txt)
        self.assertIn("can_persist", txt)

    def test_it_admits_when_there_is_no_reading_at_all(self):
        a = _artifact()
        a.benchmark_data = {}
        txt = build_report(artifact=a)
        self.assertIn("没有性能读数", txt)


class TestWriteReport(unittest.TestCase):
    def test_it_lands_in_the_artifact_dir(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_report(d, artifact=_artifact(),
                             task=_task())
            self.assertEqual(REPORT_NAME, p.name)
            self.assertTrue(p.exists())
            self.assertIn("best-known-good", p.read_text(encoding="utf-8"))

    def test_a_broken_artifact_does_not_raise_but_leaves_a_trace(self):
        """写报告失败不该让一次跑完的实验失败 —— 但也不许静默: 文件里留异常原文"""
        class Boom:
            @property
            def benchmark_data(self):
                raise RuntimeError("半成品 artifact")
        with tempfile.TemporaryDirectory() as d:
            p = write_report(d, artifact=Boom())
            self.assertTrue(p.exists())
            self.assertIn("报告生成失败", p.read_text(encoding="utf-8"))


# ─── B 组: 真跑一遍 ────────────────────────────────────────────────

def _task():
    return TaskSchema(
        task_id="report-e2e",
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0,
                                   max_error=1e-3))



class TestTheStaticAuditGapIsInTheDeliverable(unittest.TestCase):
    """§21.4/§22.3 的覆盖缺口只有**一个**人类可读的下游: 这一节 (CHECKLOG I90)

    血缘里的 `precompile.coverage.uncovered_rules` 在本片之前没有任何读者 ——
    报告里读起来像「源码级幻觉 API 查过了」。这一组钉住「有缺口 ⇒ 有条目」「没缺口 ⇒
    不许多一条」「没判过 ⇒ 不说话」三态。
    """

    def _with_verdict(self, uncovered, outcomes=None):
        a = _artifact()
        a.iteration_history = [
            {"round": 1, "action": "generate",
             "levels_passed": {"L1": True, "L2": True, "L3": True},
             "agent_action": {"precompile": {
                 "coverage": {"language": "python", "mode": "ast",
                              "covered_rules": ["write_scope"],
                              "uncovered_rules": list(uncovered)},
                 "outcomes": outcomes or []}}},
        ]
        return a

    def _gap_items(self, artifact):
        return [i for i in collect_unable(artifact=artifact).items if "没有量具" in i.claim]

    def test_an_uncovered_rule_becomes_an_item_with_its_detail(self):
        a = self._with_verdict(
            ["capability"],
            [{"rule": "capability", "state": "unknown",
              "detail": "源码级「幻觉 API」没有判据: 仓里没有任何语言的 API 面清单"}])
        text = _unable_text(a)
        self.assertIn("没有量具", text)
        self.assertIn("capability", text)
        self.assertIn("API 面清单", text, "缺口要带逐条详情, 不是一个光秃秃的判据名")

    def test_no_gap_means_no_item(self):
        """反向: 凹缺口为空时不许多印一条 (这一节一旦复读就会被跳过不读)"""
        self.assertEqual([], self._gap_items(self._with_verdict([])))

    def test_a_run_without_a_verdict_says_nothing(self):
        """三态: 没有血缘 / 没走 §21 的路径 = **没判过**, 不是「没有缺口」"""
        self.assertEqual([], self._gap_items(_artifact()))

    def test_a_not_applicable_rule_is_not_a_gap(self):
        """`not_applicable` 不进这一节: 「这一份候选没有对象」不是「这一层查不了」(I90)"""
        a = self._with_verdict(
            [], [{"rule": "new_dependency", "state": "not_applicable",
                  "detail": "没有父代 ⇒ 这条判据没有对象"}])
        self.assertEqual([], self._gap_items(a))

    def test_only_the_last_adjudicated_round_is_read(self):
        """更早轮次的缺口若已经补上, 不再印 —— 报告说的是「结束时还立不住的事」"""
        a = _artifact()
        a.iteration_history = [
            {"round": 1, "agent_action": {"precompile": {
                "coverage": {"uncovered_rules": ["capability"]}, "outcomes": []}}},
            {"round": 2, "agent_action": {"precompile": {
                "coverage": {"uncovered_rules": []}, "outcomes": []}}},
        ]
        self.assertEqual([], self._gap_items(a))


class TestThroughARealRun(unittest.TestCase):
    """模块写好了 ≠ 接上了。这一组断言的是**产物目录里真有那份文件**。

    变异: 删掉 `conductor._execute_inner` 结尾那次 `write_report(...)`,
    下面两条都红 (A 组仍然全绿 —— 这正是要分开两组的原因)。
    """

    CODE = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
    SKETCH = {"algorithm": "matmul",
              "tiling": {"block_m": 16, "block_n": 16, "block_k": 16},
              "memory_plan": {}, "pipeline": [], "optimization_hints": []}

    class _Llm:
        def chat(self, messages, **kw):
            return TestThroughARealRun.CODE

        def chat_json(self, messages, **kw):
            return dict(TestThroughARealRun.SKETCH)

    def _run(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                      llm=self._Llm(), max_optimize_rounds=1)
        c.store.output_dir = Path(tmp.name)
        rng = np.random.default_rng(0)
        inputs = {"A": rng.standard_normal((32, 32)).astype(np.float32),
                  "B": rng.standard_normal((32, 32)).astype(np.float32)}
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            art = c.execute(_task(), inputs, None, target_gflops=0.0)
        return art, buf.getvalue()

    def test_the_run_writes_a_report_next_to_the_json(self):
        art, out = self._run()
        self.assertIsNotNone(getattr(art, "saved_dir", None), "前提: 产物落盘了")
        path = Path(art.saved_dir) / REPORT_NAME
        self.assertTrue(path.exists(),
                        f"产物目录里没有 {REPORT_NAME} —— 报告模块没接上")
        self.assertIn("报告已写出", out)
        txt = path.read_text(encoding="utf-8")
        for section in ("best-known-good", "证据链", "我无法确立的事"):
            self.assertIn(section, txt)

    def test_the_written_report_speaks_about_this_run(self):
        """产物不能是一份通用模板: 它必须带着这次运行的读数与指纹"""
        art, _ = self._run()
        txt = (Path(art.saved_dir) / REPORT_NAME).read_text(encoding="utf-8")
        self.assertIn(str(art.kernel_id), txt)
        self.assertIn("matmul", txt.lower(), "算子名要出现")
        # 诚实性检查: 报告说的重测状态必须与 artifact 上的一致 (这一句最容易写成
        # 「永远说有」或「永远说没有」)
        if getattr(art, "remeasure", None) is None:
            self.assertIn("**没有**做重测", txt)
        else:
            self.assertIn("有: ", txt)

    def test_the_static_audit_gap_reaches_the_written_report(self):
        """§21.4 源码级「幻觉 API」判不了 ⇒ 必须出现在这份**真跑出来**的报告里

        这一条走的是真的 `Conductor.execute`（不是手搓 artifact）: 覆盖表在血缘里,
        而把血缘读成一句话的只有 `report.collect_unable`（CHECKLOG I90）。
        """
        art, _ = self._run()
        txt = (Path(art.saved_dir) / REPORT_NAME).read_text(encoding="utf-8")
        self.assertIn("没有量具", txt)
        self.assertIn("capability", txt)

    def test_the_implementation_verdict_reaches_the_written_report(self):
        """§17.1 的实现裁决是交付物第 1 件 ⇒ 真跑一遍以后必须在报告里（带理由）

        今天它只能是 `inconclusive`（两条证据没有生产者, 见 CHECKLOG I91）—— 这一条钉
        「生产路径写进去了」+「报告读了它」, 而不是裁决的取值本身。
        """
        art, _ = self._run()
        txt = (Path(art.saved_dir) / REPORT_NAME).read_text(encoding="utf-8")
        self.assertIn("实现裁决 (§17.1)", txt)
        self.assertIn("inconclusive", txt)
        self.assertIn("门 3", txt, "理由要一起印出来, 否则读不出「为什么判不了」")


if __name__ == "__main__":
    unittest.main(verbosity=2)