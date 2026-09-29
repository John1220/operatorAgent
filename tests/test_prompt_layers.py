"""C2 热/冷两层上下文的守卫 (决策与实测见 docs/research/20 第三节、37)

要守住的性质只有一条大原则: **prompt 里只许有热层, 而且只许出现一次**。
冷层 (逐轮累计的 hint 日志) 留在 sketch 与快照里, 不进 Coder prompt。
"""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.measure_prompt_layers import (  # noqa: E402
    CapturingLLM, EmptyDB, initial_sketch, perf_data, task)
from operator_agent.agents import CoderAgent, SearchAgent, _clean, _hint_log  # noqa: E402
from operator_agent.interface_contract import broadcast_calling_convention  # noqa: E402
from operator_agent.schemas import UnifiedSketch  # noqa: E402


def _coder_prompt(sketch):
    llm = CapturingLLM()
    CoderAgent(kernel_db=EmptyDB(), llm=llm)._llm_generate_code(sketch, task())
    return llm.prompts[0]


def _advanced(n_rounds, llm_path=True):
    """走真实的 SearchAgent 路径推进 n 轮, 返回最后一轮的 sketch"""
    llm = CapturingLLM()
    searcher = SearchAgent(llm=llm)
    sketch = initial_sketch()
    for _ in range(n_rounds):
        sketch, _ = (searcher._llm_optimize(sketch, perf_data(30.0)) if llm_path
                     else searcher._rule_optimize(sketch, perf_data(30.0)))
    return sketch


class TestLayerSemantics(unittest.TestCase):

    def test_hot_layer_falls_back_to_log_on_version_one(self):
        """Designer 出来的 v1 还没有「本轮」概念, 此时热层就是那份日志

        回退写反了 (返回 []) 的话, 第一轮生成代码时 Coder 会完全看不到设计建议 ——
        这才是这层设计真正的风险所在。
        """
        s = UnifiedSketch(optimization_hints=["a", "b"])
        self.assertEqual(["a", "b"], s.hot_hints())
        s2 = UnifiedSketch(optimization_hints=["a", "b"], hints_this_version=["c"])
        self.assertEqual(["c"], s2.hot_hints())

    def test_cold_log_dedupes_but_hot_layer_keeps_this_round_verbatim(self):
        s = initial_sketch()
        for _ in range(4):
            s, _ = SearchAgent(llm=CapturingLLM())._rule_optimize(s, perf_data(30.0))
        self.assertEqual(len(set(s.optimization_hints)), len(s.optimization_hints),
                         "冷层日志里还有重复条目 —— 去重没生效")
        self.assertEqual(["x", "y"], _hint_log(UnifiedSketch(optimization_hints=["x", "y"]),
                                               ["x", "y"]))
        self.assertTrue(s.hints_this_version)
        self.assertLessEqual(len(s.hints_this_version), len(s.optimization_hints))

    def test_non_string_hints_never_reach_the_prompt(self):
        """new_hints 来自 LLM 的 JSON: 一个 dict 混进来就会以 Python repr 印进 prompt"""
        self.assertEqual(["ok"], _clean([{"a": 1}, 7, None, "", "  ", "ok"]))
        s, _ = SearchAgent(llm=CapturingLLM())._rule_optimize(initial_sketch(),
                                                              perf_data(30.0))
        self.assertTrue(all(isinstance(h, str) for h in s.hints_this_version))

    def test_round_trip_keeps_both_layers_and_prompt_dict_keeps_neither(self):
        s = _advanced(2)
        back = UnifiedSketch.from_dict(s.to_dict())
        self.assertEqual(s.optimization_hints, back.optimization_hints)
        self.assertEqual(s.hints_this_version, back.hints_this_version)
        d = s.to_prompt_dict()
        self.assertNotIn("optimization_hints", d)
        self.assertNotIn("hints_this_version", d)
        # 结构信息不许跟着一起丢
        self.assertIn("tiling", d)
        self.assertEqual(s.tiling.block_m, d["tiling"]["block_m"])


class TestPromptIsHotOnly(unittest.TestCase):

    def test_each_hot_hint_appears_exactly_once(self):
        """两个注入点 (sketch JSON 与 {{optimization_hints}}) 飘成一个的老毛病

        改之前实测每条 hint 出现 2 次, 规则回退路径上因为日志不去重最多到 20 次。
        """
        for rounds in (0, 3, 6):
            with self.subTest(rounds=rounds):
                sketch = initial_sketch() if rounds == 0 else _advanced(rounds)
                prompt = _coder_prompt(sketch)
                for h in sketch.hot_hints():
                    self.assertEqual(1, prompt.count(h),
                                     f"热 hint 在 prompt 里出现 {prompt.count(h)} 次: {h[:40]}")

    def test_cold_only_hints_are_absent_from_the_prompt(self):
        sketch = _advanced(6)
        prompt = _coder_prompt(sketch)
        cold_only = [h for h in sketch.optimization_hints
                     if h not in sketch.hot_hints()]
        self.assertTrue(cold_only, "6 轮之后冷层应当已经攒出热层之外的内容, 否则测不出什么")
        for h in cold_only:
            self.assertEqual(0, prompt.count(h), f"冷层 hint 漏进了 prompt: {h[:40]}")

    def test_prompt_size_stops_growing_after_the_first_optimization(self):
        """规模曲线必须压平: 后半段每轮增幅 < 30 字符

        额度留给 tiling 数字本身的变化 (block_n 随轮次变), 那不是历史堆积。
        改之前: LLM 路径 6 轮 +6.4%, 规则路径 +17.2%。
        """
        for llm_path in (True, False):
            with self.subTest(llm_path=llm_path):
                llm = CapturingLLM()
                searcher = SearchAgent(llm=llm)
                coder = CoderAgent(kernel_db=EmptyDB(), llm=llm)
                t = task()
                sketch = initial_sketch()
                sizes = []
                for _ in range(6):
                    llm.prompts.clear()
                    coder._llm_generate_code(sketch, t)
                    sizes.append(len(llm.prompts[0]))
                    sketch, _ = (searcher._llm_optimize(sketch, perf_data(30.0))
                                 if llm_path else
                                 searcher._rule_optimize(sketch, perf_data(30.0)))
                per_round = (sizes[-1] - sizes[2]) / (len(sizes) - 3)
                self.assertLess(per_round, 30,
                                f"每轮仍涨 {per_round:.1f} 字符: {sizes}")

    def test_pipeline_coder_path_is_hot_too(self):
        """CoderPipeline 是第二条 coder 路径 —— 只测第一条会漏掉它

        行为断言而不是「源码里还有 hot_hints() 字样」: 后者在 K5 那种「把这一处改回
        冷层、另一处留着」的变异下会跟着文本走, 量不到实际注入的内容。
        """
        from operator_agent.coder_pipeline import CoderPipeline

        llm = CapturingLLM()
        pipe = CoderPipeline(llm=llm, kernel_db=EmptyDB())
        sketch = _advanced(6)
        cold_only = [h for h in sketch.optimization_hints if h not in sketch.hot_hints()]
        self.assertTrue(cold_only)
        pipe._sample_one("步骤1: 分块", sketch, task(), "", 0.1,
                         broadcast_calling_convention(None))
        prompt = llm.prompts[-1]
        for h in cold_only:
            self.assertEqual(0, prompt.count(h), f"冷层 hint 漏进 pipeline 的 prompt")
        for h in sketch.hot_hints():
            self.assertEqual(1, prompt.count(h))

    def test_guard_is_not_a_no_op(self):
        """把冷层硬塞回 prompt, 上面的断言必须会红 —— 否则这条守卫是在空转"""
        s = initial_sketch()
        s.optimization_hints = ["冷层专有句子-不可注入-XYZ"]
        s.hints_this_version = []
        prompt = _coder_prompt(s)
        self.assertIn("冷层专有句子-不可注入-XYZ", prompt,
                      "v1 时 hot_hints() 退回全量日志, 所以这里应当出现; 若不然说明回退了")
        s2 = _advanced(3)
        s2.optimization_hints = list(s2.optimization_hints) + ["冷层专有句子-ABC"]
        self.assertIn("冷层专有句子-ABC", s2.optimization_hints)
        self.assertNotIn("冷层专有句子-ABC", _coder_prompt(s2))


class TestRenderSitesUseTheHotLayer(unittest.TestCase):
    """coder 侧两条渲染路径都必须经 hot_hints(); 冷层只在 _hint_log 里被读

    这是「算了没人读」的反面版本: 新加一条 coder 路径时如果直接
    `sketch.optimization_hints` 去拼 prompt, 堆积会静默回来。
    """

    def test_cold_log_is_read_only_by_the_dedup_helper(self):
        src = (ROOT / "operator_agent" / "agents.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        readers = sorted({n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                          and any(isinstance(x, ast.Attribute)
                                  and x.attr == "optimization_hints"
                                  and isinstance(x.value, ast.Name)
                                  and x.value.id == "sketch"
                                  for x in ast.walk(n))})
        self.assertEqual(["_hint_log"], readers,
                         f"这些地方还在直接读冷层: {readers}")

    def test_both_coder_paths_render_hot_hints(self):
        for rel in ("operator_agent/agents.py", "operator_agent/coder_pipeline.py"):
            with self.subTest(path=rel):
                src = (ROOT / rel).read_text(encoding="utf-8")
                self.assertIn("hot_hints()", src, f"{rel} 不再经热层")


if __name__ == "__main__":
    unittest.main(verbosity=2)
