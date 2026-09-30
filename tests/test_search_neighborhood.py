"""P1-4 首片 (docs/design/generation-side-review.md + doc 14): LLM 建议降为先验

旧实现: SearchAgent 的一次 LLM 跳点被直接采纳, 失败了分不清「方向不行」
还是「没搜到」。首片改动:
  - tiling 的格点邻域 (每维 ±1 步, 一次只动一维, 可归因) 交给确定性排序;
  - roofline 方向判据: memory-bound 优先增大 BLOCK_K, compute-bound 优先
    减小 BLOCK_M/N, unknown 不猜方向 (离先验近者优先);
  - 先验参与排序而非直接采纳 —— 启发式给出明确方向时可以位移先验;
  - Conductor 跨轮轮换排队的邻域候选 (真机 L3 当裁判), 不再每轮重问 LLM。
BO/bandit 的 surrogate 需要实测分布, 是后续片。

运行: python -m pytest tests/test_search_neighborhood.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.agents import SearchAgent
from operator_agent.conductor import Conductor
from operator_agent.mock_gpu import MockGPU
from operator_agent.kernel_db import KernelDatabase
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec, UnifiedSketch,
    TilingStrategy, TaskType, TaskComplexity, Precision,
)

GOOD_KERNEL = """import numpy as np

def kernel(A, B):
    return A @ B
"""


class MockSearchLLM:
    """只应答 SearchAgent 的 chat_json (优化提案)"""

    def __init__(self, proposal=None):
        self.calls = 0
        self._proposal = proposal or {}

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        self.calls += 1
        return dict(self._proposal)


def _sketch(tiling=None, dsl="triton", version=1) -> UnifiedSketch:
    return UnifiedSketch(algorithm="matmul", version=version, target_dsl=dsl,
                         tiling=tiling or TilingStrategy(64, 64, 32))


def _perf(bound=None, **kw):
    d = {"gflops": 10.0, "speedup": 0.6, "target_speedup": 1.0}
    if bound:
        d["roofline_bound"] = bound
    d.update(kw)
    return d


# ─── 1. 格点邻域 ─────────────────────────────────────────────

class TestLatticeNeighborhood(unittest.TestCase):

    def test_single_step_neighbors_dedup_and_validate(self):
        """64x64x32 的邻域 = 每维 ±1 格共 6 个, 无中心, 全部过 schema 校验"""
        neighbors = SearchAgent._lattice_neighbors(TilingStrategy(64, 64, 32))
        keys = {(t.block_m, t.block_n, t.block_k) for t in neighbors}
        self.assertEqual(keys, {(32, 64, 32), (128, 64, 32),
                                (64, 32, 32), (64, 128, 32),
                                (64, 64, 16), (64, 64, 64)})
        for t in neighbors:
            self.assertEqual(t.validate(), [], f"非法邻域点: {t}")

    def test_boundary_clamped(self):
        """格点边界不越界: block_k=16 只能向上; block_m=512 只能向下"""
        neighbors = SearchAgent._lattice_neighbors(TilingStrategy(512, 16, 16))
        keys = {(t.block_m, t.block_n, t.block_k) for t in neighbors}
        self.assertIn((256, 16, 16), keys)
        self.assertNotIn((1024, 16, 16), keys)
        self.assertIn((512, 16, 32), keys)
        self.assertNotIn((512, 16, 8), keys)

    def test_off_lattice_center_not_guessed(self):
        """中心不在合法格点: 该维不动 (交给 schema 校验报错), 不猜"""
        neighbors = SearchAgent._lattice_neighbors(TilingStrategy(48, 64, 32))
        keys = {(t.block_m, t.block_n, t.block_k) for t in neighbors}
        self.assertTrue(keys)
        for m, n, k in keys:
            self.assertEqual(m, 48, "非法中心的 block_m 不该被改动")


# ─── 2. 先验降级与方向排序 ──────────────────────────────────

class TestPriorDemotion(unittest.TestCase):

    def test_memory_bound_displaces_prior_toward_larger_k(self):
        """memory-bound: 先验 128x128x32 被邻域 k+1 步 (block_k=64) 位移

        这条就是「建议降为先验」的可观察形态: LLM 给的 tiling 不再自动获胜。
        """
        llm = MockSearchLLM({"suggestion": "增大 tile",
                             "new_tiling": {"block_m": 128, "block_n": 128,
                                            "block_k": 32}})
        search = SearchAgent(llm=llm)
        chosen, _ = search.run(_sketch(), _perf(bound="memory"))
        self.assertEqual((chosen.tiling.block_m, chosen.tiling.block_n,
                          chosen.tiling.block_k), (128, 128, 64))
        self.assertTrue(search.last_alternatives, "邻域候选没有排队")
        # 队列里不应再出现与选中者相同的 tiling
        for alt in search.last_alternatives:
            self.assertNotEqual((alt.tiling.block_m, alt.tiling.block_n,
                                 alt.tiling.block_k), (128, 128, 64))

    def test_compute_bound_prefers_smaller_mn(self):
        """compute-bound: 先验 128x128x64 让位给 m-1 步 (occupancy 方向)"""
        llm = MockSearchLLM({"suggestion": "调整",
                             "new_tiling": {"block_m": 128, "block_n": 128,
                                            "block_k": 64}})
        search = SearchAgent(llm=llm)
        chosen, _ = search.run(_sketch(), _perf(bound="compute"))
        self.assertEqual(chosen.tiling.block_m * chosen.tiling.block_n, 64 * 128)

    def test_unknown_bound_keeps_prior_first(self):
        """unknown: 不猜方向 —— 先验 (距离 0) 仍是第一, 邻域按距离排队"""
        llm = MockSearchLLM({"suggestion": "换思路",
                             "new_tiling": {"block_m": 128, "block_n": 128,
                                            "block_k": 32}})
        search = SearchAgent(llm=llm)
        chosen, _ = search.run(_sketch(), _perf())
        self.assertEqual((chosen.tiling.block_m, chosen.tiling.block_n,
                          chosen.tiling.block_k), (128, 128, 32),
                         "unknown 方向下先验不该被位移")

    def test_noop_prior_yields_to_neighborhood(self):
        """LLM 没动 tiling (先验==origin) 且有邻域可试: 不再原样重跑"""
        llm = MockSearchLLM({"suggestion": "保持", "new_hints": ["vectorize"]})
        search = SearchAgent(llm=llm)
        origin = _sketch(TilingStrategy(64, 64, 32))
        chosen, _ = search.run(origin, _perf(bound="memory"))
        self.assertNotEqual((chosen.tiling.block_m, chosen.tiling.block_n,
                             chosen.tiling.block_k), (64, 64, 32),
                             "原样重跑等于空烧一轮")
        # 结构性判断保留: hints 来自 LLM 提案
        self.assertIn("vectorize", chosen.hints_this_version)

    def test_non_triton_dsl_adopts_directly(self):
        """python/cuda_c: block 参数不是真实旋钮, 维持直接采纳, 队列为空"""
        llm = MockSearchLLM({"suggestion": "python 优化",
                             "new_tiling": {"block_m": 128, "block_n": 128,
                                            "block_k": 32}})
        search = SearchAgent(llm=llm)
        chosen, _ = search.run(_sketch(dsl="python"), _perf(bound="memory"))
        self.assertEqual((chosen.tiling.block_m, chosen.tiling.block_n,
                          chosen.tiling.block_k), (128, 128, 32))
        self.assertEqual(search.last_alternatives, [])


# ─── 3. Conductor 跨轮轮换 (集成) ───────────────────────────

class _FakeSearch:
    """替身 SearchAgent: 只被调用一次, 之后 Conductor 应改用轮换队列"""

    def __init__(self, alternatives):
        self.calls = 0
        self._alternatives = alternatives

    def run(self, sketch, perf_data):
        self.calls += 1
        new_sketch = UnifiedSketch(algorithm=sketch.algorithm,
                                   version=sketch.version + 1,
                                   target_dsl=sketch.target_dsl,
                                   tiling=TilingStrategy(128, 64, 32))
        self.last_alternatives = list(self._alternatives)
        return new_sketch, "先验"


class _SteadyLLM:
    """Designer/Planner/Coder 共用: chat_json 回 sketch, chat 回好代码 (L1/L2 恒过)"""

    def __init__(self):
        self.chat_calls = []

    def chat(self, messages, temperature=None, max_tokens=None, raw=False, caller_tag="Unknown"):
        self.chat_calls.append(messages)
        return GOOD_KERNEL

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        return {"algorithm": "matmul",
                "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                "memory_plan": {}, "pipeline": [], "optimization_hints": []}


class TestConductorRotation(unittest.TestCase):

    def test_rotation_replaces_repeated_llm_asks(self):
        """L3 连败 → 第 2 次优化起轮换队列, Search (LLM) 不再被调用"""
        llm = _SteadyLLM()
        alts = [_sketch(TilingStrategy(32, 64, 32), dsl="python", version=3),
                _sketch(TilingStrategy(256, 64, 32), dsl="python", version=4)]
        conductor = Conductor(backend=MockGPU(),
                              kernel_db=KernelDatabase(jsonl_path=None),
                              llm=llm, max_optimize_rounds=3)
        fake = _FakeSearch(alts)
        conductor.search = fake

        seen_sketches = []
        real_run = conductor.coder.run

        def spy_run(sketch, task, **kw):
            seen_sketches.append(sketch)
            return real_run(sketch, task, **kw)

        conductor.coder.run = spy_run

        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
            target=TargetSpec(hardware="mock_gpu", dsl="python",
                              compute_units=64, warp_size=32),
            # min_speedup 高到 L3 恒败, 制造连续的优化轮
            constraints=ConstraintSpec(precision=Precision.FP32,
                                       min_speedup=50.0, max_error=1e-5),
        )
        import numpy as np
        rng = np.random.default_rng(0)
        inputs = {"A": (rng.random((8, 4)) * 0.1).astype(np.float32),
                  "B": (rng.random((4, 8)) * 0.1).astype(np.float32)}
        conductor.execute(task, inputs, target_gflops=0.0)

        # Search 只被问了一次; 第 2 轮起是轮换, 不是重新问 LLM
        self.assertEqual(fake.calls, 1,
                         "轮换没有生效 —— 每轮都在重新问 Search")
        # 轮次血缘: r2 用 Search 选中的先验 (128x64x32), r3 消费队列头 alt1
        # (32x64x32) —— r3 的 sketch 不再来自 LLM, 而是排队候选
        self.assertGreaterEqual(len(seen_sketches), 3,
                                "第 3 轮没有发生 —— 轮换候选没被消费")
        prior_t = seen_sketches[1].tiling
        self.assertEqual((prior_t.block_m, prior_t.block_n, prior_t.block_k),
                         (128, 64, 32))
        rotated_t = seen_sketches[2].tiling
        self.assertEqual((rotated_t.block_m, rotated_t.block_n, rotated_t.block_k),
                         (32, 64, 32),
                         "第 3 轮没有消费队列头的邻域候选")


if __name__ == "__main__":
    unittest.main(verbosity=2)
