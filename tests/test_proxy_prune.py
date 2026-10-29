"""§12 的便宜代理剪枝**接在生产链路上了没有**

`tests/test_candidates.py` 证的是判据本身算得对; 这里证三处接线:
① Verifier 把每 SM smem 上界发布进 L3 metrics (与 L1 预检同一个 caps 真源);
② Conductor 把上界与任务 `elem_bytes` 放进 `perf_data`；精度不是实际共享缓冲区证据；
③ SearchAgent 缺少 harness 资源契约时保留点，且「不知道」与「放得下」分得开。

③ 是这一族最常见的失效形状: 判据写好了、有单测, 但生产路径上没人读它。

运行: python3 -m unittest tests.test_proxy_prune -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.agents import SearchAgent
from operator_agent.candidates import ParamSpace, TemplateCandidate, template_resource_binding
from operator_agent.conductor import Conductor
from operator_agent.kernel_db import KernelDatabase
from operator_agent.resource_check import SharedMemoryEvidence
from operator_agent.schemas import TilingStrategy, UnifiedSketch
from tests.test_measurement_validity import (
    LaunchDominatedGPU, NET_MS_MATMUL_4096,
)
from tests.test_search_belief_gate import _LoopLlm, _inputs, _task


def _sketch(block_m=128, block_n=64, block_k=32):
    return UnifiedSketch(algorithm="matmul", version=1,
                         tiling=TilingStrategy(block_m=block_m, block_n=block_n,
                                               block_k=block_k),
                         memory_plan={}, pipeline=[], target_dsl="triton",
                         optimization_hints=[], hints_this_version=[])


class _RecordingSearch(SearchAgent):
    """记下每轮拿到的 `perf_data` —— 剪枝的输入是不是真的到了这里, 只能这样看"""

    def __init__(self):
        super().__init__(llm=None, prompt_manager=None)
        self.perf_seen = []

    def run(self, sketch, perf_data):
        self.perf_seen.append(dict(perf_data))
        return sketch, "桩: 不动"


class TestTheLimitReachesTheSearcher(unittest.TestCase):

    maxDiff = 600

    def test_the_verifier_publishes_the_shared_memory_limit(self):
        """L3 metrics 里必须有这个键, 且 > 0 (mock 后端报的是 H20: 228 KB)"""
        c = Conductor(backend=LaunchDominatedGPU(NET_MS_MATMUL_4096),
                      kernel_db=KernelDatabase(jsonl_path=None), llm=_LoopLlm(),
                      max_optimize_rounds=3, heartbeat_interval=1000)
        search = _RecordingSearch()
        c.search = search
        with contextlib.redirect_stdout(io.StringIO()):
            c.execute(_task(min_speedup=5.0), _inputs(), None, target_gflops=0.0)
        self.assertTrue(search.perf_seen, "前提: 这一跑要走优化分支")
        perf = search.perf_seen[0]
        self.assertIn("shared_mem_per_sm_kb", perf, "硬件上界仍须传到搜索侧")
        self.assertEqual(228, perf["shared_mem_per_sm_kb"], "H20 的 caps 是 228 KB")
        self.assertEqual(4, perf["elem_bytes"], "fp32 声明精度 = 4 字节")

    def test_without_the_key_the_pruner_says_it_did_nothing(self):
        """缺键 = **未生效**, 不许被读成「剪过了没事」"""
        a = SearchAgent(llm=None, prompt_manager=None)
        kept, note = a._prune_by_smem([_sketch(), _sketch(256, 128, 64)],
                                      {"elem_bytes": 4})
        self.assertEqual(2, len(kept))
        self.assertEqual("", note, "三态: 不知道 ⇒ 不剪且不报")


class TestThePrunerActsInTheProductionPath(unittest.TestCase):

    def test_large_and_small_neighbors_remain_unknown_without_storage_contracts(self):
        a = SearchAgent(llm=None, prompt_manager=None)
        big = _sketch(512, 512, 128)
        small = _sketch(64, 64, 32)
        kept, note = a._prune_by_smem([small, big],
                                      {"shared_mem_per_sm_kb": 64, "elem_bytes": 4})
        self.assertEqual([small, big], kept, "大小 tile 都没有物理存储证据，不能假拒")
        self.assertIn("代理剪枝", note)
        self.assertIn("未知", note)
        self.assertNotIn("下界", note)
        self.assertNotIn("摘掉", note)

    def test_pruning_a_whole_neighborhood_still_leaves_a_round_to_run(self):
        a = SearchAgent(llm=None, prompt_manager=None)
        pts = [_sketch(256, 256, 128), _sketch(512, 512, 128)]
        kept, note = a._prune_by_smem(pts, {"shared_mem_per_sm_kb": 1,
                                            "elem_bytes": 4})
        self.assertEqual(2, len(kept), "物理占用未知 ⇒ 保留待原生检查")
        self.assertIn("未知", note)
        self.assertIn("没摘不等于放得下", note)
        self.assertNotIn("全超上界", note, "unknown 不是全部已拒后的防饿死回退")

    def test_the_note_reaches_the_suggestion_printed_upstream(self):
        """剪枝的说明要跟着 suggestion 走 —— 只在内部记账等于没说"""
        a = SearchAgent(llm=None, prompt_manager=None)
        prior = _sketch(128, 64, 32)
        chosen, alts, note = a._neighborhood_search(
            prior, origin=_sketch(64, 64, 32),
            perf_data={"shared_mem_per_sm_kb": 1, "elem_bytes": 4})
        self.assertIn("代理剪枝", note)
        self.assertIn("未知", note)

    def test_user_perf_data_cannot_supply_trusted_resource_receipts(self):
        a = SearchAgent(llm=None, prompt_manager=None)
        sketches = [_sketch(64, 64, 32), _sketch(512, 512, 128)]
        receipts = {}
        for index, sketch in enumerate(sketches):
            tile = sketch.tiling
            point = TemplateCandidate("lattice", "tiling", {
                "BLOCK_M": tile.block_m, "BLOCK_N": tile.block_n,
                "BLOCK_K": tile.block_k, "_idx": index})
            binding = template_resource_binding(point)
            receipts[binding] = SharedMemoryEvidence(
                binding=binding, kind="native_compiler", producer="user-perf-data-claim",
                native_bytes=1024 if index == 0 else 512 * 1024)
        kept, note = a._prune_by_smem(sketches, {
            "shared_mem_per_sm_kb": 64, "elem_bytes": 4,
            "resource_evidence": receipts,
            "shared_memory_evidence": receipts,
            "shared_bytes_per_cta": 512 * 1024,
            "estimated_smem_bytes": 512 * 1024,
        })
        self.assertEqual(sketches, kept,
                         "perf_data 里的用户自报值不能升级为 harness 可信证据")
        self.assertIn("未知", note)
        self.assertNotIn("摘掉", note)


class TestTheSwitchIsWiredIntoTheController(unittest.TestCase):

    def test_a_template_backend_stops_the_run_loudly(self):
        """后端声明了参数空间 ⇒ 控制器 fail-closed, 而不是把它当源码型继续跑"""

        class _TemplateGPU(LaunchDominatedGPU):
            def param_space(self, contract=None):
                return ParamSpace(template_id="cutlass_sm80_gemm",
                                  axes={"BM": (64, 128)}, backend="cutlass")

        c = Conductor(backend=_TemplateGPU(NET_MS_MATMUL_4096),
                      kernel_db=KernelDatabase(jsonl_path=None), llm=_LoopLlm(),
                      max_optimize_rounds=3, heartbeat_interval=1000)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(NotImplementedError) as ctx:
                c.execute(_task(min_speedup=5.0), _inputs(), None, target_gflops=0.0)
        self.assertIn("cutlass_sm80_gemm", str(ctx.exception))

    def test_a_code_type_backend_runs_normally(self):
        """反向: 代码型后端 (今天全部) 不受这个开关影响"""
        search = _RecordingSearch()
        c = Conductor(backend=LaunchDominatedGPU(NET_MS_MATMUL_4096),
                      kernel_db=KernelDatabase(jsonl_path=None), llm=_LoopLlm(),
                      max_optimize_rounds=3, heartbeat_interval=1000)
        c.search = search
        with contextlib.redirect_stdout(io.StringIO()):
            art = c.execute(_task(min_speedup=5.0), _inputs(), None, target_gflops=0.0)
        self.assertIsNotNone(art)


if __name__ == "__main__":
    unittest.main(verbosity=2)
