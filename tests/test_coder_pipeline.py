"""CoderPipeline 流程测试 (Plan→并行采样→剪枝→修复→回退)

覆盖 6 个特性:
1. Plan 阶段产出结构化计划
2. Code 阶段仅纯代码 (提取后无解释)
3. 并行采样 3 候选 (不同 temperature)
4. compile-gate 早退剪枝 (P1-1): 收集通过者 (≤ L2_FINALISTS) 而非「首个通过
   即定终身」—— 返回值仍是第一名 (默认候选), 决赛名单与选择语义的验收在
   tests/test_finalist_selection.py
5. 全部失败→有状态修复循环 (≤2轮) 修复成功
6. 修复仍败→规则模板回退

运行: python3 tests/test_coder_pipeline.py  或  python3 -m unittest tests.test_coder_pipeline -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent.coder_pipeline import CoderPipeline, RepairState
from operator_agent.planner import PlannerAgent
from operator_agent.resource_check import MemCaps
from operator_agent.rule_templates import RuleTemplateCoder
from operator_agent.kernel_db import KernelDatabase
from operator_agent.prompt_manager import PromptManager
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec, UnifiedSketch,
    TilingStrategy,
    TaskType, TaskComplexity, Precision,
)

PYTHON_MATMUL = """import numpy as np

def kernel(A, B):
    return A @ B
"""

BROKEN_SYNTAX = "import numpy as np\ndef kernel(A, B:\n    return A @ B"
BROKEN_TRUNCATED = "import numpy as np\ndef kernel(A, B):\n    return (A @ B"

FENCED_MATMUL = f"```python\n{PYTHON_MATMUL}```"
REPAIR_MARKER = "待修复代码"
PLAN_MARKER = "实现计划"


class MockLLM:
    """可编程 LLM: 区分 chat_json(plan) / chat(采样/修复), 按 temperature 与 prompt 内容回复"""

    def __init__(self, chat_handler=None, plan=None):
        self.chat_calls: list[tuple[list, float]] = []
        self.json_calls: list[list] = []
        self._chat_handler = chat_handler or (lambda prompt, temp: "")
        self._plan = plan or {
            "algorithm": "matmul", "kernel_signature": "def kernel(A, B)",
            "jit_signature": "", "steps": ["alloc", "launch"],
            "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
            "risks": ["boundary"], "notes": "",
        }

    def chat(self, messages, temperature=None, max_tokens=None, raw=False, caller_tag="Unknown"):
        self.chat_calls.append((messages, temperature))
        return self._chat_handler(messages[-1]["content"], temperature or 0.1)

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        self.json_calls.append(messages)
        return dict(self._plan)


def _task(dsl="python", kernel="matmul") -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel, complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="mock_gpu", dsl=dsl, compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.5, max_error=1e-5),
    )


def _sketch(dsl="python") -> UnifiedSketch:
    return UnifiedSketch(algorithm="matmul", version=1, target_dsl=dsl)


# ─── 1. Plan 阶段 ────────────────────────────────────────────

class TestPlanStage(unittest.TestCase):

    def test_planner_outputs_structured_plan(self):
        llm = MockLLM()
        planner = PlannerAgent(KernelDatabase(), llm=llm)
        plan = planner.run(_task(), _sketch())
        self.assertEqual(plan["algorithm"], "matmul")
        self.assertIn("kernel_signature", plan)
        self.assertIn("steps", plan)
        self.assertEqual(len(llm.json_calls), 1)

    def test_planner_fallback_without_llm(self):
        planner = PlannerAgent(KernelDatabase(), llm=None)
        plan = planner.run(_task(), _sketch())
        self.assertEqual(plan["algorithm"], "matmul")
        self.assertTrue(plan["steps"])

    def test_pipeline_runs_plan_first(self):
        """Pipeline 先调 Planner (chat_json), 再采样 (chat)"""
        llm = MockLLM(chat_handler=lambda p, t: FENCED_MATMUL)
        pipe = CoderPipeline(llm, KernelDatabase())
        pipe.run(_sketch(), _task())
        self.assertEqual(len(llm.json_calls), 1)   # Plan 先行
        self.assertGreaterEqual(len(llm.chat_calls), 1)  # 随后采样


# ─── 2/3. 并行采样 + 仅纯代码 ────────────────────────────────

class TestParallelSampling(unittest.TestCase):

    def test_three_candidates_different_temperatures(self):
        seen_temps = []

        def handler(prompt, temp):
            seen_temps.append(temp)
            return FENCED_MATMUL if PLAN_MARKER not in prompt else ""  # plan 是 json, 不走 chat

        llm = MockLLM(chat_handler=handler)
        pipe = CoderPipeline(llm, KernelDatabase())
        pipe.run(_sketch(), _task())
        self.assertEqual(sorted(seen_temps), sorted(CoderPipeline.SAMPLE_TEMPERATURES))

    def test_sampled_output_is_pure_code(self):
        """采样后经 _extract_code, 不含 fence/解释"""
        llm = MockLLM(chat_handler=lambda p, t: f"这是代码:\n{FENCED_MATMUL}\n希望有用")
        pipe = CoderPipeline(llm, KernelDatabase())
        code = pipe.run(_sketch(), _task())
        self.assertTrue(code.startswith("import numpy"))
        self.assertNotIn("```", code)
        self.assertNotIn("希望有用", code)

    def test_first_pass_candidate_selected(self):
        """3 候选中第 2 个通过, 应选首个通过项"""
        seq = iter([BROKEN_SYNTAX, PYTHON_MATMUL, BROKEN_TRUNCATED])

        def handler(prompt, temp):
            return next(seq) if PLAN_MARKER not in prompt else ""

        llm = MockLLM(chat_handler=handler)
        pipe = CoderPipeline(llm, KernelDatabase())
        code = pipe.run(_sketch(), _task())
        self.assertIn("def kernel", code)
        self.assertIn("A @ B", code)

    def test_all_pass_returns_first(self):
        llm = MockLLM(chat_handler=lambda p, t: FENCED_MATMUL)
        pipe = CoderPipeline(llm, KernelDatabase())
        code = pipe.run(_sketch(), _task())
        self.assertIn("def kernel", code)


# ─── 4. 编译剪枝 (compile-gate = 早退剪枝, 选择在 Conductor 侧 L1+L2) ──

class TestCompileSelection(unittest.TestCase):

    def test_injected_compile_fn_used(self):
        calls = []

        def strict_compile(code, dsl):
            calls.append(len(code))
            return (False, ["injected: always fail"])

        llm = MockLLM(chat_handler=lambda p, t: PYTHON_MATMUL)
        pipe = CoderPipeline(llm, KernelDatabase(), compile_fn=strict_compile)
        # 全部"编译失败" -> 走修复 -> 仍败 -> 回退模板
        code = pipe.run(_sketch(), _task())
        self.assertGreater(len(calls), 0)   # 注入的 compile_fn 被调用
        self.assertIn("def kernel", code)   # 回退到规则模板


# ─── 5. 有状态修复循环 ──────────────────────────────────────

class TestRepairLoop(unittest.TestCase):

    def test_repair_succeeds_on_second_round(self):
        """采样全失败 -> 修复轮1失败 -> 修复轮2成功"""
        repair_calls = {"n": 0}

        def handler(prompt, temp):
            if REPAIR_MARKER in prompt:
                repair_calls["n"] += 1
                return BROKEN_SYNTAX if repair_calls["n"] == 1 else PYTHON_MATMUL
            return BROKEN_SYNTAX  # 采样全失败

        llm = MockLLM(chat_handler=handler)
        pipe = CoderPipeline(llm, KernelDatabase())
        code = pipe.run(_sketch(), _task())
        self.assertIn("def kernel", code)
        self.assertLessEqual(repair_calls["n"], CoderPipeline.MAX_REPAIR_ROUNDS)

    def test_repair_state_records_history(self):
        state = RepairState(max_attempts=2)
        state.record_failure("code1", ["err1"])
        state.record_failure("code2", ["err2"])
        self.assertEqual(len(state.failures), 2)
        self.assertTrue(state.should_continue())
        state.attempts = 2
        self.assertFalse(state.should_continue())

    def test_repair_exhausted_falls_to_template(self):
        """采样失败 + 修复 2 轮仍败 -> 规则模板"""
        llm = MockLLM(chat_handler=lambda p, t: BROKEN_SYNTAX)
        pipe = CoderPipeline(llm, KernelDatabase())
        code = pipe.run(_sketch(), _task())
        self.assertIn("def kernel", code)
        # matmul python 模板内容
        self.assertIn("A @ B", code)


# ─── 6. 规则模板回退 ────────────────────────────────────────

class TestRuleTemplateFallback(unittest.TestCase):

    def test_matmul_python_template(self):
        coder = RuleTemplateCoder()
        code = coder.generate(_task(dsl="python", kernel="matmul"), _sketch())
        self.assertIn("A @ B", code)
        self.assertIn("def kernel", code)

    def test_matmul_triton_template(self):
        coder = RuleTemplateCoder()
        code = coder.generate(_task(dsl="triton", kernel="matmul"), _sketch("triton"))
        self.assertIn("@triton.jit", code)
        self.assertIn("def kernel", code)

    def test_softmax_template(self):
        coder = RuleTemplateCoder()
        code = coder.generate(_task(kernel="softmax"), _sketch())
        self.assertIn("np.exp", code)
        self.assertIn("def kernel", code)

    def test_unsupported_returns_empty(self):
        coder = RuleTemplateCoder()
        code = coder.generate(_task(kernel="exotic_op"), _sketch())
        self.assertEqual(code, "")

    def test_pipeline_falls_back_when_all_fail(self):
        llm = MockLLM(chat_handler=lambda p, t: BROKEN_SYNTAX)
        pipe = CoderPipeline(llm, KernelDatabase())
        code = pipe.run(_sketch(), _task())
        self.assertIn("def kernel", code)


# ─── 集成: CoderAgent 注入 pipeline ──────────────────────────

class TestCoderAgentWithPipeline(unittest.TestCase):

    def test_agent_uses_pipeline_when_injected(self):
        from operator_agent.agents import CoderAgent
        llm = MockLLM(chat_handler=lambda p, t: FENCED_MATMUL)
        pipe = CoderPipeline(llm, KernelDatabase())
        coder = CoderAgent(KernelDatabase(), llm=llm, pipeline=pipe)
        code = coder.run(_sketch(), _task())
        self.assertIn("def kernel", code)

    def test_agent_legacy_without_pipeline(self):
        """未注入 pipeline 时走 legacy 单线程生成 (兼容)"""
        from operator_agent.agents import CoderAgent
        llm = MockLLM(chat_handler=lambda p, t: PYTHON_MATMUL)
        coder = CoderAgent(KernelDatabase(), llm=llm)
        code = coder.run(_sketch(), _task())
        self.assertIn("def kernel", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ─── I75-G1: compile-gate 的资源门 ─────────────────────────────

TRITON_BIG_TILE = """import triton
import triton.language as tl

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tl.store(c_ptr, acc)

def kernel(A, B):
    _mm[(1,)](A, B, A, 1, 1, 1, BLOCK_M=256, BLOCK_N=256, BLOCK_K=64)
    return A
"""
# Historical proxy read this accumulator as 384KiB shared; no physical layout is proved.
TRITON_SMALL_TILE = TRITON_BIG_TILE.replace(
    "BLOCK_M=256, BLOCK_N=256, BLOCK_K=64", "BLOCK_M=64, BLOCK_N=64, BLOCK_K=32")


class TestResourceGate(unittest.TestCase):
    """I75-G1: 不可行 tile 在采样处剪掉 (不烧 SSH), 与 L1 同一份估算、同一漏拦口径"""

    def _pipe(self, caps):
        llm = MockLLM()          # 门测试不该调 LLM; 调到会炸出可读错误
        pipe = CoderPipeline(llm, KernelDatabase(),
                             compile_fn=lambda code, dsl: (bool(code), []))
        pipe.caps = caps
        return pipe

    def test_a_logical_tile_is_not_pruned_without_shared_storage_evidence(self):
        pipe = self._pipe(MemCaps(228))
        self.assertEqual([TRITON_BIG_TILE], pipe._prune_to_finalists(
            [TRITON_BIG_TILE], "triton", "matmul"))

    def _proven_oversized(self):
        from operator_agent.resource_check import (SharedMemoryEvidence, assess_shared_memory,
                                                   source_resource_binding)
        binding = source_resource_binding(TRITON_BIG_TILE)
        return assess_shared_memory(SharedMemoryEvidence(binding=binding, kind='native_compiler',
            producer='controlled-test-compiler', native_bytes=384*1024), MemCaps(228), binding=binding)

    def test_an_explicit_shared_resource_refusal_still_reaches_the_gate(self):
        from unittest.mock import patch
        with patch('operator_agent.coder_pipeline.check_resource_feasibility',
                   return_value=self._proven_oversized()):
            self.assertEqual([], self._pipe(MemCaps(228))._prune_to_finalists(
                [TRITON_BIG_TILE], 'triton', 'matmul'))

    def test_a_fitting_tile_passes(self):
        pipe = self._pipe(MemCaps(228))
        self.assertEqual([TRITON_SMALL_TILE], pipe._prune_to_finalists(
            [TRITON_SMALL_TILE], "triton", "matmul"))

    def test_missing_caps_lets_it_through(self):
        """caps 缺 ⇒ 漏拦 (与 L1 同口径): 门不许比 L1 更严, 否则 caps 缺键的任务
        会在生成侧全军覆没而没人知道为什么"""
        pipe = self._pipe(None)
        self.assertEqual([TRITON_BIG_TILE], pipe._prune_to_finalists(
            [TRITON_BIG_TILE], "triton", "matmul"))

    def test_the_resource_reason_reaches_the_repair_baseline(self):
        """修复基线的 issues 带着上界约束 —— 修代码的提示从此有硬数字 (G3 的流程半边)"""
        pipe = self._pipe(MemCaps(228))
        from unittest.mock import patch
        with patch('operator_agent.coder_pipeline.check_resource_feasibility',
                   return_value=self._proven_oversized()):
            best = pipe._best_failure([TRITON_BIG_TILE], "triton", "matmul")
        self.assertIsNotNone(best)
        _, issues = best
        self.assertTrue(any("资源预检" in i and "228" in i for i in issues), issues)

    def test_the_conductor_injects_caps_from_hardware(self):
        """接线证人: hw_info → detect_hardware_caps → pipeline.caps (execute 内注入)"""
        import contextlib
        import io
        from operator_agent.conductor import Conductor
        from operator_agent.mock_gpu import MockGPU
        from tests.test_template_queue import _KernelLLM, _task, _inputs

        c = Conductor(backend=MockGPU(), llm=_KernelLLM(),
                      kernel_db=KernelDatabase(jsonl_path=None))
        with contextlib.redirect_stdout(io.StringIO()):
            c.execute(_task(), _inputs(), target_gflops=0.0)
        self.assertIsNotNone(c._coder_pipeline, "默认路径应构造 pipeline")
        self.assertIsNotNone(c._coder_pipeline.caps, "caps 应在 hw_info 后注入")


# ─── I75 后续: 确定性机械改写 (prompt 约束真机无效 → 生成后机械对齐) ───

TRITON_DOT_TILE = """import triton
import triton.language as tl

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    acc = tl.dot(a, b, acc)
    tl.store(c_ptr, acc)

def kernel(A, B):
    _mm[(1,)](A, B, A, 1, 1, 1, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return A
"""


def _sketch_646432() -> UnifiedSketch:
    """sketch 64×64×32 —— I75 真机场景 (H20 smem 228KiB + fp32 matmul)"""
    return UnifiedSketch(
        algorithm="matmul", version=1, target_dsl="triton",
        tiling=TilingStrategy(block_m=64, block_n=64, block_k=32),
    )


class TestMechanicalRewrite(unittest.TestCase):
    """I75 后续: LLM 无视 prompt 硬约束 (256 大 tile / fp32 不带 input_precision),
    生成后用确定性文本改写兜住 —— 见 coder_pipeline.normalize_candidate"""

    def _pipe(self) -> CoderPipeline:
        return CoderPipeline(
            MockLLM(), KernelDatabase(),
            compile_fn=lambda code, dsl: (bool(code), []))

    def _task_fp(self, precision: Precision) -> TaskSchema:
        t = _task(dsl="triton")
        t.constraints = ConstraintSpec(precision=precision, min_speedup=0.5,
                                       max_error=1e-5)
        return t

    def test_big_tile_rewritten_to_sketch(self):
        """256 大 tile + sketch(64,64,32) → 三维改回 sketch 值, notes 非空"""
        pipe = self._pipe()
        code, notes = pipe.normalize_candidate(
            TRITON_BIG_TILE, _sketch_646432(), _task(dsl="triton"))
        self.assertIn("BLOCK_M=64", code)
        self.assertIn("BLOCK_N=64", code)
        self.assertIn("BLOCK_K=32", code)
        self.assertNotIn("BLOCK_M=256", code)
        self.assertNotIn("BLOCK_N=256", code)
        self.assertTrue(any("BLOCK_M 256→64" in n for n in notes), notes)
        self.assertTrue(notes)

    def test_matching_tile_returned_verbatim(self):
        """BLOCK 已等于 sketch → 原样 (同一字符串对象), notes 为空"""
        pipe = self._pipe()
        code, notes = pipe.normalize_candidate(
            TRITON_SMALL_TILE, _sketch_646432(), _task(dsl="triton"))
        self.assertIs(code, TRITON_SMALL_TILE)
        self.assertEqual(notes, [])

    def test_no_block_literals_returned_verbatim(self):
        """无 BLOCK 字面量 (triton) / 纯 python numpy 代码 → 原样"""
        pipe = self._pipe()
        sketch = _sketch_646432()
        no_block = ("import triton\nimport triton.language as tl\n\n"
                    "@triton.jit\ndef kernel(A):\n    return A\n")
        for raw, dsl in ((no_block, "triton"), (PYTHON_MATMUL, "python")):
            code, notes = pipe.normalize_candidate(raw, sketch, _task(dsl=dsl))
            self.assertIs(code, raw)
            self.assertEqual(notes, [])

    def test_fp32_dot_pinned_to_ieee(self):
        """fp32 + tl.dot(a, b, acc) 无 input_precision → 追加 ieee"""
        pipe = self._pipe()
        code, notes = pipe.normalize_candidate(
            TRITON_DOT_TILE, _sketch_646432(), _task(dsl="triton"))
        self.assertIn('tl.dot(a, b, acc, input_precision="ieee")', code)
        self.assertTrue(any("ieee" in n for n in notes), notes)

    def test_dot_already_with_input_precision_untouched(self):
        """已含 input_precision → 不重复追加, 原样返回"""
        pipe = self._pipe()
        already = TRITON_DOT_TILE.replace(
            "tl.dot(a, b, acc)", 'tl.dot(a, b, acc, input_precision="ieee")')
        code, notes = pipe.normalize_candidate(
            already, _sketch_646432(), _task(dsl="triton"))
        self.assertIs(code, already)
        self.assertEqual(notes, [])

    def test_fp16_dot_not_pinned(self):
        """精度非 fp32 (fp16) → 不钉 ieee"""
        pipe = self._pipe()
        code, notes = pipe.normalize_candidate(
            TRITON_DOT_TILE, _sketch_646432(), self._task_fp(Precision.FP16))
        self.assertIs(code, TRITON_DOT_TILE)
        self.assertNotIn("input_precision", code)
        self.assertEqual(notes, [])

    def test_nested_dot_not_pinned_leaves_note(self):
        """嵌套括号的 tl.dot 匹配不到 → 代码不动, 留「人工注意」note"""
        pipe = self._pipe()
        nested = TRITON_DOT_TILE.replace(
            "tl.dot(a, b, acc)", "tl.dot(a, tl.trans(b), acc)")
        code, notes = pipe.normalize_candidate(
            nested, _sketch_646432(), _task(dsl="triton"))
        self.assertNotIn("input_precision", code)
        self.assertTrue(any("未钉住" in n for n in notes), notes)

    def test_missing_sketch_dim_left_alone(self):
        """sketch 缺某维 (SimpleNamespace 伪装) → 该维不改写, 只改有依据的"""
        pipe = self._pipe()
        sketch = SimpleNamespace(tiling=SimpleNamespace(block_m=64))  # 缺 N/K
        code, notes = pipe.normalize_candidate(
            TRITON_BIG_TILE, sketch, _task(dsl="triton"))
        self.assertIn("BLOCK_M=64", code)      # M 有依据 → 改
        self.assertIn("BLOCK_N=256", code)     # N 无依据 → 不动
        self.assertIn("BLOCK_K=64", code)      # K 无依据 → 不动
        self.assertEqual(notes, ["BLOCK_M 256→64 (sketch)"])

    def test_run_level_big_tile_survives_the_gate(self):
        """端到端证人: LLM 恒返 256 大 tile (真机 I75 行为), 机械改写后过
        资源门 (改写前 384KiB > 228KiB 会被剪光) → last_finalists 非空"""
        llm = MockLLM(chat_handler=lambda p, t: TRITON_BIG_TILE)
        pipe = CoderPipeline(llm, KernelDatabase(jsonl_path=None),
                             compile_fn=lambda code, dsl: (bool(code), []))
        pipe.caps = MemCaps(228)
        code = pipe.run(_sketch_646432(), _task(dsl="triton", kernel="matmul"))
        self.assertTrue(pipe.last_finalists)
        self.assertIn("BLOCK_M=64", code)
        self.assertNotIn("BLOCK_M=256", code)
