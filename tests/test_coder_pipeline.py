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
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent.coder_pipeline import CoderPipeline, RepairState
from operator_agent.planner import PlannerAgent
from operator_agent.rule_templates import RuleTemplateCoder
from operator_agent.kernel_db import KernelDatabase
from operator_agent.prompt_manager import PromptManager
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec, UnifiedSketch,
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
