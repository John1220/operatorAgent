"""FixerAgent 与 Conductor 修复闭环的单元测试

覆盖:
1. _truncate_error: 错误日志截断 (短日志原样 / 长日志保头尾)
2. _extract_code:   代码提取 (代码块 / 无代码块 / 提取失败)
3. FixerAgent.run:  prompt 注入正确性、无 LLM 报错、提取失败报错
4. Conductor 闭环:  L1 失败 -> Fixer 修复 -> 验证通过 (端到端)
5. Conductor 预算:  修复持续失败 -> 达到最大轮次后终止

运行: python3 tests/test_fixer.py  或  python3 -m unittest tests.test_fixer -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent.fixer import FixerAgent
from operator_agent.conductor import Conductor
from operator_agent.mock_gpu import MockGPU
from operator_agent.kernel_db import KernelDatabase
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision,
    VerificationResult, VerificationLevel,
)

# ─── 测试用 kernel 代码 ──────────────────────────────────────

VALID_KERNEL = """import numpy as np

def kernel(A, B):
    return A @ B
"""

BROKEN_KERNEL = """import numpy as np

def kernel(A, B:
    return A @ B
"""

FENCED_VALID_KERNEL = f"```python\n{VALID_KERNEL}```"

# Fixer prompt 模板 (fixer_repair.md) 中的特征标记，用于区分 Coder/Fixer 调用
FIXER_PROMPT_MARKER = "待修复代码"


# ─── Mock LLM ────────────────────────────────────────────────

class MockLLM:
    """可编程 LLM mock: 记录全部调用，chat 按 handler 策略回复"""

    def __init__(self, chat_handler=None):
        self.chat_calls: list[list[dict]] = []
        self.chat_json_calls: list[list[dict]] = []
        self._chat_handler = chat_handler or (lambda prompt: "")
        self._sketch = {
            "algorithm": "matmul",
            "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
            "memory_plan": {},
            "pipeline": [],
            "optimization_hints": [],
        }

    def chat(self, messages, temperature=None, max_tokens=None, raw=False, caller_tag="Unknown"):
        self.chat_calls.append(messages)
        return self._chat_handler(messages[-1]["content"])

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        self.chat_json_calls.append(messages)
        return dict(self._sketch)

    def fixer_prompts(self) -> list[str]:
        return [c[-1]["content"] for c in self.chat_calls
                if FIXER_PROMPT_MARKER in c[-1]["content"]]


def _make_task() -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python",
                          compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32,
                                   min_speedup=0.5, max_error=1e-5),
    )


def _make_inputs() -> dict:
    rng = np.random.default_rng(0)
    return {"A": (rng.random((8, 4)) * 0.1).astype(np.float32),
            "B": (rng.random((4, 8)) * 0.1).astype(np.float32)}


# ─── 1. 错误日志截断 ─────────────────────────────────────────

class TestTruncateError(unittest.TestCase):

    def test_short_log_unchanged(self):
        log = "SyntaxError: invalid syntax (line 3)"
        self.assertEqual(FixerAgent._truncate_error(log), log)

    def test_empty_log(self):
        self.assertEqual(FixerAgent._truncate_error(""), "")
        self.assertEqual(FixerAgent._truncate_error(None), "")

    def test_long_log_keeps_head_and_tail(self):
        head = "E" * 1000
        tail = "T" * 1000
        log = head + "M" * 3000 + tail
        out = FixerAgent._truncate_error(log, max_chars=1000)
        self.assertTrue(out.startswith(head[:500]))
        self.assertTrue(out.endswith(tail[-500:]))
        self.assertIn("省略", out)
        self.assertLess(len(out), len(log))

    def test_custom_limit(self):
        log = "x" * 500
        out = FixerAgent._truncate_error(log, max_chars=100)
        self.assertLessEqual(len(out), 100 + 40)  # 100 + 省略标记


# ─── 2. 代码提取 ─────────────────────────────────────────────

class TestExtractCode(unittest.TestCase):

    def test_fenced_block(self):
        raw = f"这是修复后的代码:\n{FENCED_VALID_KERNEL}\n希望有帮助"
        self.assertEqual(FixerAgent._extract_code(raw), VALID_KERNEL.strip())

    def test_thinking_tag_stripped(self):
        raw = f"<think>分析一下错误原因...</think>\n{FENCED_VALID_KERNEL}"
        self.assertEqual(FixerAgent._extract_code(raw), VALID_KERNEL.strip())

    def test_plain_code_without_fence(self):
        raw = f"修复如下\n{VALID_KERNEL}"
        self.assertEqual(FixerAgent._extract_code(raw), VALID_KERNEL.strip())

    def test_triton_code_preserved(self):
        triton = "@triton.jit\ndef matmul_kernel(...):\n    pass"
        raw = f"{triton}\n```"
        self.assertEqual(FixerAgent._extract_code(raw), triton)

    def test_extract_dedents_indented_fence(self):
        """LLM 把代码块放在缩进列表内时, 提取后必须去缩进以保证可编译"""
        raw = ("代码如下:\n"
               "    ```python\n"
               "    import triton\n"
               "    import torch\n"
               "    def kernel(A, B):\n"
               "        return A @ B\n"
               "    ```")
        out = FixerAgent._extract_code(raw)
        self.assertTrue(out.startswith("import triton"))
        self.assertNotIn("\n    import", out)  # 行首缩进已去除
        compile(out, "<t>", "exec")  # 去缩进后可编译

    def test_extract_dedents_indented_plain(self):
        """前言 + 一致缩进的纯代码: 提取并去缩进后可编译"""
        raw = "以下是修复后的代码:\n    import numpy as np\n    def kernel(A, B):\n        return A @ B"
        out = FixerAgent._extract_code(raw)
        self.assertTrue(out.startswith("import numpy"))
        compile(out, "<t>", "exec")

    def test_no_code_returns_empty(self):
        self.assertEqual(FixerAgent._extract_code("无法修复，建议重写"), "")

    def test_no_semantic_rewrite(self):
        """提取器不得做 float32->float64 之类的语义改写"""
        code = "import triton\nacc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)"
        self.assertEqual(FixerAgent._extract_code(code), code)


# ─── 3. FixerAgent 单元行为 ──────────────────────────────────

class TestFixerAgent(unittest.TestCase):

    def test_prompt_contains_error_and_code(self):
        llm = MockLLM(chat_handler=lambda p: FENCED_VALID_KERNEL)
        fixer = FixerAgent(llm=llm)
        failed = [VerificationResult(
            level=VerificationLevel.L1_COMPILE, passed=False,
            message="编译失败: Missing 'kernel' function",
            error_log="SyntaxError: invalid syntax",
        )]
        fixed, summary = fixer.run(BROKEN_KERNEL, _make_task(), failed)

        prompt = llm.chat_calls[0][-1]["content"]
        self.assertIn("编译失败", prompt)          # 失败消息已注入
        self.assertIn("SyntaxError", prompt)       # 错误日志已注入
        self.assertIn("def kernel(A, B:", prompt)  # 原始失败代码已注入
        self.assertEqual(fixed, VALID_KERNEL.strip())
        self.assertIn("L1", summary)

    def test_long_error_log_truncated_in_prompt(self):
        llm = MockLLM(chat_handler=lambda p: FENCED_VALID_KERNEL)
        fixer = FixerAgent(llm=llm)
        failed = [VerificationResult(
            level=VerificationLevel.L1_COMPILE, passed=False,
            message="编译失败", error_log="x" * 10000,
        )]
        fixer.run(BROKEN_KERNEL, _make_task(), failed)
        prompt = llm.chat_calls[0][-1]["content"]
        self.assertNotIn("x" * 10000, prompt)
        self.assertIn("省略", prompt)

    def test_no_llm_raises(self):
        fixer = FixerAgent(llm=None)
        with self.assertRaises(RuntimeError):
            fixer.run(BROKEN_KERNEL, _make_task(), [])

    def test_unextractable_reply_raises(self):
        llm = MockLLM(chat_handler=lambda p: "我不知道怎么修")
        fixer = FixerAgent(llm=llm)
        with self.assertRaises(ValueError):
            fixer.run(BROKEN_KERNEL, _make_task(), [
                VerificationResult(level=VerificationLevel.L1_COMPILE,
                                   passed=False, message="编译失败"),
            ])


# ─── 4. Conductor 修复闭环 (端到端) ─────────────────────────

class TestConductorRepairLoop(unittest.TestCase):

    def _make_conductor(self, llm, max_rounds=3) -> Conductor:
        return Conductor(
            backend=MockGPU(),
            kernel_db=KernelDatabase(),
            llm=llm,
            max_optimize_rounds=max_rounds,
        )

    def test_l1_failure_repaired_and_passed(self):
        """Coder 首轮产出坏代码 -> L1 失败 -> Fixer 修复 -> 全部验证通过"""
        def handler(prompt: str) -> str:
            if FIXER_PROMPT_MARKER in prompt:
                return FENCED_VALID_KERNEL      # Fixer: 返回修复后的好代码
            return BROKEN_KERNEL                # Coder: 首轮返回坏代码

        llm = MockLLM(chat_handler=handler)
        conductor = self._make_conductor(llm)
        artifact = conductor.execute(
            _make_task(), _make_inputs(), target_gflops=0.0,
        )

        self.assertTrue(artifact.all_passed)
        self.assertIn("def kernel", artifact.code)
        self.assertEqual(len(llm.fixer_prompts()), 1)   # Fixer 恰好调用 1 次
        # Coder 预检重试 3 次 (P1-1) + Fixer 1 次
        self.assertEqual(len(llm.chat_calls), 4)
        # 修复 prompt 中必须携带 L1 失败信息
        self.assertIn("L1", llm.fixer_prompts()[0])

    def test_budget_exhausted_stops_with_failure(self):
        """修复持续失败 -> 达到最大轮次后终止，不无限循环"""
        llm = MockLLM(chat_handler=lambda p: BROKEN_KERNEL)
        conductor = self._make_conductor(llm, max_rounds=3)
        artifact = conductor.execute(
            _make_task(), _make_inputs(), target_gflops=0.0,
        )

        self.assertFalse(artifact.all_passed)
        # Coder 预检重试 3 次 (P1-1) + Fixer 2 次 (末轮不再修复)
        self.assertEqual(len(llm.chat_calls), 5)
        self.assertEqual(len(llm.fixer_prompts()), 2)   # 末轮不再修复
        # 最终验证结果停留在 L1 失败
        l1 = artifact.verification_results[0]
        self.assertEqual(l1.level, VerificationLevel.L1_COMPILE)
        self.assertFalse(l1.passed)

    def test_fixer_exception_breaks_gracefully(self):
        """Fixer 提取不到有效代码 -> 优雅终止而非抛异常"""
        def handler(prompt: str) -> str:
            if FIXER_PROMPT_MARKER in prompt:
                return "无法修复"                 # Fixer 回复无法提取代码
            return BROKEN_KERNEL

        llm = MockLLM(chat_handler=handler)
        conductor = self._make_conductor(llm)
        artifact = conductor.execute(
            _make_task(), _make_inputs(), target_gflops=0.0,
        )
        self.assertFalse(artifact.all_passed)
        self.assertEqual(len(llm.fixer_prompts()), 1)   # 修复失败后即终止


if __name__ == "__main__":
    unittest.main(verbosity=2)
