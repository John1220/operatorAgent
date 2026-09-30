"""P1-1 (docs/design/generation-side-review.md): 候选选择的验收测试

审查文档换掉的验收方式 (golden 断言两路径「产出等价」对随机路径不可实施),
按新口径钉四件事:
  1. 默认翻转: Conductor(llm=...) 默认走 CoderPipeline, 单体路径是逃生门;
  2. 剪枝结构: 3 路采样照旧; 截断候选不进决赛; 决赛 ≤ L2_FINALISTS 个,
     按 temperature 序; 修复/模板产物是单候选;
  3. 选择语义: 决赛候选逐个跑 L1+L2, 首个通过者胜出; L3 只跑胜者一次
     (选择权在验证侧, 机时不随候选数全额翻倍 —— 「前 2 名上 L2」折中);
  4. 选择不是门: 决赛全败时保持默认候选交完整验证, Conductor 的 Fixer
     照常接手 (pipeline 路径的修复闭环端到端)。

运行: python -m pytest tests/test_finalist_selection.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent.conductor import Conductor
from operator_agent.coder_pipeline import CoderPipeline
from operator_agent.mock_gpu import MockGPU
from operator_agent.kernel_db import KernelDatabase
from operator_agent.prompt_manager import PromptManager
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec, UnifiedSketch,
    TaskType, TaskComplexity, Precision,
)

# ─── 测试用代码 ──────────────────────────────────────────────

RIGHT = """import numpy as np

def kernel(A, B):
    return A @ B
"""

# 形状对、值错: 过预检 (语法/结构)、过 L1 (可编译可执行), 只在 L2 数值比对时暴露
WRONG = """import numpy as np

def kernel(A, B):
    return A @ A.T
"""

BROKEN_TRUNCATED = "import numpy as np\ndef kernel(A, B):\n    return (A @ B"

# 模板标记: 区分 pipeline 内部修复 (coder_repair.md) 与 Conductor 的 Fixer (fixer_repair.md)
PIPELINE_REPAIR_MARKER = "未通过预检"     # 仅 coder_repair.md 开头有
FIXER_MARKER = "## 错误信息"              # 仅 fixer_repair.md 有


class MockLLM:
    """可编程 LLM: chat 按 (prompt 标记, temperature) 分流; chat_json 回 sketch/plan 通用 dict"""

    def __init__(self, by_temp=None, repair_reply=None, fixer_reply=None):
        self.chat_calls: list[tuple[str, float]] = []   # (prompt, temperature)
        self._by_temp = by_temp or {}                   # {temperature: code}
        self._repair_reply = repair_reply
        self._fixer_reply = fixer_reply

    def chat(self, messages, temperature=None, max_tokens=None, raw=False, caller_tag="Unknown"):
        prompt = messages[-1]["content"]
        self.chat_calls.append((prompt, temperature))
        if PIPELINE_REPAIR_MARKER in prompt and self._repair_reply is not None:
            return self._repair_reply
        if FIXER_MARKER in prompt and self._fixer_reply is not None:
            return self._fixer_reply
        return self._by_temp.get(temperature, "")

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        # Designer 与 Planner 共用: 两边都只 setdefault 自己缺的字段
        return {
            "algorithm": "matmul",
            "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
            "memory_plan": {},
            "pipeline": [],
            "optimization_hints": [],
        }


def _task() -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python",
                          compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32,
                                   min_speedup=0.5, max_error=1e-5),
    )


def _sketch() -> UnifiedSketch:
    return UnifiedSketch(algorithm="matmul", version=1, target_dsl="python")


def _inputs() -> dict:
    rng = np.random.default_rng(0)
    return {"A": (rng.random((8, 4)) * 0.1).astype(np.float32),
            "B": (rng.random((4, 8)) * 0.1).astype(np.float32)}


def _pipeline(llm) -> CoderPipeline:
    return CoderPipeline(llm=llm, kernel_db=KernelDatabase(jsonl_path=None),
                         prompt_manager=PromptManager())


def _spy(verifier, method_name) -> list:
    """包住 verifier 的某级方法, 记录每次收到的 kernel_code (不换行为)"""
    seen: list = []
    real = getattr(verifier, method_name)

    def wrapper(kernel_code, *args, **kwargs):
        seen.append(kernel_code)
        return real(kernel_code, *args, **kwargs)

    setattr(verifier, method_name, wrapper)
    return seen


# ─── 1. 默认翻转 ─────────────────────────────────────────────

class TestDefaultFlip(unittest.TestCase):

    def test_pipeline_is_the_default_path(self):
        """P1-1 主张本体: 有 LLM 就默认走流水线 (单体路径不再是默认)"""
        conductor = Conductor(backend=MockGPU(),
                              kernel_db=KernelDatabase(jsonl_path=None),
                              llm=MockLLM())
        self.assertIsNotNone(conductor.coder.pipeline,
                             "默认路径不是 CoderPipeline —— 翻转被回退了")

    def test_legacy_path_is_the_escape_hatch(self):
        """单体路径保留: use_coder_pipeline=False 显式关掉"""
        conductor = Conductor(backend=MockGPU(),
                              kernel_db=KernelDatabase(jsonl_path=None),
                              llm=MockLLM(),
                              use_coder_pipeline=False)
        self.assertIsNone(conductor.coder.pipeline)
        self.assertEqual(conductor.coder.last_finalists, [],
                         "单体路径无决赛概念, 名单应为空")


# ─── 2. 剪枝结构 (管线单元级) ────────────────────────────────

class TestPruneStructure(unittest.TestCase):

    def test_three_samples_truncated_pruned_finalists_capped(self):
        """3 路采样照旧; 截断候选被剪出决赛; 决赛 ≤ 2 且按 temperature 序"""
        llm = MockLLM(by_temp={0.1: BROKEN_TRUNCATED, 0.4: WRONG, 0.7: RIGHT})
        pipeline = _pipeline(llm)
        code = pipeline.run(_sketch(), _task(), test_inputs=_inputs())
        # 采样恰好 3 次 (temps 0.1/0.4/0.7 各一)
        temps = [t for _, t in llm.chat_calls if _is_sample(llm, t)]
        self.assertEqual(sorted(temps), [0.1, 0.4, 0.7])
        # 截断者被剪: 不在决赛, 也不作为返回值
        self.assertNotIn(BROKEN_TRUNCATED, pipeline.last_finalists)
        self.assertNotEqual(code, BROKEN_TRUNCATED)
        # 决赛 = [WRONG(t0.4), RIGHT(t0.7)], 上限 L2_FINALISTS=2, temperature 序
        # (比较用 strip: 清洗会剥掉候选的尾部换行)
        self.assertEqual([c.strip() for c in pipeline.last_finalists],
                         [WRONG.strip(), RIGHT.strip()])
        # 返回值 = 决赛第一名 (默认候选)
        self.assertEqual(code.strip(), WRONG.strip())

    def test_repair_product_is_a_single_finalist(self):
        """全部采样失败 → 修复产物是单候选 (选择退化为不选)"""
        llm = MockLLM(by_temp={t: BROKEN_TRUNCATED for t in (0.1, 0.4, 0.7)},
                      repair_reply=RIGHT)
        pipeline = _pipeline(llm)
        code = pipeline.run(_sketch(), _task(), test_inputs=_inputs())
        self.assertEqual(code.strip(), RIGHT.strip())
        self.assertEqual(len(pipeline.last_finalists), 1)
        self.assertEqual(pipeline.last_finalists[0].strip(), RIGHT.strip())


def _is_sample(llm, temperature) -> bool:
    """采样调用 = handler 按 temperature 分流命中 (非修复/Fixer prompt)"""
    # 简化判据: 该 temperature 在 by_temp 里有显式映射且 prompt 不含修复标记
    return temperature in llm._by_temp


# ─── 3+4. 选择语义 (Conductor 端到端, 默认路径) ──────────────

class TestSelectionByL2(unittest.TestCase):

    def _conductor(self, llm) -> Conductor:
        # 不传 use_coder_pipeline: 端到端必须在**默认路径**上成立
        return Conductor(backend=MockGPU(),
                         kernel_db=KernelDatabase(jsonl_path=None),
                         llm=llm)

    def test_second_finalist_wins_and_l3_runs_once(self):
        """低温候选 (默认) L2 失败 → 高温候选救场; L3 只跑胜者一次"""
        llm = MockLLM(by_temp={0.1: WRONG, 0.4: RIGHT, 0.7: WRONG})
        conductor = self._conductor(llm)
        l2_codes = _spy(conductor.verifier, "_verify_function")
        l3_codes = _spy(conductor.verifier, "_verify_performance")

        artifact = conductor.execute(_task(), _inputs(), target_gflops=0.0)

        self.assertTrue(artifact.all_passed)
        self.assertEqual(artifact.code.strip(), RIGHT.strip(),
                         "L2 胜者没有取代 L2 失败的默认候选")
        # L2 按温度序跑过两个决赛候选: 先 WRONG (败) 后 RIGHT (胜)
        self.assertEqual([c.strip() for c in l2_codes],
                         [WRONG.strip(), RIGHT.strip()])
        # L3 只跑胜者一次 —— 「前 2 名上 L2」折中要省的就是这里
        self.assertEqual([c.strip() for c in l3_codes], [RIGHT.strip()])

    def test_all_finalists_failing_l2_keeps_default_and_fixer_takes_over(self):
        """决赛全败 → 保持默认候选交完整验证 → Conductor 的 Fixer 接手闭环

        选择不是门: 「没有候选过 L2」是 Fixer 的输入, 不是任务失败。
        这条同时覆盖 pipeline 路径的修复闭环端到端 (test_fixer.py 的
        TestConductorRepairLoop 只钉了单体路径)。
        """
        llm = MockLLM(by_temp={t: WRONG for t in (0.1, 0.4, 0.7)},
                      fixer_reply=RIGHT)
        conductor = self._conductor(llm)
        l2_codes = _spy(conductor.verifier, "_verify_function")

        artifact = conductor.execute(_task(), _inputs(), target_gflops=0.0)

        self.assertTrue(artifact.all_passed)
        self.assertEqual(artifact.code.strip(), RIGHT.strip())
        # L2 序列: 决赛两个 WRONG 各一次 (选择循环) + 默认候选完整验证再跑一次
        # (全败路径的已知小代价: 完整验证需要完整的失败证据喂 Fixer, happy path
        # 不走这里) + 修复轮的 RIGHT 一次
        self.assertEqual([c.strip() for c in l2_codes],
                         [WRONG.strip(), WRONG.strip(), WRONG.strip(),
                          RIGHT.strip()])
        # 真 Fixer 被调用过 (fixer_repair.md 的标记), 且只被调用一次
        fixer_prompts = [p for p, _ in llm.chat_calls if FIXER_MARKER in p]
        self.assertEqual(len(fixer_prompts), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
