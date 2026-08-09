"""Planner Agent - Plan 阶段
单一职责: 根据 TaskSchema + Sketch + RAG，输出结构化实现计划 (JSON)
不生成代码，只规划算法/签名/步骤/风险点，供 Coder 阶段消费"""

import json
from typing import Optional
from .schemas import TaskSchema, UnifiedSketch
from .kernel_db import KernelDatabase
from .prompt_manager import PromptManager


class PlannerAgent:
    """实现规划器 - 输出算法选择、kernel/wrapper 签名、实现步骤、风险点"""

    def __init__(self, kernel_db: KernelDatabase, llm=None,
                 prompt_manager: Optional[PromptManager] = None):
        self.kernel_db = kernel_db
        self.llm = llm
        self.pm = prompt_manager or PromptManager()
        self.name = "Planner"
        self._agent_role = "designer"   # 复用 designer 的 SKILL 切片
        self._skill_name = "triton_skill"

    def run(self, task: TaskSchema, sketch: UnifiedSketch) -> dict:
        """输出实现计划 dict; 无 LLM 时走最小回退计划"""
        if self.llm is None:
            return self._fallback_plan(task, sketch)

        similar = self.kernel_db.search(
            algorithm=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=task.target.dsl,
        )
        rag_context = self._build_rag_context(similar)

        prompt = self.pm.render(
            "planner_plan",
            _agent_role=self._agent_role,
            _skill_name=self._skill_name,
            kernel_type=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=task.target.dsl,
            precision=task.constraints.precision.value,
            sketch=json.dumps(sketch.to_dict(), ensure_ascii=False, indent=2),
            rag_context=rag_context or "无",
        )
        result = self.llm.chat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=2048,
            caller_tag="Planner",
        )
        # 补全缺失字段
        result.setdefault("algorithm", sketch.algorithm)
        result.setdefault("kernel_signature", "def kernel(A, B)")
        result.setdefault("jit_signature", "")
        result.setdefault("steps", [])
        result.setdefault("tiling", {
            "block_m": sketch.tiling.block_m,
            "block_n": sketch.tiling.block_n,
            "block_k": sketch.tiling.block_k,
        })
        result.setdefault("risks", [])
        result.setdefault("notes", "")
        print(f"  [Planner] 计划: algorithm={result['algorithm']}, "
              f"steps={len(result['steps'])}, risks={len(result['risks'])}")
        return result

    @staticmethod
    def _build_rag_context(similar: list) -> str:
        if not similar:
            return ""
        lines = []
        for i, r in enumerate(similar[:3]):
            lines.append(f"案例{i+1}: {r.algorithm}, perf={r.performance} GFLOPS, tags={r.tags}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_plan(task: TaskSchema, sketch: UnifiedSketch) -> dict:
        """无 LLM 时的最小计划 (从 sketch 推断)"""
        kt = task.source.kernel_type
        is_triton = task.target.dsl == "triton"
        return {
            "algorithm": sketch.algorithm,
            "kernel_signature": f"def kernel(A, B)" if "matmul" in kt else "def kernel(X)",
            "jit_signature": (f"@triton.jit def {kt}_kernel(ptrs..., M, N, K, strides..., "
                              "BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr)")
                             if is_triton else "",
            "steps": ["分配输出", "计算 grid", "启动 kernel", "返回输出"],
            "tiling": {"block_m": sketch.tiling.block_m,
                       "block_n": sketch.tiling.block_n,
                       "block_k": sketch.tiling.block_k},
            "risks": ["边界处理", "stride 非连续"],
            "notes": "无 LLM 回退计划",
        }
