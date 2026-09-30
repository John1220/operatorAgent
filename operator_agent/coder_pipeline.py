"""Coder Pipeline - Coder 执行流程编排 (优化版)
流程: Plan → 并行采样3 (不同 temperature) → 独立编译筛选 → 有状态修复循环(≤2轮) → 规则模板回退

单一职责: 编排代码生成流程, 不直接做 LLM 调用 (委托 Planner/采样器/Repairer/模板)
编译筛选: 默认用本地语法预检 (compile_fn 可注入真实 L1 编译器)
"""

import concurrent.futures
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from .planner import PlannerAgent
from .rule_templates import RuleTemplateCoder
from .fixer import FixerAgent
from .interface_contract import broadcast_calling_convention
from .schemas import TaskSchema, UnifiedSketch
from .prompt_manager import PromptManager


@dataclass
class RepairState:
    """有状态修复循环的状态: 记录失败候选与已尝试修复, 避免重复"""
    attempts: int = 0
    max_attempts: int = 2
    failures: list = field(default_factory=list)   # (code, issues)
    tried_fixes: list = field(default_factory=list)  # 修复说明历史

    def record_failure(self, code: str, issues: list):
        self.failures.append((code, issues))

    def should_continue(self) -> bool:
        return self.attempts < self.max_attempts


class CoderPipeline:
    """
    Coder 优化流程编排器

    特性:
    1. Plan 阶段: PlannerAgent 生成结构化实现计划
    2. Code 阶段: 基于 plan 并行采样 3 个候选 (temperature 0.1/0.4/0.7)
    3. 独立编译筛选: 逐个预检候选, 取首个通过项
    4. 有状态修复循环: 全部失败时, 携带历史最多修复 2 轮
    5. 规则模板回退: 修复仍败则用 RuleTemplateCoder 兜底
    """

    SAMPLE_TEMPERATURES = (0.1, 0.4, 0.7)
    MAX_REPAIR_ROUNDS = 2
    MAX_REF_CODE_CHARS = 2000
    GEN_MAX_TOKENS = 8192   # 治截断: 代码生成单独调高, 不被全局 max_tokens 卡住
    # P1-1 (generation-side-review): compile-gate 只作早退剪枝, 不再「首个通过即
    # 返回」。编译通过 ≠ 正确 —— 真正的选择权在 Conductor 侧的 L1+L2 (真机),
    # 这里最多放行 L2_FINALISTS 个候选去参加, 机时不随候选数全额翻倍。
    L2_FINALISTS = 2

    def __init__(
        self,
        llm,
        kernel_db,
        prompt_manager: Optional[PromptManager] = None,
        rule_coder: Optional[RuleTemplateCoder] = None,
        compile_fn: Optional[Callable] = None,
    ):
        self.llm = llm
        self.kernel_db = kernel_db
        self.pm = prompt_manager or PromptManager()
        self.planner = PlannerAgent(kernel_db, llm=llm, prompt_manager=self.pm)
        self.rule_coder = rule_coder or RuleTemplateCoder()
        # compile_fn: (code, dsl) -> (passed: bool, issues: list[str])
        # 默认 = 本地语法/结构预检; 生产可注入 Verifier L1 真实编译
        self.compile_fn = compile_fn or self._default_compile
        self._agent_role = "coder"
        self._skill_name = "triton_skill"
        # 最近一次 run() 留给 Conductor 做 L1+L2 选择的决赛候选 (P1-1)。
        # ≤ L2_FINALISTS 个; 修复/模板产物算单候选; 全败为空。
        self.last_finalists: list[str] = []

    def run(self, sketch: UnifiedSketch, task: TaskSchema, optimization_recipes=None,
            test_inputs=None, **kwargs) -> str:
        """主流程: Plan → 采样 → 剪枝 → 修复 → 回退

        返回值仍是单个字符串 (默认候选 = 决赛名单第一名); 决赛名单整体暴露在
        `self.last_finalists`, 由 Conductor 逐个跑 L1+L2 做最终选择 (P1-1)。
        """
        dsl = task.target.dsl
        recipes_ctx = optimization_recipes or ""
        # 三个采样线程共用同一份广播 (契约不随 temperature 变), 也避免每个线程各算一遍
        calling_convention = broadcast_calling_convention(test_inputs)

        # 1. Plan 阶段
        print(f"\n  [CoderPipeline] Plan 阶段...")
        plan = self.planner.run(task, sketch)
        if recipes_ctx:
            plan["_optimization_recipes"] = recipes_ctx

        # 2. 并行采样
        print(f"  [CoderPipeline] 并行采样 {len(self.SAMPLE_TEMPERATURES)} 候选 "
              f"(temps={self.SAMPLE_TEMPERATURES})...")
        candidates = self._parallel_sample(plan, sketch, task, calling_convention)

        # 3. compile-gate 早退剪枝 (P1-1): 收集通过者而不是「首个通过即返回」
        finalists = self._prune_to_finalists(candidates, dsl)
        self.last_finalists = list(finalists)
        if finalists:
            print(f"  [CoderPipeline] {len(finalists)}/{len(candidates)} 候选通过编译剪枝 "
                  f"(决赛上限 {self.L2_FINALISTS}); 最终选择在 Conductor 侧 L1+L2")
            return finalists[0]

        # 4. 有状态修复循环
        print(f"  [CoderPipeline] 全部候选未通过, 进入修复循环 (max {self.MAX_REPAIR_ROUNDS} 轮)...")
        repaired = self._repair_loop(candidates, sketch, task, dsl)
        if repaired:
            print(f"  [CoderPipeline] 修复成功 ({len(repaired)} chars)")
            self.last_finalists = [repaired]
            return repaired

        # 5. 规则模板回退
        print(f"  [CoderPipeline] 修复未通过, 回退规则模板...")
        tmpl = self.rule_coder.generate(task, sketch)
        if tmpl:
            print(f"  [CoderPipeline] 规则模板已生成 ({len(tmpl)} chars)")
            self.last_finalists = [tmpl]
            return tmpl

        # 最终兜底: 返回最优失败候选 (交由上层 Verifier 判定)
        best = self._best_failure(candidates, dsl)
        self.last_finalists = []
        if best:
            print(f"  [CoderPipeline] 无模板, 返回最优失败候选")
            return best[0]
        return ""

    # ─── 并行采样 ────────────────────────────────────────────

    def _parallel_sample(self, plan: dict, sketch: UnifiedSketch,
                         task: TaskSchema, calling_convention: str) -> list[str]:
        """并行生成 N 个候选 (不同 temperature)"""
        ref_code = self._ref_code(task, sketch)
        plan_str = json.dumps(plan, ensure_ascii=False, indent=2)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.SAMPLE_TEMPERATURES)
        ) as pool:
            futures = [
                pool.submit(self._sample_one, plan_str, sketch, task, ref_code, temp,
                            calling_convention)
                for temp in self.SAMPLE_TEMPERATURES
            ]
            results = [f.result() for f in futures]
        valid = sum(1 for c in results if c)
        print(f"  [CoderPipeline] 采样完成: {valid}/{len(results)} 非空")
        return results

    def _sample_one(self, plan_str: str, sketch: UnifiedSketch,
                    task: TaskSchema, ref_code: str, temperature: float,
                    calling_convention: str) -> str:
        """单次采样 (Code 阶段: 仅输出纯代码)"""
        hot = sketch.hot_hints()
        optimization_hints = "\n".join(f"- {h}" for h in hot) if hot else "无"

        prompt = self.pm.render(
            "coder_from_plan",
            _agent_role=self._agent_role,
            _skill_name=self._skill_name,
            plan=plan_str,
            kernel_type=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=sketch.target_dsl,
            precision=task.constraints.precision.value,
            block_m=sketch.tiling.block_m,
            block_n=sketch.tiling.block_n,
            block_k=sketch.tiling.block_k,
            ref_code=ref_code or "无",
            optimization_hints=optimization_hints,
            calling_convention=calling_convention,
        )
        raw = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self.GEN_MAX_TOKENS,
            caller_tag="CoderPipeline.Coder",
        )
        return FixerAgent._extract_code(raw)

    # ─── 编译剪枝 ────────────────────────────────────────────

    def _prune_to_finalists(self, candidates: list, dsl: str) -> list[str]:
        """compile-gate 早退剪枝 (P1-1): 编译通过只是入围, 不是入选

        旧 `_select_first_pass` 取首个通过者直接定终身 —— 编译通过 ≠ 正确,
        选择权停在最弱的验证信号上。现在只做两件事:
          1. 剪掉空候选与预检失败者 (截断/语法错不进后续任何环节);
          2. 按 temperature 序 (0.1→0.4→0.7, 低温 = 更贴 plan 的先验) 放行
             至多 L2_FINALISTS 个进入决赛, 由 Conductor 侧 L1+L2 真机定夺。
        """
        finalists = []
        for code in candidates:
            if len(finalists) >= self.L2_FINALISTS:
                break
            if not code:
                continue
            ok, _issues = self.compile_fn(code, dsl)
            if ok:
                finalists.append(code)
        return finalists

    def _best_failure(self, candidates: list, dsl: str):
        """选 issues 最少的失败候选作为修复基线; 返回 (code, issues) 或 None"""
        scored = []
        for code in candidates:
            if not code:
                continue
            ok, issues = self.compile_fn(code, dsl)
            scored.append((len(issues), code, issues))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        return scored[0][1], scored[0][2]

    # ─── 有状态修复循环 ──────────────────────────────────────

    def _repair_loop(self, candidates: list, sketch: UnifiedSketch,
                     task: TaskSchema, dsl: str) -> Optional[str]:
        state = RepairState(max_attempts=self.MAX_REPAIR_ROUNDS)
        best = self._best_failure(candidates, dsl)
        if not best:
            return None
        base_code, base_issues = best

        while state.should_continue():
            state.attempts += 1
            state.record_failure(base_code, base_issues)
            print(f"  [CoderPipeline] 修复轮 {state.attempts}/{state.max_attempts}: "
                  f"{'; '.join(base_issues[:3])}")

            fixed, note = self._repair(base_code, base_issues, state, sketch, task)
            state.tried_fixes.append(note)
            if not fixed:
                break
            ok, issues = self.compile_fn(fixed, dsl)
            if ok:
                return fixed
            base_code, base_issues = fixed, issues   # 基于修复版继续
        return None

    def _repair(self, code: str, issues: list, state: RepairState,
                sketch: UnifiedSketch, task: TaskSchema) -> tuple[str, str]:
        """基于预检失败 + 历史状态修复代码"""
        history = self._format_history(state)
        prompt = self.pm.render(
            "coder_repair",
            _agent_role=self._agent_role,
            _skill_name=self._skill_name,
            code=code,
            issues="\n".join(f"- {i}" for i in issues),
            history=history,
            kernel_type=task.source.kernel_type,
            dsl=task.target.dsl,
            precision=task.constraints.precision.value,
        )
        raw = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=self.GEN_MAX_TOKENS,
            caller_tag="CoderPipeline.Repair",
        )
        return FixerAgent._extract_code(raw), "LLM 修复"

    @staticmethod
    def _format_history(state: RepairState) -> str:
        if not state.failures:
            return "无"
        lines = []
        for i, (_, issues) in enumerate(state.failures[-2:]):
            lines.append(f"尝试{i+1}: {'; '.join(issues)}")
        return "\n".join(lines)

    # ─── 辅助 ────────────────────────────────────────────────

    def _ref_code(self, task: TaskSchema, sketch: UnifiedSketch) -> str:
        similar = self.kernel_db.search(
            algorithm=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=sketch.target_dsl,
        )
        if not similar:
            return ""
        code = similar[0].code
        if len(code) > self.MAX_REF_CODE_CHARS:
            code = code[:self.MAX_REF_CODE_CHARS] + "\n# ... (参考代码已截断)"
        return code

    @staticmethod
    def _default_compile(code: str, dsl: str) -> tuple:
        """默认编译检查 = 本地语法/结构预检 (与 CoderAgent._validate_code 等价)
        生产环境可注入 Verifier 的真实 L1 编译器"""
        from .agents import CoderAgent
        return CoderAgent._validate_code(code, dsl)
