"""Coder Pipeline - Coder 执行流程编排 (优化版)
流程: Plan → 并行采样3 (不同 temperature) → 独立编译筛选 → 有状态修复循环(≤2轮) → 规则模板回退

单一职责: 编排代码生成流程, 不直接做 LLM 调用 (委托 Planner/采样器/Repairer/模板)
编译筛选: 默认用本地语法预检 (compile_fn 可注入真实 L1 编译器)
"""

import concurrent.futures
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..generate.planner import PlannerAgent
from ..core.rule_templates import RuleTemplateCoder
from ..generate.fixer import FixerAgent
from ..generate.generation_context import (build_generation_context, generation_request,
                                 make_candidate_record)
# _extract_block_sizes 是同包 (resource_check) 的私有名, 这里直接引用是刻意的:
# BLOCK_* 解析必须与资源预检 (I75-G1) / L1 用**同一份**实现, 两份解析器迟早漂移
from ..decide.resource_check import check_resource_feasibility, _extract_block_sizes
from ..core.schemas import TaskSchema, UnifiedSketch
from ..generate.prompt_manager import PromptManager


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
        # I75-G1: compile-gate 的资源上界 (MemCaps / HardwareCaps / None)。
        # 由 Conductor 在 hw_info 到手后注入 (构造时还没有); None / 无 smem 键
        # ⇒ 资源预检跳过 (宁可漏拦, 与 L1 同一口径)。
        self.caps = None
        self._agent_role = "coder"
        self._skill_name = "triton_skill"
        # 最近一次 run() 留给 Conductor 做 L1+L2 选择的决赛候选 (P1-1)。
        # ≤ L2_FINALISTS 个; 修复/模板产物算单候选; 全败为空。
        self.last_finalists: list[str] = []
        self.generation_context = None
        self.candidate_records: list[dict] = []
        self.last_candidate_record = None
        self._sample_records: list[dict] = []
        self._finalist_records: list[dict] = []
        self._best_failure_record = None
        self._last_repair_record = None

    def _context(self, task, sketch, test_inputs=None):
        return build_generation_context(
            task, sketch, test_inputs, producer='CoderPipeline',
            strategy='parallel-sample-normalize-gate-repair-template',
            policy=dict(sample_temperatures=list(self.SAMPLE_TEMPERATURES),
                max_repair_rounds=self.MAX_REPAIR_ROUNDS,
                max_output_tokens=self.GEN_MAX_TOKENS,
                mechanical_rewrite=True, template_fallback=True,
                render_dsl=sketch.target_dsl),
            prompt_manager=self.pm, skill_name=self._skill_name,
            agent_role=self._agent_role)

    def _remember(self, record):
        # Rechecking an identical local gate does not represent a model call or
        # another proposal. Preserve its receipt once, without changing it.
        if not any(r['record_hash'] == record['record_hash'] for r in self.candidate_records):
            self.candidate_records.append(record)
        return record

    def _gate_receipt(self, code, ok, issues, record):
        if record is None:
            return None
        parent = next((r['source'] for r in self.candidate_records
                       if r['source_hash'] == record['parent_source_hash']), None)
        return self._remember(make_candidate_record(
            record['context'], code, request=record['request'],
            attempt=record['attempt'], status='prechecked' if ok else 'precheck_failed',
            source_author=record['source_author'], parent_source=parent,
            transformation=record['transformation'],
            reason=None if ok else 'compile_gate_refused'))

    def run(self, sketch: UnifiedSketch, task: TaskSchema, optimization_recipes=None,
            test_inputs=None, **kwargs) -> str:
        """主流程: Plan → 采样 → 剪枝 → 修复 → 回退

        返回值仍是单个字符串 (默认候选 = 决赛名单第一名); 决赛名单整体暴露在
        `self.last_finalists`, 由 Conductor 逐个跑 L1+L2 做最终选择 (P1-1)。
        """
        self.candidate_records = []
        self.last_candidate_record = None
        self._sample_records = []
        self._finalist_records = []
        self._best_failure_record = None
        self._last_repair_record = None
        self.generation_context = self._context(task, sketch, test_inputs)
        facts = self.generation_context['task']
        dsl = facts['dsl']
        recipes_ctx = optimization_recipes or ""
        # 三个采样线程共用同一份广播 (契约不随 temperature 变), 也避免每个线程各算一遍
        calling_convention = self.generation_context['interface']['broadcast']

        # 1. Plan 阶段
        print(f"\n  [CoderPipeline] Plan 阶段...")
        plan = self.planner.run(task, sketch)
        if recipes_ctx:
            plan["_optimization_recipes"] = recipes_ctx

        # 2. 并行采样
        print(f"  [CoderPipeline] 并行采样 {len(self.SAMPLE_TEMPERATURES)} 候选 "
              f"(temps={self.SAMPLE_TEMPERATURES})...")
        candidates = self._parallel_sample(plan, sketch, task, calling_convention)

        # 3. 确定性机械改写 (I75 后续): prompt 层约束已被真机证明对该模型无效
        #    (无视 sketch 写 256 大 tile / fp32 不带 input_precision), 生成后
        #    机械对齐兜底 —— 不再指望模型「自觉」
        normalized, normalized_records = [], []
        for index, code in enumerate(candidates):
            record = self._sample_records[index] if index < len(self._sample_records) else None
            code, record = self._normalize_recorded(code, sketch, task, record)
            normalized.append(code)
            normalized_records.append(record)
        candidates, self._sample_records = normalized, normalized_records

        # 4. compile-gate 早退剪枝 (P1-1): 收集通过者而不是「首个通过即返回」
        finalists = self._prune_to_finalists(candidates, dsl, task.source.kernel_type)
        self.last_finalists = list(finalists)
        if finalists:
            print(f"  [CoderPipeline] {len(finalists)}/{len(candidates)} 候选通过编译剪枝 "
                  f"(决赛上限 {self.L2_FINALISTS}); 最终选择在 Conductor 侧 L1+L2")
            self.last_candidate_record = self._finalist_records[0]
            return finalists[0]

        # 5. 有状态修复循环
        print(f"  [CoderPipeline] 全部候选未通过, 进入修复循环 (max {self.MAX_REPAIR_ROUNDS} 轮)...")
        repaired = self._repair_loop(candidates, sketch, task, dsl)
        if repaired:
            print(f"  [CoderPipeline] 修复成功 ({len(repaired)} chars)")
            self.last_finalists = [repaired]
            return repaired

        # 6. 规则模板回退
        print(f"  [CoderPipeline] 修复未通过, 回退规则模板...")
        tmpl = self.rule_coder.generate(task, sketch)
        if tmpl:
            print(f"  [CoderPipeline] 规则模板已生成 ({len(tmpl)} chars)")
            self.last_finalists = [tmpl]
            self.last_candidate_record = self._remember(make_candidate_record(
                self.generation_context, tmpl, source_author='rule_template'))
            return tmpl

        # 最终兜底: 返回最优失败候选 (交由上层 Verifier 判定)
        best = self._best_failure(candidates, dsl)
        self.last_finalists = []
        if best:
            print(f"  [CoderPipeline] 无模板, 返回最优失败候选")
            self.last_candidate_record = self._best_failure_record
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
            context = self.generation_context or self._context(task, sketch)
            futures = [
                pool.submit(self._sample_one, plan_str, sketch, task, ref_code, temp,
                            calling_convention, _record_context=context,
                            _attempt=index, _with_record=True)
                for index, temp in enumerate(self.SAMPLE_TEMPERATURES, 1)
            ]
            outcomes = [f.result() for f in futures]
        # Workers never mutate shared receipt state. Merge in the original
        # temperature order, even when responses finish in another order.
        self._sample_records = [item[1] for item in outcomes]
        self.candidate_records.extend(self._sample_records)
        error = next((item[2] for item in outcomes if item[2] is not None), None)
        if error is not None:
            raise error
        results = [item[0] for item in outcomes]
        valid = sum(1 for c in results if c)
        print(f"  [CoderPipeline] 采样完成: {valid}/{len(results)} 非空")
        return results

    def _sample_one(self, plan_str: str, sketch: UnifiedSketch,
                    task: TaskSchema, ref_code: str, temperature: float,
                    calling_convention: str, *, _record_context=None,
                    _attempt=1, _with_record=False) -> str:
        """单次采样 (Code 阶段: 仅输出纯代码)"""
        hot = sketch.hot_hints()
        optimization_hints = "\n".join(f"- {h}" for h in hot) if hot else "无"

        context = _record_context or self._context(task, sketch)
        facts = context['task']
        prompt = self.pm.render(
            "coder_from_plan",
            _agent_role=self._agent_role,
            _skill_name=self._skill_name,
            plan=plan_str,
            kernel_type=facts['kernel_type'],
            hardware=facts['hardware'],
            dsl=sketch.target_dsl,
            precision=facts['precision'],
            block_m=sketch.tiling.block_m,
            block_n=sketch.tiling.block_n,
            block_k=sketch.tiling.block_k,
            ref_code=ref_code or "无",
            optimization_hints=optimization_hints,
            calling_convention=calling_convention,
        )
        messages = [{"role": "user", "content": prompt}]
        request = generation_request(self.llm, messages, caller='CoderPipeline.Coder',
                                     temperature=temperature, max_tokens=self.GEN_MAX_TOKENS)
        failure_status = 'model_failed'
        try:
            raw = self.llm.chat(
                messages,
                temperature=temperature,
                max_tokens=self.GEN_MAX_TOKENS,
                caller_tag="CoderPipeline.Coder",
            )
            failure_status = 'no_candidate'
            code = FixerAgent._extract_code(raw)
            record = make_candidate_record(context, code, request=request, attempt=_attempt,
                                           status='proposed' if code else 'no_candidate')
        except Exception as error:
            record = make_candidate_record(context, None, request=request, attempt=_attempt,
                                           status=failure_status, reason=type(error).__name__)
            if _with_record:
                return '', record, error
            self._remember(record)
            raise
        if _with_record:
            return code, record, None
        self._remember(record)
        return code

    # ─── 确定性机械改写 (I75 后续) ────────────────────────────

    def normalize_candidate(self, code: str, sketch: UnifiedSketch,
                            task: TaskSchema) -> tuple[str, list[str]]:
        """生成后确定性改写: 违反 sketch 的 BLOCK_* 常数改回 sketch 值;
        fp32 的 tl.dot 钉住 input_precision="ieee"。

        为什么不再靠 prompt (CHECKLOG I75 真机证据): LLM 无视 prompt 硬约束
        坚持 BLOCK_M=BLOCK_N=256 (sketch 说 64×64×32, smem 384KiB > H20 的
        228KiB), 且 fp32 的 tl.dot 不带 input_precision 走 TF32 (差 ~2.98e-3,
        容差 1e-4 必挂 L2)。prompt 层约束对该模型无效 ⇒ 唯一可靠的位置是
        生成之后的机械文本改写。

        返回 (code, notes); 无可改写内容时**原样返回原字符串对象**。
        """
        # 非 triton / 空代码: 没有可改写的对象, 原样返回
        if not code or task.target.dsl != "triton":
            return code, []
        notes: list[str] = []
        new_code = code

        # ── 1) tiling 改写: BLOCK_M/N/K 与 sketch 不符 → 改回 sketch 值 ──
        # sketch 缺某维 (getattr 不到 / 非 int) 时该维**不改写** —— 只改有依据的;
        # BLOCK_M/N/K 之外的 BLOCK_* (如 BLOCK_SIZE) 没有 sketch 对应物, 同样跳过
        tiling = getattr(sketch, "tiling", None)
        targets: dict[str, int] = {}
        for blk, attr in (("BLOCK_M", "block_m"), ("BLOCK_N", "block_n"),
                          ("BLOCK_K", "block_k")):
            v = getattr(tiling, attr, None)
            if isinstance(v, int) and not isinstance(v, bool):
                targets[blk] = v
        for name, old in _extract_block_sizes(new_code):
            target = targets.get(name)
            if target is None or old == target:
                continue
            # 精确 token 替换 (该「名字=旧值」对的全部出现): 覆盖解析器认识的
            # 两种书写 —— `BLOCK_M=256` (含 `: tl.constexpr` / 空格变体) 与
            # autotune dict 的 `'BLOCK_M': 256`; (?!\d) 防止 256 咬中 2560 的前缀
            pat = (rf"(\b{name}\s*(?::\s*tl\.constexpr\s*)?=\s*){old}(?!\d)"
                   rf"|(['\"]{name}['\"]\s*:\s*){old}(?!\d)")
            rewritten = re.sub(
                pat, lambda m: (m.group(1) or m.group(2)) + str(target), new_code)
            if rewritten != new_code:   # 确实改到了才记 note (没改到 = 无依据可改)
                new_code = rewritten
                notes.append(f"{name} {old}→{target} (sketch)")

        # ── 2) ieee 钉住: fp32 下 tl.dot 默认走 TF32, 精度必挂 L2 (I75 实测) ──
        if (task.constraints.precision.value == "fp32"
                and "tl.dot(" in new_code
                and "input_precision" not in new_code):
            pinned = 0

            def _pin(m: "re.Match") -> str:
                nonlocal pinned
                args = m.group(1)
                if not args.strip():   # `tl.dot()` 空参不追加
                    return m.group(0)
                pinned += 1
                return f'tl.dot({args}, input_precision="ieee")'

            # 单行正则: 参数内再出现括号 (嵌套) 即匹配不到 —— 匹配不到的交给人工
            pinned_code = re.sub(r"tl\.dot\(([^()]*)\)", _pin, new_code)
            if pinned:
                new_code = pinned_code
                notes.append(f'tl.dot 钉住 input_precision="ieee" ({pinned} 处)')
            else:
                # 代码里有 tl.dot( 却一处都匹配不上 (嵌套括号/跨行) —— 跳过并留痕
                notes.append("tl.dot 跨行/嵌套, ieee 未钉住(人工注意)")

        return new_code, notes

    def _normalize_and_report(self, code: str, sketch: UnifiedSketch,
                              task: TaskSchema) -> str:
        """normalize_candidate 的接线封装: 改写发生时打一行日志供人工复核"""
        return self._normalize_recorded(code, sketch, task)[0]

    def _normalize_recorded(self, code, sketch, task, record=None):
        original = code
        code, notes = self.normalize_candidate(code, sketch, task)
        if notes:
            print(f"  [CoderPipeline] 机械改写: {', '.join(notes)}")
        if code != original and self.generation_context is not None:
            record = self._remember(make_candidate_record(
                self.generation_context, code,
                attempt=record['attempt'] if record is not None else 1,
                source_author='mechanical_transform', parent_source=original,
                transformation=notes))
        return code, record

    # ─── 编译剪枝 ────────────────────────────────────────────

    def _gate_check(self, code: str, dsl: str, kernel_type: str = "") -> tuple:
        """compile-gate 的两道便宜门收在一处 (I75-G1): 语法 + **资源预检**。

        与 L1 共用 check_resource_feasibility。逻辑 tile、dtype 与 num_stages 不证明
        共享存储；无明确资源证据时交后续原生检查。unknown 不改变语法门的实际结果。
        """
        ok, issues = self.compile_fn(code, dsl)
        issues = list(issues or [])
        if ok and code:
            rep = check_resource_feasibility(code, self.caps, kernel_type)
            if not rep.feasible:
                ok = False
                issues.append(rep.as_message())
        return ok, issues

    def _prune_to_finalists(self, candidates: list, dsl: str,
                            kernel_type: str = "") -> list[str]:
        """compile-gate 早退剪枝 (P1-1): 编译通过只是入围, 不是入选

        旧 `_select_first_pass` 取首个通过者直接定终身 —— 编译通过 ≠ 正确,
        选择权停在最弱的验证信号上。现在只做两件事:
          1. 剪掉空候选与预检失败者 (截断/语法错/**tile 放不下**不进后续任何环节);
          2. 按 temperature 序 (0.1→0.4→0.7, 低温 = 更贴 plan 的先验) 放行
             至多 L2_FINALISTS 个进入决赛, 由 Conductor 侧 L1+L2 真机定夺。
        """
        finalists = []
        self._finalist_records = []
        for index, code in enumerate(candidates):
            if len(finalists) >= self.L2_FINALISTS:
                break
            if not code:
                continue
            ok, issues = self._gate_check(code, dsl, kernel_type)
            record = self._sample_records[index] if index < len(self._sample_records) else None
            receipt = self._gate_receipt(code, ok, issues, record)
            if ok:
                finalists.append(code)
                self._finalist_records.append(receipt)
            elif any("资源预检" in i for i in issues):
                # 单独说一句: 这类失败是「本地就能判死」的, 之前它烧的是真机往返
                print(f"  [CoderPipeline] 候选被资源预检剪掉 (未动 SSH): {issues[-1][:110]}")
        return finalists

    def _best_failure(self, candidates: list, dsl: str, kernel_type: str = ""):
        """选 issues 最少的失败候选作为修复基线; 返回 (code, issues) 或 None"""
        scored = []
        self._best_failure_record = None
        for index, code in enumerate(candidates):
            if not code:
                continue
            ok, issues = self._gate_check(code, dsl, kernel_type)
            record = self._sample_records[index] if index < len(self._sample_records) else None
            receipt = self._gate_receipt(code, ok, issues, record)
            scored.append((len(issues), code, issues, receipt))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        self._best_failure_record = scored[0][3]
        return scored[0][1], scored[0][2]

    # ─── 有状态修复循环 ──────────────────────────────────────

    def _repair_loop(self, candidates: list, sketch: UnifiedSketch,
                     task: TaskSchema, dsl: str) -> Optional[str]:
        state = RepairState(max_attempts=self.MAX_REPAIR_ROUNDS)
        best = self._best_failure(candidates, dsl, task.source.kernel_type)
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
            # 修复产物同样过机械改写 (I75 后续): 修复的 LLM 与生成的 LLM 是同一个,
            # 一样会把 256 大 tile 写回来 —— 修完再机械对齐一次 sketch
            fixed, record = self._normalize_recorded(fixed, sketch, task, self._last_repair_record)
            # 复检也过资源门 (I75-G1): 修完又写回大 tile 的产物不许再烧 SSH
            ok, issues = self._gate_check(fixed, dsl, task.source.kernel_type)
            receipt = self._gate_receipt(fixed, ok, issues, record)
            if ok:
                self.last_candidate_record = receipt
                return fixed
            base_code, base_issues = fixed, issues   # 基于修复版继续
        return None

    def _repair(self, code: str, issues: list, state: RepairState,
                sketch: UnifiedSketch, task: TaskSchema) -> tuple[str, str]:
        """基于预检失败 + 历史状态修复代码"""
        history = self._format_history(state)
        context = self.generation_context or self._context(task, sketch)
        facts = context['task']
        prompt = self.pm.render(
            "coder_repair",
            _agent_role=self._agent_role,
            _skill_name=self._skill_name,
            code=code,
            issues="\n".join(f"- {i}" for i in issues),
            history=history,
            kernel_type=facts['kernel_type'],
            dsl=facts['dsl'],
            precision=facts['precision'],
        )
        messages = [{"role": "user", "content": prompt}]
        request = generation_request(self.llm, messages, caller='CoderPipeline.Repair',
                                     temperature=0.1, max_tokens=self.GEN_MAX_TOKENS)
        failure_status = 'model_failed'
        try:
            raw = self.llm.chat(
                messages,
                temperature=0.1,
                max_tokens=self.GEN_MAX_TOKENS,
                caller_tag="CoderPipeline.Repair",
            )
            failure_status = 'no_candidate'
            fixed = FixerAgent._extract_code(raw)
            record = make_candidate_record(context, fixed, request=request, attempt=state.attempts,
                status='proposed' if fixed else 'no_candidate', parent_source=code)
        except Exception as error:
            self._last_repair_record = self._remember(make_candidate_record(
                context, None, request=request, attempt=state.attempts,
                status=failure_status, parent_source=code, reason=type(error).__name__))
            raise
        self._last_repair_record = self._remember(record)
        return fixed, "LLM 修复"

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
        from ..generate.agents import CoderAgent
        return CoderAgent._validate_code(code, dsl)
