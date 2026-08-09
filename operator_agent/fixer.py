"""Fixer Agent 模块
职责单一: 根据 L1(编译)/L2(功能) 验证失败信息修复 kernel 代码

设计要点:
- 输入: 失败代码 + TaskSchema + 验证失败结果列表 (级别/消息/错误日志)
- 输出: (修复后的代码, 修复说明, Reflexion 结构化教训)
- 错误日志经 _truncate_error 截断压缩，避免撑爆 LLM 上下文
- 代码提取 (_extract_code) 与 Coder 的生成清理逻辑解耦，
  同时兼容 Triton 与纯 Python kernel，不做任何语义改写
- Reflexion: 修复成功后调用 produce_reflexion() 产出 {root_cause, lessons,
  error_keywords, applicable_hardware} 结构化教训供 FailureDatabase 沉淀
"""

import re
import textwrap
from typing import Optional
from .schemas import TaskSchema, VerificationResult
from .prompt_manager import PromptManager
from .kernel_db import FailureRecord  # 用于回写 Reflexion 教训


class FixerAgent:
    """
    Fixer Agent (错误修复)
    职责: 当 L1/L2 验证失败时，分析错误原因并生成修复后的代码
    基于 LLM 驱动修复
    """

    MAX_ERROR_CHARS = 1500  # 单条错误日志注入 prompt 的最大长度
    MAX_FIX_ATTEMPTS = 3    # 修复代码的预检重试次数

    def __init__(self, llm=None, prompt_manager: Optional[PromptManager] = None,
                 failure_db=None):
        self.llm = llm
        self.pm = prompt_manager or PromptManager()
        self.failure_db = failure_db  # 失败案例库 (可选): 检索历史修复经验
        self.name = "Fixer"
        # 修复代码时必须遵守与 Coder 相同的 SKILL 禁令/必需模式
        self._agent_role = "coder"
        self._skill_name = "triton_skill"

    def run(
        self,
        code: str,
        task: TaskSchema,
        failed_results: list[VerificationResult],
    ) -> tuple[str, str]:
        """
        修复失败代码 (带预检+重试闭环，和 CoderAgent._llm_generate_code 一致)

        关键防御:
        - 当 task.target.dsl != "triton" (昆仑芯/昇腾等) 时, 不注入 triton_skill
          否则 LLM 看到 SKILL 里的 tl.* 指令会继续输出 Triton -> 编译无限失败
        - 修复生成后走与 Coder 同一套 _validate_code 预检 (含"非 triton 禁止 Triton API")
          有问题就带反馈最多重试 MAX_FIX_ATTEMPTS 次
        """
        if self.llm is None:
            raise RuntimeError("LLM 未配置，无法执行修复")

        self._last_task = task
        self._last_failed_results = failed_results
        self._last_original_code = code

        error_details = self._format_failures(failed_results)
        # Reflexion 增强: 先按 kernel_type/dsl 上下文取, 再用 error_summary 相似度补充
        history_cases = self._build_history_cases(task, failed_results)

        # ── SKILL 选择策略 ──
        effective_skill = (
            self._skill_name if (task.target.dsl or "").lower() == "triton" else None
        )
        dsl = task.target.dsl

        base_prompt = self.pm.render(
            "fixer_repair",
            _agent_role=self._agent_role,
            _skill_name=effective_skill,
            kernel_type=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=dsl,
            precision=task.constraints.precision.value,
            error_details=error_details,
            history_cases=history_cases or "无",
            code=code,
        )

        levels = ", ".join(r.level.value for r in failed_results)
        feedback = ""
        fixed = ""
        succeeded = False
        for attempt in range(1, self.MAX_FIX_ATTEMPTS + 1):
            raw = self.llm.chat(
                [{"role": "user", "content": base_prompt + feedback}],
                temperature=0.1,
                max_tokens=4096,
                caller_tag="Fixer",
            )
            fixed = self._extract_code(raw)
            if not fixed:
                issues = ["Fixer 未能从 LLM 回复中提取有效代码"]
            else:
                _, issues = self._validate_code(fixed, dsl)
            if not issues:
                succeeded = True
                if attempt > 1:
                    print(f"  [Fixer] 第 {attempt} 次修复通过预检")
                break
            print(f"  [Fixer] 第 {attempt} 次修复预检未通过: {'; '.join(issues)}")
            bullets = "\n".join(f"- {i}" for i in issues)
            feedback = (
                f"\n\n## 上次修复未通过预检 (尝试 {attempt}/{self.MAX_FIX_ATTEMPTS})\n{bullets}\n"
                f"请重新输出完整修复代码: 以 `import` 开头，包含 kernel 入口函数，"
                f"括号配对完整，不要输出解释；特别注意当前 dsl={dsl!r} 的 API 禁令。"
            )
        if not fixed:
            raise ValueError("Fixer 未能从 LLM 回复中提取有效代码")

        summary = f"基于 {len(failed_results)} 个失败结果修复 ({levels})"
        print(f"  [Fixer] LLM 修复完成: {len(fixed)} chars")
        # 仅当修复通过预检时才留下可 Reflexion 的上下文 (避免"假修复"入库)
        if succeeded:
            self._last_fixed_code = fixed
            self._last_fix_summary = summary
        return fixed, summary

    def produce_reflexion(self) -> Optional[FailureRecord]:
        """Reflexion: 修复成功后, 让 LLM 生成结构化自我反思并封装为 FailureRecord。

        返回 FailureRecord (fixed=True, 带 lessons/error_keywords/root_cause)
        供上游 Conductor 写入 FailureDatabase 沉淀经验。
        若无 last_* 上下文或 LLM 未配置则返回 None。
        """
        if not hasattr(self, '_last_fixed_code') or not self._last_fixed_code:
            return None
        if self.llm is None:
            return None

        task = getattr(self, '_last_task', None)
        failed = getattr(self, '_last_failed_results', [])
        original = getattr(self, '_last_original_code', '')
        if task is None:
            return None

        error_details = self._format_failures(failed)
        levels = "+".join(sorted({r.level.value for r in failed})) or "L?"

        prompt = self.pm.render(
            "fixer_reflexion",
            _agent_role="coder",
            _skill_name=None,  # 反思不注入 skill
            kernel_type=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=task.target.dsl,
            error_details=error_details,
            fix_summary=self._last_fix_summary,
        )

        try:
            result = self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
                caller_tag="Fixer.Reflexion",
            )
        except Exception as e:
            print(f"  [Fixer] Reflexion LLM 调用失败, 跳过沉淀: {e}")
            return None

        # 构造 FailureRecord (固定 failure_id, 避免重复写入)
        import uuid, hashlib, time
        sig = hashlib.sha1(
            f"{task.source.kernel_type}|{task.target.dsl}|{levels}|{error_details[:200]}".encode()
        ).hexdigest()[:12]

        lessons = result.get("lessons") or []
        if isinstance(lessons, str):
            lessons = [lessons.strip()]
        lessons = [str(x).strip() for x in lessons if str(x).strip()]

        kw = result.get("error_keywords") or []
        if isinstance(kw, str):
            kw = [k.strip() for k in re.split(r"[,;，；]", kw) if k.strip()]
        kw = [str(x).strip() for x in kw if str(x).strip()]

        hw = result.get("applicable_hardware") or [task.target.hardware]
        if isinstance(hw, str):
            hw = [x.strip() for x in re.split(r"[,;，；]", hw) if x.strip()]
        hw = [str(x).strip() for x in hw if str(x).strip()]

        return FailureRecord(
            failure_id=f"fix-{sig}-{uuid.uuid4().hex[:4]}",
            kernel_type=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=task.target.dsl,
            level=levels,
            error_summary=error_details[:500],
            fix_summary=self._last_fix_summary,
            fixed=True,
            created_at=time.time(),
            root_cause=str(result.get("root_cause", "")).strip()[:300],
            lessons=lessons[:10],
            error_keywords=kw[:15],
            applicable_hardware=hw[:10],
        )

    @staticmethod
    def _validate_code(code: str, dsl: str = "python") -> tuple:
        """
        修复后预检 —— 与 CoderAgent._validate_code 保持完全一致的规则集,
        确保修复后的代码至少能过静态关卡 (截断/语法/入口/跨硬件 API 禁令)
        """
        if not code or not code.strip():
            return False, ["代码为空"]
        issues = []

        for open_ch, close_ch in (("(", ")"), ("[", "]"), ("{", "}")):
            if code.count(open_ch) != code.count(close_ch):
                issues.append(f"括号 {open_ch}{close_ch} 不配对 (疑似输出截断)")
                break

        stripped = code.rstrip()
        if stripped.endswith(("(", ",", ":", "\\", "+", "-", "*", "/", "=")):
            issues.append("代码结尾悬空 (疑似输出截断)")

        if "def kernel" not in code:
            issues.append("缺少 kernel 入口函数")

        # 跨硬件 API 禁令: dsl != "triton" 时禁止 Triton (昆仑芯/昇腾等新硬件核心防线)
        if (dsl or "").lower() != "triton":
            if "import triton" in code or "from triton" in code or "@triton.jit" in code:
                issues.append(
                    f"禁止: dsl={dsl!r} 但代码里出现了 Triton API (import triton / @triton.jit 等)。"
                    " 昆仑芯/昇腾/其他非 Nvidia 硬件必须使用纯 Python/PyTorch/NumPy，不能用 Triton/CUDA。"
                )

        if (dsl or "").lower() == "triton" and "@triton.jit" not in code:
            issues.append("缺少 @triton.jit 装饰器")

        dsl_lower = (dsl or "").lower()
        if dsl_lower == "cuda_c":
            # C++ Extension 预检: 必须含 load_inline 和 C++ 源码
            if "load_inline" not in code:
                issues.append("cuda_c 代码缺少 load_inline 调用 (PyTorch C++ Extension JIT 编译)")
            if "_CPP_SOURCE" not in code and "cpp_sources" not in code:
                issues.append("cuda_c 代码缺少 C++ 源码 (_CPP_SOURCE 或 cpp_sources)")
        elif dsl_lower in ("python", "triton"):
            try:
                compile(code, "<generated>", "exec")
            except SyntaxError as e:
                issues.append(f"语法错误: {e.msg} (第 {e.lineno} 行)")

        return (not issues, issues)

    def _build_history_cases(self, task: TaskSchema,
                             failed_results: Optional[list[VerificationResult]] = None) -> str:
        """从失败案例库检索相似历史案例 (已修复案例优先)，注入 prompt。

        Reflexion 增强:
          1. search_by_context(kernel_type/dsl/hardware) 取基础同类案例
          2. 若有 failed_results, search_by_error(error_text) 再取 "错误词元相似" 案例
          3. 去重合并后, 每条案例输出 error_summary + fix_summary + Reflexion lessons
        """
        if not self.failure_db:
            return ""
        base = self.failure_db.search_by_context(
            kernel_type=task.source.kernel_type,
            dsl=task.target.dsl,
            hardware=task.target.hardware,
            limit=2,
        )
        error_matches: list = []
        if failed_results:
            err_text = "\n".join(r.error_log or r.message for r in failed_results)
            error_matches = self.failure_db.search_by_error(
                err_text,
                kernel_type=task.source.kernel_type,
                dsl=task.target.dsl,
                limit=2,
            )
        # 合并去重 (按 failure_id)
        seen = set()
        merged: list = []
        for r in base + error_matches:
            if r.failure_id in seen:
                continue
            seen.add(r.failure_id)
            merged.append(r)
        if not merged:
            return ""
        lines = []
        for i, c in enumerate(merged):
            tag = ("[错误相似度命中]" if c in error_matches else "[上下文命中]")
            lines.append(f"历史案例{i + 1} {tag}: [{c.level}] {c.error_summary}")
            if c.root_cause:
                lines.append(f"  根因: {c.root_cause}")
            if c.fix_summary:
                lines.append(f"  修复方式: {c.fix_summary}")
            if c.lessons:
                lines.append("  可复用教训:")
                for l in c.lessons[:5]:
                    lines.append(f"    - {l}")
            if c.applicable_hardware:
                lines.append(f"  适用硬件: {', '.join(c.applicable_hardware)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 错误信息组装与压缩
    # ------------------------------------------------------------------

    def _format_failures(self, failed_results: list[VerificationResult]) -> str:
        """将失败结果格式化为注入 prompt 的错误描述 (日志已截断)"""
        if not failed_results:
            return "无详细错误信息"
        parts = []
        for r in failed_results:
            part = f"[{r.level.value}] {r.message}"
            if r.error_log:
                part += f"\n错误日志:\n{self._truncate_error(r.error_log)}"
            parts.append(part)
        return "\n\n".join(parts)

    @classmethod
    def _truncate_error(cls, log: str, max_chars: Optional[int] = None) -> str:
        """
        截断过长的错误日志: 保留头部 (首个 error 上下文) 与尾部 (最终报错行)，
        中间以省略标记连接，避免长编译错误栈撑爆上下文
        """
        if not log:
            return ""
        limit = max_chars or cls.MAX_ERROR_CHARS
        log = log.strip()
        if len(log) <= limit:
            return log
        head = limit // 2
        tail = limit - head
        omitted = len(log) - limit
        return f"{log[:head]}\n... [省略 {omitted} 字符] ...\n{log[-tail:]}"

    # ------------------------------------------------------------------
    # 代码提取 (语言无关，不做语义改写)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code(raw: str) -> str:
        """
        从 LLM 回复中提取代码:
        1. 剥离 thinking 标签
        2. 优先提取 ``` 代码块
        3. 无代码块时定位代码起始行 (import/from/@triton.jit/def/#include)
        返回空串表示提取失败 (由调用方决定重试或终止)
        """
        code = raw.strip()

        for end_tag in ("</think>", "</thinking>"):
            idx = code.find(end_tag)
            if idx != -1:
                code = code[idx + len(end_tag):].strip()
                break

        m = re.search(
            r"```(?:python|triton|cpp|cuda|c\+\+)?\s*\n(.*?)```",
            code,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            # 去除代码块整体缩进 (LLM 常把 fence 放在列表/引用内导致行首带缩进)
            return textwrap.dedent(m.group(1)).strip()

        lines = code.split("\n")
        start = None
        for i, line in enumerate(lines):
            if re.match(r"^(import |from |@triton\.jit|def |#include)", line.strip()):
                start = i
                break
        if start is None:
            return ""

        body = "\n".join(lines[start:])
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        return textwrap.dedent(body).strip()

    def __repr__(self) -> str:
        return f"FixerAgent(llm={self.llm is not None})"
