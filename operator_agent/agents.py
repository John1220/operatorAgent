"""Agent 算子生成模块
Designer: LLM 驱动设计 UnifiedSketch
Coder:   LLM 驱动生成目标代码
Search:   LLM 驱动性能优化建议
prompt 模板均从 prompts/ 目录加载，支持热更新"""

import json
from typing import Optional
from .candidates import TemplateCandidate, prune_by_smem
from .resource_check import caps_from_metrics
from .schemas import (
    TaskSchema, UnifiedSketch, TilingStrategy, MemoryPlan,
    PipelineStage, TaskComplexity,
)
from .kernel_db import KernelDatabase
from .prompt_manager import PromptManager
from .fixer import FixerAgent
from .interface_contract import broadcast_calling_convention
from .generation_context import (build_generation_context, generation_request,
                                 make_candidate_record)
from .triton_opt_rules import TritonOptimizationRules


def _clean(hints) -> list:
    """只留字符串: new_hints 来自 LLM 的 JSON, 一个 dict 混进来就会以 Python repr
    的形式印进 prompt"""
    return [h for h in (hints or []) if isinstance(h, str) and h.strip()]


def _hint_log(sketch: UnifiedSketch, new_hints: list) -> list:
    """冷层累计日志: 去重、保序、不随轮数重复

    规则回退路径每轮生成**同文案**的 hint, 原先是 `旧 + 新` 直接拼, 于是 6 轮后
    26 条里绝大部分是同一句 (实测每条在 Coder prompt 里出现 20 次)。去重只作用在
    冷层; 热层 (hints_this_version) 永远是本轮原样。
    """
    seen, out = set(), []
    for h in _clean(sketch.optimization_hints) + _clean(new_hints):
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


class DesignerAgent:
    """
    Designer Agent
    职责: 根据算子 Spec + 硬件信息 + RAG 案例，生成 Unified Sketch
    基于 LLM 驱动生成
    """

    def __init__(self, kernel_db: KernelDatabase, llm=None, prompt_manager: Optional[PromptManager] = None):
        self.kernel_db = kernel_db
        self.llm = llm
        self.pm = prompt_manager or PromptManager()
        self.name = "Designer"
        self._agent_role = "designer"
        self._skill_name = "triton_skill"

    MAX_SKETCH_ATTEMPTS = 3   # Sketch 生成 + 最多 2 次带 schema 校验反馈重试

    def run(self, task: TaskSchema, hardware_info: dict) -> UnifiedSketch:
        if self.llm is None:
            raise RuntimeError("LLM 未配置，规则模板已移除")

        similar = self.kernel_db.search(
            algorithm=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=task.target.dsl,
        )

        feedback = ""
        sketch = None
        for attempt in range(1, self.MAX_SKETCH_ATTEMPTS + 1):
            sketch = self._llm_generate_sketch(task, hardware_info, similar, feedback)
            issues = sketch.validate()
            if not issues:
                if attempt > 1:
                    print(f"  [Designer] 第 {attempt} 次尝试 Sketch 通过 schema 校验")
                break
            # 仅输出非 warning 项作为真正失败依据, warnings 单独 print
            hard_errors = [e for e in issues if not e.startswith("[WARNING]")]
            for w in [e for e in issues if e.startswith("[WARNING]")]:
                print(f"  [Designer] {w}")
            if not hard_errors:
                # 全部是 warnings, 允许放行 (允许探索新算子/新硬件)
                break
            print(f"  [Designer] 第 {attempt} 次 Sketch schema 校验失败: {'; '.join(hard_errors[:3])}")
            feedback = (
                "\n\n[上一轮 Designer 输出未通过本地 schema 校验, 请按以下修正后重新生成 JSON:\n"
                + "\n".join(f"  - {e}" for e in hard_errors)
                + "\n  仍返回严格同结构的 JSON (algorithm/tiling/memory_plan/pipeline/target_dsl/optimization_hints), 不要输出解释或代码.]\n"
            )

        print(f"  [Designer] LLM 生成 Sketch: algorithm={sketch.algorithm}, "
              f"tiling=({sketch.tiling.block_m}x{sketch.tiling.block_n}x{sketch.tiling.block_k})")
        if similar:
            print(f"  [Designer] RAG 命中 {len(similar)} 个相似案例")
        return sketch

    def _llm_generate_sketch(self, task: TaskSchema, hardware_info: dict, similar: list, feedback: str = "") -> UnifiedSketch:
        rag_context = self._build_rag_context(similar)

        # ---- 提取硬件约束用于 tile 选择 ----
        cc = str(hardware_info.get("compute_capability", ""))
        hw_name = str(hardware_info.get("name", "")).lower()
        # 根据 compute_capability 推断共享内存和 SM 参数
        # A100: 164KB/SM (可配置到 48KB 默认), 2048 threads/SM, 32 blocks/SM
        # H100: 228KB/SM, 2048 threads/SM, 32 blocks/SM
        is_a100 = cc.startswith("8.0") or "a100" in hw_name
        is_h100 = cc.startswith("9.0") or "h100" in hw_name or "h200" in hw_name
        if is_h100:
            shared_mem_kb, max_threads, max_blocks = 228, 2048, 32
        elif is_a100:
            shared_mem_kb, max_threads, max_blocks = 164, 2048, 32
        else:
            shared_mem_kb = hardware_info.get("memory", {}).get("local", {}).get("size", "128KB")
            if isinstance(shared_mem_kb, str):
                shared_mem_kb = int(shared_mem_kb.replace("KB", "").replace("/SM", "").strip()) if "KB" in shared_mem_kb else 128
            max_threads = hardware_info.get("compute_units", 128) * 2  # rough estimate
            max_blocks = 32

        # 可用共享内存 (扣除系统预留 ~10%)
        usable_shared_mem_kb = int(shared_mem_kb * 0.9)

        # ── SKILL 选择策略 ──
        # triton_skill 中包含大量 tl.* / @triton.jit 专用指令
        # 当 dsl != "triton" (例如昇腾等硬件的 python / cuda_c 兼容层) 时, 强制不注入该 skill
        # 否则 LLM 很容易被诱导写出 Triton 代码 -> 在非 NVIDIA 硬件上编译必挂
        effective_skill = (
            self._skill_name if (task.target.dsl or "").lower() == "triton" else None
        )

        prompt = self.pm.render(
            "designer_sketch",
            _agent_role=self._agent_role,
            _skill_name=effective_skill,
            kernel_type=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=task.target.dsl,
            precision=task.constraints.precision.value,
            complexity=task.source.complexity.value,
            hardware_info=json.dumps(hardware_info, ensure_ascii=False, indent=2),
            rag_context=rag_context or "无",
            shared_mem_kb=str(shared_mem_kb),
            usable_shared_mem_kb=str(usable_shared_mem_kb),
            max_threads_per_sm=str(max_threads),
            max_blocks_per_sm=str(max_blocks),
        )

        # 追加上一轮校验反馈 (不为空时, 让 LLM 根据错误修正 JSON 输出)
        if feedback:
            prompt = prompt + feedback

        result = self.llm.chat_json(messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=2048, caller_tag="Designer")

        return UnifiedSketch(
            algorithm=result.get("algorithm", task.source.kernel_type),
            version=1,
            tiling=TilingStrategy(**result.get("tiling", {})),
            memory_plan=MemoryPlan(**result.get("memory_plan", {})),
            pipeline=[PipelineStage(**p) for p in result.get("pipeline", [])],
            target_dsl=task.target.dsl,
            optimization_hints=result.get("optimization_hints", []),
        )

    @staticmethod
    def _build_rag_context(similar: list) -> str:
        if not similar:
            return ""
        lines = []
        for i, r in enumerate(similar[:3]):
            line = f"案例{i+1}: algorithm={r.algorithm}, hardware={r.hardware}, "
            line += f"performance={r.performance} GFLOPS, tags={r.tags}"
            if r.sketch:
                line += f", sketch_tiling={r.sketch.get('tiling', {})}"
            lines.append(line)
        return "\n".join(lines)


class CoderAgent:
    """
    Coder Agent
    职责: 根据 Sketch + RAG 参考，生成目标代码
    基于 LLM 驱动生成
    """

    def __init__(self, kernel_db: KernelDatabase, llm=None, prompt_manager: Optional[PromptManager] = None,
                 pipeline=None):
        self.kernel_db = kernel_db
        self.llm = llm
        self.pm = prompt_manager or PromptManager()
        self.name = "Coder"
        self._agent_role = "coder"
        self._skill_name = "triton_skill"
        # 可选注入 CoderPipeline (Plan→并行采样→筛选→修复→回退)
        # 默认 None: 走 legacy 单线程生成+预检重试, 保持兼容
        self.pipeline = pipeline
        self.candidate_records = []
        self.last_candidate_record = None

    @property
    def last_finalists(self) -> list:
        """Pipeline 模式下最近一次 run 的决赛候选 (P1-1, ≤ CoderPipeline.L2_FINALISTS)

        读者是 Conductor: 对决赛候选逐个跑 L1+L2, 用真机正确性做最终选择。
        单体路径 (pipeline=None) 无此概念, 返回空列表 —— 调用方据此跳过选择。
        修复/模板回退的产物是单候选, 也在名单里 (len=1, 选择退化为不选)。
        """
        return self.pipeline.last_finalists if self.pipeline is not None else []

    def run(self, sketch: UnifiedSketch, task: TaskSchema, optimization_recipes=None,
            test_inputs=None, **kwargs) -> str:
        if self.llm is None:
            raise RuntimeError("LLM 未配置，规则模板已移除")

        # 注入了 CoderPipeline 时走优化流程 (Plan→并行采样→筛选→修复→回退)
        if self.pipeline is not None:
            try:
                code = self.pipeline.run(sketch, task, optimization_recipes=optimization_recipes,
                                         test_inputs=test_inputs, **kwargs)
            finally:
                self.candidate_records = list(getattr(self.pipeline, 'candidate_records', []))
                self.last_candidate_record = getattr(self.pipeline, 'last_candidate_record', None)
            print(f"  [Coder] Pipeline 生成代码: {len(code)} chars, language={sketch.target_dsl}")
            return code

        code = self._llm_generate_code(sketch, task, optimization_recipes=optimization_recipes,
                                       test_inputs=test_inputs)
        print(f"  [Coder] LLM 生成代码: {len(code)} chars, language={sketch.target_dsl}")
        return code

    MAX_GENERATE_ATTEMPTS = 3   # 首次生成 + 最多 2 次带预检反馈重试
    MAX_REF_CODE_CHARS = 2000   # 注入 prompt 的参考代码上限 (摘要级注入, 控制上下文)

    def _llm_generate_code(self, sketch: UnifiedSketch, task: TaskSchema,
                           optimization_recipes=None, test_inputs=None) -> str:
        self.candidate_records = []
        self.last_candidate_record = None
        context = build_generation_context(task, sketch, test_inputs,
            producer='CoderAgent', strategy='single-with-precheck-retry',
            policy=dict(max_attempts=self.MAX_GENERATE_ATTEMPTS, render_dsl=sketch.target_dsl),
            prompt_manager=self.pm, agent_role=self._agent_role,
            skill_name=self._skill_name if sketch.target_dsl == 'triton' else None)
        similar = self.kernel_db.search(algorithm=task.source.kernel_type, hardware=task.target.hardware, dsl=sketch.target_dsl)
        ref_code = similar[0].code if similar else ""
        if len(ref_code) > self.MAX_REF_CODE_CHARS:
            ref_code = ref_code[:self.MAX_REF_CODE_CHARS] + "\n# ... (参考代码已截断)"

        if sketch.target_dsl == "triton":
            template_name = "coder_generate_triton"
            effective_skill = self._skill_name
        elif sketch.target_dsl == "cuda_c":
            # C++ (PyTorch C++ Extension): 使用专用模板, 不注入 triton_skill
            template_name = "coder_generate_cuda_c"
            effective_skill = None
        else:
            template_name = "coder_generate"
            # 非 triton DSL: 不注入 triton_skill (tl.* / @triton.jit 暗示会严重干扰非 NVIDIA 硬件的代码生成)
            effective_skill = None

        hot = sketch.hot_hints()
        optimization_hints = "\n".join(f"- {h}" for h in hot) if hot else "无"
        # 配方**不再折进 hints**。以前 `optimization_recipes` 这个值算出来了却没传给
        # render, 而 `coder_generate_triton.md` 的「## 优化配方 (CRITICAL - MUST APPLY ALL)」
        # 那一段底下写的就是 `{{optimization_recipes}}` ⇒ 每一条 Triton 生成的 prompt 里,
        # 那个位置发给模型的是字面量 `{{optimization_recipes}}`。折进 hints 是绕过, 不是修:
        # 同一个模板一边说"这里放配方"一边什么都不放, 而 hints 那一块从此装两种东西。
        # (与 `schemas.py` 里 `to_prompt_dict` 那条"hint 只留一个通道"是同一条纪律。)
        recipes_text = optimization_recipes or "（本轮没有新增配方，按上述提示优化即可）"
        # 入口契约由 harness 说, 不由模型猜: 远端固定 `kernel(**输入)` (doc 41), 而
        # 输入名只有调用方知道。不广播的话, prompt 里唯一一处签名是 Planner 按
        # kernel_type 猜出来的 `def kernel(A, B)` ⇒ 对不上的任务只能等一次 SSH 往返抛出
        # TypeError 才知道契约长什么样。
        calling_convention = context['interface']['broadcast']

        prompt = self.pm.render(
            template_name,
            _agent_role=self._agent_role,
            _skill_name=effective_skill,
            kernel_type=context['task']['kernel_type'],
            hardware=context['task']['hardware'],
            dsl=sketch.target_dsl,
            precision=context['task']['precision'],
            sketch=json.dumps(sketch.to_prompt_dict(), ensure_ascii=False, indent=2),
            ref_code=ref_code or "无",
            optimization_hints=optimization_hints,
            optimization_recipes=recipes_text,
            calling_convention=calling_convention,
        )
        # tiling 的三个数不单独传: `{{sketch}}` 那份 JSON 里就带 `tiling.block_m/n/k`
        # (`UnifiedSketch.to_dict`), 单独传第二遍是同一份内容的第二个注入点。

        # 生成 -> 预检 -> 带反馈重试: 拦截输出截断/语法错误, 避免必然失败的编译轮
        feedback = ""
        code = ""
        for attempt in range(1, self.MAX_GENERATE_ATTEMPTS + 1):
            messages = [{"role": "user", "content": prompt + feedback}]
            request = generation_request(self.llm, messages, caller='Coder', temperature=0.1, max_tokens=40960)
            try:
                raw = self.llm.chat(messages, temperature=0.1, max_tokens=40960, caller_tag="Coder")
            except Exception as error:
                record = make_candidate_record(context, None, request=request, attempt=attempt,
                    status='model_failed', reason=type(error).__name__)
                self.candidate_records.append(record)
                self.last_candidate_record = record
                raise
            try:
                code = self._clean_code(raw)
            except Exception as error:
                record = make_candidate_record(context, None, request=request, attempt=attempt,
                    status='no_candidate', reason=type(error).__name__)
                self.candidate_records.append(record)
                self.last_candidate_record = record
                raise
            try:
                ok, issues = self._validate_code(code, sketch.target_dsl)
            except Exception as error:
                record = make_candidate_record(context, code, request=request, attempt=attempt,
                    status='precheck_failed', reason=type(error).__name__)
                self.candidate_records.append(record)
                self.last_candidate_record = record
                raise
            record = make_candidate_record(context, code, request=request, attempt=attempt,
                status='prechecked' if ok else 'precheck_failed')
            self.candidate_records.append(record)
            self.last_candidate_record = record
            if ok:
                if attempt > 1:
                    print(f"  [Coder] 第 {attempt} 次尝试通过预检")
                return code
            print(f"  [Coder] 代码预检失败 ({'; '.join(issues)})")
            feedback = self._build_retry_feedback(issues)
        print(f"  [Coder] 预检连续未通过，返回最后一次生成结果")
        return code

    @staticmethod
    def _validate_code(code: str, dsl: str = "python") -> tuple:
        """
        生成后预检 (不进沙箱的静态检查):
        - 截断检测: 括号不配对 / 结尾悬空 (max_tokens 截断的典型特征)
        - 结构检测: kernel 入口函数 / @triton.jit 装饰器
        - 语法检测: Python 系 DSL 用 compile() 验证语法
        - cuda_c 特检: 必须含 load_inline 和 _CPP_SOURCE
        返回 (是否通过, 问题列表)
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

        # 只要 dsl != "triton" (即 python/cuda_c/opencl 等) 就禁止出现 import triton / @triton.jit
        # 这是昇腾等非 Nvidia 硬件的关键防线
        if dsl != "triton":
            if "import triton" in code or "from triton" in code or "@triton.jit" in code:
                issues.append(
                    f"禁止: dsl={dsl!r} 但代码里出现了 Triton API (import triton / @triton.jit 等)。"
                    " 昇腾/其他非 Nvidia 硬件必须使用纯 Python/PyTorch/NumPy，不能用 Triton/CUDA。"
                )

        if dsl == "triton" and "@triton.jit" not in code:
            issues.append("缺少 @triton.jit 装饰器")

        # cuda_c (PyTorch C++ Extension) 专用预检
        if dsl == "cuda_c":
            if "load_inline" not in code:
                issues.append("cuda_c 代码缺少 load_inline 调用 (PyTorch C++ Extension JIT 编译)")
            if "_CPP_SOURCE" not in code and "cpp_sources" not in code:
                issues.append("cuda_c 代码缺少 C++ 源码 (_CPP_SOURCE 或 cpp_sources)")
            # 不走 Python compile() — 因为代码中包含 C++ 字符串，虽然 Python 语法合法，
            # 但 compile 无法检查 C++ 内容，直接跳过即可
        elif dsl in ("python", "triton"):
            try:
                compile(code, "<generated>", "exec")
            except SyntaxError as e:
                issues.append(f"语法错误: {e.msg} (第 {e.lineno} 行)")

        return (not issues, issues)

    @staticmethod
    def _build_retry_feedback(issues: list) -> str:
        bullets = "\n".join(f"- {i}" for i in issues)
        return (
            f"\n\n## 上次生成未通过预检\n{bullets}\n"
            f"请重新输出完整代码: 以 `import` 开头，包含 kernel 入口函数，"
            f"括号配对完整，不要中途截断，不要输出解释。"
        )

    @staticmethod
    def _clean_code(raw: str) -> str:
        """清理 LLM 生成的代码"""
        code = raw.strip()

        for end_tag in ["</think>", "</thinking>"]:
            idx = code.find(end_tag)
            if idx != -1:
                code = code[idx + len(end_tag):]
                break

        code = code.strip()

        if code.startswith("```"):
            lines = code.split("\n")
            start = 1
            if start < len(lines) and lines[start].strip().startswith(("python", "triton", "cpp", "cuda", "c++")):
                start += 1
            code = "\n".join(lines[start:])

        if code.rstrip().endswith("```"):
            code = code.rstrip()[:-3]

        code = code.strip()

        # 非 Triton 代码: 走通用提取，避免下方 Triton 专用清理管道
        # 因起始标记不匹配把纯 Python/CUDA 代码误删为空串
        if not any(m in code for m in ("import triton", "@triton.jit", "from triton", "triton.language")):
            extracted = FixerAgent._extract_code(code)
            return extracted if extracted else code

        valid_start_markers = ("import triton", "import torch", "from triton", "@triton.jit")
        found_idx = -1
        for marker in valid_start_markers:
            idx = code.find(marker)
            if idx >= 0 and (found_idx == -1 or idx < found_idx):
                found_idx = idx
        if found_idx >= 0:
            code = code[found_idx:]

        code = code.strip()

        lines = code.split("\n")
        cleaned_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith(valid_start_markers):
                in_code = True
                
            if in_code:
                if stripped == "```":
                    break
                if stripped.startswith("**") and not "@triton" in stripped and not "def " in stripped and not "import" in stripped:
                    break
                if stripped.startswith("*   ") or stripped.startswith("    *   "):
                    continue
                if stripped.startswith("    *") and len(stripped) > 6 and not stripped[6].isspace():
                    continue
                if "`" in stripped and ("import" in stripped or "def " in stripped or "@triton" in stripped):
                    stripped = stripped.replace("`", "")
                    line = " " * (len(line) - len(stripped)) + stripped
                cleaned_lines.append(line)

        if cleaned_lines:
            first_line_stripped = cleaned_lines[0].strip()
            first_line_indent = len(cleaned_lines[0]) - len(first_line_stripped)
            
            if first_line_indent == 0:
                pass
            else:
                min_indent = None
                for line in cleaned_lines:
                    stripped = line.strip()
                    if stripped:
                        indent = len(line) - len(stripped)
                        if min_indent is None or indent < min_indent:
                            min_indent = indent
                if min_indent and min_indent > 0:
                    cleaned_lines = [line[min_indent:] if len(line) >= min_indent else line for line in cleaned_lines]

        code = "\n".join(cleaned_lines).strip()

        if code.startswith("@triton.jit"):
            lines = code.split("\n")
            if len(lines) > 1 and lines[1].strip().startswith("def "):
                if lines[1].startswith("    def "):
                    min_indent = 4
                    fixed_lines = [lines[0]]
                    for line in lines[1:]:
                        if len(line) >= min_indent and line[:min_indent].isspace():
                            fixed_lines.append(line[min_indent:])
                        else:
                            fixed_lines.append(line)
                    code = "\n".join(fixed_lines)

        # 清理 docstring：只删除"无意义"的 LLM 残留 markdown 文本 (如包含 ``` 或 >10 行超长、
        # 或出现 ## 标题、含有 __all__/author 等与算法无关的 docstring)；保留 kernel 算法 docstring
        for quote in ['"""', "'''"]:
            while quote in code:
                start_idx = code.index(quote)
                end_idx = code.find(quote, start_idx + 3)
                if end_idx == -1:
                    # 未闭合的三引号：尽量保留到行末
                    end_of_line = code.find("\n", start_idx)
                    if end_of_line == -1:
                        code = code[:start_idx]
                    else:
                        code = code[:start_idx] + code[end_of_line + 1:]
                    continue
                body = code[start_idx + 3:end_idx]
                keep = True
                body_lines = [l.strip() for l in body.split("\n") if l.strip()]
                if not body_lines:
                    keep = False
                if any(tok in body for tok in ("```", "****", "__all__", "__version__", "Author:", "License")):
                    keep = False
                if any(l.startswith("## ") or l.startswith("### ") for l in body_lines):
                    keep = False
                # 单独的参数/返回 doc 超过 14 行往往是 LLM 在写说明书而非注释
                if len(body_lines) > 14:
                    # 包含至少一个"算法/Flash/分块/累加/指针/softmax/归约"等关键词则保留
                    kw = ("算法", "Flash", "分块", "累加", "指针", "softmax", "归约",
                          "精度", "边界", "causal", "Tensor Core", "Tiling", "tile")
                    if not any(k.lower() in body.lower() for k in kw):
                        keep = False
                if not keep:
                    code = code[:start_idx] + code[end_idx + 3:]
                else:
                    # 保留 docstring: 将开始/结束引号都替换为占位符,
                    # 避免 while 循环再次把开始引号当作"未闭合"而删掉整行
                    code = code[:start_idx] + "\x00\x03\x04" + body + "\x00\x01\x02" + code[end_idx + 3:]
            # 最后还原起止引号占位符
            code = code.replace("\x00\x03\x04", quote).replace("\x00\x01\x02", quote)

        lines = code.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # 仅删除明显由 LLM 生成的"结构化注释"（# ## 标题、# --- 分隔、过长的 TODO 占位），
            # 保留普通中文/英文行注释 (# 参数说明 / # 加载 tile / # softmax 更新 等)
            if stripped.startswith("#") and not stripped.startswith("#!"):
                body = stripped.lstrip("#").strip()
                if not body:
                    continue  # 空的 ## 注释
                if (body.startswith("## ") or body.startswith("### ")
                        or len(set(body)) == 1 and body[0] in "-=*~"):
                    continue  # # ## 标题 / # ---- 分隔线
                if body.lower().startswith(("note:", "todo:", "fixme:", "warning:")) and len(body) > 60:
                    continue
            cleaned_lines.append(line)

        code = "\n".join(cleaned_lines).strip()

        if "@triton.jit" in code and "import triton" not in code:
            code = "import triton\nimport triton.language as tl\nimport torch\n" + code
        elif "@triton.jit" in code:
            if "import triton.language as tl" not in code:
                import_triton_idx = code.find("import triton")
                if import_triton_idx >= 0:
                    end_of_line = code.find("\n", import_triton_idx)
                    if end_of_line > 0:
                        code = code[:end_of_line] + "\nimport triton.language as tl" + code[end_of_line:]
            if "import torch" not in code:
                import_triton_idx = code.find("import triton")
                if import_triton_idx >= 0:
                    end_of_line = code.find("\n", import_triton_idx)
                    if end_of_line > 0:
                        code = code[:end_of_line] + "\nimport torch" + code[end_of_line:]

        return code.strip()


class SearchAgent:
    """
    Search Agent (性能优化)
    职责: 当 L3 验证未通过时，分析瓶颈并提出优化建议
    基于 LLM 驱动分析

    P1-4 首片 (generation-side-review + doc 14): LLM 建议降为「先验」——
    tiling 这一低维规整空间交给确定性邻域搜索, LLM 保留结构级判断
    (hints/memory_plan/邻域中心)。旧实现的一次跳点直接被采纳, 失败了
    分不清「方向不行」还是「没搜到」。首片 = roofline 启发式排序 +
    Conductor 跨轮轮换 (真机 L3 当裁判, 改进即停); BO/bandit 的 surrogate
    需要实测分布, 留给后续片。
    """

    def __init__(self, llm=None, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm
        self.pm = prompt_manager or PromptManager()
        self.name = "Search"
        self._agent_role = "search"
        self._skill_name = "triton_skill"
        # P1-4: 最近一次 run() 排好序的邻域候选 (不含本轮选中者), 供 Conductor
        # 跨轮轮换。命名与 CoderPipeline.last_finalists 同族。
        self.last_alternatives: list[UnifiedSketch] = []

    def run(self, sketch: UnifiedSketch, perf_data: dict) -> tuple[UnifiedSketch, str]:
        if self.llm is None:
            raise RuntimeError("LLM 未配置，规则模板已移除")

        # §10.2 第三行: token 预算用尽 ⇒ **退化成纯数值搜索**, 不问 LLM 要结构改动。
        # 注意这不是「LLM 不在」(那条路会 raise: 规则模板已不再是缺 LLM 时的替补),
        # 而是「LLM 在, 但这一轮的额度花完了」—— 退化的后果正好是规则路径。
        # §10.3: 结构升级门未通过 ⇒ **只做参数搜索**（走规则/格点路径）。
        # 这条与上面那条 token 退化共用**同一条通道**（perf_data）、但是两件事:
        # 「额度花完了」是预算问题, 「三条件没同时成立」是判据问题 —— 读同一份
        # perf_data 的下游必须能分开说出是哪一条（键名不同 ⇒ 理由可归因）。
        _esc_denied = perf_data.get("structural_escalation_denied")
        if _esc_denied:
            prior_sketch, suggestion = self._rule_optimize(sketch, perf_data)
            print(f"  [Search] 结构升级门未通过 ⇒ 只做参数搜索 (规则模板): "
                  f"{'; '.join(str(r) for r in _esc_denied)}")
        elif perf_data.get("llm_budget_exhausted"):
            prior_sketch, suggestion = self._rule_optimize(sketch, perf_data)
            print(f"  [Search] token 预算用尽 ⇒ 纯数值搜索 (规则模板): {suggestion}")
        else:
            try:
                prior_sketch, suggestion = self._llm_optimize(sketch, perf_data)
            except Exception as e:
                # LLM 优化失败时回退到规则优化, 不中断整个任务
                print(f"  [Search] LLM 优化失败 ({e}), 回退规则优化")
                prior_sketch, suggestion = self._rule_optimize(sketch, perf_data)

        # P1-4: tiling 交给邻域搜索 (仅 triton —— python/cuda_c 的 block 参数
        # 不是真实旋钮, 轮换它们只会空烧轮次); 其余 DSL 维持「直接采纳」。
        if (sketch.target_dsl or "").lower() == "triton":
            prior_sketch, self.last_alternatives, note = self._neighborhood_search(
                prior_sketch, origin=sketch, perf_data=perf_data)
            if note:
                suggestion = f"{suggestion} ({note})"
            if self.last_alternatives:
                print(f"  [Search] P1-4 邻域: {len(self.last_alternatives)} 个候选排队, "
                      f"下轮 L3 再败时按序轮换")
        else:
            self.last_alternatives = []
        print(f"  [Search] LLM 优化建议: {suggestion}")
        return prior_sketch, suggestion

    # ─── P1-4 邻域搜索 ────────────────────────────────────────

    def _neighborhood_search(self, prior: UnifiedSketch, origin: UnifiedSketch,
                             perf_data: dict) -> tuple:
        """先验参与排序而非直接采纳; 返回 (chosen, alternatives, override_note)

        分工 (doc 14 的口径): LLM 出结构 (hints/memory_plan) 与中心点,
        tiling 的格点邻域交给确定性排序。排序只重排不删, 先验在池首位 +
        稳定排序 ⇒ 打平时先验仍胜 —— 只有启发式给出明确方向时才会位移。
        """
        center = prior.tiling
        origin_key = (origin.tiling.block_m, origin.tiling.block_n,
                      origin.tiling.block_k)
        # 与 origin 相同的 tiling 刚失败过, 不排进队列 (重复尝试不可归因)
        neighbor_sketches = [
            self._variant_of(prior, t) for t in self._lattice_neighbors(center)
            if (t.block_m, t.block_n, t.block_k) != origin_key
        ]
        # §12 的**便宜代理剪枝** (不花 GPU 秒): 把 smem 明显放不下的邻域点先摘掉,
        # 否则它会占掉一轮真值测量去买一个必然失败的 launch。上界取不到时不剪, 且
        # 留一句「未生效」——「没有上界」不许被读成「剪过了没事」(三态)。
        neighbor_sketches, prune_note = self._prune_by_smem(neighbor_sketches, perf_data)
        pool = [prior] + neighbor_sketches
        # 先验没动 tiling 且有别的可试: 再选它等于原样重跑一轮, 让位给邻域
        prior_key = (center.block_m, center.block_n, center.block_k)
        if prior_key == origin_key and neighbor_sketches:
            pool = neighbor_sketches
        ranked = self._rank(pool, center, perf_data)
        chosen, alternatives = ranked[0], ranked[1:]
        # 排队候选 version 顺延: 轮换进来时血缘打印单调递增
        for i, alt in enumerate(alternatives, start=1):
            alt.version = prior.version + i
        note = ""
        if chosen is not prior:
            t, c = chosen.tiling, center
            note = (f"邻域排序选中 {t.block_m}x{t.block_n}x{t.block_k}, "
                    f"LLM 先验 {c.block_m}x{c.block_n}x{c.block_k} 降为候选")
        if prune_note:
            note = f"{note} | {prune_note}" if note else prune_note
        return chosen, alternatives, note

    @staticmethod
    def _prune_by_smem(sketches: list, perf_data: dict) -> tuple:
        """返回候选与资源说明；raw perf_data 不提供可信 shared layout。

        caps 不是存储证据，任务精度也不能补出物理 shared 字节。缺证据保留候选，
        后续仍走原生检查和原有预算；不从用户测量字典注入权威资源回执。
        """
        points = []
        for i, s in enumerate(sketches):
            t = s.tiling
            points.append(TemplateCandidate(
                backend="lattice", template_id="tiling",
                params={"BLOCK_M": t.block_m, "BLOCK_N": t.block_n,
                        "BLOCK_K": t.block_k, "_idx": i}))
        # 上界的载体与口径只有一处 (`resource_check.caps_from_metrics`): 这里不再自己
        # 造一个 `_Caps`, 免得「搜索侧算出来的上界」与「预检/模板队列算出来的」两套。
        res = prune_by_smem(points, caps_from_metrics(perf_data))
        if res.no_limit:
            return list(sketches), ""
        kept = [sketches[c.params["_idx"]] for c in res.kept]
        note = res.as_note()
        if not note and res.kept_all_to_avoid_starvation:
            note = res.as_note()
        return kept, note

    @staticmethod
    def _variant_of(prior: UnifiedSketch, tiling: "TilingStrategy") -> UnifiedSketch:
        """同代候选: 复制先验的结构性字段, 只换 tiling"""
        return UnifiedSketch(
            algorithm=prior.algorithm, version=prior.version,
            tiling=tiling, memory_plan=prior.memory_plan, pipeline=prior.pipeline,
            target_dsl=prior.target_dsl,
            optimization_hints=prior.optimization_hints,
            hints_this_version=prior.hints_this_version,
        )

    @staticmethod
    def _lattice_neighbors(tiling: "TilingStrategy") -> list:
        """中心每维 ±1 格的单步邻域; 一次只动一维 (可归因: 变了哪个旋钮一目了然)

        中心不在合法格点上时不猜 (交给 schema 校验报错), 返回该维不动。
        单步邻域 ≤ 6 个, 天然有界, 不需额外截断。
        """
        lattices = (
            ("block_m", sorted(TilingStrategy._ALLOWED_BLOCK_M_N)),
            ("block_n", sorted(TilingStrategy._ALLOWED_BLOCK_M_N)),
            ("block_k", sorted(TilingStrategy._ALLOWED_BLOCK_K)),
        )
        out, seen = [], set()
        center = (tiling.block_m, tiling.block_n, tiling.block_k)
        for dim, lattice in lattices:
            cur = getattr(tiling, dim)
            if cur not in lattice:
                continue
            i = lattice.index(cur)
            for j in (i - 1, i + 1):
                if not (0 <= j < len(lattice)):
                    continue
                vals = dict(block_m=tiling.block_m, block_n=tiling.block_n,
                            block_k=tiling.block_k)
                vals[dim] = lattice[j]
                key = (vals["block_m"], vals["block_n"], vals["block_k"])
                if key == center or key in seen:
                    continue
                seen.add(key)
                out.append(TilingStrategy(**vals))
        return out

    @staticmethod
    def _rank(pool: list, center: "TilingStrategy", perf_data: dict) -> list:
        """roofline 启发式排序 (首片; 不是 BO —— surrogate 需要实测分布)

        方向判据 (doc 14 §二的 roofline 直觉):
          - memory-bound (AI 低于脊点): 每 tile 算术强度不足 → 优先增大 BLOCK_K
          - compute-bound: 疑似 occupancy 不足 → 优先减小 BLOCK_M/N
          - unknown: 不猜方向, 「离先验近者优先」的确定性序

        **方向不可达时不许倒退, 也不改猜另一个方向** (P1-4 边界修正): 中心已经顶在
        格点边界 (如 BLOCK_K=128) 时, 邻域里没有任何点能「继续增大 K」。旧写法按
        `-block_k` 排序, 边界处头名会落到**那个减小 K 的点**上; 先验在池里时又会被
        `-vol` 位移到更大的 M/N tile —— 也就是「启发式没给出方向, 先验却被位移」,
        与「只有给出明确方向才位移」的承诺不符。
        现在把「不得倒退」写成显式罚位: 目标轴上比中心差的候选一律垫底, 其余按
        (目标轴, 离中心步数, 体积) 排。于是方向不通时**先验 (步数 0) 胜出**;
        no-op 先验已被剔出池时, 退回各自的口径 (memory 更大 tile / compute 更小 tile)。
        """
        # `roofline_bound` 由 conductor 用 `.get(..., "unknown")` 填, 恒为非空字符串,
        # 所以这里**没有 `or bound_type` 回退** —— 那会是一段永不执行的代码 (旧写法
        # 就有, 已删)。脊点算不出时它是 "unknown", 与 optimization_planner 拒绝拿
        # bound_type 当可比轴的口径一致: 宁可说不知道, 不拿猜测穿轴。
        bound = str(perf_data.get("roofline_bound", "unknown")).lower()
        l_mn = sorted(TilingStrategy._ALLOWED_BLOCK_M_N)
        l_k = sorted(TilingStrategy._ALLOWED_BLOCK_K)
        center_area = center.block_m * center.block_n

        def steps(a, b, lattice):
            if a in lattice and b in lattice:
                return abs(lattice.index(a) - lattice.index(b))
            return 0

        def key(s: UnifiedSketch):
            t = s.tiling
            vol = t.block_m * t.block_n * t.block_k
            dist = (steps(t.block_m, center.block_m, l_mn)
                    + steps(t.block_n, center.block_n, l_mn)
                    + steps(t.block_k, center.block_k, l_k))
            if bound == "memory":
                backward = 0 if t.block_k >= center.block_k else 1
                return (backward, -t.block_k, dist, -vol)
            if bound == "compute":
                area = t.block_m * t.block_n
                backward = 0 if area <= center_area else 1
                return (backward, area, dist, vol)
            return (0, 0, dist, vol)

        return sorted(pool, key=key)

    def _llm_optimize(self, sketch, perf_data) -> tuple[UnifiedSketch, str]:
        # ── 没有调参资格的那一轮: **不许编数字** ──
        # 上游按设计 §10.4 的伪码把没资格的数挡在信念之外, 于是这里可能拿到一个
        # **不含任何性能数字**的 perf_data。原来的 `.get(k, 0)` 会把 0.00 印进 prompt,
        # LLM 拿到一个不存在的"事实"再据此改方向 —— 与「缺键 ≠ 0」是同一件事,
        # 只是发生在 prompt 里 (而 prompt 是唯一保证被模型看到的通道)。
        # `tuning_refused` 是上游显式给的信号, 不是靠「键少」猜出来的。
        refused = perf_data.get("tuning_refused")
        _NONE = "无可用读数"

        def _num(value, fmt="{:.2f}"):
            return _NONE if refused else fmt.format(value)

        speedup = perf_data.get("speedup", 0)
        target_speedup = perf_data.get("target_speedup", 1.0)
        achieve_pct = speedup / target_speedup * 100 if target_speedup > 0 else 0.0

        # ---- 提取 Roofline 分析数据 (Verifier 注入) ----
        hw_peak_gflops = perf_data.get("hw_peak_gflops", 0)
        hw_peak_bandwidth = perf_data.get("hw_peak_bandwidth_gbs", 0)
        gflops_util = perf_data.get("gflops_utilization_pct", 0)
        bw_util = perf_data.get("bandwidth_utilization_pct", 0)
        ai = perf_data.get("arithmetic_intensity", 0)
        ridge_point = perf_data.get("ridge_point", 0)
        roofline_bound = perf_data.get("roofline_bound", "unknown")
        hw_name = perf_data.get("hw_name", "unknown")
        hw_cc = perf_data.get("hw_compute_capability", "unknown")

        # ---- 构建优化历史 (避免重复失败的策略) ----
        history_lines = perf_data.get("optimization_history", [])
        history_text = "无" if not history_lines else "\n".join(
            f"- 第{i}轮: tile={h.get('tiling', 'N/A')}, GFLOPS={h.get('gflops', 0):.2f}, "
            f"策略={h.get('stage', 'N/A')}, 结果={h.get('outcome', 'N/A')}"
            for i, h in enumerate(history_lines, 1)
        )

        # ── SKILL 选择策略 ──
        # triton_skill 的 §3 是 NVIDIA 专用的性能优化优先级
        # 当 sketch.target_dsl != "triton" 时, 不注入以免把优化方向带偏
        effective_skill = (
            self._skill_name if (sketch.target_dsl or "").lower() == "triton" else None
        )

        prompt = self.pm.render(
            "search_optimize",
            _agent_role=self._agent_role,
            _skill_name=effective_skill,
            sketch=json.dumps(sketch.to_dict(), ensure_ascii=False, indent=2),
            # `target_speedup` 是**任务常量**不是读数 ⇒ 照常印 (它给模型一个标尺)
            perf_availability=(
                "⚠️ **本轮没有任何可用的性能读数**（上一次测量没有调参资格，"
                "按设计 §4.3 不得驱动调参）。下面表里的数字一律不可用 —— "
                "请不要假设任何性能事实，只做**结构性**改进（访存/流水/融合/hints），"
                "并把 `new_tiling` 留空。"
                if refused else ""),
            speedup=_num(speedup),
            target_speedup=f"{target_speedup:.2f}",
            achieve_ratio=_num(achieve_pct, "{:.1f}%"),
            current_gflops=_num(perf_data.get("gflops", 0)),
            memory_bandwidth_gbs=_num(perf_data.get("memory_bandwidth_gbs", 0)),
            # ---- Roofline 数据 ----
            hw_peak_gflops=_num(hw_peak_gflops, "{:.1f}"),
            hw_peak_bandwidth_gbs=_num(hw_peak_bandwidth, "{:.1f}"),
            gflops_utilization_pct=_num(gflops_util, "{:.1f}%"),
            bandwidth_utilization_pct=_num(bw_util, "{:.1f}%"),
            arithmetic_intensity=_num(ai, "{:.4f}"),
            ridge_point=_num(ridge_point),
            roofline_bound=roofline_bound,
            name_bound=perf_data.get("name_bound", "unknown"),
            hw_name=hw_name,
            hw_compute_capability=hw_cc,
            optimization_history=history_text,
        )

        if perf_data.get("performance_scope") == "declared_workload" and not refused:
            point = perf_data["search_objective"]
            prompt += ("\n\n【冻结的 workload 搜索目标】\n"
                "上述加速比与达标比例是整体目标；GFLOPS/带宽/roofline 是主 case 诊断，"
                "不是候选排名或整体收敛依据。不得改权重、删 case 或放宽回退上限。\n"
                + json.dumps({"identity": point["identity"], "objective": point["score"],
                              "cases": perf_data["workload_cases"]}, ensure_ascii=False, indent=2))

        # max_tokens 调高至 4096, 避免 thinking 占满后 JSON 没输出
        result = self.llm.chat_json([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=4096, caller_tag="Search")

        new_tiling = result.get("new_tiling", {})
        tiling = TilingStrategy(
            block_m=new_tiling.get("block_m", sketch.tiling.block_m),
            block_n=new_tiling.get("block_n", sketch.tiling.block_n),
            block_k=new_tiling.get("block_k", sketch.tiling.block_k),
        ) if new_tiling else sketch.tiling

        new_hints = result.get("new_hints", [])
        suggestion = result.get("suggestion", "LLM优化")

        new_sketch = UnifiedSketch(
            algorithm=sketch.algorithm, version=sketch.version + 1,
            tiling=tiling, memory_plan=sketch.memory_plan, pipeline=sketch.pipeline,
            target_dsl=sketch.target_dsl,
            optimization_hints=_hint_log(sketch, new_hints),
            hints_this_version=list(new_hints),
        )
        return new_sketch, suggestion

    def _rule_optimize(self, sketch: UnifiedSketch, perf_data: dict) -> tuple[UnifiedSketch, str]:
        """
        规则优化回退: 基于 TritonOptimizationRules 智能调整 tiling
        根据 bound 类型和硬件特性推荐最优 tiling 组合
        """
        old_bm, old_bn, old_bk = sketch.tiling.block_m, sketch.tiling.block_n, sketch.tiling.block_k
        bound = perf_data.get("bound_type", "unknown")
        hw_name = perf_data.get("hw_name", "unknown")
        hw_cc = perf_data.get("hw_compute_capability", "unknown")

        new_tiling, reason = TritonOptimizationRules.recommend_tiling(
            sketch.tiling, bound, hw_name, hw_cc
        )
        new_bm, new_bn, new_bk = new_tiling.block_m, new_tiling.block_n, new_tiling.block_k

        # 生成优化提示
        current_gflops = perf_data.get("gflops", 0)
        target_gflops = perf_data.get("target_gflops", current_gflops) if current_gflops > 0 else 0
        hints = TritonOptimizationRules.generate_optimization_hints(
            bound, current_gflops, target_gflops, new_tiling, "float32"
        )

        new_sketch = UnifiedSketch(
            algorithm=sketch.algorithm, version=sketch.version + 1,
            tiling=new_tiling, memory_plan=sketch.memory_plan,
            pipeline=sketch.pipeline, target_dsl=sketch.target_dsl,
            optimization_hints=_hint_log(sketch, hints),
            hints_this_version=list(hints),
        )
        return new_sketch, f"规则优化: tile {old_bm}x{old_bn}x{old_bk}→{new_bm}x{new_bn}x{new_bk} ({reason})"
