"""Optimization Planner — 三层架构 (Layer 0 原则 + Layer 1 能力 + Layer 2 DSL 模板)

确定性优化配方选择器，不依赖 LLM，基于规则从配方库中选择适用的配方。

三层架构:
  Layer 0 — 优化原则 (硬件/DSL 无关): tiling, fusion, double-buffering 等概念
  Layer 1 — 硬件能力: 通过 HardwareCaps 判断硬件是否支持某特性
  Layer 2 — DSL 代码模板: 同一原则在 Triton/CUDA-C 等不同 DSL 下的代码模式

适配新硬件 (如昇腾) 只需要:
  1. 在 hardware_caps.py 添加硬件能力描述
  2. (可选) 在 optimization_recipes.md 添加对应 DSL 的 *_template
  3. Planner 自动根据 caps + dsl 筛选和注入正确模板
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .hardware_caps import HardwareCaps, detect_hardware_caps, get_caps_for_target


@dataclass
class Recipe:
    """单个优化配方 (三层结构)"""
    id: str
    name: str
    applies_to: list[str] = field(default_factory=list)
    applies_when: str = "always"
    priority: int = 99
    requires_caps: list[str] = field(default_factory=list)  # Layer 1: 硬件能力要求
    expected_impact: str = ""
    description: str = ""
    # Layer 2: 按 DSL 分类的代码模板
    dsl_templates: dict[str, str] = field(default_factory=dict)
    generic_template: str = ""
    notes: str = ""
    # 运行时状态
    enabled: bool = True
    # 条件解析后
    _conditions: dict = field(default_factory=dict)

    def get_template_for_dsl(self, dsl: str) -> str:
        """根据 DSL 获取对应的代码模板，找不到时回退到 generic_template"""
        dsl_lower = (dsl or "").lower().replace("-", "_")
        # 直接匹配
        if dsl_lower in self.dsl_templates:
            return self.dsl_templates[dsl_lower]
        # triton -> python 模板也可以
        if dsl_lower == "triton" and "python" in self.dsl_templates:
            return self.dsl_templates["python"]
        # cuda_c -> cuda 兼容
        if dsl_lower in ("cuda_c", "cuda") and "cuda" in self.dsl_templates:
            return self.dsl_templates["cuda"]
        # generic fallback
        return self.generic_template


class OptimizationPlanner:
    """
    确定性优化配方选择器 (三层架构)

    用法:
        planner = OptimizationPlanner()
        plan = planner.plan(
            kernel_type="matmul", dsl="triton",
            roofline=roofline_metrics, hardware=hw_info,
            precision="fp32", round_num=1,
        )
        prompt_text = plan.to_prompt()  # 注入 Coder
    """

    # DSL 别名映射 (将不同 DSL 名映射到配方中的模板 key)
    DSL_ALIASES = {
        "triton": "triton",
        "python": "triton",
        "cuda_c": "cuda_c",
        "cuda": "cuda_c",
        "hip": "cuda_c",  # AMD HIP 语法基本兼容 CUDA
        "ascend_c": "generic",
        "bisheng": "generic",
    }

    def __init__(self, recipes_path: Optional[str] = None):
        if recipes_path is None:
            recipes_path = os.path.join(
                os.path.dirname(__file__), "skills", "optimization_recipes.md"
            )
        self.recipes_path = recipes_path
        self._all_recipes: list[Recipe] = []
        self._load_recipes()

    def _load_recipes(self):
        """从 markdown 文件解析所有 Recipe (支持多 DSL 模板)"""
        if not os.path.exists(self.recipes_path):
            return
        with open(self.recipes_path, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = re.compile(
            r"^### (R-[A-Z]+-\d+): (.+?)$(.*?)(?=^### R-|^## |\Z)",
            re.MULTILINE | re.DOTALL,
        )
        for match in pattern.finditer(content):
            rid = match.group(1)
            name = match.group(2).strip()
            body = match.group(3)
            recipe = self._parse_recipe(rid, name, body)
            if recipe:
                self._all_recipes.append(recipe)

    def _parse_recipe(self, rid: str, name: str, body: str) -> Optional[Recipe]:
        """解析单个 recipe，支持多 DSL 模板"""
        recipe = Recipe(id=rid, name=name)

        # 提取所有代码块 (按模板类型分类)
        # 格式: - **triton_template**: 或 - **cuda_c_template**: 后跟代码块
        # 先提取命名的模板段
        template_sections = {}
        # 匹配 - **xxx_template**: 然后后面的代码块
        tmpl_pattern = re.compile(
            r"- \*\*(\w+_template)\*\*:\s*\n(```\w*\n.*?```)",
            re.DOTALL,
        )
        for tm in tmpl_pattern.finditer(body):
            tmpl_name = tm.group(1).replace("_template", "")
            code_block = tm.group(2)
            # 从代码块中提取代码 (去掉 ```lang 和 ```)
            code_match = re.match(r"```\w*\n(.*?)```", code_block, re.DOTALL)
            if code_match:
                template_sections[tmpl_name] = code_match.group(1).strip()

        # 移除已提取的模板段，避免干扰元数据解析
        body_clean = tmpl_pattern.sub("", body)

        # 提取元数据
        meta_pattern = re.compile(r"^- \*\*(.+?)\*\*: (.+)$", re.MULTILINE)
        description_lines = []
        in_description = False

        for line in body_clean.split("\n"):
            m = meta_pattern.match(line)
            if m:
                in_description = False
                key = m.group(1).strip().lower()
                val = m.group(2).strip()
                if key == "applies_to":
                    # 分割逗号，但保留括号内的内容作为一个整体
                    # 例如 "elementwise (gelu, silu, add, mul)" 不应被分割
                    recipe.applies_to = self._split_applies_to(val)
                elif key == "applies_when":
                    recipe.applies_when = val
                    recipe._conditions = self._parse_conditions(val)
                elif key == "priority":
                    try:
                        recipe.priority = int(val)
                    except ValueError:
                        pass
                elif key == "requires_caps":
                    # 解析硬件能力要求，如 "has_tensor_core=True, smem>=64"
                    recipe.requires_caps = self._parse_caps_requirements(val)
                elif key == "expected_impact":
                    recipe.expected_impact = val
                elif key == "description":
                    in_description = True
                    description_lines.append(val)
            elif in_description and line.strip() and not line.startswith("-"):
                description_lines.append(line.strip())
            elif in_description and (line.startswith("```") or line.startswith("- **")):
                in_description = False

        recipe.description = " ".join(description_lines)

        # 存储 DSL 模板
        for dsl_name, code in template_sections.items():
            if dsl_name == "generic":
                recipe.generic_template = code
            else:
                recipe.dsl_templates[dsl_name] = code

        # 兼容旧格式: 如果没有 dsl_templates 但有 code_template，放入 triton
        if not recipe.dsl_templates and not recipe.generic_template:
            # 尝试从 body_clean 提取第一个 python 代码块
            cb = re.search(r"```python\n(.*?)```", body_clean, re.DOTALL)
            if cb:
                recipe.dsl_templates["triton"] = cb.group(1).strip()

        return recipe if recipe.id else None

    @staticmethod
    def _split_applies_to(val: str) -> list[str]:
        """解析 applies_to 值，正确处理括号内的逗号。
        例如 "elementwise (gelu, silu, add, mul)" -> ["elementwise (gelu, silu, add, mul)"]
        "matmul, gemm, linear" -> ["matmul", "gemm", "linear"]
        """
        result = []
        depth = 0
        current = []
        for ch in val:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                result.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            result.append(''.join(current).strip())
        return result

    @staticmethod
    def _parse_caps_requirements(caps_str: str) -> list[str]:
        """
        解析 requires_caps 字符串为能力检查列表。
        支持的格式:
          - "has_tensor_core=True"
          - "has_software_pipelining=True"
          - "l2_cache_size_kb > 0"
          - "shared_mem_per_sm_kb >= 64"
          - 无值的表示布尔要求 (如 "has_l2_cache_hint" 等同于 "has_l2_cache_hint=True")
          - "(无)" 或中文说明文字表示无硬性要求
        """
        reqs = []
        if not caps_str or caps_str.strip().lower() in ("(无)", "none", "无", "无硬性要求", "无特殊要求"):
            return reqs
        # 过滤掉纯中文描述性文字（不含 has_/比较运算符/等号的）
        for part in caps_str.split(","):
            part = part.strip().strip("()")
            if not part:
                continue
            # 如果是纯中文/英文描述（不包含 has_、>、<、=、>=、<=），跳过
            if not any(kw in part for kw in ("has_", ">", "<", "=", "registers", "l2_cache", "shared_mem", "vector_width", "sm_count")):
                continue
            reqs.append(part)
        return reqs

    @staticmethod
    def _check_caps_requirement(req: str, caps: HardwareCaps) -> bool:
        """检查单个硬件能力要求是否满足"""
        req = req.strip()
        # 布尔要求: "has_tensor_core"
        if req in ("has_tensor_core", "has_software_pipelining",
                   "has_double_buffering", "has_l2_cache_hint",
                   "has_block_reduce", "has_warp_shuffle",
                   "has_atomic_fadd", "has_fast_math"):
            return bool(getattr(caps, req, False))

        # 布尔要求显式: "has_tensor_core=True"
        bool_match = re.match(r"(has_\w+)\s*=\s*(true|false)", req, re.IGNORECASE)
        if bool_match:
            attr = bool_match.group(1)
            expected = bool_match.group(2).lower() == "true"
            return bool(getattr(caps, attr, False)) == expected

        # 数值比较: "l2_cache_size_kb > 0", "shared_mem_per_sm_kb >= 64"
        num_match = re.match(r"(\w+)\s*(>=|<=|>|<|==)\s*(\d+)", req)
        if num_match:
            attr = num_match.group(1)
            op = num_match.group(2)
            val = int(num_match.group(3))
            actual = getattr(caps, attr, 0) or 0
            if op == ">=":
                return actual >= val
            elif op == "<=":
                return actual <= val
            elif op == ">":
                return actual > val
            elif op == "<":
                return actual < val
            elif op == "==":
                return actual == val

        # vector_width_bytes >= 16 之类
        # 如果无法解析，默认通过 (不阻塞)
        return True

    def plan(
        self,
        kernel_type: str,
        dsl: str = "triton",
        roofline: Optional[dict] = None,
        hardware: Optional[dict] = None,
        hw_caps: Optional[HardwareCaps] = None,
        precision: str = "fp32",
        round_num: int = 1,
        disabled_recipes: Optional[set[str]] = None,
        active_recipes: Optional[set[str]] = None,
    ) -> "OptimizationPlan":
        """
        根据当前状态生成优化计划 (三层架构)

        Args:
            kernel_type: 算子类型
            dsl: 目标 DSL (triton, cuda_c, etc.)
            roofline: Roofline 分析结果
            hardware: 硬件信息 dict (来自 backend.get_hardware_info())
            hw_caps: 硬件能力描述 (若提供则直接使用，否则从 hardware 推断)
            precision: 精度
            round_num: 当前优化轮次
            disabled_recipes: 已证明降低性能的配方 (禁用)
            active_recipes: 已成功应用的配方 (继续注入)

        Returns:
            OptimizationPlan
        """
        disabled_recipes = disabled_recipes or set()
        active_recipes = active_recipes or set()
        roofline = roofline or {}
        hardware = hardware or {}

        # Layer 1: 获取硬件能力描述
        if hw_caps is None:
            hw_caps = detect_hardware_caps(hardware) if hardware else get_caps_for_target("nvidia_gpu")

        # DSL 标准化
        dsl_normalized = self.DSL_ALIASES.get(dsl.lower(), dsl.lower())

        kt = (kernel_type or "").lower()
        bound_type = roofline.get("bound_type", "unknown")
        gflops_util = roofline.get("gflops_utilization_pct", 0)
        bw_util = roofline.get("bandwidth_utilization_pct", 0)

        selected: list[Recipe] = []
        rejected: list[tuple[Recipe, str]] = []

        for recipe in self._all_recipes:
            # 禁用检查
            if recipe.id in disabled_recipes:
                rejected.append((recipe, "disabled (proven to hurt performance)"))
                continue
            # 算子类型匹配
            if not self._matches_kernel_type(recipe.applies_to, kt):
                rejected.append((recipe, f"wrong kernel type (needs {recipe.applies_to})"))
                continue
            # Layer 1: 硬件能力检查
            caps_ok, caps_reason = self._check_all_caps(recipe.requires_caps, hw_caps)
            if not caps_ok:
                rejected.append((recipe, f"missing hardware capability: {caps_reason}"))
                continue
            # 条件检查 (Roofline, 精度等)
            if not self._check_conditions(recipe, kt, bound_type, gflops_util, bw_util,
                                          precision, round_num, hw_caps):
                rejected.append((recipe, f"conditions not met: {recipe.applies_when}"))
                continue
            # Layer 2: 检查是否有对应 DSL 的模板 (generic 总是可用)
            template = recipe.get_template_for_dsl(dsl_normalized)
            if not template and not recipe.generic_template and not recipe.description:
                rejected.append((recipe, f"no template available for DSL '{dsl}'"))
                continue
            selected.append(recipe)

        selected.sort(key=lambda r: r.priority)

        return OptimizationPlan(
            kernel_type=kernel_type,
            dsl=dsl_normalized,
            round_num=round_num,
            selected_recipes=selected,
            rejected_recipes=rejected,
            roofline=roofline,
            precision=precision,
            hw_caps=hw_caps,
        )

    def _check_all_caps(self, reqs: list[str], caps: HardwareCaps) -> tuple[bool, str]:
        """检查所有硬件能力要求，返回 (是否满足, 失败原因)"""
        for req in reqs:
            if not self._check_caps_requirement(req, caps):
                return False, req
        return True, ""

    @staticmethod
    def _matches_kernel_type(applies_to: list[str], kernel_type: str) -> bool:
        if not applies_to:
            return False
        for at in applies_to:
            at_lower = at.lower().strip()
            if at_lower in ("all", "any"):
                return True
            # gemm 也匹配 matmul
            if at_lower == "gemm" and "matmul" in kernel_type:
                return True
            if at_lower == "matmul" and "gemm" in kernel_type:
                return True
            if at_lower in kernel_type or kernel_type in at_lower:
                return True
        return False

    @staticmethod
    def _parse_conditions(cond_str: str) -> dict:
        conds: dict = {
            "always": False,
            "compute_bound": False,
            "memory_bound": False,
            "low_utilization": 0,
            "low_bandwidth_util": 0,
            "precision": [],
            "min_round": 1,
        }
        cl = cond_str.lower().strip()
        if cl in ("always", ""):
            conds["always"] = True
            return conds
        if "compute-bound" in cl or "compute_bound" in cl:
            conds["compute_bound"] = True
        if "memory-bound" in cl or "memory_bound" in cl:
            conds["memory_bound"] = True
        if "utilization" in cl and "<" in cl:
            m = re.search(r"util(?:ization)?\s*<\s*(\d+)", cl)
            if m:
                conds["low_utilization"] = int(m.group(1))
        if "bandwidth" in cl and "<" in cl:
            m = re.search(r"bandwidth.*?<\s*(\d+)", cl)
            if m:
                conds["low_bandwidth_util"] = int(m.group(1))
        if "fp16" in cl:
            conds["precision"].append("fp16")
        if "bf16" in cl:
            conds["precision"].append("bf16")
        if "fp32" in cl and "tf32" in cl:
            conds["precision"].append("tf32")
        if "round" in cl and ">" in cl:
            m = re.search(r"round\s*>\s*(\d+)", cl)
            if m:
                conds["min_round"] = int(m.group(1)) + 1
        return conds

    @staticmethod
    def _check_conditions(
        recipe: Recipe,
        kernel_type: str,
        bound_type: str,
        gflops_util: float,
        bw_util: float,
        precision: str,
        round_num: int,
        caps: HardwareCaps,
    ) -> bool:
        conds = recipe._conditions
        if conds.get("always"):
            return True
        # compute_bound: 首轮(无 roofline 数据, bound_type=unknown)时默认视为 compute-bound
        # 因为大部分大矩阵算子都是 compute-bound
        if conds.get("compute_bound"):
            if bound_type == "memory":
                return False
            # bound_type == "compute" 或 "unknown" (首轮) 都通过
        if conds.get("memory_bound") and bound_type != "memory":
            return False
        # 低利用率条件: 首轮无数据时不做限制 (视为 0 < threshold 通过)
        low_thresh = conds.get("low_utilization", 0)
        if low_thresh and gflops_util > 0 and gflops_util >= low_thresh:
            return False
        bw_thresh = conds.get("low_bandwidth_util", 0)
        if bw_thresh and bw_util > 0 and bw_util >= bw_thresh:
            return False
        req_prec = conds.get("precision", [])
        if req_prec:
            if precision == "fp32" and "tf32" in req_prec and caps.tf32_tflops > 0:
                pass
            elif precision not in req_prec:
                if not ("matmul" in kernel_type and caps.has_tensor_core and precision == "fp32"):
                    return False
        if round_num < conds.get("min_round", 1):
            return False
        return True


@dataclass
class OptimizationPlan:
    """一次优化计划"""
    kernel_type: str
    dsl: str
    round_num: int
    selected_recipes: list[Recipe]
    rejected_recipes: list[tuple[Recipe, str]]
    roofline: dict
    precision: str
    hw_caps: HardwareCaps

    @property
    def recipe_ids(self) -> list[str]:
        return [r.id for r in self.selected_recipes]

    def to_prompt(self) -> str:
        """将选中的配方格式化为可注入 Coder prompt 的文本 (DSL-aware)"""
        if not self.selected_recipes:
            return "## 优化配方\n本轮无新增优化配方。\n"

        dsl_label = {
            "triton": "Triton",
            "cuda_c": "CUDA-C",
            "generic": "通用伪代码",
        }.get(self.dsl, self.dsl)

        lines = ["## 优化配方 (CRITICAL - MUST APPLY ALL)"]
        lines.append(f"算子类型: {self.kernel_type} | 轮次: {self.round_num} | "
                     f"精度: {self.precision} | DSL: {dsl_label}")
        lines.append(f"硬件: {self.hw_caps.vendor} {self.hw_caps.device_name} "
                     f"(Tensor Core={'Y' if self.hw_caps.has_tensor_core else 'N'}, "
                     f"SM={self.hw_caps.sm_count}, "
                     f"BW={self.hw_caps.global_mem_bandwidth_gbs:.0f}GB/s)")
        if self.roofline:
            lines.append(
                f"当前状态: bound={self.roofline.get('bound_type', '?')}, "
                f"GFLOPS利用率={self.roofline.get('gflops_utilization_pct', 0):.1f}%, "
                f"带宽利用率={self.roofline.get('bandwidth_utilization_pct', 0):.1f}%"
            )
        lines.append("")
        lines.append("以下优化配方按优先级排列，你必须严格按照每个配方的代码模式编写代码，不得遗漏。")
        lines.append("配方中的 description 解释了优化原则，代码模板给出了具体写法。")
        lines.append("")

        for recipe in self.selected_recipes:
            lines.append(f"### [{recipe.priority}] {recipe.id}: {recipe.name}")
            lines.append(f"预期收益: {recipe.expected_impact}")
            if recipe.description:
                lines.append(f"优化原则: {recipe.description}")
            # 获取当前 DSL 的模板
            template = recipe.get_template_for_dsl(self.dsl)
            if template:
                # 根据 DSL 选择代码块语言标签
                lang_tag = "python" if self.dsl == "triton" else (
                    "cuda" if self.dsl == "cuda_c" else ""
                )
                lines.append("代码模板 (必须遵循此模式):")
                lines.append(f"```{lang_tag}")
                lines.append(template)
                lines.append("```")
            elif recipe.generic_template:
                lines.append("通用参考模式 (请根据当前DSL语法实现):")
                lines.append("```")
                lines.append(recipe.generic_template)
                lines.append("```")
            if recipe.notes:
                lines.append(f"注意: {recipe.notes}")
            lines.append("")
        return "\n".join(lines)

    def summary(self) -> str:
        """简短摘要，用于日志"""
        ids = ", ".join(self.recipe_ids) if self.selected_recipes else "none"
        caps_info = f"vendor={self.hw_caps.vendor}"
        return (f"Plan(round={self.round_num}, dsl={self.dsl}, {caps_info}, "
                f"recipes=[{ids}], n_selected={len(self.selected_recipes)})")
