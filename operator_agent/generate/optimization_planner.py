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

from ..core.hardware_caps import (
    HardwareCaps,
    UnknownHardware,
    detect_hardware_caps,
    get_caps_for_target,
    unknown_caps,
)


class CapsRequirementUnparsable(ValueError):
    """`requires_caps` 里出现判不了的条件。

    判不了 ≠ 满足: 这里曾经 `return True`, 于是配方里一个拼错的字段名会让该配方在
    **任何**硬件上都入选。抛出来是为了让「无法判定」在类型上不可能被读成 True;
    策略留给调用方 —— `_check_all_caps` 把它当成不通过并写进拒绝原因, 因为一轮
    plan() 崩掉的代价比少选一条配方大。
    """

    def __init__(self, req: str):
        self.req = req
        super().__init__(f"无法判定的 requires_caps 条件: {req!r}")


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


#: 轴向的闭合词汇。只有这两个值能参与筛选; `"unknown"` / 契约测试里的 `"mixed"` /
#: 拼错的词一律按「不知道」处理。词表以 Verifier 的产出为准 (裸词), 这里不另立第二套
#: —— 两套词汇之间的那次漂移 (`memory_bound` vs `memory`) 让 bound 感知静默失效过。
AXIS_VALUES = ("memory", "compute")


def _axis_field(src: dict, key: str):
    """同一个键在两种传入形状里位置不同, 两边都读。

    生产调用方 (`conductor.py:497`) 传的是 **L3 metrics 顶层**: Verifier 在那儿平铺了
    `bound_type` / `roofline_bound` / 两侧利用率 (`verifier.py:595-606`), 而
    `peak_trusted` **只活在 `roofline_analysis` 块里**; 但 `plan()` 的 docstring 说的是
    「Roofline 分析结果」那一个块。只按一种形状读, 「我没接对」就会被读成「没有这个
    数据」—— 同一个位置错误在 doc 50 §7.5 里让一张表的表头打成了 None。
    """
    if key in src:
        return src[key]
    block = src.get("roofline_analysis")
    if isinstance(block, dict):
        return block.get(key)
    return None


def axis_evidence(src: dict) -> tuple[bool, str]:
    """这一轮的 roofline 块能不能拿来筛配方。返回 `(能用, 原因)`。

    关键在于**轴向不是测出来的**: `bound_type` 是算术强度 AI 与脊点
    (`peak_gflops / peak_bw`) 比出来的, 而 AI 来自任务声明的 shape 与精度。所以它的
    可信度取决于脊点算没算出来、以及脊点用的那个分母口径核过没有 —— 跟「这一轮的计时
    可不可信」是两个问题, 不要用 `measurement_validity` 判它 (那是计时链的门, 已经有
    自己的三态)。把两者混成一个数是 doc 49 §八 第 17 条那种「一个块里两个口径」。

    三条门各自对应一种实测过的失效:
      * `src` 为空 ⇒ 首轮压根没有上一轮的测量。
      * `roofline_bound == "unknown"` ⇒ 脊点没算出来 (peak=0), 此时 `_roofline_analysis`
        会把 `bound_type` **兜到算子名启发式** (`verifier.py:1006`), 于是同一个块里
        `roofline_bound="unknown"` 与 `bound_type="compute"` 并存 —— 后者是猜测穿了
        测量的衣服。
      * `peak_trusted is not True` ⇒ 分母来自未核口径的表行 (`⚠️` 前缀) 或根本没命中表,
        比较的两边不同源。利用率两个百分比同样是从这个分母除出来的, 一起作废。

    取不到 ⇒ **不据此过滤**, 而不是「默认视为 compute-bound」。方向是刻意选的一侧:
    基于一个无法核实的轴去**拒绝**一条配方, 会在 `rejected_recipes` 里留下一个看起来
    很合理的理由, 而没人会去查它凭什么 —— 与 `measurement.py` 里「不可信不能说已到
    上界」是同一个不对称 (误拒不报错, 误放只是多几轮)。
    """
    src = src or {}
    if not src:
        return False, "本轮没有上一轮的测量 (首轮)"
    bound = _axis_field(src, "bound_type")
    if bound not in AXIS_VALUES:
        return False, f"bound_type={bound!r} 不在轴向词汇 {AXIS_VALUES} 内"
    roofline_bound = _axis_field(src, "roofline_bound")
    if roofline_bound == "unknown":
        return False, ("roofline_bound=unknown: 脊点没算出来 (peak=0), 此时 bound_type "
                       "是兜到算子名启发式的猜测, 不是这条轴上的证据")
    if roofline_bound is not None and roofline_bound not in AXIS_VALUES:
        return False, f"roofline_bound={roofline_bound!r} 不是可比的轴"
    trusted = _axis_field(src, "peak_trusted")
    if trusted is not True:
        return False, (f"peak_trusted={trusted!r}: 分母口径未核 (未命中表 / ⚠️ 行 / 缺键), "
                       "AI 与脊点两边不同源, 判出的轴和两侧利用率都不承重")
    return True, f"bound={bound} (脊点与分母口径都可核)"


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
                os.path.dirname(os.path.dirname(__file__)), "skills", "optimization_recipes.md"
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
    def _evaluate_caps_requirement(req: str, caps: HardwareCaps) -> Optional[bool]:
        """判定单条能力要求; 返回 None 表示这条字符串根本判不了 (配方写错)。

        三态而不是 bool: 有了 None 这一态, 「判不了」就没法在下游被读成「满足」。
        """
        req = req.strip()
        # 布尔要求裸名: "has_tensor_core" 等同 "has_tensor_core=True"。
        # 按前缀判定而不是枚举字段名 —— 枚举表漏一个新字段, 该配方就会走到函数
        # 末尾的「无法判定」, 于是被塞给不具备该能力的硬件。
        if re.fullmatch(r"has_\w+", req):
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
            actual = getattr(caps, attr, 0)
            # compute_capability 之类是字符串 ("9.0"), 直接 "9.0" >= 90 会抛
            # TypeError 打断整轮 plan()。非数值一律判不满足, 由 _caps_failure_note
            # 说明是配方写错而不是硬件不够。
            if not isinstance(actual, (int, float)):
                return False
            actual = actual or 0
            if op == ">=":
                return actual >= val
            if op == "<=":
                return actual <= val
            if op == ">":
                return actual > val
            if op == "<":
                return actual < val
            return actual == val

        return None

    @staticmethod
    def _check_caps_requirement(req: str, caps: HardwareCaps) -> bool:
        """检查单个硬件能力要求是否满足; 判不了就抛, 不默认通过"""
        verdict = OptimizationPlanner._evaluate_caps_requirement(req, caps)
        if verdict is None:
            raise CapsRequirementUnparsable(req)
        return verdict

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
            roofline: 上一轮的 roofline 数据 —— 接受两种形状 (L3 metrics 顶层, 或
                `_roofline_analysis` 那个块), `_axis_field` 两边都读。它可以是**空的**
                (首轮), 也可以非空但**不承重** (分母口径未核), 后者由 `axis_evidence()`
                裁决并原样记进 `plan.axis`。
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
            if hardware:
                hw_caps = detect_hardware_caps(hardware)
            else:
                try:
                    hw_caps = get_caps_for_target("nvidia_gpu")
                except UnknownHardware:
                    # 调用方没给任何硬件信息 ⇒ 认不出。"nvidia_gpu" 这个字面量以前会
                    # 被兜到 A100 的完整能力表上, 于是一轮没做过任何探测的优化按
                    # 108 SM / 2039 GB/s 选配方。
                    hw_caps = unknown_caps("nvidia_gpu")

        # DSL 标准化
        dsl_normalized = self.DSL_ALIASES.get(dsl.lower(), dsl.lower())

        kt = (kernel_type or "").lower()
        axis_ok, axis_note = axis_evidence(roofline)
        # 轴不可用时三个值一起退到「不参与」: `bound_type="unknown"` 让 bound 条件整体
        # 跳过, 利用率归零让 `low_utilization` / `low_bandwidth_util` 那两个条件跳过
        # (它们的判断式是 `util > 0 and util >= thresh`, 0 就是「没有这个读数」)。
        # 留着原值而只跳过 bound, 会把同一个分母除出来的利用率继续用来拒绝配方。
        # 一律走 `_axis_field` 而不是 `roofline.get`: 后者只认顶层形状, 于是「轴可用但
        # 值在块里」的组合会让 axis_ok 与 bound_type 自己打架。
        bound_type = _axis_field(roofline, "bound_type") if axis_ok else "unknown"
        # `or 0` 不是装饰: `_axis_field` 对缺键返回 None, 而下面那两个利用率条件是
        # `util > 0 and ...` —— None 会在比较时抛 TypeError (原代码用 `.get(k, 0)`
        # 兜住了这个)。缺键在这里的语义就是「没有这个读数」, 与 0 同样不参与。
        gflops_util = (_axis_field(roofline, "gflops_utilization_pct") or 0) if axis_ok else 0
        bw_util = (_axis_field(roofline, "bandwidth_utilization_pct") or 0) if axis_ok else 0

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
            axis={"usable": axis_ok, "bound_type": bound_type, "note": axis_note},
        )

    def _check_all_caps(self, reqs: list[str], caps: HardwareCaps) -> tuple[bool, str]:
        """检查所有硬件能力要求，返回 (是否满足, 失败原因)"""
        for req in reqs:
            try:
                ok = self._check_caps_requirement(req, caps)
            except CapsRequirementUnparsable:
                ok = False        # 判不了按不通过处理, 原因由 _caps_failure_note 留下
            if not ok:
                return False, req + self._caps_failure_note(req, caps)
        return True, ""

    @staticmethod
    def _caps_failure_note(req: str, caps: HardwareCaps) -> str:
        """区分「硬件不够」和「配方条件写错」—— 后者不写进拒绝原因就没人会去修。"""
        if OptimizationPlanner._evaluate_caps_requirement(req, caps) is None:
            return "  [条件无法判定 (既不是 has_* 布尔也不是整数比较), 按不满足处理]"
        m = re.match(r"(\w+)\s*(?:>=|<=|>|<|==)\s*\d+", req.strip())
        if m:
            attr = m.group(1)
            if not hasattr(caps, attr):
                return f"  [HardwareCaps 无 {attr} 字段, 条件写错]"
            if not isinstance(getattr(caps, attr), (int, float)):
                return f"  [{attr}={getattr(caps, attr)!r} 非数值, 不能做数值比较]"
        return ""

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
        # 入口守卫以前要的是字面量 "utilization", 而配方实际写的是 `gflops_util < 80%`
        # ⇒ 三条配方的利用率条件从来没被解析出来过 (正则本身是认得的, 守卫先把它挡了)。
        # 放宽到 "util" 之后, 那个「不算这里说的」的语义就交给下面的否定先行断言:
        # `bandwidth_util` 那一族走 `low_bandwidth_util`, 不许被这条一并设上。
        if "util" in cl and "<" in cl:
            m = re.search(r"(?<!bandwidth_)util(?:ization)?\s*<\s*(\d+)", cl)
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
        # bound 条件: 轴不可用 (`unknown`) 时**两个方向都不参与**, 既不再「默认视为
        # compute-bound」, 也不拒绝。原实现是刻意非对称的 —— unknown 放行 compute_bound、
        # 拒绝 memory_bound, 于是一轮没做过任何测量的优化, 会把所有访存向配方挡在门外,
        # 理由是「它不是 memory-bound」, 而这句话的真实依据是「我们不知道」。
        # `AXIS_VALUES` 之外的词一律走这条路, 包括契约测试里的 "mixed" 和拼错的值。
        if bound_type in AXIS_VALUES:
            if conds.get("compute_bound") and bound_type == "memory":
                return False
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
    #: 这一轮绑定轴的裁决 (`{"usable", "bound_type", "note"}`), 由 `plan()` 填。
    #: 它必须是一个**独立字段**而不是从 `roofline` 现算: `to_prompt()` 和 `summary()`
    #: 都要说同一句话, 两处各自解读那个块就会分叉 —— 而缺省 (直接构造的 plan) 按
    #: 「不可用」处理, 于是「没有裁决」不可能被读成「有一个 bound 可以断言」。
    axis: dict = field(default_factory=dict)

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
        if self.axis.get("usable"):
            lines.append(
                f"当前状态: bound={self.axis.get('bound_type', '?')}, "
                f"GFLOPS利用率={(_axis_field(self.roofline, 'gflops_utilization_pct') or 0):.1f}%, "
                f"带宽利用率={(_axis_field(self.roofline, 'bandwidth_utilization_pct') or 0):.1f}%"
            )
        elif self.roofline:
            # 有块但不可用: 这里既不能继续印 `bound=…`, 也不能什么都不印。什么都不印会让
            # 「测过但那个分母不可信」和「首轮还没测过」在 prompt 里长得一模一样; 而印出
            # 那个值就是让模型把一个猜测当结论 —— doc 49 §八 第 17 条实测到的正是这条:
            # 同一个块里 `roofline_bound="unknown"` 与 `bound_type="compute"` 并存,
            # 而只有后者会走进 prompt。
            lines.append(
                f"瓶颈轴: 本轮**不作判定** ({self.axis.get('note', 'plan() 未给出轴向裁决')})"
                f" —— 下面的配方按算子类型与硬件能力选出, 未按 roofline 轴向筛选; "
                f"请自己从代码判断这个核是访存密集还是计算密集。"
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
        # 轴向裁决必须在日志里可见: 「这一轮的配方选择依据了什么」以前只能靠反推
        # selected 集合, 而 off 的时候恰恰是最需要它出现的那一轮。
        axis_info = ("on:" + str(self.axis.get("bound_type"))
                     if self.axis.get("usable")
                     else "off:" + str(self.axis.get("note", "plan() 未给出裁决")))
        return (f"Plan(round={self.round_num}, dsl={self.dsl}, {caps_info}, "
                f"axis=[{axis_info}], "
                f"recipes=[{ids}], n_selected={len(self.selected_recipes)})")
