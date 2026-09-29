"""配方库与硬件能力表之间的结构一致性守卫。

这一类缺陷不会让测试变红, 只会让配方**永远选不出来**或让 plan() 直接崩:

  1. 配方写 `requires_caps: has_tma=True`, 而 HardwareCaps 上没有 has_tma 字段
     → getattr 默认 False → 该配方在任何硬件上都不会被选中。R-MATMUL-07 的
     `registers_per_thread >= 255` 就是这么死掉一整条优化的 (字段存在, 但值恒为
     上限 255, 条件恒假)。
  2. 配方写 `requires_caps: compute_capability >= 90`, 而该字段是字符串 "9.0"
     → 数值比较抛 TypeError, 打断整轮 plan()。
  3. 别处 (skill 文档 / 代码注释) 引用 R-MATMUL-12, 而配方库里没有这一条
     → Coder prompt 里每轮都出现一个「关键性能优化」, Planner 永远选不出来。

三者共同点: 静默失效, 没有任何运行期信号。所以只能靠结构性断言守住。
"""

import ast
import dataclasses
import os
import re
import unittest

from operator_agent.hardware_caps import _KNOWN_HARDWARE, HardwareCaps
from operator_agent.optimization_planner import OptimizationPlanner

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPE_MD = os.path.join(REPO_ROOT, "operator_agent", "skills",
                         "optimization_recipes.md")
SKILLS_DIR = os.path.join(REPO_ROOT, "operator_agent", "skills")

_RECIPE_ID = re.compile(r"^###\s+(R-[A-Z]+-\d+)\s*:", re.MULTILINE)
_RECIPE_REF = re.compile(r"\bR-[A-Z]+-\d+\b")
_CAPS_FIELD_NAMES = {f.name for f in dataclasses.fields(HardwareCaps)}


def _recipe_text() -> str:
    with open(RECIPE_MD, encoding="utf-8") as fh:
        return fh.read()


def _recipe_ids() -> set:
    return set(_RECIPE_ID.findall(_recipe_text()))


def _recipe_blocks() -> dict:
    """{recipe_id: 该条配方的正文} —— 按 `### R-XXX-NN:` 切分。"""
    text = _recipe_text()
    ids = list(_RECIPE_ID.finditer(text))
    blocks = {}
    for i, m in enumerate(ids):
        end = ids[i + 1].start() if i + 1 < len(ids) else len(text)
        blocks[m.group(1)] = text[m.start():end]
    return blocks


def _declared_caps(block: str) -> list:
    """取出 `requires_caps` 行, 并用 Planner 自己的解析器切条件。

    必须走 _parse_caps_requirements 而不是自己写正则: 解析器会跳过纯中文描述,
    自己写一套就会测到不存在的行为。
    """
    m = re.search(r"\*\*requires_caps\*\*\s*:\s*(.+)", block)
    if not m:
        return []
    return OptimizationPlanner._parse_caps_requirements(m.group(1))


class TestCapsFieldsExist(unittest.TestCase):
    """条件 1: 配方引用的每个能力字段都必须是 HardwareCaps 上的真实字段。"""

    def test_every_referenced_attr_is_a_real_field(self):
        referenced = {}
        for rid, block in _recipe_blocks().items():
            for req in _declared_caps(block):
                attr = re.match(r"(\w+)", req).group(1)
                referenced.setdefault(attr, []).append(rid)
        missing = {a: rids for a, rids in referenced.items()
                   if a not in _CAPS_FIELD_NAMES}
        self.assertEqual(
            {}, missing,
            f"配方引用了 HardwareCaps 上不存在的字段, 这些配方永远选不出来: {missing}")

    def test_hopper_caps_are_declared_on_the_dataclass(self):
        """R-MATMUL-13/14 与 R-UNIV-05 的判据字段必须存在, 否则是死条件。"""
        for f in ("has_tma", "has_warp_specialization", "has_pdl",
                  "has_fp8_tensor_core"):
            self.assertTrue(hasattr(HardwareCaps(), f), f"HardwareCaps 缺字段 {f}")

    def test_defaults_are_conservative(self):
        """新字段的默认值必须是 False —— 未知硬件不该拿到 Hopper 专属配方。"""
        caps = HardwareCaps()
        for f in ("has_tma", "has_warp_specialization", "has_pdl",
                  "has_fp8_tensor_core"):
            self.assertFalse(getattr(caps, f), f"{f} 默认值应为 False")


class TestCapsCheckNeverRaises(unittest.TestCase):
    """条件 2: 任何已声明的条件 × 任何已知硬件, 都不允许抛异常。"""

    def test_all_declared_requirements_evaluable_on_all_known_hardware(self):
        all_reqs = set()
        for block in _recipe_blocks().values():
            all_reqs.update(_declared_caps(block))
        self.assertTrue(all_reqs, "没解析出任何 requires_caps, 解析器或格式变了")
        for name, caps in _KNOWN_HARDWARE.items():
            for req in sorted(all_reqs):
                try:
                    OptimizationPlanner._check_caps_requirement(req, caps)
                except Exception as exc:            # noqa: BLE001 - 断言的就是"不抛"
                    self.fail(f"[{name}] 检查 {req!r} 抛了 {type(exc).__name__}: {exc}")

    def test_string_valued_numeric_comparison_is_refused_not_fatal(self):
        """实测过的崩溃: compute_capability 是 "9.0", 与 int 比较抛 TypeError。"""
        caps = _KNOWN_HARDWARE["h20"]
        self.assertFalse(
            OptimizationPlanner._check_caps_requirement("compute_capability >= 90", caps))
        # 而且要说清是配方写错, 不是硬件不够 —— 否则没人会去修那行配方
        self.assertIn("非数值", OptimizationPlanner._caps_failure_note(
            "compute_capability >= 90", caps))

    def test_unknown_attr_is_refused_with_explanation(self):
        note = OptimizationPlanner._caps_failure_note("no_such_field >= 1",
                                                      _KNOWN_HARDWARE["h20"])
        self.assertIn("无 no_such_field 字段", note)

    def test_bare_has_prefix_is_not_silently_passed(self):
        """裸名 has_xxx 走前缀判定。

        旧实现是一张硬编码字段名枚举表, 表里没有的 has_xxx 会落到函数末尾的
        「无法解析则默认通过」→ 把配方塞给不具备该能力的硬件。
        """
        h20, a100 = _KNOWN_HARDWARE["h20"], _KNOWN_HARDWARE["a100"]
        self.assertTrue(OptimizationPlanner._check_caps_requirement("has_tma", h20))
        self.assertFalse(OptimizationPlanner._check_caps_requirement("has_tma", a100))
        # 一个压根不存在的 has_ 字段: 必须判不满足, 不能默认放行
        self.assertFalse(
            OptimizationPlanner._check_caps_requirement("has_nonexistent_thing", h20))


class TestRecipeCrossReferences(unittest.TestCase):
    """条件 3: 仓库里出现的每个 R-XXX-NN 都必须在配方库里真实存在。"""

    def test_no_dangling_recipe_id_anywhere_in_repo(self):
        known = _recipe_ids()
        dangling = {}
        for root, dirs, files in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs
                       if d not in {".git", "__pycache__", "triton", "logs",
                                    "node_modules", ".venv"}]
            for fn in files:
                if not fn.endswith((".md", ".py")):
                    continue
                path = os.path.join(root, fn)
                rel = os.path.relpath(path, REPO_ROOT)
                if rel == os.path.relpath(__file__, REPO_ROOT):
                    continue        # 本文件的 docstring 里有示例 ID
                try:
                    with open(path, encoding="utf-8") as fh:
                        body = fh.read()
                except (UnicodeDecodeError, OSError):
                    continue
                for rid in set(_RECIPE_REF.findall(body)) - known:
                    dangling.setdefault(rid, []).append(rel)
        self.assertEqual(
            {}, dangling,
            f"引用了配方库里不存在的配方 ID (Planner 永远选不出来): {dangling}")

    def test_recipe_ids_are_unique(self):
        ids = _RECIPE_ID.findall(_recipe_text())
        dupes = {i for i in ids if ids.count(i) > 1}
        self.assertEqual(set(), dupes, f"配方 ID 重复: {dupes}")


class TestHopperRecipesActuallySelectable(unittest.TestCase):
    """端到端: 新配方在 H20 上要被选出来, 在 A100 上要被拒绝。

    只测 _check_caps_requirement 不够 —— 还要确认 applies_when / applies_to /
    解析环节没把它们在更早的一步就筛掉 (R-MATMUL-12 的悬空引用就是这么产生的)。
    """

    def _plan(self, caps, kernel_type="matmul", bound="memory", gflops_util=30.0):
        """bound 必须用 Verifier 实际产出的**裸词** ("memory"/"compute")。

        写 "memory_bound" 会静默走进 Planner 的「既不是 memory 也不是 compute」
        分支, 于是所有 compute_bound 配方一律放行 —— 测出来的结果毫无意义。

        ⚠️ 那两个 provenance 键不是样板: Planner 从 doc 49 §八 第 17 条那次修正起,
        要求 `roofline_bound` 与 `peak_trusted` 才能确认「这一轮的轴是证据」。缺任何一个,
        整个 roofline 块就被判为不承重、bound 条件**整体不参与筛选** —— 那时这一组用例
        断言的「按轴筛」根本不会发生, 而它们看起来仍然全绿 (只会绿在错误的理由上)。
        键名用 Verifier 平铺到 L3 metrics 顶层的那个 (`gflops_utilization_pct`)。
        """
        planner = OptimizationPlanner()
        plan = planner.plan(kernel_type=kernel_type, dsl="triton", hw_caps=caps,
                            roofline={"bound_type": bound,
                                      "roofline_bound": bound,
                                      "peak_trusted": True,
                                      "gflops_utilization_pct": gflops_util})
        rejected = {r.id: why for r, why in plan.rejected_recipes}
        return {r.id for r in plan.selected_recipes}, rejected

    def test_tma_recipe_selected_on_h20_rejected_on_a100(self):
        sel_h20, _ = self._plan(_KNOWN_HARDWARE["h20"])
        sel_a100, rej_a100 = self._plan(_KNOWN_HARDWARE["a100"])
        self.assertIn("R-MATMUL-13", sel_h20, "TMA 配方在 H20 上没被选出来")
        self.assertNotIn("R-MATMUL-13", sel_a100, "A100 无 TMA, 不该拿到这条")
        self.assertIn("has_tma", rej_a100["R-MATMUL-13"])

    def test_compute_bound_recipe_rejected_on_memory_bound_roofline(self):
        """H20 是 memory-bound (ridge≈37.5), 不该拿到 compute_bound 专属配方。

        这条以前是反的: R-MATMUL-07 的条件恒假 → 永远选不出; 修好条件后又要确认
        它确实被 bound 类型挡住, 否则等于把 H100 的取向套到 H20 上。
        """
        sel, rej = self._plan(_KNOWN_HARDWARE["h20"], bound="memory")
        for rid in ("R-MATMUL-07", "R-MATMUL-14"):
            self.assertNotIn(rid, sel, f"{rid} 是 compute_bound 配方, 不该进 memory-bound 计划")
            self.assertIn("conditions not met", rej[rid])

    def test_warp_spec_recipe_selected_when_compute_bound(self):
        sel, _ = self._plan(_KNOWN_HARDWARE["h20"], bound="compute")
        self.assertIn("R-MATMUL-14", sel)

    def test_warp_spec_rejected_on_a100_even_when_compute_bound(self):
        """caps 门要独立于条件门生效: A100 无 Hopper autoWS。"""
        sel, rej = self._plan(_KNOWN_HARDWARE["a100"], bound="compute")
        self.assertNotIn("R-MATMUL-14", sel)
        self.assertIn("has_warp_specialization", rej["R-MATMUL-14"])

    def test_persistent_kernel_recipe_is_universal(self):
        for name in ("h20", "a100", "3090"):
            sel, _ = self._plan(_KNOWN_HARDWARE[name])
            self.assertIn("R-UNIV-04", sel, f"{name} 上 persistent kernel 配方未被选中")

    def test_pdl_recipe_gated_by_caps(self):
        sel_h20, _ = self._plan(_KNOWN_HARDWARE["h20"])
        sel_a100, _ = self._plan(_KNOWN_HARDWARE["a100"])
        self.assertIn("R-UNIV-05", sel_h20)
        self.assertNotIn("R-UNIV-05", sel_a100)


class TestBoundTypeVocabularyContract(unittest.TestCase):
    """bound_type 的取值集合必须被**所有**消费端认识。

    实测过的缺陷: Verifier 产出裸词 "memory"/"compute", 而 triton_opt_rules 按
    "memory_bound"/"compute_bound" 比较 —— 两个分支都进不去, 一路落到 else 的
    "mixed", bound 感知从上线起就没生效过。这类拼法漂移不会报错, 只会静默降级。

    取值集合用 AST 从 verifier 源码里现取, 不写死在测试里: 将来改了产出端,
    这个测试会跟着变, 从而逼着消费端一起改。
    """

    _BOUND_LIKE = re.compile(r"^(memory|compute|unknown|mixed)([_-]bound)?$")

    @classmethod
    def setUpClass(cls):
        path = os.path.join(REPO_ROOT, "operator_agent", "verifier.py")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_roofline_analysis"),
                  None)
        if fn is None:
            raise RuntimeError("verifier._roofline_analysis 不见了, 契约测试失去锚点")
        cls.emitted = {n.value for n in ast.walk(fn)
                       if isinstance(n, ast.Constant) and isinstance(n.value, str)
                       and cls._BOUND_LIKE.match(n.value)}

    def test_verifier_emits_a_known_vocabulary(self):
        self.assertTrue(self.emitted, "没从 verifier 里提取到任何 bound_type 字面量")
        self.assertIn("memory", self.emitted)
        self.assertIn("compute", self.emitted)

    def test_opt_rules_recognizes_every_emitted_value(self):
        from operator_agent.triton_opt_rules import TritonOptimizationRules, normalize_bound_type
        from operator_agent.schemas import TilingStrategy
        tiling = TilingStrategy(64, 64, 32)
        # 断言必须盯住**策略文本**而不是 bound 这个词: else 兜底分支会把传入的
        # bound 插进 reason 里, 于是「走错分支」在字面上看不出来 (实测漏过一次)。
        strategy = {"compute": "BLOCK_M/BLOCK_N", "memory": "BLOCK_K"}
        for bound in sorted(self.emitted):
            self.assertEqual(bound, normalize_bound_type(bound),
                             f"normalize_bound_type 没认出 verifier 产出的 {bound!r}")
            if bound not in strategy:
                continue        # unknown 落到 mixed 分支是设计意图
            _, reason = TritonOptimizationRules.recommend_tiling(
                tiling, bound, hw_name="NVIDIA H20")
            self.assertIn(strategy[bound], reason,
                          f"bound_type={bound!r} 落到了兜底分支, 没给出对应策略 "
                          f"(期望提到 {strategy[bound]}, 实际 reason={reason!r})")

    def test_opt_rules_mixed_branch_is_reachable_only_for_mixed(self):
        """兜底分支的 reason 不许冒充 compute/memory 建议。"""
        from operator_agent.triton_opt_rules import TritonOptimizationRules
        from operator_agent.schemas import TilingStrategy
        for bound in ("memory", "compute"):
            _, reason = TritonOptimizationRules.recommend_tiling(
                TilingStrategy(64, 64, 32), bound, hw_name="NVIDIA H20")
            self.assertNotIn("均衡调整", reason, f"{bound} 不该走 mixed 兜底")
        _, reason = TritonOptimizationRules.recommend_tiling(
            TilingStrategy(64, 64, 32), "unknown", hw_name="NVIDIA H20")
        self.assertIn("均衡调整", reason)

    def test_opt_rules_accepts_the_underscore_spelling_too(self):
        """两套拼法都要认 —— 消费端归一化, 不依赖产出端只用一种写法。"""
        from operator_agent.triton_opt_rules import normalize_bound_type
        for spelled in ("memory", "memory_bound", "memory-bound", "MEMORY_BOUND"):
            self.assertEqual("memory", normalize_bound_type(spelled))
        self.assertEqual("compute", normalize_bound_type("compute_bound"))
        self.assertEqual("unknown", normalize_bound_type(""))

    def test_hints_never_recommend_a_nonexistent_triton_api(self):
        """`tl.cache` 在 Triton 3.8 里根本不存在 (✅ language/core.py 无此导出)。

        提示词里出现不存在的 API, Coder 会照抄, 然后 NameError 烧掉一轮 Fixer。
        同理「fp16 累加器」是精度 bug 不是优化, 与 R-UNIV-03 直接冲突。
        """
        from operator_agent.triton_opt_rules import TritonOptimizationRules
        from operator_agent.schemas import TilingStrategy
        for bound in ("memory", "compute"):
            hints = TritonOptimizationRules.generate_optimization_hints(
                bound, current_gflops=10.0, target_gflops=100.0,
                tiling=TilingStrategy(64, 64, 32))
            joined = " ".join(hints)
            self.assertNotIn("tl.cache", joined)
            self.assertNotIn("float16 累加", joined)
            self.assertTrue(hints, f"{bound} 没产出任何提示")

    def test_hints_are_bound_specific(self):
        """hints 的分支同样会因拼法漂移退化成兜底那句泛泛建议。

        只断言「hints 非空」挡不住这种退化 —— 兜底分支也非空。
        """
        from operator_agent.triton_opt_rules import TritonOptimizationRules
        from operator_agent.schemas import TilingStrategy
        tiling = TilingStrategy(64, 64, 32)
        compute_hints = " ".join(TritonOptimizationRules.generate_optimization_hints(
            "compute", 10.0, 100.0, tiling))
        memory_hints = " ".join(TritonOptimizationRules.generate_optimization_hints(
            "memory", 10.0, 100.0, tiling))
        self.assertIn("Tensor Core", compute_hints)
        self.assertIn("BLOCK_K", memory_hints)
        self.assertIn("coalesced", memory_hints)
        # 兜底分支的特征句不该出现在已识别的 bound 上
        for h in (compute_hints, memory_hints):
            self.assertNotIn("检查内存访问模式和计算效率", h)

    def test_planner_uses_the_same_vocabulary_as_verifier(self):
        """Planner 的 memory_bound 条件必须按裸词判, 否则配方全被误杀/误放。"""
        from operator_agent.optimization_planner import OptimizationPlanner, Recipe
        from operator_agent.hardware_caps import _KNOWN_HARDWARE

        def _recipe(cond: str) -> Recipe:
            r = Recipe(id=f"t-{cond}", name="t", applies_to=["matmul"], applies_when=cond)
            r._conditions = OptimizationPlanner._parse_conditions(cond)
            return r

        r_mem, r_cmp = _recipe("memory_bound"), _recipe("compute_bound")
        args = dict(kernel_type="matmul", gflops_util=0.0, bw_util=0.0,
                    precision="fp32", round_num=1, caps=_KNOWN_HARDWARE["h20"])
        check = OptimizationPlanner._check_conditions
        self.assertTrue(check(r_mem, bound_type="memory", **args))
        self.assertFalse(check(r_mem, bound_type="compute", **args))
        self.assertTrue(check(r_cmp, bound_type="compute", **args))
        self.assertFalse(check(r_cmp, bound_type="memory", **args))


class TestSkillDocsReferenceOnlyRealRecipes(unittest.TestCase):
    """skills/ 下的其它文档若点名推荐某条配方, 也必须是真的。"""

    def test_coder_skill_references_resolve(self):
        known = _recipe_ids()
        for fn in os.listdir(SKILLS_DIR):
            if not fn.endswith(".md") or fn == os.path.basename(RECIPE_MD):
                continue
            with open(os.path.join(SKILLS_DIR, fn), encoding="utf-8") as fh:
                body = fh.read()
            missing = set(_RECIPE_REF.findall(body)) - known
            self.assertEqual(set(), missing,
                             f"{fn} 引用了不存在的配方: {sorted(missing)}")


class TestRecipeFileStructureIntact(unittest.TestCase):
    """编辑配方文件时容易顺手吃掉一个 `## ` 分节标题 (真实发生过一次)。"""

    EXPECTED_SECTIONS = [
        "Matmul / GEMM 优化配方", "通用 (算子无关) 优化配方",
        "Softmax 优化配方", "LayerNorm 优化配方",
        "Element-wise 优化配方", "Reduction 优化配方", "新硬件适配指南",
    ]

    def test_all_top_level_sections_present_and_ordered(self):
        with open(RECIPE_MD, encoding="utf-8") as fh:
            found = [m.group(1).strip()
                     for m in re.finditer(r"^##\s+(.+)$", fh.read(), re.MULTILINE)]
        self.assertEqual(self.EXPECTED_SECTIONS, found)

    def test_every_recipe_has_the_five_required_fields(self):
        required = ("applies_to", "applies_when", "priority",
                    "requires_caps", "expected_impact")
        for rid, block in _recipe_blocks().items():
            for field in required:
                self.assertIn(f"**{field}**", block, f"{rid} 缺字段 {field}")

    def test_recipe_file_is_valid_utf8_and_no_tabs(self):
        with open(RECIPE_MD, encoding="utf-8") as fh:
            text = fh.read()
        self.assertNotIn("\t", text, "配方文件里出现 tab, Markdown 缩进会错乱")


if __name__ == "__main__":
    unittest.main()
