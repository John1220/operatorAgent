"""架构守卫：**同一份事实只允许被解释一次**。

这里钉的不是某个函数的输出，而是「某条路径**不存在**」—— 这类守卫必须写成
对源码/结构的断言，而不是对行为的断言：行为断言拦不住「多解释一次」被重新加回来，
因为重新加回来的那次解释在多数输入上恰好给出同一个答案。

背景（v0.9 任务声明收敛）：`build_task` 此前会拿 `declaration["request"]` 再跑一次
规则版 `IntentParser`，再用声明覆盖其中几项。没被覆盖的字段就继承了关键词猜测，
于是「宿主以为声明的是 A、冻结契约实际是 B」。这个形状已经咬过人（`precision` 被
覆盖之后，早先按 FP32 算出的正确性套件成了脏值，只能在末尾再算一遍兜住）。

现在：`build_task` 是纯映射，`request` 只进产物与审计；关键词解析退到
`normalize_declaration` / `declaration_from_request`（**宿主侧**），它们只产出提案。
"""
import ast
import io
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

from operator_agent.app import hitl_tools as ht                             # noqa: E402
from operator_agent.core.contract import freeze_from_task                  # noqa: E402
from operator_agent.core.task_declaration import (ALLOWED_DECLARATION_KEYS,  # noqa: E402
                                                 DECLARATION_SCHEMA,
                                                 DeclarationError,
                                                 parse_declaration)

CONDUCTOR = ROOT / "operator_agent" / "app" / "conductor.py"
HITL = ROOT / "operator_agent" / "app" / "hitl_tools.py"


def _function_tree(path: Path, name: str) -> ast.AST:
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{path.name} 里找不到函数 {name}")


def _imported_names(node: ast.AST) -> set:
    """函数体里出现的 import 目标（含局部 import）与属性名。"""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom):
            names.update(alias.name for alias in child.names)
            names.update(alias.asname for alias in child.names if alias.asname)
        elif isinstance(child, ast.Import):
            names.update(alias.name for alias in child.names)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
        elif isinstance(child, ast.Name):
            names.add(child.id)
    return names


class TestTheFrozenPathDoesNotInterpretNaturalLanguage:
    def test_build_task_never_reaches_the_intent_parser(self):
        """`build_task` 里不出现 `IntentParser` —— 判据层不该有第二个语义来源。

        这是本类里唯一一条**结构**断言，也是最重要的一条：它拦的是「又加回一次解释」，
        而不是「这次解释给出了不同的值」。
        """
        used = _imported_names(_function_tree(HITL, "build_task"))
        assert "IntentParser" not in used, (
            "build_task 里又出现了 IntentParser —— 冻结路径不该再解释自然语言；"
            "要起草请走 normalize_declaration（宿主侧），别把推断塞回判据层")
        assert "draft_from_request" not in used, "同上"

    def test_the_parser_still_lives_but_only_as_a_drafter(self):
        """解析器没有消失，它退到了提案侧 —— 这条防止有人把它当死代码删掉。

        删掉它不会让任何用例变红（宿主可以直接给显式字段），但会让 CLI / 实验脚本
        失去起草能力，而那是另一个能力的静默丧失。
        """
        drafting = _imported_names(_function_tree(HITL, "normalize_declaration"))
        assert "draft_from_request" in drafting, (
            "提案侧不再起草了 —— 解析器被误当死代码删掉的形状")


class TestTheRequestTextCannotMoveTheContract:
    """改自然语言不许改判据。这是「两次需求理解」的直接反证。"""

    BASE = {"operator": "layernorm", "precision": "fp32", "shapes": [[8, 32]],
            "hardware": "mock_gpu", "dsl": "python"}

    def _hash(self, **changes):
        declaration = dict(self.BASE, **changes)
        return freeze_from_task(ht.build_task(declaration)).hash

    def test_rewording_the_request_leaves_the_contract_identical(self):
        first = self._hash(request="layernorm fp32")
        second = self._hash(request="帮我写一个 layernorm，fp32 精度，8 行 32 列")
        assert first == second, (
            "request 的措辞改变了契约哈希 —— 说明判据层又在读自然语言了")

    def test_the_request_is_still_required_and_still_recorded(self):
        """`request` 不参与判据，但**仍然必填**：它是审计字段，不是可选装饰。"""
        with pytest.raises(ht.ToolError) as caught:
            ht.build_task({k: v for k, v in self.BASE.items()})
        assert caught.value.code == "request_required"

    def test_an_unresolved_semantic_marker_still_changes_the_task(self):
        """检测器是活着的：认不出的语义必须**看得见**，不是被静默丢弃。

        这条与上面两条不矛盾 —— 前者说「措辞不该有影响」，这条说「**声明的语义范围**
        有影响」。检测器不推断任何值，只标记「这里有未解析的语义」并让下游 fail closed。
        """
        plain = ht.build_task(dict(self.BASE, request="layernorm fp32"))
        marked = ht.build_task(dict(self.BASE, request="layernorm fp32, eps=1e-5 affine=True"))
        assert not plain.source.semantic_parameters.get("unresolved_request")
        assert marked.source.semantic_parameters.get("unresolved_request")


class TestSemanticFieldsMustBeExplicit:
    """`operator` / `precision` 没有默认值：缺了就拒绝，不猜。"""

    def test_missing_operator_is_refused_with_an_actionable_list(self):
        with pytest.raises(ht.ToolError) as caught:
            ht.build_task({"request": "写个 layernorm", "precision": "fp32",
                           "shapes": [[8, 32]]})
        assert caught.value.code == "explicit_declaration_required"
        assert "operator" in caught.value.detail

    def test_an_operator_outside_the_closed_vocabulary_is_refused(self):
        with pytest.raises(ht.ToolError) as caught:
            ht.build_task({"request": "写个 x", "operator": "not_an_operator",
                           "precision": "fp32", "shapes": [[8, 32]]})
        assert caught.value.code == "unsupported_operator"

    def test_documented_defaults_are_reported_rather_than_silently_applied(self):
        """默认值可以存在，但必须**可见** —— 静默默认是「宿主以为 A、契约是 B」的另一条通道。"""
        declaration = {"request": "layernorm fp32", "operator": "layernorm",
                       "precision": "fp32", "shapes": [[8, 32]]}
        _, assumptions = parse_declaration(declaration)
        fields = {a.field for a in assumptions}
        assert {"dsl", "hardware", "framework", "min_speedup", "complexity"} <= fields
        prepared = ht.prepare(declaration, Path(tempfile.mkdtemp(prefix="absence-")))
        assert {row["field"] for row in prepared["assumptions"]} >= {"dsl", "hardware"}


class TestOneAuthorityForTheDeclarationVocabulary:
    def test_the_tool_whitelist_is_the_core_whitelist(self):
        """工具层的白名单不是一个副本，而是同一个对象。"""
        assert ht.ALLOWED_KEYS is ALLOWED_DECLARATION_KEYS

    def test_the_hardware_table_has_one_implementation(self):
        """硬件 → (CU, warp) 的映射只有一处；解析器那一份必须转调过去。"""
        from operator_agent.app import intent_parser
        source = io.open(intent_parser.__file__, encoding="utf-8").read()
        assert "hardware_profile" in source
        assert 'cu_map = {"nvidia_gpu"' not in source, (
            "解析器里又抄了一份硬件映射表 —— 两份表迟早会漂")

    def test_the_declaration_schema_is_versioned(self):
        assert DECLARATION_SCHEMA == "task-declaration-v1"
        with pytest.raises(DeclarationError) as caught:
            parse_declaration({"schema_version": "task-declaration-v0",
                               "request": "x", "operator": "relu",
                               "precision": "fp32", "shapes": [[2, 2]]})
        assert caught.value.code == "unsupported_schema_version"


class TestTheConductorIsNotReachableFromTheDeclaredPath:
    """`build_task` 不许把活儿委派给 Conductor —— 那是第二条生成路径。"""

    def test_build_task_does_not_construct_a_conductor(self):
        used = _imported_names(_function_tree(HITL, "build_task"))
        assert "Conductor" not in used