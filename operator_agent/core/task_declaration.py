"""任务声明的**唯一权威形状**：类型化、版本化、不含自然语言推导。

为什么要这一层（v0.9 / 缺口：同一份事实被解释两次）：

宿主（DSH / CLI / 别的入口）负责理解自然语言，并交出这份声明。此前 `build_task`
会拿着 `declaration["request"]` 再跑一次规则版 `IntentParser`，再用声明覆盖其中
几个字段 —— 于是**没被显式覆盖的字段全部继承关键词猜测**，产生「宿主以为声明的是
A、冻结契约实际是 B」的形状。这个形状已经咬过人：`precision` 被覆盖之后，早先按
FP32 算出的正确性套件成了脏值，只能在末尾再重算一遍兜住。

现在的纪律：

* `parse_declaration` 只做**类型化校验**与**方言收敛**，不读 `request` 的字面内容。
* `request` 仍然是必填字段，但它只进产物与审计，**不参与任何判据推导**。
* 语义上必须由调用方决定的两项（`operator` / `precision`）**没有默认值**：缺了就
  拒绝，不落默认值。这是「不支持就显式拒绝，不要靠猜测」的落点。
* 其余字段有**成文默认值**，并且**出现在 `assumptions` 里**返回给调用方 —— 默认
  可以存在，但必须是可见的，而不是悄悄发生。

关键词解析器没有消失，它退到了 `normalize_declaration`（在 `app/hitl_tools.py`），
职责从「产生契约」变成「**起草一份供宿主确认的提案**」。提案不冻结任何东西。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .schemas import Precision, SourceSpec, TaskComplexity

#: 声明的版本号。形状一变就换号，旧声明按旧规则解释或直接拒绝 —— 不做静默兼容。
DECLARATION_SCHEMA = "task-declaration-v1"

#: 声明里允许出现的键。闭合词表：多一个键就拒绝，避免「宿主以为传进去了」。
ALLOWED_DECLARATION_KEYS = frozenset({
    "schema_version",
    "request",
    "shapes",
    "operator",
    "precision",
    "dsl",
    "hardware",
    "framework",
    "min_speedup",
    "required_caps",
    "complexity",
    "notes",
})

#: 必须由调用方显式给出的语义字段。**没有默认值** —— 这两个直接决定判据。
REQUIRED_SEMANTIC_FIELDS = ("operator", "precision")

#: 成文默认值。每一项都必须在 `assumptions` 里被如实回报。
DOCUMENTED_DEFAULTS = {
    "dsl": "python",
    "hardware": "mock_gpu",
    "framework": "pytorch",
    "min_speedup": 1.0,
    "complexity": "medium",
}

#: **受支持**的精度。注意这不等于 `Precision` 枚举：枚举里有 bf16/int8，但本层没有
#: 可用的正确性 profile，也不在公开工具契约里。两张表分开的后果是 `normalize` 会把
#: bf16 标成可用、宿主照着回填、`prepare` 再拒 —— 所以这里只留一份，工具 schema 与
#: 本模块都从它生成。
SUPPORTED_PRECISIONS = ('fp32', 'fp16')

_PRECISIONS = {name: Precision(name) for name in SUPPORTED_PRECISIONS}
_COMPLEXITIES = {c.value: c for c in TaskComplexity}

#: 硬件 → (compute_units, warp_size)。这是**确定的事实查表**, 不是对请求的解释:
#: 同一个硬件名永远是同一组参数, 与 `request` 写了什么无关。它此前住在关键词解析器
#: 里 (`IntentParser._hardware_cu/_hardware_warp`), 于是「不含关键词解析的构建路径」
#: 会丢掉它 —— 挪到这里之后两条路径共用同一张表。
HARDWARE_PROFILES = {
    "nvidia_gpu": (128, 32),
    "mock_gpu": (64, 32),
    "ascend": (64, 16),
}

#: 未登记的硬件按最小可信值处理: 宁可少报算力, 不要凭空给一个更好看的数。
DEFAULT_HARDWARE_PROFILE = (64, 32)


def hardware_profile(hardware: str) -> tuple:
    """硬件名 → (compute_units, warp_size)。未登记的名字不猜, 走成文默认。"""
    return HARDWARE_PROFILES.get(str(hardware), DEFAULT_HARDWARE_PROFILE)


class DeclarationError(ValueError):
    """声明不合规。带结构化字段，便于宿主据此重填后重试。

    `code` 是稳定标识（宿主可以按它分支）；`missing_fields` / `unsupported_fields`
    是**可行动**的清单 —— 拒绝要让人知道下一步该补什么，而不是只说「你错了」。
    """

    def __init__(self, code: str, detail: str = "", *, missing_fields=(
    ), unsupported_fields=(), assumptions=()):
        self.code = code
        self.detail = detail or code
        self.missing_fields = tuple(missing_fields)
        self.unsupported_fields = tuple(unsupported_fields)
        self.assumptions = tuple(assumptions)
        super().__init__(f"{code}: {self.detail}")

    def as_dict(self) -> dict:
        payload = {"ok": False, "error": self.code, "detail": self.detail}
        if self.missing_fields:
            payload["missing_fields"] = list(self.missing_fields)
        if self.unsupported_fields:
            payload["unsupported_fields"] = list(self.unsupported_fields)
        if self.assumptions:
            payload["assumptions"] = [a.as_dict() for a in self.assumptions]
        return payload


@dataclass(frozen=True)
class Assumption:
    """一个**取了成文默认值**的字段。存在的意义是让默认值可见、可复核。"""

    field: str
    value: Any
    why: str

    def as_dict(self) -> dict:
        return {"field": self.field, "value": self.value, "why": self.why}


@dataclass(frozen=True)
class TaskDeclaration:
    """一份已通过校验的声明。构造只能走 `parse_declaration`。"""

    operator: str
    precision: str
    shapes: tuple
    request: str
    dsl: str
    hardware: str
    framework: str
    min_speedup: float
    required_caps: tuple = ()
    complexity: str = "medium"
    notes: str = ""
    schema_version: str = DECLARATION_SCHEMA

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "request": self.request,
            "shapes": [list(s) for s in self.shapes],
            "operator": self.operator,
            "precision": self.precision,
            "dsl": self.dsl,
            "hardware": self.hardware,
            "framework": self.framework,
            "min_speedup": self.min_speedup,
            "required_caps": list(self.required_caps),
            "complexity": self.complexity,
            "notes": self.notes,
        }


def _shape_rows(raw) -> tuple:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        raise DeclarationError("bad_shapes", f"shapes 必须是整数序列的序列，实际 {raw!r}")
    out = []
    for item in raw:
        if isinstance(item, (str, bytes)) or not isinstance(item, Iterable):
            raise DeclarationError("bad_shapes", f"shapes 里每一项必须是整数序列，实际 {item!r}")
        try:
            dims = tuple(int(d) for d in item)
        except (TypeError, ValueError) as exc:
            raise DeclarationError("bad_shapes", f"{item!r} 读不出整数维度: {exc}") from exc
        if any(d < 0 for d in dims):
            raise DeclarationError("bad_shapes", f"{item!r} 有负维度")
        out.append(dims)
    if not out:
        raise DeclarationError("shapes_required", "shapes 不能为空")
    return tuple(out)


def parse_declaration(raw: dict) -> tuple:
    """声明 → (TaskDeclaration, assumptions)。**不读 `request` 的字面内容。**

    纯函数：同一份声明任何时刻都得到同一份结果。返回的第二项是取了成文默认值的
    字段清单；调用方应当把它回显给宿主，而不是丢掉。
    """
    if not isinstance(raw, dict):
        raise DeclarationError("bad_declaration", "declaration 必须是一个对象")

    unknown = sorted(k for k in raw if k not in ALLOWED_DECLARATION_KEYS)
    if unknown:
        raise DeclarationError(
            "unknown_parameter",
            f"不认识的键 {unknown}；允许的是 {sorted(ALLOWED_DECLARATION_KEYS)}",
            unsupported_fields=unknown)

    version = raw.get("schema_version", DECLARATION_SCHEMA)
    if version != DECLARATION_SCHEMA:
        raise DeclarationError(
            "unsupported_schema_version",
            f"声明版本 {version!r} 不受支持；本层只认 {DECLARATION_SCHEMA!r}")

    request = raw.get("request")
    if not isinstance(request, str) or not request.strip():
        raise DeclarationError(
            "request_required",
            "declaration.request 必须是非空字符串（它只进产物与审计，不参与判据）")
    if not raw.get("shapes"):
        # 不猜：契约拥有输入是这一层的前提；没有声明就没有「评的是哪组输入」。
        raise DeclarationError(
            "shapes_required",
            "必须声明 shapes —— 这一层不接受「输入由调用方临时给」的形态")

    missing = [k for k in REQUIRED_SEMANTIC_FIELDS if not str(raw.get(k, "")).strip()]
    if missing:
        raise DeclarationError(
            "explicit_declaration_required",
            f"这几个字段必须由调用方显式给出，本层不做关键词推断：{missing}",
            missing_fields=missing)

    operator = str(raw["operator"]).strip().lower()
    if operator not in SourceSpec._KNOWN_KERNEL_TYPES:
        raise DeclarationError(
            "unsupported_operator",
            f"operator={operator!r} 不在闭合词表里；已知的是 "
            f"{sorted(SourceSpec._KNOWN_KERNEL_TYPES)}",
            unsupported_fields=[operator])

    precision_raw = str(raw["precision"]).strip().lower()
    if precision_raw not in _PRECISIONS:
        raise DeclarationError(
            "unsupported_precision",
            f"precision={precision_raw!r} 不在闭合词表里；已知的是 {sorted(_PRECISIONS)}",
            unsupported_fields=[precision_raw])

    assumptions = []
    resolved = {}
    for name, default in DOCUMENTED_DEFAULTS.items():
        if name in raw and raw[name] not in (None, ""):
            resolved[name] = raw[name]
        else:
            resolved[name] = default
            assumptions.append(Assumption(
                field=name, value=default,
                why=f"declaration 未给出 {name}，取本层的成文默认值"))

    complexity_raw = str(resolved["complexity"]).strip().lower()
    if complexity_raw not in _COMPLEXITIES:
        raise DeclarationError(
            "unsupported_complexity",
            f"complexity={complexity_raw!r} 不在闭合词表里；已知的是 {sorted(_COMPLEXITIES)}",
            unsupported_fields=[complexity_raw])

    try:
        min_speedup = float(resolved["min_speedup"])
    except (TypeError, ValueError) as exc:
        raise DeclarationError("bad_min_speedup", f"min_speedup 读不出浮点数: {exc}") from exc

    caps = raw.get("required_caps") or ()
    if isinstance(caps, (str, bytes)) or not isinstance(caps, Iterable):
        raise DeclarationError("bad_required_caps", "required_caps 必须是字符串序列")

    declaration = TaskDeclaration(
        operator=operator,
        precision=precision_raw,
        shapes=_shape_rows(raw["shapes"]),
        request=request,
        dsl=str(resolved["dsl"]),
        hardware=str(resolved["hardware"]),
        framework=str(resolved["framework"]),
        min_speedup=min_speedup,
        required_caps=tuple(sorted({str(c) for c in caps})),
        complexity=complexity_raw,
        notes=str(raw.get("notes") or ""),
        schema_version=version,
    )
    return declaration, tuple(assumptions)


def precision_of(declaration: TaskDeclaration) -> Precision:
    """声明里的精度 → `Precision` 枚举。词表已在 `parse_declaration` 收敛过。"""
    return _PRECISIONS[declaration.precision]


def complexity_of(declaration: TaskDeclaration) -> TaskComplexity:
    return _COMPLEXITIES[declaration.complexity]


def with_overrides(declaration: TaskDeclaration, **changes) -> TaskDeclaration:
    """改字段的**唯一**出口：改完仍是一份不可变声明，不会退化成裸 dict。"""
    unknown = sorted(set(changes) - set(DOCUMENTED_DEFAULTS) - set(REQUIRED_SEMANTIC_FIELDS)
                     - {"shapes", "request", "required_caps", "notes"})
    if unknown:
        raise DeclarationError("unknown_override", f"不认识的字段 {unknown}")
    return replace(declaration, **changes)


def field_state(raw: dict) -> dict:
    """给宿主看的体检结果：缺什么、不支持什么。**不抛错**，用于起草与回显。"""
    if not isinstance(raw, dict):
        return {"ok": False, "error": "bad_declaration",
                "detail": "declaration 必须是一个对象"}
    missing = [k for k in ("request", "shapes", "operator", "precision") if not raw.get(k)]
    unknown = sorted(k for k in raw if k not in ALLOWED_DECLARATION_KEYS)
    # 取值也要在这里拦：只报「缺不缺」会放过 bf16/int8 这类**有值但不支持**的提案，
    # 宿主照着回填，prepare 再拒 —— 契约就分家了。
    bad_values = []
    operator = str(raw.get("operator") or "").strip().lower()
    if operator and operator not in SourceSpec._KNOWN_KERNEL_TYPES:
        bad_values.append(operator)
    precision = str(raw.get("precision") or "").strip().lower()
    if precision and precision not in _PRECISIONS:
        bad_values.append(precision)
    return {
        "ok": not (missing or unknown or bad_values),
        "schema_version": DECLARATION_SCHEMA,
        "missing_fields": missing,
        "unsupported_fields": sorted(unknown + bad_values),
        "supported_precisions": list(SUPPORTED_PRECISIONS),
    }