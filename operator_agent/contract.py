"""契约冻结 (frozen contract) —— 无人值守下的「尺子不可改」物理收口

设计出处: `docs/design/autonomous-operator-agent-design.md` §3 / §10.5-5 /
§13.1 / §13.4。核心句:

    Agent 不能改自己的尺子 —— 否则就是 reward hacking: 放宽容差、换基线、改指标。
    契约在 CONTRACT_LINT 通过后写入只读文件并哈希; 评测器每轮校验哈希。

与既有机制的分工 (不是重复):
  * `anticheat`     —— 挡**生成的代码**回退到库算子
  * `interface_contract` —— 挡**签名**与调用约定不符
  * **本模块**      —— 挡**判据本身** (容差 / 基线定义 / 目标指标 / 数值语义) 被改写

契约哈希是评测器的拒评条件 (`rec.contract_hash != contract.hash → 拒评`),
所以它必须是**内容哈希**而不是文件 mtime: 同内容两份拷贝必须同哈希,
改一个字必须变哈希。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


class ContractError(ValueError):
    """契约不可判定 / 哈希不匹配 —— 失败关闭, 不猜"""


# 数值语义: 加速比差 2x 的那一族 (doc 44 / 设计 §3.2)
_DTYPE_KEYS = ("tf32", "accum", "allow_reduced_precision", "deterministic")


@dataclass(frozen=True)
class DTypeSemantics:
    """显式钉死的数值语义。缺项 = 契约不完整, 拒绝冻结。"""
    dtype: str                      # fp32 | fp16 | bf16 | int8 | ...
    tf32: bool = False              # fp32 matmul 是否允许 TF32 通路
    accum: str = "fp32"             # 累加精度: fp32 | fp16 | fp64
    allow_reduced_precision: bool = False
    deterministic: bool = True      # 是否允许 split-K 等非确定性

    def validate(self) -> list:
        errs = []
        if not self.dtype:
            errs.append("dtype_semantics.dtype 不能为空")
        if self.accum not in ("fp32", "fp16", "fp64", "tf32"):
            errs.append(f"accum={self.accum!r} 不在允许集合")
        return errs


@dataclass(frozen=True)
class BaselineDef:
    """基线的**定义方式**, 不是一张查表数字。

    「谁生产语义, 谁定义对手」(设计 §13.2 ⑥): 基线必须与候选在同一数值语义下。
    """
    name: str                       # pytorch_eager | cuBLAS_fp16 | reference_source | ...
    same_device: bool = True        # 同卡
    same_window: bool = True        # 交错时间窗
    same_dtype_semantics: bool = True
    source: str = ""                # 出处/说明, 只作审计

    def validate(self) -> list:
        errs = []
        if not self.name:
            errs.append("baseline.name 不能为空")
        if not (self.same_device and self.same_window and self.same_dtype_semantics):
            errs.append("基线必须同卡、同窗、同数值语义 —— 否则加速比无指称对象")
        return errs


@dataclass(frozen=True)
class ShapeDist:
    """形状分布 + 权重。单点 shape 要显式写成 weight=1 的一条。"""
    entries: tuple = ()             # ((shape_tuple, weight), ...)

    def validate(self) -> list:
        errs = []
        if not self.entries:
            errs.append("形状分布不能为空 (单点也要写成 weight=1 的一条)")
        total = 0.0
        for shape, w in self.entries:
            if not shape or not isinstance(w, (int, float)) or w <= 0:
                errs.append(f"非法形状项: shape={shape!r} weight={w!r}")
            total += float(w)
        if total <= 0:
            errs.append("形状权重之和必须 > 0")
        return errs


@dataclass(frozen=True)
class FrozenContract:
    """冻结后的契约。**只读**; `hash` 由内容决定, 改一个字段就变。"""
    kernel_type: str
    dtype_semantics: DTypeSemantics
    baseline: BaselineDef
    shapes: ShapeDist
    max_error: float
    min_speedup: float
    target_metric: str = "speedup_vs_baseline"   # 吞吐 / 延迟 / 显存 / 加速比
    required_caps: tuple = ()                    # Cap.value 字符串
    notes: str = ""
    #: 规范化 JSON 的 sha256 —— 不存进 as_dict 的计算路径, 单独算
    hash: str = field(default="", compare=False)

    def content_dict(self) -> dict:
        """参与哈希的内容 (排除 hash 字段本身与 notes 这类说明性文字)。"""
        return {
            "kernel_type": self.kernel_type,
            "dtype_semantics": asdict(self.dtype_semantics),
            "baseline": asdict(self.baseline),
            "shapes": [(list(s), float(w)) for s, w in self.shapes.entries],
            "max_error": float(self.max_error),
            "min_speedup": float(self.min_speedup),
            "target_metric": self.target_metric,
            "required_caps": sorted(self.required_caps),
        }

    def compute_hash(self) -> str:
        blob = json.dumps(self.content_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        d = self.content_dict()
        d["hash"] = self.hash or self.compute_hash()
        d["notes"] = self.notes
        return d


def freeze(kernel_type: str,
           dtype_semantics: DTypeSemantics,
           baseline: BaselineDef,
           shapes: ShapeDist,
           max_error: float,
           min_speedup: float,
           target_metric: str = "speedup_vs_baseline",
           required_caps=(),
           notes: str = "") -> FrozenContract:
    """CONTRACT_LINT + 冻结。任何 validate 失败都抛 ContractError —— **失败关闭**。

    「不可判定就交还给人」(设计 §10.1 REJECT): 这里不猜默认值。缺数值语义的契约
    会让后面所有加速比都可能是 2x 假象 (设计 §3.2 原句)。
    """
    if not (kernel_type or "").strip():
        raise ContractError("kernel_type 不能为空")
    if not isinstance(max_error, (int, float)) or isinstance(max_error, bool) or max_error <= 0:
        raise ContractError(f"max_error 必须为正数: {max_error!r}")
    if not isinstance(min_speedup, (int, float)) or isinstance(min_speedup, bool) or min_speedup <= 0:
        raise ContractError(f"min_speedup 必须为正数: {min_speedup!r}")

    errs = []
    errs += [f"dtype_semantics: {e}" for e in dtype_semantics.validate()]
    errs += [f"baseline: {e}" for e in baseline.validate()]
    errs += [f"shapes: {e}" for e in shapes.validate()]
    if errs:
        raise ContractError("契约不可判定: " + "; ".join(errs))

    c = FrozenContract(
        kernel_type=kernel_type,
        dtype_semantics=dtype_semantics,
        baseline=baseline,
        shapes=shapes,
        max_error=float(max_error),
        min_speedup=float(min_speedup),
        target_metric=target_metric,
        required_caps=tuple(sorted(required_caps or ())),
        notes=notes,
    )
    # 冻结后立刻写入 hash: 之后 as_dict() 用的就是这份
    return FrozenContract(
        kernel_type=c.kernel_type,
        dtype_semantics=c.dtype_semantics,
        baseline=c.baseline,
        shapes=c.shapes,
        max_error=c.max_error,
        min_speedup=c.min_speedup,
        target_metric=c.target_metric,
        required_caps=c.required_caps,
        notes=c.notes,
        hash=c.compute_hash(),
    )


def assert_hash(contract: FrozenContract, expected_hash: str) -> None:
    """评测器每轮校验。不匹配 = 拒评 —— 改不了契约就改不了判据。"""
    actual = contract.hash or contract.compute_hash()
    if not expected_hash or actual != expected_hash:
        raise ContractError(
            f"契约哈希不匹配: 期望 {str(expected_hash)[:12]}… 实际 {actual[:12]}… "
            f"—— 判据被改写过, 拒绝评测")


def load_dict(payload: Mapping[str, Any]) -> FrozenContract:
    """从 `as_dict()` 的产物重建 (断点续跑 / 离线复盘用)。"""
    try:
        ds = DTypeSemantics(**payload["dtype_semantics"])
        # dataclass 字段名与 asdict 一致
        bd = BaselineDef(**payload["baseline"])
        shapes = ShapeDist(entries=tuple(
            (tuple(s), float(w)) for s, w in payload["shapes"]))
        return freeze(
            kernel_type=payload["kernel_type"],
            dtype_semantics=ds,
            baseline=bd,
            shapes=shapes,
            max_error=payload["max_error"],
            min_speedup=payload["min_speedup"],
            target_metric=payload.get("target_metric", "speedup_vs_baseline"),
            required_caps=payload.get("required_caps") or (),
            notes=payload.get("notes", ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"无法从载荷重建契约: {exc}") from exc
