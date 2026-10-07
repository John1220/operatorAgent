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
import math
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


def case_key(inputs: Any) -> str:
    """一次评测的 **case 身份** = 输入集的形状签名（`A=64x64|B=64x64`）

    §6.2 条款 2 要求「候选须在**全部**形状上通过」，条款 4 要求 `required_cases` 进契约
    hash；本仓要用它判「所有必判 case 正确」就必须先能回答**本次评的是哪一条 case**。
    这个函数就是那个身份：按输入名排序 ⇒ 同一组输入任何时候都是同一个 key，产物里可离线
    复算、可逐字比对。

    为什么不是 `ShapeDist` 里那个 kernel 级 shape：设计 §6.2 的 case 是 `BenchmarkCase`
    （带**具体输入**的那一条），而今天 `ShapeDist` 只参与校验与哈希（§3.4 的欠账），实测
    还有「契约声明 256x256x256、本次评的是 64x64」的产物 —— 拿它当身份会让「评过没有」
    永远判不了。两个空间的合一是 `BenchmarkCase` 进契约那天的事（CHECKLOG I92）。

    三态：形状取不到（值没有 `.shape`）⇒ 那一位写 `?`（**不是** 0，也不是省略）；
    `inputs` 不是映射或为空 ⇒ 返回空串 = **没有 case**（调用方按「没判过」处理，
    **不许**拿空串去匹配 `required_cases`）。
    """
    if not isinstance(inputs, Mapping):
        return ""
    parts = []
    for name in sorted(inputs):
        shape: tuple = ()
        try:
            shape = tuple(int(d) for d in (getattr(inputs[name], "shape", None) or ()))
        except (TypeError, ValueError):
            shape = ()
        parts.append(f"{name}=" + ("x".join(str(d) for d in shape) if shape else "?"))
    return "|".join(parts)


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
    #: §6.2 条款 2/4：必须参与判决的 case（`case_key` 字符串，缺一不可）。
    #: 空 tuple = **契约没声明**（老契约）⇒ 判据按「未确立」处理，不是「都通过了」。
    required_cases: tuple = ()
    #: §6.2 条款 3：`((case_key, 允许的回退比例), ...)` —— 逐 case 硬回退上限。
    #: 用 tuple-of-pairs 而不是 dict：与 `ShapeDist.entries` 同款（冻结对象不可变、可哈希）。
    max_regression: tuple = ()
    notes: str = ""
    #: 规范化 JSON 的 sha256 —— 不存进 as_dict 的计算路径, 单独算
    hash: str = field(default="", compare=False)

    def content_dict(self) -> dict:
        """参与哈希的内容 (排除 hash 字段本身与 notes 这类说明性文字)。"""
        d = {
            "kernel_type": self.kernel_type,
            "dtype_semantics": asdict(self.dtype_semantics),
            "baseline": asdict(self.baseline),
            "shapes": [(list(s), float(w)) for s, w in self.shapes.entries],
            "max_error": float(self.max_error),
            "min_speedup": float(self.min_speedup),
            "target_metric": self.target_metric,
            "required_caps": sorted(self.required_caps),
        }
        # §6.2 条款 4：这两项**全部进契约 hash** —— 判据变成可改的常量就等于没有判据。
        # 但**空的声明不进**：空 = 没声明（判据面按「未确立」走），与本片之前冻结的契约
        # 是同一份判据内容 ⇒ 同一个 hash（那样旧载荷还能被 `load_dict` 复盘，而
        # 「有↔无」「值变了」照旧必须改 hash）。见 CHECKLOG I92。
        if self.required_cases:
            d["required_cases"] = list(self.required_cases)
        if self.max_regression:
            d["max_regression"] = [[str(k), float(v)] for k, v in self.max_regression]
        return d

    def regression_limit(self, case: str) -> Optional[float]:
        """这条 case 声明的允许回退比例；没为它声明 ⇒ `None`（**不是** 0）

        「没声明上限」与「声明了 0（一点也不许退）」是两件事，判据面必须分得开。
        """
        for key, value in self.max_regression:
            if str(key) == str(case):
                return float(value)
        return None

    def compute_hash(self) -> str:
        blob = json.dumps(self.content_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        d = self.content_dict()
        d["hash"] = self.hash or self.compute_hash()
        d["notes"] = self.notes
        return d


def _regression_pairs(raw) -> tuple:
    """任务侧/载荷侧的声明可能是 dict（JSON 原生）或 `(case, 值)` 序列 —— 统一成二元组序列。

    值**不在这里**转 float：转坏了就成了另一个异常类型，而「读不出来的声明」应当由
    同一处 CONTRACT_LINT 以 `ContractError` 拒绝（失败关闭）。
    """
    if not raw:
        return ()
    items = raw.items() if isinstance(raw, Mapping) else raw
    return tuple((str(k), v) for k, v in items)


def _lint_case_declarations(required_cases, max_regression) -> list:
    """§6.2 条款 2/3 的两项声明进契约前的 CONTRACT_LINT。

    失败关闭：声明了但读不出来（空串、非数值、值 ≤ -1）就拒绝冻结 —— 一个读不出来的
    「硬回退上限」会让判据面变成一句空话，而契约是没人值守时唯一不能含糊的东西。
    值必须 > -1：`1 - limit` 是「允许的最慢加速比」，limit ≤ -1 时下界 ≤ 0，判据形同虚设。
    """
    errs = []
    for case in required_cases or ():
        if not isinstance(case, str) or not case.strip():
            errs.append(f"required_cases 里有一条不是非空字符串: {case!r}")
    for item in max_regression or ():
        try:
            key, value = item
        except (TypeError, ValueError):
            errs.append(f"max_regression 的项必须是 (case, 数值) 二元组: {item!r}")
            continue
        if not isinstance(key, str) or not key.strip():
            errs.append(f"max_regression 的 case 不是非空字符串: {key!r}")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errs.append(f"max_regression[{key!r}] 必须是数值: {value!r}")
        elif not math.isfinite(float(value)) or float(value) <= -1.0:
            errs.append(f"max_regression[{key!r}] 必须 > -1 且有限: {value!r}")
    return errs


def freeze(kernel_type: str,
           dtype_semantics: DTypeSemantics,
           baseline: BaselineDef,
           shapes: ShapeDist,
           max_error: float,
           min_speedup: float,
           target_metric: str = "speedup_vs_baseline",
           required_caps=(),
           notes: str = "",
           required_cases=(),
           max_regression=()) -> FrozenContract:
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
    errs += _lint_case_declarations(required_cases, max_regression)
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
        # 顺序无语义 ⇒ 规范化成「排序去重」：同一份判据不该有两种 hash（同内容两份拷贝
        # 必须同哈希是契约哈希的第一条要求, 见模块 docstring）。
        required_cases=tuple(sorted(set(required_cases or ()))),
        max_regression=tuple(sorted((str(k), float(v)) for k, v in (max_regression or ()))),
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
        required_cases=c.required_cases,
        max_regression=c.max_regression,
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
    """从 `as_dict()` 的产物重建 (断点续跑 / 离线复盘用)。

    ⚠️ spine-review P3-15: **校验载荷自带的 hash**。原先忽略它重新 freeze(),
    于是「被改过的载荷」会得到一个合法**新**哈希, 改写检测不出来。
    """
    try:
        ds = DTypeSemantics(**payload["dtype_semantics"])
        bd = BaselineDef(**payload["baseline"])
        shapes = ShapeDist(entries=tuple(
            (tuple(s), float(w)) for s, w in payload["shapes"]))
        c = freeze(
            kernel_type=payload["kernel_type"],
            dtype_semantics=ds,
            baseline=bd,
            shapes=shapes,
            max_error=payload["max_error"],
            min_speedup=payload["min_speedup"],
            target_metric=payload.get("target_metric", "speedup_vs_baseline"),
            required_caps=payload.get("required_caps") or (),
            notes=payload.get("notes", ""),
            # 老载荷（本片之前冻结的契约）没有这两个键 ⇒ 空 = **没声明**，判据面按
            # 「未确立」走。**不**在这里补默认值：补了就等于替人写了判据（见 §4.4 三态）。
            required_cases=tuple(payload.get("required_cases") or ()),
            max_regression=_regression_pairs(payload.get("max_regression")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"无法从载荷重建契约: {exc}") from exc
    claimed = payload.get("hash")
    if claimed and claimed != (c.hash or c.compute_hash()):
        raise ContractError(
            f"载荷自带哈希 {str(claimed)[:12]}… 与内容重算 "
            f"{(c.hash or c.compute_hash())[:12]}… 不一致: 判据被改写过")
    return c


def freeze_from_task(task, *, dtype: Optional[str] = None,
                     tf32: bool = False,
                     baseline_name: str = "pytorch_eager") -> FrozenContract:
    """从 TaskSchema 冻结一份契约 (生产接线用)。

    形状取任务的输入 shape 里最大的那个二维阵当代表 (单点 weight=1);
    数值语义按 `task.constraints.precision`。**不是**完整 workload profile ——
    那要 §3.3 的形状分布, 但先把「每轮校验哈希」这条链路接上。

    §6.2 条款 2/3 的两项声明 (`required_cases` / `max_regression`) 从
    `task.constraints` **透传**: 任务侧是唯一一处「契约与输入同时在手」的地方
    (harness 既建 task 又建 inputs), 声明从这里进契约 ⇒ 进 hash ⇒ 判据不可改。
    任务没声明 ⇒ 空 (**未声明**), 判据面按「未确立」走 —— 不替调用方补默认值。
    """
    import numpy as np
    from .schemas import Precision, TaskSchema  # noqa: F401

    kt = (getattr(getattr(task, "source", None), "kernel_type", "") or "unknown").strip()
    cons = getattr(task, "constraints", None)
    prec = getattr(cons, "precision", None)
    dtype_s = dtype or (getattr(prec, "value", None) or "fp32")
    max_error = float(getattr(cons, "max_error", 1e-4) or 1e-4)
    min_speedup = float(getattr(cons, "min_speedup", 1.0) or 1.0)
    return freeze(
        kernel_type=kt,
        dtype_semantics=DTypeSemantics(dtype=dtype_s, tf32=tf32,
                                       accum="fp32" if dtype_s in ("fp32", "fp16") else "fp32"),
        baseline=BaselineDef(name=baseline_name, same_device=True,
                             same_window=True, same_dtype_semantics=True),
        shapes=ShapeDist(entries=(((64, 64, 64), 1.0),)),
        max_error=max_error,
        min_speedup=min_speedup,
        required_caps=tuple(sorted(
            getattr(getattr(task, "target", None), "required_caps", ()) or ()
        )),
        required_cases=tuple(getattr(cons, "required_cases", ()) or ()),
        # 任务侧写的是 dict（JSON 原生），冻结契约收的是 tuple-of-pairs（可哈希）
        max_regression=_regression_pairs(getattr(cons, "max_regression", None)),
    )
