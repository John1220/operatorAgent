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

# `BenchmarkCase` / `TensorSpec` 与 **case 身份** 同处一个模块（`bench_cases`）：身份与声明
# 必须一起改，分两个模块迟早会漂。`case_key` 从这里**再导出**（`contract.case_key` 仍是合法
# 名字 ⇒ verifier / conductor / tests 的老调用点一字不动；定义搬家的理由见 I94）。
from .bench_cases import (BenchmarkCase, BenchCaseError, TensorSpec,  # noqa: F401
                          case_key, cases_from_shapes, known_layouts)
from .correctness import CorrectnessSuite, as_suite

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


# `case_key` 的定义在 `bench_cases.py`（I94 搬家：与 `BenchmarkCase` 同处 ⇒ 数组路径与
# 声明路径**同一个函数**）。这里只是再导出，老调用点（`from .contract import case_key`）
# 一字不动。


#: §6.2 的**聚合口径**（目标函数）。设计原文给的是
#: `Literal["geomean", "weighted_geomean", "max_regret"]` —— py3.9 没有 `StrEnum`
#: （同 `CLAIM_LEVELS` / `IMPLEMENTATION_DECISIONS` 的做法）⇒ 常数 + 有序元组。
#: `mean` = 加权算术平均（权重当相对份额）; `worst_case` = 最差的那一条 case ——
#: 设计里 `max_regret` 要挡的正是「总体好看不能掩盖最差的一条」那个方向。
#: 设计里的 `geomean`（等权）**不再单列**：它就是所有权重都为 1 的
#: `weighted_geomean`（多一个成员只会多一条会漂的同义路）。
CASE_AGGREGATE_WEIGHTED_GEOMEAN = "weighted_geomean"
CASE_AGGREGATE_MEAN = "mean"
CASE_AGGREGATE_WORST_CASE = "worst_case"
CASE_AGGREGATES = (CASE_AGGREGATE_WEIGHTED_GEOMEAN, CASE_AGGREGATE_MEAN,
                    CASE_AGGREGATE_WORST_CASE)


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
    #: §3.4 的 `BenchmarkCase`：**带具体输入**的那一条 case（`(BenchmarkCase, ...)`）。
    #: 空 tuple = **契约没声明具体输入**（老契约）⇒ 评测的输入由调用方给，case 身份只能
    #: 事后从数组读（`case_key`）。声明了 ⇒ 契约**拥有输入**（`materialize()` 造得出同一组
    #: 数组），且 `required_cases` / `max_regression` / `case_weights` 的 key 必须都指向这些
    #: case（CONTRACT_LINT 收紧）。进 hash：**评的是什么输入**也是判据的一部分。
    cases: tuple = ()
    correctness_suite: Optional[CorrectnessSuite] = None
    #: §6.2 条款 2/4：必须参与判决的 case（`case_key` 字符串，缺一不可）。
    #: 空 tuple = **契约没声明**（老契约）⇒ 判据按「未确立」处理，不是「都通过了」。
    required_cases: tuple = ()
    #: §6.2 条款 3：`((case_key, 允许的回退比例), ...)` —— 逐 case 硬回退上限。
    #: 用 tuple-of-pairs 而不是 dict：与 `ShapeDist.entries` 同款（冻结对象不可变、可哈希）。
    max_regression: tuple = ()
    #: §6.2 / §3.4：`((case_key, 权重), ...)` —— 目标函数的权重。空 tuple = **契约没
    #: 声明**（判据面按「未确立」走, 不是「权重都是 1」）。权重之和必须 > 0（校验）。
    case_weights: tuple = ()
    #: §6.2 的**聚合口径**（`CASE_AGGREGATES` 之一）。空串 = 没声明（同上一格的三态）。
    aggregate: str = ""
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
        # §3.4：**具体输入**进 hash（同「空声明不进」的纪律）。声明的形状/dtype/seed 改一个
        # 字就是另一份判据 —— 「评的是哪组输入」被悄悄换掉而 hash 不变，是最难发现的一类
        # 判据漂移（I94）。⚠️ case 身份仍是**形状签名**（I92 的读法）: 同形状不同 dtype 的两条
        # case 在身份上分不开 ⇒ CONTRACT_LINT 直接拒绝（见 `_lint_cases`）。
        if self.cases:
            d["cases"] = [c.content_dict() for c in self.cases]
        if self.correctness_suite is not None:
            d["correctness_suite"] = self.correctness_suite.content_dict()
        # §6.2 条款 4：这两项**全部进契约 hash** —— 判据变成可改的常量就等于没有判据。
        # 但**空的声明不进**：空 = 没声明（判据面按「未确立」走），与本片之前冻结的契约
        # 是同一份判据内容 ⇒ 同一个 hash（那样旧载荷还能被 `load_dict` 复盘，而
        # 「有↔无」「值变了」照旧必须改 hash）。见 CHECKLOG I92。
        if self.required_cases:
            d["required_cases"] = list(self.required_cases)
        if self.max_regression:
            d["max_regression"] = [[str(k), float(v)] for k, v in self.max_regression]
        # 同一条纪律（条款 4 + 空声明不进）: **权重进 hash**（设计 §6.2 明写「权重进
        # 契约 hash」）—— 权重能改就等于目标函数能改; 空串/空表 = 没声明 ⇒ 不进。
        if self.case_weights:
            d["case_weights"] = [[str(k), float(v)] for k, v in self.case_weights]
        if self.aggregate:
            d["aggregate"] = str(self.aggregate)
        return d

    def case(self, key: str) -> Optional[BenchmarkCase]:
        """这条 key 对应的**具体输入**；没声明 / 没这条 ⇒ `None`（不造一个空壳出来）

        三态同族：`None` = **契约没声明这条 case 的输入**（老契约，或走「调用方自带输入」那条
        路），此时输入从哪来由调用方回答（`metrics["case_source"]`）。
        """
        for item in self.cases:
            if item.key() == str(key):
                return item
        return None

    def regression_limit(self, case: str) -> Optional[float]:
        """这条 case 声明的允许回退比例；没为它声明 ⇒ `None`（**不是** 0）

        「没声明上限」与「声明了 0（一点也不许退）」是两件事，判据面必须分得开。
        """
        for key, value in self.max_regression:
            if str(key) == str(case):
                return float(value)
        return None

    def case_weight(self, case: str) -> Optional[float]:
        """这条 case 在目标函数里的权重；没为它声明 ⇒ `None`（**不是** 0）

        「没声明权重」与「声明了 0 权重（这条不进聚合）」是两件事：前者是没判据，
        后者是一条明确的排除。判据面必须分得开（同 `regression_limit`）。
        """
        for key, value in self.case_weights:
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


def _weight_pairs(raw) -> tuple:
    """同 `_regression_pairs`：dict 或 `(case, 值)` 序列都收，值**不在这里**转 float。"""
    if not raw:
        return ()
    items = raw.items() if isinstance(raw, Mapping) else raw
    return tuple((str(k), v) for k, v in items)


def _lint_cases(cases, required_cases, max_regression, case_weights) -> list:
    """§3.4 的 `cases` 进契约前的 CONTRACT_LINT（失败关闭 + 一个空间）。

    两条**收紧**（只在声明了 cases 时生效 ⇒ 老契约一字不受影响）：

    1. **身份分不开就拒绝**：case 身份是**形状签名**（I92 的读法）⇒ 同形状不同 dtype 的两条
       case 会是同一个 key，账本与判据里都分不开 ⇒ 拒绝冻结（想分开就得把 dtype 并进身份，
       那会改已发布的 key 格式 ⇒ 另开一片）。
    2. **key 空间必须指向声明过的 case**：`required_cases` / `max_regression` / `case_weights`
       里出现一个不在 `cases` 里的 key ⇒ 拒绝冻结。契约一旦拥有输入，「这条 key 是哪组输入」
       就不该再靠猜 —— 指向不存在的 case 的判据是一条永远判不了的判据（I92 的欠账）。
    """
    errs = []
    keys = []
    for item in cases or ():
        if not isinstance(item, BenchmarkCase):
            errs.append(f"cases 里有一项不是 BenchmarkCase: {type(item).__name__}")
            continue
        errs += [f"case[{item.name!r}]: {e}" for e in item.validate()]
        keys.append(item.key())
    if not keys:
        return errs
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        errs.append(f"cases 里有两条身份相同的 case（身份 = 形状签名）: {dupes}"
                    " ⇒ 账本与判据都分不开 ⇒ 拒绝冻结")
    declared = set(keys)
    for key in required_cases or ():
        if isinstance(key, str) and key and key not in declared:
            errs.append(f"required_cases 指向的 case {key!r} 不在契约声明的 cases 里"
                        " ⇒ 拒绝冻结")
    for label, items in (("max_regression", max_regression or ()),
                         ("case_weights", case_weights or ())):
        for item in items:
            try:
                key = item[0]
            except (TypeError, ValueError):
                continue   # 形状不对由 `_lint_case_declarations` 报, 这里不抢答
            if isinstance(key, str) and key and key not in declared:
                errs.append(f"{label} 指向的 case {key!r} 不在契约声明的 cases 里"
                            " ⇒ 拒绝冻结")
    return errs

def _lint_case_declarations(required_cases, max_regression,
                            case_weights=(), aggregate="") -> list:
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
    # 权重：非负、有限，且**和 > 0** —— 全零的权重表等于「谁都不算」，聚合出来的是
    # 一个没有分母的数（不是「没声明」，那会被读成一条判据）。
    weight_sum = 0.0
    for item in case_weights or ():
        try:
            key, value = item
        except (TypeError, ValueError):
            errs.append(f"case_weights 的项必须是 (case, 数值) 二元组: {item!r}")
            continue
        if not isinstance(key, str) or not key.strip():
            errs.append(f"case_weights 的 case 不是非空字符串: {key!r}")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errs.append(f"case_weights[{key!r}] 必须是数值: {value!r}")
        elif not math.isfinite(float(value)) or float(value) < 0.0:
            errs.append(f"case_weights[{key!r}] 必须 >= 0 且有限: {value!r}")
        else:
            weight_sum += float(value)
    if case_weights and weight_sum <= 0:
        errs.append("case_weights 的权重之和必须 > 0")
    if aggregate and aggregate not in CASE_AGGREGATES:
        errs.append(f"aggregate 不在词表里 {CASE_AGGREGATES}: {aggregate!r}")
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
           max_regression=(),
           case_weights=(),
           aggregate: str = "",
           cases=(), correctness_suite=None) -> FrozenContract:
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
    errs += _lint_case_declarations(required_cases, max_regression,
                                    case_weights, aggregate)
    errs += _lint_cases(cases, required_cases, max_regression, case_weights)
    if correctness_suite is not None:
        try:
            correctness_suite = as_suite(correctness_suite)
            if dtype_semantics.deterministic and not correctness_suite.deterministic:
                errs.append("数值语义要求确定性，但 CorrectnessSuite 关闭了确定性检查")
        except (BenchCaseError, TypeError, ValueError) as exc:
            errs.append(f"correctness_suite: {exc}")
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
        # 顺序无语义 ⇒ 按**身份**排序（同「同内容两份拷贝必须同哈希」那条要求）。
        cases=tuple(sorted(cases or (), key=lambda c: c.key())),
        correctness_suite=correctness_suite,
        notes=notes,
        # 顺序无语义 ⇒ 规范化成「排序去重」：同一份判据不该有两种 hash（同内容两份拷贝
        # 必须同哈希是契约哈希的第一条要求, 见模块 docstring）。
        required_cases=tuple(sorted(set(required_cases or ()))),
        max_regression=tuple(sorted((str(k), float(v)) for k, v in (max_regression or ()))),
        case_weights=tuple(sorted((str(k), float(v)) for k, v in (case_weights or ()))),
        aggregate=str(aggregate or ""),
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
        cases=c.cases,
        correctness_suite=c.correctness_suite,
        required_cases=c.required_cases,
        max_regression=c.max_regression,
        case_weights=c.case_weights,
        aggregate=c.aggregate,
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
            case_weights=_weight_pairs(payload.get("case_weights")),
            aggregate=str(payload.get("aggregate") or ""),
            # 具体输入（老载荷没有这个键 ⇒ 空 = 没声明，输入由调用方给）
            cases=tuple(BenchmarkCase.from_dict(d)
                        for d in (payload.get("cases") or ())),
            correctness_suite=payload.get("correctness_suite"),
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

    **§3.4 起契约还拥有输入**（I94）: 任务声明了 `shapes`（或直接给 `cases` 声明）⇒ 按
    `bench_cases.cases_from_shapes` 派生 `BenchmarkCase`（**具体输入**），进 hash；派生出的
    全部 case 在没有显式 `required_cases` 时就是「全部必判」（§3.4 规范 2 的原话）。两条
    声明同时给时必须对得上，未知 kernel_type 的布局一律**拒绝**（不猜）。
    """
    import numpy as np
    from .schemas import Precision, TaskSchema  # noqa: F401

    kt = (getattr(getattr(task, "source", None), "kernel_type", "") or "unknown").strip()
    cons = getattr(task, "constraints", None)
    correctness_suite = getattr(cons, "correctness_suite", None)
    if correctness_suite is not None:
        try:
            correctness_suite = as_suite(correctness_suite)
        except (ValueError, TypeError) as exc:
            raise ContractError(f"任务声明的 correctness_suite 无效: {exc}") from exc
    prec = getattr(cons, "precision", None)
    dtype_s = dtype or (getattr(prec, "value", None) or "fp32")
    max_error = float(getattr(cons, "max_error", 1e-4) or 1e-4)
    min_speedup = float(getattr(cons, "min_speedup", 1.0) or 1.0)
    # ── §3.4: 形状分布 → **具体输入**（`BenchmarkCase`）──
    #
    # 任务**没声明**形状分布 ⇒ 契约里仍是那个代表形状（与本片之前**同一份判据内容** ⇒
    # 同一个 hash，老产物照旧复盘），也不派生任何 case（输入由调用方给）。
    # 任务**声明了** shapes ⇒ 契约按它派生 cases: 契约从此**拥有输入**。未知布局的
    # kernel_type 一律拒绝（不猜 —— 猜错会让「评的是什么」静默漂移）。
    declared_shapes = tuple(getattr(cons, "shapes", ()) or ())
    shape_entries = (tuple((tuple(int(d) for d in s), 1.0) for s in declared_shapes)
                     if declared_shapes else (((64, 64, 64), 1.0),))
    raw_cases = tuple(getattr(cons, "cases", ()) or ())
    cases = ()
    if raw_cases:
        try:
            # 两条都收: JSON 原生的 dict（落盘/传输）与进程里的 `BenchmarkCase`（真机驱动）
            cases = tuple(item if isinstance(item, BenchmarkCase)
                          else BenchmarkCase.from_dict(item) for item in raw_cases)
        except BenchCaseError as exc:
            raise ContractError(f"任务声明的 cases 读不出来: {exc}") from exc
    if declared_shapes:
        derived = cases_from_shapes(kt, shape_entries, dtype_s)
        if derived is None:
            raise ContractError(
                f"任务声明了 shapes, 但 kernel_type={kt!r} 没有输入布局表 ⇒ 造不出具体输入"
                f"（不猜布局）; 有布局的是 {known_layouts()}")
        if cases and {c.key() for c in cases} != {c.key() for c in derived}:
            raise ContractError(
                "任务同时声明了 shapes 与 cases, 但两处派生的 case 身份对不上: "
                f"{sorted(c.key() for c in cases)} vs {sorted(c.key() for c in derived)}"
                " ⇒ 拒绝冻结（两个真相来源迟早会漂）")
        if not cases:
            cases = derived
    required_cases = tuple(getattr(cons, "required_cases", ()) or ())
    if cases and not required_cases:
        # §3.4 规范 2 的**原话**是「候选须在**全部**形状上通过」—— 声明了形状集/输入集，
        # 那个「全部」就是它们。这不是替人补默认值: 派生结果写进契约（进 hash）⇒ 在产物
        # 里看得见，而且**没声明** shapes/cases 的任务一个字都不动（仍三态未确立）。
        required_cases = tuple(c.key() for c in cases)
    return freeze(
        kernel_type=kt,
        dtype_semantics=DTypeSemantics(
            dtype=dtype_s, tf32=tf32, accum="fp32",
            deterministic=(correctness_suite.deterministic
                           if correctness_suite is not None else True)),
        baseline=BaselineDef(name=baseline_name, same_device=True,
                             same_window=True, same_dtype_semantics=True),
        shapes=ShapeDist(entries=shape_entries),
        max_error=max_error,
        min_speedup=min_speedup,
        required_caps=tuple(sorted(
            getattr(getattr(task, "target", None), "required_caps", ()) or ()
        )),
        required_cases=required_cases,
        # 任务侧写的是 dict（JSON 原生），冻结契约收的是 tuple-of-pairs（可哈希）
        max_regression=_regression_pairs(getattr(cons, "max_regression", None)),
        case_weights=_weight_pairs(getattr(cons, "case_weights", None)),
        aggregate=str(getattr(cons, "aggregate", "") or ""),
        cases=cases,
        correctness_suite=correctness_suite,
    )
