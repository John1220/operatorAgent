"""§3.4 的 `BenchmarkCase`：**带具体输入**的那一条 case —— 契约与评测共用的那一个身份空间

设计 §3.4 的目标函数是 `cases: tuple[BenchmarkCase, ...]` + 权重 + 聚合口径；而本仓在
此之前有两个互不相干的空间：契约里的 `ShapeDist`（kernel 级 `(M,N,K)`，**零消费者**）与
评测侧的 case 身份（`contract.case_key` = 输入形状签名，I92 落地）。于是「契约声明要评
什么」与「评测真的评了什么」谁也管不着谁 —— I92/I93 的读法登记里那条「两个空间未合一」
说的就是它（§3.4 的欠账）。

这一片把 `BenchmarkCase` 立起来：

    TensorSpec(name, shape, dtype)        # 一个张量的**声明**（可序列化、可哈希）
    BenchmarkCase(name, tensors, seed)    # 一条 case = 一组**具体输入**

三件事必须**只有一处生产**：

1. **身份**（`BenchmarkCase.key()`）与 `case_key(inputs)` 是**同一个函数**（就在本模块）：
   数组那条路与声明这条路径给出同一个字符串。不是同一个函数的话，「契约声明的 case」永远
   匹配不上「本次评的 case」，`required_cases` 就恒为未确立 —— 那正是 I92 之后仍欠的一条。
2. **输入**（`BenchmarkCase.materialize()`）：由声明（形状 + dtype + seed）**确定性**地造出
   张量。契约拥有输入时，评测的输入就是契约里写的那一组（挡住「针对调用方挑的 shape
   特化」）；调用方仍然可以直接给数组，但那条路会记 `case_source="caller"`。
3. **形状分布 → case**（`cases_from_shapes`）：kernel 级 shape 到具体张量布局的**唯一**一张
   表。未知 kernel_type ⇒ `None`（**不猜**布局 —— 猜错会让「评的是什么」静默漂移，比缺这条
   能力危险得多）。

`case_key` 原来住在 `contract.py`（I92 落地）。搬到这里与 `BenchmarkCase` 同处：身份与声明
必须一起改，分两个模块迟早会漂。`contract.case_key` 保留为**再导出** ⇒ 老调用点
（`verifier` / `conductor` / tests）一字不动。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Tuple


class BenchCaseError(ValueError):
    """case 声明读不出来 / 造不出来。**失败关闭**：不猜布局、不补默认输入。"""


#: 契约里允许声明的数值语义（`TensorSpec.dtype`）。与 `DTypeSemantics.dtype` 同一套词，
#: 但这里**只需要**「张量怎么造」这一件事 —— 累加精度、tf32 那些属于契约的数值语义，
#: 由 `DTypeSemantics` 管，不重复声明。
CASE_DTYPES = ("fp32", "fp16", "fp64", "int32", "int64")

#: dtype → numpy 名字。`bf16` **故意不在表里**：numpy 没有这个 dtype，声明了也造不出来；
#: 与其悄悄降级成 fp32（读数与声明不符），不如拒绝（`BenchCaseError`）。
_NP_DTYPES = {"fp32": "float32", "fp16": "float16", "fp64": "float64",
              "int32": "int32", "int64": "int64"}


def case_key(inputs: Any) -> str:
    """一次评测的 **case 身份** = 输入集的形状签名（`A=64x64|B=64x64`）

    §6.2 条款 2 要求「候选须在**全部**形状上通过」，条款 4 要求 `required_cases` 进契约
    hash；本仓要用它判「所有必判 case 正确」就必须先能回答**本次评的是哪一条 case**。
    这个函数就是那个身份：按输入名排序 ⇒ 同一组输入任何时候都是同一个 key，产物里可离线
    复算、可逐字比对。

    **两条路同一个函数**：`inputs` 既可以是真张量（有 `.shape` 的数组/`TensorSpec`），也可以
    是 `BenchmarkCase.tensors`。于是「契约声明的 case」与「本次评的那组输入」在同一个空间里
    可比 —— 两个空间没合一之前，`required_cases` 是永远匹配不上的（I92 的欠账）。

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
class TensorSpec:
    """一个输入张量的**声明**（名字 + 形状 + 数值语义）。冻结、可哈希、可 JSON 往返。"""
    name: str
    shape: tuple = ()
    dtype: str = "fp32"

    def validate(self) -> list:
        errs = []
        if not isinstance(self.name, str) or not self.name.strip():
            errs.append(f"tensor.name 必须是非空字符串, 实际 {self.name!r}")
        if not isinstance(self.shape, tuple):
            errs.append(f"tensor.shape 必须是 tuple, 实际 {type(self.shape).__name__}")
        else:
            for d in self.shape:
                if not isinstance(d, int) or isinstance(d, bool) or d < 0:
                    errs.append(f"tensor.shape 的维度必须是非负 int, 实际 {self.shape!r}")
                    break
        if self.dtype not in CASE_DTYPES:
            errs.append(f"tensor.dtype 不在词表 {CASE_DTYPES} 里: {self.dtype!r}")
        return errs

    def to_dict(self) -> dict:
        return {"name": self.name, "shape": [int(d) for d in self.shape],
                "dtype": self.dtype}

    @staticmethod
    def from_dict(payload: Any) -> "TensorSpec":
        if not isinstance(payload, Mapping):
            raise BenchCaseError(f"tensor 声明必须是映射, 实际 {type(payload).__name__}")
        try:
            shape = tuple(int(d) for d in (payload.get("shape") or ()))
        except (TypeError, ValueError) as exc:
            raise BenchCaseError(f"tensor.shape 读不出来: {payload.get('shape')!r}") from exc
        spec = TensorSpec(name=str(payload.get("name") or ""), shape=shape,
                          dtype=str(payload.get("dtype") or "fp32"))
        errs = spec.validate()
        if errs:
            raise BenchCaseError("tensor 声明不可判定: " + "; ".join(errs))
        return spec


@dataclass(frozen=True)
class BenchmarkCase:
    """设计 §3.4 的 `BenchmarkCase`：**带具体输入**的那一条 case。

    `name` 只作审计（人读得懂的名字）；**身份**是 `key()` 那个形状签名 —— 两个进程各写一个
    名字不该让「同一条 case」变成两条。`seed` 决定输入的具体数值（同一个 seed + 同一份声明
    ⇒ 同一组数组），于是「契约里的输入」是**可复算**的，不是「谁跑谁造」。
    """
    name: str = ""
    tensors: tuple = ()            # (TensorSpec, ...)
    seed: int = 0

    def validate(self) -> list:
        errs = []
        if not self.tensors:
            errs.append("case 至少要有一个输入张量")
        seen = set()
        for t in self.tensors:
            if not isinstance(t, TensorSpec):
                errs.append(f"case.tensors 里有一项不是 TensorSpec, 实际 {type(t).__name__}")
                continue
            errs += [f"tensor[{t.name!r}]: {e}" for e in t.validate()]
            if t.name in seen:
                errs.append(f"case 里有两个同名张量: {t.name!r}")
            seen.add(t.name)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            errs.append(f"case.seed 必须是非负 int, 实际 {self.seed!r}")
        return errs

    def key(self) -> str:
        """这条 case 的身份 —— 与 `case_key(inputs)` **同一个函数**（就地调它，不另写一份）"""
        return case_key({t.name: t for t in self.tensors if isinstance(t, TensorSpec)})

    def seed_spec(self) -> dict:
        """可以**过线**的那份描述：seed + 每个张量的名字/形状/dtype。

        2026-10-20 起，远端按它重新物化输入并核 digest，不再整份传字节（fp16 4096³
        两个张量 ≈ 90 MB 文本，实测链路要 570 s，见评估文档 §9.8.1）。
        """
        from ..decide.remote_input_fidelity import SEED_PROTOCOL
        return {"protocol": SEED_PROTOCOL, "seed": int(self.seed),
                "tensors": [{"name": t.name, "shape": list(t.shape), "dtype": t.dtype}
                            for t in self.tensors]}

    def materialize(self) -> dict:
        """按声明造出这一组输入。**确定性**：同一份声明（含 seed）任何时候都是同一组数组。

        数值语义：浮点用 `standard_normal`（不是全 1/全 0 —— 全 1 的输入会把「索引写错」这类
        错误掩盖成正确），整数用 `integers(1, 8)`。dtype 不在表里 ⇒ 抛 `BenchCaseError`
        （**不悄悄降级**：降级会让读数与声明不符，而这种不符没人看得出来）。

        ⚠️ 具体抽取规则**不在这里**：它和远端物化共用 `remote_input_fidelity.
        materialize_seed_inputs`（一个实现两处调用）。两份实现分家之日，就是「控制端以为
        在验 A、远端实际在验 B」之时。
        """
        errs = self.validate()
        if errs:
            raise BenchCaseError("case 声明不可判定: " + "; ".join(errs))
        from ..decide.remote_input_fidelity import InputFidelityError, materialize_seed_inputs
        try:
            return materialize_seed_inputs(self.seed_spec())
        except InputFidelityError as exc:
            raise BenchCaseError(str(exc)) from exc

    def content_dict(self) -> dict:
        """进**契约 hash** 的内容（`to_dict()` 里去掉审计格 + 张量按名字规范化）

        两条纪律：
        * **审计名不进 hash**（同契约的 `notes`）: 两个进程各写一个 `name` 不该让同一条 case
          变成两份判据 —— `name` 只是给人读的;
        * **声明顺序不进 hash**: 张量按名字排序 ⇒ 同一组输入无论谁怎么写都是同一个 hash
          （内容哈希的第一条要求: 同内容两份拷贝必须同哈希）。
        """
        return {"seed": int(self.seed),
                "tensors": [t.to_dict() for t in
                            sorted(self.tensors, key=lambda t: t.name)]}

    def to_dict(self) -> dict:
        """落盘/传输的形状（**含**审计名 —— 产物里要能人读，`load_dict` 再读回来）"""
        return {"name": self.name, "seed": int(self.seed),
                "tensors": [t.to_dict() for t in self.tensors]}

    @staticmethod
    def from_dict(payload: Any) -> "BenchmarkCase":
        if not isinstance(payload, Mapping):
            raise BenchCaseError(f"case 声明必须是映射, 实际 {type(payload).__name__}")
        tensors = payload.get("tensors")
        if not isinstance(tensors, (list, tuple)):
            raise BenchCaseError(f"case.tensors 必须是列表, 实际 {type(tensors).__name__}")
        try:
            seed = int(payload.get("seed") or 0)
        except (TypeError, ValueError) as exc:
            raise BenchCaseError(f"case.seed 读不出来: {payload.get('seed')!r}") from exc
        case = BenchmarkCase(name=str(payload.get("name") or ""),
                             tensors=tuple(TensorSpec.from_dict(t) for t in tensors),
                             seed=seed)
        errs = case.validate()
        if errs:
            raise BenchCaseError("case 声明不可判定: " + "; ".join(errs))
        return case

# ── 形状分布 → case：kernel 级 shape 到具体张量布局的**唯一**一张表 ───────────────
#
# 键与 `benchmarks/drivers/harness_pass_rate.py:inputs_for` 同一套（那张表是「已经端到端跑通过的
# 输入契约」）：两处必须是同一组名字，否则 L2 的参考实现按名字逐个对上时会「找不到输入」。


def _matmul_case(kernel_type: str, shape: tuple, dtype: str, seed: int) -> "BenchmarkCase":
    m, n, k = shape
    return BenchmarkCase(
        name=f"{kernel_type}-{m}x{n}x{k}",
        tensors=(TensorSpec("A", (m, k), dtype), TensorSpec("B", (k, n), dtype)),
        seed=seed)


def _rows_case(kernel_type: str, shape: tuple, dtype: str, seed: int) -> "BenchmarkCase":
    m, n = shape
    return BenchmarkCase(
        name=f"{kernel_type}-{m}x{n}",
        tensors=(TensorSpec("X", (m, n), dtype),
                 TensorSpec("weight", (n,), dtype), TensorSpec("bias", (n,), dtype)),
        seed=seed)


#: kernel_type → (shape 的元数, 构造器)。**不在这张表里的 kernel_type 一律 `None`**
#: （不猜布局）。缺的是 attention / conv2d / csa_* 这几族：它们的输入布局要按 head 数、
#: stride、压缩块这些参数才定得下来，本仓没有一份「已验证过的」布局可以照抄 —— 编一个
#: 出来会让「评的是什么」静默漂移（正是本片要消掉的那类病），所以宁可没有。
_CASE_LAYOUTS = {
    "matmul": (3, _matmul_case),
    "gemm": (3, _matmul_case),
    "softmax": (2, _rows_case),
    "log_softmax": (2, _rows_case),
    "layernorm": (2, _rows_case),
    "rmsnorm": (2, _rows_case),
    "add": (2, _rows_case),
    "relu": (2, _rows_case),
    "gelu": (2, _rows_case),
    "silu": (2, _rows_case),
}


def cases_from_shapes(kernel_type: str, entries: Iterable, dtype: str = "fp32",
                      seed: int = 0) -> Optional[Tuple["BenchmarkCase", ...]]:
    """`ShapeDist.entries`（`((shape, weight), ...)`）→ `BenchmarkCase` 元组。

    没有布局表 ⇒ `None`（**不猜**）。权重**不回填**进 case：case 只回答「评的是哪组输入」，
    权重是目标函数那一半（`case_weights`）—— 两处都记一遍就会有两份会漂的权重。

    形状读不出来（不是整数、元数不对）⇒ 抛 `BenchCaseError`（**失败关闭**：宁可拒绝，
    也不拿一个截断/补零的形状去评）。
    """
    kt = (kernel_type or "").strip().lower()
    layout = _CASE_LAYOUTS.get(kt)
    if layout is None:
        return None
    arity, build = layout
    cases = []
    for entry in entries or ():
        try:
            shape, _weight = entry
        except (TypeError, ValueError) as exc:
            raise BenchCaseError(f"形状分布里的项必须是 (shape, weight): {entry!r}") from exc
        try:
            dims = tuple(int(d) for d in shape)
        except (TypeError, ValueError) as exc:
            raise BenchCaseError(f"{kt} 的形状读不出来: {shape!r}") from exc
        if len(dims) != arity or any(d < 0 for d in dims):
            raise BenchCaseError(
                f"{kt} 的形状必须是 {arity} 维非负整数, 实际 {shape!r}")
        cases.append(build(kt, dims, dtype, seed))
    return tuple(cases)


def case_from_inputs(inputs: Mapping, name: str = "", seed: int = 0) -> "BenchmarkCase":
    """**从真的输入**造一条 case 声明（形状 + 数值语义从数组读出来）

    用途：调用方手上已经有一组输入（未知布局的算子也在内），想让契约**拥有**它。声明由这组
    输入自己派生 ⇒ 身份必然一致（不可能出现「契约说 A、本次评的是 B」）。

    ⚠️ 记的是**形状与数值语义**那一半（case 身份就是形状签名，I92 的读法）：数值本身不在
    声明里 —— 要「数值也可复算」，走「任务声明 `shapes` ⇒ 契约派生 cases ⇒ 契约
    `materialize()` 造输入」那条路（输入由契约造）。`seed` 在两条路上都是审计格。

    读不出来（没有 `.shape` / dtype 认不出）⇒ 抛 `BenchCaseError`（**失败关闭**：一条读不
    出来的声明比「没声明」危险得多）。
    """
    if not isinstance(inputs, Mapping) or not inputs:
        raise BenchCaseError(f"case_from_inputs 需要一组非空输入, 实际 {type(inputs).__name__}")
    tensors = []
    for key in sorted(inputs):
        value = inputs[key]
        shape = getattr(value, "shape", None)
        if shape is None:
            raise BenchCaseError(f"输入[{key!r}] 没有 .shape ⇒ 读不出 case 声明")
        try:
            dims = tuple(int(d) for d in shape)
        except (TypeError, ValueError) as exc:
            raise BenchCaseError(f"输入[{key!r}] 的 shape 读不出来: {shape!r}") from exc
        dtype = str(getattr(value, "dtype", "") or "")
        tensors.append(TensorSpec(str(key), dims, _case_dtype(dtype, key)))
    case = BenchmarkCase(name=name, tensors=tuple(tensors), seed=seed)
    errs = case.validate()
    if errs:
        raise BenchCaseError("case 声明不可判定: " + "; ".join(errs))
    return case


#: numpy dtype 名 → 声明词表里的名字（只认得出映射表里的那几种）
_NUMPY_TO_CASE_DTYPE = {"float32": "fp32", "float16": "fp16", "float64": "fp64",
                         "int32": "int32", "int64": "int64"}


def _case_dtype(dtype_name: str, key: str) -> str:
    """张量的 dtype 名 → 词表里的名字；认不出就抛（不猜、不降级）"""
    plain = dtype_name.split(".")[-1] if dtype_name else ""
    mapped = _NUMPY_TO_CASE_DTYPE.get(plain) or (plain if plain in CASE_DTYPES else None)
    if mapped is None:
        raise BenchCaseError(
            f"输入[{key!r}] 的 dtype={dtype_name!r} 不在词表 {CASE_DTYPES} 里"
            f" ⇒ 读不出 case 声明（不猜）")
    return mapped


def known_layouts() -> tuple:
    """有布局表的 kernel_type（**只读**；报表/诊断用它说清「为什么这条算子没有 case」）"""
    return tuple(sorted(_CASE_LAYOUTS))



def case_spec(case: "BenchmarkCase") -> dict:
    """控制端下发的完整 spec：`seed_spec()` + 物化后的 digest。

    digest 由**控制端先算一遍**（并与真实 inputs 比对，见 `Verifier._seed_spec_for`），
    远端再算一遍；两侧一致才使用物化的那份。
    """
    from ..decide.remote_input_fidelity import seed_input_digest
    spec = case.seed_spec()
    spec["digest"] = seed_input_digest(case.materialize())
    return spec
