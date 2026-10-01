"""§12 的候选联合类型、参数空间, 与**不花 GPU 秒**的便宜代理剪枝

设计 §12 的原话：

> 接入 TVM / CUTLASS 的最大风险是接口把"候选"硬编码成源码字符串——那一刻起，
> 模板型后端就无法接入，整条链路要重写。因此 `Candidate` 必须是**联合类型**。
> ……`Backend.param_space()` 返回 `None` 的后端 = **代码型后端**，控制器自动切换成
> "LLM / 模板产出候选、bandit 只做局部精修"。这就是统一两类后端的那个开关。

今天的实际状态（2026-10-02）：**所有后端都是代码型**（没有一个实现 `param_space`），
链路上流动的候选就是 Triton 源码字符串。本模块做三件事：

1. 把**联合类型**立成数据（`SourceCandidate` / `TemplateCandidate`），并给它们一个
   **去重口径** `dedup_key`：源码候选按 α-等价结构签名（`novelty.signature_of`，与
   `iteration_history` 里那个签名**同一个实现**），模板候选按
   `(template_id, instantiator_ver, 排序后的参数)` —— 模板候选**没有源码可签名**，
   这正是「去重」在两类候选上必须分开写的原因；
2. 把参数空间立成数据：`ParamSpace.points()` 是**确定性**的笛卡尔积（轴序即枚举序，
   不许依赖 dict 的哈希顺序），以及 `prune_by_smem()` 这个便宜代理剪枝 ——
   它复用 `resource_check` 的 smem 估算器与**同一个宽松系数**，因此不花 GPU 秒就能把
   「这套 tile 根本放不下」的点摘掉；
3. 把那个**开关**写成一处判据 `param_space_of()` + `require_supported_space()`：
   `None` ⇒ 代码型（今天全部），非 `None` ⇒ 模板型，而**模板候选的构建还没实现** ⇒
   fail-closed 拒绝，不是悄悄降级。设计点名的那种风险恰恰是「悄悄把候选当成源码字符串」，
   所以这里宁可炸在一处、也不把「后端说有参数空间」读成「我们支持它」。

关于剪枝的边界（与 `resource_check` 同一口径，别在这里另立一套）：

* 估的是**数量级**：matmul 形态 `(BM*BK + BK*BN + BM*BN) * elem_bytes`，**不含**
  `num_stages` 的多缓冲（真实占用是它的若干倍）⇒ 只砍「明显放不下」的点，边界交给远端；
* 宽松系数 `SMEM_SLACK = 1.2`（超限 20% 以上才拒）与 L1 预检**共用同一个常数**，
  免得「预检放行、剪枝拒绝」这种互相矛盾的两套阈值；
* 上界取不到时**返回 None**（缺 ≠ 0 ≠ 无上限）：`smem_ok()` 的三态让调用方必须显式
  处理「不知道」，而不是把「没上界」读成「剪过了没事」。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .backend_caps import Backend
from .novelty import signature_of
from .resource_check import (
    SMEM_SLACK, estimate_matmul_smem_bytes, smem_limit_bytes,
)


# ─────────────────────────── 候选: 联合类型 ───────────────────────────

@dataclass(frozen=True)
class SourceCandidate:
    """源码型候选 (Triton / CUDA): 交出去的是**代码**"""

    backend: str
    source: str
    build_flags: Mapping[str, Any] = field(default_factory=dict)
    #: α-等价结构签名 (去重口径)。空串 = 还没算 —— 用 `source_candidate()` 构造会自动填。
    struct_sig: str = ""

    def __eq__(self, other: object) -> bool:
        # **候选的身份就是去重口径**。不这样写的话 `{a, b}` 会把 α-等价的同一份候选
        # 留两份 (字段比较里 `source` 的变量名不同 ⇒ 不相等), 而「去重」正是 §12 里
        # `struct_sig` 存在的理由。注意 dedup_key 里**故意不含** `build_flags` /
        # 源码原文: 同结构的候选是同一个候选, 编译参数不同不改变这一点。
        if not isinstance(other, (SourceCandidate, TemplateCandidate)):
            return NotImplemented
        return dedup_key(self) == dedup_key(other)

    def __hash__(self) -> int:
        # 与 `__eq__` 配套: 哈希也必须走同一个口径 (否则 set/dict 的去重行为与
        # `__eq__` 说的不一致, 同一份候选能同时留在集合里)。
        return hash(dedup_key(self))


@dataclass(frozen=True)
class TemplateCandidate:
    """模板型候选 (CUTLASS / TVM): 交出去的是**点+参数**, 代码由后端实例化

    `params` 是 `Mapping` 而不是 `str` —— 这是 §12 的全部要点: 接口一旦要求源码,
    模板后端就接不进来。
    """

    backend: str
    template_id: str
    params: Mapping[str, Any] = field(default_factory=dict)
    instantiator_ver: str = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (SourceCandidate, TemplateCandidate)):
            return NotImplemented
        return dedup_key(self) == dedup_key(other)

    def __hash__(self) -> int:
        return hash(dedup_key(self))


Candidate = Union[SourceCandidate, TemplateCandidate]


def is_template(cand: Candidate) -> bool:
    """是不是模板型候选。**按类型断, 不按有没有 `source` 断** —— 后者会把
    「模板候选恰好也带了源码」读成源码型, 而去重口径与构建路径都要跟着它分叉。"""
    return isinstance(cand, TemplateCandidate)


def source_candidate(backend: str, source: str,
                     build_flags: Optional[Mapping[str, Any]] = None) -> SourceCandidate:
    """构造源码候选并**当场算好** α-等价签名 (签名与 `iteration_history` 同源)"""
    return SourceCandidate(
        backend=backend, source=source,
        build_flags=dict(build_flags or {}),
        struct_sig=signature_of(source) or "",
    )


def _params_key(params: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """参数的身份: **按键排序**的元组 —— 字典顺序不该决定两个点是不是同一个"""
    return tuple(sorted((str(k), _freeze(v)) for k, v in dict(params).items()))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _params_key(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(v) for v in value)
    return value


def dedup_key(cand: Candidate) -> Tuple:
    """去重键。两类候选**必须分开写**：模板候选没有源码可签名。

    * 源码型: `("source", backend, struct_sig)`；签名算不出来时退到源码原文
      （原文相同仍是同一个候选），空源码退到 `("",)` —— 三种情况都**不返回 None**,
      因为「算不出签名」不代表「不该去重」。
    * 模板型: `("template", backend, template_id, instantiator_ver, 参数)`。
      `instantiator_ver` 进键: 同一个点位由不同版本的实例化器产出的是**不同的构建**。
    """
    if is_template(cand):
        assert isinstance(cand, TemplateCandidate)
        return ("template", cand.backend, cand.template_id, cand.instantiator_ver,
                _params_key(cand.params))
    assert isinstance(cand, SourceCandidate)
    ident = cand.struct_sig or cand.source or ""
    return ("source", cand.backend, ident)


def as_lineage(cand: Candidate) -> Dict[str, Any]:
    """血缘快照: 与 `iteration_history` 里那份**同形状**（签名/版本/参数）"""
    if is_template(cand):
        assert isinstance(cand, TemplateCandidate)
        return {
            "kind": "template", "backend": cand.backend,
            "template_id": cand.template_id,
            "instantiator_ver": cand.instantiator_ver,
            "params": dict(cand.params),
        }
    assert isinstance(cand, SourceCandidate)
    return {
        "kind": "source", "backend": cand.backend,
        "struct_sig": cand.struct_sig, "build_flags": dict(cand.build_flags),
    }


# ─────────────────────────── 参数空间 ───────────────────────────

@dataclass(frozen=True)
class ParamSpace:
    """模板型后端的参数空间: 轴 → 候选值。**枚举顺序确定**(轴序 × 值序)。

    `axes` 的值顺序就是枚举顺序 —— 不排序、不去重, 因为「顺序」在剪枝与截断
    (取前 N 个) 之后是**可观测**的: 悄悄排序会让「先试哪个」变成没人能预测的事。
    """

    template_id: str
    axes: Mapping[str, Tuple[Any, ...]]
    instantiator_ver: str = ""
    backend: str = ""

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("ParamSpace 必须带 template_id (模板候选的身份之一)")
        if not self.axes:
            raise ValueError("ParamSpace 的 axes 为空 ⇒ 参数空间里没有任何点")
        for name, values in self.axes.items():
            if not values:
                raise ValueError(f"参数轴 {name!r} 的候选值为空 ⇒ 空间恒为空集")

    def points(self) -> List[TemplateCandidate]:
        """笛卡尔积。轴序 = `axes` 的插入序, 最后一轴变化最快 (itertools 口径)。"""
        names = list(self.axes.keys())
        out: List[TemplateCandidate] = []
        for combo in product(*(self.axes[n] for n in names)):
            out.append(TemplateCandidate(
                backend=self.backend, template_id=self.template_id,
                params={n: v for n, v in zip(names, combo)},
                instantiator_ver=self.instantiator_ver))
        return out

    def size(self) -> int:
        n = 1
        for values in self.axes.values():
            n *= len(values)
        return n

    def as_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id, "backend": self.backend,
            "instantiator_ver": self.instantiator_ver,
            "axes": {k: list(v) for k, v in self.axes.items()},
            "size": self.size(),
        }


def param_space_of(backend: Any, contract: Any = None) -> Optional[ParamSpace]:
    """§12 的那个开关: 返回 `None` 的后端 = **代码型**。

    「没有这个方法」与「方法返回 None」都是代码型 —— 后者是设计明写的约定,
    前者是今天所有后端的实际形状（协议里的方法还没人实现）。两种情况**都**收敛到
    同一个答案, 但**不**互相掩盖: 返回值一样, `space_origin()` 能说出是哪一种。
    """
    return backend.param_space(contract)


def space_origin(backend: Any) -> str:
    """`"inherited"` = 用基类的默认 (代码型); `"overridden"` = 这个后端自己声明了空间

    为什么不留「协议方法都没有」这个状态 (第一版有): 方法**已经立在基类上**
    (`backend_caps.Backend.param_space`), 所以「没有这个方法」按设计**不可表达** ——
    那正是 `backend_caps` 模块开头第 1 条要的。于是有意义的只剩「覆写了没有」,
    而它决定控制器会不会在 `require_supported_space` 那里撞上 fail-closed。
    """
    base = getattr(Backend, "param_space")
    return "inherited" if type(backend).param_space is base else "overridden"


def require_supported_space(space: Optional[ParamSpace], backend: Any) -> Optional[ParamSpace]:
    """模板型后端（声明了非空参数空间）今天**不支持** ⇒ fail-closed

    为什么宁可炸也不降级: 设计 §12 点名的风险就是「把候选悄悄当成源码字符串」。
    今天 `TemplateCandidate` 的 `build()` 还不存在（没有实例化器），若在这里返回
    `None` 假装成代码型, 后端声明的参数空间会被无声丢弃 —— 那是「接口说谎」，
    而诚实的行为是把「我们还不支持这条路」说清楚。
    """
    if space is None:
        return None
    raise NotImplementedError(
        "模板型后端 (§12) 尚未支持: "
        # 用基类上的 `label` 而不是嗅探 `name`: 各提供方的 `name` 有的是属性、
        # 有的是 property, 而 `label` 是**声明好的**那个统一叫法。
        f"{backend.label!r} 声明了参数空间 "
        f"{space.template_id!r} ({space.size()} 个点), 但模板候选的实例化 "
        "(TemplateCandidate.build) 还没实现。当前链路只认源码型候选 —— "
        "不许把它降级成代码型后继续跑 (那正是 §12 要挡的失效)。")


# ─────────────────────── 便宜代理剪枝 (不花 GPU 秒) ───────────────────────

#: matmul 模板参数轴到 smem 估算器键名的约定 (与 `resource_check` 解析源码时的键名一致)
BLOCK_KEYS = ("BLOCK_M", "BLOCK_N", "BLOCK_K")


def tile_smem_bytes(block_m: int, block_n: int, block_k: int,
                    elem_bytes: int) -> Optional[int]:
    """一套 tile 的 smem 估算 (复用 `resource_check` 的公式与口径)"""
    return estimate_matmul_smem_bytes(
        {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
        int(elem_bytes))


def smem_ok(estimated_bytes: Optional[int], caps: Any) -> Optional[bool]:
    """三态: True 放行 / False 明显放不下 / **None 不知道**(没有上界或缺估算)。

    `None` 是刻意的第三态: 调用方必须显式决定怎么处理「不知道」, 而不是让它默认成
    「通过」—— 「缺上界」被读成「没问题」正是这一族反复出现的失效。
    """
    limit = smem_limit_bytes(caps)
    if limit is None or estimated_bytes is None:
        return None
    return estimated_bytes <= limit * SMEM_SLACK


@dataclass
class PruneResult:
    """剪枝的产物: 保留 / 摘掉(带理由) / 有没有真的剪过"""

    kept: List[TemplateCandidate] = field(default_factory=list)
    dropped: List[Tuple[TemplateCandidate, str]] = field(default_factory=list)
    #: 上界取不到 ⇒ 没剪。「没剪」必须与「剪完刚好一个都没摘」分开记。
    no_limit: bool = False
    #: 全被摘光 ⇒ 一个都没摘 (剪枝不是「让这一轮没得跑」)。理由仍留在 `dropped` 里。
    kept_all_to_avoid_starvation: bool = False

    def as_note(self) -> str:
        if self.no_limit:
            return "smem 代理剪枝: 未生效 (没有每 SM 上界)"
        if not self.dropped:
            return ""
        parts = [f"{c.params.get('BLOCK_M')}x{c.params.get('BLOCK_N')}"
                 f"x{c.params.get('BLOCK_K')}({why})" for c, why in self.dropped]
        if self.kept_all_to_avoid_starvation:
            return (f"smem 代理剪枝: 邻域 {len(self.dropped)} 点全超上界 "
                    f"⇒ 全部保留待远端判定 " + "/".join(parts))
        return f"smem 代理剪枝: 摘掉 {len(self.dropped)} 点 " + "/".join(parts)


def prune_by_smem(points: List[TemplateCandidate], caps: Any,
                  elem_bytes: int = 4) -> PruneResult:
    """按 smem 上界摘点。**全摘光时一个都不摘**（见下）。

    最后那条不是防御性代码: 剪枝的目的是「少花一次冤枉测量」, 不是「让这一轮没得跑」。
    若上界（或 elem_bytes 猜错）把邻域全砍了, 保留原集合交给远端判定, 比空手回上游好。
    """
    res = PruneResult()
    limit = smem_limit_bytes(caps)
    if limit is None:
        res.kept = list(points)
        res.no_limit = True
        return res
    kept: List[TemplateCandidate] = []
    for c in points:
        try:
            est = tile_smem_bytes(c.params["BLOCK_M"], c.params["BLOCK_N"],
                                  c.params["BLOCK_K"], elem_bytes)
        except (KeyError, TypeError, ValueError):
            kept.append(c)          # 认不出的参数轴 ⇒ 不剪 (宁可漏拦)
            continue
        if est is not None and est > limit * SMEM_SLACK:
            res.dropped.append((c, f"≈{est / 1024:.0f}KiB>{limit / 1024:.0f}KiB"))
        else:
            kept.append(c)
    if not kept and res.dropped:
        res.kept = list(points)
        res.kept_all_to_avoid_starvation = True
        return res
    res.kept = kept
    return res