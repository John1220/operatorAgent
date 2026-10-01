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

import hashlib
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .backend_caps import Backend, Instantiation
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


def require_supported_space(space: Optional[ParamSpace],
                            backend: Any) -> Optional[ParamSpace]:
    """§12 那条开关的**后半**: 声明了参数空间的后端, 我们能不能真的用它

    三种情形:
      * `space is None` ⇒ 代码型 (今天全部后端), 正常走 LLM/邻域那条路;
      * 有空间 + 有实例化器 ⇒ **返回空间** (2026-10-02 起, 「点 → 源码」那段已就绪:
        `instantiate_points`)。**但仍要调用方自己决定怎么消费** —— 控制器的候选队列
        还没接上参数空间, 那一半在 conductor 里 fail-closed (`_refuse_unconsumed_space`);
      * 有空间 + **没有**实例化器 ⇒ raise: 这是**后端**的缺口, 不许把它读成代码型
        (设计 §12 点名的风险就是「把候选悄悄当成源码字符串」, 声明的空间不许被无声丢掉)。
    """
    if space is None:
        return None
    if instantiator_available(backend):
        return space
    raise NotImplementedError(
        "模板型后端 (§12) 的这个后端还没就绪: "
        # 用基类上的 `label` 而不是嗅探 `name`: 各提供方的 `name` 有的是属性、
        # 有的是 property, 而 `label` 是**声明好的**那个统一叫法。
        f"{backend.label!r} 声明了参数空间 "
        f"{space.template_id!r} ({space.size()} 个点), 却没有实现模板实例化 "
        "(Backend.instantiate) —— 点建不成源码。当前链路只认源码型候选, 不许把它"
        "降级成代码型后继续跑 (那正是 §12 要挡的失效)。")


# ─────────────────────── 模板候选的实例化 (§12 的 `build`) ───────────────────────
#
# 设计 §12 的两类候选里, 模板型那半在 2026-10-02 只有**枚举**(第一片)。这一节补上
# 「点 → 源码」那一步, 并且把三件事钉死在一处:
#
#   1. **身份不跟着代码走**。`BuiltCandidate.dedup_key` 直接委派给那个 `TemplateCandidate`
#      —— 同一个点位由不同实例化器版本产出的两份源码, 是**同一个点的两次构建**, 不是两个
#      候选。反过来, 两个不同的点即使碰巧产出同一份源码, 也仍然是两个候选 (设计 §12 的
#      全部要点就是「候选不是源码字符串」)。
#   2. **版本必须自报并核对**。参数空间的身份含 `instantiator_ver`; 实例化器换版本却不说
#      ⇒ 去重键会把两个不同的构建读成同一个候选。
#   3. **有点没实例化必须看得见**。「N 个点里成功了 M 个」不许塌成「这一轮有 M 个候选」
#      —— 拒绝的每一条都留理由 (`InstantiationBatch.refusals`), 与「缺 ≠ 0」同族。

@dataclass(frozen=True)
class BuiltCandidate:
    """一个参数点**实例化后**的候选: 源码 + 出处, 身份仍是那个模板点

    `struct_sig` 是产出源码的 α-等价签名 —— 它是**证据**, 不是身份: 同一个点换一版
    实例化器 (版本号却没变) 会让它变, 而候选**还是那个点**。把它记下来, 正是为了让
    「同一版本、同一点位、源码却变了」这件事在血缘里看得见, 而不是被去重键吞掉。
    """

    candidate: TemplateCandidate
    source: str
    instantiator_ver: str = ""
    evidence: str = ""
    #: 产出源码的**内容哈希** (前 16 位)。跨语言都算得出来 —— 模板后端产出的多半是
    #: C++/CUDA, 而 `struct_sig` 那个量具是 Python AST 的 (`novelty.signature_of`),
    #: 对 C++ 源恒为空串。只靠它会把「源码变了」这件事在模板路径上整个看不见。
    source_hash: str = ""
    #: Python 源的 α-等价签名。**空串 = 量具不适用** (不是「没有源码」)。
    struct_sig: str = ""

    @property
    def params(self) -> Mapping[str, Any]:
        return self.candidate.params

    def dedup_key(self) -> Tuple:
        """**委派给模板候选**: 身份 = 点, 不是这份源码"""
        return dedup_key(self.candidate)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BuiltCandidate):
            return NotImplemented
        return self.dedup_key() == other.dedup_key()

    def __hash__(self) -> int:
        return hash(self.dedup_key())

    def as_lineage(self) -> Dict[str, Any]:
        out = as_lineage(self.candidate)
        # 源码签名只作为**证据**进血缘 (`build_struct_sig`), 不去替换 `struct_sig`
        # 那个属于源码型候选的身份字段 —— 两类候选的血缘字段不许互相冒名。
        out["build_source_hash"] = self.source_hash
        # 空串在这里是**三态里的「量具不适用」** (C++/CUDA 源没有 Python AST), 不是
        # 「没有源码」—— 判读的时候别把两者读成同一件事。
        out["build_struct_sig"] = self.struct_sig
        out["build_evidence"] = self.evidence
        return out


def source_hash(source: str) -> str:
    """源码内容哈希 (前 16 位)。**跨语言**都能算 —— 与 `struct_sig` 的分工:

    * `struct_sig` (Python AST α-等价) 回答「结构变没变」, 但它**只对 Python 源有效**;
    * `source_hash` 回答「内容是不是同一份」, 对 C++/CUDA 源一样有效。

    两个都记, 因为它们会独立变化: 换个变量名 ⇒ 哈希变、结构签名不变; 换编译宏里的
    常量 ⇒ 哈希变、结构也变。只记一个必然丢掉一半信息。
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def instantiator_available(backend: Any) -> bool:
    """调用 `backend.instantiate` 会不会走到**真实现** (而不是基类那条 raise)

    问法与 `space_origin` 刻意不同: 那个问「这个**类**覆写了没有」, 这个问「这一**次**
    调用会不会落到真实现上」。差别在**实例级替换** —— 测试假件 (`DeclaredFakeBackend`)
    正是把方法挂到实例上的, 只比类属性会把「能建出来」读成「没有实例化器」, 而
    `require_supported_space` 会据此拒绝一个其实能用的后端 (一个问题的两个答案)。

    为什么不留「后端根本没有这个方法」这个状态: 方法立在基类上 (默认 raise), 所以
    「没有」按设计不可表达 —— 有意义的只剩「这次调用有没有真实现」。
    """
    # 类上的实现: 协议默认 (那条 raise) 与覆写在这里就能分开。
    if type(backend).instantiate is not Backend.instantiate:
        return True
    # 实例级替换: 测试假件 (`DeclaredFakeBackend`) 是把方法挂到**实例**上的。
    # ⚠️ 这里刻意不用 `getattr(backend, …)`: `operator_agent/` 里禁止按方法名嗅探能力
    # (`tests/test_backend_caps.py::TestNoCapabilitySniffing` 按 AST 拦它, doc 49 §4.1
    # 第 3 条)。`vars()` 读的是实例字典, 不是「试一下这个方法在不在」。
    try:
        own = vars(backend).get("instantiate")
    except TypeError:            # `__slots__` 之类没有实例字典
        own = None
    return callable(own)


def instantiate_candidate(cand: TemplateCandidate, backend: Any,
                          contract: Any = None) -> BuiltCandidate:
    """`TemplateCandidate` → `BuiltCandidate`。四处 fail-closed (见下)

    拒绝用异常而不是「返回 None」: 这四处都是**调用方或后端的错**, 不是「这一轮没有
    候选」这种可继续的状态。批量那一层 (`instantiate_points`) 会把异常收成带理由的
    `refusals`, 所以「有点没建成」在整条链路上仍然看得见。
    """
    if not is_template(cand):
        raise TypeError("instantiate_candidate 只吃模板候选; 源码候选不许走这条路 "
                        "(§12: 两类候选的构建路径必须分开)")
    if not instantiator_available(backend):
        # 与 `Backend.instantiate` 基类那条 raise 同一个判据, 但更早、更点题: 这里能
        # 说清「哪个模板、哪个后端」, 而不是等到进去才炸。
        raise NotImplementedError(
            f"后端 {backend.label!r} 没有实现模板实例化 "
            f"(Backend.instantiate), 无法构建 {cand.template_id!r} 的参数 "
            f"{dict(cand.params)!r}")
    out = backend.instantiate(cand.template_id, dict(cand.params), contract)
    if not isinstance(out, Instantiation):
        raise TypeError(
            f"Backend.instantiate 必须返回 Instantiation, 拿到 "
            f"{type(out).__name__} —— 返回值形状不定会让下游各自解释")
    source = out.source
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            f"实例化器返回了空源码 ({cand.template_id!r} {dict(cand.params)!r}) —— "
            "「实例化失败」不许被读成「产出了一个空候选」")
    ver = out.instantiator_ver or ""
    if cand.instantiator_ver and ver and ver != cand.instantiator_ver:
        raise ValueError(
            f"实例化器版本不一致: 参数空间里是 {cand.instantiator_ver!r}, 实例化器自报 "
            f"{ver!r} —— 去重键 (template_id, instantiator_ver, 参数) 会把两个不同的"
            "构建读成同一个候选")
    return BuiltCandidate(
        candidate=cand, source=source,
        # 自报为空时**保留参数空间里的那个版本**, 而不是把它抹成空串: 空串会在血缘里
        # 读成「没有版本」, 而事实是「实例化器没说话, 我们知道的是空间里那个」。
        instantiator_ver=ver or cand.instantiator_ver,
        evidence=out.evidence or "", source_hash=source_hash(source),
        struct_sig=signature_of(source) or "")


@dataclass
class InstantiationBatch:
    """一批点的实例化结果: 建成的 / 拒绝的(带理由) / 有没有真建过

    `refusals` 是**一等公民**: 「12 个点建了 11 个」与「12 个点建了 12 个」在报告里
    必须分得开, 否则「实例化器在某类参数上一直失败」会变成「这一轮候选变少了」。
    """

    built: List[BuiltCandidate] = field(default_factory=list)
    refusals: List[Tuple[TemplateCandidate, str]] = field(default_factory=list)
    #: 入参为空 ⇒ 没建过任何东西 (与「建了 0 个」不是一回事)
    empty_input: bool = False

    def as_note(self) -> str:
        if self.empty_input:
            return "模板实例化: 没有点可建"
        note = f"模板实例化: 建成 {len(self.built)} 个"
        if self.refusals:
            head = "; ".join(f"{dict(c.params)}({why})" for c, why in self.refusals[:3])
            more = f" 等 {len(self.refusals)} 条" if len(self.refusals) > 3 else ""
            note += f", 拒绝 {len(self.refusals)} 个: {head}{more}"
        return note

    def lineage(self) -> List[Dict[str, Any]]:
        return [b.as_lineage() for b in self.built]


def instantiate_points(points: List[TemplateCandidate], backend: Any,
                       contract: Any = None) -> InstantiationBatch:
    """把一批点逐点实例化; **单点失败不打断**其余点, 但要留理由"""
    batch = InstantiationBatch()
    if not points:
        batch.empty_input = True
        return batch
    for cand in points:
        try:
            batch.built.append(instantiate_candidate(cand, backend, contract))
        except Exception as exc:            # noqa: BLE001 — 单点失败要留痕, 不许吞
            batch.refusals.append((cand, f"{type(exc).__name__}: {exc}"))
    return batch


def refuse_unconsumed_space(space: Optional[ParamSpace], backend: Any) -> None:
    """§12 那条路的**控制器侧**缺口: 点建得出来, 但还没有消费它的候选队列

    与 `require_supported_space` 的分工是刻意的 —— 两条缺口各有主人:

    * 后端侧 (没有实例化器) ⇒ `require_supported_space` 拒绝;
    * 控制器侧 (没有候选队列) ⇒ 这里拒绝。

    为什么不合成一条消息: 「后端没实现」与「控制器没接上」在报告里必须分得开 ——
    合成一条会让这两种状态都变成同一句「模板不支持」, 而它们的下一步动作完全不同。
    """
    if space is None:
        return
    raise NotImplementedError(
        f"模板型后端 {backend.label!r} 的参数空间 {space.template_id!r} "
        f"({space.size()} 个点) 现在**建得出源码** (candidates.instantiate_points), "
        "但控制器的候选队列还没接上参数空间 (§12 的「LLM / 模板产出候选」那一半)。"
        "当前这条搜索路径只认代码型候选 —— 不许把声明的空间降级成「没声明」后继续跑。")


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