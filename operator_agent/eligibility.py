"""资格位 (eligibility) —— 一个观测值被允许造成什么后果

设计出处: `docs/design/autonomous-operator-agent-design.md` §4.2 / §4.3 / §13.3。
核心句 (§4.3):

    资格决定「这个数被允许造成什么后果」, 而不只是「过没过」。

与 `measurement.MeasurementAssessment` 的关系 (不是重命名, 也不 import 它):
那边是**三档有效性** (valid / degraded / uninformative) 加四个用途位
(`can_gate` / `can_disable_recipe` / `can_count_stagnation` / `l3_pass_is_refused`);
本模块是设计文档要求的**五位资格**, 把用途拆得更细:

    can_verdict        能判对 / 错
    can_tune           能改配置 / 参数
    can_persist        能写长期记忆 (永久生效)
    can_report         能对外报数
    can_count_progress 能算一次有效迭代

关键方向 (§4.3 两条设计意图的第一条): `can_persist` / `can_count_progress`
**只在 VALID 放开** —— 一次噪声读数永远不能永久拉黑一个配方, 也永远不能算作收敛证据。
所以 DEGRADED 的位型是 `(verdict, tune, persist, report, progress) = (1, 1, 0, 1, 0)`。

第 3 条与第 6 条刻意不同 (§4.3 第二条): 降级是「数不可信」, REAL_DEVICE 缺失是
「数不存在」。后者整组拒绝, 因为它测的根本不是契约要求的东西 —— 与
`MeasurementAssessment.l3_pass_is_refused` 同一条边界, 但落在资格位上是**全 False**,
而不只是拒绝 `passed`。

纯函数 + 简单布尔入参。调用方把 `MeasurementAssessment` 的字段 (validity /
l3_pass_is_refused / backend_real_device) 与心跳、契约哈希、峰值、lib fallback 等
信号摊平传进来即可; 本模块不 import `measurement`, 不做重逻辑, 可离线单测。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Tuple

from .contract import (CASE_AGGREGATE_MEAN, CASE_AGGREGATE_WORST_CASE,
                       CASE_AGGREGATES)
from .source_lang import COVERAGE_AST, COVERAGE_NONE, COVERAGE_REGEX

# §4.3 第 5 条的常数: value > 参考下界 × 1.05 判为「测量错了」而不是「很棒」。
#
# 语义是**余量比例** (0.05 = 5% 余量), 阈值 = `峰值 × (1 + PEAK_SLACK)`, 不是乘数。
# 这一格踩过一次坑 (spine-review P2-9): 常数取名 `slack`、docstring 写 `×(1+slack)`,
# 而代码写的是 `×slack` —— 默认值 1.05 让两种语义**数值上巧合相等**, 于是测试全绿;
# 第二个调用方按 docstring 传 `0.05` (意为「5% 余量」) 时阈值变成 `峰值 × 0.05`,
# 几乎所有读数都会被判成「超上界」。现在常数与判据只有这一个真源, 调用方一律走
# `exceeds_measured_peak` (oracle 里那句手写 `* 1.05` 已删)。
PEAK_SLACK = 0.05


@dataclass(frozen=True)
class Eligibility:
    """五位资格掩码 (设计 §13.3)。字段顺序即文档里的叙述顺序。"""

    can_verdict: bool          # 能判对 / 错
    can_tune: bool             # 能改配置 / 参数
    can_persist: bool          # 能写长期记忆 (永久生效)
    can_report: bool           # 能对外报数
    can_count_progress: bool   # 能算一次有效迭代

    def as_dict(self) -> dict:
        """给 Observation / JSONL 用的扁平形态 (设计 §4.1 的 eligibility 字段)"""
        return {
            "can_verdict": self.can_verdict,
            "can_tune": self.can_tune,
            "can_persist": self.can_persist,
            "can_report": self.can_report,
            "can_count_progress": self.can_count_progress,
        }

    @classmethod
    def from_mapping(cls, mapping) -> Optional["Eligibility"]:
        """从产物里的扁平 dict 还原五位掩码。**缺一位 ⇒ None**（不是「默认 True」）

        与 `oracle.admit_to_corpus` 的既有口径同源（有键但掩码不全 ⇒ 判不合格），
        但这里更保守：掩码不全一律返回 `None`，由调用方按「**没有资格位**」处理 ——
        缺证据不是有效证据，而「默认补齐」正是本仓反复记过的那类自欺。
        """
        if not isinstance(mapping, Mapping):
            return None
        keys = ("can_verdict", "can_tune", "can_persist", "can_report",
                "can_count_progress")
        if any(k not in mapping for k in keys):
            return None
        return cls(*[bool(mapping[k]) for k in keys])


#: 全 False —— 规则 1~5: 这个数没有指称对象 / 测的是别的东西 / 物理上不可能
ALL_FALSE = Eligibility(False, False, False, False, False)
#: DEGRADED 位型 (规则 6~7): 判决 / 调参 / 报数可以, 持久化 / 计进度不行
DEGRADED = Eligibility(True, True, False, True, False)
#: 全 True —— 规则 8: VALID
ALL_TRUE = Eligibility(True, True, True, True, True)


def exceeds_measured_peak(value: Optional[float],
                          measured_peak: Optional[float],
                          slack: float = PEAK_SLACK) -> bool:
    """`value > 参考下界 × (1+slack)`? 任一侧缺失则不触发 (缺证据不等于超上界)

    阈值用**实测**峰值而不是 spec 表: roofline 在这里是校验边界 (§4.2 第 3 条),
    报出的 FLOPS 超过峰值说明测量错了, 不是说明很棒 —— 分母若来自规格表, 一次
    钟频拉满的读数就可能合法地「超上界」, 那样这条检查会变成假警报发生器。
    """
    if value is None or measured_peak is None:
        return False
    if measured_peak <= 0:
        return False
    return value > measured_peak * (1.0 + slack)


def _tier(validity) -> Optional[str]:
    """把 validity 归一成小写字符串; None = 调用方没给档位

    接受 `measurement.Validity` 的成员 (str 枚举) 也接受裸字符串 —— 调用方不必
    为了传个档位就 import measurement, 本模块也不必反过来依赖它。
    """
    if validity is None:
        return None
    return str(getattr(validity, "value", validity)).lower()


def derive_eligibility(
    *,
    contract_hash_ok: bool = True,
    heartbeat_ok: bool = True,
    requires_real_device: bool = False,
    real_device: Optional[bool] = None,
    l3_pass_is_refused: bool = False,
    correctness_verified: Optional[bool] = None,
    lib_fallback: bool = False,
    static_guards: Optional[bool] = None,
    peak_exceeded: bool = False,
    cv_exceeded: bool = False,
    launch_dominance_exceeded: bool = False,
    partition_drift: bool = False,
    validity=None,
) -> Eligibility:
    """按设计 §4.3 的 8 条规则推导五位资格, 优先级从高到低

    1  契约哈希不匹配                            → 全 False (调用方已拒评, 这里也拒)
    2  落在心跳失败的标定窗口内                   → 全 False (尺子自己坏了)
    3  契约要求 REAL_DEVICE 但 real_device=False   → 全 False (数没有指称对象)
       `l3_pass_is_refused=True` 同条: 评估器已经认定「通过」无对象
       `correctness_verified=False` 同条 (设计 §14-10 / §20.4): 这个候选的正确性
       没有凭据 («这个核算得对» 没被验证过) ⇒ 它的速率读数是**关于另一个对象**的。
       只有 `verify_l3_only` 这条绕开 L2 的入口才可能带出 False —— 正常路径上
       数值错的候选到不了 L3 (L2 失败即返回)。
    4  检测到 lib fallback                        → 全 False (它在测别的东西)
    5  value > 参考下界 × 1.05 / 档位已是 uninformative → 全 False
    6  cv 超阈值 或 launch 下限占比超阈值          → DEGRADED (判决/调参/报数, 不持久化/不计进度)
    7  partition 漂移 / 档位已是 degraded          → DEGRADED (同 6 的位)
    8  静态守卫**不适用**于这份源码 (`static_guards=False`) → DEGRADED (CHECKLOG I65)
       「查不了」不是「没有问题」: 反作弊与接口契约两道守卫都是 Python 形状的, 对一份
       C++/CUDA 模板源它们判不了。数字本身照旧可判、可调、可报, 但**不许进长期记忆、
       不许计进度** —— 与 6/7 同族: 低资格 ≠ 没测准, 而是这个数不足以支撑**永久**结论。
    9  其余                                       → VALID, 全 True

    `validity` 是**桥接**入参: 调用方若已经有 `MeasurementAssessment.validity`,
    直接传进来, 污染、出处矛盾等没单列成布尔的信号也会落到正确的位型上。
    它只降不升 —— 显式布尔判得更重时以布尔为准 (例如 peak_exceeded=True 时即使
    validity 说 valid 也整组拒绝)。

    关于 `real_device=None`: 与 `False` 不是一回事。None 表示调用方没做出处检查,
    老路径的行为是不追究 (缺键 ≠ 检查过、不合格); 只有明确的 False 才触发规则 3。
    这条与 `MeasurementAssessment.as_metrics` 里 `is False` 的写法同一边界 ——
    否则一份诚实的本地 mock 运行会因为载荷老实写了 device="cpu" 而被判死。
    """
    tier = _tier(validity)

    # 规则 1: 契约是尺子。哈希不匹配 = 判据被改写, 任何数字都无资格 (§10.5-5)
    if not contract_hash_ok:
        return ALL_FALSE

    # 规则 2: 心跳失败窗口内的观测全部失效 (§10.5-1 / §13.5)
    if not heartbeat_ok:
        return ALL_FALSE

    # 规则 3: 数不存在 ≠ 数不可信。整组拒绝 (§4.3 第二条设计意图)
    # `correctness_verified=False` 收在同一族: 没有正确性凭据的速率读数说的是另一个对象
    # (§14-10 / §20.4)。三态 —— None (没带凭据) **不改判**, 与 real_device 同一边界。
    if (l3_pass_is_refused or (requires_real_device and real_device is False)
            or correctness_verified is False):
        return ALL_FALSE

    # 规则 4: 静默回退到库 = 候选根本没跑生成代码 (§4.2 第 1 条, 最常见的自欺)
    if lib_fallback:
        return ALL_FALSE

    # 规则 5: 超物理上界 = 测量错了。UNINFORMATIVE → 全 False
    if peak_exceeded or tier == "uninformative":
        return ALL_FALSE

    # 规则 6 / 7: 降级。可判决、可调参、可报数, 但不得持久化、不得计进度
    if cv_exceeded or launch_dominance_exceeded or partition_drift or tier == "degraded":
        return DEGRADED

    # 规则 8: 静态守卫不适用 (CHECKLOG I65)。三态 —— `None` (无法确立/没观测) **不改判**:
    # 与 `real_device=None` 同一边界 (缺键 ≠ 检查过且不合格)。只有明确的 False 才降级。
    if static_guards is False:
        return DEGRADED

    # 规则 9
    return ALL_TRUE


# ─── §4.4 声明等级 (ClaimLevel)：能力位之外的「这条结论现在处于什么地位」 ─────
#
# 设计原文是一个 `StrEnum`；本仓跑 3.9（`StrEnum` 要 3.11），所以与
# `measurement.Validity` 同款：**字符串常数 + 升序元组**，产物里记字符串。
#
# 与资格位的关系：`can_*` 回答「这个数能干什么」，声明等级回答「这条结论站到了哪一级、
# 还差哪一件证据」。两者的判据是**分开的**：等级只由本模块这一段推导，消费方（入库判据 /
# 报告）不许自己再排一次序。

CLAIM_UNOBSERVED = "unobserved"      # 没有可用测量 —— 不能判决、不能学习
CLAIM_OBSERVED = "observed"          # 有原始观测，但资格不足 —— 只能诊断与继续探索
CLAIM_PROVISIONAL = "provisional"    # 当前运行内可比较 —— 可局部搜索, 不能晋级 best
CLAIM_VALIDATED = "validated"        # 通过正确性与测量门禁 —— 可参与正式比较
CLAIM_PROMOTABLE = "promotable"      # 通过独立确认 —— 可以替换运行级 best
CLAIM_PERSISTED = "persisted"        # 进入长期知识库 —— 可跨任务复用

#: **顺序即升序**（§4.4：等级只能逐级上升，每一级上升都要有对应证据）。
CLAIM_LEVELS: Tuple[str, ...] = (
    CLAIM_UNOBSERVED, CLAIM_OBSERVED, CLAIM_PROVISIONAL,
    CLAIM_VALIDATED, CLAIM_PROMOTABLE, CLAIM_PERSISTED,
)


def claim_rank(level: Optional[str]) -> int:
    """等级的序号（越大越高）。**不在词表里 ⇒ -1** —— fail-closed：读不懂的等级
    不比任何一级高（与「缺证据不是有效证据」同一条纪律）。"""
    try:
        return CLAIM_LEVELS.index(str(level))
    except ValueError:
        return -1


def at_least(level: Optional[str], floor: str) -> bool:
    """`level >= floor`? 任一侧不在词表里 ⇒ False"""
    return claim_rank(level) >= claim_rank(floor) >= 0


#: §22.3：静态审计的**覆盖强度**决定「这份源码的结论最多能站到哪一级」。
#: 为什么上限按 `mode` 而不按 `uncovered_rules` 是否为空：见 `claim_ceiling`。
CLAIM_CEILING_BY_COVERAGE: Dict[str, str] = {
    COVERAGE_AST: CLAIM_PERSISTED,        # 看得穿源码（Python AST）⇒ 不设限
    COVERAGE_REGEX: CLAIM_VALIDATED,      # 真实但更弱：运行内可比, 不足以永久
    COVERAGE_NONE: CLAIM_PROVISIONAL,     # 判不了（无法确立 ≠ 通过）
}


def claim_ceiling(coverage_mode: Optional[str]) -> str:
    """覆盖强度 → 等级上限。词表外的值（含 `None` / 空串 = **没判过**）按最保守那档。

    **为什么用 `mode` 而不是「`uncovered_rules` 非空就压级」**：后者会自锁 ——
    §21.4 第 7 条（引用不存在的能力）的**源码半边**对**每一份** patch 候选都是 `unknown`
    （仓里没有任何语言的 API 面清单 ⇒ **有对象、没量具**），于是 `uncovered_rules`
    永远至少是 `{capability}`。拿它压级 ⇒ 每一份候选都到不了 `promotable` ⇒ 入库率归零，
    而那把尺子拦的是**量具缺失**，不是候选的问题。`mode` 说的是「这份源码的静态检查
    能力」，那才是 §22.3 的判据面；`uncovered_rules` 作为证据进报告
    （`report._uncovered_rules_from_lineage`）。

    （2026-10-07 之前 `uncovered_rules` 还把 `not_applicable` 一起收进来，见 CHECKLOG
    I90 —— 那让「这一层查不了」多报两条。改成只收 `unknown` 之后这条自锁的理由不变，
    只是集合从「混着的四条」变成「永远的那一条」。）
    """
    return CLAIM_CEILING_BY_COVERAGE.get(str(coverage_mode or "").strip().lower(),
                                         CLAIM_PROVISIONAL)


@dataclass(frozen=True)
class ClaimRecord:
    """一次读数的声明等级 + 它为什么停在那一级（产物里记这个）。"""

    level: str
    ceiling: str
    coverage_mode: str
    reasons: Tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"level": self.level, "ceiling": self.ceiling,
                "coverage_mode": self.coverage_mode, "reasons": list(self.reasons)}


def claim_level(*,
                eligibility: Optional[object] = None,
                validity: Optional[object] = None,
                correctness_verified: Optional[bool] = None,
                coverage_mode: Optional[str] = None,
                independent_confirmation: Optional[bool] = None,
                persisted: bool = False) -> ClaimRecord:
    """按 §4.4 的阶梯推导声明等级（纯函数；**等级的唯一推导处**）。

    阶梯（顺序即依赖，每一级都要有对应证据，缺证据不许升）：

    | 等级 | 需要的证据 |
    |---|---|
    | `unobserved` | 没有可用测量：没有资格位且没有档位，或 `can_verdict=False`（整组拒绝） |
    | `observed` | 有原始观测但资格不足：有档位没资格位，或 `can_tune=False` |
    | `provisional` | 有 `can_tune` 的读数；但档位不是 `valid`，或正确性没有凭据 |
    | `validated` | 档位 `valid` **且** 正确性有凭据（§20 / 规则 9） |
    | `promotable` | 通过一次独立确认（§10.5-4） |
    | `persisted` | 入库判据通过（§24.3） |

    最后**逐级上升不许越级**的保障落在一处：`level = min(阶梯, 覆盖上限)`（§22.3）。
    覆盖不足只**压级**，不改变任何资格位（`can_*` 仍然由 §4.3 的八条规则说了算）——
    「coverage 不足只能影响 ClaimLevel 与 promotion，不能替代执行沙箱」。

    三态纪律（设计与本仓的基线）：
      * `eligibility=None`（没经过收口点）⇒ 按「没有资格位」处理，不许补齐；
      * `correctness_verified=None`（没带凭据）⇒ **不升** `validated`（缺证据不是有效证据）；
      * `coverage_mode=None`（没判过）⇒ 上限按最保守那档（`provisional`）。

    ⚠️ **`promotable` 今天在产物里很少出现**：本仓的「独立确认」与「入库」是**同一步**
    （`oracle.admit_to_corpus` 的第 3 道门），通过独立确认那一刻就已经入库 ⇒ 直接
    `persisted`。`promotable` 今天的角色是**门槛**（入库门要求覆盖上限 ≥ 它），
    等 §24.3 的 promotion policy 从入库里独立出来之后才会成为终态。不许把它读成
    「今天有一条路径能停在 promotable」。
    """
    elig = eligibility
    if elig is not None and not isinstance(elig, Eligibility):
        elig = Eligibility.from_mapping(elig)

    ceiling = claim_ceiling(coverage_mode)
    mode = str(coverage_mode or "")
    tier = _tier(validity)
    reasons: list = []

    # ── 阶梯（从下往上，只取**证据齐全**的最高一级）──
    if persisted:
        base = CLAIM_PERSISTED
    elif independent_confirmation is True:
        base = CLAIM_PROMOTABLE
    elif elig is None and tier is None:
        base = CLAIM_UNOBSERVED
        reasons.append("没有资格位也没有档位 ⇒ 没有可用测量")
    elif elig is not None and not elig.can_verdict:
        base = CLAIM_UNOBSERVED
        reasons.append("can_verdict=False（整组拒绝）⇒ 这个数没有指称对象")
    elif elig is None or not elig.can_tune:
        base = CLAIM_OBSERVED
        reasons.append("有原始观测但资格不足（" +
                       ("没有资格位" if elig is None else "can_tune=False") + "）")
    elif tier != "valid" or correctness_verified is not True:
        base = CLAIM_PROVISIONAL
        if tier != "valid":
            reasons.append(f"档位 {tier or '未报'} ≠ valid ⇒ 未达 validated")
        else:
            reasons.append("正确性没有凭据（correctness_verified 不是 True）⇒ 未达 validated")
    else:
        base = CLAIM_VALIDATED

    level = base
    if claim_rank(ceiling) < claim_rank(CLAIM_PERSISTED):
        # 上限本身也要印出来（哪怕它没有压住当前这一级）—— 否则「为什么进不了语料」
        # 在产物里只剩入库门那句拒因，而拒因的量在覆盖这一格。
        reasons.append(f"静态审计覆盖 {mode or '未判'} ⇒ 等级上限 {ceiling} (§22.3)")
        if claim_rank(ceiling) < claim_rank(base):
            level = ceiling
    if claim_rank(level) < claim_rank(CLAIM_PERSISTED) and not persisted:
        reasons.append(f"没有入库 ⇒ 未达 {CLAIM_PERSISTED}")
    return ClaimRecord(level=level, ceiling=ceiling, coverage_mode=mode,
                       reasons=tuple(reasons))


# ─── §17.1 实现裁决 (ImplementationDecision)：交付物第 1 件里那一个显式结论 ─────
#
# 设计原文是一个 `StrEnum`；本仓跑 3.9（`StrEnum` 要 3.11），所以与
# `measurement.Validity` / 上面的 `CLAIM_*` 同款：**字符串常数 + 元组**，产物里记字符串。
#
# 为什么这一格必须接进同一套资格位（设计 §17.1 的原话）：本文反复防范的失效模式是
# 「噪声读数造成永久后果」——「因为一次噪声读数就建议你别上自定义核」与「因为一次噪声
# 读数就永久拉黑一个配方」是同一失效模式的两个方向。裁决本身就是一个会被人**拿去部署**
# 的结论，所以它读的证据与 §4.4/§22.3 是同一套，不许在下游另开一个没有闸门的新口子。

#: 判据是**严格**越限, 但浮点减法在「恰好等于上限」时会给出 1e-17 级的正差 —— 那是
#: 表示误差, 不是回退。留 1e-9 的余量: 远大于表示误差, 远小于任何真实回退的尺度。
REGRESSION_SLACK = 1e-9

DECISION_USE_CUSTOM = "use_custom"
DECISION_USE_LIBRARY = "use_library"
DECISION_INCONCLUSIVE = "inconclusive"

#: 词表（顺序即从「敢用」到「判不了」），产物里记字符串。
IMPLEMENTATION_DECISIONS: Tuple[str, ...] = (
    DECISION_USE_CUSTOM, DECISION_USE_LIBRARY, DECISION_INCONCLUSIVE,
)


@dataclass(frozen=True)
class DecisionRecord:
    """一次运行的实现裁决 + 它为什么是这个结论（§17.1；产物里记这个）。

    三个证据字段是**三态**（`True` / `False` / `None`）：`None` = **未确立**，
    不是 False —— 「没量过回退」与「量过、没有回退」是两件事（与 §4.4/§22.3 同一条纪律）。
    """

    decision: str
    claim_level: str
    claim_ceiling: str
    can_report: Optional[bool]
    required_cases_ok: Optional[bool]
    regression_over_limit: Optional[bool]
    reasons: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.decision not in IMPLEMENTATION_DECISIONS:
            raise ValueError(f"裁决不在词表里: {self.decision!r} "
                             f"(表 {IMPLEMENTATION_DECISIONS})")

    def as_dict(self) -> dict:
        return {"decision": self.decision, "claim_level": self.claim_level,
                "claim_ceiling": self.claim_ceiling, "can_report": self.can_report,
                "required_cases_ok": self.required_cases_ok,
                "regression_over_limit": self.regression_over_limit,
                "reasons": list(self.reasons)}


def _claim_record_of(claim_record) -> ClaimRecord:
    """把产物里的 `claim_level` 摊平成 `ClaimRecord`（形状不对/缺键 ⇒ 等级读成空串）

    空串的 `claim_rank` 是 -1 ⇒ 一档都够不上（fail-closed）。这里**不**补任何缺省等级：
    「没经过 §4.4」与「等级很低」都上不了 `use_custom`，但前者要在理由里说出来。
    """
    if isinstance(claim_record, ClaimRecord):
        return claim_record
    if isinstance(claim_record, Mapping):
        return ClaimRecord(level=str(claim_record.get("level") or ""),
                           ceiling=str(claim_record.get("ceiling") or ""),
                           coverage_mode=str(claim_record.get("coverage_mode") or ""),
                           reasons=tuple(claim_record.get("reasons") or ()))
    return ClaimRecord(level="", ceiling="", coverage_mode="")


def required_cases_ok(required_cases, evaluated,
                      ) -> Tuple[Optional[bool], Tuple[str, ...]]:
    """§6.2 条款 2 的判据面：「所有必判 case 都正确」—— 三态 + 逐 run 理由

    * `required_cases`：契约声明的 case（`contract.case_key` 字符串）
    * `evaluated`：本次**评过**的 case → 该 case 上的正确性（`True` / `False` / `None`）
      —— 键在 = 评过；值 `None` = 评过但没得出正确性结论（三态，**不当 False**）

    三条读法（登记在 CHECKLOG I92）：

      * 契约**没声明** ⇒ `None`：这不是「都通过了」，是**没有判据**。这里要是 vacuous
        truth，那「没声明」就比「声明了且都过」更容易拿到 `use_custom` —— 正好反了。
      * 声明了但本次**没评过** ⇒ `None`（未确立），理由里点名缺哪几条：§6.2 条款 2 要求
        在**全部**形状上通过，没评过的形状不是「通过」。
      * 有 case **评过且判成不正确** ⇒ `False`（唯一能给出否定的情形）。正确性不通过的
        候选本来就进不了 best-known-good，这里的 `False` 是给理由用的，不是新判决。
    """
    required = tuple(required_cases or ())
    if not required:
        return None, ("契约没有声明 required_cases（老契约 / 任务没声明）"
                      "⇒ 「所有必判 case 正确」没有判据 ⇒ 未确立",)
    if not isinstance(evaluated, Mapping):
        evaluated = {}
    missing = [c for c in required if c not in evaluated]
    if missing:
        seen = ", ".join(sorted(str(k) for k in evaluated)) or "（本次一条 case 都没记）"
        return None, (f"契约要求的 case {sorted(missing)} 本次没有评过"
                      f"（本次评过：{seen}）⇒ 「全部通过」未确立",)
    unjudged = [c for c in required if evaluated.get(c) is None]
    if unjudged:
        return None, (f"必判 case {sorted(unjudged)} 评过但没有正确性结论 ⇒ 未确立",)
    failed = [c for c in required if evaluated.get(c) is False]
    if failed:
        return False, (f"必判 case {sorted(failed)} 上正确性不通过 ⇒ §6.2 条款 2 不成立",)
    return True, (f"{len(required)} 条必判 case 本次都评过且都正确",)


def regression_over_limit(max_regression, speedups,
                          target_metric: str = "speedup_vs_baseline",
                          ) -> Tuple[Optional[bool], Tuple[str, ...]]:
    """§6.2 条款 3 的判据面：「有 case 的回退超过硬上限」—— 三态 + 逐 run 理由

    * `max_regression`：契约声明的 `((case, 允许回退比例), ...)`（也接受 dict）
    * `speedups`：本次量到的 case → 加速比（`None` = 量过但没得出数值）
    * 判据：`1 - speedup > limit` ⇒ 越限（`limit` 是**允许的回退比例**；0 = 一点都不许退；
      负值 = 要求比基线更快）

    四条读法（登记在 CHECKLOG I92）：

      * 契约**没声明**上限 ⇒ `None`（没有判据，不是「都没越限」）；
      * 声明了上限、但本次**没量到那条 case 的加速比** ⇒ `None`（未确立）—— 「没量到」
        不许当成「没越限」，否则「一次噪声读数造成永久后果」就换个方向回来了；
      * 目标指标**不是** `speedup_vs_baseline` ⇒ `None`：今天只有这一种口径，吞吐/延迟的
        「回退」要另一套换算，宁可未确立也不猜（与 §3.2 缺数值语义即拒绝同一条纪律）；
      * 上限声明在**本次没评的 case** 上 ⇒ 那一条**不进判据**（它既不越限也不阻塞），
        但理由里会说「本次没量到它」—— 拿一条没人量过的 case 去挡 `use_custom` 才是自锁。
    """
    if not max_regression:
        return None, ("契约没有声明 max_regression ⇒ 回退越限没有判据 ⇒ 未确立",)
    if not isinstance(max_regression, (Mapping, list, tuple)):
        return None, (f"max_regression 的形状读不出来 " 
                      f"({type(max_regression).__name__}) ⇒ 未确立",)
    pairs = (max_regression.items() if isinstance(max_regression, Mapping)
             else max_regression)
    limits: dict = {}
    for item in pairs:
        try:
            key, value = item
        except (TypeError, ValueError):
            return None, (f"max_regression 里有一项读不出来: {item!r} ⇒ 未确立",)
        try:
            limits[str(key)] = float(value)
        except (TypeError, ValueError):
            return None, (f"max_regression[{key!r}] 不是数值: {value!r} ⇒ 未确立",)
    if target_metric != "speedup_vs_baseline":
        return None, (f"目标指标是 {target_metric} ⇒ 今天只有 speedup_vs_baseline 口径的"
                      f"回退判据 ⇒ 未确立",)
    if not isinstance(speedups, Mapping):
        speedups = {}
    measured: dict = {}
    for case in limits:
        try:
            value = speedups.get(case)
            measured[case] = None if value is None else float(value)
        except (TypeError, ValueError):
            measured[case] = None
    unmeasured = sorted(c for c in limits if measured.get(c) is None)
    if unmeasured:
        still = sorted(c for c in limits if c not in unmeasured)
        tail = f"（本次量到加速比的：{still}）" if still else ""
        return None, (f"契约给了 case {unmeasured} 的回退上限, 但本次没有它们的加速比"
                      f" ⇒ 未确立{tail}",)
    over = [(c, 1.0 - measured[c], limits[c]) for c in sorted(limits)
            if (1.0 - measured[c]) > limits[c] + REGRESSION_SLACK]
    if over:
        detail = "、".join(f"{c} 回退 {r:.4f} > 上限 {lim:g}" for c, r, lim in over)
        return True, (f"{detail} ⇒ §6.2 条款 3「候选不得晋级」",)
    return False, ("每条声明了回退上限的 case 都在限内",)


def aggregate_speedup(case_weights, speedups, aggregate,
                      target_metric: str = "speedup_vs_baseline",
                      ) -> Tuple[Optional[float], Tuple[str, ...]]:
    """§6.2 的**目标函数**: 把逐 case 的加速比聚合成一个数（三态 + 逐 run 理由）

    * `case_weights`: 契约声明的 `((case, 权重), ...)`（也接受 dict）
    * `speedups`: 本次量到的 case → 加速比（`None` = 量过但没得出数值）
    * `aggregate`: `CASE_AGGREGATES` 之一; 空串 = 没声明

    四条读法（登记在 CHECKLOG I93）:

      * 没声明权重**或**没声明口径 ⇒ `None` —— **不替人补默认**（补一个「等权」/
        「加权几何平均」就等于替人写了目标函数, 而同一条纪律 §3.2 写过: 缺数值语义
        即拒绝）;
      * 声明了、但有一条带权重的 case **没量到** ⇒ `None`（点名缺哪几条）: 分母不全
        ⇒ 这个数没有指称对象。宁可未确立, 也不给一个「少算了最差那条」的数;
      * 几何平均要求每条加速比 **> 0**（`ln` 的定义域）⇒ 有一条 ≤ 0 就 `None`;
        `mean` / `worst_case` 不受此限 —— `worst_case` 取最差的那一条, 正是设计里
        `max_regret` 要挡的方向（总体好看不能掩盖最差的一条）;
      * 目标指标不是 `speedup_vs_baseline` ⇒ `None`（同 `regression_over_limit`:
        吞吐/延迟的聚合要另一套换算, 今天不存在）。

    ⚠️ **今天这个数是「数据」不是「门槛」**: 候选准入仍由主 case 的
    `speedup_vs_baseline >= min_speedup` 判（§16 反模式「用单形状的加速比作为成功
    标准」还没收口 —— 见 CHECKLOG I93 的仍欠那一条）。把它直接接进门槛会让
    「没声明权重的调用方」整片失守, 那是另一种自锁。
    """
    if not case_weights:
        return None, ("契约没有声明 case_weights ⇒ 目标函数没有权重 ⇒ 未确立",)
    if not aggregate:
        return None, ("契约没有声明 aggregate（聚合口径）⇒ 未确立",)
    if aggregate not in CASE_AGGREGATES:
        return None, (f"aggregate 不在词表里 {CASE_AGGREGATES}: {aggregate!r} ⇒ 未确立",)
    if not isinstance(case_weights, (Mapping, list, tuple)):
        return None, (f"case_weights 的形状读不出来 "
                      f"({type(case_weights).__name__}) ⇒ 未确立",)
    if target_metric != "speedup_vs_baseline":
        return None, (f"目标指标是 {target_metric} ⇒ 今天只有 speedup_vs_baseline "
                      f"口径的聚合 ⇒ 未确立",)
    pairs = (case_weights.items() if isinstance(case_weights, Mapping)
             else case_weights)
    weights: dict = {}
    for item in pairs:
        try:
            key, value = item
        except (TypeError, ValueError):
            return None, (f"case_weights 里有一项读不出来: {item!r} ⇒ 未确立",)
        try:
            weights[str(key)] = float(value)
        except (TypeError, ValueError):
            return None, (f"case_weights[{key!r}] 不是数值: {value!r} ⇒ 未确立",)
    if not isinstance(speedups, Mapping):
        speedups = {}
    measured: dict = {}
    for case in weights:
        try:
            value = speedups.get(case)
            measured[case] = None if value is None else float(value)
        except (TypeError, ValueError):
            measured[case] = None
    unmeasured = sorted(c for c in weights if measured.get(c) is None)
    if unmeasured:
        still = sorted(c for c in weights if c not in unmeasured)
        tail = f"（本次量到加速比的：{still}）" if still else ""
        return None, (f"契约给了 case {unmeasured} 的权重, 但本次没有它们的加速比"
                      f" ⇒ 未确立{tail}",)
    total = sum(weights.values())
    if total <= 0:
        return None, (f"case_weights 的权重之和是 {total:g} ⇒ 聚合没有分母 ⇒ 未确立",)
    if aggregate == CASE_AGGREGATE_WORST_CASE:
        worst = min(measured[c] for c in weights)
        return worst, (f"最差的一条 case 是 {worst:g}x（{len(weights)} 条带权重的 case）",)
    if aggregate == CASE_AGGREGATE_MEAN:
        value = sum(weights[c] * measured[c] for c in weights) / total
        return value, (f"加权算术平均 = {value:g}x（{len(weights)} 条 case）",)
    bad = sorted(c for c in weights if measured[c] <= 0)
    if bad:
        return None, (f"几何平均要求每条加速比 > 0, 但 {bad} 不是 ⇒ 未确立",)
    value = math.exp(sum(weights[c] * math.log(measured[c]) for c in weights) / total)
    return value, (f"加权几何平均 = {value:g}x（{len(weights)} 条 case）",)

def implementation_decision(*, eligibility=None, claim_record=None,
                            required_cases_ok: Optional[bool] = None,
                            regression_over_limit: Optional[bool] = None,
                            evidence_notes=(),
                            ) -> DecisionRecord:
    """按 §17.1 推导实现裁决（纯函数；**裁决的唯一推导处**）。

    设计原文的门槛（§17.1 末段原句）：

        `USE_CUSTOM` 要求 `ClaimLevel ≥ VALIDATED` 且所有 `required_cases` 正确；
        `USE_LIBRARY` 至少要求 `can_report`；其余一律 `INCONCLUSIVE`。

    落成**四条门**，每一门都在产物里留下理由（理由是这个字段的一半：读报告的人要能分出
    「判了：回退越限 ⇒ 用库」与「没判成 ⇒ 保守落 inconclusive」）：

    0. `can_report is not True`（没有资格位 / 位是 False / 掩码不齐）⇒ `inconclusive`：
       连「这个读数能对外报」都没确立，此时没有可裁决的对象。
    1. 三条「自定义」的证据**全中** ⇒ `use_custom`：声明等级 ≥ `validated`（§4.4）、
       `required_cases_ok is True`、`regression_over_limit is False`（§6.2 条款 3）。
    2. `regression_over_limit is True` ⇒ `use_library`：这是设计点名的三条「用库」理由里
       今天**有数据模型**的那一条（§6.2 条款 3 明写「`USE_CUSTOM` 也不成立」）。
    3. 其余 ⇒ `inconclusive`（**含 `required_cases_ok is None`**）。

    `evidence_notes`：**逐 run** 的证据来源说明（由 `required_cases_ok` / `regression_over_limit`
    两个纯函数给出，见 §6.2 条款 2/3）。它进理由**最前面**：同一个 `None` 在不同运行里
    原因不同（契约没声明 / 契约要求的 case 本次没评过 / 目标指标不是加速比 / 没量到那条
    case），读报告的人要能分出是哪一种。裁决本身**不**再解释证据来源（单一真源）。

    ⚠️ 三条读法与边界（与设计原句一起读，登记在 CHECKLOG I91 / I92）：

      * 「`USE_LIBRARY` 至少要求 `can_report`」读成**必要条件**，不是「`can_report` ⇒ 用库」。
        理由是本段自己的立论：`USE_LIBRARY` 是一个会被人拿去部署的**否定建议**，而
        「本次没量到」「契约没声明 required_cases」都不是候选不适用 —— 把它们读成
        「建议你别上自定义核」正是本段点名的失效模式。所以门 3 才是默认落点。
      * `required_cases_ok is False`（有的必判 case 不正确）今天也落 `inconclusive`：
        设计给 `USE_LIBRARY` 的三条理由（稳定慢 / 维护成本 / 回退越限）都**预设自定义
        实现是正确的**，而正确性不成立的候选本来就进不了 best-known-good。
      * 「在目标 workload 上稳定快」「维护成本不划算」**不在这里再判一遍**：前者是
        `ClaimLevel` 的阶梯（`validated` 要正确性凭据、`promotable` 要独立确认 —— §4.4），
        后者今天没有数据模型。单一真源，不重造。

    三态纪律：`required_cases_ok` / `regression_over_limit` 的 `None`（未确立）既不算
    「坏」也不算「好」—— 它挡住 `use_custom`，但也不给出 `use_library` 的依据。
    """
    elig = eligibility
    if elig is not None and not isinstance(elig, Eligibility):
        elig = Eligibility.from_mapping(elig)
    rec = _claim_record_of(claim_record)
    level = rec.level or None
    can_report = getattr(elig, "can_report", None) if elig is not None else None
    # 逐 run 的证据来源说明排在最前：同一个 `None` 在不同运行里原因不同（见 docstring）。
    reasons: list = [str(n) for n in (evidence_notes or ())]

    def _record(decision: str, more) -> DecisionRecord:
        return DecisionRecord(decision=decision, claim_level=rec.level,
                              claim_ceiling=rec.ceiling, can_report=can_report,
                              required_cases_ok=required_cases_ok,
                              regression_over_limit=regression_over_limit,
                              reasons=tuple(reasons) + tuple(more))

    if can_report is not True:
        why = "没有资格位（没经过收口点）" if elig is None else f"can_report={elig.can_report}"
        return _record(DECISION_INCONCLUSIVE,
                       [f"资格位 {why} ⇒ 连「这个读数能对外报」都没有确立 "
                        f"⇒ 判不了 (门 0)"])

    blockers: list = []
    if not at_least(level, CLAIM_VALIDATED):
        blockers.append("声明等级 " + (f"{level} < {CLAIM_VALIDATED}" if level
                                   else "缺失（没有 §4.4 的记录）") + " (门 1)")
    if required_cases_ok is not True:
        blockers.append("required_cases 不是全对：" +
                        ("未确立（逐 run 的原因见「证据说明」）" if required_cases_ok is None
                         else "有必判 case 不正确") + " (门 1)")
    if regression_over_limit is not False:
        blockers.append("回退越限不是「没有」：" +
                        ("未确立（逐 run 的原因见「证据说明」）"
                         if regression_over_limit is None else "有回退越限") +
                        " (§6.2 条款 3)")
    if not blockers:
        return _record(DECISION_USE_CUSTOM,
                       ["声明等级 ≥ validated、required_cases 全对、没有回退越限 "
                        "⇒ 三条证据全中 (门 1)"])
    if regression_over_limit is True:
        return _record(DECISION_USE_LIBRARY,
                       blockers + ["有回退越限 ⇒ §6.2 条款 3「候选不得晋级」⇒ 自定义不成立 "
                                   "⇒ 建议用库 (门 2)"])
    reasons.extend(blockers)
    return _record(DECISION_INCONCLUSIVE,
                   ["「自定义」的三条证据没有一条被否定、也没有一条被确立 ⇒ "
                    "既上不了 use_custom, 也没有「建议用库」的依据 (门 3)"])


