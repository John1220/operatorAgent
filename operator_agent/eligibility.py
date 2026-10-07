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
from typing import Dict, Mapping, Optional, Tuple

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
    `PrecompileVerdict.coverage` 把 `not_applicable` 与 `unknown` 一起收进
    `uncovered_rules`，而 §21.4 第 7 条（引用不存在的能力）对**每一份** patch 候选都是
    `not_applicable`（仓里没有任何语言的 API 面清单）。拿它压级 ⇒ 每一份候选都到不了
    `promotable` ⇒ 入库率归零，而那把尺子拦的是**量具缺失**，不是候选的问题。
    `mode` 说的是「这份源码的静态检查能力」，那才是 §22.3 的判据面；
    `uncovered_rules` 作为证据进报告（`confirm.UnableToEstablish.extend_metrics`）。
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

