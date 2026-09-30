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
from typing import Optional

# §4.3 第 5 条的常数: value > 实测峰值 × 1.05 判为「测量错了」而不是「很棒」。
# 这里只钉住常数本身; 阈值比较在调用方 (或 exceeds_measured_peak) 做, 免得资格推导
# 再去读硬件数据。
PEAK_SLACK = 1.05


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


#: 全 False —— 规则 1~5: 这个数没有指称对象 / 测的是别的东西 / 物理上不可能
ALL_FALSE = Eligibility(False, False, False, False, False)
#: DEGRADED 位型 (规则 6~7): 判决 / 调参 / 报数可以, 持久化 / 计进度不行
DEGRADED = Eligibility(True, True, False, True, False)
#: 全 True —— 规则 8: VALID
ALL_TRUE = Eligibility(True, True, True, True, True)


def exceeds_measured_peak(value: Optional[float],
                          measured_peak: Optional[float],
                          slack: float = PEAK_SLACK) -> bool:
    """`value > 实测峰值 × (1+slack)`? 任一侧缺失则不触发 (缺证据不等于超上界)

    阈值用**实测**峰值而不是 spec 表: roofline 在这里是校验边界 (§4.2 第 3 条),
    报出的 FLOPS 超过峰值说明测量错了, 不是说明很棒 —— 分母若来自规格表, 一次
    钟频拉满的读数就可能合法地「超上界」, 那样这条检查会变成假警报发生器。
    """
    if value is None or measured_peak is None:
        return False
    if measured_peak <= 0:
        return False
    return value > measured_peak * slack


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
    lib_fallback: bool = False,
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
    4  检测到 lib fallback                        → 全 False (它在测别的东西)
    5  value > 实测峰值 × 1.05 / 档位已是 uninformative → 全 False
    6  cv 超阈值 或 launch 下限占比超阈值          → DEGRADED (判决/调参/报数, 不持久化/不计进度)
    7  partition 漂移 / 档位已是 degraded          → DEGRADED (同 6 的位)
    8  其余                                       → VALID, 全 True

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
    if l3_pass_is_refused or (requires_real_device and real_device is False):
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

    # 规则 8
    return ALL_TRUE
