"""测量有效性判定 (docs/research/00 缺口 1, 设计见 docs/research/01 第五节)

L3 的数字有两种「不可信」, 而它们都不该被当成「性能不达标」:

1. **launch 地板支配** —— 实测时间几乎全是 kernel launch 开销, 描述的是驱动派发
   而不是核的质量。实测: layernorm 1024×768 读写 3.15 MB, roofline 下界 0.86 µs,
   而 launch 地板 18.69 µs → 地板占比 95.6%。那一轮报出的「0.65x」「带宽利用率
   5.4%」「memory-bound 需优化」全部是在描述 launch 开销。决定性旁证是
   torch_compile 反而比 eager 慢 2.5× —— 编译开销在这个尺寸根本摊不平。
2. **基准被污染** —— 测量窗口内 GPU 上有别的进程在跑, 加速比不可比。

`VerificationResult.passed` 只有两态, 于是这两种情况都被记成失败, 再被 Conductor
拿去禁用配方、累计停滞轮次。已记录的后果: matmul 那轮触发污染 (测量前 util=14%)
→ gflops 被压低 → 判「未改善」→ 把**实际有效**的配方拉黑; 而
`max_stagnant_rounds=2`, 两次误判就终止整个搜索。这台机器上恒有 7 个其他进程,
不是边缘情况。

本模块只做判定: 不含硬件数据 (launch 地板由调用方从 hardware_caps 取), 不做 I/O,
不依赖 torch。定位与 `tolerance.py` 相同 —— 单一权威判据 + 可离线单测。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Validity(str, Enum):
    """测量有效性的三个档位 (docs/research/01 表, 第五节第 2 条)"""

    VALID = "valid"                    # 可参与全部闭环决策
    DEGRADED = "degraded"              # 判定可用, 但不得驱动配方禁用/停滞计数
    UNINFORMATIVE = "uninformative"    # 只作诊断信息呈现, 不参与任何闭环决策


# launch_dominance = 地板 / 候选 median 时间。
# 0.10 与 0.30 是 docs/research/01 第二节的经验分界 (<10% 时实测/下界已收敛到
# 1.1×; >30% 时曲线仍完全平坦), **不是推导值**, 改动需回归验证。
LAUNCH_DOMINANCE_VALID = 0.10
LAUNCH_DOMINANCE_UNINFORMATIVE = 0.30

# 沿用 verifier 里既有的硬污染阈值 (>5%), 不另立新数。
# 只统计 util 而不统计「有其他进程」: 这台机器上恒有 7 个其他进程, 若把「有进程」
# 也算污染, 每次运行都会被标记, 信号就失效了。
CONTAMINATION_UTIL_PCT = 5.0


@dataclass
class MeasurementAssessment:
    """一次 L3 测量的有效性判定结果

    三个 can_* 是给 Conductor 用的: 它不需要知道地板和 util 是多少, 只需要知道
    「这个数字能不能拿去做这个决定」。把用途而不是档位暴露出去, 是为了让调用方
    不必在自己的代码里重新解释一遍分档表 (那正是缺口 1 原来的失效方式:
    verifier 算出了 contaminated, 却没有任何下游读它)。
    """

    validity: Validity
    reasons: List[str] = field(default_factory=list)
    launch_dominance: Optional[float] = None     # None = 地板未知, 无从判定
    launch_floor_us: Optional[float] = None
    net_of_floor_time_ms: Optional[float] = None  # 扣除地板后的净时间
    contaminated: bool = False

    @property
    def can_gate(self) -> bool:
        """能否用于 L3 的通过/不通过判定"""
        return self.validity is not Validity.UNINFORMATIVE

    @property
    def can_disable_recipe(self) -> bool:
        """能否据此禁用配方 —— 只有 VALID 可以"""
        return self.validity is Validity.VALID

    @property
    def can_count_stagnation(self) -> bool:
        """能否据此累计停滞轮次 / 触发提前终止 —— 只有 VALID 可以"""
        return self.validity is Validity.VALID

    def as_metrics(self) -> dict:
        """摊平进 VerificationResult.metrics 的部分"""
        out = {
            "measurement_validity": self.validity.value,
            "measurement_validity_reasons": list(self.reasons),
            "measurement_can_gate": self.can_gate,
            "measurement_can_disable_recipe": self.can_disable_recipe,
            "measurement_can_count_stagnation": self.can_count_stagnation,
        }
        if self.launch_dominance is not None:
            out["launch_dominance"] = round(self.launch_dominance, 4)
        if self.launch_floor_us is not None:
            out["launch_floor_us"] = self.launch_floor_us
        if self.net_of_floor_time_ms is not None:
            out["net_of_floor_time_ms"] = round(self.net_of_floor_time_ms, 5)
        return out


def launch_dominance(launch_floor_us: Optional[float], median_time_ms: Optional[float]) -> Optional[float]:
    """地板占候选 median 时间的比例; 任一侧缺失或非正则返回 None (无从判定)"""
    if not launch_floor_us or launch_floor_us <= 0:
        return None
    if not median_time_ms or median_time_ms <= 0:
        return None
    return (launch_floor_us / 1000.0) / median_time_ms


def net_of_floor(launch_floor_us: Optional[float], median_time_ms: Optional[float]) -> Optional[float]:
    """扣除 launch 地板后的净时间 (ms), 下限截到 0

    前提是「候选与基线的 launch 开销相同」。对 Triton vs cuBLAS 这个假设并不严格
    成立 (两者的派发路径不同, 真实差值未测, 见 docs/research/01 第六节未决问题 2),
    所以净值只作为第二指标呈现, 不替代实测值。
    """
    dom = launch_dominance(launch_floor_us, median_time_ms)
    if dom is None:
        return None
    return max(0.0, median_time_ms * (1.0 - dom))


def net_of_floor_speedup(launch_floor_us: Optional[float],
                         candidate_ms: Optional[float],
                         baseline_ms: Optional[float]) -> Optional[float]:
    """扣除地板后的加速比 (第二指标)。前提同 net_of_floor。

    任一侧本身就被地板支配时返回 None。从一个几乎全是 launch 开销的时间里减掉
    地板, 剩下的是噪声而不是计算时间: 实测 layernorm 1024×768 的 PyTorch Eager
    基线 median=20.6 µs 而地板 20.24 µs, 净剩 0.36 µs —— 低于该 shape 0.86 µs
    的 roofline 下界, 物理上不可能, 说明差值全是噪声。报一个由噪声相除得来的
    数字比不报更糟。
    """
    for ms in (candidate_ms, baseline_ms):
        dom = launch_dominance(launch_floor_us, ms)
        if dom is None or dom > LAUNCH_DOMINANCE_UNINFORMATIVE:
            return None
    c = net_of_floor(launch_floor_us, candidate_ms)
    b = net_of_floor(launch_floor_us, baseline_ms)
    if not c or not b:
        return None
    return b / c


def assess(median_time_ms: Optional[float] = None,
           launch_floor_us: Optional[float] = None,
           gpu_util_pct: Optional[float] = None) -> MeasurementAssessment:
    """综合两个信号给出有效性判定, 取更差的一档

    污染判为 DEGRADED 而不是 UNINFORMATIVE: `benchmark_with_baselines` 是同机同卡
    依次测候选与各基线的, 两者落在同一个污染窗口里, 比值仍保留部分意义; 但绝对
    gflops 与比值都被扰动过, 不足以支撑「这个配方没用」这种结论。
    """
    reasons: List[str] = []
    validity = Validity.VALID

    dom = launch_dominance(launch_floor_us, median_time_ms)
    if dom is None:
        # 地板未知 (hardware_caps 里没有这台设备的实测值) 时不猜。缺证据不等于无效,
        # 但必须显式说出来, 否则「没检查」会被读成「检查过了」。
        if launch_floor_us is None:
            reasons.append("launch 地板未知, 未做地板支配检查")
    elif dom > LAUNCH_DOMINANCE_UNINFORMATIVE:
        validity = Validity.UNINFORMATIVE
        reasons.append(
            f"launch 地板占实测时间 {dom * 100:.1f}% (> {LAUNCH_DOMINANCE_UNINFORMATIVE * 100:.0f}%): "
            f"数字描述的是驱动派发开销, 不是核质量; 换更大的 shape")
    elif dom > LAUNCH_DOMINANCE_VALID:
        validity = Validity.DEGRADED
        reasons.append(
            f"launch 地板占实测时间 {dom * 100:.1f}% (> {LAUNCH_DOMINANCE_VALID * 100:.0f}%): "
            f"判定可用, 但不得据此禁用配方或累计停滞")

    contaminated = gpu_util_pct is not None and gpu_util_pct > CONTAMINATION_UTIL_PCT
    if contaminated:
        reasons.append(f"测量前 GPU util={gpu_util_pct}%, 基准被污染, 加速比不可比")
        if validity is Validity.VALID:
            validity = Validity.DEGRADED

    return MeasurementAssessment(
        validity=validity,
        reasons=reasons,
        launch_dominance=dom,
        launch_floor_us=launch_floor_us,
        net_of_floor_time_ms=net_of_floor(launch_floor_us, median_time_ms),
        contaminated=contaminated,
    )
