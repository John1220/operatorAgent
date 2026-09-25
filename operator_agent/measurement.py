"""测量有效性判定 (docs/research/00 缺口 1, 设计见 docs/research/01 第五节)

L3 的数字有三种「不可信」, 而它们都不该被当成「性能不达标」:

1. **launch 地板支配** —— 实测时间几乎全是 kernel launch 开销, 描述的是驱动派发
   而不是核的质量。实测: layernorm 1024×768 读写 3.15 MB, roofline 下界 0.86 µs,
   而 launch 地板 18.69 µs → 地板占比 95.6%。那一轮报出的「0.65x」「带宽利用率
   5.4%」「memory-bound 需优化」全部是在描述 launch 开销。决定性旁证是
   torch_compile 反而比 eager 慢 2.5× —— 编译开销在这个尺寸根本摊不平。
2. **基准被污染** —— 测量窗口内 GPU 上有别的进程在跑, 加速比不可比。两条互补的
   证据: 测量**前**的 util 快照 (硬污染), 以及最慢样本/中位样本 (窗口**中途**插
   进来的抢占 —— util 快照看不见它, 而 iqr/median 对单个离群样本也完全不敏感)。
3. **样本自身在抖** —— 30 个样本的 iqr/median 大到超过被测差异的量级, median
   看着权威其实是噪声。这条以前完全没检查: runner 跑了 n 个样本, 只报 min/median/
   avg/max, 从不看它们散不散 (docs/research/17 第二节第 1 条)。

`VerificationResult.passed` 只有两态, 于是这三种情况都被记成失败, 再被 Conductor
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

# 样本离散度 iqr/median 的门限 (docs/research/17 第二节第 1 条)。
#
# 0.10 取自 PyTorch `torch.utils.benchmark` 的 `_IQR_WARN_THRESHOLD` (✅ 一手核实
# `torch/utils/benchmark/utils/common.py`), 它是「结果可以报到几位有效数字」的判据。
# NVBench 的 `same.relative_dispersion_ceiling` 是 0.02 (✅ 默认 TOML), 但那是**锁频
# 专用机**的值; 本机是 8 卡共享、恒有 7 个其他进程, 直接套 0.02 会让几乎所有测量
# 落到最严的一档, 信号同样失效 —— 与上面 util 阈值只取 5% 是同一个理由。
#
# 0.30 这一档没有外部锚点, 是按「与 launch dominance 的 0.10/0.30 保持同一节奏」
# 取的, 属经验值, 改动需回归验证。
DISPERSION_DEGRADED = 0.10
DISPERSION_UNINFORMATIVE = 0.30

# 「测量窗口内发生过抢占」的旁证: 最慢样本 / 中位样本。
#
# iqr/median 对这个完全不敏感 (29 个 1.0 + 1 个 50.0 → iqr = 0), 而当决策统计量是
# median 时这正是想要的性质。问题在于 gpu_util 只在测量**前**采一次快照, 窗口中途
# 插进来的邻居进程它同样看不见 —— 于是「测量期间发生过抢占」这件事以前没有任何
# 证据通道。max/median 是唯一的廉价旁证。
#
# ⚠️ 比值**强烈依赖核的绝对时长**。真机标定 (benchmarks/calibrate_dispersion_thresholds.py,
# H20, 30 轮 × 30 样本 × 4 个核, 2026-09-25):
#
#     核                        median      max/median 的 p50 / max
#     空核                       4.5 µs        3.84 / 18.26
#     layernorm 1024×768         8.4 µs        4.31 /  8.00
#     layernorm 16384×6144       256 µs        1.04 /  1.38
#     matmul 4096³ fp16         1030 µs        1.00 /  1.04
#
# 一次固定量级的宿主抖动 (~30 µs) 对 4.5 µs 的空核是 18 倍, 对 1 ms 的 matmul 只有
# 1.04 倍 —— 单一绝对阈值必然两头不讨好。解法不是继续调阈值, 而是**只在比值有意义的
# 区间用它**: 短核本来就被 launch 地板支配判成 UNINFORMATIVE, 那时 max/median 大是
# 常态、没有额外信息量; 只在 launch_dominance <= LAUNCH_DOMINANCE_VALID (核足够长)
# 时启用。该区间内实测最坏值 1.38, 取 2.0 留约 45% 余量。
INTRUDER_MAX_OVER_MEDIAN = 2.0


def _worse(a: "Validity", b: "Validity") -> "Validity":
    """取更差的一档。三个信号各自独立判定, 合成时不做加权 —— 任一项不可信,
    整个数字就不该拿去做闭环决策。"""
    order = {Validity.VALID: 0, Validity.DEGRADED: 1, Validity.UNINFORMATIVE: 2}
    return a if order[a] >= order[b] else b


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
    contaminated: bool = False
    dispersion: Optional[float] = None           # iqr/median, None = 样本不足或未上报
    max_over_median: Optional[float] = None      # 窗口内抢占的旁证, 与 dispersion 互补

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
        if self.dispersion is not None:
            out["dispersion_iqr_over_median"] = self.dispersion
        if self.max_over_median is not None:
            out["dispersion_max_over_median"] = self.max_over_median
        return out


def launch_dominance(launch_floor_us: Optional[float], net_time_ms: Optional[float]) -> Optional[float]:
    """端到端时间里 launch 开销占的比例; 任一侧缺失或非正则返回 None (无从判定)

    runner 报的是**冲刷 L2 后**的净执行时间 (Triton `do_bench` 口径), 事件窗口里
    已经不含 GPU 空等 launch 命令的那一段。而用户真正付出的端到端时间是
    「launch 开销 + 净执行」, 所以占比是 `floor / (floor + net)`。

    不能用 `floor / net`: 真机标定 (benchmarks/calibrate_l2_flush_effect.py, H20,
    torch 2.14.0+cu130 / triton 3.8.0) 显示同一个空 Triton 核

        不冲刷 median = 20.03 µs      冲刷 median = 4.64 µs

    也就是说「20.24 µs 的 launch 地板」量的其实是**空闲 GPU 上事件窗口吃进的
    launch 延迟**, 不是一个恒定的执行开销 —— 用 do_bench 的 256 MB 冲刷把 GPU
    填满后, 同一个空核只剩 4.64 µs。冲刷前后是两个口径, 拿旧口径的地板去除新
    口径的净时间会算出 >1 的值 (实测 layernorm 1024×768 得到 2.81), 那正是口径
    混用的信号。
    """
    if not launch_floor_us or launch_floor_us <= 0:
        return None
    if not net_time_ms or net_time_ms <= 0:
        return None
    floor_ms = launch_floor_us / 1000.0
    return floor_ms / (floor_ms + net_time_ms)


def assess(median_time_ms: Optional[float] = None,
           launch_floor_us: Optional[float] = None,
           gpu_util_pct: Optional[float] = None,
           dispersion: Optional[float] = None,
           max_over_median: Optional[float] = None) -> MeasurementAssessment:
    """综合四个信号给出有效性判定, 取更差的一档

    污染判为 DEGRADED 而不是 UNINFORMATIVE: `benchmark_with_baselines` 是同机同卡
    依次测候选与各基线的, 两者落在同一个污染窗口里, 比值仍保留部分意义; 但绝对
    gflops 与比值都被扰动过, 不足以支撑「这个配方没用」这种结论。

    离散度同理: 30 个样本的 iqr/median 超过 0.10 说明这台机器此刻在抖, median
    本身仍是无偏的, 但它的精度撑不住「加速比 1.02x」这种量级的结论。

    max_over_median 也判 DEGRADED 而不是更差: median 对少量离群样本是稳的, 所以
    这个信号说明的是「窗口内有抢占」, 而不是「median 不可信」。
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
        validity = _worse(validity, Validity.DEGRADED)

    if dispersion is not None and dispersion >= 0:
        if dispersion > DISPERSION_UNINFORMATIVE:
            validity = _worse(validity, Validity.UNINFORMATIVE)
            reasons.append(
                f"样本离散度 iqr/median={dispersion:.3f} "
                f"(> {DISPERSION_UNINFORMATIVE:.2f}): 抖动量级已超过被测差异, "
                f"这一轮的数字无法支撑任何结论; 重测或换更长的 shape")
        elif dispersion > DISPERSION_DEGRADED:
            validity = _worse(validity, Validity.DEGRADED)
            reasons.append(
                f"样本离散度 iqr/median={dispersion:.3f} "
                f"(> {DISPERSION_DEGRADED:.2f}): median 仍无偏, 但精度撑不住小差异, "
                f"不得据此禁用配方或累计停滞")

    # 只在「核足够长、launch 地板不支配」的区间里用比值判抢占。短核的 max/median
    # 天然就是几倍到十几倍 (真机实测空核 p50=3.84), 而那个区间已被上面的地板门
    # 判成 UNINFORMATIVE, 再加一条只会把常态误报成污染。阈值来源见常量注释。
    intruder_regime_ok = dom is not None and dom <= LAUNCH_DOMINANCE_VALID
    intruder = (intruder_regime_ok and max_over_median is not None
                and max_over_median > INTRUDER_MAX_OVER_MEDIAN)
    if intruder:
        reasons.append(
            f"最慢样本是中位样本的 {max_over_median:.1f} 倍 "
            f"(> {INTRUDER_MAX_OVER_MEDIAN:.0f}): 测量窗口内有进程抢占, "
            f"而测量前的 util 快照看不见它; median 仍可用, 但不足以支撑小差异结论")
        validity = _worse(validity, Validity.DEGRADED)

    return MeasurementAssessment(
        validity=validity,
        reasons=reasons,
        launch_dominance=dom,
        launch_floor_us=launch_floor_us,
        contaminated=contaminated,
        dispersion=dispersion,
        max_over_median=max_over_median,
    )
