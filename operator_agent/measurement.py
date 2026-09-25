"""测量有效性判定 (docs/research/00 缺口 1, 设计见 docs/research/01 第五节)

L3 的数字有三种「不可信」, 而它们都不该被当成「性能不达标」:

1. **launch 地板支配** —— 实测时间几乎全是 kernel launch 开销, 描述的是驱动派发
   而不是核的质量。实测: layernorm 1024×768 读写 3.15 MB, roofline 下界 0.86 µs,
   而 launch 地板 18.69 µs → 地板占比 95.6%。那一轮报出的「0.65x」「带宽利用率
   5.4%」「memory-bound 需优化」全部是在描述 launch 开销。决定性旁证是
   torch_compile 反而比 eager 慢 2.5× —— 编译开销在这个尺寸根本摊不平。
2. **基准被污染** —— 测量窗口内 GPU 上有别的进程在跑, 加速比不可比。证据是测量
   **前**的 util 快照 (且须排掉自己上一轮的残留, 见 docs/research/33)。最慢样本/
   中位样本本来是想补的第二条证据 (窗口**中途**的抢占, util 快照看不见), 但实测
   它被样本数系统性抬高, 目前只作诊断。
3. **样本自身在抖** —— 30 个样本的 iqr/median 大到超过被测差异的量级, median
   看着权威其实是噪声。这条以前完全没检查: runner 跑了 n 个样本, 只报 min/median/
   avg/max, 从不看它们散不散 (决策与出处: docs/research/20 §3.1 D1)。

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


class Verdict(str, Enum):
    """两轮性能之间的四态判决 (吸纳自测量方法论一系, docs/research/20 第三节 D4)

    二元的「改善 / 没改善」把两件不同的事压成一件:
      - 真的更慢 (有证据说明这轮的配方有害)
      - 落在噪声里 (没有证据, 只是这次没测出差别)
    前者才可以禁用配方 —— 那是**永久**决策; 后者只该累计停滞轮次。原来的写法是
    `current > best` 的严格比较, 于是「完全打平」和「差 0.0001%」都会去禁用配方。
    """

    FAST = "faster"            # 超出带宽的改善
    SAME = "same"              # 落在带宽内: 计停滞, 但不据此禁用配方
    SLOW = "slower"            # 超出带宽的退步: 唯一可以禁用配方的一态
    AMBIGUOUS = "ambiguous"    # 证据不足 (没有可比前值, 或测量本身不可信)


# 判决带宽 = 3 × 实测的「同一份代码连跑多次的加速比散布」。
#
# 实测值来自 H20 GPU 2、无注入、每档连跑 8 次 L3 (docs/research/34 第四节表):
#   valid (matmul 4096³)         跨次 iqr 0.02%
#   degraded (layernorm 16384×6144)   0.13%
#   uninformative (layernorm 1024×768) 0.66% (范围宽到 2.1%)
#
# 外部 (SOL-ExecBench 一系) 的 0.5%/2% 是锁频专用机上的值, 不能照抄: 0.5% 在 valid
# 档是真实噪声底的 25 倍 (会把真的小改善判成「没变」), 在 uninformative 档又比实测
# 范围 2.1% 还窄 (一切都落 SAME/AMBIGUOUS) —— 这正是 doc 20 当初暂缓 D4 的理由,
# 现在有了本机的数就按档分开设。
NOISE_BAND = {
    Validity.VALID: 0.001,
    Validity.DEGRADED: 0.004,
    Validity.UNINFORMATIVE: 0.02,
}

# 绑定轴上达到 roofline 上界的这个比例, 就认为「再优化也不会有 measurable 收益」。
# 缺口 4 的正解在这里: 判据该看**离上界还有多远**, 而不是继续拿一个标量比大小。
# 0.90 是经验值 (真机上 matmul 4096³ fp16 的 Triton 核实测 101% GFLOPS 利用率,
# 峰值表本身有 ±5% 的余量), 改动需回归验证。
ROOFLINE_CONVERGED_PCT = 90.0


def binding_attainment(metrics: dict) -> Optional[float]:
    """绑定轴上的 roofline 达成率 (%), 拿不到返回 None

    memory-bound 看带宽利用率、compute-bound 看 GFLOPS 利用率。取两者的 **max**
    而不是按 bound 分支选一个: roofline 的语义就是「离上界多远由紧的那条轴决定」,
    分支选指标会在两条轴接近时选错 (而 `roofline_bound` 这个字符串本身还有过
    memory/compute 与 name/bound 两套拼法飘掉的历史, 见 docs/research/15 第五节)。
    """
    vals = []
    for key in ("gflops_utilization_pct", "bandwidth_utilization_pct"):
        v = metrics.get(key)
        if isinstance(v, (int, float)) and v >= 0:
            vals.append(float(v))
    return max(vals) if vals else None


def roofline_converged(metrics: dict) -> bool:
    """已经贴着 roofline 上界 —— 继续换配方只会白烧 GPU 与 LLM token"""
    if (metrics.get("measurement_can_gate") is False
            or metrics.get("benchmark_contaminated")):
        return False          # 不可信的测量不能说「已经到顶」
    a = binding_attainment(metrics)
    return a is not None and a >= ROOFLINE_CONVERGED_PCT


def verdict(current: Optional[float], previous: Optional[float],
            validity: Validity) -> tuple[Verdict, str]:
    """两轮性能值的四态判决。current/previous 是同一轴上的标量 (越大越好)。

    打平不算退步: `SAME` 与 `SLOW` 的区别就是「能不能禁用配方」的分界。
    """
    if current is None or previous is None or current <= 0 or previous <= 0:
        return Verdict.AMBIGUOUS, "无可比的性能值 (缺 current 或 previous)"
    if validity is Validity.UNINFORMATIVE:
        return Verdict.AMBIGUOUS, "测量档位 uninformative: 判决带宽比信号本身宽"
    band = NOISE_BAND[validity]
    delta = (current - previous) / previous
    if delta > band:
        return Verdict.FAST, f"+{delta:.4%} > 带宽 {band:.2%}"
    if delta < -band:
        return Verdict.SLOW, f"{delta:.4%} < -带宽 {band:.2%}"
    return Verdict.SAME, f"{delta:+.4%} 落在带宽 ±{band:.2%} 内"


# launch_dominance = 地板 / 候选 median 时间。
# 0.10 与 0.30 是 docs/research/01 第二节的经验分界 (<10% 时实测/下界已收敛到
# 1.1×; >30% 时曲线仍完全平坦), **不是推导值**, 改动需回归验证。
LAUNCH_DOMINANCE_VALID = 0.10
LAUNCH_DOMINANCE_UNINFORMATIVE = 0.30

# 沿用 verifier 里既有的硬污染阈值 (>5%), 不另立新数。
# 只统计 util 而不统计「有其他进程」: 这台机器上恒有 7 个其他进程, 若把「有进程」
# 也算污染, 每次运行都会被标记, 信号就失效了。
CONTAMINATION_UTIL_PCT = 5.0

# 样本离散度 iqr/median 的门限 (阈值由真机标定得出: docs/research/20 第四节)。
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
# ⚠️ 比值同时依赖核的绝对时长和样本数, 因此目前只作诊断、不改 validity 档位。
# 真机数据 (H20, 2026-09-26, docs/research/33):
#
#     核                    median   实际样本数 runs   max/median
#     layernorm 1024×768    7.2 µs        750         19.17
#     layernorm 16384×6144  164 µs        239          1.86   ← util=0%, 无邻居插队
#     matmul 4096³ fp16     980 µs         48          1.12
#
# 原先 benchmarks/calibrate_dispersion_thresholds.py 标定时刻意固定 30 样本/轮,
# 所以只看见第 1 层偏差 (绝对时长), 没看见第 2 层 (顺序统计量随 n 抬升) —— 那是
# 标定脚本自身的方法缺陷。阈值保留在此, 供 reasons 里那句观察引用; 要重新赋予判据
# 地位, 须先换成对 n 稳健的统计量。
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
    intruder_suspected: bool = False             # 仅诊断; 为何不作判据见 assess() 内说明

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
        out["measurement_intruder_suspected"] = self.intruder_suspected
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
    """综合判定测量有效性: 三个信号取更差的一档, 第四个只作观察

    参与判定的三个: launch 地板支配 (`median_time_ms` vs `launch_floor_us`)、
    测量前 util 污染 (`gpu_util_pct`)、样本主体离散度 (`dispersion` = iqr/median)。

    污染判为 DEGRADED 而不是 UNINFORMATIVE: `benchmark_with_baselines` 是同机同卡
    依次测候选与各基线的, 两者落在同一个污染窗口里, 比值仍保留部分意义; 但绝对
    gflops 与比值都被扰动过, 不足以支撑「这个配方没用」这种结论。

    离散度同理: iqr/median 超过 0.10 说明这台机器此刻在抖, median 本身仍是无偏的,
    但它的精度撑不住「加速比 1.02x」这种量级的结论。

    max_over_median **不改变档位**, 只写进 reasons 并置 `intruder_suspected` —— 它被
    顺序统计量效应污染, 实测干净测量就已打到 1.86。详见 INTRUDER_MAX_OVER_MEDIAN
    注释与 docs/research/33。
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

    # max_over_median **只作观察, 不改变 validity 档位**。
    #
    # 真机验证 (docs/research/33) 推翻了把它当判据的设计。两层偏差同向叠加:
    #   1. 绝对时长 —— 一次 ~30 µs 的宿主抖动对 4.5 µs 的核是 18 倍, 对 1 ms 的核
    #      只有 1.04 倍;
    #   2. 顺序统计量 —— runner 按毫秒预算自动加采样 (实测 runs = 750/239/48),
    #      样本越多 max 越靠尾部。而能通过「核足够长」这个前提的核必然净时间
    #      >= 9x 地板 ~= 182 µs, 那个长度正好换来 ~274 个样本 —— 用得上这条判据的
    #      区间, 恰好是 max 被抬得最高的区间。
    # 实测 layernorm 16384x6144 在 util=0%、完全没有邻居插队时就已报出 1.86,
    # 逼近阈值 2.0。拿它当判据会系统性地把干净测量判成污染, 即把闭环静默关掉 ——
    # 那正是缺口 1 最初的病。
    #
    # 仍写进 reasons: 缺口 1 的教训是「算了没人读」, reasons 会进 Fixer /
    # SearchAgent 的 prompt。但措辞必须自我声明不影响判定, 否则下游会当判据读。
    # 想救回这条判据得换成对 n 稳健的形式 (如「> 2xmedian 的样本占比」)。
    intruder = (max_over_median is not None
                and max_over_median > INTRUDER_MAX_OVER_MEDIAN)
    if intruder:
        reasons.append(
            f"最慢样本是中位样本的 {max_over_median:.1f} 倍 (阈值 "
            f"{INTRUDER_MAX_OVER_MEDIAN:.0f}, 实测样本数越多该值天然越高): "
            f"窗口内可能有抢占, 也可能只是长尾; **仅诊断, 不影响本轮判定**")

    return MeasurementAssessment(
        validity=validity,
        reasons=reasons,
        launch_dominance=dom,
        launch_floor_us=launch_floor_us,
        contaminated=contaminated,
        dispersion=dispersion,
        max_over_median=max_over_median,
        intruder_suspected=intruder,
    )
