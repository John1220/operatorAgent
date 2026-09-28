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

上面三条问的都是「**测得准不准**」。2026-09-28 又补了两条问「**测的是不是这个东西**」
(doc 49 §3.5 第 1/2/3 处, 那一节记的失效方式叫「算了没人读」):

4. **出处不对** —— 数字可能压根不是测出来的: `MockGPU` 无 CUDA 时按硬件规格**估**
   (`"mode": "simulated"`), `LocalExecutor` 在控制机 CPU 上真测 (那不是加速卡的成绩)。
   两者以前都在产物里与真机读数**权威性等价**。判据走 `backend_caps` 的声明而不是
   字符串匹配 `device`/`name`: 任务在 `target.required_caps` 里声明它要什么, 执行侧
   声明它有什么, 缺就是 UNINFORMATIVE 且 L3 不许 `passed=True`。载荷自报的
   `mode`/`device` 只用作**与声明对照**的第二手证据 (两者打架说明连声明都不可信)。
5. **环境漂移** —— 器件名/torch/triton 版本/执行侧身份与上一次运行不一致时, 这一轮的
   加速比与上一轮不可比。按本项目「观测先于判定」的处置: **只降到 DEGRADED**
   (不禁用配方、不计停滞), 不据此判核失败, 也没有阈值 —— 一份「变了哪些键」的清单
   还没有分布支撑, 现在定阈值就是编一个门。

`VerificationResult.passed` 只有两态, 于是前三种情况都被记成失败, 再被 Conductor
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

from .backend_caps import Cap


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
# 0.90 这个数以前写的理由是「真机上 matmul 4096³ fp16 实测 101% GFLOPS 利用率,
# 峰值表本身有 ±5% 余量」—— 那条**已经不成立了** (2026-09-26 核实): 101% 不是余量,
# 是分母用错了对象 (拿 cuBLAS 8192³ 的一次读数当上界)。分母换成 dense 规格 148 TFLOPS
# 后同一个核真机复测 **94.7%**, 于是 90 的含义从「比库实现的成绩还差一点」变成
# 「离规格上限还差 10%」。
#
# ⚠️ 两条轴的可达性因此**不对称**, 别把它当一个通用常数读 (2026-09-26 真机):
#   - compute 轴: 本机实测最好的 fp16 核 140.2 TFLOPS = 规格的 94.7%, 出口打得开,
#     但只剩 4.7% 余量 (对着旧的 138.2 分母是「随便就打满」)。
#   - memory 轴: 本机 D2D 拷贝实测最好 3357 GB/s = 4000 规格的 **84%** (带宽阶梯见
#     docs/research/42), 低于 90 —— 这条轴上出口**永不打开**, memory-bound 核会把
#     轮次跑满 (由停滞出口兜底)。改之前分母是 3680, 同一个读数 91.2% 是能出口的。
#   这是这次吸纳**故意付出的代价**: 分母偏小 (旧 138.2 更偏小) 会把还能优化的核**提前**
#   判成到顶, 而多烧几轮是可恢复的。正解是按 shape 查这台卡的可达上界 (缺口 2 的持久化
#   记录), 而不是把阈值调回 84% 去迁就一个更好的分母。
# ⚠️ 参照 KernelBench-Hard 的一句话: 人写的 Hopper 核常见只到峰值的 60~85% —— 按
# 这个量级, 90 是一个**偏乐观**的出口, 打不开它才是常态 (docs/research/42)。
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


def persist_decision(metrics: dict) -> tuple:
    """这次运行的核值不值得写进跨运行经验库 (缺口 2)。返回 (入库?, 理由)

    写成判据而不是写在 conductor 里, 是为了让它有单测与变异守卫 —— 缺口 2 原先的
    失效方式正是「快照一直在写, 但没人读」(docs/research/43 第三节)。

    门比 Voyager 的 `if success` 严三条, 理由都来自本机实测:
      - `measurement_can_gate` (缺口 1 的产物): 「通过」可以是被 launch 地板喂出来的
        假信号 (doc 31/32: 同一个 layernorm 核换个计时口径, 加速比从 0.895x 翻到 1.346x)。
        持久化会把这种假信号**固化成下一次检索的样本**, 比一次性误判更贵。
      - `backend_real_device` (doc 49 §3.5 第 2/3 处): 执行侧没声明 `Cap.REAL_DEVICE`
        时, 那个加速比要么是按规格**估**的 (`MockGPU` 的 `mode="simulated"`), 要么是
        控制机 CPU 上真测的 (`LocalExecutor`)。两者都不是「这张卡上这个核有多快」的
        证据, 而语料是按 (algorithm, hardware) 检索的 —— 一条 mock 成绩会被下一次
        真机运行当示范取出来。用 `is False` 而不是 `not`: 老产物没这个键, 不做追溯性重判。
      - score 用 speedup 而不是 gflops: 对 memory-bound 算子 gflops 整个失真 (缺口 4)。
    """
    if metrics.get("measurement_can_gate") is False:
        return False, "测量不可信 (can_gate=False): 不能把假信号固化成语料"
    if metrics.get("backend_real_device") is False:
        return False, ("执行侧未声明 REAL_DEVICE: 数字来自模拟或 CPU, "
                       "不是这张卡上的成绩, 不能当语料")
    if metrics.get("benchmark_contaminated"):
        return False, "窗口被抢占污染 (util 超阈值)"
    speedup = metrics.get("speedup_vs_baseline")
    if not isinstance(speedup, (int, float)) or speedup <= 1.0:
        return False, f"相对同机基线没有加速 (speedup={speedup})"
    return True, (f"speedup={speedup:.3f}x, "
                  f"validity={metrics.get('measurement_validity', '未知')}")


def roofline_converged(metrics: dict) -> bool:
    """已经贴着 roofline 上界 —— 继续换配方只会白烧 GPU 与 LLM token"""
    if (metrics.get("measurement_can_gate") is False
            or metrics.get("benchmark_contaminated")):
        return False          # 不可信的测量不能说「已经到顶」
    # 分母自己不可信时同样不能说「到顶」: 峰值表里没核过口径的行可能偏小 2 倍
    # (sparse/dense 混过), 那会把还能优化的核提前判成已到上限。用 `is False` 而不是
    # `not ...`: 老记录里没这个键时保持原有行为 (默认开), 新链路一律带 True/False。
    if (metrics.get("roofline_analysis") or {}).get("peak_trusted") is False:
        return False
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


# ─── 数字的出处 (doc 49 §3.5 第 2/3 处) ────────────────────────────────────────
#
# 两个字符串是**载荷自报**的值, 只用来与执行侧的能力声明对照, 不用来推断能力:
#   `"simulated"` = `MockGPU._benchmark_triton_simulated` 按硬件规格估出来的那份;
#   `"cpu"`       = `LocalExecutor` / `MockGPU._benchmark_python` / runner 的
#                   `"device": str(device) if lang == "torch" else "cpu"` 三处会打的标。
# 判据本身走 `Cap` (声明), 见 `provenance_conflicts` 的 docstring。
SIMULATED_MODE = "simulated"
CPU_DEVICE = "cpu"


def provenance_conflicts(backend_caps, required_caps,
                         reported_mode: Optional[str] = None,
                         reported_device: Optional[str] = None) -> tuple:
    """这批数字的**出处**站不站得住。返回 `(能力缺口, 自相矛盾)`, 都是字符串 tuple

    为什么要两份而不是一份: 它们的后果不同。

    * **能力缺口** = 任务在 `target.required_caps` 里声明了它要什么, 执行侧的
      `capabilities` 里没有。这是 doc 49 §五 第 4 片的验收条件所要拦的那件事:
      `mode == "simulated"` 的 L3 不许 `passed=True` (当任务声明了 `REAL_DEVICE`)。
      数字不是「测得不准」, 是**根本没测** —— 一份按规格估出来的加速比, 它的
      「通过」与「不通过」都没有对象。
    * **自相矛盾** = 声明与载荷自己报的对不上。声明 `REAL_DEVICE` 却报
      `mode="simulated"` / `device="cpu"`, 说明提供方要么在说谎、要么中途回退了
      (真有其事: `MockGPU` 有 CUDA 时声明 `REAL_DEVICE`, 而跑 python 核走的是
      `_benchmark_python`, 载荷老老实实写着 `device="cpu"`); 反过来, 载荷自报
      `simulated` 而后端没声明 `SIMULATED_PERF`, 是一次**未声明的降级**
      (doc 49 §3.3 那一族)。

    ⚠️ 刻意**不**用 `device`/`name` 的子串去推断能力 (那正是 doc 49 §3.1 记的三处
    名字匹配的毛病): 这里只判「等于 `cpu`」这一个精确值, 且只在它**与声明打架**时
    才算证据。`"cuda (fallback)"` 这类串一律不认 —— 认了就等于把能力判断又搬回
    字符串匹配。
    """
    caps = frozenset(backend_caps or ())
    required = frozenset(required_caps or ())
    shortfall = tuple(sorted(c.value for c in (required - caps)))

    declares_real = Cap.REAL_DEVICE in caps
    mode = str(reported_mode or "").strip().lower()
    device = str(reported_device or "").strip().lower()

    conflicts: List[str] = []
    if declares_real and mode == SIMULATED_MODE:
        conflicts.append(f"声明了 {Cap.REAL_DEVICE.value}, 载荷却自报 "
                         f"mode={reported_mode!r} (性能是按规格估的)")
    if declares_real and device == CPU_DEVICE:
        conflicts.append(f"声明了 {Cap.REAL_DEVICE.value}, 载荷却自报 "
                         f"device={reported_device!r} (这一轮在 CPU 上测的)")
    if mode == SIMULATED_MODE and Cap.SIMULATED_PERF not in caps:
        conflicts.append(f"载荷自报 mode={reported_mode!r}, 后端却没声明 "
                         f"{Cap.SIMULATED_PERF.value} (未声明的降级)")
    return shortfall, tuple(conflicts)


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
    #: 任务要求、而执行侧没声明的能力 (`Cap.value` 字符串, 已排序)
    capability_shortfall: tuple = ()
    #: 声明与载荷自报互相打架的地方 (doc 49 §3.5 第 2/3 处)
    provenance_conflict: tuple = ()
    #: 任务声明要的能力, 用来决定「矛盾」够不够格拒绝 L3 通过
    required_caps: tuple = ()
    #: 执行侧是否声明了 `Cap.REAL_DEVICE`。None = 调用方没给能力信息 (老路径)
    backend_real_device: Optional[bool] = None
    #: 与上一次运行相比变了哪些环境项; 空 = 比过了没漂移或无从比较 (见 EnvironmentDrift)
    environment_drift: tuple = ()

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

    @property
    def l3_pass_is_refused(self) -> bool:
        """L3 的 `passed` 不许为 True —— 与 `can_gate` 刻意分成两条

        `can_gate=False` 说的是「这个数字不足以判定通过/不通过」, 它**不**把 passed
        拉成 False: launch 地板支配 / 离散度超标的那一档, 核可能确实是对的, 只是数字
        撑不住结论, 而既有链路 (含真机标定过的 layernorm 1024×768) 一直按这个语义跑。

        出处不对是另一回事。任务声明了要真卡, 执行侧给不出 (或给了却自报在 CPU 上测),
        这时「通过」这个结论**没有对象** —— 那份加速比不是测得不准, 是压根没测。
        doc 49 §五 第 4 片的验收条件就是这一条: `mode == "simulated"` 的 L3 结果
        不许 `passed=True` (当任务声明了 `REAL_DEVICE`)。
        """
        if self.capability_shortfall:
            return True
        # 声明与载荷打架时, 只有「任务真的要过真卡」才够格拒绝通过: 否则一份本地
        # mock 运行会因为载荷老实写了 device="cpu" 而被判失败, 那是把观测当判据。
        return bool(self.provenance_conflict) and Cap.REAL_DEVICE.value in self.required_caps

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
        # `backend_real_device` 是**判据的输入**而不是装饰: `persist_decision` 读它决定
        # 「这份成绩能不能进跨运行语料」。有了它, 一份纯模拟的基准与一份真机实测的基准
        # 在产物里不再权威性等价 (doc 49 §3.5 第 2 处)。
        # 只在调用方给了能力信息时才写 —— 缺键与 False 是两件事, 老产物走 `is False`
        # 判定, 不会被追溯性重判。同一批里的 shortfall/矛盾/拒绝是同一个理由: 没做过
        # 出处检查时不写, 免得下游把「没检查」读成「检查过、干净的」。
        if self.backend_real_device is not None:
            out["backend_real_device"] = self.backend_real_device
            out["measurement_capability_shortfall"] = list(self.capability_shortfall)
            out["measurement_provenance_conflict"] = list(self.provenance_conflict)
            # 这个布尔是**判据的输出**: 没有它, 下游想区分「核太慢所以没过」与
            # 「压根没测所以没资格过」就只能去 string-match message 里那个 ⛔。
            out["measurement_l3_pass_refused"] = self.l3_pass_is_refused
        # 漂移这里落成列表: 空 = 本轮没有漂移信号。注意「比过了没变」与「没有上一次可比」
        # 在这一层分不开 (conductor 两条都传 None) —— 那个区分在产物级的
        # `environment_drift.comparable` 里, 不在轮级 metrics 里。
        out["measurement_environment_drift"] = list(self.environment_drift)
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
           max_over_median: Optional[float] = None,
           backend_caps: Optional[frozenset] = None,
           required_caps: Optional[frozenset] = None,
           reported_mode: Optional[str] = None,
           reported_device: Optional[str] = None,
           environment_drift: Optional[List[str]] = None) -> MeasurementAssessment:
    """综合判定测量有效性: 信号各自独立判定, 合成时取更差的一档

    问「测得准不准」的三个: launch 地板支配 (`median_time_ms` vs `launch_floor_us`)、
    测量前 util 污染 (`gpu_util_pct`)、样本主体离散度 (`dispersion` = iqr/median)。

    问「测的是不是这个东西」的两个 (doc 49 §3.5): 出处 (`backend_caps` 对
    `required_caps`, 加上载荷自报的 `reported_mode`/`reported_device` 作对照) 与
    环境漂移 (`environment_drift`, 由 `environment.compare_fingerprints` 算出)。

    污染判为 DEGRADED 而不是 UNINFORMATIVE: `benchmark_with_baselines` 是同机同卡
    依次测候选与各基线的, 两者落在同一个污染窗口里, 比值仍保留部分意义; 但绝对
    gflops 与比值都被扰动过, 不足以支撑「这个配方没用」这种结论。

    离散度同理: iqr/median 超过 0.10 说明这台机器此刻在抖, median 本身仍是无偏的,
    但它的精度撑不住「加速比 1.02x」这种量级的结论。

    环境漂移也停在 DEGRADED, 而且**没有阈值**: 「torch 从 2.14 变成 2.15」到底该不该
    让一次运行作废, 本机还没有分布能回答。按项目「观测先于判定」的处置, 先把
    「变了哪些键」这件事变得可见 (产物 + 日志), 并且不让它去驱动永久决策。

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

    # ── 出处 (doc 49 §3.5 第 2/3 处) ──────────────────────────────────
    # `backend_caps is None` 表示调用方压根没提供能力信息 (老路径 / 直接调 assess 的
    # 单测), 与「提供了空集」必须分开: 后者是 `LocalExecutor` 如实声明的样子, 该判;
    # 前者无从判, 只能显式记一句。
    shortfall: tuple = ()
    conflicts: tuple = ()
    real_device: Optional[bool] = None
    if backend_caps is None:
        if required_caps:
            reasons.append("任务声明了 required_caps 但调用方没提供后端能力, 未做出处检查")
    else:
        real_device = Cap.REAL_DEVICE in frozenset(backend_caps)
        shortfall, conflicts = provenance_conflicts(
            backend_caps, required_caps, reported_mode, reported_device)
        if shortfall:
            validity = _worse(validity, Validity.UNINFORMATIVE)
            reasons.append(
                f"任务要求的能力执行侧没有: {', '.join(shortfall)} —— 这批数字不是"
                f"在满足要求的设备上测出来的, 「通过/不通过」都没有对象")
        for c in conflicts:
            validity = _worse(validity, Validity.UNINFORMATIVE)
            reasons.append(f"出处自相矛盾: {c}")

    # ── 环境漂移 (doc 49 §3.5 第 1 处) ────────────────────────────────
    drift = tuple(environment_drift or ())
    if drift:
        validity = _worse(validity, Validity.DEGRADED)
        reasons.append(
            "环境指纹与上一次运行不一致 (" + "; ".join(drift) + "): "
            "本轮加速比与上一次不可比, 不得据此禁用配方或累计停滞; "
            "**不据此判核失败** (观测先于判定)")

    return MeasurementAssessment(
        validity=validity,
        reasons=reasons,
        launch_dominance=dom,
        launch_floor_us=launch_floor_us,
        contaminated=contaminated,
        dispersion=dispersion,
        max_over_median=max_over_median,
        intruder_suspected=intruder,
        capability_shortfall=shortfall,
        provenance_conflict=conflicts,
        required_caps=tuple(sorted(c.value for c in frozenset(required_caps or ()))),
        backend_real_device=real_device,
        environment_drift=drift,
    )
