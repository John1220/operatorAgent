"""标定心跳 (Calibration heartbeat) —— 无人值守下唯一能发现「度量器悄悄坏了」的机制

设计出处: `docs/design/autonomous-operator-agent-design.md` §10.5-1 / §13.5 / §4.3 第 2 条:

    每 N 次真值测量塞两个已知参照: 一个已知好 kernel、一个故意写慢的 kernel。
    二者都在带内 → 尺子可信; 任一偏出 → 该卡宣布 RULER_BROKEN:
    停止基于数字决策、该窗口观测全部失效、不持久化、直接进 REPORT。

为什么必须成对塞, 而不是只测一个「已知好」:

* 只有已知好核 —— 读数整体偏慢/偏快时, 若不做绝对锚定就看不出来;
  而绝对锚点 (期望中位时间) 本身跨机器不可移植, 只能由调用方从预检标定注入。
* 只有比值 (慢核/好核) —— 整个 harness 被缩放 10× 时比值不变, 尺子仍然「看起来对」。
* **一对 + 两个比值**: `known_good_ratio` 抓绝对偏置, `known_slow_ratio` 抓分离度
  (尺子还分不分得出快慢)。二者都在带内才 ok, 任一偏出 → RULER_BROKEN。

`known_good_ratio = 好核读数 / expected_good`(绝对锚点, 无锚点则不参与判定);
`known_slow_ratio = 慢核读数 / 好核读数`(期望显著 ≥2×, 慢核故意多算几遍)。

默认参照对是纯 Python 小 matmul 与重复算 N 遍的慢核 —— 可跑、可离线单测,
不依赖 GPU/torch。真机路径由调用方注入自己的 `bench_fn` 与参照核源码。

与既有机制的分工 (不是重复):
  * `measurement.py`     —— 问「这一次读数准不准 / 是不是这个东西」
  * `contract.py`        —— 问「判据本身有没有被改写」
  * **本模块**           —— 问「**尺子本身**还准不准」—— 参照物读数偏出带宽

定位与 `tolerance.py` / `measurement.py` 相同: 纯函数 + 可离线单测, 不做 I/O。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

#: 失败原因前缀。与 `confirm.py` 报告侧的 "标定心跳失败 (RULER_BROKEN)" 对齐。
RULER_BROKEN = "RULER_BROKEN"


# ─── 默认参照对: 纯 Python 小 matmul (好核) vs 多算几遍的慢核 ─────────────────
#
# 参照核的**源码形态**为什么必须存在 (而不是只留可调用形态):
# 真机路径要让参照物走**远端那把尺子**—— `backend.benchmark(code, inputs)`, 也就是
# 部署 → runner → exec → 计时 → JSON 回传这一整条路。只用 `default_bench_fn` 在控制机
# 上计时, 测的是**控制机的 `perf_counter`**: 远端的尺子坏了它照样报绿, 那是「报表上
# 显示为已检查」的变体。`backend_bench_fn` 就是走这条路的那把尺子。
#
# ⚠️ 它**不是**「与候选完全同一条计时通道」—— 说清楚边界:
# `run_kernel.py:_detect_language` 看 import 分派; 参照核不带 triton/torch import →
# 判成 `numpy` → 走 `_benchmark_torch_numpy` 的 else 分支 → `_timed_runs(..., False)`
# → 远端 `perf_counter`, 回传 `"device": "cpu"`。triton/torch 候选走的是 CUDA event
# 那条分支。所以这一对参照抓得住的是: 部署/回传/exec/时钟/顺序坏掉, 以及**在 Python
# 分支上**「计时已经分不出快慢」; 抓不住**CUDA event 分支**坏掉。要覆盖后者得塞一对
# triton 参照核 (按任务的 DSL 选), 目前**没做** —— 见 docs/design/spine-review.md P1-5。
#
# 也别给这里加「payload 自报 device=cpu 就判坏」的检查: 参照核本来就该在 CPU 上跑,
# 加了会变成每次心跳都误报 RULER_BROKEN (把「本就不该是 GPU 数」读成「出处不对」)。
# 两种形态由**同一个定义体** `_MATMUL_BODY` 生成, 不并存两份会各自漂的拷贝。

_MATMUL_BODY = """\
    n, p, m = len(A), len(B), len(B[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            aik = A[i][k]
            row = out[i]
            bk = B[k]
            for j in range(m):
                row[j] += aik * bk[j]
    return out
"""

#: 慢核比好核多算几遍。**烘进源码**, 不作为 `inputs` 里的标量传 —— 2026-10-01 真机
#: 实测抓到的: `SSHExecutor._serialize_inputs` 把每个输入都变成「shape+dtype+seed,
#: 远端按 seed 重新生成」, 于是 `repeats=8` 到了远端是一个**按 int64 截断的随机标量**
#: (实测恒为 0), 慢核只跑 1 遍 → 比值 1.0 → 健康的真机上心跳宣布 RULER_BROKEN。
#: 烘进源码之后参照对只依赖数组输入, 三个后端 (ssh/local/http) 行为一致。
SLOW_REPEATS = 8

KNOWN_GOOD_SOURCE = '''\
def kernel(A, B):
    """已知好参照核: 小矩阵乘。故意朴素但正确 —— 它是尺子的「准星」。"""
''' + _MATMUL_BODY

#: 故意写慢的参照核: 同样的 matmul 重复算 `SLOW_REPEATS` 遍, 只留最后一遍。
#: 分离度是**设计出来的**, 不是测出来的期望 —— 重复 8 遍时慢核读数应约 8× 好核,
#: 远超默认下界 `min_slow_ratio=2.0`, 于是「带内」判据对计时噪声不敏感。
#: 遍数写死在源码里 (见 `SLOW_REPEATS` 的说明): 走 `inputs` 传标量在 SSH 后端上会被
#: 重新生成, 那是「参照对静默退化成一个核」的入口。
KNOWN_SLOW_SOURCE = '''\
def kernel(A, B):
    """故意写慢的参照核: 同样的 matmul 重复算 {repeats} 遍, 只留最后一遍。"""
    out = None
    for _ in range({repeats}):
        out = _plain(A, B)
    return out


def _plain(A, B):
'''.format(repeats=SLOW_REPEATS) + _MATMUL_BODY


def _from_source(source: str, name: str):
    """把源码形态 exec 成可调用形态, 并把源码挂在 `__source__` 上。

    `__name__` 显式写回原样: `heartbeat` 的调用方 (以及 `_fixed_bench` 那类假 bench)
    靠核名分辨好核/慢核, 而 exec 出来的函数一律叫 `kernel` —— 不改名会让「慢核」
    被当成好核, 方向检查静默失效。
    """
    ns: dict = {}
    exec(compile(source, f"<calibration:{name}>", "exec"), ns)
    fn = ns["kernel"]
    fn.__name__ = name
    fn.__source__ = source
    return fn


known_good_kernel = _from_source(KNOWN_GOOD_SOURCE, "known_good_kernel")
known_slow_kernel = _from_source(KNOWN_SLOW_SOURCE, "known_slow_kernel")


def default_inputs(n=24):
    """默认参照输入: 两个小方阵。尺寸刻意不大 —— 心跳要便宜, 不该烧机时。"""
    A = [[float((i * n + j) % 17) * 0.1 for j in range(n)] for i in range(n)]
    B = [[float((i + 3 * j) % 13) * 0.1 for j in range(n)] for i in range(n)]
    return (A, B)


def default_bench_fn(code, inputs, warmup=1, samples=5):
    """默认 bench: 对可调用参照核计时, 返回中位毫秒 (cost, 越大越慢)。

    ⚠️ 它只量**控制机**: 拿它当心跳的 bench_fn 时, 远端/runner 的计时通道坏了是
    看不出来的。真机路径用 `backend_bench_fn`。保留它是为了离线单测与无后端场景。

    调用方注入的 `bench_fn(code, inputs) -> float` 同此签名: 返回值一律当 **cost**
    (median_ms) 解读。若你的 bench 报的是「越大越好」的 ratio, 请在包装层取倒数。
    """
    fn = code
    for _ in range(max(0, int(warmup))):
        fn(*inputs)
    xs = []
    for _ in range(max(1, int(samples))):
        t0 = time.perf_counter()
        fn(*inputs)
        xs.append((time.perf_counter() - t0) * 1e3)
    xs.sort()
    return xs[len(xs) // 2]


def backend_bench_fn(backend, *, warmup: int = 1, repeat: int = 3):
    """让**远端那把尺子**量参照物: `heartbeat` 的 bench_fn 走 `backend.benchmark`。

    与 `default_bench_fn` 的分工、以及它覆盖到哪/覆盖不到哪 (Python 计时分支 vs
    triton 候选的 CUDA event 分支), 见上面源码形态那段的说明。`inputs` 是
    `default_inputs()` 的元组形态, 这里转成调用约定要求的 dict —— **只放数组**:
    标量走 `inputs` 在 SSH 后端上会被重新生成 (见 `SLOW_REPEATS` 那段), 参照核
    需要的常数一律烘进源码。

    读数不可用时**抛**而不是返回一个假数字 —— `heartbeat` 会把异常折成
    `ok=False`(RULER_BROKEN): 「尺子读不出数」本来就该按坏处理, 不该静默放行。
    """
    def bench(kernel, inputs):
        source = getattr(kernel, "__source__", None)
        if not source:
            raise ValueError(
                f"参照核没有源码形态, 后端量不了它: {kernel!r} —— 只有由 "
                "KNOWN_GOOD_SOURCE / KNOWN_SLOW_SOURCE 生成的核才带 __source__")
        A, B = inputs
        kwargs = {"A": A, "B": B}
        payload = backend.benchmark(source, kwargs, warmup=warmup, repeat=repeat)
        if not isinstance(payload, dict):
            raise ValueError(f"后端没有回载荷: {payload!r}")
        # `_transport_failure` 那类失败**以字典形态返回**而不是抛 (`remote_executor`),
        # 所以这里必须显式看 success —— 否则「没听到回答」会被读成一次读数。
        if payload.get("success") is False:
            raise RuntimeError(f"后端基准失败: {payload.get('error') or payload!r}")
        cost = payload.get("median_time_ms")
        if cost is None:
            cost = payload.get("avg_time_ms")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost <= 0:
            raise ValueError(f"后端没回可用的计时: {payload!r}")
        return float(cost)

    return bench


# ─── 结果与判定 ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HeartbeatResult:
    """一次标定心跳的判定。`ok=False` = RULER_BROKEN, 该窗口观测全部失效。"""
    ok: bool
    epoch: int
    reasons: Tuple[str, ...]
    known_good_ratio: float
    known_slow_ratio: float
    window_id: str


def heartbeat(bench_fn: Callable[[Any, Any], float],
              inputs: Any = None,
              *,
              epoch: int = 0,
              window_id: str = "",
              expected_good: Optional[float] = None,
              good_band: Tuple[float, float] = (0.5, 2.0),
              min_slow_ratio: float = 2.0,
              max_slow_ratio: Optional[float] = None,
              good_kernel: Any = None,
              slow_kernel: Any = None) -> HeartbeatResult:
    """塞一对已知参照核进 `bench_fn`, 判尺子是否还可信。

    参数
    ----
    bench_fn:
        调用方注入的测量函数 `bench_fn(code, inputs) -> cost`。返回值当 median_ms
        (越大越慢) 解读。注入点在这里, 所以单测不需要 GPU/torch。
    expected_good:
        已知好核的期望 cost (绝对锚点)。来自预检标定, **不能**由同一个 bench_fn
        自己量自己 (那样抓不到 bench 整体偏置)。缺省 None = 只做分离度检查,
        `known_good_ratio` 记 1.0 且不参与判定。
    good_band:
        `known_good_ratio = good_cost / expected_good` 的接受带。偏出 (偏慢或偏快)
        都算 RULER_BROKEN —— 方向检查的两臂都在这一条里。
    min_slow_ratio / max_slow_ratio:
        `known_slow_ratio = slow_cost / good_cost` 的接受带。默认下界 2.0
        (慢核应显著更慢); 上界缺省不设 (慢核再慢也不构成「尺子坏了」)。

    判定: 两个比值**都在带内**才 `ok=True`; 任一偏出 → RULER_BROKEN。
    """
    g_kern = known_good_kernel if good_kernel is None else good_kernel
    s_kern = known_slow_kernel if slow_kernel is None else slow_kernel
    if inputs is None:
        inputs = default_inputs()
    wid = window_id or f"hb-{epoch}"

    def _broken(*reasons: str) -> HeartbeatResult:
        tagged = tuple(
            r if r.startswith(RULER_BROKEN) else f"{RULER_BROKEN}: {r}" for r in reasons
        )
        return HeartbeatResult(
            ok=False, epoch=epoch, reasons=tagged,
            known_good_ratio=float("nan"), known_slow_ratio=float("nan"),
            window_id=wid,
        )

    try:
        good_cost = float(bench_fn(g_kern, inputs))
        slow_cost = float(bench_fn(s_kern, inputs))
    except Exception as exc:  # bench 炸了 = 尺子读不出数, 失败关闭
        return _broken(f"bench_fn 调用失败: {exc}")

    if not (math.isfinite(good_cost) and math.isfinite(slow_cost)):
        return _broken(f"非有限读数: good={good_cost!r} slow={slow_cost!r}")
    if good_cost <= 0 or slow_cost <= 0:
        return _broken(f"非正读数: good={good_cost!r} slow={slow_cost!r}")

    # 绝对锚点: 抓「读数整体偏慢/偏快」。无锚点时不参与判定 (记 1.0, 不是测出来的)。
    if expected_good is not None:
        if not (isinstance(expected_good, (int, float)) and math.isfinite(expected_good)
                and expected_good > 0):
            return _broken(f"expected_good 非法: {expected_good!r}")
        known_good_ratio = good_cost / float(expected_good)
    else:
        known_good_ratio = 1.0

    # 分离度: 抓「尺子分不出快慢」。慢核故意多算, 期望显著更大。
    known_slow_ratio = slow_cost / good_cost

    reasons = []
    if expected_good is not None:
        lo, hi = good_band
        if not (lo <= known_good_ratio <= hi):
            reasons.append(
                f"已知好核比值 {known_good_ratio:.4g} 偏出带宽 [{lo}, {hi}] "
                f"(good={good_cost:.4g} expected={float(expected_good):.4g})")
    lo_s = float(min_slow_ratio)
    hi_s = float(max_slow_ratio) if max_slow_ratio is not None else math.inf
    if not (lo_s <= known_slow_ratio <= hi_s):
        reasons.append(
            f"已知慢核比值 {known_slow_ratio:.4g} 偏出期望带 [{lo_s}, {hi_s}] "
            f"(slow={slow_cost:.4g} good={good_cost:.4g}, 慢核应显著更慢)")

    if reasons:
        return HeartbeatResult(
            ok=False, epoch=epoch,
            reasons=tuple(
                r if r.startswith(RULER_BROKEN) else f"{RULER_BROKEN}: {r}" for r in reasons
            ),
            known_good_ratio=known_good_ratio,
            known_slow_ratio=known_slow_ratio,
            window_id=wid,
        )

    evidence = []
    if expected_good is not None:
        evidence.append(f"good_ratio={known_good_ratio:.4g} 在带宽 {good_band} 内")
    else:
        evidence.append("无绝对锚点, 仅核验分离度")
    evidence.append(f"slow_ratio={known_slow_ratio:.4g} 在带宽 [{lo_s}, {hi_s}] 内")
    return HeartbeatResult(
        ok=True, epoch=epoch, reasons=tuple(evidence),
        known_good_ratio=known_good_ratio,
        known_slow_ratio=known_slow_ratio,
        window_id=wid,
    )


# ─── 小工具 ───────────────────────────────────────────────────────────────────

def heartbeat_due(n_measurements: int, interval: int) -> bool:
    """累计真值测量次数是否到了该塞参照对的时刻 (设计 §10.5-1「每 N 次」)。

    `n_measurements` 是**累计**真值测量次数 (调用方自己维护), 到 `interval`
    的正整数倍时触发。0 次测量不触发 —— 还没有尺子可标定。
    """
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        raise ValueError(f"interval 必须为正整数: {interval!r}")
    if not isinstance(n_measurements, int) or isinstance(n_measurements, bool):
        raise ValueError(f"n_measurements 必须为整数: {n_measurements!r}")
    if n_measurements <= 0:
        return False
    return n_measurements % interval == 0


def ruler_broken_window(heartbeat_result: Optional[HeartbeatResult],
                        observation_window_id: str = "") -> bool:
    """该观测是否落在心跳失败的标定窗口内 (设计 §4.3 第 2 条 → 资格全 False)。

    心跳 `ok=True` → 哪个窗口都可用; `ok=False` → 同窗观测全部作废、不持久化。
    失败关闭: 观测没写 window_id 时按同窗处理, 不猜「也许不是这一窗」。
    跨窗观测不受牵连 —— 一张卡的尺子坏了, 不代表别的窗口/别的卡也坏 (§13.5)。
    """
    if heartbeat_result is None or heartbeat_result.ok:
        return False
    if not observation_window_id:
        return True
    return observation_window_id == heartbeat_result.window_id
