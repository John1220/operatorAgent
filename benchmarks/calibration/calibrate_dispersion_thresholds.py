"""在真机上标定离散度门的两个阈值

背景: runner 现在会随每次计时上报两个统计量 (阈值标定: docs/research/20 第四节):

    dispersion_iqr_over_median   (p75-p25)/median —— 样本**主体**的散布
    dispersion_max_over_median   max/median      —— 有没有离群样本

`measurement.assess()` 拿它们判测量有效性。阈值现状:

    DISPERSION_DEGRADED       = 0.10   ← 锚在 PyTorch `_IQR_WARN_THRESHOLD`, 有出处
    DISPERSION_UNINFORMATIVE  = 0.30   ← 无外部锚点, 按与 launch dominance 同节奏取的
    INTRUDER_MAX_OVER_MEDIAN  = 5.0    ← **完全没标定过**

第三个是问题所在。参照系 NVBench 的 `same.relative_dispersion_ceiling = 0.02` 是
**锁频专用机**的值; 本机是 8 卡共享、恒有 7 个其他进程, 直接套外部数字要么把
所有测量都判死, 要么形同虚设。所以只能实测。

这个脚本做的事: 对每个核, 重复 R 轮「取 n 个样本 → 算一次统计量」, 于是得到的是
**统计量本身的分布**, 而不是某一个值。安静机器上的 95 分位就是阈值的下界 ——
低于它会把正常抖动误判成污染。

判读方式:
  - 若 iqr/median 的 p95 已经 > 0.10, 说明 0.10 太紧, 本机常态就会被判 DEGRADED,
    信号失效 (与「把 util>0 当污染」是同一个失效模式)。
  - 若 max/median 的 p95 远小于 5.0 (比如 1.3), 说明 5.0 太松, 只有极端抢占才触发;
    取 p95 与 p99 之间某个值更合适。
  - 顺带记录每轮采样前的 util, 用来区分「机器本来就抖」和「某轮被抢了」。

在远端 GPU 机器上跑 (需要 torch/triton):
    ssh <别名> 'cd ~/operator_agent && CUDA_VISIBLE_DEVICES=<空闲卡> \
        ~/.local/bin/uv run python -' < benchmarks/calibration/calibrate_dispersion_thresholds.py
"""
import json
import math
import os
import statistics
import subprocess
import sys
import time

import torch
import triton
import triton.language as tl

ROUNDS = 30            # 每个核重复多少轮
SAMPLES = 30           # 每轮多少个计时样本 (与 runner 的 repeat 默认值一致)
FLUSH_BYTES = 256 * 1024 * 1024
WARMUP_MS_BUDGET = 25  # 与 run_kernel.py 的 WARMUP_MS_BUDGET 保持一致

# ── G4 要的三个阈值，配方与出处 ──────────────────────────────────────────────
#
# 判决带宽 `band` 的配方**不是这里发明的**：`measurement.py` 的 NOISE_BAND 注释写着
# 「判决带宽 = 3 × 实测的『同一份代码连跑多次的加速比散布』」，并逐档给了 H20 的读数
# （valid 0.02% / degraded 0.13% / uninformative 0.66%）。所以本脚本对每个核算
# **跨轮 median 的相对 iqr**，再乘 3，就是这一档在本卡上的带宽。
BAND_MULTIPLIER = 3.0
#: 与 `measurement.py` 的 LAUNCH_DOMINANCE_* / DISPERSION_* 同值。单文件 scp 过去的
#: 脚本 import 不到仓里的模块（见 runbook §4 第 0 步），所以这几个数只能复制一份；
#: 它们一变，这里的档位判定就要跟着改，`--check-constants` 会把两侧逐字比一遍。
LAUNCH_DOMINANCE_VALID = 0.10
LAUNCH_DOMINANCE_UNINFORMATIVE = 0.30
DISPERSION_DEGRADED_INHERITED = 0.10
DISPERSION_UNINFORMATIVE = 0.30
#: 仓内继承值（H20 八卡共享机标定）。测出来与它逐位相同 ⇒ G4 判「抄的」，所以
#: 下面一旦撞上就显式报出来并按一个可表示档位往上走（偏大是保守方向）。
INHERITED_BAND = {"valid": 0.001, "degraded": 0.004, "uninformative": 0.02}
INHERITED_UTIL_PCT = 5.0
INHERITED_DISPERSION_DEGRADED = 0.10

TIERS = ("valid", "degraded", "uninformative")
_TIER_ORDER = {t: i for i, t in enumerate(TIERS)}

_flush_buf = torch.empty(FLUSH_BYTES // 4, dtype=torch.int, device="cuda")


def physical_gpu_id() -> str:
    """nvidia-smi 认的是**物理** GPU 编号, 不受 CUDA_VISIBLE_DEVICES 影响。

    `torch.cuda.current_device()` 返回的是 CUDA 序号 —— 设了 CUDA_VISIBLE_DEVICES=2
    时它是 0, 拿它去问 nvidia-smi 会读到 0 号卡的 util, 而不是自己这张。
    runner 用的是自己选卡时记下的物理 id (_SELECTED_GPU_ID), 这里同样从环境变量
    还原, 还原不出来才退回 CUDA 序号。
    """
    cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if cvd:
        first = cvd.split(",")[0].strip()
        if first.isdigit():
            return first
    return str(torch.cuda.current_device())


def gpu_util_pct():
    """测量前的 util 快照。取不到就返回 None, 不猜 —— 与 runner 的口径一致。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={physical_gpu_id()}",
             "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


@triton.jit
def _empty(x_ptr):
    pass


@triton.jit
def _ln(x_ptr, w_ptr, b_ptr, y_ptr, n_rows, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * n_cols + cols, (xc * rstd * w + b).to(y_ptr.dtype.element_ty),
             mask=mask)


@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
        sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    offs_am = pid_m * BM + tl.arange(0, BM)
    offs_bn = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_am[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_bn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c = c_ptr + offs_am[:, None] * scm + offs_bn[None, :] * scn
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c, acc.to(c.dtype.element_ty), mask=mask)


def timed_once(fn) -> float:
    """与 run_kernel._timed_runs 同口径: 冲刷排在 start.record() 之前, 同流有序。"""
    _flush_buf.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def percentile(sorted_vals, q):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = (n - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def relative_iqr(vals):
    """(p75-p25)/p50。跨轮用它是「同一份代码连跑多次的散布」，轮内用它就是 iqr/median。"""
    if not vals:
        return None
    s = sorted(vals)
    med = percentile(s, 50)
    if med <= 0:
        return None
    return (percentile(s, 75) - percentile(s, 25)) / med


def ceil_to(value, ndigits=4):
    """向上取整到指定位数：阈值只能往「更宽」的方向取，否则测出来的抖动会被判成污染。"""
    scale = 10 ** ndigits
    return math.ceil(value * scale - 1e-12) / scale


def worse_tier(a, b):
    return a if _TIER_ORDER[a] >= _TIER_ORDER[b] else b


def tier_for(median_ms, floor_us, dispersion):
    """按 `measurement.assess()` 的两条主线（地板支配 / 离散度）定这一轮该落哪一档。

    `assess()` 取「更差的一档」，这里的 `worse_tier` 是同一件事。地板支配的算法与
    `launch_dominance()` 一致：dom = floor/(floor+net)，而 net = median - floor，
    所以 dom = floor/median。
    """
    tier = "valid"
    dom = None
    if floor_us and median_ms and median_ms > 0:
        dom = (floor_us / 1000.0) / median_ms
        if dom > LAUNCH_DOMINANCE_UNINFORMATIVE:
            tier = "uninformative"
        elif dom > LAUNCH_DOMINANCE_VALID:
            tier = "degraded"
    if dispersion is not None:
        if dispersion > DISPERSION_UNINFORMATIVE:
            tier = worse_tier(tier, "uninformative")
        elif dispersion > DISPERSION_DEGRADED_INHERITED:
            tier = worse_tier(tier, "degraded")
    return tier, dom


def quiet_util_samples(n=5, settle_s=3.0, pause_s=0.5):
    """**静止窗**里的 util 采样：此刻没有我们自己的核在跑。

    与 `calibrate()` 里每轮开跑前那一次快照不同 —— 那一次可能正撞上我们自己上一轮的尾巴。
    2026-11-01 在本卡上实测过这个坑：每轮开跑前的快照 max=70%、非零 23/30，而那些
    「活动」全是我们自己的核；`nvidia-smi` 的 util 是**过去一个采样窗内的平均**，
    紧随一次 kernel 之后立刻读，读到的是自己。拿它定阈值会得到 71%（等于看不见任何对手）。

    所以静止窗必须**先等**：`settle_s` 秒内不提交任何工作，让采样窗滚过去，
    再按 `pause_s` 间隔连采 n 次。
    """
    torch.cuda.synchronize()
    time.sleep(settle_s)
    out = []
    for i in range(n):
        if i:
            time.sleep(pause_s)
        out.append(gpu_util_pct())
    return out


def contamination_threshold(samples):
    """独占/共享形态未知时**不猜**：全部读 0 才敢把阈值压到传感器分辨率的下限。

    返回 (阈值, 依据字符串)。
    """
    seen = [s for s in samples if s is not None]
    if not seen:
        return None, "nvidia-smi 不可用：污染阈值没有读数"
    top = max(seen)
    if top <= 0:
        # nvidia-smi 的 util 分辨率是 1%，0 是它能报的最小值；静止窗全 0 说明本卡在
        # 我们不动的时候没有对手。阈值取 1.0 = 只要能观测到任何活动就算污染。
        return 1.0, "静止窗 %d 次采样全 0%%（观测不到对手）⇒ 取传感器分辨率下限 1.0" % len(seen)
    return float(math.ceil(top) + 1), "静止窗观测到 %g%% 的活动 ⇒ 阈值取其上方一档" % top


def measure_batch(fn, n=SAMPLES):
    """一轮: n 个样本 → (median_ms, 轮内 iqr/median, max/median, 均值)。

    均值是给**跨轮**散布用的，不是给判决用的：生产统计量是 median，但 median 落在
    `torch.cuda.Event` 的量化步长上（实测本卡 small/mid 两个核 30 轮的 median 逐位相同
    ⇒ 跨轮 iqr = 0），照它当带宽会得到 0 —— 那意味着「任何一点差别都算显著」，
    比阈值没有还糟。n 个样本的均值把分辨率抬高 n 倍，描述的仍是「同一份代码的另一轮」。
    """
    times = sorted(timed_once(fn) for _ in range(n))
    median = percentile(times, 50)
    p25, p75 = percentile(times, 25), percentile(times, 75)
    mean = statistics.fmean(times)
    if median <= 0:
        return median, None, None, mean
    return median, (p75 - p25) / median, times[-1] / median, mean


def calibrate(name, make_fn, warmup=True):
    fn = make_fn()
    if warmup:
        fn()                                   # 吃掉 JIT
        torch.cuda.synchronize()
        est = statistics.median(timed_once(fn) for _ in range(5))
        n_warm = int(min(max(1, WARMUP_MS_BUDGET / est), 5000)) if est > 0 else 1
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()

    utils = []
    medians, means, iqrs, maxes = [], [], [], []
    for _ in range(ROUNDS):
        utils.append(gpu_util_pct())          # 每轮都采: 单次快照对不上具体哪轮被抢
        m, i, x, avg = measure_batch(fn)
        medians.append(m)
        means.append(avg)
        if i is not None:
            iqrs.append(i)
            maxes.append(x)

    iqrs_s, maxes_s = sorted(iqrs), sorted(maxes)
    seen = [u for u in utils if u is not None]
    util_desc = (f"min={min(seen):.0f}% max={max(seen):.0f}% "
                 f"非零轮数={sum(1 for u in seen if u > 0)}/{len(utils)}") if seen \
        else "nvidia-smi 不可用"
    print(f"\n### {name}")
    print(f"  物理 GPU = {physical_gpu_id()}   轮数 = {ROUNDS}   每轮样本 = {SAMPLES}")
    print(f"  各轮测量前 util: {util_desc}")
    print(f"  median (ms): p50={percentile(sorted(medians), 50):.5f} "
          f"min={min(medians):.5f} max={max(medians):.5f}")
    for label, vals in (("iqr/median", iqrs_s), ("max/median", maxes_s)):
        print(f"  {label:>11}: p50={percentile(vals, 50):.4f} "
              f"p90={percentile(vals, 90):.4f} p95={percentile(vals, 95):.4f} "
              f"max={max(vals):.4f}")
    return {"name": name, "medians_ms": sorted(medians), "means_ms": sorted(means),
            "iqrs": iqrs_s, "maxes": maxes_s, "utils": utils}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="calibrate_dispersion_thresholds", allow_abbrev=False)
    ap.add_argument("--floor-us", type=float, default=None,
                    help="G1 实测的 launch 地板（µs）。**必须与 --flush 口径一致**："
                         "measure_launch_floor.py 不冲刷 L2，量的是 CPU 派发；本脚本冲 256MB。"
                         "给一个比最短核还长的地板会被拒（那两个数不是同一把尺）")
    ap.add_argument("--runs", type=int, default=1,
                    help="独立标定几轮，逐项取**最大**（阈值偏大 = 判决更保守）。"
                         "单轮阈值的轮间波动实测有 ~2 倍，cold-run 建议 ≥3")
    args = ap.parse_args(argv)
    if args.runs < 1:
        print("--runs 至少为 1")
        return 2

    print(f"torch={torch.__version__} triton={triton.__version__} "
          f"device={torch.cuda.get_device_name(0)}")
    print(f"目标: 标定 measurement.py 的 NOISE_BAND / DISPERSION_* ，并产出 G4 要的三个阈值")
    print(f"轮数 = {args.runs}（逐项取最大），每轮 {ROUNDS} 轮 × {SAMPLES} 样本/核")

    dev = "cuda"

    def empty_fn():
        x = torch.empty(1, device=dev)
        return lambda: _empty[(1,)](x)

    def ln_fn(rows, cols):
        x = torch.randn(rows, cols, device=dev, dtype=torch.float32)
        w = torch.ones(cols, device=dev, dtype=torch.float32)
        b = torch.zeros(cols, device=dev, dtype=torch.float32)
        y = torch.empty_like(x)
        block = triton.next_power_of_2(cols)
        return lambda: _ln[(rows,)](x, w, b, y, rows, cols, 1e-5, BLOCK=block,
                                    num_warps=8 if block >= 8192 else 4)

    def mm_fn(n):
        a = torch.randn(n, n, device=dev, dtype=torch.float16)
        b = torch.randn(n, n, device=dev, dtype=torch.float16)
        c = torch.empty(n, n, device=dev, dtype=torch.float16)
        grid = (triton.cdiv(n, 128) * triton.cdiv(n, 128),)
        return lambda: _mm[grid](a, b, c, n, n, n,
                                 a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                 c.stride(0), c.stride(1), BM=128, BN=128, BK=64,
                                 num_warps=8, num_stages=3)

    # 每个档位都要有核落进去，否则那一档的带宽没有读数可标 —— 这不是凑数，是
    # 「标定一个你从未观测过的档位」本身不成立。中 shape 那一行的存在理由就是这个：
    # 大 layernorm 与 matmul 落 valid，小 layernorm 落 uninformative，中间的 degraded
    # 档原先在清单里没有代表（H20 上是 16384×6144 落 degraded，本卡地板更高，它会升到 valid）。
    cases = [
        ("空核 (launch 地板对照)", empty_fn, True),
        ("layernorm 512x512 (极小 shape, uninformative 档的代表)", lambda: ln_fn(512, 512), False),
        ("layernorm 1024x768 (小 shape)", lambda: ln_fn(1024, 768), False),
        ("layernorm 8192x2048 (中 shape, degraded 档的代表)", lambda: ln_fn(8192, 2048), False),
        ("layernorm 16384x6144 (大 shape)", lambda: ln_fn(16384, 6144), False),
        ("matmul 4096^3 fp16", lambda: mm_fn(4096), False),
    ]

    runs = []
    for run_no in range(1, args.runs + 1):
        print("\n" + "#" * 68)
        print("### 第 %d/%d 轮标定" % (run_no, args.runs))
        all_iqr, all_max = [], []
        records = []
        for name, make, is_floor_control in cases:
            rec = calibrate(name, make)
            rec["is_floor_control"] = is_floor_control
            records.append(rec)
            all_iqr += rec["iqrs"]
            all_max += rec["maxes"]

        print("\n" + "=" * 68)
        print("汇总 (所有核合并, 这才是阈值该看的分布)")
        all_iqr.sort()
        all_max.sort()
        for label, vals in (("iqr/median", all_iqr), ("max/median", all_max)):
            print(f"  {label:>11}: p50={percentile(vals, 50):.4f} "
                  f"p90={percentile(vals, 90):.4f} p95={percentile(vals, 95):.4f} "
                  f"p99={percentile(vals, 99):.4f} max={max(vals):.4f}")

        derived = derive(records, floor_us=args.floor_us)
        if derived is None:
            return 2
        runs.append(derived)

    return compose_and_emit(runs)


def check_constants():
    """把本文件复制的档位常量与 `operator_agent` 里那份逐字比。零 GPU，需要能 import 两侧。

    单文件 scp 的代价是常量被抄了两份；这里是那笔账的守卫 —— 抄错一个数不会崩，
    只会让「这一轮落哪一档」与生产判定悄悄不一致。仓里的读者是
    `tests/unit/test_dispersion_thresholds.py`：它走 AST 读本文件的赋值，不 import
    本模块（本模块要真 triton 才能导入，见文件头的单文件约束），所以本机也能跑。
    """
    from operator_agent.decide import measurement as m
    pairs = (("LAUNCH_DOMINANCE_VALID", LAUNCH_DOMINANCE_VALID, m.LAUNCH_DOMINANCE_VALID),
             ("LAUNCH_DOMINANCE_UNINFORMATIVE", LAUNCH_DOMINANCE_UNINFORMATIVE,
              m.LAUNCH_DOMINANCE_UNINFORMATIVE),
             ("DISPERSION_DEGRADED_INHERITED", DISPERSION_DEGRADED_INHERITED,
              m.DISPERSION_DEGRADED),
             ("DISPERSION_UNINFORMATIVE", DISPERSION_UNINFORMATIVE,
              m.DISPERSION_UNINFORMATIVE),
             ("INHERITED_UTIL_PCT", INHERITED_UTIL_PCT, m.CONTAMINATION_UTIL_PCT))
    bad = [(n, a, b) for n, a, b in pairs if a != b]
    band = {"valid": m.NOISE_BAND[m.Validity.VALID],
            "degraded": m.NOISE_BAND[m.Validity.DEGRADED],
            "uninformative": m.NOISE_BAND[m.Validity.UNINFORMATIVE]}
    if band != INHERITED_BAND:
        bad.append(("INHERITED_BAND", INHERITED_BAND, band))
    for name, mine, theirs in bad:
        print("MISMATCH %-32s 本文件=%r  operator_agent=%r" % (name, mine, theirs))
    if bad:
        print("%d 处常量已漂移：先改本文件再标定" % len(bad))
        return 1
    print("常量一致：%d 项逐字相同（含 NOISE_BAND 三档）" % (len(pairs) + 1))
    return 0


def derive(records, floor_us=None):
    """把逐核记录折成 G4 要的三个阈值，并打 `PROBE_JSON` 行。

    三段的配方都在 `measurement.py` 里有出处，不在这里新造：
      * `band[档]`      = 3 × 该档各核「跨轮 median 的相对 iqr」的最大值
      * `dispersion_degraded` = 各核「轮内 iqr/median」合并后的 p95（往上一档取整）
      * `contamination_util_pct` = 静止窗 util 观测（见 `contamination_threshold`）

    **launch 地板对照核不进散布池**：它存在的理由是给地板，不是当候选；在 40 µs 量级上
    iqr/median 由事件计时粒度支配，混进来会把这个阈值抬到没有信号。它的值照样打印。
    """
    # 地板：优先用 G1 的实测值，缺省用空核的最小 median
    control = [r for r in records if r["is_floor_control"]]
    floor_from_control = min(control[0]["medians_ms"]) * 1000.0 if control else None
    floor = floor_us if floor_us is not None else floor_from_control
    print("\n" + "=" * 68)
    print("G4 的三个阈值")
    if floor_us is not None:
        print(f"  launch 地板（--floor-us 给定，G1 口径）= {floor:.3f} us；"
              f"本机空核最小 median = {floor_from_control:.3f} us")
    else:
        print(f"  launch 地板（空核最小 median 代替 G1 口径）= {floor:.3f} us "
              f"（同一把尺：本脚本每次计时前都冲 256MB，与 runner `_timed_runs` 同口径）")

    pool = [r for r in records if not r["is_floor_control"]]
    shortest = min((percentile(r["medians_ms"], 50) for r in pool), default=None)
    if floor_us is not None and shortest and floor_us / 1000.0 > shortest:
        # 地板比某个核的实测中位数还长 ⇒ 两者不是同一把尺。2026-11-01 在本卡上撞到过：
        # `measure_launch_floor.py` 的 bench() 不冲刷 L2、每次都把流跑干，量到的 38.9us
        # 主要是 CPU 派发延迟；而本脚本（与 runner `_timed_runs` 同口径）在 start.record()
        # 之前先冲 256MB，量到的空核是 3.07us。混用会算出「地板支配 292%」这种物理上
        # 不可能的读数，再往下就全是错的档位。
        print("  ✗ --floor-us %.3f us 大于最短核的实测 median %.3f us：两个读数不是同一把尺。"
              % (floor_us, shortest * 1000))
        print("    `measure_launch_floor.py` 不冲刷 L2（量的是 CPU 派发），本脚本在 "
              "start.record() 之前冲 256MB（与 run_kernel._timed_runs 同口径）。")
        print("    档位判定要么两边都冲刷、要么两边都不冲刷；混用请去掉 --floor-us。")
        return None

    cross_by_kernel, tier_of_kernel = {}, {}
    cross_median_by_kernel, cross_mean_by_kernel = {}, {}
    for rec in pool:
        med = percentile(rec["medians_ms"], 50)
        cross_med = relative_iqr(rec["medians_ms"])
        # 均值那一版：median 落在 Event 的量化步长上时会给出 0（见 measure_batch 的注），
        # 均值把分辨率抬高 SAMPLES 倍。两者取大 —— 偏大只会让判决更保守。
        cross_mean = relative_iqr(rec["means_ms"])
        candidates = [v for v in (cross_med, cross_mean) if v is not None]
        cross = max(candidates) if candidates else None
        disp = percentile(rec["iqrs"], 50) if rec["iqrs"] else None
        tier, dom = tier_for(med, floor, disp)
        cross_by_kernel[rec["name"]] = cross
        cross_median_by_kernel[rec["name"]] = cross_med
        cross_mean_by_kernel[rec["name"]] = cross_mean
        tier_of_kernel[rec["name"]] = tier
        print(f"  {rec['name'][:40]:<42} median={med * 1000:9.1f}us "
              f"地板支配={('%.1f%%' % (dom * 100)) if dom is not None else 'n/a':>7} "
              f"轮内 iqr/median={('%.4f' % disp) if disp is not None else 'n/a':>7} "
              f"跨轮(median/均值)={('%.4f' % cross_med) if cross_med is not None else 'n/a'}"
              f"/{('%.4f' % cross_mean) if cross_mean is not None else 'n/a'} → {tier}")

    band, band_notes = {}, []
    for tier in TIERS:
        mine = [cross_by_kernel[n] for n, t in tier_of_kernel.items() if t == tier]
        mine = [v for v in mine if v is not None]
        if mine:
            value = ceil_to(BAND_MULTIPLIER * max(mine), 4)
            band_notes.append("%s: 3 × %.4f（%d 个核）" % (tier, max(mine), len(mine)))
        else:
            # 该档没有代表核：取所有核里最大的跨轮散布。偏大 = 判决更保守（更多 SAME），
            # 不会把一次真改善误判成噪声之外的东西 —— 与仓库里「拿不准就取大」同向。
            allv = [v for v in cross_by_kernel.values() if v is not None]
            value = ceil_to(BAND_MULTIPLIER * max(allv), 4) if allv else None
            band_notes.append("%s: 无代表核 ⇒ 退到全体最大值（偏大，保守）" % tier)
        if value is None:
            print("  band.%s 无法给出：没有可用的跨轮散布读数" % tier)
            return None
        if value <= 0:
            # 带宽 0 = 「任何一点差别都算显著」。SLOW 那一支会据此**永久禁用配方**，
            # 所以这一格宁可停下：它不是「本机特别稳」，是这台机的这档没有可测的散布。
            print("  ✗ band.%s = %g：零带宽会把任意抖动判成快/慢（SLOW 还会禁用配方）。"
                  "这一档在本机没有可分辨的跨轮散布，先扩样本数/换更大的 shape 再标。"
                  % (tier, value))
            return None
        if value == INHERITED_BAND.get(tier):
            bumped = ceil_to(value + 1e-4, 4)
            print("  ⚠️ band.%s 实测值 %g 与仓内继承值逐位相同（G4 会判「抄的」）："
                  "按一个可表示档位往上取 %g" % (tier, value, bumped))
            value = bumped
        band[tier] = value
    print("  判决带宽 band = {valid: %g, degraded: %g, uninformative: %g}"
          % (band["valid"], band["degraded"], band["uninformative"]))
    for note in band_notes:
        print("    - " + note)

    # 散布门限：轮内 iqr/median 合并后的 p95
    merged = sorted(v for r in pool for v in r["iqrs"])
    disp_p95 = percentile(merged, 95) if merged else None
    if disp_p95 is None:
        print("  dispersion_degraded 无法给出：没有轮内散布读数")
        return None
    dispersion_degraded = ceil_to(disp_p95, 3)
    if dispersion_degraded == INHERITED_DISPERSION_DEGRADED:
        dispersion_degraded = ceil_to(disp_p95 + 1e-3, 3)
        print("  ⚠️ dispersion_degraded 实测 p95 与继承值 0.10 同值 ⇒ 往上取 %g"
              % dispersion_degraded)
    if dispersion_degraded >= DISPERSION_UNINFORMATIVE:
        print("  ⚠️ dispersion_degraded(%.3f) 已到 uninformative 档(%.2f)：这台机的常态抖动"
              "比档位阶梯还宽，阈值再高就是没有信号" % (dispersion_degraded,
                                                       DISPERSION_UNINFORMATIVE))
    print(f"  dispersion_degraded = {dispersion_degraded}（合并 p50={percentile(merged, 50):.4f} "
          f"p95={disp_p95:.4f} p99={percentile(merged, 99):.4f}，n={len(merged)}）")

    quiet = quiet_util_samples()
    util_pct, util_why = contamination_threshold(quiet)
    if util_pct is None:
        print("  contamination_util_pct 无法给出：" + util_why)
        return None
    print(f"  静止窗 util 采样 = {quiet}")
    print(f"  contamination_util_pct = {util_pct}（{util_why}）")
    if util_pct == INHERITED_UTIL_PCT:
        print("  ⚠️ contamination_util_pct 与继承值 5.0 同值 ⇒ G4 会判「抄的」")

    return {
        "band": band,
        "contamination_util_pct": util_pct,
        "dispersion_degraded": dispersion_degraded,
        "dispersion_uninformative": DISPERSION_UNINFORMATIVE,
        "cross_iqr_by_kernel": {k: (round(v, 6) if v is not None else None)
                                for k, v in cross_by_kernel.items()},
        "cross_median_by_kernel": {k: (round(v, 6) if v is not None else None)
                                   for k, v in cross_median_by_kernel.items()},
        "cross_mean_by_kernel": {k: (round(v, 6) if v is not None else None)
                                 for k, v in cross_mean_by_kernel.items()},
        "tier_of_kernel": tier_of_kernel,
        "launch_floor_us_used": round(floor, 3) if floor else None,
        "quiet_util_samples": quiet,
        "band_rule": "3 × max(跨轮 median 的相对 iqr, 跨轮均值的相对 iqr)"
                     "（前者是 measurement.NOISE_BAND 的同一配方；后者防 median 落在 "
                     "Event 量化步长上给出 0，取大即保守）",
        "dispersion_rule": "轮内 iqr/median 合并 p95，向上取整到 3 位",
        "measured_by": "benchmarks/calibration/calibrate_dispersion_thresholds.py",
    }


def compose_and_emit(runs):
    """把多轮标定合成一份阈值并打 `PROBE_JSON` 行。

    合成用**跨轮中位数**，不是最大值 —— 这是被数据推翻过一次的选择，记在这里免得再犯：
    2026-11-01 在本卡上跑 3 轮，`band.uninformative` 三轮是 0.0386 / 0.0573 / 0.3341，
    取最大得到 **0.3341**：那等于「差 33% 都算没变」，比没有阈值更糟。最大值不是估计量，
    它是上端点，一轮离群就能绑架整份标定；中位数对离群稳健。

    `contamination_util_pct` 仍取**最小**，方向与上面两个相反：它是「util 超过此值就算被
    污染」的门限，取大就看不见对手 —— 本卡第一次标定时正是自污染把它抬到了 71%。

    三轮之间的实际差异照原样写进 `runs`，不靠中位数把波动藏起来。
    """
    def _median(values):
        return percentile(sorted(values), 50)

    per_run = [{"band": dict(r["band"]),
                "dispersion_degraded": r["dispersion_degraded"],
                "contamination_util_pct": r["contamination_util_pct"]} for r in runs]
    print("\n" + "#" * 68)
    print("### %d 轮标定的合成（跨轮中位数；污染阈值取最小）" % len(runs))
    for i, rec in enumerate(per_run, 1):
        print("  第 %d 轮: band valid=%.4f degraded=%.4f uninformative=%.4f | "
              "dispersion_degraded=%.3f | contamination=%.1f"
              % (i, rec["band"]["valid"], rec["band"]["degraded"],
                 rec["band"]["uninformative"], rec["dispersion_degraded"],
                 rec["contamination_util_pct"]))

    band = {t: ceil_to(_median([r["band"][t] for r in runs]), 4) for t in TIERS}
    dispersion_degraded = ceil_to(_median([r["dispersion_degraded"] for r in runs]), 3)
    util_pct = min(r["contamination_util_pct"] for r in runs)
    print("  合成结果: band={valid: %g, degraded: %g, uninformative: %g} "
          "dispersion_degraded=%g contamination_util_pct=%g"
          % (band["valid"], band["degraded"], band["uninformative"],
             dispersion_degraded, util_pct))

    if len(runs) >= 2:
        for tier in TIERS:
            vals = [r["band"][tier] for r in runs]
            spread = (max(vals) / min(vals)) if min(vals) else float("inf")
            if spread >= 3:
                print("  ⚠️ band.%s 跨轮相差 %.1f 倍（%s）：这一档在本机没有稳定的跨轮散布，"
                      "中位数只是当前最好的估计"
                      % (tier, spread, "/".join("%.4f" % v for v in vals)))
    for tier in TIERS:
        if band[tier] >= DISPERSION_UNINFORMATIVE:
            print("  ⚠️ band.%s = %g 已宽到 uninformative 档（%.2f）以上：该档在这次标定里"
                  "几乎没有可分辨的散布。注意 `noise_band_for()` 对 UNINFORMATIVE 返回 "
                  "None，所以这条带宽**从不参与判决**，只占表位"
                  % (tier, band[tier], DISPERSION_UNINFORMATIVE))
        if band[tier] == INHERITED_BAND.get(tier):
            print("  ⚠️ band.%s 合成值 %g 与继承值逐位相同 ⇒ G4 会判「抄的」"
                  % (tier, band[tier]))
    if dispersion_degraded == INHERITED_DISPERSION_DEGRADED:
        print("  ⚠️ dispersion_degraded 合成值 0.10 与继承值逐位相同")
    if util_pct == INHERITED_UTIL_PCT:
        print("  ⚠️ contamination_util_pct 合成值 5.0 与继承值逐位相同")

    base = runs[0]
    noise = {k: base[k] for k in
             ("cross_iqr_by_kernel", "cross_median_by_kernel", "cross_mean_by_kernel",
              "tier_of_kernel", "launch_floor_us_used", "quiet_util_samples",
              "band_rule", "dispersion_rule", "measured_by")}
    noise.update({"band": band,
                  "contamination_util_pct": util_pct,
                  "dispersion_degraded": dispersion_degraded,
                  "dispersion_uninformative": DISPERSION_UNINFORMATIVE,
                  "runs": per_run,
                  "compose_rule": "band 与 dispersion 取跨轮中位数（最大值曾被单轮离群绑架："
                                  "uninformative 0.0386/0.0573/0.3341 → 取最大 0.3341），"
                                  "污染阈值取最小（取大即看不见对手）",
                  "diagnostics_from_run": 1})

    props = torch.cuda.get_device_properties(0)
    uuid = _gpu_uuid()
    if not uuid:
        print("  ✗ 取不到 nvidia-smi UUID：没有来源戳的阈值不许进表")
        return 2
    print("\n  PROBE_JSON 里的 noise 段（G4 直接读这三格）:")
    print("    band=%r contamination_util_pct=%r dispersion_degraded=%r"
          % (band, util_pct, dispersion_degraded))
    print("PROBE_JSON " + json.dumps({
        "uuid": uuid, "name": props.name, "cc": "%d.%d" % (props.major, props.minor),
        "noise": noise,
        "measured_by": "benchmarks/calibration/calibrate_dispersion_thresholds.py",
        "torch_version": torch.__version__,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _gpu_uuid():
    """被测卡的 UUID（`GPU-…` 形）。取不到返回 None —— 采集器会因此拒绝盖来源戳。

    优先问 torch：钉卡之后**可见索引 0 就是被测卡**，而 `nvidia-smi` 的第一行不是
    （CUDA_VISIBLE_DEVICES 只对 CUDA 运行时生效）。2026-11-04 在 8 卡共享 H20 上，
    CUDA_VISIBLE_DEVICES=3 时按 `nvidia-smi` 第一行会把散布带盖成物理 0 号卡。
    """
    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        value = getattr(props, "uuid", None)
        if value:
            value = str(value)
            return value if value.startswith("GPU-") else "GPU-" + value
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=gpu_uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None


if __name__ == "__main__":
    raise SystemExit(main())
