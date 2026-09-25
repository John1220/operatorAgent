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
        ~/.local/bin/uv run python -' < benchmarks/calibrate_dispersion_thresholds.py
"""
import os
import statistics
import subprocess

import torch
import triton
import triton.language as tl

ROUNDS = 30            # 每个核重复多少轮
SAMPLES = 30           # 每轮多少个计时样本 (与 runner 的 repeat 默认值一致)
FLUSH_BYTES = 256 * 1024 * 1024
WARMUP_MS_BUDGET = 25  # 与 run_kernel.py 的 WARMUP_MS_BUDGET 保持一致

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


def measure_batch(fn, n=SAMPLES):
    """一轮: n 个样本 → (median_ms, iqr/median, max/median)"""
    times = sorted(timed_once(fn) for _ in range(n))
    median = percentile(times, 50)
    p25, p75 = percentile(times, 25), percentile(times, 75)
    if median <= 0:
        return median, None, None
    return median, (p75 - p25) / median, times[-1] / median


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
    medians, iqrs, maxes = [], [], []
    for _ in range(ROUNDS):
        utils.append(gpu_util_pct())          # 每轮都采: 单次快照对不上具体哪轮被抢
        m, i, x = measure_batch(fn)
        medians.append(m)
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
    return iqrs_s, maxes_s


def main():
    print(f"torch={torch.__version__} triton={triton.__version__} "
          f"device={torch.cuda.get_device_name(0)}")
    print(f"目标: 标定 measurement.py 的 DISPERSION_* 与 INTRUDER_MAX_OVER_MEDIAN")

    dev = "cuda"
    all_iqr, all_max = [], []

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

    cases = [
        ("空核 (launch 地板对照)", empty_fn),
        ("layernorm 1024x768 (小 shape, 曾被判 uninformative)", lambda: ln_fn(1024, 768)),
        ("layernorm 16384x6144 (大 shape)", lambda: ln_fn(16384, 6144)),
        ("matmul 4096^3 fp16", lambda: mm_fn(4096)),
    ]
    for name, make in cases:
        i, x = calibrate(name, make)
        all_iqr += i
        all_max += x

    print("\n" + "=" * 68)
    print("汇总 (所有核合并, 这才是阈值该看的分布)")
    all_iqr.sort()
    all_max.sort()
    for label, vals in (("iqr/median", all_iqr), ("max/median", all_max)):
        print(f"  {label:>11}: p50={percentile(vals, 50):.4f} "
              f"p90={percentile(vals, 90):.4f} p95={percentile(vals, 95):.4f} "
              f"p99={percentile(vals, 99):.4f} max={max(vals):.4f}")
    print("\n建议: DISPERSION_DEGRADED 取 iqr/median 的 p95 上方一点;")
    print("      INTRUDER_MAX_OVER_MEDIAN 取 max/median 的 p99 上方一点。")
    print("      两者都要高于本机常态, 否则信号每次运行都会触发, 等于没有信号。")


if __name__ == "__main__":
    main()
