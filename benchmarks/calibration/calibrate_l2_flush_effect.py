"""在真机上标定: L2 冲刷对 CUDA-event 计时窗口的影响

背景: 给 runner 加上 Triton do_bench 式的 L2 冲刷后, layernorm 1024×768 的候选
median 从 32.5 µs 掉到 7.2 µs, 而 launch 地板是 20.24 µs —— 于是
`launch_dominance = floor / median` 算出了 2.81, 一个物理上不可能的值 (>1)。

假设: 冲刷用的 `buf.zero_()` 要写 256 MB, 在 GPU 上是 ~70 µs 的活。它排在
`start.record()` 之前且同流有序, 所以 GPU 在记录 start 时是**忙**的 —— 下一条
kernel 的 launch 命令早已入队, 事件窗口因此量不到 launch 等待, 只剩纯执行时间。
若不冲刷, GPU 在 start 时是空闲的, 事件窗口会把 launch 延迟整段吃进去。

这个脚本就是去证/证伪这件事: 同一个核, 冲刷与不冲刷各测一次, 再加一个空核做
对照。若「空核不冲刷 ≈ 20 µs、冲刷 ≈ 1-2 µs」, 假设成立, 那么
`launch_dominance` 的分母口径必须改。

在远端 GPU 机器上跑 (需要 torch/triton):
    ssh <别名> 'cd ~/operator_agent && CUDA_VISIBLE_DEVICES=<空闲卡> \
        ~/.local/bin/uv run python -' < benchmarks/calibration/calibrate_l2_flush_effect.py
"""
import statistics

import torch
import triton
import triton.language as tl

FLUSH_BYTES = 256 * 1024 * 1024
_flush_buf = torch.empty(FLUSH_BYTES // 4, dtype=torch.int, device="cuda")

print(f"torch={torch.__version__} triton={triton.__version__} "
      f"device={torch.cuda.get_device_name(0)}")
print(f"冲刷缓冲 = {FLUSH_BYTES // 1024 // 1024} MB")


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
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols, (xc * rstd * w + b).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BM + tl.arange(0, BM)
    off_n = pid_n * BN + tl.arange(0, BN)
    a_ptrs = a_ptr + off_m[:, None] * sam + tl.arange(0, BK)[None, :] * sak
    b_ptrs = b_ptr + tl.arange(0, BK)[:, None] * sbk + off_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    tl.store(c_ptr + off_m[:, None] * scm + off_n[None, :] * scn, acc.to(tl.float16))


def bench(fn, n_warmup, n_repeat, flush):
    """与 runner 的 _timed_runs 同口径: CUDA event, 每轮 synchronize, 取 median"""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_repeat):
        if flush:
            _flush_buf.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)   # µs
    return statistics.median(ts), min(ts)


def bench_flush_cost(n_repeat=50):
    """单独量一次 zero_() 的 GPU 耗时, 用来判断冲刷本身的代价"""
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _flush_buf.zero_()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return statistics.median(ts)


print("\n[0] 冲刷本身的 GPU 耗时 (256 MB zero_)")
cost = bench_flush_cost()
print(f"    median = {cost:.2f} µs   → 折算带宽 {FLUSH_BYTES / (cost * 1e-6) / 1e12:.2f} TB/s")

# ── 空核: 假设的决定性对照 ──
x = torch.zeros(1024, device="cuda")
print("\n[1] 空 Triton 核 (只有 launch, 没有任何计算)")
for flush in (False, True):
    med, mn = bench(lambda: _empty[(1,)](x), n_warmup=200, n_repeat=500, flush=flush)
    print(f"    flush={str(flush):5s}  median={med:8.2f} µs   min={mn:8.2f} µs")

# ── layernorm 1024×768: 项目里判过 uninformative 的那一档 ──
rows, cols = 1024, 768
X = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
W = torch.ones(cols, device="cuda", dtype=torch.float16)
B = torch.zeros(cols, device="cuda", dtype=torch.float16)
Y = torch.empty_like(X)
BLOCK = triton.next_power_of_2(cols)
traffic_mb = rows * cols * 2 * 2 / 1e6
print(f"\n[2] layernorm {rows}×{cols} fp16  (访存 {traffic_mb:.2f} MB, "
      f"roofline 下界 {traffic_mb * 1e6 / 3.68e12 * 1e6:.2f} µs)")
for flush in (False, True):
    med, mn = bench(lambda: _ln[(rows,)](X, W, B, Y, rows, cols, 1e-5, BLOCK=BLOCK),
                    n_warmup=200, n_repeat=500, flush=flush)
    print(f"    flush={str(flush):5s}  median={med:8.2f} µs   min={mn:8.2f} µs")

# ── matmul 4096³: 判过 valid 的那一档 ──
n = 4096
A = (torch.randn(n, n, device="cuda", dtype=torch.float16) * 0.1)
Bm = (torch.randn(n, n, device="cuda", dtype=torch.float16) * 0.1)
C = torch.empty(n, n, device="cuda", dtype=torch.float16)
BM, BN, BK = 64, 64, 32
grid = (triton.cdiv(n, BM), triton.cdiv(n, BN))
print(f"\n[3] matmul {n}³ fp16  (访存 {3 * n * n * 2 / 1e6:.1f} MB)")
for flush in (False, True):
    med, mn = bench(lambda: _mm[grid](A, Bm, C, n, n, n, A.stride(0), A.stride(1),
                                      Bm.stride(0), Bm.stride(1), C.stride(0), C.stride(1),
                                      BM=BM, BN=BN, BK=BK),
                    n_warmup=20, n_repeat=50, flush=flush)
    print(f"    flush={str(flush):5s}  median={med:8.2f} µs   min={mn:8.2f} µs")

# ── PyTorch eager 基线同样两种口径, 用来判断加速比会不会翻 ──
print(f"\n[4] layernorm {rows}×{cols} 的 PyTorch Eager 基线")
for flush in (False, True):
    med, mn = bench(lambda: torch.nn.functional.layer_norm(X, (cols,), W, B, 1e-5),
                    n_warmup=200, n_repeat=500, flush=flush)
    print(f"    flush={str(flush):5s}  median={med:8.2f} µs   min={mn:8.2f} µs")

print("\n[5] 判读")
print("    若空核 flush=False 远大于 flush=True, 说明不冲刷时事件窗口吃进了 launch 延迟,")
print("    冲刷把 GPU 填满从而把 launch 藏在窗口之外 —— 那么 median 的口径就变了,")
print("    `launch_dominance = floor / median` 的分母不再是「含 launch 的端到端时间」。")
