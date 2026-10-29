"""Baseline Triton + quick config probe: where does the space sit vs cuBLAS?"""
import torch
import triton

from bench_common import N, setup, time_fn, parquet_free_report, _flush_buf
from triton_ref import matmul

dev = torch.device("cuda:0")
a, b, c = setup()
flush = _flush_buf(dev, 256 << 20)
ref = torch.matmul(a, b)

# correctness of the reference triton kernel first
out = matmul(a, b)
err = (out.float() - ref.float()).abs().max().item()
print(f"triton ref max abs err vs cublas: {err:.5f} (fp16 accum values ~ +-64)")

configs = [
    # (BM, BN, BK, GROUP_M, warps, stages)
    (128, 128, 32, 8, 8, 3),
    (128, 128, 64, 8, 8, 3),
    (128, 128, 64, 8, 8, 4),
    (128, 256, 64, 8, 8, 3),
    (256, 128, 64, 8, 8, 3),
    (128, 128, 128, 8, 8, 3),
    (64, 256, 64, 8, 8, 3),
    (256, 128, 32, 8, 8, 3),
]
results = []
for bm, bn, bk, gm, w, s in configs:
    try:
        f = lambda: matmul(a, b, bm, bn, bk, gm, w, s)
        o = f()
        e = (o.float() - ref.float()).abs().max().item()
        st = time_fn(f, iters=60, warmup=15, flush=flush)
        tag = f"w{w}s{s} err{e:.3f}"
        print(parquet_free_report(f"tri {bm}x{bn}x{bk} g{gm}", st, tag))
        results.append((st["median_ms"], (bm, bn, bk, gm, w, s)))
    except Exception as ex:
        print(f"tri {bm}x{bn}x{bk} g{gm} w{w}s{s} FAILED: {type(ex).__name__}: {str(ex)[:160]}")

results.sort()
print("\nbest:", results[0] if results else None)
print("cublas median: 1.036 ms")