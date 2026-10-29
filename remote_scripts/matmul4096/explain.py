"""Closure experiment: why is 64x64x64/w4/s4 faster than cuBLAS *only* at 4096^3?

Conjecture (inductively derived from the measurements, to be tested):
  (a) cuBLAS's chosen kernel leaves DRAM bandwidth on the table (its tile is
      larger than the tiny CTA budget needs), while the small 64x64 tile
      maximises the number of CTAs resident -> more memory-level parallelism.
  (b) The Triton kernel's false dependency on out-of-range A rows at `load C`
      is a pure win here: ~10.7% of A's DRAM traffic never happens.
  (c) Both effects only pay off while the problem is DRAM-bound; once the
      working set is small enough to be L2-resident, compute wins and the
      small-tile kernel falls behind.

Each experiment flips one variable.
"""
import torch

from bench_common import time_fn, tflops, _flush_buf
from triton_ref import matmul as matmul_ref

dev = torch.device("cuda:0")
CFG = (64, 64, 64, 4, 4, 4)
flush = _flush_buf(dev, 256 << 20)


def bench(M, N, K, cfg=CFG, iters=60):
    torch.manual_seed(0)
    a = torch.randn(M, K, device=dev, dtype=torch.float16)
    b = torch.randn(K, N, device=dev, dtype=torch.float16)
    c = torch.empty(M, N, device=dev, dtype=torch.float16)
    f_cb = lambda: torch.matmul(a, b, out=c)
    f_tr = lambda: matmul_ref(a, b, *cfg)
    cb = time_fn(f_cb, iters=iters, warmup=20, flush=flush)["median_ms"]
    tr = time_fn(f_tr, iters=iters, warmup=20, flush=flush)["median_ms"]
    flops = 2 * M * N * K
    return cb, tr, flops / (cb * 1e-3) / 1e12, flops / (tr * 1e-3) / 1e12


print("=== E1: the 'redundant A traffic' effect vs shape alignment ===")
print("K below 4096 makes the out-of-range tile touch real data; watch Triton TF/s.")
print(f"{'shape':<22s} {'cuBLAS ms':>10s} {'Triton ms':>10s} {'cb TF/s':>9s} {'tr TF/s':>9s} {'tr/cb':>7s}")
for K in [4096, 4000, 3000, 2048, 1024, 512, 64]:
    M = N = 4096
    cb, tr, tcb, ttr = bench(M, N, K)
    print(f"{M}x{N}x{K:<10d} {cb:>10.4f} {tr:>10.4f} {tcb:>9.1f} {ttr:>9.1f} {ttr/tcb:>6.3f}")

print("\n=== E2: same kernel, different configs across sizes (regime map) ===")
cfgs = [(64, 64, 64, 4, 4, 4), (64, 64, 32, 4, 4, 3), (128, 128, 64, 8, 8, 3),
        (128, 256, 64, 8, 8, 3)]
print(f"{'shape':<20s} {'cuBLAS':>9s} " + " ".join(f"{str(c[:3]):>16s}" for c in cfgs))
for (M, N, K) in [(4096, 4096, 4096), (2048, 2048, 2048), (1024, 1024, 1024),
                  (8192, 8192, 8192)]:
    torch.manual_seed(0)
    a = torch.randn(M, K, device=dev, dtype=torch.float16)
    b = torch.randn(K, N, device=dev, dtype=torch.float16)
    c = torch.empty(M, N, device=dev, dtype=torch.float16)
    cb = time_fn(lambda: torch.matmul(a, b, out=c), iters=40, warmup=15,
                 flush=flush)["median_ms"]
    cells = []
    for cfg in cfgs:
        try:
            t = time_fn(lambda cfg=cfg: matmul_ref(a, b, *cfg), iters=40,
                        warmup=15, flush=flush)["median_ms"]
            cells.append(f"{t:9.4f}({cb/t:5.3f}x)")
        except Exception:
            cells.append("   n/a        ")
    print(f"{M:>5d}x{N:>5d}x{K:<5d} {cb:>9.4f} " + " ".join(cells))