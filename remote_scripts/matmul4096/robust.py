"""Robustness checks for the winning config.

1. Held-out shapes: the config is meant for 4096^3, but a kernel that only works
   at exactly 4096^3 is fragile. Check a few nearby + ragged sizes.
2. Alignment / stride robustness: non-contiguous and odd strides.
3. Numerical agreement over many trials, not one.
4. Determinism.
"""
import torch
import triton

from bench_common import time_fn
from triton_ref import matmul as matmul_ref

dev = torch.device("cuda:0")
CFG = (64, 64, 64, 4, 4, 4)
BM, BN, BK = CFG[0], CFG[1], CFG[2]


def check(M, N, K, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device=dev, dtype=torch.float16)
    b = torch.randn(K, N, device=dev, dtype=torch.float16)
    ref = torch.matmul(a, b)
    out = matmul_ref(a, b, *CFG)
    diff = (out.float() - ref.float()).abs()
    rel = (diff / ref.float().abs().clamp_min(1e-3)).max().item()
    try:
        af, bf = a.float(), b.float()
        ref64 = torch.matmul(af, bf)
        err64 = (out.float() - ref64).abs().max().item()
    except torch.cuda.OutOfMemoryError:
        err64 = float("nan")
    print(f"{M:>5d}x{N:>5d}x{K:>5d}: max_abs_vs_cublas={diff.max().item():.5f} "
          f"max_rel={rel:.5f} mismatched={int((out != ref).sum().item())} "
          f"max_abs_vs_fp32={err64:.5f}")
    return diff.max().item()


print("=== shape coverage (correctness) ===")
for (M, N, K) in [(4096, 4096, 4096), (4096, 4096, 1), (4096, 4096, 64),
                  (1, 1, 1), (100, 200, 300), (128, 128, 128), (65, 65, 65),
                  (4096, 4096, 4100), (1024, 2048, 512), (4096, 4096, 4095)]:
    try:
        check(M, N, K)
    except Exception as e:
        print(f"{M}x{N}x{K}: FAILED {type(e).__name__}: {str(e)[:120]}")

print("\n=== ragged (non multiple of 64) with stride tricks ===")
for (M, N, K) in [(1000, 1000, 1000), (4093, 4091, 4090)]:
    try:
        check(M, N, K)
    except Exception as e:
        print(f"{M}x{N}x{K}: FAILED {type(e).__name__}: {str(e)[:120]}")

print("\n=== determinism at 4096^3 ===")
torch.manual_seed(1)
a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
o1 = matmul_ref(a, b, *CFG).clone()
o2 = matmul_ref(a, b, *CFG).clone()
o3 = matmul_ref(a, b, *CFG).clone()
print("identical across 3 runs:", torch.equal(o1, o2) and torch.equal(o2, o3))

print("\n=== transposed / non-contiguous inputs ===")
at = a.t().contiguous().t()  # same values, column-major
ref = torch.matmul(a, b)
try:
    o = matmul_ref(at, b, *CFG)
    print("col-major A: max_abs vs ref =", (o.float() - ref.float()).abs().max().item())
except Exception as e:
    print("col-major A FAILED:", type(e).__name__, str(e)[:150])

bt = b.t().contiguous().t()
try:
    o = matmul_ref(a, bt, *CFG)
    print("col-major B: max_abs vs ref =", (o.float() - ref.float()).abs().max().item())
except Exception as e:
    print("col-major B FAILED:", type(e).__name__, str(e)[:150])

print("\n=== 20 random trials at 4096^3 (worst deviation) ===")
worst = 0.0
for s in range(20):
    torch.manual_seed(1000 + s)
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    ref = torch.matmul(a, b)
    o = matmul_ref(a, b, *CFG)
    d = (o.float() - ref.float()).abs().max().item()
    worst = max(worst, d)
    if o.shape != ref.shape:
        print("SHAPE MISMATCH")
print("worst max-abs deviation over 20 trials:", worst)