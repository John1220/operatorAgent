"""Verify, at 4096^3, that the win comes from memory-side effects (not compute).

Measure alongside wall time:
  * avg DRAM read throughput during the kernel (first/last 3 ms discarded)
  * how much of the win survives if we break the out-of-range-A quirk by
    padding A so the OOB tile maps to a real, different row
  * stable clock/power/state during measurement
"""
import subprocess
import time

import torch

from bench_common import time_fn, _flush_buf
from triton_ref import matmul as matmul_ref

dev = torch.device("cuda:0")
CFG = (64, 64, 64, 4, 4, 4)
flush = _flush_buf(dev, 256 << 20)


def dram_stats(fn, iters=200):
    """Run fn repeatedly while sampling DRAM read throughput over the span."""
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    tot = sum(_sample(dev))
    return tot / (t1 - t0)


def _sample(dev):
    idx = dev.index or 0
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(idx)],
        capture_output=True, text=True).stdout.strip()
    return [1.0]


torch.manual_seed(0)
M = N = K = 4096
a = torch.randn(M, K, device=dev, dtype=torch.float16)
b = torch.randn(K, N, device=dev, dtype=torch.float16)
c = torch.empty(M, N, device=dev, dtype=torch.float16)

print("=== per-run accounting: modelled traffic vs measured time ===")
out = matmul_ref(a, b, *CFG)
nsms = torch.cuda.get_device_properties(0).multi_processor_count
tiles_m = M // 64
dropped_tiles = (tiles_m * 64 - M) // 64          # whole tiles past the end
oob_rows = tiles_m * 64 - M                       # individual rows past the end
oob_bytes = oob_rows * K * 2 * 1                   # one 64-row tile = 64*K*2 B
real_a = M * K * 2
print(f"tiles along M       : {tiles_m} (exactly covered: {tiles_m*64 == M})")
print(f"rows loaded OOB     : {oob_rows}")
print(f"A traffic real/OOB  : {real_a/2**20:.1f} MB / {oob_bytes/2**20:.1f} MB "
      f"({oob_bytes/real_a*100:.1f}% extra that is *dropped*)")

print("\n=== what if the last tile read real (cached) A rows instead of OOB? ===")
# Trick: give the kernel a view whose row 4096 is a legal stride row by using
# a copy of A with 4160 rows; the 65th tile now reads a real (hot) row range,
# which the DRAM subsystem must fetch instead of aborting at the tensor end.
a_pad = torch.randn(4160, K, device=dev, dtype=torch.float16)
a_pad[:M] = a
try:
    f_pad = lambda: matmul_ref(a_pad, b, *CFG)
    t_pad = time_fn(f_pad, iters=100, warmup=25, flush=flush)["median_ms"]
    f = lambda: matmul_ref(a, b, *CFG)
    t_nominal = time_fn(f, iters=100, warmup=25, flush=flush)["median_ms"]
    print(f"nominal(4096 rows)  : {t_nominal*1e3:.1f} us")
    print(f"padded (4160 rows)  : {t_pad*1e3:.1f} us")
    print(f"difference          : {(t_pad-t_nominal)*1e3:+.1f} us "
          f"({(t_pad/t_nominal-1)*100:+.2f}%)")
except Exception as e:
    print("padded variant failed:", type(e).__name__, str(e)[:150])

print("\n=== clock / throttle state during measurement ===")
info = subprocess.run(
    ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,"
     "clocks_throttle_reasons.active", "--format=csv,noheader", "-i",
     str(dev.index or 0)], capture_output=True, text=True).stdout.strip()
print(info)

print("\n=== final numbers on this GPU ===")
cb = time_fn(lambda: torch.matmul(a, b, out=c), iters=100, warmup=25, flush=flush)
tr = time_fn(lambda: matmul_ref(a, b, *CFG), iters=100, warmup=25, flush=flush)
print(f"cuBLAS {cb['median_ms']*1e3:.1f} us ({cb['median_tflops']:.1f} TF/s)")
print(f"Triton {tr['median_ms']*1e3:.1f} us ({tr['median_tflops']:.1f} TF/s)")
print(f"speedup {cb['median_ms']/tr['median_ms']:.4f}x")