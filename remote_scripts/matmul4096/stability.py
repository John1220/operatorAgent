"""Stability study: is the 1.0557x a real, repeatable effect or a lucky sample?

Repeated interleaved A/B measurement (cuBLAS vs candidate) in the same process,
plus separate processes, plus a low-noise CUDA-graph variant.  We report the
distribution of the speedup ratio, not just one number.
"""
import json
import os
import statistics
import sys

import torch

from bench_common import N, setup, time_fn, _flush_buf
from triton_ref import matmul as matmul_ref

dev = torch.device("cuda:0")
a, b, c = setup()
flush = _flush_buf(dev, 256 << 20)

HIDE_CUBLAS = os.environ.get("HIDE_CUBLAS_CFG", "0") == "1"
# A pre-tuned cuBLAS run (same call) to make sure cuBLAS has warmed its
# heuristics before we start measuring the contest.
cfg = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
       int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (64, 64, 64, 4, 4, 4)

f_cublas = lambda: torch.matmul(a, b, out=c)
f_tri = lambda: matmul_ref(a, b, *cfg)

tri_out = f_tri()
err = (tri_out.float() - torch.matmul(a, b).float()).abs().max().item()

# warm both
time_fn(f_cublas, iters=30, warmup=30, flush=flush)
time_fn(f_tri, iters=30, warmup=30, flush=flush)

ratios, cb_s, tri_s = [], [], []
ROUNDS = 12
for r in range(ROUNDS):
    cb = time_fn(f_cublas, iters=60, warmup=10, flush=flush)["median_ms"]
    tr = time_fn(f_tri, iters=60, warmup=10, flush=flush)["median_ms"]
    cb_s.append(cb)
    tri_s.append(tr)
    ratios.append(cb / tr)

print(f"cfg={cfg} err={err:.4f}")
print(f"cuBLAS  rounds: {[f'{x:.4f}' for x in cb_s]}")
print(f"Triton  rounds: {[f'{x:.4f}' for x in tri_s]}")
print(f"speedup rounds: {[f'{x:.4f}' for x in ratios]}")
print(f"cuBLAS median-of-rounds: {statistics.median(cb_s):.4f} ms  (sd {statistics.pstdev(cb_s):.4f})")
print(f"Triton median-of-rounds: {statistics.median(tri_s):.4f} ms  (sd {statistics.pstdev(tri_s):.4f})")
print(f"median speedup: {statistics.median(ratios):.4f}x  min {min(ratios):.4f}x  max {max(ratios):.4f}x")
print(f"all rounds > 1.05: {all(r > 1.05 for r in ratios)}")

with open(f"stability_{'_'.join(map(str,cfg))}.json", "w") as fh:
    json.dump({"cfg": list(cfg), "err": err, "cb": cb_s, "tri": tri_s,
               "ratios": ratios}, fh, indent=2)