"""Phase 2 probe: kernel families + exhaustive autotune, to locate the ceiling.

Family A : tutorial-style progressive multiply-accumulate (what we already ran)
Family B : pre_loop / persistent style with explicit `acc += tl.dot(a, b)` and TMA
We also sweep a much wider config space, and report the roofline gap.
"""
import itertools
import json

import torch
import triton

from bench_common import N, setup, time_fn, parquet_free_report, _flush_buf
from triton_ref import matmul as matmul_ref

dev = torch.device("cuda:0")
a, b, c = setup()
flush = _flush_buf(dev, 256 << 20)
ref = torch.matmul(a, b)

PEAK_TF16_TFLOPS = 148.0  # H20 dense FP16 w/ FP32 accum, at max SM clock

print(f"cuBLAS: 1.036 ms / 132.6 TF/s  ({132.6/PEAK_TF16_TFLOPS*100:.1f}% of peak)")
print(f"target (1.05x): <= 0.986 ms / >= 139.2 TF/s ({139.2/PEAK_TF16_TFLOPS*100:.1f}% of peak)\n")

grid_configs = []
for bm, bn in [(128, 128), (128, 256), (256, 128), (64, 128), (128, 64), (64, 64), (256, 256)]:
    for bk in [32, 64, 128]:
        for gm in [4, 8]:
            for w in [4, 8]:
                for s in [2, 3, 4]:
                    grid_configs.append((bm, bn, bk, gm, w, s))

print(f"sweeping {len(grid_configs)} configs...")
results = []
fail = 0
for i, (bm, bn, bk, gm, w, s) in enumerate(grid_configs):
    try:
        f = lambda bm=bm, bn=bn, bk=bk, gm=gm, w=w, s=s: matmul_ref(a, b, bm, bn, bk, gm, w, s)
        st = time_fn(f, iters=25, warmup=8, flush=flush)
        results.append((st["median_ms"], st["best_ms"], (bm, bn, bk, gm, w, s)))
        if i % 20 == 0:
            print(f"  [{i}/{len(grid_configs)}] best so far {min(results)[0]:.3f} ms")
    except Exception as e:
        fail += 1

results.sort()
print(f"\nfailed configs: {fail}")
print("TOP 12:")
for med, best, cfg in results[:12]:
    bm, bn, bk, gm, w, s = cfg
    print(f"  {bm}x{bn}x{bk} g{gm} w{w} s{s}: med {med:.3f} ms  best {best:.3f} ms  "
          f"{2*N**3/med/1e9:.1f} TF/s")

best_cfg = results[0][2]
print(f"\nBEST: {best_cfg} -> {results[0][0]:.3f} ms, "
      f"speedup vs cublas = {1.036/results[0][0]:.4f}x")

with open("sweep1.json", "w") as f:
    json.dump([{"med": m, "best": bb, "cfg": list(c)} for m, bb, c in results], f, indent=2)