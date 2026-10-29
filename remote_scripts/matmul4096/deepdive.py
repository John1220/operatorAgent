"""Deep-dive around the 64x64 tile region; check battier dependency + tail effects.

Key questions:
  1. Is 0.983 ms reproducible and is the margin stable (1.0538x is only just over)?
  2. Which 64x64 variants really win, and what separates them?
  3. Does L2 flush / graph timing change the ranking?  (Guard against a
     measurement artifact masquerading as a speedup.)
  4. What does the PTX look like for the winner?
"""
import json
import statistics

import torch
import triton

from bench_common import N, setup, time_fn, parquet_free_report, _flush_buf
from triton_ref import matmul as matmul_ref

dev = torch.device("cuda:0")
a, b, c = setup()
flush = _flush_buf(dev, 256 << 20)
ref = torch.matmul(a, b)


def cublas_ms(n=100, use_graph=False, fl=True):
    f = lambda: torch.matmul(a, b, out=c)
    return time_fn(f, iters=n, warmup=20, flush=flush if fl else None,
                   use_cuda_graph=use_graph)["median_ms"]


base = cublas_ms()
print(f"cuBLAS median (L2 flush, event timing): {base:.4f} ms\n")

cands = [
    (64, 64, 64, 4, 4, 4), (64, 64, 64, 8, 4, 4),
    (64, 64, 32, 4, 4, 3), (64, 64, 32, 4, 4, 4),
    (64, 64, 32, 8, 4, 3), (64, 64, 32, 8, 4, 4),
    (64, 128, 32, 8, 4, 3), (64, 128, 64, 8, 4, 3),
    (128, 64, 64, 8, 4, 3), (128, 64, 32, 8, 4, 4),
    (64, 64, 128, 4, 4, 3), (64, 64, 64, 4, 2, 4),
    (64, 64, 64, 4, 4, 4),  # duplicate: noise check
]

rows = []
for cfg in cands:
    bm, bn, bk, gm, w, s = cfg
    try:
        f = lambda: matmul_ref(a, b, bm, bn, bk, gm, w, s)
        out = f()
        err = (out.float() - ref.float()).abs().max().item()
        ev = time_fn(f, iters=100, warmup=25, flush=flush)
        gr = time_fn(f, iters=100, warmup=25, flush=flush, use_cuda_graph=True)
        rows.append({
            "cfg": cfg, "err": err,
            "med": ev["median_ms"], "sd": ev["stddev_ms"], "best": ev["best_ms"],
            "graph": gr["median_ms"],
            "spd": base / ev["median_ms"],
        })
    except Exception as ex:
        print(f"{cfg} FAILED {type(ex).__name__} {str(ex)[:100]}")

rows.sort(key=lambda r: r["med"])
print(f"{'config':<26s} {'err':>6s} {'med':>8s} {'sd':>7s} {'graph':>8s} {'TF/s':>7s} {'vs cuBLAS':>10s}")
for r in rows:
    bm, bn, bk, gm, w, s = r["cfg"]
    print(f"{bm}x{bn}x{bk} g{gm} w{w} s{s:<3d} {r['err']:>6.3f} {r['med']:>8.4f} "
          f"{r['sd']:>7.4f} {r['graph']:>8.4f} {2*N**3/r['med']/1e9:>7.1f} {r['spd']:>9.4f}x")

with open("deepdive.json", "w") as f:
    json.dump({"cublas_ms": base, "rows": [dict(r, cfg=list(r["cfg"])) for r in rows]}, f, indent=2)