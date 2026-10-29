"""Final validation harness for matmul_best.py.

Produces the numbers quoted in the report:
  * correctness at 4096^3 (bit-exact) + ragged shapes with the same kernel
  * interleaved cuBLAS-vs-Triton A/B timing, repeated rounds
  * final timing under three independent timing methods
"""
import json
import statistics

import torch

from bench_common import time_fn, _flush_buf, N
from matmul_best import matmul

dev = torch.device("cuda:0")
flush = _flush_buf(dev, 256 << 20)
torch.manual_seed(0)
a = torch.randn(N, N, device=dev, dtype=torch.float16)
b = torch.randn(N, N, device=dev, dtype=torch.float16)
c = torch.empty(N, N, device=dev, dtype=torch.float16)
ref = torch.matmul(a, b)

print("=== correctness ===")
out = matmul(a, b)
print("4096^3 bit-exact vs cuBLAS:", torch.equal(out, ref))
for (M, Nn, K) in [(1, 1, 1), (65, 65, 65), (1000, 1000, 1000), (4096, 4096, 4000),
                   (300, 5000, 70), (4096, 4096, 4096)]:
    torch.manual_seed(0)
    aa = torch.randn(M, K, device=dev, dtype=torch.float16)
    bb = torch.randn(K, Nn, device=dev, dtype=torch.float16)
    o = matmul(aa, bb)
    r = aa @ bb
    d = (o.float() - r.float()).abs().max().item()
    rel_ok = torch.allclose(o.float(), r.float(), atol=0.5, rtol=2e-3)
    print(f"  {M}x{Nn}x{K}: max_abs={d:.4f} allclose(fp16 tol)={rel_ok} bit_exact={torch.equal(o, r)}")

print("\n=== interleaved A/B (12 rounds x 60 iters) ===")
f_cb = lambda: torch.matmul(a, b, out=c)
f_tr = lambda: matmul(a, b)
time_fn(f_cb, iters=30, warmup=30, flush=flush)
time_fn(f_tr, iters=30, warmup=30, flush=flush)
cbs, trs, ratios = [], [], []
for _ in range(12):
    cb = time_fn(f_cb, iters=60, warmup=10, flush=flush)["median_ms"]
    tr = time_fn(f_tr, iters=60, warmup=10, flush=flush)["median_ms"]
    cbs.append(cb); trs.append(tr); ratios.append(cb / tr)
cb_med, tr_med = statistics.median(cbs), statistics.median(trs)
print(f"cuBLAS : {cb_med*1e3:.1f} us  ({2*N**3/(cb_med*1e-3)/1e12:.1f} TF/s)  sd {statistics.pstdev(cbs)*1e3:.1f} us")
print(f"Triton : {tr_med*1e3:.1f} us  ({2*N**3/(tr_med*1e-3)/1e12:.1f} TF/s)  sd {statistics.pstdev(trs)*1e3:.1f} us")
print(f"speedup: {statistics.median(ratios):.4f}x  (min {min(ratios):.4f}x, max {max(ratios):.4f}x)")
print(f"every round > 1.05x: {all(r > 1.05 for r in ratios)}")
print(f"every round > 1.04x: {all(r > 1.04 for r in ratios)}")

print("\n=== timing-method cross-check ===")
t_event = time_fn(f_tr, iters=100, warmup=25, flush=flush)["median_ms"]
t_graph = time_fn(f_tr, iters=100, warmup=25, flush=flush, use_cuda_graph=True)["median_ms"]
print(f"event-based median : {t_event*1e3:.1f} us")
print(f"CUDA-graph median  : {t_graph*1e3:.1f} us")
print(f"cuBLAS graph       : {time_fn(f_cb, iters=100, warmup=25, flush=flush, use_cuda_graph=True)['median_ms']*1e3:.1f} us")

peak_tf = 2 * 78 * 4096 * 1980 / 1e6  # dtype-agnostic TC peak at max clock
print(f"\nH20 FP16 dense peak @1980 MHz: ~{peak_tf:.0f} TF/s")
print(f"Triton fraction of peak     : {2*N**3/(tr_med*1e-3)/1e12/peak_tf*100:.1f}%")
print(f"cuBLAS fraction of peak     : {2*N**3/(cb_med*1e-3)/1e12/peak_tf*100:.1f}%")

json.dump({"cublas_us": cb_med*1e3, "triton_us": tr_med*1e3,
           "speedup_median": statistics.median(ratios),
           "speedup_min": min(ratios), "speedup_max": max(ratios),
           "bit_exact": torch.equal(out, ref),
           "graph_us": t_graph*1e3},
          open("final.json", "w"), indent=2)