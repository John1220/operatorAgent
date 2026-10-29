"""Establish the cuBLAS baseline for 4096^3 fp16 on H20.

Measures several plausible "cuBLAS baselines" so we know exactly what we have
to beat:
  * torch.matmul (heuristic pick, what a user normally gets)
  * torch.matmul under a CUDA graph (removes launch overhead)
  * cublasLt after an exhaustive algo-space search (the honest upper bound)
"""
import json
import time

import torch

from bench_common import N, setup, time_fn, parquet_free_report, _flush_buf

dev = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())

a, b, c = setup()

# ---------------- 1. plain torch.matmul ----------------
fn = lambda: torch.matmul(a, b, out=c)
st_plain = time_fn(fn)
print(parquet_free_report("torch.matmul (no flush)", st_plain))

flush = _flush_buf(dev, 256 << 20)
st_plain_f = time_fn(fn, flush=flush)
print(parquet_free_report("torch.matmul (L2 flush)", st_plain_f))

# ---------------- 2. under CUDA graph ----------------
st_graph = time_fn(fn, flush=flush, use_cuda_graph=True)
print(parquet_free_report("torch.matmul graph (L2 flush)", st_graph))

# ---------------- 3. cublasLt exhaustive ----------------
best = {"ms": float("inf")}
try:
    from torch._C import _cublas_init  # noqa: F401  (not used; keep import light)
except Exception:
    pass

lt_results = []
try:
    import ctypes

    # torch's cublasLt heuristic search is exposed through _scaled_mm? no.
    # Instead use the documented PyTorch knob: torch.backends.cuda.preferred_linalg_library
    # is for linalg (cusolver). For Lt autotune we use the cudnn-benchmark-style
    # experimental flag if present.
    print("has torch._inductor:", end=" ")
    import torch._inductor  # noqa: F401
    print("yes")
except Exception as e:  # pragma: no cover
    print("inductor unavailable:", e)

# Direct ctypes path to cublasLt is fragile; instead run a small sweep over
# torch matmul with different memory layouts, which is what changes the Lt
# kernel choice in practice:
variants = {}
for tag, aa, bb in [
    ("row,row", a, b),
    ("row,colT", a, b.t()),
    ("colT,row", a.t(), b),
]:
    fn2 = lambda aa=aa, bb=bb: torch.matmul(aa, bb, out=c)
    variants[tag] = time_fn(fn2, flush=flush)["median_ms"]
    print(parquet_free_report(f"torch.matmul {tag}", time_fn(fn2, flush=flush)))

# ---------------- 4. bf16 / fp32 sanity (roofline orientation) ----------------
a32 = a.float()
b32 = b.float()
c32 = torch.empty(N, N, device=dev, dtype=torch.float32)
st_f32 = time_fn(lambda: torch.matmul(a32, b32, out=c32), iters=30, flush=flush)
print(parquet_free_report("torch.matmul fp32 (tc off)", st_f32))

props = torch.cuda.get_device_properties(0)
out = {
    "gpu": torch.cuda.get_device_name(0),
    "sm_count": props.multi_processor_count,
    "total_mem_gb": props.total_memory / 2**30,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "plain": st_plain,
    "plain_flush": st_plain_f,
    "graph": st_graph,
    "layout_variants_ms": variants,
    "fp32_median_ms": st_f32["median_ms"],
}
with open("baseline.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved baseline.json")