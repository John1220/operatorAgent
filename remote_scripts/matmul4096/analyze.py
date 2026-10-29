"""Extract the concrete launch parameters Triton chose for the winning kernel.

Important: the winner uses num_warps=4, so with 64x64 tiles each thread holds only
8 accumulator floats -> the interesting question is what the *tile and warp-level
shape* became after Triton's layout selection, and whether WGMMA or
mma.sync is used.
"""
import json

import torch
import triton

from bench_common import setup
from triton_ref import matmul as matmul_ref

dev = torch.device("cuda:0")
a, b, c = setup()
avg = lambda *x: sum(x) / len(x)

warmup = triton.testing.do_bench(lambda: matmul_ref(a, b, 64, 64, 64, 4, 4, 4), rep=200)
q = triton.testing.do_bench(lambda: matmul_ref(a, b, 64, 64, 64, 4, 4, 4),
                            quantiles=[0.2, 0.5, 0.8], rep=200)
print("do_bench quantiles .2/.5/.8:", [f"{x:.4f}" for x in q])
ref = torch.matmul(a, b)
o = matmul_ref(a, b, 64, 64, 64, 4, 4, 4)
print("max abs err:", (o.float() - ref.float()).abs().max().item())
print("exact bitwise equal:", torch.equal(o, ref))
print("fraction of differing elements:",
      (o != ref).float().mean().item())

# print compiled kernel metadata
from triton_ref import matmul_kernel  # noqa: E402
best = matmul_kernel.warmup(
    a, b, c, 4096, 4096, 4096,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=4, num_warps=4, num_stages=4,
    grid=(1,),
)
print("\n=== compiled kernel ===")
print("name:", best.name)
print("num_warps:", best.metadata.num_warps, "num_stages:", best.metadata.num_stages)
print("shared:", best.metadata.shared, "bytes")
print("asm keys:", list(best.asm.keys()))
sass = best.asm.get("sass", "")
ptx = best.asm["ptx"]
for mac in ["wgmma.mma_async.sync.aligned.m64n64k16", "wgmma.mma_async.sync.aligned.m64n32k16",
            "wgmma.mma_async.sync.aligned.m64n16k16", "mma.sync.aligned.m16n8k16",
            "cp.async.bulk.tensor", "cp.async.ca", "cp.async.cg", "stmatrix", "ldmatrix"]:
    print(f"  {mac:<45s} ptx_count={ptx.count(mac)}")
print("SASS available:", bool(sass), len(sass))