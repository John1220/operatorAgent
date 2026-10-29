"""Quantify the mechanisms at 4096^3 directly (no guessing from models).

Three orthogonal ablations, each isolating one variable:
  A. A-padding test : does an out-of-range A tile read real DRAM?
     -> compare nominal vs a "safe" kernel variant that clamps instead of mask-
        loads. If the speedup shrinks, the quirk is worth exactly that much.
  B. B layout test  : is the transposed-B (wgmma tnspB) path the advantage?
     -> benchmark with B stored transposed, contiguous.
  C. Tile-count test: is it occupancy? Sweep the 64x64 family at K values that
     keep the tile grid identical, so kernel shape is constant.
"""
import torch
import triton
import triton.language as tl

from bench_common import time_fn, _flush_buf
from triton_ref import matmul as matmul_ref

dev = torch.device("cuda:0")
CFG = (64, 64, 64, 4, 4, 4)
flush = _flush_buf(dev, 256 << 20)
M = N = K = 4096

torch.manual_seed(0)
a = torch.randn(M, K, device=dev, dtype=torch.float16)
b = torch.randn(K, N, device=dev, dtype=torch.float16)
c = torch.empty(M, N, device=dev, dtype=torch.float16)
ref = torch.matmul(a, b)


@triton.jit
def matmul_safe(a_ptr, b_ptr, c_ptr, M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr):
    """Same schedule as the reference kernel, but every global access is masked
    (no out-of-range addresses, so no reliance on the end-of-tensor behaviour)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_ = tl.load(a_ptr + offs_am[:, None] * K + offs_k[None, :],
                     mask=mask_am[:, None] & mask_k[None, :], other=0.0)
        b_ = tl.load(b_ptr + offs_k[:, None] * N + offs_bn[None, :],
                     mask=mask_k[:, None] & mask_bn[None, :], other=0.0)
        acc = tl.dot(a_, b_, acc)
    c_ = acc.to(tl.float16)
    tl.store(c_ptr + offs_am[:, None] * N + offs_bn[None, :], c_,
             mask=mask_am[:, None] & mask_bn[None, :])


def run_safe():
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    matmul_safe[grid](a, b, c, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
                      GROUP_M=4, num_warps=4, num_stages=4)
    return c


print("=== A. masked-safe variant vs reference kernel (same schedule) ===")
run_safe()
torch.cuda.synchronize()
err_safe = (c.float() - ref.float()).abs().max().item()
t_safe = time_fn(run_safe, iters=100, warmup=25, flush=flush)["median_ms"]
t_ref = time_fn(lambda: matmul_ref(a, b, *CFG), iters=100, warmup=25, flush=flush)["median_ms"]
t_cb = time_fn(lambda: torch.matmul(a, b, out=c), iters=100, warmup=25, flush=flush)["median_ms"]
print(f"safe   : {t_safe*1e3:8.1f} us  (err {err_safe:.4f})  vs cuBLAS {t_cb/t_safe:.4f}x")
print(f"ref    : {t_ref*1e3:8.1f} us                    vs cuBLAS {t_cb/t_ref:.4f}x")
print(f"cuBLAS : {t_cb*1e3:8.1f} us")

print("\n=== B. B stored transposed & contiguous ===")
bt = b.t().contiguous()  # shape (N, K), row-major
c2 = torch.empty(M, N, device=dev, dtype=torch.float16)


@triton.jit
def matmul_nt(a_ptr, b_ptr, c_ptr, M, N, K,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BLOCK_M)
    npn = tl.cdiv(N, BLOCK_N)
    nig = GROUP_M * npn
    gid = pid // nig
    fpm = gid * GROUP_M
    gsm = min(npm - fpm, GROUP_M)
    pid_m = fpm + ((pid % nig) % gsm)
    pid_n = (pid % nig) // gsm
    om = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    on = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    ok = tl.arange(0, BLOCK_K)
    ap = a_ptr + om[:, None] * K + ok[None, :]
    bp = b_ptr + on[None, :] * K + ok[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        ap += BLOCK_K
        bp += BLOCK_K
    oc = tl.store(c_ptr + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * N
                  + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :],
                  acc.to(tl.float16))


def run_nt():
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    matmul_nt[grid](a, bt, c2, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
                    GROUP_M=4, num_warps=4, num_stages=4)
    return c2


run_nt(); torch.cuda.synchronize()
print("NT err:", (c2.float() - ref.float()).abs().max().item())
t_nt = time_fn(run_nt, iters=100, warmup=25, flush=flush)["median_ms"]
print(f"B^T contiguous : {t_nt*1e3:8.1f} us  vs cuBLAS {t_cb/t_nt:.4f}x")

print("\n=== C. tile-grid sweep at constant 4096^2 output ===")
for cfg in [(64, 64, 64, 4, 4, 4), (64, 64, 32, 4, 4, 4), (64, 64, 128, 4, 4, 3),
            (128, 128, 64, 8, 8, 3)]:
    t = time_fn(lambda cfg=cfg: matmul_ref(a, b, *cfg), iters=80, warmup=20,
                flush=flush)["median_ms"]
    bm = cfg[0]
    nblk = (M // bm) * (N // cfg[1])
    print(f"  {cfg[:3]} -> {nblk:5d} blocks: {t*1e3:8.1f} us ({t_cb/t:.4f}x)")