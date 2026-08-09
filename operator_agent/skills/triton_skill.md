# Triton Kernel Development Skill Specification

> Version: 1.0 | Applies to: CoderAgent, DesignerAgent, SearchAgent
> This document encodes expert-level GPU kernel optimization knowledge as structured constraints.
> Agents MUST follow all rules below when generating, analyzing, or optimizing Triton code.

---

## 1. Absolute Prohibitions (MUST NOT Violate)

### 1.1 Output Format
- NO markdown code fences (```) in generated code
- ALLOW and ENCOURAGE concise kernel-level docstrings (triple-quoted) at the top of `@triton.jit` kernel or `def kernel` wrapper describing: algorithm name, input/output shapes, precision strategy, numerical stability trick (≤12 lines)
- ALLOW and ENCOURAGE **Chinese line comments** on key logic: ptr/stride initialization, tile load + mask, K/N loop pointer advancement, tl.dot Tensor Core calls, softmax/reduction online updates, causal mask shape, wrapper grid + strides mapping. Comments should let a reviewer understand the algorithm structure in ≤30 seconds
- NO explanation text before or after code (outside of docstring/comments)
- NO thinking process, reasoning steps, or self-correction notes in output
- NO "Here is the code", "Sure", "Okay" or any conversational filler
- Output MUST start with `import` and end with `return C` (or equivalent return statement of kernel wrapper)

### 1.2 Code Content
- NO `print()` statements
- NO `assert` statements except for shape compatibility checks
- NO hardcoded output values or magic numbers (except well-known constants)
- NO external library imports beyond `triton`, `triton.language`, `torch`, `math`
- NO `torch.nn.functional.*` calls inside kernel or wrapper
- NO fallback to PyTorch built-in operations (e.g. `torch.matmul`) as the primary computation
- NO dynamic shape assumptions: kernel MUST work for any valid input dimensions within the tiling constraints

### 1.3 Kernel Structure
- NO missing `def kernel(A, B)` wrapper function
- NO wrapper that takes non-tensor arguments as primary inputs
- NO `if __name__ == "__main__"` block
- NO class definitions

---

## 2. Required Patterns (MUST Follow)

### 2.1 Imports
```
import triton
import triton.language as tl
import torch
```

### 2.2 Kernel Function Signature
```python
@triton.jit
def <name>_kernel(
    ptrs..., M, N, K, strides...,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
```
- All tile size parameters MUST use `tl.constexpr` type annotation
- Pointer parameters come first, then dimensions, then strides, then constexpr block sizes

### 2.3 Wrapper Function Signature
```python
def kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
```
- MUST accept `torch.Tensor` inputs and return a single `torch.Tensor`
- MUST extract shape and stride from input tensors
- MUST compute grid using `triton.cdiv()`
- MUST use `torch.empty()` for output allocation
- MUST pass strides explicitly to kernel

### 2.4 Core Kernel Logic Pattern (HIGH-PERFORMANCE)
```python
# Step 1: L2 cache optimized block ordering
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m

# Step 2: Integer bound assumptions (critical for compiler optimization)
tl.assume(pid_m >= 0)
tl.assume(pid_n >= 0)
tl.assume(stride_am > 0)
tl.assume(stride_ak > 0)
tl.assume(stride_bn > 0)
tl.assume(stride_bk > 0)
tl.assume(stride_cm > 0)
tl.assume(stride_cn > 0)

# Step 3: Pointer initialization with modulo (enables bitwise AND)
offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
offs_k = tl.arange(0, BLOCK_K)
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

# Step 4: K-loop with pointer advancement + 3-arg tl.dot
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator)  # 3-arg fused accumulate
    a_ptrs += BLOCK_K * stride_ak  # pointer advancement
    b_ptrs += BLOCK_K * stride_bk

# Step 5: Cast and store with mask
c = accumulator.to(tl.float16)
offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

**CRITICAL - Performance Techniques (MUST ALL apply):**
1. **1D grid + GROUP_SIZE_M**: Never use 2D grid `(M, N)`. Always use 1D grouped ordering for L2 cache reuse.
2. **tl.assume hints**: Add `pid_m >= 0`, `pid_n >= 0`, and all stride `> 0` assumptions. This enables compiler to optimize address arithmetic.
3. **Modulo for M/N bounds**: Use `% M` and `% N` instead of `< M` and `< N`. When dimensions are power-of-2, compiler converts to bitwise AND.
4. **Pointer advancement**: Initialize pointers before K-loop, then `ptr += BLOCK_K * stride` inside loop. Never recompute full addresses each iteration.
5. **3-arg tl.dot**: Use `tl.dot(a, b, accumulator)` not `accumulator += tl.dot(a, b)`. This generates fused FMA instructions.
6. **K-mask optimization**: Use `offs_k < K - k * BLOCK_K` not `(k_start + offs_k) < K`. Simpler arithmetic.
7. **fp32 accumulator**: Always accumulate in fp32, cast to target dtype only at final store.

---

## 3. Optimization Priority System

> When optimizing a kernel, follow this priority order. Higher priority optimizations yield larger speedups.

### Priority 1: Algorithm-Level (>50% impact)
Apply these FIRST before any other optimization:

| Optimization | Description | When to Apply |
|---|---|---|
| Kernel Fusion | Combine multiple element-wise ops into one kernel | Always if multiple ops in pipeline |
| Shared Memory Tiling | Load data into shared memory for reuse | When data is accessed multiple times (e.g. matmul K dimension) |
| Memory Coalescing | Ensure consecutive threads access consecutive memory addresses | Always; verify with stride patterns |

### Priority 2: Hardware Utilization (20-50% impact)
Apply these AFTER Priority 1 optimizations are in place:

| Optimization | Description | When to Apply |
|---|---|---|
| Vectorized Load/Store | Use `tl.load` with larger element types (e.g. load float4 instead of float) | When memory access is the bottleneck |
| Loop Unrolling | Unroll the K-loop for better instruction-level parallelism | When loop body is small and predictable |
| Occupancy Tuning | Adjust BLOCK_M, BLOCK_N, BLOCK_K to maximize active warps | When register/spill pressure allows |
| Warp-Level Primitives | Use `tl.reduce`, `tl.broadcast`, `tl.trans` for cross-lane ops | For reductions (softmax, layernorm) |

### Priority 3: Fine-Grained Tuning (<20% impact)
Apply these LAST, only when Priority 1 and 2 are exhausted:

| Optimization | Description | When to Apply |
|---|---|---|
| Mixed Precision | Use fp16/bf16 for accumulation where numerical error is acceptable | When target precision allows |
| Double Buffering | Pre-load next tile while computing current tile | When shared memory is underutilized |
| Prefetching | Load data ahead of time using software pipelining | For memory-bound kernels |
| Instruction-Level Parallelism | Rearrange independent operations to fill pipeline | As final micro-optimization |

---

## 4. Tile Size Selection Guidelines

### 4.1 Default Tile Sizes by Kernel Type

| Kernel | BLOCK_M | BLOCK_N | BLOCK_K | Notes |
|---|---|---|---|---|
| Matmul (FP32) | 64 | 64 | 32 | Balanced for most GPUs |
| Matmul (FP16/BF16) | 128 | 128 | 32 | Larger tiles for tensor cores |
| Attention (QK^T) | 128 | 128 | 32 | Flash attention style |
| Softmax | 1024 | 1 | 1 | Full row reduction |
| LayerNorm | 1024 | 1 | 1 | Full row reduction |
| Element-wise | 1024 | 1 | N/A | Vectorized map |
| Conv2d | 128 | 128 | 16 | Depends on filter size |

### 4.2 Tile Size Constraints
- BLOCK_M * BLOCK_N * BLOCK_K should not exceed hardware shared memory limits
- BLOCK_M, BLOCK_N should be multiples of warp_size (typically 32)
- For small matrices (M, N, K < 256), reduce tile sizes to avoid out-of-bounds
- BLOCK_M must be <= M, BLOCK_N must be <= N, BLOCK_K must be <= K

---

## 5. Common Patterns by Kernel Type

### 5.1 Matrix Multiplication (Matmul / GEMM)
```
Required techniques (ALL mandatory for high performance):
1. tl.dot() for Tensor Core accumulation — 3-arg form: tl.dot(a, b, accumulator)
2. 1D grid + GROUP_SIZE_M grouped ordering for L2 cache optimization
3. Pointer advancement: init ptrs outside K-loop, increment inside
4. tl.assume() hints for pid and stride bounds
5. Modulo-based M/N bounds (% M, % N) instead of comparison
6. K-mask with remaining K (K - k * BLOCK_K) instead of absolute offset
7. fp32 accumulator with final cast to target dtype
Pattern: tiled matmul with BLOCK_M x BLOCK_K and BLOCK_K x BLOCK_N tiles
Key: Ensure correct stride calculation for both row-major and column-major layouts
Config: Include autotune with 8+ configs covering tile sizes 32-256, BLOCK_K 32-128
```

### 5.2 Attention (QK^T V)
```
Required: Flash attention pattern (online softmax)
Pattern: tile Q and K, compute attention scores, apply softmax online, accumulate V
Key: Use row-wise max and exp-sum for numerical stability
```

### 5.3 Softmax
```
Required: Online softmax (two-pass: max then sum-exp)
Pattern: Load full row, compute max, compute exp-sum, normalize
Key: Numerical stability via max subtraction
```

### 5.4 Layer Normalization
```
Required: Single-pass mean/variance computation
Pattern: Load full row, compute mean, compute variance, normalize
Key: Use warp reductions for efficiency
```

---

## 6. Correctness Checklist

Before finalizing any kernel, verify ALL items:

- [ ] `import triton` and `import triton.language as tl` are present
- [ ] `@triton.jit` decorator is applied to kernel function
- [ ] BLOCK_M, BLOCK_N, BLOCK_K use `tl.constexpr` type annotation
- [ ] `tl.program_id(axis)` is used for block indexing
- [ ] `tl.arange` is used for intra-block offsets
- [ ] `tl.load` / `tl.store` are used for memory access
- [ ] `tl.zeros` is used for accumulator initialization
- [ ] `def kernel(A, B)` wrapper exists and returns a tensor
- [ ] Grid is computed with `triton.cdiv()`
- [ ] Strides are passed explicitly (not relying on default layout)
- [ ] Output tensor is allocated with `torch.empty()`
- [ ] No markdown fences. Docstrings (≤12 lines) and Chinese line comments on key logic are NOT forbidden but are actually encouraged
- [ ] Code starts with `import` and ends with the wrapper return
- [ ] No hardcoded dimensions (works for any valid input size)
- [ ] (额外) 关键逻辑已写中文注释: ptr/stride 初始化、tile 加载+mask、K/N 循环指针推进、tl.dot、softmax/归约在线更新、causal mask、wrapper 的 grid+strides 传递
