# Triton Kernel Development Skill Specification

> Version: 1.1 | Applies to: CoderAgent, DesignerAgent, SearchAgent
> This document encodes expert-level GPU kernel optimization knowledge as structured constraints.
> Agents MUST follow all rules below when generating, analyzing, or optimizing Triton code.
> ABI, layout, precision and reduction semantics come from the automatically injected
> operator-generation-spec and frozen calling_convention; examples here do not override them.

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
- NO missing Python `kernel` wrapper with the actual declared keyword inputs
- NO invented wrapper parameters; preserve scalar inputs when explicitly declared by the caller
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

### 2.2 GEMM Helper Example (only for a declared GEMM task)
```python
@triton.jit
def <name>_kernel(
    ptrs..., M, N, K, strides...,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
```
- All tile size parameters MUST use `tl.constexpr` type annotation
- Pointer parameters come first, then dimensions, then strides, then constexpr block sizes

### 2.3 Wrapper Interface
- Follow the shared generation spec and the current dynamic calling_convention
- Choose a launch grid for the actual operator; use `triton.cdiv()` when division into tiles is needed
- Output allocation must match the declared shape, dtype and device

### 2.4 GEMM Example (conditional, not a reduction or elementwise template)
This example assumes nonzero dimensions and positive strides. Apply such assumptions
only if the task establishes them; zero-stride or empty inputs need their declared handling.
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
c = accumulator.to(c_ptr.dtype.element_ty)
offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

**GEMM optimization options, subject to the task and measured evidence:**
1. Grouped ordering may improve L2 reuse; select launch dimensions for the workload.
2. Compiler assumptions require proven task preconditions.
3. Modulo-based loads require nonzero dimensions and correct masked stores.
4. Pointer advancement and the three-argument `tl.dot` are implementation options.
5. K-tail masks must preserve the full declared domain.
6. Precision and casts follow the shared generation spec and frozen semantics.

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
| Mixed Precision | Use only arithmetic modes explicitly permitted by the frozen contract | Never infer accumulation precision from input/output dtype |
| Double Buffering | Pre-load next tile while computing current tile | When shared memory is underutilized |
| Prefetching | Load data ahead of time using software pipelining | For memory-bound kernels |
| Instruction-Level Parallelism | Rearrange independent operations to fill pipeline | As final micro-optimization |

---

## 4. Tile Size Selection Guidelines

### 4.1 Initial Tile Candidates (not universal constraints)

| Kernel | BLOCK_M | BLOCK_N | BLOCK_K | Notes |
|---|---|---|---|---|
| Matmul (FP32) | 64 | 64 | 32 | Balanced for most GPUs |
| Matmul (FP16/BF16) | 128 | 128 | 32 | Larger tiles for tensor cores |
| Attention (QK^T) | 128 | 128 | 32 | Flash attention style |
| Softmax | task-dependent | task-dependent | N/A | Compare whole-row/chunked layouts |
| LayerNorm | task-dependent | task-dependent | N/A | Compare whole-row/chunked layouts |
| Element-wise | 1024 | 1 | N/A | Vectorized map |
| Conv2d | 128 | 128 | 16 | Depends on filter size |

### 4.2 Tile Size Constraints
- Estimate actual buffers using their dtypes, stages and the backend's resource limits
- GEMM tile/warp choices do not apply automatically to reductions or elementwise operators
- Padded tiles may exceed logical dimensions only with correct masks and declared empty-input behavior
- Preserve caller-owned tiling and search budgets

---

## 5. Common Patterns by Kernel Type

### 5.1 Matrix Multiplication (Matmul / GEMM)
```
Possible techniques when task preconditions and budgets permit:
1. tl.dot() for Tensor Core accumulation — 3-arg form: tl.dot(a, b, accumulator)
2. 1D grid + GROUP_SIZE_M grouped ordering for L2 cache optimization
3. Pointer advancement: init ptrs outside K-loop, increment inside
4. tl.assume() hints only for established pid and stride bounds
5. Modulo-based M/N bounds (% M, % N) instead of comparison
6. K-mask with remaining K (K - k * BLOCK_K) instead of absolute offset
7. accumulation and output casts according to the frozen precision contract
Pattern: tiled matmul with BLOCK_M x BLOCK_K and BLOCK_K x BLOCK_N tiles
Key: Ensure correct stride calculation for both row-major and column-major layouts
Config: Autotuning is allowed only when explicitly included in the task's search budget
```

### 5.2 Attention (QK^T V)
```
Required: Flash attention pattern (online softmax)
Pattern: tile Q and K, compute attention scores, apply softmax online, accumulate V
Key: Use row-wise max and exp-sum for numerical stability
```

### 5.3 Softmax
```
Semantics: Use the shared generation spec's softmax rules, not LayerNorm variance
Layout: Choose whole-row or chunked/online processing for the workload and resources
```

### 5.4 Layer Normalization
```
Semantics: Use the shared generation spec's LayerNorm rules, not softmax normalization
Layout: Choose a stable reduction algorithm and whole-row/chunked layout for the task
```

---

## 6. Correctness Checklist

Before finalizing any kernel, verify ALL items:

- [ ] `import triton` and `import triton.language as tl` are present
- [ ] `@triton.jit` decorates a differently named compute helper, not the Python wrapper
- [ ] BLOCK_M, BLOCK_N, BLOCK_K use `tl.constexpr` type annotation
- [ ] `tl.program_id(axis)` is used for block indexing
- [ ] `tl.arange` is used for intra-block offsets
- [ ] `tl.load` / `tl.store` are used for memory access
- [ ] The shared ABI, layout, precision and reduction requirements are satisfied
- [ ] The Python wrapper uses the actual dynamic calling_convention and returns its declared output
- [ ] Grid and allocation match this operator's declared inputs and outputs
- [ ] No markdown fences. Docstrings (≤12 lines) and Chinese line comments on key logic are NOT forbidden but are actually encouraged
- [ ] Code starts with `import` and ends with the wrapper return
- [ ] No hardcoded dimensions (works for any valid input size)
- [ ] (额外) 关键逻辑已写中文注释: ptr/stride 初始化、tile 加载+mask、K/N 循环指针推进、tl.dot、softmax/归约在线更新、causal mask、wrapper 的 grid+strides 传递
