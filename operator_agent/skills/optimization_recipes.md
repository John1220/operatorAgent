# Optimization Recipes — GPU 算子优化配方库

> **设计原则**: 每条配方是一个硬件无关的优化原则 (Layer 0)，
> 通过 `requires_caps` 声明需要的硬件能力 (Layer 1)，
> 然后按 DSL 提供具体代码模板 (Layer 2)。
> Planner 根据 `hardware_caps` + `dsl` 自动匹配和选择。

配方格式:

```
### <ID>: <名称>
- **applies_to**: matmul | softmax | layernorm | elementwise | reduce | all
- **applies_when**: 条件 (gflops_util < X, memory_bound, always, ...)
- **priority**: 1-10 (1=最优先)
- **requires_caps**: 硬件能力要求 (has_tensor_core, has_software_pipelining, ...)
- **expected_impact**: 预期性能提升
- **description**: 优化原则描述 (硬件/DSL 无关)
- **triton_template**: Triton DSL 代码模板 (若 dsl=triton 时注入)
- **cuda_c_template**: CUDA-C 代码模板 (若 dsl=cuda_c 时注入)
- **xpu_template**: 昆仑芯 XPU 代码模板 (若 dsl=xpu 时注入)
- **generic_template**: 通用伪代码模板 (其他 DSL 参考)
```

---

## Matmul / GEMM 优化配方

### R-MATMUL-01: L2 Cache Tiling Grouping
- **applies_to**: matmul, gemm, linear
- **applies_when**: always
- **priority**: 1
- **requires_caps**: l2_cache_size_kb > 0
- **expected_impact**: +10-20% on large matrices
- **description**:
  将 2D grid launch 改为 1D grid + GROUP_SIZE 分组调度，使同一组内连续的 block
  共享 A 矩阵的行，提高 L2 cache 命中率。当矩阵很大时，默认 2D 按列调度会导致
  L2 频繁替换。分组调度让 M 维相邻 block 时间上接近执行，复用 L2 中的 A tile。

- **triton_template**:
```python
# 1D grid launch + grouped ordering
grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

@triton.jit
def matmul_kernel(..., GROUP_SIZE_M: tl.constexpr = 8):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ... 其余逻辑与 2D grid 相同
```

- **cuda_c_template**:
```cuda
// 使用 cudaLaunchKernel 的 1D grid + block swizzle
const int GROUP_SIZE_M = 8;
int num_pid_m = (M + BLOCK_M - 1) / BLOCK_M;
int num_pid_n = (N + BLOCK_N - 1) / BLOCK_N;
int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
int group_id = blockIdx.x / num_pid_in_group;
int first_pid_m = group_id * GROUP_SIZE_M;
int group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M);
int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
```

- **xpu_template**:
```c
// 昆仑芯 XPU 类似 CUDA 但用 xpuLaunchKernel
// 分组调度逻辑与 CUDA 相同，注意 sub_group 大小可能不同
const int GROUP_SIZE_M = 8;
int num_pid_m = (M + BLOCK_M - 1) / BLOCK_M;
int num_pid_n = (N + BLOCK_N - 1) / BLOCK_N;
int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
int group_id = blockIdx.x / num_pid_in_group;
int pid_m = (group_id * GROUP_SIZE_M) +
           ((blockIdx.x % num_pid_in_group) % min(num_pid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M));
int pid_n = (blockIdx.x % num_pid_in_group) / min(num_pid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M);
```

- **generic_template**:
```
// 优化原则: 将 block 按 GROUP_SIZE 分组，同一组内 block 在 M 维连续，
// 以提高 L2 cache 对 A 矩阵的命中率。
pid = block_index_1d
num_pid_m = ceil(M / BLOCK_M)
num_pid_n = ceil(N / BLOCK_N)
group_id = pid / (GROUP_SIZE * num_pid_n)
pid_m = group_id * GROUP_SIZE + (pid % (GROUP_SIZE * num_pid_n)) % min(GROUP_SIZE, num_pid_m - group_id*GROUP_SIZE)
pid_n = (pid % (GROUP_SIZE * num_pid_n)) / min(GROUP_SIZE, num_pid_m - group_id*GROUP_SIZE)
```

---

### R-MATMUL-02: Autotune Configuration Search
- **applies_to**: matmul, gemm, linear
- **applies_when**: always
- **priority**: 2
- **requires_caps**: (无硬性要求，越多 SM 越能从 autotune 获益)
- **expected_impact**: +30-100% (critical for reaching hardware peak)
- **description**:
  不同矩阵形状和 GPU 对 tile 大小/warp 数/流水线阶段数有不同最优配置。
  提供多组 (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) 组合，
  运行时自动 benchmark 选择最优配置。配置数量建议 4-10 组覆盖大/中/小 tile。
  BLOCK_K 必须是 `tensor_core_m_k_n[2]` 的倍数 (通常是 16 的倍数) 以触发 Tensor Core。

- **triton_template**:
```python
@triton.autotune(
    configs=[
        # 大 tile + 深流水线: 适合大矩阵, 充分复用共享内存与寄存器 (A100 4096 方阵实测最优区)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    # BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M 现在是 tl.constexpr 参数
```
> 配置数量建议 8-12 组, 覆盖大/中/小 tile。BLOCK_K 必须是 `tensor_core_m_k_n[2]` 的倍数 (通常是 16 的倍数) 以触发 Tensor Core。

- **cuda_c_template**:
```cuda
// CUDA 中通常使用多个 kernel + 运行时选择
// 或者使用 CUTLASS 的 tile size dispatcher
// 简单做法: 定义多个模板实例，首次调用时 benchmark 选择
template<int BM, int BN, int BK, int NWARPS, int NSTAGES>
__global__ void matmul_kernel(...);

// 首次运行时遍历候选配置，记录最快的，后续直接调用最优版本
```

- **xpu_template**:
```c
// 昆仑芯 XPU 类似 CUDA，建议提供 3-5 组 tile 配置
// 注意 BLOCK_K 应对齐 XPU Tensor Core 的 K 维要求
// 参考: (128,128,32), (64,128,32), (128,64,32) 等组合
```

- **generic_template**:
```
// 优化原则: 枚举 tile 配置集合 C = {(BM,BN,BK,warp,stage)...}
// 首次运行时对每个 c ∈ C 执行 benchmark，选择最快的 c*
// 后续相同 shape 直接使用 c*
// BM/BN 应使每个 block 计算量足够大以摊销 launch 开销
// BK 应为 tensor core K 维度整数倍
```

---

### R-MATMUL-03: Tensor Core Input Casting
- **applies_to**: matmul, gemm, linear
- **applies_when**: precision != fp32 OR tf32 enabled
- **priority**: 2
- **requires_caps**: has_tensor_core=True
- **expected_impact**: +2-4x (触发 Tensor Core vs CUDA Core)
- **description**:
  矩阵乘法中，将输入 load 后 cast 为低精度 (fp16/bf16/tf32) 再做 `dot`，
  累加器保持 fp32。这是触发 Tensor Core 的必要条件。
  - fp32 输入 + tf32 compute: `tl.dot(..., input_precision="tf32")`
  - fp32 输入 + fp16 compute: load as fp16 (精度损失)，累加 fp32
  - bf16 输入: 直接用 bf16 dot
  BLOCK_K 必须是 Tensor Core K 维度的倍数 (NVIDIA: 16)。

  **`allow_tf32` 已废弃**，改用 `input_precision`。Triton 3.8 的 `tl.dot`
  签名是 `dot(input, other, acc=None, input_precision=None, allow_tf32=None, ...)`，
  docstring 明写 `allow_tf32: *Deprecated.*`，且两者**互斥**（同时传会 assert 失败）。
  合法取值：`"tf32"` / `"tf32x3"` / `"ieee"` / `"bf16x3"` / `"bf16x6"`，
  **默认就是 `"tf32"`**（不是 `"ieee"`）。

  ⚠️ **默认值是 `"tf32"` 这件事会直接决定 L2 过不过。** 声明 fp32 的任务，
  容差表给的是 atol/rtol=1e-4，而 tf32 只有 10 位尾数。H20 真机实测
  (1024³ fp32 matmul，只改这一个参数)：

  | 写法 | max_abs | matched_ratio | L2 (atol/rtol=1e-4) |
  |---|---|---|---|
  | `tl.dot(a, b)`（默认 tf32） | 1.22e-03 | 0.3163 | ❌ FAIL |
  | `tl.dot(a, b, input_precision="ieee")` | **0.0** | **1.000000** | ✅ PASS |

  所以**任务声明 fp32 且容差按 fp32 给时，必须显式写 `input_precision="ieee"`**，
  否则一个数值上完全正确的核会被判失败，而失败信息看不出是少写了一个参数。
  见 `docs/research/30-L2数值门真机验证.md` 第六节。

  ⚠️ **`tf32x3` 在 H20 上是负收益，在 H100 上是大幅正收益 —— 不要跨型号照搬。**
  `tf32x3` 用三次 tf32 dot 拼出接近 fp32 的精度，所以吞吐约为 tf32 的 1/3：

  | | fp32 (ieee) | tf32 | tf32x3 ≈ tf32/3 | tf32x3 相对 ieee |
  |---|---|---|---|---|
  | H100 | 67 TFLOPS | 989 | ≈330 | **4.9× 更快** ✅ |
  | H20 | 31.6 TFLOPS | 71.6 | ≈23.9 | **0.76× 更慢** ❌ |

  H20 的算力被砍到 138 TFLOPS(fp16)/71.6(tf32) 而带宽保留 3.68 TB/s，
  tf32 相对 fp32 只有 2.27× 而不是 H100 的 14.8×，三次拆分就把优势拆没了。
  **H20 上要高精度就老老实实用 `"ieee"`，别试 `tf32x3`。**
  (⚠️ 上表的 tf32x3 列是按峰值算力 1/3 推算，未真机实测；fp32/tf32 两列是
  `hardware_caps.py` 里 H20 的实测值与 H100 的 datasheet 值。)

  另注：`tl.dot` 的 tf32 是**截断而非舍入**，会引入系统性 bias。走 TMA 描述符时
  可用 `TensorDescriptor(..., round_f32_to_tf32=True)` 改成舍入（Triton 3.7.0+）。

- **triton_template**:
```python
# TF32 模式 (fp32 输入但用 TF32 Tensor Core) —— 也是 tl.dot 的默认行为
a = tl.load(a_ptrs, mask=mask_a, other=0.0)
b = tl.load(b_ptrs, mask=mask_b, other=0.0)
acc += tl.dot(a, b, input_precision="tf32")

# IEEE 模式 (任务声明 fp32 且容差按 fp32 给时**必须**显式写, 否则默认走 tf32 而 L2 挂)
acc += tl.dot(a, b, input_precision="ieee")

# FP16 输入 cast 模式 (更快但有精度损失)
a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float16)
b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float16)
acc += tl.dot(a, b)  # acc 是 fp32
```

- **cuda_c_template**:
```cuda
// CUDA 中使用 wmma::mma_sync 或 nvcuda::wmma
#include <mma.h>
using namespace nvcuda;
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::fill_fragment(c_frag, 0.0f);
wmma::load_matrix_sync(a_frag, a_ptr, lda);
wmma::load_matrix_sync(b_frag, b_ptr, ldb);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

- **xpu_template**:
```c
// 昆仑芯 XPU 使用 xpu::dot 或内置 tensor intrinsic
// 具体 API 参考 XPU 编程指南，核心是 load→cast→tensor dot→fp32 accumulate
```

- **generic_template**:
```
// 优化原则:
// 1. 输入 tile 加载后 cast 为 tensor core 原生精度
// 2. 矩阵乘使用 tensor 指令 (非逐元素乘加)
// 3. 累加器使用较高精度 (fp32)
// 4. K 维 tile 大小对齐 tensor core K 维度
```

---

### R-MATMUL-04: Software Pipelining (Double Buffering)
- **applies_to**: matmul, conv, any compute-bound kernel with inner loop
- **applies_when**: compute_bound AND gflops_util < 80%
- **priority**: 3
- **requires_caps**: has_software_pipelining=True
- **expected_impact**: +20-40% on compute-bound kernels
- **description**:
  利用 async copy 和多阶段流水线，将全局内存加载与计算重叠。
  Triton 通过 `num_stages` 参数控制；CUDA 通过 `cp.async` + shared memory 双缓冲。
  当 shared memory 足够时，更多 stage = 更好地隐藏内存延迟。
  注意: num_stages 过大会导致 shared memory 溢出，需在 autotune 中权衡。

- **triton_template**:
```python
# num_stages 在 @triton.autotune configs 中设置，或直接:
@triton.jit(num_stages=3)
def matmul_kernel(...):
    # 编译器自动插入 async copy 和 double buffering
    for k in range(0, K, BLOCK_K):
        # Triton 自动流水线化这些 load 和 dot
        a = tl.load(a_ptrs, mask=..., other=0.0)
        b = tl.load(b_ptrs, mask=..., other=0.0)
        acc += tl.dot(a, b)
```

- **cuda_c_template**:
```cuda
// 使用 cp.async 手动双缓冲
__shared__ half As[2][BLOCK_M][BLOCK_K];
__shared__ half Bs[2][BLOCK_K][BLOCK_N];
// 预加载第一个 tile
cp_async(As[0], ...); cp_async(Bs[0], ...);
cp_async_commit();
cp_async_wait<NSTAGES-1>();
__syncthreads();
for (int k = 0; k < K; k += BLOCK_K) {
    int cur = (k / BLOCK_K) % 2;
    int next = 1 - cur;
    // 预取下一个 tile (async)
    if (k + BLOCK_K < K) {
        cp_async(As[next], ...);
        cp_async(Bs[next], ...);
        cp_async_commit();
    }
    // 计算当前 tile
    compute(As[cur], Bs[cur]);
    // 等待预取完成
    cp_async_wait<NSTAGES-1>();
    __syncthreads();
}
```

- **xpu_template**:
```c
// 昆仑芯 XPU 支持 async copy，使用 __xpu_memcpy_async 或内置 pipeline pragma
// 双缓冲逻辑与 CUDA 相同，注意 shared memory bank conflict
```

- **generic_template**:
```
// 优化原则:
// 1. 分配 N 组 shared memory buffer (N = pipeline depth)
// 2. 计算 buffer i 时，async 预取 buffer i+1
// 3. 使用 sync 确保数据就绪
// 4. N 值通过 autotune 选择
```

---

### R-MATMUL-05: Stride Assumptions for Compiler Optimization
- **applies_to**: matmul, any kernel with stride parameters
- **applies_when**: always
- **priority**: 4
- **requires_caps**: (无特殊要求)
- **expected_impact**: +5-15% (帮助编译器消除边界检查)
- **description**:
  向编译器提供 stride 正性/对齐假设，让编译器可以更好地做向量化和地址计算优化。
  在 Triton 中用 `tl.assume`，在 CUDA 中用 `__builtin_assume` 或 `__assume`。
  这对非对齐输入尤其有效。

- **triton_template**:
```python
# Kernel 开头添加 stride 假设
tl.assume(stride_am > 0)
tl.assume(stride_bk > 0)
tl.assume(stride_bn > 0)
tl.assume(stride_cm > 0)
```

- **cuda_c_template**:
```cuda
// Kernel 入口处添加
__builtin_assume(stride_am > 0);
__builtin_assume(stride_bk > 0);
__builtin_assume(stride_bn > 0);
```

- **xpu_template**:
```c
// 昆仑芯 XPU 支持 __builtin_assume 或类似 pragma
__builtin_assume(stride_am > 0);
```

- **generic_template**:
```
// 在 kernel 开头对所有 stride 参数添加 >0 的假设
// 帮助编译器进行地址计算优化和向量化
```

---

### R-MATMUL-06: L2 Cache Eviction Policy
- **applies_to**: matmul, conv
- **applies_when**: compute_bound AND l2_cache_size_kb > 0
- **priority**: 5
- **requires_caps**: has_l2_cache_hint=True
- **expected_impact**: +5-10% on large matrices
- **description**:
  对于不会被重用的临时数据 (如 K 循环中用完即弃的 A/B tile)，使用 cache eviction
  提示让 L2 不缓存它们，把 L2 空间留给会被重用的 C 累加器或下一个 tile。
  Triton 用 `eviction_policy`，CUDA 用 `.cg` (cache global) 或 `.cs` (cache streaming)。

- **triton_template**:
```python
# A/B tile 是流数据 (使用一次不再需要)，用 evict_first 或 evict_last
a = tl.load(a_ptrs, mask=mask_a, other=0.0, eviction_policy='evict_last')
b = tl.load(b_ptrs, mask=mask_b, other=0.0, eviction_policy='evict_first')
# C 结果会被写回，使用默认 cache 策略
```

- **cuda_c_template**:
```cuda
// 使用 PTX cache modifier: .cg (cache at global level) .cs (streaming)
float val = __ldg(&a[row][col]);  // read-only cache
// 或者使用 asm volatile with .cs modifier for streaming data
```

- **xpu_template**:
```c
// 昆仑芯 XPU 若不支持 cache hint，则省略此优化
// 若支持，使用对应的 load with streaming hint API
```

- **generic_template**:
```
// 优化原则:
// 对只读一次的流数据使用 non-cached 或 streaming load
// 对会被重用的数据 (C 累加器) 保持正常缓存策略
// 若硬件不支持 cache hint，此优化自动跳过
```

---

### R-MATMUL-07: Register Tiling for K Dimension
- **applies_to**: matmul, gemm
- **applies_when**: gflops_util < 50% AND compute_bound
- **priority**: 6
- **requires_caps**: has_tensor_core=True
- **expected_impact**: +10-25%
- **description**:
  在 K 维循环内对小块进行展开 (unroll)，让编译器将更多数据放入寄存器，
  减少 shared memory 访问。通过 BLOCK_K=64 配合 #pragma unroll 实现。
  注意: 过度 unroll 会导致寄存器溢出到 local memory (spill)，反而降低性能。

  ⚠️ 本条的 `applies_when` / `requires_caps` 原先写的是
  `registers_per_thread >= 255` 与 `>= 128`，那是**死条件**：
  `registers_per_thread` 是硬件常量，`hardware_caps.py` 里 A100/H100/H20 全部
  硬编码为 255，不是运行期可观测量，所以条件恒真、等于没有条件。已改为按
  bound 类型与算力利用率触发（真正的判据是「compute-bound 但利用率低」，
  即算力没吃满，而不是寄存器数量）。
  另外这条是 pre-TMA 时代的微优化：走 R-MATMUL-13 的 tensor descriptor 之后，
  搬运与流水由硬件/TMA 接管，本条的收益会进一步缩小。

- **triton_template**:
```python
# Triton 编译器通常自动处理 K-loop unroll
# 可以通过更大的 BLOCK_K (64 vs 32) 隐式实现 register tiling
# 确保 BLOCK_K 是 16 的倍数
```

- **cuda_c_template**:
```cuda
// K 循环展开
#pragma unroll
for (int kk = 0; kk < BLOCK_K; kk += 16) {
    // 每次处理 16 个 K 元素 (一个 tensor core tile)
    mma_sync(c_frag, a_frag, b_frag, c_frag);
}
```

- **xpu_template**:
```c
// 类似 CUDA，使用 #pragma unroll 对 K 维内循环展开
```

- **generic_template**:
```
// 优化原则:
// 1. BLOCK_K 取适中大小 (32/64)，平衡寄存器使用和数据复用
// 2. 对内层 K 循环 unroll 以减少分支和指令开销
// 3. 监控寄存器使用量，避免 spill
```

---

### R-MATMUL-08: Pointer Advancement (Incremental Update)
- **applies_to**: matmul, gemm, linear
- **applies_when**: always
- **priority**: 1
- **requires_caps**: (无特殊要求)
- **expected_impact**: +5-15% (reduces address computation overhead)
- **description**:
  在 K 循环前初始化 A/B 的首块指针，循环内通过 `+= BLOCK_K * stride` 增量推进，
  避免每次迭代重新计算完整地址 `base + offset * stride`。
  这消除了 K 维指针的冗余乘法运算。
  同时 K 边界 mask 使用 `offs_k < K - k * BLOCK_K` 而非 `k_offs < K`，
  减少 mask 计算开销。

- **triton_template**:
```python
# Step 1: 初始化首块指针 (K=0 时)
offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
offs_k = tl.arange(0, BLOCK_K)
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

# Step 2: K 循环内增量推进
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    # 关键: 增量推进而非重新计算
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

- **cuda_c_template**:
```cuda
// CUDA 中使用指针增量
float* a_tile = &A[pid_m * BLOCK_M * stride_am];
float* b_tile = &B[pid_n * BLOCK_N * stride_bn];
for (int k = 0; k < K; k += BLOCK_K) {
    // load a_tile, b_tile
    // compute
    a_tile += BLOCK_K * stride_ak;
    b_tile += BLOCK_K * stride_bk;
}
```

- **generic_template**:
```
// 优化原则:
// 1. 循环外初始化首块 tile 指针
// 2. 循环内使用 ptr += stride_increment 增量推进
// 3. K 边界使用 K - k * BLOCK_K 计算当前剩余长度
// 4. 避免每轮重新计算 base + full_offset * stride
```

---

### R-MATMUL-09: 3-arg tl.dot Fused Accumulation
- **applies_to**: matmul, gemm, linear
- **applies_when**: always
- **priority**: 1
- **requires_caps**: has_tensor_core=True
- **expected_impact**: +10-20% (fused multiply-accumulate, reduced register pressure)
- **description**:
  使用 `tl.dot(a, b, accumulator)` 三参数形式直接将 dot 结果累加到 accumulator，
  而非 `accumulator += tl.dot(a, b)` 两参数形式。
  三参数形式在底层生成单次 FMA (fused multiply-accumulate) 指令，
  减少寄存器读写和中间结果存储。

- **triton_template**:
```python
# 正确: 3-arg fused accumulation
accumulator = tl.dot(a, b, accumulator)
# 错误: 2-arg + separate add
# accumulator += tl.dot(a, b)
```

- **cuda_c_template**:
```cuda
// WMMA 中默认就是 fused accumulate
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
// 等效于: c_frag += a_frag * b_frag
```

- **generic_template**:
```
// 优化原则:
// 使用 fused multiply-add (FMA) 将乘积直接累加到目标寄存器
// 避免中间结果写入寄存器再读回
```

---

### R-MATMUL-10: Modulo-Based Bounds Optimization
- **applies_to**: matmul, gemm, linear
- **applies_when**: always
- **priority**: 2
- **requires_caps**: (无特殊要求)
- **expected_impact**: +3-8% (enables bitwise AND for power-of-2 dimensions)
- **description**:
  使用 `% M` 和 `% N` 模运算替代 `< M` 和 `< N` 比较来处理 M/N 边界。
  当 M 或 N 是 2 的幂时，编译器可将 `% M` 优化为位与 `(value & (M-1))`，
  比整数比较更快。配合 `tl.assume(pid_m >= 0)` 确保编译器可以安全转换。
  K 维度边界继续使用 mask (`offs_k < K - k * BLOCK_K`)。

- **triton_template**:
```python
# M/N 边界用 modulo 优化
offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
# tl.assume 帮助编译器安全转换
tl.assume(pid_m >= 0)
tl.assume(pid_n >= 0)
# Store 时仍用 mask (因为 C 写入需要检测完整边界)
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

- **cuda_c_template**:
```cuda
// CUDA 中使用模运算或位与
int row_idx = (blockIdx.y * BLOCK_M + threadIdx.y) % M;
// 若 M 是 2 的幂: row_idx = (blockIdx.y * BLOCK_M + threadIdx.y) & (M - 1)
```

- **generic_template**:
```
// 优化原则:
// 1. M/N 维度使用 modulo 替代比较
// 2. 当维度是 2^n 时编译器可优化为位与
// 3. 需配合 pid >= 0 的 assume 确保转换安全
// 4. K 维度因循环递增特性继续用 mask 边界
```

---

### R-MATMUL-11: Optimized K-Dimension Mask
- **applies_to**: matmul, gemm, linear
- **applies_when**: always
- **priority**: 3
- **requires_caps**: (无特殊要求)
- **expected_impact**: +2-5% (simpler mask computation)
- **description**:
  K 循环中的边界 mask 使用 `offs_k < K - k * BLOCK_K` 替代 `(k_start + offs_k) < K`。
  前者编译器可优化为单次减法比较，后者需要加法 + 比较。
  当 K 不是 BLOCK_K 倍数时，最后一轮迭代的 mask 会更精确。

- **triton_template**:
```python
# 优化: 在循环中使用当前剩余 K 长度
for k in range(0, tl.cdiv(K, BLOCK_K)):
    remaining_k = K - k * BLOCK_K
    a = tl.load(a_ptrs, mask=offs_k[None, :] < remaining_k, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < remaining_k, other=0.0)
    # ...
```

- **generic_template**:
```
// 优化原则:
// mask 使用 remaining = total - offset 代替 absolute offset < total
// 减少整数运算复杂度
```

---

### R-MATMUL-12: Divisibility-Based Mask Elimination
- **applies_to**: matmul, gemm, linear
- **applies_when**: always
- **priority**: 5
- **requires_caps**: has_tensor_core=True
- **expected_impact**: +3-5% (A100 实测 0.78x → 0.81x cuBLAS)
- **description**:
  当且仅当 `K % BLOCK_K == 0` 时，去掉 K 循环内所有 `tl.load` 的 mask/other，
  并用 `tl.assume()` 把整除性告诉编译器。mask 会阻止向量化访存并增加谓词寄存器压力。
  K 不整除 BLOCK_K 时保留 R-MATMUL-11 的 mask 版本。

  ⚠️ 本条曾是一个**悬空引用**：`prompts/coder_generate_triton.md:37` 以「关键性能
  优化」名义要求 Coder 应用 R-MATMUL-12，但配方库里根本没有这一条（最大编号停在
  R-MATMUL-11）。Planner 是从本文件解析配方的，所以它**永远选不出 R-MATMUL-12**，
  Coder 只能看到 prompt 里那一行没有模板的描述。现已补齐。
  这正是本项目反复踩的那类缺陷：文档声称有、代码里没有。

  ⚠️ 走 R-MATMUL-13 的 tensor descriptor 之后本条自动失效 —— TMA 的越界由硬件按
  描述符 shape 自动截断，根本不写 mask。两条不要同时用。

- **triton_template**:
```python
# host 侧或 kernel 内先声明整除性
IS_DIVISIBLE: tl.constexpr = (K % BK == 0)

for k in range(0, tl.cdiv(K, BK)):
    if IS_DIVISIBLE:
        a = tl.load(a_ptrs)                      # 无 mask → 可向量化
        b = tl.load(b_ptrs)
    else:
        a = tl.load(a_ptrs, mask=k_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[None, :], other=0.0)
    acc = tl.dot(a, b, acc)
    a_ptrs += BK * sam
    b_ptrs += BK * sbk

# 在 wrapper 里给编译器提示 (Triton 会据此特化)
tl.assume(K % BK == 0)
```

---

### R-MATMUL-13: TMA Tensor Descriptor (Hopper+)
- **applies_to**: matmul, gemm, linear, attention
- **applies_when**: always
- **priority**: 1
- **requires_caps**: has_tma=True
- **expected_impact**: ⚠️ 未在 H20 上实测。官方 tutorial 08 的基准把 "Triton + TMA"
  与 "Triton" 并列对比；Triton 2026-03 meetup 记录里的说法是「现在要拿到好性能就
  需要 tensor descriptor API」。**按「H20 上比手工指针算术更可能拿到带宽」采纳，
  但期望值需真机标定后再写死。**
- **description**:
  用 `tl.make_tensor_descriptor` 建 device 侧描述符，让 TMA 硬件负责搬运。
  收益不只是「少写 mask」：R-MATMUL-01/05/08/10/11/12 这一整批 pre-TMA 微优化
  （modulo 边界、stride 假设、指针增量、K-mask 化简、整除去 mask）在描述符下
  **全部由硬件自动处理**，不需要手写。

  **H20 上这条比在 H100 上更重要**：H20 算力被砍到 fp16 138.2 TFLOPS 而带宽保留
  3.68 TB/s，ridge point 只有 ≈37.5 FLOP/B（H100 ≈295），几乎所有算子都是
  memory-bound —— 取数效率就是天花板。

  **硬约束（违反则编译或运行期直接失败，✅ 一手核实 Triton 3.8.0 源码）**：
  - `base` 必须 **16 字节对齐**（`core.py:2467`）
  - 除最后一维外，所有 stride × 元素大小必须是 16 字节的倍数；**最后一维必须
    contiguous**（`core.py:2469-2470`）
  - 只支持 **2–5 维**（`core.py:2478`）
  - `block_shape` **每一维都必须是 2 的幂**，且总 numel 不得超过
    `TRITON_MAX_TENSOR_NUMEL`（`_utils.py:63-74` 的 `validate_block_shape`，
    不满足直接 `raise ValueError`）
  - **device 侧描述符必须先注册 allocator**：`triton.set_allocator(...)`
    （`triton/__init__.py:24` 顶层导出）。漏了这一步 kernel 启动即失败。
  - `padding_option` 默认 `"zero"`

  ⚠️ 能力检测函数 `supports_tma()` **不是公开 API** —— 它是 tutorial 08 里的本地
  辅助函数（`tutorials/08-grouped-gemm.py:42`），判据是
  `torch.cuda.get_device_capability()[0] >= 9`。生成的核不能 `import` 它，
  要么自己复制这个判据，要么直接依赖本配方的 `requires_caps: has_tma=True`
  由 Planner 把关（推荐后者）。

  ⚠️ **block pointer 已在 Triton 3.8 彻底删除**：`tl.make_block_ptr` / `tl.advance`
  现在直接 `raise NotImplementedError("Block pointers have been removed in favor of
  the tensor descriptor API")`（`core.py:2434-2438`）。任何含 block pointer 的
  旧资料/旧配方照抄进来会直接崩，不要吸纳。

- **triton_template**:
```python
import torch, triton, triton.language as tl

def kernel(A, B):
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    # device 侧描述符必须配 allocator, 否则启动失败
    triton.set_allocator(lambda size, align, _: torch.empty(size, dtype=torch.int8, device="cuda"))
    BM, BN, BK = 128, 128, 64          # 每维必须是 2 的幂
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _mm_tma[grid](A, B, C, M, N, K, BM=BM, BN=BN, BK=BK)
    return C

@triton.jit
def _mm_tma(a_ptr, b_ptr, c_ptr, M, N, K,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # shape 用真实维度, 越界由 TMA 按 shape 自动截断 —— 所以不需要任何 mask
    a_d = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_d = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    c_d = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(tl.cdiv(K, BK)):
        a = a_d.load([pid_m * BM, ki * BK])
        b = b_d.load([pid_n * BN, ki * BK])
        acc = tl.dot(a, b.T, acc)          # 3-arg 形式, 见 R-MATMUL-09
    c_d.store([pid_m * BM, pid_n * BN], acc.to(c_d.dtype))
```

---

### R-MATMUL-14: Warp Specialization (Hopper autoWS)
- **applies_to**: matmul, gemm, attention
- **applies_when**: compute_bound AND gflops_util < 70%
- **priority**: 3
- **requires_caps**: has_warp_specialization=True
- **expected_impact**: ⚠️ **不要预期 1.5-2×**。可查到的数字全部来自别的卡：
  PyTorch 官方博客的 1.5–2× 是 **B200 上的 FlashAttention fwd**（且基于 Helion
  autotuning + ptxas 高级配置）；Tawa 论文（arXiv:2510.14719）在 **H100** 上是
  GEMM ≤1.1× cuBLAS、attention 1.2× over Triton。**H20 上没有任何公开数据**，
  而 warp specialization 解决的是「计算与搬运重叠」，H20 算力已非瓶颈，
  收益应低于 H100 —— 这是推断，需真机实测。
- **description**:
  `tl.range(..., warp_specialize=True)` 把 TMA 搬运与 MMA 分到不同 warp group，
  消除流水线气泡。Hopper 走的是独立的编译 pass
  `nvidia.passes.hopper.add_hopper_warpspec`（✅ `third_party/nvidia/backend/
  compiler.py`，`capability // 10 in [8, 9]` 分支），Blackwell 才走通用的
  `add_warp_specialize`。该 Hopper pass 于 2025-06-23（PR #7136）打通，
  即 **Triton v3.4.0（2025-07-30）起可用**。

  ⚠️ tutorial 09 里那句注释 "FIXME: This only works on Blackwell right now"
  **是过时的**，不要因为它就跳过 Hopper。

  ⚠️⚠️ **上游版本的门槛极窄，而且失败是静默的**
  （✅ 一手读 `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp`）：

  | 条件 | 不满足时的行为 |
  |---|---|
  | `num_warps` **必须恰好 == 4**（`if (numWarps != 4) return;`） | **静默不做 warp spec**，不报错、不警告 |
  | 该 loop 的 `num_stages > 1` | 静默跳过（loop 根本不被收集） |
  | 函数内**任何** `scf.if` 都不能带 else 分支 | **静默整体放弃**（注释写明是 FIXME） |
  | 任务分区失败 | `signalPassFailure()` → **硬编译错误**（不是降级） |

  「静默不做」是最坏的一种失败：代码照跑、结果照对、性能没变，而你会以为已经
  用上了 warp specialization。**必须靠真机计时对比来确认它生效了**，没有别的信号。

  **前置条件**：Hopper + warp_specialize **必须用 device 侧描述符**（tutorial 06：
  `if supports_host_descriptor() and not (is_hopper() and warp_specialize)`），
  所以 R-MATMUL-13 的 `triton.set_allocator` 是本条的前提。

  ⚠️ **与 R-MATMUL-02 的 autotune 配置表直接冲突**：现有 12 组 config 里有 8 组是
  `num_warps=8`，在 Hopper 上会让 warp spec 静默失效。两条一起用时必须把 config
  收敛到 `num_warps=4`。`benchmarks/triton_autotune_matmul.py` 是同一份表的拷贝，
  要同步改。

  attention 场景 tutorial 06 还额外过滤掉 cc9.0 上 `BLOCK_M*BLOCK_N < 128*128
  且 num_warps == 8` 的 config。

- **triton_template**:
```python
# num_warps 必须是 4, 否则整条优化静默失效
for ki in tl.range(0, tl.cdiv(K, BK), num_stages=NS, warp_specialize=True):
    a = a_d.load([pid_m * BM, ki * BK])
    b = b_d.load([pid_n * BN, ki * BK])
    acc = tl.dot(a, b.T, acc)

# launch: _mm[grid](..., num_warps=4, num_stages=NS)
# 注意 kernel 内不能出现带 else 的 if, 否则同样静默放弃
```

---

## 通用 (算子无关) 优化配方

### R-UNIV-01: Boundary Mask Handling
- **applies_to**: all
- **applies_when**: always (除非矩阵维度保证是 block_size 的整数倍)
- **priority**: 1
- **requires_caps**: (无)
- **expected_impact**: 正确性保证
- **description**:
  所有 load/store 必须使用 mask 处理边界，确保任意输入尺寸都能正确运行。
  不允许假设 M/N/K 是 block 大小的整数倍。mask 逻辑:
  对于 2D tile，mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)。

- **triton_template**:
```python
# 边界 mask 原则 (配合 R-MATMUL-08/10/11 使用):
# M/N 边界: 使用 modulo 优化 (见 R-MATMUL-10)
# offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
# offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
# K 边界: 使用剩余长度 mask (见 R-MATMUL-11)
# mask = offs_k < K - k * BLOCK_K
# Store 时使用标准 mask
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, result, mask=c_mask)
# Load 时: boundary 外数据必须设为 0 (other=0.0)
# a = tl.load(a_ptrs, mask=k_mask, other=0.0)
```

- **cuda_c_template**:
```cuda
int row = blockIdx.y * BLOCK_M + threadIdx.y;
int col = blockIdx.x * BLOCK_N + threadIdx.x;
if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row*lda + k] * B[k*ldb + col];
    }
    C[row*ldc + col] = sum;
}
```

- **xpu_template**:
```c
// 与 CUDA 相同: if (row < M && col < N) 边界检查
```

- **generic_template**:
```
// 所有 load/store 操作必须有边界检查
// mask = (row < M) AND (col < N)
// other=0 用于边界外的 load
```

---

### R-UNIV-02: Contiguous Memory Assumption
- **applies_to**: all
- **applies_when**: input is contiguous (row-major with standard strides)
- **priority**: 2
- **requires_caps**: vector_width_bytes >= 16
- **expected_impact**: +10-30% (vectorized load/store)
- **description**:
  当输入张量是 contiguous 时 (stride[-1] == 1)，编译器可以生成向量化加载指令
  (float4, int4, etc.)，单次加载 16 字节，减少内存指令数。
  Triton 中通过正确传递 stride 并让编译器推断 contiguous 模式实现。
  wrapper 函数中必须正确传递 stride_am/stride_bk/stride_bn/stride_cm/stride_cn。

- **triton_template**:
```python
# Wrapper 函数中传递正确的 strides
def matmul(A, B, C, M, N, K, ...):
    grid = ...
    # contiguous 2D: stride_am=K, stride_ak=1 for row-major A
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)
    matmul_kernel[grid](A, B, C, M, N, K, stride_am, stride_ak, ...)
```

- **cuda_c_template**:
```cuda
// 使用 float4 向量化加载 (当 alignment 允许时)
if ((reinterpret_cast<uintptr_t>(ptr) % 16) == 0) {
    float4 val = *reinterpret_cast<const float4*>(ptr);
} else {
    // fallback to scalar load
}
```

- **xpu_template**:
```c
// 昆仑芯 XPU 支持向量化加载，使用 xpu::vector<T,N> 或 float4 类型
```

- **generic_template**:
```
// 优化原则:
// 1. wrapper 中正确传递所有 stride 参数
// 2. 不要硬编码 stride=1，从 tensor.stride() 获取
// 3. 保证数据按行优先连续存储时能触发向量化
```

---

### R-UNIV-03: Output dtype Casting
- **applies_to**: all
- **applies_when**: precision == fp32
- **priority**: 3
- **requires_caps**: (无)
- **expected_impact**: 正确性保证
- **description**:
  累加器使用 fp32 保证精度，最终 store 前 cast 为输出 dtype。
  不要在整个计算中使用低精度累加，否则会有严重的精度累积误差。

- **triton_template**:
```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(...).to(tl.float16)
    b = tl.load(...).to(tl.float16)
    acc += tl.dot(a, b)  # fp32 accumulate
# Store 时 cast 回输出类型
tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=...)
```

- **cuda_c_template**:
```cuda
float acc = 0.0f;
for (int k = 0; k < K; ++k) {
    acc += __half2float(a[k]) * __half2float(b[k]);
}
c[row*ldc+col] = acc;  // 自动 cast 到输出类型
```

- **xpu_template**:
```c
// 同 CUDA: 累加器用 fp32，store 时 cast 到目标类型
```

- **generic_template**:
```
// 1. 累加器始终使用 fp32
// 2. 输入可以 cast 到低精度触发 tensor core
// 3. 输出时 cast 到目标 dtype
```

---

### R-UNIV-04: Persistent Kernel + 按占用率定 grid
- **applies_to**: all
- **applies_when**: always
- **priority**: 2
- **requires_caps**: sm_count > 0
- **expected_impact**: ⚠️ 未在 H20 上实测。官方 tutorial 02 对 fused softmax 给的是
  「persistent 版本比 naive 快若干倍」的量级，但那是 **行归约类算子**（每行一个
  program，行数远大于 SM 数）的场景。**matmul 这类每个 tile 工作量已经很大的算子
  收益有限**，不要无差别套用。
- **description**:
  普通写法是 `grid = (cdiv(M,BM) * cdiv(N,BN),)` —— 有多少 tile 就发多少 program，
  数量可能是 SM 数的几十上百倍，调度与 prologue 开销按 program 数摊。

  Persistent kernel 反过来：**只发「刚好填满 GPU」个 program**，每个 program 用
  一个内层循环依次处理多个 tile。program 数从「tile 数」变成「SM 数 × 占用率」，
  prologue/epilogue 只跑一次。

  关键是 grid 大小不能拍脑袋，要按实际寄存器与 shared memory 用量算占用率
  （✅ 一手核实 tutorial `02-fused-softmax.py` 的 `calculate_programs` 路径）:

  ```python
  kernel._init_handles()          # 必须先 init, 否则 n_regs / metadata 还没填
  n_regs    = kernel.n_regs
  size_smem = kernel.metadata.shared
  occupancy = min(NUM_REGS // (n_regs * WARP_SIZE * num_warps),
                  SIZE_SMEM  // size_smem)
  num_programs = min(NUM_SM * occupancy, n_rows)
  ```

  H20 上的常量（✅ `hardware_caps.py` 的 h20 条目）: `NUM_SM=78`、
  `NUM_REGS=65536`（`registers_per_sm`）、`WARP_SIZE=32`、
  `SIZE_SMEM=228 KB`（`shared_mem_per_sm_kb`）。

  ⚠️ `_init_handles()` / `kernel.n_regs` / `kernel.metadata.shared` 都是
  **Triton 运行时私有 API**（下划线开头），不是稳定契约，版本升级可能改名。
  生成的核里用要包 try/except 并给保守回退值，不能让它成为唯一的 grid 来源。

  **循环级流水线**：persistent 的内层循环要把 `num_stages` 写在 `tl.range` 上而
  不是只写在 launch 参数上，否则跨 tile 的取数不会与计算重叠:

  ```python
  for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, num_stages=NS, flatten=True):
      ...
  ```

  `flatten=True` 会把嵌套循环压平成一个流水，是让跨 tile 边界也能重叠的开关。

- **triton_template**:
```python
NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count  # H20 = 78

def kernel(A, B, NUM_SMS=78):
    ...
    grid = (min(NUM_SMS, num_tiles),)      # ← 不再是 num_tiles 本身
    _mm_persistent[grid](A, B, C, M, N, K, NUM_SMS=NUM_SMS, BM=BM, BN=BN, BK=BK)

@triton.jit
def _mm_persistent(a_ptr, b_ptr, c_ptr, M, N, K,
                   NUM_SMS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    num_pid_m, num_pid_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    start_pid = tl.program_id(0)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # 每个 program 认领 start_pid, start_pid+NUM_SMS, start_pid+2*NUM_SMS, ...
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, num_stages=3, flatten=True):
        pid_m, pid_n = tile_id // num_pid_n, tile_id % num_pid_n
        ...  # load / dot
```

---

### R-UNIV-05: Programmatic Dependent Launch (PDL)
- **applies_to**: all
- **applies_when**: always
- **priority**: 4
- **requires_caps**: has_pdl=True
- **expected_impact**: ⚠️ **本条在 L3 基准里量不出来，甚至会把测量带偏**。
  PDL 消除的是**相邻两个核之间**的启动空档，而我们每次只测一个核的一次调用。
  期望收益只在真实推理流水线（连续多个算子）里成立。**不要为了跑分而用。**
- **description**:
  PDL 让第 N+1 个核在第 N 个核**还没跑完**时就开始启动与执行 prologue，把启动
  延迟藏进上一个核的尾巴里。Hopper（cc ≥ 9.0）起可用。

  三个动作缺一不可:

  | 位置 | 写法 | 漏掉的后果 |
  |---|---|---|
  | kernel 装饰器 | `@triton.jit(launch_pdl=True)` | 依赖启动根本不生效 |
  | 启动点 | `kernel[grid](..., launch_pdl=True)` | 同上（两处都要写） |
  | 读上游数据之前 | `tl.extra.cuda.gdc_wait()` | **静默读到未写完的数据 → 结果错** |
  | 本核数据写完之后 | `tl.extra.cuda.gdc_launch_dependents()` | 后继核无法提前启动，白开 PDL |

  ⚠️ `gdc_wait()` 的位置是**正确性问题**不是性能问题：PDL 让后继核提前跑起来了，
  如果不在读输入前 wait，就会读到前一个核还没写完的 buffer。这类 bug 在小 shape
  上常常碰巧不出现（前核跑得快），换个大 shape 才炸，是典型的隐蔽错误。

  ⚠️ 能力检测函数 `supports_pdl()` 与 `supports_tma()` 一样**不是公开 API**，是
  tutorial 里的本地辅助函数，判据 `torch.cuda.get_device_capability()[0] >= 9`。
  生成的核不要 import 它，交给本配方的 `requires_caps: has_pdl=True` 由 Planner 把关。

  ⚠️⚠️ **与 L3 测量的冲突**：runner 的计时循环是背靠背重复调用同一个核。开了 PDL
  之后相邻两次重复会**互相重叠**，测出来的是吞吐而不是单次延迟，数字会明显偏快，
  而生产里这个核只被调用一次。若候选核使用了 `launch_pdl=True`，L3 的数字不可与
  未使用 PDL 的基线直接比较。

- **triton_template**:
```python
@triton.jit(launch_pdl=True)                       # ← 装饰器上要写
def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.extra.cuda.gdc_wait()                       # ← 读上游数据之前
    x = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(y_ptr + offs, x * 2, mask=offs < n)
    tl.extra.cuda.gdc_launch_dependents()          # ← 写完之后再放行后继

# 启动点同样要写, 只写一处不生效:
kernel[grid](x, y, n, BLOCK=1024, launch_pdl=True)
```

---

## Softmax 优化配方

### R-SOFTMAX-01: Online Softmax (Normalization Split)
- **applies_to**: softmax
- **applies_when**: always
- **priority**: 1
- **requires_caps**: (无)
- **expected_impact**: 正确性 + 避免数值溢出
- **description**:
  使用 online softmax 算法 (Milakov & Gimelshein 2018): 两遍扫描或单遍 online 算法
  计算 max 和 sum，避免 fp16/bf16 溢出。行方向 (dim=-1) softmax 中每个 block 处理一行或多行。

- **triton_template**:
```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_idx * n_cols + col_offs
    mask = col_offs < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_ptrs = output_ptr + row_idx * n_cols + col_offs
    tl.store(output_ptrs, softmax_output, mask=mask)
```

- **generic_template**:
```
// online softmax:
// 1. 加载整行 (或分块加载)
// 2. max_val = max(row)
// 3. exp_row = exp(row - max_val)
// 4. sum_exp = sum(exp_row)
// 5. softmax = exp_row / sum_exp
```

---

### R-SOFTMAX-02: Warp/Block Reduction
- **applies_to**: softmax, layernorm, reduce
- **applies_when**: sequence_length > block_size (需要 warp/block 间 reduce)
- **priority**: 2
- **requires_caps**: has_block_reduce=True
- **expected_impact**: +30-50% for long sequences
- **description**:
  当序列长度超过单个 block 大小时，使用多 block 协作 + shared memory 归约。
  Triton 中使用 `tl.reduce` 或显式 warp shuffle。CUDA/XPU 中使用 warp shuffle + shared memory。

- **triton_template**:
```python
# Triton 2.1+ 有 tl.reduce 原语
# 对于超长序列，可以使用 atomic max/sum 做跨 block 归约
row_max = tl.max(row, axis=0)
# 如果需要跨 block: 用 atomic_max 写全局 max
```

- **cuda_c_template**:
```cuda
// Warp reduce using shuffle
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
// Block reduce via shared memory
__shared__ float smem[32];
float val = warp_reduce_max(thread_val);
if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = val;
__syncthreads();
```

- **generic_template**:
```
// 1. 每个 block 计算局部 max/sum
// 2. Warp shuffle 做 warp 内 reduce
// 3. Shared memory 做 warp 间 reduce
// 4. 若跨 block，使用 atomic 或多 kernel
```

---

### R-SOFTMAX-03: Fused Scale/Mask/Softmax (Flash Attention 风格)
- **applies_to**: softmax (in attention context)
- **applies_when**: attention 计算中 Q@K^T 后立即 softmax
- **priority**: 3
- **requires_caps**: shared_mem_per_sm_kb >= 64
- **expected_impact**: +2-4x (避免 materialize 到 HBM)
- **description**:
  在 attention 中，将 scale、causal mask、softmax 融合到一个 kernel 中，
  避免将中间结果 (QK^T 矩阵) 写回 HBM。这是 Flash Attention 的核心思想。

- **triton_template**:
```python
# 融合: S = Q @ K^T / sqrt(d) + causal_mask
#      P = softmax(S)
# 在 matmul kernel 内部直接做 softmax，不存 S 到全局内存
```

- **generic_template**:
```
// 融合 scale + mask + softmax + matmul
// QK 计算后在 shared memory/register 中直接 softmax，再乘以 V
// 不将 attention matrix 写回 HBM
```

---

## LayerNorm 优化配方

### R-LN-01: Fused Compute (Mean/Var/Norm in one pass)
- **applies_to**: layernorm, rmsnorm
- **applies_when**: always
- **priority**: 1
- **requires_caps**: (无)
- **expected_impact**: +20% vs naive two-pass
- **description**:
  使用 Welford online algorithm 单遍计算 mean 和 variance，减少 HBM 读取次数。
  对于 fp16/bf16 输入，加载时 cast 到 fp32 计算。

- **triton_template**:
```python
# 单遍 online 算法计算 mean/var (或两遍更稳定)
# Triton 实现通常两遍: 第一遍 mean/var, 第二遍 normalize+scale+bias
```

- **generic_template**:
```
// Pass 1: load row, compute mean, compute variance
// Pass 2: normalize (x - mean) / sqrt(var + eps), apply gamma/beta
// 对于 RMSNorm 只需要 pass 1 计算 rms
```

---

### R-LN-02: Vectorized Load/Store with float4
- **applies_to**: layernorm, softmax, elementwise
- **applies_when**: hidden_size is multiple of 4/8
- **priority**: 2
- **requires_caps**: vector_width_bytes >= 16
- **expected_impact**: +20-30%
- **description**:
  当 hidden_dim 是 4/8 的倍数时，使用 float4/half8 向量化加载，减少内存事务数。

- **triton_template**:
```python
# Triton 编译器在 contiguous + 对齐时自动向量化
# 确保 BLOCK_SIZE 是 4 的倍数
BLOCK_SIZE: tl.constexpr = 1024  # 必须是 2 的幂且 >= 16
```

- **generic_template**:
```
// 确保 hidden_dim 对齐 vector_width_bytes
// 使用 vector type (float4, half8) 进行 load/store
```

---

### R-LN-03: Shared Memory for Reduction
- **applies_to**: layernorm, rmsnorm, softmax
- **applies_when**: hidden_size > 256 (N > 256)
- **priority**: 1
- **requires_caps**: has_shared_memory=True
- **expected_impact**: +30-50% (减少全局内存读取, 降低寄存器压力)
- **description**:
  当 N 较大时，将输入数据加载到 shared memory 后进行归约计算，
  避免 N 维直接全部加载到寄存器导致寄存器溢出 (register spilling)。
  使用分块策略：一次处理 BLOCK_SIZE 列 (如 256)，在 shared memory 中归约。
  对于 N=768 的场景，分 3 块处理 (256+256+256)。

- **triton_template**:
```python
@triton.jit
def layernorm_kernel(X_ptr, Y_ptr, W_ptr, B_ptr, M, N, stride_x, stride_y, eps,
                     BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * stride_x
    # 分块加载 + 归约: 逐块累加 sum 和 sum_sq
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sum_val += x
        sum_sq += x * x
    total_sum = tl.sum(sum_val, axis=0)
    total_sq = tl.sum(sum_sq, axis=0)
    mean = total_sum / N
    var = total_sq / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    # 第二遍: 重新加载数据, normalize + scale + bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std * w + b
        tl.store(Y_ptr + row * stride_y + cols, y.to(tl.float16), mask=mask)
```

- **generic_template**:
```
// 优化原则:
// 1. 分块加载 (BLOCK_SIZE=256) 避免寄存器溢出
// 2. 两遍方法: 第一遍计算 mean/var, 第二遍 normalize
// 3. 使用 shared memory 缓存中间结果
// 4. 对于 N <= 256, 可以单遍在寄存器中完成
```

---

### R-LN-04: Warp-Level Reduction
- **applies_to**: layernorm, rmsnorm, softmax
- **applies_when**: always
- **priority**: 2
- **requires_caps**: has_warp_primitives=True
- **expected_impact**: +10-20% (更快的归约)
- **description**:
  使用 warp-level 原语 (tl.sum, tl.multiple_of) 替代逐元素累加，
  利用硬件 warp shuffle 加速归约操作。
  Triton 编译器自动将 tl.sum 优化为 warp shuffle + shared memory 归约。

- **triton_template**:
```python
# tl.sum 内部使用 warp shuffle, 无需显式写 warp 归约
# 关键: 使用 axis=0 在列维度上归约
mean = tl.sum(x, axis=1) / N  # 行级归约
# 如果 BLOCK_SIZE > 1024, 需要分步归约
# 对于 N <= 1024, tl.sum 单次即可完成
```

- **generic_template**:
```
// 优化原则:
// 1. 使用 tl.sum(axis=...) 而非手动 for 循环累加
// 2. 对于大 N (>1024), 分步归约: 先分块 sum, 再总 sum
// 3. 利用 warp 原语避免串行累加
```

---

## Element-wise 优化配方

### R-ELEM-01: Element-wise Fusion
- **applies_to**: elementwise (gelu, silu, add, mul, relu, etc.)
- **applies_when**: multiple consecutive elementwise ops
- **priority**: 1
- **requires_caps**: (无)
- **expected_impact**: +2x per fused op (减少 HBM 读写)
- **description**:
  将多个连续的 element-wise 操作融合到一个 kernel 中，避免中间结果写回 HBM。
  例如: bias + gelu + dropout 融合为一个 kernel。

- **triton_template**:
```python
@triton.jit
def fused_bias_gelu_dropout_kernel(X, Bias, Out, N, BLOCK_SIZE: tl.constexpr, p: float, seed: int):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    b = tl.load(Bias + (offs % Bias.shape[0]), mask=mask)
    x = x + b
    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    x = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # Dropout
    random = tl.rand(seed, offs)
    x = tl.where(random > p, x / (1.0 - p), 0.0)
    tl.store(Out + offs, x, mask=mask)
```

- **generic_template**:
```
// 将所有 point-wise 操作合并到一个 kernel
// chain: op3(op2(op1(x))) 在一次 load/store 中完成
// 不产生中间 tensor
```

---

## Reduction 优化配方

### R-REDUCE-01: Tree Reduction with Warp Shuffle
- **applies_to**: reduce (sum, max, min), layernorm, softmax
- **applies_when**: reduction dim > warp_size
- **priority**: 1
- **requires_caps**: has_warp_shuffle=True
- **expected_impact**: +2-5x vs atomic reduction
- **description**:
  使用 warp shuffle 做 tree reduction，避免 shared memory bank conflict。
  最后一个 warp 汇总所有 warp 的部分结果。

- **triton_template**:
```python
# Triton 中直接用 tl.sum/tl.max 即可，编译器自动生成最优 reduction
row_sum = tl.sum(numerator, axis=0)
```

- **cuda_c_template**:
```cuda
template<int BLOCK_SIZE>
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    // First warp reduces warp results
    if (wid == 0) {
        val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

- **xpu_template**:
```c
// 昆仑芯 XPU 使用 __xpu_shfl_down_sync 或内置 reduce 原语
// 逻辑与 CUDA 相同
```

- **generic_template**:
```
// 1. 每个线程计算局部 partial result
// 2. Warp shuffle 做 warp 内 tree reduce
// 3. Shared memory 做 warp 间 reduce
// 4. 第一个线程输出最终结果
```

---

## 新硬件适配指南

为新 GPU (如昆仑芯) 添加支持时，只需以下步骤:

1. **在 `hardware_caps.py` 中添加硬件能力描述**:
```python
_KNOWN_HARDWARE["kunlunxin_r3"] = HardwareCaps(
    vendor="kunlunxin", device_name="Kunlunxin R3",
    fp32_tflops=..., fp16_tflops=...,
    has_tensor_core=True,
    global_mem_bandwidth_gbs=...,
    sub_group_size=...,  # 通常 32
    has_software_pipelining=True/False,
    has_l2_cache_hint=True/False,
    has_warp_shuffle=True/False,
    supported_dsls={"xpu", "xpu-c", "python"},
    # ... 其他字段
)
```

2. **在配方文件中添加对应的 DSL 模板** (可选，generic_template 作为 fallback):
   - 为关键配方 (R-MATMUL-01/02/03, R-UNIV-01, R-REDUCE-01) 添加 `xpu_template`
   - 其他配方可先用 generic_template，后续逐步优化

3. **在 `remote_scripts/` 中添加 XPU 执行脚本**:
   - 参考 `run_kernel.py`，实现编译、运行、benchmark、baseline 测量

4. **Planner 自动适配**:
   - 不需要修改 Planner 代码，它根据 `caps` 自动跳过不满足 `requires_caps` 的配方
   - 自动选择匹配 `dsl` 的代码模板
