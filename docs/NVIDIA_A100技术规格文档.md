# NVIDIA A100 技术规格文档

> 本文档汇总 NVIDIA A100 Tensor Core GPU 的技术规格，供算子开发 Agent 做硬件适配参考。
> 数据来源：NVIDIA 官方白皮书、CUDA 编程指南、Triton 官方文档等。

---

## 一、产品概览

### 1.1 产品定位

NVIDIA A100 是基于 **Ampere 架构**的旗舰级数据中心 GPU，专为 AI 训练、HPC 和数据分析设计。它是首个采用 NVIDIA Ampere 架构的 GPU，引入了第三代 Tensor Core、多实例 GPU (MIG)、TF32 精度等创新特性。

- 产品形态：SXM4 / PCIe 4.0
- 制程工艺：7nm（台积电）
- 晶体管数量：542 亿
- 发布时间：2020 年 5 月
- CUDA 计算能力：8.0 (sm_80)

### 1.2 核心规格速览

| 规格项 | SXM4 版本 | PCIe 版本 |
|--------|-----------|-----------|
| **FP64** | 9.7 TFLOPS | 9.7 TFLOPS |
| **FP64 Tensor Core** | 19.5 TFLOPS | 19.5 TFLOPS |
| **FP32** | 19.5 TFLOPS | 19.5 TFLOPS |
| **TF32 Tensor Core** | 156 TFLOPS | 156 TFLOPS |
| **BF16 Tensor Core** | 312 TFLOPS | 312 TFLOPS |
| **FP16 Tensor Core** | 312 TFLOPS | 312 TFLOPS |
| **INT8 Tensor Core** | 624 TOPS | 624 TOPS |
| **显存容量** | 40 GB / 80 GB HBM2e | 40 GB / 80 GB HBM2e |
| **显存带宽** | 1.555 TB/s (40GB) / 2.0 TB/s (80GB) | 1.555 TB/s (40GB) / 2.0 TB/s (80GB) |
| **PCIe 接口** | N/A (NVLink) | PCIe Gen4 x16 |
| **NVLink** | 600 GB/s (第三代) | 不支持 |
| **TGP / TDP** | 400W | 300W |
| **多实例 GPU (MIG)** | 最多 7 个实例 | 最多 7 个实例 |

---

## 二、硬件架构

### 2.1 Ampere 架构核心特性

1. **第三代 Tensor Core**
   - 支持 TF32：FP32 精度下自动使用 Tensor Core，无需修改代码
   - 支持 FP64：HPC 场景也能用上 Tensor Core
   - 支持稀疏：1:2 结构化稀疏，理论算力翻倍
   - 每个 SM 有 4 个 Tensor Core

2. **多实例 GPU (MIG)**
   - 单块 A100 可划分为最多 7 个独立 GPU 实例
   - 每个实例有独立的 SM、显存、L2 cache、显存控制器
   - 支持多种分区配置（1g.5gb / 2g.10gb / 3g.20gb / 4g.20gb / 7g.40gb 等）

3. **第三代 NVLink**
   - 600 GB/s 带宽（SXM 版本）
   - 单 GPU 最多 12 条链路
   - 支持多 GPU P2P 通信

4. **结构稀疏 (Structural Sparsity)**
   - 1:2 稀疏模式（每 4 个权重中 2 个为零）
   - Tensor Core 自动利用稀疏性，理论吞吐翻倍

### 2.2 计算单元

| 项目 | 规格 |
|------|------|
| **SM 数量** | 108 (完整 A100) |
| **每个 SM 的 CUDA Core** | 64 |
| **总 CUDA Core 数** | 6,912 |
| **每个 SM 的 Tensor Core** | 4 |
| **总 Tensor Core 数** | 432 |
| **每个 SM 的 RT Core** | 无（Ampere 数据中心版无 RT Core） |
| **每个 SM 的 SFU** | 4 |
| **Warp 大小** | 32 线程 |
| **每个 SM 最大 Warp 数** | 64 |
| **每个 SM 最大线程数** | 2048 |
| **每个 SM 最大 Block 数** | 32 |
| **每个 Block 最大线程数** | 1024 |

### 2.3 内存层级

| 层级 | 大小 / 规格 | 带宽 / 延迟 | 说明 |
|------|-------------|-------------|------|
| **寄存器** | 256 KB / SM (65536 个 32-bit 寄存器) | 单周期 | 每个线程最多 255 个寄存器 |
| **L1 / Shared Memory** | 164 KB / SM（可配置划分） | ~19 TB/s | 默认 128 KB Shared + 16 KB L1 + 20 KB 纹理缓存 |
| **L2 Cache** | 40 MB（统一） | 约 3.5 TB/s | 支持 cache management / 控制驻留 |
| **HBM2e 显存** | 40 GB / 80 GB | 1.555 / 2.0 TB/s | 5 个 / 6 个 显存 stack，12 个 内存控制器 |
| **PCIe** | PCIe 4.0 x16 | 32 GB/s 双向 | 主机↔GPU 传输 |

**Shared Memory 配置**：每个 SM 的 164 KB 可按以下比例划分：
- 0 KB shared + 164 KB L1
- 8 KB shared + 156 KB L1
- 16 KB shared + 148 KB L1
- 32 KB shared + 132 KB L1
- 64 KB shared + 100 KB L1
- 100 KB shared + 64 KB L1
- 132 KB shared + 32 KB L1
- 164 KB shared + 0 KB L1（Triton 默认优化目标）

**Shared Memory Bank 结构**：
- 32 个 bank（对应 warp 32 线程）
- 每个 bank 宽度：4 bytes（FP32）/ 8 bytes（FP64）
- 连续地址跨 bank 存储
- 同一 warp 内同一 bank 同时访问产生 bank conflict

---

## 三、精度支持与 Tensor Core

### 3.1 精度矩阵

| 精度 | CUDA Core | Tensor Core | 说明 |
|------|-----------|-------------|------|
| FP64 | ✅ 9.7 TFLOPS | ✅ 19.5 TFLOPS | HPC 全精度 |
| FP32 | ✅ 19.5 TFLOPS | ✅ 156 TFLOPS (TF32) | TF32 自动加速 FP32 计算 |
| TF32 | ❌ | ✅ 156 TFLOPS | Ampere 新增，FP32 范围 + FP19 精度 |
| BF16 | ❌ | ✅ 312 TFLOPS | 与 FP32 相同范围，低精度 |
| FP16 | ✅ 19.5 TFLOPS | ✅ 312 TFLOPS | 半精度训练/推理 |
| INT8 | ❌ | ✅ 624 TOPS | 量化推理 |
| INT4 | ❌ | ✅ 1248 TOPS (稀疏) | 极低位宽推理 |

### 3.2 Tensor Core 矩阵形状 (WMMA)

| 精度 | M × N × K | 说明 |
|------|-----------|------|
| FP16 / BF16 | 16×16×16 | 标准形状 |
| FP16 / BF16 | 32×8×16 | 瘦高形状 |
| FP16 / BF16 | 8×32×16 | 宽扁形状 |
| TF32 | 16×16×8 | TF32 是 19-bit mantissa，K 减半 |
| FP64 | 8×8×4 | FP64 Tensor Core |
| INT8 | 16×16×32 | INT8 量化 |

### 3.3 TF32 说明

TF32（Tensor Float 32）是 Ampere 架构新增的精度格式：
- 保留 FP32 的 8-bit 指数（动态范围相同）
- 使用 19-bit 尾数（比 FP32 的 23-bit 略少）
- **默认启用**：`torch.matmul`、`F.conv2d` 等操作在 Ampere 上自动使用 TF32
- 可用 `torch.backends.cuda.matmul.allow_tf32 = False` 禁用
- 精度损失通常 < 1e-5，对大多数训练场景可忽略

---

## 四、软件栈与编程模型

### 4.1 软件栈层级

| 层级 | 组件 | 说明 |
|------|------|------|
| **驱动层** | NVIDIA Driver | 支持 CUDA 11.0+ |
| **CUDA Runtime** | CUDA Toolkit | CUDA 编程模型、CUDA C/C++ |
| **库** | cuBLAS / cuDNN / cuFFT / NCCL / Thrust | 高性能数学与通信库 |
| **编译器** | NVCC / NVRTC / PTXAS | CUDA 编译器、JIT 编译 |
| **DSL** | Triton | Python 风格的 GPU 编程 DSL |
| **框架** | PyTorch / TensorFlow / JAX | 深度学习框架 |
| **推理优化** | TensorRT / vLLM / FasterTransformer | 推理加速引擎 |
| **性能分析** | Nsight Systems / Nsight Compute / nvprof | Profiler 工具 |
| **调试** | cuda-gdb / Nsight Debugger | 调试工具 |

### 4.2 算子开发方式

| 方式 | 语言 | 难度 | 性能上限 | 适用场景 |
|------|------|------|----------|----------|
| **Triton** | Python DSL | 低 | 接近 cuBLAS | 大多数算子，Agent 首选 |
| **CUDA C++** | C++ / CUDA | 中高 | 理论峰值 | 极致性能优化 |
| **cuBLAS / cuDNN** | 库调用 | 低 | 厂商最优 | 标准算子直接调用 |
| **Thrust** | C++ STL-like | 低 | 中等 | 通用并行算法 |
| **PyTorch CUDA Ext** | C++ / CUDA | 中 | 高 | 自定义 PyTorch 算子 |
| **TVM / Ansor** | Schedule DSL | 中高 | 高 | 自动调优 |

### 4.3 Triton 支持

A100 是 Triton DSL 的主要目标平台之一：
- 完全支持 Triton 所有特性（auto-tune、dot、tensor cores 等）
- Triton 官方教程以 A100 为基准
- `tl.dot` 自动调度到 Tensor Core（FP16/BF16/TF32）
- 支持 `tl.dot` 的 `allow_tf32=True` 参数
- 支持 shared memory 配置优化

---

## 五、性能优化要点

### 5.1 Roofline 模型

**A100 SXM4 80GB 的关键 Roofline 参数**：
- 计算峰值（FP16 Tensor Core）：312 TFLOPS
- 显存带宽：2.0 TB/s
- Ridge point（拐点）：312e12 / 2e12 = **156 FLOP/Byte**
- 含义：算子算术强度 > 156 FLOP/Byte 时为计算-bound，否则为访存-bound

**常见算子的算术强度参考**：
| 算子 | 算术强度 | 类型 |
|------|----------|------|
| Matmul (大矩阵) | ~ M/4 (M=K=N 时) | 计算-bound |
| Conv (大特征图) | 高 | 计算-bound |
| Elementwise (add/mul) | 0.33 | 访存-bound |
| Softmax | ~2 | 访存-bound |
| LayerNorm | ~3 | 访存-bound |
| Reduce (sum/mean) | 0.25 | 访存-bound |

### 5.2 Triton 算子优化要点

1. **Tiling 策略**
   - BLOCK_M / BLOCK_N / BLOCK_K 的选择
   - 大矩阵用大 tile（如 128×128），小矩阵用小 tile
   - GROUP_SIZE_M 优化 L2 缓存命中率

2. **内存访问优化**
   - Coalesced memory access（合并访存）
   - Shared Memory 复用 + bank conflict 避免
   - Vectorized load/store（向量化访存）

3. **Tensor Core 利用**
   - 确保 `tl.dot` 的形状匹配 WMMA 指令
   - 使用 `allow_tf32=True`（FP32 输入时）
   - 数据排布优化（连续维度对齐）

4. **软件流水线**
   - `num_stages`（预取阶段数，A100 推荐 3-5）
   - `num_warps`（warp 数，A100 推荐 4-8）
   - 计算与访存重叠

5. **Auto-tuning**
   - 使用 `@triton.autotune` 搜索最优配置
   - 覆盖不同的 BLOCK 大小、stages、warps 组合

### 5.3 常见性能优化手段效果

| 优化技术 | 典型提升 | 适用场景 |
|----------|----------|----------|
| 向量化访存 | 2-4x | 所有访存密集型算子 |
| Shared Memory Tiling | 2-10x | 有数据复用的算子 |
| Tensor Core (FP16→TC) | 8-16x | 矩阵乘、卷积 |
| 软件流水线 | 1.2-2x | 计算访存重叠 |
| Auto-tune | 1.2-3x | 所有算子 |
| 算子融合 | 1.5-5x | 连续访存密集型算子 |

---

## 六、基线性能参考

### 6.1 cuBLAS 性能参考（FP16 Tensor Core）

| 矩阵大小 (M=N=K) | 理论 TFLOPS | 实测 TFLOPS（约） | 达到率 |
|-------------------|------------|------------------|--------|
| 128 | 312 | ~50 | ~16% |
| 256 | 312 | ~120 | ~38% |
| 512 | 312 | ~220 | ~71% |
| 1024 | 312 | ~280 | ~90% |
| 2048 | 312 | ~300 | ~96% |
| 4096 | 312 | ~305 | ~98% |
| 8192 | 312 | ~310 | ~99% |

> 注：小矩阵受启动延迟和调度开销影响，达不到峰值；大矩阵接近理论峰值。

### 6.2 基线选择策略（NVIDIA 平台）

| 基线 | 含义 | 用途 | 预期加速比 |
|------|------|------|-----------|
| **cuBLAS (FP16/BF16)** | NVIDIA 官方 BLAS 库 | 性能天花板参考 | Triton 优化版 0.8-1.0x |
| **cuBLAS (TF32)** | TF32 精度 cuBLAS | FP32 算子的公平对比 | Triton TF32 0.9-1.1x |
| **cuBLAS (FP32)** | 纯 FP32 CUDA Core | FP32 全精度基线 | Triton 可 5-8x (用 TF32) |
| **torch.compile** | TorchInductor 编译 | 编译器优化基线 | 通常比 cuBLAS 慢 |
| **朴素实现** | 手写 for 循环等 | 最低门槛 | 10-100x 以上 |

---

## 七、开发工具

### 7.1 性能分析工具

| 工具 | 用途 | 级别 |
|------|------|------|
| **Nsight Compute** | Kernel 级细粒度分析（指令、内存、寄存器等） | Kernel 级 |
| **Nsight Systems** | 系统级时间线（CUDA API、kernel、memcpy、NCCL 等） | 系统级 |
| **nvprof** | 老版 profiling 工具（已逐步被 Nsight 替代） | 两级 |
| **cuda-event** | 代码内手动计时 | 代码级 |
| **Triton autotune** | Triton 内置自动调优 | Triton 专用 |

### 7.2 调试工具

| 工具 | 用途 |
|------|------|
| **cuda-gdb** | CUDA 代码调试（断点、单步、检查显存等） |
| **cuda-memcheck** | 内存越界、未初始化访问检测 |
| **compute-sanitizer** | cuda-memcheck 的继任者，功能更强 |
| **Nsight Compute Debugger** | GUI 调试器 |

### 7.3 其他开发工具

| 工具 | 用途 |
|------|------|
| **NVCC** | CUDA C/C++ 编译器 |
| **NVRTC** | 运行时编译（JIT） |
| **cuobjdump** | CUDA 二进制反汇编、资源统计 |
| **PTXAS** | PTX → SASS 汇编器 |
| **nvidia-smi** | GPU 状态监控（类似任务管理器） |
| **nvtop** | 终端 GPU 监控（类 htop） |

---

## 八、与算子开发 Agent 的关联

### 8.1 Agent 已利用的 A100 特性

1. **Triton DSL** — 主要代码生成目标，CoderAgent 专门有 Triton prompt
2. **TF32 自动加速** — Verifier 基线选择已考虑 TF32 精度匹配
3. **cuBLAS 天花板** — 以 cuBLAS 为黄金标准衡量算子性能
4. **Roofline 分析** — L3 验证自动计算 Roofline 指标，指导 SearchAgent 优化
5. **Auto-tune 配方** — Optimization Recipes 中包含 autotune 优化

### 8.2 HardwareCaps 中的 A100 配置

`hardware_caps.py` 中 A100 的关键参数：
- FP32: 19.5 TFLOPS
- FP16: 312 TFLOPS (Tensor Core)
- TF32: 156 TFLOPS
- 显存带宽: 1.555 TB/s (40GB) / 2.0 TB/s (80GB)
- SM 数量: 108
- Shared Memory: 164 KB/SM (可配置)
- L2 Cache: 40 MB
- Warp 大小: 32
- 寄存器/线程: 255

### 8.3 优化 Recipe 适用情况

| Recipe | A100 适用性 | 说明 |
|--------|------------|------|
| Tiling 策略 | ✅ 完全适用 | BLOCK_M/N/K 选择 |
| Auto-tune | ✅ 完全适用 | @triton.autotune |
| Tensor Core 利用 | ✅ 完全适用 | tl.dot + WMMA |
| 软件流水线 | ✅ 完全适用 | num_stages / num_warps |
| Vectorized 访存 | ✅ 完全适用 | tl.load/store vectorized |
| Shared Memory 优化 | ✅ 完全适用 | bank conflict 避免 |
| L2 Cache 优化 | ✅ 完全适用 | GROUP_SIZE_M / cache_hints |
| 算子融合 | ✅ 完全适用 | 减少内存访问 |

---

## 九、公开文档资源

### 9.1 官方文档

| 资源 | URL |
|------|-----|
| NVIDIA A100 白皮书 | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-1623830.pdf |
| Ampere 架构白皮书 | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf |
| CUDA C++ Programming Guide | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ |
| Triton 官方文档 | https://triton-lang.org/main/ |
| Triton 矩阵乘教程 | https://triton-lang.cn/main/getting-started/tutorials/03-matrix-multiplication.html |
| Nsight Compute 文档 | https://docs.nvidia.com/nsight-compute/ |
| cuBLAS 文档 | https://docs.nvidia.com/cuda/cublas/ |

---

*文档整理时间：2026-08-02*
*信息来源：NVIDIA 官方白皮书、CUDA 编程指南、Triton 官方文档*
