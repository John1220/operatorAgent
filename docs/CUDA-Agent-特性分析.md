# CUDA-Agent 完整特性分析报告

> 项目地址：https://github.com/BytedTsinghua-SIA/CUDA-Agent
> 项目主页：https://cuda-agent.github.io/
> 数据集：https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K
> 研究机构：ByteDance Seed × 清华大学 AIR（SIA-Lab）

---

## 一、项目定位与核心成就

CUDA-Agent 是**首个已知通过大规模 Agentic RL（智能体强化学习）训练、在 CUDA Kernel 生成任务上超越 Claude Opus-4.6、Gemini 3 Pro 等顶级闭源模型**的系统。它在 KernelBench 基准上达到 SOTA，在所有难度级别上稳定超越 `torch.compile` 基线，最难场景下领先优势尤其显著。

### 关键指标（KernelBench）

| 指标 | Overall | Level-3（最难） |
|---|---|---|
| Pass Rate（通过率） | 98.8% | 94% |
| Faster Rate vs. Eager | 98.4% | 94% |
| Faster Rate vs. Compile | 96.8% | 90% |
| Speed-up vs. Eager | 2.60x | 1.80x |
| Speed-up vs. Compile | 2.11x | 1.52x |

对比闭源基线：Claude Opus 4.5 与 Gemini 3 Pro 的 faster rate vs. compile 仅约 66.4%–69.6%，speed-up 约 1.42x–1.46x。CUDA-Agent 在 Level-3 上领先约 +40 个百分点。

---

## 二、整体架构设计：三大核心组件

CUDA-Agent 由三个相互配合的核心组件构成：

1. **可扩展数据合成（Scalable Data Synthesis）** —— 产生高质量训练任务
2. **技能增强的 CUDA 开发环境（Skill-Augmented Agent Environment）** —— 提供可靠执行反馈
3. **稳定的长时序 RL 训练流水线（Stable Long-Horizon RL Training）** —— 让策略持续学习

这三者共同构成"数据 → 环境 → 训练"的完整闭环。

---

## 三、Agent 协作模式与环境设计

### 3.1 Agent 工作循环（ReAct-style）

Agent 采用 **ReAct 风格工作流**，配合编码工具与 CUDA 技能规范（SKILL.md）实现迭代式开发：

```
profile 原始 PyTorch → 实现 CUDA kernel/bindings → 在 GPU 沙箱编译
   ↑                                                          ↓
   └────────── 迭代 debug / 性能优化 ←── verify + profile ←────┘
```

**标准工作流**：
1. 对原始 PyTorch 模型做性能 profiling，定位瓶颈
2. 在 `kernels/` 目录实现自定义 CUDA/C++ kernel 及其 Python binding
3. 通过 `utils/compile.sh` 在 GPU 沙箱中编译扩展
4. 运行 `utils/verification`（正确性）与 `utils/profiling`（性能）
5. 根据 profiler 反馈迭代优化，直至达到性能目标

### 3.2 `agent_workdir` 标准化工作区结构

仓库提供的 `agent_workdir` 是一个标准化的 Agent 工作示例，覆盖"实现 → 编译 → 验证 → 性能分析 → 迭代"全流程：

```
agent_workdir/
├── SKILL.md                  # Agent 执行的工作流约束与优化规则（核心"技能规范"）
├── model.py                  # 原始 PyTorch 基线模型（不可修改）
├── model_new.py              # Agent 用自定义 CUDA 扩展优化的模型（Agent 工作区）
├── binding.cpp               # 共享 Python binding 注册主模块（不可修改）
├── binding_registry.h        # 注册系统基础设施（不可修改）
├── kernels/                  # Agent 唯一可写入区域：自定义 CUDA/C++ kernels + bindings
└── utils/                    # 不可修改的工具集
    ├── compile.py / compile.sh    # 扩展构建脚本
    ├── verification.py            # 正确性验证
    └── profiling.py               # 性能对比（vs baseline & torch.compile）
```

**关键设计**：基础设施（`binding.cpp`、`binding_registry.h`、`utils/`）被锁定，Agent 只能在 `kernels/` 和 `model_new.py` 中工作，这保证了环境可控、防作弊。

---

## 四、SKILL.md：技能规范（Agent 的"领域知识注入"）

SKILL.md 是项目的**知识库/技能规范设计核心**——它把专家级 CUDA 优化经验以规则形式注入 Agent，是 RAG/知识库设计的体现。

### 4.1 严格禁令（防止捷径行为）

- **C++ 中禁止使用任何 `torch::*` 或 `torch::nn::functional::*`**
- `model_new.py` 中禁止使用 torch 运算（只允许 tensor 创建与自定义 op）
- 禁止第三方库（仅允许 cuBLAS 做 GEMM、cuDNN 做 Conv）
- 禁止修改 `utils/`、`binding.cpp`、`binding_registry.h`

### 4.2 允许的操作（约束生成空间）

- C++：纯 CUDA kernel、cuBLAS（GEMM）、cuDNN（Conv 强制要求）
- Python：`torch.tensor` 创建、自定义扩展 op、tensor 属性
- 内存：仅 `torch::empty_like` 分配

### 4.3 文件分离设计（工程优化）

- `.cu` 文件：纯 CUDA kernel + `extern "C"` launcher（无 PyTorch 依赖）
- `.cpp` 文件：cuDNN/cuBLAS 调用
- `_binding.cpp` 文件：PyTorch tensor 处理 + Python 绑定
- **分离的目的**：加快编译速度，便于独立调优

### 4.4 运行时调参机制

通过 binding 传入 `config` 参数，实现**无需重新编译即可切换 kernel 配置**：

```cpp
// .cu 中提供模板化 launcher
template<int BLOCK_SIZE, int TILE_SIZE>
__global__ void my_kernel_impl(...);

extern "C" void my_kernel_launcher(..., int config, cudaStream_t stream) {
    switch(config) {
        case 0: my_kernel_impl<256, 16><<<...>>>(...); break;
        case 1: my_kernel_impl<128, 32><<<...>>>(...); break;
        ...
    }
}
```

### 4.5 优化策略优先级体系

SKILL.md 给出明确的优化优先级，相当于专家知识注入：

**Priority 1：算法级优化（>50% 影响）**
- Kernel fusion（减少访存）
- Shared memory tiling（数据复用）
- Memory coalescing（连续访问）

**Priority 2：硬件利用率（20%–50% 影响）**
- 向量化加载（float2/float4）
- Warp 级原语（`__shfl_sync`、`__ballot_sync`）
- Occupancy 调优（block size、寄存器使用）

**Priority 3：精细调优（<20% 影响）**
- 指令级并行
- 混合精度（FP16/TF32）
- Prefetching 与 double buffering

### 4.6 迭代要求

- **正确性失败必须迭代直到通过**（无例外）
- **性能目标**：最低比 `torch.compile` 快 5%（≤ 0.95× baseline），目标追求 20%+ 提升
- 迭代策略：前 1–2 轮达到最低门槛，后续 3–5 轮激进推进
- **完成前强制清理**：`kernels/` 仅保留最终版本，删除所有中间尝试

### 4.7 优化 Checklist（完整知识库）

涵盖 Memory Coalescing、Kernel Fusion、Shared Memory、Grid-Stride Loops、Boundary Checks、Vectorized Memory、Warp Primitives、Occupancy Tuning、Bank Conflict Avoidance、Loop Unrolling、Tensor Cores（WMMA/MMA）、Mixed Precision、Persistent Kernels、CUDA Graphs、Double Buffering 等十余项优化检查点。

---

## 五、代码生成策略

### 5.1 代码生成范式

Agent 生成的代码遵循"分离 + 模板化 + 可调参"范式：

1. **`.cu` 文件**：纯 CUDA 实现，使用 template 参数化（BLOCK_SIZE、TILE_SIZE），通过 `extern "C"` launcher 暴露 C 接口
2. **`_binding.cpp` 文件**：处理 PyTorch tensor 校验（is_cuda / is_contiguous / dtype）、内存分配（`torch::empty_like`）、流获取（`c10::cuda::getCurrentCUDAStream`），并暴露 `config` 参数
3. **`model_new.py`**：`__init__` 与原 Model 完全一致（保证 `state_dict` 兼容），`forward` 中**仅调用自定义 op**，禁止 torch 运算

### 5.2 模型权重兼容性

`cuda_model.load_state_dict(torch_model.state_dict())` —— 新模型必须保持与原模型一致的参数结构，这是验证脚本的关键要求。

### 5.3 编译目标

```bash
TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh
```

`TORCH_CUDA_ARCH_LIST=9.0` 指向 SM90（Hopper 架构），表明训练与评测主要面向 H100/H200 类硬件。

---

## 六、验证方法（双轨制）

### 6.1 正确性验证（`utils/verification.py`）

**5 输入随机检查 + 反作弊机制**：

```python
num_checks = 5
for i in range(num_checks):
    torch_inputs, cuda_inputs = build_inputs()  # 每次重新生成输入
    torch_output = torch_model(*torch_inputs)
    with block_torch_functional():              # 屏蔽 torch.nn.functional
        cuda_output = cuda_model(*cuda_inputs)
    check_equal(cuda_output, torch_output)
```

**关键反作弊设计**：
- **`block_torch_functional` 上下文管理器**：在 CUDA 模型前向时屏蔽所有 `torch.nn.functional.*` 调用，防止 Agent 偷偷回退到 PyTorch 实现
- **5 次不同输入检查**：防止硬编码输出
- **容差**：`atol=1e-2, rtol=1e-2`
- **递归类型检查**：支持 Tensor / list / tuple / dict / 标量的嵌套输出结构

### 6.2 性能 Profiling（`utils/profiling.py`）

**三路对比 + 同步预热**：

```python
warmup_iters = 5
run_iters = args.iters  # 默认 10

# 使用 torch.profiler，统计 device_time（CUDA 设备时间）
cuda_time   = benchmark_model(cuda_model,   cuda_inputs,   warmup, iters)
torch_time  = benchmark_model(torch_model,  torch_inputs,  warmup, iters)
compile_time = benchmark_model(torch_compile_model, torch_inputs, warmup, iters)
```

**关键设计**：
- 基于 `torch.profiler` 统计 **CUDA device_time**（而非 wall time），更准确反映 kernel 真实性能
- `torch.cuda.synchronize()` 在测量前后同步，避免异步流干扰
- 同时对比三个对象：原始 PyTorch（Eager）、`torch.compile`、自定义 CUDA 扩展
- 支持 `--single-run` 单独运行某个目标（用于 RL 环境）

---

## 七、优化手段总览

### 7.1 Agent 层面

- **ReAct 迭代**：profile → 实现 → 编译 → verify → profile → 优化，多轮闭环
- **运行时调参**：通过 `config` 参数切换 kernel 配置，无需重编译
- **激进优化策略**：不回退到慢版本，正确性失败时修复而非降级

### 7.2 Kernel 层面（SKILL.md 注入）

- Kernel Fusion、Shared Memory Tiling、Memory Coalescing
- 向量化加载（float2/float4）、Warp 原语（`__shfl_sync`）
- Occupancy 调优、Bank Conflict 规避、Loop Unrolling
- Tensor Cores（WMMA/MMA）、混合精度（FP16/TF32）
- Persistent Kernels、CUDA Graphs、Double Buffering

### 7.3 工程层面

- `.cu` 与 `_binding.cpp` 分离，加快编译
- 模板化 kernel + `extern "C"` launcher
- 基础设施锁定，防止 Agent 篡改验证逻辑

---

## 八、RL 训练流水线（核心算法贡献）

### 8.1 分阶段训练设计（稳定长时序 RL）

训练分阶段进行，以稳定 CUDA 编码的长时序 RL：

**Stage 1：单轮 PPO 预热**
- 先做 single-turn PPO，改进基础 CUDA 生成能力
- 为后续多轮 Agent 训练打下基础

**Stage 2：Actor 初始化（RFT）**
- 使用 **Rejection Fine-Tuning（RFT）**：从采样轨迹中筛选正收益轨迹
- **过滤无效循环与非法工具调用模式**，降低策略崩溃风险

**Stage 3：Critic 初始化（Value Pretraining）**
- 对 Critic 做 value pretraining，使早期 advantage 估计可靠

**Stage 4：多轮 Agentic RL**
- 在完成 actor/critic 初始化后进入完整多轮 Agent RL

### 8.2 长时序训练能力

得益于多阶段设计，训练可稳定支持：
- **上下文长度高达 128k**
- **训练阶段最多 150 轮**
- **评测阶段最多 200 轮**
- 奖励持续增长

### 8.3 奖励设计（基于里程碑的离散奖励）

- **鲁棒的奖励调度**：基于里程碑的离散奖励（correctness + speedup 双里程碑）
- **目标要求**：通过正确性检查 + 比 `torch.compile` 快至少 5%

### 8.4 反奖励作弊控制（关键工程贡献）

为防止策略学到捷径行为，环境设计了多重反作弊机制：

| 控制项 | 说明 |
|---|---|
| 保护验证/分析脚本 | `utils/` 不可修改 |
| 禁止 fallback 调用 | `block_torch_functional` 屏蔽 |
| 5 输入正确性检查 | 防硬编码输出 |
| 同步预热 profiling | 防异步流作弊 |
| 禁止网络检索 | 防外部求助 |
| 运行时范围控制 | 1ms–100ms |

这些约束确保策略学习强调**真实 kernel 质量**而非捷径行为。

---

## 九、数据集：CUDA-Agent-Ops-6K

### 9.1 规模与格式

- **6,000 条合成训练样本**
- 3 个字符串列：`ops`（算子描述）、`data_source`（来源标签）、`code`（可执行 PyTorch 代码）
- 无空值

### 9.2 三阶段数据构建流水线

**Stage 1：种子问题爬取**
- 从 `torch` 和 `transformers` 挖掘参考算子
- 每个任务表示为可运行的 PyTorch 算子逻辑（含 `__init__` 与 `forward`）

**Stage 2：LLM 组合合成**
- 采样最多 5 个 torch 算子，顺序组合成融合任务
- 超越单算子模式，增加任务多样性与优化复杂度

**Stage 3：执行驱动过滤与去污染**
- 必须在 eager 与 `torch.compile` 两种模式下均可执行
- 移除随机性算子（保证可复现）
- 移除退化输出（常量/不可区分输出）—— **反作弊检查**
- 控制运行时在 1ms–100ms（eager 模式）—— **工作负载控制**
- 移除与 KernelBench 高度相似的任务 —— **去污染**

### 9.3 数据来源标签

- `torch#N`：来自 torch，N 表示组合的算子数（1–5）
- `transformers`：来自 transformers 库（如 `MPNetLMHead_2`）

### 9.4 数据多样性示例

数据涵盖极广，从单算子（如 `["torch.sigmoid"]`）到 5 算子融合（如 `["torch.einsum", "nn.AdaptiveAvgPool3d", "nn.LeakyReLU", "nn.Conv3d", "nn.ConvTranspose1d"]`），覆盖 Conv、Attention、Norm、Loss、LSTM、Embedding 等各类算子。

---

## 十、硬件支持

- **训练/评测目标架构**：SM90（Hopper，`TORCH_CUDA_ARCH_LIST=9.0`），即 H100/H200 类 GPU
- **允许的底层库**：cuBLAS（GEMM）、cuDNN（Conv/ConvTranspose 强制要求）
- **运行环境**：GPU 沙箱，使用 `sudo` 运行验证与分析工具
- **语言构成**：Python 68.2%、C++ 21.7%、CUDA 8.5%、Shell 1.6%

---

## 十一、RAG / 知识库设计

CUDA-Agent 的"知识库"通过 **SKILL.md 技能规范**实现，而非传统向量检索式 RAG：

### 11.1 知识注入方式

- **结构化规则文档**：SKILL.md 以 Markdown 形式提供完整的工作流约束、文件结构、代码模板、优化策略、检查清单、常见问题
- **专家经验编码**：优化优先级体系（算法级 > 硬件级 > 精细级）将专家经验显式编码
- **代码模板**：提供 `.cu`、`_binding.cpp`、`model_new.py` 的完整模板，减少 Agent 的格式错误

### 11.2 与传统 RAG 的差异

- **无检索**：知识以静态规范形式注入，禁止网络检索（反作弊）
- **约束式**：知识既是指导也是约束（禁令 + 允许列表）
- **环境耦合**：知识与不可修改的工具脚本（`utils/`）共同构成"硬环境"

---

## 十二、与现有方法的差异

| 维度 | 现有方法 | CUDA-Agent |
|---|---|---|
| 训练方式 | 训练无关精炼 / 固定执行反馈循环 | 大规模 Agentic RL |
| 能力来源 | 外部精炼 | 内在优化能力 |
| 环境 | 简单反馈 | 技能增强 + 反作弊 |
| 数据 | 人工标注（昂贵） | 可扩展合成流水线 |
| 训练稳定性 | 未明确 | 分阶段 + RFT + Value Pretraining |

---

## 十三、开源内容

项目开源了三部分以支持社区：

1. **`agent_workdir`**：标准化 Agent 工作环境（含 SKILL.md、工具脚本、示例模型）
2. **CUDA-Agent-Ops-6K 数据集**：6,000 条训练样本（HuggingFace）
3. **SKILL.md**：专家级 CUDA 优化技能规范

注：模型权重与完整 RL 训练代码未开源，当前仓库主要发布数据与环境以支持 LLM-based CUDA 生成社区。

---

## 十四、团队

**作者**：Weinan Dai\*、Hanlin Wu\*、Qiying Yu、Huan-ang Gao、Jiahao Li、Chengquan Jiang、Weiqiang Lou、Yufan Song、Hongli Yu、Jiaze Chen、Wei-Ying Ma、Ya-Qin Zhang、Jingjing Liu、Mingxuan Wang、Xin Liu、Hao Zhou†

**机构**：ByteDance Seed × 清华大学 AIR（SIA-Lab）

**GitHub 贡献者**：wu-han-lin（Hanlin Wu）、nwiad（Dai, Weinan）

---

## 总结

CUDA-Agent 的核心创新在于把"数据合成 + 技能增强环境 + 稳定长时序 RL"三者系统化结合：用可扩展合成解决数据稀缺，用 SKILL.md + 锁定工具集 + 反作弊机制提供可靠执行反馈，用分阶段训练（单轮预热 → RFT → Value Pretraining → 多轮 RL）稳定 128k 上下文、150–200 轮的长时序训练。其反作弊设计（屏蔽 torch.nn.functional、5 输入检查、同步预热、禁网）与分阶段训练设计是工程上最具参考价值的部分。
