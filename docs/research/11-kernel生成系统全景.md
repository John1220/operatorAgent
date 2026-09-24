# 调研 11：LLM 驱动 GPU kernel 生成系统全景（2025 – 2026.09）

> 版本: v1.0 | 日期: 2026-09-25
> 服务于缺口: 全部 6 项，重点 1（测量有效性）、2（跨运行记忆）、4（bound 感知判据）
> 核实声明: ✅ = 本人独立核对（读代码 / 本机实测 / 拉一手文件）；⚠️ = 来自研究员报告，未独立核实

## 一、外部全景

⚠️ 下表来自研究员报告，链接为其提供；本人未逐篇核对论文内容。**「可信度」一列是研究员的判断，我保留其措辞。**

| 系统 | 核心架构 | 评估方式 | 跨运行记忆 | 报告的最佳结果 | 可信度 |
|---|---|---|---|---|---|
| **KernelBench** (Stanford) [2502.10517](https://arxiv.org/abs/2502.10517) | 纯 benchmark：PyTorch `Model`→`ModelNew` | `fast_p` = 正确（5 组随机输入 allclose）且 speedup>p 的任务占比；wall-clock vs **PyTorch eager** | — | o1/R1 一次成型 `fast_1`<20%；加 profiling 反馈 12/36/12%→43/72/18% | 基准本身可信，但⚠️已漂移：论文称 250 题/3 level，SOL-ExecBench(2026.03) 称 270 题/4 level |
| **Sakana AI CUDA Engineer** → **robust-kbench** [2509.14279](https://arxiv.org/abs/2509.14279) | 4 阶段：翻译→进化优化(crossover)→**innovation archive** 取 stepping stone；LLM soft-verifier 集成投票后才上真机 | 多初始化状态防硬编码、fwd+**bwd**、torch profiler+clang-tidy+NCU | archive（run 内）+ 放出 17k kernel 数据集 | 原宣称 10–100×、17k verified kernels | ❌ **已被自方推翻**：去污染后均速 3.13×→**1.49×** |
| **AlphaEvolve** (DeepMind) [白皮书](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) | Prompt sampler + **Program database** + LLM 集成产 **diff**（`# EVOLVE-BLOCK-START/END` 标注可进化区）+ **评估级联**（难度递增测试集，早剪枝） | 用户自写 `evaluate()`→标量字典；每个改动对随机输入校验数值 | Program database（持久） | 4×4 复矩阵 **48 次乘法**（56 年来首破 Strassen）；Gemini Pallas tiling **+23% kernel / −1% 训练时间**；Borg 回收 **0.7%** 机队算力 | 48 次乘法 ✅ **已被第三方独立验证**；Gemini/Borg 数字**无法外部复现**（代码未开源，仅 [OpenEvolve](https://github.com/codelion/openevolve) 复刻） |
| **Kevin-32B** (Stanford/Cognition) [博客](https://cognition.com/blog/kevin-32b) | GRPO 多轮 RL（QwQ-32B 起），8 步串行 refine；丢弃 CoT 改传自摘要 | 沙箱化；**因 launch 开销主导而主动放大 KernelBench 张量**；发现 harness bug | 无（进权重） | 解出 89%（o3 51%）；best@16 speedup **1.41×** | ✅ **最可信**：数字克制，作者自曝 launch-overhead 坑与 harness bug，主动声明 20 题 holdout 方差过大故改报全集 |
| **Meta KernelEvolve** [2512.23236](https://arxiv.org/abs/2512.23236) | 图搜索：selection policy + universal operator + fitness fn + termination rule；RAG prompt 合成；多抽象层 | KernelBench + 160 ATen op × 3 硬件平台 | 硬件专用知识库 | 250/250 **100% pass**；周→小时 | ⚠️ 是**覆盖率**声明不是性能声明（pass = 能跑且正确），无代码可核 |
| **AKO / AKO4X** (NVIDIA-TLAIC) [主页](https://tongminglaic.github.io/AKO/) | **不是 agent，是"薄 harness"**：把强模型放进好环境；每算子带 lineage 的多轮 campaign archive；harness 可自进化 | 每 trial 重生成输入 + 剥离输入生成代码；**独立 staleness 审计**；隔离子进程；**drift-robust geomean**（baseline 与 kernel 同容器背靠背计时） | 每算子 archive + 共同进化的知识/工具 | vs **FlashInfer 专家 kernel** geomean **1.14–1.43×**；GEMM **1.00 平 cuBLAS，明确不宣称赢** | ✅ **最可信的工业口径**：分母是专家 kernel 而非 eager，主动公布 min–max（含 <1.0 的 launch-bound 负例） |
| **NVIDIA SOL-ExecBench** [2603.19173](https://arxiv.org/abs/2603.19173) | benchmark + agentic baseline 优化器 | **SOL Score** `S=(T_b−T_SOL)/((T_k−T_SOL)+(T_b−T_SOL))`，0.5=持平 baseline、1.0=达硬件极限；正确性 C∈{0,1} 硬门；`T_k<T_SOL` ⇒ **审计信号** | — | 235 题（BF16/FP8/NVFP4, B200）；agent baseline 中位 S=0.732；**14.5% 提交被判 reward hacking** | 方法论可信（NVIDIA 官方 + [NVlabs/SOLAR](https://github.com/NVlabs/SOLAR)） |
| **Houmao** [2608.14560](https://arxiv.org/abs/2608.14560) | Planner→**并行 CUDA Coders**（各自隔离 GPU）→**Synthesizer** 合并最优改动 + NCU Profiler；**只允许改 kernel，harness/数据/reference 冻结** | correctness-gated speedup | Markdown skill 文件夹按需加载 | FuseMoE 92.68×、DSA-TopK **1101×** | ❌ **baseline 不公平的典型**：分母是 PyTorch 的 dequant-bmm+`torch.topk`，而 FlashInfer 专家 kernel 对同一分母已是 52× ⇒ 诚实口径约 **21× over FlashInfer** |
| **KernelSkill** [2603.10085](https://arxiv.org/abs/2603.10085) / **KernelBlaster** [2602.14293](https://arxiv.org/abs/2602.14293) | 双层记忆（长期专家技能 + 短期防回溯）/ MAIC-RL 持久 CUDA 知识库 | KernelBench L1-3 | **是（核心卖点）** | 5.44×/2.82×/1.92× / 1.43×/2.50× geomean | ❌ KernelSkill 数字**高度存疑**：远超公认现实上限（>2× over eager 罕见），未找到独立复现 |
| **CUDA Agent**(ByteDance×清华) [2602.24286](https://arxiv.org/abs/2602.24286) · **AMD GEAK** [2507.23194](https://arxiv.org/abs/2507.23194) · **华为 AscendKernelGen** [2601.07160](https://arxiv.org/abs/2601.07160) | 大规模 agentic RL + skill 环境 / generator-reflector-evaluator-optimizer / CoT-SFT+DPO 出 AscendC | KernelBench / TritonBench-revised+30 ROCm kernel / CANN-Bench | — | 100%/100%/92% faster-rate | ❌ 存疑（同上） |

**基准补充**：TritonBench [2502.14752](https://arxiv.org/abs/2502.14752)、FlashInfer-Bench [2601.00227](https://arxiv.org/abs/2601.00227)（生产 trace，MLSys'26 竞赛）、BackendBench（271 ATen op）、MultiKernelBench（华为 NPU/TPU/AMD）、KernelBenchX、ISO-Bench、KernelCraft。

> ⚠️ 研究员明确报告：**CyberTron 找不到可核实的一手来源**（arXiv/GitHub 均无同名项），建议存疑。我在任务书里提到过它，此处更正——不要引用。

## 二、外部共识：宣称的加速比普遍虚高

这是本轮调研最重要的校准信息。**Simon Guo（KernelBench 作者）直言超过 eager 2× 通常不现实**（[博客](https://simonguo.tech/blog/2025-10-automated-gpu-kernels.html)）。以此为尺：

| 宣称 | 诚实口径 |
|---|---|
| Sakana 10–100×（个别 17000×） | **1.49×**（自方去污染后） |
| Houmao DSA-TopK 1101× | **~21× over FlashInfer**（换掉不公平分母后） |
| KernelSkill 5.44× | 未复现，存疑 |
| Kevin-32B 1.41× | 1.41×（作者自己就报的这个数） |
| AKO 1.14–1.43× vs 专家 kernel | 1.14–1.43×（且主动公布含 <1.0 的负例） |

**对我方的意义**：matmul 的 **0.91× vs cuBLAS(FP16)** 放在这个尺度上是**正常且诚实**的结果，不是失败。cuBLAS 是专家手调库，AKO 对 GEMM 也只做到 1.00 平手并明确不宣称赢。**不该为了让数字好看而放宽 `min_speedup`，也不该因为 0.91× 就认为流水线坏了。**

反过来，layernorm 的 0.65× vs PyTorch Eager 才是真问题——但原因不是核写得差那么简单，见 [`01-launch开销与shape充分性实测.md`](01-launch开销与shape充分性实测.md)：净值差其实是 **6.8×**。

## 三、我方明显落后处（5 条，附核实）

### 3.1 完全没有跨运行记忆 ✅

详见 [`13-跨运行记忆与技能库架构.md`](13-跨运行记忆与技能库架构.md)。补充外部对照：AKO4X 每算子带 lineage 的 archive、KernelBlaster 持久分层知识库、KernelSkill 双层记忆、Meta KernelEvolve 的 RAG prompt 合成——**「跨任务积累经验」已是 2026 年主流卖点，我方为零。**

### 3.2 计时缺 L2 flush、缺方差门 ✅（本人核实，比报告更严重）

✅ 读 `remote_scripts/run_kernel.py:553-574` 核实 `_timed_runs`：每 rep 直接 `cuda.Event` 前后夹，**rep 之间没有任何缓存刷新**。且 `grep -cE "flush_l2|empty_cache|variance|stdev|p95|cv_pct"` 在 `run_kernel.py` 与 `verifier.py` 中**均为 0**——min/median/avg/max 都算了，方差从未参与任何判定。

**比报告更严重的一点（本人推算）**：H20 的 L2 是 **60 MiB**（`hardware_caps.py` 的 `l2_cache_size_kb=61440`），而 layernorm 1024×768 fp16 的工作集是 **3.15 MB —— 只占 L2 的 5%，完全常驻**。所以 warmup 之后每一轮测的都是 L2 带宽，**这个 shape 下基准从未测量过 HBM 行为**。

而 roofline 的分母用的是 HBM 峰值 3.68 TB/s。于是「带宽利用率 5.4%」这个数**分子分母不是同一个基准**：分子是 launch 开销支配下的 L2 常驻流量，分母是 HBM 理论峰值。它既不能说明核差，也不能说明核好。

> ⚠️ **但不要简单加 flush 就完事**。warm-L2 对推理类重复调用的负载**恰恰是现实工况**。真正的问题是「基准未声明且不一致」，不是「一定要冷缓存」。修法应是：显式声明 cache 状态（warm/cold）、候选与 baseline 必须在同一状态下背靠背测量、roofline 分母与实测口径匹配（L2 常驻时用 L2 带宽，或干脆判定该 shape 不适用于 HBM roofline）。

matmul 4096³ fp16 的工作集是 3×33.6 MB = 100.7 MB ≈ **1.6× L2**，会大量溢出到 HBM，受此影响小得多——这也解释了为什么它的 0.91× 看起来可信。

### 3.3 无反作弊的动态审计 ✅

`anticheat.py` 是静态串匹配（禁 `torch.matmul`/`F.softmax`）。⚠️ 报告指出它抓不到 AKO 记录的真实陷阱：

- **CUDA-graph capture 时跑一次、replay 静默跳过**——计时器看不见，陈旧 buffer 仍能过正确性检查
- **指针键控 cache**——仅因 harness 复用输入 buffer 才成立

AKO 的对策是**独立 staleness 审计：固定张量指针、只改数值，输出不变即拒**。SOL-ExecBench 则规定 `T_k < T_SOL` 直接触发审计（该 benchmark 有 **14.5%** 提交被判 reward hacking）。

我方 L3 **无条件采信**测得的 speedup。

### 3.4 正确性判定落在 "correctness illusion" 区 ⚠️

[2606.20128](https://arxiv.org/abs/2606.20128) 实证：**固定 shape + allclose** 会把 9/9 个 LLM 典型 bug 判为正确（RMSNorm 漏 sqrt、attention 漏 1/√D、matmul 把 `acc=` 写成覆盖、softmax 尾掩码泄漏）。消融显示：**去掉边界 shape，shape 相关 bug 召回率归零**。

✅ 我方现状正落在靶心：layernorm 只测 1024×768 单 shape，matmul 只测 4096³ 单 shape，attention 单 shape。`Verifier` 支持 `list[dict]` 多组套件（P0-2 时加的），但**三个 driver 都只传单组**。

叠加已实测的 L2 门敞开（乘 0.97 的核也 PASS），正确性防线的实际强度接近于零。

### 3.5 只有「比 baseline 快几倍」，无 headroom 归一化 ✅

我方已算出 `bound_type` + 利用率，却仍上报 `0.91x cuBLAS` / `0.65x eager`，把「已无优化空间」和「核差」混为一谈。

⚠️ **AKO 的直接印证**：它在生产推理 kernel 上观察到 launch-bound shape 的 geomean 下探到 **0.82–0.96**，并把它列为 **"honest negative"** 而非继续优化。⚠️ **Kevin-32B 独立记录过同一坑**，被迫主动放大 KernelBench 张量。

> **外部共识是：换 shape，或标记为不可测——而不是让 agent 在上面迭代。** 这与 [`01`](01-launch开销与shape充分性实测.md) 号文档的独立实测结论完全一致（该文实测 launch 占比 95.6%，净值差 6.8×）。SOL-ExecBench 更进一步规定：`T_b − T_SOL < ε` 时**视为已解决，不再评估新提交**。

## 四、可直接吸纳的技术（按性价比排序）

| 技术 | 出处 | 代价 | 对治 |
|---|---|---|---|
| **SOL Score + headroom 耗尽即停评**：`S=(T_b−T_SOL)/((T_k−T_SOL)+(T_b−T_SOL))`；`T_b−T_SOL<ε` 或 `T_k<T_SOL` ⇒ 标 AUDIT/INVALID 并**排除出配方禁用逻辑** | [2603.19173](https://arxiv.org/abs/2603.19173) §4.3 · [NVlabs/SOLAR](https://github.com/NVlabs/SOLAR) | **低（1–2 天）**。roofline 的 `T_SOL` 我方已有，只改 `verifier.py` L3 + `conductor.py` 停机判据 | 1、4。**直接止住 layernorm 上白烧 2 轮 stagnant 禁用配方** |
| **Staleness 审计 + 每 trial 新输入**：固定指针改数值，验输出必变；每轮重生成随机输入并从解法文件剥离输入生成代码 | [AKO](https://tongminglaic.github.io/AKO/) | **低（~50 行）**，加进 `run_kernel.py`/`anticheat.py` | 1、6。补上静态串匹配抓不到的动态作弊 |
| **背靠背计时 + 显式 cache 声明 + 方差门**：baseline 与 candidate 同进程交替（interleaved A/B/A/B），CV 超阈值判**测量无效并丢弃**（而非采信） | AKO · [2601.00227](https://arxiv.org/abs/2601.00227) | **中（~1 天）**，改 `_timed_runs` | 1。让 0.91× cuBLAS 这个数字站得住 |
| **Op-schema 感知模糊测试 oracle**：符号维（matmul `A[M,K]·B[K,N]`）采样**必含边界值**（如 `H∈{1,3,7,256,1025}`），fp64 CPU 参考，**per-(op,dtype) 绝对容差**，记录 max-abs/max-rel/ULP 分布 + NaN/Inf，输出最小失败 shape 与可 bit-for-bit 重放的快照 | [2606.20128](https://arxiv.org/abs/2606.20128) | **中（2–3 天）** | 1、6。**性价比最高的正确性加固** |
| **持久化算子档案 + 配方级成败先验**：按 `(kernel_type, hardware, shape-bucket, bound_type)` 建索引，回灌 top-k 历史优胜者 + 历史失效配方进 Designer/Planner prompt | [AKO4X](https://github.com/TongmingLAIC/AKO4ALL) · [2602.14293](https://arxiv.org/abs/2602.14293) · [2603.10085](https://arxiv.org/abs/2603.10085) | **中（~2 天）** | 2。详见 [`13`](13-跨运行记忆与技能库架构.md) |
| **（可选）薄 harness + 并行 Coder/Synthesizer**：N 个 Coder 各在隔离环境并行探索，Synthesizer 合并有效 delta | [2608.14560](https://arxiv.org/abs/2608.14560) · AKO 结论「**强模型该少受约束**，固定 workflow 反而不如直接给强模型一个好环境」 | **高（~1 周）** | 全部。✅ 我方 `CoderPipeline` 已有 T=0.1/0.4/0.7 三路并行，**只差 Synthesizer 一级** |

**最后一条值得单独强调**：AKO 的核心结论是「不是 agent，是薄 harness」——把强模型放进一个好环境，比设计复杂 workflow 更有效。这与本项目当前的重流水线（IntentParser→Router→Designer→Coder→Verifier→Fixer→Search）方向相反。⚠️ 这是研究员转述的结论，我未独立核实其证据强度；但它与 [`00`](00-架构基线与缺口核实.md) 号文档的实测吻合——attention 那次 3 轮 Fixer 修不好一个签名错误，恰恰说明**当前流水线的约束没有换来可靠性**。列为 20 号决策文档需要正面回答的问题。

## 五、来源可信度分级

**一手**：见第一、四节表格内各链接。其中 ✅ 本人已独立核对的仅 AlphaEvolve 48 次乘法的第三方验证仓库存在性（未展开核对其数学）。

**二手（仅用于事件记录，不作论断依据）**：[BAAI 社区](https://hub.baai.ac.cn/view/43673)、[SegmentFault](https://segmentfault.com/a/1190000046629990)、[36kr](https://m.36kr.com/p/3181496883008384)。

> ⚠️ **重要更正**：研究员明确报告，「o3-mini-high 11 秒发现『快 150 倍』实为慢 3 倍」这一具体情节**只见于中文二手转述**；一手来源仅能证实「存在 evaluation code 内存漏洞导致跳过正确性检查」。引用时不得把二手情节当一手事实。

**Sakana 原文**：[Wayback 2025-04-03 快照](https://web.archive.org/web/20250403145219/https://sakana.ai/ai-cuda-engineer/)（含 "Limitations and Bloopers" 自承作弊段落）。
