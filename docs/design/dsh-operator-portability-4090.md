# 算子开发插件的显卡可迁移性评估（2026-10-30）

状态：**评估文档，未在 4090 或任何 Ada 卡上执行编译、测量、SSH 或模型调用。**
本文只回答两个问题：换成 4090 类显卡后插件是否同样适用、8 GB 显存是否够用。
它不修改任何判据、容差、预算或门槛，不改变 S4/S5/S6 的既有结论，也不把 H20 的
数值外推成新设备的结论。文中 4090 的规格数字是公开规格的转述，标注为**未在本仓探测**；
本仓自己的峰值表把该行标成"口径未核"，见 §4 G3。

## 1. 结论

| 层 | 是否依赖具体型号 | 依据 |
|---|---|---|
| 插件本体（`operator_run` / `verify` / `repair` / `bench`、MCP/CLI、取消、留存、证据包） | **不依赖** | 认卡靠探测，未命中硬件表就走"什么都不承诺"的未知路径（[hardware_caps.py:458](../../operator_agent/hardware_caps.py)）；换机只改 SSH 别名（[remote_executor.py:614](../../operator_agent/remote_executor.py)） |
| 已有数值证据（S4/S5 结果、峰值、脊点、launch 地板、噪声带） | **完全依赖** | 全部是 H20 实测；4090 脊点约 327 FLOP/B，与 H20 的约 37 相差近 9 倍，优化取向相反 |
| 实验装置（S5 的 `h20` 后端代号、固定主机/UUID/型号校验） | **硬编码** | [s5_formal_run.py:39](../../benchmarks/s5_formal_run.py)、[s5_verify_service.py:68](../../benchmarks/s5_verify_service.py)、[s5_workflow_preflight.py:188](../../benchmarks/s5_workflow_preflight.py) 等 |
| 标定产物（floor / L2 冲刷 / dispersion / `NOISE_BAND`） | **完全依赖** | 都按 8 卡共享 H20 标定（[measurement.py:92](../../operator_agent/measurement.py)） |

一句话：**依赖的不是 H20 这个型号，而是"每张卡先标定、再出结论"这条纪律。**
迁移是可以做的工程，不是重写；但 S4/S5 的成绩一条都不能继承，且必须先把 §4 的
五个缺口补上，否则新卡上的性能结论会比 H20 上更松。

本文只把[H20 冻结记录](../status/dsh-operator-gpu-freeze-20261030.md)里那条既有边界
（"跨卡验证未做……不能声称跨 GPU 通用"）展开成可执行的迁移清单，不越过它。

## 2. 型号无关的部分

| 能力 | 代码位置 | 行为 |
|---|---|---|
| 换机器 | [remote_executor.py:614](../../operator_agent/remote_executor.py) | 默认 `algorithm-h20`，可用 `REMOTE_SSH_HOST` 或 ssh config 别名覆盖，由 Host 块解析端口/用户/密钥 |
| 认卡 | [hardware_caps.py:458](../../operator_agent/hardware_caps.py) | 命中 `_KNOWN_HARDWARE` 才用表内规格；未命中从 `unknown_caps()` 起，只按 compute capability 推能在整个受支持架构区间保证的中立事实（共享内存、warp shuffle、块归约可声明；TMA/FP8 等一律 False） |
| 认不出就报错 | [hardware_caps.py:545](../../operator_agent/hardware_caps.py) | `get_caps_for_target()` 未命中直接抛 `UnknownHardware`，不兜底成 A100 的规格 |
| 型号混淆守卫 | [hardware_caps.py:328](../../operator_agent/hardware_caps.py) | `_match_known_key` 要求键前后不紧跟数字，"nvidia h200" 不会吃到 `h20` 的 4.0 TB/s |
| 资源预检 | [resource_check.py:254](../../operator_agent/resource_check.py) | smem 未知时返回"上界未知，交由原生检查"，既不误拒也不假装剪枝 |
| Hopper 专属配方 | [optimization_planner.py:103](../../operator_agent/optimization_planner.py) | TMA / warp specialization / PDL / FP8 由 `requires_caps` 门控，Ada 上判 False 就选不出来 |
| 卡号固定 | [remote_executor.py:747](../../operator_agent/remote_executor.py) | 通过 `nvidia-smi` 选卡并钉住，单卡机器同样成立（索引 0） |
| LLM / 控制面 | — | 模型走远端 API，控制机只需 Python，不需要本地 GPU |

结论：4090 不会被当成 H20 使用。上表是刻意的设计（缺证据不许折成一个能当分母的数），
不是巧合。

## 3. 与 H20 绑定的部分

### 3.1 数值

| 项 | H20（本仓实测/推导） | 4090（公开规格，未在本仓探测） |
|---|---|---|
| FP16 dense | 148 TFLOPS | 约 330 TFLOPS（FP16 累加；FP32 累加约减半） |
| 显存带宽 | 4.0 TB/s | 约 1.008 TB/s |
| 脊点（算力/带宽） | 约 37 FLOP/B | 约 327 FLOP/B |
| launch 地板 | 20.24 µs（L2 冲刷口径 4.64 µs） | 未测 |
| 共享内存/SM | 228 KiB | 未测（Ada 量级约 100 KiB，须以探测为准） |
| L2 | 60 MiB | 未测（AD102 量级 72 MiB，须以探测为准） |

脊点差近 9 倍意味着：H20 上"memory-bound 友好"的配方取向在 4090 上不成立，
两边同一形状的 bound 判定可以相反。**已有的加速比、bound 判定、配方禁用记录
一律不得跨卡引用。** 另外 H20 资源已于 2026-10-30 释放，之后不再新增真机证据
（[GPU 冻结记录](../status/dsh-operator-gpu-freeze-20261030.md)）。

### 3.2 实验装置里的 `h20` 是硬编码

这些是"真机验证后端"的代号，不是插件能力：就地改成 4090 而不重新冻结，
等于把旧分母套到新批次上。

| 位置 | 校验内容 | 迁移动作 |
|---|---|---|
| [s5_formal_run.py:39](../../benchmarks/s5_formal_run.py) | `GATEWAY_HOST = 'algorithm-h20'` | 改为新主机别名并重冻结计划 |
| [s5_verify_service.py:68](../../benchmarks/s5_verify_service.py) | `backend` 只接受 `mock`/`h20`，且非 mock 分支当前返回"real device adapter not wired" | 新增后端标签并接线，或明确本批不用该服务 |
| [s5_workflow_preflight.py:188](../../benchmarks/s5_workflow_preflight.py) | GPU 型号必须为 `NVIDIA H20` | 改为按探测名/cc 断言 |
| [s5_workflow_preflight.py:41](../../benchmarks/s5_workflow_preflight.py) | 预检要求空闲显存 ≥ 8192 MiB | 小卡不可能满足，必须重设（见 §5） |
| [s5_ruler_preflight.py:75](../../benchmarks/s5_ruler_preflight.py) | 固定 GPU 索引/UUID/型号 | 换 UUID 与型号 |
| [workload_codegen_h20.py:291](../../benchmarks/workload_codegen_h20.py) | `'H20' not in name` 即拒 | 同上 |
| [s5_preflight_plan.py:34](../../benchmarks/s5_preflight_plan.py) | `verify_backend` 只允许 `mock`/`h20` | 同 s5_verify_service |
| [test_nvidia_matmul.py:109](../../test_nvidia_matmul.py) | 驱动里写死 `compute_units=78` | 只影响这三个历史驱动，不属插件路径 |
| [config.yaml](../../config.yaml) | 注释即规格索引，**全仓无模块读取** | 仅需同步注释，改它不会生效 |

### 3.3 标定产物

floor、L2 冲刷口径、样本离散度阈值、`NOISE_BAND` 都是 H20 的实测值
（[measurement.py:83-92](../../operator_agent/measurement.py)；离散度与污染阈值在
同文件 894/906 行），必须在目标卡上重标。
脚本已存在：`benchmarks/calibrate_achievable_peak.py`、`benchmarks/cuda_calibration_h20.py`、
`benchmarks/durable_profile_h20.py`（后两者的 `--host/--gpu` 是参数，改参数即可，
但产物目录名带 h20，应按新设备另立目录，不覆盖旧证据）。

## 4. 迁移会立刻暴露的缺口（按危害排序）

### G1 未知卡的 launch 地板不检查，也不降档

`assess()` 在 `launch_floor_us is None` 时只写一条 reason，`validity` 保持 `VALID`
（[measurement.py:1211](../../operator_agent/measurement.py)），于是 `can_gate` 仍为 True
（[measurement.py:1036](../../operator_agent/measurement.py)），连"可禁用配方"也为 True。

H20 上小形状会被 `uninformative` 拦掉；同一份代码在未标定的卡上会把它当成可裁决读数，
而禁用配方是**永久**决策。这是本次评估里唯一"方向性变松"的缺口：**先标 floor 并落表，
再谈任何性能结论。**

### G2 共享内存未知时被解析成 128 KiB

[agents.py:117](../../operator_agent/agents.py) 把 `memory.local.size`（未知时为字符串
`"unknown"`）解析成 128 KiB，再乘 0.9 作为 Designer 的 tile 预算。Ada 量级约 100 KiB，
即提示值偏高约 28%。后果是编译失败多烧几轮（L1 会拒，不会静默错值），但会直接烧预算。
同源的隐患是 [agents.py:111](../../operator_agent/agents.py) 只有 `8.0`/`9.0` 两个分支，
Ada（8.9）落到 else 分支。

### G3 峰值行口径未核，roofline 轴向不参与筛选

[verifier.py:157](../../operator_agent/verifier.py) 有 `4090` 行，但出处标为
`⚠️口径未核`（[verifier.py:177](../../operator_agent/verifier.py)），于是
`peak_trusted` 不为 True，`axis_evidence()` 判轴不可用：配方筛选退化为不过滤、
"已到上界"的出口永不打开。能跑，但拿不到 H20 上那种判停质量。
另外 `_PEAK_BY_CC["8.9"]` 按 L40S 填（[verifier.py:206](../../operator_agent/verifier.py)），
是"宁可偏大"的口径，同样不能用来判停。

### G4 噪声带与污染阈值按 8 卡共享机标定

`NOISE_BAND`（0.1% / 0.4% / 2%）、`CONTAMINATION_UTIL_PCT = 5`
（[measurement.py:894](../../operator_agent/measurement.py)）、
`DISPERSION_DEGRADED = 0.10`（[measurement.py:906](../../operator_agent/measurement.py)）
的注释都写明依据是"8 卡共享、恒有 7 个其他进程"的 H20。若 4090 是接显示器的工作站，
util 长期 >5% 会让每轮都 `degraded`；反之若独占，带宽可能比阈值更干净。需要重标。

### G5 显存与形状的继承边界

96 GB 与 24 GB（或 8 GB）不是一个池子。现用的 L2 套件与冻结形状（见 §5）都在
1 GB 量级以内，可以直接沿用；但任何 96 GB 级的形状不构成 4090 的可执行条件，
也不能因为"H20 上跑过"就当作新卡的验收样本。

## 5. 显存与规模边界（8 GB 是否够用）

### 5.1 实际用过的规模

| 来源 | 形状 | 单张量 | 三张量合计 |
|---|---|---|---|
| [S4 真机四项](../status/dsh-operator-s4-gpu6-results-20261027.md) | MatMul 64³、31×47×19、Softmax 7×33、LayerNorm 7×33 | KiB 级 | 可忽略 |
| [S5 六任务](../../benchmarks/s5_minimal_tasks.py) | 96×128×64、37×53×23、12×128、11×79、10×128、13×83 | KiB 级 | 可忽略 |
| [GPU 冻结](../status/dsh-operator-gpu-freeze-20261030.md) | 4096³ fp16 | 32 MiB | 约 96 MiB |
| 同上 | 8192³ fp16 | 128 MiB | 约 384 MiB |
| 同上 | LayerNorm 16384×6144 fp32 | 384 MiB | 约 1.1 GiB |

attention 的参考实现走 `F.scaled_dot_product_attention`
（[run_kernel.py:476](../../remote_scripts/run_kernel.py)），是 O(S·D) 内存，
不会物化 S×S；只有 SDPA 不可用时的 manual 分支才按 S² 分配。

注意最后一行的来历：16384×6144 的单张量是 402 MB，**超过 256 MiB 的 JSON 上限**，
是先把 seed 路径放宽到 512 MiB 之后才跑通的（冻结记录 §2.6–§2.7）。也就是说
这一档的成败由字节预算决定，不由显存决定。

### 5.2 峰值估算与真正的上限

显存侧估算：

> 峰值 ≈ 2–3 × 最大单张量 + 约 1 GB（CUDA context / torch）+ 0.26 GB（L2 冲刷缓冲）
> + cuBLAS / torch.compile 工作区

按此，8 GB 可给单张量约 2 GB，能装下 matmul 到约 2 万³ 的张量。但**实际先撞到的是
传输字节预算，不是显存**：

| 路径 | 单张量上限 | 出处 |
|---|---|---|
| JSON 传输（输入 + 标量同一预算） | 256 MiB | [input_fidelity.py:17](../../operator_agent/input_fidelity.py)、冻结记录 §2.6 |
| seed 可复算路径 | 512 MiB | [input_fidelity.py:20](../../operator_agent/input_fidelity.py)、冻结记录 §2.7 |

把字节上限换成形状：512 MiB 允许的最大方阵边长，fp16 约 16384、fp32 约 11585。
即 fp16 matmul 在现有协议下的形状上限约 16384³，fp32 约 11585³ —— 这些张量的峰值
（单张量 512 MiB、三张约 1.5 GiB）仍然远小于 8 GB。

**所以 8 GB 足够跑完整个开发闭环**，包括 L1/L2、L3 计时与修复循环；真正卡住更大
工作负载的是传输协议的单张量预算，而不是卡上的显存。生产路径本身也没有显存门槛：
控制面无总显存检查，只有 S5 驱动层有预检策略。

### 5.3 但 8 GB 会收窄三件事

1. **候选写爆显存不会被记成候选的错。** `_INFRA_MARKERS` 含 `"oom"` / `"out of memory"`
   （[failure_route.py:93](../../operator_agent/failure_route.py)），OOM 归"基础设施失败"：
   不算迭代、不学习、重试。卡越小，候选自己的内存缺陷越容易被记成环境未知。
2. **可裁决的读数更少。** 8 GB 档的卡算力与带宽都低，小形状上 launch 地板占比更高，
   `uninformative` 更多。S4 在 H20 上已经是"完整正确 4/4、性能可裁决 0/4"，
   换更小的卡不会改善；缓解只能靠更大的形状，而形状先受 256/512 MiB 的传输预算限制。
3. **设备预检阈值要重设。** 8192 MiB（[s5_workflow_preflight.py:41](../../benchmarks/s5_workflow_preflight.py)）
   在 8 GB 卡上等于要求整卡全空；4096 MiB（[s5_ruler_preflight.py:76](../../benchmarks/s5_ruler_preflight.py)）
   尚可；通用默认 1024 MiB（[_preparation.py:21](../../benchmarks/_preparation.py)）无问题。

补充：峰值显存已经在被记录，只作观测不参与判定
（[verifier.py:1946](../../operator_agent/verifier.py) 的 `candidate_peak_memory_bytes` /
`baseline_peak_memory_bytes`），所以小卡上还能顺带攒"我们的核多吃多少显存"的数据，
不需要为这项观测预留大显存。

## 6. 迁移的最小动作

按顺序做，每步都能独立验收；任何一步没做完就不进入下一步的性能结论。

1. **让 runner 自描述硬件。** ✅ 已做（2026-11-01，`0.6.6`/`0.1.15`）：`gpu_info` 现在从
   `torch.cuda.get_device_properties` 逐字段探测每 SM 共享内存、块可 opt-in 的字节数、L2 大小、
   每 SM 线程数、寄存器数与 warp 宽度，带 `probed_by`（[run_kernel.py:1734](../../remote_scripts/run_kernel.py#L1734)）；
   **读不到就省键**，另给一条 `hardware_report_missing` 说明「不许改用兜底常量」。
   验收（未知卡不再落进 G2 的 128 KiB 兜底）由 `tests/unit/test_smem_budget.py` 的
   runner↔Designer 键名守卫与变异表 M17 钉住。
2. **修 G2 的解析分支。** ✅ 已做（同一批）：Designer 侧提出
   [`smem_budget()`](../../operator_agent/generate/agents.py#L51)，顺序为
   **实测 > 按 cc 的常量表 > 被显式申报的 legacy 字段 > 抛 `HardwareProbeFailed`**；
   两处 `128` 兜底（else 分支与 legacy 字符串解析失败的 `else 128`）一起删掉。
   验收：未知卡上 Designer 拿到的是**拒绝**而不是一个偏大的数——`{}`、cc 8.9 无实测、
   以及 CPU 回退那句 `"N/A"` 三条输入各自都红过（变异表 M10–M12）。
   副作用要写在脸上：**老 runner 在 cc 8.9 上现在会抛**，换卡必须连 runner 一起部署。
3. **加目标卡的硬件条目并实测峰值。** 用 `peak_probe` 实测替代 ⚠️ 行；验收：
   `peak_trusted=True` 且出处字符串不再以 ⚠️ 开头。
4. **标定 floor / L2 冲刷 / dispersion / `NOISE_BAND`**，把 floor 落进硬件条目。
   验收：`launch_dominance` 不再返回 None（G1 恢复检查）。
5. **跑一次新身份的小批验收**（S4 的 3–5 个任务，任务/预算/模型/Suite 重新冻结）。
   验收：与 S4 同口径出表——完整正确、性能可裁决、未知、成本四栏。
6. **再决定 S5。** 若要在新卡上重做对照，先改 §3.2 的装置并重新冻结 24 格；
   不做事后替换，不沿用 8192 MiB 预检，不把 H20 分母带进新批次。

不需要动的部分：prompts/skills（文本，Hopper 倾向的配方会被 `requires_caps` 过滤）、
MCP/CLI 协议、取消语义、留存与证据包、S5 的 macOS 读取隔离（`sandbox-exec` 在控制机侧，
与 GPU 无关）、L2 自动套件的生成逻辑（形状由契约轴界定）。

## 7. 证据边界

- 本文没有在 4090/Ada 上执行任何编译、测量、SSH 或模型调用；§3.1 的 4090 数字是公开规格
  转述，不是本仓探测值，也不是实测可达值。
- 本仓自己的峰值表已把该行标为"口径未核"，因此本文的脊点比较只用来说明"差异极大"，
  不作为任何判停依据。
- G1–G5 是源码对照的结论，每条都给了定位；它们是"未验证的迁移风险"，
  不是已复现的缺陷，也不是已修复项。
- S4/S5/S6 的既有结论、分母与未完成状态不因本文改变；H20 之后不再新增真机证据。
- 本文未修改任何代码、配置或实验装置。

## 8. 参考

- [封版待执行计划](dsh-operator-completion-plan-20261026.md)、[S4 真机结果](../status/dsh-operator-s4-gpu6-results-20261027.md)、[GPU 冻结验收](../status/dsh-operator-gpu-freeze-20261030.md)
- [资源预检证据边界](resource-precheck-evidence.md)、[S5 两组接入方案](s5-plugin-only-comparison-20261029.md)、[S5 隔离与监督边界](s5-isolation-and-supervision-20261028.md)
- [插件说明](../../integrations/dsh-operator/README.md)