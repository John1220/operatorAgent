# 融合任务头寸预登记：residual add + LayerNorm

**2026-11-17 起草 · 2026-11-17 冻结。** 用户已确认形状、参考加法精度与测量卡；
容差已在 GPU7 标定并写进 registry。本文件只冻结「先验证提速空间」这一步
（[v0.8.1 下一步](../status/v081/README.md#v081-next-steps) 第 2 条），**不**冻结 A/C 对照。
A/C 要等本文件的头寸门通过后另立预登记。

## 1. 研究问题

> 在 H20 同一张卡、同一组输入上，一份手写的 fused `residual add + LayerNorm` 核，
> 相对当前 eager 实现（`X + R` 与 `F.layer_norm` 两个 kernel），中位时延能否降低
> **至少 2.5% 且超过当窗噪声带**？

这是**头寸门**：过门才值得投入普通 Agent vs 插件的对照；不过门就停，不烧模型预算。
它不回答「插件有没有价值」，也不构成任何性能主张。

## 2. 与 P1 / P2 / D2 的关系（别混轴）

| 轮 | 问什么 | 结论 |
|---|---|---|
| P1 | 一次交付核的性能 parity | `C_worse_observed` |
| P2 | optimize 诊断提示的增量 | `no_increment_observed` |
| D2 Q1 | 参数层单独有没有余量 | `no_increment`（1.31%，带外阈内） |
| D2 结构试验 | 整行加载能不能省一层 | `structure_hurts`（−26.21%） |
| **本轮** | **融合能不能减少一整遍读写、换来 ≥2.5%** | **待测** |

前三轮都在单一 LayerNorm 核的「参数/结构」上找余量，读数为负。融合换的是**内存流量**：
eager 序列要读 X、读 R、写中间和、再读中间和、写 Y（多一遍读写），单核只需读 X、读 R、写 Y。

## 3. 任务定义（已确认）

- **函数语义**：`Y = LayerNorm(X + R)`，沿末轴，总体方差（除以列数），`eps=1e-5`，
  无 affine（无 weight/bias），fp16 输入输出，X 与 R 同形，Y 与 X 同形。
- **输入**：`X`、`R` 两个 fp16 张量，形状 `[32768, 6144]`（2026-11-17 用户确认）。
- **正确性参考**：native eager 序列 —— `S = X + R`（**fp16 加法**，用户 2026-11-17 确认）→
  `F.layer_norm(S, S.shape[-1:], weight=None, bias=None, eps=1e-5)`（fp32 内部累加）。
  参考在远端 GPU 上构造，不下载到控制端。
- **性能基线**：同一个 eager 序列（2 个 kernel），`baseline_type=pytorch_eager`；
  不拿 cuDNN / 厂商融合库当基线。
- **候选**：手写 Triton 单核：读 X、R → fp32 统计 → 写 Y；允许候选在 fp32 中求和
  （这是融合核的自然做法，容差标定要覆盖它，而不是禁止它）。
- **不变式**：不自造 `eps`/`affine`/轴；不支持 bf16、fp8、affine、非末轴、非 2D。

## 4. 正确性 profile 注册（不能借 LayerNorm v2 的判据）

当前 registry 只有 `builtin-layernorm-fp16-cuda-v2` 和 `builtin-attention-fp16-cuda-v1`；
融合族**必须新注册**，不能把 v2 的套件/容差挪过来用。

### 4.1 身份

- 新 profile：`builtin-fused-layernorm-residual-fp16-cuda-v1`
- 新 kernel_type：`fused_layernorm_residual`（规则解析走新关键词，且在 `layernorm` 之前）
- 新受信任参考 primitive：单独文件 + 单独 pin 的 SHA256（不修改 `builtin_layernorm.py`，
  避免改动 v2/attention 已冻结的参考身份）
- 参考执行身份：`ExecutionIdentity` 的 `primitive_sha256` 指向新 primitive；
  `observe_execution(profile)` 按 profile 选择 primitive；runner 的 `execution_receipt`
  与 executor 的部署清单同步扩展
- `fp_fusion` 语义字段记录「候选允许融合；参考本身是 eager 两 kernel」

### 4.2 套件

显式 32 条 case（不走二维 axis 生成器），沿用 v2 已经标定过的 shapes/patterns 矩阵，
但每条 case 有两个输入 `X`/`R`：

- rows `1/7/8/9`、cols `1/31/32/33`、pattern `random/zeros/constant/large/small`；
- 追加 v2 的 20 条单列/非二进制/近常量/大小变化/正负大值矩阵；
- contiguous 与真实 strided 各覆盖；
- 输出元素数 = X 的元素数（`expected_output_elements` 新分支）。

### 4.3 容差（**必须真机标定后注册，不许先填数**）

不从 LayerNorm v2 借 `atol=1e-5`。标定矩阵（H20，同 torch/triton）：

| 臂 | 用途 |
|---|---|
| good：fp32 统计的融合核 | 必须全 case 通过 |
| alt：fp16 加法路径 / 不同归约顺序 | 记录误差分布；决定容差是否放它 |
| bad：漏加 R / 错 eps / 错方差（减 1） | 必须被拒绝 |

`rtol=1e-2`、`rho=1.0` 沿用受信任比较器的 AND 口径。**2026-11-17 GPU7 标定结果**
（`logs/fused-calib-20261117-{c,d,e,f}/`，真机 L2：32 条套件 case + 主用例）：

| 变体 | atol | L2 | max_abs | bad_elements | matched |
|---|---:|---|---:|---:|---:|
| good（fp32 统计） | 5e-3 | 通过 | 3.90625e-3 | 0 | 1.0 |
| alt（fp16 加法路径） | 5e-3 | 通过 | 3.90625e-3 | 0 | 1.0 |
| bad（漏加 R） | 5e-3 | 拒绝 | 4.53 | 1.99 亿 | 0.88% |

⇒ 注册 **atol=5e-3**：最小的圆整值，同时放行 good/alt、拒绝 bad。它比 layernorm v2 的
1e-5 宽，原因是参考的 fp16 加法在近零输出上产生固有 3.9e-3 差异；不是把 v2 的值挪过来。
标定不成立 ⇒ profile 不注册、任务判 `unknown`，不许先跑头寸再补容差。

## 5. 噪声探针通道

- 新 preflight op：`--op fused_layernorm_residual --shape 32768,6144 --dtype fp16`。
- 探针 kernel 与候选同通道：读 X、R → 末轴 LN → 写 Y，必须自己写 Triton
  （远端 `shield_torch()` 会拦原生算子）。
- **不复用** LayerNorm 的 channel：输入从 1 份变 2 份，带宽结构不同；
  按 [任务族选型 §3](../design/v08/task-family-selection.md) 逐任务登记窗口。

## 6. 头寸试验

- **fused candidate**：手写 Triton 单核（源码 `benchmarks/experiments/v08/kernels/fused_ln_residual.py`，
  哈希进冻结件），跑完整 L2（32 case + 主用例）后再做 `purpose=final`、`stop_after=L3` 的独立终评。
- **eager baseline**：就是受信任参考本身，不是第二个候选；它由同一次 L3 在**同一窗口**
  测出（receipt 的 `baseline_type=pytorch_eager` 与 `baseline_time_ms`），与候选的
  `median_time_ms` 配对。这样不需要一个"候选壳"去调用 native 算子（那会被反作弊屏蔽，
  也不是我们要测的东西）。
- 主指标：`diff_pct = (baseline_time_ms − candidate_median_ms) / baseline_time_ms × 100`（fused 更快为正）。
- 门槛：`≥ effect_threshold(2.5%)` 且 `> band`（当窗 preflight 实测）⇒ `headroom_confirmed`；
  `within band` ⇒ `no_increment`；`fused 更慢超带` ⇒ `fused_hurts`。
- 附报（只作上下文，不改判）：绝对加速比、launch 地板占比、`measurement_validity`、
  两臂源码哈希与当窗 preflight。
- 终止：探针 `action != start` ⇒ hold；任一臂 L2 不过或 L3 非 `valid` ⇒ 该臂出局，
  整轮 `unknown`，不修核凑读数；campaign 墙钟超预算 ⇒ hold。

## 7. 冻结身份与预算（2026-11-17）

| 项 | 值 |
|---|---|
| 任务声明 | `fp16 32768x6144 residual add layernorm`，shapes `[[32768,6144]]` |
| profile | `builtin-fused-layernorm-residual-fp16-cuda-v1` |
| 参考 digest | `c9aea6ac22afb4edc6a05c85b9fd3a9b9b9b70d2e2b1a552c4cf9bdffe8792cf` |
| 套件 digest | `652f65b16440864795cf60bd75bd3e93146dd26dfce3b0e6b6d2ccf195cf4aa7`（32 case） |
| 标定 atol | `5e-3`（§4.3） |
| 契约 hash | `88f248ae4725b0e2…` |
| 候选源码 hash | `500c84d0fb50c386…`（`kernels/fused_ln_residual.py`） |
| 探针 channel | `fused-layernorm-residual-32768x6144-fp16-cuda-event` |
| 测量卡 | GPU7，`GPU-52ca6155-d8bc-a416-847a-6f0c71db1552` |
| 计划 hash | `d0497326a4d86c55…`（`logs/fusion-headroom-plan-20261117.json`） |

- 预算：**模型调用 0 次**；L2 ×1（探索）+ 独立 L3 ×1（终评，内含 L2 前置）、当窗探针 ×2；
  campaign ≤ 2400 s。
- 证据目录：`logs/fusion-headroom-20261117/`（工作区日志，不纳入 Git）。

## 8. 决策状态（2026-11-17 已定）

| 项 | 定值 |
|---|---|
| 形状 | `32768×6144`（用户确认；与 P2 同形状，访存受限） |
| 参考加法精度 | fp16 加法（用户确认；自然 eager 写法，基线真实） |
| 容差流程 | H20/GPU7 上先标定 good/alt/bad，再把 atol 写死进 registry（§4.3，已执行） |
| 测量卡 | GPU7（用户确认） |

## 9. 实施备注

- 两输入共 768 MiB，原 512 MiB 的传输上限会把合法任务挡在 L2 之外；`MAX_STORAGE_BYTES`
  与 `MAX_SEED_STORAGE_BYTES` 同步放宽到 1 GiB（传输安全门，不是判据；不改有效性/容差）。
- 新噪声通道 `--op fused_layernorm_residual`；探针读 X/R、末轴 LN、写 Y，与候选同通道。

## 10. 头寸结果（2026-11-17，GPU7）

计划 hash `d0497326…`、候选源码 `500c84d0…`、契约 `88f248ae…`。

| 项 | 值 |
|---|---:|
| fused 候选中位 | 0.331968 ms |
| eager 基线中位（同窗） | 0.739968 ms |
| 相对差 | **+55.14%**（fused 快 2.229×） |
| 当窗带宽（终评前） | 0.418209%（起始 0.490596%） |
| 终评有效性 | `valid`（launch 地板 5.75%，137 个配对样本，0 harness peer） |
| 完整 L2 | 32/32 |

判读 `headroom_confirmed`，`formal_gain_claim=false`。这是**手写融合核 vs eager 两 kernel**
的头寸门，不是插件交付读数；它只回答"值不值得做下一步 A/C 对照"。回执
`logs/fusion-headroom-20261117b/{receipt,judgement}.json`。

## 11. 边界与对外表述

- 结论限一条任务、一个形状、一份手写融合核、登记的 H20 卡与当窗噪声；无显著性、无 A/C 含义。
- 头寸通过**不等于**插件能交付这样的核；那要另做 A/C 对照（同源码、同预算、同目标，
  记录达标与否与达标耗时）。
- 不写「性能全面达标」「显著优于 DSH」；不把手写核读数归因于插件。
