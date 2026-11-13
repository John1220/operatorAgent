# R2 预登记：v0.7.2 对 v0.8 的增量对照

2026-11-07 · v0.8 · FP16 计划 `operator-v08-increment-r2` 与 FP32 计划
`operator-v08-increment-r2-fp32-v2`。当前执行步骤见 [R2 跑法](v08-r2-runbook.md)。

本文件是**跑之前**冻结的判定规则，配合 [r2_plan.py](../../benchmarks/experiments/v08/r2_plan.py) 使用。
R2 回答的是 [v08-preparation §2](../design/v08-preparation.md#L23) 表格第二行那个问题：
同样的冻结预算下，加上"瓶颈诊断 + 候选比较"能不能让**终稿性能**更好。

它**不是**"插件比普通 DSH 强"的证明（那是 R1，已降级为副产物）。

## 1. 冻结了什么

| 项 | 冻结值 | 依据 |
|---|---|---|
| 两臂 | B = v0.7.2 封版工件；C = 同宿主 + 启用 `operator_context(phase=optimize)` | 只差 optimize 阶段，比的是增量不是工作流 |
| 任务 | FP16 `matmul-4096` 与 FP32 `matmul-4096-fp32` 各自登记，均为 Triton 4096³ | FP32 试测余量更大，见 [候选余量](../status/v08/candidate-headroom-20261107.md) |
| 噪声门 | 两份计划各自指定目标精度的 CUDA event 探针。每对开跑前重测 | FP16 初测带宽 0.203%，FP32 初测 0.10% |
| 最小可检效应 | **2.5%** | 分别为初测带宽的 12.3 倍和 25 倍 |
| 主指标 | 终稿加速比的**配对差值**，仅限终评 L3 为 `valid` 的配对 | `degraded` 读数只作观测 |
| 系列停止 | 连续 **2** 对没有 `valid` ⇒ 记 `unknown`，不启动第三对 | 两对都不可判是系统性窗口问题 |
| 终评预留 | 每对两臂各做一次独立终评，各自绑定终稿源码哈希 | 验证预算里保留终评额度 |
| 预算 | 已冻结。每系列 2 对，宿主的每臂预算见 [R2 跑法](v08-r2-runbook.md) | 新计划须填全预算后冻结 |

## 2. 两个容易混的数字

| 名字 | 是什么 | 用在哪 |
|---|---|---|
| `min_speedup`（契约里，如 1.0） | 任务**要求**达到多少 | 判定候选达没达标 |
| 最小可检效应（预登记里，2.5%） | 这次实验**能看出**多大差别 | 判定这次配对值不值得跑 |

写这份文件时我第一版填的是 2.0%，被 `validate()` 直接拦下——因为
`0.203% × 10 = 2.03%`，2.0 不够。**规则先于数字生效，这是好事**，所以最终登记 2.5%。

## 3. 怎么冻结

```bash
python -c "
from benchmarks.experiments.v08 import r2_plan
r2_plan.freeze(dict(pairs=2, max_model_calls=12, max_output_tokens_per_call=8192,
                    total_output_token_reservations=98304, max_verifications=6,
                    wall_clock_s=2400, gpu_minutes=60), '/path/outside/repo/preregister.json')
"
```

写盘用 `open('x')`：**不允许覆盖**已冻结的计划。要改判据只能新开计划、新编号，
和 [v0.7.2 的"不得原位改判据"](../../docs/status/v072-todo.md) 同一条纪律。

`freeze(..., task_id=...)` 选计划：缺省 `matmul-4096`（fp16，编号 `operator-v08-increment-r2`），
`matmul-4096-fp32` 给 fp32（编号 `operator-v08-increment-r2-fp32-v2`）。门控与可判性随
`task_id` 取该任务自己那次探针的读数，不由另一个精度继承。

## 4. 它禁止什么

`validate()` 对以下情况直接拒绝：

- `formal_gain_claim` 或 `formal_comparison_credit` 为真（预登记不能自带结论）；
- 计划里出现 `outcome` 字段（跑完才能有结果）；
- 冻结时预算有空项；
- 最小可检效应没有越过 `窗口带宽 × 10`；
- 系列停止规则不是"连续 2 对无 valid"；
- 内容与 `hash` 不符（改一个数就会被发现）；
- 门控命令探的 dtype 与任务声明的 `precision` 不一致（`--dtype` 缺省即 fp16，
  "命令里没写"不等于"跟着任务走"）；
- `detectability.measured_band_pct` 与 `measurement_gate.observed_band_pct` 不相等
  （同一个带宽存了两份，重新冻结时只改一份就会自相矛盾）；
- 计划编号在 `WITHDRAWN_PLAN_IDS` 里（指名拒绝并给出撤回理由，而不是落进"编号不认识"那条通用消息）。

后三条是 2026-11-07 审查补的：fp32 首版计划登记了 fp32 任务，却配一条探 fp16 的门控命令、
沿用 fp16 的 0.203% 带宽，而当时的校验器全部放行。门控与可判性现在**按任务取值**
（`PROBES`），每个任务的 command / channel / band / window 都来自它自己那次探针回执。

## 5. 每对怎么跑

1. **门**：`preflight` 落 `hold` ⇒ 这一对不启动，不计失败、不消耗模型/验证预算，等下一个窗口。
2. **两臂**：同宿主、同模型、同冻结预算；顺序交替（B→C、C→B），避免把漂移算到某一臂头上。
3. **终评**：两臂各跑一次独立终评，绑定各自提交源码的哈希。任一臂 L2 不全过或 L3 非 `valid`，这一对记为不可判。
4. **停**：连续 2 对不可判 ⇒ 记 `unknown` 收工。

## 6. 这条系列不能得出的结论

写在计划里，也写在这里：

- 两臂都用插件，**不能**据此说"插件优于普通 Agent"。
- `sampling_comparability` 仍是 `not_established`（A 臂无同等硬门、账本未记
  `temperature/n/enable_thinking`），所以只报**配对观测**，不报绝对收益。
- 任务没拿到 `valid` 终评时，那是关于**窗口**的结论，不是关于候选的结论。

## 7. 两臂的执行环境（2026-11-07 核实）

两臂必须用**各自独立安装**，否则来回切换工作树会让身份漂移：

| | 臂 B | 臂 C |
|---|---|---|
| 源码 | `git worktree` @ tag `v0.7.2`（`$HOME/op-agent-delivery/v072-arm-b`） | 当前分支 `codex/v0.8` 工作树 |
| Python | `$HOME/op-agent-delivery/v072-venv/bin/operator-agent*` | `$HOME/op-agent-delivery/v08-venv/bin/operator-agent*` |
| `TOOL_VERSION` | 12 | **12**（工具面相同，这是两臂可比的前提） |
| `decide.noise_preflight` | 无 | 有 |
| `generate.optimize_diagnoser` | 无 | 有 |

实际 DSH profile 的入口与启动前核对项见 [R2 跑法](v08-r2-runbook.md#1-环境)。以会话实际
生效的 MCP 入口为准，`desktop` 的 patch 里有旧入口和后写生效的 v0.7.2 入口。

## 8. 现状

- 计划模板与校验器已完成（[r2_plan.py](../../benchmarks/experiments/v08/r2_plan.py)），13 个用例钉住上述规则。
- FP16 已完成一对流程试跑。FP32 首版在派发前撤回，v2 已冻结，待首对原生会话。
- **attention 计划 `operator-v08-increment-r2-attention-v2` 已冻结**（hash `756f17d44e148788…`），
  任务 `attention-prefill-4096`、门控通道 `attention-1x16x4096x128-fp16-nocausal-cuda-event`、带宽 `0.1%`；
  登记的是 nocausal 主用例（`prepare` 目前没有 causal 输入）。
  **v1（S=1024）已按名字撤回**：那个核只有 96 µs、launch 地板占 17.3% ⇒ 每次配对结构性不可判。
  见 [A3](../status/v08/a3-attention-r2-preregistration-20261108.md)。
- 噪声门的实现见 [v08-noise-preflight.md](v08-noise-preflight.md)；本窗口实测见
  [真机回执记录](../status/v08/real-receipt-20261107.md)。
