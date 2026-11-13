# A3：attention 进 R2 预登记（2026-11-08）

| 项 | 值 |
|---|---|
| 计划 | `operator-v08-increment-r2-attention-v1`，**已冻结** |
| 计划哈希 | `2669865924bad093…` |
| 任务 | `attention-prefill-1024`：`fp16 attention prefill 1x16x1024x128`，Triton，`min_speedup=1.0` |
| 门控通道 | `attention-1x16x1024x128-fp16-nocausal-cuda-event`，带宽 `0.1%` |
| 预算 | 沿用 fp32-v2 同一套：2 对、30 次模型派发、8 次验证、2400 s、45 GPU 分钟 |
| 证据 | `output/v08-attention-20261108/preregister-attention-v1.json`；门控回执 `output/…/c5/gate-attention-20261108-nocausal.json` |

## 1. 为什么登记的是 nocausal

`prepare` 目前没有把 `causal` 提成可声明的输入（标量默认 `false`），所以**可注册的主用例就是
`causal=False`**；`causal=True` 只出现在 8 条正确性 case 里。门控必须量登记的那个任务，
于是同一张卡上先后量了两条：

| 探针 | 中位 | 判决带宽 | 用途 |
|---|---:|---:|---|
| `--causal` | 0.097280 ms | 0.1%（实测散布 0.0%） | 只作对照，**不进计划** |
| 无 `--causal` | 0.096224 ms | 0.1%（实测 0.0998%） | **登记用** |

两条都在 `action=start`、`predicted_tier=valid`，且都与同窗 L3 的 `0.09648 ms` 对得上（差 0.3–0.8%）。
把 causal 那条当登记依据会犯 fp32 计划当年犯的同一类错：门的读数来自**另一个任务**。

## 2. 校验器同时收紧了三格

原来只绑 dtype（fp32 计划继承 fp16 探针那次的教训）。本轮把同一条纪律扩到算子、尺寸与 mask：

| 新拦下的情况 | 消息 |
|---|---|
| 任务声明是 attention，门控命令探 matmul（`--op` 缺省即 matmul） | `is 'attention' but the gate command probes 'matmul'` |
| 命令探的尺寸不是声明的形状 | `declares [[1, 16, 1024, 128]] but the gate command probes [...]` |
| 登记 `mask=nocausal` 但命令带 `--causal` / 反了 | `registers mask=... but the gate command disagrees about --causal` |
| 通道名没说 mask 形态 | `the gate channel ... does not name mask=nocausal` |

算子的解析走**同一个** `build_task`（不另立映射表）。两份已冻结的老计划（fp16、fp32-v2）
继续逐条通过——新增校验只收紧新计划，不改它们的读数。

## 3. 任务为什么值得选

| 证据 | 值 | 来源与性质 |
|---|---|---|
| 余量 | 39.8% | 100 − 60.2% 达成率；**粗估模型** `attention_coarse_v1`，不是实测峰值 |
| 基线差距 | 0.8826× | 一个**正确**的 triton 核只到 SDPA 的 0.88×；读数档位 `degraded`，故只作选题依据，不作性能主张 |
| 门控带宽 | 0.1% | 2.5% 登记效应 = 25 倍余量 |

三条都写进了计划里的 `why` / `measured_baseline_gap` / `notes`，冻结后不可改。

## 4. 这份计划不能得出的结论（继承 + 新增）

- 两臂都用插件，只差 optimize：比的是增量，不是"插件优于普通 Agent"。
- `sampling_comparability` 仍 `not_established`：只报配对观测。
- **新增**：attention 的达成率来自粗估模型（忽略 softmax 与中间搬运），`reference_lower_bound=false`，
  天花板出口关闭；8 条 case 是**正确性**范围，不是 8 个性能 case —— L3 只测主用例，
  且没有声明 `case_weights` / `max_regression`。

## 5. 开跑前还缺

1. 一份 `valid`（非 `degraded`）的 L3：S=1024 的 launch 地板占 17.3%，换 S=4096 主用例才会掉到 10% 以下。
   计划本身允许 `degraded` 对不成立——连续两对非 `valid` 就记 `unknown` 收工。
2. 两臂接线（A 臂同等预算硬门 + 统一独立终评）仍没做，这一条与 attention 无关，是所有 R2 任务共同的前置。