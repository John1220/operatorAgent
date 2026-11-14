# v0.8 A/C 工作流轮：DSH 与 DSH+插件开发算子的对照报告

**日期**：2026-11-09 ～ 2026-11-10
**任务**：同一个 kernel 开发任务，两条路径各跑一遍，看**最终源码能否通过独立终评**
**结论一句话**：强模型上两臂打平（3–3）；**弱模型上差别可复现**——预登记 2 轮、每轮配对差均 +1
（A 1/3 与 2/3，C 2/3 与 3/3），按预设规则判 **`reproduced`**、`formal_gain_claim=true`。
插件臂交付只用 **3–9 次**模型调用，裸臂每格都用满 20 次。墙钟方向不一致（§6b），
性能与"开发人时"均不在结论范围内。

---

## 1. 两条臂是什么

| 臂 | 宿主能看到的工具 | 拿到规格的途径 |
|---|---|---|
| **A（裸 DSH）** | 普通 shell / 读写 / 编辑 + `check.py`（stdlib 客户端，提交走 supervisor） | 只有 prompt 里那几句：任务声明、ABI、数值契约、规则、预算 |
| **C（DSH + 插件）** | 普通工具 **关掉**，只有插件提供的 7 个 MCP 工具（`operator_context` / `operator_verify` / …） | `operator_context` 一次性发布：ABI、数值契约、判据、被测布局、规则、候选 suite 清单 |

两臂**共用同一个 supervisor 判据**：同一份冻结契约、同一个 L2 套件、同一份 H20、
同一次"独立终评"（`verification_purpose=final`）。

## 2. 判定标准

- **主指标**：该格的最终源码在独立终评里**通过完整 L2**（含派生 suite）算"交付"；
- **副指标**：达到交付所花的**模型调用次数**；
- **预算**：每格 20 次模型调用、每次预留 40960 输出 token、墙钟 2400 s、验证 8 次、终评预留 1 次；
- **采样身份**：`model / temperature / n / max_tokens / enable_thinking / reasoning_effort`
  六字段必须两臂逐字一致 —— 每条都能从 `cell/outer-requests/*.json` 里核对出来。

---

## 3. 结果一：强模型（deepseek-v4.1-flash-1017，reasoning_effort=medium）

| 任务 | A（裸 DSH） | C（DSH+插件） |
|---|---|---|
| matmul-4096-fp32 ¹ | ✗ L2 `1.16e-03`（用了 tf32x3） | ✓ 交付（13 次调用） |
| attention-prefill-4096 | ✓ 交付（20 次） | ✓ 交付（10 次） |
| softmax-16384x6144-fp16 | ✓ 交付（20 次） | ✓ 交付（5 次） |
| matmul-2048x16384x2048-fp16 | ✓ 交付（20 次） | ✓ 交付（18 次） |
| **合计** | **3/4** | **4/4** |

¹ 这一对用的是 `qwen3.8-2.4t-a95b`（当时唯一有额度的模型），其余三格是 deepseek。
**按同模型的三格算：A 3/3、C 3/3 —— 打平。**

## 4. 结果二：弱模型（Qwen3.8-27B，xhigh）——**预登记复现轮**

单轮那次的 +3 不能直接当结论，所以按预登记跑了 **2 轮**（计划 v6，冻结 hash `42e01fca…`），
规则是"**每一轮**配对差都要 ≥ +1，任一轮掉 0 判 null 且不追加轮次"。

| 轮 | 任务 | A（裸 DSH） | C（DSH+插件） |
|---|---|---|---|
| r1 | matmul-4096 | ✓（20 次调用） | ✓（9 次） |
| r1 | attention | ✗ `s4096-causal-strided` 挂（20） | ✗ L2 里换 case 编译崩（5） |
| r1 | softmax | ✗ 主用例过、strided 挂（20） | ✓（3 次） |
| r2 | matmul-4096 | ✓（20） | ✓（6 次） |
| r2 | attention | ✓（20） | ✓（9 次） |
| r2 | softmax | ✗ strided 挂（20） | ✓（3 次） |
| | **合计** | **R1 1/3 · R2 2/3** | **R1 2/3 · R2 3/3** |

```
R1 配对差 +1 · R2 配对差 +1   （两轮 sampling_comparability = established）
ac_aggregate → verdict = reproduced, formal_gain_claim = true
范围：weak host (Qwen3.8-27B, xhigh) on tasks that one generation can deliver
```

### 三条必须一起读的限制

1. **单格不稳定，两臂都抖**：attention 在 A 是 ✗→✓、在 C 也是 ✗→✓；只有 softmax 稳定
   （C 2/2 交付、A 0/2 失败）。所以"每轮 ≥ +1"成立，但**没有一个格子可以当成稳定单点**。
2. **调用数差距比交付率更一致**：A 每格都用满 20 次；C 交付用 3–9 次。
3. **A 在 softmax 上两轮同一个死法**：主用例 `1.91e-06` 通过、`strided` case 挂——
   正好对应"插件给了 `tested_layouts` / 裸臂没被告知"这条机制。

---

## 5. 机制：插件到底提供了什么

A 臂的三种失败，正好一一对应插件"一次性说全"的三类事实：

| A 的失败 | 它缺的事实 | 插件的出处 |
|---|---|---|
| `CompilationError` / `Missing 'kernel' function` / 签名不符 | **ABI**：入口名、参数名、**按值直传的标量**（如 attention 的 `causal`） | `authoritative.interface` |
| `nan`、`1.16e-03`（tf32x3） | **数值契约**：`tf32:false`、`accum:fp32`、`atol`、AND 判据 | `authoritative.contract` |
| 主用例过、只有 strided case 挂 | **被测布局**：suite 里真的会构造非连续输入 | `suite_intent.tested_layouts` |
| 调 `torch.softmax` 被判反作弊 | **规则**：禁止用原生实现顶替算子 | `rules.anti_cheat` |

所以差距不是"插件让模型更聪明"，而是：**插件把规格从"模型得自己猜到/试出来"变成"一次给全"**。
弱模型猜错的概率高、试错的预算又少，差距因此被放大；强模型本来就能补齐这些事实，差距消失。

## 6. 七个仪器缺陷（每一个都曾伪装成"某一边能力不行"）

这一轮真正的成本不在跑实验，而在**判据链本身不可靠**。逐个修掉之前，每一次"哪边输了"
的读数都可能是假的：

| # | 缺陷 | 曾一度被读成 |
|---|---|---|
| 1 | 线上输出上限被压到 8192（应为 32768/预留 40960） | 模型太弱、写不完 |
| 2 | `--arm C` 实际跑的是 A 臂（CLI 分派写死） | 插件没用 |
| 3 | A 臂的 ABI 少传了标量 `causal` | 裸臂不会写 attention |
| 4 | 流式截止按 62.5 tok/s 算（实测 32–53 tok/s） | 服务太慢 |
| 5 | MCP 工具调用 60 s 超时（attention 的 L2 要数分钟） | 插件臂自己放弃 |
| 6 | 两臂采样身份不同（A `xhigh` / C `medium`，因为策略补丁只加在 A 的进程里） | 插件带来 +3 增益 |
| 7 | 网关硬顶每次输出预留 ≤ 40960，而计划登记了 65536 | "64k 的计划跑不起来"（其实是护栏在拒绝） |

第 6 条是**在写这份报告的过程中**被 `sampling_identity` 核对抓出来的——修完重跑复现轮，
+1 依然成立，但这次它有证据。第 7 条让 v5 计划的 12 格一格都没起来，表现为"模型不响应"。

此外还补了一条判定纪律：**仪器失败 ≠ 候选失败**。钉卡失败（`required GPU selection previously
failed`）、别人抢卡导致的 `CUDA out of memory`、墙钟超时，全部记 `outcome=unknown` 并重跑，
不计入任何一臂的胜负。

## 6b. 成本读数（`measured` 计数 + `recomputed` 墙钟）

### 复现轮（v6，2 轮 × 3 任务）

| 轮/任务 | A 调用 | C 调用 |
|---|---:|---:|
| r1 matmul-4096 | 20 | 9 |
| r1 attention | 20 | 5 |
| r1 softmax | 20 | 3 |
| r2 matmul-4096 | 20 | 6 |
| r2 attention | 20 | 9 |
| r2 softmax | 20 | 3 |
| **合计** | **120** | **35** |

A 每一格都顶到 20 次上限；C 交付用 3–9 次。**调用数差距（约 3.4×）比交付率差距更稳定。**


调用数与验证数来自盘上一一对应的记录（一次请求一份 `request-NNNN.json`、一次提交一份回执），
墙钟由回执 `created_at` 与请求文件时间戳相减得来。**A/C 未交替运行**，墙钟只作辅助。

**一处计量勘误（2026-11-13）**：v6r1-attention-C 失败格回执里 `model_dispatches=None`——
早退路径（无提交通过 L2）当时绕过了计量，上表该格的 5 次是从盘上 5 份 `request-NNNN.json` 数的。
冻结回执**不改**：轮哈希覆盖计量字段，改它会让 `reproduction-verdict.json` 引用的轮哈希失配
（已用原时间戳 + metering 复算 round-1 哈希 `fa3eab80…` 一致，链完整）。活路径已修：
两条早退（A/C）现在都过 `_counts()`，失败格也带 `measured` 计量与出处。

### 弱宿主（27B/xhigh）

| 任务 | 臂 | 结果 | 调用 | 验证 | 到首个完整 L2 |
|---|---|---|---:|---:|---:|
| matmul-4096-fp32 | A | ✗ failed | 20 | 2 | — |
| matmul-4096-fp32 | C | ✓ 交付 | **6** | 1 | 203s |
| attention-prefill-4096 | A | ✗ failed | 20 | 2 | — |
| attention-prefill-4096 | C | ✓ 交付 | **4** | 1 | 253s |
| softmax-16384x6144-fp16 | A | ✗ failed | 20 | 2 | — |
| softmax-16384x6144-fp16 | C | ✓ 交付 | **3** | 1 | 140s |
| matmul-2048x16384x2048-fp16 | A | ✓ 交付 | 20 | 2 | 226s |
| matmul-2048x16384x2048-fp16 | C | ✓ 交付 | **3** | 1 | 196s |

### 强模型（deepseek-v4.1-flash/medium）

| 任务 | 臂 | 结果 | 调用 | 验证 | 到首个完整 L2 |
|---|---|---|---:|---:|---:|
| attention-prefill-4096 | A | ✓ | 20 | 2 | 230s |
| attention-prefill-4096 | C | ✓ | **10** | 6 | 63s |
| softmax-16384x6144-fp16 | A | ✓ | 20 | 2 | 130s |
| softmax-16384x6144-fp16 | C | ✓ | **5** | 2 | 35s |
| matmul-2048x16384x2048-fp16 | A | ✓ | 20 | 2 | 92s |
| matmul-2048x16384x2048-fp16 | C | ✓ | 18 | 3 | **316s** |

### 这两张表能主张什么、不能主张什么

- **调用数**：C 一致地更少（弱宿主 3–6 vs 20；强模型 5–18 vs 20）。这是**实测**的，可以主张。
- **墙钟**：方向**不一致**。强模型上 attention 3.6× 更快、softmax 3.7× 更快，但 tall-K
  C 反而 3.4× 更慢（它交了 3 次、每次都要跑 L2，验证时间计入了墙钟）。
  所以只能说"墙钟在部分任务上更短"，不能统一说"更快"。
- **人时不在范围内**：开发者人时不可得、外部无同口径，任何"开发更快"的表述都不成立。
  能主张的是"**无人值守交付的调用数与墙钟**"。

## 7. 限制（不能外推的部分）

1. **单卡、格内 n=1**：复现共 2 轮 × 3 任务，每格每轮 n=1（更早的 attention 探点重复过 3 次）；
   判据是预登记的「每轮 ≥ +1 任务」这一最小可分辨单位，**不是统计显著性**，不给显著性结论；
2. **模型换了三次**：deepseek-v4.1-flash、qwen3.8-2.4t、Qwen3.8-27B（额度与可用性所迫），
   跨轮数字不能混算；同一轮内采样身份已核对一致；
3. **正式额度分两层**：`ac_compare` 的 `formal_comparison_credit` 恒 False——**单轮**观测不授予
   正式额度；只有 `ac_aggregate` 在全部轮次 ≥ 最小可分辨单位时判 `reproduced`，才授予
   `formal_gain_claim=formal_comparison_credit=true`。本轮结论在聚合层（范围见头条），
   单轮回执里的 False 不与它矛盾；
4. **性能未进比较**：交付的格子里 L3 有效性多为 `None`/`degraded`，所以本轮只谈"能不能交付"，
   不谈"快不快"；
5. **softmax / tall-K 的 suite 覆盖面**仍是声明范围，未证明任意输入域；
6. **单格不稳定**：复现轮里 attention 两臂都是 ✗→✓，所以"每轮 ≥ +1"成立但单点不可复现；
7. **共享机器**：钉卡在 2（个别格落到 3）且逐格记录；别人抢卡会造成 OOM——
   这类格按"仪器失败"记为 unknown 并重跑，不计胜负；
8. **没有测到 repair/optimize 闭环**：C 臂的交付都是 3–9 次一次成型，
   插件在"改错"与"调优"上的价值本轮未涉及。

## 8. 复现

```bash
# 一格的完整链路（A 臂 / C 臂）
.venv/bin/python -m benchmarks.experiments.v08.ac_live \
  --plan  output/v08-ac-20261108/preregister-ac-2.json \
  --task-id attention-prefill-4096 --arm C \
  --root   output/v08-ac-20261108/rep --json-out output/v08-ac-20261108/rep.json \
  --work-dir /root/operator-o8-a1-161201 \
  --ssh-host <h20-host> --ssh-port <ssh-port> --ssh-user <user> --ssh-key ~/.ssh/<key>

# 整轮判读
.venv/bin/python -c "from benchmarks.experiments.v08 import ac_compare; ..."   # 见 round-9.json
```

环境变量：`AC_MODEL_BASE_URL / AC_MODEL_ID / AC_MODEL_KEY_ENV`、`AC_REASONING_EFFORT`、
`AC_UPSTREAM_S_PER_1K`（按实测 tok/s 给，本地 vLLM ≈30、云端 ≈26）。

## 9. 证据索引

| 内容 | 路径 |
|---|---|
| 计划 v2（冻结，20 次/40960/819200） | `output/v08-ac-20261108/preregister-ac-2.json` |
| 强模型整轮 | `output/v08-ac-20261108/round-7.json` |
| **弱模型整轮（推荐引用）** | `output/v08-ac-20261108/round-9.json` |
| 采样身份核对（六字段） | `output/v08-ac-20261108/sampling-identity-weak.json` |
| attention 3 次重复 | `output/v08-ac-20261108/q27x-attn-A.json`、`q27x-attn2-A.json`、`q27x-attn3-A.json`（C 臂同名） |
| 逐格回执 | `output/v08-ac-20261108/` 下的 `cell-*`、`pair4-*`、`q27x-*`、`q27xf-*` 目录 |
| **复现判读（结论依据）** | `output/v08-ac-20261108/reproduction-verdict.json` |
| 复现轮每轮判读 | `output/v08-ac-20261108/v6-round-1.json`、`v6-round-2.json` |
| 复现计划 v6（冻结） | `output/v08-ac-20261108/preregister-ac-6.json` |
| 详细日志（本报告的依据） | `docs/status/v08/ac-workflow-round-todo-20261108.md` §2b–§3u |