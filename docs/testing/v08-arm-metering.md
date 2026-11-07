# 两臂模型计量对账（S4）

2026-11-07 · v0.8 S4

八格对照的最后结论里有一条一直没变：`sampling_comparability=not_established`、费用与人工分钟
`unknown`。原因不是没人记账，而是**两臂记的不是同一件事**：B 有插件的前置预留账本，
A 只有 provider 看到的那份。这条命令把两边读进来，逐项说清哪里可比、哪里不可比。

## 1. 怎么跑

```bash
python -m operator_agent usage-reconcile \
    --run-dir /path/to/B-run \
    --a /path/to/arm-a.json \
    --json-out /path/to/metering.json
```

`--run-dir` 需要两个文件（都由插件在任务绑定时写）：

| 文件 | 提供什么 |
|---|---|
| `host-task-policy.json` | 冻结预算：`limits.wall_clock_s`、`max_model_calls`、`max_output_tokens_per_call`、`total_output_tokens_reservations` |
| `host-model-budget.json` | 逐次派发记录：`provider`、`model`、`max_output_tokens_reserved`、`status`、`usage{inputTokens,outputTokens,totalTokens,…}`、`reserved_at`/`finished_at` |

## 2. A 臂日志契约

一个 JSON 文档，一次派发一行。**逐条记，不聚合**——聚合过的日志无法对账。

```json
{
  "arm": "A",
  "source": "provider proxy access log",
  "metering": "provider_proxy",
  "budget": {"max_calls": 30, "max_output_tokens_per_call": 32768, "wall_clock_s": 2400},
  "dispatches": [
    {"status": "answered", "provider": "aliyun", "model": "deepseek-v3",
     "temperature": 0.1, "n": 1, "max_output_tokens": 32768, "enable_thinking": true,
     "reserved_at": 1762400000000, "finished_at": 1762400012000,
     "usage": {"inputTokens": 8123, "outputTokens": 2044, "totalTokens": 10167}},
    {"status": "failed", "provider": "aliyun", "model": "deepseek-v3",
     "temperature": 0.1, "n": 1, "max_output_tokens": 32768, "enable_thinking": true,
     "usage": null}
  ]
}
```

规则：

- `status` ∈ `reserved / answered / failed / cancelled / abandoned`。**失败的调用也要记**，
  否则两臂的派发数不可比。
- `usage` 为 `null` 表示 provider 没给计数 —— 那是**未知**，不是 0。任何一次未知都会让
  这一臂的 token 合计变成 `null`（半截求和长得像总量，是对账最容易引用错的地方）。
- `usage.outputTokens` 必须存在；各计数器必须是非负整数。
- 采样字段能记就记。记不到的字段会在回执里以 `unrecorded_sampling` 点名，**不会被当成相等**。

## 3. 回执怎么读

| 字段 | 含义 |
|---|---|
| `metering_comparable` | 两臂是否计的是同一批派发（用量已知、无未结算、无超预留） |
| `sampling_comparability` | `established` 仅当采样身份**逐项已记录且相等** |
| `checks` | 逐项布尔 + 明细，`blockers` 是非通过项的可读摘要 |
| `unrecorded_sampling` / `mismatched_sampling` | 没记的、记了但不一致的字段 |
| `arms.<A/B>` | 派发数、状态分布、用量已知/未知、token 合计、预算、窗口、违规 |
| `cost` | 恒为 `unknown`：token 计数不是发票，本命令不推断价格 |

## 4. 它证明什么、不证明什么

- **证明**：两臂各自的派发是否被完整计量、预算定义是否相同、采样字段是否一致。
- **不证明**：覆盖完整（provider 日志可能漏掉绕过它的派发；插件账本只覆盖成功 prepare
  之后的主会话原生派发 —— 这是 `coverage=declared`）。也不证明任何一臂的**结果**更好。
- **不做**：不从 token 推算费用，不把「两边数字接近」升级成「同口径」。

## 5. 现在会拦下什么

按当前插件的账本形状，跑真任务会得到 `not_established`，`blockers` 里点名：

- `sampling.recorded: temperature: A=0.1 B=None; n: A=1 B=None; enable_thinking: A=True B=None`
  —— 原生派发路径上插件看不到这三个字段，所以没记。
- `budget.max_model_calls: A=None vs B=30` —— A 臂还没有同等的调用硬门（历史不对称）。

两条都必须先在产物里成立，`sampling_comparability` 才可能变成 `established`。
把 `temperature/n/enable_thinking` 记进账本属于插件侧改动（工具契约会从 `12` 推到 `13`），
本条命令只负责判定，不负责放宽。