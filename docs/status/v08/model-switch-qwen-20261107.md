# 换模型：qwen3.8-2.4t-a95b 的实测与配置（2026-11-07）

## 1. 端点与可用性

| 项 | 值 |
|---|---|
| 端点 | `https://llm-4lvki6ksuwa2fuhf.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` |
| 模型 id | `qwen3.8-2.4t-a95b`（在该端点 182 个模型里确实存在）|
| responses API | ✅ 200 |
| chat/completions | ✅ 200 |
| 工具调用 | ✅ 返回 `tool_calls`，`arguments` 是合法 JSON |
| 流式 | ✅ `chat.completion.chunk` 正常 |

## 2. 一条硬约束：**它强制思考**

```http
POST /chat/completions  {"model":"qwen3.8-2.4t-a95b", ..., "enable_thinking": false}
→ 400 invalid_parameter_error
  "The value of the enable_thinking parameter is restricted to True."
```

后果：**reasoning token 永远计入 output**。实测一次 `max_tokens=16` 的极简请求，
`completion_tokens=21`，其中 `reasoning_tokens=18`。

对 host 模式的影响：

1. 插件的 `max_output_tokens_per_call`（默认 8192）要**按"思考 + 正文"合计**留量，
   否则会出现"想完了但没写代码"的调用；
2. 账本里的 `outputTokens` 会把思考算进去，`max_output_tokens_reserved` 预留不足会触发
   `provider_output_cap_violation`，之后该任务会被拒绝继续派发；
3. 因此用这个模型跑系列时，预算必须重新冻结（单次 8192 → 建议 16384 起，累计预留同比放大）。

## 3. 已做的配置

两个 profile 的 `aliyun` provider 下都加了该模型（与现在能用的 `deepseek-v4.1-flash-1017`
同一个 key 路径，`api: openai-responses`）：

| profile | aliyun 模型列表 |
|---|---|
| `desktop`（臂 B） | `deepseek-v4.1-flash-1017`、**`qwen3.8-2.4t-a95b`** |
| `v08-arm-c`（臂 C） | 同上 |

改动前都留了 `.yml.bak-qwen-<时间戳>` 备份；两份 YAML 均已解析验证。

`contextWindow` 填的是 **262144**，这是保守取值、不是端点声明的值 —— 第一次真跑时要观察是否会因上下文过长被拒。

## 4. 对 R2 的影响

- **R2 不能中途换模型**：两臂必须同一个模型，否则差值里混着模型差异。
- 换模型 = 新系列、新计划编号（与换任务同一条纪律）。
- 账本会记 `provider` / `model`，所以"这次用的是 qwen"可核；但 `temperature` / `n` /
  `enable_thinking` 仍不入账，`sampling_comparability` 依旧不会自动成立。

## 5. 下一步要它做什么

这个模型值得用在 **R3**（同任务、不同强弱的模型，看"开/不开 optimize"的增益是否随模型变弱而上升），
而不是 R2。R2 保持 deepseek，先把"管道 + 任务选择"这两件事收干净。