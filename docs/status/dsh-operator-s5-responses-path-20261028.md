# S5：DSH 默认 Responses 路径经预算网关（2026-10-28）

状态：**默认 Responses 协议已接到预算网关；一次 256 输出上限的真实短请求通过。仍不是正式 S5、三组对照或隔离验收。**

## 为什么需要这一步

此前网关只接受 `/v1/chat/completions`，而 DSH 默认 `aliyun` 提供方使用 OpenAI **Responses**（`/v1/responses`）。已完成的真实预检走的是配置里的 Completions 兼容路径，不能代表默认路径。本轮把 Responses 接入同一账本与同一监督传输，并让真实 DSH agent（而非仅 `LlmRuntime`）走通。

## 改动

- `benchmarks/s5_gateway_policy.py`：新增 `protocol='responses'`，只接受响应体和 agent 实际发送的字段（`model/input/max_output_tokens/stream/store/tools/tool_choice/temperature/instructions/prompt_cache_key`），仍要求显式输出上限、`store=false`、单次 `n` 语义，未知字段直接拒绝。
- `benchmarks/s5_gateway_transport.py`：允许 `/v1/responses` 与 `/compatible-mode/v1/responses`；流式校验改为“最后一个事件必须是 `response.completed` 且 `response.status=completed`”，`response.failed`/`response.incomplete`/截断一律 unknown；终止事件同样暂存到 EOF 后才放行。
- `benchmarks/s5_gateway_http.py`：按协议只开放对应路径，错误协议路径返回 400。
- `benchmarks/s5_fake_responses_upstream.py`：回环 Responses 假上游，可产出正常、截断、失败三种流，只记录请求哈希与有限字段。
- `benchmarks/s5_responses_turn.py`：一次 DSH agent 回合 → 网关 → 上游，先冻结计划（端点、模型、上限、墙钟、源码哈希）再执行一次；记录账本、上游成帧结论与清理状态。
- `benchmarks/s5_arm_env.py`：安装面支持 `api='responses'`、模型 ID 与输出上限参数化；三组旧路径不变。

`tests/test_s5_responses_gateway.py` 18 项；S5 全部专项测试 **246 项通过**（2026-10-28）。

## 真实短请求

冻结计划：端点 `https://llm-4lvki6ksuwa2fuhf.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/responses`，模型 `deepseek-v4.1-flash-1017`，监督传输，1 次调用，输出上限 256，墙钟 60 秒，固定提示 `Reply with exactly OK.`，无重试、无 GPU。

- 计划 SHA256 `f0bc44581e079797f8c70ae1ab16ef0a6034a24959c145e4a1ead46c1c06fcc9`；结果 `passed`，墙钟 1.959 秒。
- 账本 **1 次预留、1 次完成**，无中止；上游成帧结论 `answered`（完整 SSE 且终止事件合法）。
- 最终文本 `OK`（2 字节，SHA256 `565339bc4d33d72817b583024112eb7f5cdf3e5eef0252d6ec1b9c9a94e12bb3`）。
- provider usage 仍为 **unknown**：Responses 路径未解析原始 usage，账本保守记 unknown，不能当计费依据。
- 直接 worker 已回收（`direct_child_reaped_descendants_unknown`）；逃逸后代与“服务商已停止”仍为 unknown。
- 原始记录 `output/plugin_completion/20261028-s5-responses-real-v3/`（15 文件、约 47 KiB，Git 忽略）。扫描全部文件未发现凭据值；凭据只在受监督传输内存中使用，DSH 侧只有占位符。

## 两次本机失败（各 0 次模型调用）

真实调用前有两次尝试在**本地**失败，账本均为 0 预留，未产生服务商请求：

- v1：`arm_dir` 为相对路径，DSH 以工作目录解析覆盖文件导致读不到 `patch.json`；已改为绝对路径并加回归测试。
- v2：安装面把模型名写死为 `fake-model`，与计划的真实模型不一致，网关按设计以 429 拒绝；已把模型 ID 与输出上限参数化并加测试。

两次失败保留目录与记录，作为“计划不可重入 + 逐次新身份”的实际演练；v3 是唯一一次真实请求。

## 仍然未知

1. 三组仍跑假模型；用真实模型跑三组、每组预算与同任务对照尚未做。
2. Responses 的工具调用流（`function_call` 事件）与多轮反馈未经真实模型验证；本次请求虽带一个计划模式工具定义，但没有触发工具。
3. H20 统一验证入口未接；`h20` 后端仍返回 unknown。
4. 隔离未建立：A 组仍可读本仓，网关仍非“不可绕过”的唯一出口（同机可直连服务商）。
5. usage/计费：Responses 原始 usage 未解析，token 与金额都未知。