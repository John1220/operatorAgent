# S5：单次真实模型网关预检（2026-10-28）

状态：**一次真实请求通过；不是正式 S5 对照或隔离验收。**

## 范围

经用户同意，固定一次调用，模型为 DSH 配置中的 `deepseek-v4.1-flash-1017`，输出上限 256 tokens，固定单条用户消息 `Reply with exactly OK.`，无 tools、无 GPU、无重试。请求从已安装的 DSH `LlmRuntime` / pi-ai 经本机预算网关和受监督进程发往配置的阿里云兼容端点；这不是完整 headless agent。

当前 DSH 默认 `aliyun` 使用 **OpenAI Responses**；此次使用配置中的 `aliyun-native` **Chat Completions** 兼容路径。两者在预检前只读核对为同一主机、模型 ID 和凭据引用，但协议不同；本次不证明默认 Responses 路径可用或两种协议等价。

凭据由短生命周期监督进程只读现有 `0600` DSH 凭据文件，传入网关进程内存及受监督 worker 的 stdin；DSH 客户端只使用占位凭据。计划不可重入，失败或超时也不重试。执行前对网关、兼容路径、DSH 客户端及预算的本机假上游回归 **68 项通过**。

## 结果

- 结果 `passed`，网关账本 **1 次预留、1 次完成记录**，上限 256 tokens；监督器观测到完整上游 SSE、DSH 完成事件和 2 字节响应，其 SHA256 与 `OK` 一致。墙钟 1.334 秒。共享账本本身按原网关保守策略仍记 provider outcome unknown，不因本次预检结果回写推断。
- 从完整上游 SSE 原始字段取得 usage：`prompt_tokens=35`、`completion_tokens=13`、`total_tokens=48`。原始响应只在内存中解析、未保存，落盘数字是预检监督器记录，不能从证据包独立重放验证；具体金额和计费口径未知，不能把 token 数直接当账单。
- 直接 worker 已回收（`direct_child_reaped_descendants_unknown`）；会后只读进程检查未发现该 worker 命令残留。逃逸后代和服务商侧是否停止仍为 unknown，不能由此推断系统隔离或完整进程树清理。
- [预检材料](../../output/plugin_completion/20261028-s5-real-gateway-preflight-v1/) 约 52 KiB，包含冻结计划、不可重入标记、共享账本与快照、客户端脚本和结果；被 Git 忽略，需单独备份。计划 SHA256 `ac3c1b306fa79a5bcf2ad3b6c147a188734532bbca7ad533e48515b3fe18a8eb`；结果 SHA256 `220b382527a32f6e2b8ef98fad9ea6e6f3c722cfeb7671f5c52e3c0bc3ad2429`。执行后扫描 6 个落盘文件，未发现本次凭据值；这只覆盖扫描的目录及该值。

## 限制

该预检只验证受控单次路径的兼容性，不证明 DSH 的默认 Responses 协议、完整 agent、网络/文件隔离、不可绕过的统一预算、精确账单或三组对照。没有进行算子生成、候选验证或性能测量。S5 和 S6 的状态不因这次预检改变。
