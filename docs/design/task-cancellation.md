# 任务取消与中断（S2）

工具契约从 6 升为 **7**。取消面向一次具体的 `run_dir + invocation_id`；后台自主运行和前台 verify/repair/bench 都有独立实例。判定、资格、入库门和默认生成策略不变。

## 使用

后台 `operator_run` 回包立即包含 `invocation_id`。同步 MCP 长调用可发送 `notifications/cancelled`，其中 `requestId` 必须与正在执行的 JSON-RPC 请求 ID 相同；通知自身没有响应，原请求最终返回取消错误或更早完成的真实结果。

也可显式调用：

```json
{"name":"operator_cancel","arguments":{"run_dir":"/tmp/my-run","invocation_id":"0123456789abcdef0123456789abcdef"}}
```

随后调用 `operator_status(run_dir, invocation_id)` 查看终态。未提供实例且目录没有自主运行状态时，status 列出该目录中的工具调用实例。

```sh
python -m operator_agent cancel --run-dir /tmp/my-run --invocation-id <id>
python -m operator_agent status --run-dir /tmp/my-run --invocation-id <id>
```

DSH 支持 `/op cancel <run_dir> <id>`、`/op status <run_dir> [id]`。含空格的目录可以加引号。CLI verify/repair 收到 Ctrl-C 或 SIGTERM 时请求取消、等待本机清理、输出 JSON，取消退出码为 **130**。直接 import `hitl_tools.verify/repair/bench` 仍是同步库调用；中断保证由 CLI/MCP 的监督适配器提供。

`accepted=true` 只表示取消请求已留存，不表示执行已停止。前台工具的实例和路径也会加入最终回包；调用尚未返回时，可通过 status 或 `tool-calls/` 下的状态文件发现实例。

## 状态与竞争

| 情况 | 行为 |
|---|---|
| 启动后立即取消 | 启动前即建立不可覆盖的实例身份；监督进程收到早期请求后不启动 worker |
| 运行中取消 | 拥有 Popen 的监督进程清理本机 worker 和已发现的进程组，写 `phase=cancelled`、`stopped_reason=user_cancelled` |
| 重复取消 | 复用第一次请求及其时间，不重置预算、不创建新任务 |
| 已完成或失败 | `accepted=false`、`reason=already_terminal`，保留原结果 |
| 取消与完成/失败竞争 | 比较共享单调时钟；更早写下的有效 worker 结果保留，否则取消胜出 |
| 取消与墙钟截止竞争 | 采用更早的控制事件；截止仍为既有预算停止语义 |
| owner 消失或 PID 已复用 | 返回 `lost` / `owner_unavailable`，清理保持未知，不按客户端 PID 发信号 |
| 身份错误或无身份的历史目录 | 明确拒绝；不把旧目录或进程追认为当前任务 |

`done` 表示工具/流程结束，仍须查看其判定和停止原因。被取消的任务不因部分源码、报告或原始结果已落盘而取得成功或入库资格。更早完成的独立验证事实可以保留；未完成步骤不补造结果或 `complete.json`。

取消不会回滚已经完成的外部动作或语料写入；它负责停止本次执行、保存事实，不新授予正确性或入库资格。

## 进程与文件所有权

取消 API 只原子发布请求，**不接收任意 PID 并杀进程**。监督者只操作自己持有的 Popen 及已观察到、启动身份仍匹配的进程组。观察过的 PGID 如已属于另一个启动身份，跳过并记录 `stale_groups_skipped`。发现或信号/回收失败记录 `cleanup_unknown`；`all_descendants_proven=false` 不变，快速脱离且从未观察到的进程不在已证实范围内。

MCP 读入线程只读取协议行，主循环处理状态/取消并轮询 worker；库的 stdout 重定向局限在工作进程内，不与协议输出线程争用。正常结束、取消和宿主 SIGINT/SIGTERM 都经过监督清理。不可捕获的 SIGKILL、机器故障或文件系统写入失败可能留下 `lost`；不能据此声称资源已回收。

后台任务使用：

```text
<run>/
  launch.json           # 不可覆盖的 invocation 身份，先于异步启动
  supervisor.json       # launcher 记录的监督进程 PID/启动身份
  cancel-request.json   # 第一次取消请求，原子创建，不覆盖
  autonomous.json
  run-status.json       # 监督进程发布的状态
  worker-result.json    # worker 自己的结果，按 invocation/完成时点核对
```

前台长调用使用 `<run>/tool-calls/<id>/`，其中保存 launch、请求、独立执行日志、worker 结果、最终回包和状态。请求包含用户提交的源码或文件引用；凭据值仍通过环境/既有凭据引用提供，不加入请求。S1 的候选与逐轮验证目录保持独立。

请求及完成顺序使用 `task_clock.monotonic()`；POSIX 显式读取系统 `CLOCK_MONOTONIC`，避免 macOS Python 3.9 的进程独立起点。生产任务预算和模型调用截止采用同一时钟。严格单候选实验的进程内计时口径保持。

## 预算与外部执行

取消不会删除、修改或退还派发前的模型/验证预留。终态保留取消时可得的账本、部分产物路径和清理回执；缺失账本保持未知，不补零。前台 HITL 工具没有因此获得一份虚构的自主任务总账本。

本机 worker 回收与远端/服务商停止是不同事实。自 2026-10-30 起，CLI/MCP 与自主运行的 **persistent SSH** 默认在 worker 退出前关闭本次拥有的会话：远端关闭共用最多 10 秒，监督者在取消/总时限到达时给 worker 最多 12 秒清理机会，再进入既有本机终止流程。无需 S4 专用环境开关。只有所有已启动会话握手完整且各自的精确进程身份清理得到静默回执，`cleanup.remote_execution` 才为 `quiescent`；启动身份不完整、清理失败或超时仍为 `unknown`。一次性 SSH、其它后端、SIGKILL 和机器故障不获得该保证。

`cleanup.provider_call=unknown`、`all_descendants_proven=false` 与自主预算的 `remote_cancellation_proven=false` 继续保留：本次持有的远端 runner 退出不能证明服务商停止推理/计费，也不能证明任意子孙进程都已停止。只杀本机 SSH/HTTP 进程仍不构成远端静默证据。GPU2 的真实执行中取消、RPC 超时及两次后续完整 L2 验收见[中断与恢复验收](../status/gpu-interrupt-recovery-20261030.md)。S2 的历史离线结果不被升级为 GPU 证据。

不实现暂停/恢复、取消后的自动重试或任务目录复用。重新运行使用新任务目录；历史产物保留。Q1 外内层对照的暂停决定仍按 [HITL 边界 §7](hitl-generation-boundary.md#7-我无法确立的事) 执行。
