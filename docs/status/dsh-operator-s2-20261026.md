# 算子开发插件 S2：任务取消与中断（2026-10-26）

> 状态：**已完成**。本项目完整回归、Golden 复核、DSH 宿主契约和本地中断复现通过。
> 对应[封版计划 S2](../design/dsh-operator-completion-plan-20261026.md#5-s2任务取消与中断)。只使用本机假 HTTP、受控进程与 CPU/mock；未调用真实模型、SSH 或 GPU。

## 1. 已实现

工具契约 **6 → 7**，DSH 插件 **0.1.6 → 0.1.7**。新增 `operator_cancel`、CLI `cancel`、`/op cancel`，均要求任务目录与实例 ID；status 支持按实例查询。完整使用及状态语义见[取消协议](../design/task-cancellation.md)。

后台 run 在异步启动前先保存不可覆盖的 launch 身份，回包中的 ID 与监督进程、worker 结果和取消请求一致。启动中即可请求取消；同目录重复启动拒绝，不覆盖历史任务。取消只发布请求，由拥有 Popen 的监督进程执行本机清理，客户端不提供任意 PID。

前台 verify/repair/bench 经 CLI/MCP 监督适配器在独立工作进程执行。MCP 执行期间仍可响应 ping/status/cancel 与 `notifications/cancelled`；取消通知按原 JSON-RPC ID 匹配，没有通知响应。原长请求收到明确取消错误或更早完成的真实结果。CLI 的 Ctrl-C/SIGTERM 在清理后输出 JSON，取消退出码为 130；MCP 宿主 SIGINT/SIGTERM 也会清理前台 worker。

新增 `cancelled` / `user_cancelled`。启动、重复取消、已结束、owner 消失、PID 复用、取消与失败/完成/截止竞争都有明确分支。更早完成的有效结果保留；取消胜出时不把局部成功当作整段完成。进程组清理核对已观察的启动身份；已复用的 PGID 跳过，发现/清理失败保持 `cleanup_unknown`。

已发生的模型/验证预留不退款，候选源码、已完成的逐轮结果和不完整目录均保留；未完成步骤不补造完整回执。终态给出预算、部分产物位置及清理事实。库的 stdout 只进入 worker 日志，协议输出由 MCP 主循环统一写出。

## 2. 跨进程计时修复

专项测试发现本机 macOS Python 3.9 的 `time.monotonic()` 采用各进程自己的起点：父进程已运行数十秒时，worker 完成时点可能仍为零点几秒。直接比较会错误裁决取消/完成顺序，也影响跨进程任务截止。

新增 `task_clock.monotonic()`，POSIX 使用系统 `CLOCK_MONOTONIC`；监督进程、worker 结果、取消请求、任务快照及生产模型预算共享该时钟。严格单候选实验继续使用原有进程内时钟。独立新进程的读数必须落在父进程前后两个读数之间，已加入回归。协议实现清单由 **61 → 64**，加入取消、长调用监督与时钟模块；历史证据按既有漂移规则识别。

## 3. 已确认边界

- `accepted=true` 是请求留存回执，需继续查终态。已结束的任务保留原结果；无法联系 owner 时返回 lost 与清理未知。
- `worker_reaped` 和已观察组清理只说明本机事实。`all_descendants_proven=false`；未被发现的快速脱离进程、不可捕获 SIGKILL 和机器故障不被证明已清理。
- `remote_execution=unknown`、`provider_call=unknown`，预算保留 `remote_cancellation_proven=false`。本机 HTTP/SSH 子进程停止不证明服务商推理、计费或远端 GPU 已静默。
- 直接 import HITL 库函数仍是同步调用，中断保证在 CLI/MCP 适配器；没有增加恢复、自动重试或目录复用。
- 判据、数值容差、入库门和默认 Coder 策略保持；S3–S6 未启动，Q1 暂停决定未变更。

真实远端清理验证仍纳入后续 S4 受控真机计划，不能用本机假服务测试替代。

## 4. 验证

- 入口、预算、协议、证据留存与进程清理专项：**233 通过**；随后新增宿主退出清理与显式 MCP 取消路由，S2 专项 **21 通过**。
- DSH 插件及真实宿主契约：先前 **57 通过**；全量结束后仅补正 lost 取消回执及其 JS 测试，最终 **58 通过**，覆盖含空格路径、实例参数、终态显示与清理未知。
- 首次完整命令误收集了 vendored Triton 自身的测试，出现 76 项环境/插件收集错误；已保留该日志，并改用本项目 `pytest tests`。不把它计为完成的回归。
- Golden 首轮 **3 PASS / 1 FAIL**：ReLU 本地 CPU 延迟从基线 0.000700 ms 到 0.000900 ms，超过既有 20% 窗口。未改 registry/阈值；完整回归后单独复核。
- **最终本项目全量：5384 通过、1 跳过、2 项既有非有限值 RuntimeWarning，305.30 秒**。363 个被冻结文件在该次回归结束时哈希一致。Python 实现和测试随后未变；仅补正 2 个 DSH 文件中的 lost 回执及测试，单独重跑宿主全套。
- **Golden 独立复核：4 PASS / 0 REVIEW / 0 FAIL / 0 SKIP**。ReLU 复测为 0.000700 ms；首轮 CPU 计时波动及最终日志均保留，未改基线、容忍范围或测量代码。结果是本地 CPU/mock 回归，不是 GPU 性能成绩。
- S0 的 **11 个归档、2085 个文件**完整性核验通过；文档链接、代码块与 `git diff --check` 通过。

## 5. 持久材料

目录：[output/plugin_completion/20261026-s2/](../../output/plugin_completion/20261026-s2/)。它被 Git 忽略，应随交付另行备份。

- `implementation-and-tests.full-regression.tar.gz` / `source-manifest.full-regression.json`：完整 Python 回归前固定的 **363 个文件**。
- `implementation-and-tests.tar.gz` / `source-manifest.json`：最终 DSH 文案补正后的 **363 个文件**，清单列明与全量快照不同的 2 个 JS 文件。最终宿主测试后再次检查全部文件哈希。
- `pytest-full.log`、`dsh-tests.log`、Golden 日志及 `validation-summary.json`：最终结果、源文件校验和历史失败记录。
- `reproduce_s2.py` / `reproduction.json`：两个生产入口的本地复现，后台运行与 MCP repair 都在本机假 HTTP 阻塞期间取消。
- `cancelled-autonomous/`：同一实例的 launch/request/status、调用预留及清理回执。取消后模型预留为 **1**、剩余 **1**，重复取消返回已结束。
- `cancelled-mcp-repair/`：原始失败源码、独立工具调用、已完成的初始验证（L1 通过、L2 失败）；父 repair 没有伪造 `complete.json`。

复现共派发 **2 次本机假 HTTP 请求**。没有真实模型质量、GPU 性能或远端静默结论。S0/S1 的只读历史材料保持原样。
