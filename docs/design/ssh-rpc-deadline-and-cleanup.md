# 常驻 SSH RPC：全程截止时间与 owned-runner 清理

日期：2026-10-08。范围是 NVIDIA 常驻 SSH 行协议；不是 GPU watchdog、敌意候选沙箱或完整 RPC 幂等协议。

## 1. 这次解决什么

此前 `_RemoteSession._call_locked()` 在 `stdin.write/flush` **之后**才开始等待超时。
远端停止读请求或大输入填满管道时，发送自身没有截止时间；`close()` 也用同一条阻塞写
发送 shutdown。会话坏掉后 `_start_session()` 还可能起新进程，却没有确认旧 runner 停止。

发现现场见 [默认正确性配置的早先失败](default-correctness-profiles.md#4-h20真实生产链路正反验收)。
不能把控制端没拿到回答，翻译成“候选算错了”，或直接重试一个可能还在飞的 GPU 请求。

## 2. 一次请求，一条 deadline

`_RemoteSession.call(payload, timeout=None)` 在入口创建 `time.monotonic()` 绝对截止时间：

```
等待会话锁 → JSON/UTF-8 编码 → 非阻塞管道发送 → 响应队列等待 → 校验 id/result
                          同一个 deadline，不逐阶段重置 timeout
```

- stdin 只有 `os.write` 一个写者；`os.set_blocking(False)` + selector，以最多 64 KiB 分块。
  没有缓冲写，因此不用可能阻塞的 flush；帧仍为一行 JSON，不改远端数值协议。
- 编码在线程内完成，控制端只按剩余时间等字节。迟到的编码线程**不持有/不写 stdin**。
  这是限时等待，不是可硬杀的 CPU 编码进程；已开始的编码可短暂继续，不承诺硬实时或内存隔离。
- `timeout=None` 使用会话默认；0、负数、NaN、Inf 拒绝，不能因为 `timeout or default`
  意外延长显式预算。锁等待超时的请求没有发送，不破坏持锁请求的会话。
- 发过部分帧后超时、EOF、错位、非对象 result 都使会话 broken。完整回答须 id 精确匹配
  当前整数序号；后续调用不消费迟到回答，不追加新 JSON 或 shutdown。
- `SessionError` 发布 `session_state/session_phase/request_id/bytes_sent`，不含请求大缓冲。

**边界**：deadline 从已构造的 RPC payload 进入 `call()` 开始；不包含调用者的 NumPy storage
打包、任务生成、部署/选卡、独立启动握手。那些阶段有各自上限，不能据此宣称整条任务链已统一截止时间。
stdout/stderr 排水与解析仍在线程内；不声称对恶意无限输出有资源隔离。

## 3. 失效会话不能自动复活

`SSHExecutor` 保留失效会话；握手失败也保留进程和早期身份，不丢失半开的 runner。
调用不会自动新建、复用流或重试。清理未确立时保存在 `_pending_session`；下次调用在发送前拒绝。

`close(grace=10)` 返回结构化证据：

```json
{
  "local_quiescent": true,
  "shutdown_ack": false,
  "remote": {"quiescent": true, "identity_verified": true, "signals": ["SIGTERM"]},
  "quiescent": true
}
```

健康流可在有限预算内 shutdown；坏流只关 stdin/等本地 SSH/必要时终止本地进程。
**本地 SSH 退出不是远端静默**。总清理预算同时覆盖本地收尾与独立控制命令。
失败时可再次 `close()`，只重试生命周期控制，不重放结果未知的 GPU 请求。
只有本地和远端均确认静默，才解除恢复阻断；新建会话是调用者后续的显式行为。

## 4. 谁有权终止 runner

每次启动生成 128-bit 随机 session token，通过 runner 环境变量传入。
`runner_server.py` **在 Torch/CUDA import 前**回报 PID、UID、`/proc` start_ticks、绝对脚本路径。
独立 `session_control.py` 不 import Torch，不依赖已坏的 RPC 流。

清理在 Linux 上打开 pidfd，再校验 `/proc`：

1. UID 等于读回身份及控制进程 UID；
2. start_ticks 等于该进程出生时间；
3. cmdline 独立参数精确包含该绝对 `runner_server.py` 路径（不做子串匹配）；
4. environ 精确包含本次 session token。

只有四项成立才向 **pidfd** 发 SIGTERM，有限等待后必要时 SIGKILL，最后读回该进程退出。
pidfd 固定目标，避免检查后 PID 复用把信号打给别人。PID 已不存在/已退出也有明确证据。
身份缺失、字段错位、属主/路径/token 不一致、缺 pidfd 支持或控制通道超时：
**不发送信号、不声称静默、不自动恢复**。不用通用 `pkill`，不杀共享目录下的同名 runner。

独立 scratch 的删除还沿用目录随机名、属主、0700 与残留进程读回；未确认清理则保留目录。
控制命令的错误只保留类别，不落盘 session token 或 SSH argv。

**边界**：只证明本次 runner 退出，不证明它的任意子孙进程、所有 CUDA context 或设备租约
均被回收；同 UID 的敌意候选不在这份身份方案的威胁模型内。HTTP/XPU 没有迁移。

## 5. unknown 必须到达调度与写入侧

`SessionError` 统一归入 `transport_outcome=outcome_unknown`，包括 timeout/EOF/desync/broken。
远端明确返回的 `success=false` 仍为 answered，不混淆真正的候选数值失败。

- 编译出口不把 SessionError 包成候选 SyntaxError；L1 保留 transport_unknown。
- L2 在缺 input fidelity 回读之前保留传输状态；suite verdict 为 unknown。
  case 或 repeat 超时后，其余 case 记 not_run，不再发送请求。
- 决赛选择遇到失效传输即停止，不尝试其他候选；Conductor 走基础设施 HOLD，停止原因
  `transport_unknown_hold`，不启动 Fixer，不继续轮次/L3/确认，不进负样本库。
- 普通确定的编译/数值失败与原有 Golden 基线/容忍窗口不变。

## 6. H20 验收（真实读数与传输对照分开）

入口：`python3 benchmarks/rpc_deadline_h20.py`；可加 `--gpu 2` 指定物理卡，忙卡必须拒绝。
既有运行时、私有随机 0700 目录、9 份共享源码 SHA 校验，不 install/sync、不计时性能。
两次 GPU 2 被探测为忙，安全拒绝，未创建 scratch/执行 GPU 工作；一次空闲复查后在 GPU 2
完成 6/6。最终复验自动选择空闲 GPU 7，固定该卡；不是强行抢 GPU 2。

最终成功编号：`rpc-deadline-h20-20261008T104105209963Z-95218-1`，**6/6 符合预期**。
H20 物理 GPU 7 / Torch 实际 cuda:0；Torch 2.14.0+cu130 / Triton 3.8.0。
早先 GPU 2 的成功编号 `rpc-deadline-h20-20261008T103005700068Z-88625-1` 同样保留。

| 对照 | 证据 | 实测 |
|---|---|---|
| 中载荷 | 1 MiB fp32、真实 CUDA 全量 ReLU/输入保真 | 2.48 s，误差 0，coverage=1 |
| 大载荷 | 16 MiB fp32、真实 CUDA 全量 ReLU/输入保真 | 44.52 s，误差 0，coverage=1 |
| 完全停止读取 | 16 MiB JSON blob，**SSH fixture，不执行 CUDA** | send 超时 0.505 s，已发送 2,228,224 B |
| 只读取 1024 B 后停止 | 同上 | send 超时 0.502 s，已发送 2,232,320 B |
| 已接收但不回应 | **SSH fixture，不执行 CUDA** | response 超时 0.509 s |
| 清理后的新会话 | 原值/strided 真实 CUDA L2 | 数值与保真通过 |

三条失效对照都拒绝重启，独立校验身份后只 SIGTERM 本次 runner，确认退出；
清理分别 1.93 / 2.62 / 1.96 s。最后正常 runner 退出、scratch 删除均读回。
证据在 ignored `output/rpc_deadline_h20/<编号>/report/summary.json`；不提交机器/用户路径。
这些时延不是吞吐或性能收益结论；不宣称任意大输入/完整网络故障组合已验收。

## 7. 离线守卫

`tests/test_rpc_deadline.py`：真实管道、部分发送、共享预算、慢编码与迟到隔离、锁竞争、
大 Unicode 帧、关闭取消、早期身份、坏会话禁止自动恢复、UID/出生时间/路径/token 的反向对照、
pidfd 限时等待、L1/L2/suite/决赛/Conductor HOLD/负样本写入等。
卡住发送的测试有外层受控子进程截止时间，不允许旧实现把 pytest 永久挂起。
`tests/test_rpc_deadline_h20_acceptance.py`：验收缺证据/CPU/部分覆盖/失效清理必须拒绝，
远端 cleanup 必须在可能失败的本地产物写入之前。

新增 48 项守卫；完整本地验证结果登记在状态文档。离线 fake pidfd 守卫不是 Linux/GPU
实测的替代品。运行项目回归应使用 `python -m pytest tests -q`，不是扫描含 vendored Triton
源码的仓库根（第三方 GPU 测试不属于本项目的本地回归集）。

最终本地全量：**2966 passed / 1 skipped / 2 warnings**（本机 CUDA 缺失跳过；已知非有限参考
负例的两条 warning 仍验证 unknown）。之后串行 Golden：**PASS 4 / REVIEW 0 / FAIL 0 / SKIP 0**。
基线、容忍窗口和性能计时代码未变。

临时副本逐项变异：**7/7 捕获**——阻塞管道、响应阶段重置预算、失效会话自动重启、
跳过 owner token、缺身份冒充静默、未知传输进入负样本、L3 丢失 unknown。
首轮 **6/7**：预算重置虽然在迟到回答后被 `_remaining` 拒绝，但用例的时间余量太宽，
没有抓住“过期后才返回”；将晚回答对照与 deadline 拉开后复验 7/7。
这是有限关键变异标定，不是完整故障/变异覆盖率。
