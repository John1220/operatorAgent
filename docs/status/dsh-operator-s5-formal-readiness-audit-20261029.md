# S5 P2：正式执行条件差距审计（2026-10-29）

状态：**正式 24 格尚不能放行**。现有文件读取边界、受监督模型请求及共同验证服务可以复用；明确缺口是外层网络出口约束和远端候选执行边界。初版为静态审计，随后补做了仅使用本机合成回环端点的网络缺口复现（见第 5 节）；没有请求真实模型、GPU、SSH 或外部服务，也未更改生产服务。

本报告服务于 [S5 最小完成计划 P2](../design/s5-minimum-completion-plan-20261029.md)。它不是“隔离通过报告”；静态发现的缺失可以否定当前放行资格，但不能代替补齐实现后的运行时拒绝探针。

## 1. 当前已存在的有效机制

| 机制 | 证据与范围 |
|---|---|
| A/C 文件面分离，C 仅暴露插件/代理工件 | [plugin_env.configure](../../benchmarks/s5_plugin_env.py:84) 生成文件白名单，隔离外层源码/账本；[wrap](../../benchmarks/s5_plugin_env.py:115) 比较策略和运行时身份，禁止无围栏回退。既有实际 DSH read/bash/import/路径与链接拒绝见 [文件隔离结果](dsh-operator-s5-read-isolation-20261029.md) 和当前 [A/C 设计](../design/s5-plugin-only-comparison-20261029.md)。本次没有重新执行探针。 |
| DSH 启动环境精简，provider 凭据不放进受试 env | [s5_arm_env.py:77](../../benchmarks/s5_arm_env.py:77) 从白名单环境构建；模型 provider 指向监督器 gateway，受试环境中只有 placeholder key。验证用的一次运行能力文件是正常测试入口，不等于服务商密钥。 |
| 辅助功能显式关闭 | [s5_arm_env.py:55](../../benchmarks/s5_arm_env.py:55) 的 KEEP_DISABLED 包括标题模型、重试、子 agent、web、workflow；[build:80](../../benchmarks/s5_arm_env.py:80) 写入实际 patch。该静态配置不证明任意新脚本/子进程无法另行请求模型，仍需唯一出口。 |
| 经网关的请求先预留再交接 | [GatewayPolicy.stream:214](../../benchmarks/s5_gateway_policy.py:214) 调用 reserve、dispatch 后才打开固定 adapter；[监督 transport:141](../../benchmarks/s5_gateway_supervisor.py:141) 经私有 stdin 传入 endpoint/key，启动受监督 worker，超时清理自己的进程组。缺原始 usage 仍记 missing。 |
| MCP 只开放固定开发面 | [MCPRelay:28](../../benchmarks/s5_plugin_rpc.py:28) 限制 prepare/verify/capabilities；[参数与候选路径检查:130](../../benchmarks/s5_plugin_rpc.py:130) 限定工作区、单一源码、固定契约/后端/L2，拒绝链接。代理只连配置的回环 MCP 端点，见 [proxy:18](../../benchmarks/s5_plugin_proxy.py:18)。 |
| 等额开发验证与独立最终预留 | [DevelopmentService:132](../../benchmarks/s5_development_service.py:132) 以锁串行扣额，提交前写 request/source，失败不退款；最终只能经监督器 Python API，不是 HTTP/MCP 工具。候选/契约及反馈身份绑定位于 [submit:173](../../benchmarks/s5_development_service.py:173)。 |
| unknown 不假冒候选数值失败 | [s5_outcomes.py:2](../../benchmarks/s5_outcomes.py:2) 将网关故障、反馈/开发验证 unknown、证据不完整归设施错误。正式 24 格驱动仍须覆盖“设施停止、后续未执行”的集成回归；近期成功无法证明历史远端传输 unknown 根因消失。 |

## 2. 两个阻断条件

### G1：当前外层策略不限制网络

[s5_seatbelt.profile:72–99](../../benchmarks/s5_seatbelt.py:72) 以 `(allow default)` 开始，只明确拒绝文件读取、写入和 Apple Event；没有 network 默认拒绝、目标地址/端口白名单或代理出口限制。A/C [plugin_env.configure:102](../../benchmarks/s5_plugin_env.py:102) 直接复用该策略，并明确记录 `network_isolation_verified=false`。

因此，当前配置没有从操作系统层禁止 DSH 的 bash/子进程建立其他可达网络连接，包括非本轮网关、其他回环服务或代理。这是策略缺失的静态结论；本次没有实测某个外部服务可达，也没有证明受试进程取得服务商凭据。清理环境、隐藏 key 与正确记账覆盖的是正常配置路径，不足以证明所有派发均受本账本限制。

[GatewayServer:1–6](../../benchmarks/s5_gateway_http.py:1) 自身也明确是回环监听器而非网络隔离；监听器的请求检查和固定上游不能约束没连它的进程。同 UID 的其他宿主进程也不因为 gateway 存在而被隔离。

最小后续：保留现有 macOS DSH，先验证 Seatbelt 是否能按一次运行所需的精确 gateway/development/MCP 回环端点，默认拒绝其他网络和可形成旁路的 IPC；增加实际 DSH read/bash 子进程的 fake-server 探针，覆盖允许端口、非允许端口、IPv4/IPv6、域名/IP直连和代理路径。正常服务必须仍可用，不能只验证“所有网络都坏了”。能否完整覆盖系统代理/IPC要由实测确定，不预先声称该改动已足够正式放行。

### G2：GPU runner 直接在 SSH 用户上下文加载候选

一次性 [SSHExecutor:838–857](../../operator_agent/remote_executor.py:838) 直接在远端工作目录调用 `.venv/bin/python` 或 `uv run python`；常驻 [session:895–908](../../operator_agent/remote_executor.py:895) 同样直接启动 Python runner。调用路径没有命名空间、独立 UID、文件挂载白名单、环境清理或网络拒绝包装。

[run_kernel._load_module:327–344](../../remote_scripts/run_kernel.py:327) 和 [run_compile:367–379](../../remote_scripts/run_kernel.py:367) 仅把代码放入临时文件后 `exec_module`，候选模块顶层代码在 runner 同一解释器中执行。“临时模块”不是 OS 隔离。[runner_server:125–132](../../remote_scripts/runner_server.py:125) 也在同进程分派 `run_action`。故当前不能证明候选无法访问该 SSH 用户有权访问的文件、继承环境和网络，也不能把候选与验证服务/参考实现当作互相隔离的信任域。

本审计没有读取远端 HOME/env，不声称远端现存 provider key、SSH key 或其他秘密已被读取。问题是执行权限缺少受验证约束，而非已发生泄露的证据。

[session_control.py:1–5](../../remote_scripts/session_control.py:1) 明确只为可信 runner 提供生命周期控制；[45–94](../../remote_scripts/session_control.py:45) 对目标 PID 的 UID、出生时间、脚本、token 与 pidfd 进行核验，有助于避免误杀，但不隔离候选。候选与 runner 共用进程时，session token 也在该进程环境中；“owned PID 已停止”不是所有逃逸后代、其他网络调用或服务商工作均停止的证明。MCP relay 的 [cleanup:317–329](../../benchmarks/s5_plugin_rpc.py:317) 和模型 worker 的 [cleanup:93–110](../../benchmarks/s5_gateway_supervisor.py:93) 同样保留 descendants unknown，不能把这一诚实标记提升为完整清理通过。

最小后续：先为远端候选引入按验证实例的限制 worker，使用独立临时 HOME/work/cache、清理 env、只读运行时/runner、所需 GPU 设备白名单、无外网和受控生命周期组；评估 Linux `bwrap`/命名空间或受限用户是否能满足所需 CUDA/Triton 行为。监督器账本、连接凭据和独立参考材料不应暴露给候选。涉及参考与候选同解释器可互相影响的部分，需明确“可信合作候选”与真正隔离判定的边界，不能仅给整段 runner 加一个临时目录就宣称解决。

在无模型、无 GPU 的文件/网络/子进程探针通过前，不改现有真机默认路径；之后仍须针对实际 GPU worker 路径核验兼容性、输入保真与归属清理。不能用 CPU/fake 通过替代 CUDA 路径的最终资格。

## 3. 是否必须迁移 DSH 到 Linux

**目前没有证据要求整套 DSH 迁移。** 可先尝试“现有 macOS 外层收紧出口 + Linux 远端受限候选 worker”，保留已验收的模型网关、MCP 和共同验证服务。这样可以避免重做 Linux DSH/Node 原生依赖闭包。

但 G2 已超出改几项提示或冻结参数的规模，涉及执行适配、GPU设备访问、Triton缓存、输入/参考边界和取消清理。若现有用户/内核不支持所需命名空间、GPU兼容性不成立，或需部署常驻特权服务，必须先向用户提供具体变更和成本再决定；不自动下载整套依赖、不改共享宿主服务，也不通过降低“正式”的定义绕过它。

| 子工作 | 可直接推进的最小检查 | 成本边界 |
|---|---|---|
| G1 | 精确端口策略原型 + 本机假端点正常/绕行矩阵 | 相对集中；实际可行性仍需探针，未知 IPC 能力不能计通过 |
| G2 | 已有远端能力材料审阅 + 限制 worker 设计 + CPU canary/进程探针 | 需执行适配和GPU兼容验证，不能承诺一次小改完成 |
| 整套 Linux DSH 迁移 | 暂不做 | 只有现有拓扑确实不能满足边界时才评估 |
| 24 格真实对照 | 暂不启动 | 模型费用、GPU占用待正式门通过；静态审计不产生其授权或执行证据 |

## 4. 留存与下一次放行需要的证据

静态证据目录：`output/plugin_completion/20261029-s5-minimum-p2-audit-v1/`，包含 `source-manifest.json` 与 `audit.json`。摘要哈希为 `70a147a3b09a41fc0ae4c74c48384d5568fc650574671bad0dfded1c5a348058`；记录 `formal_ready=false`、真实模型调用 0、GPU执行 0、SSH调用 0。清单绑定本次审阅的 15 个文件；之后源码变更需重新核对，不能把静态快照当新执行的凭据。

下一次正式放行至少要绑定运行时/源码/策略哈希、实际 A/C 外层出口拒绝与允许探针、受限 GPU候选路径的文件/网络/生命周期证据、辅助调用/预算记账一致性，以及新 study 驱动故障停止和完整分母回归。已有无模型 readiness 只证明此前声明的文件与工具路径范围，不能重命名成这份正式报告。

在这些证据不存在前，正式执行入口应在读取 provider 凭据、查询/占用 GPU、创建执行记录或发出请求之前拒绝 study 派发。P1 可以继续完成并用模拟后端验收；P2 本次完成的是差距审计，不是正式隔离实现或资格通过。

## 5. 本机无模型实测：非网关回环端口可达

新增 `benchmarks/s5_network_audit_probe.py`，用真实 `s5_plugin_env.build` 和 `wrap` 生成 A/C 实际文件围栏，分别运行 Python 直连与 `/bin/sh` 子进程。监督器只启动两个自己拥有的 `127.0.0.1` 合成 HTTP 监听器：一个代表声明网关，另一个代表非声明 sentinel 端口。没有启动 DSH 模型循环；探针使用的是相同外层包装和进程树策略，不把它描述成真实模型绕行行为。

无围栏正向对照通过。围栏中的 A/C × 两种路径共四项，都成功访问两个端口，服务端请求路径与客户端回复一致；证实当前边界没有限制到本轮网关端点。所有归属监听器都已停止。没有探测实际 provider、宿主代理、外网或远端凭据，不能由本次回环实测断言这些具体目标已经可达。

结果保存在 `output/plugin_completion/20261029-s5-minimum-p2-audit-v1/network-probe/summary.json`，状态 `bypass_reproduced`，哈希 `60ad793ebfde6dfd2e1bbb4a9bbdea5e0e83e37221db3e01e977f27bebb15944`，`formal_ready=false`。实际策略文件、策略 SHA256、客户端输出、服务端请求和清理结果均保留。首次环境沙箱拒绝本机监听的初始化失败另存 `network-probe-initial/`；允许本机监听后重新运行完成，不把初次环境拒绝当成被测文件围栏已限制网络的证明。

[P1 验收](dsh-operator-s5-minimum-p1-20261029.md)的 252 项相关回归及真实入口拒绝证据已经完成；正式 study 入口目前会明确拒绝，不因任意 JSON 报告自称通过而放行。静态审计与本机探针只确认 P2 缺口，未建立 G1/G2 修复后的执行资格。
