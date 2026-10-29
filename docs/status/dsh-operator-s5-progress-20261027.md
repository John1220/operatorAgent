# S5 三组对照准备进度（2026-10-27）

状态：**进行中；当前研究范围为原生 DSH vs 仅经插件发布接口接入的 DSH。A/C 六格批次部分执行后因验证器传输 unknown 停止**。设计见[两组插件接入方案](../design/s5-plugin-only-comparison-20261029.md)，旧三组 30 次方向性运行及九格预检原样保留，不转记为新两组结果。用户要求完成 S5 后再继续 S6；此前 Q1 单独对照维持暂停，不因本次研究范围调整恢复。

最新补充：独立 C-only 三任务诊断已全部通过最终 L2（20 次模型调用、约 16 分钟）；三格实际均走公共检查入口，插件验证工具的真实模型使用仍待验收。详见 [C-only 结果](dsh-operator-s5-plugin-c-diagnostic-20261029.md)。

## 已有材料

- [S5 三组对照方案](../design/dsh-operator-s5-comparison-design-20261027.md)：A 裸 DSH、B DSH + core CLI、C DSH + 同一 core CLI + 插件；三组要共同计入外层与内层成本，不能把声明式候选来源当路径证明。
- [性能量尺预检](../design/s5-performance-ruler-preflight-20261027.md)：两项真实 H20 测试得到可裁决的单次读数，无模型调用、无独立性能确认；不作为正式 S5 样本。
- [14 项任务草案](../../output/plugin_completion/20261027-s5/task-manifest-draft.json)，内容哈希 `8663285abafdd8fbbb3cfd9d0c6affa726c97cc0e1d27ab11555825eb0e12d26`；构建脚本 `benchmarks/s5_task_manifest.py`。前十项逐字段保留原质量集的主输入与两个 holdout；四项新增候选暴露与性能可测性未知。草案尚未封版或用于正式调用。
- 本机 `dsh --profile headless --help` 在临时 `DSH_HOME` 中可运行，未初始化或更改用户真实 profile。此检查只证明帮助入口可启动，不证明三组隔离、模型调用记账或端到端实验可用。
- 2026-10-28 增加共享 SQLite 预算原型、显式启用的内层桥及外层 DSH waterfall 原型：预留不退款，停止状态持久化，缺失或不可信的 provider usage 保持 unknown；外层和内层故障会尝试停止共同账本。交接记录只证明交接意图，不证明请求实际到达服务商。账本故障或替换时可能无法持久写入停止，须由监督器独立停止进程并保留 unknown；同 UID 下文件检查也不是防恶意篡改隔离。
- 跨进程并发预算、实际 Python bounded client、repair 接线、生产预算、汇总及完整回归中失败项的复测合计 **118 项通过**；本机安装的 DSH waterfall 配合 fake adapter 的 Node 测试 **17 项通过**，无真实模型调用。mock 终评、任务草案及量尺预检离线测试 **21 项通过**；其中新增区分确定的接口不匹配与传输未知。此前完整项目回归先后发现 S5 账本新建文件违反单一写入口约束，以及量尺预检异常退出后未恢复环境变量、影响后续 SSH 状态测试；两处均已修正，相关测试 **104 项通过**。最终完整回归 **5629 通过、1 跳过、2 条非有限数值边界测试 warning**（2026-10-28）。
- [隔离与监督实施边界](../design/s5-isolation-and-supervision-20261028.md)：已确认仅换 cwd/DSH_HOME 不能隔离 A；正式执行仍需受控模型网关、统一验证服务及可审计的真机适配。既有 S5 与暂停的 Q1 是不同研究范围，S5 不恢复 Q1 或改变默认 Coder。
- `benchmarks/s5_summary.py` 提供离线草案排程与汇总原型：固定 14 × 3 行和交替顺序，保留未执行/未知，拒绝重复或计划外记录，缺失 usage 不写成零；配对只计算双方完整报告的布尔结果。原型仅检查调用方报告格式，不核验产物及来源，不能当作正式结果；相关测试 **12 项通过**。
- 在预算代码变更后，实际 DSH 插件包的 `npm test` **58 项通过**。事务内任务身份校验、初始化期替换和日志故障注入的共享账本专项测试 **32 项通过**。
- `benchmarks/s5_fake_gateway.py` 是进程内、可信假 adapter 的离线预算网关原型，支持 JSON/SSE 字节透传，覆盖并发调用、取消、响应大小和账本错误；**15 项测试通过**。它不是 HTTP 服务、真实模型网关或隔离机制，fixture callable 也不受代码沙箱约束；无法证明实际流量全部受到控制。真实网关必须成为唯一跨层预留权威，不得对同一请求同时在两侧共享桥重复预留。
- `benchmarks/s5_fake_http.py` 为上述 fixture 提供仅绑定 `127.0.0.1` 临时端口的 HTTP 适配，兼容试验范围内两种输出上限字段及客户端扩展；**12 项离线测试通过**，包括真实 Python `LLMClient` 子进程与假 adapter 的联测、SSE/tool-call 字节和共享预算。另有 **1 项**使用已安装 DSH LLM runtime 与 pi-ai adapter 的回环联测通过：外层流式请求和 Python 内层请求共享总额度，后续请求被拒绝；该测试并非完整 DSH agent，也不是可出网的真实网关或隔离证明。
- 已通过 npm 元数据生成 [Linux DSH 依赖锁草案](../../output/plugin_completion/20261027-s5/linux-runtime-draft/package-lock.json)，包含 Linux 原生可选依赖，尚未下载安装完整包或验证启动；运行时版本和依赖审查仍未完成。
- [Linux 宿主下载试探与成本](dsh-operator-s5-runtime-probe-20261028.md)：完整 CLI 依赖与缓存合计约 1 GiB 时主动中止，未取得完整依赖或 Linux Node，尚未启动；真实网关和统一 GPU 入口仍需独立开发和验证。
- [最小纵向试验](dsh-operator-s5-vertical-trial-20261028.md)：本机 macOS DSH headless → 一个假模型请求 → 一个可信 fixture 的 mock L2 已跑通；主输入与两个 holdout 通过，专项测试 43 项通过。该结果不证明 Linux、真实模型/GPU 或隔离，也不计入正式 S5。
- [工具反馈纵向试验](dsh-operator-s5-tool-feedback-trial-20261028.md)：DSH 实际调用本机实验验证工具，两次假模型请求之间收到经过候选与回执哈希核对的 mock L2 反馈；最终专项回归 49 项通过。该试验没有测试本仓插件/MCP 或真实资源。
- [真实 MCP 最小纵向试验](dsh-operator-s5-mcp-vertical-20261028.md)：DSH 通过仓库 MCP `operator_verify` 调用 mock 后端，核验主输入、完整 12 项 suite、worker 与第二轮反馈；专项回归 33 项通过。`/op` 斜杠命令、Linux 和真实资源未在这次试验中验证。
- [预算转发网关假上游验证](dsh-operator-s5-gateway-prototype-20261028.md)：真实 DSH/pi-ai 和 Python 客户端通过进程监督、共享账本与本机假上游完成联测；71 项专项回归通过。该阶段未测试真实上游；系统级隔离和金额硬上限仍未建立。
- [单次真实模型网关预检](dsh-operator-s5-real-gateway-preflight-20261028.md)：固定 1 次、256 输出上限的短请求通过，原始上游报告 35 输入、13 输出 tokens；使用同模型同主机的 Completions 兼容路径，不是 DSH 默认 Responses 路径，也不代表正式 S5 或隔离验收。
- [三组启动路径与统一验证接线](dsh-operator-s5-three-group-preflight-20261028.md)：A 裸 DSH、B DSH + core CLI、C DSH + 同一 CLI + 插件按冻结计划各跑一次同一任务，脚本化假模型、共享账本与 supervisor 侧统一 mock 验证全部通过（三组同 verdict 哈希，每组 1 次假调用、usage unknown）；该预检在计划中预先声明**不计入正式分母**。同机 A 组仍可读本仓，裸 DSH 未被证明；真实模型路径、H20 统一验证、工具面差异与隔离仍未闭合。
- [DSH 默认 Responses 路径经预算网关](dsh-operator-s5-responses-path-20261028.md)：网关新增 `/v1/responses` 协议（含 `response.completed` 终止校验），真实 DSH agent → 网关 → 回环假上游完成联测，截断/失败流记 unknown；随后一次 256 输出上限的真实短请求通过（计划 `f0bc4458…`，1 预留 1 完成，最终文本 `OK`，usage 仍 unknown）。此前两次本地失败（相对路径、模型名不一致）各 0 次模型调用并已修正加测。
- [统一验证服务与一次真实 H20 L2](dsh-operator-s5-verify-service-h20-20261028.md)：监督器侧统一提交入口（mock 与 SSH 两条后端），候选先落盘、服务自备契约、证据要求套件完整与输入保真，缺项记 unknown；一次固定 GPU 7 的真实 H20 L2 复核通过（106.6 秒、12 case、无模型调用、后端未降级）。三组尚未接入该服务。
- [三组最终提交接入统一验证服务](dsh-operator-s5-three-group-service-20261028.md)：A/B/C 的最终候选都交给同一个监督器侧服务（自己 prepare、候选先落盘、结论只写一次）；一次 mock 后端三组运行 passed 3/3、汇总 `fc8862dd…`。两个契约哈希（质量集草案 `da7f3620…`、生产验证器 `96cdb38…`）按任务身份绑定并在记录中分开说明。
- [三组对照执行结果](dsh-operator-s5-formal-results-20261028.md)：10 项已知 exposure 任务 × A/B/C 各一次，共 30 次全部执行（A 6/10、B 5/10、C 7/10 含 1 unknown），配对 B−A 1 胜 2 负 7 平、C−B 2 胜 0 负 7 平；matmul 三任务三组全败。实时模型、真实 H20 验证、固定分母，但 **A 未隔离、开发期反馈不对称、每格仅 1 次**，因此不给正式 S5 credit。
- 三组接线同时修正了环境漂移：`PYTHONNOUSERSITE=1` 会隐藏用户 site-packages 中的 numpy，导致 MCP/工具纵向试验失败；现改为显式冻结依赖目录（931 文件树哈希）并被 MCP、工具与三组入口共用。S5 专项回归 **256 项通过**（2026-10-28）。

## 正式实验前必须闭合

1. 统一外层 DSH 与内层 Python 的跨进程调用预留、失败和取消留痕、原始 API usage 的缺失判定，以及所有辅助模型调用的覆盖。普通 session usage 不是完整派发账本。**外层 DSH 的 Responses 协议已接入网关并做过一次真实短请求，内层 Python 与 usage 判定仍未闭合。**
2. 三组真实启动路径与工作区隔离、A 组不挂载 core/插件时的公平执行环境、候选路径证明及一致的独立终评，须通过 mock/fake-provider 及故障注入。**三组已接入统一验证入口（mock 后端 3/3 通过），入口本身有一次真实 H20 通过；但三组尚未用真实模型和真机后端运行，A 组仍可读本仓，各组工具面未实际使用，因此本项未闭合。**
3. 确定并冻结最终任务、评估器、调用及耗时预算、身份与资源；明确三组研究问题不自动恢复 Q1。实际性能资格不够的任务记 unknown。
4. 实验过程中保留各组全部分母、失败和中断；资源被占、远端清理未知或账本不可证明时停止，不通过扩大预算或反复重跑挑选结果。

此前 S6 的隔离安装与脱敏案例包只属于准备工作，不在 S5 未完成时被追认为 S6 完成；下一阶段继续遵循用户指定顺序。

## 2026-10-29：任务语义与开发期闭环修正

已复核 [v6 的 30 次历史运行](dsh-operator-s5-formal-results-20261028.md)：汇总可重算，但 matmul 的三维提示与二维评测不一致，C 未观测到 operator MCP 验证，不能作模型能力或插件闭环收益归因。历史材料保留，新结果不可覆盖旧分母。

已实现三组共同的监督 L2 服务、每组 2 次开发期 + 1 次独立最终预留，通过实际 DSH 的公共命令/CLI/MCP 接线保留候选与反馈顺序；用户指定后续 medium/单次 32K，总输出预留同步为 786432，宿主上下文与传输界限同时调整。最终全量 5827 通过、1 跳过；30 项本地专项、相关模块 135 项、DSH 58 项及 Golden 4/4 通过。详见 [2026-10-29 闭环接线验收](dsh-operator-s5-workflow-wiring-20261029.md)。这次不调用真实模型/GPU；完整 Responses/GPU预检、隔离与正式样本仍未完成，S6 未开始。

## 2026-10-29：medium/32K 真实闭环预检未放行

按独立计划冻结三任务三组 9 格，实际派发 2 格后停止：A/matmul-tail 的公共入口反馈闭环与独立终评均通过完整真实 H20 L2；B 用尽 24 次模型额度而未提交候选，且真实读到了 A 候选前 80 行。固定 9 格为 1 passed、8 unknown（7 未执行），不与历史 30 格混合，不作插件增益解释。实际留存 29 次模型调用预留和 2 次真实验证；GPU 7 结束检查空闲，无其他服务变更。详见 [真实预检结果与跨组污染证据](dsh-operator-s5-workflow-preflight-20261029.md)。下一项为无模型的文件读取隔离验收，再决定新批次；正式 S5 与 S6 未完成。

## 2026-10-29：读取边界无模型探针通过，新九格 v2 冻结未执行

实现 macOS 外层文件白名单，将受试 HOME/work/tmp 移出仓库；B/C 使用相同只读 core 副本，C 额外可读插件，A 无 core 访问。实际 DSH 的 read/bash 与公共入口、CLI、MCP 在无模型费用的脚本化验收中通过；跨组/假凭据/私有账本、路径与链接绕行被拒绝，反馈闭环保留。全部相关回归 409 项通过。见 [读取隔离验收与限制](dsh-operator-s5-read-isolation-20261029.md)。新九格预检 v2 已绑定报告及源码身份冻结，尚无真实请求；旧污染批次保留。网络/系统代理等完整隔离条件、正式 S5 和 S6 均不据此放行。

## 2026-10-29：最小实施与隔离九格 v3 设施停止

已实现 6 个前瞻任务 × 2 次 × ABC 的调度和配对汇总、每格独立预算/证据，以及普通预算停止继续、设施错误停批的规则。按用户要求先执行独立九格，36 格未运行。九格 v3 实际派发 3 格：A/matmul 最终 L2 通过，B 耗尽 24 次调用预算后仍调度 C，C 因 JSONL 输出解析错误停止后续 6 格；共 47 次模型调用、3 次真实验证，固定分母 1 passed、8 unknown，不作插件收益推断。

已用无模型复现定位 MCP SDK 资源查询向 stdout 写 `console.debug` 的路径，统一启动层改为 stderr，且在解析前留存原始输出。相关回归 77 项通过，三组无模型读取/反馈探针 v5 通过。v2 未执行计划、v3 停止结果及旧源码快照均保留；新真实执行须重新冻结计划，本轮未自动付费重跑。详见 [最小实施与实际结果](dsh-operator-s5-minimal-20261029.md)及[最小对照设计](../design/s5-minimal-study-20261029.md)。

## 2026-10-29：GPU2 共享九格 v4 全部派发，门槛未通过

用户授权继续后，将 GPU2 的设备身份和当前 7 个常驻 PID 冻结，显存至少 8192 MiB、利用率至多 5%，每格前检查，默认独占规则保持。相关回归 118 项通过。独立九格全部派发并结束：A 3/3、B 1/3、C 0/3 最终通过；另四格调用预算耗尽、一格 Softmax/B 网关 502 设施未知。共 135 次模型调用、4 次开发和 4 次最终真实 GPU 验证，各格墙钟合计约 29.3 分钟；原始结果和源码快照保留。GPU2 结束恢复至剩余 22246 MiB，未停止其他服务。

JSONL 留存及转录审计完整；C 的真实 MCP 验证未被使用，探索开销消耗预算，结果不支持插件增益声明。502 细分根因缺少留痕，无法归为模型失败，预检门槛仍未通过。见 [完整结果、局限与下一步](dsh-operator-s5-gpu2-preflight-20261029.md)。未执行 36 格，S5/S6 未封版。

## 2026-10-29：按发布使用方式收敛两组

用户要求仅比较原生 DSH 与 DSH + 插件，去掉直接 CLI、core 源码可访问性等额外条件。已更新[当前设计](../design/s5-plugin-only-comparison-20261029.md)与完成清单：删除 B，插件的 core 依赖保留在后台，C 仅经正常发布接口使用；两组共享任务、原生工具、基础测试条件、额度和独立终评。首轮仍限定外层写核与插件验证，不扩展到内层自动生成/修复。

本次只有文档变更，没有修改实验驱动、运行新探针或调用真实模型/GPU。旧 C 的 CLI/PYTHONPATH/core 挂载仍属于旧实现，不能冒充新边界；下一步先实现并验收两组的真实入口和源码隔离，再考虑独立 6 格预检。前瞻 6 任务 × 2 次 × 2 组的候选规模为 24 格，尚未冻结或运行。历史结果不删列、不覆盖、不合并；S5/S6 未封版。

## 2026-10-29：A/C 无模型接线验收通过

新增 [插件限定 readiness 报告](../../output/plugin_completion/20261029-s5-plugin-only-readiness-v1/summary.json)：实际 DSH + 脚本化假模型下，A/C 各 10 个调用完成；A 的 core/仓库/对照候选/凭据/账本读取、导入、CLI 和子进程读取均被拒，共同 `check.py` 可读。C 仅多出暂存插件工件，通过真实 MCP 发现 schema/instructions，并以同一监督服务完成一次失败和一次通过验证。两组都耗用 2 次开发验证，额外提交被共同额度拒绝；流程观察器确认首次候选在 DSH 步骤 5、首次验证反馈在步骤 6，未注入额外提醒或调用。

独立 `s5_plugin_workflow.py` 已加入六格 A/C 冻结/执行入口，固定 medium/32K/24 次每格/2400 秒；计划绑定 readiness、任务服务、GPU 资源策略及源码身份。调用预算耗尽不被当作设施错误；网关/验证/隔离设施 unknown 会停止后续派发。执行器只接受精确计划哈希，并逐格重新检查 GPU2 共享资源条件。旧 ABC 入口/计划不被新执行器接受。

限定回归 **134 项通过**。另以临时目录实际冻结并重新加载一个六格 A/C 计划，绑定三个指定任务、medium/32K/24 次额度、readiness 和 mock 的 GPU pin；没有保留或执行该临时计划。readiness 与计划接线均为本机 mock/脚本化无模型验收，真实模型调用 0、GPU 验证 0。readiness 报告明确 `network_isolation_verified=false`；未证明只允许连预算网关或抵御同 UID 恶意进程。新的持久六格计划尚未冻结或执行，真实运行需重新核对网络/IPC 要求与 GPU2 资源；历史 ABC 结果不变。

## 2026-10-29：A/C 六格真实批次因验证 unknown 停止

首次真实启动在 DSH 启动接线处触发重复 `tools=True` 的 `TypeError`，预算账本确认模型预留 0、GPU 验证 0；记录保存在 `output/plugin_completion/20261029-s5-plugin-only-ac-v1/execution-v1/`。修正 A/C 环境适配器并增加同参数组合回归后，针对性回归 12 项通过；以当前源码重跑无模型 readiness，A/C 各 10 个脚本化请求通过，再冻结新计划并派发。

新计划哈希 `22830ea3e66ac3c8b062cba1cbffc7faa7dbafab52834f84030ba2a775b792b2`，结果汇总哈希 `7fc3ff37b52ad6f41016ab3445349ffc7090ed5325985aaf06a68ae89629fd72`。共派发 3/6 格，25 次模型调用预留；provider usage 均缺失，费用未知。Matmul A/C 都以 8 次调用通过独立最终 L2，墙钟分别 407.036 秒和 479.934 秒；单个任务的 C 多约 73 秒，样本不足以推断插件影响。C 调用了 `operator_capabilities`，但最终开发验证经两组共有的公开 Python 检查入口提交，未观察到插件 `operator_verify` 调用；本配对没有证明实际插件验证能力带来增益。

第三格 LayerNorm-C 用 9 次调用；开发 L2 通过，但最终 CorrectnessSuite 的重复检查出现远程验证传输 unknown。模型网关没有上游失败记录；错误属于验证器结果未知，不能判作候选数值失败。按设施 unknown 停批规则，LayerNorm-A 与两格 Softmax 未派发。固定分母汇总为 passed 2、failed 0、unknown 4，`inference_allowed=false`、`formal_s5_credit=false`。结束后 GPU2 观察为空闲（22,246 MiB、0%）；没有停止其他常驻服务。readiness 的 `network_isolation_verified=false` 仍成立。

完整证据与限制见[A/C 实测记录](dsh-operator-s5-plugin-only-ac-results-20261029.md)。下轮提速可先跑单任务 A/C 配对（2 格），但要先定位本次验证器传输 unknown 并重新冻结计划；本批 unknown 格不自动重跑。


## 2026-10-29：C-only 三任务全部通过

增加独立的 `freeze --mode c-only` 诊断入口，默认 A/C 六格保持原协议。新计划哈希 `7657260bfbefcef3235fe8cd248fe8a48322ebb7dacf8b7c92296defc3fb3f40`；重新通过无模型 readiness 后，在 GPU2 派发三个 C 格。Matmul、LayerNorm、Softmax 均通过开发与独立最终 L2，模型调用分别 6、7、7 次，累计约 16 分 4 秒；汇总 passed 3、failed 0、unknown 0，哈希 `4ca123267510aae5415c4cfdd437870ed772413e3889392064e49f514f888914`。相关环境/驱动/汇总回归 52 项通过。

三格模型均经公共 `check.py` 提交开发验证，`plugin_verification_observed=false`；仅 Matmul、LayerNorm 调用了插件 capabilities。结果说明 C 配置下三个任务可完成，尚不证明插件验证工具被模型实际使用或带来增益。旧 A/C 的 unknown 不变，不补旧格；本轮 `formal_s5_credit=false`、`inference_allowed=false`。下一步优先检查插件使用说明与工具发现，单独取得真实 `operator_verify` 使用证据。详见 [C-only 结果与证据边界](dsh-operator-s5-plugin-c-diagnostic-20261029.md)。
