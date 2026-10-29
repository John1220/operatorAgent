# 算子开发插件 S0：固定起点验收（2026-10-26）

> 状态：**已完成**，对应[封版计划 S0](../design/dsh-operator-completion-plan-20261026.md#3-s0固定起点)。
> 执行范围：本地源码、入口和历史证据清点、归档及哈希核验。未调用真实模型、SSH 或 GPU，未改变生产实现、默认生成策略或既有暂停决定。

## 1. 固定的实现身份

采集于 **2026-10-26 15:45:39（Asia/Shanghai）**。源码快照先于本报告及完成状态更新生成，避免报告与自身快照互相引用。

| 项目 | 本次记录 |
|---|---|
| HEAD | `2f2d281dc0ee7e828eeaec43dcc0b1c09e2ef736` |
| 分支 | `design/autonomous-spine` |
| 源码范围 | 全部非忽略的已跟踪及未跟踪常规文件，共 **604** 个，13,630,322 bytes |
| 已跟踪 / 未跟踪 | **508 / 96**，未跟踪实现与测试已纳入快照 |
| 工作区文件清单身份 | `61d87fdaa29b573246ae165481388eab4041d054e06a2fc3bca928605ef901af` |
| 运行时清单 | **133** 个文件，覆盖 Python 核心、runner、prompts、skills、插件运行资源及打包配置 |
| 运行时身份 | `bf0211d4c718537b4e4787d0811c43324459237ac7c5cdb59f69b5e8704a20a8` |
| 现有协议清单 | **57** 个文件；这是既有 `protocol_identity()` 范围，不等于全部运行时文件 |
| Python 核心 / DSH 插件 | `0.6.0 / 0.1.6`，两者是独立版本号 |

HEAD 不能独立复现当前实现。归档同时保存工作区相对 HEAD 的 binary diff、已暂存 diff、未暂存 diff、索引条目、状态及所有文件的字节数与 SHA256；完整源码归档补齐 Git diff 不包含的未跟踪文件。

快照不包含 `.git`、被忽略的本地配置/凭据、环境、vendored 检出、运行产物和外部依赖。它固定本仓源码，不声称单靠此包就能还原远端 H20 环境。未读取环境变量、私有配置或凭据文件；归档执行了密钥、私钥、token 与 URL 凭据形态检查，保留明确属于负向测试的 `.invalid` 合成 URL。

本地取证解释器为 Python **3.9.12**，其已安装元数据包括 NumPy **2.0.2**、pytest **8.4.2**、PyTorch **2.8.0**；没有安装 Triton 和 `operator-agent` distribution，核心按源码树读取。这些是**本地取证环境**，不是历史或当前远端 GPU 环境。

## 2. 实际入口能力清单

下表来自当前源码、函数签名、AST 与 MCP schema 检查；没有执行工具任务。完整字段与证据位置见持久目录中的 `entrypoints.json`。

| 入口 | 实际生成/验证路径 | 预算及状态 | 取消边界 |
|---|---|---|---|
| `operator_run` | `Conductor → 普通 Coder`，共享规约/技能/生成记录；控制器验证、修复/搜索及最终确认 | 默认 1800 秒本机执行时限、24 次模型调用预留、20 单位输入组验证派发预留；最终确认份额保留，标定为已声明例外。异步启动，`operator_status` 返回四态、预算和停止原因 | 无用户取消入口；本机截止监督存在，不能据此宣称远端/服务商已停止 |
| `operator_repair` | 已有源码 → Verifier → 资格检查/诊断 → Fixer → 重新验证；源码或 AST 未变则停止 | 默认最多 2 轮；模型单次 4096 输出 token、120 秒超时；后端默认 300 秒。**未接入 run 的任务总账本与整次硬截止**；同步返回及落盘 | 无独立取消；同步 MCP 通知不能中断在途调用 |
| `operator_verify` | 调用方提供源码，核对冻结契约后由 Verifier 判定，可停在 L1/L2/L3 | 后端默认 300 秒，**不是整次多操作验证的总时限**；同步返回；当前报告覆盖和源码留存缺口属于 S1 | 无独立取消；`stop_after=L1` 的局部通过不等于完整正确 |
| `operator_bench` | 调用方提供多份源码，逐臂 `verify_l3_only`；每臂与基线同窗配对 | 默认候选 cap=16，可显式配置；120 秒 deadline **只在臂之间检查**，不硬中断在途单臂；同步返回 `requested/measured/partial` | 无独立取消；结果为 exploration，不能作正确性、入库或交付裁决 |

补充核验：

- [autonomous 的 Conductor 构造](../../operator_agent/autonomous.py)显式设置 `use_coder_pipeline=False`，AST 检查确认为 `False`。本次未开启 Pipeline。
- MCP 当前公开 **7** 个工具；没有取消或单候选生成工具。`bench` 当前通过 MCP/Python 暴露，CLI parser 没有独立 `bench` 子命令。
- [MCP server](../../operator_agent/mcp_server.py)逐条同步处理请求，取消通知被忽略且不返回响应；[宿主插件](../../integrations/dsh-operator/index.mjs)对子进程命令的取消不等于后台 run 的任务取消。
- `candidate_origin` 是声明式归因，不参与裁决；不能靠该字段认定实际走过某个生成器。
- `run` 的 `done` 代表流程结束，预算停止或性能目标未达也可能结束；须结合停止原因与验证结果判读。

## 3. 历史案例清单

共登记 **15** 个案例；其中 **10** 个有可保存的原始证据集合，**5** 个目前只有文档记录。10 组证据共 **1,481** 个文件、25,749,541 bytes，已按原字节归档到持久目录；缓存文件不纳入。

“文档记录”与“原始报告存在”分开标注。本次核验的是保留字节与来源关系，没有重跑历史模型、候选或 GPU，也没有重新授予任何候选资格。

| ID | 历史任务/模型 | L2 / L3 记录 | 测量资格与限制 | 原始证据状态 |
|---|---|---|---|---|
| H01 | 10-18 HITL R1，IEEE fp32 1024³；外层模型身份未知 | 文档称 L2 过；L3 0.36x、未达标 | degraded，IEEE 与 TF32 比较语义不同 | `oa-h20` 目录为空；仅文档 |
| H02 | 10-18 HITL R2，fp16 1024³；外层模型身份未知 | 文档称 L2 过；L3 1.08x 但未取得裁决通过 | uninformative，不能记为有效性能收益 | 同上；仅文档 |
| H03 | 10-18 HITL R3，fp16 4096³；外层模型身份未知 | 文档称 L2/L3 过、约 1.06x | 文档称 valid，原始读数已不可复核 | 同上；仅文档 |
| H04 | 早期自主修复，文档称 27B、具体模型未知 | 文档称 L2/L3 过、1.05x | degraded，不具入库/进展资格 | 自动化临时目录为空；仅文档 |
| H05 | 六个种植缺陷修复；具体模型未知 | 文档称 5/6 一轮完整 L2 通过；未测 L3 | 不是总体修复成功率或性能证据 | 文档未定位原始目录；仅文档 |
| H06 | 4096³ 本地留存验证；模型/原实现身份未知 | 第 1/2 轮失败，第 3 轮 L2/L3 过、1.0561x | 第 3 轮记录 valid；共同报告可能只对应末轮 | `output/matmul4096_h20/` 已归档 |
| H07 | 普通 Coder 的 4096³ 自主运行；精确模型未知 | L2 过，L3 目标未达、1.0036x | valid，但 `all_passed=false` | `output/runs/matmul-4096-fp16/` 已归档 |
| H08 | 两份人工候选的 bench / verify | 外层候选有独立验证；bench 1.0575x / 1.0045x | 每臂 valid，bench 本身不可裁决；不是自主搜索收益 | `output/runs/matmul-4096-theirs/` 已归档 |
| H09 | 两任务真实配对，`deepseek-v4.1-flash-1017` | 两组初始 1/2、最终 2/2；L3 未测 | 诊断独有修复 0，只有 L2 证据 | 近期临时证据完整留存并已归档 |
| H10 | 固定列 stride 故障，同上模型 | 两组均修好 1/1；L3 未测 | 诊断增量修复 0；不含新生成 | 同上 |
| H11 | 十任务三候选，同上模型 | 首槽 8/10、组合 9/10；L3 未测 | 新增成功来自未知首候选；非当前生产 Pipeline 对照 | 同上，含历史源码快照 |
| H12 | 两问题任务的边界修复，同上模型 | 新首槽 2/2，六候选 3/6；L3 未测 | 固定源码人工隔离另列；不能拼成十任务 10/10 | 同上，含历史源码快照 |
| H13 | 两已知故障的诊断修复，同上模型 | 两组各 2/2；L3 未测 | 诊断独有修复 0，增量收益未证明 | 同上，含历史源码快照 |
| H14 | 有界 LayerNorm 单候选；文档标 `Qwen3.8-27B` | 主输入数值失败，suite/L3 未执行 | 没有接受候选 | `workload-codegen-mn841e1w` 已归档 |
| H15 | 有界修复原样返回；文档标 `Qwen3.8-27B` | Verifier 前拒绝，没有新 L2/L3 | `generation_failed / repair_unchanged_source` | `workload-codegen-_conj61k` 已归档 |

出处：[早期插件实验](../dsh-plugin-evaluation.md)、[bench 实测](../design/hitl-bench-sweep.md)、[两任务配对](../research/operator-quality-real-paired-20261023.md)、[固定故障配对](../research/operator-quality-seeded-repair-20261023.md)、[十任务三候选](../research/operator-quality-portfolio-20261024.md)、[边界修复](../research/operator-quality-boundary-fixes-20261024.md)、[两故障诊断](../research/operator-quality-diagnosed-repair-20261024.md)、[有界生成](../design/single-candidate-codegen-h20-acceptance.md)、[原样修复拒绝](../design/bounded-numerical-repair.md)。

机器清单 `history-inventory.json` 记录每个可定位主报告的 SHA256、schema、历史 evaluator/规约身份及模型名来源。H09–H13 的五份历史 evaluator 均不同于当前 evaluator；按原身份保留，不绕过源码漂移校验、不写回历史报表。H09/H10 没有在对应目录找到完整 evaluator 源码快照，仅有记录身份与候选/验证材料；其他记录是否足以完整重建，仍按各自清单判断。

## 4. 持久保存位置与复核

正式目录：[output/plugin_completion/20261026-s0/](../../output/plugin_completion/20261026-s0/)。它位于项目本地持久目录，**被 Git 忽略，不随提交自动备份**；S6 交付时应另行复制/打包，不能仅分享本报告链接。

| 文件 | 用途 |
|---|---|
| `workspace-source.tar.gz` / `workspace-manifest.json` | 当前全部非忽略源码及逐文件身份，含未跟踪实现 |
| `worktree-vs-head.patch` / `staged-vs-head.patch` / `unstaged-vs-index.patch` | 捕获工作区、暂存区与 HEAD 的差异 |
| `status.porcelain.txt` / `index.entries.bin` | Git 状态及索引条目 |
| `entrypoints.json` / `runtime-identities.json` | 实际入口、工具 schema、本地环境及协议身份 |
| `history-inventory.json` / `history-archives.json` | 15 案例解释与历史原始文件的逐文件清单 |
| `history-*.tar.gz` | 10 组原始证据副本；原目录未修改 |
| `capture_operator_s0.py` / `verify_s0.py` / `COMPLETE.json` | 采集逻辑、只读核验入口与完成回执 |

在仓库根目录执行：

```sh
python3 output/plugin_completion/20261026-s0/verify_s0.py --check-current-runtime .
```

该命令检查归档文件与每个成员的 SHA256，并比较当前 133 个运行时文件。文档在快照后更新不改变这份运行时身份。若 S1 等后续工作修改运行时，则应去掉 `--check-current-runtime .` 复核历史包；新执行另建新快照，不能修改 S0 记录来让当前版本“匹配”。

归档仅用于读取和历史复核。没有自动恢复预算、解封执行计划、执行候选、连接远端或重新验证性能的动作。哈希与只读文件权限提供完整性检查，不提供对同 UID 恶意改写的签名认证。

## 5. 验收与后续范围

- [x] HEAD、Git 三层 diff、96 个未跟踪文件和 604 个逐文件哈希已固定。
- [x] 四条开发入口的策略、验证、预算、状态和取消能力已记录。
- [x] `operator_run` 的普通 Coder / Pipeline 关闭状态已用 AST 与源码确认。
- [x] 15 个历史案例已区分文档记录、原始证据、L2/L3、测量资格和实现身份。
- [x] 10 组仍存在的原始证据已保存到持久目录，原始目录未修改；空目录不算完整证据。
- [x] 归档与当前运行时核验通过，伪造文件哈希与损坏归档的负向检查均被拒绝。

S0 只完成起点固定。S1 报告/源码留存、S2 取消、S3 预检误拒、S4–S6 验收/对照/交付仍未执行；[Q1 暂停决定](../design/hitl-generation-boundary.md#7-我无法确立的事)保持。
