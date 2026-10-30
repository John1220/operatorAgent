# S5 有限范围结项报告（2026-10-30）

**结论：真实插件入口与受控修复功能已有证据，插件整体优于原生 DSH 的结论未建立。** 用户决定按 [GPU 有限时的优先级](../design/s5-gpu-limited-closeout-20261030.md)收尾，本轮不再扩大对照。原六任务 × A/C × 两次重复的 24 格研究仍未完成；本报告结束本轮研究投入，不授予完整对照通过资格，也不据此认定插件无效。

## 1. 已完成什么

- LayerNorm-tail 的一次真实 DSH 插件会话，经 `operator_verify` 提交到 H20 GPU2，开发反馈返回 DSH，最终候选通过独立完整 L2（12-case suite）。
- 一个预先注入的 LayerNorm 数值缺陷，经真实插件验证被拒，DSH 修改后复验通过。首跑最终源码提取失败保留；修正提取后，明确追加的一次独立 GPU 终评通过，未改写首跑。
- 工程上已有候选/契约哈希绑定、逐轮留存、调用预算、取消和未知结果判定。它们分别有功能验收，不能由此推出插件能提高模型开发成功率。
- 对照研究保留所有分母与异常。多次中断反映了工具发现、任务语义、传输和执行版本稳定性问题；不能将这些问题统称为模型写错算子。

## 2. 各批次分别保留

表中“未知”包含未执行、设施 unknown，以及历史协议下记录为 unknown 的预算内未完成。“未执行”是其子集；不与未知再相加。通过指该协议下最终完整 L2 通过，不表示性能目标达成。各行是独立计划，**不求合并正确率，不拼接不同版本或协议的成绩**。

| 批次与原始汇总 | 计划格数 | 已派发 | 通过 | 确定失败 | 未知 | 其中未执行 | 调用预留 | 主要限制 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| [历史 ABC v6](../../output/plugin_completion/20261028-s5-formal-v6/aggregate.json) | 30 | 30 | 18 | 11 | 1 | 0 | 90 | A 可读本仓、开发反馈不对称、Matmul 提示/评测维度混杂；none/8192 的旧模型配置 |
| [工作流九格 v1](../../output/plugin_completion/20261029-s5-workflow-preflight-v1/aggregate.json) | 9 | 2 | 1 | 0 | 8 | 7 | 29 | B 读到 A 候选；预算未完成与跨组污染，未放行 |
| [工作流九格 v3](../../output/plugin_completion/20261029-s5-workflow-preflight-v3/aggregate.json) | 9 | 3 | 1 | 0 | 8 | 6 | 47 | 一格预算未完成，一格 headless 输出解析设施错误 |
| [GPU2 九格 v4](../../output/plugin_completion/20261029-s5-workflow-preflight-v4/aggregate.json) | 9 | 9 | 4 | 0 | 5 | 0 | 135 | 四格调用预算未完成、一格网关链路 unknown；C 未验证候选 |
| [插件限定 A/C 六格](../../output/plugin_completion/20261029-s5-plugin-only-ac-v1/execution-v2/aggregate.json) | 6 | 3 | 2 | 0 | 4 | 3 | 25 | LayerNorm-C 终评传输 unknown；C 开发验证走公共入口 |
| [C-only 三任务诊断](../../output/plugin_completion/20261029-s5-plugin-c-diagnostic-v1/execution/aggregate.json) | 3 | 3 | 3 | 0 | 0 | 0 | 20 | 安装插件的 DSH 完成三任务，但均经公共入口验证；没有 A 配对 |
| [合作环境研究 v1](../../output/plugin_completion/20261029-s5-cooperative-study-v1/execution/aggregate.json) | 24 | 13 | 12 | 0 | 12 | 11 | 124 | 首轮 A/C 各 6/6；第二轮首格 idle timeout；C 未用插件 verify |
| [合作环境研究 v2](../../output/plugin_completion/20261030-s5-cooperative-study-v2/execution/aggregate.json) | 24 | 1 | 0 | 0 | 24 | 23 | 9 | 第二次开发验证发现实现哈希变化，独立终评 unknown；C 未运行 |

九格 v1/v3/v4 的 `executed_runs=9` 是包含未执行占位记录的历史汇总字段，本表“已派发”使用其独立 `dispatched_runs`，没有将占位记为真实会话。预算未完成保留原始状态，不能以 failed 列为零解释为“模型从未失败”。

旧 ABC v6 目录名含 `formal`，不代表满足正式对照标准。历史校准和接线中断（包括插件限定首次 `execution-v1` 在真实模型/GPU 启动前中断）仍保留在原目录，不混入上述有效执行批次。预检 v2 的冻结材料也不计为真实已执行格。

上表只收正确率口径的对照批次。另有[性能 A/C 实测](dsh-operator-s5-kernel-performance-results-20261030.md)（2026-10-30，GPU2 与 GPU7；首轮 4 格加两次 follow-up 共预留 138 次模型调用）属性能口径，**不计入上表分母**，同样 `formal_s5_credit=false`。它补充的是本项目此前缺失的两项读数：唯一一次同条件 A/C 性能配对（GPU7 LayerNorm，C 快 0.24% 即 `1.0024×`；首轮差 1.34% 即 `1.0134×`，均低于预注册的 1.05× 交付目标，且 C 终稿 SHA256 `df189e4c…e0fd` 与共同起始 seed 相同），以及历次批次中第一次真机观测到 `operator_bench`（3 次成功）与 `operator_verify` 在同一 DSH 会话内实际生效。两个口径分别保留，不合并成同一个正确率或增益结论。

## 3. 对照能说明什么

[合作环境 v1](dsh-operator-s5-cooperative-study-20261029.md)首轮 A/C 均 6/6，模型调用分别 76、46，墙钟分别 2571.281、2016.817 秒。C 六格均发现过 capabilities，但七次开发验证全部走公共 `check.py`，没有 `operator_verify`。一次重复、未完成分母及工具实际选择，使这些成本差异不足以归因于插件验证能力，不能用于“减少调用 X%”的简历表述。

[v2](dsh-operator-s5-cooperative-study-v2-20261030.md)没有再次触发 300 秒 idle timeout，但只完成首格的模型会话。第二次验证期间 `remote_executor.py` 哈希改变，底层 L1/L2 通过仍被证据检查保留为 unknown；最终验证仅保留 `ValueError` 类型。修改来源和该异常的完整根因未确定，不能声称已根治全部设施问题。

当前研究未评估内层自动生成/repair、性能调优、人工使用便利性及其他模型/宿主；不能将本轮对照外推到这些能力。用户已允许后移完整网络唯一出口与恶意候选沙箱，本次结项不再以前述要求阻挡成果交付；限制继续明确保存。

## 4. 两个可复核的插件案例

| 案例 | 可复核证据 | 已成立的事实 | 保留边界 |
|---|---|---|---|
| 真实插件入口 | [入口 audit](../../output/plugin_completion/20261029-s5-plugin-entry-acceptance-v1/execution/acceptance-audit.json)、[验收报告](dsh-operator-s5-plugin-entry-acceptance-20261029.md) | MCP 提交、GPU 反馈返回、最终候选源码身份匹配；12/12 完整 L2；5 次模型调用 | 单任务、历史版本；不能证明反馈导致了质量提升 |
| 数值错误与修复 | [首跑汇总](../../output/plugin_completion/20261029-s5-plugin-repair-acceptance-v1/execution/aggregate.json)、[追加终评](../../output/plugin_completion/20261029-s5-plugin-repair-acceptance-v1/final-recovery/summary.json)、[验收报告](dsh-operator-s5-plugin-repair-acceptance-20261029.md) | 原候选 L1 过/L2 数值失败；模型补上方差除数并经插件复验通过；恢复终评完整 L2 12/12 | 注入缺陷；首跑最终交付失败。追加终评新增模型 0、验证 1，不能改称首跑成功或自然修复率 |

S4 的 [四任务与非连续布局修复](../design/dsh-operator-s6-case-demo-20261027.md)可作 core/CLI 补充材料，性能结果为 4/4 未知；不能将这四项全部说成实际 DSH MCP 插件案例，也不能说已实现性能加速。

### 可直接复用的性能判定证据

另有 [2026-10-30 GPU7 冻结验收](dsh-operator-gpu-freeze-20261030.md)，不能因 S4 的四项性能未知而遗漏。已读取以下原始结果，核对文件 SHA256、`implementation_stable=true`、L1/L2/L3 均通过及测量资格：

| workload | 测量资格 | 原始基线类型 | 观测加速比 | 原始结果 |
|---|---|---|---:|---|
| 4096³ fp16 Matmul | valid，can_gate=true | `pytorch_eager_fp16` | 1.0569× | [result](../../output/plugin_completion/20261030-gpu-freeze/matmul4096/verify-1/result.json) |
| 8192³ fp16 Matmul | valid，can_gate=true | `pytorch_eager_fp16` | 1.0239× | [result](../../output/plugin_completion/20261030-gpu-freeze/matmul8192/verify-1/result.json) |
| 16384×6144 fp32 LayerNorm | valid，can_gate=true | `pytorch_eager` | 1.4135× | [result](../../output/plugin_completion/20261030-gpu-freeze/layernorm16384x6144-v4/verify-1/result.json) |

这些是同卡 CLI/core 的性能链路与手写候选测量，**不是 DSH+插件相对原生 DSH 的提升**，不加入 S5 的 A/C 分母。读数仅限该 GPU、形状、运行环境与基线；独立性能确认和跨卡泛化未建立。LayerNorm 大输入经三处 seed/传输修正后获得 valid，中间拒绝与取消记录继续保留，可作为具体工程问题案例。性能指标已有这份真实证据，本轮不为增加数字再重测相同案例。

该验收报告中“GPU 释放后不再新增”的表述属于 GPU7 那轮工作的结束状态，不用来推断本次 GPU2 当前安装包验收是否可执行。设备/授权/资源仍以本次 P1 的新观察为准。取消记录亦只明确本机 worker 回收；远端执行/服务商终止的回执 unknown 不提升为全部已停。

## 5. 交付与后续状态

本轮 P0 工件在 [20261030-s5-closeout-v1](../../output/plugin_completion/20261030-s5-closeout-v1/) 内生成，由主任务的最终交付清单确认状态：

- 完整实现已冻结为独立只读快照，780 文件；[源码清单](../../output/plugin_completion/20261030-s5-closeout-v1/private/source-manifest.json)绑定逐文件 SHA256。源码归档 `private/complete-source.tar.gz` 为 5,032,677 bytes。保留未提交工作区内容，不把当前 Git HEAD 当作全部交付身份。
- [历史证据清单](../../output/plugin_completion/20261030-s5-closeout-v1/private/historical-evidence-manifest.json)与 `private/historical-evidence-private.tar.gz` 保存 24,405 文件、413,398,274 原始 bytes，归档大小 156,873,522 bytes；28 个 symlink 别名跳过并留记。归档已逐文件验证。**副本位置为单份本地副本，与原件在同一文件系统；用户已于 2026-10-30 决定接受单副本，异地备份不列为待办**。已知残余风险：H20 与昆仑芯 P800 均已下线，这些读数不可再生，故单副本失效等于全部原始证据失效。原始档案可能含连接/路径元数据，须私密保存。
- 已离线构建 wheel `0.6.0` 与插件 `0.1.9`。版本号沿用现有仓库，工件由本轮 SHA256 区分；尚未发布，不能与旧同版本包互换验收。
- 当前工件在仓库外的临时干净 venv/npm prefix 离线安装通过；`pip check`、CLI capabilities、真实 MCP 握手/8 工具发现、Python/mock 完整 L2 均通过。实际 DSH runtime/SDK 配合已安装插件的宿主契约检查为 **9 passed、0 skipped**，见 [CPU 安装摘要](../../output/plugin_completion/20261030-s5-closeout-v1/installation/cpu-acceptance.json)。初次 CPU fixture 将 Python 矩阵乘法错声明为 Triton，被反作弊拒绝的记录独立保留，修正声明后通过；未修改判据或产品源码。
- [脱敏插件案例 ZIP](../../output/plugin_completion/20261030-s5-closeout-v1/shareable/s5-plugin-cases.zip)已生成，16,193 bytes、14 文件；SHA256 `e74e0e621f2fbbad87de6488b1fa43bedb2efce26c5b0991a4b7e7375929bb8d`。包含两条历史 LayerNorm 案例、四份候选源码、有限 L2/反馈投影、原始交付失败与追加终评事实。标准库核验器在仓库外 `python -I -S` 通过，相关正常/篡改拒绝测试 9 项通过；不执行候选或 GPU。只分享这个脱敏 ZIP，私有原始档案另存。
- **[P1 当前安装包真实 H20 MCP 验收已通过](dsh-operator-s5-installed-gpu-acceptance-20261030.md)**：仓库外使用已安装 MCP 命令提交一个历史通过候选，完整 L2 12/12 通过，111.481 秒，新增模型 0、验证 1。安装树与 15 个远端 runner 文件执行前后稳定，候选/契约/complete receipt 相符。这是当前工件的标准客户端入口验收，不是新的 DSH 模型会话、修复或 A/C 实验。
- 完整 A/C 重复对照后移。S6 已按用户调整后的范围完成安装、打包和演示材料准备；录制视频未做，完整对照的未完成状态原样保留。详见 [S6 验收](dsh-operator-s6-acceptance-20261030.md)。

`output/` 被 Git 忽略，交付包因此不在版本控制与公开仓内。副本位置按 §5 记为单份本地副本：用户已决定接受，不再要求复制到其它存储，「本地文件存在不等于异地保存」在这里是**已接受的决定**，不是待补项。三分钟回放及简历口径见 [演示脚本](../design/dsh-operator-s5-resume-demo-20261030.md)。

## 6. 本报告如何复核

本报告的历史结果核对未调用真实模型、GPU 或 SSH；P1 是单独冻结的新增执行，另记实际结果。离线核对了上表八份 aggregate 的 canonical 内容哈希及逐格 passed/failed/unknown、派发与未执行计数；插件限定六格、C-only 三格、合作环境 v1/v2 另外用当前离线汇总器重算，完整结果与原 aggregate 相等。入口 audit 的 `accepted=true`、MCP/反馈/终评三项检查，以及修复恢复 summary 的 `accepted=true`、`original_run_accepted=false` 已读取核对。

这证明所列数字可从本地留存记录重建，不构成对历史 GPU 事实的签名认证，也不升级原实验的因果或统计资格。原始记录未改写，异版本结果未合并。
