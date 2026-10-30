# 当前安装工件的 H20 MCP 验收（2026-10-30）

状态：**通过。** 按 GPU 有限收尾计划，完成一次仓库外已安装工件的真实 MCP → H20 完整 L2 验收。新增模型调用 **0**，完整验证 **1**，墙钟 **111.481 秒**。没有重新生成候选，没有补跑 A/C，没有做 L3 或性能调参。

## 工件与入口

- 从独立只读的完整源码快照构建；源码快照 780 文件，当前 wheel 实际安装的产品文件 134 个，协议身份 73 文件，远端部署集合 15 文件。逐文件核对安装内容与源码快照一致，执行前后再次检查安装身份。
- wheel `operator_agent-0.6.0-py3-none-any.whl`，SHA256 `8cd75a3864d7a300fd71f33360f81c3ba92d240b8889cb16289a525497898906`。
- 插件 `dsh-operator-agent-0.1.9.tgz`，SHA256 `881e8384598a3ff360b16eea901eed56ed31cb9cbeb6c69c604239bb5ecb6af6`。两包版本沿用仓库，通过内容哈希区别于此前同版本构建；未发布到包仓库。
- 临时干净 venv/npm prefix 离线安装，在仓库外运行；不继承系统 site-packages，不提供 `PYTHONPATH`。依赖检查、CLI/mock 完整 L2、MCP 初始化/8 工具发现通过；已安装插件配合实际 DSH runtime/SDK 的宿主契约检查 **9 passed、0 skipped**。
- 本次使用 DSH 随附的标准 MCP SDK，启动插件配置指定的已安装 `operator-agent-mcp` 命令，真实调用 `operator_prepare` 与一次 `operator_verify`；不导入仓库 core 执行验证。它是安装工件的 MCP 入口验收，没有启动 DSH 模型会话，不冒充模型主动选择插件或新的修复实验。

## 固定输入与结果

任务为 `layernorm fp32 triton`，shape `[7,33]`，沿用生产完整 L2；候选复用历史真实入口验收最终源码，不由本次模型生成。

| 项 | 结果 |
|---|---|
| 候选 SHA256 | `df189e4c0ab4bef63000147d5ad2f5844b805ea1797c099df21dd2edce69e0fd` |
| 契约 SHA256 | `96cdb38deafce9282485b5beaa555a738e573a1c0892292554b9d7f70685c7a7` |
| L1 / 完整 L2 | 通过 / **12/12 通过** |
| 主输入最大绝对误差 | `2.384185791015625e-7` |
| 输入保真 | 主输入及各 suite case 通过 |
| 实现身份 | 安装树前后稳定、证据 `implementation_stable=true` |
| 远端 runner | 15 文件执行前后完整 SHA256 匹配；仅使用现有环境，不部署覆盖 |
| GPU | H20 GPU2，UUID `GPU-13c6c4b4-45ce-8963-1122-df2b3d72fb76` |
| 资源 | 执行前后均 22246 MiB 空闲、利用率 0%，七个原有已知进程；共享卡，无独占租约 |
| runner 环境 | Linux x86_64，Python 3.10.12，torch 包元数据 2.14.0、triton 3.8.0、NumPy 2.2.6 |

执行前冻结计划 `211546990762e85f80c2f28b55ec876b06f4d4d6f12a04f735481435f8094263`。最多一次验证，验证 timeout 600 秒，外围客户端 630 秒、外层进程 660 秒；不增加调用额度，不自动重试。MCP 回包、完整判定及 complete receipt 均保留，候选/契约/回执身份核验通过。

## 证据与结论

目录：[本轮交付](../../output/plugin_completion/20261030-s5-closeout-v1/)。其中 `installation/` 保存安装检查、实际宿主契约检查及一次性驱动；`p1-gpu/` 保存冻结计划、MCP 请求/反馈、设备/环境快照、复制到持久目录的完整任务证据和 [summary](../../output/plugin_completion/20261030-s5-closeout-v1/p1-gpu/summary.json)。summary SHA256 为 `0d76068831d294cc3709c78c87c739eb5ab770112e1a81226ba41ac86c6b143b`，最终证据 SHA256 为 `e4b513b2572a72e5ba34a670c9826fcc8c08dc84638e6d6024373453e9f9d55c`。

本次填补“当前可安装工件在真实 GPU 上可用”的缺口，不证明插件比原生 DSH 更好，不增加历史 24 格分母，不覆盖此前版本未知结果。没有新性能数字；已有 CLI/core 性能证据按其原报告引用。资源查询只是结束时观察，不升级为所有远端/服务商后代已终止的证明。
