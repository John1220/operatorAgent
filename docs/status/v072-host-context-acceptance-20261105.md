# v0.7.2 外层上下文工作流验收（2026-11-05）

本轮完成 [DSH 外层工作流](../design/dsh-host-context-workflow.md)的开发闭环：DSH 主模型取得领域材料、组装最终上下文、生成及修复候选；Python 插件冻结任务、提供材料和诊断、执行验证。当前为 Python `0.7.2.dev2` / 插件 `0.2.2-rc.3` 开发预览，工具契约 `11`，没有封版或公开发布；v0.7.1 标签与交付保持原样。

## 实现边界

- 新组合包默认 `OPERATOR_AGENT_WORKFLOW=host`，只提供 capabilities/prepare/context/verify/status/cancel。run/repair/bench 在工具发现及实际调用两层均被拒绝，独立 CLI 和 legacy MCP 保留。
- `operator_context` 返回有哈希的权威事实、规约、技能与原始角色模板，由 DSH 选择材料和组装最终 messages。模板不是已经渲染好的请求；Python 不调用内部模型。
- host 验证账本跨进程加锁、跨工具调用累计，固定预留一次独立终评。同源码需先完整 L2 通过，终评使用新的完整证据；终评开始后关闭开发。来源、契约、后端和证据漂移均拒绝，测试 seed/值不导出。
- invocation 私有绑定使监督进程确认回收后只结算自己的预留。取消和截止竞争保留原证据，拒收结果不再获得最终候选资格；额度不退款。空闲时间计入墙钟，在飞 MCP worker 由 supervisor 监督截止。
- 模型调用与 token 仍属于 DSH 主会话，产品明确标记 `not_enforced_by_plugin`；本轮没有实现或宣称任务级累计模型硬预算。测试控制器的调用上限单列，费用保持 unknown。

## 安装工件与真实主模型

所有场景在隔离 HOME/DSH_HOME、新 Python 3.12 venv 中安装 wheel/tgz，Python 使用 `-I`。宿主为 DSH `0.2.1-rc.2` / Node `24.14.1`。真实外层采用已有阿里云 OpenAI Completions 路由 `deepseek-v4.1-flash-1017`；宿主使用其已有凭据，MCP worker 不接收模型凭据或 modelRelay。

测试驱动只给任务、预算和已授权执行目标，控制超时及记录/导出。工具选择、候选源码和修复来自真实 DSH 主模型，没有用外层 HTTP fixture 代决策。原生 `llm/stream` 观察请求、用量和下一次模型请求里的工具正文；`tools/result` 保留调用顺序与完整回包。

| 场景 | 工件 | 实际主模型调用 | 结果 |
|---|---|---:|---|
| FakeBackend 从零生成及独立终评 | dev0 / rc.1 | 5 | L2 通过，内层模型调用 0 |
| 错误 NumPy 候选验证 → 反馈 → 外层修复 → 独立终评 | dev0 / rc.1 | 7 | 初始 L2 失败，修复后完整 L2 通过，内层模型调用 0 |
| H20 FP16 Softmax `4096×4096` 生成及独立终评 | dev0 / rc.1 | 5 | 主用例及 12 个 L2 case 通过，L3 未通过、uninformative |
| 取消结算补齐后的外层生成复验 | dev1 / rc.2 | 5 | L2 通过，内层模型调用 0 |
| 最终边界小修后的外层生成复验 | dev2 / rc.3 | 5 | L2 通过，内层模型调用 0 |

以上全部调用均有原生用量回执，未观察到辅助模型调用。模型调用数并非费用；主会话及服务商是否停止执行不能由本地回收推断。

领域材料在后续主模型请求中逐字匹配，所有交付检查均 `exact=true / spill=false`。dev0 的生成、修复与 H20 正文分别约 18,632 / 19,455 / 30,245 个 UTF-16 字符，低于默认宿主 spill 阈值；最终 dev2 另保留独立交付回执。材料自身限制为 40 KiB，完整 context MCP 文本限制为 48,000 UTF-16 字符；超限拒绝，不伪称完整到达。

## H20 取消、恢复和独立终评

dev1 安装 wheel 另起一个 600 秒、最多 3 次验证的任务。等本次持久 SSH 会话 ready 且预留绑定 invocation 后，通过 MCP requestId 取消：本机 `worker_reaped=true`、远端回执 `quiescent`，账本该 owner 为 `cancelled`、unfinished=0，已花预留仍为 1。

随后在同一任务中使用第二次探索预留恢复验证，再使用原先预留的 final 做独立终评。最终 reserved=3、phase=`final_complete`，没有新建任务、重置截止或退款；该场模型调用 0。候选由 dev0 的真实 DSH 外层生成，dev1 只做取消恢复及复评，不将生成归属改记到新工件。

- 源码 SHA256：`a70b54e7313e610e0d670827acfc78b299dc1f0ee3b2f6c8b1e060b5a355a82a`。
- 契约 SHA256：`8fd7e3ed178667c3e6ade373b032c98edef82454de6393edcde72bdda6b8413c`。
- dev1 最终报告 SHA256：`272ceec90ee53c0456d6bdf32b5fe673d806d64bbb5f72c5ac7db8d6ba340a79`。
- 主用例全量比对 16,777,216 个元素，最大误差 `1.91e-6`；冻结套件全部 12 个 case 通过。
- 前后核对同一 H20、驱动、运行时及 15 个 runner 文件；没有部署或替换远端环境。
- 最终读数 speedup=`3.3523`，launch 地板占 `41.8%`，`validity=uninformative / L3=false`，不得作为性能提升、长期记忆或公平 A/B 收益结论。

取消恢复脚本初次在读取完整证据的可选 `cases_total` 字段时发生 KeyError，原失败 summary 和协议保留。检查改为从完整 case 列表核对后，在原任务剩余的 final 额度内继续完成；没有删除失败或重置预算。

## 工件、回归与后续

最终开发工件 SHA256：

| 工件 | SHA256 |
|---|---|
| `operator_agent-0.7.2.dev2-py3-none-any.whl` | `17c677c251ed8ed6e9e3747d936958efce2b8b700fc411aeb3aba822eac5b3cc` |
| `operator_agent-0.7.2.dev2.tar.gz` | `988d46181631b9406538a8ee0397d61311e91621abf56502f5a60db120bafcca` |
| `dsh-operator-agent-0.2.2-rc.3.tgz` | `13e6742a8575b01af51af57dc8ee518543947290415017698b6d568855b63bf7` |

取消修复后的相关 Python 回归 **400 passed / 12 subtests passed**（35.06 秒）；最后预算参数入口小修的边界/协议/仓库守卫 **129 passed**（16.13 秒）。这两批有重叠，不相加、不称全量回归。插件 67 项通过、0 失败、0 跳过；最终工件的离线干净安装与隔离 DSH 安装/配置/卸载结果随交付清单归档。

早期受限沙箱中的进程/回环测试失败、旧八工具/配置断言失效及对应复验日志均保留。正式公开发布和 v0.7.2 封版尚未执行。后续需要补有任务身份的 DSH 模型累计预算、会话恢复及中途追加要求的专项验收；性能诊断与搜索增强仍归 v0.8。
