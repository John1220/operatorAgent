# v0.7.2 会话生命周期修正验收

日期：2026-11-06。候选 Python `0.7.2rc2` / DSH 插件 `0.2.2-rc.5`，工具契约 `12`。仍为本地候选，不据此声称 v0.7.2 正式封版。

## 问题与修正

原用户 ReLU 任务已完成独立终评，但同会话后续消息仍被原任务的 600 秒截止拦截，报 `Task wall clock exhausted before model dispatch.`。预算应约束任务执行范围，不能覆盖会话的全部寿命。

- final 返回后，原轮模型汇总继续计入旧任务，直到原生 `turn/end` 才进入结果讨论。
- 到期或模型额度耗尽时停止当前执行轮，下一轮允许普通讨论。普通讨论由 DSH 正常计费，只允许只读算子工具，不恢复旧任务执行额度。
- 用户直接输入 `/op new [run_dir]` 后，可在同一主会话 prepare 下一项独立任务；`/op status` 无参查询阶段。两个命令不调用模型。
- 授权来自原生用户 `command/run` 与成功 `command/done` 的配对；模型文字、工具回包及继承的分支事件不能授予权限。成功 prepare 消费授权，目录不可复用。
- 重启后从原生事件恢复阶段。讨论时只读加载旧模型账本，不重绑 lease、不更新时间锚点。未知 prepare 结果、在飞模型/验证或 worker 未回收时拒绝切换。

实现入口：`integrations/dsh-operator/host-session-lifecycle.mjs`、`host-model-budget.mjs`、`host-model-ledger.mjs`、`index.mjs`；技术边界见[设计说明](../design/dsh-host-context-workflow.md)。

## 回归证据

| 检查 | 结果与范围 |
|---|---|
| Python 相关回归 | 157 passed；host workflow、预算、resume、MCP、安装助手与 hygiene。日志 `/private/tmp/operator-v072-lifecycle-python-complete.log` |
| 插件源码回归 | 138 passed / 0 failed / 1 skipped；新增未确认 prepare 用例通过。唯一跳过为未开启的 legacy relay 集成；日志 `/private/tmp/operator-v072-lifecycle-plugin-final.log` |
| 原生 DSH 生命周期 | 真实 DSH + 本地 HTTP provider，无服务商凭据或 GPU：同会话任务 A→讨论→原生 `/op new`→跨进程任务 B；未授权 prepare 拒绝；耗尽后讨论及旧 verify 拒绝；旧账本字节保持不变。包含在上述通过项中 |
| 审查 | 独立只读审查未发现额外预算重置或讨论态执行绕过；补上未知 prepare 的切换拒绝及重启用例 |

初步候选的离线 wheel 安装、`pip check`、FakeBackend 安装冒烟和独立 DSH profile 安装/组合/卸载均通过。最终包验证和个人 DSH 升级结果在执行后追加；保留初步包身份，不用源码通过替代最终包验收。

该轮未重跑 Python 全量，也未新增真实付费模型或 GPU 算子测量。rc1 的完整回归与真实算子历史证据见[原候选验收](v072-budget-resume-acceptance-20261106.md)，不能直接写成 rc2 的全量复验。

## 用户任务保全基线

旧 session：`session-e21a4967-bc99-493f-85db-21dd3907d8d1`。旧任务目录 `/Users/.../.local/share/operator-agent/trials/relu-9ede13ba`；升级前 `host-model-budget.json` SHA256：`c2a00716c606224b2da3fb55a7c533d6b05a025a2dec1457354c56a56d5dbda2`，共 5 条模型预留记录。

上述用户文件仅用于核对原任务未被重置；不修改历史报告中的原版本、调用数或失败记录。


## 最终候选与本机升级结果

最终构建快照保存在 `/Users/.../op-agent-delivery/v0.7.2rc2-lifecycle/candidate/`；回归日志、包安装/卸载回执及本机核对保存在同目录 `evidence/`。快照中的指南及本记录截至构建前；安装后补充以本文及独立回执为准，不修改已验工件。

| 最终工件 | SHA256 |
|---|---|
| `operator_agent-0.7.2rc2-py3-none-any.whl` | `cc4e0f7f367b860cc57b0e04542c81696d771b662b12eb5500664863747cb647` |
| `operator_agent-0.7.2rc2.tar.gz` | `95a923eb7c0493f35ea1642163506d6b3d1a0805307dd13d3ac7da9ea5f2383e` |
| `dsh-operator-agent-0.2.2-rc.5.tgz` | `07d6e4d5e4d1a3e996383059677298046ca46221071e5450439d58fd05c0b77b` |

- 最终包插件回归 **139 passed / 0 failed / 0 skipped**，包含原生 DSH 生命周期、模型预算、legacy relay 集成；测试 provider 在本地，不使用付费凭据。原日志 `/private/tmp/operator-v072-lifecycle-plugin-packaged.log`。
- 最终 wheel 干净安装、`pip check`、FakeBackend 修复/终评/取消回收通过；独立 DSH profile 离线安装、完整配置组合及卸载通过。`final-smoke/smoke-result.json` 与 `final-profile/manifest.json` 分别留证。
- 个人 DSH NEXT 通过原生插件管理更新到 `0.2.2-rc.5`，两个组件运行中；配置改指向独立 rc2 Python，rc1 环境保留。配置原件与回执：`/Users/.../.dsh/operator-install-receipts/desktop-asqjn7ay/`。
- 在原用户 session 中执行无模型命令 `/op status`，实际显示“结果讨论”；`/op capabilities` 显示 Python `0.7.2rc2`、tool `12`。旧模型账本仍为上述 SHA256、5 条记录，逐字未变。
- 本机仅执行安装、状态及能力核对，没有代用户运行新的真实模型 case，也没有自动执行 `/op new` 授权任务。

下一步用户可在原会话直接 `/op new`，看到成功回执后发送下一条新任务提示词；必须使用全新任务目录。`/op new` 不延长旧任务，讨论仍由 DSH 正常计费。
