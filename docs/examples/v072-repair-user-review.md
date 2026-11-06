# v0.7.2 用户修复 case 复核

复核日期：2026-11-06。会话 `session-e21a4967-bc99-493f-85db-21dd3907d8d1` 的第 4 轮；本机候选 Python `0.7.2rc3` / 插件 `0.2.2-rc.6`。仅核对原生会话、源码、完整证据及账本，没有重跑模型、验证或修改任务。

结论：**指定的失败→反馈→外层修复→复验→同源码独立终评功能闭环符合预期。** [21 项机器核对](v072-repair-user-review.json)全部通过；自动入口与自然语言体验的证据边界另见下文。

任务目录：`/Users/.../.local/share/operator-agent/trials/relu-repair-06292db7`。新 task_id 为 `2ce7aae534e8489a95a3a9469e1ff9b5`，与旧任务独立；ReLU/fp32/[4,8]/Python 契约与旧任务相同，没有因本次修复修改判据。

| 轮次 | 目的 | L1 | L2 | 实际结果 |
|---|---|---|---|---|
| 1 | explore，故意错误候选 | 通过 | 失败 | 原样提交 `np.maximum(X, 0) + 1`；主输入 32/32 错，最大误差 1；12 个扩展 case 未运行，不能说它们各自失败 |
| 2 | explore，修复后 | 通过 | 通过 | 删除错误偏移，输出按运行时 shape/dtype 分配；最大误差 0，完整套件 12/12 |
| 3 | 独立 final | 通过 | 通过 | 与第 2 轮源码逐字相同，新验证轮次，完整套件 12/12 |

实际顺序为 prepare → generate context → verify(explore 失败) → repair context(轮次 1) → verify(explore 通过) → verify(final) → 汇总。三次工具提交的 code 与落盘 candidate.py 逐字相同；三份 complete.json 的所有成员哈希通过；三条监督调用均 done、worker_reaped=true，验证均在原截止前完成。未使用 shell、文件工具、子代理或 legacy run/repair/bench。

首轮源码 SHA256：`08abe3ce713fdb99d1a08f56b73f2daaec3c542debe17f8f9e309d932016f299`。修复与终评共同源码 SHA256：`9e1f490149ecc74ca07748cbdafe9ca5e3ef88585139935b182521aa9c85986d`。

## 诊断与分工

repair context 的 candidate、feedback 与 diagnosis 都绑定首轮源码。插件提供 L2 数值反馈（bad_elements=32、max_error=1、输入保真已核对）及 debug_fix 路由；专用确定性诊断器返回 `unknown / unsupported_task_contract`、findings=[]。因此不能称为诊断器自动识别了 ReLU 偏移根因：根因定位及代码修复来自 DSH 外层对反馈和源码的分析。host context 记录 model_calls=0、context_owner=host。

该 unknown 未被当作通过，实际修复仍经过独立 verifier；这符合既定分工。当前诊断器对这一 ReLU 错误没有专用覆盖。

## 最终预算与任务保全

- 验证使用 3/4 次，分别为 2 次 explore、1 次 final；已完成独立终评，账面剩余验证不重开旧任务开发。
- 绑定后模型调用 **6/12**，全部 answered，剩余 6 次；输出预留 **49152/98304**，剩余 49152。观察到的 API 输出 token 合计 3311，费用未知。
- 新轮共 7 条主模型回答，首次 prepare 决策在绑定前，故不在上述 6 条中。最终总结包含在 6 条内。
- 自然语言汇总明确声明用量为总结前快照：5 条、剩余 7 次；本次最终账本多计一次总结，口径正确。
- 旧任务 `relu-9ede13ba/host-model-budget.json` SHA256 仍为 `c2a00716c606224b2da3fb55a7c533d6b05a025a2dec1457354c56a56d5dbda2`，与升级前相同，未补额度或改写。
- 全程 FakeBackend / test_only，无 GPU/SSH、L3 或性能收益结论。

## 这次没有单独证明的体验

1. **完全无命令自动入口。** 本轮消息到 prepare 之间没有新增 `/op new`，但原生事件 seq63–64 已存在此前成功的用户 `/op new`，授权一直保留到本轮 prepare。故实际运行也满足手动入口条件，不能将它单独作为“从未执行命令也自动切换”的验收。该自动路径已有本地 provider 集成测试；真实服务商下需在本轮完成后的讨论阶段，直接发下一项任务且不先执行 `/op new` 再验证。
2. **自然语言需求即可使用。** 本轮仍使用明确指定工具、预算、故意错误源码和流程的详细验收提示词，不能替代简短自然语言需求测试。

## 产物

- [失败报告](/Users/.../.local/share/operator-agent/trials/relu-repair-06292db7/verify-1/report.md)
- [修复后的源码](/Users/.../.local/share/operator-agent/trials/relu-repair-06292db7/verify-2/candidate.py)
- [独立终评报告](/Users/.../.local/share/operator-agent/trials/relu-repair-06292db7/verify-3/report.md)
- [模型账本](/Users/.../.local/share/operator-agent/trials/relu-repair-06292db7/host-model-budget.json)
