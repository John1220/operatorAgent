# v0.7.2 用户自助 ReLU 试跑复核

复核日期：2026-11-06。会话 `session-e21a4967-bc99-493f-85db-21dd3907d8d1`，DSH NEXT 宿主 `0.2.1-rc.4`，已安装候选 Python `0.7.2rc1` / 插件 `0.2.2-rc.4`。本次仅检查已有记录，没有重跑任务、调用模型或改写任务账本。

结论：**CPU host 功能闭环符合预期，存在两处外层行为/汇总问题。** 不将其扩大为 GPU、性能、成本收益或整个新宿主组合的完整兼容验收。

## 已核对的结果

| 项目 | 实际结果 |
|---|---|
| 声明 | ReLU、fp32、`[4,8]`、Python、`backend=fake`、停在 L2 |
| 工作流 | prepare → generate context → explore verify → 同源码 final verify |
| 源码归属 | 会话中两次 verify 的 code 均与对应 candidate.py 哈希相同；由外层提交，candidate_origin=outer |
| 探索验证 | L1/L2 通过；主用例 32 元素全部比较，最大误差 0，完整 CorrectnessSuite 12/12 通过 |
| 独立终评 | 新的验证轮次和完整证据，L1/L2 及 12/12 case 通过；两轮源码逐字相同 |
| 证据完整性 | 两轮 complete.json 列出的所有成员哈希通过；源码、契约、binding 和任务身份一致 |
| 状态与清理 | final_complete；两条监督调用均 done，worker_reaped=true，未超原截止 |
| 模型预算 | 成功绑定后 5/12 次，全部 answered；输出预留 40960/98304，剩余 7 次与 57344 token 预留；该部分 API 输出计数 4077，费用未知 |
| 主模型总量口径 | 会话有 6 条主模型回答，其中首次准备请求发生在绑定前；另有 1 条辅助标题请求事件，不计任务主模型预算 |
| 测试性质 | FAKE/test_only=true；gpu_acceptance=false、performance_evidence=false，没有执行 L3 |

上下文的持久工具回包与 contexts 中保存的材料相同，回包无 spill 标记。普通用户会话没有验收 observer 的实际 provider 请求观察记录，因此本次不宣称已重新证明材料在服务商请求中逐字完整到达。

## 两处需要完善的表现

1. **收尾多尝试了一次 bash。** 外层在独立终评后尝试用 shell 列目录、核对哈希，违反了本任务只用七个 operator 工具的要求。持久工具回包为 isError=true，明确被任务 guard 拦截，没有执行该 shell。这证明工具边界生效，但外层行为仍可收敛以避免浪费模型回合。
2. **自然语言总结的预算是旧快照。** 总结写“模型剩余 9、输出预留剩余 73728”，来自 final 工具回包时的 3 条模型记录。其后还有一次产生 bash 意图的主模型调用和最终汇总调用，均已记账。最终应以 host-model-budget.json 的 **5 条、剩余 7、输出预留剩余 57344** 为准。预算硬门没有漏计；待改进的是汇总展示口径。

独立终评已关闭任务开发，账面剩余的两次验证不重新授予开发资格。原任务可查询证据，后续独立实验需明确新目标、新会话与新目录。

## 查看实际产物

- [最终源码](/Users/.../.local/share/operator-agent/trials/relu-9ede13ba/verify-2/candidate.py)
- [最终报告](/Users/.../.local/share/operator-agent/trials/relu-9ede13ba/verify-2/report.md)
- [最终执行日志](/Users/.../.local/share/operator-agent/trials/relu-9ede13ba/verify-2/execution.log)
- [最终完整证据](/Users/.../.local/share/operator-agent/trials/relu-9ede13ba/verify-2/result.json)
- [模型账本](/Users/.../.local/share/operator-agent/trials/relu-9ede13ba/host-model-budget.json)
- [验证账本](/Users/.../.local/share/operator-agent/trials/relu-9ede13ba/host-task.json)
- [24 项核对结果 JSON](/private/tmp/operator-v072-user-review.f_s1__23/review.json)

任务 ID：`7eff510b8c274d50ad51ed8e8a721977`。契约 SHA256：`f29dd76ad83821486bb5d9972e80502e90bc92762810e8e5dabc073ccef1b9f3`。源码 SHA256：`b9f754d617fd26d58e070dd2291a4a3837d2ae301560188ad1c63942ad570b19`。
