# 自然语言需求与自动任务入口复核

2026-11-06；会话 `session-e21a4967-bc99-493f-85db-21dd3907d8d1` 第 5、6 轮。仅检查已有日志、账本和产物，不重跑模型、不修改原任务。当前安装 Python `0.7.2rc3` / 插件 `0.2.2-rc.6`。

结论：**普通讨论与自然语言新需求的自动任务切换均符合预期。** [16 项核对](v072-natural-user-review.json)全部通过。本次是实际 DSH 用户会话及真实模型产生的调用与源码，CPU FakeBackend 执行，始终 test_only；不扩展为 GPU 或性能验收。

## 普通讨论

用户只说：“上一个实现为什么会出错？解释一下就好。”

原生事件 seq109–116 为第 5 轮：一条正常模型解释，无 tool/call、无 prepare/verify，无新任务。两项旧任务预算保持原状态：最早 ReLU 模型账本 SHA256 仍为 `c2a00716c606224b2da3fb55a7c533d6b05a025a2dec1457354c56a56d5dbda2`；上一修复任务的模型 state_hash 仍为 `096ef81d60acd69bb2bc65ec1013d01c98adad159f5be7c3b50ae4ce1460aea3`，6 条记录、lease generation=1、无新 lease history。普通讨论由 DSH 正常调用模型，不计旧 task 额度。

## 简短新需求

用户只说：“再帮我实现一个 FP32 ReLU，输入形状是 4×8，在本机 CPU 上验证正确性并给出源码，暂不测性能。”

没有指定 MCP 工具、预算、run_dir、backend=fake、独立终评流程或命令。DSH 自行执行 capabilities → prepare → generate context → verify(explore) → verify(final) → 给出完整源码。

新目录 `/Users/.../.local/share/operator-agent/trials/relu-4x8-cpu-3ad91e57`，新 task_id `5d11e4f638994fdc95e305968533846f`。DSH 选择 Python、hardware=cpu、FakeBackend 及 600 秒 / 4 验证 / 12 模型调用的预算，未向用户索要技术参数。

此前 seq63–64 的手动 `/op new` 已被上一修复任务 seq76–77 的成功 prepare 消费；回放到 seq107 及 seq116 时 authorization 均为 null。seq76 之后没有任何 command/run，新任务 prepare 在 seq128 成功，生命周期 generation 从 1 增加到 2。因此本次可以证明实际用户会话走了自动入口，不沿用上一次修复 case 的手动授权。

| 项目 | 结果 |
|---|---|
| 首轮 explore | L1/L2 通过，12/12 正确性套件，max_error=0 |
| 独立 final | 新轮次 L1/L2 通过，12/12，max_error=0 |
| 源码一致性 | 两轮 candidate.py 逐字相同，SHA256 `0ea99cb8afad343f4f995f9f4a5ff6a280f4e7d0ff61bec040e77a9a51d7c9c0` |
| 完整证据 | 两份 complete.json 的全部成员哈希通过 |
| 清理 | 两条监督调用均 done，worker_reaped=true |
| 模型 | 新任务共 6 次主模型请求，其中 2 次在绑定前准备，4 次绑定后（含总结），均结算；绑定后输出预留 32768，观察到 API 输出 3551 token，费用未知 |
| 原任务 | 两项旧任务记录保留，新旧 task_id 均不同，没有继续旧任务或补额度 |
| 工具与范围 | 仅 host 算子工具；没有 shell、文件工具、子代理、GPU/SSH 或 L3 性能测量 |

## 输出体验仍可完善

流程层面无需用户提供详细验收脚本，但最终回复仍展示较多 ABI、后端、套件、预算术语。日常回复可优先给源码、验证结果与报告链接，将内部执行细节留给日志或用户追问。

解释旧错误时，模型关于容差的表述有误：“正数侧 atol 先被击穿”不足以判定失败。此次实际 verifier 的规则是绝对误差同时超过 atol 与相对容差才失败；对足够大的正数，`+1` 未必超相对容差。本次原始 32 元素全部失败有真实回执支持，模型的泛化解释不能替代该回执。该说明问题不影响本次两轮的实际验证结果。

## 查看产物

- [最终源码](/Users/.../.local/share/operator-agent/trials/relu-4x8-cpu-3ad91e57/verify-2/candidate.py)
- [独立终评报告](/Users/.../.local/share/operator-agent/trials/relu-4x8-cpu-3ad91e57/verify-2/report.md)
- [模型账本](/Users/.../.local/share/operator-agent/trials/relu-4x8-cpu-3ad91e57/host-model-budget.json)
