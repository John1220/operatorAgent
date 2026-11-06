# v0.7.2 可选自助验收

2026-11-06。适用正式本地版 Python **0.7.2** / 插件 **0.2.2** / 工具契约 **12**。日常直接描述需求即可，无需 `/op new`，见[使用指南](v072-quickstart.md)。以下详细提示词用于可选验收，不是使用前提；步骤本身不代表正式工件已完成验收。

历史 [ReLU 自助试跑](../examples/v072-relu-user-review.md)对应 rc1/rc.4；[错误候选修复](../examples/v072-repair-user-review.md)及[普通讨论与自然语言新任务](../examples/v072-natural-user-review.md)对应 rc3/rc.6，均保留原身份与结论。原生测试基线为 DSH `0.2.1-rc.2` / Node `24.14.1`，上述真实用户功能记录来自 DSH NEXT `0.2.1-rc.4`；不扩大为完整宿主矩阵或 GPU/性能验收。下列清单可用于正式安装后的独立复测。

下面三个用例都用本机 CPU FakeBackend，仍由 DSH 已选择的真实模型生成或决策，会消耗模型额度；结果始终 test_only。每个用例使用一个**新的任务目录**，明确作为独立实验。可以分别新建主会话，也可以在同一主会话顺序执行：上一项结束后直接发送下一份新 case 提示词，由 DSH 判断意图并 prepare 全新目录，自动创建独立任务。`/op new` 仅作手动备选；不能在原任务仍有未结算请求或清理未知时并行开启下一项。

## 1. 先测错误候选修复

在新主会话或上一项已结束的原主会话，使用[完整提示词 01-repair.txt](../examples/v072-test-prompts/01-repair.txt)。先把其中历史试跑目录换成未使用的新目录。

它先提交一份有意写错的 ReLU：`np.maximum(X, 0) + 1`，让第一次 L2 失败；再由外层取得该失败轮次的 repair context，自主修复，最后做同源码独立终评。

预期顺序：

```text
prepare → context(generate) → verify(explore，L2失败)
→ context(repair，绑定失败轮次) → verify(explore，完整L2通过)
→ verify(final，同修复后源码，完整L2通过)
```

重点查看：

- 第一轮确实执行错误候选并保留失败，不能把先修好后再测称为修复通过。
- repair context 绑定第一次失败的源码、契约和轮次。
- 修复后的源码哈希与错误候选不同，成功探索和 final 的源码相同。
- 完整 L2 的所有 case 均执行，没有减少 case 或放宽容差；失败的验证预留不退。
- 收尾仅从 operator 回包取得路径和哈希，不再尝试 shell；最终模型用量以结算账本为准。

该用例预算为 600 秒、4 次验证（保留 1 次 final）、12 次绑定后模型请求，通常目标路径需要 3 次验证；是否顺利完成以实际模型和回执为准。

## 2. 再测暂停与追加要求

在新主会话或上一项已结束的原主会话，直接发送 [02-steer-start.txt](../examples/v072-test-prompts/02-steer-start.txt)。本轮只 prepare、取得 generate context，随后回复 `READY_FOR_ADDITIONAL_REQUIREMENT` 并等待，不应提前 verify。

看到等待回执后，保留当前活动任务，**不要执行 `/op new`**，在同一个会话发送 [02-steer-continue.txt](../examples/v072-test-prompts/02-steer-continue.txt)，追加源码注释 `# USER_NOTE_V072_STEER`，要求先 resume，再完成生成、探索和独立终评。

重点查看：

- task_id、契约、原 deadline 及调用历史不变，没有重新 prepare 或补额度。
- 新注释确实出现在实际验证的 candidate.py 中，不能只在聊天里答应。
- 保持同一声明，完整 L2 和同源码独立 final 通过。

墙钟从 prepare 起计算，包含等待时间；本例为 600 秒，建议看到等待回执后尽快发送第二段。它先验证普通用户追加消息与 resume；真正进程重启恢复可在此路径走通后单独测试，不能把普通暂停冒称崩溃恢复。

## 3. 最后测模型预算耗尽

为本用例使用新主会话，或上一项已结束的原主会话，直接发送 [03-budget-stop.txt](../examples/v072-test-prompts/03-budget-stop.txt)。该独立用例故意把绑定后模型请求上限设为 **1 次**、累计输出预留设为 **8192 token**。唯一额度用于取得 context，下一次模型请求应由硬门拒绝。

**本例的预期是被停止，不是交付源码。** 不应要求模型在已无信用后生成总结，也不应因此加额度或换目录。

无需模型，直接查询：

```text
/op status
/op status <实际任务目录>
```

核对 DSH 的实际预算拒绝错误，以及 host-model-budget.json 的 1 条预留、剩余调用 0、输出预留 8192；不要只依据模型自然语言“我已经停止”。重复查询不能修改模型信用。此时 Python 的验证预算可能仍未消耗、phase 仍为 awaiting_host，应与模型额度一起阅读，不能只看 Python 任务相位。无参 `/op status` 查询的是会话执行/讨论阶段。

待停止轮结束后，在同一会话询问已有结果或预算含义，应能正常讨论；这会按 DSH 正常计费，但旧任务账本的调用数与输出预留仍保持 1 / 8192。讨论阶段不能继续旧任务验证。下一项独立任务直接发送新 case 要求即可，旧任务截止和消耗仍保留，不能把讨论或新任务解释为给旧任务补额度。

## 4. 同会话切换与重启检查

完成上述任一任务、原轮汇总结束后，先用无参 `/op status` 确认讨论阶段，并记录旧任务预算。普通讨论后复查旧账本不增加；再在同一主会话直接发送使用新目录的下一项任务，确认 task_id 与 run_dir 不同。旧任务的契约、截止、消耗和失败记录应保持不变。

还可在讨论阶段或 `/op new` 成功后重启 DSH，恢复同一 session，再查 `/op status`。应保留讨论或等待新任务的阶段，不应从早先 prepare 回执重新绑定已结束任务。若有未知在飞状态应明确拒绝新执行，先查询/取消；不能靠重启获得信用。候选的跨进程恢复证据见[自动入口验收](../status/v072-auto-task-acceptance-20261106.md)；本节是复测清单，不单凭步骤冒称正式包已完成崩溃恢复验收。

## 目录和结果回传

| 用例 | 历史提示词中的预选目录 |
|---|---|
| 错误候选修复 | `~/.local/share/operator-agent/trials/relu-repair-06292db7` |
| 暂停与追加要求 | `~/.local/share/operator-agent/trials/relu-steer-8d4f49c0` |
| 模型预算耗尽 | `~/.local/share/operator-agent/trials/relu-budget-8a2e549c` |

这些是历史提示词预选的目录，其中修复目录已用于上述用户验收；复测前应在提示词中换成未承载任务的全新绝对目录，并同步替换查询命令。每份提示词用于一次独立测试；已准备后，继续或调查时保留原目录与所有失败，不通过换目录给同一任务续预算。

跑完每项，提供 DSH session ID 即可按[首个试跑的复核方式](../examples/v072-relu-user-review.md)检查：实际工具顺序、失败反馈、候选哈希、独立证据、预算结算及 worker 清理。安装和日志位置见[试跑指南](v072-quickstart.md)。
