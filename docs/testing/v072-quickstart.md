# v0.7.2 自助使用指南

更新：2026-11-06。正式本地版身份为 Python **0.7.2** / DSH 插件 **0.2.2** / 工具契约 **12**，未向 PyPI/npm 公开发布。下面给出正式版安装目标与使用步骤；历史候选的安装成功记录不能代替正式工件的安装回执与实际 UI 核对。

## 1. 安装并核对版本

本机正式版目标目录为 `~/.local/share/operator-agent/v0.7.2/`。安装完成后，Python 3.12 环境应提供：

```text
~/.local/share/operator-agent/v0.7.2/venv/bin/operator-agent
~/.local/share/operator-agent/v0.7.2/venv/bin/operator-agent-mcp
```

DSH Desktop 的安装与配置分两步：

1. 在 **DSH → 插件 → 添加插件 → 安装第三方插件** 从本机正式目录复制 tgz 的实际绝对路径，替换下面占位符后输入，安装后启用（GUI 不展开 `~` 或 `$HOME`）：

   ```text
   /absolute/path/operator-agent/v0.7.2/artifacts/dsh-operator-agent-0.2.2.tgz
   ```

2. 安装后运行配置助手，备份原配置并写入两处 Python 绝对路径：

   ```bash
   "$HOME/.local/share/operator-agent/v0.7.2/examples/configure-v072.command"
   ```

   追加 `--dry-run` 可只预览。配置检查通过后重开 DSH，在插件页确认 `0.2.2` 启用且两个组件运行中，再直接输入 `/op capabilities`，核对 Python `0.7.2` 与工具契约 `12`。配置检查不能代替实际加载验证。

**Desktop profile 由 Electron 应用独占管理，普通 CLI 不能安装或导出该 profile；退出应用也不能解除限制。** Desktop 必须先用 GUI 安装，再用助手的 `--configure-only` 配置。上面的快捷助手已包含该参数。自定义非 desktop profile 可由[安装助手](../../benchmarks/delivery/install_dsh.py)的 `--profile` 完成安装和配置。

助手保留原 YAML 内容并维护两条覆盖记录；JSON patch 保持其他条目语义。原件和回执保存在私有备份目录。检查失败会恢复原 patch，但不代表已安装的插件包已自动卸载。换机器时需重新安装并替换本文绝对路径。

DSH 中还需有能正常聊天的模型。默认 host 模式使用 DSH 当前模型，无需 Python `LLM_API_KEY` 或 `modelRelay`。本机 CPU 验证无需 GPU、Torch/Triton 或 SSH；FakeBackend 的执行结果始终为 `FAKE / test_only`，外层真实模型仍正常消耗模型额度。

## 2. 直接描述需求

在 DSH 主会话中发送：

```text
帮我实现一个 FP32 ReLU，输入形状是 4×8，
在本机 CPU 上验证正确性并给出源码，暂不测性能。
```

DSH 会选择执行参数与全新任务目录，取得领域材料，生成源码、验证并完成同源码独立终评。日常使用无需手写工具顺序、预算或任务目录；若缺少影响任务的条件，DSH 应先澄清。

上一项结束后，可在**同一主会话**直接发送下一项独立需求，无需 `/op new`。例如“再帮我验证一个新的 FP32 ReLU，输入形状改为 8×16”。解释已有结果仍是普通讨论；含糊的“继续”应先澄清。每轮最多成功开启一项新任务，旧任务截止、预算与证据不重置；未结算请求、在飞验证或未知清理状态会阻止切换。`/op new [新的绝对目录]` 保留为用户直接输入的手动入口。

这条自然语言入口已有 DSH NEXT `0.2.1-rc.4` 上的[真实用户复核](../examples/v072-natural-user-review.md)，该记录对应 Python `0.7.2rc3` / 插件 `0.2.2-rc.6`，保留候选身份，不冒称正式工件已重新执行同一任务。

## 3. 怎么判断跑通、查看结果

本例的成功条件是探索验证与独立终评均完整 L2 通过、源码相同，最终 `phase=final_complete` 且 `test_only=true`。它不要求 L3，也不能得出 GPU 性能结论；工具 `ok=true` 不能单独代替正确性通过。

可让 DSH 展示源码、验证结果与报告链接。无需模型时，直接输入：

```text
/op status
/op status <实际任务目录>
```

无参命令查看会话处于执行、讨论或等待新任务的阶段；带目录命令查看该任务预算、候选/报告路径与在飞实例。模型汇总可能引用结算前快照，最终用量以 `host-model-budget.json` 为准。

| 任务目录内的路径 | 内容 |
|---|---|
| `declaration.json`、`contract.json` | 原任务与冻结判据 |
| `contexts/*.json` | 交给 DSH 的领域材料 |
| `verify-N/candidate.py` | 该轮实际验证源码 |
| `verify-N/execution.log`、`result.json`、`report.md`、`complete.json` | 执行日志、判定与完整证据 |
| `host-task-policy.json`、`host-task.json` | 固定预算、验证消耗、终评资格和状态 |
| `host-model-budget.json` | 绑定后的主模型请求与输出预留 |
| `tool-calls/` | 监督调用、取消与 worker 清理记录 |

DSH 对话和工具卡可查看外层过程。历史验收的 `events.jsonl`、`tool-results.jsonl` 来自额外 observer，普通 GUI 交互不会自动生成同名文件。需要 CLI `--json` 日志时应另建隔离 headless profile，不能用普通 CLI 启动 desktop profile。

## 4. 追加要求、停止与下一项

- 活动任务中可直接追加源码要求；保持冻结契约时，DSH 用 `operator_resume` 核对原任务并继续原预算。墙钟包含等待时间，不因暂停、重启或恢复延长。
- 独立终评后的原轮汇总仍计入旧任务；原轮结束或预算停止后进入讨论。普通讨论按 DSH 正常计费，不写旧任务模型账本，也不重新开放旧任务开发。
- 模型额度耗尽后仍可用 `/op status` 取回源码和报告路径。取消在飞验证使用 `/op cancel <run_dir> <invocation_id>`，随后查询终态与 worker 清理；停止模型生成和取消验证需分别核对。
- 改变 shape、精度等冻结条件属于独立任务，直接表达新需求即可；不能通过新目录给旧任务补额度。重启同一 session 也不恢复已花额度。

## 5. 可选验收与历史证据

想检查失败修复、用户干预和预算耗尽，可按[后续测试指南](v072-next-cases.md)使用详细验收提示词；这些脚本是测试选项，不是日常使用前提。

已完成的候选验收分别保留其原身份：[rc1 ReLU 用户试跑](../examples/v072-relu-user-review.md)、[rc2 生命周期验收及安装回执](../status/v072-session-lifecycle-acceptance-20261106.md)、[rc3 自动入口验收](../status/v072-auto-task-acceptance-20261106.md)、[rc3 错误修复](../examples/v072-repair-user-review.md)与[自然语言入口](../examples/v072-natural-user-review.md)。旧安装目录和回执用于审计与回退，不作为正式版当前路径。

原生集成测试以 DSH `0.2.1-rc.2` / Node `24.14.1` 为基线；DSH NEXT `0.2.1-rc.4` 已有上述用户功能闭环，不扩大为完整宿主兼容矩阵。CPU 流程跑通后，如需已授权的 GPU/SSH 测试，可参考历史 [H20 实例](../examples/v072-h20-softmax.md)并先准备匹配的主机与运行时；不能只改后端名称便声称具备真机环境或性能资格。完整安装说明见[插件 README](../../integrations/dsh-operator/README.md)，支持边界见[支持清单](v072-support.md)。
