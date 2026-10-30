# DSH 算子性能优化 A/C 首轮实测（2026-10-30）

本轮是有限 pilot，不是 S5 正式分母，也没有完成预注册的 8 格、两次 L3 确认。当前证据不支持“插件整体优于原生 DSH”。总体设计与停止规则见[实验计划](../design/s5-kernel-performance-comparison-20261030.md)，GPU 起点和 readiness 见[预检记录](dsh-operator-s5-kernel-performance-preflight-20261030.md)。

## 执行摘要

用户确认 GPU7 可用后，两类 workload 并行运行：LayerNorm A/C 固定在 GPU2，Matmul A/C 固定在 GPU7。Matmul A 格在 24 次模型调用预留耗尽后没有终稿；按预注册停止规则，后续 repeat-2 不派发。

| Workload / 组 | 模型调用预留 | 插件 `operator_verify` | 插件 `operator_bench` | 最终 L2 | 候选 median | PyTorch 基线 median | 相对 PyTorch | 独立 L3 |
|---|---:|---:|---:|---|---:|---:|---:|---|
| LayerNorm A | 18 | 不适用 | 不适用 | 通过 | 0.229248 ms | 0.320608 ms | 1.3985× | valid，195 对样本 |
| LayerNorm C | 20 | 3 次成功 | 0 次 | 通过 | 0.226208 ms | 0.320960 ms | 1.4189× | valid，191 对样本 |
| Matmul C | 19 | 0 次成功，2 次拒绝 | 0 次 | 通过 | 1.050592 ms | 1.036640 ms | 0.9867× | valid，47 对样本 |
| Matmul A | 24，预算耗尽 | 不适用 | 不适用 | 仅 1 次开发候选 L2 通过；无最终提交 | 无 | 无 | unknown | 未运行 |

LayerNorm 的单次最终读数给出 `A latency / C latency = 0.229248 / 0.226208 = 1.0134×`，即 C 快约 1.34%。这低于计划的 1.05× 交付目标，而且 A/C 使用的计划哈希不同，因此只能作为方向性 pilot 观察，不能算严格同版配对结论。

Matmul C 的候选通过完整 L2，但独立 L3 比 PyTorch 基线慢约 1.33%。A 没有最终候选，不能计算 A/C 性能差异。没有 workload 完成两次独立 L3 确认。

## 插件路径

LayerNorm C 的 relay 记录了 3 次已转发的 `operator_verify`，提交 3 个不同候选；3 次验证的 GPU 身份和进程集合检查通过。期间另有 2 次请求被拒绝，之后模型修正路径并成功使用验证工具。C 没有调用 `operator_bench`，所以这轮观察到的是正确性反馈，不是插件 benchmark 反馈。

Matmul C 的 DSH 转录有两次 `operator_verify` 调用尝试，但 relay 均拒绝，`verify_calls=0`、`candidate_hashes=[]`。两次请求都把 `run_dir` 指向工作目录 `/work`，而冻结任务目录是 `/work/task`；第一次还显式指定了 `backend=mock`。DSH 收到的错误文本只有 `Frozen development surface refused this request`，没有提供可修正的具体字段，因此模型没有纠正请求。GPU 前后快照只证明资源状态稳定，不代表验证执行。Matmul C 的 L2/L3 是外层 supervisor 对最终候选做的验证与测量，不能归功于插件反馈。

## Runner 与实验偏差

首个 GPU7 C 启动在模型派发前因 `TypeError` 退出，模型调用数和 GPU 执行数均为 0。原因是性能 C 不挂载公共验证客户端，但启动配置仍读取该客户端的 timeout/config。随后将 MCP timeout 显式绑定到 relay timeout，并允许无公共客户端的隔离配置。

GPU2 LayerNorm A 在修复前计划下运行，18 次模型调用后生成候选并通过最终 L2；旧 runner 随后因性能记录的 `development_audit=None` 崩溃，未写入外层 slot record 和对应的 slot 后置快照。保留的最终 L2 记录与候选被恢复，随后单独冻结并执行了一次合格的独立 L3。该 A 结果的 L3 有自己的完整资源快照，但模型格没有完整外层收尾证据；C 使用修复后的计划哈希，因此这对结果不是同一实现快照下的正式配对。

修复后，GPU2 LayerNorm C 完成 20 次调用预留并通过 L2；GPU7 Matmul C 完成 19 次调用预留并通过 L2/L3。GPU7 Matmul A 用满 24 次调用预留，只有一次开发 L2 通过，无最终答案，结果为 `call_budget_exhausted/unknown`。因此停止后续格，没有自动重跑或加预算。

4 个真实模型格共预留 81 次调用。所有调用的 provider usage 字段都缺失，账单金额未知；网关诊断没有记录 upstream transport failure。DSH transcript 中的 usage 不是服务商账单证据。所有记录均为 `formal_s5_credit=false`。

## LayerNorm 定向配对

首轮结束后，追加一次 GPU7 LayerNorm repeat-1 A/C 定向配对，计划 hash `98e1f4b5c52835e417201ef7286e465c3fca88f7835183ab6e60801f60548c15`。同一模型、seed、契约与代码快照，执行顺序为 C→A。该 follow-up 独立于原 8 格计划，不能补成原计划的完整分母。

此前一次路由尝试向 relay 传入 `/work/task`，宿主 relay 实际要求参与者工作区内的相对路径 `task`。因此工具请求被拒绝，C 外层候选虽然通过 L2，插件成功工具数仍为 0；该格保留为 unknown，没有启动 A。模型共预留 18 次调用，provider usage 缺失。随后修正路径说明与 relay 工具面并重新冻结新计划。

| 组 | 模型调用预留 | `operator_bench` | `operator_verify` | 最终 L2 | 独立 L3 候选 median | PyTorch 基线 median | 相对 PyTorch | L3 样本 | 有效性 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C：DSH + 插件 | 20 | 3 次成功 | 1 次成功 | 通过 | 0.227744 ms | 0.321408 ms | 1.4113× | 193 | valid，can gate |
| A：原生 DSH | 19 | 不适用 | 不适用 | 通过 | 0.228288 ms | 0.321312 ms | 1.4075× | 192 | valid，can gate |

两组使用相同 GPU7 UUID `GPU-52ca6155-d8bc-a416-847a-6f0c71db1552`；开发工具调用和独立评测均通过资源身份与进程集合检查。独立评测期间 GPU 无其他 compute-app 进程，前后快照均为 97,353 MiB 空闲、0% 利用率。直接比较为 `A latency / C latency = 0.228288 / 0.227744 = 1.0024×`，C 快约 0.24%，没有达到 1.05× 的 A/C 目标。这里相对 PyTorch 的 1.4113× 和 1.4075× 是各自终稿的基准成绩，不等于 A/C 加速比。

C 虽真实使用 3 次 `operator_bench` 并由 `operator_verify` 完成最终 L2，但它选择的终稿 SHA256 `df189e4c0ab4bef63000147d5ad2f5844b805ea1797c099df21dd2edce69e0fd` 与共同起始 seed 相同。A 的终稿 SHA256 为 `ba407ea0c1ee96ff9c80aa6a69715a655c4e36c8b80f4a6f2f8ef025364a339b`。因此本组证明了插件工具实际接入且有一次有效配对读数，没有显示出有实际意义或可重复的性能优势。

修正配对共预留 39 次模型调用；连同前述失败路由尝试，共 57 次 follow-up 预留。provider usage 仍全部缺失，实际账单金额未知。

截至本报告，首轮 4 格加两次 follow-up 尝试共预留 138 次模型调用；所有相关 provider usage 均缺失，不据此估算账单金额。

## 结论边界

首轮证明了 LayerNorm C 路径能在真实 DSH 会话中提交候选并获得 GPU `operator_verify` 反馈；GPU7 定向配对进一步确认 `operator_bench` 与 `operator_verify` 都可在真实会话中工作。但这两轮都没有证明插件整体提升算子性能：首轮 LayerNorm 差异约 1.34%，定向配对差异约 0.24%，均低于 5% A/C 目标；定向配对 C 最终仍选择 seed。Matmul C 在首轮未成功使用插件验证且性能低于 PyTorch 基线，A 格因预算没有终稿。

此前预检中的 LayerNorm `1.4179×/1.4173×` 和 Matmul `1.0362×/1.0363×` 是共同起点的真实 GPU 量尺读数，没有模型调用，不是 A/C 优化结果。它们只说明本轮 seed 通过了 L2 与测量资格门。

## 证据

- LayerNorm A 的最终 L2：[记录](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v1-execution/runs/layernorm-16384x6144-repeat-1/repeat-1/A/supervised-verification/003-final/record.json)；独立 L3：[结果](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v2-gpu2/final-evaluations/recovered-original-A/result.json)。
- LayerNorm C 的模型格：[记录](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v2-gpu2/runs/layernorm-16384x6144-repeat-1/repeat-1/C/record.json)；MCP/GPU 反馈：[relay 快照](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v2-gpu2/runs/layernorm-16384x6144-repeat-1/repeat-1/C/mcp-relay-snapshot.json)；独立 L3：[结果](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v2-gpu2/final-evaluations/plugin-C/result.json)。
- Matmul C 的模型格：[记录](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v2-gpu7/runs/matmul-4096-repeat-1/repeat-1/C/record.json)；拒绝的插件调用与资源观测：[relay 快照](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v2-gpu7/runs/matmul-4096-repeat-1/repeat-1/C/mcp-relay-snapshot.json)；独立 L3：[结果](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v2-gpu7/final-results/03-matmul-4096-repeat-1-C.json)。
- Matmul A 预算耗尽与停止窗口：[记录](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v2-gpu7/records/04-matmul-4096-repeat-1-A.json)、[窗口汇总](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v2-gpu7/window-result-03-04.json)。
- 修复后的零模型 readiness：[A/C DSH 探针](../../output/plugin_completion/20261030-s5-kernel-performance-v1/readiness-repaired/summary.json)、[性能 MCP 探针](../../output/plugin_completion/20261030-s5-kernel-performance-v1/readiness-repaired/perf-mcp/result.json)。
- GPU7 定向配对的计划与窗口：[计划](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v4-layernorm-followup-gpu7/plan.json)、[窗口结果](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v4-layernorm-followup-gpu7/window-result-01-02.json)；C 的插件调用与资源记录：[relay 快照](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v4-layernorm-followup-gpu7/runs/layernorm-16384x6144-repeat-1/repeat-1/C/mcp-relay-snapshot.json)；A/C 独立 L3：[C 结果](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v4-layernorm-followup-gpu7/final-evaluations/01-layernorm-16384x6144-repeat-1-C/result.json)、[A 结果](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v4-layernorm-followup-gpu7/final-evaluations/02-layernorm-16384x6144-repeat-1-A/result.json)。
- 首次错误路由尝试及停止记录：[C 格记录](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v3-layernorm-followup-gpu7/records/01-layernorm-16384x6144-repeat-1-C.json)、[relay 快照](../../output/plugin_completion/20261030-s5-kernel-performance-v1/trial-v3-layernorm-followup-gpu7/runs/layernorm-16384x6144-repeat-1/repeat-1/C/mcp-relay-snapshot.json)。
