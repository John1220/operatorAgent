# S5 算子性能对比：预检与首轮状态（2026-10-30）

本记录对应[性能优化 A/C 对照计划](../design/s5-kernel-performance-comparison-20261030.md)。首次快照只读取 H20 主机设备和进程状态，没有调用模型或启动候选。用户先确认 GPU2 可用于实验，随后确认 GPU7 也可用；起点预检与 readiness 均完成，首轮模型 pilot 已执行。详细结果见[首轮实测记录](dsh-operator-s5-kernel-performance-results-20261030.md)。

## 资源门

首次只读快照中，远端 `algorithm-h20` 八张卡均有 compute-app 进程，因此没有卡满足独占条件。用户确认 GPU2 当前没有进程在使用，可以用于本实验；随后复查仍看到驻留进程，但 GPU2 利用率为 0%。本实验据此采用共享卡测量策略，不宣称独占。

| GPU | UUID | 型号 | 空闲显存 MiB | 利用率 | 计算进程 | 可派发 |
|---:|---|---|---:|---:|---|---|
| 0 | `GPU-e4f8f9cd-7125-adf0-dd9d-5b9c03a00415` | H20 | 4735 | 0% | 有 | 否 |
| 1 | `GPU-74130a78-6b1f-3d44-077f-0d7e2fecba69` | H20 | 2792 | 100% | 有 | 否 |
| 2 | `GPU-13c6c4b4-45ce-8963-1122-df2b3d72fb76` | H20 | 22246 | 0% | 7 个已知驻留进程，用户确认暂不使用 | 共享模式可派发 |
| 3 | `GPU-634bbc76-9ab6-220a-48a8-27247d7acbe2` | H20 | 14756 | 0% | 有 | 否 |
| 4 | `GPU-82f3be28-9c30-8d69-47a5-9cafb70131c1` | H20 | 1390 | 0% | 有 | 否 |
| 5 | `GPU-0c16a8be-9d4a-61fc-e703-0c659f2544d4` | H20 | 3906 | 0% | 有 | 否 |
| 6 | `GPU-0f61d0d6-767e-44a6-0cd0-a43892bf17e2` | H20 | 354 | 0% | 有 | 否 |
| 7 | `GPU-52ca6155-d8bc-a416-847a-6f0c71db1552` | H20 | 97353 | 0% | 有 | 否 |

用户确认 GPU7 可用于本实验后，派发前复查到 GPU7 空闲 97353 MiB、利用率 0%、未列出 compute-app；按无驻留进程的共享策略使用，仍不声称持有独占租约。GPU2 保持原有 7 个已知驻留进程和共享策略。

GPU2 复查值仍为 22246 MiB 空闲、0% 利用率，并列出以下 7 个驻留 PID：`162589, 822889, 1641644, 1677855, 3885613, 3890552, 3916094`。按用户确认，将这些 PID 纳入共享卡 allowlist；它们仍作为驻留进程留档。每次验证前后要求 GPU2 UUID 不变、空闲显存至少 8192 MiB、利用率不高于 5%，且 PID 只能来自此 allowlist；前后 PID 集合变化则该格记 unknown。正式 L3 仍必须由生产量尺判为 `valid`、`measurement_can_gate=true` 且 `benchmark_contaminated=false`。进程驻留本身不会否决，但也不会把共享卡描述为独占。

## 当前进度

- [x] 将 A/C 性能对比计划写入设计文档；LayerNorm 和 Matmul 起点文件 SHA256 与计划记录一致。
- [x] 首次只读 GPU 资源快照；用户随后确认 GPU2 可供本实验使用。
- [x] 将 GPU2 冻结为共享卡目标，并记录用户确认的 7 个允许驻留 PID 和逐次资源检查规则。
- [x] 核实两个 seed 的原任务尺寸为 LayerNorm `7×33`、Matmul `64³`；本实验会按目标大形状重新构建契约。
- [x] 两个共同起点各两次完整 L2→L3 预检；未调用模型。
- [x] A/C 无模型入口探针：普通 DSH 公共验证路径、真实 MCP `operator_verify` 和性能 MCP `operator_bench` 工具发现均通过；没有提交候选。
- [x] 修复性能 C 启动和 runner 收尾问题后，重新冻结 GPU2/GPU7 计划并重跑零模型 readiness。
- [x] GPU2 LayerNorm、GPU7 Matmul 首轮并行执行；其中 Matmul A 达到调用预算后 unknown，按计划停止后续 repeat-2。
- [x] 追加 GPU7 LayerNorm C→A 定向配对；C 的真实 `operator_bench` / `operator_verify` 调用、两组完整 L2 与独立 L3 均有记录；A/C 差异约 0.24%，未达到 5% 目标。
- [ ] 完成预注册的 8 格和每个终稿的两次独立确认；本轮未完成，不能据此作整体性能优势结论。

## 起点验证

四次验证使用重新构建的大形状契约、相同冻结 seed 源码及生产完整 L2/L3。LayerNorm 完整套件 12/12、Matmul 15/15 均通过；L3 `measurement_validity=valid`、`measurement_can_gate=true`、`benchmark_contaminated=false`。GPU2 验证前后利用率均为 0%，7 个驻留 PID 集合完全一致。读数来自用户确认暂不使用 GPU 的共享卡，不属于独占卡读数。

| workload / 次数 | candidate median ms | 基线 median ms | 加速比 | 配对样本 | L2 / L3 | 墙钟秒 | 证据 |
|---|---:|---:|---:|---:|---|---:|---|
| LayerNorm 16384×6144 / 1 | 0.226176 | 0.320704 | 1.4179× | 192 | 12/12，通过 / valid | 28.634 | [结果](../../output/plugin_completion/20261030-s5-kernel-performance-v1/preflight/layernorm-seed-1-escalated/result.json) |
| LayerNorm 16384×6144 / 2 | 0.226240 | 0.320640 | 1.4173× | 191 | 12/12，通过 / valid | 28.545 | [结果](../../output/plugin_completion/20261030-s5-kernel-performance-v1/preflight/layernorm-seed-2/result.json) |
| Matmul 4096³ fp16 / 1 | 1.000864 | 1.037136 | 1.0362× | 48 | 15/15，通过 / valid | 23.987 | [结果](../../output/plugin_completion/20261030-s5-kernel-performance-v1/preflight/matmul-seed-1/result.json) |
| Matmul 4096³ fp16 / 2 | 0.998352 | 1.034560 | 1.0363× | 48 | 15/15，通过 / valid | 20.627 | [结果](../../output/plugin_completion/20261030-s5-kernel-performance-v1/preflight/matmul-seed-2/result.json) |

这些是本轮共同起点的资格读数，不是 A/C 组结果，也不证明 DSH 或插件生成/优化了候选。Matmul 相对 PyTorch 的读数约 1.036×，只用于本轮起点比较；不与旧 GPU7 结果合并。

第一次 LayerNorm 执行在本机沙箱拒绝 SSH 时结束，结果保留为 unknown，用时 1.166 秒；没有 `resources-before.json` 或派发记录，验证 CLI 未启动。随后在 SSH 执行权限获批后，以新的一次性目录完成预检；首次 unknown 不计作 GPU 验证。

## A/C 无模型入口

- [修复后的普通 DSH / MCP readiness](../../output/plugin_completion/20261030-s5-kernel-performance-v1/readiness-repaired/summary.json)：A/C 均通过；两臂共 20 次 fake-provider 请求，只用于 UI/工具链路探针；真实模型调用 0、GPU 执行 0。
- [修复后的性能 MCP 工具发现](../../output/plugin_completion/20261030-s5-kernel-performance-v1/readiness-repaired/perf-mcp/result.json)：真实 core MCP server 初始化及工具发现通过，暴露 `operator_bench`、`operator_capabilities`、`operator_prepare`、`operator_verify`；候选提交、验证、基准运行均为 0，真实模型调用 0、GPU 执行 0。

当前 A 探针未提供 SSH 凭据，证明的是原生 DSH 普通工具和公共 L2 入口可用，不证明 A 能从该隔离环境自行访问远端 H20。A/C 的不同 GPU 反馈能力属于“原生 DSH 环境 vs 安装插件”这套产品对照的一部分，结论必须限定该处理条件，按臂报告实际 GPU 调用成本。首轮真实模型结果和插件工具拒绝原因见[实测记录](dsh-operator-s5-kernel-performance-results-20261030.md)。

普通 readiness 探针首次也曾在沙箱内因 loopback bind 被拒绝，未启动 DSH/GPU/真实 provider；正式 no-model readiness 在扩展权限通道重跑通过。该探针不证明 A/C 正式实验的真实模型反馈质量。

首轮每格使用 DeepSeek V4.1 Flash 1017、Responses、medium reasoning、每请求最多 32K output token、最多 24 次调用及 2400 秒墙钟上限。所有 81 次模型调用预留的 provider usage 均缺失，账单金额未知。共享卡快照不构成 GPU 租约。
