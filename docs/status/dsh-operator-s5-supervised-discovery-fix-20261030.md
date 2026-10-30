# S5 监督模式插件发现说明修正（2026-10-30）

状态：**接口修正与验收完成**。相关回归 148 项通过；实际 DSH 的无模型 A/C readiness 通过；完整回归 **6113 项通过、1 项跳过**，Golden CPU/mock **4/4 PASS**。没有请求真实模型/GPU，也没有修改已停批的 24 格计划、提示、候选或结果。

## 问题与修正

[上一批观察](dsh-operator-s5-cooperative-study-20261029.md)中，六个 C case 均发现并调用了 capabilities，但七次开发验证全部选择公共入口。监督任务要求固定后端、禁止 host，工具 schema 和 capabilities 却沿用普通 SSH 必须 host 的说明，展示内容与实际转发规则不一致。

修正位于 `operator_agent/mcp_server.py`，仅在明确启用 `OPERATOR_AGENT_S5_DEV_*` 的监督模式生效：

- `tools/list` 的 verify 参数只保留 run_dir、code/code_file、candidate_origin、固定 backend、L2 和有界 timeout；不再展示 host、port、user、key_path、persistent、target_gflops、local、L1/L3 等本模式不接收的选项。
- schema 默认 backend 来自当前配置；实际 verify 调用也应用相同默认值，省略参数时不会误落到 mock。未准备好配置时拒绝发现，不回退展示普通直连能力。
- initialize instructions 和工具描述明确任务已准备，使用任务提供的 run_dir，无须重复 prepare；连接由监督器持有，不提供 host/凭据，公共和 MCP 验证共用额度，独立终评不能当开发工具调用。
- capabilities 只返回当前配置的委托后端，`configured=true`、`delegated=true`、`requires=[]`，但 `connectivity_verified=false`；本机 Python/Torch/GPU 信息单列 `local_runtime`，不再用本机 SSH 环境缺失表达委托服务不可用。
- 普通未启用监督模式的 MCP 接口不变：SSH 仍有 host 要求，默认 backend 仍为 mock，原有工具面及 L1/L2/L3 保留。

这是监督模式接口的契约修正，没有给 C 单独增加实验任务提示、强制使用计数或额外额度。不能据此声称真实模型会自然选择插件，或插件已经改善完成率。

## 验收与中途发现

相关回归覆盖普通/监督两种 schema、实际 initialize、capabilities 不泄露端点/令牌、不把配置等价于连通性、默认参数实际派发、无效配置拒绝，以及真实 MCP relay 的失败后通过反馈与共同额度。

初次实际 DSH 探针中，C 的 schema 已正常发现，但消息原文匹配失败：探针先 `json.dumps` 整个消息，再查找未转义的说明原文；新说明包含双引号，JSON 转义引发假阴性。已把 `benchmarks/s5_plugin_probe.py` 改为递归检查实际消息文本，增加字符串/内容块两种正例与缺失内容负例，没有取消说明必须进入 DSH 的检查。

修正后独立 readiness 中 A/C 各完成 10 个脚本化假模型请求；两组都用完两次开发验证，额外公开提交被拒；C 发现实际 schema 和新 instructions，经真实 MCP 取得先失败、后通过反馈，最终额度尚未使用。它证明说明传递、参数和调用链一致，不是自然模型选择或真机正确性的新成绩。

证据：`output/plugin_completion/20261030-s5-supervised-discovery-fix-v1/`。初次失败探针保存在 `readiness/`，通过版本为 `readiness-v2/summary.json`，哈希 `c42cdcaf3be458b063cc8906a56cbf527f37d47ae4b5df4b6cfb3a9729e4e688`。本机监听和 MCP 使用的是假服务、mock 后端；真实模型调用 0、GPU 执行 0。

完整回归命令 `python3.9 -m pytest tests/ -q`，耗时 504.75 秒，日志 `full-regression.log`；两条 warning 来自既有非有限参考值负例的 NumPy subtract。Golden 命令 `python3.9 tests/golden_regression.py compare`，四项 CPU/mock 全通过，日志 `golden.log`，不是 GPU 性能成绩。五个变更源码/测试及 SHA256 清单位于 `frozen-source/` 和 `source-manifest.json`。

## 后续

旧批 13/24 停止及其结果继续保持。若检验自然使用率，应以当前源码身份和新的 readiness 重新冻结版本，保留相同共同任务提示和公开验证机会；不能在旧版本格上补跑后覆盖，也不能把脚本化 MCP 调用当成模型主动选择。
