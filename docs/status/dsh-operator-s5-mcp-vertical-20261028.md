# S5：真实 MCP 的最小纵向试验（2026-10-28）

状态：**本机 DSH → 本仓 operator MCP → mock L2 → DSH 反馈链路跑通；不是正式 S5 或真机验证。**

## 固定范围

沿用本机 macOS DSH Desktop `0.2.1-rc.2` 和 Node `v24.14.1`，使用全新临时 HOME/DSH_HOME 和工作区；headless overlay 同时插入仓库 `integrations/dsh-operator/index.mjs` 与实际 DSH MCP client，后者通过 stdio 启动本仓 `operator_agent.mcp_server`。只使用回环假模型与占位凭据；关闭 retry、标题模型、telemetry、web 和试验不需要的工具。配置限制不构成操作系统隔离，MCP 本身也未被证明强制只允许 mock 或一次验证。

监督器在 DSH 会话**开始前**调用真实 `hitl_tools.prepare`，创建 LayerNorm fp32 `[7,33]`、Python/mock_gpu 的任务与契约；这一操作不计为 DSH 工具调用。假模型第一轮固定发起真正的 `mcp__operator__operator_verify`，参数包含预先审核的固定 Python fixture、`backend=mock`、`stop_after=L2`。mock 执行不是敌意源码沙箱，本次不接受真实模型或不可信源码。第二次假模型请求必须携带同一个工具调用 ID 的实际 MCP 反馈，之后才返回固定结束文本。

## 结果

- 一次运行 `passed`，墙钟 2.247 秒；共享账本记录 **2 次假模型请求**，真实 MCP worker 完成 **1 次 mock 验证**。provider usage 为 unknown，不能把 DSH 合成的零值当作服务商计费。
- 真实后端解析为 `MockGPU`，未降级；L1、L2 通过。完整独立 result 包含 **1 组主输入和 12 条完整 correctness suite case**，包含逐 case 与重复运行的输入保真记录。这与上次独立 mock 评估器的主输入加两个 holdout 不是相同评测，不继承其 holdout 结论。
- DSH `tool_call`、`tool_result`、第二次请求的 feedback 与同一 `call_operator_verify_01` 绑定；核验器重新核对了模型账本顺序与请求哈希、实际 MCP 参数、worker `request.json`/终态、候选源码、冻结契约、完整 L2、轮次 `complete.json` 中所有文件哈希及 MCP 回包哈希。`candidate_origin` 为 `undeclared`，不声称来源证明。
- 候选 SHA256 `f5c78090d9c648e82019e652487d03d3b837294bc196679e17da901f04c677f1`；契约哈希 `96cdb38deafce9282485b5beaa555a738e573a1c0892292554b9d7f70685c7a7`；suite 哈希 `cf2c9a8d96f4edeec475697c2548a963594a2fb0737e1f6c8ffaf618ecfeb24c`。
- [原始记录](../../output/plugin_completion/20261028-s5-vertical-mcp-v1/) 约 328 KiB，含 headless 事件、预算、反馈观察、完整 MCP 任务产物及交叉核验；被 Git 忽略，需单独备份。`result.json` SHA256 `34d50de215197e8ed9c1892fb44b113f9421dae696231c71f28d3ab23728780b`。
- 实际 headless/MCP、结果重算、前一工具试验与负向情形的专项回归 **33 项通过**；未重新运行整个项目测试集。

## 限制

该试验证实 DSH 的真实 MCP client、仓库 MCP server 和 `operator_verify` mock 路径完成了一次工具反馈；虽然 overlay 插入了仓库 DSH 插件，**本次没有独立验证 `/op` 斜杠命令的注册或执行**。固定脚本只是约束了本次调用，不证明受试进程不能调用其他 MCP 工具或绕过额度。没有真实模型、GPU、Linux 或系统级隔离，不证明插件对真实开发质量的增益。
