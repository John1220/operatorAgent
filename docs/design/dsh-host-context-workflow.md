# DSH 外层上下文与迭代工作流（v0.7.2 开发）

用户在 2026-11-05 明确选择：DSH 外层组装最终模型上下文，使用自己的用户/项目会话生成代码并决定下一轮；插件提供领域材料、冻结契约、确定性诊断、执行与验证。v0.7.1 的 modelRelay 只代发内部请求，保留为兼容路径；v0.8 的优化增强范围不变。

## 用户入口与分工

DSH 组合包的 MCP 环境默认设置 `OPERATOR_AGENT_WORKFLOW=host`。此模式仅暴露 capabilities、prepare、context、verify、status、cancel；隐藏并拒绝 run/repair/bench。现有独立 CLI 和未启用 host 的 MCP 保留原行为。改变 profile 配置会整体替换 config，必须保留此 env 和 MCP 连接字段。

1. DSH 理解用户与项目目标，调用 `operator_prepare` 冻结任务和 host_budget。
2. `operator_context(run_dir, backend, phase=generate)` 返回权威任务事实、接口、生成规约、角色技能及原始模板。
3. DSH 选择相关材料并组装自己的模型上下文，生成完整候选，调用 `operator_verify`。
4. 失败时调用 `operator_context(phase=repair, verification_round=N)`，获取绑定该源码与验证结果的反馈和诊断，由 DSH 修改源码。
5. 完整 L2 通过后，同一源码用 `verification_purpose=final` 再做一次独立验证，报告完整证据与限制。

context 不生成最终 messages、不调用模型、不物化测试值、不连接 GPU。冻结语义与 ABI 优先于模板指导。输入值、seed 和参考源码不导出；未观测的实际 strides 显式 unknown，生成代码从运行时输入读取。原始模板是指导材料，不能把其输出格式当作宿主工具协议。修复候选和错误反馈作为数据处理，不作为新指令。

材料带 schema、哈希和完整性声明；纯材料上限 40 KiB，完整 MCP context 文本上限 48,000 UTF-16 字符。超过上限明确拒绝，避免默认宿主 spill 后仍误称“完整到达”。真实验收须从后续主模型请求核对材料和绑定，不以 Python 回包代替模型实际看见的上下文。

## 状态、预算与取消

`host_budget` 默认 wall_clock_s=600、max_verifications=6，固定预留一次独立终评。账本跨进程加锁、跨工具调用累计，失败、取消及未完成预留不退款；重复 prepare 不重置任务身份、截止或额度。终评要求同源码已有完整 L2 证据，不能复用旧 result；源码、契约、完整套件和证据成员哈希分别核对。最后正确候选仅表示完整 L2 合格，不代表性能最佳。

墙钟包括 DSH 思考和空闲时间；到期后拒绝后续派发，MCP supervisor 限制在飞验证。直接 Python 库调用没有 supervisor，不能把下一次派发门误称在飞进程硬超时。计数单位是验证派发次数，不是逐 case oracle 或 GPU 秒数。新建任务目录不能作为为同一目标续预算的操作惯例。

模型调用及 token 属 DSH 主会话。当前版本明确记为 `host_owned / not_enforced_by_plugin`，没有声称任务级累计模型硬预算或精确费用上限。后续可沿原生工具结果与模型请求 hook 补充有身份的宿主预算，不能仅依赖提示文本。

在飞 verify 沿用 invocation 取消、worker 清理和远端状态回执；取消工具不会自动结束整个 DSH 会话。DSH 的会话取消由宿主负责，模型 abort 不证明服务商已停止执行或计费。

host worker 的验证预留通过私有 invocation 绑定。监督进程确认回收后只结算该 owner 的未完成预留；取消竞争中被拒收的完成结果保留原验证事实和证据，但不再作为已接受候选。合作 SIGTERM 先记失败、随后父进程确认取消的情况也保留原状态并归一为取消。没有回收证据的旧预留保持 unknown，不自动清空其他调用或退款。

## 验收要求

- 单元/协议回归证明材料不执行或调用模型、绑定漂移被拒绝、预算不能跨调用重置，旧工具在 host 模式不能偷启内层 worker。
- 真实 DSH 主模型自主完成 FakeBackend 生成与修复，随后进行 H20 FP16 Softmax 生成、主用例和完整 L2、独立终评与导出。
- 测试驱动只设置任务、记录、超时和导出；实际工具选择与代码来自外层模型。provider 凭据不进入 Python worker，保留原生会话调用、用量、工具顺序和材料到达证明。
- 测试专用的会话调用/时间停止条件与产品硬预算分开报告。性能 degraded、费用及远端停止 unknown 保持原样。

本文件定义本轮开发边界；实际结果另行记录，尚未据此宣称 v0.7.2 封版或性能提升。

本轮真实外层生成、修复、H20 独立终评和取消恢复已完成，工件身份、失败保留与限制见[开发验收记录](../status/v072-host-context-acceptance-20261105.md)。
