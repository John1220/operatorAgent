# S5 最小工具反馈纵向试验（2026-10-28）

状态：**本机离线工具调用与反馈链路跑通；并非正式 S5 对照。**

## 实验范围

一个 `layernorm-tail` 任务，使用本机 macOS 已安装的 DSH Desktop `0.2.1-rc.2` headless、临时 HOME/DSH_HOME 和工作目录。模型是固定脚本的回环假服务；工具是仅用于试验、由 DSH 注册的 `validate_kernel`，通过回环 HTTP 调用 mock 验证服务。插件加载配置关闭 retry、标题模型、telemetry、web 和其他执行工具；配置限制不构成操作系统隔离。

假模型的第一条流式响应请求 `validate_kernel`，参数为事先审核的固定 Python fixture。验证服务最多接受一次实际验证，并只允许可信 fixture 进入 mock 执行；未知源码不执行且消耗额度。完整 mock L2 覆盖主输入和两个 holdout。验证在单线程 worker 中运行，避免多线程父进程为每个 suite case 反复启动解释器；它不提供生成代码的安全沙箱。

假模型的第二次请求必须实际携带同一 `tool_call_id` 的工具反馈，且该反馈字节的 SHA256 必须等于验证服务回执；否则假模型拒绝继续。第一次和第二次请求分别经同一共享账本预留和完成，最多 2 次，每次输出预留最多 1024 tokens；provider usage 保持 unknown。最终文本只代表假模型脚本响应，不作为验证通过证据。

## 结果与证据

- 第三次运行 `passed`，墙钟 2.324 秒；2 次假请求、1 次 mock 验证。DSH 事件显示一次 `validate_kernel` 调用、一次同 ID 的工具完成、第二次 step 和最终结束；服务实际收到的源码请求、返回结果以及第二次模型请求的工具反馈经哈希交叉核对。
- tool call ID `call_s5_mock_01`；候选源码 SHA256 `f5c78090d9c648e82019e652487d03d3b837294bc196679e17da901f04c677f1`；工具反馈 SHA256 `0812da6e536f354ad199d0c0d75fe2e320534565a86de018f04b5a58daac4dab`。
- [第三次原始材料](../../output/plugin_completion/20261028-s5-vertical-tool-v3/) 包含 headless 事件、两次预算记录、验证请求与完整结果、假服务观察和交叉核验；约 376 KiB，目录被 Git 忽略，需单独备份。`result.json` SHA256 `a1d38dfb18b78b62a1cc5b11774b3593d4bbb40b7a1f0d6475018652e58ad2e1`。最终版核验器已从这些材料重新核对并得到 `complete`；这些是同机监督器的记录，不是独立隔离或来源证明。
- [第一次材料](../../output/plugin_completion/20261028-s5-vertical-tool-v1/) 中工具链路实际发生，但监督器漏传验证请求数，交叉核验保守记为 unknown；[第二次材料](../../output/plugin_completion/20261028-s5-vertical-tool-v2/) 通过了当时的核验；第三次增加服务端源码请求哈希绑定，使用新目录，没有覆盖前两次。
- 最终专项回归 **49 项通过**，包括真实本机 headless 联测、预算、未知源码不执行、工具回执与证据缺失/不匹配；未重新执行整个项目测试集。此前 mock 验证在线程中为每个 case 启动独立解释器而超时的尝试不是通过结果；已改为单线程 worker 并复测。

## 边界

这次验证的是**DSH 实验工具**与假模型的工具反馈流程，尚未接入仓库现有 `dsh-operator` 插件或 MCP，不能宣称插件接线通过。没有调用真实模型、GPU，也没有验证 Linux 宿主、网络/文件隔离、实际 provider 费用或三组公平性。下一步若继续，应单独决定是否用现有插件/MCP 和 mock 后端替换实验工具，再考虑 Linux 和真实资源。
