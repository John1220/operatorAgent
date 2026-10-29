# S5 最小纵向试验结果（2026-10-28）

状态：**本机离线纵向链路跑通；不是正式 S5，也不是 Linux / 真实模型 / GPU 验收。**

## 范围和链路

一个固定的 `layernorm-tail` 任务，通过本机已安装的 macOS DSH Desktop `0.2.1-rc.2` headless CLI 发起一个 agent 会话；Node `v24.14.1`，临时 HOME、DSH_HOME 和工作目录，模型端点是只监听 `127.0.0.1` 的假 HTTP 服务。关闭试验配置中的 retry、标题模型、telemetry、web 和执行工具；这只是配置约束，**不是操作系统隔离**。假服务返回预先审核的固定 Python 源码，不提供模型生成质量证据。

假网关经同一 SQLite 账本限制最多一次请求和 1024 输出 token 预留，实际记下一次预留、一次已完成的假响应；实际 provider usage 为 unknown。DSH 事件中的合成零值不当作 provider 用量。监督器仅在 final 文本逐字节等于可信 fixture 时才交给 mock 验证；其他文本为 unknown，不执行。验证预算一次，使用同一 `layernorm-tail` 任务的主输入和两个 holdout，逐输入执行现有完整 mock L2 套件。

## 结果与记录

- 第二次运行的纵向结果 `passed`：一个假请求、一次 mock 验证；主输入、holdout 317 和 holdout 619 各自通过，三组输入下各有 12 条 mock suite case；监督器报告墙钟 3.221 秒。没有真实模型调用、GPU 执行、性能测量或费用。
- 候选源码 SHA256 `f5c78090d9c648e82019e652487d03d3b837294bc196679e17da901f04c677f1`；草案任务清单哈希 `8663285abafdd8fbbb3cfd9d0c6affa726c97cc0e1d27ab11555825eb0e12d26`；契约 `da7f3620b37d8c4936d4655482073ecaf7261beda913b28a7ec4bcf1dcd6678c`；suite `cf2c9a8d96f4edeec475697c2548a963594a2fb0737e1f6c8ffaf618ecfeb24c`。
- [第二次原始材料](../../output/plugin_completion/20261028-s5-vertical-v2/) 约 356 KiB，包含 `result.json`、fake 预算账本和快照、headless 事件、固定候选和 mock 验证明细；目录被 Git 忽略，需单独备份。`result.json` SHA256 `e4ee9b786d9ddeafd9e00187aac5b115b35c496fbeee8dbc7c93148edd2c6ec9`。
- [第一次材料](../../output/plugin_completion/20261028-s5-vertical-v1/) 保留为 unknown：headless 和假模型已经结束，但监督器把 `status.phase=turn_end` 错读为事件类型，未进入 mock 验证；修正后在新目录执行第二次，没有覆盖首次记录。
- 当前脚本哈希：`s5_vertical_trial.py` 为 `3fbc0b06ee8ababd76443e1a6296fc7d9eb2c7681ad92453cba6fcf975a3e53c`；`s5_vertical_headless.py` 为 `e526605423ef9256b05731fb4d331e60ebfbcdc58f4753ef72a296cac7b27e2c`；`s5_vertical_validation.py` 为 `39ea75063a4b23e8b7da9ab7977a05c382b5555731f6d12c6ed0c370b0f2326c`。这些是试验后的本地身份记录，非完整依赖或环境封版。
- 覆盖 headless、假 HTTP、验证器、未知候选拒绝和整体链路的专项测试 **43 项通过**；未重新执行整个项目测试集。

## 判断

这个小试验说明：可以复用现有 DSH 会话、假网关和 mock 验证，以很小的新增代码及本机现有依赖打通一条可留痕的链路。它**不证明** Linux headless 可运行、真实模型请求无法绕过网关、候选代码被隔离、真实 GPU 输入保真或远端清理，也不代表 A/B/C 三组实验结果。接真实服务之前必须分别解决这些问题；是否投入完整实验基础设施仍须按成本和目标判断。
