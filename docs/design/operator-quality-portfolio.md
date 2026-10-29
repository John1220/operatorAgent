# 三候选完整 L2 对照

`benchmarks/operator_quality.py --portfolio` 冻结三个独立生成槽位：布局优先、数值稳定优先、分块计算。每槽最多一次生成、一次完整候选验证，不调用 Fixer，不自动重试。任务仍使用现有十任务集、三个独立主输入及完整正确性配置；参考实现和容差不变。

所有槽位均执行，即使第一槽已经通过。首槽是嵌套的单次生成基线，组合结果选择最早完整 L2 通过的源码。没有通过且存在未知结果时，任务保持未知；失败、超时、坏响应与预检拒绝均留在原分母中。报告分别记录失败基线被挽回、未知基线被解决，不把两者混为修复收益。

每槽独立模型账本、调用上限和输入物化；每任务共享通过只读预检的固定 GPU 后端。生成请求提供完整公开 shape、dtype、pattern、layout 域和冻结 dtype 语义，隐藏具体数组及验收种子。三个策略名称只是生成指引：源码哈希及不含位置信息的 AST 哈希衡量实际结构差异，不能证明语义多样性。预检查拒绝的已生成源码也保存并计费。

报告包含全部三槽和一次 readiness 成本。tokens 保留来源；缺失 API usage 保持未知，不归零。首槽基线费用已经包含在组合总费用内，不能再次相加。模型账单金额、GPU 时间上限和独占隔离没有保证。本流程只验证 L2，性能始终为 `not_measured`。

```sh
python benchmarks/operator_quality.py --portfolio --backend ssh --host HOST --output plan.json
python benchmarks/operator_quality.py --preflight plan.json --host HOST --output NEW_PREFLIGHT_DIR
python benchmarks/operator_quality.py --execute plan.json --host HOST --output NEW_RUN_DIR
```

计划默认只写声明，不触发模型或设备。模型与 endpoint 来自既有配置，凭据通过受保护文件提供，不写入计划。真实执行要求模型、端点、目标主机和源码身份匹配。回放使用 `--portfolio --replay replies.json`；文件为 `{case_id: [JSON响应1, JSON响应2, JSON响应3]}`，仅作 CPU 流程验证，不构成真实生成质量证据。

可在不同操作系统进程执行互不重叠的任务子计划；不在线程间共享输出捕获。`merge_portfolio_reports` 仅接受完整、无重叠、控制一致且通过验证的子报告，按主计划顺序汇总，不选择或重跑结果。并行进程可能共享 GPU，本流程不测性能，也不声称设备隔离。
