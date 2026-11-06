# v0.7.2 模型预算与恢复候选验收

日期：2026-11-06。当前候选为 Python `0.7.2rc1` / DSH 插件 `0.2.2-rc.4` / 工具契约 `12`，分支 `codex/v0.7.2`，尚未封版。开发基线 `1aa2539` 之后的工作区通过显式源码快照构建；本记录不改写 [dev2 的工件身份与验收](v072-host-context-acceptance-20261105.md)。

## 工件身份

| 工件 | SHA256 |
|---|---|
| `operator_agent-0.7.2rc1-py3-none-any.whl` | `51dfd0fd0e89667c7ff24b70cf5a84270885e038870145a97ec8a42d2ad82cf0` |
| `operator_agent-0.7.2rc1.tar.gz` | `25d73153609e3ed7499501bcc28f2611ef3db48bb55f0c09046aa689f9329ccb` |
| `dsh-operator-agent-0.2.2-rc.4.tgz` | `d3c506a96a3e0e22e55012fbc93538e894b5d2479b9b48cd12117288817792ed` |

源码清单 SHA256 为 `fe3ad4ddde99e597c8b738ab141f6fdcb6f28b5c5a1d72468acfde0ba56cd3ad`。工作目录为 `/private/tmp/operator-v072-closeout.rVQIRa`；工件、原始回执和日志保留在该目录，最终归档与源码提交待正式封版步骤完成。

## 已完成验收

| 项目 | 结果与回执 |
|---|---|
| 原生模型硬门 | 插件完整测试 **116 passed，0 failed，0 skipped**；显式启用 host budget 和 modelRelay 的真实 DSH 本地 provider 用例。调用耗尽、输出预留耗尽均在 provider 前拒绝；同 session 重启不补额度；在飞主模型取消后预留不退且无额外重试。日志 `/private/tmp/operator-v072-plugin-clock-final.log` |
| 截止与恢复边界 | JS 账本 26 项、hook 19 项通过；坏成功回包重启仍封锁；墙钟回拨、停机恢复和前跳耗尽后回正不增加时间。时钟使用现场 Python 剩余量和同 boot 的持久 Node 单调锚点，不比较 Python/Node 的单调 epoch |
| Python 接线 | 相关工具/上下文/预算/恢复 145 项通过；后续新增时钟的跨语言预算/恢复组合 59 项通过。批次有重叠，不相加。MCP host 七工具包含 `operator_resume`，legacy 工具面保留 |
| 干净安装 | 新 CPython 3.12.16 venv 从本地 wheelhouse 安装候选，`pip check` 通过；`profile-rc1/manifest.json` 证明隔离 DSH 安装、完整配置合成和卸载通过；`smoke-rc1/smoke-result.json` 证明 CLI/MCP legacy 与真实 DSH 离线冒烟通过，均为 `test_only` |
| 耗尽后的无模型出口 | `no-model-exit/run-rc1/result.json` 通过。手工 fixture 账本耗尽 1 次/128 token 后，实际安装 `/op status` 取得原失败源码/报告和有效剩余额度；同任务真实在飞验证经 `/op cancel` 取消，确认 worker 已回收。原 task/deadline/policy/契约/失败证据不变，模型账本逐字不变，验证预留不退。此项实际模型/provider/SSH 派发全为 0，手工账本明确 `test_only`，不冒称 provider 请求 |
| 真实模型恢复与干预 | `native-resume-rc1/summary.json` 通过。外层取得材料后收到原生 `agent.steer` 用户暂停要求，正常退出；同 cwd/DSH_HOME/session 重启，新增用户注释进入 native options、实际 provider 请求及候选源码。原 task_id、契约、起点/deadline、policy 字节和历史调用记录不变，自主完成探索和同源码独立终评 |
| 真实模型修复 | `native-repair-rc1` 保留首次错误源码的 L2 失败、绑定源码的 repair context、后续异源码完整 L2 和同源码独立终评。原驱动因模型省略错误 seed 的末尾一个 LF 而判失败；原 `summary.json` 保持 false。独立 `repair-evidence-review.json` 核对唯一差异为末尾 LF、AST 相同、实际失败源码/证据哈希一致及整个修复闭环通过。没有删除失败、补调用或重置预算 |
| H20 新候选 | `native-gpu-rc1/summary.json` 通过。真实 DSH 外层自主生成 FP16 Softmax `4096×4096`，主用例全量比较 **16,777,216 元素**、最大误差 **1.91e-6**，完整 **12 个 L2 case** 及同源码独立终评通过。源码和报告已导出；`L3=false`、`validity=uninformative`，不证明性能提升 |

三项真实模型任务均使用 DSH 已配置的 `aliyun-native / deepseek-v4.1-flash-1017`，由外层模型选择工具并编写/修复源码，测试脚本只设置声明、限制、记录、用户干预和导出。未使用确定性外层工具选择 fixture。全部 5 次 context 交付均在后续真实主模型请求中逐字核对完整正文、无 spill；Python 内层模型派发为 **0**。

| 任务 | 全部观察到的主模型调用 | 成功绑定后的账本调用 | 输出 token 预留 |
|---|---:|---:|---:|
| 恢复与用户干预 | 8 | 7 | 57344 |
| 错误候选修复 | 7 | 6 | 49152 |
| H20 生成和独立终评 | 5 | 4 | 32768 |

每个任务冻结模型上限 `12 / 8192 / 98304`，验证派发上限 4（固定保留 1 次 final）、墙钟 600 s。每项的第一次准备请求发生于绑定前，明确不计入任务模型账本；辅助模型请求均为 0。有效 API usage 单列，费用与服务商实际停止仍为 unknown。

H20 源哈希为 `0698da97ab61741b7adc8a16035d6c5015ee58fcadb189f05c5aab40be41531a`，契约哈希为 `8fd7e3ed178667c3e6ade373b032c98edef82454de6393edcde72bdda6b8413c`。前后检查同一 H20 3 号 GPU、驱动、运行时和 15 个 runner 文件；不部署远端运行时。预检主机 load 为 6.19/192 核，性能资格仍按实际 `uninformative` 保留。

## 回归结果与尚未完成

全量回归已完成。Python 首轮为 **6874 passed / 10 failed / 2 skipped**，失败包括旧 backend/Fixer 测试替身未跟上现有接口、两处旧工具版本断言，以及时钟接线过程中跨语言测试尚未同步；均已定位并修复，首轮日志原样保留。第二轮完整 `tests/` 于 2026-11-06 10:17 结束：**6885 passed / 0 failed / 1 skipped**（666.28 s）。唯一 skip 为需真实 CUDA 设备的 `test_remote_input_fidelity::test_real_cuda_triton_runner_and_mock_path_read_back_actual_device_views`。证据为 `regression/python-full-final.log` 与 `python-full-final.xml`（failures=0 / errors=0）。插件完整测试 116 passed / 0 failed / 0 skipped（`/private/tmp/operator-v072-plugin-clock-final.log`）；本机复跑 112 passed / 0 failed / 4 skipped，原因是该机 DSH 安装缺少测试所需的 `node_modules/@deepseek-ai/cordis` 布局，不构成失败。

尚未完成：

- Python `0.7.2` / 插件 `0.2.2` 正式工件、最终安装/回退与源码复评、归档、源码提交和本地标签尚未完成。公开发布或远端推送未执行。

入口与支持边界见[封版待办](v072-todo.md)、[支持清单](../testing/v072-support.md)和[外层分工设计](../design/dsh-host-context-workflow.md)。
