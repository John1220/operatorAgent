# S5 受控合作环境 A/C 对照 v2（2026-10-30）

状态：**已因验证实现运行中变化而停批，未完成对照**。汇总于北京时间 2026-10-30 09:06:45 落盘：只派发 Matmul fp16 aligned / repeat-1 / A，后续 23 格未执行。本批使用已修正的监督模式工具说明，以及适配器 idle 930 秒/网关硬截止 900 秒；不覆盖旧批。

## 实际结果（2026-10-30 复核）

- 固定分母 24 格：0 passed、0 确定失败、24 unknown，其中 1 格设施错误、23 格未执行。C 尚未运行，不能比较 A/C，也不能判断修正后的插件自然使用情况。
- 已派发格模型预留/完成均为 9 次，DSH 正常退出（returncode 0、stderr 为空），网关 failures 为空；记录墙钟 412.397 秒。本格没有重现此前 300 秒 idle timeout，但单格不证明该类问题全部消除。provider 账单金额未知。
- 第一次开发验证通过真实 H20 完整 L2（15 case），候选 SHA 为 `e21fbf49…396a9f`。第二次开发验证底层 L1/L2 也报告通过，但证据中的 `implementation_stable=false`，监督结论因此为 unknown，理由 `implementation not stable`。
- 第二次开发验证的前后实现哈希明确显示 `operator_agent/remote_executor.py` 从 `6bd2cf999e73931419fff446dad23afc1ebac6d2873d9148e19528db3fde6eea` 变为 `081f2618de8b1d935862872011509ec328c1db7742f1a21019179dabeb0bdd33`。当前文件又与冻结哈希一致；记录能证明运行期间变动，不能确定修改来源或原因。
- 最终候选已取得并保存（SHA `ce677b75…a714b`），独立终评返回 `verification unavailable: ValueError`，没有取得最终完整 L2 结论。服务加载计划会核对实现身份，前述变化与身份拒绝相符；现有记录仅保留异常类型，不能据此断言该 ValueError 的唯一根因。
- 开发候选通过不替代最终候选终评，不改记模型数值失败，不自动补跑或与旧批拼分母。S5 仍未完成，S6 未开始。

下一步应把测试使用的完整验证实现（包括 `operator_agent/`、`remote_scripts/` 和监督驱动）固定到独立只读快照，避免工作区并行编辑影响运行；先做无模型身份稳定检查，再安排真实测试。本次复核只读取证据并更新报告，没有启动新模型/GPU 请求。

## 固定条件

- 计划：`6012c4a43ed7ac3a5260c3c09c3292a98f9d0af452a036d30338341b74903db7`。
- 六项前瞻任务、A/C 各两次重复，共 24 格；清单哈希 `79fb8b03ee0514d67a354e24b78622fb668a7d27e3592ad3d1144ee2dc74bbd2`。
- `deepseek-v4.1-flash-1017`，Responses，medium，单次最多 32768 输出，每格 24 次调用/786432 输出预留/2400 秒开发会话；每格 2 次开发 L2 + 1 次独立最终 L2，每次验证最多 600 秒。
- 单请求监督硬截止仍为 900 秒，DSH provider `streamIdleTimeoutMs=930000`。A/C 配置相同，停止规则和额度不增加。
- 全批上限 576 次模型请求预留、18874368 输出预留、72 次验证；这些不是预计使用量或账单金额，真实费用未知。
- H20 GPU2，UUID `GPU-13c6c4b4-45ce-8963-1122-df2b3d72fb76`。启动前重新观察：22246 MiB 空闲、利用率 0%、七个原有已知进程。逐格按固定身份、至少 8192 MiB 空闲、利用率不高于 5% 和冻结 PID 名单检查；共享 GPU 不用于内核性能增益结论。
- 策略 `cooperative-exploratory-v1`，保留文件/工具边界、统一验证标准、共同预算和独立终评。完整网络唯一出口、恶意候选沙箱和预算绝对不可绕过不作已建立声明，`formal_s5_credit=false`。
- 普通编译/数值失败及调用耗尽保留结果，继续下一格；设施/身份/资源 unknown 停止后续派发，剩余格保留未执行，不自动重试。

## 本批与历史区别

已修正[监督接口说明冲突](dsh-operator-s5-supervised-discovery-fix-20261030.md)和[宿主提前 idle timeout](dsh-operator-s5-stream-timeout-fix-20261030.md)。共用任务提示保持正常插件可选使用，不向 C 临时添加强制调用要求；能力仍明确“已配置不等于已探测连通”。

新 readiness 的 A/C 各 10 次脚本化请求通过，真实模型/GPU 0，哈希 `ccc03dacc5488a41e0fce12730e69c868d75fce402b4caf0e807e94a947f37ca`。超时相关回归 183 项通过后，才冻结并启动真实批次。新旧版本结果分别报告，旧批第一轮 A/C 各 6/6 和第二轮停批不作为本批补格。

## 证据与收尾

证据目录：`output/plugin_completion/20261030-s5-cooperative-study-v2/`，包含 `readiness/`、`frozen-plan/`、`frozen-source/` 和 `execution/`。本次格记录位于 `execution/runs/study-matmul-fp16-aligned/repeat-1/A/record.json`，三次验证回执位于其 `supervised-verification/`；已生成 `execution/aggregate.json`，哈希 `9508441165a2168dd4b7fc85746ed161760415736fa12535b99b7272f558dac8`。

最终按全部分母报告正确率、调用/耗时、失败/unknown、未执行及 C 是否自然使用插件，允许无增益或成本更高。只有完整结果与报告具备后，才能据实收尾当前探索性 S5；启动不等于完成，也不升级成严格隔离实验。
