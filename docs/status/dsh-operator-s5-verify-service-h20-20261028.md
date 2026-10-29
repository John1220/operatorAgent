# S5：统一验证服务与一次真实 H20 L2（2026-10-28）

状态：**监督器侧统一验证入口已实现（mock 与 SSH 两条后端），一次真实 H20 GPU 7 的 L2 复核通过。** 这是"共同真实验证路径"的接线与单次演练，不是三组对照，也不是性能或质量结论。

## 服务契约

`benchmarks/s5_verify_remote.py`：三组最终候选都提交到这里，由**监督器**执行，而不是受试 agent。

- 计划冻结：声明（任务/形状/精度/后端/DSL）、停止级别（固定 L2）、墙钟、GPU 身份（索引 + UUID + 名称）、本地代码与远端 runner 脚本哈希；写一次、哈希可复核。
- 提交顺序：先落盘候选哈希 → 服务自己 `prepare` 出契约（不接受 agent 提供的契约）→ 由服务写候选文件 → 调生产 CLI `operator_agent verify --stop-after L2`；判定只写一次，不回灌给同一组。
- 证据核验：`operator-tool-evidence-v1` 完整、实现稳定、契约身份一致、后端未降级；L2 通过时还要求完整 correctness suite 全通过、`num_input_sets=1`、输入保真读数全部 verified；L2 拒绝时要求拒绝可归因到 correctness 门。任何缺项、身份漂移、超时、非 JSON 输出 → `unknown`，不写成通过。
- 后端：`mock`（可信本地 fixture）与 `h20`（生产 `RemoteExecutor` SSH + 持久会话）；GPU 固定要求索引/UUID/名称三者齐全，与生产选卡逻辑一致。

测试 `tests/test_s5_verify_remote.py` 10 项：计划不可重入、契约绑定、mock 通过与拒绝、重复提交拒绝、超时/非 JSON 记为 unknown、证据被篡改时拒绝、GPU 身份校验与下发。

## 真实 H20 复核

任务沿用 S4 已冻结的 LayerNorm fp32 `[[7,33]]` triton 声明；候选复用 S4 GPU 6 批次里最终 L2 通过的 `repair-1.py`（1156 字节），因此本轮测的是**服务路径**，不是候选质量或模型能力。无模型调用，固定一次，不重试。

| 项目 | 结果 |
|---|---|
| 计划 SHA256 | `735ddbac9fdc123e4c38b825028c74e5dc61a2d64eaf46b7ad8904126bf94ec2` |
| 状态 / 墙钟 | passed / 106.572 秒 |
| 级别 | L1 通过、L2 通过（12 case suite 全通过） |
| 后端 | `SSHExecutor@algorithm-h20(nvidia)`，`degraded=false` |
| GPU | 固定 GPU 7（`CUDA_VISIBLE_DEVICES=7`，运行前 util 0%、free 95.1 GB） |
| 输入保真 | `input_fidelity_readings` 全部 verified，`num_input_sets=1` |
| 证据 | evidence `e3eaf1a1…`、receipt `7c8111cc…`、payload `0ca39887…`，tool_version 9 |

材料在 `output/plugin_completion/20261028-s5-verify-remote-h20-v2/`（27 文件、约 260 KiB，Git 忽略），含设备快照、计划、提交、候选、完整证据与结论。用**更严格**的规则（套件完整性 + 输入保真）对存证离线复核，仍通过；没有重跑 GPU。

### 第一次尝试失败（0 次模型调用、未完成编译）

v1 只下发了 GPU 索引 `OPERATOR_AGENT_REQUIRED_GPU=7`，生产选卡要求索引/UUID/名称三者一致，于是在硬件探测阶段失败（L1 FAILED，7.341 秒）。这是接线缺陷：已改成必须同时冻结 UUID 与名称并加测试；v2 是唯一一次完整 GPU 验证。v1 材料保留。

## 限制

- 只验证一个任务、一个候选、一次运行；不证明其他形状、其他 DSL 或纳秒级性能，也不产生性能结论。
- "GPU 7 空闲"是运行前的只读快照，不是租约；运行中可能有其他作业开始，证据里的 `gpu_lease=false` 是刻意的。
- 远端清理只到进程级：运行后只读检查未见 `runner_server.py` 残留（pgrep 会匹配到自身命令行，需按此解读），但不证明逃逸后代或服务商侧状态。
- 服务仍是"配合的调用"：它能证明经过它的验证被记录，不能阻止同机绕过；隔离仍未建立。
- 三组尚未接入该服务；下一跳是把 A/B/C 的最终提交都改成调用它，先用 mock 跑通三组，再谈真实模型的三组对照。