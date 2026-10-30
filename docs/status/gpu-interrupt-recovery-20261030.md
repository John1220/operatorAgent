# GPU 执行中取消、超时与后续任务验收（2026-10-30）

状态：**GPU2 四项真实验收全部通过**。本轮修复默认入口对常驻 SSH 会话的清理行为，然后顺序执行取消、正常新任务、超时、正常新任务。模型调用 **0**、L3/性能测试 **0**；没有新增架构专用优化。

## 默认行为修复

原受控 S4 测试需要 `OPERATOR_AGENT_S4_GRACEFUL_CLEANUP=1`；普通 CLI/MCP 与自主运行取消只保证本机 worker 被回收，远端状态为 unknown。现在常驻 SSH 默认通过共同的 owned-worker scope 清理自己注册的后端，关闭预算 10 秒、监督者宽限 12 秒；普通成功退出也先完成会话关闭。取消/截止期间重复 SIGINT/SIGTERM 不打断清理回执，单个后端关闭异常不能遗漏其余已注册后端。

只有本次会话的身份检查与退出回执成立，才报告 `remote_execution=quiescent`。一次性 SSH、其它后端、身份不完整或清理失败仍不能证明远端停止；服务商状态与任意子孙进程仍为 unknown/未证明。没有自动重试超时请求、修改正确性/性能判据或退还已消耗的预算。

## 真实执行与结果

用户指定优先 GPU2。执行前固定该卡身份，观察空闲显存 22,246 MiB、利用率 0%；这是共享卡快照，不是独占租约。使用现有 Python/Torch/Triton 环境，把原样生产 runner 及共享模块上传到随机 0700 私有目录，完整核对源码哈希。没有安装/同步依赖或覆盖共享 runner。

慢候选是固定 Triton ReLU：先真正 launch 并同步，再提交固定 **40,000,000,000 cycles** 的 CUDA 延迟；没有无界循环。只在本次私有 runner 的同一 PID、出生时间和 GPU 身份下，取得两份间隔超过 0.5 秒的正 SM 活动观测，才认可进入实际 GPU 执行。握手、显存驻留与整卡总利用率不能替代该证据。这是中断生命周期探针，不是性能候选或加速结果。

| 顺序 | 测试 | 实际结果 | 整次 CLI 验证及后置观察耗时 |
|---|---|---|---:|
| 1 | GPU 执行中通过公开 CLI cancel | 同 PID 正 SM 1% / 45%；退出码 130、cancelled、worker 回收、远端 quiescent；未生成完整验证回执 | 13.678 秒 |
| 2 | 取消后在新目录运行正常候选 | 真实 CUDA，L1 与完整 L2 **12/12** 通过，输入保真与确定性检查成立；关闭 quiescent | 10.337 秒 |
| 3 | GPU 执行中的 8 秒 RPC 超时 | 同 PID 正 SM 2% / 52%；L1 通过，L2 `outcome_unknown / timeout / response`；没有 L3。工具正常返回未知判定，CLI 退出码 0；关闭 quiescent | 17.432 秒 |
| 4 | 超时后在新目录运行正常候选 | 真实 CUDA，L1 与完整 L2 **12/12** 通过；关闭 quiescent | 10.594 秒 |

上述耗时包含启动、验证、清理和后置观察，**不是取消响应时间或 GPU 核耗时**。超时是传输/执行结果未知，不是数值失败；实际路由为 infra/hold、不计搜索迭代、不污染学习。后续任务均是新的显式调用，不是恢复旧进程、复用旧通过结果或自动重试在飞请求。

四次结束后的独立观察均无本次私有 runner 的 CUDA 进程；显卡均恢复 22,246 MiB 空闲、利用率 0%。最终精确身份救援检查没有发现任何需要救援的 owner（owners=[]），私有目录已确认删除。代码执行前后哈希一致。

## 证据与复现

原始证据在 Git 忽略目录 `output/plugin_completion/20261030-gpu-interrupt-recovery-v1/`，含事前计划、109 个冻结源码文件、实际环境、固定候选、逐次 CLI 请求/结果、SM 活动样本、生产清理回执、完整 L2 判定与逐文件 SHA256 清单。独立离线重载核对 **213 个文件、30 项检查全部通过**，包括 L1/L2 的严格阶段、两次新任务的 12 个 case 各自首测/重复 CUDA 输入回读、超时 infra/hold 与不污染学习、取消观测先于取消请求、源码与候选绑定，以及所有清理回执。审计文件为 validation 目录的 `independent-evidence-audit.json`；不产生新的 GPU 请求或历史真实性签名。

四项来自同一冻结实现、一次顺序执行，没有按结果重跑。测试的是本轮源码；此前交付的 wheel 不自动包含这次默认清理修复。本地归档 `output/plugin_completion/20261030-gpu-interrupt-recovery-v1-evidence.tar.gz` 为 920,749 bytes，SHA256 `ce0fd0c13d796d25849e87d344ae799b9aadcaec81631c1fa5a0eff6138ca164`。该归档与原件在同一本地文件系统，按此前接受的单份本地保存口径留存。

驱动 `benchmarks/gpu_interrupt_recovery.py` 默认只冻结本地计划。显式执行需指定新的输出目录；指定卡繁忙或显存不足即停止，不改选其它卡。

```sh
python benchmarks/gpu_interrupt_recovery.py --gpu 2 --output /path/to/new-plan
python benchmarks/gpu_interrupt_recovery.py --gpu 2 --output /path/to/new-execution --execute
```

GPU 型号和 UUID 从执行时探测，不是驱动常量。该验收只覆盖受信任固定候选、持有的常驻 SSH runner 与显式新任务，不证明任意失控 GPU kernel 的 watchdog、敌意候选隔离、服务商停止或所有后代退出。

## 本地回归

默认清理相关专项 **85 项通过**；验收器正反例专项 **9 项通过**。后者拒绝只有握手/显存、非本次 PID、PID 复用、无 SM 活动、只有本机退出、启动阶段超时或未知验证进入 L3 等假通过。

全量 **6,172 passed / 1 skipped / 2 warnings**，506.34 秒；skip 是本机 CUDA 不可用，warnings 是原有非有限参考负例。

首次串行 Golden 为 **3 PASS / 1 FAIL**：CPU/mock Softmax 功能通过，平均延迟从历史 0.0062 ms 到 0.0869 ms。该次仅 5 个计时样本，min=0.0138 ms、max=0.3761 ms，单个最大值约占总耗时 86.6%；没有保存逐样本数组或首次 CPU 负载，不能确证长尾根因。Golden 直接调用 Conductor/MockGPU，不经过本次修改的监督入口；MockGPU、Conductor、Verifier、Golden 脚本和 registry 均未改动。首失败日志保留，不刷新基线或改容忍窗口；另做一次明确标注的诊断性串行复核。

一次诊断性串行复核为 **4 PASS / 0 REVIEW / 0 FAIL / 0 SKIP**，Softmax 为 0.0060 ms（历史基线 0.0062 ms）。诊断前记录本机 11 个逻辑 CPU、load average 约 4.01/4.10/4.06，实际 pytest 进程为 0；两次读数说明 CPU 计时不稳定，不能借复核通过追改首轮或声称已证明其根因。没有继续重复测试。

全量及两次 Golden 日志保存在 `output/plugin_completion/20261030-gpu-interrupt-recovery-validation/`；`validation-summary.json` 分列完整测试、首次比较、诊断性复核和独立证据审计。
