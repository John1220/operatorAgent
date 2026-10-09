# 冻结运行契约持久化与失败关闭恢复

日期：2026-10-08；验收更新：2026-10-09。接续 `reference-identity-and-v2-profile.md`。

## 1. 修的是什么

之前参考身份只在 Verifier 内存中按 task_id 绑定。重新启动进程后，恢复快照会
沿用旧 best/验证结果；快照损坏、缺失或任务不匹配还会静默重新生成。
这不能证明“现在的候选、输入、尺子”仍是旧证据描述的对象。

现在 **显式 resume 必须有完整且一致的证据，否则 unknown/HOLD**。
不会自动改成 fresh run，不调用 Designer/Coder/Fixer，不把未知变成候选失败，
也不因此写正/负语料。合法恢复仍须重新运行候选的 L1/L2/L3。

## 2. 文件与绑定

生产 staging 和最终 artifact 都有：

| 文件 | 含义 |
|---|---|
| `contract.json` | 完整 FrozenContract，沿用原内容 hash/可信注册表校验 |
| `run_manifest.json` | `frozen-run-v1`；run_id、契约/参考/suite、实际 workload、协议、控制参数和执行环境 |
| `execution_snapshot.json` | 原状态 schema + run_binding，包括当前/待修复候选摘要、整个 checkpoint 摘要 |
| `confirmation.json` | 实际做过独立重测才生成；两次 verification_id/binding、第二次指标、ConfirmRecord |

新 run 的 staging 用 UUID 命名，不再依赖 task_id 加秒级时间戳。
contract/manifest 写成 0444；这用于减少误修改，**不是安全隔离**。
result.json 只登记摘要和 resume/audit 键，不整体序列化任意 metadata。
每份新 L1/L2/L3 结果携带 run_id、manifest/contract/workload/protocol/candidate 摘要
和新的 verification_id。确认不能拿候选 B 的结果替候选 A 背书。

### 实际 workload，不仅是 shape

通过已有 `describe_inputs` 记录每组调用输入的 shape、dtype、stride、offset、
storage 分组/可写性、整个 backing storage 内容摘要、logical values 摘要，
以及标量类型和内容摘要。不把大数组/base64 写进 manifest。

因此同 shape 换值、fp16 换 fp32、连续换转置/offset view、换标量或 alias，
都不是同一组证据。FrozenContract 的历史代表 shape 不冒充实际调用输入。

完整任务声明还绑定 source/reference_source、语义参数、目标/编译配置；
外部 code_path 非空时绑定该文件内容。任务创建时间不参与身份。

### 测量协议

摘要覆盖明确列出的生产 evaluator、correctness/reference/input-fidelity、
admission/confirm、控制端 executor 和远端 runner 源文件，不只写一个手工版本标签。
target_gflops、heartbeat_interval、调用方 expected_output 也进入 manifest。
代码升级甚至相关文件注释变化都会让旧 resume HOLD——刻意保守，不自动迁移。

环境保存已有稳定指纹，硬件回报了 selected_gpu/driver/CUDA/UUID 时也绑定它们；
不伪造缺失字段。v2 仍必须通过实际运行时参考身份探测/回执检查。
没有把 MockGPU 的声明当成真实 GPU 证明。

## 3. 时序

```text
声明 / 正确性静态预检 / 解析实际 inputs
  -> 若 resume：读快照 + 两份 sidecar，重新冻结当前任务并比对全部身份
  -> 参考身份 preflight、已有环境准备、硬件身份检查
  -> 完整 contract + manifest 原子持久化
  -> Router / Designer / Coder
  -> 候选出炉后再次检查 live task / inputs / sidecar
  -> 实际 L1 / 全套 L2 / L3，盖本次结果绑定戳
  -> 再核 live identity，seal checkpoint
  -> 最终 artifact 复制同一份不可变 sidecar
  -> 独立重测，同一候选绑定，持久化 confirmation
  -> 原有 admission gates / 报告
```

文件不一致的恢复在 GPU/路由/生成前停止；实际执行环境不一致必须探测之后才知道，
同样在生成前停止。轮首、候选生成后、验证后、快照写入和确认/入库前都核对绑定。

## 4. 合法恢复也不复用旧通过结果

- 保留轮次和 sketch、current_code/pending_fix；下一轮直接验证保存的候选，
  不重新让模型生成，也不拿模板选择替换它。
- 旧 verification_results、iteration_history 和 best 写入 resume_audit，标明
  `previous_process_audit_only`，不再作当前进程的通过凭据。
- best、停滞、优化信念、applied/disabled recipes 和 verified parent 从本次证据重建。
  不拿旧基率、坏点/性能读数支配新测量。
- 新验证通过后，如要入库，仍走原来的独立 L3 确认和资格门。
- 没有剩余轮次直接 HOLD；不会零次新验证就再次晋级。
- final snapshot 保留真正尚未消费的 pending_fix，而不是收尾时无条件丢弃。

按 task_id 检索时有不可读记录不能猜“旧的那份一定是对的”，要求明确快照路径。
旧无 sidecar/无 seal 快照不自动补齐；只能查看历史，或由用户明确发起新运行。

## 5. 原子性与非有限诊断

同目录唯一临时文件、文件 fsync、os.replace、目录 fsync；写失败清理临时文件，
不破坏旧完整 checkpoint。sidecar 存在但不同/损坏不能被“重新写正确”掩盖。
先让 sidecar durable，之后才发布 sealed checkpoint。

这是**逐文件原子写，不是三文件事务或事件日志**。一半 sidecar 不构成有效恢复凭据。
生产关键写失败转 HOLD，不能按以前“快照旁路失败忽略”继续晋级。

checkpoint/confirmation 审计里的 NaN/Inf 显式保存为 `$nonfinite` 标签，
不填零、不把非有限参考变成候选 passed；manifest 中数值声明仍严格拒绝非有限值。

## 6. 明确边界

1. 哈希检测正常漂移/损坏，不防恶意同 UID 同时重写文件、seal 和注册代码。
2. 不证明候选内部 fp32 算术、全部传递依赖或内存 bytecode；v2 的既有限定仍适用。
3. 输入必须由调用方可再提供/契约可物化；manifest 不储存完整 tensor checkpoint。
4. 暂无全局事件事务、跨进程 exactly-once admission 或累计预算消费恢复。
   当前会话仍受 BudgetPlan 约束，但不是完整的 durable budget ledger。
5. 这一轮不是原搜索轨迹逐步等价恢复：保存候选，丢弃旧调优信念并重新取得证据。
6. 不能从 L3.passed 推断可入库；真实测量 validity/can_gate/eligibility 和双次确认
   必须仍独立成立。没有放宽容差、launch 地板、噪声带或准入门槛。

## 7. 验证入口

```sh
python -m pytest tests/test_run_contract.py tests/test_resume_integrity.py -q
python -m pytest tests -q
python tests/golden_regression.py compare
python benchmarks/calibrate_run_contract_guards.py
python benchmarks/durable_profile_h20.py --gpu 2
python benchmarks/durable_profile_h20.py --gpu 2 --resume-only
python benchmarks/durable_profile_h20.py --gpu 2 --workload 1024x1024 --workload 8192x8192
```

H20 driver 固定三个调用方 workload：1024×1024、4096×4096、4096×8192，
fp16 周期整数输入；候选无 shape/value 特化，仍须过原 32 组参考校准 suite。
显式 `--workload` 只允许这三组和扩展 8192×8192，按命令声明的顺序执行；
该选择在环境准备/任何测量前完成，并作为实际输入进入 manifest。
恢复和漂移负例使用首组实际 workload，不偷换成历史默认 shape。
min_speedup 在任何测量前固定为 1.05。
只 stub 生成，实际执行 Conductor/L1/L2/L3/确认/入库；在隔离 corpus 验证落盘。
另测独立 Verifier 的合法恢复、输入/容差变化的零 RPC HOLD。
至少一个真实正向 admission 才能标记整体 completed；否则保留 functional 结果，
如实报告性能/入库未确立。全部只用 owned private scratch，最终核对清理。

## 8. 验收记录（2026-10-09）

### 本地守卫与回归

- durable-run 专项测试 47 条：真实 Conductor/ArtifactStore I/O、跨进程恢复、
  损坏/缺失/symlink、实际输入身份、声明/协议/环境漂移、关键写失败、双次候选绑定。
- H20 driver 增加 6 条离线参数/回执测试，拒绝超出白名单的无限制 shape 和冲突的验收模式；
  未知 RPC 的缺失参考回执必须记录为验收失败，不能提前抛异常丢失该 trial 的 trace。
- 最终全量测试：3120 passed、1 skipped、2 warnings；两条 warning 来自既有非有限参考负例。
- Golden Regression：PASS 4 / REVIEW 0 / FAIL 0 / SKIP 0；未改注册基线/数值门。
- 独立副本的 clean 通过；8/8 守卫变异被断言杀死，不把收集/import 错误算成功。
  覆盖 seal、实际 workload、声明漂移、sidecar、只读契约、恢复重新生成、耗尽轮次晋级、
  忽略关键写失败。首轮 sidecar 变异曾存活，补 live-sidecar 负例后复测通过。
  原始日志保存在
  `output/run_contract_guard_calibration/run-contract-guards-20261009T015412784841Z-17963-1/report.json`。

### 真实 H20 恢复验收

最终源码运行 `durable-profile-h20-20261009T014438805558Z-2322-1`：

1. 新运行实际 L1、32 组 suite 加主输入的 L2、真实 L3、独立确认。
2. 另一 Conductor/Verifier 恢复保存候选，零 Designer/Coder 调用，重新编译/验证。
3. 同 shape 改输入值：unknown/HOLD，零 RPC、生成和 corpus 写入。
4. 仍满足 v2 声明规则、但进一步收紧 max_error：unknown/HOLD，同样零副作用。

**功能 4/4；不等于性能/入库通过**。该小 workload 的主测和确认均 uninformative，
所以 positive_admissions 为空、整体 completed=false（driver 非零退出是刻意的）。
owned runner 收到 shutdown 回执，quiescent=true、removed=true、signals=[]，
forbidden_actions=[]。
证据位于 `output/durable_profile_h20/durable-profile-h20-20261009T014438805558Z-2322-1/summary.json`。

### 首轮代表 workload：拒绝入库也是验收事实

| 主输入 shape | 主测表观加速比 | 独立确认表观加速比 | 主测 launch 占比 | 资格 |
|---|---:|---:|---:|---|
| 1024×1024 | 1.4118× | 1.4054× | 74.11% | uninformative / can_gate=false |
| 4096×4096 | 1.7735× | 1.7761× | 44.87% | uninformative / can_gate=false |
| 4096×8192 | 1.7039× | 1.7017× | 31.15% | uninformative / can_gate=false |

三组数值验证通过、实际执行了主测和确认；**不能把这些表观加速比发布成可信收益**。
launch 地板仍固定 20.24 µs，30% 门未改，三组均未写 corpus。
原始记录：`output/durable_profile_h20/durable-profile-h20-20261008T131553946484Z-64309-1/`。

该首轮整体 functional_completed=false：阈值负例最初使用“放宽 10 倍”，
被既有 v2 静态参考门先拒绝，而 harness 错误地只期待 durable-run 标志。
随后改为合法的“收紧一半”负例，以上真实恢复复测 4/4 通过；旧失败记录未删除。
更早一次 harness 未创建 report 根目录的失败也保留，不能计入成功样本。

### 扩展 8192×8192：输入发送阶段超时，尚无 L3 证据

扩展运行 `durable-profile-h20-20261009T015239243495Z-14094-1`，在首次测量前
声明两组 workload：1024×1024、8192×8192；阈值仍为 1.05，数值/测量门均未改。
小 workload 仍 uninformative；8192×8192 的 128 MiB 输入在 180 s RPC 截止前
只发送了 10,059,776 字节：`session_phase=send`、`session_state=timeout`、
`request_id=149`、`transport_outcome=outcome_unknown`。

生产侧正确保留 unknown，未把它当成数值失败，未运行 32 组 suite/L3，未入库。
不能据此推断大 shape 算不对、性能差、或者已跨过 launch 地板。
本次破损会话没有 shutdown 回执；控制通道核对本次 runner 的生命周期身份后
发 SIGTERM，读回 owned process exited，最终 quiescent=true、removed=true。
没有 broad kill 或 forbidden action。

旧版 harness 在核对缺失回执时抛异常，遗漏该 trial 的 compact trace，但原始
verification.json、sealed snapshot 和完整运行日志仍保留。
已改为累积验收错误、保存 trace，不把异常/缺回执变成通过，并添加离线负例。
这项 harness 修复不冒充成功的大 workload 真机复测。
原始结果位于 `output/durable_profile_h20/durable-profile-h20-20261009T015239243495Z-14094-1/`。

**当前结论：durable identity/resume 已验收；可信正向 L3/实际 corpus admission 尚未确立。**
下一步先解决大输入重复 JSON/SSH 传输：内容摘要绑定的 session-owned 输入缓存/分块传输，
仍要求执行侧完整 readback 并保持 alias/stride/offset，保留端到端截止与 owned cleanup。
之后再做大 workload 的 L1/完整 L2/主测/独立确认/真实入库；不是放宽地板或用缺证据的
表观速度替代性能验收。
