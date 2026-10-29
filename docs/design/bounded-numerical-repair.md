# 一次有界数值修复：失败源码、反馈与独立重验

日期：2026-10-15。接续[单候选数值失败记录](single-candidate-codegen-h20-acceptance.md#16-新计划获批执行包装接口通过padding-方差错误再次被数值门拦住)。

## 1. 目的与非目标

此前模型已连续产出把 padding 计入中心化方差的源码；继续增加通用提示并没有解决
问题。本阶段让模型看到**自己的失败源码和明确的数值反馈**，只允许生成一份修复
候选，然后重新验证。不是把可信模板当模型答案，也不由控制器替模型修改源码。

这是独立、显式启用的修复模式，普通生成仍为 `repairs=0`，失败后不会自动调用
Fixer。修复模式为 `single-candidate-repair-v1`、`repair_index=1`，caller 仅允许
`SingleCandidateRepair`，新 journal 仍只有一次请求、4096 输出 tokens 预留。
保留相同契约、native 参考、32-case suite、输入、容差、权重、10 倍接线目标、原
ruler、测量资格与确认/入库门。源码不同不等于修复成功。

第一版只接受 **L1 通过、primary 全量数值比较确实失败、suite 全部尚未运行**的
原始单候选运行。生成、接口、编译、传输、参考身份未知、suite 内失败、性能不足
等原因不在本版范围；不接受修复的修复。不能把 HOLD/unknown 包装成数值反馈。

## 2. 输入与证据血缘

`benchmarks/_numerical_repair.py` 只读核对父运行：

1. 原 plan hash、生成声明、模型原始请求/响应、usage、调用预算及 source hash；
2. 原 evaluator 的只读源码副本与其历史 hash，而不是要求旧实现等于当前实现；
3. artifact、verification、checkpoint seal、manifest、契约、工作负载、参考、suite
   和逐轮结果的交叉绑定；原始 compile/verify 回执及输入保真/参考身份；
4. 原失败未获性能、确认或入库资格，且 owned cleanup 记录明确 removed/quiescent。

父输入只读取描述和 hash，plan-only 不分配 432 MiB 数组，不执行或 import 候选。
单文件最多 2 MiB、总读取最多 16 MiB，拒绝 symlink/path traversal。所用证据按
相对路径/hash 冻结，复制到新目录 `frozen-parent/`；发起外部动作前继续检查原件、
副本与 evaluator。旧结果保持失败，不 resume，不回填或覆盖历史。

`operator_agent/bounded_repair.py` 严格限制外发 repair 对象：

- 原模型失败源码原文；父 plan/run/manifest/verification/source/evidence 身份 hash；
- 每个 primary 的 shape/case key、通过状态、全量比较数、bad 数、coverage、最大
  绝对/相对误差、匹配比例、nonfinite 数及未放宽的 atol/rtol/要求匹配比例；
- suite 声明数量与实际执行为 0 的事实。

反馈字段为固定白名单，不发送原始错误文本、rationale、输入数组、输出样本、
API key、模板源码或仓库文件。父源码是历史模型产物，不是可信候选模板。输入
名/shape/dtype/stride、完整契约/参考/suite/sketch/包装调用约定仍属于原生成上下文。
检查有限数值、精确类型、元素计数、匹配比例、case 与阈值一致性；额外字段即拒绝。

模型源码及其注释是待分析数据，不能成为执行指令。AST/precheck 仍仅作剪枝，
并不证明数值正确、无恶意或运行隔离。hash/seal 只防漂移，不防同 UID 恶意伪造。

## 3. 一次性额度与执行

显式 `--repair-parent-run <原运行目录>` 默认仅生成新计划。新载荷包含历史反馈，
**旧计划的授权不能覆盖它**；`--execute` 仍要求新计划路径、完整 hash 和另行批准。

执行前先消费普通 prepared-plan claim，再在固定
`output/workload_codegen_h20/repair-claims/<parent-identity-hash>.json` 原子、排他地
消费父候选的唯一修复机会。换 plan、输出目录或新 journal 都不能恢复这份额度。
失败、取消、准备未完成或崩溃都不退款；两个 claim 不做失败回滚。live guard 检查
claim 仍属于本次 plan 和 execution root。plan-only 不创建 claim。

`SingleCandidateCoder` 复用原有单调用/严格 JSON/原文保留/静态检查/Conductor 接线：
请求、反馈 hash、父候选 hash、父身份 hash、修复序号与新 source hash 一起进入血缘。
返回完全相同的源码被固定原因 `repair_unchanged_source` 拒绝；新源码不修补、不去
fence、不 fallback。源码改动不能获得正确性资格，仍须走 fresh verifier；第 6 节
随后增加 AST 未变即拒绝的守卫，使仅改注释/格式也不会进入 Verifier。

新候选完整重验 L1/L2；primary 和原 suite/repeat 都通过后才允许主测。上限仍是
1 次 compile RPC、66 次 verify RPC、2 个主测窗口；仅生产通过路径允许 2 个确认
窗口。一次 compile RPC 不是底层 JIT specialization 次数上限。失败即停，没有
第二次修复或挑选窗口，失败不能入库。最多 512 RPC/512 commands、1800 秒派发，
另留 60 秒确认 owned 清理。

执行仍是 H20 GPU 2、owned scratch、远端同 UID；没有敌意代码沙箱、网络硬隔离、
显存 quota 或独占 GPU lease。不安装、不修改共享 runtime、不杀共享进程。客户端
HTTP deadline 也不能证明服务端取消计算或精确限制供应商账单。

## 4. 本地阶段：计划冻结与回归结果

本阶段针对父运行 `workload-codegen-mn841e1w`，父源码 hash 为
`b3f95ac94f85e6027b54df920ef48425f200185c466eaeec89b21b1a539b8c95`，父计划已消费。
新计划只读冻结在 `output/workload_codegen_h20/workload-codegen-jrg52ja3/plan.json`：

- 完整 hash：`c507d78d65452fd5765d6873762c2aa0897130abad26cacc5cb31cf0b455801f`；
- production protocol：`d9be3ade6e9365948f6e41800e5f6bedcbf113844c97e2705487ff16680782ab`；
- 87 份当前 evaluator 源码，103 份父证据/历史源码副本；
- 冻结请求 17906 bytes，最多一次；模型为 `Qwen3.8-27B`，明文 HTTP，输出预留
  4096 tokens、超时 60 秒，无重试/fallback。具体主机地址仅在本地审批回执中披露，
  不写入公开文档；审批时不能用 endpoint hash 代替完整地址。

本地阶段没有外发、SSH、GPU、修复候选、性能成绩或入库。计划与父修复 claim 均
未消费，尚需针对此载荷的新批准。离线响应和 CUDA 回执是假件，不构成真机验收。
本地验证日志与不可变性核对保留在
`output/workload_codegen_validation/repair-local-dxtfejlt/`，最终成绩见该目录总结。

本阶段新增 35 项离线回归，专项 330 passed；全量 4114 passed / 1 skipped，2 个
原有 nonfinite warnings；随后串行 Golden 4 PASS / 0 REVIEW / 0 FAIL / 0 SKIP。
未刷新 registry 或容差，也未补测找绿。2943 份受保护历史文件、原 ruler 与 registry
hash 保持不变；当前生产源码/测试 hash 与验证前快照一致。没有 commit/push。

## 5. 获批真实执行：模型原样返回失败源码，修复守卫拒绝

2026-10-16，用户明确批准第 4 节的具体数据外发与隔离限制后，消费计划
`c507d78d6545`，以及父身份
`97574ba0293c306590bf89e7e371b19f74b7d8364b4c6e75eaea445fbf81417b`
的唯一修复机会。真实执行目录为
`output/workload_codegen_h20/workload-codegen-_conj61k/`，审批、预检及只读复核在
`output/workload_codegen_validation/repair-acceptance-59u_n_jl/`。

结果是 **`generation_failed` / `precheck: repair_unchanged_source`**，exit=1，
completed/accepted/deployment_accepted 均 false。不是网络错误、JSON 格式错误、
新候选编译错误或新一轮数值失败：模型成功返回严格 `{code,rationale}`，但源码
955 bytes 与父源码逐字节相同，hash 仍为
`b3f95ac94f85e6027b54df920ef48425f200185c466eaeec89b21b1a539b8c95`。

模型 rationale 把失败归因于大归约的精度问题，并声称改成了先均值、后中心化方差
的两阶段计算；但父源码原本已是这种结构，本次没有任何代码改动，也没有排除
中心化后的无效 padding lane。解释文字不是修复证据，不执行同一段已知失败源码
去重新确认旧结论。该守卫按预期在 Verifier 前停止，没有机械补代码或重试。

| 项目 | 本次真实结果 |
|---|---|
| 模型调用 | 1 次，caller=`SingleCandidateRepair`，17906-byte 请求 |
| 响应 / 调用墙钟 | 2472 bytes / 约 11.302 秒，reported model 与 finish=stop 正确 |
| 服务端 usage | 5865 input + 516 output = 6381 tokens；4096 输出预留已消费，不退款 |
| 准备证据 | 7 阶段全部通过，15 份回执 |
| GPU 查询快照 | GPU 2 为 H20，free 25312 MiB、utilization 0%；不是独占 lease |
| 远端工作量 | 4 RPC / 18 commands；4 个 RPC 都是 gpu_info |
| 候选验证 | Verifier/compile/verify/主测/确认/原生心跳均 0 |
| suite / 入库 | suite 执行 0，无完成候选 artifact、确认或入库 |
| 清理 | quiescent/removed、shutdown ack 均确认，owned PID absent，signals=[] |

CostTracker 约 `$0.002275` 是估算，不是供应商账单。此次没有新候选的正确性或性能
读数；父运行的错误数值只作为冻结反馈，不是本次新增测量。

生产正向 generation auditor 因预算已 aborted 而保持 completed=false，并保留
固定拒绝原因；这不意味着没有收到模型响应。另一个只读失败审计重新核对了真实
请求/响应、原文一致性、修复血缘、usage/费用重建、准备回执、两个 claim 与清理，
证据核对完成。raw/durable 候选验收不适用，没有把失败审计升级成候选资格。

审计前后 3208 份历史文件、父候选、准备计划、原 ruler 与 Golden registry 未变。
生产 protocol 仍为 `d9be3ade6e9365948f6e41800e5f6bedcbf113844c97e2705487ff16680782ab`，
本次真实执行阶段未改生产源码/测试，未重跑或回填第 4 节成绩，未 commit/push。

**两个 claim 已消费，父候选的修复额度已耗尽。** 不能复制父目录、换新计划或换
journal 继续请求。下一步应先做离线真实响应回归与诊断反馈设计，检查修复说明
和源码差异是否一致；未来实验须在新的明确任务/预算政策下审阅，不能借此绕过
本父候选的一次上限。本次没有继续实现新策略或发送新请求。

## 6. 随后离线补强：Repair Diagnoser 与 AST 伪修复检查

新增[离线修复诊断层](repair-diagnoser.md)：固化真实响应，保守识别未屏蔽中心化
方差的源码模式，给出绑定源码位置、独立标量反例和修复要求的诊断包。诊断只能是
hypothesis，未知不算通过；不执行或替模型改源码，不改变冻结判据。

新增守卫拒绝注释/格式等 AST 未变的伪修复；AST 改动本身仍不能证明修复。诊断
通过显式 `single-candidate-repair-v2` 上下文接入，默认 v1 声明不自动追加反馈。
独立入口只能生成 offline preview，无 execute/模型/GPU/额度操作，本阶段没有
新验收计划，也没有重置第 5 节已耗尽的父修复额度。新 payload 不获旧批准覆盖。

新增 49 项离线回归，专项 379 passed，全量 4163 passed / 1 skipped / 2 原有
warnings，随后串行 Golden 4 PASS。3439 份历史文件、原 ruler、registry 与旧
claim 不变。本阶段无真实外发/远端派发，未 commit/push。

### 随后补齐诊断预览的只读复核

新增独立 audit CLI，从冻结 case/response 重算诊断、源码证据与预览上下文。新
预览格式 v2 绑定文件清单和生产者源码快照；旧 v1 仅核对可重算的一致性，不能
补造缺失的生产者身份。两者都不是模型执行、授权、GPU 资格或新修复机会。
缺件、重 hash 篡改、协议漂移和 unknown 诊断的多余 context 均拒绝；不写旧证据，
不触碰额度，不创建真机计划。见[诊断预览独立复核](repair-diagnoser.md#8-后续离线接线独立只读复核与-v2-预览清单)。

新增 43 项离线回归，专项 422 passed，全量 4206 passed / 1 skipped / 2 原有
warnings，随后串行 Golden 4 PASS；3511 份历史文件、原 ruler、registry 与已
消费 claim 不变。本阶段无真实模型/远端派发，无 commit/push。
