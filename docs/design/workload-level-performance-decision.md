# Workload 级性能决策与确认

日期：2026-10-09。接续 [持久化入库 H20 验收](durable-corpus-h20-acceptance.md)。

## 1. 问题与范围

上一轮三个真实 admission 已证明 durable corpus 的提交与幂等恢复，但其契约没有声明
required cases、权重、回退上限，最终 implementation_decision 仍为 inconclusive。
`persisted` 不等于整个 workload 都值得部署。

现有实现已有多 case 账本、聚合判据和三态建议。本轮不重写这些能力，修复其生产接线：

1. 非主 case 原先只提取加速比，没有逐 case 的测量资格；
2. 首测按聚合判门，独立确认与入库却仍只读主 case；
3. 必判覆盖/缺失回退证据未在永久副作用前闭合；
4. 失败产物没有完整部署建议，报告仍声称聚合不进门槛。

范围为固定候选的 workload 评测、确认、建议与提交。没有验收真实 LLM 搜索能力，
也没有把搜索中的主 case GFLOPS/bandit/roofline 状态整体重构为分布优化器。

## 2. 冻结与逐 case 资格

继续复用 FrozenContract 的 `cases`、`required_cases`、`case_weights`、`aggregate`、
`max_regression` 和 `min_speedup`。数值参考、精度、输入原值、完整 evaluator 与量具
profile 仍由 durable manifest 冻结；Agent 不得改写。

非主 case 现在进入与主 case 相同的 L3/Oracle 流程，只跳过递归账本和整体目标判门。
每行保留原始耗时/加速比、配对身份/窗口、timing/reference receipt、心跳、有效性、
资格掩码和拒绝原因。主 case 不重复测，同形状身份仍按既有规则去重。

workload 资格是以下集合中各 case 掩码的交集：

```text
required_cases ∪ weighted_cases ∪ regression_limited_cases
```

没有这些声明时，显式退回主 case 口径；额外诊断 case 不偷偷改变已声明目标。
缺失 case 或资格不完整按 all-false 合成，不能删除坏 case 后重新归一化权重。
原始不合格读数仍保留在账本，但不进入聚合。degraded 可按原策略用于有限判决，
不能借用主 case 的 valid 写入长期记忆；uninformative 不支持整体结论。

`workload_qualification` 发布实际 scope、有效性、资格和原因；顶层 Observation 的
资格/档位与之保持一致，逐 case 的原 Oracle observation 不被覆盖。

## 3. 一个目标口径贯穿判门、确认与入库

`eligibility.decision_speedup` 是显式目标的公共读口：

- 有 objective receipt：读取聚合值，缺失/非有限不退回主 case；
- legacy primary_case：保留旧单 case 读法并明确标注范围。

入库 score、确认 speedup、逐轮 speedup 留痕与 Search 提示使用这一读口；主 case
的 `speedup_vs_baseline` 保留为诊断。目标达标不额外要求每条 case 都超过聚合门槛，
但显式硬回退上限优先，回退证据缺失也不能通过。有回退声明但没有聚合声明时，
回退门仍生效。已声明加速比目标却无 baseline，不得换成绝对 GFLOPS 成功。

ConfirmRecord 增加 `performance_basis` 与 `workload_evidence`。
`confirm_metrics` 是入库与显式 workload 建议共享的确认入口：

1. 校验两次完整 frozen workload contract 的内容哈希和实际 contract_hash；
2. 对照冻结契约核对 case/权重/聚合/回退/门槛，不能只比两份自报声明；
3. 首次整体资格可持久化，第二次整个 scope 可判，整体目标与回退门都通过；
4. 从账本复算资格和目标，逐行值/档位/哈希/掩码必须与 Oracle receipt 一致；
5. 每条真实 case 两臂同窗，所有首测/确认 case 窗口互不复用；删掉设备标记不豁免；
6. 继续要求原确认协议的独立性、相同契约和门槛。

单 case 的显式 workload 也保留完整账本/窗口；旧格式不被追溯升级为新协议。
第二次仍执行新 L3，而不是再跑完整 L2 suite。允许 degraded 第二次确认的既有规则
未放宽；它必须是所有 scope 都可用且整体仍通过的第二窗口。

入库 intent/seal 与恢复审计核对的是同一个目标 score 和完整确认凭据。仅恢复副作用
不重放 GPU/LLM/预算/bandit，也不把历史结果重新颁发为 live credential。

## 4. 建议与失败边界

复用既有 use_custom / use_library / inconclusive 判据，补充性能依据：

- 显式目标建议 use_custom 需要完整目标判门与独立确认，不等同于文件落盘成功；
- 有资格的明确回退越限可建议 use_library，即使 L3 未通过；
- 覆盖缺失、坏尺、目标未达/未确认不能建议 use_custom；证据不足落 inconclusive。

失败产物也发布 claim、objective、decision 和未入库投影，报告不再遗漏负结论。
测量未知不是候选正确性失败。拒绝部署建议不代表可以把基础设施故障学成坏配方。

## 5. 验收方式

本地确定性后端假载荷验证边界，不能冒称真实 GPU 性能：非主 case 污染/离散度/地板/
跨卡/跨窗/缺失/契约漂移，二次聚合失败，主 case 未达但整体达标，回退越限，NaN/Inf，
Oracle 值不一致、坏尺、双边缩窄自报 scope、缺失窗口，以及实际本地 seal/新进程恢复。

真机驱动 `benchmarks/workload_decision_h20.py` 默认为只写本地 frozen plan。
计划固定 FP16 LayerNorm unfused-fp32 候选、v2 参考、1.05× 门槛、等权几何平均和
每 case 5% 回退上限。不是来自真实业务流量的权重，也不宣称覆盖全部 shape：

- 正例：8192×15360 与 8192×12288，两条输入各自严格低于原 256 MiB 上限；
- 地板负例：额外声明并测量 32×64 tiny case，不调整尺子让它通过；
- 缺失负例：声明两个大 case，但只提供 8192×15360；缺失另一大 case，必须给出
  缺失证据而非错误核结论；
- 权重漂移 resume：改变已冻结权重，零该 trial RPC/生成/语料副作用 HOLD。

明确回退越限用本地 policy fixture 验收，不伪造真机慢核证据。
源码、候选、参考、声明、预算与计划在远端启动前冻结；每次原始结果均保留，不按
读数扩 shape、改权重、降门槛或重试到 valid。实际 corpus 必须有合法 seal 和 fresh
replay；清理必须证明 owned runner 停止且私有 scratch 已删除。

```sh
python benchmarks/workload_decision_h20.py                 # local plan only
python benchmarks/cuda_calibration_h20.py --gpu 2 --ruler-profile
python benchmarks/workload_decision_h20.py --execute --gpu 2 \
  --cuda-ruler-profile /absolute/path/to/new/ruler-profile.json
# 只补未完成负例；旧正例仅供身份审计，不恢复成 live credential，也不重跑性能。
python benchmarks/workload_decision_h20.py --execute --gpu 2 \
  --cuda-ruler-profile /absolute/path/to/new/ruler-profile.json \
  --continue-negatives-from /absolute/path/to/positive/artifact \
  --negative-trial missing-declared-case --negative-trial resume-weight-drift
python -m pytest tests -q
python tests/golden_regression.py compare
```

驱动区分功能边界验收和正例性能验收，诚实 HOLD 不等于真机验收完成。
本节描述计划与实现，不把上一轮两个独立 primary 的读数拼成新版 workload 证据。

## 6. 本轮结果

### 6.1 本地回归

- `python -m pytest tests -q`：**3520 passed、1 skipped、2 warnings**。
  warnings 是既有非有限参考负例触发的 NumPy subtraction 警告，未屏蔽。
- Golden 最终串行执行：**4/4 PASS**，没有改 registry 或容忍窗口。
  初次与全量测试并行时 LayerNorm 曾落在容忍窗口外（3/4）；该日志保留于
  `/tmp/operator-workload-golden.log`。随后两个无并行测试负载的串行结果均 4/4。
  这属于 Mock/CPU Golden 的调度噪声排查，不是挑选 H20 性能窗口。
- 日志：`output/workload_decision_validation/full-tests-final-stable.log`、
  `golden-final.log`；确定性边界用例不宣称 GPU 性能。

### 6.2 固定 workload 正例

设备为 `algorithm-h20` 的物理 GPU 2，选用新预检 profile：

| 项目 | 结果 |
| --- | --- |
| profile hash | `97dec1cd4d762682d713271146ac1fcd76b4c3b813842247083edd61d9348899` |
| calibration identity | `38f243e601321b5cde955804f81c29397569bff261476c468eebc368c5e17d0d` |
| native launch floor / absolute anchor | 4.64 µs / 0.0104 ms |
| 完整生产 protocol hash | `e2a0a70f0844cb6933ca3610792b1252ba4100443f57cc44abad25648ce93001` |
| 首测 objective / 确认 objective | **1.9864502611× / 1.9862898278×** |
| 最终建议 / claim | **use_custom / persisted** |
| 实际 sealed events | **1**；fresh replay 与原记录完全一致 |
| 新进程恢复两次 | **already_committed**；相同 admission ID，语料字节未变 |

这不是调低默认地板：legacy 20.24 µs 路径与原 256 MiB 输入存储协议上限均未改。
新 profile 来自候选试验之前的两个独立 native audits，固定后用于全部试验。

每个大 case 的首测与独立确认都 valid，严格数值验证与 v2 32-case suite 通过。
实际耗时和原始比值如下，聚合为等权几何平均，而不是主 case 的别名：

| case | 首测中位耗时 | 首测 speedup | 确认中位耗时 | 确认 speedup |
| --- | ---: | ---: | ---: | ---: |
| 8192×15360 | 149.184 µs | 2.0182× | 149.104 µs | 2.0187× |
| 8192×12288 | 117.056 µs | 1.9552× | 117.088 µs | 1.9544× |

四个 case 测量窗口不重复、每个窗口两臂配对一致；首测/确认分别有独立 native
heartbeat 与绝对 anchor 审核。设备仍有 7 个其他 resident peers，观测 util 为 0，
不能声称独占卡。首测 IQR/median 为 0.0073 / 0.0036；确认 0.0079 / 0.0044。
确认 max/median 为 1.1948 / 1.566，长尾原样保留，没有删样本或追加窗口。

正例产物：
`output/workload_decision_h20/workload-decision-nnl_dfpm/artifacts/two-large/layernorm/nvidia_gpu/20261009_181125_1d6d624f`。
真实隔离语料：该 run 下 `corpus/two-large/kernel_db.jsonl`。

- admission ID：`0981db73bd1fabd971501cf2536cb3acc7408370e50d2e5d45de105d0e0c12dc`。
- record hash：`26a53baf4d001482d80d9ac731b548e4374262dfd7b4cd553b45c8bf26f98600`。
- 最终 corpus hash：`3a883f7c26702077f3a4ba2d312eefe7c450d78913ab2364dfe9308cd9a3deb8`。

### 6.3 负例与失败历史

本轮验收是**三个冻结 session 中四个指定 trial 的组合审计**，不是一次全成功
campaign。旧的 partial summary 没有被覆盖为 completed；完整失败轨迹继续保留。

1. 最初 JSON tuple/list 表示差异误触契约漂移，候选未派发、未测量、未入库。
   修复为 JSON normal form 比较并增加回归用例后重新冻结计划。
2. 主 campaign 正例成功，但后续 v2 suite 的 reference preflight 被低估，256 RPC
   预算耗尽；tiny 与缺失 trial 不完整，不冒称数值错误或已验收。
   根据固定调用账目把**驱动传输预算**改为有界 512 RPC，门槛、shape、权重和量具未改。
3. 第一次负例补测中 tiny 完整通过正确性检查；32×64 的 5.6 µs / 原始 1.2114×
   被 Oracle 判为 **uninformative / launch-limited**。整体 `inconclusive`，未确认、未入库。
   同 session 后续缺失 trial 的实际输入选择与计划不一致，结果不计入验收；
   排查修改了驱动源，下一权重 trial 被 frozen evaluator 检查拒绝派发，也不计验收。
   两类诊断均保留，增加了实际 supplied_cases 派发前与计划严格一致的检查。
4. 再单独冻结**仅缺失与权重漂移**两个未完成 trial：缺失试验只提供第一个大 case，
   已测 case 数值正确，但另一大 case 缺失，整体五个资格位全 false，objective=None，
   建议 `inconclusive`、零 corpus 副作用。权重漂移 resume 为 **identity_mismatch**，
   该 trial 零 RPC、零 Designer/Coder、零入库。

明确回退越限仍只由本地 `[2.0, 0.90]` policy fixture 验证：几何平均达门槛，但第二条
超 5% 硬回退上限，建议 `use_library`。没有将它伪装成真实 H20 慢核。

| evidence session | plan hash | 被验收的 trial | 原 summary |
| --- | --- | --- | --- |
| `workload-decision-nnl_dfpm` | `31d6b517d8122d0490c15a6043125ca28f00702eee5981b28028171b18ab341b` | two-large | completed=false；正例 performance_completed=true |
| `workload-decision-4mgau4kt` | `7f568c0473f506f3afd833c544f94dff29fa6013910adfd66fede3906d6c6a3a` | tiny-required-weighted | completed=false；tiny errors=[] |
| `workload-decision-7b3zfc4k` | `050fdf099a3c1a4082106172dd75a63f74614c0701492cb55f751a21d3ad3cc9` | missing-declared-case、resume-weight-drift | completed=true；不包含正例性能重测 |

各计划保留自己的驱动源码 hash；前两版本驱动也单独保存，供重新核验，不强行用
新版源码匹配旧计划。全部选定试验的生产 evaluator/protocol、候选、参考、契约、
量具保持相同；历史正例仅作只读身份上下文，没有颁发新的 live credential。

### 6.4 审计与边界

独立组合审计写入 `output/workload_decision_validation/actual-audit.json`，completed=true。
它核对选定 trial、计划/源码 hash、原始 Oracle 账本、目标复算、配对窗口、量具心跳、
seal 与 fresh replay；12 个性能/心跳窗口全部独立；三个负例的隔离语料为空。
在全部 remote owned runner shutdown ACK、PID quiescent、私有 scratch removed 之后，
才执行两个新进程 CLI recovery，日志为同目录 `recovery-0.log` / `recovery-1.log`。
全部清理 signals=[]、forbidden_actions=[]；未安装依赖、改共享运行时或杀共享进程。

只证明固定候选在这**两个声明的大 case、等权目标、冻结精度语义**下可选自定义实现。
不能推成全 shape 性能保证、真实业务流量最优或 LLM 自主搜索已验收。所有真实入库
位于本轮隔离 corpus，并未自动推广进项目的其他长期语料。
