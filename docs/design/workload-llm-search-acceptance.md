# 有界真实 LLM 提案搜索验收

日期：2026-10-10。接续 [固定候选最终协议验收](workload-search-final-protocol-acceptance.md)。

## 1. 本阶段证明什么

用现有配置中的真实模型，替换固定的初始提案与 Search 提案：首轮由模型选择
`num_warps`，第二轮由模型阅读首轮**完整且合格的 workload** 反馈，再选择一个
未使用的参数。两轮都走生产 Conductor、参考身份、L1/L2/L3、预算、选优、保存、
普通重载与 sealed checkpoint 校验。

这是**受限参数提案**，不是自由源码生成：模型只能返回 `num_warps` 和 rationale；
代码来自可信 unfused-fp32 LayerNorm 模板，只允许 `num_warps ∈ {4,8,16}`。
不能据此宣称 Designer/CoderPipeline/Fixer 的通用自主开发能力已经验收。
没有固定“4 然后 8”的模型桩；实际顺序由保留的模型响应决定，非法/重复选择即停。

## 2. 执行前冻结

驱动：`benchmarks/workload_llm_search_h20.py`；模型传输：
`benchmarks/_bounded_proposals.py`。默认 plan-only，不读真实模型配置、不调用 API、
不分配大输入、不 SSH。`--execute` 同时需要显式 GPU 和原 ruler profile。

- 既有 algorithm-h20 runtime、物理 GPU 2；只使用本次 owned scratch/runner。
- workload、参考、精度语义、v2 32-case 正确性 suite、等权几何平均、每 case 5%
  回退上限，均沿用前一验收。8192×15360 / 8192×12288，各输入 packet <256 MiB。
- min_speedup=10× 仍是**强制搜索的验收门槛**，不是新的业务性能目标。
- 原 profile hash：`97dec1cd4d762682d713271146ac1fcd76b4c3b813842247083edd61d9348899`。
  不重标 launch floor、不按新结果调整门槛、形状、权重或窗口。
- 两个不同候选，每个一次完整主测；max_rounds=3、oracle_total=5、heartbeat=1、
  总派发 1800 秒、512 RPC/commands，确认格不可挪用；零修复、零补测。
- 模型最多 **2 次请求（失败也占格）**，单次 max_tokens=512，输出 token 预留合计
  1024；请求 UTF-8 body ≤16 KiB，响应 ≤64 KiB，父进程强制单次 60 秒 deadline。
- 只使用本地已配置的单一 endpoint/model；禁用代理、redirect、fallback、试探请求、
  格式修复和自动重试。响应自报 model 必须与冻结请求一致；这不是模型权重的证明。
- LLM 总 token 账本仍由生产 BudgetPlan 跟踪，但**硬边界**是前置调用次数、请求字节
  与输出 token 预留；没有 tokenizer 不能宣称输入 tokens 或供应商账单被精确封顶。

计划在第一笔 API 请求前落盘为只读文件，包含模型 endpoint 的 hash、请求参数、
prompt、模板、完整 evaluator sources 和测量协议。另保留逐文件只读源码快照；
不需要为了冻结而自动提交用户工作区。旧试验不可覆盖或作为实时凭据复用。
模型 key 不进入 argv、证据或日志，只通过子进程 stdin 传给 transport。

## 3. 闭环和失败行为

1. 请求**先持久化 reserved 记录再派发**；成功响应/usage/自报 model/finish_reason
   保留原文与 hash。失败也保留调用格，不追加候选来“找个通过的”。
2. 模型只能给声明内的整数参数和非空理由；额外 source/门槛/依赖字段、重复参数、
   空输出、截断、未知模型、超预算 usage 都 fail-closed。
3. 第二次请求的反馈独立从首轮 raw L1/L2/L3 重建：自身 candidate/contract binding、
   全 scope 的资格与目标、两个 case 的 speedup/validity/window。GFLOPS 不进入
   排名反馈；首轮 failed/degraded/缺 case 不会换成虚构数字继续调参。
4. materializer 与 Verifier 入口都检查冻结计划，拒绝第三候选、重复派发、未提案
   源码或 Fixer 替代品。派发失败仍占候选格。模板照常经过生产静态/资源/执行门。
5. Conductor 会在第 2 轮结束时问 Search；此时参数预算已满，只返回**原 sketch**
   让下一轮进入 Oracle-budget stop，不调用模型、不创建第三候选，也不退回规则搜索。
6. 模型 lineage 审计核对真实 request→response→参数→源码→完整 workload；普通
   ArtifactStore 保存/重载还必须保存本次 usage。费用只是仓库价格表估算，不是账单。
7. 成功的两份完整 VALID、低于 10× 的观察，复用不改动的 raw-workload/durable
   auditors。候选列表是从真实响应证明的**审计输入**，另存 `audit-input.json`；
   不改写执行前计划，也不伪称模型源码在调用前已知。失败/部分/degraded 证据原样
   保存，但 summary.completed 不能为 true。
8. 整体通过还要求 shutdown ACK、owned PID quiescence、scratch removal。API/准备/
   评测失败同样进入 finally 清理，不安装依赖、不编辑共享 runtime、不杀共享进程。

性能/入库口径保持原生产政策。本阶段预期低于 10×，没有 confirmation/admission，
不写入长期 corpus、不授权 use_custom，不把“接线通过”当成“业务性能达标”。

## 4. 本地验证与真机记录

本地 fixture 明确使用 fake model 与 CPU/fake timings，只验证守卫和生产 hook；
不作为模型调用或 H20 性能证据。覆盖前置预算、deadline、无 key 落盘、非法输出、
响应/反馈/source 血缘篡改、模型身份、坏 plan、零修复、准备失败清理，以及生产
Conductor 两轮选优和第二次 API 失败后保存第一轮赢家。

本次最终源码的本地结果：

- 新增守卫 **51 passed**；连同搜索/持久化专项合计 **140 passed**。
- 全量 **3660 passed / 1 skipped / 2 warnings**，约 192.3 秒；warning 仍来自已有
  nonfinite correctness fixture，没有新增失败。
- 全量结束后串行 Golden：**4 PASS / 0 REVIEW / 0 FAIL / 0 SKIP**。默认容差不改，
  registry hash 保持 `38836e092efd7ca9773ce414c39271ab0f050e29318b27d3de9d57e417d60819`。
- 生产 measurement protocol 仍为
  `f14423ca3d71dd7a6fdb29500c8f25d5c05c76277ad12daef69f7c788a2290a6`；本次只增验收
  驱动、守卫和文档，没有修改生产协议，也没有 commit/push。
- 执行前 expected-plan hash：
  `7e58497cdecfebda395e37e88794cce6aa680d50f0f7790c39bf8a79d83822ef`。
  计划原文保持只读，真实试验与它完全一致。

### 4.1 授权边界

第一次执行申请被安全 auto-review 拒绝，因此当时 **model calls=0 / GPU dispatches=0**。
本地只读检查确认模型 endpoint 与已授权 algorithm-h20 SSH 主机一致，但 API 是
公网明文 HTTP。没有拆分命令、替换 endpoint 或绕过拒绝。

用户随后对具体目的地、明文风险、冻结契约/内置参考、合成 workload 声明、首轮
完整反馈、可信模板源码，以及最多两调用/GPU 2/owned scratch/无重试，明确回复
**“批准本次有界验收”**。复核同一冻结计划后才执行，授权记录保存在
`output/workload_llm_search_validation/explicit-approval.json`；真实地址/key 不复制进
此文档、冻结源码或 Git。授权前 `execution-status.json` 是历史快照，非最终状态。

### 4.2 真实结果

- run root：`output/workload_llm_search_h20/workload-llm-search-s60diq5p`。
- 原始 artifact：`artifacts/layernorm/nvidia_gpu/20261010_103845_4b603b41/`。
- plan hash 同上；生产 protocol hash 未变，完整 evaluator sources 及只读源码快照
  在调用前冻结。新增驱动仍未 commit/push；不以旧 HEAD 假称包含未提交的实现。
- 真实模型两次请求，无预检请求、fallback、重试、输出修复或 Fixer 调用；requested
  与 reported model 都是 Qwen3.8-27B，finish_reason=stop，reasoning_tokens=0。

| 提案 | 来源 | API 输入/输出 tokens | 模型调用墙钟 |
| --- | --- | --- | --- |
| num_warps=8 | InitialProposal | 3894 / 216 | 5.390 秒 |
| num_warps=16 | SearchProposal，读取首轮完整合格反馈 | 4882 / 199 | 4.591 秒 |

总输入 **8776**、输出 **415**、合计 **9191 tokens**，usage 均由 API 返回，不是字符
估算。输出预留仍为 1024；每调用上限 512。仓库价格表估计约 $0.003048，**不是供应商
账单或实际收费证明**。实际请求字节、响应字节和调用墙钟都在冻结边界内。

| 模型候选 | 8192×15360 | 8192×12288 | 等权几何平均 | 资格 |
| --- | --- | --- | --- | --- |
| warps-8，第 1 轮 | 1.7347× | 1.9576× | 1.8427828738× | 两 case valid |
| warps-16，第 2 轮 | 1.6984× | 1.8064× | 1.7515678006× | 两 case valid |

两轮 L1、两组主输入 L2、v2 32-case 正确性 suite 均通过。第二轮整体约 **−4.950%**，
判决 SLOW；不能因为这是新提案就覆盖赢家。首轮只建 baseline、reward 为空；bandit
两次 pull、一次计分。系统保留首轮自身 source/sketch/完整证据，selected_round=1；
source hash：`e73d43489e0a2a2a17ace0d37383e1452e1c0dbaa3ede8cefe783713c72f1179`。

这是一次真实而**未带来性能提升**的参数搜索，不能宣称模型找到更优配置、全 shape
最优或自由代码生成能力已验收。两个候选都低于冻结的 10× 门槛，整体 L3 未通过；
decision=inconclusive，无 confirmation/admission，隔离 corpus 为空，没有 use_custom。

实际 **284 RPC / 18 commands**；Oracle 4/5（calibration=2、explore=2、confirm=0），
确认格保留。按既有策略 calibration phase 计划格为 1，over_plan=1 如实记录，但未
突破总预算或挪用确认格。墙钟检查约 398.1 秒，低于 1800 秒；第 3 轮在新候选生成前
以 budget_exhausted 停止。结束时的 Search 交接不再调用模型、不生成第三候选。

四个 case 窗和两个 native heartbeat 窗均互异，与上次最终协议试验窗口无交集。
两个 absolute-anchor heartbeat 都为 OK。GPU 仍为 shared-device，不升级成独占或
无干扰保证。owned shutdown ACK、PID absent、quiescent=true、scratch removed=true；
signals=[]，forbidden_actions=[]，没有终止共享进程或修改共享 runtime。

### 4.3 原始磁盘与独立复算

原始 `summary.completed=true/errors=[]`，model/durable auditors 同样通过。模型
request→原始 response→参数→source→首轮反馈的血缘一致，第二次请求确实带有
本次首轮完整合格目标及两个 case 证据；不是以固定提案桩冒充真实 Search。

普通 ArtifactStore 重载保留首轮赢家的代码、sketch、完整 verification/benchmark 和
usage。完整 checkpoint seal/实际输入/controls/协议/best 经生产 reader 校验；
current_code 仍是第二轮探针，报告选优与逐轮链一致。没有补字段、改写旧 checkpoint、
save 或 resume 来制造通过。

之后从已落盘的 model-call、proposal、rounds/trace 重新构造账本和审计输入，独立
复算 model/raw-workload/durable proofs，均与原 summary 完全一致。只读复算的新增
API/GPU 次数为 0；普通重载不恢复历史资格为 live state。所有本次原始文件前后
hash 相等；两个旧试验各 18 项文件也保持原值。

真机清理和独立只读复算结束后再次串行 Golden，仍为 **4 PASS / 0 REVIEW / 0 FAIL /
0 SKIP**；默认容差与 registry hash 未改。此轮没有生产源码变更，不以改动协议后的
离线重标或额外模型/GPU 试验来解释旧数据。

日志与冻结资料：`output/workload_llm_search_validation/` 中的
`targeted-final-source.log`、`full-tests-frozen-source.log`、`golden-final.log`、
`before-execution.json`、`expected-plan.json`、`execution-status.json`（授权前历史状态）、
`explicit-approval.json`、`h20-authorized.log`、`authorized-actual-audit.json`、
`offline-authorized-audit-final.log`、`golden-after-authorized.log`、`final-summary.json`
（最终状态）。

## 5. 结论与下一步

**真实受限 LLM 参数提案→全 workload 真机反馈→正确保留赢家→原始产物重载→有界
退出与清理**的验收闭环已完成，不替代性能达标、长期入库或通用自主开发证明。
下一步可单独设计有界的自由代码候选生成/修复阶段；须重新冻结提案权限与调用预算，
不能沿用本次两次参数请求的授权无限扩展为源码编辑或额外模型/GPU 调用。

后续 [有界单候选代码生成接线](bounded-code-generation.md) 已完成本地实现与守卫。
它有新的生产源码/协议身份，尚未做新的真实模型/GPU 验收；本文旧运行的冻结源码、
原始数据和验收范围保持不变。
