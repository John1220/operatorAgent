# 冻结 workload 目标的搜索闭环

日期：2026-10-10。基于上一轮已提交的 workload 决策与确认（`3d07f1a`）。

## 1. 为什么继续改搜索

上一轮已让评测、确认、建议、持久化按声明的 workload objective 判决，但搜索仍有
三个口径断点：最佳值/进步主要读主 case GFLOPS，bandit 回报没有统一到目标，预算或
末轮失败后默认返回最后一版。另有主 case 的 legacy 停滞位不能代表整个 scope。

本轮把**显式目标任务**的搜索判断统一起来，不把旧 primary-GFLOPS 任务偷偷升级为
多形状结论，也不宣称实现业务流量分布优化、全 shape 最优或真实 LLM 自主搜索。

## 2. 一个冻结目标，一个候选凭据

新增 `operator_agent/search_objective.py`。身份包含 schema、basis、contract_hash、
aggregate、case_weights、required_cases、max_regression、min_speedup。

每次读数必须满足：

1. 完整 workload contract 与冻结契约一致，receipt 声明的判据没有改变；
2. scope 的资格由同一个 `workload.qualification` 复算，顶层 mask/档位与之相符；
3. 每行原始数值、档位、掩码、契约身份与 Oracle observation 一致；
4. 用既有 `aggregate_speedup`、`regression_over_limit`、`performance_verdict` 复算；
5. 真实 case 两臂同窗、case 窗口互异，当前会话已见窗口不能再次计入搜索；
6. 源码与 L1/L2/L3 verification binding 一致，正确性已由 L1/L2 确立。

**未达 min_speedup 不等于不可搜索**：1.6× 未达 10× 目标，但仍可作为有资格的
优化基线。相反，缺失 coverage、bad ruler、missing/cross/reused window、回退越限或
声明漂移不能凭主 case 更快成为赢家。原始不可用数保留为诊断，score=None，不补 0。

## 3. 资格与后果

| 后果 | 需要的条件 |
| --- | --- |
| 更新 Search 的软信念 | score 可用、hard constraints 成立、整体 can_tune=true |
| best 候选、bandit 有效回报、进度/停滞 | 整体 can_count_progress=true，正确性与身份成立 |
| 独立确认/长期入库/use_custom | 继续使用既有 workload 确认与 admission；搜索赢家不豁免 |

degraded 可以给软信念提供方向，但不能覆盖 best 或增加有效停滞。没有本次可用信念
时不给 Search 编造性能数字；已有信念时沿用上次合格读数。

每个 L3 试验只记一次 bandit pull，包括最后一轮和直接通过的候选。第一个合格样本
是 baseline，reward=None，而不是“提升 0”。后续回报是：

```text
(本轮 workload score - 本轮之前 best score) / 本轮之前 best score
```

回报在更新 best **之前**计算，避免把新赢家与自己比较而得到 0。没有资格的试验只
记尝试与真实成本，不记分。bandit 另外拒绝 NaN/Inf reward 和 cost。

候选保留按有效 score 最大值、同分保留先出现者；进度显著性仍调用既有 D4
`measurement.verdict`。因此“保留稍高读数的候选”不等于“宣布显著 FAST”：SAME
仍计为无显著改善，只有 SLOW 才禁用未成功应用的配方。

主 case GFLOPS、带宽与 roofline 仍是**提议方向的诊断/启发式**，不是整体排名。
Search prompt 标注作用范围，并附冻结目标与逐 case 证据。整体 roofline 退出必须
所有 scope case 都有可靠的参考 ceiling 证据，不能拿主 case 到顶提前终止。

未声明目标的旧任务继续走原 GFLOPS 搜索分支；没有给旧记录伪造 objective。

## 4. 预算结束与最后一轮失败

保存 best 的源码、源码 hash、sketch、round、score 和**该候选自己的**完整分级结果。
返回产物的源码、编译信息与 benchmark 必须来自同一候选，不能拼接末轮结果。

预算耗尽、末轮更差/不合格甚至 L1/L2 失败时，可以返回本会话较早的最佳合格候选。
如果该候选未达目标，L3 仍失败，decision 仍 inconclusive，不补确认、不写长期语料。
“最优已观测候选”不意味着满足部署条件，更不是全搜索空间最优。

最后一轮 degraded 虽过 L3、但选优资格不足时，不能借它替换较早 valid 候选；如果
保留的合格候选未达门槛，最终 reason=`qualified_candidate_hold`，同时保留
`last_probe_stop`，不让 `passed` 的探针出口伪装成最终成功。

## 5. 快照、恢复与冻结身份

快照新增 `search_objective`，保存目标身份与 best 候选的完整审计凭据；其内容进入
checkpoint digest。原 `current_code` / pending_fix / verification_results 继续记录最后
探针/下一步状态，`best_gflops` 不承载 workload 加速比。

恢复在任何 provisioning、RPC、generation 之前验证目标、score、正确性和候选绑定。
best 的验证载荷必须是完整、唯一的 L1/L2/L3，passed 为严格布尔、metrics 为映射；
null/list metrics、整数 passed、重复 level 或非列表记录按身份 HOLD，而非在派发后
才抛异常，也不会读成通过。
可以把历史 best **源码**作为新提案，但不恢复 score、eligibility、stagnation、bandit、
已见 live windows 或旧验证结果。本会话新建空 SearchObjective，强制 fresh L1/L2/L3。
历史数据仅在 `resume_audit` 中供审计，不能恢复为实时进度或入库资格。

普通产物也显式登记 `search_objective`、`candidate_selection`、`last_probe_stop` 三个
顶层键；不整包输出 metadata，不给 legacy 补目标。`ArtifactStore.load` 回读这些审计
字段、最终停因与逐轮血缘；sketch 使用 schema 的 `from_dict`，避免把序列化的
`global/local/register`、`from/to` 当作 dataclass 参数导致重载异常。**普通 load 不
等于 resume，不授予任何实时资格。** 报告标明保留轮次、目标分数、较早候选回退、
末次探针与最终停因，并把主 case/GFLOPS 标为诊断；逐轮表增加 score/reward/判决。

`search_objective.py`、bandit 与 selection 实现纳入完整 protocol identity，因此不同
协议的旧 checkpoint 不会被新语义自动接纳。旧语料也没有追溯性换分。

## 6. 验收

### 6.1 本地确定性用例

非真实 GPU 性能：主 case/GFLOPS 排名与 workload 排名反转；硬回退、次 case 地板/
噪声/缺失/漂移/复用窗口；degraded 软信念但无永久后果；预算耗尽保留最佳候选而非
末轮；同门槛独立确认与 sealed 入库；恢复必须新测；resealed 但 criterion/score/source/
Oracle mask/binding 变更的 checkpoint 在派发之前拒绝；单 case ceiling 不终止整体。

```sh
python -m pytest tests/test_search_objective.py tests/test_workload_search_driver.py \
  tests/test_search_artifact_persistence.py -q
python -m pytest tests -q
python tests/golden_regression.py compare
```

### 6.2 H20 有界两候选接线验收

驱动 `benchmarks/workload_search_h20.py` 默认只写本地计划，不分配大输入、不 SSH。
显式执行时使用 algorithm-h20 GPU 2、既有隔离 runtime 与 owned scratch，固定：

- FP16 LayerNorm、unfused FP32 中间语义、同一 v2 reference；仅 num_warps=4/8 不同。
- 两个大 case：8192×15360 / 8192×12288，输入各自低于原 256 MiB 存储上限。
- 等权几何平均、每 case 5% 硬回退上限；**min_speedup=10×** 只用于让固定候选
  进入搜索流程，未达就未达，不调整到观察值附近。
- 只生成两候选、各一轮完整 L1/L2 suite/L3；每 case 一个新配对窗口，无 retry。
- 最多 3 轮、oracle_total=5（预留确认格不挪用）、1800 秒、512 RPC/512 commands；
  LLM token 预算固定 128，但生成和下一候选提案为桩，实际不调用 LLM。
- 复算两候选目标与奖励、核对实际 RPC/输入/参考/heartbeat/window、赢家源码与结果，
  必须 budget stop、inconclusive、未确认且隔离 corpus 为空。

```sh
python benchmarks/workload_search_h20.py  # local plan only
python benchmarks/workload_search_h20.py --execute --host algorithm-h20 --gpu 2 \
  --cuda-ruler-profile /absolute/path/to/ruler-profile.json
```

首次远端尝试 `workload-search-lwso9j9l` 的驱动误把 llm_tokens 写为 0，触发已有预算
正整数校验，未生成、编译、验证或测量候选（trace 只有两个 gpu_info）。失败计划与
日志保留；owned runner/scratch 清理已确认。修正仅驱动预算为正整数，并把预算构造
前移到任何 SSH 之前，再冻结新计划；候选与生产测量/搜索策略未改。

### 6.3 真机记录与验收范围

本轮唯一进入候选测量的计划：

- run root：`output/workload_search_h20/workload-search-918lhb1j`。
- plan hash：`84bf7cfea83f9645a122ed3bab01e115a9d5675e9d2d224ef2fcde709db32f81`。
- contract hash：`407e4e52bdafe1bedc5f9e43f15f2df3b888e1bb8bf8676057a1211324e2d5bf`。
- measured protocol hash：`3f3c0a9cc1992e4e2f924cb9b942f6827d3d8e60bc4d8ba07c06c6a1c48d0af7`。

| 候选 | 8192×15360 | 8192×12288 | 等权几何平均 | 搜索资格 |
| --- | --- | --- | --- | --- |
| warps-4（第 1 轮） | 2.0180× | 1.9538× | 1.9856405516× | 两 case valid |
| warps-8（第 2 轮） | 1.7352× | 1.9549× | 1.8417769898× | 两 case valid |

两轮真实 L1、主测 L2 与 v2 32-case 正确性套件通过，均未达冻结的 **10×** 门槛。
第一轮只建 baseline，reward=None；第二轮 reward≈−0.072452（−7.2452%），进度
判决 SLOW。保留第 1 轮源码、sketch、该候选自己的结果；checkpoint 的 current_code
仍是第 2 轮探针。不是“最后一版即赢家”。bandit 2 次 pull、1 次计分，无重复计分。

两轮共 4 个 case 配对窗、2 个 native heartbeat 窗，6 个窗口互异，单 case 两臂同窗。
参考 receipt、执行输入保真、实际 RPC 与指标、心跳标尺均复核，无改权重/门槛、补测
或挑窗。实际 284 RPC / 18 commands；Oracle 花费 4/5，confirm=0，确认格保留。
calibration 的 phase 计划格为 1，按既有心跳策略记了 2 次（over_plan=1），这是原账本
明确披露的 phase 超额，不是额外试验或挪用确认预算。墙钟检查约 318.7 秒。

第 3 轮开始前按预算拒绝新的探索，stop=`budget_exhausted`，decision=`inconclusive`，
corpus 为空、无 confirmation 文件、无新 admission。虽然数字约 1.986×，**不是上一轮
已入库记录的复用，也不是本轮部署达标**。GPU 仍为 shared-device 证据，不宣称独占。
generation/Search proposal 为固定桩，因此也不算真实 LLM 自主搜索验收。

`summary.json` completed=true/errors=[]；owned runner shutdown ack、PID quiescence
与 scratch removal 均确认，无 signal、共享进程终止或共享 runtime 修改。

### 6.4 真机之后的持久化补丁（没有新 GPU 证据）

真机清理结束后，离线重建发现 `result.json` 未登记选优字段，且旧 sketch loader 会对
MemoryPlan 的 `global` 键抛 TypeError。修复普通保存/重载与报告；**没有为此重测或
修改原始 H20 文件、计划、checkpoint、receipt 或源文件 hash**。
最终检查还补充了 checkpoint best 验证载荷的结构校验；这不改变读数、排名、奖励
或测量判据，但会让损坏记录在恢复前按身份 HOLD。

原实测源码（完整 81 项 evaluator sources）保存于
`output/workload_search_validation/measured-sources/`；原 raw round/trace 可独立复算
选优、奖励、候选绑定、budget stop 与未入库，离线审计见 `actual-audit.json`。

最终 protocol hash：`f14423ca3d71dd7a6fdb29500c8f25d5c05c76277ad12daef69f7c788a2290a6`。
与实测 protocol 不同的登记源文件为 `operator_agent/artifact_store.py` 与
`operator_agent/search_objective.py`（上述恢复载荷结构校验）；报告也有修改，属于额外
evaluator source，但不在 measurement_protocol 文件列表里。
**最终协议尚无新一轮 GPU 验收**，不能把旧协议的实测窗口改挂到最终协议上；按协议
身份检查，原 checkpoint 也不会被最终版本直接续跑。

使用最终 writer/loader 对原实测产物做一次**独立目录的本地审计投影**：选优字段
明确来自原 sealed checkpoint + iteration history，测量 manifest 保留旧协议，所有
原文件 hash 保持不变。投影 report 标注不是新 GPU run，无 resume/确认/入库，详情
见 `output/workload_search_validation/persistence-audit-final.json`。这项验证只证明最终
保存→重载→报告不丢真实候选的身份与证据，不替代最终协议 GPU 测试。

### 6.5 本地最终回归

- 新增搜索、驱动安全与产物重载专项：**64 passed**。
- 最终全量：**3584 passed / 1 skipped / 2 warnings**，约 192.2 秒；两条 warning
  来自已有 nonfinite correctness fixture 的 subtract，不是本轮新增失败。
- 全量结束后串行 Golden compare：**PASS 4 / REVIEW 0 / FAIL 0 / SKIP 0**。
- 默认容差未改，registry 文件 SHA256 前后均为
  `38836e092efd7ca9773ce414c39271ab0f050e29318b27d3de9d57e417d60819`。
- 最终 `git diff --check` 通过；无新 push。本轮改动尚未 commit。

日志：`output/workload_search_validation/targeted-final.log`、`full-tests-complete.log`、
`golden-final.log`。早期失败的夹具/驱动尝试日志保留，但不当成最终通过或新真机读数。
