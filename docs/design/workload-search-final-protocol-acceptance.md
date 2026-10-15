# Workload 搜索最终协议复验

日期：2026-10-10。接续 [搜索闭环设计](workload-objective-search.md) 的最终协议缺口。

## 1. 范围与冻结

上一阶段源码已本地提交为 `d33a8a4`，此前本地全量 3584 passed，Golden 4 PASS。
生产 protocol hash 为
`f14423ca3d71dd7a6fdb29500c8f25d5c05c76277ad12daef69f7c788a2290a6`。
本次**不改生产实现**，仅增强 H20 验收驱动和本地守卫，再对这个最终协议采集一次
新证据。驱动及守卫在执行前本地提交为 `fd78902`，完整 evaluator sources 与它自身
hash 进入新计划；执行时 Git 工作区干净。未 push。

沿用原候选、workload、参考、标尺和预算：

- algorithm-h20，物理 GPU 2，既有 runtime，只有本次 owned scratch/runner。
- FP16 LayerNorm，unfused FP32 语义；两个候选只差 num_warps=4/8。
- 8192×15360 / 8192×12288；等权几何平均，每 case 5% 回退上限，min_speedup=10×。
- 原 ruler profile：`97dec1cd4d762682d713271146ac1fcd76b4c3b813842247083edd61d9348899`。
  不重标 launch floor、不放宽资格，也不根据新读数调整门槛。
- max_rounds=3、oracle_total=5、确认格保留、wall_clock=1800 秒，512 RPC/commands。
  LLM token 预算 128 为合法占位，本次提案仍是固定桩，实际无 LLM 调用。
- 每候选只执行一次完整 L1/L2/L3；新 case/heartbeat 窗口，不复用旧测量证据，不挑窗。
- 无安装、共享 runtime 编辑、共享进程终止；清理须确认 shutdown ack、owned PID
  quiescence、scratch removal。失败/降级保留原证据，不自动追加试验求通过。

这不是换门槛后的性能重跑，也不在多次试验中挑较大的加速比。旧协议与最终协议的
原始试验都保留；本次结论只来自本次冻结计划。

## 2. 磁盘证据是验收条件

`benchmarks/workload_search_h20.py:audit_durable` 是只读检查：

1. 从普通 ArtifactStore 重载真实目录；不补 metadata，不 save，不 resume，不派 RPC。
2. result 的 `search_objective`/`candidate_selection` 与 sealed checkpoint、原候选数据
   一致，独立复算 score 和稳定同分选优；不是只信内存返回对象。
3. 赢家 source/sketch/完整 L1/L2/L3 与该候选原始结果一致；仅允许已知的派生发布
   注释有差异，不能拼接末轮的 metrics。benchmark 也必须属于同一赢家。
4. 通过生产 `load_checkpoint` 验证完整 manifest、实际输入、controls、协议、seal 与
   best 载荷；只反序列化供审计，不把历史凭据恢复为实时资格。current_code 仍是末探针。
5. 原报告的选优段与逐轮证据链必须和重载产物的渲染一致。报告生成失败也不能算验收。
6. budget stop、inconclusive、未确认/未 admission 必须同时成立；受检文件前后 hash
   相等。审计通过的 file hashes 与 protocol hash 随 summary 落盘。

local-plan-only 不触碰 transport；预算不合法时在 SSH/输入分配之前拒绝。最终
summary.completed 必须同时满足主证据、磁盘审计和 owned cleanup，缺一不可。
固定源码与顺序还在 Verifier 编译/派发前检查，拒绝重复、第三候选、重排序和临时
Fixer 替换；不是等跑完多余窗口以后才在审计中发现越界。

## 3. 本地守卫（不是 GPU 性能）

确定性 fake timing 经完整生产保存→重载，对“较早候选赢”和“末候选赢”都验真。
负向测试覆盖选优字段缺失、轮次/score/source/sketch 变更、末轮验证/benchmark
替换、报告不一致、坏 seal、resealed best、协议漂移、symlink 与伪确认/admission。
失败不得修补文件或派发新测量。

```sh
python -m pytest tests/test_workload_search_driver.py -q
python -m pytest tests -q
python tests/golden_regression.py compare
```

- 新驱动守卫 36 项；连同搜索/重载专项，**89 passed**。
- 最终冻结源码的全量：**3609 passed / 1 skipped / 2 warnings**，约 179.6 秒。
  两个 warning 来自已有 nonfinite correctness fixture，不是新的功能失败。
- 全量和真机工作结束后串行 Golden compare：**PASS 4 / REVIEW 0 / FAIL 0 / SKIP 0**。
  registry SHA256 前后均为
  `38836e092efd7ca9773ce414c39271ab0f050e29318b27d3de9d57e417d60819`，默认容差未改。
- 日志：`output/workload_search_final_validation/targeted-final.log`、
  `full-tests-final.log`、`golden-final.log`。

## 4. 最终记录

### 4.1 本次新协议试验

- run root：`output/workload_search_final_h20/workload-search-lcw_onp9`。
- source commit：`fd78902d329aa8c55fad5590e9227c2e02241abf`。
- plan hash：`f70d2df8fc863fbe7fc392552db333f05f2f37294027a98fbbda8d77507b85dd`。
- protocol hash：`f14423ca3d71dd7a6fdb29500c8f25d5c05c76277ad12daef69f7c788a2290a6`。
- contract hash：`407e4e52bdafe1bedc5f9e43f15f2df3b888e1bb8bf8676057a1211324e2d5bf`。
- 本次计划与执行前只读 expected-plan 完全一致。与上次计划比，候选、case、参考、
  profile、门槛、权重和预算都相同，只有协议/源码清单变化。

| 候选 | 8192×15360 | 8192×12288 | 等权几何平均 | 资格 |
| --- | --- | --- | --- | --- |
| warps-4，第 1 轮 | 2.0185× | 1.9543× | 1.9861406169× | 两 case valid |
| warps-8，第 2 轮 | 1.7346× | 1.9573× | 1.8425885542× | 两 case valid |

两轮 L1、两组主测 L2、v2 32-case 正确性套件均通过。性能凭据 valid，但都低于
冻结的 **10×** 门槛，因此两轮整体 L3 均未通过。这是预期的有界搜索接线验收，
不是部署性能达标，也不宣称绝对峰值/全 shape 最优。

首轮仅建立 baseline，reward/进度比较均为空；第二轮 reward=
`-0.07227688788692398`（约 −7.2277%），判决 SLOW。正确保留第一轮代码、sketch
及该候选完整证据；第二个 case 虽略快也不能覆盖较差的整体目标。bandit 两次 pull，
一次计分。final artifact 的 candidate hash 为
`5cce7c80eafd1ecf534082be24a09b58ab02eb5cd7c6423883589ca2580b4db1`。

4 个 case 配对窗、2 个 native heartbeat 窗均互异，与上次 6 个窗口的交集为空。
两次 absolute-anchor heartbeat 均 OK；参考 receipt、输入保真、实际 RPC 与 scope
指标通过核对。测量仍为 shared-device（7 个其他进程、观测瞬时 util=0%），不升级成
exclusive-device 或无干扰保证。没有临时追加候选、补测、修改标尺或挑更好的窗口。

### 4.2 预算与退出

实际 284 RPC / 18 commands，Oracle 花费 4/5（calibration=2、explore=2、confirm=0），
确认格保留，LLM tokens=0。按原策略 heartbeat 的 calibration phase 计划格为 1，
记录 over_plan=1，但不超过总预算或挪用确认格。墙钟检查约 254.7 秒，1800 秒上限
未放宽。第 3 轮之前拒绝新探索，stop=`budget_exhausted`。

decision=`inconclusive`，隔离 corpus 为空；无 confirmation/admission 文件，没有
use_custom 或新入库。H20 验收通过指的是“真实测量→正确选优→持久化→预算退出”
接线成立，**不等于该候选已满足业务部署门槛**。

### 4.3 原始磁盘产物与清理

最终原始产物目录：

```text
output/workload_search_final_h20/workload-search-lcw_onp9/artifacts/
  layernorm/nvidia_gpu/20261010_095442_88f19d43/
```

`summary.completed=true/errors=[]`，`durable.completed=true/errors=[]`。本次不再
使用上一阶段的“离线补字段投影”：原始 result 就有选优身份，普通重载的 source、
sketch、score 与原候选证据一致；完整 checkpoint seal、实际输入/controls/最终协议
经过生产 reader 校验。checkpoint 仍记录第 2 轮探针，没有把末轮覆盖成赢家。
原 report 的选优段与逐轮表和重载后的渲染一致。

只读磁盘审计记录 9 个受检文件的 hash，前后完全相同。owned runner shutdown ack、
owned PID absent、quiescent=true、scratch removed=true；无 signal，forbidden_actions
为空，没有终止共享进程或修改共享 runtime。

之后又从已落盘的 rounds/trace 和普通 ArtifactStore 重载结果独立复算，并重建
确定性主输入验证完整 checkpoint。主 proof 与 durable proof 均与原 summary 完全
相同，未 save/resume/派 RPC/新测 GPU。旧试验所有 18 个文件 hash 与本次原始文件
hash 均保持不变。budget/bandit 摘要作为显式磁盘审计输入，不恢复为 live state。

独立结果见 `output/workload_search_final_validation/actual-final-audit.json`，冻结控制
和完整 81 项源码快照见该目录的 `before-execution.json`、`expected-plan.json`、
`frozen-sources/`；真机日志为 `h20-final.log`。

## 5. 结论与下一步

**最终协议的固定两候选 H20 + 原始产物重载验收缺口已关闭**。生产代码仍是已提交
的同一协议，没有把旧测量重新挂到新协议、改写旧 checkpoint 或复用旧资格。

本次 generation/Search proposal 仍为固定桩；真实 LLM 搜索尚未验收。下一阶段应在
这套证据/预算/清理门上接入真实提案，使用新的冻结任务和有界预算，单独报告模型
调用成本、失败修复、实际候选数与合格 workload 证据，不能用本次结果代替自主性证明。

下一阶段的第一小步见 [有界真实 LLM 提案验收](workload-llm-search-acceptance.md)：
先验受限 launch 参数提案，不把可信模板 materialization 冒充自由源码生成。
