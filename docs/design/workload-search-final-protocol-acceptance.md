# Workload 搜索最终协议复验

日期：2026-10-10。接续 [搜索闭环设计](workload-objective-search.md) 的最终协议缺口。

## 1. 范围与冻结

上一阶段源码已本地提交为 `d33a8a4`，此前本地全量 3584 passed，Golden 4 PASS。
生产 protocol hash 为
`f14423ca3d71dd7a6fdb29500c8f25d5c05c76277ad12daef69f7c788a2290a6`。
本次**不改生产实现**，仅增强 H20 验收驱动和本地守卫，再对这个最终协议采集一次
新证据。驱动也要在执行前提交，完整 evaluator sources 与它自身 hash 进入新计划。

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

驱动专项与最终全量、Golden、真机结果待本次冻结执行后回填；不更改
registry 或容差，不把前一阶段结果误写成本次新协议 GPU 验收。

## 4. 最终记录

待本次唯一冻结试验结束后填写新 plan/protocol hash、真机档位与分数、磁盘审计、
旧窗口互斥和清理证据。满足以上范围后才关闭最终协议缺口；真实 LLM 搜索另行验收。
