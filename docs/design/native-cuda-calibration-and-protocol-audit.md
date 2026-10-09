# 原生 CUDA 标定心跳与计时协议审计

日期：2026-10-09。接续 [有界 workload 与入库验收](bounded-workload-and-corpus-acceptance.md)。

后续实现与新版本实测见 [版本化 CUDA 量具、独立锚点与真实入库](versioned-cuda-ruler-and-admission.md)。
下文保留本阶段 separation-only 与旧地板 degraded 的历史结论，不追认成新版本结果。

## 1. 问题与不变量

240 MiB 主输入已执行完整正确性 suite、主测和独立确认，但 149 µs 左右的净读数
仍受既有 H20 20.24 µs 地板约束，launch dominance 约 11.9%，为 degraded。
继续放大 shape 已接近 256 MiB 原值输入协议上限，也不是解决计时口径差异的办法。

既有地板来自不冲刷 L2 的空核测量；生产 L3 的 `_timed_runs` 每样本在 event
窗口之外冲刷 256 MiB L2 buffer。这两条通道不能只因同样叫“空核耗时”就混用。
另外，旧心跳是远端 CPU/Python matmul + perf_counter，不能认证 CUDA event 分支。

本轮**不修改**地板、10%/30% validity 分档、数值容差、参考语义、1.05× 门槛、
完整 suite、独立确认或 corpus admission。不把新审计值自动写回 hardware_caps。

## 2. 单一生产计时入口

新增共享源 `operator_agent/cuda_calibration.py`，部署为 runner 同级模块，先于服务端。
常驻服务端专属 `cuda_calibration_probe` action，不扩展 CUDA/XPU kernel action 表。

- 只允许预定义 `audit` / `heartbeat` 模式与协议 id/action/mode 字段。
  任意源码、输入、路径、shape、repeat 或其他配置在 GPU 工作前拒绝。
- 固定原值 fp32 2048×1024 输入，重复 `[-1,0,1,2]`，8 MiB。
  NumPy storage authority 与实际 Torch storage/view/值/设备回读逐项匹配。
- 固定源码：空 Triton kernel；单次完整 copy 好核；连续八次真实 copy launch 慢核。
  两个输出先与完整输入比较。慢核只用于审计尺子，不掺进任务候选或 workload。
- 好/慢两臂直接调用生产 `_bench_pair_interleaved(..., 10, 30, True)`：
  候选 shield、预算估时、最多 5000 样本、每块最多 10 次、每样本 L2 冲刷，均复用。
- audit 的空核与好核直接调用生产 `_timed_runs`，分别 flush / no-flush，各 200 样本；
  每个核先固定 warmup 25 次，所有原始样本保留。
- 另记 200 个 host dispatch + device execution + synchronize 墙钟样本，不冲刷 L2。
  它是**另一轮观测**，不是纯 launch cost，不与 event 样本相减，也不作为地板。

event 时间也不能简单称为“纯设备指令时间”：无 flush、GPU 空闲时，host 派发间隔
可能落入 event 窗口；冲刷产生的同流排队改变这一关系。审计区分的是**计时通道口径**，
不是声称 kernel 本身由于冲刷而变快。host/event 独立样本也不保证可相加或大小单调。

## 3. 身份冻结与失败关闭

标定身份 SHA256 覆盖：

1. 固定参照源码、真实输入 descriptor/字节摘要、全部固定计时配置；
2. 物理 GPU 号、CUDA 可见集/index/count/name、默认 stream；
3. GPU UUID、PCI bus id、驱动、Torch/CUDA/Triton 版本；
4. 实际部署的 run_kernel、共享标定模块、anticheat、两份 fidelity 模块全文摘要。

每次回执还必须证明：两臂 sample count 有界且相同；每臂/结束复读设备相同；
实际 timing window 相同；读数有限且为正；full input/readback 和两个正确输出齐全。
分离度同时看 ratio-of-medians 与 paired median，取较保守者，要求 ≥2。

离线控制端另核本地预期源码/固定输入，不能只相信远端自报 identity_hash。
预声明**两次** audit，不按读数追加尝试或只保留通过窗口；两次必须相同身份、不同实际窗口。
成功并确认 owned runner 清理后，atomic/fsync 写入 readonly `calibration.json`。
读取时重新验证两份实际回执及完整冻结摘要，限制文件 2 MiB。

Conductor 新的可选 `cuda_calibration_identity_hash` **只收紧**判定，并进入 frozen run controls：
后续心跳身份不匹配即 RULER_BROKEN，不能换尺子继续测；resume 改这个值也不继承旧结果。
标定模块本身已加入 production measurement protocol 源码摘要。

观察到 CUDA 环境时，到期心跳必须走 `NATIVE_CUDA_CALIBRATION` 能力。
目前由 NVIDIA persistent SSH 提供；one-shot/XPU/Local/HTTP 不虚报这个能力。
能力缺失、超时、身份未知、读数无效或快慢不可分，不静默退回 CPU 心跳。
CPU/离线既有心跳仍保留。原有心跳 cadence 默认 5 不变。
Oracle 注册 channel、identity hash 和实际 evidence，不能被候选 raw provenance 改写。

### 诚实的边界：不是绝对锚定

本轮 CUDA 心跳是 **separation-only**。`known_good_ratio=1` 是“未做绝对检查”的占位值，
不是用刚量出的好核自除得到的证明；回执明确 `absolute_anchor_checked=false`。
整条计时通道若同时缩放 10×，快慢比仍不变，当前心跳抓不住。对此有明确负向边界测试。
冻结身份能阻止换设备/版本/源码冒充同一标定，不能替代独立绝对时间锚点。

## 4. 实际 H20 审计

运行目录：`output/cuda_calibration_h20/cuda-calibration-h20-20261009T040702027205Z-87286-1`。
固定 GPU 2，私有 owned runner，现有 runtime；没有安装/sync 或共享环境改动。

| 通道，中位 µs | audit 1 | audit 2 |
|---|---:|---:|
| 空核，CUDA event，无 L2 flush | 18.672 | 18.688 |
| 空核，CUDA event，L2 flush | 4.512 | 4.480 |
| 空核，host end-to-end，无 flush | 16.797 | 16.571 |
| 好核，CUDA event，无 flush | 30.928 | 29.312 |
| 好核，CUDA event，L2 flush | 10.400 | 10.400 |
| 好核，host end-to-end，无 flush | 23.671 | 23.552 |
| 慢/好两臂保守分离度 | 7.4844× | 7.1109× |

每个 audit 六个 channel 各 200 个原始样本，共 1200 个；另有生产预算决定的好/慢 paired 统计。
两次窗口分别 `cb297c6be65f` / `17ce0be8b746`，非同一回执复用。

- identity：`f6164643fddaa8c480b81a29a11b8e62d2154842407f2193658ab9c8a48ee20c`。
- 冻结完整 audit receipt：`ad32c861ad10df3548d6098cb36999f7525eefb81ea60cf99d53659e29235d3b`。
- Runtime：Torch `2.14.0+cu130` / CUDA `13.0` / Triton `3.8.0` / driver `580.105.08`。
- GPU UUID：`GPU-13c6c4b4-45ce-8963-1122-df2b3d72fb76`，PCI `00000000:A2:00.0`。
- precheck 是启动前 idle/free-memory 检查；audit 的 before/after telemetry 不是持续无干扰证明，
  且采样前已跑过 paired probes，util 可能来自本探针。SM 1980 MHz、memory 2619 MHz、throttle=0
  只是这两次快照观测，不声称锁频。
- shutdown ACK、remote quiescent、私有 scratch removed 均确认；signals=[]、forbidden_actions=[]。

结论：历史约 20 µs 无 flush 口径与当前约 4.5 µs flush event 口径确实不同。
这份证据**支持需要 versioned channel-specific 地板**，但不是自动调整 production ruler 的授权。

## 5. 冻结身份后的同 workload 验收

`durable_profile_h20.py --cuda-calibration <calibration.json>` 在任何 backend/输入分配前核验冻结回执，
要求显式相同物理 GPU，并把 receipt/identity hash 与 heartbeat_interval=1 写入预声明计划。
原 240/192 MiB workload、值模式、候选、参考与所有数值/admission 门槛保持不变。
主测、独立确认与合法 resume 都必须实际跑原生心跳并匹配冻结身份。
输入/threshold drift HOLD 仍要求零 RPC、零生成、零 corpus 变化。

实际运行目录：`output/durable_profile_h20/durable-profile-h20-20261009T040856053546Z-89383-1`。
预声明计划 hash：`9fddb0ee081725a4507717756d792748c5785f688b4055961fc502687da32c01`。

| 主输入 / 过程 | 主测净耗时 µs | 主测比值 | 独立确认比值 | 地板占比 | 主测 / 确认 |
|---|---:|---:|---:|---:|---|
| 240 MiB，8192×15360 | 149.152 | 2.0185× | 2.0185× | 11.95% | degraded / degraded |
| 192 MiB，8192×12288 | 116.928 | 1.9579× | 1.9581× | 14.76% | degraded / degraded |
| 合法 resume，240 MiB | 149.056 | 2.0198× | 2.0187× | 11.96% | degraded / degraded |

比值保留为这几个窗口的描述性观测，**不是可信持久化性能结论**。
完整 32-case suite 与重复确定性检查全部通过；原始 CUDA 参考 receipt 全部复核。
三个过程各有两个独立 native 心跳，总计六个不同的实际 timing window；
保守分离度依次为 7.3037/7.3526、7.4645/7.4057、7.5527/7.4181×。
各次主测与确认的 Oracle metadata 均实际登记 native channel、冻结身份和 GPU evidence。
三个 durable confirmation 文件另行复核：相同身份/候选、不同 verification/window，未借旧确认。
生产 manifest 中的完整 evaluator 源码摘要与当前源码一致。

- 合法 resume：没有 Designer/Coder 调用，重新 compile、完整 L2、native 心跳、主测及确认。
- 输入值/阈值漂移：HOLD；零 RPC、零生成、零 corpus 变化。
- 功能验收 `functional_completed=true`，5/5；性能/真实入库验收仍 `completed=false`。
- `positive_admissions=[]`；未写正例或失败 corpus。未实际走到 positive corpus append/replay，
  不能声称入库闭环已完成。
- owned runner shutdown ACK、remote quiescent、scratch removed；signals=[]、forbidden_actions=[]。

两次审计和同 workload 重跑都保留全部证据，不通过时没有自适应追加尝试。

### 回归与守卫标定

验证目录：`output/cuda_calibration_validation`。

- 新增 54 个离线/实际 CPU 服务端测试；这些不当作 CUDA 真机证据。
- 全量：3265 passed / 1 skipped；两条既有 nonfinite-reference RuntimeWarning 保留。
- Golden compare：4/4 PASS，registry 未刷新。
- 关键守卫 mutation：clean pass，9/9 **断言杀死**，不是 import/collection failure。
  覆盖身份字段、input readback、逐臂设备、有限计时、native 派发、冻结身份漂移、
  好/慢分离度、独立 audit window、identity hash 校验。所有突变在隔离副本执行。
- 单独生成 `evidence-audit.json`，复核六个 native 窗口、当前 production protocol、
  三个 confirmation、所有数值 receipt、旧地板未变和零 corpus 写入。

可复现命令：

```bash
python benchmarks/cuda_calibration_h20.py --gpu 2
python benchmarks/durable_profile_h20.py --gpu 2 \
  --workload 8192x15360 --workload 8192x12288 \
  --cuda-calibration /absolute/path/to/calibration.json
python -m pytest tests -q
python tests/golden_regression.py compare
```

审计回执不是跨版本通行证；源码、GPU 或运行时改变后必须重新独立标定并冻结。

## 6. 下一步条件

1. 先独立定义并冻结 channel-specific floor 的统计口径、设备/运行时键与更新审批流程。
2. 引入独立、不可由当次心跳自归一化的 CUDA 绝对锚点及容忍带，覆盖整体缩放负控。
3. 新标定版本启用后重新冻结任务、重跑同 workload/full suite/主测/独立确认；
   只有真实 valid 才尝试 admission，并验证实际 corpus append + fresh replay。

不能把本轮旧地板下的 degraded 回执追认成新版本 valid，或复用旧确认直接入库。
