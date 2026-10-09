# 版本化 CUDA 量具、独立绝对锚点与真实入库

日期：2026-10-09。接续 [原生 CUDA 标定与协议审计](native-cuda-calibration-and-protocol-audit.md)。

## 1. 本轮解决什么

历史 H20 地板 20.24 µs 与生产 L3 的 L2-flush event 通道口径不同。
仅证明好核/慢核能够区分，不足以抓住整个 event 通道同时快或慢 10×。
本轮新增显式版本 `cuda-flush-ruler-profile-v1`，先冻结独立预检，再测任务。

**不变项**：默认 hardware_caps 地板仍 20.24 µs；launch dominance 的 10%/30%
分档、精度容差、参考身份、1.05× 性能门、完整正确性 suite、独立确认、admission 均不改。
旧 degraded 回执保持原结论，不能换一个数字追认为 valid 或直接继承旧确认入库。
没有自动热更新、从候选读数推导地板、扩大输入协议上限或人为拖慢候选。

## 2. 预声明 policy 与独立证据

`operator_agent/cuda_ruler_profile.py` 是唯一 profile 派生与校验权威。
不接受手填 floor/anchor，也不接受“改数值后重新算 SHA 就可信”。读取时从完整两次
原生 audit 回执重新派生 profile，并比较全部内容、policy 和 hash。

| 项目 | 固定规则 |
|---|---|
| 预检 | 恰好两次固定 audit、相同完整身份、不同实际 paired window；全保留 |
| 地板 | 两次空核 flush 各 200 样本，分别取 nearest-rank P95，再取较大值 |
| 好核锚点 | 两次生产 paired 好核 median 的中位数；不是另一个单臂计时实现 |
| 预检质量 | 空核与好核 IQR/median ≤10%；两窗口各自 median 比 ≤1.10 |
| 绝对容忍带 | 当前好核 / 独立预检好核 ∈[0.5, 2.0]，沿用既有心跳容忍带 |
| 配对负控 | 保守慢/好分离度 ≥2；ratio-of-medians 与 paired median 取较小者 |
| cadence | 显式启用 profile 时 heartbeat_interval 必须为 1；每次主测/确认重测 |

容忍带用于发现粗粒度量具失效，**不等于精确时钟认证或锁频**；例如边界上的 2×
偏移仍在带内。预检稳定性规则拒绝明显不稳定的标定，不据此重试直到得到较小地板。

### 2.1 两种独立性，不可混为一谈

1. **独立测量窗口**：任务心跳的好核时间除以预先冻结的好核时间。
   分子/分母不能来自当次同一窗口；主测、确认也不能复用预检或心跳 window。
   因而后续两臂同比例 0.1×/10×，即使慢/好比不变，也不能自归一化为 1。
2. **独立时钟**：每次预检另跑 3 次固定长 batch，每 batch 1024 次真实 copy 调用；
   同一个 batch 同时被生产 `_timed_runs(..., 1, True, flush_l2=True)` CUDA event
   与 host `perf_counter` 观察，要求每次 CUDA/host 比在 [0.5, 2.0]。
   这阻止预检 event 已整体错 10×、后续还用同一错尺而看似匹配。

host batch 外围包含冲刷、event 管理和 synchronize，不是单核净时间，也不作为 floor。
长 batch 降低这些外围固定成本的占比；与此前另一次 host 单核样本不同，此处观察的是
**同一次执行**。每次 copy 的输出立即丢弃，不保留 1024×8 MiB 输出。
没有复制 event 循环、引入用户配置的 repeat/shape 或将 host 与 event 时间相减。

## 3. 身份绑定与失败关闭

原生回执继续绑定完整部署源码、固定参考/原值输入、计时配置、物理 GPU/可见集、
UUID/PCI、驱动/Torch/CUDA/Triton 与默认 stream。新增 profile 模块本身也加入
部署和 production measurement protocol 全文摘要。

生产 `run_benchmark_with_baselines` 使用同一个 `ruler_identity` 生成 timing receipt，
绑定实际 L3 两臂 sample count、逐臂/结束设备、同窗计时与结束默认 stream。
其中固定参考/输入字段是**标定仪器身份**；任务 workload 仍由自己的原值 readback 和
run manifest 绑定，不能拿 8 MiB 标定输入冒充 240 MiB 主输入。

启用新地板前必须同时成立：

- 本窗口原生 CUDA 心跳通过、实际完成独立绝对检查，profile hash/identity/比值/原始
  evidence 与重新计算的 anchor check 完全一致；separation-only 心跳不够资格。
- 实际 L3 receipt 的 channel、全部身份、requested warmup/repeat=10/30、默认 stream 匹配。
- candidate/baseline/paired sample count 相同且在 30..5000；block 在 1..10。
- candidate/baseline/after 设备一致；两臂 window 相同且不复用两个预检或本次心跳 window。

缺失、不匹配或绝对锚点失败：记录 `ruler_identity_error` / `RULER_BROKEN`，
measurement 为 uninformative，L3 不许 passed，Oracle 全资格撤销；**不退回旧地板或 CPU**。
无 profile 的旧路径保持原行为，不额外改变 Golden 的 legacy metrics 形状。

profile hash、标定 identity hash、heartbeat cadence 进入 frozen run controls。
只改变 profile 选择的 resume，也必须在 routing/LLM/GPU 前 HOLD；源码改变则完整
protocol hash 改变。新版本必须重新预检并重新测量，不是跨版本通行证。

信任边界仍是可信控制端和 owned runner：SHA 绑定证明完整性/一致性，不是签名认证，
也不防恶意 owner/same UID 伪造所有回执、同时篡改两个时钟或跨进程事务攻击。

## 4. 实际 H20 独立预检

最终预检：`output/cuda_calibration_h20/cuda-calibration-h20-20261009T045122489633Z-6281-1`。
固定 GPU 2、现有 runtime、私有 owned runner，未安装/sync 或修改共享环境。

| 观测 | audit 1 | audit 2 |
|---|---:|---:|
| 实际 paired window | `2122d80ee8ae` | `a35aafb8856a` |
| 空核 no-flush median | 19.600 µs | 19.408 µs |
| 空核 flush median | 4.576 µs | 4.608 µs |
| 空核 flush P95 | 4.800 µs | 4.768 µs |
| 生产 paired 好核 median | 10.496 µs | 10.496 µs |
| 生产 paired 好核 IQR/median | 2.13% | 1.83% |
| 保守慢/好分离度 | 7.6420× | 7.2577× |
| 同 batch CUDA/host 比，3 次 | 0.99339 / 0.99423 / 0.99465 | 0.99253 / 0.99442 / 0.99474 |

因此派生保守 floor **4.800000227987766 µs**、expected_good **0.010496 ms**：

- profile：`10ff2b43842dfd035017fcf24ac037f81a4556700a3d443a4b4de8b06624c770`。
- calibration identity：`2b524238408372ca5b2da5a6b2f62f552af32abd2d7ab168818278d95a462838`。
- 完整 calibration：`bcaa99b60ad64c8a0d27f892889d521950491b4634b7baafa473a9acad5081c8`。
- Torch `2.14.0+cu130` / CUDA `13.0` / Triton `3.8.0` / driver `580.105.08`。
- GPU UUID `GPU-13c6c4b4-45ce-8963-1122-df2b3d72fb76`，PCI `00000000:A2:00.0`。

审计 telemetry 是快照且包含探针自身活动，不是持续无竞争或锁频证明。
两次 audit 后 shutdown ACK、owned PID absent/quiescent、scratch removed 均确认，
signals=[]、forbidden_actions=[]；才 atomic/fsync 写出 readonly profile。

### 过程中的失败和版本变更保留

首次运行 `cuda-calibration-h20-20261009T044324767858Z-1968-1` 在 runtime SSH 探测时
连接关闭，没有 GPU audit、没有创建 scratch；完整失败 summary 保留。
做过一次启动前只读检查和一次有界 setup retry，不是按性能结果重试。
随后产生过未加同 batch 独立时钟的中间 profile（hash `462bb24b…`），没有用于候选验收。
加入独立时钟后源码/身份改变，重新完整预检，最终只使用上表的新版本。
这不是事后追认中间或更早的历史回执。

## 5. 同 workload 的真实主测、确认与入库

运行：`output/durable_profile_h20/durable-profile-h20-20261009T045546001660Z-9093-1`。
使用显式冻结 profile；原 8192×15360（240 MiB）/8192×12288（192 MiB）、值模式、
候选和参考不变，未自适应增加尝试或挑最快窗口。
候选 SHA：`ea4e9fe9363d79d0b75b3be7fabcbe51f161fa6b1e4d9d0b84968765e5ba55c8`。

预声明 plan hash：`f082534e8862cc406d248c00a1bd64475a05ff13caea8103daa022a5e2cd2c99`。
实际 evaluator protocol hash：`8b07561577933d178e27c694c5ee90e73c50dada1524fa6d15f791fe7ec70661`。

| 主输入 / 过程 | 主测净时间 | 主测 / 确认比值 | 主测地板占比 | 主测 / 确认 | 实际入库 ID |
|---|---:|---:|---:|---|---|
| 240 MiB，8192×15360 | 149.216 µs | 2.0169× / 2.0174× | 3.12% | valid / valid | `0e49657e` |
| 192 MiB，8192×12288 | 117.248 µs | 1.9522× / 1.9530× | 3.93% | valid / valid | `7cc7a77c` |
| 合法 resume，240 MiB | 149.152 µs | 2.0178× / 2.0172× | 3.12% | valid / valid | `179e4d82` |

所有比值仅对这些预声明主 case、该 frozen evaluator 及本次窗口成立。
它们是**新测得的 valid 观测**，不是旧读数被追认；候选没有变得更慢或被换成别的核。
每个过程重新 compile、主输入正确性、完整 32-case suite/重复确定性，再主测和独立确认。
实际 trace 每个过程为 1 compile、65 verify、2 benchmark 和 2 native heartbeat；
确认是新的性能窗口，不声称它又独立重复了一遍 32-case suite。
合法 resume 零 Designer/Coder；输入值/阈值漂移均 HOLD，零 RPC/生成/corpus 副作用。

独立审计在 driver 返回后重新读实际文件，而非只读 `admission.kept`：

1. 三份 manifest 的完整源码摘要与当前 evaluator 相同，controls 固定同一 profile/identity/cadence。
2. 六个原生心跳实际好核/锚点比为 0.99695..1.0，慢/好比为 6.4894..6.6906；
   从原始 evidence 重算 anchor check，再重建 HeartbeatResult 验证真实 L3 timing receipt。
3. 两次预检、六个心跳、六个主测/确认计时 window **14 个全部不同**。
4. 三份实际 confirmation 文件重读：候选/contract/workload/protocol 同一身份，
   verification/window 独立，确认自身 valid 且 ≥1.05；无旧确认借用。
5. 实际 JSONL 是 **3 个 kernel append event**，三条完整记录匹配候选源码、任务字段、
   主测 valid score 与 run binding；新建 KernelDatabase 重放全部精确恢复，skipped lines=0。
   driver 还逐次验证只追加一个 event、此前前缀不变；failure corpus 零写入。
6. shutdown ACK、owned PID absent/quiescent、私有 scratch removed 均确认，
   signals=[]、forbidden_actions=[]。

最终 `functional_completed=true`（5/5）、`completed=true`、三个 positive admission。
独立审计保存于 `output/cuda_ruler_profile_validation/evidence-audit.json`，
实际 corpus 位于上述运行目录的 `corpus/kernel_db.jsonl`。

## 6. 回归与守卫标定

本轮验证日志：`output/cuda_ruler_profile_validation`。

- profile 单测覆盖保守派生、重算 hash 不能授权改尺、预检缺失/抖动、独立时钟粗缩放、
  后续绝对锚点同比例缩放、设备/runtime/source/stream 漂移、sample parity、window 重用、
  separation-only 不能启用、新尺缺证据不能 fallback、Oracle 撤销和 resume 换尺 HOLD。
- 新量具守卫 mutation 在临时隔离副本执行：clean pass，**14/14 断言杀死**，
  不是 import/collection failure；含独立 clock、禁止自归一化、floor 取较大 P95、
  profile 派生权威、生产身份/样本、window 独立性、拒绝 fallback、Oracle 与 frozen controls。
- profile 专项 **47 passed**；加上 durable contract 专项为 **94 passed**。
- 全量 **3312 passed，1 skipped，2 warnings**；两条既有 nonfinite-reference warning 保留。
  Golden **4/4 PASS**，未刷新 registry。先前全量发现 legacy 路径多余 metrics 键时，
  修复行为兼容性后重跑，没有为通过测试而修改 Golden 基线。
- 离线 fake receipt/缩放负例不是在 H20 上篡改计时器的证据，二者不混称。

## 7. 复现与后续边界

```sh
python benchmarks/cuda_calibration_h20.py --gpu 2 --ruler-profile
python benchmarks/durable_profile_h20.py --gpu 2 \
  --workload 8192x15360 --workload 8192x12288 \
  --cuda-ruler-profile /absolute/path/to/ruler-profile.json
python -m pytest tests -q
python tests/golden_regression.py compare
```

只能在匹配预检源码/设备/runtime 的新观测上启用，未向全局或其他 GPU 推广。
本次 isolated corpus 的 append/fresh replay 也不提供跨进程 exactly-once、崩溃事务或
跨任务语料可信性承诺。性能结论只覆盖预声明主 workload；32-case suite 证明数值覆盖，
不证明每个 shape 都加速，尤其不能把 launch-dominated 小 shape 当成性能正例。

后续的本地单写者提交/显式幂等恢复见
[本地语料持久化提交与恢复](durable-corpus-admission-and-recovery.md)。本页 H20 证据属于
`9fbc099` 对应的 frozen evaluator；该后续协议改变源码身份，需要重新预检和测量，
不能直接沿用本页 profile 作为新版验收。
