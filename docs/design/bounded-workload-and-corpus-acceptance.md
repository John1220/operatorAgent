# 有界大 workload 与可信 corpus admission 验收

日期：2026-10-09。接续 [session-owned 输入缓存](session-owned-input-cache.md)。

## 1. 本轮目的

128 MiB 输入已能执行完整 L2、主测与确认，但净执行时间约 82 µs，既有 launch 地板
20.24 µs 占端到端时间约 19.7%，仍为 degraded，不具备长期持久化资格。

本轮扩大**事先声明的真实输入**，不是在候选里增加无效计算、sleep、循环次数或派发开销。
候选仍为原 `unfused-fp32` 通用实现，不对主输入 shape/值特化。
参考身份、fp16 输出/累加语义、完整 32-case suite、数值容差、launch 地板、测量窗口、
1.05× 门槛、180 s RPC 截止、独立确认与 production admission 均不修改。

## 2. 预声明计划与资源边界

固定选择两组，按以下顺序完整执行，不根据读数临时追加 shape 或“重测直到通过”：

| 顺序 | 主输入 shape | fp16 backing storage | 对应原候选 tile 上界 |
|---|---|---:|---:|
| 1 | 8192×15360 | 240 MiB | 16384 columns |
| 2 | 8192×12288 | 192 MiB | 16384 columns |

行宽扩大带来的 next-power-of-two tile 是既有通用候选的真实行为，不是本轮人为插入的工作。
实际值仍是明确声明的 `[-1,0,1,2]` 重复模式；完整 suite 继续包含随机、布局、尾部、
精度等独立负控边界。这不是一般模型输入的压缩率/性能结论。

- CLI 只允许有限白名单；原默认三组和 resume-only 小输入不变。
- 重复 workload、未列入白名单的 shape、互斥验收模式在任何 backend/输出副作用前拒绝。
- 分配前按 fp16 storage + 空 scalar wire 的 2 bytes 核对 256 MiB 协议上限。
  恰好 256 MiB 的 ndarray 也不是合法整批输入：没有为 scalar wire 留空间。
- `plan.json` 在 remote preparation/输入分配前以 atomic write + fsync 落盘，mode=0444。
  记录两组顺序、字节估算、值模式、候选 SHA、参考 SHA、门槛、timeout、scope 和计划摘要。
- 生产 `run_manifest.json` 仍另行冻结**实际字节/view/标量**、任务、环境与 evaluator 源码。
  新主输入是新工作负载身份，不借用旧 8192×8192 的证据。
- 256 MiB 是 input storage/scalar 协议上限，不是 host RSS 或总显存承诺。
  物化、Torch readback、输出与测量冲刷有临时副本；只使用既有闲置 GPU 检查和私有 runner。

240 + 192 MiB 超过 256 MiB session cache 容量，切换 workload 时允许正常 LRU 淘汰。
合法恢复第一个 workload 可能再次上传 L2 输入，**不能要求跨淘汰零上传**；但主测/确认
必须复用刚验证的完整 blob，仍由 `transfer_errors` 检查零重复上传。

## 3. 验收的是实际写入，不是 metadata 标记

生产 `admit_to_corpus` 仍是唯一入库判据；本轮仅加强 harness 对其实际后果的读回审计。
首测必须 valid / can_gate / can_persist；独立重测按照原规则可为 usable valid 或 degraded，
不能把“主测 valid”擅自改成“两次都必须 valid”，也不能把 uninformative 当确认。

声称 `kept=true` 时，harness 额外核对：

1. 实际 `confirmation.json` 与返回的 ConfirmRecord 一致。
2. 首测/确认的 run、manifest、contract、workload、protocol、candidate 摘要一致。
3. verification_id 与实际 candidate timing window 均不同；确认自身 metrics 与绑定一致。
4. 确认可用、独立、speedup 有限且不低于 1.05；bool/NaN、缺失/过期记录不得放行。
5. corpus 必须在本次 before 快照之后追加恰好一条实际 kernel event，不能仅有任意文件增长。
6. event 的 kernel_id、代码、任务、valid 档位、score、compile_info 与当前候选证据完全一致；
   compile_info 的 run_binding 必须与本次主测一致，候选 SHA 必须匹配实际保存代码。
7. **新建 KernelDatabase 从磁盘重放**，读回同一条完整记录，没有跳过坏行。

审计结果保存在每个 trial 的 `corpus_readback`；正向 admission 列表只收录所有检查无误且
实际读回成功的 trial。否决时 corpus 字节必须不变；两种 resume 漂移继续要求零 RPC、
Designer/Coder 调用及 corpus 副作用。

这证明本次单写者、隔离 corpus 的实际写入，不新增跨进程 exactly-once、崩溃事务、
hostile 同 UID 隔离或跨任务语料可信性承诺。

## 4. 运行与声明范围

```sh
python -m pytest tests/test_durable_profile_driver.py -q
python -m pytest tests -q
python tests/golden_regression.py compare
python benchmarks/durable_profile_h20.py --gpu 2 \
  --workload 8192x15360 --workload 8192x12288
```

只 stub Designer/Coder；Conductor/L1/L2/L3/确认/admission/persistence 全走生产链路。
所有失败、degraded、拒绝与清理事实都保留，不仅展示较快的样本。

性能 scope 是**每组预声明主 case 单独成立的结论**，没有 aggregate 权重，也不是
“所有 shape 已验证更快”或生产部署建议。full suite 是数值覆盖，不是全部 shape 的性能覆盖。
如果本轮仍不能入库，保持 completed=false，不通过缩门、偷换地板或省略证据收口。

## 5. 验收结果

### 本地

- driver 专项 **47 passed**，新增 **40** 条预声明/资源上限/真实 JSONL 重放/确认身份负例。
  单测使用显式离线构造的测量 fixture，不冒充 CUDA 性能证据。
- 全量 **3211 passed，1 skipped，2 warnings**。warning 为既有 nonfinite-reference 负例，
  没有屏蔽或放宽精度判据。
- Golden Regression **4/4 PASS**；未修改 golden registry。

本地日志、H20 日志、审计摘要与当前 evaluator 源摘要：
`output/bounded_workload_validation/bounded-workload-20261009T031738997361Z-60509-1/`。

### 真实 H20：接近 valid 边界，但本轮仍不入库

固定计划 SHA：`3459b902d6872db89952e382fb254564ac5be518ed5fbe43995a2b20c9e21f02`。
运行：`output/durable_profile_h20/durable-profile-h20-20261009T031130385434Z-56545-1/`。
没有修改候选 SHA `ea4e9fe9363d79d0b75b3be7fabcbe51f161fa6b1e4d9d0b84968765e5ba55c8`。

| trial | 净执行时间 | 主测表观加速比 | 确认表观加速比 | 主测 launch 占比 | 主测/确认档位 |
|---|---:|---:|---:|---:|---|
| 8192×15360，240 MiB | 149.280 µs | 2.0116× | 2.0142× | 11.94% | degraded / degraded |
| 8192×12288，192 MiB | 116.672 µs | 1.9608× | 1.9620× | 14.78% | degraded / degraded |
| 合法恢复 240 MiB | 149.184 µs | 2.0129× | 2.0120× | 11.95% | degraded / degraded |

三次都实际重新执行 L1、主输入与完整 32-case suite（每个 suite case 确定性重复通过）、
真实 L3 与独立确认。源码部署摘要、原生参考回执、实际 CUDA readback 和运行绑定核对无误。
三份实际 `confirmation.json` 另经新确认审计函数读回通过，身份相同、verification_id 与
候选计时 window 不同；这验证确认审计，不等于确认可越过首次 degraded 的入库拒绝。
两组新工作负载摘要不同，不复用旧 128 MiB 或彼此的数值证据。
合法恢复零 Designer/Coder，fresh compile/verification；输入值与阈值漂移均 unknown/HOLD，
零 RPC/生成/corpus 副作用。**functional_completed=true，5/5 无验收错误**。

传输事实：

| trial | L2 上传的 raw bytes | chunk JSON 实际 wire bytes | chunks | 主测/确认重传量 |
|---|---:|---:|---:|---:|
| 首次 240 MiB | 251,658,240 | 1,958,573 | 480 | 0 / 0 |
| 首次 192 MiB | 201,326,592 | 1,566,906 | 384 | 0 / 0 |
| 恢复 240 MiB | 251,658,240 | 1,959,142 | 480 | 0 / 0 |

恢复中的 L2 再次上传是预期的有界 LRU 淘汰后行为，不是跨 session/cache 的隐式重建或
候选结果缓存；每次 L3/确认都独立物化与测量。wire 计数只包含 chunk JSON 请求。

三次主测均 `can_gate=true`、但 `can_persist=false`；原 admission 门全部拒绝。
**positive_admissions=[]，completed=false**，没有正/负 corpus 写入。
新增正向 corpus 读回审计本轮只取得离线测试证据；真实 runner 没有正向 admission，
所以 `corpus_readback=null`，**不能声称已在 H20 上验证实际入库/重放闭环**。
driver 退出码 1 是这项未完成状态，不是候选算错或 harness 异常。

本次正常 shutdown 有回执，owned runner `quiescent=true`、scratch `removed=true`，
`signals=[]`、`forbidden_actions=[]`。失败/拒绝记录全部保留，没有事后追加 workload 或
重复挑选较快读数。

## 6. 下一步边界

维持 20.24 µs 地板和 10% valid 边界时，净执行时间需超过 182.16 µs 才能跨过该项门。
240 MiB 已接近 256 MiB 输入上限，实测仍只有 149.28 µs；本轮并没有证明“再略增一点”
就能成功，也没有穷举其他行宽/布局。

因此不应继续盲目放大输入或人为拖慢候选。后续优先做**独立测量口径审计/标定**：
核对既有 launch 地板的标定方式、当前净执行计时与端到端时间的关系，保留配对负控，
若要更新标定必须先形成证据与新的 evaluator 身份，不能为了本次过门直接改小数字。
之后再冻结下一组 workload/口径并做真实正向 admission；原精度、确认与入库门仍不可关闭。
