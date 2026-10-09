# Session-owned 大输入分块传输与缓存

日期：2026-10-09。接续 [冻结运行契约与恢复](durable-run-contract-and-resume.md)。

## 1. 问题与不变量

此前 8192×8192 fp16 主输入有 128 MiB storage，整块 base64 嵌入单行 JSON。
真实 H20 在 180 s RPC 截止前仅发送约 10 MiB，返回 send timeout/unknown；
因此没有完整 L2 或 L3，不能解释成候选数值/性能失败。

本轮只优化**实际输入的传输**：不缓存候选输出、Torch/CUDA tensor、数值通过凭据或计时结果。
不改参考、容差、32 组 suite、测量窗口、launch 地板、确认或 admission 门。
L3 和独立确认仍是新 runner 调用、新输入对象、新计时窗口。

## 2. 协议与部署

- 小于 1 MiB 的 storage 继续既有 `numpy-storage-json-v1` 内联精确协议。
- 常驻 NVIDIA 会话的大 storage 使用 `numpy-session-inputs-v1` envelope。
- 非常驻/不支持的平台保留原路径；新版缓存握手缺失时，**不悄悄回退**巨型内联或 shape/seed。
- PreparedInputs 是 JSON-safe manifest；不可变 raw bytes 只在控制端对象属性里，
  不被 JSON serializer 当成 wire 数据。标量/字节上限和 view decoder 沿用同一份 authority。
- manifest 包含每个 storage 的 SHA-256/长度、完整 tensor view 与标量 wire metadata，
  全部内容进入 manifest hash。run_manifest 仍绑定实际值/布局、任务和 evaluator 源码。
- `session_inputs.py` 部署为共享源的 sibling，位于消费者 runner_server 前；新增源摘要
  进入 frozen run 的 measurement_protocol。旧 evaluator 的 resume 会 HOLD，不能无声升级。

```text
ping -> 本次进程的随机 session_id + 协议/分块/容量/raw,zlib
input_begin(session_id, 完整 manifest) -> 校验 + 预留容量 + missing storages
input_chunk(manifest_hash, storage_hash, 连续 offset, encoding, bounded base64)
  -> 有界无损解码 + offset 回执（offset 按原始字节计算）
input_commit -> 完整长度/SHA + scalar/view 校验 -> 原子发布整批输入
verify / benchmark_with_baselines(同一 manifest + session_id)
  -> 重新核全部 storage 字节摘要
  -> 原 decoder，fresh NumPy roots -> fresh Torch storage/views
  -> 实际 Torch 输入 readback -> 原数值/计时流程
```

只有 server 能创建 in-process ResolvedJSONInputs。普通 JSON 字段不能冒充 raw-byte carrier；
kernel runner 仍读既有精确协议，不新增设备 action，也不接受客户端任意文件路径。
三条 upload control action 仅由 resident server 处理，纳入控制端/服务端的 AST 协议守卫。

## 3. 生命周期与资源

缓存仅在**本次 runner 的进程内存**，不建共享文件、不存全局 cache，不跨 session 复用。
session_id 与 owned-process 清理 token 是不同概念；nonce 用于防过期引用，不是 hostile-code 认证。

- 单 chunk 最多 512 KiB raw bytes；输入 storage 加标量仍受既有 256 MiB 上限。
- 每块尝试标准库 zlib level=1，仅压缩结果更小时发送 zlib，否则发送 raw。
  摘要、长度、offset、容量都以**未压缩的原始字节**为准，不改变实际输入。
  接收端 `decompress(..., 512 KiB + 1)` 限制分配，拒绝超限、截断、尾随数据、
  拼接流和未知 encoding；不调用无界 decompress/flush。
- 常驻 blob 容量 256 MiB；一个 pending upload transaction，容量计算包含待接收预留字节。
- LRU 只淘汰不属于当前整批输入的完整 blob；当前依赖不能被自身上传挤掉。
- chunk 必须严格连续，不接受重放/重叠/bool offset；不完整输入不可见。
- commit 先验证整批，再发布 immutable bytes；内容相同的 storage 可去重传输，
  但每个原 storage group 仍单独物化，**不虚构 alias**。
- 候选修改本次输入不会污染缓存原件，下一次重新物化。
- 正常 shutdown/EOF 清空缓存；破损会话继续沿既有独立控制通道确认 owned runner 已退出，
  不能把本地 SSH 退出当成远端静默，也不 broad kill。

256 MiB 是存储/预留容量，不是进程 RSS 上界：immutable promotion、NumPy/Torch 物化及
实际 readback 有有界的临时副本。本轮没有 GPU tensor cache 或跨进程持久化缓存。

## 4. 截止与失败关闭

一次调用的 lock、每个 upload RPC、commit、最终候选 RPC 共用同一个 monotonic deadline；
不能每块重置 timeout，把大输入传输变成无限运行。既有初始化/本地输入准备阶段不冒充
计时窗口；数值/性能测量仍在原位置。新协议不增加生产默认 timeout。

无缓存握手、nonce/manifest/长度/摘要/offset 回执不符、未提交或过期 batch：
input identity unknown，客户端标记 session broken。没有自动 retry chunk、重建会话或
输入降级；原 L2 unknown/HOLD 路径不调用 Fixer、不写正/负 corpus。
远端整批元数据错误也在 candidate source exec 之前拒绝。

每次响应登记 `input_transfer`：session/manifest、完整内容校验、实际引用 storage 字节数，
以及控制端统计的 uploaded_bytes/uploaded_chunks/reused_storages、chunk_encodings
和 uploaded_wire_bytes。uploaded_bytes 统计未压缩输入；uploaded_wire_bytes 统计实际发送的
完整 chunk JSON 请求（含 base64、字段与换行，不含 begin/commit/最终 RPC）。它是传输审计信息，
**不能替代实际 Torch readback、参考身份、数值门或测量有效性**。
L2/L3 的登记字段和真实 H20 compact trace 保留它；harness 要求 L3/确认零重复上传。

## 5. 明确边界

1. 本轮优化大 ndarray backing storage；巨大标量仍受旧字节边界，但未增加 scalar 分块协议。
2. Hash/nonce 处理正常漂移、损坏和生命周期错配，不证明 hostile 同 UID/Python 候选隔离。
3. 原 view/标量规则不放宽；Torch 不支持负 stride/readonly 时仍无法确立，不转连续副本。
4. 缓存命中不是 L2/L3 通过；每次都重新执行，不复用结果或跳过数值 suite。
5. 本次不实现 durable input blob checkpoint、跨 session reuse 或 exactly-once admission。
6. 无损压缩不是任意输入的吞吐保证：不可压缩输入仍发 raw，低带宽时可能超时/HOLD。
   真实验收主输入是明确声明的重复模式，不能把其压缩比推广到随机数据或一般模型输入。

## 6. 验证入口

```sh
python -m pytest tests/test_session_inputs.py tests/test_rpc_deadline.py tests/test_remote_input_fidelity.py -q
python benchmarks/calibrate_session_input_guards.py
python -m pytest tests -q
python tests/golden_regression.py compare
python benchmarks/durable_profile_h20.py --gpu 2 --workload 1024x1024 --workload 8192x8192
```

专项覆盖真实 manager、shared decoder、部署 sibling 源后的常驻进程/管道、实际 Torch CPU
readback、内容去重不造 alias、候选写入不污染原件、LRU/partial/nonce/commit/回执/共同截止，
以及压缩/raw 自动选择、解压炸弹、截断/拼接/尾随流和分配上限。
CPU 测试不冒充 native CUDA 或性能证据。

变异标定在私有副本完成：clean 必须通过；manifest hash、nonce、storage hash、未提交
使用、offset、容量、上传回执、最终输入凭据、每块重置 deadline、无界解压共 10 个负控。
仅 pytest assertion failure 算杀死，collect/import 错误不计入。

## 7. 本地验收

- session-input 专项 **50 passed**；包含 runner/driver/shared-scratch 守卫的组合 **87 passed**。
- 全量 `pytest tests -q`：**3171 passed，1 skipped，2 warnings**。两条 warning 来自既有
  nonfinite-reference 负例；未改容差或过滤告警。
- Golden Regression：**4/4 PASS**。
- 变异标定：clean 通过，**10/10 assertion-killed**，没有以导入错误冒充守卫有效。

本地日志、变异报告与源码摘要保存于
`output/session_input_validation/session-input-final-20261009T025533484949Z-49779-1/`。

## 8. 真实 H20：传输已通，不等于可信正向入库

### 分块但未压缩的首轮

`durable-profile-h20-20261009T023509243451Z-36347-1`：1024×1024 主输入只上传一次，
L3/独立确认零重传；8192×8192 的 128 MiB 输入仍在共同 180 s 截止内上传超时，
`session_phase=response`、`request_id=199`、`bytes_sent=699340`。
该 bytes_sent 是最后一条请求的发送量，**不是完整 storage 已成功上传量**。
没有得到大输入候选的数值/计时结果，也没有入库。

破损会话没有 shutdown 回执；独立控制通道验证本次 owned runner 身份后发 SIGTERM，
确认 quiescent/removed=true，没有 broad kill 或 forbidden action。
旧 harness 在提前 HOLD 没有 resume_status 时另有空值异常；已修复为保留失败证据，
不是把 HOLD 改成恢复成功。原失败记录保留。

### 有界无损压缩后的完整功能验收

`durable-profile-h20-20261009T024734964373Z-42180-1`：预先固定 1024×1024、8192×8192，
门槛仍为 1.05。两组实际 L1、主输入 + 完整 32-case suite、L3 和独立确认均执行；
合法恢复零 Designer/Coder、重新编译/验证；两种漂移均 HOLD，零 RPC/生成/corpus 写入。
**functional_completed=true，5/5 无 harness 验收错误**。

| 主输入 | 首次未压缩上传量 | chunk 请求实际 wire bytes | zlib chunk 数 | 主测/确认重传量 |
|---|---:|---:|---:|---:|
| 1024×1024 fp16 | 2,097,152 | 16,305 | 4 | 0 / 0 |
| 8192×8192 fp16 | 134,217,728 | 1,044,518 | 256 | 0 / 0 |

以上 wire bytes 只含 chunk 请求，不是全流程网络计费量。执行侧同时提供完整 storage
内容校验、原 view/scalar 解码和实际 Torch readback；参考身份仍逐数值 RPC 核对。

大 workload 主测表观 **1.6845×**、确认 **1.6848×**，但 launch 占比 **19.71%**，
主测为 **degraded / can_gate=true / can_persist=false**；小 workload 为 uninformative。
两者均被原 admission 门拒绝，corpus 无正向写入；**positive_admissions=[]，completed=false**。
不能把“ALL PASSED”或两个表观加速比写成可信持久化收益。

正常 shutdown 有回执，quiescent/removed=true、signals=[]、forbidden_actions=[]。
原始 summary、compact trace、verification、sealed snapshot 保存在上述 run root。

### 同一 workload 的一次独立会话复测

`durable-profile-h20-20261009T025239840692Z-46477-1`：仅声明原 8192×8192，
源码/候选/输入/测量门保持不变，不做“重复直到通过”的选择。
功能与恢复检查 **4/4**；首次上传 134,217,728 raw bytes，chunk JSON 共 1,044,429 bytes，
主测/确认和合法恢复中的 L2/L3/确认均零重传。

主测表观 **1.6865×**、确认 **1.6858×**；合法恢复表观 **1.6860× / 1.6866×**。
主测 launch 占比 **19.74%**、恢复 **19.75%**，两者仍 degraded，均拒绝入库。
functional_completed=true，但 positive_admissions=[]、completed=false；无正/负 corpus 写入。
正常 shutdown、quiescent/removed=true、signals=[]、forbidden_actions=[]。
这次结果与上一轮一起保留，不只展示较快读数。

H20 launch 地板仍沿用既有 20.24 µs 标定，不是本轮动态调低；缓存只解决传输问题。
当前尚欠 **valid 主测 + 原独立确认门通过 + 实际 corpus admission**；若继续扩展，
应事先声明有界的大工作负载与其身份，或独立标定测量口径，不能降低 validity/精度门。

后续验收：[有界大 workload 与实际 corpus 读回](bounded-workload-and-corpus-acceptance.md)。
上文两轮 degraded 是本阶段历史记录，不把它们改写成 valid。
