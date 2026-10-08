# 远端 L2 与 Torch 输入保真协议

日期：2026-10-08。承接 [本地输入保真](input-fidelity-protocol.md)。
本片实现 NVIDIA SSH 的验证路径和本地真实 CUDA/Triton 输入转换；
**离线协议测试不等于远端 GPU 真机验收**。

## 为什么分开 L2 与 L3

旧 SSH 协议只发 shape/dtype/seed，runner 重新抽随机数；`run_verify` 又将输入
`.contiguous()`，fp16 精度参数还会隐式转换输入 dtype。它测到的是相似输入，
不是 CorrectnessSuite 声明的值、布局和 dtype。

现在 NVIDIA `SSHExecutor.verify_function` 无条件使用原值协议，`RemoteExecutor`
转发实际内层的 `correctness_input_support`。未声明套件的调用也收到原始输入 dtype，
`precision` 只选验证容差、不再在 exact 路径偷偷转型。

`execute`、`benchmark`、`benchmark_with_baselines` 的旧传输未迁移；L3 仍按
shape/dtype/seed 生成自己的工作负载。套件输入仍不进入 L3 的 case/权重。
**本片不宣称 L3 的任意调用方原值或布局已保真**。

## `numpy-storage-json-v1`

沿用本地 storage/view 模型，差异只有 JSON 编码：

- `storages` 用 base64 字节串，不发逐元素 JSON 数组，不用 seed 代替原值。
- `tensors` 保留 shape、dtype.str、字节 strides/offset、storage 分组和 writeable。
- 标量用显式类型；NumPy 标量用 dtype+字节，float/complex 用 IEEE 字节。
  NaN/Inf 不依赖 JSON 非标准数值 token；bool、bytes、None、np.float32 不做类型合并。
- 共享 storage 只传一次。storage 与标量载荷有 256 MiB 上限，tensor 逻辑大小及
  view 地址范围复用本地校验。base64 长度在解码分配前检查。
  这不是整个 JSON 文本、元数据数量或进程内存的统一配额。
- 未知协议、非法 base64、越界 view 不退回随机生成或连续副本。

`input_fidelity.py` 和 `remote_input_fidelity.py` 由 `_remote_files` 排在 runner 前同步；
推送的是控制端同一份源码，不维护远端副本。一次性 SSH 与常驻服务共用 `run_verify`。

## 实际 Torch 对象

`to_torch_inputs` 先按 storage 建 uint8 buffer，将整块 storage 复制到目标 device，
然后用真实 untyped storage 构建显式 dtype/shape/stride/storage_offset 的 view。
逐输入 `.to(dtype)` / `.contiguous()` 不再出现在 exact L2 路径。
同 storage 的多个输入共享目标 storage，控制端对象与候选对象隔离。

`torch_input_readback` 在候选/外挂参考源码执行前，读取实际 Torch tensor：
dtype、shape、字节 stride、字节 offset、storage 分组/长度/完整哈希、逻辑值哈希和
实际 device。对账使用与 NumPy 相同的 `numpy-storage-v1` 回读字段；
`representation=torch-storage-v1` 表示这些字段来自 Torch，不是 NumPy 对象。

控制端仍在后端调用前保存预期描述，严格核类型和值。证据缺失/不兼容为 None，
不一致为 False；两者均使声明套件 unknown，不进入正式 L3。成功、数值失败及形状
失败都保留证据，完整 JSON 进产物；报告额外打印实际 tensor device。
声明 `REAL_DEVICE` 的远端还必须提供每个输入 tensor 的同一加速设备回读：CPU/meta
或缺少 device 证据不能晋级为真机正确性。离线测试后端只声明 REMOTE_VERIFY，不声明 REAL_DEVICE。

本地 `MockGPU._execute_triton_cuda` 共用上述转换与回读。没有真实 CUDA/Triton 时的
模拟参考仍不提供候选输入证据，不能冒充套件通过。保真转换明确拒绝时不走模拟兜底。

## 不能表达的语义不伪造

- Torch 不支持负 stride；拒绝，不转连续副本。
- Torch 没有 NumPy 的 readonly 属性。readonly 输入拒绝，回读不照抄声明。
  因此当前 `CorrectnessCase(layout="broadcast")` 由 `np.broadcast_to` 构造的只读
  输入在 Torch 路径仍为 unknown；**不能宣称 GPU broadcast 套件已全通过**。
  可写零 stride view 在转换层可保留，但它不等于只读 broadcast 的语义。
- 非原生字节序、Torch 不支持的 dtype、未按元素大小对齐的 stride/offset 拒绝。
- 未接通 XPU/HTTP、旧远端 runner 无回读证据时，声明套件仍保持 unknown。
- 不证明 backward、in-place/aliasing 算子语义、GPU launch/fallback 审计、OS 隔离。
  同 uid 文件改写、恶意候选反射伪造报告等不是此协议解决的攻击面。

## 比对、确定性和参考边界

远端返回完整实际输出的 dtype/shape/bytes SHA-256，先于 fp32 比对转换。
缺指纹时确定性 unknown；两次完整指纹不同，即使误差低于容差也失败。
输入/输出哈希的 CPU 回读在 L2 进行，不计入正式 L3 的 kernel 计时。

参考输入独立复制，候选写输入不会污染参考或控制端。exact 路径声明了
`reference_source` 却没成功建立参考时，不静默改用内建参考：返回 reference_error，
套件 unknown。非有限参考只统计可比较元素，透传 reference_nonfinite 与 coverage，
不靠 NaN 比较结果假装全量通过。

## 验收

守卫：`tests/test_remote_input_fidelity.py`。包含 strict JSON 标量/存储往返、
Torch 实际 view/offset/共享 storage 回读、正确与错误正 stride 核的 L2→L3 对照、
readonly 拒绝、真实连续化/证据篡改拦截、完整输出指纹、非有限/错误参考，以及
将部署清单复制到独立目录后由新 Python 进程运行 runner 的测试。

2026-10-08 验收记录：

- 本片定向测试：53 passed / 1 skipped；与本地/套件/能力测试合跑 157 passed / 1 skipped。
- 完整测试：`python -m pytest tests/ -q`，2751 passed / 1 skipped。
- Golden 串行 compare 有一次 PASS 3 / FAIL 1：LayerNorm 平均值 0.0250 ms，
  min/median 0.0155 ms、max 0.0502 ms，越过 0.0140 ms 基线的原容忍窗口。
  随后只做一次串行复测：PASS 4 / FAIL 0，LayerNorm 0.0155 ms；
  在临时目录运行已提交的 `8f57074` 作对照：PASS 4 / FAIL 0，LayerNorm 0.0157 ms。
  该异常未稳定复现，数据提示小样本 CPU 计时波动，**不等于证明性能永不回归**。
  未修改基线、阈值或 benchmark 计时路径，也不把前一次失败抹成通过。
- 七个关键变异在临时副本中全部被捕获：偷偷 contiguous（4 项红）、精度参数隐式转输入（1）、
  控制端丢弃回读（2）、执行后回读（1）、输出先转 fp32 再算指纹（1）、
  错误参考当错核而非 unknown（2）、CPU 回退冒充真实设备（1）。
  这是七个关键门禁负例，不是完整变异覆盖率声明。

本机只有 CPU：真实 CUDA/Triton 用例显式 skip，不以 CPU 测试替代真机。
下一验收应在真实 NVIDIA 执行侧运行该用例并留设备与回读证据，然后补自动边界生成/覆盖 lint。
