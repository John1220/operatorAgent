# 本地 Python 输入保真协议

日期：2026-10-08。目标：让 CorrectnessSuite 声明的输入与候选实际收到的输入可逐项对账。
不是敌意候选沙箱，也不是 aliasing/in-place 语义验收器。

## 路径

`MockGPU._execute_python` 无条件请求 `run_candidate(preserve_inputs=True)`。
父侧 `input_fidelity.encode_inputs` → pickle 传输协议载荷 → 子侧 `decode_inputs` 重建视图 →
**候选源码 exec 前** `describe_inputs` 回读 → 数值结果和回读证据一并返回。
fork 和 subprocess 共用同一段子侧程序。

裸 ndarray pickle 会把逻辑元素打包，丢失 stride、offset 和共享 storage。因此协议不直接
pickle ndarray，而是发送 storage 字节和 view 元数据。通用 exec_guard 调用仍可走旧路径；
本地 MockGPU Python 不提供环境变量关闭这条协议的入口。

## `numpy-storage-v1` 载荷

```text
protocol: numpy-storage-v1
storages: [bytes, ...]          # 同一 backing storage 只传一次，物理字节顺序
tensors:
  <input name>:
    shape: [int, ...]
    dtype: NumPy dtype.str      # 含字节序，不悄悄变成 fp32
    strides: [int, ...]         # 字节 stride，允许负数和 0
    offset: int                # 相对最终 ndarray backing root 的字节偏移
    storage: int               # storage 索引
    writeable: bool
scalars: {<input name>: value}  # 原类型/原值，含 np.float32 和 causal=False
```

子侧每块 storage 建一份 u8 backing buffer，再以显式 shape/dtype/offset/strides 构造 ndarray。
覆盖 contiguous、切片、转置、Fortran、负 stride、零 stride broadcast、只读视图、0D 和空数组。
同 storage 的输入重建在同一块 buffer 上；这保留传输层共享关系，**不代表算子的 aliasing
语义已通过验收**。父侧输入不会因子侧候选写入而改变。

仅支持普通、无对象的数值 ndarray 与简单标量。ndarray subclass、object/结构化 dtype、
stride_tricks 的未知 backing 来源等明确拒绝，不把它们偷偷转成 contiguous。

边界检查：dtype、shape/stride/offset 类型、storage 索引、负 stride 的最低/最高地址、
逻辑 tensor 字节数和 storage 总字节数。字节上限为 256 MiB；这是协议的输入界限，
不是整个进程的内存上限。越界 view 在取字节/重建前拒绝。

## 回读与门禁

子侧对**重建后的实际对象**计算：

- shape / dtype / strides / offset / storage 分组 / writeable；
- backing storage 字节数和完整 SHA-256；
- 逻辑元素值 SHA-256（C 顺序，包含非有限值的实际位模式）；
- 标量类型和原值指纹。

控制端在调用后端**之前**保存预期输入描述，再与返回的 `input_fidelity` 比较。
提供方的 `correctness_input_support` 只是能力声明，不能替代这次执行的证据。

`input_fidelity_verified` 三态：True=严格一致，False=不一致，None=没证据/协议不兼容。
比较同时核字段类型，`True` 不能冒充整数 1。后两者均令套件 unknown、不进入正式 L3。
正确输出本身不能证明候选收到了正确布局。

证据进入逐 case L2 metrics、`verification.json`、重复确定性检查和报告；报告打印执行侧
dtype/shape/strides/offset/只读属性及值指纹。完整预期/回读和 storage 哈希保留在 JSON 中。

参考侧：内建 Torch 参考遇到只读或负 stride 时，只在参考侧按逻辑值复制，以满足
`torch.from_numpy` 限制；不复制候选输入。外挂 reference_source 的语义不在这里偷偷改写。

## 验收与剩余边界

守卫：`tests/test_input_fidelity.py`。包括真实 fork/subprocess 回读、offset/负 stride、
共享 storage、readonly、错误描述拒绝、预期值提前快照、候选改输入时仍报告调用前证据、
缺证据/证据变更 fail-closed，以及正确/错误 stride 和 broadcast 核的 L2→L3 对照。

2026-10-08 验收结果：

- 输入保真守卫 36 项；与套件守卫合跑 83 项全部通过。
- 完整测试：`python -m pytest tests/ -q`，2698 passed。
- 完整测试结束后串行 Golden compare：PASS 4 / REVIEW 0 / FAIL 0 / SKIP 0；未改基线。
- 四个关键变异均被定向测试捕获：重建后偷偷 contiguous、忽略回读不一致、
  缺证据冒充已验证、改成候选执行后才回读。分别触发 4、5、1、1 项失败。
  这是四个关键负例的验收，不是完整变异覆盖率声明。

本地第二阶段不涉及 SSH/HTTP/Triton/CUDA；同日后续的 NVIDIA L2 / Torch 实现见
[远端输入保真](remote-input-fidelity-protocol.md)，真机验收仍单列，不由本地结果代替。
GPU 上实际 tensor 的 stride、dtype、storage_offset 必须由 GPU runner 自己回报。
backward、aliasing/in-place 语义、sanitizer、GPU launch 路径检查仍是独立欠账。
本协议不解决同 uid 文件篡改、恶意反射修改子进程报告、pickle 的安全性或 OS 隔离。
