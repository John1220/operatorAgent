# CorrectnessSuite：实现与边界（含本地与 NVIDIA L2 输入保真）

日期：2026-10-08。对应主设计 §20；实现状态以测试为准，不宣称 §20 全部完成。
本地输入保真协议见 [input-fidelity-protocol](input-fidelity-protocol.md)。
NVIDIA L2 / Torch 协议和真机验收边界见 [remote-input-fidelity-protocol](remote-input-fidelity-protocol.md)。

## 接入路径

`ConstraintSpec.correctness_suite` → `freeze_from_task` → `FrozenContract` 内容哈希。
`TaskSchema.to_dict()` 保存完整套件声明，旧任务未声明时不写这个字段；旧契约哈希不变。

`Verifier.verify()` / 候选选择使用的 `stop_after=L2` 都在 L2 执行套件。
调用方原始输入仍要通过，不能替代套件。套件数组不传入 L3，不改变性能 case 的权重、
身份或目标函数。失败和无法确立均令 `L2.passed=False`，不进入正式 L3 比较或晋级。

## 数据模型

- `CorrectnessCase(case_id, inputs, pattern, layout)`：`inputs` 复用 `BenchmarkCase` / `TensorSpec`。
- `CorrectnessSuite(cases, deterministic=True, required_checks=())`：冻结对象，也接受 JSON 声明。
- `case_id` 唯一；同 shape 的不同 dtype、seed、值分布或布局不去重。
- 套件内容、输入形状/dtype/seed、值模式、布局、确定性要求和额外检查全部进契约 hash。
- 值模式：random / zeros / constant（1）/ large（100）/ small（1e-4）/ nan / inf。
  后两者能构造输入，但当前参考一旦非有限只能报 unknown，不算已验证了 NaN/Inf 语义。
  large/small 是固定尺度压力样本，**不是**覆盖整个 dtype 值域的保证。
- 布局模式：contiguous / strided（最后一维步长 2）/ broadcast（最后一维零 stride）。
  **能在控制端构造不等于后端能保真执行**。

声明示例（此处 shape 仅示范，不是自动满足全部边界类别的默认套件）：

```python
from operator_agent import CorrectnessCase, CorrectnessSuite, ConstraintSpec
from operator_agent.bench_cases import BenchmarkCase, TensorSpec

suite = CorrectnessSuite(cases=(
    CorrectnessCase("tail-fp32", BenchmarkCase(tensors=(
        TensorSpec("A", (7, 5), "fp32"),
        TensorSpec("B", (5, 9), "fp32"),
    ), seed=1)),
    CorrectnessCase("zero-fp16", BenchmarkCase(tensors=(
        TensorSpec("A", (1, 5), "fp16"),
        TensorSpec("B", (5, 9), "fp16"),
    ), seed=2), pattern="zeros"),
))
constraints = ConstraintSpec(correctness_suite=suite)
# 可序列化的声明：suite.content_dict()
```

## 三态和证据

`L2.metrics["correctness"]` 记录 passed / failed / unknown、套件哈希和逐 case 结果。
失败 case 带原始数值读数（包括首个错误位置）；未执行项明确记 `not_run`。
没有参考、非有限参考、空/不完整输出比对、缺少保真输入协议、静态守卫不适用，均不能通过。
fp16/fp32 case 按自身声明选择参考和分精度容差；不支持的 dtype 不自动降级。
确定性检查重新生成同一输入，重复 L2，并比较**原始输出 dtype/shape/bytes** 指纹，
不使用 float32 比对后的数组指纹，以免精度转换抹掉微小差异。

unknown 经候选凭据和入库正确性判据保持为 `None`；失败路由选择 HOLD，
不把覆盖缺口当候选 bug 交给 Fixer 或沉淀错误教训。
报告直接读取 L2 证据，即使没有任何 L3 读数也打印套件和无法确立的检查。

## 目前可确立的范围

MockGPU 的 **Python 数值 ndarray** 路径能保留输入值、dtype、strides、offset 和只读属性，
并由执行侧在候选 exec 前回读；已验证 contiguous / strided / broadcast 输入及多 shape、
非对齐尾块、tiny/batch=1、fp16/fp32、随机/零/常数/固定大小尺度、两次输出一致性。
这证明的是声明的有限样本，不是所有 shape、所有语义上的正确性证明。

## 仍未实现，不能宣称完成

1. **复杂 backing 来源**：本地普通数值 ndarray 的 storage/view 传输已接通；
   ndarray subclass、object dtype、stride_tricks 的未知 backing 来源等仍拒绝，不猜指针。
   回读证据缺失或不一致时仍是 unknown，不能只靠后端声明或正确的输出数值放行。
2. **远端套件**：NVIDIA SSH L2 原值/storage 协议及真实 Torch 回读已接通，离线测试通过；
   GPU 真机验收尚未完成，XPU/HTTP 未接通。`Backend.correctness_input_support` 默认拒绝，
   肯定声明也必须核本次执行证据。L3 的旧生成协议不在本片改造范围。
3. **Triton 路径**：本地真实 CUDA 与 NVIDIA runner 使用 Torch storage/view 回读；
   readonly/负 stride 不能表达，仍 unknown；无 CUDA 的模拟参考仍不能证明原候选正确性。
4. backward / sanitizer / racecheck / aliasing / in-place / 指定非 fp32 累加精度，
   若写进 `required_checks`，当前全部 unknown。没有执行器，不补假通过。
5. 没有自动推导每类边界的生成器，也没有“每类边界各一”的覆盖 lint。
   shape 仍由调用方明确声明；未声明 CorrectnessSuite 的旧任务仍走原 L2，
   **本轮没有把全部旧调用方迁移成强制套件模式**。
6. 两次独立执行的指纹一致不是任意次数、同进程连续 launch 的确定性证明。
   完整 launch/fallback 探针仍依赖已有静态/运行时盾，未新增 GPU launch 审计器。

## 验收

新增 47 条守卫；在临时副本中做了四种关键变异：跳过套件（4 条红）、unknown 放行（2 条红）、
套件不进契约 hash（3 条红）、先转 float32 再算输出指纹（1 条红）；4/4 被测试拒绝。
这不是完整变异覆盖率，只是对四处承重门的负向检查。

第一阶段回归：全量 `2662 passed`；Golden compare 串行运行 `PASS 4 / REVIEW 0 / FAIL 0`。

`tests/test_correctness_suite.py`：契约往返/篡改拒绝、同 shape 不同 dtype、输入可复算、
shape/尾块/dtype/零值特化错核止步 L2、确定性失败、缺参考/部分非有限/能力缺口、
unknown 不污染教训、L2 报告、Conductor 候选选择和产物落盘。

本地第二阶段守卫：`tests/test_input_fidelity.py`（fork/subprocess、值与布局回读、
正确/错误 stride 和 broadcast 核对照）；原有布局缺口测试改成“缺回读证据仍 unknown”。

第三阶段守卫：`tests/test_remote_input_fidelity.py`，原值 JSON、独立部署 runner、Torch
实际对象回读、确定性和 L2→L3 对照；真实 CUDA/Triton 用例需要相应设备，否则明确 skip。
下一阶段先补 GPU 真机验收，再补自动边界生成和覆盖 lint；不能把 CPU 协议测试当成 GPU 证据。
