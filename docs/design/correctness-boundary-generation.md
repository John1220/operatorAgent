# 自动边界生成与覆盖 lint（第一版）

日期：2026-10-08。实现：`operator_agent/correctness_boundaries.py`。
对应 §20 / §3.4 的有限边界样本，不宣称完整 §20 已完成。
首批默认入口与多算子 H20 链路验收见 [可信默认配置](default-correctness-profiles.md)。

## 目的与信任边界

“已有 case 全部通过”与“规定的边界类别齐全”不是一件事。本片补齐测试集合的
生产者与检查者，但不替代既有 L2 数值比对、输入保真、确定性或实际设备证据。

可信 harness **显式声明**可用于 L2 采样的合法正整数轴范围、anchor、对齐尺度、
输入张量的轴绑定、支持 dtype 与布局。范围可以是算子合法域的一个子域；调用方主输入
仍独立验证，性能 workload 不由此替换。**不从候选源码/BLOCK 参数猜域，不从 L3 shape
列表擅自推断新的输入合法性，不由候选挑选/删改测试**。

第一版域模型是正整数矩形盒 + 共享轴相等绑定。额外整除条件、离散合法 shape、
复杂参数关系不能自动表达；harness 必须先判断该声明是否确实是合法子域。
没有这样的声明时，不猜 attention/conv 等输入语义，也不默认迁移旧任务。
绑定必须列全输入张量；本版不自动生成必需的标量参数，带这类参数的算子仍需可信
harness 提供合适的输入/参考适配，不能只声明张量 shape 就声称完整支持。

## 接口与声明

```python
from operator_agent import (
    BoundaryAxis, TensorAxes, BoundaryPolicy, ConstraintSpec,
    generate_correctness_suite, lint_coverage,
)

# matmul: A[M,K], B[K,N]；共享 K 的一致性由绑定检查，而不是 case 名字。
policy = BoundaryPolicy(
    axes=tuple(BoundaryAxis(a, minimum=1, maximum=5, anchor=4, alignment=4)
               for a in ("m", "n", "k")),
    tensors=(TensorAxes("A", ("m", "k")), TensorAxes("B", ("k", "n"))),
    dtypes=("fp16", "fp32"),
    patterns=("random", "zeros", "constant", "large", "small"),
    layouts=("contiguous", "strided"),
    seed=17,
    max_cases=128,
    max_input_bytes=16 * 1024 * 1024,
)

# 两种接法，得到同一份规范化 suite / hash：
suite = generate_correctness_suite(policy)    # 立即生成声明，不分配 ndarray
assert lint_coverage(suite)["status"] == "passed"
constraints = ConstraintSpec(correctness_suite=suite)

# 延迟到现有任务/契约/验证入口展开；不是一份没有消费者的独立 plan。
constraints = ConstraintSpec(correctness_suite={"coverage_policy": policy.content_dict()})
```

`CorrectnessSuite.coverage_policy` 可附到手工套件上。手工 case **不会被自动补齐**；
lint 缺项就拒绝。“省略 cases + 有 policy”表示自动生成；显式 `cases=[]` / `None`
仍是错误，不替调用者修成一份通过的套件。

`as_suite`、`TaskSchema.to_dict`、`freeze_from_task` 共用展开逻辑。任务序列化保存展开后的
完整 case 声明和 policy；生成过程不修改原任务。轴/输入声明顺序规范化，但每个张量的
轴顺序有语义，不能随便排序。默认输入 dtype 只支持 fp16/fp32，不伪造 bf16/fp64 转型。

## 固定规则 `axis-boundary-v1`

在显式域内，选择以下有限 shape：

- anchor，以及可构造的全轴对齐点。
- 每个轴单独取 minimum / maximum，其他轴保持 anchor。
- 允许 1 时，每轴单独取 1（batch=1/tiny 是否有该语义，由声明者负责）。
- 每轴最靠近 anchor 的域内对齐点上下邻点 `-1 / +1`；等距时选较小对齐点。
- 每轴一个合法非对齐点，作为 tail 见证；不靠名字“tail”判断。

没有合法的对齐点、邻点或 1 时，报告逐项 `inapplicable` 和原因，不越界造样本。
这不同于“本应存在而被漏掉”。例如域 `[3,5]`、alignment=8 没有合法对齐点，但仍有 tail。
“maximum”表示声明子域的上界，不保证是硬件意义上的极大输入。

所有 shape 见证必须在**每个声明 dtype**下用随机连续输入测试，不能拿零值 case 代替。
另外，每个 dtype 都有每种声明值模式与布局的见证。布局扰动必须实际可构造（现有 factory
要求所有输入最后一维 >1）；anchor 太小时选其他合法边界点，没有合适点就拒绝，不删要求。

不做 shape × dtype × pattern × layout 的全量笛卡尔积：覆盖的是上述固定见证，
不是全部组合。相同内容（含 dtype、seed、pattern、layout）只生成一次，不按 L3 形状 key
合并不同精度/值/布局。变化生成算法须升级 recipe_version，不让旧契约悄悄换尺子。

生成与 lint **都不分配输入数组**。实际值仍由可信 `CorrectnessCase.materialize()` 生成，
并走已有执行侧回读。NumPy/执行环境变化仍需环境留痕，不能仅凭 seed 忽略版本漂移。

## 预算与 lint

`max_cases` 与每组 `max_input_bytes` 是显式拒绝条件：不缩小 maximum、不少测 dtype、
不裁掉 tail、不悄悄停在前 N 项。输入字节估算包含 strided backing 的双倍存储；
它不是整个 Python 进程、随机生成临时数组、输出或 GPU workspace 的统一内存配额。

lint 根据输入名、rank、共享轴尺寸、实际声明 shape/dtype/pattern/layout 判见证；
重复命名/增加数量/把对齐样本改名“tail”不能补缺项。越域、混合 dtype、不一致 K、
不可构造布局及超预算均报错。随机基准和 contiguous 必须显式保留。

结果区分 `passed / missing / invalid / not_declared`，含 policy_hash、recipe_version、
逐规则 case_id 见证、missing、errors、inapplicable 与范围说明。
未声明 policy 的旧套件是 `not_declared`，不是 lint passed；旧内容和契约 hash 不变。

## 生产硬门禁与报告

```text
可信 policy / 手工 suite
    → 确定性展开与 coverage lint
    → 完整声明冻结进契约 hash
    → L1 / L2 原值回读、数值与确定性
    → 仅全部通过后进入 L3
```

- 契约冻结拒绝 missing/invalid；改变 policy、范围、alignment、seed 或预算会改变 hash。
- `Verifier.correctness_preflight` 是纯声明检查：覆盖缺口在编译/验证器硬件探测前产生
  **L2 unknown**，不调用执行器或性能 oracle。私有 L2 入口也保留该门。
- Conductor 在路由/LLM/环境 provisioning 前做同一预检查；不要求注入的旧 scripted verifier
  增加新方法。前置拒绝与现有入口错误一致，直接返回无候选的 artifact，不走正式候选保存
  流程；其 L2 结果可直接交 `build_report`，缺口不能被 Fixer 或失败语料库当成候选 bug。
- 原有 credential / HOLD / 晋级链保留 unknown。通过 lint **不意味着数值通过**；报告与
  `L2.metrics["correctness"]["coverage_lint"]` 同源，明确说明声明见证不是运行证据。
- 自动套件不进 `constraints.cases/shapes`、L3 case 权重或性能目标；性能输入还是原 workload。

## 验收与边界

守卫：`tests/test_correctness_boundaries.py`，包含独立的几何点预期、严格声明/预算、
手工冒名与漏精度、契约往返/篡改、缺项不编译/不调用 L3、HOLD、不沉淀候选教训、
Conductor 自动展开/保存有效结果、正确与错误 shape/dtype/stride 核、非有限参考及
只读 broadcast 协议缺口。离线 Torch 测试只声明 REMOTE_VERIFY，不能冒充 REAL_DEVICE。

本地验证（2026-10-08，Python 3.9.12 / Torch 2.8.0 CPU）：

- 新边界守卫 **59 项**；边界、H20 驱动与 scratch 联合测试 **104 passed**。
- 全量 pytest **2837 passed / 1 skipped / 2 warnings**。跳过项需本机 CUDA；两条 warning
  来自刻意制造非有限参考的 oracle 差值，相关测试确认结果保持 unknown、不得进入 L3。
- 全量结束后串行 Golden compare：**PASS 4 / REVIEW 0 / FAIL 0 / SKIP 0**。
  这是原有 MockGPU 回归窗口，不是新增真实设备性能证据；未改基线、容忍窗口或计时代码。

六个承重变异在临时副本中均被捕获：跳过 Verifier 预检查、跳过 Conductor 预检查、
policy 不进 hash、邻点错成 ±2、零值冒充随机 shape 见证、冻结不做覆盖 lint。
这是六个关键门禁的负向标定，不是完整变异覆盖率。

真机入口：`python3 benchmarks/input_fidelity_h20.py --boundaries --gpu 2`。
2026-10-08 H20 GPU 2 / 实际 cuda:0（Torch 2.14.0+cu130、Triton 3.8.0）：

- 新声明：rows=[1,9]、anchor=8、alignment=8；cols=[1,33]、anchor=8、alignment=32；
  X[rows,cols]，fp16/fp32、五种值模式、contiguous/strided。
- **26 个自动 case、42 项覆盖规则**；正确 Triton ReLU 全部数值、原值回读及两次指纹通过。
- 忽略 stride 的核对齐主输入通过、两个精度的 stride case failed；错误尾块核的 12 个
  case failed（含 fp16/fp32）。二者 lint 都 passed，但都未进入 L3。
- 加上之前 11 项协议验收，共 **14/14 符合预期**（6 passed、6 failed、2 unknown）；
  185 次 verify、183 份实际 CUDA 回读、L3 尝试/benchmark 请求均为 0，清理已读回。
- 私有产物编号 `input-fidelity-h20-20261008T080510885334Z-39571-1`，位于 ignored
  `output/input_fidelity_h20/<编号>/report/summary.json`；不提交真实主机/用户路径。

**仍未覆盖**：全部旧任务强制声明覆盖规则；任意合法域/全部组合；
完整 NaN/Inf 语义、累加精度、backward、aliasing/in-place、sanitizer、GPU launch 探针；
自动 offset/转置/负 stride 用例。Torch readonly broadcast 仍 unknown。
大值=100 的 fp16 GEMM（例如 K=8 得到 80,000）参考会溢出，仍保留该 case 和 unknown，
不能因为覆盖 lint 通过就把非有限参考变成通过；示例 K=4 的大值样本是有限的对照。
