# 无人值守算子开发 Agent —— 优化方向与实施路线

| 项 | 值 |
|---|---|
| 文档状态 | Draft v1 |
| 日期 | 2026-10-03 |
| 适用对象 | `autonomous-operator-agent-design.md` 及其 Stage 0–5 实现 |
| 目标 | 在不破坏现有“可信评测脊柱”的前提下，补齐正确性、Agent Runtime、执行隔离、统计判决和产品化边界 |
| 基本立场 | **先修边界和不变量，再增加模型能力；先让系统不会自信地犯错，再让它更会搜索** |

---

## 0. 摘要

当前设计已经形成了一个较强的“可信算子评测与搜索脊柱”：契约冻结、同卡配对、资格审计、标定心跳、候选血缘、失败分类、预算和“无法确立”报告，都是正确的基础。

下一阶段不应继续简单增加 Agent 角色或 Prompt，而应围绕以下五个方向收敛：

1. **规范、实现状态和研究日志分离**，让文档重新成为可靠的设计合同；
2. **把正确性提升到与性能可信度同等的地位**，建立完整的算子语义与错误候选门禁；
3. **把测量判决形式化**，处理自适应搜索、噪声、选择偏差和多 shape 目标；
4. **补齐 Agent Runtime 与候选执行沙箱**，让 LLM 真正成为受约束的提议器，而不是隐式控制器；
5. **补齐持久化、调度和多卡运行时**，让系统可以长期无人值守、可恢复、可复现。

目标状态不是“LLM 写出更多 Kernel”，而是：

```text
需求 / 参考实现
    ↓
冻结的算子契约
    ↓
可追踪的设计与候选
    ↓
隔离构建与运行
    ↓
正确性门禁
    ↓
资格审计与成对测量
    ↓
统计判决与受约束搜索
    ↓
确认、晋级、报告和长期记忆
```

---

## 1. 当前判断

### 1.1 已经形成的优势

当前设计中应当保留、不应因追求简化而删除的核心原则：

- 评测器是唯一的结果收口点；
- 契约在运行前冻结并哈希化；
- Agent 不能修改容差、基线和目标指标；
- 基线和候选必须同卡配对测量；
- 低资格结果不能产生永久后果；
- 基础设施失败不计入优化进度；
- 性能不达标优先进入搜索，而不是盲目修复代码；
- 数值调参主要交给模板、搜索和 bandit，LLM 负责结构性变化；
- best-known-good、实验日志和“无法确立”章节必须同时交付；
- 所有候选都必须保留父代和可复现血缘。

这些原则构成系统的“信任边界”，优化时只能增强，不能弱化。

### 1.2 当前主要短板

| 优先级 | 短板 | 后果 |
|---|---|---|
| P0 | 规范和实现状态混在同一文档 | 无法判断哪些是合同、哪些是当前事实、哪些是历史修复记录 |
| P0 | 部分安全策略仍是 opt-in | 无人值守模式可能默认跳过二次确认、预算或部分审计 |
| P0 | roofline 收敛与参考 Kernel 下界混用 | 可能把“接近参考实现”误报成“接近硬件峰值” |
| P0 | 正确性轴弱于性能轴 | 性能测量很严格，但错误 Kernel、特化 Kernel、框架语义不完整仍可能漏过搜索门禁 |
| P1 | LLM Agent Runtime 只有原则、缺少正式协议 | Stage 4 落地时容易退化为不可控的长 Prompt + 自由改代码 |
| P1 | 统计判决缺少完整定义 | 自适应搜索可能把噪声最大的候选挑成 winner |
| P1 | 候选执行隔离未形成完整安全边界 | 静态守卫无法覆盖宏、头文件、动态加载和运行时作弊 |
| P1 | 事件日志、恢复和 exactly-once 语义不完整 | 崩溃恢复后可能重复扣预算、重复学习或覆盖状态 |
| P2 | 多卡当前主要是身份和配对约束，尚未形成调度系统 | 多卡并行、租约、抢占、故障转移仍是接口承诺 |
| P2 | 长期记忆的迁移、过期、撤销和选择偏差没有充分建模 | 错误经验可能通过 bandit 或跨任务复用放大 |

---

## 2. 目标架构

下一阶段建议将系统分成五个平面，而不是继续把所有能力都放进 Oracle 或 Conductor：

```text
┌─────────────────────────────────────────────────────────────┐
│                    Control Plane 控制平面                    │
│  Task Intake · Contract · Policy · Planner · Agent Runtime   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Candidate / Action
┌──────────────────────────────▼──────────────────────────────┐
│                   Execution Plane 执行平面                    │
│ Materializer · Build Sandbox · Runner · Device Scheduler    │
└──────────────────────────────┬──────────────────────────────┘
                               │ Artifact / Raw Observation
┌──────────────────────────────▼──────────────────────────────┐
│                     Trust Plane 信任平面                      │
│ Correctness Oracle · Qualification · Pairing · Adjudicator  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Claims / Decisions
┌──────────────────────────────▼──────────────────────────────┐
│                      Data Plane 数据平面                      │
│ Event Log · Artifact Store · Measurement DB · Corpus        │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                 Operations Plane 运维平面                     │
│ Auth · Quota · Health · Trace · Alert · Audit · Cleanup      │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 平面之间的硬边界

| 边界 | 规则 |
|---|---|
| Agent → Execution | Agent 只能提交结构化 Candidate / Patch，不能直接控制 benchmark 进程 |
| Candidate → Harness | Candidate 不能修改 harness、baseline、contract、日志和评分代码 |
| Execution → Trust | Trust Plane 只消费带完整 provenance 的 artifact 和 observation |
| Trust → Corpus | 只有通过 promotion policy 的 claim 才能进入长期记忆 |
| Data → Report | 报告中的数字只能来自不可变事件和已收口的 Observation |
| Operations → All | 所有任务、设备、模型调用和 artifact 都必须可审计 |

---

## 3. 优化原则

### 3.1 安全不变量默认开启

以下能力不应再作为普通运行参数提供关闭开关：

```text
contract hash 校验
候选 / 基线来源校验
lib fallback 检查
同卡成对测量
心跳和环境漂移检查
候选不能修改 harness
长期记忆的独立确认
不可变实验日志
```

可以配置的是预算和搜索强度，而不是安全语义：

```text
允许配置：
- 最大 wall clock
- 最大 GPU 秒
- 最大 oracle 次数
- 最大 LLM token
- 探索 / 精修比例
- benchmark 样本数
- 最小效应量

不允许 Agent 自行修改：
- 容差
- 基线实现
- 资格判据
- promotion policy
- 任务语义
```

### 3.2 先定义 Claim，再定义布尔资格位

现有 `can_*` 位继续保留，但增加一个面向产品和报告的声明等级：

```python
class ClaimLevel(StrEnum):
    UNOBSERVED = "unobserved"
    OBSERVED = "observed"
    PROVISIONAL = "provisional"
    VALIDATED = "validated"
    PROMOTABLE = "promotable"
    PERSISTED = "persisted"
```

推荐语义：

| 等级 | 含义 | 允许的后果 |
|---|---|---|
| `UNOBSERVED` | 没有可用测量 | 不能判决、不能学习 |
| `OBSERVED` | 有原始观测，但资格不足 | 只能诊断和继续探索 |
| `PROVISIONAL` | 当前运行内可比较 | 可用于局部搜索，不能晋级长期 best |
| `VALIDATED` | 通过正确性和测量门禁 | 可以参与正式候选比较 |
| `PROMOTABLE` | 通过独立确认 | 可以替换运行级 best |
| `PERSISTED` | 进入长期知识库 | 可跨任务复用，但必须带分区与 TTL |

这样可以避免把 `can_tune=True` 同时解释为“允许继续试探”和“允许基于该结果改变策略”。

### 3.3 设计、实现、证据必须分开

规范文档只描述：

```text
接口
不变量
状态机
判决规则
数据 schema
验收标准
```

实现状态单独维护：

```text
已实现
已单测
已真机
已端到端
已生产验证
```

研究日志单独维护：

```text
为什么改变设计
哪次实验推翻了什么
外部论文和实验条件
历史误判和修正
```

---

## 4. 方向一：补齐算子契约和正确性轴

### 4.1 扩展 `OperatorContract`

算子契约不应只描述性能目标，还应描述完整可观察语义：

```python
@dataclass(frozen=True)
class OperatorContract:
    # 语义
    operator_name: str
    formula: str | None
    reference_impl: ArtifactRef
    numerical_semantics: NumericalSemantics
    error_policy: ErrorPolicy

    # 输入输出
    tensors: tuple[TensorSpec, ...]
    supported_dtypes: tuple[DType, ...]
    supported_layouts: tuple[Layout, ...]
    supported_strides: StridePolicy
    alignment: AlignmentPolicy
    aliasing: AliasingPolicy
    dynamic_shape_policy: ShapePolicy

    # 正确性
    correctness_suite: CorrectnessSuite
    tolerance_policy: TolerancePolicy
    deterministic: bool
    backward_policy: BackwardPolicy

    # 性能
    workload_profile: WorkloadProfile
    objective: PerformanceObjective
    regression_limits: RegressionLimits

    # 工程和环境
    backend_requirements: BackendRequirements
    topology: TopologySpec | None
    contract_hash: str
```

### 4.2 正确性测试套件必须成为 Stage 门禁

当前 Stage 1 的验收重点偏向测量器本身。下一阶段应将正确性纳入同一个硬门禁，至少覆盖：

```text
基础随机输入
边界 shape
非对齐尾块
batch=1 / 极大 shape / 极小 shape
不同 dtype 和累加精度
非 contiguous stride
broadcast
零、常数、极大值、极小值
NaN / Inf
确定性
aliasing / in-place
backward / gradient（如果契约要求）
sanitzer / racecheck
```

验收中必须有明确的错误候选：

```text
数值整体缩放
尾块越界
只对特定 shape 正确
忽略 stride
错误 broadcast
静默 fallback
```

目标不是测试数量多，而是确保：

> **错误候选不能进入正式性能比较，更不能替换 best。**

### 4.3 增加 `USE_CUSTOM / USE_LIBRARY / INCONCLUSIVE` 裁决

对外部报告而言，只有“候选比 baseline 快”是不够的。还需要判断最终应该使用哪条路径：

```python
class ImplementationDecision(StrEnum):
    USE_CUSTOM = "use_custom"
    USE_LIBRARY = "use_library"
    INCONCLUSIVE = "inconclusive"
```

示例：

```text
自定义实现正确且在目标 workload 上稳定快       → USE_CUSTOM
自定义实现正确但稳定慢或维护成本不划算           → USE_LIBRARY
测量资格不足、结果冲突、环境不稳定               → INCONCLUSIVE
```

这样可以避免“自定义 Kernel 能跑”被误报成“应该采用自定义 Kernel”。

### 4.4 增加特化和空转探针

防止候选只在 benchmark 中的几个 shape 上工作，而对其他输入退化或不执行真正路径：

```text
输入 shape 扰动
输入值扰动
layout / stride 扰动
候选输出扰动检测
kernel launch 路径检查
库函数 fallback 检查
```

探针结果应进入报告，而不是只作为内部 debug 信息。

---

## 5. 方向二：形式化测量和判决

### 5.1 重新命名峰值概念

将以下三个概念分开：

```python
peak_trusted:
    规格或文档口径经过检查

peak_measured:
    真正的硬件峰值 microbenchmark 经过实测

reference_lower_bound:
    已知参考 Kernel 在当前 workload 上的实测下界
```

`reference_lower_bound` 可以用于：

- 判断候选是否接近当前参考实现；
- 生成 roofline-like 诊断；
- 作为搜索排序特征。

但只有 `peak_measured` 才能打开真正的硬件峰值停止条件。

建议将：

```text
roofline_converged
```

改成：

```text
reference_ceiling_reached
```

除非分母确实来自独立、compute-bound、按精度标定的硬件峰值测试。

### 5.2 四态判决必须有数学定义

建议以成对差值作为基本统计量：

```text
Δ_i = candidate_i - baseline_i
```

然后基于稳健统计和置信区间判决：

```text
FAST:
    upper_confidence_bound(Δ) < -minimum_effect

SLOW:
    lower_confidence_bound(Δ) > +minimum_effect

SAME:
    confidence_interval(Δ) 完全落在 noise_band 内

AMBIGUOUS:
    其余情况
```

需要在契约或测量策略中固定：

```text
最小效应量
置信水平
最小样本数
最大样本数
warmup 策略
异常值处理
配对顺序
随机化种子
多重比较策略
```

### 5.3 处理自适应搜索的选择偏差

Bandit 直接从“是否击败当前 best”学习，会受到以下影响：

```text
当前 best 不断变化
候选测量次数不一致
噪声大的候选更容易偶然胜出
搜索过程本身选择了最终样本
```

建议分成三层：

```text
探索层：低成本、弱判决、允许不确定
比较层：同卡配对、固定样本和统计判决
晋级层：独立样本、独立窗口、严格确认
```

`best` 只能由晋级层改写，Bandit 可以使用探索层信号，但不能把探索层信号直接当作永久事实。

### 5.4 多 shape 目标必须正式进入评分器

性能目标应支持：

```python
@dataclass(frozen=True)
class PerformanceObjective:
    cases: tuple[BenchmarkCase, ...]
    weights: tuple[float, ...]
    aggregate: Literal["geomean", "weighted_geomean", "max_regret"]
    required_cases: tuple[str, ...]
    max_regression: Mapping[str, float]
```

一个候选不能因为在一个 shape 上快很多，就掩盖关键 shape 上的严重回退。

建议同时使用：

```text
加权总体目标
关键 shape 硬回退约束
P50 / P95 稳定性约束
显存和编译时间约束
```

---

## 6. 方向三：建立正式的 Agent Runtime

### 6.1 Agent 只提交结构化动作

Agent 不应直接操作 Conductor 的内部状态，而应提交：

```python
@dataclass(frozen=True)
class AgentAction:
    action_id: str
    parent_candidate: CandidateID | None
    kind: Literal[
        "propose_template",
        "propose_source_patch",
        "repair_compile",
        "repair_correctness",
        "request_profiler",
        "request_new_shape",
        "explain_failure",
    ]
    patch: Patch | None
    rationale: str
    expected_bottleneck: Bottleneck | None
    requested_tools: tuple[str, ...]
    token_budget: int
```

控制器只接受通过 schema、权限和预算校验的 Action。

### 6.2 结构性变化和局部修复分开

建议保留现有的 `fix_not_rewrite` 思路，并进一步形式化：

```text
编译错误：局部修复优先
数值错误：局部修复优先，并要求误差形态诊断
边界错误：局部修复或回退父代
性能不达标：进入搜索，不默认改代码
算法瓶颈：允许结构性变更
接口 / ABI 破坏：拒绝候选，回退父代
```

在“数值错误”中增加误差形态：

```text
tail_block
whole_row
whole_tensor
broadcast_axis
precision_accumulation
nonfinite
```

误差形态比单个 `first_bad_index` 更适合决定“修复还是重写”。

### 6.3 Prompt 上下文只使用结构化证据

每次 LLM 调用只提供：

```text
当前契约摘要
父候选 diff
最近一次失败分类
首个错误和误差形态
相关 profiler 指标
可用工具列表
剩余预算
允许修改的文件
```

不要把完整历史日志无条件塞入上下文。

报告生成也应优先使用模板渲染：

```text
事实数字：由事件日志和 Observation 注入
结论状态：由 Adjudicator 生成
解释文字：LLM 可以润色，但不能改变事实和状态
```

### 6.4 LLM 提议必须经过独立检查

LLM 生成的候选在进入编译前依次检查：

```text
文件写入范围
patch 是否只修改授权路径
contract / harness / baseline 是否被修改
依赖是否新增
ABI 是否变化
候选是否与父代重复
是否引用不存在的能力
```

---

## 7. 方向四：候选执行和构建隔离

### 7.1 目录权限

建议将一次运行拆成：

```text
run_root/
├── contract/       # Agent 只读
├── harness/        # Agent 只读
├── baseline/       # Agent 不可见或只读 artifact
├── candidate/      # Agent 只写
├── build/          # 构建进程写
├── measurements/   # Oracle 写
└── report/         # Report Writer 写
```

候选不能写入：

```text
contract/
harness/
baseline/
measurements/
report/
```

### 7.2 进程与资源隔离

至少需要：

```text
编译超时
运行超时
GPU watchdog
CPU / memory quota
文件系统白名单
网络默认关闭
进程树清理
artifact 大小限制
设备租约超时和回收
```

### 7.3 静态检查不是唯一防线

对于 Python、Triton、CUDA C、模板实例化，静态检查能力不同。

建议把源代码审计输出明确成：

```python
@dataclass(frozen=True)
class AuditCoverage:
    language: str
    mode: Literal["ast", "typed_ast", "regex_only", "none"]
    covered_rules: frozenset[str]
    uncovered_rules: frozenset[str]
```

但是：

```text
coverage 不足只能影响 ClaimLevel 和 promotion
不能替代执行沙箱
```

---

## 8. 方向五：持久化、恢复和可复现

### 8.1 采用 append-only event log

每个重要动作都记录事件：

```python
@dataclass(frozen=True)
class Event:
    run_id: str
    event_id: str
    sequence: int
    event_type: str
    candidate_id: str | None
    payload_hash: str
    created_at: str
    parent_event_id: str | None
```

建议事件类型至少包括：

```text
RUN_CREATED
CONTRACT_FROZEN
CANDIDATE_PROPOSED
BUILD_RESERVED
BUILD_COMPLETED
CORRECTNESS_COMPLETED
MEASUREMENT_RESERVED
MEASUREMENT_COMPLETED
OBSERVATION_COMMITTED
BELIEF_UPDATED
CANDIDATE_PROMOTED
CONFIRMATION_COMPLETED
MEMORY_ADMITTED
RUN_ABORTED
REPORT_EMITTED
```

### 8.2 明确 exactly-once 语义

预算和状态更新必须能回答：

```text
这次 GPU 测量是否已经计费？
这次 observation 是否已经进入 bandit？
这次确认是否已经完成？
重启后是否允许重新执行？
重复事件是否会改变结果？
```

建议每个副作用都有幂等键：

```text
measurement_id
build_id
promotion_id
memory_admission_id
```

### 8.3 重新定义缓存键

除 `BuildKey` 和 `PartitionKey` 外，增加：

```python
@dataclass(frozen=True)
class ToolchainKey:
    compiler: str
    compiler_version: str
    runtime: str
    runtime_version: str
    headers_hash: str
    abi_hash: str

@dataclass(frozen=True)
class MeasurementProtocolKey:
    harness_hash: str
    benchmark_config_hash: str
    profiler_version: str
    warmup_policy_hash: str
    sample_policy_hash: str
```

性能测量缓存至少应绑定：

```text
候选
设备分区
工具链
harness
测量协议
契约
```

---

## 9. 方向六：多卡运行时和调度

### 9.1 三态字段必须在接口中显式表达

未观测不能被编码成 `False` 或 `0.0`：

```python
@dataclass(frozen=True)
class DeviceLease:
    device: DeviceKey
    exclusive: bool | None
    external_load: float | None
    lease_id: str
    source: str | None
    expires_at: str | None
```

### 9.2 `interference_domain` 必须有来源

来源优先级建议：

```text
硬件拓扑探测
MIG / MPS / 容器信息
时钟域和共享缓存信息
调度器配置
人工声明
unknown
```

如果来源为 `unknown`，则：

```text
允许单卡串行
不允许宣称异域并行
报告中必须写明未知
```

### 9.3 多卡能力的最小闭环

在实现真正异域并行前，至少先完成：

```text
设备发现
租约获取
租约续期
租约释放
任务取消
worker 心跳
GPU reset / worker 故障处理
同卡配对
跨节点产物汇聚
```

多卡并行应该是调度层能力，而不是只在 `DeviceKey` 上添加一个字段。

---

## 10. 方向七：记忆和跨任务迁移

### 10.1 记忆分层

建议将记忆分为：

```text
Run Memory:
    当前任务的候选、失败和信念

Partition Memory:
    某算子 / shape / 硬件分区上的有效配置

Transferable Knowledge:
    算法模式、DSL 陷阱、编译器缺陷

Negative Knowledge:
    已验证的失败签名和适用范围
```

### 10.2 记忆条目必须带有效期

每条长期记忆应包含：

```python
@dataclass(frozen=True)
class MemoryEntry:
    key: str
    claim: str
    evidence_ids: tuple[str, ...]
    partition: PartitionKey
    confirmed_count: int
    last_verified_at: str
    expires_at: str | None
    invalidated_by: tuple[str, ...]
    confidence: float
```

驱动、库、编译器、架构或 harness 变化后，相关记忆应自动进入：

```text
STALE
```

而不是继续作为强先验使用。

### 10.3 避免把搜索 winner 当成知识

只有“赢过当前 best”不能直接形成可迁移知识。

进入长期记忆至少需要：

```text
正确性通过
测量资格足够
独立重测通过
没有关键 shape 回退
硬件 / 工具链分区明确
报告中声明适用范围
```

---

## 11. 文档治理方向

建议将当前设计文档拆分如下：

```text
docs/design/autonomous-operator-agent-design.md
    规范：目标、架构、接口、不变量、验收标准

docs/design/autonomous-operator-agent-optimization-directions.md
    本文：优化方向、路线、优先级和迁移策略

docs/status/autonomous-operator-agent-status.md
    当前实现状态和真机验证状态

docs/decisions/
    ADR：关键设计决策和取舍

docs/research/
    外部证据、实验记录、CHECKLOG
```

设计正文中不应保留以下内容：

```text
“上次误判已撤回”
“今天还没有接上”
“某次 CHECKLOG 修复了什么”
“某个实现缺口后来已补齐”
```

这些内容应放入状态或决策文档，并通过链接引用。

设计正文只保留：

```text
当前规范
当前接口
当前不变量
当前验收要求
```

---

## 12. 优先级路线图

现有 Stage 0–5 继续保留。这里使用 `Wave` 表示优化工作波次，避免和已有 Stage 混淆。

### Wave 0：规范收口与安全默认值

**目标：让文档和实现不再对同一语义给出两套答案。**

任务：

- 拆分规范 / 状态 / 研究日志；
- 明确安全不变量和可配置参数；
- 二次确认、契约校验、fallback 校验在生产模式默认开启；
- 修正 `DeviceLease` 三态类型；
- 将 `Backend.instantiate()` 纳入正式协议；
- 将 `roofline_converged` 改成准确的语义名称；
- 定义 `ClaimLevel` 和 promotion policy。

退出标准：

```text
同一个问题在规范、代码、报告中只有一个语义
所有安全门都有默认值测试
所有接口字段都能表达 unknown
```

### Wave 1：正确性和候选晋级门禁

**目标：任何错误候选都不能进入正式性能晋级。**

任务：

- 扩展 OperatorContract；
- 完善 correctness suite；
- 将错误候选纳入 Stage 1 验收；
- 增加 shape / stride / dtype / nonfinite / determinism 测试；
- 增加特化和 fallback 探针；
- 增加 `USE_CUSTOM / USE_LIBRARY / INCONCLUSIVE`；
- 完成错误形态诊断。

退出标准：

```text
错误候选无法替换 best
错误候选无法进入长期记忆
关键 shape 的性能回退会被报告
无法判定时输出 INCONCLUSIVE
```

### Wave 2：统计判决和多 shape 目标

**目标：让 FAST / SAME / SLOW / AMBIGUOUS 成为可复算的统计协议。**

任务：

- 定义成对差值和置信区间；
- 定义最小效应量和 noise band；
- 将样本策略、随机种子和 warmup 固化到协议；
- 引入多 shape aggregate 和关键 shape 回退约束；
- 将探索、比较、晋级三种测量区分开；
- 处理多重比较和 winner's curse。

退出标准：

```text
相同原始样本可离线重算判决
不同搜索顺序不会改变统计协议
最终 best 必须经过独立晋级测量
```

### Wave 3：Agent Runtime 与执行沙箱

**目标：让 LLM 成为受约束的候选提议器。**

任务：

- 定义 `AgentAction`；
- 定义局部修复和结构重写边界；
- 引入候选 workspace 权限隔离；
- 编译 / 运行超时与 GPU watchdog；
- 禁止候选修改 harness、baseline、contract 和日志；
- Prompt 只使用结构化证据；
- 报告数字完全由系统注入。

退出标准：

```text
Agent 不能改变评分规则
Agent 不能修改评测基础设施
Agent 产生的每个 patch 都可审计、可回滚
编译失败和正确性失败都有结构化修复路径
```

### Wave 4：事件日志、恢复和多卡调度

**目标：实现真正的长期无人值守运行。**

任务：

- append-only event log；
- exactly-once 预算和晋级语义；
- artifact manifest；
- lease expiry / renewal / recovery；
- 多卡 worker；
- 异域并行和故障转移；
- 任务取消、重试和清理。

退出标准：

```text
任意阶段杀死进程都可以恢复
恢复后不重复学习、不重复晋级、不丢失 best
多卡任务可以观测、取消和回收
```

### Wave 5：跨任务记忆和结构性优化

**目标：在不污染长期知识的前提下扩大复用。**

任务：

- 记忆分层和 TTL；
- stale / invalidate 机制；
- 结构性 LLM 提议；
- 算法级变体；
- 记忆消融和迁移收益评估；
- 资格审计的真实双臂消融。

退出标准：

```text
跨任务复用收益有独立对照
错误记忆可以撤销
算法级变体不会绕过正确性和晋级门禁
```

---

## 13. 不建议现在做的事情

在 Wave 0–2 完成前，不建议：

1. 增加更多 Agent 角色；
2. 先优化 Prompt，再补正确性和统计协议；
3. 支持大量后端而没有统一的候选执行边界；
4. 将更多 profiler 指标直接塞给 LLM；
5. 允许 LLM 自由重写整个仓库；
6. 把参考 Kernel 的测量下界称为硬件 roofline；
7. 让 DEGRADED 结果直接改写 best 或长期记忆；
8. 通过环境变量提供“一键关闭安全门”；
9. 在没有恢复语义的情况下扩大多卡并行；
10. 用一次真机成功运行代替系统性验收。

当前系统最大的风险不是“不会生成新 Kernel”，而是：

```text
在错误的契约、错误的基线、错误的测量或错误的恢复状态上，
无人值守地做出一个看起来很合理的结论。
```

---

## 14. 验收指标

### 14.1 安全与可信度

```text
错误候选晋级率 = 0
修改 contract / harness 的候选接受率 = 0
fallback 候选进入长期记忆率 = 0
未确认结论进入长期记忆率 = 0
恢复后重复晋级率 = 0
```

### 14.2 正确性

```text
关键 shape 覆盖率
stride / dtype / nonfinite 覆盖率
错误候选捕获率
错误定位平均粒度
正确性失败自动修复率
```

### 14.3 性能

```text
FAST / SLOW 判决可复算率
独立重测一致率
多 shape 目标达成率
关键 shape 回退逃逸率
测量协议漂移检测率
```

### 14.4 Agent

```text
首次编译通过率
编译错误局部修复率
正确性错误局部修复率
无效重写比例
每个有效候选的 token 成本
LLM 提议对最终收益的边际贡献
```

### 14.5 系统运行

```text
断点恢复成功率
任务取消回收成功率
设备租约泄漏率
重复测量率
日志完整率
多卡调度利用率
```

---

## 15. 最终目标状态

完成上述优化后，系统应具备以下性质：

```text
1. 需求在运行前被冻结，Agent 不能修改尺子。
2. 错误候选在正确性门禁被拦截，不能靠性能掩盖错误。
3. 每个性能结论都带设备、工具链、harness、契约和测量协议指纹。
4. 探索结果、正式比较结果和长期结论具有不同权限。
5. LLM 只能提交结构化、可审计、可回滚的动作。
6. 候选代码不能修改评测基础设施。
7. 多 shape 目标不会被单个 shape 的偶然收益掩盖。
8. best、预算、信念和记忆在崩溃后可以一致恢复。
9. 无法确立时，系统会明确报告无法确立，而不是强行给出结论。
10. 跨任务复用有证据、有分区、有过期和撤销机制。
```

最终系统应当更像：

```text
一个可审计的自动实验平台
+ 一个受约束的算子工程 Agent
+ 一个硬件感知的搜索器
+ 一个不可被候选代码修改的评测器
```

而不是：

```text
一个可以不断重写 Kernel 的 LLM 循环。
```
