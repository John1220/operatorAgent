# 无人值守算子开发 Agent —— 设计规范

| 项 | 值 |
|---|---|
| 文档性质 | **规范**（接口 / 不变量 / 状态机 / 判决规则 / schema / 验收标准） |
| 实现状态 | [`docs/status/autonomous-operator-agent-status.md`](../status/autonomous-operator-agent-status.md) —— 本文**不写**实现进度 |
| 研究日志 | [`docs/research/CHECKLOG.md`](../research/CHECKLOG.md) 与 `docs/research/*.md` —— 本文**不写**"哪次实验推翻了什么" |
| 优化路线 | [`autonomous-operator-agent-optimization-directions.md`](autonomous-operator-agent-optimization-directions.md) —— 波次、优先级与迁移策略 |
| 目的 | 给出一个可自动运行、自主搜索并产出**可审计结论**的 GPU 算子开发 agent 的架构与接口约定 |
| 独立性 | 本文不依赖任何既有实现；所有接口为从零定义 |
| 运行模式 | **无人值守**（autonomous）：无人在环内裁决歧义、发现失效 |
| 目标硬件 | **多卡**（同构与异构并存） |
| DSL 边界 | 初期 Triton / CUDA；后端接口预留 **TVM / CUTLASS 模板型后端** |

> ⚠️ **编号稳定性是本文的契约之一。** §1–§19 的编号被 `operator_agent/` 的模块 docstring、
> `tests/` 与其他文档**按号引用**（代码内 `设计 §X` 引用 68 处）。因此修订时**只增小节、不重排**，
> 新增平面一律追加在 §20 之后。改动编号等于同时改坏几十处不可见的引用。

---

## 0. 怎么读这份文档

### 0.1 三份文档的边界

| 文档 | 只回答 | 不回答 |
|---|---|---|
| **本文（规范）** | 应该是什么：接口、不变量、状态机、判决规则、schema、验收标准 | 现在做到了哪一步（那是状态）；为什么这么改（那是研究日志） |
| `docs/status/` | 现在是什么、被哪个可运行的守卫守着、边界在哪 | 应该是什么 |
| `docs/research/` | 为什么改变设计、哪次实验推翻了什么、历史误判 | 当前口径（历史结论**可以被推翻**，规范与状态不会） |

**规范正文里不许出现**："上次误判已撤回"、"今天还没有接上"、"某次 CHECKLOG 修复了什么"、
"某个实现缺口后来已补齐"。这些属于状态或研究日志，正文只留**当前**规范，用链接指过去。

### 0.2 逻辑顺序与编号的关系

逻辑上的依赖顺序是：**先可信 → 再可搜 → 最后可生成**（§15）。

但编号不是这个顺序：§1–§19 是历史编号（稳定引用），后补的四个平面在 §20–§26。读者若要按依赖
顺序读，建议：§1 → §2 → §3（契约与不变量）→ §4 → §6（度量与判决）→ **§20（正确性轴）** →
§9–§10（搜索与控制）→ **§21–§23（Agent Runtime / 执行隔离 / 持久化）** → §13–§14（接口与验收）。

---

## 1. 设计立场

### 1.1 一句话定位

> 这个系统的成败不在"会不会写 kernel"，而在"**能不能信自己测出来的数**"。
> 因此它是一个**以可信托管评测器为脊柱的闭环实验系统**，LLM 只是其中一个可替换的提议器，而不是一个"会写代码的助手"。

### 1.2 问题本质

| 维度 | 算子开发的现实 | 对设计的含义 |
|---|---|---|
| 正确性 | 便宜、客观、可自动判定（参考实现 + 容差） | 可以直接闭环，不需要人判 |
| 性能 | 可测，但**噪声大、极易自欺**（回退到库、TF32、launch 开销） | 度量本身是最大失效点 |
| 搜索空间 | 大但结构化、局部连续（tile / stage / warp / 向量化 / layout） | 数值调参不该交给 LLM |
| 迭代成本 | 一次编译 + 实测 = 秒到分钟级 | 样本效率 > 生成质量 |
| 硬件 | 配方不跨卡/跨架构迁移 | 记忆必须按硬件指纹分区 |
| 目标 | 通常不是单形状，而是一个形状分布 | 目标函数要聚合，不能只看一个 shape |

**写出来是容易的部分；知道它是不是真的更快、且测得是它本身，才是产品。**

### 1.3 无人值守带来的四条硬约束

| 约束 | 原因 | 设计后果 |
|---|---|---|
| **契约必须在开跑前冻结** | 中途没人裁决歧义，它会把错误的目标优雅地优化 8 小时 | 开跑前做契约 lint + 预检；冻结后**只读且哈希化** |
| **Agent 不能改自己的尺子** | 否则就是 reward hacking：放宽容差、换基线、改指标 | 容差 / 基线定义 / 目标指标由外部提供，每轮校验哈希 |
| **每个动作必须可恢复** | 没有人类来收拾残局 | 全程 checkpoint + 幂等 + 断点续跑（schema 见 §23） |
| **歧义必须失败关闭** | 测不准时它只能"猜"，而猜出来的数会污染记忆 | "无法可靠测量"是一等结论，触发停止而非继续 |

> 无人值守系统里，**最危险的不是报错，是自信地报一个错的数**。

### 1.4 安全不变量默认开启

安全语义**不是运行参数**。下面这一组能力在无人值守模式下没有"关掉"这个选项，也不通过环境变量暴露
（一个"一键关掉安全门"的开关会最先被用在无人值守的夜里）：

```
contract hash 校验 · 候选/基线来源校验 · lib fallback 检查 · 同卡成对测量
心跳与环境漂移检查 · 候选不能修改 harness · 长期记忆的独立确认 · 不可变实验日志
```

可以配置的只有**预算与搜索强度**（wall clock / GPU 秒 / oracle 次数 / LLM token / 探索-精修比例 /
benchmark 样本数 / 最小效应量）。完整清单与理由见 §3.5。

---

## 2. 总体架构

### 2.1 一次运行的流水线

```
        ┌──────────── 契约层 Contract（冻结、只读、哈希化） ────────────┐
        │ 语义 / 形状分布 / dtype 数值语义 / 基线定义 / 目标指标 / 拓扑    │
        └──────────────────────────┬──────────────────────────────────┘
                                   │
   ┌──── 记忆层 ────┐   ┌──────────▼──── 提议层 Proposer ──────────┐
   │ 硬件画像        │◄─►│ 参数化模板 + bandit（主） / LLM 结构变体（辅） │
   │ 配置库 / 失败签名 │   └──────────┬──────────────────────────────┘
   └────────┬───────┘              │ 候选（源码型 | 模板型 + 谱系）
            │                      ▼
            │   ┌──── 评测器 Oracle（脊柱：无状态、可审计、唯一收口点）────┐
            └──►│ G0 契约合规 → G1 编译 → G2 正确性 → G3 度量 + 资格       │
                └──────────────────────────┬────────────────────────────┘
                                           │ 观测（值 + 不确定度 + 溯源 + 资格位掩码 + ClaimLevel）
                                           ▼
                ┌──── 判决层 Adjudicator / 信念状态 ────┐
                │ 与同卡配对基线比 → FAST/SAME/SLOW/AMBIGUOUS │
                │ 更新信念 / 失败归类 / 决定下一步动作         │
                └──────────────────────────┬────────────────┘
                                           ▼
                预算控制器：继续搜索 / 收敛收工 / 不安全中止 / 交还给人
                                           │
                                           ▼
        产物：best-known-good + 不可变实验日志 + 报告（含"无法确立"章节）+ 实现裁决 + 复现命令
```

`G2 正确性` 是这条流水线上的**硬门禁**，不是 G3 的副产品：错误候选不许进入正式性能比较，更不许替换
`best`。它的完整规范在 §20。

### 2.2 目标平面划分

单次运行的流水线（§2.1）是**数据流**；落到工程结构上，系统分成五个平面，
而不是把所有能力都塞进 Oracle 或 Conductor：

```
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

**平面之间的硬边界**（违反任何一条都是设计缺陷，不是实现细节）：

| 边界 | 规则 |
|---|---|
| Agent → Execution | Agent 只能提交结构化 Candidate / Patch（§21），不能直接控制 benchmark 进程 |
| Candidate → Harness | Candidate 不能修改 harness、baseline、contract、日志和评分代码（§22） |
| Execution → Trust | Trust Plane 只消费带完整 provenance 的 artifact 与 observation |
| Trust → Corpus | 只有通过 promotion policy 的 claim 才能进入长期记忆（§4.4 / §24） |
| Data → Report | 报告中的数字只能来自不可变事件与已收口的 Observation |
| Operations → All | 所有任务、设备、模型调用和 artifact 都必须可审计 |

---

## 3. 契约层：把模糊需求变成可机器判定的东西

### 3.1 契约必填项

契约必须包含，且缺一不可：

1. **参考实现**（最重要的一件产物）：同时定义正确性、测试数据生成、性能基线。
2. **数值语义显式钉死**（见 §3.2）。
3. **形状分布 + 权重**（见 §3.4）：目标是 workload profile（含边缘形状：非对齐尾块、batch=1、极大/极小 k），
   优化目标是**加权几何平均**，而不是单点。
4. **基线定义方式**：不是"查表得到的数字"，而是"同一套 harness、同一时刻、同一张卡、同样数值语义下对手跑的"。
5. **目标指标 + 阈值**：吞吐 / 延迟 / 显存，以及"达到什么程度算收工"。
6. **拓扑约束**（多卡算子）：world size、通信后端、拓扑；决定性能模型是否仍适用 roofline。

契约在 `CONTRACT_LINT`（§3.6）通过后写入只读文件并哈希；评测器每轮校验哈希。

### 3.2 数值语义显式钉死

**fp32 是 TF32 还是全精度、累加精度、是否允许 split-K 的非确定性 —— 必须在契约里写死。**

理由不是洁癖：基线的加速比只差一个数值语义就能差约 2×（TF32 打开与关闭）。契约里缺这一项时，
后续所有加速比都可能是 2× 的假象；而修正它意味着重跑而不是改数。

因此：**缺失即拒绝，不猜默认值**（"不可判定就交还给人"，同 §10.1 的 `REJECT` 出口）。
数值语义随观测进入 `Provenance.dtype_semantics`，`Eligibility` 与报告都读它。

### 3.3 扩展后的 `OperatorContract`

契约不只是"性能目标 + 容差"，而是**完整的可观察语义**。目标 schema：

```python
@dataclass(frozen=True)
class OperatorContract:
    # ── 语义 ──
    operator_name: str
    formula: str | None
    reference_impl: ArtifactRef
    numerical_semantics: NumericalSemantics      # §3.2
    error_policy: ErrorPolicy

    # ── 输入输出 ──
    tensors: tuple[TensorSpec, ...]
    supported_dtypes: tuple[DType, ...]
    supported_layouts: tuple[Layout, ...]
    supported_strides: StridePolicy
    alignment: AlignmentPolicy
    aliasing: AliasingPolicy
    dynamic_shape_policy: ShapePolicy

    # ── 正确性 ──
    correctness_suite: CorrectnessSuite          # §20.1
    tolerance_policy: TolerancePolicy
    deterministic: bool
    backward_policy: BackwardPolicy

    # ── 性能 ──
    workload_profile: WorkloadProfile
    objective: PerformanceObjective              # §3.4
    regression_limits: RegressionLimits

    # ── 工程与环境 ──
    backend_requirements: BackendRequirements
    topology: TopologySpec | None
    contract_hash: str
```

设计约束：

- **不可变性**：`frozen=True`，字段集变化即 `contract_hash` 变化；评测器每轮校验（§13.4 硬性语义 1）。
- **迁移策略**：本 schema 取代 §13.1 的 `FrozenContract` 字段集。hash 因此变化 ⇒ 旧产物的
  `contract_hash` 与新产物**不可隐式互认**（按 partition 隔离，不做静默迁移）。这是刻意的：
  契约一变，"同一个 run" 这个说法就不成立了。
- **加字段优先于改语义**：新增能力一律加可选字段；已发布字段的语义只许**收紧**（例如容差表只许变严）。

### 3.4 形状分布、多 shape 目标与回退约束

形状分布同时是**两张**要求，不能只当性能目标：

```python
@dataclass(frozen=True)
class PerformanceObjective:
    cases: tuple[BenchmarkCase, ...]
    weights: tuple[float, ...]
    aggregate: Literal["geomean", "weighted_geomean", "max_regret"]
    required_cases: tuple[str, ...]        # 必须参与判决的 shape（缺一不可）
    max_regression: Mapping[str, float]    # 逐 shape 硬回退上限
```

**规范**：

1. **性能**用形状分布 + 权重，聚合成加权几何平均（或 `max_regret`，当"最差 shape 不能太差"是首要目标时）。
2. **正确性**用形状集，**必须每类边界各一**（非对齐尾块 / batch=1 / 极大 / 极小 k / 非 contiguous 等），
   候选须在**全部**形状上通过 —— 这是唯一能挡住"针对测试 shape 特化"的东西，因此它是契约条款，
   不是测试细节。
3. **关键 shape 的硬回退约束**优先于总体聚合：一个候选不能因为在某个 shape 上快很多，
   就掩盖关键 shape 上的严重回退。回退越限 ⇒ 该候选**不得晋级**（`USE_CUSTOM` 也不成立）。
4. 聚合口径、权重、`required_cases` 全部进契约 hash；**评测侧只许执行，不许重算权重**。

### 3.5 安全不变量与可配置面

**不允许配置**（无开关、无环境变量、不可被 Agent 或任务参数改写）：

```
容差          基线实现        资格判据        promotion policy     任务语义
契约哈希校验   候选/基线来源校验   lib fallback 检查   同卡成对测量
心跳与环境漂移检查   候选不能修改 harness   长期记忆的独立确认   不可变实验日志
```

**允许配置**（属于预算与搜索强度，不属于安全语义）：

```
最大 wall clock · 最大 GPU 秒 · 最大 oracle 次数 · 最大 LLM token
探索 / 精修比例 · benchmark 样本数 · 最小效应量
```

三条实现纪律：

1. **策略层必须先于开关存在**：配置的载体（文件 / 环境变量 / 契约字段）与优先级必须先定义成一个
   显式的 policy 层，然后才谈"哪些键可以配"。今天"配置"这个说法如果没有载体，就等于没有可配置面。
2. **默认值本身就是判据**：一个"默认关"的安全装置等于不存在——无人值守下没人会去打开它。
   所有安全门的默认值必须由用例钉住（"默认配置下这条路走不通"）。
3. **消融不许做成产品开关**：§14-9 需要"关掉资格审计的后果"这一臂，但那一臂**只能在 harness 侧 patch**，
   命令行不留开关 —— 一个"一键关安全门"的环境变量就是那把尺子。

### 3.6 `CONTRACT_LINT`：花 GPU 时间之前的退出点

清单（全部通过才允许 `PREFLIGHT`）：

```
参考实现存在吗？        容差能区分对错吗？        形状分布覆盖边缘形状吗？
正确性形状集覆盖每类边界吗？  数值语义钉死了吗？      基线可复现吗？        后端能力满足吗？
```

> 第 4 条是新增的：此前"覆盖边缘形状吗"这一条**只对性能负责**，正确性依赖多形状套件的效果
> 不是契约的一部分。一个后来者删掉"多输入套件"时，没有一条设计条款会反对他 —— 现在有了。

---

## 4. 度量与资格：脊柱

### 4.1 测量的产物不是数字，是一条观测记录

```json
{
  "value": 412.3, "unit": "gflops",
  "stat": {"median": 0, "p10": 0, "p90": 0, "cv": 0.031, "n": 30},
  "provenance": {
    "device": "...", "real_device": true, "arch": "sm90",
    "dtype_semantics": {"tf32": false, "accum": "fp32"},
    "source_hash": "...", "build_cache_hit": false,
    "no_lib_fallback": true, "baseline_call": "...",
    "calibration_epoch": 7, "contract_hash": "..."
  },
  "claim_level": "validated",
  "validity": "valid",
  "eligibility": {
    "can_verdict": true, "can_tune": true, "can_persist": true,
    "can_report": true, "can_count_progress": true
  },
  "evidence": ["launch_floor_ratio=0.02", "no_lib_fallback", "paired_baseline_same_device"],
  "reasons": []
}
```

### 4.2 资格从哪来（五类检查）

1. **它测的是不是"我的" kernel**：编译产物 hash、是否真的走到生成代码路径、
   运行时把库函数替换成直接报错、检查 dispatch 有无静默回退。**实践中最常见的自欺。**
2. **数值语义对不对**：TF32 / 精度 / 累加。
3. **是否超物理上界**：roofline 当**校验边界**用——报出的 FLOPS 超过峰值说明测量错了，
   不是说明很棒。
4. **信噪比**：短 kernel 的 launch 下限占比、重复采样离散度。
5. **环境一致性**：硬件 / 驱动 / 库版本指纹（`PartitionKey`）有无漂移。

### 4.3 资格推导策略（本层即产品）

```
1  契约哈希不匹配                              → 全 False，并 abort
2  落在心跳失败的标定窗口内                     → 全 False
3  契约要求 REAL_DEVICE 但 real_device=False    → 全 False（这个数没有指称对象）
3b 正确性凭据为假（correctness_verified=False）  → 全 False（同上：这个数说的是另一个对象）
4  检测到 lib fallback                          → 全 False（它在测别的东西）
5  value > 实测峰值 × 1.05                      → UNINFORMATIVE，全 False
6  cv 超阈值 或 launch 下限占比超阈值            → DEGRADED：
        can_verdict = can_tune = can_report = True
        can_persist = can_count_progress = False          ← 关键
7  partition 漂移（驱动/库/卡变了）               → DEGRADED，跨漂移比较一律作废
8  静态守卫**不适用**于这份源码                   → DEGRADED（同 6/7 的位）：
        「查不了」不是「没有问题」—— 反作弊与接口契约两道守卫都是 Python 形状的，
        对一份 C++/CUDA 模板源判不了（判据：`source_lang.classify`，三态，见 §19.3）
9  其余                                         → VALID，全 True
```

三条设计意图：

- **低资格 ≠ 没测准，而是"这个数不足以支撑结论"**。`can_persist` / `can_count_progress`
  只在 VALID 放开，因此一次噪声读数**永远不能永久拉黑一个配方，也永远不能算作收敛证据**。
- **第 3 条与第 6 条刻意不同**：降级是"数不可信"，第 3 条是"数不存在"。
  后者必须直接拒绝 `passed`，因为它测的根本不是契约要求的东西。
  **正确性凭据（3b）收在第 3 条这一族**：一个"还没确认它算得对"的核，它的速率读数不是测得
  不准，而是**没有指称对象**。这条只可能在 `verify_l3_only`（唯一绕开 L2 的入口）上触发 ——
  正常路径上数值错的候选到不了 L3。三态：**`False` 才拒，缺键（未观测）不改判**。
- **第 8 条为什么排在 3/4/5 之后**：「这份源码没被静态守卫查过」不比"测的是别的库"更严重 ——
  数字本身是真的，不足以支撑的只是**永久**结论（同 6/7：低资格 ≠ 没测准）。
  三态：只有**明确**「不适用」才降级；「未观测」（老路径没带这个值）**不改判** ——
  缺键 ≠ 判过且不合格（同 `real_device=None` 那一边界）。

> **核心原则：资格决定"这个数被允许造成什么后果"，而不只是"过没过"。**

**资格位表**：

| 规则 | `can_verdict` | `can_tune` | `can_persist` | `can_report` | `can_count_progress` |
|---|---|---|---|---|---|
| 1 / 2 / 3 / 3b / 4 / 5 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 6 / 7 / 8（DEGRADED） | ✓ | ✓ | ✗ | ✓ | ✗ |
| 9（VALID） | ✓ | ✓ | ✓ | ✓ | ✓ |

`l3_pass_is_refused` **不并进 `can_verdict`**：它是"这个数从来没被测量过"（规则 3），
与"数不足以支撑结论"（规则 6/7/8）是两件事，后者从不改写 `passed`。

### 4.4 `ClaimLevel`：面向产品与报告的声明等级

`can_*` 是**能力位**（这个数能干什么），但它不足以回答"这条结论现在处于什么地位"。
再加一个**声明等级**，并把"允许的后果"写死在等级上：

```python
class ClaimLevel(StrEnum):
    UNOBSERVED  = "unobserved"     # 没有可用测量
    OBSERVED    = "observed"       # 有原始观测，但资格不足
    PROVISIONAL = "provisional"    # 当前运行内可比较
    VALIDATED   = "validated"      # 通过正确性与测量门禁
    PROMOTABLE  = "promotable"     # 通过独立确认
    PERSISTED   = "persisted"      # 进入长期知识库
```

| 等级 | 含义 | 允许的后果 |
|---|---|---|
| `UNOBSERVED` | 没有可用测量 | 不能判决、不能学习 |
| `OBSERVED` | 有原始观测，但资格不足 | 只能诊断与继续探索 |
| `PROVISIONAL` | 当前运行内可比较 | 可用于局部搜索，**不能**晋级长期 `best` |
| `VALIDATED` | 通过正确性 + 测量门禁 | 可参与正式候选比较 |
| `PROMOTABLE` | 通过独立确认 | 可以替换运行级 `best` |
| `PERSISTED` | 进入长期知识库 | 可跨任务复用，**必须**带分区与 TTL（§24） |

**等级只能逐级上升，且每一级上升都要有对应证据**；不允许从 `OBSERVED` 直接跳到 `VALIDATED`。
升级的判据：

```
OBSERVED    → PROVISIONAL : 有一条 can_tune 的读数
PROVISIONAL → VALIDATED   : L2 正确性通过（§20）且档位 valid（规则 9）
VALIDATED   → PROMOTABLE  : 一次独立重测仍然不劣（§10.5-4，不变量）
PROMOTABLE  → PERSISTED   : promotion policy 通过（§24.3）
```

这样"允许继续试探"与"允许基于该结果改变策略"就不会再被同一个 `can_tune=True` 同时表达。

---

## 5. 提议层：LLM 做什么、不做什么

| 角色 | 手段 | 说明 |
|---|---|---|
| **主路径** | **参数化模板 + bandit / 网格** | 在 (BM, BN, BK, stages, warps, 向量宽度, layout, split-K) 空间里选点 |
| **结构创新** | LLM 生成源码变体 | 模板族覆盖不了时（融合、online-softmax 变体、算法替换） |
| **修复** | LLM + 错误日志 | 仅编译失败 / 正确性失败 |
| **报告** | LLM 写散文 | **不写数字**：所有数字由实验日志注入 |

**数值调参不交给 LLM**——LLM 猜 tile 参数又慢又不准；对一张给定卡，20~40 次带剪枝的搜索
通常优于 40 次 LLM 重写，且更便宜可靠。

**每次候选必须带父代（lineage）**：候选 = 相对某个已知好版本的 diff，
从而可以二分定位回归，并且永不丢失 last-known-good。

**去重**：对生成代码做结构等价签名（α-等价）+ 配置 hash，避免预算烧在重复点上。

**结构升级有门禁**：LLM 的结构性改动只在 §10.3 三条件同时成立时允许调用；
其余时间它只能作为修复器与报告写手。这是"LLM 是受约束的提议器"这条定位的可执行形式 ——
它的**正式协议**（结构化动作、权限、上下文边界）见 §21。

---

## 6. 判决层：显式信念状态

### 6.1 四态判决

- 与基线比较输出四态：`FAST / SAME / SLOW / AMBIGUOUS`，噪声带随资格等级变化。
  **只有 `SLOW`（且资格足够）才允许改配置或拉黑。**
- 维护**假设图 / 实验日志**：每个候选为何被生成、测出了什么、因此更新了什么信念。
  这既是决策依据，也是最终给人看的证据链。
- **停滞计数只在有效轮次上累加**（被污染的一轮不算进度，也不该结束循环）。

### 6.2 四态判决的数学定义

四态必须是**可复算的统计协议**，而不是"看起来快了"。基本统计量是**同一配对窗口内的成对相对差**：

```
样本：候选臂与基线臂在同一设备、同一测量窗内交错测得 (candidate_i, baseline_i), i = 1..n
统计量：Δ_i = (candidate_i - baseline_i) / baseline_i          # 越小越快
判决：
  FAST       UCB(Δ) < -minimum_effect
  SLOW       LCB(Δ) > +minimum_effect
  SAME       CI(Δ) 完全落在 ±noise_band 之内
  AMBIGUOUS  其余（含样本不足、资格不足、CI 跨越两个区间）
```

**契约或测量策略必须固定**（以下各项进测量协议指纹，见 §9.4）：

```
最小效应量 · 置信水平 · 最小样本数 · 最大样本数 · warmup 策略
异常值处理 · 配对顺序 · 随机化种子 · 多重比较策略
```

三条纪律：

1. **同一份原始样本必须能离线重算出同一个判决**：判决用到的每个参数（含真正用过的那条噪声带）
   都要随观测落盘，不许"记下的带宽"与"判决用过的带宽"是两个数。
2. **`AMBIGUOUS` 是结论，不是失败**：它是"证据不足"，因此既不能禁用配方，也不能计入停滞。
3. **资格不足时不许判决**：档位 `uninformative` ⇒ 直接 `AMBIGUOUS`，因为那条带宽比信号本身还宽。

### 6.3 三层测量与选择偏差

bandit 直接从"是否击败当前 best"学习，会同时受到四个效应污染：`best` 不断变化、候选测量次数不一致、
噪声大的候选更容易偶然胜出、搜索过程本身选择了最终样本（winner's curse）。
因此把测量分成三层，**权限逐层收紧**：

| 层 | 成本 | 判决强度 | 允许的后果 |
|---|---|---|---|
| 探索层 | 低 | 弱判决，允许不确定 | 只喂 bandit 排序（`PROVISIONAL`） |
| 比较层 | 中 | 同卡配对 + 固定样本 + §6.2 统计判决 | 参与正式候选比较（`VALIDATED`） |
| 晋级层 | 高 | 独立样本、独立窗口、严格确认 | **只有这一层能改写 `best`**（`PROMOTABLE`） |

**`best` 只能由晋级层改写；bandit 可以使用探索层信号，但不能把探索层信号当作永久事实。**
单调性由此成立：任何时刻的 `best` 都带一条晋级层证据。

---

## 7. 记忆层：区分可迁移与不可迁移

| 类型 | 例子 | 存储键 |
|---|---|---|
| 可迁移 | 算法模式、DSL 陷阱、硬件通用坑 | (算子族, 形状类) |
| 不可迁移 | 这张卡上最佳 tile | (算子, 形状类, **PartitionKey**) |
| 负样本 | 崩溃签名、某形状触发编译器 bug、某配置必 spill | 失败签名 |

**入库有门禁**：

- 负样本需"**跨 N 个不同任务复现**"才算一条可复用教训；
- 数值容差类与基础设施类的失败**一律不进学习库**
  —— 从"误差超阈值"能学到的唯一东西就是"放宽阈值"；
- 任何永久结论需**两次确认**（§10.5-4，**不变量，不可关闭**）。

记忆的**分层、有效期、失效与撤销**见 §24；promotion policy（哪条结论有资格进库）见 §24.3。

---

## 8. 失败分类与修复策略（价值密度最高的一层）

### 8.1 五类路由

任何修复动作之前，先做**失败归类**，不同类走完全不同的策略：

| 类别 | 表现 | 策略 | 是否算迭代 |
|---|---|---|---|
| 编译失败 | 类型 / layout / DSL 误用 | LLM 拿错误 + 代码修，便宜 | 是 |
| 正确性失败 | 数值错、边界错 | 调试模式：缩小形状、打中间量、与参考逐块对比 | 是 |
| 性能不达标 | 能跑对但慢 | **搜索失败，不是 bug**：换配置 / 换结构，别去"修代码" | 是 |
| 基础设施 | SSH / 驱动 / OOM | 退避重试，**绝不污染学习，也不计进度** | 否 |
| 静默退化 | 全过，但不是你以为的那个东西 | 靠 §4.3 的溯源检查拦住（最危险的一类） | 否 |

路由的**内容**（而不是类别名）决定动作，且不许由调用方自觉：

```
编译错误        → 局部修复优先
数值错误        → 局部修复优先，并要求误差形态诊断（§8.2）
边界错误        → 局部修复，或回退父代
性能不达标      → 进入搜索，不默认改代码
算法瓶颈        → 允许结构性变更（仍受 §10.3 门禁）
接口 / ABI 破坏 → 拒绝候选，回退父代
```

### 8.2 误差形态：比"第一个坏点"更适合决定修还是重写

第一个失配点（`first_bad_index` / `first_bad_actual` / `first_bad_expected`）回答"哪错了"，
但"修 vs 重写"最直接的依据是**误差的形态**。数值失败必须再产出一个形态标签：

```
tail_block              误差集中在尾块 / 非对齐边界        → 修边界守卫
whole_row               整行错（广播轴选错 / 索引少一维）   → 修索引
whole_tensor            整体等比偏差（缩放 / 累加精度错）   → 多半是数值语义，修或回退
broadcast_axis          某一维广播错                       → 修广播语义
precision_accumulation  误差随 k 增长（累加精度不足）        → 改累加策略，不是修索引
nonfinite               出现 NaN / Inf                     → 修数值稳定性
```

形态标签随失败记录进 `FailureDatabase`，并与"修 vs 重写"的路由判定绑在同一处（不许调用方各判一次）。

四条规范：

1. **形态是证据，不是判据**：它进理由文本与 Fixer 的上下文，**不改路由决策**。
   判歪的代价是"提示指歪了"，不是"候选被误杀" —— 这条边界要写死在实现里。
2. **判不出来就给"无法确立"，不许猜**：一个"看着像但不成立"的标签会把修复引到错的地方，
   比没有标签更坏。因此每种形态都必须说得出**量具是什么**；没有量具的那几种（今天
   `precision_accumulation` 需要"误差对归约维度的依赖"这类证据）留在未确立态，
   并在报告里如实出现，而不是先占一个名字。
3. **规则只有一处**：元素级归约由各执行路径在自己的数组后端上做（本地 numpy / 远端 torch），
   分类规则必须共用同一份实现，并由一条"同一掩码两边给同一标签"的一致性用例钉住。
4. **触发条件是"这一组判失败"，不是"有坏点"**：含 NaN/Inf 时坏点计数是 0（NaN 的比较恒假），
   而那恰恰是最该说出口的一种形态 —— 挂在坏点计数上的话，它会是唯一永远报不出来的那一类。

---

## 9. 搜索策略：分层 + 便宜代理先剪枝

### 9.1 层级

```
算法级（融合 / split-K / flash 式在线 softmax）
   └─ 分块级（BM / BN / BK / stages / warps / split-K）
        └─ 微观级（向量宽度 / layout / swizzle / 异步拷贝 / cluster）
```

**从粗到细，不平铺搜索。**

### 9.2 两级 oracle

| 级别 | 手段 | 成本 |
|---|---|---|
| 便宜代理 | 编译是否通过、寄存器 / spill 数、共享内存占用、occupancy 估算、PTX/SASS 静态检查 | 秒级，无 GPU |
| 真值 oracle | 上卡实测 | 秒~分钟，GPU 秒 |

**先用代理剪掉明显差的，再花 GPU 时间去测真值。** 多数设计只做后者，很烧钱。

便宜代理的**产出也是证据**：它同时是 §10.3 条件 2 的量具输入（spill / smem / 占用率 / bank conflict）。
没有产出者的量具等于门关着 —— 因此"量具 → 生产者 → 判据"三者必须成套交付。

### 9.3 缓存

| 缓存 | 键 | 能否跨卡共享 |
|---|---|---|
| 编译产物 | `BuildKey(backend, arch, source_hash, flags)` | **能**（同 arch 共享） |
| 测量结果 | `(candidate_hash, PartitionKey)` | **绝对不能**（时钟 / 温度 / 竞争不同） |

把这两者混成一个键，是多卡系统最常见的错误。

### 9.4 缓存键的扩展：工具链与测量协议

两个键还不够。性能测量缓存必须能回答"这两次测量是不是**同一把尺子**量的"，
因此至少再绑两组指纹：

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

性能测量缓存至少绑定：**候选 · 设备分区 · 工具链 · harness · 测量协议 · 契约**。
任何一项变化 ⇒ 缓存未命中（**不做"近似命中"**：测量缓存跨协议复用是最难发现的自我欺骗）。

---

## 10. 无人值守控制器

### 10.1 状态机

```
INTAKE ──► CONTRACT_LINT ──► PREFLIGHT ──► EXPLORE ──► EXPLOIT ──► CONVERGE ──► REPORT
   │            │                │             │            │            │
   │            │                │             └────────────┴────────────┘
   │            │                │                     （按预算回流）
   │            │                └─► ABORT(环境 / 标定失败)
   └────────────┴─► REJECT(契约不可判定，交还给人)
```

`CONTRACT_LINT` 与 `PREFLIGHT` 是**两个真实的退出点**：
自动跑系统必须在花 GPU 时间之前就敢说"这事我判不了"。

- `CONTRACT_LINT`：见 §3.6 清单。
- `PREFLIGHT`：harness 自检（§11）、硬件画像探测、**峰值算力/带宽的真实标定**
  （roofline 的分母只能来自实测，不能来自 spec 表）。

### 10.2 三种预算，分开管

| 预算 | 角色 | 典型分配 |
|---|---|---|
| wall clock | 硬上限，兜底 | 任务级 |
| **真值 oracle 次数**（GPU 测量次数） | 真正的货币 | 5% 预检标定 / 55% 探索 / 30% 精修 / **10% 最终确认** |
| LLM token | 只在结构级介入时消耗 | 独立管；超了退化成纯数值搜索 |

最后 10% 的"最终确认"预算**不可挪用**：任何要写进长期记忆或对外报出的结论，
必须留有**一次独立重测**的预算。这条挡掉绝大多数"一次侥幸读数变成永久真理"。

**三条边界**（写进规范，因为它们决定预算能不能被信任）：

1. **切分顺序是判据的一部分**：先切预留，再 explore、calibration，余数归 refine，
   使四格之和恒等于总数。顺序不能反 —— 余数归 confirm 会让预留**不单调**（total=3 有 1 次、
   total=4 反而 0 次，那是取整事故，不是设计意图）。
2. **心跳只记账、不被拦**：让预算饿死标定，等于把"尺子有没有坏"变成一件可以因为缺钱而跳过的事
   （§10.5-5 的同族病）。超支只报警。
3. **分相预算是选择的约束，不是计划**：有格子 ⇒ 约束；一格都没有 ⇒ 不拿计划当约束（记 `over_plan`）。
   被**强制**执行的只有"总数 − 预留"与"预留专属 confirm"两条硬边界；计划被突破只留痕，不停运行
   —— 交付 best-known-good 优先。

### 10.3 每轮的决策策略

候选选择用 bandit，不用 LLM：

```
score(c) = P(beats_best | 观测) × 预期增益 / 预期测量秒

选择：
  ε 概率 → 最不确定 / 未探索区域（信息优先）
  1-ε    → 期望增益最高（收益优先）
```

**三块原料各自三态**：

- `P(beats_best)`：无观测 ⇒ `None`；观测过但一次没赢 ⇒ **取小的正数，不取 0**
  （"没赢过"不是"不可能赢"）。
- **预期增益**：只从**赢**的那些轮算 —— 把负回报摊进来等于把负证据读成正证据；一次没赢 ⇒ `None`。
- **预期测量秒（分母）**：本臂的平均测量秒；本臂没有就用全局均值兜底并**标注来源**；
  全局也没有 ⇒ `score=None` 且写明缺哪一块。

> **分母的货币**是控制端观测到的**测量墙钟**（不是卡上时间），因为它与 §10.2 的墙钟预算同一种货币、
> 可离线复算。**但它是"该臂的边际成本"，不是全局常数**：分母必须逐臂取值，
> 否则除法在退化做无事可做的事。

**抽签不需要随机源**：`draw = hash(seed, round)`，种子与 ε 随产物落盘 ⇒ 逐轮抽签值**可离线复算**。

**何时才升级到 LLM 做结构性改动**（三条件同时满足，否则不许调用）：

1. 参数搜索已经平台化（K 轮 valid 且 SAME/AMBIGUOUS）；
2. 平台化有**可归因的硬限制**证据，且证据**说得出出处**：
   `spill-bound` / `smem-bound` / `占用率上限` / `bank conflict` 四类之一；
3. 还存在一个可探索的结构方向（融合 / split-K / 算法替换）。

条件 2 的三条纪律（否则这道门会以两种相反的方式失效）：

- **缺证据 = 条件不成立**（fail-closed）：量具取不到输入时，不许用默认值假装测过，
  也不许用"感觉调不动了"的代理量把门撬开；
- **同一件事只许一个认领者**：更紧的那一侧认领，另一条不报 —— 否则条件 2 会看起来证据翻倍；
- **`SLOW` 不算平台化**：退步是"还在动"，平台化要的是"既没变好也没变坏"的平坦区。

> 这条门禁很重要：LLM 在自动跑里最大的价值是**结构性跳跃**，
> 最大的风险是被当成万能调参器反复重写同一份代码。

### 10.4 停止条件（自动跑最难的部分）

| 条件 | 判据 | 备注 |
|---|---|---|
| **达标** | 达到目标指标，测量 `valid`，且**重测确认** | 重测是硬要求（§10.5-4） |
| **预算耗尽** | 任一预算触顶 | 交付 best-known-good，**不装作成功** |
| **可靠停滞** | K 轮 valid 无改善，且**有希望的区域已穷尽** | 必须"穷尽"才算，不能只因轮数 |
| **硬件峰值收敛** | 利用率接近**实测硬件峰值**，且分母是真正的峰值 microbenchmark | 见 §13.5：分母语义必须与出口名一致 |
| **不安全中止** | 心跳失败 / 环境漂移 / 连续基础设施失败 | 中止且**不写入任何记忆** |

覆盖判据三态（"有希望的区域已穷尽"必须能表达"无法确立"）：

```
exhausted    有一个可枚举的候选全集，且每个点都至少观测过一次
             ⇒ 与 K 轮无改善同时成立才停（停机理由点名两半）
unexplored   还有候选没试过 ⇒ 不停；把"达到阈值却没停"记进 declined
             （没有这一格，"因为没穷尽所以继续跑"与"停滞判据根本没人读"在产物里长得一样）
unknown      无法确立 ⇒ 停滞条件不成立
```

**空集不算已穷尽**（空队列自证可停是最危险的那种写反）。代码型路径的候选按需产生，
没有可列举的全集 ⇒ 覆盖判据恒 `unknown`，其收口靠轮次与 §10.2 预算 —— **不许**把"轮数够了"读成"停滞"。

要点：**"无法再改进"这个结论本身必须可信。**
否则要么烧卡到天长地久，要么错误宣布胜利。

### 10.5 自动跑专属安全装置

1. **标定心跳（Calibration heartbeat）**
   每 N 次真值测量或每 T 分钟，往 harness 里塞两个已知参照：一个已知好 kernel、
   一个故意写慢的 kernel。二者都在带内 → 尺子可信；任一偏出 → 该卡宣布 `RULER_BROKEN`：
   停止基于数字决策、该窗口观测全部失效、不持久化、直接进 REPORT。
   **这是无人值守下唯一能发现"度量器悄悄坏了"的机制。**

2. **环境漂移**
   每次测量记录硬件 / 驱动 / 库 / 时钟指纹。漂移出现时，**跨漂移的比较一律作废**
   （不是"打折使用"），旧记忆标记为需重测。

3. **抗污染（Poison-proof）**
   以下三类事件**不允许**更新信念、累加停滞计数、写长期记忆：
   基础设施失败、资格为 `uninformative` 的测量、落在心跳失败窗口内的所有测量。

4. **永久结论需两次确认**（**不变量：不可配置、不可关闭**）
   任何进长期语料库的结论（"这个配方在这张卡上更快"）必须：
   一次 `valid` 且 FAST **+** 一次独立重测仍然不劣。

   这条**不是运行参数**。把它做成开关，就等于让"默认配置下单次读数即可入库"成为常态，
   而这套系统花在 `confirm` / 预留预算 / 报告单列一行上的全部成本，防的正是这件事。
   机时的去处是明确的：§10.2 已把 10% 划成**不可挪用**的 confirm 预留，多测这一次不占探索与精修。

   **独立性边界必须写进方法本身**：给的是"另跑一次"，新的计时窗口、同一台机器、同一份输入、
   同一张卡 —— 它抓得住"第一次是侥幸"，抓不住"这台机器的尺子系统性偏了"（后者是标定心跳的活）。

5. **编辑不了自己的判据**
   契约、容差、基线调用、目标指标在 `CONTRACT_LINT` 通过后写入只读文件并哈希；
   评测器每轮校验哈希，agent 只能读。**这是护栏底线，不是可选项。**

### 10.6 控制器伪代码（浓缩版）

```python
contract = lint_and_freeze(user_spec)                  # 不可判定 → 直接交还给人
if not preflight_ok(contract): abort("environment")    # 含峰值标定

best = None
while not stop(contract):
    if heartbeat_due():                                # §10.5-1
        if not calibrate(): abort("ruler broken")      # 不持久化、不学习

    c = propose(contract, belief, budget)              # bandit 优先；LLM 需门禁（§10.3）
    if proxy_prune(c): continue                        # 便宜代理剪枝，不花 GPU 秒

    obs = oracle.evaluate(c, contract)                 # 值 + 不确定度 + 溯源 + 资格 + ClaimLevel
    if obs.eligibility.can_tune:
        belief.update(c, obs)                          # FAST / SAME / SLOW / AMBIGUOUS（§6.2）
    if obs.eligibility.can_count_progress:
        stagnation.update(belief)

    if (obs.eligibility.can_persist and obs.correctness_passed    # §20 正确性门禁
            and beats(best, obs)):
        if confirm_by_remeasure(c, reserve_budget):    # §10.2 的 10%，不可挪用
            best = c                                   # 只有这里能改写 best（晋级层，§6.3）

    checkpoint(belief, best, ledger)                   # 幂等 + 可续跑（§23）

report(best, log, unable_to_establish=[...])           # 显式列出没确立的东西
```

**`can_tune=False` 的轮次照样跑满预算、不许提前终止**：资格不足是"这个数不进信念"，
不是"停下来"。这两件事在产物里必须长得不一样。

---

## 11. 多卡：干扰域与同卡配对

### 11.1 身份

```python
@dataclass(frozen=True)
class DeviceKey:
    uuid: str                  # 最稳的锚，别用 index
    node: str
    index: int
    arch: str                  # sm90 / sm80 / sm75
    mig: str | None            # MIG 切片
    interference_domain: str   # 同 L2 / 同时钟域 / 同 NVLink 域 → 同域不可并行测量
```

### 11.2 放置策略（按优先级）

1. **同域串行，异域并行。**
   不同 `interference_domain` 的候选可并行以省 wall clock；同一域内**必须串行**
   —— 否则共享 L2 / 时钟域会互相污染，而 cv 会诚实地变大，把观测打到 DEGRADED。

2. **同卡配对（paired baseline）。**
   一个候选的基线必须租到**同一张卡、同一时间窗**。
   **A 卡基线 vs B 卡候选的"加速比"直接作废**——多卡下最容易犯、也最难发现的错。

   配对是**观测复核**而不仅是声明：设备与窗口各自有观测来源，且各是三态
   （`same` / 拒绝 / `unobserved`）。**缺观测不拒评**，但**观测到不同** ⇒ 拒评 +
   `passed=False`（加速比没有指称对象）。测量窗与心跳窗是两组参数，混用会恒真或恒假。

3. **交错测量。**
   candidate / baseline **交替 + 每轮 L2 flush**，而不是"先跑基线，再跑候选"，
   否则热/时钟状态变化会污染比较。

4. **竞争检测。**
   租不到独占卡时，记录外部占用并在 `evidence` 中显式标注；观测降为 DEGRADED
   而不是失败——**诚实降级优于静默污染**。

   三态纪律（这一条是数据模型要求，不是实现细节）：`未观测` 不许被编码成 `False` 或 `0.0`。
   `exclusive=None` 表示"没有观测到独占性"，`external_load=None` 表示"没观测"，
   而 `external_load=0.0` 才是"观测到干净" —— 把前者写成后者，就是拿"没测"冒充"干净"。

5. **异构卡。**
   不同 `arch` → 不同分区、不同编译缓存、不同记忆。`best` 只在同一 partition 内有效；
   跨 arch 迁移是**显式声明的操作**，绝不隐式共享。

6. **多卡算子。**
   若算子本身跨卡，契约加 `TopologySpec`；性能模型变成 compute / comm overlap，
   **roofline 分母不再适用**，需额外维度衡量重叠率，基线也必须跨卡配对测。

### 11.3 多卡能力的最小闭环

"多卡"是**调度层能力**，不是在 `DeviceKey` 上多加一个字段。在宣称任何形式的异域并行之前，
下面这一串必须逐项存在（缺一项则只能单卡串行，且报告必须写明）：

```
设备发现 · 租约获取 · 租约续期 · 租约释放 · 任务取消 · worker 心跳
GPU reset / worker 故障处理 · 同卡配对 · 跨节点产物汇聚
```

`interference_domain` 的来源优先级：

```
硬件拓扑探测 · MIG / MPS / 容器信息 · 时钟域与共享缓存信息 · 调度器配置 · 人工声明 · unknown
```

**来源为 `unknown` 时**：允许单卡串行，**不允许**宣称异域并行，报告里必须写明未知。

---

## 12. 可插拔后端：代码可选是核心

接入 TVM / CUTLASS 的最大风险是接口把"候选"硬编码成源码字符串——
那一刻起，模板型后端就无法接入，整条链路要重写。

因此 `Candidate` 必须是**联合类型**：

```python
@dataclass(frozen=True)
class SourceCandidate:            # Triton / CUDA
    backend: str
    source: str
    build_flags: Mapping[str, Any]
    struct_sig: str               # α-等价结构签名（去重）

@dataclass(frozen=True)
class TemplateCandidate:          # CUTLASS / TVM：不产代码，只在空间里选点
    backend: str
    template_id: str
    params: Mapping[str, int | str | bool]
    instantiator_ver: str

Candidate = SourceCandidate | TemplateCandidate
```

`Backend.param_space()` 返回 `None` 的后端 = **代码型后端**，控制器自动切换成
"LLM / 模板产出候选、bandit 只做局部精修"。这就是统一两类后端的那个开关。

四条规范：

1. **去重口径分开写**：源码型走 α-等价结构签名；模板型走 `template_id + instantiator_ver + 排序后的参数`
   —— 模板候选没有源码可签名。两点同源仍是两个候选，同点两次构建仍是同一个候选。
2. **身份委派给模板点，源码变化进证据**：`source_hash`（跨语言内容哈希）与 `struct_sig`（Python AST）
   记录实际构建出来的东西；空串 = 该量具对这门语言不适用（不是"通过"）。
3. **实例化必须是协议方法**：`Backend.instantiate` 默认 `raise`，五道守卫（类型 / 实例化器可达 /
   返回形状 / 非空源码 / 版本自报一致）用异常拒绝，批量收成带理由的 `refusals`。
   不许靠 `hasattr` 嗅探"有没有这个方法"。
4. **便宜代理剪枝与 L1 资源预检共用同一个估算器与同一个真源**，否则"证据"与"预检"是两套算法。

   S3 落地边界：逻辑 tile、任务 dtype、累加器和 num_stages 不足以证明共享存储。
   拒绝、剪枝和硬限制归因共用 `assess_shared_memory`；只有 harness 提供且身份/作用域
   匹配的显式 shared layout 或原生资源回执可产生物理字节结论，否则保持 unknown。
   `feasible=True` 只表示预检未拒绝，不能代替原生编译/launch、正确性或占用率证据。

---

## 13. Stage 1 接口草案

### 13.1 身份与分区

```python
@dataclass(frozen=True)
class PartitionKey:            # 「记忆与测量」的分区
    device_uuid: str
    arch: str
    driver: str
    runtime: str
    backend_ver: tuple[str, str]

@dataclass(frozen=True)
class BuildKey:                # 「编译产物」的分区 —— 与上面刻意不同
    backend: str
    arch: str
    source_hash: str
    build_flags_hash: str
```

工具链与测量协议指纹见 §9.4，缓存绑定清单同节。

### 13.2 后端协议

```python
class Capability(StrEnum):
    REAL_DEVICE = "real_device";  FP8 = "fp8";  TF32 = "tf32";  FP64 = "fp64"
    TENSOR_CORE = "tensor_core";  ASYNC_COPY = "async_copy"
    CLUSTER = "cluster";          MULTI_GPU = "multi_gpu";  DETERMINISTIC = "deterministic"


class Backend(Protocol):
    name: str
    version: str

    # ① 静态能力声明 —— 契约 lint 用它做「零 GPU 时间」的前置否决
    def capabilities(self, device: DeviceKey) -> frozenset[Capability]: ...

    # ② 参数空间 —— 模板型后端（CUTLASS/TVM）的核心；源码型后端可返回 None
    def param_space(self, contract: Contract) -> ParamSpace | None: ...

    # ③ 构建 —— 源码型 / 模板型统一入口
    def build(self, cand: Candidate, device: DeviceKey) -> BuildResult: ...

    # ④ 静态资源画像 —— 便宜代理 oracle 的来源（可返回 None）
    def static_resources(self, art: ArtifactRef) -> ResourceReport | None: ...

    # ⑤ ABI —— 调用约定由后端广播，不让模型猜
    def abi(self) -> Abi: ...

    # ⑥ 基线 —— 必须由后端给出「同一数值语义下的对手实现」
    def baseline(self, contract: Contract, device: DeviceKey) -> BaselineRef: ...

    # ⑦ 实例化 —— 模板型后端把「参数点」变成可构建候选；代码型后端默认 raise
    def instantiate(self, template_id: str, params: Mapping[str, Any]) -> Instantiation: ...
```

⑥ 放在后端而非编排层，是因为**正确的基线取决于后端产生的数值语义**
（TF32 还是全精度 fp32，差约 2x）。**谁生产语义，谁定义对手。**

```python
@dataclass(frozen=True)
class ResourceReport:
    regs_per_thread: int | None
    spills: int | None
    smem_bytes: int | None
    occupancy_est: float | None
    backend_specific: dict     # nvcc -Xptxas -v / triton metadata / cutlass tile desc
```

### 13.3 观测与资格

```python
class Validity(StrEnum):
    VALID = "valid"; DEGRADED = "degraded"; UNINFORMATIVE = "uninformative"


@dataclass(frozen=True)
class Eligibility:
    can_verdict: bool          # 能判对 / 错
    can_tune: bool             # 能改配置 / 参数
    can_persist: bool          # 能写长期记忆（永久生效）
    can_report: bool           # 能对外报数
    can_count_progress: bool   # 能算一次有效迭代


@dataclass(frozen=True)
class Provenance:
    device: DeviceKey
    partition: PartitionKey
    real_device: bool
    dtype_semantics: DTypeSemantics    # tf32 on/off、累加精度
    source_hash: str
    build_cache_hit: bool
    no_lib_fallback: bool              # 是否验证过没静默回退到库
    baseline_call: str                 # 基线是否同一张卡、同一窗口
    calibration_epoch: int
    contract_hash: str
    correctness: CorrectnessVerdict    # §20：通过 / 失败 / 无法确立（三态）


@dataclass(frozen=True)
class Observation:
    metric: str
    value: float
    unit: str
    stat: Dispersion                   # median / p10 / p90 / cv / n
    provenance: Provenance
    validity: Validity
    eligibility: Eligibility
    claim_level: ClaimLevel            # §4.4
    evidence: list[str]                # 判据来源，人可读
    reasons: list[str]                 # 降级 / 不可用的原因
```

资格推导见 §4.3，声明等级见 §4.4，正确性三态见 §20。

### 13.4 评测器：唯一收口点

```python
class Oracle(Protocol):
    def evaluate(self, rec: CandidateRecord, contract: Contract,
                 device: DeviceKey) -> Observation: ...
```

三条硬性语义写在 `evaluate` 内部（而非调用方）：

1. `rec.contract_hash != contract.hash` → **拒评**。
   这是反 reward hacking 的物理收口：**改不了契约就改不了判据。**
2. 基线与候选**同卡、同窗口、交错测量**。
3. 每次测量后**重新校验标定 epoch**，而不是只在循环开头校验一次。

另加一条（§20 的落点）：**G2 正确性未通过 ⇒ 不产生可用于晋级的性能观测**。
正确性不是性能观测的一个字段，而是它的前置条件。

### 13.5 放置、标定与峰值的命名

```python
@dataclass(frozen=True)
class DeviceLease:
    device: DeviceKey
    exclusive: bool | None        # None = 未观测到独占性
    external_load: float | None   # None = 未观测；0.0 = 观测到干净
    harness_peers: int | None     # 同卡其他 harness 进程（声明生命期 = 对方整个进程）
    source: str                   # 这条租约从哪个观测装出来的；空串 = 出处不明
    lease_id: str | None
    expires_at: str | None


class Placement(Protocol):
    def lease(self, contract: Contract, n: int) -> list[DeviceLease]: ...


class Calibrator(Protocol):
    def probe_peak(self, device: DeviceKey) -> PeakProfile | None
    def reference_lower_bound(self, device: DeviceKey) -> PeakProfile | None
    def heartbeat(self, device: DeviceKey, epoch: int) -> HeartbeatResult
```

**三个概念必须分开，名字必须各归各位**（混用会让停止条件说假话）：

```python
peak_trusted          # 规格 / 文档口径经过检查（dense/sparse 没混、算术自洽）
peak_measured         # 真正的硬件峰值 microbenchmark 实测（独立、compute-bound、按精度标定）
reference_lower_bound # 已知参考核在当前 workload 上的实测可达值 —— 是下界
```

| 概念 | 可以用于 | **不可以**用于 |
|---|---|---|
| `peak_trusted` | 校验边界（超过它说明测量错了）、契约 lint | 收敛判定（它只是"口径核过"） |
| `reference_lower_bound` | 接近参考实现的诊断、搜索排序特征、报告 | **硬件峰值停止条件** |
| `peak_measured` | **硬件峰值收敛出口** | —— |

因此：

- 硬件峰值的停止条件**只有** `peak_measured` 能打开；`peak_trusted` 与 `reference_lower_bound`
  都不行（前者是"口径核过"，后者是下界 ⇒ 利用率会偏大、出口会偏松）。
- 收敛出口的语义名必须是 **`reference_ceiling_reached`**（贴着当前参考实现的天花板），
  **不得**叫 `roofline_converged` —— 除非分母确实来自独立、compute-bound、按精度标定的硬件峰值测试。
- `probe_peak` 探测失败 ⇒ 硬件峰值出口**永久关闭**。关着只少一条停止条件
  （停滞 / 预算 / 不安全中止三条仍在）；开着会把还能优化的核提前判成到顶，方向是不对称的。
- **按精度发布与读取**：实测峰值是 `{精度: TFLOPS}`，键由**这一轮选分母用的那个** `compute_used` 得到；
  取不到就当作未实测（**不许**借别的精度 —— tf32 与 fp32 差近一倍）。
- **失败一律是"继续关着"**：仿真后端（无 `REAL_DEVICE`）、档位不是 valid、离散度未知、
  AI 低于脊点、高出规格 5% 以上 —— 每一条都拒绝，并**把理由落进产物**，
  于是"没标定"与"没接线"分得开。
- **`heartbeat` 逐卡独立**：一张卡被降频 / 被占，不代表别的卡坏。
  每个 `Observation` 记录自己的 `calibration_epoch`，失败窗口内的观测全部失效。

---

## 14. Stage 1 验收标准

这是**标定"标定器"本身**的测试集。全绿才允许往上盖 Stage 3（搜索）与 Stage 4（LLM 提议）。

### 14.1 门禁用例

| # | 测试 | 期望 |
|---|---|---|
| 1 | 每张卡上测已知好 kernel | VALID，跨次 cv < 阈值 |
| 2 | 测故意写慢的 kernel | 判决正确给出 SLOW |
| 3 | 注入一个"回退到库"的假候选 | 资格全 False，被拒 |
| 4 | 后台起一个占卡任务再测 | 降为 DEGRADED，**不是**错误通过 |
| 5 | 中途 kill 进程再启动 | 日志无损坏、可续跑、best 未丢 |
| 6 | 人为改契约文件 | 评测器拒评（哈希不匹配） |
| 7 | 把候选和基线放到不同卡 | 配对检查拒绝该观测 |
| 8 | 关掉心跳 | 该窗口观测全部不可用，不写记忆 |
| 9 | **同一批轮次跑两臂**：A 照常 / B 关掉资格审计的**后果** | 五态判决，且"没有收益"必须写进报告 |
| **10** | **喂一个数值错的候选**（缩放 / 丢一个输入 / 尾块越界） | **被 G2 拒，且理由指向该处**；不产生可用于晋级的性能观测 |

第 9 条与前面八条性质不同：前八条问"资格审计**在不在岗**"，第 9 条问"它在不在**值**"。

**第 10 条是硬要求，不是补充用例**：前九条全部落在度量 / 资格 / 契约哈希 / 配对 / 窗口 / 心跳上，
一条都不验证"**错的 kernel 会被抓住**"。缺了它，一个完全不检查数值的实现只要通过前九条，
就能按本文的规则合法地盖起 Stage 3 与 Stage 4。§14 应视为两节：
**14-A「尺子可信」**（1–9）与 **14-B「尺子量的是对的东西」**（10）；前者不蕴含后者。

### 14.2 第 9 条的消融协议

**协议**（跑法是两臂各跑一次）：

- 两臂只差一件事：B 组让资格位不再拦任何东西（判据**照算照记**，只是后果不再受限）。
  ⚠️ 这个"关掉"**不许做成产品开关**（§3.5 规则 3）—— B 组只能在 harness 侧 patch 出来
  （`oracle.admit_to_corpus` 忽略 `can_persist`，控制器把每个读数都当有效），命令行里没有这个开关。
- 三个数全部从 `iteration_history` 离线可算（所以能复盘，不必重跑）：最终 `best`、
  真值测量轮数（§10.2 的货币）、**脏永久后果**轮数（读数没资格造成永久后果却造成了：
  覆盖 `best` / 永久禁用配方 —— 就是 §16 那条"让一个噪声大的数永久拉黑配置"）。
- 判决五态：`无法比较` / `没打到点上` / `没有收益` / `证据不足` / `有收益`。
  带宽取最小的那一档噪声带（跨运行比较的可信度低于轮内比较）。
  关键的两条：**"门没被需要过"不算"门没用"**（否则从无信息量的实验里读出结论）；
  **"没有收益"这一向必须写进报告**。

在双臂真机对照完成之前，"资格审计有收益"这句话**不许对外说**。

### 14.3 验收指标

| 组 | 指标 |
|---|---|
| **安全与可信度** | 错误候选晋级率 = 0 · 修改 contract/harness 的候选接受率 = 0 · fallback 候选进入长期记忆率 = 0 · 未确认结论进入长期记忆率 = 0 · 恢复后重复晋级率 = 0 |
| **正确性** | 关键 shape 覆盖率 · stride / dtype / nonfinite 覆盖率 · 错误候选捕获率 · 错误定位平均粒度 · 正确性失败自动修复率 |
| **性能** | FAST / SLOW 判决可复算率 · 独立重测一致率 · 多 shape 目标达成率 · 关键 shape 回退逃逸率 · 测量协议漂移检测率 |
| **Agent** | 首次编译通过率 · 编译错误局部修复率 · 正确性错误局部修复率 · 无效重写比例 · 每个有效候选的 token 成本 · LLM 提议对最终收益的边际贡献 |
| **系统运行** | 断点恢复成功率 · 任务取消回收成功率 · 设备租约泄漏率 · 重复测量率 · 日志完整率 · 多卡调度利用率 |

---

## 15. 实施顺序（顺序是刻意的）

```
Stage 0  契约冻结 + 只读哈希护栏 + 预检 / 峰值标定
Stage 1  评测器 + 资格模型 + 标定心跳 + checkpoint/幂等/续跑   ← 尺子（含尺子的自检）
Stage 2  多卡放置：干扰域 + 同卡配对（若一开始就是多卡，则与 1 并行）
Stage 3  参数模板 + bandit 搜索 + 停止条件                     ← 可搜
Stage 4  LLM 结构提议（带三条件门禁）
Stage 5  记忆与跨任务迁移（带两次确认）
```

**先可信、再可搜、最后可生成。**

反过来做（先搭一个很酷的多 agent 生成流水线）几乎必然得到一个
"每次都能产出 kernel、但没人知道数是真的假的"的系统。

> 注：若确定从一开始就是多卡，Stage 2 不是"后期工程"，而应与 Stage 1 同期，
> 因为"基线同卡配对"是评测器语义的一部分，不是外挂的调度优化。

**两个横切的补充**（Stage 之外，但决定 Stage 能不能往上盖）：

- **正确性与性能同级**：G2 正确性门禁（§20）必须在 Stage 3 之前是硬门禁，否则"能跑对但慢"
  与"又快又错"在流水线里长得一样。
- **执行隔离与持久化**（§22 / §23）不随 Stage 递延：Stage 1 就需要"候选改不了 harness"
  与"崩溃后不重复计费"。它们是**不变量**，不是工程量。

优化波次（Wave 0–5）、优先级与迁移策略见
[`autonomous-operator-agent-optimization-directions.md`](autonomous-operator-agent-optimization-directions.md) §12。

---

## 16. 反模式清单

- 相信 LLM 自称的加速比。
- 基线取自查表或上次运行的结果。
- 让一个噪声大的数永久拉黑配置、或算作收敛。
- 让基础设施失败驱动优化决策。
- 让 LLM 做数值自动调参。
- 一次成型交付，没有同 harness 的对照测量。
- 在"正确性大概没问题"的状态下开始调性能。
- 用单形状的加速比作为成功标准。
- 让 agent 能改写容差 / 基线 / 目标指标。
- 把编译缓存和测量缓存用同一个键。
- 跨卡比较候选与基线的性能。
- 把候选代码和评测 harness 放进同一个可写上下文（§22）。
- 让"未观测"以 `False` / `0.0` / 空串的形式进入数据模型（§11.2 三态纪律）。
- 把参考实现的下界叫成硬件峰值（§13.5）。
- 用环境变量提供"一键关闭安全门"。

**正模式**（可执行的规则，而不是"不要做"）：

| 规则 | 出处 |
|---|---|
| **资格决定这个数被允许造成什么后果，而不只是过没过** | §4.3 |
| **数不存在**（规则 3）与**数不足以支撑结论**（6/7/8）必须分开，二者对决策的后果相反 | §4.3 |
| **"查不了"不是"没问题"**：静态守卫适用性三态，"未观测"不改判 | §4.3 规则 8 |
| **缺证据 = 门关着**（fail-closed），且不许用代理量撬门 | §10.3 条件 2 |
| **同一件事只许一个认领者**（否则证据看起来翻倍） | §10.3 条件 2 |
| **口径的切分顺序是判据的一部分**（先切预留，否则预留不单调） | §10.2 |
| **判决用过的参数必须随观测落盘**（不许"记下的"与"用过的"是两个数） | §6.2 |
| **门的默认值本身就是判据**：默认关的安全装置等于不存在 | §3.5 |
| **消融不许做成产品开关**：能一键关掉的尺子会先被用在无人值守的夜里 | §3.5 / §14.2 |
| **一个从不说"我测不准"的自动系统，它的所有数字都不可信** | §17 |

---

## 17. 交付物（一次自动跑结束）

1. **best-known-good**：代码 + 完整配置（永不因未确认的结果被覆盖）+ **实现裁决**
2. **不可变实验日志**（事件日志）：每次尝试、每个读数的溯源与资格
3. **报告**：结论 + 证据链 + **"我无法确立的事"**（一等公民章节）
4. **复现命令** + 硬件 / 驱动 / 库 / 工具链 / 测量协议指纹

### 17.1 实现裁决：`USE_CUSTOM / USE_LIBRARY / INCONCLUSIVE`

"候选比 baseline 快"不足以支撑一个部署决定。交付物必须带一个显式裁决：

```python
class ImplementationDecision(StrEnum):
    USE_CUSTOM     = "use_custom"
    USE_LIBRARY    = "use_library"
    INCONCLUSIVE   = "inconclusive"
```

```
自定义实现正确、在目标 workload 上稳定快、无关键 shape 回退   → USE_CUSTOM
自定义实现正确但稳定慢 / 维护成本不划算 / 有回退越限           → USE_LIBRARY
资格不足、结论冲突、环境不稳定                                → INCONCLUSIVE
```

**裁决字段本身受资格与声明等级约束**：`USE_CUSTOM` 要求 `ClaimLevel ≥ VALIDATED`
且所有 `required_cases` 正确；`USE_LIBRARY` 至少要求 `can_report`；
其余一律 `INCONCLUSIVE`。

理由不只是完整性：本文反复防范的失效模式是"噪声读数造成永久后果"。
**"因为一次噪声读数就建议你别上自定义核"与"因为一次噪声读数就永久拉黑一个配方"
是同一失效模式的两个方向** —— 因此裁决必须接进同一套资格位，否则等于在下游开了个没有闸门的新口子。

### 17.2 报告的纪律

- 第 3 件的"无法确立"章节是自动跑系统的信誉所在：

  > **一个从不说"我测不准"的自动系统，它的所有数字都不可信。**

- **空章节必须带依据。** 空清单有两种来源（"都站得住" / "压根没收集"），在产物里长得一样；
  所以为空时要把逐项核过的那张清单印出来。核不过的项不会消失，它们变成章节里的条目——
  "没检查"不许伪装成"检查过且好"。
- **`can_persist=False` 不是"测不准"，是"不许记住"。** 它不进"无法确立"章节，
  而是在 best-known-good 里明写一行"这条没有进长期记忆"。
- **报告的数字只能来自不可变事件与已收口的 Observation**：
  事实数字由事件日志与 Observation 注入，结论状态由判决层生成，解释文字 LLM 可以润色，
  但**不能改变事实和状态**（§21.3）。

---

## 18. 待定问题

| # | 问题 | 影响面 | 现状 |
|---|---|---|---|
| 1 | 算子本身是否可能跨卡（通信 / TP / EP）？ | 决定 `TopologySpec` 与性能模型是否要引入 comm overlap | 开放 |
| 2 | 外部负载占卡的容忍策略：降级继续，还是排队等独占？ | 决定 `Placement` 的等待策略与 wall clock 预算 | 开放（§11.2-4 已定"诚实降级优于静默污染"，等待策略未定） |
| 3 | 停机后的续跑语义：从 best 继续，还是从信念状态继续？ | 决定 checkpoint 粒度与日志 schema | **已定**：从信念状态续跑，checkpoint 三元组 = `belief + best + ledger`（§23.2）；各轮的幂等键见 §23.3 |
| 4 | Triton 与 CUDA 是否共享同一套参数空间抽象？ | 决定 `param_space()` 的表达力与 bandit 的通用性 | 开放 |
| 5 | 策略层（§3.5）的载体与优先级：文件 / 环境变量 / 契约字段？ | 决定"哪些键可以配"能不能落地 | 开放 —— Wave 0 的第 0 项 |

---

## 19. 相关工作与可借鉴点（外部证据）

> 这一节是**外部证据**，不是本文的设计主张。所有数字来自 2026 年的公开工作，
> 用于检验 §1–§18 的取舍是否站得住。**注意：跨系统数字不可比**（见 §19.4）。

### 19.1 三条并存的路线

| 路线 | 代表 | 归因到的杠杆 | 关键数字 |
|---|---|---|---|
| 强模型 + 短闭环，**免训练** | [CudaForge](https://arxiv.org/abs/2511.01884) | **反馈设计** | KernelBench 250 题 / RTX6000：正确率 97.6%、平均 **1.677×**、Fast1 70.8% |
| **训模型** + agentic RL | [CUDA Agent](https://arxiv.org/abs/2602.24286)（字节 Seed + 清华） | 训练 + loop（**loop 权重最大**） | 通过率 98.8%，vs torch.compile **2.11×** |
| **确定性编排** + 并行采样早停 | [KernelFalcon](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) | **编排**（早停 / 上下文隔离） | KernelBench 250 题正确率 **100%**（未报加速比） |
| 修复优先（不重写） | [Don't Regenerate, Debug](https://arxiv.org/abs/2608.02712)（AscendC/NPU） | **修复闭环 + 知识库 + 全覆盖守卫** | Pass@1 **66.7%** vs 重写 25.9%@1 / 40.7%@3，**省 92.8% token** |

### 19.2 消融给出的归因（这是本节的核心）

**CudaForge——把"正确率"和"速度"的来源拆开了：**

| 配置 | 正确率 | 加速比 |
|---|---|---|
| 强模型单次 | 57.6% | 0.680× |
| 强模型自反思 | 90.8% | 1.107× |
| 强模型 + 纠正循环 | **97.6%** | 仅 1.222× |
| 强模型 + 优化反馈（无纠正） | 88.4% | **1.509×** |
| **完整（纠正 + profiler 优化反馈）** | **97.6%** | **1.677×** |
| 塞**全量** NCU 指标（对照） | 100% | 1.414×，且 40min / $1（完整配置 26.5min / $0.3） |

→ **纠正循环买正确率，优化循环买速度，两者不能互相替代；且原始 profiler 数据必须筛过再喂。**

**CUDA Agent——leave-one-out：**

| 去掉什么 | 通过率 | 加速比 |
|---|---|---|
| **去掉 agent loop** | **77.1%** | **0.69×** ← 掉得最狠 |
| 去掉 robust reward | 96.8% | 1.25× |
| 去掉 RFT 初始化 | 95.6% | 1.05× |
| 去掉 critic 价值预训练 | 98.6% | 1.00× |

→ **连"训模型"路线自己也承认：最大杠杆是 agent loop（harness），不是训练技巧。**

**Don't Regenerate, Debug——2×2 因子实验：**

- **方法主效应 +20.4pp > 模型主效应 +13.0pp**
- 去掉知识库 → 13/27；去掉全覆盖守卫 → 12/27
- 作者自注：各配对 McNemar 检验 p > 0.05，属该 27 算子队列上的**方向性观察**

**Atrex-Bench + AKA（阿里，[2607.14541](https://arxiv.org/abs/2607.14541)）——生产 trace + roofline 口径：**

- 最强模型（GPT-5.5 Sagg=0.107、Opus 0.059）**均未超过生产 kernel**
- **正确率被 PyTorch fallback 抬高**（benchmark 口径缺陷）
- 同一 base model 套 harness 前后：`chunk_gated_delta` 0.03→**1.20×**、`attention_forward` **6.7×**
- 其结论：**① 换 harness > 换强模型；② 强模型对 planner 语义不敏感；③ 最强模型 vanilla 仅约 10% roofline，套 harness 后反超**
- 作者自注：无组件消融，属受控 case study

**CUDAnalyst（[2605.26720](https://arxiv.org/abs/2605.26720)，ICML 2026）——归因分析层：**

- **单个工具/反馈的贡献接近零甚至为负**；增益来自**多个反馈的交互**
- 显式规划仅在反馈对齐时有益；摘要能帮弱模型但不替代规划
- 全部使用开源模型（无闭源强模型）

### 19.3 对本文设计的检验

**被外部证据印证的：**

| 本文主张 | 外部证据 |
|---|---|
| §4.3 检查 ④ 必须检测 lib fallback | Atrex **实测**：正确率被 PyTorch fallback 抬高 |
| §9.2 两级 oracle，别把原始 profiler 灌给模型 | 全量 NCU 1.414× vs 筛 24 指标 1.677× |
| §8 性能不达标 ≠ bug，走搜索而非修复 | CudaForge：纠正 1.222× / 优化 1.509×，不可互换 |
| §1.2 迭代与产物必须是端到端可判定的 | Atrex：单 kernel 均匀平均 + 软件基线偏离生产 |
| §15 "先可信、再可搜、最后可生成" | 三篇独立工作一致指向 **harness 是主要杠杆** |

**需要修正或补强的：**

1. **"强模型 + 简单 agent 够用"被否。** 三篇均不支持。推论：§5 提议层要加一条硬约束——
   **不新增任何不与现有反馈发生交互的工具/角色**（CUDAnalyst：单工具贡献近零或为负）。
2. **near-miss 走修复而非重写。** §8「正确性失败」行要补：**编译通过、仅数值失败时，修 > 重写**
   （省 92.8% token，且方法效应大于模型效应）—— 这正是 §8.2 误差形态要服务的那条路由。
3. **全覆盖守卫是有效组件。** 去掉后 12/27（Don't Regenerate）→ 支持 §13.2 的能力声明 +
   §4.3 的静态审计 / 反作弊**必须保留**，不能作为"简化"砍掉。
   **但静态守卫不是唯一防线**：它对宏、头文件、动态加载和运行时作弊无能为力 ⇒
   它只能影响声明等级与晋级，**不能替代执行沙箱**（§22）。
4. **"换 harness > 换强模型"意味着预算分配要偏向 harness 工程**，而不是偏向换更大的模型。

### 19.4 两条必须保留的怀疑

1. **跨系统数字不可比。** KernelFalcon 只报正确率**不报加速比**；CudaForge 报 1.677×；
   CUDA Agent 报 2.11×。基线、容差、硬件、评测协议各不相同。那篇
   [IJCAI 2026 survey](https://arxiv.org/abs/2601.15727) 自述：**没有跨系统对比表，没有任何消融**，
   其引用数字"非受控对比，不能归因"。这恰好再次证明 §3 冻结契约与 §11 同卡配对基线的必要性。
2. **§4 那层"资格审计"在上述任何工作中都没有对照。** 既可能说明它是真空白（差异化机会），
   也可能说明它在主流的**同卡、短程、单次评测**协议下并不必要——噪声被评测口径掩盖了。
   本文的判断是：它在你确定要做的**多卡、长跑、无人值守**场景下会重新出现，
   但在 Stage 1 验收（§14）通过之前，不应对外声称它带来了收益。

---

## 20. 正确性轴与候选晋级门禁

> 本节是新增平面。放在 §20 而不是插进 §4，是为了不重排已被代码引用的编号（§0.2）。

**原则：正确性与性能可信度同级。** 性能测量做得再严，只要正确性不是同一道硬门禁，
"又快又错"就与"能跑对但慢"在流水线里长得一样 —— 而前者是**回归**，不是进展。

### 20.1 正确性测试套件是契约的一部分

`CorrectnessSuite`（§3.3）至少覆盖：

```
基础随机输入 · 边界 shape · 非对齐尾块 · batch=1 / 极大 shape / 极小 shape
不同 dtype 与累加精度 · 非 contiguous stride · broadcast
零 / 常数 / 极大值 / 极小值 · NaN / Inf · 确定性 · aliasing / in-place
backward / gradient（若契约要求）· sanitizer / racecheck（若后端支持）
```

三条规范：

1. **形状集必须每类边界各一**，且候选须在**全部**形状上通过（§3.4 条款 2）——
   这是唯一能挡住"针对测试 shape 特化"的东西。
2. **容差只能收紧不能放宽**：任务自带的 `max_error` 与分精度容差表取**更严**的那一侧；
   驱动传进来的宽松值变成 no-op，而不是继续放行错核。
3. **`correctness` 是三态**（通过 / 失败 / 无法确立）：没有参考实现、参考侧自己非有限、
   或守卫不适用于这份源码时，是"无法确立"而**不是**"通过"。

### 20.2 错误候选清单（必须被拒）

验收中必须有明确的错误候选（§14.1 第 10 条）：

```
数值整体缩放 · 尾块越界 · 只对特定 shape 正确 · 忽略 stride
错误 broadcast · 静默 fallback（回退到库）
```

> **目标不是测试数量多，而是确保：错误候选不能进入正式性能比较，更不能替换 `best`。**

### 20.3 特化与空转探针

只靠"多形状套件"是**未被声明的副业**；探针必须显式存在，并且**结果进报告**（不是内部 debug）：

```
输入 shape 扰动 · 输入值扰动 · layout / stride 扰动
候选输出扰动检测 · kernel launch 路径检查 · 库函数 fallback 检查
```

探针的两种结论都必须可表达：**"没发现问题"** 与 **"这个探针不适用于这份候选"** ——
后者不许被读成前者（§4.3 规则 8 的同一条纪律）。

### 20.4 晋级门禁（不变量）

```
G2 正确性未通过（或无法确立） ⇒ 不产生可用于晋级的性能观测，ClaimLevel 不得 ≥ VALIDATED
篡改 contract / harness          ⇒ 候选被拒，不是"降级"
关键 shape 回退越限               ⇒ 该候选不得晋级（§3.4 条款 3）
```

---

## 21. Agent Runtime：结构化动作与权限

> 本节是新增平面。它把"LLM 是受约束的提议器"从原则变成协议；
> Stage 4 落地时若没有它，必然退化成"不可控的长 Prompt + 自由改代码"。

### 21.1 Agent 只提交结构化动作

Agent **不直接操作** Conductor 的内部状态，只提交动作：

```python
@dataclass(frozen=True)
class AgentAction:
    action_id: str
    parent_candidate: CandidateID | None
    kind: Literal[
        "propose_template", "propose_source_patch", "repair_compile",
        "repair_correctness", "request_profiler", "request_new_shape", "explain_failure",
    ]
    patch: Patch | None
    rationale: str
    expected_bottleneck: Bottleneck | None
    requested_tools: tuple[str, ...]
    token_budget: int
```

控制器**只接受**通过 schema 校验、权限校验与预算校验的动作。
"自由文本 + 直接改状态"没有入口 —— 这是可审计与可回滚的前提。

### 21.2 结构性变更与局部修复的边界

§8.1 的路由在这里成为**动作权限**；错误形态（§8.2）是"修 vs 重写"的直接依据：

```
编译错误        → repair_compile（局部修复优先）
数值错误        → repair_correctness（局部优先，必须带误差形态）
边界错误        → repair_correctness，或回退父代
性能不达标      → 进入搜索，不默认改代码
算法瓶颈        → 允许结构性变更（仍受 §10.3 三条件门禁）
接口 / ABI 破坏 → 拒绝候选，回退父代
```

### 21.3 上下文只使用结构化证据

每次 LLM 调用**只**提供：

```
当前契约摘要 · 父候选 diff · 最近一次失败分类 · 首个错误与误差形态
相关 profiler 指标 · 可用工具列表 · 剩余预算 · 允许修改的文件
```

**不要把完整历史日志无条件塞入上下文**（外部证据：全量 NCU 指标 1.414× vs 筛过 1.677×；
单个工具贡献近零甚至为负）。报告生成同样如此：

```
事实数字：由事件日志与 Observation 注入
结论状态：由判决层生成
解释文字：LLM 可以润色，但**不能改变事实和状态**
```

### 21.4 进编译之前的独立检查

LLM 生成的候选在进入编译前依次检查，任一项不过即拒绝：

```
文件写入范围 · patch 是否只修改授权路径 · contract / harness / baseline 是否被修改
依赖是否新增 · ABI 是否变化 · 候选是否与父代重复 · 是否引用不存在的能力
```

"引用不存在的能力"（幻觉 API）必须在**零 GPU 时间**的成本上被拦住（§13.2 ① 的能力声明）。

---

## 22. 候选执行与构建隔离

> 本节是新增平面。静态守卫能拦住库调用，拦不住文件系统、网络与进程 ——
> **静态检查不是唯一防线，也不能替代沙箱。**

### 22.1 目录权限：一次运行的布局

```
run_root/
├── contract/       # Agent 只读
├── harness/        # Agent 只读
├── baseline/       # Agent 不可见或只读 artifact
├── candidate/      # Agent 只写
├── build/          # 构建进程写
├── measurements/   # Oracle 写
└── report/         # Report Writer 写
```

候选**不能**写入 `contract/`、`harness/`、`baseline/`、`measurements/`、`report/`。
这条不是"约定"，而是权限位：违反时写入必须**失败**，而不是"事后被审计发现"。

### 22.2 进程与资源隔离

至少需要：

```
编译超时 · 运行超时 · GPU watchdog · CPU / memory quota
文件系统白名单 · 网络默认关闭 · 进程树清理 · artifact 大小限制
设备租约超时与回收
```

**为什么这是不变量而不是优化**：候选代码在评测进程里运行时，它与 harness 共享地址空间、
文件系统与网络。没有隔离，"候选不能修改 harness"就只能靠候选自觉 ——
而无人值守下没人会检查它是否自觉。

### 22.3 静态检查的覆盖必须显式

不同语言/形态的静态检查能力不同（Python AST / typed AST / 只能正则 / 完全没有）。
审计输出必须把覆盖情况表达成数据，而不是"有/没有问题"：

```python
@dataclass(frozen=True)
class AuditCoverage:
    language: str
    mode: Literal["ast", "typed_ast", "regex_only", "none"]
    covered_rules: frozenset[str]
    uncovered_rules: frozenset[str]
```

**coverage 不足只能影响 `ClaimLevel` 与 promotion，不能替代执行沙箱。**
（三态纪律：`mode="none"` 是"无法确立"，不是"通过" —— §4.3 规则 8。）

---

## 23. 持久化、恢复与 exactly-once

> 本节是新增平面。无人值守长跑的核心风险不是崩溃，而是**崩溃恢复后重复计费、重复学习、
> 或覆盖状态** —— 一个"看起来合理地继续跑"的恢复，比一次干净的失败危险得多。

### 23.1 append-only 事件日志

每个重要动作记录事件；日志只追加，不改写：

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

事件类型至少包括：

```
RUN_CREATED · CONTRACT_FROZEN · CANDIDATE_PROPOSED · BUILD_RESERVED · BUILD_COMPLETED
CORRECTNESS_COMPLETED · MEASUREMENT_RESERVED · MEASUREMENT_COMPLETED
OBSERVATION_COMMITTED · BELIEF_UPDATED · CANDIDATE_PROMOTED
CONFIRMATION_COMPLETED · MEMORY_ADMITTED · RUN_ABORTED · REPORT_EMITTED
```

**`*_RESERVED` 先于动作、`*_COMPLETED` 后于动作**：没有"预留"这一格，
"这次 GPU 测量是否已经计费"在崩溃后就无法回答。

### 23.2 续跑语义（回答 §18 #3）

**从信念状态续跑，不是从 `best` 续跑。** checkpoint 是三元组：

```
checkpoint = (belief, best, ledger)
```

- 只恢复 `best` 会丢掉"哪些点试过、哪些配方被禁用、停滞计数到几轮" ⇒ 恢复后重复探索；
- 只恢复 `belief` 会丢掉"不可复现的最好版本" ⇒ 交付物消失；
- 只有 `ledger` 一起恢复，"预算没被重复花"才成立。

各轮的幂等键：

```
measurement_id · build_id · promotion_id · memory_admission_id
```

### 23.3 exactly-once 语义

恢复逻辑必须能回答：

```
这次 GPU 测量是否已经计费？ · 这次 observation 是否已经进入 bandit？
这次确认是否已经完成？ · 重启后是否允许重新执行？ · 重复事件是否会改变结果？
```

**规范**：每个副作用都必须幂等；重复事件**不得**改变结果（不是"通常不会"）。
预算与晋级是仅有的两个不可逆副作用，因此它们的幂等键必须在动作**之前**落盘。

缓存键见 §9.3 与 §9.4（`BuildKey` / `PartitionKey` / `ToolchainKey` / `MeasurementProtocolKey`）。

---

## 24. 记忆治理：分层、有效期与撤销

### 24.1 记忆分层

```
Run Memory            当前任务的候选、失败和信念（随 run 结束而失效）
Partition Memory      某算子 / shape / 硬件分区上的有效配置（绑 PartitionKey）
Transferable Knowledge 算法模式、DSL 陷阱、编译器缺陷（跨分区，但要声明适用范围）
Negative Knowledge    已验证的失败签名和适用范围
```

### 24.2 每条长期记忆必须带有效期

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

驱动、库、编译器、架构或 harness 变化后，相关记忆自动进入 **`STALE`**，
而**不是**继续作为强先验使用。`invalidated_by` 让"错误经验"可被撤销，而不是只能被覆盖。

### 24.3 promotion policy：什么才算知识

只有"赢过当前 `best`"**不能**形成可迁移知识。进入长期记忆至少需要：

```
正确性通过 · 测量资格足够 · 独立重测通过（§10.5-4）
没有关键 shape 回退 · 硬件 / 工具链分区明确 · 报告中声明适用范围
```

对应 `ClaimLevel.PERSISTED`（§4.4）。**没有 TTL 与撤销机制的记忆层，
在长期无人值守下会把一次偶然胜利放大成跨任务的系统性错误。**

---

## 25. 文档治理：规范 / 状态 / 研究日志

### 25.1 三份文档

```
docs/design/autonomous-operator-agent-design.md                    ← 规范（本文）
docs/status/autonomous-operator-agent-status.md                    ← 当前实现与真机状态
docs/design/autonomous-operator-agent-optimization-directions.md   ← 优化方向、波次与迁移
docs/research/                                                     ← 外部证据、实验记录、CHECKLOG
README.md                                                          ← 路由：新人第一眼，上面几份的分工与入口
```

### 25.2 设计正文不许出现的内容

```
"上次误判已撤回" · "今天还没有接上" · "某次 CHECKLOG 修复了什么" · "某个实现缺口后来已补齐"
```

这些内容属于状态或研究日志，**通过链接引用**。设计正文只保留当前规范、当前接口、
当前不变量、当前验收要求。

**为什么这是治理而非洁癖**：行为变更与状态更新共用一个文件 ⇒ `git diff` 不可读 ⇒
review 事实上会转移到研究日志；而**当更正需要写进历史（而不是覆盖它）时，
说明承载机制选错了地方** —— 读者必须读完几百字才能知道现在的口径是什么。

### 25.3 ✅ 必须从验收测试派生

实现状态的"✅"必须写成"`tests/xxx.py::test_yyy` 存在且通过"，
**而不是**一句人写的段落。这样"落地状态"这个概念**不可能腐化**："已落地"要么有守卫，
要么不成立。状态文档的记录口径见其自身说明。

---

## 26. 优化波次

Stage（§15）描述**能力顺序**；Wave 描述**优化工作波次**，两者不复用编号：

```
Wave 0  规范收口与安全默认值      → 同一问题在规范、代码、报告中只有一个语义
Wave 1  正确性与候选晋级门禁      → 错误候选无法替换 best、无法进入长期记忆
Wave 2  统计判决与多 shape 目标   → 相同原始样本可离线重算判决
Wave 3  Agent Runtime 与执行沙箱  → Agent 不能改变评分规则、不能修改评测基础设施
Wave 4  事件日志、恢复与多卡调度  → 任意阶段杀死进程都可恢复且不重复晋级
Wave 5  跨任务记忆与结构性优化    → 跨任务复用收益有独立对照；错误记忆可撤销
```

每一波的任务清单、前置依赖与退出标准，以及"不建议现在做的事情"清单，见
[`autonomous-operator-agent-optimization-directions.md`](autonomous-operator-agent-optimization-directions.md) §12–§13。

**Wave 0 的第 0 项是策略层载体**（§3.5 规则 1 / §18 #5）：在定义"哪些键可以配"之前，
必须先定义配置的载体与优先级。否则"安全不变量默认开启"与"不许一键关安全门"两条都无处落脚。

