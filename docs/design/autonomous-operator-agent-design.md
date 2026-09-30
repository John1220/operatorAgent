# 无人值守算子开发 Agent —— 设计文档

| 项 | 值 |
|---|---|
| 状态 | 设计草案（接口未实现） |
| 目的 | 给出一个可自动运行、自主搜索并产出**可审计结论**的 GPU 算子开发 agent 的架构与接口约定 |
| 独立性 | 本文不依赖任何既有实现；所有接口为从零定义 |
| 运行模式 | **无人值守**（autonomous）：无人在环内裁决歧义、发现失效 |
| 目标硬件 | **多卡**（同构与异构并存） |
| DSL 边界 | 初期 Triton / CUDA；后端接口预留 **TVM / CUTLASS 模板型后端** |

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
| **每个动作必须可恢复** | 没有人类来收拾残局 | 全程 checkpoint + 幂等 + 断点续跑 |
| **歧义必须失败关闭** | 测不准时它只能"猜"，而猜出来的数会污染记忆 | "无法可靠测量"是一等结论，触发停止而非继续 |

> 无人值守系统里，**最危险的不是报错，是自信地报一个错的数**。

---

## 2. 总体架构

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
                                           │ 观测（值 + 不确定度 + 溯源 + 资格位掩码）
                                           ▼
                ┌──── 判决层 Adjudicator / 信念状态 ────┐
                │ 与同卡配对基线比 → FAST/SAME/SLOW/AMBIGUOUS │
                │ 更新信念 / 失败归类 / 决定下一步动作         │
                └──────────────────────────┬────────────────┘
                                           ▼
                预算控制器：继续搜索 / 收敛收工 / 不安全中止 / 交还给人
                                           │
                                           ▼
        产物：best-known-good + 不可变实验日志 + 报告（含"无法确立"章节）+ 复现命令
```

---

## 3. 契约层：把模糊需求变成可机器判定的东西

契约必须包含，且缺一不可：

1. **参考实现**（最重要的一件产物）：同时定义正确性、测试数据生成、性能基线。
2. **数值语义显式钉死**：fp32 是 TF32 还是全精度、累加精度、是否允许 split-K 的非确定性。
   — 这一步不做，后面所有加速比都可能是 2x 的假象。
3. **形状分布 + 权重**：目标是 workload profile（含边缘形状：非对齐尾块、batch=1、极大/极小 k），
   优化目标是**加权几何平均**，而不是单点。
4. **基线定义方式**：不是"查表得到的数字"，而是"同一套 harness、同一时刻、同一张卡、同样数值语义下对手跑的"。
5. **目标指标 + 阈值**：吞吐 / 延迟 / 显存，以及"达到什么程度算收工"。
6. **拓扑约束**（多卡算子）：world size、通信后端、拓扑；决定性能模型是否仍适用 roofline。

契约在 `CONTRACT_LINT` 通过后写入只读文件并哈希；评测器每轮校验哈希。

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
4  检测到 lib fallback                          → 全 False（它在测别的东西）
5  value > 实测峰值 × 1.05                      → UNINFORMATIVE，全 False
6  cv 超阈值 或 launch 下限占比超阈值            → DEGRADED：
        can_verdict = can_tune = can_report = True
        can_persist = can_count_progress = False          ← 关键
7  partition 漂移（驱动/库/卡变了）               → DEGRADED，跨漂移比较一律作废
8  其余                                         → VALID，全 True
```

两条设计意图：

- **低资格 ≠ 没测准，而是"这个数不足以支撑结论"**。`can_persist` / `can_count_progress`
  只在 VALID 放开，因此一次噪声读数**永远不能永久拉黑一个配方，也永远不能算作收敛证据**。
- **第 3 条与第 6 条刻意不同**：降级是"数不可信"，第 3 条是"数不存在"。
  后者必须直接拒绝 `passed`，因为它测的根本不是契约要求的东西。

> **核心原则：资格决定"这个数被允许造成什么后果"，而不只是"过没过"。**

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

---

## 6. 判决层：显式信念状态

- 与基线比较输出四态：`FAST / SAME / SLOW / AMBIGUOUS`，噪声带随资格等级变化。
  **只有 `SLOW`（且资格足够）才允许改配置或拉黑。**
- 维护**假设图 / 实验日志**：每个候选为何被生成、测出了什么、因此更新了什么信念。
  这既是决策依据，也是最终给人看的证据链。
- **停滞计数只在有效轮次上累加**（被污染的一轮不算进度，也不该结束循环）。

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
- 任何永久结论需**两次确认**（见 §10.5）。

---

## 8. 失败分类与修复策略（价值密度最高的一层）

任何修复动作之前，先做**失败归类**，不同类走完全不同的策略：

| 类别 | 表现 | 策略 | 是否算迭代 |
|---|---|---|---|
| 编译失败 | 类型 / layout / DSL 误用 | LLM 拿错误 + 代码修，便宜 | 是 |
| 正确性失败 | 数值错、边界错 | 调试模式：缩小形状、打中间量、与参考逐块对比 | 是 |
| 性能不达标 | 能跑对但慢 | **搜索失败，不是 bug**：换配置 / 换结构，别去"修代码" | 是 |
| 基础设施 | SSH / 驱动 / OOM | 退避重试，**绝不污染学习，也不计进度** | 否 |
| 静默退化 | 全过，但不是你以为的那个东西 | 靠 §4.3 的溯源检查拦住（最危险的一类） | 否 |

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

### 9.3 缓存

| 缓存 | 键 | 能否跨卡共享 |
|---|---|---|
| 编译产物 | `BuildKey(backend, arch, source_hash, flags)` | **能**（同 arch 共享） |
| 测量结果 | `(candidate_hash, PartitionKey)` | **绝对不能**（时钟 / 温度 / 竞争不同） |

把这两者混成一个键，是多卡系统最常见的错误。

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

- `CONTRACT_LINT`：参考实现存在吗？容差能区分对错吗？形状分布覆盖边缘形状吗？
  数值语义钉死了吗？基线可复现吗？后端能力满足吗？
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

### 10.3 每轮的决策策略

候选选择用 bandit，不用 LLM：

```
score(c) = P(beats_best | 观测) × 预期增益 / 预期 GPU 秒

选择：
  ε 概率 → 最不确定 / 未探索区域（信息优先）
  1-ε    → 期望增益最高（收益优先）
```

**何时才升级到 LLM 做结构性改动**（三条件同时满足，否则不许调用）：

1. 参数搜索已经平台化（K 轮 valid 且 SAME/AMBIGUOUS）；
2. 平台化有**可归因的硬限制**证据（spill-bound、smem-bound、占用率上限、bank conflict 报表）
   —— 不是"感觉调不动了"；
3. 还存在一个可探索的结构方向（融合 / split-K / 算法替换）。

> 这条门禁很重要：LLM 在自动跑里最大的价值是**结构性跳跃**，
> 最大的风险是被当成万能调参器反复重写同一份代码。

### 10.4 停止条件（自动跑最难的部分）

| 条件 | 判据 | 备注 |
|---|---|---|
| **达标** | 达到目标指标，测量 `valid`，且**重测确认** | 重测是硬要求 |
| **预算耗尽** | 任一预算触顶 | 交付 best-known-good，**不装作成功** |
| **可靠停滞** | K 轮 valid 无改善，且**有希望的区域已穷尽** | 必须"穷尽"才算，不能只因轮数 |
| **roofline 收敛** | 利用率接近峰值，**且峰值分母已标定** | 峰值未标定 → 此出口**永久关闭** |
| **不安全中止** | 心跳失败 / 环境漂移 / 连续基础设施失败 | 中止且**不写入任何记忆** |

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

4. **永久结论需两次确认**
   任何进长期语料库的结论（"这个配方在这张卡上更快"）必须：
   一次 `valid` 且 FAST **+** 一次独立重测仍然不劣。

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

    c = propose(contract, belief, budget)              # bandit 优先；LLM 需门禁
    if proxy_prune(c): continue                        # 便宜代理剪枝，不花 GPU 秒

    obs = oracle.evaluate(c, contract)                 # 值 + 不确定度 + 溯源 + 资格
    if obs.eligibility.can_tune:
        belief.update(c, obs)                          # FAST / SAME / SLOW / AMBIGUOUS
    if obs.eligibility.can_count_progress:
        stagnation.update(belief)

    if obs.eligibility.can_persist and beats(best, obs):
        if confirm_by_remeasure(c, reserve_budget):    # §10.2 的 10%
            best = c                                   # 只有这里能改写 best

    checkpoint(belief, best)                           # 幂等 + 可续跑

report(best, log, unable_to_establish=[...])           # 显式列出没确立的东西
```

---

## 11. 多卡：干扰域与同卡配对

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

放置策略（按优先级）：

1. **同域串行，异域并行。**
   不同 `interference_domain` 的候选可并行以省 wall clock；同一域内**必须串行**
   —— 否则共享 L2 / 时钟域会互相污染，而 cv 会诚实地变大，把观测打到 DEGRADED。

2. **同卡配对（paired baseline）。**
   一个候选的基线必须租到**同一张卡、同一时间窗**。
   **A 卡基线 vs B 卡候选的"加速比"直接作废**——多卡下最容易犯、也最难发现的错。

3. **交错测量。**
   candidate / baseline **交替 + 每轮 L2 flush**，而不是"先跑基线，再跑候选"，
   否则热/时钟状态变化会污染比较。

4. **竞争检测。**
   租不到独占卡时，记录 `external_load` 并在 `evidence` 中显式标注；观测降为 DEGRADED
   而不是失败——**诚实降级优于静默污染**。

5. **异构卡。**
   不同 `arch` → 不同分区、不同编译缓存、不同记忆。`best` 只在同一 partition 内有效；
   跨 arch 迁移是**显式声明的操作**，绝不隐式共享。

6. **多卡算子。**
   若算子本身跨卡，契约加 `TopologySpec`；性能模型变成 compute / comm overlap，
   **roofline 分母不再适用**，需额外维度衡量重叠率，基线也必须跨卡配对测。

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


@dataclass(frozen=True)
class Observation:
    metric: str
    value: float
    unit: str
    stat: Dispersion                   # median / p10 / p90 / cv / n
    provenance: Provenance
    validity: Validity
    eligibility: Eligibility
    evidence: list[str]                # 判据来源，人可读
    reasons: list[str]                 # 降级 / 不可用的原因
```

资格推导见 §4.3。

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

### 13.5 放置与标定

```python
@dataclass(frozen=True)
class DeviceLease:
    device: DeviceKey
    exclusive: bool
    external_load: float       # 探测到的外部占用


class Placement(Protocol):
    def lease(self, contract: Contract, n: int) -> list[DeviceLease]: ...


class Calibrator(Protocol):
    def probe_peak(self, device: DeviceKey) -> PeakProfile | None   # 实测峰值，roofline 的分母
    def heartbeat(self, device: DeviceKey, epoch: int) -> HeartbeatResult
```

- **`probe_peak` 必须实测**：拿一个已知好 GEMM 打到真实峰值，而不是读 spec 表。
  探测失败 → **roofline 收敛出口永久关闭**。
- **`heartbeat` 逐卡独立**：一张卡被降频 / 被占，不代表别的卡坏。
  每个 `Observation` 记录自己的 `calibration_epoch`，失败窗口内的观测全部失效。

---

## 14. Stage 1 验收标准

这是**标定"标定器"本身**的测试集。全绿才允许往上盖 Stage 3（搜索）与 Stage 4（LLM 提议）。

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

---

## 15. 实施顺序（顺序是刻意的）

```
Stage 0  契约冻结 + 只读哈希护栏 + 预检 / 峰值标定
Stage 1  评测器 + 资格模型 + 标定心跳 + checkpoint/幂等/续跑   ← 尺子（含尺子的自检）
Stage 2  多卡放置：干扰域 + 同卡配对（若一开始就多卡，则与 1 并行）
Stage 3  参数模板 + bandit 搜索 + 停止条件
Stage 4  LLM 结构提议（带三条件门禁）
Stage 5  记忆与跨任务迁移（带两次确认）
```

**先可信、再可搜、最后可生成。**

反过来做（先搭一个很酷的多 agent 生成流水线）几乎必然得到一个
"每次都能产出 kernel、但没人知道数是真的假的"的系统。

> 注：若确定从一开始就是多卡，Stage 2 不是"后期工程"，而应与 Stage 1 同期，
> 因为"基线同卡配对"是评测器语义的一部分，不是外挂的调度优化。

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

---

## 17. 交付物（一次自动跑结束）

1. **best-known-good**：代码 + 完整配置（永不因未确认的结果被覆盖）
2. **不可变实验日志**（JSONL）：每次尝试、每个读数的溯源与资格
3. **报告**：结论 + 证据链 + **"我无法确立的事"**（一等公民章节）
4. **复现命令** + 硬件 / 驱动 / 库指纹

第 3 条的"无法确立"章节是自动跑系统的信誉所在。
**一个从不说"我测不准"的自动系统，它的所有数字都不可信。**

---

## 18. 待定问题

| # | 问题 | 影响面 |
|---|---|---|
| 1 | 算子本身是否可能跨卡（通信 / TP / EP）？ | 决定 `TopologySpec` 与性能模型是否要引入 comm overlap |
| 2 | 外部负载占卡的容忍策略：降级继续，还是排队等独占？ | 决定 `Placement` 的等待策略与 wall clock 预算 |
| 3 | 停机后的续跑语义：从 best 继续，还是从信念状态继续？ | 决定 checkpoint 粒度与日志 schema |
| 4 | Triton 与 CUDA 是否共享同一套参数空间抽象？ | 决定 `param_space()` 的表达力与 bandit 的通用性 |