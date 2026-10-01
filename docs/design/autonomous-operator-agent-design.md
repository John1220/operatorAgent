# 无人值守算子开发 Agent —— 设计文档

| 项 | 值 |
|---|---|
| 状态 | **Stage 0–1 已落地**（见文末「落地状态」）；Stage 2+ 仍为设计 |
| 目的 | 给出一个可自动运行、自主搜索并产出**可审计结论**的 GPU 算子开发 agent 的架构与接口约定 |
| 独立性 | 本文不依赖任何既有实现；所有接口为从零定义 |
| 运行模式 | **无人值守**（autonomous）：无人在环内裁决歧义、发现失效 |
| 目标硬件 | **多卡**（同构与异构并存） |
| DSL 边界 | 初期 Triton / CUDA；后端接口预留 **TVM / CUTLASS 模板型后端** |

---

## 落地状态（`design/autonomous-spine` 分支）

| 设计节 | 产物 | 状态 |
|---|---|---|
| §3 契约冻结 + 哈希 | `operator_agent/contract.py`（含 `freeze_from_task` / `load_dict` 校验载荷 hash） | ✅ |
| §4.3 五位资格 | `operator_agent/eligibility.py` | ✅ |
| §10.5-1 标定心跳 | `operator_agent/calibration.py`（成对参照双比值） | ✅ |
| §10.5-4 两次确认 | `operator_agent/confirm.py` + 唯一判据 `oracle.admit_to_corpus` | ✅ **但是 opt-in**：判据默认要求重测，生产把它接到 `require_double_confirm`（默认 `False`，机时承诺）⇒ **默认配置下单次读数即可入库**；理由文本里会写明本次没要求 |
| §13.4 Oracle 收口 | `operator_agent/oracle.py`，已接 **L3 出口**（`verifier._assess_measurement`） | ✅ |
| §8 失败分类路由 | `operator_agent/failure_route.py`（五路 + `fix_not_rewrite`） | ✅ |
| §14-9 资格审计对照 | `benchmarks/ablation_eligibility.py` + `tests/test_ablation_eligibility.py` | ⏳ **比较器已落地、两臂未跑**：五态判据（含"没打到点上"这一态与"没有收益必须写进报告"），带宽取 `NOISE_BAND[uninformative]`；真机对照需要同卡两次完整循环（要 LLM）⇒ "审计有收益"仍不许声称 |
| §17「无法确立」 | `confirm.UnableToEstablish` + **`report.py` / `report.md` 产物** | ✅ 2026-10-01 补齐：第 3 件交付物此前完全没有产物（收集器只被两个 benchmark 调过）。空章节带依据、`can_persist=False` 单列一行、入库拒因随报告落盘 |
| §14 验收八条 | `tests/test_stage1_acceptance.py`（离线） + `benchmarks/smoke_spine_h20.py`（真机 H20） | ✅ **真机已跑**（2026-10-01，8×H20 的 `algorithm-h20`，走生产 L3 入口）：0/1/2/3/7/8 绿、4 见下、5 未覆盖（需 LLM，`LLM_API_KEY` 未设）、6 纯离线。**真机抓出两处「算了但没接上」**（CHECKLOG I46 发布的资格位不是收口点算的那份 ⇒ §14-8 在生产上没生效；I48 远端运行时盾的证据不落地），都已修 + 守卫；§14-7 的运行时检查**已接上喂给者**（I47 同卡那半 2026-10-01 修：远端按臂采身份 ⇒ `measurement_scope` 由观测算出三态、跨卡时资格全 False 且 `passed=False`；**窗口那一半仍无喂给者**，明写在 I47）。见 [docs/research/63](../research/63-真机§14冒烟与它抓出来的两个真源.md) |
| §11 多卡放置 / §9.3 双键 | `operator_agent/placement.py`（DeviceKey/干扰域/同卡配对/竞争降级） | ✅ **同卡配对已有观测喂给者**（2026-10-01：远端逐臂报卡 ⇒ `observed_pairing` 三态 ⇒ oracle 拒评 + `passed=False`；`placement.plan_leases`/`DeviceLease` 仍**零生产调用者**，见 §15「没有实现物」表） |
| §13.5 峰值实测（`probe_peak`） | `operator_agent/peak_probe.py` + `hardware_caps` 存放处 + `verifier._publish_peak_probe`/按精度读取 | ✅ 2026-10-02：判据七条（仿真读数 / 精度键 / 出处 / 非有限正数 / 探针自己的档位 / 离散度未知 / 没打到算力峰值 / 高于规格 5%）+ **按精度**发布进 `HardwareCaps.measured_peak_tflops`（字典）+ 收敛出口按**这一轮分母的精度键**读取；零额外 GPU 秒（复用同一次 L3 的基线臂）。**边界**：发布的是**下界**（基线臂的实测可达值，不是器件极限）⇒ 出口偏松；只在**本进程**内生效（不跨运行）；`MockGPU` 一律拒发（无 `REAL_DEVICE`）；真机**未复跑**（本机无 GPU）—— 见 [doc 70](../research/70-实测峰值的判据与按精度发布.md) |
| §10.2 三种预算 | `operator_agent/budget.py`（`BudgetPlan` + `BudgetLedger`）+ 轮次开头两道门 | ✅ 2026-10-02：oracle 次数（**预留不可挪用**）/ 墙钟 / token 三条接上并落 `artifact.metadata["budget"]`；心跳只记账不被拦。**边界**：`explore`/`refine` 不分相、按 GPU 秒未做、预算默认全关（opt-in）—— 见 [doc 67](../research/67-三种预算与不可挪用的预留.md) |
| §10.3 候选选择 / LLM 升级门 | — | ⏳ **半片**：三条件升级门**已落地**（`escalation.py` + conductor 判决点接线 + SearchAgent 发布通道，13/13 变异红，CHECKLOG I53→I57/doc 69）—— 条件 2 今天只有 `smem-bound` 一条量具、且只在上界~×1.2 那条带里成立；`score = P(beats_best)×预期增益/预期GPU秒` 与 ε-greedy **未做**（要 GPU 秒口径） |
| §5 bandit 调参 | `operator_agent/bandit.py`（确定性 UCB1）+ `Conductor` 信念门 | ✅ 2026-10-02：`perf_data` / `optimization_history` / 回报统计三条信念通道按 `can_tune` / `can_count_progress` 收口（CHECKLOG I52），prompt 不再把「没有读数」渲染成 `0.00`；状态落 `artifact.metadata["bandit"]`。**臂空间只有 tiling**、`c` 未标定、证据**不进快照**（只覆盖本次运行）；**真机未复跑** —— 见 [doc 66](../research/66-搜索侧的信念门与bandit.md) |
| §12 模板后端 / `param_space()` | `operator_agent/candidates.py` + `backend_caps.Backend.param_space` + 便宜代理剪枝 | ✅ 2026-10-02 **枚举半片**：候选联合类型、分开写的去重口径（α-等价签名 / 模板参数+实例化器版本）、`ParamSpace` 确定性枚举、`prune_by_smem`（复用 L1 的 smem 估算器与同一个 `SMEM_SLACK`）、开关 `param_space_of` + fail-closed。**边界**：模板候选的**实例化 `build` 未实现**（声明了参数空间 ⇒ 控制器拒绝，不降级）；剪枝只估单缓冲；上界缺 ⇒ 三态不剪；参数轴只有 tiling 三维有产出者 —— 见 [doc 68](../research/68-候选联合类型与便宜代理剪枝.md) |

**审查**：`spine-review.md` 记了 P0–P3；P0 三项、P1-4/P1-5/P1-6/P2-8/P2-9/P2-10 与 P3-11/12/13/15/16/18 已修；P3-14/P3-17 仍未修。**入库判据只有一处**（`oracle.admit_to_corpus`，2026-10-01 收口：此前它零生产调用者，`_save_to_db` 自己又写了一遍那三道门且口径不同）。  
**尚未宣称收益的那道门已过**（§19.4）：设计原文是「真机 §14 冒烟之前，不对外说资格审计带来收益」——
2026-10-01 真机 H20 跑完（doc 63），**这道门确实值钱**：八条里两条 ❌ 都不是逻辑错，而是「算好的判定没接到
发布出去的那一格」（发布的资格位取错了来源、远端盾的证据只活在错误文本里）。⇒ 现在可以说的只有这两句：
① 真机上「坏窗口的读数」不再自称可入库（同一判据、同一参数、两臂只差心跳，坏窗口被拒 / 好窗口被收）；
② 这条链路上的「算了一处、判在另一处」在这个真机场景里已被消掉并钉住。**收益本身仍未量化**（§14-4 的
竞争降级在真机上没造出竞争：共享卡上「后台占卡」这一臂缺一次成功证据；§14-5 未覆盖）。

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

> **实现状态（2026-10-02）**：三条预算的账本落在 `operator_agent/budget.py`
> （`BudgetPlan` 限额 + `BudgetLedger`），接线四处 —— 轮次最前面两道 fail-closed 的门
> （墙钟到点 / oracle 次数用尽，各自一条停因，都在**开机时之前**）、测量点记账与
> `_true_measurements` 同点、心跳只记账**不被拦**（饿死标定 = 把「尺子有没有坏」变成缺钱
> 就能跳过的事）、token 超预算 ⇒ 退化成规则路径（纯数值搜索）。账本落
> `artifact.metadata["budget"]`。**「不可挪用」的实现**：非 confirm 的开销只许花到
> `总数 - 预留份额`，`confirm` 只能花自己那一格；份额**先切预留**（余数归 refine），
> 四格之和恒等于总数。
> **边界**：`explore` / `refine` 今天**不分相**（那需要 §10.3 的「平台化」判据 —— 该判据已落地
> （doc 69），但分相本身仍未做；见
> [CHECKLOG I53](../research/CHECKLOG.md)）⇒ 被强制执行的只有「总数 - 预留」与「预留专属
> confirm」两条，`phase_caps` 是计划；三种预算**默认全关**（opt-in —— 默认生效的机时承诺
> 会改变每一次真机运行）；按 GPU **秒**的预算**未做**（今天只有次数）。见
> [CHECKLOG I54](../research/CHECKLOG.md) 与 [doc 67](../research/67-三种预算与不可挪用的预留.md)。

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
| **roofline 收敛** | 利用率接近峰值，**且峰值分母已标定** | 「标定」= **实测过**（`peak_measured`），不是「表行口径核过」（`peak_trusted`）；未实测 → 此出口**永久关闭**（今天所有真机都在这一格） |
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

> **实现状态（2026-10-02）**：`belief.update` 那一格已落地 —— `can_tune=False` 的读数
> **不进信念**（`Conductor` 的 `last_tunable_perf` 门），下一轮提议只能从上一次**有资格**的
> 读数出发；`optimization_history`（会印进 prompt 的历史）与 bandit 回报统计按同一判据收口，
> prompt 层不许把「没有读数」渲染成 `0.00`。**它不是「停下来」**：`can_tune=False` 的轮次
> 照样跑满预算、不许提前终止（真机用例守着这一条）。见 [CHECKLOG I52](../research/CHECKLOG.md)
> 与 [doc 66](../research/66-搜索侧的信念门与bandit.md)。

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

> **实现状态（2026-10-02，第一片 = 枚举半片）**：`operator_agent/candidates.py` 立了
> 联合类型（`SourceCandidate` / `TemplateCandidate`）、**分开写的去重口径**
> （源码型走 α-等价结构签名，与 `iteration_history` 同一个 `novelty.signature_of`；
> 模板型走 `template_id + instantiator_ver + 排序后的参数` —— 模板候选没有源码可签名）、
> 参数空间枚举（`ParamSpace.points()`，轴序即枚举序、顺序确定）、
> **便宜代理剪枝**（`prune_by_smem`：复用 `resource_check` 的 smem 估算器与**同一个**
> 宽松系数 `SMEM_SLACK = 1.2`，不花 GPU 秒就能摘掉「明显放不下」的点）、
> 以及那个开关（`param_space_of` + `require_supported_space`）。协议方法
> `param_space()` 立在 `backend_caps.Backend` 上，默认返回 `None` = 代码型 ——
> 于是「没有这个方法」按设计**不可表达**（不靠 `hasattr` 嗅探）。
> 上界由 Verifier 从 `HardwareCaps` 发布进 L3 metrics（`shared_mem_per_sm_kb`），
> 与 L1 资源预检**同一个真源**。
> **边界**：① **模板候选的实例化（`build`）未实现** ⇒ 后端若声明了参数空间，控制器
> **fail-closed 拒绝**（`NotImplementedError`，点名模板）而不是降级成代码型继续跑 ——
> §12 要挡的正是「把候选悄悄当成源码字符串」；② 剪枝只估**单缓冲**（不含 `num_stages`
> 的多缓冲）⇒ 只砍明显超限的点，边界交给远端；③ 上界取不到时**返回 None（三态）**，
> 剪枝**不生效**且明说，不当成「剪过了没事」；④ 全摘光时**一个都不摘**（剪枝是省冤枉钱，
> 不是让这一轮没得跑）；⑤ 参数轴今天只有 tiling 三维用得上（`stages` / `warps` /
> 向量宽度 / `layout` / split-K 还没有产出者 —— 它们要等模板实例化）。见
> [CHECKLOG I56](../research/CHECKLOG.md) 与 [doc 68](../research/68-候选联合类型与便宜代理剪枝.md)。

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

  **落地口径（2026-10-01 校正）**：这句话里其实有**两问**，必须分开存放 ——
  ① `peak_trusted`：表行**口径核过**没有（dense/sparse 没混、算术自洽）。H20 那行的出处
  自己写着「flash-attention dense 表 148/4.0 + 算术推导；**无公开官方页**」，它算核过。
  ② `peak_measured`：**真探测过一次**没有。它落在 `HardwareCaps.measured_peak_tflops`
  （`None` = 从没探测过），随 `roofline_analysis["peak_measured"]` 发布，
  **收敛出口只认这一个位**（`measurement.roofline_converged`）。

  混成一个标志的实测后果：H20 上这两问答案不同（核过 + 从没探测），而出口因为前者被
  **开着** —— 真机 4096³ 的 Roofline 行里那个分母就是这么来的。⇒ 在 `probe_peak` 落地
  之前**没有任何设备**做过实测，收敛出口在所有真机上都是关的（日志里明写
  `⚠️未实测⇒收敛出口关闭`）。**这是设计要的方向，不是缺陷**：关着只少一条停止条件
  （停滞 / 预算 / 不安全中止三条仍在），而开着会把还能优化的核提前判成到顶。

  **落地口径（2026-10-02）**：`operator_agent/peak_probe.py` 的判据 + 存放处 +
  读取三层已接上，实现要点四条：

  1. **样本是「已知好参考核」的基线臂，不是另一个 benchmark**：同一次 L3 调用里已经
     跑过的那条臂（同机 / 同形状 / 同精度），实测可达值 = `flops / baseline_time_ms`
     （分子就是 roofline 用的那一份，见 `_ai_numer_denom`）。**零额外 GPU 秒**，因此
     不占 §10.2 那 5% 的标定额度（那个额度留给心跳）。
  2. **它是下界，不是器件极限**：语义是「这台机器这个精度**至少**能跑这么快」⇒ 拿它当
     分母，利用率偏大、出口偏松。约束靠判据那七条（尤其「样本必须 compute-bound」与
     「探针自己必须是 valid 档」）；残余风险（参考核自己没打到顶）**明写**在模块里 ——
     那时该修参考核，不是放宽判据。
  3. **按精度发布与读取**：`measured_peak_tflops` 是 `{精度: TFLOPS}`，键由
     `peak_probe.precision_key_of` 从**这一轮选分母用的那个** `compute_used` 得到；
     取不到就当作未实测（**不许**借别的精度 —— tf32 与 fp32 差近一倍）。
  4. **失败一律是「继续关着」**：仿真后端（无 `REAL_DEVICE`）、档位不是 valid、离散度
     未知、AI < 脊点、高出规格 5% 以上 …… 每一条都拒绝并**把理由落进产物**
     （`metrics["peak_probe"]`），于是「没标定」与「没接线」分得开。

  **今天仍然关着的地方**（不许读成已开）：① 跨运行不持久（只在进程内，下一次运行重新
  标定）；② 标定发生在第 1 轮 L3 之后 ⇒ 出口从**下一轮**起才可能开；③ 真机未复跑。
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
| 9 | **同一批轮次跑两臂**：A 照常 / B 关掉资格审计的**后果** | 五态判决，且"没有收益"必须写进报告 |

第 9 条与前面八条性质不同：前八条问"资格审计**在不在岗**"，第 9 条问"它在不在**值**"。

**协议**（`benchmarks/ablation_eligibility.py` 是比较器，跑法是两臂各跑一次）：

- 两臂只差一件事：B 组让资格位不再拦任何东西（判据**照算照记**，只是后果不再受限）。
  ⚠️ 这个"关掉"**不许做成产品开关** —— 设计 §10.5-5 说 agent 不能编辑自己的尺子，而一个
  "一键关掉安全门"的环境变量就是那把尺子，它会先被用在无人值守的夜里。所以 B 组只能在
  harness 侧 patch 出来（`oracle.admit_to_corpus` 忽略 `can_persist`，控制器把每个读数都
  当有效），命令行里没有这个开关。
- 三个数全部从 `iteration_history` 离线可算（所以能复盘，不必重跑）：最终 `best`、
  真值测量轮数（§10.2 的货币）、**脏永久后果**轮数（读数没资格造成永久后果却造成了：
  覆盖 `best` / 永久禁用配方 —— 就是 §16 那条"让一个噪声大的数永久拉黑配置"）。
- 判决五态：`无法比较` / `没打到点上` / `没有收益` / `证据不足` / `有收益`。
  带宽取 `NOISE_BAND[uninformative]` = 2%（跨运行比较的可信度低于轮内比较）。
  关键的两条：**"门没被需要过"不算"门没用"**（否则从无信息量的实验里读出结论）；
  **"没有收益"这一向必须写进报告** —— 设计 §19.4 缺的正是它。

**状态**：比较器与五态判据已落地（`tests/test_ablation_eligibility.py`，含"不许有开关"
的政策守卫）；**两臂真机对照 ⏳ 未跑**（需要在同一张卡上各跑一次完整循环，要 LLM），
所以"资格审计有收益"这句话今天**仍然不许对外说** —— 边界与 §19.4 一致。

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

**这两件今天还没有实现物**（2026-10-01 核对，写在这里免得把承诺读成现状）：

| 缺的东西 | 现在的真实状态 | 后果 |
|---|---|---|
| `probe_peak`（实测峰值） | ✅ **2026-10-02 已落地**（判据 + 按精度发布 + 按精度读取，零额外 GPU 秒）：`operator_agent/peak_probe.py` | 收敛出口不再是「永远关」：同一轮 L3 里基线臂过判据 ⇒ 下一轮起有分母。**仍受的约束**：分母是下界（出口偏松）、不跨运行、真机未复跑 —— [doc 70](../research/70-实测峰值的判据与按精度发布.md) |
| `Placement.lease`（多卡放置） | `DeviceLease` 只有定义、无生产者；真机是**单卡串行 + 一次调用跑完两臂** | §11 的「异域并行省 wall clock」是**纸面承诺**；`pairing_conflict` 的跨卡检查没有生产喂给者 |

两件都排进 Stage 3 之前 —— 前者决定停止条件能不能用，后者决定多卡是不是真的多卡。

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

落地口径（2026-10-01 补齐）：第 3 件以前**完全没有产物** —— 收集器
（`confirm.UnableToEstablish`）早就写好了，但产品链路跑完一次只留下 JSON 与 print。
现在由 `operator_agent/report.py` 写 `report.md` 到产物目录（best-known-good +
逐轮证据链 + "我无法确立的事" + 指纹/续跑入口），在 `Conductor` 循环结束、**入库判定
之后**写出。两条形状上的纪律：

- **空章节必须带依据。** 空清单有两种来源（"都站得住" / "压根没收集"），在产物里长得
  一样；所以为空时要把逐项核过的那张清单印出来（有过 L3 读数 / `can_gate` / 档位 /
  **心跳检查过且是好的** / 未被抢占 / **峰值已实测** / 环境漂移已比较 / 契约哈希核过）。
  核不过的项不会消失，它们变成章节里的条目——"没检查"不许伪装成"检查过且好"。
- **`can_persist=False` 不是"测不准"，是"不许记住"。** 它不进无法确立章节，而是在
  best-known-good 里明写一行"这条没有进长期记忆"。

---

## 18. 待定问题

| # | 问题 | 影响面 |
|---|---|---|
| 1 | 算子本身是否可能跨卡（通信 / TP / EP）？ | 决定 `TopologySpec` 与性能模型是否要引入 comm overlap |
| 2 | 外部负载占卡的容忍策略：降级继续，还是排队等独占？ | 决定 `Placement` 的等待策略与 wall clock 预算 |
| 3 | 停机后的续跑语义：从 best 继续，还是从信念状态继续？ | 决定 checkpoint 粒度与日志 schema |
| 4 | Triton 与 CUDA 是否共享同一套参数空间抽象？ | 决定 `param_space()` 的表达力与 bandit 的通用性 |

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

- **方法主效应 +20.4pp > 模型主效应 +13.0pp**（Kimi K3 强于 Qwen3.8-Max）
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
   （省 92.8% token，且方法效应大于模型效应）。
3. **全覆盖守卫是有效组件。** 去掉后 12/27（Don't Regenerate）→ 支持 §13.2 的能力声明 + §4.3 的
   静态审计 / 反作弊**必须保留**，不能作为"简化"砍掉。
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