# 生成侧问题审查 —— 对照「验证中心」设计方案的差距

| 项 | 值 |
|---|---|
| 审查对象 | 主仓当前实现（生成侧与决策侧全链路） |
| 审查基准 | 「验证中心」设计方案：生成侧给 LLM 最少自由度，信任全部来自验证侧 |
| 基准来源 | **2026-09-30 人工确认**（此前为默认假设，未记录出处）。⚠️ 本基准与 `docs/research/11-kernel生成系统全景.md` §四末 记录的 AKO 结论「强模型该少受约束」**相反** —— 这是一次显式取舍，理由与代价见 §三 |
| 读数条件 | 2026-09-30，`b1bc65a`（净树；本文档与 `benchmarks/smoke_spine_h20.py` 为未跟踪新增） |
| 证据方式 | 源码 grep + 行号引用；CLAUDE.md 自述与代码冲突处以代码为准 |
| 审查结论 | 验证与信任侧已成熟；**问题集中在生成侧默认路径与 UX 诚实性** |
| 落地状态 | **2026-09-30 全部落地**: P2-3 `e8121bc` / P1-2 `288a714` / P1-1 `9864c3b` / P1-4 首片 `c17a84a`。P1-4 的 BO/bandit surrogate（doc 14 完整版）留给后续片 |

> §一 的问题条目按**原始编号序**排列（便于交叉引用稳定），逐条的实际优先级以「结论摘要」表为准。
> 两处事实错误已按作者确认处理（2026-09-30）：P1-2 算式更正、P3-5 撤回，过程与证据记在文末「附录：勘误」。

## 结论摘要

| 级别 | 条目 | 一句话 |
|---|---|---|
| **P1** | 1 | CoderPipeline 已实现但**默认关闭**，主路径仍是单体 Coder 裸调 LLM；且它的 `compile-gate` 用的是**最弱**的验证信号 |
| **P1** | 2 | 检索排名是「相关性 + 加分」，**历史 best 可被相关性淹没**（同 DSL 内）甚至被**排除**；gflops 回退对 memory-bound 失真（项目自述未修） |
| **P1** | 4 | SearchAgent 参数空间探索浅：LLM 直觉 + 规则兜底，Roofline 数据未闭环指导搜索。**升自 P2** —— 在「生成侧最小自由度」基准下，这是最直接的落点（把提议者从 LLM 直觉换成搜索策略） |
| **P2** | 3 | `OPERATOR_AGENT_GPU` 本地设置**静默无效**：SSHExecutor 自动选卡并覆盖，不跑 harness 无感知 |
| ~~P3~~ | 5 | **已撤回**：所引 conductor.py 注释是已修记录（本轮改的口径），被读反；详见 §一 P3-5 与附录勘误 |

对照组（判定为**正确**、不应改动的部分）见文末 §四。

---

## 一、问题详述

### P1-1 CoderPipeline 默认关闭

**证据**

```
operator_agent/conductor.py:260:        use_coder_pipeline: bool = False,
operator_agent/conductor.py:286:        if use_coder_pipeline and llm is not None:
```

**现状**：Planner → 3 路并行采样（T=0.1/0.4/0.7）→ compile-gate 先过编译 → 有状态修复（≤2 轮）→ 规则模板兜底，这条流水线**代码已全部存在**，但必须显式开 `use_coder_pipeline=True` 才走。默认路径是单体 `CoderAgent` + 最多 3 次重试。

**后果**

- 单次 LLM 调用的方差直接变成任务级方差：同一个任务跑两次，一次 L1 全过一次全挂，是默认路径的常态
- compile-gate「先编译再选」省 SSH 往返的收益，只有显式开开关的用户拿得到
- CLAUDE.md 把 pipeline 写成「Optional」，测试覆盖也以单体路径为主 → 流水线路径的回归保护弱于默认路径

**建议**（理由已按基准重述）

1. **把 pipeline 翻转为默认**，单体路径保留为 `use_coder_pipeline=False` 的调试逃生门。
2. **翻转的理由要改。** 不是"给 LLM 更多自由度"（那与审查基准相反），而是**把选择权从生成侧搬到验证侧**：3 路并行采样是让 LLM *多提议*，真正被搬走的是"**由谁决定留哪一个**"。
3. **选择信号要在基准下加强。** 当前 `compile-gate` 是**最弱的验证信号**（编译通过 ≠ 正确）。基准说"信任**全部**来自验证侧"，那么选择就不该停在第 1 级。建议：`compile-gate` 只作**早退剪枝**（省 SSH 往返），真正的选择放在 **L2 正确性**之后；L3 一侧已由 `verifier` 的资格位把守（见 `docs/design/spine-review.md`）。**代价要明说**：L2 是真机执行，幸存候选全上 L2 会把 compile-gate 剪枝省下的 SSH 往返花回去一部分；预算敏感时可折中为「compile-gate 剪枝后取前 2 名上 L2」——选择权仍在验证侧，机时不随候选数全额翻倍。
4. **验收方式要换。** 原先设想「补一条 golden 回归确认两路径产出等价」在方法学上不可实施——两条路径都是**随机生成**（T=0.1/0.4/0.7 采样），golden 测试的语义是钉死确定输出，对随机路径断言"产出等价"照字面做不出来。改为：
   - 固定 seed / mock LLM：断言 pipeline 路径下 `compile-gate` 的候选数 = 3、且被截断的候选不进 compile
   - 断言**两条路径各自都有** L1/L2/L3 端到端用例（要的是**覆盖**，不是"等价"）

### P1-2 检索排名不偏好历史 best

**证据**

```
operator_agent/kernel_db.py:515:            score += relevance + _perf_weight(r)
operator_agent/kernel_db.py:45-49:
    if record.score > 0:
        return min(record.score, 3.0) / 3.0
    if not record.measurement_validity:
        return 0.0
    return min(record.performance, 100.0) / 100.0
```

`_perf_weight` 上限 1.0，与 DSL 匹配（+3.0）、词元重叠（**+2.0/词**）在同一条轴上相加 → **相关性高的旧差样本可以排在实测最优前面**。CLAUDE.md 自述「ranking does not yet prefer the historical best」，与代码一致。

**两个比原文更硬的细节**（`kernel_db.py:489-515`）

- `else: continue` —— **DSL 不匹配的记录被直接丢弃**。所以"相关性淹没 best"只在**同一 DSL 内**成立；问题范围比"全库"窄。
- `if (algorithm or tags) and relevance == 0: continue` —— 零词元重叠的记录被**排除**，不是"排在后面"。最坏形态是"**实测最优根本没进候选池**"，比"被淹没"更硬。

**已缓解的部分**（避免误伤）：`_perf_weight` 的口径 3 已把 6 条手写 seed 的先验 `performance` 踢出加分——旧写法里全库权重最高的是没测过的 `seed_triton_001`（0.85），真机实测的 8 条只有 0.35–0.44，**没测过的数字赢过测过的**。这个倒挂已修。

**残余问题**（kernel_db.py:38-43 自述，记在 doc 49 §八-14，未修）：

- 口径 2 的 gflops 回退对 memory-bound 算子失真——gflops 高的访存核是错误示范
- 正确口径「离可达上界的距离」缺两个前置：`KernelRecord.bound_type` 没有写者；doc 42 的可达上界是按 shape 索引的曲线，标量分母装不下

**建议**：短期——同一 (kernel_type, hardware, dsl) 分组内先按 `score` 硬排序再截 limit，让历史 best 不被相关性淹没；长期——按 doc 49 §八-14 补 `bound_type` 写者后再换口径，不要「先加字段再加排序」。

### P1-4（升自 P2-4）SearchAgent 参数空间探索浅

**现状**：L3 失败后的调参靠 LLM 读 Roofline 指标给建议，LLM 失败回退 `TritonOptimizationRules`。两个来源都是**点估计**，没有搜索结构：

- Roofline 给出了利用率、arithmetic intensity，但这些数字只进 prompt，不驱动任何系统性探索（如按利用率分桶决定调 block 还是调 pipeline stages）
- tiling 空间（block_m/n/k × num_warps × num_stages）是低维规整空间，贝叶斯优化或进化策略都适用，目前完全靠 LLM 直觉跳点
- `optimization.max_rounds=3` 的预算下，LLM 跳错 2 次就触顶退出

**后果**：L3 优化轮的效果方差大，且失败模式不可归因——分不清是「这个方向就是不行」还是「LLM 没搜到」。

**与仓库已有研究的关系（本节的定位要修正）**：这条**不是新发现**。`docs/research/14-搜索策略与硬件感知目标函数.md` 已调研完，并且比本节的建议更具体：

| doc 14 原行 | 内容 |
|---|---|
| :96 | 「**贝叶斯优化** … ✅✅ **最适合替换 `TritonOptimizationRules` 回退**」← 正是本节的建议 |
| :97 | 「**Bandit**（UCB / Hyperband）… ✅✅ **最适合做「配方选择」**」 |
| :98 | 「**KernelBand**（LLM + 分层 UCB）… ✅✅ **几乎就是我方目标架构**」 |
| :119 | 「我方已有基础：`CoderPipeline` 已实现三路并行采样 + compile-select + ≤2 轮 stateful repair … **缺的是把「选哪一路继续」从固定规则换成 bandit**」 |

⇒ 问题应重述为「**已采纳但未落地，卡在哪**」，而不是「未发现」。（同篇 doc 14 开篇那条「比缺口 4 更紧急的 bug」，即 `best_gflops` 首轮自锁，**已修**：`conductor.py:834` 现为 `best_gflops <= 0`。）

**升级理由**：在「生成侧给 LLM 最少自由度」基准下，把提议者从 **LLM 直觉**换成**低维空间上的搜索策略**，是这条基准最直接的落点。因此本条从 P2 升为 **P1**，并建议排在 P1-1 之后立即做。

**建议**：把 SearchAgent 的建议从「直接采纳」降级为「先验」，在其邻域做小规模系统搜索（如高斯过程 surrogate，3 轮内每轮试 2-3 个邻域点）。Roofline 利用率作为 surrogate 的特征输入，闭环才真正成立。

### P2-3 `OPERATOR_AGENT_GPU` 本地设置静默无效

**证据**

```
operator_agent/remote_executor.py:383:  env = f"OPERATOR_AGENT_GPU={gpu} " if gpu is not None else ""
remote_scripts/run_kernel.py:244:      pinned = os.environ.get("OPERATOR_AGENT_GPU", "").strip()
```

链路是：SSHExecutor **不读**本地环境变量 → 自己按（低利用率, 高显存余量）选卡 → 把**自己的选择**注入远端 env。用户在本地设 `OPERATOR_AGENT_GPU=2`，实际跑在 GPU 7，**全程无报错**（doc 49 §851 的原始事故）。

**已修复的侧面**（避免误伤）：

- 报表侧已修（#52）：`backend_caps.py:121` 起，harness 从后端拿 `selected_gpu`，不再读用户 env
- `harness_pass_rate.py:1318` 已有警告：本地设置与实际卡不一致时会打印 ⚠️

**残余问题**：警告只在跑 harness 时出现。普通 `Conductor.execute()` 路径下，用户的选卡意图被静默覆盖，没有任何日志说「你设的 2 被忽略了，实际用 7」。

**建议**：SSHExecutor 选卡后与本地 env 比对一次，不一致就 WARN 一行。成本一行日志，消除一类「我以为我在测卡 2」的实验污染。

### P3-5 观测写入重定向的交互坑 —— **已撤回（2026-09-30 作者确认）**

**撤回原因**：初稿把 `conductor.py:1300-1303` 与 `1415-1416` 两条注释读反了——它们是**已修的记录**，不是"仍存的自述"：注释原文明写「两处口径是这轮改的」「取 `self.store.output_dir` 而不是写死的 `"output"`」，`_resolve_snapshot_path` docstring 也写「以前不返回这些」。实质核查同样不支持本条：`grep -rn '"output"' operator_agent/` 命中的 `kernel_db.py:261/762`（各有自己的 env 处理）与 `artifact_store.py:33`（env 读不到时的 fallback），均非读取侧的 CWD 坑。

**现状**：观测写入重定向目前**没有已知问题**。若未来重立此条，需先重找证据——初稿的证据链（两条已修注释）不成立。

---

## 二、与我方案的差异定位

| 维度 | 方案设计 | 当前项目 | 差异性质 |
|---|---|---|---|
| Coder 生成 | 并行采样流水线为默认；**选择在验证侧** | 流水线存在但默认关闭；选择停在 `compile-gate`（最弱信号） | **实现已有，默认未翻 + 选择信号过弱**（P1-1） |
| 检索排名 | 历史 best 硬排序 | 相关性 + 加分混合（同 DSL 内可淹没 / 可排除） | **机制差距**（P1-2） |
| 参数搜索 | 贝叶斯/进化 | LLM 点估计 | **机制差距**（P1-4，doc 14 已采纳未落地） |
| GPU 选择 | 用户意图应显式生效或被显式拒绝 | 静默覆盖 | **UX 诚实性**（P2-3） |
| 生成侧自由度 | 压到最小（模板/参数为主） | 仍是自由画 sketch + 自由写代码 | **基准级差距，见 §三** |
| 验证/信任/写回 | 三级门控 + 门控写回 | 已实现且更细 | 无差距，见 §四 |

## 三、基准声明与升级门

### 3.1 一次显式取舍：与 AKO 相反

表头基准与外部最可信的工业路线正面冲突：

| | 主张 |
|---|---|
| **本基准** | 生成侧给 LLM **最少**自由度，信任全部来自验证侧 |
| **AKO** | 「**强模型该少受约束**，固定 workflow 反而不如直接给强模型一个好环境」 |

出处与状态：`docs/research/11-kernel生成系统全景.md` §四末 **已把这件事列为 20 号决策文档需要正面回答的问题**；`docs/research/20-吸纳决策与阶段总结.md:325` 记「业界已经给了一个反向答案」。**所以本基准是一次显式否决，不是默认值。**

**AKO 是什么**（供读者定位）：**AKO = Agentic Kernel Optimization**，GitHub 组织 `TongmingLAIC`，MIT。

- `AKO4ALL` = Agentic Kernel Optimization **for All**：**18 个 blob、0 行循环代码**——"agent" 是一颗 16KB `SKILL.md` + 51.6KB `bench.py` 契约（doc 17 拉全 tree 数过 ✅）
- `AKO4X` = Agentic Kernel Optimization — **advanced & eXtensible**：有代码，含 `templates/skills/{triton,cuda,cute-dsl,tilelang,profiler-ncu,sanitizer}`、`scripts/cheat_check_modal.py`
- 成绩（doc 11 评为「最可信的工业口径」）：vs **FlashInfer 专家 kernel** geomean **1.14–1.43×**；GEMM **1.00 平 cuBLAS，明确不宣称赢**；主动公布 min–max，含 launch-bound **0.82–0.96** 的 honest negative

**冲突的边界**（避免把冲突说大）：AKO 反对的是**流程约束**（固定阶段序列、强制中间产物），不是**输出约束**（ABI / 模板 / 参数空间）。两者在"环境质量"与"测量卫生"上完全一致——AKO 的可信恰恰来自 **staleness 审计**与 **drift-robust geomean**（baseline 与 kernel 同容器背靠背计时）。

| 维度 | AKO | 本基准 | |
|---|---|---|---|
| 环境质量（工具 / profiler / 反馈） | 要好 | 要好 | ✅ 一致 |
| 测量卫生（背靠背、staleness 审计） | 严格 | 严格 | ✅ 一致 |
| 流程约束（固定 stage 序列） | 反对 | 未表态 | ⚠️ 待定 |
| **输出自由度（写什么代码）** | **让强模型写** | **压到最小** | ❌ **冲突** |

**真正要回答的只有最后一行。**

### 3.2 本基准的落点与代价

- **落点**：按基准推到底，`DesignerAgent`（自由画 sketch）与 `CoderAgent`（自由写代码）**本身就是被约束的对象** → 默认路径应为**模板实例化 + 参数选择**，LLM 只保留三件事：契约解析、结构升级、带错误日志的受限修复。
- **代价**：模板/搜索路线把上限锁在「单算子比 cuBLAS 快一点」（AKO 对 GEMM 也只是 1.00 平手）。**结构性收益（算子融合、megakernel）在这条路线上拿不到。**

### 3.3 升级门（不补 = 声明放弃结构性收益）

三条**同时**满足，才允许 LLM 做结构性改动（设计文档 §5）：

1. 参数搜索已平台化（K 轮 `valid` 且 SAME / AMBIGUOUS；**K 待标定**——没有机时分布之前给默认值，等于编一个门）
2. 平台化有**可归因的硬限制**证据（spill / smem / occupancy / bank conflict）
3. 还存在可探索的结构方向

### 3.4 待办

- [ ] 在 `doc 20` 里正面回答 AKO 的反向结论——本基准是一次**显式否决**，需要留下否决的理由

## 四、对照组：审查后判定**正确**的部分

以下条目经过同样的源码核查，结论是实现**优于**方案描述，不应被后续修改破坏：

| 条目 | 证据位置 | 为什么是对的 |
|---|---|---|
| `l3_pass_is_refused` 独立于 `can_gate` | `measurement.py` | 低 tier 不改写 passed（数字不可信≠失败），provenance 缺失直接拒过（根本没测过）；且只在任务显式要求 REAL_DEVICE 时触发，不冤枉本地 mock |
| 只有 `SLOW` 淘汰配方 | `measurement.py` verdict() | 四态判定 + 每 tier 噪声带，ties 和无证据不触发永久黑名单 |
| baseline 远端选定 | `verifier.py` 只信 `baseline_selected` | 控制面不接受本地指定的 baseline，防 TF32-off 虚增 1.9× |
| 反作弊双保险 | `anticheat.py` + `run_kernel.py shield_torch()` | 静态 + 运行时缺一不可，源于「假 kernel 通过三级验证」的真实事故 |
| FailureDB 类别门控 | `kernel_db.py` | 数值容忍/基础设施失败不可学——从「error exceeded threshold」唯一能学的是「放宽检查」，这个排除是对的 |
| `_perf_weight` 口径 3 | `kernel_db.py:45-49` | 未实测的先验数零加分，修掉了 seed 倒挂实测的旧账 |
| 每 host 串行锁 | `remote_executor.py` | 共享 GPU 防并发 benchmark 污染 |
| novelty 签名只观察 | `novelty.py` + `iteration_history` | α-等价签名记录但不据此跳过，避免过早剪枝 |
| HTTPExecutor capabilities 为空集 | `remote_executor.py:892` | 无硬件端点就诚实返回空集（而非 None），任何 `require(Cap.REAL_DEVICE)` 的消费方在构造时就拒绝 |
| 资格位 fail-closed | `verifier.py` L3 出口 | 资格推导异常时写 `ALL_FALSE` + `eligibility_derive_error`，不让守卫静默消失（`spine-review` P0-3） |

## 五、本次审查未覆盖

- 国产硬件路径（Kunlun/Ascend）的 cuda_c DSL 链路——约束在 project_memory 有记录，未做源码级核查
- `tests/` 的覆盖率充分性——只确认了存在性，未评估断言强度
- prompt 模板（`operator_agent/prompts/*.md`）与 skill 注入的实际内容质量

---

## 附录：勘误（2026-09-30 作者确认，均已落地）

| 条目 | 问题 | 证据 | 处理结果 |
|---|---|---|---|
| **P3-5** | **证据被读反**：`conductor.py:1302-1303` 与 `1415-1416` 两条注释都是**已修的记录**，不是"仍存的自述" | 原文含「**两处口径是这轮改的**」「取 `self.store.output_dir` **而不是**写死的 `"output"`」；`_resolve_snapshot_path` docstring 另写「**以前不返回这些**」。实质证据也不支持：`grep -rn '"output"' operator_agent/` 命中 `kernel_db.py:261/762`（另有自己的 env）与 `artifact_store.py:33`（env 读不到时的 fallback），都不是读取侧的 CWD 坑 | ✅ 已撤回：§一 P3-5 就地标注撤回原因，编号保留稳定 |
| **P1-2** | **算式抄错两处**：原文写"token 重叠（+0.5/词）" | 代码是 `relevance += 2.0 * len(query_tokens & record_tokens)`（**+2.0/词**）；`+0.5` 是 `elif r.dsl == "python": score += 0.5`（python 通用回退加分）。方向不变（`_perf_weight` 上限 1.0 被相关性淹没），但引用的机制错了 | ✅ 已更正：正文改为 **+2.0/词**（`+0.5` 系 python DSL 回退分，不涉词元重叠）；两个新细节（`else: continue` 的 DSL 限定、`relevance == 0` 的排除）已在 §一 P1-2 |

---

## 修订记录

| 日期 | 读数条件 | 改动 |
|---|---|---|
| 2026-09-30 | `b1bc65a` | 初稿 |
| 2026-09-30 | `b1bc65a` | 按基准确认落实四条：① 表头补「基准来源 + 对 AKO 的反向选择声明」，读数条件补 HEAD 短哈希；② P2-4 升为 **P1-4** 并补 doc 14 对照；③ P1-1 理由重述为「选择权搬去验证侧」+ `compile-gate` 降级为剪枝 + 换掉不可实施的 golden 验收；④ 新增 §三「基准声明与升级门」。另把两处已知事实错误移入文末勘误（原文保留） |
| 2026-09-30 | `b1bc65a` | 勘误经作者确认落地：**P3-5 撤回**（所引注释为已修记录，证据读反，撤回说明就地保留）、**P1-2 算式更正**（+2.0/词）；另按审查反馈补两处——P1-1 建议 3 增加真机成本权衡与折中方案（前 2 名上 L2）、§3.3 升级门第 1 条标注 K 待标定 |
| 2026-09-30 | `c17a84a` | 四项开发落地：P2-3（选卡 WARN+5 例）/ P1-2（两级排序+4 例）/ P1-1（pipeline 默认+剪枝+L2 选择，6 例验收）/ P1-4 首片（先验+邻域+轮换，9 例）。旧用例断言单体路径调用形状的钉 `use_coder_pipeline=False`；P1-4 的 §3.3 升级门第 1 条（参数搜索平台化）自此有了第一格读数 |