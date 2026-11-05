# 内层 / 外层模型的分工边界（HITL 生成侧）

> 历史设计记录。2026-11-05 用户明确选择由 DSH 外层组装最终上下文并驱动生成/修复；v0.7.2 的新默认分工见[外层工作流](dsh-host-context-workflow.md)。本文保留原调查与旧路径事实，不作为新 host 模式“写核必须交给内层”的要求。

日期：2026-10-25。本文回答一个问题：**算子开发里「写核」与「决定下一步试什么」这两步，该归外层模型（DSH 侧）还是内层模型（本仓 `LLMClient`）？**

前提：目标是**从 0 开发出可用算子**，算子经验库（`kernel_db` / `failure_db`）弱化甚至不用。这个前提很重要——它废掉了内层 agent 最常被引用的一条优势（入库门），所以必须重新数一遍剩下的。

**这不是什么**：不复述工具契约、数据布局与拒绝码（那是 [`dsh-plugin-implementation.md`](dsh-plugin-implementation.md)）；不复述实测读数与 Phase 顺序（那是 [`../dsh-plugin-evaluation.md`](../dsh-plugin-evaluation.md) §9）；**不重开任何已决边界**（见 §1 的两次纠正）。

---

## 1. 调查过程，以及两次自我纠正

这份文档的价值有一半在纠正里，所以按时序写。

**第一步**：先问「插件形式是否比内层 agent 更合适」。查到的第一条内层独有优势是语料入库——`oracle.admit_to_corpus` 只在 [`conductor.py:3677`](../../operator_agent/conductor.py#L3677)（`_produce_remeasure` → 两次确认 → 入库）接线，`hitl_tools.py` 里没有这条路径。成立。

**第二步**：但前提说语料库弱化，这条优势作废。于是重新查：拿掉入库之后，内层还剩什么是外层结构上够不着的。得到三条（§3）。

**纠正一：「HITL 路径不入库」不是缺口，是已决边界。**
我一度把它当成插件路径的功能缺口。`../dsh-plugin-evaluation.md` §9.13 结尾已经明确写过：这一层**不做两次确认、不做入库**，`report.md` 自己如实写「没有做重测」，「这是**如实的**，不是缺陷」。本文不重开这个决定，只记一句：**拿掉语料库这个前提，与仓库既有立场一致，不是妥协。**

**纠正二：「补一个生成候选的工具」不是新提议，是被刻意推迟的选项。**
我一度把「中间缺一格：外层描述清楚 → 内层生成一个候选 → 外层判」当成新发现。同一份评估文档里它已经被记过三次：§9.3（行 435）列为**可选**工具 `operator_generate_candidate`（走 `SingleCandidateCoder`，一次有界）；§9.4（行 477）列为「外层写核质量太差」时的**退路**；§二（行 29）已经指出 L2 粒度可行正是因为**单步原语齐备**（`SingleCandidateCoder`、`Verifier.verify(..., stop_after=)`、`failure_route.classify`、`report.write_report`）。

所以本文的贡献不是提出它，而是给出**「什么时候该启用它」的判据**（§5）——原评估里写的是「必须先实测才能下结论」，但没写实测要看哪一格。

---

## 2. 边界该按什么画

不按阶段画（设计归谁 / 写核归谁 / 判定归谁），按**「谁持有做这个决定所需的信息」**画。

这条原则的好处是它天然对齐仓库里已经硬门的东西：契约是意图的冻结形式，由外层声明、由代码冻结；判定是可验证的事实，共享；生成需要注入知识与累积状态，归内层。

| 决定 | 归谁 | 为什么不能换 |
|---|---|---|
| 意图澄清、契约声明（`shapes` / `precision` / `min_speedup` / `required_caps`） | **外层** | 只在对话里。内层 `LLMClient` 是一次性无状态调用，拿不到「可用」对这个用户意味着什么 |
| 交付判定：读数够不够、要不要再花一轮、报告怎么讲给人听 | **外层** | 判定层给 `route` / `eligibility` / `conflict` 这些**事实**，「所以呢」不是事实 |
| 资源与路径取舍：换 host、先冒烟、这台机器现在能不能占 | **外层** | 同上，且人在环上 |
| 写核 | **内层** | 需要 `GENERATION_SPEC` / `skills/triton_skill.md` 注入、多路采样、smem 剪枝编译门（§3-1、§3-4） |
| 参数搜索：下一个点试哪里 | **内层** | 需要 roofline 排序 + 剪枝 + 跨轮奖励（§3-2） |
| 跨轮记忆与回退 | **内层** | `RunState`（§3-3） |
| L1/L2/L3 判定、`exec_guard`、资源预检、seed+digest 输入保真、契约漂移拒绝 | **共享（已硬门）** | 别动，见 §3 末尾 |

一句话：**判定纪律是共享的，生成知识与搜索策略是内层独有的。**

---

## 3. 拿掉语料库之后，内层还剩什么（按优先级）

### 3-1 生成侧领域知识从未暴露给外层（最硬的一条）

`PromptManager` 按 Agent 角色把 `skills/triton_skill.md` 切片注入（[`prompt_manager.py:103`](../../operator_agent/prompt_manager.py#L103) 的筛选 + `:116` 的 `_SKILL_SECTION_AGENTS` 映射），消费者是 Designer / Coder / Search / Planner / CoderPipeline / Fixer。`SingleCandidateCoder` 另有一份 `GENERATION_SPEC`（[`single_candidate_coder.py:28`](../../operator_agent/single_candidate_coder.py#L28) 的 `PROMPT` 拼接）。

外层模型能看到的只有 [`mcp_server.py:56`](../../operator_agent/mcp_server.py#L56) 的 `INSTRUCTIONS`——**全部是「怎么读回包」，一个字都没有 Triton 该怎么写**。

唯一的泄漏点是 `repair` 内部的 Fixer：它确实带技能（[`fixer.py:303`](../../operator_agent/fixer.py#L303) `_skill_name = "triton_skill"`、`:302` `_agent_role = "coder"`），但那些内容只活在**内层那一次 LLM 调用里**；回到外层的只有截断后的 `fix_summary`（[`hitl_tools.py:791`](../../operator_agent/hitl_tools.py#L791)，160 字符上限）。

后果：外层模型写核是**裸写**。它可能很强，但拿不到这个仓里沉淀的 smem 上限、tile 选择、累加器精度这些约束。这是「从 0 到可用」主路径上最关键的一份知识，而它被锁在内层。

一个很说明问题的细节：[`hitl_tools.py:49`](../../operator_agent/hitl_tools.py#L49) 已经 import 了 `generation_spec_identity`——**判定层知道那份 spec 的身份（能校验它有没有漂），却没有接上能执行它的生成器。**

### 3-2 搜索是策略，`bench` 只是量尺

内层：`SearchAgent` 的邻域格点 + roofline 排序 + smem 剪枝，`Bandit` 用累积奖励重排队列（`conductor.py` 的 `selection.choose` 一线）。

外层只有 `bench`（[`hitl_tools.py:540`](../../operator_agent/hitl_tools.py#L540)），而它的 docstring 把边界写得很清楚：测**调用方给的**候选、逐臂各自与基线同窗配对、**回包没有 `levels` / `route` / `eligibility` / `passed`**——「这不是省略，是设计」，因为一旦 `bench` 能出裁决，调用方就会拿它替代 `verify`。

它不生成、不排序、不剪枝。所以「下一个点试哪里」全在外层模型手里，而它没有剪枝——真机一次完整判定的成本在 `dsh-plugin-implementation.md` §9 第 3 条记着，逐个试是结构性的效率劣势，不只是知识劣势。

### 3-3 跨轮记忆，以及一处需要确认的没接线

内层的 `RunState`（[`conductor.py:309`](../../operator_agent/conductor.py#L309)）跨轮携带 `verified_parent`（回退到上一个已验证的核）、recipe 的 applied/disabled 账、`best_gflops`、停滞计数、bandit、round 去重签名、环境漂移指纹、snapshot 存/恢复。

`repair` 的循环（[`repair_loop.py:112`](../../operator_agent/repair_loop.py#L112) 的 `repair()`）状态只活在**一次调用内**；跨调用 `hitl_tools` 只留 run 目录里的文件。

**具体一处**：[`hitl_tools.py:780`](../../operator_agent/hitl_tools.py#L780) 构造的是 `FixerAgent(llm=llm)`，**没传 `failure_db`** ⇒ [`fixer.py:299`](../../operator_agent/fixer.py#L299) 默认 `None` ⇒ 渲染成 `history_cases="无"`（[`fixer.py:412`](../../operator_agent/fixer.py#L412)）。而 Conductor 传了（[`conductor.py:585`](../../operator_agent/conductor.py#L585)）。**同一个修复器，两条路上能力不同。**

⚠️ **这可能是刻意的，不要直接改。** `FailureDatabase()` 的默认构造会绑到仓库自己的 `output/failure_db.jsonl`（`../research/49-后端能力声明与静默降级.md` 记过这条后果链），而 `tests/test_hitl_tools.py` 钉住「语料不写回仓库」。所以「不传」是**安全默认**。要补的话应该传一个 **run 目录内**的实例，照 `run_autonomous` 重定向 `OPERATOR_AGENT_KERNEL_DB` / `_FAILURE_DB` 的同一套做法（`dsh-plugin-implementation.md` §6）。**先确认是刻意还是漏接线，再动。**

### 3-4 多路采样 + 编译门（次要，但便宜）

`CoderPipeline` 的 `SAMPLE_TEMPERATURES = (0.1, 0.4, 0.7)`（[`coder_pipeline.py:52`](../../operator_agent/coder_pipeline.py#L52)）：Plan → 三路并行采样 → 机械改写 → smem 剪枝的编译门 → 有状态修复（≤2 轮）→ 规则模板兜底。

外层模型通常一次给一个候选，而且对它来说每次采样是一次完整推理，不会自发这么放大成功率。

### 内层没有优势的地方（同样重要，别把这些当成理由）

判定纪律**两条路都吃得到**：`exec_guard` 在后端里强制（[`mock_gpu.py:431`](../../operator_agent/mock_gpu.py#L431)、`remote_executor.py` 的 `run_compile` 一线），资源可行性预检在 `Verifier` 的 L1 里（[`verifier.py:486`](../../operator_agent/verifier.py#L486) 起）。输入 seed+digest 保真、`bench_cases` 套件、persistent SSH、`write_report` 也都是共享的。

---

## 4. 真正的设计难点：这条边界现在是**建议性**的

`operator_verify` 的 schema 明明白白接受 `code`（[`mcp_server.py:118`](../../operator_agent/mcp_server.py#L118) 起的 `anyOf: [{code}, {code_file}]`），而且**必须**接受——`dsh-plugin-implementation.md` §0.4 的**工作流 B 就是「外层写候选，判定层只判」**，那是明确设计过的用法，不能砍。`INSTRUCTIONS` 现在会**劝退**外层自己从头写核（§4.3），但**不禁止**：劝退是措辞，禁止是门，而 (C) 那条门已经被否掉了。

于是失效模式是：**外层模型自己裸写一个核 → 判定层照样判 → 判过了照样报「通过」→ 事后从证据里看不出来这个核是谁写的、有没有走过 spec 与技能注入。** 不是出错，是静默走了一条能力更弱的路。

三个应对，取舍清楚：

| | 做法 | 成本 | 失效模式 |
|---|---|---|---|
| **(A)** ✅ | 纯声明：`INSTRUCTIONS` 里写清「写核请用 `repair` / `run`」 | 零 | 靠自觉。**已落地，但写不到多硬，见 §4.3** |
| **(B)** ✅ | **可归因**：回包与 `report.md` 里记候选来源（`outer` / `inner_fixer` / `inner_coder`） | 一个参数 + 报告一行，**不改任何判定语义** | 无（不阻止，只记录）。**已落地，见 §4.1** |
| **(C)** | 强制门：`verify` 拒绝外层候选进 L3 | 高 | **废掉工作流 B**，不建议 |

**推荐 (B)。** 理由是它和这个仓一贯的模式一致——`input_seed_fallback: true`、`grade=exploration`、`validity=degraded` 都是同一条纪律：**不阻止，但让读数可归因**。裸写的核不该被禁，但**该被标记**。

附带的、也是更重要的好处：有了这一格，「外层裸写的通过率 vs 内层生成的通过率」就变成可统计的量，边界该画在哪可以用数据回答，而不是靠感觉争论。

### 4.1 (B) 已落地（2026-10-25）

`candidate_origin` 现在在 `verify` / `repair` 的回包、`verify-N.json` / `repair-N.json` 与 `report.md` 里。实现时定下来的几条，比原提议更具体：

| | 决定 | 为什么 |
|---|---|---|
| **缺省值** | 省略 ⇒ 记 `undeclared`，**不是省略这一格** | 省略就等于把「走了哪条路」重新变成猜。`hitl_tools.UNDECLARED_ORIGIN` |
| **词表** | 闭合三项 `outer` / `inner_fixer` / `inner_coder`（`hitl_tools.CANDIDATE_ORIGINS`），MCP schema 的 `enum` 直接取这一份 | schema 与实现各写一份 = 「工具说收 A、实现只认 B」。守卫 `test_the_schema_enum_and_the_runtime_vocabulary_are_one_list` |
| **词表外的值** | 拒为 `unknown_candidate_origin`，拒绝理由里列出词表 | 与既有拒绝码同一条纪律：拒绝要**可归因** |
| **`repair` 多加一格** | `final_candidate_origin` 是**派生**的：有任一轮 `action="fixed"` ⇒ `inner_fixer`，否则等于传入那格的声明值 | 「内层真改过源码」这件事工具自己知道，不该让调用方声明 |
| **`bench` 刻意不加** | — | 它结构上不出裁决字段（`hitl_tools.py:567` 那段注释），而 §5 要比的是 L2 通过率，不经过它 |
| **报告只在有键时多一行** | `report._candidate_origin_lines` | `Conductor` 那条路的 artifact 不带这个键；无条件加行会改动既有报告与 golden 产物 |
| **回包契约版本** | `TOOL_VERSION` 2 → 3 | 它的注释就写着「改回包形状就要 +1」 |

**这一格治什么、不治什么**：它治「看不见」，不治「说谎」。来源是**声明式**的，所以只和调用方的诚实一样可靠——外层模型完全可以把自己写的核标成 `inner_coder`。这是 (B) 相对 (C) 的固有代价，也是为什么 (C) 那条（强制门）没有被选：它会废掉工作流 B。

代码：[`hitl_tools.py`](../../operator_agent/hitl_tools.py) 的 `CANDIDATE_ORIGINS` / `_origin_of` / `verify` / `repair`；[`report.py`](../../operator_agent/report.py) 的 `_candidate_origin_lines`。守卫：`tests/test_hitl_tools.py` 的 `TestCandidateOriginIsAttributionNotJudgement` 与 `TestRepairAttributesTheFinalCodeNotTheInput`。

### 4.2 落地时顺带撞出来的一条死路

`operator_agent repair` 这个子命令**一次都跑不起来**：repair 的 argparse 没有 `--code-stdin`，而 `_read_code` 对三个来源都做 `bool(args.code_stdin)` ⇒ `AttributeError`。而 `dsh-plugin-implementation.md` §0.5.1 的工作流 D 正是这条命令。已修（补上那个选项），并加了守卫 `TestTheRepairCliPathIsAlive`——**少一个选项不是「功能少一点」，是整条路死掉，而且死在没人测的分支上**。

### 4.3 (A) 也落地了（2026-10-26），但措辞写不到多硬——因为工具面缺一格

`mcp_server.INSTRUCTIONS` 现在明确「**写核优先交给内层，别自己从头写**」，并按已确认的方向
（外层当调度器与判读者、内层当生成器）写清两条例外：**人写的核照常走 `operator_verify`**
（工作流 B 是设计过的用法，不能劝退），以及「要单个候选又不值得起一次整段」时允许自己写、
但必须如实标 `outer`。`_ORIGIN` 的 schema 描述同步改成「**如实填**，不要为了好看填
`inner_coder`」——原来那句「自己写的核就填 outer」读起来像在认可裸写。

**为什么第二条例外必须留着**：从零整段只有 `operator_run`，那是**无人值守的整段**（要等它跑完，
中间拿不回控制权）；`operator_repair` 又必须**已有一份核**。所以「给我一个候选，我来判、我来
决定下一步」这个最常见的情况下，外层模型除了裸写**没有别的选项**。这是 §5 那个中间格
（`operator_generate_candidate`）的**第二条独立论据**：

- 第一条（§3-1）是**能力**：内层有注入，外层没有；
- 这一条是**指令无处可指**：就算想说「别自己写」，也没有一个便宜的工具可以指过去。

**顺带核准的一件事：各条内层路径的注入并不相同**，别把它们当一回事（`INSTRUCTIONS` 里也这么写了）：

| 路径 | 带什么 | 不带什么 |
|---|---|---|
| `operator_run` → `Conductor` → `CoderAgent`（当前无人值守入口显式关闭 Pipeline） | `GENERATION_SPEC` + `triton_skill` 注入 + 优化配方 + 单候选预检重试 + 控制器验证/修复 | 不默认执行 Pipeline 的三温度并行采样 |
| 显式启用的 `Conductor` → `CoderPipeline` | `GENERATION_SPEC` + 规划 + `triton_skill` 注入 + 三温度采样 + smem 剪枝编译门 + 跨轮搜索/回退 | — |
| `operator_repair` → `FixerAgent` | `GENERATION_SPEC` + coder 角色的 `triton_skill` 注入 + 失败反馈与绑定当前源码的诊断 | 不做规划、多候选采样或参数搜索；`history_cases` 在本路径恒为「无」（§3-3） |
| `SingleCandidateCoder` | `GENERATION_SPEC` + `anticheat.static_audit` + 接口契约静态门 + 有界一次调用 | **不走 `PromptManager`，所以没有 `triton_skill`** |

Fixer 的共享规约来自 [`PromptManager.render()`](../../operator_agent/prompt_manager.py#L182)：`fixer_repair` 模板命中 `fixer_` 前缀，渲染后会自动前置 `GENERATION_SPEC`；coder 技能则按角色填入模板。只查看 `fixer.py` 是否直接引用该常量，会漏掉这层间接注入。

因此，`operator_repair` 同时具有共享规约和 coder 技能；`operator_run` 串起生成、验证与控制器迭代，三温度并行采样则要求显式启用 Pipeline。路径之间的区别在流程能力，而不是 Fixer 缺少共享规约。

2026-10-26 起，上述生成器共用[生成上下文和候选记录协议](shared-generation-context.md)，显式记录实际策略、技能身份、请求哈希及候选血缘。该协议没有增加技能注入、改变调用预算或证明内层质量优势；外层/人工候选也不会因来源声明而获得生成记录。

---

## 5. 启用 `operator_generate_candidate` 的判据

不是「现在就该加」，而是「**有了 (B) 的读数再决定**」。判据：

- 若 `outer` 裸写的 L2 通过率**显著低于** `inner_coder` ⇒ 启用，把「写」交回内层，人仍能看到并干预每一步（这正是 §9.4 行 477 写的退路）。
- 若相当或更高 ⇒ 不启用，省一个工具面与一份要维护的 schema。

**需要的前置**：一组能区分两臂的实测。但**别把两个问题混成一个**（本文早先混过，2026-10-25 改）：

| | 问的是 | 要什么夹具 | 现状 |
|---|---|---|---|
| **Q1 生成** | 外层裸写能不能**写对** | 任务集 | ✅ `quality_suite.py` 的十任务冻结套件（形状/dtype/seed 已冻） |
| **Q2 修复** | 外层能不能**按处方改对** | 5 类 seeded bug | ⚠️ stride 齐、尾块 mask 半个（`tests/fixtures/single_candidate_unmasked_variance.txt`）、dtype 只有编译失败那版；**广播轴与 K 循环不存在** |

`../dsh-plugin-evaluation.md` §9.13 第 4 条那批（尾块 mask、stride、广播轴、K 循环、dtype）是 **Q2**，不是 Q1。要回答本节判据的是 **Q1**，而 Q1 **不需要写任何夹具**——所以顺序是 Q1 先。

### 5.1 真要跑 Q1，这三条会咬人（2026-10-25 查过）

1. **现有 harness 的「内层臂」不是真的内层。** `quality_evaluation.py` 把 `GENERATION_SPEC` 手工拼成 system message，**没走 `PromptManager` / `skills/triton_skill.md`**。拿「裸写」去比已有的 8/10（`../research/operator-quality-portfolio-20261024.md`），测的是 **spec 注入**值多少，不是 §3-1 那条论点（spec **+** 技能 **+** 采样）——两臂的知识差被削掉一半，数字看着像答案，测的不是那个问题。要先把内层臂接到真生成器上。
2. **harness 全都绕开 `hitl_tools`**（自己 `new Verifier`），所以 §4.1 那一格 `candidate_origin` **一个都捡不到**。要让证据带来源，判定得改走 `hitl_tools.verify(..., candidate_origin=...)`。
3. **模型这个混淆项是天然控住的，别浪费掉。** 已有那批真机内层数据用的是 `deepseek-v4.1-flash-1017`，而 DSH 外层的默认模型**也是它**。所以两臂同模型不是妥协，是现成条件。另外**臂 O 不需要 DSH、也不需要人在环**：「外层裸写」的可复现形式就是「同一个模型 + 只给任务声明的裸 prompt + 一次调用、无采样」，脚本能发。跑之前先标定量具——断言臂 O 的 prompt 不含 `GENERATION_SPEC`/`triton_skill` 的任何一段、臂 I 的含，否则「两臂只差 prompt」是假设不是事实。

机时量级：L2-only、十任务 × 两臂，按 portfolio 实测（十任务 × 三槽 1238 s、≤3 并发）外推约 **800–1250 s**。历次质量报表都**没有**跑过 L3。

**顺序：先 (B)（已落地）→ 再跑 Q1 → 再决定要不要中间格。** 反过来做（先加中间格）等于先假设内层更强，而那恰好是 §9.4 里唯一标注为「未验证」的假设。

**边界最终由人判读**：这一节给的是判据与量具，不是把结论交给脚本。(B) 的作用是让判读时**分得清哪个核是哪条路来的**，而不是自动得出「边界画对了」。

---

## 6. 边界与既有文档的关系（不要在这里找这些）

| 想找 | 去哪 |
|---|---|
| 工具契约、数据布局、拒绝码、后端语义 | [`dsh-plugin-implementation.md`](dsh-plugin-implementation.md) |
| 决策过程、Phase 顺序、真机实测读数 | [`../dsh-plugin-evaluation.md`](../dsh-plugin-evaluation.md) §9 |
| `SingleCandidateCoder` 的有界性（一次调用 / 预算 / 静态门） | [`bounded-code-generation.md`](bounded-code-generation.md)、[`single-candidate-codegen-h20-acceptance.md`](single-candidate-codegen-h20-acceptance.md) |
| 入库门与两次确认 | `../dsh-plugin-evaluation.md` §9.13 结尾那段、[`durable-corpus-admission-and-recovery.md`](durable-corpus-admission-and-recovery.md) |
| `failure_db` 的默认构造为什么会绑仓库语料 | `../research/49-后端能力声明与静默降级.md` |

---

## 7. 我无法确立的事

- **没有实测外层模型写 Triton 的质量。** §9.4 那个未验证假设依然未验证，样本只有 2 个核（§十的诚实账）。本文 §5 给的是判据，不是读数。
- **(B) 已落地，但 `candidate_origin` 的读数还是空的。** §4.1 那一格只是让归因**可记**；「外层裸写 vs 内层生成」的通过率对照**一条数据都还没有**，要等 §5 的 Q1 跑完。所以现在仍然不能说边界画对了。
- **2026-10-25 的决定：停在 (B)，不搭 Q1 的对照实验。** 理由是归因可记就够了，边界**由人看实测结果判读**，不交给脚本得出结论。所以 §5.1 那三条是**给将来要跑的人**的前置清单，不是待办承诺。
- §5.1 的三条已逐条核过代码（`quality_evaluation.py` 里 grep 不到 `PromptManager`/`triton_skill`，且只在 `:251` import 了 `_backend`；`quality_suite._CASES` 确为十条冻结任务），但**没有跑过任何一臂**。
- **`undeclared` 会是绝大多数存量的取值。** 这一格从 `TOOL_VERSION` 3 才开始记，之前的 `verify-N.json` / `report.md` 里没有它；跨版本比较通过率时，旧回合要按「未知来源」处理，不能当成 `outer`。
- **§3-3 那处 `failure_db` 是刻意还是漏接线，我没有答案。** 代码里两种解释都成立（不传 = 安全默认；不传 = 少了 Reflexion 历史），提交信息里没查到理由。所以那里写的是「先确认，再动」。
- **行号会漂。** 当前是脏工作区（`design/autonomous-spine` 分支），引用按本文落笔时的树。`../dsh-plugin-evaluation.md` §十记过同一条教训：任何估算应钉 commit。
- §3-2 里 `SearchAgent` / `Bandit` 的具体行号我没有逐一复核（只核了它们**不在** `hitl_tools` 的 import 里），要用请先 grep。
