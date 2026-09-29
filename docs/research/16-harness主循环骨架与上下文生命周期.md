# 调研 16：主流 agent harness 的主循环骨架与上下文生命周期

> 起因：用户 2026-09-29 提问「现在的主流 agent harness 框架，在完成一个复杂任务时，核心逻辑是怎样的」。
> 现有 `1x-` 文档里，`10-` 只覆盖 DeepSeek Harness 一个对象，`11-` 是 kernel 生成系统全景，
> **「通用 harness 主循环长什么样」这个主题在本仓是空白**。本篇补这个位置。
>
> **证据标注约定**：✅ = 本人用 curl 直取 raw 源码或官方文档原文并核对过行号；
> ⚠️ = 由调研 agent 转述、或只读到摘要、或路径已失效无法复核。**本篇的承重数字我尽量自己重取了一遍**，
> 取不到的都留在 ⚠️，没有为了让表好看而升级标注。

---

## 一、一句话结论：笨循环 + 聪明外围

主循环在所有主流实现里几乎同构，而且是**刻意做得笨**的。LangChain 的官方定义就是全部：
*"An agent is a model calling tools in a loop until a given task is complete."*

必备件只有四个：**模型调用 → 工具执行 → observation 回填 → 终止判定 + 步数上限**。
其余全部是可选件，且**多数默认关闭**（见第八节）。

复杂度不在循环里，而是外移到了三处：**① 上下文生命周期、② 工具接口形状、③ 验证信号从哪来**。
这三处才是各家真正拉开差距的地方，也是本篇的主体。

---

## 二、循环上限：从 10 到「无限」，说明它是成本旋钮而非架构决策

| harness | 上限 | 证据 |
|---|---|---|
| OpenAI Agents SDK | `DEFAULT_MAX_TURNS = 10`（`src/agents/run_config.py:45`） | ✅ 本人 curl 核对 |
| smolagents | `max_steps = 20`，`while not final_answer and step <= max` | ⚠️ agent 转述，本人未复核 |
| Gemini CLI | `const MAX_TURNS = 100;`（`packages/core/src/core/client.ts:79`，在 `:928`/`:974` 用于 replay 截断） | ✅ 本人 curl 核对 |
| Codex CLI | `turn.rs:426` 起 `loop{}`，靠 `:448/:701/:755/:817` 多处 `break` 退出；**该文件内 grep 不到 `max_turn`/`turn_count`** | ✅ 本人 curl 核对（单文件级结论，不代表全仓无上限） |
| Aider | `max_reflections = 3`（`aider/coders/base_coder.py:101`） | ✅ 本人 curl 核对 |
| OpenHands | `DEFAULT_MAX_ITERATIONS = 500` | ⚠️ **无法复核**：agent 引的是 TS 客户端文件；Python 侧 `openhands/controller/agent_controller.py` 与 `agenthub/codeact_agent/codeact_agent.py` 在 main 上均 **404** |

跨度 10 → 500 → 无限（Gemini 的 `maxSessionTurns` 默认 `-1`）。这个数不像「架构选择」，
更像「一次跑下来愿意付多少钱」。别把它当设计指标抄。

---

## 三、上下文生命周期（本篇最高价值的一节）

长任务撑爆 context 是硬约束，三家的做法有具体数字可查：

### 3.1 Codex CLI：按 token 数触发，且压缩目标写着「保前缀缓存」

- `effective_context_window_percent` **默认 95**：`default_effective_context_window_percent() -> i64 { 95 }`
  （`codex-rs/protocol/src/openai_models.rs:389-392`），硬上限按
  `context_window * percent / 100` 算（同文件 `:521`）。另有可显式指定的
  `auto_compact_token_limit: Option<i64>`（`:460`）。 ✅ 本人 curl 核对
- 压缩策略的注释写明「**从头裁剪以保前缀缓存，近期消息原样保留**」（agent 引 `compact.rs:313`）⚠️ 未复核。
  若属实，这是个值得注意的取向：**它优化的是 KV cache 命中率，不是信息保留率**。

### 3.2 Gemini CLI：按百分比触发，三档保留策略

`packages/core/src/context/chatCompressionService.ts`（**注意：在 `context/` 下，不在 `services/` 下**）：

- `DEFAULT_COMPRESSION_TOKEN_THRESHOLD = 0.5`（`:45`）→ 用到模型 token 上限的 **50%** 就压，不是逼近上限才压。
- `RECENT_TURNS_PROTECTED = 3`（`:50`）→ 最近 3 条工具响应**全保**。
- `COLLAPSED_FUNCTION_RESPONSE_MAX_BYTES = 2048; // 2 KB`（`:88`）→ 更早的 functionResponse **折叠成 2KB 预览**，不是整条丢弃。
- 摘要由 LLM 生成，且有一道二次校验（`:595` 生成、`:643` 校验）⚠️ 行号来自 agent。

✅ 三个常量与行号本人 curl 核对过。

> **转述失真的实例**：调研 agent 报的路径是 `core/src/services/chatCompressionService.ts`，
> 该路径 **404**，实际在 `core/src/context/`。这正是本项目反复踩的那类坑——
> 二手转述的文件名/路径不能直接写进文档，否则后人按路径找不到东西，
> 还会误判「这个机制不存在」。

### 3.3 Claude Code：文档宣称级，无数字

官方文档原文 *"automatically compacts conversation history when you approach context limits"*，
保留 "important code and decisions"；`/compact` 可带指令，`CLAUDE.md` 里可写 "Compact instructions" 定制保留内容。
**触发阈值的具体百分比 / token 数：not reported**（文档没给，本体闭源无法核对）。 ✅ 官方文档原文

三家的共同套路（Anthropic 侧总结成三件套）：**压缩**、**把进度写进外部文件**（`claude-progress.txt` 类）、
**just-in-time retrieval**（context 里只存路径/ID，运行时再 load）。

---

## 四、工具接口形状 > prompt（ACI）

SWE-agent 论文（arxiv 2405.15793）摘要层面我能确认的只有立论本身：
*"we posit that LM agents represent a new category of end users... would benefit from specially-built interfaces to the software they use"*，
且 *"custom agent-computer interface (ACI) significantly enhances an agent's ability to create and edit code files, navigate entire repositories, and execute tests"*。 ✅ 摘要原文（本人取到）

论文 Table 1/Table 3 的消融数字（同一模型下 full-file 显示 12.7 → 100 行窗口 viewer 18.0；
保留全历史 15.0 → 只留最近 5 条 observation 18.0；无专用编辑器 10.3）：⚠️ **仅由 agent 转述，
我只取到摘要、未读到表格，数值不作数**。方向性结论（接口形状显著影响成功率）与摘要立论一致，
但要用这些数字必须先复核原文表格。

### 4.1 Aider 是「接口优先」最实装的样本 ✅ 以下均本人 curl 核对

- **编辑格式按模型覆写**：`Model.edit_format: str = "whole"`（`aider/models.py:131`）是基础默认，
  随后按模型名逐个覆写成 `"diff"`（`:439/:449/:456/:465`）。也就是说**没有「最好的编辑格式」，只有配对模型的格式**。
- **编辑失败不是重试，而是把报错当下一轮输入**：`apply_edits` 抛的 ValueError 被塞进
  `self.reflected_message`，由 `run_one` 的循环喂回模型自我修正；预算就是 `max_reflections = 3`
  （`base_coder.py:101`）。**报错信息的措辞是接口的一部分。**
- **按需加载而非全量塞**：repo map 用 tree-sitter 抽 symbol + PageRank 排序 ⚠️（agent 转述），
  但只给「符号级摘要」，全文必须 `/add`；模型引用未加载的文件时，反射消息提示它去 add。
- **`dirty_commits` 的真实语义**：签名默认 `dirty_commits=True`（`base_coder.py:309`），
  `check_for_dirty_commit()`（`:2175-2177`）的注释原文
  *"We need a committed copy of the file in order to /undo, so skip this"*——
  它是**为了 `/undo` 能回退而先把脏状态提交快照**，**不是**「测试不过就不提交」。
  这条很容易读反，本人特意核到函数体。✅

---

## 五、验证信号与终止判定

- Claude Code 把循环写成 `gather context → take action → verify work → repeat`，
  verify 手段 = lint / 截图 / LLM-as-judge；且**生成者与评审者是分开的 agent**，
  公开理由是躲「自评偏置」和 context 焦虑。 ✅ 官方工程博客原文（agent 取，具体页未复验）
- OpenHands 的 goal 循环带裁判位（`status: running|complete|capped|interrupted` + "Last judge verdict"）⚠️ TS 客户端类型，服务端未见。
- Aider 的 `auto_test = False`（`base_coder.py:106`）—— **自动跑测试出厂是关的**，
  `--test-cmd` 默认空。开了以后失败输出进同一个 `reflected_message` 预算。 ✅ 本人 curl 核对
- Codex CLI 的循环退出条件之一看起来朴素得反常：**模型不再请求工具就退**（`turn.rs:448` 的 `break`）。
  也就是说这一派把「何时算完成」的判定权**交给了模型自述**，没有客观门。

---

## 六、单 agent vs 多 agent：判据是耦合度，两篇不矛盾

| | Cognition "Don't Build Multi-Agents" | Anthropic multi-agent research system |
|---|---|---|
| 结论 | 单线程 + 压缩模型 | coordinator + 3–5 并行 subagent |
| 适用前提 | 写代码：决策互相隐含、强耦合 | 只读、可并行搜索、token 充裕 |
| 有没有数字 | **全文 0 指标，纯定性** | 内部 eval 90.2%，代价 "15× more tokens than chats" |

⚠️ 两者均 agent 转述，本人未复验原文。但「Cognition 那篇没有量化证据」这条**方向上可信**，
它通篇是工程复盘文体；而 Anthropic 自己就写了失效边界（需要所有 agent 共享同一 context 的领域不适用）。

**判据**：子任务能否切成**互不隐含决策**的并行块 + 读多写少。满足就上，不满足别上。

Claude Code 的子 agent 机制与此一致且更保守（✅ 官方文档）：*"Subagents run in their own context with their own set of allowed tools"*，
回流的是**摘要而非整条 trace**；且 *"Claude Code never applies a 'bypassPermissions' value"* 给子 agent。

角色分工范式确在退潮：AutoGen README 已进 maintenance mode ⚠️；CrewAI 的
role/goal/backstory 是 `prompts.py` 里三行 `.replace()` ⚠️（agent 读源码，本人未复核）。

---

## 七、「听着先进、实则空心」清单

| 机制 | 实际是什么 | 标注 |
|---|---|---|
| OpenAI 的 handoff | 只是一个名为 `transfer_to_<agent>` 的**普通工具**，不是编排器 | ⚠️ |
| Codex `update_plan` / Gemini `write_todos` | 工具确实注册了，但**无任何框架强制**（不校验、不提醒漂移），纯 system prompt 引导 | ⚠️ |
| LangGraph 持久化 | `compile(checkpointer=None)` —— 不显式传就没有断点续跑/HITL | ⚠️ |
| smolagents "30% fewer steps" | 引的是 CodeAct 论文（2402.01030），**不是自测数** | ⚠️ |
| CrewAI 角色分工 | 三行字符串插值 | ⚠️ |
| ADK / Pydantic AI 的「状态机」 | 把 while 循环换成节点遍历，语义等价，不是架构升级 | ⚠️ |
| Aider `auto_test` | 出厂 `False` | ✅ |

这一节整体可信度低于前几节，因为它几乎全部来自 agent 转述、我没有逐条复核。
**但它是本项目最需要的一类**：我方已三次被「论文说 vs 仓库做」不一致坑过
（Ecdysis 的「复发优先」只是 prompt 里的形容词、AdaExplore 出厂 `memory_update: false`、
OpenEvolve 的 embedding 去重出厂关闭）。要把这里任何一条当依据做决定，必须先自己读到那一行。

---

## 八、对我方的含义（对照 `operator_agent/conductor.py`）

我方与主流有个**结构性差别**，而且这个差别是好事：

- 主流把循环放在**模型手里**——模型自己决定何时停（Codex 的 `break` 就是「不再请求工具」）。
- 我方 `Conductor`（`conductor.py:114`）是**外部确定性编排器**：
  Router（`:40`）→ Designer → Coder → Verifier → Search → Fixer，LLM 只是工位上的工人；
  终止判定在 `_measurement_drives_loop(l3_result, round_num)`（`:934`），
  **归测量结果，不归模型自述**。按第七节那张表，这一条已经优于多数主流实现。
- `max_optimize_rounds` 默认 **3**（`conductor.py:130`）看着比人家的 10~500 小得多，
  但**单位不同**：我方一轮 = 生成并验证一个候选，人家一轮 = 一次 LLM 往返。**不可直接比，也别去「对齐」这个数。**
- 断点续跑我方已有（`resume_from` → `execution_snapshot.json`，`:173`、`:958` 起的一组 snapshot 存取函数）。
  对照 LangGraph「不传 checkpointer 就没有持久化」，我方这块**不落后**。

按本篇三条量尺排的我方位置：

1. **验证信号：最强**。数值对拍 + 计时是硬执行，强度高于 LLM-as-judge。
2. **上下文生命周期：最该补**。第 3.2 节 Gemini 那套「最近 3 条全保 + 更早折叠成 2KB 预览」
   是一条可直接抄的具体机制——我方 `docs/research/37-热冷两层上下文` 已在同一方向上，
   但缺少**可写进代码的档位数字**。第 3.1 节「从头裁剪以保前缀缓存」是另一个视角：
   若我方 prompt 组装存在重复长前缀，裁剪顺序会影响生成端缓存命中。
3. **接口形状：次该补**。Aider 的「编辑失败 → 报错原文进下一轮、共用反射预算」与我方
   `FixerAgent` + `fixer.py` 的思路同源，但我方没有显式的**修正预算上限**概念。

本篇**不吸纳**的：步数上限常量（成本旋钮，不是设计）、多 agent 编排（我方 kernel 搜索
不一定能切成互不隐含决策的并行块，按第六节判据默认不上）。

---

## 九、取证方法与未复核项（诚实账）

- **本机到 GitHub 直连不可达**：`curl https://raw.githubusercontent.com/...` 返回 `http=000`；
  WebFetch 三次全部超时。**必须走本机既有的本地 HTTP 代理**（`curl -x`，端口按本机环境，不写进本仓）才能取到（`http=200`）。
  后来者若照抄本篇路径去核，**必须带代理**，否则会误判「文件不存在」。
- 本人复核并命中的：OpenAI Agents SDK `DEFAULT_MAX_TURNS`、Gemini `MAX_TURNS` 与三个压缩常量、
  Codex `default_effective_context_window_percent = 95` 与 `turn.rs` 循环结构、
  Aider `max_reflections`/`auto_test`/`edit_format`/`dirty_commits`、SWE-agent 摘要立论。
- **复核失败/取不到的**（全部留在 ⚠️，未升级标注）：
  OpenHands 的 Python 侧 controller（main 上 404，代码已迁走）；
  SWE-agent Table 1/3 的具体数值（只取到摘要）；
  Claude Code 的 auto-compact 阈值（闭源，文档未给数字）；
  第六节两篇博客、第七节除 `auto_test` 以外的全部条目。
- 本篇**没有改动任何代码**。落地另开提交。
- ⚠️ **我方代码行号会漂**：本篇写作期间，`conductor.py:134` 处被并发的改动插入了 4 行，
  使我最初写下的 `:907`（`_measurement_drives_loop`）与 `:169`（`resume_from`）当场失准，
  已按当前树改回 `:934` 与 `:173`。**引用自己仓库的行号也要在收尾时重取一次**，
  外部源码的行号反而更稳定（我 curl 的是 main 的固定快照）。

---

## 十、来源

源码（取 raw 时需带本机代理，见第九节）：
- `openai/openai-agents-python` — `src/agents/run_config.py:45`
- `google-gemini/gemini-cli` — `packages/core/src/core/client.ts:79`、`packages/core/src/context/chatCompressionService.ts:45,50,88`
- `openai/codex` — `codex-rs/protocol/src/openai_models.rs:389,460,521`、`codex-rs/core/src/session/turn.rs:426,448`
- `Aider-AI/aider` — `aider/models.py:131,439`、`aider/coders/base_coder.py:101,106,309,2175`
- `All-Hands-AI/OpenHands` — main 分支 `openhands/controller/agent_controller.py` **404**（迁走）

文档与论文：
- LangChain agents docs（循环定义）
- Claude Code / Agent SDK docs：`costs`、`best-practices`、`agent-sdk/permissions`、Agent Skills
- Anthropic 工程博客：multi-agent research system、effective context engineering、
  effective harnesses for long-running agents、building-c-compiler
- Cognition — *Don't Build Multi-Agents*
- SWE-agent — arxiv 2405.15793 · ReAct 2210.03629 · Reflexion 2303.11366 · Self-Refine 2303.17651 ·
  Voyager 2305.16291 · MAST 2503.13657 · CodeAct 2402.01030 · Kimi K2 2507.20534
