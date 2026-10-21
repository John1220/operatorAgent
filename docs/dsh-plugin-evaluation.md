# 算子开发功能做成 DSH 插件 —— 评估

> 日期: 2026-10-18 | 性质: **决策评估**（不是规范，不是状态）
> 评估对象: 本仓 HEAD `6ecf13e`（v0.6.0，含未提交的 repair-diagnoser / bounded-repair 改动）
> 对照对象: 本机 DSH Desktop 2.0.18 自带的 `@deepseek-ai/dsh*` **0.2.1-rc.2**，profile `~/.dsh/profiles/desktop/`
> **证据约定**: 【V】= 本次直接读到文件/行号或状态文档实测数字；【I】= 判断或推断。本文不改任何代码，未跑任何插件、未跑测试、未连真机。

---

## 〇、结论摘要

1. **Python 侧不可能直接是 DSH 插件**（【V】DSH 插件是 Cordis 的 ESM 模块，由 Node loader 装载）。所以「做成插件」在物理上只有三种形态：**(A1) MCP 旁路**（零 JS，一条配置行）、**(A2) JS 宿主插件 + Python sidecar**、**(B) 深度迁移**（把生成/编排搬进 DSH）。
2. **推荐 A 类，且第一条落地的应该是 A1**；**不建议 B** —— B 会打掉本项目最贵的那个性质：**终止判定归测量、不归模型自述**（doc 16 §八 把它写成对主流 harness 的结构性优势）。
3. **插件是叠加，不是替换。** 现状（仓库内 driver + 离线全量测试）继续当唯一真相，插件只做**人机交互入口**。这与 doc 10 §七「不引入 DSH 的代码/插件体系」不矛盾：这里借的是**它的宿主设施**（工具/命令/后台任务/审批/会话日志），不是把项目迁进 Cordis。
4. **最大的三个代价**，按被低估的程度排序：
   - **打包前置缺口**：本仓没有 `pyproject.toml`/`setup.py`，`prompts/`+`skills/` 靠 `Path(__file__).parent` 定位，而 `run_contract.protocol_identity()` **硬编码仓库相对路径去 hash 47 个源文件** ⇒ 一移动目录，run identity 就变，**续跑与对账当场失效**。不先修这个，「稳定入口」是假的。
   - **配置面搬家**：全部配置现在走环境变量；DSH 的 `scrubbedParentEnv()` 会先洗掉 `*KEY*/*SECRET*/*TOKEN*` 与 `DSH_*` 再叠加显式覆盖 ⇒ 要么显式传参（可能把密钥写进 profile YAML，违反本仓凭据纪律），要么只传**引用**。两边都要新代码。
   - **两套账、两套日志、两个模型**：外层 DSH 的 token 与会话日志、内层 `llm_client.CostTracker` 与 `output/` 产物，谁都不自动知道对方存在。必须显式规定「哪一类事实以哪一侧为准」。
5. **最大的收益**（不是「能跑在 DSH 里」这种空洞说法，而是可检验的四条）：
   - 三个复制粘贴的 driver（269/274/438 行）收敛成**一个稳定入口**——这件事与 DSH 无关，本来就该做；
   - 长跑（一次 `execute` 分钟~小时级）拿到**进度、中断、完成通知**，而本仓**已经有**可复用的持久状态（`execution_snapshot.json` + `resume_from` + `run_contract`）；
   - 任务语义（shape/精度/硬件/容差/预算）从「改代码重跑」变成**对话里改**；
   - 入口成本从「4 类环境变量 + 读 driver」降到一句话。
6. **一条必须在接口上钉死的红线**：DSH 的审批/沙箱只约束**外层能不能发起这次运行**，**不许穿透到内层的安全不变量**。工具签名里不得出现任何形如 `require_double_confirm` / `skip_anticheat` / `kerneldb=off` / 放宽容差的字段。本仓已经因为「能一键关掉的尺子会先被用在无人值守的夜里」删过一次构造参数（`conductor.py:417-424` 的注释就是墓志铭），插件化最容易把它重新引进来。

### 2026-10-18 追加（HITL 场景 + 开源形态复核）——结论有修订

7. **场景是「先 HITL、后自动化」**，这与本仓已写进规范的「**无人值守**」立场正面冲突（§7.1）。出路是**把 HITL 限制在「目标与花费」面**：人只做「改目标 / 加候选 / 加预算」三件事，**人不改判**（§7.2）。这样判定面零开关，`oracle` 仍是唯一入库门。
8. **HITL 的粒度分三档**（§7.3）：L0 事前授权（**本仓已有**：`workload_codegen_h20.py` 的 prepare → `--approved-plan`/`--plan-hash`，执行时重算并比 digest）、L1 轮间暂停（**没有**，且与无人值守立场冲突）、**L2 每步一个模型回合（推荐）**。L2 可行的原因是**单步原语已经存在**：`SingleCandidateCoder`、`Verifier.verify(..., stop_after=)`、`failure_route.classify`、`report.write_report`（§7.4）。
9. **开源形态确实是 skill**（用户记得没错，doc 17 已一手核过；本轮 2026-10-18 **重新联网复核**，见 §8.5）：AKO4ALL 是**纯 skill**（16,156 B `SKILL.md`，顶层无循环代码，★376）；AKO4X 是**Python agent + 下发 skill 载荷**；NVIDIA/skills ★3635 有 **418 个 skill 目录（17 个 kernel 向）+ SkillEvaluator 三层评测 + evals/ + 签名**。**但**：所有可信的 kernel skill 都**配可执行脚本 + 评测/反作弊层**（判据：有无校验基建，doc 17 §五），而 Intel/HF `xpu-kernels`、minho-fan `kernel-loop` 都是「**skill 管交互与规则、闸门/服务管验证**」。
10. **⚠️ DSH 生态里已经有人做了同一件事**：`xieTwim/dsh-kernel-opt`（MIT，★5，2026-08 起）是一个 DSH 插件，标题即「**实时查看并干预算子优化过程**」——面板 + 按预算运行并自行收尾的循环 + 可随时打断 + 可选的第二模型监督 + **评测器无关的双通道契约** + 信任级标注 + 收尾复测（§8.5(d)）。它的评测器改写自 AKO4ALL，兼容范围是 DSH 0.1.x（本机 0.2.1-rc.2 需版本豁免）。⇒ **面板与循环的轮子它造过了，判定层谁都没占**；本仓的差异化只在**判定层**。
11. **修订后的推荐（覆盖前文 §6.2 的顺序）**：形态 **C = skill（知识）+ 判定工具（每一步）+ Conductor 后端（后期自动化）**，三层共用一个 Python 入口（§9.1）。**第一条落地建议走「甲方案」**：先当 `dsh-kernel-opt` 的评测器（输出契约行 / 注册名字含 `kernel_evaluate` 的工具），用最小代价验证「人看得见每一步」是否真的解决问题；它的契约装不下本仓的定向事实集与 `claim_level`，所以只当第一步、不当终点（§9.2）。

---

## 一、现状基线（一手）

### 1.1 规模与形态【V】

| 项 | 值 |
|---|---|
| `operator_agent/` | 34,674 行 / 75 个 `.py` |
| 最大的 8 个模块 | `conductor.py` 3887 · `verifier.py` 2895 · `remote_executor.py` 1995 · `measurement.py` 1442 · `agent_runtime.py` 1359 · `exec_guard.py` 1155 · `kernel_db.py` 992 · `agents.py` 892 |
| 测试 | `tests/` 168 个文件 / 3,161 个 test 函数 / 51,721 行；`conftest.py` 把 `output/`、`logs/` 重定向到 tmp 并关掉两个持久化库（`tests/conftest.py:23-60`） |
| LLM 调用点 | 10 处：`agents.py:153`(Designer) `:283`(Coder) `:842`(Search) · `coder_pipeline.py:192`(sample) `:381`(repair) · `fixer.py:411`(repair) `:513`(reflexion) · `planner.py:47` · `intent_parser.py:107` · `single_candidate_coder.py:493` |
| Prompt / Skill | 11 个模板 `operator_agent/prompts/*.md`；2 个 skill `skills/triton_skill.md`(11KB) + `skills/optimization_recipes.md`(60KB)，按 `## N.` 切段经 `{{skill}}` 注入 |
| 打包 | **没有** `pyproject.toml` / `setup.py` / `setup.cfg` / `requirements.txt`；driver 靠 `sys.path.insert(0, ...)`（`test_nvidia_matmul.py:24`） |
| 落盘 | `output/<kernel_type>/<hardware>/<ts>_<id>/{code,sketch,result,benchmark,execution_snapshot,contract,run_manifest}` + `output/kernel_db.jsonl` + `logs/*.log`；本机当前 `output/` 下 **13,623** 份 `execution_snapshot.json`、`kernel_db.jsonl` 119,607 B【V】 |
| 依赖 | 控制面零第三方（`urllib` + stdlib）；`numpy` 必需；`torch`/`triton` 仅真机；远端额外需要 `ssh`/`scp`/`uv` |

### 1.2 入口与调用链【V】

```
python test_nvidia_matmul.py                       # 269 行（layernorm 274 / attention 438）
 └─ build_llm()  → LLMClient(urllib, /chat/completions, temperature=.3)
 └─ ssh_config() → RemoteExecutor(mode="ssh").require(Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE)
 └─ TaskSchema(...) 硬编码: kernel_type / hardware / dsl / compute_units=78 / precision / shapes / min_speedup / max_error
 └─ expected_output 由 driver 自己用 numpy 或 torch 算（layernorm 内联 import torch）
 └─ Conductor(backend, llm, max_optimize_rounds=3).execute(task, test_inputs, expected_output, target_gflops)
      └─ _execute_inner → correctness_preflight ×2 → run_contract.declaration → Router
         → 每轮 _run_round: Designer(LLM) → Coder(LLM, CoderPipeline) → Verifier(L1/L2/L3+roofline+oracle)
                            → 不过则 Fixer(LLM) / Search(LLM) → 回头
         → _finalize_run → report.md / result.json / 入库判定
```

三个 driver 之间 `build_llm` / `ssh_config` / `ssh_label` / `print_report` 是**复制粘贴**的；`config.yaml` 全仓无消费者（`import yaml` 零命中）。

### 1.3 现状的优缺点（先说公正话，再谈插件）

**优点**
- **零外部运行时**：不需要 Node / pnpm / profile 这一层。这对一个「零重依赖」立场的项目是真实资产。
- **确定性外部编排器**：LLM 只是工位上的工人；`_measurement_drives_loop`（`conductor.py:3194`）把终止权交给测量结果，`single_candidate_coder` 甚至断言 `max_calls == 1`。doc 16 §八 论证过这**优于**多数主流 harness。
- **安全不变量不可配置**：日志 / Kernel DB / Failure DB / failure gate / 双确认都没有环境变量开关（`_harness_controls.py` 只给进程内 harness hook）。入口越少，这条越容易守。
- **证据链完整**：contract hash（47 个源文件）、`run_manifest.json`、`execution_snapshot.json`、append-only 语料。这是本项目真正的产品。
- **测试可离线全跑**：3,161 条测试 + MockGPU，一分钟级，不碰 GPU/LLM。

**缺点（也就是插件能补上的那部分）**
- **入口成本高**：跑一次要会设 `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` / `REMOTE_SSH_*`，还要读一个 270~440 行的 driver 才知道哪些值是硬编码的。
- **无进度、无中断、无追问**：一次 `execute` 分钟~小时级，只能盯 stdout 或翻 `logs/`。改一个 shape 只能改代码重跑。
- **结果消费面是 print**：`operator_agent/` 里 258 处 `print(`（`conductor.py` 129 处），返回的是 dataclass + free-form `metrics: dict`。对外部宿主不友好。
- **不能跨工作区复用**：必须在这个仓库根目录跑，产物落进这个仓库。
- **没有自然语言入口**：`IntentParser` 有，但 driver 不走它。

---

## 二、DSH 侧事实（一手，含落点）

### 2.1 插件是什么、怎么装【V】

- 插件 = **ESM/Node 模块**，导出 `{name, inject, Config?, apply(ctx, config)}`（或 `export default class extends Service`）。实证：`node_modules/@deepseek-ai/dsh-tool-todo/lib/index.js:11-20`；已装第三方 `~/.dsh/profiles/desktop/node_modules/beauticode-dsh/`（v1.0.26）同形。
- 装载 = **profile**：`~/.dsh/profiles/desktop/package.json` 的 `dsh.profile.bundles`（当前 `["@deepseek-ai/dsh-base","@deepseek-ai/dsh-web-app","beauticode-dsh","dshmarket"]`）+ 该目录 `cordis.patch.yml`（顶层 YAML 数组，操作为 `id` 覆盖 / `disabled` / `insert`）。
- 安装四条路：`dsh plugin add`（内部走 pnpm）、Web 侧 Plugins 页、`plugin_manager` 工具（standard preset 里默认关）、**直接按绝对路径 insert**（`dsh-app-boot/README.zh.md:65`：插件名可以是绝对文件系统路径 / file: URL —— **不必发布 npm，也不必带 `dsh` 清单键**）。
- 版本兼容在**导入前**强制：`engines.dsh` 优先，否则 `@deepseek-ai/dsh*` 的 peer 范围必须匹配唯一的运行时版本；不匹配的行变 `disabled: true`，要 `allow-version` 显式豁免。

### 2.2 可用扩展点（与本项目相关的五处）【V】

| 扩展点 | 签名/落点 | 对上本项目的哪一块 |
|---|---|---|
| 工具 | `ctx.tools.register({name, description, parameters /*JSON Schema*/, output:{schema, render}, timeoutMs?, execute(args, exec)})`（`dsh-tools/lib/index.js:2974`；`run_code` 为保留名） | 「算子开发」暴露成一个工具调用 |
| 斜杠命令 | `ctx.commands.register({name, description, input:{hint}, handler})`（`dsh-commands/lib/index.js:266`），**不进模型上下文**，落 `command/run` 会话事件 | `/op run` 这类无需模型回合的入口 |
| 系统提示段 | `ctx.systemPrompt.section({name, order, text})`（`dsh-system-prompt/lib/index.js:253`）；MCP 服务器的 `instructions` 也作为字面文本进系统提示词 | 「什么时候该用 operator_* 工具、什么时候不该」 |
| 后台任务 | `ctx.jobs.start({kind,label,owner,outputLimitBytes,toolCall,run(handle)})`（`dsh-jobs-local:412`），句柄有 `updateProgress`/`append`，**结算自动转成会话内通知，不用轮询**；需要 `dsh-tool-jobs` 在场 | 长跑的进度/完成/取消 |
| 审批与沙箱 | `ctx.approval`（`dsh-user-approval:128`，policy `ask|never`）；sandbox 模式 `read-only|workspace-write|danger-full-access`；工具可带 `sandbox_permissions` 升级参数并走 `approveEscalation`（`dsh-tool-bash` 就是这么做的） | 「真机 SSH / 花费 / 写语料」三件事的闸门 |
| 子进程 | `ctx.subprocess.spawn(spec)`；导出的 `scrubbedParentEnv()` 会**先剔除 `*KEY*\|*PASSWORD*\|*SECRET*\|*TOKEN*` 与全部 `DSH_*`**，再叠加显式 env 覆盖 | 起 Python sidecar；也是配置面搬家的根源 |
| MCP | `dsh-mcp-client` 一条配置行（`lib/index.js:780`）：`{transport:"stdio", serverName, command, args, env, cwd, toolCallTimeoutMs, reconnect}` → 工具名 `mcp__<serverName>__<tool>`；`toolCallTimeoutMs` 默认 **60,000 ms** | 零 JS 的旁路 |
| Skill 文件 | `<root>/<name>/SKILL.md` 或 `<name>.md`，frontmatter 必填 `name`/`description`，可选 `whenToUse`/`user-invocable`/`disable-model-invocation`；扫描 `<projectRoot>/.dsh/skills`（rank 100）等六级根 | 现成的 `skills/*.md` 可以改造成 DSH skill |
| 会话投影 | `ctx.sessionProjections.register({key, stateSchema, init, apply, wire})` | 可选：把「当前 run 进度」画进会话 |

**两个关键边界**【V】：
- **宿主插件代码跑在宿主进程里，不受工作区沙箱限制**（plugin-manager README 原文：「已安装的 Host 代码在宿主进程内运行，不受工作区沙箱限制」）。沙箱只约束**经由 `ctx.shell`/`ctx.subprocess` 的调用**。这直接决定了「审批摩擦」有多大。
- **MCP 拿不到 jobs / 命令 / 会话投影 / UI**；MCP 只给工具 + 一段 `instructions` 文本。

### 2.3 本机组合的现状【V】

- base 组合里只有 `mcp-resources`（`dsh-base/cordis.patch.yml:492`），**没有 `dsh-mcp-client`** ⇒ 走 MCP 旁路要自己插一行 `@deepseek-ai/dsh-mcp-client`。
- profile 的 `cordis.patch.yml` 里已有本地 LLM provider 配置（`llm-pi-ai` 的 `providers.tokendance` 等）与 `agent-default-model`。**这给了「外层模型配置复用给内层」的具体抓手**：插件/sidecar 可以从同一份来源取 `baseURL`/`apiKeyEnv`/`model`。
- 客户端插件的 HMR 靠**已构建产物**的 mtime 轮询（`dsh-client-hmr`，`/plugins/events` SSE），改源码不会自动生效（`dsh-web-app/cordis.patch.yml` 原话：这条链「在重建 watcher 真正重写 client bundle 之前是空闲的」）⇒ 若要做 UI，得自带构建管线；不做 UI 就别碰 `dsh.client`。

---

## 三、三种形态

### A1 —— MCP 旁路（零 JS）
Python 侧起一个 MCP stdio server（本仓零重依赖，帧协议用 stdlib 实现即可），profile 插一行 `dsh-mcp-client`。模型看到 `mcp__operator__*`。

### A2 —— JS 宿主插件 + Python sidecar
写一个 ESM 插件（照 `beauticode-dsh` 的形状，约 200~400 行）：注册工具 + `/op` 命令 + 系统提示段（+ 可选 jobs）。工具 `execute` 内部 spawn Python sidecar（宿主进程内 spawn，不受工作区沙箱限制），或按路径 insert 安装。

### B —— 深度迁移（**不建议**，列在此处当反面参照）
把 Designer/Coder/Fixer/Search 拆成 DSH 工具、把编排交给 DSH 的模型循环。后果：
- **双大脑**：DSH 的模型循环与 Conductor 的确定性编排器抢「下一步做什么」。
- **丢掉终止权**：`_measurement_drives_loop` 与 `single_candidate_coder` 的 `max_calls==1` 是刻意设计；交给模型自述就退化成主流形态。
- **11 个 prompt 模板 + `agent_runtime` 的 ```agent-action 信封 + `coder_pipeline` 的并行采样会全部搁浅**：这些是单次 prompt→text 管线，没有工具调用概念（全仓 `tool_calls`/`tools`/`function_call`/`mcp`/`json-rpc` **零命中**）。
- 唯一能去掉双循环的 B 变体是「只用本仓的判定层（verifier/measurement/eligibility/oracle），生成层重写在 DSH 里」—— 那是把 3.5 万行里最值钱的判定层与最耦合的生成层切开重做，成本与风险都远超本次评估范围。

---

## 四、优点（A 类，逐条可检验）

| # | 优点 | 为什么成立（证据） |
|---|---|---|
| 1 | **入口从「读 driver + 设 4 类环境变量」降到一句话** | 三个 driver 里的 task 构造是硬编码的（`test_nvidia_matmul.py:98-117`），插件把它变成工具参数 |
| 2 | **三个 driver 收敛成一个稳定入口** | 269+274+438 行里 `build_llm`/`ssh_config`/`print_report` 三份复制；这件事与 DSH 无关，本来就该做 |
| 3 | **长跑拿到进度 / 中断 / 完成通知** | `ctx.jobs` + `updateProgress` + 结算通知；而本仓**已有**可读的持久状态（`execution_snapshot.json`、`resume_from`、`run_contract` 的原子写），status 工具直接读产物即可，几乎不需要新管道 |
| 4 | **任务语义进入对话** | shape/精度/硬件/容差/预算现在是代码里的字面量，可以变成工具签名里的参数 |
| 5 | **复用 DSH 的宿主设施** | 会话落盘与检索、通知、子 agent、多模型切换、审批 —— 这些自己写一遍都不便宜 |
| 6 | **双层可审计** | DSH session 是 append-only 的「模型可见即已记录」，本仓是 contract hash + run_manifest + append-only 语料。两条链可以互指（但必须规定谁是真相，见 §五.5） |
| 7 | **模型分层** | 外层用便宜快的模型做交互与任务构造，内层继续用调过的内核生成模型（`temperature=0.1`/`max_tokens=40960` 是内核生成调过的值，不该被外层默认值覆盖） |
| 8 | **分发与复用** | 可按绝对路径 insert，不必发 npm；可以只在某些工作区/某些 preset 启用；dev 与 prod 用 profile 切换 |
| 9 | **倒逼本仓收口** | 入口一旦变成对外契约，258 处 `print`、free-form `metrics`、环境变量配置面就藏不住了 —— 这正是 doc 20/49 想要的方向 |
| 10 | **可选的人在环** | 真机 SSH / `danger-full-access` / 花费上限可以挂 `ctx.approval`。**默认必须是 `ask` 下的例外通道，不是默认路径**（无人值守是设计立场） |

---

## 五、缺点与代价（按被低估程度排序）

### 5.1 打包前置缺口：`protocol_identity()` 硬编码仓库相对路径【V，最要命】
`run_contract.protocol_identity()`（`run_contract.py:64-95`）对 **47 个源文件**做 repo-relative hash，用来标识「这次运行的协议身份」。一旦代码被搬进 site-packages / npm 包 / 另一个目录，这个 hash 的输入就变了 ⇒ 续跑对账、跨运行比较、golden 全部失真。
连带：没有 `pyproject.toml`；`prompts/`+`skills/` 靠 `Path(__file__).parent` 定位（`prompt_manager.py:46-48`）；`ArtifactStore`/`KernelDatabase` 默认往**仓库内**写。
**结论：任何插件形态都必须先有一个「可安装/可定位的 Python 入口」，否则插件只是把不稳定性包了一层。**

### 5.2 配置面搬家与凭据纪律的正面冲突【V】
现在凭据靠自己的一套纪律（不进仓 + `~/.ssh/config` + 环境变量 + `tests/test_repo_hygiene.py` 机械扫描）。插件化会多出一个配置落点：
- 走 MCP：`env: {...}` 写在 profile YAML 里 —— **若把 `LLM_API_KEY` 直接写进去，等于把密钥搬进一个可能被 git/npm 带走的文件**。
- 走 JS 插件：`scrubbedParentEnv()` 会把 `*KEY*/*SECRET*/*TOKEN*` 全洗掉，**必须显式覆盖**；显式覆盖又回到上一条。
- 正确形态是**只传引用**（env 名 / 文件路径 / `~/.ssh/config` 别名），值由 Python 侧解析 —— 这与 doc 10 §四(a) 记的 DSH credentials seam「配置携带引用，provider 拥有值」是同一条原则。**但这要新写一层。**

### 5.3 沙箱与审批的摩擦：两种都不是好答案【V】
- 走 `ctx.subprocess`/`ctx.shell`：落进 `workspace-write` 默认策略与 `ask` 审批。而本仓要写 `output/`+`logs/`（可能不在会话工作区）、要 SSH 出网、要在远端 `uv init`。**一条命令一次批准 = 无人值守直接失效。**
- 给 `danger-full-access`：把本该收口的边界放开。
- 折中（推荐）：**宿主插件在宿主进程内自己 spawn**（实证不受工作区沙箱限制），把审批留在**工具级** —— 一次批准一个 run，而不是一条命令一次。

### 5.4 桥本身是新的失效面，且本仓已经交过学费【V】
「进程起不来 / 跑完但输出错 / 流错位」三态折叠成 `success=False` 这个缺陷，本仓在 `runner_server.py` 的 `id` 回显设计里专门修过一次（doc 60：`TimeoutExpired` 与「核跑完了但输出不对」同形，Fixer 会对一个不知道成没成的现场开处方）。
桥必须照同一条纪律做：**id 回显、正交上报（超时 ≠ 失败 ≠ 未回显）、异常原文过落盘边界、单一事实来源**。这是要新写的代码 + 新写的守卫。

### 5.5 双账本 / 双日志 / 两个模型：谁都不自动知道对方存在
| 事实 | 现状落点 | 插件化后落点 | 必须规定 |
|---|---|---|---|
| token 花费 | `llm_client.CostTracker`（按 caller 分账），进产物 | DSH 自己的 token meter | 报表里两个数**并排印**，各自写清口径（doc 58「缓存语义猜错方向」的教训：口径不写清，读者会把两个数当同一个） |
| 运行日志 | `logs/*.log` + `capture_execution` 抓走的 258 处 stdout | DSH session jsonl | **内核事实以本仓产物为准，会话事实以 DSH session 为准** —— 这是 doc 10 §二「正交结果独立上报」的跨系统版本 |
| 模型 | `LLM_BASE_URL`/`LLM_MODEL` | profile 的 `agent-default-model` | 内层绝不能默默继承外层模型（内核生成的温度/上限是调过的） |

### 5.6 时间尺度的错配【V】
一次 `Conductor.execute` 是分钟~小时级；DSH 的工具调用与 MCP 的 `toolCallTimeoutMs` 默认 **60 s**（可调），审批是「一次调用」粒度。
⇒ **「起 + 查」两段式基本是必须的**：`run` 立刻返回 run_id，`status`/`report` 读产物。好处是本仓已有可读的持久状态，代价是「取消」需要自己定义并测（进程组、远端 runner 清理 —— `session_control.py` 已有 owned-runner 身份与清理的雏形，但那是 SSH 那一层）。

### 5.7 版本耦合：拿主线稳定性换体验【V】
DSH 是 **0.2.1-rc.2**，官方明示 developer preview / 会有破坏性变更（doc 10 §一 已引用过原文）；而插件版本兼容在**导入前**强制，不匹配直接拒载。
⇒ 把插件做成**唯一入口**是不可接受的。**必须保留 driver 与离线测试路径不删**，插件只做叠加。

### 5.8 测试无法共享【V】
3,161 条离线测试在 DSH 侧跑不了（它们在 `tests/` 里，靠 `conftest.py` 的进程内 hook 隔离）。插件侧只能加**契约测试**：工具 schema、命名空间前缀、超时与取消、三态失败上行、参数白名单。不要试图把内核测试搬进 DSH。

### 5.9 上下文挤压的方向是反的
本仓的 prompt 是按 §21.3 精挑的八项证据（`structured_evidence`）；外层若把内层日志**全量**拉进上下文，会同时污染两边的预算。而**薄封装天然规避了这件事**（内层 prompt 完全不受外层影响）—— 这正是推荐 A 而不是 B 的又一条理由。反过来，外层只需拿到**结构化摘要 + 产物路径**。

### 5.10 立场冲突：DSH 的审批天然是「可关的开关」
设计 §3.5 的不变量清单（日志 / Kernel DB / Failure DB / failure gate / 双确认）在 DSH 侧没有对应物。插件化最大的风险不是技术，而是**顺手把这些做成工具参数**。
必须写死一张**允许进工具签名的参数白名单**（见 §六.3），并加结构守卫（照 `tests/test_spine_wiring.py` 的「没有开关」写法）。

---

## 六、对比表与推荐路线

### 6.1 四形态对照

| 维度 | 现状（仓库 driver） | A1 MCP 旁路 | A2 JS 插件 + sidecar | B 深度迁移 |
|---|---|---|---|---|
| 语言/代码量 | Python，3 份 driver | Python，+1 个 MCP server | +200~400 行 ESM + Python 入口 | 重写生成层 |
| 安装 | clone + 环境变量 | 插一行 `dsh-mcp-client` | 按路径 insert 或发 npm | 不适用 |
| 自然语言入口 | ✗ | ✓ | ✓ | ✓ |
| 斜杠命令 | ✗ | ✗ | ✓ | ✓ |
| 系统提示段 | ✗（内层自有） | △（只能靠 MCP `instructions`） | ✓ | ✓ |
| 长跑进度/通知/取消 | ✗（盯 stdout） | △（自己轮询状态） | ✓（`ctx.jobs`） | ✓ |
| 审批闸门 | ✗ | △（调用级） | ✓（工具级） | ✓ |
| 会话落盘/审计 | △（自有产物） | ✓ + 自有产物 | ✓ + 自有产物 | ✓ |
| 保住「终止归测量」 | ✓ | ✓ | ✓ | **✗** |
| 新增失效面 | — | 桥的三态、超时 | + 版本兼容、HMR/构建 | 全部 |
| 维护成本 | 低（但是三个 driver） | 低-中 | 中 | 极高 |
| 建议 | 保留 | **先做** | 有价值再加 | 不做 |

### 6.2 推荐路线（分阶，最小代价）

**Phase 0（与 DSH 无关，无条件该做）**
把三份 driver 收敛成一个稳定入口：
```
python -m operator_agent.cli run --request <file|-> --backend ssh|mock --run-root <dir> --json-out <path>
```
输出一份**结构化 run 结果**：`run_id` / 产物目录 / 轮次数 / 最好加速比 / `claim_level` + 五资格位 / 停因 / 拒绝理由 / 花费。同时修 `protocol_identity()` 的路径依赖。
验收：同一份 JSON 在 MockGPU 与真机各跑通一次；`tests/` 全绿不变；driver 行数显著下降。

**Phase 1（A1：验证价值，零 JS）**
MCP stdio server，暴露 4 个工具：
- `operator_run`：请求（自然语言或结构化）→ 起一次开发，**立刻**返回 `run_id` + 阶段；
- `operator_status`：`run_id` → 阶段/轮次/当前最好/是否在等 LLM/是否卡住；
- `operator_report`：`run_id` → 结构化摘要 + 产物路径（**只给摘要与路径，不灌日志**）；
- `operator_capabilities`：不跑活，回答本机/远端有没有可用 GPU、SSH 别名、可用 DSL。
profile 插一行 `@deepseek-ai/dsh-mcp-client`；`toolCallTimeoutMs` 调大或干脆只用于 `status/report`。
验收：DSH 里一句话 → MockGPU 端到端出报告；真机一次；桥的**三态失败**（起不来 / 跑完失败 / 流错位）各有一条契约测试。

**Phase 2（A2：要命令与提示词时才做）**
小 ESM 插件：`/op run|status|report` 斜杠命令 + `ctx.systemPrompt.section` 写清「什么时候用、什么时候**不要**用（没有 GPU 时、纯离线回归时）」。可选把工具直接注册成 `operator_*`（而非 `mcp__operator__*` 命名空间）。

**Phase 3（长跑与闸门）**
用 `ctx.jobs` 包一次 run（`updateProgress` 喂阶段/轮次，结算通知替代轮询）；把「真机 / 花费 / 写语料」三件事接 `ctx.approval`，**只作为外层闸门**。

**Phase 4（可选）**
把 `skills/triton_skill.md`、`optimization_recipes.md` 转成 DSH skill 格式（补 frontmatter + `whenToUse` + **不适用条件**）。注意 doc 17 的判据：**没有校验基建的「技能包」等于 prompt 形容词** —— 转格式不等于有价值。

### 6.3 参数白名单（接口红线，建议直接写成守卫）

| 可以进工具签名 | 绝对不许 |
|---|---|
| 任务语义（kernel_type / dsl / 精度） | 任何放宽 L2 容差 / 跳过 L1 静态守卫的字段 |
| 形状、case、目标函数权重 | `require_double_confirm` / `skip_remeasure` 之类 |
| 后端选择（`ssh`/`mock`）、SSH 别名**引用** | 关闭 Kernel DB / Failure DB / 日志 |
| 预算与上限（`max_rounds`、LLM 调用数上限）—— §10.2 明确「花不花这次钱」是任务级决定 | 降低 failure gate、放大 `heartbeat_interval` |
| `run_root`、超时、是否后台 | 「本次测量是否可信」的判定权（那是 measurement/eligibility 的职责，不许由外层 LLM 代答） |

### 6.4 反面清单（明确不做）
1. 不把 Designer/Coder/Fixer/Search 拆成 DSH 工具（双大脑、双计费、丢终止权）。
2. 不把 `kernel_db`/`failure_db` 换成 DSH session/storage —— 语料是**判据的产物**，不是会话记录。
3. 不为插件在 Python 侧新增任何安全开关（§3.5）。
4. 不让外层 LLM 判「这次测量是否可信」。
5. 不为「能在 DSH 里跑」放宽任何容差或跳过反作弊。
6. 不把 `output/`+`logs/` 的真相搬进会话 —— 会话只放摘要与指针。

### 6.5 什么时候「值得」，什么时候「不值得」

**值得**（满足两条以上）：你要频繁在对话里改 task/shape/精度；你需要「一个 run 的进度与中断」作为一等体验；你要把这份能力交给不懂环境变量的人；你能接受多一层桥 + DSH prerelease 的版本风险。

**不值得**：主要用途是无人值守批处理 / 基准对照（那用 `harness_pass_rate` 这类 driver/benchmark 更合适）；你要的是「入口越少越好的可审计一次运行」；你的运维不接受 Node/profile 这一层。

**建议的最终立场**：**叠加，不替换。** driver 与离线全量测试继续当唯一真相；插件只是人机交互入口。这与 doc 10 §七「不采纳其代码/插件体系」不冲突 —— 那次拒绝的是**把项目迁进 Cordis**，这次要的是**借它的宿主设施**。

> ⚠️ **2026-10-18 追加（场景更新）**：用户补充了真实场景 —— **前期 HITL（能看到和干预进程），后期若效果足够再完全自动化**；并追问开源算子开发工具的形态。**§六 的推荐顺序因此被修订**：起步形态不是 A1/A2（整段包装恰好把进程藏起来），而是 **形态 C = skill（知识）+ 判定工具（每一步）+ Conductor 后端（后期自动化）**。论证见 **§七–§九**。

---

## 七、场景更新：先 HITL、后自动化

### 7.1 先说一个正面冲突：HITL 与本仓已写进规范的设计立场相反

`README.md:8` 与设计规范 §0 表都写着：运行模式「**无人值守** —— 无人在环内裁决歧义或发现失效」。
HITL 不是一个入口层的小改动，它触到的是一条**成文立场**。从本仓自己的证据反推当初为什么这么定：判定权归测量（`conductor.py:3194` `_measurement_drives_loop`），而**人一旦能改判，`oracle.admit_to_corpus` 那道唯一入库门就不再唯一**。本仓为「能一键关掉的尺子会先被用在无人值守的夜里」删过一次构造参数（`conductor.py:417` 的墓志铭）—— **HITL 是同一族风险的"人肉版本"**。

### 7.2 出路：HITL 只开在「目标与花费」面，判定面不开孔

人的介入只有三种**合法**形态：

1. **改目标**：算子 / 形状 / 精度 / 硬件 / 容差要求；
2. **加候选**：自己写一版、让模型再写一版、指定修哪一版；
3. **加预算**：上不上真机、L3 跑几次、LLM 调用上限。

**人不改判**：不得推翻 L2 失败、不得宣布一次测量有效、不得绕过反作弊。
在实现上这意味着 HITL 是「**再来一次**」，不是「**override**」—— 正好落在 `failure_route` 的五路处方与 `oracle` 唯一门禁上，**不需要任何新开关**。这条同时解决了 §5.10 那条立场冲突：DSH 的审批只管「能不能发起这次花费」，管不到判决。

### 7.3 三个粒度（代价递增）

| 粒度 | 人能看到 | 人的动作 | 承载物 | 代价 / 风险 |
|---|---|---|---|---|
| **L0 事前授权** | 计划全貌：任务 / 契约 / 预算 / 目标卡 | 批准或否 | **已有**：`benchmarks/workload_codegen_h20.py` 的 prepare → `--approved-plan` + `--plan-hash`；执行时**重新推导计划并比 digest**（`workload_codegen_h20.py:647-649` "execution differs from separately authorized exact plan"） | 低。**但看不到每轮** |
| **L1 轮间暂停** | 每轮：候选 / 判定 / 下一步 | 继续 / 改方向 / 停 | **无**，需要在 Conductor 里加钩子 | 中-高。与「无人值守」立场冲突；且 `execution_snapshot.json` 是**崩溃恢复**用的，不是「等人」用的，暂停语义要重做 |
| **L2 每步一个模型回合** | 每一步：构思 → 写码 → 验证 → 处方 | 随时插话、直接改代码、换方向 | **原语已有**（见 8.4） | 中。外层循环取代 Conductor 当驱动器 |

**推荐 L2 起步，L0 作为后期的自动化开关。** 理由就是用户的原话：「**看到和干预进程**」是主需求，而整段包装（A1/A2）恰好把进程藏起来——一次调用就是一个分钟~小时的黑盒。

### 7.4 L2 现在就可做，因为单步原语已经存在（本节最重要的一条）

这些原语都是依赖注入的、**不依赖 Conductor**：

| 原语 | 落点 | 说明 |
|---|---|---|
| `SingleCandidateCoder(llm, task, sketch, test_inputs, policy=None)` | `single_candidate_coder.py:395` | 一次 LLM 调用产出**一个**候选：无 Planner、无采样、无修复、无重试；返回严格 JSON（`code` + `rationale`），带静态预检 |
| `Verifier(backend, *, cuda_ruler_profile=None)` + `verify(kernel_code, task, test_inputs, expected_output=None, target_gflops=..., stop_after=...)` | `verifier.py:309` / `:326` | **只吃源码文本**，不要求候选来自本仓的 Coder ⇒ 外层模型（或人）写的核可以走完全同一条 L1/L2/L3 + measurement + eligibility + oracle 链 |
| `IntentParser(llm=None)` | `intent_parser.py:52` | 自然语言 → `TaskSchema`，**规则回退可用**（无 LLM 也能跑） |
| `with_default_correctness(task)` | `correctness_profiles.py:144` | 正式新任务入口，带 `builtin-l2-v1` 套件标记 |
| `failure_route.classify(...)` | `failure_route.py:106` | 五路处方（`fix` / `debug_fix` / `search` / `retry` / `hold`），**内容决定而不是类别名决定** —— 正好当"给人和模型看的下一步" |
| `report.write_report(artifact_dir, *, artifact=..., task=...)` | `report.py:671` | 独立可用，产出 `report.md`（含「我无法确立的事」章节） |
| `correctness_preflight(task, test_inputs=None)` | `verifier.py:556` | 静态，零 GPU 秒 |

⇒ **「skill 负责知识、判定工具负责判决、DSH 负责循环与人机界面」不是重写，而是把已有原语暴露出来。**
而且因为走的是**同一套** `Verifier`/`measurement`/`eligibility`/`oracle`，HITL 阶段的产物与自动化阶段**同格式、同判据**，语料与血缘不分叉 —— 否则前期攒的经验到自动化那天会全部作废（这正是本仓最贵的缺陷类型：「判据算了但没人调」的镜像）。

**诚实边界（不要读大）**：判定**判据**在 `Verifier` 里（含 L3 出口的 oracle 观察，`verifier._publish_oracle_observation`），但**运行外壳**仍长在 Conductor 上 —— run identity（`run_contract.declaration`）、角色目录（`run_layout`）、产物落盘（`ArtifactStore`）、入库（`Conductor._save_to_db` → `oracle.admit_to_corpus`）都在 `_execute_inner`/`_finalize_run` 里。所以 Phase 0 真正要做的事是：**把这些外壳从 Conductor 里抽成一个可被两种驱动共用的薄层**（或让 HITL 工具驱动一个"每轮只跑一次"的精简 Conductor 路径）。这不是新算法，但也**不是零工作量**。

### 7.5 HITL 的验收标准（可判）

1. 任意时刻人能回答三件事：**现在到哪一步** / **上一个判定为什么**（含 refusal 原因与 `failure_route` 处方）/ **下一个候选从哪来**；
2. 人的每次介入在产物里**有迹可循**（改动进血缘，而不是聊掉）；
3. 判定面**零开关**：工具签名里没有「放行 / 跳过 / 关库」类字段（§6.3 的白名单继续成立）。

---

## 八、开源算子开发工具的形态：确实是 skill

**用户记得没错。本仓 `docs/research/17-算子开发agent与skill实现盘点.md`（2026-09-29）已经一手核过这件事**，结论是原文标题那句：「**这条路线已经收敛成『发一个 skill』，不是『写一个 agent』**」。

### 8.1 doc 17 的一手证据（【V】本仓文档，标注为 ✅ 的那些）

| 项目 | license / ★ | 形态 | 关键事实 |
|---|---|---|---|
| `TongmingLAIC/AKO4ALL` | MIT / ★370 | **纯 skill** | git tree 全量 = **18 个 blob**，`SKILL.md` 16,156 B + `bench/kernelbench/bench.py` 51,677 B + `HINTS.md` + `ITERATIONS.md`；**没有一行 agent 循环代码** —— 循环是 Claude Code 读完 `SKILL.md` 之后的行为 |
| `TongmingLAIC/AKO4X` | MIT / ★75 | **agent + 产物 + 反作弊 + 离散度** | `master/master.py` + `templates/skills/{triton,cuda,cute-dsl,tilelang,bench,profiler-ncu,sanitizer}` + `scripts/cheat_check_modal.py` + `reference/` 九个真产物目录 |
| `NVIDIA/skills` | Apache-2.0 / ★3478 | **skill 库（官方占位）** | 每技能一目录：`SKILL.md`(frontmatter) + `references/` + `examples/` + **`evals/evals.json`** + `BENCHMARK.md` + 签名 `skill.oms.sig` + **`scripts/validate_*.py`**；含 `skills/triton/`、`tilegym-converting-cutile-to-triton` |
| `NVIDIA/SOL-ExecBench` | Apache-2.0 / ★301 | 评测器（benchmark） | 只有题目集与评测器 |
| `AMD-AGI/GEAK` | Apache-2.0 / ★181 | agent（ROCm） | 另一硬件线 |
| `tensormux/kernel-skills` | MIT / ★79 | skill 库 | — |

### 8.2 doc 17 立的那把尺（本轮最值得记住的一句）

> **有没有配套校验基建（frontmatter 触发语 / schema / evals / proof / validate 脚本），就是「技能包」与「prompt 形容词」的分界线。**

- 有基建：`NVIDIA/skills`（evals + sig + validate 脚本）、`AKO4ALL`（`description` 里把**用户可能的原话措辞**列全 + `HINTS.md`/`ITERATIONS.md` 固定格式 + `bench.py` 契约）。
- 没基建：`operator_agent/skills/*.md` **今天属于后者** —— 它们是按 `## N.` 切段、由 `{{skill}}` 静态注入的 prompt 片段，**没有 frontmatter、没有触发语、没有不适用条件、没有校验器**（doc 17 §七 已判定）。

### 8.3 skill 形态**装不下**什么（这正是本仓的差异化资产）

AKO4ALL 的 `bench.py` 已经是这一类里最"重"的契约了，但对照本仓，它没有：

- **测量有效性三态**（`measurement.assess`：valid / degraded / uninformative + launch dominance）；
- **资格五位**（`eligibility.derive_eligibility`：这个数被允许造成什么后果）；
- **同卡配对与竞争检测**（`placement` + `gpu_claim`）；
- **反作弊两层**（静态 AST + 运行时盾）与**唯一入库门**（`oracle.admit_to_corpus`）；
- **冻结契约与 run identity**（`contract` / `run_contract`，47 个源文件的协议 hash）；
- **处方式失败路由**（`failure_route` 五路 + `fix_not_rewrite`）。

⇒ 所以正确的分工是：**skill 装知识（方法、不变量、何时不适用），本仓装判定**。这两者相加才同时具备「NVIDIA/skills 的形制」与「AKO4X 那个级别的判定」，而今天没有开源项目同时具备。

### 8.4 对我方最直接的三条可吸纳（doc 17 §七，代价已标）

| 动作 | 参照 | 代价 |
|---|---|---|
| 给 `skills/*.md` 补 frontmatter 触发语 + 「**When this skill does NOT apply**」段 | `AKO4ALL/SKILL.md` | 低，纯文本 |
| 抄 `bench.py` 的**单契约文件**形状（输入生成 / 正确性 / 计时 / 落盘只在一处） | `AKO4ALL/bench/kernelbench/bench.py` | 中 |
| 反作弊从静态串匹配升级成**动态 staleness 审计**（固定指针改数值，输出不变即拒） | `AKO4X/scripts/cheat_check_modal.py` | 低（约 50 行，doc 11 §四 已估） |

> **本文的取舍**：§8.3 的判定层是本次评估里**唯一比开源参照物强的地方**，因此形态选择必须围绕它展开 —— 这就是下一节「三层架构」的理由。

### 8.5 2026-10-18 复核（本轮亲身取数，含一个**直接对手**）

本节全部是**本轮亲自 curl 取到原文**的一手结果（本机 `http_proxy=127.0.0.1:7887`；`api.github.com` 取到约 40 次后按 IP 限流，`raw.githubusercontent.com` 全程可用）。**这推翻了 §十 里"未重新联网复核"那条边界。**

**(a) doc 17 的两条主要结论都还成立，但要按新数字更新：**

| 项目 | 2026-09-29（doc 17） | 2026-10-18（本轮） | 形态 |
|---|---|---|---|
| `TongmingLAIC/AKO4ALL` | MIT ★370 | **MIT ★376，push 2026-09-15**，`SKILL.md` 仍是 16,156 B（字节数与 doc 17 完全一致） | **skill bundle**（顶层无循环代码）。README 自述 *"packaged as a single Claude Code skill… The skill is a single SKILL.md protocol document"*，装法 `git clone … ~/.claude/skills/ako4all` |
| `TongmingLAIC/AKO4X` | MIT ★75 | **MIT ★81，push 2026-09-15**，有 `pyproject.toml`（hatchling, `requires-python>=3.10`） | **独立 Python agent**（`master/master.py` 39,594 B + `spawn.py` 41,085 B），但它**同时**下发 `templates/skills/{triton,cuda,cpp,cute-dsl,tilelang,bench,benchmark,profiler-ncu,sanitizer}/SKILL.md` 与 `templates/agent/claude.json` ⇒ **skill = 知识载荷，Python master = 循环** |
| `NVIDIA/skills` | ★3478，含 `skills/triton/`（⚠️ 未复验） | **★3635，push 2026-10-17**，8037 tree 条目，**418 个 skill 目录、其中 17 个 kernel/GPU 向**；**`skills/triton/` 今天 404**，README 目录表里也没有它 ⇒ doc 17 那条要么已改名/合并、要么当时就是误记。Triton 方向的正主是 `tilegym-*`（`tilegym-converting-cutile-to-triton/SKILL.md` 本轮实测 17,064 B，frontmatter 含 `name/description/license/tools/metadata{…}`，比 AKO4ALL 的两个键丰富得多） | **skill 目录 + 评测基建 + 四宿主插件清单**（`.claude-plugin/`、`.codex-plugin/`、`.cursor-plugin/`、`.agents/plugins/`） |

NVIDIA 那套的**校验基建已经成型**（直接坐实 doc 17 §五那把尺）：`docs/evaluating-agent-skills.mdx` 写明 **SkillEvaluator 三层**（静态校验+安全扫描 → 语义去重 → **带/不带 skill 的实机 agent 对照**），且 *"Tier 3 cannot run without a task set… Provide one at `evals/evals.json`"*；`skills/tilegym-converting-cutile-to-triton/BENCHMARK.md` 记了**评测日期 2026-10-16**、evaluator v1.5.6、两个宿主 agent、5 个任务（1 正 4 负）、k8s 沙箱、数据集 sha256、correctness 80→100%。装法是 `npx skills add nvidia/skills`（vercel-labs 的 skills CLI）+ 每宿主 `--agent cursor` 这类参数。

**(b) 本轮新发现的两个 skill 项目（都晚于 doc 17）：**

- `minho-fan/kernel-opt-agent`：*"four **Claude Code / Codex skills** for GPU kernel work"* —— `kernel-KBS`（知识库检索）/ `kernel-benchmark` / `kernel-profile`（NCU 剖析与瓶颈分类）/ **`kernel-loop`（迭代优化编排器）**。`kernel-loop` 的自我定位值得逐字抄：
  > "This is an **orchestration skill**. It does not own profiling, KBS search, or benchmarking scripts. It owns only the **loop rules, output layout, local report templates, and the iteration gate**."
  且 *"The loop is valid only when every version is correct, profiled, evidence-backed, and **accepted by the gate script** before the next version is created"*（gate = `scripts/check_iteration_gate.sh`）。
  ⇒ 这是**「skill 定规则 + 确定性闸门说了算」**的现成先例，与本节推荐的形态同构，且比 doc 17 晚。
- `slowlyC/agent-gpu-skills`：`cuda-skill` / `cutlass-skill` / `tilelang-skill` / `triton-skill`，`install.sh --agent codex|cursor|claude-code` 装到 `~/.cursor/skills/` 这类目录 —— **纯知识检索类**（文档快照 + 上游源码检索入口），不执行候选。

**(c) 反例（说明「skill」这个词被用得多松）**：`tensormux/kernel-skills` 的 SKILL.md **完全没有 YAML frontmatter**，靠旁挂 `skill.json` + npm registry，README 明说 *"not a benchmark harness… not an autonomous coding agent… does not execute generated code"*。另一头，Intel/HF 的 `xpu-kernels` skill 把整套 CoVeR 循环压进 `SKILL.md + scripts/ + references/`，**但编译/上传由外部 Rust CLI `kernel-builder` 承担** —— 又是「skill 管交互、服务管验证」。

**(d) ⚠️ 最重要的发现：DSH 生态里已经有人做了同一件事。**
`xieTwim/dsh-kernel-opt`（**MIT**，★5，创建 2026-08-13，push 2026-09-07，TypeScript）—— **一个 DSH 插件，README 标题就是「实时查看并干预算子优化过程」**：

| 维度 | 它怎么做（本轮一手读到） |
|---|---|
| 形态 | **插件**，不是 skill：`package.json` 有 `dsh.bundle.patch` 与 `dsh.client{platform:"web", inject:[…dsh-client-runtime, …dsh-client-ui-slots, …dsh-client-locale]}`；`private: true` ⇒ 从 git 装：`dsh plugin --profile web add "github:xieTwim/dsh-kernel-opt"` |
| 给人看什么 | 会话里多一个「评测」页签：延迟曲线、状态、当前方案、最近五次评测、来源命令、监督记录；数据**全部从会话日志投影派生**（零插件侧状态）⇒ 回放与实时渲染一致 |
| 干预怎么发生 | 「随时发消息即可改变方向」；循环可随时打断（停止按钮与中断回合**立刻生效且不做收尾**）；用户消息优先于循环要发的内容 |
| 自动化怎么收口 | 「按预算运行并**自行收尾**」：预算只计优化评测，用完或连续两轮无新评测 ⇒ 先要一次收尾（装回最优版本 + finalize + 总结）再停 |
| 评测器 | **不内置也不适配任何评测逻辑**：① 契约行 `KERNEL_EVAL={"artifact":…,"latency_ms":…,"correct":true}`（模型经 bash 跑评测脚本，面板从 shell 结果解析）；② 工具结果 JSON（**任何名字含 `kernel_evaluate` 的工具**默认命中） |
| 信任模型 | 面板**明牌标注信任级**：`工具`（模型伪造不了）vs `智能体评测`（模型可控的 stdout，旁边附产生它的命令行）；`correct !== true` / reward-hack / error 的点**永不参与 best**；收尾时插件**自己把那条命令重跑一遍**做复测（可关；命令超 300 字符则不可复测并明示） |
| 它自己撞到的测量问题 | 同一份没改过的 kernel 在 13 个 Modal 容器里自报 **×35.5 / ×46.3 / ×53.2 / ×35.5**，改用**统一分母**重算后 **×34.9 / ×35.5 / ×34.9 / ×35.5**；并写明「**分母漂移只能报，不能修**」，阈值依据是同一台 A100 上重测参考的 5.4% vs 每容器各测一次的 53% |
| 去重与幻影点 | 契约行按 payload 在**整个会话内**去重（实测 15 次评测因 job + 轮询被数成 20 个点）；纯读取命令与存档类工具不计点，后台任务读取器照常计点 |
| 安装前提 | 「面板属于 DSH 的 **web** 界面，所以插件要装进 web profile」；需要 `PATH` 里有 pnpm；`lib/` 已提交，插件本身不需要构建 |

**三条直接推论**（本轮最有决策价值的部分）：

1. **「薄封装 + 实时面板 + 可打断 + 按预算自行收尾」这条路已被一个 MIT 插件走通**；而它自述的兼容范围是 **DSH 0.1.0-rc.6 — 0.1.2-rc.1**（peer `<0.2`），**本机运行时是 0.2.1-rc.2** ⇒ 直接装会落在版本豁免那一档（`allow-version`）。这既证明"插件路线可行"，也**用实例坐实了 §5.7 标记的版本耦合风险**。
2. **它的信任模型是本仓判定层的"轻量版"**：工具通道 vs 自报通道、reward-hack 标记、`correct!==true` 不进 best、收尾复测 —— 与 `measurement_validity` / `eligibility` / `oracle.admit_to_corpus` / 两轮确认同域，但**它止步于"这个数可不可信"的第一层，本仓还管"这个数被允许造成什么后果"**。
3. **它的评测器是 AKO4ALL 的 `bench.py` 改写来的**（README 致谢写明）⇒ 生态里的实际拼装方式是：**知识/契约来自 skill 生态，面板与循环来自宿主插件，判定层谁都没占**。本仓要占的正是**第三格**——而它已经被量化出必要性了（上表 ×35.5→×34.9 那组数就是 `measurement.assess` 要治的病）。

---

## 九、修订后的推荐（覆盖 §6.2 的 Phase 顺序）

### 9.1 三层，共用一个 Python 入口

| 层 | 内容 | 服务于 |
|---|---|---|
| **知识层** | **1 个 DSH skill**：`SKILL.md` + `references/`，从 `operator_agent/skills/*.md` 改造（补 frontmatter 触发语 + 何时不适用 + 怎么读处方） | HITL 与自动化都读它 |
| **判定层（工具）** | `operator_prepare`（NL/结构化 → 冻结契约 + 套件 + 输入 + **run_root**）· `operator_verify`（源码 → L1/L2/L3 + 正交事实集 + 处方）· `operator_report` · `operator_capabilities`；（可选）`operator_generate_candidate`（走 `SingleCandidateCoder`，一次有界） | **前期 HITL 的每一步** |
| **编排层（后端）** | 现成的 `Conductor`，一个 `operator_run_autonomous` 工具 | **后期完全自动化** |

三个入口共用 Phase 0 做出来的 Python CLI 契约与同一套产物格式 ⇒ **不是三条路，是一条路的两端**。

### 9.2 与 `dsh-kernel-opt` 的相对位置（决定要不要自己写面板）

§8.5(d) 那个插件把「面板 + 循环 + 信任级」这一层已经做完了，而且它**评测器无关**。所以本仓有三种相对位置，成本差别很大：

| 位置 | 做什么 | 成本 | 何时选 |
|---|---|---|---|
| **甲：做它的评测器** | 让本仓的验证路径产出一条契约行（或注册一个名字含 `kernel_evaluate` 的工具，结果带同名字段），把本仓的判定塞进它的 `native_metrics`（数值映射）+ `advisory`（文本），复用它的面板与信任级 | **低**（一个输出适配器 + 一次契约测试） | 想**最快**验证「人看得见每一步」这个假设是否真的解决问题 |
| **乙：自己写插件，判定层照旧** | 自己出 ESM 插件 + preset（照它的形状），但把判定完全换成 `Verifier`/`measurement`/`eligibility` | 中（面板/投影要自己做；它用「会话日志投影」的写法可直接借鉴） | 甲的契约装不下本仓要暴露的东西时 |
| **丙：只用 skill + MCP 工具，不要面板** | 靠对话本身当"面板" | 低-中 | 接受"没有曲线图"，只要每步可读可干预 |

**注意甲的边界**：它的契约是**为评测点设计的**（哪个文件、是否正确、延迟、分母），塞得下"这一轮的数"，但**塞不下**定向事实集、覆盖缺口、`claim_level`、`ImplementationDecision`、"我无法确立的事"。所以甲不适合当终点，只适合当**第一步验证**：先用最小代价证明「人的可见性与干预」这件事值得做，再决定要不要走乙。

**另一条必须提前想清的**：走乙时会不会与它**重复造轮子**。诚实的说法是：**面板与循环的轮子它已经造了，本仓的差异化只在判定层**。所以乙的正当理由是"本仓的判定必须原生地进产物与报告"，而不是"我们也能画曲线"。

### 9.3 Phase 顺序（相对 §6.2 的变化）

1. **Phase 0（不变，无条件先做）**：稳定 CLI + 结构化 run 结果 + 修 `protocol_identity()` 的路径依赖。
   **✅ 部分完成**：CLI（`python -m operator_agent prepare|verify|capabilities`）已落地，见 §9.6；
   `protocol_identity()` 的路径依赖**仍未修**（它还不影响 mock 路径，但打包前必须修）。
2. **Phase 1（改）**：**skill + 判定工具**，在 DSH 里以 L2 粒度跑通一次 HITL 全流程（人看得见每一步、能改代码、能改目标）。装载方式三选一：MCP（零 JS）／小 ESM 插件／**甲方案（直接做 `dsh-kernel-opt` 的评测器）**。
   **判定工具侧 ✅ 已落地**（§9.6）；**已装进 DSH**（MCP，§9.6 续）；
   **整段自动化入口 ✅ 已落地**（`operator_run`/`operator_status`，§9.8）。
3. **Phase 2（新）**：把 `skills/*.md` 转成规范 DSH skill（frontmatter + `whenToUse` + 不适用条件），并**补校验基建**（否则按 doc 17 的尺子，它仍是 prompt 形容词）。
4. **Phase 3（原 A1/A2 降级到此）**：`operator_run_autonomous` = 整段交给 Conductor；用 `ctx.jobs` 包长跑、`ctx.approval` 管花费与真机。**这一步等价于「效果足够之后才完全自动化」**。
   ✅ **可行的前置证据已有**：§9.7 的三轮里，判定/处方/资格/报告都已经在真机上产出；但「外层写错→按处方改对」的对照还没做，**所以现在还不能说「效果足够」**。
5. **Phase 4（可选）**：动态 staleness 反作弊、单契约 `bench` 文件等 §8.4 的吸纳项。

### 9.4 为什么这个顺序，而不是先做整段包装

| | 先做整段包装（原 §6.2） | 先做 skill + 判定工具（本节） |
|---|---|---|
| 人的可见性 | 只有起止两头，中间是黑盒 | 每一步都在对话里 |
| 干预 | 只能停/重跑 | 改目标、改代码、换方向，随时 |
| 前期产出 | 一份最终产物 | 每轮的判定与处方（**可复盘的决策记录**） |
| 后期自动化 | 已经到了 | 增量：加一个工具，判据与产物完全不变 |
| 主要风险 | 前期看不见 → 无法建立对判定的信任 | 外层模型写的核质量未知（**需要实测**） |

最后一行是这套方案唯一的**未验证假设**：外层模型（DSH 侧，当前 profile 是 `deepseek-v4.1-flash`）自己写 Triton 的质量，是否足以让判定层给出有意义的信号。**这一点必须先用几轮真机实测才能下结论**；如果外层写核质量太差，退路是保留 `operator_generate_candidate`（走本仓的 `SingleCandidateCoder`，prompt 与约束都是调过的），人仍能看到并干预每一步，只是"写"这一步交回内层模型。

### 9.5 第一步验证：**已跑通**（2026-10-18 实测）

产物：`benchmarks/demo_hitl_chain.py`（零 LLM / 零 GPU / 不联网）+ `tests/test_demo_hitl_chain.py`（**11 条守卫，1.34 s**）。
运行：`python3 benchmarks/demo_hitl_chain.py`。

链路（全程不经过 `Conductor`）：

```
IntentParser(llm=None)          # 规则分支，零 LLM
 → with_default_correctness     # builtin-l2-v1 套件
 → 外层声明 shapes=((64,64,64),)
 → freeze_from_task             # 契约哈希 + required_cases
 → contract.cases[0].materialize()   # 契约拥有输入，两次造逐元素相同
 → Verifier(backend=MockGPU()).verify(code, task, inputs, stop_after=...)
 → failure_route.classify(...)  # 五路处方
 → report.write_report(...)     # report.md +「我无法确立的事」
```

四个场景的实测读数（**这一栏是屏幕上的真值，未修饰**）：

| 场景 | 判定 | 处方 | 资格掩码 V T P R C |
|---|---|---|---|
| A 错的核（结果 ×0.9） | L1 ✓ / L2 ✗（`max_error=2.9`，`first_bad@0`，形态 `whole_tensor`） | `debug_fix`，`fix_not_rewrite=True`，带"整张量错"的方向提示 | — |
| B 对的核，契约**没**要求真机 | L1 ✓ / L2 ✓（`max_error=0`）/ L3 ✓（`speedup≈1.0–1.3x`） | `hold`（全部通过） | **✓✓✓✓✓** |
| B' **同一个核**，契约 `required_caps=('real_device',)` | L1 ✓ / L2 ✓ / **L3 ✗，`l3_pass_refused=True`** | `search`（见下面的接缝） | **·····** |
| C Triton 源码，`stop_after=L1` | 只有 L1 ✓（不谎报 L2/L3） | `hold` | — |

**三个结论，按重要性：**

1. **判定层确实可以脱离 Conductor 被串起来**，而且**零 GPU、零 LLM、不联网、1.3 秒**。所以「HITL 形态不是重写」这句话现在是实测过的，不再是推断。
2. **B 与 B' 的杠杆只有契约里的一格 `required_caps`**：同一份源码、同一份输入、同一条读数，`validity` 从 `valid` 变 `uninformative`，五位从全开变全关，理由字符串逐字点名「任务要求的能力执行侧没有: real_device —— 这批数字不是在满足要求的设备上测出来的，『通过/不通过』都没有对象」。**这不需要任何开关**，正是设计 §3.5 想要的那种形状。
3. **⚠️ 撞出一个接缝（本节最值钱的一条）**：`failure_route.classify(verify_results)` 只按「哪一级失败 + 文本标记」定路，**不读资格位**。于是 B' 的处方仍然说「性能不达标 ⇒ 去搜索」，且 `counts_as_iteration=True` —— 而资格层刚说这个数不能驱动任何后果。
   - Conductor 那边另有 `can_tune` 门兜着（不会真的拿它去搜索），所以这不是线上缺陷；
   - 但 **HITL 形态里，外层读处方的正是人/模型** ⇒ 工具**不许只给 `route`**，必须把 `route` 与资格掩码**并列**给，并在二者冲突时让冲突可见。这一条应当直接写进工具契约与守卫。

**另两条工程事实（都是踩出来的，不是读出来的）：**
- `write_report` **不建目录**：落点由调用方负责（Conductor 那边是 `run_layout` 的活）。HITL 工具必须自己建 run_root。
- `IntentParser` 对 `nvidia_gpu` 默认 `dsl=triton`，所以外层拿到的 task **必须显式钉住** `hardware`/`dsl`/`shapes`/`required_caps` —— 否则"在 mock 上跑"这件事不会出现在契约里。
- 顺带更正一处常见误读：`MockGPU` 的 **Python 路径 `mode=actual`**（它真测的是控制机 CPU 的 numpy），不是 `simulated`。所以"模拟读数一律 uninformative"只对需要真机/`SIMULATED_PERF` 的那条路成立；B 拿到 `valid` 是**按设计正确的**（诚实的本地 mock 不该被自动判死）。

### 9.6 工具壳已落地：`prepare` / `verify` / `capabilities`（2026-10-18）

这是 §6.2 的 Phase 0 + Phase 1 的前半：一个**进程契约**，让任何外层宿主（DSH 插件 / MCP /
脚本）不必 import 本仓内部对象。

| 文件 | 作用 |
|---|---|
| `operator_agent/hitl_tools.py` | 三个动作：`prepare`（声明 → 冻结契约 + 契约自带输入 + 回合身份）、`verify`（源码 → 正交判定 + 处方 + **冲突裁决** + 报告）、`capabilities`（只读，不跑活不联网） |
| `operator_agent/cli.py` + `operator_agent/__main__.py` | `python -m operator_agent prepare\|verify\|capabilities` |
| `tests/test_hitl_tools.py` | **24 条守卫**（3.3 s） |
| `docs/user_guide.md` §3.1 方式三 | 给人看的最短用法 |

**四条契约（每条都有守卫）：**

1. **stdout 只有一行 JSON**：判定层的进度输出被收进 `run-dir/verify-N.log`。退出码 `0` =
   判完了（候选没过也是 0），`2` = 被拒绝（带 `error`/`detail`），`1` = 意外。
2. **判据不许被调用方放宽**：`ALLOWED_KEYS` 只有 9 个（`request`/`shapes`/`hardware`/`dsl`/
   `precision`/`min_speedup`/`required_caps`/`framework`/`notes`）；`max_error`/`atol`/
   `skip_anticheat`/`require_double_confirm`/`kernel_db`/`failure_gate` 等一出现就
   `forbidden_parameter`，且**拒绝理由会写明它属于 §3.5 的哪一条**。
3. **回合身份可复算、改判据会被抓**：`prepare` 落 `declaration.json` + `contract.json`；
   `verify` 重新冻结并比哈希，不一致返回 `contract_drift`（不按新判据继续）——
   与 `workload_codegen_h20` 的「执行必须等于被授权的计划」同一条纪律。
4. **处方与资格打架必须显式**：§9.5 撞出的接缝在这里变成 `conflict` 字段，冲突时
   `action = hold_for_measurement`。**`classify` 本身一个字没改**（它还被 Conductor 用着），
   改的是这一层的呈现与建议。

#### 2026-10-18 追加：MCP server 已落地，并在真宿主里跑通

| 文件 | 作用 |
|---|---|
| `operator_agent/mcp_server.py` | 最小 MCP stdio server（**标准库实现，没有引第三方 SDK**）：`initialize` / `tools/list` / `tools/call` / `ping`，其余显式 `-32601` |
| `tests/test_mcp_server.py` | **17 条守卫**：握手与版本协商、通知不响应、坏行不打断流、schema 白名单，以及**起真进程验证 stdout 只有协议行** |
| `~/.dsh/profiles/desktop/cordis.patch.yml`（已备份） | 一行 `insert` → `@deepseek-ai/dsh-mcp-client`，`serverName: operator` |

**真宿主里的读数**（本会话里直接调的，不是模拟）：`mcp__operator__operator_capabilities`、
`mcp__operator__operator_prepare`、`mcp__operator__operator_verify` 全部就位；错的核走完
L1→L2 得到 `debug_fix` + `whole_tensor`；契约声明 `required_caps=('real_device',)` 时
得到 `validity=uninformative`、五位全关、`conflict.action=hold_for_measurement`。

**踩到的两件事（都是装进真宿主才暴露的）：**

1. **profile 的 patch 语法**：`- id: <新行>` 是「覆盖已有行」，不是插入 —— 第一次写就得到
   `patch: entry "mcp-operator" not found` 的告警（行根本没加载）。正确的是
   `- insert: [{id: …, name: …}]`。
2. **工具回包会撑爆上下文**：第一版把 `correctness.cases[*]` 的 `metrics`/`repeat_metrics`
   原样带出，单次 L2 回包 **40 KB**（里面是逐 case 的 storage/view 回读明细），宿主直接把
   结果 spill 到临时文件。裁成「只留 verdict/reason/case 身份」后 **4.8 KB**，明细仍留在
   `report.md` 与 `verify-N.json`。**这条已写成守卫**（回包 < 12 KB、逐 case 不带 metrics）。

**后端（第 3 步）也已接线：**

| backend | 状态 | 边界 |
|---|---|---|
| `mock` | ✅ 默认 | 控制机 numpy；`mode=actual`，但不是加速卡 |
| `local` | ✅ 已接线 | `LocalExecutor` 能力集是**空集** ⇒ 没有输入回读协议，带套件的 L2 判 `unknown`（**不是「候选错了」**） |
| `ssh` | ✅ 已接线，**真机未测** | 必须显式给 `host`；**不提供 `auto` 降级**（`host_required`），因为「以为在真机上、其实在控制机」是这里最贵的误读 |

**还缺的（有意不做，不是忘了）：**
- `ssh` 只做了结构接线与缺参拒绝，**没有一次真机读数**；host 选择、并发占用与清理语义
  仍在 §9.7 里。
- 没有 ESM 插件（斜杠命令 / 系统提示段 / 面板）—— 那要等判定层的使用体感被验证过。

### 9.7 真机三连（2026-10-18，H20 / `algorithm-h20`）

**外层写核的是本会话的模型（我），不是本仓的 `Designer/Coder`** —— 这正是 HITL 形态要验的那件事：
外层写码、判定层只判。判定全部经 `backend=ssh` 落到真机。

| 轮 | 候选 / 契约 | L1 | L2 | L3 | 档位 | 资格 V T P R C | 干预点 |
|---|---|---|---|---|---|---|---|
| **R1** | IEEE fp32（`input_precision="ieee"`）/ fp32 1024³ | ✅ 11 s | ✅ 2m13（15 case，`max_error=0`） | ❌ **0.36x** vs cuBLAS(TF32) | `degraded`（launch 12.5%） | ✓ ✓ · ✓ · | 工具给出「**基线是 TF32，契约是 fp32 ⇒ 不是同数值语义的比较**」 |
| **R2** | fp16 fp32-acc / fp16 1024³ | ✅ | ✅（15 case） | ❌ 但 **1.08x** | **`uninformative`**（launch **44.9%**） | · · · · · | `conflict.action=hold_for_measurement`：候选确实更快，但**这个数不许驱动任何后果** |
| **R3** | 同一个 fp16 核 / fp16 **4096³** | ✅ | ✅ | ✅ **1.06x** vs cuBLAS(FP16)，140 TFLOPS，0.98 ms | **`valid`** | **✓ ✓ ✓ ✓ ✓** | 无冲突，读数可用 |

三条结论（都是真机读数，不是推断）：

1. **HITL 链路在真机上成立**：`prepare → verify(ssh) → 处方/资格 → 报告` 全程跑通；单次全量 verify ≈
   **2m20s**（其中 L1 单独 ≈ 11 s），产物全部落在 `run_dir`。
2. **判据真的会拦下「看起来很好的数」**：R2 的 1.08x 是诱人的，但 launch 地板占了 44.9% ⇒
   档位 `uninformative`、五位全关、`conflict` 要求 `hold_for_measurement`。**同一个核**换到 4096³
   就变成 `valid` + 五位全开 —— 也就是说「shape 选得够不够大」是 HITL 里人该管的事，而工具会把
   这件事说出来。
3. **契约与基线的数值语义会打架，而且这个打架是可归因的**：`baseline_plan()` 对 fp32 matmul 的
   首选基线是 `pytorch_eager_tf32`（理由：Triton 的 `tl.dot` 默认走 TF32），而 `freeze_from_task()`
   默认 `tf32=False`。两者同时成立时，IEEE fp32 候选天然跑不过 —— 这一条现在会在回包的 `notes`
   里点名（`hitl_tools._judgement_notes`），**判定语义一个字没动**。

**关于「外层写核质量」· 诚实账**：两版核（IEEE fp32、fp16）都是**第一次就过 L1+L2**，没有一次
返工；但我只写了 2 个核、且都是结构相同的朴素分块 matmul，**样本量不足以下任何结论**。
真正需要的是「外层在收到处方后能不能改对」——那要专门跑一次「先写错、再按 `failure_route`
处方修」的对照，本次没做。

### 9.8 Phase 3 入口已落地：`operator_run` / `operator_status`（2026-10-18）

`(c) 连整段自动化也交给 DSH 发起` 的入口做完了：**一拍启动、四态轮询**，跑的是本仓自己的
`Conductor`（无人值守那条路），不是外层模型。

| 动作 | 行为 |
|---|---|
| `operator_run` | 先冻结契约（与 `prepare` 同一套），再以**独立进程** (`start_new_session`) 拉起 `python -m operator_agent.autonomous <run_dir>`，**立刻**返回 `pid`/`run_dir`/`contract_hash` |
| `operator_status` | 读 `run-status.json`：`running` / `done` / `failed` / **`lost`**（进程没了但没写终态 —— 单独一档，不许读成"还在跑"）+ `conductor.json` 摘要 + 日志尾部 |

**实测（本地假 LLM 端点，零真实凭据）**：`run` 返回 pid 后 **1.1 s** 完成，`summary`：
`all_passed=True, rounds=1, levels={L1,L2,L3}=True, speedup=1.25, validity=valid`；
报告落在 `run/output/matmul/mock_gpu/<ts>_<id>/report.md`。
日志里还能看到这条路**确实重新开了两道门**：

```
[Confirm] 独立重测: 首次 1.25x → 重测 1.1765x (valid)
[KernelDB] 未入库 (只留在本次进程): 执行侧未声明 REAL_DEVICE: 数字来自模拟或 CPU …
```

也就是说：`verify`（HITL 那条）只判定、不重测不入库；`run`（无人值守这条）**重测与入库门都回来了**。

**⚠️ 这条路必须配内层 LLM**（`Conductor` 自己设计/写核）。没配就**立刻拒绝**
（`llm_not_configured`，不起进程）。密钥只认两种来源：`LLM_API_KEY` 或
**`LLM_API_KEY_FILE`（文件引用）**。宿主侧的写法是把引用放进 MCP 行的 `env`，值不落盘：

```yaml
    env:
      LLM_BASE_URL: https://your-endpoint/v1
      LLM_MODEL: your-inner-model
      LLM_API_KEY_FILE: ~/.config/operator_agent/llm.key   # 0600, 仓库外
```

**踩到并修掉的一个真问题（值得单独记）**：第一版在子进程里写
`OPERATOR_AGENT_KERNEL_DB=off` / `FAILURE_DB=off` 想做隔离 —— **关不掉**。2026-10-08 起
普通环境变量只能**重定向**，`off` 会被解析回 `<repo>/output/*.jsonl`（日志里
`FailureDB 重放 95 条` 就是读的仓库那份）。于是无人值守运行会**读写仓库语料**。
修法是把两个变量指向 `run_dir/*.jsonl`；守卫 `test_the_repo_corpus_is_untouched`
在运行前后比对仓库语料的 `(mtime, size)`。

**2026-10-19 追加：真机闭环跑通，并在过程中抓到一个判据侧的真 bug。**

第一次真机 `operator_run`（fp16 4096³ / Triton / ssh / max_rounds=2）：**130 s，L1 就挂了** ——
模型选的 `BLOCK=256×256×64` 确实放不下（估算 384 KiB > H20 的 228 KiB），预检把它拦在
控制侧、**零 GPU 机时**。Fixer 也确实改对了（删掉 256 配置、换成 128×256，还写了一行
注释说明）。但**同一个错误把修好的代码又拒了一次**：

> `# 修复: 移除 BLOCK_M=256 的超大 tile 配置 (smem 384KiB > 228KiB 上限)`

`resource_check._extract_block_sizes` 的正则把**注释里的 `BLOCK_M=256` 读成了真实配置**
（`max()` 一取就是 256）⇒ 正确的修复被预检自己挡死。这是判据侧的**假阳性**，比漏报更难查
——它长得像「候选没改」。

修法（`resource_check._strip_comments_and_prose`）：用 `tokenize` 的 token 位置**原位抹掉**
注释与 docstring（只放行 autotune 的 `'BLOCK_M'` 这类键），再跑原有的两条正则。守卫 4 条
（注释 / docstring / autotune 最坏配置仍要被读到 / 语法坏掉时走兜底）。

**修完重跑（同一份声明、同一台卡）**：

| 轮 | 发生了什么 | 判定 | 耗时 |
|---|---|---|---|
| 1 | 27B 写 256×256×64 → 预检拒绝（**这次是合法的**） | L1 ✗ | ~2 min |
| 1' | Fixer 改成 128×256（4684 chars）→ 预检放行 | L1 ✓ → L2 ✓ → L3 ✓ | |
| 2 | 收尾 | **ALL PASSED，1.05x，139.7 TFLOPS** | **866.8 s（14.5 min）** |

- 档位 **`degraded`**（理由逐字：`测量前 GPU util=50%, 基准被污染, 加速比不可比`）；
  资格 `V ✓ T ✓ P ✗ R ✓ C ✗` —— 数可用、可调参、可报，但**不入库、不算进展**。
- 成本拆解：130 s 的第一轮几乎全在 LLM + 部署；866 s 的那次主要花在 **L2 的 15 个 case**
  （每个 case 一次一次性 SSH + 远端 import + Triton JIT）。

**这条闭环同时证明了四件事**：DSH 发起 ✓、Conductor 自主修 ✓、真卡判定 ✓、判据自身的
假阳性会被修掉并留下守卫 ✓。**还欠的**：预检的**元素宽度**仍是从源码猜的
（`_elem_bytes_from_code` 认到累加器里的 `tl.float32` 就按 4 字节算 ⇒ fp16 的 256×256 tile
被估成 384 KiB 而其实只有 192 KiB）。这条今天**没动**：改动会让 L1 对 fp16 任务放宽，
属于判定语义变更，要单独评估（见 §9.13 第 7 条）。

**成本分解（2026-10-20，同一份声明、同一台卡、同一个模型，只改 `persistent`）**

| 配置 | 总耗时 | 分解 |
|---|---|---|
| `persistent=false`（10-19） | **866.8 s** | 未逐阶段计时 |
| `persistent=true`（10-20） | **730.9 s（−16%）** | LLM+Fixer **123 s** · L1 **3 s** · **L2 490 s（67%）** · L3 108 s |

### 9.8.1 更正：L2 的钱不在“逐 case”，在**主 case 的输入传输**

上面的结论是**错的**（2026-10-20 逐项实测后更正）。三次测量：

| 测什么 | 读数 |
|---|---|
| `_verify_single_set`（**主 case** 4096³ fp16） | **572.4 s** |
| `_verify_single_set`（套件 case，32×32 级） | **3.1 s / case**（15 个 ≈ 46 s） |
| 裸 `ssh host 'cat >/dev/null'` 传 64 MB | **229 s（≈0.28 MB/s）** |

输入协议是 base64 的原始 storage 字节（`encode_json_inputs`）：fp16 4096³ 两个张量 = 67 MB 原始
→ ~90 MB base64 文本；按实测链路 ≈ 300–570 s。**这就是 L2 的全部成本**，而“把 15 个套件 case
批成一次调用”最多只省 46 s。

所以真正的杠杆是**别再传这些字节**。唯一判据中性的做法是：**远端按契约里的 seed 重新
物化输入，回传 digest，控制端核对一致后才用**（不一致就退回传字节）。契约本来就拥有
`seed`/`shape`/`dtype`，`BenchmarkCase.materialize()` 是确定性的 —— 这样 90 MB 变成几 KB，
但**必须**保留 digest 校验，否则就把「验证真实输入」换成了「验证长得像的输入」，
那正是这个仓库明确拒绝的事。

**2026-10-20 已落地（seed + digest + 字节回退）**：

| 测什么 | 之前 | 之后 |
|---|---|---|
| 主 case 4096³ 一次 `_verify_single_set` | **572.4 s** | **16.5 s** |
| 套件 case（32×32 级） | 3.1 s | 1.6 s |
| 端到端 `operator_run`（同声明/同卡/同模型） | 730.9 s | **240.6 s（−67%）** |

实现：控制端 `Verifier._seed_spec_for()` 先**自己物化一遍**、逐字节比 digest，一致才把字节
换成 `{"protocol":"numpy-seed-v1","seed":…,"tensors":[…],"digest":…}`（**231 字节**）；
远端 `run_kernel` 收到 `inputs_spec` 后**再物化一次、再比一次**，对不上回
`input_seed_mismatch`，控制端立刻退回传字节并在回包里记 `input_seed_fallback=True`。
`verify` 与 `benchmark_with_baselines` 两条路都接了（否则钱只是从 L2 挪到 L3）。
两次真机运行**都没有触发回退**（两侧 numpy 一致）。
守卫：`tests/test_input_seed_spec.py`（14 条，含「改一个数/换 dtype/换 seed 都要变」「回退必须留痕」）。

同一份代码两次读数的对照也顺手拿到了：`util=50%` 那次 **1.05x**，`util=100%` 这次 **0.97x**，
两次档位都是 `degraded`、都写明「基准被污染」。这就是为什么系统**不许**用这两次读数去调参
或禁用配方 —— 数在，但不可比。

**边界**：没有取消动作（只能按 pid 杀）；MCP 侧的 `operator_run` 加工具后需要重连
（本会话已重连，5 个工具都在）。

### 9.9 处方前后对照 + MCP 真机（2026-10-19）

**(A) 外层模型能不能按处方改对** —— 这是“效果足够再自动化”的唯一判据。做法：故意种一个
可被处方定位的真 bug，看处方说什么、改了几个字符、再判是否通过。

| 轮 | 候选 | 判定 | 处方 | 人（外层）做了什么 |
|---|---|---|---|---|
| **R1** | `tl.load(..., other=1.0)`（两处；真实 typo） | 主 case **通过**（4096 是块大小倍数 ⇒ mask 不触发），**套件 15/15 全挂**：每条 `whole_tensor`，连 `zeros` 也挂，误差 ≈28–31 | `route=debug_fix, fix_not_rewrite=True`；**聚合 `error_shape=None`** →“误差形态未确立，按首个坏点定位，别猜” | 读逐 case 证据（`zeros` 挂 + `whole_tensor` + 误差量级）⇒ 判定“被 mask 的 lane 贡献了非零常数”，改 **2 个字符** |
| **R2** | `other=0.0` | **L1/L2/L3 全过**：套件 15 case passed，1.05x vs cuBLAS(FP16)，`valid`，五位全开 | — | — |

读数与成本：R1 **4m57**、R2 **6m14**（套件失败那轮更贵）。**结论：一次改对**——但这是 n=1，
只能说明“这条路能走通”，不能当成功率。

**过程中最有价值的一条**：主 case 通过、套件失败时，`failure_route` 的**聚合形态是 `None`**
（三态里的“未确立”），处方因此退回“按首个坏点定位”。真正定住方向的是**逐 case 证据**
（`zeros` 这种输入也挂 ⇒ 只可能是常量污染）。这既说明处方没吹牛（不知道就说不知道），
也说明**逐 case 证据必须留在工具回包里**——否则外层只剩一句“未确立”。

**(B) MCP → 真机**：
- 重启 MCP server 后（日志：`connection lost; reconnecting in 500ms` → `reconnected and re-synced tools`），
  `operator_verify(backend=ssh)` 成功：`resolved=SSHExecutor@algorithm-h20(nvidia)`、L1 通过、
  **回包是干净 JSON** —— 这同时是 §9.6 那个“后端 print 污染 stdout”修复在**真宿主**里的验收。
- `operator_run` 在宿主里 fail-closed：返回 `llm_not_configured`（没起进程），符合设计。
- 运维事实：**加/改工具后要等 MCP 重连**（`reconnect.enabled` 默认开，500 ms 起指数退避），
  而**本会话的工具清单要到下一回合才刷新**；`pkill -f operator_agent.mcp_server` 是安全的
  重载方式（宿主会自动重连并 re-sync）。

### 9.11 自动修复能力：n=6（2026-10-20）

方法：写 6 个**各对应一类真实缺陷**的 Triton 核（都 L1 可编译），先用判定层确认落点
（不花 LLM 的钱），再用新加的 `operator_repair`（判 → 处方 → 内层 Fixer → 再判）跑一遍，
`stop_after=L2`，最大 2 轮。

| 缺陷 | 首次失败 | 处方形态 | 一轮后 |
|---|---|---|---|
| 尾块 `other=1.0`（两处） | L2（主 case 过、套件挂） | `debug_fix`，形状 `None` | ✅ 修好 |
| A 的两个 stride 传反 | L2（主 case 挂） | `debug_fix`，`whole_tensor` | ❌ **没修好** |
| B 的 K 掩码用了 N 边界 | L2（套件） | `debug_fix`，`None` | ✅ |
| K 循环只累加到 `K//2` | L2（主 case 挂） | `debug_fix`，`whole_tensor` | ✅ |
| 输出按 `(N,M)` 分配 | L2（套件；方阵主 case 掩盖了它） | `debug_fix`，`None` | ✅ |
| `torch.matmul` 回退 | L2（反作弊） | `debug_fix`，`None` | ✅ |

**读数：5/6 一轮改对（83%），每例 25–34 秒**（两次判定 + 一次 LLM 调用）。这是第一条
“闭环能把错核改对”的数字，可以拿去支撑“效果是否足够”的判断了（n=6，仍不是统计）。

**唯一那次失败很值得记**：模型**照着处方走**了 —— `whole_tensor` 的处方写着「先核对数值
语义与输出缩放（dtype / 累加精度 / 是否漏了一次 scale）」，于是它把 `tl.dot` 改成了
`input_precision="ieee"`，而真正的 bug 是两个 stride 传反（它自己的修复注释就是证据）。
也就是说：**这一次不是模型不会修，是处方指错了方向**。`whole_tensor` 这个形态无法区分
「值全错」的多种成因（dtype / 缩放 / 索引），而处方只给了 dtype 那一支。

⇒ 下一步若继续提升，杠杆是**处方的可分辨性**（例如把「主 case 就错」与「只有套件错」、
「形状相关」与「值与形状都相关」分开），而不是换更大的模型。

### 9.12 发布形态与宿主插件（2026-10-20）

插件从“一个 MCP 旁路”变成**可安装的两层**：

| 层 | 产物 | 验证 |
|---|---|---|
| Python | `pyproject.toml`（`operator-agent` / `operator-agent-mcp` 两个命令）+ `remote_scripts/__init__.py` + `protocol_identity()` 改包相对解析 + `MANIFEST.in` | 装到临时目录后**从 /tmp** 跑 `--selftest`、11 个 prompt 模板、47 条 identity 全对 |
| DSH | `integrations/dsh-operator/`：npm 壳（`dsh.bundle.patch` + `engines.dsh`）+ ESM 宿主插件（`/op` 命令 + `tool:operator` 提示段） | Node 测试 8 条；装进本机 profile 后**提示段确实出现在系统提示里**、MCP 六工具同时可用、宿主日志无报错 |

**本机当前形态**：源码树（不 pip install）——MCP 行用 `<abs-python> -m operator_agent.mcp_server`，
插件行用 `index.mjs` 的绝对路径 + `PYTHONPATH` 指到仓库。

**顺手记两个坑**（都是这次踩的）：
1. `pip < 21.3` 会把本地包复制到临时目录再构建，而 `output/` 里的 `0400` 产物删不掉 ⇒
   构建以 `PermissionError` 收尾（wheel 已生成）。用 `--use-feature=in-tree-build` 或从干净树构建。
2. **斜杠命令与工具/提示段的重载时机不同**：后者热加载，前者常要重启一次宿主。

**仍未定**：License（`UNLICENSED`）。它只挡 `npm publish`，不挡本地安装。

### 9.13 还没做的

1. 多卡/并发与清理语义：目前全在单卡、串行、`util=0` 的窗口里，**没有测竞争与共存**。
2. **取消动作**（现在只能按 pid 杀）；`operator_run` 的真机一次已在 §9.8 补上（866.8 s，1.05x）。
3. 内层 LLM 的凭据接线（§9.8 的 `LLM_API_KEY_FILE` 写法还没落到本机 profile —— 需要用户
   提供端点与密钥文件，我没有、也不该替你造一个）。
4. 处方质量的**统计**（A 只是 n=1）：要回答“外层改一次的成功率”，得跑 5~10 个不同 bug
   （尾块 mask、stride、广播轴、K 循环、dtype）并记“几次改对”。
5. 套件失败时**聚合形态**的改进：现在退回“未确立”，可考虑按 case 证据聚类给出方向 ——
   但那是**新判据**，要单独评估（不能为了让处方好看而发明一个形态）。
6. ~~输入传输~~ **已落地（2026-10-20）**：seed + digest + 字节回退，端到端 730.9 → **240.6 s**。
   ⚠️ 边界：远端必须能复算出同一份字节（依赖两侧 numpy 版本一致）；不一致时会退回传字节，
   表现为**慢但正确**，并可凭 `input_seed_fallback` 归因。
7. 预检的**元素宽度**：`_elem_bytes_from_code` 会读到累加器的 `tl.float32` ⇒ fp16 任务的
   256×256 tile 被按 4 字节估（384 KiB），而真实 tile 是 2 字节（192 KiB）。应改成用**任务
   声明的 precision**（`verifier` 和 `conductor` 都拿得到），源码推断只做兜底。这会放宽 L1，
   是判定语义变更，要单独评估 + 守卫。

**一条必须记住的边界**：这一层**不做两次确认、不做入库**。那是 `Conductor` 的门
（`_produce_remeasure` → `oracle.admit_to_corpus`），HITL 路径只产出判定与报告 ——
R3 的 `report.md` 自己就写着「两次独立确认: **没有**做重测」，这是**如实的**，不是缺陷。
等到 Phase 3 把整段交回 `Conductor` 时，那两道门才重新生效。

---

## 十、未核实项与边界（诚实账）

- 我没有**写或跑过任何 DSH 插件**；MCP 行能否在本 profile 一次插通，只有文档与代码证据，**无实测**。本机也没有装 Python `mcp` 包（`import mcp` 报 ModuleNotFoundError）—— 本文的 A1 预设是「stdlib 自实现帧协议」。
- 「宿主插件不受工作区沙箱限制」来自 `dsh-plugin-manager` README 原文【V 读到文本】，**未做越权实测**。
- DSH 的上下文压缩阈值 / 策略未核实：装机包里没有 `docs/`（`package.json#files` 声明了 `docs/**` 但未随包），README 引用的 `../../../docs/...` 链接在本机是**断链**。
- `dsh-client-hmr`、`dsh.client` 客户端插件的构建链路只读到文档与产物形状【V】，未验证 `pnpm run dev:web` 的实际行为。
- 本仓的数字取自已说明的当前树（HEAD `6ecf13e` + 脏工作区：`run_contract.py`、`single_candidate_coder.py` 等有未提交改动）。**任何打包估算应钉 commit，这份评估的引用行号也会漂**（doc 16 §九 记过同一条教训）。
- 3,161 个测试「一分钟内跑完」是本仓文档的说法（`docs/user_guide.md:105`），本次**未跑**。
- §八 的开源形态：**2026-10-18 本轮已重新联网复核**，一手结果在 §8.5（本机 `http_proxy=127.0.0.1:7887`；`api.github.com` 约 40 次后限流、`raw.githubusercontent.com` 全程可用）。**仍未复核的**：`NVIDIA/skills` 的 17 个 kernel 技能我只逐字读了 1 个 `SKILL.md` 与它的 `BENCHMARK.md`；`KernelFlow-ops/cuda-optimized-skill`、`Shekswess/gpu-kernel-skills`、`flashinfer-bench` 内嵌的 10 个 skill、`tensormux` 的 35 技能计数、`cfregly/gpu-perf-tune` 的 32 技能计数，**都只有元数据或二手转述，未逐文件核**。
- `dsh-kernel-opt`：§8.5(d) 全部取自它的 README / `package.json` / `preset.yml` / `docs/eval-contract.md` **原文**，但**我没有安装、没有运行它**，也没有核过它的 `lib/` 与 `src/loop.ts` 实现；它 README 里提到的 "AKO Runtime MCP" 接口**只有那一句话，未找到任何 MCP 配置或 schema**。
- 本仓数字取自已说明的当前树（HEAD `6ecf13e` + 脏工作区：`run_contract.py`、`single_candidate_coder.py` 等有未提交改动）。**任何打包估算应钉 commit，这份评估的引用行号也会漂**（doc 16 §九 记过同一条教训）。
- **§9.5 的链路、§9.6 的工具壳与 MCP、§9.7 的真机三连都已在 2026-10-18 实测跑通**。仍缺：§9.8 的三件（处方前后对照、并发/清理、MCP 侧真机调用）；「外层模型写核质量」只有 2 个核的样本，**不足以下结论**。
- §9.5 里那个「接缝」是按**当前树**实测的（`failure_route.classify` 不读资格位）。如果有人在那之前改了 classify、或外层改成先读资格再读处方，这条要重测 —— 演示里那行 `⚠️ 接缝:` 就是它的读数入口。
- §9.4 最后一行那个未验证假设（外层模型自己写 Triton 的质量）**没有数据**：DSH 侧当前默认模型是 `deepseek-v4.1-flash`（profile 的 `agent-default-model`），我没有任何它在 Triton 生成上的实测读数。
- **本次新增** 9 个文件（`benchmarks/demo_hitl_chain.py`、`tests/test_demo_hitl_chain.py`、`operator_agent/hitl_tools.py`、`operator_agent/cli.py`、`operator_agent/__main__.py`、`operator_agent/mcp_server.py`、`tests/test_mcp_server.py`、`operator_agent/autonomous.py`、`tests/test_autonomous.py`）**+ 1 个测试**（`tests/test_hitl_tools.py`），并把用法写进 `docs/user_guide.md`；**未改任何既有模块的代码**。守卫跑了相关的 **89 条全绿**（工具壳 30 + MCP 17 + 无人值守 10 + 演示 11 + 卫生/日志 21），**不是全量 3,161 条**；另有 DSH 宿主内的真实 MCP 调用验收（§9.6）、H20 真机三轮（§9.7）与无人值守整段运行的离线验收（§9.8）。没改 `docs/research/README.md` 的索引；若要纳入正式研究编号，另开一篇并补 CHECKLOG 更符合本仓治理习惯。
