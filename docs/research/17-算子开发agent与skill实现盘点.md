# 调研 17：算子开发 agent 的**可实现**版图 —— agent 与 skill 两种形态

> 版本: v1.0 | 日期: 2026-09-29
> 起因：用户问「当前业内有算子开发 agent 的一些实现吗，agent 或 skill」。
> **与 [`11-`](11-kernel生成系统全景.md) 的分工**：11 号盘的是**论文侧**（架构、宣称数字、可信度分级），
> 本篇盘的是**仓库侧** —— 能不能 clone、license 是什么、跑起来要什么、
> 以及「agent」这个东西在 2026 年到底以什么形态分发。
>
> **核实声明**：✅ = 本人本轮用 `api.github.com` 元数据 / git tree / raw 源码直接取到并核对；
> ⚠️ = 调研 agent 转述、本人未复验。**本轮又抓到一处转述失真**，见 §六。

---

## 一、结论先行：这条路线已经收敛成「发一个 skill」，不是「写一个 agent」

最该看的一个是 `TongmingLAIC/AKO4ALL`。我把它的 git tree 整个拉下来数过（✅）：

```
blob 总数 = 18
   16156  SKILL.md                              ← 「agent」的全部逻辑在这里
   51677  bench/kernelbench/bench.py            ← 唯一的可执行契约
    9718  bench/kernelbench/GUIDE.md
    17975  README.md
    1600  bench-wrapper.sh
     750  HINTS.md      714  ITERATIONS.md      ← 提示与迭代日志的固定格式
       0  knowledge/.gitkeep  source/.gitkeep  ← 运行时才填充的槽位
    + 6 个 png、LICENSE、.gitignore、workspace.gitignore
```

**没有一行 agent 循环代码。** 循环是 Claude Code 读完 16KB `SKILL.md` 之后的行为。
这印证了 [`11-`](11-kernel生成系统全景.md) §四 末尾那条当时只敢标 ⚠️ 的判断——
AKO 的「薄 harness + 强模型」不是口号，它的**实现形态就是一颗 skill 文件**。

`SKILL.md` 的 frontmatter 原文（✅ 本人读到）值得直接抄结构：

> ```
> name: ako4all
> description: Drive an agentic loop that iteratively optimizes a GPU kernel for maximum
>   speedup. Use this skill whenever the user wants to optimize / speed up / benchmark a GPU
>   kernel (CUDA, Triton, TileLang, C++, Python), mentions AKO / AKO4ALL / AKO4X / agentic
>   kernel optimization, asks to "make this kernel faster", or has a kernel they want measured
>   against a PyTorch reference. The skill handles setup, profiling (ncu), correctness
>   checking, iteration logging, and git commits. Bootstraps a workspace in any directory
>   the user points at.
> ```

正文第一件事不是讲算法，而是 **"When this skill applies" 加一段 "Does NOT apply when"**
（想从零写新核 → 不适用；要别人 review → 转 `codex:rescue`；非 kernel 的泛性能建议 → 不适用），
然后才是循环本体一句话：`profile → modify → benchmark → log → commit`，
以及 **First action 先敲定 workspace**（"the directory the loop runs in"，靠「四处看看」而非固定清单来盘点输入）。

> **对我方最直接的一条**：`description` 那段把**触发条件写全**（含用户可能的原话措辞），
> 这正是「progressive disclosure」能工作的前提——启动只注入 name+description，
> 模型据此判断要不要读全文。我方 `operator_agent/skills/*.md` 没有这一层，见 §七。

---

## 二、能 clone 就跑的实现（★/license/push 均 ✅ 本人取 api 元数据）

| 仓库 | license | ★ / 最近 push | 有没有 agent 循环 | 放产物了吗 | 跑起来要什么 |
|---|---|---|---|---|---|
| `TongmingLAIC/AKO4ALL` | **MIT** | 370 / 2026-09-15 | ❌ 无代码，是 skill（§一） | ⚠️ 只给脚手架 | NVIDIA GPU，本地即可，不需云 |
| `TongmingLAIC/AKO4X` | **MIT** | 75 / 2026-09-15 | ✅ 有（见 §三） | ✅ **放了**（§三实证） | Modal 云（要账号）**或**本地 CUDA ⚠️ |
| `TongmingLAIC/KernelBand` | Apache-2.0 | 7 / 2026-07-02 | ⚠️ `agents/optim_agent.py` 单文件主循环 | ⚠️ 放了训练数据 json | NVIDIA GPU |
| `NVIDIA/SOL-ExecBench` | Apache-2.0 | 301 / 2026-07-15 | ❌ 只有评测器 | 题目集 | NVIDIA GPU |
| `AMD-AGI/GEAK` | Apache-2.0 | 181 / **2026-09-29 仍活跃** | ⚠️ | ⚠️ kernel 集未开 tree 核 | **ROCm/MI300X**（非我方硬件） |
| `codelion/openevolve` | ⚠️ 未取到 | — | ✅ 通用进化框架 | 每任务 `config.yaml+evaluator.py+initial_program.py` 契约 | 只要 LLM API key |
| `tensormux/kernel-skills` | **MIT** | 79 / 2026-06-21 | — 是 skill 库 | ⚠️ | — |
| `NVIDIA/skills` | Apache-2.0 | **3478** / 2026-09-25 | — skill 库 | ✅（§四 tree 实证） | — |
| `flashinfer-ai/flashinfer-bench` | ⚠️ Apache-2.0 ★286 | 2026-09-25 ⚠️ | ❌ 基准 | 生产 trace | NVIDIA GPU |
| Kevin-32B | — | — | — | ⚠️ HF 上权重确认存在（`cognition-ai/Kevin-32B`） | — |

**两个「不存在/未开源」的确认**：
- ⚠️ `ScalingKernelLLM/kernel-agent` —— 我上一轮凭印象提的这个名字，本人 API 取回 **404，确实不存在**，别再搜。
- ⚠️ **Meta KernelEvolve 没有开源**（agent 搜遍 GitHub 只命中一个第三方结果仓）。
  这条重要：`11-` §一 表格里给它的 100% pass 声明**无代码可核**的判断成立，且现在多一条佐证。

---

## 三、AKO4X：目前唯一「循环 + 产物 + 反作弊 + 离散度」四全的（✅ 本人取 contents API）

```
master/            MASTER.md, master.py, harness-ledger.md
templates/skills/  bench, benchmark, cpp, cuda, cute-dsl, profiler-ncu, sanitizer, tilelang, triton
scripts/           cheat_check_modal.py, backfill_parent_txt.py, benchmark_adapter.py,
                   campaign_start.py, run_local.py, run_local_profile.py,
                   run_local_sanitize.py, diff_trajectory.py, pack_solution.py, CLAUDE.md
reference/         dsa-sparse-attention, dsa-topk-indexer, gdn-decode, gdn-prefill,
                   gemm-n2048-k4096, gqa-paged-decode-h32-kv8-d128-ps1,
                   mla-paged-decode-…, mla-paged-prefill-…, moe-fp8-block-sc…
```

为什么这一格对我方价值最高（三条都是**同题已有代码可参照**，不是概念）：

1. **`scripts/cheat_check_modal.py`** —— [`03-`](03-反作弊与正确性防线实测.md) 我方 `anticheat.py` 是静态串匹配，
   抓不到「CUDA-graph capture 时跑一次、replay 静默跳过」这类动态作弊；
   `11-` §3.3 记下 AKO 的对策是「固定张量指针只改数值，输出不变即拒」，**现在那 8KB 实现就在仓里**。
2. **`reference/*/variance.json` + `parent.txt`**（文件名 ⚠️ 由 agent 报，目录 ✅ 本人核）——
   离散度门（[`33-`](33-离散度门真机验证.md)）与候选血缘（[`47-`](47-候选结构去重键与跳过验证的代价.md)）在业界有落盘格式先例。
3. **`master/harness-ledger.md`** —— harness 自己记账（[`48-`](48-端到端成功率量具.md) 的同类），
   且 `templates/skills/` 按 DSL 分包（triton/cuda/cute-dsl/tilelang）+ `profiler-ncu`/`sanitizer` 两个
   **工具型技能**——这正好是 §一 那颗 skill 的「多技能版」。

⚠️ 未核：各文件大小与内部实现细节；`MASTER.md` 与 `SKILL.md` 的分工。要借鉴必须先读原文。

---

## 四、skill 形态这一格：NVIDIA 官方已占位（✅ 本人取 tree，5331 blob）

`NVIDIA/skills`（Apache-2.0，★3478，push 2026-09-25），自述 *"Agent Skills for NVIDIA products —
install into Claude Code…"*。kernel 向的技能目录（✅ 本人从 tree 里筛出）：

```
skills/tilegym-adding-cutile-kernel/          SKILL.md 8230 · BENCHMARK.md · evals/evals.json
                                              · skill-card.md · skill.oms.sig
skills/tilegym-converting-cutile-to-triton/   ← 与我方 Triton 路线直接对口
skills/tilegym-converting-cutile-to-julia/    SKILL.md 6932 · references/{api-mapping,
                                              critical-rules,debugging,testing}.md
                                              · examples/0N_{add,matmul,softmax}/ 成对可执行代码
                                              · scripts/validate_cutile_jl.py
skills/triton/                                ← agent 漏报的一个
```

**这套目录形制就是「有加载机制」的样子**：每技能一目录 + `SKILL.md`(frontmatter) +
`references/` + `evals/evals.json` + `BENCHMARK.md` + 签名文件 `skill.oms.sig` +
`scripts/validate_*.py`（**技能自带校验器**）。

其余：⚠️ `tensormux/kernel-skills`（license ✅ MIT，★79 ✅；「35 个技能目录 + schema + 索引脚本 + proof」**未复验**）；
⚠️ `cfregly/gpu-perf-tune`（Agent Skills + MCP 双形态，**★1** —— 只能当存在性证据，不是参照物）；
⚠️ **Anthropic 官方 `anthropics/skills` 的 21 个技能里 GPU/kernel 为零**（若属实，说明官方 examples 这格空着，
NVIDIA 是先占的那个）；**Nsight Compute 的 MCP server：未找到**（只有 Nsight *Graphics*）。

---

## 五、判据：怎么区分「技能包」和「prompt 里的形容词」

本轮反复出现同一把尺，值得立成规则：

> **有没有配套校验基建（frontmatter 触发语 / schema / evals / proof / validate 脚本），
> 就是「技能包」与「prompt 形容词」的分界线。**

- ✅ 有基建：`NVIDIA/skills`（evals + sig + validate 脚本）、`AKO4ALL`（frontmatter 写全触发措辞 +
  `HINTS.md`/`ITERATIONS.md` 固定格式 + `bench.py` 契约）。
- ⚠️ 结构上不具备 progressive disclosure：`anthony-maio/triton-skills` 是 `skills/SKILL.md` +
  9 个**平铺** `triton-*.md`，无 per-skill 目录 —— 更像一次性灌满 context。

这正是本项目踩过三次的同一个坑的 skill 版本（Ecdysis 的「复发优先」只是形容词、
AdaExplore 出厂 `memory_update: false`、OpenEvolve 的 embedding 去重出厂关闭）。
已进 [`CHECKLOG.md`](CHECKLOG.md) 台账。

---

## 六、本轮翻出来的更正

1. **`docs/research/11` 指错了仓库（两处，已在本文档同日修正）**：它把 `NVlabs/SOLAR` 当成
   SOL-ExecBench 的官方实现（L19 可信度列、L93 出处列各一次）。
   ✅ 本人核到 SOLAR 的真实描述是 **"Speed of Light Analysis for ML Model Runtime"**，
   是 einsum/timeloop 方向的**解析建模**，与那个 benchmark 实现无关；
   SOL-ExecBench 的正主是 `NVIDIA/SOL-ExecBench`（Apache-2.0，★301）。
   已挂号 CHECKLOG。
2. **`11-` 那张全景表整列原本 ⚠️，现在可升 ✅ 的三条**：AKO 系代码确实存在且是 **MIT**、
   GEAK **仍在活跃**（push 2026-09-29）、AlphaEvolve **确实未自开源**（只有第三方复刻）。
3. **转述失真一例（本轮新抓）**：agent 报 `AKO4ALL` 里 `bench/kernelbench/bench.py` 为 51.6KB ✅ 命中，
   但同一条报告里 NVIDIA 的技能只提了 `tilegym-converting-cutile-to-triton`，
   实际 tree 里 `skills/triton/` 与 `-to-julia` 也在 —— **漏报不等于没有**，
   所以本篇凡是「清单类」结论都以 tree 为准，不以 agent 的点列为准。

---

## 七、对我方的含义（不吸纳什么，也要写）

**可吸纳，按性价比：**

| 动作 | 参照 | 代价 |
|---|---|---|
| 给 `operator_agent/skills/` 两个 md 补 frontmatter 触发语（把用户可能的措辞列进 `description`）+ 「When this skill does NOT apply」段 | `AKO4ALL/SKILL.md`（✅ 原文见 §一） | **低**，纯文本 |
| 抄 `bench.py` 的**契约文件**形状：一个文件同时管「输入生成 / 正确性 / 计时 / 结果落盘」，让 harness 与 agent 的边界只有一处 | `AKO4ALL/bench/kernelbench/bench.py` 51.6KB（✅ 存在，⚠️ 内容未读） | 中 |
| 反作弊从静态串匹配升级成**动态 staleness 审计**（固定指针改数值，输出不变即拒） | `AKO4X/scripts/cheat_check_modal.py`（✅ 存在） | 低（`11-` §四 已估 ~50 行） |
| 迭代日志与提示分文件、格式固定（`HINTS.md` / `ITERATIONS.md` / `harness-ledger.md`） | AKO4ALL + AKO4X | 低 |

**不吸纳**：AKO4X 的 Modal 云编排（我方是自有 H20 机，云依赖是负债不是收益）；
`tilegym-*` 的 CuTe-DSL 方向（我方 DSL 是 Triton，但 `-to-triton` 那颗技能值得**读它的
`references/critical-rules.md` 类硬约束写法**，不抄内容）。

---

## 八、取证方法与未核项（诚实账）

- **网络前提同 [`16-`](16-harness主循环骨架与上下文生命周期.md) §九**：本机到 GitHub 直连 `http=000`、
  `WebFetch` 静默超时、`gh` CLI 未装。本轮全部走本地代理。
- **zsh 词分割坑**：把 `-x http://…` 存进变量再 `$VAR` 展开，在 **zsh 下不分割**，
  整串被当成一个参数 → 全部 `http=000`，看起来像「代理也不通」。改成显式写在命令行后才 200。
  **这是一次「量具自己骗我」**：差点据此得出「外部仓都取不到」的结论。
- ✅ **本人核到的**：全部列出的 api.github.com 元数据（描述/license/★/push 日期）；
  `AKO4ALL` 完整 tree + `SKILL.md` 前 32 行原文；`AKO4X` 四个关键目录的 contents 列表；
  `NVIDIA/skills` tree 里 tilegym/triton 技能目录；`ScalingKernelLLM/kernel-agent` = 404。
- ⚠️ **未核（照抄会出错）**：AKO4X 各文件大小与实现、`MASTER.md` 内容；KernelBand 内部；
  `tensormux` 的「35 技能」计数；`anthropics/skills` 零 GPU 技能这条；
  flashinfer-bench 与 openevolve 的 star 数；Kevin-32B 训练代码是否开源；
  `11-` 论文侧数字本篇**一律未重核**（那 110 行的证据等级不因本篇改变）。

---

## 九、来源

一手（本轮取到，需带本地代理）：
- `https://api.github.com/repos/{TongmingLAIC/AKO4ALL, TongmingLAIC/AKO4X, TongmingLAIC/KernelBand,`
  `NVIDIA/SOL-ExecBench, NVlabs/SOLAR, AMD-AGI/GEAK, NVIDIA/skills, tensormux/kernel-skills,`
  `cfregly/gpu-perf-tune, ScalingKernelLLM/kernel-agent}`
- `https://raw.githubusercontent.com/TongmingLAIC/AKO4ALL/main/SKILL.md`（前 32 行）
- git trees：`AKO4ALL`（18 blob 全列）、`NVIDIA/skills`（5331 blob 内筛 tilegym/triton）

相关内部文档：[`11-`](11-kernel生成系统全景.md) 论文侧全景 · [`16-`](16-harness主循环骨架与上下文生命周期.md) 主循环骨架 ·
[`13-`](13-跨运行记忆与技能库架构.md) 技能库选型 · [`03-`](03-反作弊与正确性防线实测.md) 反作弊 ·
[`33-`](33-离散度门真机验证.md) 离散度 · [`48-`](48-端到端成功率量具.md) 量具
