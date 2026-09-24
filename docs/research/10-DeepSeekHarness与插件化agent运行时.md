# 调研 10：DeepSeek Harness (DSH) 与插件化 agent 运行时

> 版本: v1.0 | 日期: 2026-09-24
> 服务于缺口: 1（测量有效性无第三态）、5（isinstance 分派 + 静默降级）、6（接口契约隐式）、2（跨运行记忆）
> 核实声明: 标注 ✅ 的论断由本人直接拉取一手文件逐字核对；标注 ⚠️ 的来自研究员报告，**未独立核实**，落地前需自行验证。

## 一、"dsh" 是什么（已核实）

**dsh = DeepSeek Harness**，DeepSeek 官方开源的 agent 运行时。

| 项 | 值 | 来源 |
|---|---|---|
| 仓库 | `deepseek-ai/deepseek-harness` | ✅ GitHub API |
| 描述 | "DeepSeek Harness: Everything is a Plugin." | ✅ GitHub API |
| License / 语言 | MIT / TypeScript | ✅ GitHub API |
| topics | `ai-agents`, `cordis`, **`dsh`**, `dsh-plugin` | ✅ GitHub API（`dsh` 确为其官方缩写） |
| 创建 / 最近推送 | 2026-08-13 / 2026-09-24（当日仍在推） | ✅ GitHub API |
| stars | 234,982 | ✅ GitHub API |
| 底层框架 | Cordis（`cordiverse/cordis`） | ⚠️ 未独立核实 |

**成熟度警告（✅ 研究员引自官方 `SAFETY.md` 与 README，措辞强烈）**：项目处于 developer preview，官方明说「未经安全审计，不得视为安全或生产可用」「THERE WILL BE COMPATIBILITY-BREAKING CHANGES」，最新版本全是 prerelease。

**因此本项目的定位是：借鉴其设计原则，不引入其代码。** 它是 TypeScript + Cordis 插件体系，与本项目（Python、零重依赖）技术栈完全不同，直接引入不可行也不必要。价值在于它把一批「bug 类型 → 防御规则」写成了可引用的成文原则。

## 二、最高价值的一条：正交结果独立上报（缺口 1）

✅ 逐字核对自 `docs/defensive-patterns.md` 第 7-9 行：

> **Report orthogonal outcomes independently**
>
> A result can be several things at once — a process can time out AND exit 0 because it trapped the signal. Surface each independent fact (`timedOut`, `signal`, `exitCode`) on its own; **never nest one flag's report inside another's branch, or a caller reads a cut-short run as a clean success.**

文档开头对这一类规则的定位（✅ 同文件第 5 行）：

> Hard-won bug-class rules: each pattern below is a class of defect that **actually shipped or nearly shipped here**, stated as the rule that prevents its recurrence.

### 与本项目缺口 1 的对应关系

这不是泛泛的类比，是同一个缺陷类：

| DSH 的反模式 | 本项目的实例 |
|---|---|
| 一个进程「既 timeout 又 exit 0」被压成单一结果 | 一次 L3 基准「既测出了加速比、又被同机其他进程污染」被压成 `passed: bool` |
| `timedOut` 埋在 `exitCode` 的分支里 | `benchmark_contaminated` 埋在 `metrics` dict 和 message 字符串里，`passed` 照旧 |
| 调用方把被截断的运行读成干净成功 | Conductor 把被污染的测量读成「性能未改善」，进而**禁用实际有效的优化配方** |

本项目的 `VerificationResult`（`operator_agent/verifier.py`）当前形态：

```
level / passed: bool / message: str / metrics: dict / error_log
```

`passed` 是唯一的控制流依据，`metrics` 是无类型的自由 dict。已核实 `grep contaminat conductor.py fixer.py agents.py` **零命中**——污染标志不参与任何决策。

### 建议的落地形态

把 `passed` 拆成正交事实集，每个独立可查询，控制流按需组合而非读单一布尔：

```
compiled: bool            # L1
numerics_ok: bool         # L2 数值
anticheat_ok: bool        # L2 反作弊
signature_ok: bool        # L1 接口契约 (缺口 6)
perf_measured: bool       # L3 是否产出性能数据
measurement_validity:     # 见 docs/research/01-launch开销与shape充分性实测.md 第五节
  valid | degraded | uninformative
baseline_valid: bool      # baseline 本身是否可信
roofline_regime:          # compute | memory | unknown (缺口 4)
  compute | memory | unknown
launch_dominance: float   # floor / candidate_time
```

关键约束（直接照搬 DSH 的措辞）：**任何一个标志的上报都不得嵌在另一个标志的分支里**。当前代码恰好违反——污染信息只在 `passed` 已定之后追加到 message 字符串。

代价：中。需要同步改 Conductor 的判据读取处（`conductor.py:425-463`）与 Fixer/Search 的输入。但这是缺口 1 的正确形态，比「加一个 `contaminated` 布尔」更彻底。

## 三、次高价值：基线是产物，不是审查（缺口 1 的延伸）

✅ 逐字核对自 `docs/postmortem/0002-js-expression-disabled-filesystem-tools.md` 第 46 行：

> **A snapshot refresh is fixture production, not correctness review.** Semantic impossibilities such as a missing registered tool need assertions **independent of the expected output**.

该事故的机制（✅ 同文件第 19、34 行）：文件系统插件被一个写错位置的条件表达式永久禁用，工具全部 `UNKNOWN_TOOL`；但快照框架「把任何可复现的 transcript 都当作有效行为」，刷新快照时**先重写了期望输出，才轮到语义断言去拒绝缺失的工具**，于是回归被固化成「期望行为」，测试全绿。

### 对本项目的直接警示

本项目有 `tests/golden_registry.json` + `tests/golden_regression.py`（当前仅 4 条 MockGPU 基线，`git_hash` 为空）。同一失效模式完全适用：

- 若允许「用当前运行结果刷新 golden 基线」，那么一次被 launch 开销压扁的 layernorm 运行（见 `01-launch开销与shape充分性实测.md`：报告 0.65x，净值差 6.8x）会被写进基线，此后**这个 6.8 倍的退化就成了"期望值"**。
- L3 的 baseline 缓存（`verifier.py` 的 `_baseline_cache`，按「算子 + 输入签名」缓存）是同一风险的另一处：缓存的 baseline 若来自被污染的一次测量，后续所有对比都失真。

### 建议的落地形态

1. **禁止自动刷新 golden 基线**，刷新必须显式触发且需要人看 diff。
2. 加**不经由基线的语义断言**——DSH 的做法是让快照框架直接拒绝 `UNKNOWN_TOOL` 这类「语义上不可能正确」的结果。本项目的对应物：L2 反作弊信号与接口契约检查必须**先于**任何性能基线的刷新/缓存写入通过。
3. `golden_registry.json` 的 `git_hash` 必须真实填写，否则无法判断基线对应哪个版本的代码。

代价：小。

## 四、Capability seam 三角色（缺口 5、6、3）

✅ 核对自 `docs/capability-seams.md` 第 5 行：

> A service can be a core spine service, **a swappable capability seam**, a bundle/composition point, or a standalone service. The graph shows **the package that owns the service declaration, known implementation packages, and packages that consume the service directly**.

即每个可替换能力有明确三角色：**Definition（拥有 `ctx.<key>`）/ Provider（实现）/ Consumer（消费）**，且该表由 `scripts/gen-doc-graphs.ts` 机器生成并带完整性守卫（✅ 同文件第 661 行）。

✅ 两个可直接借用的 provider 契约措辞（同文件表格内）：

- `ctx.sandbox`（第 638 行）：「Consumers hand over **the exact argv they are about to spawn**; same-world backends wrap it under a per-call policy and **report enforcement**.」
- `ctx.credentials`（第 605 行）：「**Configuration carries references to secrets; providers own the values.** Consumers resolve per operation, so a rotated credential reaches the very next request; the settings controller exposes **value-free views and write-only storage**.」

### 对本项目的意义

**(a) `ctx.credentials` 那条正是本项目 2026-09-23 脱敏改造的独立印证。** 当时的做法是：仓库里只留 ssh 别名（reference），真实 IP/端口/用户名/密钥由 `~/.ssh/config` 的 Host 块持有（provider owns the values）。DSH 把这条写成了成文原则，说明选型正确，可作为后续同类决策的依据。

**(b) `ctx.sandbox` 那条直接对治缺口 5。** 本项目 `verifier.py:62` 用 `_is_remote = isinstance(backend, RemoteExecutor)` 做行为分派，`RemoteExecutor` 还会静默降级 SSH→HTTP→Local，而 HTTP 路径返回编造的硬件信息（`remote_executor.py:660`：`compute_units:128 / 40GB / 2TB/s`）。按 seam 模式应改为：

- **Definition**：`Backend` 协议 + `capabilities` 声明（是否真设备、是否支持同机 baseline、是否可测带宽、是否可校准 launch 地板）
- **Provider**：`SSHExecutor` / `HTTPExecutor` / `LocalExecutor` / `MockGPU` 各自声明能力，**缺失能力必须显式失败，不得静默填默认值**
- **Consumer**：`Verifier` 按 capability 分支，不再 isinstance

⚠️ 研究员另报告了两条相关规则（未独立核实，落地前请查原文）：
- Consumer「要么在激活时 require 该服务，要么显式失败」，DSH 明写禁止 "silently defaulting a missing host value"
- Postmortem 0004：「An adapter must preserve structured failures owned by the seam below it instead of replacing them with its own nearest generic category.」

第二条与本项目高度相关：`RemoteExecutor` 把 SSH 失败降级为 LocalExecutor 正是「用最近的泛化类别替换下层的结构化失败」。

**(c) 硬件查表收敛（缺口 3）。** 同一模式可用：`HardwareProfile` 作 Definition，静态表（`hardware_caps._KNOWN_HARDWARE`、`verifier._PEAK_BY_NAME/_PEAK_BY_CC`）与 SSH 远端探测各作一个 Provider，所有 Consumer（Designer prompt、Verifier roofline、Planner 条件、driver）只从 Definition 取值。这直接消除「同一份 H20 事实散落 4 处、脊点曾错 4.07×」的问题。

代价：中-大。建议与缺口 3 的 `HardwareCalibrator` 合并为一次改造。

## 五、append-only + 类型指纹的持久化（缺口 2）

⚠️ 以下来自研究员报告，**未独立核实**：DSH 的会话日志是 append-only、按代版本化的 `session.vN.jsonl[.zstd]`，已提交代次永不重命名/覆盖/删除，迁移是相邻 `vN→vN+1` 的独立包，类型带 SHA-256 指纹机械校验；配套不变量是「**Model-visible means logged**」——模型所见上下文必须可从日志重建。

若属实，这套东西**不需要数据库**就能满足缺口 2 的需求，且与本项目「零重依赖」约束相容（jsonl + hashlib 都是标准库）。

对本项目的映射：
- `kernel_db` 落 jsonl 追加，按 schema 版本命名（`kernels.v1.jsonl`），已写入的代次不改
- 每类持久化 record 算 SHA-256 指纹存清单，读回时机械校验，schema 漂移立刻暴露而不是静默产出脏数据
- 迁移只写相邻一步（v1→v2），不做跨版本跳跃

⚠️ **但有一个前置条件**：必须先有缺口 1 的 `measurement_validity`，否则持久化只是把被污染的测量固化成语料——这与第三节的 postmortem 0002 是同一个失效模式，只是发生在知识库而非快照。

代价：中。

## 六、其余可借用的小规则（✅ 均核对自 `defensive-patterns.md`）

| 规则（原文要点） | 对本项目的意义 |
|---|---|
| **Never hand untrusted output the ambient environment or predictable paths**：spawned commands 用洗过的 env（丢弃 `*KEY*`/`*SECRET*`/`*TOKEN*`/`*PASSWORD*`）；临时文件用 0700 私有目录 + 随机名 + 独占创建 | 直接对应 `docs/问题识别与技术方案.md` 的 **P2-2（密钥与日志泄露，至今未实施）**。本项目把 LLM 生成的代码经 SSH 送到远端执行，`remote_scripts/run_kernel.py` 用 `tempfile.NamedTemporaryFile` 写代码——需核查是否继承了含 `LLM_API_KEY` 的环境 |
| **Unlink link-shaped paths**：可能是符号链接的路径要先 `lstat().isSymbolicLink()` 再 `unlink`，递归删除只用于已知真实目录 | ⚠️ **本项目现在正好有这个形状**：2026-09-23 在远端把默认 `work_dir`（`~/operator_agent`）做成了指向数据盘既有 uv 环境的符号链接，而 `SSHExecutor.deploy()` 会往 `{work_dir}/run_kernel.py` 写、`_setup_uv_env` 会在其中 `uv init`。当前只做 `mkdir -p` 与 scp，尚安全；但**任何未来加入的清理/重建逻辑都必须先判链接**，否则递归删除会穿过软链打到数据盘上的真实目录 |
| **Dispose must reach quiescence, not just request it**：teardown 要 await 子进程真正退出，且先关监听再杀进程 | 对应远程执行的清理：SSH 起的 `uv run` 若超时被放弃，远端进程是否真的终止？未核查，列为待办 |
| **Honor public contracts on BOTH sides**：一个结果有多种表示时，在公开 API 边界归一化，让 consumer 不必猜异常来自哪一层 | 对应缺口 6：`module.kernel(**inputs)` 的调用契约应显式化并在 L1 静态校验，而不是让 Fixer 从运行时 TypeError 反推 |

## 七、结论与建议

**采纳（按代价从低到高）**：

1. **正交结果独立上报**（第二节）→ 缺口 1 的目标形态。这是本轮调研最有价值的单条产出，建议作为缺口 1 的实现蓝本，而非「加一个 contaminated 布尔」。
2. **禁止自动刷新基线 + 不经基线的语义断言**（第三节）→ 代价小，且防的是一个一旦发生就极难发现的失效模式。
3. **env 洗白与临时文件权限**（第六节）→ 对应 P2-2，代价小。
4. **软链感知的路径操作**（第六节）→ 本项目已实际引入软链，属预防性加固，代价极小。

**改造后采纳**：

5. **Capability seam 三角色**（第四节）→ 缺口 5 + 3 + 6 的统一解法，代价中-大，建议与 `HardwareCalibrator` 合并为一次改造。
6. **append-only + 类型指纹持久化**（第五节）→ 缺口 2，但**必须排在缺口 1 之后**。

**不采纳**：

- DSH 的代码本身、Cordis 插件体系、profile/bundle 组合机制。技术栈不兼容（TypeScript vs Python），且项目处于 prerelease、官方明示会有破坏性变更。本项目要的是它成文的防御原则，不是它的运行时。

## 八、来源

一手（✅ 本人已拉取核对）：
- [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) · [docs/defensive-patterns.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/defensive-patterns.md) · [docs/capability-seams.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/capability-seams.md) · [postmortem/0002](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/postmortem/0002-js-expression-disabled-filesystem-tools.md)
- 注：本机 `raw.githubusercontent.com` 不可达，以上内容经 GitHub contents API 取 base64 解码核对。各文档均有 `.zh.md` 中文版可对照。

一手（⚠️ 研究员提供，未独立核对）：
- [docs/architecture.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md) · [docs/cordis-primer.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-primer.md) · [docs/persistence-catalog.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/persistence-catalog.md) · [docs/subsystems/invariants.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/invariants.md) · [postmortem/0004](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/postmortem/0004-landlock-partial-notice-misclassified-child-failures.md) · [Discussions（批评来源）](https://github.com/deepseek-ai/deepseek-harness/discussions) · [SAFETY.md](https://github.com/deepseek-ai/deepseek-harness/blob/master/SAFETY.md) · [npm @deepseek-ai/dsh](https://www.npmjs.com/package/@deepseek-ai/dsh) · [arXiv:2608.25512](https://arxiv.org/abs/2608.25512)

二手（仅作线索，不作论断依据）：
- [「模型无关的主板」](https://blog.csdn.net/Flying_Fish_roe/article/details/164190000) · [harness/framework/SDK 分类](https://segmentfault.com/a/1190000048297339) · [runoob 介绍](https://www.runoob.com/deepseek-harness/deepseek-harness-intro.html)

**已识别的污染源**：CSDN 上大量「实战/入门/避坑」类文章（`weixin_29055699`、`weixin_42651281`、`m0_63299656` 等）内容同质、无一手细节，疑为 SEO 批量生成；另有一批第三方蹭名仓库（`dsh-desktop`、`deepseek-harness-desktop`、各类 awesome-list）与官方无关。后续调研须避开。
