# dsh-operator-agent

把 [operator-agent](../../README.md) 的**判定层**接成 DeepSeek Harness（DSH）的工具：
判一个算子核能不能编译、对不对、快不快，读数可不可信，以及按处方自动修。

**它不是什么**：不是新的 agent。判定全部来自 `operator-agent` 既有模块
（`Verifier` / `measurement` / `eligibility` / `anticheat` / `failure_route`），
这个包只提供 DSH 侧的安装壳与配置。

---

## 1. 前置

- DSH `0.2.1-rc.2`（本包按这个版本声明 `engines.dsh`；升版前先看 `CHANGELOG.md`）
- Python ≥ 3.9，且 `operator-agent` 可被宿主进程导入
- 真机执法需要 `ssh` 与目标机器的 `~/.ssh/config` 别名（可选，先跑 mock 也行）

## 2. 安装

### 2.1 Python 侧

```bash
# 从仓库装（当前）
cd /path/to/operatorAgent && pip install .

# 或从发布源（尚未上架）
pip install operator-agent
```

装完应有两个命令：`operator-agent`（CLI）与 `operator-agent-mcp`（MCP stdio server）。

### 2.2 从本地构建工件安装

本地交付的 wheel 与 DSH `.tgz` 尚未发布到包仓库。先将 wheel 安装到 DSH 启动时可访问的 Python 环境，再把 `.tgz` 加到目标 profile：

```bash
python3 -m pip install numpy /path/to/operator_agent-0.6.6-py3-none-any.whl
dsh plugin --profile <profile> add /path/to/dsh-operator-agent-0.1.15.tgz
```

确认安装命令可见：

```bash
command -v operator-agent
command -v operator-agent-mcp
```

如果 DSH 从 GUI 启动，GUI 未继承安装 Python 环境的 `PATH`，应在 profile 的 `cordis.patch.yml` 中将 `mcp-operator` 与 `operator-tools` 的命令改成对应绝对路径。卸载 `.tgz` 与源码目录安装相同，见 §2.4。

本次验收的工件哈希和平台边界见 [S6 验收记录](../../docs/status/dsh-operator-s6-acceptance-20261030.md)。本地验证保存的 NumPy wheel 仅适用于 macOS 14+ arm64 / CPython 3.9；其他平台应安装适配本机的 NumPy，不要复用该 wheel。

### 2.3 DSH 侧

```bash
# 本地目录安装（不需要发布 npm）
dsh plugin --profile <profile> add /path/to/operatorAgent/integrations/dsh-operator
```

> **`desktop` 这个 profile 有前置条件**：DSH Desktop 自己拥有它，CLI 会要求你先
> **完全退出 Desktop** 再跑（否则直接拒绝）。更顺的是在 GUI 里装：侧边栏的
> **插件**页，或让模型用 `plugin_manager` 工具——两者都不需要退出。

本包的 `cordis.patch.yml` 会插**两条**记录：`mcp-operator`（八个工具 + server 的
`instructions`）与 `operator-tools`（只有斜杠命令 `/op`；给模型的提示段不在这里，
见 §5.5）。手工安装时两条都要复制。

> **装成组合包，别装成手工条目**：只有前者会出现在 GUI 插件页（有图标、可开关）。
> 两者的区别、迁移步骤、以及「`engines.dsh` 失配导致组合包被静默跳过」这个坑，
> 见 [`docs/design/dsh-profile-and-bundles.md`](../../docs/design/dsh-profile-and-bundles.md)。

也可以直接从源码树跑（没 `pip install` 时）——把插件的 `command` 换成本机 Python：

```yaml
    - id: operator-tools
      name: dsh-operator-agent
      config:
        command: <abs-path-to-python3>
        args: ["-m", "operator_agent"]
        env:
          PYTHONPATH: <abs-path-to-operatorAgent>
```

（注意 `name:` 用包名需要 `dsh plugin add`；按**绝对路径**插入时写 `index.mjs` 的路径。）

然后重连 MCP：

```bash
pkill -f operator_agent.mcp_server     # 宿主会在 500ms 后自动重连并 re-sync
```

> 工具表是 `mcp_server.py` 里的**静态表**，所以加/改工具要**重启 server 进程**（就是上面
> 这条）——不是宿主不肯刷新。宿主本身**支持**热刷新（`dsh-mcp-client` 收
> `notifications/tools/list_changed` 就重新同步），只是这个 server 还没有动态工具表；
> 哪天有了，在 `initialize` 里声明 `capabilities.tools.listChanged` 并补发那条通知即可，
> 不用重连。**当前会话仍要到下一回合**才能看到新工具。
>
> 重连是**有上限的**：默认 500 ms 起、指数退避到 30 s、最多 10 次；耗尽后这个 server 的
> 工具会**整体注销**，只能重新加载插件或重启宿主才能回来。所以别把 `pkill` 当日常刷新手段。

如果 `operator-agent-mcp` 不在宿主的 `PATH` 里（常见于装进 venv），把
`cordis.patch.yml` 的 `command` 改成本机绝对路径，或改用：

```yaml
        command: python3
        args: ["-m", "operator_agent.mcp_server"]
```

### 2.4 卸载 / 禁用

三种手段，效果不一样，别混：

| 手段 | 效果 |
|---|---|
| **GUI 侧边栏 插件 页**：关开关（禁用）/ 删除 | 推荐。删除会同时摘掉 profile 的 `dependencies` 与 `dsh.profile.bundles`；禁用只是让 bundle 不生效（patch 不再插入），文件还在 |
| `dsh plugin --profile <profile> remove dsh-operator-agent` | 实测（本机 CLI `0.1.1-rc.2`）**会同时**摘掉 profile 的 `dependencies` 与 `dsh.profile.bundles`。你的 CLI 版本不同的话，卸完 `grep dsh-operator-agent <profile>/package.json` 确认一眼即可。`desktop` profile 还要求先完全退出 Desktop |
| 纯手工 | 从 profile（或你放这两行的那个 patch）里删掉 `mcp-operator` 与 `operator-tools` 两条，再把名字从 `dsh.profile.bundles` 去掉 |

两半是**独立**的：卸载插件不影响 Python 侧的 `operator-agent` CLI；反过来
`pip uninstall operator-agent` 会让八个 MCP 工具全部变成「命令不存在」——`/op` 会明说
找不到命令，宿主日志里记一条 `operator-tools: 找不到命令 …`（error 级，同类只记一次）。

## 3. 配置内层 LLM（只有 `operator_run` / `operator_repair` 需要）

`operator_prepare` / `operator_verify` / `operator_capabilities` **不需要** LLM。
需要的是「让本仓自己设计/写核/修复」的那两条路。在 mcp-client 那一行加 `env`
（只放**引用**，不要把密钥值写进 YAML）：

```yaml
        env:
          LLM_BASE_URL: https://your-endpoint/v1
          LLM_MODEL: your-inner-model
          LLM_API_KEY_FILE: ~/.config/operator_agent/llm.key   # 0600，仓库外
          # 阿里云百炼的思考开关使用顶层字段；其他服务按其接口选择
          LLM_THINKING_PARAMETER: enable_thinking
```

也可以用 `LLM_API_KEY` 直接给值（不推荐）。没配时这两条工具会立刻返回
`llm_not_configured`，**不会**起一个跑不出结论的进程。

`operator_run` 和 `operator_repair` 共用客户端的密钥解析：显式密钥、`LLM_API_KEY`、
`LLM_API_KEY_FILE` 依次取值。文件为空或不可读时拒绝调用；文件内容不会写进环境变量。
`LLM_THINKING_PARAMETER` 默认是 `chat_template_kwargs`；设置为 `enable_thinking` 时，
请求改用顶层字段。思考仍默认关闭，输出预算不变。更换服务后也应清空不再使用的
`LLM_FALLBACK_MODEL`，避免主模型失败后切回旧模型。

> 为什么必须写在 `env:` 里：DSH 会先洗掉子进程环境里名字**含 `KEY`/`PASSWORD`/`SECRET`/`TOKEN`**
> 的变量（`/KEY|PASSWORD|SECRET|TOKEN/i`），**外加所有 `DSH_*`**，再应用显式覆盖。
> `LLM_API_KEY_FILE` 含 `KEY`，靠全局环境变量传不进去；`DSH_HOME` 之类也一样传不进去。
>
> `/op`（宿主插件自己起的那条子进程）用的是**同一套规则**：先洗、再合并配置里的 `env`。

## 4. 验证装好了

```bash
operator-agent-mcp --selftest     # 不依赖宿主，走一遍握手
dsh --profile <profile> --dump-config | grep mcp-operator
```

然后在会话里让模型调一次 `operator_capabilities`：返回 `backends.{mock,local,ssh}.wired`
就说明链路通了。

## 5. 八个工具

工具契约 **6** 起，`verify` / `repair` / `bench` 各轮独立留存源码与完整结果。
回包的 `evidence` 指向完整事实，`verify` / `repair` 还给出源码路径与完整 SHA256；
每轮报告独立，旧报告保持原样。`repair` 每次实际验证也有独立证据，`bench` 仍仅供探索。
`complete.json` 只证明留存完成，不代表源码正确或性能达标。数据布局见[实现说明](../../docs/design/dsh-plugin-implementation.md#3-数据布局一个-run-目录)。

| 工具 | 一句话 | 要 LLM |
|---|---|---|
| `operator_capabilities` | 这台机器能做什么（只读，不联网、不探测 SSH） | 否 |
| `operator_prepare` | 声明 → 冻结契约 + 契约自带的输入 + 回合身份 | 否 |
| `operator_verify` | 对**外层给的**源码判 L1/L2/L3，返回正交事实 + 处方 + 资格 | 否 |
| `operator_bench` | **多个**候选只跑 L3 的批量测量（探索级，**不可裁决**）：调参/扫网格用 | 否 |
| `operator_repair` | 给一份**已有的**核：判 → 处方 → 内层模型修 → 再判（≤N 轮） | 是 |
| `operator_run` | 起一次**无人值守**整段（本仓自己设计/写核/修），立刻返回 pid | 是 |
| `operator_cancel` | 按 run/invocation 请求取消，再查 status 确认终态 | 否 |
| `operator_status` | 查无人值守的相位：`running` / `done` / `failed` / `cancelled` / **`lost`** | 否 |

**耗时与超时**：真机上一次**完整**判定（L1→L3、L2 的 15 个边界 case）实测 **866 s**，所以
`cordis.patch.yml` 里 `toolCallTimeoutMs` 设成 **3600 s**（60 min：单次 866 s，`repair` 默认
2 轮约 30 min）。**别往回调小**。契约 7 起 server 支持 MCP 取消通知；客户端超时本身仍不是本机/远端停止的证明，须查终态和清理回执；
更长的工作请用 `operator_run` + `operator_status` 轮询（`run` 立刻返回 pid，不占一次调用）。

工具契约 **7** 起新增 `operator_cancel`、CLI `cancel` 和 `/op cancel <run_dir> <invocation_id>`。
后台 run 立即返回实例 ID；同步 verify/repair/bench 由独立 worker 执行，支持 MCP 取消通知。
CLI Ctrl-C/SIGTERM 会清理本机 worker 后返回取消 JSON（退出码 130）。
`accepted` 只表示请求已记录；需查 `status` 的终态、预算及清理回执。未完成的候选不因此通过，已发生调用不退款，远端与服务商是否停止保持未知。
完整语义和目录见 [任务取消与中断](../../docs/design/task-cancellation.md)。

### 5.1 生成与修复质量

核心生成/修复提示词已共用版本化规约。`operator_prepare` 返回规约身份；`operator_repair` 默认加入与当前源码、契约及数值反馈绑定的诊断，并返回 `initial_full_l2_passed`、`full_l2_passed` 和完整证据文件位置。当前诊断识别器聚焦 LayerNorm 尾部归约 mask；其他模式明确返回 `unknown`。诊断是修复假设，最终结果仍需完整验证。

`operator_verify` / `operator_repair` 另有一格**归因**（不是判据）：`candidate_origin` 记这份源码自称是谁写的（`outer` = 你或人写的、`inner_fixer`、`inner_coder`），省略则回包与 `report.md` 记 `undeclared`；`operator_repair` 还回一个**派生**的 `final_candidate_origin`（有任一轮真改了源码就是 `inner_fixer`，那一格不接受声明）。它不影响任何通过与否，存在的目的是让「外层裸写的核」与「走过规约/技能注入的核」事后分得开——动机与判据见[分工边界](../../docs/design/hitl-generation-boundary.md)。

无实际源码或 AST 变化的修复会停止，避免反复验证相同失败候选。输入保真或参考身份未知时先恢复观测资格。固定 10 任务的计划、回放与对照命令见[质量闭环说明](../../docs/design/operator-quality-improvement.md)；CPU 回放不代表新模型或 GPU 质量提升。

新增的[配对评测入口](../../docs/design/operator-quality-paired-evaluation.md)用 `--paired` 生成同一初始候选，两组独立修复；SSH 评测会先检查 GPU、现有运行时及 runner，预检失败不会请求模型。`--preflight PLAN` 可单独检查环境，不需要模型凭据。

## 5.5 斜杠命令 `/op`

命令在**宿主进程**里执行，`/op status` 的输出直接给人看、不进模型上下文：

```
/op capabilities          这台机器能做什么（只读，不跑活）
/op status <run_dir>      查无人值守的相位（running/done/failed/lost）
/op help                  用法
```

`run_dir` 的解析规则（**相对路径不是按宿主进程的 cwd 解析**——GUI 进程的 cwd 多半是 `/`，
按它解析会报「run-status.json 不存在」，看着像 run 丢了）：

| 写法 | 解析成 |
|---|---|
| `/abs/path` | 原样 |
| `runs/2026-10` | **会话工作区**下的 `runs/2026-10`（`ctx.workingDirectory`） |
| `~/runs/x` | 你的 home 下的 `runs/x` |
| `"my runs/x"` | 去掉引号后按上面的规则；路径里的空格**整段保留** |

查不到 `workingDirectory` 服务时原样传给 CLI（不因为一个可选依赖把命令变成不可用）。真 profile 里这个服务由 `@deepseek-ai/dsh-base` 装上，而 `ctx.get` 对**没 inject** 的服务
也读得到 —— 两条都在本机验过，所以这条解析在真机上是生效的。
输出里会回显**实际读取的目录**，所以相对路径写歪了一眼能看出来。

需要「判定/修复/从头自动化」时用模型工具（`mcp__operator__*`）：命令只是旁路，
不替模型做判定。

### 输出长什么样（以及为什么首行必须是回执）

`/op` 的结果在界面里是一个**默认折叠**的节点（宿主 `ui-chat` 的 `GenericCommandCard`），
不是一条助手回复 —— 它不产生轮次，也不会有「已完成」标记。折叠摘要的样式是
`white-space: nowrap; text-overflow: ellipsis`，也就是**只有首行可见，放不下的直接截掉**。

所以插件把**回执**放在首行：

| 情形 | 首行 |
|---|---|
| `/op capabilities` | `✓ 已执行 · 算子判定层 0.6.0 · 后端 local/mock/ssh · 10 个算子有输入布局` |
| `/op status …` | `✓ 查询完成 · 相位 done（进程已结束）` |
| 判定层拒绝 / 未装 / 非 JSON | `✗ …`（原因跟在后面） |
| 被宿主取消 | `已取消`（取消是人的正常操作，不挂 `✗`） |

两条硬约束，改首行前必须知道：

- **回执必须在前**。没有它的话，折叠态的「跑成了」和「失败了」长得一模一样 ——
  都是命令名加一段摘要，用户只能点开才知道结果。早期版本就是这样被误读成
  「命令没反应」的。`✓` 和 `✗` 是唯一在收起状态下可分辨的信号。
- **首行要短**，且不放路径、不放低频字段（`tool` 号挪到明细）。摘要 nowrap 截断，
  塞长了会把真正有用的部分（版本/后端/覆盖数/相位）挤到省略号之后。
- **不进模型上下文**是设计而非缺陷。想让模型用上这些信息，必须走 `mcp__operator__*`
  工具，`/op` 的文本模型永远看不到。

`test/plugin.test.mjs` 里有一组测试钉住这些（首行含 ✓、长度上限、tool 号不进首行、
四类失败都有 `✗`、明细仍完整、空后端不崩），改格式化逻辑时先让它们红。

> 给模型的**使用说明**（什么时候该用、回来怎么读）不存在于本插件里：它是 Python 侧
> `operator_agent/mcp_server.py` 的 `initialize.instructions`，宿主会自动渲染成
> `mcp:operator` 提示段。只有一处来源，换任何 MCP 客户端也都在 —— 插件只负责 `/op`。

细节（契约、拒绝码、数据布局、扩展点）见
[实现说明](../../docs/design/dsh-plugin-implementation.md)。

## 6. 威胁模型（请读完再上真机）

- **它会在你的 GPU 主机上执行模型写的代码。** `verify` / `repair` / `run` 的候选源码会被
  送到远端执行；`exec_guard` 提供了墙钟、rlimit、进程组与逐项强制读数，但**不是沙箱**
  （同 uid、无容器、网络未关）。
- **它不是敌意代码的隔离器**：共享机器上请只跑你愿意信任的候选。
- **MCP 工具不经过 DSH 的审批层**（MCP 客户端拿不到 `ctx.approval`）：谁能用这个 DSH，
  谁就能在配置的目标主机上跑代码。
- **凭据**：`LLM_API_KEY` / `LLM_API_KEY_FILE` 只被 Python 侧读取，不会写进产物；
  `LLM_API_KEY_FILE` 建议 `0600` 且放在仓库外。

## 7. 已知限制

- 只接线了 `mock` / `local` / `ssh` 三个后端；**没有 `auto` 降级**（不给 host 就报
  `host_required`，避免「以为在真机上、其实在控制机」）。
- `ssh` 单主机、单卡、串行；没有多卡/并发调度。
- **取消的外部边界**：契约 7 支持按实例取消后台任务和前台长调用；本机清理后远端 GPU/服务商是否停止仍为未知。不可捕获退出可能留下 `lost`。
  取消 `/op status` 只停止旁路查询；取消任务使用 `/op cancel <run_dir> <invocation_id>` 并确认终态。
- 没有 UI/曲线面板（那一格由其他插件占位；本包的差异化在判定层）。
- 契约 8 的资源预检只采信明确的共享存储证据；逻辑 tile、累加器 dtype、动态/多配置的占用保持未知，交给原生检查。见 [S3 资源边界](../../docs/design/resource-precheck-evidence.md)。

## 8. 构建与发版（维护者）

```bash
npm test        # 58 条：注册面/行为/格式化契约（真 CLI fixture）+ 真宿主契约
```

`npm test` 里有两层守卫，别只跑一层：

| 文件 | 钉住什么 | 依赖 |
|---|---|---|
| `test/plugin.test.mjs`（49 条） | 注册面、`/op` 行为、路径解析、超时/取消、`scrubEnv`/`normalizeConfig`、宿主日志（同类只记一次）、两个 formatter 吃**真 CLI fixture** | 无（替身 ctx + 假 CLI） |
| `test/host-contract.test.mjs`（9 条） | 把插件注册进**装在机器上的** DSH 自己的 `cordis` + `dsh-commands` 服务真执行（含 `agent` 透传），并用真 `readPluginMeta` 读一遍「脸」、用真 `evaluatePluginCompatibility` 核 `engines.dsh`；最后一条用**真 MCP SDK 客户端**连真 server 验 `instructions` 到达客户端 | DSH 安装；最后一条还需要能 import 判定层的 python3。找不到就**跳过**（`DSH_APP_ROOT` / `OPERATOR_AGENT_PYTHON`） |

第二层是必要的：替身 ctx 只会照单全收，而宿主的 `normalizeDefinition` / `normalizeResult`
会拒绝不合规的命令定义与返回值；元数据那几条更只认**真模块解析器 + `exports` 映射**。

**给插件配「脸」的三个资源**（宿主 `readPluginMeta` 读的就是这三样，规则已在
`tests/hygiene/test_packaging.py::TestTheDshPluginFace` 里镜像一份）：

- `"./package.json"` **必须**在 `exports` 里——宿主拿裸包名走真解析器读 manifest；少了这条，
  manifest 读不到 ⇒ 卡片标题退化成 `dsh-operator-agent`，`icon` 也永远声明不了（静默，不报错）。
- `locale/en.json` 是宿主**列目录枚举语言**的入口，缺了等于整个 locale 不存在；每个 locale
  文件还要被 `"./locale/*.json"` 导出、进 `files`。
- `icon` 只能是包内相对路径，SVG/PNG/JPEG/WebP，≤256 KiB，且要进 `files`。

```bash
# 从**干净的树**构建（见下面的坑）
python -m pip wheel . --no-deps -w dist/
```

⚠️ **坑**：`pip < 21.3` 会先把本地包**复制**到临时目录再构建，而这个仓库的 `output/`
里有 `run_layout` 写的 `0400` 产物文件 —— 复制之后 pip 删不掉，构建以 `PermissionError`
收场（wheel 其实已生成，但退出码是 1）。两个解法任选：

- `pip wheel . --use-feature=in-tree-build`（pip 21.1+），或升级 pip；**或**
- 从干净导出构建：`rsync -a --exclude output --exclude logs --exclude triton --exclude .git ./ /tmp/oa-src/`
  然后在 `/tmp/oa-src` 里构建。

发布形态（2026-10-30 起）：`private` 已改为 `false`、`LICENSE`/`NOTICE` 已随包、可以发布。
版本号现推进到 `0.1.15`（Python 侧 `0.6.6`）——`0.6.1` 曾同时指平铺 wheel 与分层源码，同号两物，
换号理由与身份断链的明细见 `docs/phase1/delivery-identity-rebind-20261031.md`。
实际执行顺序是 Python 包（PyPI）→ npm 壳（`npm publish`）→ README 里补一行 `dsh plugin add dsh-operator-agent`。
工具契约仍是 **9**，本次没有改工具行为。逐字节副本与发版形态由
`tests/hygiene/test_packaging.py::TestTheDshReleaseShape` 和 CI 的 `release-shape` 作业守住
（`.github/workflows/ci.yml`）。若将来把 `operator-dev-agent-design/` 的静态资源打进
发布物，记得把 `THIRD_PARTY_NOTICES.md` 也列进 `files`。

## 9. License

**Apache-2.0**，见仓库根目录的 [`LICENSE`](../../LICENSE) 与 [`NOTICE`](../../NOTICE)
（版权人：`the operator-agent authors`）。npm 包里各带一份逐字节副本
（[LICENSE](LICENSE)、[NOTICE](NOTICE)），因为安装方拿到的只有包内文件。
第三方组件（ECharts / Mermaid）保留各自许可，见 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)。
