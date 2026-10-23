# Changelog

## 0.1.4

**文档更正**：上一版写错的两处事实，用隔离 `DSH_HOME` + 本机 CLI 实测推翻。

- **`engines.dsh` 失配 ≠「装不上」**。`dsh plugin --profile <p> add …` 是 pnpm 透传，
  **不做版本门禁**：用 CLI `0.1.1-rc.2` 装 `engines.dsh: ^0.2.1-rc.2` 的包，装得上。
  真正的把关在 **boot** —— `dsh-app-boot` 解析 profile 的 bundle 列表时
  `evaluatePluginCompatibility` 不过就抛错、被捕获进 `skippedBundles`，在 stderr 留一行
  `dsh: skipping profile bundle "<name>": <reason>`，然后照常启动。
  所以失配的表现是「**装好了但没生效**」，比报错更难查；`host-contract` 里那条
  `engines.dsh` 守卫正是为这个留的。
- **`dsh plugin … remove` 会同步 `dsh.profile.bundles`**。原来说「只摘 `dependencies`、
  得手工删 bundles、否则启动报 `skipping profile bundle`」是错的：实测 `add` 与 `remove`
  都会把 profile 的 `dsh.profile.bundles` 改成 `[base, web-app]` / `[base, web-app, 本包]`。
  README §2.3 已按实测改写（并留了「你的 CLI 版本不同就 `grep` 一眼确认」的话）。
- 顺手把「`ctx.get` 对未 inject 的服务也读得到 + `dsh-base` 装了 `dsh-working-directory`」
  这两条实测结论写进 README §5.5：`/op status` 的相对路径解析在真 profile 里是生效的，
  不是死代码。

## 0.1.3

给模型的说明**只剩一处来源**（判定层的回包契约没动，`/op` 行为没动）。

- **提示段搬进 server 的 `initialize.instructions`**：原来同一段规则有三份副本（插件的
  `tool:operator` 系统提示段、MCP 工具描述、设计文档），改一处就得记得同步另两处。
  现在它是 `operator_agent/mcp_server.py` 的 `INSTRUCTIONS`，由宿主
  `dsh-mcp-client` 的 `registerServerContext` 渲染成 `mcp:operator` 段（order 取
  `MCP_SERVERS`、`interpolate:false`、懒取），**任何 MCP 客户端**都吃得到，不限于 DSH。
  插件随之只剩 `/op` 一件事，`inject` 从 `["commands", "systemPrompt"]` 收成
  `["commands"]`（少一个启动依赖）。
- **端到端守住了「说明真的过线」**：新增一条测试，用宿主自己依赖的那个
  `@modelcontextprotocol/client` 起真 stdio 连接，断言 `getInstructions()` 里含
  `eligibility` / `hold_for_measurement` 且小于宿主 `maxInstructionBytes`（32 KiB，
  超了是**拒绝连接**而不是截断）。找不到 python 就跳过并说明原因。
- 指令文本顺带对非 DSH 客户端友好：说明里点明「DSH 里这些工具显示为
  `mcp__operator__*`」，其它客户端看到的是 `operator_*`。

## 0.1.2

`/op` 的可用性与安全性收口 + 两条防失配守卫（判定层的回包契约没动）。

- **相对路径按会话工作区解析**（`ctx.workingDirectory.get(agent.session)`）。原来相对
  `run_dir` 是按**宿主进程的 cwd** 解析的（GUI 进程多半在 `/`），于是「run 在但路径不对」
  被报成「run-status.json 不存在」，看着像 run 丢了。命令的 invocation 里本来就有
  `agent`（`{commandId, agent, rawInput, attachments, signal}`），我们只接了后两个。
  `~/x` 展开到 home；拿不到 `workingDirectory` 时原样传下去（不因为一个可选依赖把命令
  变成不可用）；输出回显**实际读取的目录**。
- **含空格的路径不再被切碎**：`parseInput` 原来按空白切开再拼，`/op status /a/my runs/x`
  直接散架。现在 `rest` 是原样剩余整段，成对引号会被去掉。
- **`/op` 的子进程环境按宿主同一套规则清洗**（`scrubEnv`：名字含
  `KEY`/`PASSWORD`/`SECRET`/`TOKEN` 或带 `DSH_` 前缀的一律不给）。`command` 是可配置的，
  原来等于把宿主密钥交给一个可替换的二进制；显式 `config.env` 在清洗**之后**合并，
  所以真要传仍传得进去。
- **配置校验**（`normalizeConfig`）：`comand:` 拼错原来会让 `/op` **静默**换成默认二进制，
  `timeoutMs: "30s"` 会静默变回 60000。现在未知键/类型不对都在宿主日志里 warn 一次
  （`timeoutMS` 这种大小写错误会给「正确写法是 timeoutMs」的提示）。
- 两条防失配守卫：真宿主测试里用**宿主自己的** `evaluatePluginCompatibility` + 装在机器上的
  版本核 `engines.dsh`（失配**不会**被 `dsh plugin` 拦住，而是在 boot 时把这个 bundle 跳过并
  在 stderr 打 `dsh: skipping profile bundle …`，见 0.1.4 的更正）；Python 侧钉住 patch 行里非官方
  的 `name:` 必须等于包名（写错是「装上了但 boot 不起来」）。两条都用变异验证过会红。
- 日志去重机制合并成一处（`said`），同类只记一次；配置告警与运行时失败各自成组。

## 0.1.1

对着 DSH `0.2.1-rc.2` 的源码逐条核对后的修补（判定层的回包契约没动）。

- **`toolCallTimeoutMs` 600 s → 3600 s**：真机一次完整判定实测 866.8 s
  （评估文档 §9.8），600 s 会在判定完之前先被 MCP 层掐掉；`repair` 默认 2 轮约 30 min。
  超时不是优雅退出（server 同步逐行读，会算完再写一个没人认领的响应），所以宁可给足。
- **`/op status` 不再把终态显示成「在跑」**：Python 侧终态给的 `process_alive` 是 `null`
  （不是 `false`），按死活判会把一个 `done` 的 run 显示成「进程 在跑」。现在按相位判，
  四态各有各的文案，未知相位不猜（`进程状态未知`）。
- **`ok:false` 的回包不再伪装成成功**：`/op capabilities|status` 现在回 `kind:"error"`
  并只打一行原因，不再让 `capabilities` 打出一排「?」。
- **`engines.dsh` 由精确钉死改为 `^0.2.1-rc.2`**：原来的写法只有 `0.2.1-rc.2` 判通过，
  升一个 rc（或 0.2.1/0.2.2）都要人工 `allow-version --accept-risk`。
- 测试 8 条 → 15 条：新增真 CLI 输出 fixture（`test/fixtures/`）、四态文案、`ok:false`、
  超时分支（`err.killed`）、`config.env` 合并。Python 侧同步加两条跨边界守卫
  （壳读的字段仍然存在、真机耗时是超时的下界）。
- 修 fixture 暴露的一处测试断言错误：真 CLI 的 JSON 是 `sort_keys=True`，
  `backends` 顺序是 `local / mock / ssh`。

### 0.1.1（续）：代码卫生与事实修正

- **`/op` 的取消真的生效**：`handle` 收到了 `signal` 却丢掉，`execFile` 支持它。
  现在接上（`runCli(..., signal)`），并把取消排在超时**前面**——abort 也会让 `err.killed`
  为 true，先判 `AbortError` 才不会把「人取消了」读成「超时」。测试：18 条，含
  「真的把子进程收掉」（50 ms 内返回，实测 51 ms）。
- 命令补 `definitionId: "dsh-operator-agent"`（UI 侧跨重启配对的稳定身份）。
- **新增真宿主契约测试**（`test/host-contract.test.mjs`，4 条）：用机器上装着的 DSH 自己的
  `@deepseek-ai/cordis` + `@deepseek-ai/dsh-commands` 起真服务，把插件注册进去、真执行一次
  命令，验「服务侧看到的 name/definitionId/hint/段顺序」「handler 的返回值过宿主的
  `normalizeResult`」「取消由宿主挡下（真 `AbortError`）」「error 文案非空」。
  找不到 DSH 安装就跳过（`DSH_APP_ROOT` 可指定），所以不依赖 Python、也不假装通过。
  替身 ctx 只会照单全收，这一层才验得出契约漂移。合计 22 条。
- 删掉两处**死码/误导**：`ctx.get("systemPrompt")` 兜底分支（`inject` 是必需语义，
  缺了 fiber 根本不启动，那段永远不会执行）、以及 `apply` 返回的 `registrations` 数组
  （cordis 不看 apply 返回值；生命周期本来就归它，注册的 effect 挂在调用方 ctx 上）。
- **文档事实修正**（对着 DSH `0.2.1-rc.2` 源码核过）：
  - 「工具清单在 server 启动时固定…这是宿主的行为」**是错的**。宿主收
    `notifications/tools/list_changed` 就重新同步（`listChanged.tools.onChanged`）；
    真正的原因是这个 server 的工具表在进程里是静态的 ⇒ 改它要重启**进程**。
    同时写明重连**有上限**（500 ms→30 s 退避、10 次，耗尽后工具整体注销）。
  - 环境清洗的准确边界：`/KEY|PASSWORD|SECRET|TOKEN/i` **外加所有 `DSH_*`**；并点明
    `/op` 那条路用的是宿主完整环境，两条子进程路径的 env 策略本就不同。
  - 补 `desktop` profile 的安装前置条件（DSH Desktop 拥有它，CLI 要求先完全退出；
    更顺的是 GUI 插件页 / `plugin_manager` 工具）。
  - 设计文档 §9.1 的两处过期描述（「只有 YAML 与文档，没有 JS」「License 未定 /
    `UNLICENSED`」）、`mcp_server.py` docstring 的「三个动作/三个工具」（实际六个）。

### 0.1.1（续）：宿主里的「脸」、日志、卸载路径

对着 DSH `0.2.1-rc.2` 的 `readPluginMeta` / `iconOf` / `dictionariesOf` 源码补的：

- **`exports` 补 `"./package.json"` 与 `"./locale/*.json"`**。这是真缺口：宿主拿裸包名走
  **真模块解析器**读 `<包名>/package.json`，我们原来只导出 `"."` ⇒ 抛
  `ERR_PACKAGE_PATH_NOT_EXPORTED`、被吞成 `undefined` ⇒ 插件页标题**静默退化**成裸包名
  `dsh-operator-agent`，而且 `manifest.icon` 这条图标路径也一起断了。
  （bundle/patch 的解析走磁盘路径，不受影响——所以这只是「脸」，不是装不上。）
- 新增 `icon.svg`（36×36，与官方同规格）、`locale/en.json`、`locale/zh.json`
  （`meta.title` / `meta.description`；`en.json` 是宿主枚举语言的入口），`files` 与
  `repository` 一并补齐。
- **失败在宿主日志里留痕**：`/op` 的失败原来只在命令输出里可见，`ctx.logger` 一次没用。
  现在「命令不存在」记 error、超时/非 JSON 记 warn，且**同类只记一次**（`reported` 随插件
  实例生灭）——取消不记，那是人的正常操作。
- README 补「2.3 卸载 / 禁用」：GUI 插件页 / CLI / 纯手工三条路，并点明 CLI 那条是**透传
  给 pnpm** 的（`add` / `remove` 都会顺带同步 `dsh.profile.bundles` —— 见 0.1.4 的更正）。
- 守卫：`test/host-contract.test.mjs` 用**真 `readPluginMeta`** 经 `node_modules` 符号链接
  读一遍元数据（去掉任一条 exports 都会红，已用变异验证）；`tests/test_packaging.py`
  新增 `TestTheDshPluginFace` 把宿主的 locale/icon 规则镜像一份（不需要装 DSH）。
  合计 JS 26 条 / Python 侧 64 条通过。

## 0.1.0

首个可安装版本。

- 六个工具：`operator_capabilities` / `operator_prepare` / `operator_verify` /
  `operator_repair` / `operator_run` / `operator_status`。
- 输入按 seed 物化（`numpy-seed-v1`）+ 双向 digest 核对，不一致自动退回传字节
  （`input_seed_fallback` 留痕）。实测真机一次判定从 572 s 降到 16.5 s。
- `operator_repair`：判 → 处方 → 内层模型修 → 再判。6 类真实缺陷里 5 类一轮改对。
- `operator_run`：独立进程跑本仓 `Conductor`，四态查询（含 `lost`），语料/日志/产物
  全部收在 run 目录内。
- 资源预检：注释与 docstring 里的 `BLOCK_*` 不再被当成配置（真机上曾把正确的修复再拒一次）。
- Python 侧首次提供 `pyproject.toml`；`protocol_identity()` 改为包相对路径，
  源码树与 site-packages 两种布局都成立。
- 宿主插件（`index.mjs`）：斜杠命令 `/op capabilities|status|help` + 一段
  `tool:operator` 系统提示段（说明 `mcp__operator__*` 何时用、`route` 与 `eligibility`
  必须一起读）。命令走 `operator-agent` CLI，**不复制判定逻辑**；8 条 Node 测试。
