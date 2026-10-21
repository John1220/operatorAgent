# Changelog

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
  给 pnpm** 的，不会改 `dsh.profile.bundles`（不改就会在启动时报 `skipping profile bundle`）。
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
