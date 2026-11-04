# Changelog

## 0.2.0-rc.2

配套 Python `0.7.0rc2`，工具契约 `10`，仍未正式发布。本版把 2026-11-04 真机跑出来的
缺陷逐条修掉，并首次跑完一整轮八格（记录见 `docs/status/phase1-eight-cell-20261104.md`）。

**修掉的缺陷（每一条都有回归用例）**

- **长响应被自己的网关掐断**：SSE 每片段都带封套，32K 输出在远未到 token 上限时就撞上
  1 MiB / 4096 块的老界，DSH 侧只看到 `Stream ended without finish_reason`。
  实测：同一请求重复 6 次，5 次卡在 1,048,xxx 字节。现在按输出上限缩放
  （32K ⇒ 32 MiB / 65536 块，4K 档保持原值）。本轮八格 178 条流记录 0 失败。
- **远端编译错误被记成基础设施故障**：`_INFRA_MARKERS` 含裸词 `ssh`，而编译错误消息形如
  `Remote compile error via SSH: ...`，于是被判 `infra_retry`（一次白烧 1148 s / 20 次调用）。
  现在编译错误优先，且裸词 `ssh` 换成具体形态。
- **预算耗尽被记成宿主故障**：改读账本 `remaining_calls`，记 `budget_calls_exhausted`。
- **宿主退出会丢掉已通过的候选**：验证服务的判定与宿主生死无关，现在取「最后一次
  各关全过的提交」；只在工作进程显式声明预算停止/通过后结束时才允许触发独立终评。
- **A 臂没有停滞规则**：连续 2 次 **valid** 的「跑对但没更快」即关闭验证服务
  （`no_gain_2_rounds`）；无效测量与编译失败不计入。
- **单个 case 传输未知会废掉整格**：已确认未知后补发一次新请求（不重试在飞请求）。
- **验证每次重连**：常驻会话默认开启。实测 L1+L2 384.9 s → 60.0 s，全量 405 s → 31.9 s。
- **PDL 配方教了一个编译不过的写法**：`@triton.jit(launch_pdl=True)` 在 Triton 3.7.1 上必失败，
  已改为只在启动点开。

**已知不合格**

- `sampling_comparability` 仍为 `not_established`（DSH 外层 usage 对账缺失，费用 unknown）。
- 八格本轮只有 softmax 一对可判；`relu-large` 的 1.009 落在判决带宽内，不能当收益。
- 主机满载（load 188/192）时 valid 带宽涨到 10–14%，此时任何 speedup 比较都不可判。

## 0.2.0-rc.1

v0.7 本地交付候选，配套 Python `0.7.0rc1`，工具契约 `10`，尚未正式发布。

- 新增显式 FakeBackend：CPU Python 验证、脚本模型、修复场景与预算/取消检查；输出始终标记 test_only，不能取得 GPU 性能验收资格。
- 修正 calibrated FP16 LayerNorm v2 的 X-only 输入契约，以及声明覆盖精度后的 profile 重算。
- 提供随包离线示例和 Python 3.12 安装验证；从 wheel/tgz 的干净环境验收与源码树检查分开记录。
- 本候选不包含正式同卡八格收益结论；已知限制和回退步骤见 v0.7 交付说明。

## 0.1.16

升号原因在 wheel 内，而且改的是**被读得最多的两个文件**：`core/hardware_caps.py`（新增
`TRITON_MIN_CC_MAJOR` / `triton_cc_verdict()` / `require_triton_support()` / `TritonUnsupported`）
与 `app/conductor.py`（紧跟硬件探测之后做一次 CC 下限预检）。

**行为变更**：目标设备申报出 `compute_capability < 7.0`（TITAN X 5.2 / P4、P40、P100 6.x）时，
运行在**第一次 LLM 调用之前**抛 `TritonUnsupported` 并停住，理由里带读到的 cc、设备名、下限与后果。
以前这类卡会一路跑到编译期，而失败路由的 `_INFRA_MARKERS` 只认 oom 那一族 ⇒
架构不支持被记成**候选失败**，跑完交出的是一张假的能力表（钱也照烧）。

- 7.x（Volta / Turing）只打一行 ⚠️ 提示：无 cp.async、bf16/tf32 视代次缺失、工具链正在移出 sm_70。
- **不拦「问不到 cc」的那些**：MockGPU、CPU 回退、`gfx90a` 一律判 unknown 后放行——那是探测诚实性
  的活（`smem_budget()` 与迁移门采集器管），混进来会把无 GPU 的开发路径整条打死。
- Python 侧同步 `0.6.7`；工具契约仍为 **9**，MCP 工具面一条命令都没动。
- **协议摘要本期第二次换号**：`b937fd8e…` → `5b9b0c8f…`（明细见 `docs/phase1/delivery-identity-rebind-20261031.md` §5.5）。
- 已知未覆盖：`verify` / `repair` **不经 Conductor**，那条路上架构不支持仍会被记成候选失败。
  不动 `_INFRA_MARKERS` 是有意的——加个 marker 会把它变成**无限重试**（这类卡永远不会成功），
  正确的形状是一个「环境不可用」终态，那需要单独设计，不在本次顺手改。

## 0.1.15

升号原因在 wheel 内，且**这次连随包 runner 一起改了**（`remote_scripts/run_kernel.py` 在
交付身份的 73 个文件里，改它必然换身份）：

- runner 的 `gpu_info` 现在自描述硬件：每 SM 共享内存、块可 opt-in 的字节数、L2 大小、
  每 SM 线程数、寄存器数、warp 宽度，并带 `probed_by`。**读不到就省键**，另给一条
  `hardware_report_missing` 说明——不许改用兜底常量。
- Designer 的 tile 预算（`generate/agents.py` 的 `smem_budget()`）改成
  **实测 > 按 cc 的常量表 > 被显式申报的 legacy 字段 > 抛**。原来最后一级是 `128` KB，
  它对 Ada（100 KB/SM）偏大 28%，而且让「没探测过硬件」在产物里长得像探测过。

**使用者可见的行为变更**：降级梯子的终点（CPU 回退 `LocalExecutor`，它的
`memory.local.size` 是 `"N/A"`）从此**不是一台可生成的 GPU**——Designer 在烧掉 LLM 调用之前
抛 `HardwareProbeFailed`，而不是借给它一个不存在的共享内存池子。

- Python 侧同步 `0.6.6`；工具契约仍为 **9**，MCP 工具面一条命令都没动。
- 两个共享内存上限（每 SM 与每块 opt-in）**只要有一个就够排 tile 预算**：`shared_memory_per_multiprocessor`
  不是每个 torch build 都暴露，要求两个都有等于按 torch 版本判卡。只有 opt-in 时按它取值
  （Hopper 227/228、Ada 99/100，取到的是偏小的那个，方向保守）。
- 同批（都不随包发布）：迁移门采集器 `benchmarks/probes/collect_unknown_card_probe.py` 接上
  `--from-probe`，`measure_launch_floor.py` 与 `calibrate_achievable_peak.py` 尾巴各打一行
  `PROBE_JSON {...}`（带本卡 uuid，取不到就省键）；门与采集器、runner 与 Designer 之间的字段契约
  共 59 条用例（`test_unknown_card_gates` 21 / `test_unknown_card_probe` 19 / `test_smem_budget` 19），
  变异表 `benchmarks/mut_specs/unknown_card_gates.py` 扩到 21 行（行为 18 / 结构 3，全 KILLED）。

## 0.1.14

一行没改插件壳本身，升号原因还是 wheel 内的 Python 代码：`app/cli.py` 新增三条子命令
`operator-agent ledger-new | ledger-fill | ledger-report`（一期 T3.7 人时表的填写入口），
`core/human_ledger.py` 的边界说明随之更新（**入口是 CLI，仍然不接 MCP 工具回包**，那会改工具契约）。

- Python 侧同步 `0.6.5`；工具契约仍为 **9**，MCP 工具面一条命令都没动。
- 同批新增未知卡迁移门探针 `benchmarks/probes/unknown_card_gates.py`（不随包发布），
  阶段 B runbook 的第 0 步与第 6 步现在直接调这两个入口。

## 0.1.13

换号原因仍然只是文档：一期清单、身份重绑记录与阶段 A 台账迁入 `docs/phase1/` 后，随包文字里的
文档路径又变了一次（本 `CHANGELOG.md` 与 `README.md`），且 Python 包根的模块清单 docstring 改成
了分层写法（`operator_agent/__init__.py`，在 wheel 内）。Python 侧同步 `0.6.4`。

- 工具行为、schema、宿主契约未变（工具契约仍为 **9**）。
- 一期文档索引现在在 `docs/phase1/README.md`；阶段 B 的执行前冻结计划在 `docs/phase1/stage-b-runbook-20261101.md`。

## 0.1.12

换号原因只有一件：一期文档搬进 `docs/phase1/` 后，随包的 `README.md` 与本 `CHANGELOG.md` 里引用的
文档路径变了，Python 侧 `human_ledger.py` 的模块说明也变了 —— 这些字节都在发布物里，按 `0.1.10`
立下的「版本号 ↔ 内容一一对应」规矩必须换号，不能沿用 `0.1.11` 的哈希。

- Python 侧同步升到 `0.6.3`；`0.6.2`/`0.1.11` 两个工件与其 SHA256 原样保留，不覆盖、不复用其验收。
- 工具行为、schema、宿主契约均未变（工具契约仍为 **9**，`npm test` 仍 58 条）。
- 文档位置：一期清单与两份记录现在在 `docs/phase1/`，索引见 `docs/phase1/README.md`。

## 0.1.11

换号只因 Python 侧身份变了：`operator_agent/` 从包根平铺分为五层，`protocol_identity()` 的 73 条清单里 70 条路径键
带层、59 个文件内容随 import 重写而变，聚合摘要从 `78993c98…` 变为 `e8c7f2c6…`。npm 壳自身的行为、schema 与宿主契约
均未变更（工具契约仍为 **9**，`npm test` 仍 58 条）。

- 与 `0.6.1` 的关系必须说清：在盘的 `0.6.1` wheel 是平铺布局，仓库 HEAD 的 `0.6.1` 源码是分层布局，
  同号两物正是 `0.1.10` 修掉过的缺陷，因此 Python 侧升 `0.6.2`、壳侧同步升 `0.1.11`。
- 分层前的冻结记录（S5/S6、GPU 中断恢复、发布形态）从此只读、不可当新身份引用，见
  `docs/phase1/delivery-identity-rebind-20261031.md`。

## 0.1.10

首个按公开发布形态打包的版本；工具契约仍为 **9**，工具行为、schema 与指令均未变更。

- 包内随附 `LICENSE` 与 `NOTICE`（Apache-2.0 §4 要求分发时随附许可副本）。此前 `.tgz`
  只声明 `license` 字段、不带许可正文，安装方拿不到条文。
- `private` 由 `true` 改为 `false`，可 `npm publish`；版本号从 `0.1.9` 前进到 `0.1.10`，
  避免「同版本号、不同内容」——0.1.9 那两份本地构建继续用各自 SHA256 区分。
- 两份许可文件是仓库根目录的**逐字节副本**，由 `tests/hygiene/test_packaging.py` 守卫，防止改一处忘另一处。

## 0.1.9

- 工具契约 9：修复入口支持任务墙钟、模型调用和验证额度，并在派发前记录预算；失败调用不退款。
- 逐级保存完成时间、源码/契约身份与已知 API usage 水位；缺失 usage 保持未知。
- S4 专用受控 SSH 模式可校验指定设备及现有 runner，持有的持久会话支持有界清理；无回执或强杀后远端状态仍为未知。

## 0.1.8

- 工具契约 8：资源预检不再把 dtype 字符串、逻辑 tile、累加器或流水参数提升为物理 shared 存储；不明确时记录 unknown 并继续原生检查。
- L1 保存 `resource_precheck` 的状态、证据来源和逐组配置；实际编译仍按原有结果判定。
- 源码与模板剪枝、结构升级门共用有证据的资源边界；混合缓冲区各自计算 dtype/copies，保留既有上限与宽松系数。
- autotune 多配置不拼轴；单 binary 回执不能代表多配置或多 launch。完整边界见仓库 S3 记录。

## 0.1.7

- 工具契约 7：新增 `operator_cancel`、CLI cancel 与 `/op cancel <run_dir> <invocation_id>`。
- 同步 verify/repair/bench 通过受监督工作进程执行，MCP 可在调用期间响应取消通知；CLI Ctrl-C/SIGTERM 清理后返回取消 JSON。
- status 展示实例身份、cancelled 终态、本机清理和远端/服务商未知状态；重复取消和已结束调用保留原结果。
- 取消不退还已有模型/验证预留，不补造成功结果。生产任务跨进程时间统一系统单调时钟，兼容 macOS Python 3.9。

## 0.1.6

命令执行完**有回执**了：首行以 `✓` / `✗` 开头，收起状态下一眼能看出跑成没有。

- **为什么要回执**：宿主折叠摘要只有首行可见（`nowrap` + `ellipsis`），而
  `GenericCommandCard` 对成功和失败只改一个 `data-state` 属性 —— 收起时两者的
  文字外观几乎一样。没有回执的话，「执行了」这件事仍然要用户点开才知道，
  这正是上一版没解决干净的部分。
- **成功**：`✓ 已执行 · 算子判定层 0.6.0 · 后端 local/mock/ssh · 10 个算子有输入布局`
  （53 字符）；`/op status` 是 `✓ 查询完成 · 相位 done（进程已结束）`（23 字符）。
- **失败四类都有 `✗` 且说清类型**：判定层拒绝 → `✗ 判定层拒绝`；命令没装 →
  `✗ 未执行 · not_installed`；回包不是 JSON → `✗ 读数不可用`；缺参数/未知子命令 →
  `✗ 缺参数` / `✗ 未知子命令`。**取消不挂 `✗`**（那是人的正常操作，写 `已取消`）。
- **给首行腾预算**：`tool` 号从首行挪到明细的 `工具版本：tool 1` 行 —— 它是回包契约
  版本，和判定层版本不是一回事，不能合并，但也不该占折叠态的位置。
- **`/op status` 相位不再重复**：首行已经写了相位，展开态不再回声一遍。
- **`/op help` 也带 `✓`**，并说明折叠时只看首行、点它展开。
- **测试从 42 条加到 50 条**：首行 ✓ 开头、长度上限（capabilities ≤ 60、status ≤ 40）、
  长路径不进首行、tool 号不进首行但留在明细、四类失败各自的 `✗`、帮助也带回执。
- **顺带修了三条旧测试的断言落点**：它们用假 CLI 把路径塞进 `phase` 字段，再去断言
  `相位：<路径>`；相位改成只出现在首行后这些断言失去依托，改为断言权威落点
  `目录：<路径>`（含 `host-contract` 那条真宿主测试）。

## 0.1.5

`/op capabilities` 换个首行，修掉「命令看起来没执行」这个误读。

- **根因不在插件，在宿主的折叠摘要**：`/op` 的结果是 `GenericCommandCard` 节点，
  **默认折叠**，折叠时把 `command/done` 的整段文本塞进单行摘要（换行被压平、溢出折行）。
  多行的 capabilities 输出因此在折叠态长得像页眉，用户会以为命令没响应 —— 实际上
  每次 `command/done` 都是 `success`，约 200 ms 返回。
- **首行改为自带摘要**：`算子判定层 0.6.0（tool 1）· 后端 local/mock/ssh ·
  10 个算子有输入布局`（52 字符）。折叠态一行就能读完，明细仍在后面，点开展开。
  没砍任何字段，只调整了首行的信息密度。
- **`/op help` 明说输出形态**：结果显示在对话里、是可折叠节点、不进模型上下文、
  不产生轮次。原来这层语义只在 README 里，用的人看不到。
- **测试从 39 条加到 42 条**：钉住首行含后端与覆盖数、首行 ≤ 60 字符（不铺算子名）、
  明细十个算子一个不少、后端为空时首行不崩也不编造。README §5.5 补了
  「为什么首行必须是摘要」，免得下次改格式化时把这条重新踩掉。

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
  读一遍元数据（去掉任一条 exports 都会红，已用变异验证）；`tests/hygiene/test_packaging.py`
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
