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

### 2.2 DSH 侧

```bash
# 本地目录安装（不需要发布 npm）
dsh plugin --profile <profile> add /path/to/operatorAgent/integrations/dsh-operator
```

> **`desktop` 这个 profile 有前置条件**：DSH Desktop 自己拥有它，CLI 会要求你先
> **完全退出 Desktop** 再跑（否则直接拒绝）。更顺的是在 GUI 里装：侧边栏的
> **插件**页，或让模型用 `plugin_manager` 工具——两者都不需要退出。

本包的 `cordis.patch.yml` 会插**两条**记录：`mcp-operator`（六个工具）与
`operator-tools`（斜杠命令 `/op` + 一段 system prompt）。手工安装时两条都要复制。

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

### 2.3 卸载 / 禁用

三种手段，效果不一样，别混：

| 手段 | 效果 |
|---|---|
| **GUI 侧边栏 插件 页**：关开关（禁用）/ 删除 | 推荐。删除会同时摘掉 profile 的 `dependencies` 与 `dsh.profile.bundles`；禁用只是让 bundle 不生效（patch 不再插入），文件还在 |
| `dsh plugin --profile <profile> remove dsh-operator-agent` | ⚠️ 这条是**透传给 pnpm** 的（`dsh plugin --profile <name> <pnpm-args...>`），它只摘 `dependencies`，**不会**改 `dsh.profile.bundles`。所以要再手工把名字从 profile `package.json` 的 `dsh.profile.bundles` 里删掉，否则启动时会报 `skipping profile bundle …`。`desktop` profile 还要求先完全退出 Desktop |
| 纯手工 | 从 profile（或你放这两行的那个 patch）里删掉 `mcp-operator` 与 `operator-tools` 两条，再把名字从 `dsh.profile.bundles` 去掉 |

两半是**独立**的：卸载插件不影响 Python 侧的 `operator-agent` CLI；反过来
`pip uninstall operator-agent` 会让六个 MCP 工具全部变成「命令不存在」——`/op` 会明说
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
```

也可以用 `LLM_API_KEY` 直接给值（不推荐）。没配时这两条工具会立刻返回
`llm_not_configured`，**不会**起一个跑不出结论的进程。

> 为什么必须写在 `env:` 里：DSH 会先洗掉子进程环境里名字**含 `KEY`/`PASSWORD`/`SECRET`/`TOKEN`**
> 的变量（`/KEY|PASSWORD|SECRET|TOKEN/i`），**外加所有 `DSH_*`**，再应用显式覆盖。
> `LLM_API_KEY_FILE` 含 `KEY`，靠全局环境变量传不进去；`DSH_HOME` 之类也一样传不进去。

## 4. 验证装好了

```bash
operator-agent-mcp --selftest     # 不依赖宿主，走一遍握手
dsh --profile <profile> --dump-config | grep mcp-operator
```

然后在会话里让模型调一次 `operator_capabilities`：返回 `backends.{mock,local,ssh}.wired`
就说明链路通了。

## 5. 六个工具

| 工具 | 一句话 | 要 LLM |
|---|---|---|
| `operator_capabilities` | 这台机器能做什么（只读，不联网、不探测 SSH） | 否 |
| `operator_prepare` | 声明 → 冻结契约 + 契约自带的输入 + 回合身份 | 否 |
| `operator_verify` | 对**外层给的**源码判 L1/L2/L3，返回正交事实 + 处方 + 资格 | 否 |
| `operator_repair` | 给一份**已有的**核：判 → 处方 → 内层模型修 → 再判（≤N 轮） | 是 |
| `operator_run` | 起一次**无人值守**整段（本仓自己设计/写核/修），立刻返回 pid | 是 |
| `operator_status` | 查无人值守的相位：`running` / `done` / `failed` / **`lost`** | 否 |

**耗时与超时**：真机上一次**完整**判定（L1→L3、L2 的 15 个边界 case）实测 **866 s**，所以
`cordis.patch.yml` 里 `toolCallTimeoutMs` 设成 **3600 s**（60 min：单次 866 s，`repair` 默认
2 轮约 30 min）。**别往回调小**——超时不是优雅退出，server 会继续算完再写一个没人认领的响应；
更长的工作请用 `operator_run` + `operator_status` 轮询（`run` 立刻返回 pid，不占一次调用）。

## 5.5 斜杠命令 `/op`

命令在**宿主进程**里执行，`/op status` 的输出直接给人看、不进模型上下文：

```
/op capabilities          这台机器能做什么（只读，不跑活）
/op status <run_dir>      查无人值守的相位（running/done/failed/lost）
/op help                  用法
```

需要「判定/修复/从头自动化」时用模型工具（`mcp__operator__*`）：命令只是旁路，
不替模型做判定。

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
- **无人值守运行**没有取消动作（只能按 pid 杀；状态会停在 `running`，下一次查询判 `lost`）。
  （`/op` 命令本身可以被宿主取消——那是宿主侧的子进程，与判定层的运行不是一回事。）
- 没有 UI/曲线面板（那一格由其他插件占位；本包的差异化在判定层）。
- 资源预检的元素宽度目前从源码推断（fp16 任务的 `256×256` tile 可能被误判为超限）。

## 8. 构建与发版（维护者）

```bash
npm test        # 26 条：注册面/行为/格式化契约（真 CLI fixture）+ 真宿主契约
```

`npm test` 里有两层守卫，别只跑一层：

| 文件 | 钉住什么 | 依赖 |
|---|---|---|
| `test/plugin.test.mjs` | 注册面、`/op` 行为、超时/取消、`config.env`、宿主日志（同类只记一次）、两个 formatter 吃**真 CLI fixture** | 无（替身 ctx + 假 CLI） |
| `test/host-contract.test.mjs` | 把插件注册进**装在机器上的** DSH 自己的 `cordis` + `dsh-commands` 服务真执行，并用真 `readPluginMeta` 读一遍「脸」 | DSH 安装；找不到就**跳过**（`DSH_APP_ROOT` 可指定） |

第二层是必要的：替身 ctx 只会照单全收，而宿主的 `normalizeDefinition` / `normalizeResult`
会拒绝不合规的命令定义与返回值；元数据那几条更只认**真模块解析器 + `exports` 映射**。

**给插件配「脸」的三个资源**（宿主 `readPluginMeta` 读的就是这三样，规则已在
`tests/test_packaging.py::TestTheDshPluginFace` 里镜像一份）：

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

发布顺序：Python 包（PyPI）→ npm 壳（把 `private` 改成 `false`）→ README 里补一行
`dsh plugin add dsh-operator-agent`。许可已定为 Apache-2.0；若将来把
`operator-dev-agent-design/` 的静态资源打进发布物，记得把 `THIRD_PARTY_NOTICES.md`
也列进 `files`。

## 9. License

**Apache-2.0**，见仓库根目录的 [`LICENSE`](../../LICENSE) 与 [`NOTICE`](../../NOTICE)
（版权人：`the operator-agent authors`）。
第三方组件（ECharts / Mermaid）保留各自许可，见 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)。
