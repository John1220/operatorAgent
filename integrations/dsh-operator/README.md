# dsh-operator-agent

把 [operator-agent](../../README.md) 的**判定层**接成 DeepSeek Harness（DSH）的工具：
判一个算子核能不能编译、对不对、快不快，读数可不可信，并向 DSH 外层提供生成与修复材料。

**它不是什么**：不是新的 agent。判定全部来自 `operator-agent` 既有模块
（`Verifier` / `measurement` / `eligibility` / `anticheat` / `failure_route`），
这个包提供 DSH 侧的安装配置、`/op` 命令与原生主模型预算 hook。

首次试用 v0.7.2，请从[自助试跑指南](../../docs/testing/v072-quickstart.md)开始：先用自然语言描述 CPU ReLU 需求，再按需查看安装、证据和可选验收步骤。正式本地版为 Python **0.7.2** / 插件 **0.2.2** / 工具契约 **12**，未向 PyPI/npm 公开发布。

---

## 1. 前置

- 原生集成测试基线为 DSH `0.2.1-rc.2` / Node `24.14.1`；`engines.dsh` 从该宿主版本声明兼容。DSH NEXT `0.2.1-rc.4` 已有 CPU 用户功能证据，见[自然语言入口复核](../../docs/examples/v072-natural-user-review.md)，不扩大为完整宿主兼容矩阵。
- 默认开发与安装验收使用 Python 3.12；最低兼容版本仍为 3.9，且 `operator-agent` 可被宿主进程导入
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

### 2.2 从本地正式工件安装

本地正式交付为 Python `0.7.2` / 插件 `0.2.2`，工具契约 **12**，尚未发布到包仓库。本机安装目标为 `~/.local/share/operator-agent/v0.7.2/`；实际完成状态以正式安装回执和 UI 核对为准，不沿用候选回执。

先把 wheel 装入 DSH 启动时可访问的 Python 3.12 环境：

```bash
python3.12 -m venv "$HOME/.local/share/operator-agent/v0.7.2/venv"
"$HOME/.local/share/operator-agent/v0.7.2/venv/bin/python" -m pip install "$HOME/.local/share/operator-agent/v0.7.2/artifacts/operator_agent-0.7.2-py3-none-any.whl"
```

也可以让插件自带助手一次完成「环境 + 配置」：

```bash
# 用本地 wheel 建 venv 并安装，再写 profile 配置
npx dsh-operator-agent-setup --wheel /path/to/operator_agent-0.7.2-py3-none-any.whl
# 已经有 venv 时，只做校验与配置
npx dsh-operator-agent-setup --python "$HOME/.local/share/operator-agent/v0.7.2/venv/bin/python"
```

它会起一次真实 MCP 握手核对七个 host 工具，写前备份原 `cordis.patch.yml` 并留回执，重复执行幂等；加 `--dry-run` 只预览。**插件包本身仍要在 GUI 安装**，助手不碰 Desktop profile 的安装锁。

本机装了 [uv](https://docs.astral.sh/uv/) 的话，也可以完全不建 venv——让 uvx 按钉死的版本解析 Python 包：

```bash
npx dsh-operator-agent-setup --runtime uvx
```

它把 profile 写成 `uvx --from dsh-operator-agent==0.7.2 operator-agent-mcp`（以及同版本的 `/op` CLI）。
首次启动会下载 wheel，之后走 uv 缓存；版本与插件配对，不会漂移。没有 uv 时助手会直接报错，
退回 `--python`/`--wheel` 形态即可。

**Desktop 使用 GUI 安装插件**：在 DSH 的 **插件 → 添加插件 → 安装第三方插件** 输入 `/absolute/path/operator-agent/v0.7.2/artifacts/dsh-operator-agent-0.2.2.tgz`，请从本机正式目录复制实际绝对路径替换占位符（GUI 不展开 `~` 或 `$HOME`）。安装后启用，再运行配置助手：

```bash
"$HOME/.local/share/operator-agent/v0.7.2/examples/configure-v072.command"
```

该助手用 `--configure-only` 备份配置、写入 CLI/MCP 两处绝对路径并检查七个 host 工具；追加 `--dry-run` 可预览。Desktop profile 由 Electron 独占，普通 CLI 不能安装或导出，退出应用也不能解除。不要将下方自定义 profile 的命令用于 desktop。

自定义非 desktop profile 可以通过[安装助手](../../benchmarks/delivery/install_dsh.py)的 `--profile <profile>` 一次安装并配置，或使用普通 CLI：

```bash
dsh plugin --profile <非desktop的profile> add /path/to/dsh-operator-agent-0.2.2.tgz
```

GUI 通常不继承 venv 的 `PATH`，因此 `mcp-operator` 与 `operator-tools` 均需使用相应 Python 环境中的绝对命令路径。配置助手检查不代替实际加载：重开 DSH，确认插件页 `0.2.2` 的两个组件运行中，并用 `/op doctor` 核对（CLI/MCP 是否可执行、工具契约与版本配对、已接线后端；失败项直接给修复命令）。卸载见 §2.4。

当前边界见 [v0.7.2 支持清单](../../docs/testing/v072-support.md)。历史候选的[生命周期验收](../../docs/status/v072-session-lifecycle-acceptance-20261106.md)与[自动入口验收](../../docs/status/v072-auto-task-acceptance-20261106.md)保留原版本和安装身份。前版边界与回退见 [v0.7.1 支持清单](../../docs/testing/v071-support.md)及[交付记录 §12](../../docs/testing/v07-delivery.md#12-v071-本地封版)。legacy 离线示例随包位于 `examples/v07/`。旧 [S6 验收记录](../../docs/status/dsh-operator-s6-acceptance-20261030.md)仅覆盖当时工件；其 NumPy wheel 面向 CPython 3.9，Python 3.12 应使用匹配的平台依赖。

### 2.3 DSH 侧

```bash
# 自定义非 desktop profile：本地源码安装（不需要发布 npm）
dsh plugin --profile <非desktop的profile> add /path/to/operatorAgent/integrations/dsh-operator
```

> **`desktop` profile 由 Electron 独占管理**：普通 CLI 会直接拒绝管理它，退出 Desktop 也不能解除。
> 请在 DSH 的 **插件 → 添加插件 → 安装第三方插件** 输入本地包路径，安装后启用。
> 当前本机的配置助手可自动备份并配置 Python 路径，见[自助试跑指南](../../docs/testing/v072-quickstart.md)。

本包的 `cordis.patch.yml` 会插**两条**记录：`mcp-operator`（host 模式七个工具 + server 的
`instructions`）与 `operator-tools`（斜杠命令 `/op` 和原生模型预算 hook；给模型的提示段不在这里，
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
| `dsh plugin --profile <profile> remove dsh-operator-agent` | 用于自定义非 desktop profile；同时摘掉 profile 的依赖与 bundle。desktop 请在 GUI 插件页卸载 |
| 纯手工 | 从 profile（或你放这两行的那个 patch）里删掉 `mcp-operator` 与 `operator-tools` 两条，再把名字从 `dsh.profile.bundles` 去掉 |

两半是**独立**的：卸载插件不影响 Python 侧的 `operator-agent` CLI；反过来
`pip uninstall operator-agent` 会让 MCP 工具变成「命令不存在」——`/op` 会明说
找不到命令，宿主日志里记一条 `operator-tools: 找不到命令 …`（error 级，同类只记一次）。

## 3. 模型调用模式：默认 host 与 legacy 配置

**v0.7.2 组合包默认由 DSH 外层驱动。** MCP 环境设置 `OPERATOR_AGENT_WORKFLOW: host`，工作流为 `operator_prepare → operator_context → DSH 写代码 → operator_verify → DSH 修复 → final verify`。插件提供领域事实与指导，DSH 负责最终上下文、模型选择和迭代；此模式不需要 Python 的模型凭据或 `modelRelay`。验证预算跨调用持久化，并预留一次同源码的独立终评。默认 `operator-tools.config.hostModelBudget: {}` 启用主模型发送前硬门：prepare/resume 成功绑定后按任务预留模型次数及输出 token，重启不补额度。详见 §5 和[外层工作流说明](../../docs/design/dsh-host-context-workflow.md)。

以下配置针对 v0.7.1 和兼容的内层模式。新版本需要在 MCP 环境显式设置 `OPERATOR_AGENT_WORKFLOW: legacy`，并在 `operator-tools.config` 设置 `hostModelBudget: false`；保留 MCP 的完整连接字段，才会按 legacy 模式使用 `operator_run`、`operator_repair` 和 `operator_bench`。两项必须一起修改，避免把 legacy 回执当作 host 任务绑定。

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

### 3.1 v0.7.1 · 通过 DSH 发出内层模型请求

v0.7.1 当时使用 Python `0.7.1` 和 DSH 插件 `0.2.1`；此处保留该 legacy 入口说明。兼容的 DSH 可以为 `operator_run` 和 `operator_repair` 发出内层模型请求。Python 继续组装角色消息、推进生成或修复循环，并在任务账本中记录请求和用量。DSH 使用指定的宿主模型路由及该路由已有的凭据。这个入口默认关闭，独立 CLI 和现有 MCP 配置可继续使用各自的模型客户端。

在本机可信的隔离 profile 中，先生成单独的转发凭据。使用一个自己拥有、权限为 `0700` 的目录，并为每个 profile 使用单独的回环端口和凭据文件。下面的命令会创建新目录和权限为 `0600` 的文件。

```sh
relay_dir="$(mktemp -d)"
(umask 077; node -e 'process.stdout.write(require("node:crypto").randomBytes(32).toString("base64url") + "\n")' > "$relay_dir/token")
printf '%s\n' "$relay_dir/token"
```

安装插件后，在该 profile 的覆盖配置中填入实际路径、空闲端口，以及 DSH 已配置的一组 provider/model。DSH 会整体替换每个条目的 `config`，因此需要保留 MCP 的连接字段。

```yaml
- id: operator-tools
  config:
    command: /absolute/path/to/venv/bin/operator-agent
    hostModelBudget: false
    modelRelay:
      provider: YOUR_DSH_PROVIDER
      model: YOUR_DSH_MODEL
      port: 43718
      tokenFile: /absolute/path/to/private-relay-dir/token
- id: mcp-operator
  config:
    serverName: operator
    transport: stdio
    command: /absolute/path/to/venv/bin/operator-agent-mcp
    toolCallTimeoutMs: 3600000
    failOnStartupError: true
    env:
      OPERATOR_AGENT_WORKFLOW: legacy
      LLM_BASE_URL: http://127.0.0.1:43718/v1
      LLM_API_KEY_FILE: /absolute/path/to/private-relay-dir/token
      LLM_MODEL: YOUR_DSH_MODEL
      LLM_FALLBACK_MODEL: ""
      LLM_THINKING_PARAMETER: enable_thinking
```

转发凭据只供同一 profile 的 Python 客户端使用，服务商密钥保留在 DSH。转发服务只监听本机 IPv4 回环地址，接受指定模型的非流式文本请求，并随 profile 卸载。调用超时或连接断开时会请求中止宿主调用。Python 继续按既有账本扣减已派发调用，服务商是否已经停止以可观测证据为准。缺失的服务商用量保持未知。当前路由在 profile 配置中固定，宿主主会话切换模型时可单独调整这条路由。恢复直连时删除 `modelRelay`，把 MCP 的 `env` 恢复为原模型配置并重新加载 profile。

宿主路由还需要正确的服务商兼容配置。Python 的 `LLM_THINKING_PARAMETER` 只控制本地转发请求；发往服务商的字段由 DSH 适配器决定。本轮阿里云 OpenAI Completions 路由要求顶层 `enable_thinking: false` 和 `max_tokens`，在隔离 profile 的对应模型条目中使用以下兼容字段，已核对实际出站请求。保留该条目的其他字段及 provider 中的其他模型；其他服务按其实际协议配置。

```yaml
compat:
  thinkingFormat: qwen
  maxTokensField: max_tokens
```

这里的 `qwen` 是 DSH 适配器发送顶层思考开关的格式名称。仅声明 `reasoningEfforts` 支持 `off`，在本轮原配置中没有发出关闭字段，曾导致 Designer 输出截断。截断结果会被拒绝并保留已消耗的调用预算。

## 4. 验证装好了

```bash
"$HOME/.local/share/operator-agent/v0.7.2/venv/bin/operator-agent-mcp" --selftest
# 下条只用于自定义非 desktop profile
dsh --profile <非desktop的profile> --dump-config | rg mcp-operator
```

Desktop 重开后应在插件页核对插件 `0.2.2` 启用且两个组件运行中，再直接 `/op capabilities` 核对 Python `0.7.2` / 工具契约 `12`。在会话中让模型调用 `operator_capabilities` 可进一步核对 MCP 链路；`backends.{mock,local,ssh}.wired` 只说明接线，不证明已通过 GPU/SSH 验收。

## 5. 工具与工作流

host 模式提供七个工具：`operator_capabilities`、`operator_prepare`、`operator_context`、`operator_verify`、`operator_resume`、`operator_status`、`operator_cancel`。插件不组装最终 messages 或调用内层模型。

`operator_prepare.host_budget` 的默认值为：

```json
{
  "wall_clock_s": 600,
  "max_verifications": 6,
  "max_model_calls": 12,
  "max_output_tokens_per_call": 8192,
  "total_output_token_reservations": 98304
}
```

这些值在 prepare 时冻结。普通 verify 消耗探索额度，固定保留一次同源码独立终评；完整 L2 通过后设置 `verification_purpose: final` 再验证，随后开发关闭。`operator_context` 通过 `phase: generate/repair/optimize` 按需返回材料，repair/optimize 还需 `verification_round`。性能观测无资格时不因存在 optimize 材料就继续调参。

### 5.0 host 模型预算、恢复与取材

默认组合包同时启用 `hostModelBudget: {}`。此配置只接受空对象，限额由 Python 的冻结 policy 提供；不要在 profile 里另配一套额度。原生 hook 在绑定后的主模型请求到达 provider 前预留一次调用和该次 `maxTokens`，并限制单次输出。失败、取消、输出少于预留或 usage 缺失均不退款；usage 缺失保持 unknown。它约束的是**成功 prepare/resume 绑定后的主模型派发**，不包括绑定前准备、标题等辅助模型和 legacy 转发，不是全会话或精确账单上限。final 后的模型汇总继续计入原任务。

本版支持一个主会话同时绑定一个活动任务；上一项结束后，用户可在同一会话直接提出下一项自然语言需求，无需 `/op new`。绑定后允许上述七个算子工具，拒绝 shell、子代理、PTC 和其他工具路径；fork/委托会话不支持 host 开发。请在 prepare 前取得需要的项目文件与背景；DSH 已有会话背景和用户后续消息仍参与外层上下文。绑定后通过 `operator_context` 取得当前任务/阶段材料，验证回包和 `operator_status` / `operator_resume` 提供候选与证据路径。当前没有通用的绑定后文件读取工具，超出这套取材方式的工作流需另行支持；这项白名单是本版预算支持边界。

同一 DSH session 进程重启时，从本会话原生用户轮次、prepare/resume 回执、可选的手动切换命令及 turn/end 恢复阶段和绑定，在首个后续主模型请求前核对原账本。也可调用 `operator_resume(run_dir, backend)` 读取现有任务的 `next_action`、`next_context`、最近验证与最后完整 L2 候选；恢复不重新 prepare，不改变 task_id、契约、deadline 或已花额度。模型 owner 无法确认已退出时保留不确定状态并拒绝继续；确认旧进程退出后可将遗留请求记为 abandoned，额度仍保留。旧 policy 没有模型字段时不会自动补额度，损坏或身份不符的账本会拒绝绑定。改变 shape、精度等冻结判据需要明确的新任务，可在同一主会话的新用户轮次顺序开启；不能靠换目录给原任务补额度。

验证终评预留不保证还剩足够的模型调用来发起终评或汇总，模型用完时必须如实停止。用户可直接运行 `/op status <run_dir>` 查看预算、候选源码与报告路径、在飞 invocation ID，再用 `/op cancel <run_dir> <invocation_id>` 取消验证；这两条命令不调用模型，即使模型额度已耗尽仍可用。DSH 会话取消由宿主处理；本地 abort 不证明服务商已停止执行或计费。

以下为 legacy 模式的原工具说明；该模式另提供 `operator_context`，但不自动启用跨调用的 host 验证预算。

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
/op status                查看当前会话阶段
/op status <run_dir>      查看任务状态、预算、候选与证据
/op new [new_run_dir]     可选手动新任务入口（日常直接描述新需求即可）
/op cancel <run_dir> <invocation_id>  按实例取消验证，再查 status 确认清理
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

host 任务的 status 还显示验证与模型预算、最后完整 L2 候选或最近失败验证的源码/报告路径，以及在飞工具实例。`host_enforced_receipt` 表示 Python 已核对宿主账本收据，不单凭该收据证明当前 hook 正在运行或服务商执行已经停止。`not_bound`、`receipt_invalid` 和旧 policy 状态会明确显示，不作为有额度的证明。

需要开发决策和工具验证时用模型工具（`mcp__operator__*`）；旁路命令供用户直接查询、取回路径与取消。

### 输出长什么样（以及为什么首行必须是回执）

`/op` 的结果在界面里是一个**默认折叠**的节点（宿主 `ui-chat` 的 `GenericCommandCard`），
不是一条助手回复 —— 它不产生轮次，也不会有「已完成」标记。折叠摘要的样式是
`white-space: nowrap; text-overflow: ellipsis`，也就是**只有首行可见，放不下的直接截掉**。

所以插件把**回执**放在首行：

| 情形 | 首行 |
|---|---|
| `/op capabilities` | `✓ 已执行 · 算子判定层 0.7.2 · 后端 local/mock/ssh · 10 个算子有输入布局` |
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
> `mcp:operator` 提示段。只有一处来源，换任何 MCP 客户端也都在。JS 插件另负责 `/op` 与 DSH 原生预算 hook；其他 MCP 客户端不能仅凭 Python policy 声称具备 DSH 主模型硬门。

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
npm test        # 注册/命令/格式、模型预算、生命周期与原生宿主集成
```

`npm test` 里有两层守卫，别只跑一层：

| 文件 | 钉住什么 | 依赖 |
|---|---|---|
| `test/plugin.test.mjs` | 注册面、`/op` 行为、路径解析、超时/取消、`scrubEnv`/`normalizeConfig`、宿主日志（同类只记一次）、两个 formatter 吃**真 CLI fixture** | 无（替身 ctx + 假 CLI） |
| `test/host-contract.test.mjs` | 把插件注册进**装在机器上的** DSH 自己的 `cordis` + `dsh-commands` 服务真执行（含 `agent` 透传），并用真 `readPluginMeta` 读一遍「脸」、用真 `evaluatePluginCompatibility` 核 `engines.dsh`；最后一条用**真 MCP SDK 客户端**连真 server 验 `instructions` 到达客户端 | DSH 安装；最后一条还需要能 import 判定层的 python3。找不到就**跳过**（`DSH_APP_ROOT` / `OPERATOR_AGENT_PYTHON`） |

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

当前正式本地交付身份为 Python **0.7.2** / 插件 **0.2.2** / 工具契约 **12**，未执行 PyPI/npm 公开发布。正式工件从保留源码快照构建并核对包内身份；本机安装目标为 `~/.local/share/operator-agent/v0.7.2/`，安装成功需另有回执及 UI 版本核对。

历史 `0.1.16` 插件 / Python `0.6.7` 的换号原因仍见[身份重绑定记录](../../docs/phase1/delivery-identity-rebind-20261031.md)，不作为当前版本说明。`private: false` 及随包 `LICENSE`/`NOTICE` 说明发布形态，不代表已上架。逐字节副本与发版形态由
`tests/hygiene/test_packaging.py::TestTheDshReleaseShape` 和 CI 的 `release-shape` 作业守住
（`.github/workflows/ci.yml`）。若将来把 `operator-dev-agent-design/` 的静态资源打进
发布物，记得把 `THIRD_PARTY_NOTICES.md` 也列进 `files`。

## 9. License

**Apache-2.0**，见仓库根目录的 [`LICENSE`](../../LICENSE) 与 [`NOTICE`](../../NOTICE)
（版权人：`the operator-agent authors`）。npm 包里各带一份逐字节副本
（[LICENSE](LICENSE)、[NOTICE](NOTICE)），因为安装方拿到的只有包内文件。
第三方组件（ECharts / Mermaid）保留各自许可，见 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)。

## v0.7.2 自然语言任务入口

直接发送“帮我实现一个 FP32 ReLU，输入 4×8，在本机 CPU 验证正确性并给出源码，暂不测性能”即可；无需指定工具、预算、目录或 `/op new`。已有任务结束后，新用户轮次的独立需求可在同一主会话自动开启全新目录。解释结果或查询状态保持讨论；含糊的“继续”应先澄清。意图由 DSH 模型判断，插件执行轮次、目录、账本和清理硬门，每轮最多成功开启一项任务，不能同轮自动循环建目录补额度。真实用户证据见[自然语言入口复核](../../docs/examples/v072-natural-user-review.md)，保留 rc3/rc.6 的历史身份。

独立终评后原轮总结仍计入任务预算，原轮结束后进入结果讨论。预算耗尽或到期的原轮停止，之后可继续讨论；普通讨论使用 DSH 模型正常计费，旧任务不能继续执行。新任务不改变旧任务的截止、消耗和证据。

`/op new [新的绝对目录]` 保留为用户直接输入的手动入口，不调用模型；命令后下一次 prepare 才创建新契约和预算。`/op status` 无参数查看会话阶段，带目录查看相应任务。存在未结算模型请求、在飞验证或未知清理状态时禁止切换，需等待结算或 `/op cancel` 并确认终态。重启从本会话原生事件和账本恢复阶段，已结束任务不会因旧 prepare 回执重新占用会话。
