# v0.7 候选版：无需租卡的安装与交付

本说明针对本地候选工件，**v0.7 尚未正式发布**。产品版本、Python wheel 和 DSH 插件包版本分别记录，以交付目录里的身份清单与 SHA256 为准。下面的离线流程使用脚本模型及 CPU FakeBackend，不需要模型账户、SSH 或 GPU。

## 1. 在干净 Python 3.12 环境安装

取得本次交付目录中的 `operator_agent-*.whl`、`dsh-operator-agent-*.tgz`、身份/哈希清单及 `examples/v07/`。从仓库外运行，不使用源码目录上的 `PYTHONPATH`。若目录中保存了多个版本，先按清单选择唯一工件，不能将多份 wheel 同时安装。

```sh
python3.12 -m venv /absolute/path/to/v07-venv
/absolute/path/to/v07-venv/bin/python -m pip install /absolute/path/to/operator_agent-VERSION-py3-none-any.whl
/absolute/path/to/v07-venv/bin/python -m pip check
/absolute/path/to/v07-venv/bin/operator-agent capabilities
```

wheel 依赖 NumPy，正常安装会由包管理器解析并下载本机适用版本。若交付目录含本平台的 `wheelhouse/`，可用 `pip install --no-index --find-links /absolute/path/to/wheelhouse <wheel>`；不能跨 Python ABI、操作系统或 CPU 架构复用 NumPy wheel。控制端离线检查无需安装 Torch/Triton；目标 GPU 运行时另行验收。

`capabilities` 应报告当前 Python 为 3.12、`backends.fake.wired=true`、`backends.fake.test_only=true`。能力发现为只读，不能据此认为真实 GPU 可用。

## 2. CLI 与 MCP 离线完整流程

示例 `offline_smoke.py` 只用标准库驱动已安装包。它从独立工作目录运行 `python -I -m operator_agent` 或 MCP stdio server，明确固定 `backend=fake` 和 `fake.llm=fake`。每次选择一个不存在的新输出目录：

```sh
/absolute/path/to/v07-venv/bin/python /absolute/path/to/examples/v07/offline_smoke.py \
  --entry cli --scenario success --output /absolute/path/to/new-cli-run

/absolute/path/to/v07-venv/bin/python /absolute/path/to/examples/v07/offline_smoke.py \
  --entry mcp --scenario correctness-repair --output /absolute/path/to/new-mcp-repair

/absolute/path/to/v07-venv/bin/python /absolute/path/to/examples/v07/offline_smoke.py \
  --entry mcp --scenario cancel --output /absolute/path/to/new-mcp-cancel
```

成功及修复示例的预期终态是 `done`、L1/L2 通过、L3 不通过、`all_passed=false`。取消示例会等待 fake 编译进入阻塞后按本次 `invocation_id` 请求取消，再查询到 `cancelled`。MCP 示例保留 `initialize`、工具发现、调用和响应回执；CLI 示例保留原始 stdout/stderr 与退出码。

`export/` 包含独立复制的 `code.py`、`report.md`、`status.json` 和内容哈希清单；取消任务可无终稿，只导出状态和清单。原始运行证据仍位于同一次输出目录的 `run/`，导出目录可独立查看。报告应保留 `FAKE / test_only` 标识；这些结果不能作为真实 GPU 编译、数值或性能资格。

成功任务可将导出的终稿交给另一份新验证目录，检查独立终评入口。声明应与原任务一致；此调用只验证，不生成源码、不调用模型：

```sh
/absolute/path/to/v07-venv/bin/operator-agent prepare \
  --request relu --dsl python --shapes 4,8 --run-dir /absolute/path/to/new-final-check
/absolute/path/to/v07-venv/bin/operator-agent verify \
  --run-dir /absolute/path/to/new-final-check --backend fake \
  --code-file /absolute/path/to/new-cli-run/export/code.py \
  --json-out /absolute/path/to/new-final-check/final-cli.json
```

仍应核对 L1/L2 通过、L3 不具备资格和 `test_only`；不能将重新验证解释为真实 GPU 交付通过。

脚本模型目前只覆盖 ReLU、Matmul。示例采用 ReLU 完整 L2；显式 `correctness-repair` 检查失败候选→新源码→重新验证。`operator_repair` 工具本身仍需要真实内层模型，不是这里的零模型入口。通过 `operator_run` 的 fake 故障场景检查修复循环不会发 API 请求。

## 3. 安装到隔离 DSH profile

需要当前插件声明兼容的 DSH 和 Node 版本。先用独立 `DSH_HOME` 与新 profile 完成检查；命令中的路径需替换成真实绝对路径，不操作个人 `desktop` profile：

```sh
DSH_HOME=/absolute/path/to/new-dsh-home dsh plugin --profile v07-check add /absolute/path/to/dsh-operator-agent-VERSION.tgz
DSH_HOME=/absolute/path/to/new-dsh-home dsh --profile v07-check --dump-config
```

合成配置应包含 `mcp-operator` 和 `operator-tools`。前者连接 MCP，后者提供 `/op`；只有一条配置不足以完成入口检查。GUI 启动的 DSH 常不继承终端 venv 的 PATH，需用 `examples/v07/profile-absolute-paths.patch.yml` 中的形态覆盖两个命令，分别指向干净 venv 的 `operator-agent-mcp`、`operator-agent`，并重新加载该 profile。配置中不需要模型密钥或 `PYTHONPATH`。

进入该 profile 后执行 `/op capabilities`，应显示已安装包的版本与 Python 3.12；已有运行可用 `/op status /absolute/path/to/run` 查询。DSH 侧工具名为 `mcp__operator__operator_run`、`mcp__operator__operator_status`、`mcp__operator__operator_cancel`。

提供给 `operator_run` 的离线参数为：

```json
{
  "request": "relu",
  "shapes": [[4, 8]],
  "dsl": "python",
  "backend": "fake",
  "fake": {"scenario": "correctness-repair", "profile": "small", "llm": "fake"},
  "max_rounds": 2,
  "wall_clock_s": 30,
  "max_llm_calls": 8,
  "oracle_total": 20,
  "run_dir": "/absolute/path/to/new-dsh-fake-run"
}
```

再以返回的 `run_dir`、`invocation_id` 查询 `operator_status` 到终态。直接 stdio 示例无需外层模型；若通过 DSH 的 Agent 自动调用工具，外层模型依然需要另外配置，可能产生费用。交付验收使用的本地假 HTTP 服务属于测试装置，不作为产品内置模型提供。

## 4. 已知边界

| 项目 | 当前口径 |
|---|---|
| 本地安装与 fake 流程 | 验证包可导入、工具接线、预算/取消、归档与导出 |
| GPU 实例 | 每次租赁后重新检测并固定 index/UUID/name，核对运行时及探针；本流程不提供 GPU 验收 |
| 性能 | Fake L3 始终不具备验收资格，不能将固定耗时视为真实性能 |
| 支持清单 | 四算子 FP16/FP32 是候选范围，实际实例组合待真机收口 |
| 正式对照 | 模型采样可比性、四任务八格、同卡独立终评另行冻结 |
| Python 兼容 | 开发默认 3.12；最低版本及 CI 兼容范围以包元数据为准 |
| 候选代码 | 进程约束不等于对敌意代码的安全隔离 |

## 5. 回退与卸载

保留旧版 wheel、tgz 和哈希清单，回退优先新建另一份 venv/profile 安装旧工件，再将目标 profile 的两条绝对命令切回旧环境。这样当前候选环境、历史运行和导出结果仍可保留用于复核。

```sh
DSH_HOME=/absolute/path/to/new-dsh-home dsh plugin --profile v07-check remove dsh-operator-agent
/absolute/path/to/v07-venv/bin/python -m pip uninstall operator-agent
```

移除后核对 profile 的 dependencies 与 bundle 注册均无 `dsh-operator-agent`。卸载 Python 包与 DSH 插件是独立操作；卸载不应删除已有的运行/导出证据。若使用临时 venv 与 profile，可在确认其中没有需要保留的结果后删除整个临时目录。

## 6. 交付验收记录

2026-11-03 在 macOS（arm64）上实做了一遍。下面是**实测读数**，不是模板。

> **`examples/v07/` 在哪**：它在**插件包里**（`integrations/dsh-operator/examples/v07/`），
> 并随 `npm pack` 进 tgz（解包后是 `package/examples/v07/`）。本文 §1–§4 写的
> `examples/v07/...` 指的就是这一份 —— 交付时把 tgz 解开就能用，不需要另找。
> 本节的三个场景**全部用发货包里那份**跑的（`package/examples/v07/offline_smoke.py`）。

### 6.1 工件身份

| 件 | 版本 | 大小 | SHA256 |
|---|---|---|---|
| `operator_agent-0.7.0rc1-py3-none-any.whl` | 0.7.0rc1 | 965.3 KB | `aa3cc8c3064fafce5d5affb01bbe48e34824b4afa697cf10622e0f16fe11826d` |
| `dsh-operator-agent-0.2.0-rc.1.tgz` | 0.2.0-rc.1 | 37.0 KB | `3813dbc5dfda284a790a62ab53379ddbd0412833584428763ca4e7938744e72c` |

环境：Python **3.12.16**（`python3.12 -m venv`）、pip 解析到 **numpy 2.5.4**、npm **11.11.0**、pnpm **11.10.0**。
宿主侧**有两套 dsh，别混**：PATH 上的 `dsh` 是 npm 全局的 `@deepseek-ai/dsh@0.1.1-rc.2`（2025-08-29 装的老 CLI）；用户实际用的是**桌面端 DSH NEXT 2.0.21-next**，其 bundle 声明 `@deepseek-ai/dsh@0.2.1-rc.2`（构建工作树名 `adapt-dsh-v0.2.1-rc.4`）。插件要求 `engines.dsh ^0.2.1-rc.2` ⇒ **桌面端满足**，老 CLI 不满足。

⚠️ **两处与本说明的偏差，必须记下来**：

1. **构建方式不是 `uv build`**：本机到 `pypi.org` 的 TLS 握手失败（`error sending request for url https://pypi.org/simple/setuptools/`），`uv build` 的隔离构建拉不到 setuptools。改用 `python -m pip wheel . --no-build-isolation --no-deps`（本地 setuptools 61.2.0）。wheel 内容不受影响，但**「干净机器上按本文构建」这一步没有验证**。
2. **§6.4 的 profile 装的是"老 CLI"，不是桌面端**。本轮 `dsh plugin add` 用的是 PATH 上那个 `@deepseek-ai/dsh@0.1.1-rc.2`（并配了独立 `DSH_HOME`，没碰 `~/.dsh` / `~/.dsh-beta`）。它**不满足**插件要求的 `^0.2.1-rc.2`，所以那一步证明的是"配置形态与 MCP 通道正确"，**不是**"在用户真实宿主上装好了"。桌面端（2.0.21-next，内嵌 `@deepseek-ai/dsh@0.2.1-rc.2`）上的安装与 `/op` 验收仍未做 —— 那会写进用户的活动 profile，需先经同意。

### 6.2 干净安装

```
$ python3.12 -m venv .../v07-venv
$ .../v07-venv/bin/python -m pip install operator_agent-0.7.0rc1-py3-none-any.whl
    → Successfully installed numpy-2.5.4 operator-agent-0.7.0rc1
$ .../v07-venv/bin/python -m pip check
    → No broken requirements found.
$ .../v07-venv/bin/operator-agent capabilities
    version=0.7.0rc1  tool_version=10  python=3.12.16
    fake wired=True test_only=True / local mock ssh wired=True
$ cd /tmp && .../v07-venv/bin/python -I -c "import operator_agent; print(operator_agent.__file__)"
    → .../v07-venv/lib/python3.12/site-packages/operator_agent/__init__.py
```

最后一行是关键：`-I` 隔离模式下解析到 site-packages，说明驱动的是**装好的包**，不是源码树。

### 6.3 CLI / MCP 示例（三个场景全过）

| 场景 | 入口 | 终态 | `summary.levels` | `summary.all_passed` | 导出 |
|---|---|---|---|---|---|
| success | cli | `done` | L1✓ L2✓ L3✗ | false | code.py / report.md / status.json / manifest.json |
| correctness-repair | mcp | `done` | L1✓ L2✓ L3✗ | false | 同上 |
| cancel | mcp | **`cancelled`** | — | false | **只 manifest.json + status.json**（无终稿，与本文一致） |

`summary.validity` 三次都是 `uninformative`（fake 没有 launch 地板），`test_only: true`、`gpu_acceptance: false`。
取消场景：借 fake 的 `timeout`（编译阶段阻塞）造出可取消窗口 → 按本次 `invocation_id` 调 `operator_cancel` → 轮询到 `cancelled`。
MCP 回执逐个留档：`rpc-001…010.json`（每个都带 `request` 与 `reply` 两半）+ `mcp-stderr.log`；实测序列为 `initialize` → `tools/list` → 8× `tools/call`（含 `operator_run` 与轮询用的 `operator_status`）。

独立终评入口（零模型调用，对导出的终稿重跑）：

```
$ operator-agent prepare --request relu --dsl python --shapes 4,8 --run-dir .../final-check
$ operator-agent verify  --run-dir .../final-check --backend fake --code-file .../cli-success/export/code.py
    → L1 True / L2 True（12 case）/ L3 False，validity=uninformative
```

与本文「L1/L2 通过、L3 不具备资格」一致；导出目录每件带 SHA256 并标 `test_only: true`。

### 6.4 桌面端（DSH NEXT 2.0.21-next）上的真实安装与一条**真缺陷**

在活动 profile `~/.dsh/profiles/desktop` 上做（先备份到 `desktop.backup-*`）：

```
$ DSH_HOME=~/.dsh dsh plugin --profile desktop add dsh-operator-agent-0.2.0-rc.1.tgz --offline
    dependencies: + dsh-operator-agent file:...   bundles: [..., dsh-operator-agent]   (+3 -7)
```

**发现的真缺陷**：该 profile 的 patch 层里那条 `mcp-operator` 写的是

```
command: ~/opt/miniconda3/bin/python3
args: [-m, operator_agent.mcp_server]        ← 顶层没有这个模块
```

真模块是 `operator_agent.app.mcp_server`。而该条目所在 bundle 的 `failOnStartupError: false`
把它变成了**静默不可用**：本应用日志里连续记着

```
host.stderr  | ~/opt/miniconda3/bin/python3: No module named operator_agent.mcp_server
host.kernel.mcp-client | mcp-client(operator): connection attempt failed: SdkError: Connection closed
                       | retrying in 16000ms (attempt 6/10) … (attempt 7/10)
```

改成 `operator_agent.app.mcp_server` 后，应用热重载并成功拉起了服务：

```
$ ps -ef | grep operator_agent.app.mcp_server
  pid=53966 ppid=44701(桌面端)  ~/opt/miniconda3/bin/python3 -m operator_agent.app.mcp_server
```

且其后日志里**没有** operator/mcp 相关的 error 或 warn。`patch: entry "mcp-operator" not found`
这条 warn 也随之消失（它此前说明 patch 里那两条 `- id:` 覆盖在目标 bundle 未安装时是空转的）。

⚠️ **`+3 -7`**：`pnpm add` 会顺带修剪 profile 自己的 `node_modules`（7 个 extraneous 包）。
实测其他 bundle 的 `not found` 警告集合在安装前后一致，未观察到功能损失；但这是**要盯的副作用**。

### 6.5 隔离 DSH profile（配置形态验证，用老 CLI）

```
$ DSH_HOME=.../dsh-home dsh plugin --profile v07-check add dsh-operator-agent-0.2.0-rc.1.tgz
    → Done in 445ms
$ DSH_HOME=.../dsh-home dsh --profile v07-check --dump-config | grep -E "id: (mcp-operator|operator-tools)"
    - id: mcp-operator      command: operator-agent-mcp
    - id: operator-tools    command: operator-agent
```

两条入口都在。随后按 `examples/v07/profile-absolute-paths.patch.yml` 的形态改成干净 venv 的绝对路径（GUI DSH 不继承终端 PATH）：

```
- id: mcp-operator    config: command: .../v07-venv/bin/operator-agent-mcp
- id: operator-tools  config: command: .../v07-venv/bin/operator-agent
```

用配置里那条绝对命令实测 MCP 通道：`initialize` → `serverInfo {name: operator, version: 0.7.0rc1}`；`tools/list` 8 个工具；`operator_capabilities` 回 `version=0.7.0rc1 tool_version=10 fake_wired=True`。

> 踩到一次：`cordis.patch.yml` 原本是空数组 `[]`，直接往后追加块条目会让 YAML 失效（`--dump-config` 报行号）。必须把 `[]` 换成块序列。

### 6.6 `/op capabilities` 实测（用户 2026-11-03 在应用里执行）

```
✓ 已执行 · 算子判定层 0.7.0rc1 · 后端 fake/local/mock/ssh · 10 个算子有输入布局
工具版本：tool 10（回包契约版本，与判定层版本不是一回事）
已接线后端：fake / local / mock / ssh
运行时：python 3.9.12 · torch 有 · triton 有
有输入布局的算子：add, gelu, gemm, layernorm, log_softmax, matmul, relu, rmsnorm, silu, softmax
```

入口、版本、tool 10、四个后端全对，`/op` 这条斜杠命令确实通了 —— **§6.4 修的那条模块路径是本轮唯一阻塞**。

⚠️ 这次 `/op` 报的 `python 3.9.12` 说明当时入口跑的是 **conda base + PYTHONPATH 源码树**，不是 §6.1 的工件。
后续已两次改口径：先换 `py12`（§6.6 末），再换成**交付 wheel 的 venv**（§6.7，`python 3.12.16`）。

**2026-11-03 后续：解释器已换成 conda 的 `py12`（3.12.13）**，两处 `command` 都改掉后应用热重载，
新进程确认跑在 3.12 上、无报错：

```
$ ps -ef | grep operator_agent.app.mcp_server
  pid=59897 ppid=44701 | ~/opt/miniconda3/envs/py12/bin/python -m operator_agent.app.mcp_server
$ ~/opt/miniconda3/envs/py12/bin/python -V
  Python 3.12.13
```

⚠️ **base 升不到 3.12**：`conda.anaconda.org` 现在 HTTP 000（不可达），`conda install python=3.12`
装不了；而且 base 里那套 torch 2.8 是给 3.9 编的，换解释器等于全部重装。所以走的是
「已有 env `py12`」这条路，base 未动。

⚠️ **`py12` 里没有 torch**（只有 numpy 2.5.1，triton 是个没有 `__version__` 的空壳）。
这意味着：`/op` 与 MCP 的判定/契约/预算功能正常（纯 Python），但 `backend=local` 的真算力检查
在这个解释器上跑不了 —— 真机走 `backend=ssh`，不受影响。**仓库自己的 pytest 仍需 base 那套
（3.9.12 + torch 2.8）**，所以没有把 `py12` 设成 shell 默认环境：那会让 `python -m pytest` 直接
`ModuleNotFoundError: torch`。若确实要把默认环境换成 py12，在 `~/.zshrc` 的 conda init 块后加一行
`conda activate py12` 即可，但先确认本仓的测试工作流能接受没有 torch。

### 6.7 收口：入口已切到**交付的 wheel**

按 §6.6 最后那条把 profile 两处 `command` 换成 §6.2 的干净 venv（并去掉 `-m ...`、`cwd`、`PYTHONPATH`），
应用热重载后：

```
$ ps -ef | grep operator-agent-mcp
  pid=60980 ppid=44701 | .../v07-venv/bin/operator-agent-mcp     # 进程 cwd = ~/.dsh/profiles/desktop
$ cd /tmp && .../v07-venv/bin/python -c "import operator_agent; print(operator_agent.__file__)"
  .../v07-venv/lib/python3.12/site-packages/operator_agent/__init__.py
```

两条命令各自端到端验过（从**中立 cwd** `/tmp` 跑，排除源码树遮蔽）：

| 命令（`/op capabilities` 与 MCP 的背后） | 结果 |
|---|---|
| `.../v07-venv/bin/operator-agent capabilities` | `version=0.7.0rc1 tool=10 **python=3.12.16** torch=False` |
| `.../v07-venv/bin/operator-agent-mcp` 握手 + `operator_capabilities` | `serverInfo {name: operator, version: 0.7.0rc1}`；`python=3.12.16`；后端 `fake/local/mock/ssh` 全 wired |

⚠️ `torch=False` 是**预期的**：交付 wheel 只依赖 NumPy，控制面不需要 torch；真机走 `backend=ssh`，
远端 runner 自带 torch 2.9.1+cu128。但 `backend=local` 的真算力检查在这条入口上跑不了 —— 这是
「控制面零第三方依赖」这个设计的直接后果，不是缺陷。

### 6.8 仍然**待验收**

- **`/op` 与 DSH Agent 驱动的一次工具调用**：需要真实 DSH 会话（外层模型另配），本轮没做。上面的结果只证明配置合成与 MCP 通道可用。
- ~~`/op capabilities` 在 wheel 上的最终确认~~ **已闭（2026-11-04）**：用户在应用里执行，报 `python 3.12.16`，
  即入口跑的是 §6.1 的交付工件，不是源码树。
- **从 DSH 插件入口走完一次完整流程**（声明 → 生成/修复 → 状态/取消 → 独立终评 → 源码/报告导出）。
  §6.7 只证明入口与 capabilities 可用；一期清单 §5.2 第 4 条要的是**整条工作流经宿主**跑通，
  `CLI 试跑不能代替宿主入口验收`。可先用 `backend=fake` 在应用里跑一次，零成本。
- **`mcp__operator__*` 八个工具在会话里被实际调用一次**：MCP 进程已起、握手已验，但还没有一次真实的工具调用留痕。
- **`+3 -7` 那 7 个被修剪的包**：未逐个核对是否为其他插件所需（安装前后 `not found` 集合一致，暂判无损）。
- **干净机器上 `uv build`**：本机网络不通（6.1 偏差 1）。
- **卸载与回退演练**：只做了安装，`plugin remove` / `pip uninstall` / 切回旧 venv 未跑。
- 源码树测试不能外推为安装包通过；本文所有 fake 结果都是 `test_only`，不是 GPU 编译/数值/性能资格。

### 6.9 本轮对活动 profile 的改动与回滚

| 改动 | 回滚 |
|---|---|
| `~/.dsh/profiles/desktop/package.json` 加 `dsh-operator-agent` 依赖与 bundle | 从 `desktop.backup-<ts>/package.json` 覆盖回去，再 `node <dsh-runtimes pnpm> install --frozen-lockfile --offline` |
| `~/.dsh/profiles/desktop/cordis.patch.yml` 的 `-m operator_agent.mcp_server` → `.app.mcp_server` | 同上，用备份的 `cordis.patch.yml` 覆盖 |
| `node_modules`（`+3 -7`） | 同上，`pnpm install --frozen-lockfile --offline` 会复原 |

备份目录：`~/.dsh/profiles/desktop.backup-20261103232716`、`desktop.backup-20261103232928`。
卸载：`DSH_HOME=~/.dsh dsh plugin --profile desktop remove dsh-operator-agent`。

## 7. 本轮改动清单（2026-11-03）

| 文件 | 改动 |
|---|---|
| `docs/testing/v07-delivery.md` §6 | 由模板改为**实测记录**：工件哈希、干净安装、三场景、独立终评入口、DSH profile、5 项待验收 |
| `integrations/dsh-operator/examples/v07/` | **未改动**。本轮为写 §6 才发现它已经存在（22:21 建，`package.json` 的 `files` 已含 `examples/v07/**`，`npm pack` 会自动收）。我曾另写一份放到仓根 `examples/v07/`，发现重复后**已删除**，并把一度加进 `ALLOWED_ROOT` 的 `examples` 也还原 —— 保持单一真源：示例随插件包发，而不在仓根再造一份 |

工件与示例落在交付目录 `~/op-agent-delivery/v0.7.0rc1/`（`manifest.json` + `artifacts-SHA256.txt`），解包后的插件在 `.../unpacked/package/`，三份运行证据在 `.../runs-shipped/`。
回归：`tests/unit tests/hygiene` **5990 passed / 1 skipped**（skip 为需要真实 CUDA 的那条）。

## 8. 无需租卡准备收口（2026-11-04，独立候选）

本节保留 §6–§7 的用户实测记录，补充另一轮独立工件和隔离环境的验收。未改动个人 `desktop` profile。工作目录为 `/private/tmp/operator-v07-candidate-20261103/`，最终身份、依赖和回执以其中 `delivery.json`、`CONTENT-SHA256.json` 为准。

| 工件 | SHA256 |
|---|---|
| Python `0.7.0rc1` wheel | `bc79d14ea5d77fbfbbd7d92807a1d56ed6cb60ed4f9b9db586444224198e54b1` |
| 插件 `0.2.0-rc.1` tgz | `3813dbc5dfda284a790a62ab53379ddbd0412833584428763ca4e7938744e72c` |
| Python sdist | `6c4f253b1ae15806c070c62964497e5814809be3390d67523288e586dd904bbc` |

本节 wheel 与 §6.1 的 `aa3cc8…` 是不同 ZIP 工件：已逐文件比较，两者的 `operator_agent/`、`remote_scripts/` 运行内容完全相同，差异在构建元数据与许可文件位置。仍分别保留哈希，不将隔离环境的结果记作对个人活动 profile 的新操作。

本轮已完成：

- Python 3.12.16 新 venv 从 wheel 安装，`-I` 和仓外工作目录确认导入来自该 venv 的 `site-packages`；逐字节复核 145 个已安装运行文件。
- 最终 tgz 随包的 CLI/MCP × 成功、正确性修复、取消，共六种示例全部通过；独立终评完成完整 L2，L3 不具备资格，源码/报告导出逐项核对 SHA256。
- 通过实际桌面端内嵌 DSH `0.2.1-rc.2`、Node `24.14.1` 和最终解包插件驱动 `operator_run` / `operator_status`。外层为本地 HTTP fixture，内层为脚本 fake；同时由真实 MCP 验证 prepare、verify、修复、终评和 cancel。
- 使用同一内嵌 DSH 的私有 `HOME` / `DSH_HOME` 实际 add、合成配置和 remove；命令固定为 clean venv 的绝对路径。移除后依赖、bundle、包目录及合成工具全部消失；从交付包 `tools/` 运行该检查也通过。
- 利用本地 `wheelhouse/` 实际执行无网络安装、`pip check`、卸载及重新安装；卸载后两个 console script 消失，重装后 145 个运行文件与 wheel 一致。

`wheelhouse/` 中 NumPy `2.5.4` 仅适用于 **CPython 3.12 / macOS 14+ / arm64**。其他平台需单独取得匹配依赖，不能外推本机读数。交付辅助脚本负责复制导出，核心 CLI 未新增 `export` 命令。

Python 3.12 开发环境现已独立安装 NumPy、Torch 和测试依赖；不需要借用 §6 所述的旧 conda 测试环境。首轮大范围回归为 6006 passed / 1 skipped / 2 failed，保留原日志；失败项涉及候选版本号守卫及实验 DSH 启动/流转发。修复后相关范围 **124 passed**（46.31 秒），插件测试 **58 passed**；从交付包自带工具与源码快照重建也已通过。另修正遗留隔离测试误读“禁止访问 site-packages”为失败，以及把旧 CPython 3.9 NumPy 注入 3.12 的测试环境问题。命令和日志摘要记录在 `delivery.json`，未将局部复验写成一次新的全量通过。测试框架另有多线程 fork 的弃用提示，本轮未改执行模型。

以上闭合的是无需租卡的安装与交付准备。真实 GPU/运行时支持、性能资格、正式模型采样对齐及同卡四任务八格继续待验收；公开发布未执行。

## 9. rc2 干净安装与宿主入口验收（2026-11-04，本机）

针对 `0.7.0rc2` / 插件 `0.2.0-rc.2`（提交 `d4b27b8`，工件哈希见交付目录 `manifest.json`）。

环境与偏差：本机到 `pypi.org` 返回 `http=000`（不可达），因此 venv 用
`--system-site-packages` 建（numpy 取自宿主 conda py12 的 2.5.1），wheel 用
`pip install --no-deps` 装入。**「无网络下重新解析依赖」这一步仍然没有验证过**，其余步骤都是真跑。

| 步骤 | 命令要点 | 读数 |
|---|---|---|
| 干净 venv 安装 | `python3.12 -m venv --system-site-packages` + `pip install --no-deps <wheel>` | 安装成功；`pip check` = **No broken requirements found** |
| 能力发现 | `operator-agent capabilities`（在 `/tmp` 下跑，排除源码树遮蔽） | `version=0.7.0rc2`、`tool_version=10`、`torch_installed=false`、`triton_installed=false`（控制面零第三方依赖，符合设计） |
| 随包示例 · CLI 成功 | `examples/v07/offline_smoke.py --entry cli --scenario success` | `ok=true / phase=done / test_only=true / gpu_acceptance=false` |
| 随包示例 · MCP 修复 | `--entry mcp --scenario correctness-repair` | `ok=true / phase=done` |
| 随包示例 · MCP 取消 | `--entry mcp --scenario cancel` | `ok=true / phase=cancelled` |
| 导出物 | `export/` | `code.py` / `report.md` / `status.json` / `manifest.json` 四件齐备 |
| 独立终评 | `prepare` + `verify --backend fake --code-file export/code.py`（零模型调用） | `L1✓ L2✓ L3✗`（fake 无性能资格），`execution_context.test_only=true`、`gpu_acceptance=false` |
| 隔离 DSH profile | `DSH_HOME=<临时> dsh plugin add <tgz> --offline` → `--dump-config` → `remove` | 合成配置含 `mcp-operator` 与 `operator-tools` 两条入口；卸载成功 |

⚠️ 隔离 profile 用的是 PATH 上的 `dsh@0.1.1-rc.2`，**不满足**插件 `engines: ^0.2.1-rc.2`，
所以这一步证明的是**配置形态与打包正确**，不是「在满足版本要求的宿主上装好」。桌面端
（内嵌 `0.2.1-rc.2`）上的安装与 `/op` 实跑属用户环境，未在本轮改动。

**结论**：rc2 工件在干净 venv 里可安装、可运行、可导出、可独立复评，示例与配置形态正确；
仍然缺的是「在符合 engines 的宿主上从插件入口跑一次完整真实任务」，以及无网络重建依赖。
