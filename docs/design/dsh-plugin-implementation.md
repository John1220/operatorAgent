# DSH 插件实现说明（判定层工具面）

日期：2026-10-19。目标：把**已有的判定层**包成 DeepSeek Harness（DSH）能直接调的工具面，
让人或外层模型「一拍一步」地驱动算子开发，或把整段交回本仓的 `Conductor`。

**这不是什么**：不是 ESM/Cordis 插件（没有 `package.json`、没有 npm 包）；不做 UI/曲线面板；
**不改任何判定语义**（容差、反作弊、资格、入库门一个字没动）。

决策与实测读数在 [`docs/dsh-plugin-evaluation.md`](../dsh-plugin-evaluation.md)（§9.5–§9.13），
本文只回答「**怎么用**、它由什么组成、契约是什么、坏了怎么查、怎么加第六个工具」。

---

## 0. 怎么用

### 0.1 三种入口，调的是同一批函数

| 入口 | 场景 | 怎么起 |
|---|---|---|
| **DSH 里的工具** | 人/模型在对话里一步步驱动 | 装一次（§6），之后直接说“用 operator_* 工具…” |
| **CLI** | 脚本、排障、不进 DSH | `python -m operator_agent <子命令>` |
| **协议自检** | 验证 MCP 实现本身 | `python -m operator_agent.mcp_server --selftest` |

先确认环境（只读，不跑活、不联网）：

```bash
python -m operator_agent capabilities
```

返回里 `backends.{mock,local,ssh}.wired` 说明哪条路已接线；`runtime.torch_installed` /
`triton_installed` 说明这台机器装了什么；`kernel_types_with_input_layout` 是**有输入布局**
的算子（不在表里的算子 `prepare` 会明确拒绝，不猜布局）。

### 0.1.1 斜杠命令 `/op`（宿主插件，可选）

装了 `integrations/dsh-operator` 之后还会有两条宿主侧的记录：MCP 的六个工具，加上
`/op` 命令与一段系统提示段。命令**不经过模型**，输出直接给人看：

```
/op capabilities          这台机器能做什么（只读）
/op status <run_dir>      无人值守的相位（running/done/failed/lost）
```

命令只调 `operator-agent` CLI（与 MCP 工具同源），**不复制任何判定逻辑**；
命令不存在时返回「pip install operator-agent」而不是抛异常。

`<run_dir>` 的相对路径按**会话工作区**解析（`ctx.workingDirectory`），不是按宿主进程的
cwd——GUI 进程的 cwd 多半是 `/`，按它解析会把「run 在但路径不对」报成「run-status.json
不存在」。`~/x` 展开到 home，含空格整段保留（`rest` 是原样剩余整段，不按空白切），
输出回显实际读取的目录。

### 0.2 装进 DSH（只做一次）

装的是**两条记录**：一条 MCP（六个工具），一条宿主插件（`/op` 命令 + 提示段）。
打包好的版本在 [`integrations/dsh-operator/`](../../integrations/dsh-operator/README.md)，
它的 `cordis.patch.yml` 一次插两条。

**形态一：pip 装（发布的形态）**

```bash
pip install .
dsh plugin --profile <profile> add /path/to/operatorAgent/integrations/dsh-operator
```

**形态二：源码树跑（不碰 Python 环境，本机当前用的是这种）**

```yaml
- insert:
    - id: mcp-operator
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: operator
        transport: stdio
        command: <abs-python>
        args: ["-m", "operator_agent.mcp_server"]
        cwd: <abs-repo>
        toolCallTimeoutMs: 600000
        failOnStartupError: false

    - id: operator-tools
      name: <abs-repo>/integrations/dsh-operator/index.mjs
      config:
        command: <abs-python>
        args: ["-m", "operator_agent"]
        env:
          PYTHONPATH: <abs-repo>
        timeoutMs: 60000
```

之后**重连 MCP**（工具表是 `mcp_server.py` 里的静态表 ⇒ 改它要重启 server **进程**）：

```bash
pkill -f operator_agent.mcp_server      # 宿主 500ms 后自动重连并 re-sync
```

模型侧会出现六个工具：`mcp__operator__operator_{prepare,verify,repair,run,status,capabilities}`；
宿主侧出现 `/op` 命令与 `tool:operator` 提示段。**本会话的工具列表要到下一回合才刷新** ——
这是宿主的刷新时机，不是插件的问题。**斜杠命令有时要重启一次宿主才出现在命令面板**
（工具与提示段是热加载的，命令面板不一定是）。

### 0.3 工作流 A：冒烟（不碰 GPU，30 秒）

用 mock 后端把「声明 → 判定」走一遍，确认链路与回包字段：

```bash
python -m operator_agent prepare \
  --request "把 64x64x64 的 fp32 矩阵乘法写成算子" \
  --shapes 64,64,64 --run-dir /tmp/op-smoke --min-speedup 0.5

printf 'def kernel(A, B):\n    return (A @ B) * 0.9\n' > /tmp/bad.py
python -m operator_agent verify --run-dir /tmp/op-smoke --code-file /tmp/bad.py --stop-after L2
```

应看到 `levels={"L1":true,"L2":false}`、`route.route="debug_fix"`、`route.error_shape`，
以及一条指向「整张量错 ⇒ 核对数值语义/输出缩放」的处方。

### 0.4 工作流 B：手写核 + 真机（HITL，人看得见每一步）

外层（人或模型）写候选，判定层只判。四步：

```bash
# 1) 声明：形状、精度、必须真机
python -m operator_agent prepare \
  --request "把 4096x4096x4096 的 fp16 矩阵乘法用 Triton 写成算子" \
  --shapes 4096,4096,4096 --hardware nvidia_gpu --dsl triton --precision fp16 \
  --require-cap real_device --min-speedup 1.0 --run-dir /tmp/op-r1

# 2) 先只到 L1（便宜：不执行、只编译）
python -m operator_agent verify --run-dir /tmp/op-r1 --code-file cand.py \
  --backend ssh --host <ssh-alias> --stop-after L1

# 3) 正确性（15 个边界 case；真机上这一步最贵）
python -m operator_agent verify --run-dir /tmp/op-r1 --code-file cand.py \
  --backend ssh --host <ssh-alias> --stop-after L2

# 4) 全量（含性能与 roofline）
python -m operator_agent verify --run-dir /tmp/op-r1 --code-file cand.py \
  --backend ssh --host <ssh-alias>
```

`<ssh-alias>` 是 `~/.ssh/config` 里的别名；**没有 `auto` 降级**，不给 host 会直接
`host_required`。每一步都会在 `/tmp/op-r1` 留下 `verify-N.json`（回包）、`verify-N.log`
（过程）、`report.md`（报告）。

### 0.5 工作流 C：整段自动化（无人值守）

一拍启动、四态轮询；跑的是本仓 `Conductor`（它自己设计/写核/修/搜索）。

```bash
python -m operator_agent run \
  --request "把 4096x4096x4096 的 fp16 矩阵乘法用 Triton 写成算子" \
  --shapes 4096,4096,4096 --hardware nvidia_gpu --dsl triton --precision fp16 \
  --require-cap real_device --min-speedup 1.0 \
  --backend ssh --host <ssh-alias> --max-rounds 2 --run-dir /tmp/op-auto
# → {"phase":"running","pid":12345,...}   立刻返回

python -m operator_agent status --run-dir /tmp/op-auto
# → running / done / failed / lost
```

需要**内层 LLM**（见 §6），没配会立刻 `llm_not_configured`、不起进程。
`done` 之后看 `/tmp/op-auto/conductor.json`（`all_passed` / `levels` / `speedup` / `validity` /
`rounds` / 报告目录）与 `autonomous.log` 的尾 20 行。

**在 DSH 里等价于**：`operator_run` 一次，然后隔一会儿 `operator_status` 一次；不要用一次
工具调用去等它跑完（真机一次可达 15 分钟，超过 `toolCallTimeoutMs` 的默认值）。

### 0.5.1 工作流 D：给一份已有的核，让它自己修

`operator_run` 是**从零生成**；如果你手上已经有一份核（人写的、上一轮的、别处拷来的），
用 `repair`：

```bash
python -m operator_agent repair --run-dir /tmp/op-r1 --code-file cand.py \
  --max-rounds 2 --stop-after L2 --backend ssh --host <ssh-alias>
```

它每轮做：`verify` → 读 `failure_route` 处方 → `Fixer`（内层模型）→ 再 `verify`；
`hold`（这一轮没判出对象）时**不修**，只说明理由。实测 6 类缺陷里 5 类一轮改对。

### 0.6 回包先看哪几格

| 字段 | 读法 |
|---|---|
| `levels` | `{"L1":…,"L2":…,"L3":…}`，只含**跑到过的**级（`stop_after` 之后的级不出现） |
| `route` | 处方：`fix` / `debug_fix` / `search` / `retry` / `hold` + `fix_not_rewrite` |
| `eligibility` | 五位资格：这个数**被允许造成什么后果**（判对错/可调参/可入库/可报数/算进展） |
| `conflict` | **必读**：`present=true` 时处方与资格打架，按 `action=hold_for_measurement` 办 |
| `validity` | 测量档位：`valid` / `degraded` / `uninformative` |
| `notes` | 可判的读法提示（如「基线是 TF32，契约是 fp32 ⇒ 不是同语义比较」） |
| `report` | 报告路径；「我无法确立的事」在里面 |

口诀：**`route` 与 `eligibility` 一起读**；两者冲突时先修判据（形状/精度/`required_caps`），
别照着 `route` 去迭代。

### 0.7 常见拒绝怎么处理

| 现象 | 原因 | 动作 |
|---|---|---|
| 工具列表里没有 `operator_*` | MCP server 没连上/工具清单过期 | `pkill -f operator_agent.mcp_server`，等重连 |
| `/op` 补全不出来 | 命令面板要重启宿主才刷新 | 重启 DSH；配置不用改 |
| `llm_not_configured` | 无人值守缺内层 LLM | 在 MCP 行 `env:` 里配 `LLM_*`（§6） |
| `host_required` | `backend=ssh` 没给 host | 加 `--host <ssh-alias>` |
| `backend_not_wired` | backend 名字不在 `mock/local/ssh` | 改值；没有 `auto` |
| `contract_drift` | 声明在回合之间被改过 | 重新 `prepare`，别在旧回合上继续 |
| `run_not_started` | `status` 查了个没跑过的目录 | 先 `run` |
| `phase=lost` | 进程没了、终态没写 | 看 `autonomous.log` 尾部；按 pid 已不存在 |
| `output is not valid JSON`（旧版） | 库的 print 漏进 stdout | 已修；出现在旧进程就重连 |

### 0.8 卸载/禁用

删掉 profile 里那段 `insert`，`pkill -f operator_agent.mcp_server` 即可；仓库侧不需要改任何东西
（磁盘上的 run 目录留在你给的位置，不会被清理）。

---

## 1. 架构（一行图）

```
DSH  ──(MCP stdio, JSON-RPC)──▶  operator_agent.mcp_server
                                      │
                                      ├─▶ operator_agent.hitl_tools ──▶ 既有判定层
                                      │      prepare / verify / capabilities      contract / Verifier /
                                      │                                          failure_route / report
                                      └─▶ (独立进程) operator_agent.autonomous
                                             └─▶ Conductor（无人值守整段）
```

`operator_agent.cli` 是同一层的**进程入口**（`python -m operator_agent …`），不经过 MCP，
供脚本与排障使用；两条路调的是同一批函数。

---

## 2. 六个工具的契约

| 工具 | 做什么 | 必填 | 立刻返回 |
|---|---|---|---|
| `operator_capabilities` | 只读：这台机器能做什么（运行时、可用算子、后端是否接线） | — | 能力表；**不跑活、不联网、不探测 SSH** |
| `operator_prepare` | 声明 → 冻结契约 + 契约自带输入 + 回合身份 | `request` `shapes` `run_dir` | `contract_hash` / `cases` / `inputs` |
| `operator_verify` | 对**外层给的**候选源码判 L1/L2/L3 | `run_dir` + (`code` \| `code_file`) | 正交判定 + 处方 + 资格 + 报告路径 |
| `operator_run` | 起一次**无人值守**整段（本仓 `Conductor` 自己设计/写核） | `request` `shapes` `run_dir` | `pid` / `phase=running` / `status` 路径 |
| `operator_status` | 查无人值守的相位 | `run_dir` | `running` / `done` / `failed` / **`lost`** |
| `operator_repair` | 给一份**已有的**核：判 → 处方 → 内层模型修 → 再判 | `run_dir` + (`code` \| `code_file`) | 每轮的判定/处方/动作，最终是否通过 |

**声明（declaration）白名单**（写进契约的只有这些）：
`request` `shapes` `hardware` `dsl` `precision` `min_speedup` `required_caps` `framework` `notes`。

**运行选项**（只影响怎么跑，不进判据）：
`max_rounds` `backend` `host` `port` `user` `key_path` `work_dir` `timeout` `persistent` `target_gflops`。

两套词表**刻意分开**：前者决定「判什么」，后者决定「怎么跑」；混用会被
`unknown_parameter` 拒（2026-10-19 在真宿主里撞过一次，见 `tests/test_mcp_server.py::test_run_splits_tool_options_from_the_declaration`）。

**不允许**出现在任何参数里的（出现即 `forbidden_parameter`，拒绝理由会写明属于哪一条）：
`max_error` / `atol` / `rtol` / `skip_anticheat` / `require_double_confirm` / `skip_remeasure` /
`kernel_db` / `failure_db` / `disable_logging` / `failure_gate`。见 `hitl_tools.FORBIDDEN_KEYS`。

---

## 3. 数据布局（一个 run 目录）

调用方拥有布局；工具只往里写，不猜别的位置。

```text
<run_dir>/
  declaration.json      # 输入的唯一真相（verify 会重算并比 hash）
  contract.json         # {hash, contract}：冻结契约 + 身份
  run.json              # task_id / 创建时间 / 工具版本
  verify-N.json         # 第 N 次判定的回包（裁剪版，<12 KB）
  verify-N.log          # 判定层的过程输出（stdout 被收在这里）
  report.md             # 最近一次的报告（best-known-good + 证据链 + 我无法确立的事）
  # 下面只在 operator_run 之后出现：
  autonomous.json       # 启动参数（backend/max_rounds/…）
  run-status.json       # 相位机（running → done|failed，含 pid 与错误原文）
  conductor.json        # 摘要（all_passed / levels / speedup / validity / rounds / 报告目录）
  autonomous.log        # 子进程 stdout+stderr
  output/ log/          # 子进程把 OPERATOR_AGENT_* 重定向到这里
  kernel_db.jsonl       # 同上（语料默认**运行内**，不写仓库那份）
  failure_db.jsonl
  candidate/            # Conductor 的 §22.1 布局（过门的候选落这里）
```

**身份怎么保证**：`verify` 会重新冻结契约并和 `contract.json` 里的 hash 比；不一致返回
`contract_drift`，**不按新判据继续**（与 `benchmarks/workload_codegen_h20.py` 的
「执行必须等于被授权的计划」同一条纪律）。

### 3.1 输入传输：seed + digest（2026-10-20）

L2/L3 的输入**不再整份传字节**。控制端先确认「这组输入就是契约里某条 case 物化出来的那份」，
然后只发一段描述（**231 字节**）：

```json
{"protocol":"numpy-seed-v1","seed":0,
 "tensors":[{"name":"A","shape":[4096,4096],"dtype":"fp16"}, …],
 "digest":"<sha256>"}
```

- **控制端**（`Verifier._seed_spec_for`）：按契约的 `seed`/`shape`/`dtype` 物化一次，
  与真实 inputs **逐字节比 digest**；不一致就退回传字节。带标量/非 ndarray 的输入一律不走这条路。
- **远端**（`run_kernel` 的 `verify` / `benchmark*` 分支）：再物化一次、再比一次；
  对不上回 `input_seed_mismatch`。
- **回退**：控制端收到 `input_seed_mismatch` 就用老的字节协议重发，并在回包里记
  `input_seed_fallback: true` —— **慢但正确**，而且可归因。
- 抽取规则只有一份实现（`remote_input_fidelity.materialize_seed_inputs`），
  `BenchmarkCase.materialize` 也调它；两份实现分家就是「控制端以为在验 A、远端在验 B」。

实测：主 case 4096³ 一次判定 **572 s → 16.5 s**；端到端无人值守 **730.9 s → 240.6 s**。
守卫：`tests/test_input_seed_spec.py`。

---

## 4. 状态机与拒绝码

```text
operator_run:  (未开始) ──llm 缺失──▶ llm_not_configured（不起进程）
                   └──start──▶ running ──子进程写终态──▶ done | failed
                                    └──进程没了、终态没写──▶ lost（查询时判出，不写盘）
```

`lost` 是**独立一档**：`process_alive=false` 但 `phase` 仍是 `running` 时返回 `lost`。
读成 `running` 会让人一直等下去（`tests/test_autonomous.py::test_a_dead_pid_without_final_state_is_lost_not_running`）。

| 拒绝码 | 含义 | 处理 |
|---|---|---|
| `request_required` / `shapes_required` | 声明缺必填项 | 补上；`shapes` 是契约拥有输入的前提，**不猜** |
| `bad_shapes` / `unknown_cap` | 形状或能力名不合法 | 修值；`Cap` 是闭合词汇 |
| `forbidden_parameter` | 触碰安全面（见 §2） | 去掉；判据不可由调用方放宽 |
| `unknown_parameter` | 键不在白名单 | 见 §2 两套词表 |
| `contract_refused` | 冻结失败（无布局表 / 套件无效 / 声明冲突） | 看 `detail`；`attention/conv2d` 目前没有输入布局 |
| `contract_has_no_inputs` | 契约派生不出 case | 声明 `shapes` |
| `inputs_not_materializable` | dtype 造不出张量 | 换 `precision` |
| `backend_not_wired` | `backend` 不在 `mock/local/ssh` | 见 §5 |
| `host_required` | `backend=ssh` 但没给 host | 给 `~/.ssh/config` 的别名；**没有 auto 降级** |
| `code_required` / `code_source_ambiguous` | 源码缺失或 `code`/`code_file` 同时给 | 只给一个 |
| `run_not_prepared` | 目录里没有 `declaration.json` | 先 `operator_prepare` |
| `contract_drift` | 回合之间判据被改过 | 重新 prepare，别在旧回合上继续 |
| `llm_not_configured` | 无人值守缺内层 LLM | 见 §6 |
| `run_not_started` | `status` 查一个没跑过的目录 | 先 `operator_run` |
| `unknown_tool` | 工具名不存在 | 见 §2 |

---

## 5. 后端（`backend`）

| 值 | 实际执行 | 边界 |
|---|---|---|
| `mock`（默认） | 控制机 numpy（`MockGPU`） | `mode=actual` 但**不是加速卡**；不需要 GPU |
| `local` | `RemoteExecutor(mode="local")` | 能力集是**空集** ⇒ 没有输入回读协议，带套件的 L2 判 `unknown`（**不是候选错了**） |
| `ssh` | `RemoteExecutor(mode="ssh")` 到 `host` | 必须显式 `host`；选卡按利用率；`persistent` 开常驻会话（doc 62） |

**没有 `auto`**：`auto` 会一路降到 CPU，表现为「以为在真机上、其实在控制机」，是这条路上最贵的误读。

---

## 6. 配置面

| 变量 | 谁用 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | 无人值守（内层） | OpenAI 兼容端点，带 `/v1` |
| `LLM_MODEL` | 同上 | 模型 id |
| `LLM_API_KEY` 或 `LLM_API_KEY_FILE` | 同上 | 值或**文件引用**；文件建议 0600、放仓库外 |
| `OPERATOR_AGENT_OUTPUT_DIR` / `_LOG_DIR` | 子进程 | 由 `autonomous.py` **强制**重定向进 run 目录 |
| `OPERATOR_AGENT_KERNEL_DB` / `_FAILURE_DB` | 子进程 | 同上；⚠️ 写 `off` **关不掉**持久化（只能重定向），所以这里指向 run 目录里的文件 |

**为什么必须写在宿主 MCP 行的 `env:` 里**：DSH 会先洗掉子进程环境里名字含
`KEY`/`PASSWORD`/`SECRET`/`TOKEN` 的变量（`/KEY|PASSWORD|SECRET|TOKEN/i`）、**外加所有
`DSH_*`**，再应用显式覆盖。`LLM_API_KEY_FILE` 含 `KEY`，靠全局环境变量传不进去；
`DSH_HOME` 之类也一样。`/op` 那条路（宿主插件自己 `execFile`）现在用**同一套规则**
（`scrubEnv` + 配置里的 `env` 在其后合并）：`command` 是可配置的，不该因为换了个包装脚本
就把宿主密钥交出去。两条路的差别只剩「谁提供基础环境」，清洗规则是同一条。

```yaml
# ~/.dsh/profiles/<profile>/cordis.patch.yml
- insert:
    - id: mcp-operator
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: operator
        transport: stdio
        command: <abs-path-to-python3>
        args: [-m, operator_agent.mcp_server]
        cwd: <abs-path-to-this-repo>
        toolCallTimeoutMs: 600000
        failOnStartupError: false
        env:
          LLM_BASE_URL: <your-endpoint>/v1
          LLM_MODEL: <your-inner-model>
          LLM_API_KEY_FILE: ~/.config/operator_agent/llm.key
```

---

## 7. 与既有模块的边界

- **谁判**：`Verifier`（L1/L2/L3）、`measurement`（有效性三态）、`eligibility`（资格五位）、
  `oracle`（唯一入库门）、`anticheat`（静态 + 运行时）、`failure_route`（处方）。
- **谁不判**：`hitl_tools` 只做「读声明 → 调判定 → 组回包」。它**不改**任何判据。
- **两次确认 / 入库只在 `operator_run` 那条路上**（`Conductor` 的 `_produce_remeasure` →
  `oracle.admit_to_corpus`）。`verify` 只判定、不重测、不入库 —— 报告里会如实写「没有做重测」。
- **`route` 与 `eligibility` 必须一起读**：`failure_route.classify` 不读资格位（实测接缝），
  所以回包里多了 `conflict`：处方说「去搜索/算一轮」而资格说这个数不能驱动任何后果时，
  `conflict.action = hold_for_measurement`。**`classify` 本身没改**。

---

## 8. 测试与守卫（要改这里，先跑这些）

| 文件 | 钉住什么 |
|---|---|
| `tests/test_hitl_tools.py` | 白名单/黑名单不相交、回合身份可复算、契约漂移被拒、`conflict`、回包 <12 KB、语料不写回仓库、后端拒绝 |
| `tests/test_mcp_server.py` | 握手/版本协商/通知不响应/坏行不打断流、schema 无安全参数、**真进程 stdout 只有协议行**、参数拆分 |
| `tests/test_autonomous.py` | 缺 LLM 拒绝、独立进程跑完整段（本地假端点）、四态相位、仓库语料不被写 |
| `tests/test_demo_hitl_chain.py` | 零 GPU/零 LLM 演示里每条结论仍在屏幕上 |
| `tests/test_resource_precheck.py` | 资源预检读代码不读注释（§9.8 那个假阳性的守卫） |
| `tests/test_input_seed_spec.py` | 输入按 seed 物化 + 双向 digest + 字节回退；打包后 `protocol_identity` 仍成立 |
| `tests/test_repair_loop.py` | 修复环的控制流（pass 不修 / hold 不修 / 只提问不修 / 轮数耗尽） |
| `tests/test_packaging.py` | 版本一致、runner 随包发、模板是 package-data、运行期目录不进 sdist、shim 的 patch 存在 |
| `integrations/dsh-operator/test/plugin.test.mjs` | 插件注册面 + `/op` 行为（8 条，`npm test`） |

**加第六个工具**：① `mcp_server._tool_definitions()` 加 schema；② `_call_tool()` 加一个分支
（注意把**运行选项**从 declaration 里摘出来）；③ 若加新拒绝码，补进本文 §4 与测试；
④ **重启 server 进程**：`pkill -f operator_agent.mcp_server`，宿主会在 500 ms 后自动重连并
re-sync，**本会话的工具列表要到下一回合才刷新**。宿主本身支持热刷新
（收 `notifications/tools/list_changed` 即重新同步），是这张表在进程里是静态的才要重启；
重连有上限（默认退避 500 ms→30 s、10 次），别把它当日常刷新手段。

---

## 9. 已知边界与坑（都实测过）

1. **资源预检读注释**（已修，2026-10-19）：Fixer 在修复说明里写 `BLOCK_M=256`，预检把注释
   当配置，把正确的修复又拒一次。现在用 tokenize 原位抹掉注释/docstring。
2. **元素宽度仍从源码猜**（未修）：`_elem_bytes_from_code` 读到累加器的 `tl.float32` 就按
   4 字节算 ⇒ fp16 任务的 `256×256` tile 被估成 384 KiB（实际 192 KiB）。修复要改成用任务
   声明的 `precision`，属判定语义变更，待单独评估（评估文档 §9.13-7）。
3. **L2 的真实成本**：15 个 case 每个一次一次性 SSH + 远端 import + Triton JIT，真机一次
   完整判定实测 **866 s**（其中 L3 只占小头）。
4. **没有取消动作**：只能按 pid 杀；`run-status.json` 会停在 `running`，查询时判为 `lost`。
5. **`toolCallTimeoutMs`**：MCP 默认 60 s。`verify` 在真机上**一次完整判定就是 866 s**（见上面第 3 条），
   所以配置里设成 **3600 s**（60 min）——按 866 s 的单次成本 + `repair` 默认 2 轮（约 30 min）留余量。
   超时**不是优雅退出**：server 同步逐行读，客户端 abort 后它仍会算完再写一个没人认领的响应。
   更长的无人值守用 `run` + `status` 轮询，不要占用一次工具调用。

---

## 9.1 发布形态（2026-10-20 起）

| 侧 | 产物 | 说明 |
|---|---|---|
| Python | `pyproject.toml`（name `operator-agent`，两个 console script：`operator-agent` / `operator-agent-mcp`） | `pip install .` 后 `python -m operator_agent*` 与两个命令都可用；`remote_scripts/` 作为**兄弟顶层包**一起装（`remote_scripts/__init__.py` 就是为此存在） |
| DSH | `integrations/dsh-operator/`（`package.json` + `dsh.bundle.patch` → `cordis.patch.yml` + `index.mjs` + `icon.svg` + `locale/*.json` + 测试 + README + CHANGELOG） | 宿主插件只做两件事：`/op` 命令与一段 `tool:operator` 提示段，**不复制判定逻辑**；`dsh plugin add <path>` 或手改 profile 都行（`desktop` profile 需先退出 DSH Desktop，或用 GUI 插件页） |

**插件在宿主里的「脸」**（对着 `dsh-app-boot` 的 `readPluginMeta` 核过）：标题/描述/图标来自
`<包名>/package.json`、`<包名>/locale/<lang>.json`、`icon` 三处，全走**真模块解析器**，
所以 `exports` 里必须有 `"./package.json"` 与 `"./locale/*.json"`。少了不是报错而是**静默
退化**（标题变成裸包名、图标没有）——bundle 与 patch 的解析走磁盘路径，不受 `exports` 影响，
即「装得上，但没脸」。规则镜像在 `tests/test_packaging.py::TestTheDshPluginFace`，并用真
`readPluginMeta` 在 `test/host-contract.test.mjs` 里现场验一遍。

**已修的打包前置**：`protocol_identity()` 改成**包相对**解析（源码树与 site-packages 两种
布局都成立；缺文件时报出缺哪一个，而不是埋在几十条哈希里）。守卫在
`tests/test_input_seed_spec.py::TestProtocolIdentitySurvivesAPackagedLayout`。

**License**：Apache-2.0（`LICENSE` / `NOTICE` / `THIRD_PARTY_NOTICES.md` 在仓库根，
`tests/test_packaging.py::TestLicensing` 钉住「一处声明与三处字段互相指认」）。
`package.json` 目前仍是 `private: true`，所以**尚未** `npm publish`——发版时把它改成
`false` 并补 `dsh plugin add dsh-operator-agent` 一行。

**构建坑**：`pip < 21.3` 会把本地包复制到临时目录，而 `output/` 里的 `0400` 产物删不掉 ⇒
构建以 `PermissionError` 结束（wheel 已生成）。用 `--use-feature=in-tree-build` 或从干净
导出构建（见 `integrations/dsh-operator/README.md` §8）。

## 10. 指路

- 决策与读数：[`docs/dsh-plugin-evaluation.md`](../dsh-plugin-evaluation.md)（§9.5 工具壳、§9.6 MCP、
  §9.7 真机三连、§9.8 自动化入口 + 真机闭环、§9.9 处方对照、§9.11 自动修复统计、§9.12 发布形态、§9.13 待办）
- 用法：[`docs/user_guide.md`](../user_guide.md) §3.1 方式三（CLI）/ 方式四（接进 DSH）
- 代码：`operator_agent/mcp_server.py`（握手与工具表）、`hitl_tools.py`（判定封装）、
  `autonomous.py`（无人值守执行体）、`cli.py`（同源进程入口）
