# R2 跑法：两臂两个 profile

2026-11-07 · 计划 `operator-v08-increment-r2`（见 [预登记](v08-r2-preregistration.md)）

## 1. 环境

| | 臂 B（对照） | 臂 C（增量） |
|---|---|---|
| DSH profile | `desktop`（日常那个） | `v08-arm-c`（新建，由 desktop 复制） |
| MCP 入口 | `~/.local/share/operator-agent/v0.7.2/venv/bin/operator-agent-mcp` | `/Users/zh/op-agent-delivery/v08-venv/bin/operator-agent-mcp` |
| 源码 | tag `v0.7.2`（wheel 装进 site-packages） | 分支 `codex/v0.8`（worktree 可编辑安装） |
| `TOOL_VERSION` | 12 | 12（工具面相同 —— 两臂可比的前提） |
| 有 `phase=optimize` 的瓶颈诊断 | 否 | 是 |

两个 profile 的 `cordis.patch.yml` 都已核实可解析，指向的 console script 都在：

```bash
/tmp $ ~/.local/share/operator-agent/v0.7.2/venv/bin/operator-agent capabilities   # 工具契约 12
/tmp $ /Users/zh/op-agent-delivery/v08-venv/bin/operator-agent capabilities       # 工具契约 12
```

**切换**：DSH 的 profile 选择器 → `v08-arm-c`，重启会话。当前这个会话跑的是臂 B。

## 2. 每对的固定动作

**① 门（每对开跑前，跑一次）**

从分支 venv 跑，与两臂无关，避免把门的实现混进某一臂：

```bash
/Users/zh/op-agent-delivery/v08-venv/bin/operator-agent preflight \
    --backend ssh --host h20 --shape 4096,4096,4096 --reps 5 --repeat 8 \
    --planned-effect-pct 2.5 --json-out <evidence>/pair-N/gate.json
```

`action != "start"` ⇒ 这一对**不启动**、不计失败、不消耗模型或验证预算。等下一个窗口。

**② 两臂各跑一次任务**

```bash
# 各自独立的 run 目录，且在仓外
<arm-venv>/bin/operator-agent prepare \
    --request "fp16 4096x4096x4096 matmul, triton" \
    --shapes 4096,4096,4096 --hardware nvidia_gpu --dsl triton --precision fp16 \
    --min-speedup 1.0 --run-dir <evidence>/pair-N/arm-{B,C}/run
```

然后**宿主模型**驱动 `operator_context → verify` 直到终稿，最后跑一次独立终评。
每臂的模型预算是冻结计划里的 30 次调用 / 8192 单次输出 / 245760 累计预留。

**③ 顺序交替**

第 1 对：B → C；第 2 对：C → B。避免把机器漂移算到某一臂头上。

**④ 记什么**

每对 `<evidence>/pair-N/` 下必须有：`gate.json`、`arm-B/`、`arm-C/` 两个 run 目录、
两臂各自 `verify-*/result.json` 与其终评回执、以及一份 `pair.json`（写明终评 L3 的
`measurement_validity`、speedup、源码哈希）。

## 3. 判定（写死，不许临场改）

- 一对**可比**的条件：两臂终稿都过完整 L2，且终评 L3 都是 `valid`，且终评绑定的是各自提交的源码哈希。
- 主指标：两臂终稿 speedup 的**配对差值**。
- `degraded` 的读数只作观测，不进入差值；`uninformative` 直接让这一对记为不可判。
- 连续 **2** 对不可判 ⇒ 系列记 `unknown` 收工，不开第三对。

## 4. 还没落的东西

- `pair.json` 的比较器还没写成代码（`benchmarks/experiments/v08/r2_compare.py` 待补）。
  在此之前按第 3 节的规则人工判，且把人工判定的依据完整留在 `pair.json` 里。
- `pairs=2` 的预算已冻结；若第一对就判不出，按停止规则收工，不追加预算。