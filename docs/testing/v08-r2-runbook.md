# R2 跑法：两臂两个 profile

2026-11-07 · 本轮用 `operator-v08-increment-r2-fp32-v2`，任务为 FP32 Triton Matmul 4096³。
FP16 首对是单独计划的试跑，证据保留在原目录。见 [预登记](v08-r2-preregistration.md)。

## 1. 环境

| | 臂 B（对照） | 臂 C（增量） |
|---|---|---|
| DSH profile | `desktop`（日常那个） | `v08-arm-c`（新建，由 desktop 复制） |
| MCP 入口 | `~/.local/share/operator-agent/v0.7.2/venv/bin/operator-agent-mcp` | `$HOME/op-agent-delivery/v08-venv/bin/operator-agent-mcp` |
| 源码 | tag `v0.7.2`（wheel 装进 site-packages） | 分支 `codex/v0.8`（worktree 可编辑安装） |
| `TOOL_VERSION` | 12 | 12（工具面相同 —— 两臂可比的前提） |
| 有 `phase=optimize` 的瓶颈诊断 | 否 | 是 |

两个 profile 的 `cordis.patch.yml` 都已核实可解析，指向的 console script 都在：

```bash
/tmp $ ~/.local/share/operator-agent/v0.7.2/venv/bin/operator-agent capabilities   # 工具契约 12
/tmp $ $HOME/op-agent-delivery/v08-venv/bin/operator-agent capabilities       # 工具契约 12
```

在 DSH 中分别选择 `desktop` 和 `v08-arm-c`，每臂开一个全新会话。开跑前核对两边的实际
MCP 入口、工具契约、宿主版本、provider 和模型。两臂使用同一用户任务及起点，各自只读取本臂的
历史和回执。把这些身份与采样参数的观测状态随回执保存。

## 2. 每对的固定动作

**① 门（每对开跑前，跑一次）**

从分支 venv 跑，与两臂无关，避免把门的实现混进某一臂：

先从 H20 记录目标卡的索引、UUID 和设备名。在探针进程与两个 profile 的 MCP worker 环境中
设置同一组 `OPERATOR_AGENT_REQUIRED_GPU`、`OPERATOR_AGENT_REQUIRED_GPU_UUID` 和
`OPERATOR_AGENT_REQUIRED_GPU_NAME`。按回执核对实际设备，无法绑定该卡时暂停该对。

```bash
# 先将 r2_pair_dir 设为本次 FP32 v2 配对的独立证据目录。
mkdir -p "$r2_pair_dir"
"$HOME/op-agent-delivery/v08-venv/bin/operator-agent" preflight \
    --backend ssh --host h20 --shape 4096,4096,4096 --dtype fp32 \
    --reps 5 --repeat 8 --planned-effect-pct 2.5 --json-out "$r2_pair_dir/gate.json"
```

同时核对 `action == "start"`、通道 `matmul-4096x4096x4096-fp32-cuda-event` 和
`effective_band_pct ≤ 0.25`（2.5% 登记效应的 1/10）。任一条件不满足就保留回执、等待下一窗口。
未启动的窗口不计为一次配对。

**② 两臂各跑一次任务**

从原生会话调用 `operator_prepare`，两臂只替换 `run_dir`，分别指向各自的仓外目录。

```json
{
  "request": "fp32 4096x4096x4096 matmul, triton",
  "shapes": [[4096, 4096, 4096]],
  "hardware": "nvidia_gpu",
  "dsl": "triton",
  "precision": "fp32",
  "min_speedup": 1.0,
  "run_dir": "<pair-dir>/arm-B/run",
  "host_budget": {
    "wall_clock_s": 2400,
    "max_verifications": 8,
    "max_model_calls": 30,
    "max_output_tokens_per_call": 8192,
    "total_output_token_reservations": 245760
  }
}
```

在首次验证前，核对两臂的 `host-task-policy.json` 与 `host-model-budget.json`、DSH 已绑定任务，
以及验证额度中已保留一次终评。计划中的 `gpu_minutes: 45` 单独记录，宿主工具只冻结上述五项。
宿主模型驱动 `operator_context → operator_verify`，两臂串行使用 `backend=ssh`、`host=h20`
和同一块 GPU。C 臂记录与候选轮次绑定的 `phase=optimize` 诊断。两臂分别用
`verification_purpose: final` 验证选中的同一份源码，再核对终评绑定的源码哈希。

**③ 顺序交替**

FP32 v2 第 1 对按 B → C，第 2 对按 C → B，降低机器漂移对单臂的影响。

**④ 记什么**

每对 `<evidence>/pair-N/` 下必须有：`gate.json`、`arm-B/`、`arm-C/` 两个 run 目录、
两臂各自 `verify-*/result.json` 与其终评回执、以及一份 `pair.json`（写明终评 L3 的
`measurement_validity`、speedup、源码哈希）。同时保留计划哈希、GPU 身份、两臂的预算回执
和模型用量，区分预留与实际消耗。

### attention 对的额外动作

门的命令按计划里那条（`--op attention --shape 1,16,4096,128 --dtype fp16`，**不带** `--causal`）：

```bash
"$HOME/op-agent-delivery/v08-venv/bin/operator-agent" preflight \
    --backend ssh --host h20 --op attention --shape 1,16,4096,128 --dtype fp16 \
    --reps 5 --repeat 8 --planned-effect-pct 2.5 --json-out "$r2_pair_dir/gate.json"
```

核对 `action == "start"` 与通道 `attention-1x16x4096x128-fp16-nocausal-cuda-event`。
（S=1024 的核只有 96 µs、launch 地板占 17.3% ⇒ 每次配对结构性不可判，那份计划已按名字撤回。）
两臂的 runner 必须是**装了 `torch==2.14.0`** 的那个（A1 起部署会自己钉版本并回读核对；
身份门对 attention 与 layernorm v2 一视同仁）。任务声明用
`fp16 attention prefill 1x16x1024x128`，契约里应看到 `max_error=3e-4`、
`correctness_profile=builtin-attention-fp16-cuda-v1` 与 8 条 case。

## 3. 判定（写死，不许临场改）

- 一对**可比**的条件：两臂终稿都过完整 L2，且终评 L3 都是 `valid`，且终评绑定的是各自提交的源码哈希。
- 主指标：两臂终稿 speedup 的**配对差值**。
- `degraded` 的读数只作观测，不进入差值；`uninformative` 直接让这一对记为不可判。
- 连续 **2** 对不可判 ⇒ 系列记 `unknown` 收工，不开第三对。

## 4. 还没落的东西

- 比较器已有代码：`benchmarks/experiments/v08/r2_compare.py`。它要求两臂都是
  `verification_purpose=final` 的回执、L2 通过、**源码哈希按文件重算**与回执一致，
  并在两臂 L3 都 `valid` 时才算可比；`degraded` 不给差值，连续两对不可判即 `stop`。
  用法：
  `python -m benchmarks.experiments.v08.r2_compare --plan <plan.json> --arm B=<final.json>
   --arm C=<final.json> --gate <gate.json> --json-out <pair-1.json>`
- `pairs=2` 的预算已冻结。达到预算或连续两对不可判时收工。
