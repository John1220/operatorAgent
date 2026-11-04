# 一期模型、SSH 与 DSH 入口接入

本入口用于检查 A/B 执行链路。A 是独立的模型生成→普通验证→反馈修正循环；B 可显式选择生产自主 worker（`core`）或已安装 DSH + 当前源码插件（`dsh`）。不会自动切换入口。计划 ID 是 `phase1-entry-integration-v1`，与正式四任务八格 `phase1-agent-comparison-v2` 分开。

`freeze-connected` 只读取配置和本地身份，不读模型凭据、不访问模型或 GPU。`run-connected` 才会调用配置指定的模型及后端。默认 B 入口为 `dsh`；需要 macOS Seatbelt、Node 和已安装 DSH，缺失时明确失败。此实现复用 S5 的网关、DSH 启动及源码插件挂载方式；不等于干净安装工件的验收。

## 配置与执行

将如下示例保存到仓外的 `connected-config.json`；替换模型地址、ID 和连接信息。凭据只写环境变量名，不能将密钥直接放入 JSON。模型需支持 `/v1/chat/completions`。示例是单任务接入检查；冻结正式对照前仍需核定四任务、实际请求采样参数和完整身份。

```json
{
  "backend": "ssh",
  "entry_b": "dsh",
  "model": {
    "base_url": "https://model.example/v1",
    "id": "replace-with-model-id",
    "api_key_env": "PHASE1_MODEL_KEY",
    "temperature": 0.2,
    "transport": "live"
  },
  "connection": {
    "host": "gpu-host",
    "port": 22,
    "user": "worker",
    "work_dir": "/opt/operator-runtime",
    "timeout": 300
  },
  "gpu": {"index": 0, "uuid": "GPU-replace-from-probe", "name": "replace-from-probe"},
  "probe": "/absolute/path/card-probe.json",
  "tasks": [{
    "task_id": "matmul-check",
    "declaration": {"request": "matmul fp32", "precision": "fp32", "shapes": [[256, 256, 256]], "dsl": "triton"}
  }],
  "budget": {"wall_clock_s": 2400, "max_llm_calls": 30, "oracle_total": 20},
  "max_rounds": 2,
  "final_wall_clock_s": 120
}
```

SSH 使用**已准备的专用远端运行时目录**，其中 `.venv/bin/python` 及 runner 文件必须符合当前源码身份。执行期间启用 `OPERATOR_AGENT_EXISTING_REMOTE_ONLY=1`：缺环境或哈希不符会停止，需要在正式冻结前完成环境准备。不会自动租卡或安装、覆盖 runner。A、B 与终评串行复用该运行时，本地候选、工作目录、证据和终评分别保存。租卡型号变化后必须重新采集探针并冻结新计划；探针五门全部通过、UUID 一致后才接受 SSH 配置，每个执行器还会核对 index/UUID/name。

```sh
python -B -m benchmarks.experiments.phase1 freeze-connected \
  --config /absolute/path/connected-config.json \
  --output /absolute/path/new-frozen-plan

# 在运行环境提供 PHASE1_MODEL_KEY；下面命令会实际使用模型及指定 SSH 实例。
python -B -m benchmarks.experiments.phase1 run-connected \
  --plan /absolute/path/new-frozen-plan/plan.json \
  --output /absolute/path/new-execution

python -B -m benchmarks.experiments.phase1 summary --execution /absolute/path/new-execution
python -B -m benchmarks.experiments.phase1 cancel --execution /absolute/path/new-execution
```

目录必须是仓外的新目录。冻结后实现、DSH 入口、探针变化会拒绝继续派发，需要重新冻结；只读 `summary` 不要求旧实现仍在当前树，也不需要模型凭据。

本地功能检查可改为 `backend=fake`、`dsl=python`，删除 `connection/gpu/probe`。模型端仍有两种明确配置：实际模型用 `transport=live`；本地测试服务用 `transport=local_fixture` 且地址必须为 `http://127.0.0.1:端口/v1`。Fake 后端或 fixture 模型都会永久标记 `test_only`。完全不发模型请求的检查沿用[离线演练](phase1-rehearsal.md)。

## 预算、证据与结论边界

- 每格同一 SQLite 预算池预留模型调用和输出额度。B 的 DSH 外层网关与内层生产账本共用该池；失败、未完成请求不返还额度。MCP 只暴露能力、一次 `operator_run` 及该调用的状态/取消，任务、后端、预算由父进程注入，不能通过重复运行重置验证预算。
- 每格由父进程控制截止、取消和进程组回收。终稿绑定源码与契约哈希，在独立目录进行一次普通验证，另受终评截止控制。原始失败、取消和未执行格位保留。
- 模型账本保存实际请求的模型、采样参数、输出上限和请求摘要；DSH 外层请求另存。内层按角色使用采样参数，DSH 也可能省略参数，因此当前 `sampling_comparability=not_established`，不能据此声称完成同口径正式收益对照。
- API usage 仅计已校验的服务端计数；缺失保持 unknown，调用预留不能当作计费调用数，费用保持 unknown。现有 DSH 外层流式网关未提供 usage 对账，不能用内层读数补齐。密钥只留在可信 worker 的运行环境，DSH 子进程仅取得本地网关占位凭据和本格 MCP capability。
- 保存 DSH 原始输出、MCP 请求/回复和来源摘要；握手、工具发现、实际委派及 Agent 终态轮询都有记录才可标记入口已观察到。会话中的人工接管及外部编辑尚无完整来源，人工总数与分钟仍为 unknown。工具观察范围在汇总中单列。

`summary.json` 输出统一模型预算、验证预留、独立完整 L2、观测事件和缺失项。不将接入检查升级为正式对照、GPU 性能资格或已安装插件验收。实际 SSH 的逐项 L3 原始结果可供复查，版本验收仍按一期清单执行。

## v0.7 候选支持范围

| 项目 | 当前范围 |
|---|---|
| 算子 / 精度 | Matmul、ReLU、Softmax、LayerNorm；FP16 / FP32 |
| 真机源代码 / 后端 | Triton / 显式 SSH，按租赁实例检测并固定 GPU 身份 |
| 离线源代码 / 后端 | CPU Python / FakeBackend，test_only |
| LayerNorm | 非 affine、最后一轴、eps=1e-5；FP16 使用 calibrated v2，输入只有 `X` |
| 布局 | 完整内建 L2，包括派生的 strided case；每格一组声明形状 |
| 运行时 | 当前源码 runner、Python/Torch/Triton/CUDA 的实际实例组合另行验收 |

此表是候选范围，尚未成为全部组合的真机发布承诺。旧式未声明 profile 的 affine LayerNorm 契约保留兼容，不属于本接入范围。

> **2026-11-04 起冻结版见 [v0.7 支持清单](v07-support.md)**：那一份逐行标了证据等级（真机 / 仅契约 / 仅离线 / 不支持），并写明失效条件。本节的表作为候选阶段的快照保留，不再作为边界依据。

## 本地验证

后续开发与验证默认使用 **Python 3.12**（仓库根目录 `.python-version`）。当前本机默认 `python` 为 3.12.16。下面的 3.9.12 读数保留为当时实际使用的环境记录，尚未由 3.12 重跑替代。

新增测试用本地 HTTP 服务经过真实 `LLMClient`、共享预算、受管子进程和普通 CPU 验证执行 A/B，检查独立终评、只读回放、取消、失败及凭据不落盘。MCP 单元测试经过真实生产 handler，核对冻结参数、调用归属与 DSH 配置构造；实际宿主由下方集成用例覆盖。

2026-11-03，使用 `~/opt/miniconda3/bin/python`（Python 3.9.12）验证：核心声明/契约修正后的 `tests/unit tests/hygiene` 为 **5948 passed / 1 skipped**，275.83 秒；skip 为需要真实 CUDA 的用例。随后接入层补强的最终范围 `tests/unit/phase1 tests/hygiene tests/acceptance/phase1` 为 **349 passed**，58.46 秒，没有重跑无关的发布全量。

最终范围包含一次**实际已安装 DSH 会话**：使用本地假 HTTP 模型，通过真实插件/MCP 完成握手、发现、单次委派、终态轮询和独立完整 L2；A/B 模型预算均可核，DSH 外层与内层确实共用额度。该检查证明本地接线可用，模型是 fixture、后端是 fake，仍不计真 GPU、付费模型对照或干净安装工件验收。

本轮临时证据位于 `/private/tmp/operator-phase1-core-20261103.{log,xml}`、`/private/tmp/operator-phase1-final-20261103.{log,xml}`，解释器/身份/日志摘要清单为 `/private/tmp/operator-phase1-validation-20261103.json`。完整单元回归覆盖相同的核心协议树；最后的接入实现身份由最终子集绑定。这些是本机临时验证记录，尚未进入正式交付证据归档。
