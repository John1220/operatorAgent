# 无 GPU 的组件功能检查

`FakeBackend` 是显式选择的离线后端，不检查本机 CUDA、不连接 SSH，也不根据本机是否有 GPU 切换路径。它实现现有 `Backend` 接口，接入正常的 Conductor、输入回读、验证、预算、归档、状态与取消流程。

当前实现可从源码树使用，工具契约版本为 `10`，尚未打包发布。脚本模型支持 **ReLU、matmul**；候选使用 CPU Python/NumPy，不能用它检查 Triton/CUDA 编译或 GPU 数值正确性。

## 运行一次

使用项目的 Python 环境（需安装 NumPy），把运行目录放在仓库外：

```sh
FAKE_RUN_ROOT=$(mktemp -d /tmp/operator-agent-fake.XXXXXX)
python -m operator_agent run \
  --backend fake --dsl python --request relu --shapes 8,16 \
  --run-dir "$FAKE_RUN_ROOT/run" \
  --wall-clock-s 60 --max-llm-calls 8 --oracle-total 20

python -m operator_agent status --run-dir "$FAKE_RUN_ROOT/run"
```

`run` 异步返回 `invocation_id`，继续用 `status` 查终态。默认使用 `fake` 模型，无需模型地址或密钥，不读取密钥文件、不发送 API 请求；已有 LLM 环境变量也不会把这条运行自动变成真实调用。

正常运行的预期结果：

```json
{
  "phase": "done",
  "execution_context": {"mode": "fake", "test_only": true, "gpu_acceptance": false},
  "summary": {
    "levels": {"L1": true, "L2": true, "L3": false},
    "all_passed": false,
    "validity": "uninformative"
  }
}
```

这里 L1/L2 表示 **CPU 候选**通过了现有语法、接口和完整正确性套件；L3 明确不具备 GPU 验收资格。`phase=done` 表示任务结束，不能据此声称 GPU 算子交付成功。日志中 L3 的 `FAILED` 与 `all_passed=false` 是预期行为。

## 场景与硬件规格

`--fake-scenario` 默认 `success`：

| 场景 | 注入与预期行为 |
|---|---|
| `success` | 正常生成、CPU 执行、完整 L2 和归档；L3 保持测试证据 |
| `compile-repair` | 首份源码编译失败；相同源码重复尝试仍失败，Fixer 改源码后恢复。使用 `--max-rounds 2` |
| `correctness-repair` | 首份源码的输出被注入数值错误；真实 L2 失败，修复后的新源码再走完整 L2。使用 `--max-rounds 2` |
| `no-improvement` | 候选与基线固定为 1 ms，检查相同观测的处理；不能用它宣称已验证真实优化收敛，测试读数不累计有效停滞 |
| `timeout` | 编译调用持续阻塞，交由真实监督进程按任务墙钟终止，或由用户取消 |
| `transport-unknown` | 返回“执行结果未知”，L2 不通过，不运行 L3；没有实际网络请求 |
| `hardware-missing` | 硬件信息读取抛错，任务失败，不执行候选 |

例如检查自动修复：

```sh
python -m operator_agent run \
  --backend fake --fake-scenario correctness-repair \
  --dsl python --request relu --shapes 8,16 \
  --max-rounds 2 --run-dir "$FAKE_RUN_ROOT/repair"
```

`--fake-profile small|large|unknown` 提供声明式规格。`small`/`large` 的每 SM 共享内存为 48/228 KiB，用于检查规划和资源适配；`unknown` 缺失这项信息，让现有流程暴露未知或拒绝。它们不对应真实卡型，不伪造 GPU UUID、驱动或真实性能。换 profile 会改变后端指纹。

验证预算可以直接缩小现有 `--max-llm-calls`、`--oracle-total`、`--wall-clock-s`，没有另造一套 fake 预算。脚本回复占用逻辑模型调用预留，但账本里没有 provider dispatch/API usage，费用记录为零、来源为 `test_fixture`。

取消复用现有入口：

```sh
python -m operator_agent cancel --run-dir "$FAKE_RUN_ROOT/run" --invocation-id <run返回的ID>
python -m operator_agent status --run-dir "$FAKE_RUN_ROOT/run"
```

取消场景可先用 `--fake-scenario timeout` 启动另一份新目录，在 `fake-events.jsonl` 出现 `blocked` 后取消。核对终态 `cancelled` 与 `cleanup.worker_reaped`；清理无法证明的字段仍保留 unknown。

## Fake GPU 配真实模型

已有正常模型配置时，显式添加 `--llm-mode live`：

```sh
python -m operator_agent run \
  --backend fake --llm-mode live --dsl python \
  --request relu --shapes 8,16 --max-llm-calls 8 \
  --run-dir "$FAKE_RUN_ROOT/live-model"
```

这会调用现有 `LLMClient`，产生真实模型费用，仍执行 CPU Python 候选并保留 test_only 标记。NumPy/math 以外的 import 和动态代码/IO 形式会被后端拒绝，不能借此直接运行模型生成的 Torch/Triton/CUDA 代码。现有 `exec_guard` 提供进程和资源约束，不是敌意代码的安全沙箱。

MCP 的 `operator_run` 同样支持：

```json
{
  "request": "relu",
  "shapes": [[8, 16]],
  "dsl": "python",
  "backend": "fake",
  "fake": {"scenario": "compile-repair", "profile": "small", "llm": "fake"},
  "max_rounds": 2,
  "run_dir": "/tmp/operator-agent-fake-example"
}
```

CLI/MCP 的 `verify`、`repair` 也可选择 `backend=fake`，使用默认硬件与场景；`verify` 检查给定 Python 源码，`repair` 沿用真实模型客户端。场景和脚本模型选项属于自主 `run`。发布工件和 DSH 宿主安装后的行为仍需单独验收，源码树检查不代替它们。

## 结果隔离与租卡前后的分工

自主运行目录保存 `autonomous.json`、`run-status.json`、`conductor.json`、`model-journal/`、`fake-events.jsonl`、候选、验证记录及 `output/` 内的代码和报告。报告开头显示 `FAKE / test_only`；启动、运行、异常、超时、取消和完成结果均保留 `execution_context`。

fake Conductor 要求显式 `run_root`，使用独立的 `fake-kernel_db.jsonl`、`fake-failure_db.jsonl`、日志与产物目录，不接收外部语料库实例。性能读数不能获得验收、调优、进度或生产入库资格，即使固定加速比达到门槛，也不把 L3 判为通过。正向语料不会因这次 fake 运行入库。

建议日常先跑 fake 模型和 fake 后端，检查组件功能；需要核对模型接线时显式启用真实模型。租到 GPU 后再用 `ssh` 后端读取实际硬件/运行时、检查资源限制，验证 Triton/CUDA 编译、GPU 正确性和同卡性能。不同卡型的成绩按各自设备身份留存，fake 数值不能补齐一期对照实验或真机验收。

相关实现回归为 `tests/unit/fake_backend/` 与 `tests/acceptance/fake_backend/`，覆盖实际候选执行、修复、证据标记、隔离、模型预算、墙钟、取消，以及传输/硬件失败。

## 本次验证记录（2026-11-02）

执行 `python -B -m pytest tests/unit tests/hygiene tests/acceptance/fake_backend -q`：**5718 passed、1 skipped**，其中 fake 专项 52 项。另在清空模型配置后通过实际 CLI 启动 ReLU 任务并查询终态：`done`，L1/L2 通过，L3 不通过，`validity=uninformative`、`test_only=true`。这些记录仅证明源码树的离线组件检查，不是发布工件或真机 GPU 验收。
