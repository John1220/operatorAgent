# 一期 A/B 自动执行与证据回放

当前提供可运行的 **离线演练**：A 臂使用独立基线生成/验证循环，B 臂复用 `operator_run` 的生产自主 worker。两臂按相同的预算政策运行，每格有独立工作目录、共享模型预算池与验证预留；最终源码另做一次独立验证。

本入口为 `python -m benchmarks.experiments.phase1`，仅支持 fake 模型与 fake 后端，零 GPU、零外部模型请求。B 的实际入口是生产 CLI 所用 worker 的适配器，**未经过 DSH 宿主或安装后的插件**。本轮不能记为插件收益或正式对照完成。

## 运行和回放

```sh
P1_REHEARSAL_ROOT=$(mktemp -d /tmp/operator-phase1.XXXXXX)

python -B -m benchmarks.experiments.phase1 freeze \
  --output "$P1_REHEARSAL_ROOT/frozen" \
  --scenario correctness-repair --max-rounds 2

python -B -m benchmarks.experiments.phase1 run \
  --plan "$P1_REHEARSAL_ROOT/frozen/plan.json" \
  --output "$P1_REHEARSAL_ROOT/execution"

python -B -m benchmarks.experiments.phase1 summary \
  --execution "$P1_REHEARSAL_ROOT/execution"
```

运行目录必须是仓外新目录，已有计划和执行目录不会覆盖。运行后直接阅读 `execution/summary.md`；`summary.json` 保存机器可读结果。`summary` 命令只读取留存证据，不调用模型、不重新执行候选、不补填缺失数据。

演练使用四个小型 fp32 fixture：matmul 8³、ReLU 8×16、matmul 16³、ReLU 4×8；顺序为任务 1/3 A→B、任务 2/4 B→A。它们与正式 `phase1-agent-comparison-v2` 的四个 fp16 真机任务分别编号，**不是正式八格的替代成绩**。

正常或修复场景中，两臂最终完整 L2 应通过；L3 不通过、性能读数不具备验收资格，均为预期。错误候选仍由真实 CPU 验证器拒绝，最终验证使用固定源码哈希与同一契约，开发故障注入不会在独立终评中重新注入。

## 预算、故障与取消

冻结时可设置 `--wall-clock-s`、`--max-llm-calls`、`--oracle-total`。默认每格为 30 秒、8 次逻辑模型调用、20 次按输入组计算的验证额度；只能在正式计划的 2400 秒、30 次调用、20 次验证以内调整。这里的验证额度不是 GPU launch 次数或 GPU 秒数。独立终评固定最多一次，默认另有 30 秒墙钟，不占开发调用池，其开销单独列出。

每格复用 `s5_shared_budget` 与生产模型账本，两臂同一配置，不在重试或角色切换时重置；验证使用 `BudgetLedger/BudgetPlan`，确认份额不能被开发借用。父进程负责墙钟和进程组清理，模型账本与验证账本分别归档。脚本回复只有逻辑预留与本地 fixture 完成事实，不写 provider dispatch、API usage 或虚构共享池完成事件。

可选场景与 [FakeBackend](fake-backend.md) 一致。测试取消时，先用 `--scenario timeout` 冻结并启动，在另一个终端执行：

```sh
python -B -m benchmarks.experiments.phase1 cancel \
  --execution "$P1_REHEARSAL_ROOT/execution"
```

取消写入绑定本次 invocation 的请求，由拥有 worker 的执行进程处理。当前格保留取消和清理回执，其余格保留 `not_started`；超时、失败和中断不会从汇总分母消失。结束后使用 `summary` 核对格位和 `cleanup`，无法证明的清理事实仍为 unknown。修改源码或量具后执行会拒绝旧计划，需要另起计划和目录；历史留存仍可只读回放。

## 汇总来源与未知值

每格保留 `cell-config.json`、`budget.sqlite`、`record.json`、开发源码与验证记录、`model-journal/`、`verification-budget.json`、`events.json`，以及独立终评的 `selected.py` 和 `final/`。汇总逐项核对模型预算上限、local/shared 请求哈希与预留、验证额度及确认份额、终稿/契约哈希和终评判定；缺记录、孤儿预留或不一致都限制预算可核结论。

事件来源须包含实际文件路径、SHA256 和记录定位，回放会核对原始字节。Agent 的代码修改、shell 与重试归 Agent，失败和重试均留存；B 臂的非委托候选开发动作单列绕过记录。初始任务/配置与运行期人工接管分开，不推断阅读、思考或人工分钟。

失败事件计数保留原始判定，因此也包含预期的 test_only L3 拒绝。查看具体 stage、工具回执与来源可区分数值错误、预算停止和测试证据不具备验收资格。

**当前适配器只覆盖其自身工具执行记录，尚未连接 DSH 会话和外部编辑观测源。** 因此人工介入可显示 `observed_count=0`，总数 `count=null`、覆盖不完整；不能把它解释为无人介入或节省人时。采集函数可以汇总有来源的人工事件，但接入真实来源仍是后续工作。

## 仍需接通的正式入口

后续需接入真实基线 Agent 与 DSH 插件臂的实际入口、共同模型/采样配置、宿主会话和编辑观测源，核对真实卡与量具后冻结正式任务。当前离线命令拒绝 live 模型、SSH 和继承的旧 S5 验证委托，避免把演练运行成真实资源调用。

本实现不发布 v0.7 工件，也不完成正式真机比较、支持清单或干净环境 DSH 安装验收。进度见[一期清单](../phase1/operator-development-phase1-execution-20261031.md)。

## 验证记录（2026-11-02）

`tests/unit/phase1`、仓库守卫及受影响的共享/内层/任务预算子集合计 **277 passed**；`tests/acceptance/phase1/test_rehearsal.py` **4 passed**。后者覆盖真实进程的八格正确性修复、首格取消保留其余七格、父进程截止回收，以及最后一格终评取消。八格两臂各四份独立 L2 通过，预算可核、provider dispatch 为零；回放结果稳定，人工总数保持 unknown。实际 CLI `freeze` 的场景、profile 与预算传递也已检查。这些均为离线 fixture 读数。
