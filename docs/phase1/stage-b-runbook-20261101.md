# 阶段 B 执行准备：预算冻结、控制变量、记录字段与 runbook（2026-11-01）

| 项目 | 说明 |
|---|---|
| 文档性质 | 一期 W3 产出：**执行前**的冻结计划，不是执行结果。本文没有任何新的真机读数 |
| 对应任务 | [一期清单](operator-development-phase1-execution-20261031.md) T2.5、T3.3、T3.4、T3.5、T3.6 |
| 上游判据 | [路线 §5.1 最小工作流](../design/operator-development-product-roadmap.md#L172-L180)、[§5.4 继续/转向/停止](../design/operator-development-product-roadmap.md#L207-L220)、[§6.2 公平对照](../design/operator-development-product-roadmap.md#L239-L246)、[§6.3 口径](../design/operator-development-product-roadmap.md#L248-L261) |
| 场景依据 | [阶段 A 台账 §4–§5](stage-a-ledger-20261031.md)：选定「目标硬件上对已有正确实现做性能确认与优化」 |
| 资源前提 | 制定时**零 GPU**；H20 与 P800 已下线。本计划等一张可用的卡，不预设日历 |

## 1. T2.5 预登记（跑之前定死，跑之后不改）

### 1.1 主评价问题

选**固定开发预算**（路线 §6.1 的问题 2）：预算结束时，比较两臂的达标任务数与最终性能。
不选「固定验收目标」的原因写在台账里——同条件 A/C 对照依赖已下线的 H20，那条主张永久不可补。

### 1.2 预算（当事人 2026-11-01 给定）

| 预算项 | 上限 | 落到的参数 | 单位说明 |
|---|---|---|---|
| 模型调用 | **每任务 ≤30 次** | `--max-llm-calls 30`（CLI 默认 24，必须显式改） | 调用次数，不是 token |
| 墙钟 | **每任务 ≤40 分钟 = 2400 s** | `--wall-clock-s 2400`（CLI 默认 1800） | **开发预算**，含排队与等待；不是 GPU 时间上限，也不是 kernel 耗时（路线 §5.4 明令不许混） |
| 真值 oracle 次数 | 20 次（沿用 CLI 默认 `--oracle-total 20`） | 默认值 | 与上面两项分列，不可互相挪用 |
| 本轮规模 | 4 个任务 × 2 个臂 = 8 格；**无预登记未执行格**（派发前逐格登记） | — | 全 8 格终态都要有归宿（成功/失败/未知） |

### 1.3 达标定义（预先约定）

一个任务在**该臂预算内**同时满足：① 完整 L2 通过（全部 case + 输入保真 + 确定性检查）；② 至少一次 `measurement_validity=valid` 的 L3 读数且不低于约定回退约束；③ 终稿源码哈希与判定绑定一致。
只满足 ① 记「正确候选」，不记达标（路线 §1 的三档验收层次）。

### 1.4 收益判定与可检出单位

| 结果 | 判定 |
|---|---|
| 插件臂达标数 − 基线臂达标数 ≥ **+1**，且额外成本（§3 记录）不抵消改善 | 有收益，进入 C 的候选功能评估 |
| 两臂达标数相同 | 无净收益；除非最终性能有可复核差异，否则停止扩大该工作流投入 |
| 插件臂更少 | 转向信号，按路线 §5.4 收缩 |

**+1 是这套设计的最小可检出单位**（每臂 4 格），不是统计显著性——写在这里防止把 4 格差异读成结论。要更细的分辨力必须**先加格子再跑**，不许跑完追加。

### 1.5 重复方式与停止规则

- 每臂每任务 **1 格**，预算用尽即停；同任务再跑一次要作为新格登记，不复用旧终态。
- 停止条件（任一）：预算耗尽 / 设施中断 / 迁移门未满足（§4 前置）/ 连续两轮在 roofline 效率与运行时间上都无有意义改善。
- **不得**为过门槛临时追加预算、改达标定义或删除任务；需要新一轮就写新计划、用新身份。

## 2. T3.3 控制变量清单（逐项固定，差异先写后跑）

| 变量 | 固定方式 | 允许的跨臂差异 |
|---|---|---|
| 模型与采样 | 同一 provider、同一 model id、同一 `reasoning effort`（S5 教训：8192 输出预算会被内部推理吃满） | 无 |
| 硬件 | 同一张卡、同一 UUID（运行时探测，不写死），共享卡须留 `peers` 观测 | 无 |
| 任务输入 | 同一 `--shapes`、同一 dtype/stride 声明、同一参考实现文件哈希 | 无 |
| 起始代码 | 同一 seed 候选（SHA256 记录在案） | 无 |
| 预算 | §1.2 三档 | 无 |
| 工具面 | 基线臂 = 当事人实际在用的 Agent + 常用脚本，**保留其正常验证与执行能力**（不许削成残废基线）；插件臂走 §4 命令序列 | 仅入口不同 |
| 文件系统可见性 | 隔离声明：基线臂若可读仓，必须在计划里写明「隔离未建立」，不得称其为裸环境（S5 已因此返工） | 记录差异，不隐瞒 |

## 3. T3.4 绕过与额外成本记录字段

路线 §6.2 要求计入「插件发现、工具调用成功与否、用户绕过工具、额外配置与纠错成本」。落成一张表，每格一行，**缺项写 unknown，不补 0**：

| 字段 | 取值 | 来源 |
|---|---|---|
| `discovery_ok` | yes/no/unknown | `operator-agent capabilities` 退出码 + 回执 |
| `tool_calls_attempted` / `succeeded` / `refused` | 整数 | `tool-calls/<id>/run-status.json`；refusal 有独立字段（发布形态 §1 第 5 项的可修正回显） |
| `bypassed_tool` | 工具名列表 | 人工填报：本应调用却改用手上脚本/ssh 的情形 |
| `config_fix_minutes` | 自报量级档 | `core/human_ledger.py` 的 `minutes_band`（T3.7），只准填档不准填分钟 |
| `human_involved_stages` | 六环节四元组 | 同上；与机器秒**分列**，报表不许折叠成合计 |
| `machine_wall_s` | 秒 | `execution` 记录；注明含启动与清理，不当性能指标 |
| `outcome` | pass/fail/unknown + 原因 | 终态；基础设施中断记 infra/hold，不进搜索迭代 |

一条硬规矩：**`bypassed_tool` 非空的格，不得算作「插件工作流的收益」**——L6 那次会话里 `operator_run`/`repair` 用了 0 次、45 次 `bash` 手搓赢了插件，正是这个字段要抓的情形。

## 4. T3.5 runbook（拿到卡当天可直接执行）

前置未满足就**停在第 0 步并报告**，不许静默降级继续跑（[可迁移性 G1–G5](../design/dsh-operator-portability-4090.md#L85-L126)）：

```sh
# 0. 迁移门（T3.2 只落了探针形状；五道门任一不过即停）
python3 -B -m benchmarks.probes.unknown_card_gates --selftest
#    ↑ 零机时自检：证明「无读数 ⇒ 五门全 unknown」「继承值 ⇒ 不得 pass」两条反例是活的
#      真跑前先在卡上产出一份探测 JSON（device / launch_floor / shared_memory / peak /
#      noise / shapes，每块都要带 source_uuid），再喂给门：
python3 -B -m benchmarks.probes.unknown_card_gates --from-json ./card-probe.json
#    退出码非 0（有 fail 或 unknown）⇒ 停在这里，不开跑；不静默降级
operator-agent capabilities --json-out ./cap.json

# 1. 冻契约（每格一个新 run-dir，调用方拥有布局）
operator-agent prepare --declaration ./task-01.json --run-dir ./b/01 \
  --shapes "<shape>" --hardware "<探测值>" --dsl triton --precision fp16 --min-speedup 1.0

# 2. 基线臂：当事人实际在用的 Agent + 常用脚本（保留验证与执行能力，全程记 §3 字段）

# 3. 插件臂：整段委托（预算必须显式改，CLI 默认 24 次 / 1800 s）
operator-agent run --declaration ./task-01.json --run-dir ./b/01 \
  --backend ssh --host <别名> --persistent \
  --max-llm-calls 30 --wall-clock-s 2400 --oracle-total 20 --json-out ./b/01/run.json

# 4. 中途查询与取消（取消回执不等于终态）
operator-agent status --run-dir ./b/01
operator-agent cancel --run-dir ./b/01 --invocation-id <id> && operator-agent status --run-dir ./b/01

# 5. 独立终评（不复用开发期判定；对终稿源码重新 verify）
operator-agent verify --run-dir ./b/01 --code-file <终稿> --stop-after L3 \
  --backend ssh --host <别名> --persistent --json-out ./b/01/final.json

# 6. 填人时表（T3.7 的 CLI 入口；没问过的格子留 unknown，量级档与秒分列）
operator-agent ledger-new    --run-dir ./b/01
operator-agent ledger-fill   --run-dir ./b/01 --stage analyze --involved yes \
  --action-taken "读 L3 回执判是否值得再跑" --reason "门槛在 ±3% 刀刃上" --minutes-band 5-15
operator-agent ledger-report --run-dir ./b/01 --machine-wall-json ./b/01/wall.json
```

预期读数与停止条件：

| 步 | 期望 | 不满足时 |
|---|---|---|
| 0 | 探针 `--selftest` 退 0（两条反例活着）；`--from-json` 对真实探测输出应五门全 pass；`capabilities` 无静默默认值 | 任一门 fail 或 unknown ⇒ 停止并记录，不开跑。**这道门的存在理由**：`measurement.assess()` 缺地板时仍给 `VALID`，未标定的卡会把小形状读数当成可裁决读数，而禁用配方是永久决策 |
| 1 | `prepare` 产出契约 + 输入冻结，`--hardware` 来自探测而非仓内常量 | 停止（这是 G1/G2 的坑位） |
| 3 | 终态 `done/failed/cancelled`；`lost` 必须能说清是进程没了还是结果未知 | 该格记未知，不重跑覆盖 |
| 3 后 | `cleanup.remote_execution ∈ {quiescent, not_started, unknown}`；本机回收与远端停止分开报 | 未证 quiescent 就写未证，不当成功 |
| 5 | 终评与开发期判定身份一致（候选哈希绑定） | 不一致即该格作废，不折算收益 |

GPU 型号、UUID、free memory 一律运行时探测入档；共享卡快照不是独占租据，`peers` 观测必须留下。

## 5. T3.6 evaluator 身份重绑（本次实算）

分层与分目录后，各 workload 的 evaluator 身份已变；下面四条摘要在 `HEAD = 34e8166` 上由 `digest()` 实算，**自本日起算**：

| 身份 | 条目数 | 摘要 |
|---|---:|---|
| `workload_decision_h20.evaluator_sources()` | 121 | `1570599f1045af2c80970950849717c38ceea22a039b88f6375aaa57bd9f2ede` |
| `workload_codegen_h20.sources()` | 124 | `3d76e9f0d0d21ffdc0a62ba98647020760f7a27119de7b054bc0830d4a6a854c` |
| `workload_llm_search_h20.sources()` | 124 | `62b5c1e698e37e476ac15bb30a0f96d1dc0f9776b536f4fd810ea037cc2a81c6` |
| `quality_evaluation.evaluator_identity()` | — | `4603b2514b64f896536947cde2fc2e3aa13940ce9f180163d0abed161625a095` |

协议身份本身的历史摘要**不是只有一个**——从留存记录里反算出来是三代，且与在盘 wheel 那一代互不相等：

| 摘要 | 条目 | 留存位置 |
|---|---:|---|
| `fdb76a11e04e75c6…` | 64 | `20261026-s2` 的 1 份记录 |
| `9914985a98d6efe7…` | 70 | `20261027-s4` 的 6 份记录 |
| `78993c983003b9a8…` | 73 | 在盘 `0.6.1` wheel 解包复算 |
| `e8c7f2c6ffa99576…` | 73（带层键） | 当前 HEAD 与 `0.6.3`/`0.6.2` wheel 解包 |

三条规则：

1. **旧批次不得与新身份混表报数**，也不能声称「接着上轮跑」；引用旧读数时按上表标明它属于哪一代。
2. 本次实测：**证据树里没有任何留存 JSON 带 `evaluator_sources` 字段**（`grep` 全树为 0），只有 `measurement_protocol` 7 份、`protocol_hash` 18 份。所以旧 evaluator 身份**不可复算**，只能靠记录里留存的摘要字符串比对；缺摘要就记「不可核」，不倒推。
3. 「73 文件」这个数在 S5/S6 与发布形态里指的不是同一代清单（上表为证）。以后写「协议身份 N 文件」必须同时给摘要，否则等于没写。

## 6. 本文没有做的事

- 没有跑任何真机或模型；没有新增能力（剖析、形状扫描、多候选裁决工具仍为零，见一期清单 §6 不做项）。
- 没有替 T3.1 选卡。**T3.2 只落了探针形状**（`benchmarks/probes/unknown_card_gates.py`，七行变异表 7/7 拦住、26 条用例全绿），**五道门一条读数都没有**——那些必须来自真实卡上的探测输出，本文不猜；探针的全部作用就是让继承值与缺读数不许变成绿。
- 预算与门槛的数值取自当事人 2026-11-01 的指示；改动它们需要新计划与新身份，不在本文授权内。

## 修订记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-11-01 | 创建：T2.5 预登记（固定预算问题 + 每任务 ≤30 调用 / ≤2400 s 墙钟 + 4×2 设计 + +1 最小可检出单位）、T3.3 控制变量表、T3.4 绕过与额外成本字段（含「bypass 非空不算收益」硬规矩）、T3.5 七步 runbook（命令与 CLI 默认值实测核对）、T3.6 四代协议身份与四条新 evaluator 摘要 | `operator_agent/app/cli.py:39-145` 参数实读；`digest()` 实算四条 evaluator 身份与三代协议摘要；证据树 `grep` 全树查 `evaluator_sources`（0 命中）与 `measurement_protocol`（7 命中） |
