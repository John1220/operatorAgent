# 阶段 B 执行准备：预算冻结、控制变量、记录字段与 runbook（2026-11-01）

| 项目 | 说明 |
|---|---|
| 文档性质 | 一期 W3 执行计划与试跑记录；§0 为当前修订方案，§7.1–§7.4 为旧轮真机事实，不能混作新轮结果 |
| 所属产品版本 | v0.7 基础可用版；本 runbook 是该版对照与任务执行子计划，完成实验不等于完成可用版本交付 |
| 对应任务 | [一期清单](operator-development-phase1-execution-20261031.md) T2.5、T3.3、T3.4、T3.5、T3.6 |
| 上游判据 | [路线 §5.1 最小工作流](../design/operator-development-product-roadmap.md#L172-L180)、[§5.4 继续/转向/停止](../design/operator-development-product-roadmap.md#L207-L220)、[§6.2 公平对照](../design/operator-development-product-roadmap.md#L239-L246)、[§6.3 口径](../design/operator-development-product-roadmap.md#L248-L261) |
| 场景依据 | [阶段 A 台账 §4–§5](stage-a-ledger-20261031.md)：选定「目标硬件上对已有正确实现做性能确认与优化」 |
| 资源前提 | 制定时**零 GPU**；H20 与 P800 已下线。本计划等一张可用的卡，不预设日历 |

## 0. 当前方案：Agent 自动对照与事件记录（2026-11-01 修订）

用户已要求取消为评估而新增的人工执行、逐格填表工作。**两臂均由 Agent 开发、执行和记录；人工表不再是阶段 A 出口或对照开跑条件。** 以下修订优先于本文旧的填表流程。修订方案编号 `phase1-agent-comparison-v2`，目前是**待正式入口接线、待最终冻结的计划**。2026-11-02 已实现[离线编排与汇总演练](../testing/phase1-rehearsal.md)，不等于正式八格或宿主观测源已经完成。

**版本归属更新**：按[长期路线 §5](../design/operator-development-product-roadmap.md)，本计划纳入 v0.7；原阶段 A/B 只是历史工作分类。实验计划编号保持独立，预算与达标判据沿用本节约定。v0.7 还须通过支持清单、干净安装、宿主入口及源码/报告导出验收，见[一期清单 §5.2](operator-development-phase1-execution-20261031.md)。

**2026-11-04 实施更新**：[两阶段预登记](../testing/phase1-study.md)已完成，模型沿用用户阿里云 DeepSeek，A 已补为原生 DSH 基线，所有实际请求在预算预留前统一采样。本机预登记已留档；GPU/运行时/runner 实测绑定及正式八格尚未进行，本文旧轮事实仍只作探索记录。

### 0.1 新轮与旧轮的关系

§7.4 的四格插件臂与四格终评保留为探索记录：中途换过任务、修改过地板口径，且源码未提交。**新轮重新登记 4×2 格，两臂全部重新执行**；不把后来补跑的四格与旧轮拼成严格对照，也不改写旧契约、哈希或判定。旧轮“独立终评 1/6”已过时，实际四个选定插件格均已终评；旧人工表全 unknown，不能算有效人工投入读数。

### 0.2 对照与冻结内容

| 项 | 新轮约定 |
|---|---|
| 基线臂 A | Agent + 常规 shell/SSH、代码编辑与验证脚本，保留正常生成、调试和执行能力；由 Agent 自主完成，不要求用户手工开发。运行前固定实际 Agent 入口、工具清单、提示词与脚本哈希；不能笼统写成“当事人的工具” |
| 插件臂 B | 同一模型通过插件委托工作流完成开发；使用哪个宿主、CLI/MCP 入口和工具契约须入计划。直接用 CLI 的结果不得声称验证了 DSH 插件发现或宿主接入收益 |
| 四个任务 | matmul 4096³、relu 16384×6144、softmax 4096²、relu 1024×768，均为 fp16；参考实现、种子、完整 dtype/stride/case 集合与哈希在开跑前冻结。LayerNorm 契约问题另列缺陷，不在途中换题 |
| 共同条件 | 同一 provider/model/采样参数与 reasoning effort、同一 GPU UUID 与运行时、同一起始材料、同一终评器。两臂均有完整 L2/L3 验证能力；逐格独立上下文，不得读取对方候选、反馈或解题记录；可见性不能隔离时明确报告 |
| 开发预算 | 每格 ≤30 次模型调用、≤2400 s 墙钟、oracle 20 次；包含该格全部开发调用与工具等待，不按 Agent 角色重置。独立终评单列开销并计入验收总投入，终评次数与资源界限必须在派发前冻结 |
| 顺序与失败 | 单卡串行交错：任务 1、3 为 A→B，任务 2、4 为 B→A；保存每格 peers/污染观测。失败、超时、中断和未执行均保留；重试另登记，不挑最佳结果。代码/量具变化则停止本轮并另起身份 |
| 判定 | 沿用完整 L2 + 至少一次 valid 的 L3 + speedup≥1.0 + 终稿哈希绑定；另报噪声带、独立终评和是否带内打平，不能把带内涨跌解释为稳定性能收益。+1 仍仅为四任务的探索信号，不是统计显著性或可重复收益证明 |

### 0.3 自动记录什么

由执行侧从原始会话、模型账本、工具调用与执行回执汇总。**本节定义正式采集契约。离线演练已实现账本、验证与工具事件汇总及有来源事件的回放；宿主会话和外部编辑观测源尚未接入，仍记缺覆盖。现有 `ledger-*` 仅为可选自报工具。** 不要求用户补填任何字段。

| 数据 | 取证与缺失规则 |
|---|---|
| 达标、首次完整 L2、终评性能与有效性 | 关联源码/契约哈希和原始 verdict；未达到保留原因，不以成功样本代替全部格位 |
| 调用、token、开发与终评墙钟、验证次数、失败与重试 | 取模型账本、工具与执行时间戳；两臂同口径。未覆盖内部模型调用时记预算不可核，不将该格作为同预算收益证据；费用没有 usage/账单则 unknown |
| 人工介入事件 | 仅从有来源的用户消息、可归属的外部编辑或人工接管回执提取；记录 run/arm/stage、事件时间、actor、动作、原始证据位置，原因只取明确表述，无法判断则 unknown。Agent 写代码、重试或调用 shell 均归 Agent，不自动算人工 |
| 覆盖范围 | 保存会话/工具/外部编辑各来源是否完整与观测起止；只报“观测到 N 次”。完整观测中无事件可报 observed=0；缺来源则 unknown，不能推断没有人介入。初始任务与统一配置单列，不混入运行期接管 |
| 绕过与额外成本 | 由工具轨迹识别插件臂是否绕开委托，并记录执行者；基线臂正常用 shell 不是绕过。插件臂绕过的格不计插件工作流收益，格位保留；配置/纠错的机器耗时单列，不冒充人时 |

人工活跃分钟、阅读和思考时间继续 unknown；已有自报 CLI 保留为可选工具，不主动发问补表。事件次数只能描述可观测接管，**不能证明节省人时、总成本更低或整体经济收益**。自动采集缺人工观测源不阻止实验执行；缺核心预算或终评证据则限制比较结论。

### 0.4 执行顺序与当前待办

- [x] 2026-11-04 完成[两阶段预登记及实例绑定入口](../testing/phase1-study.md)：四任务完整契约、八格顺序、共同预算与采样策略、本地源码/提示资源/DSH/依赖身份；补齐原生 DSH A 臂，实际两臂本地 fixture 验证通过。模型沿用用户阿里云 DeepSeek 配置，GPU/远端运行时/部署回执待租卡后绑定；不能据此勾选正式开跑或结果。
- [ ] 固定本轮源码、量具与远端 runner 身份，核对当前树、工件、协议和 evaluator 摘要；登记 A/B 入口、任务与全部八格，冻结计划内容摘要，不沿用试跑时的摘要。
- [x] 实现离线八格编排、共同预算/验证预留核对、独立终评与自动事件汇总，提供零模型、零 GPU 执行和日志回放；覆盖 Agent 动作不算人工、缺源不报零、失败/重试及未执行不丢失。小 fixture 与正式任务分开编号，全部标记 test_only。
- [x] 完成[显式模型 / SSH / DSH 接入实现](../testing/phase1-connected.md)，复用共同模型预算、固定任务工具参数、独立终评及来源归档；提供 core 与 dsh 两个明确入口，缺宿主不降级。该勾选不计实际宿主/真机验收。
- [ ] 核对正式基线与 DSH 插件臂的实际采样参数、冻结两臂入口与来源覆盖；宿主会话/外部编辑观测缺失仍 unknown。当前 connected 计划与正式四任务八格分开，不以接入测试代替正式执行。
- [ ] 在目标卡核对 G1–G5、运行时和共享观测，串行交错运行新一轮八格；逐格保留终态、候选及证据。
- [ ] 完成独立终评并自动汇总两臂达标数、性能、资源和可观测接管事件；将未知与解释边界写入报告。用户无需逐格执行或填表。

正式新轮尚未执行，每格调用/墙钟预算未扩大；离线实现的来源范围与验收边界见演练说明。

## 1. T2.5 原轮预登记（历史冻结口径，新轮按 §0 另起身份）

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
| 工具面 | 基线臂由 Agent 使用固定的常用脚本自主执行，保留正常验证与执行能力；插件臂走委托工作流，入口与哈希按 §0 冻结 | 工作流入口与工具组织不同 |
| 文件系统可见性 | 隔离声明：基线臂若可读仓，必须在计划里写明「隔离未建立」，不得称其为裸环境（S5 已因此返工） | 记录差异，不隐瞒 |

## 3. T3.4 绕过与额外成本记录字段

路线 §6.2 要求计入「插件发现、工具调用成功与否、用户绕过工具、额外配置与纠错成本」。落成一张表，每格一行，**缺项写 unknown，不补 0**：

| 字段 | 取值 | 来源 |
|---|---|---|
| `discovery_ok` | yes/no/unknown | `operator-agent capabilities` 退出码 + 回执 |
| `tool_calls_attempted` / `succeeded` / `refused` | 整数 | `tool-calls/<id>/run-status.json`；refusal 有独立字段（发布形态 §1 第 5 项的可修正回显） |
| `bypassed_tool` | 工具名列表 + actor + 证据 | 自动关联工具轨迹；基线臂正常 shell 不算绕过；缺轨迹 unknown |
| 配置/纠错成本 | 机器耗时、失败/重试事件 | 自动关联执行回执；旧 `config_fix_minutes` 无自报即 unknown，不要求填报 |
| 人工介入观测 | 事件 + 来源覆盖 + 观测范围 | 按 §0.3 自动采集；旧 `human_involved_stages` 无证据保留 unknown，非开跑/出口门槛 |
| `machine_wall_s` | 秒 | `execution` 记录；注明含启动与清理，不当性能指标 |
| `outcome` | pass/fail/unknown + 原因 | 终态；基础设施中断记 infra/hold，不进搜索迭代 |

一条硬规矩：**`bypassed_tool` 非空的格，不得算作「插件工作流的收益」**——L6 那次会话里 `operator_run`/`repair` 用了 0 次、45 次 `bash` 手搓赢了插件，正是这个字段要抓的情形。

## 4. T3.5 命令参考（新轮需先完成 §0.4 的入口与采集实施）

前置未满足就**停在第 0 步并报告**，不许静默降级继续跑（[可迁移性 G1–G5](../design/dsh-operator-portability-4090.md#L85-L126)）：

```sh
# 0. 迁移门（五道门任一不过即停）
python3 -B -m benchmarks.probes.unknown_card_gates --selftest
#    ↑ 零机时自检：证明「无读数 ⇒ 五门全 unknown」「继承值 ⇒ 不得 pass」两条反例是活的

#    门的输入由采集器在目标机组装（读 torch 设备属性 + nvidia-smi UUID）。
#    五个脚本都是**单文件 scp** 过去跑（它们没法 import 门里的 schema，所以键名靠
#    tests/unit/test_unknown_card_probe.py 与 test_dispersion_thresholds.py 的静态守卫钉住）：
for f in benchmarks/measurement/measure_launch_floor.py \
         benchmarks/calibration/calibrate_achievable_peak.py \
         benchmarks/calibration/calibrate_dispersion_thresholds.py \
         benchmarks/probes/collect_shape_inventory.py \
         benchmarks/probes/collect_unknown_card_probe.py; do
  scp "$f" <ssh别名>:~/
done
ssh <ssh别名> "python3 ~/measure_launch_floor.py        > ~/floor.txt"   # G1：本卡 launch 地板
ssh <ssh别名> "python3 ~/calibrate_achievable_peak.py   > ~/peak.txt"    # G3：本卡峰值与 D2D 带宽
ssh <ssh别名> "python3 ~/calibrate_dispersion_thresholds.py --runs 3 \
      > ~/noise.txt"                                                     # G4：散布带与污染阈值
ssh <ssh别名> "python3 ~/collect_shape_inventory.py --dtype fp16 \
      --shape <声明形状> --shape ... --out ~/shapes.txt"                  # G5：形状在本卡跑一遍
ssh <ssh别名> "python3 ~/collect_unknown_card_probe.py --sharing exclusive \
      --from-probe ~/floor.txt --from-probe ~/peak.txt \
      --from-probe ~/noise.txt --from-probe ~/shapes.txt --out ~/card-probe.json"
scp <ssh别名>:~/card-probe.json ./
#      —— `--sharing` 必须是**标定当时**的共享形态；接显示器改 desktop，不许留空。
#         四份实测尾巴各打一行 `PROBE_JSON {...}`，里面带**本卡 uuid**；uuid 与本卡不符
#         或缺 uuid ⇒ 采集器直接拒（别卡的地板/峰值/阈值/形状不许混用）。
#         `--runs 3` 不是保险丝而是必需品：本卡实测单轮阈值轮间差 2–6 倍（runbook §7 第 5 条）。
#         `--shape` 是**声明**（与 `operator-agent prepare --shapes` 同一套写法），
#         采集器不替任务猜形状；空清单会被拒，而不是让 G5 落进 unknown 假装"还没核对"。
python3 -B -m benchmarks.probes.unknown_card_gates --from-json ./card-probe.json
#    退出码非 0（有 fail 或 unknown）⇒ 停在这里，不开跑；不静默降级
#    —— 控制面还有一道硬门：`operator-agent run` 探测到 cc<7.0 直接抛 `TritonUnsupported`，
#       一次 LLM 调用都不发。`verify`/`repair` 不经这道门（§6），所以别拿它们试不认识的卡。
operator-agent capabilities --json-out ./cap.json

# 1. 冻契约（每格一个新 run-dir，调用方拥有布局）
operator-agent prepare --declaration ./task-01.json --run-dir ./b/01 \
  --shapes "<shape>" --hardware "<探测值>" --dsl triton --precision fp16 --min-speedup 1.0

# 2. 基线臂：Agent 使用固定常规脚本自主开发（入口/预算控制待按 §0.4 落地）

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

# 6. 执行侧自动归档并汇总 §0.3 指标与来源覆盖（待实施，无现成 CLI 命令）
#    ledger-new/fill/report 仅保留为可选自报工具，不要求用户逐格填写。
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
- **CC 下限预检只覆盖 `run` 这一条路**：门在 `Conductor._execute_inner` 的硬件探测之后
  （`core/hardware_caps.py` 的 `require_triton_support()`），低于 7.0 就抛 `TritonUnsupported`
  且一次 LLM 调用都不发。而 `verify` / `repair` **不经 Conductor**（见 `hitl_tools.py` 的头注），
  那条路上「这台编不出码」仍然会被失败路由记成**候选失败**。没有顺手往 `_INFRA_MARKERS`
  加 marker 是有意的：那会让它变成**无限重试**（这类卡永远不会成功），正确的形状是一个
  「环境不可用」终态，需要单独设计——本文把它记成缺口，不记成已修。
- **7.x 的 `sunset` 分支只是一行 print**，改坏它今天没有任何用例会红。按「报缺陷前先证明有活入口」
  的纪律，这条不假装覆盖（变异表 `triton_cc_floor.py` 头部同步记了），它只是提示，判据不依赖它。
- 没有替 T3.1 选卡。**T3.2 只落了探针形状**（`benchmarks/probes/unknown_card_gates.py`，21 行变异表 21/21 拦住、门与接面共 59 条用例全绿）。~~**五道门一条读数都没有**~~ —— 2026-11-01 起 G4/G5 也有了产出方（§7 第 6/7 条），第一份真机读数在 §7 第 6 条，本文不猜数。
- 预算与门槛的数值取自当事人 2026-11-01 的指示；改动它们需要新计划与新身份，不在本文授权内。
- **不替表内阈值背书**：G4 那三个数只保证「来源是本卡 uuid + 三格都有值」，不保证数值稳定。本卡实测轮间差 2–6 倍（§7 第 6 条），门对此一无所知 —— 那是标定质量，属标定脚本与 `--runs` 的事，不是门的事。

## 7. 目标卡：RTX 4070（当事人 2026-11-01 确认：独占、ssh 可达）

> **卡源同日有变**：改成公开算力市场的价目表，逐档判定与排序见
> [`docs/research/80-算力市场机型可用性与连接方式核查.md`](../research/80-算力市场机型可用性与连接方式核查.md)。
> 本节留着当**样板**：判据（G1–G5、分母实测、共享形态显式申报）对 3070 / 3090 / A100 同型适用，
> 差异只在那几张的具体代价（8 GB 要裁标定脚本、3090 行是 ⚠️ 未核口径、A100-40GB 命中 80GB 分母）。

CC 8.9（Ada，AD104）落在仓内硬件表的 **8.9 行**，Triton 可用、有 Tensor Core 与 bf16、12 GB 显存。
四条已知后果都不需要卡就能定下来（第 2 条已在 2026-11-01 闭合，其余三条要卡上的读数）：

| # | 事实（仓内实读） | 后果与处置 |
|---|---|---|
| 1 | [verifier.py 的 8.9 行](../../operator_agent/decide/verifier.py#L206) 按 **L40S** 填（362 TFLOPS fp16 / 864 GB/s），而 `_PEAK_BY_NAME` 没有 4070 条目 ⇒ 走 CC 兜底 | 分母偏大是仓里**有意的不对称策略**（偏大只是不出口，偏小会提前停）。**归因先前写错了**：cc 行的出处串不带 ⚠️ ⇒ `peak_trusted` 其实是 **True**（[verifier.py:2524](../../operator_agent/decide/verifier.py#L2524) 只问口径核没核）；真正关着判停出口的是**「分母没实测过」**（[verifier.py:1875](../../operator_agent/decide/verifier.py#L1875) `未实测⇒天花板出口关闭`，§13.5 把这两问刻意分开）。后果不变：不在本卡跑一次 `calibrate_achievable_peak.py` 就没有出口，与机型无关（G3 验收） |
| 2 | ~~`generate/agents.py` 的 smem 分支只有 `8.0`/`9.0`，8.9 落 else~~ **本次已修**：runner 的 `gpu_info` 自描述 `hardware_report`（每 SM 共享内存 / 块 opt-in 字节 / L2 / 每 SM 线程数 / 寄存器 / warp，**读不到就省键**），Designer 侧改成 [`smem_budget()`](../../operator_agent/generate/agents.py#L51) 按 **实测 > cc 表 > legacy 申报 > 抛** 取值，`128` 兜底删掉；两个上限**有一个就够**（`shared_memory_per_multiprocessor` 不是每个 torch build 都暴露，只有 opt-in 时按它取 ⇒ Ada 上 99 ≤ 100，保守方向） | 后果变了，而且是两条：**老 runner（没有 `hardware_report`）在 cc 8.9 上现在直接抛 `HardwareProbeFailed`**，所以 4070 上必须部署本轮的 `remote_scripts/run_kernel.py`；CPU 回退（`memory.local.size = "N/A"`）不再是可生成目标——`128 KB` 借给它的这条路断了。`⚠️ cc 8.9 表内常量`不存在，实测值才是唯一来源 |
| 3 | 独占机器 | G4 的 `--sharing exclusive` 如实可填；这条消掉了消费卡最常见的杀手（桌面占用使 util 长期 >5%、每轮 degraded）。**但阈值本身仍是 H20 八卡共享机那组**，必须重测，否则是反向错（太松） |
| 4 | 预检阈值有两档，前一版本文只说了一个数（8192）——那是**共享模式**那档 | 独占默认档是 `min_free_mib=4096`（[s5_workflow_preflight.py:22](../../benchmarks/experiments/s5/s5_workflow_preflight.py#L22)），12 GB 卡上可过；`shared-known-processes` 那档的 8192 被 [s5_shared_gpu.py:31](../../benchmarks/experiments/s5/s5_shared_gpu.py#L31) 的 `validate_policy` **钉死为常数**（不是可配项），12 GB 卡上等于要求整卡 2/3 全空——按独占口径就不必碰它。通用默认 `min_free_mb=1024`（[_preparation.py:21](../../benchmarks/_preparation.py#L21)）无问题。三条都只在 **S5 驱动层**生效，阶段 B 走 `operator-agent run` 不吃这套策略 |

再加两条口径约束：4070 的带宽约为 H20 的 1/7，**别期待复现 1.41× 那类读数**；且这张卡上的全部结论都是**新身份新基线**，不得与 H20 的冻结记录同表（§5 规则）。形状上限仍由传输预算 256/512 MiB 决定，与显存无关。

## 7.1 第一张真机读数：RTX 3080 20 GB（2026-11-01，gpushare）

来源：公开算力市场一台 `i-2.gpushare.com:38210` 的容器，Ubuntu 22.04 / torch 2.9.1+cu128 /
Triton 3.5.1，**cc 8.6（68 SM、L2 5 MB、smem 100 KB/SM、1536 线程）**。本节是 §4 第 0 步
第一次跑到底的产物，**五门全绿**（`all_green=True`，退出码 0）。

| 门 | 本卡读数 | 与仓内继承值的差 |
|---|---|---|
| G1 地板 | **38.912 µs**（min，200 样本；median **76.8**） | H20 的 `launch_floor_us` 是 **20.24**（median，同口径）⇒ 本卡 **3.8×**。⚠️ 两者都是**不冲刷 median**；同一空核在 L2 冲刷口径下本卡只有 **3.07 µs**（H20 是 4.64 µs） |
| G2 共享内存 | 100 KB/SM，opt-in 101376 B | 走「实测」分支，不是 128 兜底 |
| G3 带宽 | **706.4 GB/s**（4096 MB D2D；8 MB 只有 520.8） | cc 8.6 行写 1555 ⇒ 偏大 **2.20×** |
| G3 峰值 | fp16 **65.44** / bf16 **65.90** / tf32 **33.36** / fp32 **22.14** TFLOPS | cc 8.6 行 156/312/312/19.5 ⇒ 偏大 **2.38× / 4.73× / 9.35×**，但 fp32 **偏小 0.88×** |
| G4 噪声带 | band = {valid **0.0372**, degraded **0.2225**, uninformative **0.0943**}；dispersion_degraded **0.077**；污染阈值 **1.0** | 继承 {0.001, 0.004, 0.02} / 0.10 / 5.0，逐格都不同 |
| G5 形状 | 1024×768 / 4096×4096 / 16384×6144 / 16384×16384（fp16，实占 3 MiB → 1 GiB） | 无跨卡样本，全在本卡池内 |

**六条要带走的结论**（都是这次跑出来的，不是推的）：

1. **地板读数有两个口径，别混用**。`measure_launch_floor.py` 的 `bench()` 不冲刷 L2、
   每轮把流跑干 ⇒ 量到 38.9 µs，主要是 **CPU 派发延迟**；`calibrate_dispersion_thresholds.py`
   （与 runner `_timed_runs` 同口径）在 `start.record()` 之前冲 256 MB ⇒ 空核只有 **3.07 µs**。
   混用会算出「地板支配 292%」这种物理上不可能的读数。标定脚本现在**直接拒**一个比最短核还长的
   `--floor-us`。G1 那个 38.9 该不该换口径，是一笔要单独定的账（改它等于改 H20 的历史读数）。
2. **cc 8.6 的兜底行对这张卡偏大 2–9 倍**，其中 tf32 偏大 9.35× 意味着 tf32 核的达成率上限
   只有 ~11%，天花板出口实质不存在；**fp32 偏小 0.88× 是唯一危险的一格**（偏小会提前停）。
   实测 `calibrate_achievable_peak.py` 之后 G3 才转绿。
3. **`nvidia-smi` 的 util 在「刚跑完」时读的是自己**。第一次标定的静止窗读到 `[70, 70, 0, 0, 0]`，
   据此算出的污染阈值是 **71%**（等于看不见任何对手）。加 3 s 静置窗后读 `[0,0,0,0,0]` ⇒ 阈值 1.0。
   **runner 的 `gpu_util_pct` 是测量前单次快照**，同一族自污染在那边也可能发生 —— 记成缺口，本轮没改。
4. **`torch.cuda.Event` 的量化步长会把跨轮散布打成 0**。本卡 5 个核里 4 个的 30 轮 median
   逐位相同 ⇒ 跨轮 iqr 恰好 0 ⇒ 若照搬「3 × 跨轮 iqr」的配方，会得到**带宽 0**
   （任何一点差别都算显著，SLOW 还会据此永久禁用配方）。所以跨轮散布取
   `max(跨轮 median 的相对 iqr, 跨轮均值的相对 iqr)`，并且**零带宽直接拒**。
5. **单轮阈值不能直接用**：同一张卡连跑 3 轮，`band.degraded` 是 0.2225 / 0.0351 / 0.2299（差 **6.5 倍**），
   `band.uninformative` 是 0.1664 / 0.0421 / 0.0943（4 倍）。取最大曾被单轮离群绑架
   （`uninformative` 一轮 0.3341 ⇒「差 33% 都算没变」），所以合成用**跨轮中位数**、
   污染阈值取**最小**，三轮原值全部写进产物 `runs`。冷跑用 `--runs 3`。
6. **degraded 档在这张卡上没有稳定的可测散布**。落这一档的形状中位数在 10–30 µs 之间，
   跨轮抖动被 launch 噪声支配 ⇒ 带宽只能是 20% 量级。含义要说清楚：**degraded 档的读数
   在本卡支撑不了小于 ~22% 的判决**。这与「degraded 不得驱动配方禁用」同向（保守），
   但不是「已经标定好了」。

另记两条**待办缺口**（本轮没做，别当成已完成）：

- **G5 的形状清单是声明 + 本卡验跑，不是发现的"可验收形状集"**。哪些形状该进来由任务决定
  （`operator-agent prepare --shapes`）；本卡只是证明它们跑得通、字节数够。16384×16384 fp16
  实占 1 GiB，是 512 MiB 传输预算下的上限形状。
- **20 GB 的 3080 不是标准 SKU**（标准是 10/12 GB）。容器内能观测到的一切支持独占
  （整卡两功能 `0000:06:00.0/.1`、device id `0x2206`、无 vGPU、无 MIG、静止窗全 0），
  但容器 PID 命名空间看不到宿主机上的其他容器 ⇒ 这仍**不是租据**，与 research/80 判
  「3090 Ti 8 GB 不可选」是同一族风险。

## 7.2 第一格真跑：链路通了，卡在端点吞吐（2026-11-01）

在 §7.1 那张卡上按 §4 走到底，**没有跑出任何性能读数**。链路逐段通了、最后一段没通，
如实记下来：

| 段 | 结果 |
|---|---|
| `capabilities` | `0.6.7` / 契约 9，三个 backend 全 `wired` |
| `prepare` | 契约冻结，`contract_hash = 7770b8be…`（matmul 4096³ fp16） |
| `verify --stop-after L2`（零模型调用） | **L1 + L2 全绿**：15 例正确性套件 passed，最大误差 0 |
| `run` 的环境段 | ssh / uv / python 3.11.12 / torch 2.9.1+cu128 / triton 3.5.1 / cuda 可用 / GPU 0 固定 / 环境指纹入档 |
| Designer | Sketch 生成成功（`tiled_matmul_fp16`, tiling 128x128x32），RAG 命中 1 例 |
| Planner | 14 个配方入 Plan（规则分支，零模型调用） |
| **Coder** | **`LLMCallRefused`**：`call_seconds=120` 用尽，`wall_seconds=120.007`，`status=failed` |

### 真因：端点只有 4.4–7.7 tok/s

`Qwen3.8-27B`（仓外 vLLM 端点，地址按仓库脱敏约定不落盘：`LLM_BASE_URL` 由环境变量给，形如 `http://<host>:<port>/v1`，免 key）实测三次：

| max_tokens | completion_tokens | 墙钟 | 吞吐 |
|---:|---:|---:|---:|
| 40960（Coder 实发） | — | 120.0 s（超时截断） | — |
| 1168 | 1168 | 152.4 s | 7.7 tok/s |
| 400 | 400 | 65.0 s | 6.1 tok/s |
| 100 | 100 | 22.9 s | 4.4 tok/s |

三个尺寸的 tok/s 同一量级 ⇒ **是逐 token 慢，不是排队慢**。

### 这笔账与冻结预算冲突

§1.2 冻结的是「每任务 ≤30 次模型调用 / ≤2400 s 墙钟」。2400 s × 7 tok/s ≈ **1.7 万输出
token**，而一个 Coder 调用自己就要 2–4 千 token（120 s 只够 ~800 token，实测连它的一半
都不到）。⇒ **这份冻结预算在这条端点上不可执行**，与卡、与代码都无关。

按 §1.5「不得为过门槛临时追加预算」，**不擅自把 `call_seconds` 或墙钟调大再跑一遍**。
出路只有两条，都要人来定：

1. 换一个够快的端点（同一份预算即可原样执行；预登记不用改）；
2. 写**新计划 + 新身份**，按本端点重述预算（runbook §1.2 的数值来自当事人 2026-11-01
   的指示，改它不在执行侧授权内）。

### 这一段路上修掉的三处「设施问题伪装成候选失败」

同一个形状出现了三次，值得单独记：三段都是**用例两侧各自绿、中间那一跳没人看**。

| # | 位置 | 症状 | 处置 |
|---|---|---|---|
| 1 | `deploy()` → `_setup_uv_env()` | `uv add numpy torch triton` 装了 **torch 2.15.0+cu132**，而驱动是 CUDA 12.8 ⇒ 远端 `torch.cuda.is_available()` 为 **False** ⇒ 候选被判 `unknown`（"没有 CUDA 执行设备，不模拟为通过"） | 用文档里的冻结准入路径 `OPERATOR_AGENT_EXISTING_REMOTE_ONLY=1`：远端 `.venv` 自己建（`--system-site-packages` 指向系统 torch 2.9.1+cu128），`deploy()` 只校验哈希、不 provision |
| 2 | `SSHExecutor.get_gpu_info()` | 按字面量重建字典，**丢掉 runner 自描述的 `hardware_report`** ⇒ `smem_budget()` 落到 cc 常量表（无 8.6 行）⇒ `HardwareProbeFailed` | 透传该键（3 行）+ 3 条「缝合处」用例；变异「去掉透传」⇒ 2 failed |
| 3 | `autonomous.py` 的 `call_seconds` | HTTP 超时被报成 `LLMCallRefused`，读起来像模型拒绝 | **未修**，记成缺口：这个终态应区分「端点慢/超时」与「模型拒绝」 |

第 2 条最值得记：它两侧都有守卫（`TestRunnerReportsTheSameKey` 守写者、
`TestSmemBudget` 守读者），唯独中间那一跳没人看，于是**两个组件各自的用例全绿而集成是坏的**。

## 7.3 第一格插件臂跑完（2026-11-01）：正确、但没达标

§7.2 卡在端点吞吐之后换了端点（百炼 `deepseek-v4.1-flash-1017`，实测 **131–187 tok/s**，
比原来那条 7.7 tok/s 快 20 倍）⇒ **原冻结预算原样可执行**，预登记一字未改。

| 项 | 读数 |
|---|---|
| 预算用量 | **2 次模型调用 / 1902 output token / 10.6 s 模型墙钟**（上限 30 次 / 2400 s） |
| L1 | passed（编译） |
| L2 | **passed**：15/15 case、最大误差 **0.00e+00**、含 strided 与非连续输入 |
| L3 | **failed**：`speedup = 0.9972 < min_speedup = 1.0` |
| `measurement_validity` | **valid**，资格位五项全 True（`can_verdict/tune/persist/report/count_progress`） |
| 吞吐读数 | `gflops = 54626.7`（54.6 TFLOPS），`bound_type = compute`，验证墙钟 159.8 s |
| 相对**实测**可达上界 | 54626.7 / 65440 = **83.5%** |
| 候选 | 97 行 Triton，1D 网格 + GROUP_SIZE_M 分组、K 循环指针推进、3 参数 `tl.dot`，`sha256[:12]=af80228792ca` |

### 三条结论

1. **这一格「不达标」是判据口径造成的，不是核差**。−0.28% 的差距远在本会话自己标定出来的
   **valid 档判决带宽 3.72%** 之内，按 `verdict()` 的带内比较是 `SAME`；但任务级达标用的是
   `speedup_vs_baseline >= min_speedup` 的**硬标量比较**，不查那张带宽表。⇒ 在一个 3.7% 噪声底
   的卡上要求 `speedup ≥ 1.0`，对一个已经跑到 83.5% 可达峰值的核来说基本是掷硬币。**这一条要进
   阶段 B 的判读，但本轮不改判据**（改它等于改 §1.3 的达标定义）。
2. **roofline 分母的偏大在本轮是活的、不是理论的**。`iteration_history.roofline` 里
   `hw_peak_gflops = 312000`、`hw_peak_bandwidth_gbs = 1555` —— 正是 cc 8.6 那行 A100-40GB 的数，
   而本卡实测是 65440 / 706.4 ⇒ 对这两条分别偏大 **4.77× / 2.20×**。G3 测出来的账，在产物里
   原样出现。
3. **基线臂没跑**。§1.2 的 4×2 是「插件臂 4 格 + 基线臂 4 格」，基线臂是当事人自己在用的
   Agent + 常用脚本，本轮只跑了插件臂一格 ⇒ **没有任何跨臂比较**，`+1 最小可检出单位` 这条路还差一半。

### 归档

证据树 `output/plugin_completion/20261101-phase1-card0-and-cell01-v1/`（`manifest.json` +
`artifacts-SHA256.txt`，16 件全部 `shasum -c` 复算通过）：`probe/` 五件（G1–G5 原始读数 +
门的输出）、`cell-01/` 十件（候选源码、报告、`result.json`、`benchmark.json`、`sketch.json`、
契约、声明、模型调用账本）。

⚠️ **manifest 里记着 `dirty: true`**：这一格跑的是**未提交**的工作区（本轮四处修复都还没进任何
提交），所以上面那些哈希是「跑的是什么」的唯一身份。要复现必须先提交。

## 7.4 插件臂四格跑齐（2026-11-01）：一格可用、一格打平、两格真失败

§7.3 之后继续跑，中间穿过三处阻塞（下面逐条记）。**八次运行、六个数据点**：

| 格位 | 任务 | L1 | L2 | L3 | speedup | validity | 资格位 | 调用/输出 token | 墙钟 |
|---|---|---|---|---|---|---|---|---|---|
| 01 | matmul 4096³ fp16 | ✓ | ✓ 15/15 | ✗ | 0.9972（终评 **0.97x vs cuBLAS**） | valid → 终评 **degraded** | — | 2 / 1902 | 198 s |
| 02 | relu 16384×6144（旧地板） | ✓ | ✓ | ✓ | 1.0018 | **degraded** | 4/5（不可 persist） | 4 / 1712 | 183 s |
| **02c** | relu（**修好地板后重跑**） | ✓ | ✓ | **✓** | **1.0072** | **valid** | **5/5 全 True** | 3 / 1554 | 209 s |
| 03 | softmax 4096² | ✓ | ✓ | ✗ | 0.8261 | valid | — | 4 / 2401 | 171 s |
| 03b | softmax 重跑 | ✓ | ✓ | ✗ | 0.4718 | degraded | — | 3 / 2185 | 203 s |
| 04 | relu 1024×768（负对照） | ✓ | ✗ | — | — | — | — | 4 / 1639 | 154 s |
| — | layernorm 16384×6144 | — | ✗ | — | — | — | — | **0 / 0** | **2 s** |
| — | layernorm 1024×768 | — | ✗ | — | — | — | — | **0 / 0** | **2 s** |

预算侧全部远低于冻结上限（≤30 次调用 / ≤2400 s）：最多一格用了 4 次调用、2401 token、209 s。

### 三处阻塞（都记成缺陷，不是调参）

1. **layernorm 结构性不可跑**。`build_task` 给 layernorm 生成 `X/weight/bias`（affine），而
   内建 L2 profile `correctness_profiles.py:126` 只物化 `X`，集合不等即
   `BenchCaseError: 不支持 affine/自定义 eps 等变体` ⇒ **一次 LLM 调用都不发、0 机时**。
   `verifier.py:1011` 的注释早就写着这个不对称（只在"重传输入"那一处做了绕行），
   但 profile 那道门没有。两个 layernorm 格位因此换成 `relu`（内建 profile 支持 `X` 单输入）。
2. **地板口径（本节最重要的一条）**。先按 review 的建议把 **76.8 µs**（不冲刷 median）记进
   `_KNOWN_HARDWARE["3080"]`，紧接着 **relu 那格自己把这条路堵死了**：它跑出 **700.9 GB/s**，
   而本卡实测带宽上界是 **706.4 GB/s**（99.2%）。若 574.4 µs 里真有 76.8 µs 是 launch 开销，
   搬运期就得跑 **809 GB/s > 上界** —— 物理上不可能。**开销的上界只有 4.5 µs**，与冲刷口径的
   **3.07 µs** 吻合、与 76.8 µs 差 17×。字段因此记 3.07，76.8 留在条目注释里。
   口径的直接后果：用 76.8 会把 **768 µs 以下的核全部判 degraded**（3/4 的格带一个不存在的
   测量缺陷进表）；同任务重跑的 02c 验证了这点 —— validity 从 degraded 变 **valid**、资格位 5/5。
3. **`_KNOWN_HARDWARE` 缺 3080 时 `launch_floor_us=None`** ⇒ `launch_dominance()` 返回 None ⇒
   launch 支配这条轴**没有读数**，4 µs 与 2.5 ms 的核在这一轴上一样。补上条目才修好。

### 三条结论

1. **工作流能产出贴 roofline 的核**：relu 那格 **704.7 GB/s = 实测上界的 99.8%**，
   `speedup=1.0072`、资格位全 True —— 这是本轮唯一一格既达标又可用于裁决的。
2. **`min_speedup=1.0` 在量噪声方向**：matmul `0.9972`（83.5% 算力上界）判红、relu
   `1.0072`（99.8% 带宽上界）判绿，两格都贴着各自上界，差别只有 0.28%/0.72%，而本卡
   **valid 档噪声底是 3.72%**。判据未改（§1.3 的达标定义不在执行侧授权内），只记结论。
3. **候选方差压倒了臂间差异**：同一份 softmax 声明跑两次，基线稳（0.164→0.176 ms，7%），
   候选抖（0.199→0.374 ms，**88%**）。§1.4 的「+1 最小可检出单位（每臂 4 格）」以「格与格
   可比」为前提，这条现在有反例：达标数更像在测"模型这次手气"。

### 四格独立终评（runbook 步 5，零模型调用）

| 格 | 终评读数 | validity | 结论 |
|---|---|---|---|
| 01 matmul | `2.515/2.508 ms` vs **cuBLAS(FP16)**（1.00x，实为 0.9972） | **degraded**（离散 0.123） | 不达标，但差距在 3.72% 带内 |
| **02c relu** | `0.5714/0.5755 ms` vs **PyTorch Eager**（**1.01x**） | **valid** | **达标且可用** —— 唯一一格 |
| 03 softmax | `0.1935/0.1597 ms` vs PyTorch Eager（**0.83x**） | valid | 不达标，差距**远在带外** |
| 04 relu 小 | L2 再次失败（strided 输入算错） | — | 不可判 |

两条必须一起读的旁证：① **cuBLAS 基线自己从 2.187 ms 变到 2.508 ms（+15%）** —— 同一份源码两次终评，
"相对基线"这个量本身有基线抖动；② 两次终评都记着「卡与 1 个其他进程共享」（`gpu_claim` 的 peer 观测），
即这两轮的读数是在有 peer 的情况下取的。

### 还差的

- **旧轮基线臂 0/4**：无跨臂比较；按 §0 新轮由 Agent 执行两臂，不要求用户手工补跑。
- **四个选定插件格独立终评已完成**（见本节上表）；旧人工表仅一份且全 unknown，取消补表待办，不计作人时证据。
- **负对照没测到想测的东西**：relu 1024×768 那格在 **L2** 就红了 —— 生成的核在 **strided 输入**
  上算错（误差 2.90），没能走到「测不准时说测不准」那一步。L2 拦住了它是正确的，但负对照
  的意图（uninformative 档的自律）仍然没有读数。
- **协议身份已变成 `e5d88ef0…`**（改 `hardware_caps.py` 动了那 73 文件身份），`0.6.8` **未升**、
  包**未重建** ⇒ 这一批格跑的是未提交工作区，manifest 里照旧记 `dirty: true`。

## 修订记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-11-02 | 新增离线 A/B 编排、共同预算与验证预留核对、独立终评、事件汇总和只读回放；正式 Agent/宿主/真机入口仍待接，八格小 fixture 不记正式对照 | [演练实现与验证记录](../testing/phase1-rehearsal.md)；定向回归 277 passed，生命周期 4 passed |
| 2026-11-01（方案修订） | 新增 §0：两臂由 Agent 执行、自动归档机器指标与可观测介入事件；取消强制人工填表和人时出口门槛；旧轮保留探索记录，新轮八格重新冻结执行。明确采集与预算控制尚待实施，修正终评剩余项 | 用户本轮明确要求调整方案；§7.4 试跑的换题、地板修订、四格终评及 dirty 身份事实 |
| 2026-11-01 | 创建：T2.5 预登记（固定预算问题 + 每任务 ≤30 调用 / ≤2400 s 墙钟 + 4×2 设计 + +1 最小可检出单位）、T3.3 控制变量表、T3.4 绕过与额外成本字段（含「bypass 非空不算收益」硬规矩）、T3.5 七步 runbook（命令与 CLI 默认值实测核对）、T3.6 四代协议身份与四条新 evaluator 摘要 | `operator_agent/app/cli.py:39-145` 参数实读；`digest()` 实算四条 evaluator 身份与三代协议摘要；证据树 `grep` 全树查 `evaluator_sources`（0 命中）与 `measurement_protocol`（7 命中） |
| 2026-11-01 | 门的输入侧补齐：新增采集器 `collect_unknown_card_probe.py`（读 torch 设备属性与 nvidia-smi UUID，**拒绝给共享形态与 smem 兜底值**），runbook 第 0 步接上它；新增 §7 记录目标卡 RTX 4070 的四个已知后果（8.9 行按 L40S 偏大 ⇒ 判停不出口、`agents.py` 缺 8.9 分支、独占消掉污染杀手、8192 MiB 预检阈值要重设） | 采集器契约 8 条用例（与门的字段对齐、没测的段不许出现在产物里）；变异表扩到 9 行后重跑 **9/9 KILLED**，跑后 `git status` 只剩本次新增文件；`_PEAK_BY_CC` 与 `agents.py` 分支为本次实读 |
| 2026-11-01（同日第二次） | §7 第 2 条**闭合**：`gpu_info` 自描述 `hardware_report`，Designer 侧新增 `smem_budget()`（实测 > cc 表 > legacy 申报 > 抛），`128` 兜底与 legacy 分支的 `else 128` 一起删掉 ⇒ **老 runner 在 cc 8.9 上现在会抛**，4070 必须部署本轮 `remote_scripts/run_kernel.py`。§7 第 4 条**更正**：先前只写 8192 是半句话，实际独占默认 `min_free_mib=4096`、8192 属 `shared-known-processes` 且被 `validate_policy` 钉死为常数，三条都只在 S5 驱动层生效。第 0 步补上两份实测（地板 / 峰值带宽）与 `--from-probe` 的接线，并写明 G4/G5 仍无读数。过程中查出并修掉一处会让 4070 **开不了跑**的口径：`shared_memory_per_multiprocessor` 不是每个 torch build 都暴露，于是门与采集器改成**两个上限有一个就够**（全没有才拒），Designer 只有 opt-in 时按 opt-in 取（Ada 99 ≤ 100，保守方向） | `smem_budget()` 与 19 条新用例（含 `LocalExecutor`「N/A」申报被拒、MockGPU 申报可读的反向对照、opt-in 单独可读）；用例总数 34→59；变异表 21 行重跑 **21/21 KILLED**（行为 18 / 结构 3），跑后 `git status` 无残留、五处改动文件 grep 不到 `if False:`；`s5_workflow_preflight.py:22`、`s5_shared_gpu.py:31`、`_preparation.py:21` 三处阈值本次实读；H20 归档 `raw/hw/nvml_attrs.json` 用来核对 torch 属性名拼法 |
| 2026-11-01（同日第三次） | **§7 第 1 条的归因更正**：先前写「`peak_trusted` 不为 True ⇒ 判停出口不开」是错的——cc 行的出处串不带 ⚠️，`peak_trusted` 实际为 **True**；出口认的是**「分母实测过」没有**（`未实测⇒天花板出口关闭`），§13.5 刻意把「口径核过」与「探测过」分开。后果与处置不变（新卡必须实测）。另在 §7 顶部记下**卡源同日有变**并指到 [`docs/research/80`](../research/80-算力市场机型可用性与连接方式核查.md)：市场机型逐档判定 + 首选 3070 8 GB / 同代选 3090 24 GB | [verifier.py:2524](../../operator_agent/decide/verifier.py#L2524) 与 [verifier.py:1875](../../operator_agent/decide/verifier.py#L1875) 两行本次实读；`_PEAK_CC_SOURCE` 无 ⚠️ 前缀为实测所得；市场价目表与连接方式取自其公开文档三页 |
| 2026-11-01（同日第四次） | 第 0 步补一句**控制面硬门**（`operator-agent run` 探测到 cc<7.0 直接抛 `TritonUnsupported`，一次 LLM 都不发）；§6 加两条「本文没有做的事」：① 这道门只在 `run` 上——`verify`/`repair` 不经 Conductor，那条路上架构不支持仍被记成候选失败，而**不往 `_INFRA_MARKERS` 加 marker 是有意的**（会变无限重试：这类卡永远不会成功），需要的是「环境不可用」终态；② 7.x 的 `sunset` 分支只是 print、没有活入口，改坏它今天不会红 | `hitl_tools.py` 头注「不经过 Conductor」实读；`decide/failure_route.py` 的 `_INFRA_MARKERS` 实读（只含 oom 那一族）；变异表 `benchmarks/mut_specs/triton_cc_floor.py` 4/4 KILLED；`operator_agent: 6.1 抛 / 7.5 sunset / 8.6 ok / 无 cc unknown` 是在**解包后的 0.6.7 wheel 里**跑的 |
| 2026-11-01（同日第五次） | **G4/G5 的产出方补齐**，第 0 步第一次跑到底：新增 `collect_shape_inventory.py`（G5 的形状清单：声明 + 本卡分配跑通 + 实测字节数 + 盖本卡 uuid，空声明拒绝），`calibrate_dispersion_thresholds.py` 加 `PROBE_JSON` 尾巴与 `--runs`（G4 的三个阈值：band / dispersion_degraded / contamination_util_pct），采集器 `assemble()` 合并 `noise` 与 `shapes` 两段并**只认本卡 uuid**（别卡形状在入口就拒、band 键集拼错就拒、半截标定就拒）。新增 §7.1 记第一张真机的五门读数与六条结论 | 五门 **all_green=True / exit 0**（RTX 3080 20 GB，cc 8.6）；用例 59 → 81 条（采集器 19→27、新增静态守卫 14 条；门 21 与 smem 19 未动），变异 **9/9 KILLED**（含 M2b：把 payload 产出处的一个键改名 —— 用「全模块任一字典覆盖三键」的宽松写法时它曾逃掉）；三处真机踩到的口径坑各配一条守卫：`--floor-us` 比最短核还长 ⇒ 拒、band 为 0 ⇒ 拒、静止窗先静置 3 s |
| 2026-11-01（同日第六次） | **第一格真跑：链路通到底、卡在端点吞吐**。`prepare`/`verify L2`/`run` 的环境段与 Designer 全通（L2 15 例正确性套件全绿、最大误差 0），**Coder 在 `call_seconds=120` 上超时**。实测端点 `Qwen3.8-27B` 只有 **4.4–7.7 tok/s**（三个尺寸同量级 ⇒ 逐 token 慢）：2400 s 冻结墙钟 ≈ 1.7 万输出 token，而一个 Coder 调用就要 2–4 千 ⇒ **冻结预算在这条端点上不可执行**。按 §1.5 不擅自加预算，出路是换端点或写新计划+新身份。另记三处同族缺陷（设施问题伪装成候选失败）与 §7.2 的处置表 | `verify --stop-after L2` 全绿；`model-journal` 里 `call2 status=failed / wall_seconds=120.007`；端点吞吐三次实测 7.7 / 6.1 / 4.4 tok/s；`autonomous.log` 的环境自检七项全 OK；远端 `.venv` 重建后 `torch 2.9.1+cu128 cuda True`；新增缝合处用例 3 条（变异「去掉透传」2 failed） |
| 2026-11-01（同日第七次） | **第一格插件臂跑完**：换百炼 `deepseek-v4.1-flash-1017`（131–187 tok/s，比原端点快 20 倍）⇒ 冻结预算原样可执行，预登记未改。结果 **L1+L2 全绿（15/15 case，最大误差 0）、L3 不达标（speedup 0.9972 < 1.0）**，用量 2 次调用 / 1902 token / 10.6 s。三条结论：① −0.28% 远在本会话实测的 valid 档带宽 3.72% 之内，判「不达标」是硬标量比较不查带宽表所致，本轮不改判据；② `roofline.hw_peak_gflops=312000` / `bandwidth=1555` 证明 cc 8.6 兜底行的偏大（4.77×/2.20×）在产物里是活的；③ **基线臂没跑，没有任何跨臂比较**。产物归档进证据树并按 16 件 SHA256 复算通过 | `cell-01/result.json` 的 `iteration_history[0]`（`levels_passed` / `speedup` / `eligibility` / `roofline`）；`benchmark.json`（`gflops=54626.6699`、`bound_type=compute`、`timing_receipt` 六件源码哈希）；`model-journal/summary.json`（2 次调用、1902 token）；端点吞吐三档实测 131.4 / 186.9 tok/s；`shasum -a 256 -c` 16/16 OK |
| 2026-11-01（同日第八次） | **插件臂四格跑齐**（八次运行、六个数据点）：02c `1.0072 / valid / 资格位 5/5 / 704.7 GB/s = 实测带宽上界 99.8%`（本轮唯一既达标又可用）；01 matmul `0.9972` 打平（终评 0.97x vs cuBLAS、degraded）；03/03b softmax `0.8261`/`0.4718` 真失败且候选跨次抖 88%；04 relu 小形状在 L2 因 strided 输入算错（误差 2.90）。三处阻塞：① **layernorm 结构性不可跑**（`build_task` 给 affine，内建 profile 只物化 X ⇒ 0 调用 0 机时），两个格位换成 relu；② **地板口径由数据定死**——relu 跑到 700.9 GB/s（上界 706.4），若真有 76.8 µs 开销则搬运期需 809 GB/s，物理不可能 ⇒ 字段记冲刷口径 3.07 µs，76.8 留注释；用 76.8 会让 768 µs 以下全部假降级；③ `_KNOWN_HARDWARE` 缺 3080 时 launch 支配无读数 | 六格 `conductor.json` 与 `benchmark.json`；`model-journal/summary.json`（调用数/token/墙钟）；relu 那格的 `memory_bandwidth_gbs=700.9` 与 `calibrate_achievable_peak` 的 706.4 反推；`tests/unit/test_hardware_caps_3080.py` 7 条（变异「改回 76.8」⇒ 2 failed）；`correctness_profiles.py:126` 与 `verifier.py:1011` 实读 |
| 2026-11-01（同日第九次） | **§7.4 落盘 + 六格归档**：`output/plugin_completion/20261101-phase1-stage-b-cells-v1/`（6 个格位 + 2 个被 profile 挡住的格位骨架，76 件、`shasum -a 256 -c` 76/76 OK、manifest 记 `dirty:true` 与协议摘要 `e5d88ef0…`）。另**改了一条既有用例的期望**：`test_a_runner_that_does_not_report_still_refuses` → `..._falls_back_to_the_table...`，因为给 3080 补了 `_KNOWN_HARDWARE` 条目之后，runner 不自描述时 `memory.local.size` 不再是 "unknown" 而是 "100KB/SM"，落在「实测 > cc 表 > legacy 申报 > 抛」的第三级 —— 旧断言把「表里没有这张卡」当成了不变量 | 归档 76 件哈希复算；`tests/unit/test_smem_budget.py` 23 passed（新增「未知卡仍拒」反向对照）；`tests/hygiene` 59 passed；§7.4 的六格读数取自各自 `conductor.json` / `benchmark.json` / `result.json` |
| 2026-11-01（同日第十次） | **四格独立终评补齐**（步 5，零模型调用）：**02c relu 终评仍为 `L3 True / validity=valid`（1.01x vs PyTorch Eager）⇒ 唯一达标且可用的一格**；01 matmul 终评 `1.00x vs cuBLAS`（实为 0.9972、validity=degraded）；03 softmax `0.83x`、valid；04 relu 小 L2 再次失败。两处旁证：cuBLAS 基线两次终评差 **+15%**（2.187→2.508 ms），且两次都观测到「卡与 1 个其他进程共享」。产物落 `RESULTS.md` / `RESULTS.json` / `cell-*/final.json`（归档 82 件，`shasum -c` 82/82 OK） | 四份 `final.json` 的 `verdicts[L3].message` 原文；`RESULTS.md` 汇总表；`artifacts-SHA256.txt` 82 件复算；resolver 走 `OPERATOR_AGENT_EXISTING_REMOTE_ONLY=1`（零 provision） |
| 2026-11-01（版本规划） | 对照子计划归属 v0.7；保持独立实验计划编号与既定预算/判据，明确实验完成后仍须通过安装、宿主入口、结果导出和交付包验收 | 用户本轮版本迭代要求；长期版本路线与一期清单 §5.2 |
