# 共享生成上下文与候选记录

2026-10-26，落实开发评审 R4 中“各生产入口统一构建上下文、返回统一候选记录”。

## 入口与策略

`operator_agent/generation_context.py` 提供纯函数接口：

- `build_generation_context()`：构建任务、冻结契约、sketch、输入布局、调用约定、规约/技能身份与策略声明。
- `generation_request()`：绑定实际发送消息的哈希、配置模型名、调用标签、温度、输出上限和已有预算水位。
- `make_candidate_record()` / `validate_candidate_record()`：创建和检查统一候选记录。
- `candidate_summary()`：生成不含源码和完整上下文的工具回包摘要。

各生成器仍保留原来的返回类型，通过 `candidate_records` 和 `last_candidate_record` 暴露统一记录；Fixer 的 `FixOutcome.candidate_records` 同时提供本次调用的独立快照。这样已有字符串、二元组及动作信封调用者保持兼容。

| 生成器 | 策略标识 | 保留的行为 |
|---|---|---|
| `CoderAgent` | `single-with-precheck-retry` | 最多三次生成/预检，反馈重试 |
| `CoderPipeline` | `parallel-sample-normalize-gate-repair-template` | 规划、三温度采样、机械改写、编译/资源门、最多两轮修复、模板回退 |
| `FixerAgent` | `repair` | 原有失败反馈、诊断、最多三次预检尝试、动作信封 |
| `SingleCandidateCoder` | `bounded-single` / `bounded-repair` | 原有冻结上下文、一次调用、静态门、严格预算与独立 lineage 审计 |
| `Conductor` 模板参数空间 | `declared-template-space` | 现有实例化器直接生成；不调用模型 |

`autonomous.main()` 当前仍显式设置 `use_coder_pipeline=False`，所以 `operator_run` 的生成器是普通 `CoderAgent`。调用方显式启用流水线才有并行采样等能力。本次没有切换默认策略、增加模型请求或新增 MCP 工具。

上下文包含共享事实，不意味着每条路径发送同样的 prompt。各适配器从共享上下文取事实，按现有模板投影；实际消息由 `request.messages_hash` 绑定。技能身份记录实际选用的角色和内容哈希；有界路径仍不注入 coder 技能。

缺失或暂不支持的输入元数据，在普通路径中明确记为 `unknown`，不因此新增拒绝条件；有界路径继续严格要求明确接口。输入元数据只包含 shape、dtype、stride 和标量类型，不包含张量元素或标量输入值。参考源码只额外记录内容哈希；完整语义约束仍由已有冻结契约负责。

## 候选记录

记录 schema 为 `operator-candidate-record-v1`，包含：

- `producer`、`strategy`、完整 `context`、`context_hash`；
- 请求身份 `request`、尝试序号 `attempt`；
- 候选源码 `source`、`source_hash`、保留方式 `source_retention`、修复父源码 `parent_source_hash`；
- `source_author`：模型、机械改写或规则模板；
- `status`、`static_precheck`、固定原因码或异常类型，以及 `record_hash`。

预检失败、请求失败、无候选、只提问也保留记录。无候选时 `source_retention=absent`，源码及其哈希为 `null`；完整源码记录为 `full`；仅保留源码哈希的记录为 `hash_only`，源码为 `null` 但哈希保留，不能误读成没有生成候选。流水线按温度槽顺序合并并行记录；机械改写另建记录并绑定原源码，不覆盖原始模型候选；规则模板不会被归因给模型。

一份模型候选可能还有独立的机械改写或门检查记录，**记录条数不等于模型调用次数**。实际调用和 token 用量继续从已有请求账本/成本跟踪器读取。`configured_model` 只是配置名，不是服务端实际响应模型的证明。

`prechecked` 仅表示通过原有本地预检。所有记录固定保留 `candidate_execution=false`、`correctness_proven=false`、`gpu_measurements=0`、`restored_eligibility=false`。这些字段描述生成记录本身的证明范围；随后通过独立验证也不改写生成记录。哈希检查只检测内容漂移，不认证作者，不恢复验证资格。

## 记录的传递和保存

- `Conductor` 收集每次 Coder/Fixer 调用新增的记录，包括提问后的第二次修复和异常退出。跨任务重新清空，跨轮累积，不重收上一轮记录。控制器副本使用 `without_candidate_source()` 保留哈希和完整上下文，源码不进入审计副本；生成器原记录和返回源码不被修改。这保持了 §21/§22 中被动作门拒绝的源码不能留在运行目录的既有边界，审计文件不会成为第二条源码写入通道。
- 返回产物 `artifact.metadata.candidate_records` 包含上述记录，`selected_source_hash` 单独绑定最终保存源码。最后生成的候选不一定是保留的最佳候选。哈希保留模式只能检查记录结构和内容一致性，无法从记录本身重算已省略源码的哈希；已接受候选可结合正式源码文件核对。
- `ArtifactStore` 显式注册这两个键，将它们写入 `result.json` 顶层并在加载时恢复；其他 metadata 不随之整体导出。
- 生成过程中，在已有 `run_root` 或运行绑定目录中更新 `generation-records.json`。因此模型调用失败、还没进入最终保存阶段时也有记录。证据写入失败会打印错误类型，并保留原来的模型异常。
- 有界生成另写只读 `candidate-record.json`，原有 `generation-context.json`、`candidate.json` 和 lineage 审计保持原协议。
- `repair_loop.repair()` 返回完整记录与 `selected_source_hash`；MCP `operator_repair` 将完整记录存入 `repair-N.json`，回包只含统一摘要，轮次通过 `candidate_record_hashes` 引用，避免重复大段源码和上下文。工具契约版本升级为 4。
- 无人值守摘要记录候选记录数量、最终源码哈希与证据路径；失败状态也提供已存在的生成证据路径。

外层/人工传入 `operator_verify` 的源码没有经过这些生成器，仍只保留原有来源声明，不凭空补造生成记录。旧产物缺少记录时也不会被追认为经过某个生成器。恢复运行的记录仅描述本次新调用，不回填历史候选。

## 范围与验证

本次收敛生产生成/修复适配器，未改动 `quality_evaluation` / `quality_portfolio` 的历史实验生成臂，也未把其旧证据重标为生产流水线结果；这些实验仍有各自冻结计划与请求账本。Q1 外层/内层对照继续暂停，`operator_generate_candidate` 仍未开放。

验证使用本地模型替身、模拟后端和现有回归套件，覆盖实际发送消息、原有调用次数、失败记录、并行顺序、改写血缘、跨任务清空、提问后二次修复、落盘/加载和 MCP 精简回包。这次重构建立复用和比较的基础，不据此声称真实算子正确率或性能提高。

本次验证结果：Python 全量运行 5276 通过、1 跳过；包含新增成功/失败状态回包的入口专项 88 通过；DSH 插件 50 通过；`git diff --check` 通过。Golden 首轮 3/4 通过，LayerNorm 的功能验证通过但 CPU 延迟超出历史基线；三次独立复核为 24.3、32.7、16.6 微秒，历史基线为 14 微秒，不能视为稳定的性能对照通过。三次候选与保留的历史基线源码逐字相同（SHA-256 `6504ad444caeb099b6162bbd924ef4c0a6123b529ecae7ce11ea3676dbaa2547`）；尚未确立计时差异与本次改动的因果关系，未修改 Golden 基线或容忍范围。
