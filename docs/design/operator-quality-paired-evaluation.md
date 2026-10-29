# 真机预检与同候选配对评测

本轮在[质量闭环](operator-quality-improvement.md)上增加两个能力：付费调用前确认现有执行环境可用；生成一次初始源码，让两个修复策略从同一候选开始。默认仍只生成计划。开发验证使用离线假服务与 CPU 回放，没有调用真实模型或 GPU。

## 预检发生在模型之前

普通质量运行和配对运行均按「创建后端 → 预检 → 创建模型客户端 → 生成」执行。预检失败或证据未知时保留任务及分母，模型调用数为零。

当前支持 NVIDIA 的一次性 SSH 后端。检查包括真实设备能力、已固定的 GPU 编号及 UUID、现有 `.venv/bin/python` 中的 PyTorch/Triton/NumPy/CUDA 身份、已部署 runner 及同级依赖的完整 SHA256，以及一个 257 元素的 Triton 向量加控制核。控制核验证真实 JIT、启动、同步和尾部处理；它不替代候选的完整 L2，也不提供性能资格。

后端通过 `READONLY_QUALITY_PREFLIGHT` 声明能力，并实现 `quality_preflight()`；门面按同一接口转发。缺失能力或证据未知时停止，调用方不按后端类型猜测是否支持预检。

预检不安装依赖、不部署或更新 runner。控制核使用临时目录与临时编译缓存。只有检查全部通过，才把同一后端标记为已核实现存 runner，后续用已验证的 `.venv/bin/python` 运行，避免 `uv run` 自动同步依赖。常驻后端暂时拒绝放行。环境或 runner 未准备好时应先单独完成部署，再重新预检；预检不会用自动安装把问题隐藏起来。

独立预检不需要模型凭据，也不物化任务输入：

```bash
python3 benchmarks/operator_quality.py --paired --backend ssh --host <ssh-alias> \
  --model "$LLM_MODEL" --case layernorm-tail --case matmul-strided-fp32 \
  --output /tmp/operator-paired-plan.json
python3 benchmarks/operator_quality.py --preflight /tmp/operator-paired-plan.json \
  --host <ssh-alias> --output /tmp/operator-readiness
```

预检证据包含通过/失败/未知的阶段、可公开的组件名、设备与运行时身份、runner 哈希身份及控制核结果，不保留 SSH 参数或异常原文。每次正式执行仍重新预检，旧证据不是可复用的放行凭证。探针 SSH 等待最长 60 秒（或更短的后端超时）；这不证明远端 GPU 工作已被取消，也不提供设备独占。

## 两组共享源码，分别修复和验证

`--paired` 创建新的配对计划，固定 baseline 和 diagnosis 两组，无需同时指定 `--strategy`：

1. 每个任务只生成一次，把初始源码写到只读的 `shared/<case>/candidate.py`，记录源码哈希。
2. 两组各自读取同一份源码，重新物化主输入和两个独立种子输入，保持非连续布局。不会复制上一组使用过的数组。
3. 两组共享已预检并固定 GPU 的后端，但分别创建验证器、修复客户端和调用账本；只改变诊断开关。按任务序号交替先后顺序。
4. 两组分别执行完整 L2；失败后的新源码仍需重新验证。保存的两份 `candidate-1.py` 必须与共同初始源码哈希相同。

默认 `max_calls=2` 表示一次共享生成、每组最多一次修复，**一个配对任务最多实际调用三次**。生成费用只计入 `shared.costs` 一次，组报告的成本只包含该组修复，`total_actual_costs` 为共享加两组的实际总和。初始候选已经通过时修复调用为零。失败调用照计，未返回 usage 时 token 保持未知。每组账本独立，某组失败不会给另一组增加或减少调用额度。

`max_verifications` 对两组分别生效，默认每组两次。`campaign_seconds` 分别限制共享生成账本和各修复账本的调用活动，不是整个配对任务或 GPU 的硬超时。配对计划要求 `max_calls >= 2`，不以伪造一笔修复额度支持零预算。

```bash
python3 benchmarks/operator_quality.py --execute /tmp/operator-paired-plan.json \
  --host <ssh-alias> --output /tmp/operator-paired-run
```

执行目录必须不存在。计划冻结任务、模型配置、预算、代码及预检策略身份；源码变动后需要新计划。报告保留共享生成、每组候选/验证/修复及调用账本。`arms/<arm>/report.json` 沿用原质量报告格式，`report.json` 汇总共同初始状态和配对修复指标。

初始生成失败、预检失败和未知观测都保留完整任务分母。只有同一初始源码在两组均被确立为可修复的正确性失败，才计入 `eligible_failed_pairs`。初始观测冲突会标记 `initial_observation_conflict` 并排除修复收益计算。初始已经通过的候选不计为修复成功。比较仍是有限任务集上的描述，不提供统计显著性结论。

## 固定初始候选的修复对照

诊断扩展后，若需要检验某个既知缺陷，使用显式的固定候选计划。CLI 在规划时读取源码的原始 UTF-8 字节，将正文与 SHA256 冻结进新计划；执行时不再读取来源文件，也不继承旧验证结果或旧调用额度。源代码上限为每任务 32768 bytes，必须恰好覆盖选定任务。未传 `--case` 时由 `--initial-source` 的任务顺序确定范围。

```bash
quality_run_root=$(mktemp -d)
python3 benchmarks/operator_quality.py --paired \
  --initial-source layernorm-tail=/path/to/failed-candidate.py \
  --backend ssh --host <ssh-alias> --model "$LLM_MODEL" \
  --output "$quality_run_root/plan.json"
python3 benchmarks/operator_quality.py --preflight "$quality_run_root/plan.json" \
  --host <ssh-alias> --output "$quality_run_root/readiness"
python3 benchmarks/operator_quality.py --execute "$quality_run_root/plan.json" \
  --host <ssh-alias> --output "$quality_run_root/run"
```

计划 schema 为 `operator-seeded-paired-quality-plan-v1`，报告为 `operator-seeded-paired-quality-report-v1`。Python API 为 `make_seeded_paired_plan(initial_sources={case_id: source}, case_ids=[...], ...)`。默认 `max_calls=1`、`max_verifications=2`：每组最多一次修复、两次候选验证，一个任务总计最多两次模型调用。共享阶段只有后端预检和不可变候选保存，不创建生成模型或生成账本，共享模型调用与 tokens 均为零。

两组仍各自物化输入，重新验证初始候选、调用独立 Fixer，再做完整 L2。报告同时检查计划、共享候选和两组首发源码绑定；初始化失败不能套用旧的通过结果。保留 CRLF、Unicode 等原始源码字节，不用文本读回的换行归一化改变哈希。

此模式的首次 L2 表示**固定候选的缺陷复现**，不能当作更新模型的首发生成质量。历史来源的报告与计划哈希只作溯源，不能成为新一轮执行资格或预算恢复凭证。`--initial-source` 只用于规划或离线回放；执行和预检以冻结计划为准。

固定候选模式新增核心回归 38 项、CLI 回归 30 项，覆盖零共享生成、严格来源/预算/费用、初始化失败、两组独立输入、完整 L2、报告绑定，以及 CRLF/Unicode 源码字节保真。冻结源码后的完整回归为 **4775 通过、1 跳过**；Golden **4 项通过**，DSH 插件与宿主契约 **39 项通过**。原有生成配对路径保持兼容。

## 无模型的回放

配对回放文件按任务区分三个响应流：

```json
{
  "layernorm-tail": {
    "generation": ["严格 JSON 字符串，包含 code 和 rationale"],
    "baseline": ["Fixer 的完整代码块响应"],
    "diagnosis": ["Fixer 的完整代码块响应"]
  }
}
```

```bash
python3 benchmarks/operator_quality.py --paired --replay /tmp/paired-responses.json \
  --case layernorm-tail --output /tmp/operator-paired-replay
```

回放不做 SSH 预检，使用 CPU 候选验证接线。响应由回放作者指定，合成 token 带 `replay_fixture` 标签；它不能证明诊断、模型升级或 GPU 性能的收益。真实小规模 L2 对照通过后，再扩展十个任务；L3 性能测量仍单独进行。

## 验证记录（2026-10-23）

冻结源码后的完整离线回归：**4597 通过、1 跳过**；DSH 插件及宿主契约 **39 项通过**；Golden **4 项通过**。前一轮的六项守卫失败已清理。新增回归覆盖预检失败时零模型调用、GPU 固定与环境复用、同源/输入隔离、独立预算、共享成本与耗时、1024-token 修复预算、初始观测冲突及报告篡改拒绝。

十个任务的手写控制响应回放中，两组均首次完整 L2 **9/10**、修复后 **10/10**；共享生成 **10 次**，两组各修复 **1 次**，实际合计 **12 次回放调用**。一个共同初始失败在两组均修好，无初始观测冲突。该结果验证接线与成本归集，未调用真实模型或 GPU，也未测量原生性能。

随后完成了[两任务真实模型与 H20 配对验收](../research/operator-quality-real-paired-20261023.md)：两组首次完整 L2 均为 **1/2**，修复后均为 **2/2**，实际调用 **4 次**、**15,297 tokens**。共同初始失败来自 LayerNorm 漏处理列跨距；现有诊断返回 `unknown`，两组均依靠验证反馈修复，本次没有诊断独有修复收益。未运行 L3。
