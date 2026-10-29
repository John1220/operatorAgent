# 算子生成与修复质量闭环

当前实现把「统一生成规约、诊断驱动修复、固定任务对照」接入现有 Python 核心和 DSH 的 MCP 工具。更换模型后仍需独立验证；规约和诊断本身不授予正确性、性能或入库资格。

后续增加的[真机预检与同候选配对评测](operator-quality-paired-evaluation.md)会在模型调用前检查现有 SSH 环境，并让两组共享一次生成的源码及成本。原有两次独立生成的 baseline/diagnosis 对照入口保留，评估诊断时优先使用 `--paired`。

最新[重复故障 H20 修复对照](../research/operator-quality-diagnosed-repair-20261024.md)已验证 rank=2 条件分支下的列 stride 与单列原生编译诊断：两组各一次修好 2/2，诊断覆盖 2/2，额外修复收益 0。该结果验证真实证据接线，尚不证明总体通过率提升。

## 共享生成规约

`operator_agent/generation_spec.py` 是唯一的公共规约正文。`PromptManager` 将它注入 Designer、Planner、Coder、Fixer 和 Search 的提示词；单候选有界生成器使用同一正文。规约身份包含版本和正文 SHA256，进入评测计划及执行协议身份。

规约约束实际调用签名、输出分配、独立 stride、尾部 mask、累加精度、逻辑归约长度、LayerNorm 稳定方差与 Softmax 稳定归约。模板里与这些规则冲突的固定 tile 和通用算法要求已清理。实际任务与冻结参考契约始终优先，标量输入和调用方明确指定的参数必须保留。

针对真实验收暴露的边界，规约进一步说明：JIT 中可能被特化的标量使用 `tl.cast` 等标量兼容转换；大偏置 LayerNorm 可在契约允许等价统计时先平移，再在平移域中计算居中残差、方差和输出，避免重建大均值后再次相减。输出分配尺寸必须在 Python wrapper 作用域中定义；这些指导不改变 epsilon、精度或验收规则。

单候选预检支持直接 `torch.empty` / `torch.empty_like` 分配调用的合法 `memory_format=` 常量，`preserve_format` 仅用于 `empty_like`。常量不能逃逸为别名或可调用对象，计算 API 仍不允许。分配格式与实际形状、布局是否兼容，以及候选数值是否正确，仍由独立完整 L2 验证。

默认正确性 profile 的 LayerNorm、Softmax、ReLU 使用单个 `X` 输入。冻结 shape 时按已检查的 profile 派生输入，修正旧布局表多派生 weight/bias 的问题。未声明该 profile 的旧任务保持原布局语义。

## 诊断驱动修复

`operator_repair` 的每轮流程为：验证 → 失败资格检查 → 源码与反馈诊断 → Fixer → 源码变化检查 → 重新验证。

诊断绑定当前源码、冻结契约和反馈的哈希；Fixer 调用前重新计算绑定。当前在线识别器支持已声明 LayerNorm 中「masked load 后中心化，归约时遗漏尾部 mask」、[保真布局与源码寻址共同确立的列跨距缺失](stride-repair-diagnosis.md)，以及[绑定原生编译节点与单列输入的标量转换失败](compile-repair-diagnosis.md)。列 stride 规则还支持已证明 rank=2 的身份条件分支和 anchor load。诊断提供具体位置、独立标量见证或 typed 编译证据及修复义务。它支持显式原生参考的 fp16 路径，以及已检查的内置 fp32 profile。其他模式返回 `unknown`，仍可由原有失败处方指导修复。

数值诊断要求充分且一致的比较数据；编译诊断要求精确原生异常类型、源码与 AST 节点绑定，不能靠报错文字推断。完整同轮验证结果传给 Fixer 重新核验，失败集合须与处方一致。传输结果未知、参考身份未知、输入保真未证实或性能观测缺少可行动资格时，修复暂停。诊断不执行或改写候选，不补回历史预算，不改变参考、容差和测量门槛。

源码相同，或只改变注释/格式而 AST 相同，会停止并记录原因，不重复验证已失败候选。更改源码后必须重新验证。`full_l2_passed` 单独记录完整套件事实；空结果、只有 L1 或缺少完整套件不能作为成功。

MCP 返回紧凑诊断和证据文件位置；完整位置、见证及逐轮反馈保存在 `repair-N.json`。`operator_prepare` 和 `operator_repair` 还返回共享规约身份，便于审计。

## 固定任务和对照

`operator_agent/quality_suite.py` 固定 10 个任务：3 个 Matmul、4 个 LayerNorm、3 个 Softmax，覆盖 fp16/fp32、整齐与尾部形状、非连续布局、大均值小方差、常量和极端值。每个任务验证一个主挑战输入、两个独立随机种子输入，以及原有完整可信正确性套件；不替换或缩减套件。

任务清单固定 shape、dtype、pattern、layout、种子、参考契约和套件身份。种子与具体输入值不进入生成请求；种子在可复现清单中公开，不能视为秘密测试集。该有限集合的结果不代表所有形状或工作负载。

`benchmarks/operator_quality.py` 默认只生成计划，既不请求模型，也不执行候选：

```bash
python3 benchmarks/operator_quality.py --output /tmp/quality-plan.json
```

默认每任务最多 2 次模型调用、每次预留 4096 输出 token、2 次候选验证。生成和 Fixer 共用持久调用账本，失败、缺失响应和无效修复都计入成本。请求/响应字节、调用超时和模型调用活动时间也有上限。这些限制不等于精确账单上限、整个任务的硬超时或 GPU 时间上限。

`baseline` 和 `diagnosis` 两组均使用共享生成规约及相同验证规则；差别仅为是否加入结构化诊断。此对照测量诊断的增量效果，不直接测量新旧规约的差别。真实模型与旧模型也可以分别冻结计划比较；除 `model`、`strategy` 外的控制项和任务必须一致，修改聚合数字会被报告重算检查拒绝。若需分别归因，比较时一次只改变一个因素。

离线回放输入是「任务 ID → 原始模型响应字符串列表」的 JSON。首条响应格式为严格 JSON `{"code": "完整源码", "rationale": "说明"}`；修复响应沿用 Fixer 的代码块格式。回放执行 Python/NumPy 候选，token 成本为带 `replay_fixture` 标签的合成零值，不能用于估算真实模型成本。

```bash
python3 benchmarks/operator_quality.py --replay /tmp/responses.json \
  --strategy baseline --output /tmp/quality-baseline
python3 benchmarks/operator_quality.py --replay /tmp/responses.json \
  --strategy diagnosis --output /tmp/quality-diagnosis
python3 benchmarks/operator_quality.py --compare \
  /tmp/quality-baseline/report.json /tmp/quality-diagnosis/report.json
```

真实运行必须显式选择 SSH、冻结模型配置与主机别名，再执行计划。凭据沿用环境变量或仓库外的密钥文件，不进入计划：

```bash
python3 benchmarks/operator_quality.py --backend ssh --host <ssh-alias> \
  --model "$LLM_MODEL" --strategy diagnosis --output /tmp/quality-real-plan.json
python3 benchmarks/operator_quality.py --execute /tmp/quality-real-plan.json \
  --host <ssh-alias> --output /tmp/quality-real-run
```

执行目录必须不存在；其中保留计划、模型调用账本、候选源码、验证结果、诊断、失败记录及报告。计划与源码身份漂移时要求重新冻结。哈希用于检查一致性，不提供签名认证。执行沿用现有同 UID 后端，不提供独立设备隔离；相同 SSH 别名也不自动证明相同 GPU 状态。

报告保留全部任务分母，统计首次、最后一次及预算内完整 L2 通过率，以及所有尝试的调用数、token、耗时和每个成功任务的成本。未知成本保持未知。当前执行器只执行 L2，原生性能显示 `not_measured`、比率为 `null`；mock/replay 永不取得原生性能信用。性能与部署结论仍需另行执行既有 L3、确认和入库流程。

最初接线阶段使用离线测试与 CPU 回放，只证明接线、规则及计费归集工作正常。当时尚未测量更新模型的真实 GPU 结果；后续独立真实实验见上方验收记录，有限样本不提供总体提升结论。

## 开发验证记录（2026-10-22）

相关回归 283 项、DSH 插件及宿主契约 39 项通过。CPU 回放对照使用相同手写控制响应，两组均首次 9/10、修复后 10/10、共 11 次回放调用，证明失败成本与重新验证已接通。Golden 在并行负载时有两项计时超窗，空闲复测四项全部通过；未刷新基线或修改阈值。

完整回归首轮为 4467 通过、20 失败、1 跳过。更新本次增加的协议文件清单，并让未声明 seed 的旧后端调用省略可选 `input_seed` 后，相关失败均通过复测。当时剩余 6 项既有守卫失败：后台进程未统一经过 exec_guard；工具审计 action 被 runner 静态扫描当作远端载荷；NVIDIA/XPU 的 `inputs_spec` 键不一致；CLI 示例固定临时路径；以及复用 runner 守卫的 pluggable-reference 检查。

后续[配对评测实现](operator-quality-paired-evaluation.md)已清理这六项：可信后台 worker 统一从 exec_guard 启动；扫描器追踪实际发送的载荷；XPU 对不支持的 seed 输入明确拒绝并要求重传具体输入；CLI 示例使用私有随机目录。相应语义和守卫回归已通过，当前验证记录见配对评测文档。
