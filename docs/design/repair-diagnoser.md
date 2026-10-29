# 离线 Repair Diagnoser：证据驱动的修复诊断

日期：2026-10-16。接续[原样返回源码的真实修复失败](bounded-numerical-repair.md#5-获批真实执行模型原样返回失败源码修复守卫拒绝)。

## 1. 目标与状态

上次模型声称将单遍计算改为两阶段计算，却返回完全相同的失败源码。本阶段补齐
真实响应回归、AST 伪修复检查，以及带源码位置、独立反例和修复要求的诊断包。
**只完成本地实现、离线回放和上下文预览，没有新增模型、SSH 或 GPU 调用。**

不修补模型源码，不生成可信 Triton 答案模板，不改变参考、容差、suite、workload、
测量资格或确认/入库门。旧计划和父候选 claim 不删除、不续用、不重置。

## 2. 真实响应与伪修复守卫

- `tests/fixtures/repair_unchanged_response.json` 保存原始响应 2472 bytes，hash
  `b494c41c4270d4df52883762d03d6bda3e34bd1b6c31e9f4f0032e57bc287222`。
- `tests/fixtures/repair_unchanged_case.json` 保存冻结上下文与来源 hash：失败模型
  源码、契约、输入元数据和白名单反馈，无输入数组、密钥、真实端点或原始日志。

解码源码仍是 955 bytes、hash
`b3f95ac94f85e6027b54df920ef48425f200185c466eaeec89b21b1a539b8c95`。
回放响应不是重新调用模型或重测 GPU。原历史目录保持只读。

`bounded_repair.source_change` 分别记录精确原文 hash 和无位置属性的 Python AST
hash。原文相同仍用 `repair_unchanged_source`；原文不同但 AST 相同用新增固定原因
`repair_no_structural_change`。两个修复模式均拒绝，并保留原始响应和比较回执。
普通生成没有父源码，不使用修复比较。

AST 改动不等于语义修复：加无用变量也能改变 AST，但仍须 fresh L1/L2，失败不得
进入 L3。本版不做变量重命名归一化、代数化简或程序等价判定，也不通过理解任意
rationale 授予资格。`rationale_has_source_change_evidence` 仅表示存在 AST 改动，
不是对解释内容的语义认证。格式、注释及相同浮点值的不同拼写不能买到修复资格。

## 3. 狭窄且保守的源码模式识别

`operator_agent/repair_diagnoser.py` 纯分析数据，不 import/exec 模型源码。第一版
仅识别直线执行的单 JIT helper 与单 wrapper 模式：wrapper 从二维 X.shape 最后
一轴取 N，以 next_power_of_2(N) 取 block，传递 N/block/冻结 epsilon；helper
构造 lane/mask，masked load 的 other=0 后转 FP32，计算逻辑 N 归一化的均值，
中心化，再对**未重新屏蔽**的平方和归约。方差必须实际接入归一化和 FP16 输出
store 的 def-use 链，而非一段未使用的可疑代码。

重定义、分支、多 helper/入口、别名变体、复杂调用、其他归约轴，以及 N/block/
epsilon 绑定不明等均不支持。上限为源码 32 KiB、AST 4096 节点、finding 1 项、
每个 excerpt 512 bytes、诊断包 16 KiB。至少一条失败观察必须是 tail shape；
只有 power-of-two case 失败时不用此模式解释。其他写法，包括已 masked 的方差，
返回 unknown，不将其认证为正确。

未知或超限时 `state=unknown`、findings 为空，不产生 v2 诊断上下文。识别成功
也仅为 `supported_hypothesis`，不是完整程序语义、GPU 算术或布局证明；不能凭
误差与 shape 断言真实输入均值，或认定所有误差均由这一处引起。

## 4. 诊断包与独立反例

诊断绑定 source/feedback/contract/input_metadata/parent identity hash，包含：

- 10 处源码 line/column/end span、精确 excerpt 与 hash；excerpt 是不可信源码
  数据，不能成为执行指令；
- 与失败观察兼容的 tail shape、block 长度，以及关系式
  `variance_bias = (BLOCK_N - N) * mean**2 / N`；
- 修复要求：中心化平方和归约排除无效 lane，仍以逻辑 N 归一化，保留 wrapper、
  独立 stride、冻结参考与所有验证门。不提供候选模板或机械替换补丁。

反例只用内置三个标量 `[0,1,2]`：N=3、block=4、mean=1、总体方差 2/3，多出的
padding 方差 1/3，未屏蔽归约得到 1。使用冻结 epsilon 后，两种归一化结果的
最大差约 0.22474。由可信 Python 标量演算得到，不执行候选，不把 CPU 演算冒充
native CUDA 参考，也不把该形状加入原 suite 或当验收资格。

校验会从冻结输入重新计算整个诊断；重新 hash 后篡改位置、数值、修复要求或资格
仍被拒绝。hash 只防漂移，不防同 UID 恶意伪造全部父证据。

## 5. 显式 v2 接线，不自动扩充 v1

`SingleCandidateCoder(..., repair=..., diagnosis=...)` 只有显式提供且验证了
supported diagnosis 才用 `single-candidate-repair-v2`。新 prompt 强调诊断是假设，
rationale、注释或格式改动不是修复。新血缘绑定 diagnosis hash 和 source_change。
调用前后漂移、未知诊断、无 repair 的诊断及审计血缘不一致均拒绝。

默认普通生成和 repair v1 的声明/prompt/context 不自动加诊断。AST 伪修复守卫是
新增限制，生产 protocol 因而改变；历史原始事实不覆盖，也不可把旧快照当当前
实现。v2 仍仅有一个 `SingleCandidateRepair` caller、一次调用预算和一份新候选，
不继承父结果，不自动 Search/Fixer，不跳过完整正确性、测量、确认或入库门。

本阶段没有给 H20 执行 CLI 增加启用诊断的开关，没有生成新的可执行验收计划。

## 6. 只能离线的 preview

```bash
python -B benchmarks/repair_diagnosis.py \
  --case tests/fixtures/repair_unchanged_case.json \
  --response tests/fixtures/repair_unchanged_response.json \
  --output output/repair_diagnosis
```

入口拒绝 `--execute`，没有 endpoint、SSH、数组物化或 claim 操作。核对原始响应
hash、model/finish/envelope 与上下文绑定后，创建全新 0700 目录，只读冻结输入、
diagnosis、response_review、generation_preview 与 summary。输入有字节上限，
拒绝 symlink；输出不覆盖旧目录。summary 明确 executable_plan=false、approved=
false、model/SSH/GPU calls=0、repair_credit_reset=false。

本次预览目录为
`output/workload_codegen_validation/repair-diagnoser-local-kgpmcr2i/previews/repair-diagnosis-9stkyzx6/`，
diagnosis hash 为 `9db8bfd393d5e4494bde9c75e35a4fe5d08e3a3222060d9eb16066d3835d308c`，
v2 preview context hash 为 `006c266367d6ba61210a05af2941f2617cd62dd629661153bad8e03c8f72ff2f`。

**预览不是执行计划或授权。** 新上下文有源码片段、诊断和内置反例的三个标量，
未来外发必须披露这些新增内容，不能沿用此前 17906-byte 请求的批准。父候选
修复额度已耗尽，预览不会恢复它，不能复制父目录或换计划追加尝试。

## 7. 离线验证

新增 49 项回归，专项 379 passed：真实响应、源码 span、反例、unsupported/unknown、
重 hash 篡改、伪修复、新 prompt/血缘、调用前漂移零请求、一次预算、fresh L2
失败不进 L3、preview 不接触执行/额度。测试中的响应回放、CUDA 标签与回执均为
离线数据/假件，不构成真实修复成功或 H20 性能结论。

全量、串行 Golden 与历史不可变性结果保留在
`output/workload_codegen_validation/repair-diagnoser-local-kgpmcr2i/` 的最终总结中。

最终全量为 4163 passed / 1 skipped / 2 原有 nonfinite warnings，随后串行 Golden
4 PASS / 0 REVIEW / 0 FAIL / 0 SKIP；未刷新 registry、改容差或补测找绿。
3439 份受保护历史文件、原 ruler、registry 与已消费的 claim 未变；生产/测试源码
与验证前快照一致。production protocol 为
`7e14a361f243eb688e863b35d22a78e74272869da2ac85de8c4f49f0e12b308c`。
本阶段真实模型/SSH/GPU 派发均为 0，未创建执行计划，未 commit/push。

## 8. 后续离线接线：独立只读复核与 v2 预览清单

新增 `benchmarks/repair_diagnosis.py::audit_preview` 和 CLI：

```bash
python -B benchmarks/repair_diagnosis.py --audit <preview-directory>
```

只输出 JSON 到 stdout，完整一致返回 0；缺件、篡改、未知格式或协议漂移返回 1。
禁止同时传 case/response/output，不建输出目录，不改预览文件，不调用生成器写入
分支、模型客户端、SSH、数组物化或额度接口。审计报告应保存在预览目录外，不能
向旧证据目录塞入额外报告。

### 8.1 从冻结输入重算，不相信 producer summary 的通过标记

只读流程先限定文件清单，再读取冻结 case/response，调用可信诊断逻辑重新计算
diagnosis、response_review 与 generation context，并逐项核对持久化结果。不是
重新执行保存的生产者源码，不运行候选，也不信任模型解释。检查内容包括：

- 原始响应/source/feedback/contract/parent identity 绑定、源码位置和数学反例；
- prompt、policy、诊断血缘、context hash、source_change 及所有 false 资格字段；
- 当前生产 measurement protocol，冻结输入 hash、版本、文件清单和摘要；
- 读取前后完整文件 hash、模式和清单一致，防止复核过程中证据变化。

使用规范 JSON digest 区分类型，不能把 `false` 偷换成 `0` 或把计数换成 bool。
篡改后重新生成文件 hash/manifest hash，仍不能替代从冻结输入重算派生证据。
错误只保留固定原因，不回显任意文件正文、模型文字、路径或异常原文。

### 8.2 新预览格式与旧版兼容

新 writer 输出 `offline-repair-diagnosis-preview-v2`：保存 `producer-source.py`
数据快照，summary.files 绑定除 summary 外所有文件，summary.hash 再绑定整个
声明。候选修复上下文仍是 `single-candidate-repair-v2`；预览文件格式 v2 与模型
上下文版本不是同一概念。原诊断内容、prompt、生产协议、验证门与额度均未放宽。

写入前预先序列化并检查大小，再创建全新 0700 目录，逐件只读冻结，以最终 summary
作为完成标记。限制每文件 64 KiB、总计 256 KiB、最多 7 个固定名称文件；审计用
惰性 scandir，到第 8 个条目即拒绝，不遍历任意大目录。拒绝 symlink、目录代替
文件、可写或缺失文件、未知额外文件。unknown 诊断不得带 generation preview。

已有 v1 预览仍可在匹配的生产协议下重算核对，但没有生产者源码快照，只记录
`declared_hash_only_no_snapshot`。v2 记录
`snapshot_hash_verified_not_execution_attested`：仅确认快照 hash 一致，不证明
这个程序历史上确实执行过，更不会执行这份源码来验证它。

**completed=true 仅代表冻结数据内部一致。** 同 UID 可以改掉全部原始数据并
重算所有派生物，内部一致性不是历史真实性、恶意代码隔离、人类授权或 GPU
正确性证明。报告始终 executable_plan=false、approved=false、restored_eligibility=
false、repair_credit_reset=false。生产协议漂移时明确拒绝，不能把新解释当旧资格。

### 8.3 本轮实际复核

历史 v1 目录保持不变；新增 v2 目录在
`output/workload_codegen_validation/diagnosis-audit-local-3q5rz0ju/previews/repair-diagnosis-_errua30/`。
两份独立只读审计均 completed=true、evidence_consistent=true；v1 与 v2 分别保留
上述不同的生产者身份限定。诊断 hash 仍为
`9db8bfd393d5e4494bde9c75e35a4fe5d08e3a3222060d9eb16066d3835d308c`，没有生成新候选。

新增 43 项离线回归，专项 422 passed，覆盖 supported/unknown、旧版兼容、重新
hash 的证据篡改、摘要类型偷换、缺件/额外件、目录/字节上限、证据读取中变化、
协议漂移、生产者源码只作数据，以及 CLI 零写入/零执行/零额度操作。
最终全量、串行 Golden、历史与 claim 不可变性记录在本轮目录的 final-summary 中。

本轮全量 4206 passed / 1 skipped / 2 原有 nonfinite warnings；随后串行 Golden
4 PASS / 0 REVIEW / 0 FAIL / 0 SKIP，未更新 registry、改容差或补测找绿。
3511 份历史文件、原 ruler、registry 与已消费 claim 不变；本轮只改离线 producer/
auditor、测试和文档，生产 protocol 仍为
`7e14a361f243eb688e863b35d22a78e74272869da2ac85de8c4f49f0e12b308c`。
真实模型/SSH/GPU 派发为 0，未创建执行计划、重置额度或 commit/push。
