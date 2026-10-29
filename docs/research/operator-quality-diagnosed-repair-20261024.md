# 重复跨距与单列编译错误：真实诊断修复对照

日期：2026-10-24。接续[上一轮边界验收](operator-quality-boundary-fixes-20261024.md)仍出现的列 stride 遗漏和单列 JIT 标量转换错误。

两类错误已接入有界诊断与真实修复流程。使用 `deepseek-v4.1-flash-1017` 和 H20 的新对照中，**基线、诊断两组各修好 2/2 个案例，诊断覆盖 2/2，额外修复收益为 0**。这证明新增诊断能绑定真实证据并进入模型请求，尚不能证明它提高总体修复成功率。

## 真实配对结果

| 保留缺陷 | 基线：初始 → 一次修复后完整 L2 | 诊断：初始 → 一次修复后完整 L2 | 基线 / 诊断调用 | 基线 / 诊断 API tokens |
| --- | --- | --- | ---: | ---: |
| `layernorm-large-mean`：输入列 stride 缺失，含 rank=2 身份条件分支与 anchor load | 失败 → 通过 | 失败 → 通过 | 1 / 1 | 5,951 / 8,236 |
| `layernorm-constant-strided`：单列输入的 `n_cols.to(tl.float32)` 编译失败 | 失败 → 通过 | 失败 → 通过 | 1 / 1 | 6,243 / 7,901 |

符合条件的共同初始失败为 **2 对**，两组初始观测冲突 **0 对**；诊断独有成功、基线独有成功均 **0 对**，最终未知 **0**。四臂分别重新物化输入和验证原源码，没有沿用历史失败或通过凭据。每份修复均重新通过 **3 份主输入、12 个可信套件用例和确定性检查**。

列 stride 初始故障均发生在 `auto-cols32-rows8-fp32-random-strided`，256 个元素全部不匹配，最大误差 `4.74771785736084`。单列初始故障发生在 `auto-cols1-rows8-fp32-random-contiguous`，有实际编译节点证据，没有数值比较数据；两类失败分别使用数值与编译诊断条件。

本轮从已选定的历史失败源码开始，**没有重新生成初始候选**，不能将已知缺陷复现写成更新模型的首发生成失败，也不是新的十任务总体成绩。

## 新增实现与真实证据

[实现说明](../design/compile-repair-diagnosis.md)记录具体边界：

- 列 stride 新规则要求已核验二维 CUDA 输入、非单位列 stride 与失败数值，识别特定身份条件分支、独立 shape 字段、输出分配和双 load。复制、clone、其他条件或不明寻址保持 unknown。
- 新共享模块只捕获精确原生 `CompilationError`，绑定实际 JIT helper 的源码、AST 节点及绝对位置。只有候选调用阶段、已核验输入回读、CUDA 设备和已知传输结果才能传递该字段；错误文字不产生证据。
- 单列诊断还绑定冻结套件与实际输入列数，证明被调用 `.to` 的参数直接来自输入列数，且未被重绑定。真实 native 节点是源码第 26 行的 `n_cols.to` 属性，位置 `26:8–26:17`。
- Fixer 接收完整同轮验证结果，复核 L1 与失败集合，再重新计算诊断绑定。该修正避免仅传失败项时丢失 L1 通过事实。

两组列 stride 修复都补齐独立的输入、输出行列 stride，保留 anchor、平移域的均值和居中方差、mask、FP32 转换与 epsilon `1e-5`。两组均移除了在冻结 rank=2 输入上恒为身份的条件分支；该修改不能推断为保持任意其他 rank 的行为。

列 stride 基线解释误称原 wrapper 强制执行 `contiguous()`；原条件分支在 rank=2 时实际返回原 X。诊断组对此解释准确。此为单例解释质量观察，验收仍取决于源码与 fresh 完整 L2。

单列两组修复的唯一 AST 改动均为 `n_cols.to(tl.float32)` → `tl.cast(n_cols, tl.float32)`；其他计算、wrapper、stride、mask、分配与 epsilon 保持不变。忽略注释和位置后，两组修复 AST 相同。独立源码审查未执行候选，实际通过依据来自新 H20 验证。

两份诊断请求中的完整 JSON 均与对应 `repair.json` 第一轮对象逐字段一致，基线请求没有该诊断段。source/contract/feedback 绑定、原始 HTTP 请求响应字节与 hash、模型返回身份及 API usage 均经独立重算。

## 预算、预检异常与证据保留

新计划每臂最多一次修复、两次候选验证：正式对照合计 **4 次模型请求、8 次候选验证、28,331 API tokens**。无缺失或估算 usage。基线合计 12,194 tokens，诊断合计 16,137 tokens；费用未作为本轮优化目标。配对执行约 **910.8 秒（15.2 分钟）**，不含准备、测试和探测。

首次零模型探测在预检阶段得到 `probe_transport / RuntimeError / unknown`，在候选执行前停止，模型调用和候选验证均为 0。原投影未保存更具体的传输异常，**原因未确立**；后续只读诊断通过不能反推首次失败原因。

首次失败记录没有覆盖。另行冻结最多一次只读预检诊断、零模型零候选验证；随后冻结新的零模型探测计划，最多一次候选验证。第二次探测确实复现单列编译失败并产生有效诊断，完整 L2 仍保持失败；然后才执行原先尚未启动的配对计划。实际候选验证合计 **9 次**，模型上限始终为 **4 次**。

正式计划 hash：`20207314a446dd4722273481a805a5b60b4ca70d323a4f423f8c75199a7f9a1d`。第二次探测计划 hash：`2c7b5e0e51a2c3a6be01bd93317002f58556256cbf8bbd7a4acc56ff5277ef0f`。

H20、CUDA 13.0、PyTorch `2.14.0+cu130`、Triton 3.8.0、NumPy 2.2.6 与 15 个部署文件均重新核验。每个 case 的两臂共用固定 GPU 后端并各自重新验证；复用现有环境，未安装依赖。未运行 L3，不提供性能、GPU 时间或部署资格结论。

## 开发验证与后续

新增 **285 项**离线正反例；冻结源码后完整回归 **5,207 通过、1 跳过**，保留 2 条既有非有限值测试警告。Golden **4 项通过**，DSH 插件与宿主契约 **39 项通过**。生产身份覆盖 54 个文件，源码及测试快照共 119 个文件；731 份历史证据经哈希核验保持不变。

参考、容差、完整正确性套件、调用上限与准入规则没有放宽。诊断只产生 `supported_hypothesis`，不机械改写候选、不授予通过、不恢复历史资格。旧日志缺少 typed 编译证据时仍返回 unknown。

下一阶段应优先收集首次修复仍失败、误修或多个缺陷并存的独立样本，在相同新预算下比较额外修复率。继续增加已能被两组一次修好的简单样本，难以判断诊断是否提高了成功率。

完整证据：[配对报告](/private/tmp/operator-diagnosed-repair-20261024-sDdhVo/run/report.json)、[独立验收](/private/tmp/operator-diagnosed-repair-20261024-sDdhVo/validation.json)、[请求与源码复核](/private/tmp/operator-diagnosed-repair-20261024-sDdhVo/request-source-review.json)、[原生节点复核](/private/tmp/operator-diagnosed-repair-20261024-sDdhVo/native-node-review.json)、[续行预算](/private/tmp/operator-diagnosed-repair-20261024-sDdhVo/continuation-policy.json)、[源码快照](/private/tmp/operator-diagnosed-repair-20261024-sDdhVo/source-snapshot-sha256.json)。
