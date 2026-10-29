# 列 stride 与单列编译失败的有界诊断

日期：2026-10-24。接续[边界修复验收](../research/operator-quality-boundary-fixes-20261024.md)中仍失败的三个候选。诊断进入现有 `operator_repair` / Fixer 链，不增加生成、重试或验证预算。

## 列 stride 识别扩展

原直线寻址规则保持不变。新增规则仅在可信套件、实际 CUDA 输入回读和失败数值比较共同确立二维非单位列 stride 后，识别以下组合：wrapper 的 `X = X.contiguous() if X.dim() != 2 else X`、独立 `shape[0]` / `shape[1]`、独立输出分配、主 load 与行 anchor load。

回读已证明 rank=2，因此该特定条件表达式走原输入分支；这不是对任意复制操作的豁免。无条件 `contiguous`、clone、不同条件、其他输入重绑定、复杂寻址和不明调用均返回 unknown。地址分析仍须证明实际主 load 使用 `row * row_stride + column`，并缺少输入列 stride；注释中出现或没有出现某个名称不构成证据。

## 原生编译节点证据

共享 `compile_evidence.py` 仅捕获精确类型的 `triton.compiler.errors.CompilationError`，并要求异常发生在实际 `module.kernel` 调用阶段。模块导入、参考执行、异常子类及拼接的错误文字不能产生证据。

证据只含 schema、producer、phase、完整候选 source hash、JIT helper 名、AST 节点 hash 与绝对位置、evidence hash。异常的 `src` 必须与候选中唯一的 JIT helper AST 对应，`node` 必须匹配该函数中的真实节点。允许 compiler source 省略 decorator，通过 AST 字段路径映射回完整源码；不从 traceback 或自由文本猜行号。

Verifier 再次绑定源码，并要求候选明确失败、输入保真回读成立、设备为 CUDA、张量处在同一 CUDA 设备，以及已知传输结果。参考错误和身份未知不传递编译证据。该字段部署在 runner 之前；生产协议同时覆盖证据模块和诊断模块的源码 hash。

## 单列标量转换假设

在线诊断还须核验同轮 L1 通过、失败 L2、冻结套件 hash/case/dtype/layout、实际二维单列输入及完整输入回读。窄 AST 分析证明 `.to(tl.float32)` 的 receiver 是没有被重绑定的 helper 参数，启动实参直接来自 wrapper 输入的列数。编译节点必须精确落在该转换、其方法属性或直接赋值上。

普通 tensor 的 `.to`、stride / pointer 参数、维度别名或重绑定、未知调用、其他控制流和绑定不明的启动均不支持。支持三份保留失败源码的行式与二维 tile 写法，不据此推断其他程序或编译器的行为。

finding 提供五处源码位置和修复义务：使用 `tl.cast` 等兼容标量的转换，保留逻辑归约长度、stride、mask、epsilon、输出 dtype 和所有验证门。状态始终是 `supported_hypothesis`，不自动修改候选或断言完整根因。新源码仍须 fresh L1、三份主输入、完整可信套件与确定性检查。

## 证据传递与上限

修复循环给 Fixer 传递完整的同轮验证结果，用于重新核验 L1 与诊断绑定；失败处方仍只读取失败项。Fixer 检查完整结果中的失败集合与修复请求一致后才调用模型，防止投影丢失权限事实或遗漏失败项。只读哈希检查用于一致性校验，不是证据签名。

源码最多 32 KiB，AST 最多 4096 节点。编译证据与标量规则限定 AST 深度 32 层；stride 寻址与依赖分析的递归最多 32 层，标量数据流最多 1024 步。每个 excerpt 最多 512 bytes、最多一项 finding、诊断包最多 16 KiB。超限、歧义或未知证据返回 unknown。无候选执行、机械替换、正确性授予、历史预算恢复或 L3 性能结论。

离线测试包含保留源码、已去除连接信息的历史失败元数据、合成 typed 事件及拒绝边界。合成事件只证明接线。真实编译证据和修复效果另以新 H20 计划验收，并保留原始请求、响应、调用账本和全部失败结果。
