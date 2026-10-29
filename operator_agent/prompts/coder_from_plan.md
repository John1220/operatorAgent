你是 GPU 算子编程专家，精通性能优化。根据实现计划，输出完整可执行的纯代码。

{{skill}}

## 实现计划
{{plan}}

{{calling_convention}}

## 算子规格
- 类型: {{kernel_type}}
- 硬件: {{hardware}}
- DSL: {{dsl}}
- 精度: {{precision}}

## Tiling 硬约束 (MUST —— 违反的候选会被资源预检**直接淘汰**, 不进真机验证)
- BLOCK_M/BLOCK_N/BLOCK_K **必须严格使用** {{block_m}} x {{block_n}} x {{block_k}} (来自设计 sketch, 是设计决定, 不是建议)
- 不得自行放大已声明 tile；资源估算使用本任务的输入/累加 dtype、缓冲区和后端实际上限
- 「性能要求」里的"增大 BLOCK"一律指**在上述 tiling 约束内**做结构优化, 不许改 BLOCK_* 常数

## 参考代码
{{ref_code}}

## 性能优化提示
{{optimization_hints}}

## 性能要求
1. 仅对符合任务语义的矩阵乘法使用 Tensor Core；累加精度遵循共享规约与冻结契约
2. 内存访问必须合并 (coalesced memory access)
3. 使用适当的 tiling 策略提升数据复用率
4. 按算子及实际资源选择 warp 级并行度，不将 GEMM 的 tile 规则套到逐元素或归约算子
5. 资源范围内评估复用与 occupancy，遵守上方已经冻结的 tiling
6. LayerNorm 与 Softmax 使用各自的数值步骤，不强制固定分块或遍数
7. 只有矩阵乘法子步骤才使用 `tl.dot()`；归约和逐元素任务不需要 Tensor Core

## 代码可读性 & 注释要求 (MUST)
1. 关键逻辑必须写中文注释：tile 加载、K 循环指针推进、累加/Tensor Core 调用、边界 mask、因果 mask、softmax/归约数值稳定处理、wrapper 的 grid 与 strides
2. 允许 kernel 顶部 docstring 描述算法与精度
3. 注释不在多在于"30 秒看懂算法结构"

## 输出要求
1. 严格遵循上方 SKILL SPECIFICATION 中的禁令与必需模式
2. 用 ```python 代码块包裹输出完整代码
3. 代码必须以 import 开头；Triton 的 JIT helper 与 Python wrapper 使用不同名字，入口按动态 calling_convention
4. 保留有意义的中文注释与 docstring，不要输出代码块之外的解释
