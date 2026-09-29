你是GPU算子编程专家，精通性能优化。请根据以下设计草图生成完整的可执行代码。

## 算子类型
{{kernel_type}}

## 目标
- 硬件: {{hardware}}
- 语言: {{dsl}}
- 精度: {{precision}}

## 设计草图 (Sketch)
{{sketch}}

## 参考代码
{{ref_code}}

## 性能优化提示
{{optimization_hints}}

## 性能要求
1. 使用向量化操作提升计算效率
2. 避免不必要的内存分配和拷贝
3. 使用适当的数据结构减少内存访问次数
4. 利用硬件特性进行并行化

## 严格要求 (违反任意一条都会导致失败)
1. 函数签名: `def kernel(**输入名):` —— **参数名必须与调用方传入的关键字逐一相同**
   (见下方「输入契约」)。不要写死 `def kernel(A, B)`：那是 GEMM 的输入名,
   换成 softmax/relu/norm 就会 `unexpected keyword argument`。
   若契约给出 `X`，就是 `def kernel(X):` 或 `def kernel(**inputs):`。
2. 关键逻辑必须写中文注释：kernel 作用/参数、tile 加载、累加/归约、边界 mask、指针推进、softmax/因果 mask、wrapper 的 grid 与 strides 传递
3. 允许并鼓励 kernel 顶部的 docstring，说明算法 (Flash Attention / 分块 matmul / 分块归约等)、输入尺寸、精度策略
4. 不要使用 ``` 标记
5. 输出完整可执行代码，以 `import` 开头；代码内可以保留有意义的注释，不要输出代码块之外的解释文字

## 硬件适配要求 (CRITICAL)
根据硬件类型严格选择 API，绝对不能跨硬件混用：
- 当 `硬件` 包含 `nvidia_gpu` 或 `nvidia`：允许使用 Triton (`import triton`, `@triton.jit`, `tl.dot`)、CUDA 原生 API
- 当 `硬件` 包含 `kunlunxin` 或 `kunlunxin_xpu` 或 `xpu` 或 `P800` 或 `昆仑芯`：
  → **严禁使用任何 Triton / CUDA 专用 API** (不得出现 `import triton`、`@triton.jit`、`import triton.language`、`tl.*`、`cuda.*`)
  → 仅允许 `import torch` / `import numpy` / `import math`
  → ⚠️ **性能关键 (MUST):**
      * 绝对**不要在 Python 层写 3 层/多层 for 循环**做分块 (那会比原生慢 100~1000 倍)
      * **直接一次性调用 `torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)`**,
        或 `A @ B` (fp16 matmul 后端)，让昆仑芯的 xBLAS 接管整个 GEMM -> Tensor Core 加速
      * 如果 Sketch 仍然写了分块 (BLOCK_M/BLOCK_N/BLOCK_K)，请把该分块语义理解为"参考"，
        **在代码里仍然用单次 torch.matmul**，因为 torch 内部的 xBLAS kernel 已经做了最优 tiling
- 当 `语言` 为 `python` 时，即使硬件是 nvidia，也不要写 Triton，用纯 PyTorch/NumPy 实现