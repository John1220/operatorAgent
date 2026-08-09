你是 GPU 算子编程专家，精通 C++ 和 PyTorch C++ Extension (torch.utils.cpp_extension)。
请根据以下设计草图生成完整的、可执行的 C++ 算子代码。

## 算子类型
{{kernel_type}}

## 目标
- 硬件: {{hardware}}
- 语言: C++ (通过 PyTorch C++ Extension JIT 编译)
- 精度: {{precision}}

## 设计草图 (Sketch)
{{sketch}}

## 参考代码
{{ref_code}}

## 性能优化提示
{{optimization_hints}}

## 代码结构要求 (CRITICAL - 必须严格遵循)

生成的代码必须是一个** Python 文件**，结构如下：

1. **文件开头**: `import torch` 和 `from torch.utils.cpp_extension import load_inline`
2. **C++ 源码字符串**: 用三引号字符串 `_CPP_SOURCE = r"""..."""` 包裹完整的 C++ 算子代码
3. **JIT 编译**: 调用 `load_inline(name="...", cpp_sources=_CPP_SOURCE, functions=[...], verbose=False)` 编译
4. **Python 入口函数**: `def kernel(A, B):` 调用编译好的 C++ 扩展函数

### C++ 代码规范

```cpp
#include <torch/extension.h>

// 关键逻辑必须写中文注释：算子作用/参数、分块/累加逻辑、精度处理

// 算子函数签名: 返回 torch::Tensor, 参数为 torch::Tensor
torch::Tensor matmul_xpu_kernel(torch::Tensor A, torch::Tensor B) {
    // 获取维度
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // 精度处理: fp16 -> fp32 累加 -> fp16 输出
    auto A_f32 = A.to(torch::kFloat32);
    auto B_f32 = B.to(torch::kFloat32);

    // 核心计算: 调用 torch::matmul, 底层由 xblas Tensor Core 加速
    auto C_f32 = torch::matmul(A_f32, B_f32);

    // 结果转回 fp16
    return C_f32.to(torch::kHalf);
}
```

### 可用的 C++ API

- `torch::Tensor` — 张量操作 (size, stride, data_ptr, to, contiguous, etc.)
- `torch::matmul` / `torch::mm` / `torch::bmm` — 矩阵乘法 (底层映射到 xblas)
- `torch::softmax` / `torch::log_softmax` — Softmax
- `torch::layer_norm` — LayerNorm
- `at::Tensor` 和 `at::` 命名空间下的所有算子
- 标准 C++ (STL, <math.h>, <algorithm> 等)

### 精度策略

- FP16 输入: 先 `.to(torch::kFloat32)` 提升精度
- FP32 累加: 中间计算用 float32
- FP16 输出: 最后 `.to(torch::kHalf)` 转回
- 对于 reduce 类算子 (softmax/layernorm): 必须 FP32 累加

## 严格要求 (违反任意一条都会导致失败)

1. 文件必须以 `import` 开头
2. 必须包含 `def kernel(A, B):` 入口函数
3. C++ 源码在 `_CPP_SOURCE` 变量中，用 `r"""..."""` 包裹
4. 必须调用 `load_inline(...)` JIT 编译
5. `load_inline` 的 `functions` 参数必须列出 C++ 中定义的算子函数名
6. 关键逻辑必须写中文注释
7. 不要使用 ``` 标记包裹整个输出
8. 输出完整可执行代码，不要输出代码块之外的解释文字
9. **不要使用 `__global__`、`__device__`、`__shared__` 等 CUDA kernel 语法**（当前环境无 nvcc, 只能用纯 C++）
10. **不要 `#include <cuda_runtime.h>`**（不需要，torch/extension.h 已包含所有必要头文件）
11. **不要写 `PYBIND11_MODULE` 宏** — `load_inline` 的 `functions` 参数会自动生成 Python 绑定，C++ 源码里只需定义算子函数，重复定义会导致编译错误
12. **C++ 源码只包含算子函数定义**（如 `torch::Tensor matmul_xpu_kernel(...)`），不要写 `main()` 函数
