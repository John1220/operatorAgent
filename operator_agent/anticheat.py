"""反作弊模块
防止生成的 kernel 通过回退调用原生算子 (torch.matmul / F.softmax 等) 蒙混过关

两道防线:
1. static_audit: 静态审计 - 在代码执行前扫描禁用 API 调用模式 (按 DSL 分流)
2. shield_torch: 运行时屏蔽 - 执行期间将 torch 原生算子替换为抛错桩

规则按 DSL 区分:
- triton/cuda_c/opencl/sycl: 禁止 torch 计算 API 与 NumPy 计算回退
  (wrapper 只允许分配内存/启动 kernel，不允许在 CPU 上算出结果)
- python: NumPy 是实现介质所以允许，仅禁止 torch 计算 API
"""

import re
from contextlib import contextmanager
from typing import Optional


class CheatDetectedError(RuntimeError):
    """验证期间检测到禁用的原生算子调用"""
    pass


# ─── 静态审计规则 ────────────────────────────────────────────

_TORCH_COMPUTE = (
    r"\btorch\s*\.\s*(matmul|mm|bmm|baddbmm|addmm|einsum|softmax|log_softmax|"
    r"conv1d|conv2d|conv3d|conv_transpose1d|conv_transpose2d|conv_transpose3d)\s*\("
)
_TORCH_FUNCTIONAL = r"\btorch\s*\.\s*nn\s*\.\s*functional\b"
_F_COMPUTE = (
    r"\bF\s*\.\s*(softmax|log_softmax|linear|conv1d|conv2d|conv3d|"
    r"scaled_dot_product_attention|layer_norm|batch_norm)\s*\("
)
_NUMPY_COMPUTE = r"\b(?:np|numpy)\s*\.\s*(matmul|dot|einsum|vdot|tensordot)\s*\("

_TORCH_RULES = ("torch 原生算子", [_TORCH_COMPUTE, _TORCH_FUNCTIONAL, _F_COMPUTE])
_NUMPY_RULES = ("NumPy 原生算子", [_NUMPY_COMPUTE])

_GPU_DSL_RULES = [_TORCH_RULES, _NUMPY_RULES]    # 真正 GPU DSL: 两者都不允许
_PYTHON_DSL_RULES = [_TORCH_RULES]              # Python DSL: 允许 NumPy, 禁止 torch


def _strip_non_kernel_regions(code: str) -> str:
    """
    剔除代码中「不算做 kernel 实现」的区域，避免误报：
    1. `if __name__ == "__main__":` 之后的所有行 (测试代码 / reference 比较)
    2. Module-level docstring (kernel 只关心 def kernel/fuser 等函数体)
       → docstring 里常写 "参考结果用 torch.matmul 验证" 这种说明文字
    返回: 适合进行 static_audit 的「核心代码片段」
    """
    if not code:
        return code

    # 1. 去掉 __main__ 主函数 (从该行到末尾)
    m = re.search(r'^\s*if\s+__name__\s*==\s*["\']__main__["\']\s*:', code, re.MULTILINE)
    if m:
        code = code[:m.start()]

    # 2. 去掉三引号 module-level docstring (开头的 """ 或 ''')
    #    仅匹配文件最开头的 docstring，不处理函数内部 docstring
    stripped = code.lstrip()
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            end = stripped.find(quote, len(quote))
            if end >= 0:
                # 移除开头 docstring，保留剩余代码
                code = stripped[end + len(quote):]
                break

    return code


def _hardware_allows_torch_compute(hardware: str) -> bool:
    """
    对于「昇腾 NPU / 其他非 NVIDIA 自研加速卡」+ python DSL 组合：
    此时根本无法写 Triton/CUDA kernel，唯一能激活厂商自研 Tensor Core 的路径
    就是调用 torch.matmul / torch.mm 等高层 API (由厂商的 torch 后端接管下放)。

    在这种组合下如果还禁止 torch.* compute，等于强制写 6 层嵌套 Python 循环，
    不仅性能差 1000 倍、还根本不可能通过 benchmark。因此这里把这些硬件列入白名单：
    对它们 torch.matmul 等不算「作弊」，而算「正确的硬件调用方式」。
    """
    if not hardware:
        return False
    hw = hardware.lower()
    return any(tok in hw for tok in (
        "ascend", "npu", "昇腾", "huawei",          # 昇腾 / 华为
        "cambricon", "mlu",                         # 寒武纪
        "hygon", "dc", "海光",                      # 海光
        "loongson", "ls",                           # 龙芯
        "phytium", "ft",                            # 飞腾
        "zhaoxin", "兆芯",                          # 兆芯
        "sunway", "神威",                           # 神威
        "rv", "riscv", "玄铁",                      # RISC-V 生态
    ))


def static_audit(code: str, dsl: str = "python", hardware: Optional[str] = None) -> list[str]:
    """
    静态审计: 扫描代码中禁用的原生算子调用

    Args:
        code: 待审计的 kernel 代码
        dsl: 目标 DSL (python 与其他 GPU DSL 规则不同)
        hardware: 目标硬件 (新增, 可选)；用于对白名单自研加速卡放开 torch.* 计算

    Returns:
        违规描述列表 (空列表 = 通过)
    """
    rules = _PYTHON_DSL_RULES if (dsl or "").lower() == "python" else _GPU_DSL_RULES

    # ── 硬件白名单：昇腾等自研加速卡 + 任何 DSL，放开 torch compute 限制 ──
    #   cuda_c DSL: C++ 源码中的 torch::matmul / at::matmul 等 (正则匹配 Python 的 torch.matmul,
    #   不匹配 C++ 的 ::, 天然不报); 且这些硬件的 Tensor Core 只能通过 torch API 激活
    if _hardware_allows_torch_compute(hardware):
        # 只保留 numpy 禁令 (python DSL 本来就允许 numpy，因此实际上两者均无)
        rules = [] if (dsl or "").lower() in ("python", "cuda_c") else [_NUMPY_RULES]

    # 审计范围: 去掉 __main__ 测试段与模块级 docstring
    code_for_audit = _strip_non_kernel_regions(code)

    violations = []
    for label, patterns in rules:
        for pattern in patterns:
            for m in re.finditer(pattern, code_for_audit):
                # 换算成原始 code 中的行号: 从原始 code 的起点数到 m.start() 即可
                # 由于 _strip_non_kernel_regions 目前只删除尾部 (__main__) / 开头 (module docstring)，
                # 用 code_for_audit[:m.start()].count("\n") + offset 即可
                offset_lines = 0
                if code_for_audit is not code:
                    # 计算 code_for_audit 在原始 code 中的起始行号 (用于 module docstring 删除偏移)
                    idx_in_orig = code.find(code_for_audit[: min(len(code_for_audit), 80)].strip())
                    if idx_in_orig > 0:
                        offset_lines = code[:idx_in_orig].count("\n")
                line_no = code_for_audit[:m.start()].count("\n") + 1 + offset_lines
                violations.append(
                    f"第 {line_no} 行: 检测到 {label} 调用 `{m.group(0)}`"
                )
    return violations


# ─── 运行时屏蔽 ──────────────────────────────────────────────

# 执行期间需要替换为抛错桩的 torch 函数
_SHIELD_TARGETS = {
    "torch": [
        "matmul", "mm", "bmm", "baddbmm", "addmm", "einsum",
        "softmax", "log_softmax",
        "conv1d", "conv2d", "conv3d",
        "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    ],
    "torch.nn.functional": [
        "softmax", "log_softmax", "linear",
        "conv1d", "conv2d", "conv3d",
        "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
        "layer_norm", "batch_norm", "gelu", "relu",
        "scaled_dot_product_attention",
    ],
}


def _make_blocked(qualified_name: str):
    def _blocked(*args, **kwargs):
        raise CheatDetectedError(
            f"反作弊: 验证执行期间禁止调用原生算子 {qualified_name}()"
        )
    _blocked.__name__ = f"blocked_{qualified_name}"
    return _blocked


@contextmanager
def shield_torch(hardware: Optional[str] = None):
    """
    运行时屏蔽上下文: 将 torch 原生计算算子临时替换为抛 CheatDetectedError 的桩，
    退出时恢复原实现。torch 未安装时为 no-op。

    参数 (新增):
        hardware: 目标硬件，用于对白名单自研加速卡 (昇腾/寒武纪等) 关闭屏蔽。
                  这些硬件只能通过 torch.matmul / torch.mm 等高层 API 激活其厂商
                  Tensor Core 实现，屏蔽了反而等于禁用了唯一的高性能路径。

    用法:
        with shield_torch(hardware=task.target.hardware):
            exec(kernel_code, ns)   # kernel 内 import torch
            ns["kernel"](**inputs)  # 对 NVIDIA GPU: torch.matmul 抛错；白名单硬件: 正常放行
    """
    try:
        import torch
    except ImportError:
        yield
        return

    # ── 自研加速卡白名单: 不做任何 torch 屏蔽 ──
    if _hardware_allows_torch_compute(hardware or ""):
        yield
        return

    originals = []  # (module, attr_name, original_fn)
    modules = {"torch": torch}
    try:
        import torch.nn.functional as F
        modules["torch.nn.functional"] = F
    except ImportError:
        pass

    for mod_name, attr_names in _SHIELD_TARGETS.items():
        module = modules.get(mod_name)
        if module is None:
            continue
        for attr in attr_names:
            if hasattr(module, attr):
                originals.append((module, attr, getattr(module, attr)))
                setattr(module, attr, _make_blocked(f"{mod_name}.{attr}"))
    try:
        yield
    finally:
        for module, attr, original in originals:
            setattr(module, attr, original)
