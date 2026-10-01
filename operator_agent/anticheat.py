"""反作弊模块
防止生成的 kernel 通过回退调用原生算子 (torch.matmul / F.softmax 等) 蒙混过关

两道防线:
1. static_audit: 静态审计 - 在代码执行前扫描禁用 API 调用模式 (按 DSL 分流)
2. shield_torch: 运行时屏蔽 - 执行期间将 torch 原生算子替换为抛错桩

规则按 DSL 区分:
- triton/cuda_c/opencl/sycl: 禁止 torch 计算 API 与 NumPy 计算回退
  (wrapper 只允许分配内存/启动 kernel，不允许在 CPU 上算出结果)
- python: NumPy 是实现介质所以允许，仅禁止 torch 计算 API

static_audit 走 AST 而不是正则 (docs/research/03 洞 2): 正则要求
`torch.` / `F.` / `np.` 前缀 + 函数名 + `(`，因此 `A @ B`、`A.matmul(B)`、
`from torch import matmul`、`getattr(torch,"mat"+"mul")`、`torch.nn.LayerNorm(...)`
五种自然写法全部漏过。AST 能解析 import 别名、做 getattr 常量折叠，并且天然
不会匹配 docstring 与注释里的散文 (正则版必须靠 _strip_non_kernel_regions
预先剔除 module docstring 才不误报)。
"""

import ast
import re
from contextlib import contextmanager
from typing import Optional


class CheatDetectedError(RuntimeError):
    """验证期间检测到禁用的原生算子调用"""
    pass


# ─── lib_fallback 生产者 (spine-review P1-4) ───────────────────
# WHY: eligibility 规则 4 / oracle / verifier 都读 `lib_fallback`, 但此前全仓
# **没有任何写入点** → 该护栏恒为 False, 报表上却显示「已检查」。恒 False 的护栏
# 比没有更危险 (spine-review §P1-4)。这里是唯一生产者: shield_torch 拦到调用时
# 记下被调符号, 上游用 `lib_fallback_metrics()` 搬进 metrics。

_last_lib_fallback: Optional[str] = None


def _record_lib_fallback(qualified_name: str) -> None:
    """记下一次运行时库回退命中 (被 shield 桩拦到的被调符号)。"""
    global _last_lib_fallback
    _last_lib_fallback = qualified_name


def take_lib_fallback() -> Optional[str]:
    """取走并清空最近一次运行时命中; None = 自上次取走以来未命中。"""
    global _last_lib_fallback
    hit = _last_lib_fallback
    _last_lib_fallback = None
    return hit


# 盾抛出来的那句话的**唯一**出处: 远端失败路径靠它把「真的发生了库回退」认回来。
# 为什么需要认文本 (2026-10-01 真机 §14-3): `take_lib_fallback()` 读的是**本地**全局
# 变量, 而盾跑在**远端进程**里 —— 把名字拼出来的假候选 (`getattr(torch, "mat"+"mul")`)
# 静态审计看不见, 只有远端那把盾能拦, 可它的证据只活在一句错误文本里, 于是
# `derive_eligibility(lib_fallback=...)` 规则 4 对唯一真正需要它的那条路径恒 False。
_RUNTIME_CHEAT_MARK = "反作弊: 验证执行期间禁止调用原生算子 {name}()"


def _runtime_cheat_message(qualified_name: str) -> str:
    """盾的那句错误 —— 生产者与解析器共用这一个格式, 不许各写一遍。"""
    return _RUNTIME_CHEAT_MARK.format(name=qualified_name)


def lib_fallback_from_error(error_text: Optional[str]) -> Optional[str]:
    """从**远端**回来的错误文本里认领一次运行时库回退, 返回被拦符号 (认不出 → None)。

    只认自家文案 (前缀与 `_make_blocked` 同一个常量), 不做模糊匹配: 否则任何含
    "matmul" 字样的报错都会被读成库回退, 而这一位会去驱动资格规则 4。
    """
    if not error_text:
        return None
    prefix = _RUNTIME_CHEAT_MARK.split("{name}", 1)[0]
    if prefix not in error_text:
        return None
    tail = error_text.split(prefix, 1)[1].strip()
    name = tail.split("(", 1)[0].strip()
    return name or "unknown"


def lib_fallback_metrics(violations: Optional[list] = None, *,
                         remote_error: Optional[str] = None) -> dict:
    """生产者 (spine-review P1-4): 把反作弊命中写成 metrics 片段。

    两个来源:
      1. `shield_torch` 运行时拦到的调用 (`take_lib_fallback`);
      2. `static_audit` 的违规列表 (调用方传入 `violations`)。

    返回 `{"lib_fallback": True, "lib_fallback_evidence": ...}` 或空 dict。
    上游把它 update 进 metrics 之后, `derive_eligibility(lib_fallback=...)`
    规则 4 才有触发的可能 —— 否则这一位永远是 False。
    """
    evidence = []
    hit = take_lib_fallback()
    if hit:
        evidence.append(f"runtime:{hit}")
    remote_hit = lib_fallback_from_error(remote_error)
    if remote_hit:
        evidence.append(f"runtime:{remote_hit}")
    for v in (violations or [])[:3]:
        evidence.append(f"static:{v}")
    if not evidence:
        return {}
    return {"lib_fallback": True, "lib_fallback_evidence": "; ".join(evidence)}


# ─── 禁用清单 (static_audit 与 shield_torch 共用同一份, 避免两处漂移) ───

_BLOCKED_TORCH_FUNCS = frozenset({
    "matmul", "mm", "bmm", "baddbmm", "addmm", "einsum",
    "softmax", "log_softmax",
    "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
})

_BLOCKED_FUNCTIONAL_FUNCS = frozenset({
    "softmax", "log_softmax", "linear",
    "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "layer_norm", "batch_norm", "gelu", "relu",
    "scaled_dot_product_attention",
})

_BLOCKED_NUMPY_FUNCS = frozenset({"matmul", "dot", "einsum", "vdot", "tensordot"})

# 张量方法形式 (A.matmul(B)): 接收者解析不出模块归属时按方法名判。
# 这些名字是 torch 独有的，任何 DSL 下出现都意味着回退。
_BLOCKED_METHODS = frozenset({
    "matmul", "mm", "bmm", "baddbmm", "addmm", "einsum",
    "softmax", "log_softmax", "scaled_dot_product_attention",
})

# 解析后落在这几个根下的表达式走「路径规则」而不是方法名规则 —— 否则
# `tl.dot(a, b)` 会被 `.dot` 误杀，而那正是 Triton kernel 里合法的张量核调用。
_KNOWN_MODULE_ROOTS = frozenset({"torch", "numpy", "triton"})


# ─── 正则规则 (仅作 AST 解析失败时的兜底，见 static_audit) ──────────

_TORCH_COMPUTE = (
    r"\btorch\s*\.\s*(?:" + "|".join(sorted(_BLOCKED_TORCH_FUNCS)) + r")\s*\("
)
_TORCH_FUNCTIONAL = r"\btorch\s*\.\s*nn\s*\.\s*functional\b"
_F_COMPUTE = (
    r"\bF\s*\.\s*(?:" + "|".join(sorted(_BLOCKED_FUNCTIONAL_FUNCS)) + r")\s*\("
)
_NUMPY_COMPUTE = (
    r"\b(?:np|numpy)\s*\.\s*(?:" + "|".join(sorted(_BLOCKED_NUMPY_FUNCS)) + r")\s*\("
)

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
    返回: 适合进行正则兜底审计的「核心代码片段」

    AST 路径不需要本函数: docstring 在 AST 里是单个 Constant 节点，不会被
    当成调用；__main__ 段则按行号裁掉 (见 _find_main_cutoff)。
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


# ─── AST 审计 ────────────────────────────────────────────────

def _find_main_cutoff(tree: ast.AST) -> Optional[int]:
    """返回 `if __name__ == "__main__":` 的行号; 该段是测试代码, 不参与审计"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                and t.left.id == "__name__" and len(t.ops) == 1
                and isinstance(t.ops[0], ast.Eq) and len(t.comparators) == 1
                and isinstance(t.comparators[0], ast.Constant)
                and t.comparators[0].value == "__main__"):
            return node.lineno
    return None


def _const_str(node) -> Optional[str]:
    """求值字符串常量表达式; 支持 "mat" + "mul" 这种拼接 (getattr 混淆)"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
        left, right = _const_str(node.left), _const_str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


class _AuditVisitor(ast.NodeVisitor):
    """按 DSL 收集禁用调用。行号直接取自 AST 节点，无需正则版的偏移换算。"""

    def __init__(self, is_python_dsl: bool, cutoff: Optional[int]):
        self.is_python_dsl = is_python_dsl
        self.cutoff = cutoff
        self.aliases: dict = {}      # 局部名 -> 解析后的点号模块路径
        self.violations: list = []

    # ── 作用域外的代码不审计 ──
    def _skipped(self, node) -> bool:
        return self.cutoff is not None and getattr(node, "lineno", 0) >= self.cutoff

    def visit_Import(self, node):
        if not self._skipped(node):
            for a in node.names:
                # `import torch.nn.functional as F` -> F = torch.nn.functional
                # `import torch`                  -> torch = torch
                self.aliases[a.asname or a.name.split(".")[0]] = a.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if not self._skipped(node):
            mod = node.module or ""
            for a in node.names:
                # `from torch import matmul`             -> matmul = torch.matmul
                # `from torch.nn import functional as Fn` -> Fn = torch.nn.functional
                self.aliases[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
        self.generic_visit(node)

    def _resolve(self, node) -> Optional[str]:
        """把表达式解析成点号路径; 解析不出 (如函数调用的返回值) 返回 None"""
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self._resolve(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def _fold_getattr(self, node) -> Optional[str]:
        """getattr(torch, "mat" + "mul") -> "torch.matmul"; 折叠不了返回 None"""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2):
            return None
        base, attr = self._resolve(node.args[0]), _const_str(node.args[1])
        return f"{base}.{attr}" if base and attr is not None else None

    def _report(self, node, what: str):
        self.violations.append(f"第 {node.lineno} 行: 检测到 {what}")

    def _path_violation(self, path: str) -> Optional[str]:
        """按解析后的点号路径判违规; 返回违规描述或 None"""
        root = path.split(".")[0]
        if root == "torch":
            if path == "torch.nn.functional" or path.startswith("torch.nn.functional."):
                fn = path.rsplit(".", 1)[-1]
                if fn in _BLOCKED_FUNCTIONAL_FUNCS:
                    return f"torch.nn.functional.{fn}() 调用"
                return None
            if path.startswith("torch.nn."):
                # 实例化 nn 模块 == 直接用 PyTorch 的实现，kernel 里没有正当理由
                return f"torch.nn 模块实例化 `{path.split('.', 2)[-1]}`"
            parts = path.split(".")
            if len(parts) == 2 and parts[1] in _BLOCKED_TORCH_FUNCS:
                return f"torch 原生算子 `{path}()`"
            return None
        if root == "numpy" and not self.is_python_dsl:
            fn = path.rsplit(".", 1)[-1]
            if fn in _BLOCKED_NUMPY_FUNCS:
                return f"NumPy 原生算子 `{path}()` (GPU DSL 下不得在 host 侧算出结果)"
        return None

    def visit_Call(self, node):
        if self._skipped(node):
            self.generic_visit(node)
            return

        func = node.func

        # getattr(torch, "mat" + "mul")(A, B) —— 常量折叠后按普通路径判
        folded = self._fold_getattr(func)
        if folded is not None:
            what = self._path_violation(folded)
            if what:
                self._report(node, f"{what} (经 getattr 常量折叠还原)")
            self.generic_visit(node)
            return

        if isinstance(func, ast.Attribute):
            base = self._resolve(func.value)
            if base and base.split(".")[0] in _KNOWN_MODULE_ROOTS:
                # 接收者是已知模块 (torch./numpy./triton.) → 按解析出的路径判。
                # 不能退化成按方法名判: `tl.dot(a, b)` 的 `.dot` 是 Triton 里
                # 合法的张量核调用，按名字会误杀。
                path = self._resolve(func)
                what = self._path_violation(path) if path else None
                if what:
                    self._report(node, what)
            elif func.attr in _BLOCKED_METHODS:
                # 接收者解析不出模块归属 (是个张量变量) → 按方法名判
                self._report(node, f"原生算子方法 `.{func.attr}()`")
        elif isinstance(func, ast.Name):
            # `from torch import matmul` 之后的裸名调用
            path = self._resolve(func)
            what = self._path_violation(path) if path else None
            if what:
                self._report(node, what)

        self.generic_visit(node)

    def visit_BinOp(self, node):
        # `A @ B` 在 python DSL 下是 NumPy 的合法矩阵乘 (NumPy 是实现介质)，
        # 只在 GPU DSL 下才算回退 —— 见 docs/research/03 的更正记录。
        if (isinstance(node.op, ast.MatMult) and not self.is_python_dsl
                and not self._skipped(node)):
            self._report(node, "`@` 矩阵乘运算符 (GPU DSL 下应使用 tl.dot 等原生指令)")
        self.generic_visit(node)


def _audit_by_ast(code: str, dsl: str) -> Optional[list]:
    """AST 审计; 代码无法解析时返回 None 让调用方退回正则兜底"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    visitor = _AuditVisitor(is_python_dsl=(dsl == "python"),
                            cutoff=_find_main_cutoff(tree))
    visitor.visit(tree)
    return visitor.violations


def _audit_by_regex(code: str, rules) -> list:
    """正则兜底: 仅在 AST 解析失败时使用 (fail-closed, 不因解析不出就放行)"""
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
    normalized_dsl = (dsl or "").lower()
    is_python_dsl = normalized_dsl == "python"

    # ── 硬件白名单：昇腾等自研加速卡放开 torch compute 限制 ──
    #   cuda_c DSL: C++ 源码中的 torch::matmul / at::matmul 等写在字符串常量里，
    #   AST 只会看到一个 Constant 节点，天然不报；且这些硬件的 Tensor Core
    #   只能通过 torch API 激活
    #   白名单下唯一保留的禁令是「GPU DSL 不得用 NumPy 在 host 侧算出结果」，
    #   而 python / cuda_c 两个 DSL 连这条都不适用 → 直接放行。
    whitelisted = _hardware_allows_torch_compute(hardware)
    numpy_only = whitelisted and normalized_dsl not in ("python", "cuda_c")
    if whitelisted and not numpy_only:
        return []

    violations = _audit_by_ast(code, normalized_dsl)
    if violations is not None:
        if not numpy_only:
            return violations
        return [v for v in violations if "NumPy" in v]

    # AST 解析失败 (代码有语法错) → 退回正则。宁可两套都跑也不要因为解析不出
    # 就放行: L1 的 compile() 预检通常会先拦下语法错，但 static_audit 是公开
    # 函数，不能假设调用顺序。
    if numpy_only:
        rules = [_NUMPY_RULES]
    else:
        rules = _PYTHON_DSL_RULES if is_python_dsl else _GPU_DSL_RULES
    return _audit_by_regex(code, rules)


# ─── 运行时屏蔽 ──────────────────────────────────────────────

# 执行期间需要替换为抛错桩的 torch 函数。
# 直接从 static_audit 的禁用清单派生 —— 两道防线共用一份定义，静态审计拦下的
# 运行时也一定拦得住，不会出现「审计放过但运行时炸」或反过来的错配。
_SHIELD_TARGETS = {
    "torch": sorted(_BLOCKED_TORCH_FUNCS),
    "torch.nn.functional": sorted(_BLOCKED_FUNCTIONAL_FUNCS),
}


def _make_blocked(qualified_name: str):
    def _blocked(*args, **kwargs):
        # 先记后抛: 异常可能被上游吞掉, 但 lib_fallback 生产者必须留下痕迹
        # (spine-review P1-4)。上游用 `lib_fallback_metrics()` 搬进 metrics。
        _record_lib_fallback(qualified_name)
        raise CheatDetectedError(_runtime_cheat_message(qualified_name))
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
