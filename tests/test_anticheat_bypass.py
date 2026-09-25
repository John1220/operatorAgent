"""反作弊静态审计回归测试

对应 docs/research/03-反作弊与正确性防线实测.md 第二节的 8 个实测用例，
以及 docs/research/12-正确性验证与reward-hacking防护.md 的吸纳决策。

历史: 这 8 个用例最初有 5 个是漏过的，当时按「单个向量 × 单个 DSL」的粒度逐个
挂 `@unittest.expectedFailure` —— 粒度必须这么细，因为 expectedFailure 只在方法
内**全部** subTest 通过时才翻成 unexpected success，把 dict 塞进一个 subTest
循环会让部分修复被完全掩盖。static_audit 改用 AST 后 10 个标记全部翻成
unexpected success，于是摘掉标记，本文件从「缺陷清单」转为「回归防线」。

最后一项 (运行时屏蔽没进真机 runner) 修好后, 本文件已无 expectedFailure。

纯本地测试, 不需要 GPU / LLM / 远端。
"""

import ast
import unittest
from pathlib import Path, PurePosixPath

from operator_agent.anticheat import static_audit

# ─── 两个 DSL 下都必须拦下 ───────────────────────────────────
# python DSL 允许 NumPy (它是实现介质)，但下列写法语义明确指向 torch，
# 与 NumPy 无关，所以两个 DSL 下都算回退。
# 这些都不是刻意构造的对抗样本: `A @ B` 与 `A.matmul(B)` 是 PyTorch 里写矩阵乘
# 最自然的两种形态, LLM 在 Triton 路径遇挫后「退一步用 torch 兜底」时最可能写出。

CAUGHT_BOTH_DSL = {
    "torch.matmul 直呼": "def kernel(A,B):\n    import torch\n    return torch.matmul(A,B)\n",
    "torch.einsum": "def kernel(A,B):\n    import torch\n    return torch.einsum('ij,jk->ik',A,B)\n",
    "换别名 import functional": (
        "def kernel(X):\n    import torch.nn.functional as Fn\n    return Fn.softmax(X,-1)\n"
    ),
    "方法形式 .matmul()": "def kernel(A,B):\n    return A.matmul(B)\n",
    "from torch import matmul": (
        "def kernel(A,B):\n    from torch import matmul\n    return matmul(A,B)\n"
    ),
    "getattr + 字符串拼接": (
        'def kernel(A,B):\n    import torch\n    return getattr(torch,"mat"+"mul")(A,B)\n'
    ),
    "torch.nn 模块实例化": (
        "def kernel(X,w,b):\n    import torch\n"
        "    return torch.nn.LayerNorm(X.shape[-1]).to(X.device)(X)\n"
    ),
}

# ─── 仅 GPU DSL 下必须拦下 ───────────────────────────────────
# `A @ B` 是歧义写法: 对 numpy 数组它是合法矩阵乘，对 torch 张量它是回退。
# python DSL 下 NumPy 是实现介质 (anticheat.py 模块 docstring 明说)，本仓自己的
# tests/test_p0_3.py::PYTHON_MATMUL 与 tests/test_m1_hardening.py::PYTHON_MATMUL
# 正是这一形态，属正样本。详见 docs/research/03 的更正记录。

CAUGHT_GPU_DSL_ONLY = {
    "@ 运算符 (__matmul__)": "def kernel(A,B):\n    return A @ B\n",
}

# ─── 合法 kernel: 不得误报 ───────────────────────────────────
# 这两条是 AST 实现的主要误报风险点。`tl.dot` 的方法名 `.dot` 恰好也在 NumPy
# 禁用名单里，只能靠「接收者解析到 triton 模块就走路径规则、不退化成按方法名判」
# 来放过 —— 少了这一步，每一个正经的 Triton matmul kernel 都会被判作弊。

LEGIT_TRITON = """import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, BLOCK_M: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
    acc += tl.dot(a_ptr, b_ptr)
    tl.store(c_ptr, acc)

def kernel(A, B):
    M, K = A.shape
    C = torch.empty((M, M), device=A.device, dtype=torch.float32)
    return C
"""

LEGIT_PYTHON_NUMPY = """import numpy as np

def kernel(A, B):
    return A @ B
"""


class TestStaticAuditCatchesBothDSL(unittest.TestCase):
    """两个 DSL 下都必须拦下的写法，不得回归"""

    def test_triton(self):
        for name, code in CAUGHT_BOTH_DSL.items():
            with self.subTest(case=name):
                self.assertTrue(static_audit(code, "triton"),
                                f"{name} 本该被拦下, 现在漏过了 (回归)")

    def test_python(self):
        for name, code in CAUGHT_BOTH_DSL.items():
            with self.subTest(case=name):
                self.assertTrue(static_audit(code, "python"),
                                f"{name} 本该被拦下, 现在漏过了 (回归)")


class TestStaticAuditCatchesGpuDSL(unittest.TestCase):
    """歧义写法只在 GPU DSL 下断言"""

    def test_triton(self):
        for name, code in CAUGHT_GPU_DSL_ONLY.items():
            with self.subTest(case=name):
                self.assertTrue(static_audit(code, "triton"), f"{name} 漏过 (回归)")

    def test_cuda_c(self):
        for name, code in CAUGHT_GPU_DSL_ONLY.items():
            with self.subTest(case=name):
                self.assertTrue(static_audit(code, "cuda_c"), f"{name} 漏过 (回归)")


class TestStaticAuditNoFalsePositive(unittest.TestCase):
    """反向保护: 收紧规则时最容易打死的就是这几条"""

    def test_legit_triton_kernel_passes(self):
        """`tl.dot` 与 `torch.empty` 都是合法用法，不得被误拦"""
        self.assertEqual(static_audit(LEGIT_TRITON, "triton"), [])

    def test_python_dsl_allows_numpy_at_operator(self):
        """python DSL 下 numpy 的 `@` 是实现介质

        若将来为拦 torch 的 `A @ B` 而把 @ 运算符一律禁掉, 这个用例会失败 ——
        那是有意的。
        """
        self.assertEqual(static_audit(LEGIT_PYTHON_NUMPY, "python"), [])

    def test_python_dsl_allows_numpy_dot_method(self):
        """`.dot()` 是 numpy 数组的合法方法, python DSL 下不得拦"""
        code = "import numpy as np\n\ndef kernel(A,B):\n    return A.dot(B)\n"
        self.assertEqual(static_audit(code, "python"), [])

    def test_numpy_fallback_still_blocked_under_gpu_dsl(self):
        """同一份代码换到 GPU DSL 下就该拦 —— 证明按 DSL 分流真的在起作用"""
        code = "import numpy as np\n\ndef kernel(A,B):\n    return np.matmul(A,B)\n"
        self.assertTrue(static_audit(code, "triton"))

    def test_docstring_mention_not_flagged(self):
        """docstring 里提到 torch.matmul 不算调用

        正则版必须靠 _strip_non_kernel_regions 预删 module docstring 才不误报；
        AST 版里它只是一个 Constant 节点，天然不匹配。
        """
        code = ('"""本 kernel 用 tl.dot 实现，参考结果用 torch.matmul 验证。"""\n'
                "import triton.language as tl\n\n"
                "def kernel(A,B):\n    return tl.dot(A,B)\n")
        self.assertEqual(static_audit(code, "triton"), [])

    def test_main_block_not_flagged(self):
        """`if __name__ == "__main__":` 里的自测代码不算 kernel 实现"""
        code = ("def kernel(A,B):\n    return A*B\n\n"
                'if __name__ == "__main__":\n'
                "    import torch\n"
                "    ref = torch.matmul(A, B)\n")
        self.assertEqual(static_audit(code, "triton"), [])


class TestHardwareWhitelist(unittest.TestCase):
    """自研加速卡白名单: torch.* 是唯一能激活厂商 Tensor Core 的路径

    重构 static_audit 时这一段分支被改写过 (原先有两条等价的 `return []` 与一处
    被后一行覆盖的 rules 赋值)，这里钉住行为，避免改回去。
    """

    def test_ascend_python_dsl_allows_torch_matmul(self):
        code = "import torch\n\ndef kernel(A,B):\n    return torch.matmul(A,B)\n"
        self.assertEqual(static_audit(code, "python", hardware="Ascend910"), [])

    def test_ascend_triton_dsl_allows_torch_but_blocks_numpy(self):
        torch_code = "import torch\n\ndef kernel(A,B):\n    return torch.matmul(A,B)\n"
        self.assertEqual(static_audit(torch_code, "triton", hardware="ascend"), [])
        np_code = "import numpy as np\n\ndef kernel(A,B):\n    return np.matmul(A,B)\n"
        self.assertTrue(static_audit(np_code, "triton", hardware="ascend"),
                        "白名单只放开 torch, 用 NumPy 在 host 侧算出结果仍然是作弊")

    def test_nvidia_not_whitelisted(self):
        code = "import torch\n\ndef kernel(A,B):\n    return torch.matmul(A,B)\n"
        self.assertTrue(static_audit(code, "python", hardware="nvidia_gpu"))


class TestSyntaxErrorFailClosed(unittest.TestCase):
    """解析不出时代入正则兜底，不得因为 AST 抛错就放行"""

    def test_broken_code_still_audited_by_regex(self):
        # 括号未闭合 → ast.parse 抛 SyntaxError；正则仍应抓到 torch.matmul
        code = "def kernel(A,B:\n    import torch\n    return torch.matmul(A,B\n"
        self.assertTrue(static_audit(code, "triton"),
                        "AST 解析失败时必须退回正则, 不能直接返回空列表")


# ─── 真机 runner 的运行时屏蔽结构断言 ─────────────────────────
#
# runner 里的调用点分三类, 屏蔽只能覆盖第一类:
#   候选 (candidate)  用户交上来的 kernel, 必须被 shield 包住
#   参考 (reference)  _reference_torch 产出的 torch 原生实现, 绝不能被包住
#   基线 (baseline)   eager / fp16 / tf32 / torch.compile, 同样绝不能被包住
# 用「被调用者的名字」来分类, 因为 runner 里有两种形态: 直接调
# `module.kernel(**inputs)` / `kernel_func(**inputs)`, 以及把可调用对象交给
# `_bench_callable(fn, ...)` / `_timed_runs(fn, ...)` 去跑 —— 后者要靠首参才能
# 区分候选和基线。
#
# 首参有两种形态, 都得认:
#   `_bench_callable(candidate, ...)`     candidate 是 lambda, 但名字本身就有语义
#   `_timed_runs(call, ...)`              call = lambda: kernel_func(...) —— 名字
#                                         毫无语义, 必须顺着绑定解析到 lambda 体里
#                                         那个真正的被调用者
# 只认名字的话, 后一种形态会整个从检查里消失: 把 `kernel_func(**inputs)` 重构成
# `call = lambda: kernel_func(**inputs)` + `_timed_runs(call, ...)` 之后, 结构断言
# 会静默少覆盖两个函数。这个漏洞是真实发生过的 (见 test_shield_sites_are_non_trivial)。

_CANDIDATE_CALLEES = frozenset({"kernel", "kernel_func", "candidate"})
_REFERENCE_CALLEES = frozenset({"reference", "fp16_reference", "compiled"})
# 首参是「待跑的可调用对象」的那些调度函数
_BENCH_DISPATCHERS = frozenset({"_bench_callable", "_timed_runs", "_resolve_counts"})


def _has_shield_item(with_node) -> bool:
    return any(
        isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Name)
        and item.context_expr.func.id == "shield_torch"
        for item in with_node.items
    )


def _shield_regions(tree) -> list:
    """所有 `with shield_torch()` 覆盖的行区间 [(起, 止), ...]"""
    regions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and _has_shield_item(node):
            end = max((n.lineno for n in ast.walk(node) if hasattr(n, "lineno")),
                      default=node.lineno)
            regions.append((node.lineno, end))
    return regions


def _lambda_bindings(tree) -> dict:
    """`name = lambda: <callee>(...)` → {name: callee 标签}

    只收 lambda 体里**确实调用了**候选/参考的绑定。`fp16_reference = lambda:
    a @ b` 的体是 BinOp 不是 Call, 因此不进表 —— 它靠名字本身分类就够了。
    """
    bindings = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Lambda):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            inner = node.value.body
            if isinstance(inner, ast.Call):
                label = _direct_callee_label(inner)
                if label in _CANDIDATE_CALLEES or label in _REFERENCE_CALLEES:
                    bindings[target.id] = label
    return bindings


def _direct_callee_label(call: ast.Call):
    """只看这个 Call 自己调了谁, 不做首参解析 (供 _lambda_bindings 复用)"""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr                                  # module.kernel(...)
    if isinstance(func, ast.Name):
        return func.id                                    # kernel_func(...) / reference()
    return None


def _callee_label(call: ast.Call, bindings: dict):
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr                                  # module.kernel(...)
    if isinstance(func, ast.Name):
        if func.id in _BENCH_DISPATCHERS and call.args and isinstance(call.args[0], ast.Name):
            name = call.args[0].id
            # 顺着绑定解析到 lambda 体里真正的被调用者; 解析不到就用名字本身
            # (`candidate` / `reference` / `compiled` 这些名字自带语义)
            return bindings.get(name, name)
        return func.id                                    # reference() / compiled()
    return None


def _execution_lineno(node, parents):
    """lambda 的定义处不是执行处 —— 把 lambda 体内的调用折算到它真正被跑的行。

    - `candidate = lambda: module.kernel(...)`: lambda 绑到了名字上, 真正的执行点
      是后面的 `_bench_callable(candidate, ...)`, 那一条已被单独分类; 这里返回
      None 表示跳过, 否则会把它误判成「shield 外的候选调用」。
    - `_timed_runs(lambda: kernel_func(...), ...)`: lambda 直接当实参就地执行,
      折算成 `_timed_runs` 那一行。
    """
    cur = node
    while True:
        parent = parents.get(id(cur))
        if parent is None:
            return node.lineno
        if isinstance(parent, ast.Lambda):
            grand = parents.get(id(parent))
            if isinstance(grand, ast.Call) and any(a is parent for a in grand.args):
                return grand.lineno
            return None
        cur = parent


def _kernel_call_sites(tree) -> list:
    """返回 [(被调用者标签, 行号, 是否被 shield 覆盖), ...]"""
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    bindings = _lambda_bindings(tree)
    regions = _shield_regions(tree)
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        label = _callee_label(node, bindings)
        if label not in _CANDIDATE_CALLEES and label not in _REFERENCE_CALLEES:
            continue
        line = _execution_lineno(node, parents)
        if line is None:
            continue
        sites.append((label, line, any(lo <= line <= hi for lo, hi in regions)))
    return sites


class TestRuntimeShieldCoverage(unittest.TestCase):
    """洞 1 (docs/research/03): 真机 runner 曾经没有运行时反作弊

    shield_torch 原先只在 LocalExecutor 与 MockGPU 生效, 而唯一产出真机 L2/L3
    数字的是 remote_scripts/run_kernel.py 这条路径 —— 静态审计漏掉的写法在真机
    上就畅通无阻。

    这里是结构测试而不是行为测试: 真机路径没法在单测里跑。但只断言「源码里出现
    shield_torch」太弱, 一个 import 就能让它变绿。所以按 AST 断言三件事:
      1. 每个候选调用点都落在某个 `with shield_torch()` 区间内;
      2. 每个参考/基线调用点都落在区间**外** —— 参考实现本来就要合法调用
         torch.matmul, 套进去会把整个 L2/L3 打死;
      3. shield 不得出现在 for 循环体内 —— 进出各二十多次 setattr, 在 18.69µs
         的 launch 底噪上是 2~3 倍开销, 逐次包裹会毁掉 L3 计时。
    """

    @classmethod
    def setUpClass(cls):
        cls.runner = Path(__file__).resolve().parent.parent / "remote_scripts" / "run_kernel.py"
        cls.src = cls.runner.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.src)

    def test_shield_comes_from_anticheat_module(self):
        """必须是同一份 shield_torch, 不能在 runner 里另抄一套禁用清单"""
        imported = any(
            isinstance(node, ast.ImportFrom) and node.module == "anticheat"
            and any(a.name == "shield_torch" for a in node.names)
            for node in ast.walk(self.tree)
        )
        self.assertTrue(imported, "runner 未从 anticheat 导入 shield_torch")

    def test_candidate_calls_are_shielded(self):
        outside = [f"{label} @ 第 {line} 行"
                   for label, line, covered in _kernel_call_sites(self.tree)
                   if label in _CANDIDATE_CALLEES and not covered]
        self.assertEqual([], outside,
                         f"这些候选调用点不在任何 shield_torch 区间内: {outside}")

    def test_reference_calls_are_not_shielded(self):
        inside = [f"{label} @ 第 {line} 行"
                  for label, line, covered in _kernel_call_sites(self.tree)
                  if label in _REFERENCE_CALLEES and covered]
        self.assertEqual([], inside,
                         f"参考/基线被 shield 包住, 它们调用 torch 原生算子是合法的: {inside}")

    def test_shield_never_inside_a_loop(self):
        offenders = []
        for node in ast.walk(self.tree):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.With) and _has_shield_item(sub):
                    offenders.append(sub.lineno)
        self.assertEqual([], offenders,
                         f"shield_torch 出现在循环体内 (第 {offenders} 行), 会变成逐次屏蔽")

    def test_shield_sites_are_non_trivial(self):
        """兜底: 真出现新的候选执行路径时, 上面几条断言不该因为集合为空而空转"""
        sites = _kernel_call_sites(self.tree)
        self.assertGreaterEqual(len([s for s in sites if s[0] in _CANDIDATE_CALLEES]), 8)
        self.assertGreaterEqual(len([s for s in sites if s[0] in _REFERENCE_CALLEES]), 4)


class TestShieldIsActuallyDeployed(unittest.TestCase):
    """runner 里 import anticheat 之后, 部署链必须把 anticheat.py 一起推上去

    否则每次远端调用都 ImportError。_remote_files() 的顺序也有讲究: 依赖排在
    runner 前面, 中途失败只会得到「新依赖 + 旧 runner」(能跑), 不会得到
    「新 runner + 缺依赖」(直接崩)。
    """

    def test_anticheat_deployed_before_runner(self):
        from operator_agent.remote_executor import SSHExecutor

        ex = SSHExecutor(host="h", user="u")
        names = [PurePosixPath(remote).name for _, remote in ex._remote_files()]
        self.assertIn("anticheat.py", names)
        self.assertLess(names.index("anticheat.py"), names.index("run_kernel.py"))

    def test_deployed_anticheat_is_the_one_under_test(self):
        """推上去的必须是 operator_agent/anticheat.py 本身, 不是副本"""
        from operator_agent import anticheat

        deployed = Path(__file__).resolve().parent.parent / "operator_agent" / "anticheat.py"
        self.assertEqual(deployed.resolve(), Path(anticheat.__file__).resolve())


if __name__ == "__main__":
    unittest.main(verbosity=2)
