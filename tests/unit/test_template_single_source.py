"""模板单一真源的守卫: 同一份 kernel 源码只许有一个定义处。

起因是两个真实发生过的形态, 不是假想:

1) **逐字重复**。SOFTMAX_TRITON 曾同时存在于 kernel_db._TRITON_SOFTMAX_CODE (语料种子)
   与 rule_templates._SOFTMAX_TRITON (回退实现), 675 字符逐字相同, 两份都在跑。
   却没有一条断言让它们同源: 改一处, 另一处静默留在旧版。

2) **手抄, 而且逐字比较抓不到**。benchmarks/measurement/measure_heartbeat_real.py 里那份 triton
   参照核是抄来的, 但签名被重新折行 ⇒ 文本不相等。它的注释写着"取自
   rule_templates.py 的 matmul 模板"。模板一改, 这个**量具**就会静静地量另一个核,
   而注释继续撒谎 —— 这正是本仓反复记的那类缺陷: 判据的出处没有机械约束。

⚠️ 本守卫**不**要求把已刻意分叉的两份 matmul 合并: 语料种子 (带边界 mask / BLOCK=128)
与回退模板 (无 mask / BLOCK=64) 承担不同角色, 合并任一个都会改掉另一个的语义。
理由写在 operator_agent/kernel_templates.py 的模块 docstring 里, 并由本文件的
TestTheDeliberateDivergenceIsRecorded 钉住。

运行: python -m pytest tests/unit/test_template_single_source.py -q
"""
import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from operator_agent import kernel_templates as kt
from operator_agent.kernel_db import KernelDatabase
from operator_agent.rule_templates import RuleTemplateCoder
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                    TargetSpec, TaskComplexity, TaskSchema,
                                    TaskType)

CANONICAL = {
    "SOFTMAX_TRITON": kt.SOFTMAX_TRITON,
    "MATMUL_TRITON": kt.MATMUL_TRITON,
    "MATMUL_NUMPY": kt.MATMUL_NUMPY,
    "SOFTMAX_NUMPY": kt.SOFTMAX_NUMPY,
}
BENCH = REPO / "benchmarks" / "measurement" / "measure_heartbeat_real.py"


def _task(kernel="softmax", dsl="triton") -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel, complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl=dsl, compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.5, max_error=1e-5),
    )


def _string_constants(path) -> list:
    """文件里所有字符串字面量 (含嵌套) —— 用来找第二份副本"""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _copies_in(path) -> list:
    """返回该文件里逐字等于某个 canonical 模板的名字列表"""
    hits = []
    for s in _string_constants(path):
        for name, text in CANONICAL.items():
            if s == text:
                hits.append(name)
    return hits


def _jit_ast(source):
    """source 里 @triton.jit 那个函数的 AST (忽略格式); 取不到或解析失败返回 None"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        if not isinstance(n, ast.FunctionDef):
            continue
        if any((isinstance(d, ast.Attribute) and d.attr == "jit")
               or (isinstance(d, ast.Name) and d.id == "jit")
               for d in n.decorator_list):
            return ast.dump(n, annotate_fields=False)
    return None


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("_tpl_bench", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestOneDefinitionOfEachSharedTemplate(unittest.TestCase):

    def test_softmax_seed_and_fallback_are_the_same_object(self):
        db = KernelDatabase(jsonl_path="off")
        seeds = [r for r in db._records.values()
                 if r.algorithm == "softmax" and r.dsl == "triton"]
        self.assertEqual(1, len(seeds), "softmax/triton 种子应当只有一条")
        self.assertIs(seeds[0].code, kt.SOFTMAX_TRITON,
                      "语料种子不再引用单一真源 —— 又抄了一份")

    def test_rule_template_fallback_returns_the_single_source(self):
        code = RuleTemplateCoder().generate(_task("softmax", "triton"), None)
        self.assertIs(code, kt.SOFTMAX_TRITON,
                      "回退实现不再引用单一真源 —— 又抄了一份")

    def test_no_second_literal_copy_inside_the_package(self):
        offenders = []
        for p in sorted((REPO / "operator_agent").rglob("*.py")):
            if "__pycache__" in str(p) or p.name == "kernel_templates.py":
                continue
            for name in _copies_in(p):
                offenders.append("%s 里有一份逐字相同的 %s" % (p.name, name))
        self.assertEqual([], offenders, "同一份模板出现了第二个定义处: %s" % offenders)


class TestTheInstrumentDerivesItsKernel(unittest.TestCase):
    """量具的参照核必须是**派生**的, 不是手抄的 —— 见模块 docstring 第 2 条"""

    def test_benchmark_body_is_derived_from_the_single_source(self):
        mod = _load_benchmark()
        self.assertEqual(kt.jit_kernel_source(kt.MATMUL_TRITON), mod.TRITON_BODY,
                         "量具的参照核不再由单一真源派生")

    def test_the_kernel_the_instrument_measures_is_the_template_kernel(self):
        """真正要守的那条: 量具量到的核, 就是模板的核 (AST 级, 与格式无关)"""
        mod = _load_benchmark()
        self.assertIsNotNone(_jit_ast(kt.MATMUL_TRITON))
        self.assertEqual(_jit_ast(kt.MATMUL_TRITON), _jit_ast(mod.TRITON_GOOD_SRC))
        self.assertEqual(_jit_ast(kt.MATMUL_TRITON), _jit_ast(mod.TRITON_SLOW_SRC))

    def test_the_benchmark_holds_no_literal_copy_of_the_kernel(self):
        """重新手抄会被抓 —— 即使把签名折行 (逐字比较正是这样漏掉的)"""
        canon = _jit_ast(kt.MATMUL_TRITON)
        for s in _string_constants(BENCH):
            if s != kt.MATMUL_TRITON and _jit_ast(s) == canon:
                self.fail("benchmarks/measurement/measure_heartbeat_real.py 里又出现了一份手抄的核 "
                          "(逐字不同但 AST 相同 ⇒ 只有 AST 级守卫抓得到)")


class TestTheDeliberateDivergenceIsRecorded(unittest.TestCase):
    """两份 matmul 是刻意不同, 不是待清理的重复: 谁想合并, 先来解释这一条"""

    def test_the_two_matmul_implementations_are_deliberately_different(self):
        db = KernelDatabase(jsonl_path="off")
        seeds = [r for r in db._records.values()
                 if r.algorithm == "matmul" and r.dsl == "triton"]
        self.assertEqual(1, len(seeds))
        seed_code = seeds[0].code
        self.assertNotEqual(seed_code, kt.MATMUL_TRITON,
                            "两份 matmul 合并了 —— 请先读 kernel_templates.py 的模块 docstring")
        self.assertIn("mask=", seed_code, "语料正样本必须能处理不能整除的 shape")
        self.assertNotIn("mask=", kt.MATMUL_TRITON, "保底模板本来就不带 mask")


class TestTheGuardIsNotANoOp(unittest.TestCase):
    """反向标定: 每条守卫都要能变红, 否则它是永真的"""

    def test_the_scanner_catches_a_synthetic_second_copy(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "fake_copy.py"
            fake.write_text("TPL = " + repr(kt.SOFTMAX_TRITON) + chr(10), encoding="utf-8")
            self.assertEqual(["SOFTMAX_TRITON"], _copies_in(fake),
                             "扫描器没有抓到放在别处的第二份副本 ⇒ 它是永真的")

    def test_ast_comparison_sees_through_reformatting(self):
        """关键一条: 把签名折行 (正是原手抄的形态) 后逐字不等, 而 AST 仍相等"""
        canon = kt.MATMUL_TRITON
        sig_old = ("def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, "
                   "stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, "
                   "BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):")
        sig_new = ("def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K," + chr(10) +
                   "                  stride_am, stride_ak, stride_bk, stride_bn, "
                   "stride_cm, stride_cn," + chr(10) +
                   "                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, "
                   "BLOCK_K: tl.constexpr):")
        self.assertIn(sig_old, canon, "模板签名变了 —— 这条反向标定要跟着更新")
        reformatted = canon.replace(sig_old, sig_new)
        self.assertNotEqual(reformatted, canon, "折行后应当逐字不等")
        self.assertEqual(_jit_ast(canon), _jit_ast(reformatted),
                         "AST 级比较必须看穿格式差异 —— 否则它挡不住手抄")

    def test_jit_kernel_source_refuses_input_without_a_jit_function(self):
        with self.assertRaises(ValueError):
            kt.jit_kernel_source("x = 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
