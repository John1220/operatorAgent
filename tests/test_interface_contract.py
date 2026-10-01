"""缺口 6: 核函数接口契约静态检查的行为测试

契约只有一条, 且从未写进代码: 远端固定 `module.kernel(**inputs)` 全关键字调用。
这条检查要在**动 GPU 之前**把签名不匹配拦下来 —— 所以除了「判对」，更要紧的是
「拦截时确实没碰后端」。
"""
import unittest

from operator_agent.backend_caps import Backend
from operator_agent.interface_contract import check_kernel_interface
from operator_agent.verifier import Verifier


def _msg(src, keys):
    # 判定对象现在带**适用性** (CHECKLOG I65): 本文件测的是「问题描述」那一半,
    # 所以统一取 `problem`。适配性那一半在 tests/test_guard_applicability.py。
    return check_kernel_interface(src, keys).problem


class TestContractChecker(unittest.TestCase):
    def test_historical_failure_case_is_caught(self):
        """真实历史: attention 连烧 3 轮在 `kernel() got an unexpected keyword 'causal'`"""
        src = "import torch\ndef kernel(Q, K, V):\n    return Q @ K.transpose(-1, -2)\n"
        msg = _msg(src, ["Q", "K", "V", "causal"])
        self.assertIsNotNone(msg, "多出来的 causal 必须被拦下")
        self.assertIn("causal", msg)
        self.assertIn("根因", msg, "处方式: 要说清为什么会炸")
        self.assertIn("改法", msg, "处方式: 要给可执行的下一步")

    def test_matching_signature_passes(self):
        """反向: 不能写成「什么都拦」—— 那比不拦更糟"""
        src = "import torch\ndef kernel(Q, K, V, causal=False):\n    return Q\n"
        self.assertIsNone(_msg(src, ["Q", "K", "V", "causal"]))

    def test_optional_param_the_caller_omits_is_not_a_problem(self):
        """调用方不传一个有默认值的参数完全合法 (这一条曾把能跑的核打回去重画)"""
        src = "def kernel(A, B, scale=1.0):\n    return A\n"
        self.assertIsNone(_msg(src, ["A", "B"]))

    def test_extra_input_is_caught_even_when_names_are_close(self):
        src = "def kernel(X, weight):\n    return X\n"
        msg = _msg(src, ["X", "weight", "bias"])
        self.assertIsNotNone(msg)
        self.assertIn("bias", msg)

    def test_required_param_the_caller_never_sends_is_caught(self):
        src = "def kernel(A, B, eps):\n    return A\n"
        msg = _msg(src, ["A", "B"])
        self.assertIsNotNone(msg)
        self.assertIn("eps", msg)

    def test_kwargs_escape_hatch_is_accepted(self):
        src = "def kernel(A, B, **kw):\n    return A\n"
        self.assertIsNone(_msg(src, ["A", "B", "causal", "anything"]))

    def test_positional_only_cannot_be_passed_by_keyword(self):
        src = "def kernel(A, B, /):\n    return A\n"
        msg = _msg(src, ["A", "B"])
        self.assertIsNotNone(msg, "positional-only 在全关键字调用下必然炸")
        self.assertIn("positional-only", msg)

    def test_last_top_level_def_wins_even_with_a_jit_decoy(self):
        """与 Python 语义一致: 顶层重复 `def kernel` 以后一个为准。

        顺带说明为什么走 AST: 真机实测 `inspect.signature(JITFunction)` 只会给出
        `(*args, **kwargs)` —— 用它做检查会永远放行 (见 docs/research/41)。
        """
        src = ("import triton\nimport triton.language as tl\n"
               "@triton.jit\ndef kernel(q_ptr, k_ptr, n, BLOCK: int):\n    pass\n"
               "def kernel(q, k):\n    return q\n")     # 顶层重复定义: 后一个生效
        msg = _msg(src, ["q", "k", "v"])
        self.assertIsNotNone(msg, "最后一个 def kernel 才是真正被调用的那个")
        self.assertIn("v", msg)

    def test_alias_assignment_is_not_guessed(self):
        """`kernel = something` 在后面把真身换掉了 —— 静态看不出参数表, 必须放行。

        ⚠️ 这条要连着那个「看起来会不匹配」的 `def kernel` 一起写: 只写别名而没有一个
        待判的 def, 检查会因为「找不到 def kernel」而返回 None, 于是别名规则本身没被
         exercised (第一版就是这么假绿的 —— 变异掉别名分支后测试仍然全绿)。
        """
        src = ("def kernel(A, B):\n    return A\n"
               "import functools\nkernel = functools.partial(kernel)\n")
        self.assertIsNone(_msg(src, ["A", "B", "extra"]),
                          "真身被别名换掉时不得凭 def 那行的签名判失败")

    def test_mismatch_in_a_later_input_set_is_still_caught(self):
        """多组输入套件要逐组判: 第 2 组多带一个键, 那一组运行时就会炸"""
        src = "def kernel(A, B):\n    return A\n"
        sets = [["A", "B"], ["A", "B", "scale"]]
        msgs = [check_kernel_interface(src, k).problem for k in sets]
        self.assertIsNone(msgs[0])
        self.assertIsNotNone(msgs[1], "只看第一组会把后面几组的不匹配整个漏掉")

    def test_conditional_definition_is_not_guessed(self):
        src = ("if True:\n    def kernel(A, B):\n        return A\n"
               "else:\n    def kernel(A):\n        return A\n")
        self.assertIsNone(_msg(src, ["A", "B"]))

    def test_missing_kernel_def_is_left_to_later_levels(self):
        src = "import torch\nhelper = 1\n"
        self.assertIsNone(_msg(src, ["A", "B"]))

    def test_no_input_info_means_nothing_to_check(self):
        src = "def kernel(A, B):\n    return A\n"
        self.assertIsNone(_msg(src, None))
        self.assertIsNone(_msg(src, []))

    def test_syntax_error_is_not_this_checkers_business(self):
        """语法错由 L1 的 compile() 预检负责; 这里不能抢话也不能崩"""
        self.assertIsNone(_msg("def kernel(A, B\n", ["A", "B"]))


class _RecordingBackend(Backend):
    """只在被调用时才记账 —— 「拦下了但后端也被碰了」等于没省下那一轮

    能力是**空集**: 它只是个记账器, 不是执行设备。空集意味着任何可选方法被调用时
    都会从基类存根抛出 `CapabilityMissing`, 而不是像从前那样安静地返回一个像样的值
    (旧版这里写了 `deploy() -> True`, 而它根本没部署任何东西)。
    """

    def __init__(self):
        self.compile_calls = 0
        self.execute_calls = 0

    @property
    def capabilities(self) -> frozenset:
        return frozenset()

    def __str__(self):
        return "RecordingBackend"

    def compile(self, code):
        self.compile_calls += 1
        return True

    def execute(self, *a, **kw):
        self.execute_calls += 1
        raise AssertionError("不该走到执行")

    def benchmark(self, *a, **kw):
        raise AssertionError("不该走到基准")

    def get_hardware_info(self):
        return {}


class TestL1RejectsBeforeTouchingBackend(unittest.TestCase):
    """判据不只是「返回失败」，而是「失败发生在动 GPU / 发远程之前」"""

    def setUp(self):
        self.backend = _RecordingBackend()
        self.v = Verifier(self.backend)

    def test_mismatch_never_reaches_the_remote_compile(self):
        code = "import torch\n\n\ndef kernel(A, B):\n    return A @ B\n"
        r = self.v._verify_compile(code, "triton", [["A", "B", "causal"]])
        self.assertFalse(r.passed)
        self.assertEqual(0, self.backend.compile_calls,
                         "L1 拦下了却仍然打了远程编译 —— 省的那一轮没省下")
        self.assertEqual(r.metrics.get("compile_status"), "interface_mismatch")
        self.assertIn("causal", r.error_log)

    def test_matching_code_does_reach_the_backend(self):
        """反向守卫: 若两条测试都失败, 说明检查无条件拦截"""
        code = "import torch\n\n\ndef kernel(A, B, causal=False):\n    return A @ B\n"
        r = self.v._verify_compile(code, "triton", [["A", "B", "causal"]])
        self.assertTrue(r.passed, r.message)
        self.assertEqual(1, self.backend.compile_calls)

    def test_only_later_input_set_mismatching_still_blocks(self):
        """逐组判这条要走完 Verifier: 变异掉循环 (只看第一组) 必须让本用例变红"""
        code = "import torch\n\n\ndef kernel(A, B):\n    return A @ B\n"
        r = self.v._verify_compile(code, "triton", [["A", "B"], ["A", "B", "scale"]])
        self.assertFalse(r.passed,
                         "第 2 组会传 scale 而 kernel() 不收 —— 只看第 1 组就放行了")
        self.assertEqual(0, self.backend.compile_calls)


if __name__ == "__main__":
    unittest.main()
