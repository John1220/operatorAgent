"""§8 失败路由的离线标定

五路各一条方向检查 + §19.3 的「编译过、仅数值错 → 修不重写」+ 「不算迭代/不污染学习」。
"""
import unittest

from types import SimpleNamespace

from operator_agent.decide.failure_route import (ROUTE_DEBUG_FIX, ROUTE_FIX, ROUTE_HOLD,
                                          ROUTE_RETRY, ROUTE_SEARCH, classify)


def _r(level, passed, msg="", log=""):
    return SimpleNamespace(level=SimpleNamespace(value=level),
                           passed=passed, message=msg, error_log=log)


class TestFiveRoutes(unittest.TestCase):
    def test_compile_fail_goes_to_fix(self):
        r = classify([_r("L1_COMPILE", False, log="SyntaxError: invalid syntax")])
        self.assertEqual(ROUTE_FIX, r.route)
        self.assertTrue(r.counts_as_iteration)
        self.assertTrue(r.pollutes_learning)
        self.assertFalse(r.fix_not_rewrite)

    def test_numeric_fail_goes_to_debug_fix_not_rewrite(self):
        """§19.3: 编译过、仅数值错 —— 修 > 重写 (外部实测省 92.8% token)"""
        r = classify([_r("L1_COMPILE", True),
                      _r("L2_FUNCTION", False, log="最大误差=1.2e0")])
        self.assertEqual(ROUTE_DEBUG_FIX, r.route)
        self.assertTrue(r.fix_not_rewrite, "数值失败被当成重写了")
        self.assertTrue(r.counts_as_iteration)

    def test_perf_fail_goes_to_search_not_fix(self):
        r = classify([_r("L1_COMPILE", True),
                      _r("L2_FUNCTION", True),
                      _r("L3_PERFORMANCE", False, log="加速比 0.8x")])
        self.assertEqual(ROUTE_SEARCH, r.route)
        self.assertFalse(r.fix_not_rewrite, "性能不达标去修代码 —— §8 明确说不是 bug")

    def test_infra_fail_retries_and_does_not_learn(self):
        r = classify([], backend_error="SSH timeout after 30s")
        self.assertEqual(ROUTE_RETRY, r.route)
        self.assertFalse(r.counts_as_iteration, "基础设施失败算了进度")
        self.assertFalse(r.pollutes_learning, "基础设施失败污染了学习")

    def test_infra_in_verification_text_also_routes_retry(self):
        r = classify([_r("L3_PERFORMANCE", False,
                        log="No usable remote baseline; ssh connection reset")])
        self.assertEqual(ROUTE_RETRY, r.route)
        self.assertFalse(r.pollutes_learning)

    def test_reference_uninformative_holds(self):
        # 用 kernel_db 分类器认得的类名, 不在本文件复述 marker 原文
        r = classify([_r("L2_FUNCTION", False,
                        log="reference_uninformative: 参考侧没有可比对象")])
        self.assertEqual(ROUTE_HOLD, r.route)
        self.assertFalse(r.pollutes_learning)
        self.assertFalse(r.counts_as_iteration)

    def test_lib_fallback_is_silent_degradation(self):
        r = classify([_r("L3_PERFORMANCE", False)], lib_fallback=True)
        self.assertEqual(ROUTE_HOLD, r.route)
        self.assertEqual("silent", r.kind)
        self.assertFalse(r.pollutes_learning)

    def test_all_passed_holds(self):
        r = classify([_r("L1_COMPILE", True), _r("L2_FUNCTION", True)])
        self.assertEqual(ROUTE_HOLD, r.route)


class TestPriorityOrdering(unittest.TestCase):
    def test_silent_beats_compile(self):
        """「什么都没判」盖过「判出了坏结果」—— 与 kernel_db 标记优先级同向"""
        r = classify([
            _r("L1_COMPILE", False, log="SyntaxError"),
            _r("L2_FUNCTION", False, log="reference_uninformative"),
        ])
        self.assertEqual(ROUTE_HOLD, r.route)

    def test_infra_beats_perf(self):
        r = classify([_r("L3_PERFORMANCE", False, log="ssh timeout")])
        self.assertEqual(ROUTE_RETRY, r.route)

    def test_compile_plus_numeric_still_fixes(self):
        r = classify([
            _r("L1_COMPILE", False, log="SyntaxError"),
            _r("L2_FUNCTION", False, log="最大误差=9"),
        ])
        self.assertEqual(ROUTE_FIX, r.route)
        self.assertFalse(r.fix_not_rewrite)


class TestRouteIsJsonSafe(unittest.TestCase):
    def test_as_dict_roundtrips(self):
        import json
        r = classify([_r("L2_FUNCTION", False, log="最大误差=1")])
        d = json.loads(json.dumps(r.as_dict()))
        self.assertTrue(d["fix_not_rewrite"])


if __name__ == "__main__":
    unittest.main()


class TestRemoteCompileErrorIsNotInfra(unittest.TestCase):
    """远端编译失败不能被判成基础设施。

    2026-11-04 真机：候选按配方写了 `@triton.jit(launch_pdl=True)`，Triton 3.7.1 的
    `jit()` 没有这个参数 ⇒ `TypeError: jit() got an unexpected keyword argument`。
    消息长这样：`Remote compile error via SSH: Compile error: TypeError: ...`，
    而 `_INFRA_MARKERS` 那时含裸词 `"ssh"` ⇒ 分类成 infra ⇒ 这一轮 `infra_retry`，
    「不计进度、退避重试」，白烧 1148 s / 20 次调用。
    """

    LOG = ("L1 [FAIL] 编译失败: Remote compile error via SSH: Compile error: "
           "TypeError: jit() got an unexpected keyword argument 'launch_pdl'")

    def test_compile_error_over_ssh_goes_to_fix(self):
        r = classify([_r("L1_COMPILE", False, log=self.LOG)])
        self.assertEqual(ROUTE_FIX, r.route, "远端编译错误被当成了基础设施")
        self.assertEqual("compile", r.kind)
        self.assertTrue(r.counts_as_iteration)
        self.assertTrue(r.pollutes_learning, "编译错误应当进负样本/教训, 不是基础设施噪音")

    def test_real_ssh_connect_failure_still_retries(self):
        r = classify([_r("L3_PERFORMANCE", False,
                         log="ssh: connect to host gpu-host.invalid port 2222: Connection timed out")])
        self.assertEqual(ROUTE_RETRY, r.route, "真正的连接失败必须仍然是基础设施")
