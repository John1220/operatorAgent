"""缺口 5 第 2 片的守卫: `mode="ssh"` 是承诺, 降级必须是数据 (doc 49 §五 第 2 片)

修的是三处:
  S3 `_init_backend` 的梯子 —— `mode="ssh"` 连不上时试 HTTP、再无条件落到 LocalExecutor,
     唯一信号是一行 print (不抛、不存标志、没有 strict 模式);
  S4 `LocalExecutor.check_connection()` 恒 True —— 与 S3 合起来, 梯子**永远以成功结束**;
  S6 `_lock_key()` 按**请求的** mode 取键 —— `mode="auto"` 降级到本地后键是 `auto:<host>`,
     而 `if self._mode == "local"` 永远不触发, 于是两个都跑在本机 CPU 上的后端拿到两把锁。
以及 A3: 三个 `test_nvidia_*` 驱动用显示名子串 `"LocalExecutor" in remote.name` 猜到底连上
没有, 猜错时**凭空造一台 SM 数是真的假 H20** (`MockGPU(name="MockH20", compute_units=78)`)
—— 名字与 SM 数都是真的, 只有「它是模拟器」这件事是假的, 于是产出的性能数字看起来完全像
H20 的。

⚠️ 全部离线: 探测被替换成必失败/必成功的假传输, 一个用例都不碰网络。
"""
import ast
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import operator_agent.execute.remote_executor as rx
from operator_agent.core.backend_caps import Backend, Cap, concrete_backends
from operator_agent.execute.remote_executor import (
    BackendUnavailable, LocalExecutor, RemoteExecutor, Resolution,
)
from tests.backend_fakes import fake_backend

REPO = Path(__file__).resolve().parents[2]
DRIVERS = ("test_nvidia_matmul.py", "test_nvidia_attention.py", "test_nvidia_layernorm.py")


class _Dead:
    """探测必失败的传输。只需要 `label` 与 `probe_connection` —— 梯子只用这两样"""

    def __init__(self, label, reason):
        self.label = label
        self.name = label
        self.host = label
        self._reason = reason

    def probe_connection(self):
        return False, self._reason


class _Live:
    """探测必成功的传输, 内层是一个声明了能力的假件"""

    def __init__(self, label, inner):
        self.label = label
        self.name = label
        self._inner = inner

    def probe_connection(self):
        return True, ""

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _dead_ssh(reason="Connection refused (fake)"):
    return lambda **kw: _Dead("SSHExecutor@fake(nvidia)", reason)


def _dead_http(reason="未配置 base_url"):
    return lambda **kw: _Dead("HTTPExecutor@fake", reason)


def _live_ssh(inner=None):
    # 声明 SAME_DEVICE_BASELINE 就**必须**给出 benchmark_with_baselines —— 假件在构造期
    # 双向检查这件事, 所以这里不能只声明不实现 (第一次写这条用例时就被它拦下了)。
    if inner is None:
        inner = fake_backend(
            caps={Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE},
            label="SSHExecutor@fake(nvidia)",
            benchmark_with_baselines=lambda *a, **kw: {"success": True})
    return lambda **kw: _Live("SSHExecutor@fake(nvidia)", inner)


# ─── S3: 承诺兑现不了就抛, 不降级 ────────────────────────────

class TestSshIsAPromiseNotAPreference(unittest.TestCase):

    def test_unreachable_ssh_raises_instead_of_degrading_to_cpu(self):
        with mock.patch.object(rx, "SSHExecutor", _dead_ssh()), \
                mock.patch.object(rx, "HTTPExecutor", _dead_http()):
            ex = RemoteExecutor(mode="ssh")
            with self.assertRaises(BackendUnavailable) as ctx:
                ex.backend
        self.assertNotIsInstance(getattr(ex, "_backend", None), LocalExecutor,
                                 "抛了却仍然把 LocalExecutor 塞进了 _backend")

    def test_the_exception_carries_every_probe_reason(self):
        """旧的 check_connection() 把异常吞成 False ⇒ 「为什么连不上」永远查不到"""
        with mock.patch.object(rx, "SSHExecutor", _dead_ssh("Authentication failed (fake)")), \
                mock.patch.object(rx, "HTTPExecutor", _dead_http()):
            ex = RemoteExecutor(mode="ssh")
            with self.assertRaises(BackendUnavailable) as ctx:
                ex.backend
        self.assertEqual([("SSHExecutor@fake(nvidia)", "Authentication failed (fake)")],
                         ctx.exception.resolution.attempts)
        self.assertIn("Authentication failed", str(ctx.exception),
                      "异常文本里必须能读出原因, 否则又回到只有一行 print")

    def test_ssh_mode_does_not_try_http(self):
        """承诺是「SSH」, 不是「随便给我一个能跑的」"""
        calls = []
        with mock.patch.object(rx, "SSHExecutor", _dead_ssh()), \
                mock.patch.object(rx, "HTTPExecutor",
                                  lambda **kw: calls.append(1) or _Dead("HTTPExecutor@fake", "")):
            with self.assertRaises(BackendUnavailable):
                RemoteExecutor(mode="ssh").backend
        self.assertEqual([], calls, "mode=ssh 却去探了 HTTP —— 那正是被拆掉的那级梯子")

    def test_a_reachable_ssh_still_resolves(self):
        """反向对照: 不能写成「什么都抛」—— 那比不抛更糟"""
        with mock.patch.object(rx, "SSHExecutor", _live_ssh()), \
                mock.patch.object(rx, "HTTPExecutor", _dead_http()):
            ex = RemoteExecutor(mode="ssh")
            self.assertIn(Cap.REAL_DEVICE, ex.capabilities)
            self.assertFalse(ex.resolution.degraded)
            self.assertEqual("SSHExecutor@fake(nvidia)", ex.resolution.resolved)

    def test_failure_is_not_cached_as_a_usable_object(self):
        """抛过之后再访问仍然抛 —— 失败状态不该被缓存成一个能用的后端"""
        with mock.patch.object(rx, "SSHExecutor", _dead_ssh()), \
                mock.patch.object(rx, "HTTPExecutor", _dead_http()):
            ex = RemoteExecutor(mode="ssh")
            for _ in range(2):
                with self.assertRaises(BackendUnavailable):
                    ex.backend


# ─── S3 的另一半: auto 仍可降级, 但降级是数据 ────────────────

class TestAutoDegradesVisibly(unittest.TestCase):

    def _degraded(self):
        patcher_ssh = mock.patch.object(rx, "SSHExecutor", _dead_ssh())
        patcher_http = mock.patch.object(rx, "HTTPExecutor", _dead_http())
        with patcher_ssh, patcher_http:
            ex = RemoteExecutor(mode="auto")
            ex.resolution            # 触发解析
        return ex

    def test_auto_falls_back_to_cpu(self):
        ex = self._degraded()
        self.assertIsInstance(ex.backend, LocalExecutor)

    def test_degradation_is_recorded_as_data(self):
        res = self._degraded().resolution
        self.assertTrue(res.degraded)
        self.assertEqual("auto", res.requested)
        self.assertEqual("LocalExecutor (CPU)", res.resolved)
        self.assertEqual(2, len(res.attempts), "两级探测都要留痕")

    def test_degradation_shows_up_in_capabilities(self):
        """这就是 A3 那个显示名子串检查的替代品"""
        ex = self._degraded()
        self.assertEqual(frozenset(), ex.capabilities)
        self.assertNotIn(Cap.REAL_DEVICE, ex.capabilities)
        with self.assertRaises(Exception) as ctx:
            ex.require(Cap.REAL_DEVICE)
        self.assertIn("LocalExecutor", str(ctx.exception))

    def test_the_facade_invents_no_capability_of_its_own(self):
        """**注入对照**: 「真解析到 LocalExecutor 却仍宣称 REAL_DEVICE」必须不可能

        门面的能力恒等于内层的能力 —— 一个都不加。这条断言与
        `test_backend_caps.py` 里「LocalExecutor 声明空集」那条连成一条链:
        降级 ⇒ 内层是 LocalExecutor ⇒ 能力空集 ⇒ 没有 REAL_DEVICE。
        链上任何一环被人改掉, 这两条里至少有一条会红。
        """
        inner = LocalExecutor()
        ex = self._degraded()
        self.assertEqual(inner.capabilities, ex.capabilities)
        ex2 = RemoteExecutor(mode="local")
        self.assertEqual(LocalExecutor().capabilities, ex2.capabilities)


# ─── S4: 梯子不再有一级「恒成功」 ────────────────────────────

class TestNoAlwaysSucceedingRung(unittest.TestCase):

    def test_local_executor_has_no_connection_probe(self):
        """`check_connection()` 恒 True 已被删掉

        对 CPU 回退问「能不能连上」没有意义 —— 它不是远端。留着它, 梯子的最后一级就
        永远成功, 于是「连不上」与「连上了一个 CPU 回退」在返回值上无法区分。
        删掉而不是改成返回 False: 降级到它是梯子的**终点**, 不是一次探测的结果。
        """
        for name in ("check_connection", "probe_connection"):
            self.assertFalse(hasattr(LocalExecutor(), name),
                             f"LocalExecutor 又有 {name} 了 —— 梯子会重新获得一级恒成功")

    def test_both_transports_probe_with_a_reason(self):
        for cls, kwargs in ((rx.SSHExecutor, dict(host="h", user="u")),
                            (rx.HTTPExecutor, dict(base_url=""))):
            inst = cls(**kwargs)
            self.assertTrue(hasattr(inst, "probe_connection"), f"{cls.__name__} 缺 probe_connection")
        # HTTP 的空 base_url 必须**离线**就给出原因, 不去连任何地址
        ok, reason = rx.HTTPExecutor(base_url="").probe_connection()
        self.assertFalse(ok)
        self.assertTrue(reason, "探测失败却没给原因")


class TestProbesReportWhyTheyFailed(unittest.TestCase):
    """**真的** `probe_connection` 必须把原因带出来 —— 不是假件带出来

    变异标定抓到的缺口 (M5 逃逸): 上面 `TestSshIsAPromiseNotAPreference` 里的 reason 断言
    喂的是假传输, 所以把真 `SSHExecutor.probe_connection` 的 `except` 改回「吞掉异常只返
    False」时**一条用例都不红** —— 守卫测的是我的假件, 不是产品代码。这一类用例必须驱动
    真实现, 只把最外层那次 I/O 换成必抛。
    """

    def test_ssh_probe_carries_the_exception_type_and_message(self):
        ex = rx.SSHExecutor(host="h", user="u")
        ex._ssh_exec = lambda *a, **kw: (_ for _ in ()).throw(
            OSError("Connection refused (fake)"))
        ok, reason = ex.probe_connection()
        self.assertFalse(ok)
        self.assertIn("OSError", reason,
                      "旧写法把异常吞成 False ⇒ 日志里只剩「SSH 不可用」, 分不清是 DNS、"
                      "认证还是超时 (同文件其余每条错误都带 type(e).__name__, 只有这条没有)")
        self.assertIn("Connection refused", reason)

    def test_http_probe_carries_the_exception_type_and_message(self):
        # 用一个非法 scheme, 不发起任何网络请求
        ok, reason = rx.HTTPExecutor(base_url="not-a-url").probe_connection()
        self.assertFalse(ok)
        self.assertTrue(reason)
        self.assertNotEqual("", reason)

    def test_http_probe_reports_a_non_200_health_check(self):
        """/health 回了但不是 200: 也要有原因, 不能只返 False"""
        class _Resp:
            status = 503

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        ex = rx.HTTPExecutor(base_url="http://fake.invalid")
        with mock.patch.object(rx.urllib.request, "urlopen", return_value=_Resp()):
            ok, reason = ex.probe_connection()
        self.assertFalse(ok)
        self.assertIn("503", reason)


# ─── S6: 锁按解析后的后端取键 ────────────────────────────────

class TestLockKeyFollowsTheResolvedBackend(unittest.TestCase):

    def test_a_degraded_auto_and_an_explicit_local_share_one_lock(self):
        """S6 的具体后果: 两个都跑在本机 CPU 上的后端以前拿到**两把**锁

        旧写法: `mode="auto"` 降级后键是 `auto:<host>`, `mode="local"` 的键是 `local`,
        而 `if self._mode == "local"` 对前者永远不触发 ⇒ 它们可以并发争同一份 CPU。
        """
        with mock.patch.object(rx, "SSHExecutor", _dead_ssh()), \
                mock.patch.object(rx, "HTTPExecutor", _dead_http()):
            degraded = RemoteExecutor(mode="auto")
            degraded.resolution
        explicit = RemoteExecutor(mode="local")
        self.assertEqual(explicit._lock_key(), degraded._lock_key())
        self.assertEqual("local", degraded._lock_key())

    def test_two_ssh_backends_to_the_same_host_share_one_lock(self):
        """同一张卡不因 platform 不同而分成两把锁 —— 锁保护的是卡, 不是配置"""
        a = rx.SSHExecutor(host="h", user="u", platform="nvidia")
        b = rx.SSHExecutor(host="h", user="u", platform="xpu")
        self.assertEqual(a.lock_identity(), b.lock_identity())
        self.assertNotEqual(a.label, b.label, "前提: 这两个后端的 label 确实不同")

    def test_different_hosts_get_different_locks(self):
        a = rx.SSHExecutor(host="h1", user="u")
        b = rx.SSHExecutor(host="h2", user="u")
        self.assertNotEqual(a.lock_identity(), b.lock_identity())


# ─── A3: 驱动不许再凭空造一台假 H20 ──────────────────────────

class TestDriversDoNotFabricateADevice(unittest.TestCase):
    """三个驱动各抄了一遍显示名子串检查 + 假 H20 兜底, 现在都必须删干净

    这条守卫读**源码**而不是跑驱动: 驱动要 LLM 端点与真机, 用例里跑不了。
    """

    def _sources(self):
        return {name: (REPO / name).read_text(encoding="utf-8") for name in DRIVERS}

    def test_no_mock_h20_anywhere(self):
        hits = [f"{name}:{i + 1}"
                for name, src in self._sources().items()
                for i, line in enumerate(src.splitlines())
                if "MockH20" in line and not line.lstrip().startswith("#")]
        self.assertEqual([], hits, f"驱动仍在造 SM 数是真的假 H20: {hits}")

    def test_no_display_name_substring_dispatch(self):
        hits = [f"{name}:{i + 1}"
                for name, src in self._sources().items()
                for i, line in enumerate(src.splitlines())
                if '"LocalExecutor" in' in line and not line.lstrip().startswith("#")]
        self.assertEqual([], hits, f"驱动仍在用显示名子串判断连没连上: {hits}")

    def test_every_driver_declares_what_it_needs(self):
        """反向对照: 删掉兜底之后必须**留下**一个会响的检查, 而不是什么都不查"""
        for name, src in self._sources().items():
            tree = ast.parse(src)
            calls = [n for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "require"]
            self.assertTrue(calls, f"{name} 没有声明它需要哪些能力")
            attrs = {a.attr for c in calls for a in ast.walk(c)
                     if isinstance(a, ast.Attribute)}
            self.assertIn("REAL_DEVICE", attrs, f"{name} 没要求 REAL_DEVICE")

    def test_mock_gpu_is_no_longer_imported_by_the_drivers(self):
        """留着不用的 import 会让下一个人以为兜底还在"""
        for name, src in self._sources().items():
            tree = ast.parse(src)
            imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                        for a in n.names}
            if "MockGPU" in imported:
                used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                self.assertIn("MockGPU", used, f"{name} import 了 MockGPU 却不再使用")


class TestActualDeviceIsReportable(unittest.TestCase):
    """#52: 「这一轮实际跑在哪张卡上」是执行侧的事实, 必须**问得到**

    量具的报表以前写 `os.environ["OPERATOR_AGENT_GPU"]` —— 那是用户**想**的卡,
    而 `SSHExecutor` 从不读那个变量。修法的第一步是把事实暴露成一个只读视图,
    报表改读它。基类必须有这个 property: 否则「有没有这个信息」又只能靠 `hasattr` 猜
    (doc 49 §4.1 第 1 条那条病)。
    """

    def test_the_property_exists_on_every_concrete_backend(self):
        for cls in concrete_backends():
            self.assertTrue(hasattr(cls, "selected_gpu"), f"{cls.__name__} 少了 selected_gpu")

    def test_reading_it_never_triggers_a_round_trip(self):
        """问它在用哪张卡不许变成「再花一次 ssh」"""
        ex = rx.SSHExecutor(host="probe.invalid")
        with mock.patch.object(ex, "_ssh_exec") as exec_spy:
            self.assertIsNone(ex.selected_gpu)
        exec_spy.assert_not_called()

    def test_a_resolved_pin_is_reported(self):
        ex = rx.SSHExecutor(host="probe.invalid")
        ex._gpu_pin = 7
        self.assertEqual(7, ex.selected_gpu)

    def test_the_two_ways_to_have_no_card_both_read_none(self):
        """-1 (解析过但不固定) 与 None (还没解析) 对读者的意思一样: 这一轮没有固定卡号

        区分它们靠 `capabilities` 里有没有 `GPU_PINNING`, 不靠往值里塞哨兵数 ——
        「没有卡」与「第 -1 号卡」不能共用一个数。
        """
        for pin in (None, -1):
            ex = rx.SSHExecutor(host="probe.invalid", platform="iluvatar")
            ex._gpu_pin = pin
            self.assertIsNone(ex.selected_gpu)

    def test_the_facade_forwards_the_inner_value(self):
        ex = rx.RemoteExecutor(mode="ssh")
        inner = mock.Mock()
        inner.selected_gpu = 5
        inner.capabilities = frozenset({Cap.GPU_PINNING})
        ex._backend = inner
        self.assertEqual(5, ex.selected_gpu)

    def test_the_facade_of_a_cpu_backend_reads_none(self):
        ex = rx.RemoteExecutor(mode="local")
        self.assertIsNone(ex.selected_gpu)
        self.assertNotIn(Cap.GPU_PINNING, ex.capabilities)


if __name__ == "__main__":
    unittest.main(verbosity=2)
