"""后端硬件探测的诚实性守卫 (doc 49 §3.3 的 S1 / S2 / S7, 加本轮新查出的第 4、5 处)。

五个缺陷同型: **问不到的时候给一个看起来像样的答案**。

  S1  `SSHExecutor.get_gpu_info()` 的 `except` 分支返回一整套 A100 量级虚构字典
      (128 SM / 40GB / 2TB/s / 128KB/SM / 19TB/s)。同文件 happy path 的注释早就写着
      这些数字「已从 happy path 删掉 (H20 实为 4.0TB/s + 228KB/SM)」—— 修了一个分支,
      没修它的异常分支。唯一的读者是 Designer 的 tiling 约束 (`agents.py` 读
      `memory.local.size`), 所以后果是**朝保守方向收缩而不报错**: 编造的 128KB/SM 比
      真实的 228KB/SM 小, 核会一直不快, 没人知道为什么。
  S2  `HTTPExecutor.get_hardware_info()` **无条件**返回同一套数字 —— 比 S1 更彻底,
      因为它连探测都没试 (这个协议只有 /health 与三个 /api 端点, 没有硬件端点)。
  第 4 处  S1 修完之后才发现的: nvidia happy path 的 `memory.local.bandwidth` 写死
      "19TB/s", 而且**就在**「取不到就标 unknown, 不再编造数字」那行注释的下面两行。
      L1/共享内存带宽在两个真源里都没有字段 (runner 不报, HardwareCaps 只有 bank_count)。
  S7  `_setup_uv_env()` 非 nvidia 分支 `except` 里 print 一行「警告」然后
      `_uv_setup = True; return True` —— 谎话还会被**缓存住**, 第二次调用在第一行就返回。
  第 5 处  同函数 nvidia 分支的 `.python-version` 写失败只 print 就继续, 最后仍 `return True`;
      而这一段存在的全部理由就是「残留错误 pin 会让 uv add 全盘失败」。

每条断言都做过变异标定: 把修复改回旧写法, 对应用例必须变红 (结果记在 doc 49 §7.11)。
"""

import ast
import json
import unittest
from pathlib import Path
from unittest import mock

from operator_agent.backend_caps import Cap, CapabilityMissing
from operator_agent.remote_executor import (
    HardwareProbeFailed,
    HTTPExecutor,
    SSHExecutor,
)

_MODULE_SOURCE = Path("operator_agent/remote_executor.py").read_text(encoding="utf-8")

#: 旧写法在失败路径上编造的那几个具体值。任何一个重新出现在可执行语句里都算回归。
FABRICATED = ("40GB", "2TB/s", "128KB/SM", "19TB/s", "NVIDIA GPU (Remote")

_WORK_DIR = "/tmp/probe_work_dir"

#: runner 的 `gpu_info` 在 H20 上的真实形状 (字段名取自 remote_scripts/run_kernel.py:1366-1398)
H20_PAYLOAD = {
    "name": "NVIDIA H20",
    "compute_capability": "9.0",
    "total_memory_gb": 96.0,
    "multiprocessor_count": 78,
    "device": "cuda",
    "device_count": 8,
    "selected_gpu": 3,
    "torch_version": "2.7.0",
    "triton_version": "3.3.0",
    "gpu_memory": [],
}


def _ssh(platform="nvidia"):
    """构造一个不碰网络的 SSHExecutor (构造函数只做字符串拼装)"""
    ex = SSHExecutor(host="probe.invalid", platform=platform, work_dir=_WORK_DIR)
    ex._deployed = True          # 跳过 deploy() 的懒触发
    return ex


def _remote(fail_on=None, calls=None):
    """`_ssh_exec` 替身: 模拟一台**什么都没有的全新远端**。

    两个 `test -f` 探测一律抛 RuntimeError —— 这正是代码预期的「uv 没装 / 项目没初始化」,
    于是安装与 init 两级在每条用例里都真的被执行 (否则 `curl` 与 `uv init` 那两条变异
    根本跑不到, 守卫会假装覆盖它们)。
    """
    def fake(cmd, *a, **k):
        if calls is not None:
            calls.append(cmd)
        if "test -f" in cmd:
            raise RuntimeError(f"not found: {cmd[:40]}")
        if fail_on and fail_on in cmd:
            raise RuntimeError(f"模拟失败: {cmd[:40]}")
        if "which python3" in cmd:
            return "/usr/bin/python3\n"
        if "import sys;print" in cmd:
            return "3.10\n"
        return ""
    return fake


class TestSshProbeFailureRaises(unittest.TestCase):
    """S1: 探测失败必须**响**, 不许返回一整套规格"""

    def test_every_failure_source_raises(self):
        """四种失败来源各一条 (分支维度在异常类型: 旧写法对四者一律返回虚构字典)"""
        sources = (RuntimeError("ssh exited 255"),
                   json.JSONDecodeError("Expecting value", "<html>", 0),
                   ValueError("No JSON found in remote output"),
                   OSError("timed out"))
        for err in sources:
            ex = _ssh()
            with self.assertRaises(HardwareProbeFailed, msg=type(err).__name__):
                with mock.patch.object(ex, "_ssh_exec_json", side_effect=err):
                    ex.get_gpu_info()

    def test_deploy_failure_raises_too(self):
        """`deploy()` 也在 try 里 —— 它失败时旧写法同样返回虚构字典"""
        ex = _ssh()
        ex._deployed = False
        with self.assertRaises(HardwareProbeFailed):
            with mock.patch.object(ex, "deploy", side_effect=RuntimeError("scp refused")):
                ex.get_gpu_info()

    def test_no_fabricated_number_survives_a_failure(self):
        """已知会被看见的信号: 编造值不许出现在异常消息里"""
        ex = _ssh()
        with self.assertRaises(HardwareProbeFailed) as ctx:
            with mock.patch.object(ex, "_ssh_exec_json", side_effect=RuntimeError("boom")):
                ex.get_gpu_info()
        for token in FABRICATED:
            self.assertNotIn(token, str(ctx.exception), f"异常消息里带着 {token!r}")

    def test_message_keeps_the_underlying_cause(self):
        """postmortem 0004: 不许把下层原因换成自己最近的泛型类别"""
        ex = _ssh()
        with self.assertRaises(HardwareProbeFailed) as ctx:
            with mock.patch.object(ex, "_ssh_exec_json",
                                   side_effect=RuntimeError("Permission denied (publickey)")):
                ex.get_gpu_info()
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertIn("Permission denied (publickey)", str(ctx.exception))
        self.assertIn("RuntimeError", str(ctx.exception),
                      "要带类型名 —— 只有消息文本就没法按类名归因")

    def test_message_names_which_backend_failed(self):
        ex = _ssh()
        with self.assertRaises(HardwareProbeFailed) as ctx:
            with mock.patch.object(ex, "_ssh_exec_json", side_effect=RuntimeError("boom")):
                ex.get_gpu_info()
        self.assertIn(ex.label, str(ctx.exception))

    def test_a_failed_probe_leaves_no_cache(self):
        """抛出去之后不许留下半成品缓存: 下一次真探测成功时应该拿到真值"""
        ex = _ssh()
        with self.assertRaises(HardwareProbeFailed):
            with mock.patch.object(ex, "_ssh_exec_json", side_effect=RuntimeError("boom")):
                ex.get_gpu_info()
        self.assertIsNone(ex._gpu_info_cache)
        with mock.patch.object(ex, "_ssh_exec_json", return_value=dict(H20_PAYLOAD)):
            self.assertEqual(78, ex.get_gpu_info()["compute_units"])


class TestSshHappyPathsAreProbed(unittest.TestCase):
    """第 4 处 + 「派生值确实来自探测」的正面对照 (否则上面的守卫可以靠恒抛来绿)"""

    def _info(self, payload, platform="nvidia"):
        ex = _ssh(platform)
        with mock.patch.object(ex, "_ssh_exec_json", return_value=dict(payload)):
            return ex.get_gpu_info()

    def test_nvidia_local_bandwidth_is_unknown_not_19TB(self):
        self.assertEqual("unknown",
                         self._info(H20_PAYLOAD)["memory"]["local"]["bandwidth"])

    def test_nvidia_derives_from_the_probe_and_the_single_source_table(self):
        info = self._info(H20_PAYLOAD)
        self.assertEqual(78, info["compute_units"])
        self.assertEqual("228KB/SM", info["memory"]["local"]["size"],
                         "H20 的共享内存必须是探测/表里的 228, 不是编造的 128")
        self.assertEqual("4.0TB/s", info["memory"]["global"]["bandwidth"])
        self.assertEqual("96.0GB", info["memory"]["global"]["size"])
        self.assertEqual("NVIDIA H20", info["name"])

    def test_unrecognized_card_gets_unknowns_not_a_guess(self):
        """认不出的卡: 两个带宽与 local size 都必须是 unknown (推断路径给 0)"""
        info = self._info(dict(H20_PAYLOAD, name="NVIDIA Whatever-Next",
                               multiprocessor_count=132))
        self.assertEqual("unknown", info["memory"]["global"]["bandwidth"])
        self.assertEqual("unknown", info["memory"]["local"]["size"])
        self.assertEqual("unknown", info["memory"]["local"]["bandwidth"])
        self.assertEqual(132, info["compute_units"], "SM 数是真探测的, 不许一起 unknown")

    def test_non_nvidia_platform_reports_unknown_bandwidth(self):
        """这个分支是「接入下一个平台的模板」, 所以它也不许带具体型号的数"""
        payload = {"device_name": "iLUVATAR Something", "multiprocessor_count": 64,
                   "warp_size": 32, "total_memory_gb": 32.0, "device": "cuda",
                   "compute_capability": "unknown", "torch_version": "2.4.0"}
        info = self._info(payload, platform="iluvatar")
        self.assertEqual("unknown", info["memory"]["global"]["bandwidth"])
        self.assertEqual("unknown", info["memory"]["local"]["size"])
        self.assertEqual("unknown", info["memory"]["local"]["bandwidth"])
        self.assertEqual(64, info["compute_units"])
        blob = json.dumps(info, ensure_ascii=False)
        for token in FABRICATED:
            self.assertNotIn(token, blob)


class TestHttpHardwareInfoRefuses(unittest.TestCase):
    """S2: 这个协议问不到对面是谁, 所以只能抛"""

    def test_get_hardware_info_raises(self):
        ex = HTTPExecutor(base_url="http://probe.invalid:1")
        with self.assertRaises(HardwareProbeFailed) as ctx:
            ex.get_hardware_info()
        self.assertIn("gpu_info", str(ctx.exception), "要说清缺的是哪个端点")
        for token in FABRICATED:
            self.assertNotIn(token, str(ctx.exception))

    def test_message_lists_the_endpoints_that_do_exist(self):
        """处方式消息: 拿到这条的人要知道下一步能做什么"""
        ex = HTTPExecutor(base_url="http://probe.invalid:1")
        with self.assertRaises(HardwareProbeFailed) as ctx:
            ex.get_hardware_info()
        msg = str(ctx.exception)
        for endpoint in ("/health", "/api/compile", "/api/execute", "/api/benchmark"):
            self.assertIn(endpoint, msg)
        self.assertIn('mode="ssh"', msg, "要给出可执行的替代路径")

    def test_fixing_s2_does_not_buy_a_real_device_claim(self):
        """⚠️ 这条钉的是**决定**而不是代码: 能力要跟着凭据走。

        旧文档写「等 S2 修掉再声明 REAL_DEVICE」—— 那个条件现在满足了, 但满足了不等于该声明:
        这里唯一的凭据是「编译/执行都成功」, 它既可能来自 GPU 也可能来自 CPU。
        """
        ex = HTTPExecutor(base_url="http://probe.invalid:1")
        self.assertEqual(frozenset(), ex.capabilities)
        with self.assertRaises(CapabilityMissing):
            ex.require(Cap.REAL_DEVICE)


class TestEnvironmentSetupReportsFailures(unittest.TestCase):
    """S7 + 第 5 处: `_setup_uv_env` 的每一级失败都不许变成 True"""

    def test_non_nvidia_mkdir_failure_returns_false(self):
        ex = _ssh("iluvatar")
        with mock.patch.object(ex, "_ssh_exec",
                               side_effect=_remote(f"mkdir -p {_WORK_DIR}")):
            self.assertFalse(ex._setup_uv_env())

    def test_the_failure_is_not_cached_as_success(self):
        """旧写法 `_uv_setup = True` 会让谎话在第二次调用第一行就返回"""
        ex = _ssh("iluvatar")
        calls = []
        with mock.patch.object(ex, "_ssh_exec",
                               side_effect=_remote(f"mkdir -p {_WORK_DIR}", calls)):
            self.assertFalse(ex._setup_uv_env())
            self.assertFalse(ex._uv_setup)
            self.assertFalse(ex._setup_uv_env())
        self.assertGreaterEqual(len([c for c in calls if "mkdir" in c]), 2,
                                "失败被缓存住了, 第二次根本没再试")

    def test_non_nvidia_success_path_still_succeeds(self):
        """反向对照: 修复不许把这条路变成恒假"""
        ex = _ssh("iluvatar")
        calls = []
        with mock.patch.object(ex, "_ssh_exec", side_effect=_remote(None, calls)):
            self.assertTrue(ex._setup_uv_env())
        self.assertTrue(ex._uv_setup)
        calls.clear()
        self.assertTrue(ex._setup_uv_env())
        self.assertEqual([], calls, "已就绪时第二次调用应当直接短路, 不再跑命令")

    def test_python_version_pin_failure_returns_false(self):
        """第 5 处: 这一段的全部理由是防止残留错误 pin; 没做成就不能报成功"""
        ex = _ssh()
        calls = []
        with mock.patch.object(ex, "_ssh_exec",
                               side_effect=_remote(".python-version", calls)):
            self.assertFalse(ex._setup_uv_env())
        self.assertFalse(any("uv add" in c for c in calls), "pin 没写成还继续装依赖")
        self.assertFalse(ex._uv_setup)

    def test_every_nvidia_step_failure_returns_false(self):
        """五个失败点各一条 (用例数按分支算, 不按函数算)"""
        for step in (f"mkdir -p {_WORK_DIR}", "curl -LsSf", "uv init",
                     ".python-version", "uv add numpy"):
            ex = _ssh()
            with mock.patch.object(ex, "_ssh_exec", side_effect=_remote(step)):
                self.assertFalse(ex._setup_uv_env(), msg=step)
            self.assertFalse(ex._uv_setup, msg=step)

    def test_full_nvidia_success_path_returns_true(self):
        ex = _ssh()
        calls = []
        with mock.patch.object(ex, "_ssh_exec", side_effect=_remote(None, calls)):
            self.assertTrue(ex._setup_uv_env())
        self.assertTrue(ex._uv_setup)
        for needle in ("uv add numpy torch triton", ".python-version", "uv init"):
            self.assertTrue(any(needle in c for c in calls), f"{needle} 这一步根本没被执行")


# ── 源码级守卫 ────────────────────────────────────────────────
# 上面那些用例都通过 mock 驱动真实现; 这一组盯的是「别处再抄一份编造值」。

def _code_constants(source: str) -> set:
    """源码里**可执行语句**中的字符串常量, 剥掉所有 docstring。

    必须剥 docstring: 修复说明里逐字引用了 "40GB"/"19TB/s" 来解释为什么不许那么写
    (`HardwareProbeFailed` 的 docstring、S1/S2/第 4 处的注释), 不剥就把说明当成缺陷。
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            node.body = body[1:]
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _dict_values(source: str, class_name: str, method_name: str) -> list:
    """`class_name.method_name` 里所有字典字面量的 (键, 值节点) 对"""
    tree = ast.parse(source)
    out = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                and n.name == class_name]:
        for fn in [n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)
                   and n.name == method_name]:
            for node in ast.walk(fn):
                if isinstance(node, ast.Dict):
                    out.extend((k.value, v) for k, v in zip(node.keys, node.values)
                               if isinstance(k, ast.Constant))
    return out


class TestSourceCarriesNoFabricatedSpec(unittest.TestCase):
    """`remote_executor.py` 的可执行语句里不许再出现那五个编造值"""

    def test_guard_is_not_vacuous_on_synthetic_source(self):
        """解析型守卫必须先证明它真解析出了东西 (合成输入, 两个方向都要验)"""
        synthetic = (
            '"""docstring 里提到 "40GB" —— 不许算命中"""\n'
            "def f():\n"
            '    return {"size": "40GB"}\n'
        )
        found = _code_constants(synthetic)
        self.assertIn("40GB", found, "解析器空转: 真代码里的编造值它也认不出来")
        self.assertNotIn('docstring 里提到 "40GB" —— 不许算命中', found,
                         "docstring 没被剥掉: 它会喂出假阳性")

    def test_real_file_yields_a_populated_constant_set(self):
        found = _code_constants(_MODULE_SOURCE)
        self.assertGreater(len(found), 50,
                           f"只解析出 {len(found)} 个字符串常量 —— 解析区域大概找错了")

    def test_no_fabricated_literals_in_executable_statements(self):
        found = _code_constants(_MODULE_SOURCE)
        hits = sorted(t for t in found for f in FABRICATED if f in t)
        self.assertEqual([], hits, f"编造的硬件值重新出现在可执行语句里: {hits}")

    def test_hardware_sized_fields_are_never_literals(self):
        """`get_gpu_info` 返回字典里的 `compute_units` / `warp_size` 必须是表达式。

        挡的是「把某个具体型号的数直接写进返回字典」—— 旧写法四处都是这种字面量, 而它
        是这一族里最难从行为上区分的一种 (值看着完全合理)。范围限定在 `SSHExecutor.get_gpu_info`,
        因为 `LocalExecutor` 的 `compute_units: 1` 是**有意**标记「这不是加速卡」, 不是编造规格。
        """
        pairs = _dict_values(_MODULE_SOURCE, "SSHExecutor", "get_gpu_info")
        scanned = {k for k, _ in pairs}
        for must_see in ("compute_units", "warp_size", "memory", "name"):
            self.assertIn(must_see, scanned, f"解析器在 get_gpu_info 里没看到 {must_see} 键")
        offenders = [f"{k} @ line {v.lineno}" for k, v in pairs
                     if k in ("compute_units", "warp_size") and isinstance(v, ast.Constant)]
        self.assertEqual([], offenders, f"硬件尺寸字段被写成了字面量: {offenders}")


if __name__ == "__main__":
    unittest.main()
