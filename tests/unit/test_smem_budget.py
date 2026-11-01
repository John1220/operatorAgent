# -*- coding: utf-8 -*-
"""Designer 的共享内存预算：实测值优先，读不到就拒。

这条分支的旧写法是 `else 128`——「没探测过硬件」在产物里长得和探测过一模一样，
而且对 Ada（100 KB/SM）偏大 28%：tile 预算按一个不存在的池子排，代价全烧在 L1 轮次上。
这正是 `docs/design/dsh-operator-portability-4090.md` §6 第 1、2 步要修的东西。

`_llm_generate_sketch` 需要 LLM 客户端才走得到，所以判定被抽成了 `smem_budget()`。
**抽出函数不等于接上了**：最后一组用例直接读源码，确认调用方真的在调它。
"""
import ast
import unittest
from pathlib import Path

from operator_agent.execute.remote_executor import HardwareProbeFailed
from operator_agent.generate.agents import smem_budget

#: 被测函数的源文件。被调用方是不是真的在调它，只能从写它的这份源码里读，
#: 不能靠我记得「上次是这样接的」。
_AGENTS_SOURCE = (Path(__file__).resolve().parents[2]
                  / "operator_agent/generate/agents.py").read_text(encoding="utf-8")

REPORTED_4070 = {"name": "NVIDIA GeForce RTX 4070", "compute_capability": "8.9",
                 "hardware_report": {"shared_memory_per_sm_kb": 100,
                                     "max_threads_per_sm": 1536,
                                     "l2_cache_size_bytes": 100663296,
                                     "probed_by": "torch.cuda.get_device_properties"}}
UNKNOWN_CARD = {"name": "NVIDIA Some New Card", "compute_capability": "10.0"}


def _calls_to(source, func_name, callee):
    """`func_name` 函数体里对 `callee(...)` 的直接调用行号（docstring 里的字不算）。"""
    out = []
    for fn in [n for n in ast.walk(ast.parse(source))
               if isinstance(n, ast.FunctionDef) and n.name == func_name]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == callee:
                out.append(node.lineno)
    return out


class TestProbeBeatsTable(unittest.TestCase):

    def test_reported_limit_is_used_on_an_unknown_cc(self):
        self.assertEqual((100, 1536, 32), smem_budget(REPORTED_4070))

    def test_reported_limit_beats_the_cc_constant(self):
        """228 是 H100 表里的数；探测器报了 227 就必须用 227。"""
        info = {"compute_capability": "9.0", "hardware_report":
                {"shared_memory_per_sm_kb": 227, "max_threads_per_sm": 2048}}
        self.assertEqual(227, smem_budget(info)[0])

    def test_threads_default_to_2048_when_only_smem_is_reported(self):
        info = {"compute_capability": "8.9", "hardware_report":
                {"shared_memory_per_sm_kb": 100}}
        self.assertEqual((100, 2048, 32), smem_budget(info))

    def test_zero_reported_value_is_not_treated_as_a_reading(self):
        """`0` 是「没读到」而不是「这台卡有 0 KB」——不许拿它去排 tile。"""
        info = {"compute_capability": "8.9", "hardware_report":
                {"shared_memory_per_sm_kb": 0}}
        with self.assertRaises(HardwareProbeFailed):
            smem_budget(info)

    def test_per_sm_limit_wins_when_both_are_reported(self):
        """两个上限同时报时取 per-SM：这是 Designer 历史上一直在用的那个口径。"""
        info = {"compute_capability": "8.9", "hardware_report":
                {"shared_memory_per_sm_kb": 100, "shared_memory_per_block_optin_bytes": 99 * 1024}}
        self.assertEqual(100, smem_budget(info)[0])

    def test_opt_in_bytes_alone_is_enough(self):
        """`shared_memory_per_multiprocessor` 不是每个 torch build 都暴露。

        只有 opt-in 时按它排预算：Hopper 是 227/228、Ada 是 99/100，**取到的是偏小的那个**，
        方向保守——偏小不会炸共享内存，偏大才让候选在编译期 OOR、白烧一轮。
        """
        info = {"compute_capability": "8.9", "hardware_report":
                {"shared_memory_per_block_optin_bytes": 99 * 1024, "warp_size": 32}}
        self.assertEqual(99, smem_budget(info)[0])

    def test_a_report_with_only_unrelated_fields_is_not_a_reading(self):
        info = {"compute_capability": "8.9", "hardware_report":
                {"l2_cache_size_bytes": 100663296, "probed_by": "torch"}}
        with self.assertRaises(HardwareProbeFailed):
            smem_budget(info)


class TestFallbackTableStillWorks(unittest.TestCase):
    """反向对照：修复不是「一律 raise」，否则老 runner 的卡全部跑不动。"""

    def test_known_cc_rows_are_unchanged(self):
        for cc, kb in (("9.0", 228), ("8.0", 164), ("9.0a", 228)):
            self.assertEqual(kb, smem_budget({"compute_capability": cc})[0], cc)

    def test_device_name_fills_in_when_cc_is_absent(self):
        self.assertEqual(228, smem_budget({"name": "NVIDIA H100 80GB HBM3"})[0])
        self.assertEqual(164, smem_budget({"name": "A100-SXM4-40GB"})[0])

    def test_legacy_string_field_is_parsed(self):
        info = {"compute_capability": "8.6", "compute_units": 128,
                "memory": {"local": {"size": "100KB/SM"}}}
        self.assertEqual((100, 256, 32), smem_budget(info))

    def test_legacy_unparseable_string_is_refused_not_guessed(self):
        """旧写法在这里 `else 128`：兜底病灶换了个分支继续存在。"""
        for bad in ("unknown", "N/A", ""):
            with self.assertRaises(HardwareProbeFailed, msg=bad):
                smem_budget({"compute_capability": "8.6",
                             "memory": {"local": {"size": bad}}})


class TestRefusalIsTheLastResort(unittest.TestCase):

    def test_unknown_card_without_a_probe_raises(self):
        with self.assertRaises(HardwareProbeFailed) as ctx:
            smem_budget(UNKNOWN_CARD)
        msg = str(ctx.exception)
        self.assertIn("128KB", msg, "拒的时候要明说拒的是那个兜底值")
        self.assertIn("10.0", msg, "要把收到的 cc 回显出来，否则没人知道缺的是哪张卡")

    def test_empty_hardware_info_raises(self):
        with self.assertRaises(HardwareProbeFailed):
            smem_budget({})


class TestRealBackendsAgainstTheSameRule(unittest.TestCase):
    """两个真后端的字典直接喂进同一条规则，期望值不手抄。"""

    def test_cpu_fallback_is_not_allowed_to_borrow_a_gpu_pool(self):
        """降级梯子的终点（`LocalExecutor`）报 `"N/A"`：它不是一台 GPU。

        旧写法在这里给出 128 KB，于是「CPU 回退」在 Designer 的 prompt 里长得像一台
        有 128 KB/SM 的卡。现在这条路径**在烧掉 LLM 调用之前**就停。
        """
        from operator_agent.execute.remote_executor import LocalExecutor
        with self.assertRaises(HardwareProbeFailed):
            smem_budget(LocalExecutor().get_hardware_info())

    def test_the_mock_backend_still_declares_a_readable_budget(self):
        """反向对照：拒绝编造不等于一律 raise。`MockGPU` 自己申报了 `local.size`。"""
        from operator_agent.execute.mock_gpu import MockGPU
        shared_mem_kb, threads, blocks = smem_budget(MockGPU().get_hardware_info())
        self.assertGreater(shared_mem_kb, 0, "模拟器申报的预算读不出来")
        self.assertGreater(threads, 0)
        self.assertEqual(32, blocks)


class TestRunnerReportsTheSameKey(unittest.TestCase):
    """`gap ①` 的写者侧：runner 报的键名必须就是 Designer 读的那个。

    这两头隔着一个进程（控制机 vs GPU 机上的 `run_kernel.py`），改一边不会让另一边变红。
    键名从 **runner 自己的源码**里解析出来核对，不抄进测试——抄的那份从抄下就开始漂。
    """

    def setUp(self):
        source = (Path(__file__).resolve().parents[2] / "remote_scripts/run_kernel.py"
                  ).read_text(encoding="utf-8")
        self.strings = set()
        seen = 0
        for fn in [n for n in ast.walk(ast.parse(source))
                   if isinstance(n, ast.FunctionDef) and n.name == "get_gpu_info"]:
            seen += 1
            for node in ast.walk(fn):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.strings.add(node.value)
        self.assertEqual(1, seen, "没解析出 get_gpu_info 的函数体，下面两条在空转")

    def test_every_key_the_designer_reads_is_reported_by_the_runner(self):
        for key in ("hardware_report", "shared_memory_per_sm_kb",
                    "shared_memory_per_block_optin_bytes", "max_threads_per_sm",
                    "l2_cache_size_bytes", "probed_by"):
            self.assertIn(key, self.strings,
                          "runner 不再上报 %s：Designer 会退回拒绝，迁移门读不到共享内存" % key)
        # 读不到时要走「明说没读到」那条分支，而不是省掉键或编一个数
        self.assertIn("hardware_report_missing", self.strings,
                      "runner 少了「探测失败」的显式出口，缺字段与没探测又长得一样了")

    def test_the_probe_keys_cannot_be_silently_renamed(self):
        """反向核对：torch 的属性名也留在源码里，改错属性名同样会被抓到。"""
        for attr in ("shared_memory_per_multiprocessor", "shared_memory_per_block_optin",
                     "L2_cache_size"):
            self.assertIn(attr, self.strings, "%s 不是 torch 的属性名了？" % attr)


class TestCallSiteIsWired(unittest.TestCase):
    """抽了函数却没人调 = 缺陷原样还在，而上面所有用例照样全绿。"""

    def test_llm_generate_sketch_calls_smem_budget(self):
        source = _AGENTS_SOURCE
        self.assertEqual(1, len(_calls_to(source, "_llm_generate_sketch", "smem_budget")),
                         "Designer 不再走 smem_budget()，上面那批用例守的是死代码")

    def test_the_parser_is_not_vacuous(self):
        """解析器看不见调用时上面那条会「绿」在空转上——先证明它看得见。"""
        self.assertGreaterEqual(
            len(_calls_to(_AGENTS_SOURCE, "smem_budget", "HardwareProbeFailed")), 1,
            "_calls_to 根本没在解析函数体，上面那条接线守卫在空转")
        self.assertEqual([], _calls_to(_AGENTS_SOURCE, "no_such_function", "smem_budget"))


if __name__ == "__main__":
    unittest.main()
