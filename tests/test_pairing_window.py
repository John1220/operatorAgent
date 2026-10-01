"""§11-2 的「同一时间窗」那一半：观测、发布、拒评 (CHECKLOG I47 的窗口半格)

设计原话: 「一个候选的基线必须租到**同一张卡、同一时间窗**。A 卡基线 vs B 卡候选的
『加速比』直接作废」；以及「交错测量」—— candidate / baseline 交替 + 每轮 L2 flush。

设备那一半早有喂给者 (doc 63)；窗口这一半此前**没有**：oracle 的 `pairing_conflict`
窗口比较两侧都为 None ⇒ 只比设备。本文件钉四件事：

1. **生产者是真的**：`remote_scripts/run_kernel.py` 的 `_bench_pair_interleaved` 一次调用
   生成一扇窗, 两臂各自带回**自己那批样本所属的窗**, 两次调用拿到的窗不同 (不是常量)。
2. **三态**：`observed_windows` 缺观测返回 `None`/`unobserved`（不是空串 —— 空串会让
   oracle 那边变成一次恒真的比较）。
3. **判决只有一处**：观测到不同窗 ⇒ oracle 用 `pairing_conflict` 拒评, 稳定 token
   `pairing_conflict`, 且 provenance 里两个窗口都落盘可复核。
4. **心跳窗与测量窗不是一回事**: 传错那一组参数时心跳会被判成「别窗读数」—— 所以两条
   读数必须各自成组 (用例里的心跳窗口与测量窗口刻意取不同的值)。

运行: python3 -m unittest tests.test_pairing_window -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "remote_scripts"))

import torch  # noqa: F401  (run_kernel 的计时路径需要 torch)

from operator_agent.placement import (WINDOW_DIFFERENT, WINDOW_SAME,
                                      WINDOW_UNOBSERVED, observed_windows)
from tests.test_measurement_validity import _matmul_inputs, _task  # noqa: F401


class TestTheRunnerProducer(unittest.TestCase):
    """真跑一遍生产者的 CPU 路径 (`device_is_cuda=False`), 不 mock 计时"""

    def _pair(self):
        import run_kernel
        a = torch.arange(4096, dtype=torch.float32)
        b = torch.arange(4096, dtype=torch.float32)
        cand = lambda: (a * 2.0).sum()
        base = lambda: (b + 0.0).sum()
        return run_kernel, run_kernel._bench_pair_interleaved(
            cand, base, warmup=1, repeat=4, device_is_cuda=False)

    def test_both_arms_carry_the_window_they_were_measured_in(self):
        _, (cand_stats, base_stats, paired) = self._pair()
        wins = paired.get("arm_windows")
        self.assertIsInstance(wins, dict, "两臂的窗口观测必须落进 paired")
        self.assertEqual({"candidate", "baseline"}, set(wins))
        self.assertTrue(wins["candidate"], "窗口 id 不许是空串 (空串 = 恒真比较)")
        self.assertEqual(wins["candidate"], wins["baseline"],
                         "同一次交错调用的两臂必须在同一扇窗里")
        self.assertNotIn("arm_windows", cand_stats,
                         "不塞进逐臂 stats: 那是样本统计的容器, 窗口是配对属性")

    def test_two_calls_are_two_windows(self):
        """窗口 id 必须是**每扇窗**的, 不是模块常量 —— 否则这道比较永远为真"""
        run_kernel, first = self._pair()
        _, second = self._pair()
        self.assertNotEqual(first[2]["arm_windows"]["candidate"],
                            second[2]["arm_windows"]["candidate"])


class TestTheControlSideTriState(unittest.TestCase):

    def test_missing_observation_is_none_not_empty_string(self):
        for pair in ((None, "w1"), ("w1", None), (None, None)):
            c, b, scope = observed_windows(*pair)
            self.assertIsNone(c, pair)
            self.assertIsNone(b, pair)
            self.assertEqual(WINDOW_UNOBSERVED, scope, pair)

    def test_same_window_is_observed_not_assumed(self):
        self.assertEqual(("w1", "w1", WINDOW_SAME), observed_windows("w1", "w1"))

    def test_a_different_window_is_a_different_window(self):
        self.assertEqual(WINDOW_DIFFERENT, observed_windows("w1", "w2")[2])


class TestTheRefusalPath(unittest.TestCase):
    """观测到不同窗 ⇒ 拒评；同窗 ⇒ 不拒；没观测 ⇒ 跳过 (与设备那半同一边界)"""

    def _run(self, arm_windows, arm_identity=None, heartbeat_window="hb-window"):
        import contextlib
        import io

        from operator_agent.calibration import HeartbeatResult
        from operator_agent.verifier import Verifier
        from tests.backend_fakes import fake_backend
        from tests.test_measurement_validity import H20_HW_INFO

        bench = {"success": True, "device": "cuda:0", "gflops": 1000.0,
                 "memory_bandwidth_gbs": 500.0,
                 "candidate": {"median_time_ms": 1.0, "min_time_ms": 1.0},
                 "baselines": {"pytorch_eager": {"median_time_ms": 1.0,
                                                 "min_time_ms": 1.0}},
                 "baseline_selected": "pytorch_eager"}
        paired = {}
        if arm_windows is not None:
            paired["arm_windows"] = arm_windows
        if arm_identity is not None:
            paired["arm_identity"] = arm_identity
        if paired:
            bench["paired"] = paired
        backend = fake_backend(benchmark_with_baselines=lambda *a, **kw: bench,
                               get_hardware_info=lambda: dict(H20_HW_INFO))
        # 心跳窗与测量窗**刻意取不同值**: 两组参数混用时心跳会被判成别窗读数
        hb = HeartbeatResult(ok=True, epoch=1, reasons=(), known_good_ratio=1.0,
                             known_slow_ratio=1.0, window_id=heartbeat_window)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = Verifier(backend)._verify_performance(
                _PYTHON_MATMUL, _matmul_inputs(), 0.0, _task(),
                heartbeat=hb, heartbeat_expected=True)
        return result

    def test_a_cross_window_pairing_is_refused(self):
        r = self._run({"candidate": "w1", "baseline": "w2"})
        self.assertFalse(r.passed, "跨窗的加速比没有指称对象")
        self.assertEqual("cross_window_refused",
                         r.metrics["measurement_window_scope"])
        obs = r.metrics["observation"]
        self.assertIn("pairing_conflict", obs["reasons"],
                      "拒评的稳定 token 要能被代码读")
        self.assertIn("不在同一时间窗", obs["refused"] or "")
        self.assertEqual("w2", obs["provenance"]["baseline_window"],
                         "窗口要落盘: 拒评理由必须能复核")

    def test_a_same_window_pairing_passes_this_check(self):
        r = self._run({"candidate": "w1", "baseline": "w1"})
        self.assertEqual("same_window", r.metrics["measurement_window_scope"])
        self.assertNotIn("pairing_conflict", r.metrics["observation"]["reasons"])
        self.assertEqual({"candidate": "w1", "baseline": "w1"},
                         r.metrics["arm_windows"])

    def test_no_window_observation_skips_the_check(self):
        """老 payload / 非远端路径 ⇒ 不声称同窗, 也不拒评 (与设备那半同一边界)"""
        r = self._run(None)
        self.assertEqual("unobserved", r.metrics["measurement_window_scope"])
        self.assertEqual({}, r.metrics["arm_windows"])
        self.assertNotIn("pairing_conflict", r.metrics["observation"]["reasons"])

    def test_a_window_without_arm_identity_still_gets_judged(self):
        """报了窗口、没报逐臂身份 ⇒ 窗口那道比较**照样执行** (进入条件不许挂在设备上)"""
        r = self._run({"candidate": "w1", "baseline": "w9"})
        self.assertIn("pairing_conflict", r.metrics["observation"]["reasons"])
        self.assertIn("不在同一时间窗", r.metrics["observation"]["refused"] or "")

    def test_a_cross_device_pairing_still_reports_the_device(self):
        """设备那半没被窗口改动碰坏: 跨卡仍然以**卡**为理由拒评"""
        ident = {"candidate": {"visible_index": 0, "name": "NVIDIA H20"},
                 "baseline": {"visible_index": 1, "name": "NVIDIA H20"}}
        r = self._run({"candidate": "w1", "baseline": "w1"}, arm_identity=ident)
        self.assertEqual("cross_device_refused", r.metrics["measurement_scope"])
        self.assertIn("不在同一张卡", r.metrics["observation"]["refused"] or "")

    def test_the_heartbeat_window_is_a_different_thing(self):
        """测量窗与心跳窗取不同值: 心跳仍然算「本窗检查过」(两组参数不混用)"""
        r = self._run({"candidate": "m-window", "baseline": "m-window"},
                      heartbeat_window="hb-window")
        self.assertEqual("same_window", r.metrics["measurement_window_scope"])
        self.assertFalse(r.metrics["observation"]["ruler_broken"],
                         r.metrics["observation"].get("reasons"))


_PYTHON_MATMUL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"


class TestTheControlSideReuseIsVisible(unittest.TestCase):
    """非远程路径的缓存基线是**跨窗复用** (§11-2 的窗口前提在参照臂上不成立)

    今天不作判决 (参照臂不是设备租约臂, 复用是声明的优化), 但它必须**看得见**:
    产物里 `baseline_reused`、消息里一句明说 —— 否则「跨窗」永远只是一个沉默的捷径。
    """

    def test_a_reused_baseline_says_so_in_the_artifact_and_the_message(self):
        import contextlib
        import io

        from operator_agent import MockGPU
        from operator_agent.verifier import Verifier

        verifier = Verifier(MockGPU())
        task = _task()
        inputs = _matmul_inputs()
        # 两轮 L3: 第二轮必然命中基线缓存 (同一 (算子, 输入签名))
        outputs = []
        for _ in range(2):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r = verifier._verify_performance(_PYTHON_MATMUL, inputs, 0.0, task)
            outputs.append((r, buf.getvalue()))
        first, second = outputs
        self.assertFalse(first[0].metrics.get("baseline_reused"),
                         "第一轮的基线是本轮现量的")
        self.assertTrue(second[0].metrics.get("baseline_reused"),
                        "第二轮的基线来自本轮之前的窗口")
        self.assertIn("跨窗", second[1], "消息里要写出来, 不能只在产物里")

    def test_a_fresh_baseline_is_not_labelled_as_reused(self):
        from operator_agent import MockGPU
        from operator_agent.verifier import Verifier

        verifier = Verifier(MockGPU())
        baseline = verifier._measure_baseline(_task(), _matmul_inputs())
        self.assertFalse(baseline["baseline_reused"])


if __name__ == "__main__":
    unittest.main(verbosity=2)