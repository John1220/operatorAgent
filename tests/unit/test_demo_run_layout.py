"""演示脚本的守卫: `benchmarks/demos/demo_run_layout.py` 说得对, 而且今天仍然跑得起来

为什么给**演示**配守卫（本仓「每句 ✅ 都要有可运行的守卫」的同一条）: 演示里每一行
读数都是一句判断（「越权写被拒」「裸写抛 PermissionError」「判据拿真路径仍然命中」）。
判据改名、模块搬家、写者表被改 —— 这些都不会让脚本报错，只会让它印出一张读起来
仍然通顺、但已经不成立的图。那是 I15/I11 那一族（守卫在不成立的前提下恒绿）在
文档侧的形态: 屏幕上的图比注释更容易被当成证据。

这里只问三件事:
1. 演示四个场景全部跑完, 且它**自称的结论**逐条出现（越权五个格、裸写两层、同源两条）;
2. 屏幕上**没有** `!!`（脚本里每一处 `!!` 都是「本不该发生的事发生了」）;
3. 边界那四句仍在屏幕上 —— §22.1 的这一半**不许**被读成「候选被挡住了」。

运行: python3 -m unittest tests.test_demo_run_layout -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.demos import demo_run_layout as demo  # noqa: E402
from operator_agent.core import run_layout as rl  # noqa: E402


class TestTheRunLayoutDemo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demo.main()
        cls.out = buf.getvalue()

    def test_it_runs_and_never_prints_a_violation_marker(self):
        self.assertNotIn("!!", self.out,
                         "演示里出现了 `!!` —— 那几行正是「本不该发生的事发生了」")

    def test_every_protected_dir_is_shown_refusing_the_candidate(self):
        for name in rl.PROTECTED_DIR_NAMES:
            self.assertIn(f"candidate → {name}/ 拒", self.out, f"{name}/ 那行没了")
            self.assertIn(f"格内残留=[]", self.out)

    def test_both_layers_of_the_raw_write_are_shown(self):
        self.assertIn("PermissionError（目录写位那层）", self.out)
        self.assertIn("PermissionError（文件写位那层）", self.out)
        self.assertIn("0o400", self.out, "artifact 的权限位读数没了")

    def test_the_same_source_claims_are_on_screen(self):
        self.assertIn("is run_layout.PROTECTED_DIR_NAMES = True", self.out)
        self.assertIn("candidate_path(root) == layout.path('candidate/code.py') = True",
                      self.out)
        for name in rl.PROTECTED_DIR_NAMES:
            self.assertIn(f"真路径 {name}/x.py → violation", self.out)

    def test_the_design_seven_dirs_all_appear(self):
        for name in rl.DIR_NAMES:
            self.assertIn(name, self.out)

    def test_the_boundaries_are_on_screen(self):
        """§22.1 的 harness 侧一半**不许**被读成「候选被挡住了」/「有沙箱」"""
        for phrase in ("候选与 harness **同一个 uid**", "远端 runner 侧的布局与权限位",
                       "不是沙箱", "没有生产调用点"):
            self.assertIn(phrase, self.out, f"边界那句没了: {phrase}")


if __name__ == "__main__":
    unittest.main()