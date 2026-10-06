"""演示脚本的守卫: `benchmarks/demo_exec_guard.py` 说得对, 而且今天仍然跑得起来

为什么给**演示**配守卫（本仓「每句 ✅ 都要有可运行的守卫」的同一条）: 演示里每一行读数
都是一句判断（「超时被杀」「连孙子一起收了」「文件停在上限上」「设不上如实报」）。
上限默认值改了、执行后端换了、`status` 词表动了 —— 这些都不会让脚本报错，只会让它
印出一张读起来仍然通顺、但已经不成立的图。那是 I15/I11 那一族（守卫在不成立的前提下
恒绿）在文档侧的形态: 屏幕上的图比注释更容易被当成证据。

这里只问四件事:
1. 演示四个场景全部跑完, 且它**自称的读数**逐条出现（超时/文件大小/进程数/产物/CPU 五档,
   以及「设不上」那一条的原始原因）;
2. 屏幕上**没有** `!!`（脚本里每一处 `!!` 都是「本不该发生的事发生了」）;
3. 「本不该发生」的几句话**不**出现（「留了孤儿」「截断后当通过」「漏进了 harness」）;
4. 边界那几句仍在屏幕上 —— §22.2 这一片**不许**被读成「候选被隔离了」。

运行: python3 -m unittest tests.test_demo_exec_guard -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import demo_exec_guard as demo  # noqa: E402
from operator_agent import exec_guard as eg  # noqa: E402


class TestTheExecGuardDemo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demo.main()
        cls.out = buf.getvalue()

    def test_it_runs_and_never_prints_a_violation_marker(self):
        self.assertNotIn("!!", self.out,
                         "演示里出现了 `!!` —— 那几行正是「本不该发生的事发生了」")

    def test_every_status_it_claims_is_actually_on_screen(self):
        for needle in (
                "status=ok",                          # 正常候选
                f"status={eg.STATUS_TIMEOUT} kind={eg.KIND_WALL}",       # 墙钟
                f"kind={eg.KIND_FILE_SIZE}",          # 单文件大小
                f"kind={eg.KIND_PROCESSES}",          # 进程数
                f"kind={eg.KIND_ARTIFACT}",           # 产物上限
                f"kind={eg.KIND_CPU}",                # CPU 时间
                "SIGXCPU",
        ):
            self.assertIn(needle, self.out, f"演示少了这一档读数: {needle}")

    def test_the_two_things_that_must_not_be_read_as_success(self):
        self.assertIn("连孙子一起收了", self.out)
        self.assertIn("产物被读回来了吗? False", self.out)
        self.assertIn("失败（不是截断后当通过）", self.out)
        self.assertNotIn("留了孤儿", self.out)
        self.assertNotIn("截断后当通过了", self.out)
        self.assertNotIn("漏进了 harness", self.out)

    def test_the_not_enforced_item_is_shown_with_its_raw_reason(self):
        self.assertIn(f"{eg.NOT_ENFORCED}", self.out)
        self.assertIn("not_enforced  requested", self.out.replace("   ", " "))
        self.assertIn("内存那一档: state=not_enforced", self.out.replace("   ", " "))
        self.assertIn("ValueError: current limit exceeds maximum limit", self.out,
                      "「设不上」没有把原始原因带上屏幕")
        self.assertIn("fully_enforced=False", self.out)

    def test_the_boundaries_are_still_on_screen(self):
        for phrase in ("同一个 uid", "网络**没关**", "没有容器/cgroup",
                       "没有文件系统白名单", "没有 GPU watchdog", "设备租约回收",
                       "远端 runner 侧零覆盖", "CUDA 执行路径"):
            self.assertIn(phrase, self.out.replace("\n", " "),
                          f"边界那一句不见了: {phrase}")


if __name__ == "__main__":
    unittest.main(verbosity=2)