"""§22.1 布局的**生产接线**（切片 C）：`run_root` 有调用点，候选真的落在 `candidate/` 那一格

四件事，每件都要有证人：

 ① `new_run_root` 是「一次运行一个新根」的唯一出处（`base_dir` 的归属仍是调用方）；
 ② **过门**的候选走唯一写入口落盘：内容 = 这一轮交给后端的那份、权限位 `0400`、
    路径与 `layout.path("candidate/code.py")` 是同一个；
 ③ **被 §21 拒掉**的候选一个字符都不留（内容级否证，与「零远端载荷」同一条纪律）；
 ④ `Conductor` 里 `write_for` 只有一个调用点，且真机驱动真的在传 `run_root`
    —— 「声明 + 无生产调用点」是本仓记过多次的形状（I82 的边界说的就是这一格）。

⚠️ 不许读成：`report` / `measurements` / `kernel_db` 那些既有产物也进了布局 —— 它们今天
    **没搬**，布局保护的是候选源码这一件（见状态文档 §22 行与 CHECKLOG I88）。
"""

import contextlib
import io
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent.conductor import Conductor  # noqa: E402
from operator_agent.kernel_db import KernelDatabase  # noqa: E402
from operator_agent.run_layout import (  # noqa: E402
    DIR_NAMES, LayoutError, PROTECTED_DIR_NAMES, new_run_root, run_layout,
)
from tests.unit.test_agent_runtime_wiring import (  # noqa: E402
    _CountingGPU, _ScriptedLlm, _inputs, _task,
)

ROOT = Path(__file__).resolve().parents[2]

#: 被 §21 拒掉的那份候选的**内容特征**（Fixer 加了一个外部依赖 ⇒ §21.4 第 4 条）
REFUSED_MARK = "import torch"


class TestTheRunRootConstructor(unittest.TestCase):
    """「一次运行一个新根」只有这一处，`base_dir` 的归属仍是调用方"""

    def test_two_runs_get_two_roots_under_the_base(self):
        base = tempfile.mkdtemp(prefix="runroot-")
        first, second = new_run_root(base), new_run_root(base)
        self.assertNotEqual(first, second, "两次运行拿到同一个根 ⇒ 上一轮的候选会被覆盖")
        for root in (first, second):
            self.assertTrue(root.startswith(base + os.sep), root)

    def test_a_bad_base_is_refused(self):
        for bad in ("", "   ", None, 0):
            with self.assertRaises(LayoutError):
                new_run_root(bad)

    def test_the_label_cannot_escape_the_base(self):
        base = tempfile.mkdtemp(prefix="runroot-")
        root = new_run_root(base, "../../evil/label")
        self.assertTrue(root.startswith(base + os.sep), root)
        self.assertNotIn("..", os.path.basename(root))


class TestAnAdmittedCandidateLandsInTheLayout(unittest.TestCase):

    def _run(self, run_root):
        backend = _CountingGPU()
        conductor = Conductor(backend=backend, kernel_db=KernelDatabase(jsonl_path=None),
                              llm=_ScriptedLlm(), max_optimize_rounds=2,
                              heartbeat_interval=1000, run_root=run_root,
                              use_coder_pipeline=False)
        with contextlib.redirect_stdout(io.StringIO()):
            artifact = conductor.execute(_task(), _inputs())
        return conductor, artifact, backend

    def test_the_file_is_the_code_that_reached_the_backend(self):
        root = tempfile.mkdtemp(prefix="runroot-live-")
        conductor, _, backend = self._run(root)
        path = os.path.join(root, "candidate", "code.py")
        self.assertTrue(os.path.exists(path), "过门的候选没有落盘")
        text = Path(path).read_text(encoding="utf-8")
        self.assertIn(text, backend.seen,
                      "盘上那份与交给后端的那份不是同一份 —— 判据指着的路径与产物脱钩了")
        # 权限位与路径都要与布局口径同源
        self.assertEqual(0o400, stat.S_IMODE(os.stat(path).st_mode))
        self.assertEqual(conductor.layout.path("candidate/code.py"), path)

    def test_a_refused_candidate_leaves_nothing(self):
        """被拒的候选**一个字符都不留**（它也从没到过后端）"""
        root = tempfile.mkdtemp(prefix="runroot-refused-")
        _, artifact, backend = self._run(root)
        rounds = [(h["round"], h["agent_action"]["accepted"])
                  for h in artifact.iteration_history if h.get("agent_action")]
        self.assertEqual([(1, True), (2, False)], rounds,
                         f"这条腿没有「过门 + 被拒」两轮: {rounds}")
        self.assertFalse(any(REFUSED_MARK in code for code in backend.seen),
                         "被拒的候选还是被送去后端了")
        survivors = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                body = Path(dirpath, name).read_text(encoding="utf-8", errors="replace")
                if REFUSED_MARK in body:
                    survivors.append(os.path.join(dirpath, name))
        self.assertEqual([], survivors, f"被拒的候选留下了痕迹: {survivors}")

    def test_round_history_uses_only_candidate_measurement_and_report_writers(self):
        """S1 retains post-gate history without opening unrelated layout roles."""
        root = tempfile.mkdtemp(prefix="runroot-slots-")
        self._run(root)
        for name in DIR_NAMES:
            entries = os.listdir(os.path.join(root, name))
            if name == "candidate":
                self.assertIn("code.py", entries)
                self.assertTrue(any(entry.startswith('round-') for entry in entries))
            elif name == "measurements":
                self.assertTrue(entries)
                self.assertTrue(all(entry.startswith(('round-', 'verification-event-')) for entry in entries))
                self.assertTrue(any(entry.startswith('verification-event-') for entry in entries))
            elif name == "report":
                self.assertTrue(entries)
                self.assertTrue(all(entry.startswith('round-') for entry in entries))
            else:
                self.assertEqual([], entries, f"{name}/ 里有东西 —— 谁写的？")

    def test_no_run_root_means_no_write(self):
        conductor = Conductor(backend=_CountingGPU(),
                              kernel_db=KernelDatabase(jsonl_path=None))
        self.assertIsNone(conductor.layout)
        self.assertIsNone(conductor._materialize_candidate("def kernel(A, B):\n    return A @ B\n",
                                                           _task()))
        self.assertEqual("candidate/code.py",
                         conductor.agent_runtime.authorized_roots[0] + "/code.py")


class TestWriteForHasAProductionCaller(unittest.TestCase):
    """结构性：唯一写入口**真的有生产调用者**，而且「真机驱动在传 run_root」不许悄悄回退"""

    def test_conductor_calls_write_for_exactly_once(self):
        src = (ROOT / "operator_agent" / "conductor.py").read_text(encoding="utf-8")
        self.assertEqual(1, src.count("write_for("),
                         "唯一写入口在控制器里出现了不止一次（或一次都没有）"
                         "—— §22.1 的主张是「只有一个写入口」")

    def test_the_real_machine_drivers_pass_a_run_root(self):
        drivers = ("harness_pass_rate.py", "agent_action_h20.py", "smoke_spine_h20.py")
        missing = [name for name in drivers
                   if "run_root=new_run_root(" not in
                   (ROOT / "benchmarks" / name).read_text(encoding="utf-8")]
        self.assertEqual([], missing,
                         f"这些真机驱动不再传 run_root: {missing} —— "
                         "回到「声明 + 无生产调用点」了")

    def test_the_protected_slots_stay_read_only(self):
        """落盘之后受保护格的权限位不许被写窗带走（写窗只该开在受保护格上，`finally` 复原）"""
        root = tempfile.mkdtemp(prefix="runroot-perm-")
        layout = run_layout(root)
        layout.write_for("candidate", "candidate/code.py", "def kernel(A, B):\n    return A @ B\n")
        for name in PROTECTED_DIR_NAMES:
            mode = stat.S_IMODE(os.lstat(os.path.join(root, name)).st_mode)
            self.assertEqual(0o500, mode, f"{name}/ 的权限位被留成了 {oct(mode)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
