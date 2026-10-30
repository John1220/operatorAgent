"""日志目录的三条不变量 (跑测试曾经把真实运行现场轮掉)

机制: `Conductor.run()` 无条件进 `capture_execution()`, 它一边往 `<repo>/logs/`
写文件, 一边调用 `cleanup_old_logs(keep_count=10)`。所以每跑一次测试集, 仓库里
真实的运行日志就被 mock 现场挤掉一批。

对策是 `OPERATOR_AGENT_LOG_DIR` (指到别处 / `off` 关掉) 与 `OPERATOR_AGENT_LOG_KEEP`
(放宽保留数), `tests/conftest.py` 在测试会话里设前缀到 tmp。
这里钉住这三条别再悄悄退回去。
"""
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from operator_agent.agent_logger import capture_execution

REPO_LOGS = Path(__file__).resolve().parents[2] / "logs"


@contextmanager
def _in_dir(path):
    """临时把 cwd 换到 path —— capture_execution 里的相对路径是按 cwd 解析的。"""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class TestLogDirEnv(unittest.TestCase):
    def test_env_var_redirects_the_log_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"OPERATOR_AGENT_LOG_DIR": d}):
                with capture_execution(task_id="t1") as path:
                    print("hello")
                self.assertIsNotNone(path)
                self.assertEqual(Path(path).parent, Path(d),
                                 "env 变量没生效的话就会写回仓库的 logs/")
                self.assertIn("hello", Path(path).read_text(encoding="utf-8"))

    def test_off_writes_no_file_at_all(self):
        """值是字面量 off 时既不建目录也不落文件, 但输出必须照常透传。

        ⚠️ 第一版只对比了仓库 `logs/`, 于是漏掉了真正的失效模式: `off` 同时是一个
        **合法的相对路径**, 绕过哨兵分支后被解析成 `<cwd>/off/x.log`, 测试跑完在仓库根
        留下一整个 `off/` 目录而断言全绿 (2026-09-26 实际发生, 现场 `off/x.log` 里就是
        本用例那句 "still on stdout")。所以观察面必须换成「cwd 下多出来的任何东西」,
        而不是「我以为它会写的地方」。
        """
        with tempfile.TemporaryDirectory() as cwd:
            before = set(REPO_LOGS.glob("*.log")) if REPO_LOGS.exists() else set()
            with mock.patch.dict(os.environ, {"OPERATOR_AGENT_LOG_DIR": "off"}), \
                    _in_dir(cwd):
                with capture_execution(task_id="t2", log_name="x") as path:
                    print("still on stdout")
                self.assertIsNone(path)
                left_behind = sorted(os.listdir(cwd))
            after = set(REPO_LOGS.glob("*.log")) if REPO_LOGS.exists() else set()
            self.assertEqual([], left_behind,
                             f'值为 "off" 时不该在 cwd 留下 {left_behind}')
            self.assertEqual(before, after, '值为 "off" 时不该写仓库 logs/')

    def test_repo_logs_untouched_when_env_is_set(self):
        """回归守卫: 环境变量生效期间, 仓库 logs/ 一个文件都不许多"""
        before = set(REPO_LOGS.glob("*.log")) if REPO_LOGS.exists() else set()
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"OPERATOR_AGENT_LOG_DIR": d}):
                for i in range(3):
                    with capture_execution(task_id=f"tid{i}"):
                        print("x")
        after = set(REPO_LOGS.glob("*.log")) if REPO_LOGS.exists() else set()
        self.assertEqual(before, after,
                         f"仓库 logs/ 被写动了: 多了 {after - before}")


class TestKeepCount(unittest.TestCase):
    def _touch(self, d, n):
        for i in range(n):
            (d / f"log{i}.log").write_text("x", encoding="utf-8")
            # mtime 递增, 让「按最新保留」有确定顺序
            os.utime(d / f"log{i}.log", (1e9 + i, 1e9 + i))

    def test_keep_env_widens_the_retention_window(self):
        """默认 keep=10 是为「随手跑几次」设的; 长周期实验要能放宽, 否则要复查的现场被删。

        ⚠️ 用例必须造在「两条分支结果不同」的地方: 第一版只放了 5 个文件, keep=10 与
        keep=100 都不删任何东西, 于是把 KEEP 的读取整个变异掉测试照样绿 (逃逸过一次)。
        """
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            self._touch(dp, 14)          # 超过默认 10, "有没有读 env" 才看得出差别
            with mock.patch.dict(os.environ, {"OPERATOR_AGENT_LOG_DIR": d,
                                              "OPERATOR_AGENT_LOG_KEEP": "100"}):
                with capture_execution(task_id="t"):
                    pass
            self.assertEqual(15, len(list(dp.glob("*.log"))),
                             "放宽 keep 之后旧日志不该被删 (14 个原有 + 1 个新建)")

    def test_default_keep_still_prunes(self):
        """反向: 不设 env 时仍按默认 10 条清理 —— 别把这条机制改成「永不删」。

        ⚠️ 真实语义是「先清到 keep 条, 再写本次这一条」, 所以稳态是 keep+1 个文件。
        第一版断言写成 <=10 是错的, 而且是**先写测试再读代码**才会犯的那种。
        """
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            self._touch(dp, 14)
            with mock.patch.dict(os.environ, {"OPERATOR_AGENT_LOG_DIR": d}):
                os.environ.pop("OPERATOR_AGENT_LOG_KEEP", None)
                with capture_execution(task_id="t"):
                    pass
            self.assertEqual(11, len(list(dp.glob("*.log"))),
                             "清理发生在写入之前: 稳态应是 10 个旧的 + 本次 1 个")


if __name__ == "__main__":
    unittest.main()
