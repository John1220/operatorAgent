"""gpu_claim 协议 (CHECKLOG I68) 的单元测试 —— 纯本地, 不需要 GPU / SSH

每一组失效方式都对应 I68 实测里的一个具体形态:
  - 时刻采样看不见「还在 import 的对手」⇒ 声明生命期必须覆盖**整个进程**;
  - 崩溃没拆的陈旧声明不许永远占着卡 (pid 活性检查);
  - 写了一半的坏文件 / 别的协议的 .json 不许让 scan 抛 (观测面不许炸);
  - 写不进 (权限/只读) 不许挡测量 (尽力而为, 声明不是入场券);
  - 拆除必须真的发生在进程退出时 (atexit), 且不许挂到注入的假 pid 上。

运行: python3 -B -m unittest tests.test_gpu_claim -v
"""
import atexit
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "remote_scripts"))

import gpu_claim  # noqa: E402


class TestWriteAndScan(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_live_peer_on_the_same_gpu_is_seen(self):
        """核心判据: 对手活着、同卡 ⇒ scan 看得见 (对手 = 本测试进程, 天然活着)。"""
        path = gpu_claim.write_claim(2, pid=os.getpid(), directory=self.dir)
        self.assertTrue(path and os.path.exists(path), "写声明必须真的落盘")
        peers = gpu_claim.scan_claims(2, exclude_pid=999999, directory=self.dir)
        self.assertEqual(1, len(peers))
        self.assertEqual(os.getpid(), peers[0]["pid"])
        self.assertEqual(2, peers[0]["gpu"])

    def test_self_is_excluded(self):
        """exclude_pid 的意义: L3 扫描时本进程自己的声明不算「对手」。"""
        gpu_claim.write_claim(2, pid=4242, directory=self.dir)
        self.assertEqual([], gpu_claim.scan_claims(2, exclude_pid=4242,
                                                   directory=self.dir))

    def test_another_gpu_is_not_my_business(self):
        gpu_claim.write_claim(3, pid=os.getpid(), directory=self.dir)
        self.assertEqual([], gpu_claim.scan_claims(2, exclude_pid=1,
                                                   directory=self.dir))

    def test_a_dead_peers_stale_claim_is_ignored(self):
        """崩溃没拆的声明: pid 死了就不算占用 —— 否则一次崩溃永远污染后续测量。"""
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        self.assertIsNotNone(proc.returncode, "前提: 对手确实退出了")
        gpu_claim.write_claim(2, pid=proc.pid, directory=self.dir)
        self.assertEqual([], gpu_claim.scan_claims(2, exclude_pid=1,
                                                   directory=self.dir))

    def test_malformed_and_foreign_files_are_skipped(self):
        """scan 是观测面: 坏 JSON / 坏形状 / 非本协议文件一律跳过, 不许抛。"""
        with open(os.path.join(self.dir, "claim-1-abc.json"), "w") as fh:
            fh.write("{ not json")
        with open(os.path.join(self.dir, "note.json"), "w") as fh:
            json.dump({"pid": "not-an-int", "gpu": 2}, fh)
        with open(os.path.join(self.dir, "claim-3-ghi.json"), "w") as fh:
            json.dump({"gpu": 2}, fh)                 # 缺 pid
        with open(os.path.join(self.dir, "README"), "w") as fh:
            fh.write("not json at all")
        self.assertEqual([], gpu_claim.scan_claims(2, exclude_pid=1,
                                                   directory=self.dir))

    def test_the_written_file_is_owner_private(self):
        path = gpu_claim.write_claim(0, pid=os.getpid(), directory=self.dir)
        self.assertEqual(0o600, stat.S_IMODE(os.stat(path).st_mode),
                         "0600: 共享机上别的租户不许可写 (doc 52 落点纪律)")

    def test_an_unwritable_location_returns_none_not_an_error(self):
        """尽力而为: 目录造不出来 (这里是「路径被一个文件挡住」) ⇒ None, 不抛。"""
        blocker = os.path.join(self.dir, "afile")
        with open(blocker, "w") as fh:
            fh.write("")
        self.assertIsNone(gpu_claim.write_claim(
            0, directory=os.path.join(blocker, "sub")))

    def test_scanning_a_missing_directory_is_empty(self):
        self.assertEqual([], gpu_claim.scan_claims(
            0, exclude_pid=1, directory=os.path.join(self.dir, "nope")))

    def test_remove_claim_is_quiet_about_everything(self):
        path = gpu_claim.write_claim(0, pid=os.getpid(), directory=self.dir)
        gpu_claim.remove_claim(path)
        self.assertFalse(os.path.exists(path))
        gpu_claim.remove_claim(path)      # 再拆一次 (已不存在) 不炸
        gpu_claim.remove_claim(None)      # atexit 钩子里不许因 None 抛

    def test_an_injected_pid_never_registers_atexit(self):
        """单测注入的假 pid 不许挂到**真实进程**的退出钩子上 (会拆错别人的文件)。"""
        calls = []
        real_register = atexit.register

        def _spy(fn, *a):
            calls.append(a)
            return real_register(fn, *a)

        atexit.register = _spy
        try:
            path = gpu_claim.write_claim(2, pid=777, directory=self.dir)
        finally:
            atexit.register = real_register
        self.assertEqual([], calls, "注入 pid 的写入不许注册 atexit")
        self.assertTrue(path and os.path.exists(path), "但文件是真的写了")


class TestClaimDirResolution(unittest.TestCase):

    def test_env_overrides_the_rendezvous_dir(self):
        old = os.environ.get("OPERATOR_AGENT_GPU_CLAIM_DIR")
        try:
            os.environ["OPERATOR_AGENT_GPU_CLAIM_DIR"] = "/tmp/claims-x"
            self.assertEqual("/tmp/claims-x", gpu_claim.claim_dir())
        finally:
            if old is None:
                os.environ.pop("OPERATOR_AGENT_GPU_CLAIM_DIR", None)
            else:
                os.environ["OPERATOR_AGENT_GPU_CLAIM_DIR"] = old

    def test_default_is_a_fixed_home_path(self):
        """会合目录**必须**固定可发现 (协议要求); 不可预测性落在文件名上, 不在目录上。"""
        old = os.environ.pop("OPERATOR_AGENT_GPU_CLAIM_DIR", None)
        try:
            self.assertTrue(gpu_claim.claim_dir().endswith(".operator_agent_gpu_claims"))
        finally:
            if old is not None:
                os.environ["OPERATOR_AGENT_GPU_CLAIM_DIR"] = old


class TestAtexitCleanup(unittest.TestCase):

    def test_a_real_process_removes_its_own_claim_on_exit(self):
        """声明的拆除必须真的发生在进程退出时 —— 这条只能用真子进程看。"""
        with tempfile.TemporaryDirectory() as tmp:
            code = (
                "import sys, os\n"
                "sys.path.insert(0, sys.argv[1])\n"
                "import gpu_claim\n"
                "p = gpu_claim.write_claim(5)\n"
                "assert p and os.path.exists(p)\n"
                "print(p)\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", code, str(ROOT / "remote_scripts")],
                capture_output=True, text=True,
                env=dict(os.environ, OPERATOR_AGENT_GPU_CLAIM_DIR=tmp))
            self.assertEqual(0, proc.returncode, proc.stderr)
            path = proc.stdout.strip().splitlines()[-1]
            self.assertFalse(os.path.exists(path), "退出后声明必须被 atexit 拆掉")


if __name__ == "__main__":
    unittest.main(verbosity=2)
