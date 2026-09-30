"""共享机上的可预测路径 + 「拆除要等到静默」的吸纳 (#69)。

一手出处：DeepSeek Harness 本地 vendored 树（`~/repos/github/deepseek-harness`，
HEAD `47f943859b`，2026-08-13）`docs/defensive-patterns.md`：

* `:27` **"Never hand untrusted output the ambient environment or predictable paths"** —
  *"Temp/spill files use a private (0700) dir, random names, and exclusive owner-only
  opens (`'wx'`, `0o600`) — predictable world-readable paths invite symlink races and
  disclosure."*
* `:19` **"Dispose must reach quiescence, not just request it"** —
  *"A teardown that issues kills/aborts but returns before the work stops leaves orphans."*

对这两条本仓的失效方式**不是** DSH 说的 disclosure（我们在意的文件是自己写的探针脚本），
而是**测量完整性**：那台 GPU 机是多用户的，而原来两个驱动把一个固定名的脚本 scp 上去再
**执行它** —— 路径公开 ⇒ 我们跑的那份未必是我们写的那份；而 `terminate()` 不等 ⇒ 远端那条
自带 +300s 自我期限的干扰流可能继续占着卡 ⇒ 下一次「安静读数」其实是被我们自己污染的。
这条与 doc 33/34 花两轮消除的「邻居抢卡」是同一个变量。

扫描范围是 `benchmarks/` + `operator_agent/` + `remote_scripts/`，**不含 `tests/`**：
测试里那些 `/tmp/...` 是喂给假 SSH 的**惰性字符串**（断言的是命令文本，从不真在共享机上
创建），把它们算进命中会让守卫的名单里混进不需要修的东西 —— 这里明写这个边界，而不是
悄悄加白名单。

运行: python3 -B -m unittest tests.test_shared_host_scratch_paths -v

变异标定 16 条（首轮 15 条里有 1 条**逃逸**，那条改成了下面的 AST 判据）：
去 `O_EXCL`→`test_write_refuses_an_existing_path`+`_does_not_follow_a_symlink` /
`0o600`→`0o644`→`test_written_file_is_0600` / 拿掉 `write_private` 里的落点门、
以及把 `ensure_private` 的 raise 短路→`test_permission_guard_fires_on_a_live_path`
（这条用例的存在本身是因为「写在 `private_dir` 里的守卫在死分支上，删掉无人变红」）/
只 `terminate` 不 `wait`→`test_terminate_then_wait_and_read_back`+
`test_kill_when_terminate_is_not_enough` / 不做 `pgrep` 读回→上面两条+
`test_no_residue_claim_without_a_read_back` / 不发 `pkill`、去掉 `shlex.quote`→
`test_a_path_needing_escaping_is_quoted` / 远端名退回固定名或 pid→
`test_remote_names_are_not_predictable` / 本地目录换成 `/tmp/opag-fixed`→5 条 /
驱动退回裸 `terminate()`、chmod 那行只留注释、绕开 `private_dir` 用 `mkdtemp`→
`test_the_drivers_use_the_helpers_rather_than_being_deleted` / 驱动里新写一处
`/tmp/_util_probe.py`→ratchet 与驱动门**两条同时响**。
"""

import ast
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from _scratch import (ensure_private, private_dir, remote_scratch_path,  # noqa: E402
                      stat_mode, stop_interferer, write_private)

SCAN_DIRS = ("benchmarks", "operator_agent", "remote_scripts")

#: 仍然留在共享路径上的文件, 以及**具体是哪几个**。两个都**不是本仓今天能修的**:
#: `remote_executor.py` 与 `run_kernel_xpu.py` 正被另一个会话编辑 (不碰别人的在途文件),
#: 而那批路径全是昆仑芯/xBLAS 的 JIT 缓存目录与一把全局锁 —— 用户明确说过「昆仑芯屏蔽即可」。
#: 钉的是**去重后的路径集合**而不是数量: 数量对「换掉一个、少掉一个」恒盲 (量具先标定
#: 那条纪律的第 22 条 —— 钉集合要用差集等式, 子集/计数式断言对集合变动是瞎的)。
RATCHET = {
    "operator_agent/remote_executor.py": {
        "/tmp/xpujitc-temp", "/tmp/kXPU3", "/tmp/.xblas_cache", "/tmp/.xblas_jit.lock",
    },
    "remote_scripts/run_kernel_xpu.py": {
        "/tmp/kXPU3", "/tmp/xpujitc-temp", "/tmp/.xblas_cache", "/tmp/.xblas_jit.lock",
    },
}


def _shared_path_hits():
    """AST 扫字符串常量里的 `/tmp/<固定名>`（注释不在 AST 里 ⇒ 说明文字不算命中）。"""
    import re
    pat = re.compile(r"/tmp/[A-Za-z0-9._\-]{3,}")
    out = {}
    for d in SCAN_DIRS:
        for p in sorted((ROOT / d).glob("*.py")):
            try:
                tree = ast.parse(io.open(p, encoding="utf-8", errors="replace").read())
            except SyntaxError as exc:            # 读不动就说读不动, 别当成「没有命中」
                raise AssertionError(f"守卫解析失败: {p}: {exc}") from exc
            found = [m.group(0) for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)
                     for m in pat.finditer(n.value)]
            if found:
                out[str(p.relative_to(ROOT))] = found
    return out


class _FakeProc:
    """只记录被怎么对待的假 Popen。"""

    def __init__(self, terminate_never_exits=False):
        self.calls = []
        self._terminate_never_exits = terminate_never_exits
        self._waited_once = False

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")

    def wait(self, timeout=None):
        self.calls.append(f"wait(timeout={timeout})")
        if self._terminate_never_exits and not self._waited_once:
            self._waited_once = True
            raise subprocess.TimeoutExpired(["ssh", "host", "cmd"], timeout or 0)
        return -9 if "kill" in self.calls else 0


class _FakeRemote:
    def __init__(self, pgrep_output=""):
        self.commands = []
        self._pgrep = pgrep_output

    def __call__(self, cmd):
        self.commands.append(cmd)
        if cmd.startswith("pgrep"):
            return self._pgrep
        return ""


class TestPrivateScratchSemantics(unittest.TestCase):
    def test_private_dir_is_owner_only(self):
        d = private_dir("guard-test-")
        try:
            self.assertEqual(0o700, stat_mode(d) & 0o777,
                             "目录带 group/other 位 ⇒ 同机其他用户能读我们写进去的东西")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_written_file_is_0600(self):
        d = private_dir("guard-test-")
        try:
            f = write_private(d, "probe.py", "print(1)\n")
            self.assertEqual(0o600, stat_mode(f) & 0o777)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_write_refuses_an_existing_path(self):
        """exclusive open 的意义: 已存在时报错, 而不是覆盖掉别人的东西。"""
        d = private_dir("guard-test-")
        try:
            write_private(d, "a.py", "first")
            with self.assertRaises(FileExistsError):
                write_private(d, "a.py", "second")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_write_does_not_follow_a_symlink(self):
        """DSH 那条点名的失效形态: 世界可写目录里的固定名会被做成符号链接。

        这里用**真的 symlink** 验 `write_private` 不被带跑 —— 如果它退化成
        `Path.write_text`, 目标文件就会被我们覆盖, 而这条会红。
        """
        d = private_dir("guard-test-")
        try:
            victim = write_private(d, "victim.txt", "不是我们的文件")
            link = d / "probe.py"
            os.symlink(str(victim), str(link))
            with self.assertRaises(FileExistsError):
                write_private(d, "probe.py", "payload")
            self.assertEqual("不是我们的文件", victim.read_text(encoding="utf-8"),
                             "符号链接的目标被覆盖了 ⇒ exclusive open 没起作用")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_permission_guard_fires_on_a_live_path(self):
        """守卫必须在**用例能走到的**分支上: 直接给它一个 0755 的目录, 要求它炸。

        写在 `private_dir` 里的那个 if 只有 mkdtemp 反常时才执行 —— 删掉它没有一条
        用例会红, 所以这条纪律必须由一条能构造出该状态的用例来守。
        """
        d = Path(tempfile.mkdtemp(prefix="guard-test-"))
        try:
            d.chmod(0o755)
            with self.assertRaises(RuntimeError) as cm:
                ensure_private(d)
            self.assertIn("mode=0o755", str(cm.exception))
            with self.assertRaises(RuntimeError):
                write_private(d, "probe.py", "payload")   # 落点目录也走同一道门
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_the_guard_passes_a_real_private_dir(self):
        d = private_dir("guard-test-")
        try:
            self.assertIs(d, ensure_private(d))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_remote_names_are_not_predictable(self):
        a, b = remote_scratch_path("_contender.py"), remote_scratch_path("_contender.py")
        self.assertNotEqual(a, b, "同一个名字两次给出同一路径 = 还是固定名")
        self.assertTrue(a.endswith(".py"), "uv 按扩展名判定, 后缀不能丢")
        self.assertTrue(a.startswith("/tmp/"), "落点仍然是共享机上唯一的公共可写区")
        # 绝对锚点: 随机后缀的长度 (8 个 hex = 32 bit)。写死是防「退化成 pid 或时间戳」——
        # 那两个在共享机上都是可猜/会撞的。
        token = a.rsplit("-", 1)[1].split(".")[0]
        self.assertEqual(8, len(token), f"随机后缀应该是 8 位 hex, 实际 {token!r}")


class TestTeardownReachesQuiescence(unittest.TestCase):
    def test_terminate_then_wait_and_read_back(self):
        proc, remote = _FakeProc(), _FakeRemote()
        path = "/tmp/_contender-deadbeef.py"
        ev = stop_interferer(proc, remote, path)
        self.assertEqual(["terminate", "wait(timeout=10.0)"], proc.calls,
                         "只 request 不等 = 留孤儿 (DSH :19)")
        self.assertEqual(3, len(remote.commands), "pkill + pgrep + rm 三步都要有")
        self.assertIn("_contender-deadbeef.py", remote.commands[0])
        self.assertTrue(remote.commands[0].startswith("pkill -f "))
        self.assertTrue(remote.commands[2].startswith("rm -f "))
        self.assertEqual(0, ev["remote_still_running"])
        self.assertTrue(ev["quiescent"])
        self.assertNotIn("warning", ev)

    def test_a_path_needing_escaping_is_quoted(self):
        """`shlex.quote` 对安全字符**不加引号**，所以上面那条断言引号不是证据。
        真正的判据是：路径里带 shell 元字符时它必须被包住，否则那个路径就是命令注入的入口
        （远端路径来自我们自己，但这条是「装配环节不许靠假设」的同一族纪律）。"""
        remote = _FakeRemote()
        nasty = "/tmp/_c-0.py; touch /tmp/pwned"
        stop_interferer(_FakeProc(), remote, nasty)
        self.assertTrue(remote.commands[0].startswith("pkill -f '"),
                        f"没有 quote 就会多出第二条命令: {remote.commands[0]}")
        self.assertNotIn("touch", remote.commands[0].split("'")[0],
                        "注入的部分必须整个落在引号里")

    def test_kill_when_terminate_is_not_enough(self):
        proc = _FakeProc(terminate_never_exits=True)
        ev = stop_interferer(proc, _FakeRemote(), "/tmp/_c-12345678.py")
        self.assertIn("kill", proc.calls, "terminate 超时后必须升级并等到退出")
        self.assertEqual(["terminate", "wait(timeout=10.0)", "kill", "wait(timeout=None)"],
                         proc.calls)
        self.assertIn("kill", ev["local_wait"])

    def test_residual_process_is_surfaced_not_swallowed(self):
        """读回的证据要能改变结论: 远端还有进程时不许报成 quiescent。"""
        remote = _FakeRemote(pgrep_output="12345\n12346\n")
        ev = stop_interferer(_FakeProc(), remote, "/tmp/_c-abcdefab.py")
        self.assertEqual(2, ev["remote_still_running"])
        self.assertNotIn("quiescent", ev, "还有残留却报成静默 = 下一次读数被当成安静值")
        self.assertIn("不干净", ev["warning"])

    def test_no_residue_claim_without_a_read_back(self):
        """反空转: `pgrep` 这一步不能省。少了它, 上面那条「有残留」的判据就无从成立。"""
        remote = _FakeRemote()
        stop_interferer(_FakeProc(), remote, "/tmp/_c-00000000.py")
        self.assertTrue(any(c.startswith("pgrep") for c in remote.commands),
                        f"没有读回证据的拆除只是请求: {remote.commands}")


def _called_names(path: Path) -> set:
    """AST 取该文件里**真的被调用到**的名字。

    不用 `in text`: 变异标定 M12 撞出来的正是这个 —— 驱动把 `stop_interferer(...)` 调用
    退回成裸 `terminate()` 时，那行 import 还留在文本里，文本判据恒真地绿。同理注释里
    提一句 `chmod 600` 也不算做过（§八 那条「清点必须去注释」）。
    """
    with io.open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=str(path))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                out.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                out.add(fn.attr)
    return out


def _string_constants(path: Path) -> str:
    with io.open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=str(path))
    return "\n".join(n.value for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str))


class TestSharedPathRatchet(unittest.TestCase):
    def test_no_new_fixed_scratch_paths_in_the_scanned_tree(self):
        hits = _shared_path_hits()
        for f, where in sorted(hits.items()):
            self.assertEqual(sorted(RATCHET.get(f, set())), sorted(set(where)),
                             f"{f} 的共享路径清单与登记不一致: 现在 {sorted(set(where))} / "
                             f"登记 {sorted(RATCHET.get(f, set()))}。新增的要修 "
                             f"(私有目录 + 随机名), 确实要留的请把清单与理由一起改进 RATCHET。")
        self.assertEqual(set(RATCHET), set(hits),
                         f"登记的名单与实扫不一致。新命中: {set(hits) - set(RATCHET)}; "
                         f"已修完可以从名单里删掉: {set(RATCHET) - set(hits)}")

    def test_the_scanner_actually_finds_things(self):
        """反空转: 上面那条断言在「解析面窄过被测代码」时会恒真地绿 (doc 49 §八 那条陷阱)。
        这里要求扫描器确实命中了登记的那些字符串, 并且至少有一处来自 shell 字符串内部
        (`remote_executor.py` 的 `mkdir -p /tmp/...`), 证明它不是只认整串字面量。"""
        hits = _shared_path_hits()
        self.assertGreaterEqual(sum(len(v) for v in hits.values()), 12,
                                f"只扫出 {hits} —— 解析面可能已经窄过代码")
        executor = hits.get("operator_agent/remote_executor.py", [])
        self.assertTrue(any("/tmp/kXPU3" == h for h in executor),
                        f"shell 字符串里的路径没被认出来: {executor}")

    def test_the_drivers_use_the_helpers_rather_than_being_deleted(self):
        """ratchet 归零的另一条退路是「把用路径的代码删了」—— 这条盯住那两个驱动还在。

        判据是 **AST 里的调用**，不是文本出现: 变异标定 M12 显示文本判据只守住了半扇门
        —— 调用退回裸 `terminate()` 时 import 行仍带着 `stop_interferer` 这个词。
        """
        want = {"private_dir", "write_private", "remote_scratch_path"}
        for name in ("verify_interleaved_ab.py", "verify_util_decontamination.py"):
            called = _called_names(ROOT / "benchmarks" / name)
            self.assertTrue(want <= called, f"{name} 不再从 _scratch 拿落点了: 缺 {sorted(want - called)}")
            self.assertNotIn("mkdtemp", called,
                             f"{name} 绕开 private_dir 直接用 mkdtemp, 权限不再被核")
        called = _called_names(ROOT / "benchmarks" / "verify_interleaved_ab.py")
        self.assertIn("stop_interferer", called, "干扰流的拆除退回成 terminate() 了")
        strings = _string_constants(ROOT / "benchmarks" / "verify_interleaved_ab.py")
        self.assertIn("chmod 600", strings, "远端上传后要显式收权限 (scp 保留源 mode)")


class TestSmokeBurnerArm(unittest.TestCase):
    """smoke_spine_h20.py 的竞争臂 —— #69 纪律在它身上的落法

    旧写法两处违规 (ratchet 撞出来的): 固定名 `/tmp/burner_gpu{gpu}.log`
    (世界可写区的可预测路径) 与拆除用内容模式 `pkill -f '2048,2048'`
    (会误伤任何恰好带着这串的进程, 且只请求不读回)。这条钉住新写法的形状。
    """

    def test_burner_command_carries_its_random_path_as_the_kill_marker(self):
        """MARK = 落点本体: pkill/pgrep/rm 三个动作对着同一个不可预测路径。

        burner 是 `python -c` 内联代码, 落点本身不在它的 argv 里 —— 所以必须
        显式把落点塞进 argv (MARK= 赋值), `pkill -f <落点>` 才指得到它。
        """
        import smoke_spine_h20 as smoke
        log = "/tmp/burner-1a2b3c4d.log"
        cmd = smoke._burner_command(3, log)
        self.assertIn(f'MARK="{log}"', cmd)
        self.assertIn(f"> {log} 2>&1", cmd)

    def test_smoke_uses_the_helpers_not_ad_hoc_paths(self):
        """驱动不许绕开 helper 自己拼路径/拆除 (判据 = AST 里的调用, 非文本)"""
        called = _called_names(ROOT / "benchmarks" / "smoke_spine_h20.py")
        self.assertIn("remote_scratch_path", called,
                      "落点不再从 _scratch 拿随机名了")
        self.assertIn("stop_interferer", called,
                      "拆除退回成裸 pkill —— 没有读回的拆除只是请求")


if __name__ == "__main__":
    unittest.main(verbosity=2)
