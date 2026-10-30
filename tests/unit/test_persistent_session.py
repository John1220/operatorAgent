"""常驻会话的协议守卫 (doc 62)

`_RemoteSession` 是「一条 SSH 连接 + 一个常驻远端进程」的那层框架。它的失效方式都很
难查, 所以每一条都得有用例钉着:

- **超时与否不是一回事**: 没回显而进程还活着 = 这一次请求的世界未知; 回显了但
  `success=false` = 跑了且失败, 可归因。折成一个 `False` 就是 doc 60 记的那个缺陷。
- **id 对不上就是流错位**: 之后每一条都不能信, 不能按「收到一行 JSON」当回答。
- **stdout 上的杂音不等于错位**: uv / triton 会往 stdout 打东西, 那些行既不是回答
  也不该判会话死 (真错位会以 id 对不上出现)。
- **stderr 必须有人读**: 管道写满 64KB 没人读, 远端就阻塞在 print 上, 表现成「测量
  突然全部超时」。
- **一条会话是共享通道**: 两个并发调用会互相读走对方的回显。

测试不需要 SSH / GPU: 用当前解释器跑一段内联脚本当远端, 或用真的
`remote_scripts/runner_server.py`。
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from operator_agent.remote_executor import SessionError, _RemoteSession

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "remote_scripts" / "runner_server.py"

# run_kernel **import 时**会写进程级 GPU claim (CHECKLOG I68)。本文件在本地拉起真
# runner_server ⇒ 不处理的话它会往真实 ~/.operator_agent_gpu_claims/ 写声明 ——
# 测试不许动用户的 home (conftest「测试现场与真实现场分开」的同一条纪律)。指到
# 本模块自己的 tmp, 子进程经 env 继承; claim 本身是尽力而为, 不影响协议测试。
_CLAIM_TMP = tempfile.TemporaryDirectory(prefix="operator_agent_test_claims_")
os.environ["OPERATOR_AGENT_GPU_CLAIM_DIR"] = _CLAIM_TMP.name

_ECHO = (
    "import sys, json\n"
    "for line in sys.stdin:\n"
    "    req = json.loads(line)\n"
    "    print(json.dumps({'id': req.get('id'), 'result': {'echo': req.get('action')}}),"
    " flush=True)\n"
)
_WRONG_ID = (
    "import sys, json\n"
    "for line in sys.stdin:\n"
    "    req = json.loads(line)\n"
    "    print(json.dumps({'id': 999999, 'result': {'echo': 'wrong'}}), flush=True)\n"
)
_SLEEPY = (
    "import sys, json, time\n"
    "line = sys.stdin.readline()\n"
    "time.sleep(5)\n"
    "print(json.dumps({'id': json.loads(line).get('id'), 'result': {}}), flush=True)\n"
)
_NOISY = (
    "import sys, json\n"
    "print('uv warning: something on stdout', flush=True)\n"
    "for line in sys.stdin:\n"
    "    req = json.loads(line)\n"
    "    print('triton: another stray line', flush=True)\n"
    "    print(json.dumps({'id': req.get('id'), 'result': {'echo': req.get('action')}}),"
    " flush=True)\n"
)


def _fake(body: str, timeout: float = 10.0) -> _RemoteSession:
    sess = _RemoteSession([sys.executable, "-u", "-c", body], timeout=timeout,
                          label="fake-remote")
    sess.start()
    return sess


def _real_server(timeout: float = 60.0) -> _RemoteSession:
    sess = _RemoteSession([sys.executable, "-u", str(SERVER)], timeout=timeout,
                          label="local-server")
    sess.start()
    return sess


class TestProtocolMachinery(unittest.TestCase):
    def tearDown(self):
        for sess in getattr(self, "_sessions", []):
            sess.close(grace=0.2)
        self._sessions = []

    def _track(self, sess):
        self._sessions = getattr(self, "_sessions", []) + [sess]
        return sess

    def test_round_trip_echoes_the_id(self):
        s = self._track(_fake(_ECHO))
        self.assertEqual({"echo": "ping"}, s.call({"action": "ping"}))

    def test_ids_advance_and_are_echoed_every_time(self):
        s = self._track(_fake(_ECHO))
        for i in range(1, 6):
            self.assertEqual({"echo": f"a{i}"}, s.call({"action": f"a{i}"}))
            self.assertEqual(i, s._next_id)

    def test_timeout_is_its_own_state_not_a_failure(self):
        """没回显 + 进程还活着 ⇒ 世界未知。**不能**折成 success=False。"""
        s = self._track(_fake(_SLEEPY, timeout=0.3))
        with self.assertRaises(SessionError) as ctx:
            s.call({"action": "benchmark"})
        self.assertEqual("timeout", ctx.exception.session_state)
        self.assertFalse(s.alive)

    def test_desync_is_detected(self):
        s = self._track(_fake(_WRONG_ID))
        with self.assertRaises(SessionError) as ctx:
            s.call({"action": "ping"})
        self.assertEqual("desync", ctx.exception.session_state)

    def test_eof_is_detected_when_the_remote_dies(self):
        s = self._track(_fake("pass"))
        with self.assertRaises(SessionError) as ctx:
            s.call({"action": "ping"})
        self.assertEqual("eof", ctx.exception.session_state)

    def test_stdout_noise_is_not_treated_as_desync(self):
        """uv/triton 的杂音走 stdout 是常态; 它不是回答, 也不该判会话死"""
        s = self._track(_fake(_NOISY))
        self.assertEqual({"echo": "ping"}, s.call({"action": "ping"}))
        self.assertEqual({"echo": "ping"}, s.call({"action": "ping"}))
        self.assertTrue(any("uv warning" in x for x in s._noise_tail))

    def test_concurrent_calls_do_not_steal_each_others_replies(self):
        """一条会话是共享通道 —— 锁在里面, 不靠调用方守纪律"""
        s = self._track(_fake(_ECHO))
        with ThreadPoolExecutor(max_workers=8) as pool:
            got = list(pool.map(lambda i: s.call({"action": f"c{i}"}), range(16)))
        self.assertEqual([{"echo": f"c{i}"} for i in range(16)], got)

    def test_close_is_idempotent_and_reaches_the_process(self):
        s = self._track(_fake(_ECHO))
        s.call({"action": "ping"})
        s.close(grace=2.0)
        self.assertFalse(s.alive)
        self.assertIsNotNone(s._proc.poll(), "关闭后本地进程必须已经退出")
        s.close(grace=0.1)          # 再关一次不许炸

    def test_a_bad_request_does_not_kill_the_session(self):
        s = self._track(_real_server())
        bad = s.call({"action": "definitely_not_an_action"})
        self.assertFalse(bad.get("success"))
        self.assertIn("Unknown action", bad.get("error", ""))
        self.assertTrue(s.alive, "一个坏请求不该带走整个会话")
        ok = s.call({"action": "ping"})
        self.assertTrue(ok.get("success"))


class TestRealServerOverTheWire(unittest.TestCase):
    """用**真的** `runner_server.py` 当远端 (本地解释器, 不需要 SSH/GPU)"""

    def test_ping_reports_the_pinned_card(self):
        s = _real_server()
        try:
            r = s.call({"action": "ping"})
            self.assertTrue(r.get("pong"))
            self.assertIn("selected_gpu", r)
            self.assertIsInstance(r["selected_gpu"], int)
        finally:
            s.close(grace=3.0)

    def test_a_real_action_goes_through_shared_dispatch(self):
        """`gpu_info` 走的是与一次性路径**同一份** `run_kernel.run_action`"""
        s = _real_server()
        try:
            r = s.call({"action": "gpu_info"})
            self.assertIn("name", r)
            self.assertIn("device", r)
        finally:
            s.close(grace=3.0)

    def test_malformed_line_gets_an_error_but_the_server_survives(self):
        """非 JSON 行由服务端答一个 id=null 的错误, 然后继续服务"""
        proc = subprocess.Popen([sys.executable, "-u", str(SERVER)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        try:
            proc.stdin.write("this is not json\n")
            proc.stdin.flush()
            first = json.loads(proc.stdout.readline())
            self.assertIsNone(first["id"])
            self.assertFalse(first["result"]["success"])
            proc.stdin.write(json.dumps({"id": 7, "action": "ping"}) + "\n")
            proc.stdin.flush()
            second = json.loads(proc.stdout.readline())
            self.assertEqual(7, second["id"])
            self.assertTrue(second["result"]["pong"])
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_shutdown_exits_cleanly(self):
        s = _real_server()
        s.call({"action": "ping"})
        s.close(grace=5.0)
        self.assertIsNotNone(s._proc.poll(), "shutdown 之后服务端该自己退出")


class TestSessionStartup(unittest.TestCase):
    """握手: 会话开起来时必须先确认服务端**真的应答**, 并记下它钉住的那张卡

    `ping` 因此有了生产调用者 (否则它就是一条死分派, 契约测试会响)。卡号必须是读数:
    选卡发生在远端 runner import 时, 一次会话固定在同一张卡上, 「固定在哪张」不该是推断。

    用 `_session_argv` 这条接缝把「远端」换成本地进程 —— 不需要 SSH, 也不需要 GPU。
    """

    def _executor(self, argv_factory):
        from operator_agent.remote_executor import SSHExecutor
        ex = SSHExecutor(persistent=True, timeout=30)
        ex._deployed = True          # 测试不碰远端, 跳过 deploy
        ex._gpu_pin = -1             # 已解析成「不固定」⇒ _pinned_gpu() 不再走 ssh
        ex._session_argv = argv_factory
        self.addCleanup(ex.close)
        return ex

    def test_the_handshake_records_the_pinned_card(self):
        ex = self._executor(lambda cmd: [sys.executable, "-u", str(SERVER)])
        r = ex._session_call({"action": "gpu_info"})      # 首次调用触发握手
        self.assertIn("name", r)
        self.assertIsInstance(ex.session_gpu, int)
        self.assertTrue(ex._session.alive)

    def test_a_server_that_never_answers_the_handshake_fails_closed(self):
        ex = self._executor(lambda cmd: [sys.executable, "-u", "-c", "pass"])
        with self.assertRaises(SessionError) as ctx:
            ex._session_call({"action": "gpu_info"})
        self.assertEqual("eof", ctx.exception.session_state)
        self.assertIsNone(ex._session, "握手失败不许留下一个半开的会话")
        self.assertIsNone(ex.session_gpu)

    def test_persistent_is_refused_on_platforms_without_run_action(self):
        """不给「悄悄退化成一次性」的路 —— 声明了常驻就得真常驻"""
        from operator_agent.remote_executor import SSHExecutor
        with self.assertRaises(ValueError):
            SSHExecutor(platform="kunlunxin", persistent=True)


class TestOneDispatchNotTwo(unittest.TestCase):
    """常驻服务与一次性路径共用同一份分派 —— 抄第二套 if 链就会慢慢漂开"""

    def test_server_calls_the_runner_action(self):
        src = SERVER.read_text(encoding="utf-8")
        tree = ast.parse(src)
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertIn("run_action", called,
                      "服务端必须调 run_kernel.run_action, 不能自己写一套分派")

    def test_runner_exposes_exactly_one_dispatch_site(self):
        """分派现场只许出现在一个函数里 (elif 链本身是多个 If 节点, 数 If 没意义)"""
        src = (ROOT / "remote_scripts" / "run_kernel.py").read_text(encoding="utf-8")
        tree = ast.parse(src)

        def compares_action(scope):
            return any(isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                       and isinstance(n.test.left, ast.Name) and n.test.left.id == "action"
                       for n in ast.walk(scope))

        sites = [n.name for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and compares_action(n)]
        self.assertEqual(["run_action"], sites,
                         f"分派现场在 {sites} —— 只许在 run_action 里有一处")

    def test_ping_and_shutdown_stay_out_of_the_kernel_runner(self):
        """服务端专属 action 不许进 kernel runner 的分派 —— 契约测试按 action 集合
        对照两个平台的 runner, 混进去会让它以为 kernel runner 也支持 ping"""
        src = (ROOT / "remote_scripts" / "run_kernel.py").read_text(encoding="utf-8")
        for name in ('"ping"', "'ping'", '"shutdown"'):
            self.assertNotIn(name, src, f"kernel runner 里出现了服务端专属 action {name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)