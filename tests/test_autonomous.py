"""无人值守整段运行（Phase 3）的守卫：**进程分离 + 四态相位**

这一层的失败形态比成功形态重要，所以钉住的是：

1. **没配内层 LLM 就拒绝**（`llm_not_configured`）—— 不许起一个跑不出结论的进程；
2. **起得来、跑得完**：用一个本地假的 OpenAI 兼容端点把 `Conductor` 那整段跑完
   （零 GPU、零真实凭据、不联网），并检查 `run-status.json` / `conductor.json` / 报告；
3. **相位四态**：`running` / `done` / `failed` / **`lost`**（进程没了但没写终态 ——
   这一档最容易被读成"还在跑"）；
4. 子进程的落点收在 run 目录里（不写仓库、不写 home 的默认位置）。

运行: python3 -m pytest tests/test_autonomous.py -q
"""

import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile

from operator_agent import hitl_tools as ht

SKETCH = {"algorithm": "matmul",
          "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
          "memory_plan": {}, "pipeline": [], "optimization_hints": []}
CODE = "def kernel(A, B):\n    return A @ B\n"


class _FakeLLM(BaseHTTPRequestHandler):
    """假 OpenAI 兼容端点：Designer 要 JSON，Coder 要代码。"""

    def do_POST(self):                                            # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = "".join(m.get("content", "") for m in body.get("messages", []))
        content = json.dumps(SKETCH, ensure_ascii=False) if "设计专家" in prompt else CODE
        payload = json.dumps({"choices": [{"message": {"content": content},
                                           "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 10, "completion_tokens": 10,
                                        "total_tokens": 20}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):                                 # 静音
        pass


class _Server:
    def __enter__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLLM)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def declaration(**over):
    base = {"request": "把 64x64x64 的 fp32 矩阵乘法写成算子",
            "shapes": [[64, 64, 64]], "min_speedup": 0.5}
    base.update(over)
    return base


class TestItRefusesWithoutAnInnerLlm(unittest.TestCase):

    def test_no_endpoint_means_no_process(self):
        saved = {k: os.environ.pop(k, None) for k in
                 ("LLM_BASE_URL", "LLM_API_KEY", "LLM_API_KEY_FILE")}
        try:
            run = Path(tempfile.mkdtemp(prefix="auto-nollm-")) / "run"
            with self.assertRaises(ht.ToolError) as ctx:
                ht.run_autonomous(declaration(), run)
            self.assertEqual(ctx.exception.code, "llm_not_configured")
            self.assertFalse((run / ht.STATUS_FILE).exists())
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


class TestItRunsTheWholeThing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.saved = {k: os.environ.get(k) for k in
                     ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")}
        cls.tmp = Path(tempfile.mkdtemp(prefix="auto-run-"))
        cls.round_dir = cls.tmp / "run"
        with _Server() as srv:
            os.environ["LLM_BASE_URL"] = f"http://127.0.0.1:{srv.port}/v1"
            os.environ["LLM_API_KEY"] = "fake-key"
            os.environ["LLM_MODEL"] = "fake-model"
            repo_output = Path(__file__).resolve().parents[1] / "output"
            cls.repo_corpus_before = {
                name: ((repo_output / name).stat().st_mtime,
                       (repo_output / name).stat().st_size)
                for name in ("kernel_db.jsonl", "failure_db.jsonl")
                if (repo_output / name).exists()}
            cls.started = ht.run_autonomous(declaration(), cls.round_dir,
                                            backend="mock", max_rounds=1)
            cls.status = cls._wait(srv, timeout=180)

    @classmethod
    def _wait(cls, srv, timeout):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            last = ht.autonomous_status(cls.round_dir)
            if last.get("phase") in ("done", "failed", "lost"):
                return last
            time.sleep(0.5)
        return last

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_it_returns_immediately_with_a_pid_and_a_status_path(self):
        self.assertTrue(self.started["ok"])
        self.assertEqual(self.started["phase"], "running")
        self.assertIsInstance(self.started["pid"], int)
        self.assertTrue(Path(self.started["status"]).name == ht.STATUS_FILE)

    def test_the_round_finishes_and_the_summary_is_written(self):
        self.assertEqual(self.status["phase"], "done", self.status)
        self.assertIsNone(self.status["error"])
        summary = self.status["summary"]
        self.assertTrue(summary["all_passed"])
        self.assertGreaterEqual(summary["rounds"], 1)
        self.assertIn("L1", summary["levels"])
        self.assertIn("L3", summary["levels"])

    def test_the_report_lands_inside_the_run_dir(self):
        """`Conductor` 把 report.md 写进**产物目录**（run/output/.../<id>/），不是 run 根。

        这个区别是刻意的：`verify` 那条路每一步写 `run/report.md`（覆盖式、一次回合），
        无人值守这条路每次运行一个产物目录（追加式、可回放）。
        """
        saved_dir = Path(self.status["summary"]["path"] or "")
        self.assertTrue(saved_dir.is_dir(), saved_dir)
        report = saved_dir / "report.md"
        self.assertTrue(report.is_file(), f"没找到 {report}")
        self.assertIn("自动跑报告", report.read_text(encoding="utf-8"))
        # macOS 的 /var 是指向 /private/var 的软链 —— 必须 resolve 之后比较
        self.assertTrue(str(report.resolve()).startswith(str(self.round_dir.resolve())),
                        f"报告跑到 run 目录外面去了: {report}")

    def test_nothing_is_written_outside_the_run_dir(self):
        for child in (self.round_dir / "output", self.round_dir / "log"):
            self.assertTrue(child.exists(), child)
        status = json.loads((self.round_dir / ht.STATUS_FILE).read_text(encoding="utf-8"))
        # ⚠️ `OPERATOR_AGENT_*_DB=off` **关不掉持久化**（2026-10-08 起普通环境变量只能
        # 重定向）；写 "off" 会被解析回仓库里那份语料。所以验的是「重定向进 run 目录」。
        self.assertEqual(status["corpus"], {"kernel_db": "run-local",
                                            "failure_db": "run-local"})

    def test_the_repo_corpus_is_untouched(self):
        repo_output = Path(__file__).resolve().parents[1] / "output"
        for name, before in self.repo_corpus_before.items():
            path = repo_output / name
            self.assertTrue(path.exists(), name)
            self.assertEqual((path.stat().st_mtime, path.stat().st_size), before,
                             f"{name} 在本次无人值守运行期间被改过 —— 语料写回仓库了")

    def test_the_log_is_kept_next_to_the_round(self):
        log = (self.round_dir / "autonomous.log").read_text(encoding="utf-8")
        self.assertIn("[autonomous] done", log)


class TestThePhaseMachine(unittest.TestCase):

    def test_a_never_started_dir_is_not_reported_as_running(self):
        out = ht.autonomous_status(Path(tempfile.mkdtemp(prefix="auto-none-")))
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "run_not_started")

    def test_a_dead_pid_without_final_state_is_lost_not_running(self):
        """这一档最危险：进程没了、终态没写。读成 running 会让人一直等。"""
        run = Path(tempfile.mkdtemp(prefix="auto-lost-"))
        (run / ht.STATUS_FILE).write_text(json.dumps({
            "phase": "running", "pid": 999999, "started_at": time.time(),
            "backend": "mock", "max_rounds": 1}), encoding="utf-8")
        out = ht.autonomous_status(run)
        self.assertEqual(out["phase"], "lost")
        self.assertFalse(out["process_alive"])

    def test_a_finished_run_is_left_alone(self):
        run = Path(tempfile.mkdtemp(prefix="auto-done-"))
        (run / ht.STATUS_FILE).write_text(json.dumps({
            "phase": "done", "pid": 1, "started_at": 1, "finished_at": 2,
            "duration_s": 1}), encoding="utf-8")
        out = ht.autonomous_status(run)
        self.assertEqual(out["phase"], "done")
        self.assertIsNone(out["process_alive"])


if __name__ == "__main__":
    unittest.main()
