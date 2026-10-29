"""MCP stdio server 的守卫：协议面与「stdout 只许走协议行」

这一层是给**宿主**用的（DSH 通过 `@deepseek-ai/dsh-mcp-client` 驱动它），所以两件事会
直接影响线上：协议对不对、以及 stdout 有没有被库的 print 污染（污染 = 客户的 JSON-RPC
流错位，表现是「工具时好时坏」，最难查）。

这里钉住：
1. 握手/版本协商/通知不响应/工具清单/调用/未知方法/坏行（坏行不打断流）；
2. **工具 schema 里没有放宽判据的参数**（与 `hitl_tools.ALLOWED_KEYS` 同一条纪律）；
3. 子进程真跑一遍：**每一行 stdout 都是合法 JSON**（这条只有起真进程才测得出）。

运行: python3 -m pytest tests/test_mcp_server.py -q
"""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from operator_agent import mcp_server as srv  # noqa: E402


def conversation(requests, *, log=None):
    instream = io.StringIO("".join(
        (r if isinstance(r, str) else json.dumps(r)) + "\n" for r in requests))
    outstream = io.StringIO()
    srv.serve(instream, outstream, log=log or io.StringIO())
    return [json.loads(line) for line in outstream.getvalue().splitlines() if line.strip()]


class TestHandshake(unittest.TestCase):

    def test_it_negotiates_a_supported_protocol_version(self):
        replies = conversation([{
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}},
        }])
        self.assertEqual(len(replies), 1)
        result = replies[0]["result"]
        self.assertEqual(result["protocolVersion"], "2025-06-18")
        self.assertEqual(result["serverInfo"]["name"], "operator")
        self.assertIn("tools", result["capabilities"])

    def test_it_ships_usage_instructions_for_the_model(self):
        """使用说明**跟着 server 走**（`initialize.instructions`），所以契约钉在这里。

        这段文字原来在 DSH 宿主插件里（`tool:operator` 提示段），和工具描述、设计文档
        是三份会互相漂的副本。搬到 server 之后，DSH 的 mcp-client 会自动把它注册成
        `mcp:operator` 段（`registerServerContext`），换别的 MCP 客户端也照样拿到。
        """
        replies = conversation([{
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }])
        instructions = replies[0]["result"]["instructions"]
        self.assertTrue(instructions.strip())
        # 三条硬要求：怎么读回包（route 与资格一起读）、冲突时怎么办、典型顺序。
        self.assertIn("eligibility", instructions)
        self.assertIn("hold_for_measurement", instructions)
        self.assertIn("operator_verify", instructions)
        self.assertIn("mcp__operator__", instructions, "告诉模型它在 DSH 里叫什么名字")
        # DSH 默认 maxInstructionBytes = 32 KiB；留足余量，别把客户端上限顶满。
        self.assertLess(len(instructions.encode("utf-8")), 8 * 1024)

    def test_an_unknown_protocol_falls_back_to_our_latest(self):
        replies = conversation([{
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "1999-01-01"},
        }])
        self.assertEqual(replies[0]["result"]["protocolVersion"],
                         srv.SUPPORTED_PROTOCOLS[0])

    def test_notifications_get_no_response(self):
        replies = conversation([
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "notifications/cancelled",
             "params": {"requestId": 7}},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        ])
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["id"], 2)

    def test_an_unknown_method_is_a_protocol_error(self):
        replies = conversation([{"jsonrpc": "2.0", "id": 3, "method": "no/such"}])
        self.assertEqual(replies[0]["error"]["code"], -32601)

    def test_a_broken_line_does_not_break_the_stream(self):
        log = io.StringIO()
        replies = conversation(["{not json", '{"jsonrpc":"2.0","id":9,"method":"ping"}'],
                               log=log)
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["id"], 9)
        self.assertIn("坏行", log.getvalue())

    def test_a_batch_is_answered_line_by_line(self):
        replies = conversation([[
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        ]])
        self.assertEqual([r["id"] for r in replies[0]], [1, 2])


class TestToolSurface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.replies = conversation([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])
        cls.tools = {t["name"]: t for t in cls.replies[0]["result"]["tools"]}

    def test_the_eight_actions_are_exposed(self):
        self.assertEqual(set(self.tools),
                         {"operator_prepare", "operator_verify", "operator_bench",
                          "operator_run", "operator_status", "operator_repair",
                          "operator_capabilities", "operator_cancel"})

    def test_every_tool_has_a_closed_json_schema(self):
        for name, tool in self.tools.items():
            schema = tool["inputSchema"]
            self.assertEqual(schema["type"], "object", name)
            self.assertFalse(schema.get("additionalProperties", True), name)
            self.assertTrue(tool.get("description"), name)

    def test_prepare_requires_exactly_request_shapes_run_dir(self):
        self.assertEqual(set(self.tools["operator_prepare"]["inputSchema"]["required"]),
                         {"request", "shapes", "run_dir"})

    def test_no_schema_parameter_can_relax_the_judgement(self):
        """与 hitl_tools 的白名单同一条纪律：schema 里出现这些名字就是开了口子。"""
        forbidden = {"max_error", "atol", "rtol", "skip_anticheat",
                     "require_double_confirm", "kernel_db", "failure_db",
                     "failure_gate", "disable_logging"}
        names = set()
        for tool in self.tools.values():
            names |= set(tool["inputSchema"].get("properties") or {})
        self.assertEqual(names & forbidden, set())

    def test_verify_takes_code_or_code_file_but_not_both(self):
        schema = self.tools["operator_verify"]["inputSchema"]
        self.assertIn("anyOf", schema)
        self.assertEqual(set(schema["required"]), {"run_dir"})


class TestToolCalls(unittest.TestCase):

    def test_capabilities_is_readable_json_and_marks_backends(self):
        replies = conversation([{
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "operator_capabilities", "arguments": {}},
        }])
        body = json.loads(replies[0]["result"]["content"][0]["text"])
        self.assertTrue(body["ok"])
        self.assertTrue(body["backends"]["mock"]["wired"])

    def test_prepare_then_verify_through_the_protocol(self):
        run = Path(tempfile.mkdtemp(prefix="mcp-round-")) / "run"
        replies = conversation([
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "operator_prepare", "arguments": {
                    "request": "把 64x64x64 的 fp32 矩阵乘法写成算子",
                    "shapes": [[64, 64, 64]], "run_dir": str(run), "min_speedup": 0.5}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "operator_verify", "arguments": {
                    "run_dir": str(run), "stop_after": "L1",
                    "code": "def kernel(A, B):\n    return A @ B\n"}}},
        ])
        prepared = json.loads(replies[0]["result"]["content"][0]["text"])
        verified = json.loads(replies[1]["result"]["content"][0]["text"])
        self.assertTrue(prepared["ok"])
        self.assertTrue(verified["ok"])
        self.assertEqual(verified["levels"], {"L1": True})
        self.assertNotIn("isError", replies[0]["result"])

    def test_a_refusal_is_iserror_with_the_reason_visible(self):
        replies = conversation([{
            "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "operator_prepare", "arguments": {
                    "request": "x", "shapes": [[64, 64, 64]],
                    "run_dir": tempfile.mkdtemp(prefix="mcp-bad-"), "max_error": 1.0}},
        }])
        result = replies[0]["result"]
        self.assertTrue(result.get("isError"))
        body = json.loads(result["content"][0]["text"])
        self.assertEqual(body["error"], "forbidden_parameter")

    def test_run_splits_tool_options_from_the_declaration(self):
        """`backend`/`max_rounds` 是运行选项，不许混进 declaration（会被白名单拒）。

        没配内层 LLM 时它会 fail-closed 成 `llm_not_configured` —— 这正好说明**参数拆分
        走通**了（否则报的是 `unknown_parameter`）。
        """
        import os
        saved = {k: os.environ.pop(k, None) for k in
                 ("LLM_BASE_URL", "LLM_API_KEY", "LLM_API_KEY_FILE")}
        try:
            replies = conversation([{
                "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                    "name": "operator_run", "arguments": {
                        "request": "把 64x64x64 的 fp32 矩阵乘法写成算子",
                        "shapes": [[64, 64, 64]],
                        "run_dir": tempfile.mkdtemp(prefix="mcp-run-"),
                        "backend": "mock", "max_rounds": 1}}},
            ])
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
        result = replies[0]["result"]
        body = json.loads(result["content"][0]["text"])
        self.assertTrue(result.get("isError"))
        self.assertEqual(body["error"], "llm_not_configured", body)

    def test_an_unknown_tool_is_an_error_result_not_a_crash(self):
        replies = conversation([{
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }])
        self.assertTrue(replies[0]["result"].get("isError"))
        self.assertEqual(json.loads(replies[0]["result"]["content"][0]["text"])["error"],
                         "unknown_tool")


class TestStdoutIsProtocolOnly(unittest.TestCase):
    """只有起**真进程**才测得出：库里的 print 会不会漏进 stdout。"""

    def test_every_stdout_line_is_one_jsonrpc_message(self):
        requests = "\n".join(json.dumps(r) for r in [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                        "clientInfo": {"name": "subprocess", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "operator_capabilities", "arguments": {}}},
        ]) + "\n"
        proc = subprocess.run(
            [sys.executable, "-m", "operator_agent.mcp_server"],
            input=requests, capture_output=True, text=True, cwd=str(ROOT), timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 3, f"期望 3 条响应，实际 stdout:\n{proc.stdout!r}")
        for line in lines:
            payload = json.loads(line)
            self.assertEqual(payload["jsonrpc"], "2.0")

    def test_the_selftest_runs(self):
        proc = subprocess.run(
            [sys.executable, "-m", "operator_agent.mcp_server", "--selftest"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("tools=['operator_prepare'", proc.stdout)


if __name__ == "__main__":
    unittest.main()
