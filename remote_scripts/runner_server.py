"""常驻远端执行服务 —— 一条 SSH 连接、一个进程、多请求

为什么要有它 (真机实测, doc 62)：`SSHExecutor` 每次调用都新起一个远端进程，固定开销
≈4.5 s，而测量本身只有毫秒级 ——

    ssh 握手+session            0.57 s
    import torch                1.09 s
    CUDA 上下文首次创建          0.20 s
    首个 CUDA 张量              0.24 s
    runner 本体 import          2.76 s
    ── 同一进程内连跑三次测量 ──   9.7 / 9.0 / 9.3 ms

一次心跳 8.6 s 墙钟里，两条臂的测量合计只有 18.8 ms。这份服务把固定开销摊到整个会话。

协议 (行分隔 JSON, 一请求一行、一响应一行)：

    请求  {"id": <JSON 标量>, "action": "...", ...}
    响应  {"id": <原样回显>, "result": {...}}

**`id` 回显是归因设施，不是方便**：客户端据此把三种状态分开 —— 没回显（进程死了，或
还在算）、回显且 `result.success=false`（跑了且失败，可归因到这一请求）、回显的 id 不是
自己发的（流错位）。一次性路径把这三态折成一个 `success=False`（doc 60 记的那个读数
缺陷：「超时」与「核跑完了但输出不对」同形）。

只服务端认得的 action：`ping`（存活 + 报钉住的卡）/ `shutdown`（干净退出）。
**不放进 `run_kernel.run_action` 的分派链** —— `tests/test_runner_contract.py` 按 action
集合对照两个平台的 kernel runner，加进去会让它以为 kernel runner 也支持 `ping`。

stdout 只许走协议行：处理请求期间把 `sys.stdout` 换成 stderr，免得任何库的 `print`
把流打断（真错位时客户端那条 id 校验仍然抓得住，不会静默串行）。

⚠️ 一个真实的语义变化：选卡发生在 `run_kernel` **import 时**（它在那儿设
`CUDA_VISIBLE_DEVICES`），所以一次会话会把卡**钉住**。对可比性是好事（同一会话里候选与
基线、各轮之间都在同一张卡上），代价是会话中途出现更空的卡也不会换。
"""
import json
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 协议行只能写进真 stdout；处理期间 sys.stdout 会被临时换成 stderr
_REAL_STDOUT = sys.stdout

# Publish lifecycle identity BEFORE expensive Torch imports / CUDA startup.
# A failed handshake must still leave enough evidence to clean up our runner.
from session_control import TOKEN_ENV, runner_identity  # noqa: E402
if os.environ.get(TOKEN_ENV):
    try:
        _REAL_STDOUT.write(json.dumps({"session_identity": runner_identity(__file__)}) + "\n")
        _REAL_STDOUT.flush()
    except OSError:
        pass  # Non-Linux: no /proc identity; client must not claim remote cleanup.

# Same dispatch as the one-shot runner, never a second kernel action table.
import run_kernel  # noqa: E402
from session_inputs import SessionInputs, PROTOCOL as INPUT_CACHE_PROTOCOL  # noqa: E402

_INPUTS = SessionInputs()


def _write(obj: dict) -> None:
    _REAL_STDOUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _REAL_STDOUT.flush()


def _reply(req_id, result: dict) -> None:
    """统一的响应形状 —— 只有一种 (一形两状那类 formatter bug 见 CHECKLOG I29)"""
    _write({"id": req_id, "result": result})


def _serve_one(line: str) -> bool:
    """处理一行请求。返回 False 表示该收工了 (shutdown / 流结束)"""
    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        _reply(None, {"success": False, "error": f"Invalid JSON: {e}"})
        return True
    if not isinstance(req, dict):
        _reply(None, {"success": False,
                      "error": f"请求必须是 JSON 对象: {type(req).__name__}"})
        return True

    req_id = req.get("id")
    action = req.get("action", "execute")

    if action == "shutdown":
        _reply(req_id, {"success": True, "shutdown": True})
        return False
    if action == "ping":
        # 报出钉住的卡 —— 会话把它固定下来了，客户端该知道是哪一张
        _reply(req_id, {"success": True, "pong": True, "pid": os.getpid(),
                        "input_cache": _INPUTS.advertisement(),
                        "selected_gpu": run_kernel._SELECTED_GPU_ID})
        return True

    # Session-only control actions, never additions to run_kernel's device table.
    proof = None
    try:
        if action == "input_begin":
            _reply(req_id, _INPUTS.begin(req.get("session_id"), req.get("manifest")))
            return True
        if action == "input_chunk":
            _reply(req_id, _INPUTS.chunk(req.get("session_id"), req.get("manifest_hash"),
                req.get("storage_hash"), req.get("offset"), req.get("data"), req.get("encoding")))
            return True
        if action == "input_commit":
            _reply(req_id, _INPUTS.commit(req.get("session_id"), req.get("manifest_hash")))
            return True
        inputs = req.get("inputs")
        if isinstance(inputs, dict) and inputs.get("protocol") == INPUT_CACHE_PROTOCOL:
            if action not in {"verify", "benchmark_with_baselines"}:
                raise ValueError("session inputs unsupported for this action")
            req["inputs"], proof = _INPUTS.resolve(inputs)
    except Exception as exc:  # Input preparation never becomes candidate failure.
        _INPUTS.abort()
        _reply(req_id, {"success": False, "input_identity_unknown": True,
                        "error": "session input unknown: " + str(exc)})
        return True

    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = run_kernel.run_action(req)
    except Exception as e:            # 一个坏请求不该带走整个会话
        result = {"success": False, "error": f"{type(e).__name__}: {e}",
                  "traceback_tail": traceback.format_exc().strip().splitlines()[-6:]}
    finally:
        sys.stdout = saved

    if not isinstance(result, dict):
        result = {"success": False, "error": f"run_action 返回了非载荷: {result!r}"}
    if proof is not None:
        result["input_transfer"] = proof
    _reply(req_id, result)
    return True


def main() -> int:
    try:
        for line in sys.stdin:            # EOF = 父进程关掉了 stdin
            line = line.strip()
            if not line:
                continue
            if not _serve_one(line):
                break
    finally:
        _INPUTS.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
