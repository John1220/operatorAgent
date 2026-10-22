"""最小 MCP stdio server：把 `hitl_tools` 的**六个**工具暴露给宿主的 agent

为什么是这个形态（`docs/dsh-plugin-evaluation.md` §6.2 / §9.6）：

DSH 的插件必须是 Cordis 的 ESM 模块，而本仓是 Python —— 两者之间**官方支持**的接缝只有
两个：`ctx.subprocess` 与 MCP。MCP 这条不需要写任何 JS：宿主侧加一条
`@deepseek-ai/dsh-mcp-client` 配置行，工具就以 `mcp__<serverName>__<tool>` 出现。

**没有引第三方 SDK**：MCP 的 stdio 传输就是「一行一个 JSON-RPC 2.0 消息」，用标准库足够，
与本仓「控制面零第三方依赖」的立场一致（也避免把 `mcp` 包变成新的部署前提）。

协议面（只实现会被用到的哪些，其余显式报错，不假装支持）：

    initialize                 → 版本协商 + capabilities.tools
    notifications/initialized  → 无响应（通知没有 id）
    notifications/cancelled    → 无响应（本 server 的调用是同步的，取消靠进程退出）
    ping                       → {}
    tools/list                 → 六个工具的 JSON Schema（静态表：改了要重启进程）
    tools/call                 → 转发给 hitl_tools；拒绝时 isError=true + 可归因的文本

硬约束：**stdout 只许走协议行**。任何别的 print 都会把流打断（DSH 那条 `id` 校验会报
「流错位」），所以这里在启动时把 `sys.stdout` 换成 stderr，只在 `_emit` 里用真正的 stdout。
这与 `remote_scripts/runner_server.py` 的写法同源（同一个教训：doc 60）。

用法：

    python -m operator_agent.mcp_server            # 宿主通过 stdio 驱动
    python -m operator_agent.mcp_server --selftest # 不依赖宿主，自己走一遍握手（给人看）
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any, BinaryIO, Optional, TextIO

from . import __version__
from .hitl_tools import (ToolError, autonomous_status, capabilities, prepare,
                         repair, run_autonomous, verify)

SERVER_NAME = "operator"
SERVER_VERSION = __version__

#: 本 server 认的协议版本（新→旧）。协商时：客户端给的支持就用它的，
#: 不支持就回自己最新的一个 —— SDK 会据此决定是否继续。
SUPPORTED_PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05", "2024-10-07")

#: 给模型的**使用说明**，走 `initialize` 的 `instructions` 字段。
#:
#: 为什么放这儿而不是写进 DSH 宿主插件：任何 MCP 客户端都会把 server 的 instructions
#: 渲染进上下文（DSH 把它做成 `mcp:operator` 提示段，order 取 `MCP_SERVERS`），所以这段
#: 规则只有**一处**来源，不会和工具描述、设计文档漂成三个版本；换客户端也照样在。
#: 长度上限由客户端定（DSH 默认 `maxInstructionBytes` 32 KiB），这里保持精简。
INSTRUCTIONS = """\
算子判定层：GPU 算子（Triton/CUDA）的编译、正确性、性能判定与修复。

什么时候用：用户要把一个算子写成 / 改好 / 测快，或问「这个核对不对、快不快、读数可不可信」时。
什么时候**不要**用：纯 CPU / 非算子任务；没有 GPU 却要做性能结论（`backend=mock` 的读数
不代表加速卡成绩）。

典型顺序：`operator_prepare` 先冻结契约（声明 shapes / precision / required_caps），再
`operator_verify` 判候选；已有核要修就用 `operator_repair`；从头自动化用 `operator_run`
加 `operator_status` 轮询。（DSH 里这些工具显示为 `mcp__operator__*`。）

读回包时：`levels` 看判到哪一级；**`route` 与 `eligibility` 必须一起读** —— 两者冲突时
`conflict.action=hold_for_measurement`，此时不要照着 `route` 迭代，先修判据（形状 / 精度 /
required_caps）。`validity=degraded` 或 `uninformative` 的读数不驱动调参或入库。
"""

_SHAPES = {"type": "array", "minItems": 1,
           "items": {"type": "array", "minItems": 1,
                     "items": {"type": "integer", "minimum": 0}},
           "description": "形状分布：matmul 是 [M,N,K]，逐元素/归一化是 [M,N]"}


def _tool_definitions() -> list[dict]:
    return [
        {
            "name": "operator_prepare",
            "description": ("把一个算子任务冻结成可复算的「回合」：返回契约哈希、契约自带的"
                            "输入、必须参与的 case。之后用 operator_verify 判定候选源码。"
                            "不跑 GPU、不调 LLM；判据（容差/反作弊）不可由本工具放宽。"),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["request", "shapes", "run_dir"],
                "properties": {
                    "request": {"type": "string", "description": "自然语言请求"},
                    "shapes": _SHAPES,
                    "run_dir": {"type": "string", "description": "回合目录（绝对路径更好；调用方拥有布局）"},
                    "hardware": {"type": "string", "description": "默认 mock_gpu"},
                    "dsl": {"type": "string", "description": "默认 python（真机上是 triton）"},
                    "precision": {"type": "string", "enum": ["fp32", "fp16"]},
                    "min_speedup": {"type": "number", "description": "性能门槛"},
                    "required_caps": {"type": "array", "items": {"type": "string"},
                                      "description": "执行侧必须具备的能力，如 real_device"},
                    "framework": {"type": "string", "description": "默认 pytorch"},
                    "notes": {"type": "string", "description": "只进产物，不进判据"},
                },
            },
        },
        {
            "name": "operator_verify",
            "description": ("对一份候选内核源码做 L1/L2/L3 判定（不生成、不修复、不搜索），"
                            "返回正交事实 + 处方 + 资格五位 + 一份报告。"
                            "**route 与 eligibility 必须一起读**：两者冲突时 conflict.action 会是 "
                            "hold_for_measurement，此时不要照着 route 走。"),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["run_dir"],
                "anyOf": [{"required": ["code"]}, {"required": ["code_file"]}],
                "properties": {
                    "run_dir": {"type": "string"},
                    "code": {"type": "string", "description": "候选源码文本"},
                    "code_file": {"type": "string", "description": "候选源码文件路径（与 code 二选一）"},
                    "stop_after": {"type": "string", "enum": ["L1", "L2", "L3"]},
                    "backend": {"type": "string", "enum": ["mock", "local", "ssh"],
                                "default": "mock"},
                    "host": {"type": "string",
                             "description": "backend=ssh 必需：主机名或 ~/.ssh/config 别名"},
                    "port": {"type": "integer"},
                    "user": {"type": "string"},
                    "key_path": {"type": "string", "description": "私钥路径（引用；值不落库）"},
                    "work_dir": {"type": "string"},
                    "timeout": {"type": "integer", "default": 300},
                    "persistent": {"type": "boolean", "default": False},
                    "target_gflops": {"type": "number"},
                },
            },
        },
        {
            "name": "operator_run",
            "description": ("起一次**无人值守**的整段运行：本仓的 Conductor 自己设计/写核/修/搜索，"
                            "立刻返回 pid 与 run_dir，之后用 operator_status 轮询。"
                            "需要宿主给内层 LLM 配置（LLM_BASE_URL / LLM_MODEL / LLM_API_KEY_FILE），"
                            "否则立刻拒绝而不是假装在跑。"),
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "required": ["request", "shapes", "run_dir"],
                "properties": {
                    "request": {"type": "string"},
                    "shapes": _SHAPES,
                    "run_dir": {"type": "string"},
                    "hardware": {"type": "string"},
                    "dsl": {"type": "string"},
                    "precision": {"type": "string", "enum": ["fp32", "fp16"]},
                    "min_speedup": {"type": "number"},
                    "required_caps": {"type": "array", "items": {"type": "string"}},
                    "framework": {"type": "string"},
                    "notes": {"type": "string"},
                    "max_rounds": {"type": "integer", "default": 1},
                    "backend": {"type": "string", "enum": ["mock", "local", "ssh"], "default": "mock"},
                    "host": {"type": "string"},
                    "port": {"type": "integer"},
                    "user": {"type": "string"},
                    "key_path": {"type": "string"},
                    "work_dir": {"type": "string"},
                    "timeout": {"type": "integer", "default": 300},
                    "persistent": {"type": "boolean", "default": False},
                    "target_gflops": {"type": "number"},
                },
            },
        },
        {
            "name": "operator_status",
            "description": ("查一次无人值守运行的相位。四态：running / done / failed / **lost**"
                            "（进程没了但没写终态 —— 不许读成还在跑）。"),
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "required": ["run_dir"],
                "properties": {"run_dir": {"type": "string"}},
            },
        },
        {
            "name": "operator_repair",
            "description": ("给一份**已有的**内核源码：判 → 读处方 → 调内层模型修 → 再判，最多 N 轮。"
                            "判据全在既有模块里；处方为 hold（这一轮没判出对象）时不修。"
                            "需要内层 LLM（与 operator_run 同一套配置）。"),
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "required": ["run_dir"],
                "anyOf": [{"required": ["code"]}, {"required": ["code_file"]}],
                "properties": {
                    "run_dir": {"type": "string"},
                    "code": {"type": "string"},
                    "code_file": {"type": "string"},
                    "max_rounds": {"type": "integer", "default": 2},
                    "stop_after": {"type": "string", "enum": ["L1", "L2", "L3"]},
                    "backend": {"type": "string", "enum": ["mock", "local", "ssh"], "default": "mock"},
                    "host": {"type": "string"},
                    "port": {"type": "integer"},
                    "user": {"type": "string"},
                    "key_path": {"type": "string"},
                    "work_dir": {"type": "string"},
                    "timeout": {"type": "integer", "default": 300},
                    "persistent": {"type": "boolean", "default": False},
                    "target_gflops": {"type": "number"},
                },
            },
        },
        {
            "name": "operator_capabilities",
            "description": ("只读：这台机器上判定层能做什么（装了什么运行时、哪些算子有输入布局、"
                            "哪些后端已接线）。不跑活、不联网、不探测 SSH。"),
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
    ]


def _jsonrpc_ok(msg_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _jsonrpc_err(msg_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def _text_result(payload: dict, *, is_error: bool = False) -> dict:
    """`tools/call` 的结果：正文是 JSON 文本，模型与宿主都能直接读。"""
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    out = {"content": [{"type": "text", "text": body}]}
    if is_error:
        out["isError"] = True
    return out


#: `operator_run` 的**运行选项**（相对 declaration 的白名单）。两套词表分开是有意的：
#: declaration 决定"判什么"，运行选项决定"怎么跑"，前者不许被后者污染。
_RUN_OPTION_KEYS = ("max_rounds", "backend", "host", "port", "user", "key_path",
                    "work_dir", "timeout", "persistent", "target_gflops")


def _call_tool(name: str, arguments: dict) -> dict:
    if name == "operator_capabilities":
        return _text_result(capabilities())
    if name == "operator_repair":
        args = dict(arguments or {})
        run_dir = args.pop("run_dir", None)
        code = args.pop("code", None)
        code_file = args.pop("code_file", None)
        if bool(code) == bool(code_file):
            raise ToolError("code_source_ambiguous", "code 与 code_file 必须给且只给一个")
        if code_file:
            code = Path(code_file).read_text(encoding="utf-8")
        return _text_result(repair(run_dir, code, **args))
    if name == "operator_status":
        return _text_result(autonomous_status(dict(arguments or {}).get("run_dir", "")))
    if name == "operator_run":
        # 工具层参数 ≠ 声明参数：`backend`/`max_rounds`/ssh 那几项属于**运行选项**，
        # 不能混进 declaration（白名单会拒）。2026-10-19 在真宿主里撞到过一次。
        args = dict(arguments or {})
        run_dir = args.pop("run_dir", None)
        if not run_dir:
            raise ToolError("run_dir_required", "operator_run 需要 run_dir")
        options = {k: args.pop(k) for k in _RUN_OPTION_KEYS if k in args}
        return _text_result(run_autonomous(args, run_dir, **options))
    if name == "operator_prepare":
        args = dict(arguments or {})
        run_dir = args.pop("run_dir", None)
        if not run_dir:
            raise ToolError("run_dir_required", "prepare 需要 run_dir")
        return _text_result(prepare(args, run_dir))
    if name == "operator_verify":
        args = dict(arguments or {})
        run_dir = args.pop("run_dir", None)
        code = args.pop("code", None)
        code_file = args.pop("code_file", None)
        if bool(code) == bool(code_file):
            raise ToolError("code_source_ambiguous", "code 与 code_file 必须给且只给一个")
        if code_file:
            code = Path(code_file).read_text(encoding="utf-8")
        return _text_result(verify(run_dir, code, **args))
    raise ToolError("unknown_tool", f"没有这个工具: {name}")


def handle_message(message: dict) -> Optional[dict]:
    """一条请求 → 一条响应；通知返回 None（JSON-RPC 的通知不许有响应）。"""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _jsonrpc_err(None, -32600, "Invalid Request")
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params") or {}

    if msg_id is None and isinstance(method, str):
        # 通知：initialized / cancelled / 别的 —— 一律不响应（也不假装处理了）。
        return None

    if method == "initialize":
        wanted = params.get("protocolVersion")
        version = wanted if wanted in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
        return _jsonrpc_ok(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            # 使用说明跟着 server 走（见 INSTRUCTIONS 的注释）：客户端渲染进上下文，
            # 于是宿主的提示段与工具描述不会各说一套。
            "instructions": INSTRUCTIONS,
        })
    if method == "ping":
        return _jsonrpc_ok(msg_id, {})
    if method == "tools/list":
        return _jsonrpc_ok(msg_id, {"tools": _tool_definitions()})
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            return _jsonrpc_ok(msg_id, _call_tool(str(name), arguments))
        except ToolError as exc:
            # 工具层拒绝：**不是**协议错误 —— 让模型看见原因（isError），
            # 而不是把一个可归因的失败伪装成 server 崩了。
            return _jsonrpc_ok(msg_id, _text_result(exc.as_dict(), is_error=True))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return _jsonrpc_ok(msg_id, _text_result(
                {"ok": False, "error": "tool_failed", "detail": repr(exc)}, is_error=True))
    return _jsonrpc_err(msg_id, -32601, f"Method not found: {method}")


def serve(instream: TextIO, outstream: TextIO, *, log: Optional[TextIO] = None) -> None:
    """逐行读 JSON-RPC，逐行写响应。空行/坏行只记 stderr，不打断流。"""
    log = log or sys.stderr
    for raw in instream:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            log.write(f"[mcp] 坏行（不是 JSON）: {exc}\n")
            continue
        if isinstance(message, list):                      # 批量：逐条处理
            replies = [r for r in (handle_message(m) for m in message) if r is not None]
            if replies:
                outstream.write(json.dumps(replies, ensure_ascii=False) + "\n")
                outstream.flush()
            continue
        reply = handle_message(message)
        if reply is None:
            continue
        outstream.write(json.dumps(reply, ensure_ascii=False) + "\n")
        outstream.flush()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return _selftest()
    # stdout 只许走协议行：所有库的 print 都改道 stderr（与 runner_server 同源）。
    real_out = sys.stdout
    sys.stdout = sys.stderr
    try:
        serve(sys.stdin, real_out)
    finally:
        sys.stdout = real_out
    return 0


def _selftest() -> int:
    """不依赖宿主，自己走一遍握手 —— 给人看「装了它之后会发生什么」。"""
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                    "clientInfo": {"name": "selftest", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "operator_capabilities", "arguments": {}}},
    ]
    instream = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    outstream = io.StringIO()
    serve(instream, outstream)
    lines = [json.loads(l) for l in outstream.getvalue().splitlines() if l.strip()]
    print(f"协议对话: 发出 {len(requests)} 条，收到 {len(lines)} 条响应"
          f"（通知不响应）")
    for reply in lines:
        if "error" in reply:
            print(f"  id={reply['id']} error={reply['error']['message']}")
            continue
        result = reply["result"]
        if "tools" in result:
            print(f"  id={reply['id']} tools={[t['name'] for t in result['tools']]}")
        elif "content" in result:
            body = json.loads(result["content"][0]["text"])
            print(f"  id={reply['id']} capabilities.ok={body.get('ok')}")
        else:
            print(f"  id={reply['id']} protocolVersion={result.get('protocolVersion')}"
                  f" server={result.get('serverInfo')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
