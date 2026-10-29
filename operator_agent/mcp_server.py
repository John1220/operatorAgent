"""最小 MCP stdio server：把 `hitl_tools` 的八个工具暴露给宿主的 agent

为什么是这个形态（`docs/dsh-plugin-evaluation.md` §6.2 / §9.6）：

DSH 的插件必须是 Cordis 的 ESM 模块，而本仓是 Python —— 两者之间**官方支持**的接缝只有
两个：`ctx.subprocess` 与 MCP。MCP 这条不需要写任何 JS：宿主侧加一条
`@deepseek-ai/dsh-mcp-client` 配置行，工具就以 `mcp__<serverName>__<tool>` 出现。

**没有引第三方 SDK**：MCP 的 stdio 传输就是「一行一个 JSON-RPC 2.0 消息」，用标准库足够，
与本仓「控制面零第三方依赖」的立场一致（也避免把 `mcp` 包变成新的部署前提）。

协议面（只实现会被用到的哪些，其余显式报错，不假装支持）：

    initialize                 → 版本协商 + capabilities.tools
    notifications/initialized  → 无响应（通知没有 id）
    notifications/cancelled    → 请求取消对应的前台 worker；通知不响应
    ping                       → {}
    tools/list                 → 八个工具的 JSON Schema（静态表：改了要重启进程）
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
import queue
import signal
import threading
from pathlib import Path
from typing import Any, BinaryIO, Optional, TextIO

from . import __version__
from .hitl_tools import (CANDIDATE_ORIGINS, ToolError, autonomous_status, cancel_task, bench,
                         capabilities, prepare, repair, run_autonomous, verify)

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

`operator_run` 和 `operator_repair` 的任务预算由 `wall_clock_s`、`max_llm_calls`、`oracle_total` 指定；后者按验证
case 派发预留计数，最终确认预留不可挪用，不是 GPU 秒数。`operator_status` 的 `budget` 与
`stopped_reason` 说明用量和停止原因；`done` 包括预算停止，仍需查看验证结果。

取消使用 `operator_cancel(run_dir, invocation_id)`，再查 `operator_status` 确认 cancelled
或其他终态。后台 run 立刻返回 invocation_id；同步 verify/repair/bench 可用 MCP 取消通知，
也可从 operator_status(run_dir) 列出的工具调用取得实例 ID。accepted 只是请求已记录；
本机清理不证明远端 GPU 或服务商已停止。已发生的调用/测量不退款。

**写核优先交给内层，别自己从头写**：从零整段用 `operator_run` —— 当前入口使用普通 Coder，
带共享规约、`triton_skill`、优化配方和预检重试，再由控制器验证/修复；三温度采样等 Pipeline
能力需要显式启用，不是该工具默认策略。已有一份核要改
用 `operator_repair` —— 内层 Fixer 带共享 `GENERATION_SPEC`、coder 角色的 `triton_skill`
注入与绑定当前源码的诊断；该路径不做规划、多候选采样或参数搜索（三条内层路径的能力并不相同）。
理由是这些注入你在对话里一个都拿不到，而内层那条路有真机十任务的实测通过率，你自己写这条
没有。**两条例外**：① **人写的核照常走 `operator_verify`** —— 那是设计过的用法，不要劝人
别写，这条只针对「你自己动手写」；② 目前**没有**「只生成一个候选」的工具，所以如果你要的
是单个候选、又不值得起一次整段 `operator_run`，那就自己写，并在 `candidate_origin` 里
**如实**填 `outer`。

要**扫参数或对比多个候选**（tile / warps / stages 这类网格，或同一任务的多份源码）用
`operator_bench`：一次调用测一批，逐臂给 speedup 与 validity。它的回包 `grade=exploration`
**没有**裁决字段 —— 那是设计，不是缺字段：读数只能用来选方向，别拿它当交付结论，
survivors 仍要走 `operator_verify`。也别再手搓 ssh 基准脚本。

读回包时：`levels` 看判到哪一级；**`route` 与 `eligibility` 必须一起读** —— 两者冲突时
`conflict.action=hold_for_measurement`，此时不要照着 `route` 迭代，先修判据（形状 / 精度 /
required_caps）。`validity=degraded` 或 `uninformative` 的读数不驱动调参或入库。
修复回包的 `full_l2_passed=true` 才表示完整正确性套件通过；单独 L1 通过不是正确性。
`diagnosis` 是绑定当前源码与失败观察的假设，不是通过凭据；未知时不猜原因。
原样返回或只改注释的修复会停止，完整诊断和源码改动记录在 `evidence` 文件中。

`candidate_origin`（`verify`/`repair` 的可选参数）是**归因**格，不是判据：你写的或人写的核
填 `outer`，省略则回包与 `report.md` 记 `undeclared`（缺省是「记成没声明」，不是「不记」）。
它不影响任何通过与否。`repair` 另回一个**派生**的 `final_candidate_origin` —— 只要有哪一轮
真改了源码就是 `inner_fixer`，那一格不接受声明。
"""

_SHAPES = {"type": "array", "minItems": 1,
           "items": {"type": "array", "minItems": 1,
                     "items": {"type": "integer", "minimum": 0}},
           "description": "形状分布：matmul 是 [M,N,K]，逐元素/归一化是 [M,N]"}

#: 候选来源。**闭合词表只有一处来源**（`hitl_tools.CANDIDATE_ORIGINS`），schema 与实现
#: 各写一份就是「工具说收 A、实现只认 B」。省略 ⇒ 回包记 `undeclared`，不是省略这一格。
_ORIGIN = {"type": "string", "enum": list(CANDIDATE_ORIGINS),
           "description": ("这份源码是谁写的（归因用，**不进判据**）：outer = 你/人写的，"
                           "inner_fixer = operator_repair 的内层模型改的，"
                           "inner_coder = 本仓生成器产的。省略则回包记 undeclared。"
                           "**如实填**：这一格是用来事后区分「裸写」与「走过 spec/技能注入」的，"
                           "不是为了好看 —— 自己写的核填 outer，不要填 inner_coder。")}


def _tool_definitions() -> list[dict]:
    definitions = [
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
                    "candidate_origin": _ORIGIN,
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
            "name": "operator_bench",
            "description": ("对**多个**候选只跑 L3 的批量测量（探索级，**不可裁决**）：一次部署、"
                            "逐臂与基线同窗配对，只回答「哪个方向更快」。已有 verdict 字段一律不出，"
                            "要交付结论仍须 operator_verify。调参/扫网格用它，别再手搓 ssh 基准。"),
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "required": ["run_dir", "candidates"],
                "properties": {
                    "run_dir": {"type": "string", "description": "已 prepare 的回合目录（复用其契约）"},
                    "candidates": {
                        "type": "array", "minItems": 1,
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["name"],
                            "anyOf": [{"required": ["code"]}, {"required": ["code_file"]}],
                            "properties": {
                                "name": {"type": "string"},
                                "code": {"type": "string"},
                                "code_file": {"type": "string"},
                            },
                        },
                    },
                    "backend": {"type": "string", "enum": ["mock", "local", "ssh"], "default": "mock"},
                    "host": {"type": "string"},
                    "port": {"type": "integer"},
                    "user": {"type": "string"},
                    "key_path": {"type": "string"},
                    "work_dir": {"type": "string"},
                    "timeout": {"type": "integer", "default": 300},
                    "persistent": {"type": "boolean", "default": True},
                    "max_candidates": {"type": "integer", "default": 16},
                    "deadline_s": {"type": "number", "default": 120},
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
                    "wall_clock_s": {"type": "number", "exclusiveMinimum": 0,
                                     "default": 1800.0, "description": "整个任务的墙钟时限（秒）"},
                    "max_llm_calls": {"type": "integer", "minimum": 1,
                                      "default": 24, "description": "整个任务允许派发的模型调用总数，失败不退款"},
                    "oracle_total": {"type": "integer", "minimum": 1,
                                     "default": 20, "description": "验证 case 派发额度，确认预留不可挪用；不表示原始 GPU launch 次数或 GPU 秒数"},
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
            "name": "operator_cancel",
            "description": ("请求取消指定 run/invocation。只写请求，由拥有任务的监督进程清理；"
                            "accepted 不是终态确认，随后查 operator_status。已结束/重复请求幂等。"),
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "required": ["run_dir", "invocation_id"],
                "properties": {"run_dir": {"type": "string"},
                    "invocation_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"}},
            },
        },
        {
            "name": "operator_status",
            "description": ("查自主任务或指定 invocation 的 running/done/failed/cancelled/lost。"
                            "仅前台工具目录且未指定实例时列出 tool_calls；lost 的清理状态未知。"),
            "inputSchema": {
                "type": "object", "additionalProperties": False,
                "required": ["run_dir"],
                "properties": {"run_dir": {"type": "string"},
                    "invocation_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"}},
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
                    "candidate_origin": _ORIGIN,
                    "max_rounds": {"type": "integer", "default": 2},
                    "wall_clock_s": {"type": "number", "exclusiveMinimum": 0,
                                     "default": 1800.0,
                                     "description": "修复任务总墙钟时限（秒）"},
                    "max_llm_calls": {"type": "integer", "minimum": 1, "default": 24,
                                      "description": "修复任务模型调用预留，失败不退款"},
                    "oracle_total": {"type": "integer", "minimum": 1, "default": 20,
                                     "description": "验证输入组派发额度，确认预留不可挪用"},
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
    from .s5_verify_client import enabled
    if enabled():
        return [tool for tool in definitions if tool['name'] in
                ('operator_prepare', 'operator_verify', 'operator_capabilities')]
    return definitions


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
                    "work_dir", "timeout", "persistent", "target_gflops",
                    "wall_clock_s", "max_llm_calls", "oracle_total")


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
            code = Path(code_file).read_bytes().decode("utf-8")
        return _text_result(repair(run_dir, code, **args))
    if name == "operator_status":
        return _text_result(autonomous_status(**dict(arguments or {})))
    if name == "operator_cancel":
        args = dict(arguments or {})
        if not args.get('run_dir') or not args.get('invocation_id'):
            raise ToolError('cancellation_refused', 'run_dir and invocation_id are required')
        return _text_result(cancel_task(**args))
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
            code = Path(code_file).read_bytes().decode("utf-8")
        return _text_result(verify(run_dir, code, **args))
    if name == "operator_bench":
        args = dict(arguments or {})
        run_dir = args.pop("run_dir", None)
        if not run_dir:
            raise ToolError("run_dir_required", "operator_bench 需要 run_dir")
        candidates = args.pop("candidates", None)
        return _text_result(bench(run_dir, candidates, **args))
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
        from .s5_verify_client import enabled
        wanted = params.get("protocolVersion")
        version = wanted if wanted in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
        return _jsonrpc_ok(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            # 使用说明跟着 server 走（见 INSTRUCTIONS 的注释）：客户端渲染进上下文，
            # 于是宿主的提示段与工具描述不会各说一套。
            "instructions": ("S5 supervised development experiment: prepare then verify with "
                             "stop_after=L2 and the supplied backend. Verification is delegated "
                             "to the common service; read status and feedback, revise source in "
                             "the outer agent. No repair/run/bench or direct GPU execution. "
                             "The final independent evaluation is not a development tool."
                             if enabled() else INSTRUCTIONS),
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
    """Read controls while foreground workers run; only this thread emits JSON."""
    from .tool_calls import LONG_TOOLS, ToolCall
    log = log or sys.stderr
    inbox = queue.Queue()
    active = {}
    ended = False
    interrupted = False

    def read_lines():
        try:
            for raw in instream:
                inbox.put(raw)
        finally:
            inbox.put(None)

    def emit(reply):
        if reply is None:
            return
        outstream.write(json.dumps(reply, ensure_ascii=False) + '\n')
        outstream.flush()

    def key(value):
        return (type(value).__name__, json.dumps(value, sort_keys=True))

    def submit(message, done):
        if not isinstance(message, dict):
            done(handle_message(message))
            return
        params = message.get('params') or {}
        method, msg_id = message.get('method'), message.get('id')
        if (message.get('jsonrpc') == '2.0' and method == 'notifications/cancelled'
                and msg_id is None):
            job = active.get(key(params.get('requestId')))
            if job:
                job[0].cancel()
            done(None)
            return
        if (message.get('jsonrpc') == '2.0' and msg_id is not None
                and method == 'tools/call' and params.get('name') in LONG_TOOLS):
            identity = key(msg_id)
            if identity in active:
                done(_jsonrpc_err(msg_id, -32600, 'Request ID already active'))
                return
            try:
                active[identity] = (ToolCall(params['name'], params.get('arguments') or {}),
                                    msg_id, done)
            except (ToolError, OSError, ValueError) as exc:
                body = exc.as_dict() if isinstance(exc, ToolError) else dict(
                    ok=False, error='tool_failed', detail=repr(exc))
                done(_jsonrpc_ok(msg_id, _text_result(body, is_error=True)))
            return
        done(handle_message(message))

    def dispatch(message):
        if not isinstance(message, list):
            submit(message, emit)
            return
        if not message:
            emit(_jsonrpc_err(None, -32600, 'Invalid Request'))
            return
        batch = dict(remaining=len(message), replies={})
        def collect(index, reply):
            if reply is not None:
                batch['replies'][index] = reply
            batch['remaining'] -= 1
            if batch['remaining'] == 0 and batch['replies']:
                emit([batch['replies'][i] for i in sorted(batch['replies'])])
        for index, entry in enumerate(message):
            submit(entry, lambda reply, index=index: collect(index, reply))

    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
    previous = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, on_signal)
    except ValueError:
        pass
    threading.Thread(target=read_lines, daemon=True).start()
    try:
        while (not ended or active) and not interrupted:
            try:
                raw = inbox.get(timeout=.05)
                if raw is None:
                    ended = True
                elif raw.strip():
                    try:
                        dispatch(json.loads(raw))
                    except json.JSONDecodeError as exc:
                        log.write(f'[mcp] 坏行（不是 JSON）: {exc}\n')
            except queue.Empty:
                pass
            for identity, (call, msg_id, done) in list(active.items()):
                response = call.poll()
                if response is not None:
                    del active[identity]
                    done(_jsonrpc_ok(msg_id, response))
    finally:
        try:
            for call, _, _ in active.values():
                call.close()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return interrupted


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return _selftest()
    # stdout 只许走协议行：所有库的 print 都改道 stderr（与 runner_server 同源）。
    real_out = sys.stdout
    sys.stdout = sys.stderr
    try:
        interrupted = serve(sys.stdin, real_out)
    finally:
        sys.stdout = real_out
    return 130 if interrupted else 0


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
