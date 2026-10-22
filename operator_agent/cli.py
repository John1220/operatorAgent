"""命令行壳：`python -m operator_agent prepare|verify|capabilities`

为什么要有 CLI（`docs/dsh-plugin-evaluation.md` §6.2 Phase 0）：

外层（DSH 插件 / MCP server / 任何脚本）需要一个**稳定的进程契约**，而不是 import 本仓
的内部对象。这个入口只保证三件事：

1. **stdout 只有一行 JSON**（判定层的进度输出被 `hitl_tools` 收进 run 目录）；
2. **退出码**：0 = 判完了（哪怕候选没过），2 = 被拒绝（可归因的 `error`），1 = 意外；
3. **不猜**：拒绝时给 `error` + `detail`，不给「看起来像成功」的空壳。

用法：

    python -m operator_agent capabilities
    python -m operator_agent prepare --request "把 64x64x64 的 fp32 矩阵乘法写成算子" \\
        --shapes 64,64,64 --run-dir /tmp/op-run
    python -m operator_agent verify --run-dir /tmp/op-run --code-file cand.py --stop-after L2

与 `benchmarks/demo_hitl_chain.py` 的关系：那个演示给人看；这个 CLI 给程序调。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .hitl_tools import (ToolError, autonomous_status, capabilities, prepare,
                         repair, run_autonomous, verify)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="operator_agent", allow_abbrev=False,
                                description="算子开发判定层的一次一回合 CLI")
    sub = p.add_subparsers(dest="action", required=True)

    cap = sub.add_parser("capabilities", help="不跑活：这台机器上这个壳能做什么")
    cap.add_argument("--json-out", help="同时写一份 JSON 到文件")

    pre = sub.add_parser("prepare", help="把一份声明冻结成可复算的回合")
    pre.add_argument("--request", help="自然语言请求")
    pre.add_argument("--declaration", help="声明 JSON 文件（与上面的参数二选一/覆盖）")
    pre.add_argument("--shapes", action="append", default=[],
                     help="形状，可重复；如 --shapes 64,64,64")
    pre.add_argument("--hardware", default=None)
    pre.add_argument("--dsl", default=None)
    pre.add_argument("--precision", default=None)
    pre.add_argument("--min-speedup", type=float, default=None)
    pre.add_argument("--require-cap", action="append", default=[],
                     help="执行侧必须具备的能力，可重复（如 real_device）")
    pre.add_argument("--framework", default=None)
    pre.add_argument("--run-dir", required=True, help="回合目录（调用方拥有布局）")
    pre.add_argument("--json-out", help="同时写一份 JSON 到文件")

    ver = sub.add_parser("verify", help="对一份候选源码做判定（不生成、不修复、不搜索）")
    ver.add_argument("--run-dir", required=True)
    ver.add_argument("--code-file", help="候选源码文件")
    ver.add_argument("--code", help="候选源码（短片段用）")
    ver.add_argument("--code-stdin", action="store_true", help="从 stdin 读候选源码")
    ver.add_argument("--stop-after", choices=["L1", "L2", "L3"], default=None)
    ver.add_argument("--backend", default="mock", choices=["mock", "local", "ssh"])
    ver.add_argument("--host", help="ssh 后端的主机（~/.ssh/config 别名即可）")
    ver.add_argument("--port", type=int, default=None)
    ver.add_argument("--user", default=None)
    ver.add_argument("--key-path", default=None)
    ver.add_argument("--work-dir", default=None)
    ver.add_argument("--timeout", type=int, default=300)
    ver.add_argument("--persistent", action="store_true",
                     help="SSH 常驻会话（doc 62）；默认关")
    ver.add_argument("--target-gflops", type=float, default=0.0)
    ver.add_argument("--json-out", help="同时写一份 JSON 到文件")

    run = sub.add_parser("run", help="起一次无人值守的整段运行（Conductor 自己设计/写核）")
    run.add_argument("--request", help="自然语言请求")
    run.add_argument("--declaration", help="声明 JSON 文件")
    run.add_argument("--shapes", action="append", default=[])
    run.add_argument("--hardware", default=None)
    run.add_argument("--dsl", default=None)
    run.add_argument("--precision", default=None)
    run.add_argument("--min-speedup", type=float, default=None)
    run.add_argument("--require-cap", action="append", default=[])
    run.add_argument("--framework", default=None)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--max-rounds", type=int, default=1)
    run.add_argument("--backend", default="mock", choices=["mock", "local", "ssh"])
    run.add_argument("--host", default=None)
    run.add_argument("--port", type=int, default=None)
    run.add_argument("--user", default=None)
    run.add_argument("--key-path", default=None)
    run.add_argument("--work-dir", default=None)
    run.add_argument("--timeout", type=int, default=300)
    run.add_argument("--persistent", action="store_true")
    run.add_argument("--target-gflops", type=float, default=0.0)
    run.add_argument("--json-out", help="同时写一份 JSON 到文件")

    rep = sub.add_parser("repair", help="给一份已有的核：判→按处方修→再判（最多 N 轮）")
    rep.add_argument("--run-dir", required=True)
    rep.add_argument("--code-file", help="候选源码文件")
    rep.add_argument("--code", help="候选源码（短片段用）")
    rep.add_argument("--max-rounds", type=int, default=2)
    rep.add_argument("--stop-after", choices=["L1", "L2", "L3"], default=None)
    rep.add_argument("--backend", default="mock", choices=["mock", "local", "ssh"])
    rep.add_argument("--host", default=None)
    rep.add_argument("--port", type=int, default=None)
    rep.add_argument("--user", default=None)
    rep.add_argument("--key-path", default=None)
    rep.add_argument("--work-dir", default=None)
    rep.add_argument("--timeout", type=int, default=300)
    rep.add_argument("--persistent", action="store_true")
    rep.add_argument("--target-gflops", type=float, default=0.0)
    rep.add_argument("--json-out", help="同时写一份 JSON 到文件")

    st = sub.add_parser("status", help="查一次无人值守运行的相位（running/done/failed/lost）")
    st.add_argument("--run-dir", required=True)
    st.add_argument("--json-out", help="同时写一份 JSON 到文件")
    return p


def _declaration_from_args(args) -> dict:
    decl: dict = {}
    if args.declaration:
        decl.update(json.loads(Path(args.declaration).read_text(encoding="utf-8")))
    if args.request is not None:
        decl["request"] = args.request
    if args.shapes:
        shapes = []
        for item in args.shapes:
            parts = [p for p in item.replace("x", ",").split(",") if p.strip()]
            shapes.append([int(p) for p in parts])
        decl["shapes"] = shapes
    for key, value in (("hardware", args.hardware), ("dsl", args.dsl),
                       ("precision", args.precision),
                       ("min_speedup", args.min_speedup),
                       ("framework", args.framework)):
        if value is not None:
            decl[key] = value
    if args.require_cap:
        decl["required_caps"] = list(args.require_cap)
    return decl


def _read_code(args) -> str:
    sources = [bool(args.code_file), bool(args.code), bool(args.code_stdin)]
    if sum(sources) != 1:
        raise ToolError("code_source_ambiguous",
                        "code-file / code / code-stdin 里必须且只能给一个")
    if args.code_file:
        return Path(args.code_file).read_text(encoding="utf-8")
    if args.code:
        return args.code
    return sys.stdin.read()


def _emit(payload: dict, json_out: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if json_out:
        Path(json_out).write_text(text + "\n", encoding="utf-8")
    print(text)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "capabilities":
            payload = capabilities()
        elif args.action == "prepare":
            payload = prepare(_declaration_from_args(args), args.run_dir)
        elif args.action == "run":
            payload = run_autonomous(
                _declaration_from_args(args), args.run_dir,
                max_rounds=args.max_rounds, backend=args.backend, host=args.host,
                port=args.port, user=args.user, key_path=args.key_path,
                work_dir=args.work_dir, timeout=args.timeout,
                persistent=args.persistent, target_gflops=args.target_gflops)
        elif args.action == "repair":
            payload = repair(args.run_dir, _read_code(args),
                             max_rounds=args.max_rounds, stop_after=args.stop_after,
                             backend=args.backend, host=args.host, port=args.port,
                             user=args.user, key_path=args.key_path,
                             work_dir=args.work_dir, timeout=args.timeout,
                             persistent=args.persistent,
                             target_gflops=args.target_gflops)
        elif args.action == "status":
            payload = autonomous_status(args.run_dir)
        else:
            payload = verify(args.run_dir, _read_code(args),
                             stop_after=args.stop_after, backend=args.backend,
                             target_gflops=args.target_gflops,
                             host=args.host, port=args.port, user=args.user,
                             key_path=args.key_path, work_dir=args.work_dir,
                             timeout=args.timeout, persistent=args.persistent)
    except ToolError as exc:
        _emit(exc.as_dict(), getattr(args, "json_out", None))
        return 2
    except (OSError, json.JSONDecodeError) as exc:
        _emit({"ok": False, "error": "io_error", "detail": repr(exc)},
              getattr(args, "json_out", None))
        return 2
    _emit(payload, getattr(args, "json_out", None))
    # 判完了就是 0 —— 候选没过不是 CLI 的失败（那在 payload 里）。
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
