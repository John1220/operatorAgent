"""命令行壳：`python -m operator_agent prepare|verify|capabilities`

为什么要有 CLI（`docs/dsh-plugin-evaluation.md` §6.2 Phase 0）：

外层（DSH 插件 / MCP server / 任何脚本）需要一个**稳定的进程契约**，而不是 import 本仓
的内部对象。这个入口只保证三件事：

1. **stdout 只有一行 JSON**（判定层的进度输出被 `hitl_tools` 收进 run 目录）；
2. **退出码**：0 = 判完了（哪怕候选没过），2 = 被拒绝（可归因的 `error`），1 = 意外；
3. **不猜**：拒绝时给 `error` + `detail`，不给「看起来像成功」的空壳。

用法：

    python -m operator_agent capabilities
    op_workspace=$(mktemp -d)
    python -m operator_agent prepare --request "把 64x64x64 的 fp32 矩阵乘法写成算子" \\
        --shapes 64,64,64 --run-dir "$op_workspace/run"
    python -m operator_agent verify --run-dir "$op_workspace/run" --code-file cand.py --stop-after L2
    # 六环节人时表（阶段 A「人在哪些环节出现」的当场读数，与机器秒分列）
    python -m operator_agent ledger-new  --run-dir "$op_workspace/run"
    python -m operator_agent ledger-fill --run-dir "$op_workspace/run" \\
        --stage analyze --involved yes --action-taken "读 L3 回执并判是否值得再跑" \\
        --reason "门槛在 ±3% 刀刃上" --minutes-band 5-15
    python -m operator_agent ledger-report --run-dir "$op_workspace/run"

子命令：capabilities / prepare / verify / run / repair / status / cancel / ledger-{new,fill,report}。

与 `benchmarks/demos/demo_hitl_chain.py` 的关系：那个演示给人看；这个 CLI 给程序调。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..app.hitl_tools import (CANDIDATE_ORIGINS, ToolError, autonomous_status, cancel_task, capabilities,
                         prepare, run_autonomous)
from ..core import human_ledger
from ..core.human_ledger import LedgerError


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
    ver.add_argument("--candidate-origin", choices=list(CANDIDATE_ORIGINS), default=None,
                     help="候选来源（归因，不进判据）；不给则回包记 undeclared")
    ver.add_argument("--stop-after", choices=["L1", "L2", "L3"], default=None)
    ver.add_argument("--backend", default="mock", choices=["mock", "local", "ssh", "fake"])
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
    run.add_argument("--wall-clock-s", type=float, default=1800.0,
                     help="任务总墙钟时限（秒）")
    run.add_argument("--max-llm-calls", type=int, default=24,
                     help="任务内模型调用总额度，失败调用同样计数")
    run.add_argument("--oracle-total", type=int, default=20,
                     help="验证 case 派发额度，最终确认预留不可挪用")
    run.add_argument("--backend", default="mock", choices=["mock", "local", "ssh", "fake"])
    from ..execute.fake_backend.config import SCENARIOS, PROFILES
    run.add_argument("--fake-scenario", choices=SCENARIOS,
                     help="backend=fake 的故障场景；默认 success")
    run.add_argument("--fake-profile", choices=PROFILES,
                     help="声明式测试硬件规格；默认 small")
    run.add_argument("--llm-mode", choices=["fake", "live"],
                     help="仅 backend=fake：默认 fake，不读凭据、不发模型请求")
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
    # `_read_code` 对三个来源都做 `bool(...)`，少这一个就是 AttributeError（不是「少个选项」）：
    # 没有它，CLI 的 repair 路径一次都跑不起来。
    rep.add_argument("--code-stdin", action="store_true", help="从 stdin 读候选源码")
    rep.add_argument("--candidate-origin", choices=list(CANDIDATE_ORIGINS), default=None,
                     help="**传进来那份**的来源（归因，不进判据）；不给则记 undeclared")
    rep.add_argument("--max-rounds", type=int, default=2)
    rep.add_argument("--wall-clock-s", type=float, default=1800.0,
                     help="修复任务总墙钟时限（秒）")
    rep.add_argument("--max-llm-calls", type=int, default=24,
                     help="修复任务模型调用总额度，失败不退款")
    rep.add_argument("--oracle-total", type=int, default=20,
                     help="验证输入组派发额度，确认预留不可挪用")
    rep.add_argument("--stop-after", choices=["L1", "L2", "L3"], default=None)
    rep.add_argument("--backend", default="mock", choices=["mock", "local", "ssh", "fake"])
    rep.add_argument("--host", default=None)
    rep.add_argument("--port", type=int, default=None)
    rep.add_argument("--user", default=None)
    rep.add_argument("--key-path", default=None)
    rep.add_argument("--work-dir", default=None)
    rep.add_argument("--timeout", type=int, default=300)
    rep.add_argument("--persistent", action="store_true")
    rep.add_argument("--target-gflops", type=float, default=0.0)
    rep.add_argument("--json-out", help="同时写一份 JSON 到文件")

    st = sub.add_parser("status", help="查任务/工具调用状态（含 cancelled/lost）")
    st.add_argument("--run-dir", required=True)
    st.add_argument("--invocation-id")
    st.add_argument("--json-out", help="同时写一份 JSON 到文件")
    cancel = sub.add_parser('cancel', help='按 run/invocation 请求取消，再用 status 确认终态')
    cancel.add_argument('--run-dir', required=True)
    cancel.add_argument('--invocation-id', required=True)
    cancel.add_argument('--json-out')

    led_new = sub.add_parser('ledger-new', help='在一个 run 目录里建一张六环节人时表（全部 unknown）')
    led_new.add_argument('--run-dir', required=True)
    led_new.add_argument('--json-out')
    led_fill = sub.add_parser('ledger-fill', help='填一格：人是否介入 / 做了什么 / 为什么 / 量级档')
    led_fill.add_argument('--run-dir', required=True)
    led_fill.add_argument('--stage', required=True,
                          help='| '.join(human_ledger.STAGES))
    led_fill.add_argument('--involved', choices=list(human_ledger.INVOLVED_VALUES),
                          help='不传表示「没问过」——保持 unknown，不等于 no')
    led_fill.add_argument('--action-taken', dest='action_taken', help='人具体做了什么')
    led_fill.add_argument('--reason', help='为什么必须人出手（干预原因单独记）')
    led_fill.add_argument('--minutes-band', dest='minutes_band',
                         choices=list(human_ledger.MINUTE_BANDS),
                         help='自报量级档，只准填档不准填分钟')
    led_fill.add_argument('--json-out')
    led_rep = sub.add_parser('ledger-report', help='打印人自报档与机器秒两列（不合算，不折叠）')
    led_rep.add_argument('--run-dir', required=True)
    led_rep.add_argument('--machine-wall-json', dest='machine_wall_json',
                         help='按环节给的机器秒数 JSON；缺则整列留空而不是补 0')
    led_rep.add_argument('--json-out')
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
        return Path(args.code_file).read_bytes().decode("utf-8")
    if args.code:
        return args.code
    return sys.stdin.read()


def _emit(payload: dict, json_out: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if json_out:
        Path(json_out).write_text(text + "\n", encoding="utf-8")
    print(text)


def _ledger(args) -> dict:
    """六环节人时表的新建 / 填格 / 打印。表在 run 目录里，路径知识只有一处。"""
    path = Path(args.run_dir) / human_ledger.LEDGER_FILENAME
    if args.action == "ledger-new":
        if path.exists():
            raise ToolError("ledger_exists",
                            "%s 已有表：请编辑它而不是覆盖——已答的格子一旦被重写，"
                            "「问过并答了」就退化成「没问过」" % path)
        ledger = human_ledger.new_ledger(Path(args.run_dir).name)
        human_ledger.write_ledger(path, ledger)
        return dict(ok=True, ledger=str(path), **human_ledger.counts(ledger))
    if args.action == "ledger-fill":
        ledger = human_ledger.read_ledger(path)
        human_ledger.record(ledger, args.stage, involved=args.involved,
                            action=args.action_taken, reason=args.reason,
                            minutes_band=args.minutes_band)
        human_ledger.write_ledger(path, ledger)
        return dict(ok=True, ledger=str(path), stage=args.stage,
                    entry=ledger["entries"][args.stage], **human_ledger.counts(ledger))
    ledger = human_ledger.read_ledger(path)
    machine = None
    if args.machine_wall_json:
        machine = json.loads(Path(args.machine_wall_json).read_text(encoding="utf-8"))
    return dict(ok=True, ledger=str(path), counts=human_ledger.counts(ledger),
                rows=human_ledger.as_report_rows(ledger, machine),
                columns=("human_minutes_band", "machine_wall_s"),
                note="两列分开印：本命令不提供把量级档与秒合算的入口")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "capabilities":
            payload = capabilities()
        elif args.action == "prepare":
            payload = prepare(_declaration_from_args(args), args.run_dir)
        elif args.action == "run":
            fake = {key: value for key, value in (
                ('scenario', args.fake_scenario), ('profile', args.fake_profile),
                ('llm', args.llm_mode)) if value is not None}
            payload = run_autonomous(
                _declaration_from_args(args), args.run_dir,
                max_rounds=args.max_rounds, backend=args.backend, host=args.host,
                port=args.port, user=args.user, key_path=args.key_path,
                work_dir=args.work_dir, timeout=args.timeout,
                persistent=args.persistent, target_gflops=args.target_gflops,
                wall_clock_s=args.wall_clock_s, max_llm_calls=args.max_llm_calls,
                oracle_total=args.oracle_total, fake=fake or None)
        elif args.action == "repair":
            from ..app.tool_calls import run_tool_call
            payload = run_tool_call('operator_repair', dict(run_dir=args.run_dir, code=_read_code(args),
                             max_rounds=args.max_rounds, stop_after=args.stop_after,
                             backend=args.backend, host=args.host, port=args.port,
                             user=args.user, key_path=args.key_path,
                             work_dir=args.work_dir, timeout=args.timeout,
                             persistent=args.persistent,
                             target_gflops=args.target_gflops,
                             wall_clock_s=args.wall_clock_s,
                             max_llm_calls=args.max_llm_calls,
                             oracle_total=args.oracle_total,
                             candidate_origin=args.candidate_origin))
        elif args.action == "status":
            payload = autonomous_status(args.run_dir, args.invocation_id)
        elif args.action == 'cancel':
            payload = cancel_task(args.run_dir, args.invocation_id)
        elif args.action in ("ledger-new", "ledger-fill", "ledger-report"):
            payload = _ledger(args)
        else:
            from ..app.tool_calls import run_tool_call
            payload = run_tool_call('operator_verify', dict(run_dir=args.run_dir, code=_read_code(args),
                             stop_after=args.stop_after, backend=args.backend,
                             target_gflops=args.target_gflops,
                             host=args.host, port=args.port, user=args.user,
                             key_path=args.key_path, work_dir=args.work_dir,
                             timeout=args.timeout, persistent=args.persistent,
                             candidate_origin=args.candidate_origin))
    except ToolError as exc:
        _emit(exc.as_dict(), getattr(args, "json_out", None))
        return 2
    except LedgerError as exc:
        # 人时表的拒绝要能自我纠正：消息里带合法取值，调用方不用猜
        _emit({"ok": False, "error": "ledger_refused", "detail": str(exc)},
              getattr(args, "json_out", None))
        return 2
    except (OSError, json.JSONDecodeError) as exc:
        _emit({"ok": False, "error": "io_error", "detail": repr(exc)},
              getattr(args, "json_out", None))
        return 2
    _emit(payload, getattr(args, "json_out", None))
    if payload.get('phase') == 'cancelled':
        return 130
    # 判完了就是 0 —— 候选没过不是 CLI 的失败（那在 payload 里）。
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
