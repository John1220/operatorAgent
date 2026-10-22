"""无人值守整段运行（Phase 3 的后端执行体）

由 `hitl_tools.run_autonomous()` 以**独立进程**拉起：一拍启动、之后用
`hitl_tools.autonomous_status()` 查进度。为什么是独立进程而不是在 MCP server 里同步跑：

* 一次 `Conductor.execute` 是分钟~小时级，宿主侧的工具调用超时按「一次请求」算；
* 崩了要能看出「是这一轮失败」还是「进程没了」—— 两件事分开报；
* 宿主重启不该杀掉已经跑起来的实验。

它只做三件事：**读 `declaration.json` → 跑 `Conductor` → 把结论写进 `run-status.json`**。
判据一行都不在这里（契约冻结、L1/L2/L3、资格、入库门全在既有模块里），这一层只负责
「把一次无人值守运行装进一个目录」。

身份与状态：
    <run>/autonomous.json   启动参数（backend / max_rounds / target_gflops）
    <run>/run-status.json   相位机：running → done | failed（含 pid 与错误原文）
    <run>/conductor.json    完成时的摘要（artifact / 停因 / 最好读数 / 报告路径）
    <run>/autonomous.log    子进程的 stdout+stderr
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

STATUS = "run-status.json"
SUMMARY = "conductor.json"
CONFIG = "autonomous.json"


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                              default=str), encoding="utf-8")
    os.replace(tmp, path)


def _backend(name: str, cfg: dict):
    from .hitl_tools import _backend as make
    return make(name,
                host=cfg.get("host"), port=cfg.get("port"), user=cfg.get("user"),
                key_path=cfg.get("key_path"), work_dir=cfg.get("work_dir"),
                timeout=int(cfg.get("timeout") or 300),
                persistent=bool(cfg.get("persistent")))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m operator_agent.autonomous <run_dir>", file=sys.stderr)
        return 2
    run = Path(argv[0]).resolve()
    status_path = run / STATUS
    started = time.time()
    cfg = json.loads((run / CONFIG).read_text(encoding="utf-8"))
    declaration = json.loads((run / "declaration.json").read_text(encoding="utf-8"))

    status = {"phase": "running", "pid": os.getpid(), "started_at": started,
              "backend": cfg.get("backend"), "max_rounds": cfg.get("max_rounds"),
              "run_dir": str(run),
              # 语料落点也收在 run 目录：这一版的语料是**运行内**的，不写入仓库那份。
              "corpus": {"kernel_db": "run-local", "failure_db": "run-local"}}
    _atomic_json(status_path, status)

    # 落点收进 run 目录：无人值守的运行不该把产物撒到仓库或用户 home 的默认位置。
    os.environ["OPERATOR_AGENT_OUTPUT_DIR"] = str(run / "output")
    os.environ["OPERATOR_AGENT_LOG_DIR"] = str(run / "log")
    # ⚠️ 这两个**不能写 "off"**：2026-10-08（§3.5）之后普通环境变量只能**重定向**，
    # 关不掉持久化 —— `off` 会被解析回 `<repo>/output/*.jsonl`，于是子进程会读写仓库
    # 里那份语料。2026-10-18 实测就撞上了（日志里 `FailureDB 重放 95` 来自仓库）。
    os.environ["OPERATOR_AGENT_KERNEL_DB"] = str(run / "kernel_db.jsonl")
    os.environ["OPERATOR_AGENT_FAILURE_DB"] = str(run / "failure_db.jsonl")
    # 密钥只认**文件引用**或环境变量，且从不回显/落盘。
    key_file = os.environ.get("LLM_API_KEY_FILE")
    if key_file and not os.environ.get("LLM_API_KEY"):
        try:
            os.environ["LLM_API_KEY"] = Path(key_file).expanduser().read_text(
                encoding="utf-8").strip()
        except OSError as exc:
            status.update(phase="failed", error="llm_key_unreadable",
                          detail=f"LLM_API_KEY_FILE 读不出来: {exc}")
            _atomic_json(status_path, status)
            return 1

    try:
        from .conductor import Conductor
        from .contract import freeze_from_task
        from .hitl_tools import build_task
        from .llm_client import LLMClient

        task = build_task(declaration)
        contract = freeze_from_task(task)
        inputs = contract.cases[0].materialize()
        llm = LLMClient(max_tokens=4096, timeout=int(cfg.get("llm_timeout") or 120))
        backend = _backend(str(cfg.get("backend") or "mock"), cfg)

        conductor = Conductor(
            backend=backend, llm=llm,
            max_optimize_rounds=int(cfg.get("max_rounds") or 1),
            # 一条候选路径：这一层是「入口」，不是搜索策略。换 pipeline 会改候选选择口径，
            # 那件事应当单独评估，不该由「有没有这个工具」顺手改掉。
            use_coder_pipeline=False,
            run_root=str(run),
        )
        artifact = conductor.execute(task, inputs,
                                     target_gflops=float(cfg.get("target_gflops") or 0.0))

        metrics = getattr(artifact, "benchmark_data", None) or {}
        summary = {
            "phase": "done",
            "artifact_id": getattr(artifact, "kernel_id", ""),
            "all_passed": bool(getattr(artifact, "all_passed", False)),
            "stopped_reason": metrics.get("stop_reason"),
            "levels": {r.level.value: bool(r.passed)
                       for r in (getattr(artifact, "verification_results", None) or [])},
            "speedup": metrics.get("speedup_vs_baseline"),
            "gflops": metrics.get("gflops"),
            "validity": metrics.get("measurement_validity"),
            "path": getattr(artifact, "saved_dir", ""),
            "rounds": len(getattr(artifact, "iteration_history", None) or []),
            "finished_at": time.time(),
            "duration_s": round(time.time() - started, 1),
        }
        _atomic_json(run / SUMMARY, summary)
        status.update(phase="done", finished_at=summary["finished_at"],
                      duration_s=summary["duration_s"],
                      summary_path=str(run / SUMMARY))
        _atomic_json(status_path, status)
        print(f"[autonomous] done in {summary['duration_s']}s: "
              f"all_passed={summary['all_passed']} speedup={summary['speedup']}",
              file=sys.stderr)
        return 0
    except Exception as exc:                                   # noqa: BLE001
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-12:])
        status.update(phase="failed", error=type(exc).__name__, detail=str(exc)[:800],
                      traceback=tail, finished_at=time.time(),
                      duration_s=round(time.time() - started, 1))
        _atomic_json(status_path, status)
        print(f"[autonomous] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
