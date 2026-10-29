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
    <run>/run-status.json   相位机：running → done | failed | cancelled
    <run>/conductor.json    完成时的摘要（artifact / 停因 / 最好读数 / 报告路径）
    <run>/autonomous.log    子进程的 stdout+stderr
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from .task_clock import monotonic
import traceback
import uuid
import signal
from pathlib import Path

STATUS = "run-status.json"
SUMMARY = "conductor.json"
CONFIG = "autonomous.json"
WORKER_RESULT = "worker-result.json"


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


def _worker_main(run, deadline, started_monotonic, invocation_id=None) -> int:
    from .task_budget import TaskBudgetConfig, snapshot, stopped_reason
    started = time.time()
    cfg = json.loads((run / CONFIG).read_text(encoding="utf-8"))
    declaration = json.loads((run / "declaration.json").read_text(encoding="utf-8"))
    policy = TaskBudgetConfig.from_dict(cfg.get('task_budget'))

    status = {"phase": "running", "pid": os.getpid(), "started_at": started,
              "backend": cfg.get("backend"), "max_rounds": cfg.get("max_rounds"),
              "run_dir": str(run),
              # 语料落点也收在 run 目录：这一版的语料是**运行内**的，不写入仓库那份。
              "corpus": {"kernel_db": "run-local", "failure_db": "run-local"}}
    status.update(task_budget=policy.as_dict(), invocation_id=invocation_id)
    conductor = None

    # 落点收进 run 目录：无人值守的运行不该把产物撒到仓库或用户 home 的默认位置。
    os.environ["OPERATOR_AGENT_OUTPUT_DIR"] = str(run / "output")
    os.environ["OPERATOR_AGENT_LOG_DIR"] = str(run / "log")
    # ⚠️ 这两个**不能写 "off"**：2026-10-08（§3.5）之后普通环境变量只能**重定向**，
    # 关不掉持久化 —— `off` 会被解析回 `<repo>/output/*.jsonl`，于是子进程会读写仓库
    # 里那份语料。2026-10-18 实测就撞上了（日志里 `FailureDB 重放 95` 来自仓库）。
    os.environ["OPERATOR_AGENT_KERNEL_DB"] = str(run / "kernel_db.jsonl")
    os.environ["OPERATOR_AGENT_FAILURE_DB"] = str(run / "failure_db.jsonl")
    try:
        from .conductor import Conductor
        from .contract import freeze_from_task
        from .hitl_tools import build_task
        from .llm_client import LLMClient
        from .s5_inner_budget import production_budget
        from .budget import BudgetPlan

        task = build_task(declaration)
        contract = freeze_from_task(task)
        inputs = contract.cases[0].materialize()
        call_timeout = min(float(cfg.get('llm_timeout') or 120), policy.wall_clock_s)
        ledger = production_budget(policy.llm_limits(call_timeout), run/'model-journal',
                                   deadline=deadline)
        llm = LLMClient(max_tokens=4096, timeout=call_timeout, call_budget=ledger)
        backend = _backend(str(cfg.get("backend") or "mock"), cfg)

        conductor = Conductor(
            backend=backend, llm=llm,
            max_optimize_rounds=int(cfg.get("max_rounds") or 1),
            # 一条候选路径：这一层是「入口」，不是搜索策略。换 pipeline 会改候选选择口径，
            # 那件事应当单独评估，不该由「有没有这个工具」顺手改掉。
            use_coder_pipeline=False,
            run_root=str(run),
            budget=BudgetPlan(oracle_total=policy.oracle_total, wall_clock_s=policy.wall_clock_s),
        )
        artifact = conductor.execute(task, inputs,
                                     target_gflops=float(cfg.get("target_gflops") or 0.0))

        metrics = getattr(artifact, "benchmark_data", None) or {}
        generation = getattr(artifact, 'metadata', None) or {}
        summary = {
            "phase": "done",
            "artifact_id": getattr(artifact, "kernel_id", ""),
            "all_passed": bool(getattr(artifact, "all_passed", False)),
            "verification_all_passed": bool(getattr(artifact, "all_passed", False)),
            "stopped_reason": (generation.get('stop') or {}).get('reason') or metrics.get("stop_reason"),
            "levels": {r.level.value: bool(r.passed)
                       for r in (getattr(artifact, "verification_results", None) or [])},
            "speedup": metrics.get("speedup_vs_baseline"),
            "gflops": metrics.get("gflops"),
            "validity": metrics.get("measurement_validity"),
            "path": getattr(artifact, "saved_dir", ""),
            "rounds": len(getattr(artifact, "iteration_history", None) or []),
            "candidate_record_count": len(generation.get('candidate_records', [])),
            "selected_source_hash": generation.get('selected_source_hash'),
            "generation_evidence": (str(run / 'generation-records.json')
                                    if (run / 'generation-records.json').is_file() else None),
            "finished_at": time.time(),
            "duration_s": round(time.time() - started, 1),
        }
        budget = snapshot(run, policy, started_monotonic)
        reason = stopped_reason(budget)
        if summary['stopped_reason'] in ('budget_exhausted', 'wall_clock_exhausted'):
            summary['all_passed'] = False
        if reason:
            summary.update(stopped_reason=reason, all_passed=False)
        summary.update(task_budget=policy.as_dict(), budget=budget)
        _atomic_json(run / SUMMARY, summary)
        status.update(phase="done", finished_at=summary["finished_at"],
                      duration_s=summary["duration_s"],
                      summary_path=str(run / SUMMARY), stopped_reason=summary['stopped_reason'],
                      budget=budget)
        status['finished_monotonic'] = monotonic()
        _atomic_json(run / WORKER_RESULT, status)
        print(f"[autonomous] done in {summary['duration_s']}s: "
              f"all_passed={summary['all_passed']} speedup={summary['speedup']}",
              file=sys.stderr)
        return 0
    except Exception as exc:                                   # noqa: BLE001
        failed_monotonic = monotonic()
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-12:])
        status.update(phase="failed", error=type(exc).__name__, detail=str(exc)[:800],
                      traceback=tail, finished_at=time.time(),
                      duration_s=round(time.time() - started, 1))
        if (run / 'generation-records.json').is_file():
            status['generation_evidence'] = str(run / 'generation-records.json')
        budget = snapshot(run, policy, started_monotonic, now=failed_monotonic)
        reason = stopped_reason(budget)
        status.update(budget=budget, stopped_reason=reason)
        if reason:
            status.update(phase='done', error=None, detail=None)
            _atomic_json(run / SUMMARY, dict(phase='done', all_passed=False,
                stopped_reason=reason, task_budget=policy.as_dict(), budget=budget,
                finished_at=status['finished_at'], duration_s=status['duration_s']))
        status['finished_monotonic'] = failed_monotonic
        _atomic_json(run / WORKER_RESULT, status)
        print(f"[autonomous] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0 if reason else 1


def _owned_groups(worker_pid):
    from .exec_guard import task_process_groups
    return task_process_groups(worker_pid)


def _stop_worker(process, observed_groups=(), graceful_s=0):
    from .exec_guard import stop_task_worker
    return stop_task_worker(process, observed_groups, discover=_owned_groups,
                            graceful_s=graceful_s)


def _start_worker(run, deadline, started_monotonic, invocation_id):
    from .exec_guard import start_task_worker
    return start_task_worker([sys.executable, '-m', 'operator_agent.autonomous',
        '--worker', str(run), str(deadline), str(started_monotonic), invocation_id],
        cwd=str(Path(__file__).resolve().parents[1]))


def _main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 5 and argv[0] == '--worker':
        run = Path(argv[1]).resolve()
        graceful = (os.environ.get('OPERATOR_AGENT_S4_GRACEFUL_CLEANUP') == '1'
                    and (json.loads((run/CONFIG).read_text()).get('backend') == 'ssh')
                    and bool(json.loads((run/CONFIG).read_text()).get('persistent')))
        if not graceful:
            return _worker_main(run, float(argv[2]), float(argv[3]), argv[4])
        from .backend_ownership import begin_scope, finish_scope
        begin_scope(run, argv[4])
        previous = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        try:
            return _worker_main(run, float(argv[2]), float(argv[3]), argv[4])
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                finish_scope()
            finally:
                signal.signal(signal.SIGTERM, previous)
    if len(argv) != 1:
        print('usage: python -m operator_agent.autonomous <run_dir>', file=sys.stderr)
        return 2
    from .task_budget import TaskBudgetConfig, snapshot, stopped_reason
    from .task_cancellation import (launch_identity, read_json, stop_event,
                                    partial_artifacts, valid_invocation, cancellation,
                                    request_cancel)
    from .exec_guard import process_identity
    run = Path(argv[0]).resolve()
    started_monotonic, started = monotonic(), time.time()
    try:
        cfg = json.loads((run / CONFIG).read_text(encoding='utf-8'))
        policy = TaskBudgetConfig.from_dict(cfg.get('task_budget'))
    except (OSError, ValueError, TypeError) as exc:
        _atomic_json(run / STATUS, dict(phase='failed', pid=os.getpid(),
            error=type(exc).__name__, detail=str(exc), stopped_reason='invalid_task_budget',
            finished_at=time.time(), started_at=started))
        return 1
    deadline = started_monotonic + policy.wall_clock_s
    invocation_id = cfg.get('invocation_id')
    if invocation_id is None:  # legacy/manual supervisor entry gets its own identity
        invocation_id = uuid.uuid4().hex
        launch_identity(run, invocation_id, kind='autonomous')
    launch = read_json(run/'launch.json') or {}
    if not valid_invocation(invocation_id) or launch.get('invocation_id') != invocation_id:
        return 1  # never overwrite another invocation's status or signal its PID
    status = dict(phase='running', stage='executing', pid=os.getpid(), started_at=started,
        backend=cfg.get('backend'), max_rounds=cfg.get('max_rounds'), run_dir=str(run),
        task_budget=policy.as_dict(), invocation_id=invocation_id,
        corpus={'kernel_db':'run-local', 'failure_db':'run-local'})
    status['pid_identity'] = process_identity(os.getpid())
    status['budget'] = snapshot(run, policy, started_monotonic)
    _atomic_json(run / STATUS, status)
    reason, event_at = stop_event(run, invocation_id, deadline)
    if reason == 'user_cancelled':
        status.update(phase='cancelled', stopped_reason=reason, finished_at=time.time(),
            cancellation=cancellation(run, invocation_id), partial_artifacts=partial_artifacts(run),
            cleanup=dict(status='no_worker_started', worker_reaped=True,
                         remote_execution='unknown', provider_call='unknown'))
        _atomic_json(run / SUMMARY, dict(phase='cancelled', all_passed=False,
            stopped_reason=reason, task_budget=policy.as_dict(), budget=status['budget']))
        _atomic_json(run / STATUS, status)
        return 1
    def interrupted(signum, frame):
        request_cancel(run, invocation_id)
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        process = _start_worker(run, deadline, started_monotonic, invocation_id)
    except OSError as exc:
        status.update(phase='failed', error=type(exc).__name__, detail=str(exc),
            finished_at=time.time(), stopped_reason='worker_start_failed')
        _atomic_json(run / STATUS, status)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        return 1
    status['worker_pid'] = process.pid
    last_update = 0.
    from .exec_guard import OwnedGroups
    observed_groups = OwnedGroups([process.pid])
    reason = event_at = None
    graceful_remote = (os.environ.get('OPERATOR_AGENT_S4_GRACEFUL_CLEANUP') == '1'
                       and cfg.get('backend') == 'ssh' and bool(cfg.get('persistent')))
    try:
        while process.poll() is None:
            remaining = deadline-monotonic()
            reason, event_at = stop_event(run, invocation_id, deadline)
            if reason:
                break
            groups, discovered = _owned_groups(process.pid)
            observed_groups.update(groups)
            observed_groups.discovery_complete &= discovered
            if monotonic()-last_update >= 1.:
                status['budget'] = snapshot(run, policy, started_monotonic)
                _atomic_json(run / STATUS, status)
                last_update = monotonic()
            try:
                process.wait(timeout=min(.25, max(.001, remaining)))
            except subprocess.TimeoutExpired:
                pass
    finally:
        cleanup = (_stop_worker(process, observed_groups, 12)
                   if reason and graceful_remote else _stop_worker(process, observed_groups))
    try:
        result = json.loads((run / WORKER_RESULT).read_text())
    except (OSError, ValueError):
        result = {}
    same_invocation = result.get('invocation_id') == invocation_id
    finished = result.get('finished_monotonic') if same_invocation else None
    valid_finish = type(finished) in (int, float) and started_monotonic <= finished <= monotonic()
    budget = snapshot(run, policy, started_monotonic,
                      now=finished if valid_finish else None)
    reason, event_at = stop_event(run, invocation_id, deadline)
    result_won = valid_finish and (event_at is None or finished <= event_at)
    if reason and not result_won:
        phase = 'cancelled' if reason == 'user_cancelled' else 'done'
        status.update(phase=phase, stopped_reason=reason,
            error=None, detail=None, budget=budget)
        _atomic_json(run / SUMMARY, dict(phase=phase, all_passed=False,
            stopped_reason=reason, task_budget=policy.as_dict(), budget=budget,
            finished_at=time.time(), duration_s=round(time.time()-started, 3)))
    elif valid_finish and result.get('phase') in ('done', 'failed'):
        status.update(result, pid=os.getpid(), started_at=started, budget=budget)
    else:
        status.update(phase='failed', error='WorkerExited',
            detail='worker exited without a terminal result', stopped_reason='worker_failed', budget=budget)
    from .backend_ownership import cleanup_state
    cleanup.update(remote_execution=(cleanup_state(run, invocation_id)
                                     if graceful_remote else 'unknown'), provider_call='unknown')
    status.update(finished_at=time.time(), duration_s=round(time.time()-started, 3), cleanup=cleanup,
        cancellation=cancellation(run, invocation_id), partial_artifacts=partial_artifacts(run))
    if (run/'generation-records.json').is_file():
        status['generation_evidence'] = str(run/'generation-records.json')
    _atomic_json(run / STATUS, status)
    for sig, handler in previous.items():
        signal.signal(sig, handler)
    return 0 if status['phase'] == 'done' else 1


def main(argv=None) -> int:
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return _main(argv)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
