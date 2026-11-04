"""Serial A/B rehearsal with parent-owned deadlines, cancellation and cleanup."""
import hashlib
import os
from pathlib import Path
import sys
import time
import uuid

from operator_agent.app.autonomous import _atomic_json
from operator_agent.execute import exec_guard, s5_shared_budget as shared
from operator_agent.execute.task_clock import monotonic

from . import plan as plans


def _offline_environment():
    from operator_agent.execute.s5_verify_client import enabled
    if enabled():
        raise ValueError('offline rehearsal refuses inherited S5 verification delegation')


def cancel(root):
    root = Path(root).resolve()
    launch = plans.read(root/'execution.json')
    if launch.get('schema') != 'phase1-execution-v1':
        raise ValueError('not a phase1 execution')
    if launch.get('phase') != 'running':
        return dict(accepted=False, phase=launch['phase'])
    marker = root/'cancel.json'
    try:
        plans.write_once(marker, dict(invocation_id=launch['invocation_id'],
                                      requested_at=time.time()))
    except FileExistsError:
        if marker.is_symlink() or plans.read(marker).get('invocation_id') != launch['invocation_id']:
            raise ValueError('cancel request invocation mismatch')
    return dict(accepted=True, invocation_id=launch['invocation_id'], phase='requested')


def _cancelled(root, invocation):
    marker = root/'cancel.json'
    return marker.is_file() and plans.read(marker).get('invocation_id') == invocation


def _supervise(argv, directory, deadline, root, invocation):
    groups = exec_guard.OwnedGroups()
    process = None
    reason = None
    started = monotonic()
    try:
        process = exec_guard.start_task_worker(argv, cwd=str(plans.REPO),
                                              log_path=directory/'execution.log')
        groups.add(process.pid)
        while process.poll() is None:
            current, complete = exec_guard.task_process_groups(process.pid)
            groups.update(current)
            groups.discovery_complete &= complete
            if _cancelled(root, invocation):
                reason = 'user_cancelled'
                break
            if monotonic() >= deadline:
                reason = 'budget_wall_clock'
                break
            time.sleep(.05)
    except KeyboardInterrupt:
        cancel(root)
        reason = 'user_cancelled'
    finally:
        cleanup = (exec_guard.stop_task_worker(process, groups) if process is not None else
                   dict(status='no_worker_started', worker_reaped=True))
    return dict(returncode=process.returncode if process else None, stopped_reason=reason,
                started_monotonic=started, deadline_monotonic=deadline,
                finished_monotonic=monotonic(), elapsed_s=monotonic()-started, cleanup=cleanup)


def _final_evaluation(plan, task, cell, selected, root, invocation):
    source = Path(selected['selected_source']).resolve()
    if cell/'work' not in source.parents or source.is_symlink():
        raise ValueError('selected candidate must be inside its own work directory')
    data = source.read_bytes()
    source_hash = hashlib.sha256(data).hexdigest()
    if source_hash != selected.get('source_hash'):
        raise ValueError('development selected source binding mismatch')
    if selected.get('contract_hash') != task['contract_hash']:
        raise ValueError('development contract binding mismatch')
    pinned = cell/'selected.py'
    with pinned.open('xb') as stream:
        stream.write(data)
    config = dict(cell_id=selected['cell_id'], plan_hash=plan['hash'],
        run_dir=str(cell/'final'), source=str(pinned), source_hash=source_hash,
        declaration=task['declaration'], contract_hash=task['contract_hash'],
        profile=plan['fake']['profile'])
    plans.write_once(cell/'final-config.json', config)
    started = monotonic()
    deadline = started+plan['final_evaluation']['wall_clock_s']
    outcome = _supervise([sys.executable, '-B', '-m',
        'benchmarks.experiments.phase1.finalize', str(cell/'final-config.json')],
        cell, deadline, root, invocation)
    receipt = cell/'final'/'final-result.json'
    outcome.update(receipt=str(receipt) if receipt.is_file() else None, source_hash=source_hash,
                   contract_hash=task['contract_hash'], max_verifications=1)
    if receipt.is_file() and not outcome['stopped_reason']:
        if plans.read(receipt)['finished_monotonic'] > deadline:
            outcome['stopped_reason'] = 'budget_wall_clock'
    return outcome


def _run_cell(plan, task, slot, root, invocation):
    cell = root/'cells'/slot['cell_id']
    work = cell/'work'
    started = monotonic()
    deadline = started+plan['budget']['wall_clock_s']
    ledger = cell/'budget.sqlite'
    task_id = 'p1-'+plans.digest([plan['hash'], slot['cell_id']])[:24]
    limits = dict(task_id=task_id, max_calls=plan['budget']['max_llm_calls'],
        max_output_tokens=40960, total_output_tokens=40960*plan['budget']['max_llm_calls'],
        wall_clock_s=plan['budget']['wall_clock_s'])
    shared.create(ledger, **limits)
    config = dict(schema='phase1-cell-config-v1', **slot, plan_hash=plan['hash'],
        run_dir=str(work), declaration=task['declaration'], backend='fake',
        fake=plan['fake'], connection={}, budget=plan['budget'],
        shared_ledger=str(ledger), shared_task_id=task_id,
        started_monotonic=started, deadline_monotonic=deadline,
        max_rounds=plan['max_rounds'])
    plans.write_once(cell/'cell-config.json', config)
    module = plan['entries'][slot['arm']]
    outcome = _supervise([sys.executable, '-B', '-m', module, str(cell/'cell-config.json')],
                         cell, deadline, root, invocation)
    record = dict(schema='phase1-cell-record-v1', **slot, plan_hash=plan['hash'],
        phase='failed', reason=outcome['stopped_reason'], development=outcome,
        execution_context=plan['execution_context'], final_evaluation=None,
        selected_source_hash=None, started_at=time.time()-outcome['elapsed_s'])
    path = work/('baseline-result.json' if slot['arm'] == 'A' else 'delegate-result.json')
    if path.is_file():
        selected = plans.read(path)
        if (selected.get('cell_id') != slot['cell_id'] or selected.get('plan_hash') != plan['hash']
                or selected.get('contract_hash') != task['contract_hash']):
            raise ValueError('worker result identity mismatch')
        record.update(phase=selected['phase'], worker_result=str(path),
                      reason=record['reason'] or selected.get('reason'))
        if selected.get('selected_source') and not outcome['stopped_reason']:
            record['final_evaluation'] = _final_evaluation(plan, task, cell, selected, root, invocation)
            record['selected_source_hash'] = record['final_evaluation']['source_hash']
            final_stop = record['final_evaluation'].get('stopped_reason')
            if final_stop:
                record.update(phase='cancelled' if final_stop == 'user_cancelled' else 'done',
                              reason=final_stop)
    if outcome['stopped_reason']:
        record['phase'] = 'cancelled' if outcome['stopped_reason'] == 'user_cancelled' else 'done'
    if record.get('reason') is None:
        record['reason'] = 'worker_missing_terminal_receipt'
    record['finished_at'] = time.time()
    _atomic_json(cell/'record.json', record)
    return record


def run(plan_path, output):
    _offline_environment()
    plan = plans.load(plan_path)
    root = plans.fresh_directory(output)
    plans.write_once(root/'plan.json', plan)
    invocation = uuid.uuid4().hex
    execution = dict(schema='phase1-execution-v1', plan_hash=plan['hash'],
        invocation_id=invocation, phase='running', started_at=time.time(),
        execution_context=plan['execution_context'], formal_comparison_credit=False)
    _atomic_json(root/'execution.json', execution)
    tasks = {task['task_id']: task for task in plan['tasks']}
    for slot in plan['schedule']:
        cell = root/'cells'/slot['cell_id']
        cell.mkdir(parents=True)
        _atomic_json(cell/'record.json', dict(schema='phase1-cell-record-v1', **slot,
            plan_hash=plan['hash'], phase='not_started', reason='not_dispatched',
            execution_context=plan['execution_context']))
    try:
        for slot in plan['schedule']:
            if _cancelled(root, invocation):
                execution['phase'] = 'cancelled'
                break
            plans.validate(plan)
            try:
                record = _run_cell(plan, tasks[slot['task_id']], slot, root, invocation)
            except Exception as exc:
                path = root/'cells'/slot['cell_id']/'record.json'
                record = plans.read(path)
                record.update(phase='failed', reason=type(exc).__name__, detail=str(exc))
                _atomic_json(path, record)
            if record['phase'] == 'cancelled':
                execution['phase'] = 'cancelled'
                break
        else:
            execution['phase'] = 'done'
    except (KeyboardInterrupt, ValueError) as exc:
        execution.update(phase='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                         reason=type(exc).__name__, detail=str(exc))
    finally:
        execution['finished_at'] = time.time()
        _atomic_json(root/'execution.json', execution)
    from .summary import summarize
    summary = summarize(root)
    _atomic_json(root/'summary.json', summary)
    (root/'summary.md').write_text(markdown(summary), encoding='utf-8')
    return summary


def markdown(summary):
    lines = ['# Phase 1 离线执行回放', '',
        '> FAKE / test_only：不计入真机对照、GPU 验收或 DSH 宿主验收。', '',
        '| 格位 | 终态 | 独立 L2 | 模型预留 | 预算可核 | 人工介入 |',
        '|---|---|---|---|---|---|']
    for row in summary['cells']:
        lines.append('| {cell_id} | {phase} | {l2} | {calls} | {audit} | {human} |'.format(
            **row, l2=row['final_evaluation'].get('full_l2_passed'),
            calls=row['model_budget'].get('reserved_calls'), audit=row['budget_auditable'],
            human=row['events']['count'] if row['events']['count'] is not None else 'unknown'))
    lines += ['', '人工分钟与未覆盖的介入来源为 unknown；不据此声称节省人时或插件收益。', '']
    return '\n'.join(lines)
