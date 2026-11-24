"""Invocation-bound cancellation requests. Only the owning supervisor signals."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time
from ..execute.task_clock import monotonic

from ..core.run_layout import publish_once_artifact

LAUNCH = 'launch.json'
CANCEL = 'cancel-request.json'
TERMINAL = frozenset(('done', 'failed', 'cancelled', 'lost'))


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def valid_invocation(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{32}', value) is not None


def publish_json(path, value):
    publish_once_artifact(Path(path), (json.dumps(value, ensure_ascii=False,
        sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))


def launch_identity(run, invocation, *, kind, **fields):
    if not valid_invocation(invocation):
        raise ValueError('invalid invocation identity')
    publish_json(Path(run)/LAUNCH, dict(schema='operator-task-launch-v1',
        invocation_id=invocation, kind=kind, created_at=time.time(), **fields))


def cancellation(run, invocation):
    value = read_json(Path(run)/CANCEL)
    if value is None or value.get('invocation_id') != invocation:
        return None
    stamp = value.get('requested_monotonic')
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= 0:
        return None
    return value


def call_status(directory, invocation):
    """一次 invocation 现在是什么状态：**只从任务账本读**。

    两层都要看，因为两条路的调用者都问这一个问题：
      * `tool-calls/<inv>/` 是**一次调用**（Job 层）；
      * 运行根是**整段运行**（任务层），autonomous 的 invocation 记在任务事件上。
    账本里没有它 ⇒ 空（没有状态可报，让调用方去问存活证据），不猜、也不去翻旧文件。
    """
    directory = Path(directory)
    from ..core import task_journal
    try:
        view = task_journal.invocation_view(directory, invocation)
        if view is not None:
            return dict(invocation_id=invocation, phase=view['phase'],
                        stopped_reason=view.get('stopped_reason') or '',
                        cleanup=view.get('cleanup'), pid=view.get('pid'),
                        pid_identity=view.get('pid_identity'), source='task_journal',
                        journal_phase=view['journal_phase'], result=view.get('result'),
                        result_reason=view.get('result_reason') or '',
                        terminal=view['terminal'])
        task = task_journal.task_view(directory)
    except task_journal.JournalError:
        return dict(phase='unknown', source='task_journal_unusable')  # 读不动 ≠ 没有
    if task is None or task.get('invocation') not in (None, '', invocation):
        return {}
    return dict(invocation_id=invocation, phase=task['phase'],
                stopped_reason=task.get('reason') or '', cleanup=task.get('cleanup'),
                pid=task.get('pid'), pid_identity=task.get('pid_identity'),
                source='task_journal', journal_phase=task['journal_phase'],
                terminal=task['terminal'])


def request_cancel(run_dir, invocation):
    """Write one request; never send a signal, mutate status or refund budgets."""
    if not valid_invocation(invocation):
        raise ValueError('invalid invocation identity')
    run = Path(run_dir).resolve()
    launch = read_json(run/LAUNCH)
    if not launch or launch.get('invocation_id') != invocation:
        child = run/'tool-calls'/invocation
        launch = read_json(child/LAUNCH)
        if not launch or launch.get('invocation_id') != invocation:
            raise ValueError('invocation mismatch or legacy run without cancellation identity')
        run = child
    status = call_status(run, invocation)
    context = ({'execution_context': launch['execution_context']}
               if isinstance(launch.get('execution_context'), dict) else {})
    if status.get('phase') in TERMINAL:
        return dict(ok=True, accepted=False, phase=status['phase'],
            invocation_id=invocation, run_dir=str(run), reason='already_terminal',
            cleanup=status.get('cleanup'), stopped_reason=status.get('stopped_reason'), **context)
    owner = status if status.get('pid') else (read_json(run/'supervisor.json') or {})
    if owner.get('pid'):
        from ..execute.exec_guard import process_identity
        from ..app.hitl_tools import _pid_alive
        actual = process_identity(owner['pid'])
        if (not _pid_alive(owner['pid']) or (actual and owner.get('pid_identity')
                and actual != owner['pid_identity'])):
            return dict(ok=True, accepted=False, phase='lost', invocation_id=invocation,
                run_dir=str(run), reason='owner_unavailable', cleanup=dict(status='cleanup_unknown',
                    remote_execution='unknown', provider_call='unknown'), **context)
    value = dict(schema='operator-task-cancel-v1', invocation_id=invocation,
        requested_at=time.time(), requested_monotonic=monotonic())
    try:
        publish_json(run/CANCEL, value)
    except FileExistsError:
        existing = cancellation(run, invocation)
        if existing is None:
            raise ValueError('existing cancellation request does not match invocation')
        value = existing
    return dict(ok=True, accepted=True, phase=status.get('phase', 'running'),
        invocation_id=invocation, run_dir=str(run), cancellation='requested',
        requested_at=value['requested_at'], hint='Query operator_status for terminal acknowledgement.',
        **context)


def stop_event(run, invocation, deadline):
    request = cancellation(run, invocation)
    if request and (deadline is None or request['requested_monotonic'] <= deadline):
        return 'user_cancelled', request['requested_monotonic']
    if deadline is not None and monotonic() >= deadline:
        return 'budget_wall_clock', deadline
    return None, None


def partial_artifacts(run):
    run = Path(run)
    return {name: str(run/name) for name in ('model-journal', 'verification-budget.json',
        'generation-records.json', 'output', 'candidate', 'tool-calls') if (run/name).exists()}
