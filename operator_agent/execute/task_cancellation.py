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
    status = read_json(run/'run-status.json') or {}
    if status.get('invocation_id') not in (None, invocation):
        raise ValueError('status invocation mismatch')
    if status.get('phase') in TERMINAL:
        return dict(ok=True, accepted=False, phase=status['phase'],
            invocation_id=invocation, run_dir=str(run), reason='already_terminal',
            cleanup=status.get('cleanup'), stopped_reason=status.get('stopped_reason'))
    owner = status if status.get('pid') else (read_json(run/'supervisor.json') or {})
    if owner.get('pid'):
        from ..execute.exec_guard import process_identity
        from ..app.hitl_tools import _pid_alive
        actual = process_identity(owner['pid'])
        if (not _pid_alive(owner['pid']) or (actual and owner.get('pid_identity')
                and actual != owner['pid_identity'])):
            return dict(ok=True, accepted=False, phase='lost', invocation_id=invocation,
                run_dir=str(run), reason='owner_unavailable', cleanup=dict(status='cleanup_unknown',
                    remote_execution='unknown', provider_call='unknown'))
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
        requested_at=value['requested_at'], hint='Query operator_status for terminal acknowledgement.')


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
