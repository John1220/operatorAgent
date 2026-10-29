"""Audited graceful cleanup for backends owned by a supervised worker.

No token or remote identity is persisted. Abrupt death cannot be certified and
must remain unknown; this is not isolation against hostile same-UID code.
"""
from __future__ import annotations

import math
import time
import uuid
from pathlib import Path

from .task_cancellation import publish_json, valid_invocation

_scope = None


def begin_scope(directory, invocation_id):
    global _scope
    if not valid_invocation(invocation_id):
        raise ValueError('invalid owned invocation')
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError('owned scope directory does not exist')
    _scope = dict(root=root, invocation_id=invocation_id, backends=[], started=0, ready=0)


def register_backend(backend, *, persistent):
    if _scope is not None:
        _scope['backends'].append((backend, persistent is True))
    return backend


def session_starting():
    if _scope is not None:
        _scope['started'] += 1


def session_ready(identity):
    """A ping and an identity were observed; no identity/token is persisted."""
    if _scope is None:
        return
    from remote_scripts.session_control import validate_identity
    validate_identity(identity)
    _scope['ready'] += 1
    publish_json(_scope['root']/('owned-session-ready-'+uuid.uuid4().hex+'.json'),
        dict(schema='operator-owned-session-ready-v1', invocation_id=_scope['invocation_id'],
             ready_at=time.time(), scope='owned persistent session; no GPU result'))


def finish_scope(grace=10.0):
    global _scope
    scope, _scope = _scope, None
    if scope is None:
        return None
    rows = []
    deadline = time.monotonic() + max(0., grace)
    for backend, persistent in scope['backends']:
        try:
            remaining = max(0., deadline-time.monotonic())
            result = backend.close(grace=remaining) if remaining > 0 else None
            rows.append(dict(persistent=persistent,
                quiescent=(result.get('quiescent') if isinstance(result, dict) else None)))
        except Exception:
            rows.append(dict(persistent=persistent, quiescent=None))
    if scope['started'] != scope['ready']:
        # A signal during subprocess startup may arrive before the backend can
        # retain the session handle. A later successful close is not evidence
        # that the unobserved remote process never started or has stopped.
        state = 'unknown'
    elif not scope['backends']:
        state = 'not_started'
    elif any(not row['persistent'] or row['quiescent'] is not True for row in rows):
        state = 'unknown'
    elif scope['started']:
        state = 'quiescent'
    else:
        state = 'not_started'
    receipt = dict(schema='operator-owned-cleanup-v1', invocation_id=scope['invocation_id'],
        completed_at=time.time(), remote_execution=state, sessions_started=scope['started'],
        sessions_ready=scope['ready'], backend_count=len(rows), backends=rows,
        scope='Owned persistent sessions only; provider calls remain unknown.')
    publish_json(scope['root']/'owned-cleanup.json', receipt)
    return receipt


def cleanup_state(directory, invocation_id):
    from .task_cancellation import read_json
    value = read_json(Path(directory)/'owned-cleanup.json')
    if (not value or value.get('schema') != 'operator-owned-cleanup-v1'
            or value.get('invocation_id') != invocation_id
            or value.get('remote_execution') not in ('quiescent', 'not_started', 'unknown')):
        return 'unknown'
    started, ready, count, rows = (value.get('sessions_started'), value.get('sessions_ready'),
                                   value.get('backend_count'), value.get('backends'))
    completed = value.get('completed_at')
    if (type(completed) not in (int, float) or not math.isfinite(completed) or completed <= 0
            or any(type(x) is not int or x < 0 for x in (started, ready, count))
            or ready > started or not isinstance(rows, list) or len(rows) != count
            or any(not isinstance(row, dict) or type(row.get('persistent')) is not bool
                   or (row.get('quiescent') is not None and
                       type(row.get('quiescent')) is not bool) for row in rows)):
        return 'unknown'
    if value['remote_execution'] == 'quiescent' and (
            not count or not started or ready != started or any(not row['persistent'] or row['quiescent'] is not True
                                             for row in rows)):
        return 'unknown'
    if value['remote_execution'] == 'not_started' and (started or any(
            not row['persistent'] or row['quiescent'] is not True for row in rows)):
        return 'unknown'
    return value['remote_execution']
