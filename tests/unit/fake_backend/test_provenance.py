"""Fake provenance survives compact foreground replies and interrupted calls."""
import json
import os
from types import SimpleNamespace

import pytest

from operator_agent.app import hitl_tools as ht, tool_calls
from operator_agent.execute import task_cancellation

CODE = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'
DECL = dict(request='relu', shapes=[[4, 8]], dsl='python', min_speedup=0.)


def assert_context(value, model='not_used'):
    context = value['execution_context']
    assert context['test_only'] is True
    assert context['gpu_acceptance'] is False
    assert context['performance_evidence'] is False
    assert context['model_mode'] == model


def test_short_verify_and_bench_replies_retain_test_provenance(tmp_path):
    ht.prepare(DECL, tmp_path)
    result = ht.verify(tmp_path, CODE, backend='fake', stop_after='L2')
    assert result['passed'] is True
    assert_context(result)
    for verdict in result['verdicts']:
        assert verdict['metrics']['test_only'] is True
        assert verdict['metrics']['gpu_acceptance'] is False
    assert_context(json.loads((tmp_path/'verify-1.json').read_text()))
    batch = ht.bench(tmp_path, [dict(name='candidate', code=CODE)], backend='fake')
    assert_context(batch)
    assert batch['best'] is None
    assert_context(batch['arms'][0])


def test_repair_budget_refusal_report_retains_fake_provenance(tmp_path, monkeypatch):
    ht.prepare(DECL, tmp_path)
    monkeypatch.setattr(ht, '_llm_problem', lambda: None)
    # Refuse before backend/model construction, with no tagged verifier results.
    result = ht.repair(tmp_path, CODE, backend='fake', oracle_total=1)
    assert result['stopped_reason'] == 'verification_budget_exhausted'
    assert_context(result, 'live')
    from pathlib import Path
    assert 'FAKE / test_only' in Path(result['report']).read_text()


@pytest.mark.parametrize('outcome', ['cancel', 'budget', 'worker_error', 'start_error'])
def test_supervised_foreground_terminal_replies_and_status_are_marked(
        tmp_path, monkeypatch, outcome):
    ht.prepare(DECL, tmp_path)
    clock = [1000.]
    monkeypatch.setattr(tool_calls, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(task_cancellation, 'monotonic', lambda: clock[0])
    from operator_agent.execute import task_budget
    monkeypatch.setattr(task_budget, 'monotonic', lambda: clock[0])
    process = SimpleNamespace(pid=999999, returncode=None)
    process.poll = lambda: process.returncode
    def start(directory):
        if outcome == 'start_error':
            raise OSError('fixture start failure')
        return process
    monkeypatch.setattr(tool_calls, '_start_call_worker', start)
    monkeypatch.setattr(tool_calls.exec_guard, 'task_process_groups',
        lambda pid: (tool_calls.exec_guard.OwnedGroups(), True))
    monkeypatch.setattr(tool_calls.exec_guard, 'stop_task_worker',
        lambda *a, **kw: dict(worker_reaped=True, status='fixture_cleanup'))
    call = tool_calls.ToolCall('operator_repair', dict(run_dir=str(tmp_path),
        code=CODE, backend='fake', wall_clock_s=1))
    assert_context(ht.autonomous_status(tmp_path, call.invocation_id), 'live')
    if outcome == 'cancel':
        assert_context(call.cancel(), 'live')
    elif outcome == 'budget':
        clock[0] += 2
    elif outcome == 'worker_error':
        process.returncode = 1
    body = json.loads(call.poll()['content'][0]['text'])
    assert body['error'] == dict(cancel='task_cancelled', budget='task_budget_exhausted',
        worker_error='worker_failed', start_error='worker_start_failed')[outcome]
    assert_context(body, 'live')
    assert_context(ht.autonomous_status(tmp_path, call.invocation_id), 'live')
    assert_context(call.cancel(), 'live')  # terminal acknowledgement
    stored = json.loads((call.directory/'result.json').read_text())
    assert_context(json.loads(stored['content'][0]['text']), 'live')


def test_cancel_lost_owner_retains_launch_provenance(tmp_path, monkeypatch):
    from operator_agent.execute.fake_backend.config import tool_context
    identity = 'a'*32
    context = tool_context('fake')
    task_cancellation.launch_identity(tmp_path, identity, kind='tool_call', **context)
    task_cancellation.publish_json(tmp_path/'run-status.json', dict(
        invocation_id=identity, phase='running', pid=os.getpid()))
    monkeypatch.setattr(ht, '_pid_alive', lambda pid: False)
    result = task_cancellation.request_cancel(tmp_path, identity)
    assert result['reason'] == 'owner_unavailable'
    assert_context(result)
