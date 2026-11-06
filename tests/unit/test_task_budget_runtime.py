"""Production supervisor/worker tests. Local fake HTTP only, no model or GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

# 进程/截止/取消时序敏感: 并行运行时抖动, 用 -m serial 单独串行跑。
pytestmark = pytest.mark.serial

from operator_agent.app import autonomous, hitl_tools as ht
from operator_agent.execute.task_budget import TaskBudgetConfig, snapshot
from tests.unit.test_autonomous import SKETCH, CODE


class Server:
    def __init__(self, replies, delay=0):
        self.calls = []
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                outer.calls.append(body)
                time.sleep(delay)
                content = replies[min(len(outer.calls)-1, len(replies)-1)]
                data = json.dumps(dict(choices=[dict(message=dict(content=content), finish_reason='stop')],
                    usage=dict(prompt_tokens=10, completion_tokens=10, total_tokens=20))).encode()
                try:
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            def log_message(self, *a):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True

    def __enter__(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()


def launch(tmp_path, monkeypatch, server, **budget):
    monkeypatch.setenv('LLM_BASE_URL', f'http://127.0.0.1:{server.server.server_port}/v1')
    monkeypatch.setenv('LLM_MODEL', 'local-budget-fixture')
    monkeypatch.setenv('LLM_API_KEY', 'synthetic-budget-fixture-key')
    run = tmp_path/'run'
    response = ht.run_autonomous(dict(request='matmul fp32', dsl='python',
        shapes=[[8, 8, 8]], min_speedup=0), run, backend='mock', **budget)
    end = time.monotonic()+30
    while time.monotonic() < end:
        status = ht.autonomous_status(run)
        if status.get('phase') in ('done', 'failed', 'lost'):
            return run, response, status
        time.sleep(.03)
    pytest.fail('local budget supervisor did not finish')


def test_call_budget_stops_before_coder_and_is_visible_in_status(tmp_path, monkeypatch):
    with Server([json.dumps(SKETCH)]) as server:
        run, started, status = launch(tmp_path, monkeypatch, server, max_llm_calls=1, wall_clock_s=20)
        assert len(server.calls) == 1
    assert status['phase'] == 'done' and status['stopped_reason'] == 'budget_llm_calls'
    assert status['pid'] == started['pid']
    calls = status['budget']['model_calls']
    assert calls['reserved'] == 1 and calls['remaining'] == 0
    assert status['summary']['all_passed'] is False
    journal = json.loads((run/'model-journal/call-001.json').read_text())
    assert journal['caller'] == 'Designer' and journal['status'] == 'answered'


def test_precheck_retry_uses_same_task_ledger_and_confirmation_is_reserved(tmp_path, monkeypatch):
    invalid = 'def kernel(A, B):\n    return A @'
    with Server([json.dumps(SKETCH), invalid, CODE]) as server:
        run, _, status = launch(tmp_path, monkeypatch, server, max_llm_calls=3, wall_clock_s=30)
        assert len(server.calls) == 3
    assert status['phase'] == 'done', status
    assert status['stopped_reason'] == 'passed'
    assert status['summary']['all_passed']
    calls = [json.loads(path.read_text()) for path in sorted((run/'model-journal').glob('call-*.json'))]
    assert [row['caller'] for row in calls] == ['Designer', 'Coder', 'Coder']
    assert status['budget']['model_calls']['remaining'] == 0
    assert status['budget']['model_calls']['exhausted_reason'] is None
    verification = status['budget']['verification']
    assert verification['spent']['confirm'] == 1
    assert verification['spent_all'] <= 20
    assert invalid not in '\n'.join(path.read_text() for path in (run/'model-journal').glob('*.json'))


def test_remaining_deadline_interrupts_blocked_model_request(tmp_path, monkeypatch):
    with Server([json.dumps(SKETCH)], delay=5) as server:
        started = time.monotonic()
        _, _, status = launch(tmp_path, monkeypatch, server, wall_clock_s=2)
        elapsed = time.monotonic()-started
        assert len(server.calls) == 1
    assert elapsed < 4
    assert status['phase'] == 'done' and status['stopped_reason'] == 'budget_wall_clock'
    assert status['summary']['all_passed'] is False
    assert status['budget']['model_calls']['reserved'] == 1
    assert status['budget']['remote_cancellation_proven'] is False


def test_supervisor_kills_blocked_worker_and_separate_session_descendant(tmp_path, monkeypatch):
    run = tmp_path/'run'
    run.mkdir()
    (run/'autonomous.json').write_text(json.dumps(dict(task_budget=dict(wall_clock_s=.5))))
    marker = tmp_path/'escaped-descendant'
    child = 'import time;from pathlib import Path;time.sleep(1);Path('+repr(str(marker))+').write_text("survived")'
    worker = 'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(child)+'],start_new_session=True);time.sleep(60)'
    def start(*a):
        return subprocess.Popen([sys.executable, '-c', worker], start_new_session=True)
    monkeypatch.setattr(autonomous, '_start_worker', start)
    started = time.monotonic()
    assert autonomous.main([str(run)]) == 0
    assert time.monotonic()-started < 2
    status = json.loads((run/'run-status.json').read_text())
    assert status['stopped_reason'] == 'budget_wall_clock'
    time.sleep(.7)
    assert not marker.exists()


def test_unstarted_or_unreadable_ledgers_stay_unknown(tmp_path):
    policy = TaskBudgetConfig()
    result = snapshot(tmp_path, policy, 10., now=11.)
    assert result['model_calls']['reserved'] is None
    assert result['verification'] is None
    journal = tmp_path/'model-journal'
    journal.mkdir()
    (journal/'call-001.json').write_text('invalid')
    assert snapshot(tmp_path, policy, 10., now=11.)['model_calls']['reserved'] is None


def test_unknown_task_policy_fields_are_rejected():
    with pytest.raises(ValueError):
        TaskBudgetConfig.from_dict(dict(max_callz=3))


def test_deleted_reservation_is_unknown_instead_of_zero(tmp_path):
    from operator_agent.execute.llm_budget import ProductionLLMCallBudget
    policy = TaskBudgetConfig()
    ledger = ProductionLLMCallBudget(policy.llm_limits(), tmp_path/'model-journal')
    identity = dict(model='fixture')
    ledger.bind(identity)
    ledger.reserve(identity, json.dumps(dict(max_tokens=100)), 'Coder', 1)
    assert snapshot(tmp_path, policy, time.monotonic())['model_calls']['reserved'] == 1
    (tmp_path/'model-journal/call-001.json').unlink()
    result = snapshot(tmp_path, policy, time.monotonic())
    assert result['model_calls']['reserved'] is None
    assert result['model_calls']['remaining'] is None
    assert result['model_calls']['journal_state'] == 'unknown_or_inconsistent'


def test_late_supervisor_observation_preserves_earlier_worker_failure(tmp_path, monkeypatch):
    run = tmp_path/'run'
    run.mkdir()
    (run/'autonomous.json').write_text(json.dumps(dict(task_budget=dict(wall_clock_s=.1))))
    class Finished:
        pid = 123456789
        def poll(self): return 1
        def wait(self, **kwargs): return 1
    def start(run, deadline, started, invocation):
        (run/autonomous.WORKER_RESULT).write_text(json.dumps(dict(phase='failed',
            invocation_id=invocation, error='OriginalFailure', detail='original error',
            finished_monotonic=started+.01)))
        time.sleep(.15)  # The supervisor observes the result after its deadline.
        return Finished()
    monkeypatch.setattr(autonomous, '_start_worker', start)
    monkeypatch.setattr(autonomous, '_stop_worker', lambda *a: dict(status='fixture'))
    assert autonomous.main([str(run)]) == 1
    status = json.loads((run/'run-status.json').read_text())
    assert status['phase'] == 'failed' and status['error'] == 'OriginalFailure'
    assert not status['budget']['wall_clock']['expired']


def test_cleanup_discovery_failure_is_reported_unknown(monkeypatch):
    process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    monkeypatch.setattr(autonomous, '_owned_groups', lambda pid: ({pid}, False))
    try:
        cleanup = autonomous._stop_worker(process)
        assert cleanup['status'] == 'cleanup_unknown' and cleanup['worker_reaped']
        assert cleanup['all_descendants_proven'] is False
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
