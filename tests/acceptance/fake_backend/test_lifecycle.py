"""Real supervisor/worker, no GPU rental and no external model endpoint."""
import json
from pathlib import Path
import time

import pytest

from operator_agent.app import hitl_tools as ht


def wait_for(run, predicate, timeout=30):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        status = ht.autonomous_status(run)
        if predicate(status):
            return status
        time.sleep(.05)
    pytest.fail('fake task did not reach expected state: ' + repr(ht.autonomous_status(run)))


def launch(tmp_path, monkeypatch, *, scenario='success', **options):
    for name in ('LLM_BASE_URL', 'LLM_API_KEY', 'LLM_API_KEY_FILE'):
        monkeypatch.delenv(name, raising=False)
    run = tmp_path/'run'
    started = ht.run_autonomous(dict(request='relu', shapes=[[4, 8]], dsl='python'),
        run, backend='fake', fake={'scenario': scenario}, max_rounds=2,
        wall_clock_s=options.pop('wall_clock_s', 25), **options)
    assert started['execution_context']['test_only'] is True
    return run, started


@pytest.mark.parametrize('scenario', ['success', 'compile-repair', 'correctness-repair',
                                     'no-improvement', 'transport-unknown', 'hardware-missing'])
def test_normal_pipeline_and_injected_failures(tmp_path, monkeypatch, scenario):
    run, _ = launch(tmp_path, monkeypatch, scenario=scenario)
    status = wait_for(run, lambda s: s.get('phase') in ('done', 'failed', 'lost'))
    assert status['execution_context']['fake']['scenario'] == scenario
    assert status['phase'] != 'lost'
    assert status['execution_context']['gpu_acceptance'] is False
    events = [json.loads(line) for line in (run/'fake-events.jsonl').read_text().splitlines()]
    if scenario == 'hardware-missing':
        assert status['phase'] == 'failed' and 'hardware-missing' in status['detail']
        assert not any(row['operation'] == 'compile' for row in events)
        return
    assert status['phase'] == 'done', status.get('detail')
    summary = status['summary']
    assert summary['all_passed'] is False
    if scenario == 'transport-unknown':
        assert summary['levels']['L2'] is False
        assert not any(row['operation'] == 'benchmark' for row in events)
        return
    assert summary['levels'] == {'L1': True, 'L2': True, 'L3': False}
    assert summary['validity'] == 'uninformative'
    saved = Path(summary['path'])
    assert 'FAKE / test_only' in (saved/'report.md').read_text()
    assert not (run/'kernel_db.jsonl').exists()
    assert not (run/'fake-kernel_db.jsonl').exists()  # no positive admission
    if scenario.endswith('-repair'):
        assert len({event['source_hash'] for event in events if event['operation'] == 'compile'}) >= 2
        calls = [json.loads(p.read_text()) for p in (run/'model-journal').glob('call-*.json')]
        assert any(row['caller'] == 'Fixer' for row in calls)


def test_fake_model_call_budget_stops_before_coder(tmp_path, monkeypatch):
    run, _ = launch(tmp_path, monkeypatch, max_llm_calls=1)
    status = wait_for(run, lambda s: s.get('phase') in ('done', 'failed', 'lost'))
    assert status['phase'] == 'done' and status['stopped_reason'] == 'budget_llm_calls'
    assert status['budget']['model_calls']['reserved'] == 1
    assert status['summary']['all_passed'] is False
    assert status['execution_context']['test_only'] is True


def test_matmul_fixture_uses_normal_full_correctness_suite(tmp_path, monkeypatch):
    monkeypatch.delenv('LLM_BASE_URL', raising=False)
    run = tmp_path/'run'
    ht.run_autonomous(dict(request='matmul fp32', shapes=[[8, 8, 8]], dsl='python'),
                      run, backend='fake', wall_clock_s=25)
    status = wait_for(run, lambda s: s.get('phase') in ('done', 'failed', 'lost'))
    assert status['phase'] == 'done'
    assert status['summary']['levels'] == {'L1': True, 'L2': True, 'L3': False}


def test_real_deadline_stops_blocked_fake_backend(tmp_path, monkeypatch):
    run, _ = launch(tmp_path, monkeypatch, scenario='timeout', wall_clock_s=3.)
    status = wait_for(run, lambda s: s.get('phase') in ('done', 'failed', 'lost'), timeout=10)
    assert status['phase'] == 'done' and status['stopped_reason'] == 'budget_wall_clock'
    assert status['cleanup']['worker_reaped'] is True
    assert status['execution_context']['test_only'] is True
    assert status['summary']['execution_context']['gpu_acceptance'] is False


def test_real_cancellation_stops_blocked_fake_backend(tmp_path, monkeypatch):
    run, started = launch(tmp_path, monkeypatch, scenario='timeout')
    def blocked(status):
        path = run/'fake-events.jsonl'
        if status.get('phase') in ('done', 'failed', 'lost'):
            pytest.fail('task terminated before cancellation: ' + repr(status))
        return path.is_file() and any(json.loads(line)['operation'] == 'blocked'
                                      for line in path.read_text().splitlines())
    wait_for(run, blocked)
    ht.cancel_task(run, started['invocation_id'])
    status = wait_for(run, lambda s: s.get('phase') != 'running')
    assert status['phase'] == 'cancelled' and status['stopped_reason'] == 'user_cancelled'
    assert status['cleanup']['worker_reaped'] is True
    assert status['summary']['execution_context']['test_only'] is True
