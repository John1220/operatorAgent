"""Real offline A/B workers, independent finals and owned-process termination."""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from benchmarks.experiments.phase1 import plan as plans, runner, summary
from operator_agent.execute.task_clock import monotonic


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    # Parent and freshly launched workers must not inherit a live connection.
    for name in tuple(os.environ):
        if name.startswith(('LLM_', 'OPERATOR_AGENT_S5_DEV_', 'OPERATOR_AGENT_S5_SHARED_')):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv('OPERATOR_AGENT_S5_TASK_ID', raising=False)


def freeze(tmp_path, scenario):
    path = tmp_path/'frozen'
    plan = plans.freeze(path, fake={'scenario': scenario})
    return path/'plan.json', plan


def test_real_eight_cell_repair_rehearsal_keeps_auditable_source_bound_evidence(tmp_path):
    plan_path, plan = freeze(tmp_path, 'correctness-repair')
    root = tmp_path/'execution'
    result = runner.run(plan_path, root)
    assert result['phase'] == 'done'
    assert result['planned_cells'] == result['retained_cells'] == 8
    assert [row['arm'] for row in result['cells']] == ['A', 'B', 'B', 'A', 'A', 'B', 'B', 'A']
    assert result['execution_context']['test_only'] is True
    assert result['gpu_acceptance'] is False
    assert result['formal_comparison_credit'] is False
    assert result['dsh_entry_verified'] is False
    assert all(arm['terminal'] == arm['correct_candidates'] == 4 for arm in result['arms'].values())
    tasks = {task['task_id']: task for task in plan['tasks']}
    for row in result['cells']:
        assert row['phase'] == 'done', row
        assert row['budget_auditable'] is True, row['problems']
        assert row['model_budget']['provider_dispatches'] == 0
        assert row['model_budget']['api_usage'] is None
        assert row['events']['count'] is None
        assert row['events']['observed_failure_count'] >= 1
        assert row['events']['observed_retry_count'] >= 1
        final = row['final_evaluation']
        assert final['complete'] is True and final['full_l2_passed'] is True, row
        assert final['levels'] == {'L1': True, 'L2': True, 'L3': False}
        cell = root/'cells'/row['cell_id']
        pinned = hashlib.sha256((cell/'selected.py').read_bytes()).hexdigest()
        record = plans.read(cell/'record.json')
        receipt = plans.read(cell/'final'/'final-result.json')
        assert pinned == final['source_hash'] == record['selected_source_hash'] == receipt['source_hash']
        assert final['contract_hash'] == tasks[row['task_id']]['contract_hash']
        assert receipt['plan_hash'] == plan['hash']
        assert receipt['result']['execution_context']['test_only'] is True
        for event in row['events']['raw_events']:
            source = event['source']
            assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
        # Failed L2 and retries remain in the original event/evidence files.
        assert row['events']['failed_events'] and row['events']['retry_events']
        assert row['first_full_l2']['source_hash'] == pinned
    assert summary.summarize(root) == result
    assert plans.read(root/'summary.json') == result


def test_real_cancel_keeps_first_cell_cancelled_and_other_seven_not_started(tmp_path):
    plan_path, plan = freeze(tmp_path, 'timeout')
    root = tmp_path/'execution'
    events = root/'cells'/plan['schedule'][0]['cell_id']/'work'/'fake-events.jsonl'
    outcomes, failures = [], []
    def cancel_when_blocked():
        end = time.monotonic()+15
        while time.monotonic() < end:
            try:
                rows = [json.loads(line) for line in events.read_text().splitlines()]
                if any(row['operation'] == 'blocked' for row in rows):
                    outcomes.append(runner.cancel(root))
                    return
            except (OSError, ValueError):
                pass
            time.sleep(.02)
        failures.append('first fake worker never reached the blocking compile call')
        if (root/'execution.json').is_file():
            runner.cancel(root)
    canceller = threading.Thread(target=cancel_when_blocked, daemon=True)
    canceller.start()
    result = runner.run(plan_path, root)
    canceller.join(timeout=1)
    assert not canceller.is_alive() and not failures
    assert outcomes and outcomes[0]['accepted'] is True
    assert result['phase'] == 'cancelled'
    assert len(result['cells']) == 8
    assert result['cells'][0]['phase'] == 'cancelled'
    first = plans.read(root/'cells'/plan['schedule'][0]['cell_id']/'record.json')
    assert first['development']['stopped_reason'] == 'user_cancelled'
    assert first['development']['cleanup']['worker_reaped'] is True
    assert all(row['phase'] == 'not_started' for row in result['cells'][1:])
    assert summary.summarize(root) == result


def test_real_parent_deadline_reaps_a_blocked_worker(tmp_path):
    root = tmp_path/'execution'
    root.mkdir()
    worker = root/'worker'
    worker.mkdir()
    blocked = worker/'blocked'
    invocation = 'a'*32
    plans.write_once(root/'execution.json', dict(schema='phase1-execution-v1',
        phase='running', invocation_id=invocation))
    script = 'from pathlib import Path;import time;Path(%r).touch();time.sleep(60)' % str(blocked)
    result = runner._supervise([sys.executable, '-B', '-c', script], worker,
        monotonic()+.6, root, invocation)
    assert blocked.is_file()
    assert result['stopped_reason'] == 'budget_wall_clock'
    assert result['cleanup']['worker_reaped'] is True
    assert result['returncode'] is not None
    assert result['elapsed_s'] < 5


def test_cancel_during_last_independent_final_cannot_finish_batch_as_done(tmp_path, monkeypatch):
    plan_path, plan = freeze(tmp_path, 'success')
    root = tmp_path/'execution'
    last_id = plan['schedule'][-1]['cell_id']
    run_cell = runner._run_cell
    seen = []
    def earlier_cells(plan, task, slot, root, invocation):
        if slot['cell_id'] == last_id:
            return run_cell(plan, task, slot, root, invocation)
        record = dict(slot, schema='phase1-cell-record-v1', plan_hash=plan['hash'],
            phase='done', reason='fixture_complete', execution_context=plan['execution_context'],
            development=dict(elapsed_s=0.), final_evaluation=None)
        runner._atomic_json(root/'cells'/slot['cell_id']/'record.json', record)
        return record
    def supervised(argv, cell, deadline, root, invocation):
        started = monotonic()
        outcome = dict(returncode=0, stopped_reason=None, started_monotonic=started,
            deadline_monotonic=deadline, finished_monotonic=started, elapsed_s=0.,
            cleanup=dict(worker_reaped=True))
        if argv[3].endswith('.finalize'):
            seen.append(runner.cancel(root))
            outcome.update(returncode=-15, stopped_reason='user_cancelled')
            return outcome
        config = plans.read(cell/'cell-config.json')
        work = cell/'work'
        work.mkdir()
        candidate = work/'candidate.py'
        candidate.write_text('import numpy as np\ndef kernel(X):\n return np.maximum(X, 0)\n')
        plans.write_once(work/'baseline-result.json', dict(cell_id=last_id,
            plan_hash=plan['hash'], phase='done', reason='fixture_complete',
            selected_source=str(candidate), source_hash=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            contract_hash=next(t['contract_hash'] for t in plan['tasks']
                               if t['task_id'] == config['task_id'])))
        return outcome
    monkeypatch.setattr(runner, '_run_cell', earlier_cells)
    monkeypatch.setattr(runner, '_supervise', supervised)
    result = runner.run(plan_path, root)
    assert seen and seen[0]['accepted'] is True
    assert result['phase'] == 'cancelled'
    assert [row['phase'] for row in result['cells']] == ['done']*7+['cancelled']
    final = plans.read(root/'cells'/last_id/'record.json')['final_evaluation']
    assert final['stopped_reason'] == 'user_cancelled'
    assert final['cleanup']['worker_reaped'] is True
