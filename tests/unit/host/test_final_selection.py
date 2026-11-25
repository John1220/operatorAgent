"""Final source selection is a journal projection, not a second state file."""
from pathlib import Path

import pytest

from operator_agent.app import hitl_tools as ht
from operator_agent.core.run_contract import code_hash
from operator_agent.execute import host_task_budget as hb
from tests.unit.test_host_task_budget import budget_state, checked, prepared, rewrite_budget_state


SEED = 'import torch\n\ndef kernel(x):\n    return x\n'
CAND = 'import torch\n\ndef kernel(x):\n    return x + 1\n'


def layout(tmp_path, *, best='seed'):
    run = tmp_path/'task'
    run.mkdir()
    seed = run/'seed.py'
    seed.write_text(SEED)
    cand = run/'verify-2'/'candidate.py'
    cand.parent.mkdir()
    cand.write_text(CAND)
    best_candidate = (dict(source_hash=code_hash(SEED), round=1, source=str(seed),
                           role='baseline') if best == 'seed' else
                      dict(source_hash=code_hash(CAND), round=2, source=str(cand),
                           role='candidate'))
    return run, dict(seed=dict(source_hash=code_hash(SEED), source=str(seed)),
                     best_candidate=best_candidate)


def test_final_defaults_to_ledger_seed_when_seed_retained(tmp_path):
    run, snapshot = layout(tmp_path, best='seed')
    actual, selection = ht._recommended_final(snapshot, run, CAND, None)
    assert actual == SEED
    assert selection['recommended_source'] == code_hash(SEED)
    assert selection['submitted_source'] == code_hash(CAND)
    assert selection['actual_source'] == code_hash(SEED)
    assert selection['override'] is False
    assert selection['reason'] == 'ledger_recommended_default'


def test_final_override_with_reason_uses_submitted_source(tmp_path):
    run, snapshot = layout(tmp_path, best='seed')
    actual, selection = ht._recommended_final(
        snapshot, run, CAND, 'host accepts the only L2 candidate despite latency')
    assert actual == CAND
    assert selection['override'] is True
    assert selection['override_reason'] == 'host accepts the only L2 candidate despite latency'
    assert selection['actual_source'] == code_hash(CAND)


def test_final_uses_recommended_candidate_when_candidate_best(tmp_path):
    run, snapshot = layout(tmp_path, best='candidate')
    actual, selection = ht._recommended_final(snapshot, run, CAND, None)
    assert actual == CAND
    assert selection['recommended_source'] == code_hash(CAND)
    assert selection['reason'] == 'submitted_matches_recommendation'


def test_changed_recommended_source_is_refused(tmp_path):
    run, snapshot = layout(tmp_path, best='candidate')
    Path(snapshot['best_candidate']['source']).write_text(CAND + '\n# changed\n')
    with pytest.raises(ht.ToolError, match='final_selection_source'):
        ht._recommended_final(snapshot, run, CAND, None)


def test_final_selection_lands_atomically_with_reservation(tmp_path):
    run, budget = prepared(tmp_path, hb.HostTaskLimits(max_verifications=3))
    snapshot, _ = checked(budget)
    source_hash = snapshot['last_correct_candidate']['source_hash']

    def add_performance(state):
        state['records'][0]['performance'] = dict(
            median_time_ms=0.2, measurement_validity='valid',
            measurement_can_persist=True, selected_gpu=3, gpu_uuid='GPU-test',
            timing_identity_hash='tid', timing_protocol='protocol', timing_channel='channel',
            torch_version='torch', triton_version='triton', cuda_version='cuda',
            l2_flush_bytes=1, pair_repeat=1, sm_clock_mhz_before=1980.0,
            sm_clock_mhz_after=1980.0, started_at=1.0, completed_at=2.0,
            benchmark_contaminated=False, measurement_intruder_suspected=False,
            l3_passed=True, speedup_vs_baseline=1.0)

    rewrite_budget_state(run, add_performance)
    snapshot = budget.snapshot()
    assert snapshot['best_candidate']['source_hash'] == source_hash
    code = Path(snapshot['best_candidate']['source']).read_text()
    actual, selection = ht._recommended_final(snapshot, run, code, None)
    reservation = budget.reserve(code_hash(actual), purpose='final', final_selection=selection)
    state = budget_state(run)
    assert state['final_selection'] == selection
    assert state['records'][-1]['id'] == reservation['id']
    assert not (run/'final-selection.json').exists()
    assert not (run/'best.json').exists()
    budget.finish(reservation['id'], outcome='failed')
    assert budget.snapshot()['final_selection'] == selection


def test_verify_final_records_selection_in_the_journal(tmp_path):
    run = tmp_path/'task'
    declaration = dict(operator='relu', request='relu fp32', shapes=[[4, 8]], precision='fp32')
    ht.prepare(declaration, run, host_budget=dict(max_verifications=3, wall_clock_s=600))
    ht.verify(run, 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n',
              backend='fake', stop_after='L2')
    result = ht.verify(run, 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n',
                       backend='fake', stop_after='L2', verification_purpose='final')
    assert result['final_selection']['reason'] == 'no_recommendation'
    assert budget_state(run)['final_selection'] == result['final_selection']
    assert result['host_task']['phase'] == 'final_complete'


def test_mcp_exposes_explicit_override_reason():
    from operator_agent.app import mcp_server
    tools = {tool['name']: tool for tool in mcp_server._tool_definitions(host_mode=True)}
    assert 'override_reason' in tools['operator_verify']['inputSchema']['properties']


def test_invalid_final_selection_cannot_spend_a_reservation(tmp_path):
    run, budget = prepared(tmp_path, hb.HostTaskLimits(max_verifications=3))
    snapshot, _ = checked(budget)
    before = (run/'journal.jsonl').read_bytes()
    selection = dict(schema=hb.FINAL_SELECTION_SCHEMA, override=False,
                     actual_source='f'*64)
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.reserve(snapshot['last_correct_candidate']['source_hash'],
                       purpose='final', final_selection=selection)
    assert error.value.code == 'host_final_selection'
    assert (run/'journal.jsonl').read_bytes() == before
    assert budget.snapshot()['reserved'] == 1
