"""Resume an optimization task from seed/baseline evidence, not latest L2 alone."""
import json
from pathlib import Path

import pytest

from operator_agent.app import hitl_tools as ht
from operator_agent.app import host_workflow as hw
from operator_agent.core.run_contract import code_hash
from operator_agent.execute import host_task_budget as hb


DECL = dict(operator='relu', request='relu fp32', shapes=[[4, 8]], precision='fp32')
SEED = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'
BUDGET = dict(max_verifications=6, wall_clock_s=600)


def performance():
    return dict(median_time_ms=0.2, measurement_validity='valid',
                measurement_can_persist=True, selected_gpu=3, gpu_uuid='GPU-test',
                timing_identity_hash='tid', timing_protocol='protocol', timing_channel='channel',
                torch_version='torch', triton_version='triton', cuda_version='cuda',
                l2_flush_bytes=1, pair_repeat=1, sm_clock_mhz_before=1980.0,
                sm_clock_mhz_after=1980.0, started_at=1.0, completed_at=2.0,
                benchmark_contaminated=False, measurement_intruder_suspected=False,
                l3_passed=True, speedup_vs_baseline=1.0)


def seed_record(*, passed=True, measured=True):
    return dict(id=1, source_hash=code_hash(SEED), purpose='explore', role='baseline',
                status='completed', full_l2_passed=passed, finished_after_deadline=False,
                evidence='/tmp/verify-1/result.json',
                performance=performance() if passed and measured else None)


def snapshot(**overrides):
    record = seed_record()
    value = dict(task_id='t', contract_hash='c'*64, phase='awaiting_host', terminal=False,
                 unfinished=0, can_reserve_exploration=True, can_reserve_final=True,
                 records=[record], last_correct_candidate=dict(
                     source_hash=code_hash(SEED), evidence='/tmp/verify-1/result.json'),
                 model_budget={'calls': None}, stop_reason=None, expired=False,
                 remaining_verifications=4,
                 seed=dict(sealed=True, source='/tmp/seed.py', source_hash=code_hash(SEED),
                           status='established'),
                 baseline=dict(source_hash=code_hash(SEED), round=1,
                               evidence='/tmp/verify-1/result.json', performance=performance()),
                 best_candidate=dict(source_hash=code_hash(SEED), round=1,
                                     evidence='/tmp/verify-1/result.json', performance=performance()),
                 candidates=[], noise_band=dict(pct=0.1, source='nominal_table'))
    value.update(overrides)
    return value


def resume_with(tmp_path, monkeypatch, value):
    run = tmp_path/'task'
    ht.prepare(DECL, run, host_budget=BUDGET, seed_source=SEED)
    monkeypatch.setattr(hb.HostTaskBudget, 'snapshot',
                        lambda self, readonly=False: value)
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, {'levels': {'L2': True}}, run/'verify-1'))
    return hw.resume(run)


def test_seed_baseline_gate_blocks_other_sources_until_eligible_l3(tmp_path):
    run = tmp_path/'task'
    receipt = dict(schema='noise-preflight-v1', channel='relu-cuda-event', ok=True,
                   action='start', test_only=False,
                   identity=dict(backend='ssh', shape=[4, 8],
                                 device=dict(index=3, name='NVIDIA H20',
                                             uuid='GPU-634bbc76-9ab6-220a-48a8-27247d7acbe2')),
                   effective_band_pct=0.4, nominal_band_pct=0.1,
                   planned_effect_pct=2.5, created_at='2026-11-24T00:00:00+00:00')
    prepared = ht.prepare(DECL, run, host_budget=BUDGET, seed_source=SEED,
                          preflight_receipt=receipt)
    budget = hb.HostTaskBudget(run, prepared['contract_hash'])
    assert budget.snapshot()['noise_band']['pct'] == 0.4
    with pytest.raises(ht.ToolError) as error:
        ht.verify(run, SEED + '\n# candidate\n', backend='mock', stop_after='L3')
    assert error.value.code == 'host_seed_baseline_required'
    ht.verify(run, SEED, backend='mock', stop_after='L3')
    snapshot = budget.snapshot()
    assert snapshot['seed']['status'] == 'established'
    assert snapshot['baseline']['source_hash'] == code_hash(SEED)
    assert snapshot['records'][0]['role'] == 'baseline'
    ht.verify(run, SEED + '\n# candidate\n', backend='mock', stop_after='L2')


def test_seed_pending_asks_for_baseline_verification(tmp_path, monkeypatch):
    value = resume_with(tmp_path, monkeypatch, snapshot(
        seed=dict(sealed=True, source='/tmp/seed.py', source_hash=code_hash(SEED), status='pending'),
        baseline=None, best_candidate=None, last_correct_candidate=None, records=[],
        can_reserve_final=False))
    assert value['next_action'] == 'verify_seed'
    assert value['next_context']['phase'] == 'verify_seed'


def test_established_baseline_with_remaining_budget_optimizes_instead_of_final(tmp_path, monkeypatch):
    value = resume_with(tmp_path, monkeypatch, snapshot())
    assert value['next_action'] == 'optimize'
    assert value['next_context']['phase'] == 'optimize'
    assert value['next_context']['verification_round'] == 1


def test_exhausted_exploration_finals_the_best(tmp_path, monkeypatch):
    value = resume_with(tmp_path, monkeypatch, snapshot(can_reserve_exploration=False))
    assert value['next_action'] == 'final_verify'
    assert value['selected_verification'] == 1


def test_failed_seed_stops_with_a_reason(tmp_path, monkeypatch):
    value = resume_with(tmp_path, monkeypatch, snapshot(
        seed=dict(sealed=True, source='/tmp/seed.py', source_hash=code_hash(SEED), status='l2_failed'),
        baseline=None, best_candidate=None, last_correct_candidate=None,
        can_reserve_final=False, records=[seed_record(passed=False)]))
    assert value['next_action'] == 'report_stopped'
    assert value['stop_reason'] == 'seed_not_correct_for_contract'


def test_infra_held_seed_verification_retries_once(tmp_path, monkeypatch):
    evidence = tmp_path/'verify-1-result.json'
    evidence.write_text(json.dumps(dict(
        route=dict(route='hold', kind='infra', counts_as_iteration=False),
        verdicts=[dict(level='L2', passed=False, metrics=dict(
            reference_identity_unknown=True, transport_outcome='outcome_unknown'))])))
    record = seed_record(passed=False)
    record['evidence'] = str(evidence)
    common = dict(seed=dict(sealed=True, source='/tmp/seed.py', source_hash=code_hash(SEED),
                            status='l2_failed'), baseline=None, best_candidate=None,
                  last_correct_candidate=None, can_reserve_final=False)
    value = resume_with(tmp_path, monkeypatch, snapshot(**common, records=[record]))
    assert value['next_action'] == 'verify_seed'
    second = dict(record, id=2)
    value = resume_with(tmp_path, monkeypatch, snapshot(**common, records=[record, second]))
    assert value['next_action'] == 'report_stopped'


def test_unmeasurable_baseline_retries_then_stops(tmp_path, monkeypatch):
    record = seed_record(measured=False)
    common = dict(baseline=None, best_candidate=None, last_correct_candidate=None,
                  records=[record], can_reserve_final=False)
    value = resume_with(tmp_path, monkeypatch, snapshot(**common))
    assert value['next_action'] == 'verify_seed'
    value = resume_with(tmp_path, monkeypatch, snapshot(
        **common, can_reserve_exploration=False))
    assert value['next_action'] == 'report_stopped'
    assert value['stop_reason'] == 'baseline_unavailable'
