"""Frozen A/C driver contracts; no model, GPU or live verification calls."""
import copy
import json

import pytest

from benchmarks.experiments.s5 import s5_formal_run, s5_plugin_probe
from benchmarks.experiments.s5 import s5_plugin_workflow as workflow
from operator_agent.run_contract import digest


def fake_readiness(tmp_path):
    return dict(report_path=str(tmp_path / 'readiness.json'), report_sha256='a' * 64,
        runtime=dict(schema='s5-plugin-runtime-v1', root=str(tmp_path / 'runtime'), files={}),
        runtime_files_hash='b' * 64, isolation_implementation={})


def freeze_fixture(tmp_path, monkeypatch, arms=workflow.ARMS, mode=None, exploratory=False):
    readiness = fake_readiness(tmp_path)
    monkeypatch.setattr(s5_plugin_probe, 'load_readiness', lambda path: readiness)

    declarations = {}
    def freeze_service(path, **kwargs):
        path.mkdir(parents=True)
        declarations[str(path)] = dict(kwargs)
        return 'c' * 64

    monkeypatch.setattr(workflow.s5_verify_remote, 'freeze', freeze_service)
    monkeypatch.setattr(workflow.s5_verify_remote, 'load_plan',
        lambda path, sha: dict(declarations[str(path.parent)], contract_hash='d' * 64,
                              mode='ssh', stop_after='L2'))
    monkeypatch.setattr(s5_formal_run, '_read', lambda path: {})
    monkeypatch.setattr(s5_formal_run, 'check_task_inputs', lambda row, contract: dict(operands=[]))
    output = tmp_path / 'frozen'
    policy = dict(mode='shared-known-processes', min_free_mib=8192, max_util_pct=5,
                  allowed_compute_pids=[])
    gpu = dict(index=2, uuid='GPU-12345678-1234-1234-1234-123456789abc', name='NVIDIA H20')
    mode = mode or ('ac' if arms == workflow.ARMS else 'c-only')
    plan = workflow.freeze(output, readiness_report=readiness['report_path'], gpu=gpu,
        resource_policy=policy, arms=arms,
        case_ids=(workflow.STUDY_CASES if mode == 'study' else workflow.PLUGIN_ENTRY_CASES
                  if mode in ('plugin-entry', 'plugin-repair') else workflow.PREFLIGHT_CASES),
        repeats=2 if mode == 'study' else 1, mode=mode, exploratory=exploratory)
    return output, plan


def test_new_plan_freezes_only_a_c_with_same_budget_and_fixed_six_slots(tmp_path, monkeypatch):
    output, plan = freeze_fixture(tmp_path, monkeypatch)
    loaded = workflow.load_plan(output / 'plan.json')
    assert loaded['hash'] == plan['hash']
    assert loaded['arms'] == ['A', 'C']
    assert loaded['planned_runs'] == 6
    assert loaded['budget'] == workflow.BUDGET
    assert loaded['model']['reasoning_effort'] == 'medium'
    assert loaded['model']['id'] == workflow.MODEL
    assert loaded['limits']['max_model_reservations'] == 6 * 24
    assert loaded['limits']['max_output_reservations'] == 6 * 24 * 32768
    assert [slot['arm'] for slot in loaded['schedule']] == list('ACCAAC')
    assert loaded['formal_s5_credit'] is False
    assert loaded['counts_toward_denominator'] is False


def test_c_only_diagnostic_freezes_three_plugin_slots_without_comparison(tmp_path, monkeypatch):
    output, plan = freeze_fixture(tmp_path, monkeypatch, workflow.C_ONLY_ARMS)
    loaded = workflow.load_plan(output / 'plan.json')
    assert loaded['hash'] == plan['hash']
    assert loaded['schema'] == workflow.C_ONLY_SCHEMA
    assert loaded['arms'] == ['C']
    assert loaded['planned_runs'] == 3
    assert [slot['arm'] for slot in loaded['schedule']] == ['C', 'C', 'C']
    assert loaded['limits']['max_model_reservations'] == 3 * 24
    assert loaded['limits']['max_output_reservations'] == 3 * 24 * 32768
    assert loaded['formal_s5_credit'] is False
    assert loaded['counts_toward_denominator'] is False
    assert 'no A/C comparison' in loaded['purpose']


def test_plugin_entry_plan_requires_one_layernorm_c_slot_and_tool_path(tmp_path, monkeypatch):
    output, plan = freeze_fixture(tmp_path, monkeypatch, workflow.C_ONLY_ARMS, 'plugin-entry')
    loaded = workflow.load_plan(output / 'plan.json')
    assert loaded['hash'] == plan['hash']
    assert loaded['schema'] == workflow.PLUGIN_ENTRY_SCHEMA
    assert loaded['mode'] == 'plugin-entry'
    assert loaded['arms'] == ['C']
    assert loaded['planned_runs'] == 1
    assert loaded['cases'][0]['case_id'] == 'layernorm-tail'
    assert loaded['cases'][0]['acceptance_instruction'] == workflow.PLUGIN_ENTRY_INSTRUCTION
    assert loaded['schedule'] == [dict(case_id='layernorm-tail', repeat_id=1, arm='C')]
    assert loaded['limits']['max_model_reservations'] == 24
    assert loaded['formal_s5_credit'] is False
    assert loaded['counts_toward_denominator'] is False


def test_repair_plan_binds_seed_and_rejects_rehashed_seed_drift(tmp_path, monkeypatch):
    output, plan = freeze_fixture(tmp_path, monkeypatch, workflow.C_ONLY_ARMS, 'plugin-repair')
    loaded = workflow.load_plan(output / 'plan.json')
    assert loaded['planned_runs'] == 1
    assert loaded['schema'] == workflow.PLUGIN_REPAIR_SCHEMA
    assert loaded['cases'][0]['initial_candidate'] == workflow.REPAIR_SEED_PATH.read_text()
    altered = copy.deepcopy(plan)
    altered['cases'][0]['initial_candidate'] += '\n'
    altered.pop('hash')
    altered['hash'] = digest(altered)
    path = output / 'altered.json'
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='source drift'):
        workflow.load_plan(path)


def test_frozen_plan_rejects_a_rehashed_legacy_arm_set(tmp_path, monkeypatch):
    output, plan = freeze_fixture(tmp_path, monkeypatch)
    altered = copy.deepcopy(plan)
    altered['arms'] = ['A', 'B', 'C']
    altered.pop('hash')
    altered['hash'] = digest(altered)
    path = output / 'legacy.json'
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='configuration or source drift'):
        workflow.load_plan(path)


def test_freezer_refuses_a_different_budget_before_writing_plan(tmp_path, monkeypatch):
    readiness = fake_readiness(tmp_path)
    monkeypatch.setattr(s5_plugin_probe, 'load_readiness', lambda path: readiness)
    output = tmp_path / 'bad-budget'
    budget = dict(workflow.BUDGET, max_calls=48)
    with pytest.raises(ValueError, match='frozen medium/32K budget'):
        workflow.freeze(output, readiness_report=readiness['report_path'],
            gpu=dict(index=2, uuid='GPU-12345678-1234-1234-1234-123456789abc', name='NVIDIA H20'),
            resource_policy=dict(mode='shared-known-processes', min_free_mib=8192,
                max_util_pct=5, allowed_compute_pids=[]), budget=budget)
    assert not output.exists()


def test_c_only_summary_preserves_unknown_and_never_makes_gain_claim():
    plan = dict(schema=workflow.C_ONLY_SCHEMA, hash='frozen-c-plan', arms=['C'],
        cases=[dict(case_id=name) for name in workflow.PREFLIGHT_CASES], repeats=1,
        planned_runs=3,
        schedule=workflow._case_order(
            [dict(case_id=name) for name in workflow.PREFLIGHT_CASES], 1,
            workflow.C_ONLY_ARMS), formal_s5_credit=False)
    records = [
        dict(case_id='matmul-tail-fp16', arm='C', repeat_id=1,
             plan_hash=plan['hash'], status='passed', execution_state='finished'),
        dict(case_id='layernorm-tail', arm='C', repeat_id=1,
             plan_hash=plan['hash'], status='unknown', execution_state='interrupted'),
    ]
    result = workflow.aggregate_c_diagnostic(plan, records)
    assert result['counts'] == dict(passed=1, failed=0, unknown=2)
    assert result['dispatched_runs'] == 2
    assert result['rows'][-1]['execution_state'] == 'not_executed'
    assert result['inference_allowed'] is False
    assert result['formal_s5_credit'] is False


def test_c_only_summary_rejects_a_foreign_record():
    plan = dict(schema=workflow.C_ONLY_SCHEMA, hash='frozen-c-plan', arms=['C'],
        cases=[dict(case_id=name) for name in workflow.PREFLIGHT_CASES], repeats=1,
        planned_runs=3,
        schedule=workflow._case_order(
            [dict(case_id=name) for name in workflow.PREFLIGHT_CASES], 1,
            workflow.C_ONLY_ARMS), formal_s5_credit=False)
    row = dict(case_id='foreign', arm='C', repeat_id=1, plan_hash=plan['hash'],
               status='passed', execution_state='finished')
    with pytest.raises(ValueError, match='record'):
        workflow.aggregate_c_diagnostic(plan, [row])


def plugin_entry_fixture():
    receipt = dict(ingress='operator_mcp', sequence=1,
        source_sha256='dev-source', service_response_sha256='dev-response',
        tool_result_sha256='tool-result')
    row = dict(case_id='layernorm-tail', repeat_id=1, arm='C', plan_hash='plugin-plan',
        status='passed', execution_state='finished', candidate_sha256='final-source',
        development_audit=dict(complete=True, plugin_verification_observed=True,
                               receipts=[receipt]),
        development=dict(records=[dict(phase='development', status='passed',
                                       response_sha256='dev-response')]),
        verification=dict(status='passed', full_l2_passed=True))
    plan = dict(schema=workflow.PLUGIN_ENTRY_SCHEMA, mode='plugin-entry',
        hash='plugin-plan', arms=['C'], cases=[dict(case_id='layernorm-tail')],
        planned_runs=1, schedule=[dict(case_id='layernorm-tail', repeat_id=1, arm='C')],
        formal_s5_credit=False)
    return plan, row


def test_plugin_entry_acceptance_requires_plugin_submission_feedback_and_final_pass():
    plan, row = plugin_entry_fixture()
    result = workflow.aggregate_plugin_entry(plan, [row])
    assert result['accepted'] is True
    assert result['status'] == 'passed'
    assert result['checks'] == dict(operator_verify_submitted=True,
        l2_feedback_returned_to_dsh=True, final_candidate_passed=True)
    assert result['development_receipts'][0]['source_sha256'] == 'dev-source'
    assert result['final_candidate_sha256'] == 'final-source'
    assert result['formal_s5_credit'] is False


def test_plugin_entry_accepts_legacy_passed_field_as_full_l2_verdict():
    plan, row = plugin_entry_fixture()
    row['verification'] = dict(status='passed', passed=True)
    result = workflow.aggregate_plugin_entry(plan, [row])
    assert result['accepted'] is True
    assert result['checks']['final_candidate_passed'] is True


def test_plugin_entry_does_not_override_explicit_unknown_final_l2():
    plan, row = plugin_entry_fixture()
    row['verification'] = dict(status='passed', passed=True, full_l2_passed=None)
    result = workflow.aggregate_plugin_entry(plan, [row])
    assert result['accepted'] is False
    assert result['checks']['final_candidate_passed'] is False


def repair_fixture():
    plan, row = plugin_entry_fixture()
    plan.update(mode='plugin-repair', schema=workflow.PLUGIN_REPAIR_SCHEMA)
    plan['cases'][0]['initial_candidate_sha256'] = 'seed'
    records = [dict(phase='development', sequence=1, status='failed',
                    source_sha256='seed', response_sha256='failed-feedback',
                    result=dict(passed=False, levels=dict(L1=True, L2=False))),
               dict(phase='development', sequence=2, status='passed',
                    source_sha256='final-source', response_sha256='passed-feedback')]
    row['development']['records'] = records
    row['development_audit']['receipts'] = [dict(ingress='operator_mcp',
        sequence=r['sequence'], source_sha256=r['source_sha256'],
        service_response_sha256=r['response_sha256']) for r in records]
    row['verification']['source_sha256'] = 'final-source'
    return plan, row


def test_repair_acceptance_requires_failed_seed_bound_feedback_revision_and_final():
    plan, row = repair_fixture()
    result = workflow.aggregate_plugin_entry(plan, [row])
    assert result['accepted'] is True
    assert result['schema'] == 's5-plugin-repair-acceptance-summary-v1'
    assert all(result['checks'].values())


@pytest.mark.parametrize('broken', ['seed', 'compile', 'receipt', 'revision', 'final', 'audit'])
def test_repair_rejects_missing_or_misbound_failure_repair_chain(broken):
    plan, row = repair_fixture()
    first, second = row['development']['records']
    if broken == 'seed': first['source_sha256'] = 'not-the-seed'
    if broken == 'compile': first['result']['levels']['L1'] = False
    if broken == 'receipt': row['development_audit']['receipts'][0]['service_response_sha256'] = 'foreign'
    if broken == 'revision': second['source_sha256'] = 'seed'
    if broken == 'final': row['verification']['source_sha256'] = 'foreign'
    if broken == 'audit': row['development_audit']['complete'] = False
    assert workflow.aggregate_plugin_entry(plan, [row])['accepted'] is False


def test_repair_final_extracts_single_code_block_and_preserves_verified_bytes():
    verified = 'def kernel(X):\n    return X'
    raw = 'L1 and L2 passed. Final source:\n\n```python\n' + verified + '\n```'
    source, normalization = workflow.extract_repair_final(raw, verified)
    assert source == verified
    assert normalization == 'verified_repair_single_source_newline_aligned'


@pytest.mark.parametrize('raw', [
    '```python\ndef kernel(X):\n    return X + 1\n```',
    '```python\ndef kernel(X):\n    return X\n```\n```python\nprint(1)\n```',
    'No code supplied',
    '```python\ndef kernel(X)\n    return X\n```',
])
def test_repair_final_refuses_changed_ambiguous_or_invalid_source(raw):
    with pytest.raises((ValueError, SyntaxError)):
        workflow.extract_repair_final(raw, 'def kernel(X):\n    return X')


@pytest.mark.parametrize('change', [
    {'development_audit': dict(complete=True, plugin_verification_observed=False, receipts=[])},
    {'development': dict(records=[dict(phase='development', status='unknown',
                                        response_sha256='dev-response')])},
    {'verification': dict(status='failed', full_l2_passed=False)},
])
def test_plugin_entry_acceptance_does_not_pass_without_all_three_evidence_links(change):
    plan, row = plugin_entry_fixture()
    row.update(change)
    result = workflow.aggregate_plugin_entry(plan, [row])
    assert result['accepted'] is False
    assert result['status'] == 'not_demonstrated'
    assert result['inference_allowed'] is False


@pytest.mark.parametrize('infrastructure_stop', [False, True])
def test_c_only_executor_dispatches_no_a_and_stops_on_infrastructure_unknown(
        tmp_path, monkeypatch, infrastructure_stop):
    output, plan = freeze_fixture(tmp_path, monkeypatch, workflow.C_ONLY_ARMS)
    monkeypatch.setattr(workflow, 'resources', lambda plan: dict(ready=True))
    called = []

    def run_one(plan, case, arm, output, api_key, *, repeat_id):
        called.append((case['case_id'], arm))
        stopped = infrastructure_stop and len(called) == 1
        return dict(case_id=case['case_id'], arm=arm, repeat_id=repeat_id,
                    plan_hash=plan['hash'], status='unknown' if stopped else 'passed',
                    execution_state='interrupted' if stopped else 'finished',
                    outcome_class='infrastructure_error' if stopped else 'completed')

    monkeypatch.setattr(workflow, 'run_one', run_one)
    result = workflow.execute(output / 'plan.json', plan['hash'],
                              tmp_path / 'execution', api_key='fake-key')
    assert all(arm == 'C' for _, arm in called)
    assert len(called) == (1 if infrastructure_stop else 3)
    assert result['counts'] == (dict(passed=0, failed=0, unknown=3)
                                if infrastructure_stop else dict(passed=3, failed=0, unknown=0))
    assert result['inference_allowed'] is False
    assert result['formal_s5_credit'] is False


def test_cli_preserves_ac_comparison_inference_flag(monkeypatch, capsys):
    monkeypatch.setattr(workflow, 'execute', lambda *args: dict(hash='summary',
        counts=dict(passed=6, failed=0, unknown=0),
        comparisons={'C-minus-A': dict(inference_allowed=True)}))
    workflow.main(['execute', '--plan', 'unused.json', '--plan-hash', 'frozen',
                   '--output', 'unused'])
    assert json.loads(capsys.readouterr().out)['inference_allowed'] is True


def test_study_freezes_exact_prospective_24_slots_and_shared_limits(tmp_path, monkeypatch):
    output, plan = freeze_fixture(tmp_path, monkeypatch, mode='study')
    loaded = workflow.load_plan(output / 'plan.json')
    assert loaded == plan
    assert plan['schema'] == workflow.STUDY_SCHEMA
    assert plan['status'] == 'frozen_study_candidate'
    assert plan['repeats'] == 2 and plan['planned_runs'] == 24
    assert plan['manifest_hash'] == workflow.STUDY_MANIFEST_HASH
    assert plan['model']['stream_idle_timeout_ms'] == 930000
    assert plan['model']['stream_idle_timeout_ms'] > plan['model']['request_wall_seconds'] * 1000
    assert plan['formal_readiness'] is None
    assert plan['formal_s5_credit'] is False
    assert plan['limits']['max_model_reservations'] == 576
    assert plan['limits']['max_output_reservations'] == 18874368
    assert plan['limits']['max_verifications'] == 72
    keys = {(s['case_id'], s['repeat_id'], s['arm']) for s in plan['schedule']}
    assert len(keys) == 24
    assert [slot['arm'] for slot in plan['schedule'][:4]] == list('ACCA')
    assert [slot['arm'] for slot in plan['schedule'][12:16]] == list('CAAC')
    assert all('initial_candidate' not in c and 'acceptance_instruction' not in c for c in plan['cases'])


def test_explicit_exploratory_plan_preserves_budget_and_limits_without_claiming_formal_pass(tmp_path, monkeypatch):
    from benchmarks.experiments.s5.s5_study_readiness import require_study_readiness, EXPLORATORY_POLICY
    output, plan = freeze_fixture(tmp_path, monkeypatch, mode='study', exploratory=True)
    assert workflow.load_plan(output / 'plan.json') == plan
    facts = require_study_readiness(plan)
    assert facts['execution_policy'] == EXPLORATORY_POLICY
    assert facts['formal_s5_credit'] is False
    assert all(value is False for value in facts['limitations'].values())
    assert plan['planned_runs'] == 24 and plan['budget'] == workflow.BUDGET


@pytest.mark.parametrize('change', ['claim_isolation', 'hash', 'policy'])
def test_exploratory_dispatch_requires_explicit_consistent_scope(tmp_path, monkeypatch, change):
    from benchmarks.experiments.s5.s5_study_readiness import require_study_readiness, FormalReadinessRefused
    _, plan = freeze_fixture(tmp_path, monkeypatch, mode='study', exploratory=True)
    if change == 'claim_isolation': plan['exploratory_limitations']['network_only_gateway'] = True
    if change == 'hash': plan['hash'] = 'foreign'
    if change == 'policy': plan['execution_policy'] = 'skip-all-checks'
    with pytest.raises(FormalReadinessRefused): require_study_readiness(plan)


@pytest.mark.parametrize('change', ['shape', 'precision', 'text', 'order', 'seed', 'repeat', 'schedule', 'limits', 'context', 'idle'])
def test_study_rejects_rehashed_task_or_schedule_drift(tmp_path, monkeypatch, change):
    output, plan = freeze_fixture(tmp_path, monkeypatch, mode='study')
    altered = copy.deepcopy(plan)
    if change == 'shape': altered['cases'][0]['shape'][0] += 1
    if change == 'precision': altered['cases'][0]['precision'] = 'fp32'
    if change == 'text': altered['cases'][0]['text'] += '\nForce operator_verify.'
    if change == 'order': altered['cases'][0]['order'] = ['C', 'A']
    if change == 'seed': altered['cases'][0]['initial_candidate'] = 'def kernel(X): return X'
    if change == 'repeat': altered['repeats'] = 1
    if change == 'schedule': altered['schedule'][0]['repeat_id'] = 2
    if change == 'limits': altered['limits']['max_model_reservations'] += 1
    if change == 'context': altered['model']['context_window'] = 16384
    if change == 'idle': altered['model']['stream_idle_timeout_ms'] = 300000
    altered.pop('hash')
    altered['hash'] = digest(altered)
    path = output / 'altered.json'
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='drift'):
        workflow.load_plan(path)


@pytest.mark.parametrize('change', ['gpu_index', 'gpu_uuid', 'host', 'timeout', 'backend'])
def test_study_resource_observation_and_verifier_must_use_same_frozen_device(tmp_path, monkeypatch, change):
    output, plan = freeze_fixture(tmp_path, monkeypatch, mode='study')
    altered = copy.deepcopy(plan)
    verification = altered['verification']
    if change == 'gpu_index': verification['gpu']['index'] = 7
    if change == 'gpu_uuid': verification['gpu']['uuid'] = 'GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
    if change == 'host': verification['host'] = 'other-host'
    if change == 'timeout': verification['timeout'] = 1800
    if change == 'backend': verification['backend'] = 'mock'
    altered.pop('hash')
    altered['hash'] = digest(altered)
    path = output / 'altered-verification.json'
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='GPU identity drift'):
        workflow.load_plan(path)


def test_study_formal_gate_refuses_before_resources_credentials_or_output(tmp_path, monkeypatch):
    from benchmarks.experiments.s5 import s5_study_readiness
    output, plan = freeze_fixture(tmp_path, monkeypatch, mode='study')
    def deny(plan): raise ValueError('formal readiness unavailable')
    monkeypatch.setattr(s5_study_readiness, 'require_formal_readiness', deny)
    monkeypatch.setattr(workflow, 'resources', lambda p: pytest.fail('GPU observation before gate'))
    from benchmarks.experiments.s5 import s5_workflow_preflight
    monkeypatch.setattr(s5_workflow_preflight, 'credential', lambda: pytest.fail('credential before gate'))
    with pytest.raises(ValueError, match='formal readiness'):
        workflow.execute(output / 'plan.json', plan['hash'], tmp_path / 'execution')
    assert not (tmp_path / 'execution').exists()


@pytest.mark.parametrize('mode', [None, 'ac', 'study'])
def test_direct_study_run_cannot_bypass_formal_gate_by_removing_mode(tmp_path, monkeypatch, mode):
    from benchmarks.experiments.s5.s5_study_readiness import FormalReadinessRefused
    _, plan = freeze_fixture(tmp_path, monkeypatch, mode='study')
    if mode is None: plan.pop('mode')
    else: plan['mode'] = mode
    monkeypatch.setattr(workflow.s5_development_service, 'DevelopmentService',
        lambda *a, **k: pytest.fail('service constructed before formal gate'))
    monkeypatch.setattr(workflow.s5_gateway_http, 'GatewayServer',
        lambda *a, **k: pytest.fail('gateway constructed before formal gate'))
    with pytest.raises(FormalReadinessRefused):
        workflow.run_one(plan, plan['cases'][0], 'A', tmp_path / 'execution', 'unused')
    assert not (tmp_path / 'execution').exists()


@pytest.mark.parametrize('stop_at', [None, 7])
def test_study_offline_executor_keeps_24_slots_and_stops_only_on_infrastructure(
        tmp_path, monkeypatch, stop_at):
    from benchmarks.experiments.s5 import s5_study_readiness
    output, plan = freeze_fixture(tmp_path, monkeypatch, mode='study')
    # This test proves scheduling only; bypass is explicit and no real transport is constructed.
    monkeypatch.setattr(s5_study_readiness, 'require_formal_readiness', lambda p: {'test_only': True})
    monkeypatch.setattr(workflow, 'resources', lambda p: dict(mock_only=True))
    dispatched = []
    def run_one(plan, case, arm, output, api_key, *, repeat_id):
        dispatched.append((case['case_id'], repeat_id, arm))
        index = len(dispatched)
        status = 'passed' if index % 3 == 0 else 'failed' if index % 3 == 1 else 'unknown'
        outcome = 'completed' if status == 'passed' else 'candidate_rejected' if status == 'failed' else 'call_budget_exhausted'
        if index == stop_at: status, outcome = 'unknown', 'infrastructure_error'
        return dict(case_id=case['case_id'], repeat_id=repeat_id, arm=arm,
            plan_hash=plan['hash'], status=status, execution_state='finished' if status != 'unknown' else 'interrupted',
            outcome_class=outcome, model_calls=dict(reserved=3, completed=3), wall_seconds=5,
            development_audit=dict(plugin_verification_observed=False))
    monkeypatch.setattr(workflow, 'run_one', run_one)
    result = workflow.execute(output / 'plan.json', plan['hash'], tmp_path / 'execution', api_key='fake-key')
    assert result['schema'] == 's5-plugin-study-summary-v1'
    assert result['planned'] == 24 and len(result['rows']) == 24
    assert len(dispatched) == (24 if stop_at is None else stop_at)
    assert len(set(dispatched)) == len(dispatched)
    assert len(list((tmp_path / 'execution/records').glob('*.json'))) == 24
    assert result['comparisons']['C-minus-A']['inference_allowed'] is (stop_at is None)
    assert result['formal_s5_credit'] is False
    assert sum(r['execution_state'] == 'not_executed' for r in result['rows']) == (0 if stop_at is None else 17)
    with pytest.raises(FileExistsError):
        workflow.execute(output / 'plan.json', plan['hash'], tmp_path / 'execution', api_key='fake-key')


@pytest.mark.parametrize('malformed', [False, True])
def test_study_run_one_uses_same_final_extraction_without_development_history_and_isolates_repeats(
        tmp_path, monkeypatch, malformed):
    import time
    from types import SimpleNamespace
    from benchmarks.experiments.s5 import s5_study_readiness
    from operator_agent.run_contract import code_hash
    output, plan = freeze_fixture(tmp_path, monkeypatch, mode='study')
    case = plan['cases'][0]
    contract_hash = case['service_contract']
    source = 'def kernel(X):\n    return X\n'
    raw = ('Done.\n\n```python\n' + source + '```' if not malformed else
           '```python\n' + source + '```\n```python\n' + source + '```')
    monkeypatch.setattr(s5_study_readiness, 'require_formal_readiness', lambda p: {'test_only': True})
    class Service:
        def __init__(self, plan_path, sha, path, **kwargs):
            self.plan = dict(contract_hash=contract_hash, declaration={}, mode='ssh', timeout=600)
            self.output, self.records, self.arm = path, [], kwargs['arm']
        def __enter__(self): return self
        def __exit__(self, *args): pass
        url, token = 'http://127.0.0.1:1/development/verify', 'fake-token'
        def final(self, code):
            assert code == source
            result = dict(status='passed', full_l2_passed=True,
                source_sha256=code_hash(code), contract_hash=contract_hash)
            self.records.append(dict(phase='final', sequence=1, status='passed',
                source_sha256=code_hash(code), completed_monotonic=time.monotonic(), result=result))
            return result
        def snapshot(self):
            return dict(records=self.records, development_used=0, final_used=len(self.records))
    class Context:
        def __init__(self, *args, **kwargs):
            self._server = SimpleNamespace(gateway=SimpleNamespace(last_budget_refusal=None))
        def __enter__(self): return self
        def __exit__(self, *args): pass
        base_url = 'http://127.0.0.1:1/v1'
        def failure_snapshot(self): return dict(schema='s5-upstream-failures-v1', failures=[], truncated=False)
        def snapshot(self): return dict(mock_only=True)
    monkeypatch.setattr(workflow.s5_development_service, 'DevelopmentService', Service)
    monkeypatch.setattr(workflow.s5_gateway_supervisor, 'SupervisedTransport', Context)
    monkeypatch.setattr(workflow.s5_gateway_http, 'GatewayServer', Context)
    monkeypatch.setattr(workflow.s5_plugin_rpc, 'MCPRelay', Context)
    monkeypatch.setattr(workflow.s5_verify_remote, 'load_plan', lambda *a: dict(declaration={}))
    monkeypatch.setattr(workflow, 'prepare', lambda *a: dict(contract_hash=contract_hash))
    monkeypatch.setattr(workflow.s5_plugin_env, 'build', lambda arm, **kw:
        dict(arm=arm, work=str(kw['arm_dir'] / 'work')))
    monkeypatch.setattr(workflow.s5_arm_env, 'start_headless', lambda *a, **k:
        dict(returncode=0, stderr='', events=[dict(type='final', text=raw),
             dict(phase='turn_end', reason=dict(kind='completed'))]))
    monkeypatch.setattr(workflow.s5_arm_env, 'cleanup_development', lambda *a: None)
    monkeypatch.setattr(workflow.s5_seatbelt, 'archive_work', lambda *a: dict(complete=True))
    execution = tmp_path / 'runs'
    for repeat in (1, 2):
        for arm in ('A', 'C'):
            row = workflow.run_one(plan, case, arm, execution, 'fake-key', repeat_id=repeat)
            path = execution / 'runs' / case['case_id'] / ('repeat-%d' % repeat) / arm
            assert (path / 'raw-answer.txt').read_text() == raw
            assert row['development']['development_used'] == 0
            assert row['outcome_class'] == ('no_submission' if malformed else 'completed')
            assert row['development']['final_used'] == (0 if malformed else 1)
            expected_id = 'plugin-' + digest([plan['hash'], case['case_id'], repeat, arm])[:24]
            assert workflow.shared.snapshot(path / 'budget.sqlite', expected_task_id=expected_id)['reserved_calls'] == 0
            if malformed:
                assert row['final_submission_refused'] is True
                assert not (path / 'candidate.txt').exists()
            else:
                assert (path / 'candidate.txt').read_text() == source
                assert row['candidate_sha256'] == code_hash(source)
            with pytest.raises(FileExistsError):
                workflow.run_one(plan, case, arm, execution, 'fake-key', repeat_id=repeat)
