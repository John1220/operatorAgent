"""Frozen A/C driver contracts; no model, GPU or live verification calls."""
import copy
import json

import pytest

from benchmarks import s5_formal_run, s5_plugin_probe, s5_plugin_workflow as workflow
from operator_agent.run_contract import digest


def fake_readiness(tmp_path):
    return dict(report_path=str(tmp_path / 'readiness.json'), report_sha256='a' * 64,
        runtime=dict(schema='s5-plugin-runtime-v1', root=str(tmp_path / 'runtime'), files={}),
        runtime_files_hash='b' * 64, isolation_implementation={})


def freeze_fixture(tmp_path, monkeypatch, arms=workflow.ARMS, mode=None):
    readiness = fake_readiness(tmp_path)
    monkeypatch.setattr(s5_plugin_probe, 'load_readiness', lambda path: readiness)

    def freeze_service(path, **kwargs):
        path.mkdir(parents=True)
        return 'c' * 64

    monkeypatch.setattr(workflow.s5_verify_remote, 'freeze', freeze_service)
    monkeypatch.setattr(workflow.s5_verify_remote, 'load_plan',
        lambda path, sha: dict(declaration={}, contract_hash='d' * 64))
    monkeypatch.setattr(s5_formal_run, '_read', lambda path: {})
    monkeypatch.setattr(s5_formal_run, 'check_task_inputs', lambda row, contract: dict(operands=[]))
    output = tmp_path / 'frozen'
    policy = dict(mode='shared-known-processes', min_free_mib=8192, max_util_pct=5,
                  allowed_compute_pids=[])
    gpu = dict(index=2, uuid='GPU-12345678-1234-1234-1234-123456789abc', name='NVIDIA H20')
    mode = mode or ('ac' if arms == workflow.ARMS else 'c-only')
    plan = workflow.freeze(output, readiness_report=readiness['report_path'], gpu=gpu,
        resource_policy=policy, arms=arms,
        case_ids=(workflow.PLUGIN_ENTRY_CASES if mode in ('plugin-entry', 'plugin-repair')
                  else workflow.PREFLIGHT_CASES), mode=mode)
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
