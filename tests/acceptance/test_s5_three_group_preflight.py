"""Offline and local checks for the S5 three-group preflight wiring."""
import json
import shutil

import pytest

from benchmarks.experiments.s5 import s5_arm_env, s5_preflight_plan, s5_runtime, s5_three_group_run, s5_verify_service
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT
from benchmarks.experiments.s5.s5_vertical_validation import BAD_FIXTURE, GOOD_FIXTURE

DSH_READY = bool(shutil.which('node')) and (DSH_ROOT / 'dsh' / 'lib' / 'bin.js').is_file()


def _plan(**kwargs):
    return s5_preflight_plan.build(**kwargs)


def _patch_ids(patch):
    ids = []
    for entry in patch:
        if entry.get('id'):
            ids.append(entry['id'])
        ids.extend(item.get('id') for item in entry.get('insert', []) if item.get('id'))
    return ids


def _record(arm, *, case_id='layernorm-tail', status='passed', state='finished', passed=True):
    return dict(schema=s5_three_group_run.SCHEMA_RECORD, arm=arm, case_id=case_id,
                status=status, execution_state=state,
                final_l2=dict(full_l2_passed=passed, source_sha256='a' * 64,
                              contract_hash='b' * 64, verdict_sha256='c' * 64),
                budget=dict(reserved=1, completed=1, answered=1, aborted=False, usage_present=0),
                probe=dict(core_importable=arm != 'A', cli_on_path=arm != 'A',
                           repo_canary_readable=True, matches_expectation=True))


def test_plan_hash_is_stable_and_persisted_once(tmp_path):
    first, second = _plan(), _plan()
    assert first['hash'] == second['hash']
    assert first['arms'] == ['A', 'B', 'C'] and first['repeats'] == 1
    path = tmp_path / 'plan.json'
    s5_preflight_plan.write(first, path)
    with pytest.raises(FileExistsError):
        s5_preflight_plan.write(second, path)
    loaded = s5_preflight_plan.load(path)
    assert loaded['hash'] == first['hash']
    tampered = json.loads(path.read_text())
    tampered['budget']['max_model_calls'] = 99
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        s5_preflight_plan.load(path)


def test_preflight_plan_never_claims_the_formal_denominator():
    plan = _plan()
    assert plan['preflight'] is True
    assert plan['formal_s5_credit'] is False
    assert plan['counts_toward_denominator'] is False
    assert plan['status'] == 'preflight_not_formal'
    with pytest.raises(ValueError):
        _plan(repeats=2)
    with pytest.raises(ValueError):
        _plan(arms=('A', 'C', 'B'))
    with pytest.raises(ValueError):
        _plan(case_id='not-a-manifest-task')


def test_task_text_matches_the_earlier_vertical_runs():
    from benchmarks.experiments.s5.s5_vertical_trial import TASK as vertical_task
    assert s5_preflight_plan.TASK == vertical_task


def test_dependency_freeze_includes_the_supplied_numpy_distribution_metadata(tmp_path):
    site = tmp_path / 'site'; (site / 'numpy').mkdir(parents=True)
    (site / 'numpy/__init__.py').write_text('__version__ = "9.8.7"\n')
    metadata = site / 'numpy-9.8.7.dist-info'; metadata.mkdir()
    (metadata / 'METADATA').write_text('Name: numpy\nVersion: 9.8.7\n')
    facts = s5_runtime.materialize(site, tmp_path / 'frozen')
    assert facts['copied'] == ['numpy', 'numpy-9.8.7.dist-info']
    assert (tmp_path / 'frozen/numpy-9.8.7.dist-info/METADATA').read_bytes() == (metadata / 'METADATA').read_bytes()
    (site / 'numpy-1.0.0.dist-info').mkdir()
    with pytest.raises(ValueError, match='exactly one numpy distribution'):
        s5_runtime.materialize(site, tmp_path / 'ambiguous')


def test_arm_surfaces_differ_only_by_installed_components(tmp_path):
    plans = {}
    for arm in ('A', 'B', 'C'):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        plans[arm] = s5_arm_env.build(arm, base_url='http://127.0.0.1:9/v1', arm_dir=arm_dir)
    ids = {arm: _patch_ids(built['patch']) for arm, built in plans.items()}
    assert 'mcp-operator' not in ids['A'] and 'operator-tools' not in ids['A']
    assert 'mcp-operator' not in ids['B'] and 'operator-tools' not in ids['B']
    assert 'mcp-operator' in ids['C'] and 'operator-tools' in ids['C']
    assert 'PYTHONPATH' not in plans['A']['env']
    assert plans['B']['env']['PYTHONPATH'] == plans['C']['env']['PYTHONPATH']
    assert plans['A']['cli_shim'] is None and plans['B']['cli_shim']
    assert plans['C']['patch_sha256'] != plans['B']['patch_sha256']
    assert plans['A']['surface']['core_cli'] is False
    for arm in ('A', 'B', 'C'):
        disabled = {entry['id'] for entry in plans[arm]['patch'] if entry.get('disabled')}
        assert {'tool-bash', 'tool-fs', 'tool-subagent', 'web'} <= disabled


def test_probe_is_recorded_as_environment_fact_not_isolation(tmp_path):
    arm_dir = tmp_path / 'probe-A'
    built = s5_arm_env.build('A', base_url='http://127.0.0.1:9/v1', arm_dir=arm_dir)
    facts = s5_arm_env.probe(built)
    assert facts['isolation_verified'] is False
    assert facts['linux_host_verified'] is False
    assert facts['core_importable'] is False and facts['cli_on_path'] is False
    assert facts['matches_expectation'] is True
    # On a shared host the same-UID arm can still read the repository; the
    # harness records that instead of turning the env setup into a claim.
    assert facts['repo_canary_readable'] is True
    with pytest.raises(ValueError):
        s5_arm_env.surface('D')


def test_verification_refuses_unplanned_or_duplicate_submission(tmp_path):
    plan = _plan()
    with pytest.raises(s5_verify_service.VerificationRefused):
        s5_verify_service.submit(plan, arm='D', candidate_text=GOOD_FIXTURE,
                                 run_dir=tmp_path / 'x')
    with pytest.raises(s5_verify_service.VerificationRefused):
        s5_verify_service.submit(plan, arm='A', candidate_text='  ',
                                 run_dir=tmp_path / 'y')
    target = tmp_path / 'once'
    s5_verify_service.submit(plan, arm='A', candidate_text=GOOD_FIXTURE, run_dir=target)
    with pytest.raises(s5_verify_service.VerificationRefused):
        s5_verify_service.submit(plan, arm='A', candidate_text=GOOD_FIXTURE, run_dir=target)
    assert json.loads((target / 'submission.json').read_text())['candidate_sha256']


def test_mock_backend_passes_and_fails_without_formal_credit(tmp_path):
    plan = _plan()
    good = s5_verify_service.submit(plan, arm='B', candidate_text=GOOD_FIXTURE,
                                    run_dir=tmp_path / 'good')
    assert good['status'] == 'passed' and good['full_l2_passed'] is True
    assert good['mock_only'] is True and good['formal_s5_credit'] is False
    assert good['backend'] == 'mock' and good['remote_backend_executed'] is False
    assert good['contract_hash'] == plan['contract_hash']
    bad = s5_verify_service.submit(plan, arm='B', candidate_text=BAD_FIXTURE,
                                   run_dir=tmp_path / 'bad')
    assert bad['status'] == 'failed' and bad['full_l2_passed'] is False
    assert s5_verify_service.load_result(tmp_path / 'bad' / 'result.json')['status'] == 'failed'


def test_unwired_real_backend_stays_unknown(tmp_path):
    plan = _plan(verify_backend='h20')
    result = s5_verify_service.submit(plan, arm='C', candidate_text=GOOD_FIXTURE,
                                      run_dir=tmp_path / 'h20')
    assert result['status'] == 'unknown' and result['full_l2_passed'] is None
    assert result['mock_only'] is False and result['remote_backend_executed'] is False
    assert 'unknown' in result['reason']
    assert not (tmp_path / 'h20' / 'result.json').read_text().count('"passed"')


def test_identity_drift_is_refused_rather_than_trusted(tmp_path):
    plan = json.loads(json.dumps(_plan()))
    plan['contract_hash'] = 'd' * 64
    result = s5_verify_service.submit(plan, arm='A', candidate_text=GOOD_FIXTURE,
                                      run_dir=tmp_path / 'drift')
    assert result['status'] == 'unknown'
    assert 'contract identity drift' in result['reason']


def test_verify_service_rejects_a_plan_that_claims_formal_credit(tmp_path):
    plan = json.loads(json.dumps(_plan()))
    plan['formal_s5_credit'] = True
    with pytest.raises(s5_verify_service.VerificationRefused):
        s5_verify_service.submit(plan, arm='A', candidate_text=GOOD_FIXTURE,
                                 run_dir=tmp_path / 'claim')


def test_aggregate_is_pure_and_requires_every_arm():
    plan = _plan()
    records = [_record(arm) for arm in plan['arms']]
    first = s5_three_group_run.aggregate(plan, records)
    second = s5_three_group_run.aggregate(plan, list(reversed(records)))
    assert first['hash'] == second['hash']
    assert first['counts'] == dict(passed=3, failed=0, unknown=0)
    assert first['wiring_complete'] is True
    assert first['formal_s5_credit'] is False and first['counts_toward_denominator'] is False
    with pytest.raises(ValueError):
        s5_three_group_run.aggregate(plan, records[:2])
    with pytest.raises(ValueError):
        s5_three_group_run.aggregate(plan, records + [records[0]])


def test_aggregate_keeps_unknown_and_foreign_tasks_out_of_passes():
    plan = _plan()
    records = [_record('A', status='unknown', state='interrupted', passed=None),
               _record('B'), _record('C')]
    summary = s5_three_group_run.aggregate(plan, records)
    assert summary['counts'] == dict(passed=2, failed=0, unknown=1)
    assert summary['execution_states'] == dict(finished=2, interrupted=1, unknown=0)
    assert summary['wiring_complete'] is False
    assert summary['rows'][0]['final_l2']['full_l2_passed'] is None
    bad = [_record('A', case_id='other-task'), _record('B'), _record('C')]
    with pytest.raises(ValueError):
        s5_three_group_run.aggregate(plan, bad)


def test_summary_rows_keep_the_shape_the_offline_summary_expects():
    plan = _plan()
    summary = s5_three_group_run.aggregate(plan, [_record(arm) for arm in plan['arms']])
    for row in summary['rows']:
        assert row['case_id'] == plan['case_id']
        assert row['execution_state'] == 'finished'
        assert set(row['final_l2']) >= {'full_l2_passed', 'source_sha256', 'contract_hash'}
        assert row['api_usage'] is None


def test_arm_failure_is_recorded_as_unknown_not_a_crash(tmp_path, monkeypatch):
    plan = _plan()
    monkeypatch.setattr(s5_three_group_run, '_headless',
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('boom')))
    record = s5_three_group_run.run_arm(plan, 'B', tmp_path / 'B')
    assert record['status'] == 'unknown'
    assert record['execution_state'] == 'interrupted'
    assert record['error'] == 'RuntimeError' and record['final_l2'] is None
    assert record['formal_s5_credit'] is False and record['gpu_used'] is False
    assert (tmp_path / 'B' / 'budget.json').is_file()


@pytest.mark.skipif(not DSH_READY or not s5_runtime.available(),
                    reason='installed DSH or frozen runtime deps unavailable')
def test_three_arm_preflight_end_to_end(tmp_path):
    plan = _plan()
    plan_path = tmp_path / 'plan.json'
    s5_preflight_plan.write(plan, plan_path)
    output = tmp_path / 'run'
    summary = s5_three_group_run.run(plan_path, output)
    assert summary['counts'] == dict(passed=3, failed=0, unknown=0)
    assert summary['execution_states'] == dict(finished=3, interrupted=0, unknown=0)
    assert summary['wiring_complete'] is True
    assert summary['formal_s5_credit'] is False
    verdicts = {row['final_l2']['verdict_sha256'] for row in summary['rows']}
    assert len(verdicts) == 1
    for arm in ('A', 'B', 'C'):
        record = json.loads((output / 'records' / ('arm-' + arm + '.json')).read_text())
        assert record['surface']['core_cli'] == (arm != 'A')
        assert record['surface']['plugin'] == (arm == 'C')
        assert record['probe']['matches_expectation'] is True
        assert record['probe']['repo_canary_readable'] is True
        assert record['isolation_verified'] is False
        assert record['usage_evidence'] == 'missing' and record['api_usage'] is None
        assert record['budget']['reserved'] == 1 and record['budget']['answered'] == 1
        assert record['evidence']['provider_requests'] == 1
        assert record['gpu_used'] is False and record['real_model_called'] is False
    again = s5_three_group_run.aggregate(
        plan, [json.loads((output / 'records' / ('arm-' + arm + '.json')).read_text())
               for arm in plan['arms']])
    assert again['hash'] == summary['hash']
    with pytest.raises(FileExistsError):
        s5_three_group_run.run(plan_path, output)
