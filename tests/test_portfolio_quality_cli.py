"""The public portfolio command freezes three explicit generation slots."""
import copy
import json

import pytest

from benchmarks import operator_quality as cli
from operator_agent import quality_evaluation as q
from operator_agent.quality_pairing import make_paired_plan
from operator_agent.quality_portfolio import make_portfolio_plan


CASE = 'layernorm-tail'
SCHEMA = 'operator-portfolio-quality-plan-v1'


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return str(path)


def test_portfolio_plan_is_pure_defaults_to_all_ten_cases_and_saves_exclusively(
        tmp_path, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('portfolio planning must not create a model, device, or concrete inputs')
    monkeypatch.setattr(cli, 'run_portfolio_plan', forbidden)
    monkeypatch.setattr(q, 'LLMClient', forbidden)
    monkeypatch.setattr(q.quality_suite, 'evaluation_inputs_for', forbidden)
    output = tmp_path / 'plan.json'
    assert cli.main(['--portfolio', '--output', str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan == json.loads(output.read_text())
    assert plan['schema'] == SCHEMA
    assert len(plan['base_plan']['manifest']['cases']) == 10
    assert plan['base_plan']['controls']['budget']['max_calls'] == 1
    assert plan['base_plan']['controls']['budget']['max_verifications'] == 1
    assert plan['portfolio']['repair_calls'] == 0
    assert [slot['strategy'] for slot in plan['portfolio']['slots']] == [
        'layout_first', 'numerics_first', 'tiled']
    with pytest.raises(FileExistsError):
        cli.main(['--portfolio', '--output', str(output)])


def test_portfolio_plan_preserves_selected_case_order_and_remote_controls(capsys, monkeypatch):
    monkeypatch.setenv('LLM_BASE_URL', 'https://model.invalid/v1')
    assert cli.main(['--portfolio', '--model', 'declared-model', '--backend', 'ssh',
                     '--host', 'synthetic-host', '--case', CASE,
                     '--case', 'matmul-strided-fp32']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [row['case_id'] for row in plan['base_plan']['manifest']['cases']] == [
        CASE, 'matmul-strided-fp32']
    assert plan['base_plan'] == make_portfolio_plan(
        model='declared-model', endpoint='https://model.invalid/v1', backend='ssh',
        host='synthetic-host', case_ids=[CASE, 'matmul-strided-fp32'])['base_plan']


@pytest.mark.parametrize('extra', [
    ['--paired'], ['--strategy', 'baseline'], ['--strategy', 'diagnosis'],
    ['--initial-source', 'layernorm-tail=missing.py'],
    ['--compare', 'missing-left.json', 'missing-right.json'],
])
def test_portfolio_rejects_conflicting_controls_before_reading_or_writing(
        tmp_path, capsys, extra):
    output = tmp_path / 'not-created'
    with pytest.raises(SystemExit) as error:
        cli.main(['--portfolio', *extra, '--output', str(output)])
    assert error.value.code == 2
    assert '--portfolio fixes three generation strategies' in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize('flag', [[], ['--portfolio']])
def test_execute_dispatches_frozen_portfolio_schema_with_optional_flag(
        tmp_path, capsys, monkeypatch, flag):
    plan = make_portfolio_plan(case_ids=[CASE])
    plan_path = write_json(tmp_path / 'plan.json', plan)
    output = str(tmp_path / 'run')
    seen = []
    def run(declaration, directory, **kwargs):
        seen.append((declaration, directory, kwargs))
        return {'portfolio_execution_dispatched': True}
    def forbidden(*args, **kwargs):
        pytest.fail('portfolio plans must not dispatch to ordinary or repair-paired runners')
    monkeypatch.setattr(cli, 'run_portfolio_plan', run)
    monkeypatch.setattr(cli, 'run_plan', forbidden)
    monkeypatch.setattr(cli, 'run_paired_plan', forbidden)
    assert cli.main([*flag, '--execute', plan_path, '--output', output,
                     '--host', 'synthetic-host']) == 0
    assert seen == [(plan, output, {'host': 'synthetic-host'})]
    assert json.loads(capsys.readouterr().out) == {'portfolio_execution_dispatched': True}


@pytest.mark.parametrize('action', ['--execute', '--preflight'])
@pytest.mark.parametrize('factory', [q.make_plan, make_paired_plan])
def test_portfolio_flag_rejects_a_different_frozen_plan_before_work(
        tmp_path, capsys, monkeypatch, action, factory):
    plan_path = write_json(tmp_path / 'other-plan.json', factory(case_ids=[CASE]))
    def forbidden(*args, **kwargs):
        pytest.fail('mismatched flag must fail before invoking any runner')
    for name in ('run_plan', 'run_paired_plan', 'run_portfolio_plan', 'run_preflight'):
        monkeypatch.setattr(cli, name, forbidden)
    output = tmp_path / 'not-created'
    with pytest.raises(SystemExit) as error:
        cli.main(['--portfolio', action, plan_path, '--output', str(output)])
    assert error.value.code == 2
    assert '--portfolio requires a portfolio frozen plan' in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize('flag', [[], ['--portfolio']])
def test_portfolio_preflight_validates_and_unwraps_without_generation(
        tmp_path, capsys, monkeypatch, flag):
    plan = make_portfolio_plan(backend='ssh', host='synthetic-host', case_ids=[CASE])
    plan_path = write_json(tmp_path / 'plan.json', plan)
    output = str(tmp_path / 'preflight')
    seen = []
    def preflight(declaration, directory, **kwargs):
        seen.append((declaration, directory, kwargs))
        return {'model_calls': 0}
    def forbidden(*args, **kwargs):
        pytest.fail('preflight must not generate or verify candidates')
    monkeypatch.setattr(cli, 'run_preflight', preflight)
    monkeypatch.setattr(cli, 'run_portfolio_plan', forbidden)
    assert cli.main([*flag, '--preflight', plan_path, '--host', 'synthetic-host',
                     '--output', output]) == 0
    assert seen == [(plan['base_plan'], output, {'host': 'synthetic-host'})]
    assert json.loads(capsys.readouterr().out) == {'model_calls': 0}


def test_portfolio_preflight_rejects_tampered_slot_contract_before_backend(
        tmp_path, monkeypatch):
    plan = copy.deepcopy(make_portfolio_plan(case_ids=[CASE]))
    plan['portfolio']['slots'][1]['strategy'] = 'hidden-extra-candidate'
    plan_path = write_json(tmp_path / 'tampered-plan.json', plan)
    def forbidden(*args, **kwargs):
        pytest.fail('tampered portfolio contract must fail before preflight')
    monkeypatch.setattr(cli, 'run_preflight', forbidden)
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError):
        cli.main(['--preflight', plan_path, '--output', str(output)])
    assert not output.exists()


def test_portfolio_replay_dispatches_three_raw_responses_per_case(
        tmp_path, capsys, monkeypatch):
    responses = {CASE: [json.dumps({'code': 'def kernel(X):\n    return X\n',
                                   'rationale': f'Candidate {slot}'})
                        for slot in (1, 2, 3)]}
    responses_path = write_json(tmp_path / 'responses.json', responses)
    seen = []
    def run(plan, directory, **kwargs):
        seen.append((plan, directory, kwargs))
        return {'portfolio_replay_dispatched': True}
    def forbidden(*args, **kwargs):
        pytest.fail('portfolio replay must not dispatch to repair-paired or ordinary runners')
    monkeypatch.setattr(cli, 'run_portfolio_plan', run)
    monkeypatch.setattr(cli, 'run_plan', forbidden)
    monkeypatch.setattr(cli, 'run_paired_plan', forbidden)
    output = str(tmp_path / 'run')
    assert cli.main(['--portfolio', '--replay', responses_path, '--case', CASE,
                     '--output', output]) == 0
    assert len(seen) == 1
    plan, directory, kwargs = seen[0]
    assert plan['schema'] == SCHEMA
    assert [row['case_id'] for row in plan['base_plan']['manifest']['cases']] == [CASE]
    assert directory == output
    assert kwargs == {'responses': responses}
    assert json.loads(capsys.readouterr().out) == {'portfolio_replay_dispatched': True}


def test_portfolio_replay_runs_all_three_slots_and_selects_the_first_full_l2_pass(
        tmp_path, capsys, monkeypatch):
    bad = 'def kernel(X):\n    return X * 0\n'
    good = ('import numpy as np\ndef kernel(X):\n'
            '    z = X.astype(np.float32)\n'
            '    z = z - z.mean(axis=-1, keepdims=True)\n'
            '    return (z / np.sqrt((z*z).mean(axis=-1, keepdims=True)+1e-5)).astype(X.dtype)\n')
    responses = {CASE: [json.dumps(dict(code=source, rationale='Authored CPU control.'))
                        for source in (bad, good, good + '# Third slot still runs.\n')]}
    responses_path = write_json(tmp_path / 'responses.json', responses)
    def forbidden(*args, **kwargs):
        pytest.fail('explicit portfolio replay must not construct a live model or repair a slot')
    monkeypatch.setattr(q, 'LLMClient', forbidden)
    monkeypatch.setattr(q, '_repair_candidate', forbidden)
    output = tmp_path / 'run'
    assert cli.main(['--portfolio', '--replay', responses_path, '--case', CASE,
                     '--output', str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['metrics']['baseline_full_l2']['count'] == 0
    assert report['metrics']['portfolio_full_l2']['count'] == 1
    assert report['metrics']['rescued_failed_baselines'] == 1
    assert report['costs']['model_calls']['known_total'] == 3
    assert report['tasks'][0]['selected_slot'] == 2
    assert [row['full_l2_passed'] for row in report['candidates']] == [False, True, True]
    assert [row['model_calls'] for row in report['candidates']] == [1, 1, 1]
    for slot in (1, 2, 3):
        assert (output / 'cases' / CASE / f'slot-{slot}' / 'verification-1.json').is_file()


@pytest.mark.parametrize('action', ['--execute', '--preflight', '--replay'])
def test_portfolio_execution_actions_require_explicit_new_output_directory(capsys, action):
    with pytest.raises(SystemExit) as error:
        cli.main(['--portfolio', action, 'missing.json'])
    assert error.value.code == 2
    assert 'requires --output NEW_DIRECTORY' in capsys.readouterr().err
