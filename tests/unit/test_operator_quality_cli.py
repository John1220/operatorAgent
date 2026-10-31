"""Public quality commands select pure planning, readiness, or explicit replay."""
import json

import pytest

from benchmarks.experiments.misc import operator_quality as cli
from operator_agent.decide.quality_evaluation import make_plan
from operator_agent.decide.quality_pairing import make_paired_plan


def test_paired_plan_only_is_pure_and_published_exclusively(tmp_path, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('planning must not run a model or a device')
    monkeypatch.setattr(cli, 'run_paired_plan', forbidden)
    plan = tmp_path/'plan.json'
    assert cli.main(['--paired', '--case', 'layernorm-tail', '--output', str(plan)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(plan.read_text())
    assert json.loads(plan.read_text())['schema'] == 'operator-paired-quality-plan-v1'
    with pytest.raises(FileExistsError):
        cli.main(['--paired', '--case', 'layernorm-tail', '--output', str(plan)])


@pytest.mark.parametrize('paired', [False, True])
def test_execute_dispatches_from_frozen_schema(tmp_path, capsys, monkeypatch, paired):
    plan = (make_paired_plan if paired else make_plan)(case_ids=['layernorm-tail'])
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    seen = []
    def run(declaration, output, **kwargs):
        seen.append(declaration)
        return {'paired': paired}
    monkeypatch.setattr(cli, 'run_paired_plan' if paired else 'run_plan', run)
    assert cli.main(['--execute', str(path), '--output', str(tmp_path/'run')]) == 0
    assert seen == [plan]
    assert json.loads(capsys.readouterr().out) == {'paired': paired}


def test_paired_preflight_unwraps_validated_plan_without_execution(tmp_path, capsys, monkeypatch):
    plan = make_paired_plan(backend='ssh', host='synthetic-host', case_ids=['layernorm-tail'])
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    seen = []
    def preflight(declaration, output, **kwargs):
        seen.append((declaration, kwargs))
        return {'model_calls': 0}
    monkeypatch.setattr(cli, 'run_preflight', preflight)
    assert cli.main(['--preflight', str(path), '--host', 'synthetic-host',
                     '--output', str(tmp_path/'preflight')]) == 0
    assert seen == [(plan['base_plan'], {'host': 'synthetic-host'})]
    assert json.loads(capsys.readouterr().out)['model_calls'] == 0


def test_paired_replay_uses_phase_responses_and_cpu_full_suite(tmp_path, capsys):
    source = ('import numpy as np\ndef kernel(X):\n'
              '    z=X-X.mean(axis=-1,keepdims=True)\n'
              '    return z/np.sqrt((z*z).mean(axis=-1,keepdims=True)+1e-5)\n')
    replies = tmp_path/'replies.json'
    replies.write_text(json.dumps({'layernorm-tail': {
        'generation': [json.dumps(dict(code=source, rationale='Authored CPU control.'))]}}))
    assert cli.main(['--paired', '--replay', str(replies), '--case', 'layernorm-tail',
                     '--output', str(tmp_path/'run')]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['total_actual_costs']['model_calls']['known_total'] == 1
    assert all(report['arms'][arm]['metrics']['first_full_l2']['count'] == 1
               for arm in ('baseline', 'diagnosis'))


def test_paired_rejects_single_strategy_before_work(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(['--paired', '--strategy', 'baseline'])
    assert error.value.code == 2
    assert 'already fixes both strategies' in capsys.readouterr().err
