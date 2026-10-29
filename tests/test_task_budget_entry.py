"""Finite production budgets reach the worker without starting a real worker."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from operator_agent import cli, hitl_tools as ht, mcp_server


DEFAULT = dict(wall_clock_s=1800.0, max_llm_calls=24, oracle_total=20)
EXPLICIT = dict(wall_clock_s=12.5, max_llm_calls=7, oracle_total=11)


@pytest.fixture
def launcher(monkeypatch):
    from operator_agent import exec_guard
    start = Mock(return_value=SimpleNamespace(pid=12345))
    monkeypatch.setattr(exec_guard, 'start_background_worker', start)
    monkeypatch.setattr(ht, '_llm_problem', lambda: None)
    return start


def declaration():
    return dict(request='relu fp32', shapes=[[7, 33]], precision='fp32')


@pytest.mark.parametrize('options, expected', [({}, DEFAULT), (EXPLICIT, EXPLICIT)])
def test_run_freezes_budget_and_returns_it_before_launch(tmp_path, launcher, options, expected):
    run = tmp_path/'run'
    response = ht.run_autonomous(declaration(), run, **options)
    config = json.loads((run/'autonomous.json').read_text())
    assert response['task_budget'] == config['task_budget'] == expected
    assert response['phase'] == 'running' and response['pid'] == 12345
    assert launcher.call_count == 1
    assert launcher.call_args.args[0][-1] == str(run)
    frozen = json.loads((run/'declaration.json').read_text())
    assert not set(expected).intersection(frozen)


@pytest.mark.parametrize('name,value', [
    ('wall_clock_s', True), ('wall_clock_s', 0), ('wall_clock_s', -1),
    ('wall_clock_s', float('nan')), ('wall_clock_s', float('inf')),
    ('wall_clock_s', '5'), ('wall_clock_s', None),
    ('max_llm_calls', True), ('max_llm_calls', 0), ('max_llm_calls', -1),
    ('max_llm_calls', 2.0), ('max_llm_calls', '2'),
    ('oracle_total', True), ('oracle_total', 0), ('oracle_total', -1),
    ('oracle_total', 3.5), ('oracle_total', None),
])
def test_bad_budget_is_refused_without_creating_run_or_launching(tmp_path, launcher, name, value):
    run = tmp_path/'run'
    with pytest.raises(ht.ToolError) as caught:
        ht.run_autonomous(declaration(), run, **{name: value})
    assert caught.value.code == 'invalid_task_budget'
    assert not run.exists() and not launcher.called


def test_mcp_forwards_budget_as_run_options_not_frozen_semantics(tmp_path, launcher):
    run = tmp_path/'run'
    response = mcp_server._call_tool('operator_run', dict(declaration(), run_dir=str(run), **EXPLICIT))
    result = json.loads(response['content'][0]['text'])
    assert result['task_budget'] == EXPLICIT and launcher.call_count == 1
    assert json.loads((run/'autonomous.json').read_text())['task_budget'] == EXPLICIT


@pytest.mark.parametrize('extra', [{'max_llm_callz': 7}, {'task_budget': EXPLICIT}])
def test_unknown_budget_options_are_not_silently_discarded(tmp_path, launcher, extra):
    with pytest.raises(ht.ToolError) as caught:
        mcp_server._call_tool('operator_run', dict(declaration(), run_dir=str(tmp_path/'run'), **extra))
    assert caught.value.code == 'unknown_parameter'
    assert not launcher.called


def test_mcp_schema_declares_finite_positive_budget_options():
    tool = next(tool for tool in mcp_server._tool_definitions() if tool['name'] == 'operator_run')
    properties = tool['inputSchema']['properties']
    assert not tool['inputSchema']['additionalProperties']
    assert set(DEFAULT).issubset(mcp_server._RUN_OPTION_KEYS)
    for key, value in DEFAULT.items():
        assert properties[key]['default'] == value
    assert properties['wall_clock_s']['exclusiveMinimum'] == 0
    assert properties['max_llm_calls']['minimum'] == properties['oracle_total']['minimum'] == 1
    assert ht.TOOL_VERSION == 9


def test_cli_forwards_all_three_budget_options(tmp_path, monkeypatch, capsys):
    run = Mock(return_value=dict(ok=True, task_budget=EXPLICIT))
    monkeypatch.setattr(cli, 'run_autonomous', run)
    result = cli.main(['run', '--request', 'relu fp32', '--shapes', '7,33',
        '--run-dir', str(tmp_path/'run'), '--wall-clock-s', '12.5',
        '--max-llm-calls', '7', '--oracle-total', '11'])
    assert result == 0
    assert {key: run.call_args.kwargs[key] for key in EXPLICIT} == EXPLICIT
    assert json.loads(capsys.readouterr().out)['task_budget'] == EXPLICIT


def test_cli_nonfinite_budget_is_a_structured_refusal(tmp_path, capsys):
    result = cli.main(['run', '--request', 'relu fp32', '--shapes', '7,33',
        '--run-dir', str(tmp_path/'run'), '--wall-clock-s', 'nan'])
    assert result == 2
    assert json.loads(capsys.readouterr().out)['error'] == 'invalid_task_budget'
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('owner', ['status', 'summary'])
def test_status_returns_budget_usage_and_stop_reason(tmp_path, owner):
    budget = dict(reserved_calls=7, remaining_calls=0, oracle_spent=2)
    audit = dict(task_budget=EXPLICIT, budget=budget, stopped_reason='budget_llm_calls')
    status = dict(phase='done', pid=12345)
    if owner == 'status':
        status.update(audit)
    else:
        (tmp_path/'conductor.json').write_text(json.dumps(dict(phase='done', **audit)))
    (tmp_path/'run-status.json').write_text(json.dumps(status))
    result = ht.autonomous_status(tmp_path)
    assert result['phase'] == 'done'
    assert result['task_budget'] == EXPLICIT and result['budget'] == budget
    assert result['stopped_reason'] == 'budget_llm_calls'


def test_legacy_status_does_not_invent_task_limits(tmp_path):
    (tmp_path/'run-status.json').write_text(json.dumps(dict(phase='done', pid=12345)))
    result = ht.autonomous_status(tmp_path)
    assert result['task_budget'] is result['budget'] is result['stopped_reason'] is None
