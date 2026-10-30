"""Offline checks for the Responses-protocol budget gateway path."""
import json
from pathlib import Path
import shutil

import pytest

from benchmarks.experiments.s5 import s5_arm_env
from benchmarks.experiments.s5 import s5_responses_turn as turn
from benchmarks.experiments.s5.s5_fake_responses_upstream import FakeResponsesUpstream
from benchmarks.experiments.s5.s5_gateway_http import GatewayServer
from benchmarks.experiments.s5.s5_gateway_policy import GatewayPolicy, PolicyRefused
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT
from operator_agent import s5_shared_budget as shared

DSH_READY = bool(shutil.which('node')) and (DSH_ROOT / 'dsh' / 'lib' / 'bin.js').is_file()
TASK_ID = 'responses-fixture'


def _ledger(tmp_path, **limits):
    path = tmp_path / 'ledger.sqlite'
    options = dict(max_calls=2, max_output_tokens=256, total_output_tokens=512, wall_clock_s=120)
    options.update(limits)
    shared.create(path, task_id=TASK_ID, **options)
    return path


def _agent_request(**overrides):
    request = dict(model='fake-model', stream=True, store=False, max_output_tokens=256,
                   input=[dict(role='user', content=[dict(type='input_text', text='hi')])],
                   prompt_cache_key='session-1',
                   tools=[dict(type='function', name='exit_plan_mode', parameters={})])
    request.update(overrides)
    return request


def _drain(policy, request, participant='outer'):
    return list(policy.stream(request, participant=participant))


def test_policy_accepts_the_agent_request_shape_and_binds_output(tmp_path):
    ledger = _ledger(tmp_path)
    policy = GatewayPolicy(ledger, task_id=TASK_ID, model='fake-model', protocol='responses',
                           adapter=lambda request: [b'data: {}\n\n'])
    assert _drain(policy, _agent_request()) == [b'data: {}\n\n']
    snapshot = shared.snapshot(ledger, expected_task_id=TASK_ID)
    assert snapshot['reserved_calls'] == 1 and snapshot['completed_calls'] == 1


@pytest.mark.parametrize('overrides', [
    dict(max_output_tokens=None),
    dict(model='other-model'),
    dict(store=True),
    dict(prompt_cache_key=7),
    dict(tools={'name': 'x'}),
    dict(input=[]),
    dict(unexpected_field=1),
])
def test_policy_refuses_unbounded_or_unexpected_requests(tmp_path, overrides):
    ledger = _ledger(tmp_path)
    policy = GatewayPolicy(ledger, task_id=TASK_ID, model='fake-model', protocol='responses',
                           adapter=lambda request: [b'ok'])
    request = _agent_request()
    request.update(overrides)
    with pytest.raises(PolicyRefused):
        _drain(policy, request)
    assert shared.snapshot(ledger, expected_task_id=TASK_ID)['reserved_calls'] == 0


def test_second_request_is_refused_once_the_shared_budget_is_spent(tmp_path):
    ledger = _ledger(tmp_path, max_calls=1, total_output_tokens=256)
    policy = GatewayPolicy(ledger, task_id=TASK_ID, model='fake-model', protocol='responses',
                           adapter=lambda request: [b'ok'])
    _drain(policy, _agent_request())
    with pytest.raises(PolicyRefused):
        _drain(policy, _agent_request())
    assert shared.snapshot(ledger, expected_task_id=TASK_ID)['reserved_calls'] == 1


def test_arm_paths_are_absolute_so_dsh_can_find_the_patch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    built = s5_arm_env.build('A', base_url='http://127.0.0.1:9/v1',
                             arm_dir=Path('relative-arm'), api='responses')
    assert Path(built['home']).is_absolute() and Path(built['work']).is_absolute()


def test_arm_model_id_propagates_to_the_provider_patch(tmp_path):
    built = s5_arm_env.build('A', base_url='http://127.0.0.1:9/v1',
                             arm_dir=tmp_path / 'arm', api='responses',
                             model='deepseek-test-model', max_tokens=256)
    patch = json.dumps(built['patch'])
    assert 'deepseek-test-model' in patch and '"maxTokens": 256' in patch
    assert 'fake-model' not in patch


def test_gateway_serves_only_the_responses_path(tmp_path):
    import urllib.error
    import urllib.request
    ledger = _ledger(tmp_path)
    with GatewayServer(ledger, task_id=TASK_ID, model='fake-model', participant='outer',
                       adapter=lambda request: [b'data: {"type":"response.completed"}\n\n'],
                       protocol='responses') as gateway:
        body = json.dumps(_agent_request()).encode()
        request = urllib.request.Request(gateway.base_url + '/responses', data=body,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 200
        wrong = urllib.request.Request(gateway.base_url + '/chat/completions', data=body,
                                       headers={'Content-Type': 'application/json'})
        with pytest.raises(urllib.error.HTTPError) as failed:
            urllib.request.urlopen(wrong, timeout=10)
        assert failed.value.code == 400


@pytest.mark.skipif(not DSH_READY, reason='installed DSH headless unavailable')
def test_one_agent_turn_passes_through_the_gateway(tmp_path):
    with FakeResponsesUpstream(text='OK') as upstream:
        result = turn.run(tmp_path / 'turn', upstream_url=upstream.url, api_key='upstream-key')
    assert result['status'] == 'passed'
    assert result['model_calls']['reserved'] == 1 and result['model_calls']['completed'] == 1
    assert result['upstream_framing'] == 'answered'
    assert result['usage_evidence'] == 'missing' and result['provider_usage'] is None
    assert result['real_provider_call'] is False and result['gpu_used'] is False
    events = json.loads((tmp_path / 'turn' / 'headless-events.json').read_text())
    finals = [event['text'] for event in events if event.get('type') == 'final']
    assert finals == ['OK']
    assert len(upstream.requests) == 1
    sent = upstream.requests[0]
    assert sent['model'] == 'fake-model' and sent['max_output_tokens'] == 256
    assert sent['stream'] is True and sent['store'] is False
    assert sent['authorization'] == 'Bearer upstream-key'
    assert sent['tool_count'] >= 1


@pytest.mark.skipif(not DSH_READY, reason='installed DSH headless unavailable')
@pytest.mark.parametrize('mode', ['truncated', 'failed'])
def test_incomplete_upstream_never_passes(tmp_path, mode):
    options = dict(truncate_after=5) if mode == 'truncated' else dict(terminal='failed')
    with FakeResponsesUpstream(text='OK', **options) as upstream:
        result = turn.run(tmp_path / mode, upstream_url=upstream.url, api_key='upstream-key')
    assert result['status'] == 'unknown'
    assert result['upstream_framing'] == 'unknown'
    assert result['usage_evidence'] == 'missing' and result['provider_usage'] is None
    assert result.get('final_text_sha256') is None


def test_frozen_plan_refuses_rerun_and_tampering(tmp_path):
    directory = tmp_path / 'plan'
    with FakeResponsesUpstream() as upstream:
        sha = turn.freeze(directory, endpoint=upstream.url)
        assert len(sha) == 64
        with pytest.raises(FileExistsError):
            turn.freeze(directory, endpoint=upstream.url)
        with pytest.raises(ValueError):
            turn.run_planned(tmp_path / 'never', directory / 'plan.json', 'f' * 64,
                             api_key='upstream-key')
        plan = json.loads((directory / 'plan.json').read_text())
        plan['max_output_tokens'] = 2048
        (directory / 'plan.json').write_text(json.dumps(plan))
        with pytest.raises(ValueError):
            turn.run_planned(tmp_path / 'never', directory / 'plan.json', sha,
                             api_key='upstream-key')
    assert not (tmp_path / 'never').exists()


@pytest.mark.skipif(not DSH_READY, reason='installed DSH headless unavailable')
def test_planned_turn_runs_once_against_the_fixture(tmp_path):
    with FakeResponsesUpstream(text='OK') as upstream:
        sha = turn.freeze(tmp_path / 'run', endpoint=upstream.url)
        result = turn.run_planned(tmp_path / 'run' / 'exec', tmp_path / 'run' / 'plan.json',
                                  sha, api_key='upstream-key', upstream_kind='fixture')
    assert result['status'] == 'passed' and result['upstream_framing'] == 'answered'
    assert result['model_calls']['reserved'] == 1
    with pytest.raises(FileExistsError):
        turn.run_planned(tmp_path / 'run' / 'exec', tmp_path / 'run' / 'plan.json', sha,
                         api_key='upstream-key', upstream_kind='fixture')


@pytest.mark.skipif(not DSH_READY, reason='installed DSH headless unavailable')
def test_client_credential_and_prompt_never_reach_the_upstream(tmp_path):
    with FakeResponsesUpstream(text='OK') as upstream:
        result = turn.run(tmp_path / 'cred', upstream_url=upstream.url, api_key='upstream-key')
    assert result['status'] == 'passed'
    launch = json.loads((tmp_path / 'cred' / 'launch.json').read_text())
    assert launch['env']['S5_FAKE_API_KEY'] == 'placeholder'
    for row in upstream.requests:
        assert row['authorization'] == 'Bearer upstream-key'
        assert 'placeholder' not in json.dumps(row)
    recorded = (tmp_path / 'cred' / 'budget.json').read_text()
    assert 'upstream-key' not in recorded and 'placeholder' not in recorded