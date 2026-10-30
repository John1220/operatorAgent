"""Regression for task semantics and equal, bounded development feedback."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchmarks.experiments.s5 import s5_formal_run as workflow
from benchmarks.experiments.s5 import s5_verify_remote, s5_arm_env, s5_runtime
from benchmarks.experiments.s5.s5_development_service import DevelopmentService, encoded
from benchmarks.experiments.s5.s5_development_evidence import audit
from benchmarks.experiments.s5.s5_gateway_http import GatewayServer
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT
from benchmarks.experiments.s5.s5_vertical_tool_provider import _sse
from benchmarks.experiments.s5.s5_vertical_validation import BAD_FIXTURE, GOOD_FIXTURE
from operator_agent import hitl_tools, s5_verify_client as client, s5_shared_budget as shared
from operator_agent.run_contract import code_hash

DECLARATION = dict(request='layernorm fp32', shapes=[[7, 33]], hardware='mock_gpu',
                   dsl='python', precision='fp32')
READY = bool(shutil.which('node')) and (DSH_ROOT / 'dsh/lib/bin.js').is_file() and s5_runtime.available()


def service(tmp_path, **kwargs):
    sha = s5_verify_remote.freeze(tmp_path / 'plan', declaration=DECLARATION, backend='mock')
    return DevelopmentService(tmp_path / 'plan/plan.json', sha, tmp_path / 'feedback',
                              arm=kwargs.pop('arm', 'A'), **kwargs)


def env(service):
    return {client.PREFIX + k: str(v) for k, v in dict(URL=service.url, TOKEN=service.token,
            CONTRACT=service.plan['contract_hash'], BACKEND='mock', TIMEOUT=70).items()}


@pytest.mark.parametrize('row', workflow._quality_tasks()[1], ids=lambda r: r['case_id'])
def test_prompt_operands_match_the_prepared_service_primary(tmp_path, row):
    hitl_tools.prepare(workflow.declaration(row), tmp_path / 'task')
    contract = json.loads((tmp_path / 'task/contract.json').read_text())
    identity = workflow.check_task_inputs(row, contract)
    text = workflow.task_text(row)
    for operand in identity['operands']:
        shape = ', '.join(map(str, operand['shape']))
        assert (operand['name'] + '[' + shape + ']') in text or (
            operand['name'] == 'X' and '[' + shape + ']' in text)
    if row['kernel_type'] == 'matmul':
        bad = copy.deepcopy(contract)
        bad['contract']['cases'][0]['tensors'][0]['shape'] = row['shape']
        with pytest.raises(ValueError, match='disagree'):
            workflow.check_task_inputs(row, bad)
        bad = copy.deepcopy(contract)
        bad['contract']['cases'][0]['tensors'][0]['dtype'] = 'int32'
        with pytest.raises(ValueError, match='disagree'):
            workflow.check_task_inputs(row, bad)
    assert 'revise them from its feedback' in text
    assert 'run the kernel locally' in text


def test_old_plans_are_readable_for_analysis_but_never_reexecuted(tmp_path):
    with pytest.raises(ValueError, match='newly frozen'):
        workflow.run_one(dict(schema='s5-formal-plan-v1'), {}, 'A', tmp_path, 'unused')
    assert not (tmp_path / 'runs').exists()


def test_b_c_have_identical_usage_instructions():
    assert workflow.workflow_instructions('B', '/tmp/task', backend='ssh', max_development=2) == workflow.workflow_instructions('C', '/tmp/task', backend='ssh', max_development=2)


def test_failure_then_revision_and_separate_final_have_immutable_evidence(tmp_path):
    s = service(tmp_path)
    first = s.develop(BAD_FIXTURE, s.plan['contract_hash'])
    second = s.develop(GOOD_FIXTURE, s.plan['contract_hash'])
    assert first['status'] == 'failed' and first['feedback']
    assert second['status'] == 'passed'
    denied = s.develop(GOOD_FIXTURE, s.plan['contract_hash'])
    assert denied['error'] == 'development_budget_exhausted'
    final = s.final(GOOD_FIXTURE)
    assert final['status'] == 'passed' and final['phase'] == 'final'
    assert final['passed'] is True and final['full_l2_passed'] is True
    assert s.final(BAD_FIXTURE)['error'] == 'verification_closed'
    snapshot = s.snapshot()
    assert snapshot['development_used'] == 2 and snapshot['final_used'] == 1
    for index, record in enumerate(snapshot['records'], 1):
        d = s.output / ('%03d-%s' % (index, record['phase']))
        assert code_hash((d / 'candidate.py').read_text()) == record['source_sha256']
        assert hashlib.sha256(encoded(json.loads((d / 'response.json').read_text()))).hexdigest() == record['response_sha256']
        assert record['previous_sequence'] == (index - 1 or None)
        assert record['completed_monotonic'] >= record['started_monotonic']
    assert len(list(s.output.glob('*/evaluation/result.json'))) == 3
    with pytest.raises(FileExistsError):
        DevelopmentService(s.plan_path, s.plan_sha256, s.output, arm='A')


def test_unknown_consumes_quota_and_drift_never_gets_a_free_retry(tmp_path):
    s = service(tmp_path, max_development=1)
    unknown = s.develop('unreviewed source is never executed in mock mode', s.plan['contract_hash'])
    assert unknown['status'] == 'unknown'
    assert not list(s.output.glob('*/evaluation'))
    assert s.develop(GOOD_FIXTURE, s.plan['contract_hash'])['error'] == 'development_budget_exhausted'
    assert s.final(GOOD_FIXTURE)['status'] == 'passed'


def test_parallel_attempts_cannot_overspend(tmp_path):
    s = service(tmp_path, max_development=1)
    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(lambda _: s.develop('not executed', s.plan['contract_hash']), range(2)))
    assert sum(r.get('error') == 'development_budget_exhausted' for r in rows) == 1
    assert s.snapshot()['development_used'] == 1


def test_expired_development_does_not_spend_final_reserve(tmp_path):
    s = service(tmp_path)
    s.deadline = 0
    assert s.develop(GOOD_FIXTURE, s.plan['contract_hash'])['error'] == 'development_budget_exhausted'
    assert s.final(GOOD_FIXTURE)['status'] == 'passed'


def test_core_opt_in_never_falls_back_to_a_local_backend(tmp_path, monkeypatch):
    run = tmp_path / 'task'
    hitl_tools.prepare(DECLARATION, run)
    monkeypatch.setenv(client.PREFIX + 'URL', 'http://127.0.0.1:9/development/verify')
    monkeypatch.setattr(hitl_tools, '_make_backend', lambda *a, **k: pytest.fail('local backend reached'))
    with pytest.raises(hitl_tools.ToolError, match='incomplete'):
        hitl_tools.verify(run, GOOD_FIXTURE, backend='mock', stop_after='L2')
    for function, args in [(hitl_tools.repair, (run, GOOD_FIXTURE)),
                           (hitl_tools.bench, (run, [])),
                           (hitl_tools.run_autonomous, (DECLARATION, run))]:
        with pytest.raises(hitl_tools.ToolError) as exc:
            function(*args)
        assert exc.value.code == 's5_development_only'
    from operator_agent.mcp_server import _tool_definitions
    with pytest.raises(ValueError, match='incomplete'):
        _tool_definitions()


def test_response_binding_checked_and_transport_options_cannot_change_task(tmp_path, monkeypatch):
    with service(tmp_path) as s:
        for k, v in env(s).items(): monkeypatch.setenv(k, v)
        for kwargs in [dict(stop_after='L3', backend='mock'),
                       dict(stop_after='L2', backend='ssh'),
                       dict(stop_after='L2', backend='mock', host='another-host')]:
            with pytest.raises(ValueError):
                client.delegated_verify(tmp_path, GOOD_FIXTURE,
                    recorded_hash=s.plan['contract_hash'], **kwargs)
        with pytest.raises(ValueError):
            client.submit(GOOD_FIXTURE, contract_hash='0' * 64)
        assert s.snapshot()['development_used'] == 0
        assert client.submit(BAD_FIXTURE)['status'] == 'failed'
        assert client.submit(GOOD_FIXTURE)['status'] == 'passed'
        assert client.submit(GOOD_FIXTURE)['error'] == 'development_budget_exhausted'


def test_http_rejects_final_task_override_auth_and_duplicate_fields(tmp_path):
    import http.client
    from urllib.parse import urlsplit
    with service(tmp_path) as s:
        u = urlsplit(s.url)
        def post(body, path=u.path, token=s.token):
            c = http.client.HTTPConnection(u.hostname, u.port, timeout=5)
            try:
                c.request('POST', path, body=body, headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
                r = c.getresponse(); r.read(); return r.status
            finally: c.close()
        assert post(b'{}', token='wrong') == 403
        assert post(b'{}', path='/final') == 400
        assert post(b'{"code":"a","code":"b","contract_hash":"x"}') == 400
        assert post(encoded(dict(code=GOOD_FIXTURE, contract_hash=s.plan['contract_hash'], backend='h20'))) == 400
        assert post(encoded(dict(code=GOOD_FIXTURE, contract_hash='0' * 64))) == 400
        assert s.snapshot()['development_used'] == 0


def feedback_in_text(text):
    """Find the JSON projection inside either bash stdout or MCP tool content."""
    for i, char in enumerate(text):
        if char != '{': continue
        try: value, _ = json.JSONDecoder().raw_decode(text[i:])
        except ValueError: continue
        if isinstance(value, dict) and value.get('schema') == 's5-development-feedback-v1':
            return value
    raise ValueError('actual service feedback absent from next model request')


class RepairProvider:
    """Fake model: bad source -> real failed feedback -> good source -> pass."""
    def __init__(self, arm, task, svc):
        self.arm, self.task, self.service = arm, task, svc
        self.requests = []

    def __call__(self, request):
        number = len(self.requests) + 1
        record = dict(sequence=number)
        self.requests.append(record)
        if number > 1:
            replies = [m for m in request['messages'] if m.get('role') == 'tool']
            latest = replies[-1]
            record['reply_preview'] = str(latest.get('content'))[:2000]
            assert latest['tool_call_id'] == 'repair_' + str(number - 1)
            feedback = feedback_in_text(latest['content'])
            actual = self.service.snapshot()['records'][number - 2]
            record['raw_tool_reply_sha256'] = hashlib.sha256(latest['content'].encode()).hexdigest()
            # Real CLI/MCP supervisors add invocation_id/tool_call_dir. Bind
            # the service projection separately from these transport receipts.
            feedback = {k: feedback[k] for k in actual['result']}
            assert feedback == actual['result']
            record.update(feedback_sha256=hashlib.sha256(encoded(feedback)).hexdigest(),
                          source_sha256=feedback['source_sha256'])
            assert record['feedback_sha256'] == actual['response_sha256']
            assert feedback['status'] == ('failed' if number == 2 else 'passed'), feedback
        if number == 3:
            return _sse({'content': GOOD_FIXTURE}, 'stop')
        if number > 3: raise ValueError('unexpected fake request')
        source = BAD_FIXTURE if number == 1 else GOOD_FIXTURE
        if self.arm == 'C':
            name = 'mcp__operator__operator_verify'
            args = dict(run_dir=str(self.task), code=source, backend='mock', stop_after='L2')
        else:
            name = 'bash'
            command = ('s5-verify --code-file candidate.py' if self.arm == 'A' else
                       'operator-agent verify --run-dir ' + shlex.quote(str(self.task)) +
                       ' --code-file candidate.py --stop-after L2 --backend mock')
            args = dict(description='Verify the fixed mock candidate',
                        command='printf %s ' + shlex.quote(source) + ' > candidate.py && ' + command)
        call = dict(index=0, id='repair_' + str(number), type='function',
                    function=dict(name=name, arguments=json.dumps(args)))
        return _sse({'tool_calls': [call]}, 'tool_calls')


@pytest.mark.skipif(not READY, reason='installed DSH or frozen Python deps unavailable')
@pytest.mark.parametrize('arm', ['A', 'B', 'C'])
def test_actual_dsh_ingresses_receive_failure_then_pass_and_final_is_separate(tmp_path, arm):
    with service(tmp_path, arm=arm) as svc:
        workspace = tmp_path / 'arm'
        task = workspace / 'work/task'
        if arm != 'A': hitl_tools.prepare(DECLARATION, task)
        provider = RepairProvider(arm, task, svc)
        ledger = tmp_path / 'model.sqlite'
        shared.create(ledger, task_id='repair-' + arm, max_calls=3, max_output_tokens=2048,
                      total_output_tokens=6144, wall_clock_s=120)
        with GatewayServer(ledger, task_id='repair-' + arm, model='fake-model',
                           participant='outer', adapter=provider, max_calls_limit=3) as gateway:
            conf = dict(url=svc.url, token=svc.token, contract=svc.plan['contract_hash'],
                        backend='mock', timeout=70)
            built = s5_arm_env.build(arm, base_url=gateway.base_url, arm_dir=workspace,
                                    tools=True, max_tokens=2048, development=conf)
            Path(built['home']).mkdir(parents=True)
            Path(built['work']).mkdir(parents=True, exist_ok=True)
            response = s5_arm_env.start_headless(built, 'Use the supervised tools to develop the fixed test kernel.', timeout=90)
        (tmp_path / 'debug.json').write_text(json.dumps(dict(response=response, requests=provider.requests)))
        assert response['returncode'] == 0, dict(response=response, requests=provider.requests,
                                                development=svc.snapshot())
        assert len(provider.requests) == 3
        assert shared.snapshot(ledger)['reserved_calls'] == 3
        assert svc.snapshot()['development_used'] == 2
        assert svc.snapshot()['final_used'] == 0
        events = response['events']
        calls = [e for e in events if e.get('type') == 'tool_call']
        assert len(calls) == 2
        assert all(e['tool'] == ('mcp__operator__operator_verify' if arm == 'C' else 'bash') for e in calls)
        proof = audit(events, svc.snapshot(), arm=arm)
        assert proof['complete'], proof
        assert proof['plugin_verification_observed'] is (arm == 'C')
        tampered = copy.deepcopy(events)
        reply = next(e for e in tampered if e.get('type') == 'tool_result')
        reply['result'] = reply['result'].replace('"status": "failed"', '"status": "passed"')
        assert audit(tampered, svc.snapshot(), arm=arm)['complete'] is False
        final = workflow._final_text(events)
        assert code_hash(final) == code_hash(GOOD_FIXTURE)
        assert svc.final(final)['status'] == 'passed'
        assert svc.snapshot()['final_used'] == 1
        (tmp_path / 'workflow-audit.json').write_text(json.dumps(proof, indent=2, sort_keys=True))
        (tmp_path / 'verification-budget.json').write_text(json.dumps(svc.snapshot(), indent=2, sort_keys=True))
        assert len(provider.requests) == 3  # final verdict never goes back to fake model
        assert svc.token not in json.dumps(workflow.redacted_launch(built))
        assert not (workspace / 'patch.json').exists()
        assert not (workspace / 'development-client.json').exists()
        if arm == 'A':
            assert 'PYTHONPATH' not in built['env']
            assert built['cli_shim'] is None


def test_workflow_freeze_binds_semantics_quotas_and_new_execution_identity(tmp_path):
    plan = workflow.freeze(tmp_path / 'frozen')  # freeze never dispatches model/GPU
    assert plan['schema'] == workflow.SCHEMA
    assert plan['implementation'] == workflow.runtime_identity()
    assert plan['development']['max_development'] == 2
    assert plan['development']['final_reserve'] == 1
    assert plan['model']['reasoning_effort'] == 'medium'
    assert plan['model']['context_window'] == 1000000
    assert plan['budget']['max_output_tokens'] == 32768
    assert plan['budget']['total_output_tokens'] == 24 * 32768
    assert len(plan['cases']) == 10
    assert all(c['input_semantics']['operands'] for c in plan['cases'])
    assert workflow.load_plan(tmp_path / 'frozen/plan.json') == plan


def test_run_one_uses_supervised_final_and_preserves_early_exit_wall_time(tmp_path, monkeypatch):
    sha = s5_verify_remote.freeze(tmp_path / 'services/case', declaration=DECLARATION, backend='mock')
    plan = dict(schema=workflow.SCHEMA, implementation=workflow.runtime_identity(), hash='test-plan',
        budget=dict(max_calls=3, max_output_tokens=2048, total_output_tokens=6144, wall_clock_s=120),
        model=dict(endpoint='http://127.0.0.1:9/v1/responses', id='fake-model', reasoning_effort='none'),
        development=dict(max_development=2))
    case = dict(case_id='case', service_sha256=sha, text='fixed mock task')
    monkeypatch.setattr(s5_arm_env, 'start_headless', lambda *a, **k: dict(returncode=0, stderr='',
        events=[dict(type='final', text=GOOD_FIXTURE)]))
    result = workflow.run_one(plan, case, 'A', tmp_path, 'placeholder')
    assert result['status'] == 'passed'
    assert result['development']['final_used'] == 1
    assert result['development']['development_used'] == 0
    assert result['candidate_sha256'] == code_hash(GOOD_FIXTURE)
    assert result['model_calls']['reserved'] == 0
    assert result['gateway_diagnostics']['calls_created'] == 0
    assert result['gateway_diagnostics']['failures'] == []
    assert (tmp_path / 'runs/case/A/gateway-diagnostics.json').is_file()
    assert result['real_model_called'] is None  # no unsupported claim from configuration alone
    assert result['wall_seconds'] > 0
    monkeypatch.setattr(s5_arm_env, 'start_headless', lambda *a, **k: dict(returncode=1, stderr='error', events=[]))
    result = workflow.run_one(plan, case, 'B', tmp_path, 'placeholder')
    assert result['status'] == 'unknown' and result['execution_state'] == 'interrupted'
    assert result['wall_seconds'] is not None
    assert result['development']['final_used'] == 0


@pytest.mark.skipif(not READY, reason='installed DSH or frozen Python deps unavailable')
def test_medium_reasoning_and_32k_reach_the_actual_responses_upstream(tmp_path, monkeypatch):
    from benchmarks.experiments.s5.s5_fake_responses_upstream import FakeResponsesUpstream
    from benchmarks.experiments.s5.s5_gateway_supervisor import SupervisedTransport
    from benchmarks.experiments.s5.s5_gateway_policy import GatewayPolicy
    observed = []
    original_request = GatewayPolicy._request_responses
    def observe(self, request, participant):
        observed.append(request)
        return original_request(self, request, participant)
    monkeypatch.setattr(GatewayPolicy, '_request_responses', observe)
    ledger = tmp_path / 'model.sqlite'
    shared.create(ledger, task_id='medium-32k', max_calls=1, max_output_tokens=32768,
                  total_output_tokens=32768, wall_clock_s=120)
    with FakeResponsesUpstream(text='OK') as upstream:
        transport = SupervisedTransport(upstream.url, 'fake-upstream-key',
            wall_seconds=workflow.REQUEST_WALL_SECONDS, max_response_bytes=workflow.RESPONSE_BYTES)
        with GatewayServer(ledger, task_id='medium-32k', model='fake-model',
            participant='outer', adapter=transport, max_calls_limit=1, protocol='responses',
            reasoning_effort='medium', max_response_bytes=workflow.RESPONSE_BYTES) as gateway:
            built = s5_arm_env.build('A', base_url=gateway.base_url, arm_dir=tmp_path / 'arm',
                api='responses', max_tokens=32768, context_window=workflow.CONTEXT_WINDOW,
                reasoning_effort='medium')
            Path(built['home']).mkdir(parents=True)
            Path(built['work']).mkdir(parents=True)
            response = s5_arm_env.start_headless(built, 'Reply OK.', timeout=45)
    assert response['returncode'] == 0, dict(response=response,
        observed=[{k: v for k, v in row.items() if k not in ('input', 'tools')} for row in observed])
    assert workflow._final_text(response['events']) == 'OK'
    assert len(upstream.requests) == 1
    assert upstream.requests[0]['max_output_tokens'] == 32768
    assert upstream.requests[0]['reasoning'] == {'effort': 'medium', 'summary': 'auto'}
    assert shared.snapshot(ledger)['reserved_output_tokens'] == 32768
    (tmp_path / 'model-request-metadata.json').write_text(json.dumps(dict(
        reasoning=upstream.requests[0]['reasoning'],
        max_output_tokens=upstream.requests[0]['max_output_tokens'],
        model=upstream.requests[0]['model'], real_provider_call=False, gpu_used=False), indent=2))
    with pytest.raises(ValueError, match='context window'):
        s5_arm_env.build('A', base_url='http://127.0.0.1:9/v1',
                        arm_dir=tmp_path / 'invalid', max_tokens=32768, context_window=16384)


def test_long_reasoning_transport_limits_remain_bounded(tmp_path):
    from benchmarks.experiments.s5.s5_gateway_supervisor import SupervisedTransport
    from benchmarks.experiments.s5.s5_gateway_transport import MAX_RESPONSE_BYTES, MAX_WALL_SECONDS
    transport = SupervisedTransport('http://127.0.0.1:9/v1/responses', 'placeholder',
        wall_seconds=900, max_response_bytes=64 * 1024 * 1024)
    assert transport.max_response_bytes == workflow.RESPONSE_BYTES
    with pytest.raises(ValueError):
        SupervisedTransport('http://127.0.0.1:9/v1/responses', 'placeholder',
                            wall_seconds=MAX_WALL_SECONDS + 1)
    with pytest.raises(ValueError):
        SupervisedTransport('http://127.0.0.1:9/v1/responses', 'placeholder',
                            max_response_bytes=MAX_RESPONSE_BYTES + 1)


@pytest.mark.parametrize('extra', [dict(include=['unknown']),
    dict(include=['reasoning.encrypted_content', 'reasoning.encrypted_content']),
    dict(reasoning=dict(effort='medium', summary='unbounded')),
    dict(reasoning=dict(effort='medium', extra=True))])
def test_reasoning_extensions_still_reject_unknown_fields_before_spending(tmp_path, extra):
    from benchmarks.experiments.s5.s5_gateway_policy import GatewayPolicy, PolicyRefused
    path = tmp_path / 'budget.sqlite'
    shared.create(path, task_id='reasoning', max_calls=1, max_output_tokens=32768,
                  total_output_tokens=32768, wall_clock_s=120)
    policy = GatewayPolicy(path, task_id='reasoning', model='fake', protocol='responses',
        adapter=lambda _: pytest.fail('invalid request reached upstream'), reasoning_effort='medium')
    request = dict(model='fake', stream=True, store=False, input='test', max_output_tokens=32768)
    request.update(extra)
    with pytest.raises(PolicyRefused):
        list(policy.stream(request, participant='outer'))
    assert shared.snapshot(path)['reserved_calls'] == 0
