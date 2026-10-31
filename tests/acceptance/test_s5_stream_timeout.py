"""Scaled real DSH/Responses timeout tests using owned loopback services only."""
import hashlib
import json
from pathlib import Path
import time

import pytest

from benchmarks.experiments.s5 import s5_arm_env as arms
from benchmarks.experiments.s5 import s5_formal_run
from benchmarks.experiments.s5.s5_fake_responses_upstream import FakeResponsesUpstream
from benchmarks.experiments.s5.s5_gateway_http import GatewayServer
from benchmarks.experiments.s5.s5_gateway_supervisor import SupervisedTransport
from operator_agent.execute import s5_shared_budget as budget

READY = bool(arms.NODE) and (arms.DSH_ROOT / 'dsh/lib/bin.js').is_file()


def run_fixture(tmp_path, *, idle_ms, wall_seconds, header_delay=0, output_delay=0, truncate=None):
    ledger=tmp_path/'model.sqlite'
    budget.create(ledger,task_id='idle-fixture',max_calls=1,max_output_tokens=256,
                  total_output_tokens=256,wall_clock_s=60)
    with FakeResponsesUpstream(header_delay_seconds=header_delay,
            output_delay_seconds=output_delay,truncate_after=truncate) as upstream:
        transport=SupervisedTransport(upstream.url,'synthetic-only',wall_seconds=wall_seconds)
        with GatewayServer(ledger,task_id='idle-fixture',model='fake-model',participant='outer',
                adapter=transport,protocol='responses',max_calls_limit=1) as gateway:
            built=arms.build('A',base_url=gateway.base_url,arm_dir=tmp_path/'arm',
                api='responses',max_tokens=256,stream_idle_timeout_ms=idle_ms)
            Path(built['home']).mkdir(parents=True)
            Path(built['work']).mkdir()
            started=time.monotonic()
            response=arms.start_headless(built,'Reply OK.',timeout=20)
            elapsed=time.monotonic()-started
        # Gateway teardown closes the listener, not necessarily the active
        # handler. Observe bounded worker completion before inspecting receipts.
        for _ in range(100):
            snapshot=budget.snapshot(ledger)
            if snapshot['completed_calls']==1:break
            time.sleep(.02)
        diagnostics=transport.failure_snapshot()
        assert len(upstream.requests)==1
    result=dict(response=response,elapsed=elapsed,budget=snapshot,diagnostics=diagnostics,
        stream_idle_timeout_ms=idle_ms,request_wall_seconds=wall_seconds,
        real_model_calls=0,gpu_executions=0,
        adapter_sha256=hashlib.sha256((arms.DSH_ROOT/'dsh-llm-pi-ai/lib/index.js').read_bytes()).hexdigest())
    (tmp_path/'observation.json').write_text(json.dumps(result,indent=2)+'\n')
    assert snapshot['reserved_calls']==1  # no retry and no refund
    return result


@pytest.mark.skipif(not READY,reason='installed DSH required')
def test_short_adapter_idle_timeout_reproduces_host_exit_before_request_deadline(tmp_path):
    result=run_fixture(tmp_path,idle_ms=700,wall_seconds=5,output_delay=1.8)
    assert result['response']['returncode']!=0
    diagnostics=arms.headless_failure_diagnostics(result['response'])
    assert diagnostics['kind']=='adapter_stream_idle_timeout'
    assert diagnostics['idle_timeout_ms']==700
    assert diagnostics['upstream_root_cause']=='unknown'


@pytest.mark.skipif(not READY,reason='installed DSH required')
@pytest.mark.parametrize('delay_location',['headers','output'])
def test_aligned_adapter_allows_valid_silent_gap_inside_request_budget(tmp_path,delay_location):
    result=run_fixture(tmp_path,idle_ms=6000,wall_seconds=5,
        header_delay=1.8 if delay_location=='headers' else 0,
        output_delay=1.8 if delay_location=='output' else 0)
    assert result['response']['returncode']==0
    assert s5_formal_run._final_text(result['response']['events'])=='OK'
    assert result['diagnostics']['failures']==[]
    assert result['budget']['completed_calls']==1


@pytest.mark.skipif(not READY,reason='installed DSH required')
def test_gateway_deadline_still_terminates_stalled_request_before_adapter_idle(tmp_path):
    result=run_fixture(tmp_path,idle_ms=6000,wall_seconds=1.2,header_delay=5)
    assert result['response']['returncode']!=0
    assert result['elapsed']<6
    assert arms.headless_failure_diagnostics(result['response']) is None
    assert result['diagnostics']['failures'][0]['code']=='timeout'
    assert result['budget']['completed_calls']==1


@pytest.mark.skipif(not READY,reason='installed DSH required')
def test_disconnected_partial_stream_does_not_become_a_successful_answer(tmp_path):
    result=run_fixture(tmp_path,idle_ms=6000,wall_seconds=5,truncate=4)
    assert result['response']['returncode']!=0
    assert result['diagnostics']['failures']
    assert result['budget']['completed_calls']==1


@pytest.mark.parametrize('value',[0,-1,True,1.5,2147483648])
def test_invalid_idle_timeout_cannot_silently_disable_watchdog(tmp_path,value):
    with pytest.raises(ValueError,match='stream idle timeout'):
        arms.build('A',base_url='http://127.0.0.1:9/v1',arm_dir=tmp_path/'arm',
                   stream_idle_timeout_ms=value)


def test_known_host_timeout_diagnostic_is_not_a_claim_about_provider_root_cause():
    assert arms.headless_failure_diagnostics(dict(returncode=0,stderr='')) is None
    assert arms.headless_failure_diagnostics(dict(returncode=1,stderr='other error')) is None


def test_idle_timeout_is_an_infrastructure_fault_not_call_budget_exhaustion():
    from benchmarks.experiments.s5.s5_outcomes import classify
    record=dict(status='unknown',execution_state='interrupted',
        development_audit=dict(complete=True),development=dict(records=[]),
        gateway_diagnostics=dict(schema='s5-upstream-failures-v1',failures=[],truncated=False),
        headless_diagnostics=dict(kind='adapter_stream_idle_timeout',idle_timeout_ms=300000))
    assert classify(record)=='infrastructure_error'


def test_timeout_is_set_on_provider_profile_and_normal_default_is_unchanged(tmp_path):
    plain=arms.build('A',base_url='http://127.0.0.1:9/v1',arm_dir=tmp_path/'plain')
    changed=arms.build('A',base_url='http://127.0.0.1:9/v1',arm_dir=tmp_path/'changed',
                       stream_idle_timeout_ms=930000)
    def profile(built):
        return next(row for row in built['patch'] if row.get('id')=='llm-pi-ai')['config']['providers']['fixture']
    assert 'streamIdleTimeoutMs' not in profile(plain)
    assert profile(changed)['streamIdleTimeoutMs']==930000
