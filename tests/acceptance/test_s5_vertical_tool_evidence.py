"""Cross-check facts rather than trusting the fake model's final statement."""
import json
import hashlib
import pytest
from benchmarks.experiments.s5 import s5_vertical_tool_evidence as evidence
from benchmarks.experiments.s5.s5_vertical_validation import GOOD_FIXTURE
from operator_agent.core.run_contract import code_hash


def sample():
    source=code_hash(GOOD_FIXTURE); sha=lambda x:hashlib.sha256(x.encode()).hexdigest()
    contract,suite,manifest='1'*64,'2'*64,'3'*64
    ids=('layernorm-tail-primary','layernorm-tail-holdout-317','layernorm-tail-holdout-619')
    v=dict(schema='s5-vertical-mock-validation-v1',status='passed',candidate_sha256=source,
           verification_attempts=1,max_verification_attempts=1,mock_only=True,
           formal_s5_credit=False,gpu_used=False,untrusted_source_executed=False,
           contract_hash=contract,suite_hash=suite,draft_manifest_hash=manifest,
           mock_evidence=dict(full_l2_passed=True,source_hash=source,draft_contract_hash=contract,
                suite_hash=suite,inputs=[dict(case_id=x,full_l2_passed=True,source_hash=source) for x in ids]))
    response=evidence._canonical_response(v); response_hash=sha(response)
    request_hash=sha(json.dumps({'source':GOOD_FIXTURE},ensure_ascii=False,separators=(',',':')))
    validation=dict(sequence=1,start_monotonic=14.,end_monotonic=15.,request_sha256=request_hash,
                    response_sha256=response_hash,result=v)
    calls=[dict(call_id='a'*32,participant='outer',outcome='answered',api_usage=None,
                request_sha256='a'*64,reserved_monotonic=10.,dispatched_monotonic=11.,completed_monotonic=13.),
           dict(call_id='b'*32,participant='outer',outcome='answered',api_usage=None,
                request_sha256='b'*64,reserved_monotonic=16.,dispatched_monotonic=17.,completed_monotonic=19.)]
    ledger=dict(task_id='trial',schema='s5-shared-call-budget-v1',aborted=False,max_calls=2,
                reserved_calls=2,completed_calls=2,network_interception_proven=False,calls=calls)
    obs=dict(validation_requests=1,requests=[dict(request_sha256='a'*64,observed_monotonic=12.,tool_feedback=[]),
                       dict(request_sha256='b'*64,observed_monotonic=18.,tool_feedback=[
                           dict(tool_call_id='call_1',content_sha256=response_hash)])])
    events=[dict(type='tool_call',callId='call_1',tool='validate_kernel',input={'source':GOOD_FIXTURE}),
            dict(type='tool_result',callId='call_1',status='completed',result=response),
            dict(type='status',phase='step_start'),
            dict(type='status',phase='turn_end',reason={'kind':'completed'}),
            dict(type='final',text='untrusted claim')]
    return events,ledger,validation,obs


def check(data):
    return evidence.evaluate(*data,task_id='trial',expected_candidate_sha256=code_hash(GOOD_FIXTURE),
                             expected_tool_name='validate_kernel')


def test_cross_checks_complete_and_model_final_text_is_not_evidence():
    data=sample(); result=check(data)
    assert result['evidence_status']=='complete' and result['fake_calls']==2
    assert not result['formal_s5_credit'] and result['provider_usage'] is None
    data[0][-1]['text']='model could claim anything'
    assert check(data)['evidence_status']=='complete'


@pytest.mark.parametrize('mutator',[
    lambda e,l,v,o: l.update(reserved_calls=1),
    lambda e,l,v,o: o.update(validation_requests=2),
    lambda e,l,v,o: l['calls'][1].update(request_sha256='c'*64),
    lambda e,l,v,o: l['calls'][1].update(reserved_monotonic=14.5),
    lambda e,l,v,o: l['calls'][0].update(api_usage={'prompt_tokens':0}),
    lambda e,l,v,o: o['requests'][1]['tool_feedback'][0].update(tool_call_id='wrong'),
    lambda e,l,v,o: o['requests'][1]['tool_feedback'][0].update(content_sha256='0'*64),
    lambda e,l,v,o: v.update(response_sha256='0'*64),
    lambda e,l,v,o: v.update(request_sha256='0'*64),
    lambda e,l,v,o: v['result'].update(candidate_sha256='0'*64),
    lambda e,l,v,o: v['result'].update(verification_attempts=True),
    lambda e,l,v,o: v['result']['mock_evidence']['inputs'][1].update(full_l2_passed=None),
    lambda e,l,v,o: e[0].update(input={'source':'not a fixture'}),
    lambda e,l,v,o: e[1].update(callId='wrong'),
    lambda e,l,v,o: e[1].update(result='fake success'),
    lambda e,l,v,o: e.pop(2),
    lambda e,l,v,o: e[3].update(reason={'kind':'failed'}),
])
def test_incomplete_or_inconsistent_evidence_is_unknown(mutator):
    data=sample(); mutator(*data)
    assert check(data)['evidence_status']=='unknown'


def test_malformed_evidence_does_not_crash():
    assert check((None,{}, {}, {}))['evidence_status']=='unknown'
