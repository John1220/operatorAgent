"""Recalculate actual mock tool evidence and reject contradictory metadata."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from benchmarks.experiments.s5.s5_vertical_mcp_evidence import evaluate
from benchmarks.experiments.s5.s5_vertical_validation import GOOD_FIXTURE
from operator_agent.app.hitl_tools import prepare
from operator_agent.app.tool_calls import run_tool_call

TOOL='mcp__operator__operator_verify'


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    run=tmp_path_factory.mktemp('mcp-evidence')/'run'
    prepared=prepare({'request':'layernorm fp32','shapes':[[7,33]],'precision':'fp32'},run)
    args=dict(run_dir=str(run),code=GOOD_FIXTURE,backend='mock',stop_after='L2')
    payload=run_tool_call('operator_verify',args)
    call_dir=Path(payload['tool_call_dir'])
    raw=json.loads((call_dir/'result.json').read_text())['content'][0]['text']
    finished=json.loads((call_dir/'worker-result.json').read_text())['finished_monotonic']
    first,second='1'*64,'2'*64; cid='fixture-1'; body_hash=hashlib.sha256(raw.encode()).hexdigest()
    events=[dict(type='tool_call',callId=cid,tool=TOOL,input=args),
            dict(type='tool_result',callId=cid,status='completed',result=raw),
            dict(type='status',phase='step_start'),
            dict(type='status',phase='turn_end',reason={'kind':'completed'}),
            dict(type='final',text='fixture final')]
    calls=[dict(call_id='a'*32,participant='outer',outcome='answered',api_usage=None,
                request_sha256=first,reserved_monotonic=finished-10,
                dispatched_monotonic=finished-9,completed_monotonic=finished-5),
           dict(call_id='b'*32,participant='outer',outcome='answered',api_usage=None,
                request_sha256=second,reserved_monotonic=finished+1,
                dispatched_monotonic=finished+2,completed_monotonic=finished+4)]
    ledger=dict(schema='s5-shared-call-budget-v1',task_id='fixture',aborted=False,max_calls=2,
                reserved_calls=2,completed_calls=2,network_interception_proven=False,calls=calls)
    observations=dict(requests=[dict(request_sha256=first,observed_monotonic=finished-8,tool_feedback=[]),
                    dict(request_sha256=second,observed_monotonic=finished+3,
                         tool_feedback=[dict(tool_call_id=cid,content_sha256=body_hash)])])
    return events,ledger,observations,run,prepared['contract_hash']


def check(e):
    events,ledger,observations,run,contract=e
    return evaluate(events,ledger,observations,task_id='fixture',run_dir=run,
                    expected_tool_name=TOOL,expected_contract_hash=contract)


def test_real_mock_verifier_files_and_binding_are_accepted(evidence):
    assert check(evidence)['evidence_status']=='complete'


@pytest.mark.parametrize('target', ['source','backend','feedback','call_id','budget','contract','event'])
def test_unmatched_or_missing_records_remain_unknown(evidence,target):
    events,ledger,observations,run,contract=copy.deepcopy(evidence)
    if target=='source': events[0]['input']['code']='def kernel(X): return X'
    if target=='backend': events[0]['input']['backend']='ssh'
    if target=='feedback': observations['requests'][1]['tool_feedback'][0]['content_sha256']='f'*64
    if target=='call_id': events[1]['callId']='other'
    if target=='budget': ledger['calls'][1]['api_usage']={'prompt_tokens':0}
    if target=='contract': contract='f'*64
    if target=='event': events.pop(2)
    assert check((events,ledger,observations,run,contract))['evidence_status']=='unknown'


def test_changed_candidate_or_missing_receipt_member_is_unknown(evidence,monkeypatch):
    original=Path.read_bytes
    def replaced(self):
        if self.name=='result.json' and self.parent.name=='verify-1': return b'changed'
        return original(self)
    monkeypatch.setattr(Path,'read_bytes',replaced)
    assert check(evidence)['reason']=='evidence receipt hash mismatch'
