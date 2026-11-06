"""Host recovery reads the original task; it never creates new budget credit."""
import hashlib
import json
from pathlib import Path

import pytest

from operator_agent.app import hitl_tools as ht
from operator_agent.app.host_workflow import resume
from operator_agent.core.run_contract import code_hash
from operator_agent.execute import host_task_budget as hb

DECL=dict(request='relu fp32',shapes=[[4,8]],precision='fp32',dsl='python')
WRONG='import numpy as np\ndef kernel(X):\n    return X.copy()\n'
CORRECT='import numpy as np\ndef kernel(X):\n    return np.maximum(X,0)\n'


def prepared(tmp_path,**budget):
    run=tmp_path/'task';value=ht.prepare(DECL,run,host_budget=dict(max_verifications=3,**budget))
    return run,value


def test_fresh_task_resume_is_read_only_and_keeps_host_owned_user_context(tmp_path,monkeypatch):
    from operator_agent.execute import llm_client
    run,first=prepared(tmp_path)
    monkeypatch.setattr(llm_client,'LLMClient',lambda **kwargs:pytest.fail('resume called a model'))
    before={name:(run/name).read_bytes() for name in (hb.POLICY,hb.STATE,'declaration.json','contract.json','run.json')}
    value=resume(run,backend='fake')
    assert value['task_id']==first['host_task']['task_id'] and value['next_action']=='generate'
    assert value['host_task']['reserved']==0 and value['model_calls']==0
    assert value['next_context']==dict(run_dir=str(run),backend='fake',phase='generate')
    assert 'context' not in value and 'messages' not in value and 'records' not in value['host_task']
    assert value['user_instruction_owner']=='host_conversation_not_frozen_task_mutation'
    assert value['host_task']['limits']['max_model_calls']==12
    assert all((run/name).read_bytes()==data for name,data in before.items())
    assert not (run/'contexts').exists()


def test_resume_failed_candidate_returns_bound_repair_material_without_refund(tmp_path):
    run,_=prepared(tmp_path)
    result=ht.verify(run,WRONG,backend='fake',stop_after='L2',candidate_origin='outer')
    before=(run/hb.STATE).read_bytes()
    value=resume(run,backend='fake')
    assert value['next_action']=='repair' and value['selected_verification']==result['round']
    assert value['next_context']==dict(run_dir=str(run),backend='fake',phase='repair',verification_round=result['round'])
    assert value['candidate']['source_hash']==code_hash(WRONG)
    assert value['host_task']['reserved']==1 and (run/hb.STATE).read_bytes()==before
    assert value['latest_completed_verification']==result['round']


def test_resume_prefers_accepted_final_source_when_exploration_is_spent(tmp_path):
    run,_=prepared(tmp_path)
    correct=ht.verify(run,CORRECT,backend='fake',stop_after='L2')
    ht.verify(run,WRONG,backend='fake',stop_after='L2')
    value=resume(run,backend='fake')
    assert value['next_action']=='final_verify'
    assert value['latest_completed_verification']==2 and value['selected_verification']==correct['round']
    assert value['next_context'] is None
    assert value['candidate']['source_hash']==code_hash(CORRECT)
    final=ht.verify(run,CORRECT,backend='fake',stop_after='L2',verification_purpose='final')
    terminal=resume(run,backend='fake')
    assert terminal['next_action']=='report_final' and terminal['next_context'] is None
    assert terminal['host_task']['reserved']==3 and terminal['host_task']['remaining_final']==0
    assert terminal['candidate']['evidence']==final['evidence']


def test_resume_preserves_unknown_inflight_owner_without_stealing_it(tmp_path):
    run,first=prepared(tmp_path)
    budget=hb.HostTaskBudget(run,first['contract_hash'])
    budget.reserve(code_hash(CORRECT),owner_invocation_id='a'*32)
    before=(run/hb.STATE).read_bytes()
    value=resume(run,backend='fake')
    assert value['next_action']=='inspect_in_flight' and value['next_context'] is None
    assert value['host_task']['unfinished']==1 and value['unfinished_owners'][0]['owner_invocation_id']=='a'*32
    assert (run/hb.STATE).read_bytes()==before


def test_resume_refuses_contract_evidence_and_backend_drift(tmp_path):
    run,_=prepared(tmp_path)
    result=ht.verify(run,WRONG,backend='fake',stop_after='L2')
    with pytest.raises(ht.ToolError,match='verification_refused'):
        resume(run,backend='ssh')
    path=Path(result['source']);path.chmod(0o600);path.write_text(CORRECT)
    with pytest.raises(ht.ToolError,match='verification_refused'):
        resume(run,backend='fake')
    declaration=run/'declaration.json';value=json.loads(declaration.read_text());value['shapes']=[[8,8]]
    declaration.write_text(json.dumps(value))
    with pytest.raises(ht.ToolError,match='contract_drift'):
        resume(run,backend='fake')


def test_resume_does_not_upgrade_legacy_run_or_accept_new_contract_fields(tmp_path):
    run=tmp_path/'legacy';ht.prepare(DECL,run)
    with pytest.raises(ht.ToolError,match='host_task_required'):resume(run,backend='fake')
    run,_=prepared(tmp_path)
    before=hashlib.sha256((run/'contract.json').read_bytes()).hexdigest()
    with pytest.raises(TypeError):resume(run,backend='fake',shapes=[[8,8]])
    assert hashlib.sha256((run/'contract.json').read_bytes()).hexdigest()==before


def test_resume_rejects_uncompleted_round_and_reports_closed_expired_task(tmp_path,monkeypatch):
    run,_=prepared(tmp_path,wall_clock_s=10)
    with pytest.raises(ht.ToolError,match='verification_required'):
        resume(run,backend='fake',verification_round=1)
    monkeypatch.setattr(hb.HostTaskBudget,'_elapsed',staticmethod(lambda state:11.0))
    value=resume(run,backend='fake')
    assert value['next_action']=='report_stopped' and value['next_context'] is None
    assert value['host_task']['expired'] is True and value['host_task']['remaining_s']==0
