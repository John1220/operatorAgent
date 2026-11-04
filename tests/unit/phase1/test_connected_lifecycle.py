"""Supervised failures and read-only reconciliation must never grant false credit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.experiments.phase1 import connected, connected_plan, connected_summary
from benchmarks.experiments.phase1.plan import read, write_once
from operator_agent.app import hitl_tools as ht
from operator_agent.execute import s5_shared_budget as shared
from operator_agent.execute.llm_client import LLMClient
from operator_agent.execute.s5_inner_budget import SharedProductionBudget
from operator_agent.execute.task_budget import TaskBudgetConfig
from operator_agent.execute.task_clock import monotonic

CODE='import numpy as np\ndef kernel(X):\n    return np.maximum(X,0)\n'
DECLARATION=dict(request='relu fp32',precision='fp32',dsl='python',shapes=[[4,8]])


def replace_json(path,value):
    if path.exists():path.unlink()
    write_once(path,value)


@pytest.fixture
def cell(tmp_path,monkeypatch):
    monkeypatch.setattr(connected_plan,'implementation_identity',lambda: {'stable':'fixture'})
    config=dict(backend='fake',entry_b='core',model=dict(base_url='http://127.0.0.1:43210/v1',
        id='local-fixture',api_key_env='PHASE1_TEST_KEY',transport='local_fixture',temperature=.2),
        tasks=[dict(task_id='relu',declaration=DECLARATION)],max_rounds=1,
        budget=dict(wall_clock_s=30.,max_llm_calls=4,oracle_total=20),final_wall_clock_s=20.)
    plan=connected_plan.freeze(tmp_path/'frozen',config)
    root=tmp_path/'execution';directory=root/'cells/relu-A';directory.mkdir(parents=True)
    slot=plan['schedule'][0];task=plan['tasks'][0]
    cfg=connected.configuration(plan,task,slot,directory,monotonic())
    return dict(root=root,path=directory,plan=plan,slot=slot,task=task,config=cfg)


def outcome(started,deadline,*,returncode=0,stopped_reason=None):
    return dict(returncode=returncode,stopped_reason=stopped_reason,
        started_monotonic=started,deadline_monotonic=deadline,finished_monotonic=monotonic(),
        elapsed_s=0.,cleanup=dict(status='stopped',worker_reaped=True))


@pytest.mark.parametrize('failure,phase,reason',[
    ('nonzero','failed','final_worker_failed'),('missing','failed','final_worker_failed'),
    ('late','failed','budget_wall_clock'),('cancel','cancelled','user_cancelled')])
def test_cell_propagates_independent_final_failure_and_cancel(cell,monkeypatch,failure,phase,reason):
    calls=[]
    def supervise(argv,directory,deadline,root,invocation):
        calls.append(argv)
        cfg=read(argv[-1])
        if argv[3].endswith('connected_worker'):
            work=Path(cfg['run_dir']);work.mkdir()
            code=work/'candidate.py';code.write_text(CODE)
            write_once(work/'connected-result.json',dict(cell_id=cfg['cell_id'],plan_hash=cfg['plan_hash'],
                contract_hash=cell['task']['contract_hash'],execution_context=cfg['execution_context'],
                phase='done',reason='candidate_verified',selected_source=str(code),
                source_hash=hashlib.sha256(code.read_bytes()).hexdigest()))
            return outcome(cfg['started_monotonic'],deadline)
        assert argv[3].endswith('connected_final')
        receipt=Path(cfg['run_dir'])/'final-result.json'
        if failure!='missing':
            write_once(receipt,dict(finished_monotonic=deadline+1 if failure=='late' else monotonic()))
        return outcome(deadline-cell['plan']['final_wall_clock_s'],deadline,
            returncode=1 if failure=='nonzero' else 0,
            stopped_reason='user_cancelled' if failure=='cancel' else None)
    monkeypatch.setattr(connected,'_supervise',supervise)
    record=connected._cell(cell['plan'],cell['task'],cell['slot'],cell['root'],'owned-invocation')
    assert record['phase']==phase
    assert record['reason']==reason
    assert len(calls)==2
    assert read(cell['path']/'record.json')==record
    assert (cell['path']/'selected.py').read_text()==CODE


def test_development_nonzero_exit_cannot_trigger_independent_final(cell,monkeypatch):
    calls=[]
    def supervise(argv,directory,deadline,root,invocation):
        calls.append(argv)
        cfg=read(argv[-1]);work=Path(cfg['run_dir']);work.mkdir()
        code=work/'candidate.py';code.write_text(CODE)
        write_once(work/'connected-result.json',dict(cell_id=cfg['cell_id'],plan_hash=cfg['plan_hash'],
            contract_hash=cell['task']['contract_hash'],execution_context=cfg['execution_context'],
            phase='done',reason='candidate_verified',selected_source=str(code),
            source_hash=hashlib.sha256(code.read_bytes()).hexdigest()))
        return outcome(cfg['started_monotonic'],deadline,returncode=1)
    monkeypatch.setattr(connected,'_supervise',supervise)
    result=connected._cell(cell['plan'],cell['task'],cell['slot'],cell['root'],'owned-invocation')
    assert result['phase']=='failed' and result['reason']=='worker_failed'
    assert result['final_evaluation'] is None
    assert len(calls)==1 and not (cell['path']/'selected.py').exists()


@pytest.mark.parametrize('tamper',['source_hash','contract','outside','symlink'])
def test_final_refuses_unbound_candidate_before_starting_worker(cell,monkeypatch,tamper):
    work=cell['path']/'work';work.mkdir()
    candidate=work/'candidate.py';candidate.write_text(CODE)
    selected=dict(selected_source=str(candidate),source_hash=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  contract_hash=cell['task']['contract_hash'])
    if tamper=='source_hash':selected['source_hash']='f'*64
    elif tamper=='contract':selected['contract_hash']='f'*64
    elif tamper=='outside':
        candidate=cell['root']/'outside.py';candidate.write_text(CODE);selected['selected_source']=str(candidate)
    else:
        target=work/'target.py';candidate.rename(target);candidate.symlink_to(target)
    def forbidden(*args,**kwargs):raise AssertionError('unbound source reached worker')
    monkeypatch.setattr(connected,'_supervise',forbidden)
    with pytest.raises(ValueError):
        connected._final(cell['plan'],cell['task'],cell['slot'],cell['path'],selected,
                         cell['config'],cell['root'],'owned-invocation')
    assert not (cell['path']/'selected.py').exists()


@pytest.fixture(scope='module')
def real_verification(tmp_path_factory):
    # One actual CPU verification gives the reconciliation cases normal full-L2 evidence.
    work=tmp_path_factory.mktemp('connected-final-verification')
    ht.prepare(DECLARATION,work)
    result=ht.verify(work,CODE,backend='fake')
    assert result['levels']==dict(L1=True,L2=True,L3=False)
    return result


def retained_final(cell,verified):
    directory=cell['path'];source=directory/'selected.py';source.write_text(CODE)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    cfg=dict(cell['config'],run_dir=str(directory/'final'),source=str(source),source_hash=source_hash,
             contract_hash=cell['task']['contract_hash'])
    write_once(directory/'final-config.json',cfg)
    started=monotonic()
    receipt=dict(schema='phase1-connected-final-v1',cell_id=cell['slot']['cell_id'],
        plan_hash=cell['plan']['hash'],source_hash=source_hash,contract_hash=cell['task']['contract_hash'],
        verification_count=1,execution_context=deepcopy(cell['plan']['execution_context']),finished_monotonic=started+.1,
        result=deepcopy(verified))
    write_once(directory/'final/final-result.json',receipt)
    record=dict(cell_id=cell['slot']['cell_id'],final_evaluation=dict(returncode=0,stopped_reason=None,
        started_monotonic=started,deadline_monotonic=started+20,source_hash=source_hash))
    return record,cfg,receipt


def test_final_summary_accepts_actual_full_l2_but_never_fake_gpu_credit(cell,real_verification):
    record,_,_=retained_final(cell,real_verification)
    problems=[]
    result=connected_summary._final(cell['path'],record,cell['task'],cell['plan'],cell['config'],problems)
    assert result['complete'] is True,problems
    assert result['full_l2_passed'] is True
    assert result['gpu_acceptance'] is False and result['performance_evidence'] is False
    assert problems==[]


@pytest.mark.parametrize('tamper',['returncode','stopped','missing','late','source','contract','context',
    'config_backend','config_model','config_source','config_declaration','count','count_bool','levels'])
def test_final_summary_rejects_tampered_identity_config_or_terminal_evidence(cell,real_verification,tamper):
    record,cfg,receipt=retained_final(cell,real_verification)
    directory=cell['path']
    if tamper=='returncode':record['final_evaluation']['returncode']=1
    elif tamper=='stopped':record['final_evaluation']['stopped_reason']='user_cancelled'
    elif tamper=='missing':(directory/'final/final-result.json').unlink()
    elif tamper=='late':receipt['finished_monotonic']=record['final_evaluation']['deadline_monotonic']+1
    elif tamper=='source':(directory/'selected.py').write_text('different source')
    elif tamper=='contract':receipt['contract_hash']='f'*64
    elif tamper=='context':receipt['execution_context']['test_only']=False
    elif tamper=='config_backend':cfg['backend']='ssh'
    elif tamper=='config_model':cfg['model']=dict(cfg['model'],id='different-model')
    elif tamper=='config_source':cfg['source']='/tmp/different.py'
    elif tamper=='config_declaration':cfg['declaration']=dict(cfg['declaration'],shapes=[[16,16]])
    elif tamper=='count':receipt['verification_count']=2
    elif tamper=='count_bool':receipt['verification_count']=True
    elif tamper=='levels':receipt['result']['levels']['L3']=True
    replace_json(directory/'final-config.json',cfg)
    if tamper!='missing':replace_json(directory/'final/final-result.json',receipt)
    problems=[]
    result=connected_summary._final(directory,record,cell['task'],cell['plan'],cell['config'],problems)
    assert result['complete'] is False
    assert result['full_l2_passed'] is None
    assert result['gpu_acceptance'] is False


def retained_budget(cell,*,usage=True):
    cfg=cell['config'];directory=cell['path']
    shared.create(directory/'budget.sqlite',task_id=cfg['shared_task_id'],max_calls=4,
                  max_output_tokens=40960,total_output_tokens=4*40960,wall_clock_s=30.)
    budget=SharedProductionBudget(TaskBudgetConfig.from_dict(cfg['budget']).llm_limits(30.),
        directory/'work/model-journal',shared_path=directory/'budget.sqlite',
        task_id=cfg['shared_task_id'],deadline=cfg['deadline_monotonic'])
    client=LLMClient(base_url=cfg['model']['base_url'],model=cfg['model']['id'],api_key='synthetic-only',
        temperature=.2,max_tokens=4096,enable_thinking=False,timeout=30.,call_budget=budget)
    body=json.dumps(dict(model=cfg['model']['id'],messages=[dict(role='user',content='fixture')],
                         max_tokens=4096,temperature=.2,stream=False),sort_keys=True,separators=(',',':'))
    call,_=budget.reserve(client._bounded_identity(),body,'Coder',30.)
    budget.mark_dispatched(call)
    budget.finish(call,response=CODE,usage_source='api_usage' if usage else None,
        usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15) if usage else None)
    return budget


def test_model_budget_reconciles_actual_shared_and_local_reservations(cell):
    retained_budget(cell)
    problems=[]
    result=connected_summary._budget(cell['path'],cell['config'],problems)
    assert result['complete'] is True,problems
    assert result['reserved_calls']==result['provider_dispatches']==1
    assert result['api_usage']['known_calls']==1
    assert result['api_usage']['totals']==dict(prompt_tokens=10,completion_tokens=5)
    assert result['cost_usd'] is None
    assert problems==[]


def test_missing_provider_usage_stays_unknown_without_losing_budget_reservation(cell):
    retained_budget(cell,usage=False)
    result=connected_summary._budget(cell['path'],cell['config'],[])
    assert result['complete'] is True
    assert result['reserved_calls']==1
    assert result['api_usage']['known_calls']==0
    assert result['api_usage']['unknown_calls']==1
    assert result['api_usage']['totals'] is None


@pytest.mark.parametrize('tamper',['request_hash','usage','caller','reservation','identity','missing_call'])
def test_model_budget_refuses_local_changes_even_when_local_summary_matches_them(cell,tamper):
    retained_budget(cell)
    journal=cell['path']/'work/model-journal'
    local=read(journal/'summary.json');call=read(journal/'call-001.json')
    if tamper=='request_hash':call['request_sha256']='f'*64
    elif tamper=='usage':call['api_usage']['completion_tokens']=500
    elif tamper=='caller':call['caller']='Fixer'
    elif tamper=='reservation':call['output_tokens_reserved']=4000
    elif tamper=='identity':
        identity=read(journal/'identity.json');identity['model']='different-model'
        replace_json(journal/'identity.json',identity);local['identity']=identity
    elif tamper=='missing_call':(journal/'call-001.json').unlink()
    if tamper!='missing_call':
        replace_json(journal/'call-001.json',call)
        local['attempts']=[call]
    replace_json(journal/'summary.json',local)
    problems=[]
    result=connected_summary._budget(cell['path'],cell['config'],problems)
    assert result['complete'] is False
    assert result['reserved_calls']==1  # an inconsistent journal must not refund the shared reservation
    assert problems


@pytest.mark.parametrize('tamper',['request_hash','usage','reservation','cap'])
def test_model_budget_refuses_shared_changes_that_disagree_with_local_journal(cell,monkeypatch,tamper):
    retained_budget(cell)
    snapshot=shared.snapshot(cell['path']/'budget.sqlite',expected_task_id=cell['config']['shared_task_id'])
    if tamper=='request_hash':snapshot['calls'][0]['request_sha256']='f'*64
    elif tamper=='usage':snapshot['calls'][0]['api_usage']['completion_tokens']=500
    elif tamper=='reservation':snapshot['calls'][0]['output_tokens_reserved']=4000
    else:snapshot['max_calls']=30
    monkeypatch.setattr(shared,'snapshot',lambda *args,**kwargs:deepcopy(snapshot))
    problems=[]
    result=connected_summary._budget(cell['path'],cell['config'],problems)
    assert result['complete'] is False
    assert result['reserved_calls']==1
    assert problems
