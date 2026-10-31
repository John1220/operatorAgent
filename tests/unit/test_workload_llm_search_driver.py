"""Deterministic local guards. NO model endpoint, GPU or performance claims."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from benchmarks import _bounded_proposals as transport
from benchmarks.drivers import workload_llm_search_h20 as driver
from operator_agent.core import run_contract as rc


def response(warps=4, **changes):
    r=dict(model='fixture-model',choices=[dict(finish_reason='stop',message=dict(
        content=json.dumps(dict(num_warps=warps,rationale='bounded fixture proposal'))))],
        usage=dict(prompt_tokens=100,completion_tokens=25))
    r.update(changes)
    return json.dumps(r)


def model(tmp_path):
    cfg=dict(url='https://fixture.invalid/v1',model='fixture-model',key='SECRET_DO_NOT_RETAIN')
    return transport.BoundedModel(cfg,tmp_path,transport.model_identity(cfg))


def test_default_plan_has_no_config_model_transport_or_input_allocation(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('plan reached external work')
    monkeypatch.setattr(driver,'model_config',forbidden)
    monkeypatch.setattr(transport,'dispatch',forbidden)
    monkeypatch.setattr(driver.wd,'BoundedProfileSSH',forbidden)
    monkeypatch.setattr(driver.wd,'primary',forbidden)
    assert driver.run(driver.parser().parse_args(['--output',str(tmp_path)]))==0
    root=next(tmp_path.iterdir());plan=json.loads((root/'plan.json').read_text())
    assert plan['hash']==rc.digest({k:v for k,v in plan.items() if k!='hash'})
    assert plan['model'] is None and plan['model_limits']['max_calls']==2
    assert plan['controls']['budget']['oracle_total']==7
    assert plan['budget_accounting']['unit']=='candidate_verification_dispatch_per_input_group'
    assert plan['budget_accounting']['fixed_dispatch_sequence_unchanged']
    assert plan['proposal_policy']['source_author']=='trusted_template'
    assert plan['proposal_policy']['max_repairs']==0
    assert plan['contract']['min_speedup']==10.
    assert (root/'plan.json').stat().st_mode&0o777==0o444
    for name,sha in plan['evaluator_sources'].items():
        assert rc.code_hash((root/'frozen-sources'/name).read_text())==sha
    assert not json.loads((root/'summary.json').read_text())['completed']


@pytest.mark.parametrize('argv',[['--execute'],['--execute','--gpu','2']])
def test_explicit_execution_controls_required_before_config(tmp_path,monkeypatch,argv):
    monkeypatch.setattr(driver,'model_config',lambda *a:pytest.fail('config accessed'))
    with pytest.raises(ValueError,match='--gpu AND --cuda-ruler-profile'):
        driver.run(driver.parser().parse_args(['--output',str(tmp_path)]+argv))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('argv',[['--repeat','2'],['--min-speedup','1'],['--max-calls','10'],['--repair','1'],['--model','other']])
def test_cli_cannot_relax_frozen_limits(argv):
    with pytest.raises(SystemExit):driver.parser().parse_args(argv)


def test_two_calls_reserved_before_dispatch_and_credentials_never_retained(tmp_path,monkeypatch):
    m=model(tmp_path); payloads=[]
    def send(payload):
        path=tmp_path/('model-call-%02d.json'%(len(payloads)+1))
        retained=json.loads(path.read_text())
        assert retained['status']=='reserved'
        assert 'SECRET_DO_NOT_RETAIN' not in path.read_text()
        payloads.append(payload)
        return response(4 if len(payloads)==1 else 8)
    monkeypatch.setattr(transport,'dispatch',send)
    for stage in ('InitialProposal','SearchProposal'):
        assert m.ask([dict(role='user',content='bounded')],stage)['num_warps'] in driver.WARPS
    with pytest.raises(ValueError,match='budget exhausted before dispatch'):
        m.ask([], 'SearchProposal')
    assert len(payloads)==2 and m.cost_tracker.total_calls==2
    assert all(p['key']=='SECRET_DO_NOT_RETAIN' for p in payloads)
    assert sum(r['output_tokens_reserved'] for r in m.records)==1024
    assert all((tmp_path/('model-call-%02d.json'%(i+1))).stat().st_mode&0o777==0o444 for i in range(2))
    assert 'SECRET_DO_NOT_RETAIN' not in json.dumps(m.records)+repr(m)


@pytest.mark.parametrize('kind',('network','truncated','empty','bad_json','wrong_model','over_tokens','negative_usage'))
def test_failed_attempt_stays_counted_and_retains_evidence_without_retry(tmp_path,monkeypatch,kind):
    m=model(tmp_path); sent=[]
    r=json.loads(response())
    if kind=='truncated':r['choices'][0]['finish_reason']='length'
    elif kind=='empty':r['choices'][0]['message']['content']=''
    elif kind=='bad_json':r['choices'][0]['message']['content']='```json\n{}\n```'
    elif kind=='wrong_model':r['model']='undeclared-fallback'
    elif kind=='over_tokens':r['usage']['completion_tokens']=513
    elif kind=='negative_usage':r['usage']['prompt_tokens']=-1
    def send(payload):
        sent.append(payload)
        if kind=='network':raise RuntimeError('secret URL/key MUST NOT BE RETAINED')
        return json.dumps(r)
    monkeypatch.setattr(transport,'dispatch',send)
    with pytest.raises(ValueError,match='no retry'):m.ask([], 'InitialProposal')
    assert len(sent)==len(m.records)==1
    retained=json.loads((tmp_path/'model-call-01.json').read_text())
    assert retained['status']=='failed'
    assert 'MUST NOT BE RETAINED' not in json.dumps(retained)
    if kind!='network':assert retained['response_text']==json.dumps(r)


def test_request_byte_bound_is_before_reservation_and_transport(tmp_path,monkeypatch):
    m=model(tmp_path)
    monkeypatch.setattr(transport,'dispatch',lambda *a:pytest.fail('oversized dispatch'))
    with pytest.raises(ValueError,match='byte limit'):
        m.ask([dict(role='user',content='中'*16384)],'InitialProposal')
    assert m.records==[] and not list(tmp_path.iterdir())


def test_model_config_drift_and_unknown_caller_fail_before_dispatch(tmp_path,monkeypatch):
    m=model(tmp_path)
    monkeypatch.setattr(transport,'dispatch',lambda *a:pytest.fail('drift dispatch'))
    with pytest.raises(ValueError,match='undeclared'):m.ask([], 'Fixer')
    m._config['url']='https://another.invalid'
    with pytest.raises(ValueError,match='drift'):m.ask([], 'InitialProposal')
    assert not m.records


def test_failed_durable_reservation_cannot_spend_call(tmp_path,monkeypatch):
    m=model(tmp_path)
    monkeypatch.setattr(transport,'atomic_json',lambda *a,**k:(_ for _ in ()).throw(OSError('disk full')))
    monkeypatch.setattr(transport,'dispatch',lambda *a:pytest.fail('unrecorded dispatch'))
    with pytest.raises(OSError):m.ask([], 'InitialProposal')


def test_campaign_deadline_and_old_attempt_refused_before_transport(tmp_path,monkeypatch):
    m=model(tmp_path)
    monkeypatch.setattr(transport,'dispatch',lambda *a:pytest.fail('forbidden dispatch'))
    m.deadline=0
    with pytest.raises(ValueError,match='campaign time'):m.ask([], 'InitialProposal')
    m.deadline=float('inf')
    (tmp_path/'model-call-01.json').write_text('old evidence')
    with pytest.raises(ValueError,match='overwrite'):m.ask([], 'InitialProposal')
    assert (tmp_path/'model-call-01.json').read_text()=='old evidence'


def test_child_deadline_and_no_credentials_in_argv(tmp_path,monkeypatch):
    calls=[]
    def run(argv,**kwargs):
        calls.append((argv,kwargs))
        assert 'SECRET' not in ' '.join(argv)
        assert kwargs['timeout']==60 and kwargs['capture_output']
        raise subprocess.TimeoutExpired(argv,60)
    monkeypatch.setattr(transport.subprocess,'run',run)
    with pytest.raises(ValueError,match='child killed, no retry'):
        transport.dispatch(dict(url='https://fixture.invalid',key='SECRET',body={}))
    assert len(calls)==1


def test_redirect_is_not_followed():
    assert transport.NoRedirect().redirect_request(None,None,302,'Found',{},'https://other.invalid') is None


def test_worker_caps_response_bytes_and_does_not_log_error_bodies(monkeypatch):
    class R:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self,n):
            assert n==65537
            return b'x'*n
    class O:
        def open(self,*a,**k):return R()
    monkeypatch.setattr(transport.urllib.request,'build_opener',lambda *a:O())
    assert transport.worker(dict(url='https://fixture.invalid',key='SECRET',body={}))=={'error':'response_size_limit'}
    def failed(*a,**k):raise RuntimeError('SECRET_DO_NOT_RETAIN')
    monkeypatch.setattr(O,'open',failed)
    assert transport.worker(dict(url='https://fixture.invalid',key='SECRET',body={}))=={'error':'RuntimeError','http_status':None}


@pytest.mark.parametrize('proposal',[
    dict(num_warps=True,rationale='x'),dict(num_warps=32,rationale='x'),
    dict(num_warps=4,rationale=''),dict(num_warps=4,rationale=0),
    dict(num_warps=4,rationale='x',source='import os'),
    dict(num_warps=4,rationale='x',min_speedup=1),dict(num_warps=4,rationale='x'*1001),
])
def test_model_cannot_edit_source_contract_or_resource_space(proposal):
    with pytest.raises(ValueError,match='unsafe'):driver.parse_proposal(proposal,[])


def test_duplicate_proposal_no_retry():
    with pytest.raises(ValueError,match='duplicate'):
        driver.parse_proposal(dict(num_warps=4,rationale='repeat'),[4])


def test_materialization_changes_only_approved_launch_knob():
    assert driver.source(4).replace('num_warps=4','num_warps=16')==driver.source(16)
    assert 'enable_fp_fusion=False' in driver.source(8)


def test_generated_audit_input_is_not_a_new_frozen_plan():
    _,plan=driver.declaration(driver.parser().parse_args([]))
    before=copy.deepcopy(plan)
    generated=[dict(source=driver.source(8),source_hash=rc.code_hash(driver.source(8)))]
    projection=driver.audit_input(plan,generated)
    assert plan==before and 'candidates' not in plan
    assert 'hash' not in projection and projection['frozen_plan_hash']==plan['hash']
    assert projection['schema']!=plan['schema']
    assert projection['candidates'][0]['hash']==generated[0]['source_hash']


def session_fixture(tmp_path,monkeypatch):
    m=model(tmp_path)
    task,plan=driver.declaration(driver.parser().parse_args([]),m.identity)
    rc.atomic_json(tmp_path/'plan.json',plan,readonly=True)
    s=driver.ProposalSession(m,tmp_path,plan,task,[])
    replies=iter((response(4),response(8)))
    monkeypatch.setattr(transport,'dispatch',lambda *a:next(replies))
    # Explicit fake full-workload feedback, never machine evidence.
    monkeypatch.setattr(driver,'feedback',lambda *a:dict(objective=1.5,cases={'fixture-a':1,'fixture-b':2}))
    s.design();code=s.generate();s.before_verify(code);s.rounds.append((code,[]))
    s.search(None,{});code=s.generate();s.before_verify(code);s.rounds.append((code,[]))
    return s


def test_real_response_lineage_audit_reads_only_and_checks_two_candidates(tmp_path,monkeypatch):
    s=session_fixture(tmp_path,monkeypatch)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    monkeypatch.setattr(transport,'dispatch',lambda *a:pytest.fail('audit called model'))
    proof=driver.audit_model(s,tmp_path)
    assert proof['completed'] and not proof['errors'] and proof['attempts']==2
    assert proof['output_token_reservations']==1024
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError,match='before model dispatch'):s.propose('SearchProposal')
    with pytest.raises(ValueError,match='generation'):s.generate()
    with pytest.raises(ValueError,match='before verification'):s.before_verify(s.generated[0])
    with pytest.raises(ValueError,match='zero'):s.refuse_repair()


@pytest.mark.parametrize('drift',('response','feedback','source','reservation','caller','dispatch','missing_file','returned_model'))
def test_audit_refuses_torn_or_forged_model_source_lineage(tmp_path,monkeypatch,drift):
    s=session_fixture(tmp_path,monkeypatch)
    if drift=='response':s.model.records[1]['response_text']=response(16)
    elif drift=='feedback':s.model.records[1]['request']['messages'][1]['content']='{}'
    elif drift=='source':s.proposals[1]['source']+='\n# injected'
    elif drift=='reservation':s.model.records[1]['output_tokens_reserved']=1024
    elif drift=='caller':s.model.records[1]['stage']='Fixer'
    elif drift=='dispatch':s.dispatched.append('third')
    elif drift=='returned_model':s.model.records[1]['reported_model']='other'
    else:(tmp_path/'model-call-02.json').unlink()
    # Even changing both memory and disk must not satisfy cross-source checks.
    if drift not in ('missing_file','source'):
        rc.atomic_json(tmp_path/'model-call-02.json',s.model.records[1])
    if drift=='source':rc.atomic_json(tmp_path/'proposal-02.json',s.proposals[1])
    assert driver.audit_model(s,tmp_path)['errors']


def test_feedback_uses_full_workload_not_primary_or_gflops(tmp_path,monkeypatch):
    from tests.unit.test_workload_search_driver import durable_fixture
    from operator_agent.core.contract import freeze_from_task
    art,c,t,rounds,plan=durable_fixture(tmp_path,monkeypatch)
    data=driver.feedback(rounds[:1],freeze_from_task(t))
    assert data['objective']==2. and len(data['cases'])==2
    assert 'gflops' not in json.dumps(data)
    for case,row in rounds[0][1][-1].metrics['cases'].items():
        assert data['cases'][case]['speedup']==row['speedup_vs_baseline']
        assert data['cases'][case]['candidate_window']==row['arm_windows']['candidate']
    changed=copy.deepcopy(rounds[:1]);changed[0][1][-1].metrics['run_binding']['candidate_hash']='swapped'
    with pytest.raises(ValueError,match='binding drift'):driver.feedback(changed,freeze_from_task(t))


def test_failed_first_observation_cannot_drive_second_model_call(tmp_path,monkeypatch):
    from tests.unit.test_workload_search_driver import durable_fixture
    from operator_agent.core.contract import freeze_from_task
    art,c,t,rounds,plan=durable_fixture(tmp_path,monkeypatch)
    rounds[0][1][0].passed=False
    with pytest.raises(ValueError,match='no correct'):driver.feedback(rounds[:1],freeze_from_task(t))


def test_degraded_feedback_cannot_buy_model_tuning(tmp_path,monkeypatch):
    from tests.unit.test_workload_search_driver import durable_fixture
    from operator_agent.core.contract import freeze_from_task
    art,c,t,rounds,plan=durable_fixture(tmp_path,monkeypatch)
    rounds[0][1][-1].metrics['measurement_validity']='degraded'
    with pytest.raises(ValueError,match='unqualified'):driver.feedback(rounds[:1],freeze_from_task(t))


@pytest.mark.parametrize('drift',('plan','contract','source'))
def test_plan_drift_blocks_model_or_gpu_dispatch(tmp_path,monkeypatch,drift):
    m=model(tmp_path);task,plan=driver.declaration(driver.parser().parse_args([]),m.identity)
    rc.atomic_json(tmp_path/'plan.json',plan,readonly=True)
    s=driver.ProposalSession(m,tmp_path,plan,task,[])
    monkeypatch.setattr(transport,'dispatch',lambda *a:pytest.fail('drift reached model'))
    if drift=='plan':
        altered=copy.deepcopy(plan);altered['proposal_policy']['max_candidates']=3
        rc.atomic_json(tmp_path/'plan.json',altered)
    elif drift=='contract':task.constraints.min_speedup=1.
    else:monkeypatch.setattr(driver,'sources',lambda:{'changed':'implementation'})
    with pytest.raises(ValueError,match='drift'):s.design()
    assert m.records==[]
    with pytest.raises(ValueError,match='drift'):s.before_verify(driver.source(4))


def test_prepare_failure_still_cleans_owned_backend_without_model_retry(tmp_path,monkeypatch):
    events=[]
    class B:
        forbidden_actions=[];trace=[];work_counts={}
        def __init__(self,*a,**k):events.append('construct')
        def prepare(self,*a):events.append('prepare');raise ValueError('fixture preparation failed')
        def cleanup(self):events.append('cleanup');return dict(quiescent=True,removed=True)
    monkeypatch.setattr(driver,'model_config',lambda *a:dict(url='https://fixture.invalid',model='fixture-model',key=''))
    monkeypatch.setattr(driver.wd,'ruler',lambda *a:(None,{'identity_hash':'a'*64}))
    monkeypatch.setattr(driver.wd,'BoundedProfileSSH',B)
    monkeypatch.setattr(driver.wd,'cleanup_errors',lambda *a:[])
    monkeypatch.setattr(transport,'dispatch',lambda *a:pytest.fail('prepare failure spent model call'))
    a=driver.parser().parse_args(['--output',str(tmp_path),'--execute','--gpu','2','--cuda-ruler-profile','fixture'])
    assert driver.run(a)==1 and events==['construct','prepare','cleanup']
    summary=json.loads((next(tmp_path.iterdir())/'summary.json').read_text())
    assert not summary['completed'] and summary['model']['attempts']==0
    assert summary['cleanup']['quiescent'] and summary['errors']


@pytest.mark.parametrize('second_fails',(False,True))
def test_production_conductor_consumes_real_response_hooks_and_saves_own_winner(tmp_path,monkeypatch,second_fails):
    """Fake model + CPU/fake timings; exercises hooks, NOT remote acceptance."""
    import contextlib
    import io
    from operator_agent import Conductor, KernelDatabase
    from operator_agent.execute.budget import BudgetPlan
    from operator_agent.core.contract import freeze_from_task
    from operator_agent.core.kernel_db import FailureDatabase
    from operator_agent.core.artifact_store import ArtifactStore
    from tests.unit.test_workload_evidence import task,inputs,backend,CODE
    m=model(tmp_path);t=task(min_speedup=10.)
    _,plan=driver.declaration(driver.parser().parse_args([]),m.identity)
    plan['contract']=rc.json_value(freeze_from_task(t).as_dict())
    plan['hash']=rc.digest({k:v for k,v in plan.items() if k!='hash'})
    rc.atomic_json(tmp_path/'plan.json',plan,readonly=True)
    # CPU runnable source in THIS fixture only. Live materializer is unchanged.
    monkeypatch.setattr(driver,'source',lambda w:CODE+'# num_warps='+str(w)+'\n')
    sent=[]
    def send(payload):
        sent.append(payload)
        if second_fails and len(sent)==2:raise ValueError('fixture second model failure')
        return response(4 if len(sent)==1 else 8)
    monkeypatch.setattr(transport,'dispatch',send)
    b,calls=backend();bench=b.benchmark_with_baselines
    def paired(code,*a,**k):
        p=bench(code,*a,**k)
        p['baselines']['pytorch_eager']['median_time_ms']=.4 if 'warps=4' in code else .3
        return p
    b.benchmark_with_baselines=paired
    s=driver.ProposalSession(m,tmp_path,plan,t,[])
    c=Conductor(backend=b,llm=m,kernel_db=KernelDatabase(str(tmp_path/'corpus.jsonl'),persist_retrievals=False),
        failure_db=FailureDatabase(str(tmp_path/'failure.jsonl')),use_coder_pipeline=False,
        max_optimize_rounds=3,budget=BudgetPlan(oracle_total=5,llm_tokens=40000))
    c.store.output_dir=tmp_path/'artifacts'
    c.designer=SimpleNamespace(run=s.design);c.coder=SimpleNamespace(run=s.generate)
    c.search=SimpleNamespace(run=s.search);c.fixer=SimpleNamespace(run_with_action=s.refuse_repair)
    original=c.verifier.verify
    def capture(*a,**k):
        s.before_verify(k['kernel_code'])
        results=original(*a,**k);s.rounds.append((k['kernel_code'],results));return results
    c.verifier.verify=capture
    log=io.StringIO()
    with contextlib.redirect_stdout(log):art=c.execute(t,[inputs(16),inputs(32)],target_gflops=0.)
    loaded=ArtifactStore(str(tmp_path)).load(art.saved_dir)
    assert art.code==s.generated[0]==loaded.code and not art.all_passed
    assert not (tmp_path/'corpus.jsonl').exists()
    assert len(sent)==2 and len(m.records)==2
    assert loaded.metadata['candidate_selection']['selected_round']==1
    assert art.metadata['stop']['reason']==('search_error' if second_fails else 'budget_exhausted'), log.getvalue()
    assert len(calls)==(2 if second_fails else 4) and len(s.rounds)==(1 if second_fails else 2)
    assert driver.audit_model(s,tmp_path)['completed'] is (not second_fails)
    persisted=json.loads((Path(art.saved_dir)/'result.json').read_text())
    assert persisted['llm_cost']==rc.json_value(art.metadata['llm_cost'])
    if not second_fails:
        proof=driver.fixed.audit_durable(art,s.rounds,plan,task=t,input_sets=[inputs(16),inputs(32)],controls=c._run_controls)
        assert proof['completed'] and proof['selected_round']==1
