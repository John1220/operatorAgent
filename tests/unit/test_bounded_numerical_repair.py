"""Offline mock lineage/negative controls. NO real HTTP, SSH, CUDA or speed claims."""
import contextlib
import copy
import io
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from benchmarks import _numerical_repair as nr
from benchmarks.drivers import workload_codegen_h20 as d
from operator_agent import llm_client as lc
from operator_agent.bounded_repair import repair_lineage, validate_repair
from operator_agent.llm_budget import LLMCallBudget, LLMCallLimits, LLMCallRefused
from operator_agent.reference_contract import check_observed
from operator_agent.run_contract import atomic_json, code_hash, digest, json_value
from operator_agent.schemas import VerificationLevel as Level, VerificationResult
from operator_agent.search_objective import results_dict
from operator_agent.single_candidate_coder import SingleCandidateCoder, audit_generation
from tests.unit.test_single_candidate_coder import conductor, response
from tests.unit.test_workload_codegen_driver import retain_model
from tests.unit.test_reference_contract import ProtocolSSH

SOURCE = (Path(__file__).resolve().parents[1]/'fixtures/single_candidate_unmasked_variance.txt').read_text()
# Static mock response only, not a production mechanical rewrite or GPU proof.
CHANGED = SOURCE + '\nfixture_revision = 1\n'


@pytest.fixture
def parent(tmp_path, monkeypatch):
    shapes = ((2,32),(3,33))
    original_task = d.wd.task
    monkeypatch.setattr(d.wd,'LARGE',shapes)
    monkeypatch.setattr(d.wd,'task',lambda name: original_task(name,shapes=shapes))
    monkeypatch.setattr(nr,'CLAIMS',tmp_path/'claims')
    root=tmp_path/'parent';root.mkdir()
    args=d.parser().parse_args(['--gpu','2','--model-url','https://fixture.invalid/v1','--model','fixture'])
    task,plan=d.declaration(args);d.freeze(root,plan)
    xs=[{'X':np.ones(shape,np.float16)} for shape in shapes]
    client=d.PlanLLMClient(base_url=args.model_url,model='fixture',temperature=.1,timeout=60,
        call_budget=LLMCallBudget(d.LIMITS,root/'model-journal'))
    client.frozen_plan=plan;client.plan_guard=lambda:None
    coder=SingleCandidateCoder(client,task,d.sketch(),xs)
    monkeypatch.setattr(lc,'bounded_dispatch',lambda *a:response(SOURCE))
    b=ProtocolSSH();c=conductor(root,coder,b)
    receipt=check_observed(d.registered_reference(),d.registered_reference().execution.content_dict())
    primary=[]
    for i,shape in enumerate(shapes):
        size=int(np.prod(shape)); ok=i==0
        primary.append(dict(success=ok,output_shape=list(shape),output_dtype='torch.float16',
            max_error=.001 if ok else .04,max_rel_error=.001 if ok else .03,
            matched_ratio=1. if ok else 0.,bad_elements=0 if ok else size,compared_elements=size,
            coverage=1.,reference_nonfinite=0,atol=1e-5,rtol=.01,required_matched_ratio=1.,
            device='cuda',reference_type='pytorch_eager',reference_receipt=receipt))
    suite=d.wd.as_suite(plan['contract']['correctness_suite'])
    numeric={k:v for k,v in primary[-1].items() if k not in ('success','output_shape')}
    numeric.update(correctness_primary_verdict='failed',correctness=dict(suite_hash=suite.digest(),
        cases=[dict(case_id=case.case_id,verdict='not_run') for case in suite.cases]))
    c.verifier._verify_compile=lambda *a,**k:VerificationResult(Level.L1_COMPILE,True,metrics={'static_guards':True})
    c.verifier._verify_function=lambda *a,**k:VerificationResult(Level.L2_FUNCTION,False,'fake numerical failure',metrics=numeric)
    output=io.StringIO()
    with contextlib.redirect_stdout(output): art=c.execute(task,xs,target_gflops=0.)
    assert art.saved_dir, (art.metadata, art.verification_results, output.getvalue())
    retain_model(root,client)
    directory=Path(art.saved_dir); manifest=d.read_object(directory/'run_manifest.json')
    trace=[dict(request=dict(action='compile',code=SOURCE),response={'success':True})]
    for res, original in zip(primary,manifest['inputs']):
        res['input_fidelity']=dict(representation='torch-storage-v1',observed=original,tensor_devices={'X':'cuda:0'})
        trace.append(dict(request=dict(action='verify',code=SOURCE,inputs={'tensors':original},
            precision='fp16',max_error=1e-5,reference_contract=plan['contract']['reference_contract']),response=res))
    atomic_json(root/'evidence.json',dict(rounds=[dict(code=SOURCE,results=results_dict(art.verification_results))],
        dispatched=[code_hash(SOURCE)],trace=trace,forbidden_actions=[]),readonly=True)
    atomic_json(root/'summary.json',dict(plan_hash=plan['hash'],status='correctness_failed',completed=False,
        accepted=False,deployment_accepted=False,errors=[],model_attempts=1,verifier_dispatches=1,
        saved_dir=str(directory),cleanup_status={'state':'removed'},cleanup={'quiescent':True,'removed':True}),readonly=True)
    return root,task,xs,args


def repair_coder(tmp_path, task, xs, repair):
    limits=LLMCallLimits(**dict(d.asdict(d.LIMITS),allowed_callers=('SingleCandidateRepair',)))
    llm=lc.LLMClient(base_url='https://fixture.invalid/v1',model='fixture',temperature=.1,timeout=60,
        call_budget=LLMCallBudget(limits,tmp_path/'repair-journal'))
    return SingleCandidateCoder(llm,task,d.sketch(),xs,repair=repair)


def mutate(path, change):
    data=json.loads(path.read_text());change(data);path.chmod(0o600);path.write_text(json.dumps(data))


def test_parent_projection_is_read_only_allowlisted_and_not_new_eligibility(parent):
    root,t,xs,args=parent
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    repair,proof=nr.load_parent(root,t)
    assert repair['source']==SOURCE and repair['repair_index']==1
    assert len(repair['feedback']['primary'])==2
    assert repair['feedback']['suite']==dict(declared_cases=32,executed_cases=0,verdict='not_run')
    raw=json.dumps(repair)
    for excluded in ('error_log','first_bad_actual','first_bad_expected','input_fidelity','storage_digest',str(root)):
        assert excluded not in raw
    assert proof['identity_hash']==digest(repair['parent'])
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['extra','bool_count','nan','coverage','threshold','shape','all_passed',
    'source','parent','index','suite','ratio','dtype','negative','contract'])
def test_feedback_and_parent_fields_fail_closed(parent,change):
    root,t,xs,args=parent;r,_=nr.load_parent(root,t); row=r['feedback']['primary'][1]
    if change=='extra':r['feedback']['log']='PRIVATE arbitrary model instructions'
    elif change=='bool_count':row['bad_elements']=True
    elif change=='nan':row['max_error']=float('nan')
    elif change=='coverage':row['coverage']=.5
    elif change=='threshold':row['rtol']=.5
    elif change=='shape':row['shape']=[1,99]
    elif change=='all_passed':row.update(passed=True,bad_elements=0,matched_ratio=1.)
    elif change=='source':r['source']+='\n'
    elif change=='parent':r['parent']['run_id']='../bad'
    elif change=='index':r['repair_index']=2
    elif change=='suite':r['feedback']['suite']['executed_cases']=1
    elif change=='ratio':row['matched_ratio']=.5
    elif change=='dtype':row['output_dtype']='torch.float32'
    elif change=='negative':row['max_error']=-1.
    else:r['contract_hash']='a'*64
    if change!='nan':r['feedback_hash']=digest(r['feedback'])
    with pytest.raises(ValueError):validate_repair(r,d.generation_declaration(t)['context']['contract'],d.generation_declaration(t)['context']['input_metadata'])


@pytest.mark.parametrize('change',['summary','cleanup','checkpoint','source','verification','model','snapshot',
    'manifest','trace','chain','confirmation','symlink','fidelity'])
def test_parent_loader_rejects_torn_unknown_or_ineligible_parent(parent,change):
    root,t,xs,args=parent;a=Path(d.read_object(root/'summary.json')['saved_dir'])
    if change=='summary':mutate(root/'summary.json',lambda v:v.update(status='generation_failed'))
    elif change=='cleanup':mutate(root/'summary.json',lambda v:v['cleanup'].update(removed=False))
    elif change=='checkpoint':mutate(a/'execution_snapshot.json',lambda v:v.update(round_done=2))
    elif change=='source':(a/'code.py').write_text(CHANGED)
    elif change=='verification':mutate(a/'verification.json',lambda v:v[1].update(passed=True))
    elif change=='model':mutate(root/'model-journal/candidate.json',lambda v:v.update(code=CHANGED))
    elif change=='snapshot':(root/'frozen-sources/operator_agent/verifier.py').chmod(0o600);(root/'frozen-sources/operator_agent/verifier.py').write_text('changed')
    elif change=='manifest':mutate(a/'run_manifest.json',lambda v:v.update(workload_hash='f'*64))
    elif change=='trace':mutate(root/'evidence.json',lambda v:v['trace'][-1]['response'].update(compared_elements=0))
    elif change=='chain':mutate(root/'plan.json',lambda v:v['generation'].update(repairs=1))
    elif change=='confirmation':atomic_json(a/'confirmation.json',{})
    elif change=='fidelity':mutate(root/'evidence.json',lambda v:v['trace'][-1]['response']['input_fidelity'].update(observed={}))
    else:
        p=a/'code.py';p.unlink();p.symlink_to(root/'absent')
    with pytest.raises((ValueError,KeyError)):nr.load_parent(root,t)


def test_repair_source_is_exact_one_model_response_with_bound_parent(parent,tmp_path,monkeypatch):
    root,t,xs,args=parent;r,p=nr.load_parent(root,t);coder=repair_coder(tmp_path,t,xs,r)
    send=Mock(return_value=response(CHANGED));monkeypatch.setattr(lc,'bounded_dispatch',send)
    assert coder.run(d.sketch(),t,test_inputs=xs)==CHANGED
    assert coder.declaration()==d.generation_declaration(t,r)
    assert coder.last_generation['repair']==repair_lineage(r)
    body=json.loads(send.call_args.args[0]['body_text'])
    assert json.loads(body['messages'][1]['content'])['repair']==r
    assert send.call_count==1 and coder.llm.call_budget.summary()['attempts'][0]['caller']=='SingleCandidateRepair'
    assert coder.check_candidate(CHANGED)['precheck_only']
    # Reconstruct the repair caller's cost/source journal with the same auditor.
    retain_model(tmp_path,coder.llm)
    journal=coder.llm.call_budget.root
    journal.rename(tmp_path/'model-journal')
    gen=coder.declaration()
    identity=coder.llm.call_budget.summary()['identity']
    request=d.request_body(gen,identity);raw=json.dumps(request,ensure_ascii=False,allow_nan=False)
    plan=dict(generation=gen,model=identity,model_limits=json_value(d.asdict(coder.llm.call_budget.limits)),
              request=dict(body=request,sha256=code_hash(raw),bytes=len(raw.encode())))
    assert d.audit_model(tmp_path,plan)['completed']
    (tmp_path/'model-journal').rename(journal)
    with pytest.raises(LLMCallRefused):coder.run(d.sketch(),t,test_inputs=xs)
    lineage=copy.deepcopy(coder.last_generation);lineage['repair']['parent_candidate_hash']='f'*64
    with pytest.raises(ValueError):audit_generation(coder.declaration(),coder.llm.call_budget.summary(),lineage,CHANGED)


def test_identical_source_burns_one_call_without_rewrite_or_retry(parent,tmp_path,monkeypatch):
    root,t,xs,args=parent;r,_=nr.load_parent(root,t);coder=repair_coder(tmp_path,t,xs,r)
    send=Mock(return_value=response(SOURCE));monkeypatch.setattr(lc,'bounded_dispatch',send)
    with pytest.raises(LLMCallRefused):coder.run(d.sketch(),t,test_inputs=xs)
    assert coder.last_generation['code']==SOURCE
    assert coder.last_generation['failure']==dict(stage='precheck',reason='repair_unchanged_source')
    with pytest.raises(LLMCallRefused):coder.run(d.sketch(),t,test_inputs=xs)
    assert send.call_count==1 and coder.llm.call_budget.summary()['aborted']


def test_fresh_failed_repair_never_inherits_parent_pass_or_reaches_l3(parent,tmp_path,monkeypatch):
    root,t,xs,args=parent;r,_=nr.load_parent(root,t);coder=repair_coder(tmp_path,t,xs,r)
    send=Mock(return_value=response(CHANGED));monkeypatch.setattr(lc,'bounded_dispatch',send)
    b=ProtocolSSH();c=conductor(tmp_path,coder,b);counts=[]
    c.verifier._verify_compile=lambda *a,**k:counts.append('L1') or VerificationResult(Level.L1_COMPILE,True,metrics={'static_guards':True})
    c.verifier._verify_function=lambda *a,**k:counts.append('L2') or VerificationResult(Level.L2_FUNCTION,False,'new fake failure')
    forbidden=lambda *a,**k:pytest.fail('extra search/fixer/performance')
    c.fixer.run_with_action=forbidden;c.search.run=forbidden;c.verifier._verify_performance=forbidden
    with contextlib.redirect_stdout(io.StringIO()):art=c.execute(t,xs,target_gflops=0.)
    assert counts==['L1','L2'] and not art.all_passed and len(art.iteration_history)==1
    assert not any(r['action']=='benchmark_with_baselines' for r in b.requests)
    assert send.call_count==1 and not (tmp_path/'corpus.jsonl').exists()
    assert art.metadata['code_generation']['lineage']['repair']==repair_lineage(r)


def test_parent_claim_survives_new_plan_and_fresh_output(parent,tmp_path):
    root,t,xs,args=parent;r,p=nr.load_parent(root,t);nr.require_available(p)
    nr.claim(p,'a'*64,tmp_path/'first');before=nr.claim_path(p).read_bytes()
    nr.check_claim(p,'a'*64,tmp_path/'first')
    with pytest.raises(ValueError):nr.check_claim(p,'b'*64,tmp_path/'first')
    with pytest.raises(ValueError):nr.check_claim(p,'a'*64,tmp_path/'second')
    for action in (lambda:nr.require_available(p),lambda:nr.claim(p,'b'*64,tmp_path/'second')):
        with pytest.raises(ValueError,match='consumed'):action()
    assert nr.claim_path(p).read_bytes()==before
    assert nr.claim_path(p).stat().st_mode & 0o777==0o444
    args.repair_parent_run=str(root)
    with pytest.raises(ValueError,match='consumed'):d.declaration(args)


def test_plan_only_repair_freezes_exact_feedback_without_network_arrays_or_claim(parent,tmp_path,monkeypatch):
    root,t,xs,args=parent
    forbidden=lambda *a,**k:pytest.fail('plan performed external work/allocation')
    monkeypatch.setattr(d,'CodegenSSH',forbidden);monkeypatch.setattr(d,'PlanLLMClient',forbidden)
    monkeypatch.setattr(d.wd,'primary',forbidden);monkeypatch.setattr(lc,'bounded_dispatch',forbidden)
    args.repair_parent_run=str(root);args.output=str(tmp_path/'prepared')
    assert d.run(args)==0
    prepared=next(Path(args.output).iterdir());plan=d.read_object(prepared/'plan.json')
    assert plan['generation']['repairs']==1 and plan['model_limits']['allowed_callers']==['SingleCandidateRepair']
    assert plan['disclosure']['failed_model_source'] and plan['disclosure']['previous_campaign_feedback']
    assert not plan['disclosure']['tensor_values'] and not plan['disclosure']['template_source']
    assert not nr.claim_path(plan['repair_parent']).exists()
    assert not (prepared/'execution-claim.json').exists()
    assert json.loads(plan['request']['body']['messages'][1]['content'])['repair']['source']==SOURCE
    d.check_plan(prepared/'plan.json',plan,t,frozen_sources=True)
    p=prepared/'frozen-parent/evidence.json';p.chmod(0o600);p.write_text('{}')
    with pytest.raises(ValueError):d.check_plan(prepared/'plan.json',plan,t,frozen_sources=True)


def test_repair_prepare_failure_burns_parent_before_external_work(parent,tmp_path,monkeypatch):
    root,t,xs,args=parent
    monkeypatch.setattr(d.wd,'ruler',lambda *a:({'hash':'a'*64},{'identity_hash':'b'*64}))
    args.repair_parent_run=str(root);args.cuda_ruler_profile='offline-fixture';args.output=str(tmp_path/'plans')
    assert d.run(args)==0
    prepared=next(Path(args.output).iterdir())/'plan.json';plan=d.read_object(prepared)
    events=[]
    class FailPrepare:
        trace=[];forbidden_actions=[];work_counts={}
        def __init__(self,**kwargs):pass
        def prepare(self,*a):
            assert nr.claim_path(plan['repair_parent']).exists()
            assert (prepared.parent/'execution-claim.json').exists()
            events.append('prepare');raise ValueError('offline prepare failure')
        def cleanup(self):events.append('cleanup');return dict(quiescent=True,removed=True)
    monkeypatch.setattr(d,'CodegenSSH',FailPrepare)
    monkeypatch.setattr(lc,'bounded_dispatch',lambda *a:pytest.fail('unexpected HTTP'))
    args.execute=True;args.approved_plan=str(prepared);args.plan_hash=plan['hash']
    assert d.run(args)==1 and events==['prepare','cleanup']
    assert nr.claim_path(plan['repair_parent']).exists()
    with pytest.raises(ValueError,match='consumed'):d.run(args)
