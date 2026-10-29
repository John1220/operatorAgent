"""Offline/mocked guards only. No real HTTP, SSH, CUDA or performance claims."""
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks import workload_codegen_h20 as d
from operator_agent import llm_client as lc
from operator_agent import run_contract as rc
from operator_agent.artifact_store import ArtifactStore
from operator_agent.llm_budget import LLMCallBudget, LLMCallLimits, LLMCallRefused
from operator_agent.schemas import VerificationLevel as Level, VerificationResult
from operator_agent.single_candidate_coder import SingleCandidateCoder
from tests.test_single_candidate_coder import CODE, response, conductor
from tests.test_workload_evidence import task, inputs, backend


def args(tmp_path, *extra):
    return d.parser().parse_args(['--output', str(tmp_path)]+list(extra))


def test_default_plan_has_no_model_config_transport_or_large_allocation(tmp_path, monkeypatch):
    forbidden = lambda *a, **k: pytest.fail('plan spent model/GPU/allocation')
    monkeypatch.setattr(d, 'CodegenSSH', forbidden)
    monkeypatch.setattr(d, 'PlanLLMClient', forbidden)
    monkeypatch.setattr(d.wd, 'primary', forbidden)
    monkeypatch.setenv('LLM_BASE_URL', 'http://do-not-use.invalid')
    monkeypatch.setenv('LLM_API_KEY', 'NEVER_DISCLOSE')
    assert d.run(args(tmp_path)) == 0
    root = next(tmp_path.iterdir()); plan = d.read_object(root/'plan.json')
    assert root.stat().st_mode & 0o777 == 0o700
    assert (root/'plan.json').stat().st_mode & 0o777 == 0o444
    assert plan['model'] is None and 'request' not in plan
    assert 'NEVER_DISCLOSE' not in json.dumps(plan)
    assert plan['contract']['min_speedup'] == 10.
    assert plan['model_limits']['max_calls'] == 1
    assert plan['model_limits']['total_output_token_reservations'] == 4096
    assert plan['generation']['max_candidates'] == 1 and plan['generation']['repairs'] == 0
    assert plan['controls']['max_rounds'] == 1
    assert plan['controls']['budget']['oracle_total'] == 20
    assert plan['budget_accounting']['unit'] == 'candidate_verification_dispatch_per_input_group'
    assert plan['budget_accounting']['fixed_dispatch_sequence_unchanged']
    assert plan['runtime_boundary']['hostile_code_sandbox'] is False
    assert not plan['runtime_boundary']['vram_cap'] and not plan['runtime_boundary']['network_isolated']
    assert [m['storage_bytes']//1024**2 for m in plan['input_bounds']] == [240,192]
    summary = d.read_object(root/'summary.json')
    assert not summary['completed'] and not summary['accepted']
    assert summary['model_attempts'] == summary['verifier_dispatches'] == 0
    assert not (root/'model-journal').exists()
    t, _ = d.declaration(args(tmp_path)); d.check_plan(root/'plan.json', plan, t, frozen_sources=True)


def test_explicit_model_plan_freezes_exact_payload_but_does_not_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_API_KEY', 'PRIVATE_KEY')
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: pytest.fail('prepare sent HTTP'))
    assert d.run(args(tmp_path,'--model-url','https://fixture.invalid/v1','--model','fixture')) == 0
    p = d.read_object(next(tmp_path.iterdir())/'plan.json')
    raw = json.dumps(p['request']['body'], ensure_ascii=False, allow_nan=False)
    assert p['request']['sha256'] == rc.code_hash(raw)
    assert p['request']['bytes'] == len(raw.encode()) < 65536
    assert p['model']['credential_present'] is True
    assert 'PRIVATE_KEY' not in json.dumps(p) and 'https://fixture.invalid' not in json.dumps(p)
    assert p['disclosure']['tensor_values'] is False and p['disclosure']['template_source'] is False
    context = json.loads(p['request']['body']['messages'][1]['content'])
    assert context['input_metadata'] == [{'X':dict(shape=[8192,15360],dtype='float16',strides=[30720,2])},
                                        {'X':dict(shape=[8192,12288],dtype='float16',strides=[24576,2])}]
    assert context['contract']['reference_contract'] == p['contract']['reference_contract']


@pytest.mark.parametrize('extra', [('--execute',), ('--execute','--gpu','2'),
    ('--execute','--gpu','2','--cuda-ruler-profile','unused'), ('--approved-plan','unused'),
    ('--model','fixture'), ('--model-url','https://fixture.invalid')])
def test_missing_explicit_controls_refused_before_files_transport_or_arrays(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(d, 'CodegenSSH', lambda *a, **k: pytest.fail('unapproved SSH'))
    monkeypatch.setattr(d.wd, 'primary', lambda *a: pytest.fail('unapproved arrays'))
    with pytest.raises(ValueError): d.run(args(tmp_path,*extra))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('extra', [('--max-calls','2'), ('--max-rounds','2'), ('--min-speedup','1'),
    ('--repair','1'), ('--repeat','1'), ('--allow-fallback',), ('--resume-from','old'), ('--gpu','8')])
def test_cli_cannot_expand_candidates_relax_ruler_or_resume(tmp_path, extra):
    with pytest.raises(SystemExit): args(tmp_path,*extra)


@pytest.mark.parametrize('url', ['file:///etc/passwd','https://a:b@fixture.invalid/v1',
    'https://fixture.invalid?key=secret','https://fixture.invalid/#secret','https://fixture.invalid:bad'])
def test_bad_endpoint_refused_without_disclosing_it(tmp_path, url):
    with pytest.raises(ValueError, match='unsafe'): d.declaration(args(tmp_path,'--model-url',url,'--model','fixture'))


@pytest.mark.parametrize('what', ['plan','task','source','snapshot','symlink','context'])
def test_frozen_plan_and_exact_sources_drift_refused(tmp_path, monkeypatch, what):
    t, plan = d.declaration(args(tmp_path)); root=tmp_path/'frozen'; root.mkdir(); d.freeze(root,plan)
    if what=='task': t.constraints.min_speedup=1.
    elif what=='source': monkeypatch.setattr(d,'sources',lambda:{'different':'source'})
    elif what=='snapshot':
        p=root/'frozen-sources'/'operator_agent'/'conductor.py';p.chmod(0o600);p.write_text('different')
    elif what=='symlink':
        p=root/'plan.json';p.unlink();p.symlink_to(tmp_path/'missing')
    else:
        changed=copy.deepcopy(plan)
        if what=='plan': changed['controls']['max_rounds']=2
        else: changed['generation']['context']['input_metadata'][1]['X']['strides'][0]=2
        rc.atomic_json(root/'plan.json',changed)
    with pytest.raises((ValueError, OSError)): d.check_plan(root/'plan.json',plan,t,frozen_sources=True)


def test_prepared_plan_claim_is_exclusive_and_failure_does_not_refund(tmp_path):
    p=tmp_path/'plan.json';rc.atomic_json(p,{'unused':True})
    d.claim_execution(p,tmp_path/'first',dict(hash='a'*64))
    first=(tmp_path/'execution-claim.json').read_bytes()
    with pytest.raises(ValueError,match='consumed'):d.claim_execution(p,tmp_path/'second',dict(hash='a'*64))
    assert (tmp_path/'execution-claim.json').read_bytes()==first


def test_dangling_claim_symlink_cannot_be_overwritten(tmp_path):
    p=tmp_path/'plan.json';rc.atomic_json(p,{})
    marker=tmp_path/'execution-claim.json';marker.symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError,match='consumed'):d.claim_execution(p,tmp_path/'run',dict(hash='a'*64))
    assert marker.is_symlink() and not (tmp_path/'missing').exists()


def generation_fixture(tmp_path, monkeypatch):
    t=task(min_speedup=10.);t.target.dsl='triton';xs=[inputs(16),inputs(32)]
    sketch=d.UnifiedSketch(algorithm='matmul',target_dsl='triton')
    client=d.PlanLLMClient(base_url='https://fixture.invalid/v1',model='fixture',api_key='PRIVATE_KEY',
        temperature=.1,timeout=60,call_budget=LLMCallBudget(d.LIMITS,tmp_path/'model-journal'))
    coder=SingleCandidateCoder(client,t,sketch,xs)
    gen=coder.declaration();identity=client.call_budget.summary()['identity'];body=d.request_body(gen,identity)
    raw=json.dumps(body,ensure_ascii=False,allow_nan=False)
    plan=dict(generation=gen,contract=gen['context']['contract'],model=identity,
        model_limits=rc.json_value(d.asdict(d.LIMITS)),measurement_protocol=rc.protocol_identity(),
        request=dict(body=body,sha256=rc.code_hash(raw),bytes=len(raw.encode())))
    client.frozen_plan=plan;client.plan_guard=lambda:None
    send=Mock(return_value=response());monkeypatch.setattr(lc,'bounded_dispatch',send)
    return t,xs,coder,client,plan,send


def retain_model(root,client):
    rc.atomic_json(root/'model-budget.json',client.call_budget.summary(),readonly=True)
    rc.atomic_json(root/'llm-usage.json',client.cost_tracker.summary(),readonly=True)


def test_production_request_matches_frozen_bytes_and_journal_audit_is_readonly(tmp_path, monkeypatch):
    t,xs,coder,client,plan,send=generation_fixture(tmp_path,monkeypatch)
    assert coder.run(coder._sketch,t,test_inputs=xs)==CODE
    assert send.call_count==1
    assert rc.code_hash(send.call_args.args[0]['body_text'])==plan['request']['sha256']
    retain_model(tmp_path,client)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    proof=d.audit_model(tmp_path,plan)
    assert proof['completed'] and proof['source']==CODE and proof['model_calls']==1
    assert proof['gpu_measurements']==0 and not proof['restored_eligibility']
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('what', ['request','identity','guard'])
def test_model_plan_guard_runs_before_call_reservation(tmp_path,monkeypatch,what):
    t,xs,coder,client,plan,send=generation_fixture(tmp_path,monkeypatch)
    if what=='request':plan['request']['body']['messages'][0]['content']='unapproved'
    elif what=='identity':plan['model']['model']='other'
    else:client.plan_guard=lambda:(_ for _ in ()).throw(ValueError('plan drift'))
    with pytest.raises(LLMCallRefused):coder.run(coder._sketch,t,test_inputs=xs)
    assert send.call_count==0 and client.call_budget.summary()['reserved_calls']==0


@pytest.mark.parametrize('what', ['missing','symlink','call','extra_call','candidate','usage','budget','declaration','plan_request'])
def test_offline_model_audit_rejects_missing_torn_forged_evidence(tmp_path,monkeypatch,what):
    t,xs,coder,client,plan,_=generation_fixture(tmp_path,monkeypatch)
    coder.run(coder._sketch,t,test_inputs=xs);retain_model(tmp_path,client)
    def change(path,fn):
        raw=json.loads(path.read_text());fn(raw);path.chmod(0o600);path.write_text(json.dumps(raw))
    root=tmp_path/'model-journal'
    if what=='missing':(root/'call-001.json').unlink()
    elif what=='symlink':
        p=root/'candidate.json';data=p.read_bytes();p.unlink();q=tmp_path/'other';q.write_bytes(data);p.symlink_to(q)
    elif what=='call':change(root/'call-001.json',lambda r:r.update(response_text='{}'))
    elif what=='extra_call':rc.atomic_json(root/'call-002.json',{})
    elif what=='candidate':change(root/'candidate.json',lambda r:r.update(code=CODE+'\n# forged'))
    elif what=='usage':change(tmp_path/'llm-usage.json',lambda r:r.update(total_tokens=1))
    elif what=='budget':change(tmp_path/'model-budget.json',lambda r:r.update(reserved_calls=0))
    elif what=='declaration':change(root/'generation-declaration.json',lambda r:r.update(repairs=1))
    else:plan['request']['bytes']+=1
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    proof=d.audit_model(tmp_path,plan)
    assert not proof['completed'] and proof['errors']
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def product_fixture(tmp_path,monkeypatch,phase='valid_below_target'):
    t,xs,coder,client,plan,_=generation_fixture(tmp_path,monkeypatch)
    def change(p,n,count):
        if phase=='degraded' and n==32:p['gpu_util_pct']=80
    b,_=backend(change);c=conductor(tmp_path,coder,b)
    c.verifier._verify_compile=lambda *a,**k:VerificationResult(Level.L1_COMPILE,phase!='l1_failed',
        'fake compile only',error_log='original mock diagnostic' if phase=='l1_failed' else None,metrics={'static_guards':True})
    c.verifier._verify_function=lambda *a,**k:VerificationResult(Level.L2_FUNCTION,phase!='l2_failed',
        'fake correctness only',metrics={'max_error':0.})
    original=c.verifier.verify;rounds=[]
    def capture(*a,**k):
        rs=original(*a,**k);rounds.append((k['kernel_code'],rs));return rs
    c.verifier.verify=capture
    with contextlib.redirect_stdout(io.StringIO()):art=c.execute(t,xs,target_gflops=0.)
    retain_model(tmp_path,client);proof=d.audit_model(tmp_path,plan)
    assert proof['completed']
    return t,xs,c,art,rounds,plan,proof


@pytest.mark.parametrize('phase',['valid_below_target','degraded','l1_failed','l2_failed'])
def test_single_product_audit_reads_failure_and_degraded_without_backfill(tmp_path,monkeypatch,phase):
    t,xs,c,art,rounds,plan,model=product_fixture(tmp_path,monkeypatch,phase)
    folder=Path(art.saved_dir);before={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
    monkeypatch.setattr(ArtifactStore,'save',lambda *a:pytest.fail('audit wrote artifact'))
    proof=d.audit_durable(art,rounds,plan,model,task=t,input_sets=xs,controls=c._run_controls)
    assert proof['completed'],proof
    assert proof['status']==('correctness_failed' if phase in ('l1_failed','l2_failed') else phase)
    assert proof['gpu_measurements']==0 and not proof['resumed'] and not proof['restored_eligibility']
    assert before=={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}


@pytest.mark.parametrize('what',['source','sketch','verification','benchmark','report','checkpoint',
    'model_budget','lineage','usage','selection','confirmation','symlink'])
def test_single_product_audit_rejects_forged_missing_or_repaired_product(tmp_path,monkeypatch,what):
    t,xs,c,art,rounds,plan,model=product_fixture(tmp_path,monkeypatch)
    folder=Path(art.saved_dir)
    def change(name,fn):
        p=folder/name;raw=json.loads(p.read_text());fn(raw);p.write_text(json.dumps(raw))
    if what=='source':(folder/'code.py').write_text(CODE+'\n# repaired')
    elif what=='sketch':change('sketch.json',lambda r:r.update(version=9))
    elif what=='verification':change('verification.json',lambda r:r[0].update(passed=False))
    elif what=='benchmark':change('benchmark.json',lambda r:r.update(gflops=9999))
    elif what=='report':(folder/'report.md').write_text('fictional acceptance')
    elif what=='checkpoint':change('execution_snapshot.json',lambda r:r.update(round_done=2))
    elif what=='model_budget':change('result.json',lambda r:r['llm_call_budget'].update(reserved_calls=2))
    elif what=='lineage':change('result.json',lambda r:r['code_generation']['lineage'].update(source_author='template'))
    elif what=='usage':change('result.json',lambda r:r['llm_cost'].update(total_tokens=0))
    elif what=='selection':change('result.json',lambda r:r['candidate_selection'].update(selected_round=2))
    elif what=='confirmation':rc.atomic_json(folder/'confirmation.json',{})
    else:
        p=folder/'sketch.json';q=tmp_path/'sketch-copy';q.write_bytes(p.read_bytes());p.unlink();p.symlink_to(q)
    before={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
    proof=d.audit_durable(art,rounds,plan,model,task=t,input_sets=xs,controls=c._run_controls)
    assert not proof['completed'] and proof['errors']
    assert before=={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}


def ssh_fixture(monkeypatch):
    _,plan=d.declaration(d.parser().parse_args([]));events=[]
    monkeypatch.setattr(d.wd.BoundedProfileSSH,'_ssh_exec_json',lambda self,p:events.append(copy.deepcopy(p)) or {})
    b=d.CodegenSSH(plan_guard=lambda:events.append('plan'),candidate_guard=lambda code:events.append('source'),plan=plan,host='fixture')
    payload=dict(action='benchmark_with_baselines',code=CODE,precision='fp16',kernel_type='layernorm',
        reference_source='',reference_contract=d.registered_reference().content_dict(),warmup=10,repeat=30,
        baseline_priority=['pytorch_eager'],inputs={'tensors':{'X':{'shape':[8192,15360]}}})
    return b,payload,events


def test_remote_candidate_case_order_and_window_cap_count_before_dispatch(monkeypatch):
    b,p,events=ssh_fixture(monkeypatch)
    for i in range(4):
        p['inputs']['tensors']['X']['shape']=[8192,15360 if i%2==0 else 12288]
        b._ssh_exec_json(p)
    assert len(b.bench_dispatches)==4
    before=len([r for r in events if isinstance(r,dict)])
    with pytest.raises(ValueError,match='retried'):b._ssh_exec_json(p)
    assert len([r for r in events if isinstance(r,dict)])==before


@pytest.mark.parametrize('what',['order','precision','reference','repeat','code'])
def test_remote_refuses_drift_before_transport(monkeypatch,what):
    b,p,events=ssh_fixture(monkeypatch)
    if what=='order':p['inputs']['tensors']['X']['shape']=[8192,12288]
    elif what=='precision':p['precision']='fp32'
    elif what=='reference':p['reference_contract']={}
    elif what=='repeat':p['repeat']=1
    else:b.candidate_guard=lambda code:(_ for _ in ()).throw(ValueError('source drift'))
    with pytest.raises(ValueError):b._ssh_exec_json(p)
    assert not any(isinstance(r,dict) for r in events)


def test_failed_remote_dispatch_does_not_refund_window(monkeypatch):
    b,p,_=ssh_fixture(monkeypatch)
    monkeypatch.setattr(d.wd.BoundedProfileSSH,'_ssh_exec_json',lambda *a:(_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(TimeoutError):b._ssh_exec_json(p)
    assert len(b.bench_dispatches)==1
    with pytest.raises(ValueError,match='retried'):b._ssh_exec_json(p)


def test_compile_and_heartbeat_caps_are_pre_io(monkeypatch):
    b,_,events=ssh_fixture(monkeypatch)
    b._ssh_exec_json(dict(action='compile',code=CODE))
    with pytest.raises(ValueError,match='second'):b._ssh_exec_json(dict(action='compile',code=CODE))
    for _ in range(2):b._ssh_exec_json(dict(action='cuda_calibration_probe',mode='heartbeat'))
    with pytest.raises(ValueError,match='extra'):b._ssh_exec_json(dict(action='cuda_calibration_probe',mode='heartbeat'))
    assert len([r for r in events if isinstance(r,dict)])==3


def test_broken_plan_does_not_block_owned_cleanup(monkeypatch):
    b,_,events=ssh_fixture(monkeypatch)
    b.plan_guard=lambda:(_ for _ in ()).throw(ValueError('drift'))
    with pytest.raises(ValueError):b.guard('rpc')
    b.cleaning=True;b.guard('rpc')
    b._ssh_exec_json(dict(action='shutdown'))
    assert events[-1]==dict(action='shutdown')


def prepared_execution(tmp_path,monkeypatch):
    # Fake profile identity solely for transport-negative tests, never a ruler.
    monkeypatch.setattr(d.wd,'ruler',lambda *a:({'hash':'a'*64},{'identity_hash':'b'*64}))
    extras=['--gpu','2','--cuda-ruler-profile','fixture','--model-url','https://fixture.invalid/v1','--model','fixture']
    assert d.run(args(tmp_path,*extras))==0
    prepared=next(tmp_path.iterdir())/'plan.json';plan=d.read_object(prepared)
    a=args(tmp_path,*extras,'--execute','--approved-plan',str(prepared),'--plan-hash',plan['hash'])
    return prepared,a


def test_authorized_hash_and_model_destination_mismatch_fail_before_new_files_or_transport(tmp_path,monkeypatch):
    prepared,a=prepared_execution(tmp_path,monkeypatch);before=set(tmp_path.iterdir())
    monkeypatch.setattr(d,'CodegenSSH',lambda *a,**k:pytest.fail('drift reached SSH'))
    a.model_url='https://unapproved.invalid/v1'
    with pytest.raises(ValueError,match='authorized exact plan'):d.run(a)
    assert set(tmp_path.iterdir())==before and not (prepared.parent/'execution-claim.json').exists()


def test_prepare_failure_still_cleans_and_retains_zero_model_attempt(tmp_path,monkeypatch):
    prepared,a=prepared_execution(tmp_path,monkeypatch);events=[]
    class B:
        trace=[];forbidden_actions=[];work_counts={}
        def __init__(self,**kwargs):events.append('construct')
        def prepare(self,*a):events.append('prepare');raise ValueError('failed')
        def cleanup(self):events.append('cleanup');return dict(quiescent=True,removed=True)
    monkeypatch.setattr(d,'CodegenSSH',B)
    monkeypatch.setattr(lc,'bounded_dispatch',lambda *a:pytest.fail('prepare failure spent HTTP'))
    monkeypatch.setattr(d.wd,'primary',lambda *a:pytest.fail('prepare failure allocated inputs'))
    assert d.run(a)==1 and events==['construct','prepare','cleanup']
    root=next(p for p in tmp_path.iterdir() if p!=prepared.parent);s=d.read_object(root/'summary.json')
    assert s['model_attempts']==s['verifier_dispatches']==0 and not s['completed']
    assert s['cleanup']['quiescent'] and s['status']=='preparation_failed'
    assert (root/'evidence.json').is_file() and (prepared.parent/'execution-claim.json').is_file()


def test_claim_cannot_dispatch_again_even_after_prepare_failure(tmp_path,monkeypatch):
    prepared,a=prepared_execution(tmp_path,monkeypatch)
    d.claim_execution(prepared,tmp_path/'previous',d.read_object(prepared))
    monkeypatch.setattr(d,'CodegenSSH',lambda *a,**k:pytest.fail('spent a consumed plan'))
    monkeypatch.setattr(lc,'bounded_dispatch',lambda *a:pytest.fail('spent a consumed plan'))
    assert d.run(a)==1
    root=next(p for p in tmp_path.iterdir() if p!=prepared.parent)
    assert d.read_object(root/'summary.json')['model_attempts']==0


def raw_fixture(tmp_path):
    """Entirely FABRICATED receipt/Oracle/RPC fields; no CUDA or allocations.

    Exercises the independent read-only auditor, not a numerical implementation
    or machine qualification. The regular production integration is above.
    """
    from tests.test_workload_decision_driver import _metrics
    from operator_agent.reference_contract import check_observed
    t,plan=d.declaration(d.parser().parse_args(['--gpu','2']))
    spec=d.registered_reference();receipt=check_observed(spec,spec.execution.content_dict())
    contract=d.load_dict(plan['contract']);suite=d.wd.as_suite(contract.correctness_suite)
    binding=dict(candidate_hash=rc.code_hash(CODE),contract_hash=contract.hash)
    trace=[dict(request=dict(action='compile',code=CODE),response={'success':True})]
    fidelity=dict(representation='torch-storage-v1',observed={'X':{}},tensor_devices={'X':'cuda:0'})
    def rpc(action,key,res):
        shape=list(map(int,key.split('=')[1].split('x')));size=shape[0]*shape[1]*2
        packet=dict(protocol=d.wd.INPUT_PROTOCOL,hash=rc.digest(shape),
            storages=[dict(bytes=size)],scalars={},tensors={'X':{'shape':shape}})
        res['input_transfer']=dict(protocol=d.wd.INPUT_PROTOCOL,manifest_hash=packet['hash'],
            storage_bytes=size,uploaded_bytes=size,content_verified=True)
        trace.append(dict(request=dict(action=action,code=CODE,inputs=packet,precision='fp16',
            max_error=contract.max_error,reference_contract=spec.content_dict(),
            warmup=10,repeat=30,baseline_priority=['pytorch_eager']),response=res))
    def numeric():
        return dict(success=True,device='cuda',output_digest='f'*64,max_error=0.,max_rel_error=0.,
            matched_ratio=1.,bad_elements=0,atol=1e-5,rtol=.01,required_matched_ratio=1.,
            reference_receipt=receipt)
    for key in plan['supplied_cases']:rpc('verify',key,numeric())
    rows=[]
    for case in suite.cases:
        m=numeric();m.update(input_fidelity_verified=True,input_fidelity=fidelity)
        rows.append(dict(case_id=case.case_id,shape_key=case.inputs.key(),verdict='passed',
                         determinism='passed',metrics=copy.deepcopy(m),repeat_metrics=copy.deepcopy(m)))
        rpc('verify',case.inputs.key(),numeric());rpc('verify',case.inputs.key(),numeric())
    m=_metrics(t,'offline-main');m['run_binding']=binding
    for key,row in m['cases'].items():
        rpc('benchmark_with_baselines',key,dict(success=True,device='cuda',selected_gpu=2,
            reference_receipt=receipt,paired=dict(arm_windows=row['arm_windows'],arm_identity=row['arm_identity']),
            candidate=dict(median_time_ms=.2),baselines={'pytorch_eager':{'median_time_ms':.2*row['speedup_vs_baseline']}},
            baseline_selected='pytorch_eager'))
    results=[VerificationResult(Level.L1_COMPILE,True,metrics={'run_binding':binding}),
        VerificationResult(Level.L2_FUNCTION,True,metrics=dict(run_binding=binding,input_fidelity_verified=True,
            input_fidelity=fidelity,correctness=dict(verdict='passed',suite_hash=suite.digest(),
            coverage_lint={'status':'passed'},cases=rows))),VerificationResult(Level.L3_PERFORMANCE,False,metrics=m)]
    art=SimpleNamespace(code=CODE,all_passed=False,metadata={'corpus_admission':{'kept':False}})
    return t,plan,art,[(CODE,results)],trace,tmp_path/'empty-corpus.jsonl'


def test_readonly_raw_cuda_audit_accepts_only_complete_frozen_one_candidate_fixture(tmp_path):
    t,plan,art,rounds,trace,corpus=raw_fixture(tmp_path)
    before=copy.deepcopy((rounds,trace))
    proof=d.audit_raw(art,rounds,trace,plan,task=t,corpus=corpus)
    assert proof['completed'] and proof['accepted'],proof
    assert proof['status']=='valid_below_target' and not proof['deployment_accepted']
    assert len(proof['windows'])==2 and proof['objective']['score']>1.
    assert proof['gpu_measurements']==0 and (rounds,trace)==before and not corpus.exists()


@pytest.mark.parametrize('what',['missing_suite','extra_verify','repeat_digest','suite_metrics','primary',
    'compile','other_code','receipt','transfer','case_rpc','benchmark_policy','extra_window','reused_window','threshold','binding','corpus'])
def test_raw_audit_refuses_torn_forged_or_extra_evidence_without_remeasure(tmp_path,what):
    t,plan,art,rounds,trace,corpus=raw_fixture(tmp_path)
    l2=rounds[0][1][1];m=rounds[0][1][2].metrics
    if what=='missing_suite':l2.metrics['correctness']['cases'].pop()
    elif what=='extra_verify':trace.insert(2,copy.deepcopy(trace[1]))
    elif what=='repeat_digest':l2.metrics['correctness']['cases'][0]['repeat_metrics']['output_digest']='a'*64
    elif what=='suite_metrics':l2.metrics['correctness']['cases'][0]['metrics']['max_error']=9.
    elif what=='primary':trace[1]['response']['success']=False
    elif what=='compile':trace[0]['response']['success']=False
    elif what=='other_code':trace[0]['request']['code']+='\n# repaired'
    elif what=='receipt':trace[1]['response']['reference_receipt']={}
    elif what=='transfer':trace[1]['response']['input_transfer']['content_verified']=False
    elif what=='case_rpc':trace[-1]['response']['candidate']['median_time_ms']=5.
    elif what=='benchmark_policy':trace[-1]['request']['repeat']=1
    elif what=='extra_window':trace.append(copy.deepcopy(trace[-1]))
    elif what=='reused_window':m['cases'][plan['supplied_cases'][1]]['arm_windows']=m['cases'][plan['supplied_cases'][0]]['arm_windows']
    elif what=='threshold':m['performance_verdict']['min_speedup']=1.
    elif what=='binding':rounds[0][1][0].metrics['run_binding']={}
    else:corpus.write_text('unconfirmed append')
    proof=d.audit_raw(art,rounds,trace,plan,task=t,corpus=corpus)
    assert not proof['completed'] and not proof['accepted'] and proof['errors'],proof


def test_raw_degraded_is_auditable_but_never_counts_as_valid_acceptance(tmp_path):
    from operator_agent.eligibility import DEGRADED
    from operator_agent.workload import apply_qualification
    t,plan,art,rounds,trace,corpus=raw_fixture(tmp_path)
    m=rounds[0][1][2].metrics;row=m['cases'][plan['supplied_cases'][1]]
    row.update(measurement_validity='degraded',eligibility=DEGRADED.as_dict())
    row['observation'].update(validity='degraded',eligibility=DEGRADED.as_dict())
    apply_qualification(m,d.load_dict(plan['contract']),plan['supplied_cases'][0])
    proof=d.audit_raw(art,rounds,trace,plan,task=t,corpus=corpus)
    assert proof['completed'] and not proof['accepted'] and proof['status']=='degraded',proof


def test_correctness_failure_keeps_evidence_without_benchmark_acceptance(tmp_path):
    t,plan,art,rounds,trace,corpus=raw_fixture(tmp_path)
    results=rounds[0][1];results[1].passed=False;results.pop()
    trace=[r for r in trace if r['request']['action']!='benchmark_with_baselines']
    proof=d.audit_raw(art,rounds,trace,plan,task=t,corpus=corpus)
    assert proof['completed'] and not proof['accepted'] and proof['status']=='correctness_failed',proof
    trace.append(dict(request={'action':'benchmark_with_baselines','code':CODE},response={}))
    assert not d.audit_raw(art,rounds,trace,plan,task=t,corpus=corpus)['completed']


@pytest.mark.parametrize('bad_confirmation',[False,True])
def test_unexpected_target_pass_still_needs_independent_full_workload_confirmation(tmp_path,bad_confirmation):
    from dataclasses import asdict
    from operator_agent.confirm import ConfirmRecord
    from operator_agent.workload import confirmation_evidence
    from tests.test_workload_decision_driver import _metrics
    t,plan,art,rounds,trace,corpus=raw_fixture(tmp_path)
    binding=dict(candidate_hash=rc.code_hash(CODE),contract_hash=plan['contract']['hash'],
        run_id='a'*32,manifest_hash='b'*64,workload_hash='c'*64,protocol_hash='d'*64,verification_id='first')
    first=_metrics(t,'main',(12.,15.));first['run_binding']=binding
    second=_metrics(t,'confirmation',(11.,14.));second['run_binding']=dict(binding,verification_id='second')
    templates={key:copy.deepcopy(rpc) for key,rpc in zip(plan['supplied_cases'],trace[-2:])}
    for m in (first,second):
        for key,row in m['cases'].items():
            res=dict(success=True,device='cuda',selected_gpu=2,reference_receipt=row['reference_receipt'],
                paired=dict(arm_windows=row['arm_windows'],arm_identity=row['arm_identity']),
                candidate=dict(median_time_ms=.2),baselines={'pytorch_eager':{'median_time_ms':.2*row['speedup_vs_baseline']}},
                baseline_selected='pytorch_eager')
            template=copy.deepcopy(templates[key])
            res['input_transfer']=template['response']['input_transfer'];template['response']=res
            if m is first:
                index=-2 if key==plan['supplied_cases'][0] else -1;trace[index]=template
            else:trace.append(template)
    rounds[0][1][-1].metrics=first;rounds[0][1][-1].passed=True
    record=ConfirmRecord(ok=True,speedup=d.wd.decision_speedup(second),validity='valid',independent=True,
        contract_hash=plan['contract']['hash'],performance_basis='objective',workload_evidence=confirmation_evidence(second))
    art.all_passed=True;art.remeasure=record;folder=tmp_path/'artifact';folder.mkdir();art.saved_dir=str(folder)
    if bad_confirmation:second['run_binding']['verification_id']='first'
    rc.atomic_json(folder/'confirmation.json',dict(first_binding=binding,second_binding=second['run_binding'],
        confirmation=asdict(record),second_metrics=second))
    proof=d.audit_raw(art,rounds,trace,plan,task=t,corpus=corpus)
    assert proof['accepted'] is (not bad_confirmation),proof
    assert not proof['deployment_accepted'] and not corpus.exists()
    assert len([r for r in trace if r['request']['action']=='benchmark_with_baselines'])==4


def test_cleanup_failure_retains_evidence_and_restores_environment(tmp_path,monkeypatch):
    import os
    prepared,a=prepared_execution(tmp_path,monkeypatch)
    before={k:os.environ.get(k) for k in ('OPERATOR_AGENT_LOG_DIR','OPERATOR_AGENT_LOG_KEEP')}
    class B:
        trace=[];forbidden_actions=[];work_counts={}
        def __init__(self,**k):pass
        def prepare(self,*a):raise ValueError('CANDIDATE_SECRET_EXCEPTION_TEXT')
        def cleanup(self):raise TimeoutError('unconfirmed')
    monkeypatch.setattr(d,'CodegenSSH',B)
    monkeypatch.setattr(lc,'bounded_dispatch',lambda *a:pytest.fail('failure retried model'))
    assert d.run(a)==1
    root=next(p for p in tmp_path.iterdir() if p!=prepared.parent);s=d.read_object(root/'summary.json')
    assert not s['completed'] and not s['accepted'] and s['cleanup']['quiescent'] is False
    assert (root/'evidence.json').is_file() and (root/'model-budget.json').is_file()
    assert 'CANDIDATE_SECRET_EXCEPTION_TEXT' not in json.dumps(s)
    assert {k:os.environ.get(k) for k in before}==before


def test_actual_input_drift_after_prepare_cleans_without_model_or_candidate_dispatch(tmp_path,monkeypatch):
    import numpy as np
    from operator_agent.reference_contract import check_observed
    prepared,a=prepared_execution(tmp_path,monkeypatch);events=[]
    spec=d.registered_reference()
    class B:
        trace=[];forbidden_actions=[];work_counts={}
        scratch='/tmp/opag-fidelity-fixture0'
        def __init__(self,**k):pass
        def prepare(self,*a):return dict(selected=dict(index=2,name='NVIDIA H20'))
        def reference_identity(self,*a):return dict(reference_receipt=check_observed(spec,spec.execution.content_dict()))
        def cleanup(self):events.append('cleanup');return dict(quiescent=True,removed=True)
    monkeypatch.setattr(d,'CodegenSSH',B)
    monkeypatch.setattr(d.wd,'primary',lambda *a:{'X':np.zeros((2,2),np.float16)})
    monkeypatch.setattr(lc,'bounded_dispatch',lambda *a:pytest.fail('input drift spent HTTP'))
    assert d.run(a)==1 and events==['cleanup']
    root=next(p for p in tmp_path.iterdir() if p!=prepared.parent);s=d.read_object(root/'summary.json')
    assert s['model_attempts']==s['verifier_dispatches']==0 and not s['accepted']
