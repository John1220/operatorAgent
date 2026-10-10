"""Local frozen-plan safety tests only; no SSH, GPU or performance claims."""
import copy
import json
from pathlib import Path
import pytest
from benchmarks import workload_search_h20 as driver
from operator_agent import run_contract as rc


def args(tmp_path,*argv):
    return driver.parser().parse_args(['--output',str(tmp_path)]+list(argv))


def test_local_plan_allocates_no_inputs_or_transport(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('local plan attempted SSH/allocation')
    monkeypatch.setattr(driver.wd,'BoundedProfileSSH',forbidden)
    monkeypatch.setattr(driver.wd,'primary',forbidden)
    assert driver.run(args(tmp_path))==0
    root=next(tmp_path.iterdir());plan=json.loads((root/'plan.json').read_text())
    assert root.stat().st_mode&0o777==0o700
    assert (root/'plan.json').stat().st_mode&0o777==0o444
    assert plan['hash']==rc.digest({k:v for k,v in plan.items() if k!='hash'})
    assert [c['name'] for c in plan['candidates']]==['warps-4','warps-8']
    assert all(c['hash']==rc.code_hash(c['source']) for c in plan['candidates'])
    assert plan['contract']['min_speedup']==10. and plan['contract']['aggregate']=='weighted_geomean'
    assert plan['controls']['oracle_total']==5 and plan['controls']['max_rounds']==3
    from operator_agent.budget import BudgetPlan
    budget=BudgetPlan(**plan['controls']['budget'])
    assert budget.oracle_total==5 and budget.llm_tokens==128
    assert plan['controls']['max_rpcs']==512 and plan['controls']['wall_clock_s']==1800
    assert plan['input_storage_limit']==256*1024*1024
    assert not plan['retry_windows'] and not plan['adaptive']
    assert not json.loads((root/'summary.json').read_text())['completed']


@pytest.mark.parametrize('argv',[['--execute'],['--execute','--gpu','2']])
def test_missing_explicit_execution_controls_fail_before_side_effects(tmp_path,argv):
    with pytest.raises(ValueError,match='--gpu AND --cuda-ruler-profile'):driver.run(args(tmp_path,*argv))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('argv',[['--min-speedup','1'],['--repeat','1'],['--gpu','8'],['--weights','2,1']])
def test_cli_cannot_relax_frozen_search_or_measurement_policy(tmp_path,argv):
    with pytest.raises(SystemExit):args(tmp_path,*argv)


def test_two_sources_only_change_fixed_launch_config():
    first,second=driver.candidates()
    assert first['source'].replace('num_warps=4','num_warps=8')==second['source']
    assert 'enable_fp_fusion=False' in first['source']


@pytest.mark.parametrize('completed,source,allowed', ((0,0,True),(1,1,True),
    (0,1,False),(1,0,False),(2,0,False),(2,1,False),(0,-1,False)))
def test_candidate_guard_prevents_reordering_retry_third_or_repaired_source(completed,source,allowed):
    cs=driver.candidates();plan=dict(candidates=cs)
    code=cs[source]['source'] if source>=0 else cs[0]['source']+'\n# undeclared repair'
    rounds=[(cs[i%2]['source'],[]) for i in range(completed)]
    if allowed:driver.check_candidate(plan,rounds,code)
    else:
        with pytest.raises(ValueError,match='refused before verification'):driver.check_candidate(plan,rounds,code)


@pytest.mark.parametrize('which',('plan','task','source'))
def test_guard_rejects_drift_before_dispatch(tmp_path,monkeypatch,which):
    a=args(tmp_path);t,plan=driver.declaration(a);path=tmp_path/'plan.json';rc.atomic_json(path,plan)
    driver.check_plan(path,plan,t)
    if which=='task':t.constraints.min_speedup=1.
    elif which=='source':monkeypatch.setattr(driver.wd,'evaluator_sources',lambda:{'changed':'source'})
    else:
        changed=copy.deepcopy(plan);changed['candidates'].reverse();rc.atomic_json(path,changed)
    with pytest.raises(ValueError,match='drift'):driver.check_plan(path,plan,t)


def test_invalid_budget_is_rejected_before_any_transport_or_input_allocation(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('invalid budget reached transport/allocation')
    monkeypatch.setattr(driver.wd,'BoundedProfileSSH',forbidden)
    monkeypatch.setattr(driver.wd,'primary',forbidden)
    monkeypatch.setattr(driver,'declaration',lambda a:(None,dict(controls=dict(
        budget=dict(oracle_total=5,wall_clock_s=1800,llm_tokens=0)))))
    a=args(tmp_path,'--execute','--gpu','2','--cuda-ruler-profile','unused-before-budget')
    with pytest.raises(ValueError):driver.run(a)
    assert not list(tmp_path.iterdir())


def durable_fixture(tmp_path,monkeypatch,second_wins=False):
    """Production save/load with deterministic fake timings, NOT GPU evidence."""
    from operator_agent.verifier import Verifier
    from operator_agent.budget import BudgetPlan
    from operator_agent.contract import freeze_from_task
    from tests.test_search_objective import execute
    from tests.test_workload_evidence import task
    rounds=[];original=Verifier.verify
    def capture(self,*a,**k):
        results=original(self,*a,**k);rounds.append((k['kernel_code'],results));return results
    monkeypatch.setattr(Verifier,'verify',capture)
    ratios=((1.5,1.5),(2.,2.)) if second_wins else ((2.,2.),(3.,.95))
    art,c,_,_,_=execute(tmp_path,ratios=ratios,max_rounds=6,budget=BudgetPlan(oracle_total=3))
    t=task(min_speedup=10.)
    plan=dict(contract=rc.json_value(freeze_from_task(t).as_dict()),measurement_protocol=rc.protocol_identity())
    return art,c,t,rounds,plan


@pytest.mark.parametrize('second_wins',(False,True))
def test_durable_audit_reloads_actual_selected_product_and_sealed_checkpoint_without_writes(tmp_path,monkeypatch,second_wins):
    from tests.test_workload_evidence import inputs
    art,c,t,rounds,plan=durable_fixture(tmp_path,monkeypatch,second_wins)
    folder=Path(art.saved_dir)
    before={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
    def forbidden(*a,**k):raise AssertionError('read-only audit attempted transport/save')
    monkeypatch.setattr(driver.wd,'BoundedProfileSSH',forbidden)
    monkeypatch.setattr(driver.ArtifactStore,'save',forbidden)
    result=driver.audit_durable(art,rounds,plan,task=t,input_sets=[inputs(16),inputs(32)],controls=c._run_controls)
    assert result['errors']==[] and result['completed']
    assert result['selected_round']==(2 if second_wins else 1) and result['score']==2.
    assert result['candidate_hash']==rc.code_hash(art.code)
    assert result['checkpoint_verified'] and result['protocol_hash']==rc.digest(plan['measurement_protocol'])
    assert result['gpu_measurements']==0 and not result['resumed']
    assert before=={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}


@pytest.mark.parametrize('drift',('missing_selection','missing_search','selected_round','score','source','sketch',
    'verification','benchmark','report','checkpoint','resealed_best','protocol','symlink','confirmation','admission'))
def test_durable_audit_rejects_missing_or_swapped_evidence_without_repairing_files(tmp_path,monkeypatch,drift):
    from tests.test_workload_evidence import inputs
    from operator_agent.search_objective import results_dict
    art,c,t,rounds,plan=durable_fixture(tmp_path,monkeypatch)
    folder=Path(art.saved_dir)
    def change(name,modify):
        p=folder/name;raw=json.loads(p.read_text());modify(raw);p.write_text(json.dumps(raw))
    if drift=='missing_selection':change('result.json',lambda d:d.pop('candidate_selection'))
    elif drift=='missing_search':change('result.json',lambda d:d.pop('search_objective'))
    elif drift=='selected_round':change('result.json',lambda d:d['candidate_selection'].update(selected_round=2))
    elif drift=='score':change('result.json',lambda d:d['search_objective']['best'].update(score=100.))
    elif drift=='source':(folder/'code.py').write_text(rounds[-1][0])
    elif drift=='sketch':change('sketch.json',lambda d:d.update(version=100))
    elif drift=='verification':(folder/'verification.json').write_text(json.dumps(results_dict(rounds[-1][1])))
    elif drift=='benchmark':(folder/'benchmark.json').write_text(json.dumps(rounds[-1][1][-1].metrics))
    elif drift=='report':(folder/'report.md').write_text('claimed winner without evidence')
    elif drift=='checkpoint':change('execution_snapshot.json',lambda d:d.update(round_done=100))
    elif drift=='resealed_best':
        p=folder/'execution_snapshot.json';s=json.loads(p.read_text())
        s['search_objective']['best']['score']=100.;rc.seal(s,c._run_binding);rc.atomic_json(p,s)
    elif drift=='protocol':plan['measurement_protocol']={'unmeasured':'different'}
    elif drift=='symlink':
        p=folder/'sketch.json';copy_path=tmp_path/'sketch-copy.json';copy_path.write_bytes(p.read_bytes())
        p.unlink();p.symlink_to(copy_path)
    else:(folder/('confirmation.json' if drift=='confirmation' else 'admission.json')).write_text('{}')
    before={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
    def forbidden(*a,**k):raise AssertionError('failed audit attempted transport/save')
    monkeypatch.setattr(driver.wd,'BoundedProfileSSH',forbidden)
    monkeypatch.setattr(driver.ArtifactStore,'save',forbidden)
    result=driver.audit_durable(art,rounds,plan,task=t,input_sets=[inputs(16),inputs(32)],controls=c._run_controls)
    assert not result['completed'] and result['errors']
    assert before=={p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
