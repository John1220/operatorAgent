"""Local frozen-plan safety tests only; no SSH, GPU or performance claims."""
import copy
import json
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


@pytest.mark.parametrize('which',('plan','task','source'))
def test_guard_rejects_drift_before_dispatch(tmp_path,monkeypatch,which):
    a=args(tmp_path);t,plan=driver.declaration(a);path=tmp_path/'plan.json';rc.atomic_json(path,plan)
    driver.check_plan(path,plan,t)
    if which=='task':t.constraints.min_speedup=1.
    elif which=='source':monkeypatch.setattr(driver.wd,'evaluator_sources',lambda:{'changed':'source'})
    else:
        changed=copy.deepcopy(plan);changed['candidates'].reverse();rc.atomic_json(path,changed)
    with pytest.raises(ValueError,match='drift'):driver.check_plan(path,plan,t)
