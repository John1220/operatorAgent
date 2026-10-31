"""Excluded preflight denominators, stop policy and real-tool timeout wiring."""
import json
from pathlib import Path
import pytest
from benchmarks.experiments.s5 import s5_workflow_preflight as pilot, s5_formal_run as workflow
from benchmarks.experiments.s5 import s5_arm_env


@pytest.fixture(autouse=True)
def unit_readiness(monkeypatch):
    # These tests exercise selection/stop policy, not the OS proof validator.
    from benchmarks.experiments.s5 import s5_isolation_probe
    monkeypatch.setattr(s5_isolation_probe,'load_readiness',
                        lambda path:dict(report_path='unit-fixture',runtime={}))


def test_freeze_selects_three_tasks_and_preserves_user_model_settings(tmp_path):
    plan, policy = pilot.freeze(tmp_path / 'plan',isolation_report='unit-fixture')
    assert [c['case_id'] for c in plan['cases']] == list(pilot.TASKS)
    assert [c['order'] for c in plan['cases']] == [['A','B','C'], ['B','C','A'], ['C','A','B']]
    assert plan['preflight'] and not plan['counts_toward_denominator']
    assert plan['planned_runs'] == 9 and policy['max_verifications'] == 27
    assert policy['max_model_reservations'] == 216
    assert plan['model']['reasoning_effort'] == 'medium'
    assert plan['budget']['max_output_tokens'] == 32768
    assert workflow.load_plan(tmp_path / 'plan/plan.json') == plan


def test_bad_selection_cannot_create_a_plan(tmp_path):
    for cases in [[], ['layernorm-tail']*3, ['missing','layernorm-tail','softmax-tail']]:
        with pytest.raises(ValueError):
            workflow.freeze(tmp_path/'bad', case_ids=cases, preflight=True)
    assert not (tmp_path/'bad').exists()


def test_unknown_stops_all_remaining_dispatch_with_fixed_denominator(tmp_path, monkeypatch):
    plan, policy = pilot.freeze(tmp_path / 'plan',isolation_report='unit-fixture')
    monkeypatch.setattr(pilot, 'resources', lambda p: {'gpu_index': 7})
    monkeypatch.setattr(pilot, 'credential', lambda: 'placeholder')
    calls = []
    def run(p,c,a,*args):
        calls.append((c['case_id'],a))
        return dict(case_id=c['case_id'],arm=a,plan_hash=p['hash'],status='unknown',
                    execution_state='interrupted',model_calls=dict(reserved=1),
                    wall_seconds=0.1,repo_path_mentions=0)
    monkeypatch.setattr(workflow, 'run_one', run)
    summary = pilot.execute(tmp_path / 'plan', plan['hash'], policy['hash'])
    assert len(calls) == 1 and summary['dispatched_runs'] == 1
    assert summary['planned_runs'] == 9 and summary['counts']['unknown'] == 9
    assert summary['workflow_gate_passed'] is False
    assert len(list((tmp_path/'plan/records').glob('*.json'))) == 9
    with pytest.raises(FileExistsError):
        pilot.execute(tmp_path/'plan',plan['hash'],policy['hash'])
    assert len(calls) == 1


def test_development_cli_and_mcp_allow_full_service_timeout(tmp_path):
    conf=dict(url='http://127.0.0.1:9/development/verify', token='temporary-test-token',
              contract='a'*64, backend='ssh', timeout=610)
    for arm in 'ABC':
        built=s5_arm_env.build(arm, base_url='http://127.0.0.1:9/v1',arm_dir=tmp_path/arm,
                               tools=True, development=conf)
        bash=next(r for r in built['patch'] if r.get('id')=='bash-sandbox')
        assert bash['config']['timeoutMs']==bash['config']['maxTimeoutMs']==640000
        if arm=='C':
            entries=[v for r in built['patch'] for v in r.get('insert',[])]
            mcp=next(r for r in entries if r['id']=='mcp-operator')
            assert mcp['config']['toolCallTimeoutMs']==640000


def test_no_unisolated_preflight_can_be_frozen(tmp_path):
    with pytest.raises(ValueError,match='read isolation report'):
        pilot.freeze(tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()


def test_budget_stops_do_not_cancel_other_cells(tmp_path,monkeypatch):
    plan,policy=pilot.freeze(tmp_path/'plan',isolation_report='unit-fixture')
    monkeypatch.setattr(pilot,'resources',lambda p:{'gpu_index':7})
    monkeypatch.setattr(pilot,'credential',lambda:'placeholder')
    calls=[]
    def run(p,c,a,*args):
        calls.append(a)
        return dict(case_id=c['case_id'],arm=a,plan_hash=p['hash'],status='unknown',
            execution_state='interrupted',budget_refusal='call_budget_exhausted',
            development_audit=dict(complete=True,plugin_verification_observed=False),
            model_calls=dict(reserved=24),wall_seconds=.1,repo_path_mentions=0)
    monkeypatch.setattr(workflow,'run_one',run)
    s=pilot.execute(tmp_path/'plan',plan['hash'],policy['hash'])
    assert len(calls)==9 and s['dispatched_runs']==9 and s['stopped_reason']is None
    assert s['mcp_used_runs']==0 and s['counts']['unknown']==9


def test_launch_refusal_still_retains_every_planned_cell(tmp_path,monkeypatch):
    plan,policy=pilot.freeze(tmp_path/'plan',isolation_report='unit-fixture')
    monkeypatch.setattr(pilot,'resources',lambda p:{'gpu_index':7})
    monkeypatch.setattr(pilot,'credential',lambda:'placeholder')
    monkeypatch.setattr(workflow,'run_one',lambda *a,**kw:(_ for _ in ()).throw(ValueError('drift')))
    s=pilot.execute(tmp_path/'plan',plan['hash'],policy['hash'])
    assert s['dispatched_runs']==1 and s['counts']['unknown']==9
    assert s['stopped_reason'] and len(list((tmp_path/'plan/records').glob('*.json')))==9


def test_shared_gpu_pin_and_policy_are_frozen_for_all_services(tmp_path,monkeypatch):
    gpu=dict(index=2,uuid='GPU-13c6c4b4-45ce-8963-1122-df2b3d72fb76',name='NVIDIA H20')
    plan,policy=pilot.freeze(tmp_path/'shared',isolation_report='unit-fixture',
                            gpu=gpu,shared_compute_pids=[10,20])
    assert plan['verification']['gpu']==gpu
    assert plan['resource_policy']==policy['resources']
    assert policy['resources']['allowed_compute_pids']==[10,20]
    for c in plan['cases']:
        service=json.loads((tmp_path/'shared/services'/c['case_id']/'plan.json').read_text())
        assert (service['gpu_index'],service['gpu_uuid'],service['gpu_name'])==(2,gpu['uuid'],gpu['name'])
    from benchmarks.experiments.s5 import s5_shared_gpu
    observed={}
    def snapshot(pin,p,env):
        observed.update(pin=pin,policy=p);return {'shared':True}
    monkeypatch.setattr(s5_shared_gpu,'snapshot',snapshot)
    monkeypatch.setattr(pilot,'RESOURCE_POLICY',{'modified':'must not be used for shared'})
    assert pilot.resources(plan)=={'shared':True}
    assert observed['policy']==policy['resources'] and observed['pin']['gpu_index']==2


def test_shared_resource_policy_drift_refuses_before_credentials(tmp_path,monkeypatch):
    from operator_agent.core.run_contract import digest
    plan,policy=pilot.freeze(tmp_path/'plan',isolation_report='unit-fixture')
    policy['resources']=dict(mode='shared-known-processes',min_free_mib=8192,
                             max_util_pct=5,allowed_compute_pids=[1])
    policy['hash']=digest({k:v for k,v in policy.items()if k!='hash'})
    (tmp_path/'plan/execution-policy.json').write_text(json.dumps(policy))
    monkeypatch.setattr(pilot,'credential',lambda:pytest.fail('credential reached'))
    monkeypatch.setattr(pilot,'resources',lambda p:pytest.fail('resource dispatch reached'))
    with pytest.raises(ValueError,match='identity drift'):
        pilot.execute(tmp_path/'plan',plan['hash'],policy['hash'])
