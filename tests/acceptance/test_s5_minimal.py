import copy
import json
import pytest
from benchmarks.experiments.s5 import s5_formal_run as flow, s5_minimal_tasks as tasks, s5_minimal_summary as stats
from benchmarks.experiments.s5.s5_outcomes import classify,fatal
from operator_agent.app.hitl_tools import prepare


def test_new_tasks_have_disjoint_primary_specs_and_correct_operands(tmp_path):
    from benchmarks.experiments.s5.s5_task_manifest import manifest
    old={(r['kernel_type'],tuple(r['shape']),r['precision'])for r in manifest()['tasks']}
    rows=tasks.manifest()['tasks'];assert len(rows)==6
    assert not any((r['kernel_type'],tuple(r['shape']),r['precision'])in old for r in rows)
    for r in rows:
        root=tmp_path/r['case_id'];prepare(flow.declaration(r),root)
        assert flow.check_task_inputs(r,json.loads((root/'contract.json').read_text()))


def test_repeat_schedule_is_complete_and_rotated(tmp_path):
    p=flow.freeze(tmp_path/'plan',task_manifest=tasks.manifest(),repeats=2)
    assert p['planned_runs']==36 and len(p['schedule'])==36
    assert len({(r['case_id'],r['repeat_id'],r['arm'])for r in p['schedule']})==36
    assert flow.load_plan(tmp_path/'plan/plan.json')==p
    assert p['schedule'][0]['arm']!=p['schedule'][18]['arm']


def test_stop_classification_preserves_budget_and_unused_tools():
    row=dict(status='unknown',execution_state='interrupted',budget_refusal='call_budget_exhausted',
             development_audit=dict(complete=True,plugin_verification_observed=False))
    assert classify(row)=='call_budget_exhausted' and not fatal(row)
    assert fatal(dict(row,error='identity_drift'))
    assert fatal(dict(row,development_audit=dict(complete=False)))
    assert fatal(dict(row,development=dict(records=[dict(status='unknown')])))
    assert classify(dict(row,status='failed',budget_refusal=None))=='candidate_rejected'
    assert not fatal(dict(row,status='passed',budget_refusal=None))
    assert fatal(dict(row,budget_refusal=None))  # unknown 429 != known budget stop


def grid():
    plan=dict(hash='p',cases=[dict(case_id='one'),dict(case_id='two')],repeats=2)
    records=[dict(plan_hash='p',case_id=c,repeat_id=r,arm=a,status='passed'if a=='C'else'failed',
       execution_state='finished',model_calls=dict(reserved=2),wall_seconds=3,
       development_audit=dict(plugin_verification_observed=False))
       for c in ('one','two')for r in (1,2)for a in 'ABC']
    return plan,records


def test_paired_task_ci_and_all_assigned_costs():
    p,r=grid();s=stats.aggregate(p,r,bootstrap_samples=100)
    assert s['comparisons']['C-minus-B']['success_rate_difference']==1
    assert s['comparisons']['C-minus-B']['task_cluster_percentile_95']==[1,1]
    assert len(s['comparisons']['C-minus-B']['paired_task_differences'])==2
    assert s['groups']['B']['costs']['calls']['observed_sum']==8
    assert s['groups']['C']['mcp_used']==0  # not excluded
    assert stats.aggregate(p,r,bootstrap_samples=100)==s


def test_missing_unknown_and_duplicates_never_create_positive_inference():
    p,r=grid();s=stats.aggregate(p,r[:-1],bootstrap_samples=100)
    assert s['planned']==12 and s['received']==11
    assert not s['comparisons']['C-minus-B']['inference_allowed']
    assert s['groups']['C']['costs']['calls']['unknown_rows']==1
    with pytest.raises(ValueError):stats.aggregate(p,r+[r[0]],bootstrap_samples=100)
    bad=copy.deepcopy(r);bad[0]['plan_hash']='wrong'
    with pytest.raises(ValueError):stats.aggregate(p,bad,bootstrap_samples=100)


def test_repeated_runs_have_separate_evidence_and_budgets(tmp_path,monkeypatch):
    from benchmarks.experiments.s5 import s5_verify_remote, s5_arm_env
    from benchmarks.experiments.s5.s5_vertical_validation import GOOD_FIXTURE
    decl=dict(request='layernorm fp32',shapes=[[7,33]],hardware='mock_gpu',dsl='python',precision='fp32')
    sha=s5_verify_remote.freeze(tmp_path/'services/case',declaration=decl,backend='mock')
    p=dict(schema=flow.SCHEMA,implementation=flow.runtime_identity(),hash='repeat-test',repeats=2,
       budget=dict(max_calls=1,max_output_tokens=2048,total_output_tokens=2048,wall_clock_s=120),
       model=dict(endpoint='http://127.0.0.1:9/v1/responses',id='fake',reasoning_effort='medium'),
       development=dict(max_development=2))
    monkeypatch.setattr(s5_arm_env,'start_headless',lambda *a,**kw:dict(returncode=0,stderr='',events=[dict(type='final',text=GOOD_FIXTURE)]))
    for r in (1,2):
        row=flow.run_one(p,dict(case_id='case',service_sha256=sha,text='fixed'), 'A',tmp_path,'fake',repeat_id=r)
        assert row['status']=='passed' and row['repeat_id']==r
        assert(tmp_path/('runs/case/repeat-%02d/A/budget.json'%r)).is_file()
    with pytest.raises(FileExistsError):
        flow.run_one(p,dict(case_id='case',service_sha256=sha,text='fixed'),'A',tmp_path,'fake',repeat_id=1)
