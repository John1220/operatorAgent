"""Deterministic production search fixtures -- NOT measured GPU performance."""
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operator_agent import Conductor, KernelDatabase
from operator_agent.agents import SearchAgent
from operator_agent.bandit import Bandit
from operator_agent.budget import BudgetPlan
from operator_agent.contract import freeze_from_task
from operator_agent.kernel_db import FailureDatabase
from operator_agent import run_contract as rc
from operator_agent.schemas import UnifiedSketch, TilingStrategy
from operator_agent.search_objective import SearchObjective, reading, validate_snapshot
from tests.test_workload_evidence import A, B, CODE, backend, inputs, measure, task


def codes(n):
    return [CODE + "# candidate-" + str(i) + "\n" for i in range(n)]


def execute(tmp_path, ratios=((2.2, 1.0), (2.0, 1.8)), *, change=None,
            t=None, max_rounds=None, budget=None, resume_from=None, fail_compile=False):
    sources = codes(len(ratios))
    b, calls = backend()
    original = b.benchmark_with_baselines
    dispatch = []
    def bench(code, x, *args, **kwargs):
        i = sources.index(code)
        n = x["A"].shape[0]
        p = original(code, x, *args, **kwargs)
        p["baselines"]["pytorch_eager"]["median_time_ms"] = .2 * ratios[i][0 if n == 16 else 1]
        # Deliberately misleading primary GFLOPS, independent of objective.
        p["gflops"] = 1000. * (len(sources) - i)
        if change: change(p, i, n)
        dispatch.append((i, n))
        return p
    b.benchmark_with_baselines = bench
    if fail_compile:
        compile_original=b.compile
        def compile_failing(code):
            if code==sources[-1]:raise RuntimeError('fixture compiler rejected last candidate')
            return compile_original(code)
        b.compile=compile_failing
    c = Conductor(backend=b, kernel_db=KernelDatabase(str(tmp_path/'corpus.jsonl'), persist_retrievals=False),
        failure_db=FailureDatabase(str(tmp_path/'failures.jsonl')), use_coder_pipeline=False,
        max_optimize_rounds=max_rounds or len(sources), budget=budget)
    c.store.output_dir = tmp_path/'artifacts'
    c.designer.run = lambda *a, **k: UnifiedSketch(algorithm="matmul", target_dsl="python")
    generated = []
    def generate(*a, **k):
        code = sources[len(generated)]
        generated.append(code)
        return code
    c.coder = SimpleNamespace(run=generate)
    beliefs = []
    def search(sketch, perf):
        beliefs.append(copy.deepcopy(perf))
        return UnifiedSketch(algorithm="matmul", target_dsl="python", version=sketch.version+1,
                             tiling=TilingStrategy(128,64,32)), "fixed local proposal"
    c.search = SimpleNamespace(run=search)
    with contextlib.redirect_stdout(io.StringIO()):
        art = c.execute(t or task(min_speedup=10.), [inputs(16),inputs(32)], target_gflops=0.,
                        resume_from=resume_from)
    return art, c, beliefs, dispatch, generated


def test_lower_primary_and_gflops_but_better_workload_wins(tmp_path):
    art, c, beliefs, calls, generated = execute(tmp_path)
    assert calls == [(0,16),(0,32),(1,16),(1,32)]
    assert art.code == generated[1] and art.benchmark_data['speedup_vs_baseline'] == 2.0
    assert art.metadata['search_objective']['best']['score'] == 1.9
    first, second = [r['search_objective'] for r in art.iteration_history]
    assert first['score'] == 1.6 and first['reward'] is None
    assert second['score'] == 1.9 and second['comparison'] == 'FAST'
    assert second['reward'] == pytest.approx((1.9-1.6)/1.6)
    assert art.iteration_history[1]['permanent_effects']['best_updated']
    assert not art.iteration_history[1]['permanent_effects']['stagnation_counted']
    assert c._verdict_tail == ['FAST']
    assert art.metadata['bandit']['total_scored'] == 1
    assert beliefs[0]['speedup'] == 1.6 and beliefs[0]['performance_scope']=='declared_workload'
    assert not art.all_passed and not (tmp_path/'corpus.jsonl').exists()


def test_worse_last_candidate_does_not_replace_best_artifact_or_checkpoint(tmp_path):
    art, c, _, _, generated = execute(tmp_path, ratios=((2.,2.), (3.,.95)))
    assert art.code == generated[0]
    assert art.metadata['candidate_selection']['selected_round']==1
    assert art.metadata['candidate_selection']['fallback_to_best']
    assert art.benchmark_data['performance_verdict']['value']==2.
    assert art.iteration_history[-1]['search_objective']['comparison']=='SLOW'
    snap=json.loads((Path(art.saved_dir)/'execution_snapshot.json').read_text())
    assert snap['current_code']==generated[1]  # Last probe remains resumable audit data.
    assert snap['search_objective']['best']['code']==generated[0]
    assert snap['best_gflops']==0.  # Never put a workload score into a GFLOPS slot.
    validate_snapshot(snap['search_objective'],freeze_from_task(task(min_speedup=10.)),2)


@pytest.mark.parametrize('bad', ('floor','noise','regression','missing','contract','reused_window'))
def test_secondary_bad_evidence_cannot_win_or_count_progress(tmp_path,bad):
    def change(p,i,n):
        if bad=='reused_window' and i==1:
            w='fixture-window-'+('1' if n==16 else '2')
            p['paired']['arm_windows']=dict(candidate=w,baseline=w)
            return
        if i!=1 or n!=32:return
        if bad=='floor':p['candidate']['median_time_ms']=.001
        elif bad=='noise':p['candidate']['dispersion_iqr_over_median']=.2
        elif bad=='regression':p['baselines']['pytorch_eager']['median_time_ms']=.1
        elif bad=='missing':p.update(success=False,error='fixture missing observation')
        else:p['contract_hash']='different'
    art,c,_,_,generated=execute(tmp_path,ratios=((2.,2.),(5.,5.)),change=change)
    assert art.code==generated[0]
    point=art.iteration_history[-1]['search_objective']
    assert not point['can_select'] and not point['can_count_progress']
    assert point['reward'] is None
    assert not art.iteration_history[-1]['permanent_effects']['best_updated']
    assert not art.iteration_history[-1]['permanent_effects']['stagnation_counted']
    assert c._verdict_tail==[]
    assert art.metadata['bandit']['total_scored']==0
    assert not (tmp_path/'corpus.jsonl').exists()


def test_degraded_can_update_soft_belief_but_not_winner_or_stagnation(tmp_path):
    def change(p,i,n):
        if i==1 and n==32:p['candidate']['dispersion_iqr_over_median']=.2
    art,c,beliefs,_,_=execute(tmp_path,ratios=((1.5,1.5),(2.,2.),(1.5,1.5)),change=change)
    p=art.iteration_history[1]['search_objective']
    assert p['can_tune'] and not p['can_select'] and not p['can_count_progress']
    assert beliefs[1]['speedup']==2.
    assert art.metadata['candidate_selection']['selected_round']==1
    assert not art.iteration_history[1]['permanent_effects']['stagnation_counted']


def test_budget_exhaustion_returns_best_without_granting_target_pass(tmp_path):
    art,c,_,calls,generated=execute(tmp_path,ratios=((2.,2.),(3.,1.),(5.,5.)),
                                  max_rounds=6,budget=BudgetPlan(oracle_total=3))
    assert len(calls)==4 and len(generated)==2
    assert art.metadata['stop']['reason']=='budget_exhausted'
    assert art.code==generated[0] and not art.all_passed and art.remeasure is None
    assert art.metadata['budget']['spent']['confirm']==0
    assert not (tmp_path/'corpus.jsonl').exists()


def test_actual_workload_pass_records_winner_then_independent_confirmation(tmp_path):
    art,c,_,calls,_=execute(tmp_path,ratios=((2.,2.),),t=task())
    assert art.all_passed and calls==[(0,16),(0,32),(0,16),(0,32)]
    assert art.metadata['search_objective']['best']['score']==2.
    assert art.metadata['corpus_admission']['kept']
    assert art.metadata['bandit']['total_scored']==0  # First sample is baseline, not a 0 reward.
    assert sum(a['pulls'] for a in art.metadata['bandit']['arms'].values())==1


def test_resume_restores_source_not_historical_score_or_qualification(tmp_path):
    first,c,_,_,_=execute(tmp_path/'old',ratios=((2.,2.),))
    art,c,_,calls,generated=execute(tmp_path/'new',ratios=((.95,.95),),max_rounds=2,resume_from=first.saved_dir)
    assert not generated and calls==[(0,16),(0,32)]
    assert art.resume_status['status']=='resumed'
    assert art.metadata['resume_audit']['search_objective']['best']['score']==2.
    fresh=art.iteration_history[0]['search_objective']
    assert fresh['best_before'] is None and fresh['reward'] is None and fresh['score']==.95
    assert art.metadata['search_objective']['best']['score']==.95
    assert not art.all_passed and not (tmp_path/'new'/'corpus.jsonl').exists()


@pytest.mark.parametrize('mutation',('identity','score','code','qualifications','required_scope','bindings'))
def test_corrupt_search_checkpoint_fails_before_transport_or_generation(tmp_path,mutation):
    art,c,_,_,_=execute(tmp_path,ratios=((2.,2.),))
    path=Path(art.saved_dir)/'execution_snapshot.json';s=json.loads(path.read_text())
    point=s['search_objective'];best=point['best']
    if mutation=='identity':point['identity']['case_weights'][A]=2.
    elif mutation=='score':best['score']=100.
    elif mutation=='code':best['code']+='\n# substituted'
    elif mutation=='bindings':best['verification_results'][-1]['metrics']['run_binding']['candidate_hash']='bad'
    else:
        m=best['verification_results'][-1]['metrics']
        if mutation=='qualifications':m['cases'][B]['eligibility']['can_count_progress']=False
        else:m['workload_qualification']['scope_cases']=[A]
    # Even a resealed envelope cannot change the criterion or swap receipts.
    rc.seal(s,c._run_binding);rc.atomic_json(path,s)
    next_art,_,_,calls,generated=execute(tmp_path/'resume',max_rounds=2,resume_from=art.saved_dir)
    assert next_art.resume_status['status']=='identity_mismatch'
    assert not calls and not generated


def test_primary_roofline_ceiling_cannot_stop_workload_search(monkeypatch):
    r,_=measure(t=task(min_speedup=10.)); obj=SearchObjective.for_task(task(min_speedup=10.))
    seen=[]
    def ceiling(m):
        seen.append(m)
        return m is r.metrics['cases'][A]
    monkeypatch.setattr('operator_agent.search_objective.reference_ceiling_reached',ceiling)
    assert not obj.at_reference_ceiling(r.metrics)
    assert len(seen)==2


@pytest.mark.parametrize('mutation',('nan','inf','bool','value','criterion','oracle','top_mask'))
def test_malformed_objective_never_falls_back_to_high_primary(mutation):
    r,_=measure(t=task(min_speedup=10.));m=r.metrics
    if mutation in ('nan','inf','bool','value'):
        m['performance_verdict']['value']={'nan':float('nan'),'inf':float('inf'),'bool':True,'value':99.}[mutation]
    elif mutation=='criterion':m['performance_verdict']['case_weights'][A]=3.
    elif mutation=='oracle':m['cases'][B]['observation']['ruler_broken']=True
    else:m['eligibility']['can_count_progress']=False
    p=reading(m,freeze_from_task(task(min_speedup=10.)))
    assert p['score'] is None and not p['can_select'] and p['reasons']


@pytest.mark.parametrize('bad',(float('nan'),float('inf'),-float('inf')))
def test_bandit_rejects_nonfinite_reward_and_cost(bad):
    b=Bandit();b.update('a',bad,eligible=True,cost_s=bad)
    assert b.total_scored==0 and b.pulls('a')==1 and b.mean_cost_all() is None


def test_search_prompt_labels_objective_and_primary_diagnostics(tmp_path):
    art,c,beliefs,_,_=execute(tmp_path)
    prompts=[]
    llm=SimpleNamespace(chat_json=lambda messages,**k: prompts.append(messages[0]['content']) or {})
    SearchAgent(llm=llm)._llm_optimize(UnifiedSketch(algorithm='matmul',target_dsl='python'),beliefs[0])
    assert '冻结的 workload 搜索目标' in prompts[0]
    assert '不是候选排名或整体收敛依据' in prompts[0]
    assert '"objective": 1.6' in prompts[0] and A in prompts[0] and B in prompts[0]


def test_primary_improvement_cannot_hide_workload_slowdown(tmp_path):
    art,c,_,_,_=execute(tmp_path,ratios=((2.,2.),(3.,.91)))
    p=art.iteration_history[1]['search_objective']
    assert p['score']==pytest.approx(1.955) and p['reward']<0
    assert p['comparison'] in ('SLOW','SAME')
    assert art.metadata['candidate_selection']['selected_round']==1
    assert art.iteration_history[1]['permanent_effects']['stagnation_counted']


def test_legacy_search_is_explicitly_unchanged():
    assert SearchObjective.for_task(task(case_weights={},aggregate='',required_cases=(),max_regression={})) is None


def test_degraded_last_pass_cannot_displace_qualified_below_target_winner(tmp_path):
    def change(p,i,n):
        if i==1 and n==32:p['candidate']['dispersion_iqr_over_median']=.2
    art,c,_,calls,generated=execute(tmp_path,ratios=((1.5,1.5),(2.,2.)),change=change,t=task(min_speedup=1.8))
    assert art.code==generated[0] and not art.all_passed and art.remeasure is None
    assert art.metadata['stop']['reason']=='qualified_candidate_hold'
    assert art.metadata['last_probe_stop']['reason']=='passed'
    assert len(calls)==4 and not (tmp_path/'corpus.jsonl').exists()


@pytest.mark.parametrize('bad',('passed','missing_window','cross_window','same_case_windows'))
def test_workload_attestation_and_window_integrity_are_required(bad):
    r,_=measure(t=task(min_speedup=10.));m=r.metrics
    if bad=='passed':m['performance_verdict']['passed']=True
    elif bad=='missing_window':m['cases'][B].pop('arm_windows')
    elif bad=='cross_window':m['cases'][B]['arm_windows']['baseline']='other'
    else:m['cases'][B]['arm_windows']=m['cases'][A]['arm_windows']
    point=reading(m,freeze_from_task(task(min_speedup=10.)))
    assert point['score'] is None and not point['can_select']


def test_last_compile_failure_returns_earlier_correct_best_with_its_own_results(tmp_path):
    art,c,_,calls,generated=execute(tmp_path,fail_compile=True)
    assert len(calls)==2 and len(generated)==2
    assert art.metadata['stop']['reason']=='l1_l2_failed'
    assert art.iteration_history[-1]['levels_passed']['L1'] is False
    assert art.code==generated[0] and not art.all_passed
    assert {r.level.value:r.passed for r in art.verification_results}==dict(L1=True,L2=True,L3=False)
    assert all(r.metrics['run_binding']['candidate_hash']==rc.code_hash(art.code) for r in art.verification_results)
    assert art.compile_info['run_binding']['candidate_hash']==rc.code_hash(art.code)
    assert not (tmp_path/'corpus.jsonl').exists()


def test_no_qualified_candidate_does_not_invent_a_winner_or_zero_reward(tmp_path):
    def change(p,i,n):p['candidate']['median_time_ms']=.001
    art,c,beliefs,_,_=execute(tmp_path,change=change)
    assert art.metadata['search_objective']['best'] is None
    assert art.metadata['candidate_selection']['selected_round'] is None
    assert art.metadata['bandit']['total_scored']==0
    assert all(r['search_objective']['reward'] is None for r in art.iteration_history)
    assert beliefs[0]['tuning_refused']=='can_tune=False' and 'speedup' not in beliefs[0]


def test_checkpoint_cannot_replace_numeric_one_with_boolean_true(tmp_path):
    art,c,_,_,_=execute(tmp_path,ratios=((1.,1.),))
    snapshot=json.loads((Path(art.saved_dir)/'execution_snapshot.json').read_text())['search_objective']
    snapshot['best']['score']=True
    with pytest.raises(ValueError,match='identity invalid'):
        validate_snapshot(snapshot,freeze_from_task(task(min_speedup=10.)),1)


@pytest.mark.parametrize('malformed', ('null_metrics', 'list_metrics', 'integer_passed', 'duplicate', 'not_list'))
def test_malformed_best_verification_holds_before_any_dispatch(tmp_path, malformed):
    art, c, _, _, _ = execute(tmp_path, ratios=((2., 2.),))
    path = Path(art.saved_dir)/'execution_snapshot.json';snap = json.loads(path.read_text())
    best = snap['search_objective']['best'];rows = best['verification_results']
    if malformed == 'null_metrics':rows[0]['metrics'] = None
    elif malformed == 'list_metrics':rows[0]['metrics'] = []
    elif malformed == 'integer_passed':rows[0]['passed'] = 1
    elif malformed == 'duplicate':rows[2] = copy.deepcopy(rows[0])
    else:best['verification_results'] = {'L1': True}
    rc.seal(snap, c._run_binding);rc.atomic_json(path, snap)
    held, _, _, calls, generated = execute(tmp_path/'resume', max_rounds=2, resume_from=art.saved_dir)
    assert held.resume_status['status'] == 'identity_mismatch'
    assert not calls and not generated
