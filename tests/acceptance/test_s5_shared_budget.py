"""Concurrency, fail-closed and accounting behavior of the shared S5 ledger."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from operator_agent.execute import s5_shared_budget as budget

ROOT=Path(__file__).resolve().parents[2]
REQ='a'*64


def make(tmp_path, **overrides):
    values=dict(task_id='s5-demo-01',max_calls=4,max_output_tokens=100,
                total_output_tokens=400,wall_clock_s=60.)
    values.update(overrides); path=tmp_path/'ledger.sqlite'; budget.create(path,**values); return path


def reserve(path, participant='outer', caller='loop', tokens=50):
    return budget.reserve(path,participant=participant,caller=caller,
                          max_output_tokens=tokens,request_sha256=REQ)


def command(*args):
    return subprocess.run([sys.executable,'-m','operator_agent.execute.s5_shared_budget',*map(str,args)],
                          cwd=ROOT,text=True,capture_output=True,timeout=30)


def test_concurrent_outer_and_inner_processes_share_one_exact_cap(tmp_path):
    path=make(tmp_path,max_calls=5,max_output_tokens=40,total_output_tokens=200)
    def attempt(i):
        r=command('reserve','--ledger',path,'--participant','outer' if i%2 else 'inner',
            '--caller','loop','--max-output-tokens',40,'--request-sha256',REQ)
        return r.returncode,json.loads(r.stdout)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results=list(pool.map(attempt,range(20)))
    success=[r for code,r in results if code==0]
    assert len(success)==5
    assert len({r['call_id'] for r in success})==5
    assert sorted(r['sequence'] for r in success)==[1,2,3,4,5]
    assert all(r.get('error')=='call_budget_exhausted' for code,r in results if code!=0)
    s=budget.snapshot(path)
    assert s['reserved_calls']==5 and s['remaining_calls']==0 and s['reserved_output_tokens']==200
    assert s['usage_unknown_calls']==5 and s['network_interception_proven'] is False


def test_dispatched_failures_and_worker_death_do_not_refund(tmp_path):
    path=make(tmp_path,max_calls=2)
    a=reserve(path); budget.dispatch(path,call_id=a['call_id'])
    budget.finish(path,call_id=a['call_id'],outcome='failed',usage_evidence='missing')
    # A different process reserves, then exits before dispatch; it remains spent.
    p=command('reserve','--ledger',path,'--participant','inner','--caller','Coder',
              '--max-output-tokens',50,'--request-sha256',REQ)
    assert p.returncode==0
    with pytest.raises(budget.BudgetError,match='call_budget_exhausted'): reserve(path)
    s=budget.snapshot(path)
    assert s['reserved_calls']==2 and s['completed_calls']==1
    assert s['calls'][0]['outcome']=='failed' and s['calls'][1]['dispatched_monotonic'] is None
    assert s['usage_unknown_calls']==2


def test_single_dispatch_and_finish_are_not_replayable(tmp_path):
    path=make(tmp_path); a=reserve(path); cid=a['call_id']
    with pytest.raises(budget.BudgetError,match='dispatch_not_recorded'):
        budget.finish(path,call_id=cid,outcome='failed')
    budget.dispatch(path,call_id=cid)
    with pytest.raises(budget.BudgetError,match='dispatch_already_recorded'): budget.dispatch(path,call_id=cid)
    budget.finish(path,call_id=cid,outcome='answered',usage={'prompt_tokens':20,'completion_tokens':7,'total_tokens':27},usage_evidence='provider')
    with pytest.raises(budget.BudgetError,match='finish_already_recorded'): budget.finish(path,call_id=cid,outcome='failed')
    row=budget.snapshot(path)['calls'][0]
    assert row['api_usage']['total_tokens']==27 and row['usage_evidence']=='provider'


def test_unverified_zero_is_not_promoted_to_provider_usage(tmp_path):
    path=make(tmp_path); cid=reserve(path)['call_id']; budget.dispatch(path,call_id=cid)
    budget.finish(path,call_id=cid,outcome='unknown',usage={'prompt_tokens':0,'completion_tokens':0},usage_evidence='unverified')
    s=budget.snapshot(path); assert s['usage_unknown_calls']==1 and s['calls'][0]['api_usage'] is None


@pytest.mark.parametrize('usage',[
    {'prompt_tokens':True,'completion_tokens':3},
    {'prompt_tokens':10,'completion_tokens':51},
    {'prompt_tokens':10,'completion_tokens':3,'total_tokens':0},
    {'prompt_tokens':10,'completion_tokens':3,'cache_read_tokens':8,'cache_write_tokens':4},
    {'prompt_tokens':10,'completion_tokens':3,'provider_extension':'text'},
])
def test_invalid_provider_usage_does_not_get_recorded(tmp_path,usage):
    path=make(tmp_path); cid=reserve(path)['call_id']; budget.dispatch(path,call_id=cid)
    with pytest.raises(budget.BudgetError,match='invalid_provider_usage'):
        budget.finish(path,call_id=cid,outcome='answered',usage=usage,usage_evidence='provider')
    assert budget.snapshot(path)['calls'][0]['outcome'] is None


def test_output_reservations_are_separate_from_call_count(tmp_path):
    path=make(tmp_path,max_calls=4,max_output_tokens=100,total_output_tokens=70)
    reserve(path,tokens=45)
    with pytest.raises(budget.BudgetError,match='output_budget_exhausted'): reserve(path,tokens=26)
    reserve(path,participant='inner',tokens=25)
    with pytest.raises(budget.BudgetError,match='per_call_output_exhausted'): reserve(path,tokens=101)
    assert budget.snapshot(path)['reserved_output_tokens']==70


def test_clock_jump_or_mismatched_boot_cannot_extend_allowance(tmp_path,monkeypatch):
    path=make(tmp_path)
    original=budget.time.time
    monkeypatch.setattr(budget.time,'time',lambda:original()+10)
    with pytest.raises(budget.BudgetError,match='clock_identity_changed'): reserve(path)


def test_append_only_and_no_recreate(tmp_path):
    path=make(tmp_path); reserve(path)
    with pytest.raises(budget.BudgetError,match='ledger_exists'):
        budget.create(path,task_id='s5-demo-01',max_calls=4,max_output_tokens=100,total_output_tokens=400,wall_clock_s=60)
    with sqlite3.connect(path) as conn:
        with pytest.raises(sqlite3.IntegrityError): conn.execute('DELETE FROM reservations')
        with pytest.raises(sqlite3.IntegrityError): conn.execute('UPDATE settings SET value=?',('{}',))
    assert budget.snapshot(path)['reserved_calls']==1


def test_cli_never_emits_request_or_credentials(tmp_path):
    path=make(tmp_path)
    p=command('reserve','--ledger',path,'--participant','outer','--caller','loop',
              '--max-output-tokens','10','--request-sha256',REQ)
    d=json.loads(p.stdout); assert p.returncode==0 and d['ok'] and set(d)=={'ok','call_id','sequence'}
    p=command('dispatch','--ledger',path,'--call-id',d['call_id']); assert p.returncode==0
    p=command('finish','--ledger',path,'--call-id',d['call_id'],'--outcome','answered',
              '--usage-json','{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}',
              '--usage-evidence','provider'); assert p.returncode==0
    assert 'prompt_tokens' not in p.stdout


def test_abort_is_durable_irreversible_and_preserves_completion(tmp_path):
    path=make(tmp_path)
    running=reserve(path); ready=reserve(path)
    budget.dispatch(path,call_id=running['call_id'])
    assert budget.abort(path,reason='audit_failure')['already_aborted'] is False
    again=budget.abort(path,reason='manual_stop')
    assert again['already_aborted'] is True and again['reason']=='audit_failure'
    with pytest.raises(budget.BudgetError,match='campaign_aborted'): reserve(path)
    with pytest.raises(budget.BudgetError,match='campaign_aborted'): budget.dispatch(path,call_id=ready['call_id'])
    budget.finish(path,call_id=running['call_id'],outcome='failed',usage_evidence='missing')
    s=budget.snapshot(path); assert s['aborted'] is True and s['abort_reason']=='audit_failure'
    assert s['calls'][0]['outcome']=='failed' and s['calls'][1]['dispatched_monotonic'] is None
    with sqlite3.connect(path) as conn:
        with pytest.raises(sqlite3.IntegrityError): conn.execute('DELETE FROM aborts')
        with pytest.raises(sqlite3.IntegrityError): conn.execute('UPDATE aborts SET reason=?',('manual_stop',))


def test_concurrent_abort_orders_dispatches_across_processes(tmp_path):
    path=make(tmp_path,max_calls=20,max_output_tokens=40,total_output_tokens=800)
    calls=[reserve(path,participant='inner',tokens=40)['call_id'] for _ in range(20)]
    def operation(i):
        if i==7:
            p=command('abort','--ledger',path,'--reason','audit_failure')
        else:
            p=command('dispatch','--ledger',path,'--call-id',calls[i])
        return p.returncode,json.loads(p.stdout)
    with ThreadPoolExecutor(max_workers=15) as pool:
        results=list(pool.map(operation,range(20)))
    assert results[7][0]==0
    for i,(code,value) in enumerate(results):
        if i!=7 and code!=0: assert value['error']=='campaign_aborted'
    s=budget.snapshot(path); assert s['aborted'] and s['abort_reason']=='audit_failure'
    assert all(c['dispatched_monotonic']<=s['aborted_monotonic'] for c in s['calls'] if c['dispatched_monotonic'] is not None)
    with pytest.raises(budget.BudgetError,match='campaign_aborted'): reserve(path)
    active=next((c for c in s['calls'] if c['dispatched_monotonic'] is not None),None)
    if active: budget.finish(path,call_id=active['call_id'],outcome='cancelled')
    assert budget.snapshot(path)['completed_calls']==int(active is not None)


def test_abort_cli_is_cross_process_and_nonleaking(tmp_path):
    path=make(tmp_path)
    p=command('abort','--ledger',path,'--reason','audit_failure')
    assert p.returncode==0 and json.loads(p.stdout)['aborted'] is True
    p=command('reserve','--ledger',path,'--participant','outer','--caller','loop',
              '--max-output-tokens','10','--request-sha256',REQ)
    assert p.returncode==2 and json.loads(p.stdout)['error']=='campaign_aborted'


def test_wrong_expected_task_never_changes_another_ledger(tmp_path):
    path=make(tmp_path)
    cid=reserve(path)['call_id']
    for operation in (
        lambda: budget.snapshot(path,expected_task_id='other'),
        lambda: budget.reserve(path,participant='inner',caller='loop',max_output_tokens=10,
                               request_sha256=REQ,expected_task_id='other'),
        lambda: budget.dispatch(path,call_id=cid,expected_task_id='other'),
        lambda: budget.abort(path,reason='audit_failure',expected_task_id='other'),
    ):
        with pytest.raises(budget.BudgetError,match='task_identity_mismatch'): operation()
    s=budget.snapshot(path)
    assert not s['aborted'] and s['reserved_calls']==1 and s['calls'][0]['dispatched_monotonic'] is None
    p=command('abort','--ledger',path,'--reason','manual_stop','--expected-task-id','other')
    assert p.returncode==2 and json.loads(p.stdout)['error']=='task_identity_mismatch'
