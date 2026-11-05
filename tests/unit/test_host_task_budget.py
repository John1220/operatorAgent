"""Host verification credit persists across calls, failures and processes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from operator_agent.app import hitl_tools as ht
from operator_agent.core.run_contract import code_hash
from operator_agent.execute import host_task_budget as hb


CORRECT = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'
WRONG = 'import numpy as np\ndef kernel(X):\n    return X.copy()\n'


def prepared(tmp_path, limits=None):
    run = tmp_path/'task'
    value = ht.prepare(dict(request='relu fp32', shapes=[[4, 8]], precision='fp32'), run)
    hb.initialize_host_task(run,value['contract_hash'],limits)
    return run,hb.HostTaskBudget(run,value['contract_hash'])


def checked(budget, code=CORRECT, purpose='explore'):
    reservation=budget.reserve(code_hash(code),purpose=purpose)
    result=verification(budget,code)
    return budget.finish(reservation['id'],result=result),result


def verification(budget,code):
    # A legacy child run creates authentic oracle evidence without invoking
    # host-workflow integration (which independently reserves its own calls).
    run=budget.run/'verification-fixture'
    if not run.exists():
        ht.prepare(dict(request='relu fp32',shapes=[[4,8]],precision='fp32'),run)
    return ht.verify(run,code,backend='fake',stop_after='L2',candidate_origin='outer')


def test_same_policy_cannot_recreate_credit_or_change_contract(tmp_path):
    run,budget=prepared(tmp_path)
    reserved=budget.reserve(code_hash(CORRECT))
    first=budget.snapshot()
    again=hb.initialize_host_task(run,budget.contract_hash)
    assert again['task_id']==first['task_id'] and again['reserved']==1
    assert again['model_budget']['enforcement']=='not_enforced_by_plugin'
    assert again['model_budget']['calls'] is None
    with pytest.raises(hb.HostTaskBudgetError,match='policy differs'):
        hb.initialize_host_task(run,budget.contract_hash,hb.HostTaskLimits(max_verifications=10))
    with pytest.raises(hb.HostTaskBudgetError):
        hb.initialize_host_task(run,'f'*64)
    assert budget.snapshot()['records'][0]['id']==reserved['id']


def test_exploration_protects_final_and_failed_calls_never_refund(tmp_path):
    run,budget=prepared(tmp_path,hb.HostTaskLimits(max_verifications=3))
    failed=budget.reserve(code_hash(WRONG))
    budget.finish(failed['id'],outcome='failed')
    snap,result=checked(budget)
    assert snap['reserved']==2 and snap['remaining_exploration']==0
    assert snap['remaining_final']==1 and snap['last_correct_candidate']['source_hash']==code_hash(CORRECT)
    assert snap['last_correct_candidate']['scope']=='latest_complete_l2_only_not_performance_best'
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.reserve(code_hash(CORRECT))
    assert error.value.code=='host_verification_budget'
    final,result=checked(budget,purpose='final')
    assert final['reserved']==3 and final['remaining_final']==0 and final['unfinished']==0
    assert final['records'][0]['status']=='failed'
    with pytest.raises(hb.HostTaskBudgetError):
        budget.reserve(code_hash(CORRECT),purpose='final')


def test_final_requires_prior_same_source_and_new_independent_evidence(tmp_path):
    _,budget=prepared(tmp_path)
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.reserve(code_hash(CORRECT),purpose='final')
    assert error.value.code=='host_final_source'
    _,first=checked(budget)
    reservation=budget.reserve(code_hash(CORRECT),purpose='final')
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.finish(reservation['id'],result=first)
    assert error.value.code=='host_task_evidence'
    final=verification(budget,CORRECT)
    completed=budget.finish(reservation['id'],result=final)
    assert completed['records'][-1]['evidence']!=completed['records'][0]['evidence']
    assert completed['records'][-1]['full_l2_passed'] is True


def test_cancelled_and_unfinished_reservations_remain_spent(tmp_path):
    run,budget=prepared(tmp_path)
    first=budget.reserve(code_hash(CORRECT))
    budget.finish(first['id'],outcome='cancelled')
    second=budget.reserve(code_hash(WRONG))
    reopened=hb.HostTaskBudget(run,budget.contract_hash).snapshot()
    assert reopened['reserved']==2 and reopened['unfinished']==1
    assert reopened['records'][0]['status']=='cancelled'
    assert reopened['records'][1]['status']=='reserved'
    assert reopened['records'][1]['full_l2_passed'] is None
    assert second['id']==2 and reopened['last_correct_candidate'] is None


def test_incomplete_l2_or_empty_suite_does_not_grant_final_credit(tmp_path):
    _,budget=prepared(tmp_path)
    reservation=budget.reserve(code_hash(CORRECT))
    result=verification(budget,CORRECT)
    path=Path(result['evidence'])
    full=json.loads(path.read_text())
    l2=next(v for v in full['verdicts'] if v['level']=='L2')
    l2['metrics']['correctness']['cases']=[]
    path.chmod(0o600)
    path.write_text(json.dumps(full))
    receipt=path.parent/'complete.json'
    complete=json.loads(receipt.read_text())
    complete['files_sha256']['result.json']=hashlib.sha256(path.read_bytes()).hexdigest()
    receipt.chmod(0o600)
    receipt.write_text(json.dumps(complete))
    snap=budget.finish(reservation['id'],result=result)
    assert snap['records'][0]['full_l2_passed'] is False and snap['last_correct_candidate'] is None
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.reserve(code_hash(CORRECT),purpose='final')
    assert error.value.code=='host_final_source'


def test_changed_completed_evidence_is_refused_without_refund(tmp_path):
    _,budget=prepared(tmp_path)
    reservation=budget.reserve(code_hash(CORRECT))
    result=verification(budget,CORRECT)
    path=Path(result['evidence'])
    path.chmod(0o600)
    path.write_text(path.read_text()+' ')
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.finish(reservation['id'],result=result)
    assert error.value.code=='host_task_evidence'
    assert budget.snapshot()['reserved']==1 and budget.snapshot()['unfinished']==1


def test_idle_time_counts_and_next_dispatch_stops_at_deadline(tmp_path,monkeypatch):
    clock=[100.0]
    monkeypatch.setattr(hb,'monotonic',lambda:clock[0])
    run,budget=prepared(tmp_path,hb.HostTaskLimits(wall_clock_s=10))
    first=budget.snapshot()
    clock[0]+=11
    reopened=hb.HostTaskBudget(run,budget.contract_hash)
    snap=reopened.snapshot()
    assert snap['expired'] is True and snap['remaining_s']==0
    assert snap['deadline_epoch']==first['deadline_epoch']
    with pytest.raises(hb.HostTaskBudgetError) as error:
        reopened.reserve(code_hash(CORRECT))
    assert error.value.code=='host_task_deadline'
    assert reopened.snapshot()['reserved']==0


def test_partial_initialization_and_policy_drift_fail_closed(tmp_path):
    run,budget=prepared(tmp_path)
    (run/hb.STATE).unlink()
    assert hb.host_task_enabled(run) is True
    with pytest.raises(hb.HostTaskBudgetError):
        hb.initialize_host_task(run,budget.contract_hash)


@pytest.mark.skipif(os.name!='posix',reason='host workflow requires POSIX locking')
def test_competing_processes_cannot_spend_reserved_final_credit(tmp_path):
    run,budget=prepared(tmp_path,hb.HostTaskLimits(max_verifications=4))
    script='''
import sys
from operator_agent.execute.host_task_budget import HostTaskBudget,HostTaskBudgetError
try:
 HostTaskBudget(sys.argv[1],sys.argv[2]).reserve(sys.argv[3])
 print('reserved')
except HostTaskBudgetError as error:
 print(error.code)
'''
    processes=[subprocess.Popen([sys.executable,'-c',script,str(run),budget.contract_hash,code_hash(CORRECT)],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(8)]
    outputs=[]
    for process in processes:
        stdout,stderr=process.communicate(timeout=45)
        assert process.returncode==0,stderr
        outputs.append(stdout.strip())
    assert outputs.count('reserved')==3
    assert outputs.count('host_verification_budget')==5
    snap=budget.snapshot()
    assert snap['reserved']==3 and snap['remaining_exploration']==0 and snap['remaining_final']==1


def test_legacy_prepared_run_is_not_implicitly_enabled(tmp_path):
    run=tmp_path/'legacy'
    ht.prepare(dict(request='relu fp32',shapes=[[4,8]],precision='fp32'),run)
    assert hb.host_task_enabled(run) is False
    ht.verify(run,CORRECT,backend='fake',stop_after='L2')
    assert hb.host_task_enabled(run) is False


def test_parent_directory_alias_does_not_break_evidence_binding(tmp_path):
    actual=tmp_path/'actual'
    actual.mkdir()
    alias=tmp_path/'alias'
    alias.symlink_to(actual,target_is_directory=True)
    run=alias/'task'
    value=ht.prepare(dict(request='relu fp32',shapes=[[4,8]],precision='fp32'),run)
    hb.initialize_host_task(run,value['contract_hash'])
    budget=hb.HostTaskBudget(run,value['contract_hash'])
    snap,_=checked(budget)
    assert snap['last_correct_candidate']['source_hash']==code_hash(CORRECT)


def test_symlink_state_is_refused_instead_of_reset(tmp_path):
    run,budget=prepared(tmp_path)
    copy=run/'copied-state.json'
    copy.write_bytes((run/hb.STATE).read_bytes())
    (run/hb.STATE).unlink()
    (run/hb.STATE).symlink_to(copy)
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.snapshot()
    assert error.value.code=='host_task_journal'


def test_rewritten_credit_without_journal_checksum_is_refused(tmp_path):
    run,budget=prepared(tmp_path)
    budget.reserve(code_hash(CORRECT))
    path=run/hb.STATE
    value=json.loads(path.read_text())
    value['records']=[]
    path.write_text(json.dumps(value))
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.reserve(code_hash(CORRECT))
    assert error.value.code=='host_task_binding'


@pytest.mark.parametrize('outcome,phase',[
    ('completed','final_complete'),('failed','final_failed'),('cancelled','final_failed')])
def test_reserved_final_closes_all_future_development_without_refunding(tmp_path,outcome,phase):
    _,budget=prepared(tmp_path)
    first,_=checked(budget)
    assert first['phase']=='awaiting_host' and first['terminal'] is False
    final=budget.reserve(code_hash(CORRECT),purpose='final')
    running=budget.snapshot()
    assert running['phase']=='finalizing' and running['development_closed'] is True
    assert running['can_reserve_exploration'] is False and running['can_reserve_final'] is False
    assert running['terminal'] is False
    for purpose in ('explore','final'):
        with pytest.raises(hb.HostTaskBudgetError) as error:
            hb.HostTaskBudget(budget.run,budget.contract_hash).reserve(code_hash(CORRECT),purpose=purpose)
        assert error.value.code=='host_task_finalized'
    result=verification(budget,CORRECT) if outcome=='completed' else None
    snap=budget.finish(final['id'],outcome=outcome,result=result)
    assert snap['phase']==phase and snap['terminal'] is True
    assert snap['development_closed'] is True and snap['reserved']==2
    assert snap['remaining_exploration']==4 and snap['remaining_final']==0
    assert snap['final_verification']['status']==outcome
    assert snap['last_correct_candidate']['source_hash']==code_hash(CORRECT)
    for purpose in ('explore','final'):
        with pytest.raises(hb.HostTaskBudgetError) as error:
            budget.reserve(code_hash(CORRECT),purpose=purpose)
        assert error.value.code=='host_task_finalized'
    assert budget.snapshot()['reserved']==2


def test_final_cannot_overlap_in_flight_exploration(tmp_path):
    _,budget=prepared(tmp_path)
    checked(budget)
    active=budget.reserve(code_hash(WRONG))
    snap=budget.snapshot()
    assert snap['phase']=='verifying' and snap['can_reserve_final'] is False
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.reserve(code_hash(CORRECT),purpose='final')
    assert error.value.code=='host_final_in_flight'
    snap=budget.finish(active['id'],outcome='cancelled')
    assert snap['reserved']==2 and snap['remaining_final']==1
    final=budget.reserve(code_hash(CORRECT),purpose='final')
    assert final['id']==3 and budget.snapshot()['phase']=='finalizing'


def test_exhausted_phases_distinguish_reserved_final_from_no_correct_candidate(tmp_path):
    _,budget=prepared(tmp_path,hb.HostTaskLimits(max_verifications=2))
    snap,_=checked(budget)
    assert snap['phase']=='awaiting_final' and snap['terminal'] is False
    assert snap['can_reserve_final'] is True and snap['can_reserve_exploration'] is False
    other=tmp_path/'other'
    other.mkdir()
    _,wrong_budget=prepared(other,hb.HostTaskLimits(max_verifications=2))
    reservation=wrong_budget.reserve(code_hash(WRONG))
    snap=wrong_budget.finish(reservation['id'],outcome='failed')
    assert snap['phase']=='exhausted' and snap['terminal'] is True
    assert snap['stop_reason']=='exploration_exhausted_without_correct_candidate'


@pytest.mark.skipif(os.name!='posix',reason='host workflow requires POSIX locking')
def test_other_process_cannot_resume_exploration_after_final_reservation(tmp_path):
    run,budget=prepared(tmp_path)
    checked(budget)
    budget.reserve(code_hash(CORRECT),purpose='final')
    script='''
import sys
from operator_agent.execute.host_task_budget import HostTaskBudget,HostTaskBudgetError
try:
 HostTaskBudget(sys.argv[1],sys.argv[2]).reserve(sys.argv[3])
 print('unexpected dispatch')
except HostTaskBudgetError as error:
 print(error.code)
'''
    result=subprocess.run([sys.executable,'-c',script,str(run),budget.contract_hash,code_hash(CORRECT)],
        capture_output=True,text=True,timeout=30,check=True)
    assert result.stdout.strip()=='host_task_finalized'
    assert budget.snapshot()['reserved']==2


@pytest.mark.parametrize('invalid',['', 'a'*31, 'a'*33, 'A'*32, 'g'*32, 12, True])
def test_invalid_owner_identity_cannot_reserve_or_settle(tmp_path,invalid):
    _,budget=prepared(tmp_path)
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.reserve(code_hash(CORRECT),owner_invocation_id=invalid)
    assert error.value.code=='host_task_owner'
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.settle_owner(invalid)
    assert error.value.code=='host_task_owner'
    assert budget.snapshot()['reserved']==0


def test_settlement_only_touches_matching_unfinished_owner_and_never_refunds(tmp_path):
    _,budget=prepared(tmp_path,hb.HostTaskLimits(max_verifications=8))
    checked(budget)
    owner_a,owner_b='a'*32,'b'*32
    first=budget.reserve(code_hash(WRONG),owner_invocation_id=owner_a)
    other=budget.reserve(code_hash(WRONG),owner_invocation_id=owner_b)
    unbound=budget.reserve(code_hash(WRONG))
    second=budget.reserve(code_hash(CORRECT),owner_invocation_id=owner_a)
    before=budget.snapshot()
    settled=hb.HostTaskBudget(budget.run,budget.contract_hash).settle_owner(owner_a)
    assert settled['settled_reservation_ids']==[first['id'],second['id']]
    assert settled['reserved']==before['reserved']==5
    assert settled['remaining_verifications']==before['remaining_verifications']
    rows={r['id']:r for r in settled['records']}
    assert rows[first['id']]['status']==rows[second['id']]['status']=='cancelled'
    assert rows[other['id']]['status']==rows[unbound['id']]['status']=='reserved'
    assert settled['unfinished']==2
    assert budget.settle_owner(owner_a,outcome='failed')['settled_reservation_ids']==[]
    assert budget.snapshot()['records'][first['id']-1]['status']=='cancelled'
    budget.settle_owner(owner_b,outcome='failed')
    budget.finish(unbound['id'],outcome='failed')
    final=budget.reserve(code_hash(CORRECT),purpose='final',owner_invocation_id='c'*32)
    assert final['id']==6 and budget.snapshot()['phase']=='finalizing'


@pytest.mark.parametrize('outcome',['cancelled','failed'])
def test_reaped_final_owner_settles_final_without_reopening_development(tmp_path,outcome):
    _,budget=prepared(tmp_path)
    checked(budget)
    final=budget.reserve(code_hash(CORRECT),purpose='final',owner_invocation_id='a'*32)
    snap=budget.settle_owner('a'*32,outcome=outcome)
    assert snap['settled_reservation_ids']==[final['id']]
    assert snap['phase']=='final_failed' and snap['terminal'] is True
    assert snap['reserved']==2 and snap['remaining_final']==0
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.reserve(code_hash(CORRECT))
    assert error.value.code=='host_task_finalized'
    assert budget.settle_owner('a'*32)['settled_reservation_ids']==[]


def test_l2_completed_after_deadline_is_not_a_final_source(tmp_path,monkeypatch):
    _,budget=prepared(tmp_path,hb.HostTaskLimits(wall_clock_s=100))
    reservation=budget.reserve(code_hash(CORRECT))
    result=verification(budget,CORRECT)
    monkeypatch.setattr(budget,'_elapsed',lambda state:101.0)
    snap=budget.finish(reservation['id'],result=result)
    assert snap['records'][0]['full_l2_passed'] is True
    assert snap['records'][0]['finished_after_deadline'] is True
    assert snap['last_correct_candidate'] is None and snap['can_reserve_final'] is False
    assert snap['phase']=='exhausted'


@pytest.mark.skipif(os.name!='posix',reason='host workflow requires POSIX locking')
def test_parallel_owner_reservations_are_settled_independently(tmp_path):
    run,budget=prepared(tmp_path)
    owners=['a'*32,'b'*32]
    script='''
import sys
from operator_agent.execute.host_task_budget import HostTaskBudget
HostTaskBudget(sys.argv[1],sys.argv[2]).reserve(sys.argv[3],owner_invocation_id=sys.argv[4])
'''
    processes=[subprocess.Popen([sys.executable,'-c',script,str(run),budget.contract_hash,
        code_hash(CORRECT),owner],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for owner in owners]
    for process in processes:
        stdout,stderr=process.communicate(timeout=45)
        assert process.returncode==0,stderr
    snap=budget.settle_owner(owners[0])
    rows={r['owner_invocation_id']:r for r in snap['records']}
    assert rows[owners[0]]['status']=='cancelled' and rows[owners[1]]['status']=='reserved'
    assert snap['reserved']==2 and snap['unfinished']==1


def test_supervisor_cancel_race_revokes_only_rejected_owners_completed_credit(tmp_path):
    _,budget=prepared(tmp_path)
    owner_a,owner_b='a'*32,'b'*32
    first=budget.reserve(code_hash(CORRECT),owner_invocation_id=owner_a)
    first_result=verification(budget,CORRECT)
    budget.finish(first['id'],result=first_result)
    changed=CORRECT+'\n# a separately proposed candidate\n'
    second=budget.reserve(code_hash(changed),owner_invocation_id=owner_b)
    second_result=verification(budget,changed)
    completed=budget.finish(second['id'],result=second_result)
    assert completed['last_correct_candidate']['source_hash']==code_hash(changed)
    untouched=budget.settle_owner(owner_b)
    assert untouched['settled_reservation_ids']==[]
    assert untouched['records'][1]['status']=='completed'
    rejected=budget.settle_owner(owner_b,reject_completed=True)
    rows=rejected['records']
    assert rejected['settled_reservation_ids']==[second['id']]
    assert rows[1]['status']=='cancelled' and rows[1]['full_l2_passed'] is True
    assert rows[1]['verification_status_before_settlement']=='completed'
    assert rows[1]['accepted_by_supervisor'] is False
    assert rows[1]['verification_finished_epoch']==completed['records'][1]['finished_epoch']
    assert rows[1]['verification_finished_after_deadline'] is False
    assert Path(rows[1]['evidence']).is_file()
    assert rows[0]['status']=='completed' and rejected['reserved']==2
    assert rejected['last_correct_candidate']['source_hash']==code_hash(CORRECT)
    assert budget.settle_owner(owner_b,reject_completed=True)['settled_reservation_ids']==[]
    with pytest.raises(hb.HostTaskBudgetError) as error:
        budget.reserve(code_hash(changed),purpose='final')
    assert error.value.code=='host_final_source'
    final=budget.reserve(code_hash(CORRECT),purpose='final',owner_invocation_id='c'*32)
    final_result=verification(budget,CORRECT)
    done=budget.finish(final['id'],result=final_result)
    assert done['phase']=='final_complete'
    late_cancel=budget.settle_owner('c'*32,outcome='failed',reject_completed=True)
    assert late_cancel['phase']=='final_failed' and late_cancel['reserved']==3
    assert late_cancel['records'][-1]['full_l2_passed'] is True
    assert late_cancel['records'][-1]['accepted_by_supervisor'] is False


def test_cooperative_system_exit_failure_is_normalized_by_owning_cancel_supervisor(tmp_path):
    _,budget=prepared(tmp_path,hb.HostTaskLimits(max_verifications=8))
    owner,other='a'*32,'b'*32
    own=budget.reserve(code_hash(CORRECT),owner_invocation_id=owner)
    try:
        raise SystemExit('cooperative SIGTERM')
    except BaseException:
        failed=budget.finish(own['id'],outcome='failed')
    original=failed['records'][0]
    other_reservation=budget.reserve(code_hash(WRONG),owner_invocation_id=other)
    budget.finish(other_reservation['id'],outcome='failed')
    already_settled=budget.reserve(code_hash(WRONG),owner_invocation_id=owner)
    budget.settle_owner(owner,outcome='failed')
    assert budget.settle_owner(owner)['settled_reservation_ids']==[]
    normalized=budget.settle_owner(owner,outcome='cancelled',reject_completed=True)
    rows={r['id']:r for r in normalized['records']}
    assert normalized['settled_reservation_ids']==[own['id']]
    assert rows[own['id']]['status']=='cancelled'
    assert rows[own['id']]['verification_status_before_settlement']=='failed'
    assert rows[own['id']]['verification_finished_epoch']==original['finished_epoch']
    assert rows[own['id']]['verification_finished_after_deadline']==original['finished_after_deadline']
    assert rows[own['id']]['accepted_by_supervisor'] is False
    assert rows[other_reservation['id']]['status']=='failed'
    assert rows[already_settled['id']]['status']=='failed'
    assert rows[already_settled['id']]['settled_by']=='owning_supervisor'
    assert normalized['reserved']==3 and normalized['remaining_verifications']==5
    assert budget.settle_owner(owner,reject_completed=True)['settled_reservation_ids']==[]
