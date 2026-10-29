"""Offline summary keeps planned denominators and unknowns explicit."""
import pytest
from benchmarks import s5_summary as s5
from benchmarks.s5_task_manifest import manifest
from operator_agent.run_contract import digest


def record(task, arm, passed, *, state='finished', usage=None):
    return dict(case_id=task['case_id'], arm=arm, execution_state=state,
        final_l2=dict(full_l2_passed=passed, contract_hash=task['contract_hash'], source_sha256='a'*64),
        api_usage=usage)


def test_schedule_is_reproducible_and_keeps_all_three_arms_per_task():
    value=s5.schedule()
    assert len(value['rows'])==42
    assert value['hash']==digest({k:v for k,v in value.items() if k!='hash'})
    assert value==s5.schedule()
    assert [r['sequence'] for r in value['rows']]==list(range(1,43))
    assert [[r['arm'] for r in value['rows'][i:i+3]] for i in range(0,9,3)] == [
        ['A','B','C'], ['B','C','A'], ['C','A','B']]
    assert len({(r['case_id'],r['arm']) for r in value['rows']})==42


def test_empty_summary_never_turns_unexecuted_or_unknown_cost_into_zero():
    value=s5.summarize([])
    assert value['formal_s5_credit'] is False and value['artifact_identity_verified'] is False
    assert len(value['rows'])==42
    for group in value['groups'].values():
        assert group['planned']==14 and group['counts']['unexecuted']==14
        assert group['usage']['prompt_tokens'] is None and group['usage']['completion_tokens'] is None
    for pair in value['pairs'].values():
        assert pair['planned']==14 and pair['comparable']==0
        assert pair['unknown']==14 and pair['mean_difference'] is None


def test_pairs_only_compare_finished_identity_bound_boolean_reports():
    task=manifest()['tasks'][0]
    records=[record(task,'A',False),record(task,'B',True),record(task,'C',True,state='interrupted')]
    summary=s5.summarize(records)
    ba=summary['pairs']['B-minus-A']; cb=summary['pairs']['C-minus-B']
    assert ba['comparable']==1 and ba['mean_difference']==1 and ba['unknown']==13
    assert cb['comparable']==0 and cb['mean_difference'] is None and cb['unknown']==14
    assert summary['groups']['A']['counts']['reported_fail']==1
    assert summary['groups']['B']['counts']['reported_pass']==1
    assert summary['groups']['C']['counts']['unknown']==1
    assert summary['groups']['A']['usage']['unknown']==1


@pytest.mark.parametrize('field,value', [('contract_hash','b'*64),('source_sha256','bad'),
                                        ('full_l2_passed',1),('full_l2_passed',None)])
def test_malformed_final_evidence_stays_unknown(field,value):
    task=manifest()['tasks'][0]; item=record(task,'A',True); item['final_l2'][field]=value
    result=s5.summarize([item]); row=next(r for r in result['rows'] if r['case_id']==task['case_id'] and r['arm']=='A')
    assert row['status']=='unknown' and row['full_l2_passed'] is None


def test_usage_missing_and_unverified_are_unknown_not_synthesized_zero():
    tasks=manifest()['tasks']
    full={'evidence':'provider_complete','prompt_tokens':12,'completion_tokens':3}
    unverified={'evidence':'unverified','prompt_tokens':0,'completion_tokens':0}
    result=s5.summarize([record(tasks[0],'A',True,usage=full),record(tasks[1],'A',True,usage=unverified)])
    usage=result['groups']['A']['usage']
    assert usage==dict(reported_complete=1, unknown=1, prompt_tokens=12, completion_tokens=3, total_is_partial=True)


@pytest.mark.parametrize('usage', [
    {'evidence':'provider_complete','prompt_tokens':True,'completion_tokens':0},
    {'evidence':'provider_complete','prompt_tokens':-1,'completion_tokens':0},
    {'evidence':'provider_complete','completion_tokens':0},
])
def test_invalid_usage_never_becomes_known(usage):
    task=manifest()['tasks'][0]; value=s5.summarize([record(task,'A',True,usage=usage)])
    assert value['groups']['A']['usage']['unknown']==1
    assert value['groups']['A']['usage']['prompt_tokens'] is None


def test_duplicate_unplanned_and_invalid_records_rejected():
    task=manifest()['tasks'][0]; item=record(task,'A',True)
    with pytest.raises(ValueError,match='duplicate'): s5.summarize([item,item])
    with pytest.raises(ValueError,match='unplanned'): s5.summarize([dict(item,arm='D')])
    with pytest.raises(ValueError,match='identity'): s5.summarize([dict(item,case_id=[])])
    with pytest.raises(ValueError,match='state'): s5.summarize([dict(item,execution_state='banana')])
