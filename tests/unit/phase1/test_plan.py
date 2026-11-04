"""Freeze the rehearsal's scope, entry points, budgets and all eight slots."""
from copy import deepcopy

import pytest

from benchmarks.experiments.phase1 import plan as plans
from operator_agent.core.run_contract import digest


@pytest.fixture(autouse=True)
def stable_implementation(monkeypatch):
    # Other workers may still be editing protocol files while these tests run.
    identity = dict(protocol={'fixture.py': 'a'*64}, harness={'plan.py': 'b'*64})
    monkeypatch.setattr(plans, 'implementation_identity', lambda: deepcopy(identity))


def resign(value):
    value['hash'] = digest({key: item for key, item in value.items() if key != 'hash'})
    return value


def test_freeze_retains_all_eight_interleaved_slots_with_one_common_budget(tmp_path):
    path = tmp_path/'rehearsal'
    plan = plans.freeze(path)
    assert plans.load(path/'plan.json') == plan
    assert len(plan['tasks']) == 4
    assert len(plan['schedule']) == 8
    assert [slot['arm'] for slot in plan['schedule']] == ['A', 'B', 'B', 'A', 'A', 'B', 'B', 'A']
    assert len({slot['cell_id'] for slot in plan['schedule']}) == 8
    for task in plan['tasks']:
        assert {slot['arm'] for slot in plan['schedule'] if slot['task_id'] == task['task_id']} == {'A', 'B'}
        assert 'budget' not in task
    assert all(set(slot) == {'cell_id', 'task_id', 'arm'} for slot in plan['schedule'])
    assert plan['budget'] == dict(wall_clock_s=30., max_llm_calls=8, oracle_total=20)
    assert plan['entries'] == dict(A='benchmarks.experiments.phase1.baseline',
                                  B='benchmarks.experiments.phase1.delegate')
    assert plan['final_evaluation']['max_verifications'] == 1
    assert plan['execution_context']['test_only'] is True
    assert plan['execution_context']['gpu_acceptance'] is False
    assert plan['formal_comparison_credit'] is False
    assert plan['dsh_entry_verified'] is False


def test_rehearsal_can_reduce_the_common_development_budget(tmp_path):
    policy = dict(wall_clock_s=1.5, max_llm_calls=1, oracle_total=1)
    plan = plans.freeze(tmp_path/'reduced', budget=policy)
    assert plans.validate(plan)['budget'] == policy


@pytest.mark.parametrize('field,value', [
    ('max_llm_calls', 31), ('wall_clock_s', 2401), ('oracle_total', 21)])
def test_cannot_expand_the_registered_development_budget(tmp_path, field, value):
    policy = dict(wall_clock_s=30., max_llm_calls=8, oracle_total=20)
    policy[field] = value
    destination = tmp_path/'too-large'
    with pytest.raises(ValueError, match='cannot enlarge'):
        plans.freeze(destination, budget=policy)
    assert not destination.exists()
    plan = plans.freeze(tmp_path/'valid')
    plan['budget'] = policy
    with pytest.raises(ValueError, match='budget expansion'):
        plans.validate(resign(plan))


def test_freeze_and_validate_reject_live_model_configuration(tmp_path):
    with pytest.raises(ValueError, match='fake model'):
        plans.freeze(tmp_path/'live', fake={'llm': 'live'})
    assert not (tmp_path/'live').exists()
    plan = plans.freeze(tmp_path/'fake')
    plan['fake']['llm'] = 'live'
    with pytest.raises(ValueError, match='cannot dispatch live'):
        plans.validate(resign(plan))


def test_no_rehearsal_artifacts_are_created_under_the_repository(tmp_path, monkeypatch):
    repo = tmp_path/'source'
    repo.mkdir()
    monkeypatch.setattr(plans, 'REPO', repo)
    for destination in (repo, repo/'evidence'):
        with pytest.raises(ValueError, match='outside the source tree'):
            plans.freeze(destination)
    assert not (repo/'plan.json').exists()
    assert not (repo/'evidence').exists()


def test_hash_tampering_is_refused_without_a_recomputed_identity(tmp_path):
    plan = plans.freeze(tmp_path/'plan')
    plan['budget']['max_llm_calls'] = 1
    with pytest.raises(ValueError, match='hash mismatch'):
        plans.validate(plan)


@pytest.mark.parametrize('tamper', ['task', 'contract', 'order', 'missing_slot', 'arm_budget'])
def test_rehash_does_not_authorize_changed_tasks_or_schedule(tmp_path, tamper):
    plan = plans.freeze(tmp_path/'plan')
    if tamper == 'task':
        plan['tasks'][0]['declaration']['shapes'] = [[16, 8, 8]]
    elif tamper == 'contract':
        plan['tasks'][0]['contract_hash'] = 'f'*64
    elif tamper == 'order':
        plan['schedule'][0], plan['schedule'][1] = plan['schedule'][1], plan['schedule'][0]
    elif tamper == 'missing_slot':
        plan['schedule'].pop()
    else:
        plan['schedule'][0]['budget'] = dict(max_llm_calls=30)
    with pytest.raises(ValueError, match='fixture task|contract drift|schedule'):
        plans.validate(resign(plan))


@pytest.mark.parametrize('field', ['formal_comparison_credit', 'dsh_entry_verified'])
def test_rehash_cannot_promote_offline_rehearsal_to_formal_credit(tmp_path, field):
    plan = plans.freeze(tmp_path/'plan')
    plan[field] = True
    with pytest.raises(ValueError, match='cannot claim formal/DSH'):
        plans.validate(resign(plan))


def test_test_only_execution_context_cannot_be_weakened(tmp_path):
    plan = plans.freeze(tmp_path/'plan')
    plan['execution_context']['test_only'] = False
    plan['execution_context']['gpu_acceptance'] = True
    with pytest.raises(ValueError, match='execution context mismatch'):
        plans.validate(resign(plan))


def test_rehashed_arbitrary_worker_entry_is_not_a_valid_offline_plan(tmp_path):
    plan = plans.freeze(tmp_path/'plan')
    plan['entries']['A'] = 'operator_agent.app.cli'
    with pytest.raises(ValueError):
        plans.validate(resign(plan))


@pytest.mark.parametrize('metadata', [
    dict(id='live-provider-model', transport='network', provider_dispatch=True),
    dict(id='fake-fixture-v1', transport='none', provider_dispatch=True)])
def test_rehashed_model_metadata_must_still_describe_the_scripted_model(tmp_path, metadata):
    plan = plans.freeze(tmp_path/'plan')
    plan['model'] = metadata
    with pytest.raises(ValueError):
        plans.validate(resign(plan))


@pytest.mark.parametrize('count', [0, 2, True, 1.0])
def test_final_evaluation_is_one_explicit_integer_reservation(tmp_path, count):
    plan = plans.freeze(tmp_path/'plan')
    plan['final_evaluation']['max_verifications'] = count
    with pytest.raises(ValueError, match='evaluation policy'):
        plans.validate(resign(plan))


def test_implementation_drift_refuses_dispatch_but_allows_explicit_read_only_inspection(tmp_path, monkeypatch):
    directory = tmp_path/'plan'
    frozen = plans.freeze(directory)
    monkeypatch.setattr(plans, 'implementation_identity', lambda: dict(protocol={'new': 'c'*64}, harness={}))
    with pytest.raises(ValueError, match='implementation drift'):
        plans.load(directory/'plan.json')
    assert plans.load(directory/'plan.json', check_implementation=False) == frozen


def test_freeze_never_replaces_a_previous_plan(tmp_path):
    path = tmp_path/'plan'
    frozen = plans.freeze(path)
    original = (path/'plan.json').read_bytes()
    with pytest.raises(FileExistsError):
        plans.freeze(path, fake={'profile': 'large'})
    assert (path/'plan.json').read_bytes() == original
    assert plans.load(path/'plan.json') == frozen
