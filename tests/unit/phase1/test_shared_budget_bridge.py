"""Shared quota limits stay normal task stops; audit faults remain fail closed."""
import json

import pytest

from operator_agent.execute import s5_shared_budget as shared
from operator_agent.execute.llm_budget import (
    LLMCallLimits, LLMCallRefused, ProductionBudgetExhausted)
from operator_agent.execute.s5_inner_budget import SharedProductionBudget
from operator_agent.execute.task_budget import TaskBudgetConfig, snapshot, stopped_reason


IDENTITY = {'model': 'offline-budget-test'}


def create(tmp_path, **changes):
    path = tmp_path/'shared.sqlite'
    options = dict(task_id='phase1-cell', max_calls=5, max_output_tokens=20,
                   total_output_tokens=100, wall_clock_s=120)
    options.update(changes)
    shared.create(path, **options)
    limits = LLMCallLimits(max_calls=5, max_output_tokens=30,
        total_output_token_reservations=150, allowed_callers=('Coder',),
        call_seconds=10, campaign_seconds=120)
    budget = SharedProductionBudget(limits, tmp_path/'model-journal',
        shared_path=path, task_id='phase1-cell')
    budget.bind(IDENTITY)
    return path, budget


def reserve(budget, tokens=10):
    return budget.reserve(IDENTITY, json.dumps({'max_tokens': tokens}), 'Coder', 5)


def assert_task_stop(tmp_path, reason, stop):
    receipt = json.loads((tmp_path/'model-journal'/'refusal.json').read_text())
    assert receipt['reason'] == reason and receipt['before_dispatch'] is True
    view = snapshot(tmp_path, TaskBudgetConfig(wall_clock_s=120, max_llm_calls=5),
                    0., now=0.)
    assert view['model_calls']['journal_state'] == 'known'
    assert stopped_reason(view) == stop


def test_outer_and_inner_reservations_exhaust_without_poison_or_refund(tmp_path):
    path, budget = create(tmp_path, max_calls=2)
    call, _ = reserve(budget)
    budget.finish(call, outcome='answered', usage_source='test_fixture')
    shared.reserve(path, participant='outer', caller='agent', max_output_tokens=10,
                   request_sha256='a'*64)
    with pytest.raises(ProductionBudgetExhausted) as caught:
        reserve(budget)
    assert caught.value.reason == 'llm_call_budget'
    assert_task_stop(tmp_path, 'llm_call_budget', 'budget_llm_calls')
    state = shared.snapshot(path)
    assert state['reserved_calls'] == 2 and not state['aborted']
    assert all(row['dispatched_monotonic'] is None for row in state['calls'])
    assert budget.summary()['reserved_calls'] == 1
    with pytest.raises(shared.BudgetError, match='call_budget_exhausted'):
        shared.reserve(path, participant='outer', caller='retry', max_output_tokens=1,
                       request_sha256='b'*64)


@pytest.mark.parametrize('per_call', [True, False])
def test_shared_output_limits_publish_normal_refusal(tmp_path, per_call):
    path, budget = create(tmp_path, total_output_tokens=20)
    if not per_call:
        shared.reserve(path, participant='outer', caller='agent', max_output_tokens=15,
                       request_sha256='a'*64)
    with pytest.raises(ProductionBudgetExhausted) as caught:
        reserve(budget, 25 if per_call else 10)
    assert caught.value.reason == 'llm_output_budget'
    assert_task_stop(tmp_path, 'llm_output_budget', 'budget_llm_output')
    state = shared.snapshot(path)
    assert state['reserved_calls'] == (0 if per_call else 1)
    assert not state['aborted'] and budget.summary()['reserved_calls'] == 0


def test_shared_deadline_precedes_later_local_deadline(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(shared, '_now', lambda: clock[0])
    path, budget = create(tmp_path, wall_clock_s=1)
    clock[0] += 2  # Within the shared clock-identity tolerance, beyond its cap.
    assert budget.remaining_seconds() > 1
    with pytest.raises(ProductionBudgetExhausted) as caught:
        reserve(budget)
    assert caught.value.reason == 'task_wall_clock'
    assert_task_stop(tmp_path, 'task_wall_clock', 'budget_wall_clock')
    state = shared.snapshot(path)
    assert state['reserved_calls'] == 0 and not state['aborted']


def test_unknown_shared_reservation_failure_still_stops_all(tmp_path, monkeypatch):
    path, budget = create(tmp_path)
    def broken(*args, **kwargs):
        raise shared.BudgetError('ledger_unavailable')
    monkeypatch.setattr(shared, 'reserve', broken)
    with pytest.raises(LLMCallRefused, match='campaign stop recorded') as caught:
        reserve(budget)
    assert not isinstance(caught.value, ProductionBudgetExhausted)
    state = shared.snapshot(path)
    assert state['aborted'] and state['reserved_calls'] == 0


def test_refusal_journal_failure_cannot_be_misreported_as_normal_limit(tmp_path, monkeypatch):
    from operator_agent.execute import llm_budget
    path, budget = create(tmp_path, max_calls=1)
    reserve(budget)
    publish = llm_budget.atomic_json
    def fail_refusal(path, *args, **kwargs):
        if path.name == 'refusal.json':
            raise OSError('fixture journal failure')
        return publish(path, *args, **kwargs)
    monkeypatch.setattr(llm_budget, 'atomic_json', fail_refusal)
    with pytest.raises(LLMCallRefused, match='reservation_refusal') as caught:
        reserve(budget)
    assert not isinstance(caught.value, ProductionBudgetExhausted)
    state = shared.snapshot(path)
    assert state['aborted'] and state['reserved_calls'] == 1
    assert state['calls'][0]['dispatched_monotonic'] is None
