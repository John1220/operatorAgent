"""复现聚合器的规则（用户 2026-11-10 选定 ①A / ③A）。"""
import json

import pytest

from benchmarks.experiments.v08 import ac_aggregate, ac_plan


@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    output = tmp_path_factory.mktemp('plan') / 'plan.json'
    budget = dict(cells=6, gpu_minutes=180, max_model_calls=20,
                  max_output_tokens_per_call=40960,
                  total_output_tokens_reservations=819200, max_verifications=8,
                  wall_clock_s=2400, final_reserve=1)
    return ac_plan.freeze(budget, output, version='v6')


def _round(plan, diff, *, complete=True, comparability='established'):
    return dict(plan_hash=plan['hash'], hash='round-hash',
                sampling_comparability=comparability,
                delivery=dict(paired_difference_C_minus_A=diff, round_complete=complete))


def test_every_round_at_the_unit_is_reproduced(plan):
    body = ac_aggregate.aggregate(plan, [_round(plan, 1)] * 2)
    assert body['verdict'] == 'reproduced'
    assert body['formal_gain_claim'] is True
    assert body['scope'].startswith('weak host')


def test_any_round_below_the_unit_is_null_and_stops(plan):
    body = ac_aggregate.aggregate(plan, [_round(plan, 1), _round(plan, 0)])
    assert body['verdict'] == 'null'
    assert body['formal_gain_claim'] is False
    assert '不再追加轮次' in body['reason']


def test_incomplete_or_mismatched_rounds_are_inconclusive_not_null(plan):
    assert ac_aggregate.aggregate(plan, [_round(plan, 1)])['verdict'] == 'inconclusive'
    assert ac_aggregate.aggregate(
        plan, [_round(plan, 1), _round(plan, 1, complete=False)])['verdict'] == 'inconclusive'
    assert ac_aggregate.aggregate(
        plan, [_round(plan, 1), _round(plan, 1, comparability='not_established')])['verdict'] == 'inconclusive'


def test_the_reproduction_plan_must_register_its_rule(plan):
    from benchmarks.experiments.v08.ac_plan import digest

    broken = json.loads(json.dumps(plan))
    broken.pop('reproduction')
    broken['hash'] = digest({key: value for key, value in broken.items() if key != 'hash'})
    problems = ac_plan.validate(broken)
    assert problems and any('复现' in problem for problem in problems)


def test_the_plan_budget_stays_within_the_gateway_guard():
    """网关硬性拒绝每次输出预留 > 40960（`PolicyRefused` → 429）。

    2026-11-10 实测：把预留提到 65536 后 12 格全部 `invalid`/429，一格都没起来。
    """
    from benchmarks.experiments.v08.ac_plan import FROZEN_PER_CELL

    assert FROZEN_PER_CELL['max_output_tokens_per_call'] <= 40960
