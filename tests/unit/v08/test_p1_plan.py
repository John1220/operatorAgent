"""P1 性能 parity 预登记：计划校验 + 判读量具（零机时）。"""
import json

import pytest

from benchmarks.experiments.v08 import p1_plan

BUDGET = dict(pairs=3, max_model_calls=0, max_verifications=6, wall_clock_s=2400, gpu_minutes=60)


def test_design_validates_as_draft():
    plan = p1_plan.design()
    assert plan['status'] == 'draft_awaiting_budget'
    assert plan['formal_parity_claim'] is False
    assert p1_plan.validate(plan) == []


def test_registered_pairs_bind_the_delivered_contracts():
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task
    by_id = {row['task_id']: row for row in p1_plan.TASKS}
    for pair in p1_plan.PAIRS:
        declaration = by_id[pair['task_id']]['declaration']
        assert freeze_from_task(build_task(declaration)).hash == pair['contract_hash']
    assert len(p1_plan.PAIRS) == 3
    assert {p['task_id'] for p in p1_plan.PAIRS} == {'matmul-4096-fp32', 'attention-prefill-4096'}


def test_freeze_writes_once_and_validates(tmp_path):
    out = tmp_path / 'p1.json'
    plan = p1_plan.freeze(BUDGET, out)
    assert plan['status'] == 'frozen'
    assert plan['hash']
    assert p1_plan.validate(plan) == []
    assert json.loads(out.read_text())['plan_id'] == p1_plan.PLAN_ID
    with pytest.raises(FileExistsError):
        p1_plan.freeze(BUDGET, out)


def test_freeze_refuses_generation_budget(tmp_path):
    with pytest.raises(ValueError, match='max_model_calls must be 0'):
        p1_plan.freeze(dict(BUDGET, max_model_calls=2), tmp_path / 'x.json')


def test_freeze_refuses_pair_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match='budget.pairs'):
        p1_plan.freeze(dict(BUDGET, pairs=2), tmp_path / 'x.json')


def test_validate_refuses_contract_drift():
    plan = p1_plan.design()
    plan['pairs'][0]['contract_hash'] = '0' * 64
    problems = p1_plan.validate(plan)
    assert any('does not match the re-frozen task contract' in p for p in problems)


def test_validate_refuses_bad_source_hash():
    plan = p1_plan.design()
    plan['pairs'][0]['A_source_sha256'] = 'not-a-hash'
    assert any('64-hex source_sha256' in p for p in p1_plan.validate(plan))


def test_validate_refuses_probe_dtype_mismatch():
    plan = p1_plan.design()
    plan['probes']['matmul-4096-fp32']['command'] = \
        plan['probes']['matmul-4096-fp32']['command'].replace('--dtype fp32', '--dtype fp16')
    assert any('precision=fp32 but probe probes fp16' in p for p in p1_plan.validate(plan))


def test_validate_refuses_claim_and_l2_final():
    plan = p1_plan.design()
    plan['formal_parity_claim'] = True
    assert any('formal_parity_claim' in p for p in p1_plan.validate(plan))
    plan = p1_plan.design()
    plan['final_evaluation']['stop_after'] = 'L2'
    assert any('stop_after L3' in p for p in p1_plan.validate(plan))


# ── 判读量具 ──────────────────────────────────────────────────────────────

def test_judge_pair_parity_worse_faster_and_not_decidable():
    within = p1_plan.judge_pair(a_median_ms=1.0, c_median_ms=1.0005, a_valid=True, c_valid=True, band_pct=0.1)
    assert within['decidable'] and within['verdict'] == 'parity_within_band'
    worse = p1_plan.judge_pair(a_median_ms=1.0, c_median_ms=1.01, a_valid=True, c_valid=True, band_pct=0.1)
    assert worse['verdict'] == 'C_worse'
    faster = p1_plan.judge_pair(a_median_ms=1.0, c_median_ms=0.98, a_valid=True, c_valid=True, band_pct=0.1)
    assert faster['verdict'] == 'C_faster'
    invalid = p1_plan.judge_pair(a_median_ms=1.0, c_median_ms=1.0, a_valid=True, c_valid=False, band_pct=0.1)
    assert invalid['decidable'] is False and invalid['verdict'] == 'not_decidable'


def test_judge_series_unknown_worse_and_parity():
    nd = dict(decidable=False, verdict='not_decidable')
    assert p1_plan.judge_series([nd, nd])['verdict'] == 'unknown'
    assert p1_plan.judge_series([nd, nd])['formal_parity_claim'] is False
    worse = dict(decidable=True, verdict='C_worse')
    ok = dict(decidable=True, verdict='parity_within_band')
    assert p1_plan.judge_series([ok, worse])['verdict'] == 'C_worse_observed'
    assert p1_plan.judge_series([ok, worse])['formal_parity_claim'] is False
    holds = p1_plan.judge_series([ok, dict(decidable=True, verdict='C_faster'), nd])
    assert holds['verdict'] == 'parity_holds'
    assert holds['formal_parity_claim'] is True
    assert holds['decidable_pairs'] == 2
    assert 'no gain claim' in holds['boundaries']