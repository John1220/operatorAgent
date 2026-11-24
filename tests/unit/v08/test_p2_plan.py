import math
from datetime import datetime, timezone

import pytest

from benchmarks.experiments.v08 import p2_plan
from operator_agent.decide.noise_preflight import run_preflight


def _gate(**changes):
    gate = run_preflight(lambda: 1.0, channel=p2_plan.PROBE_CHANNEL,
                         identity=dict(backend='ssh', shape=list(p2_plan.SHAPE)),
                         planned_effect_pct=p2_plan.EFFECT_THRESHOLD_PCT)
    return dict(gate, **changes)


def _final(source_hash, evidence, median_ms):
    return dict(
        ok=True, source_hash=source_hash, contract_hash=p2_plan.CONTRACT_HASH,
        verification_purpose='final', stop_after='L3', evidence_complete=True,
        evidence=evidence, validity='valid', levels=dict(L1=True, L2=True, L3=False),
        verdicts=[
            dict(level='L2', passed=True, metrics=dict(correctness=dict(
                verdict='passed', cases=[dict(case_id=case_id, verdict='passed')
                                        for case_id in p2_plan.required_case_ids()]))),
            dict(level='L3', passed=False, metrics=dict(
                measurement_validity='valid', baseline_type='pytorch_eager',
                median_time_ms=median_ms)),
        ])


@pytest.mark.parametrize(('off_ms', 'on_ms', 'expected'), [
    (0.26808, 0.261378, 'optimize_helps'),
    (100.0, 97.6, 'no_increment'),
    (100.0, 100.1, 'no_increment'),
    (100.0, 101.0, 'optimize_hurts'),
])
def test_pair_uses_independent_medians_and_registered_effect(off_ms, on_ms, expected):
    hashes = dict(on='a' * 64, off='b' * 64)
    seed = _final(p2_plan.SEED_SOURCE_HASH, 'seed/result.json', 105.0)
    receipts = dict(on=_final(hashes['on'], 'on/result.json', on_ms),
                    off=_final(hashes['off'], 'off/result.json', off_ms))
    gate = _gate()
    result = p2_plan.judge_pair(gate=gate, seed_receipt=seed,
                               receipts=receipts, source_hashes=hashes)
    assert result['decidable'] and result['verdict'] == expected
    assert result['band_pct'] == gate['effective_band_pct']
    assert result['difference_pct'] == pytest.approx((off_ms - on_ms) / off_ms * 100)
    assert result['formal_gain_claim'] is False


def test_gate_refuses_shape_drift_and_effect_obscured_by_noise():
    with pytest.raises(ValueError, match='32768x6144'):
        p2_plan.gate_band(_gate(identity=dict(backend='ssh', shape=[16384, 6144])))
    with pytest.raises(ValueError, match='2.5%'):
        p2_plan.gate_band(_gate(effective_band_pct=2.5))
    with pytest.raises(ValueError, match='did not permit'):
        p2_plan.gate_band(_gate(action='hold'))


def test_final_requires_bound_full_correctness_even_if_top_level_l2_passed():
    receipt = _final('a' * 64, 'on/result.json', 0.3)
    receipt['verdicts'][0]['metrics']['correctness']['cases'].pop()
    with pytest.raises(ValueError, match='full bound L2'):
        p2_plan.final_metrics(receipt, source_hash='a' * 64)
    with pytest.raises(ValueError, match='candidate source'):
        p2_plan.final_metrics(receipt, source_hash='b' * 64)


@pytest.mark.parametrize('median_ms', [math.nan, math.inf, 0, -1, True])
def test_final_refuses_unusable_medians(median_ms):
    receipt = _final('a' * 64, 'on/result.json', median_ms)
    with pytest.raises(ValueError, match='finite positive candidate median'):
        p2_plan.final_metrics(receipt, source_hash='a' * 64)


def test_pair_refuses_shared_final_evidence():
    seed = _final(p2_plan.SEED_SOURCE_HASH, 'seed/result.json', 0.3)
    final = _final('a' * 64, 'once/result.json', 0.3)
    result = p2_plan.judge_pair(gate=_gate(), seed_receipt=seed, receipts=dict(on=final, off=final),
                               source_hashes=dict(on='a' * 64, off='a' * 64))
    assert result['decidable'] is False
    assert 'distinct final evaluations' in result['reasons'][0]


def seed_budget(evaluation, state):
    """预算记录住在任务账本里：落一条 `budget_opened`（形状与 Python 写的相同）。"""
    import json as _json
    (evaluation / 'journal.jsonl').write_text(_json.dumps(dict(
        schema='operator-task-journal-v2', rev=1, kind='budget_opened',
        task_id='fixture-task', budget=dict(host_task=dict(state, records=state['records'])))) + '\n',
        encoding='utf-8')


def _archive_pair(root, *, seed_code, order, started, plan=None, pair_id=None):
    import json

    from operator_agent.core.run_contract import code_hash

    def store(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    gate = _gate(created_at=datetime.fromtimestamp(started, timezone.utc).isoformat())
    store(root / 'preflight-final.json', gate)
    receipts, hashes = {}, {}
    for arm, code, median in [('seed', seed_code, 0.25), ('off', 'off code', 0.5),
                              ('on', 'on code', 0.4)]:
        digest = code_hash(code)
        verification = root / arm / 'evaluation' / 'verify-2'
        path = verification / 'result.json'
        receipt = _final(digest, str(path), median)
        receipt.update(backend='ssh', created_at=started + 1,
                       binding=dict(contract_hash=p2_plan.CONTRACT_HASH, source_hash=digest))
        receipt['verdicts'][-1]['metrics']['selected_gpu'] = p2_plan.GPU['index']
        store(path, receipt)
        (verification / 'candidate.py').write_text(code, encoding='utf-8')
        store(verification / 'complete.json', dict(
            evidence_complete=True, binding=receipt['binding'],
            files_sha256={'candidate.py': digest, 'result.json': p2_plan._sha(path)}))
        seed_budget(verification.parent, dict(records=[
            dict(purpose='explore'), dict(purpose='final', status='completed', full_l2_passed=True,
                finished_after_deadline=False, evidence=str(path), source_hash=digest),
        ]))
        receipts[arm], hashes[arm] = receipt, digest
        if arm == 'seed':
            continue
        directory = root / arm
        (directory / 'candidate.py').write_text(code, encoding='utf-8')
        store(directory / 'generation.json', dict(source_hash=digest, search=dict(path='llm')))
        journal = directory / 'model' / 'journal'
        store(journal / 'limits.json', p2_plan.BUDGET)
        store(journal / 'identity.json', dict(model=p2_plan.MODEL, endpoint_sha256='a' * 64))
        requests = []
        for index, caller in enumerate(('Search', 'Coder'), 1):
            request_hash = str(index) * 64
            store(journal / f'call-{index:03}.json', dict(
                caller=caller, status='answered', dispatched_at_utc='present',
                request_sha256=request_hash, output_tokens_reserved=p2_plan.SAMPLING['max_tokens']))
            wire = {k: v for k, v in p2_plan.SAMPLING.items() if k != 'schema'}
            wire.update(model=p2_plan.MODEL, stream=False)
            requests.append(dict(caller=caller, request_sha256=request_hash, effective=wire))
        (journal.parent / 'sampling-inner.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in requests), encoding='utf-8')
    seed_receipt = receipts.pop('seed')
    hashes.pop('seed')
    result = dict(p2_plan.judge_pair(gate=gate, seed_receipt=seed_receipt, receipts=receipts,
                                    source_hashes=hashes), arm_order=order)
    path = root / 'pilot.json'
    store(path, result)
    if plan is not None:
        store(root / 'bindings.json', dict(
            formal_plan_hash=plan['hash'], formal_pair_id=pair_id, arm_order=order,
            model=plan['model'], sampling=plan['sampling'], gpu=plan['gpu'],
            contract_hash=plan['contract_hash'], seed_source_hash=plan['seed_source_hash'],
            rounds_per_arm=1))
    return path


def test_formal_freeze_and_replay_count_only_two_new_pairs(tmp_path, monkeypatch):
    import json

    from operator_agent.core.run_contract import code_hash

    seed = 'seed code'
    monkeypatch.setattr(p2_plan, 'SEED_SOURCE_HASH', code_hash(seed))
    pilot = _archive_pair(tmp_path / 'pilot', seed_code=seed, order=['off', 'on'], started=10.0)
    plan = p2_plan.freeze(pilot, tmp_path / 'plan.json')
    assert not p2_plan.validate(plan)
    assert plan['pilot']['judgement']['improvement_vs_seed_pct'] == {'on': -60.0, 'off': -100.0}
    paths = [
        _archive_pair(tmp_path / pair['pair_id'], seed_code=seed, order=pair['arm_order'],
                      started=plan['frozen_at'] + index, plan=plan, pair_id=pair['pair_id'])
        for index, pair in enumerate(plan['pairs'], 1)
    ]
    with pytest.raises(ValueError, match='exactly two distinct'):
        p2_plan.judge_series(plan, [paths[0], paths[0]])
    with pytest.raises(ValueError, match='bindings differ'):
        p2_plan.judge_series(plan, list(reversed(paths)))
    assert p2_plan.judge_series(plan, paths)['formal_gain_claim'] is True
    paths[1].write_text(json.dumps(dict(
        decidable=False, formal_gain_claim=False, verdict='held', arm_order=['on', 'off'])),
        encoding='utf-8')
    series = p2_plan.judge_series(plan, paths)
    assert series['formal_gain_claim'] is False and series['verdict'] == 'unknown'
    assert series['helps_pairs'] == 1 and series['decidable_pairs'] == 1
