"""D2 计划模块：冻结/校验、explore 读数、择优、Q1 判读与证据复算。"""
import json
from datetime import datetime, timezone

import pytest

from benchmarks.experiments.v08 import d2_plan, d2_rewrite
from operator_agent.core.run_contract import code_hash
from operator_agent.decide.noise_preflight import run_preflight

SEED_CODE = open('output/v08-p2-seed7-medium-20261116/run/output/layernorm/nvidia_gpu/'
                 '20261116_172641_c1564572/code.py', encoding='utf-8').read()


def _gate(**changes):
    gate = run_preflight(lambda: 1.0, channel=d2_plan.PROBE_CHANNEL,
                         identity=dict(backend='ssh', shape=list(d2_plan.SHAPE)),
                         planned_effect_pct=d2_plan.EFFECT_THRESHOLD_PCT)
    return dict(gate, **changes)


def _cases():
    return [dict(case_id=case_id, verdict='passed') for case_id in d2_plan.required_case_ids()]


def _explore(source_hash, median_ms, *, validity='valid', evidence='explore'):
    return dict(ok=True, source_hash=source_hash, contract_hash=d2_plan.CONTRACT_HASH,
                verification_purpose='explore', stop_after='L3', evidence=evidence,
                levels=dict(L1=True, L2=True, L3=True),
                verdicts=[
                    dict(level='L2', passed=True,
                         metrics=dict(correctness=dict(verdict='passed', cases=_cases()))),
                    dict(level='L3', passed=True, metrics=dict(
                        measurement_validity=validity, baseline_type='pytorch_eager',
                        median_time_ms=median_ms))])


def _final(source_hash, evidence, median_ms):
    return dict(ok=True, source_hash=source_hash, contract_hash=d2_plan.CONTRACT_HASH,
                verification_purpose='final', stop_after='L3', evidence_complete=True,
                evidence=evidence, validity='valid', levels=dict(L1=True, L2=True, L3=False),
                verdicts=[
                    dict(level='L2', passed=True,
                         metrics=dict(correctness=dict(verdict='passed', cases=_cases()))),
                    dict(level='L3', passed=False, metrics=dict(
                        measurement_validity='valid', baseline_type='pytorch_eager',
                        median_time_ms=median_ms))])


@pytest.fixture()
def plan(tmp_path):
    return d2_plan.freeze(SEED_CODE, tmp_path / 'plan.json')


def test_freeze_registers_16_distinct_points_and_seed_identity(plan):
    assert not d2_plan.validate(plan)
    assert len(plan['points']) == 16
    assert len({point['source_hash'] for point in plan['points']}) == 16
    seed_point = next(point for point in plan['points'] if point['params'] == d2_plan.SEED_POINT)
    assert seed_point['source_hash'] == d2_plan.SEED_SOURCE_HASH
    assert plan['budget']['max_model_calls'] == 0
    assert plan['formal_gain_claim'] is False


def test_freeze_refuses_drifted_seed(tmp_path):
    with pytest.raises(ValueError, match='frozen hash'):
        d2_plan.freeze(SEED_CODE + '\n# drift\n', tmp_path / 'plan.json')


GPU7 = dict(index=7, uuid='GPU-52ca6155-d8bc-a416-847a-6f0c71db1552', name='NVIDIA H20')


def test_freeze_binds_a_custom_card_without_touching_p2_identity(tmp_path):
    plan = d2_plan.freeze(SEED_CODE, tmp_path / 'plan.json', gpu=GPU7)
    assert plan['gpu'] == GPU7
    assert not d2_plan.validate(plan)
    assert d2_plan.GPU['index'] == 3  # P2 的冻结卡身份不被 D2 换卡改动
    seed_point = next(p for p in plan['points'] if p['params'] == d2_plan.SEED_POINT)
    assert seed_point['source_hash'] == d2_plan.SEED_SOURCE_HASH


def test_freeze_refuses_malformed_gpu_identity(tmp_path):
    with pytest.raises(ValueError, match='GPU identity'):
        d2_plan.freeze(SEED_CODE, tmp_path / 'plan.json',
                       gpu=dict(index=7, uuid='', name='NVIDIA H20'))


@pytest.mark.parametrize('bad', [None, dict(index='7', uuid='x', name='n'),
                                 dict(index=7, uuid='x', name='')])
def test_validate_refuses_malformed_gpu_identity(plan, bad):
    from operator_agent.core.run_contract import digest

    tampered = dict(plan, gpu=bad)
    tampered['hash'] = digest({k: v for k, v in tampered.items() if k != 'hash'})
    assert any('GPU identity' in issue for issue in d2_plan.validate(tampered))


def test_validate_rejects_tampered_budget(plan):
    tampered = dict(plan, budget=dict(plan['budget'], campaign_seconds=99999))
    assert any('budget' in issue for issue in d2_plan.validate(tampered))
    assert any('hash' in issue for issue in d2_plan.validate(tampered))


def test_explore_metrics_reads_median_and_reports_validity():
    receipt = _explore('a' * 64, 0.3, validity='degraded')
    metrics = d2_plan.explore_metrics(receipt, source_hash='a' * 64)
    assert metrics['median_time_ms'] == 0.3
    assert metrics['measurement_validity'] == 'degraded'


def test_explore_metrics_requires_full_bound_l2():
    receipt = _explore('a' * 64, 0.3)
    receipt['verdicts'][0]['metrics']['correctness']['cases'].pop()
    with pytest.raises(ValueError, match='full bound L2'):
        d2_plan.explore_metrics(receipt, source_hash='a' * 64)
    with pytest.raises(ValueError, match='point source'):
        d2_plan.explore_metrics(_explore('a' * 64, 0.3), source_hash='b' * 64)


def test_select_best_skips_ineligible_and_breaks_ties_by_enumeration():
    records = [
        dict(params=dict(block_size=256, num_warps=2), eligible=False, median_ms=0.1),
        dict(params=dict(block_size=512, num_warps=4), eligible=True, median_ms=0.3),
        dict(params=dict(block_size=1024, num_warps=8), eligible=True, median_ms=0.3),
        dict(params=dict(block_size=2048, num_warps=16), eligible=True, median_ms=0.2),
    ]
    assert d2_plan.select_best(records)['params'] == dict(block_size=2048, num_warps=16)
    assert d2_plan.select_best(records[:3])['params'] == dict(block_size=512, num_warps=4)
    assert d2_plan.select_best([records[0]]) is None


@pytest.mark.parametrize(('seed_ms', 'best_ms', 'expected'), [
    (0.268, 0.2613, 'sweep_helps'),
    (0.268, 0.267, 'no_increment'),
    (0.268, 0.268, 'no_increment'),
    (0.268, 0.270, 'sweep_hurts'),
])
def test_judge_uses_independent_final_medians(plan, seed_ms, best_ms, expected):
    best_hash = plan['points'][8]['source_hash']
    result = d2_plan.judge_sweep(
        gate=_gate(), seed_receipt=_final(d2_plan.SEED_SOURCE_HASH, 'seed/result.json', seed_ms),
        best_receipt=_final(best_hash, 'best/result.json', best_ms),
        best_source_hash=best_hash, plan=plan)
    assert result['decidable'] and result['verdict'] == expected
    assert result['difference_pct'] == pytest.approx((seed_ms - best_ms) / seed_ms * 100)
    assert result['formal_gain_claim'] is False
    assert result['best_is_seed_point'] is False


def test_judge_refuses_unregistered_source_and_shared_evidence(plan):
    seed = _final(d2_plan.SEED_SOURCE_HASH, 'seed/result.json', 0.268)
    best = _final(plan['points'][8]['source_hash'], 'best/result.json', 0.26)
    outside = d2_plan.judge_sweep(gate=_gate(), seed_receipt=seed, best_receipt=best,
                                  best_source_hash='f' * 64, plan=plan)
    assert outside['decidable'] is False
    shared = _final(d2_plan.SEED_SOURCE_HASH, 'seed/result.json', 0.268)
    same = d2_plan.judge_sweep(gate=_gate(), seed_receipt=seed, best_receipt=shared,
                               best_source_hash=d2_plan.SEED_SOURCE_HASH, plan=plan)
    assert same['decidable'] is False
    assert 'distinct final evaluations' in same['reasons'][0]


def seed_budget(evaluation, state):
    """预算记录住在任务账本里：落一条 `budget_opened`（形状与 Python 写的相同）。"""
    import json as _json
    (evaluation / 'journal.jsonl').write_text(_json.dumps(dict(
        schema='operator-task-journal-v2', rev=1, kind='budget_opened',
        task_id='fixture-task', budget=dict(host_task=dict(state, records=state['records'])))) + '\n',
        encoding='utf-8')


def _store(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _archive_sweep(root, plan, *, medians, started=None, gate_time=None):
    """落盘一份完整扫描证据树；medians 按点标签给 explore 中位时延。"""
    gate_time = plan['frozen_at'] + 100 if gate_time is None else gate_time
    started = plan['frozen_at'] + 200 if started is None else started
    _store(root / 'bindings.json', dict(sweep_plan_hash=plan['hash'],
           contract_hash=d2_plan.CONTRACT_HASH, seed_source_hash=d2_plan.SEED_SOURCE_HASH,
           gpu=plan['gpu']))
    records = []
    for point in plan['points']:
        label = f"bs{point['params']['block_size']}-w{point['params']['num_warps']}"
        directory = root / 'points' / label
        directory.mkdir(parents=True, exist_ok=True)
        code = d2_rewrite.rewrite(SEED_CODE, **point['params'])
        (directory / 'candidate.py').write_text(code, encoding='utf-8')
        median = medians.get(label, 0.3)
        receipt = _explore(point['source_hash'], median, evidence=str(directory / 'explore.json'))
        _store(directory / 'explore.json', receipt)
        metrics = d2_plan.explore_metrics(receipt, source_hash=point['source_hash'])
        record = dict(params=point['params'], source_hash=point['source_hash'],
                      median_ms=metrics['median_time_ms'],
                      validity=metrics['measurement_validity'],
                      eligible=metrics['measurement_validity'] == 'valid')
        _store(directory / 'point.json', record)
        records.append(record)
    best = d2_plan.select_best(records)
    if best is None:
        _store(root / 'sweep.json', dict(decidable=False, formal_gain_claim=False,
                                         verdict='unknown', selected_point=None))
        return root / 'sweep.json'
    gate = _gate(created_at=datetime.fromtimestamp(gate_time, timezone.utc).isoformat())
    _store(root / 'preflight-final.json', gate)
    finals = {}
    for name, params, median in (('best', best['params'], medians['best']),
                                 ('seed', d2_plan.SEED_POINT, medians['seed'])):
        code = d2_rewrite.rewrite(SEED_CODE, **params)
        digest = code_hash(code)
        verification = root / name / 'evaluation' / 'verify-2'
        path = verification / 'result.json'
        receipt = _final(digest, str(path), median)
        receipt.update(backend='ssh', created_at=started,
                       binding=dict(contract_hash=d2_plan.CONTRACT_HASH, source_hash=digest))
        receipt['verdicts'][-1]['metrics']['selected_gpu'] = plan['gpu']['index']
        _store(path, receipt)
        (verification / 'candidate.py').write_text(code, encoding='utf-8')
        _store(verification / 'complete.json', dict(
            evidence_complete=True, binding=receipt['binding'],
            files_sha256={'candidate.py': digest, 'result.json': d2_plan._sha(path)}))
        seed_budget(verification.parent, dict(records=[
            dict(id=1, purpose='explore', status='completed', full_l2_passed=True,
                 source_hash=digest),
            dict(id=2, purpose='final', status='completed', full_l2_passed=True,
                 finished_after_deadline=False, evidence=str(path), source_hash=digest)]))
        finals[name] = receipt
    judgement = d2_plan.judge_sweep(gate=gate, seed_receipt=finals['seed'],
                                    best_receipt=finals['best'],
                                    best_source_hash=best['source_hash'], plan=plan)
    _store(root / 'sweep.json', dict(judgement, selected_point=best['params'],
                                     eligible_points=sum(r['eligible'] for r in records)))
    return root / 'sweep.json'


def test_replay_recomputes_selection_and_judgment(plan, tmp_path):
    medians = {f"bs{bs}-w{nw}": 0.30 for bs in d2_plan.BLOCK_SIZES for nw in d2_plan.NUM_WARPS}
    medians.update({'bs1024-w8': 0.25, 'best': 0.25, 'seed': 0.268})
    path = _archive_sweep(tmp_path / 'sweep', plan, medians=medians)
    replay = d2_plan.replay_sweep(path, plan)
    assert replay['verdict'] == 'sweep_helps'
    summary = json.loads(path.read_text())
    assert summary['selected_point'] == dict(block_size=1024, num_warps=8)
    assert replay['difference_pct'] == pytest.approx((0.268 - 0.25) / 0.268 * 100)


def test_replay_rejects_tampered_candidate_and_model_dispatch(plan, tmp_path):
    medians = {f"bs{bs}-w{nw}": 0.30 for bs in d2_plan.BLOCK_SIZES for nw in d2_plan.NUM_WARPS}
    medians.update({'bs1024-w8': 0.25, 'best': 0.25, 'seed': 0.268})
    root = tmp_path / 'sweep'
    path = _archive_sweep(root, plan, medians=medians)
    candidate = root / 'points' / 'bs256-w2' / 'candidate.py'
    candidate.write_text(candidate.read_text() + '\n# tamper\n', encoding='utf-8')
    with pytest.raises(ValueError, match='frozen grid hash'):
        d2_plan.replay_sweep(path, plan)
    candidate.write_text(d2_rewrite.rewrite(SEED_CODE, block_size=256, num_warps=2),
                         encoding='utf-8')
    (root / 'points' / 'bs256-w2' / 'model').mkdir()
    with pytest.raises(ValueError, match='zero-model-call'):
        d2_plan.replay_sweep(path, plan)


def test_replay_reports_ineligible_sweep_as_unknown(plan, tmp_path):
    medians = {f"bs{bs}-w{nw}": 0.30 for bs in d2_plan.BLOCK_SIZES for nw in d2_plan.NUM_WARPS}
    root = tmp_path / 'sweep'
    for point in plan['points']:
        label = f"bs{point['params']['block_size']}-w{point['params']['num_warps']}"
        directory = root / 'points' / label
        directory.mkdir(parents=True, exist_ok=True)
        code = d2_rewrite.rewrite(SEED_CODE, **point['params'])
        (directory / 'candidate.py').write_text(code, encoding='utf-8')
        receipt = _explore(point['source_hash'], 0.3, validity='degraded')
        _store(directory / 'explore.json', receipt)
        _store(directory / 'point.json', dict(params=point['params'],
               source_hash=point['source_hash'], median_ms=0.3, validity='degraded',
               eligible=False))
    _store(root / 'bindings.json', dict(sweep_plan_hash=plan['hash'],
           contract_hash=d2_plan.CONTRACT_HASH, seed_source_hash=d2_plan.SEED_SOURCE_HASH,
           gpu=plan['gpu']))
    path = root / 'sweep.json'
    _store(path, dict(decidable=False, formal_gain_claim=False, verdict='unknown',
                      selected_point=None))
    replay = d2_plan.replay_sweep(path, plan)
    assert replay['decidable'] is False and replay['verdict'] == 'unknown'