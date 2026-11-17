"""D2 执行器 mock：扫描→择优→独立终评→判读，窗口拒绝与零模型路径。"""
import json
from types import SimpleNamespace

import pytest

from benchmarks.experiments.v08 import d2_plan, d2_rewrite, d2_sweep
from operator_agent.core.run_contract import code_hash
from operator_agent.decide.noise_preflight import PROTOCOL

SEED_CODE = open('output/v08-p2-seed7-medium-20261116/run/output/layernorm/nvidia_gpu/'
                 '20261116_172641_c1564572/code.py', encoding='utf-8').read()
LABELS = [f"bs{bs}-w{nw}" for bs in d2_plan.BLOCK_SIZES for nw in d2_plan.NUM_WARPS]


@pytest.fixture()
def plan(tmp_path):
    return d2_plan.freeze(SEED_CODE, tmp_path / 'plan.json')


def _cases():
    return [dict(case_id=case_id, verdict='passed') for case_id in d2_plan.required_case_ids()]


def _receipt(code, purpose, median, *, validity='valid', evidence=None):
    return dict(ok=True, source_hash=code_hash(code), contract_hash=d2_plan.CONTRACT_HASH,
                verification_purpose=purpose, stop_after='L3', evidence=evidence or purpose,
                evidence_complete=True, validity=validity, levels=dict(L1=True, L2=True, L3=True),
                verdicts=[
                    dict(level='L2', passed=True,
                         metrics=dict(correctness=dict(verdict='passed', cases=_cases()))),
                    dict(level='L3', passed=True, metrics=dict(
                        measurement_validity=validity, baseline_type='pytorch_eager',
                        median_time_ms=median))])


def _wire(plan, medians, *, stop=None, degraded=(), l2_fail=()):
    """medians: 标签→explore 中位；'best'/'seed' 键给终评。返回 (preflight, evaluate, trace)。"""
    trace = []
    by_hash = {code_hash(d2_rewrite.rewrite(SEED_CODE, **point['params'])): point['params']
               for point in plan['points']}

    def gate(path):
        trace.append(path.name)
        return dict(schema=PROTOCOL, action='hold' if path.name == stop else 'start', ok=True,
                    channel=d2_plan.PROBE_CHANNEL,
                    identity=dict(backend='ssh', shape=list(d2_plan.SHAPE)),
                    planned_effect_pct=2.5, effective_band_pct=0.1)

    def evaluate(name, code, purpose):
        trace.append(f'{name}:{purpose}')
        if purpose == 'explore':
            params = by_hash[code_hash(code)]
            label = d2_sweep.point_label(params)
            if label in l2_fail:
                return {}
            return _receipt(code, 'explore', medians.get(label, 0.3),
                            validity='degraded' if label in degraded else 'valid',
                            evidence=f'{label}/explore')
        return _receipt(code, 'final', medians[name], evidence=f'{name}/final')

    return gate, evaluate, trace


def test_sweep_selects_valid_argmin_and_judges_independent_finals(plan, tmp_path):
    medians = {label: 0.3 for label in LABELS}
    medians.update({'bs1024-w8': 0.25, 'best': 0.25, 'seed': 0.268})
    gate, evaluate, trace = _wire(plan, medians)
    root = tmp_path / 'sweep'
    result = d2_sweep.run_sweep(root=root, plan=plan, seed_code=SEED_CODE,
                                preflight=gate, evaluate=evaluate)
    assert result['decidable'] and result['verdict'] == 'sweep_helps'
    assert result['selected_point'] == dict(block_size=1024, num_warps=8)
    assert result['eligible_points'] == 16
    assert result['formal_gain_claim'] is False
    assert trace == ['preflight-start.json', *[f'{label}:explore' for label in LABELS],
                     'preflight-final.json', 'best:final', 'seed:final']
    assert d2_sweep.read_json(root / 'sweep.json') == result
    record = d2_sweep.read_json(root / 'points' / 'bs1024-w8' / 'point.json')
    assert record['eligible'] and record['median_ms'] == 0.25


@pytest.mark.parametrize('stop', ['preflight-start.json', 'preflight-final.json'])
def test_sweep_holds_on_rejected_window(plan, tmp_path, stop):
    medians = {label: 0.3 for label in LABELS}
    medians.update({'bs1024-w8': 0.25, 'best': 0.25, 'seed': 0.268})
    gate, evaluate, trace = _wire(plan, medians, stop=stop)
    result = d2_sweep.run_sweep(root=tmp_path / 'sweep', plan=plan, seed_code=SEED_CODE,
                                preflight=gate, evaluate=evaluate)
    assert result['verdict'] == 'held' and result['decidable'] is False
    if stop == 'preflight-start.json':
        assert trace == ['preflight-start.json']
    else:
        assert 'best:final' not in trace and 'seed:final' not in trace


def test_sweep_excludes_degraded_and_l2_failed_points(plan, tmp_path):
    medians = {label: 0.3 for label in LABELS}
    medians.update({'bs256-w2': 0.1, 'bs512-w2': 0.15, 'bs1024-w8': 0.25,
                    'best': 0.25, 'seed': 0.268})
    gate, evaluate, _ = _wire(plan, medians, degraded=('bs256-w2',), l2_fail=('bs512-w2',))
    root = tmp_path / 'sweep'
    result = d2_sweep.run_sweep(root=root, plan=plan, seed_code=SEED_CODE,
                                preflight=gate, evaluate=evaluate)
    assert result['selected_point'] == dict(block_size=1024, num_warps=8)
    assert result['eligible_points'] == 14
    degraded = d2_sweep.read_json(root / 'points' / 'bs256-w2' / 'point.json')
    assert degraded['eligible'] is False and degraded['validity'] == 'degraded'
    failed = d2_sweep.read_json(root / 'points' / 'bs512-w2' / 'point.json')
    assert failed['eligible'] is False and 'reason' in failed


def test_sweep_unknown_when_no_point_is_eligible(plan, tmp_path):
    medians = {label: 0.3 for label in LABELS}
    gate, evaluate, trace = _wire(plan, medians, degraded=tuple(LABELS))
    result = d2_sweep.run_sweep(root=tmp_path / 'sweep', plan=plan, seed_code=SEED_CODE,
                                preflight=gate, evaluate=evaluate)
    assert result['verdict'] == 'unknown' and result['selected_point'] is None
    assert 'best:final' not in trace


def test_sweep_best_is_seed_point_means_no_increment(plan, tmp_path):
    medians = {label: 0.3 for label in LABELS}
    medians.update({'bs512-w4': 0.268, 'best': 0.268, 'seed': 0.268})
    gate, evaluate, _ = _wire(plan, medians)
    result = d2_sweep.run_sweep(root=tmp_path / 'sweep', plan=plan, seed_code=SEED_CODE,
                                preflight=gate, evaluate=evaluate)
    assert result['verdict'] == 'no_increment'
    assert result['best_is_seed_point'] is True
    assert result['selected_point'] == d2_plan.SEED_POINT


def test_sweep_refuses_drifted_seed(plan, tmp_path):
    gate, evaluate, _ = _wire(plan, {})
    with pytest.raises(ValueError, match='frozen hash'):
        d2_sweep.run_sweep(root=tmp_path / 'sweep', plan=plan, seed_code=SEED_CODE + '\n',
                           preflight=gate, evaluate=evaluate)


def test_sweep_bindings_carry_the_frozen_custom_card(tmp_path):
    gpu = dict(index=7, uuid='GPU-52ca6155-d8bc-a416-847a-6f0c71db1552', name='NVIDIA H20')
    plan = d2_plan.freeze(SEED_CODE, tmp_path / 'plan.json', gpu=gpu)
    medians = {label: 0.3 for label in LABELS}
    medians.update({'bs1024-w8': 0.25, 'best': 0.25, 'seed': 0.268})
    gate, evaluate, _ = _wire(plan, medians)
    root = tmp_path / 'sweep'
    result = d2_sweep.run_sweep(root=root, plan=plan, seed_code=SEED_CODE,
                                preflight=gate, evaluate=evaluate)
    assert result['decidable'] and result['verdict'] == 'sweep_helps'
    assert d2_sweep.read_json(root / 'bindings.json')['gpu'] == gpu


def test_live_explore_point_single_verify_with_valid_host_budget(tmp_path, monkeypatch):
    from operator_agent.app import hitl_tools
    from operator_agent.execute.host_task_budget import HostTaskLimits

    calls, budgets = [], []

    def prepare(_declaration, run, *, host_budget):
        run.mkdir(parents=True)
        budgets.append(host_budget)
        HostTaskLimits.from_dict(host_budget)  # 真契约校验：min-2 验证 / min-1 模型上限

    def verify(run, code, *, verification_purpose, stop_after, **_kwargs):
        calls.append((verification_purpose, stop_after))
        path = run / 'result.json'
        path.write_text('{"validity":"valid"}')
        return dict(evidence=str(path), verification_purpose=verification_purpose,
                    host_task=dict(phase='awaiting_final'))

    monkeypatch.setattr(hitl_tools, 'prepare', prepare)
    monkeypatch.setattr(hitl_tools, 'verify', verify)
    evaluate = d2_sweep.live_evaluation(tmp_path, SimpleNamespace(
        verify_timeout=10, host='test', work_dir='/tmp/test'))
    result = evaluate('bs512-w4', SEED_CODE, 'explore')
    assert calls == [('explore', 'L3')]
    assert result == dict(validity='valid', verification_purpose='explore')
    assert budgets[0]['max_verifications'] == 2 and budgets[0]['max_model_calls'] == 1


@pytest.mark.parametrize('phase', ['final_complete', 'final_failed'])
def test_live_final_requires_prior_l2_and_honors_host_phase(tmp_path, monkeypatch, phase):
    from operator_agent.app import hitl_tools

    calls = []

    def prepare(_declaration, run, *, host_budget):
        run.mkdir(parents=True)

    def verify(run, code, *, verification_purpose, stop_after, **_kwargs):
        calls.append((verification_purpose, stop_after))
        if stop_after == 'L2':
            return dict(levels=dict(L2=True))
        path = run / 'result.json'
        path.write_text('{"validity":"valid"}')
        return dict(evidence=str(path), verification_purpose='final',
                    host_task=dict(phase=phase))

    monkeypatch.setattr(hitl_tools, 'prepare', prepare)
    monkeypatch.setattr(hitl_tools, 'verify', verify)
    evaluate = d2_sweep.live_evaluation(tmp_path, SimpleNamespace(
        verify_timeout=10, host='test', work_dir='/tmp/test'))
    result = evaluate('best', SEED_CODE, 'final')
    assert calls == [('explore', 'L2'), ('final', 'L3')]
    if phase == 'final_complete':
        assert result == dict(validity='valid', verification_purpose='final')
    else:
        assert result == {}


def test_live_final_skips_dispatch_when_explore_l2_fails(tmp_path, monkeypatch):
    from operator_agent.app import hitl_tools

    calls = []

    def prepare(_declaration, run, *, host_budget):
        run.mkdir(parents=True)

    def verify(run, code, *, verification_purpose, stop_after, **_kwargs):
        calls.append((verification_purpose, stop_after))
        return dict(levels=dict(L2=False))

    monkeypatch.setattr(hitl_tools, 'prepare', prepare)
    monkeypatch.setattr(hitl_tools, 'verify', verify)
    evaluate = d2_sweep.live_evaluation(tmp_path, SimpleNamespace(
        verify_timeout=10, host='test', work_dir='/tmp/test'))
    assert evaluate('best', SEED_CODE, 'final') == {}
    assert calls == [('explore', 'L2')]