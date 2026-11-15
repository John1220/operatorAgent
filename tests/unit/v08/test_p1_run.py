"""P1 执行器编排（本地）：探针门、逐对判读、源码哈希绑定、hold 窗口。

探针与独立 L3 用 mock 替身（真机取数在 GPU 窗口）；源码解析走**真实** v6 证据树，
证明哈希绑定对着实际交付源码成立。
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.experiments.v08 import p1_plan, p1_run

AC_ROOT = 'output/v08-ac-20261108'


def _final(median, validity='valid'):
    return dict(levels={'L1': True, 'L2': True, 'L3': True}, validity=validity,
                verdicts=[dict(level='L3', metrics=dict(
                    median_time_ms=median, measurement_validity=validity,
                    speedup_vs_baseline=0.37))],
                evidence='result.json')


def _setup(monkeypatch, tmp_path, *, a_median, c_median, validity='valid', action='start'):
    plan = p1_plan.design()
    plan['budget'] = dict(pairs=3, max_model_calls=0, max_verifications=6,
                          wall_clock_s=2400, gpu_minutes=60)
    plan['status'] = 'frozen'
    plan['hash'] = p1_plan.digest({k: v for k, v in plan.items() if k != 'hash'})

    monkeypatch.setattr(p1_run, 'run_preflight',
                        lambda probe, **k: dict(action=action, effective_band_pct=0.1,
                                                channel='test-channel'))

    def fake_measure(pair, arm, task, code, **k):
        return _final(a_median if arm == 'A' else c_median, validity)

    monkeypatch.setattr(p1_run, 'measure_arm', fake_measure)
    return plan


def test_parity_holds_and_evidence_written(tmp_path, monkeypatch):
    plan = _setup(monkeypatch, tmp_path, a_median=1.0, c_median=1.0005)
    root = tmp_path / 'nested' / 'run'        # 不存在的嵌套目录：执行器必须自己建
    series = p1_run.run(plan, host='h', work_dir='/w', ac_root=AC_ROOT, root=root,
                        backend='mock', gpu_env={'OPERATOR_AGENT_REQUIRED_GPU': '3'})
    assert root.is_dir()
    assert series['verdict'] == 'parity_holds'
    assert series['formal_parity_claim'] is True
    assert series['decidable_pairs'] == 3
    assert (root / 'series.json').is_file()
    for pair in plan['pairs']:
        record = json.loads((root / pair['pair_id'] / 'pair.json').read_text())
        assert record['judgment']['verdict'] == 'parity_within_band'
        # 源码哈希绑定对着真实交付源码核过。
        assert record['arms']['A']['source_hash'] == pair['A_source_sha256']
        assert record['arms']['C']['source_hash'] == pair['C_source_sha256']


def test_c_worse_observed(tmp_path, monkeypatch):
    plan = _setup(monkeypatch, tmp_path, a_median=1.0, c_median=1.01)
    series = p1_run.run(plan, host='h', work_dir='/w', ac_root=AC_ROOT, root=tmp_path, backend='mock')
    assert series['verdict'] == 'C_worse_observed'
    assert series['formal_parity_claim'] is False


def test_degraded_window_is_unknown(tmp_path, monkeypatch):
    plan = _setup(monkeypatch, tmp_path, a_median=1.0, c_median=1.0, validity='degraded')
    series = p1_run.run(plan, host='h', work_dir='/w', ac_root=AC_ROOT, root=tmp_path, backend='mock')
    assert series['verdict'] == 'unknown'
    assert series['decidable_pairs'] == 0


def test_held_window_does_not_measure(tmp_path, monkeypatch):
    plan = _setup(monkeypatch, tmp_path, a_median=1.0, c_median=1.0, action='hold')
    series = p1_run.run(plan, host='h', work_dir='/w', ac_root=AC_ROOT, root=tmp_path, backend='mock')
    assert series['verdict'] == 'unknown'
    assert series['gate_action'] == 'hold'
    assert not (tmp_path / plan['pairs'][0]['pair_id']).exists()


def test_source_hash_mismatch_refuses(tmp_path):
    pair = dict(p1_plan.PAIRS[0])
    pair['A_source_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='source hash'):
        p1_run.resolve_source(pair, 'A', AC_ROOT)


def test_measure_arm_retries_transport_unknown(tmp_path, monkeypatch):
    from benchmarks.experiments.v08 import ac_driver
    pair = p1_plan.PAIRS[0]
    task = next(t for t in p1_plan.TASKS if t['task_id'] == pair['task_id'])
    calls = []

    def fake_final(run_dir, code, **kw):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / 'contract.json').write_text(json.dumps({'hash': pair['contract_hash']}))
        calls.append(str(run_dir))
        if len(calls) == 1:
            return dict(final=dict(levels={'L2': False}, verdicts=[
                dict(level='L2', metrics={'correctness': {'verdict': 'unknown'}})]))
        return dict(final=dict(levels={'L1': True, 'L2': True, 'L3': True}, validity='valid',
                               verdicts=[dict(level='L2', metrics={'correctness': {'verdict': 'passed'}}),
                                         dict(level='L3', metrics={'median_time_ms': 1.0,
                                                                   'measurement_validity': 'valid'})]))

    monkeypatch.setattr(ac_driver, 'final_evaluation', fake_final)
    final = p1_run.measure_arm(pair, 'A', task, 'code', host='h', work_dir='/w',
                               run_dir=tmp_path / 'A', backend='mock', host_budget={})
    assert len(calls) == 2                       # unknown 触发了一次重试
    assert calls[1].endswith('-retry1')          # 重试用新 run_dir
    assert p1_run._l2_status(final) == 'passed'


def test_measure_arm_stops_on_decisive_failure(tmp_path, monkeypatch):
    from benchmarks.experiments.v08 import ac_driver
    pair = p1_plan.PAIRS[0]
    task = next(t for t in p1_plan.TASKS if t['task_id'] == pair['task_id'])
    calls = []

    def fake_final(run_dir, code, **kw):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / 'contract.json').write_text(json.dumps({'hash': pair['contract_hash']}))
        calls.append(str(run_dir))
        return dict(final=dict(levels={'L2': False}, verdicts=[
            dict(level='L2', metrics={'correctness': {'verdict': 'failed'}})]))

    monkeypatch.setattr(ac_driver, 'final_evaluation', fake_final)
    final = p1_run.measure_arm(pair, 'A', task, 'code', host='h', work_dir='/w',
                               run_dir=tmp_path / 'A', backend='mock', host_budget={})
    assert len(calls) == 1                       # 数值失败是决定性的，不重试
    assert p1_run._l2_status(final) == 'failed'


def test_run_preflight_creates_receipt_parent(tmp_path, monkeypatch):
    """真实 CLI 写回执不建父目录（首轮全量就栽在这）：执行器必须自己建。"""
    out = tmp_path / 'nested' / 'run' / 'gate-x.json'

    def fake_run(argv, **kw):
        # 模拟 CLI _emit：父目录存在才写得进，否则 FileNotFoundError。
        json_out = argv[argv.index('--json-out') + 1]
        Path(json_out).write_text('{"action":"start","effective_band_pct":0.1}')
        return SimpleNamespace(stderr='', returncode=0)

    monkeypatch.setattr(p1_run.subprocess, 'run', fake_run)
    probe = {'command': 'python -m operator_agent preflight --backend ssh --host <HOST> '
                        '--shape 4096,4096,4096 --dtype fp32 --reps 5 --repeat 8'}
    gate = p1_run.run_preflight(probe, host='h', work_dir='/w', gpu_env={}, out_path=out)
    assert gate['action'] == 'start'
    assert out.is_file()