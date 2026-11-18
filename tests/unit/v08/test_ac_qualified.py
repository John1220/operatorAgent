"""A/C qualified-delivery classification (fusion + L3 threshold)."""
import json

import pytest

from benchmarks.experiments.v08 import ac_fixture, ac_driver
from benchmarks.experiments.v08.ac_fixture import _finish_cell
from benchmarks.experiments.v08.ac_plan import FUSED_TASK

FUSED = '''import triton
import triton.language as tl

@triton.jit
def _k(x_ptr, r_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pass

def kernel(X, R):
    _k[(1,)](X, R, X, X.numel(), BLOCK=1024)
    return X
'''
TWO = '''import triton

@triton.jit
def _add(x_ptr):
    pass

@triton.jit
def _ln(x_ptr):
    pass

def kernel(X, R):
    _add[(1,)](X)
    _ln[(1,)](X)
    return X
'''


def _body(*, l2=True, validity='valid', speedup=2.0):
    return dict(delivered=l2,
                explore=dict(levels={'L1': True, 'L2': True}),
                final=dict(levels={'L1': True, 'L2': l2, 'L3': validity == 'valid'},
                           verdicts=[
                               dict(level='L2', passed=l2, metrics={}),
                               dict(level='L3', passed=validity == 'valid', metrics=dict(
                                   measurement_validity=validity,
                                   speedup_vs_baseline=speedup))]))


def _run(tmp_path, monkeypatch, *, source, body):
    (tmp_path / 'cell').mkdir(exist_ok=True)
    (tmp_path / 'run').mkdir(exist_ok=True)
    selected = tmp_path / 'candidate.py'
    selected.write_text(source, encoding='utf-8')
    monkeypatch.setattr(ac_driver, 'final_evaluation', lambda *a, **k: body)
    return _finish_cell('t', 'C', selected, root=tmp_path / 'cell',
                        declaration=FUSED_TASK['declaration'],
                        budget=dict(wall_clock_s=2400, max_model_calls=20),
                        backend='fake', dispatches=3, errors=[], returncode=0,
                        connection={})


def test_qualified_delivery_needs_l2_valid_l3_threshold_and_single_kernel(tmp_path, monkeypatch):
    record = _run(tmp_path, monkeypatch, source=FUSED, body=_body())
    assert record['qualified'] is True
    assert record['delivery_class'] == 'qualified'
    assert record['fusion_static']['passed'] is True
    assert record['final_speedup'] >= 1.5
    assert record['final_validity'] == 'valid'


def test_correct_but_slow_or_two_kernel_is_only_correct_only(tmp_path, monkeypatch):
    slow = _run(tmp_path, monkeypatch, source=FUSED, body=_body(speedup=1.2))
    assert slow['qualified'] is False and slow['delivery_class'] == 'correct_only'
    two = _run(tmp_path, monkeypatch, source=TWO, body=_body())
    assert two['qualified'] is False and two['delivery_class'] == 'correct_only'
    assert two['fusion_static']['passed'] is False


def test_l2_failure_is_failed_not_correct_only(tmp_path, monkeypatch):
    record = _run(tmp_path, monkeypatch, source=FUSED, body=_body(l2=False))
    assert record['qualified'] is False and record['delivery_class'] == 'failed'


def test_repair_acceptance_only_requires_full_l2_and_final_l2(tmp_path, monkeypatch):
    """修复模式的判据只看“修好没有”：不比性能、不查融合结构。"""
    (tmp_path / 'cell').mkdir(exist_ok=True)
    (tmp_path / 'run').mkdir(exist_ok=True)
    selected = tmp_path / 'candidate.py'
    selected.write_text(FUSED, encoding='utf-8')
    # 修复格：L3 不存在（stop_after=L2 的终评）也要能判达标
    body = dict(delivered=True, explore=dict(levels={'L1': True, 'L2': True}),
                final=dict(levels={'L1': True, 'L2': True}, verdicts=[
                    dict(level='L2', passed=True, metrics={'correctness': {'cases': [
                        dict(case_id=f'c{i}', verdict='passed') for i in range(32)]}})]))
    monkeypatch.setattr(ac_driver, 'final_evaluation', lambda *a, **k: body)
    record = _finish_cell('t', 'A', selected, root=tmp_path / 'cell',
                          declaration=FUSED_TASK['declaration'],
                          budget=dict(wall_clock_s=2400, max_model_calls=10),
                          backend='fake', dispatches=5, errors=[], returncode=0,
                          connection={}, acceptance='repair')
    assert record['qualified'] is True
    assert record['delivery_class'] == 'qualified'
    assert record['final_validity'] is None   # 修复模式不要求 L3
