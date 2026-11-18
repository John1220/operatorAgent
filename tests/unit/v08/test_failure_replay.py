"""Failure-replay matching rules (no GPU)."""
import json

import pytest

from benchmarks.experiments.v08 import failure_replay as fr


def _case(tmp_path, expected=None):
    case = dict(schema='v08-failure-replay-case-v1', case_id='c',
                declaration=dict(request='fp16 8x8 relu', shapes=[[8, 8]],
                                 precision='fp16', dsl='triton', hardware='nvidia_gpu'),
                expected=expected or dict(levels=dict(L1=True, L2=False),
                                          producer='triton.compiler.errors.CompilationError',
                                          line=30, column=11))
    (tmp_path / 'case.json').write_text(json.dumps(case), encoding='utf-8')
    return case


def test_a_matching_failure_is_a_successful_replay(tmp_path):
    case = _case(tmp_path)
    observed = dict(levels=dict(L1=True, L2=False),
                    compile_failure=dict(producer='triton.compiler.errors.CompilationError',
                                         line=30, column=11))
    matched, why = fr._matches(case, observed)
    assert matched and 'matches' in why


def test_a_different_line_or_producer_is_not_the_same_failure(tmp_path):
    case = _case(tmp_path)
    base = dict(producer='triton.compiler.errors.CompilationError', line=30, column=11)
    for change in (dict(line=31), dict(column=1), dict(producer='SyntaxError')):
        observed = dict(levels=dict(L1=True, L2=False),
                        compile_failure=dict(base, **change))
        matched, why = fr._matches(case, observed)
        assert matched is False and 'expected' in why


def test_levels_must_match_too(tmp_path):
    case = _case(tmp_path)
    observed = dict(levels=dict(L1=True, L2=True), compile_failure={})
    matched, why = fr._matches(case, observed)
    assert matched is False and 'L2' in why


def test_the_case_manifest_pins_the_contract_and_the_original_source_hash():
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / 'benchmarks/repro/fused_ln_compile_20261118'
    case = json.loads((root / 'case.json').read_text(encoding='utf-8'))
    source = (root / 'candidate.py').read_text(encoding='utf-8')
    assert fr._sha(source) == case['origin']['source_hash']
    assert case['contract_hash'].startswith('09333958')
    assert case['expected']['levels'] == {'L1': True, 'L2': False}
    fixed = (root / 'candidate_fixed.py').read_text(encoding='utf-8')
    assert 'next_power_of_2' in fixed and 'tl.where(cols < N, s - mean, 0.0)' in fixed
