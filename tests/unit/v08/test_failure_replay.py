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


def test_the_tail_case_pins_the_specific_failing_shape():
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / 'benchmarks/repro/fused_ln_tail_20261118'
    case = json.loads((root / 'case.json').read_text(encoding='utf-8'))
    source = (root / 'candidate.py').read_text(encoding='utf-8')
    assert fr._sha(source) == case['origin']['source_hash']
    assert case['contract_hash'].startswith('09333958')
    failing = case['expected']['failing_cases']
    assert [row['case_id'] for row in failing] == ['fused-ln-residual-1x33-random-contiguous']
    assert failing[0]['max_error'] == 0.099609375 and failing[0]['matched_ratio'] == 0.1515
    assert case['expected']['passing_cases'] == ['fused-ln-residual-1x1-random-contiguous']
    fixed = (root / 'candidate_fixed.py').read_text(encoding='utf-8')
    assert 'tl.where(mask, s - mean, 0.0)' in fixed and 'idx * r_inner' in fixed


def test_the_runner_matches_case_level_expectations(tmp_path):
    case = _case(tmp_path, expected=dict(
        levels=dict(L1=True, L2=False),
        passing_cases=['ok'], failing_cases=[dict(case_id='bad', verdict='failed',
                                                  max_error=0.1, matched_ratio=0.1515)]))
    rows = [dict(case_id='ok', verdict='passed', metrics=dict(bad_elements=0)),
            dict(case_id='bad', verdict='failed',
                 metrics=dict(max_error=0.1, matched_ratio=0.1515, bad_elements=28))]
    matched, why = fr._matches(case, dict(levels=dict(L1=True, L2=False),
                                          compile_failure={}, cases=rows))
    assert matched and 'named cases' in why

    drifted = [dict(rows[0]), dict(case_id='bad', verdict='failed',
                                   metrics=dict(max_error=0.2, matched_ratio=0.1515))]
    matched, why = fr._matches(case, dict(levels=dict(L1=True, L2=False),
                                          compile_failure={}, cases=drifted))
    assert matched is False and 'max_error' in why


def test_identity_is_part_of_the_pass_criteria(tmp_path):
    """源码哈希与契约哈希不能只当记录字段：回放跑错源码/契约也要判不成立。"""
    case = _case(tmp_path, expected=dict(levels=dict(L1=True, L2=False)))
    case['contract_hash'] = 'c' * 64
    case['origin'] = dict(source_hash='s' * 64)
    observed = dict(levels=dict(L1=True, L2=False), compile_failure={}, cases=[],
                    contract_hash='c' * 64, source_hash='s' * 64)
    matched, why = fr._matches(case, observed, source_name='candidate.py')
    assert matched and 'named cases' in why

    drifted_contract = dict(observed, contract_hash='d' * 64)
    matched, why = fr._matches(case, drifted_contract, source_name='candidate.py')
    assert matched is False and 'contract_hash' in why

    drifted_source = dict(observed, source_hash='x' * 64)
    matched, why = fr._matches(case, drifted_source, source_name='candidate.py')
    assert matched is False and 'source_hash' in why

    # 补丁源码本来就不是那份字节，不按修前哈希判否
    matched, why = fr._matches(case, drifted_source, source_name='candidate_fixed.py')
    assert matched is True


def test_the_matmul_case_pins_the_strided_layout_failure():
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / 'benchmarks/repro/matmul_strided_20261107'
    case = json.loads((root / 'case.json').read_text(encoding='utf-8'))
    source = (root / 'candidate.py').read_text(encoding='utf-8')
    assert fr._sha(source) == case['origin']['source_hash']
    assert case['contract_hash'].startswith('6e337b0f')
    failing = case['expected']['failing_cases'][0]
    assert failing['case_id'] == 'auto-k4-m32-n32-fp16-random-strided'
    assert failing['max_error'] == 9.69140625 and failing['matched_ratio'] == 0.0029296875
    fix = (root / 'historical_fix.py').read_text(encoding='utf-8')
    assert 'sa_m' in fix and 'sb_n' in fix
    # 修前源码里不该出现 stride 形参
    assert 'sa_m' not in source


def test_the_attention_case_pins_all_five_causal_failures():
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / 'benchmarks/repro/attention_causal_20261109'
    case = json.loads((root / 'case.json').read_text(encoding='utf-8'))
    source = (root / 'candidate.py').read_text(encoding='utf-8')
    assert fr._sha(source) == case['origin']['source_hash']
    assert case['contract_hash'].startswith('e9f75d60')
    failing = [row['case_id'] for row in case['expected']['failing_cases']]
    assert len(failing) == 5 and all('causal' in name for name in failing)
    assert case['expected']['passing_cases'] == ['s1024-nocausal']
    assert {row['max_error'] for row in case['expected']['failing_cases']} == {
        0.001953125, 0.0009765625}
