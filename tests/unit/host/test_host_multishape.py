"""Host verification evaluates every declared shape and applies the workload gate."""
import json
from pathlib import Path

from operator_agent.app import hitl_tools as ht
from operator_agent.decide.eligibility import performance_verdict


SHAPES = [[4, 8], [8, 16], [16, 32]]
CODE = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'


def test_host_verify_materializes_every_declared_shape(tmp_path):
    run = tmp_path/'run'
    ht.prepare(dict(operator='relu', request='relu fp32', shapes=SHAPES,
                    precision='fp32'), run)
    result = ht.verify(run, CODE, backend='mock', stop_after='L3')
    full = json.loads(Path(result['evidence']).read_text())
    l3 = next(row for row in full['verdicts'] if row['level'] == 'L3')
    assert set(l3['metrics']['cases']) == {'X=4x8', 'X=8x16', 'X=16x32'}
    assert l3['metrics']['workload_qualification']['scope_cases'] == [
        'X=16x32', 'X=4x8', 'X=8x16']


def test_large_shape_gain_cannot_hide_a_small_shape_regression():
    passed, basis, reason, value = performance_verdict(
        1.5, aggregate='weighted_geomean',
        case_weights={'X=16x32': 3.0, 'X=4x8': 1.0},
        case_speedups={'X=16x32': 2.0, 'X=4x8': 0.8},
        max_regression={'X=4x8': 0.05}, regression_over_limit_value=True)
    assert passed is False and basis == 'objective'
    assert '回退优先于聚合' in reason
    assert value > 1.5
