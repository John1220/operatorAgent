"""D2 确定性改写器标定：差分逐字、核体不动、模式不符即拒。"""
import ast
import json
from pathlib import Path

import pytest

from benchmarks.experiments.v08 import d2_rewrite
from operator_agent.core.run_contract import code_hash

SEED_JSON = Path('output/v08-p2-seed7-medium-20261116/seed.json')


@pytest.fixture(scope='module')
def seed_code():
    manifest = json.loads(SEED_JSON.read_text(encoding='utf-8'))
    code = Path(manifest['source']).read_text(encoding='utf-8')
    assert code_hash(code) == manifest['source_hash']
    return code


def _jit_dump(source):
    return sorted(ast.dump(node) for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef)
                  and any(isinstance(d, ast.Attribute) and d.attr == 'jit'
                          for d in node.decorator_list))


def test_rewrite_changes_only_the_two_launch_literals(seed_code):
    out = d2_rewrite.rewrite(seed_code, block_size=1024, num_warps=8)
    assert 'BLOCK_SIZE = 1024' in out and 'num_warps=8' in out
    assert 'BLOCK_SIZE = 512' not in out and 'num_warps=4' not in out
    old, new = seed_code.splitlines(), out.splitlines()
    assert len(old) == len(new)
    changed = [(a, b) for a, b in zip(old, new) if a != b]
    assert len(changed) == 2
    for a, b in changed:
        assert ''.join(ch for ch in a if not ch.isdigit()) == \
            ''.join(ch for ch in b if not ch.isdigit())
    assert _jit_dump(seed_code) == _jit_dump(out)
    assert d2_rewrite.resolve_launch(out) == d2_rewrite.LaunchKnobs(1024, 8)


def test_seed_point_rewrite_is_byte_identical(seed_code):
    assert d2_rewrite.rewrite(seed_code, block_size=512, num_warps=4) == seed_code


def test_resolve_reads_the_frozen_knobs(seed_code):
    assert d2_rewrite.resolve_launch(seed_code) == d2_rewrite.LaunchKnobs(512, 4)


def test_grid_enumeration_is_deterministic_and_distinct(seed_code):
    points = d2_rewrite.grid_points((256, 512, 1024, 2048), (2, 4, 8, 16))
    assert len(points) == 16
    assert points[0] == dict(block_size=256, num_warps=2)
    assert points[-1] == dict(block_size=2048, num_warps=16)
    assert points[5] == dict(block_size=512, num_warps=4)
    hashes = {code_hash(d2_rewrite.rewrite(seed_code, **params)) for params in points}
    assert len(hashes) == 16
    assert code_hash(seed_code) in hashes


@pytest.mark.parametrize('kwargs', [
    dict(block_size=500, num_warps=4),
    dict(block_size=512, num_warps=3),
    dict(block_size=0, num_warps=4),
    dict(block_size=512, num_warps=True),
])
def test_rewrite_refuses_invalid_targets(seed_code, kwargs):
    with pytest.raises(d2_rewrite.RewriteError):
        d2_rewrite.rewrite(seed_code, **kwargs)


def test_rewrite_refuses_source_without_launch():
    with pytest.raises(d2_rewrite.RewriteError, match='launch'):
        d2_rewrite.rewrite('def kernel(X):\n    return X\n', block_size=512, num_warps=4)


def test_rewrite_refuses_two_launches(seed_code):
    lines = seed_code.splitlines(keepends=True)
    launch = next(i for i, line in enumerate(lines) if '_layernorm_helper[grid]' in line)
    doubled = ''.join(lines[:launch + 4]) + ''.join(lines[launch:])
    with pytest.raises(d2_rewrite.RewriteError, match='恰有一个'):
        d2_rewrite.rewrite(doubled, block_size=1024, num_warps=8)


def test_rewrite_refuses_non_constant_num_warps(seed_code):
    warped = seed_code.replace('num_warps=4', 'num_warps=w')
    with pytest.raises(d2_rewrite.RewriteError):
        d2_rewrite.rewrite(warped, block_size=1024, num_warps=8)


def test_rewrite_refuses_ambiguous_block_size_assignment(seed_code):
    warped = seed_code.replace('BLOCK_SIZE = 512', 'BLOCK_SIZE = 512\n    BLOCK_SIZE = 512')
    with pytest.raises(d2_rewrite.RewriteError, match='恰有一处'):
        d2_rewrite.rewrite(warped, block_size=1024, num_warps=8)