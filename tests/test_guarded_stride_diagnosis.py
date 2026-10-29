"""Address evidence for a retained rank-two identity guard; no kernel execution."""
import ast
from pathlib import Path

import pytest

from operator_agent.repair_stride_diagnoser import (
    GUARDED_RECOGNIZER,
    guarded_column_stride_sites,
    missing_column_stride_sites,
)
from operator_agent.run_contract import code_hash


SOURCE = (Path(__file__).parent / 'fixtures' / 'layernorm_guarded_missing_column_stride.txt').read_text()


def sites(source=SOURCE, *, rank=2):
    return guarded_column_stride_sites(source, ast.parse(source), input_rank=rank)


def test_retained_source_requires_explicit_rank_evidence_and_returns_bound_original_sites():
    assert GUARDED_RECOGNIZER == 'rank-two-guarded-anchor-missing-input-column-stride-v1'
    assert code_hash(SOURCE) == 'e46c35695f78516ac1aa2fefa77ee19268bbbb7c8ccbfbc1ab9a3e623cab34e4'
    tree = ast.parse(SOURCE)
    original = ast.dump(tree, include_attributes=True)
    found = guarded_column_stride_sites(SOURCE, tree, input_rank=2)
    assert ast.dump(tree, include_attributes=True) == original
    assert found
    roles = {n['role']: n for n in found}
    assert {'input_rank_guard', 'input_rows', 'input_columns', 'output_allocation',
            'helper_launch', 'input_row_stride', 'row_index', 'column_index',
            'input_load', 'input_row_anchor', 'output_store'} == set(roles)
    assert roles['input_load']['excerpt'] == 'x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)'
    assert roles['input_row_anchor']['excerpt'] == 'anchor = tl.load(x_ptr)'
    for site in found:
        assert site['trust'] == 'untrusted_source_data'
        assert site['excerpt_sha256'] == code_hash(site['excerpt'])
        # All source evidence names the original statement rather than a rewritten AST.
        lines = SOURCE.splitlines(keepends=True)
        excerpt = ''.join(lines[site['line'] - 1:site['end_line']])
        assert site['excerpt'] in excerpt
    # The old grammar deliberately remains narrower.
    assert missing_column_stride_sites(SOURCE, ast.parse(SOURCE)) is None
    with pytest.raises(TypeError, match='input_rank'):
        guarded_column_stride_sites(SOURCE, ast.parse(SOURCE))


@pytest.mark.parametrize('rank', [None, True, False, 0, 1, 3, '2', 2.0, [2]])
def test_unknown_or_non_two_rank_never_allows_assuming_identity_branch(rank):
    assert sites(rank=rank) is None


@pytest.mark.parametrize('source', [
    SOURCE.replace('    if BLOCK < 1:\n        BLOCK = 1\n', ''),
    SOURCE.replace('anchor = tl.load(x_ptr)\n    anchor = tl.cast(anchor, tl.float32)',
                   'anchor = tl.cast(tl.load(x_ptr), tl.float32)'),
    SOURCE.replace('anchor = tl.load(x_ptr)\n    anchor = tl.cast(anchor, tl.float32)',
                   'anchor = tl.load(x_ptr).to(tl.float32)'),
    SOURCE.replace('stride_x', 'input_row_step').replace('cols', 'logical_column')
          .replace('x_ptr', 'original_input_row').replace('anchor', 'row_origin'),
    SOURCE.replace('x_ptr + cols,', 'cols + x_ptr,'),
], ids=['no_block_floor', 'inline_cast', 'inline_to', 'renamed_locals', 'commuted_addition'])
def test_equivalent_narrow_source_forms_preserve_bound_address_evidence(source):
    assert sites(source)


@pytest.mark.parametrize('old,new', [
    ('X = X.contiguous() if X.dim() != 2 else X', 'X = X.contiguous()'),
    ('X.contiguous() if X.dim() != 2 else X', 'X.clone() if X.dim() != 2 else X'),
    ('X.contiguous() if X.dim() != 2 else X', 'X if X.dim() != 2 else X.contiguous()'),
    ('X.contiguous() if X.dim() != 2 else X', 'X.contiguous() if X.dim() == 2 else X'),
    ('X.contiguous() if X.dim() != 2 else X', 'X.contiguous() if X.dim() != 3 else X'),
    ('X.contiguous() if X.dim() != 2 else X', 'X.contiguous() if X.ndim != 2 else X'),
    ('X.contiguous() if X.dim() != 2 else X', 'X.contiguous() if X.is_contiguous() else X'),
    ('X.contiguous() if X.dim() != 2 else X', 'X.contiguous(memory_format=torch.contiguous_format) if X.dim() != 2 else X'),
    ('X.contiguous() if X.dim() != 2 else X', 'X.contiguous() if X.dim() != 2 else X.clone()'),
    ('rows = X.shape[0]', 'rows = X.shape[1]'),
    ('N = X.shape[1]', 'N = X.shape[-1]'),
    ('torch.empty((rows, N), dtype=X.dtype, device=X.device)', 'torch.empty((N, rows), dtype=X.dtype, device=X.device)'),
    ('torch.empty((rows, N), dtype=X.dtype, device=X.device)', 'torch.empty((rows, N), dtype=torch.float32, device=X.device)'),
    ('torch.empty((rows, N), dtype=X.dtype, device=X.device)', 'torch.empty_like(X)'),
    ('if BLOCK < 1:', 'if BLOCK <= 1:'),
    ('BLOCK = 1\n', 'BLOCK = 2\n'),
    ('X.stride(0), Y.stride(0)', 'X.stride(1), Y.stride(0)'),
    ('X.stride(0), Y.stride(0)', 'Y.stride(0), Y.stride(0)'),
    ('X.stride(0), Y.stride(0)', 'X.shape[1], Y.stride(0)'),
    ('x_ptr + cols,', 'x_ptr + cols * 2,'),
    ('x_ptr + cols,', 'x_ptr + cols * stride_x,'),
    ('x_ptr + cols,', 'x_ptr + cols * X.stride(1),'),
    ('x_ptr = X + row * stride_x', 'x_ptr = Y + row * stride_x'),
    ('x_ptr = X + row * stride_x', 'x_ptr = X + row * stride_y'),
    ('anchor = tl.load(x_ptr)', 'anchor = tl.load(x_ptr + cols)'),
    ('anchor = tl.load(x_ptr)', 'anchor = tl.load(x_ptr, other=0)'),
    ('anchor = tl.cast(anchor, tl.float32)', 'anchor = tl.cast(anchor, tl.float16)'),
    ('anchor = tl.cast(anchor, tl.float32)', 'anchor = anchor + 1'),
    ('shifted = tl.where(mask, x - anchor, 0.0)', 'shifted = tl.where(mask, x, 0.0)'),
    ('shifted = tl.where(mask, x - anchor, 0.0)', 'shifted = tl.where(mask, anchor, 0.0)'),
    ('x = tl.load', 'x = getattr(tl, "load")'),
    ('y = centered * rstd', 'y = dynamic_function(centered)'),
    ('mask = cols < N', 'mask = cols <= N'),
    ('other=0.0', 'other=1.0'),
    ('y_ptr + cols,', 'x_ptr + cols,'),
    ('tl.store(y_ptr + cols, y.to(tl.float32), mask=mask)', 'tl.store(y_ptr + cols, y.to(tl.float32), mask=cols < N)'),
    ('_layernorm_kernel[(rows,)]', '_layernorm_kernel[(N,)]'),
    ('return Y', 'return X'),
], ids=['real_copy', 'clone_branch', 'copy_on_rank_two', 'equality_guard', 'different_rank',
        'different_rank_api', 'layout_guard', 'contiguous_options', 'copied_identity_arm',
        'wrong_rows', 'negative_column_index', 'transposed_allocation', 'different_dtype',
        'different_allocation', 'different_floor_condition', 'different_floor_value',
        'column_as_row_stride', 'output_as_input_row_stride', 'shape_as_stride',
        'explicit_column_factor', 'wrong_stride_factor', 'corrected_column_stride',
        'output_input_pointer', 'wrong_helper_row_stride', 'vector_anchor', 'anchor_keywords',
        'anchor_downcast', 'anchor_reassignment', 'unused_anchor', 'unused_main_load',
        'dynamic_load', 'dynamic_call', 'wrong_mask', 'wrong_padding', 'input_store',
        'different_store_mask', 'wrong_grid', 'wrong_output'])
def test_copies_mutations_noncanonical_guards_and_unsupported_addresses_stay_unknown(old, new):
    assert old in SOURCE
    assert sites(SOURCE.replace(old, new)) is None


@pytest.mark.parametrize('prefix,statement', [
    ('    X = X.contiguous()', '    X = X.clone()\n'),
    ('    rows = X.shape[0]', '    X[0] = 0\n'),
    ('    x_ptr = X + row * stride_x', '    X = Y\n'),
    ('    x = tl.load', '    alias = X\n    alias[0] = 0\n'),
    ('    x = tl.load', '    tl.store(X, 0)\n'),
    ('    y = centered * rstd', '    if N > 0:\n        ignored = 1\n'),
    ('    y = centered * rstd', '    hidden = X.copy_()\n'),
    ('    y = centered * rstd', '    hidden = tl.load(X)\n'),
    ('    y = centered * rstd', '    anchor = tl.cast(anchor, tl.float32)\n'),
])
def test_additional_source_effects_and_control_flow_do_not_extend_the_grammar(prefix, statement):
    assert sites(SOURCE.replace(prefix, statement + prefix)) is None
