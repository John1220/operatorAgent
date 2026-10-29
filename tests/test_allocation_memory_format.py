"""Static allocation compatibility only; no tensor execution or L2 claims."""
import pytest

from operator_agent.run_contract import code_hash
from operator_agent.single_candidate_coder import PrecheckRefused, precheck
from tests.test_single_candidate_coder import CODE


# Exact retained model source from the 2026-10-24 portfolio, never executed here.
HISTORICAL = '''import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(X, Y, n_rows, n_cols, stride_xr, stride_xc, stride_yr, stride_yc, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x_ptrs = X + row * stride_xr + cols * stride_xc
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols

    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    y = centered * rstd

    y_ptrs = Y + row * stride_yr + cols * stride_yc
    tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=mask)


def kernel(X):
    assert X.dim() == 2
    n_rows = X.shape[0]
    n_cols = X.shape[1]

    Y = torch.empty_like(X, memory_format=torch.contiguous_format)

    stride_xr = X.stride(0)
    stride_xc = X.stride(1)
    stride_yr = Y.stride(0)
    stride_yc = Y.stride(1)

    BLOCK = triton.next_power_of_2(n_cols)
    if BLOCK < 1:
        BLOCK = 1

    eps = 1e-5

    _layernorm_kernel[(n_rows,)](X, Y, n_rows, n_cols, stride_xr, stride_xc, stride_yr, stride_yc, eps, BLOCK=BLOCK)
    return Y
'''
HISTORICAL_HASH = '924b790ae5c1a48e19e413131ac179e569768652c1a25e578b5a9bd38bcf1d38'
METADATA = [{'A': {}, 'B': {}}]


def with_body(body):
    return CODE.replace('return torch.empty_like(A)', body)


def test_original_rejected_source_passes_only_static_allocation_precheck_unchanged():
    assert code_hash(HISTORICAL) == HISTORICAL_HASH
    assert precheck(HISTORICAL, [{'X': dict(shape=[9, 17], dtype='float32')}]) is None
    assert code_hash(HISTORICAL) == HISTORICAL_HASH


@pytest.mark.parametrize('allocation', ['empty', 'empty_like'])
@pytest.mark.parametrize('format_name', ['contiguous_format', 'channels_last', 'channels_last_3d'])
def test_direct_format_keyword_supported_on_existing_allocation_apis(allocation, format_name):
    # Rank constraints and numeric correctness belong to the ordinary verifier.
    argument = 'A.shape' if allocation == 'empty' else 'A'
    source = with_body('return torch.'+allocation+'('+argument+', memory_format=torch.'+format_name+')')
    assert precheck(source, METADATA) is None


def test_preserve_format_is_specific_to_empty_like():
    assert precheck(with_body('return torch.empty_like(A, memory_format=torch.preserve_format)'), METADATA) is None
    with pytest.raises(PrecheckRefused) as caught:
        precheck(with_body('return torch.empty(A, memory_format=torch.preserve_format)'), METADATA)
    assert caught.value.reason == 'torch_not_allocation'


@pytest.mark.parametrize('body', [
    'return torch.contiguous_format',
    'fmt = torch.contiguous_format\n    return torch.empty_like(A, memory_format=fmt)',
    'return torch.empty_like(A, memory_format=torch.contiguous_format())',
    'return torch.contiguous_format(A)',
    'return torch.empty_like(A, torch.contiguous_format)',
    'return torch.empty_like(A, dtype=torch.contiguous_format)',
    'return torch.empty_like(A, memory_format=[torch.contiguous_format])',
    'return torch.empty_like(A, memory_format=(torch.contiguous_format,))',
    'return torch.empty_like(A, memory_format=torch.contiguous_format or None)',
    'return helper(memory_format=torch.contiguous_format)',
    'return torch.matmul(A, B, memory_format=torch.contiguous_format)',
    'return torch.zeros_like(A, memory_format=torch.contiguous_format)',
    'alloc = torch.empty_like\n    return alloc(A, memory_format=torch.contiguous_format)',
    'return torch.empty_like(A, **dict(memory_format=torch.contiguous_format))',
    'return torch.empty_like(A, memory_format=torch.contiguous_format.name)',
    'torch.contiguous_format = 1\n    return A',
    'del torch.contiguous_format\n    return A',
    'return torch.empty_like(A, memory_format=torch.invalid_format)',
])
def test_constants_cannot_escape_allocation_keyword_or_become_operations(body):
    with pytest.raises(PrecheckRefused) as caught:
        precheck(with_body(body), METADATA)
    assert caught.value.reason == 'torch_not_allocation'


@pytest.mark.parametrize('prefix', [
    'torch = A',
    't = torch',
    'helper(torch)',
    'torch.empty_like = helper',
    'torch.empty_like.rebound = helper',
    'for torch in []:\n        pass',
    'def torch():\n        pass',
    'alias = lambda torch: torch',
    'try:\n        pass\n    except Exception as torch:\n        pass',
])
def test_newly_admitted_values_require_unrebound_unescaped_module(prefix):
    body = prefix+'\n    return torch.empty_like(A, memory_format=torch.contiguous_format)'
    with pytest.raises(PrecheckRefused) as caught:
        precheck(with_body(body), METADATA)
    assert caught.value.reason == 'torch_not_allocation'


@pytest.mark.parametrize('mutation', ['missing_import', 'rebound_import', 'argument_shadow'])
def test_spelling_alone_is_not_a_trusted_format_constant(mutation):
    source = with_body('return torch.empty_like(A, memory_format=torch.contiguous_format)')
    if mutation == 'missing_import':
        source = source.replace('import torch\n', '')
    elif mutation == 'rebound_import':
        source = source.replace('import triton.language as tl', 'import triton.language as tl\nfrom triton.language import sqrt as torch')
    else:
        source = source.replace('def kernel(A, B):', 'def kernel(A, B, torch):')
    with pytest.raises(PrecheckRefused) as caught:
        precheck(source, METADATA)
    assert caught.value.reason == 'torch_not_allocation'


@pytest.mark.parametrize('mutation,reason', [
    ('compute', 'torch_not_allocation'),
    ('decorator', 'kernel_entry_decorated'),
    ('jit', 'jit_helper_missing'),
    ('interface', 'kernel_interface_refused'),
    ('return', 'kernel_return_missing'),
])
def test_format_compatibility_keeps_existing_wrapper_and_computation_guards(mutation, reason):
    source = HISTORICAL
    if mutation == 'compute':
        source = source.replace('return Y', 'return torch.softmax(X, dim=1)')
    elif mutation == 'decorator':
        source = source.replace('def kernel(X):', '@triton.jit\ndef kernel(X):')
    elif mutation == 'jit':
        source = source.replace('@triton.jit\n', '')
    elif mutation == 'interface':
        source = source.replace('def kernel(X):', 'def kernel(X, Y):')
    else:
        source = source.replace('return Y', 'return None')
    with pytest.raises(PrecheckRefused) as caught:
        precheck(source, [{'X': {}}])
    assert caught.value.reason == reason
