"""Static fusion check used by the A/C qualified-delivery gate."""
import pytest

from benchmarks.experiments.v08.fusion_check import (FusionCheckError, check_fused_source,
                                                     require_fused)

GOOD = '''import triton
import triton.language as tl

@triton.jit
def _kernel(x_ptr, n, BLOCK: tl.constexpr):
    pass

def kernel(X):
    _kernel[(1,)](X, X.numel(), BLOCK=1024)
    return X
'''

TWO_KERNELS = '''import triton

@triton.jit
def _add(x_ptr):
    pass

@triton.jit
def _ln(x_ptr):
    pass

def kernel(X):
    _add[(1,)](X)
    _ln[(1,)](X)
    return X
'''

NO_LAUNCH = '''@triton.jit
def _kernel(x_ptr):
    pass

def kernel(X):
    return X
'''

NO_WRAPPER = '''@triton.jit
def _kernel(x_ptr):
    pass
'''


def test_single_jit_with_one_launch_passes():
    result = check_fused_source(GOOD)
    assert result['passed'] is True
    assert result['jit_kernels'] == ['_kernel']
    assert result['launch_sites'] == 1
    assert require_fused(GOOD) == result


def test_two_triton_kernels_are_not_a_fused_delivery():
    result = check_fused_source(TWO_KERNELS)
    assert result['passed'] is False
    assert 'exactly one' in result['reason']
    with pytest.raises(FusionCheckError):
        require_fused(TWO_KERNELS)


def test_missing_launch_or_wrapper_fails():
    assert check_fused_source(NO_LAUNCH)['passed'] is False
    assert check_fused_source(NO_WRAPPER)['passed'] is False


def test_syntax_error_is_refused_not_guessed():
    with pytest.raises(FusionCheckError):
        check_fused_source('def kernel(:\n')
    with pytest.raises(FusionCheckError):
        check_fused_source('')
