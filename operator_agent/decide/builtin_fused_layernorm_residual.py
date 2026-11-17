"""Versioned native reference primitive for fused residual-add LayerNorm.

Semantics: ``Y = layer_norm(X + R)`` along the last axis, population variance,
``eps=1e-5``, no affine, fp16 in and out. This is the eager sequence the task
declares as its current implementation and performance baseline: one fp16 add
and one native LayerNorm, i.e. two kernels with an fp16 intermediate.

The whole file is fingerprinted by the trusted registry
(``FUSED_LN_RESIDUAL_PRIMITIVE_SHA256``). Editing it without re-pinning that
hash and bumping the profile version is a reference-identity change, not a
refactor.
"""


def native_residual_layernorm(X, R):
    import torch
    import torch.nn.functional as F
    if X.device.type != "cuda" or X.dtype != torch.float16 or X.ndim != 2:
        raise ValueError("fused residual LayerNorm v1 requires rank-two CUDA fp16 X")
    if (R.device != X.device or R.dtype != X.dtype or R.shape != X.shape):
        raise ValueError("fused residual LayerNorm v1 requires R with X's shape/device/dtype")
    return F.layer_norm(X + R, X.shape[-1:], weight=None, bias=None, eps=1e-5)
