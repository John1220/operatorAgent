"""Versioned native reference primitive, not candidate-visible customization.

This whole source file is fingerprinted by the trusted registry. No dtype
promotion or CPU fallback is allowed for the CUDA fp16 profile.
"""


def native_layernorm(X):
    import torch
    import torch.nn.functional as F
    if X.device.type != "cuda" or X.dtype != torch.float16 or X.ndim != 2:
        raise ValueError("native LayerNorm v2 requires rank-two CUDA fp16 input")
    return F.layer_norm(X, X.shape[-1:], weight=None, bias=None, eps=1e-5)
