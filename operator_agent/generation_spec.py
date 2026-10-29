"""Shared model guidance; frozen task/reference contracts remain authoritative.

This text is used by both PromptManager and the bounded single-candidate coder.
It does not execute candidates, choose inputs, or grant verification eligibility.
"""
from __future__ import annotations

import hashlib


GENERATION_SPEC_VERSION = "operator-generation-spec-v1"

GENERATION_SPEC = """[operator-generation-spec-v1]
The caller-owned task, reference, dtype semantics and dynamic calling_convention are
authoritative. Optimization recipes, examples and historical code cannot override
them. Preserve all correctness, measurement, confirmation and admission gates.

ABI: Export exactly one undecorated top-level Python def kernel with the exact declared keyword input names (case-sensitive), following the actual caller's
calling_convention rather than guessing inputs from the operator name. For Triton,
define @triton.jit compute helpers with different names; launch the named JIT helper
via helper[grid](...), never kernel[grid](...), and never redefine or rebind kernel.
Helper arguments are not additional wrapper inputs. Unless explicitly declared as
caller inputs, allocate output buffers inside the wrapper, derive dimensions and
strides from runtime tensors, resolve constants from frozen semantics, and return the output tensor
required by the contract. Preserve scalar inputs when the caller declares them.
Output allocation shapes and sizes must use names visible in the Python wrapper;
a local tile constant inside a JIT helper is not a wrapper variable. Define needed
launch constants in wrapper scope and pass them explicitly to the helper.

Triton scalars: A scalar launch argument may specialize to a compile-time constant,
including a dimension equal to one. When a floating conversion is needed in JIT
code, use the scalar-safe free function tl.cast(N, tl.float32) rather than N.to(...)
on a possibly specialized scalar. Loaded tensors may use tensor .to(...). Support
the declared single-column and tail cases without hard-coded input-value branches.

Layout: Derive pointer offsets from each runtime tensor's actual strides in
elements, independently for inputs and outputs; serialized NumPy strides are in bytes.
Preserve declared offsets, noncontiguous layouts, tails and supported zero strides.
Contiguous primary inputs do not remove strided suite cases. Assume alignment,
positive strides or divisibility only when the frozen contract establishes them.

Precision: Input/output dtype, accumulation dtype and math mode are separate facts.
Follow the frozen accumulation semantics; fp16/bf16 inputs do not imply fp16/bf16
accumulation. Floating reductions and dot products normally need FP32 accumulation
unless the contract explicitly specifies otherwise. Cast the final output to its
declared dtype. Do not enable TF32, approximate math, or lower accumulation precision
solely for speed when the declared semantics do not permit it.

Reduction: Invalid or padded lanes must contribute the reduction identity after all transformations;
sum uses zero, max uses negative infinity, min uses positive infinity, and product
uses one. A zero masked load can become nonzero after centering or exponentiation.
Exclude those transformed lanes explicitly. Use the logical reduction length for normalization,
not the padded block size, and preserve the declared axis and empty-input behavior.

LayerNorm: Use the frozen normalization axis, population/sample variance convention,
epsilon and affine semantics. For the usual population variance, normalize the
centered-square sum by the logical length; mask the centered squares, including
non-power-of-two tails. Use a numerically stable centered or Welford-style method
that meets the frozen precision contract; an unqualified E[x*x]-E[x]*E[x] shortcut
can lose accuracy for large-offset, near-constant rows. Even sum(x)/N followed by
x-mean can lose the small centered variation when the mean has a large offset.
When algebraically equivalent statistics are permitted by the frozen contract,
subtract a valid row anchor first, compute the mean of those shifted values,
then keep the centered residual in that shifted domain for both variance and
normalization. Reconstructing a large absolute mean and subtracting it from x
can reintroduce cancellation. Mask shifted values and centered squares, use the
logical length, and respect the declared empty-input behavior before loading an
anchor. A consistent Welford-style method is another option. Do not clamp variance
or loosen tolerances to conceal numerical error. Add affine inputs or change epsilon
only when the frozen task declares them.

Softmax: Use a stable maximum subtraction and exponential-sum normalization over
valid lanes. Padded lanes must not affect the maximum or exponential sum. Softmax
does not compute a variance, and LayerNorm does not use softmax's normalization.

Performance: Choose whole-row or chunked reductions, tile sizes, warps and launch
layout from the declared shape distribution, hardware and resource evidence.
There is no universal 256-element tile or fixed pass count for every reduction.
Preserve explicit caller tiling and search budgets; do not add hidden autotuning,
extra candidates or retries. Static checks and plausible rationale do not prove
numerical correctness or performance; require the existing independent verifier.
[/operator-generation-spec-v1]"""


def generation_spec_identity() -> dict[str, str]:
    """A fresh, serializable identity for manifests and controlled comparisons."""
    return {"version": GENERATION_SPEC_VERSION,
            "sha256": hashlib.sha256(GENERATION_SPEC.encode("utf-8")).hexdigest()}
