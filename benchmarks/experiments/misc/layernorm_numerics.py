"""Trusted, diagnostic-only LayerNorm calibration; never an L2 reference swap.

Works as a sibling module in an isolated remote runner. Inputs have ALREADY been
quantized to fp16. Keep unrounded fp64 truth separate from output-rounded truth:
half output quantization is not evidence of a faulty fp32 accumulation.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from operator_agent.core import tolerance
else:  # Never pick a stale installed package over the SHA-checked siblings.
    import tolerance

EPSILON = 1e-5
SEMANTICS = dict(operator="layernorm", normalized_axis=-1, affine=False,
                 epsilon=EPSILON, variance="population", input_dtype="float16",
                 output_dtype="float16", candidate_accum="explicit_fp32",
                 eager_accum="implementation_defined_not_attested")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def compare(actual, expected, max_error=1e-5):
    """Production fp32 AND gate, unchanged thresholds; diagnostics only."""
    a, e = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    if a.shape != e.shape or not a.size:
        raise ValueError("calibration requires equal, nonempty output shapes")
    atol, rtol, rho = tolerance.effective_atol("fp16", max_error)
    finite = bool(np.isfinite(a).all() and np.isfinite(e).all())
    if not finite:
        return dict(passed=False, finite=False, bad_elements=None, max_abs=None,
                    matched_ratio=None, atol=atol, rtol=rtol, rho=rho)
    error = np.abs(a - e)
    bad = (error > atol) & (error / (np.abs(e) + tolerance.EPS) > rtol)
    n_bad = int(bad.sum())
    passed, ratio = tolerance.evaluate(n_bad, a.size, False, rho)
    return dict(passed=passed, finite=True, bad_elements=n_bad,
                max_abs=float(error.max()), matched_ratio=ratio,
                atol=atol, rtol=rtol, rho=rho)


def centered(value, dtype):
    x = np.asarray(value, dtype=dtype)
    delta = x - x.mean(axis=-1, keepdims=True)
    variance = (delta * delta).mean(axis=-1, keepdims=True)
    return delta / np.sqrt(variance + dtype(EPSILON))


def measure(inputs, reference, device, provenance):
    """Observe the actual production closure and independent diagnostic arms.

    The reference closure is supplied by Verifier / run_kernel._reference_torch,
    not invented here. Promoting input changes the implementation path and is
    recorded as a DIFFERENT arm, never used to rejudge production L2.
    """
    import torch
    import torch.nn.functional as F
    if __package__:
        from operator_agent.decide.remote_input_fidelity import to_torch_inputs, torch_input_readback
    else:
        from remote_input_fidelity import to_torch_inputs, torch_input_readback

    if set(inputs) != {"X"}:
        raise ValueError("calibration only supports X, no affine/custom epsilon")
    x_np = inputs["X"]
    if x_np.dtype != np.float16 or x_np.ndim != 2 or not x_np.size:
        raise ValueError("calibration requires nonempty rank-two fp16 input")
    values = to_torch_inputs(inputs, device)
    x = values["X"]
    if str(x.device) != str(torch.device(device)):
        raise ValueError("requested calibration device was not used")

    def array(y):
        if isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy()
        return np.asarray(y)

    with torch.no_grad():
        native = array(reference())
        repeated = array(reference())
        view = array(F.layer_norm(x, x.shape[-1:], eps=EPSILON))
        promoted = array(F.layer_norm(x.float(), x.shape[-1:], eps=EPSILON).half())
        eager64 = array(F.layer_norm(x.double(), x.shape[-1:], eps=EPSILON))
        cpu_native = None
        if x.is_cuda:
            # Eliminate the cross-version confound: same installed Torch,
            # different backend. This is not substituted for CUDA production.
            cpu_x = to_torch_inputs(inputs, "cpu")["X"]
            cpu_native = array(F.layer_norm(cpu_x, cpu_x.shape[-1:], eps=EPSILON))
    truth = centered(x_np, np.float64)
    rounded = truth.astype(np.float16)
    fp32 = centered(x_np, np.float32).astype(np.float16)
    arms = dict(production_eager=native, preserved_view_eager=view,
                promoted_eager_fp32=promoted, centered_fp32=fp32)
    if cpu_native is not None:
        arms["same_runtime_cpu_eager"] = cpu_native
    identity = dict(semantics=SEMANTICS, torch=str(torch.__version__),
                    numpy=np.__version__,
                    cuda_runtime=torch.version.cuda, device=str(x.device),
                    device_name=torch.cuda.get_device_name(x.device) if x.is_cuda else "CPU",
                    entrypoint="torch.nn.functional.layer_norm", provenance=provenance,
                    cpu_capability=torch.backends.cpu.get_cpu_capability(),
                    cpu_threads=torch.get_num_threads(),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    shared_source_hashes={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                        for name, path in (
                            ("layernorm_numerics.py", __file__),
                            ("core/tolerance.py", tolerance.__file__),
                            ("decide/remote_input_fidelity.py", sys.modules[to_torch_inputs.__module__].__file__),
                            ("decide/input_fidelity.py", to_torch_inputs.__globals__["fidelity"].__file__))})
    out = dict(reference_identity=identity, reference_identity_hash=digest(identity),
               input_fidelity=torch_input_readback(values),
               shape=list(x_np.shape), reference_output_dtype=str(native.dtype),
               reference_repeat_exact=bool(np.array_equal(native, repeated)),
               oracle=dict(kind="independent_centered_numpy_fp64_on_quantized_input",
                           unrounded_dtype="float64", rounded_dtype="float16",
                           eager64_max_abs=float(np.abs(eager64 - truth).max()),
                           quantization_max_abs=float(np.abs(rounded.astype(np.float64)-truth).max())),
               arms={})
    for name, y in arms.items():
        out["arms"][name] = dict(output_dtype=str(y.dtype),
            device="cpu" if name in ("same_runtime_cpu_eager", "centered_fp32") else str(x.device),
            vs_production=compare(y, native), vs_rounded_oracle=compare(y, rounded),
            max_abs_to_unrounded=float(np.abs(y.astype(np.float64) - truth).max()),
            output_min=float(y.min()), output_max=float(y.max()))
    return out


def remote_probe(payload, run_kernel):
    """Harness-only dispatch; never installed in production action registry."""
    if __package__:
        from operator_agent.decide.remote_input_fidelity import decode_json_inputs
    else:
        from remote_input_fidelity import decode_json_inputs
    inputs = decode_json_inputs(payload["inputs"])
    reference, kind = run_kernel._reference_torch("layernorm", inputs)
    if kind != "pytorch_eager" or reference is None:
        raise ValueError("actual production LayerNorm reference missing")
    source = Path(run_kernel.__file__).read_bytes()
    result = measure(inputs, reference, "cuda:0", dict(
        factory="run_kernel._reference_torch", source_sha256=hashlib.sha256(source).hexdigest()))
    return dict(success=True, measurement=result)
