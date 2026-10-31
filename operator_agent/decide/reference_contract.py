"""Trusted reference definition + execution identity, shared with CUDA runner.

Identity is checked BEFORE candidate loading and again before reference use.
This detects honest drift, not arbitrary same-uid hostile in-memory rewriting.
Native accumulation is implementation-pinned, NOT inferred from output dtype.
"""
from dataclasses import asdict, dataclass, fields
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess

PROFILE_V2 = "builtin-layernorm-fp16-cuda-v2"
# Filled from isolated read-only attestation, never discovered/accepted from a
# candidate or auto-updated after a numerical failure.
PINNED_RUNTIME = dict(
    torch_version="2.14.0+cu130", torch_git="08187d9e0fba026dc8217405802ab5381dc88d90", cuda_runtime="13.0",
    device="cuda:0", device_name="NVIDIA H20", compute_capability="9.0",
    driver_version="580.105.08",
    torch_config_sha256="41c1a5f5d2721940d25beeaf04aa6b47f1c0ce25788158609496d0ee2b63fe8e",
    functional_sha256="95ff403085bb179477a01df63acfddf59dfac0fb58829e99a9c8b5c2fe98899b",
    layer_norm_callable_sha256="b1034c3890e9ce8671a772ec9995b79833f1e0d90ffde7e3fdc893abd5ca464f",
    primitive_sha256="a4d7536fab09938c84b18c1390aba0025274cf36fb075a7babd5c5e16e06e5a0",
    native_libraries=(
        ("libtorch_cpu.so", "3b720eed2395ae1903cb6cf56fe3defd5776997c09adf48f3f716cb272c22356"),
        ("libtorch_cuda.so", "9ed4b912d08ae0a59c1638e9077b6d8f0439d7bd680e4a1a274441d42dd7b3f9"),
        ("libtorch_python.so", "a5dfbc797469f387b004e432e0543e1006e729608ad9a2c8a6e50e4ae98b5b2f")),
    deterministic_algorithms=False,
)


class ReferenceIdentityError(ValueError):
    pass


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ExecutionIdentity:
    torch_version: str
    torch_git: str
    cuda_runtime: str
    device: str
    device_name: str
    compute_capability: str
    driver_version: str
    torch_config_sha256: str
    functional_sha256: str
    layer_norm_callable_sha256: str
    primitive_sha256: str
    native_libraries: tuple
    deterministic_algorithms: bool

    def content_dict(self):
        d = asdict(self)
        d["native_libraries"] = [list(p) for p in self.native_libraries]
        return d


@dataclass(frozen=True)
class ReferenceContract:
    execution: ExecutionIdentity
    profile: str = PROFILE_V2
    definition: str = "native-pytorch-layernorm-last-axis-no-affine-v2"
    input_dtype: str = "fp16"
    candidate_accum: str = "fp32_declared_not_arithmetic_attestation"
    output_dtype: str = "fp16"
    reference_accum: str = "native_implementation_pinned"
    epsilon: float = 1e-5
    normalized_axis: int = -1
    variance: str = "population"
    affine: bool = False
    fp_fusion: str = "allowed_subject_to_unchanged_tolerance"
    deterministic_outputs: bool = True
    comparison: str = "fp32_and"
    rtol: float = .01
    required_matched_ratio: float = 1.0

    def content_dict(self):
        d = asdict(self)
        d["execution"] = self.execution.content_dict()
        return d

    def digest(self):
        return content_hash(self.content_dict())


def registered_reference():
    return ReferenceContract(ExecutionIdentity(**PINNED_RUNTIME))


def _strict_fields(raw, cls):
    expected = {f.name for f in fields(cls)}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ReferenceIdentityError(f"{cls.__name__} fields missing/unknown")


def as_reference(raw):
    if isinstance(raw, ReferenceContract):
        raw = raw.content_dict()
    _strict_fields(raw, ReferenceContract)
    d = dict(raw)
    _strict_fields(d["execution"], ExecutionIdentity)
    identity = dict(d["execution"])
    libraries = identity["native_libraries"]
    if (not isinstance(libraries, (tuple, list)) or len(libraries) != 3
            or any(not isinstance(p, (tuple, list)) or len(p) != 2 for p in libraries)):
        raise ReferenceIdentityError("native library manifest incomplete")
    identity["native_libraries"] = tuple(sorted(tuple(p) for p in libraries))
    if {p[0] for p in identity["native_libraries"]} != {
            "libtorch_cpu.so", "libtorch_cuda.so", "libtorch_python.so"}:
        raise ReferenceIdentityError("native library manifest drifted")
    for k, v in identity.items():
        if k == "native_libraries":
            if any(type(v) is not str or not re.fullmatch(r"[0-9a-f]{64}", v) for _, v in identity[k]):
                raise ReferenceIdentityError("native library fingerprint invalid")
        elif k == "deterministic_algorithms":
            if type(v) is not bool:
                raise ReferenceIdentityError("determinism flag must be bool")
        elif type(v) is not str or not v:
            raise ReferenceIdentityError(f"execution.{k} missing")
        elif k.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ReferenceIdentityError(f"execution.{k} fingerprint invalid")
    if (type(d["affine"]) is not bool or type(d["deterministic_outputs"]) is not bool
            or type(d["normalized_axis"]) is not int
            or any(type(d[k]) is not float for k in ("epsilon", "rtol", "required_matched_ratio"))
            or any(type(d[k]) is not str for k in
                   ("profile", "definition", "input_dtype", "candidate_accum", "output_dtype",
                    "reference_accum", "variance", "fp_fusion", "comparison"))):
        raise ReferenceIdentityError("semantic types cannot be coerced")
    d["execution"] = ExecutionIdentity(**identity)
    spec = ReferenceContract(**d)
    if spec != registered_reference():
        raise ReferenceIdentityError("reference definition/execution not in trusted v2 registry")
    return spec


# Cache only while file identity/stat agrees. Recheck on every request; never
# cache an arbitrary probe's success. Stat checks are NOT hostile-uid security.
_FILE_HASHES = {}


def _file_hash(path):
    path = Path(path)
    before = path.stat()
    stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    cached = _FILE_HASHES.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if stamp != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ReferenceIdentityError("reference file changed during fingerprinting")
    _FILE_HASHES[str(path)] = (stamp, h.hexdigest())
    return h.hexdigest()


def observe_execution():
    import torch
    import torch.nn.functional as F
    if __package__:
        from ..decide import builtin_layernorm
    else:
        import builtin_layernorm
    if not torch.cuda.is_available():
        raise ReferenceIdentityError("no CUDA reference; CPU fallback forbidden")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    drivers = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        text=True, timeout=5).strip().splitlines()
    if not drivers or len(set(drivers)) != 1:
        raise ReferenceIdentityError("driver identity unavailable/ambiguous")
    if (F.layer_norm.__module__ != "torch.nn.functional"
            or F.layer_norm.__qualname__ != "layer_norm"):
        raise ReferenceIdentityError("native layer_norm callable replaced")
    lib = Path(torch.__file__).parent / "lib"
    return ExecutionIdentity(
        torch_version=str(torch.__version__), torch_git=str(torch.version.git_version),
        cuda_runtime=str(torch.version.cuda), device=f"cuda:{device}", device_name=props.name,
        compute_capability=f"{props.major}.{props.minor}", driver_version=drivers[0],
        torch_config_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        functional_sha256=_file_hash(F.__file__), primitive_sha256=_file_hash(builtin_layernorm.__file__),
        layer_norm_callable_sha256=hashlib.sha256(inspect.getsource(F.layer_norm).encode()).hexdigest(),
        native_libraries=tuple((n, _file_hash(lib / n)) for n in
            ("libtorch_cpu.so", "libtorch_cuda.so", "libtorch_python.so")),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled()).content_dict()


def check_observed(spec, observed):
    spec = as_reference(spec)
    expected = spec.execution.content_dict()
    if (not isinstance(observed, dict) or observed != expected
            or type(observed.get("deterministic_algorithms")) is not bool):
        changed = sorted(k for k in expected if not isinstance(observed, dict) or observed.get(k) != expected[k])
        raise ReferenceIdentityError("execution identity mismatch: " + ",".join(changed))
    return dict(status="passed", reference_hash=spec.digest(), observed=observed,
                observed_hash=content_hash(observed))


def execution_receipt(raw):
    try:
        return check_observed(raw, observe_execution())
    except ReferenceIdentityError:
        raise
    except Exception as exc:
        raise ReferenceIdentityError("reference identity probe unavailable: " + type(exc).__name__) from exc


def validate_gate(raw, precision, max_error, reference_source="", kernel_type="layernorm"):
    spec = as_reference(raw)
    if (precision != "fp16" or kernel_type != "layernorm" or reference_source
            or type(max_error) not in (int, float) or not 0 < max_error <= 1e-5):
        raise ReferenceIdentityError("v2 operator/dtype/source/tolerance mismatch")
    if __package__:
        from ..core import tolerance
    else:
        import tolerance
    atol, rtol, rho = tolerance.effective_atol(precision, max_error)
    if (atol, rtol, rho) != (max_error, spec.rtol, spec.required_matched_ratio):
        raise ReferenceIdentityError("v2 numerical gate drifted")
    return spec


def validate_receipt(raw, receipt):
    if not isinstance(receipt, dict) or receipt.get("status") != "passed":
        raise ReferenceIdentityError("reference receipt absent/not passed")
    expected = check_observed(raw, receipt.get("observed"))
    if receipt != expected:
        raise ReferenceIdentityError("reference receipt hashes/fields drifted")
    return expected
