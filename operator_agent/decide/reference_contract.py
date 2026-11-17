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
#: attention 的 profile。语义与 layernorm 不同（没有 eps 归一化、没有 affine、softmax 沿末轴），
#: 所以它是**另一份**契约，不是同一份改默认值。
PROFILE_ATTENTION = "builtin-attention-fp16-cuda-v1"
#: 融合族：`Y = LayerNorm(X + R)`。参考是 eager 两 kernel（fp16 add + native LN），
#: primitive 单独成文件并单独 pin hash —— 不改 `builtin_layernorm.py`，因此不影响
#: layernorm v2 / attention 已冻结的参考身份。
PROFILE_FUSED_LN_RESIDUAL = "builtin-fused-layernorm-residual-fp16-cuda-v1"
#: `operator_agent/decide/builtin_fused_layernorm_residual.py` 的 SHA256。
#: 改那个文件就必须同步改这里并换 profile 版本，否则参考身份对不上。
FUSED_LN_RESIDUAL_PRIMITIVE_SHA256 = (
    "ee1d6cd42989aef91adefa713b1a0f3c6f08fdd5ca9fb664bb3abde467c875c2")
# Filled from isolated read-only attestation, never discovered/accepted from a
# candidate or auto-updated after a numerical failure.
PINNED_RUNTIME = dict(
    # 版本串的真源在 `core.runtime_pin.TORCH_VERSION`（部署模板 execute 层读它，
    # 而 execute 不能 import decide）。这里**不能** import: 本文件在远端是被当成
    # 顶层模块平铺加载的（scp 到 work_dir 根），任何 `from ..` 都会 ImportError，
    # 身份回执随之变空。两份字面量由 tests/unit/v08/test_deploy_runtime_pin.py 钉住。
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


#: 受信任参考的注册表：**按 profile 取值**。当前只有 layernorm 一项 —— attention 要在
#: C2b（容差标定）拿到实测数字之后才允许登记，在那之前它不在表里，`as_reference()`
#: 会对它 fail-closed（"不在受信任注册表里"），而不是接受一份拍出来的容差。
#:
#: 为什么要注册表而不是就地改默认值：`ReferenceContract` 的字段与默认值本身就是
#: layernorm 的语义，改它们会让 layernorm 的 `digest()` 变化 —— 那等于把所有已冻结的
#: layernorm 回执一起作废。
#: attention 的受信任参考。容差来自 2026-11-07 H20 标定（见
#: docs/design/v08/attention-family.md §14）：
#:   - `atol=3e-4, rtol=1e-2, rho=1.0` 让"好实现"全元素通过；
#:   - 同一组数字把"错缩放(1/sqrt(S))"判否（通过率 0.013）、"causal 差一位"判否（0.49）；
#:   - **但它抓不住"没做 max 平移"** —— 正常随机输入下 exp 不溢出，必须靠套件里的
#:     "极值输入"那条 case 才能抓到。容差不是充分条件，这一点写在契约外。
ATTENTION_REFERENCE = ReferenceContract(
    ExecutionIdentity(**PINNED_RUNTIME),
    profile=PROFILE_ATTENTION,
    definition="native-pytorch-sdpa-mha-prefill-fp16-v1",
    input_dtype="fp16",
    candidate_accum="fp32_declared_not_arithmetic_attestation",
    output_dtype="fp16",
    reference_accum="native_implementation_pinned",
    epsilon=0.0,                       # attention 没有 eps 归一化参数
    normalized_axis=-1,                # softmax 沿末轴
    variance="softmax_last_axis",
    affine=False,
    fp_fusion="forbid_native_sdpa",    # 候选不得调用 SDPA
    deterministic_outputs=True,        # 同运行两次逐位相同（2026-11-07 H20 实测）
    comparison="fp32_and",
    rtol=1e-2,
    required_matched_ratio=1.0)

#: 融合 residual+LayerNorm 的受信任参考。语义与 layernorm v2 相同，只多了一步
#: `X + R`（fp16，eager 序列的第一个 kernel）；primitive 单独文件、单独 pin hash。
#: `atol` 由 2026-11-17 H20/GPU7 的 good/alt/bad 分离标定给出，不从 v2 借值。
FUSED_LN_RESIDUAL_REFERENCE = ReferenceContract(
    ExecutionIdentity(**dict(PINNED_RUNTIME,
                             primitive_sha256=FUSED_LN_RESIDUAL_PRIMITIVE_SHA256)),
    profile=PROFILE_FUSED_LN_RESIDUAL,
    definition="native-pytorch-residual-add-layernorm-last-axis-no-affine-fp16-v1",
    input_dtype="fp16",
    candidate_accum="fp32_declared_not_arithmetic_attestation",
    output_dtype="fp16",
    reference_accum="native_implementation_pinned",
    epsilon=1e-5,
    normalized_axis=-1,
    variance="population",
    affine=False,
    # 参考本身是 eager 两 kernel；候选允许把 add 与 LN 融成一个核，容差不变。
    fp_fusion="candidate_fusion_allowed_reference_is_two_eager_kernels",
    deterministic_outputs=True,
    comparison="fp32_and",
    rtol=1e-2,
    required_matched_ratio=1.0)

REGISTERED = {PROFILE_V2: ReferenceContract(ExecutionIdentity(**PINNED_RUNTIME)),
              PROFILE_ATTENTION: ATTENTION_REFERENCE,
              PROFILE_FUSED_LN_RESIDUAL: FUSED_LN_RESIDUAL_REFERENCE}


def registered_reference(profile=PROFILE_V2):
    """按 profile 取受信任参考；未知 profile 抛（不兜默认）。"""
    try:
        return REGISTERED[profile]
    except KeyError:
        raise ReferenceIdentityError(
            f"no registered reference for profile {profile!r}; "
            f"registered: {sorted(REGISTERED)}") from None


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
    try:
        expected = registered_reference(spec.profile)
    except ReferenceIdentityError:
        raise
    if spec != expected:
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


def observe_execution(profile=PROFILE_V2):
    import torch
    import torch.nn.functional as F
    if __package__:
        from ..decide import builtin_fused_layernorm_residual, builtin_layernorm
    else:
        import builtin_fused_layernorm_residual
        import builtin_layernorm
    if profile == PROFILE_FUSED_LN_RESIDUAL:
        primitive = builtin_fused_layernorm_residual
    elif profile in (PROFILE_V2, PROFILE_ATTENTION):
        primitive = builtin_layernorm
    else:
        raise ReferenceIdentityError(f"no registered reference primitive for profile {profile!r}")
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
        functional_sha256=_file_hash(F.__file__), primitive_sha256=_file_hash(primitive.__file__),
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
        return check_observed(raw, observe_execution(as_reference(raw).profile))
    except ReferenceIdentityError:
        raise
    except Exception as exc:
        raise ReferenceIdentityError("reference identity probe unavailable: " + type(exc).__name__) from exc


#: 每个 profile **标定的绝对容差本身**，不是上限。数字来自标定，不是随手写的：
#: layernorm 1e-5（既有 v2 标定），attention 3e-4（2026-11-07 H20 网格：
#: 好实现全过、错缩放与 causal 差一位被判否）。
#:
#: 为什么必须按相等绑定而不是只当天花板：`ConstraintSpec.max_error` 的全局默认是
#: 1e-5，那恰好是 **layernorm** 的标定值。只查 `<= ceiling` 时，attention 会静默继承
#: 一个比自己的标定严 30× 的阈值（实测把 max_abs=1.22e-4 的正确核判成 99.9608% 匹配），
#: 而调用方无权纠正 —— max_error/atol/rtol 都在 hitl_tools.FORBIDDEN_KEYS 里。
#: layernorm 的默认正好等于自己的标定值，所以这个洞只在 attention 上显形。
REGISTERED_ATOL = {'builtin-layernorm-fp16-cuda-v2': 1e-5,
                   'builtin-attention-fp16-cuda-v1': 3e-4,
                   # 2026-11-17 GPU7 标定（eager fp16 add 参考）：
                   # good（fp32 统计）max_abs=3.90625e-3、alt（fp16 add）同量级，
                   # bad（漏加 R）max_abs=4.53、匹配率 1.18%。5e-3 是最小的圆整值，
                   # 同时放行 good/alt、拒绝 bad；不是从 layernorm v2 借来的 1e-5。
                   'builtin-fused-layernorm-residual-fp16-cuda-v1': 5e-3}
_PROFILE_OPERATOR = {'builtin-layernorm-fp16-cuda-v2': 'layernorm',
                     'builtin-attention-fp16-cuda-v1': 'attention',
                     'builtin-fused-layernorm-residual-fp16-cuda-v1':
                         'fused_layernorm_residual'}


def registered_atol(raw):
    """按 profile 取标定的绝对容差。

    给 **不做数值比较** 的调用点用（L3 计时、dispatch 前置门）：那里查的是
    profile/算子/精度身份，atol 槽该填注册值，而不是写死某一个 profile 的数。
    带任务容差的调用点（契约冻结、profile_check、L2 run_verify）必须自己传进来，
    由 validate_gate 判相等。
    """
    profile = as_reference(raw).profile
    try:
        return REGISTERED_ATOL[profile]
    except KeyError:
        raise ReferenceIdentityError(
            f"no registered absolute tolerance for profile {profile!r}") from None


def validate_gate(raw, precision, max_error, reference_source="", kernel_type="layernorm"):
    spec = as_reference(raw)
    expected_operator = _PROFILE_OPERATOR.get(spec.profile)
    registered = REGISTERED_ATOL.get(spec.profile)
    if expected_operator is None or registered is None:
        raise ReferenceIdentityError("unknown reference profile for the numerical gate")
    if (precision != "fp16" or kernel_type != expected_operator or reference_source
            or type(max_error) not in (int, float)):
        raise ReferenceIdentityError("operator/dtype/source/tolerance mismatch")
    if max_error != registered:
        raise ReferenceIdentityError(
            f"absolute tolerance {max_error} is not the value calibrated for {spec.profile} "
            f"({registered}); 容差按 profile 绑定，不接受全局默认漏过来的值")
    if __package__:
        from ..core import tolerance
    else:
        import tolerance
    atol, rtol, rho = tolerance.effective_atol(precision, max_error)
    if (atol, rtol, rho) != (max_error, spec.rtol, spec.required_matched_ratio):
        raise ReferenceIdentityError("numerical gate drifted")
    return spec


def validate_receipt(raw, receipt):
    if not isinstance(receipt, dict) or receipt.get("status") != "passed":
        raise ReferenceIdentityError("reference receipt absent/not passed")
    expected = check_observed(raw, receipt.get("observed"))
    if receipt != expected:
        raise ReferenceIdentityError("reference receipt hashes/fields drifted")
    return expected
