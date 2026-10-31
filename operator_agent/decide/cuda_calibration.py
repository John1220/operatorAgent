"""Fixed native CUDA ruler probes, shared with the owned persistent runner.

Reuse production timing, never copy its CUDA event loop. This is provenance and
separation evidence, NOT an automatic launch-floor update or an absolute anchor.
No caller-supplied source, shape, repetitions or paths are executable here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

try:
    from ..decide.input_fidelity import describe_inputs
    from ..decide.remote_input_fidelity import to_torch_inputs, torch_input_readback
except ImportError:  # same source deployed as a runner sibling
    from input_fidelity import describe_inputs
    from remote_input_fidelity import to_torch_inputs, torch_input_readback

PROTOCOL = "native-cuda-ruler-v1"
CHANNEL = "production-cuda-events-interleaved-l2-flush"
SHAPE = (2048, 1024)
AUDIT_WARMUP = 25
AUDIT_SAMPLES = 200
MIN_SLOW_RATIO = 2.0
CONFIG = dict(pair_warmup=10, pair_repeat=30, warmup_ms_budget=25,
              repeat_ms_budget=50, max_repeats=5000, block_max=10,
              l2_flush_bytes=256 * 1024 * 1024,
              audit_warmup=AUDIT_WARMUP, audit_samples=AUDIT_SAMPLES,
              clock_batch_repeats=1024, clock_samples=3)

REFERENCE_SOURCE = '''import torch
import triton
import triton.language as tl
@triton.jit
def _empty():
    pass
@triton.jit
def _copy(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,0)
    tl.store(Y+i,x,i<N)
def empty(X):
    _empty[(1,)]()
def kernel(X):
    Y=torch.empty_like(X)
    _copy[(triton.cdiv(X.numel(),1024),)](X,Y,X.numel(),1024)
    return Y
def slow(X):
    Y=torch.empty_like(X)
    for _ in range(8):
        _copy[(triton.cdiv(X.numel(),1024),)](X,Y,X.numel(),1024)
    return Y
'''


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fixed_inputs():
    # Exact, finite, nonconstant values; generation independent of Torch RNG.
    return {"X": np.tile(np.array([-1, 0, 1, 2], dtype=np.float32),
                         math.prod(SHAPE) // 4).reshape(SHAPE)}


def source_hashes():
    here = Path(__file__).resolve().parent
    runner = here / "run_kernel.py"
    if not runner.is_file():
        runner = here.parents[1] / "remote_scripts" / "run_kernel.py"
    paths = [runner] + [here / n for n in (
        "cuda_calibration.py", "cuda_ruler_profile.py", "anticheat.py", "input_fidelity.py", "remote_input_fidelity.py")]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_request(mode, request):
    if mode not in ("audit", "heartbeat") or type(mode) is not str:
        raise ValueError("native CUDA probe requires fixed audit/heartbeat mode")
    if set(request) - {"id", "action", "mode"}:
        raise ValueError("native CUDA probe forbids arbitrary code/inputs/options")


def _positive(x):
    return type(x) in (int, float) and math.isfinite(x) and x > 0


def validate_probe(payload, *, mode, selected_gpu, sources=None):
    """Controller-side fail-closed receipt validation, also used by heartbeat.

    This validates observed identities and bounded evidence, not an adversarial
    same-UID runner. An audit is never authority to change a qualification gate.
    """
    validate_request(mode, {"mode": mode})
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise ValueError("native CUDA probe failed: " + str(payload))
    ident = payload.get("identity")
    if not isinstance(ident, dict) or payload.get("identity_hash") != digest(ident):
        raise ValueError("native CUDA identity hash mismatch")
    expected = dict(protocol=PROTOCOL, channel=CHANNEL, config=CONFIG,
                    sources=source_hashes() if sources is None else sources,
                    reference_sha256=hashlib.sha256(REFERENCE_SOURCE.encode()).hexdigest(),
                    inputs=describe_inputs(fixed_inputs()))
    for k, v in expected.items():
        if ident.get(k) != v:
            raise ValueError("native CUDA identity mismatch: " + k)
    if payload.get("mode") != mode or payload.get("absolute_anchor_checked") is not False:
        raise ValueError("native CUDA mode/anchor scope mismatch")
    if type(selected_gpu) is not int or selected_gpu < 0 or ident.get("selected_gpu") != selected_gpu:
        raise ValueError("native CUDA physical GPU unknown/mismatched")
    device = ident.get("device", {})
    if (device.get("visible_index") != 0 or device.get("count") != 1
            or device.get("cuda_visible_devices") != str(selected_gpu)
            or not device.get("name") or device.get("name") == "cpu"):
        raise ValueError("native CUDA device visibility mismatch")
    runtime = ident.get("runtime", {})
    if any(type(runtime.get(k)) is not str or not runtime[k]
           for k in ("torch", "cuda", "triton", "driver", "gpu_uuid", "pci_bus_id")):
        raise ValueError("native CUDA runtime identity incomplete")
    if ident.get("default_stream") is not True:
        raise ValueError("native CUDA unexpected stream")
    readback = payload.get("input_fidelity", {})
    if (readback.get("observed") != expected["inputs"]
            or readback.get("tensor_devices") != {"X": "cuda:0"}
            or readback.get("representation") != "torch-storage-v1"
            or payload.get("outputs_exact") != {"good": True, "slow": True}):
        raise ValueError("native CUDA exact input/output evidence missing")
    good, slow, pair = (payload.get(k, {}) for k in ("good", "slow", "paired"))
    n = good.get("runs")
    if (type(n) is not int or not CONFIG["pair_repeat"] <= n <= CONFIG["max_repeats"]
            or slow.get("runs") != n or pair.get("paired_runs") != n):
        raise ValueError("native CUDA sample counts mismatch")
    for stats in (good, slow):
        if any(not _positive(stats.get(k)) for k in ("min_time_ms", "median_time_ms", "max_time_ms")):
            raise ValueError("native CUDA nonfinite/nonpositive timing")
        if not stats["min_time_ms"] <= stats["median_time_ms"] <= stats["max_time_ms"]:
            raise ValueError("native CUDA timing order mismatch")
    windows = pair.get("arm_windows", {})
    wid = windows.get("candidate")
    if type(wid) is not str or not wid or windows.get("baseline") != wid:
        raise ValueError("native CUDA timing window mismatch")
    if pair.get("arm_identity") != dict(candidate=device, baseline=device, after=device):
        raise ValueError("native CUDA per-arm device drift")
    block = pair.get("paired_block_samples")
    if type(block) is not int or not 1 <= block <= CONFIG["block_max"]:
        raise ValueError("native CUDA block count mismatch")
    if not _positive(pair.get("speedup_paired_median")):
        raise ValueError("native CUDA paired ratio invalid")
    # Slow/good must separate in BOTH ratio-of-medians and paired observations.
    ratio = min(slow["median_time_ms"] / good["median_time_ms"], pair["speedup_paired_median"])
    if mode == "audit":
        samples = payload.get("samples", {})
        keys = {f"{kernel}-{channel}" for kernel in ("empty", "good")
                for channel in ("flush", "no-flush", "host-e2e")}
        if set(samples) != keys:
            raise ValueError("native CUDA audit sample channels missing")
        for xs in samples.values():
            if (not isinstance(xs, list) or len(xs) != AUDIT_SAMPLES
                    or any(not _positive(x) for x in xs)):
                raise ValueError("native CUDA audit samples invalid")
        if not isinstance(payload.get("telemetry"), dict):
            raise ValueError("native CUDA audit telemetry missing")
    return ratio


def native_heartbeat(backend, *, epoch, window_id, expected_identity_hash="", ruler_profile=None):
    from ..core.backend_caps import Cap
    from ..decide.calibration import HeartbeatResult, RULER_BROKEN
    backend.require(Cap.NATIVE_CUDA_CALIBRATION)
    try:
        payload = backend.calibration_probe(mode="heartbeat")
        ratio = validate_probe(payload, mode="heartbeat", selected_gpu=backend.selected_gpu)
        if expected_identity_hash and payload["identity_hash"] != expected_identity_hash:
            raise ValueError("native CUDA frozen calibration identity drift")
        ok = ratio >= MIN_SLOW_RATIO
        reasons = ("CUDA separation-only; no independent absolute anchor",)
        good_ratio = 1.0
        check = None
        if ruler_profile is not None:
            from ..decide.cuda_ruler_profile import anchor_reading
            check = anchor_reading(payload, ruler_profile)
            good_ratio = check["good_ratio"]
            ok = check["ok"]
            payload = dict(payload, anchor_check=check)
            reasons = ("CUDA independent frozen preflight absolute anchor checked",)
            if not check["ok"]:
                reasons += (f"{RULER_BROKEN}: good absolute ratio {good_ratio:.4g}, "
                            f"required band {check['good_ratio_band']}",)
        if not ok:
            reasons += (f"{RULER_BROKEN}: native slow/good separation {ratio:.4g} < {MIN_SLOW_RATIO}",)
        return HeartbeatResult(ok, epoch, reasons, good_ratio, ratio, window_id,
            channel=CHANNEL, identity_hash=payload["identity_hash"],
            evidence=payload, absolute_anchor_checked=check is not None,
            ruler_profile_hash=check["profile_hash"] if check else "")
    except Exception as exc:
        return HeartbeatResult(False, epoch, (f"{RULER_BROKEN}: native CUDA probe: {exc}",),
            float("nan"), float("nan"), window_id, channel=CHANNEL)


def freeze_audits(receipts, selected_gpu):
    """Two independent windows, all retained; no retry-until-pass selection."""
    if type(receipts) is not list or len(receipts) != 2:
        raise ValueError("calibration requires exactly two predeclared audit windows")
    for receipt in receipts:
        if validate_probe(receipt, mode="audit", selected_gpu=selected_gpu) < MIN_SLOW_RATIO:
            raise ValueError("calibration audit cannot distinguish good/slow CUDA references")
    first, second = receipts
    if first["identity_hash"] != second["identity_hash"]:
        raise ValueError("calibration audit identities differ")
    if first["paired"]["arm_windows"]["candidate"] == second["paired"]["arm_windows"]["candidate"]:
        raise ValueError("calibration requires independent production timing windows")
    content = dict(protocol=PROTOCOL, identity_hash=first["identity_hash"],
        identity=first["identity"], audit_receipts=receipts, floor_update_authorized=False,
        absolute_anchor_checked=False)
    return dict(content, hash=digest(content))


def read_calibration(path):
    path = Path(path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("calibration receipt exceeds bounded size")
    frozen = json.loads(path.read_text())
    if not isinstance(frozen, dict):
        raise ValueError("calibration receipt must be an object")
    expected = freeze_audits(frozen.get("audit_receipts"),
                            frozen.get("identity", {}).get("selected_gpu"))
    if frozen != expected:
        raise ValueError("frozen calibration content/hash mismatch")
    return frozen


def ruler_identity(runner):
    """Observed timing-channel identity, shared by probe and production L3.

    Fixed reference/input fields identify the calibration instrument, NOT the
    candidate workload. L3 input readback/run_binding identifies that separately.
    """
    import subprocess
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("native CUDA probe has no CUDA device; CPU fallback forbidden")
    import triton
    if (runner._L2_FLUSH_BYTES != CONFIG["l2_flush_bytes"]
            or runner.WARMUP_MS_BUDGET != CONFIG["warmup_ms_budget"]
            or runner.REPEAT_MS_BUDGET != CONFIG["repeat_ms_budget"]
            or runner.MAX_REPEATS != CONFIG["max_repeats"]):
        raise RuntimeError("production CUDA timing configuration drift")
    before = runner._arm_identity(True)
    gpu = runner._SELECTED_GPU_ID
    smi = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=driver_version,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
        text=True, timeout=10).strip().split(",")
    if len(smi) != 3:
        raise RuntimeError("native CUDA driver/GPU identity unreadable")
    return dict(protocol=PROTOCOL, channel=CHANNEL, config=CONFIG, sources=source_hashes(),
        reference_sha256=hashlib.sha256(REFERENCE_SOURCE.encode()).hexdigest(),
        inputs=describe_inputs(fixed_inputs()), selected_gpu=gpu, device=before,
        runtime=dict(torch=str(torch.__version__), cuda=str(torch.version.cuda),
                     triton=str(triton.__version__), driver=smi[0].strip(),
                     gpu_uuid=smi[1].strip(), pci_bus_id=smi[2].strip()),
        default_stream=torch.cuda.current_stream() == torch.cuda.default_stream())


def run_probe(mode, request, runner):
    """Trusted runner control action; refuse unknown options BEFORE GPU work."""
    validate_request(mode, request)
    import torch
    ident = ruler_identity(runner)
    before, gpu = ident["device"], ident["selected_gpu"]
    inputs = to_torch_inputs(fixed_inputs(), torch.device("cuda"))
    readback = torch_input_readback(inputs)
    module = runner._load_module(REFERENCE_SOURCE)
    good = lambda: module.kernel(**inputs)
    slow = lambda: module.slow(**inputs)
    empty = lambda: module.empty(**inputs)
    outputs = {"good": bool(torch.equal(good(), inputs["X"])),
               "slow": bool(torch.equal(slow(), inputs["X"]))}
    if not all(outputs.values()):
        raise RuntimeError("native CUDA fixed reference output incorrect")
    good_stats, slow_stats, paired = runner._bench_pair_interleaved(good, slow, 10, 30, True)
    result = dict(success=True, mode=mode, identity=ident, identity_hash=digest(ident),
        absolute_anchor_checked=False, input_fidelity=readback, outputs_exact=outputs,
        good=good_stats, slow=slow_stats, paired=paired)
    if mode == "audit":
        samples = {}
        telemetry = {"before": runner.read_gpu_state()}
        for name, fn in (("empty", empty), ("good", good)):
            for _ in range(AUDIT_WARMUP):
                fn()
            torch.cuda.synchronize()
            for flush in (False, True):
                samples[name + ("-flush" if flush else "-no-flush")] = runner._timed_runs(
                    fn, AUDIT_SAMPLES, True, flush_l2=flush)
            # Distinct host dispatch+device+sync observation, never a net ruler.
            xs = []
            for _ in range(AUDIT_SAMPLES):
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                xs.append((time.perf_counter() - t0) * 1000)
            samples[name + "-host-e2e"] = xs
        telemetry["after"] = runner.read_gpu_state()
        # The SAME bounded batch is observed by production events and host
        # monotonic time. Discard each output, never retain 1024 x 8 MiB.
        def clock_batch():
            for _ in range(CONFIG["clock_batch_repeats"]):
                good()
        clock_samples = []
        for _ in range(CONFIG["clock_samples"]):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            event_ms = runner._timed_runs(clock_batch, 1, True, flush_l2=True)[0]
            host_ms = (time.perf_counter() - t0) * 1000
            clock_samples.append(dict(cuda_ms=event_ms, host_ms=host_ms))
        result.update(samples=samples, telemetry=telemetry,
            clock_crosscheck=dict(protocol="cuda-batch-vs-host-monotonic-v1",
                batch_repeats=CONFIG["clock_batch_repeats"], samples=clock_samples),
            sample_medians_ms={k: statistics.median(v) for k, v in samples.items()})
    if runner._arm_identity(True) != before:
        raise RuntimeError("native CUDA device changed during probe")
    validate_probe(result, mode=mode, selected_gpu=gpu)
    return result
