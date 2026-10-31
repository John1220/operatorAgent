"""Exact SSH L2 transport + actual runner/Torch readback; GPU checks are explicit.

CPU runner tests exercise the protocol, NOT GPU performance/launch proof.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from operator_agent import MockGPU, Verifier
from operator_agent.decide.input_fidelity import InputFidelityError, describe_inputs, check_readback
from operator_agent.decide.remote_input_fidelity import (JSON_PROTOCOL, encode_json_inputs,
    decode_json_inputs, to_torch_inputs, torch_input_readback, torch_output_digest)
from operator_agent.execute.remote_executor import SSHExecutor, RemoteExecutor
from operator_agent.core.backend_caps import Cap
from operator_agent.core.schemas import KernelArtifact, VerificationLevel, VerificationResult
from operator_agent.core.report import build_report
from tests.unit.test_correctness_suite import case, task, CorrectnessSuite
from tests.unit.test_input_wire_format import _runner

GOOD = '''import torch
def kernel(A, B):
    return (A[:, :, None] * B[None, :, :]).sum(dim=1)
'''
BAD_DENSE = '''import torch
def kernel(A, B):
    a = A.as_strided(A.shape, (A.shape[1], 1), storage_offset=0)
    b = B.as_strided(B.shape, (B.shape[1], 1), storage_offset=0)
    return (a[:, :, None] * b[None, :, :]).sum(dim=1)
'''


@pytest.fixture(scope="module")
def runner():
    return _runner()


def _views():
    x = np.arange(80, dtype=np.float32).reshape(10, 8)
    writable_broadcast = np.ndarray((3, 7), dtype=x.dtype, buffer=x, strides=(32, 0))
    return [x, x[1::2, 1::2], x.T, np.asfortranarray(x), writable_broadcast,
            np.array(3, dtype=np.int32), x[:0, :]]


@pytest.mark.parametrize("index", range(7))
def test_json_and_torch_roundtrip_preserves_actual_storage_and_views(index):
    sent = {"X": _views()[index], "eps": np.float32(1e-5), "causal": False}
    packet = json.loads(json.dumps(encode_json_inputs(sent), allow_nan=False))
    restored = decode_json_inputs(packet)
    assert describe_inputs(restored) == describe_inputs(sent)
    tensors = to_torch_inputs(restored, "cpu")
    receipt = torch_input_readback(tensors)
    assert check_readback(describe_inputs(sent), receipt)["input_fidelity_verified"] is True
    assert receipt["tensor_devices"] == {"X": "cpu"}


@pytest.mark.parametrize("scalar", [None, False, 3, "causal", b"\x00\xff", 1e-5,
    float("nan"), float("inf"), complex(1, -2), np.float16(0.5), np.int64(8)])
def test_typed_scalars_survive_strict_json_without_pickle_or_nonfinite_tokens(scalar):
    sent = {"scalar": scalar}
    wire = json.loads(json.dumps(encode_json_inputs(sent), allow_nan=False))
    assert describe_inputs(decode_json_inputs(wire)) == describe_inputs(sent)


def test_transport_preserves_zero_nonfinite_dtype_offset_not_seed_generation(runner):
    x = np.array([0, np.nan, np.inf, -np.inf, 2, 0], dtype=np.float16)
    sent = {"X": x[1::2], "eps": np.float32(1e-5)}
    packet = json.loads(json.dumps(encode_json_inputs(sent), allow_nan=False))
    assert "_generate" not in packet and "seed" not in json.dumps(packet)
    assert describe_inputs(runner._deserialize_inputs(packet)) == describe_inputs(sent)


@pytest.mark.parametrize("bad", ["version", "base64", "scalar_kind", "offset", "readonly_type"])
def test_malformed_wire_is_rejected_not_regenerated(bad):
    wire = encode_json_inputs({"X": np.ones(2, dtype=np.float32), "flag": False})
    if bad == "version":
        wire["protocol"] = "unknown-v2"
    elif bad == "base64":
        wire["storages"][0] = "not base64!"
    elif bad == "scalar_kind":
        wire["scalars"]["flag"]["kind"] = []
    elif bad == "offset":
        wire["tensors"]["X"]["offset"] = -1
    else:
        wire["tensors"]["X"]["writeable"] = 1
    with pytest.raises(InputFidelityError):
        decode_json_inputs(wire)


def test_json_caps_storage_and_scalar_payloads(monkeypatch):
    import operator_agent.decide.input_fidelity as fidelity
    packet = encode_json_inputs({"X": np.ones(32, dtype=np.float32)})
    monkeypatch.setattr(fidelity, "MAX_STORAGE_BYTES", 64)
    with pytest.raises(InputFidelityError, match="上限"):
        decode_json_inputs(packet)
    with pytest.raises(InputFidelityError, match="上限"):
        encode_json_inputs({"message": "x" * 100})


def test_device_views_share_storage_but_never_mutate_controller_inputs():
    x = np.arange(20, dtype=np.float32)
    sent = {"A": x[1::2], "B": x[3::2]}
    values = to_torch_inputs(sent, "cpu")
    assert check_readback(describe_inputs(sent), torch_input_readback(values))["input_fidelity_verified"] is True
    values["A"][1] = 999
    assert values["B"][0].item() == 999
    assert x[3] == 3


@pytest.mark.parametrize("kind", ["negative", "readonly", "unaligned", "endian"])
def test_unrepresentable_torch_inputs_refused_without_dense_fallback(kind):
    x = np.arange(20, dtype=np.float32)
    bad = {"negative": x[::-1], "readonly": np.broadcast_to(x[:1], (5,)),
           "unaligned": np.ndarray((2,), dtype=np.float32, buffer=x, offset=1),
           "endian": x.astype(">f4")}[kind]
    with pytest.raises(InputFidelityError):
        to_torch_inputs({"X": bad}, "cpu")


class _ProtocolOnlySSH(SSHExecutor):
    @property
    def capabilities(self):
        return frozenset({Cap.REMOTE_VERIFY})  # No REAL_DEVICE claim in CPU protocol tests.


def _offline_verifier(runner, monkeypatch, real_device=False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend = (SSHExecutor if real_device else _ProtocolOnlySSH)(host="offline")
    backend._deployed = True
    backend.compile = lambda code: True
    backend.get_hardware_info = lambda: {"device": "cpu", "name": "offline protocol test"}
    def call(payload):
        assert payload["action"] == "verify"
        # Same JSON boundary as SSH stdin and persistent-session line framing.
        p = json.loads(json.dumps(payload, allow_nan=False))
        return runner.run_verify(p["code"], p["inputs"], p["kernel_type"], p["max_error"],
                                 p["precision"], p["reference_source"])
    backend._ssh_exec_json = call
    verifier = Verifier(backend)
    verifier._verify_performance = Mock(return_value=VerificationResult(
        level=VerificationLevel.L3_PERFORMANCE, passed=True))
    return verifier, backend


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_real_runner_positive_stride_suite_passes_and_emits_full_repeat_evidence(runner, monkeypatch, dtype):
    verifier, _ = _offline_verifier(runner, monkeypatch)
    suite = CorrectnessSuite((case("stride", dtype=dtype, layout="strided"),
                              case("zeros", dtype=dtype, pattern="zeros")))
    results = verifier.verify(GOOD, task(suite), case(m=8, n=8, k=8).materialize())
    assert results[1].passed, results[1].message
    c = results[1].metrics["correctness"]
    assert c["verdict"] == "passed" and c["primary_inputs"]
    for row in c["cases"]:
        assert row["determinism"] == "passed"
        assert row["metrics"]["input_fidelity_verified"] is True
        assert row["repeat_metrics"]["input_fidelity_verified"] is True
        assert row["metrics"]["device"] == "cpu"  # This is NOT a GPU acceptance test.
        assert row["metrics"]["output_digest"] == row["repeat_metrics"]["output_digest"]
    verifier._verify_performance.assert_called_once()
    report = build_report(artifact=KernelArtifact(verification_results=results))
    assert "输入保真: 是" in report and "strides=" in report and "device=cpu" in report
    json.dumps(results[1].metrics, allow_nan=False)


def test_dense_assumption_kernel_passes_aligned_primary_but_fails_remote_suite(runner, monkeypatch):
    verifier, _ = _offline_verifier(runner, monkeypatch)
    suite = CorrectnessSuite((case("stride", layout="strided"),))
    results = verifier.verify(BAD_DENSE, task(suite), case(m=8, n=8, k=8).materialize())
    row = results[1].metrics["correctness"]["cases"][0]
    assert row["verdict"] == "failed"
    assert row["metrics"]["input_fidelity_verified"] is True
    verifier._verify_performance.assert_not_called()


def test_readonly_broadcast_is_unknown_not_silently_copied(runner, monkeypatch):
    verifier, _ = _offline_verifier(runner, monkeypatch)
    suite = CorrectnessSuite((case("broadcast", layout="broadcast"),))
    results = verifier.verify(GOOD, task(suite), case(m=8, n=8, k=8).materialize())
    row = results[1].metrics["correctness"]["cases"][0]
    assert row["verdict"] == "unknown" and "只读" in row["reason"]
    assert row["metrics"]["input_fidelity_verified"] is None
    verifier._verify_performance.assert_not_called()


def test_real_device_claim_cannot_promote_correct_cpu_fallback(runner, monkeypatch):
    verifier, _ = _offline_verifier(runner, monkeypatch, real_device=True)
    results = verifier.verify(GOOD, task(CorrectnessSuite((case(),))), case().materialize())
    assert results[1].metrics["correctness"]["verdict"] == "unknown"
    assert results[1].metrics["input_fidelity_verified"] is True
    assert "CPU" in results[1].metrics["correctness_execution_error"]
    verifier._verify_performance.assert_not_called()


def test_torch_readback_cannot_echo_numpy_arrays_as_device_tensors():
    with pytest.raises(InputFidelityError, match="NumPy"):
        torch_input_readback({"X": np.ones(2, dtype=np.float32)})


@pytest.mark.parametrize("field", ["strides", "dtype", "offset", "value_digest", "storage_digest"])
def test_remote_tampered_runtime_receipt_blocks_l3(runner, monkeypatch, field):
    verifier, backend = _offline_verifier(runner, monkeypatch)
    original = backend._ssh_exec_json
    def tamper(payload):
        result = original(payload)
        result["input_fidelity"]["observed"]["A"][field] = "changed"
        return result
    backend._ssh_exec_json = tamper
    results = verifier.verify(GOOD, task(CorrectnessSuite((case(),))), case().materialize())
    assert results[1].metrics["correctness"]["verdict"] == "unknown"
    assert results[1].metrics["input_fidelity_verified"] is False
    verifier._verify_performance.assert_not_called()


def test_actual_backend_contiguous_conversion_cannot_hide_behind_correct_output(runner, monkeypatch):
    verifier, _ = _offline_verifier(runner, monkeypatch)
    original = runner.to_torch_inputs
    monkeypatch.setattr(runner, "to_torch_inputs", lambda inputs, device:
        {k: v.contiguous() if isinstance(v, torch.Tensor) else v
         for k, v in original(inputs, device).items()})
    results = verifier.verify(GOOD, task(CorrectnessSuite((case(layout="strided"),))),
                              case(m=8, n=8, k=8).materialize())
    row = results[1].metrics["correctness"]["cases"][0]
    assert row["verdict"] == "unknown"
    assert row["metrics"]["input_fidelity_verified"] is False
    verifier._verify_performance.assert_not_called()


def test_candidate_mutation_does_not_change_pre_exec_receipt_or_reference(runner, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    x = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    result = runner.run_verify("def kernel(X):\n    X.fill_(0)\n    return X\n",
        encode_json_inputs({"X": x}), "relu", 1e-5, "fp32")
    assert result["success"] is False and result["max_error"] > 0
    assert check_readback(describe_inputs({"X": x}), result["input_fidelity"])["input_fidelity_verified"] is True
    assert x[2, 1] == 10


def test_exact_runner_precision_is_tolerance_not_input_cast(runner, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    x = np.full((3, 4), 0.1, dtype=np.float32)
    result = runner.run_verify("def kernel(X):\n    return X\n", encode_json_inputs({"X": x}),
                              "relu", 1e-5, "fp16")
    assert result["success"] is True
    assert check_readback(describe_inputs({"X": x}), result["input_fidelity"])["input_fidelity_verified"] is True
    assert result["output_digest"] == torch_output_digest(torch.from_numpy(x))


def test_remote_full_output_digest_catches_below_tolerance_nondeterminism(runner, monkeypatch):
    verifier, _ = _offline_verifier(runner, monkeypatch)
    code = GOOD.replace("return (A[:, :, None] * B[None, :, :]).sum(dim=1)",
        "out = (A[:, :, None] * B[None, :, :]).sum(dim=1).to(torch.float64)\n"
        "    return out + torch.rand_like(out) * 1e-10")
    results = verifier.verify(code, task(CorrectnessSuite((case(),))), case().materialize())
    row = results[1].metrics["correctness"]["cases"][0]
    assert row["verdict"] == "failed" and "确定性" in row["reason"]
    assert row["metrics"]["max_error"] < 1e-5
    verifier._verify_performance.assert_not_called()


def test_missing_full_output_digest_is_unknown_not_deterministic(runner, monkeypatch):
    verifier, backend = _offline_verifier(runner, monkeypatch)
    original = backend._ssh_exec_json
    def strip(payload):
        result = original(payload)
        result.pop("output_digest", None)
        return result
    backend._ssh_exec_json = strip
    results = verifier.verify(GOOD, task(CorrectnessSuite((case(),))), case().materialize())
    row = results[1].metrics["correctness"]["cases"][0]
    assert row["verdict"] == "unknown" and "指纹" in row["reason"]
    verifier._verify_performance.assert_not_called()


def test_triton_without_actual_cuda_cannot_be_simulated_to_a_pass(runner, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = runner.run_verify("import triton\ndef kernel(X): return X\n",
        encode_json_inputs({"X": np.ones(4, dtype=np.float32)}), "relu", 1e-5)
    assert result["success"] is False and "CUDA" in result["input_fidelity_error"]
    assert "input_fidelity" not in result


def test_remote_reference_nonfinite_is_unknown_even_if_candidate_is_finite(runner, monkeypatch):
    verifier, _ = _offline_verifier(runner, monkeypatch)
    t = task(CorrectnessSuite((case(),)))
    t.source.reference_source = ("def reference(A, B):\n"
        "    out = (A[:, :, None] * B[None, :, :]).sum(dim=1)\n"
        "    out[0, 0] = float('nan')\n    return out\n")
    results = verifier.verify(GOOD, t, case().materialize())
    assert results[1].metrics["correctness"]["verdict"] == "unknown"
    assert results[1].metrics["reference_nonfinite"] == 1
    verifier._verify_performance.assert_not_called()


@pytest.mark.parametrize("source", ["def broken(:", "value = 1"])
def test_invalid_declared_reference_does_not_fallback_to_builtin(runner, monkeypatch, source):
    verifier, _ = _offline_verifier(runner, monkeypatch)
    t = task(CorrectnessSuite((case(),)))
    t.source.reference_source = source
    results = verifier.verify(GOOD, t, case().materialize())
    assert results[1].metrics["correctness"]["verdict"] == "unknown"
    assert "reference_source" in results[1].metrics["reference_error"]
    verifier._verify_performance.assert_not_called()


def test_facade_forwards_support_and_unsupported_platforms_remain_unknown():
    facade = RemoteExecutor(mode="ssh")
    facade._backend = SSHExecutor(host="offline")
    assert facade.correctness_input_support(GOOD, "strided")[0] is True
    facade._backend = SSHExecutor(host="offline", platform="kunlunxin")
    assert facade.correctness_input_support(GOOD, "strided")[0] is False
    mock = MockGPU()
    assert mock.correctness_input_support("import triton\ndef kernel(X): return X", "strided")[0] is bool(
        mock._triton_available and torch.cuda.is_available())


def test_codec_deploy_sources_are_shared_and_precede_runner():
    files = SSHExecutor(host="offline")._remote_files()
    names = [Path(remote).name for _, remote in files]
    for name in ("input_fidelity.py", "remote_input_fidelity.py"):   # 远端部署名仍是裸名
        assert names.index(name) < names.index("run_kernel.py")
        assert next(local for local, remote in files
                    if Path(remote).name == name).parent.name == "decide"


def test_deployed_standalone_runner_reads_exact_json_without_repo_imports(tmp_path):
    for local, remote in SSHExecutor(host="offline")._remote_files():
        shutil.copyfile(local, tmp_path / Path(remote).name)
    sent = case(layout="strided").materialize()
    payload = {"action": "verify", "code": GOOD, "inputs": encode_json_inputs(sent),
               "kernel_type": "matmul", "max_error": 1e-5, "precision": "fp32"}
    env = {**os.environ, "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, str(tmp_path / "run_kernel.py")],
        input=json.dumps(payload, allow_nan=False), text=True, capture_output=True,
        cwd=tmp_path, env=env, timeout=30)
    assert result.returncode == 0, result.stderr
    answer = json.loads(result.stdout.splitlines()[-1])
    assert answer["success"], answer.get("error")
    assert answer["device"] == "cpu"  # Deployed import/wire contract only.
    assert check_readback(describe_inputs(sent), answer["input_fidelity"])["input_fidelity_verified"] is True


TRITON_RELU = '''import torch
import triton
import triton.language as tl
@triton.jit
def _relu(X, Y, N: tl.constexpr, C: tl.constexpr, SR: tl.constexpr, SC: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + (offs // C) * SR + (offs % C) * SC, offs < N, 0)
    tl.store(Y + offs, tl.maximum(x, 0), offs < N)
def kernel(X):
    out = torch.empty(X.shape, dtype=X.dtype, device=X.device)
    _relu[(triton.cdiv(X.numel(), 128),)](X, out, X.numel(), X.shape[1], X.stride(0), X.stride(1), 128)
    return out
'''


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA; CPU protocol tests do not count")
def test_real_cuda_triton_runner_and_mock_path_read_back_actual_device_views(runner):
    pytest.importorskip("triton")
    x = np.arange(-40, 40, dtype=np.float32).reshape(10, 8)[1::2, 1::2]
    sent = {"X": x}
    result = runner.run_verify(TRITON_RELU, encode_json_inputs(sent), "relu", 1e-5, "fp32")
    assert result["success"], result.get("error")
    assert result["device"].startswith("cuda")
    assert check_readback(describe_inputs(sent), result["input_fidelity"])["input_fidelity_verified"] is True
    local = MockGPU().execute(TRITON_RELU, sent)
    assert local.success, local.error
    assert check_readback(describe_inputs(sent), local.metrics["input_fidelity"])["input_fidelity_verified"] is True
    np.testing.assert_array_equal(local.output, np.maximum(x, 0))
