"""Typed compiler provenance tests; fake native class, no compiler or GPU claim."""
import ast
import copy
import hashlib
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from operator_agent import Verifier
from operator_agent.compile_evidence import capture_compile_failure, validate_compile_failure
from operator_agent.input_fidelity import PROTOCOL, describe_inputs
from operator_agent.measurement import TRANSPORT_KEY, TRANSPORT_UNKNOWN
from operator_agent.quality_suite import get_case, task_for
from operator_agent.schemas import VerificationLevel
from tests.backend_fakes import gpu_fake


SOURCE = '''import torch
import triton
import triton.language as tl

@triton.jit
def _normalize(X, Y, N: tl.constexpr):
    value = tl.float32(1.0)
    tl.store(Y, value)

def kernel(X):
    Y = torch.empty_like(X)
    _normalize[(1,)](X, Y, X.shape[1])
    return Y
'''
FRAGMENT = '''def _normalize(X, Y, N: tl.constexpr):
    value = tl.float32(1.0)
    tl.store(Y, value)
'''


@pytest.fixture
def compiler(monkeypatch):
    class CompilationError(Exception):
        def __init__(self, src, node):
            self.src, self.node = src, node
            self.error_message = "PRIVATE PATH AND PROSE MUST NEVER BE EVIDENCE"

        def __str__(self):
            return "synthetic compiler error"

    errors = types.ModuleType("triton.compiler.errors")
    errors.CompilationError = CompilationError
    compiler_module = types.ModuleType("triton.compiler")
    compiler_module.errors = errors
    triton = types.ModuleType("triton")
    triton.compiler = compiler_module
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.compiler", compiler_module)
    monkeypatch.setitem(sys.modules, "triton.compiler.errors", errors)
    return CompilationError


def failure(compiler, fragment=FRAGMENT):
    node = next(n for n in ast.walk(ast.parse(fragment))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "float32")
    return compiler(fragment, node)


def seal(payload):
    body = {k: v for k, v in payload.items() if k != "evidence_hash"}
    payload["evidence_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return payload


@pytest.mark.parametrize("fragment", [FRAGMENT, "@triton.jit\n" + FRAGMENT,
    FRAGMENT.replace("    ", "  ")])
def test_native_ast_maps_to_original_full_source_without_exception_prose(compiler, fragment):
    item = capture_compile_failure(failure(compiler, fragment), SOURCE, "candidate_launch")
    assert item is not None
    assert item["schema"] == "candidate-jit-compile-failure-v1"
    assert item["line"] == 7 and item["column"] == 12
    assert item["end_line"] == 7 and item["end_column"] == 27
    assert item["helper_name"] == "_normalize"
    assert validate_compile_failure(item, SOURCE) == item
    assert "PRIVATE" not in json.dumps(item)
    # The consumer cannot mutate retained producer evidence.
    assert validate_compile_failure(item, SOURCE) is not item


@pytest.mark.parametrize("phase", [None, "reference", "import", "candidate_launch_later", False])
def test_non_candidate_phases_cannot_capture_even_native_error(compiler, phase):
    assert capture_compile_failure(failure(compiler), SOURCE, phase) is None


def test_message_subclasses_and_other_exceptions_cannot_claim_native_failure(compiler):
    class Derived(compiler):
        pass
    native = failure(compiler)
    errors = [RuntimeError("CompilationError: tl.float32 at line 7"), AssertionError("compiler"),
              Derived(native.src, native.node)]
    for error in errors:
        assert capture_compile_failure(error, SOURCE, "candidate_launch") is None


def test_capture_never_formats_or_reads_exception_prose(compiler, monkeypatch):
    def forbidden(*args):
        raise AssertionError("exception prose was read")
    monkeypatch.setattr(compiler, "__str__", forbidden)
    error = failure(compiler)
    error.error_message = types.SimpleNamespace(__str__=forbidden)
    assert capture_compile_failure(error, SOURCE, "candidate_launch")


@pytest.mark.parametrize("mutation", ["different_source", "wrong_node", "wrong_location",
    "missing_node", "extra_function", "non_jit", "duplicate_helper", "too_large", "too_deep",
    "too_many_nodes"])
def test_unsupported_or_ambiguous_provenance_is_unknown(compiler, mutation):
    error, source = failure(compiler), SOURCE
    if mutation == "different_source":
        error.src = FRAGMENT.replace("1.0", "2.0")
    elif mutation == "wrong_node":
        error.node = ast.parse("foo()\n").body[0].value
    elif mutation == "wrong_location":
        error.node.lineno += 1
    elif mutation == "missing_node":
        error.node = None
    elif mutation == "extra_function":
        error.src += "\ndef extra():\n    pass\n"
    elif mutation == "non_jit":
        source = source.replace("@triton.jit\n", "")
    elif mutation == "duplicate_helper":
        source += "\n" + "@triton.jit\n" + FRAGMENT
    elif mutation == "too_large":
        source += "#" * 32769
    elif mutation == "too_many_nodes":
        source += "\n".join("unused = 1" for _ in range(1100))
    else:
        source += "x = " + "+".join("1" for _ in range(45))
    assert capture_compile_failure(error, source, "candidate_launch") is None


@pytest.mark.parametrize("field,value", [("source_hash", "0" * 64),
    ("helper_name", "kernel"), ("line", 8), ("line", True), ("column", -1),
    ("end_column", 26), ("node_ast_hash", "0" * 64), ("phase", "reference"),
    ("producer", "RuntimeError"), ("schema", "future"), ("unknown", "extra")])
def test_consumer_rechecks_even_resealed_tampered_payloads(compiler, field, value):
    item = capture_compile_failure(failure(compiler), SOURCE, "candidate_launch")
    item[field] = value
    seal(item)
    assert validate_compile_failure(item, SOURCE) is None


def test_hash_source_limits_and_compiler_node_limits_are_independent(compiler):
    item = capture_compile_failure(failure(compiler), SOURCE, "candidate_launch")
    assert validate_compile_failure(item, SOURCE + "# drift") is None
    item["evidence_hash"] = "0" * 64
    assert validate_compile_failure(item, SOURCE) is None
    error = failure(compiler)
    node = ast.Name(id="X", ctx=ast.Load())
    for _ in range(40):
        node = ast.UnaryOp(op=ast.USub(), operand=node)
    ast.copy_location(node, error.node)
    error.node = node
    assert capture_compile_failure(error, SOURCE, "candidate_launch") is None


def runner_function():
    """Execute the real function body without importing GPU-selection globals."""
    path = Path(__file__).resolve().parents[2] / "remote_scripts" / "run_kernel.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "run_verify")
    scope = {"capture_compile_failure": capture_compile_failure,
             "ReferenceIdentityError": type("ReferenceIdentityError", (Exception,), {}),
             "InputFidelityError": type("InputFidelityError", (Exception,), {}),
             "JSON_PROTOCOL": "exact", "shield_torch": nullcontext,
             "_detect_language": lambda _: "triton",
             "to_torch_inputs": lambda inputs, device: inputs,
             "torch_input_readback": lambda _: {"synthetic": "unit-test-only"},
             "_load_reference_from_source": lambda *a: (lambda: object(), "synthetic")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope


@pytest.mark.parametrize("phase", ["candidate", "reference", "module"])
def test_runner_phase_limits_evidence_to_actual_candidate_call(compiler, monkeypatch, phase):
    scope = runner_function()
    error = failure(compiler)
    def raise_error(*args, **kwargs):
        raise error
    class Device:
        type = "cuda"
        def __str__(self):
            return "cuda"
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True),
                                  device=lambda _: Device(), no_grad=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    scope["_load_module"] = (raise_error if phase == "module" else
        lambda _: types.SimpleNamespace(kernel=raise_error if phase == "candidate" else lambda **kw: object()))
    if phase == "reference":
        scope["_load_reference_from_source"] = lambda *a: (raise_error, "synthetic")
    result = scope["run_verify"](SOURCE, {}, "layernorm", 1e-5,
                                  preloaded_inputs={"X": object()})
    assert result["success"] is False
    if phase == "candidate":
        assert validate_compile_failure(result["candidate_compile_failure"], SOURCE)
        assert result["device"] == "cuda"
    else:
        assert "candidate_compile_failure" not in result


def verifier_result(compiler, mutation=None, *, suite=True):
    evidence = capture_compile_failure(failure(compiler), SOURCE, "candidate_launch")
    task = task_for(get_case("layernorm-tail"))
    if not suite:
        task.constraints.correctness_suite = None
        task.constraints.correctness_profile = None
    inputs = {"X": np.arange(7 * 33, dtype=np.float32).reshape(7, 33)}
    def remote(code, actual_inputs, *args, **kwargs):
        response = {"success": False, "error": "unchanged synthetic failure", "device": "cuda",
            "candidate_compile_failure": copy.deepcopy(evidence),
            "input_fidelity": {"protocol": PROTOCOL, "representation": "torch-storage-v1",
                "observed": describe_inputs(actual_inputs), "tensor_devices": {"X": "cuda:0"}}}
        if mutation:
            mutation(response)
        return response
    backend = gpu_fake(compile=lambda _: True, verify_function=remote,
                       get_hardware_info=lambda: {"name": "synthetic"})
    verifier = Verifier(backend)
    return verifier.verify(SOURCE, task, inputs, stop_after=VerificationLevel.L2_FUNCTION)[-1]


def test_validated_evidence_survives_full_verifier_primary_suite_failure(compiler):
    result = verifier_result(compiler)
    assert result.passed is False and result.metrics["correctness"]["verdict"] == "failed"
    assert result.metrics["input_fidelity_verified"] is True
    assert result.metrics["device"] == "cuda"
    assert validate_compile_failure(result.metrics["candidate_compile_failure"], SOURCE)
    assert result.error_log == "unchanged synthetic failure"


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(reference_error="reference failed"),
    lambda p: p.update(reference_identity_unknown=True),
    lambda p: p.update({TRANSPORT_KEY: TRANSPORT_UNKNOWN}),
    lambda p: p.update({TRANSPORT_KEY: None}),
    lambda p: p.update({TRANSPORT_KEY: "invalid"}),
    lambda p: p.update(device="cpu"),
    lambda p: p["input_fidelity"].update(representation="echoed-numpy"),
    lambda p: p["input_fidelity"].update(tensor_devices={"X": "cpu"}),
    lambda p: p["input_fidelity"].update(tensor_devices={}),
    lambda p: p["input_fidelity"].update(observed={}),
    lambda p: p["candidate_compile_failure"].update(source_hash="0" * 64),
    lambda p: p.update(success=None),
])
def test_reference_transport_fidelity_and_unbound_failures_never_propagate(compiler, mutation):
    result = verifier_result(compiler, mutation)
    assert "candidate_compile_failure" not in result.metrics


def test_no_correctness_suite_means_no_actual_input_fidelity_proof(compiler):
    assert "candidate_compile_failure" not in verifier_result(compiler, suite=False).metrics
