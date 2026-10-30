"""Storage/view transport, independent child evidence, and bad layout kernels."""
import copy
import pickle

import numpy as np
import pytest

from operator_agent import MockGPU
from operator_agent.exec_guard import run_candidate
from operator_agent.input_fidelity import (
    InputFidelityError, encode_inputs, decode_inputs, describe_inputs, check_readback,
)
from operator_agent.schemas import KernelArtifact
from operator_agent.report import build_report
from tests.unit.test_correctness_suite import case, verify, GOOD, CorrectnessSuite


def views():
    base = np.arange(80, dtype=np.float32).reshape(10, 8)
    return [base, base[1:8:2, 1::2], base[::-2, ::-1], base.T,
            np.broadcast_to(base[:3, :1], (3, 7)),
            np.asfortranarray(base), base[0:0, :], np.array(3, dtype=np.int32)]


@pytest.mark.parametrize("index", range(8))
def test_roundtrip_preserves_values_dtype_strides_offset_readonly(index):
    x = views()[index]
    inputs = {"X": x, "eps": np.float32(1e-5), "causal": False, "count": 2}
    wire = pickle.loads(pickle.dumps(encode_inputs(inputs), protocol=4))
    restored = decode_inputs(wire)
    np.testing.assert_array_equal(restored["X"], x)
    assert describe_inputs(restored) == describe_inputs(inputs)
    assert restored["X"].strides == x.strides
    assert restored["X"].dtype == x.dtype
    assert type(restored["eps"]) is np.float32
    assert restored["causal"] is False


def test_shared_storage_is_not_duplicated_or_dealiased():
    base = np.arange(20, dtype=np.float32)
    inputs = {"X": base[1::2], "Y": base[3::2]}
    wire = encode_inputs(inputs)
    assert len(wire["storages"]) == 1
    restored = decode_inputs(wire)
    assert np.shares_memory(restored["X"], restored["Y"])
    restored["X"][1] = 999
    assert restored["Y"][0] == 999
    assert base[3] != 999  # Child copy must not mutate controller inputs.


@pytest.mark.parametrize("change", ["dtype", "strides", "offset", "storage", "writeable"])
def test_invalid_view_descriptors_fail_closed(change):
    packet = encode_inputs({"X": np.ones((2, 3), dtype=np.float32)})
    bad = {"dtype": "O", "strides": [12, 100000], "offset": -1,
           "storage": True, "writeable": 1}
    packet["tensors"]["X"][change] = bad[change]
    with pytest.raises(InputFidelityError):
        decode_inputs(packet)


def test_encode_rejects_object_subclass_and_unknown_backing():
    class ArraySubclass(np.ndarray):
        pass
    for array in [np.array([object()]), np.ones(2).view(ArraySubclass),
                  np.lib.stride_tricks.as_strided(np.ones(2), shape=(50,), strides=(8,))]:
        with pytest.raises(InputFidelityError):
            encode_inputs({"X": array})


def test_storage_and_logical_size_caps_are_checked(monkeypatch):
    import operator_agent.input_fidelity as fidelity
    monkeypatch.setattr(fidelity, "MAX_STORAGE_BYTES", 32)
    with pytest.raises(InputFidelityError, match="上限"):
        encode_inputs({"X": np.ones(100, dtype=np.float32)[:1]})
    packet = encode_inputs({"X": np.ones(1, dtype=np.float32)})
    packet["tensors"]["X"].update(shape=[100], strides=[0])
    with pytest.raises(InputFidelityError, match="上限"):
        decode_inputs(packet)


def test_seed_path_can_opt_into_a_larger_explicit_limit(monkeypatch):
    import operator_agent.input_fidelity as fidelity
    monkeypatch.setattr(fidelity, "MAX_STORAGE_BYTES", 32)
    monkeypatch.setattr(fidelity, "MAX_SEED_STORAGE_BYTES", 128)
    inputs = {"X": np.ones(20, dtype=np.float32)}      # 80 B: 过 seed 上限, 不过 JSON 上限
    with pytest.raises(InputFidelityError, match="上限"):
        describe_inputs(inputs)
    observed = describe_inputs(inputs, limit=fidelity.MAX_SEED_STORAGE_BYTES)
    assert observed["X"]["storage_bytes"] == 80


@pytest.mark.parametrize("executor", ["fork", "subprocess"])
@pytest.mark.parametrize("layout", ["strided", "broadcast"])
def test_guard_child_receipt_describes_actual_inputs_in_both_executors(executor, layout):
    inputs = case(layout=layout).materialize()
    result = run_candidate(GOOD, inputs, preserve_inputs=True, executor=executor)
    assert result.ok, result.detail
    evidence = result.metrics["input_fidelity"]
    assert check_readback(describe_inputs(inputs), evidence)["input_fidelity_verified"] is True
    np.testing.assert_allclose(result.output, inputs["A"] @ inputs["B"])
    observed = evidence["observed"]["A"]
    assert observed["strides"][-1] == (0 if layout == "broadcast" else 8)


def test_child_receipt_is_taken_before_candidate_imports_or_mutation():
    inputs = {"X": np.arange(10, dtype=np.float32)[1::2]}
    expected = describe_inputs(inputs)
    result = run_candidate("def kernel(X):\n    X[:] = 0\n    return X\n",
                           inputs, preserve_inputs=True)
    assert result.ok
    assert result.metrics["input_fidelity"]["observed"] == expected
    assert not result.output.any()
    assert inputs["X"].any()


@pytest.mark.parametrize("layout", ["strided", "broadcast"])
def test_correct_layout_kernel_passes_suite_and_can_reach_l3(layout):
    results, verifier = verify(suite=CorrectnessSuite((case(layout=layout),)))
    row = results[1].metrics["correctness"]["cases"][0]
    assert row["verdict"] == "passed"
    assert row["metrics"]["input_fidelity_verified"] is True
    assert row["repeat_metrics"]["input_fidelity_verified"] is True
    assert row["metrics"]["input_fidelity"]["observed"]["A"]["strides"][-1] == (
        0 if layout == "broadcast" else 8)
    verifier._verify_performance.assert_called_once()


@pytest.mark.parametrize("layout", ["strided", "broadcast"])
def test_dense_indexing_of_strided_or_broadcast_input_is_rejected(layout):
    code = ("import numpy as np\ndef kernel(A, B):\n"
            "    a = np.ndarray(A.shape, dtype=A.dtype, buffer=A.base)\n"
            "    b = np.ndarray(B.shape, dtype=B.dtype, buffer=B.base)\n"
            "    return a @ b\n")
    results, verifier = verify(code=code, suite=CorrectnessSuite((case(layout=layout),)))
    row = results[-1].metrics["correctness"]["cases"][0]
    assert row["verdict"] == "failed"
    # Fidelity passed; the candidate is genuinely wrong, not transport-unknown.
    assert row["metrics"]["input_fidelity_verified"] is True
    verifier._verify_performance.assert_not_called()


@pytest.mark.parametrize("change", ["value_digest", "dtype", "strides", "offset", "writeable"])
def test_altered_child_evidence_cannot_claim_correctness(change, monkeypatch):
    original = MockGPU.execute
    def tampered(self, code, inputs):
        result = original(self, code, inputs)
        evidence = copy.deepcopy(result.metrics["input_fidelity"])
        evidence["observed"]["A"][change] = "changed"
        result.metrics["input_fidelity"] = evidence
        return result
    monkeypatch.setattr(MockGPU, "execute", tampered)
    results, verifier = verify(suite=CorrectnessSuite((case(layout="strided"),)))
    assert results[-1].metrics["correctness"]["verdict"] == "unknown"
    assert results[-1].metrics["input_fidelity_verified"] is False
    verifier._verify_performance.assert_not_called()


def test_correct_output_for_wrong_layout_is_not_fidelity_evidence(monkeypatch):
    original = MockGPU.execute
    def densify(self, code, inputs):
        return original(self, code, {k: np.ascontiguousarray(v) for k, v in inputs.items()})
    monkeypatch.setattr(MockGPU, "execute", densify)
    results, verifier = verify(suite=CorrectnessSuite((case(layout="strided"),)))
    row = results[-1].metrics["correctness"]["cases"][0]
    assert row["verdict"] == "unknown"
    assert row["metrics"]["input_fidelity_verified"] is False
    verifier._verify_performance.assert_not_called()


def test_missing_or_incompatible_receipt_is_unobserved():
    expected = describe_inputs({"X": np.ones(2)})
    for evidence in [None, {}, {"protocol": "future"}, {"protocol": "numpy-storage-v1"}]:
        assert check_readback(expected, evidence)["input_fidelity_verified"] is None


def test_layout_evidence_does_not_accept_bool_or_float_for_integer_fields():
    expected = describe_inputs({"X": np.ones(2)})
    for value in (False, 0.0):
        actual = copy.deepcopy(expected)
        actual["X"]["offset"] = value
        assert check_readback(expected, {"protocol": "numpy-storage-v1", "observed": actual})[
            "input_fidelity_verified"] is False


def test_nonzero_offset_and_negative_stride_reach_real_candidate_and_reference():
    a = np.arange(144, dtype=np.float32).reshape(12, 12)[1:9, 9:1:-1]
    b = np.arange(144, dtype=np.float32).reshape(12, 12)[1:9, 1:9]
    results, verifier = verify(suite=CorrectnessSuite((case(),)), primary={"A": a, "B": b})
    assert results[1].passed
    fidelity = results[1].metrics["input_fidelity_readings"][0]
    assert fidelity["input_fidelity_verified"] is True
    seen = fidelity["input_fidelity"]["observed"]["A"]
    assert seen["offset"] == (12 + 9) * 4
    assert seen["strides"] == [48, -4]
    verifier._verify_performance.assert_called_once()


def test_expected_values_are_snapshotted_before_backend_call(monkeypatch):
    original = MockGPU.execute
    def mutate(self, code, inputs):
        inputs["A"][0, 0] += 1
        return original(self, code, inputs)
    monkeypatch.setattr(MockGPU, "execute", mutate)
    results, verifier = verify(suite=CorrectnessSuite((case(),)))
    assert results[-1].metrics["correctness"]["verdict"] == "unknown"
    assert results[-1].metrics["input_fidelity_verified"] is False
    verifier._verify_performance.assert_not_called()


def test_report_contains_case_level_runtime_input_evidence():
    results, _ = verify(suite=CorrectnessSuite((case(layout="strided"),)))
    report = build_report(artifact=KernelArtifact(verification_results=results))
    assert "输入保真" in report
    assert "offset" in report and "strides" in report
