"""Offline coverage of bounded H20 workload selection; never connects to SSH."""
import pytest

from benchmarks.durable_profile_h20 import WORKLOADS, parser, receipt_errors, selected_workloads
from operator_agent.reference_contract import check_observed, registered_reference


def test_original_default_and_resume_only_workloads_are_unchanged():
    assert selected_workloads(parser().parse_args([])) == WORKLOADS
    assert selected_workloads(parser().parse_args(["--resume-only"])) == WORKLOADS[:1]


def test_explicit_workloads_are_fixed_in_declared_order():
    args = parser().parse_args(["--workload", "1024x1024", "--workload", "8192x8192"])
    assert selected_workloads(args) == ((1024, 1024), (8192, 8192))
    # A first workload other than the old default must also be the resume input.
    assert selected_workloads(parser().parse_args(["--workload", "8192x8192"]))[0] == (8192, 8192)


@pytest.mark.parametrize("argv", [["--workload", "1000000x1000000"],
                                  ["--resume-only", "--workload", "8192x8192"]])
def test_unbounded_or_ambiguous_workload_selection_is_rejected(argv):
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(argv)
    assert exc.value.code == 2


def test_missing_receipt_is_a_failed_trial_not_an_exception_that_discards_evidence():
    trace = [{"request": {"action": "verify"}, "response": {
        "transport_outcome": "outcome_unknown", "session_state": "timeout", "session_phase": "send"}},
        {"request": {"action": "compile"}, "response": {"success": True}}]
    errors = receipt_errors(registered_reference(), trace)
    assert len(errors) == 1 and "verify reference identity unknown" in errors[0]
    assert trace[0]["response"]["transport_outcome"] == "outcome_unknown"


def test_valid_receipt_still_required_for_each_numeric_rpc():
    # Fabricated offline protocol fixture only, never CUDA/performance evidence.
    reference = registered_reference()
    receipt = check_observed(reference, reference.execution.content_dict())
    trace = [{"request": {"action": action}, "response": {"reference_receipt": receipt}}
             for action in ("verify", "benchmark_with_baselines")]
    assert receipt_errors(reference, trace) == []
    trace[1]["response"]["reference_receipt"] = {"status": "passed"}
    assert len(receipt_errors(reference, trace)) == 1
