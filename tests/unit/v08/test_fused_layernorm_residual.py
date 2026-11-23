"""Fused residual+LayerNorm profile: suite, declaration routing, reference identity.

This profile is a *new* registered family, not a LayerNorm v2 variant:
- two same-shape fp16 inputs X/R, output Y = layer_norm(X + R);
- explicit 32-case suite (optimizer/reference calibration owns the tolerance);
- its own reference primitive file and pinned SHA256;
- kernel_type routing must beat the generic ``layernorm`` keyword.
"""
import hashlib
from pathlib import Path

import pytest

from operator_agent.app.hitl_tools import build_task
from operator_agent.app.intent_parser import IntentParser
from operator_agent.core.contract import freeze_from_task
from operator_agent.decide import correctness_profiles as cp
from operator_agent.decide import reference_contract as rc
from operator_agent.decide.bench_cases import cases_from_shapes, known_layouts
from operator_agent.decide.correctness import as_suite

DECLARATION = dict(operator='fused_layernorm_residual', request="fp16 32768x6144 residual add layernorm",
                   shapes=[[32768, 6144]], precision="fp16", dsl="triton",
                   hardware="nvidia_gpu", min_speedup=1.0)


def test_suite_has_32_explicit_cases_with_two_inputs():
    suite = cp.fused_ln_residual_suite()
    assert len(suite.cases) == 32
    assert suite.deterministic is True
    for case in suite.cases:
        assert sorted(t.name for t in case.inputs.tensors) == ["R", "X"]
        assert case.inputs.tensors[0].shape == case.inputs.tensors[1].shape
        assert {t.dtype for t in case.inputs.tensors} == {"fp16"}
        assert case.pattern in {"random", "zeros", "constant", "large", "small",
                                "constant_nonbinary", "near_constant", "large_varying",
                                "small_varying", "signed_large"}
    ids = {case.case_id for case in suite.cases}
    assert "fused-ln-residual-8x33-near_constant-strided" in ids
    assert "fused-ln-residual-8x1-zeros-strided" in ids


def test_suite_digest_is_stable_and_different_from_layernorm_v2():
    first = cp.fused_ln_residual_suite().digest()
    assert cp.fused_ln_residual_suite().digest() == first
    assert first != cp.v2_suite().digest()


def test_materialized_case_keeps_two_independent_inputs():
    case = next(c for c in cp.fused_ln_residual_suite().cases
                if c.case_id == "fused-ln-residual-8x33-random-contiguous")
    arrays = case.materialize()
    assert set(arrays) == {"X", "R"}
    assert arrays["X"].shape == arrays["R"].shape == (8, 33)
    assert arrays["X"].dtype.name == "float16"
    # same seed + different tensor name must not collapse R into X
    assert not (arrays["X"] == arrays["R"]).all()


def test_request_routes_to_fused_family_not_plain_layernorm():
    parses = IntentParser()
    fused = parses.parse("fp16 32768x6144 residual add layernorm")
    assert fused.source.kernel_type == "fused_layernorm_residual"
    assert fused.source.fused_ops == []          # registered semantics, not unresolved
    plain = parses.parse("fp16 32768x6144 layernorm")
    assert plain.source.kernel_type == "layernorm"
    unresolved = parses.parse("fp16 32768x6144 fused relu layernorm")
    assert unresolved.source.fused_ops == ["unresolved_fusion"]


def test_build_task_attaches_registered_profile_and_reference():
    task = build_task(DECLARATION)
    c = task.constraints
    assert task.source.kernel_type == "fused_layernorm_residual"
    assert c.correctness_profile == rc.PROFILE_FUSED_LN_RESIDUAL
    assert c.max_error == rc.REGISTERED_ATOL[rc.PROFILE_FUSED_LN_RESIDUAL]
    assert c.reference_contract["profile"] == rc.PROFILE_FUSED_LN_RESIDUAL
    assert as_suite(c.correctness_suite).digest() == cp.fused_ln_residual_suite().digest()
    assert c.reference_contract["definition"].startswith("native-pytorch-residual-add-layernorm")


def test_profile_check_accepts_same_shape_fp16_pair_and_rejects_mismatch():
    task = build_task(DECLARATION)
    import numpy as np
    good = {"X": np.zeros((8, 33), np.float16), "R": np.zeros((8, 33), np.float16)}
    assert cp.profile_check(task, [good])["status"] == "passed"
    bad = {"X": np.zeros((8, 33), np.float16), "R": np.zeros((8, 31), np.float16)}
    assert cp.profile_check(task, [bad])["status"] == "unknown"


def test_declaration_check_rejects_suite_edits_and_wrong_reference():
    task = build_task(DECLARATION)
    c = task.constraints
    suite = cp.fused_ln_residual_suite()
    check = cp.declaration_check("fused_layernorm_residual", "fp16",
                                 rc.PROFILE_FUSED_LN_RESIDUAL, suite.content_dict(),
                                 c.reference_contract)
    assert check["status"] == "passed"
    edited = cp.fused_ln_residual_suite().cases[:-1]
    from operator_agent.decide.correctness import CorrectnessSuite
    check = cp.declaration_check("fused_layernorm_residual", "fp16",
                                 rc.PROFILE_FUSED_LN_RESIDUAL,
                                 CorrectnessSuite(edited).content_dict(),
                                 c.reference_contract)
    assert check["status"] == "unknown"
    v2_reference = rc.registered_reference(rc.PROFILE_V2).content_dict()
    check = cp.declaration_check("fused_layernorm_residual", "fp16",
                                 rc.PROFILE_FUSED_LN_RESIDUAL, suite.content_dict(),
                                 v2_reference)
    assert check["status"] == "unknown"


def test_fused_reference_identity_is_pinned_and_distinct_from_v2():
    fused = rc.registered_reference(rc.PROFILE_FUSED_LN_RESIDUAL)
    layernorm = rc.registered_reference(rc.PROFILE_V2)
    primitive = (Path(rc.__file__).resolve().parent / "builtin_fused_layernorm_residual.py")
    assert hashlib.sha256(primitive.read_bytes()).hexdigest() == \
        rc.FUSED_LN_RESIDUAL_PRIMITIVE_SHA256
    assert fused.execution.primitive_sha256 == rc.FUSED_LN_RESIDUAL_PRIMITIVE_SHA256
    assert fused.execution.primitive_sha256 != layernorm.execution.primitive_sha256
    assert fused.profile == rc.PROFILE_FUSED_LN_RESIDUAL
    assert fused.epsilon == 1e-5 and fused.affine is False
    assert fused.variance == "population" and fused.normalized_axis == -1
    # round-trip through the strict validator
    assert rc.as_reference(fused.content_dict()) == fused
    with pytest.raises(rc.ReferenceIdentityError):
        rc.as_reference(dict(fused.content_dict(), profile="builtin-unknown"))


def test_gate_binds_fused_tolerance_to_the_registered_value():
    task = build_task(DECLARATION)
    contract = task.constraints.reference_contract
    atol = rc.REGISTERED_ATOL[rc.PROFILE_FUSED_LN_RESIDUAL]
    assert rc.validate_gate(contract, "fp16", atol,
                            kernel_type="fused_layernorm_residual").profile == \
        rc.PROFILE_FUSED_LN_RESIDUAL
    with pytest.raises(rc.ReferenceIdentityError):
        rc.validate_gate(contract, "fp16", atol + 1e-3,
                         kernel_type="fused_layernorm_residual")
    with pytest.raises(rc.ReferenceIdentityError):
        rc.validate_gate(contract, "fp16", atol, kernel_type="layernorm")
    with pytest.raises(rc.ReferenceIdentityError):
        rc.validate_gate(contract, "fp32", atol, kernel_type="fused_layernorm_residual")


def test_frozen_contract_owns_an_X_R_performance_case():
    task = build_task(DECLARATION)
    contract = freeze_from_task(task)
    assert "R=32768x6144|X=32768x6144" in {case.key() for case in contract.cases}
    assert contract.correctness_profile == rc.PROFILE_FUSED_LN_RESIDUAL
    assert contract.max_error == rc.REGISTERED_ATOL[rc.PROFILE_FUSED_LN_RESIDUAL]
    assert len(contract.correctness_suite.cases) == 32


def test_shape_layout_table_knows_the_fused_family():
    assert "fused_layernorm_residual" in known_layouts()
    cases = cases_from_shapes("fused_layernorm_residual", [((8, 33), 1.0)], "fp16")
    assert [t.name for t in cases[0].tensors] == ["X", "R"]


def test_capabilities_lists_the_fused_family_not_just_attention():
    from operator_agent.app.hitl_tools import capabilities
    families = capabilities()["correctness_profiles"]["families"]
    fused = families["fused_layernorm_residual"]
    assert fused["profile"] == rc.PROFILE_FUSED_LN_RESIDUAL
    assert fused["correctness_cases"] == 32
    assert "fused_layernorm_residual" in capabilities()["kernel_types_with_input_layout"]
