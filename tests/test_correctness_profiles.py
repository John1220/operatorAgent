"""Trusted profile adoption, legacy compatibility and closed production gates."""
import dataclasses
import json
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent import (Conductor, IntentParser, MockGPU, TestInputGenerator,
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec, Verifier, with_default_correctness)
from operator_agent.correctness import as_suite, credential
from operator_agent.correctness_profiles import PROFILE_VERSION, default_policy, profile_check
from operator_agent.contract import ContractError, freeze_from_task, load_dict
from operator_agent.failure_route import classify, ROUTE_HOLD
from operator_agent.schemas import Precision, UnifiedSketch, VerificationLevel as Level, VerificationResult

GOOD = {
    "relu": "import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n",
    "matmul": "def kernel(A,B):\n    return A @ B\n",
    "softmax": "import numpy as np\ndef kernel(X):\n    z=X.astype(np.float32)\n    e=np.exp(z-np.max(z,axis=-1,keepdims=True))\n    return (e/e.sum(axis=-1,keepdims=True)).astype(X.dtype)\n",
    "layernorm": "import numpy as np\ndef kernel(X):\n    z=X.astype(np.float32)\n    return ((z-z.mean(axis=-1,keepdims=True))/np.sqrt(z.var(axis=-1,keepdims=True)+1e-5)).astype(X.dtype)\n",
}


def task(op="relu", dtype="fp32"):
    return with_default_correctness(TaskSchema(source=SourceSpec(kernel_type=op),
        target=TargetSpec(dsl="python"), constraints=ConstraintSpec(precision=Precision(dtype))))


@pytest.mark.parametrize("op,dtype", [(op, dt) for op in GOOD for dt in ("fp16", "fp32")
                                     if (op, dt) != ("layernorm", "fp16")])
def test_formal_rule_entry_generates_target_precision_finite_profile(op, dtype):
    t = IntentParser().parse(f"{op} {dtype} python mock")
    assert t.constraints.correctness_profile == PROFILE_VERSION
    suite = as_suite(t.constraints.correctness_suite)
    assert suite.coverage_policy.dtypes == (dtype,)
    assert suite.coverage_policy.patterns == tuple(sorted(("random", "zeros", "constant", "large", "small")))
    inputs = TestInputGenerator.generate(t, matrix_dim=(8, 8), seed=11)
    assert set(inputs) == ({"A", "B"} if op == "matmul" else {"X"})
    assert profile_check(t, inputs)["status"] == "passed"
    v = Verifier(MockGPU())
    r = v.verify(GOOD[op], t, inputs, stop_after=Level.L2_FUNCTION)[-1]
    assert r.passed, r.metrics
    assert r.metrics["correctness"]["verdict"] == "passed"
    assert r.metrics["correctness"]["profile"]["status"] == "passed"
    assert r.metrics["correctness"]["coverage_lint"]["status"] == "passed"


def test_llm_cannot_supply_policy_or_disable_the_formal_default():
    llm = Mock()
    llm.chat_json.return_value = dict(kernel_type="softmax", precision="fp32", dsl="python",
        hardware="mock_gpu", complexity="low", correctness_profile=None,
        correctness_suite={"cases": []})
    t = IntentParser(llm=llm).parse("request")
    assert profile_check(t)["status"] == "passed"
    assert t.constraints.correctness_profile == PROFILE_VERSION


@pytest.mark.parametrize("mutation", ("suite", "policy", "pattern", "deterministic", "version", "source", "fusion", "reference", "code_path", "framework"))
def test_requirement_cannot_fall_back_to_legacy_on_mutation(mutation, tmp_path):
    t = task()
    if mutation == "suite": t.constraints.correctness_suite = None
    if mutation == "policy": t.constraints.correctness_suite = as_suite(t.constraints.correctness_suite).content_dict(); t.constraints.correctness_suite.pop("coverage_policy")
    if mutation == "pattern": t.constraints.correctness_suite["coverage_policy"]["patterns"] = ["random"]
    if mutation == "deterministic": t.constraints.correctness_suite["deterministic"] = False
    if mutation == "version": t.constraints.correctness_profile = "builtin-l2-v99"
    if mutation == "source": t.source.kernel_type = "log_softmax"
    if mutation == "fusion": t.source.fused_ops = ["add"]
    if mutation == "reference": t.source.reference_source = "def reference(X): return X"
    if mutation == "code_path": t.source.code_path = "custom.py"
    if mutation == "framework": t.source.framework = "tensorflow"
    v = Verifier(MockGPU())
    v._verify_compile = Mock(side_effect=AssertionError("compiled"))
    v._verify_performance = Mock(side_effect=AssertionError("benchmarked"))
    rs = v.verify(GOOD["relu"], t, {"X": np.ones((8, 8), np.float32)})
    assert rs[0].metrics["correctness"]["verdict"] == "unknown"
    assert credential(rs) is None
    assert classify(rs).route == ROUTE_HOLD
    v.verify_l3_only(GOOD["relu"], {}, 0, t, correctness_verified=True)
    assert v._verify_function(GOOD["relu"], {}, None, t).metrics["correctness"]["verdict"] == "unknown"
    v._verify_compile.assert_not_called(); v._verify_performance.assert_not_called()
    with pytest.raises(ContractError): freeze_from_task(t)
    c = Conductor(backend=MockGPU(), llm=Mock())
    c.designer.run = Mock(side_effect=AssertionError("LLM"))
    a = c.execute(t, {"X": np.ones((8, 8), np.float32)})
    assert a.verification_results[0].metrics["correctness"]["verdict"] == "unknown"
    c.designer.run.assert_not_called()


@pytest.mark.parametrize("op,dtype", [("attention","fp32"),("log_softmax","fp32"),("gemm","fp32"),("conv2d","fp32"),("relu","bf16"),("matmul","int8"),("layernorm","fp16")])
def test_unsupported_is_unknown_not_an_empty_passing_suite(op, dtype):
    t = task(op, dtype)
    if (op, dtype) == ("layernorm", "fp16"):
        t.constraints.correctness_profile = PROFILE_VERSION
        t.constraints.correctness_suite = None
        t.constraints.reference_contract = None
    assert t.constraints.correctness_suite is None
    r = Verifier.correctness_preflight(t)
    assert r.metrics["correctness"]["verdict"] == "unknown"


@pytest.mark.parametrize("inputs", [
    {"input": np.ones((8,8), np.float32)}, {"X": np.ones(8, np.float32)},
    {"X": np.ones((8,8), np.float16)}, {"X": np.ones((0,8), np.float32)},
    {"X": np.ones((8,8), np.float32), "eps": 1e-2},
    {"X": np.ones((8,8), np.float32), "weight": np.ones(8, np.float32)}, []])
def test_unknown_primary_semantics_stop_before_backend(inputs):
    t = task("layernorm")
    v = Verifier(MockGPU()); v._verify_compile = Mock(side_effect=AssertionError())
    r = v.verify(GOOD["layernorm"], t, inputs)[0]
    assert r.metrics["correctness"]["verdict"] == "unknown"
    v._verify_compile.assert_not_called()


def test_primary_domain_is_not_shrunk_to_the_small_sampling_box():
    t = task("matmul")
    assert profile_check(t, {"A": np.zeros((40,128),np.float32), "B":np.zeros((128,67),np.float32)})["status"] == "passed"
    assert profile_check(t, {"A": np.zeros((40,128),np.float32), "B":np.zeros((4,67),np.float32)})["status"] == "unknown"


@pytest.mark.parametrize("op",tuple(GOOD))
def test_v1_sampling_domains_and_witnesses_are_explicit_not_candidate_derived(op):
    p=default_policy(op,"fp32")
    axes={a.name:(a.minimum,a.maximum,a.anchor,a.alignment) for a in p.axes}
    if op=="matmul":
        assert axes=={"m":(1,33,32,32),"n":(1,33,32,32),"k":(1,5,4,4)}
        assert {t.name:t.axes for t in p.tensors}=={"A":("m","k"),"B":("k","n")}
    else:
        assert axes=={"rows":(1,9,8,8),"cols":(1,33,32,32)}
        assert [(t.name,t.axes) for t in p.tensors]==[("X",("rows","cols"))]
    s=as_suite({"coverage_policy":p.content_dict()})
    shapes={tuple((t.name,t.shape) for t in c.inputs.tensors) for c in s.cases
            if c.pattern=="random" and c.layout=="contiguous"}
    if op=="matmul":
        for k in (1,3,4,5): assert (("A",(32,k)),("B",(k,32))) in shapes
    else:
        for n in (1,31,32,33): assert (("X",(8,n)),) in shapes


def test_profile_roundtrip_hash_and_legacy_unchanged():
    t = task(); c = freeze_from_task(t)
    assert c.correctness_profile == PROFILE_VERSION
    assert load_dict(json.loads(json.dumps(c.as_dict()))).hash == c.hash
    assert t.to_dict()["constraints"]["correctness_profile"] == PROFILE_VERSION
    legacy = dataclasses.replace(t, constraints=dataclasses.replace(t.constraints, correctness_profile=None))
    assert freeze_from_task(legacy).hash != c.hash
    raw = c.as_dict(); raw["correctness_profile"] = "unknown-version"
    with pytest.raises(ContractError): load_dict(raw)
    old = TaskSchema()
    assert "correctness_profile" not in old.to_dict()["constraints"]
    assert "correctness_profile" not in freeze_from_task(old).content_dict()
    assert Verifier.correctness_preflight(old) is None


def test_explicit_suite_and_original_task_are_not_overwritten():
    old = TaskSchema(source=SourceSpec(kernel_type="relu"))
    new = with_default_correctness(old)
    assert old.constraints.correctness_suite is None and old.constraints.correctness_profile is None
    assert new.task_id == old.task_id
    explicit = dataclasses.replace(old, constraints=dataclasses.replace(old.constraints,
        correctness_suite=new.constraints.correctness_suite))
    assert with_default_correctness(explicit) is explicit


@pytest.mark.parametrize("attribute,value", [("code_path","custom.py"),("framework","tensorflow"),
                                            ("fused_ops",["add"]),("reference_source","def reference(X): return X"),
                                            ("semantic_parameters",{"eps":1e-3})])
def test_unknown_custom_semantics_are_not_filled_from_the_builtin_name(attribute,value):
    old = TaskSchema(source=SourceSpec(kernel_type="relu"))
    setattr(old.source, attribute, value)
    new = with_default_correctness(old)
    assert new.constraints.correctness_profile == PROFILE_VERSION
    assert new.constraints.correctness_suite is None
    assert profile_check(new)["status"] == "unknown"


@pytest.mark.parametrize("prompt",("layernorm eps=1e-3","layernorm affine=True","softmax dim=0",
                                    "matmul transpose B","fused matmul relu","matmul+relu",
                                    "log_softmax","custom_softmax"))
@pytest.mark.parametrize("use_llm",(False,True))
def test_explicit_nondefault_requests_cannot_be_erased_by_llm_or_rules(prompt,use_llm):
    llm=Mock() if use_llm else None
    if llm:
        llm.chat_json.return_value=dict(kernel_type="softmax" if prompt=="custom_softmax" else "layernorm",
                                      precision="fp32",dsl="python",complexity="low")
    t=IntentParser(llm=llm).parse(prompt)
    # A generic custom name cannot be semantically resolved by keyword guessing.
    # In LLM mode the operator label is its interpretation, not proof; custom
    # names are conservatively retained as an unresolved semantic requirement.
    assert Verifier.correctness_preflight(t).metrics["correctness"]["verdict"]=="unknown"
    assert "correctness_profile" in t.to_dict()["constraints"]


def test_unknown_profile_report_names_reason_and_default_requirement():
    from operator_agent.report import build_report
    from operator_agent.schemas import KernelArtifact
    t=task("attention")
    a=KernelArtifact(verification_results=[Verifier.correctness_preflight(t)])
    report=build_report(artifact=a,task=t)
    assert "unknown" in report and "builtin-l2-v1" in report and "attention" in report


@pytest.mark.parametrize("op", tuple(GOOD))
def test_conductor_defaults_run_and_save_suite_not_performance_workload(op, tmp_path):
    t = task(op); inputs = TestInputGenerator.generate(t, (8,8), seed=11)
    from operator_agent.artifact_store import ArtifactStore
    c = Conductor(backend=MockGPU(), max_optimize_rounds=1, use_coder_pipeline=False)
    c.store = ArtifactStore(str(tmp_path))
    c.designer.run = lambda *a: UnifiedSketch(algorithm=op, target_dsl="python")
    c.coder.run = lambda *a, **k: GOOD[op]
    c.verifier._verify_performance = Mock(return_value=VerificationResult(Level.L3_PERFORMANCE, True))
    a = c.execute(t, inputs)
    assert next(r for r in a.verification_results if r.level == Level.L2_FUNCTION).passed
    args = c.verifier._verify_performance.call_args_list[0].args
    assert args[1] is inputs or (isinstance(args[1], list) and args[1][0] is inputs)
    result = json.loads(next(tmp_path.rglob("result.json")).read_text())
    assert result["task"]["constraints"]["correctness_profile"] == PROFILE_VERSION
    assert as_suite(result["task"]["constraints"]["correctness_suite"]).digest() == as_suite(t.constraints.correctness_suite).digest()
    assert "builtin-l2-v1" in next(tmp_path.rglob("report.md")).read_text()
