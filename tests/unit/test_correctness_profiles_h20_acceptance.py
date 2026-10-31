"""Offline calibration of the acceptance driver, never real-device evidence."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.drivers import correctness_profiles_h20 as d
from operator_agent.decide.anticheat import static_audit
from operator_agent.decide.correctness import as_suite
from operator_agent.decide.correctness_profiles import default_policy
from operator_agent.core.schemas import KernelArtifact, VerificationResult, VerificationLevel as Level


def evidence():
    return dict(input_fidelity_verified=True, input_fidelity=dict(representation="torch-storage-v1",
        observed={"X":{"kind":"tensor"}},tensor_devices={"X":"cuda:0"}), output_digest="digest", max_error=0)


def fixture(expected="passed"):
    suite = as_suite({"coverage_policy": default_policy("relu","fp32").content_dict()})
    rows = [dict(case_id=c.case_id,verdict="passed",determinism="passed",metrics=evidence(),repeat_metrics=evidence()) for c in suite.cases]
    if expected == "failed": rows[0]["verdict"] = "failed"
    c = dict(verdict=expected,suite_hash=suite.digest(),coverage_lint={"status":"passed"},cases=rows)
    metrics = {**evidence(),"correctness":c}
    a = KernelArtifact(verification_results=[VerificationResult(Level.L2_FUNCTION,expected=="passed",metrics=metrics)])
    trace = []
    if expected == "passed":
        a.verification_results.append(VerificationResult(Level.L3_PERFORMANCE,False,
            metrics={"backend_real_device":True,"measurement_validity":"uninformative"}))
        trace=[dict(request={"action":"benchmark_with_baselines"},response={"success":True,"device":"cuda"})]
    return a,suite,trace


def test_all_fixed_candidates_compile_and_use_masked_safe_mutations():
    specs = d.trial_specs()
    assert len(specs) == 23
    assert len([s for s in specs if s[2]=="good"]) == 7
    for op,dt,variant in specs:
        code = d.candidate(op,variant)
        compile(code,"<acceptance>","exec")
        assert static_audit(code,"triton").violations == []
        inputs = d.primary(op,dt)
        assert sum(a.nbytes for a in inputs.values()) <= 32*1024
        assert all(a.flags.c_contiguous and np.isfinite(a).all() for a in inputs.values())
        if variant != "good": assert code != d.candidate(op,"good")


@pytest.mark.parametrize("expected",("passed","failed"))
def test_assess_accepts_full_correctness_even_when_perf_honestly_refused(expected):
    a,s,t = fixture(expected)
    assert d.assess(a,expected,s,t,False) == []


@pytest.mark.parametrize("mutation",("hash","coverage","missing","repeat","cpu","no_l3","bad_l3","primary"))
def test_assess_refuses_incomplete_evidence(mutation):
    a,s,t=fixture()
    c=a.verification_results[0].metrics["correctness"]
    if mutation=="hash": c["suite_hash"]="wrong"
    if mutation=="coverage": c["coverage_lint"]["status"]="missing"
    if mutation=="missing": c["cases"].pop()
    if mutation=="repeat": c["cases"][0]["repeat_metrics"]["output_digest"]="different"
    if mutation=="cpu": c["cases"][0]["metrics"]["input_fidelity"]["tensor_devices"]["X"]="cpu"
    if mutation=="no_l3": t=[]
    if mutation=="bad_l3": t[0]["response"]["success"]=False
    if mutation=="primary": a.verification_results[0].metrics["max_error"]=1
    assert d.assess(a,"passed",s,t,False)


@pytest.mark.parametrize("mutation",("l3","db","not_killed"))
def test_assess_bad_kernel_cannot_get_performance_or_persist(mutation):
    a,s,t=fixture("failed")
    if mutation=="l3": t=[{"request":{"action":"benchmark_with_baselines"}}]
    if mutation=="not_killed": a.verification_results[0].metrics["correctness"]["cases"][0]["verdict"]="not_run"
    assert d.assess(a,"failed",s,t,mutation=="db")


def test_compact_trace_keeps_wire_identity_without_mutating_request():
    trace=[dict(request={"action":"verify","inputs":{"storages":["abc123"],"tensors":{},"protocol":"p"}},response={"ok":True})]
    old=copy.deepcopy(trace)
    compact=d.compact_trace(trace)
    assert trace==old
    assert compact[0]["request"]["inputs"]["storages"][0]["encoded_bytes"]==6
    assert len(compact[0]["request"]["inputs"]["storages"][0]["encoded_sha256"])==64


def test_driver_prepare_failure_still_saves_failure_and_cleans_up(tmp_path,monkeypatch):
    class Backend:
        forbidden_actions=[]
        def __init__(self,**kwargs): pass
        def prepare(self,*args): raise RuntimeError("offline unavailable")
        def cleanup(self): return {"quiescent":True,"removed":True}
    monkeypatch.setattr(d,"ProfileSSH",Backend)
    # run redirects logging intentionally; don't leak that into the test session.
    import os
    monkeypatch.setenv("OPERATOR_AGENT_LOG_DIR",os.environ["OPERATOR_AGENT_LOG_DIR"])
    monkeypatch.setenv("OPERATOR_AGENT_LOG_KEEP",os.environ["OPERATOR_AGENT_LOG_KEEP"])
    assert d.run(SimpleNamespace(output=str(tmp_path),host=None,gpu=2))==1
    s=json.loads(next(tmp_path.rglob("summary.json")).read_text())
    assert s["accepted"] is False and s["cleanup"]["removed"] is True
    assert s["trials"][0]["name"]=="driver-error"
