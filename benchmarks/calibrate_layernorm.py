"""CPU / isolated H20 fp16 LayerNorm reference-semantic calibration.

CPU: python benchmarks/calibrate_layernorm.py
CUDA: python benchmarks/calibrate_layernorm.py --h20 [--gpu N]
No installs, no benchmarks, no tolerance/reference changes. A complete calibration
can establish a BLOCKER: measurement completeness is NOT default-profile support.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from benchmarks import layernorm_numerics as numerics
from benchmarks.correctness_profiles_h20 import candidate
from benchmarks.input_fidelity_h20 import AcceptanceSSH, dumps
from operator_agent.correctness import as_suite
from operator_agent.correctness_boundaries import lint_coverage
from operator_agent.correctness_profiles import default_policy, with_default_correctness
from operator_agent.input_fidelity import describe_inputs, check_readback
from operator_agent.remote_input_fidelity import encode_json_inputs
from operator_agent.remote_executor import SSHExecutor
from operator_agent.run_layout import new_run_root, run_layout, ROLE_HARNESS, ROLE_ORACLE, ROLE_REPORT_WRITER
from operator_agent.schemas import TaskSchema, SourceSpec, ConstraintSpec, Precision
from operator_agent.verifier import Verifier

CANDIDATE_VARIANTS = ("centered-fp32", "unfused-fp32")
CONTROLS = (("dense", "auto-cols32-rows8-fp16-random-strided"),
            ("tail", "auto-cols33-rows8-fp16-random-contiguous"),
            ("fp16-intermediate", "large-signed"))


def proposed_suite():
    # Only a calibration proposal: never declare unsupported fp16 as a v1 profile.
    p = dataclasses.replace(default_policy("layernorm", "fp32"), dtypes=("fp16",))
    return as_suite({"coverage_policy": p.content_dict()})


def fixtures():
    rows = [(c.case_id, "original-boundary", c.materialize()) for c in proposed_suite().cases]

    def append(name, x):
        x = np.asarray(x, dtype=np.float16)
        rows.append((name, "diagnostic-extension", {"X": x.copy()}))
        backing = np.zeros((x.shape[0], x.shape[1] * 2), np.float16)
        backing[:, ::2] = x
        rows.append((name + "-strided", "diagnostic-extension", {"X": backing[:, ::2]}))

    for name, value in (("zero", 0), ("one", 1), ("large", 100),
                        ("small", 1e-4), ("nonbinary", .1)):
        append("single-column-" + name, np.full((8, 1), value, np.float16))
    for name, values in (("constant-nonbinary", [.1]),
                         ("near-constant", [1, 1+2**-10, 1-2**-11]),
                         ("large-varying", [100, 100.0625, 99.9375]),
                         ("small-varying", [0, 1e-4, -1e-4]),
                         ("large-signed", [100, -100, 0])):
        append(name, np.tile(np.resize(np.array(values, np.float16), 33), (8, 1)))
    return rows


def reference_task():
    return TaskSchema(source=SourceSpec(kernel_type="layernorm"),
                      constraints=ConstraintSpec(precision=Precision.FP16))


def assess_measurement(result, inputs, device):
    errors = []
    reading = check_readback(describe_inputs(inputs), result.get("input_fidelity"))
    if reading.get("input_fidelity_verified") is not True:
        errors.append("input fidelity incomplete")
    if result.get("input_fidelity", {}).get("tensor_devices") != {"X": device}:
        errors.append("requested device not observed")
    identity = result.get("reference_identity", {})
    expected_sources = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
        ("layernorm_numerics.py", ROOT / "benchmarks/layernorm_numerics.py"),
        *((n, ROOT / "operator_agent" / n) for n in
          ("tolerance.py", "remote_input_fidelity.py", "input_fidelity.py")))}
    if (identity.get("semantics") != numerics.SEMANTICS or identity.get("device") != device
            or not identity.get("torch") or len(identity.get("provenance", {}).get("source_sha256", "")) != 64
            or result.get("reference_identity_hash") != numerics.digest(identity)):
        errors.append("reference identity incomplete/drifted")
    if identity.get("shared_source_hashes") != expected_sources:
        errors.append("shared measurement sources not the verified deployment")
    if result.get("reference_output_dtype") != "float16" or result.get("reference_repeat_exact") is not True:
        errors.append("native reference dtype/repeat not established")
    arms = result.get("arms", {})
    expected_arms = {"production_eager", "preserved_view_eager", "promoted_eager_fp32", "centered_fp32"}
    if device == "cuda:0":
        expected_arms.add("same_runtime_cpu_eager")
    if set(arms) != expected_arms:
        errors.append("diagnostic arms missing")
    for name, arm in arms.items():
        if arm.get("device") != ("cpu" if name in ("same_runtime_cpu_eager", "centered_fp32") else device):
            errors.append("diagnostic arm device not identified")
        for key in ("vs_production", "vs_rounded_oracle"):
            gate = arm.get(key, {})
            if (gate.get("finite") is not True or gate.get("atol") != 1e-5
                    or gate.get("rtol") != .01 or gate.get("rho") != 1.0
                    or type(gate.get("passed")) is not bool):
                errors.append("unmodified numerical reading missing")
    self_reading = arms.get("production_eager", {}).get("vs_production", {})
    if self_reading.get("passed") is not True or self_reading.get("max_abs") != 0:
        errors.append("reference self-comparison inconsistent")
    oracle = result.get("oracle", {})
    if (oracle.get("unrounded_dtype") != "float64" or oracle.get("rounded_dtype") != "float16"
            or oracle.get("kind") != "independent_centered_numpy_fp64_on_quantized_input"
            or not isinstance(oracle.get("eager64_max_abs"), (int, float))
            or not np.isfinite(oracle["eager64_max_abs"]) or oracle["eager64_max_abs"] > 1e-10
            or not isinstance(oracle.get("quantization_max_abs"), (int, float))
            or not np.isfinite(oracle["quantization_max_abs"])):
        errors.append("quantization/oracle distinction missing")
    return errors


def diagnostic_server():
    source = (ROOT / "remote_scripts/runner_server.py").read_text()
    needle = "result = run_kernel.run_action(req)"
    if source.count(needle) != 1:
        raise RuntimeError("production server dispatch seam changed")
    # Same owner identity / early publication / ping / shutdown / RPC deadlines.
    return source.replace(needle, "result = (__import__('layernorm_numerics').remote_probe(req, run_kernel) "
                          "if action == 'calibrate_layernorm' else run_kernel.run_action(req))")


class CalibrationSSH(AcceptanceSSH):
    def __init__(self, server, **kwargs):
        super().__init__(**kwargs)
        self.server = server

    def _remote_files(self):
        files = super()._remote_files()
        return [(self.server if local.name == "runner_server.py" else local, remote)
                for local, remote in files] + [(ROOT / "benchmarks/layernorm_numerics.py",
                                                self.scratch + "/layernorm_numerics.py")]

    def _ssh_exec_json(self, payload):
        if payload.get("action") == "calibrate_layernorm":
            return SSHExecutor._ssh_exec_json(self, payload)
        return super()._ssh_exec_json(payload)


def summarize(rows, expected_ids):
    ids = [r.get("case_id") for r in rows]
    complete = (len(ids) == len(expected_ids) and set(ids) == set(expected_ids)
                and bool(ids) and all(r.get("errors") == [] and r.get("measurement") for r in rows))
    blockers = [r["case_id"] for r in rows if not r.get("measurement", {}).get(
        "arms", {}).get("centered_fp32", {}).get("vs_production", {}).get("passed", False)]
    cpu_disagreement = [r["case_id"] for r in rows if r.get("measurement", {}).get(
        "arms", {}).get("same_runtime_cpu_eager", {}).get("vs_production", {}).get("passed") is False]
    return dict(measurement_complete=complete, centered_fp32_vs_eager_failures=blockers,
                same_runtime_cpu_vs_cuda_failures=cpu_disagreement,
                default_profile="unknown", profile_enabled=False,
                decision="keep_v1_disabled_no_reference_or_tolerance_change")


def assess_candidate(result, inputs):
    errors = []
    if (check_readback(describe_inputs(inputs), result.get("input_fidelity")).get(
            "input_fidelity_verified") is not True
            or result.get("input_fidelity", {}).get("tensor_devices") != {"X": "cuda:0"}):
        errors.append("candidate CUDA fidelity incomplete")
    if (result.get("reference_type") != "pytorch_eager" or result.get("coverage") != 1.0
            or not result.get("output_digest") or type(result.get("success")) is not bool
            or result.get("atol") != 1e-5 or result.get("rtol") != .01
            or result.get("required_matched_ratio") != 1.0):
        errors.append("candidate production L2 reading incomplete")
    return errors


def candidate_source(variant):
    code = candidate("layernorm", "good")
    if variant == "centered-fp32":
        return code
    if variant == "unfused-fp32":
        return code.replace("triton.next_power_of_2(X.shape[1]))",
                            "triton.next_power_of_2(X.shape[1]),enable_fp_fusion=False)")
    if variant == "fp16-intermediate":
        # Explicitly rounded low-precision intermediate, not a claim about
        # Triton's implicit reduction dtype. All memory accesses stay masked.
        return code.replace("var=tl.sum(z*z,0)/C", "var=tl.sum(z*z,0).to(tl.float16).to(tl.float32)/C")
    return candidate("layernorm", variant)


def assess_candidate_matrix(rows, expected_ids):
    expected = {(n, v) for n in expected_ids for v in CANDIDATE_VARIANTS} | {
        (n, v) for v, n in CONTROLS}
    observed = [(r.get("case_id"), r.get("variant")) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected or any(r.get("errors") != [] for r in rows):
        return False
    for row in rows:
        response = row.get("response", {})
        if row["variant"] in CANDIDATE_VARIANTS:
            repeat = row.get("repeat_response", {})
            if (not response.get("output_digest") or response["output_digest"] != repeat.get("output_digest")
                    or response.get("success") != repeat.get("success")):
                return False
        elif response.get("success") is not False or response.get("bad_elements", 0) <= 0:
            return False
    return True


def run(args):
    layout = run_layout(new_run_root(args.output, "layernorm-calibration"))
    rows, candidates = [], []
    expected = fixtures()
    summary = dict(completed=False, profile_enabled=False, default_profile="unknown", run_root=layout.root)
    backend = None
    with tempfile.TemporaryDirectory(prefix="opag-ln-calibration-") as temp:
        try:
            suite = proposed_suite()
            layout.write_for(ROLE_HARNESS, "harness/proposed-suite.json", dumps(dict(
                status="diagnostic_proposal_not_registered_profile", suite=suite.content_dict(),
                suite_hash=suite.digest(), coverage_lint=lint_coverage(suite))))
            layout.write_for(ROLE_HARNESS, "harness/driver.py", Path(__file__).read_text())
            layout.write_for(ROLE_HARNESS, "harness/layernorm_numerics.py",
                             (ROOT / "benchmarks/layernorm_numerics.py").read_text())
            if args.h20:
                server = Path(temp) / "runner_server.py"
                server.write_text(diagnostic_server())
                backend = CalibrationSSH(server, host=args.host, timeout=120)
                environment = backend.prepare(args.gpu)
                layout.write_for(ROLE_HARNESS, "harness/environment.json", dumps(environment))
                layout.write_for(ROLE_HARNESS, "harness/runner_server.py", server.read_text())
            else:
                from operator_agent.mock_gpu import MockGPU
                verifier = Verifier(MockGPU())
                provenance = dict(factory="Verifier._build_reference_fn", source_sha256=
                    hashlib.sha256((ROOT / "operator_agent/verifier.py").read_bytes()).hexdigest())

            for name, origin, inputs in expected:
                if backend:
                    result = backend._ssh_exec_json(dict(action="calibrate_layernorm",
                                                         inputs=encode_json_inputs(inputs)))
                    if result.get("success") is not True:
                        raise RuntimeError("remote calibration did not produce a measurement")
                    measurement = result["measurement"]
                else:
                    reference, kind = verifier._build_reference_fn(reference_task(), inputs)
                    if kind != "pytorch_eager":
                        raise RuntimeError("CPU calibration requires installed Torch, no fallback")
                    measurement = numerics.measure(inputs, reference, "cpu", provenance)
                errors = assess_measurement(measurement, inputs, "cuda:0" if backend else "cpu")
                rows.append(dict(case_id=name, origin=origin, measurement=measurement, errors=errors))
                if errors:
                    raise RuntimeError("calibration measurement validation failed")

            if backend:
                # Observe ALL original boundaries, not an early-exit passing subset.
                for variant in CANDIDATE_VARIANTS:
                    code = candidate_source(variant)
                    for name, origin, inputs in expected:
                        result = backend.verify_function(code, inputs, "layernorm", 1e-5, precision="fp16")
                        repeat = backend.verify_function(code, inputs, "layernorm", 1e-5, precision="fp16")
                        errors = assess_candidate(result, inputs) + assess_candidate(repeat, inputs)
                        if result.get("output_digest") != repeat.get("output_digest") or result.get("success") != repeat.get("success"):
                            errors.append("Triton output/verdict is not repeat-exact")
                        candidates.append(dict(case_id=name, variant=variant, response=result,
                                               repeat_response=repeat, errors=errors))
                for variant, name in CONTROLS:
                    inputs = next(xs for n, _, xs in expected if n == name)
                    result = backend.verify_function(candidate_source(variant), inputs,
                        "layernorm", 1e-5, precision="fp16")
                    errors = assess_candidate(result, inputs)
                    if result.get("success") is not False or result.get("bad_elements", 0) <= 0:
                        errors.append("wrong stride/tail/precision control not rejected numerically")
                    candidates.append(dict(case_id=name, variant=variant, response=result, errors=errors))
            summary.update(summarize(rows, [n for n, _, _ in expected]))
            summary["candidate_matrix_complete"] = (assess_candidate_matrix(candidates, [n for n, _, _ in expected])
                                                     if backend else None)
            summary["completed"] = summary["measurement_complete"] and (summary["candidate_matrix_complete"] is not False)
            # Calibration evidence may not silently activate the unsupported v1 profile.
            task = with_default_correctness(reference_task())
            preflight = Verifier.correctness_preflight(task)
            if (preflight is None or preflight.metrics["correctness"]["verdict"] != "unknown"
                    or task.constraints.correctness_suite is not None):
                raise RuntimeError("calibration changed production profile admission")
        except (Exception, KeyboardInterrupt) as exc:
            summary.update(completed=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            # Confirm our owned runner stopped before any fallible final writes.
            if backend:
                try:
                    summary["cleanup"] = backend.cleanup()
                except Exception as exc:
                    summary["cleanup"] = dict(quiescent=False, error=str(exc))
                if (summary["cleanup"].get("quiescent") is not True
                        or summary["cleanup"].get("removed") is not True or backend.forbidden_actions):
                    summary["completed"] = False
            layout.write_for(ROLE_ORACLE, "measurements/reference.json", dumps(rows))
            layout.write_for(ROLE_ORACLE, "measurements/candidates.json", dumps(candidates))
            summary["candidate_failures"] = {v: [r["case_id"] for r in candidates
                if r["variant"] == v and r["response"].get("success") is not True] for v in CANDIDATE_VARIANTS} if backend else None
            layout.write_for(ROLE_REPORT_WRITER, "report/summary.json", dumps(summary))
    print(dumps(summary), flush=True)
    return 0 if summary["completed"] else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h20", action="store_true")
    p.add_argument("--host", default=None)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--output", default=str(ROOT / "output/layernorm_calibration"))
    sys.exit(run(p.parse_args()))
