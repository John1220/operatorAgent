"""Small real NVIDIA L2 acceptance, NOT a performance benchmark.

Run: python3 benchmarks/input_fidelity_h20.py [--gpu 2]
     python3 benchmarks/input_fidelity_h20.py --boundaries [--gpu 2]
Uses the existing remote runtime without syncing/installing dependencies; pushes
the production runner's shared sources to a private random scratch directory.
Full evidence is private under output/, never committed with host/user paths.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from benchmarks._scratch import unmatchable  # noqa: E402
from operator_agent import (CorrectnessCase, CorrectnessSuite, ConstraintSpec,  # noqa: E402
                            SourceSpec, TargetSpec, TaskSchema, Verifier,
                            BoundaryAxis, BoundaryPolicy, TensorAxes)
from operator_agent.correctness import as_suite  # noqa: E402
from operator_agent.bench_cases import BenchmarkCase, TensorSpec  # noqa: E402
from operator_agent.contract import freeze_from_task  # noqa: E402
from operator_agent.input_fidelity import describe_inputs  # noqa: E402
from operator_agent.remote_executor import SSHExecutor  # noqa: E402
from operator_agent.report import build_report  # noqa: E402
from operator_agent.run_layout import (new_run_root, run_layout, ROLE_HARNESS,  # noqa: E402
    ROLE_CANDIDATE, ROLE_ORACLE, ROLE_REPORT_WRITER)
from operator_agent.schemas import KernelArtifact, VerificationLevel, VerificationResult  # noqa: E402

GOOD = '''import torch
import triton
import triton.language as tl
@triton.jit
def _relu(X, Y, N: tl.constexpr, C: tl.constexpr, SR: tl.constexpr,
          SC: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + (offs // C) * SR + (offs % C) * SC, offs < N, 0)
    tl.store(Y + offs, tl.maximum(x, 0), offs < N)
def kernel(X, eps=1e-5, causal=False):
    out = torch.empty(X.shape, dtype=X.dtype, device=X.device)
    _relu[(triton.cdiv(X.numel(), 32),)](X, out, X.numel(), X.shape[1],
                                      X.stride(0), X.stride(1), 32)
    return out
'''
BAD_DENSE = GOOD.replace("(offs // C) * SR + (offs % C) * SC", "offs")
# Deliberately wrong tail, but all loads/stores still masked: no GPU OOB probe.
BAD_TAIL = GOOD.replace("tl.maximum(x, 0), offs < N",
    "tl.where(offs < (N // BLOCK) * BLOCK, tl.maximum(x, 0), 0), offs < N")
BAD_MUTATION = GOOD.replace("    out = torch.empty", "    X.zero_()\n    out = torch.empty")
BAD_DETERMINISM = GOOD.replace("    return out",
    "    return out.to(torch.float64) + torch.rand_like(out, dtype=torch.float64) * 1e-10")


def case(name, shape=(7, 9), dtype="fp32", pattern="random", layout="contiguous"):
    return CorrectnessCase(name, BenchmarkCase(
        tensors=(TensorSpec("X", shape, dtype),), seed=17), pattern, layout)


def trials():
    aligned = case("aligned", (8, 8)).materialize()
    aligned.update(eps=np.float32(1e-5), causal=False)
    small = CorrectnessSuite((case("tail"),))
    core = CorrectnessSuite((case("aligned", (8, 8)), case("tail"),
        case("stride", layout="strided"), case("half-stride", (3, 5), "fp16", layout="strided"),
        *(case(p, pattern=p) for p in ("zeros", "constant", "large", "small"))))
    offset = np.arange(-40, 40, dtype=np.float32).reshape(10, 8)[1::2, 1::2]
    backing = np.array([-1, 0, 2], dtype=np.float32)
    writable_broadcast = np.ndarray((3, 7), dtype=backing.dtype, buffer=backing, strides=(4, 0))
    # (name, source, suite, primary, expected verdict, failed case / unknown reason)
    return [
        ("good-core", GOOD, core, aligned, "passed", ""),
        ("good-offset", GOOD, small, {"X": offset}, "passed", ""),
        ("good-writable-zero-stride", GOOD, small, {"X": writable_broadcast}, "passed", ""),
        ("bad-dense", BAD_DENSE, CorrectnessSuite((case("stride", layout="strided"),)), aligned, "failed", "stride"),
        ("bad-tail", BAD_TAIL, small, aligned, "failed", "tail"),
        ("bad-mutation", BAD_MUTATION, small, aligned, "failed", "primary"),
        ("bad-determinism", BAD_DETERMINISM, small, aligned, "failed", "tail"),
        ("readonly-refused", GOOD, CorrectnessSuite((case("readonly", layout="broadcast"),)), aligned, "unknown", "只读"),
        ("negative-stride-refused", GOOD, small, {"X": offset[::-1]}, "unknown", "负 stride"),
        ("good-after-negatives", GOOD, small, aligned, "passed", ""),
        ("good-one-shot", GOOD, small, aligned, "passed", ""),
    ]


def boundary_trials():
    """Extra real L2 witnesses from a trusted declaration, never candidate BLOCKs."""
    policy = BoundaryPolicy((BoundaryAxis("rows", 1, 9, 8, 8),
                             BoundaryAxis("cols", 1, 33, 8, 32)),
        (TensorAxes("X", ("rows", "cols")),), dtypes=("fp16", "fp32"), seed=17)
    raw = {"coverage_policy": policy.content_dict()}  # Exercise production deferred expansion.
    expanded = as_suite(raw)
    dense_target = next(c.case_id for c in expanded.cases if c.layout == "strided"
                        and c.inputs.tensors[0].dtype == "fp32")
    tail_target = next(c.case_id for c in expanded.cases if c.inputs.tensors[0].shape == (8, 31)
                      and c.inputs.tensors[0].dtype == "fp32" and c.pattern == "random" and c.layout == "contiguous")
    inputs = case("aligned", (8, 8)).materialize()
    return [("good-auto-boundaries", GOOD, raw, inputs, "passed", ""),
            ("bad-dense-auto-boundaries", BAD_DENSE, raw, inputs, "failed", dense_target),
            ("bad-tail-auto-boundaries", BAD_TAIL, raw, inputs, "failed", tail_target)]


def shell_path(path):
    return ('"$HOME"/' + shlex.quote(path[2:])) if path.startswith("~/") else shlex.quote(path)


def validate_scratch(path):
    p = PurePosixPath(path)
    if str(p) != path or p.parent != PurePosixPath("/tmp") or not re.fullmatch(
            r"opag-fidelity-[a-z0-9_]{8,}", p.name):
        raise ValueError("refusing non-private/non-random scratch path")
    return path


def select_gpu(csv, requested=None):
    rows = []
    for line in csv.splitlines():
        index, name, free, util = (x.strip() for x in line.split(","))
        rows.append(dict(index=int(index), name=name, free_mb=int(free), utilization=int(util)))
    idle = [r for r in rows if r["utilization"] <= 5 and r["free_mb"] >= 1024
            and (requested is None or r["index"] == requested)]
    if not idle:
        raise RuntimeError("no requested idle NVIDIA GPU with >=1 GiB free; no work launched")
    return max(idle, key=lambda r: r["free_mb"])


class AcceptanceSSH(SSHExecutor):
    """Only changes deployment location/runtime; retains production L2 transport."""

    def __init__(self, **kwargs):
        super().__init__(persistent=True, **kwargs)
        self.runtime = None
        self.scratch = None
        self.trace = []
        self.forbidden_actions = []

    def _existing_runtime(self, command):
        if self.runtime:
            command = command.replace("~/.local/bin/uv run python", shlex.quote(self.runtime))
        return command

    def _ssh_exec(self, remote_cmd, stdin_data=None):
        return super()._ssh_exec(self._existing_runtime(remote_cmd), stdin_data)

    def _session_argv(self, remote_cmd):
        return super()._session_argv(self._existing_runtime(remote_cmd))

    def _ssh_exec_json(self, payload):
        if payload.get("action") not in {"compile", "verify", "gpu_info", "ping", "shutdown"}:
            self.forbidden_actions.append(payload.get("action"))
            raise RuntimeError("acceptance forbids execute/benchmark/environment changes")
        row = {"request": payload}
        self.trace.append(row)
        try:
            row["response"] = super()._ssh_exec_json(payload)
            return row["response"]
        except Exception as exc:
            row["exception"] = repr(exc)
            raise

    def deploy(self):
        raise RuntimeError("acceptance requires explicit isolated deployment")

    def ensure_environment(self, required_pips=None):
        raise RuntimeError("acceptance never installs/syncs dependencies")

    def remote_python(self, code):
        if not self.runtime:
            raise RuntimeError("existing runtime not probed")
        return json.loads(self._ssh_exec(f"{shlex.quote(self.runtime)} -c {shlex.quote(code)}"))

    def prepare(self, requested_gpu=None):
        probe = ('import json,os,sys,torch,triton; print(json.dumps(dict('
                 'runtime=sys.executable,work_dir=os.getcwd(),cuda=torch.cuda.is_available(),'
                 'torch=torch.__version__,triton=triton.__version__)))')
        env = json.loads(self._ssh_exec(f"cd {shell_path(self.work_dir)} && "
            f"~/.local/bin/uv run --no-sync python -c {shlex.quote(probe)}"))
        if env["cuda"] is not True:
            raise RuntimeError("existing remote runtime has no CUDA; acceptance incomplete")
        self.runtime = env["runtime"]
        self.work_dir = shlex.quote(env["work_dir"])
        snapshot = self._ssh_exec("nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu "
                                  "--format=csv,noheader,nounits")
        gpu = select_gpu(snapshot, requested_gpu)
        self._gpu_pin = gpu["index"]
        created = self.remote_python('import json,tempfile,os; '
            'p=tempfile.mkdtemp(prefix="opag-fidelity-",dir="/tmp"); '
            'os.chmod(p,0o700); print(json.dumps(dict(path=p,mode=os.stat(p).st_mode & 0o777)))')
        self.scratch = validate_scratch(created["path"])
        if created["mode"] != 0o700:
            raise RuntimeError("scratch permissions are not private")
        self.remote_runner_path = self.scratch + "/run_kernel.py"
        files = self._remote_files()
        for local, remote in files:
            self._scp(local, remote)
        observed = self.remote_python('import json,hashlib,pathlib; paths=' + repr([r for _, r in files])
            + '; print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}))')
        hashes = {remote: hashlib.sha256(local.read_bytes()).hexdigest() for local, remote in files}
        if observed != hashes:
            raise RuntimeError("isolated deployment source hash mismatch")
        self._deployed = True
        info = self._ssh_exec_json({"action": "gpu_info"})
        if info.get("device") != "cuda" or info.get("selected_gpu") != self._gpu_pin or self.session_gpu != self._gpu_pin:
            raise RuntimeError("runner did not read back the selected physical GPU")
        return dict(environment=env, gpu_snapshot=snapshot, selected=gpu, device=info,
                    sources={Path(p).name: h for p, h in hashes.items()})

    def cleanup(self):
        self.close()
        if not self.scratch:
            return {"quiescent": True, "scratch_created": False}
        path = validate_scratch(self.scratch)
        # Do not remove a directory while its server is still alive, or kill others.
        left = self._ssh_exec(f"pgrep -f {shlex.quote(unmatchable(path))} || true").strip()
        if left:
            raise RuntimeError(f"scratch still has running processes ({left}); directory retained: {path}")
        result = self.remote_python('import os,json,stat,shutil; p=' + repr(path)
            + '; s=os.lstat(p); assert stat.S_ISDIR(s.st_mode) and s.st_uid==os.getuid() '
            'and stat.S_IMODE(s.st_mode)==0o700; shutil.rmtree(p); '
            'print(json.dumps(dict(quiescent=True,removed=not os.path.lexists(p))))')
        if result.get("removed") is not True:
            raise RuntimeError("scratch cleanup not confirmed")
        return result


class L2OnlyVerifier(Verifier):
    def __init__(self, backend):
        super().__init__(backend)
        self.l3_attempts = 0

    def _verify_performance(self, *args, **kwargs):
        self.l3_attempts += 1
        # Never run a benchmark, even if the gate regresses. This is NOT a pass.
        return VerificationResult(VerificationLevel.L3_PERFORMANCE, False,
                                  message="acceptance forbids performance calls")


def gpu_evidence(metrics):
    readings = metrics.get("input_fidelity_readings", [])
    if readings:
        return all(gpu_evidence(r) for r in readings)
    report = metrics.get("input_fidelity", {})
    tensors = {k for k, v in report.get("observed", {}).items() if v.get("kind") != "scalar"}
    devices = report.get("tensor_devices", {})
    return (metrics.get("input_fidelity_verified") is True and bool(tensors)
            and report.get("representation") == "torch-storage-v1"
            and set(devices) == tensors and all(d == "cuda:0" for d in devices.values()))


def assess(results, expected, target, expected_ids, l3_attempts):
    """Fail closed on missing evidence, skipped cases or an unrelated unknown."""
    errors = []
    if l3_attempts or any(r.level == VerificationLevel.L3_PERFORMANCE for r in results):
        errors.append("L3 was entered")
    if not results or results[0].level != VerificationLevel.L1_COMPILE or not results[0].passed:
        errors.append("L1 did not pass")
    l2 = next((r for r in results if r.level == VerificationLevel.L2_FUNCTION), None)
    if l2 is None:
        return errors + ["no L2 result"]
    c = l2.metrics.get("correctness", {})
    rows = c.get("cases", [])
    if c.get("verdict") != expected or l2.passed != (expected == "passed"):
        errors.append("unexpected correctness verdict")
    if {r.get("case_id") for r in rows} != set(expected_ids) or len(rows) != len(expected_ids):
        errors.append("declared case coverage missing/duplicated")
    if expected == "passed":
        if not gpu_evidence(l2.metrics):
            errors.append("primary lacks actual CUDA fidelity evidence")
        for row in rows:
            m, repeat = row.get("metrics", {}), row.get("repeat_metrics", {})
            if (row.get("verdict") != "passed" or row.get("determinism") != "passed"
                    or not gpu_evidence(m) or not gpu_evidence(repeat)
                    or not m.get("output_digest") or m["output_digest"] != repeat.get("output_digest")):
                errors.append(f"case/repeat evidence incomplete: {row.get('case_id')}")
    elif expected == "failed":
        if not gpu_evidence(l2.metrics):
            errors.append("failure lacks actual primary CUDA evidence")
        if target == "primary":
            if not l2.metrics.get("max_error", 0) > 0:
                errors.append("mutation did not cause numeric failure")
        else:
            row = next((r for r in rows if r.get("case_id") == target), {})
            # Numerically passed != bit-exact: the determinism negative adds
            # sub-tolerance noise on purpose. Retain the production comparison
            # receipt, not a second zero-error tolerance invented by this driver.
            if (not l2.metrics.get("input_fidelity_readings")
                    or l2.metrics.get("matched_ratio") != 1.0
                    or row.get("verdict") != "failed" or not gpu_evidence(row.get("metrics", {}))):
                errors.append("aligned primary/targeted numerical or determinism failure not established")
    elif target not in c.get("reason", ""):
        errors.append("unknown was not the expected unsupported layout")
    return errors


def dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def run(args):
    layout = run_layout(new_run_root(args.output, "input-fidelity-h20"))
    backend = AcceptanceSSH(host=args.host, timeout=120)
    summary = dict(scope="NVIDIA small ReLU L2 only; no performance/launch/fallback/OS-isolation proof",
                   accepted=False, trials=[], run_root=layout.root)
    layout.write_for(ROLE_HARNESS, "harness/driver.py", Path(__file__).read_text())
    try:
        environment = backend.prepare(args.gpu)
        layout.write_for(ROLE_HARNESS, "harness/environment.json", dumps(environment))
        specs = trials() + boundary_trials() if getattr(args, "boundaries", False) else trials()
        for name, code, declaration, inputs, expected, target in specs:
            if name == "good-one-shot":
                backend.close()
                backend._persistent = False
                info = backend._ssh_exec_json({"action": "gpu_info"})
                if info.get("selected_gpu") != backend._gpu_pin or info.get("device") != "cuda":
                    raise RuntimeError("one-shot runner selected a different device")
            if name == "good-auto-boundaries":
                # Keep added trials in a pinned persistent session, not one-shot
                # startup for every generated L2 case.
                backend._persistent = True
            suite = as_suite(declaration)
            task = TaskSchema(source=SourceSpec(kernel_type="relu"), target=TargetSpec(dsl="triton"),
                              constraints=ConstraintSpec(correctness_suite=declaration))
            layout.write_for(ROLE_HARNESS, f"contract/{name}.json", dumps(freeze_from_task(task).as_dict()))
            layout.write_for(ROLE_CANDIDATE, f"candidate/{name}.py", code)
            before = describe_inputs(inputs)
            verifier = L2OnlyVerifier(backend)
            start = len(backend.trace)
            results = verifier.verify(code, task, inputs,
                stop_after=VerificationLevel.L2_FUNCTION if expected == "passed" else None)
            errors = assess(results, expected, target, [c.case_id for c in suite.cases], verifier.l3_attempts)
            if before != describe_inputs(inputs):
                errors.append("controller inputs changed")
            if name == "bad-determinism":
                row = results[-1].metrics.get("correctness", {}).get("cases", [{}])[0]
                if "确定性失败" not in row.get("reason", ""):
                    errors.append("did not exercise raw-output determinism gate")
            if suite.coverage_policy is not None:
                c = results[-1].metrics.get("correctness", {})
                if c.get("coverage_lint", {}).get("status") != "passed":
                    errors.append("declared boundary coverage was not proven before actual L2")
                if c.get("suite_hash") != suite.digest():
                    errors.append("executed suite identity differs from frozen boundary suite")
            record = dict(name=name, expected=expected, accepted=not errors, errors=errors,
                l3_attempts=verifier.l3_attempts, primary_inputs=before,
                results=[dict(dataclasses.asdict(r), level=r.level.value) for r in results],
                transport=backend.trace[start:])
            layout.write_for(ROLE_ORACLE, f"measurements/{name}.json", dumps(record))
            layout.write_for(ROLE_REPORT_WRITER, f"report/{name}.md",
                             build_report(artifact=KernelArtifact(code=code, verification_results=results)))
            summary["trials"].append({k: record[k] for k in ("name", "expected", "accepted", "errors", "l3_attempts")})
            print(f"ACCEPTANCE {name}: {'PASS' if not errors else 'FAIL'} {errors}", flush=True)
        summary["accepted"] = all(r["accepted"] for r in summary["trials"]) and not backend.forbidden_actions
    except Exception as exc:
        summary["error"] = repr(exc)
        summary["accepted"] = False
    finally:
        # Remote teardown must happen even when local artifact writes fail (e.g.
        # disk full). Never put a fallible evidence write in front of cleanup.
        try:
            summary["cleanup"] = backend.cleanup()
        except Exception as exc:
            summary["cleanup"] = {"quiescent": False, "error": repr(exc)}
            summary["accepted"] = False
        summary["forbidden_actions"] = backend.forbidden_actions
        layout.write_for(ROLE_ORACLE, "measurements/transport.json", dumps(backend.trace))
        layout.write_for(ROLE_REPORT_WRITER, "report/summary.json", dumps(summary))
    print(dumps(summary), flush=True)
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None, help="SSH alias (default: backend configuration)")
    parser.add_argument("--gpu", type=int, default=None, help="physical GPU; refuses busy/low-memory cards")
    parser.add_argument("--output", default=str(ROOT / "output" / "input_fidelity_h20"))
    parser.add_argument("--boundaries", action="store_true", help="also verify generated suites and bad stride/tail kernels")
    sys.exit(run(parser.parse_args()))
