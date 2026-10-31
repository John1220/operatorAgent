"""Calibrate durable-run guards in disposable private copies, never live source."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MUTANTS = (
    ("ignore-checkpoint-seal", "operator_agent/core/run_contract.py",
     'if original != seal(copy, binding)["run_binding"]:',
     'if False and original != seal(copy, binding)["run_binding"]:'),
    ("ignore-resume-workload", "operator_agent/core/run_contract.py",
     'inputs = [describe_inputs(item) for item in input_sets]', 'inputs = [{} for item in input_sets]'),
    ("ignore-declaration-drift", "operator_agent/core/run_contract.py",
     'if m.get(key) != value:', 'if False and m.get(key) != value:'),
    ("allow-sidecar-overwrite", "operator_agent/core/run_contract.py",
     'if digest(_read(path)) != digest(binding[key]):',
     'if False and digest(_read(path)) != digest(binding[key]):'),
    ("writable-contract", "operator_agent/core/run_contract.py",
     'atomic_json(path, binding[key], readonly=True)', 'atomic_json(path, binding[key], readonly=False)'),
    ("regenerate-on-resume", "operator_agent/app/conductor.py",
     'pending_fix = pending_fix or current_code', 'pending_fix = None'),
    ("reuse-exhausted-result", "operator_agent/app/conductor.py",
     'if snap["round_done"] >= self.max_optimize_rounds:',
     'if False and snap["round_done"] >= self.max_optimize_rounds:'),
    ("ignore-critical-write", "operator_agent/app/conductor.py",
     'if self._run_binding is not None:\n                raise run_contract.RunBindingError("checkpoint write failed:',
     'if False and self._run_binding is not None:\n                raise run_contract.RunBindingError("checkpoint write failed:'),
)


def run(output):
    rows = []
    # A completely independent package root is important: protocol_identity()
    # hashes this copy, and no transient mutant touches an active H20 run.
    with tempfile.TemporaryDirectory(prefix="run-contract-calibration-") as tmp:
        base = Path(tmp)
        for name, rel, old, new in (("clean", "", "", ""),) + MUTANTS:
            root = base / name
            root.mkdir()
            for folder in ("operator_agent", "remote_scripts"):
                shutil.copytree(ROOT / folder, root / folder,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (root / "tests").mkdir()
            (root / "tests" / "__init__.py").touch()
            for file in ("test_run_contract.py", "test_resume_integrity.py"):
                shutil.copy2(ROOT / "tests" / file, root / "tests" / file)
            if name != "clean":
                path = root / rel
                source = path.read_text()
                if source.count(old) != 1:
                    raise RuntimeError("mutant anchor is not unique: " + name)
                path.write_text(source.replace(old, new, 1))
            env = dict(os.environ, PYTHONPATH=str(root), OPERATOR_AGENT_LOG_DIR=str(root / "logs"),
                       OPERATOR_AGENT_OUTPUT_DIR=str(root / "output"))
            result = subprocess.run([sys.executable, "-m", "pytest", "tests/unit/test_run_contract.py", "-q"],
                cwd=str(root), env=env, capture_output=True, text=True, timeout=120)
            log = result.stdout + result.stderr
            # Collection/import errors don't count as assertion kills.
            accepted = (result.returncode == 0 if name == "clean" else
                        result.returncode == 1 and "failed" in log and "ERROR collecting" not in log)
            rows.append(dict(name=name, accepted=accepted, returncode=result.returncode, log=log))
            print(name + ": " + ("PASS" if accepted else "FAIL"), flush=True)
    report = dict(completed=all(r["accepted"] for r in rows), trials=rows)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", help="explicit report destination; default is a new private random directory")
    args = p.parse_args()
    output = args.output or str(Path(tempfile.mkdtemp(prefix="run-contract-guard-report-")) / "report.json")
    print("REPORT " + output, flush=True)
    raise SystemExit(run(output))
