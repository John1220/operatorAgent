"""Assertion-kill calibration in private source copies; never mutate live SSH runs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = (
    ("ignore-manifest-hash", "operator_agent/session_inputs.py",
     'if body["protocol"] != PROTOCOL or _sha(manifest["hash"]) != digest(body):',
     'if body["protocol"] != PROTOCOL:'),
    ("ignore-session-nonce", "operator_agent/session_inputs.py",
     'if not self.active or session_id != self.session_id:', 'if not self.active:'),
    ("ignore-storage-digest", "operator_agent/session_inputs.py",
     'or hashlib.sha256(raw).hexdigest() != ref["sha256"]):', '):'),
    ("ignore-commit-before-use", "operator_agent/session_inputs.py",
     'if self.ready is None or digest(self.ready) != digest(manifest):',
     'if False and (self.ready is None or digest(self.ready) != digest(manifest)):'),
    ("ignore-chunk-offset", "operator_agent/session_inputs.py",
     'if buf is None or type(offset) is not int or offset != len(buf):', 'if buf is None:'),
    ("unbounded-cache", "operator_agent/session_inputs.py",
     'while sum(map(len, self.blobs.values())) + reserved > self.capacity:',
     'while False and sum(map(len, self.blobs.values())) + reserved > self.capacity:'),
    ("ignore-upload-receipts", "operator_agent/remote_executor.py",
     'if (result.get("success") is not True or result.get("session_id") != session_id\n                    or result.get("manifest_hash") != sha):',
     'if result.get("success") is not True:'),
    ("ignore-final-input-proof", "operator_agent/remote_executor.py",
     'or proof.get("manifest_hash") != sha or proof.get("content_verified") is not True',
     'or proof.get("content_verified") is not True'),
    ("renew-each-upload-deadline", "operator_agent/remote_executor.py",
     'result = self._call_locked(request, deadline=deadline)',
     'result = self._call_locked(request, deadline=time.monotonic() + self.timeout)'),
    ("unbounded-zlib-decompression", "operator_agent/session_inputs.py",
     'raw = decoder.decompress(raw, CHUNK_BYTES + 1)', 'raw = decoder.decompress(raw)'),
)


def run(output):
    rows = []
    with tempfile.TemporaryDirectory(prefix="session-input-calibration-") as tmp:
        for name, rel, old, new in (("clean", "", "", ""),) + MUTANTS:
            root = Path(tmp) / name
            root.mkdir()
            for folder in ("operator_agent", "remote_scripts"):
                shutil.copytree(ROOT / folder, root / folder,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (root / "tests").mkdir()
            shutil.copy2(ROOT / "tests/test_session_inputs.py", root / "tests/test_session_inputs.py")
            if name != "clean":
                path = root / rel
                source = path.read_text()
                if source.count(old) != 1:
                    raise RuntimeError("mutant anchor not unique: " + name)
                path.write_text(source.replace(old,new,1))
            env = dict(os.environ, PYTHONPATH=str(root), OPERATOR_AGENT_LOG_DIR=str(root / "logs"),
                       OPERATOR_AGENT_OUTPUT_DIR=str(root / "output"))
            result = subprocess.run([sys.executable,"-m","pytest","tests/test_session_inputs.py","-q"],
                cwd=root,env=env,capture_output=True,text=True,timeout=90)
            log = result.stdout + result.stderr
            accepted = (result.returncode == 0 if name == "clean" else
                        result.returncode == 1 and "failed" in log and "ERROR collecting" not in log)
            rows.append(dict(name=name,accepted=accepted,returncode=result.returncode,log=log))
            print(name + ": " + ("PASS" if accepted else "FAIL"),flush=True)
    path = Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    report = dict(completed=all(r["accepted"] for r in rows),trials=rows)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output")
    args = p.parse_args()
    output = args.output or str(Path(tempfile.mkdtemp(prefix="session-input-guard-report-")) / "report.json")
    print("REPORT " + output,flush=True)
    raise SystemExit(run(output))
