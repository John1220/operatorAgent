"""Isolated SSH deadline/backpressure + real H20 large-input L2 acceptance.

Uses the existing runtime, no installs/syncs/benchmarks. Pipe stall fixtures do
NOT execute on CUDA; numeric large-input trials must prove actual CUDA readback.
All fixture processes use production RPC and pidfd ownership cleanup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.input_fidelity_h20 import AcceptanceSSH, GOOD
from operator_agent.input_fidelity import describe_inputs, check_readback
from operator_agent.remote_input_fidelity import encode_json_inputs
from operator_agent.remote_executor import SessionError
from operator_agent.run_layout import new_run_root, run_layout, ROLE_HARNESS, ROLE_ORACLE, ROLE_REPORT_WRITER

PEER = '''import os,sys,json,time
from session_control import runner_identity
print(json.dumps(dict(session_identity=runner_identity(__file__))),flush=True)
req=json.loads(sys.stdin.readline())
print(json.dumps(dict(id=req['id'],result=dict(pong=True,selected_gpu=-1))),flush=True)
mode=os.environ['OPERATOR_AGENT_DEADLINE_MODE']
if mode=='partial': sys.stdin.buffer.read(1024)
if mode=='late':
 req=json.loads(sys.stdin.readline())
time.sleep(60)
'''


class FixtureSSH(AcceptanceSSH):
    def __init__(self, mode, **kw):
        super().__init__(**kw)
        self.mode = mode

    def _session_argv(self, cmd):
        cmd = cmd.replace("OPERATOR_AGENT_SESSION_TOKEN=",
                          f"OPERATOR_AGENT_DEADLINE_MODE={self.mode} OPERATOR_AGENT_SESSION_TOKEN=")
        return super()._session_argv(cmd)


def assess_stall(row, mode, budget):
    errors = []
    if row.get("session_state") != "timeout" or row.get("session_phase") != ("response" if mode == "late" else "send"):
        errors.append("expected timeout phase not observed")
    if not budget * .8 <= row.get("elapsed", float("inf")) < budget + 1.0:
        errors.append("request deadline was not bounded")
    if mode != "late" and not 0 < row.get("bytes_sent", 0) < row.get("payload_bytes", 0):
        errors.append("not a real partial pipe send")
    if row.get("restart_refused") is not True:
        errors.append("broken session was automatically reused/restarted")
    cleanup = row.get("cleanup", {})
    if cleanup.get("quiescent") is not True or cleanup.get("remote", {}).get("quiescent") is not True:
        errors.append("owned runner not confirmed stopped")
    if row.get("cleanup_elapsed", float("inf")) > 11.0:
        errors.append("cleanup deadline was not bounded")
    return errors


def assess_numeric(response, expected):
    fidelity = check_readback(expected, response.get("input_fidelity"))
    devices = response.get("input_fidelity", {}).get("tensor_devices", {})
    errors = []
    if response.get("success") is not True or response.get("max_error", 1) > 1e-5:
        errors.append("real CUDA numeric comparison did not pass")
    if (fidelity["input_fidelity_verified"] is not True or devices != {"X": "cuda:0"}
            or response.get("coverage") != 1.0 or not response.get("output_digest")):
        errors.append("large-input CUDA fidelity/full-output evidence incomplete")
    return errors


def run(args):
    root = Path(new_run_root(args.output, "rpc-deadline-h20"))
    layout = run_layout(str(root))
    backend = AcceptanceSSH(host=args.host, timeout=60)
    summary = {"accepted": False, "trials": []}
    fixtures = []
    try:
        environment = backend.prepare(args.gpu)
        layout.write_for(ROLE_HARNESS, "harness/environment.json", json.dumps(environment, indent=2))
        if not backend.compile(GOOD):
            raise RuntimeError("large-input candidate did not compile")
        # Real numeric/fidelity requests: medium 1 MiB, large 16 MiB. L3 is forbidden.
        for name, shape in (("medium-cuda", (512,512)), ("large-cuda", (2048,2048))):
            inputs = {"X": np.random.default_rng(17).standard_normal(shape).astype(np.float32)}
            wire = encode_json_inputs(inputs)
            started = time.monotonic()
            result = backend._ssh_exec_json(dict(action="verify", code=GOOD, inputs=wire,
                kernel_type="relu", max_error=1e-5, precision="fp32", reference_source=""))
            row = dict(name=name, scope="actual CUDA L2", input_bytes=inputs["X"].nbytes,
                       elapsed=time.monotonic()-started, response=result)
            row["errors"] = assess_numeric(result, describe_inputs(inputs))
            row["accepted"] = not row["errors"]
            summary["trials"].append(row)
        summary["numeric_session_cleanup"] = backend.close()
        if summary["numeric_session_cleanup"].get("quiescent") is not True:
            raise RuntimeError("numeric runner cleanup unconfirmed")
        directory = backend.scratch + "/transport"
        backend.remote_python('import os,json; os.mkdir(' + repr(directory) +
                              ',0o700); print(json.dumps(dict(created=True)))')
        with tempfile.TemporaryDirectory(prefix="opag-deadline-peer-") as local:
            source = Path(local) / "runner_server.py"
            source.write_text(PEER)
            os.chmod(source, 0o600)
            backend._scp(source, directory + "/runner_server.py")
        backend._scp(ROOT / "remote_scripts/session_control.py", directory + "/session_control.py")
        expected_sha = hashlib.sha256(PEER.encode()).hexdigest()
        expected_sources = {directory + "/runner_server.py": expected_sha,
                            directory + "/session_control.py": hashlib.sha256(
                                (ROOT / "remote_scripts/session_control.py").read_bytes()).hexdigest()}
        observed = backend.remote_python('import json,hashlib,pathlib; paths=' + repr(list(expected_sources))
            + '; print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}))')
        if observed != expected_sources:
            raise RuntimeError("fixture source hash mismatch")
        summary["fixture_sha256"] = expected_sha
        summary["fixture_sources"] = {Path(p).name: h for p,h in expected_sources.items()}
        for mode in ("stopped", "partial", "late"):
            ex = FixtureSSH(mode, host=args.host, timeout=30, remote_runner_path=directory + "/run_kernel.py")
            fixtures.append(ex)
            ex.runtime, ex.work_dir, ex._gpu_pin, ex._deployed = backend.runtime, backend.work_dir, -1, True
            row = dict(name=mode, scope="SSH pipe fixture; no CUDA execution")
            try:
                ex._start_session()  # Startup budget separate from measured resident RPC.
                ex.timeout = ex._session.timeout = .5
                payload = {"action": "verify", "blob": "x" * (16*1024*1024 if mode != "late" else 1024)}
                row["payload_bytes"] = len(payload["blob"])
                started = time.monotonic()
                try:
                    ex._session_call(payload)
                    row["error"] = "unexpected successful stalled request"
                except SessionError as exc:
                    row.update(session_state=exc.session_state, session_phase=exc.session_phase,
                               request_id=exc.request_id, bytes_sent=exc.bytes_sent)
                row["elapsed"] = time.monotonic() - started
                try:
                    ex._session_call({"action": "ping"})
                except SessionError as exc:
                    row["restart_refused"] = exc.session_state == "broken"
            finally:
                started = time.monotonic()
                row["cleanup"] = ex.close()
                row["cleanup_elapsed"] = time.monotonic() - started
            row["errors"] = assess_stall(row, mode, .5)
            row["accepted"] = not row["errors"]
            summary["trials"].append(row)
            if not row["accepted"]:
                raise RuntimeError("stall lifecycle acceptance failed")
        # Explicit NEW numeric session only AFTER all uncertain owners stopped.
        inputs = {"X": np.arange(-40,40,dtype=np.float32).reshape(10,8)[1::2,1::2]}
        result = backend.verify_function(GOOD, inputs, "relu", 1e-5, precision="fp32")
        errors = assess_numeric(result, describe_inputs(inputs))
        summary["trials"].append(dict(name="new-session-after-cleanup", scope="actual CUDA L2",
                                      response=result, errors=errors, accepted=not errors))
        summary["accepted"] = len(summary["trials"]) == 6 and all(r["accepted"] for r in summary["trials"])
    except (Exception, KeyboardInterrupt) as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Cleanup first, before any fallible evidence writes. No broad process kill.
        summary["fixture_final_cleanup"] = [ex.close() for ex in fixtures]
        try:
            summary["cleanup"] = backend.cleanup()
        except Exception as exc:
            summary["cleanup"] = dict(quiescent=False, error=str(exc))
        if (summary["cleanup"].get("quiescent") is not True or summary["cleanup"].get("removed") is not True
                or any(c.get("quiescent") is not True for c in summary["fixture_final_cleanup"])
                or backend.forbidden_actions):
            summary["accepted"] = False
        # Store compact responses, never 16 MiB request storages / ownership token.
        layout.write_for(ROLE_ORACLE, "measurements/trials.json", json.dumps(summary["trials"], indent=2))
        layout.write_for(ROLE_REPORT_WRITER, "report/summary.json", json.dumps(summary, indent=2))
    print(json.dumps(dict(root=str(root), accepted=summary["accepted"],
                          trials=[dict(name=r["name"],accepted=r["accepted"]) for r in summary["trials"]],
                          cleanup=summary["cleanup"], error=summary.get("error")), indent=2), flush=True)
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--output", default=str(ROOT / "output/rpc_deadline_h20"))
    sys.exit(run(parser.parse_args()))
