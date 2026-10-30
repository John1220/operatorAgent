"""Real CUDA cancellation/timeout followed by fresh full-L2 CLI verification.

No LLM, benchmark, package install, shared-runner overwrite, or automatic retry.
The slow fixture submits a finite CUDA delay after a real Triton ReLU launch.
Two per-process SM samples from our private runner must precede interruption.
Normal tests and judgement rules use the unmodified production CLI and runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.input_fidelity_h20 import AcceptanceSSH, GOOD, gpu_evidence
from operator_agent.anticheat import static_audit
from operator_agent.measurement import TRANSPORT_UNKNOWN

SLEEP_CYCLES = 40_000_000_000
SLOW = GOOD.replace("    return out", f"    torch.cuda.synchronize()\n    torch.cuda._sleep({SLEEP_CYCLES})\n    return out")
RPC_TIMEOUT = 8
TRIAL_LIMIT = 180


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    paths = list((ROOT / "operator_agent").rglob("*.py"))
    paths += list((ROOT / "remote_scripts").glob("*.py"))
    paths += [Path(__file__), ROOT / "benchmarks/input_fidelity_h20.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def active_witness(samples):
    """Handshake, resident memory, other PIDs and device-wide util are insufficient."""
    active = []
    for sample in samples:
        for row in sample.get("owned", []):
            if (type(row.get("pid")) is int and row["pid"] > 1
                    and isinstance(row.get("start_ticks"), str) and row["start_ticks"].isdigit()
                    and row.get("sm_pct", 0) > 0 and row.get("gpu_uuid") == sample.get("gpu_uuid")):
                active.append((row["pid"], row["start_ticks"], sample["observed_monotonic"]))
    return any(a[:2] == b[:2] and b[2] - a[2] >= .5 for a in active for b in active)


def terminal_errors(mode, status, body, result, samples, remaining):
    errors = []
    cleanup = (status or {}).get("cleanup") or {}
    if not active_witness(samples):
        errors.append("no two actual owned-process SM samples before interruption")
    if cleanup.get("worker_reaped") is not True or cleanup.get("remote_execution") != "quiescent":
        errors.append("production cleanup did not certify owned remote and local exit")
    if remaining.get("owned"):
        errors.append("private runner still present after cleanup")
    if (body or {}).get("passed") is True:
        errors.append("interrupted candidate incorrectly passed")
    if mode == "cancel":
        if (status or {}).get("phase") != "cancelled" or (body or {}).get("error") != "task_cancelled":
            errors.append("public cancellation did not win")
    else:
        metrics = [v.get("metrics", {}) for v in (result or {}).get("verdicts", [])]
        if not any(m.get("transport_outcome") == TRANSPORT_UNKNOWN and m.get("session_state") == "timeout"
                   and m.get("session_phase") == "response" for m in metrics):
            errors.append("no actual CUDA-request response timeout/unknown receipt")
        if any(v.get("level") == "L3" for v in (result or {}).get("verdicts", [])):
            errors.append("timed-out correctness request reached L3")
    return errors


def observe(backend, gpu, gpu_uuid):
    # Only emit our runner's identity and measurements; never emit /proc environ
    # or cmdline, which can contain ownership tokens from unrelated sessions.
    script = backend.scratch + "/runner_server.py"
    code = '''import json,os,subprocess,time
from pathlib import Path
gpu=GPU
identity=UUID
script=SCRIPT
pmon=subprocess.run(['nvidia-smi','pmon','-i',str(gpu),'-s','u','-d','1','-c','1'],capture_output=True,text=True,timeout=8)
sm={}
for line in pmon.stdout.splitlines():
 parts=line.split()
 if len(parts)>3 and parts[0].isdigit() and parts[1].isdigit() and parts[3].isdigit():
  sm[int(parts[1])]=int(parts[3])
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
device=subprocess.run(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
owned=[]
for line in apps.stdout.splitlines():
 parts=[x.strip() for x in line.split(',')]
 if len(parts)!=3 or not parts[0].isdigit() or parts[1]!=identity:continue
 pid=int(parts[0]);p=Path('/proc')/str(pid)
 try:
  if p.stat().st_uid!=os.getuid() or os.fsencode(script) not in (p/'cmdline').read_bytes().split(b'\\0')[1:]:continue
  birth=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
 except (OSError,IndexError):continue
 owned.append(dict(pid=pid,start_ticks=birth,gpu_uuid=identity,sm_pct=sm.get(pid,0),used_memory_mib=parts[2]))
print(json.dumps(dict(gpu_uuid=identity,owned=owned,pmon_ok=pmon.returncode==0,apps_ok=apps.returncode==0,device_snapshot=device.stdout.strip())))
'''.replace("GPU", repr(gpu)).replace("UUID", repr(gpu_uuid)).replace("SCRIPT", repr(script))
    result = backend.remote_python(code)
    result["observed_monotonic"] = time.monotonic()
    if result.get("pmon_ok") is not True or result.get("apps_ok") is not True:
        raise RuntimeError("per-process CUDA activity observer unavailable")
    return result


def rescue(backend):
    """Finally-only exact ownership rescue; never turns a failed test into a pass."""
    script = backend.scratch + "/runner_server.py"
    code = '''import json,os,sys
from pathlib import Path
sys.path.insert(0,DIRECTORY)
from session_control import cleanup_owned,TOKEN_ENV
script=SCRIPT
rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid() or os.fsencode(script) not in (p/'cmdline').read_bytes().split(b'\\0')[1:]:continue
  birth=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
  env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\\0') if b'=' in x)
  token=env.get(TOKEN_ENV.encode(),b'').decode()
  identity=dict(pid=int(p.name),uid=os.getuid(),start_ticks=birth,script=script)
  rows.append(cleanup_owned(identity,token,3))
 except (OSError,IndexError):continue
print(json.dumps(dict(owners=rows,quiescent=all(r.get('quiescent') is True for r in rows))))
'''.replace("DIRECTORY", repr(backend.scratch)).replace("SCRIPT", repr(script))
    return backend.remote_python(code)


def cli(root, env, *args, timeout=30):
    return subprocess.run([sys.executable, "-m", "operator_agent", *args], cwd=ROOT,
                          env=env, text=True, capture_output=True, timeout=timeout)


def trial(root, name, backend, env, gpu, gpu_uuid, *, mode=None):
    dest = root / name
    dest.mkdir()
    run = dest / "run"
    prepared = cli(root, env, "prepare", "--request", "relu fp32 triton", "--shapes", "7,33",
                   "--hardware", "nvidia_gpu", "--dsl", "triton", "--precision", "fp32",
                   "--min-speedup", "1.0", "--run-dir", str(run))
    (dest / "prepare.stdout.json").write_text(prepared.stdout)
    if prepared.returncode or not (read(dest / "prepare.stdout.json") or {}).get("ok"):
        raise RuntimeError("prepare failed; no candidate dispatched")
    candidate = dest / "candidate.py"
    candidate.write_text(SLOW if mode else GOOD)
    command = [sys.executable, "-m", "operator_agent", "verify", "--run-dir", str(run),
               "--code-file", str(candidate), "--stop-after", "L2", "--backend", "ssh",
               "--host", backend.host, "--persistent", "--timeout", str(RPC_TIMEOUT if mode == "timeout" else 120)]
    started = time.monotonic()
    samples = []
    cancelled = False
    with (dest / "stdout.json").open("w") as out, (dest / "stderr.log").open("w") as err:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            while proc.poll() is None and time.monotonic() - started < TRIAL_LIMIT:
                if mode and not active_witness(samples):
                    samples.append(observe(backend, gpu, gpu_uuid))
                    write(dest / "activity.json", samples)
                elif mode == "cancel" and not cancelled:
                    calls = list((run / "tool-calls").glob("*/launch.json"))
                    if len(calls) != 1:
                        raise RuntimeError("ambiguous invocation; no arbitrary PID signal")
                    invocation = read(calls[0])["invocation_id"]
                    cancel = cli(root, env, "cancel", "--run-dir", str(run), "--invocation-id", invocation)
                    (dest / "cancel.json").write_text(cancel.stdout)
                    if cancel.returncode or not (read(dest / "cancel.json") or {}).get("accepted"):
                        raise RuntimeError("public cancel refused")
                    cancelled = True
                else:
                    time.sleep(.1)
            if proc.poll() is None:
                raise TimeoutError("outer lifecycle limit reached")
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
    calls = list((run / "tool-calls").glob("*/run-status.json"))
    status = read(calls[0]) if len(calls) == 1 else None
    body = read(dest / "stdout.json")
    result = read(run / "verify-1/result.json")
    remaining = observe(backend, gpu, gpu_uuid)
    if mode:
        errors = terminal_errors(mode, status, body, result, samples, remaining)
        if mode == "cancel" and proc.returncode != 130:
            errors.append("cancel exit code was not 130")
        if mode == "cancel" and list(run.glob("verify-*/complete.json")):
            errors.append("cancelled verification fabricated a complete receipt")
    else:
        l2 = next((v for v in (result or {}).get("verdicts", []) if v.get("level") == "L2"), {})
        m = l2.get("metrics", {})
        correctness = m.get("correctness", {})
        errors = []
        if (proc.returncode != 0 or (body or {}).get("passed") is not True
                or l2.get("passed") is not True or correctness.get("verdict") != "passed"
                or len(correctness.get("cases", [])) != 12 or not gpu_evidence(m)):
            errors.append("fresh production CUDA full-L2 verification did not pass all 12 cases")
        if (status or {}).get("cleanup", {}).get("remote_execution") != "quiescent" or remaining.get("owned"):
            errors.append("normal follow-up session did not cleanly release its runner")
    row = dict(name=name, accepted=not errors, errors=errors, exit_code=proc.returncode,
               wall_seconds=round(time.monotonic() - started, 3),
               candidate_sha256=sha(candidate), active_gpu_witness=active_witness(samples),
               production_cleanup=(status or {}).get("cleanup"), after=remaining,
               scope="owned persistent SSH runner; provider and arbitrary descendants unproven")
    write(dest / "summary.json", row)
    return row


def main(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = fingerprint()
    plan = dict(schema="cuda-interrupt-recovery-v1", sources=sources, gpu=args.gpu,
                model_calls=0, l3_calls=0, sleep_cycles=SLEEP_CYCLES, rpc_timeout_s=RPC_TIMEOUT,
                schedule=["cancel", "normal-after-cancel", "timeout", "normal-after-timeout"],
                default_cleanup=True, retries=0)
    write(root / "plan.json", plan)
    for source in sources:
        target = root / "frozen-sources" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
    if not args.execute:
        print(json.dumps(dict(root=str(root), prepared=True, executed=False)))
        return 0
    if static_audit(SLOW, dsl="triton") or static_audit(GOOD, dsl="triton"):
        raise RuntimeError("fixed candidates fail existing audit")
    backend = AcceptanceSSH(host=args.host, timeout=120)
    summary = dict(accepted=False, model_calls=0, l3_calls=0, trials=[])
    try:
        env_info = backend.prepare(args.gpu)
        # End the setup runner before witnessing the CLI-owned process.
        if backend.close().get("quiescent") is not True:
            raise RuntimeError("setup runner did not stop")
        pin = backend.remote_python("import subprocess,json;print(json.dumps(subprocess.check_output("
            + repr(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
            + ",text=True).strip()))")
        gpu_uuid, free, utilization = [x.strip() for x in pin.split(",")]
        if int(free) < 1024 or int(utilization) > 5:
            raise RuntimeError("requested GPU became unavailable; no trial dispatched")
        write(root / "environment.json", dict(setup=env_info, pre_trial=pin))
        env = os.environ.copy()
        env.pop("OPERATOR_AGENT_S4_GRACEFUL_CLEANUP", None)
        env.update(REMOTE_RUNNER_PATH=backend.remote_runner_path,
                   OPERATOR_AGENT_EXISTING_REMOTE_ONLY="1", OPERATOR_AGENT_REQUIRED_GPU=str(args.gpu),
                   OPERATOR_AGENT_REQUIRED_GPU_UUID=gpu_uuid, OPERATOR_AGENT_REQUIRED_GPU_NAME=env_info["selected"]["name"])
        for name in plan["schedule"]:
            if fingerprint() != sources:
                raise RuntimeError("implementation drift before trial; stopped without retry")
            print(json.dumps(dict(stage=name, gpu=args.gpu)), flush=True)
            row = trial(root, name, backend, env, args.gpu, gpu_uuid,
                        mode=name if name in ("cancel", "timeout") else None)
            summary["trials"].append(row)
            write(root / "summary.json", summary)
            if not row["accepted"]:
                raise RuntimeError("trial acceptance failed; remaining schedule not dispatched")
        summary["accepted"] = len(summary["trials"]) == 4
    except (Exception, KeyboardInterrupt) as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if backend.scratch:
            try:
                summary["rescue_cleanup"] = rescue(backend)
            except Exception as exc:
                summary["rescue_cleanup"] = dict(quiescent=False, error=type(exc).__name__)
        try:
            summary["cleanup"] = backend.cleanup()
        except Exception as exc:
            summary["cleanup"] = dict(quiescent=False, error=type(exc).__name__)
        summary["implementation_stable"] = fingerprint() == sources
        if (not summary["implementation_stable"] or not summary["cleanup"].get("removed")
                or summary.get("rescue_cleanup", {}).get("owners")
                or summary.get("rescue_cleanup", {}).get("quiescent") is not True):
            summary["accepted"] = False
        write(root / "summary.json", summary)
        manifest = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
        write(root / "evidence-manifest.json", manifest)
    print(json.dumps(dict(root=str(root), accepted=summary["accepted"],
                         trials=[dict(name=r["name"], accepted=r["accepted"]) for r in summary["trials"]],
                         error=summary.get("error")), ensure_ascii=False), flush=True)
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    sys.exit(main(parser.parse_args()))
