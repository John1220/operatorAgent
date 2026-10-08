"""Linux owned-runner identity and bounded cleanup; no Torch/GPU imports.

Never pgrep/pkill a shared runner path. A pidfd pins the process while /proc
readbacks validate UID, birth time, script argv and a random session token.
This is lifecycle control for a trusted runner, not a hostile-candidate sandbox.
"""
import argparse
import json
import math
import os
import re
import select
import signal
import time
from pathlib import Path

TOKEN_ENV = "OPERATOR_AGENT_SESSION_TOKEN"


def _birth(stat_text):
    # comm (field 2) can contain spaces and parentheses. starttime is field 22.
    return stat_text.rsplit(")", 1)[1].split()[19]


def runner_identity(script):
    p = Path("/proc/self")
    return {"pid": os.getpid(), "uid": os.getuid(),
            "start_ticks": _birth((p / "stat").read_text()),
            "script": os.path.realpath(script)}


def validate_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("missing runner identity")
    pid, uid = identity.get("pid"), identity.get("uid")
    script, birth = identity.get("script"), identity.get("start_ticks")
    if (type(pid) is not int or pid <= 1 or type(uid) is not int or uid < 0
            or not isinstance(birth, str) or not birth.isdigit()
            or not isinstance(script, str) or not os.path.isabs(script)
            or Path(script).name != "runner_server.py"):
        raise ValueError("invalid runner identity")
    return identity


def _validate_owner(identity, token, proc_root=Path("/proc")):
    p = proc_root / str(identity["pid"])
    if p.stat().st_uid != identity["uid"] or identity["uid"] != os.getuid():
        raise ValueError("runner UID mismatch")
    if _birth((p / "stat").read_text()) != identity["start_ticks"]:
        raise ValueError("runner birth time mismatch")
    argv = (p / "cmdline").read_bytes().split(b"\0")
    if os.fsencode(identity["script"]) not in argv[1:]:
        raise ValueError("runner script argv mismatch")
    if os.fsencode(TOKEN_ENV + "=" + token) not in (p / "environ").read_bytes().split(b"\0"):
        raise ValueError("runner token mismatch")


def cleanup_owned(identity, token, grace=3.0):
    """Only signal a verified pidfd; missing authority always fails closed."""
    evidence = {"quiescent": False, "signals": []}
    fd = None
    try:
        validate_identity(identity)
        if not isinstance(token, str) or not re.fullmatch(r"[a-f0-9]{32}", token):
            raise ValueError("invalid session token")
        if not math.isfinite(grace) or grace <= 0:
            raise ValueError("invalid cleanup grace")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("pidfd ownership cleanup unsupported")
        try:
            fd = os.pidfd_open(identity["pid"])
        except ProcessLookupError:
            return {**evidence, "quiescent": True, "reason": "owned PID absent"}
        if select.select([fd], [], [], 0)[0]:
            return {**evidence, "quiescent": True, "reason": "owned process exited"}
        _validate_owner(identity, token)
        evidence["identity_verified"] = True
        deadline = time.monotonic() + grace
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                signal.pidfd_send_signal(fd, sig)
                evidence["signals"].append(sig.name)
            except ProcessLookupError:
                return {**evidence, "quiescent": True, "reason": "owned process exited"}
            left = max(0.0, deadline - time.monotonic())
            if select.select([fd], [], [], left * .5 if sig == signal.SIGTERM else left)[0]:
                return {**evidence, "quiescent": True, "reason": "owned process exited"}
        evidence["reason"] = "cleanup deadline reached; runner not confirmed stopped"
    except Exception as exc:
        evidence["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if fd is not None:
            os.close(fd)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--grace", type=float, default=3.0)
    args = parser.parse_args()
    print(json.dumps(cleanup_owned(json.loads(args.identity), args.token, args.grace)), flush=True)
