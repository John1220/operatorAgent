"""Local JSONL commit protocol; one writer, fail closed on ambiguous I/O.

The newline-terminated, hash-checked kernel event is the commit record. A receipt
is successful only after file and directory fsync and exact readback. Repeating
an admission reconciles this record instead of appending again. Not distributed
exactly-once, hostile-owner protection, or an exactly-once GPU/bandit protocol.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA = "kernel-admission-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def record_hash(record):
    # Admission time is assigned by the first successful append, not a new
    # request's wall clock. All other fields must match an idempotent retry.
    return digest({k: v for k, v in record.items() if k != "created_at"})


def confirmed_key(first, second):
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError("admission requires both measurement bindings")
    common = ("run_id", "manifest_hash", "contract_hash", "workload_hash",
              "protocol_hash", "candidate_hash")
    for k in common:
        v = first.get(k)
        n = 32 if k == "run_id" else 64
        if (type(v) is not str or len(v) != n or any(c not in "0123456789abcdef" for c in v)
                or second.get(k) != v):
            raise ValueError("admission measurement binding mismatch: " + k)
    ids = [x.get("verification_id") for x in (first, second)]
    if any(type(v) is not str or not v for v in ids) or ids[0] == ids[1]:
        raise ValueError("admission requires distinct verification identities")
    return dict(kind="confirmed-observation", **{k: first[k] for k in common},
                first_verification_id=ids[0], second_verification_id=ids[1])


def admission_id(key):
    if not isinstance(key, dict):
        raise ValueError("invalid admission key")
    if key.get("kind") == "confirmed-observation":
        common = {k: key.get(k) for k in ("run_id", "manifest_hash", "contract_hash",
                  "workload_hash", "protocol_hash", "candidate_hash")}
        first = dict(common, verification_id=key.get("first_verification_id"))
        second = dict(common, verification_id=key.get("second_verification_id"))
        if key != confirmed_key(first, second):
            raise ValueError("noncanonical admission key")
    elif (key.get("kind") != "legacy-record" or set(key) != {"kind", "record_hash"}
          or type(key.get("record_hash")) is not str or len(key["record_hash"]) != 64
          or any(c not in "0123456789abcdef" for c in key["record_hash"])):
        raise ValueError("invalid legacy admission key")
    return digest(dict(schema=SCHEMA, key=key))


@dataclass(frozen=True)
class CommitResult:
    status: str
    admission_id: str = ""
    reason: str = ""
    record_hash: str = ""

    @property
    def persisted(self):
        return self.status in {"committed", "already_committed"}

    def as_dict(self):
        return dict(asdict(self), persisted=self.persisted)


@dataclass
class Scan:
    records: dict = field(default_factory=dict)
    admissions: dict = field(default_factory=dict)
    retrieved: list = field(default_factory=list)
    loaded: int = 0
    skipped: int = 0
    duplicates: int = 0
    tail: bool = False
    errors: list = field(default_factory=list)


def scan(f):
    # Runtime import keeps KernelRecord as the single backwards-compatible
    # record schema. Reject incomplete updates before they overwrite a valid
    # prefix, not later while installing the final per-id map.
    from ..core.kernel_db import KernelRecord
    f.seek(0)
    rows = f.read().splitlines(keepends=True)
    result = Scan()
    for i, raw in enumerate(rows):
        if not raw.endswith(b"\n"):
            result.tail = True
            result.skipped += 1
            continue
        # An interrupted tail is preserved, but poisoned before later writes:
        # even a complete JSON object missing its newline must never turn into
        # a committed event merely because a later append supplies the newline.
        if raw.endswith(b"\x00\n"):
            result.skipped += 1
            continue
        if not raw.strip():
            continue
        try:
            ev = json.loads(raw)
            if not isinstance(ev, dict):
                raise ValueError("non-object event")
            if ev.get("event") == "kernel":
                record = ev["record"]
                if (not isinstance(record, dict) or type(record.get("kernel_id")) is not str
                        or not record["kernel_id"]):
                    raise ValueError("invalid kernel record")
                KernelRecord.from_dict(record)
                seal = ev.get("admission")
                if seal is not None:
                    if (set(seal) != {"schema", "id", "key", "record_hash", "event_hash"}
                            or seal["schema"] != SCHEMA or seal["id"] != admission_id(seal["key"])
                            or seal["record_hash"] != record_hash(record)
                            or seal["event_hash"] != digest({k:v for k,v in ev.items() if k != "admission"})):
                        raise ValueError("kernel admission hash/key mismatch")
                    old = result.admissions.get(seal["id"])
                    if old is not None:
                        if old != ev:
                            raise ValueError("conflicting duplicate admission")
                        result.duplicates += 1
                        continue
                    result.admissions[seal["id"]] = ev
                result.records[record["kernel_id"]] = record
                result.loaded += 1
            elif ev.get("event") == "retrieved":
                ids = ev["ids"]
                if not isinstance(ids, list) or any(type(x) is not str for x in ids):
                    raise ValueError("invalid retrieved event")
                # Preserve event order, not all retrievals after all updates.
                for kid in ids:
                    if kid in result.records:
                        result.records[kid] = dict(result.records[kid],
                            retrieved_count=result.records[kid].get("retrieved_count", 0) + 1)
                result.retrieved.append(ids)
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            result.skipped += 1
            result.errors.append(f"line {i + 1}: {exc}")
    return result


def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def locked_file(path, *, create=False):
    import fcntl
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("symlink corpus path")
    if create:
        missing, p = [], path.parent
        while not p.exists():
            missing.append(p)
            p = p.parent
        path.parent.mkdir(parents=True, exist_ok=True)
        for directory in reversed(missing):
            fsync_directory(directory.parent)
    with path.open("a+b" if create else "r+b", buffering=0) as f:
        # Fail closed if a second writer/replayer is unexpectedly active. The
        # supported contract remains a single owner, not lock-based scheduling.
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def fence(f, path):
    f.flush()
    os.fsync(f.fileno())
    fsync_directory(Path(path).parent)


def append(f, event, *, tail=False):
    payload = json.dumps(event, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    f.seek(0, os.SEEK_END)
    if tail and f.write(b"\x00\n") != 2:
        raise OSError("short tail-quarantine write")
    if f.write(payload) != len(payload):
        raise OSError("short corpus event write")


def commit(path, record, key):
    """Return (receipt, fenced scan). Never install a record on failed I/O."""
    aid = ""
    rh = ""
    started = False
    try:
        from ..core.kernel_db import KernelRecord
        KernelRecord.from_dict(record)
        if (type(record.get("score")) not in (int, float) or not math.isfinite(record["score"])
                or type(record.get("code")) is not str or type(record.get("kernel_id")) is not str
                or not record["kernel_id"]):
            raise ValueError("invalid admission record scalar fields")
        rh = record_hash(record)
        aid = admission_id(key)
        if key["kind"] == "legacy-record" and key["record_hash"] != rh:
            raise ValueError("legacy payload/key conflict")
        if key["kind"] == "confirmed-observation":
            first = record.get("compile_info", {}).get("run_binding")
            if (not isinstance(first, dict) or any(first.get(k) != v for k,v in key.items()
                    if k not in {"kind", "first_verification_id", "second_verification_id"})
                    or first.get("verification_id") != key["first_verification_id"]
                    or hashlib.sha256(record["code"].encode()).hexdigest() != key["candidate_hash"]):
                raise ValueError("admission key/record binding conflict")
        if path is None:
            return CommitResult("persistence_disabled", aid, "harness persistence disabled", rh), None
        with locked_file(path, create=True) as f:
            current = scan(f)
            if current.errors:
                return CommitResult("refused", aid, "corpus corrupt: " + "; ".join(current.errors), rh), None
            existing = current.admissions.get(aid)
            if existing:
                if existing["admission"]["record_hash"] != rh:
                    return CommitResult("refused", aid, "same admission id has a different payload", rh), None
                started = True
                fence(f, path)  # Reconcile a previous lost fsync/receipt outcome.
                return CommitResult("already_committed", aid, "exact journal event reconciled", rh), current
            old = current.records.get(record["kernel_id"])
            if old is not None and record["score"] <= old["score"]:
                return CommitResult("refused", aid, "existing durable score is not improved", rh), None
            event = dict(event="kernel", record=record, at=record["created_at"])
            event["admission"] = dict(schema=SCHEMA, id=aid, key=key, record_hash=rh,
                                      event_hash=digest(event))
            started = True
            append(f, event, tail=current.tail)
            fence(f, path)
            readback = scan(f)
            if readback.errors or readback.admissions.get(aid) != event:
                raise OSError("exact corpus event readback failed")
            return CommitResult("committed", aid, "fsync and exact event readback confirmed", rh), readback
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return CommitResult("outcome_unknown" if started else "failed", aid,
                            f"{type(exc).__name__}: {exc}", rh), None
