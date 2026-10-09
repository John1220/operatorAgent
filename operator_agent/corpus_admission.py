"""Prepared local admissions and explicit recovery, without measuring/learning.

The corpus journal is authoritative. admission.json is a durable intent and
receipt, never an alternative qualification policy or a restored GPU credential.
Only the same trusted artifact/evaluator/corpus may finish a prepared admission.
"""
from dataclasses import asdict
import json
from pathlib import Path

from . import corpus_journal as journal, run_contract as rc

SCHEMA = "prepared-kernel-admission-v1"
FILE = "admission.json"
_PUBLISHED_METRICS = {"claim_level", "objective", "implementation_decision"}


def _read(path):
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("admission evidence exceeds bounded size")
    return rc._read(path)


def _metrics_identity(metrics):
    return journal.digest({k:v for k,v in metrics.items() if k not in _PUBLISHED_METRICS})


def _evidence(directory, record, metrics, confirmation):
    """Read actual preexisting evidence. Hashes are integrity, not signatures."""
    from .contract import load_dict
    directory = Path(directory)
    manifest = _read(directory / rc.MANIFEST)
    contract_raw = _read(directory / rc.CONTRACT)
    contract = load_dict(contract_raw)
    if (manifest.get("schema") != rc.SCHEMA
            or manifest.get("hash") != rc.digest({k:v for k,v in manifest.items() if k != "hash"})
            or manifest.get("task_hash") != rc.digest(manifest.get("task"))
            or manifest.get("workload_hash") != rc.digest(manifest.get("inputs"))
            or manifest.get("contract_hash") != contract.hash
            or manifest.get("measurement_protocol") != rc.protocol_identity()):
        raise ValueError("admission manifest/contract/evaluator drift")
    cf = _read(directory / "confirmation.json")
    first = metrics.get("run_binding")
    second = cf.get("second_binding")
    key = journal.confirmed_key(first, second)
    expected = dict(run_id=manifest["run_id"], manifest_hash=manifest["hash"],
        contract_hash=contract.hash, workload_hash=manifest["workload_hash"],
        protocol_hash=rc.digest(manifest["measurement_protocol"]))
    from .eligibility import decision_speedup
    from .workload import confirmation_evidence
    if (any(key[k] != v for k,v in expected.items())
            or cf.get("first_binding") != first or cf.get("confirmation") != confirmation
            or record["compile_info"].get("run_binding") != first
            or record["score"] != decision_speedup(metrics)
            or record["measurement_validity"] != metrics.get("measurement_validity")
            or metrics.get("contract_hash") != contract.hash):
        raise ValueError("admission candidate/measurement/confirmation drift")
    task = manifest["task"]
    if (record["algorithm"] != task["source"]["kernel_type"]
            or record["hardware"] != task["target"]["hardware"]
            or record["dsl"] != task["target"]["dsl"]):
        raise ValueError("admission task fields drift")
    from .artifact_store import ArtifactStore
    code_file = ArtifactStore._find_code_file(directory)
    if code_file is None or code_file.is_symlink():
        raise ValueError("admission candidate file missing/symlink")
    if code_file.read_text() != record["code"] or rc.code_hash(record["code"]) != key["candidate_hash"]:
        raise ValueError("admission candidate source drift")
    disk_metrics = _read(directory / "benchmark.json")
    if _metrics_identity(disk_metrics) != _metrics_identity(metrics):
        raise ValueError("admission first metrics drift")
    sm = cf["second_metrics"]
    if (sm.get("run_binding") != second or sm.get("contract_hash") != contract.hash
            or sm.get("measurement_validity") != confirmation["validity"]
            or decision_speedup(sm) != confirmation["speedup"]
            or (confirmation.get("performance_basis") and
                (sm.get("performance_verdict") or {}).get("basis") != confirmation["performance_basis"])
            or (confirmation.get("workload_evidence") and
                confirmation_evidence(sm) != confirmation["workload_evidence"])):
        raise ValueError("admission second metrics drift")
    windows = [m.get("arm_windows", {}).get("candidate") for m in (metrics, sm)]
    if any(type(w) is not str or not w for w in windows) or windows[0] == windows[1]:
        raise ValueError("admission reused/missing performance windows")
    paths = (rc.MANIFEST, rc.CONTRACT, "confirmation.json", "verification.json", code_file.name)
    hashes = {name: rc.code_hash((directory / name).read_text()) for name in paths}
    threshold = float(task["constraints"].get("min_speedup") or 1.0)
    return key, hashes, threshold


def _request_hash(request):
    return journal.digest(request)


def _validate_envelope(envelope, db):
    request = envelope["request"]
    if (envelope.get("schema") != SCHEMA or envelope.get("hash") != _request_hash(request)
            or request.get("admission_id") != journal.admission_id(request.get("key"))
            or request.get("record_hash") != journal.record_hash(request.get("record", {}))
            or request.get("corpus_path") != db.corpus_path):
        raise ValueError("prepared admission request/hash/corpus mismatch")
    return request


def prepare(directory, db, record, metrics, confirmation, reason):
    directory = Path(directory)
    raw = record.to_dict()
    serialized_confirmation = rc.json_value(asdict(confirmation))
    key, hashes, threshold = _evidence(directory, raw, metrics, serialized_confirmation)
    request = dict(admission_id=journal.admission_id(key), key=key,
        record=raw, record_hash=journal.record_hash(raw), corpus_path=db.corpus_path,
        first_metrics=metrics, confirmation=serialized_confirmation,
        evidence_hashes=hashes, min_speedup=threshold, qualification_reason=str(reason))
    # Snapshot mutable live metrics before the claim becomes persisted.
    request = json.loads(json.dumps(request, allow_nan=False))
    path = directory / FILE
    if path.exists():
        old = _validate_envelope(_read(path), db)
        fixed = ("admission_id", "key", "record_hash", "corpus_path", "confirmation",
                 "evidence_hashes", "min_speedup", "qualification_reason")
        if (any(old.get(k) != request[k] for k in fixed)
                or _metrics_identity(old["first_metrics"]) != _metrics_identity(request["first_metrics"])):
            raise ValueError("prepared admission identity/payload conflict")
        return old
    rc.atomic_json(path, dict(schema=SCHEMA, request=request, hash=_request_hash(request), status="prepared"))
    return request


def admission_metadata(eligible, reason, result=None):
    persisted = result is not None and result.persisted
    why = str(reason)
    if eligible and not persisted:
        why += " —— 符合资格，但未确认持久化: " + (result.reason if result else "no commit receipt")
    return dict(eligible=bool(eligible), kept=bool(persisted), persisted=bool(persisted),
                reason=why, commit=result.as_dict() if result is not None else None)


def publish(directory, metadata, claim, metrics=None):
    """Registered artifact fields; atomic per-file, NOT a cross-file transaction.

    A crash between files is reconciled from admission.json and the journal.
    No unrelated result/benchmark fields are reconstructed or overwritten.
    """
    directory = Path(directory)
    path = directory / "result.json"
    if path.exists():
        data = _read(path)
        data.update(corpus_admission=metadata, claim_level=claim)
        rc.atomic_json(path, data)
    path = directory / "benchmark.json"
    if path.exists():
        data = _read(path)
        if metrics is not None:
            # The original measurement keys are immutable; only registered
            # evaluator-derived publication fields may change after measuring.
            if _metrics_identity(data) != _metrics_identity(metrics):
                raise ValueError("publication attempted to replace measured metrics")
            for k in _PUBLISHED_METRICS:
                if k in metrics:
                    data[k] = metrics[k]
        data["claim_level"] = claim
        if isinstance(data.get("implementation_decision"), dict):
            data["implementation_decision"] = dict(data["implementation_decision"],
                claim_level=claim["level"], claim_ceiling=claim["ceiling"])
        rc.atomic_json(path, data)


def finish(directory, request, db, result, claim, metrics=None):
    path = Path(directory) / FILE
    envelope = _read(path)
    if _validate_envelope(envelope, db) != request:
        raise ValueError("admission receipt no longer matches prepared request")
    envelope.update(status=result.status, commit=result.as_dict())
    rc.atomic_json(path, envelope)
    metadata = admission_metadata(True, request["qualification_reason"], result)
    publish(directory, metadata, claim, metrics)
    return metadata


def recover(directory, db):
    """Explicit local side-effect recovery. Never calls a backend, LLM or bandit."""
    from .confirm import ConfirmRecord
    from .eligibility import claim_level
    from .source_lang import classify
    from .oracle import admit_to_corpus
    from .kernel_db import KernelRecord
    try:
        request = _validate_envelope(_read(Path(directory) / FILE), db)
        key, hashes, threshold = _evidence(directory, request["record"], request["first_metrics"], request["confirmation"])
        if (key != request["key"] or hashes != request["evidence_hashes"]
                or threshold != request["min_speedup"]):
            raise ValueError("prepared admission evidence changed")
        metrics = request["first_metrics"]
        allowed, reason = admit_to_corpus(metrics, ConfirmRecord(**request["confirmation"]),
            min_speedup=threshold, requires_remeasure=True)
        if not allowed:
            raise ValueError("prepared admission no longer qualifies: " + reason)
        result = db.commit(KernelRecord.from_dict(request["record"]), key)
        claim = claim_level(eligibility=metrics.get("eligibility"),
            validity=metrics.get("measurement_validity"),
            correctness_verified=metrics.get("correctness_verified"),
            coverage_mode=classify(request["record"]["code"], request["record"]["dsl"]).coverage,
            independent_confirmation=True, persisted=result.persisted).as_dict()
        finish(directory, request, db, result, claim)
        return result
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        # No GPU credential is restored, no best/recipe/budget is replayed, and
        # no cross-version drift is silently accepted.
        return journal.CommitResult("recovery_hold", reason=f"{type(exc).__name__}: {exc}")


def main(argv=None):
    import argparse
    from .kernel_db import KernelDatabase
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", required=True)
    p.add_argument("--corpus", required=True)
    args = p.parse_args(argv)
    result = recover(args.artifact, KernelDatabase(args.corpus, persist_retrievals=False))
    print(json.dumps(result.as_dict(), ensure_ascii=False))
    return 0 if result.persisted else 1


if __name__ == "__main__":
    raise SystemExit(main())
