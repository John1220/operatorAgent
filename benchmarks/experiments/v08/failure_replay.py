"""Replay one recorded failure and check that it still fails the same way.

A replay case is a directory with:

* ``case.json`` — frozen declaration, contract hash, and the **expected** failure
  (levels, producer, line/column, root cause);
* ``candidate.py`` — the source that failed, byte-for-byte;
* optionally ``candidate_fixed.py`` — the patch under test.

The runner uses the same verification entry as production
(``operator_agent.app.hitl_tools``), so a replay is not a second judge: it is the
same L1/L2 path the arm used. It writes ``replay-<label>.json`` next to the case
and exits non-zero when the observation does not match the frozen expectation
(for the failing source) or does not pass L2 (for the fixed source).

Examples
--------

Reproduce the recorded failure::

    python -m benchmarks.experiments.v08.failure_replay run \\
        --case benchmarks/repro/fused_ln_compile_20261118 \\
        --source candidate.py --host h20 --work-dir /root/operator-p2-seed-20261116

Check the patch::

    python -m benchmarks.experiments.v08.failure_replay run \\
        --case benchmarks/repro/fused_ln_compile_20261118 \\
        --source candidate_fixed.py --expect pass --host h20 \\
        --work-dir /root/operator-p2-seed-20261116
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

from operator_agent.app import hitl_tools
from operator_agent.execute.backend_ownership import owned_worker_scope

SCHEMA = "v08-failure-replay-v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _portable(path):
    """Provenance paths in a receipt are committed, so they must not be absolute.

    Origin is a public repo: an absolute path leaks the author's home directory
    and is meaningless on another machine. Paths inside the checkout become
    repo-relative; paths outside it keep their absolute form because there is no
    portable spelling for them. Only the receipt is rewritten -- callers still
    read evidence through the absolute path they were given.
    """
    if not path:
        return path
    raw = Path(str(path))
    if not raw.is_absolute():
        return str(path)
    try:
        return str(raw.resolve().relative_to(_REPO_ROOT))
    except (ValueError, OSError):
        return str(path)


def _verdict(receipt, level):
    for verdict in (receipt or {}).get("verdicts") or []:
        if verdict.get("level") == level:
            return verdict
    return {}


def _compile_failure(receipt):
    metrics = _verdict(receipt, "L2").get("metrics") or {}
    return metrics.get("candidate_compile_failure") or {}


def observe(case, source_text, *, host, work_dir, timeout, backend="ssh", final=False):
    """Run one replay through the production verification entry."""
    budget = dict(case.get("budget") or {})
    # 每次回放一个**新** run 目录：host task 的验证预算是 write-once 的，
    # 复用同一个目录会撞 `verification budget exhausted`（实测）。
    run = Path(case["_root"]) / "run" / f"{case['_label']}-{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    hitl_tools.prepare(case["declaration"], run, host_budget=dict(
        wall_clock_s=budget.get("wall_clock_s", 900),
        max_verifications=budget.get("max_verifications", 2),
        max_model_calls=budget.get("max_model_calls", 1),
        max_output_tokens_per_call=budget.get("max_output_tokens_per_call", 8192),
        total_output_token_reservations=budget.get(
            "total_output_token_reservations", 8192)))
    worker = run / "owned-worker" / uuid.uuid4().hex
    worker.mkdir(parents=True, exist_ok=True)
    with owned_worker_scope(worker, uuid.uuid4().hex):
        response = hitl_tools.verify(run, source_text, backend=backend, host=host,
                                     work_dir=work_dir, timeout=timeout,
                                     persistent=True, stop_after="L2",
                                     candidate_origin="outer")
    evidence = response.get("evidence")
    receipt = _read(evidence) if evidence else {}
    body = dict(response=response, receipt=receipt, run_dir=str(run), evidence=evidence,
                final=None, final_evidence=None)
    if final and (receipt.get("levels") or {}).get("L2") is True:
        # 独立 L2 回执：同一 host task 上的 `purpose=final`，占预留的那 1 次。
        worker_final = run / "owned-worker" / uuid.uuid4().hex
        worker_final.mkdir(parents=True, exist_ok=True)
        with owned_worker_scope(worker_final, uuid.uuid4().hex):
            final_response = hitl_tools.verify(
                run, source_text, backend=backend, host=host, work_dir=work_dir,
                timeout=timeout, persistent=True, stop_after="L2",
                verification_purpose="final", candidate_origin="outer")
        final_evidence = final_response.get("evidence")
        body.update(final=final_response, final_evidence=final_evidence,
                    final_receipt=_read(final_evidence) if final_evidence else {})
    return body


def _case_rows(receipt):
    for verdict in (receipt or {}).get("verdicts") or []:
        if verdict.get("level") == "L2":
            metrics = verdict.get("metrics") or {}
            return (metrics.get("correctness") or {}).get("cases") or []
    return []


def _case_verdict(row):
    """One case row -> (verdict, max_error, matched_ratio).

    A case that passed but only carries ``bad_elements`` counts as passed; a
    missing verdict is ``unknown`` and never silently treated as a pass.
    """
    metrics = row.get("metrics") or {}
    verdict = row.get("verdict")
    if verdict is None:
        if metrics.get("bad_elements") == 0:
            verdict = "passed"
        elif metrics.get("bad_elements"):
            verdict = "failed"
        else:
            verdict = "unknown"
    return verdict, metrics.get("max_error"), metrics.get("matched_ratio")


def _close(observed, expected, tolerance=1e-6):
    if observed is None or expected is None:
        return False
    return abs(float(observed) - float(expected)) <= max(tolerance, abs(expected) * 1e-3)


def _matches(case, observed, *, source_name=None):
    """冻结预期 vs 观测：级别、具名用例，**以及源码/契约身份**。

    身份也要进判据，不能只记字段：2016-11-19 的复核指出，源码哈希与契约哈希此前只是
    回执里的记录项，于是「回放器跑的是另一份源码/另一份契约」也能显示 passed。
    """
    expected = case.get("expected") or {}
    want_contract = case.get("contract_hash")
    if want_contract and observed.get("contract_hash") != want_contract:
        return False, (f"contract_hash: expected {want_contract}, "
                       f"observed {observed.get('contract_hash')}")
    recorded = (case.get("origin") or {}).get("source_hash")
    replaying_recorded_source = source_name in (None, "candidate.py")
    if recorded and replaying_recorded_source and observed.get("source_hash") != recorded:
        return False, (f"source_hash: expected {recorded}, "
                       f"observed {observed.get('source_hash')}")
    levels = observed.get("levels") or {}
    for level, want in (expected.get("levels") or {}).items():
        if levels.get(level) is not want:
            return False, f"{level}: expected {want}, observed {levels.get(level)!r}"
    failure = observed.get("compile_failure") or {}
    for key in ("producer", "line", "column"):
        if key in expected and failure.get(key) != expected[key]:
            return False, f"{key}: expected {expected[key]!r}, observed {failure.get(key)!r}"
    rows = {row.get("case_id"): _case_verdict(row)
            for row in observed.get("cases") or []}
    for case_id in expected.get("passing_cases") or ():
        verdict, _, _ = rows.get(case_id, ("missing", None, None))
        if verdict != "passed":
            return False, f"{case_id}: expected passed, observed {verdict}"
    for want in expected.get("failing_cases") or ():
        case_id = want.get("case_id")
        verdict, max_error, matched = rows.get(case_id, ("missing", None, None))
        if verdict != want.get("verdict", "failed"):
            return False, f"{case_id}: expected {want.get('verdict', 'failed')}, observed {verdict}"
        if "max_error" in want and not _close(max_error, want["max_error"]):
            return False, f"{case_id}: max_error {max_error} != {want['max_error']}"
        if "matched_ratio" in want and not _close(matched, want["matched_ratio"]):
            return False, f"{case_id}: matched_ratio {matched} != {want['matched_ratio']}"
    return True, ("matches the frozen expectation "
                  "(contract + source identity + levels + the named cases)")


def run_case(*, case_dir, source_name, expect, host, work_dir, timeout, output=None,
             final=False):
    case_dir = Path(case_dir).resolve()
    case = _read(case_dir / "case.json")
    case["_root"] = str(case_dir)
    case["_label"] = Path(source_name).stem
    source_path = case_dir / source_name
    source_text = source_path.read_text(encoding="utf-8")
    observed_raw = observe(case, source_text, host=host, work_dir=work_dir,
                           timeout=timeout, final=final)
    receipt = observed_raw["receipt"]
    levels = receipt.get("levels") or {}
    failure = _compile_failure(receipt)
    observed = dict(levels=levels, compile_failure=failure,
                    cases=_case_rows(receipt),
                    contract_hash=receipt.get("contract_hash"),
                    source_hash=_sha(source_text),
                    expected_source_hash=(case.get("origin") or {}).get("source_hash"))
    if expect == "fail":
        matched, why = _matches(case, observed, source_name=source_name)
        passed = matched
    else:
        cases = ((_verdict(receipt, "L2").get("metrics") or {}).get("correctness")
                 or {}).get("cases") or []
        ok_cases = sum(1 for row in cases
                       if row.get("verdict") == "passed"
                       or (row.get("metrics") or {}).get("bad_elements") == 0)
        passed = levels.get("L2") is True and len(cases) > 0 and ok_cases == len(cases)
        matched, why = passed, (f"L2={levels.get('L2')} cases={ok_cases}/{len(cases)}")
    final_receipt = observed_raw.get("final_receipt") or {}
    if expect == "pass" and final:
        final_levels = final_receipt.get("levels") or {}
        passed = passed and final_levels.get("L2") is True
        why = f"{why}; independent final L2={final_levels.get('L2')}"
    record = dict(schema=SCHEMA, case_id=case.get("case_id"), source=source_name,
                  source_hash=observed["source_hash"], expect=expect, passed=passed,
                  why=why, levels=levels, compile_failure=failure,
                  contract_hash=receipt.get("contract_hash"),
                  evidence=_portable(observed_raw["evidence"]),
                  run_dir=_portable(observed_raw["run_dir"]),
                  final_levels=(final_receipt.get("levels") or None),
                  final_evidence=_portable(
                      observed_raw.get("final_evidence")),
                  final_verification_purpose=final_receipt.get("verification_purpose"))
    target = Path(output) if output else case_dir / f"replay-{case['_label']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in
                      ("case_id", "source", "expect", "passed", "why")}, ensure_ascii=False))
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--source", default="candidate.py",
                        help="file inside the case directory (candidate.py / candidate_fixed.py)")
    parser.add_argument("--expect", default=None, choices=["fail", "pass"],
                        help="default: case.json's expected levels decide it")
    parser.add_argument("--host", default="h20")
    # 远端工作目录默认落在 /data3：H20 的 / 只有个位数 GB（见 docs/testing/remote-workdir.md）。
    parser.add_argument("--work-dir",
                        default=os.environ.get("OPERATOR_REMOTE_WORK_DIR", "/data3/operator-work"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--final", action="store_true",
                        help="after L2 passes, also take the reserved purpose=final L2 receipt")
    args = parser.parse_args(argv)
    expect = args.expect
    if expect is None:
        case = _read(Path(args.case) / "case.json")
        expect = "fail" if (case.get("expected") or {}).get("levels", {}).get("L2") is False else "pass"
    return run_case(case_dir=args.case, source_name=args.source, expect=expect,
                    host=args.host, work_dir=args.work_dir, timeout=args.timeout,
                    output=args.output, final=args.final)


if __name__ == "__main__":
    raise SystemExit(main())
