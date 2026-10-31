"""Session-local search over one frozen workload objective.

Qualification/aggregation/regression and significance policies stay in workload,
eligibility and measurement. This module composes them, retains the candidate
that supplied a score, and serializes audit evidence -- never live credentials.
Legacy undeclared searches remain explicitly primary-GFLOPS based.
"""
import copy
import math
from collections.abc import Mapping

from ..core.contract import load_dict
from ..decide.eligibility import (aggregate_speedup, case_speedups,
                          decision_speedup, performance_verdict, regression_over_limit)
from ..decide.measurement import correctness_verified_levels, reference_ceiling_reached, verdict, Validity
from ..core.run_contract import audit_value, code_hash, digest, json_value
from ..core.schemas import VerificationLevel, VerificationResult, UnifiedSketch
from ..core.workload import qualification, scope_cases

SCHEMA = "workload-search-v1"


def identity(contract):
    return dict(schema=SCHEMA, basis="objective", contract_hash=contract.hash,
        aggregate=contract.aggregate or "", case_weights=dict(contract.case_weights),
        max_regression=dict(contract.max_regression), required_cases=list(contract.required_cases),
        min_speedup=contract.min_speedup)


def reading(metrics, contract):
    """Check the existing Oracle receipts against the frozen objective.

    Being below min_speedup is a usable search observation, not a qualification
    failure. Missing coverage or a hard regression is NOT an unconstrained win.
    """
    result = dict(identity=identity(contract), value=None, score=None, can_tune=False,
                  can_count_progress=False, can_select=False, reasons=[])
    try:
        if not isinstance(metrics, Mapping):
            raise ValueError("no L3 objective evidence")
        v = metrics.get("performance_verdict") or {}
        expected = identity(contract)
        if (type(v.get("min_speedup")) not in (int, float)
                or any(type(x) not in (int, float) for name in ("case_weights", "max_regression")
                       for x in (v.get(name) or {}).values())):
            raise ValueError("search criterion has non-numeric/bool values")
        if any(v.get(k) != expected[k] for k in expected if k not in ("schema", "contract_hash")):
            raise ValueError("search criterion differs from frozen contract")
        if (metrics.get("contract_hash") != contract.hash
                or load_dict(metrics.get("workload_contract") or {}).hash != contract.hash):
            raise ValueError("search workload contract missing/changed")
        ledger = metrics.get("cases") or {}
        scope = scope_cases(contract, "")
        actual = qualification(ledger, scope)
        q = metrics.get("workload_qualification") or {}
        if (q.get("scope_cases") != scope or q.get("eligibility") != actual["eligibility"]
                or q.get("validity") != actual["validity"]
                or metrics.get("eligibility") != actual["eligibility"]
                or metrics.get("measurement_validity") != actual["validity"]):
            raise ValueError("search qualification differs from full case scope")
        windows = []
        for case in scope:
            row = ledger.get(case) or {}
            obs = row.get("observation") or {}
            if (row.get("contract_hash") != contract.hash or obs.get("refused") or obs.get("ruler_broken")
                    or obs.get("value") != row.get("speedup_vs_baseline")
                    or obs.get("validity") != row.get("measurement_validity")
                    or obs.get("eligibility") != row.get("eligibility")
                    or obs.get("provenance", {}).get("contract_hash") != contract.hash):
                raise ValueError("missing/inconsistent case Oracle receipt: " + case)
            pair = row.get("arm_windows") or {}
            window = pair.get("candidate")
            if window or pair.get("baseline"):
                if type(window) is not str or not window or window != pair.get("baseline"):
                    raise ValueError("case has missing/cross-arm search window: " + case)
                windows.append(window)
            elif ("real_device" in contract.required_caps or row.get("backend_real_device") is True
                    or obs.get("provenance", {}).get("rule_inputs", {}).get("real_device") is True):
                raise ValueError("real case missing search window: " + case)
        if len(windows) != len(set(windows)):
            raise ValueError("search reused case windows")
        values = case_speedups(metrics)
        value, _ = aggregate_speedup(contract.case_weights, values, contract.aggregate)
        regression, _ = regression_over_limit(contract.max_regression, values)
        passed, _, _, _ = performance_verdict(contract.min_speedup, aggregate=contract.aggregate,
            case_weights=contract.case_weights, case_speedups=values,
            max_regression=contract.max_regression, regression_over_limit_value=regression)
        if v.get("passed") is not passed or v.get("regression_over_limit") is not regression:
            raise ValueError("search target/regression verdict not recomputable")
        if value is None or value != decision_speedup(metrics):
            raise ValueError("search objective missing/inconsistent")
        result["value"] = value  # Retained diagnostics, even when infeasible.
        if contract.max_regression and regression is not False:
            raise ValueError("hard regression violated or missing evidence")
        mask = actual["eligibility"]
        if mask.get("can_verdict") is not True:
            raise ValueError("workload has no verdict qualification")
        result.update(score=value, can_tune=mask.get("can_tune") is True,
                      can_count_progress=mask.get("can_count_progress") is True,
                      can_select=mask.get("can_count_progress") is True)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        result["reasons"].append(str(exc))
    return result


def results_dict(results):
    return [dict(level=r.level.value, passed=r.passed, message=r.message,
                 error_log=r.error_log, metrics=copy.deepcopy(r.metrics)) for r in results]


def results_load(rows):
    return [VerificationResult(level=VerificationLevel(r["level"]), passed=r["passed"],
        message=r["message"], error_log=r.get("error_log"), metrics=r["metrics"]) for r in rows]


class SearchObjective:
    def __init__(self, contract):
        self.contract = contract
        self.best = None
        self.seen_windows = set()  # Live window receipts NEVER restored from audit.

    @classmethod
    def for_task(cls, task):
        from ..core.contract import freeze_from_task
        contract = freeze_from_task(task)
        return cls(contract) if contract.case_weights or contract.aggregate else None

    def observe(self, *, code, sketch, results, round_num):
        l3 = next((r for r in results if r.level is VerificationLevel.L3_PERFORMANCE), None)
        point = reading(l3.metrics if l3 else None, self.contract)
        correct = correctness_verified_levels({r.level.value: r.passed for r in results})
        correct = correct and all((r.metrics.get("run_binding") or {}).get("candidate_hash") == code_hash(code)
            and (r.metrics.get("run_binding") or {}).get("contract_hash") == self.contract.hash for r in results)
        if not correct:
            point.update(can_tune=False, can_count_progress=False, can_select=False, score=None)
            point["reasons"].append("candidate L1/L2 correctness not established")
        windows = set()
        ledger = (l3.metrics.get("cases") or {}) if l3 else {}
        if isinstance(ledger, Mapping):
            for case in scope_cases(self.contract, ""):
                row = ledger.get(case)
                pair = row.get("arm_windows") if isinstance(row, Mapping) else None
                window = pair.get("candidate") if isinstance(pair, Mapping) else None
                if type(window) is str and window:
                    windows.add(window)
        if windows & self.seen_windows:
            point.update(can_tune=False, can_count_progress=False, can_select=False, score=None)
            point["reasons"].append("reused search measurement window")
        self.seen_windows.update(windows)
        before = self.best["score"] if self.best else None
        point.update(best_before=before, reward=None, comparison=None, comparison_reason="",
                     comparison_band=None, best_updated=False)
        # Compute reward BEFORE changing the winner; do not turn every win into 0.
        if point["can_count_progress"] and point["can_tune"] and before is not None:
            point["reward"] = (point["score"] - before) / before
            if not math.isfinite(point["reward"]):
                point["reward"] = None
            comparison, why, band = verdict(point["score"], before, Validity.VALID)
            point.update(comparison=comparison.name, comparison_reason=why, comparison_band=band)
        if point["can_select"] and (before is None or point["score"] > before):
            self.best = dict(round=round_num, score=point["score"], candidate_hash=code_hash(code),
                code=code, sketch=sketch.to_dict(), verification_results=results_dict(results))
            point["best_updated"] = True
        point["best_after"] = self.best["score"] if self.best else None
        return point

    def snapshot(self):
        return json_value(audit_value(dict(identity=identity(self.contract), best=copy.deepcopy(self.best),
                                           evidence_scope="current_process_selection_only")))

    def winner(self):
        if self.best is None:
            return None
        return (self.best["code"], UnifiedSketch.from_dict(copy.deepcopy(self.best["sketch"])),
                results_load(copy.deepcopy(self.best["verification_results"])))

    def at_reference_ceiling(self, metrics):
        point = reading(metrics, self.contract)
        if not point["can_count_progress"]:
            return False
        return all(reference_ceiling_reached(metrics["cases"][case])
                   for case in scope_cases(self.contract, ""))


def validate_snapshot(snapshot, contract, round_done):
    """Validate a historical proposal; caller must reverify before ALL effects."""
    if (not isinstance(snapshot, dict) or digest(snapshot.get("identity")) != digest(identity(contract))
            or snapshot.get("evidence_scope") != "current_process_selection_only"):
        raise ValueError("checkpoint search objective identity missing/changed")
    best = snapshot.get("best")
    if best is None:
        return
    if (not isinstance(best, dict) or type(best.get("round")) is not int
            or not 1 <= best["round"] <= round_done or type(best.get("code")) is not str
            or type(best.get("score")) not in (int, float)
            or best.get("candidate_hash") != code_hash(best["code"])
            or not isinstance(best.get("sketch"), dict)):
        raise ValueError("checkpoint best candidate identity invalid")
    rows = best.get("verification_results")
    if (not isinstance(rows, list) or len(rows) != 3
            or any(not isinstance(r, dict) or not isinstance(r.get("metrics"), dict)
                   or type(r.get("passed")) is not bool for r in rows)
            or {r.get("level") for r in rows} != {"L1", "L2", "L3"}):
        raise ValueError("checkpoint best candidate verification structure invalid")
    results = results_load(rows)
    if not correctness_verified_levels({r.level.value: r.passed for r in results}):
        raise ValueError("checkpoint best candidate correctness missing")
    l3 = next((r for r in results if r.level is VerificationLevel.L3_PERFORMANCE), None)
    point = reading(l3.metrics if l3 else None, contract)
    if not point["can_select"] or point["score"] != best.get("score"):
        raise ValueError("checkpoint best candidate score/qualification changed")
    for r in results:
        binding = r.metrics.get("run_binding") or {}
        if binding.get("candidate_hash") != best["candidate_hash"] or binding.get("contract_hash") != contract.hash:
            raise ValueError("checkpoint best candidate verification binding changed")
