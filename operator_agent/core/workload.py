"""Compose case evidence, never invent a second measurement policy.

All case masks come from the same production Oracle. The workload mask is their
intersection over the contract's required/weighted/regression scope. Extra
diagnostic cases do not silently change a declared objective.
"""
import copy
import json
import math
from collections.abc import Mapping

from ..decide.eligibility import ALL_FALSE, case_speedups, decision_speedup, performance_verdict, regression_over_limit


def scope_cases(contract, primary):
    keys = (set(contract.required_cases) | {k for k, _ in contract.case_weights}
            | {k for k, _ in contract.max_regression})
    return sorted(keys or {primary})


def qualification(ledger, scope):
    masks, tiers, reasons = [], [], []
    for case in scope:
        row = ledger.get(case) or {}
        mask = row.get("eligibility") or {}
        value = row.get("speedup_vs_baseline")
        complete = (isinstance(mask, Mapping)
                    and all(type(mask.get(k)) is bool for k in ALL_FALSE.as_dict())
                    and type(value) in (int, float) and math.isfinite(value) and value > 0
                    and row.get("measurement_validity") in ("valid", "degraded", "uninformative"))
        masks.append(mask if complete else ALL_FALSE.as_dict())
        tiers.append(row.get("measurement_validity") if complete else "uninformative")
        if not complete or not all(mask.values()):
            reasons.append(case + ": " + (row.get("reason") or "; ".join(
                row.get("measurement_validity_reasons") or
                (row.get("observation") or {}).get("reasons") or ["missing/ineligible case evidence"])))
    combined = {k: bool(masks) and all(m.get(k) is True for m in masks)
                for k in ALL_FALSE.as_dict()}
    tier = ("uninformative" if not tiers or "uninformative" in tiers else
            "degraded" if "degraded" in tiers else "valid")
    return dict(scope_cases=list(scope), eligibility=combined, validity=tier, reasons=reasons)


def apply_qualification(metrics, contract, primary):
    ledger = metrics.get("cases") or {primary: copy.deepcopy(metrics)}
    summary = qualification(ledger, scope_cases(contract, primary))
    metrics["workload_qualification"] = summary
    metrics["workload_contract"] = json.loads(json.dumps(contract.as_dict(), allow_nan=False))
    metrics["eligibility"] = summary["eligibility"]
    metrics["measurement_validity"] = summary["validity"]
    metrics["measurement_can_gate"] = summary["eligibility"]["can_verdict"]
    metrics["measurement_can_persist"] = summary["eligibility"]["can_persist"]
    metrics["measurement_can_count_progress"] = summary["eligibility"]["can_count_progress"]
    metrics["measurement_validity_reasons"] = list(dict.fromkeys(
        metrics.get("measurement_validity_reasons", []) + summary["reasons"]))
    # The published Observation and mask must agree. Per-case observations remain
    # unchanged in the ledger; provenance names the composed scope explicitly.
    obs = metrics.get("observation")
    if isinstance(obs, dict):
        obs.update(eligibility=summary["eligibility"], validity=summary["validity"])
        obs["reasons"] = list(dict.fromkeys(obs.get("reasons", []) + summary["reasons"]))
        obs.setdefault("provenance", {})["workload_qualification"] = copy.deepcopy(summary)


def confirmation_evidence(metrics):
    return copy.deepcopy({k: metrics[k] for k in
        ("performance_verdict", "workload_qualification", "cases", "eligibility",
         "speedup_vs_baseline", "contract_hash", "observation", "measurement_validity",
         "measurement_validity_reasons", "arm_windows", "arm_device_labels",
         "backend_real_device", "workload_contract") if k in metrics})


def confirm_workload(first, second):
    """Require the same objective, usable full scope and fresh per-case windows.

    Legacy single-case confirmation stays in confirm_persist. Production
    declared workload evidence is mandatory, never an opt-out flag.
    """
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return False, "workload confirmation evidence is not a mapping"
    v1, v2 = first.get("performance_verdict") or {}, second.get("performance_verdict") or {}
    q1, q2 = first.get("workload_qualification") or {}, second.get("workload_qualification") or {}
    if not all(isinstance(x, Mapping) for x in (v1, v2, q1, q2)):
        return False, "workload confirmation malformed criterion/qualification"
    if not q1 and v1.get("basis") != "objective":
        return True, "legacy primary case"
    from ..core.contract import load_dict
    try:
        contract = load_dict(first.get("workload_contract") or {})
        if (first.get("contract_hash") != contract.hash
                or load_dict(second.get("workload_contract") or {}).hash != contract.hash
                or second.get("contract_hash") != contract.hash):
            return False, "workload confirmation frozen contract changed"
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return False, "workload confirmation frozen contract missing/invalid: " + str(exc)
    if not q1 or not q2 or q1.get("scope_cases") != q2.get("scope_cases"):
        return False, "workload confirmation scope missing/changed"
    scope = q1.get("scope_cases")
    if (not isinstance(scope, list) or not scope
            or any(type(k) is not str or not k for k in scope)
            or scope != sorted(set(scope))):
        return False, "workload confirmation scope is malformed"
    criterion = ("basis", "min_speedup", "aggregate", "case_weights", "max_regression", "required_cases")
    if any(v1.get(k) != v2.get(k) for k in criterion):
        return False, "workload confirmation criterion changed"
    expected = dict(min_speedup=contract.min_speedup, aggregate=contract.aggregate or "",
        case_weights=dict(contract.case_weights), max_regression=dict(contract.max_regression),
        required_cases=list(contract.required_cases))
    if any(v1.get(k) != value for k, value in expected.items()):
        return False, "workload criterion differs from frozen contract"
    if v1.get("passed") is not True or v2.get("passed") is not True:
        return False, "workload objective/regression gate not confirmed"
    if (q1.get("eligibility", {}).get("can_persist") is not True
            or q2.get("eligibility", {}).get("can_verdict") is not True
            or decision_speedup(second) is None):
        return False, "workload confirmation has missing/ineligible case"
    for m, v, q in ((first, v1, q1), (second, v2, q2)):
        declared_scope = (set(v.get("required_cases") or ()) | set(v.get("case_weights") or {})
                          | set(v.get("max_regression") or {}))
        if declared_scope and sorted(declared_scope) != q["scope_cases"]:
            return False, "workload scope differs from declared criterion"
        values = case_speedups(m, case_key_hint=q["scope_cases"][0],
                               speedup_hint=m.get("speedup_vs_baseline"))
        regression, _ = regression_over_limit(v.get("max_regression"), values)
        passed, basis, _, value = performance_verdict(v.get("min_speedup", 1.0),
            aggregate=v.get("aggregate", ""), case_weights=v.get("case_weights", {}),
            case_speedups=values, primary_speedup=v.get("value") if v.get("basis") == "primary_case"
                                                 else m.get("speedup_vs_baseline"),
            max_regression=v.get("max_regression"), regression_over_limit_value=regression)
        if (not passed or basis != v.get("basis") or value != v.get("value")
                or regression != v.get("regression_over_limit")):
            return False, "workload verdict does not match measured case evidence"
    ledgers = [m.get("cases") or ({scope[0]: m} if len(scope) == 1 else {})
               for m in (first, second)]
    if any(ledgers):
        windows = []
        for m, q, ledger in zip((first, second), (q1, q2), ledgers):
            actual = qualification(ledger, q["scope_cases"])
            if actual["eligibility"] != q.get("eligibility") or actual["validity"] != q.get("validity"):
                return False, "workload summary does not match case evidence"
            for case in q["scope_cases"]:
                row = ledger.get(case) or {}
                obs = row.get("observation") or {}
                if (obs.get("refused") or obs.get("ruler_broken")
                        or obs.get("eligibility") != row.get("eligibility")
                        or obs.get("validity") != row.get("measurement_validity")
                        or obs.get("value") != row.get("speedup_vs_baseline")
                        or obs.get("provenance", {}).get("contract_hash") != contract.hash):
                    return False, "workload case Oracle receipt missing/inconsistent: " + case
                contract_hash = (first.get("contract_hash") or
                                 (first.get("observation") or {}).get("provenance", {}).get("contract_hash"))
                if contract_hash and row.get("contract_hash") != contract_hash:
                    return False, "workload case contract changed: " + case
            for case in q1["scope_cases"]:
                row = ledger.get(case) or {}
                obs = row.get("observation") or {}
                pair = row.get("arm_windows") or {}
                window = pair.get("candidate")
                if window or pair.get("baseline"):
                    if type(window) is not str or not window or window != pair.get("baseline"):
                        return False, "workload confirmation missing/cross-arm window: " + case
                    windows.append(window)
                elif (row.get("backend_real_device") is True or "real_device" in contract.required_caps
                      or obs.get("provenance", {}).get("rule_inputs", {}).get("real_device") is True):
                    return False, "workload confirmation missing real-device window: " + case
        if len(windows) != len(set(windows)):
            return False, "workload confirmation reused case window"
    return True, "same workload confirmed in fresh case windows"
