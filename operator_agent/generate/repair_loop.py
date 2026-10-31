"""给一份**已有的**核，按处方自动修到过（或 N 轮）

为什么单独有这一层：`Conductor` 的路是「从零生成 → 判 → 修」，而真实使用里经常是
「我已经有一份核（人写的、别处拷来的、上一轮的遗留），让它过」。`operator_run` 不接受
种子候选，`operator_verify` 只判不修 —— 中间缺一格。

这一层只做三件事：**判 → 读处方 → 调 Fixer → 再判**，最多 N 轮。判据全部来自既有模块
（`Verifier` / `failure_route` / `Fixer`），这里不新增任何判据、不放宽任何阈值。

它同时是「自动修复能力」的量具：每个 bug 跑一遍，记下几轮改对、初次落在哪一级。
"""

from __future__ import annotations

import contextlib
import io
from typing import Optional

from ..generate.bounded_repair import source_change
from ..decide.failure_route import classify
from ..generate.fixer import FixerAgent
from ..generate.generation_context import records_cursor, records_since
from ..decide.measurement import TRANSPORT_KEY, TRANSPORT_UNKNOWN
from ..generate.repair_diagnoser import diagnose_candidate
from ..core.run_contract import code_hash
from ..core.schemas import VerificationLevel
from ..decide.verifier import Verifier


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        value = fn(*a, **kw)
    return value, buf.getvalue()


def _full_l2_passed(results):
    """Three-state full-suite evidence; missing stages or suite stay unknown."""
    l1 = [r for r in results if r.level == VerificationLevel.L1_COMPILE]
    l2 = [r for r in results if r.level == VerificationLevel.L2_FUNCTION]
    if any(not r.passed for r in l1):
        return False
    if not l1 or not l2:
        return None
    for result in l2:
        metrics = result.metrics or {}
        if (metrics.get('correctness_primary_verdict') == 'unknown'
                or (metrics.get('correctness') or {}).get('verdict') == 'unknown'
                or metrics.get('reference_identity_unknown')
                or metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN
                or ('input_fidelity_verified' in metrics and metrics['input_fidelity_verified'] is not True)):
            return None
    verdicts = [(r.metrics.get('correctness') or {}).get('verdict') for r in l2]
    if any(not r.passed for r in l2) or 'failed' in verdicts:
        return False
    return True if all(v == 'passed' for v in verdicts) else None


def _l2_known(results):
    for result in results:
        if result.level != VerificationLevel.L2_FUNCTION:
            continue
        metrics = result.metrics or {}
        if (metrics.get('correctness_primary_verdict') == 'unknown'
                or (metrics.get('correctness') or {}).get('verdict') in ('unknown', 'failed')
                or metrics.get('reference_identity_unknown')
                or metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN
                or ('input_fidelity_verified' in metrics and metrics['input_fidelity_verified'] is not True)):
            return False
    return True


def _repair_authority(results):
    """Bad numerical code can be repaired; unqualified timing cannot drive it."""
    failures = [r for r in results if not r.passed]
    correctness_failures = [r for r in failures if r.level in
                            (VerificationLevel.L1_COMPILE, VerificationLevel.L2_FUNCTION)]
    if correctness_failures:
        for result in correctness_failures:
            metrics = result.metrics or {}
            if (metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN
                    or metrics.get('reference_identity_unknown')
                    or metrics.get('correctness_primary_verdict') == 'unknown'
                    or (metrics.get('correctness') or {}).get('verdict') == 'unknown'
                    or ('input_fidelity_verified' in metrics
                        and metrics['input_fidelity_verified'] is not True)):
                return None, dict(present=True, action='hold_for_measurement',
                    why='正确性观测未知，不能根据这份证据修复候选')
        return 'correctness_failure', dict(present=False)
    performance = [r for r in results if r.level == VerificationLevel.L3_PERFORMANCE]
    for result in performance:
        metrics = result.metrics or {}
        eligibility = (metrics.get('observation') or {}).get('eligibility') or {}
        if (eligibility.get('can_tune') is not True or metrics.get('ruler_broken')
                or metrics.get('measurement_validity') in ('uninformative', 'invalid', 'unknown')):
            return None, dict(present=True, action='hold_for_measurement',
                why='性能证据缺少可行动资格，先恢复测量资格，不据此调用 Fixer')
    return ('qualified_performance' if performance else None), dict(present=False)


def _change(parent, candidate):
    try:
        return source_change(parent, candidate)
    except (ValueError, SyntaxError, RecursionError):
        # An L1 repair may start from unparsable code (or a non-Python DSL).
        # Exact equality is still knowable; absence of an AST grants no pass.
        return dict(schema='repair-source-change-v1', parent_hash=code_hash(parent),
            candidate_hash=code_hash(candidate), parent_ast_hash=None, candidate_ast_hash=None,
            change='identical_source' if parent == candidate else 'ast_unavailable',
            correctness_proven=False)


def repair(code: str, task, test_inputs, verifier: Verifier,
           fixer: Optional[FixerAgent] = None, *,
           max_rounds: int = 2, expected_output=None,
           target_gflops: float = 0.0,
           stop_after: Optional[VerificationLevel] = None,
           use_diagnosis: bool = True, on_verification=None,
           before_verification=None) -> dict:
    """反复「判 → 修 → 再判」。返回每一轮的判定、处方与修复摘要。

    * `route.counts_as_iteration = False`（`hold`：静默退化 / 参考侧无对象）⇒ **不修**，
      直接停并说明理由 —— 那是「这一轮什么都没判」，拿它去修等于对着噪声开处方。
    * `Fixer` 只回一段说明、不给代码（模型在提问）⇒ 停，并把提问原文带出来。
    * `use_diagnosis=False` 只关闭诊断上下文，保留相同验证门、资格和伪修复检查。
      每轮完整套件结果单独用 `full_l2_passed` 三态记录，不能从 L1 或空列表推断。
    """
    if max_rounds < 1:
        raise ValueError("max_rounds 至少为 1")
    fixer = fixer or FixerAgent(llm=None)
    current = code
    rounds: list[dict] = []
    log_chunks: list[str] = []
    results = []
    passed = False
    candidate_records = []
    stopped_reason = None

    for index in range(1, int(max_rounds) + 1):
        if before_verification is not None and before_verification(index, current) is False:
            # No candidate was dispatched. Do not manufacture a completed step
            # or inherit an earlier candidate's passed flag.
            from ..core.schemas import VerificationResult
            passed = False
            stopped_reason = 'verification_budget_exhausted'
            results = [VerificationResult(level=VerificationLevel.L1_COMPILE, passed=False,
                message='Verification budget exhausted before dispatch',
                metrics=dict(compile_status='budget_not_dispatched', verification_dispatched=False))]
            break
        results, log = _quiet(verifier.verify, current, task, test_inputs,
                              expected_output, float(target_gflops), None, stop_after)
        log_chunks.append(log)
        l2 = [r for r in results if r.level == VerificationLevel.L2_FUNCTION]
        basis, conflict = _repair_authority(results)
        passed = (bool(l2) and all(r.passed for r in results) and _l2_known(results)
                  and not conflict['present'])
        route = classify(results)
        masks = [(r.metrics.get('observation') or {}).get('eligibility') for r in results]
        entry = {
            "round": index,
            "levels": {r.level.value: bool(r.passed) for r in results},
            "passed": passed,
            "route": route.as_dict(),
            "prescription": route.reason,
            "source_hash": code_hash(current),
            "full_l2_passed": _full_l2_passed(results),
            "repair_basis": basis,
            "conflict": conflict,
            "eligibility": next((dict(mask) for mask in reversed(masks) if isinstance(mask, dict)), None),
            "diagnosis": diagnose_candidate(current, task, results) if use_diagnosis else None,
        }
        rounds.append(entry)
        if on_verification is not None:
            entry['verification_evidence'] = on_verification(
                index, current, results, log)
        if passed:
            entry["action"] = "done"
            break
        if not route.counts_as_iteration:
            entry["action"] = "stop"
            entry["why"] = "处方不可据以行动（hold：这一轮没有可判的对象），不调 Fixer"
            break
        if conflict['present']:
            entry['action'] = 'hold_for_measurement'
            entry['why'] = conflict['why']
            break
        if index == int(max_rounds):
            entry["action"] = "budget_exhausted"
            break
        failed = [r for r in results if not r.passed]
        options = dict(fix_not_rewrite=bool(route.fix_not_rewrite),
                       error_shape=route.error_shape or '')
        if use_diagnosis:
            options['diagnosis'] = entry['diagnosis']
            options['diagnosis_results'] = results
        if isinstance(fixer, FixerAgent):
            options['test_inputs'] = test_inputs
        cursor = records_cursor(fixer)
        outcome = None
        try:
            outcome = _quiet(fixer.run_with_action,
                             current, task, failed,
                             **options)[0]
        except Exception as exc:                      # noqa: BLE001 — 修复失败要可归因
            entry["action"] = "fixer_failed"
            entry["why"] = f"{type(exc).__name__}: {exc}"
            break
        finally:
            generated = records_since(fixer, cursor, outcome=outcome)
            if generated:
                entry['candidate_records'] = generated
                candidate_records.extend(generated)
        if not outcome.code:
            entry["action"] = "ask"
            entry["why"] = outcome.summary or "模型只提问、未给代码"
            break
        change = _change(current, outcome.code)
        entry['source_change'] = change
        entry['fix_summary'] = outcome.summary
        if change['change'] in ('identical_source', 'same_ast'):
            entry['action'] = ('repair_unchanged_source' if change['change'] == 'identical_source'
                               else 'repair_no_structural_change')
            entry['why'] = '修复源码没有实际结构变化，不重复验证已失败候选'
            break
        entry["action"] = "fixed"
        current = outcome.code

    return {
        "passed": bool(passed),
        "rounds": rounds,
        "code": current,
        "results": results,
        "log": "\n".join(log_chunks),
        "initial_full_l2_passed": rounds[0]['full_l2_passed'] if rounds else None,
        "final_full_l2_passed": (None if stopped_reason else
                                 rounds[-1]['full_l2_passed'] if rounds else None),
        "candidate_records": candidate_records,
        "selected_source_hash": code_hash(current),
        **({'stopped_reason': stopped_reason} if stopped_reason else {}),
    }
