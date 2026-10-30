"""Formal study dispatch gate for the currently implemented launchers.

A JSON declaration cannot supply an execution boundary. The current launcher
only implements a participant file boundary; network-only egress, candidate
isolation and bounded descendant cleanup still require implementation and
probes. Consequently no current formal study can dispatch, even if a report
asserts success. Offline schedule and summary validation remain usable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from operator_agent.run_contract import digest

REPORT_SCHEMA = 's5-formal-readiness-v1'
REQUIRED_CAPABILITIES = frozenset({
    'participant_file_boundary', 'model_gateway_only_egress',
    'candidate_execution_boundary', 'owned_descendant_cleanup',
})
# This describes actual supported execution contracts, not claims made in a
# plan/report. Updating it requires implementing/probing the corresponding
# launch path and a new source-bound plan; report edits alone cannot enable it.
IMPLEMENTED_CAPABILITIES = frozenset({'participant_file_boundary'})
MAX_REPORT_BYTES = 1024 * 1024
EXPLORATORY_POLICY = 'cooperative-exploratory-v1'
EXPLORATORY_LIMITATIONS = dict(network_only_gateway=False,
    hostile_candidate_isolation=False, budget_unbypassable=False,
    formal_s5_credit=False)


def require_study_readiness(plan):
    """Dispatch under an explicitly frozen exploratory or formal contract."""
    if not isinstance(plan, dict) or plan.get('execution_policy', 'formal') == 'formal':
        return require_formal_readiness(plan)
    if (plan.get('mode') != 'study' or plan.get('schema') != 's5-plugin-study-plan-v1'
            or plan.get('execution_policy') != EXPLORATORY_POLICY
            or plan.get('exploratory_limitations') != EXPLORATORY_LIMITATIONS
            or plan.get('formal_s5_credit') is not False
            or plan.get('hash') != digest({k:v for k,v in plan.items() if k != 'hash'})):
        raise FormalReadinessRefused('exploratory policy or plan identity mismatch')
    from benchmarks.experiments.s5.s5_plugin_workflow import implementation_identity
    from benchmarks.experiments.s5.s5_plugin_probe import load_readiness
    if plan.get('implementation') != implementation_identity():
        raise FormalReadinessRefused('exploratory implementation drift')
    if load_readiness(plan['readiness']['report_path']) != plan['readiness']:
        raise FormalReadinessRefused('exploratory file/tool readiness drift')
    return dict(execution_policy=EXPLORATORY_POLICY, formal_s5_credit=False,
                limitations=dict(EXPLORATORY_LIMITATIONS))


class FormalReadinessRefused(ValueError):
    """No model, credential read, resource query or GPU dispatch is permitted."""


def require_formal_readiness(plan):
    """Check a frozen report reference and the launcher's actual capabilities.

    The driver calls this at both execute() and direct run_one() boundaries.
    The current implementation deliberately has no successful dispatch path.
    """
    if not isinstance(plan, dict) or plan.get('mode') != 'study':
        raise FormalReadinessRefused('formal readiness requires a study plan')
    reference = plan.get('formal_readiness')
    if reference is None:
        raise FormalReadinessRefused(
            'formal readiness report missing; P2 network and candidate boundaries are not verified')
    if (not isinstance(reference, dict)
            or set(reference) != {'report_path', 'report_sha256'}
            or not isinstance(reference.get('report_path'), str)
            or not isinstance(reference.get('report_sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', reference['report_sha256'])):
        raise FormalReadinessRefused('invalid formal readiness reference')
    path = Path(reference['report_path'])
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_size > MAX_REPORT_BYTES):
        raise FormalReadinessRefused('formal readiness report unavailable or unbounded')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['report_sha256']:
        raise FormalReadinessRefused('formal readiness report hash mismatch')
    try:
        report = json.loads(raw)
    except (UnicodeError, ValueError):
        raise FormalReadinessRefused('invalid formal readiness JSON') from None
    if (not isinstance(report, dict) or report.get('schema') != REPORT_SCHEMA
            or report.get('hash') != digest({k: v for k, v in report.items() if k != 'hash'})
            or report.get('implementation') != plan.get('implementation')):
        raise FormalReadinessRefused('formal readiness schema, digest or implementation mismatch')
    checks = report.get('checks')
    if (not isinstance(checks, dict)
            or any(checks.get(key) is not True for key in REQUIRED_CAPABILITIES)):
        raise FormalReadinessRefused('formal readiness boundaries not all verified')
    missing = sorted(REQUIRED_CAPABILITIES - IMPLEMENTED_CAPABILITIES)
    if missing:
        raise FormalReadinessRefused(
            'current launcher does not implement formal boundaries: ' + ', '.join(missing))
    # No executable path currently reaches here. A future launcher must add
    # identity-bound actual probe validation, rather than accepting booleans.
    raise FormalReadinessRefused('formal boundary probe validator is not implemented')
