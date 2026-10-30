"""Reconcile retained DSH tool events with supervisor-owned feedback projections.

This is a cooperative transcript check, not same-UID tamper resistance or proof
that a model causally used feedback. A final verdict never appears as a tool.
"""
from __future__ import annotations
import hashlib
import json

from benchmarks.experiments.s5.s5_development_service import encoded


def _feedback(text):
    if not isinstance(text, str):
        return None
    for i, char in enumerate(text):
        if char != '{':
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get('schema') == 's5-development-feedback-v1':
            return obj
    return None


def audit(events, snapshot, *, arm):
    calls = {}
    matches = {}
    problems = []
    rows = {r['sequence']: r for r in snapshot['records'] if r['phase'] == 'development'}
    for index, event in enumerate(events):
        if event.get('type') == 'tool_call':
            identity = event.get('callId')
            if identity in calls:
                problems.append('duplicate tool call id')
            calls[identity] = (index, event)
        if event.get('type') != 'tool_result':
            continue
        body = _feedback(event.get('result'))
        if body is None:
            continue
        seq = body.get('sequence')
        row = rows.get(seq)
        call = calls.get(event.get('callId'))
        if row is None or call is None or seq in matches or event.get('status') != 'completed':
            problems.append('unbound or duplicate development feedback')
            continue
        if any(body.get(k) != v for k, v in row['result'].items()):
            problems.append('service feedback differs from DSH tool result')
            continue
        tool = call[1].get('tool')
        if tool == 'mcp__operator__operator_verify' and arm == 'C':
            ingress = 'operator_mcp'
        elif tool == 'bash':
            command = (call[1].get('input') or {}).get('command', '')
            expected = 's5-verify' if arm == 'A' else 'operator-agent verify'
            if expected not in command:
                problems.append('unexpected verification command')
                continue
            ingress = 'public_cli' if arm == 'A' else 'core_cli'
        else:
            problems.append('unexpected development tool')
            continue
        matches[seq] = dict(sequence=seq, tool_call_id=event.get('callId'), ingress=ingress,
            source_sha256=row['source_sha256'], service_response_sha256=row['response_sha256'],
            tool_result_sha256=hashlib.sha256(event['result'].encode()).hexdigest(),
            call_event_index=call[0], result_event_index=index)
    missing = sorted(set(rows) - set(matches))
    if missing:
        problems.append('development response missing from DSH transcript')
    ordered = [matches[k] for k in sorted(matches)]
    if any(a['result_event_index'] >= b['call_event_index'] for a, b in zip(ordered, ordered[1:])):
        problems.append('feedback/revision order is not serial')
    return dict(schema='s5-development-transcript-audit-v1', complete=not problems,
                problems=problems, receipts=ordered, missing_sequences=missing,
                plugin_verification_observed=any(r['ingress'] == 'operator_mcp' for r in ordered),
                feedback_causality_proven=False, isolation_verified=False)
