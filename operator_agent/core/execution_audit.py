"""Durable level-completion facts, independent of later L3/worker termination."""
from __future__ import annotations

from pathlib import Path
import uuid

from ..core.contract import freeze_from_task
from ..core.run_contract import code_hash
from ..execute.task_cancellation import publish_json
from ..decide.tool_evidence import verification_data


def model_watermark(root):
    """Read a consistent production ledger, never fabricate zero missing usage."""
    from ..execute.task_cancellation import read_json
    journal = Path(root)/'model-journal'
    summary = read_json(journal/'summary.json')
    if not summary:
        return None
    records = [read_json(p) for p in sorted(journal.glob('call-*.json'))]
    if (summary.get('policy') != 'production_task' or summary.get('aborted')
            or summary.get('limits') != read_json(journal/'limits.json')
            or summary.get('identity') != read_json(journal/'identity.json')
            or summary.get('attempts') != records
            or type(summary.get('reserved_calls')) is not int
            or summary['reserved_calls'] != len(records)
            or any(not isinstance(row, dict) for row in records)):
        return None
    return {key: summary.get(key) for key in
            ('reserved_calls', 'dispatched_calls', 'completed_calls', 'api_usage')}


def level_recorder(root, *, layout=None):
    """A trusted harness sink; never changes judgement, retries or confirmation."""
    root = Path(root).resolve()

    def record(code, task, result, completed_results, timing):
        from ..generate.repair_loop import _full_l2_passed
        from ..core.run_contract import audit_value
        from ..core.run_layout import ROLE_ORACLE
        payload = dict(schema='operator-verification-completion-v1',
            event_id=uuid.uuid4().hex, source_hash=code_hash(code),
            contract_hash=freeze_from_task(task).hash,
            level=result.level.value, passed=bool(result.passed),
            timing=timing, full_l2_passed=_full_l2_passed(completed_results),
            model_watermark=model_watermark(root),
            verdicts=verification_data(completed_results),
            scope='Observed level completion; no confirmation or corpus credit.')
        name = 'verification-event-' + payload['event_id'] + '.json'
        if layout is not None:
            import json
            layout.write_for(ROLE_ORACLE, 'measurements/'+name,
                (json.dumps(audit_value(payload), ensure_ascii=False,
                            allow_nan=False, sort_keys=True)+'\n').encode('utf-8'))
        else:
            directory = root/'phase-events'
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            publish_json(directory/name, audit_value(payload))
    return record


def install_level_recorder(verifier, root, *, layout=None):
    # Alternative verifier implementations may not implement completion hooks;
    # their missing phase evidence must remain unknown to the S4 consumer.
    if hasattr(verifier, 'on_level_complete'):
        verifier.on_level_complete = level_recorder(root, layout=layout)
