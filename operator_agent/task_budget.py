"""Finite production task policy and read-only audit snapshots.

Limits concern local worker lifetime, model dispatches and verification-case
reservations. They are not a provider invoice or a GPU-time measurement.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from .task_clock import monotonic


@dataclass(frozen=True)
class TaskBudgetConfig:
    wall_clock_s: float = 1800.0
    max_llm_calls: int = 24
    oracle_total: int = 20

    def __post_init__(self):
        if (type(self.wall_clock_s) not in (int, float)
                or not math.isfinite(self.wall_clock_s) or self.wall_clock_s <= 0):
            raise ValueError('wall_clock_s must be finite and positive')
        for name in ('max_llm_calls', 'oracle_total'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + ' must be a positive integer')

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value=None):
        if value is None:
            return cls()
        if type(value) is not dict or set(value) - set(cls.__dataclass_fields__):
            raise ValueError('invalid task_budget fields')
        return cls(**value)

    def llm_limits(self, call_seconds=120.):
        from .llm_budget import LLMCallLimits
        return LLMCallLimits(max_calls=self.max_llm_calls, max_output_tokens=40960,
            total_output_token_reservations=self.max_llm_calls * 40960,
            max_request_bytes=1048576, max_response_bytes=1048576,
            call_seconds=call_seconds, campaign_seconds=self.wall_clock_s,
            allowed_callers=('Designer', 'Coder', 'Fixer', 'Search', 'Planner',
                             'Fixer.Reflexion', 'CoderPipeline.Coder',
                             'CoderPipeline.Repair'))


def _read(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def snapshot(run, config, started_monotonic, *, verification=None, now=None):
    """Report durable reservations, including failures and unfinished calls."""
    run = Path(run)
    elapsed = max(0., (monotonic() if now is None else now) - started_monotonic)
    journal = run / 'model-journal'
    files = sorted(journal.glob('call-*.json'))
    records = [_read(path) for path in files]
    refusal = _read(journal / 'refusal.json') or {}
    summary = _read(journal / 'summary.json') or {}
    limits, identity = _read(journal/'limits.json'), _read(journal/'identity.json')
    known = (summary.get('policy') == 'production_task' and not summary.get('aborted')
        and limits is not None and identity is not None
        and summary.get('limits') == limits and summary.get('identity') == identity
        and summary.get('attempts') == records and all(record is not None for record in records)
        and type(summary.get('reserved_calls')) is int and summary['reserved_calls'] == len(records)
        and [record.get('call') for record in records] == list(range(1, len(records)+1)))
    reserved = len(records) if known else None
    reason = (refusal.get('reason') or summary.get('stop_reason')) if known else None
    if verification is None:
        verification = _read(run / 'verification-budget.json')
    return dict(schema='operator-task-budget-v1', limits=config.as_dict(),
        wall_clock=dict(limit_s=config.wall_clock_s, elapsed_s=round(elapsed, 3),
            remaining_s=round(max(0., config.wall_clock_s-elapsed), 3),
            expired=elapsed >= config.wall_clock_s,
            scope='supervised_local_worker_lifetime'),
        model_calls=dict(limit=config.max_llm_calls, reserved=reserved,
            remaining=max(0, config.max_llm_calls-reserved) if known else None,
            failed=sum(record.get('status') not in ('reserved', 'answered')
                       for record in records) if known else None,
            in_flight=sum(record.get('status') == 'reserved' for record in records) if known else None,
            journal_state='known' if known else 'unknown_or_inconsistent',
            journal_aborted=summary.get('aborted'),
            exhausted_reason=reason, unit='pre_dispatch_reservations_no_refund'),
        verification=verification, provider_billing_cap=False,
        gpu_seconds=None, remote_cancellation_proven=False)


def stopped_reason(budget):
    if budget['wall_clock']['expired']:
        return 'budget_wall_clock'
    return {'llm_call_budget': 'budget_llm_calls',
            'llm_output_budget': 'budget_llm_output',
            'task_wall_clock': 'budget_wall_clock'}.get(
                budget['model_calls'].get('exhausted_reason'))
