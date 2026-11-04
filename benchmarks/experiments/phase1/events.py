"""Aggregate source-bound observations without inventing human effort.

This pure function checks reference structure, not source-file contents. The
capture/receipt layer must bind each path/hash/record to retained raw evidence
and supply coverage for the observation interval. These are cooperative audit
records, not a security boundary against an actor forging its own transcript.
"""
from copy import deepcopy
import math
import re


SCHEMA = 'phase1-observed-events-v1'
_SOURCES = ('conversation', 'tools', 'external_edits')
_COVERAGE = ('complete', 'partial', 'missing')
_ACTORS = ('agent', 'human', 'system', 'unknown')
_KINDS = ('user_message', 'external_edit', 'takeover', 'tool_call', 'tool_result', 'retry')
_SETUP = frozenset(('setup', 'initial_setup', 'initial_task', 'initial_config'))
_CANDIDATE_ACTIONS = frozenset(('candidate_edit', 'candidate_execute', 'candidate_verify'))
_DELEGATED = frozenset(('operator_run', 'operator_repair'))
_SUPPORT_TOOLS = frozenset(('operator_prepare', 'operator_status', 'operator_cancel',
                            'operator_capabilities'))
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _source(value):
    if not isinstance(value, dict) or not _text(value.get('path')):
        return False
    digest, record = value.get('sha256'), value.get('record')
    return (isinstance(digest, str) and bool(_HASH.fullmatch(digest))
            and ((type(record) is int and record >= 0) or _text(record)))


def _tool(value):
    return value.rsplit('__', 1)[-1] if isinstance(value, str) else None


def _timestamp(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def collect_events(events, coverage, *, arm):
    """Summarize normalized A/B observations, keeping missing data explicit.

    ``observed_count`` counts only source-bound human events during execution;
    ``count`` is unknown unless all three capture sources and all events are
    complete. Setup stages are separate and never added to runtime counts.
    Failure/retry/bypass totals likewise remain unknown when tool capture is
    incomplete; their observed counts and records are always retained.

    A source is a dict with nonempty ``path``, lowercase SHA256 ``sha256`` and
    a zero-based integer or nonempty string ``record``. ``actor`` is attributed
    by the capture layer, never inferred from an Agent invoking a shell.
    """
    if arm not in ('A', 'B'):
        raise ValueError('arm must be A or B')
    problems = []

    def issue(code, index=None, event=None):
        problems.append(dict(code=code, index=index,
            event_id=event.get('id') if isinstance(event, dict) else None))

    if not isinstance(events, list):
        issue('events_not_list')
        retained = []
    else:
        retained = deepcopy(events)
    if not isinstance(coverage, dict):
        issue('coverage_not_mapping')
        coverage = {}
    sources = {}
    for name in _SOURCES:
        state = coverage.get(name, 'missing')
        if state not in _COVERAGE:
            issue('invalid_coverage_' + name)
            state = 'missing'
        sources[name] = state
    for name in coverage:
        if name not in _SOURCES:
            issue('unknown_coverage_source')

    human, setup, retries, failed, bypass = [], [], [], [], []
    timestamps, seen = [], set()
    bypass_unknown = False
    for index, event in enumerate(retained):
        if not isinstance(event, dict):
            issue('event_not_mapping', index, event)
            continue
        before = len(problems)
        identity = event.get('id')
        if not _text(identity):
            issue('invalid_event_id', index, event)
        elif identity in seen:
            issue('duplicate_event_id', index, event)
        else:
            seen.add(identity)
        stamp = event.get('timestamp')
        if not _timestamp(stamp):
            issue('invalid_timestamp', index, event)
        if not _text(event.get('stage')):
            issue('invalid_stage', index, event)
        if event.get('actor') not in _ACTORS or event.get('actor') == 'unknown':
            issue('unknown_actor', index, event)
        if event.get('kind') not in _KINDS:
            issue('unknown_event_kind', index, event)
        for field in ('action', 'tool', 'reason', 'status'):
            if event.get(field) is not None and not isinstance(event[field], str):
                issue('invalid_' + field, index, event)
        if 'ok' in event and type(event['ok']) is not bool:
            issue('invalid_result_ok', index, event)
        if not _source(event.get('source')):
            issue('unbound_source', index, event)
        if len(problems) != before:
            continue
        timestamps.append(stamp)
        kind, actor = event['kind'], event['actor']
        if event['stage'] in _SETUP:
            setup.append(event)
        elif actor == 'human':
            human.append(event)
        if kind == 'retry':
            retries.append(event)
        if (kind == 'tool_result' and (event.get('status') in ('failed', 'error', 'refused')
                or event.get('ok') is False or bool(event.get('error')))):
            failed.append(event)
        if arm == 'B' and kind == 'tool_call':
            tool = _tool(event.get('tool'))
            if event.get('action') in _CANDIDATE_ACTIONS:
                if not tool:
                    issue('candidate_tool_unknown', index, event)
                    bypass_unknown = True
                elif tool not in _DELEGATED:
                    bypass.append(event)
            elif not event.get('action') and tool not in _DELEGATED | _SUPPORT_TOOLS:
                # A bare shell name does not tell us whether the Agent edited
                # a candidate or merely inspected its directory.
                bypass_unknown = True

    complete = not problems and all(s == 'complete' for s in sources.values())
    tools_complete = not problems and sources['tools'] == 'complete'
    bypass_complete = tools_complete and not bypass_unknown
    return dict(schema=SCHEMA, arm=arm, events=retained, coverage=sources,
        complete=complete, observed_count=len(human), count=len(human) if complete else None,
        human_events=human, setup_events=setup, human_minutes=None,
        bypass_events=bypass, observed_bypass_count=len(bypass),
        bypass_count=len(bypass) if bypass_complete else None, bypass_complete=bypass_complete,
        retry_events=retries, observed_retry_count=len(retries),
        retry_count=len(retries) if tools_complete else None,
        failed_events=failed, observed_failure_count=len(failed),
        failure_count=len(failed) if tools_complete else None, tools_complete=tools_complete,
        observation_range=dict(first_timestamp=min(timestamps) if timestamps else None,
                               last_timestamp=max(timestamps) if timestamps else None),
        problems=problems)
