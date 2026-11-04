"""Source-bound observations never turn missing capture into zero human work."""
from copy import deepcopy

import pytest

from benchmarks.experiments.phase1.events import collect_events


COVERAGE = dict(conversation='complete', tools='complete', external_edits='complete')


def event(identity='e1', **fields):
    return dict(id=identity, timestamp=100., stage='modify', actor='agent', kind='tool_call',
        source=dict(path='raw-events.jsonl', sha256='a'*64, record=0), **fields)


def changed(identity='e1', **fields):
    value = event(identity)
    value.update(fields)
    return value


def test_complete_empty_capture_can_report_observed_zero_without_human_minutes():
    result = collect_events([], COVERAGE, arm='A')
    assert result['complete'] is True
    assert result['observed_count'] == result['count'] == 0
    assert result['human_minutes'] is None
    assert result['observation_range'] == dict(first_timestamp=None, last_timestamp=None)


@pytest.mark.parametrize('source', list(COVERAGE))
@pytest.mark.parametrize('state', ['partial', 'missing'])
def test_missing_or_partial_source_does_not_become_no_intervention(source, state):
    coverage = dict(COVERAGE, **{source: state})
    result = collect_events([], coverage, arm='A')
    assert result['complete'] is False
    assert result['observed_count'] == 0
    assert result['count'] is None


def test_human_events_require_sources_and_initial_task_is_separate():
    initial = changed('initial', actor='human', kind='user_message', stage='setup')
    edit = changed('edit', actor='human', kind='external_edit', reason='fix input shape')
    takeover = changed('takeover', actor='human', kind='takeover', timestamp=102.)
    result = collect_events([initial, edit, takeover], COVERAGE, arm='A')
    assert result['count'] == result['observed_count'] == 2
    assert result['setup_events'] == [initial]
    assert result['human_events'] == [edit, takeover]
    assert result['human_events'][0]['reason'] == 'fix input shape'
    assert 'reason' not in result['human_events'][1]  # do not invent an explanation
    assert result['observation_range'] == dict(first_timestamp=100., last_timestamp=102.)


@pytest.mark.parametrize('source', [None, 'raw.jsonl', {},
    dict(path='raw', sha256='bad', record=0),
    dict(path='raw', sha256='a'*64, record=True),
    dict(path='raw', sha256='a'*64, record=-1)])
def test_unbound_human_event_stays_raw_and_makes_count_unknown(source):
    raw = changed(actor='human', kind='user_message', source=source)
    result = collect_events([raw], COVERAGE, arm='A')
    assert result['events'] == [raw]
    assert result['human_events'] == []
    assert result['observed_count'] == 0 and result['count'] is None
    assert [p['code'] for p in result['problems']] == ['unbound_source']


def test_agent_shell_is_agent_and_only_plugin_arm_bypasses_delegation():
    shell = changed(tool='bash', action='candidate_edit')
    assert collect_events([shell], COVERAGE, arm='A')['bypass_count'] == 0
    result = collect_events([shell], COVERAGE, arm='B')
    assert result['count'] == 0
    assert result['bypass_events'] == [shell]
    assert result['bypass_events'][0]['actor'] == 'agent'
    assert result['bypass_count'] == 1


@pytest.mark.parametrize('tool', ['operator_run', 'operator_repair',
    'mcp__operator__operator_run', 'mcp__operator__operator_repair'])
def test_delegated_candidate_actions_do_not_count_as_bypass(tool):
    delegated = changed(tool=tool, action='candidate_edit')
    result = collect_events([delegated], COVERAGE, arm='B')
    assert result['bypass_count'] == 0
    assert result['count'] == 0


def test_missing_shell_semantics_does_not_prove_no_bypass():
    result = collect_events([changed(tool='bash')], COVERAGE, arm='B')
    assert result['observed_bypass_count'] == 0
    assert result['bypass_count'] is None
    assert result['bypass_complete'] is False
    assert result['count'] == 0  # actor attribution is still complete


def test_failure_retry_and_refusal_records_are_not_filtered_to_successes():
    records = [changed('failure', kind='tool_result', status='failed', tool='bash'),
               changed('retry', kind='retry'),
               changed('refusal', kind='tool_result', ok=False, tool='operator_verify'),
               changed('success', kind='tool_result', ok=True, tool='operator_verify')]
    result = collect_events(records, COVERAGE, arm='A')
    assert result['events'] == records
    assert result['failure_count'] == result['observed_failure_count'] == 2
    assert result['retry_count'] == result['observed_retry_count'] == 1
    assert result['failed_events'] == [records[0], records[2]]
    assert result['retry_events'] == [records[1]]


def test_partial_tool_capture_retains_observed_failure_and_unknown_total():
    raw = changed(kind='tool_result', status='failed')
    result = collect_events([raw], dict(COVERAGE, tools='partial'), arm='A')
    assert result['failed_events'] == [raw]
    assert result['observed_failure_count'] == 1
    assert result['failure_count'] is None and result['retry_count'] is None


def test_duplicate_ids_do_not_double_count_human_events():
    raw = changed(actor='human', kind='user_message')
    result = collect_events([raw, raw], COVERAGE, arm='A')
    assert len(result['events']) == 2
    assert result['observed_count'] == 1 and result['count'] is None
    assert result['problems'][0]['code'] == 'duplicate_event_id'


@pytest.mark.parametrize('fields,code', [
    (dict(actor='unknown'), 'unknown_actor'),
    (dict(actor='robot'), 'unknown_actor'),
    (dict(kind='new_event_kind'), 'unknown_event_kind'),
    (dict(timestamp=float('nan')), 'invalid_timestamp'),
    (dict(timestamp=True), 'invalid_timestamp'),
    (dict(timestamp=-1), 'invalid_timestamp'),
    (dict(timestamp=10**1000), 'invalid_timestamp'),
    (dict(stage=''), 'invalid_stage'),
    (dict(id=None), 'invalid_event_id'),
    (dict(action=[]), 'invalid_action'),
    (dict(tool={}), 'invalid_tool'),
    (dict(reason=[]), 'invalid_reason'),
    (dict(status=False), 'invalid_status'),
    (dict(ok=0), 'invalid_result_ok'),
])
def test_invalid_rows_are_visible_and_never_make_a_complete_zero(fields, code):
    raw = changed(**fields)
    result = collect_events([raw], COVERAGE, arm='A')
    assert result['events'] == [raw]
    assert result['count'] is None
    assert result['problems'][0]['code'] == code


def test_invalid_container_and_coverage_cannot_become_empty_complete_capture():
    result = collect_events(None, None, arm='B')
    assert result['complete'] is False and result['count'] is None
    assert result['coverage'] == dict.fromkeys(COVERAGE, 'missing')
    assert {p['code'] for p in result['problems']} == {'events_not_list', 'coverage_not_mapping'}


def test_aggregation_never_mutates_caller_records_or_coverage():
    raw = [changed(actor='human', kind='external_edit')]
    original = deepcopy(raw)
    coverage = dict(COVERAGE)
    result = collect_events(raw, coverage, arm='A')
    result['human_events'][0]['source']['path'] = 'changed'
    assert raw == original and coverage == COVERAGE
