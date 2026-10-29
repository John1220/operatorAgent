"""Common instructions and offline progress use retained DSH boundaries only."""
import copy
import json
from pathlib import Path
import shlex

import pytest

from benchmarks import s5_plugin_flow as flow


def marker(phase, step, turn=1):
    return dict(type='status', phase=phase, step=step, turn=turn)


def call(identity, tool, arguments):
    return dict(type='tool_call', callId=identity, tool=tool, input=arguments)


def result(identity, text='OK', status='completed'):
    return dict(type='tool_result', callId=identity, result=text, status=status)


def fixture():
    body = dict(schema='s5-development-feedback-v1', sequence=1, phase='development',
                status='failed', source_sha256='s'*64, contract_hash='c'*64,
                feedback=[dict(message='reviewed failure')])
    snapshot = dict(development_used=1, records=[dict(sequence=1, phase='development', result=body)])
    events = [marker('step_start', 1),
              call('read-1', 'read', dict(file_path='/work/task/contract.json')),
              result('read-1', 'contract'), marker('step_end', 1),
              marker('step_start', 2),
              call('write-1', 'write', dict(file_path='/work/candidate.py', content='def kernel(x): return x')),
              result('write-1'), marker('step_end', 2),
              marker('step_start', 3),
              call('verify-1', 'bash', dict(command='/python /work/check.py --code-file candidate.py')),
              result('verify-1', json.dumps(body)), marker('step_end', 3),
              marker('step_start', 4), marker('step_end', 4),
              dict(type='final', text='This is not necessarily candidate code.')]
    return events, snapshot


def test_common_instructions_have_one_shared_flow_and_quoted_public_entry(tmp_path):
    work = tmp_path/'task with spaces'
    text = flow.common_instructions(work, 'ssh', python='/runtime/python with spaces')
    assert text == flow.common_instructions(work, 'ssh', python='/runtime/python with spaces')
    expected = ' '.join(shlex.quote(s) for s in
                        ('/runtime/python with spaces', str(work/'check.py'), '--code-file', 'candidate.py'))
    assert expected in text and str(work/'task') in text
    assert '24' in text and 'first 2 calls' in text and 'calls 6-8' in text and 'last 2' in text
    assert 'planning targets' in text and 'runtime reminders' in text
    assert 'failed and unknown submissions consume quota' in text
    assert 'normal plugin tools when available' in text
    assert 'provides no feedback' in text and 'repair/run/bench' in text
    assert 'operator-agent verify' not in text and 'shapes=' not in text


def test_plugin_entry_acceptance_requires_mcp_verify_without_public_test_path(tmp_path):
    text = flow.common_instructions(tmp_path, 'ssh', require_plugin_verify=True)
    assert 'mcp__operator__operator_verify' in text
    assert 'backend ssh' in text and 'stop_after L2' in text
    assert 'do not submit via the public check.py entry' in text
    assert 'common public test entry' not in text
    assert 'normal plugin tools when available' not in text


@pytest.mark.parametrize('kwargs', [dict(max_calls=True), dict(max_calls=2),
    dict(max_development=0), dict(max_development=True), dict(python='python3'),
    dict(backend='h20'), dict(require_plugin_verify=1)])
def test_bad_guidance_parameters_refuse(tmp_path, kwargs):
    args = dict(work_dir=tmp_path, backend='mock')
    args.update(kwargs)
    with pytest.raises(ValueError):
        flow.common_instructions(**args)


def test_steps_candidate_write_and_bound_feedback_are_observed_without_extra_calls():
    events, snapshot = fixture()
    before = copy.deepcopy((events, snapshot))
    p = flow.progress(events, snapshot)
    assert p['model_steps_observed'] == 4 and p['tool_calls'] == 3
    assert p['first_candidate_write_step'] == 2
    assert p['first_verification_step'] == 3
    assert p['development_verifications'] == p['development_feedback_observed'] == 1
    assert p['exploration_tool_calls'] == 1
    assert p['soft_targets']['first_verification_by_step_8']['state'] == 'met'
    assert p['soft_targets']['entry_confirmation_by_step_2']['state'] == 'unknown'
    assert p['final_source_present'] is None
    assert p['runtime_reminders_injected'] == p['extra_model_calls'] == 0
    assert p['tool_steps_complete'] and not p['problems']
    assert (events, snapshot) == before


def test_step_end_only_projection_assigns_pending_tools_to_completed_step():
    events, snapshot = fixture()
    events = [event for event in events if event.get('phase') != 'step_start']
    p = flow.progress(events, snapshot)
    assert p['model_steps_observed'] == 4
    assert p['first_candidate_write_step'] == 2 and p['first_verification_step'] == 3
    assert p['tool_steps_complete']


def test_missing_markers_and_bash_write_do_not_fabricate_candidate_step():
    events = [call('shell', 'bash', dict(command="cat > candidate.py <<'PY'\ndef kernel(x): return x\nPY")),
              result('shell')]
    p = flow.progress(events, dict(development_used=0, records=[]))
    assert p['first_candidate_write_step'] is None
    assert p['first_verification_step'] is None
    assert p['model_steps_observed'] == 0 and not p['tool_steps_complete']
    assert p['soft_targets']['first_candidate_write_by_step_8']['state'] == 'unknown'
    assert p['unclassified_tool_calls'] == 1


def test_failed_write_and_non_candidate_files_are_not_first_candidate():
    events = [marker('step_start', 1),
              call('bad', 'write', dict(file_path='/work/candidate.py', content='source')),
              result('bad', status='error'),
              call('other', 'write', dict(file_path='/work/notes.md', content='notes')),
              result('other'), marker('step_end', 1)]
    p = flow.progress(events, dict(development_used=0, records=[]))
    assert p['first_candidate_write_step'] is None


@pytest.mark.parametrize('change', ['unknown_sequence', 'modified_feedback', 'error_result', 'missing_call'])
def test_unbound_or_failed_feedback_cannot_establish_verification_step(change):
    events, snapshot = fixture()
    event = next(e for e in events if e.get('callId') == 'verify-1' and e['type'] == 'tool_result')
    if change in ('unknown_sequence', 'modified_feedback'):
        body = json.loads(event['result'])
        body['sequence' if change == 'unknown_sequence' else 'status'] = 999 if change == 'unknown_sequence' else 'passed'
        event['result'] = json.dumps(body)
    elif change == 'error_result':
        event['status'] = 'error'
    else:
        events = [e for e in events if not (e.get('type') == 'tool_call' and e.get('callId') == 'verify-1')]
    p = flow.progress(events, snapshot)
    assert p['first_verification_step'] is None
    assert p['development_feedback_observed'] == 0
    assert p['development_verifications'] == 1


def test_mcp_feedback_uses_same_supervisor_binding_and_counts_unknown_attempt():
    events, snapshot = fixture()
    call_event = next(e for e in events if e['type'] == 'tool_call' and e.get('callId') == 'verify-1')
    call_event.update(tool='mcp__operator__operator_verify', input=dict(code='source'))
    record = snapshot['records'][0]['result']
    record['status'] = 'unknown'
    next(e for e in events if e['type'] == 'tool_result' and e.get('callId') == 'verify-1')['result'] = json.dumps(record)
    p = flow.progress(events, snapshot)
    assert p['first_verification_step'] == 3 and p['development_verifications'] == 1
    assert p['exploration_tool_calls'] == 1


def test_late_observed_submission_is_not_met_and_no_runtime_intervention_occurs():
    events, snapshot = fixture()
    events = [marker('step_start', step) for step in range(1, 9)] + [
        marker('step_start', 9), call('verify', 'bash', dict(command='public verification')),
        result('verify', json.dumps(snapshot['records'][0]['result'])), marker('step_end', 9)]
    p = flow.progress(events, snapshot)
    target = p['soft_targets']['first_verification_by_step_8']
    assert target['state'] == 'not_met' and target['late_observed']
    assert p['runtime_reminders_injected'] == p['extra_model_calls'] == 0


def test_multiple_turns_and_multi_tool_step_are_not_counted_as_extra_model_calls():
    events = [marker('step_start', 1, 1), call('cap', 'mcp__operator__operator_capabilities', {}),
              result('cap'), call('help', 'bash', dict(command='python check.py --help')),
              result('help'), marker('step_end', 1, 1), marker('step_start', 1, 2),
              marker('step_end', 1, 2)]
    p = flow.progress(events, dict(development_used=0, records=[]))
    assert p['model_steps_observed'] == 2 and p['tool_calls'] == 2
    assert p['exploration_tool_calls'] == 2
    assert 'not provider API calls' in p['model_step_basis']


def test_final_feedback_is_not_a_development_submission_and_final_source_is_not_guessed():
    events, snapshot = fixture()
    body = snapshot['records'][0]['result']
    body['phase'] = 'final'
    events[-1] = dict(type='final', text='def kernel(x): return x')
    next(e for e in events if e.get('callId') == 'verify-1' and e['type'] == 'tool_result')['result'] = json.dumps(body)
    snapshot.update(development_used=0, records=[dict(sequence=1, phase='final', result=body)])
    p = flow.progress(events, snapshot)
    assert p['first_verification_step'] is None and p['development_verifications'] == 0
    assert p['final_source_present'] is None


def test_result_before_call_or_duplicate_ids_are_reported_as_problems():
    events = [marker('step_start', 1), result('same'),
              call('same', 'write', dict(file_path='candidate.py', content='source')),
              call('same', 'read', dict(file_path='task.json')), marker('step_end', 1)]
    p = flow.progress(events, dict(development_used=0, records=[]))
    assert p['first_candidate_write_step'] is None
    assert 'missing or duplicate tool call ID' in p['problems']
    assert 'tool result precedes its call' in p['problems']


def test_inconsistent_development_count_stays_unknown():
    events, snapshot = fixture()
    snapshot['development_used'] = 2
    p = flow.progress(events, snapshot)
    assert p['development_verifications'] is None
    assert p['problems'] == ['development count differs from service records']
