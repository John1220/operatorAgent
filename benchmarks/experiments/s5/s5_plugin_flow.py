"""Common A/C workflow guidance and offline, observation-only progress facts.

No runtime reminder or model call is issued here. DSH model steps are retained
session boundaries, not proof of the underlying provider dispatch count.
"""
from __future__ import annotations

from pathlib import Path
import shlex
import sys

from benchmarks.experiments.s5.s5_development_evidence import _feedback


def common_instructions(work_dir, backend, max_development=2, max_calls=24, *,
                        python=None, require_plugin_verify=False):
    """Return identical guidance for either arm, with its public file paths."""
    if (type(max_development) is not int or not 1 <= max_development <= 12
            or type(max_calls) is not int or max_calls < 3):
        raise ValueError('invalid workflow budget')
    if backend not in ('mock', 'ssh'):
        raise ValueError('unknown supervised backend')
    if not isinstance(work_dir, (str, Path)) or not str(work_dir).strip():
        raise ValueError('work directory required')
    work = Path(work_dir).resolve()
    interpreter = sys.executable if python is None else python
    if not isinstance(interpreter, str) or not Path(interpreter).is_absolute():
        raise ValueError('an absolute Python interpreter is required')
    command = ' '.join(shlex.quote(value) for value in
                       (interpreter, str(work/'check.py'), '--code-file', 'candidate.py'))
    if type(require_plugin_verify) is not bool:
        raise ValueError('plugin verification requirement must be boolean')
    if require_plugin_verify:
        entry_guidance = (
            'Submit the development candidate only with the DSH MCP tool '
            'mcp__operator__operator_verify. Use the prepared task run directory, backend ssh, '
            'and stop_after L2; do not submit via the public check.py entry. Read the L1/L2 '
            'result returned by the plugin in DSH, then revise only if needed. '
        )
    else:
        entry_guidance = (
            f'The common public test entry is: {command}. '
            'Use normal plugin tools when available; public and plugin submissions share one quota. '
        )
    return (
        f'Task files are already prepared at {work / "task"}. Work in {work}. '
        'The public specification already gives the required kernel signature and inputs. '
        'Write candidate.py yourself and obtain verification feedback early. '
        + entry_guidance +
        f'At most {max_development} supervised development L2 submissions are available; '
        'failed and unknown submissions consume quota too. '
        f'The supervisor fixes the verification backend ({backend}) and contract. '
        'Do not provide host, GPU, credentials or alternative verification options. '
        'No core CLI, core import or repository browsing is needed or available. '
        'Do not explore core/runner implementation files, install GPU libraries, locate another '
        'interpreter, run kernels locally, use direct SSH, or invoke repair/run/bench. '
        'After each development response, read its status and feedback and revise the candidate yourself. '
        f'The model-call cap is {max_calls}. Soft workflow targets: spend at most the first 2 calls '
        'confirming the supplied task and entry; aim to submit the first candidate within calls 6-8; '
        'on the last 2 available calls prioritize returning the best existing source. '
        'These are planning targets, not extra calls, additional quota deductions or runtime reminders. '
        'The final independent evaluation has one separate reservation, provides no feedback, '
        'and cannot be requested as a development tool. Return only the final best source code.'
    )


def _step_key(event):
    step = event.get('step')
    turn = event.get('turn', 1)
    if type(step) is not int or step < 1 or type(turn) is not int or turn < 1:
        return None
    return turn, step


def _candidate_write(event):
    value = event.get('input')
    return (event.get('tool') == 'write' and isinstance(value, dict)
            and isinstance(value.get('file_path'), str)
            and Path(value['file_path']).name == 'candidate.py'
            and isinstance(value.get('content'), str) and bool(value['content'].strip()))


def _exploration(event, verified):
    """Conservative operation categories; not a judgement of necessity."""
    if verified:
        return False
    name, value = event.get('tool'), event.get('input')
    value = value if isinstance(value, dict) else {}
    if name in ('glob', 'grep', 'list_mcp_resources', 'list_mcp_resource_templates'):
        return True
    if isinstance(name, str) and name.endswith('operator_capabilities'):
        return True
    if name == 'read':
        path = value.get('file_path')
        return Path(path).name != 'candidate.py' if isinstance(path, str) else None
    if name in ('write', 'edit') or (isinstance(name, str) and name.endswith(('operator_verify', 'operator_prepare'))):
        return False
    if name != 'bash' or not isinstance(value.get('command'), str):
        return None
    command = value['command']
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    # Do not guess at shell writes or code embedded in heredocs.
    if any(token in ('>', '>>') or token.startswith('<<') for token in tokens):
        return None
    if '--help' in tokens:
        return True
    if tokens and tokens[0] in ('ls', 'pwd', 'which', 'find', 'head', 'cat', 'rg', 'grep', 'sed', 'stat'):
        return True
    return None


def _goal(step, threshold):
    return dict(state='unknown' if step is None else 'met' if step <= threshold else 'not_met',
                first_observed_step=step, threshold=threshold,
                late_observed=step is not None and step > threshold)


def progress(events, snapshot):
    """Read-only facts from actual DSH JSONL events and service feedback records.

Only a completed direct write to candidate.py establishes a write step. Shell
writes remain unknown. A development tool result must match the supervisor's
retained feedback before it establishes a verification step.
"""
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError('DSH events must be a list of objects')
    if not isinstance(snapshot, dict):
        raise ValueError('service snapshot required')
    steps = {}
    calls = {}
    ordered = []
    pending = []
    current = None
    results = []
    problems = []
    for index, event in enumerate(events):
        phase = event.get('phase')
        if phase in ('step_start', 'step_end'):
            key = _step_key(event)
            if key is None:
                problems.append('invalid model step marker')
                current = None
                continue
            ordinal = steps.setdefault(key, len(steps) + 1)
            if phase == 'step_start':
                if pending:
                    problems.append('tool calls before a new step start have unknown step')
                    pending.clear()
                current = ordinal
            else:
                for call in pending:
                    call['step'] = ordinal
                pending.clear()
                current = None
            continue
        if event.get('type') == 'tool_call':
            identity = event.get('callId')
            item = dict(event=event, event_index=index, step=current, completed=False, verified=False)
            ordered.append(item)
            if not isinstance(identity, str) or not identity or identity in calls:
                problems.append('missing or duplicate tool call ID')
            else:
                calls[identity] = item
            if current is None:
                pending.append(item)
        elif event.get('type') == 'tool_result':
            results.append((index, event))

    records = snapshot.get('records')
    if records is not None and not isinstance(records, list):
        raise ValueError('invalid service records')
    service_rows = {}
    for record in records or []:
        if not isinstance(record, dict):
            raise ValueError('invalid service record')
        if record.get('phase') != 'development':
            continue
        sequence = record.get('sequence')
        if type(sequence) is not int or sequence < 1 or sequence in service_rows:
            problems.append('invalid or duplicate development sequence')
            continue
        service_rows[sequence] = record
    matched_sequences = set()
    for index, event in results:
        call = calls.get(event.get('callId'))
        if call is None or event.get('status') != 'completed':
            continue
        if index <= call['event_index']:
            problems.append('tool result precedes its call')
            continue
        call['completed'] = True
        feedback = _feedback(event.get('result'))
        if not isinstance(feedback, dict) or feedback.get('phase') != 'development':
            continue
        sequence = feedback.get('sequence')
        record = service_rows.get(sequence) if type(sequence) is int else None
        expected = record.get('result') if record else None
        if (not isinstance(expected, dict) or not expected or sequence in matched_sequences
                or any(feedback.get(key) != value for key, value in expected.items())):
            problems.append('unbound or duplicate development feedback')
            continue
        matched_sequences.add(sequence)
        call['verified'] = True

    writes = [call['step'] for call in ordered if call['completed'] and _candidate_write(call['event'])]
    verified_steps = [call['step'] for call in ordered if call['verified']]
    first_write = writes[0] if writes else None
    first_verification = verified_steps[0] if verified_steps else None
    used = snapshot.get('development_used')
    if type(used) is not int or used < 0:
        used = len(service_rows) if records is not None else None
    elif records is not None and used != len(service_rows):
        problems.append('development count differs from service records')
        used = None
    classifications = [_exploration(call['event'], call['verified']) for call in ordered]
    return dict(schema='s5-plugin-progress-v1', model_steps_observed=len(steps),
        model_step_basis='ordinal of retained DSH step_start/step_end markers; not provider API calls',
        tool_calls=len(ordered), exploration_tool_calls=sum(value is True for value in classifications),
        unclassified_tool_calls=sum(value is None for value in classifications),
        exploration_scope='read/search/capability and obvious shell inspection operations; necessity is not inferred',
        first_candidate_write_step=first_write, first_verification_step=first_verification,
        development_verifications=used, development_feedback_observed=len(matched_sequences),
        tool_steps_complete=all(call['step'] is not None for call in ordered),
        final_source_present=None, final_source_detection='owned by outer final-candidate extraction',
        soft_targets=dict(first_candidate_write_by_step_8=_goal(first_write, 8),
                          first_verification_by_step_8=_goal(first_verification, 8),
                          entry_confirmation_by_step_2=dict(state='unknown', reason='task comprehension is not inferred'),
                          final_source_with_last_2_calls=dict(state='unknown', reason='provider dispatch budget and final source are not inferred')),
        runtime_reminders_injected=0, extra_model_calls=0, problems=problems)
