"""Adapt the production CLI's delegated worker to a parent-owned cell deadline.

This uses the same worker as operator_run, not a DSH host or installed plugin.
The experiment parent takes the normal supervisor's process-ownership role.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

from operator_agent.app import autonomous, hitl_tools as ht
from operator_agent.execute.fake_backend import FakeConfig, execution_context

from .plan import read, write_once


def source_ref(path, record='result'):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), record=record)


def run(config):
    from operator_agent.execute.s5_verify_client import enabled
    if enabled():
        raise ValueError('offline delegation refuses inherited S5 verification service')
    fake = FakeConfig.from_dict(config['fake'])
    if config['backend'] != 'fake' or fake.llm != 'fake' or config['arm'] != 'B':
        raise ValueError('delegate adapter supports offline B cells only')
    work = Path(config['run_dir'])
    work.mkdir(parents=True, exist_ok=True)
    prepared = ht.prepare(config['declaration'], work)
    context = dict(execution_context=execution_context(fake))
    invocation = uuid.uuid4().hex
    cfg = dict(backend='fake', fake=fake.as_dict(), max_rounds=config['max_rounds'],
        task_budget=config['budget'], invocation_id=invocation, started_at=time.time(), **context)
    write_once(work/'autonomous.json', cfg)
    keys = ('OPERATOR_AGENT_S5_SHARED_LEDGER', 'OPERATOR_AGENT_S5_TASK_ID')
    previous = {key: os.environ.get(key) for key in keys}
    os.environ[keys[0]] = config['shared_ledger']
    os.environ[keys[1]] = config['shared_task_id']
    started = time.time()
    try:
        autonomous._worker_main(work, config['deadline_monotonic'],
                                config['started_monotonic'], invocation)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    raw = read(work/'worker-result.json')
    summary = read(work/'conductor.json') if (work/'conductor.json').is_file() else {}
    selected = Path(summary['path'])/'code.py' if summary.get('path') else work/'candidate'/'code.py'
    result = dict(schema='phase1-delegate-result-v1', cell_id=config['cell_id'],
        arm='B', plan_hash=config['plan_hash'], phase=raw['phase'],
        reason=raw.get('stopped_reason') or raw.get('error') or 'worker_returned',
        error=raw.get('detail'), levels=summary.get('levels', {}), all_passed=False,
        selected_source=str(selected) if selected.is_file() else None,
        source_hash=hashlib.sha256(selected.read_bytes()).hexdigest() if selected.is_file() else None,
        contract_hash=prepared['contract_hash'], budget=raw.get('budget'),
        task_budget=config['budget'], **context)
    write_once(work/'delegate-result.json', result)
    events = [dict(id='delegate-call', timestamp=started, stage='check', actor='agent',
        kind='tool_call', action='candidate_verify', tool='operator_run',
        source=source_ref(work/'autonomous.json')),
        dict(id='delegate-result', timestamp=time.time(), stage='check', actor='agent',
        kind='tool_result', action='candidate_verify', tool='operator_run',
        ok=raw['phase'] == 'done', source=source_ref(work/'delegate-result.json'))]
    # Every retained level and repair call stays visible, including failed ones.
    for index, path in enumerate(sorted(work.rglob('verification-event-*.json'))):
        value = read(path)
        events.append(dict(id='level-'+str(index), timestamp=value['timing']['completed_at'],
            stage=value['level'], actor='agent', kind='tool_result',
            action='candidate_verify', tool='operator_run', ok=value['passed'],
            source=source_ref(path)))
    for index, path in enumerate(sorted((work/'model-journal').glob('call-*.json'))):
        value = read(path)
        if 'Fixer' in str(value.get('caller')) or 'Repair' in str(value.get('caller')):
            events.append(dict(id='retry-'+str(index), timestamp=time.time(), stage='repair',
                actor='agent', kind='retry', action='candidate_verify', tool='operator_run',
                source=source_ref(path)))
    write_once(work/'events.json', events)
    return result


if __name__ == '__main__':
    print(json.dumps(run(read(sys.argv[1])), ensure_ascii=False))
