"""把「账本之前」的目录搬进账本：**只搬已经发生的事实**。

四件事：能归属才搬、不覆盖已有账本、旧文件原地不动、旧相位里没有的东西不补
（`result` 这一格在旧文件里根本不存在）。
"""
import json
from pathlib import Path

import pytest

from operator_agent.core import task_journal as journal
from operator_agent.execute.legacy_migration import MigrationRefused, migrate

CONTRACT = 'c' * 64
CALL = 'a' * 32


def legacy(tmp_path, *, task_status=None, call_status=None, run_json=True):
    run = tmp_path/'old'
    run.mkdir()
    (run/'declaration.json').write_text(json.dumps({'operator': 'relu'}))
    (run/'contract.json').write_text(json.dumps({'hash': CONTRACT, 'contract': {}}))
    if run_json:
        (run/'run.json').write_text(json.dumps({'task_id': 't-1'}))
    if task_status is not None:
        (run/'run-status.json').write_text(json.dumps(task_status))
    if call_status is not None:
        call = run/'tool-calls'/CALL
        call.mkdir(parents=True)
        (call/'run-status.json').write_text(json.dumps(call_status))
    return run


def test_a_finished_run_and_its_calls_become_one_stream(tmp_path):
    run = legacy(
        tmp_path,
        task_status=dict(phase='done', invocation_id='b'*32, stopped_reason='passed',
                         pid=123, started_at=1, finished_at=2, duration_s=1,
                         cleanup=dict(worker_reaped=True, provider_call='unknown')),
        call_status=dict(phase='done', invocation_id=CALL, tool='operator_verify',
                         stopped_reason='tool_returned', cleanup=dict(worker_reaped=True)))
    receipt = migrate(run)
    assert receipt['events'] == ['task_prepared', 'task_activated', 'job_queued',
                                 'job_started', 'job_finished', 'task_closed']
    # 旧文件原地不动：它们是这次运行当年的记录，迁移不改历史。
    assert (run/'run-status.json').is_file()
    task = journal.task_view(run)
    assert (task['phase'], task['reason']) == ('done', 'passed')
    call = journal.invocation_view(run/'tool-calls'/CALL)
    assert (call['phase'], call['outcome'], call['result']) == ('done', 'completed', 'received')
    assert call['cleanup']['worker_reaped'] is True


def test_a_dead_run_is_migrated_as_activation_only_and_reads_back_as_lost(tmp_path):
    """旧文件里的 `lost` 是「进程没了、终态没写」。搬过来只剩激活，读者重新得出 lost。"""
    run = legacy(tmp_path, task_status=dict(phase='lost', invocation_id='b'*32, pid=999999))
    migrate(run)
    assert [event['kind'] for event in journal.read_events(run)] == [
        'task_prepared', 'task_activated']
    view = journal.task_view(run)
    assert view['phase'] == 'running' and view['journal_phase'] == 'active'


def test_the_missing_result_field_is_not_invented(tmp_path):
    """旧相位里没有 `result`：能确定写 `received`，不能确定写 `unknown` 并说明理由。"""
    run = legacy(tmp_path, call_status=dict(
        phase='failed', invocation_id=CALL, tool='operator_repair',
        stopped_reason='worker_failed'))
    migrate(run)
    call = journal.invocation_view(run/'tool-calls'/CALL)
    assert (call['outcome'], call['result'], call['result_reason']) == (
        'failed', 'unknown', 'receipt_absent')
    assert call['operation'] == 'repair'


def test_dry_run_writes_nothing(tmp_path):
    run = legacy(tmp_path, task_status=dict(phase='done', invocation_id='b'*32))
    receipt = migrate(run, dry_run=True)
    assert receipt['dry_run'] is True and receipt['event_count'] >= 1
    assert journal.read_events(run) == []


def test_unattributable_directories_are_refused_not_guessed(tmp_path):
    bare = tmp_path/'bare'
    bare.mkdir()
    (bare/'run-status.json').write_text(json.dumps({'phase': 'done'}))
    with pytest.raises(MigrationRefused) as caught:
        migrate(bare)
    assert caught.value.code == 'not_migratable'          # 没有契约就没有任务身份

    no_facts = legacy(tmp_path)
    with pytest.raises(MigrationRefused) as caught:
        migrate(no_facts)
    assert caught.value.code == 'not_migratable'          # 一个旧相位文件都没有

    (tmp_path/'x').mkdir()
    (tmp_path/'y').mkdir()
    no_tool = legacy(tmp_path/'x', call_status=dict(phase='done', invocation_id=CALL))
    with pytest.raises(MigrationRefused) as caught:
        migrate(no_tool)
    assert caught.value.code == 'unattributable_status'   # 归不到任何 Job 类型

    odd = legacy(tmp_path/'y', task_status=dict(phase='sideways', invocation_id='b'*32))
    with pytest.raises(MigrationRefused) as caught:
        migrate(odd)
    assert caught.value.code == 'unknown_phase'


def test_an_existing_journal_is_never_rewritten(tmp_path):
    run = legacy(tmp_path, task_status=dict(phase='done', invocation_id='b'*32))
    migrate(run)
    before = journal.read_events(run)
    with pytest.raises(MigrationRefused) as caught:
        migrate(run)
    assert caught.value.code == 'already_migrated'
    assert journal.read_events(run) == before
