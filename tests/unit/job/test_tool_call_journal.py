"""一次被监督的调用怎么记账：**账本里只有 job_* ，且终态说清「回执到账了没有」。

钉四件事：
1. `ToolCall` 写 `task_activated` + `job_*`，**不**写 `run-status.json`（双写就是两个权威）；
2. `prepare` 缺席时它**不补** `task_prepared`（替没契约的目录伪造过去是最贵的谎）；
3. 回执读不到记 `result=unknown`，不记 `failed` —— 执行可能已经跑完；
4. 同一条错误下账本**不** `reject_completed`：没看见回执，就不能否认已完成的工作。
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operator_agent.app import hitl_tools as ht, tool_calls
from operator_agent.core import task_journal as journal
from operator_agent.execute import exec_guard
from operator_agent.execute.host_task_budget import HostTaskBudget, initialize_host_task
from operator_agent.execute.task_cancellation import publish_json

DECL = dict(operator='relu', request='relu fp32', shapes=[[4, 8]], precision='fp32',
            dsl='python')
CODE = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'
SOURCE = 'c' * 64


def _prepare(run):
    journal.append(run, 'task_prepared', task_id='fixture-task')


def _clock(monkeypatch):
    """时间固定：回执的 `finished_monotonic` 必须在 [started, now] 里才算合格。"""
    from operator_agent.execute import task_budget, task_cancellation
    clock = [1000.]
    for module in (tool_calls, task_cancellation, task_budget):
        monkeypatch.setattr(module, 'monotonic', lambda: clock[0])
    return clock


def _supervised(monkeypatch, start):
    """只换「怎么起 worker」和「怎么收 worker」；监督者本身是真的。"""
    monkeypatch.setattr(tool_calls, '_start_call_worker', start)
    monkeypatch.setattr(tool_calls.exec_guard, 'task_process_groups',
                        lambda pid: (exec_guard.OwnedGroups(), True))
    monkeypatch.setattr(tool_calls.exec_guard, 'stop_task_worker',
                        lambda *a, **kw: dict(worker_reaped=True, status='fixture_cleanup'))


def _process():
    process = SimpleNamespace(pid=999999, returncode=None)
    process.poll = lambda: process.returncode
    return process


def _write_receipt(directory):
    invocation = json.loads((directory/'launch.json').read_text())['invocation_id']
    publish_json(directory/'worker-result.json', dict(
        invocation_id=invocation, finished_monotonic=tool_calls.monotonic(),
        response=dict(content=[dict(type='text', text=json.dumps(dict(ok=True)))],
                      isError=False)))
    return invocation


def test_a_supervised_call_records_one_job_and_no_status_file(tmp_path, monkeypatch):
    from operator_agent.execute import task_budget  # noqa: F401 - patched by _clock
    _clock(monkeypatch)
    run = tmp_path/'run'
    run.mkdir()
    (run/'declaration.json').write_text('{}')
    _prepare(run)
    process = _process()

    def start(directory):
        _write_receipt(directory)
        process.returncode = 0
        return process

    _supervised(monkeypatch, start)
    call = tool_calls.ToolCall('operator_verify', dict(run_dir=str(run), code=CODE))
    try:
        assert call.poll() is not None
        kinds = [event['kind'] for event in journal.read_events(run)]
        assert kinds == ['task_prepared', 'task_activated', 'job_queued',
                         'job_started', 'job_finished']
        facts = journal.invocation_view(call.directory, call.invocation_id)
        assert facts['journal_phase'] == 'completed'
        assert (facts['outcome'], facts['result']) == ('completed', 'received')
        assert facts['operation'] == 'verify' and facts['job_id'] == call.invocation_id
        # 终态是在 worker 收完之后写的：读者按它判断能不能再发下一次。
        assert facts['cleanup']['worker_reaped'] is True
        # 双写就是两个权威。
        assert not (call.directory/'run-status.json').exists()
    finally:
        call.close()

    # 一个 Job 结束不封整条流：同一个 run 的第二条调用照样能记。
    process2 = _process()

    def start2(directory):
        _write_receipt(directory)
        process2.returncode = 0
        return process2

    _supervised(monkeypatch, start2)
    second = tool_calls.ToolCall('operator_verify', dict(run_dir=str(run), code=CODE))
    try:
        assert second.poll() is not None
        projection = journal.fold(journal.read_events(run))
        assert len(projection['jobs']) == 2
        assert journal.is_open(projection) is True
    finally:
        second.close()


def test_a_directory_without_a_journal_gets_no_fabricated_prepare(tmp_path, monkeypatch):
    """旧任务目录照旧跑，但**不**替它补一条「已准备」——它没有那份记录，就没有。"""
    _clock(monkeypatch)
    run = tmp_path/'legacy'
    run.mkdir()
    (run/'declaration.json').write_text('{}')
    process = _process()

    def start(directory):
        _write_receipt(directory)
        process.returncode = 0
        return process

    _supervised(monkeypatch, start)
    call = tool_calls.ToolCall('operator_verify', dict(run_dir=str(run), code=CODE))
    try:
        assert call.poll() is not None
        assert call.journal.mode == 'absent'
        assert not (run/'journal.jsonl').exists()
        assert call.status['journal']['mode'] == 'absent'
    finally:
        call.close()


def test_a_missing_receipt_is_unknown_and_keeps_completed_credit(tmp_path, monkeypatch):
    """worker 退出了、回执读不到：记 `unknown`，并且**不再** `reject_completed`。

    这是文档里点名的那条语义缺陷：以前这里记成 failed 并按「什么都没有完成」结算，
    可远端那次执行很可能已经跑完。账不变，记录变诚实。
    """
    _clock(monkeypatch)
    run = tmp_path/'run'
    prepared = ht.prepare(DECL, run, host_budget=dict(wall_clock_s=60, max_verifications=3))
    budget = HostTaskBudget(run, prepared['contract_hash'])
    process = _process()

    def start(directory):
        invocation = json.loads((directory/'launch.json').read_text())['invocation_id']
        budget.reserve(SOURCE, purpose='explore', owner_invocation_id=invocation)
        process.returncode = 1
        return process

    _supervised(monkeypatch, start)
    call = tool_calls.ToolCall('operator_verify', dict(run_dir=str(run), code=CODE))
    try:
        response = call.poll()
        body = json.loads(response['content'][0]['text'])
        assert body['error'] == 'worker_failed'
        assert body['result_state'] == 'unknown' and body['result_reason'] == 'receipt_absent'
        assert body['execution_may_have_completed'] is True
        facts = journal.invocation_view(call.directory, call.invocation_id)
        assert (facts['outcome'], facts['result'], facts['result_reason']) == (
            'failed', 'unknown', 'receipt_absent')
        record = budget.snapshot()['records'][0]
        assert record['status'] == 'failed'
        # 没有读到回执 ⇒ 不能声称「响应被否掉」。
        assert record['settlement_basis'] == 'worker_reaped'
        assert 'verification_status_before_settlement' not in record
    finally:
        call.close()


def test_a_receipt_that_arrives_after_the_stop_is_recorded_as_received(tmp_path, monkeypatch):
    """回执读到了、只是落在取消之后：结果**到账了**，账按 cancelled 结。

    这一支以前和「什么都没有」混在一起，读了回执却记成 failed。
    """
    clock = _clock(monkeypatch)
    run = tmp_path/'run'
    run.mkdir()
    (run/'declaration.json').write_text('{}')
    _prepare(run)
    process = _process()

    def start(directory):
        process.returncode = 0          # worker 跑完了，但回执还没落地
        return process

    _supervised(monkeypatch, start)
    call = tool_calls.ToolCall('operator_verify', dict(run_dir=str(run), code=CODE))
    try:
        assert call.cancel()['accepted']
        clock[0] += 2.                  # 回执在取消之后才落地
        _write_receipt(call.directory)
        assert call.poll() is not None
        facts = journal.invocation_view(call.directory, call.invocation_id)
        assert (facts['outcome'], facts['result']) == ('cancelled', 'received')
        assert facts['stopped_reason'] == 'user_cancelled'
        body = json.loads(call.response['content'][0]['text'])
        assert body['result_state'] == 'received'
    finally:
        call.close()

def test_a_failed_dispatch_records_no_worker_to_reap(tmp_path, monkeypatch):
    """起不来 worker 时：`result=none`（不是「未知」），且写清没有东西要回收。

    没有这条，读者只能把「清理状态未知」读成永久未结算 —— 那会让任务永远关不上。
    """
    _clock(monkeypatch)
    run = tmp_path/'run'
    run.mkdir()
    (run/'declaration.json').write_text('{}')
    _prepare(run)

    def start(directory):
        raise OSError('fixture start failure')

    _supervised(monkeypatch, start)
    call = tool_calls.ToolCall('operator_verify', dict(run_dir=str(run), code=CODE))
    try:
        body = json.loads(call.response['content'][0]['text'])
        assert body['error'] == 'worker_start_failed'
        facts = journal.invocation_view(call.directory, call.invocation_id)
        assert (facts['outcome'], facts['result']) == ('failed', 'none')
        assert facts['cleanup']['status'] == 'worker_not_started'
        assert facts['cleanup']['worker_reaped'] is False
        assert facts['journal_phase'] == 'failed'
    finally:
        call.close()
