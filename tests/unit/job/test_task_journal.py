"""journal 的守卫：**唯一权威**这件事必须自己立得住。

这些用例钉的不是某个函数的输出，而是三条纪律：
只追加、`rev` 连续、投影不缓存。任何一条破了，「唯一权威」就退回成「又一个会漂的文件」。
"""
import json
import threading
from pathlib import Path

import pytest

from operator_agent.core import task_journal


def test_an_absent_journal_is_a_state_not_an_error(tmp_path):
    """还没开始也是合法状态。用异常表达它，会让每个调用方写一遍 try。"""
    projection = task_journal.fold(task_journal.read_events(tmp_path))
    assert projection['phase'] == 'absent'
    assert projection['rev'] == 0
    assert task_journal.needs_recovery(projection) is False


def test_appends_get_contiguous_revisions(tmp_path):
    for index in range(1, 4):
        event = task_journal.append(tmp_path, 'prepared' if index == 1 else 'running',
                                    invocation='a' * 32, note=index)
        assert event['rev'] == index
    projection = task_journal.fold(task_journal.read_events(tmp_path))
    assert projection['rev'] == 3


def test_the_projection_follows_the_events(tmp_path):
    """相位由 kind 推进，结果态由 result_state 单独一格 —— 两者正交。"""
    task_journal.append(tmp_path, 'prepared', invocation='a' * 32)
    task_journal.append(tmp_path, 'verify_requested', invocation='a' * 32)
    started = task_journal.fold(task_journal.read_events(tmp_path))
    assert started['phase'] == 'queued'

    task_journal.append(tmp_path, 'verify_started', invocation='a' * 32)
    task_journal.append(tmp_path, 'completed', invocation='a' * 32, result_state='received')
    done = task_journal.fold(task_journal.read_events(tmp_path))
    assert (done['phase'], done['result']) == ('completed', 'received')
    assert task_journal.needs_recovery(done) is False


def test_result_unknown_is_not_a_failure(tmp_path):
    """两者并列，不是父子 —— 这正是「执行可能已经发生」要说清的那件事。"""
    task_journal.append(tmp_path, 'prepared', invocation='a' * 32)
    task_journal.append(tmp_path, 'completed', invocation='a' * 32, result_state='unknown')
    projection = task_journal.fold(task_journal.read_events(tmp_path))
    assert projection['phase'] == 'completed'
    assert projection['result'] == 'unknown'
    assert task_journal.needs_recovery(projection) is True


def test_a_failure_after_a_terminal_phase_is_refused(tmp_path):
    task_journal.append(tmp_path, 'prepared', invocation='a' * 32)
    task_journal.append(tmp_path, 'failed', invocation='a' * 32)
    with pytest.raises(task_journal.JournalError) as caught:
        task_journal.append(tmp_path, 'running', invocation='a' * 32)
    assert caught.value.code == 'journal_closed'


def test_a_gap_in_revisions_is_corruption_not_a_skipped_event(tmp_path):
    """断号意味着有人绕过这个模块写过 —— 那种流的顺序不可信，不许当没事。"""
    target = task_journal.path_for(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        dict(schema=task_journal.SCHEMA, rev=1, kind='prepared', invocation='a' * 32),
        dict(schema=task_journal.SCHEMA, rev=3, kind='running', invocation='a' * 32),
    ]
    target.write_text(''.join(json.dumps(line) + '\n' for line in lines), encoding='utf-8')
    with pytest.raises(task_journal.JournalError) as caught:
        task_journal.fold(task_journal.read_events(tmp_path))
    assert caught.value.code == 'journal_gap'


def test_a_half_written_line_is_refused_rather_than_guessed(tmp_path):
    target = task_journal.path_for(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"schema": "operator-task-journal-v1", "rev": 1, "kin',
                      encoding='utf-8')
    with pytest.raises(task_journal.JournalError) as caught:
        task_journal.read_events(tmp_path)
    assert caught.value.code == 'journal_corrupt'


def test_concurrent_appends_do_not_collide(tmp_path):
    """`O_APPEND` 防撕裂，`flock` 防撞号 —— 两者缺一不可。"""
    errors = []

    def writer(index):
        try:
            for _ in range(5):
                task_journal.append(tmp_path, 'verify_started',
                                    invocation=f'{index:032d}', writer=index)
        except Exception as error:           # noqa: BLE001 - 线程里只能记账
            errors.append(error)

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    events = task_journal.read_events(tmp_path)
    assert [event['rev'] for event in events] == list(range(1, 21))


def test_needs_recovery_covers_a_dead_parent(tmp_path):
    """父进程死了：相位停在非终止处、事件流不再增长。这与「还在跑」在纸上一样，
    区别只能由**时间**给出 —— 所以这里只钉住「非终止就要人看一眼」。"""
    task_journal.append(tmp_path, 'prepared', invocation='a' * 32)
    task_journal.append(tmp_path, 'verify_started', invocation='a' * 32)
    projection = task_journal.fold(task_journal.read_events(tmp_path))
    assert projection['phase'] == 'running'
    assert task_journal.needs_recovery(projection) is True