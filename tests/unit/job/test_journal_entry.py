"""跨进程追加入口：`operator-agent journal-append`。

插件（DSH 宿主）不能自己写账本 —— `rev` = 「读最大值 + 1」、落盘要 `flock`，
两件事都只在 `task_journal.append` 一处。所以它**提交**，Python 写。

这里钉三件事：
1. 合法事件真的写进去，并把写下去的那条（含 `rev`）回给提交方；
2. 账本拒绝时回稳定 `code`（CLI 退出码 2），并且 `rev` 不动；
3. 坏输入是**回执**，不是崩溃 —— 提交方要能按 code 分支。
"""
import json
from pathlib import Path
import subprocess
import sys

from operator_agent.core import task_journal as journal

REPO = Path(__file__).resolve().parents[3]


def submit(run, payload):
    proc = subprocess.run(
        [sys.executable, '-m', 'operator_agent.app.cli', 'journal-append',
         '--run-dir', str(run)],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        text=True, capture_output=True, cwd=str(REPO))
    return proc.returncode, json.loads(proc.stdout)


def test_a_submitted_event_is_written_and_echoed_back(tmp_path):
    code, body = submit(tmp_path, dict(kind='task_prepared', task_id='t-1'))
    assert code == 0 and body['ok'] is True
    assert body['event']['rev'] == 1 and body['event']['schema'] == journal.SCHEMA
    code, body = submit(tmp_path, dict(kind='task_activated'))
    assert code == 0 and body['event']['rev'] == 2
    assert body['event']['task_id'] == 't-1'      # 身份从流里继承，不必每次重传
    assert [event['kind'] for event in journal.read_events(tmp_path)] == [
        'task_prepared', 'task_activated']


def test_a_refusal_carries_a_stable_code_and_moves_nothing(tmp_path):
    submit(tmp_path, dict(kind='task_prepared', task_id='t-1'))
    submit(tmp_path, dict(kind='task_closed', outcome='completed', reason='host_closed'))
    before = journal.read_events(tmp_path)
    code, body = submit(tmp_path, dict(kind='task_closed', outcome='completed'))
    assert code == 2 and body['ok'] is False and body['error'] == 'journal_closed'
    assert journal.read_events(tmp_path) == before


def test_an_unknown_kind_or_broken_json_is_a_receipt_not_a_crash(tmp_path):
    submit(tmp_path, dict(kind='task_prepared', task_id='t-1'))
    code, body = submit(tmp_path, dict(kind='veirfy_started'))
    assert code == 2 and body['error'] == 'journal_unknown_kind'
    code, body = submit(tmp_path, '{"kind": ')
    assert code == 2 and body['error'] == 'invalid_event'
    code, body = submit(tmp_path, dict(outcome='completed'))
    assert code == 2 and body['error'] == 'invalid_event'
    assert len(journal.read_events(tmp_path)) == 1
