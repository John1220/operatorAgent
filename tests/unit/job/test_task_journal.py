"""journal 的守卫：**唯一权威**这件事必须自己立得住。

钉三件事，任何一条破了，「唯一权威」就退回成「又一个会漂的文件」：

1. 只追加，且 `rev` 连续；
2. **两层**结构 —— 一个 Job 的终态只约束它自己，只有 `task_closed` 封整条流；
3. 写入与回放**共用同一套校验**，拼错的 kind / 越界的枚举两侧都挡得住。
"""
import json
import threading

import pytest

from operator_agent.core import task_journal as j


TASK = 'task-0001'


def _prepare(run, task=TASK):
    return j.append(run, 'task_prepared', task_id=task)


def _activate(run, task=TASK):
    return j.append(run, 'task_activated', task_id=task)


def _finish(run, job_id, *, outcome='completed', result='received', **fields):
    """queued → started → finished。跳步现在会被拒，所以测试也必须走完整序列。"""
    _job(run, job_id)
    _job(run, job_id, kind='job_started')
    return _job(run, job_id, kind='job_finished', outcome=outcome, result=result, **fields)


def _job(run, job_id, *, kind='job_queued', op='verify', **fields):
    return j.append(run, kind, task_id=TASK, job_id=job_id,
                    operation=op, invocation=job_id.replace('job', 'inv'), **fields)


def test_an_absent_journal_is_a_state_not_an_error(tmp_path):
    """还没开始也是合法状态。用异常表达它，会让每个调用方写一遍 try。"""
    projection = j.fold(j.read_events(tmp_path))
    assert projection['task']['phase'] == 'absent'
    assert projection['rev'] == 0
    assert j.is_open(projection) is False


def test_appends_get_contiguous_revisions(tmp_path):
    for index, kind in enumerate(('task_prepared', 'task_activated'), start=1):
        event = j.append(tmp_path, kind, task_id=TASK)
        assert event['rev'] == index
    assert _job(tmp_path, 'job-1')['rev'] == 3
    assert j.fold(j.read_events(tmp_path))['rev'] == 3


def test_job_events_require_an_activated_task(tmp_path):
    """激活只有一条路。少了它，Job 事件不许先到 —— 拒绝要可归因。"""
    _prepare(tmp_path)
    with pytest.raises(j.JournalError) as caught:
        _job(tmp_path, 'job-1')
    assert caught.value.code == 'journal_not_activated'
    # 拒绝不许留下半条
    assert j.fold(j.read_events(tmp_path))['rev'] == 1


def test_a_finished_job_does_not_close_the_task(tmp_path):
    """**这是第一版写错的那条**：Job A 结束之后，Job B 必须还能开始。

    现有 host 流程在同一个 run 里会做三次 verify：首轮 L2 失败、修复后通过、
    再做一次独立 final。
    """
    _prepare(tmp_path)
    _activate(tmp_path)
    _job(tmp_path, 'job-a')
    _job(tmp_path, 'job-a', kind='job_started')
    _job(tmp_path, 'job-a', kind='job_finished', outcome='completed', result='received')

    # B 能开始
    _job(tmp_path, 'job-b')
    _job(tmp_path, 'job-b', kind='job_started')

    projection = j.fold(j.read_events(tmp_path))
    assert j.is_open(projection) is True
    assert projection['jobs']['job-a'] == dict(
        phase='completed', outcome='completed', result='received', result_reason='',
        operation='verify', invocation='inv-a')
    assert projection['jobs']['job-b']['phase'] == 'running'


def test_only_task_closed_seals_the_stream(tmp_path):
    """两轮结算之后再显式关闭，第三个 Job 才被拒；且拒绝时 rev 不变。"""
    _prepare(tmp_path)
    _activate(tmp_path)
    for name in ('job-a', 'job-b'):
        _finish(tmp_path, name)
    settled = j.fold(j.read_events(tmp_path))
    assert j.is_open(settled) is True
    assert set(settled['jobs']) == {'job-a', 'job-b'}

    j.append(tmp_path, 'task_closed', task_id=TASK, outcome='completed', reason='passed')
    closed = j.fold(j.read_events(tmp_path))
    assert j.is_open(closed) is False
    assert (closed['task']['outcome'], closed['task']['reason']) == ('completed', 'passed')

    before = closed['rev']
    with pytest.raises(j.JournalError) as caught:
        _job(tmp_path, 'job-c')
    assert caught.value.code == 'journal_closed'
    assert j.fold(j.read_events(tmp_path))['rev'] == before


def test_unknown_result_is_not_a_failure_and_must_say_why(tmp_path):
    """`unknown` 与 `failed` 并列，不是子类；而且必须带 result_reason。"""
    _prepare(tmp_path)
    _activate(tmp_path)
    _job(tmp_path, 'job-a')
    _job(tmp_path, 'job-a', kind='job_started')
    with pytest.raises(j.JournalError) as caught:
        _job(tmp_path, 'job-a', kind='job_finished', outcome='failed', result='unknown')
    assert caught.value.code == 'journal_missing_result_reason'

    _job(tmp_path, 'job-a', kind='job_finished', outcome='failed',
         result='unknown', result_reason='invocation_mismatch')
    row = j.fold(j.read_events(tmp_path))['jobs']['job-a']
    assert (row['outcome'], row['result'], row['result_reason']) == (
        'failed', 'unknown', 'invocation_mismatch')


def test_a_misspelled_kind_is_refused_by_both_sides(tmp_path):
    """写入侧挡住的，回放侧也要挡住 —— 否则手写进文件的坏事件就是唯一真相。"""
    _prepare(tmp_path)
    _activate(tmp_path)
    before = j.fold(j.read_events(tmp_path))['rev']
    with pytest.raises(j.JournalError) as caught:
        j.append(tmp_path, 'veirfy_started', task_id=TASK, job_id='job-a',
                 operation='verify')
    assert caught.value.code == 'journal_unknown_kind'
    assert j.fold(j.read_events(tmp_path))['rev'] == before

    # 手写一条同样的坏事件：fold 也要拒
    target = j.path_for(tmp_path)
    with open(target, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(schema=j.SCHEMA, rev=before + 1,
                                     kind='veirfy_started', job_id='job-a')) + '\n')
    with pytest.raises(j.JournalError) as caught:
        j.fold(j.read_events(tmp_path))
    assert caught.value.code == 'journal_unknown_kind'


def test_an_out_of_vocabulary_enum_is_refused(tmp_path):
    _prepare(tmp_path)
    _activate(tmp_path)
    _job(tmp_path, 'job-a')
    _job(tmp_path, 'job-a', kind='job_started')
    for fields, code in (
            (dict(outcome='exploded', result='received'), 'journal_unknown_outcome'),
            (dict(outcome='completed', result='maybe'), 'journal_unknown_result'),
    ):
        with pytest.raises(j.JournalError) as caught:
            _job(tmp_path, 'job-a', kind='job_finished', **fields)
        assert caught.value.code == code


def test_job_events_without_a_job_id_are_refused(tmp_path):
    _prepare(tmp_path)
    _activate(tmp_path)
    with pytest.raises(j.JournalError) as caught:
        j.append(tmp_path, 'job_started', task_id=TASK, operation='verify')
    assert caught.value.code == 'journal_missing_job'


def test_a_gap_in_revisions_is_corruption_not_a_skipped_event(tmp_path):
    target = j.path_for(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [dict(schema=j.SCHEMA, rev=1, kind='task_prepared', task_id=TASK),
             dict(schema=j.SCHEMA, rev=3, kind='task_activated', task_id=TASK)]
    target.write_text(''.join(json.dumps(line) + '\n' for line in lines), encoding='utf-8')
    with pytest.raises(j.JournalError) as caught:
        j.fold(j.read_events(tmp_path))
    assert caught.value.code == 'journal_gap'


def test_a_half_written_line_is_refused_rather_than_guessed(tmp_path):
    target = j.path_for(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"schema": "operator-task-journal-v2", "rev": 1, "kin',
                      encoding='utf-8')
    with pytest.raises(j.JournalError) as caught:
        j.read_events(tmp_path)
    assert caught.value.code == 'journal_corrupt'


def test_an_old_schema_stream_is_refused_not_reinterpreted(tmp_path):
    """v1 的 job 事件没有 `job_id`，无法归属。明确拒绝，不静默兼容。"""
    target = j.path_for(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(schema='operator-task-journal-v1', rev=1,
                                      kind='verify_started')) + '\n', encoding='utf-8')
    with pytest.raises(j.JournalError) as caught:
        j.read_events(tmp_path)
    assert caught.value.code == 'journal_schema_mismatch'


def test_concurrent_appends_do_not_collide(tmp_path):
    """`O_APPEND` 防撕裂，`flock` 防撞号 —— 两者缺一不可。"""
    _prepare(tmp_path)
    _activate(tmp_path)
    errors = []

    def writer(index):
        try:
            for step in range(5):
                # 每个 Job 一个唯一 id：重复排队现在会被拒（journal_job_exists）。
                _job(tmp_path, f'job-{index:02d}-{step}')
        except Exception as error:           # noqa: BLE001 - 线程里只能记账
            errors.append(error)

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    events = j.read_events(tmp_path)
    assert [event['rev'] for event in events] == list(range(1, 23))


def test_is_open_answers_only_about_the_task(tmp_path):
    """一个 Job 的回执仍是 unknown，不构成「任务还没结束」。

    这条防止恢复流程把「有个 Job 结果未知」读成「任务还开着」，从而漏掉封账。
    """
    _prepare(tmp_path)
    _activate(tmp_path)
    _finish(tmp_path, 'job-a', outcome='failed', result='unknown',
            result_reason='receipt_absent')
    projection = j.fold(j.read_events(tmp_path))
    assert projection['jobs']['job-a']['result'] == 'unknown'
    assert j.is_open(projection) is True          # 任务本身还没关

    j.append(tmp_path, 'task_closed', task_id=TASK, outcome='failed', reason='fix_error')
    closed = j.fold(j.read_events(tmp_path))
    assert j.is_open(closed) is False             # 封账之后 Job 的 unknown 不算数
    assert closed['jobs']['job-a']['result'] == 'unknown'   # 但诊断入口仍在


def test_a_job_cannot_start_on_an_empty_directory(tmp_path):
    """没有 prepare 就没有契约。凭空开 Job 是无主调用，不是「还没开始」。

    第一版的病：空目录直接 `job_started` 被接受，投影显示 task=absent + job=running ——
    一个既不属于任何任务、又在跑的东西。
    """
    with pytest.raises(j.JournalError) as caught:
        _job(tmp_path, 'job-a', kind='job_started')
    assert caught.value.code == 'journal_not_activated'
    assert j.read_events(tmp_path) == []
    assert j.fold(j.read_events(tmp_path))['task']['phase'] == 'absent'


def test_a_finished_job_cannot_return_to_running(tmp_path):
    """终态不可回退。

    第一版的病：正常完成后追加一条迟到的 `job_started`，投影会得到
    `phase=running` 与 `outcome=completed, result=received` 并存的**自相矛盾**记录 ——
    形状校验看不出来，因为它不在单条事件里，只在序列里。
    """
    _prepare(tmp_path)
    _activate(tmp_path)
    _finish(tmp_path, 'job-a', outcome='completed', result='received')
    before = j.fold(j.read_events(tmp_path))['rev']

    with pytest.raises(j.JournalError) as caught:
        _job(tmp_path, 'job-a', kind='job_started')
    assert caught.value.code == 'journal_job_closed'

    after = j.fold(j.read_events(tmp_path))
    assert after['rev'] == before
    row = after['jobs']['job-a']
    assert (row['phase'], row['outcome'], row['result']) == ('completed', 'completed', 'received')


def test_skipping_a_job_phase_is_refused(tmp_path):
    _prepare(tmp_path); _activate(tmp_path)
    with pytest.raises(j.JournalError) as caught:
        _job(tmp_path, 'job-a', kind='job_finished', outcome='completed', result='received')
    assert caught.value.code == 'journal_job_unknown'
    _job(tmp_path, 'job-a')
    with pytest.raises(j.JournalError) as caught:
        _job(tmp_path, 'job-a', kind='job_finished', outcome='completed', result='received')
    assert caught.value.code == 'journal_bad_transition'


def test_replay_refuses_an_out_of_order_stream(tmp_path):
    """手写进文件的乱序事件不能因为「形状合法」就被接受 —— 回放走同一份判据。"""
    target = j.path_for(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        dict(schema=j.SCHEMA, rev=1, kind='task_prepared', task_id=TASK),
        dict(schema=j.SCHEMA, rev=2, kind='task_activated', task_id=TASK),
        dict(schema=j.SCHEMA, rev=3, kind='job_queued', task_id=TASK,
             job_id='job-a', operation='verify'),
        dict(schema=j.SCHEMA, rev=4, kind='job_finished', task_id=TASK,
             job_id='job-a', operation='verify', outcome='completed', result='received'),
    ]
    target.write_text(''.join(json.dumps(line) + '\n' for line in lines), encoding='utf-8')
    with pytest.raises(j.JournalError) as caught:
        j.fold(j.read_events(tmp_path))
    assert caught.value.code == 'journal_bad_transition'


DECL = dict(operator='relu', request='relu fp32', shapes=[[4, 8]], precision='fp32',
            dsl='python')


def test_prepare_records_the_task_once_and_keeps_its_identity(tmp_path):
    """`task_prepared` 只有 prepare 一个写者，且重复 prepare 不追加第二条。

    重复时不追加，是因为这条事件断言的是「这个目录有契约」——它要么发生过，
    要么没发生过，不该按调用次数增长。
    """
    from operator_agent.app import hitl_tools as ht
    ht.prepare(DECL, tmp_path)
    events = j.read_events(tmp_path)
    assert [event['kind'] for event in events] == ['task_prepared']
    recorded = json.loads((tmp_path/'run.json').read_text())['task_id']
    assert events[0]['task_id'] == recorded and recorded
    ht.prepare(DECL, tmp_path)
    assert len(j.read_events(tmp_path)) == 1
    # 后面的写者不必各自去读 run.json：缺省就从流里继承同一个身份。
    j.append(tmp_path, 'task_activated')
    queued = j.append(tmp_path, 'job_queued', job_id='job-1', operation='verify',
                     invocation='inv-1')
    assert queued['task_id'] == recorded


def test_prepare_refuses_a_closed_task_directory(tmp_path):
    """封账之后再准备 = 想在同一目录开第二段工作。那件事的正确做法是新目录。"""
    from operator_agent.app import hitl_tools as ht
    ht.prepare(DECL, tmp_path)
    j.append(tmp_path, 'task_closed', outcome='completed', reason='passed')
    with pytest.raises(ht.ToolError) as caught:
        ht.prepare(DECL, tmp_path)
    assert caught.value.code == 'task_closed'


def test_prepare_refuses_on_an_unreadable_journal(tmp_path):
    """账本读不动时不再往里写：那种流已经不能当权威，继续写只会让「哪份是对的」更说不清。"""
    from operator_agent.app import hitl_tools as ht
    j.path_for(tmp_path).write_text('{"schema": "operator-task-journal-v2", "rev": 1, "kin',
                                    encoding='utf-8')
    with pytest.raises(ht.ToolError) as caught:
        ht.prepare(DECL, tmp_path)
    assert caught.value.code == 'task_journal_unusable'


def test_a_retry_after_a_refusal_continues_the_same_stream(tmp_path):
    """被拒之后重试：`rev` 必须**接着**走，不是从断点重来。

    这一条钉的是「拒绝不留痕」和「拒绝不占号」同时成立：被拒的事件不许写进流
    （否则它就成了唯一真相），也不许让下一条拿到同一个号（否则两条事件同号）。
    """
    _prepare(tmp_path)
    _activate(tmp_path)
    _finish(tmp_path, 'job-a')
    before = j.fold(j.read_events(tmp_path))

    # 迟到的终态被拒（终态不可回退）……
    with pytest.raises(j.JournalError) as caught:
        _job(tmp_path, 'job-a', kind='job_started')
    assert caught.value.code == 'journal_job_closed'
    # ……重试同一个动作仍然被拒，rev 与投影都不动。
    with pytest.raises(j.JournalError) as again:
        _job(tmp_path, 'job-a', kind='job_started')
    assert again.value.code == 'journal_job_closed'
    after = j.fold(j.read_events(tmp_path))
    assert after['rev'] == before['rev'] and after['history'] == before['history']

    # 换一条合法的动作：号码紧接在被拒之后，没有空洞。
    _job(tmp_path, 'job-b')
    assert j.fold(j.read_events(tmp_path))['rev'] == before['rev'] + 1


def test_retrying_a_terminal_event_does_not_overwrite_the_first_answer(tmp_path):
    """重复上报同一条终态是常态（监督者重试）。第一次的答案必须留住。

    终态**不可回退**也包括「不许被同一件事的第二次说法覆盖」：先 completed，
    后到的 failed 若被接受，投影就同时背着两个结论，而流里没有任何东西说哪个算数。
    """
    _prepare(tmp_path)
    _activate(tmp_path)
    _finish(tmp_path, 'job-a', outcome='completed', result='received')
    row = j.fold(j.read_events(tmp_path))['jobs']['job-a']
    with pytest.raises(j.JournalError) as caught:
        _job(tmp_path, 'job-a', kind='job_finished', outcome='failed', result='received')
    assert caught.value.code == 'journal_job_closed'
    same = j.fold(j.read_events(tmp_path))['jobs']['job-a']
    assert same == row


def test_concurrent_retries_of_one_job_queue_exactly_once(tmp_path):
    """同一条 `job_queued` 被并发重试：只许进去一次，`rev` 仍然连续。

    真实来源是「父进程不确定上一次发没发出去」——重试是正常动作，不是异常。
    这里要的是「重复的尝试既不加序号，也不留第二条排队记录」。
    """
    _prepare(tmp_path)
    _activate(tmp_path)
    accepted, refused = [], []
    barrier = threading.Barrier(4)

    def writer(index):
        barrier.wait()
        try:
            accepted.append(_job(tmp_path, 'job-a')['rev'])
        except j.JournalError as error:       # noqa: BLE001 - 只有一类预期内的拒绝
            refused.append(error)

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(accepted) == 1 and len(refused) == 3
    assert {error.code for error in refused} == {'journal_job_exists'}
    events = j.read_events(tmp_path)
    assert [event['rev'] for event in events] == list(range(1, len(events) + 1))
    assert [event['kind'] for event in events].count('job_queued') == 1
