"""Owned workers for interruptible foreground tools; stdout stays protocol-only.

The CLI/MCP parent supervises a *trusted* dispatcher, not candidate execution.
Candidate guards and judgement rules remain in the existing tool implementations.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import time
from ..execute.task_clock import monotonic
import uuid

from ..execute import exec_guard
from ..execute.fake_backend.config import tool_context
from ..execute.task_cancellation import (cancellation, launch_identity, partial_artifacts,
                                publish_json, read_json, request_cancel, stop_event)

LONG_TOOLS = frozenset(('operator_verify', 'operator_profile', 'operator_repair', 'operator_bench'))

#: 长工具各自的 Job 类型。词表在 `core.task_journal`，这里只做名字映射。
JOB_OPERATION = {'operator_verify': 'verify', 'operator_profile': 'profile',
                 'operator_bench': 'bench', 'operator_repair': 'repair'}
_REPAIR_CONTEXT = None
_HOST_CONTEXT = None
_PROFILE_CONTEXT = None


def repair_context(run):
    """Trusted worker context; never read a user-supplied deadline argument."""
    if _REPAIR_CONTEXT and Path(run).resolve() == Path(_REPAIR_CONTEXT['run_dir']):
        return dict(_REPAIR_CONTEXT)
    return None


def host_context(run):
    """Supervisor-owned binding, never accepted as a public tool argument."""
    if _HOST_CONTEXT and Path(run).resolve() == Path(_HOST_CONTEXT['run_dir']):
        return dict(_HOST_CONTEXT)
    return None


def profile_context(run):
    """Supervisor-owned profile identity/deadline; never accepted as a public argument."""
    if _PROFILE_CONTEXT and Path(run).resolve() == Path(_PROFILE_CONTEXT['run_dir']):
        return dict(_PROFILE_CONTEXT)
    return None


def _error(error, detail, **fields):
    from ..app.mcp_server import _text_result
    return _text_result(dict(ok=False, error=error, detail=detail, **fields), is_error=True)


def _start_call_worker(directory):
    return exec_guard.start_task_worker([sys.executable, '-m', 'operator_agent.app.tool_calls',
        str(directory)], cwd=str(Path(__file__).resolve().parents[2]),
        log_path=directory/'execution.log')


class _JobJournal:
    """这一次 invocation 的 Job 事件：queued → started → finished。

    两条边界是刻意的：

    * **账本不存在就不写**。缺账本时补一条 `task_prepared`，等于替一个没有契约记录的
      目录伪造「已准备」——那是这套东西里最贵的一种谎。旧目录照旧跑，只是没有账。
    * **账本拒绝时不抛给调用方**。任务封账、账本损坏时，调用仍然执行（拒绝执行是另一件
      事，得有另一个决定），但拒绝的理由如实挂在 `status['journal']` 与响应体上，
      不静默变成空操作。
    """

    def __init__(self, run, invocation, operation, status):
        from ..core import task_journal
        self.run = Path(run)
        self.invocation = invocation
        self.operation = operation
        self.status = status
        self.mode = 'journal'
        self.error = None
        try:
            events = task_journal.read_events(self.run)
        except task_journal.JournalError as exc:
            self.mode, self.error = 'refused', f'{exc.code}: {exc.detail}'
            self._publish()
            return
        if not events:
            self.mode = 'absent'
            self._publish()
            return
        try:
            if task_journal.fold(events)['task']['phase'] == 'prepared':
                # 第一个 Job 的发起方把任务从「已准备」推到「进行中」。
                task_journal.append(self.run, 'task_activated')
        except task_journal.JournalError as exc:
            self.mode, self.error = 'refused', f'{exc.code}: {exc.detail}'
        self._publish()

    def _publish(self):
        self.status['journal'] = dict(mode=self.mode, **({'error': self.error} if self.error else {}))

    def _append(self, kind, **fields):
        from ..core import task_journal
        if self.mode != 'journal':
            return
        try:
            task_journal.append(self.run, kind, job_id=self.invocation,
                                invocation=self.invocation,
                                operation=self.operation, **fields)
        except task_journal.JournalError as exc:
            # 一条写不进去，后面的也不必再试 —— 账本在这里已经不可用。
            self.mode, self.error = 'refused', f'{exc.code}: {exc.detail}'
            self._publish()

    def queued(self):
        self._append('job_queued', tool=self.status['tool'], stage='starting',
                     pid=self.status['pid'], pid_identity=self.status['pid_identity'],
                     started_at=self.status['started_at'])

    def started(self):
        """派发开始。**在起 worker 之前**写：起不来也是一次「开始了、没成」，
        而不是一条从来没有下文的排队记录 —— 后者会让读者永远等一个不会结束的 Job。"""
        self._append('job_started')

    def finished(self, receipt, *, cleanup=None, stopped_reason='', finished_at=None):
        """终态事件。**在 cleanup 之后**写：这条事件的意思是「这次调用结束了」，
        所以它落地时 worker 必须已经收完 —— 读者按它判断能不能再发下一次。"""
        fields = dict(outcome=receipt['outcome'], result=receipt['result'],
                      stopped_reason=stopped_reason or '',
                      finished_at=finished_at or time.time())
        if receipt.get('result_reason'):
            fields['result_reason'] = receipt['result_reason']
        if isinstance(cleanup, dict):
            fields['cleanup'] = dict(cleanup)
        if self.status.get('worker_pid'):
            fields['worker_pid'] = self.status['worker_pid']
        self._append('job_finished', **fields)


class ToolCall:
    """One parent-owned Popen, immutable identity, cancel marker and final receipt."""
    def __init__(self, name, arguments):
        from ..app.hitl_tools import ToolError
        from ..app.mcp_server import _guard_host_tool
        _guard_host_tool(name)
        if name not in LONG_TOOLS:
            raise ToolError('unknown_tool', str(name))
        if not isinstance(arguments, dict) or not arguments.get('run_dir'):
            raise ToolError('run_dir_required', f'{name} requires run_dir')
        self.run = Path(arguments['run_dir']).resolve()
        self.context = tool_context(arguments.get('backend'),
            model_mode='live' if name == 'operator_repair' else 'not_used')
        from ..execute.backend_ownership import owned_persistent_ssh
        self.graceful_remote = owned_persistent_ssh(
            arguments, persistent_default=name == 'operator_bench')
        self.repair_policy = None
        self.host_budget = None
        self.profile_context = None
        self.deadline = None
        if name == 'operator_repair':
            from ..execute.task_budget import TaskBudgetConfig
            try:
                self.repair_policy = TaskBudgetConfig(**{
                    key: arguments[key] for key in TaskBudgetConfig.__dataclass_fields__
                    if key in arguments})
            except (TypeError, ValueError) as exc:
                raise ToolError('invalid_task_budget', str(exc)) from exc
        if not (self.run/'declaration.json').is_file():
            raise ToolError('run_not_prepared', 'Prepare the run directory first.')
        if name == 'operator_verify':
            from ..execute.host_task_budget import HostTaskBudget, HostTaskBudgetError, host_task_enabled
            if host_task_enabled(self.run):
                from .host_workflow import prepared_task
                _, _, contract = prepared_task(self.run)
                try:
                    self.host_budget = HostTaskBudget(self.run, contract.hash)
                    remaining = self.host_budget.snapshot()['remaining_s']
                except HostTaskBudgetError as error:
                    raise ToolError(error.code, error.detail) from error
                if remaining <= 0:
                    raise ToolError('host_task_deadline', 'Host verification task deadline expired.')
        self.invocation_id = uuid.uuid4().hex
        self.started_monotonic = monotonic()
        if self.repair_policy:
            self.deadline = self.started_monotonic + self.repair_policy.wall_clock_s
        elif self.host_budget:
            self.deadline = self.started_monotonic + remaining
        if name == 'operator_profile':
            from ..execute.host_task_budget import (HostTaskBudget, HostTaskBudgetError,
                                                    host_task_enabled)
            from ..execute.profile_budget import ProfileBudget, ProfileBudgetError
            if not host_task_enabled(self.run):
                raise ToolError('host_task_required',
                                'operator_profile requires a bounded host task')
            from .host_workflow import prepared_task
            _, _, contract = prepared_task(self.run)
            try:
                task_snapshot = HostTaskBudget(self.run, contract.hash).snapshot()
                profile_snapshot = ProfileBudget(self.run).snapshot()
            except (HostTaskBudgetError, ProfileBudgetError) as error:
                raise ToolError(error.code, error.detail) from error
            remaining = task_snapshot['remaining_s']
            if task_snapshot['terminal'] or task_snapshot['expired'] or remaining < 60:
                raise ToolError('profile_deadline',
                                'the host task has less than 60 seconds available for a profile')
            if task_snapshot['unfinished']:
                raise ToolError('profile_task_busy',
                                'settle the in-flight verification before profiling')
            if profile_snapshot['in_flight']:
                raise ToolError('profile_in_flight', 'another profile is still in flight')
            if profile_snapshot['remaining'] <= 0:
                raise ToolError('profile_budget', 'profile budget exhausted (2 per task)')
            requested = arguments.get('timeout', 300)
            requested = (float(requested) if type(requested) in (int, float)
                         and requested > 0 else 300.0)
            cap = min(300.0, requested, float(remaining))
            self.profile_context = dict(
                run_dir=str(self.run), invocation_id=self.invocation_id,
                deadline=self.started_monotonic + cap + 20.0)
            self.deadline = self.profile_context['deadline']
        self.directory = self.run/'tool-calls'/self.invocation_id
        self.directory.mkdir(parents=True)
        launch_identity(self.directory, self.invocation_id, kind='tool_call', tool=name,
                        **self.context)
        args = dict(arguments, run_dir=str(self.run))
        # File references must survive the worker's fixed repository cwd.
        if args.get('code_file'):
            args['code_file'] = str(Path(args['code_file']).resolve())
        if name == 'operator_bench' and isinstance(args.get('candidates'), list):
            args['candidates'] = [dict(c, code_file=str(Path(c['code_file']).resolve()))
                if isinstance(c, dict) and isinstance(c.get('code_file'), str) else c
                for c in args['candidates']]
        request = dict(name=name, arguments=args)
        if self.repair_policy:
            request['repair_budget'] = dict(run_dir=str(self.run), directory=str(self.directory),
                invocation_id=self.invocation_id, started_monotonic=self.started_monotonic,
                deadline=self.deadline, task_budget=self.repair_policy.as_dict())
        if self.host_budget:
            request['host_budget_context'] = dict(run_dir=str(self.run),
                invocation_id=self.invocation_id, deadline=self.deadline)
        if self.profile_context:
            request['profile_context'] = dict(self.profile_context)
        publish_json(self.directory/'request.json', request)
        self.status = dict(phase='running', stage='starting', kind='tool_call', tool=name,
            pid=os.getpid(), invocation_id=self.invocation_id, started_at=time.time(),
            pid_identity=exec_guard.process_identity(os.getpid()),
            run_dir=str(self.run), tool_call_dir=str(self.directory), **self.context)
        if self.repair_policy:
            self.status['task_budget'] = self.repair_policy.as_dict()
        if self.host_budget:
            self.status['host_task'] = self.host_budget.snapshot()
        self.response = None
        self.process = None
        self.groups = exec_guard.OwnedGroups()
        self._last_discovery = 0.
        self.operation = JOB_OPERATION[name]
        self.journal = _JobJournal(self.run, self.invocation_id, self.operation, self.status)
        self.journal.queued()
        self.journal.started()
        try:
            self.process = _start_call_worker(self.directory)
            self.groups.add(self.process.pid)
            self.status.update(stage='executing', worker_pid=self.process.pid)
        except OSError as exc:
            self.status.update(phase='failed', error='worker_start_failed', detail=str(exc),
                               finished_at=time.time(),
                               # 没有 worker 起过：没有东西要被回收，也不该被读成「清理状态未知」
                               # （那会让「这次调用结束了没有」永远没有答案）。
                               cleanup=dict(worker_reaped=False, status='worker_not_started',
                                            remote_execution='unknown', provider_call='unknown'))
            self.response = _error('worker_start_failed', str(exc))
            self.receipt = dict(outcome='failed', result='none',
                                result_reason='', stopped_reason='worker_start_failed')
            self._finish()

    def cancel(self):
        return request_cancel(self.directory, self.invocation_id)

    def _repair_snapshot(self, now=None):
        from ..execute.task_budget import snapshot
        root = self.directory
        reference = read_json(root/'repair-evidence.json') or {}
        if (reference.get('invocation_id') == self.invocation_id
                and isinstance(reference.get('evidence'), str)):
            candidate = Path(reference['evidence']).resolve()
            if (candidate.parent == self.run and candidate.name.startswith('repair-')
                    and candidate.name[7:].isdigit()):
                root = candidate
        return snapshot(root, self.repair_policy, self.started_monotonic, now=now)

    def poll(self):
        if self.response is not None:
            return self.response
        reason, event_at = stop_event(self.directory, self.invocation_id, self.deadline)
        if self.process.poll() is None and not reason:
            if monotonic()-self._last_discovery >= .1:
                groups, discovered = exec_guard.task_process_groups(self.process.pid)
                self.groups.update(groups)
                self.groups.discovery_complete &= discovered
                self._last_discovery = monotonic()
                if self.repair_policy:
                    self.status['budget'] = self._repair_snapshot()
            return None
        cleanup = (exec_guard.stop_task_worker(self.process, self.groups, graceful_s=12)
                   if reason and self.graceful_remote else
                   exec_guard.stop_task_worker(self.process, self.groups))
        reason, event_at = stop_event(self.directory, self.invocation_id, self.deadline)
        from ..execute.backend_ownership import cleanup_state
        cleanup.update(remote_execution=(cleanup_state(self.directory, self.invocation_id)
                                         if self.graceful_remote else 'unknown'),
                       provider_call='unknown')
        result = read_json(self.directory/'worker-result.json') or {}
        finished = result.get('finished_monotonic')
        fault = self._receipt_fault(result)
        accepted = fault is None and (event_at is None or finished <= event_at)
        if accepted:
            self.response = result['response']
            # Tool refusal is a completed call, not a cancelled/abandoned result.
            self.status.update(phase='done', stopped_reason='tool_returned')
            if self.repair_policy:
                body = json.loads(self.response['content'][0]['text'])
                self.status['stopped_reason'] = body.get('stopped_reason') or 'tool_returned'
                self.status['budget'] = self._repair_snapshot(now=finished)
        elif reason == 'budget_wall_clock':
            self.status.update(phase='done', stopped_reason=reason)
            self.response = _error('task_budget_exhausted', 'The task wall-clock budget expired.',
                                  passed=False, phase='done', stopped_reason=reason,
                                  **self._result_facts(fault, 'stopped_before_receipt'))
        elif reason:
            self.status.update(phase='cancelled', stopped_reason='user_cancelled')
            self.response = _error('task_cancelled', 'The owning supervisor stopped this call.',
                                   phase='cancelled', stopped_reason='user_cancelled',
                                   **self._result_facts(fault, 'stopped_before_receipt'))
        else:
            self.status.update(phase='failed', stopped_reason='worker_failed')
            # 这里以前只报一句 failed。worker 退出了、回执读不到，**不等于它会话失败** ——
            # 远端那次执行很可能已经跑完。说清楚「结果未知」比猜一个结论诚实。
            self.response = _error('worker_failed', 'Worker exited without a terminal result.',
                                   execution_may_have_completed=True,
                                   **self._result_facts(fault, 'receipt_absent'))
        self.receipt = self._classify(fault=fault, accepted=accepted)
        if self.host_budget and cleanup.get('worker_reaped') is True:
            if not accepted:
                # `reject_completed` 只在**读到并否掉**一个回执时用（它确实迟到了）。
                # 读不到回执时账不动：worker 可能已经把证据落完，是我们没看见。
                self.host_budget.settle_owner(self.invocation_id,
                    outcome='cancelled' if self.receipt['outcome']=='cancelled' else 'failed',
                    reject_completed=fault is None)
        if self.profile_context and cleanup.get('worker_reaped') is True:
            self._settle_profile()
        if self.repair_policy and 'budget' not in self.status:
            self.status['budget'] = self._repair_snapshot()
        elif self.repair_policy and not accepted:
            self.status['budget'] = self._repair_snapshot()
        self.status.update(cleanup=cleanup, cancellation=cancellation(self.directory, self.invocation_id),
            finished_at=time.time(), partial_artifacts=partial_artifacts(self.run))
        self._finish()
        return self.response

    def _receipt_fault(self, result):
        """worker 的终态回执哪里不合格；合格返回 None。**只判回执，不判结果好坏。**"""
        if not result:
            return 'receipt_absent'
        if result.get('invocation_id') != self.invocation_id:
            return 'invocation_mismatch'
        finished = result.get('finished_monotonic')
        if (type(finished) not in (float, int) or finished > monotonic()
                or finished < self.started_monotonic):
            return 'invalid_timestamp'
        if not isinstance(result.get('response'), dict):
            return 'receipt_absent'
        return None

    @staticmethod
    def _result_facts(fault, stopped):
        """`result` 说的是「回执到账了没有」，与 `outcome` 正交。"""
        if fault is None:
            return dict(result_state='received')
        return dict(result_state='unknown', result_reason=stopped)

    def _classify(self, *, fault, accepted):
        """这次调用怎么记账。账不变，记录变诚实：分类只影响写下来的两个词。"""
        if accepted:
            return dict(outcome='completed', result='received', result_reason='')
        if self.status['phase'] == 'cancelled':
            outcome = 'cancelled'
        else:
            outcome = 'failed'          # 截止时间停的也按 failed 结：账不退款
        if fault is None:
            # 回执读到了，只是落在停止之后：执行确实产出了结果，只是没赶上这次调用。
            return dict(outcome=outcome, result='received', result_reason='')
        if self.status.get('error') == 'worker_start_failed':
            # 没有任何 worker 起过：这不是「未知」，是「没有」。
            return dict(outcome='failed', result='none', result_reason='')
        if self.status['phase'] == 'failed':
            return dict(outcome='failed', result='unknown', result_reason=fault)
        return dict(outcome=outcome, result='unknown', result_reason='stopped_before_receipt')

    def _settle_profile(self):
        """Settle an abandoned profile only after its worker was reaped."""
        from ..execute.profile_budget import ProfileBudget, ProfileBudgetError
        try:
            ledger = ProfileBudget(self.run)
            entry = next((row for row in ledger.snapshot()['entries']
                          if row.get('owner_invocation_id') == self.invocation_id
                          and row['status'] == 'reserved'), None)
            if entry is None:
                return
            from ..execute.remote_executor import cleanup_profile_transport
            transport = self.run / f'profile-{entry["id"]}' / 'transport.json'
            remote_cleanup = (cleanup_profile_transport(transport)
                              if transport.is_file() else 'not_started')
            phase = self.status.get('phase')
            ledger.settle_owner(
                self.invocation_id,
                outcome='cancelled' if phase == 'cancelled' else 'failed',
                error=f'supervisor settled an abandoned profile worker ({phase})',
                cleanup=remote_cleanup)
        except (ProfileBudgetError, OSError, ValueError) as error:
            self.status['profile_settlement_error'] = f'{type(error).__name__}: {error}'

    def _finish(self):
        body = json.loads(self.response['content'][0]['text'])
        body.update(invocation_id=self.invocation_id, tool_call_dir=str(self.directory),
                    **self.context)
        if self.host_budget:
            self.status['host_task'] = self.host_budget.snapshot()
            body['host_task'] = self.status['host_task']
        if self.profile_context:
            from ..execute.profile_budget import ProfileBudget, ProfileBudgetError
            try:
                self.status['profiles'] = ProfileBudget(self.run).snapshot()
                body['profiles'] = self.status['profiles']
            except ProfileBudgetError as error:
                self.status['profiles_error'] = f'{error.code}: {error.detail}'
        if self.repair_policy:
            if 'budget' not in self.status:
                self.status['budget'] = self._repair_snapshot()
            body.update(task_budget=self.repair_policy.as_dict(), budget=self.status['budget'],
                        partial_artifacts=partial_artifacts(self.run))
            reference = read_json(self.directory/'repair-evidence.json') or {}
            if reference.get('invocation_id') == self.invocation_id:
                body['partial_artifacts']['repair_evidence'] = reference.get('evidence')
            if str(self.status.get('stopped_reason') or '').startswith(('budget_', 'verification_budget_')):
                body.update(passed=False, stopped_reason=self.status['stopped_reason'])
            if self.status['phase'] in ('cancelled', 'failed'):
                body['passed'] = False
            self.status['partial_artifacts'] = dict(body['partial_artifacts'])
            body['cleanup'] = self.status.get('cleanup')
        if self.status['phase'] in ('cancelled', 'failed'):
            body.update(cleanup=self.status.get('cleanup'))
            body.setdefault('partial_artifacts', partial_artifacts(self.run))
        if self.journal.mode == 'refused':
            # 账本写不进去是**这一条调用**的异常，不静默：调用方要能看出事后查不到它。
            body['task_journal'] = dict(mode='refused', error=self.journal.error)
        self.response['content'][0]['text'] = json.dumps(body, ensure_ascii=False, sort_keys=True)
        publish_json(self.directory/'result.json', self.response)
        self.journal.finished(self.receipt, cleanup=self.status.get('cleanup'),
                              stopped_reason=self.status.get('stopped_reason') or '',
                              finished_at=self.status.get('finished_at'))

    def close(self):
        """Shutdown path: request first, then acknowledge with the same owner."""
        if self.response is None:
            self.cancel()
            self.poll()


def run_tool_call(name, arguments):
    """CLI adapter: Ctrl-C/SIGTERM request cancellation and wait for local cleanup."""
    interrupted = False
    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
    previous = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, on_signal)
    except ValueError:  # library call from a non-main thread
        pass
    call = None
    try:
        call = ToolCall(name, arguments)
        while True:
            try:
                if interrupted:
                    call.cancel()
                    interrupted = False
                response = call.poll()
                if response is not None:
                    return json.loads(response['content'][0]['text'])
                time.sleep(.05)
            except KeyboardInterrupt:
                call.cancel()
    finally:
        try:
            if call is not None:
                call.close()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    directory = Path((sys.argv[1:] if argv is None else argv)[0]).resolve()
    launch = read_json(directory/'launch.json') or {}
    request = read_json(directory/'request.json') or {}
    if request.get('name') == 'operator_verify':
        import operator_agent.app.tool_calls as canonical
        context = request.get('host_budget_context')
        canonical._HOST_CONTEXT = dict(context) if isinstance(context, dict) else None
    if request.get('name') == 'operator_profile':
        import operator_agent.app.tool_calls as canonical
        context = request.get('profile_context')
        canonical._PROFILE_CONTEXT = dict(context) if isinstance(context, dict) else None
    if request.get('name') == 'operator_repair':
        # -m executes this file as __main__; publish to the canonical module used
        # by hitl_tools, rather than accepting context via public tool arguments.
        import operator_agent.app.tool_calls as canonical
        context = request.get('repair_budget')
        canonical._REPAIR_CONTEXT = dict(context) if isinstance(context, dict) else None
    from ..app.mcp_server import handle_message
    # Any noisy library output belongs to execution.log, never the protocol pipe.
    sys.stdout = sys.stderr
    from contextlib import nullcontext
    from ..execute.backend_ownership import owned_persistent_ssh, owned_worker_scope
    graceful = owned_persistent_ssh(request.get('arguments') or {},
                                  persistent_default=request.get('name') == 'operator_bench')
    with (owned_worker_scope(directory, launch['invocation_id']) if graceful else nullcontext()):
        response = handle_message(dict(jsonrpc='2.0', id=1, method='tools/call', params=request))
        arguments = request.get('arguments') or {}
        context = tool_context(arguments.get('backend'), model_mode=(
            'live' if request.get('name') == 'operator_repair' else 'not_used'))
        if context:
            content = response['result']['content'][0]
            content['text'] = json.dumps(dict(json.loads(content['text']), **context),
                                         ensure_ascii=False, sort_keys=True)
        publish_json(directory/'worker-result.json', dict(invocation_id=launch['invocation_id'],
            finished_monotonic=monotonic(), response=response['result']))
        return 0


if __name__ == '__main__':
    sys.exit(main())
