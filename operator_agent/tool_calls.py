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
from .task_clock import monotonic
import uuid

from .autonomous import _atomic_json
from . import exec_guard
from .task_cancellation import (cancellation, launch_identity, partial_artifacts,
                                publish_json, read_json, request_cancel, stop_event)

LONG_TOOLS = frozenset(('operator_verify', 'operator_repair', 'operator_bench'))
_REPAIR_CONTEXT = None


def repair_context(run):
    """Trusted worker context; never read a user-supplied deadline argument."""
    if _REPAIR_CONTEXT and Path(run).resolve() == Path(_REPAIR_CONTEXT['run_dir']):
        return dict(_REPAIR_CONTEXT)
    return None


def _error(error, detail, **fields):
    from .mcp_server import _text_result
    return _text_result(dict(ok=False, error=error, detail=detail, **fields), is_error=True)


def _start_call_worker(directory):
    return exec_guard.start_task_worker([sys.executable, '-m', 'operator_agent.tool_calls',
        str(directory)], cwd=str(Path(__file__).resolve().parents[1]),
        log_path=directory/'execution.log')


class ToolCall:
    """One parent-owned Popen, immutable identity, cancel marker and final receipt."""
    def __init__(self, name, arguments):
        from .hitl_tools import ToolError
        if name not in LONG_TOOLS:
            raise ToolError('unknown_tool', str(name))
        if not isinstance(arguments, dict) or not arguments.get('run_dir'):
            raise ToolError('run_dir_required', f'{name} requires run_dir')
        self.run = Path(arguments['run_dir']).resolve()
        from .backend_ownership import owned_persistent_ssh
        self.graceful_remote = owned_persistent_ssh(
            arguments, persistent_default=name == 'operator_bench')
        self.repair_policy = None
        self.deadline = None
        if name == 'operator_repair':
            from .task_budget import TaskBudgetConfig
            try:
                self.repair_policy = TaskBudgetConfig(**{
                    key: arguments[key] for key in TaskBudgetConfig.__dataclass_fields__
                    if key in arguments})
            except (TypeError, ValueError) as exc:
                raise ToolError('invalid_task_budget', str(exc)) from exc
        if not (self.run/'declaration.json').is_file():
            raise ToolError('run_not_prepared', 'Prepare the run directory first.')
        self.invocation_id = uuid.uuid4().hex
        self.started_monotonic = monotonic()
        if self.repair_policy:
            self.deadline = self.started_monotonic + self.repair_policy.wall_clock_s
        self.directory = self.run/'tool-calls'/self.invocation_id
        self.directory.mkdir(parents=True)
        launch_identity(self.directory, self.invocation_id, kind='tool_call', tool=name)
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
        publish_json(self.directory/'request.json', request)
        self.status = dict(phase='running', stage='starting', kind='tool_call', tool=name,
            pid=os.getpid(), invocation_id=self.invocation_id, started_at=time.time(),
            pid_identity=exec_guard.process_identity(os.getpid()),
            run_dir=str(self.run), tool_call_dir=str(self.directory))
        if self.repair_policy:
            self.status['task_budget'] = self.repair_policy.as_dict()
        self.response = None
        self.process = None
        self.groups = exec_guard.OwnedGroups()
        self._last_discovery = 0.
        self._write_status()
        try:
            self.process = _start_call_worker(self.directory)
            self.groups.add(self.process.pid)
            self.status.update(stage='executing', worker_pid=self.process.pid)
            self._write_status()
        except OSError as exc:
            self.status.update(phase='failed', error='worker_start_failed', detail=str(exc),
                               finished_at=time.time())
            self.response = _error('worker_start_failed', str(exc))
            self._finish()

    def _write_status(self):
        _atomic_json(self.directory/'run-status.json', self.status)

    def cancel(self):
        return request_cancel(self.directory, self.invocation_id)

    def _repair_snapshot(self, now=None):
        from .task_budget import snapshot
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
                    self._write_status()
            return None
        cleanup = (exec_guard.stop_task_worker(self.process, self.groups, graceful_s=12)
                   if reason and self.graceful_remote else
                   exec_guard.stop_task_worker(self.process, self.groups))
        reason, event_at = stop_event(self.directory, self.invocation_id, self.deadline)
        from .backend_ownership import cleanup_state
        cleanup.update(remote_execution=(cleanup_state(self.directory, self.invocation_id)
                                         if self.graceful_remote else 'unknown'),
                       provider_call='unknown')
        result = read_json(self.directory/'worker-result.json') or {}
        finished = result.get('finished_monotonic')
        valid = (result.get('invocation_id') == self.invocation_id
                 and type(finished) in (float, int) and finished <= monotonic()
                 and finished >= self.started_monotonic
                 and isinstance(result.get('response'), dict))
        if valid and (event_at is None or finished <= event_at):
            self.response = result['response']
            # Tool refusal is a completed call, not a cancelled/abandoned result.
            self.status.update(phase='done', stopped_reason='tool_returned')
            if self.repair_policy:
                body = json.loads(self.response['content'][0]['text'])
                self.status['stopped_reason'] = body.get('stopped_reason') or 'tool_returned'
                self.status['budget'] = self._repair_snapshot(now=finished)
        elif reason == 'budget_wall_clock':
            self.status.update(phase='done', stopped_reason=reason)
            self.response = _error('task_budget_exhausted', 'The repair wall-clock budget expired.',
                                  passed=False, phase='done', stopped_reason=reason)
        elif reason:
            self.status.update(phase='cancelled', stopped_reason='user_cancelled')
            self.response = _error('task_cancelled', 'The owning supervisor stopped this call.',
                                   phase='cancelled', stopped_reason='user_cancelled')
        else:
            self.status.update(phase='failed', stopped_reason='worker_failed')
            self.response = _error('worker_failed', 'Worker exited without a terminal result.')
        if self.repair_policy and 'budget' not in self.status:
            self.status['budget'] = self._repair_snapshot()
        elif self.repair_policy and not (valid and (event_at is None or finished <= event_at)):
            self.status['budget'] = self._repair_snapshot()
        self.status.update(cleanup=cleanup, cancellation=cancellation(self.directory, self.invocation_id),
            finished_at=time.time(), partial_artifacts=partial_artifacts(self.run))
        self._finish()
        return self.response

    def _finish(self):
        body = json.loads(self.response['content'][0]['text'])
        body.update(invocation_id=self.invocation_id, tool_call_dir=str(self.directory))
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
        self.response['content'][0]['text'] = json.dumps(body, ensure_ascii=False, sort_keys=True)
        publish_json(self.directory/'result.json', self.response)
        self._write_status()

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
    if request.get('name') == 'operator_repair':
        # -m executes this file as __main__; publish to the canonical module used
        # by hitl_tools, rather than accepting context via public tool arguments.
        import operator_agent.tool_calls as canonical
        context = request.get('repair_budget')
        canonical._REPAIR_CONTEXT = dict(context) if isinstance(context, dict) else None
    from .mcp_server import handle_message
    # Any noisy library output belongs to execution.log, never the protocol pipe.
    sys.stdout = sys.stderr
    from contextlib import nullcontext
    from .backend_ownership import owned_persistent_ssh, owned_worker_scope
    graceful = owned_persistent_ssh(request.get('arguments') or {},
                                  persistent_default=request.get('name') == 'operator_bench')
    with (owned_worker_scope(directory, launch['invocation_id']) if graceful else nullcontext()):
        response = handle_message(dict(jsonrpc='2.0', id=1, method='tools/call', params=request))
        publish_json(directory/'worker-result.json', dict(invocation_id=launch['invocation_id'],
            finished_monotonic=monotonic(), response=response['result']))
        return 0


if __name__ == '__main__':
    sys.exit(main())
