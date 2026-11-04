"""Independent, finite baseline Agent for the offline phase-one rehearsal.

The baseline writes its own candidate and calls normal foreground verification;
it never delegates generation/repair to Conductor. One trusted worker executes
serially. The private backend scope preserves scripted fault state across its
verify calls without changing the public foreground-tool contract.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from operator_agent.app import hitl_tools as ht
from operator_agent.app.autonomous import _atomic_json
from operator_agent.core.contract import freeze_from_task
from operator_agent.core.run_contract import code_hash
from operator_agent.execute import s5_shared_budget as shared
from operator_agent.execute.budget import BudgetLedger, BudgetPlan
from operator_agent.execute.fake_backend import FakeBackend, FakeConfig, FakeLLM, execution_context
from operator_agent.execute.s5_inner_budget import SharedProductionBudget
from operator_agent.execute.task_budget import TaskBudgetConfig, snapshot
from operator_agent.execute.task_clock import monotonic


@contextmanager
def _backend_scope(backend):
    original = ht._backend

    def selected(name, **kwargs):
        return backend if name == 'fake' else original(name, **kwargs)

    ht._backend = selected
    try:
        yield
    finally:
        ht._backend = original


def _source(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                record='artifact')


def _persist_verification(run, ledger, total):
    value = ledger.as_dict()
    caps = ledger.caps
    value.update(unit='candidate_verification_dispatch_per_input_group',
        reservation_policy='pre_dispatch_non_refundable', raw_gpu_launch_cap=False,
        gpu_seconds_cap=False, remaining=dict(
            non_confirm=max(0, total-caps['confirm']-ledger.spent_all),
            confirm=max(0, caps['confirm']-ledger.spent['confirm'])))
    _atomic_json(run/'verification-budget.json', value)


def _configuration(config):
    if not isinstance(config, dict) or config.get('schema') != 'phase1-cell-config-v1':
        raise ValueError('invalid phase-one cell configuration')
    if config.get('arm') != 'A':
        raise ValueError('baseline worker accepts arm A only')
    fake = FakeConfig.from_dict(config.get('fake'))
    if config.get('backend') != 'fake' or fake.llm != 'fake':
        raise ValueError('baseline rehearsal supports explicit fake backend/model only')
    from operator_agent.execute.s5_verify_client import enabled
    if enabled():
        raise ValueError('offline baseline refuses inherited S5 verification delegation')
    if any(value is not None and value is not False for value in config.get('connection', {}).values()):
        raise ValueError('offline baseline does not accept remote connection options')
    policy = TaskBudgetConfig.from_dict(config.get('budget'))
    rounds = config.get('max_rounds', 2)
    if type(rounds) is not int or rounds <= 0:
        raise ValueError('max_rounds must be a positive integer')
    for name in ('started_monotonic', 'deadline_monotonic'):
        value = config.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid ' + name)
    if config['deadline_monotonic'] > config['started_monotonic'] + policy.wall_clock_s + .001:
        raise ValueError('baseline deadline exceeds the common wall-clock budget')
    return fake, policy, rounds


def run(config):
    """Run a fresh baseline cell; the parent owns process lifetime and cleanup."""
    run_dir = Path(config['run_dir']).resolve()
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Refuse reuse rather than replacing a previous trial's terminal evidence.
    with (run_dir/'baseline-started.json').open('x', encoding='utf-8') as handle:
        json.dump(dict(cell_id=config.get('cell_id'), started_at=time.time()), handle)
    events = []
    result = dict(schema='phase1-baseline-result-v1', cell_id=config.get('cell_id'),
        arm='A', plan_hash=config.get('plan_hash'), phase='failed', levels={},
        all_passed=False, selected_source=None, source_hash=None, contract_hash=None,
        reason=None, verification_count=0, task_budget=config.get('budget'), budget=None,
        execution_context=execution_context(FakeConfig()), error=None)
    policy = ledger = None

    def event(kind, evidence, *, stage='check', **fields):
        row = dict(id='baseline-%d' % (len(events)+1), timestamp=time.time(), stage=stage,
            actor='agent', kind=kind, action='candidate_verify', tool='baseline.verify',
            source=_source(evidence), **fields)
        events.append(row)
        _atomic_json(run_dir/'events.json', events)

    try:
        fake, policy, rounds = _configuration(config)
        result['execution_context'] = execution_context(fake)
        result['task_budget'] = policy.as_dict()
        if monotonic() >= config['deadline_monotonic']:
            result.update(phase='done', reason='budget_wall_clock')
            return result
        prepared = ht.prepare(config['declaration'], run_dir)
        result['contract_hash'] = prepared['contract_hash']
        task = ht.build_task(config['declaration'])
        input_groups = len(freeze_from_task(task).cases)
        ledger = BudgetLedger(BudgetPlan(oracle_total=policy.oracle_total,
                                       wall_clock_s=policy.wall_clock_s))
        _persist_verification(run_dir, ledger, policy.oracle_total)
        call_budget = SharedProductionBudget(policy.llm_limits(), run_dir/'model-journal',
            shared_path=config['shared_ledger'], task_id=config['shared_task_id'],
            deadline=config['deadline_monotonic'])
        llm = FakeLLM(task.source.kernel_type, call_budget=call_budget)
        backend = FakeBackend(fake, event_path=run_dir/'fake-events.jsonl')
        messages = [dict(role='system', content=(
            'Write only CPU Python/NumPy source defining kernel with the declared operands. '
            'After verification feedback revise the source yourself.')),
            dict(role='user', content=json.dumps(config['declaration'], ensure_ascii=False,
                                               sort_keys=True))]
        with _backend_scope(backend):
            for number in range(1, rounds+1):
                if monotonic() >= config['deadline_monotonic']:
                    result.update(phase='done', reason='budget_wall_clock')
                    break
                audit = shared.snapshot(config['shared_ledger'], expected_task_id=config['shared_task_id'])
                if audit['remaining_calls'] <= 0:
                    result.update(phase='done', reason='budget_llm_calls')
                    break
                phase = 'explore' if number == 1 else 'refine'
                if not ledger.oracle_may(phase, input_groups):
                    ledger.oracle_refuse(phase)
                    _persist_verification(run_dir, ledger, policy.oracle_total)
                    result.update(phase='done', reason='verification_budget_exhausted')
                    break
                code = llm.chat(messages, caller_tag='Coder' if number == 1 else 'Fixer')
                candidate = run_dir/('candidate-%d.py' % number)
                with candidate.open('x', encoding='utf-8') as handle:
                    handle.write(code)
                result.update(selected_source=str(candidate), source_hash=code_hash(code))
                # One development verification dispatch is one input-group unit,
                # matching production repair's pre-dispatch non-refundable gate.
                ledger.oracle_spend(phase, input_groups)
                _persist_verification(run_dir, ledger, policy.oracle_total)
                event('tool_call', candidate)
                result['verification_count'] += 1
                verified = ht.verify(run_dir, code, backend='fake', candidate_origin='outer')
                result['levels'] = dict(verified['levels'])
                event('tool_result', Path(verified['evidence']),
                      status='completed', ok=bool(verified['passed']), passed=verified['passed'])
                if verified['levels'].get('L1') and verified['levels'].get('L2'):
                    # The scripted performance refusal is terminal test evidence,
                    # not a reason to spend every remaining model reservation.
                    result.update(phase='done', reason='offline_correctness_complete')
                    break
                messages.extend([dict(role='assistant', content=code),
                    dict(role='user', content=json.dumps(dict(levels=verified['levels'],
                        verdicts=verified['verdicts']), ensure_ascii=False, sort_keys=True))])
                if number < rounds:
                    event('retry', Path(verified['evidence']), stage='modify',
                          reason='verification_failed')
            else:
                result.update(phase='done', reason='round_limit')
    except Exception as exc:
        result.update(phase='failed', reason='baseline_failed',
                      error=dict(type=type(exc).__name__, detail=str(exc)))
    finally:
        if policy is not None:
            result['budget'] = snapshot(run_dir, policy, config['started_monotonic'])
        _atomic_json(run_dir/'events.json', events)
        _atomic_json(run_dir/'baseline-result.json', result)
    return result


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit('usage: python -m benchmarks.experiments.phase1.baseline CELL_CONFIG')
    result = run(json.loads(Path(arguments[0]).read_text(encoding='utf-8')))
    print(json.dumps(dict(phase=result['phase'], reason=result['reason'],
                         cell_id=result['cell_id']), ensure_ascii=False, sort_keys=True))
    return 0 if result['phase'] == 'done' else 1


if __name__ == '__main__':
    raise SystemExit(main())
