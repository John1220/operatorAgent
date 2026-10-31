"""Explicit best-of-three full-L2 experiment, with a nested first-slot baseline.

All slots run even after a pass. These are three independent generations, never
repair retries. The report retains every slot, including unknown observations,
and accounts for readiness once. It makes no performance claim.
"""
from __future__ import annotations

import ast
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import re
import time

from ..decide import quality_evaluation as q, quality_suite as suite
from ..decide.correctness import as_suite
from ..core.run_contract import atomic_json, audit_value, code_hash, digest
from ..core.schemas import VerificationLevel


PLAN_SCHEMA = 'operator-portfolio-quality-plan-v1'
REPORT_SCHEMA = 'operator-portfolio-quality-report-v1'
SLOTS = ((1, 'layout_first'), (2, 'numerics_first'), (3, 'tiled'))


def _protocol():
    return dict(slots=[dict(slot=n, strategy=s) for n, s in SLOTS],
                generation_calls_per_slot=1, verifications_per_slot=1, repair_calls=0,
                selection='earliest_full_l2_pass', baseline_slot=1,
                cost_scope='all_slots_plus_readiness_once')


def make_portfolio_plan(**options):
    """Freeze the complete domain and three explicit, separately bounded slots."""
    if 'strategy' in options:
        raise ValueError('portfolio fixes its generation strategies; no repair strategy')
    options.setdefault('budget', q.QualityBudget(max_calls=1, max_verifications=1))
    base = q.make_plan(strategy='baseline', **options)
    body = dict(schema=PLAN_SCHEMA, base_plan=base, portfolio=_protocol())
    plan = dict(body, hash=digest(body))
    return validate_portfolio_plan(plan)


def validate_portfolio_plan(plan):
    if (not isinstance(plan, dict) or set(plan) != {'schema', 'base_plan', 'portfolio', 'hash'}
            or plan['schema'] != PLAN_SCHEMA
            or digest(plan['portfolio']) != digest(_protocol())
            or digest({k: v for k, v in plan.items() if k != 'hash'}) != plan['hash']):
        raise ValueError('invalid portfolio plan or protocol drift')
    base = q.validate_plan(plan['base_plan'])
    budget = base['controls']['budget']
    if (base['controls']['strategy'] != 'baseline' or budget['max_calls'] != 1
            or budget['max_verifications'] != 1):
        raise ValueError('portfolio requires one generation and one verification per slot')
    return plan


_RECIPES = {
    'matmul': (
        'Use straightforward explicit FP32 accumulation over K with runtime row and column strides. '
        'Prioritize transparent masked addressing and small portable blocks.',
        'Prioritize FP32 numerical accuracy. Use IEEE input precision for FP32 dot products, '
        'FP32 accumulators, and an explicit final output cast. Choose a different reduction organization.',
        'Use a two-dimensional output tile with a masked tiled K reduction and runtime strides. '
        'Use IEEE dot precision for FP32 inputs and FP32 accumulation; correctness precedes speed.'),
    'layernorm': (
        'Use a simple one-program-per-row centered two-pass normalization: mean then squared centered '
        'deviations. Read both runtime strides and normalize by the logical column count.',
        'Prioritize stable centered variance for large common offsets and constant rows. Consider '
        'subtracting a row anchor before the mean reduction, then center the shifted row. '
        'Keep FP32 arithmetic, population variance, epsilon and the frozen output semantics.',
        'Use a block of rows with independent last-axis reductions, or an explicitly masked multi-chunk '
        'row reduction. Each row must use its own mean and centered population variance; padded '
        'lanes must not contribute to either statistic. Support runtime strides.'),
    'softmax': (
        'Use one program per row and the stable subtract-maximum, exp, sum, divide recipe. '
        'Use runtime row and column strides and mask padded logits to negative infinity.',
        'Prioritize stable FP32 exponentiation and reduction for extreme finite logits and constants. '
        'Subtract each row maximum and exclude padded lanes from max and sum. '
        'Use a different reduction organization while preserving the frozen reference semantics.',
        'Use a tile of independent rows or a multi-chunk stable row reduction. Compute per-row '
        'maxima and denominators correctly across chunks, with runtime strides and logical masks.')}


def generation_messages(case_id, task, slot):
    """Describe the public correctness domain without seeds or concrete values."""
    if type(slot) is not int or slot not in (1, 2, 3):
        raise ValueError('invalid portfolio slot')
    request = q.generation_request(case_id, task)
    kind = task.source.kernel_type
    request['semantics'] = ({'operation': 'A @ B', 'accumulation': 'fp32'} if kind == 'matmul'
        else {'reduction_axis': -1, 'stable_subtract_max': True} if kind == 'softmax'
        else {'reduction_axis': -1, 'epsilon': 1e-5, 'affine': False, 'variance': 'population'})
    domain = list(suite.input_cases_for(case_id)) + list(as_suite(task.constraints.correctness_suite).cases)
    declarations = [dict(tensors=[asdict(t) for t in row.inputs.tensors],
                         pattern=row.pattern, layout=row.layout) for row in domain]
    request['supported_domain'] = list({digest(row): row for row in declarations}.values())
    request['generation_strategy'] = dict(slot=slot, name=SLOTS[slot-1][1],
                                         instruction=_RECIPES[kind][slot-1])
    messages = q.generation_messages(case_id, task)
    messages[0]['content'] += ('\nThis is one independent portfolio slot. Generate exactly one source '
        'using the assigned strategy where compatible with the frozen contract. Do not specialize '
        'to the primary shape or values: all declared shapes, patterns, layouts and dtypes must pass. '
        'No prior candidate or verification feedback is available. Strategy labels do not prove diversity.')
    messages[1]['content'] = json.dumps(request, ensure_ascii=False)
    return messages


def _ast_hash(source):
    try:
        return code_hash(ast.dump(ast.parse(source), include_attributes=False))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _cost(rows, field):
    return dict(known_total=sum(row[field] for row in rows if row[field] is not None),
                missing_records=sum(row[field] is None for row in rows))


def _aggregates(plan, readiness, candidates):
    tasks = []
    for declared in plan['base_plan']['manifest']['cases']:
        key = declared['case_id']
        rows = [r for r in candidates if r['case_id'] == key]
        selected = next((r for r in rows if r['full_l2_passed'] is True), None)
        verdict = True if selected else (None if any(r['full_l2_passed'] is None for r in rows) else False)
        tasks.append(dict(case_id=key, baseline_full_l2=rows[0]['full_l2_passed'],
            portfolio_full_l2=verdict, selected_slot=selected['slot'] if selected else None,
            selected_source_hash=selected['source_hash'] if selected else None,
            distinct_sources=len({r['source_hash'] for r in rows if r['source_hash'] is not None}),
            distinct_asts=len({r['ast_hash'] for r in rows if r['ast_hash'] is not None})))
    def metric(field):
        n = sum(t[field] is True for t in tasks)
        return dict(count=n, denominator=len(tasks), rate=n/len(tasks),
                    unknown_tasks=sum(t[field] is None for t in tasks))
    metrics = dict(baseline_full_l2=metric('baseline_full_l2'),
        portfolio_full_l2=metric('portfolio_full_l2'),
        rescued_failed_baselines=sum(t['baseline_full_l2'] is False and t['portfolio_full_l2'] is True for t in tasks),
        resolved_unknown_baselines=sum(t['baseline_full_l2'] is None and t['portfolio_full_l2'] is True for t in tasks))
    costs = {field: _cost(readiness + candidates, field) for field in ('model_calls', 'tokens', 'wall_seconds')}
    return dict(tasks=tasks, metrics=metrics, costs=costs, performance=dict(state='not_measured', rate=None))


def _report(plan, readiness, candidates):
    return dict(schema=REPORT_SCHEMA, plan=plan, plan_hash=plan['hash'], readiness=readiness,
                candidates=candidates, **_aggregates(plan, readiness, candidates))


def validate_portfolio_report(report):
    """Validate complete slot history and recompute all selection/accounting.

    This is structural validation. Source and verification files remain the
    evidence for recorded observations; a JSON hash alone is not authenticity.
    """
    expected = {'schema', 'plan', 'plan_hash', 'readiness', 'candidates',
                'tasks', 'metrics', 'costs', 'performance'}
    if not isinstance(report, dict) or set(report) != expected or report['schema'] != REPORT_SCHEMA:
        raise ValueError('invalid portfolio report')
    plan = validate_portfolio_plan(report['plan'])
    if report['plan_hash'] != plan['hash']:
        raise ValueError('portfolio report plan mismatch')
    readiness, candidates = report['readiness'], report['candidates']
    declarations = plan['base_plan']['manifest']['cases']
    if not isinstance(readiness, list) or len(readiness) != len(declarations) or not isinstance(candidates, list):
        raise ValueError('incomplete portfolio readiness')
    if len(candidates) != 3 * len(declarations):
        raise ValueError('incomplete portfolio candidates')
    def sha(value, optional=False):
        if not (optional and value is None) and (not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)):
            raise ValueError('invalid portfolio source/binding hash')
    for index, declared in enumerate(declarations):
        key = declared['case_id']
        ready = readiness[index]
        real = plan['base_plan']['controls']['backend'] == 'ssh'
        if (set(ready) != {'case_id', 'status', 'model_calls', 'tokens', 'wall_seconds',
                           'input_manifest_hash', 'device_identity_hash', 'identity'}
                or ready['case_id'] != key or ready['model_calls'] != 0 or ready['tokens'] != 0
                or ready['input_manifest_hash'] != digest(declared['evaluation_inputs'])
                or ready['status'] not in (('passed', 'failed') if real else ('replay', 'failed'))):
            raise ValueError('portfolio readiness binding drift')
        sha(ready['device_identity_hash'])
        if ready['device_identity_hash'] != digest(ready['identity']):
            raise ValueError('portfolio device identity drift')
        if (real and ready['status'] == 'passed'
                and (not isinstance(ready['identity'], dict) or ready['identity'].get('device') != 'cuda'
                     or not ready['identity'].get('uuid'))):
            raise ValueError('portfolio real device identity missing')
        for offset, (number, strategy) in enumerate(SLOTS):
            row = candidates[3*index+offset]
            fields = {'case_id', 'slot', 'strategy', 'status', 'full_l2_passed', 'source_hash',
                      'ast_hash', 'model_calls', 'tokens', 'wall_seconds', 'input_manifest_hash',
                      'device_identity_hash', 'execution_kind', 'token_usage_sources'}
            if (set(row) not in (fields, fields | {'error_type'})
                    or row['case_id'] != key or type(row['slot']) is not int or row['slot'] != number
                    or row['strategy'] != strategy or row['input_manifest_hash'] != ready['input_manifest_hash']
                    or row['device_identity_hash'] != ready['device_identity_hash']
                    or row['execution_kind'] != ('real_device' if real else 'replay')):
                raise ValueError('portfolio slot/binding drift')
            if row['full_l2_passed'] is not None and type(row['full_l2_passed']) is not bool:
                raise ValueError('invalid portfolio verdict')
            full = row['full_l2_passed']
            sha(row['source_hash'], True)
            sha(row['ast_hash'], True)
            if row['source_hash'] is None and (row['ast_hash'] is not None or full is not None):
                raise ValueError('portfolio observation without source')
            if row['source_hash'] is not None and row['model_calls'] != 1:
                raise ValueError('portfolio source without exactly one generation')
            if full is True and row['ast_hash'] is None:
                raise ValueError('portfolio pass without parseable source')
            if (row['status'] not in ('passed', 'failed', 'unknown', 'generation_failed', 'verification_failed', 'preflight_failed')
                    or (row['status'] == 'passed') is not (full is True)
                    or (row['status'] == 'failed') is not (full is False)
                    or type(row['model_calls']) is not int or not 0 <= row['model_calls'] <= 1
                    or (ready['status'] == 'failed' and (row['model_calls'] != 0 or row['status'] != 'preflight_failed'))):
                raise ValueError('portfolio status/call accounting drift')
    for row in readiness + candidates:
        if type(row['model_calls']) is not int or row['model_calls'] < 0:
            raise ValueError('invalid portfolio call cost')
        if row['tokens'] is not None and (type(row['tokens']) is not int or row['tokens'] < 0):
            raise ValueError('invalid portfolio token cost')
        suite._seconds(row['wall_seconds'])
    derived = _aggregates(plan, readiness, candidates)
    if any(digest(report[key]) != digest(value) for key, value in derived.items()):
        raise ValueError('portfolio selection or accounting drift')
    return report


def merge_portfolio_reports(plan, reports):
    """Combine disjoint frozen task subsets; never rerun or choose between runs.

    Separate OS processes may execute subsets because stdout capture is process
    global. This merge requires exactly one observation per declared task, with
    identical frozen controls and slot protocol. It does not assert GPU isolation.
    """
    validate_portfolio_plan(plan)
    expected = [row['case_id'] for row in plan['base_plan']['manifest']['cases']]
    collected = {}
    for report in reports:
        validate_portfolio_report(report)
        other = report['plan']
        if (other['base_plan']['controls'] != plan['base_plan']['controls']
                or other['portfolio'] != plan['portfolio']):
            raise ValueError('portfolio subset controls differ')
        for ready in report['readiness']:
            key = ready['case_id']
            if key not in expected or key in collected:
                raise ValueError('portfolio subset overlaps or has undeclared task')
            collected[key] = (ready, [r for r in report['candidates'] if r['case_id'] == key])
    if set(collected) != set(expected):
        raise ValueError('portfolio subset coverage incomplete')
    ready = [collected[key][0] for key in expected]
    rows = [row for key in expected for row in collected[key][1]]
    return validate_portfolio_report(_report(plan, ready, rows))


def run_portfolio_plan(plan, output_dir, *, responses=None, host='', client_factory=None, backend_factory=None):
    """Run every slot once. Readiness precedes all paid calls for each task."""
    validate_portfolio_plan(plan)
    replay = responses is not None
    base = plan['base_plan']
    q._validate_execution(base, replay, host)
    if replay and not isinstance(responses, dict):
        raise ValueError('replay requires a case-to-three-responses object')
    if replay:
        allowed = {row['case_id'] for row in base['manifest']['cases']}
        if any(key not in allowed or not isinstance(replies, list) or len(replies) > 3
               or any(not isinstance(reply, str) for reply in replies)
               for key, replies in responses.items()):
            raise ValueError('replay has undeclared tasks, extra slots or a non-list response')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    atomic_json(root / 'plan.json', plan, readonly=True)
    readiness, candidates = [], []
    budget = q.QualityBudget(**base['controls']['budget'])
    for declared in base['manifest']['cases']:
        case_id = declared['case_id']
        ready_root = root / 'readiness' / case_id
        ready_root.mkdir(parents=True)
        task = suite.task_for(case_id, hardware='mock_gpu' if replay else 'nvidia_gpu',
                              dsl='python' if replay else 'triton')
        backend, log = None, io.StringIO()
        started = time.monotonic()
        identity, state = dict(execution='replay'), 'replay'
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                backend = q._make_backend(case_id, replay, host, backend_factory)
                if not replay:
                    preflight = q._preflight(backend, ready_root)
                    identity, state = preflight['identity'], 'passed'
        except Exception as error:
            state, identity = 'failed', dict(execution='unavailable')
            atomic_json(ready_root / 'failure.json', dict(error_type=type(error).__name__))
        ready = dict(case_id=case_id, status=state, model_calls=0, tokens=0,
                     wall_seconds=max(0., time.monotonic()-started),
                     input_manifest_hash=digest(declared['evaluation_inputs']),
                     device_identity_hash=digest(identity), identity=identity)
        readiness.append(ready)
        atomic_json(ready_root / 'summary.json', ready)
        try:
            for number, strategy in SLOTS:
                slot_root = root / 'cases' / case_id / ('slot-%d' % number)
                slot_root.mkdir(parents=True)
                started = time.monotonic()
                client = ledger = recorder = code = None
                phase = 'preflight'
                row = dict(case_id=case_id, slot=number, strategy=strategy, status='preflight_failed',
                    full_l2_passed=None, source_hash=None, ast_hash=None, model_calls=0, tokens=0,
                    wall_seconds=0., input_manifest_hash=ready['input_manifest_hash'],
                    device_identity_hash=ready['device_identity_hash'],
                    execution_kind='replay' if replay else 'real_device')
                row['token_usage_sources'] = []
                try:
                    if state == 'failed':
                        continue
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        phase = 'generation'
                        ledger = q.LLMCallBudget(q._limits(budget, allowed_callers=('QualityGenerate',)), slot_root / 'model-journal')
                        replies = responses.get(case_id, []) if replay else []
                        reply = [replies[number-1]] if isinstance(replies, list) and len(replies) >= number else []
                        factory = ((lambda key, bound: client_factory(key, number, bound)) if client_factory else None)
                        client = q._new_client(case_id, ledger, budget, replay, reply, factory)
                        code = q._generate(case_id, task, client, slot_root, budget, replay,
                                           messages=generation_messages(case_id, task, number), check_source=False)
                        row.update(source_hash=code_hash(code), ast_hash=_ast_hash(code))
                        (slot_root / 'candidate-1.py').write_text(code)
                        if not replay:
                            q.precheck(code, q.input_metadata(list(suite.evaluation_inputs_for(case_id))))
                        recorder = q.RecordingVerifier(q.Verifier(backend), client, slot_root, case_id,
                            budget, row['execution_kind'], started=started)
                        phase = 'verification'
                        recorder.verify(code, task, list(suite.evaluation_inputs_for(case_id)),
                                        stop_after=VerificationLevel.L2_FUNCTION)
                        row.update(full_l2_passed=recorder.records[0]['full_l2_passed'],
                                   status=recorder.records[0]['status'])
                except Exception as error:
                    row.update(status=phase+'_failed', error_type=type(error).__name__)
                    atomic_json(slot_root / 'failure.json', dict(error_type=type(error).__name__, phase=phase))
                    from ..generate.single_candidate_coder import PrecheckRefused
                    if isinstance(error, PrecheckRefused):
                        atomic_json(slot_root / 'precheck-failure.json', dict(reason=error.reason))
                finally:
                    reserved = ledger.summary()['reserved_calls'] if ledger else 0
                    calls, tokens, complete = q._usage(client) if client else (reserved, 0, reserved == 0)
                    row.update(model_calls=calls, tokens=tokens if complete else None,
                               wall_seconds=max(0., time.monotonic()-started))
                    row['token_usage_sources'] = q._usage_sources(client) if client else []
                    candidates.append(row)
                    if client:
                        atomic_json(slot_root / 'usage.json', client.cost_tracker.summary())
                    atomic_json(slot_root / 'summary.json', row)
                    atomic_json(root / 'candidates.json', candidates)
        finally:
            q._close_backend(backend, ready_root)
            (ready_root / 'execution.log').write_text(log.getvalue())
        atomic_json(root / 'progress.json', dict(completed_tasks=len(readiness), total_tasks=len(base['manifest']['cases'])))
    report = _report(plan, readiness, candidates)
    validate_portfolio_report(report)
    atomic_json(root / 'report.json', report)
    return report
