"""One generated parent, two independent bounded repair arms, no hidden calls.

Planning is pure. Real execution requires the same frozen SSH/model identity as
ordinary quality runs, and preflight precedes all model construction. Replay is
CPU plumbing evidence, never a model/GPU result.
"""
from __future__ import annotations

import contextlib
import copy
import io
import os
from pathlib import Path
import time

from ..decide import quality_evaluation as q
from ..decide import quality_suite as suite
from ..execute.llm_budget import LLMCallBudget
from ..core.run_contract import atomic_json, code_hash, digest, json_value
from ..core.run_layout import ROLE_CANDIDATE, run_layout


ARMS = ('baseline', 'diagnosis')
PAIRED_PLAN_SCHEMAS = ('operator-paired-quality-plan-v1', 'operator-seeded-paired-quality-plan-v1')


def _pairing(budget, *, seeded=False):
    if seeded:
        return dict(generation_calls=0, repair_calls_per_arm=budget.max_calls,
                    arm_order='alternating_by_case',
                    cost_scope='seeded_initial_source_plus_repair_only_arms')
    if budget.max_calls < 2:
        raise ValueError('paired quality needs one generation and at least one repair call per arm')
    return dict(generation_calls=1, repair_calls_per_arm=budget.max_calls-1,
                arm_order='alternating_by_case',
                cost_scope='shared_generation_once_plus_repair_only_arms')


def make_paired_plan(**kwargs):
    """Freeze a single model and equal arm budgets; do not create any inputs."""
    if 'strategy' in kwargs:
        raise ValueError('paired quality fixes both strategies')
    base = q.make_plan(strategy='baseline', **kwargs)
    body = dict(schema='operator-paired-quality-plan-v1', base_plan=base,
                pairing=_pairing(q.QualityBudget(**base['controls']['budget'])))
    return dict(body, hash=digest(body))


def _validate_initial_sources(sources, base):
    ids = [row['case_id'] for row in base['manifest']['cases']]
    if not isinstance(sources, dict) or set(sources) != set(ids):
        raise ValueError('seed sources must exactly cover the frozen cases')
    for seed in sources.values():
        if (not isinstance(seed, dict) or set(seed) != {'source', 'sha256'}
                or not isinstance(seed['source'], str) or not seed['source'].strip()
                or len(seed['source'].encode('utf-8')) > 32768
                or seed['sha256'] != code_hash(seed['source'])):
            raise ValueError('seed source identity/size drift')


def make_seeded_paired_plan(*, initial_sources, **kwargs):
    """Freeze caller-supplied sources for fresh repair, granting no old credit.

    The default is one repair request and two candidate verifications per arm.
    Sources are plan data, never executable planning inputs or provider replies.
    """
    if 'strategy' in kwargs:
        raise ValueError('paired quality fixes both strategies')
    if kwargs.get('budget') is None:
        kwargs['budget'] = q.QualityBudget(max_calls=1)
    base = q.make_plan(strategy='baseline', **kwargs)
    if not isinstance(initial_sources, dict):
        raise ValueError('seed sources must be a case-to-source mapping')
    sources = {key: dict(source=value, sha256=code_hash(value) if isinstance(value, str) else None)
               for key, value in initial_sources.items()}
    _validate_initial_sources(sources, base)
    body = dict(schema='operator-seeded-paired-quality-plan-v1', base_plan=base,
                pairing=_pairing(q.QualityBudget(**base['controls']['budget']), seeded=True),
                initial_sources=sources)
    return dict(body, hash=digest(body))


def validate_paired_plan(plan):
    seeded = isinstance(plan, dict) and plan.get('schema') == 'operator-seeded-paired-quality-plan-v1'
    expected = {'schema', 'base_plan', 'pairing', 'hash'} | ({'initial_sources'} if seeded else set())
    if (not isinstance(plan, dict)
            or set(plan) != expected or plan.get('schema') not in PAIRED_PLAN_SCHEMAS):
        raise ValueError('invalid paired quality plan')
    if digest({k: v for k, v in plan.items() if k != 'hash'}) != plan['hash']:
        raise ValueError('paired quality plan hash drift')
    base = q.validate_plan(plan['base_plan'])
    if (base['controls']['strategy'] != 'baseline'
            or plan['pairing'] != _pairing(q.QualityBudget(**base['controls']['budget']), seeded=seeded)):
        raise ValueError('paired quality control drift')
    if seeded:
        _validate_initial_sources(plan['initial_sources'], base)
    return plan


def _costs(records):
    for row in records:
        for name in ('model_calls', 'tokens'):
            if row.get(name) is not None:
                suite._count(row[name], name)
        if row.get('wall_seconds') is not None:
            suite._seconds(row['wall_seconds'])
    return {name: dict(known_total=sum(row[name] for row in records if row.get(name) is not None),
                       missing_records=sum(row.get(name) is None for row in records))
            for name in ('model_calls', 'tokens', 'wall_seconds')}


def _absent(case_id, status, replay):
    return dict(case_id=case_id, attempt=1, status=status, full_l2_passed=None,
                performance_target_passed=None, measurement_validity=None,
                execution_kind='replay' if replay else 'real_device',
                model_calls=0, tokens=0, wall_seconds=0., inherited_shared_failure=True)


def _write_parent(path, source):
    # Use the repository's single role-aware immutable artifact writer. The
    # public audit path is an exclusive hard link to that same read-only inode.
    layout = run_layout(str(path.parent / 'source-layout'))
    artifact = layout.write_for(ROLE_CANDIDATE, 'candidate/initial.py', source)
    os.link(artifact, path)


def _build_report(plan, shared_tasks, arm_records):
    base = plan['base_plan']
    seeded = plan['schema'] == 'operator-seeded-paired-quality-plan-v1'
    declared = base['manifest']['cases']
    if [row['case_id'] for row in shared_tasks] != [row['case_id'] for row in declared]:
        raise ValueError('paired task history drift')
    for shared, declaration in zip(shared_tasks, declared):
        if shared['input_manifest_hash'] != digest(declaration['evaluation_inputs']):
            raise ValueError('paired input manifest drift')
        if seeded:
            if (type(shared.get('model_calls')) is not int or shared['model_calls'] != 0
                    or type(shared.get('tokens')) is not int or shared['tokens'] != 0):
                raise ValueError('seeded initialization cannot claim generation costs')
            if (shared['status'] not in ('seeded', 'preflight_failed', 'initialization_failed')
                    or (shared['status'] == 'seeded' and shared.get('source_hash')
                        != plan['initial_sources'][shared['case_id']]['sha256'])
                    or (shared['status'] != 'seeded' and shared.get('source_hash') is not None)):
                raise ValueError('seeded initial source evidence drift')
    parent_identity = digest([dict(case_id=row['case_id'], source_hash=row['source_hash'],
                                  input_manifest_hash=row['input_manifest_hash'],
                                  device_identity_hash=row['device_identity_hash']) for row in shared_tasks])
    reports = {}
    for arm in ARMS:
        controls = copy.deepcopy(base['controls'])
        controls.update(strategy=arm, pairing=dict(plan['pairing'], initial_candidates_hash=parent_identity))
        reports[arm] = suite.summarize(arm_records[arm], manifest=base['manifest'], controls=controls)
    pairs = []
    metrics = dict(eligible_failed_pairs=0, baseline_repairs=0, diagnosis_repairs=0,
                   diagnosis_only_repairs=0, baseline_only_repairs=0,
                   initial_observation_conflicts=0)
    for number, shared in enumerate(shared_tasks):
        key = shared['case_id']
        rows = {arm: [row for row in arm_records[arm] if row['case_id'] == key] for arm in ARMS}
        first = {arm: rows[arm][0] for arm in ARMS}
        if seeded:
            for arm in ARMS:
                actual = first[arm].get('source_hash')
                if actual is not None and actual != shared['source_hash']:
                    raise ValueError('seeded arm initial source drift')
                if shared['source_hash'] is None and (
                        len(rows[arm]) != 1 or first[arm].get('inherited_shared_failure') is not True
                        or first[arm].get('status') != shared['status']
                        or first[arm].get('model_calls') != 0):
                    raise ValueError('seeded shared failure cannot inherit arm results')
        initial = {arm: first[arm].get('full_l2_passed') for arm in ARMS}
        source_matches = (shared['source_hash'] is not None
                          and all(first[arm].get('source_hash') == shared['source_hash'] for arm in ARMS))
        conflict = bool(source_matches and initial['baseline'] != initial['diagnosis'])
        opportunity = bool(source_matches and not conflict
                           and all(initial[arm] is False
                                   and first[arm].get('repair_basis') == 'correctness_failure'
                                   and first[arm].get('repair_conflict') is False for arm in ARMS))
        final = {arm: rows[arm][-1].get('full_l2_passed') for arm in ARMS}
        successes = {arm: opportunity and final[arm] is True for arm in ARMS}
        metrics['eligible_failed_pairs'] += opportunity
        metrics['initial_observation_conflicts'] += conflict
        for arm in ARMS:
            metrics[arm+'_repairs'] += successes[arm]
        metrics['diagnosis_only_repairs'] += successes['diagnosis'] and not successes['baseline']
        metrics['baseline_only_repairs'] += successes['baseline'] and not successes['diagnosis']
        pairs.append(dict(case_id=key, initial_source_hash=shared['source_hash'],
            input_manifest_hash=shared['input_manifest_hash'], source_hashes_match=source_matches,
            arm_order=list(ARMS if number % 2 == 0 else reversed(ARMS)),
            initial_full_l2=initial, initial_observation_conflict=conflict,
            repair_opportunity=opportunity, baseline_final_full_l2=final['baseline'],
            diagnosis_final_full_l2=final['diagnosis']))
    all_records = shared_tasks + arm_records['baseline'] + arm_records['diagnosis']
    schema = 'operator-seeded-paired-quality-report-v1' if seeded else 'operator-paired-quality-report-v1'
    return dict(schema=schema, plan=plan, plan_hash=plan['hash'],
        shared=dict(tasks=shared_tasks, costs=_costs(shared_tasks)), arms=reports, pairs=pairs,
        metrics=metrics, comparison=suite.compare_reports(reports['baseline'], reports['diagnosis']),
        total_actual_costs=_costs(all_records),
        scope=('L2 only; caller-supplied fixed sources; no generation cost or model first-generation quality claim; '
               if seeded else 'L2 only; arm costs are repair-only; shared generation counted once; ')
              +
              'paired repair metrics exclude conflicting/unknown initial evidence; '
              'replay does not measure model or GPU quality')


def validate_paired_report(report):
    """Recompute aggregates and parent bindings from retained audit records."""
    if not isinstance(report, dict) or report.get('schema') not in (
            'operator-paired-quality-report-v1', 'operator-seeded-paired-quality-report-v1'):
        raise ValueError('invalid paired quality report')
    validate_paired_plan(report['plan'])
    rows = report['shared']['tasks']
    ids = [row['case_id'] for row in report['plan']['base_plan']['manifest']['cases']]
    if [row['case_id'] for row in rows] != ids:
        raise ValueError('paired task history drift')
    rebuilt = _build_report(report['plan'], rows,
                            {arm: report['arms'][arm]['attempts'] for arm in ARMS})
    if json_value(rebuilt) != json_value(report):
        raise ValueError('paired report does not match retained evidence')
    return report


def run_paired_plan(plan, output_dir, *, responses=None, host='', client_factory=None, backend_factory=None):
    """Generate once per task, then independently verify/repair the same parent."""
    validate_paired_plan(plan)
    base, replay = plan['base_plan'], responses is not None
    seeded = plan['schema'] == 'operator-seeded-paired-quality-plan-v1'
    q._validate_execution(base, replay, host)
    budget = q.QualityBudget(**base['controls']['budget'])
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    atomic_json(root / 'plan.json', plan, readonly=True)
    (root / 'shared').mkdir()
    for arm in ARMS:
        (root / 'arms' / arm).mkdir(parents=True)
    shared_tasks, arm_records = [], {arm: [] for arm in ARMS}
    for number, declared in enumerate(base['manifest']['cases']):
        case_id = declared['case_id']
        shared_root = root / 'shared' / case_id
        shared_root.mkdir()
        started, backend, client, ledger = time.monotonic(), None, None, None
        source, preflight, log = None, None, io.StringIO()
        phase = ('initialization' if seeded else 'generation') if replay else 'preflight'
        available = responses.get(case_id, {}) if replay else {}
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                backend = q._make_backend(case_id, replay, host, backend_factory)
                if not replay:
                    preflight = q._preflight(backend, shared_root)
                phase = 'initialization' if seeded else 'generation'
                if seeded:
                    source = plan['initial_sources'][case_id]['source']
                    if not replay:
                        q.precheck(source, q.input_metadata(list(suite.evaluation_inputs_for(case_id))))
                else:
                    ledger = LLMCallBudget(q._limits(budget, max_calls=1,
                        allowed_callers=('QualityGenerate',)), shared_root / 'model-journal')
                    factory = ((lambda key, book: client_factory(key, 'generation', book))
                               if client_factory else None)
                    client = q._new_client(case_id, ledger, budget, replay,
                                           available.get('generation', []), factory)
                    task = suite.task_for(case_id, hardware='mock_gpu' if replay else 'nvidia_gpu',
                                          dsl='python' if replay else 'triton')
                    source = q._generate(case_id, task, client, shared_root, budget, replay)
                _write_parent(shared_root / 'candidate.py', source)
        except Exception as error:
            source = None
            atomic_json(shared_root / 'failure.json', dict(error_type=type(error).__name__))
        finally:
            reserved = ledger.summary()['reserved_calls'] if ledger else 0
            calls, tokens, complete = q._usage(client) if client else (reserved, 0, reserved == 0)
            shared = dict(case_id=case_id, status=('seeded' if seeded else 'generated') if source is not None else phase+'_failed',
                source_hash=code_hash(source) if source is not None else None,
                input_manifest_hash=digest(declared['evaluation_inputs']),
                device_identity_hash=digest(preflight['identity']) if preflight else None,
                preflight_hash=preflight.get('hash') if preflight else None,
                model_calls=calls, tokens=tokens if complete else None,
                token_usage_sources=q._usage_sources(client) if client else [],
                wall_seconds=time.monotonic()-started)
            shared_tasks.append(shared)
            if client:
                atomic_json(shared_root / 'usage.json', client.cost_tracker.summary())
            (shared_root / 'execution.log').write_text(log.getvalue())
            atomic_json(shared_root / 'summary.json', shared)
        try:
            for arm in (ARMS if number % 2 == 0 else reversed(ARMS)):
                arm_root = root / 'arms' / arm / case_id
                arm_root.mkdir()
                if source is None:
                    arm_records[arm].append(_absent(case_id, shared['status'], replay))
                    continue
                # Re-read immutable audit bytes and rebuild layouts, never copy
                # the mutable inputs left by the first arm.
                parent = (shared_root / 'candidate.py').read_bytes().decode('utf-8')
                if code_hash(parent) != shared['source_hash']:
                    raise ValueError('paired parent source drift')
                task = suite.task_for(case_id, hardware='mock_gpu' if replay else 'nvidia_gpu',
                                      dsl='python' if replay else 'triton')
                inputs = list(suite.evaluation_inputs_for(case_id))
                arm_started, recorder, client, ledger = time.monotonic(), None, None, None
                log = io.StringIO()
                try:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        ledger = LLMCallBudget(q._limits(budget,
                            max_calls=plan['pairing']['repair_calls_per_arm'],
                            allowed_callers=('Fixer',)), arm_root / 'model-journal')
                        factory = ((lambda key, book: client_factory(key, arm, book))
                                   if client_factory else None)
                        client = q._new_client(case_id, ledger, budget, replay,
                                               available.get(arm, []), factory)
                        recorder = q.RecordingVerifier(q.Verifier(backend), client, arm_root,
                            case_id, budget, 'replay' if replay else 'real_device', started=arm_started)
                        q._repair_candidate(parent, task, inputs, recorder, arm, budget)
                except Exception as error:
                    atomic_json(arm_root / 'failure.json', dict(error_type=type(error).__name__))
                finally:
                    records = q._finish_records(recorder, client, ledger, case_id,
                                                arm_started, 'verification', replay)
                    if records[0].get('source_hash') is not None:
                        actual = code_hash((arm_root / 'candidate-1.py').read_bytes().decode('utf-8'))
                        if actual != shared['source_hash'] or actual != records[0]['source_hash']:
                            raise ValueError('paired initial candidate evidence drift')
                    arm_records[arm].extend(records)
                    if client:
                        atomic_json(arm_root / 'usage.json', client.cost_tracker.summary())
                    (arm_root / 'execution.log').write_text(log.getvalue())
        finally:
            q._close_backend(backend, shared_root)
        atomic_json(root / 'shared-tasks.json', shared_tasks)
        for arm in ARMS:
            atomic_json(root / 'arms' / arm / 'records.json', arm_records[arm])
    report = _build_report(plan, shared_tasks, arm_records)
    for arm in ARMS:
        atomic_json(root / 'arms' / arm / 'report.json', report['arms'][arm])
    atomic_json(root / 'report.json', report)
    return report
