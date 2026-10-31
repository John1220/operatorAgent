"""Freeze and execute an A/C preflight or C-only plugin diagnostic.

This is a separate protocol from the historical ABC workflow. Freezing writes
plans only; execute() is the real model/GPU dispatch boundary and requires an
exact plan hash plus a successful no-model readiness report.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from benchmarks.experiments.s5 import s5_arm_env, s5_development_service, s5_formal_run, s5_gateway_http, s5_gateway_supervisor, s5_plugin_env, s5_plugin_flow, s5_plugin_probe, s5_plugin_rpc, s5_plugin_summary, s5_outcomes, s5_seatbelt, s5_shared_gpu, s5_verify_remote, s5_task_manifest, s5_minimal_tasks
from operator_agent.execute import s5_shared_budget as shared
from operator_agent.app.hitl_tools import prepare
from operator_agent.core.run_contract import digest

SCHEMA = 's5-plugin-only-plan-v1'
C_ONLY_SCHEMA = 's5-plugin-c-diagnostic-plan-v1'
PLUGIN_ENTRY_SCHEMA = 's5-plugin-entry-acceptance-plan-v1'
PLUGIN_REPAIR_SCHEMA = 's5-plugin-repair-acceptance-plan-v1'
STUDY_SCHEMA = 's5-plugin-study-plan-v1'
STUDY_MANIFEST_HASH = '79fb8b03ee0514d67a354e24b78622fb668a7d27e3592ad3d1144ee2dc74bbd2'
STUDY_CASES = tuple(row[0] for row in s5_minimal_tasks.SPECS)
REPAIR_SEED_PATH = Path(__file__).resolve().parents[2] / 'fixtures/s5_layernorm_repair_seed.py'
PLUGIN_REPAIR_INSTRUCTION = (
    'Repair acceptance: candidate.py already contains the starting implementation. '
    'First submit that file UNCHANGED using mcp__operator__operator_verify, backend ssh, '
    'stop_after L2. Wait for its result before editing. Then diagnose the returned failure, '
    'repair candidate.py, and submit the changed file using the same MCP tool. '
    'There are only two development verifications. Do not use check.py. '
    'Return exactly the verified repaired file content, without further edits, inside one Python '
    'code fence. The supervisor independently evaluates it after DSH ends.'
)
RECORD_SCHEMA = 's5-plugin-only-run-v1'
ARMS = ('A', 'C')
C_ONLY_ARMS = ('C',)
PREFLIGHT_CASES = ('matmul-tail-fp16', 'layernorm-tail', 'softmax-extreme')
PLUGIN_ENTRY_CASES = ('layernorm-tail',)
PLUGIN_ENTRY_INSTRUCTION = (
    'Plugin entry acceptance requirement: submit your development candidate using the DSH '
    'MCP tool mcp__operator__operator_verify with the prepared task run directory, backend ssh, '
    'and stop_after L2. Do not submit development candidates through the public check.py entry. '
    'Read the tool result returned in DSH, revise only if needed, and return the final source code. '
    'The supervisor will independently evaluate the final source after this DSH turn.'
)
MODEL = 'deepseek-v4.1-flash-1017'
ENDPOINT_KEY = 'S5_RESPONSES_ENDPOINT'
DEFAULT_ENDPOINT = s5_formal_run.DEFAULT_ENDPOINT
BUDGET = dict(max_calls=24, max_output_tokens=32768,
              total_output_tokens=24 * 32768, wall_clock_s=2400)
DEVELOPMENT = dict(max_development=2, final_reserve=1)
VERIFY_TIMEOUT = 600
STUDY_STREAM_IDLE_TIMEOUT_MS = (s5_formal_run.REQUEST_WALL_SECONDS + 30) * 1000
MAX_PLAN_BYTES = 1024 * 1024


def _write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        handle.write('\n')


def implementation_identity():
    root = Path(__file__).resolve().parents[3]
    paths = ('benchmarks/experiments/s5/s5_plugin_workflow.py', 'benchmarks/experiments/s5/s5_plugin_env.py',
             'benchmarks/experiments/s5/s5_plugin_probe.py', 'benchmarks/experiments/s5/s5_plugin_flow.py',
             'benchmarks/experiments/s5/s5_plugin_summary.py', 'benchmarks/experiments/s5/s5_plugin_rpc.py',
             'benchmarks/experiments/s5/s5_plugin_proxy.py', 'benchmarks/experiments/s5/s5_formal_run.py',
             'benchmarks/experiments/s5/s5_arm_env.py', 'benchmarks/experiments/s5/s5_seatbelt.py',
             'benchmarks/experiments/s5/s5_development_service.py',
             'benchmarks/experiments/s5/s5_development_evidence.py',
             'benchmarks/experiments/s5/s5_verify_remote.py', 'benchmarks/experiments/s5/s5_gateway_http.py',
             'benchmarks/experiments/s5/s5_gateway_policy.py', 'benchmarks/experiments/s5/s5_gateway_supervisor.py',
             'benchmarks/experiments/s5/s5_gateway_transport.py', 'benchmarks/experiments/s5/s5_outcomes.py',
             'benchmarks/experiments/s5/s5_shared_gpu.py', 'benchmarks/experiments/s5/s5_runtime.py',
             'benchmarks/experiments/s5/s5_vertical_headless.py', 'benchmarks/experiments/s5/s5_task_manifest.py',
             'operator_agent/core/run_contract.py', 'operator_agent/execute/s5_shared_budget.py',
             'operator_agent/app/hitl_tools.py', 'operator_agent/execute/s5_verify_client.py',
             'operator_agent/app/mcp_server.py', 'integrations/dsh-operator/index.mjs',
             'integrations/dsh-operator/package.json',
             'benchmarks/fixtures/s5_layernorm_repair_seed.py',
             'benchmarks/experiments/s5/s5_minimal_tasks.py', 'benchmarks/experiments/s5/s5_plugin_study_summary.py',
             'benchmarks/experiments/s5/s5_candidate_extraction.py', 'benchmarks/experiments/s5/s5_study_readiness.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in paths}


def _validate_budget(budget):
    if budget != BUDGET:
        raise ValueError('plugin-only protocol requires the frozen medium/32K budget')


def _case_order(cases, repeats, arms=ARMS):
    return [dict(case_id=case['case_id'], repeat_id=repeat, arm=arm)
            for repeat in range(1, repeats + 1)
            for index, case in enumerate(cases)
            for arm in ((arms if (index + repeat - 1) % 2 == 0 else arms[::-1])
                        if len(arms) > 1 else arms)]


def freeze(output, *, readiness_report, gpu, resource_policy,
           case_ids=PREFLIGHT_CASES, endpoint=None, model=MODEL,
           budget=None, repeats=1, arms=ARMS, mode=None,
           formal_readiness_report=None, exploratory=False):
    """Freeze an A/C preflight, C-only diagnostic or plugin-entry acceptance."""
    from benchmarks.experiments.s5.s5_gateway_transport import _endpoint

    arms = tuple(arms)
    if mode is None:
        mode = 'ac' if arms == ARMS else 'c-only'
    study = mode == 'study'
    if type(exploratory) is not bool or (exploratory and not study):
        raise ValueError('exploratory policy requires study mode')
    if mode not in ('ac', 'c-only', 'plugin-entry', 'plugin-repair', 'study'):
        raise ValueError('unsupported plugin-only plan mode')
    expected_arms = ARMS if mode in ('ac', 'study') else C_ONLY_ARMS
    expected_cases = (STUDY_CASES if study else PLUGIN_ENTRY_CASES
                      if mode in ('plugin-entry', 'plugin-repair') else PREFLIGHT_CASES)
    if arms != expected_arms:
        raise ValueError('plugin-only plan supports A/C or C-only diagnostic arms')
    if model != MODEL or type(repeats) is not int or repeats != (2 if study else 1):
        raise ValueError('plugin-only protocol fixes model and repeat count')
    budget = dict(BUDGET if budget is None else budget)
    _validate_budget(budget)
    if (not isinstance(case_ids, (list, tuple)) or tuple(case_ids) != expected_cases
            or len(set(case_ids)) != len(expected_cases)):
        raise ValueError('plugin-only preflight task order is fixed')
    if (not isinstance(gpu, dict) or set(gpu) != {'index', 'uuid', 'name'}
            or type(gpu.get('index')) is not int or not 0 <= gpu['index'] <= 63
            or not isinstance(gpu.get('uuid'), str) or not isinstance(gpu.get('name'), str)):
        raise ValueError('exact GPU identity is required')
    s5_shared_gpu.validate_policy(resource_policy)
    readiness = s5_plugin_probe.load_readiness(readiness_report)
    endpoint = endpoint or os.environ.get(ENDPOINT_KEY, DEFAULT_ENDPOINT)
    _endpoint(endpoint)

    manifest = s5_minimal_tasks.manifest() if study else s5_task_manifest.manifest()
    if study and manifest['hash'] != STUDY_MANIFEST_HASH:
        raise ValueError('prospective study task manifest drift')
    rows = {row['case_id']: row for row in manifest['tasks']}
    selected = [rows[name] for name in case_ids if name in rows]
    if len(selected) != len(expected_cases):
        raise ValueError('fixed preflight tasks are unavailable')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    cases = []
    for row in selected:
        name = row['case_id']
        declaration = s5_formal_run.declaration(row)
        service_sha = s5_verify_remote.freeze(
            root / 'services' / name, declaration=declaration, backend='h20',
            host=s5_formal_run.GATEWAY_HOST, timeout=VERIFY_TIMEOUT,
            gpu_index=gpu['index'], gpu_uuid=gpu['uuid'], gpu_name=gpu['name'])
        service_plan = s5_verify_remote.load_plan(root / 'services' / name / 'plan.json',
                                                 service_sha)
        contract = s5_formal_run._read(root / 'services' / name / 'task' / 'contract.json')
        case_text = s5_formal_run.task_text(row)
        case = dict(case_id=name, kernel_type=row['kernel_type'],
            shape=list(row['shape']), precision=row['precision'], layout=row['layout'],
            text=case_text, order=list(arms),
            input_semantics=s5_formal_run.check_task_inputs(row, contract),
            service_sha256=service_sha, service_contract=service_plan['contract_hash'])
        if mode == 'plugin-entry':
            case['acceptance_instruction'] = PLUGIN_ENTRY_INSTRUCTION
        if mode == 'plugin-repair':
            seed = REPAIR_SEED_PATH.read_text(encoding='utf-8')
            case.update(acceptance_instruction=PLUGIN_REPAIR_INSTRUCTION,
                        initial_candidate=seed,
                        initial_candidate_sha256=hashlib.sha256(seed.encode()).hexdigest())
        cases.append(case)

    runs = len(cases) * len(arms) * repeats
    is_c_only = mode == 'c-only'
    schema = (STUDY_SCHEMA if study else PLUGIN_REPAIR_SCHEMA if mode == 'plugin-repair'
              else PLUGIN_ENTRY_SCHEMA if mode == 'plugin-entry'
              else C_ONLY_SCHEMA if is_c_only else SCHEMA)
    purpose = {
        'ac': 'three fixed tasks x native DSH/plugin-only DSH; wiring and budget-use preflight',
        'c-only': 'three fixed plugin-path diagnostic tasks; no A/C comparison',
        'plugin-entry': 'one LayerNorm C task requiring DSH operator_verify; acceptance only',
        'plugin-repair': 'one seeded LayerNorm C failure-feedback-repair acceptance; no gain inference',
        'study': 'six prospective tasks x A/C x two repeats; scoped contribution study',
    }[mode]
    body = dict(schema=schema, mode=mode,
        status='frozen_study_candidate' if study else 'frozen_preflight', formal_s5_credit=False,
        counts_toward_denominator=study,
        purpose=purpose,
        manifest_hash=manifest['hash'], arms=list(arms), repeats=repeats,
        model=dict(id=model, protocol='responses', endpoint=endpoint,
                   reasoning_effort='medium', context_window=s5_formal_run.CONTEXT_WINDOW,
                   request_wall_seconds=s5_formal_run.REQUEST_WALL_SECONDS,
                   max_response_bytes=s5_formal_run.RESPONSE_BYTES),
        budget=budget, development=dict(DEVELOPMENT),
        verification=dict(backend='h20', host=s5_formal_run.GATEWAY_HOST,
            gpu=dict(index=gpu['index'], uuid=gpu['uuid'], name=gpu['name']),
            stop_after='L2', timeout=VERIFY_TIMEOUT),
        resource_policy=dict(resource_policy), readiness=readiness,
        cases=cases, planned_runs=runs,
        schedule=_case_order(cases, repeats, arms), service_root=str(root / 'services'),
        implementation=implementation_identity(),
        limits=dict(max_model_reservations=runs * budget['max_calls'],
                    max_output_reservations=runs * budget['total_output_tokens'],
                    max_verifications=runs * 3,
                    stop_after_infrastructure_unknown=True,
                    ordinary_budget_exhaustion_continues=True))
    if study:
        body['model']['stream_idle_timeout_ms'] = STUDY_STREAM_IDLE_TIMEOUT_MS
        report = Path(formal_readiness_report).resolve() if formal_readiness_report is not None else None
        body['formal_readiness'] = (dict(report_path=str(report),
            report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()) if report else None)
        if exploratory:
            from benchmarks.experiments.s5.s5_study_readiness import EXPLORATORY_POLICY, EXPLORATORY_LIMITATIONS
            body.update(execution_policy=EXPLORATORY_POLICY,
                        exploratory_limitations=dict(EXPLORATORY_LIMITATIONS))
    body['hash'] = digest(body)
    _write_once(root / 'plan.json', body)
    return body


def load_plan(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PLAN_BYTES:
        raise ValueError('plugin-only plan missing or too large')
    plan = json.loads(path.read_text(encoding='utf-8'))
    stored = plan.pop('hash', None)
    if not isinstance(stored, str) or digest(plan) != stored:
        raise ValueError('plugin-only plan hash mismatch')
    plan['hash'] = stored
    arms = tuple(plan.get('arms', ()))
    mode = plan.get('mode', 'c-only' if plan.get('schema') == C_ONLY_SCHEMA else 'ac')
    study = mode == 'study'
    repeats = 2 if study else 1
    expected_arms = ARMS if mode in ('ac', 'study') else C_ONLY_ARMS
    expected_cases = (STUDY_CASES if study else PLUGIN_ENTRY_CASES
                      if mode in ('plugin-entry', 'plugin-repair') else PREFLIGHT_CASES)
    expected_schema = {'ac': SCHEMA, 'c-only': C_ONLY_SCHEMA,
                       'plugin-entry': PLUGIN_ENTRY_SCHEMA,
                       'plugin-repair': PLUGIN_REPAIR_SCHEMA, 'study': STUDY_SCHEMA}.get(mode)
    expected_runs = len(expected_cases) * len(arms) * repeats
    if (mode not in ('ac', 'c-only', 'plugin-entry', 'plugin-repair', 'study') or arms != expected_arms
            or plan.get('schema') != expected_schema
            or plan.get('status') != ('frozen_study_candidate' if study else 'frozen_preflight')
            or plan.get('formal_s5_credit') is not False
            or plan.get('counts_toward_denominator') is not study
            or plan.get('implementation') != implementation_identity()
            or [case.get('case_id') for case in plan.get('cases', [])] != list(expected_cases)
            or plan.get('planned_runs') != expected_runs
            or plan.get('repeats') != repeats
            or plan.get('schedule') != _case_order(plan.get('cases', []), repeats, arms)
            or (mode == 'plugin-entry'
                and plan['cases'][0].get('acceptance_instruction') != PLUGIN_ENTRY_INSTRUCTION)
            or (mode == 'plugin-repair' and (
                plan['cases'][0].get('acceptance_instruction') != PLUGIN_REPAIR_INSTRUCTION
                or plan['cases'][0].get('initial_candidate') != REPAIR_SEED_PATH.read_text(encoding='utf-8')
                or plan['cases'][0].get('initial_candidate_sha256') !=
                    hashlib.sha256(REPAIR_SEED_PATH.read_bytes()).hexdigest()))
            or plan.get('budget') != BUDGET
            or plan.get('model', {}).get('id') != MODEL
            or plan.get('model', {}).get('reasoning_effort') != 'medium'
            or plan.get('development') != DEVELOPMENT):
        raise ValueError('plugin-only plan configuration or source drift')
    if study:
        from benchmarks.experiments.s5.s5_study_readiness import EXPLORATORY_POLICY, EXPLORATORY_LIMITATIONS
        if (plan.get('execution_policy', 'formal') not in ('formal', EXPLORATORY_POLICY)
                or (plan.get('execution_policy') == EXPLORATORY_POLICY
                    and plan.get('exploratory_limitations') != EXPLORATORY_LIMITATIONS)):
            raise ValueError('study execution policy drift')
        manifest = s5_minimal_tasks.manifest()
        if manifest['hash'] != STUDY_MANIFEST_HASH or plan.get('manifest_hash') != manifest['hash']:
            raise ValueError('prospective study task manifest drift')
        model_fields = dict(id=MODEL, protocol='responses', reasoning_effort='medium',
            stream_idle_timeout_ms=STUDY_STREAM_IDLE_TIMEOUT_MS,
            context_window=s5_formal_run.CONTEXT_WINDOW,
            request_wall_seconds=s5_formal_run.REQUEST_WALL_SECONDS,
            max_response_bytes=s5_formal_run.RESPONSE_BYTES)
        expected_limits = dict(max_model_reservations=24 * BUDGET['max_calls'],
            max_output_reservations=24 * BUDGET['total_output_tokens'], max_verifications=72,
            stop_after_infrastructure_unknown=True, ordinary_budget_exhaustion_continues=True)
        if (type(plan.get('repeats')) is not int or type(plan.get('planned_runs')) is not int
                or any(plan['model'].get(k) != v for k, v in model_fields.items())
                or plan.get('limits') != expected_limits):
            raise ValueError('study model or aggregate budget drift')
        from benchmarks.experiments.s5.s5_gateway_transport import _endpoint
        _endpoint(plan['model'].get('endpoint'))
        for case, row in zip(plan['cases'], manifest['tasks']):
            for name in ('case_id', 'kernel_type', 'shape', 'precision', 'layout'):
                if case.get(name) != row[name]:
                    raise ValueError('study task specification drift')
            if case.get('text') != s5_formal_run.task_text(row) or case.get('order') != list(ARMS):
                raise ValueError('study task prompt or tool-path drift')
            if any(name in case for name in ('initial_candidate', 'acceptance_instruction')):
                raise ValueError('study task drift: cannot seed candidates or force plugin use')
    readiness = s5_plugin_probe.load_readiness(plan['readiness']['report_path'])
    if readiness != plan['readiness']:
        raise ValueError('plugin-only readiness drift')
    for case in plan['cases']:
        service_path = Path(plan['service_root']) / case['case_id'] / 'plan.json'
        service = s5_verify_remote.load_plan(service_path, case['service_sha256'])
        if study:
            row = next(r for r in manifest['tasks'] if r['case_id'] == case['case_id'])
            contract = s5_formal_run._read(service_path.parent / 'task/contract.json')
            if (service.get('declaration') != s5_formal_run.declaration(row)
                    or service.get('contract_hash') != case.get('service_contract')
                    or s5_formal_run.check_task_inputs(row, contract) != case.get('input_semantics')):
                raise ValueError('study service contract or input identity drift')
            expected_verification = dict(backend='h20', host=s5_formal_run.GATEWAY_HOST,
                gpu=dict(index=service.get('gpu_index'), uuid=service.get('gpu_uuid'),
                         name=service.get('gpu_name')), stop_after='L2', timeout=VERIFY_TIMEOUT)
            if (plan.get('verification') != expected_verification
                    or service.get('backend') != 'h20' or service.get('mode') != 'ssh'
                    or service.get('host') != s5_formal_run.GATEWAY_HOST
                    or service.get('timeout') != VERIFY_TIMEOUT or service.get('stop_after') != 'L2'):
                raise ValueError('study verification backend or GPU identity drift')
    s5_shared_gpu.validate_policy(plan['resource_policy'])
    return plan


def resources(plan):
    gpu = plan['verification']['gpu']
    pin = dict(host=plan['verification']['host'], gpu_index=gpu['index'],
               gpu_uuid=gpu['uuid'], gpu_name=gpu['name'])
    return s5_shared_gpu.snapshot(pin, plan['resource_policy'], os.environ.copy())


def _redact(value):
    if isinstance(value, dict):
        return {key: ('[ephemeral capability redacted]' if key.endswith('_TOKEN')
                      else _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def extract_repair_final(text, verified_source):
    """Extract one final source, requiring equality to the verified file.

    Retain the original answer separately. Only terminal newline differences
    are aligned; prose or a second source cannot replace a verified candidate.
    """
    import ast
    import re
    if not isinstance(text, str) or not isinstance(verified_source, str):
        raise ValueError('final answer and verified source required')
    if '```' in text:
        blocks = re.findall(r'^```(?:python|py)?[ \t]*\r?\n([\s\S]*?)^```[ \t]*$',
                            text, re.M)
        if text.count('```') != 2 or len(blocks) != 1:
            raise ValueError('ambiguous final code blocks')
        source = blocks[0]
    else:
        source = text
    ast.parse(source)
    if source.rstrip('\r\n') != verified_source.rstrip('\r\n'):
        raise ValueError('final source differs from verified repair')
    return verified_source, 'verified_repair_single_source_newline_aligned'


def run_one(plan, case, arm, output, api_key, *, repeat_id=1):
    """Run one real arm with a shared ledger, real plugin MCP, and final L2."""
    study = plan.get('mode') == 'study' or plan.get('schema') == STUDY_SCHEMA
    performance = plan.get('mode') == 'performance'
    if (arm not in plan.get('arms', ()) or type(repeat_id) is not int
            or repeat_id not in ((1, 2) if study or performance else (1,))):
        raise ValueError('unplanned plugin-only slot')
    if study:
        from benchmarks.experiments.s5.s5_study_readiness import require_study_readiness
        require_study_readiness(plan)
        if (case not in plan.get('cases', []) or dict(case_id=case['case_id'],
                repeat_id=repeat_id, arm=arm) not in plan.get('schedule', [])):
            raise ValueError('unplanned study slot')
    run_dir = Path(output) / 'runs' / case['case_id']
    if study or performance:
        run_dir = run_dir / ('repeat-%d' % repeat_id)
    run_dir = run_dir / arm
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    identity = ([plan['hash'], case['case_id'], repeat_id, arm]
                if study or performance else [plan['hash'], case['case_id'], arm])
    task_id = 'plugin-' + digest(identity)[:24]
    ledger = run_dir / 'budget.sqlite'
    record = dict(schema=RECORD_SCHEMA, plan_hash=plan['hash'], case_id=case['case_id'],
        repeat_id=repeat_id, arm=arm, status='unknown', execution_state='unknown',
        error=None, reason=None, candidate_sha256=None, verification=None,
        model_calls=None, dsh_usage=None, development=None, development_audit=None,
        gateway_diagnostics=None, wall_seconds=None, formal_s5_credit=False,
        isolation_verified=False, real_model_called=None, gpu_used=None)
    service = relay = built = gateway = transport = None
    events = []
    started = time.monotonic()
    participant = Path(tempfile.mkdtemp(prefix='operator-s5-plugin-only-'))
    try:
        service_path = Path(plan['service_root']) / case['case_id'] / 'plan.json'
        service = s5_development_service.DevelopmentService(
            service_path, case['service_sha256'], run_dir / 'supervised-verification',
            arm=arm, max_development=plan['development']['max_development'],
            wall_seconds=plan['budget']['wall_clock_s'])
        service.__enter__()
        work = participant / 'work'
        work.mkdir(parents=True, mode=0o700)
        prepared = prepare(s5_verify_remote.load_plan(service_path, case['service_sha256'])['declaration'],
                           work / 'task')
        if prepared['contract_hash'] != service.plan['contract_hash']:
            raise ValueError('prepared development contract drift')
        development = dict(url=service.url, token=service.token,
            contract=service.plan['contract_hash'], backend=service.plan['mode'],
            timeout=service.plan['timeout'] + 10)
        if arm == 'C':
            if performance:
                from benchmarks.experiments.s5.s5_perf_mcp import PerformanceRelay
                gpu = plan['verification']['gpu']
                relay = PerformanceRelay(work / 'task', service.plan['declaration'],
                    run_dir / 'mcp-relay',
                    pin=dict(host=plan['verification']['host'], gpu_index=gpu['index'],
                             gpu_uuid=gpu['uuid'], gpu_name=gpu['name']),
                    resource_policy=plan['resource_policy'], timeout=660,
                    max_candidates=plan['development']['max_candidate_configs'],
                    max_verify_calls=plan['development']['max_operator_verify_calls'],
                    max_bench_calls=plan['development']['max_operator_bench_calls'],
                    max_tool_gpu_calls=plan['development']['max_tool_gpu_calls'],
                    allow_prepare=not plan.get('require_plugin_tool_use'))
            else:
                relay = s5_plugin_rpc.MCPRelay(development, work / 'task',
                    service.plan['declaration'], run_dir / 'mcp-relay',
                    timeout=development['timeout'] + 30)
            relay.__enter__()
        shared.create(ledger, task_id=task_id,
            max_calls=plan['budget']['max_calls'],
            max_output_tokens=plan['budget']['max_output_tokens'],
            total_output_tokens=plan['budget']['total_output_tokens'],
            wall_clock_s=plan['budget']['wall_clock_s'])
        transport = s5_gateway_supervisor.SupervisedTransport(
            plan['model']['endpoint'], api_key,
            wall_seconds=plan['model']['request_wall_seconds'],
            max_request_bytes=1048576,
            max_response_bytes=plan['model']['max_response_bytes'])
        gateway = s5_gateway_http.GatewayServer(ledger, task_id=task_id,
            model=plan['model']['id'], participant='outer', adapter=transport,
            protocol='responses', max_request_bytes=1048576,
            max_response_bytes=plan['model']['max_response_bytes'],
            max_chunks=1048576, max_calls_limit=plan['budget']['max_calls'],
            reasoning_effort=plan['model']['reasoning_effort'])
        gateway.__enter__()
        participant_development = None if performance and arm == 'C' else development
        built = s5_plugin_env.build(arm, runtime=plan['readiness']['runtime'],
            relay=relay, base_url=gateway.base_url, arm_dir=participant,
            api='responses', model=plan['model']['id'],
            max_tokens=plan['budget']['max_output_tokens'], tools=True,
            development=participant_development,
            plugin_tool_timeout_ms=((relay.timeout + 30) * 1000
                                    if performance and arm == 'C' else None),
            context_window=plan['model']['context_window'],
            reasoning_effort=plan['model']['reasoning_effort'],
            stream_idle_timeout_ms=(plan['model']['stream_idle_timeout_ms']
                                    if study or performance else None))
        if plan.get('mode') == 'plugin-repair':
            (work / 'candidate.py').write_text(case['initial_candidate'], encoding='utf-8')
        if performance:
            seed_path = Path(plan['seed_root']) / case['seed_file']
            if seed_path.is_symlink() or not seed_path.is_file():
                raise ValueError('frozen performance seed unavailable')
            seed = seed_path.read_bytes()
            if hashlib.sha256(seed).hexdigest() != case['seed_sha256']:
                raise ValueError('frozen performance seed changed')
            (work / 'candidate.py').write_bytes(seed)
            task = case['text'] + '\n\n' + case['instructions']
            if plan.get('require_plugin_tool_use') and arm == 'C':
                task += (
                    '\n\nFor this frozen plugin-tool follow-up, optimize the supplied kernel and '
                    'create candidate_v1.py. Before returning the final source, make one real '
                    'mcp__operator__operator_bench call comparing candidate.py and '
                    'candidate_v1.py, then call mcp__operator__operator_verify on the selected '
                    'final candidate with stop_after="L2". This task is already prepared; do not '
                    'call operator_prepare. For both calls use run_dir="task" relative to this '
                    'session workspace. '
                    'For operator_bench pass candidates with code_file="candidate.py" and '
                    'code_file="candidate_v1.py". For operator_verify pass code_file for the '
                    'selected candidate. Omit backend, host, timeout, and work_dir so the frozen '
                    'relay supplies them. Do not use an absolute /work path or mock backend. The bench result '
                    'is exploratory; operator_verify must check the final candidate.'
                )
        else:
            task = case['text'] + '\n' + s5_plugin_flow.common_instructions(
                work, service.plan['mode'], plan['development']['max_development'],
                plan['budget']['max_calls'],
                require_plugin_verify=plan.get('mode') in ('plugin-entry', 'plugin-repair'))
        if plan.get('mode') in ('plugin-entry', 'plugin-repair'):
            task += '\n\n' + case['acceptance_instruction']
        if plan.get('mode') == 'plugin-repair':
            task = task.replace('Write candidate.py yourself and obtain verification feedback early.',
                'A starting candidate.py is provided; submit it unchanged before editing.')
        _write_once(run_dir / 'launch.json', _redact(built))
        _write_once(run_dir / 'task.json', dict(text=task))
        response = s5_arm_env.start_headless(built, task,
            timeout=plan['budget']['wall_clock_s'], capture_dir=run_dir / 'headless-output')
        events = response['events']
        _write_once(run_dir / 'events.json', events)
        _write_once(run_dir / 'stderr.json', dict(text=response['stderr']))
        usage = {}
        for event in events:
            if event.get('phase') == 'step_end':
                for key, value in (event.get('usage') or {}).items():
                    if type(value) is int:
                        usage[key] = usage.get(key, 0) + value
        record['dsh_usage'] = usage or None
        record['usage_provenance'] = 'DSH step_end, not provider-invoice evidence'
        endings = [event for event in events if event.get('phase') == 'turn_end']
        record['terminal_kind'] = (endings[-1].get('reason') or {}).get('kind') if endings else None
        if response['returncode'] != 0:
            record['execution_state'] = 'interrupted'
            record['reason'] = 'headless interrupted; inspect retained events and gateway diagnostics'
            record['headless_diagnostics'] = s5_arm_env.headless_failure_diagnostics(response)
            if record['headless_diagnostics']:
                record['reason'] = 'DSH adapter stream idle timeout; upstream root cause unknown'
        else:
            candidate = s5_formal_run._final_text(events)
            if not candidate or not candidate.strip():
                record['execution_state'] = 'interrupted'
                record['reason'] = 'no single nonempty final answer'
            else:
                record['execution_state'] = 'finished'
                (run_dir / 'raw-answer.txt').write_text(candidate, encoding='utf-8')
                dev_records = [r for r in service.snapshot()['records'] if r['phase'] == 'development']
                if study or performance:
                    from benchmarks.experiments.s5.s5_candidate_extraction import extract_final_source
                    candidate, normalization = extract_final_source(candidate)
                elif (plan.get('mode') == 'plugin-repair' and dev_records
                        and dev_records[-1]['status'] == 'passed'):
                    last = dev_records[-1]
                    verified_path = service.output / ('%03d-development' % last['sequence']) / 'candidate.py'
                    candidate, normalization = extract_repair_final(candidate, verified_path.read_text())
                else:
                    candidate, normalization = s5_formal_run.normalize_candidate(candidate)
                (run_dir / 'candidate.txt').write_text(candidate, encoding='utf-8')
                from operator_agent.core.run_contract import code_hash
                record['candidate_sha256'] = code_hash(candidate)
                verdict = service.final(candidate)
                record['verification'] = dict(verdict,
                    service_plan_sha256=case['service_sha256'])
                record['normalization'] = normalization
                record['status'] = verdict['status']
                if (study or performance) and (verdict.get('source_sha256') != record['candidate_sha256']
                        or verdict.get('contract_hash') != case['service_contract']):
                    raise ValueError('performance-study final verification identity mismatch')
    except Exception as exc:
        from benchmarks.experiments.s5.s5_candidate_extraction import FinalSourceError
        if (study or performance) and isinstance(exc, FinalSourceError):
            record['error'] = None
            record['reason'] = 'final source refused: no unambiguous Python candidate'
            record['final_submission_refused'] = True
            record['execution_state'] = 'interrupted'
            record['terminal_kind'] = 'completed'
        else:
            record['error'] = type(exc).__name__
            record['reason'] = 'plugin-only run interrupted; inspect retained local diagnostics'
            if record['execution_state'] == 'unknown':
                record['execution_state'] = 'interrupted'
        record['status'] = 'unknown'
    finally:
        for owner in (gateway, relay, service):
            if owner is not None:
                try:
                    owner.__exit__(None, None, None)
                except Exception:
                    record['cleanup_error'] = type(sys.exc_info()[1]).__name__
        if (run_dir / 'headless-output' / 'metadata.json').is_file():
            record['headless_output'] = json.loads(
                (run_dir / 'headless-output' / 'metadata.json').read_text())
        if gateway is not None:
            record['budget_refusal'] = gateway._server.gateway.last_budget_refusal
        if transport is not None:
            diagnostics = transport.failure_snapshot()
            record['gateway_diagnostics'] = diagnostics
            _write_once(run_dir / 'gateway-diagnostics.json', diagnostics)
        if ledger.exists():
            snapshot = shared.snapshot(ledger, expected_task_id=task_id)
            _write_once(run_dir / 'budget.json', snapshot)
            record['model_calls'] = dict(reserved=snapshot['reserved_calls'],
                completed=snapshot['completed_calls'], aborted=snapshot['aborted'],
                usage=[call.get('api_usage') for call in snapshot['calls']])
        if service is not None:
            record['development'] = service.snapshot()
            if study:
                first = next((item for item in record['development']['records']
                    if item.get('status') == 'passed' and (item.get('result') or {}).get(
                        'full_l2_passed', (item.get('result') or {}).get('passed')) is True), None)
                record['first_complete_correct'] = None if first is None else dict(
                    phase=first['phase'], sequence=first['sequence'],
                    source_sha256=first['source_sha256'],
                    elapsed_seconds=round(first['completed_monotonic'] - started, 3),
                    model_call_reservations=((record.get('model_calls') or {}).get('reserved')
                                             if first['phase'] == 'final' else None),
                    call_watermark_basis=('all DSH requests completed before independent final'
                        if first['phase'] == 'final' else
                        'unknown; development feedback lacks provider dispatch watermark'))
            _write_once(run_dir / 'development.json', record['development'])
            if not performance:
                record['development_audit'] = s5_plugin_probe.audit(
                    events, record['development'], arm=arm)
                _write_once(run_dir / 'development-audit.json', record['development_audit'])
                record['workflow_progress'] = s5_plugin_flow.progress(events, record['development'])
                _write_once(run_dir / 'workflow-progress.json', record['workflow_progress'])
        if relay is not None:
            _write_once(run_dir / 'mcp-relay-snapshot.json', relay.snapshot())
        if built is not None:
            s5_arm_env.cleanup_development(built)
            record['workspace_archive'] = s5_seatbelt.archive_work(
                built, run_dir / 'participant-work')
            record['read_boundary'] = 'macos-outer-seatbelt-whitelist'
        record['wall_seconds'] = round(time.monotonic() - started, 3)
        record['outcome_class'] = s5_outcomes.classify(
            record, require_development_audit=not performance)
        _write_once(run_dir / 'record.json', record)
    return record


def aggregate_c_diagnostic(plan, records):
    """Summarize C-only task health without producing an A/C comparison."""
    if (not isinstance(plan, dict) or plan.get('schema') != C_ONLY_SCHEMA
            or plan.get('arms') != list(C_ONLY_ARMS)
            or plan.get('formal_s5_credit') is not False
            or not isinstance(plan.get('hash'), str) or not plan['hash']):
        raise ValueError('C-only diagnostic requires its frozen plan identity')
    cases = plan.get('cases')
    if ([case.get('case_id') for case in cases or []] != list(PREFLIGHT_CASES)
            or plan.get('repeats') != 1 or plan.get('planned_runs') != len(PREFLIGHT_CASES)
            or plan.get('schedule') != _case_order(cases, 1, C_ONLY_ARMS)):
        raise ValueError('C-only diagnostic plan grid mismatch')
    if not isinstance(records, list):
        raise ValueError('C-only diagnostic records must be a list')

    expected = {case_id for case_id in PREFLIGHT_CASES}
    by_case = {}
    for row in records:
        if (not isinstance(row, dict) or row.get('arm') != 'C'
                or type(row.get('repeat_id', 1)) is not int
                or row.get('repeat_id', 1) != 1
                or row.get('case_id') not in expected
                or row.get('case_id') in by_case
                or row.get('plan_hash') != plan['hash']
                or row.get('status') not in ('passed', 'failed', 'unknown')
                or row.get('execution_state') not in
                    ('finished', 'interrupted', 'unknown', 'not_executed')):
            raise ValueError('duplicate, foreign or inconsistent C-only diagnostic record')
        if row['status'] in ('passed', 'failed') and row['execution_state'] != 'finished':
            raise ValueError('unfinished diagnostic row cannot have a definitive verdict')
        by_case[row['case_id']] = row

    rows = []
    for case_id in PREFLIGHT_CASES:
        row = by_case.get(case_id)
        if row is None:
            row = dict(case_id=case_id, repeat_id=1, arm='C', plan_hash=plan['hash'],
                       status='unknown', execution_state='not_executed',
                       outcome_class='not_executed', reason='no execution record',
                       formal_s5_credit=False)
        rows.append(row)
    body = dict(schema='s5-plugin-c-diagnostic-summary-v1', plan_hash=plan['hash'],
        arms=['C'], planned=len(PREFLIGHT_CASES), received=len(records),
        dispatched_runs=sum(row['execution_state'] != 'not_executed' for row in rows),
        counts={state: sum(row['status'] == state for row in rows)
                for state in ('passed', 'failed', 'unknown')},
        rows=rows, formal_s5_credit=False, inference_allowed=False,
        interpretation='C-only workflow diagnostic; no A/C comparison or plugin-gain claim')
    return dict(body, hash=digest(body))


def aggregate_plugin_entry(plan, records):
    """Require audited plugin submission, returned feedback and final L2 pass."""
    repair = isinstance(plan, dict) and plan.get('mode') == 'plugin-repair'
    if (not isinstance(plan, dict)
            or plan.get('schema') != (PLUGIN_REPAIR_SCHEMA if repair else PLUGIN_ENTRY_SCHEMA)
            or plan.get('mode') not in ('plugin-entry', 'plugin-repair') or plan.get('arms') != ['C']
            or plan.get('formal_s5_credit') is not False
            or not isinstance(plan.get('hash'), str) or not plan['hash']):
        raise ValueError('plugin-entry acceptance requires its frozen plan identity')
    if ([case.get('case_id') for case in plan.get('cases', [])] != list(PLUGIN_ENTRY_CASES)
            or plan.get('planned_runs') != 1 or plan.get('schedule') !=
                _case_order(plan['cases'], 1, C_ONLY_ARMS)):
        raise ValueError('plugin-entry acceptance plan grid mismatch')
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError('plugin-entry acceptance requires its single assigned record')
    row = records[0]
    if (not isinstance(row, dict) or row.get('case_id') != PLUGIN_ENTRY_CASES[0]
            or row.get('arm') != 'C' or type(row.get('repeat_id')) is not int
            or row.get('repeat_id') != 1
            or row.get('plan_hash') != plan['hash']
            or row.get('status') not in ('passed', 'failed', 'unknown')
            or row.get('execution_state') not in ('finished', 'interrupted', 'unknown')
            or (row['status'] in ('passed', 'failed') and row['execution_state'] != 'finished')):
        raise ValueError('foreign or inconsistent plugin-entry acceptance record')

    audit = row.get('development_audit') or {}
    receipts = audit.get('receipts', []) if isinstance(audit, dict) else []
    plugin_receipts = [receipt for receipt in receipts
                       if receipt.get('ingress') == 'operator_mcp']
    development = row.get('development') or {}
    dev_records = [item for item in development.get('records', [])
                   if item.get('phase') == 'development']
    feedback_returned = bool(plugin_receipts) and audit.get('complete') is True and any(
        item.get('status') == 'passed'
        and item.get('response_sha256') == receipt.get('service_response_sha256')
        for item in dev_records for receipt in plugin_receipts)
    final = row.get('verification') or {}
    final_l2_passed = final.get('full_l2_passed')
    if 'full_l2_passed' not in final:
        # Older service responses named this exact verifier verdict `passed`.
        final_l2_passed = final.get('passed')
    checks = dict(
        operator_verify_submitted=bool(plugin_receipts),
        l2_feedback_returned_to_dsh=feedback_returned,
        final_candidate_passed=(row.get('status') == 'passed'
                                and final.get('status') == 'passed'
                                and final_l2_passed is True),
    )
    if repair:
        first = dev_records[0] if dev_records else {}
        second = dev_records[1] if len(dev_records) > 1 else {}
        first_result = first.get('result') or {}
        seed_hash = plan['cases'][0].get('initial_candidate_sha256')
        receipt_by_sequence = {r.get('sequence'): r for r in plugin_receipts}
        def bound(item):
            receipt = receipt_by_sequence.get(item.get('sequence'), {})
            return bool(receipt) and (
                receipt.get('source_sha256') == item.get('source_sha256')
                and receipt.get('service_response_sha256') == item.get('response_sha256'))
        checks.update(
            unchanged_seed_failed_on_gpu=(bool(seed_hash) and first.get('source_sha256') == seed_hash
                and first.get('status') == 'failed' and first_result.get('passed') is False
                and (first_result.get('levels') or {}).get('L1') is True
                and (first_result.get('levels') or {}).get('L2') is False and bound(first)),
            revised_candidate_passed=(len(dev_records) == 2 and second.get('status') == 'passed'
                and second.get('source_sha256') not in (None, seed_hash) and bound(second)),
            final_matches_repaired_candidate=(bool(row.get('candidate_sha256'))
                and row.get('candidate_sha256') == second.get('source_sha256')
                and final.get('source_sha256') == row.get('candidate_sha256')),
        )
    accepted = all(checks.values())
    body = dict(schema=('s5-plugin-repair-acceptance-summary-v1' if repair
                        else 's5-plugin-entry-acceptance-summary-v1'),
        plan_hash=plan['hash'], planned=1, received=1, dispatched_runs=1,
        counts={state: int(row['status'] == state) for state in ('passed', 'failed', 'unknown')},
        status='passed' if accepted else 'not_demonstrated', accepted=accepted,
        checks=checks, development_receipts=plugin_receipts,
        final_candidate_sha256=row.get('candidate_sha256'), rows=[row],
        formal_s5_credit=False, inference_allowed=False,
        interpretation='single-task product-entry acceptance; no A/C or plugin-gain inference')
    return dict(body, hash=digest(body))


def execute(plan_path, expected_plan_hash, output, api_key=None):
    """Dispatch a frozen A/C batch or C-only diagnostic by exact plan hash."""
    plan = load_plan(plan_path)
    if plan['hash'] != expected_plan_hash:
        raise ValueError('plugin-only expected plan hash mismatch')
    formal_readiness = None
    if plan.get('mode') == 'study':
        from benchmarks.experiments.s5.s5_study_readiness import require_study_readiness
        formal_readiness = require_study_readiness(plan)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    initial = resources(plan)
    if not isinstance(api_key, str):
        from benchmarks.experiments.s5.s5_workflow_preflight import credential
        api_key = credential()
    if not api_key or '\n' in api_key or '\r' in api_key:
        raise ValueError('provider credential unavailable')
    _write_once(root / 'execution-started.json', dict(schema='s5-plugin-only-execution-v1',
        plan_hash=plan['hash'], started_unix=time.time(), initial_resources=initial,
        formal_readiness=formal_readiness,
        real_model_calls='not yet observed', gpu_executions=0, formal_s5_credit=False))
    records = []
    cases = {case['case_id']: case for case in plan['cases']}
    stopped = None
    for index, slot in enumerate(plan['schedule']):
        case, arm = cases[slot['case_id']], slot['arm']
        label = ('%02d-%s-repeat-%d-%s' % (index + 1, case['case_id'], slot['repeat_id'], arm)
                 if plan.get('mode') == 'study' else '%02d-%s-%s' % (index + 1, case['case_id'], arm))
        if stopped is not None:
            record = dict(case_id=case['case_id'], repeat_id=slot['repeat_id'], arm=arm,
                plan_hash=plan['hash'], status='unknown', execution_state='not_executed',
                outcome_class='not_executed', reason=stopped, formal_s5_credit=False)
        else:
            try:
                observation = resources(plan)
                _write_once(root / 'resource-checks' / (label + '.json'), observation)
                record = run_one(plan, case, arm, root, api_key, repeat_id=slot['repeat_id'])
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                record = dict(case_id=case['case_id'], repeat_id=slot['repeat_id'], arm=arm,
                    plan_hash=plan['hash'], status='unknown', execution_state='interrupted',
                    error=type(exc).__name__, reason='slot refused; inspect retained evidence',
                    model_calls=None, wall_seconds=None, formal_s5_credit=False)
                record['outcome_class'] = 'infrastructure_error'
            if record.get('outcome_class') == 'infrastructure_error':
                stopped = 'infrastructure failure at %s' % label
        _write_once(root / 'records' / (label + '.json'), record)
        records.append(record)
    if plan.get('mode') == 'study':
        from benchmarks.experiments.s5.s5_plugin_study_summary import aggregate
        summary = aggregate(plan, records)
    elif plan.get('mode') in ('plugin-entry', 'plugin-repair'):
        summary = aggregate_plugin_entry(plan, records)
    elif plan['arms'] == list(C_ONLY_ARMS):
        summary = aggregate_c_diagnostic(plan, records)
    else:
        summary = s5_plugin_summary.aggregate(plan, records)
    _write_once(root / 'aggregate.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--readiness-report', required=True, type=Path)
    freeze_parser.add_argument('--output', required=True, type=Path)
    freeze_parser.add_argument('--gpu-index', required=True, type=int)
    freeze_parser.add_argument('--gpu-uuid', required=True)
    freeze_parser.add_argument('--gpu-name', required=True)
    freeze_parser.add_argument('--allowed-pid', action='append', type=int, default=[])
    freeze_parser.add_argument('--mode', choices=('ac', 'c-only', 'plugin-entry', 'plugin-repair', 'study'), default='ac')
    freeze_parser.add_argument('--formal-readiness-report', type=Path)
    freeze_parser.add_argument('--exploratory', action='store_true')
    run_parser = sub.add_parser('execute')
    run_parser.add_argument('--plan', required=True, type=Path)
    run_parser.add_argument('--plan-hash', required=True)
    run_parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        policy = dict(mode='shared-known-processes', min_free_mib=8192,
                      max_util_pct=5, allowed_compute_pids=sorted(set(args.allowed_pid)))
        plan = freeze(args.output, readiness_report=args.readiness_report,
            gpu=dict(index=args.gpu_index, uuid=args.gpu_uuid, name=args.gpu_name),
            resource_policy=policy,
            arms=ARMS if args.mode in ('ac', 'study') else C_ONLY_ARMS,
            case_ids=(STUDY_CASES if args.mode == 'study' else PLUGIN_ENTRY_CASES
                      if args.mode in ('plugin-entry', 'plugin-repair') else PREFLIGHT_CASES),
            repeats=2 if args.mode == 'study' else 1,
            mode=args.mode, formal_readiness_report=args.formal_readiness_report,
            exploratory=args.exploratory)
        print(json.dumps(dict(plan_path=str(args.output / 'plan.json'), hash=plan['hash'],
                              planned_runs=plan['planned_runs'], arms=plan['arms']), sort_keys=True))
        return 0
    summary = execute(args.plan, args.plan_hash, args.output)
    print(json.dumps(dict(hash=summary['hash'], counts=summary['counts'],
                          inference_allowed=summary.get('comparisons', {}).get(
                              'C-minus-A', {}).get('inference_allowed',
                                  summary.get('inference_allowed', False)),
                          accepted=summary.get('accepted', False)),
                     sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
