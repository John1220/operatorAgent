"""Run the staged A/C kernel-performance trial through the production DSH path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.experiments.s5 import s5_formal_run, s5_perf_evaluator, s5_plugin_probe, s5_plugin_workflow, s5_runtime, s5_shared_gpu, s5_verify_remote
from operator_agent.run_contract import digest

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 's5-kernel-performance-trial-v1'
BASE = ROOT / 'output/plugin_completion/20261030-s5-kernel-performance-v1'
READINESS = BASE / 'readiness-repaired/summary.json'
PERF_PROBE = BASE / 'readiness-repaired/perf-mcp/result.json'
PINS = {
    2: dict(host='algorithm-h20', gpu_index=2,
           gpu_uuid='GPU-13c6c4b4-45ce-8963-1122-df2b3d72fb76', gpu_name='NVIDIA H20'),
    7: dict(host='algorithm-h20', gpu_index=7,
           gpu_uuid='GPU-52ca6155-d8bc-a416-847a-6f0c71db1552', gpu_name='NVIDIA H20'),
}
POLICIES = {
    2: dict(mode='shared-known-processes', min_free_mib=8192, max_util_pct=5,
            allowed_compute_pids=[162589, 822889, 1641644, 1677855, 3885613, 3890552, 3916094]),
    7: dict(mode='shared-known-processes', min_free_mib=8192, max_util_pct=5,
            allowed_compute_pids=[]),
}
BUDGET = dict(s5_plugin_workflow.BUDGET)
DEVELOPMENT = dict(max_development=2, max_candidate_configs=3,
                   max_operator_verify_calls=3, max_operator_bench_calls=3,
                   max_tool_gpu_calls=6)
FOLLOWUP_KIND = 'layernorm-plugin-tools-pair-v1'
INSTRUCTIONS = (
    'A starting candidate.py is provided in the task workspace. Use it as the initial '
    'implementation and optimize the exact declared workload. You may use the validation, '
    'benchmark, and other tools actually available in this DSH session; do not install '
    'packages, invoke another model, or access undeclared host resources. Submit no more than '
    'three distinct kernel configurations to GPU tools, and do not exceed the per-tool limits '
    'shown by the available interface. Keep every candidate correct for all declared inputs. '
    'Return only one final Python kernel source, with no explanation.'
)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _identity():
    paths = ('benchmarks/experiments/s5/s5_perf_trial.py', 'benchmarks/experiments/s5/s5_perf_mcp.py',
             'benchmarks/experiments/s5/s5_perf_evaluator.py', 'benchmarks/experiments/s5/s5_plugin_workflow.py',
             'benchmarks/experiments/s5/s5_outcomes.py',
             'benchmarks/experiments/s5/s5_formal_run.py',
             'benchmarks/experiments/s5/s5_plugin_env.py', 'benchmarks/experiments/s5/s5_plugin_probe.py',
             'benchmarks/experiments/s5/s5_arm_env.py', 'benchmarks/experiments/s5/s5_plugin_rpc.py',
             'benchmarks/experiments/s5/s5_verify_remote.py', 'benchmarks/experiments/s5/s5_shared_gpu.py',
             'benchmarks/experiments/s5/s5_gateway_http.py', 'benchmarks/experiments/s5/s5_gateway_supervisor.py',
             'benchmarks/experiments/s5/s5_gateway_transport.py', 'benchmarks/experiments/s5/s5_development_service.py',
             'benchmarks/experiments/s5/s5_candidate_extraction.py', 'operator_agent/mcp_server.py',
             'operator_agent/cli.py', 'integrations/dsh-operator/index.mjs')
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def _source(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('trial source missing')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('trial source changed')
    return raw


def freeze(output, *, gpu_index=2, layernorm_followup=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    if gpu_index not in PINS:
        raise ValueError('GPU index is not an approved trial target')
    pin = dict(PINS[gpu_index])
    policy = dict(POLICIES[gpu_index])
    readiness = s5_plugin_probe.load_readiness(READINESS)
    perf_probe = _read(PERF_PROBE)
    if perf_probe.get('status') != 'passed' or set(perf_probe.get('visible_tools', ())) != {
            'operator_bench', 'operator_capabilities', 'operator_prepare', 'operator_verify'}:
        raise ValueError('performance MCP readiness failed')
    s5_shared_gpu.validate_policy(policy)
    from benchmarks.experiments.s5.s5_gateway_transport import _endpoint
    endpoint = os.environ.get(s5_plugin_workflow.ENDPOINT_KEY,
                              s5_plugin_workflow.DEFAULT_ENDPOINT)
    _endpoint(endpoint)

    source_cases = (
        dict(case_id='layernorm-16384x6144', kernel_type='layernorm',
             shape=[16384, 6144], precision='fp32', layout='contiguous',
             seed_file='layernorm16384x6144.py'),
        dict(case_id='matmul-4096', kernel_type='matmul',
             shape=[4096, 4096, 4096], precision='fp16', layout='contiguous',
             seed_file='matmul4096.py'),
    )
    if layernorm_followup:
        source_cases = source_cases[:1]
    cases = []
    seeds = BASE / 'seeds'
    for item in source_cases:
        seed_path = seeds / item['seed_file']
        seed = seed_path.read_bytes()
        declaration = dict(request='%s %s triton' % (item['kernel_type'], item['precision']),
            shapes=[item['shape']], hardware='nvidia_gpu', dsl='triton',
            precision=item['precision'], min_speedup=1.0)
        row = dict(case_id=item['case_id'], kernel_type=item['kernel_type'],
                   shape=item['shape'], precision=item['precision'], layout='contiguous')
        task_text = s5_formal_run.task_text(row)
        cases_for_repeats = []
        for repeat in ((1,) if layernorm_followup else (1, 2)):
            case_id = '%s-repeat-%d' % (item['case_id'], repeat)
            service_root = output / 'services' / case_id
            service_sha = s5_verify_remote.freeze(service_root, declaration=declaration,
                backend='h20', host=pin['host'], timeout=600,
                gpu_index=pin['gpu_index'], gpu_uuid=pin['gpu_uuid'], gpu_name=pin['gpu_name'])
            service_plan = s5_verify_remote.load_plan(service_root / 'plan.json', service_sha)
            contract = _read(service_root / 'task/contract.json')
            input_semantics = s5_formal_run.check_task_inputs(row, contract)
            cases_for_repeats.append(dict(case_id=case_id, repeat_id=repeat,
                task_id=item['case_id'], kernel_type=item['kernel_type'], shape=item['shape'],
                precision=item['precision'], layout=item['layout'], text=task_text,
                instructions=INSTRUCTIONS, seed_file=item['seed_file'],
                seed_sha256=hashlib.sha256(seed).hexdigest(), declaration=declaration,
                service_sha256=service_sha, service_contract=service_plan['contract_hash'],
                input_semantics=input_semantics))
        cases.extend(cases_for_repeats)

    schedule = [dict(case_id='layernorm-16384x6144-repeat-1', repeat_id=1, arm='C'),
                dict(case_id='layernorm-16384x6144-repeat-1', repeat_id=1, arm='A')]
    if not layernorm_followup:
        schedule = [dict(case_id='layernorm-16384x6144-repeat-1', repeat_id=1, arm='A'),
                dict(case_id='layernorm-16384x6144-repeat-1', repeat_id=1, arm='C'),
                dict(case_id='matmul-4096-repeat-1', repeat_id=1, arm='C'),
                dict(case_id='matmul-4096-repeat-1', repeat_id=1, arm='A'),
                dict(case_id='layernorm-16384x6144-repeat-2', repeat_id=2, arm='C'),
                dict(case_id='layernorm-16384x6144-repeat-2', repeat_id=2, arm='A'),
                dict(case_id='matmul-4096-repeat-2', repeat_id=2, arm='A'),
                dict(case_id='matmul-4096-repeat-2', repeat_id=2, arm='C')]
    followup_instruction = bool(layernorm_followup)
    plan = dict(schema=SCHEMA, status='frozen', mode='performance', formal_s5_credit=False,
        counts_toward_denominator=False,
        scope=('LayerNorm repeat-1 C-then-A follow-up requiring actual plugin benchmark and verification'
               if layernorm_followup else
               'two declared large-shape kernels; native DSH versus DSH plus performance plugin'),
        followup_kind=FOLLOWUP_KIND if layernorm_followup else None,
        require_plugin_tool_use=followup_instruction,
        model=dict(id=s5_plugin_workflow.MODEL, protocol='responses', endpoint=endpoint,
            reasoning_effort='medium', context_window=s5_formal_run.CONTEXT_WINDOW,
            request_wall_seconds=s5_formal_run.REQUEST_WALL_SECONDS,
            max_response_bytes=s5_formal_run.RESPONSE_BYTES,
            stream_idle_timeout_ms=s5_plugin_workflow.STUDY_STREAM_IDLE_TIMEOUT_MS),
        budget=BUDGET, arms=['A', 'C'], repeats=1 if layernorm_followup else 2,
        development=DEVELOPMENT,
        verification=dict(backend='h20', host=pin['host'], gpu=dict(index=pin['gpu_index'],
            uuid=pin['gpu_uuid'], name=pin['gpu_name']), stop_after='L2', timeout=600),
        pin=pin, resource_policy=policy, readiness=readiness,
        readiness_probe_sha256=hashlib.sha256(Path(PERF_PROBE).read_bytes()).hexdigest(),
        service_root=str(output / 'services'), seed_root=str(seeds), cases=cases,
        schedule=schedule, planned_runs=len(schedule), implementation=_identity(),
        runtime_deps=s5_runtime.tree_digest(s5_runtime.deps_dir()),
        interpretation=('targeted LayerNorm tool-use follow-up; excluded from the original pilot'
                        if layernorm_followup else
                        'staged A/C product-workflow comparison; preflight results excluded'))
    plan['hash'] = digest(plan)
    _write(output / 'plan.json', plan)
    _write(output / 'frozen-sources.json', dict(implementation=plan['implementation'],
        plugin_runtime=readiness['runtime'], perf_mcp_probe_sha256=plan['readiness_probe_sha256']))
    return plan


def load_plan(path, expected_hash):
    path = Path(path)
    plan = _read(path)
    stored = plan.pop('hash', None)
    if not isinstance(stored, str) or stored != expected_hash or digest(plan) != stored:
        raise ValueError('performance trial plan identity mismatch')
    plan['hash'] = stored
    pin = plan.get('pin')
    gpu_index = pin.get('gpu_index') if isinstance(pin, dict) else None
    full_schedule = [
                dict(case_id='layernorm-16384x6144-repeat-1', repeat_id=1, arm='A'),
                dict(case_id='layernorm-16384x6144-repeat-1', repeat_id=1, arm='C'),
                dict(case_id='matmul-4096-repeat-1', repeat_id=1, arm='C'),
                dict(case_id='matmul-4096-repeat-1', repeat_id=1, arm='A'),
                dict(case_id='layernorm-16384x6144-repeat-2', repeat_id=2, arm='C'),
                dict(case_id='layernorm-16384x6144-repeat-2', repeat_id=2, arm='A'),
                dict(case_id='matmul-4096-repeat-2', repeat_id=2, arm='A'),
                dict(case_id='matmul-4096-repeat-2', repeat_id=2, arm='C')]
    followup_schedule = [
        dict(case_id='layernorm-16384x6144-repeat-1', repeat_id=1, arm='C'),
        dict(case_id='layernorm-16384x6144-repeat-1', repeat_id=1, arm='A')]
    followup = plan.get('followup_kind') == FOLLOWUP_KIND
    expected_schedule = followup_schedule if followup else full_schedule
    expected_cases = {'layernorm-16384x6144-repeat-1'} if followup else {
        'layernorm-16384x6144-repeat-1', 'layernorm-16384x6144-repeat-2',
        'matmul-4096-repeat-1', 'matmul-4096-repeat-2'}
    if (plan.get('schema') != SCHEMA or plan.get('mode') != 'performance'
            or plan.get('arms') != ['A', 'C'] or plan.get('schedule') != expected_schedule
            or plan.get('planned_runs') != len(expected_schedule)
            or plan.get('require_plugin_tool_use') is not followup
            or {case.get('case_id') for case in plan.get('cases', [])} != expected_cases
            or plan.get('implementation') != _identity()
            or plan.get('runtime_deps') != s5_runtime.tree_digest(s5_runtime.deps_dir())):
        raise ValueError('performance trial configuration or implementation drift')
    if (gpu_index not in PINS or pin != PINS[gpu_index]
            or plan.get('resource_policy') != POLICIES[gpu_index]):
        raise ValueError('performance trial GPU target or policy drift')
    if plan.get('readiness') != s5_plugin_probe.load_readiness(READINESS):
        raise ValueError('performance readiness changed')
    if hashlib.sha256(Path(PERF_PROBE).read_bytes()).hexdigest() != plan.get(
            'readiness_probe_sha256') or _read(PERF_PROBE).get('status') != 'passed':
        raise ValueError('performance MCP readiness probe changed')
    if plan.get('budget') != BUDGET or plan.get('development') != DEVELOPMENT:
        raise ValueError('performance trial budget changed')
    s5_shared_gpu.validate_policy(plan['resource_policy'])
    from benchmarks.experiments.s5.s5_gateway_transport import _endpoint
    _endpoint(plan['model']['endpoint'])
    for case in plan['cases']:
        seed = _source(Path(plan['seed_root']) / case['seed_file'], case['seed_sha256'])
        if not seed:
            raise ValueError('empty starting candidate')
        service_path = Path(plan['service_root']) / case['case_id'] / 'plan.json'
        service = s5_verify_remote.load_plan(service_path, case['service_sha256'])
        if (service.get('declaration') != case['declaration']
                or service.get('contract_hash') != case['service_contract']
                or service.get('stop_after') != 'L2'
                or service.get('gpu_index') != gpu_index
                or service.get('gpu_uuid') != pin['gpu_uuid']):
            raise ValueError('performance task service drift')
    return plan


def _pin(plan):
    return dict(host=plan['pin']['host'], gpu_index=plan['pin']['gpu_index'],
                gpu_uuid=plan['pin']['gpu_uuid'], gpu_name=plan['pin']['gpu_name'])


def _write_gpu_snapshot(plan, path):
    value = s5_shared_gpu.snapshot(_pin(plan), plan['resource_policy'],
                                   s5_perf_evaluator.environment(dict(
                                       pin=_pin(plan), resource_policy=plan['resource_policy'])))
    _write(path, value)
    return value


def _run_slot(plan, slot, root, api_key):
    case = next(item for item in plan['cases'] if item['case_id'] == slot['case_id'])
    observation_id = '%02d-%s-%s' % (plan['schedule'].index(slot) + 1,
                                     case['case_id'], slot['arm'])
    observations = root / 'resource-checks'
    before = _write_gpu_snapshot(plan, observations / (observation_id + '-before.json'))
    started = time.monotonic()
    record = s5_plugin_workflow.run_one(plan, case, slot['arm'], root, api_key,
                                        repeat_id=slot['repeat_id'])
    after = _write_gpu_snapshot(plan, observations / (observation_id + '-after.json'))
    stable = [item['pid'] for item in before['compute_apps']] == [
        item['pid'] for item in after['compute_apps']]
    record['trial_resource_check'] = dict(process_set_stable=stable,
        before=observation_id + '-before.json', after=observation_id + '-after.json')
    record['trial_wall_seconds'] = round(time.monotonic() - started, 3)
    if not stable:
        record.update(status='unknown', execution_state='interrupted',
                      reason='GPU resident process set changed during model cell')
    relay_path = root / 'runs' / case['case_id'] / ('repeat-%d' % slot['repeat_id']) / slot['arm'] / 'mcp-relay-snapshot.json'
    if relay_path.is_file():
        relay = _read(relay_path)
        if relay.get('resource_error'):
            record.update(status='unknown', execution_state='interrupted',
                          reason='performance MCP GPU resource check failed')
        if plan.get('require_plugin_tool_use') and slot['arm'] == 'C':
            successful_tools = {item.get('tool') for item in relay.get('records', [])
                                if item.get('outcome') == 'answered'
                                and 'rpc_error_code' not in item}
            observed_tools = {item.get('tool') for item in relay.get('gpu_observations', [])
                              if item.get('process_set_stable') is True}
            required = {'operator_bench', 'operator_verify'}
            if (not required <= successful_tools or not required <= observed_tools
                    or relay.get('resource_error')):
                record.update(status='unknown', execution_state='interrupted',
                    reason='C did not complete both required plugin GPU tools')
    _write(root / 'records' / (observation_id + '.json'), record)
    return record


def _evaluate_final(plan, slot, root):
    case = next(item for item in plan['cases'] if item['case_id'] == slot['case_id'])
    run = root / 'runs' / case['case_id'] / ('repeat-%d' % slot['repeat_id']) / slot['arm']
    record = _read(root / 'records' / ('%02d-%s-%s.json' % (
        plan['schedule'].index(slot) + 1, case['case_id'], slot['arm'])))
    if (record.get('status') != 'passed' or not record.get('candidate_sha256')
            or not (run / 'candidate.txt').is_file()):
        return dict(status='not_run', reason='independent L2 did not pass', qualified=False)
    eval_dir = root / 'final-evaluations' / ('%02d-%s-%s' % (
        plan['schedule'].index(slot) + 1, case['case_id'], slot['arm']))
    frozen = s5_perf_evaluator.freeze(eval_dir,
        declaration=case['declaration'], candidate_path=run / 'candidate.txt',
        pin=_pin(plan), resource_policy=plan['resource_policy'], timeout=600,
        stop_after='L3')
    return s5_perf_evaluator.execute(eval_dir / 'plan.json', frozen['hash'])


def execute(plan_path, plan_hash, output, *, start_index=1, count=2,
            independent_window=False):
    plan = load_plan(plan_path, plan_hash)
    root = Path(output).resolve()
    if type(start_index) is not int or type(count) is not int or not 1 <= start_index <= 8 or not 1 <= count <= 8:
        raise ValueError('invalid execution window')
    end_index = min(8, start_index + count - 1)
    if start_index > 1 and not independent_window:
        previous = [root / 'records' / ('%02d-%s-%s.json' % (
            index, plan['schedule'][index - 1]['case_id'], plan['schedule'][index - 1]['arm']))
            for index in range(1, start_index)]
        if any(not path.is_file() for path in previous):
            raise ValueError('prior schedule slots must be executed and retained first')
    api_key = s5_formal_run._read_key()
    _write(root / ('execution-window-%02d-%02d.json' % (start_index, end_index)),
        dict(schema='s5-kernel-performance-execution-v1', plan_hash=plan_hash,
             start_index=start_index, end_index=end_index, model_calls_not_yet_known=True,
             formal_s5_credit=False))
    records = []
    stop_reason = None
    for index in range(start_index, end_index + 1):
        slot = plan['schedule'][index - 1]
        try:
            record = _run_slot(plan, slot, root, api_key)
        except (OSError, ValueError, RuntimeError, TimeoutError,
                subprocess.SubprocessError) as exc:
            record = dict(plan_hash=plan_hash, case_id=slot['case_id'], repeat_id=slot['repeat_id'],
                arm=slot['arm'], status='unknown', execution_state='interrupted',
                reason='trial slot failed before a conclusive outcome', error=type(exc).__name__,
                formal_s5_credit=False)
            label = '%02d-%s-%s' % (index, slot['case_id'], slot['arm'])
            _write(root / 'records' / (label + '.json'), record)
        records.append(record)
        if record.get('status') == 'unknown' or record.get('execution_state') != 'finished':
            stop_reason = 'unknown or interrupted result at schedule slot %d' % index
            break
    evals = []
    if (start_index in (1, 3, 5, 7) and end_index == start_index + 1
            and len(records) == 2):
        for offset, record in enumerate(records):
            slot = plan['schedule'][start_index - 1 + offset]
            result = _evaluate_final(plan, slot, root)
            label = '%02d-%s-%s' % (start_index + offset, slot['case_id'], slot['arm'])
            _write(root / 'final-results' / (label + '.json'), result)
            evals.append(result)
    _write(root / ('window-result-%02d-%02d.json' % (start_index, end_index)),
        dict(schema='s5-kernel-performance-window-result-v1', plan_hash=plan_hash,
             schedule_start=start_index, schedule_end=end_index,
             records=records, independent_evaluations=evals, stopped_reason=stop_reason,
             next_schedule_index=(end_index + 1 if stop_reason is None else None),
             interpretation='bounded A/C trial window; no global plugin superiority claim',
             formal_s5_credit=False))
    return dict(records=records, evaluations=evals, stopped_reason=stop_reason)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--output', required=True, type=Path)
    f.add_argument('--gpu-index', type=int, choices=sorted(PINS), default=2)
    f.add_argument('--layernorm-followup', action='store_true')
    e = sub.add_parser('execute')
    e.add_argument('--plan', required=True, type=Path)
    e.add_argument('--plan-hash', required=True)
    e.add_argument('--output', required=True, type=Path)
    e.add_argument('--start-index', type=int, default=1)
    e.add_argument('--count', type=int, default=2)
    e.add_argument('--independent-window', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        plan = freeze(args.output, gpu_index=args.gpu_index,
                      layernorm_followup=args.layernorm_followup)
        result = dict(plan_path=str(args.output / 'plan.json'), plan_hash=plan['hash'],
                      planned_runs=plan['planned_runs'])
    else:
        result = execute(args.plan, args.plan_hash, args.output,
                         start_index=args.start_index, count=args.count,
                         independent_window=args.independent_window)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
