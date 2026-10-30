"""Formal-quality S5 comparison over the known-exposure quality tasks.

Pre-declared scope: the ten quality-set tasks whose exposure is already recorded
as historical, three arms each (A bare DSH, B DSH + core CLI, C DSH + core CLI +
plugin), real model through the budgeted Responses gateway, final answers judged
by the supervisor-side verification service on a pinned H20.

The four draft candidates with unknown exposure are deliberately excluded and
stay outside the denominator. Nothing here retries a task; a failed, timed-out or
unverified run is recorded as such and kept.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tempfile

from benchmarks.experiments.s5 import s5_arm_env, s5_runtime, s5_task_manifest, s5_verify_remote
from benchmarks.experiments.s5.s5_gateway_http import GatewayServer
from benchmarks.experiments.s5.s5_gateway_supervisor import SupervisedTransport
from benchmarks.experiments.s5.s5_development_service import DevelopmentService
from operator_agent import s5_shared_budget as shared
from operator_agent.run_contract import digest, protocol_identity

SCHEMA = 's5-workflow-plan-v2'
ARMS = ('A', 'B', 'C')
ROTATION = (('A', 'B', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'))
ENDPOINT_KEY = 'S5_RESPONSES_ENDPOINT'
DEFAULT_ENDPOINT = ('https://llm-4lvki6ksuwa2fuhf.cn-beijing.maas.aliyuncs.com'
                    '/compatible-mode/v1/responses')
MODEL = 'deepseek-v4.1-flash-1017'
GATEWAY_HOST = 'algorithm-h20'
GATEWAY_GPU = dict(index=7, uuid='GPU-52ca6155-d8bc-a416-847a-6f0c71db1552', name='NVIDIA H20')
# User-selected future protocol; historical v6 none/8192 remains unchanged.
OUTPUT_TOKENS = 32768
CONTEXT_WINDOW = 1000000  # model metadata in the current DSH provider profile
REQUEST_WALL_SECONDS = 900
RESPONSE_BYTES = 64 * 1024 * 1024
BUDGET = dict(max_calls=24, max_output_tokens=OUTPUT_TOKENS,
              total_output_tokens=24 * OUTPUT_TOKENS, wall_clock_s=2400)
VERIFY_TIMEOUT = 600
DEVELOPMENT = dict(max_development=2, final_reserve=1,
                   feedback='same bounded L1/L2 projection for A/B/C',
                   source_revision='outer agent; core repair/run/bench disabled',
                   final_feedback_to_agent=False)


def runtime_identity():
    root = Path(__file__).resolve().parents[3]
    files = ['benchmarks/experiments/s5/s5_formal_run.py', 'benchmarks/experiments/s5/s5_arm_env.py',
             'benchmarks/experiments/s5/s5_development_service.py', 'benchmarks/experiments/s5/s5_verify_remote.py',
             'benchmarks/experiments/s5/s5_development_evidence.py', 'benchmarks/experiments/s5/s5_gateway_http.py',
             'benchmarks/experiments/s5/s5_gateway_policy.py', 'benchmarks/experiments/s5/s5_gateway_supervisor.py',
             'benchmarks/experiments/s5/s5_gateway_transport.py',
             'benchmarks/experiments/s5/s5_seatbelt.py', 'benchmarks/experiments/s5/s5_isolation_probe.py',
             'benchmarks/experiments/s5/s5_runtime.py', 'operator_agent/cli.py',
             'benchmarks/experiments/s5/s5_outcomes.py',
             'benchmarks/experiments/s5/s5_minimal_tasks.py', 'benchmarks/experiments/s5/s5_minimal_summary.py',
             'operator_agent/mcp_server.py', 'integrations/dsh-operator/index.mjs']
    return dict(protocol=protocol_identity(), harness={name: hashlib.sha256(
        (root / name).read_bytes()).hexdigest() for name in files})


def _write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _quality_tasks():
    manifest = s5_task_manifest.manifest()
    rows = [row for row in manifest['tasks'] if row.get('exposure') == 'known_historical']
    if len(rows) != 10:
        raise ValueError('expected the ten known-exposure tasks')
    return manifest, rows


def task_text(row):
    shape = ', '.join(str(dim) for dim in row['shape'])
    if row['kernel_type'] == 'matmul':
        m, n, k = row['shape']
        body = ('Define `def kernel(A, B)` returning the matrix product C = A @ B '
                'for %s tensors of shape A[%s] and B[%s], on CUDA. ' %
                (row['precision'], f'{m}, {k}', f'{k}, {n}'))
    elif row['kernel_type'] == 'layernorm':
        body = ('Define `def kernel(X)` returning layer normalization of a %s tensor '
                'of shape [%s] over the last axis with epsilon 1e-5, on CUDA. ' %
                (row['precision'], shape))
    elif row['kernel_type'] == 'softmax':
        body = ('Define `def kernel(X)` returning the softmax of a %s tensor of shape '
                '[%s] over the last axis, on CUDA. ' % (row['precision'], shape))
    else:
        raise ValueError('unsupported kernel type')
    return ('Write a Triton kernel in Python. ' + body +
            'The kernel must be numerically correct on the full input including '
            'non-contiguous layouts. '
            'This environment has no GPU and no Torch/Triton installation: do not try to '
            'install packages, locate another interpreter, or run the kernel locally. '
            'Use the supplied supervised development verification entry to test '
            'candidates and revise them from its feedback within the stated budget. '
            'The final independent evaluation is separate and gives no further feedback. '
            'Return only the final source code, '
            'no explanation.')


def check_task_inputs(row, contract):
    """Bind the prompt's operand shapes/dtype to the actual service primary input.

    The manifest's matmul shape is (M,N,K), never a tensor shape. The production
    suite is intentionally distinct from the historical quality evaluation set.
    """
    if row['kernel_type'] == 'matmul':
        m, n, k = row['shape']
        expected = [('A', [m, k], row['precision']), ('B', [k, n], row['precision'])]
    else:
        expected = [('X', list(row['shape']), row['precision'])]
    actual = [(x['name'], x['shape'], x['dtype'])
              for x in contract['contract']['cases'][0]['tensors']]
    if actual != expected:
        raise ValueError('prompt and service operand shapes/dtype disagree')
    return dict(operands=[dict(name=name, shape=shape, dtype=dtype)
                          for name, shape, dtype in expected],
                layout_policy='must support contiguous and non-contiguous inputs',
                evaluation_regime='production builtin L2 suite, not historical holdouts')


def declaration(row):
    return dict(request='%s %s triton' % (row['kernel_type'], row['precision']),
                shapes=[list(row['shape'])], hardware='nvidia_gpu', dsl='triton',
                precision=row['precision'], min_speedup=1.0)


def freeze(output, *, endpoint=None, model=MODEL, budget=None, case_ids=None, preflight=False,
           read_isolation=None, task_manifest=None, repeats=1, gpu=None, resource_policy=None):
    """Freeze the task list, service plans, order and budgets exactly once."""
    endpoint = endpoint or os.environ.get(ENDPOINT_KEY, DEFAULT_ENDPOINT)
    budget = dict(BUDGET if budget is None else budget)
    gpu = dict(GATEWAY_GPU if gpu is None else gpu)
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError('invalid repeats')
    if task_manifest is None:
        manifest, rows = _quality_tasks()
    else:
        manifest, rows = task_manifest, task_manifest['tasks']
        if digest({k:v for k,v in manifest.items()if k!='hash'})!=manifest.get('hash'):
            raise ValueError('task manifest hash mismatch')
    if case_ids is not None:
        selected = {row['case_id']: row for row in rows}
        if (not preflight or len(case_ids) != 3 or len(set(case_ids)) != 3
                or any(name not in selected for name in case_ids)):
            raise ValueError('excluded workflow preflight requires three distinct known tasks')
        rows = [selected[name] for name in case_ids]
    elif preflight:
        raise ValueError('preflight tasks must be explicitly selected before freezing')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    services = {}
    for row in rows:
        case = row['case_id']
        sha = s5_verify_remote.freeze(root / 'services' / case, declaration=declaration(row),
                                      backend='h20', host=GATEWAY_HOST,
                                      timeout=VERIFY_TIMEOUT, gpu_index=gpu['index'],
                                      gpu_uuid=gpu['uuid'], gpu_name=gpu['name'])
        services[case] = dict(sha256=sha, declaration=declaration(row),
                              contract_hash=_read(root / 'services' / case / 'plan.json')['contract_hash'],
                              inputs=check_task_inputs(row, _read(root / 'services' / case / 'task' / 'contract.json')))
    cases = []
    for index, row in enumerate(rows):
        order = list(ROTATION[index % len(ROTATION)])
        cases.append(dict(case_id=row['case_id'], kernel_type=row['kernel_type'],
                          shape=list(row['shape']), precision=row['precision'],
                          layout=row['layout'], text=task_text(row), order=order,
                          input_semantics=services[row['case_id']]['inputs'],
                          service_sha256=services[row['case_id']]['sha256'],
                          service_contract=services[row['case_id']]['contract_hash']))
    plan = dict(schema=SCHEMA, status='frozen', formal_s5_credit=False,
                preflight=preflight, counts_toward_denominator=False,
                scope=('three known-exposure tasks x three arms; excluded workflow preflight'
                       if preflight else 'prospective task manifest with repeated ABC runs'
                       if task_manifest is not None else 'ten known-exposure quality tasks x three arms; '
                       'not the four unknown-exposure draft candidates'),
                manifest_hash=manifest['hash'], model=dict(id=model, protocol='responses',
                                                           endpoint=endpoint,
                                                           reasoning_effort='medium',
                                                           context_window=CONTEXT_WINDOW,
                                                           request_wall_seconds=REQUEST_WALL_SECONDS,
                                                           max_response_bytes=RESPONSE_BYTES),
                arms=list(ARMS), budget=budget, wall_clock_s=budget['wall_clock_s'],
                development=dict(DEVELOPMENT), implementation=runtime_identity(),
                read_isolation=read_isolation,
                resource_policy=resource_policy,
                verification=dict(backend='h20', host=GATEWAY_HOST, gpu=gpu,
                                  stop_after='L2', timeout=VERIFY_TIMEOUT),
                cases=cases, repeats=repeats, planned_runs=len(cases) * len(ARMS) * repeats,
                schedule=[dict(case_id=case['case_id'],repeat_id=r+1,arm=arm)
                          for r in range(repeats) for i,case in enumerate(cases)
                          for arm in ROTATION[(i+r)%len(ROTATION)]],
                exclusion_note='unknown-exposure candidates and performance claims are '
                               'outside this denominator; no selective reruns',
                runtime=dict(deps=s5_runtime.tree_digest(s5_runtime.deps_dir())
                             if s5_runtime.available() else None,
                             python=sys.version.split()[0]))
    plan['hash'] = digest(plan)
    _write_once(root / 'plan.json', plan)
    return plan


def load_plan(path):
    plan = _read(path)
    stored = plan.pop('hash')
    if digest(plan) != stored:
        raise ValueError('plan hash mismatch')
    plan['hash'] = stored
    if plan.get('schema') not in (SCHEMA, 's5-formal-plan-v1') or plan.get('planned_runs') != len(plan['cases']) * 3 * plan.get('repeats',1):
        raise ValueError('invalid plan')
    if 'schedule' in plan:
        expected={(c['case_id'],r,a)for c in plan['cases']for r in range(1,plan.get('repeats',1)+1)for a in ARMS}
        slots=[(r['case_id'],r['repeat_id'],r['arm'])for r in plan['schedule']]
        if len(slots)!=len(expected)or set(slots)!=expected:raise ValueError('invalid schedule')
    return plan


def _audit_repo_access(events):
    """Count tool calls whose inputs mention the repository path. Not isolation."""
    hits = 0
    tools = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict) or event.get('type') != 'tool_call':
            continue
        tools.append(event.get('tool'))
        if 'repos/operatorAgent' in json.dumps(event.get('input'), ensure_ascii=False):
            hits += 1
    return dict(tool_calls=len(tools), tools=tools, repo_path_mentions=hits)


def normalize_candidate(text):
    """Strip one markdown code fence; the same rule applies to every arm."""
    import re
    match = re.fullmatch(r'\s*```[A-Za-z0-9_+.-]*\n(.*?)\n```\s*', text, re.S)
    if match and match.group(1).strip():
        return match.group(1) + '\n', 'fenced_code_block'
    return text, 'verbatim'


def _final_text(events):
    texts = [event.get('text') for event in events
             if isinstance(event, dict) and event.get('type') == 'final']
    return texts[0] if len(texts) == 1 and isinstance(texts[0], str) else None


def workflow_instructions(arm, run_dir, *, backend, max_development):
    common = (f'You may use at most {max_development} supervised development L2 '
              'submissions. Failed/unknown attempts consume quota too. Read status '
              'and feedback, revise candidate source yourself; no inner generation, '
              'repair/run/bench, direct SSH or alternative verifier. Final evaluation '
              'has a separate one-shot reservation; do not request it as a tool. ')
    if arm == 'A':
        return common + 'Write candidate.py, then run: s5-verify --code-file candidate.py'
    # B/C receive exactly the same CLI usage note. Only C additionally discovers
    # the actual MCP tool schemas/instructions through the installed MCP client.
    return common + (f'The task is already prepared at {run_dir}. Use: '
        f'operator-agent verify --run-dir {json.dumps(str(run_dir))} '
        f'--code-file candidate.py --stop-after L2 --backend {backend}. '
        'Backend/contract are fixed by the supervisor; do not supply host or credentials.')


def redacted_launch(value):
    if isinstance(value, dict):
        return {k: ('[ephemeral capability redacted]' if k.endswith('DEV_TOKEN')
                    else redacted_launch(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redacted_launch(v) for v in value]
    return value


def run_one(plan, case, arm, output, api_key, *, repeat_id=None):
    """One workflow run, with supervised development and independent final L2."""
    if plan.get('schema') != SCHEMA or plan.get('implementation') != runtime_identity():
        raise ValueError('execution requires a newly frozen workflow plan; historical plans are read-only')
    isolation=plan.get('read_isolation')
    if isolation is not None:
        from benchmarks.experiments.s5.s5_isolation_probe import load_readiness
        if load_readiness(isolation['report_path'])!=isolation:
            raise ValueError('read isolation readiness drift')
    if repeat_id is not None and (type(repeat_id)is not int or not 1<=repeat_id<=plan.get('repeats',1)):
        raise ValueError('unplanned repeat')
    run_dir = Path(output) / 'runs' / case['case_id']
    if repeat_id is not None:run_dir=run_dir/('repeat-%02d'%repeat_id)
    run_dir=run_dir/arm
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    (Path(output) / 'records').mkdir(parents=True, exist_ok=True, mode=0o700)
    record = dict(schema='s5-workflow-run-v2', case_id=case['case_id'], arm=arm,
                  plan_hash=plan['hash'], status='unknown', execution_state='unknown',
                  error=None, reason=None, model_calls=None, wall_seconds=None,
                  candidate_sha256=None, verification=None, tools=None, development=None,
                  reasoning_effort=plan['model']['reasoning_effort'],
                  repo_path_mentions=None, real_model_called=None, gpu_used=None,
                  formal_s5_credit=False, isolation_verified=False)
    record['repeat_id']=repeat_id or 1
    started = time.monotonic()
    task_id = '%s-%s' % (case['case_id'], arm)
    if repeat_id is not None:task_id='repeat-'+digest([case['case_id'],repeat_id,arm])[:24]
    ledger = run_dir / 'budget.sqlite'
    service = None
    built = None
    events = []
    gateway=None
    transport=None
    try:
        service_path = Path(output) / 'services' / case['case_id'] / 'plan.json'
        service_plan = s5_verify_remote.load_plan(service_path, case['service_sha256'])
        shared.create(ledger, task_id=task_id, max_calls=plan['budget']['max_calls'],
                      max_output_tokens=plan['budget']['max_output_tokens'],
                      total_output_tokens=plan['budget']['total_output_tokens'],
                      wall_clock_s=plan['budget']['wall_clock_s'])
        with ExitStack() as stack:
            service = stack.enter_context(DevelopmentService(service_path, case['service_sha256'],
                run_dir / 'supervised-verification', arm=arm,
                max_development=plan['development']['max_development'],
                wall_seconds=plan['budget']['wall_clock_s']))
            response_limit = plan['model'].get('max_response_bytes', RESPONSE_BYTES)
            transport = SupervisedTransport(plan['model']['endpoint'], api_key,
                wall_seconds=plan['model'].get('request_wall_seconds', REQUEST_WALL_SECONDS),
                max_request_bytes=1048576, max_response_bytes=response_limit)
            gateway = stack.enter_context(GatewayServer(ledger, task_id=task_id,
                model=plan['model']['id'], participant='outer', adapter=transport,
                protocol='responses', max_request_bytes=1048576, max_response_bytes=response_limit,
                max_chunks=1048576, max_calls_limit=plan['budget']['max_calls'],
                reasoning_effort=plan['model']['reasoning_effort']))
            development = dict(url=service.url, token=service.token,
                contract=service_plan['contract_hash'], backend=service_plan['mode'],
                timeout=service_plan['timeout'] + 10)
            participant_dir=(Path(tempfile.mkdtemp(prefix='operator-s5-seatbelt-'))/arm
                             if isolation is not None else run_dir)
            built = s5_arm_env.build(arm, base_url=gateway.base_url, arm_dir=participant_dir,
                api='responses', model=plan['model']['id'],
                max_tokens=plan['budget']['max_output_tokens'], tools=True, development=development,
                context_window=plan['model'].get('context_window', CONTEXT_WINDOW),
                reasoning_effort=plan['model']['reasoning_effort'],
                isolation_runtime=isolation['runtime'] if isolation is not None else None)
            if isolation is not None:
                built['isolation']['file_probe_passed']=True
                built['isolation']['proof_report_sha256']=isolation['report_sha256']
            Path(built['home']).mkdir(parents=True, exist_ok=True, mode=0o700)
            Path(built['work']).mkdir(parents=True, exist_ok=True, mode=0o700)
            prepared_run = Path(built['work']) / 'task'
            if arm != 'A':
                from operator_agent.hitl_tools import prepare
                prepared = prepare(service_plan['declaration'], prepared_run)
                if prepared['contract_hash'] != service_plan['contract_hash']:
                    raise ValueError('prepared development contract drift')
            task = case['text'] + '\n' + workflow_instructions(arm, prepared_run,
                backend=service_plan['mode'], max_development=plan['development']['max_development'])
            _write_once(run_dir / 'launch.json', redacted_launch(built))
            _write_once(run_dir / 'task.json', dict(text=task))
            response = s5_arm_env.start_headless(built, task,
                timeout=plan['budget']['wall_clock_s'], capture_dir=run_dir/'headless-output')
            events = response['events']
            usage={}
            for event in events:
                if event.get('phase')=='step_end':
                    for key,value in (event.get('usage')or{}).items():
                        if type(value)is int:usage[key]=usage.get(key,0)+value
            record['dsh_usage']=usage or None
            record['usage_provenance']='DSH step_end, not raw provider verified usage'
            endings=[e for e in events if e.get('phase')=='turn_end']
            record['terminal_kind']=(endings[-1].get('reason')or{}).get('kind') if endings else None
            _write_once(run_dir / 'events.json', events)
            _write_once(run_dir / 'stderr.json', dict(text=response['stderr']))
            audit = _audit_repo_access(events)
            record['tools'] = audit['tools']
            record['repo_path_mentions'] = audit['repo_path_mentions']
            if response['returncode'] != 0:
                record['execution_state'] = 'interrupted'
                record['reason'] = 'headless interrupted; inspect terminal events (not inferred budget exhaustion)'
            else:
                candidate = _final_text(events)
                if not candidate or not candidate.strip():
                    record['execution_state'] = 'interrupted'
                    record['reason'] = 'no single nonempty final answer'
                else:
                    record['execution_state'] = 'finished'
                    (run_dir / 'raw-answer.txt').write_text(candidate, encoding='utf-8')
                    candidate, normalization = normalize_candidate(candidate)
                    record['normalization'] = normalization
                    (run_dir / 'candidate.txt').write_text(candidate, encoding='utf-8')
                    from operator_agent.run_contract import code_hash
                    record['candidate_sha256'] = code_hash(candidate)
                    # The model has ended. This result is never fed back into DSH.
                    verdict = service.final(candidate)
                    record['verification'] = dict(verdict,
                        service_plan_sha256=case['service_sha256'])
                    record['status'] = verdict['status']
    except Exception as exc:
        record['error'] = type(exc).__name__
        record['reason'] = str(exc)[:200]
        record['execution_state'] = ('interrupted' if record['execution_state'] == 'unknown'
                                     else record['execution_state'])
        record['status'] = 'unknown'
    finally:
        if (run_dir/'headless-output/metadata.json').is_file():
            record['headless_output']=_read(run_dir/'headless-output/metadata.json')
        if gateway is not None:
            record['budget_refusal']=gateway._server.gateway.last_budget_refusal
        if transport is not None:
            diagnostics=transport.failure_snapshot()
            record['gateway_diagnostics']=diagnostics
            _write_once(run_dir/'gateway-diagnostics.json',diagnostics)
        # Persist costs and elapsed time even on early exits or interrupts.
        if ledger.exists():
            snapshot = shared.snapshot(ledger, expected_task_id=task_id)
            _write_once(run_dir / 'budget.json', snapshot)
            record['model_calls'] = dict(reserved=snapshot['reserved_calls'],
                completed=snapshot['completed_calls'], aborted=snapshot['aborted'],
                usage=[call.get('api_usage') for call in snapshot['calls']])
        if service is not None:
            record['development'] = service.snapshot()
            _write_once(run_dir / 'development.json', record['development'])
            from benchmarks.experiments.s5.s5_development_evidence import audit
            record['development_audit'] = audit(events, record['development'], arm=arm)
            _write_once(run_dir / 'development-audit.json', record['development_audit'])
        if built is not None:
            s5_arm_env.cleanup_development(built)
            if built.get('isolation'):
                from benchmarks.experiments.s5.s5_seatbelt import archive_work
                record['workspace_archive']=archive_work(built,run_dir/'participant-work')
                record['read_boundary']='macos-outer-seatbelt-whitelist'
        record['wall_seconds'] = round(time.monotonic() - started, 3)
        from benchmarks.experiments.s5.s5_outcomes import classify
        record['outcome_class']=classify(record)
    return record


def run(plan_path, output, *, only_case=None, api_key=None):
    plan = load_plan(plan_path)
    if plan.get('repeats',1)!=1:
        raise ValueError('use supervised s5_workflow_preflight driver for repeated studies')
    if plan.get('schema') != SCHEMA or plan.get('implementation') != runtime_identity():
        raise ValueError('historical or drifted plan cannot authorize new execution')
    output = Path(output).resolve()
    (output / 'records').mkdir(parents=True, exist_ok=True, mode=0o700)
    api_key = api_key or _read_key()
    cases = [case for case in plan['cases'] if only_case in (None, case['case_id'])]
    done = 0
    for case in cases:
        for arm in case['order']:
            record_path = output / 'records' / ('%s-%s.json' % (case['case_id'], arm))
            if record_path.exists():
                continue
            record = run_one(plan, case, arm, output, api_key)
            _write_once(record_path, record)
            done += 1
            print('%s %s %s calls=%s wall=%s' % (case['case_id'], arm, record['status'],
                  (record['model_calls'] or {}).get('reserved'), record['wall_seconds']),
                  flush=True)
    summary = aggregate(plan, load_records(output))
    # Aggregate is derived from the records, so it is replaced atomically on resume.
    temp = output / 'aggregate.json.tmp'
    temp.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + '\n',
                    encoding='utf-8')
    temp.replace(output / 'aggregate.json')
    return summary


def load_records(output):
    records = []
    for path in sorted((Path(output) / 'records').glob('*.json')):
        records.append(_read(path))
    return records


def aggregate(plan, records):
    by_arm = {arm: [] for arm in ARMS}
    for record in records:
        if record.get('arm') not in by_arm:
            raise ValueError('unknown arm in records')
        by_arm[record['arm']].append(record)
    groups = {}
    for arm in ARMS:
        rows = by_arm[arm]
        counts = {state: sum(1 for r in rows if r['status'] == state)
                  for state in ('passed', 'failed', 'unknown')}
        calls = [r['model_calls']['reserved'] for r in rows if r.get('model_calls')]
        walls = [r['wall_seconds'] for r in rows if r.get('wall_seconds') is not None]
        groups[arm] = dict(planned=len(plan['cases']), executed=len(rows), counts=counts,
                           calls=(sum(calls) if calls else None),
                           wall_seconds=(round(sum(walls), 2) if walls else None),
                           repo_path_mentions=sum(r.get('repo_path_mentions') or 0 for r in rows))
    lookup = {(r['case_id'], r['arm']): r for r in records}
    pairs = {}
    for left, right in (('B', 'A'), ('C', 'B')):
        rows = []
        for case in plan['cases']:
            a, b = lookup.get((case['case_id'], left)), lookup.get((case['case_id'], right))
            both = (a and b and a['status'] in ('passed', 'failed')
                    and b['status'] in ('passed', 'failed'))
            rows.append(dict(case_id=case['case_id'],
                             difference=((a['status'] == 'passed') - (b['status'] == 'passed'))
                             if both else None))
        known = [row['difference'] for row in rows if row['difference'] is not None]
        pairs[left + '-minus-' + right] = dict(
            paired=len(known), unpaired=len(rows) - len(known),
            wins=sum(1 for value in known if value > 0),
            losses=sum(1 for value in known if value < 0),
            ties=sum(1 for value in known if value == 0))
    body = dict(schema='s5-formal-aggregate-v1', plan_hash=plan['hash'],
                planned_runs=plan['planned_runs'], executed_runs=len(records),
                counts={state: sum(g['counts'][state] for g in groups.values())
                        for state in ('passed', 'failed', 'unknown')},
                groups=groups, pairs=pairs,
                exclusion_note=plan['exclusion_note'],
                limits=['no performance measurement', 'A is not OS-isolated from the repository',
                        'provider token usage is recorded only when the raw stream reports it'])
    return dict(body, hash=digest(body))


def _read_key():
    text = Path(os.environ.get('S5_CREDENTIALS', str(Path.home() / '.dsh/.credentials.yaml'))).read_text()
    import re
    match = re.search(r'^\s+ALIYUN_API_KEY:\s*(.+)$', text, re.M)
    if not match:
        raise ValueError('credential unavailable')
    key = match.group(1).strip().strip('"').strip("'")
    if not key or '\n' in key:
        raise ValueError('invalid credential')
    return key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, help='create the frozen plan in this directory')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', help='run only this case id')
    args = parser.parse_args(argv)
    if args.freeze:
        plan = freeze(args.freeze)
        print(json.dumps(dict(plan=plan['hash'], runs=plan['planned_runs']), sort_keys=True))
        return 0
    if not (args.plan and args.output):
        raise SystemExit('need --plan and --output')
    summary = run(args.plan, args.output, only_case=args.case)
    print(json.dumps(dict(hash=summary['hash'], executed=summary['executed_runs'],
                          counts=summary['counts']), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
