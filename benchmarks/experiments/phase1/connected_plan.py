"""Explicitly configured model/SSH/DSH integration plans; freezing is offline."""
import hashlib
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from operator_agent.execute.task_budget import TaskBudgetConfig
from . import plan as offline
from .support import check, matrix

SCHEMA = 'phase1-connected-plan-v1'
ENTRIES = dict(A='baseline_agent', B='dsh_operator_run')
SAMPLING = 'not_established; inspect retained actual requests before study freeze'


def context(value):
    return dict(mode=value['backend'], model_mode=value['model']['transport'],
        test_only=value['backend']=='fake' or value['model']['transport']=='local_fixture',
        gpu_acceptance=False, performance_evidence=False)


def implementation_identity():
    value = offline.implementation_identity()
    paths = list((offline.REPO/'benchmarks/experiments/s5').glob('*.py'))
    paths += list((offline.REPO/'integrations/dsh-operator').glob('*'))
    value['entry_dependencies'] = {str(p.relative_to(offline.REPO)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if p.is_file()}
    return value


def host_identity(entry):
    if entry!='dsh':return None
    from .host import runtime_identity
    return runtime_identity()


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or any(c in value for c in '\r\n\x00'):
        raise ValueError('invalid '+name)
    return value


def _normalize(config):
    allowed = {'backend', 'model', 'connection', 'gpu', 'probe', 'tasks', 'budget',
               'max_rounds', 'final_wall_clock_s', 'entry_b','entry_a','sampling'}
    if not isinstance(config, dict) or set(config)-allowed:
        raise ValueError('unknown connected configuration fields')
    backend = config.get('backend', 'fake')
    if backend not in ('fake', 'ssh') or config.get('entry_b', 'dsh') not in ('core', 'dsh'):
        raise ValueError('explicit fake/ssh backend and core/dsh entry are required')
    if config.get('entry_a','loop') not in ('loop','dsh_native'):
        raise ValueError('unknown baseline entry')
    if 'sampling' in config:
        from .sampling import validate
        validate(config['sampling'])
    model = config.get('model')
    if (not isinstance(model, dict) or set(model)-{'base_url','id','api_key_env','temperature','transport'}
            or not {'base_url','id'} <= set(model)):
        raise ValueError('model requires base_url/id; credentials belong only in the runtime environment')
    endpoint = _text(model['base_url'], 'model base URL').rstrip('/')
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or not parsed.path.rstrip('/').endswith('/v1')):
        raise ValueError('model URL must end in /v1 and contain no credentials/query/fragment')
    parsed.port
    temperature = model.get('temperature', .2)
    if type(temperature) not in (int,float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError('invalid model temperature')
    credential = model.get('api_key_env', 'LLM_API_KEY')
    if not isinstance(credential, str) or not re.fullmatch('[A-Z][A-Z0-9_]{1,63}', credential):
        raise ValueError('invalid credential environment reference')
    transport = model.get('transport', 'live')
    if transport not in ('live','local_fixture'):
        raise ValueError('model transport must be live or local_fixture')
    if transport == 'local_fixture' and (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'):
        raise ValueError('local fixture transport requires an explicit IPv4 loopback URL')
    model_id=_text(model['id'],'model id')
    if len(model_id)>128:raise ValueError('model id exceeds DSH entry limit')
    model = dict(base_url=endpoint, id=model_id, api_key_env=credential,
        temperature=temperature, transport=transport, protocol='chat_completions',
        endpoint_sha256=hashlib.sha256(endpoint.encode()).hexdigest())
    connection = config.get('connection', {})
    if not isinstance(connection, dict) or set(connection)-{'host','port','user','key_path','work_dir','timeout'}:
        raise ValueError('unknown connection fields')
    gpu, probe = config.get('gpu'), config.get('probe')
    if backend == 'fake':
        if connection or gpu is not None or probe is not None:
            raise ValueError('fake backend refuses SSH, GPU pin and card probe options')
    else:
        if (not connection.get('host') or not connection.get('work_dir')
                or not isinstance(connection['work_dir'],str)
                or not connection['work_dir'].startswith('/') or connection['work_dir']=='/'
                or '..' in Path(connection['work_dir']).parts):
            raise ValueError('SSH requires host and an absolute dedicated remote work_dir')
        _text(connection['work_dir'],'remote work directory')
        if 'key_path' in connection:
            _text(connection['key_path'],'SSH key path')
        timeout=connection.get('timeout',300)
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0 < timeout <= 660:
            raise ValueError('SSH timeout must be positive and at most 660 seconds')
        for key in ('host','user'):
            if key in connection and not re.fullmatch('[A-Za-z0-9_.@-]+', _text(connection[key],key)):
                raise ValueError('invalid SSH '+key)
            if key in connection and connection[key].startswith('-'):
                raise ValueError('SSH options cannot be supplied as '+key)
        if 'port' in connection and (type(connection['port']) is not int or not 1 <= connection['port'] <= 65535):
            raise ValueError('invalid SSH port')
        if (not isinstance(gpu,dict) or set(gpu) != {'index','uuid','name'}
                or type(gpu['index']) is not int or gpu['index'] < 0):
            raise ValueError('GPU index, UUID and name are all required')
        _text(gpu['uuid'],'GPU UUID'); _text(gpu['name'],'GPU name')
        if probe is None:
            raise ValueError('SSH plan requires measured card probe JSON before freezing')
        probe=str(Path(_text(probe,'card probe path')).resolve())
    budget = TaskBudgetConfig.from_dict(config.get('budget') or dict(
        wall_clock_s=2400., max_llm_calls=30, oracle_total=20)).as_dict()
    if budget['wall_clock_s'] > 2400 or budget['max_llm_calls'] > 30 or budget['oracle_total'] > 20:
        raise ValueError('connected budget exceeds the Phase 1 limits')
    rounds = config.get('max_rounds', 2)
    if type(rounds) is not int or not 1 <= rounds <= 8:
        raise ValueError('max_rounds must be 1..8')
    final_wall = config.get('final_wall_clock_s', 120.)
    TaskBudgetConfig(wall_clock_s=final_wall)
    if final_wall > 660:
        raise ValueError('independent final evaluation is bounded at 660 seconds')
    tasks = config.get('tasks')
    if not isinstance(tasks,list) or not 1 <= len(tasks) <= 4:
        raise ValueError('freeze one to four explicitly declared integration tasks')
    seen, normalized = set(), []
    for row in tasks:
        if not isinstance(row,dict) or set(row) != {'task_id','declaration'}:
            raise ValueError('task requires task_id and declaration')
        task_id = row['task_id']
        if not isinstance(task_id,str) or not re.fullmatch('[a-z][a-z0-9-]{0,39}',task_id) or task_id in seen:
            raise ValueError('invalid or duplicate task id')
        task, contract = check(row['declaration'], backend)
        normalized.append(dict(row, contract_hash=contract.hash, kernel_type=task.source.kernel_type))
        seen.add(task_id)
    result=dict(backend=backend, model=model, connection=connection, gpu=gpu, probe=probe,
        tasks=normalized, budget=budget, max_rounds=rounds, final_wall_clock_s=final_wall,
        entry_b=config.get('entry_b','dsh'))
    for key in ('entry_a','sampling'):
        if key in config:result[key]=config[key]
    return result


def _probe(path, gpu):
    from benchmarks.probes.unknown_card_gates import assess
    source = Path(path).resolve()
    value = offline.read(source)
    if value.get('device',{}).get('uuid') != gpu['uuid']:
        raise ValueError('card probe UUID differs from the frozen GPU')
    gates = assess(value)['gates']
    if len(gates) != 5 or any(v['verdict'] != 'pass' for v in gates.values()):
        raise ValueError('all five card gates must pass before an SSH plan can be frozen')
    return dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                data=value, gates=gates)


def create(config):
    value = _normalize(config)
    probe = _probe(value['probe'],value['gpu']) if value['backend']=='ssh' else None
    plan = dict(schema=SCHEMA, plan_id='phase1-entry-integration-v1',
        target_plan_id='phase1-agent-comparison-v2', **value,
        card_probe=probe, schedule=offline.schedule(value['tasks']),
        execution_context=context(value),
        formal_comparison_credit=False, dsh_entry_verified=False,
        sampling_comparability=SAMPLING,
        entries=dict(A='benchmarks.experiments.phase1.connected_worker',
                     B='benchmarks.experiments.phase1.connected_worker'),
        support=matrix(), implementation=implementation_identity(),
        host_runtime=host_identity('dsh' if value.get('entry_a')=='dsh_native' else value['entry_b']))
    plan['hash'] = offline.digest(plan)
    return plan


def freeze(output, config):
    plan=create(config)
    root = offline.fresh_directory(output)
    offline.write_once(root/'plan.json',plan)
    return plan


def validate(plan, *, check_implementation=True):
    if not isinstance(plan,dict) or plan.get('schema') != SCHEMA:
        raise ValueError('not a connected integration plan')
    if offline.digest({k:v for k,v in plan.items() if k!='hash'}) != plan.get('hash'):
        raise ValueError('connected plan hash mismatch')
    raw = {k:plan[k] for k in ('backend','connection','gpu','probe','budget','max_rounds',
                              'final_wall_clock_s','entry_b')}
    raw['model'] = {k:v for k,v in plan['model'].items() if k in
                   ('base_url','id','api_key_env','temperature','transport')}
    for key in ('entry_a','sampling'):
        if key in plan:raw[key]=plan[key]
    raw['tasks'] = [{k:r[k] for k in ('task_id','declaration')} for r in plan['tasks']]
    normalized = _normalize(raw)
    if any(plan.get(k)!=v for k,v in normalized.items()) or plan.get('schedule') != offline.schedule(plan['tasks']):
        raise ValueError('connected task/model/schedule identity mismatch')
    expected = dict(A='benchmarks.experiments.phase1.connected_worker', B='benchmarks.experiments.phase1.connected_worker')
    if (plan.get('entries') != expected or plan.get('formal_comparison_credit') is not False
            or plan.get('dsh_entry_verified') is not False or plan.get('support') != matrix()
            or plan.get('execution_context') != context(normalized)
            or plan.get('sampling_comparability') != SAMPLING
            or plan.get('plan_id') != 'phase1-entry-integration-v1'
            or plan.get('target_plan_id') != 'phase1-agent-comparison-v2'):
        raise ValueError('integration plan cannot claim formal/installed-host acceptance')
    if check_implementation and plan['implementation'] != implementation_identity():
        raise ValueError('implementation drift; freeze a new integration plan')
    if check_implementation and plan.get('host_runtime') != host_identity('dsh' if plan.get('entry_a')=='dsh_native' else plan['entry_b']):
        raise ValueError('installed DSH runtime drift; freeze a new integration plan')
    if plan['backend']=='ssh' and _probe(plan['probe'],plan['gpu']) != plan['card_probe']:
        raise ValueError('measured card probe changed')
    if 'study' in plan:
        from .study import validate_binding
        validate_binding(plan,check_identity=check_implementation)
    return plan


def load(path, **kwargs):
    return validate(offline.read(path), **kwargs)
