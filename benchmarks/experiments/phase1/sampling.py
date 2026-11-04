"""Study-only wire policy, applied before reservation and request hashing.

Ordinary CLI/MCP behavior is unchanged. Every inherited study worker checks the
same immutable cell configuration before normalizing an actual bounded request.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path

from .plan import read

POLICY=dict(schema='phase1-wire-policy-v1',temperature=.1,max_tokens=32768,
            n=1,enable_thinking=True,reasoning_effort='medium')
FIELDS=('model','temperature','max_tokens','max_completion_tokens','n',
        'enable_thinking','chat_template_kwargs','reasoning_effort','stream')
DEFAULT_OUTPUT_CAP = 4096


def policy():return copy.deepcopy(POLICY)


def validate(value):
    try:equal=json.dumps(value,sort_keys=True,allow_nan=False)==json.dumps(POLICY,sort_keys=True)
    except (TypeError,ValueError):equal=False
    if not equal:raise ValueError('study requires the fixed wire sampling policy')
    return value


def normalized(body,value):
    validate(value)
    if not isinstance(body,dict):raise ValueError('model request must be an object')
    result=copy.deepcopy(body)
    result.pop('max_completion_tokens',None)
    result.pop('chat_template_kwargs',None)
    result.update({k:copy.deepcopy(v) for k,v in value.items() if k!='schema'})
    return result


def matches(body,value):
    # `reasoning_effort` 现在是策略里的一格（按用户实际配置取 medium），所以它出现在
    # body 里是**对的**；仍被禁的是那两个会顶掉上限与模板的别名。
    return (isinstance(body,dict) and all(type(body.get(k)) is type(v) and body[k]==v
                                         for k,v in value.items() if k!='schema')
            and not any(k in body for k in ('max_completion_tokens','chat_template_kwargs')))


def output_cap(config):
    """Use the registered limit, or the original cap for legacy integrations."""
    sampling = config.get('sampling')
    return sampling['max_tokens'] if sampling else DEFAULT_OUTPUT_CAP


def request_timeout(config):
    """Allow 16 seconds per 1,000 output tokens within the cell deadline."""
    return min(max(120.0, 16.0 * output_cap(config) / 1000.0),
               config['budget']['wall_clock_s'])


def _record(directory,participant,caller,before,after):
    item=dict(participant=participant,caller=caller,
        requested={k:before[k] for k in FIELDS if k in before},
        effective={k:after[k] for k in FIELDS if k in after},
        request_sha256=hashlib.sha256(json.dumps(after,ensure_ascii=False,allow_nan=False,
            **(dict(sort_keys=True,separators=(',',':')) if participant=='outer' else {})).encode()).hexdigest(),
        scope='sampling_intent_before_reservation_not_a_billable_call')
    path=Path(directory)/('sampling-'+participant+'.jsonl')
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(item,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_CREAT|os.O_WRONLY|os.O_APPEND,0o600)
    try:
        if os.write(fd,raw)!=len(raw):raise OSError('short sampling journal write')
    finally:os.close(fd)


def attach_gateway(gateway,config,directory):
    if 'sampling' not in config:return
    validate(config['sampling'])
    original=gateway._server.gateway._request
    def request(body,participant):
        fixed=normalized(body,config['sampling'])
        _record(directory,'outer','dsh',body,fixed)
        return original(fixed,participant)
    gateway._server.gateway._request=request


@contextmanager
def bounded(config=None):
    from operator_agent.execute.llm_client import LLMClient
    if config is None:
        path=os.environ.get('OPERATOR_AGENT_PHASE1_CELL_CONFIG')
        if not path:
            yield;return
        source=Path(path)
        expected=os.environ.get('OPERATOR_AGENT_PHASE1_CELL_SHA256')
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise ValueError('study cell configuration changed')
        config=read(source)
    if 'sampling' not in config:
        yield;return
    validate(config['sampling'])
    source=Path(config['run_dir']).parent/'cell-config.json'
    if source.is_symlink() or read(source)!=config:raise ValueError('study cell config binding mismatch')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    keys=dict(OPERATOR_AGENT_PHASE1_CELL_CONFIG=str(source),OPERATOR_AGENT_PHASE1_CELL_SHA256=digest)
    previous={k:os.environ.get(k) for k in keys}
    original=LLMClient._call_bounded_model
    def call(client,base_url,api_key,model,body,raw,caller):
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            raise ValueError('study sampling config changed before dispatch')
        fixed=normalized(body,config['sampling'])
        _record(config['run_dir'],'inner',caller,body,fixed)
        return original(client,base_url,api_key,model,fixed,raw,caller)
    LLMClient._call_bounded_model=call
    os.environ.update(keys)
    try:yield
    finally:
        LLMClient._call_bounded_model=original
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


@contextmanager
def autonomous_launcher(config):
    from operator_agent.execute import exec_guard
    if 'sampling' not in config:
        yield;return
    original=exec_guard.start_background_worker
    def launch(argv,*args,**kwargs):
        argv=list(argv)
        run_dir = Path(config['run_dir']).resolve()
        if (len(argv)!=4 or argv[1:3]!=['-m','operator_agent.app.autonomous']
                or Path(argv[3]).resolve()!=run_dir):
            raise ValueError('unexpected study autonomous launch')
        # The MCP tool has written its owned config, but has not started the
        # supervisor. Reuse the production worker's existing timeout option.
        from operator_agent.app.autonomous import _atomic_json
        worker_path = run_dir / 'autonomous.json'
        if worker_path.is_symlink():
            raise ValueError('study autonomous config must be a regular file')
        worker_config = read(worker_path)
        worker_config['llm_timeout'] = request_timeout(config)
        _atomic_json(worker_path, worker_config)
        argv[2]='benchmarks.experiments.phase1.study_autonomous'
        return original(argv,*args,**kwargs)
    exec_guard.start_background_worker=launch
    try:yield
    finally:exec_guard.start_background_worker=original
