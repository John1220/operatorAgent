"""Single independent verification; never calls a model or provisions hardware."""
import hashlib
import json
import os
from pathlib import Path
import sys

from operator_agent.app import hitl_tools as ht
from operator_agent.execute.task_clock import monotonic
from .connection import options
from .plan import read, write_once


def run(config):
    from operator_agent.execute.s5_verify_client import enabled
    if enabled():raise ValueError('inherited verification delegation refused')
    if 'study' in config:
        from .study_instance import check
        write_once(Path(config['run_dir']).parent/'final-instance-check.json',check(config))
    data=Path(config['source']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=config['source_hash']:
        raise ValueError('final source binding mismatch')
    work=Path(config['run_dir'])
    if ht.prepare(config['declaration'],work)['contract_hash']!=config['contract_hash']:
        raise ValueError('final contract binding mismatch')
    keys=dict(OPERATOR_AGENT_EXISTING_REMOTE_ONLY='1')
    for key,field in (('OPERATOR_AGENT_REQUIRED_GPU','index'),
                      ('OPERATOR_AGENT_REQUIRED_GPU_UUID','uuid'),('OPERATOR_AGENT_REQUIRED_GPU_NAME','name')):
        keys[key]=str(config['gpu'][field]) if config['backend']=='ssh' else None
    previous={key:os.environ.get(key) for key in keys}
    try:
        for key,value in keys.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        result=ht.verify(work,data.decode('utf-8'),backend=config['backend'],**options(config))
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    receipt=dict(schema='phase1-connected-final-v1',cell_id=config['cell_id'],plan_hash=config['plan_hash'],
        source_hash=config['source_hash'],contract_hash=config['contract_hash'],verification_count=1,
        execution_context=config['execution_context'],finished_monotonic=monotonic(),result=result)
    write_once(work/'final-result.json',receipt)
    return receipt


if __name__=='__main__':
    print(json.dumps(run(read(sys.argv[1])),ensure_ascii=False))
