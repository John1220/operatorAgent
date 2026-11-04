"""One independent verification with frozen source and contract bindings."""
import hashlib
import json
from pathlib import Path
import sys

from operator_agent.app import hitl_tools as ht
from operator_agent.execute.fake_backend import FakeBackend, FakeConfig
from operator_agent.execute.task_clock import monotonic

from .baseline import _backend_scope
from .plan import read, write_once


def run(config):
    from operator_agent.execute.s5_verify_client import enabled
    if enabled():
        raise ValueError('offline final evaluation refuses inherited S5 verification service')
    candidate = Path(config['source'])
    data = candidate.read_bytes()
    if hashlib.sha256(data).hexdigest() != config['source_hash']:
        raise ValueError('selected source changed before final evaluation')
    work = Path(config['run_dir'])
    prepared = ht.prepare(config['declaration'], work)
    if prepared['contract_hash'] != config['contract_hash']:
        raise ValueError('final contract does not match the frozen task')
    # Development fault injection is not repeated during independent evaluation.
    backend = FakeBackend(FakeConfig(profile=config['profile']))
    with _backend_scope(backend):
        result = ht.verify(work, data.decode('utf-8'), backend='fake')
    receipt = dict(schema='phase1-final-evaluation-v1', cell_id=config['cell_id'],
        plan_hash=config['plan_hash'], source_hash=config['source_hash'],
        contract_hash=config['contract_hash'], verification_count=1,
        finished_monotonic=monotonic(), result=result)
    write_once(work/'final-result.json', receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(run(read(sys.argv[1])), ensure_ascii=False))
