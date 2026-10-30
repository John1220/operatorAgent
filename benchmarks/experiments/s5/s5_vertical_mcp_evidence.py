"""Cross-check one offline DSH → real MCP → mock verifier experiment.

All records are local and same-user; this checks consistency, not hostile
tamper resistance, a network boundary, independent verification or GPU proof.
Missing, malformed and inconsistent evidence always yields unknown.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from benchmarks.experiments.s5.s5_vertical_validation import GOOD_FIXTURE
from operator_agent.contract import freeze_from_task
from operator_agent.hitl_tools import build_task
from operator_agent.correctness import as_suite
from operator_agent.run_contract import code_hash

HASH = re.compile(r'[a-f0-9]{64}\Z')


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('evidence file unavailable')
    return json.loads(path.read_text(encoding='utf-8'))


def evaluate(events, ledger, observations, *, task_id, run_dir, expected_tool_name,
             expected_candidate=GOOD_FIXTURE, expected_contract_hash=None):
    """Verify raw local agent, ledger, MCP worker and complete L2 evidence."""
    result = dict(schema='s5-vertical-mcp-evidence-v1', evidence_status='unknown',
        reason=None, mock_only=True, formal_s5_credit=False, isolation_verified=False,
        real_model_called=False, gpu_used=False, provider_usage=None)
    def unknown(reason):
        result['reason'] = reason
        return result
    try:
        if (not isinstance(events,list) or not isinstance(ledger,dict) or
                not isinstance(observations,dict) or not isinstance(task_id,str) or
                not isinstance(expected_tool_name,str) or expected_candidate != GOOD_FIXTURE):
            return unknown('invalid evidence identity')
        run=Path(run_dir).resolve(); source_hash=code_hash(expected_candidate)
        declaration=_read(run/'declaration.json'); stored_contract=_read(run/'contract.json')
        frozen=freeze_from_task(build_task(declaration)); contract_hash=frozen.hash
        if (not isinstance(expected_contract_hash,str) or not HASH.fullmatch(expected_contract_hash)
                or contract_hash != expected_contract_hash or stored_contract.get('hash') != contract_hash
                or stored_contract.get('contract') != json.loads(json.dumps(frozen.as_dict()))):
            return unknown('contract identity mismatch')
        if (ledger.get('schema') != 's5-shared-call-budget-v1' or ledger.get('task_id') != task_id
                or ledger.get('aborted') is not False or ledger.get('max_calls') != 2
                or ledger.get('reserved_calls') != 2 or ledger.get('completed_calls') != 2
                or ledger.get('network_interception_proven') is not False):
            return unknown('model budget evidence incomplete')
        calls=ledger.get('calls'); requests=observations.get('requests')
        if (not isinstance(calls,list) or len(calls)!=2 or not isinstance(requests,list)
                or len(requests)!=2 or calls[0].get('call_id')==calls[1].get('call_id')):
            return unknown('two distinct requests required')
        for call, request in zip(calls,requests):
            if (not isinstance(call,dict) or not isinstance(request,dict)
                    or call.get('participant')!='outer' or call.get('outcome')!='answered'
                    or call.get('api_usage') is not None or not isinstance(call.get('request_sha256'),str)
                    or not HASH.fullmatch(call['request_sha256'])
                    or request.get('request_sha256')!=call['request_sha256']
                    or not all(_number(call.get(k)) for k in ('reserved_monotonic','dispatched_monotonic','completed_monotonic'))
                    or not _number(request.get('observed_monotonic'))
                    or not call['reserved_monotonic']<=call['dispatched_monotonic']<=request['observed_monotonic']<=call['completed_monotonic']):
                return unknown('model request identity or ordering mismatch')
        if not calls[0]['completed_monotonic'] <= calls[1]['reserved_monotonic']:
            return unknown('model request ordering mismatch')
        tool_calls=[(i,e) for i,e in enumerate(events) if isinstance(e,dict) and e.get('type')=='tool_call']
        tool_results=[(i,e) for i,e in enumerate(events) if isinstance(e,dict) and e.get('type')=='tool_result']
        finals=[(i,e) for i,e in enumerate(events) if isinstance(e,dict) and e.get('type')=='final']
        if len(tool_calls)!=1 or len(tool_results)!=1 or len(finals)!=1:
            return unknown('agent tool event count mismatch')
        ci,call=tool_calls[0]; ri,tool_result=tool_results[0]; fi,_=finals[0]
        cid=call.get('callId'); args=call.get('input'); raw=tool_result.get('result')
        if (not isinstance(cid,str) or not cid or call.get('tool')!=expected_tool_name
                or not isinstance(args,dict) or set(args)!={'run_dir','code','backend','stop_after'}
                or Path(args['run_dir']).resolve()!=run or args['code']!=expected_candidate
                or args['backend']!='mock' or args['stop_after']!='L2'
                or tool_result.get('callId')!=cid or tool_result.get('status')!='completed'
                or not isinstance(raw,str) or not ci<ri<fi):
            return unknown('agent tool call mismatch')
        if not any(isinstance(x,dict) and x.get('type')=='status' and x.get('phase')=='step_start'
                   for x in events[ri+1:fi]):
            return unknown('agent follow-up step missing')
        if not any(isinstance(x,dict) and x.get('type')=='status' and x.get('phase')=='turn_end'
                   and isinstance(x.get('reason'),dict) and x['reason'].get('kind')=='completed'
                   for x in events[ri+1:fi]):
            return unknown('agent completion missing')
        response_hash=_hash(raw.encode('utf-8'))
        if (requests[0].get('tool_feedback') != [] or requests[1].get('tool_feedback') !=
                [dict(tool_call_id=cid,content_sha256=response_hash)]):
            return unknown('follow-up feedback identity mismatch')
        compact=json.loads(raw)
        if (not isinstance(compact,dict) or compact.get('ok') is not True
                or compact.get('backend')!='mock' or compact.get('source_hash')!=source_hash
                or compact.get('contract_hash')!=contract_hash or compact.get('stop_after')!='L2'
                or compact.get('round')!=1 or compact.get('levels')!={'L1':True,'L2':True}):
            return unknown('MCP response mismatch')
        resolution=compact.get('backend_resolution')
        if (not isinstance(resolution,dict) or resolution.get('requested')!='mock'
                or resolution.get('resolved')!='MockGPU' or resolution.get('degraded') is not False):
            return unknown('mock backend identity missing')
        rounds=[x for x in run.glob('verify-*') if x.is_dir()]
        round_dir=run/'verify-1'
        if len(rounds)!=1 or rounds[0]!=round_dir or round_dir.is_symlink():
            return unknown('verification round count mismatch')
        full=_read(round_dir/'result.json'); receipt=_read(round_dir/'complete.json')
        if (Path(compact.get('evidence','')).resolve()!=round_dir/'result.json'
                or (round_dir/'candidate.py').is_symlink()
                or (round_dir/'candidate.py').read_text(encoding='utf-8')!=expected_candidate):
            return unknown('candidate or result path mismatch')
        binding=dict(contract_hash=contract_hash,source_hash=source_hash,stop_after='L2',
                     tool_version=full.get('binding',{}).get('tool_version'))
        if (full.get('schema')!='operator-tool-evidence-v1' or full.get('evidence_complete') is not True
                or full.get('implementation_stable') is not True or full.get('binding')!=binding
                or full.get('contract_hash')!=contract_hash or full.get('source_hash')!=source_hash
                or full.get('backend')!='mock' or full.get('backend_resolution')!=resolution
                or full.get('contract')!=stored_contract or full.get('declaration')!=declaration
                or receipt.get('schema')!='operator-tool-evidence-complete-v1'
                or receipt.get('evidence_complete') is not True or receipt.get('binding')!=binding):
            return unknown('full MCP evidence identity mismatch')
        files=receipt.get('files_sha256')
        if not isinstance(files,dict) or not {'candidate.py','result.json','started.json'} <= set(files):
            return unknown('evidence receipt incomplete')
        for name, checksum in files.items():
            rel=Path(name)
            if (not isinstance(name,str) or rel.is_absolute() or '..' in rel.parts
                    or not isinstance(checksum,str) or not HASH.fullmatch(checksum)):
                return unknown('invalid receipt member')
            target=round_dir/rel
            if target.is_symlink() or not target.is_file() or _hash(target.read_bytes())!=checksum:
                return unknown('evidence receipt hash mismatch')
        suite=as_suite(build_task(declaration).constraints.correctness_suite)
        verdicts=full.get('verdicts')
        if not isinstance(verdicts,list): return unknown('verification verdicts missing')
        l1=[x for x in verdicts if isinstance(x,dict) and x.get('level')=='L1']
        l2=[x for x in verdicts if isinstance(x,dict) and x.get('level')=='L2']
        if len(l1)!=1 or len(l2)!=1 or l1[0].get('passed') is not True or l2[0].get('passed') is not True:
            return unknown('L1/L2 pass not established')
        metrics=l2[0].get('metrics'); correctness=metrics.get('correctness') if isinstance(metrics,dict) else None
        cases=correctness.get('cases') if isinstance(correctness,dict) else None
        wanted={c.case_id for c in suite.cases}
        if (not isinstance(cases,list) or len(wanted)!=12 or len(cases)!=12
                or correctness.get('verdict')!='passed' or correctness.get('declared_cases')!=12
                or correctness.get('suite_hash')!=suite.digest() or metrics.get('num_input_sets')!=1
                or any(not isinstance(c,dict) for c in cases)
                or {c.get('case_id') for c in cases}!=wanted
                or any(c.get('verdict')!='passed' or c.get('determinism')!='passed' for c in cases)):
            return unknown('full correctness suite incomplete')
        primary=metrics.get('input_fidelity_readings')
        if (not isinstance(primary,list) or len(primary)!=1
                or primary[0].get('input_fidelity_verified') is not True
                or any(not isinstance(c.get(k),dict) or c[k].get('input_fidelity_verified') is not True
                       for c in cases for k in ('metrics','repeat_metrics'))):
            return unknown('input fidelity incomplete')
        invocation=compact.get('invocation_id')
        if not isinstance(invocation,str) or not re.fullmatch(r'[a-f0-9]{32}',invocation):
            return unknown('MCP worker identity unavailable')
        tool_dir=run/'tool-calls'/invocation
        if Path(compact.get('tool_call_dir','')).resolve()!=tool_dir or tool_dir.is_symlink():
            return unknown('MCP worker path mismatch')
        req=_read(tool_dir/'request.json'); status=_read(tool_dir/'run-status.json')
        worker=_read(tool_dir/'worker-result.json'); tool_output=_read(tool_dir/'result.json')
        recorded_args=req.get('arguments')
        if (req.get('name')!='operator_verify' or not isinstance(recorded_args,dict)
                or set(recorded_args)!=set(args) or Path(recorded_args['run_dir']).resolve()!=run
                or any(recorded_args[k]!=args[k] for k in ('code','backend','stop_after'))
                or status.get('phase')!='done' or status.get('tool')!='operator_verify'
                or status.get('invocation_id')!=invocation or worker.get('invocation_id')!=invocation
                or not _number(worker.get('finished_monotonic'))
                or not calls[0]['completed_monotonic']<=worker['finished_monotonic']<=calls[1]['reserved_monotonic']
                or not isinstance(tool_output.get('content'),list) or len(tool_output['content'])!=1
                or tool_output['content'][0].get('text')!=raw):
            return unknown('MCP worker evidence mismatch')
        result.update(evidence_status='complete',reason='local cross-checks complete',
                      fake_calls=2,mock_verifications=1,tool_call_id=cid,
                      candidate_sha256=source_hash,contract_hash=contract_hash,
                      suite_hash=suite.digest(),tool_reply_sha256=response_hash)
        return result
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, json.JSONDecodeError):
        return unknown('malformed or unavailable evidence')
