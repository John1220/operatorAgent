"""Read-only parent evidence projection and exclusive once-per-parent claim.

Historical seals detect accidental drift, not hostile same-UID forgery. Never
resume the parent or import its source. No input allocation or external I/O.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from operator_agent.generate.bounded_repair import validate_repair
from operator_agent.core.run_contract import code_hash, digest, seal

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / 'output' / 'workload_codegen_h20' / 'repair-claims'
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024


def safe_bytes(root, name):
    root = Path(root)
    p = root / name
    if (Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink()
            or any(q.is_symlink() for q in p.parents) or not p.is_file()
            or p.stat().st_size > MAX_FILE_BYTES):
        raise ValueError('unsafe/missing/oversize parent evidence')
    data = p.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError('oversize parent evidence')
    return data


def check_files(root, hashes):
    if type(hashes) is not dict or not hashes or len(hashes) > 256:
        raise ValueError('invalid parent evidence inventory')
    total = 0
    for name, expected in hashes.items():
        data = safe_bytes(root, name); total += len(data)
        if total > MAX_TOTAL_BYTES or code_hash(data.decode()) != expected:
            raise ValueError('parent evidence drift/byte cap')


def load_parent(root, task):
    # Import at call time: the driver also imports this pure helper.
    from benchmarks.drivers import workload_codegen_h20 as d
    root = Path(root).absolute()
    hashes = {}
    total = 0

    def read(name, *, text=False, array=False):
        nonlocal total
        data = safe_bytes(root, name); total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise ValueError('parent evidence exceeds total byte cap')
        hashes[name] = code_hash(data.decode())
        if text: return data.decode()
        if array:
            # strict_object wraps this list so duplicate keys/NaN stay refused.
            return d.strict_object('{"rows":'+data.decode()+'}')['rows']
        return d.strict_object(data.decode())

    plan = read('plan.json'); summary = read('summary.json'); evidence = read('evidence.json')
    if (plan['hash'] != digest({k:v for k,v in plan.items() if k != 'hash'})
            or plan['generation'] != d.generation_declaration(task)
            or plan['generation']['repairs'] != 0 or 'repair_parent' in plan
            or summary['plan_hash'] != plan['hash'] or summary['status'] != 'correctness_failed'
            or any(summary[k] is not False for k in ('completed','accepted','deployment_accepted'))
            or summary['errors'] or summary['model_attempts'] != 1 or summary['verifier_dispatches'] != 1
            or summary['cleanup_status']['state'] != 'removed'
            or summary['cleanup'].get('quiescent') is not True or summary['cleanup'].get('removed') is not True):
        raise ValueError('parent is not a clean original numerical failure')
    for name, expected in plan['evaluator_sources'].items():
        if code_hash(read('frozen-sources/'+name, text=True)) != expected:
            raise ValueError('parent frozen evaluator drift')
    for name in ('model-budget.json','llm-usage.json'):
        read(name)
    for name in ('limits','identity','generation-context','generation-declaration','call-001','candidate'):
        read('model-journal/'+name+'.json')
    model = d.audit_model(root, plan)
    if not model['completed']:
        raise ValueError('parent model-source proof failed')
    directory = Path(summary['saved_dir'])
    try:
        relative = str(directory.relative_to(root))
    except ValueError:
        raise ValueError('parent artifact outside owned run') from None
    artifact_names = ('result.json','execution_snapshot.json','run_manifest.json','contract.json',
                      'verification.json','sketch.json','code.py')
    values = {n:read(relative+'/'+n, text=n=='code.py', array=n=='verification.json') for n in artifact_names}
    snap, manifest, result = (values[n] for n in ('execution_snapshot.json','run_manifest.json','result.json'))
    binding = dict(contract=values['contract.json'], manifest=manifest)
    body = dict(manifest); claimed = body.pop('hash')
    rounds = evidence['rounds']
    if (len(rounds) != 1 or len(rounds[0]['results']) != 2
            or evidence['dispatched'] != [code_hash(model['source'])]
            or evidence['forbidden_actions'] or rounds[0]['code'] != model['source']
            or values['code.py'] != model['source'] or snap['current_code'] != model['source']
            or snap['pending_fix'] is not None or snap['round_done'] != 1
            or snap['verification_results'] != rounds[0]['results']
            or values['contract.json'] != plan['contract'] or manifest['contract_hash'] != plan['contract']['hash']
            or manifest['schema'] != 'frozen-run-v1' or claimed != digest(body)
            or manifest['workload_hash'] != digest(manifest['inputs'])
            or manifest['task_hash'] != digest(manifest['task'])
            or manifest['reference_hash'] != d.registered_reference().digest()
            or manifest['suite_hash'] != d.wd.as_suite(plan['contract']['correctness_suite']).digest()
            or manifest['measurement_protocol'] != plan['measurement_protocol']
            or manifest['controls']['code_generation'] != plan['generation']
            or snap['run_binding'] != seal(dict(snap), binding)['run_binding']
            or snap['environment_fingerprint'] != manifest['environment']
            or result['iteration_history'] != snap['iteration_history']
            or len(snap['iteration_history']) != 1 or result['all_passed'] is not False
            or result['code_generation'] != dict(declaration=model['declaration'], lineage=model['lineage'])
            or result['llm_call_budget'] != model['budget']
            or result['stop']['reason'] != 'l1_l2_failed'):
        raise ValueError('parent durable source/checkpoint/contract drift')
    for name in ('confirmation.json','admission.json','benchmark.json'):
        p = directory/name
        if p.exists() or p.is_symlink(): raise ValueError('parent granted post-failure evidence')
    # Use the existing raw receipt/transfer checks, but never rebuild old arrays
    # or compare its historical evaluator identity to the new implementation.
    from operator_agent.core.schemas import VerificationResult, VerificationLevel
    results = [VerificationResult(VerificationLevel(r['level']), r['passed'], r['message'],
                                  metrics=r['metrics']) for r in rounds[0]['results']]
    artifact = d.ArtifactStore(str(directory)).load(str(directory))
    derived = {'claim_level','objective','implementation_decision','case_key','case_source','case_declared','case_match'}
    def core(rows):
        return [dict(level=r.level.value, passed=r.passed, message=r.message,
                     metrics={k:v for k,v in r.metrics.items() if k not in derived}) for r in rows]
    if core(artifact.verification_results) != core(results):
        raise ValueError('parent saved verification drift')
    selection = result.get('candidate_selection') or {}
    if (selection.get('selected_round') is not None or selection.get('fallback_to_best')
            or (result.get('corpus_admission') or {}).get('kept') is True):
        raise ValueError('failed parent selected/admitted a candidate')
    raw = d.audit_raw(artifact, [(model['source'],results)], evidence['trace'], plan,
                      task=task, corpus=root/'corpus'/'kernel_db.jsonl')
    if not raw['completed'] or raw['status'] != 'correctness_failed':
        raise ValueError('parent raw CUDA evidence failed')
    l1, l2 = rounds[0]['results']
    if (l1['level'] != 'L1' or l1['passed'] is not True or l2['level'] != 'L2' or l2['passed'] is not False
            or l2['metrics']['correctness_primary_verdict'] != 'failed'):
        raise ValueError('repair requires actual L1 pass/L2 numerical failure')
    suite = l2['metrics']['correctness']
    cases = plan['contract']['correctness_suite']['cases']
    if (suite['suite_hash'] != d.wd.as_suite(plan['contract']['correctness_suite']).digest()
            or [r['case_id'] for r in suite['cases']] != [r['case_id'] for r in cases]
            or any(r['verdict'] != 'not_run' for r in suite['cases'])):
        raise ValueError('unsupported parent suite state')
    expected_binding = dict(run_id=manifest['run_id'], manifest_hash=manifest['hash'],
        contract_hash=manifest['contract_hash'], workload_hash=manifest['workload_hash'],
        protocol_hash=digest(manifest['measurement_protocol']), candidate_hash=code_hash(model['source']))
    vb = l2['metrics']['run_binding']
    for r in (l1,l2):
        if r['metrics']['run_binding'] != dict(expected_binding, verification_id=vb['verification_id']):
            raise ValueError('parent verifier identity drift')
    verifies = [r for r in evidence['trace'] if r['request']['action']=='verify']
    metadata = plan['generation']['context']['input_metadata']
    if len(verifies) != len(metadata): raise ValueError('parent primary dispatch count drift')
    primary = []
    fields = ('max_error','max_rel_error','matched_ratio','bad_elements','compared_elements',
              'coverage','reference_nonfinite','atol','rtol','required_matched_ratio','output_dtype')
    if len(manifest['inputs']) != len(metadata): raise ValueError('parent manifest input count drift')
    for rpc, inputs, original in zip(verifies, metadata, manifest['inputs']):
        req, res = rpc['request'], rpc['response']
        if ({k:v['shape'] for k,v in req['inputs']['tensors'].items()} != {k:v['shape'] for k,v in inputs.items()}
                or {k:v['shape'] for k,v in original.items()} != {k:v['shape'] for k,v in inputs.items()}
                or req['reference_contract'] != plan['contract']['reference_contract']
                or req['max_error'] != plan['contract']['max_error'] or req['precision'] != 'fp16'
                or res.get('input_fidelity',{}).get('observed') != original
                or res.get('input_fidelity',{}).get('representation') != 'torch-storage-v1'
                or res.get('input_fidelity',{}).get('tensor_devices') != {'X':'cuda:0'}
                or set(original) != {'X'} or original['X']['dtype'] != '<f2'
                or original['X']['strides'] != inputs['X']['strides']
                or res['device'] != 'cuda' or res['reference_type'] != 'pytorch_eager'):
            raise ValueError('parent primary input/reference drift')
        shape = inputs['X']['shape']
        primary.append(dict(case_key='X='+'x'.join(map(str,shape)), shape=shape,
                            passed=res['success'], **{k:res[k] for k in fields}))
    failed = next(r for r in reversed(primary) if r['passed'] is False)
    if any(l2['metrics'][k] != failed[k] for k in fields):
        raise ValueError('parent numerical result/trace drift')
    feedback = dict(stage='L2', kind='primary_numerical_mismatch', primary=primary,
        suite=dict(declared_cases=len(cases), executed_cases=0, verdict='not_run'))
    repair = dict(schema='bounded-numerical-repair-v1', repair_index=1,
        parent=dict(plan_hash=plan['hash'], run_id=manifest['run_id'], manifest_hash=manifest['hash'],
            verification_id=vb['verification_id'], source_hash=code_hash(model['source']),
            evidence_hash=hashes['evidence.json']), contract_hash=plan['contract']['hash'],
        source=model['source'], feedback=feedback, feedback_hash=digest(feedback))
    validate_repair(repair, plan['contract'], metadata)
    check_files(root, hashes)
    return repair, dict(root=str(root), files=hashes, identity_hash=digest(repair['parent']))


def claim_path(parent):
    identity = parent['identity_hash']
    if len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('invalid repair claim identity')
    return CLAIMS / (identity+'.json')


def require_available(parent):
    p = claim_path(parent)
    if p.exists() or p.is_symlink():
        raise ValueError('parent repair already consumed; no retry/repair chain')


def check_claim(parent, plan_hash, root):
    from operator_agent.execute.llm_budget import strict_object
    actual = strict_object(safe_bytes(CLAIMS, claim_path(parent).name).decode())
    if actual != dict(parent_identity_hash=parent['identity_hash'], plan_hash=plan_hash,
                      execution_root=str(root), resumable=False):
        raise ValueError('repair claim owner/plan drift')


def claim(parent, plan_hash, root):
    # A fresh plan/output directory cannot refund the parent repair. Atomic
    # exclusive link; burn before any HTTP/SSH. No rollback after a crash.
    p = claim_path(parent)
    if CLAIMS.is_symlink() or any(q.is_symlink() for q in CLAIMS.parents):
        raise ValueError('unsafe repair claim directory')
    CLAIMS.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(parent_identity_hash=parent['identity_hash'], plan_hash=plan_hash,
                           execution_root=str(root), resumable=False)).encode()
    fd, tmp = tempfile.mkstemp(dir=CLAIMS)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(),0o444)
        try: os.link(tmp,p)
        except FileExistsError:
            raise ValueError('parent repair already consumed; no retry/repair chain') from None
        directory = os.open(str(CLAIMS),os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.unlink(tmp)
