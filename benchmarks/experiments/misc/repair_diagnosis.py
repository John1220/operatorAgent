"""OFFLINE ONLY: replay a retained response and freeze a diagnosis/context preview.

There is intentionally no --execute, endpoint, SSH, arrays or credit operation.
A preview is NOT an executable prepared plan or authorization for a new request.
--audit independently reads/recomputes a retained preview without writing to it.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_agent.generate.bounded_repair import source_change
from operator_agent.execute.llm_budget import strict_object
from operator_agent.generate.repair_diagnoser import diagnose_repair
from operator_agent.core.run_contract import atomic_json, code_hash, digest, protocol_identity
from operator_agent.generate.single_candidate_coder import generation_declaration, SingleCandidatePolicy


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--case')
    p.add_argument('--response')
    p.add_argument('--output')
    p.add_argument('--audit',help='read-only preview directory audit; no output files are written')
    return p


def read(path):
    p=Path(path)
    if (p.is_symlink() or any(q.is_symlink() for q in p.parents) or not p.is_file()
            or p.stat().st_size>65536):
        raise ValueError('unsafe/missing/oversize offline fixture')
    with p.open('rb') as f:
        data=f.read(65537)
    if len(data)>65536: raise ValueError('offline fixture byte cap')
    return data.decode()


def build_preview(case, response_text):
    if len(response_text.encode())>65536 or len(json.dumps(case,allow_nan=False).encode())>65536:
        raise ValueError('offline input byte cap')
    if set(case)!={'schema','context','provenance'} or case['schema']!='offline-real-repair-case-v1':
        raise ValueError('invalid offline diagnosis case')
    base=copy.deepcopy(case['context'])
    if set(base)!={'task_id','hardware','dsl','contract','reference_source','sketch','input_metadata','calling_convention','repair'}:
        raise ValueError('invalid original repair context')
    original=generation_declaration(base,SingleCandidatePolicy())
    if original['schema']!='single-candidate-repair-v1': raise ValueError('original repair v1 required')
    provenance=case['provenance']; response=strict_object(response_text)
    if (code_hash(response_text)!=provenance['response_sha256'] or len(response_text.encode())!=provenance['response_bytes']
            or response['model']!=provenance['reported_model'] or len(response['choices'])!=1
            or response['choices'][0]['finish_reason']!='stop'):
        raise ValueError('offline raw response/provenance drift')
    model=strict_object(response['choices'][0]['message']['content'])
    if (set(model)!={'code','rationale'} or type(model['rationale']) is not str
            or not 0<len(model['rationale'])<=1000 or code_hash(base['repair']['source'])!=provenance['source_sha256']
            or base['repair']['parent']['plan_hash']!=provenance['parent_plan_hash']):
        raise ValueError('offline source/envelope binding drift')
    change=source_change(base['repair']['source'],model['code'])
    diagnosis=diagnose_repair(base['repair'],base['contract'],base['input_metadata'])
    preview=None
    if diagnosis['state']=='supported_hypothesis':
        context=copy.deepcopy(base);context['repair_diagnosis']=diagnosis
        preview=generation_declaration(context,SingleCandidatePolicy())
    return dict(diagnosis=diagnosis,generation_preview=preview,response_review=dict(
        source_change=change,rationale_trust='untrusted_model_explanation_not_a_credential',
        rationale_has_source_change_evidence=change['change']=='changed_ast',
        candidate_rewritten=False,candidate_executed=False,gpu_measurements=0,
        correctness_proven=False,restored_eligibility=False))


def _summary(result, case_text, response, producer_hash, protocol):
    return dict(status=result['diagnosis']['state'],diagnosis_hash=result['diagnosis']['hash'],
        source_change=result['response_review']['source_change']['change'],
        executable_plan=False,approved=False,model_calls=0,ssh_calls=0,gpu_dispatches=0,
        repair_credit_created=False,repair_credit_reset=False,producer_source_hash=producer_hash,
        frozen_input_hashes=dict(case=code_hash(case_text),response=code_hash(response)),
        measurement_protocol=protocol)


class _AuditRefused(ValueError):
    def __init__(self,reason):
        self.reason=reason
        super().__init__(reason)


_BASE_FILES={'diagnosis.json','response_review.json','frozen-case.json','frozen-response.json','summary.json'}
_ALLOWED_FILES=_BASE_FILES | {'generation_preview.json','producer-source.py'}


def _snapshot(directory):
    root=Path(directory)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents) or not root.is_dir():
        raise _AuditRefused('unsafe_or_missing_preview')
    paths=[]
    # scandir is lazy even on Python versions where Path.iterdir uses listdir.
    with os.scandir(root) as entries:
        for entry in entries:
            paths.append(Path(entry.path))
            if len(paths)>len(_ALLOWED_FILES): raise _AuditRefused('too_many_preview_files')
    names={p.name for p in paths}
    if not _BASE_FILES<=names or not names<=_ALLOWED_FILES:
        raise _AuditRefused('missing_or_extra_preview_files')
    values={}; hashes={}; modes={}; total=0
    for p in sorted(paths):
        if p.is_symlink() or not p.is_file(): raise _AuditRefused('unsafe_preview_file')
        mode=p.stat().st_mode & 0o777
        if mode & 0o222: raise _AuditRefused('writable_preview_file')
        value=read(p); total+=len(value.encode())
        if total>262144: raise _AuditRefused('preview_total_byte_cap')
        values[p.name]=value;hashes[p.name]=code_hash(value);modes[p.name]=mode
    return values,hashes,modes


def audit_preview(directory):
    """Recompute from frozen data; no writer/client/claim or source execution.

    Internal consistency only. Even a self-consistent bundle cannot attest its
    historical producer execution, human authorization or any GPU eligibility.
    v1 has only a declared producer hash; v2 additionally checks its data snapshot.
    """
    proof=dict(schema='offline-repair-diagnosis-audit-v1',completed=False,evidence_consistent=False,
        read_only=True,errors=[],model_calls=0,ssh_calls=0,gpu_measurements=0,
        executable_plan=False,approved=False,restored_eligibility=False,repair_credit_reset=False,
        scope='frozen_data_recomputation_not_producer_execution_or_authorization_attestation')
    try:
        values,hashes,modes=_snapshot(directory)
        summary=strict_object(values['summary.json'])
        version=summary.get('schema')
        if version not in ('offline-repair-diagnosis-preview-v1','offline-repair-diagnosis-preview-v2'):
            raise _AuditRefused('unsupported_preview_schema')
        producer_hash=summary.get('producer_source_hash')
        if type(producer_hash) is not str or re.fullmatch('[0-9a-f]{64}',producer_hash) is None:
            raise _AuditRefused('invalid_producer_identity')
        result=build_preview(strict_object(values['frozen-case.json']),values['frozen-response.json'])
        expected_names=set(_BASE_FILES)
        if result['generation_preview'] is not None: expected_names.add('generation_preview.json')
        if version.endswith('-v2'): expected_names.add('producer-source.py')
        if set(values)!=expected_names: raise _AuditRefused('unexpected_preview_state_files')
        for key,value in result.items():
            if value is None: continue
            if digest(strict_object(values[key+'.json']))!=digest(value):
                raise _AuditRefused('derived_preview_evidence_drift')
        protocol=protocol_identity()
        if digest(summary.get('measurement_protocol'))!=digest(protocol):
            raise _AuditRefused('audit_protocol_drift')
        expected=dict(schema=version,**_summary(result,values['frozen-case.json'],
            values['frozen-response.json'],producer_hash,protocol))
        producer_check='declared_hash_only_no_snapshot'
        if version.endswith('-v2'):
            if hashes['producer-source.py']!=producer_hash:
                raise _AuditRefused('producer_snapshot_hash_drift')
            expected['files']={k:v for k,v in hashes.items() if k!='summary.json'}
            expected['hash']=digest(expected)
            producer_check='snapshot_hash_verified_not_execution_attested'
        if digest(summary)!=digest(expected): raise _AuditRefused('preview_summary_or_manifest_drift')
        values_after,hashes_after,modes_after=_snapshot(directory)
        if (values_after!=values or hashes_after!=hashes or modes_after!=modes):
            raise _AuditRefused('changing_preview_evidence')
        proof.update(completed=True,evidence_consistent=True,preview_schema=version,
            diagnosis_state=result['diagnosis']['state'],diagnosis_hash=result['diagnosis']['hash'],
            source_change=result['response_review']['source_change']['change'],
            context_hash=(result['generation_preview']['context_hash'] if result['generation_preview'] else None),
            producer_identity_check=producer_check,declared_producer_source_hash=producer_hash,
            audit_source_hash=code_hash(Path(__file__).read_text()),protocol_hash=digest(protocol),
            file_hashes=hashes)
    except _AuditRefused as exc:
        proof['errors'].append(exc.reason)
    except (OSError,ValueError,TypeError,KeyError,IndexError,AttributeError,RecursionError):
        # Do not echo paths, source, arbitrary JSON fields or exception text.
        proof['errors'].append('invalid_or_unreadable_preview_evidence')
    return proof


def run(args):
    if args.audit:
        if args.case is not None or args.response is not None or args.output is not None:
            raise ValueError('--audit cannot be combined with build/output arguments')
        proof=audit_preview(args.audit)
        print(json.dumps(proof,ensure_ascii=False,allow_nan=False,indent=2))
        return 0 if proof['completed'] else 1
    if args.case is None or args.response is None:
        raise ValueError('offline build requires both --case and --response')
    case_text=read(args.case);response=read(args.response)
    result=build_preview(strict_object(case_text),response)
    producer=Path(__file__).read_text()
    payload={name+'.json':json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)
             for name,value in result.items() if value is not None}
    payload.update({'frozen-case.json':case_text,'frozen-response.json':response,'producer-source.py':producer})
    summary=dict(schema='offline-repair-diagnosis-preview-v2',
        **_summary(result,case_text,response,code_hash(producer),protocol_identity()),
        files={name:code_hash(data) for name,data in payload.items()})
    summary['hash']=digest(summary)
    encoded_sizes=[len(data.encode()) for data in payload.values()]
    encoded_sizes.append(len(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2).encode()))
    if max(encoded_sizes)>65536 or sum(encoded_sizes)>262144:
        raise ValueError('offline artifact byte cap before output')
    output=Path(args.output or ROOT/'output/repair_diagnosis')
    if output.is_symlink() or any(p.is_symlink() for p in output.parents):
        raise ValueError('unsafe offline output')
    output.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='repair-diagnosis-',dir=output))
    for name,data in payload.items():
        with (root/name).open('xb') as f:
            f.write(data.encode());f.flush()
            os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
    atomic_json(root/'summary.json',summary,readonly=True)
    print('OFFLINE_PREVIEW '+str(root)+' DIAGNOSIS '+result['diagnosis']['hash'])
    return 0


if __name__=='__main__':
    raise SystemExit(run(parser().parse_args()))
