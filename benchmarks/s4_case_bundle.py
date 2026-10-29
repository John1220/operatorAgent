"""Build a sanitized, offline-verifiable S4 case bundle from retained evidence.

Only allowlisted fields are copied; historical results are not rerun or changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

EXPECTED_TASKS=('matmul-aligned','matmul-tail','softmax-tail','layernorm-tail')
EXPECTED_PLAN='cd661a10229d8461e116eb9d41b161541e443f76496657d196781fb7769778b3'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def number(x): return x if type(x) in (int,float) and math.isfinite(x) else None


def project_event(path):
    event=load(path); level=event['level']; verdict=next(v for v in reversed(event['verdicts']) if v['level']==level)
    metrics=verdict.get('metrics') or {}; watermark=event.get('model_watermark') or {}
    result=dict(schema='s4-event-projection-v1',origin_evidence_sha256=sha(path),
        contract_sha256=event['contract_hash'],source_sha256=event['source_hash'],level=level,
        passed=event['passed'],full_l2_passed=event['full_l2_passed'],
        completed_at=number(event['timing']['completed_at']),
        completed_monotonic=number(event['timing']['completed_monotonic']),
        model_watermark=dict(reserved_calls=watermark.get('reserved_calls'),
            dispatched_calls=watermark.get('dispatched_calls'),completed_calls=watermark.get('completed_calls')))
    if level=='L2':
        c=metrics['correctness']; cases=[]
        for row in c['cases']:
            m=row.get('metrics') or {}
            cases.append(dict(case_id=row['case_id'],shape_key=row.get('shape_key'),
                pattern=row.get('pattern'),layout=row.get('layout'),verdict=row.get('verdict'),
                bad_elements=number(m.get('bad_elements')),compared_elements=number(m.get('compared_elements')),
                max_error=number(m.get('max_error')),matched_ratio=number(m.get('matched_ratio')),
                input_fidelity_verified=m.get('input_fidelity_verified')))
        result['correctness']=dict(verdict=c.get('verdict'),cases=cases)
    elif level=='L3':
        result['performance']=dict(validity=metrics.get('measurement_validity'),
            can_gate=metrics.get('measurement_can_gate'),launch_dominance=number(metrics.get('launch_dominance')),
            paired_samples=number(metrics.get('paired_samples')),baseline_type=metrics.get('baseline_type'))
    else: raise ValueError('unsupported event level')
    return result


def case_source(run, source_hash):
    for p in sorted(run.rglob('*.py')):
        if sha(p)==source_hash: return p
    raise ValueError('source matching event hash missing')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='S4 gpu6-8004 evidence directory')
    parser.add_argument('--output',type=Path,required=True,help='new bundle directory; must not exist')
    args=parser.parse_args(); src=args.source.resolve(); dest=args.output.resolve()
    if dest.exists(): raise ValueError('output already exists; never overwrite evidence')
    if dest.is_relative_to(src): raise ValueError('output must be outside the source evidence directory')
    if not dest.parent.is_dir(): raise ValueError('output parent must exist')
    manifest=load(src/'evidence-manifest.json')['files']
    for rel,h in manifest.items():
        p=src/rel
        if not p.is_file() or sha(p)!=h: raise ValueError('source evidence manifest mismatch: '+rel)
    if sha(src/'frozen-plan.json')!=EXPECTED_PLAN: raise ValueError('unexpected frozen plan')
    original=load(src/'summary.json'); plan=load(src/'frozen-plan.json')
    if tuple(r['task'] for r in original['tasks'])!=EXPECTED_TASKS: raise ValueError('unexpected task denominator')
    with tempfile.TemporaryDirectory(dir=dest.parent,prefix='.s4-case-bundle-') as tmp:
        out=Path(tmp)/'bundle'; out.mkdir(); rows=[]
        shutil.copyfile(Path(__file__).with_name('s4_case_bundle_verify.py'),out/'verify.py')
        for source_row,task in zip(original['tasks'],plan['tasks']):
            name=source_row['task']; run=src/'tasks'/name; folder=out/'cases'/name; folder.mkdir(parents=True)
            if name=='layernorm-tail':
                evidence=run/'repair-1'; event_dir=evidence/'phase-events'; journal=evidence/'model-journal'; initial=load(evidence/'result.json')['initial_full_l2_passed']; status_dir=next((run/'tool-calls').iterdir())
            else:
                event_dir=run/'measurements'; journal=run/'model-journal'; initial=None; status_dir=run
            status=load(status_dir/'run-status.json')
            if status.get('phase')!='done' or (status.get('cleanup') or {}).get('remote_execution')!='quiescent': raise ValueError('unproven cleanup: '+name)
            selected=source_row['selected_source_sha256']; paths=list(event_dir.glob('verification-event-*.json'))
            events=[(p,load(p)) for p in paths]
            if not events or any(e.get('contract_hash')!=task['contract_sha256'] for _,e in events): raise ValueError('contract event mismatch: '+name)
            good=sorted(((p,e) for p,e in events if e.get('level')=='L2' and e.get('full_l2_passed') is True),key=lambda x:x[1]['timing']['completed_at'])
            final=[(p,e) for p,e in good if e.get('source_hash')==selected]
            l3=[(p,e) for p,e in events if e.get('level')=='L3' and e.get('source_hash')==selected]
            if not good or len(final)!=1 or len(l3)!=1: raise ValueError('missing bound L2/L3: '+name)
            first=good[0]
            for target,entry in (('first-l2.json',first),('final-l2.json',final[0]),('final-l3.json',l3[0])): write(folder/target,project_event(entry[0]))
            shutil.copyfile(case_source(run,first[1]['source_hash']),folder/'first.py')
            shutil.copyfile(case_source(run,selected),folder/'selected.py')
            if name=='layernorm-tail':
                failed=[(p,e) for p,e in events if e.get('level')=='L2' and e.get('full_l2_passed') is False]
                if initial is not False or len(failed)!=1: raise ValueError('missing initial repair failure')
                write(folder/'initial-l2.json',project_event(failed[0][0]))
                shutil.copyfile(case_source(run,failed[0][1]['source_hash']),folder/'initial.py')
            ledger=load(journal/'summary.json'); calls=[]
            for i,entry in enumerate(ledger['attempts'],1):
                call_path=journal/('call-%03d.json'%i); call=load(call_path)
                if call!=entry: raise ValueError('journal mismatch: '+name)
                req=call['request']; usage=call.get('api_usage') or {}
                calls.append(dict(origin_evidence_sha256=sha(call_path),call=call['call'],caller=call['caller'],status=call['status'],
                    model=req['model'],max_tokens=req['max_tokens'],usage_source=call.get('usage_source'),
                    api_usage={k:usage.get(k) for k in ('prompt_tokens','completion_tokens','total_tokens')},
                    request_sha256=call.get('request_sha256'),response_sha256=call.get('response_sha256'),
                    completed_at_utc=call.get('completed_at_utc'),completed_monotonic=number(call.get('completed_monotonic'))))
            write(folder/'calls.json',dict(schema='s4-call-projection-v1',calls=calls))
            row={k:source_row.get(k) for k in ('task','contract_sha256','selected_source_sha256','initial_full_l2_passed','final_full_l2_passed','origin','first_full_l2_seconds','first_full_l2_calls','first_full_l2_api_tokens','final_model_calls','final_api_usage','unknown_api_calls','l3_validity','l3_can_gate','launch_dominance','performance_target','independent_performance_confirmation','remote_execution','local_all_descendants_proven','provider_call')}
            row['started_at']=status['started_at']
            rows.append(row)
        summary={k:original[k] for k in ('schema','plan_sha256','denominator','full_l2_passed','performance_confirmed','performance_unknown','model_calls','prompt_tokens','completion_tokens','total_tokens','monetary_cost','scope')}; summary['tasks']=rows
        write(out/'summary.json',summary)
        (out/'README.md').write_text('''# S4 算子开发案例：脱敏离线证据包\n\n来源：2026-10-27 的 H20 GPU 6 小批 CLI 验收。四项最终候选通过有限完整 L2；四项性能测量均不可裁决，不表示性能达标。LayerNorm 初始候选失败，修复后通过。\n\n运行 `python3 verify.py` 可验证文件哈希、源码/契约绑定、L2/L3 事实和调用/token 合计；只解析文件，不执行候选、不调用模型或 GPU。`cases/` 是原始事件、账本的字段白名单脱敏投影，并保留各原文件 SHA256；不包含端点、个人路径、凭据或原始日志。\n\n该包不包含完整原始记录或实现环境；哈希证明包内一致性，不是签名、真实性认证或历史 GPU 测量重演。重建数值和真实性需原始可信归档与相应运行环境。四项正确性仅针对有限 suite；真实金额、provider 状态及所有本机后代是否停止未知。\n''',encoding='utf-8')
        files=sorted(p for p in out.rglob('*') if p.is_file())
        write(out/'manifest.json',dict(schema='s4-portable-manifest-v1',files={p.relative_to(out).as_posix():sha(p) for p in files}))
        # Test the copied verifier before the directory becomes visible.
        import subprocess,sys
        subprocess.run([sys.executable,str(out/'verify.py')],check=True)
        out.rename(dest)
    print('Created sanitized bundle:',dest)


if __name__=='__main__': main()
