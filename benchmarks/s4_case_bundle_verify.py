"""Offline structural and integrity checks for a sanitized S4 case bundle.

No candidate is imported or executed. Hashes provide consistency, not an
external signature or proof that historical GPU execution occurred.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(condition, description):
    if not condition:
        raise ValueError(description)


def sensitive(data):
    return any(re.search(p, data, re.I) for p in (
        r'https?://', r'/Users/', r'/home/', r'algorithm-h20',
        r'\b(?:api[_-]?key|authorization|bearer)\b',
        r'\bsk-[A-Za-z0-9_-]{16,}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'))


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    check(manifest.get('schema') == 's4-portable-manifest-v1', 'manifest schema')
    files = manifest.get('files')
    check(isinstance(files, dict) and files, 'manifest files')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    check(actual == set(files), 'missing or extra files')
    for rel, wanted in files.items():
        path = ROOT / rel
        check(not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve()), 'unsafe file path')
        check(digest(path) == wanted, 'file hash: ' + rel)
        # This checker contains the detector's own pattern vocabulary; scanning
        # that file would match its own literals. All exported data is scanned.
        if rel != 'verify.py':
            check(not sensitive(path.read_text(encoding='utf-8')), 'sensitive material: ' + rel)
    summary = json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
    rows=summary['tasks']
    check([r['task'] for r in rows] == ['matmul-aligned', 'matmul-tail', 'softmax-tail', 'layernorm-tail'], 'task denominator')
    totals={'calls':0,'prompt':0,'completion':0,'total':0}
    for row in rows:
        folder=ROOT/'cases'/row['task']
        initial=json.loads((folder/'first-l2.json').read_text())
        final=json.loads((folder/'final-l2.json').read_text())
        l3=json.loads((folder/'final-l3.json').read_text())
        calls=json.loads((folder/'calls.json').read_text())['calls']
        check(digest(folder/'first.py') == initial['source_sha256'], row['task']+' first source')
        check(digest(folder/'selected.py') == final['source_sha256'] == l3['source_sha256'] == row['selected_source_sha256'], row['task']+' selected source')
        check(initial['contract_sha256']==final['contract_sha256']==l3['contract_sha256']==row['contract_sha256'], row['task']+' contract binding')
        check(initial['level']=='L2' and initial['full_l2_passed'] is True and final['level']=='L2' and final['full_l2_passed'] is True, row['task']+' L2')
        check(final['correctness']['cases'] and all(c['verdict']=='passed' for c in final['correctness']['cases']), row['task']+' final cases')
        check(l3['level']=='L3' and l3['performance']['validity']=='uninformative' and l3['performance']['can_gate'] is False, row['task']+' L3 eligibility')
        check(row['performance_target']=='unknown', row['task']+' performance')
        check(len(calls)==row['final_model_calls'] and all(c['status']=='answered' and c['usage_source']=='api_usage' for c in calls), row['task']+' calls')
        for c in calls:
            u=c['api_usage']; check(u['prompt_tokens']+u['completion_tokens']==u['total_tokens'],row['task']+' usage')
        n=initial['model_watermark']['reserved_calls']; check(n==row['first_full_l2_calls'] and 0<=n<=len(calls), row['task']+' first calls')
        check(initial['model_watermark']['completed_calls']==n, row['task']+' first completed calls')
        check(all(type(c.get('completed_monotonic')) in (int,float) and c['completed_monotonic']<=initial['completed_monotonic'] for c in calls[:n]),row['task']+' first call timing')
        check(round(initial['completed_at']-row['started_at'],3)==row['first_full_l2_seconds'],row['task']+' first L2 elapsed')
        check(sum(c['api_usage']['total_tokens'] for c in calls[:n])==row['first_full_l2_api_tokens'],row['task']+' first tokens')
        check(sum(c['api_usage']['total_tokens'] for c in calls)==row['final_api_usage']['total_tokens'],row['task']+' final tokens')
        for k,field in [('prompt','prompt_tokens'),('completion','completion_tokens'),('total','total_tokens')]: totals[k]+=sum(c['api_usage'][field] for c in calls)
        totals['calls']+=len(calls)
    failed=json.loads((ROOT/'cases/layernorm-tail/initial-l2.json').read_text())
    check(digest(ROOT/'cases/layernorm-tail/initial.py')==failed['source_sha256'], 'repair initial source')
    check(failed['level']=='L2' and failed['full_l2_passed'] is False and any(c['verdict']=='failed' for c in failed['correctness']['cases']), 'repair initial failure')
    check(summary['denominator']==4 and summary['full_l2_passed']==4 and summary['performance_unknown']==4 and summary['performance_confirmed']==0, 'summary counts')
    check((summary['model_calls'],summary['prompt_tokens'],summary['completion_tokens'],summary['total_tokens'])==(totals['calls'],totals['prompt'],totals['completion'],totals['total']), 'summary token totals')
    print('PASS: %d finite L2 tasks; %d performance unknown; %d calls; %d API tokens' % (summary['full_l2_passed'], summary['performance_unknown'], totals['calls'], totals['total']))
    print('Scope: sanitized projections and hashes only; no model, GPU, correctness replay, signed provenance or performance claim.')


if __name__=='__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        print('FAIL:', str(e), file=sys.stderr)
        raise SystemExit(1)
