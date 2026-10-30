"""Export two retained S5 plugin cases; offline verification never executes a kernel.

The same standard-library-only file is copied into the bundle as verify.py.
Hashes establish internal consistency, not authenticity or a historical GPU rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


CONTRACT = '96cdb38deafce9282485b5beaa555a738e573a1c0892292554b9d7f70685c7a7'
SOURCES = {
    'entry/development.py': '374a9981093d74f3a7ef538f0cf0fc64df01fd2bab6bb567f88c90eb4e6b28ea',
    'entry/final.py': 'df189e4c0ab4bef63000147d5ad2f5844b805ea1797c099df21dd2edce69e0fd',
    'repair/initial.py': '9e4f70dd7264c0b76387d89cc9e505a9431adb002ed77ae2435c239299c17346',
    'repair/revised.py': 'ebd15b34a5f17f91dc7fe5485809cea35377c930c144cbe339fab9ed5f4417a4',
}
SENSITIVE = re.compile(r'https?://|ssh://|/Users/|/home/|/data/|algorithm-h20|'
                       r'Bearer\s|sk-[A-Za-z0-9]{16}|-----BEGIN .*PRIVATE KEY|'
                       r'(?i:api[_-]?key|access[_-]?token)\s*[=:]')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def feedback(record_path, audit_path):
    """Only transfer identities and tool/feedback ordering, never arbitrary text."""
    record, audit = load(record_path), load(audit_path)
    sequence = record['sequence']
    matches = [row for row in audit['receipts'] if row['sequence'] == sequence]
    require(audit['complete'] is True and len(matches) == 1, 'feedback audit incomplete')
    receipt = matches[0]
    encoded = json.dumps(record['result'], sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()
    require(receipt['ingress'] == 'operator_mcp'
            and receipt['source_sha256'] == record['source_sha256']
            and receipt['service_response_sha256'] == record['response_sha256']
            and hashlib.sha256(encoded).hexdigest() == record['response_sha256']
            and receipt['call_event_index'] < receipt['result_event_index'],
            'feedback identity mismatch')
    return dict(origin_record_sha256=sha(record_path), origin_audit_sha256=sha(audit_path),
                sequence=sequence, ingress=receipt['ingress'],
                source_sha256=record['source_sha256'],
                response_sha256=record['response_sha256'],
                tool_result_sha256=receipt['tool_result_sha256'],
                tool_call_id_sha256=hashlib.sha256(receipt['tool_call_id'].encode()).hexdigest(),
                call_event_index=receipt['call_event_index'],
                result_event_index=receipt['result_event_index'],
                status=record['status'], feedback_causality_proven=False)


def project_verification(evaluation, candidate, feedback_binding=None):
    evaluation = Path(evaluation)
    service_path, raw_path = evaluation/'result.json', evaluation/'task/verify-1/result.json'
    service, raw = load(service_path), load(raw_path)
    source = sha(candidate)
    require(service['source_sha256'] == source == raw['source_hash'], 'source identity mismatch')
    require(service['contract_hash'] == CONTRACT == raw['contract_hash'], 'contract mismatch')
    require(raw['implementation_stable'] is True, 'unstable historical implementation')
    l2 = next(row for row in raw['verdicts'] if row['level'] == 'L2')
    metrics, cases = l2['metrics'], l2['metrics']['correctness']['cases']
    readings = metrics.get('input_fidelity_readings')
    fidelity = (all(item.get('input_fidelity_verified') is True for item in readings)
                if readings else metrics.get('input_fidelity_verified'))
    rows = []
    for item in cases:
        values = item.get('metrics') or {}
        rows.append(dict(case_id=item['case_id'], verdict=item['verdict'],
                         input_fidelity_verified=values.get('input_fidelity_verified'),
                         max_error=number(values.get('max_error'))))
    if feedback_binding:
        require(feedback_binding['source_sha256'] == source
                and feedback_binding['status'] == service['status'], 'feedback verdict mismatch')
    return dict(schema='s5-shareable-verification-v1', origin_result_sha256=sha(raw_path),
                origin_service_sha256=sha(service_path), source_sha256=source,
                contract_sha256=CONTRACT, status=service['status'], levels=service['levels'],
                full_l2_passed=service['full_l2_passed'],
                remote_backend_executed=service['remote_backend_executed'],
                mock_only=service['mock_only'], hardware='NVIDIA H20',
                implementation_stable=raw['implementation_stable'],
                shape=[7, 33], dtype='fp32', operator='layernorm', eps=1e-5,
                input_fidelity_verified=fidelity,
                max_error=number(metrics.get('max_error')),
                compared_elements=number(metrics.get('compared_elements')),
                bad_elements=number(metrics.get('bad_elements')),
                matched_ratio=number(metrics.get('matched_ratio')),
                suite_sha256=metrics['correctness']['suite_hash'],
                suite_verdict=metrics['correctness']['verdict'], cases=rows,
                feedback=feedback_binding)


def check_verification(row, source, passed, plugin=False, sequence=None):
    require(row['source_sha256'] == source and row['contract_sha256'] == CONTRACT,
            'projected source or contract mismatch')
    require(row['hardware'] == 'NVIDIA H20' and row['remote_backend_executed'] is True
            and row['mock_only'] is False and row['implementation_stable'] is True,
            'remote execution not demonstrated')
    require(row['shape'] == [7, 33] and row['dtype'] == 'fp32'
            and row['operator'] == 'layernorm' and row['eps'] == 1e-5,
            'task scope mismatch')
    require(row['full_l2_passed'] is passed and row['levels'] == {'L1': True, 'L2': passed}
            and row['input_fidelity_verified'] is True, 'L2 state mismatch')
    cases = row['cases']
    require(len(cases) == 12 and len({r['case_id'] for r in cases}) == 12,
            'suite denominator mismatch')
    if passed:
        require(row['status'] == 'passed' and row['suite_verdict'] == 'passed'
                and row['bad_elements'] == 0 and row['matched_ratio'] == 1.0
                and all(r['verdict'] == 'passed' and r['input_fidelity_verified'] is True
                        for r in cases), 'complete suite pass not demonstrated')
    else:
        require(row['status'] == 'failed' and row['suite_verdict'] == 'failed'
                and row['bad_elements'] == row['compared_elements'] == 231
                and row['max_error'] > 2 and all(r['verdict'] == 'not_run' for r in cases),
                'initial numerical failure or unexecuted suite misreported')
    if plugin:
        binding = row['feedback']
        require(binding['source_sha256'] == source and binding['ingress'] == 'operator_mcp'
                and binding['sequence'] == sequence and binding['status'] == row['status']
                and binding['call_event_index'] < binding['result_event_index']
                and binding['feedback_causality_proven'] is False, 'plugin feedback mismatch')


def verify(directory):
    root = Path(directory)
    manifest = load(root/'manifest.json')
    require(manifest['schema'] == 's5-shareable-manifest-v1', 'manifest schema mismatch')
    files = manifest['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(not any(p.is_symlink() for p in root.rglob('*')), 'symlink refused')
    require(actual == set(files) | {'manifest.json'}, 'file list mismatch')
    for name, checksum in files.items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts
                and sha(root/rel) == checksum, 'file hash mismatch')
        if name != 'verify.py':
            require(not SENSITIVE.search((root/rel).read_text(encoding='utf-8')),
                    'sensitive text in export')
    for rel, checksum in SOURCES.items():
        require(sha(root/rel) == checksum, 'historical candidate mismatch')
    entry_dev, entry_final = load(root/'entry/development.json'), load(root/'entry/final.json')
    initial, revised, recovery = [load(root/('repair/'+name+'.json'))
                                 for name in ('initial', 'revised', 'recovery')]
    check_verification(entry_dev, SOURCES['entry/development.py'], True, True, 1)
    check_verification(entry_final, SOURCES['entry/final.py'], True)
    check_verification(initial, SOURCES['repair/initial.py'], False, True, 1)
    check_verification(revised, SOURCES['repair/revised.py'], True, True, 2)
    check_verification(recovery, SOURCES['repair/revised.py'], True)
    require(initial['feedback']['result_event_index'] < revised['feedback']['call_event_index'],
            'repair feedback order mismatch')
    original = load(root/'repair/original-final.json')
    require(original['status'] == 'failed' and original['accepted'] is False
            and original['levels'] == {'L1': False, 'L2': None}, 'original failure overwritten')
    summary = load(root/'summary.json')
    require(summary['entry_final_full_l2_passed'] is True
            and summary['repair_original_run_accepted'] is False
            and summary['repair_recovered_final_full_l2_passed'] is True
            and summary['repair_injected_defect'] is True
            and summary['recovery_additional_model_calls'] == 0
            and summary['recovery_additional_final_verifications'] == 1
            and summary['formal_s5_credit'] is False and summary['inference_allowed'] is False
            and summary['performance_evaluated'] is False
            and summary['entry_model_calls'] == 5 and summary['repair_original_model_calls'] == 9,
            'summary contradicts retained evidence')
    return dict(verified=True, cases=2, historical_only=True, executes_candidates=False,
                calls_model=False, uses_gpu=False, authenticity_attested=False)


def build(source_root, output):
    """Read fixed historical cases; never overwrite source records or an export."""
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    require(not output.exists(), 'output already exists')
    require(not output.with_suffix('.zip').exists(), 'zip already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    entry = source_root/'20261029-s5-plugin-entry-acceptance-v1'
    repair = source_root/'20261029-s5-plugin-repair-acceptance-v1'
    erun, rrun = [b/'execution/runs/layernorm-tail/C' for b in (entry, repair)]
    audit = load(entry/'execution/acceptance-audit.json')
    recovered = load(repair/'final-recovery/summary.json')
    require(audit['acceptance']['accepted'] is True and recovered['accepted'] is True,
            'historical acceptance missing')
    require(recovered['original_run_accepted'] is False
            and recovered['original_run_status'] == 'failed', 'original repair state missing')
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.s5-cases-') as temp:
        out = Path(temp)/'bundle'
        out.mkdir()
        selections = [
            ('entry/development', erun/'supervised-verification/001-development', erun, 1),
            ('entry/final', erun/'supervised-verification/002-final', erun, None),
            ('repair/initial', rrun/'supervised-verification/001-development', rrun, 1),
            ('repair/revised', rrun/'supervised-verification/002-development', rrun, 2),
            ('repair/recovery', repair/'final-recovery', rrun, None),
        ]
        for label, attempt, run, sequence in selections:
            candidate = attempt/'candidate.py'
            if label != 'repair/recovery':
                dest = out/(label+'.py')
                dest.parent.mkdir(parents=True, exist_ok=True)
                require(sha(candidate) == SOURCES[label+'.py'], 'unexpected source bytes')
                shutil.copyfile(candidate, dest)
            else:
                require(sha(candidate) == SOURCES['repair/revised.py'], 'recovery source mismatch')
            binding = feedback(attempt/'record.json', run/'development-audit.json') if sequence else None
            row = project_verification(attempt/'evaluation', candidate, binding)
            write(out/(label+'.json'), row)
        original_path = rrun/'supervised-verification/003-final/record.json'
        original = load(original_path)
        write(out/'repair/original-final.json', dict(origin_record_sha256=sha(original_path),
              status=original['status'], levels=original['result']['levels'], accepted=False,
              source_sha256=original['source_sha256'], failure='final_answer_extraction_syntax_error'))
        er, rr = load(erun/'record.json'), load(rrun/'record.json')
        require(er['candidate_sha256'] == SOURCES['entry/final.py']
                and audit['record_sha256'] == sha(erun/'record.json'), 'entry audit source mismatch')
        write(out/'summary.json', dict(schema='s5-shareable-case-summary-v1',
              origin_entry_audit_sha256=sha(entry/'execution/acceptance-audit.json'),
              origin_recovery_summary_sha256=sha(repair/'final-recovery/summary.json'),
              origin_entry_record_sha256=sha(erun/'record.json'),
              origin_repair_record_sha256=sha(rrun/'record.json'),
              entry_final_full_l2_passed=True,
              repair_original_run_accepted=recovered['original_run_accepted'],
              repair_recovered_final_full_l2_passed=recovered['result']['full_l2_passed'],
              repair_injected_defect=True, entry_model_calls=er['model_calls']['reserved'],
              repair_original_model_calls=rr['model_calls']['reserved'],
              recovery_additional_model_calls=recovered['additional_model_calls'],
              recovery_additional_final_verifications=recovered['additional_final_verifications'],
              formal_s5_credit=False, inference_allowed=False, performance_evaluated=False,
              scope='two historical LayerNorm functional cases; finite builtin L2 suite',
              authenticity='unsigned internal consistency; original private records required'))
        (out/'README.md').write_text(
            '# S5 插件真实案例：脱敏离线材料\n\n'
            '执行 `python3 verify.py` 只核对文件、候选哈希、有限 L2 套件与反馈顺序；'
            '不执行候选，不联网，不调用模型或 GPU。候选仅供阅读。\n\n'
            '来源：2026-10-29 历史 H20 共享设备验收。entry 为真实 DSH 插件提交、'
            'GPU 反馈返回和最终 12-case L2 通过。repair 从预先注入的方差缺陷开始：'
            '主输入数值失败，12 个 suite case 尚未执行；插件反馈后修复，12-case 通过。'
            '原始最终交付因源码提取而语法失败，不能说首跑成功；修正提取后另加一次'
            '独立 GPU 终评通过，新增模型调用为零。\n\n'
            'source/contract/origin SHA256 可供持有私有档案者追查；本包是白名单投影，'
            '不含原始对话和完整日志。哈希不是签名，不证明历史记录真实性，'
            '离线核对不能重演 GPU。只覆盖这些版本、任务和有限 suite，'
            '不证明插件优于原生 DSH，不证明反馈导致修复，不包含性能成绩。\n',
            encoding='utf-8')
        shutil.copyfile(Path(__file__), out/'verify.py')
        write(out/'manifest.json', dict(schema='s5-shareable-manifest-v1', files={
            p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
        verify(out)
        out.rename(output)
    zip_path = output.with_suffix('.zip')
    require(not zip_path.exists(), 'zip already exists')
    with zipfile.ZipFile(zip_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(output).as_posix(), (2026, 10, 30, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
    checksum = sha(zip_path)
    zip_path.with_suffix('.zip.sha256').write_text(checksum+'  '+zip_path.name+'\n', encoding='ascii')
    return dict(directory=output.name, zip=zip_path.name, bytes=zip_path.stat().st_size, sha256=checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify or (Path(__file__).name == 'verify.py' and not args.output):
        result = verify(args.verify or Path(__file__).resolve().parent)
    else:
        if not (args.source_root and args.output):
            parser.error('--source-root and --output are required')
        result = build(args.source_root, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
