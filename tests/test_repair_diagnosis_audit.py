"""Offline artifact integrity only; never attest historical execution or GPU work."""
import copy
import json
from pathlib import Path

import pytest

from benchmarks import repair_diagnosis as d
from operator_agent.run_contract import code_hash, digest
from tests.test_repair_diagnoser import CASE, FIXTURES, RAW


def build(tmp_path, case=None):
    source=tmp_path/'input-case.json'
    source.write_text(json.dumps(case or CASE,ensure_ascii=False))
    args=d.parser().parse_args(['--case',str(source),'--response',str(FIXTURES/'repair_unchanged_response.json'),
                               '--output',str(tmp_path/'previews')])
    assert d.run(args)==0
    return next(Path(args.output).iterdir())


def snapshot(root):
    return {p.name:(p.read_bytes(),p.stat().st_mode) for p in root.iterdir()}


def write(path, data):
    path.chmod(0o600);path.write_text(data);path.chmod(0o444)


def change(path, fn):
    data=json.loads(path.read_text());fn(data);write(path,json.dumps(data))


def reseal(root, producer=False):
    """An attacker can rehash; independent semantic recomputation must still reject."""
    summary=json.loads((root/'summary.json').read_text())
    summary.pop('hash',None)
    summary['files']={p.name:code_hash(p.read_text()) for p in root.iterdir() if p.name!='summary.json'}
    if producer:summary['producer_source_hash']=code_hash((root/'producer-source.py').read_text())
    summary['hash']=digest(summary)
    write(root/'summary.json',json.dumps(summary))


def test_v2_artifact_readback_recomputes_without_any_write(tmp_path,monkeypatch):
    root=build(tmp_path);before=snapshot(root)
    monkeypatch.setattr(d,'atomic_json',lambda *a,**k:pytest.fail('audit wrote JSON'))
    monkeypatch.setattr(d.tempfile,'mkdtemp',lambda *a,**k:pytest.fail('audit created output'))
    proof=d.audit_preview(root)
    assert proof['completed'] and proof['evidence_consistent'] and proof['read_only']
    assert proof['producer_identity_check']=='snapshot_hash_verified_not_execution_attested'
    assert proof['source_change']=='identical_source' and proof['diagnosis_state']=='supported_hypothesis'
    assert proof['errors']==[] and not proof['restored_eligibility'] and not proof['approved']
    assert not proof['executable_plan'] and not proof['repair_credit_reset']
    assert snapshot(root)==before


def test_v1_readback_does_not_invent_missing_producer_attestation(tmp_path):
    root=build(tmp_path);(root/'producer-source.py').unlink()
    summary=json.loads((root/'summary.json').read_text());summary.pop('files');summary.pop('hash')
    summary['schema']='offline-repair-diagnosis-preview-v1'
    write(root/'summary.json',json.dumps(summary));before=snapshot(root)
    proof=d.audit_preview(root)
    assert proof['completed'] and proof['producer_identity_check']=='declared_hash_only_no_snapshot'
    assert not proof['approved'] and snapshot(root)==before


def test_unknown_diagnosis_is_consistent_data_not_a_valid_repair_context(tmp_path):
    case=copy.deepcopy(CASE);repair=case['context']['repair']
    repair['source']=repair['source'].replace('x_centered * x_centered','tl.where(mask, x_centered * x_centered, 0.0)')
    repair['parent']['source_hash']=code_hash(repair['source'])
    case['provenance']['source_sha256']=code_hash(repair['source'])
    root=build(tmp_path,case)
    assert not (root/'generation_preview.json').exists()
    proof=d.audit_preview(root)
    assert proof['completed'] and proof['diagnosis_state']=='unknown' and proof['context_hash'] is None
    assert not proof['restored_eligibility'] and not proof['executable_plan']
    (root/'generation_preview.json').write_text('{}');(root/'generation_preview.json').chmod(0o444)
    reseal(root)
    assert not d.audit_preview(root)['completed']


@pytest.mark.parametrize('what',['diagnosis','witness','span','obligation','context','policy','lineage',
    'review','response','case','input_hash','summary_hash','producer','approved','bool_type','call_count',
    'source_author','extra_field','producer_identity','schema'])
def test_rehashed_tampered_evidence_never_passes(tmp_path,what):
    root=build(tmp_path)
    if what=='diagnosis':change(root/'diagnosis.json',lambda x:x.update(correctness_proven=True))
    elif what=='witness':change(root/'diagnosis.json',lambda x:x['findings'][0]['witness'].update(unmasked_variance=0.))
    elif what=='span':change(root/'diagnosis.json',lambda x:x['findings'][0]['source_sites'][0].update(line=1))
    elif what=='obligation':change(root/'diagnosis.json',lambda x:x['findings'][0].update(repair_obligations=['relax reference']))
    elif what=='context':change(root/'generation_preview.json',lambda x:x['context']['contract'].update(min_speedup=.1))
    elif what=='policy':change(root/'generation_preview.json',lambda x:x['policy'].update(max_output_tokens=8192))
    elif what=='lineage':change(root/'generation_preview.json',lambda x:x['context']['repair_diagnosis']['bindings'].update(parent_identity_hash='f'*64))
    elif what=='review':change(root/'response_review.json',lambda x:x.update(restored_eligibility=True))
    elif what=='response':write(root/'frozen-response.json',RAW+' ')
    elif what=='case':change(root/'frozen-case.json',lambda x:x['context']['repair']['feedback']['primary'][1].update(max_error=0.))
    elif what=='input_hash':change(root/'summary.json',lambda x:x['frozen_input_hashes'].update(case='f'*64))
    elif what=='summary_hash':change(root/'summary.json',lambda x:x.update(hash='f'*64))
    elif what=='producer':write(root/'producer-source.py','different producer snapshot')
    elif what=='approved':change(root/'summary.json',lambda x:x.update(approved=True))
    elif what=='bool_type':change(root/'summary.json',lambda x:x.update(approved=0))
    elif what=='call_count':change(root/'summary.json',lambda x:x.update(model_calls=True))
    elif what=='source_author':change(root/'generation_preview.json',lambda x:x.update(source_author='template'))
    elif what=='extra_field':change(root/'summary.json',lambda x:x.update(unknown_approval='yes'))
    elif what=='producer_identity':change(root/'summary.json',lambda x:x.update(producer_source_hash='../bad'))
    else:change(root/'summary.json',lambda x:x.update(schema='unknown-preview'))
    if what!='summary_hash':reseal(root)
    before=snapshot(root);proof=d.audit_preview(root)
    assert not proof['completed'] and not proof['evidence_consistent'] and proof['errors']
    assert not proof['approved'] and not proof['restored_eligibility']
    assert snapshot(root)==before


@pytest.mark.parametrize('what',['missing','extra','symlink_file','symlink_root','writable','oversize','directory','total_cap'])
def test_incomplete_unsafe_or_unbounded_artifact_is_refused(tmp_path,what):
    root=build(tmp_path);target=root/'diagnosis.json'
    if what=='missing':target.unlink()
    elif what=='extra':(root/'execution-claim.json').write_text('{}')
    elif what=='symlink_file':target.unlink();target.symlink_to(tmp_path/'absent')
    elif what=='symlink_root':
        link=tmp_path/'link';link.symlink_to(root,target_is_directory=True);root=link
    elif what=='writable':target.chmod(0o600)
    elif what=='oversize':write(target,' '*65537)
    elif what=='directory':target.unlink();target.mkdir()
    else:
        for name in ('diagnosis.json','frozen-case.json','frozen-response.json','generation_preview.json','producer-source.py'):
            write(root/name,' '*65536)
    proof=d.audit_preview(root)
    assert not proof['completed'] and proof['errors']
    assert not proof['approved'] and proof['model_calls']==proof['gpu_measurements']==0


def test_changing_evidence_is_detected_between_snapshots(tmp_path,monkeypatch):
    root=build(tmp_path);original=d._snapshot;calls=[]
    def changing(path):
        values,hashes,modes=original(path);calls.append(1)
        if len(calls)==2:hashes['diagnosis.json']='f'*64
        return values,hashes,modes
    monkeypatch.setattr(d,'_snapshot',changing)
    proof=d.audit_preview(root)
    assert not proof['completed'] and proof['errors']==['changing_preview_evidence']


def test_current_protocol_drift_is_unknown_not_a_historical_pass(tmp_path,monkeypatch):
    root=build(tmp_path)
    monkeypatch.setattr(d,'protocol_identity',lambda:{'unexpected':'version'})
    proof=d.audit_preview(root)
    assert not proof['completed'] and proof['errors']==['audit_protocol_drift']


def test_inventory_count_is_bounded_before_any_file_read(tmp_path,monkeypatch):
    root=build(tmp_path)
    for i in range(5):(root/('extra%d'%i)).write_text('{}')
    monkeypatch.setattr(d,'read',lambda *a:pytest.fail('unbounded inventory reached file reads'))
    proof=d.audit_preview(root)
    assert not proof['completed'] and proof['errors']==['too_many_preview_files']


def test_oversize_derived_artifact_is_refused_before_creating_output(tmp_path,monkeypatch):
    original=d.build_preview
    def oversize(*a):
        result=original(*a);result['response_review']['oversize']='x'*65537
        return result
    monkeypatch.setattr(d,'build_preview',oversize)
    out=tmp_path/'no-output'
    args=d.parser().parse_args(['--case',str(FIXTURES/'repair_unchanged_case.json'),
        '--response',str(FIXTURES/'repair_unchanged_response.json'),'--output',str(out)])
    with pytest.raises(ValueError,match='artifact byte cap'):d.run(args)
    assert not out.exists()


def test_producer_snapshot_is_data_never_executed_or_attested(tmp_path):
    root=build(tmp_path)
    write(root/'producer-source.py','raise AssertionError("must not execute this data snapshot")\n')
    reseal(root,producer=True)
    proof=d.audit_preview(root)
    assert proof['completed']
    assert proof['producer_identity_check']=='snapshot_hash_verified_not_execution_attested'
    assert 'not_producer_execution_or_authorization_attestation' in proof['scope']
    assert not proof['approved']


def test_audit_cli_is_read_only_and_does_not_touch_network_arrays_or_credit(tmp_path,monkeypatch,capsys):
    from benchmarks import _numerical_repair as nr, workload_codegen_h20 as driver
    from operator_agent import llm_client as lc
    root=build(tmp_path);capsys.readouterr();before=snapshot(root)
    forbidden=lambda *a,**k:pytest.fail('audit performed side effects')
    monkeypatch.setattr(d,'atomic_json',forbidden);monkeypatch.setattr(d.tempfile,'mkdtemp',forbidden)
    monkeypatch.setattr(nr,'claim',forbidden);monkeypatch.setattr(nr,'require_available',forbidden)
    monkeypatch.setattr(lc,'bounded_dispatch',forbidden);monkeypatch.setattr(driver,'CodegenSSH',forbidden)
    monkeypatch.setattr(driver.wd,'primary',forbidden)
    assert d.run(d.parser().parse_args(['--audit',str(root)]))==0
    proof=json.loads(capsys.readouterr().out)
    assert proof['completed'] and proof['model_calls']==proof['ssh_calls']==proof['gpu_measurements']==0
    assert snapshot(root)==before


@pytest.mark.parametrize('extra',[[],['--case','missing'],['--response','missing'],
    ['--audit','missing','--output','nowrite'],['--audit','missing','--case','missing']])
def test_incomplete_or_mixed_cli_controls_stop_before_output(tmp_path,extra):
    with pytest.raises(ValueError):d.run(d.parser().parse_args(extra))
    assert not list(tmp_path.iterdir())


def test_bad_audit_emits_fixed_codes_not_arbitrary_file_text(tmp_path,capsys):
    root=build(tmp_path);capsys.readouterr()
    write(root/'summary.json','{"PRIVATE_PAYLOAD": unquoted_secret}')
    assert d.run(d.parser().parse_args(['--audit',str(root)]))==1
    out=capsys.readouterr().out
    assert 'PRIVATE_PAYLOAD' not in out and 'unquoted_secret' not in out
    assert json.loads(out)['errors']==['invalid_or_unreadable_preview_evidence']
