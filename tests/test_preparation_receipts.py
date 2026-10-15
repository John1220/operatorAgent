"""Offline injection only: no SSH, model request, GPU or allocation acceptance."""
import json
import subprocess
from unittest.mock import Mock

import pytest

from benchmarks import _preparation as p
from benchmarks import workload_codegen_h20 as d
from operator_agent.remote_executor import SSHExecutor
from operator_agent import llm_client as lc
from operator_agent.run_contract import atomic_json, code_hash
from tests.test_workload_codegen_driver import prepared_execution

SECRET = 'NEVER_PERSIST_REMOTE_TEXT'


def fixture_backend(tmp_path, monkeypatch, fault=None, csv=None):
    """Use the real staged prepare and guards, fake only its transport."""
    _, plan = d.declaration(d.parser().parse_args(['--gpu', '2']))
    b = d.CodegenSSH(plan_guard=lambda: None, candidate_guard=lambda code: pytest.fail('candidate'),
        plan=plan, preparation_dir=tmp_path/'prep', host='offline')
    local = tmp_path/'trusted-runner.txt'; local.write_text('offline trusted fixture')
    remote = '/tmp/opag-fidelity-abcdefgh/run_kernel.py'
    monkeypatch.setattr(b, '_remote_files', lambda: [(local, remote)])
    calls = []; rows = csv or f'2, NVIDIA H20 {SECRET}, 8192, 0'
    def send(self, cmd, stdin_data=None):
        calls.append(cmd)
        if 'pgrep -f' in cmd: return ''
        if 'shutil.rmtree' in cmd: return json.dumps(dict(quiescent=True, removed=fault != 'cleanup'))
        if '--no-sync' in cmd:
            stage = 'runtime_probe'
            answer = json.dumps(dict(cuda=True, runtime='/runtime/python', work_dir='/runtime',
                torch=SECRET, triton=SECRET))
        elif 'nvidia-smi' in cmd:
            stage = 'gpu_query'; answer = rows
        elif 'tempfile.mkdtemp' in cmd:
            stage = 'scratch_create'; answer = json.dumps(dict(path=remote.rsplit('/', 1)[0], mode=0o700))
        elif 'hashlib' in cmd:
            stage = 'source_readback'; answer = json.dumps({remote: code_hash(local.read_text())})
        else: pytest.fail('unexpected offline command')
        assert (b.preparation.directory/f'{p.STAGES.index(stage)+1:02d}-{stage}-start.json').is_file()
        if fault == stage:
            exc = RuntimeError(SECRET); exc.ssh_rc = 255; raise exc
        if fault == 'runtime_invalid' and stage == 'runtime_probe': return SECRET
        if fault == 'cuda_unavailable' and stage == 'runtime_probe':
            return json.dumps(dict(cuda=False, runtime=SECRET, work_dir=SECRET))
        if fault == 'scratch_mode' and stage == 'scratch_create':
            return json.dumps(dict(path=remote.rsplit('/', 1)[0], mode=0o755))
        if fault == 'source_mismatch' and stage == 'source_readback': return '{}'
        return answer
    def upload(self, *args):
        calls.append('scp')
        assert (b.preparation.directory/'05-source_upload-start.json').exists()
        if fault == 'source_upload': raise subprocess.CalledProcessError(17, SECRET, stderr=SECRET)
    def rpc(self, request):
        calls.append('rpc')
        assert (b.preparation.directory/'07-runner_readback-start.json').exists()
        self.session_gpu = 2
        result = dict(device='cuda', selected_gpu=1 if fault == 'runner_readback' else 2)
        self.trace.append(dict(request=request, response=result)); return result
    monkeypatch.setattr(SSHExecutor, '_ssh_exec', send)
    monkeypatch.setattr(SSHExecutor, '_scp', upload)
    monkeypatch.setattr(d.wd.ProfileSSH, '_ssh_exec_json', rpc)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('real external transport'))
    return b, calls


def receipt_text(b):
    return '\n'.join(f.read_text() for f in b.preparation.directory.iterdir())


@pytest.mark.parametrize('fault,stage,count,reason,rc', [
    ('runtime_probe','runtime_probe',1,'ssh_command_failed',255),
    ('runtime_invalid','runtime_probe',1,'invalid_reply_or_guard',None),
    ('cuda_unavailable','runtime_probe',1,'stage_failed',None),
    ('gpu_query','gpu_query',2,'ssh_command_failed',255),
    ('scratch_create','scratch_create',3,'ssh_command_failed',255),
    ('scratch_mode','scratch_create',3,'stage_failed',None),
    ('source_upload','source_upload',4,'ssh_command_failed',17),
    ('source_mismatch','source_readback',5,'stage_failed',None),
    ('runner_readback','runner_readback',5,'stage_failed',None),
])
def test_failure_keeps_exact_stage_counts_and_no_arbitrary_text(tmp_path,monkeypatch,fault,stage,count,reason,rc):
    b, calls = fixture_backend(tmp_path,monkeypatch,fault)
    with pytest.raises((RuntimeError,ValueError,subprocess.CalledProcessError)): b.prepare(2)
    proof = b.preparation.snapshot()
    last = proof['events'][-1]
    assert proof['failed_stage'] == stage and not proof['completed']
    assert last['reason'] == reason and last['ssh_returncode'] == rc
    assert last['counts']['command'] == count
    assert len(calls) == count + (stage == 'runner_readback')
    assert SECRET not in receipt_text(b)
    assert proof['scratch_creation_attempted'] is (p.STAGES.index(stage) >= 3)
    before = list(calls)
    with pytest.raises(ValueError,match='retried'): b.prepare(2)
    assert calls == before


@pytest.mark.parametrize('csv,free,util', [('2, H20, 8192, 6',8192,6),('2, H20, 1023, 0',1023,0),
                                        ('1, H20, 8192, 0',8192,0)])
def test_gpu_ineligible_is_not_a_query_transport_failure(tmp_path,monkeypatch,csv,free,util):
    b,calls=fixture_backend(tmp_path,monkeypatch,csv=csv)
    with pytest.raises(RuntimeError): b.prepare(2)
    proof=b.preparation.snapshot()
    assert proof['failed_stage']=='gpu_select' and proof['events'][-1]['reason']=='gpu_ineligible'
    assert proof['events'][1]['state']=='passed'
    row=proof['events'][1]['details']['devices'][0]
    assert (row['free_mb'],row['utilization'])==(free,util)
    assert not proof['scratch_creation_attempted'] and len(calls)==2
    cleanup=b.cleanup(); classification=d.classify_cleanup(cleanup,b,proof)
    assert classification==dict(state='not_needed',confirmed=True,errors=[])
    assert cleanup==dict(quiescent=True,scratch_created=False) and 'removed' not in cleanup


def test_all_stages_bounded_readonly_ordered_and_no_extra_gpu_work(tmp_path,monkeypatch):
    b,calls=fixture_backend(tmp_path,monkeypatch)
    environment=b.prepare(2); proof=b.preparation.snapshot()
    assert proof['completed'] and proof['evidence_consistent'] and proof['failed_stage'] is None
    assert [e['stage'] for e in proof['events']]==list(p.STAGES)
    assert all(e['state']=='passed' for e in proof['events'])
    assert environment['selected']['index']==2 and b.counts==dict(command=5,rpc=1)
    assert len(calls)==6 and b.compile_dispatches==b.verify_dispatches==0
    assert len(list(b.preparation.directory.iterdir()))==p.POLICY['max_records']
    assert b.preparation.directory.stat().st_mode & 0o777==0o700
    for path in b.preparation.directory.iterdir():
        assert path.stat().st_mode & 0o777==0o444 and path.stat().st_size<=4096
    assert SECRET not in receipt_text(b)
    proof['events'].clear(); assert len(b.preparation.snapshot()['events'])==7
    cleanup=b.cleanup()
    assert d.classify_cleanup(cleanup,b,b.preparation.snapshot())['state']=='removed'


@pytest.mark.parametrize('fault', ['scratch_create','scratch_mode','source_upload','source_mismatch','runner_readback'])
def test_creation_attempt_or_unknown_reply_cannot_be_called_noop(tmp_path,monkeypatch,fault):
    b,_=fixture_backend(tmp_path,monkeypatch,fault)
    with pytest.raises(Exception): b.prepare(2)
    proof=b.preparation.snapshot()
    if fault=='scratch_create':
        assert b.scratch is None
        cleanup=b.cleanup()  # Raw legacy reply says "scratch_created=false".
        assert cleanup['scratch_created'] is False
        assert d.classify_cleanup(cleanup,b,proof)['state']=='unconfirmed'
    else:
        b.work_counts=dict(b.counts)
        assert d.classify_cleanup(dict(quiescent=True,scratch_created=False),b,proof)['state']=='unconfirmed'


@pytest.mark.parametrize('csv', ['',SECRET,'2, H20, NaN, 0','2, H20, 1000, -1',
    '2, H20, 1000, 101','8, H20, 1000, 0','2, H20, -1, 0','2, H20, 2147483648, 0',
    '2, H20, 8192, 0\n2, H20, 8192, 0','2, '+('x'*129)+', 1000, 0',
    '2, H20, 8192, 0\n'*9,'x'*4097])
def test_malformed_snapshot_fails_at_query_before_creation(tmp_path,monkeypatch,csv):
    b,calls=fixture_backend(tmp_path,monkeypatch,csv=csv or '\n')
    with pytest.raises(ValueError): b.prepare(2)
    proof=b.preparation.snapshot()
    assert proof['failed_stage']=='gpu_query' and proof['events'][-1]['reason']=='invalid_reply_or_guard'
    assert len(calls)==2 and b.scratch is None
    assert SECRET not in receipt_text(b)


@pytest.mark.parametrize('what',['extra','missing','changed','symlink','oversize'])
def test_receipt_readback_rejects_drift_without_backfill(tmp_path,what):
    journal=p.PreparationJournal(tmp_path/'prep',2)
    journal.step('runtime_probe',lambda: {},lambda: dict(command=1,rpc=0))
    path=journal.directory/'01-runtime_probe-finish.json'
    if what=='extra': (journal.directory/'extra.json').write_text('{}')
    elif what=='missing': path.unlink()
    elif what=='symlink': path.unlink();path.symlink_to(tmp_path/'missing')
    else:
        path.chmod(0o600); path.write_text('{}' if what=='changed' else 'x'*4097)
    before={f.name:f.read_bytes() for f in journal.directory.iterdir() if not f.is_symlink()}
    with pytest.raises(ValueError): journal.snapshot()
    assert before=={f.name:f.read_bytes() for f in journal.directory.iterdir() if not f.is_symlink()}


def test_start_receipt_failure_prevents_dispatch_and_lost_finish_is_not_success(tmp_path,monkeypatch):
    j=p.PreparationJournal(tmp_path/'prep',2); work=Mock()
    write=j._write
    monkeypatch.setattr(j,'_write',lambda *a:(_ for _ in ()).throw(OSError(SECRET)))
    with pytest.raises(OSError): j.step('runtime_probe',work,lambda:dict(command=0,rpc=0))
    work.assert_not_called()
    with pytest.raises(ValueError): j.step('runtime_probe',work,lambda:dict(command=0,rpc=0))
    work.assert_not_called()
    j=p.PreparationJournal(tmp_path/'finish-lost',2); write=j._write
    monkeypatch.setattr(j,'_write',lambda name,obj:write(name,obj) if '-start' in name else (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError): j.step('runtime_probe',work,lambda:dict(command=0,rpc=0))
    assert work.call_count==1 and not j.snapshot()['completed']
    with pytest.raises(ValueError): j.step('runtime_probe',work,lambda:dict(command=0,rpc=0))
    assert work.call_count==1


def test_concurrent_stage_cannot_dispatch_while_start_receipt_is_pending(tmp_path,monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    j=p.PreparationJournal(tmp_path/'prep',2); entered=threading.Event(); release=threading.Event()
    write=j._write; work=Mock()
    def delayed(name, value):
        if name.endswith('-start.json'):
            entered.set(); assert release.wait(5)
        write(name,value)
    monkeypatch.setattr(j,'_write',delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(j.step,'runtime_probe',work,lambda:dict(command=0,rpc=0))
        try:
            assert entered.wait(5)
            with pytest.raises(ValueError): j.step('runtime_probe',work,lambda:dict(command=0,rpc=0))
            work.assert_not_called()
        finally: release.set()
        future.result()
    assert work.call_count==1


@pytest.mark.parametrize('rc',[True,-1,256,SECRET,None])
def test_exception_return_code_and_text_cannot_inject_diagnostics(rc):
    exc=type(SECRET,(RuntimeError,),{})(SECRET); exc.ssh_rc=rc
    value=p.failure('gpu_query',exc)
    assert value==dict(reason='stage_failed',ssh_returncode=None) and SECRET not in json.dumps(value)


@pytest.mark.parametrize('what',['journal','scratch','session','pending','deployed','rpc','compile','verify','benchmark','quiescent'])
def test_noop_needs_positive_consistent_no_creation_no_runner_proof(tmp_path,monkeypatch,what):
    b,_=fixture_backend(tmp_path,monkeypatch,'gpu_query')
    with pytest.raises(RuntimeError): b.prepare(2)
    cleanup=b.cleanup(); proof=b.preparation.snapshot()
    assert d.classify_cleanup(cleanup,b,proof)['state']=='not_needed'
    if what=='journal': proof['evidence_consistent']=False
    elif what=='scratch': b.scratch='/tmp/opag-fidelity-abcdefgh'
    elif what=='session': b._session=object()
    elif what=='pending': b._pending_session=object()
    elif what=='deployed': b._deployed=True
    elif what=='rpc': b.work_counts['rpc']=1
    elif what=='compile': b.compile_dispatches=1
    elif what=='verify': b.verify_dispatches=1
    elif what=='benchmark': b.bench_dispatches=['case']
    else: cleanup['quiescent']=False
    assert d.classify_cleanup(cleanup,b,proof)['state']=='unconfirmed'
    b._session=b._pending_session=None  # Never let fake sessions reach real cleanup.


@pytest.mark.parametrize('fault,csv,stage',[('gpu_query',None,'gpu_query'),(None,'2, H20, 8192, 100','gpu_select')])
def test_driver_saves_prepare_failure_with_noop_and_zero_model_not_missing_context(tmp_path,monkeypatch,fault,csv,stage):
    prepared,args=prepared_execution(tmp_path/'runs',monkeypatch)
    b,_=fixture_backend(tmp_path,monkeypatch,fault,csv)
    # Bind this offline backend to the exact plan made by the real driver.
    b.plan=d.read_object(prepared)
    monkeypatch.setattr(d,'CodegenSSH',lambda **kwargs:b)
    monkeypatch.setattr(d.wd,'primary',lambda *a:pytest.fail('failure allocated workload'))
    monkeypatch.setattr(lc,'bounded_dispatch',lambda *a:pytest.fail('failure sent model request'))
    assert d.run(args)==1
    root=next(p for p in (tmp_path/'runs').iterdir() if p!=prepared.parent)
    summary=d.read_object(root/'summary.json'); evidence=d.read_object(root/'evidence.json')
    assert summary['status']=='preparation_failed' and not summary['accepted'] and not summary['completed']
    assert summary['cleanup_status']==dict(state='not_needed',confirmed=True,errors=[])
    assert summary['preparation']['failed_stage']==stage
    assert summary['model']['status']=='not_started' and summary['model']['evidence_consistent']
    assert summary['model']['errors']==[] and not summary['model']['completed']
    assert summary['model_attempts']==summary['verifier_dispatches']==0
    assert summary['errors']==['execution stopped: RuntimeError']
    assert evidence['preparation']==summary['preparation']
    assert 'missing/symlink evidence' not in json.dumps(summary) and SECRET not in json.dumps(summary)
    assert (prepared.parent/'execution-claim.json').exists()


@pytest.mark.parametrize('what',['extra_call','candidate','usage','counter_bool','identity','context','symlink'])
def test_zero_call_audit_rejects_fabrication_and_does_not_restore_acceptance(tmp_path,monkeypatch,what):
    prepared,args=prepared_execution(tmp_path/'runs',monkeypatch)
    b,_=fixture_backend(tmp_path,monkeypatch,'gpu_query'); b.plan=d.read_object(prepared)
    monkeypatch.setattr(d,'CodegenSSH',lambda **kwargs:b)
    assert d.run(args)==1
    root=next(p for p in (tmp_path/'runs').iterdir() if p!=prepared.parent)
    plan=d.read_object(prepared); journal=root/'model-journal'
    if what=='extra_call': atomic_json(journal/'call-001.json',{})
    elif what=='candidate': atomic_json(journal/'candidate.json',{})
    elif what=='context': atomic_json(journal/'generation-context.json',{})
    elif what=='symlink':
        path=journal/'limits.json'; path.unlink(); path.symlink_to(tmp_path/'missing')
    else:
        path=root/('llm-usage.json' if what=='usage' else 'model-budget.json')
        obj=d.read_object(path)
        if what=='usage': obj['total_calls']=1
        elif what=='identity': obj['identity']={}
        else: obj['reserved_calls']=False
        atomic_json(path,obj)
    proof=d.audit_no_model_attempt(root,plan)
    assert not proof['completed'] and not proof['evidence_consistent'] and proof['errors']


@pytest.mark.parametrize('error',[TimeoutError(SECRET),subprocess.TimeoutExpired(SECRET,1),KeyboardInterrupt(SECRET)])
def test_timeout_interrupt_have_fixed_reasons_and_consume_stage(tmp_path,error):
    j=p.PreparationJournal(tmp_path/'prep',2)
    def failed(): raise error
    with pytest.raises(type(error)): j.step('runtime_probe',failed,lambda:dict(command=1,rpc=0))
    proof=j.snapshot()
    assert proof['events'][0]['reason']==('interrupted' if isinstance(error,KeyboardInterrupt) else 'timeout')
    assert SECRET not in json.dumps(proof)
    with pytest.raises(ValueError): j.step('runtime_probe',failed,lambda:dict(command=1,rpc=0))


@pytest.mark.parametrize('counts',[{'rpc':0,'command':True},{'rpc':0,'command':513},
    {'rpc':-1,'command':0},{'rpc':0,'command':0,'extra':1}])
def test_invalid_counters_refused_before_operation(tmp_path,counts):
    j=p.PreparationJournal(tmp_path/'prep',2); work=Mock()
    with pytest.raises(ValueError): j.step('runtime_probe',work,lambda:counts)
    work.assert_not_called()
    assert list(j.directory.iterdir())==[j.directory/'policy.json']


def test_fresh_directory_and_receipt_byte_cap_do_not_expand_or_overwrite(tmp_path):
    j=p.PreparationJournal(tmp_path/'prep',2); before=(j.directory/'policy.json').read_bytes()
    with pytest.raises(FileExistsError): p.PreparationJournal(j.directory,2)
    assert (j.directory/'policy.json').read_bytes()==before
    with pytest.raises(ValueError): j._write('large.json',dict(value='x'*4097))
    assert not (j.directory/'large.json').exists()
    linked=tmp_path/'link';linked.symlink_to(j.directory,target_is_directory=True)
    with pytest.raises(ValueError): p.PreparationJournal(linked/'nested',2)
    assert not (j.directory/'nested').exists()


def test_plan_guard_failure_is_recorded_before_next_external_operation(tmp_path,monkeypatch):
    b,calls=fixture_backend(tmp_path,monkeypatch)
    guard=Mock(side_effect=[None,None,ValueError(SECRET)])
    # prepare's initial guard, runtime stage guard, then command guard.
    b.plan_guard=guard
    with pytest.raises(ValueError): b.prepare(2)
    assert calls==[]
    proof=b.preparation.snapshot()
    assert proof['failed_stage']=='runtime_probe' and proof['events'][0]['reason']=='invalid_reply_or_guard'
    assert SECRET not in receipt_text(b)


def test_zero_call_context_may_exist_but_does_not_prove_generation(tmp_path,monkeypatch):
    prepared,args=prepared_execution(tmp_path/'runs',monkeypatch)
    b,_=fixture_backend(tmp_path,monkeypatch,'gpu_query'); b.plan=d.read_object(prepared)
    monkeypatch.setattr(d,'CodegenSSH',lambda **kwargs:b)
    assert d.run(args)==1
    root=next(p for p in (tmp_path/'runs').iterdir() if p!=prepared.parent)
    plan=d.read_object(prepared); j=root/'model-journal'
    atomic_json(j/'generation-context.json',plan['generation']['context'])
    atomic_json(j/'generation-declaration.json',plan['generation'])
    proof=d.audit_no_model_attempt(root,plan)
    assert proof['status']=='refused_before_dispatch' and proof['evidence_consistent']
    assert not proof['completed'] and not proof['restored_eligibility'] and proof['errors']==[]
