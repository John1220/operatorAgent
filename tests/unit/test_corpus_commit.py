"""Real local file/Conductor/crash boundaries; offline measurements, NOT CUDA proof."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from operator_agent.core import corpus_admission as a, corpus_journal as j, run_contract as rc
from operator_agent.core.artifact_store import ArtifactStore
from operator_agent.app.conductor import Conductor
from operator_agent.core.confirm import ConfirmRecord
from operator_agent.core.kernel_db import KernelDatabase, KernelRecord
from operator_agent.core.schemas import KernelArtifact, VerificationResult, VerificationLevel as L
from tests.unit.test_resume_integrity import _task, _inputs, PYTHON_MATMUL


def bound_fixture(tmp_path):
    task = _task("offline-admission", 1.05)
    code = PYTHON_MATMUL
    binding = rc.create(task, [_inputs(8)], {"name": "offline"}, {"heartbeat_interval": 5})
    m = binding["manifest"]
    first = dict(run_id=m["run_id"], manifest_hash=m["hash"], contract_hash=m["contract_hash"],
        workload_hash=m["workload_hash"], protocol_hash=rc.digest(m["measurement_protocol"]),
        candidate_hash=rc.code_hash(code), verification_id="first-verification")
    second = dict(first, verification_id="second-verification")
    metrics = dict(run_binding=first, contract_hash=m["contract_hash"], speedup_vs_baseline=2.0,
        measurement_validity="valid", measurement_can_gate=True, backend_real_device=True,
        static_guards=True, correctness_verified=True,
        eligibility={k:True for k in ("can_verdict", "can_tune", "can_persist", "can_report", "can_count_progress")},
        arm_windows={"candidate":"first-window", "baseline":"first-window"})
    confirm = ConfirmRecord(True, 1.9, "valid", independent=True, contract_hash=m["contract_hash"])
    art = KernelArtifact(kernel_id="offline-record", task_id=task.task_id, code=code,
        benchmark_data=metrics, compile_info={"run_binding":copy.deepcopy(first)},
        remeasure=confirm, metadata={"run_binding":binding}, verification_results=[
            VerificationResult(level=level, passed=True, message="offline fixture",
                metrics={"run_binding":copy.deepcopy(first)})
            for level in (L.L1_COMPILE,L.L2_FUNCTION,L.L3_PERFORMANCE)])
    art.saved_dir = str(ArtifactStore(str(tmp_path / "artifacts")).save(art,task))
    sm = dict(copy.deepcopy(metrics), run_binding=second, speedup_vs_baseline=confirm.speedup,
              arm_windows={"candidate":"second-window", "baseline":"second-window"})
    rc.atomic_json(Path(art.saved_dir) / "confirmation.json", dict(first_binding=first, second_binding=second,
        confirmation=asdict(confirm), second_metrics=sm))
    record = KernelRecord(kernel_id=art.kernel_id, algorithm=task.source.kernel_type,
        hardware=task.target.hardware, dsl=task.target.dsl, code=code, compile_info=art.compile_info,
        score=2.0, measurement_validity="valid", tags=[task.source.kernel_type,"verified","v1"])
    path = tmp_path / "corpus.jsonl"
    db = KernelDatabase(str(path),persist_retrievals=False)
    c = Conductor.__new__(Conductor); c.kernel_db = db
    return SimpleNamespace(task=task,art=art,record=record,metrics=metrics,db=db,c=c,path=path,
        folder=Path(art.saved_dir),key=j.confirmed_key(first,second),binding=binding,
        input_sets=[_inputs(8)])


@pytest.fixture
def f(tmp_path):
    return bound_fixture(tmp_path)


def events(path):
    result=[]
    for line in path.read_bytes().splitlines() if path.exists() else []:
        try:
            row=json.loads(line)
            if isinstance(row,dict) and row.get("event")=="kernel": result.append(row)
        except (ValueError,UnicodeError): pass
    return result


def prepared(f):
    f.metrics["claim_level"]=f.c._claim_level_for(f.art,f.task,f.metrics).as_dict()
    return a.prepare(f.folder,f.db,f.record,f.metrics,f.art.remeasure,"offline approved fixture")


def test_real_commit_publishes_only_after_fence_and_exact_readback(f,monkeypatch):
    original=j.fence; observed=[]
    def fence(stream,path):
        observed.append(f.db.get(f.record.kernel_id))
        return original(stream,path)
    monkeypatch.setattr(j,"fence",fence)
    assert f.c._save_to_db(f.art,f.task)
    assert observed and all(x is None for x in observed)
    meta=f.art.metadata["corpus_admission"]
    assert meta["eligible"] and meta["kept"] and meta["persisted"]
    assert meta["commit"]["status"]=="committed"
    assert f.art.benchmark_data["claim_level"]["level"]=="persisted"
    assert len(events(f.path))==1
    reloaded=KernelDatabase(str(f.path),persist_retrievals=False)
    assert reloaded.get(f.record.kernel_id).to_dict()==f.db.get(f.record.kernel_id).to_dict()
    saved=json.loads((f.folder/"result.json").read_text())
    assert saved["corpus_admission"]==meta and saved["claim_level"]["level"]=="persisted"
    assert json.loads((f.folder/"benchmark.json").read_text())["claim_level"]["level"]=="persisted"
    assert ArtifactStore().load(str(f.folder)).metadata["corpus_admission"]["persisted"]
    assert json.loads((f.folder/a.FILE).read_text())["status"]=="committed"


def test_receipt_requires_both_file_and_directory_fsync_before_readback(f,monkeypatch):
    import stat
    stages=[]; fsync=j.os.fsync; scan=j.scan
    def sync(fd):
        stages.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        assert f.db.get(f.record.kernel_id) is None
        return fsync(fd)
    def read(stream):
        stages.append("scan")
        assert f.db.get(f.record.kernel_id) is None
        return scan(stream)
    monkeypatch.setattr(j.os,"fsync",sync);monkeypatch.setattr(j,"scan",read)
    assert f.db.commit(f.record,f.key).persisted
    assert stages==["scan","file","directory","scan"]


def test_readback_mismatch_cannot_install_memory_or_issue_success(f,monkeypatch):
    scan=j.scan;calls=[]
    def wrong(stream):
        state=scan(stream);calls.append(1)
        if len(calls)==2:
            state.admissions[j.admission_id(f.key)]["record"]["score"]=9
        return state
    monkeypatch.setattr(j,"scan",wrong)
    result=f.db.commit(f.record,f.key)
    assert result.status=="outcome_unknown" and not result.persisted
    assert f.db.get(f.record.kernel_id) is None


def test_qualification_is_preserved_when_writing_fails_and_report_does_not_lie(f,monkeypatch):
    from operator_agent.core.report import write_report
    monkeypatch.setattr(j,"locked_file",lambda *x,**kw: (_ for _ in ()).throw(PermissionError("disk readonly")))
    assert not f.c._save_to_db(f.art,f.task)
    meta=f.art.metadata["corpus_admission"]
    assert meta["eligible"] and not meta["kept"] and not meta["persisted"]
    assert meta["commit"]["status"]=="failed" and "disk readonly" in meta["reason"]
    assert f.art.benchmark_data["measurement_validity"]=="valid"
    assert f.art.benchmark_data["eligibility"]["can_persist"] is True
    assert f.art.benchmark_data["claim_level"]["level"]=="promotable"
    assert f.db.get(f.record.kernel_id) is None and not events(f.path)
    write_report(f.folder,artifact=f.art,task=f.task)
    assert "disk readonly" in (f.folder/"report.md").read_text()
    assert json.loads((f.folder/"result.json").read_text())["corpus_admission"]["persisted"] is False


@pytest.mark.parametrize("stage",["file-fsync","directory-fsync","readback"])
def test_post_write_failure_is_unknown_not_success_and_retry_does_not_append(f,monkeypatch,stage):
    req=prepared(f)
    with monkeypatch.context() as patch:
        if stage=="file-fsync": patch.setattr(j.os,"fsync",lambda fd: (_ for _ in ()).throw(OSError("fsync failed")))
        if stage=="directory-fsync": patch.setattr(j,"fsync_directory",lambda path: (_ for _ in ()).throw(OSError("dir fsync failed")))
        if stage=="readback":
            original=j.scan; calls=[]
            def bad(stream):
                calls.append(1)
                if len(calls)==2: raise OSError("readback failed")
                return original(stream)
            patch.setattr(j,"scan",bad)
        result=f.db.commit(f.record,req["key"])
        assert result.status=="outcome_unknown" and not result.persisted
        assert f.db.get(f.record.kernel_id) is None
    before=f.path.read_bytes()
    assert len(events(f.path))==1
    result=a.recover(f.folder,f.db)
    assert result.status=="already_committed" and result.persisted
    assert f.path.read_bytes()==before and len(events(f.path))==1


def test_disabled_harness_store_never_claims_durable_success(f):
    f.c.kernel_db=KernelDatabase("off")
    assert not f.c._save_to_db(f.art,f.task)
    assert f.art.metadata["corpus_admission"]["commit"]["status"]=="persistence_disabled"
    assert f.art.metadata["claim_level"]["level"]=="promotable"
    assert not f.path.exists()


def test_idempotent_live_submission_and_fresh_instance_recovery(f):
    assert f.c._save_to_db(f.art,f.task)
    before=f.path.read_bytes()
    assert f.c._save_to_db(f.art,f.task)
    assert f.art.metadata["corpus_admission"]["commit"]["status"]=="already_committed"
    assert a.recover(f.folder,KernelDatabase(str(f.path),persist_retrievals=False)).persisted
    assert f.path.read_bytes()==before and len(events(f.path))==1


def test_same_admission_with_changed_payload_is_refused_without_overwrite(f):
    assert f.db.commit(f.record,f.key).persisted
    before=f.path.read_bytes()
    changed=copy.deepcopy(f.record); changed.score=3
    result=f.db.commit(changed,f.key)
    assert result.status=="refused" and "different payload" in result.reason
    assert f.path.read_bytes()==before and f.db.get(changed.kernel_id).score==2


def test_fresh_measurement_ids_allow_another_legitimate_admission(f):
    assert f.db.commit(f.record,f.key).persisted
    other=copy.deepcopy(f.record); other.kernel_id="next-measured-record"
    first=dict(f.metrics["run_binding"],verification_id="new-first")
    second=dict(first,verification_id="new-confirm")
    other.compile_info["run_binding"]=first
    key=j.confirmed_key(first,second)
    assert j.admission_id(key)!=j.admission_id(f.key)
    assert f.db.commit(other,key).persisted and len(events(f.path))==2


def test_replay_and_installed_memory_do_not_alias_caller_or_double_count_retrievals(f):
    assert f.db.commit(f.record,f.key).persisted
    f.record.compile_info["run_binding"]["verification_id"]="mutated-caller"
    assert f.db.get(f.record.kernel_id).compile_info["run_binding"]["verification_id"]=="first-verification"
    f.db._persist_retrievals=True
    f.db.note_retrievals([f.record.kernel_id])
    assert f.db.get(f.record.kernel_id).retrieved_count==1
    f.db.replay(); f.db.replay()
    assert f.db.get(f.record.kernel_id).retrieved_count==1


@pytest.mark.parametrize("ending",[b'{"event":"kernel","record":',b'{"event":"kernel","record":{"kernel_id":"fake","algorithm":"matmul","hardware":"mock_gpu","dsl":"python","code":"# fake"}}'])
def test_unterminated_tail_is_preserved_but_cannot_be_sealed_by_next_append(f,ending):
    f.path.write_bytes(ending)
    db=KernelDatabase(str(f.path),persist_retrievals=False)
    assert db.get("fake") is None and db.disk_skipped_lines==1
    assert db.commit(f.record,f.key).persisted
    assert f.path.read_bytes().startswith(ending+b"\x00\n")
    again=KernelDatabase(str(f.path),persist_retrievals=False)
    assert again.get("fake") is None and again.get(f.record.kernel_id) is not None
    assert again.disk_skipped_lines==1 and len(events(f.path))==1


def test_retrieval_append_also_quarantines_a_torn_tail(f):
    assert f.db.commit(f.record,f.key).persisted
    with f.path.open("ab") as stream: stream.write(b'{"event":"retrieved","ids":["offline-record"]}')
    f.db._persist_retrievals=True; f.db.note_retrievals([f.record.kernel_id])
    assert KernelDatabase(str(f.path)).get(f.record.kernel_id).retrieved_count==1


@pytest.mark.parametrize("field",["score","code","timestamp","id","key"])
def test_hash_checked_events_detect_tampering_and_block_new_writes(f,field):
    assert f.db.commit(f.record,f.key).persisted
    event=events(f.path)[0]
    if field=="score": event["record"]["score"]=3
    if field=="code": event["record"]["code"]+="# changed"
    if field=="timestamp": event["record"]["created_at"]+=1
    if field=="id": event["admission"]["id"]="0"*64
    if field=="key": event["admission"]["key"]["first_verification_id"]="other"
    f.path.write_text(json.dumps(event)+"\n")
    db=KernelDatabase(str(f.path),persist_retrievals=False)
    assert db.get(f.record.kernel_id) is None and db.disk_skipped_lines==1
    before=f.path.read_bytes(); result=db.commit(f.record,f.key)
    assert result.status=="refused" and "corrupt" in result.reason and f.path.read_bytes()==before


@pytest.mark.parametrize("score",[float("nan"),float("inf"),True,"2"])
def test_bad_scalars_fail_before_creating_a_corpus_file(f,score):
    f.record.score=score
    assert not f.db.commit(f.record,f.key).persisted
    assert not f.path.exists() and f.db.get(f.record.kernel_id) is None


@pytest.mark.parametrize("fault",["source","threshold","window","verification","metrics","request","request-hash","corpus","missing"])
def test_prepared_recovery_drift_holds_before_any_append(f,monkeypatch,fault):
    req=prepared(f)
    if fault=="source": (f.folder/"code.py").write_text("# changed")
    if fault in {"window","verification","threshold"}:
        path=f.folder/"confirmation.json"; cf=json.loads(path.read_text())
        if fault=="window": cf["second_metrics"]["arm_windows"]["candidate"]="first-window"
        if fault=="verification": cf["second_binding"]["verification_id"]="first-verification"
        if fault=="threshold": cf["confirmation"]["speedup"]=.9
        path.write_text(json.dumps(cf))
    if fault=="metrics":
        p=f.folder/"benchmark.json"; m=json.loads(p.read_text()); m["speedup_vs_baseline"]=9; p.write_text(json.dumps(m))
    if fault=="request":
        p=f.folder/a.FILE; e=json.loads(p.read_text());e["request"]["record"]["score"]=9;p.write_text(json.dumps(e))
    if fault=="request-hash":
        p=f.folder/a.FILE;e=json.loads(p.read_text());e["hash"]="0"*64;p.write_text(json.dumps(e))
    if fault=="missing": (f.folder/"confirmation.json").unlink()
    db=KernelDatabase(str(f.path.parent/"different.jsonl")) if fault=="corpus" else f.db
    monkeypatch.setattr(db,"commit",lambda *x,**kw:pytest.fail("HOLD attempted commit"))
    result=a.recover(f.folder,db)
    assert result.status=="recovery_hold" and not result.persisted and not f.path.exists()


def test_prepared_recovery_refuses_evaluator_drift_without_appending(f,monkeypatch):
    prepared(f)
    monkeypatch.setattr(rc,"protocol_identity",lambda:{"changed":"0"*64})
    monkeypatch.setattr(f.db,"commit",lambda *x,**kw:pytest.fail("drift attempted commit"))
    assert a.recover(f.folder,f.db).status=="recovery_hold"


@pytest.mark.parametrize("key",[{},False,[],{"kind":"legacy-record","record_hash":"z"*64}])
def test_explicit_invalid_key_cannot_fall_back_to_legacy(f,key):
    result=f.db.commit(f.record,key)
    assert not result.persisted and not f.path.exists()
    assert f.db.get(f.record.kernel_id) is None


@pytest.mark.parametrize("field",["run_id","manifest_hash","workload_hash","candidate_hash","first_verification_id"])
def test_key_record_identity_conflicts_fail_before_opening_the_corpus(f,field):
    key=dict(f.key); key[field]="other" if field=="first_verification_id" else "0"*len(key[field])
    result=f.db.commit(f.record,key)
    assert not result.persisted and "binding conflict" in result.reason
    assert not f.path.exists()


def test_legacy_incomplete_update_does_not_hide_the_valid_prefix(f):
    f.path.write_text(json.dumps(dict(event="kernel",record=f.record.to_dict()))+"\n"+
        json.dumps(dict(event="kernel",record={"kernel_id":f.record.kernel_id}))+"\n")
    db=KernelDatabase(str(f.path),persist_retrievals=False)
    assert db.get(f.record.kernel_id).score==2 and db.loaded_from_disk==1
    assert db.disk_skipped_lines==1 and "持久化异常" in db.summary()
    before=f.path.read_bytes()
    assert db.commit(f.record,f.key).status=="refused" and f.path.read_bytes()==before


def test_exact_physical_duplicate_is_deduplicated_before_retrieval_replay(f):
    assert f.db.commit(f.record,f.key).persisted
    payload=f.path.read_bytes()
    f.path.write_bytes(payload+payload+json.dumps(dict(event="retrieved",ids=[f.record.kernel_id])).encode()+b"\n")
    db=KernelDatabase(str(f.path),persist_retrievals=False)
    assert db.loaded_from_disk==1 and db.disk_duplicate_admissions==1
    assert db.get(f.record.kernel_id).retrieved_count==1
    assert db.commit(f.record,f.key).status=="already_committed"
    assert db.get(f.record.kernel_id).retrieved_count==1


def test_different_sealed_event_for_same_admission_blocks_overwrite(f):
    assert f.db.commit(f.record,f.key).persisted
    event=events(f.path)[0]; event["record"]["score"]=3
    event["admission"]["record_hash"]=j.record_hash(event["record"])
    event["admission"]["event_hash"]=j.digest({k:v for k,v in event.items() if k!="admission"})
    with f.path.open("ab") as stream: stream.write(json.dumps(event).encode()+b"\n")
    db=KernelDatabase(str(f.path),persist_retrievals=False)
    assert db.get(f.record.kernel_id).score==2 and "conflicting duplicate" in db.persistence_error
    before=f.path.read_bytes()
    assert db.commit(f.record,f.key).status=="refused" and f.path.read_bytes()==before


def test_unexpected_second_writer_fails_closed_without_append_or_memory(f):
    with j.locked_file(f.path,create=True):
        result=f.db.commit(f.record,f.key)
        assert not result.persisted and result.status=="failed"
        assert f.db.get(f.record.kernel_id) is None and f.path.read_bytes()==b""
    assert f.db.commit(f.record,f.key).persisted


def test_partial_write_exception_remains_unknown_and_can_be_reconciled(f,monkeypatch):
    prepared(f)
    with monkeypatch.context() as patch:
        def partial(stream,event,**kw):
            payload=json.dumps(event).encode(); stream.write(payload[:len(payload)//2])
            raise OSError("short disk write")
        patch.setattr(j,"append",partial)
        assert f.db.commit(f.record,f.key).status=="outcome_unknown"
        assert f.db.get(f.record.kernel_id) is None
    tail=f.path.read_bytes()
    assert a.recover(f.folder,f.db).persisted
    assert f.path.read_bytes().startswith(tail+b"\x00\n") and len(events(f.path))==1


def test_write_failure_does_not_replace_the_existing_durable_best(f,monkeypatch):
    old=copy.deepcopy(f.record);old.score=1.5
    assert f.db.commit(old).persisted
    before=f.path.read_bytes()
    monkeypatch.setattr(j,"locked_file",lambda *x,**kw: (_ for _ in ()).throw(PermissionError("read only")))
    assert not f.c._save_to_db(f.art,f.task)
    assert f.art.metadata["corpus_admission"]["eligible"]
    assert f.db.get(f.record.kernel_id).score==1.5 and f.path.read_bytes()==before


@pytest.mark.parametrize("fault",["missing-proof","intent-write"])
def test_bound_admission_cannot_skip_preparation_and_use_legacy(f,monkeypatch,fault):
    if fault=="missing-proof": (f.folder/"confirmation.json").unlink()
    else:
        original=rc.atomic_json
        def fail(path,data,**kw):
            if Path(path).name==a.FILE: raise OSError("intent fsync failed")
            return original(path,data,**kw)
        monkeypatch.setattr(rc,"atomic_json",fail)
    monkeypatch.setattr(f.db,"commit",lambda *x,**kw:pytest.fail("unprepared admission attempted commit"))
    assert not f.c._save_to_db(f.art,f.task)
    assert f.art.metadata["corpus_admission"]["eligible"] and not f.path.exists()


def test_rehashed_intent_does_not_replace_the_actual_measurement_evidence(f,monkeypatch):
    prepared(f)
    path=f.folder/a.FILE; envelope=json.loads(path.read_text());req=envelope["request"]
    req["first_metrics"]["speedup_vs_baseline"]=9;req["record"]["score"]=9
    req["record_hash"]=j.record_hash(req["record"]);envelope["hash"]=a._request_hash(req)
    path.write_text(json.dumps(envelope))
    monkeypatch.setattr(f.db,"commit",lambda *x,**kw:pytest.fail("rehash bypass attempted commit"))
    result=a.recover(f.folder,f.db)
    assert result.status=="recovery_hold" and not f.path.exists()


def test_malformed_evidence_structure_returns_hold_before_commit(f,monkeypatch):
    prepared(f)
    path=f.folder/"confirmation.json";data=json.loads(path.read_text());data["second_metrics"]=[]
    path.write_text(json.dumps(data))
    monkeypatch.setattr(f.db,"commit",lambda *x,**kw:pytest.fail("malformed evidence attempted commit"))
    assert a.recover(f.folder,f.db).status=="recovery_hold"


def test_replay_fence_failure_is_visible_and_does_not_install_the_event(f,monkeypatch):
    assert f.db.commit(f.record,f.key).persisted
    monkeypatch.setattr(j,"fence",lambda *x: (_ for _ in ()).throw(OSError("replay fsync failed")))
    db=KernelDatabase(str(f.path),persist_retrievals=False)
    assert db.loaded_from_disk==0 and db.get(f.record.kernel_id) is None
    assert "replay fsync failed" in db.summary()


def test_publication_failure_keeps_journal_truth_and_retry_repairs_artifacts(f,monkeypatch):
    with monkeypatch.context() as patch:
        original=rc.atomic_json
        def fail(path,data,**kw):
            if Path(path).name=="benchmark.json": raise OSError("benchmark publication failed")
            return original(path,data,**kw)
        patch.setattr(rc,"atomic_json",fail)
        assert f.c._save_to_db(f.art,f.task)
        assert f.art.metadata["corpus_admission"]["persisted"]
        assert "publication failed" in f.art.metadata["corpus_admission"]["publication_error"]
    before=f.path.read_bytes()
    assert a.recover(f.folder,f.db).status=="already_committed"
    assert f.path.read_bytes()==before
    metrics=json.loads((f.folder/"benchmark.json").read_text())
    assert metrics["claim_level"]["level"]=="persisted" and metrics["measurement_validity"]=="valid"


def test_explicit_recovery_cli_is_idempotent_and_wrong_corpus_holds(f):
    prepared(f)
    cmd=[sys.executable,"-m","operator_agent.core.corpus_admission","--artifact",str(f.folder),"--corpus",str(f.path)]
    for status in ("committed","already_committed"):
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stdout+result.stderr
        assert json.loads(result.stdout)["status"]==status and len(events(f.path))==1
    before=f.path.read_bytes(); other=f.path.parent/"other.jsonl"
    result=subprocess.run(cmd[:-1]+[str(other)],capture_output=True,text=True,timeout=30)
    assert result.returncode==1 and json.loads(result.stdout)["status"]=="recovery_hold"
    assert not other.exists() and f.path.read_bytes()==before


def test_production_confirmation_producer_feeds_prepared_commit_without_replay_work(f):
    # The L3 reading is offline/scripted; the binding, confirmation sidecar,
    # qualification, commit and recovery are real production implementations.
    c=f.c;c._run_binding=f.binding;c._run_inputs=f.input_sets
    c._run_controls=f.binding["manifest"]["controls"];c._run_binding_dir=f.folder
    c._environment={"fingerprint":f.binding["manifest"]["environment"]}
    c._heartbeat_for_next_measurement=lambda:(None,False)
    import time
    from operator_agent.execute.budget import BudgetLedger, BudgetPlan
    calls=[]
    def verify(**kw):
        calls.append(kw)
        metrics=copy.deepcopy(f.metrics);metrics["speedup_vs_baseline"]=1.9
        metrics["arm_windows"]={"candidate":"actual-new-window","baseline":"actual-new-window"}
        return [VerificationResult(level=L.L3_PERFORMANCE,passed=True,message="offline L3",metrics=metrics)]
    c.verifier=SimpleNamespace(verify_l3_only=verify)
    c.run_root=''
    c._run_started=time.monotonic()
    c._budget=BudgetLedger(BudgetPlan(oracle_total=10))
    c._produce_remeasure(f.art,f.task,f.input_sets,0)
    confirmation=json.loads((f.folder/"confirmation.json").read_text())
    assert confirmation["first_binding"]==f.metrics["run_binding"]
    assert confirmation["second_binding"]["verification_id"]!="first-verification"
    assert c._save_to_db(f.art,f.task)
    assert a.recover(f.folder,KernelDatabase(str(f.path),persist_retrievals=False)).persisted
    assert len(calls)==1 and c._budget.spent['confirm']==1 and len(events(f.path))==1


@pytest.mark.parametrize("stage",["before-append","partial-line","after-append","after-fsync","after-memory","after-receipt","after-result"])
def test_actual_process_death_recovers_one_event_without_remeasuring(f,stage):
    prepared(f)
    script=r'''
import os,sys,json
from pathlib import Path
from operator_agent.core import corpus_admission as a, corpus_journal as j, run_contract as rc
from operator_agent.core.kernel_db import KernelDatabase
db=KernelDatabase(sys.argv[2],persist_retrievals=False); stage=sys.argv[3]
if stage in ('before-append','partial-line','after-append'):
 original=j.append
 def stop(stream,event,**kw):
  if stage=='partial-line':
   data=json.dumps(event).encode();stream.write(data[:len(data)//2]);os.fsync(stream.fileno())
  if stage=='after-append': original(stream,event,**kw)
  os._exit(73)
 j.append=stop
if stage=='after-fsync':
 original=j.fence
 def stop(stream,path): original(stream,path);os._exit(73)
 j.fence=stop
if stage=='after-memory':
 original=db._install_scan
 def stop(scan): original(scan);os._exit(73)
 db._install_scan=stop
if stage in ('after-receipt','after-result'):
 original=rc.atomic_json
 def stop(path,data,**kw):
  original(path,data,**kw)
  if ((stage=='after-receipt' and Path(path).name==a.FILE and data.get('status')=='committed')
      or (stage=='after-result' and Path(path).name=='result.json' and data.get('corpus_admission',{}).get('persisted'))):os._exit(73)
 rc.atomic_json=stop
result=a.recover(sys.argv[1],db)
print(result.as_dict());sys.exit(2)
'''
    result=subprocess.run([sys.executable,"-c",script,str(f.folder),str(f.path),stage],
        cwd=str(Path(__file__).resolve().parents[2]),capture_output=True,text=True,timeout=30)
    assert result.returncode==73,result.stdout+result.stderr
    db=KernelDatabase(str(f.path),persist_retrievals=False)
    result=a.recover(f.folder,db)
    assert result.persisted,result.reason
    assert len(events(f.path))==1
    before=f.path.read_bytes()
    assert a.recover(f.folder,KernelDatabase(str(f.path),persist_retrievals=False)).status=="already_committed"
    assert f.path.read_bytes()==before
    assert json.loads((f.folder/a.FILE).read_text())["commit"]["persisted"]
    assert json.loads((f.folder/"result.json").read_text())["corpus_admission"]["persisted"]
    assert ArtifactStore().load(str(f.folder)).benchmark_data["claim_level"]["level"]=="persisted"
