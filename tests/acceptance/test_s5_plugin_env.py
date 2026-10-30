"""Actual file denials and A/C public-client/real-MCP behavior, no model/GPU."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.experiments.s5 import s5_plugin_env as env, s5_plugin_probe as probe
from benchmarks.experiments.s5 import s5_seatbelt
from benchmarks.experiments.s5.s5_development_service import encoded
from operator_agent.run_contract import digest

READY=sys.platform=='darwin'and Path(s5_seatbelt.SANDBOX).is_file()


class Relay:
    def client_config(self):
        return dict(url='http://127.0.0.1:9/mcp',token='SYNTHETIC_PROXY_TOKEN',timeout=70)


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    return env.stage_runtime(tmp_path_factory.mktemp('s5-plugin-runtime')/'public')


def build(arm,root,runtime):
    return env.build(arm,runtime=runtime,relay=Relay()if arm=='C'else None,
        base_url='http://127.0.0.1:9/v1',arm_dir=root,
        tools=True,
        development=dict(url='http://127.0.0.1:9/development/verify',token='SYNTHETIC_ONLY',
            contract='a'*64,backend='mock',timeout=70))


@pytest.mark.skipif(not READY,reason='macOS Seatbelt required')
@pytest.mark.parametrize('arm',['A','C'])
def test_actual_read_matrix_core_import_and_cli_are_hidden(tmp_path,runtime,arm):
    built=build(arm,tmp_path/'participant',runtime)
    private=tmp_path/'private';private.mkdir()
    for name in('other-arm/candidate.py','host-home/credentials','supervisor-ledger'):
        p=private/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(probe.FORBIDDEN_TEXT)
    own=Path(built['work'])/'own.txt';own.write_text('OWN_CANARY')
    alias=Path(built['work'])/'alias';alias.symlink_to(private/'other-arm/candidate.py')
    provider=probe.BoundaryProvider(arm,built,private,None)
    result=subprocess.run(env.wrap(built,['/bin/sh','-c',provider.command()]),
        env=built['env'],cwd=built['work'],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==probe.matrix_expected(arm)
    assert 'PYTHONPATH'not in built['env']and built['cli_shim']is None
    assert not(Path(built['home']).parent/'bin/operator-agent').exists()
    for name in('s5-verify','s5_public_verify.py'):
        assert not(Path(built['home']).parent/'bin'/name).exists()
    read=subprocess.run(env.wrap(built,['/bin/cat',str(own)]),env=built['env'],capture_output=True,text=True)
    assert read.returncode==0 and read.stdout=='OWN_CANARY'
    for p in(private/'other-arm/candidate.py',alias):
        read=subprocess.run(env.wrap(built,['/bin/cat',str(p)]),env=built['env'],capture_output=True,text=True)
        assert read.returncode!=0 and probe.FORBIDDEN_TEXT not in read.stdout
    code='import pathlib,sys;pathlib.Path(sys.argv[1]).write_text("MODIFIED")'
    # Public plugin/proxy artifacts cannot be replaced by the participant.
    read=subprocess.run(env.wrap(built,[sys.executable,'-c',code,str(Path(runtime['root'])/'proxy.py')]),
                        env=built['env'],capture_output=True,text=True)
    assert read.returncode!=0


@pytest.mark.skipif(not READY,reason='macOS Seatbelt required')
def test_profile_runtime_drift_and_missing_boundary_refuse(tmp_path,runtime,monkeypatch):
    built=build('A',tmp_path/'participant',runtime)
    broken=copy.deepcopy(built);broken.pop('isolation')
    with pytest.raises(ValueError,match='isolation missing'):env.wrap(broken,['/bin/echo','forbidden'])
    profile=Path(built['isolation']['profile']);profile.write_text('(version 1)(allow default)')
    with pytest.raises(ValueError,match='policy drift'):env.wrap(built,['/bin/echo','forbidden'])
    changed=copy.deepcopy(runtime);changed['files']['proxy.py']='0'*64
    with pytest.raises(ValueError,match='drift'):env.verify_runtime(changed)
    monkeypatch.setattr(s5_seatbelt,'SANDBOX','/nonexistent-sandbox-runner')
    with pytest.raises(RuntimeError,match='sandbox unavailable'):env.wrap(built,['/bin/echo','forbidden'])


def test_staging_contains_no_core_or_deps(tmp_path):
    staged=env.stage_runtime(tmp_path/'runtime')
    assert set(staged['files'])=={'plugin/index.mjs','plugin/package.json','proxy.py','check.py'}
    assert not(Path(staged['root'])/'core').exists()
    assert not(Path(staged['root'])/'deps').exists()
    env.verify_runtime(staged)


def test_legacy_or_b_arm_cannot_build_new_surface(tmp_path,runtime):
    with pytest.raises(ValueError,match='only A/C'):build('B',tmp_path/'B',runtime)
    with pytest.raises(ValueError,match='real MCP relay'):
        env.build('C',runtime=runtime)
    with pytest.raises(ValueError,match='legacy isolation'):
        env.build('A',runtime=runtime,isolation_runtime={})


def test_performance_c_build_needs_no_public_verifier_config(tmp_path,runtime,monkeypatch):
    monkeypatch.setattr(env,'configure',lambda built,runtime:None)
    built=env.build('C',runtime=runtime,relay=Relay(),
        plugin_tool_timeout_ms=690000,base_url='http://127.0.0.1:9/v1',
        arm_dir=tmp_path/'participant',tools=True,development=None)
    patches=[item for patch in built['patch']for item in patch.get('insert',[])]
    mcp=next(item for item in patches if item['id']=='mcp-operator')
    assert mcp['config']['toolCallTimeoutMs']==690000
    assert 'OPERATOR_AGENT_S5_DEV_CONFIG' not in built['env']
    assert not any('PYTHONPATH' in item['config'].get('env',{}) for item in patches)


def test_plugin_timeout_must_be_explicit_without_public_verifier(tmp_path,runtime,monkeypatch):
    monkeypatch.setattr(env,'configure',lambda built,runtime:None)
    with pytest.raises(ValueError,match='bounded MCP tool timeout'):
        env.build('C',runtime=runtime,relay=Relay(),
            base_url='http://127.0.0.1:9/v1',arm_dir=tmp_path/'participant',
            tools=True,development=None)


@pytest.fixture(scope='module')
def actual_proof(tmp_path_factory):
    if not READY:pytest.skip('macOS Seatbelt required')
    root=tmp_path_factory.mktemp('s5-plugin-proof')/'acceptance'
    report=probe.run(root)
    assert report['passed'],report
    return root,report


def test_actual_dsh_common_quota_real_mcp_and_readiness(actual_proof):
    root,report=actual_proof
    assert report['real_model_calls']==report['gpu_executions']==0
    assert [r['arm']for r in report['records']]==['A','C']
    assert all(r['calls']==10 and r['development_used']==2 and r['final_used']==0
               and r['extra_public_submission_refused']for r in report['records'])
    for row in report['records']:
        progress=row['workflow_progress']
        assert progress['development_feedback_observed']==2
        assert progress['development_verifications']==2
        assert progress['runtime_reminders_injected']==progress['extra_model_calls']==0
        assert progress['first_candidate_write_step'] is not None
        assert progress['first_verification_step'] is not None
        assert not progress['problems']
    assert report['records'][0]['plugin_verification_observed']is False
    assert report['records'][1]['plugin_verification_observed']is True
    assert report['records'][1]['schema_discovered']and report['records'][1]['instructions_discovered']
    ready=probe.load_readiness(root/'summary.json')
    assert ready['runtime_files_hash']==report['runtime_files_hash']
    c=json.loads((root/'arms/C/mcp-relay/relay-snapshot.json').read_text())
    assert c['real_core_server']and not c['provider_credentials_passed']
    assert c['cleanup']=='direct_child_reaped_descendants_unknown'
    assert len([r for r in c['records']if r.get('tool')=='operator_verify'])==2
    for arm in('A','C'):
        snapshot=json.loads((root/'arms'/arm/'verification.json').read_text())
        assert [r['status']for r in snapshot['records']]==['failed','passed']
        events=json.loads((root/'arms'/arm/'events.json').read_text())
        assert probe.audit(events,snapshot,arm=arm)['complete']
        assert not any('final'==r.get('phase')for r in snapshot['records'])


def test_readiness_rejects_rehashed_boundary_claim_and_legacy_proof(actual_proof,tmp_path):
    _,report=actual_proof
    altered=copy.deepcopy(report);altered['records'][1]['matrix']['core']=True
    altered.pop('hash');altered['hash']=digest(altered)
    path=tmp_path/'fake.json';path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='evidence incomplete'):probe.load_readiness(path)
    altered=copy.deepcopy(report);altered['schema']='s5-read-isolation-probe-v1'
    altered.pop('hash');altered['hash']=digest(altered);path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='incomplete or drifted'):probe.load_readiness(path)


def test_readiness_rejects_unbound_implementation_and_report_link(actual_proof,tmp_path):
    root,report=actual_proof
    altered=copy.deepcopy(report);altered['isolation_implementation']['operator_agent/mcp_server.py']='0'*64
    altered.pop('hash');altered['hash']=digest(altered)
    path=tmp_path/'fake.json';path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='incomplete or drifted'):probe.load_readiness(path)
    link=tmp_path/'link.json';link.symlink_to(root/'summary.json')
    with pytest.raises(ValueError,match='report missing'):probe.load_readiness(link)


def test_feedback_audit_rejects_missing_or_changed_reply(actual_proof):
    root,_=actual_proof
    snapshot=json.loads((root/'arms/C/verification.json').read_text())
    events=json.loads((root/'arms/C/events.json').read_text())
    changed=copy.deepcopy(events)
    event=next(e for e in changed if e.get('type')=='tool_result'and probe._feedback(e.get('result'))is not None)
    event['result']=event['result'].replace('"status": "failed"','"status": "passed"')
    # Formatting may vary between MCP projections; force a known bound-field
    # replacement if the compact reply has no spaces.
    if event['result']==next(e['result']for e in events if e.get('callId')==event['callId']and e.get('type')=='tool_result'):
        event['result']=event['result'].replace('"failed"','"passed"',1)
    assert not probe.audit(changed,snapshot,arm='C')['complete']
    missing=[e for e in events if e.get('callId')!=event['callId']]
    assert not probe.audit(missing,snapshot,arm='C')['complete']
