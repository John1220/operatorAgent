"""OS-level read denial plus functional CLI/MCP under the outer DSH sandbox."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from benchmarks.experiments.s5 import s5_seatbelt as jail
from benchmarks.experiments.s5 import s5_arm_env, s5_runtime
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT

READY = sys.platform == 'darwin' and Path(jail.SANDBOX).is_file() and s5_runtime.available()


@pytest.fixture(scope='module')
def staged(tmp_path_factory):
    return jail.stage_runtime(tmp_path_factory.mktemp('s5-isolation')/'runtime')


def build(arm, root, runtime, **kwargs):
    return s5_arm_env.build(arm, base_url='http://127.0.0.1:9/v1', arm_dir=root,
        tools=True, development=dict(url='http://127.0.0.1:9/development/verify',
            token='SYNTHETIC_ONLY', contract='a'*64, backend='mock', timeout=70),
        isolation_runtime=runtime, **kwargs)


@pytest.mark.skipif(not READY, reason='macOS Seatbelt/runtime unavailable')
@pytest.mark.parametrize('arm', ['A','B','C'])
def test_file_visibility_matrix_and_child_processes(tmp_path, staged, arm):
    built = build(arm, tmp_path/'arm', staged)
    own = Path(built['work'])/'own.txt'; own.write_text('OWN_CANARY')
    forbidden = tmp_path/'other-arm'; forbidden.mkdir()
    other = forbidden/'candidate.py'; other.write_text('CROSS_ARM_CANARY')
    fake_home = tmp_path/'host-home';fake_home.mkdir()
    fake_key = fake_home/'credentials'; fake_key.write_text('FAKE_KEY_CANARY')
    ledger = tmp_path/'supervisor-ledger';ledger.write_text('PRIVATE_LEDGER_CANARY')
    alias = Path(built['work'])/'alias';alias.symlink_to(other)
    assert jail.run(built,['/bin/cat',str(own)]).stdout == 'OWN_CANARY'
    assert jail.run(built,[sys.executable,'-c','print("PY_OK")']).stdout.strip() == 'PY_OK'
    for label, target in [('sibling',other),('fake_credentials',fake_key),('ledger',ledger),
                          ('symlink',alias),('dotdot',Path(built['work'])/'../../other-arm/candidate.py'),
                          ('repository',jail.ROOT/'benchmarks/experiments/s5/s5_formal_run.py')]:
        # The unsandboxed control establishes that the target really exists.
        assert target.is_file()
        r=jail.run(built,['/bin/cat',str(target)])
        assert r.returncode != 0 and 'CANARY' not in r.stdout, (label,r)
        r=jail.run(built,[sys.executable,'-c','import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text())',str(target)])
        assert r.returncode != 0 and 'CANARY' not in r.stdout,(label,r)
    r=jail.run(built,['/bin/sh','-c','exec /bin/cat "$1"','probe',str(other)])
    assert r.returncode != 0 and 'CROSS_ARM_CANARY' not in r.stdout
    core=Path(staged['root'])/'core/operator_agent/__init__.py'
    plugin=Path(staged['root'])/'plugin/index.mjs'
    assert (jail.run(built,['/bin/cat',str(core)]).returncode == 0) is (arm != 'A')
    assert (jail.run(built,['/bin/cat',str(plugin)]).returncode == 0) is (arm == 'C')
    imported=jail.run(built,[sys.executable,'-c','import operator_agent;print("CORE_OK")'])
    assert (imported.returncode == 0) is (arm != 'A'), imported.stderr
    # Public runtime copies must not become writable even for B/C.
    code='import pathlib,sys;pathlib.Path(sys.argv[1]).write_text("OVERWRITTEN")'
    r=jail.run(built,[sys.executable,'-c',code,str(core)])
    assert r.returncode != 0
    # Hard links must not turn a denied inode into a readable workspace file.
    control=tmp_path/'control-hardlink';os.link(other,control)
    assert control.read_text()=='CROSS_ARM_CANARY';control.unlink()
    code='import os,pathlib,sys;os.link(sys.argv[1],sys.argv[2]);print(pathlib.Path(sys.argv[2]).read_text())'
    r=jail.run(built,[sys.executable,'-c',code,str(other),str(Path(built['work'])/'hardlink')])
    assert 'CROSS_ARM_CANARY' not in r.stdout and r.returncode != 0
    report=dict(arm=arm,positive_own_read=True,core_readable=arm!='A',plugin_readable=arm=='C',
                cross_arm_read=False,credential_read=False,ledger_read=False,
                symlink_escape=False,hardlink_escape=False,child_escape=False)
    (tmp_path/'matrix.json').write_text(json.dumps(report,indent=2))


@pytest.mark.skipif(not READY, reason='macOS Seatbelt/runtime unavailable')
def test_policy_or_runtime_drift_refuses_launch(tmp_path, staged):
    built=build('A',tmp_path/'arm',staged)
    policy=Path(built['isolation']['profile']);policy.write_text('(version 1)(allow default)')
    with pytest.raises(ValueError,match='policy drift'):
        jail.wrap(built,['/bin/echo','must-not-run'])
    altered=copy.deepcopy(staged);altered['files']['nonexistent.py']='0'*64
    with pytest.raises(ValueError,match='runtime drift'):
        jail.verify_runtime(altered)
    required=copy.deepcopy(built);required.pop('isolation')
    with pytest.raises(ValueError,match='policy missing'):
        jail.wrap(required,['/bin/echo','must-not-run'])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jail,'SANDBOX','/nonexistent-sandbox-runner')
        with pytest.raises(RuntimeError,match='sandbox unavailable'):
            jail.wrap(built,['/bin/echo','must-not-run'])


@pytest.mark.skipif(not READY, reason='macOS Seatbelt/runtime unavailable')
@pytest.mark.parametrize('arm', ['A','B','C'])
def test_actual_dsh_bootstraps_inside_outer_file_boundary(tmp_path, staged, arm):
    from benchmarks.experiments.s5.s5_fake_responses_upstream import FakeResponsesUpstream
    from benchmarks.experiments.s5.s5_gateway_supervisor import SupervisedTransport
    from benchmarks.experiments.s5.s5_gateway_http import GatewayServer
    from operator_agent import s5_shared_budget as shared
    from benchmarks.experiments.s5.s5_formal_run import _final_text
    path=tmp_path/'budget.sqlite'
    shared.create(path,task_id='boot-'+arm,max_calls=1,max_output_tokens=256,
                  total_output_tokens=256,wall_clock_s=60)
    with FakeResponsesUpstream(text='OK') as upstream:
        with GatewayServer(path,task_id='boot-'+arm,model='fake-model',participant='outer',
            adapter=SupervisedTransport(upstream.url,'fake-key',wall_seconds=15),
            protocol='responses',max_calls_limit=1) as gateway:
            built=s5_arm_env.build(arm,base_url=gateway.base_url,arm_dir=tmp_path/'arm',
                tools=True,api='responses',max_tokens=256,isolation_runtime=staged)
            response=s5_arm_env.start_headless(built,'Reply OK.',timeout=45)
    (tmp_path/'bootstrap.json').write_text(json.dumps(response,indent=2))
    assert response['returncode']==0,response['stderr']
    assert _final_text(response['events'])=='OK'


@pytest.mark.skipif(not READY, reason='macOS Seatbelt/runtime unavailable')
def test_probe_report_binds_actual_dsh_visibility_and_feedback(tmp_path):
    from benchmarks.experiments.s5 import s5_isolation_probe as probe
    from operator_agent.run_contract import digest
    report=probe.run(tmp_path/'acceptance')
    assert report['passed'] and report['real_model_calls']==report['gpu_executions']==0
    path=tmp_path/'acceptance/summary.json'
    readiness=probe.load_readiness(path)
    assert readiness['runtime_files_hash']==report['runtime_files_hash']
    report['records'][1]['matrix']['other']=True
    report.pop('hash');report['hash']=digest(report)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='evidence incomplete'):
        probe.load_readiness(path)


@pytest.mark.skipif(not READY, reason='macOS Seatbelt/runtime unavailable')
def test_workspace_archive_never_follows_private_links(tmp_path,staged):
    built=build('A',tmp_path/'arm',staged)
    private=tmp_path/'private';private.write_text('PRIVATE_CANARY')
    work=Path(built['work']);(work/'link').symlink_to(private);(work/'own').write_text('OWN')
    result=jail.archive_work(built,tmp_path/'archive')
    assert result['skipped_symlinks']==['link']
    assert not(tmp_path/'archive/link').exists()
    assert (tmp_path/'archive/own').read_text()=='OWN'


@pytest.mark.skipif(not READY, reason='macOS Seatbelt/runtime unavailable')
def test_isolated_workflow_final_evaluation_and_archive(tmp_path):
    from benchmarks.experiments.s5 import s5_isolation_probe as probe, s5_formal_run as workflow, s5_verify_remote
    from benchmarks.experiments.s5.s5_fake_responses_upstream import FakeResponsesUpstream
    from benchmarks.experiments.s5.s5_vertical_validation import GOOD_FIXTURE
    root=tmp_path/'run';root.mkdir()
    report=probe.run(tmp_path/'proof')
    assert report['passed']
    isolation=probe.load_readiness(tmp_path/'proof/summary.json')
    sha=s5_verify_remote.freeze(root/'services/case',declaration=probe.DECLARATION,backend='mock')
    with FakeResponsesUpstream(text=GOOD_FIXTURE) as upstream:
        plan=dict(schema=workflow.SCHEMA,implementation=workflow.runtime_identity(),hash='unit-isolated',
            read_isolation=isolation,budget=dict(max_calls=1,max_output_tokens=2048,
                total_output_tokens=2048,wall_clock_s=120),
            model=dict(endpoint=upstream.url,id='fake-model',reasoning_effort='medium'),
            development=dict(max_development=2))
        result=workflow.run_one(plan,dict(case_id='case',service_sha256=sha,text='Reply with the fixture.'),
                                'A',root,'fake-key')
    assert result['status']=='passed',result
    assert result['read_boundary']=='macos-outer-seatbelt-whitelist'
    assert result['workspace_archive']['complete']
    assert result['development']['final_used']==1
    assert (root/'runs/case/A/participant-work').is_dir()
