import json
import os
from pathlib import Path

import pytest

from operator_agent.app import hitl_tools as ht
from operator_agent.app import host_workflow as hw
from operator_agent.app import mcp_server as mcp
from operator_agent.app.tool_calls import LONG_TOOLS
from operator_agent.core.contract import freeze_from_task
from operator_agent.core.backend_caps import Cap
from operator_agent.core.run_contract import code_hash
from operator_agent.execute import profile_budget as pb


DECL = dict(operator='relu', request='relu fp32', shapes=[[4, 8]], precision='fp32')
SEED = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'benchmarks').is_dir())
SPIKE = ROOT / 'benchmarks' / 'experiments' / 'v082' / 'nvtx-spike'
BUDGET = dict(max_verifications=3, wall_clock_s=600)


class FakeBackend:
    def __init__(self, fail=False):
        self.fail = fail

    @property
    def capabilities(self):
        return frozenset({Cap.NCU_PROFILE})

    def profile(self, code, spec, *, artifact_dir, timeout=300, ncu_path=None,
                deadline=None):
        artifact_dir = Path(artifact_dir)
        if self.fail:
            (artifact_dir / 'ncu.log').write_text('ERR_NVGPUCTRPERM: profiling is not permitted\n')
            return dict(ok=False, error='profile_failed', detail='permission wall')
        for name, source in (('raw.csv', 'dual-seed.csv'),
                             ('reexport.csv', 'dual-seed.reimport.csv')):
            (artifact_dir / name).write_bytes((SPIKE / source).read_bytes())
        (artifact_dir / 'report.ncu-rep').write_bytes(b'fake-ncu-report')
        (artifact_dir / 'ncu-version.txt').write_text('Version 2025.3.0.0 (build 1)\n')
        (artifact_dir / 'ncu.log').write_text('')
        (artifact_dir / 'rc').write_text('0\n')
        (artifact_dir / 'result.json').write_text(json.dumps(dict(
            success=True, warmup=3, decoy=True, checksum=dict(first=1.0))))
        return dict(ok=True, rc='0', ncu=ncu_path or '/usr/local/cuda/bin/ncu',
                    ncu_version='Version 2025.3.0.0', gpu=3, command=['ncu'])


def prepared(tmp_path):
    run = tmp_path / 'task'
    ht.prepare(DECL, run, host_budget=BUDGET)
    return run, freeze_from_task(ht.build_task(DECL))


def verification(contract, *, l2=True, gpu=3, hw_name='nvidia h20',
                 arm_name='NVIDIA H20'):
    suite = getattr(contract, 'correctness_suite', None)
    cases = [dict(case_id=case.case_id, verdict='passed') for case in suite.cases] \
        if suite is not None else []
    return dict(
        schema='operator-tool-evidence-v1', evidence_complete=True,
        contract_hash=contract.hash, source_hash=code_hash(SEED),
        levels={'L1': True, 'L2': l2},
        verdicts=[
            dict(level='L2', passed=l2,
                 metrics=dict(correctness=dict(verdict='passed', cases=cases,
                                               cases_total=len(cases)))),
            dict(level='L3', passed=False, metrics=dict(
                selected_gpu=gpu, hw_name=hw_name,
                arm_identity=dict(candidate=dict(name=arm_name)),
                timing_receipt=dict(calibration_identity=dict(
                    runtime=dict(gpu_uuid='GPU-test'))) if gpu is not None else {}))],
    )


class EnvBackend(FakeBackend):
    def __init__(self, captured):
        super().__init__()
        self.captured = captured

    def profile(self, code, spec, *, artifact_dir, timeout=300, ncu_path=None,
                deadline=None):
        self.captured['gpu'] = os.environ.get('OPERATOR_AGENT_REQUIRED_GPU')
        self.captured['gpu_name'] = os.environ.get('OPERATOR_AGENT_REQUIRED_GPU_NAME')
        return super().profile(code, spec, artifact_dir=artifact_dir,
                               timeout=timeout, ncu_path=ncu_path, deadline=deadline)


def install(monkeypatch, fake):
    monkeypatch.setattr(ht, '_make_backend',
                        lambda *args, **kwargs: (fake, {'resolved': 'fake'}))


def test_profile_writes_a_bound_receipt_without_consuming_verification_budget(
        tmp_path, monkeypatch):
    run, contract = prepared(tmp_path)
    install(monkeypatch, FakeBackend())
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, verification(contract), run / 'verify-1'))
    from operator_agent.execute.host_task_budget import HostTaskBudget
    before = HostTaskBudget(run, contract.hash).snapshot()
    (run / 'baseline.json').write_text(json.dumps(dict(performance=dict(
        l2_flush_bytes=268435456,
        timing_channel='production-cuda-events-interleaved-l2-flush',
        timing_protocol='native-cuda-ruler-v1'))))

    result = ht.profile(run, 1, backend='ssh', host='h20', timeout=30)

    assert result['ok'] and result['profile_id'] == 1
    assert result['profiles']['used'] == 1 and result['profiles']['remaining'] == 1
    assert result['totals']['launches'] == 2
    assert result['totals']['dram_read_bytes'] == 805746176
    assert result['range_ok'] is True and result['decoy_seen'] is False
    launch = result['launches'][0]
    for key in ('registers_per_thread', 'shared_memory_bytes_per_block',
                'registers_limit_blocks_per_sm', 'shared_memory_limit_blocks_per_sm',
                'occupancy_limit_blocks_per_sm', 'resource_limits_blocks_per_sm'):
        assert key in launch
    assert result['cache_conditions']['profile']['cache_control'] == 'none'
    assert result['cache_conditions']['final']['l2_flush_bytes'] == 268435456
    receipt = json.loads((run / 'profile-1' / 'receipt.json').read_text())
    assert receipt['cache_conditions'] == result['cache_conditions']
    assert receipt['source_hash'] == code_hash(SEED)
    assert receipt['contract_hash'] == contract.hash
    assert receipt['case']['digest'] and receipt['case']['index'] == 0
    assert receipt['gpu']['index'] == 3
    assert receipt['reexport']['match'] is True
    complete = json.loads((run / 'profile-1' / 'complete.json').read_text())
    assert complete['evidence_complete'] is True
    assert complete['binding']['case_digest'] == receipt['case']['digest']
    after = HostTaskBudget(run, contract.hash).snapshot()
    assert len(after['records']) == len(before['records'])


def test_profile_budget_is_two_and_failures_count(tmp_path, monkeypatch):
    run, contract = prepared(tmp_path)
    install(monkeypatch, FakeBackend())
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, verification(contract), run / 'verify-1'))
    ht.profile(run, 1, backend='ssh', host='h20', timeout=30)
    ht.profile(run, 1, backend='ssh', host='h20', timeout=30)
    with pytest.raises(ht.ToolError, match='profile_budget'):
        ht.profile(run, 1, backend='ssh', host='h20', timeout=30)
    assert pb.ProfileBudget(run).snapshot()['used'] == 2


def test_failed_profile_still_consumes_one_slot(tmp_path, monkeypatch):
    run, contract = prepared(tmp_path)
    install(monkeypatch, FakeBackend(fail=True))
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, verification(contract), run / 'verify-1'))
    with pytest.raises(ht.ToolError, match='ncu_permission'):
        ht.profile(run, 1, backend='ssh', host='h20', timeout=30)
    snapshot = pb.ProfileBudget(run).snapshot()
    assert snapshot['used'] == 1 and snapshot['remaining'] == 1
    assert snapshot['entries'][0]['status'] == 'failed'
    assert 'ncu_permission' in snapshot['entries'][0]['error']


def test_profile_refuses_unqualified_inputs(tmp_path, monkeypatch):
    run, contract = prepared(tmp_path)
    install(monkeypatch, FakeBackend())
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, verification(contract, l2=False),
                                                 run / 'verify-1'))
    with pytest.raises(ht.ToolError, match='profile_l2_required'):
        ht.profile(run, 1, backend='ssh', host='h20', timeout=30)
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, verification(contract), run / 'verify-1'))
    with pytest.raises(ht.ToolError, match='profile_case'):
        ht.profile(run, 1, case_index=9, backend='ssh', host='h20', timeout=30)
    with pytest.raises(ht.ToolError, match='profile_backend'):
        ht.profile(run, 1, backend='fake', host='h20', timeout=30)
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, verification(contract, gpu=None),
                                                 run / 'verify-1'))
    with pytest.raises(ht.ToolError, match='profile_gpu_unknown'):
        ht.profile(run, 1, backend='ssh', host='h20', timeout=30)


def test_profile_pins_the_exact_nvidia_smi_name(tmp_path, monkeypatch):
    run, contract = prepared(tmp_path)
    captured = {}
    install(monkeypatch, EnvBackend(captured))
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, verification(
                            contract, hw_name='nvidia h20', arm_name='NVIDIA H20'),
                            run / 'verify-1'))
    ht.profile(run, 1, backend='ssh', host='h20', timeout=30)
    assert captured == {'gpu': '3', 'gpu_name': 'NVIDIA H20'}


def test_profile_requires_host_before_reserving(tmp_path, monkeypatch):
    run, contract = prepared(tmp_path)
    install(monkeypatch, FakeBackend())
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, verification(contract), run / 'verify-1'))
    with pytest.raises(ht.ToolError, match='host_required'):
        ht.profile(run, 1, backend='ssh', timeout=30)
    assert pb.ProfileBudget(run).snapshot()['used'] == 0


class MissingReportBackend(FakeBackend):
    def profile(self, code, spec, *, artifact_dir, timeout=300, ncu_path=None,
                deadline=None):
        result = super().profile(code, spec, artifact_dir=artifact_dir,
                                 timeout=timeout, ncu_path=ncu_path, deadline=deadline)
        (Path(artifact_dir) / 'report.ncu-rep').unlink()
        return result


def test_missing_raw_report_is_a_capture_failure(tmp_path, monkeypatch):
    run, contract = prepared(tmp_path)
    install(monkeypatch, MissingReportBackend())
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, verification(contract), run / 'verify-1'))
    with pytest.raises(ht.ToolError, match='profile_report_missing'):
        ht.profile(run, 1, backend='ssh', host='h20', timeout=30)
    entry = pb.ProfileBudget(run).snapshot()['entries'][0]
    assert entry['status'] == 'failed' and 'profile_report_missing' in entry['error']
    assert not (run / 'profile-1' / 'complete.json').exists()


def test_mcp_dispatch_reaches_the_host_only_profile_tool(tmp_path, monkeypatch):
    run, contract = prepared(tmp_path)
    install(monkeypatch, FakeBackend())
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, verification(contract), run / 'verify-1'))
    monkeypatch.setenv('OPERATOR_AGENT_WORKFLOW', 'host')
    reply = mcp._call_tool('operator_profile', dict(
        run_dir=str(run), verification_round=1, backend='ssh', host='h20', timeout=30))
    body = json.loads(reply['content'][0]['text'])
    assert body['ok'] is True and body['profile_id'] == 1
    assert body['profiles']['used'] == 1
    assert body['range_ok'] is True and body['decoy_seen'] is False
    host_tools = {tool['name']: tool for tool in mcp._tool_definitions(host_mode=True)}
    legacy_tools = {tool['name'] for tool in mcp._tool_definitions(host_mode=False)}
    assert host_tools['operator_profile']['inputSchema']['required'] == [
        'run_dir', 'verification_round', 'host']
    assert 'operator_profile' not in legacy_tools


def test_profile_is_a_long_tool():
    assert 'operator_profile' in LONG_TOOLS
