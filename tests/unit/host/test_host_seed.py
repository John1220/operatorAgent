"""v0.8.2: a host task can start from an engineer-supplied source.

The seed is sealed once (source text, hash, contract), and the window-noise
receipt is validated before it can drive the selection band.
"""
import json
from pathlib import Path

import pytest

from operator_agent.app import hitl_tools as ht, mcp_server as mcp
from operator_agent.core.run_contract import code_hash


DECL = dict(operator='relu', request='relu fp32', shapes=[[4, 8]], precision='fp32')
SEED = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'
BUDGET = dict(max_verifications=3, wall_clock_s=30)


def tool(name, **args):
    result = mcp._call_tool(name, args)
    return json.loads(result['content'][0]['text'])


def preflight(**overrides):
    receipt = dict(schema='noise-preflight-v1', channel='relu-4x8-fp32-cuda-event',
                   ok=True, action='start', test_only=False,
                   # 正式窗口证据必须绑定物理 GPU：没有 UUID 的回执只能证明"某台机器安静"。
                   identity=dict(backend='ssh', shape=[4, 8],
                                 device=dict(index=5, name='NVIDIA H20',
                                             uuid='GPU-0c16a8be-9d4a-61fc-e703-0c659f2544d4')),
                   empirical_band_pct=0.4, nominal_band_pct=0.1, effective_band_pct=0.4,
                   band_consistent=False, planned_effect_pct=2.5,
                   created_at='2026-11-19T00:00:00+00:00')
    receipt.update(overrides)
    return receipt


def test_seed_is_sealed_once_and_binds_the_contract(tmp_path, monkeypatch):
    monkeypatch.setenv('OPERATOR_AGENT_WORKFLOW', 'host')
    run = tmp_path/'task'
    prepared = tool('operator_prepare', **DECL, run_dir=str(run), host_budget=BUDGET,
                    seed_source=SEED)
    manifest = json.loads((run/'seed.json').read_text())
    assert manifest['source_hash'] == code_hash(SEED)
    assert manifest['contract_hash'] == prepared['contract_hash']
    assert (run/'seed.py').read_text() == SEED
    assert prepared['seed']['source_hash'] == code_hash(SEED)
    assert prepared['next_action'] == 'verify_seed'

    again = tool('operator_prepare', **DECL, run_dir=str(run), host_budget=BUDGET,
                 seed_source=SEED)
    assert again['seed']['source_hash'] == code_hash(SEED)
    with pytest.raises(ht.ToolError, match='seed_source_binding'):
        tool('operator_prepare', **DECL, run_dir=str(run), host_budget=BUDGET,
             seed_source=SEED + '# changed\n')


def test_seed_inputs_require_the_host_workflow(tmp_path):
    with pytest.raises(ht.ToolError, match='host_seed_requires_host_workflow'):
        ht.prepare(DECL, tmp_path/'legacy', seed_source=SEED)


def test_preflight_receipt_is_validated_and_sealed(tmp_path, monkeypatch):
    monkeypatch.setenv('OPERATOR_AGENT_WORKFLOW', 'host')
    run = tmp_path/'task'
    path = tmp_path/'preflight.json'
    path.write_text(json.dumps(preflight()))
    prepared = tool('operator_prepare', **DECL, run_dir=str(run), host_budget=BUDGET,
                    preflight_receipt=str(path), seed_source=SEED)
    assert prepared['preflight']['effective_band_pct'] == 0.4
    assert prepared['preflight']['device']['uuid'] == 'GPU-0c16a8be-9d4a-61fc-e703-0c659f2544d4'
    sealed = json.loads((run/'preflight.json').read_text())
    assert sealed['action'] == 'start' and sealed['identity']['shape'] == [4, 8]

    # 缺 GPU UUID 的回执不是正式窗口证据：封存阶段就要拒绝，而不是事后才说「不知道是哪张卡」。
    no_device = preflight(identity=dict(backend='ssh', shape=[4, 8]))
    no_uuid = preflight(identity=dict(backend='ssh', shape=[4, 8],
                                      device=dict(index=5, name='NVIDIA H20')))
    not_a_uuid = preflight(identity=dict(backend='ssh', shape=[4, 8], device=dict(uuid='GPU-5')))
    only_reason = preflight(identity=dict(backend='ssh', shape=[4, 8],
                                          device_unavailable='pinned_identity() returned None '
                                                             '(no GPU pinned in this process)'))

    broken = [
        (preflight(action='hold'), 'preflight_refused'),
        (preflight(ok=False), 'preflight_refused'),
        (preflight(test_only=True), 'preflight_refused'),
        (preflight(identity=dict(backend='ssh', shape=[8, 8])), 'preflight_refused'),
        (preflight(planned_effect_pct=0.2), 'preflight_refused'),
        (no_device, 'preflight_device_identity'),
        (no_uuid, 'preflight_device_identity'),
        (not_a_uuid, 'preflight_device_identity'),
        (only_reason, 'preflight_device_identity'),
    ]
    for index, (receipt, code) in enumerate(broken):
        other = tmp_path/f'bad-{index}.json'
        other.write_text(json.dumps(receipt))
        with pytest.raises(ht.ToolError, match=code):
            ht.prepare(DECL, tmp_path/f'run-{index}', host_budget=BUDGET,
                       seed_source=SEED, preflight_receipt=str(other))
