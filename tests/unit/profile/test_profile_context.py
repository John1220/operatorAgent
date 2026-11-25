"""Optimize context carries the source-bound NCU experiment forward."""
import hashlib
import json
from pathlib import Path

import pytest

from operator_agent.app import hitl_tools as ht
from operator_agent.app import host_workflow as hw
from operator_agent.core.run_contract import code_hash
from operator_agent.execute import profile_budget as pb


DECL = dict(operator='relu', request='relu fp32', shapes=[[4, 8]], precision='fp32')
CODE = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'


def test_optimize_context_includes_bound_profile_summary(tmp_path):
    run = tmp_path/'task'
    prepared = ht.prepare(DECL, run, host_budget=dict(max_verifications=3, wall_clock_s=600))
    verified = ht.verify(run, CODE, backend='fake', stop_after='L2')
    directory = run/'profile-1'
    directory.mkdir()
    receipt = dict(
        schema='operator-profile-v1', profile_id=1,
        verification_round=verified['round'], source_hash=verified['source_hash'],
        gpu=dict(index=3, uuid='GPU-test', name='NVIDIA H20'),
        totals=dict(launches=1, dram_read_bytes=1024, duration_ns=2048),
        launches=[dict(kernel='_relu_kernel', nvtx='candidate', l2_hit_pct=88.0,
                       warps_pct=12.0, long_scoreboard_pct=11.0)],
        range_ok=True, decoy_seen=False, reexport=dict(match=True, reasons=[]),
        cache_conditions=dict(profile=dict(cache_control='none')),
        ncu=dict(version='fixture'), wall_s=1.0)
    receipt_path = directory/'receipt.json'
    receipt_path.write_text(json.dumps(receipt))
    complete = dict(schema='operator-profile-complete-v1', evidence_complete=True,
                    binding=dict(profile_id=1, verification_round=verified['round'],
                                 source_hash=verified['source_hash']),
                    files_sha256={'receipt.json': hashlib.sha256(receipt_path.read_bytes()).hexdigest()})
    (directory/'complete.json').write_text(json.dumps(complete))
    ledger = pb.ProfileBudget(run)
    reservation = ledger.reserve(timeout_s=60)
    ledger.finish(reservation['id'], status='completed', receipt=str(receipt_path))

    result = hw.context(run, phase='optimize', backend='fake',
                        verification_round=verified['round'],
                        hypothesis='DRAM reads include a redundant full pass',
                        expected_metric_change='Reduce dram__bytes_read.sum by about one logical input')
    optimization = result['context']['optimization']
    experiment = optimization['experiment']
    assert experiment['status'] == 'observed'
    assert experiment['bottleneck_hypothesis'].startswith('DRAM reads')
    assert experiment['observed_result'][0]['totals']['dram_read_bytes'] == 1024
    assert len(optimization['profile_evidence']) == 1
    profile = optimization['profile_evidence'][0]
    assert profile['source_hash'] == code_hash(CODE)
    assert profile['totals']['dram_read_bytes'] == 1024
    assert profile['launches'][0]['l2_hit_pct'] == 88.0
    assert optimization['experiment_protocol']['required_fields'][0] == 'source_hash'
    assert prepared['contract_hash'] == result['contract_hash']

    followup = hw.context(run, phase='optimize', backend='fake',
                          verification_round=verified['round'],
                          hypothesis='Fuse the two passes into one kernel',
                          expected_metric_change='Keep L2 correctness and reduce independent L3 latency')
    history = followup['context']['optimization']['experiment_history']
    assert [row['id'] for row in history] == [experiment['id']]
    assert history[0]['observed_result'][0]['source_hash'] == code_hash(CODE)
    with pytest.raises(ht.ToolError) as error:
        hw.context(run, phase='optimize', backend='fake',
                   verification_round=verified['round'], hypothesis='only one side')
    assert error.value.code == 'experiment_incomplete'
