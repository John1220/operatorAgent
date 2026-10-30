"""Finite production verification budgets: local fake models/backends only."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.budget import BudgetPlan
from operator_agent.schemas import KernelArtifact, VerificationLevel as Level, VerificationResult
from tests.unit.test_execute_inner_behavior import _Llm, _task


def controller(tmp_path, total=20, *, pipeline=False, rounds=1):
    c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path='off'),
        llm=_Llm(), use_coder_pipeline=pipeline, max_optimize_rounds=rounds,
        heartbeat_interval=1000, budget=BudgetPlan(oracle_total=total), run_root=str(tmp_path/'run'))
    c.store.output_dir = tmp_path/'output'
    return c


def inputs(n=16):
    return {'A': np.eye(n, dtype=np.float32), 'B': np.eye(n, dtype=np.float32)}


def execute(c, data):
    with contextlib.redirect_stdout(io.StringIO()):
        return c.execute(_task(), data)


def test_full_verify_reserves_all_input_groups_before_failure_and_never_refunds(tmp_path):
    c = controller(tmp_path)
    failure = RuntimeError('fixture transport did not answer')
    def blocked(*args, **kwargs):
        receipt = json.loads((tmp_path/'run/verification-budget.json').read_text())
        assert receipt['spent']['explore'] == 2
        raise failure
    c.verifier.verify = Mock(side_effect=blocked)
    with pytest.raises(RuntimeError) as raised:
        execute(c, [inputs(16), inputs(32)])
    assert raised.value is failure
    assert c._budget.spent['explore'] == 2 and c._budget.spent['confirm'] == 0
    assert c.verifier.verify.call_count == 1
    receipt = json.loads((tmp_path/'run/verification-budget.json').read_text())
    assert receipt['reservation_policy'] == 'pre_dispatch_non_refundable'
    assert receipt['unit'] == 'candidate_verification_dispatch_per_input_group'
    assert not receipt['raw_gpu_launch_cap'] and not receipt['gpu_seconds_cap']


def test_l1_failure_consumes_reservation_even_without_l3(tmp_path):
    c = controller(tmp_path)
    c.verifier.verify = Mock(return_value=[VerificationResult(Level.L1_COMPILE, False, 'fixture syntax failure')])
    artifact = execute(c, [inputs(16), inputs(32)])
    assert not artifact.all_passed and c.verifier.verify.call_count == 1
    assert artifact.metadata['budget']['spent']['explore'] == 2
    assert artifact.metadata['budget']['spent']['confirm'] == 0


def test_finalists_reserve_separately_and_cannot_use_confirmation_allowance(tmp_path):
    c = controller(tmp_path, total=3)
    c.verifier = SimpleNamespace(verify=Mock(return_value=[VerificationResult(Level.L1_COMPILE, False)]))
    selected = c._select_finalist('candidate-a', ['candidate-a', 'candidate-b', 'candidate-c'],
                                  _task(), [inputs()], None)
    assert selected[0] == 'candidate-a' and selected[2]
    assert c.verifier.verify.call_count == 2
    assert c._budget.spent['explore'] == 2 and c._budget.spent['confirm'] == 0
    assert c._oracle_budget_refused
    receipt = json.loads((tmp_path/'run/verification-budget.json').read_text())
    assert receipt['refusals']['explore'] == 1 and receipt['remaining']['confirm'] == 1
    assert 'candidate-a' not in json.dumps(receipt)


def test_pipeline_l2_and_l3_each_reserve_and_final_confirm_uses_other_pool(tmp_path):
    c = controller(tmp_path, total=5, pipeline=True)
    artifact = execute(c, inputs())
    assert artifact.all_passed
    receipt = artifact.metadata['budget']
    assert receipt['spent']['explore'] == 2  # L1/L2 finalist, then L3-only dispatch.
    assert receipt['spent']['confirm'] == 1
    saved = json.loads((Path(artifact.saved_dir)/'result.json').read_text())
    assert saved['budget'] == receipt
    assert c.store.load(artifact.saved_dir).metadata['budget'] == receipt


def test_insufficient_confirmation_pool_prevents_backend_dispatch(tmp_path):
    c = controller(tmp_path, total=3)
    c.verifier.verify_l3_only = Mock(side_effect=AssertionError('confirmation must not dispatch'))
    artifact = KernelArtifact(code='fixture source', verification_results=[])
    c._produce_remeasure(artifact, _task(), [inputs(16), inputs(32)], 0.)
    assert not artifact.remeasure.ok and not artifact.remeasure.independent
    assert c._budget.spent['confirm'] == 0
    assert c._budget.refusals['confirm'] == 1
    assert artifact.metadata['stop']['reason'] == 'budget_exhausted'
    c.verifier.verify_l3_only.assert_not_called()


def test_confirmation_exception_still_consumes_confirm_reservation(tmp_path):
    c = controller(tmp_path, total=5)
    c.verifier.verify_l3_only = Mock(side_effect=RuntimeError('fixture confirmation failed'))
    artifact = KernelArtifact(code='fixture source', verification_results=[])
    c._produce_remeasure(artifact, _task(), [inputs()], 0.)
    assert not artifact.remeasure.ok
    assert c._budget.spent['confirm'] == 1 and c.verifier.verify_l3_only.call_count == 1
    assert json.loads((tmp_path/'run/verification-budget.json').read_text())['spent']['confirm'] == 1


def test_budget_one_stops_without_verification_or_vacuous_success(tmp_path):
    c = controller(tmp_path, total=1)
    c.verifier.verify = Mock(side_effect=AssertionError('all allowance is reserved for confirmation'))
    artifact = execute(c, inputs())
    assert not artifact.all_passed
    assert artifact.metadata['stop']['reason'] == 'budget_exhausted'
    assert artifact.metadata['budget']['spent_all'] == 0
    assert artifact.verification_results[0].metrics['budget_dispatch'] is False
    c.verifier.verify.assert_not_called()
