import threading

import pytest

from operator_agent.app import hitl_tools as ht
from operator_agent.app import host_workflow as hw
from operator_agent.app import tool_calls
from operator_agent.core.contract import freeze_from_task
from operator_agent.core.run_contract import code_hash
from operator_agent.execute import profile_budget as pb
from operator_agent.execute.host_task_budget import HostTaskBudget


DECL = dict(operator='relu', request='relu fp32', shapes=[[4, 8]], precision='fp32')
SEED = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'
BUDGET = dict(max_verifications=4, wall_clock_s=600)


def prepared(tmp_path):
    run = tmp_path / 'task'
    ht.prepare(DECL, run, host_budget=BUDGET)
    contract = freeze_from_task(ht.build_task(DECL))
    return run, contract


def test_verification_refuses_while_profile_in_flight(tmp_path):
    run, contract = prepared(tmp_path)
    budget = HostTaskBudget(run, contract.hash)
    owner = 'a' * 32
    pb.ProfileBudget(run).reserve(timeout_s=60, owner_invocation_id=owner)
    with pytest.raises(Exception) as error:
        budget.reserve(code_hash(SEED))
    assert getattr(error.value, 'code', '') in ('host_profile_in_flight', 'host_gpu_busy')
    pb.ProfileBudget(run).settle_owner(owner, outcome='cancelled')
    budget.reserve(code_hash(SEED))


def test_profile_refuses_while_verification_is_in_flight(tmp_path, monkeypatch):
    run, contract = prepared(tmp_path)
    suite = contract.correctness_suite
    cases = [dict(case_id=case.case_id, verdict='passed') for case in suite.cases]
    verification = dict(
        schema='operator-tool-evidence-v1', evidence_complete=True,
        contract_hash=contract.hash, source_hash=code_hash(SEED),
        levels={'L1': True, 'L2': True},
        verdicts=[dict(level='L2', passed=True, metrics=dict(
            correctness=dict(verdict='passed', cases=cases, cases_total=len(cases))))])
    monkeypatch.setattr(hw, '_verification_materials',
                        lambda *args, **kwargs: (SEED, verification, run / 'verify-1'))
    HostTaskBudget(run, contract.hash).reserve(code_hash(SEED))
    with pytest.raises(ht.ToolError) as error:
        ht.profile(run, 1, backend='ssh', host='h20', timeout=30)
    assert error.value.code == 'profile_task_busy'
    assert pb.ProfileBudget(run).snapshot()['used'] == 0


def test_profile_ledger_settles_owner_and_retains_the_slot(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    owner = 'b' * 32
    ledger = pb.ProfileBudget(run)
    ledger.reserve(timeout_s=60, owner_invocation_id=owner)
    settled = ledger.settle_owner(owner, outcome='cancelled', cleanup='attempted')
    assert settled['settled'] == 1
    snapshot = ledger.snapshot()
    assert snapshot['in_flight'] == 0 and snapshot['used'] == 1
    assert snapshot['entries'][0]['status'] == 'cancelled'
    assert snapshot['entries'][0]['cleanup'] == 'attempted'
    assert ledger.reserve(timeout_s=60)['id'] == 2


def test_supervisor_settles_an_abandoned_profile(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    owner = 'c' * 32
    pb.ProfileBudget(run).reserve(timeout_s=60, owner_invocation_id=owner)
    call = object.__new__(tool_calls.ToolCall)
    call.run = run
    call.invocation_id = owner
    call.status = {'phase': 'cancelled'}
    call._settle_profile()
    snapshot = pb.ProfileBudget(run).snapshot()
    assert snapshot['in_flight'] == 0 and snapshot['used'] == 1
    assert snapshot['entries'][0]['status'] == 'cancelled'
    assert snapshot['entries'][0]['cleanup'] == 'not_started'


def test_profile_and_verification_reservations_are_mutually_exclusive(tmp_path, monkeypatch):
    run, contract = prepared(tmp_path)
    owner = 'd' * 32
    profile_at_reserve = threading.Event()
    verify_reserved = threading.Event()
    original = pb.ProfileBudget.reserve

    def delayed(self, **kwargs):
        profile_at_reserve.set()
        assert verify_reserved.wait(5)
        return original(self, **kwargs)

    monkeypatch.setattr(pb.ProfileBudget, 'reserve', delayed)
    result = {}

    def profile_worker():
        try:
            result['reservation'] = pb.ProfileBudget(run).reserve(
                timeout_s=60, owner_invocation_id=owner)
        except Exception as error:               # noqa: BLE001 - asserted below
            result['error'] = error

    thread = threading.Thread(target=profile_worker)
    thread.start()
    assert profile_at_reserve.wait(5)
    verify = HostTaskBudget(run, contract.hash).reserve(code_hash(SEED))
    verify_reserved.set()
    thread.join(10)
    assert result.get('error') is not None
    assert getattr(result['error'], 'code', '') == 'profile_gpu_busy'
    HostTaskBudget(run, contract.hash).finish(verify['id'], outcome='cancelled')
    retry = pb.ProfileBudget(run).reserve(timeout_s=60, owner_invocation_id='e' * 32)
    assert retry['id'] == 1
