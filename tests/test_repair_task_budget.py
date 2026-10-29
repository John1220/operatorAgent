"""Production repair budgets: synthetic clients/workers, no HTTP, SSH or GPU."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from operator_agent import cli, hitl_tools as ht, llm_client, mcp_server, repair_loop
from operator_agent import task_budget, task_cancellation, tool_calls
from operator_agent.schemas import VerificationLevel as Level, VerificationResult


DECL = dict(request='matmul fp32', shapes=[[8, 8, 8]], min_speedup=1)
CODE = 'def kernel(A, B):\n    return A @ B\n'
LIMITS = dict(wall_clock_s=12.5, max_llm_calls=3, oracle_total=5)


class OfflineClient:
    def __init__(self, *, call_budget, **kwargs):
        self.call_budget = call_budget
        self.identity = dict(model='offline-fixture', endpoint_sha256='0'*64)
        call_budget.bind(self.identity)

    def call(self):
        budget = self.call_budget
        number, _ = budget.reserve(self.identity, json.dumps(dict(model='offline-fixture',
            max_tokens=1)), 'Fixer', 1)
        budget.finish(number, outcome='answered', wall_seconds=.001)


def outcome(code=CODE, passed=True, rounds=None):
    results = [VerificationResult(Level.L1_COMPILE, passed, 'offline fixture')]
    return dict(code=code, passed=passed, rounds=rounds or [], results=results, log='',
                initial_full_l2_passed=None, final_full_l2_passed=None,
                candidate_records=[], selected_source_hash=ht.code_hash(code))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    run = tmp_path/'run'
    ht.prepare(DECL, run)
    monkeypatch.setattr(ht, '_llm_problem', lambda: None)
    monkeypatch.setattr(llm_client, 'LLMClient', OfflineClient)
    monkeypatch.setattr(ht, '_make_backend', lambda *a, **kw: (object(), {}))
    monkeypatch.setattr(ht, 'Verifier', lambda backend: object())
    from operator_agent import fixer
    monkeypatch.setattr(fixer, 'FixerAgent', lambda llm: SimpleNamespace(llm=llm))
    monkeypatch.setattr(tool_calls, '_REPAIR_CONTEXT', None)
    return run


@pytest.mark.parametrize('key,value', [
    ('wall_clock_s', True), ('wall_clock_s', None), ('wall_clock_s', 0),
    ('wall_clock_s', float('nan')), ('wall_clock_s', float('inf')),
    ('max_llm_calls', True), ('max_llm_calls', 0), ('max_llm_calls', 1.5),
    ('oracle_total', False), ('oracle_total', 0), ('oracle_total', '5'),
])
def test_invalid_repair_budget_precedes_configuration_model_backend_and_storage(tmp_path, monkeypatch, key, value):
    unexpected = Mock(side_effect=AssertionError('must refuse before external preparation'))
    monkeypatch.setattr(ht, '_llm_problem', unexpected)
    monkeypatch.setattr(ht, '_make_backend', unexpected)
    monkeypatch.setattr(llm_client, 'LLMClient', unexpected)
    run = tmp_path/'not-created'
    with pytest.raises(ht.ToolError) as caught:
        ht.repair(run, CODE, **{key: value})
    assert caught.value.code == 'invalid_task_budget'
    assert not run.exists() and not unexpected.called


def test_no_initial_oracle_allowance_does_not_create_model_or_backend(prepared, monkeypatch):
    unexpected = Mock(side_effect=AssertionError('no model/backend before verification allowance'))
    monkeypatch.setattr(ht, '_make_backend', unexpected)
    monkeypatch.setattr(llm_client, 'LLMClient', unexpected)
    result = ht.repair(prepared, CODE, oracle_total=1)
    assert result['stopped_reason'] == 'verification_budget_exhausted'
    assert not result['passed'] and result['rounds'] == []
    assert result['budget']['verification']['spent_all'] == 0
    assert result['budget']['verification']['remaining']['confirm'] == 1
    assert not unexpected.called
    assert Path(result['evidence']).is_file()


def test_each_repair_gets_a_fresh_non_overwriting_journal_and_actual_reservations(prepared, monkeypatch):
    def loop(code, task, inputs, verifier, fixer, *, before_verification, **kwargs):
        assert before_verification(1, code)
        fixer.llm.call()
        return outcome(code)
    monkeypatch.setattr(repair_loop, 'repair', loop)
    first = ht.repair(prepared, CODE, **LIMITS)
    first_path = Path(first['evidence'])
    original = first_path.read_bytes()
    second = ht.repair(prepared, CODE, **LIMITS)
    assert first_path.read_bytes() == original
    assert first['evidence'] != second['evidence']
    for result in (first, second):
        parent = Path(result['evidence']).parent
        assert (parent/'model-journal/call-001.json').is_file()
        assert result['task_budget'] == LIMITS
        assert result['budget']['model_calls']['reserved'] == 1
        assert result['budget']['verification']['spent_all'] == 1
        assert result['budget_enforcement'] == 'cooperative_library_deadline'
        assert result['budget']['wall_clock']['scope'] == 'repair_library_cooperative_deadline'
        assert result['partial_artifacts']['repair_evidence'] == str(parent)


def test_model_limit_is_shared_across_fixer_calls_and_cannot_report_pass(prepared, monkeypatch):
    def loop(code, task, inputs, verifier, fixer, *, before_verification, **kwargs):
        assert before_verification(1, code)
        fixer.llm.call()
        fixer.llm.call()  # Production budget refuses before any real dispatch.
        return outcome(code)
    monkeypatch.setattr(repair_loop, 'repair', loop)
    result = ht.repair(prepared, CODE, max_llm_calls=1)
    assert not result['passed']
    assert result['stopped_reason'] == 'budget_llm_calls'
    assert result['budget']['model_calls']['reserved'] == 1
    assert result['budget']['model_calls']['remaining'] == 0


def test_verification_refusal_preserves_initial_reservation_without_using_confirm(prepared, monkeypatch):
    def loop(code, task, inputs, verifier, fixer, *, before_verification, **kwargs):
        assert before_verification(1, code)
        assert not before_verification(2, code)
        return outcome(code, passed=True)  # A bad loop must not override the budget stop.
    monkeypatch.setattr(repair_loop, 'repair', loop)
    result = ht.repair(prepared, CODE, oracle_total=2)
    assert not result['passed'] and result['stopped_reason'] == 'verification_budget_exhausted'
    assert result['budget']['verification']['spent_all'] == 1
    assert result['budget']['verification']['spent']['confirm'] == 0
    assert result['budget']['verification']['refusals']['refine'] == 1


def test_backend_exception_keeps_reserved_budget_and_failure_evidence(prepared, monkeypatch):
    monkeypatch.setattr(ht, '_make_backend', Mock(side_effect=RuntimeError('offline backend failed')))
    result = ht.repair(prepared, CODE)
    assert result['ok'] is False and result['passed'] is False
    assert result['error'] == 'RuntimeError'
    assert result['budget']['verification']['spent_all'] == 1
    assert Path(result['evidence']).is_file()
    assert result['partial_artifacts']['repair_evidence']


def test_real_repair_loop_consumes_the_shared_ledger_and_stops_at_the_model_limit(prepared, monkeypatch):
    from operator_agent import fixer as fixer_module
    from tests.test_repair_loop import FakeVerifier
    bad = [VerificationResult(Level.L1_COMPILE, False, 'synthetic compiler failure')]
    verifier = FakeVerifier([bad])

    class BudgetedFixer:
        def __init__(self, llm):
            self.llm = llm

        def run_with_action(self, code, *args, **kwargs):
            self.llm.call()
            return SimpleNamespace(code=code+'\n# fixture\ndef changed(): return 1\n',
                                   summary='synthetic change', envelope=None)

    monkeypatch.setattr(ht, 'Verifier', lambda backend: verifier)
    monkeypatch.setattr(fixer_module, 'FixerAgent', BudgetedFixer)
    result = ht.repair(prepared, CODE, max_rounds=4, max_llm_calls=1)
    assert result['stopped_reason'] == 'budget_llm_calls' and not result['passed']
    assert len(verifier.calls) == 2
    assert result['budget']['model_calls']['reserved'] == 1
    assert result['budget']['verification']['spent_all'] == 2
    assert result['budget']['verification']['spent']['confirm'] == 0


def test_direct_library_expiry_does_not_claim_hard_worker_termination(prepared, monkeypatch):
    from operator_agent import task_clock
    from operator_agent import llm_budget
    clock = [1000.]
    for module in (task_clock, task_budget, llm_budget):
        monkeypatch.setattr(module, 'monotonic', lambda: clock[0])

    def backend(*args, **kwargs):
        clock[0] += 2  # A direct-library call cooperates after this returns.
        return object(), {}

    monkeypatch.setattr(ht, '_make_backend', backend)
    verifier = Mock()
    monkeypatch.setattr(ht, 'Verifier', lambda backend: verifier)
    result = ht.repair(prepared, CODE, wall_clock_s=1)
    assert not result['passed'] and result['stopped_reason'] == 'budget_wall_clock'
    assert result['budget_enforcement'] == 'cooperative_library_deadline'
    assert result['budget']['wall_clock']['scope'] == 'repair_library_cooperative_deadline'
    verifier.verify.assert_not_called()


def test_cli_and_mcp_forward_repair_budgets_as_runtime_options(tmp_path, monkeypatch, capsys):
    call = Mock(return_value=dict(ok=True))
    monkeypatch.setattr(tool_calls, 'run_tool_call', call)
    assert cli.main(['repair', '--run-dir', str(tmp_path), '--code', CODE,
        '--wall-clock-s', '12.5', '--max-llm-calls', '3', '--oracle-total', '5']) == 0
    assert {key: call.call_args.args[1][key] for key in LIMITS} == LIMITS
    capsys.readouterr()
    repair = Mock(return_value=dict(ok=True))
    monkeypatch.setattr(mcp_server, 'repair', repair)
    mcp_server._call_tool('operator_repair', dict(run_dir=str(tmp_path), code=CODE, **LIMITS))
    assert {key: repair.call_args.kwargs[key] for key in LIMITS} == LIMITS


def test_repair_schema_uses_run_finite_defaults_without_exposing_them_to_verify():
    definitions = {tool['name']: tool for tool in mcp_server._tool_definitions()}
    expected = task_budget.TaskBudgetConfig().as_dict()
    repair = definitions['operator_repair']['inputSchema']['properties']
    verify = definitions['operator_verify']['inputSchema']['properties']
    assert {key: repair[key]['default'] for key in expected} == expected
    assert not set(expected).intersection(verify)
    assert ht.TOOL_VERSION == 9


@pytest.fixture
def supervised(tmp_path, monkeypatch):
    run = tmp_path/'run'
    ht.prepare(DECL, run)
    clock = [1000.]
    monkeypatch.setattr(tool_calls, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(task_cancellation, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(task_budget, 'monotonic', lambda: clock[0])
    process = SimpleNamespace(pid=999999, returncode=None)
    process.poll = lambda: process.returncode
    monkeypatch.setattr(tool_calls, '_start_call_worker', lambda directory: process)
    monkeypatch.setattr(tool_calls.exec_guard, 'process_identity', lambda pid: 'fixture-owner')
    monkeypatch.setattr(tool_calls.exec_guard, 'task_process_groups',
                        lambda pid: (tool_calls.exec_guard.OwnedGroups(), True))
    monkeypatch.setattr(tool_calls.exec_guard, 'stop_task_worker',
                        lambda *a, **kw: dict(worker_reaped=True, status='fixture_cleanup'))
    call = tool_calls.ToolCall('operator_repair', dict(run_dir=str(run), code=CODE,
                                                     wall_clock_s=1))
    return call, clock, process


def test_repair_supervisor_timeout_is_budget_stop_instead_of_cancellation(supervised):
    call, clock, _ = supervised
    clock[0] += 2
    result = json.loads(call.poll()['content'][0]['text'])
    assert result['error'] == 'task_budget_exhausted'
    assert result['passed'] is False and result['stopped_reason'] == 'budget_wall_clock'
    assert call.status['phase'] == 'done'
    assert result['budget']['wall_clock']['expired']
    assert result['budget']['model_calls']['reserved'] is None
    assert result['partial_artifacts']


def test_user_cancel_keeps_its_distinct_reason_and_budget(supervised):
    call, clock, _ = supervised
    call.cancel()
    clock[0] += .1
    result = json.loads(call.poll()['content'][0]['text'])
    assert call.status['phase'] == 'cancelled'
    assert result['stopped_reason'] == 'user_cancelled' and result['passed'] is False
    assert result['task_budget']['wall_clock_s'] == 1
    assert result['cleanup']['remote_execution'] == 'unknown'


def test_completed_before_deadline_is_not_relabelled_by_late_poll(supervised):
    call, clock, process = supervised
    response = mcp_server._text_result(dict(ok=True, passed=True))
    tool_calls.publish_json(call.directory/'worker-result.json', dict(
        invocation_id=call.invocation_id, finished_monotonic=clock[0]+.5, response=response))
    process.returncode = 0
    clock[0] += 2
    result = json.loads(call.poll()['content'][0]['text'])
    assert result['passed'] is True and call.status['phase'] == 'done'
    assert call.status['stopped_reason'] == 'tool_returned'
    assert result['budget']['wall_clock']['elapsed_s'] == .5
    assert not result['budget']['wall_clock']['expired']


def test_supervisor_freezes_start_deadline_and_all_three_limits(supervised):
    call, _, _ = supervised
    request = json.loads((call.directory/'request.json').read_text())
    context = request['repair_budget']
    assert context['deadline'] == context['started_monotonic']+1
    assert context['task_budget'] == task_budget.TaskBudgetConfig(wall_clock_s=1).as_dict()


def test_supervised_invalid_budget_creates_no_worker_or_directory(tmp_path, monkeypatch):
    start = Mock(side_effect=AssertionError('no worker for invalid budget'))
    monkeypatch.setattr(tool_calls, '_start_call_worker', start)
    with pytest.raises(ht.ToolError) as caught:
        tool_calls.ToolCall('operator_repair', dict(run_dir=str(tmp_path/'new'), code=CODE,
                                                   wall_clock_s=float('nan')))
    assert caught.value.code == 'invalid_task_budget'
    assert not (tmp_path/'new').exists() and not start.called
