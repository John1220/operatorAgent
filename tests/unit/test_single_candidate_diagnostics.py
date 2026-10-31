"""Offline response/static fixtures only: never execute the historical candidate."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.drivers import workload_codegen_h20 as driver
from operator_agent.execute import llm_client as lc
from operator_agent.generate import single_candidate_coder as sc
from operator_agent.execute.llm_budget import LLMCallRefused
from operator_agent.core.run_contract import code_hash
from tests.unit.test_single_candidate_coder import CODE, generate, response, setup, conductor
from tests.unit.test_workload_codegen_driver import generation_fixture, retain_model
from tests.unit.test_workload_evidence import backend


DUPLICATE = (Path(__file__).resolve().parents[1]/'fixtures/single_candidate_duplicate_kernel.txt').read_text()
METADATA = [{'A': {'shape': [16,16]}, 'B': {'shape': [16,16]}}]
BAD_CODE = [
    ('source_syntax_invalid', CODE+'\nSYNTAX_SECRET : :\n'),
    ('source_syntax_invalid', CODE.replace('return torch.empty_like(A)',
        'nonlocal COMPILE_SECRET\n    return torch.empty_like(A)')),
    ('kernel_entry_count', CODE.replace('fixture_jit', 'kernel')),
    ('kernel_entry_decorated', CODE.replace('def kernel(A, B):', '@triton.jit\ndef kernel(A, B):')),
    ('kernel_entry_rebound', CODE+'\nkernel = 1\n'),
    ('jit_helper_missing', CODE.replace('@triton.jit', '')),
    ('kernel_entry_launch', CODE.replace('return torch.empty_like(A)', 'return kernel[(1,)](A, B)')),
    ('undeclared_decorator', CODE.replace('@triton.jit', '@triton.autotune(configs=[])\n@triton.jit')),
    ('executable_default', CODE.replace('def kernel(A, B):', 'def kernel(A, B=print("DEFAULT_SECRET")):')),
    ('executable_assignment', CODE+'\nsecret = torch.empty((1,))\n'),
    ('executable_statement', CODE+'\nraise RuntimeError("STATEMENT_SECRET")\n'),
    ('undeclared_import', CODE+'\nimport os\n'),
    ('undeclared_import_from', CODE+'\nfrom subprocess import run\n'),
    ('dunder_access', CODE.replace('return torch.empty_like(A)', 'return torch.__dict__')),
    ('torch_not_allocation', CODE.replace('return torch.empty_like(A)', 'return torch.matmul(A, B)')),
    ('undeclared_triton_host_operation', CODE.replace('return torch.empty_like(A)', 'return triton.compile("HOST_SECRET")')),
    ('dynamic_host_operation', CODE.replace('return torch.empty_like(A)', 'return eval("CALL_SECRET")')),
    ('kernel_interface_refused', CODE.replace('def kernel(A, B):', 'def kernel(X):')),
]


def test_prompt_separates_exact_wrapper_and_named_jit_helper_without_a_template():
    assert 'exactly one undecorated top-level Python def kernel' in sc.PROMPT
    assert 'exact declared keyword input names (case-sensitive)' in sc.PROMPT
    assert 'compute helpers with different names' in sc.PROMPT
    assert 'helper[grid](...)' in sc.PROMPT and 'never kernel[grid](...)' in sc.PROMPT
    assert 'never redefine or rebind kernel' in sc.PROMPT
    assert 'No retries or alternative candidates' in sc.PROMPT
    # Naming/dispatch instructions are not an implementation or a code template.
    assert 'tl.load' not in sc.PROMPT and 'tl.store' not in sc.PROMPT


def test_exact_historical_duplicate_source_is_rejected_readonly():
    assert code_hash(DUPLICATE) == '6857d1d975367ff010bda617e72acb7998f0ebd1746fc0b5f68c4ae30643bb73'
    before = DUPLICATE
    with pytest.raises(sc.PrecheckRefused, match='one explicit kernel entry required') as caught:
        sc.precheck(DUPLICATE, [{'X': {'shape': [8192,15360]}}])
    assert caught.value.reason == 'kernel_entry_count' and DUPLICATE == before


@pytest.mark.parametrize('reason,code', BAD_CODE)
def test_static_reason_is_fixed_bounded_and_does_not_execute_source(reason, code, capsys):
    with pytest.raises(sc.PrecheckRefused) as caught:
        sc.precheck(code, METADATA)
    assert caught.value.reason == reason
    assert str(caught.value) == sc.PRECHECK_REASONS[reason]
    assert 'SECRET' not in str(caught.value)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('applicable,violations', [(False, []), (True, ['AUDIT_SECRET'])])
def test_anticheat_details_never_enter_the_failure_diagnostic(monkeypatch, applicable, violations):
    monkeypatch.setattr(sc, 'static_audit', lambda *a: SimpleNamespace(applicable=applicable, violations=violations))
    with pytest.raises(sc.PrecheckRefused) as caught:
        sc.precheck(CODE, METADATA)
    assert caught.value.reason == 'static_anticheat_refused'
    assert 'AUDIT_SECRET' not in str(caught.value)


def test_interface_problem_text_never_enters_the_failure_diagnostic(monkeypatch):
    monkeypatch.setattr(sc, 'check_kernel_interface', lambda *a: SimpleNamespace(applicable=True, problem='INTERFACE_SECRET'))
    with pytest.raises(sc.PrecheckRefused) as caught:
        sc.precheck(CODE, METADATA)
    assert caught.value.reason == 'kernel_interface_refused'
    assert 'INTERFACE_SECRET' not in str(caught.value)


def test_correctly_named_helper_launch_passes_static_check_only():
    code = CODE.replace('return torch.empty_like(A)',
        'Y = torch.empty_like(A)\n    fixture_jit[(1,)](A, Y, 16)\n    return Y')
    assert sc.precheck(code, METADATA) is None
    # This fixture is not a correct matmul: no GPU or correctness claim is made.


@pytest.mark.parametrize('reason,code', BAD_CODE+[('kernel_entry_count', DUPLICATE)])
def test_failed_original_source_has_durable_diagnostic_one_call_and_no_repair(tmp_path, monkeypatch, reason, code):
    coder, task, sketch, inputs = setup(tmp_path)
    send = Mock(return_value=response(code))
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    with pytest.raises(LLMCallRefused, match='no repair/retry/template'):
        generate(coder, task, sketch, inputs)
    with pytest.raises(LLMCallRefused, match='already attempted'):
        generate(coder, task, sketch, inputs)
    assert send.call_count == 1
    lineage = json.loads((coder.llm.call_budget.root/'candidate.json').read_text())
    assert lineage == coder.last_generation
    assert lineage['code'] == code and lineage['candidate_hash'] == code_hash(code)
    receipt = sc.generation_failure(lineage)
    assert receipt == {'stage':'precheck', 'reason':reason}
    assert len(json.dumps(receipt).encode()) <= 256 and 'SECRET' not in json.dumps(receipt)
    assert (coder.llm.call_budget.root/'candidate.json').stat().st_mode & 0o222 == 0
    budget = coder.llm.call_budget.summary()
    assert budget['reserved_calls'] == 1 and budget['output_tokens_reserved'] == 4096
    assert budget['aborted'] == 'candidate_generation_failed'
    with pytest.raises(ValueError, match='not a successful single-call campaign'):
        sc.audit_generation(coder.declaration(), budget, lineage, code)


@pytest.mark.parametrize('phase', ['model_call', 'envelope', 'precheck', 'context'])
def test_unknown_exception_text_and_class_are_not_persisted(tmp_path, monkeypatch, phase):
    coder, task, sketch, inputs = setup(tmp_path)
    send = Mock(return_value=response())
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    class EXCEPTION_CLASS_SECRET(RuntimeError):
        pass
    def fail(*a, **k):
        raise EXCEPTION_CLASS_SECRET('EXCEPTION_MESSAGE_SECRET file/network/key/raw output')
    if phase == 'model_call':
        monkeypatch.setattr(lc, 'bounded_dispatch', fail)
    elif phase == 'envelope':
        send.return_value = response(envelope={'code':CODE,'rationale':'x','extra':'EXTRA_SECRET'})
    elif phase == 'precheck':
        monkeypatch.setattr(sc, 'precheck', fail)
    else:
        def drift_after_precheck(*a):
            coder._context['task_id'] = 'CONTEXT_SECRET'
        monkeypatch.setattr(sc, 'precheck', drift_after_precheck)
    with pytest.raises(LLMCallRefused):
        generate(coder, task, sketch, inputs)
    receipt = sc.generation_failure(coder.last_generation)
    assert receipt['stage'] == phase
    assert receipt['reason'] == dict(model_call='model_call_failed',envelope='invalid_code_envelope',
        precheck='static_precheck_failed',context='generation_context_drift')[phase]
    assert 'SECRET' not in json.dumps(receipt)
    assert coder.llm.call_budget.summary()['aborted'] is not None


@pytest.mark.parametrize('receipt', [None, {}, {'stage':'precheck'},
    {'stage':'precheck','reason':'SECRET'}, {'stage':'SECRET','reason':'kernel_entry_count'},
    {'stage':'precheck','reason':'kernel_entry_count','message':'SECRET'},
    {'stage':'precheck','reason':'kernel_entry_count','code':CODE},
    {'stage':'model_call','reason':'kernel_entry_count'},
    {'stage':['precheck'],'reason':'kernel_entry_count'}, {'stage':'precheck','reason':True},
    {'stage':'precheck','reason':'x'*4096}])
def test_receipt_reader_rejects_unknown_unbounded_and_arbitrary_fields(receipt):
    with pytest.raises(ValueError, match='invalid generation failure receipt'):
        sc.generation_failure({'status':'failed','failure':receipt})


def test_failure_receipt_is_not_a_positive_generation_credential(tmp_path, monkeypatch):
    coder, task, sketch, inputs = setup(tmp_path)
    monkeypatch.setattr(lc,'bounded_dispatch',lambda *a: response())
    generate(coder, task, sketch, inputs)
    lineage = copy.deepcopy(coder.last_generation)
    lineage['failure'] = {'stage':'precheck','reason':'kernel_entry_count'}
    with pytest.raises(ValueError, match='source lineage inconsistent'):
        sc.audit_generation(coder.declaration(), coder.llm.call_budget.summary(), lineage, CODE)
    with pytest.raises(ValueError, match='invalid generation failure receipt'):
        sc.generation_failure(lineage)


@pytest.mark.parametrize('bad', ['ARBITRARY_SECRET', None, True, []])
def test_exception_cannot_carry_an_arbitrary_reason(bad):
    with pytest.raises(ValueError, match='unknown precheck reason'):
        sc.PrecheckRefused(bad)


@pytest.mark.parametrize('bad', ['EXCEPTION_REASON_SECRET', [], True])
def test_corrupted_exception_reason_falls_back_to_fixed_unknown_code(tmp_path, monkeypatch, bad):
    coder, task, sketch, inputs = setup(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: response())
    def fail(*a):
        exc = sc.PrecheckRefused('kernel_entry_count')
        exc.reason = bad
        raise exc
    monkeypatch.setattr(sc, 'precheck', fail)
    with pytest.raises(LLMCallRefused): generate(coder, task, sketch, inputs)
    assert sc.generation_failure(coder.last_generation) == {'stage':'precheck','reason':'static_precheck_failed'}
    assert coder.llm.call_budget.summary()['aborted'] == 'candidate_generation_failed'


@pytest.mark.parametrize('lineage', [None, [], 'SECRET', True])
def test_nonobject_lineage_is_refused_with_fixed_text(lineage):
    with pytest.raises(ValueError, match='invalid generation failure receipt'):
        sc.generation_failure(lineage)


def test_diagnostic_vocabularies_cannot_be_mutated():
    with pytest.raises(TypeError): sc.PRECHECK_REASONS['forged'] = 'SECRET'
    with pytest.raises(TypeError): sc.FAILURE_REASONS['forged'] = {'SECRET'}


def test_duplicate_source_never_reaches_conductor_verifier_or_followup(tmp_path, monkeypatch):
    coder, task, sketch, inputs = setup(tmp_path)
    b, cases = backend()
    c = conductor(tmp_path, coder, b)
    send = Mock(return_value=response(DUPLICATE))
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    c.verifier.verify = lambda *a, **k: pytest.fail('duplicate source dispatched')
    c.fixer.run_with_action = lambda *a, **k: pytest.fail('failure started repair')
    c.search.run = lambda *a, **k: pytest.fail('failure started another candidate')
    with pytest.raises(LLMCallRefused): c.execute(task, inputs)
    assert send.call_count == 1 and cases == []
    assert sc.generation_failure(coder.last_generation)['reason'] == 'kernel_entry_count'
    assert not list(c.store.output_dir.rglob('result.json'))


def test_driver_retains_sanitized_diagnostic_but_still_fails_readonly(tmp_path, monkeypatch):
    task, inputs, coder, client, plan, send = generation_fixture(tmp_path, monkeypatch)
    send.return_value = response(DUPLICATE)
    with pytest.raises(LLMCallRefused): coder.run(coder._sketch, task, test_inputs=inputs)
    retain_model(tmp_path, client)
    before = {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    proof = driver.audit_model(tmp_path, plan)
    assert proof['failure'] == {'stage':'precheck','reason':'kernel_entry_count'}
    assert not proof['completed'] and proof['gpu_measurements'] == 0
    assert not proof['restored_eligibility'] and send.call_count == 1
    assert before == {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_driver_rejects_tampered_diagnostic_without_echoing_it(tmp_path, monkeypatch):
    task, inputs, coder, client, plan, send = generation_fixture(tmp_path, monkeypatch)
    send.return_value = response(DUPLICATE)
    with pytest.raises(LLMCallRefused): coder.run(coder._sketch, task, test_inputs=inputs)
    retain_model(tmp_path, client)
    p = client.call_budget.root/'candidate.json'
    value = json.loads(p.read_text()); value['failure']['reason'] = 'TAMPERED_SECRET'
    p.chmod(0o600); p.write_text(json.dumps(value))  # Temp fixture only.
    proof = driver.audit_model(tmp_path, plan)
    assert not proof['completed'] and 'failure' not in proof
    assert 'TAMPERED_SECRET' not in json.dumps(proof)
    assert send.call_count == 1
