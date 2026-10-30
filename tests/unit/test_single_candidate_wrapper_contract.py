"""Offline wrapper ABI/request/return guards; historical sources stay text only."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmarks import workload_codegen_h20 as driver
from operator_agent import llm_client as lc, single_candidate_coder as sc
from operator_agent.llm_budget import LLMCallRefused
from operator_agent.run_contract import code_hash, digest
from tests.unit.test_single_candidate_coder import CODE, generate, response, setup
from tests.unit.test_single_candidate_semantics import model_fixture


HISTORICAL = (Path(__file__).resolve().parents[1]/'fixtures/single_candidate_extra_wrapper_args.txt').read_text()
HISTORICAL_HASH = '25c204a4385ca5810759e9d90ee41469000be0ce179812249016ed283d3ac119'
METADATA = [{'A': {}, 'B': {}}]


@pytest.mark.parametrize('metadata,signature,names,sets', [
    ([{'X': {}}], 'def kernel(X):', ['X'], [['X']]),
    ([{'B': {}, 'A': {}}, {'A': {}, 'B': {}}], 'def kernel(A, B):', ['A','B'], [['A','B']]),
    ([{'X': {}, 'eps': {'scalar_type':'float','value_disclosed':False}}],
        'def kernel(X, eps):', ['X','eps'], [['X','eps']]),
    ([{'A': {}, 'B': {}}, {'A': {}, 'C': {}}],
        'def kernel(A, B=None, C=None):', ['A','B','C'], [['A','B'],['A','C']]),
])
def test_calling_convention_comes_from_actual_caller_keys(metadata, signature, names, sets):
    result = sc.wrapper_calling_convention(metadata)
    assert result['schema'] == 'keyword-wrapper-v1'
    assert result['entrypoint'] == 'kernel' and result['invocation'] == 'kernel(**inputs)'
    assert result['signature'] == signature
    assert result['parameter_names'] == names and result['keyword_sets'] == sets
    assert result['extra_parameters_allowed'] is False
    assert result['parameter_style'] == 'explicit_keyword_compatible'
    assert result['returns'] == dict(kind='tensor',via='return_value',allocated_by='wrapper',
                                    shape_and_dtype='frozen_operator_semantics')
    assert result['stride_units'] == dict(input_metadata='bytes',runtime_tensor='elements')


@pytest.mark.parametrize('metadata', [None, [], {}, [{}], ['X'], [{'X':{}}, {}],
    [{'not-an-identifier':{}}], [{'class':{}}], [{1:{}}], [{'X): injected()':{}}]])
def test_signature_description_rejects_unknown_or_unrepresentable_input_names(metadata):
    with pytest.raises(ValueError):
        sc.wrapper_calling_convention(metadata)


def test_description_never_reads_values_and_has_no_shared_mutable_state():
    class Undisclosed:
        def __repr__(self):
            raise AssertionError('read a value')
    metadata = [{'X': Undisclosed()}]
    first = sc.wrapper_calling_convention(metadata)
    first['parameter_names'].append('Y')
    first['returns']['via'] = 'caller_output_argument'
    second = sc.wrapper_calling_convention(metadata)
    assert second['parameter_names'] == ['X'] and second['returns']['via'] == 'return_value'
    assert 'Undisclosed' not in json.dumps(second)


def test_live_coder_and_plan_only_driver_freeze_same_wrapper_request(tmp_path, monkeypatch):
    coder, task, sketch, inputs, plan, send = model_fixture(tmp_path, monkeypatch)
    convention = coder.declaration()['context']['calling_convention']
    assert convention['signature'] == 'def kernel(X):'
    responsibilities = convention['wrapper_responsibilities']
    assert responsibilities['output_allocation'] == 'inside_wrapper'
    assert responsibilities['shapes_and_reduction_lengths'] == 'derive_from_runtime_inputs'
    assert responsibilities['tensor_strides'] == 'read_inputs_and_outputs_independently'
    assert responsibilities['semantic_constants'] == 'frozen_contract_or_registered_reference'
    assert convention == driver.generation_declaration(task)['context']['calling_convention']
    # A tensor name Y, stride/shape names and constants are not invented inputs.
    assert convention['keyword_sets'] == [['X']] and convention['parameter_names'] == ['X']
    assert coder.run(sketch, task, test_inputs=inputs)
    raw = send.call_args.args[0]['body_text']
    assert raw == json.dumps(plan['request']['body'],ensure_ascii=False,allow_nan=False)
    request = json.loads(raw)
    context = json.loads(request['messages'][1]['content'])
    assert context['calling_convention'] == convention
    assert context['contract']['reference_contract']['epsilon'] == 1e-5
    assert 'not additional wrapper inputs' in request['messages'][0]['content']
    assert 'return the output tensor' in request['messages'][0]['content']
    assert HISTORICAL not in raw and HISTORICAL_HASH not in raw
    assert send.call_count == 1


@pytest.mark.parametrize('field', ('signature','parameters','returns','responsibilities'))
def test_calling_convention_drift_stops_before_model_reservation(tmp_path, monkeypatch, field):
    coder, task, sketch, inputs, _, send = model_fixture(tmp_path, monkeypatch)
    value = coder._context['calling_convention']
    if field == 'signature': value['signature'] = 'def kernel(X, Y):'
    if field == 'parameters': value['parameter_names'].append('Y')
    if field == 'returns': value['returns']['via'] = 'caller_output_argument'
    if field == 'responsibilities': value['wrapper_responsibilities']['output_allocation'] = 'caller'
    with pytest.raises(ValueError,match='drift'):
        coder.run(sketch, task, test_inputs=inputs)
    assert send.call_count == coder.llm.call_budget.summary()['reserved_calls'] == 0


@pytest.mark.parametrize('field', ('signature','parameters','returns','responsibilities'))
def test_self_consistent_hash_cannot_forge_caller_derived_convention(tmp_path, field):
    coder, _, _, _ = setup(tmp_path)
    declaration = copy.deepcopy(coder.declaration())
    value = declaration['context']['calling_convention']
    if field == 'signature': value['signature'] = 'def kernel(A, B, Y):'
    if field == 'parameters': value['parameter_names'].append('Y')
    if field == 'returns': value['returns']['kind'] = 'none'
    if field == 'responsibilities': value['wrapper_responsibilities']['semantic_constants'] = 'model'
    declaration['context_hash'] = digest(declaration['context'])
    with pytest.raises(ValueError,match='calling convention inconsistent'):
        sc.audit_generation(declaration, {}, {}, CODE)


@pytest.mark.parametrize('signature', [
    'def kernel(A, B, Y=None):', 'def kernel(A, B, *, Y=None):',
    'def kernel(A, B, *hidden):', 'def kernel(A, B, **hidden):',
    'def kernel(A, /, B):', 'def kernel(**inputs):',
])
def test_single_candidate_wrapper_rejects_hidden_or_undeclared_parameters(signature):
    code = CODE.replace('def kernel(A, B):', signature)
    with pytest.raises(sc.PrecheckRefused) as caught:
        sc.precheck(code, METADATA)
    assert caught.value.reason == 'kernel_interface_refused'


@pytest.mark.parametrize('signature,metadata', [
    ('def kernel(A, B):', METADATA),
    ('def kernel(*, A, B):', METADATA),
    ('def kernel(A, B=None, C=None):', [{'A':{},'B':{}},{'A':{},'C':{}}]),
])
def test_explicit_keyword_compatible_signatures_remain_allowed(signature, metadata):
    assert sc.precheck(CODE.replace('def kernel(A, B):',signature), metadata) is None


@pytest.mark.parametrize('body', [
    'pass', 'return', 'return None',
    'def nested():\n        return A',
    'class Nested:\n        def output(self):\n            return A',
    'if A is None:\n        return None\n    return',
])
def test_wrapper_requires_its_own_non_none_return_expression(body):
    code = CODE.replace('return torch.empty_like(A)', body)
    with pytest.raises(sc.PrecheckRefused) as caught:
        sc.precheck(code, METADATA)
    assert caught.value.reason == 'kernel_return_missing'


@pytest.mark.parametrize('body', [
    'if A is None:\n        return B\n    return A',
    'try:\n        return A\n    finally:\n        pass',
    'def nested():\n        return A\n    return nested()',
])
def test_value_return_presence_is_only_a_syntactic_guard(body):
    # No claim about runtime tensor type, all control paths or correct math.
    assert sc.precheck(CODE.replace('return torch.empty_like(A)',body), METADATA) is None


def test_original_historical_source_keeps_same_interface_failure_readonly():
    assert code_hash(HISTORICAL) == HISTORICAL_HASH
    with pytest.raises(sc.PrecheckRefused) as caught:
        sc.precheck(HISTORICAL,[{'X':dict(shape=[8192,15360])}])
    assert caught.value.reason == 'kernel_interface_refused'
    assert code_hash(HISTORICAL) == HISTORICAL_HASH


def test_missing_return_is_one_failed_model_source_with_fixed_receipt(tmp_path, monkeypatch):
    coder, task, sketch, inputs = setup(tmp_path)
    code = CODE.replace('return torch.empty_like(A)', 'return None')
    send = Mock(return_value=response(code))
    monkeypatch.setattr(lc,'bounded_dispatch',send)
    with pytest.raises(LLMCallRefused): generate(coder,task,sketch,inputs)
    with pytest.raises(LLMCallRefused): generate(coder,task,sketch,inputs)
    lineage = coder.last_generation
    assert lineage['code'] == code and lineage['candidate_hash'] == code_hash(code)
    assert sc.generation_failure(lineage) == dict(stage='precheck',reason='kernel_return_missing')
    assert coder.llm.call_budget.summary()['aborted'] == 'candidate_generation_failed'
    assert send.call_count == 1
    assert (coder.llm.call_budget.root/'candidate.json').stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError,match='not a successful single-call campaign'):
        sc.audit_generation(coder.declaration(),coder.llm.call_budget.summary(),lineage,code)


def test_historical_bad_interface_never_gets_repaired_or_dispatched(tmp_path, monkeypatch):
    coder, task, sketch, inputs, plan, send = model_fixture(tmp_path, monkeypatch)
    send.return_value = response(HISTORICAL)
    with pytest.raises(LLMCallRefused): coder.run(sketch,task,test_inputs=inputs)
    assert coder.last_generation['code'] == HISTORICAL
    assert coder.last_generation['candidate_hash'] == HISTORICAL_HASH
    assert sc.generation_failure(coder.last_generation) == dict(stage='precheck',reason='kernel_interface_refused')
    assert plan['generation']['context']['calling_convention']['signature'] == 'def kernel(X):'
    assert send.call_count == 1
