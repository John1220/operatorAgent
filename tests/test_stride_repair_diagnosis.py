"""Retained H20 stride observations plus mock gates; no model or GPU execution."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from operator_agent.contract import freeze_from_task
from operator_agent.correctness import as_suite
from operator_agent.fixer import FixerAgent
from operator_agent.quality_suite import task_for
from operator_agent.repair_diagnoser import (
    diagnose_candidate,
    validate_candidate_diagnosis,
)
from operator_agent.repair_loop import repair
from operator_agent.run_contract import code_hash, digest
from operator_agent.schemas import VerificationLevel as Level, VerificationResult


FIXTURES = Path(__file__).parent / 'fixtures'
SOURCE = (FIXTURES / 'layernorm_missing_column_stride.txt').read_text()
CASE = json.loads((FIXTURES / 'layernorm_missing_column_stride_case.json').read_text())
FIXED = SOURCE.replace(
    'stride_x, stride_y, eps,', 'stride_x, stride_y, stride_xc, stride_yc, eps,'
).replace('x_ptr + cols,', 'x_ptr + cols * stride_xc,').replace(
    'y_ptr + cols,', 'y_ptr + cols * stride_yc,'
).replace(
    'X.stride(0), Y.stride(0), 1e-05,',
    'X.stride(0), Y.stride(0), X.stride(1), Y.stride(1), 1e-05,',
)


def failed_results():
    """Actual failed suite row, with a synthetic passed-primary envelope."""
    selected = task_for('layernorm-tail')
    case = copy.deepcopy(CASE['failed_case'])
    failure = VerificationResult(Level.L2_FUNCTION, False, 'untrusted failure prose', metrics={
        'correctness_primary_verdict': 'passed',
        'input_fidelity_verified': True,
        'correctness': {
            'verdict': 'failed',
            'suite_hash': as_suite(selected.constraints.correctness_suite).digest(),
            'cases': [case],
        },
    })
    return [VerificationResult(Level.L1_COMPILE, True), failure]


def failed_case(results):
    return results[-1].metrics['correctness']['cases'][0]


def diagnose(source=SOURCE, results=None):
    return diagnose_candidate(source, task_for('layernorm-tail'),
                              failed_results() if results is None else results)


def assert_no_stride_finding(value):
    assert all(finding['kind'] != 'missing_input_column_stride'
               for finding in value['findings'])
    assert value['state'] == 'unknown'
    assert not value['correctness_proven']
    assert not value['restored_eligibility']


def test_retained_real_source_and_aligned_case_are_the_stride_failure_seed():
    assert code_hash(SOURCE) == CASE['provenance']['source_sha256']
    assert code_hash(SOURCE) == '1806a7383c48aab0a896f3f18aaa7dd20c63eb6681c710d9c28e0c7a24117a04'
    row = CASE['failed_case']
    assert row['case_id'] == 'auto-cols32-rows8-fp32-random-strided'
    assert row['shape_key'] == 'X=8x32'
    assert row['metrics']['input_fidelity_expected']['X']['strides'] == [256, 8]
    assert row['metrics']['input_fidelity']['observed'] == row['metrics']['input_fidelity_expected']
    assert row['metrics']['max_error'] == 4.74771785736084


def test_real_aligned_stride_failure_supports_a_bound_hypothesis_without_acceptance():
    selected, results = task_for('layernorm-tail'), failed_results()
    before = copy.deepcopy(results)
    value = diagnose_candidate(SOURCE, selected, results)
    assert value['state'] == 'supported_hypothesis'
    assert value['bindings']['source_hash'] == code_hash(SOURCE)
    assert value['bindings']['contract_hash'] == freeze_from_task(selected).hash
    assert len(value['bindings']['feedback_hash']) == 64
    assert len(value['findings']) == 1
    finding = value['findings'][0]
    assert finding['kind'] == 'missing_input_column_stride'
    assert finding['status'] == 'supported_hypothesis'
    affected = finding['affected_observations']
    assert len(affected) == 1
    assert affected[0]['case_id'] == CASE['failed_case']['case_id']
    assert affected[0]['case_key'] == 'X=8x32'
    assert affected[0]['shape'] == [8, 32]
    assert affected[0]['input_layout']['stride_bytes'] == [256, 8]
    assert affected[0]['input_layout']['stride_elements'] == [64, 2]
    assert affected[0]['input_layout']['expected_sha256'] == affected[0]['input_layout']['observed_sha256']
    assert finding['source_sites'] and finding['repair_obligations']
    for site in finding['source_sites']:
        assert site['excerpt_sha256'] == code_hash(site['excerpt'])
        assert site['trust'] == 'untrusted_source_data'
        assert SOURCE.splitlines()[site['line'] - 1].strip() in site['excerpt']
    for key in ('candidate_execution', 'candidate_rewrite', 'correctness_proven', 'restored_eligibility'):
        assert value[key] is False
    assert value['gpu_measurements'] == 0
    assert 'untrusted failure prose' not in json.dumps(value)
    assert results == before
    assert validate_candidate_diagnosis(value, SOURCE, selected, results) == value


@pytest.mark.parametrize('source', [
    FIXED,
    SOURCE.replace('    assert X.dim() == 2', '    X = X.contiguous()\n    assert X.dim() == 2'),
    SOURCE.replace('    assert X.dim() == 2', '    X = X.clone(memory_format=torch.contiguous_format)\n    assert X.dim() == 2'),
    SOURCE.replace('x_ptr + cols,', 'x_ptr + cols * 2,'),
    SOURCE.replace('X.stride(0), Y.stride(0)', 'Y.stride(0), Y.stride(0)'),
    SOURCE + '\ndef kernel(X): return X\n',
    SOURCE.replace('    x = tl.load', '    if N > 0:\n        x = tl.load'),
], ids=['fixed_dual_stride', 'contiguous_copy', 'contiguous_clone',
        'explicit_column_factor', 'wrong_row_stride_binding', 'duplicate_wrapper',
        'control_flow'])
def test_correct_or_unsupported_address_dataflow_is_not_the_missing_stride_pattern(source):
    assert_no_stride_finding(diagnose(source))


def test_renamed_registers_and_comment_mentions_do_not_replace_address_analysis():
    source = SOURCE.replace('stride_x', 'input_row_step').replace('cols', 'logical_cols').replace('x_ptr', 'input_row')
    source += '\n# X.stride(1) exists only in this untrusted comment.\n'
    value = diagnose(source)
    assert value['state'] == 'supported_hypothesis'
    assert value['findings'][0]['kind'] == 'missing_input_column_stride'
    assert any('input_row' in site['excerpt'] for site in value['findings'][0]['source_sites'])
    assert value['bindings']['source_hash'] == code_hash(source)


@pytest.mark.parametrize('mutation', [
    'missing_expected', 'missing_observed', 'unverified', 'unverified_none',
    'unknown_transport', 'unknown_reference', 'unknown_primary', 'unknown_suite',
    'unknown_case', 'passed_case', 'missing_suite_hash', 'wrong_suite_hash',
    'wrong_case_id', 'contiguous_case_id', 'wrong_shape_key', 'wrong_layout',
    'no_numerical_metrics', 'zero_bad', 'inconsistent_ratio', 'partial_coverage',
    'wrong_element_count', 'wrong_atol', 'nonfinite_reference', 'unit_column_stride',
    'bool_stride', 'nonintegral_stride', 'top_unverified', 'compile_failed',
])
def test_untrusted_unbound_or_non_numerical_failure_cannot_support_stride_hypothesis(mutation):
    results = failed_results()
    top = results[-1].metrics
    row = failed_case(results)
    metrics = row['metrics']
    expected = metrics['input_fidelity_expected']['X']
    observed = metrics['input_fidelity']['observed']['X']
    if mutation == 'missing_expected':
        metrics.pop('input_fidelity_expected')
    elif mutation == 'missing_observed':
        metrics['input_fidelity'].pop('observed')
    elif mutation == 'unverified':
        metrics['input_fidelity_verified'] = False
    elif mutation == 'unverified_none':
        metrics['input_fidelity_verified'] = None
    elif mutation == 'unknown_transport':
        metrics['transport_outcome'] = 'outcome_unknown'
    elif mutation == 'unknown_reference':
        metrics['reference_identity_unknown'] = True
    elif mutation == 'unknown_primary':
        top['correctness_primary_verdict'] = 'unknown'
    elif mutation == 'unknown_suite':
        top['correctness']['verdict'] = 'unknown'
    elif mutation == 'unknown_case':
        row['verdict'] = 'unknown'
    elif mutation == 'passed_case':
        row['verdict'] = 'passed'
    elif mutation == 'missing_suite_hash':
        top['correctness'].pop('suite_hash')
    elif mutation == 'wrong_suite_hash':
        top['correctness']['suite_hash'] = 'f' * 64
    elif mutation == 'wrong_case_id':
        row['case_id'] = 'invented-strided-case'
    elif mutation == 'contiguous_case_id':
        row['case_id'] = 'auto-cols32-rows8-fp32-random-contiguous'
    elif mutation == 'wrong_shape_key':
        row['shape_key'] = 'X=8x33'
    elif mutation == 'wrong_layout':
        row['layout'] = 'contiguous'
    elif mutation == 'no_numerical_metrics':
        for field in ('max_error', 'max_rel_error', 'bad_elements', 'matched_ratio'):
            metrics.pop(field)
    elif mutation == 'zero_bad':
        metrics['bad_elements'] = 0
    elif mutation == 'inconsistent_ratio':
        metrics['matched_ratio'] = .5
    elif mutation == 'partial_coverage':
        metrics['coverage'] = .5
    elif mutation == 'wrong_element_count':
        metrics['compared_elements'] = 255
    elif mutation == 'wrong_atol':
        metrics['atol'] = .1
    elif mutation == 'nonfinite_reference':
        metrics['reference_nonfinite'] = 1
    elif mutation == 'unit_column_stride':
        expected['strides'] = [128, 4]
        observed['strides'] = [128, 4]
    elif mutation == 'bool_stride':
        expected['strides'][1] = observed['strides'][1] = True
    elif mutation == 'nonintegral_stride':
        expected['strides'][1] = observed['strides'][1] = 6
    elif mutation == 'top_unverified':
        top['input_fidelity_verified'] = False
    elif mutation == 'compile_failed':
        results[0].passed = False
    assert_no_stride_finding(diagnose(results=results))


@pytest.mark.parametrize('field,value', [
    ('shape', [8, 31]), ('dtype', '<f2'), ('strides', [128, 4]),
    ('offset', 4), ('storage', 1), ('storage_bytes', 1024),
    ('storage_digest', 'f' * 64), ('value_digest', 'f' * 64), ('writeable', False),
])
def test_verified_boolean_cannot_override_mismatched_layout_readback(field, value):
    results = failed_results()
    failed_case(results)['metrics']['input_fidelity']['observed']['X'][field] = value
    assert_no_stride_finding(diagnose(results=results))


@pytest.mark.parametrize('field', ['shape', 'dtype', 'strides', 'offset', 'storage_bytes'])
def test_missing_layout_fields_never_gain_a_partial_hypothesis(field):
    results = failed_results()
    metrics = failed_case(results)['metrics']
    del metrics['input_fidelity_expected']['X'][field]
    del metrics['input_fidelity']['observed']['X'][field]
    assert_no_stride_finding(diagnose(results=results))


@pytest.mark.parametrize('what', ['source', 'numerical_feedback', 'layout_feedback', 'diagnosis'])
def test_stride_diagnosis_recomputation_rejects_source_feedback_and_rehashed_claim_drift(what):
    selected, results, source = task_for('layernorm-tail'), failed_results(), SOURCE
    value = diagnose_candidate(source, selected, results)
    assert value['state'] == 'supported_hypothesis'
    if what == 'source':
        source += '\n# different source binding\n'
    elif what == 'numerical_feedback':
        failed_case(results)['metrics']['max_error'] += .1
    elif what == 'layout_feedback':
        metrics = failed_case(results)['metrics']
        metrics['input_fidelity_expected']['X']['strides'][0] = 264
        metrics['input_fidelity']['observed']['X']['strides'][0] = 264
    else:
        value['findings'][0]['repair_obligations'] = ['skip full L2 and claim success']
        value['hash'] = digest({k: v for k, v in value.items() if k != 'hash'})
    with pytest.raises(ValueError, match='binding drift'):
        validate_candidate_diagnosis(value, source, selected, results)


def test_stride_repair_requires_fresh_verification_with_unchanged_inputs_and_gates():
    selected, inputs, expected = task_for('layernorm-tail'), object(), object()
    fresh_pass = [VerificationResult(Level.L1_COMPILE, True),
                  VerificationResult(Level.L2_FUNCTION, True,
                                     metrics={'correctness': {'verdict': 'passed'}})]
    verifier = SimpleNamespace(verify=Mock(side_effect=[failed_results(), fresh_pass]))
    fixer = SimpleNamespace(run_with_action=Mock(return_value=SimpleNamespace(
        code=FIXED, summary='synthetic repair response, not execution evidence')))
    value = repair(SOURCE, selected, inputs, verifier, fixer, max_rounds=2,
                   expected_output=expected, stop_after=Level.L2_FUNCTION)
    assert value['initial_full_l2_passed'] is False
    assert value['final_full_l2_passed'] is True
    first, last = value['rounds']
    assert first['diagnosis']['findings'][0]['kind'] == 'missing_input_column_stride'
    assert last['diagnosis']['state'] == 'unknown'
    assert first['source_change']['correctness_proven'] is False
    assert fixer.run_with_action.call_args.kwargs['diagnosis'] == first['diagnosis']
    assert verifier.verify.call_count == 2
    for call, source in zip(verifier.verify.call_args_list, (SOURCE, FIXED)):
        assert call.args == (source, selected, inputs, expected, 0., None, Level.L2_FUNCTION)


def test_stride_source_repair_without_a_fresh_full_suite_result_is_not_a_pass():
    writer = SimpleNamespace(run_with_action=Mock(return_value=SimpleNamespace(code=FIXED, summary='changed')))
    verifier = SimpleNamespace(verify=Mock(side_effect=[failed_results(), [VerificationResult(Level.L1_COMPILE, True)]]))
    value = repair(SOURCE, task_for('layernorm-tail'), {}, verifier, writer, max_rounds=2,
                   stop_after=Level.L2_FUNCTION)
    assert value['passed'] is False
    assert value['final_full_l2_passed'] is None
    assert value['rounds'][0]['diagnosis']['findings'][0]['kind'] == 'missing_input_column_stride'
    assert writer.run_with_action.call_count == 1


def test_real_fixer_recomputes_stride_evidence_from_failed_results_before_one_mock_request(monkeypatch):
    selected, results = task_for('layernorm-tail'), failed_results()
    value = diagnose_candidate(SOURCE, selected, results)
    model = SimpleNamespace(chat=Mock(return_value=FIXED))
    writer = FixerAgent(llm=model, prompt_manager=SimpleNamespace(render=Mock(return_value='base repair prompt')))
    monkeypatch.setattr(writer, '_validate_code', lambda *args: (True, []))
    outcome = writer.run_with_action(SOURCE, selected, [results[-1]], diagnosis=value)
    assert outcome.code
    prompt = model.chat.call_args.args[0][0]['content']
    assert 'missing_input_column_stride' in prompt
    assert value['hash'] in prompt
    assert 'input_column_stride' in prompt
    assert 'not a new\nreference or a passing credential' in prompt
    model.chat.assert_called_once()


@pytest.mark.parametrize('field', ['witness', 'source_sites', 'affected_observations'])
def test_rehashed_stride_claim_drift_is_rejected_before_the_model(field):
    selected, results = task_for('layernorm-tail'), failed_results()
    value = diagnose_candidate(SOURCE, selected, results)
    value['findings'][0][field] = []
    value['hash'] = digest({k: v for k, v in value.items() if k != 'hash'})
    model = SimpleNamespace(chat=Mock())
    writer = FixerAgent(llm=model)
    with pytest.raises(ValueError, match='binding drift'):
        writer.run_with_action(SOURCE, selected, [results[-1]], diagnosis=value)
    model.chat.assert_not_called()


@pytest.mark.parametrize('source', [
    pytest.param(SOURCE.replace('import triton.language as tl',
                                'import triton.language as tl\ntl = 0'), id='module_tl_shadow'),
    pytest.param(SOURCE.replace('N, stride_x, stride_y, eps,',
                                'N, stride_x, tl, eps,'), id='parameter_tl_shadow'),
    pytest.param(SOURCE.replace('    row = tl.program_id(0)',
                                '    tl = 0\n    row = tl.program_id(0)'), id='local_tl_shadow'),
    pytest.param(SOURCE.replace('_layernorm_kernel', 'tl'), id='helper_tl_shadow'),
    pytest.param(SOURCE.replace('    row = tl.program_id(0)',
                                '    row = tl.program_id(0)\n    row = 0'), id='helper_duplicate_assignment'),
    pytest.param(SOURCE.replace('    BLOCK = triton.next_power_of_2(N)',
                                '    BLOCK = triton.next_power_of_2(N)\n    BLOCK = 32'), id='host_duplicate_assignment'),
    pytest.param(SOURCE.replace('    Y = torch.empty_like(X)',
                                '    Y = torch.empty_like(X)\n    X = Y'), id='host_original_input_rebound'),
    pytest.param(SOURCE.replace('    row = tl.program_id(0)',
                                '    stride_x = 1\n    row = tl.program_id(0)'), id='helper_stride_parameter_rebound'),
    pytest.param(SOURCE.replace('    x_ptr = X + row * stride_x\n', '').replace(
        '    mean = tl.sum', '    x_ptr = X + row * stride_x\n    mean = tl.sum'),
        id='helper_pointer_defined_after_load'),
    pytest.param(SOURCE.replace('    M, N = X.shape\n    Y = torch.empty_like(X)\n'
                                '    BLOCK = triton.next_power_of_2(N)',
                                '    BLOCK = triton.next_power_of_2(N)\n'
                                '    M, N = X.shape\n    Y = torch.empty_like(X)'),
        id='host_length_defined_after_block_use'),
    pytest.param(SOURCE.replace('    BLOCK = triton.next_power_of_2(N)\n', '').replace(
        '    return Y', '    BLOCK = triton.next_power_of_2(N)\n    return Y'),
        id='host_block_defined_after_launch'),
    pytest.param(SOURCE.replace('    y = xc * rstd', '    y = tl.arange(0, BLOCK).to(tl.float32)'),
        id='loaded_values_unused_by_output'),
    pytest.param(SOURCE.replace('y.to(Y.dtype.element_ty)', '0.0'), id='constant_output'),
    pytest.param(SOURCE.replace('import torch\nimport triton\nimport triton.language as tl\n', '')
                 + '\nimport torch\nimport triton\nimport triton.language as tl\n',
        id='imports_after_decorated_helper'),
])
def test_shadowed_rebound_or_out_of_order_programs_cannot_prove_the_address_chain(source):
    assert_no_stride_finding(diagnose(source))


@pytest.mark.parametrize('target,field,value', [
    ('both', 'offset', False), ('both', 'storage', False), ('both', 'writeable', 1),
    ('both', 'storage_bytes', 2048.0), ('both', 'strides', [256.0, 8]),
    ('observed', 'offset', False), ('observed', 'storage', False),
    ('observed', 'writeable', 1), ('observed', 'storage_bytes', 2048.0),
    ('observed', 'shape', [8.0, 32]), ('observed', 'strides', [256, 8.0]),
    ('both', 'offset', 2048), ('both', 'offset', -4), ('both', 'offset', 2),
    ('both', 'storage_bytes', 2043), ('both', 'storage_bytes', 0),
    ('both', 'storage_bytes', 256 * 1024 * 1024 + 1),
    ('both', 'strides', [-256, 8]),
], ids=['bool_offset', 'bool_storage', 'int_writeable', 'float_storage_bytes',
        'float_stride', 'readback_bool_offset', 'readback_bool_storage',
        'readback_int_writeable', 'readback_float_storage_bytes',
        'readback_float_shape', 'readback_float_stride', 'offset_outside_storage',
        'negative_offset', 'misaligned_offset', 'storage_too_small',
        'empty_storage', 'storage_byte_cap', 'negative_stride'])
def test_layout_readback_requires_exact_types_alignment_and_storage_bounds(target, field, value):
    results = failed_results()
    metrics = failed_case(results)['metrics']
    if target == 'both':
        metrics['input_fidelity_expected']['X'][field] = value
    metrics['input_fidelity']['observed']['X'][field] = value
    assert_no_stride_finding(diagnose(results=results))


@pytest.mark.parametrize('mutation', ['wrong_pattern', 'suite_passed_with_failed_case',
                                     'missing_tensor_devices', 'cpu_tensor_device',
                                     'unindexed_cuda_device', 'missing_cuda_device',
                                     'cpu_metric_device', 'wrong_readback_protocol'])
def test_suite_and_cuda_readback_provenance_are_required(mutation):
    results = failed_results()
    top, row = results[-1].metrics, failed_case(results)
    metrics = row['metrics']
    if mutation == 'wrong_pattern':
        row['pattern'] = 'zeros'
    elif mutation == 'suite_passed_with_failed_case':
        top['correctness']['verdict'] = 'passed'
    elif mutation == 'missing_tensor_devices':
        metrics['input_fidelity'].pop('tensor_devices')
    elif mutation == 'cpu_tensor_device':
        metrics['input_fidelity']['tensor_devices']['X'] = 'cpu'
    elif mutation == 'unindexed_cuda_device':
        metrics['input_fidelity']['tensor_devices']['X'] = 'cuda'
    elif mutation == 'missing_cuda_device':
        metrics['input_fidelity']['tensor_devices'] = {}
    elif mutation == 'cpu_metric_device':
        metrics['device'] = 'cpu'
    elif mutation == 'wrong_readback_protocol':
        metrics['input_fidelity']['protocol'] = 'unverified-layout-v0'
    assert_no_stride_finding(diagnose(results=results))


def test_changed_cuda_device_preserves_a_new_hypothesis_but_invalidates_the_old_binding():
    selected, results = task_for('layernorm-tail'), failed_results()
    original = diagnose_candidate(SOURCE, selected, results)
    failed_case(results)['metrics']['input_fidelity']['tensor_devices']['X'] = 'cuda:1'
    changed = diagnose_candidate(SOURCE, selected, results)
    assert changed['state'] == 'supported_hypothesis'
    assert changed['findings'][0]['affected_observations'][0]['input_layout']['tensor_device'] == 'cuda:1'
    assert changed['bindings']['feedback_hash'] != original['bindings']['feedback_hash']
    with pytest.raises(ValueError, match='binding drift'):
        validate_candidate_diagnosis(original, SOURCE, selected, results)


@pytest.mark.parametrize('chain', ['address', 'output_dependencies'])
def test_unsupported_shared_alias_graph_is_bounded_before_exponential_expansion(chain):
    # Fewer than the AST node cap, but naive recursive alias substitution would
    # expand 2**24 paths. These unsupported graphs must stay unknown promptly.
    if chain == 'address':
        aliases = ['    address_0 = X + row * stride_x']
        aliases += [f'    address_{n} = address_{n-1} + address_{n-1}' for n in range(1, 25)]
        source = SOURCE.replace('    x_ptr = X + row * stride_x', '\n'.join(aliases))
        source = source.replace('x_ptr + cols,', 'address_24 + cols,')
    else:
        aliases = ['    unused_0 = N']
        aliases += [f'    unused_{n} = unused_{n-1} + unused_{n-1}' for n in range(1, 25)]
        source = SOURCE.replace('    y = xc * rstd', '\n'.join(aliases) + '\n    y = unused_24')
    assert_no_stride_finding(diagnose(source))


@pytest.mark.parametrize('expression', [
    'tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32) if False else 0.0',
    'False and tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)',
    '[tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32) for _ in ()]',
    '(tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32) for _ in ())',
    '(lambda: tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32))',
], ids=['conditional_dead_load', 'short_circuited_load', 'empty_comprehension',
        'unconsumed_generator', 'uncalled_lambda'])
def test_syntactically_present_but_unexecuted_load_is_not_address_evidence(expression):
    source = SOURCE.replace(
        'tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)', expression)
    assert_no_stride_finding(diagnose(source))
