"""Offline guidance/lineage guards and trusted CPU counterexamples, never CUDA claims.

The historical Triton source is read as text only. The small Python programs
below are independent test controls executed by MockGPU, not repaired candidates.
"""
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from benchmarks.drivers import workload_codegen_h20 as driver
from operator_agent import (CorrectnessCase, CorrectnessSuite, ConstraintSpec,
                            MockGPU, SourceSpec, TargetSpec, TaskSchema, Verifier)
from operator_agent import llm_client as lc, single_candidate_coder as sc
from operator_agent.bench_cases import BenchmarkCase, TensorSpec
from operator_agent.correctness import credential
from operator_agent.llm_budget import LLMCallBudget, LLMCallRefused
from operator_agent.run_contract import code_hash, json_value, protocol_identity
from operator_agent.schemas import Precision, VerificationLevel as Level, VerificationResult
from tests.unit.test_single_candidate_coder import response
from tests.unit.test_workload_codegen_driver import retain_model


HISTORICAL = (Path(__file__).resolve().parents[1]/'fixtures/single_candidate_unmasked_variance.txt').read_text()
HISTORICAL_HASH = 'bdb1698cfa2e9a6a341ecf35e2987d8d4349384ed43b23d71661b4500853b2cf'

# Trusted CPU-only controls for the existing numerical/suite gate.
GOOD_CPU = '''import numpy as np
def kernel(X):
    x = X.astype(np.float32)
    centered = x - x.mean(axis=-1, keepdims=True)
    return (centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5)).astype(X.dtype)
'''
BAD_PADDING_CPU = '''import numpy as np
def kernel(X):
    n = X.shape[-1]
    block = 1 << (n - 1).bit_length()
    x = np.pad(X.astype(np.float32), ((0, 0), (0, block - n)))
    mean = x.sum(axis=-1, keepdims=True) / n
    centered = x - mean
    variance = (centered * centered).sum(axis=-1, keepdims=True) / n
    return (centered[:, :n] / np.sqrt(variance + 1e-5)).astype(X.dtype)
'''
BAD_STRIDES_CPU = GOOD_CPU.replace('x = X.astype(np.float32)', '''if not X.flags.c_contiguous:
        X = X.base.reshape(-1)[:X.size].reshape(X.shape)
    x = X.astype(np.float32)''')


def model_fixture(tmp_path, monkeypatch):
    # Tiny metadata instead of allocating the real 432 MiB workloads.
    shapes = ((2, 32), (3, 33))
    monkeypatch.setattr(driver.wd, 'LARGE', shapes)
    task = driver.wd.task('offline-semantic-fixture', shapes=shapes)
    task.constraints.min_speedup = 10.
    inputs = [{'X': np.ones(shape, np.float16)} for shape in shapes]
    sketch = driver.sketch()
    client = driver.PlanLLMClient(base_url='https://fixture.invalid/v1', model='fixture',
        api_key='PRIVATE_KEY', temperature=.1, timeout=60,
        call_budget=LLMCallBudget(driver.LIMITS, tmp_path/'model-journal'))
    coder = sc.SingleCandidateCoder(client, task, sketch, inputs)
    gen = coder.declaration()
    assert gen == driver.generation_declaration(task)
    identity = client.call_budget.summary()['identity']
    body = driver.request_body(gen, identity)
    raw = json.dumps(body, ensure_ascii=False, allow_nan=False)
    plan = dict(generation=gen, contract=gen['context']['contract'], model=identity,
        model_limits=json_value(driver.asdict(driver.LIMITS)), measurement_protocol=protocol_identity(),
        request=dict(body=body, sha256=code_hash(raw), bytes=len(raw.encode())))
    client.frozen_plan = plan
    client.plan_guard = lambda: None
    send = Mock(return_value=response(HISTORICAL))
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    return coder, task, sketch, inputs, plan, send


def test_semantic_rules_reach_exact_frozen_request_without_values_or_template(tmp_path, monkeypatch):
    coder, task, sketch, inputs, plan, send = model_fixture(tmp_path, monkeypatch)
    assert coder.run(sketch, task, test_inputs=inputs) == HISTORICAL
    assert send.call_count == 1
    raw = send.call_args.args[0]['body_text']
    assert raw == json.dumps(plan['request']['body'], ensure_ascii=False, allow_nan=False)
    assert code_hash(raw) == plan['request']['sha256']
    assert len(raw.encode()) == plan['request']['bytes'] < driver.LIMITS.max_request_bytes
    body = json.loads(raw)
    prompt = body['messages'][0]['content']
    context = json.loads(body['messages'][1]['content'])
    hints = ' '.join(context['sketch']['optimization_hints'])
    assert 'reduction identity after all transformations' in prompt
    assert 'logical reduction length for normalization' in prompt
    assert 'serialized NumPy strides are in bytes' in prompt
    assert 'independently for inputs and outputs' in prompt
    assert 'both the mean and centered-square variance reductions' in hints
    assert 'nonzero constant and near-constant rows' in hints
    assert 'actual input row/column strides and independent output strides' in hints
    assert context['input_metadata'] == sc.input_metadata(inputs)
    assert context['input_metadata'][1]['X']['strides'] == [66, 2]
    assert all(set(row['X']) == {'shape', 'dtype', 'strides'} for row in context['input_metadata'])
    assert len(context['contract']['correctness_suite']['cases']) == 32
    assert context['contract']['reference_contract']['rtol'] == .01
    assert context['contract']['reference_contract']['required_matched_ratio'] == 1.
    assert context['reference_source'] is None
    # The semantic guidance is general, not feedback or a shape/value answer.
    for text in (prompt, hints):
        for forbidden in ('tl.load', 'tl.store', '15360', '12288', '1.3333', '0.04296875', HISTORICAL_HASH):
            assert forbidden not in text
    assert HISTORICAL not in raw and 'PRIVATE_KEY' not in raw


def test_bad_historical_math_stays_exact_and_prechecked_without_numerical_credential(tmp_path, monkeypatch):
    assert code_hash(HISTORICAL) == HISTORICAL_HASH
    coder, task, sketch, inputs, plan, send = model_fixture(tmp_path, monkeypatch)
    assert sc.precheck(HISTORICAL, sc.input_metadata(inputs)) is None
    assert coder.run(sketch, task, test_inputs=inputs) == HISTORICAL
    retain_model(tmp_path, coder.llm)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    proof = driver.audit_model(tmp_path, plan)
    assert proof['completed'] and proof['precheck_only'] and proof['model_authored_source']
    assert proof['gpu_measurements'] == 0 and proof['restored_eligibility'] is False
    assert proof['source'] == HISTORICAL and proof['candidate_hash'] == HISTORICAL_HASH
    assert coder.last_generation['status'] == 'prechecked'
    assert credential([]) is None
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(LLMCallRefused, match='already attempted'):
        coder.run(sketch, task, test_inputs=inputs)
    assert send.call_count == 1


@pytest.mark.parametrize('change', ('prompt', 'mask_hint', 'stride_hint'))
def test_semantic_guidance_drift_stops_before_request(tmp_path, monkeypatch, change):
    coder, task, sketch, inputs, _, send = model_fixture(tmp_path, monkeypatch)
    if change == 'prompt':
        monkeypatch.setattr(sc, 'PROMPT', sc.PROMPT.replace('after all transformations', 'at load only'))
    else:
        sketch.optimization_hints[2 if change == 'mask_hint' else 3] = 'undeclared semantics'
    with pytest.raises(ValueError, match='drift'):
        coder.run(sketch, task, test_inputs=inputs)
    assert send.call_count == coder.llm.call_budget.summary()['reserved_calls'] == 0


def test_stride_aware_wrapper_is_allowed_by_static_gate():
    # Independent static-only copy fixture, not a LayerNorm implementation.
    source = '''import torch
import triton
import triton.language as tl
@triton.jit
def copy_strided(X, Y, N, SX0, SX1, SY0, SY1, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * SX0 + col * SX1, mask=col < N, other=0)
    tl.store(Y + row * SY0 + col * SY1, value, mask=col < N)
def kernel(X):
    Y = torch.empty_like(X)
    copy_strided[(X.shape[0],)](X, Y, X.shape[1], X.stride(0), X.stride(1),
                               Y.stride(0), Y.stride(1), triton.next_power_of_2(X.shape[1]))
    return Y
'''
    assert sc.precheck(source, [{'X': dict(shape=[3, 33], dtype='float16', strides=[132, 4])}]) is None


def cpu_case(n=33, pattern='near_constant', layout='contiguous'):
    return CorrectnessCase('semantic-counterexample', BenchmarkCase(
        tensors=(TensorSpec('X', (3, n), 'fp32'),), seed=17), pattern=pattern, layout=layout)


def cpu_verify(source, case, primary=None):
    task = TaskSchema(source=SourceSpec(kernel_type='layernorm'), target=TargetSpec(dsl='python'),
        constraints=ConstraintSpec(precision=Precision.FP32, max_error=1e-5,
            correctness_suite=CorrectnessSuite((case,))))
    primary = primary if primary is not None else cpu_case(n=32, pattern='random').materialize()
    verifier = Verifier(MockGPU())
    # Use an explicit NumPy CPU oracle. Optional local Torch builds can differ
    # at the fp32 near-constant tolerance boundary; this test isolates padding
    # and addressing errors, not the separately pinned native CUDA reference.
    def cpu_reference(task, inputs, high_precision=False):
        x = inputs['X'].astype(np.float64 if high_precision else np.float32)
        out = (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(
            np.var(x, axis=-1, keepdims=True) + 1e-5)
        return out
    verifier._generate_reference = cpu_reference
    verifier._verify_performance = Mock(return_value=VerificationResult(Level.L3_PERFORMANCE, True))
    results = verifier.verify(source, task, primary)
    return results, verifier


@pytest.mark.parametrize('n', (3, 31, 33))
@pytest.mark.parametrize('pattern', ('near_constant', 'large_varying'))
@pytest.mark.parametrize('layout', ('contiguous', 'strided'))
def test_padding_variance_counterexample_blocks_l3_after_aligned_primary(n, pattern, layout):
    results, verifier = cpu_verify(BAD_PADDING_CPU, cpu_case(n, pattern, layout))
    assert results[0].passed
    metrics = results[-1].metrics
    assert metrics['matched_ratio'] == 1. and metrics['bad_elements'] == 0
    assert metrics['correctness']['verdict'] == 'failed'
    row = metrics['correctness']['cases'][0]
    assert row['verdict'] == 'failed' and row['metrics']['bad_elements'] > 0
    assert row['metrics']['coverage'] == 1.
    assert credential(results) is False
    verifier._verify_performance.assert_not_called()


@pytest.mark.parametrize('layout', ('contiguous', 'strided'))
def test_primary_failure_keeps_suite_not_run_and_blocks_performance(layout):
    case = cpu_case(layout=layout)
    results, verifier = cpu_verify(BAD_PADDING_CPU, case, primary=case.materialize())
    metrics = results[-1].metrics
    assert metrics['correctness_primary_verdict'] == 'failed'
    assert metrics['correctness']['verdict'] == 'failed'
    assert all(row['verdict'] == 'not_run' for row in metrics['correctness']['cases'])
    verifier._verify_performance.assert_not_called()


@pytest.mark.parametrize('n', (3, 31, 32, 33))
def test_contiguous_address_assumption_is_killed_by_existing_strided_case(n):
    results, verifier = cpu_verify(BAD_STRIDES_CPU, cpu_case(n, 'random', 'strided'))
    metrics = results[-1].metrics
    assert metrics['matched_ratio'] == 1. and metrics['bad_elements'] == 0
    assert metrics['correctness']['cases'][0]['verdict'] == 'failed'
    assert credential(results) is False
    verifier._verify_performance.assert_not_called()


@pytest.mark.parametrize('pattern', ('random', 'constant_nonbinary', 'near_constant', 'large_varying'))
@pytest.mark.parametrize('layout', ('contiguous', 'strided'))
def test_trusted_cpu_control_passes_same_cases_and_deterministic_repeat(pattern, layout):
    results, verifier = cpu_verify(GOOD_CPU, cpu_case(pattern=pattern, layout=layout))
    suite = results[1].metrics['correctness']
    assert suite['verdict'] == 'passed' and suite['cases'][0]['determinism'] == 'passed'
    assert credential(results) is True
    verifier._verify_performance.assert_called_once()
