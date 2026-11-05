"""Host receives domain materials without model calls or private test inputs."""
import copy
import json
from dataclasses import replace

import pytest

from operator_agent.app.hitl_tools import build_task
from operator_agent.core.contract import freeze_from_task
from operator_agent.core.run_contract import code_hash, digest
from operator_agent.generate.host_context import MAX_CONTEXT_BYTES, build_host_context


CODE = 'import numpy as np\ndef kernel(X):\n    return X.copy()\n'


def task(dsl='python', operator='relu'):
    return build_task(dict(request=operator+' fp32', shapes=[[4, 8]], precision='fp32',
                           dsl=dsl, hardware='mock_gpu' if dsl == 'python' else 'nvidia_gpu'))


def feedback(t, source=CODE):
    return dict(ok=True, source_hash=code_hash(source), contract_hash=freeze_from_task(t).hash,
        levels=dict(L1=True, L2=False), verdicts=[
            dict(level='L1', passed=True, metrics={}),
            dict(level='L2', passed=False, metrics=dict(max_error=1.0, coverage=1.0))],
        validity=None, eligibility=None, conflict=dict(present=False))


def all_keys(value):
    if isinstance(value, dict):
        return set(value).union(*(all_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(all_keys(v) for v in value))
    return set()


def test_host_receives_frozen_facts_abi_and_fake_restrictions_without_execution(monkeypatch):
    from operator_agent.decide.bench_cases import BenchmarkCase
    from operator_agent.execute.llm_client import LLMClient
    from operator_agent.execute.fake_backend.backend import FakeBackend
    from operator_agent.decide.verifier import Verifier

    def forbidden(*args, **kwargs):
        raise AssertionError('context must not materialize, probe, execute or call a model')

    monkeypatch.setattr(BenchmarkCase, 'materialize', forbidden)
    monkeypatch.setattr(LLMClient, '__init__', forbidden)
    monkeypatch.setattr(Verifier, 'verify', forbidden)
    monkeypatch.setattr(FakeBackend, '__init__', forbidden)
    monkeypatch.setattr(FakeBackend, 'get_hardware_info', forbidden)
    t = task()
    context = build_host_context(t, freeze_from_task(t), backend='fake')
    facts = context['authoritative']
    assert context['context_owner'] == 'host' and context['model_calls'] == 0
    assert facts['interface']['signature'] == 'def kernel(X):'
    assert facts['task']['declared_shapes'] == [[4, 8]]
    assert facts['inputs'][0]['X']['shape'] == [4, 8]
    assert facts['inputs'][0]['X']['dtype'] == 'float32'
    assert facts['inputs'][0]['X']['strides'] is None
    assert facts['runtime_layout_status'].startswith('not_observed')
    assert 'Import only numpy/math.' in facts['execution_backend']['restrictions']
    assert facts['execution_backend']['test_only'] is True
    assert facts['contract']['reference_semantics']['definition'] == 'elementwise maximum of X and zero'
    assert not {'messages', 'seed', 'reference_source', 'input_values'} & all_keys(context)


def test_generation_materials_are_unrendered_and_selected_by_role():
    t = task('triton', 'softmax')
    context = build_host_context(t, freeze_from_task(t))
    templates = context['materials']['templates']
    assert [row['name'] for row in templates] == ['designer_sketch', 'coder_generate_triton']
    assert all(row['rendering'] == 'raw_reference_template_not_messages' for row in templates)
    assert '{{declared_shapes}}' in templates[1]['text']
    skills = context['materials']['skills']
    assert {row['role'] for row in skills} == {'designer', 'coder'}
    assert all(row['sha256'] == code_hash(row['text']) for row in templates + skills)
    spec = context['rules']['generation_spec']
    assert spec['sha256'] == code_hash(spec['text'])
    assert context['authoritative']['contract']['reference_semantics']['normalized_axis'] == -1
    assert context['material_integrity']['complete'] is True
    assert context['material_integrity']['truncated'] is False
    assert {row['sha256'] for row in context['material_integrity']['required']} == {
        spec['sha256'], *(row['sha256'] for row in skills)}
    # The supported DSH text projection uses unescaped Unicode and a default
    # 12,500-token heuristic spill threshold (~50K chars). Keep headroom for
    # the MCP envelope; the full-artifact path handles larger task domains.
    assert len(json.dumps(context, ensure_ascii=False, indent=2)) < 40000
    assert len(json.dumps(context, ensure_ascii=False, separators=(',', ':')).encode()) <= MAX_CONTEXT_BYTES


def test_triton_revision_materials_fit_inline_limit_and_oversized_source_is_refused():
    t = task('triton', 'softmax')
    c = freeze_from_task(t)
    source = '''import torch
import triton
import triton.language as tl

@triton.jit
def _softmax(X, Y, N: tl.constexpr, SX0, SX1, SY0, SY1, B: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, B)
    x = tl.load(X + row * SX0 + cols * SX1, cols < N, other=-float("inf")).to(tl.float32)
    y = tl.exp(x - tl.max(x, 0))
    y = y / tl.sum(y, 0)
    tl.store(Y + row * SY0 + cols * SY1, y, cols < N)

def kernel(X):
    rows, cols = X.shape
    Y = torch.empty_like(X)
    _softmax[(rows,)](X, Y, cols, X.stride(0), X.stride(1), Y.stride(0), Y.stride(1),
                      triton.next_power_of_2(cols))
    return Y
'''
    result = build_host_context(t, c, phase='repair', source=source, verification=feedback(t, source))
    assert result['material_integrity']['truncated'] is False
    assert len(json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode()) <= MAX_CONTEXT_BYTES
    oversized = source + ('# retained candidate context\n' * 1000)
    with pytest.raises(ValueError, match='context byte limit'):
        build_host_context(t, c, phase='repair', source=oversized, verification=feedback(t, oversized))


def test_context_hash_changes_with_materials_and_returned_values_are_detached():
    t = task()
    c = freeze_from_task(t)
    first = build_host_context(t, c, backend='fake')
    second = build_host_context(t, c, backend='mock')
    assert first['hash'] == digest({k: v for k, v in first.items() if k != 'hash'})
    assert first['hash'] != second['hash']
    first['authoritative']['inputs'][0]['X']['shape'][0] = 999
    assert c.cases[0].tensors[0].shape == (4, 8)
    assert second['authoritative']['inputs'][0]['X']['shape'] == [4, 8]


def test_reference_source_and_test_seed_are_not_exported():
    t = task()
    t.source.reference_source = 'PRIVATE_REFERENCE_BODY_DO_NOT_EXPORT'
    t.constraints.correctness_profile = None
    c = freeze_from_task(t)
    changed = replace(c, cases=(replace(c.cases[0], seed=934810271),), hash='')
    changed = replace(changed, hash=changed.compute_hash())
    # Update the task-owned case too, so this is a consistent new contract.
    t.constraints.cases = tuple(case.to_dict() for case in changed.cases)
    c = freeze_from_task(t)
    context = build_host_context(t, c)
    encoded = json.dumps(context)
    assert 'PRIVATE_REFERENCE_BODY_DO_NOT_EXPORT' not in encoded
    assert '934810271' not in encoded
    assert context['authoritative']['contract']['reference_source_hash'] == code_hash(t.source.reference_source)
    assert context['authoritative']['contract']['reference_semantics']['status'] == 'unknown'


@pytest.mark.parametrize('phase', ['repair', 'optimize'])
def test_revision_context_requires_current_source_and_bound_verification(phase):
    t = task()
    c = freeze_from_task(t)
    with pytest.raises(ValueError, match='require source and verification'):
        build_host_context(t, c, phase=phase)
    for field in ('source_hash', 'contract_hash'):
        stale = feedback(t)
        stale[field] = '0' * 64
        with pytest.raises(ValueError, match='binding drift'):
            build_host_context(t, c, phase=phase, source=CODE, verification=stale)
    result = build_host_context(t, c, backend='fake', phase=phase, source=CODE, verification=feedback(t))
    assert result['candidate']['source_hash'] == code_hash(CODE)
    assert result['feedback']['levels'] == dict(L1=True, L2=False)
    assert result['diagnosis']['state'] == 'unknown'
    assert result['diagnosis']['bindings']['source_hash'] == code_hash(CODE)
    assert result['diagnosis']['bindings']['contract_hash'] == c.hash
    names = [row['name'] for row in result['materials']['templates']]
    assert names == (['fixer_repair'] if phase == 'repair' else ['search_optimize', 'coder_generate'])


def test_feedback_projection_omits_arbitrary_logs_values_and_reference_code():
    t = task()
    v = feedback(t)
    secret = 'PRIVATE_INPUT_OR_REFERENCE_LITERAL'
    v['verdicts'][1].update(message=secret, error_log=secret)
    v['verdicts'][1]['metrics'].update(seed=934810271, input_values=[secret], reference_source=secret,
        reference_identity_unknown=True)
    result = build_host_context(t, freeze_from_task(t), phase='repair', source=CODE, verification=v)
    assert secret not in json.dumps(result)
    assert not {'seed', 'input_values', 'reference_source', 'error_log'} & all_keys(result)
    assert result['feedback']['hold_reasons'] == ['unknown_reference_or_input_fidelity']


def test_inconsistent_or_incomplete_feedback_is_refused():
    t = task()
    c = freeze_from_task(t)
    for update in (dict(levels=dict(L1=True, L2=True)), dict(evidence_complete=False),
                   dict(binding=dict(source_hash='0' * 64, contract_hash=c.hash))):
        v = dict(feedback(t), **update)
        with pytest.raises(ValueError):
            build_host_context(t, c, phase='repair', source=CODE, verification=v)


def test_frozen_contract_drift_and_unsupported_modes_are_refused():
    t = task()
    c = freeze_from_task(t)
    with pytest.raises(ValueError, match='drift'):
        build_host_context(t, replace(c, min_speedup=2.0))
    for options in (dict(phase='hidden_model_loop'), dict(backend='auto')):
        with pytest.raises(ValueError):
            build_host_context(t, c, **options)
    t = task('triton')
    with pytest.raises(ValueError, match='requires python DSL'):
        build_host_context(t, freeze_from_task(t), backend='fake')


def test_source_bound_real_verification_can_supply_revision_materials():
    from operator_agent.decide.verifier import Verifier
    from operator_agent.decide.tool_evidence import verification_data
    from operator_agent.execute.fake_backend.backend import FakeBackend
    from operator_agent.core.schemas import VerificationLevel

    t = task()
    c = freeze_from_task(t)
    results = Verifier(FakeBackend()).verify(CODE, t, c.cases[0].materialize(),
                                            stop_after=VerificationLevel.L2_FUNCTION)
    v = dict(ok=True, source_hash=code_hash(CODE), contract_hash=c.hash,
        levels={row.level.value: row.passed for row in results}, verdicts=verification_data(results))
    before = copy.deepcopy(v)
    context = build_host_context(t, c, backend='fake', phase='repair', source=CODE, verification=v)
    assert context['feedback']['levels']['L1'] is True
    assert context['feedback']['levels']['L2'] is False
    assert context['diagnosis']['candidate_execution'] is False
    assert v == before
