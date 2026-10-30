"""Shared generation facts/receipts; no real model or candidate execution."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from operator_agent.agents import CoderAgent
from operator_agent.generation_context import (
    build_generation_context, candidate_summary, generation_request,
    input_metadata, make_candidate_record, validate_candidate_record,
    wrapper_calling_convention, without_candidate_source,
)
from operator_agent.interface_contract import broadcast_calling_convention
from operator_agent.prompt_manager import PromptManager
from operator_agent.quality_suite import task_for
from operator_agent.run_contract import code_hash
from operator_agent.schemas import UnifiedSketch
from operator_agent import single_candidate_coder as bounded


CODE = 'def kernel(X):\n    return X\n'


def context(inputs=None, **kwargs):
    return build_generation_context(task_for('layernorm-tail'),
        UnifiedSketch(target_dsl='triton'), inputs, producer='test',
        strategy='single', **kwargs)


def test_bounded_metadata_and_abi_keep_their_public_api():
    assert bounded.input_metadata is input_metadata
    assert bounded.wrapper_calling_convention is wrapper_calling_convention
    x = np.ones((2, 5), dtype=np.float32)[:, ::2]
    metadata = input_metadata({'X': x, 'eps': 0.123456789})
    abi = wrapper_calling_convention(metadata)
    assert metadata[0]['X']['strides'] == list(x.strides)
    assert abi['stride_units'] == dict(input_metadata='bytes', runtime_tensor='elements')
    assert abi['wrapper_responsibilities']['helper_only_arguments'] == 'derive_inside_wrapper_unless_declared_input'
    assert abi['parameter_names'] == ['X', 'eps']


@pytest.mark.parametrize('inputs', [None, {}, {'X': None}])
def test_legacy_missing_metadata_stays_unknown_and_keeps_the_existing_broadcast(inputs):
    facts = context(inputs)
    assert facts['input_metadata'] is None
    assert facts['input_metadata_status'] == 'unknown'
    assert facts['interface']['wrapper'] is None
    assert facts['interface']['broadcast'] == broadcast_calling_convention(inputs)
    with pytest.raises(ValueError):
        context(inputs, strict=True)


def test_common_semantic_facts_match_across_producers_and_are_detached_from_inputs():
    task, sketch = task_for('layernorm-tail'), UnifiedSketch(target_dsl='triton')
    inputs = {'X': np.full((7, 33), 12345.678, dtype=np.float32), 'eps': 987654.321}
    before = copy.deepcopy(inputs)
    first = build_generation_context(task, sketch, inputs, producer='CoderAgent', strategy='retry')
    second = build_generation_context(task, sketch, inputs, producer='FixerAgent', strategy='repair')
    for field in ('task', 'contract', 'input_metadata', 'interface'):
        assert first[field] == second[field]
    assert first['hash'] != second['hash']
    assert '12345' not in json.dumps(first) and '987654' not in json.dumps(first)
    first['input_metadata'][0]['X']['shape'][0] = 999
    assert inputs['X'].shape == before['X'].shape
    assert second['input_metadata'][0]['X']['shape'] == [7, 33]


def test_selected_skill_identity_is_explicit_and_single_candidate_does_not_gain_skills():
    manager = PromptManager()
    selected = manager._get_skill_for_agent('coder', 'triton_skill')
    facts = context(prompt_manager=manager, skill_name='triton_skill')
    assert facts['knowledge']['skill'] == dict(name='triton_skill', role='coder', sha256=code_hash(selected))
    assert context()['knowledge']['skill'] is None


def test_controller_receipt_omits_source_without_inventing_an_absent_candidate():
    original = make_candidate_record(context(), CODE, status='precheck_failed')
    retained = without_candidate_source(original)
    assert validate_candidate_record(retained)
    assert original['source'] == CODE and original['source_retention'] == 'full'
    assert retained['source'] is None and retained['source_retention'] == 'hash_only'
    assert retained['source_hash'] == original['source_hash'] == code_hash(CODE)
    assert retained['record_hash'] != original['record_hash']
    assert retained['context_hash'] == original['context_hash']
    assert retained['status'] == 'precheck_failed'
    assert candidate_summary(retained)['source_retention'] == 'hash_only'
    absent = make_candidate_record(context(), None, status='model_failed')
    assert without_candidate_source(absent) == absent
    retained['source_hash'] = 'invalid'
    from operator_agent.run_contract import digest
    retained['record_hash'] = digest({k: v for k, v in retained.items() if k != 'record_hash'})
    with pytest.raises(ValueError):
        validate_candidate_record(retained)


def test_request_identity_records_actual_messages_without_client_secrets_or_claimed_usage():
    llm = SimpleNamespace(model='configured-model', api_key='PRIVATE-KEY',
                          base_url='https://PRIVATE-ENDPOINT.invalid', call_budget=None)
    messages = [dict(role='user', content='PRIVATE-PROMPT')]
    request = generation_request(llm, messages, caller='Coder', temperature=.1, max_tokens=100)
    assert request['configured_model'] == 'configured-model'
    assert request['prompt_hashes'] == [code_hash('PRIVATE-PROMPT')]
    assert request['budget'] is None
    assert all(secret not in json.dumps(request) for secret in
               ('PRIVATE-KEY', 'PRIVATE-ENDPOINT', 'PRIVATE-PROMPT'))
    assert 'not_provider_response' in request['identity_scope']


@pytest.mark.parametrize('status', ['prechecked', 'precheck_failed', 'proposed', 'model_failed', 'no_candidate'])
def test_receipt_never_grants_correctness_and_compact_projection_omits_source_and_context(status):
    source = None if status in ('model_failed', 'no_candidate') else CODE
    record = make_candidate_record(context(), source, status=status)
    assert validate_candidate_record(record) is record
    assert record['source_hash'] == (code_hash(source) if source else None)
    assert record['correctness_proven'] is False
    assert record['candidate_execution'] is False
    assert record['restored_eligibility'] is False
    assert record['gpu_measurements'] == 0
    summary = candidate_summary(record)
    assert summary['source_hash'] == record['source_hash']
    assert 'source' not in summary and 'context' not in summary
    assert 'passed' not in summary and 'eligibility' not in summary
    assert len(json.dumps(summary)) < 1500


@pytest.mark.parametrize('mutation', ['source', 'context', 'correctness', 'record_hash'])
def test_tampered_receipts_are_rejected_even_when_not_used_as_verification_credentials(mutation):
    record = make_candidate_record(context(), CODE)
    if mutation == 'source': record['source'] += '# changed'
    elif mutation == 'context': record['context']['task']['hardware'] = 'changed'
    elif mutation == 'correctness': record['correctness_proven'] = True
    else: record['record_hash'] = '0' * 64
    with pytest.raises(ValueError): validate_candidate_record(record)
    with pytest.raises(ValueError): candidate_summary(record)


def test_mechanical_transforms_keep_a_separate_parent_and_author():
    initial = make_candidate_record(context(), CODE)
    revised = CODE + '# normalized\n'
    derived = make_candidate_record(context(), revised, source_author='mechanical_transform',
        parent_source=CODE, status='transformed', transformation=dict(operation='test-normalization'))
    assert initial['source'] == CODE
    assert derived['parent_source_hash'] == initial['source_hash']
    assert derived['source_author'] != initial['source_author']
    assert derived['static_precheck'] == 'not_checked'


def test_real_coder_entry_preserves_retry_count_and_source_return_and_records_rejected_attempts():
    class LLM:
        def __init__(self): self.messages = []
        def chat(self, messages, **kwargs):
            self.messages.append(copy.deepcopy(messages))
            assert kwargs == dict(temperature=.1, max_tokens=40960, caller_tag='Coder')
            return 'def kernel(' if len(self.messages) == 1 else CODE
    llm = LLM()
    coder = CoderAgent(SimpleNamespace(search=lambda **kwargs: []), llm=llm)
    task = task_for('layernorm-tail', dsl='python')
    source = coder.run(UnifiedSketch(target_dsl='python'), task, test_inputs={'X': None})
    assert source == CODE.strip()
    assert len(llm.messages) == 2
    assert [r['status'] for r in coder.candidate_records] == ['precheck_failed', 'prechecked']
    assert coder.last_candidate_record['source'] == source
    assert coder.last_candidate_record['context']['interface']['broadcast'] == broadcast_calling_convention({'X': None})
    assert all(r['correctness_proven'] is False for r in coder.candidate_records)
    assert llm.messages[0][0]['content'] != llm.messages[1][0]['content']
