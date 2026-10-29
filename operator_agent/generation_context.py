"""Shared generation facts and candidate receipts, independent of strategy.

This module never calls a model, executes a candidate, selects a winner or grants
verification eligibility. Adapters retain their existing prompts and budgets.
"""
from __future__ import annotations

import keyword
import math
import re
from copy import deepcopy

import numpy as np

from .contract import freeze_from_task
from .generation_spec import generation_spec_identity
from .interface_contract import broadcast_calling_convention
from .llm_budget import LLMCallBudget
from .run_contract import code_hash, digest, json_value


CONTEXT_SCHEMA = 'operator-generation-context-v1'
RECORD_SCHEMA = 'operator-candidate-record-v1'


def input_metadata(test_inputs):
    """Only names/layout/dtypes, never tensor elements or scalar values."""
    sets = test_inputs if isinstance(test_inputs, (list, tuple)) else [test_inputs]
    result = []
    if not sets:
        raise ValueError('explicit input interface required')
    for inputs in sets:
        if not isinstance(inputs, dict) or not inputs or any(type(k) is not str for k in inputs):
            raise ValueError('explicit named inputs required')
        row = {}
        for key, value in sorted(inputs.items()):
            if isinstance(value, np.ndarray) and not value.dtype.hasobject:
                row[key] = dict(shape=list(value.shape), dtype=str(value.dtype), strides=list(value.strides))
            elif type(value) in (bool, int, float):
                if type(value) is float and not math.isfinite(value):
                    raise ValueError('nonfinite scalar input')
                row[key] = dict(scalar_type=type(value).__name__, value_disclosed=False)
            else:
                raise ValueError('unsupported input metadata; no implicit serialization')
        result.append(row)
    return result


def wrapper_calling_convention(metadata):
    """The caller-derived bounded wrapper ABI, without input values."""
    if (not isinstance(metadata, (list, tuple)) or not metadata
            or any(type(row) is not dict or not row for row in metadata)):
        raise ValueError('explicit named input metadata required for wrapper convention')
    if any(type(name) is not str or not name.isidentifier() or keyword.iskeyword(name)
           for row in metadata for name in row):
        raise ValueError('wrapper inputs must have explicit Python parameter names')
    sets = sorted({tuple(sorted(row)) for row in metadata})
    names = sorted(set().union(*(set(row) for row in sets)))
    required = sorted(set.intersection(*(set(row) for row in sets)))
    optional = sorted(set(names) - set(required))
    signature = 'def kernel(' + ', '.join(required + [name+'=None' for name in optional]) + '):'
    return dict(schema='keyword-wrapper-v1', entrypoint='kernel', invocation='kernel(**inputs)',
        keyword_sets=[list(row) for row in sets], parameter_names=names,
        signature=signature, extra_parameters_allowed=False,
        parameter_style='explicit_keyword_compatible',
        returns=dict(kind='tensor', via='return_value', allocated_by='wrapper',
                     shape_and_dtype='frozen_operator_semantics'),
        wrapper_responsibilities=dict(output_allocation='inside_wrapper',
            shapes_and_reduction_lengths='derive_from_runtime_inputs',
            tensor_strides='read_inputs_and_outputs_independently',
            semantic_constants='frozen_contract_or_registered_reference',
            helper_only_arguments='derive_inside_wrapper_unless_declared_input'),
        stride_units=dict(input_metadata='bytes', runtime_tensor='elements'))


def _text(value):
    return value if type(value) is str else None


def build_generation_context(task, sketch=None, test_inputs=None, *, producer,
                             strategy, policy=None, parent_source=None, strict=False,
                             prompt_manager=None, skill_name=None, agent_role='coder'):
    """Build detached facts for every adapter, preserving legacy unknown inputs.

    Strict mode is used by the already-strict single-candidate adapter. Other
    callers retain their existing acceptance policy: missing facts stay unknown.
    """
    target, constraints, source = task.target, task.constraints, task.source
    precision = getattr(constraints.precision, 'value', constraints.precision)
    declared = dict(task_id=_text(getattr(task, 'task_id', None)),
        kernel_type=source.kernel_type, hardware=target.hardware,
        dsl=target.dsl, precision=precision)
    try:
        metadata = input_metadata(test_inputs)
        wrapper = wrapper_calling_convention(metadata)
    except (ValueError, TypeError):
        if strict:
            raise
        metadata, wrapper = None, None
    try:
        contract = freeze_from_task(task).as_dict()
    except (ValueError, TypeError, AttributeError):
        if strict:
            raise
        contract = None
    skill = None
    if prompt_manager is not None and skill_name:
        reader = getattr(prompt_manager, '_get_skill_for_agent', None)
        if callable(reader):
            selected = reader(agent_role, skill_name)
            if type(selected) is str and selected:
                skill = dict(name=skill_name, role=agent_role, sha256=code_hash(selected))
    sketch_data = None
    if sketch is not None and callable(getattr(sketch, 'to_dict', None)):
        try:
            sketch_data = json_value(sketch.to_dict())
        except (TypeError, ValueError):
            if strict:
                raise
    body = dict(schema=CONTEXT_SCHEMA, producer=producer, strategy=strategy,
        task=declared, contract=contract, contract_status='known' if contract else 'unknown',
        reference_source_hash=(code_hash(source.reference_source)
            if type(getattr(source, 'reference_source', None)) is str and source.reference_source else None),
        sketch=sketch_data, input_metadata=metadata,
        input_metadata_status='known' if metadata is not None else 'unknown',
        interface=dict(broadcast=broadcast_calling_convention(test_inputs), wrapper=wrapper),
        knowledge=dict(generation_spec=generation_spec_identity(), skill=skill),
        policy=policy or {},
        parent_source_hash=code_hash(parent_source) if type(parent_source) is str else None)
    # Callers receive a fresh JSON value; neither task nor input objects escape.
    body = json_value(body)
    return dict(body, hash=digest(body))


def generation_request(llm, messages, *, caller, temperature, max_tokens):
    """Identity of the sent messages and configured client, not provider proof.

    Never inspect or store credentials, raw URLs or arbitrary client reprs.
    Existing request journals remain authoritative for actual API usage.
    """
    budget = getattr(llm, 'call_budget', None)
    ledger = None
    if isinstance(budget, LLMCallBudget):
        summary = budget.summary()
        ledger = dict(reserved_calls_before=summary['reserved_calls'],
                      max_calls=budget.limits.max_calls)
    return dict(caller=caller, configured_model=_text(getattr(llm, 'model', None)),
        messages_hash=digest(messages),
        prompt_hashes=[code_hash(message['content']) for message in messages
                       if type(message.get('content')) is str],
        temperature=temperature, requested_max_output_tokens=max_tokens, budget=ledger,
        identity_scope='configured_client_and_sent_messages_not_provider_response')


def make_candidate_record(context, source, *, request=None, attempt=1,
                          status='proposed', source_author='model', parent_source=None,
                          transformation=None, reason=None):
    """Return one common receipt, including rejected and absent candidates."""
    body = {key: value for key, value in context.items() if key != 'hash'}
    if context.get('schema') != CONTEXT_SCHEMA or context.get('hash') != digest(body):
        raise ValueError('generation context drift')
    if type(attempt) is not int or attempt < 1:
        raise ValueError('positive generation attempt required')
    if source is not None and type(source) is not str:
        raise TypeError('candidate source must be text or absent')
    source = source if source else None
    record = dict(schema=RECORD_SCHEMA, producer=context['producer'],
        strategy=context['strategy'], context_hash=context['hash'], context=context,
        request=request, attempt=attempt, source_author=source_author,
        source=source, source_hash=code_hash(source) if source is not None else None,
        source_retention='full' if source is not None else 'absent',
        parent_source_hash=(code_hash(parent_source) if type(parent_source) is str
                            else context.get('parent_source_hash')),
        status=status, static_precheck=('passed' if status == 'prechecked' else
            'failed' if status == 'precheck_failed' else 'not_checked'),
        transformation=transformation, reason=reason,
        candidate_execution=False, correctness_proven=False, gpu_measurements=0,
        restored_eligibility=False)
    record = json_value(record)
    return dict(record, record_hash=digest(record))


def without_candidate_source(record):
    """Retain a receipt without persisting a proposal through the action gate.

    The producer still returns its original source. Conductor audit files must
    not become another place to write refused executable source (§21/§22).
    """
    validate_candidate_record(record)
    retained = deepcopy(record)
    if retained['source_hash'] is not None:
        retained['source'] = None
        retained['source_retention'] = 'hash_only'
        retained['record_hash'] = digest({k: v for k, v in retained.items() if k != 'record_hash'})
    return retained


def candidate_summary(record):
    """Small public projection; full source/context remain in evidence files."""
    if not isinstance(record, dict) or record.get('schema') != RECORD_SCHEMA:
        return None
    validate_candidate_record(record)
    fields = ('schema', 'producer', 'strategy', 'context_hash', 'record_hash',
              'source_author', 'source_hash', 'source_retention', 'parent_source_hash', 'attempt',
              'status', 'static_precheck', 'reason', 'correctness_proven',
              'restored_eligibility')
    return json_value({key: record[key] for key in fields})


def validate_candidate_record(record):
    """Check receipt consistency; this does not attest its author or correctness."""
    if not isinstance(record, dict) or record.get('schema') != RECORD_SCHEMA:
        raise ValueError('unknown candidate record')
    body = {key: value for key, value in record.items() if key != 'record_hash'}
    context = record.get('context')
    if not isinstance(context, dict):
        raise ValueError('candidate context missing')
    facts = {key: value for key, value in context.items() if key != 'hash'}
    source = record.get('source')
    retention = record.get('source_retention')
    source_hash = record.get('source_hash')
    source_valid = (
        (retention == 'full' and type(source) is str and source_hash == code_hash(source))
        or (retention == 'absent' and source is None and source_hash is None)
        or (retention == 'hash_only' and source is None and type(source_hash) is str
            and re.fullmatch('[0-9a-f]{64}', source_hash) is not None))
    if (record.get('record_hash') != digest(body)
            or context.get('schema') != CONTEXT_SCHEMA
            or context.get('hash') != digest(facts)
            or record.get('context_hash') != context['hash']
            or not source_valid
            or any(record.get(key) is not False for key in
                   ('candidate_execution', 'correctness_proven', 'restored_eligibility'))
            or type(record.get('gpu_measurements')) is not int
            or record['gpu_measurements'] != 0):
        raise ValueError('candidate record drift')
    return record


def records_cursor(producer):
    """Mark a producer's trace before one call, without assuming reset/append."""
    records = getattr(producer, 'candidate_records', None)
    return (id(records), len(records)) if isinstance(records, (list, tuple)) else (None, 0)


def records_since(producer, cursor, *, outcome=None):
    """Detach only this call's receipts, including model/no-candidate failures.

    An explicit outcome contains one call's records. Otherwise the cursor
    handles either a freshly replaced list or an append-only producer. Existing
    fake/legacy producers with no trace contribute no invented provenance.
    """
    records = getattr(outcome, 'candidate_records', None)
    start = 0
    if not isinstance(records, (list, tuple)) or not records:
        records = getattr(producer, 'candidate_records', None)
        if not isinstance(records, (list, tuple)):
            return []
        if id(records) == cursor[0]:
            start = cursor[1]
    return deepcopy([record for record in records[start:]
                     if isinstance(record, dict) and record.get('schema') == RECORD_SCHEMA])
