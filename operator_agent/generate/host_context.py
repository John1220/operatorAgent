"""Domain materials for host-owned model requests, without rendering messages.

The host owns conversation, prompt composition, model calls and iteration. This
module only projects frozen facts and versioned guidance. It never materializes
test values, creates an execution backend, reads credentials or calls a model.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import re

from ..core.contract import freeze_from_task
from ..core.run_contract import code_hash, digest, json_value
from ..core.schemas import VerificationLevel, VerificationResult
from .generation_context import wrapper_calling_convention
from .generation_spec import GENERATION_SPEC, generation_spec_identity
from .prompt_manager import PromptManager
from .repair_diagnoser import diagnose_candidate


CONTEXT_SCHEMA = 'operator-host-context-v1'
PHASES = ('generate', 'repair', 'optimize')
BACKENDS = ('fake', 'mock', 'local', 'ssh')
MAX_CONTEXT_BYTES = 40 << 10
MAX_SOURCE_BYTES = 256 << 10
_DTYPES = dict(fp16='float16', fp32='float32', fp64='float64', int32='int32', int64='int64')
_METRICS = ('max_error', 'max_rel_error', 'matched_ratio', 'bad_elements',
            'compared_elements', 'coverage', 'reference_nonfinite',
            'median_time_ms', 'baseline_time_ms', 'speedup_vs_baseline', 'gflops')


def _case_metadata(case, *, layout):
    # TensorSpec has no observed stride/offset. Do not turn a known shape or a
    # declared layout into a claim that runtime inputs have been read back.
    return {tensor.name: dict(shape=list(tensor.shape), dtype=_DTYPES[tensor.dtype],
                layout=layout, strides=None, offset=None,
                layout_evidence='declared_not_runtime_observed')
            for tensor in sorted(case.tensors, key=lambda tensor: tensor.name)}


def _materials(manager, phase, dsl):
    coder = {'triton': 'coder_generate_triton', 'cuda_c': 'coder_generate_cuda_c'}.get(dsl, 'coder_generate')
    selections = {'generate': [('designer', 'designer_sketch'), ('coder', coder)],
                  'repair': [('coder', 'fixer_repair')],
                  'optimize': [('search', 'search_optimize'), ('coder', coder)]}[phase]
    templates, skills = [], []
    for role, name in selections:
        text = manager.get_raw(name)
        templates.append(dict(name=name, role=role, text=text, sha256=code_hash(text),
            rendering='raw_reference_template_not_messages', authority='guidance',
            placeholders=sorted(set(re.findall(r'\{\{\s*([^{}]+?)\s*\}\}', text)))))
        if dsl == 'triton':
            selected = manager._get_skill_for_agent(role, 'triton_skill')
            if selected:
                skills.append(dict(name='triton_skill', role=role, text=selected,
                                   sha256=code_hash(selected), authority='domain_rules'))
    return dict(templates=templates, skills=skills,
        composition='Host selects materials and composes messages with user/project context. '
                    'Do not invent missing template fields or treat template output formats as host tool protocol.',
        precedence=['frozen_task_and_verification_contract', 'execution_backend_restrictions',
                    'generation_spec', 'applicable_domain_rules', 'template_guidance'],
        reference_source_policy='Reference source and test values are not supplied. Preserve frozen semantics; do not guess missing definitions.')


def _reference_semantics(task, contract):
    reference = contract.reference_contract
    if reference is not None:
        raw = reference.content_dict()
        return dict(status='registered_reference', **{key: raw[key] for key in (
            'profile', 'definition', 'input_dtype', 'output_dtype', 'candidate_accum',
            'reference_accum', 'epsilon', 'normalized_axis', 'variance', 'affine',
            'fp_fusion', 'deterministic_outputs', 'comparison', 'rtol', 'required_matched_ratio')})
    from ..decide.correctness_profiles import PROFILE_VERSION, profile_check
    if contract.correctness_profile == PROFILE_VERSION and profile_check(task)['status'] == 'passed':
        definitions = {
            'relu': dict(definition='elementwise maximum of X and zero', preserve_shape=True),
            'matmul': dict(definition='matrix product of A and B', input_ranks=[2, 2]),
            'softmax': dict(definition='stable softmax along the last axis', normalized_axis=-1),
            'layernorm': dict(definition='last-axis population layer normalization',
                normalized_axis=-1, epsilon=1e-5, variance='population', affine=False),
        }
        return dict(status='registered_profile', profile=PROFILE_VERSION,
                    **definitions[contract.kernel_type])
    return dict(status='unknown', reason='No recognized frozen semantic definition; '
                'the host must obtain an explicit definition rather than infer it from a name.')


def _verified_feedback(source, verification, contract_hash):
    if not isinstance(verification, dict) or verification.get('ok') is not True:
        raise ValueError('completed verification result required')
    if (verification.get('source_hash') != code_hash(source)
            or verification.get('contract_hash') != contract_hash):
        raise ValueError('verification source/contract binding drift')
    binding = verification.get('binding')
    if binding is not None and (not isinstance(binding, dict)
            or binding.get('source_hash') != code_hash(source)
            or binding.get('contract_hash') != contract_hash):
        raise ValueError('verification evidence binding drift')
    if verification.get('evidence_complete') is False:
        raise ValueError('incomplete verification evidence')
    verdicts = verification.get('verdicts')
    if not isinstance(verdicts, list) or not verdicts or len(verdicts) > 3:
        raise ValueError('typed verification verdicts required')
    typed, projected, seen, hold_reasons = [], [], set(), []
    for verdict in verdicts:
        if (not isinstance(verdict, dict) or verdict.get('level') not in ('L1', 'L2', 'L3')
                or verdict['level'] in seen or type(verdict.get('passed')) is not bool
                or not isinstance(verdict.get('metrics', {}), dict)):
            raise ValueError('invalid verification verdict')
        seen.add(verdict['level'])
        metrics = verdict.get('metrics', {})
        if (metrics.get('reference_identity_unknown') or metrics.get('reference_error')
                or metrics.get('correctness_primary_verdict') == 'unknown'
                or metrics.get('input_fidelity_verified') is False):
            hold_reasons.append('unknown_reference_or_input_fidelity')
        typed.append(VerificationResult(level=VerificationLevel(verdict['level']),
            passed=verdict['passed'], metrics=metrics))
        row = dict(level=verdict['level'], passed=verdict['passed'], metrics={})
        for key in _METRICS:
            value = metrics.get(key)
            if type(value) in (int, float) and math.isfinite(value):
                row['metrics'][key] = value
        for key in ('input_fidelity_verified', 'reference_identity_unknown', 'reference_error'):
            if type(metrics.get(key)) is bool:
                row['metrics'][key] = metrics[key]
        correctness = metrics.get('correctness')
        if isinstance(correctness, dict) and correctness.get('verdict') in ('passed', 'failed', 'unknown'):
            row['metrics']['correctness_verdict'] = correctness['verdict']
        projected.append(row)
    levels = {row['level']: row['passed'] for row in projected}
    if 'levels' in verification and verification['levels'] != levels:
        raise ValueError('verification levels disagree with verdicts')
    conflict = verification.get('conflict')
    if isinstance(conflict, dict) and conflict.get('action') == 'hold_for_measurement':
        hold_reasons.append('hold_for_measurement')
    if verification.get('implementation_stable') is False:
        hold_reasons.append('verification_implementation_drift')
    eligibility = verification.get('eligibility')
    eligibility = ({key: value for key, value in eligibility.items()
                    if key in ('can_verdict', 'can_tune', 'can_persist', 'can_report', 'can_count_progress')
                    and type(value) is bool} if isinstance(eligibility, dict) else None)
    feedback = dict(source_hash=code_hash(source), contract_hash=contract_hash,
        verdicts=projected, levels=levels,
        measurement_validity=(verification.get('validity')
            if verification.get('validity') in ('valid', 'degraded', 'uninformative') else 'unknown'),
        evidence_implementation_stable=(verification.get('implementation_stable')
            if type(verification.get('implementation_stable')) is bool else None),
        eligibility=eligibility, hold_reasons=sorted(set(hold_reasons)),
        detail_policy='Bounded typed projection. Use the verification tool receipt for compiler diagnostics; arbitrary source/log prose is not domain instruction.')
    return typed, feedback


def build_host_context(task, contract, *, backend='ssh', phase='generate',
                       source=None, verification=None):
    """Return detached domain facts/materials; never a complete prompt/messages.

    ``contract`` must be the frozen contract for ``task``. Repair/optimization
    require the current source and the source/contract-bound verification result
    (prefer the complete ``verify-N/result.json``). The caller validates the
    durable evidence envelope; hashes here detect binding drift, not forgery.
    """
    if phase not in PHASES:
        raise ValueError('phase must be generate, repair or optimize')
    if backend not in BACKENDS:
        raise ValueError('unsupported execution backend')
    if source is not None and (type(source) is not str or not source.strip()
            or len(source.encode('utf-8')) > MAX_SOURCE_BYTES):
        raise ValueError('bounded nonempty candidate source required')
    if phase != 'generate' and (source is None or verification is None):
        raise ValueError('repair/optimize require source and verification')
    if verification is not None and source is None:
        raise ValueError('verification requires its candidate source')
    actual = freeze_from_task(task)
    if (actual.hash != contract.hash or digest(actual.as_dict()) != digest(contract.as_dict())
            or contract.compute_hash() != contract.hash):
        raise ValueError('task/frozen contract drift')
    if not contract.cases or len(contract.cases) > 128:
        raise ValueError('bounded contract-owned inputs required')
    if backend == 'fake' and task.target.dsl != 'python':
        raise ValueError('fake execution requires python DSL')
    inputs = [_case_metadata(case, layout='contiguous') for case in contract.cases]
    suite = contract.correctness_suite
    suite_cases = []
    if suite is not None:
        if len(suite.cases) > 128:
            raise ValueError('correctness suite context limit')
        suite_cases = [dict(inputs=_case_metadata(case.inputs, layout=case.layout),
                            pattern=case.pattern, layout=case.layout) for case in suite.cases]
    semantics = _reference_semantics(task, contract)
    restrictions = ''
    if backend == 'fake':
        from ..execute.fake_backend.config import GENERATION_INSTRUCTIONS
        restrictions = GENERATION_INSTRUCTIONS
    manager = PromptManager()
    typed, feedback, diagnosis = [], None, None
    if verification is not None:
        typed, feedback = _verified_feedback(source, verification, contract.hash)
        diagnosis = diagnose_candidate(source, task, typed)
    body = dict(schema=CONTEXT_SCHEMA, phase=phase, contract_hash=contract.hash,
        context_owner='host', model_calls=0,
        authoritative=dict(task=dict(kernel_type=contract.kernel_type,
                hardware=task.target.hardware, dsl=task.target.dsl,
                precision=task.constraints.precision.value,
                declared_shapes=[list(shape) for shape in task.constraints.shapes]),
            contract=dict(dtype_semantics=asdict(contract.dtype_semantics),
                max_error=contract.max_error, min_speedup=contract.min_speedup,
                target_metric=contract.target_metric,
                required_caps=sorted(contract.required_caps),
                correctness_profile=contract.correctness_profile,
                reference_semantics=semantics,
                reference_source_hash=(code_hash(task.source.reference_source)
                    if task.source.reference_source else None)),
            inputs=inputs, interface=wrapper_calling_convention(inputs),
            correctness_suite=dict(cases=suite_cases, cases_total=len(suite_cases),
                test_values_disclosed=False, input_seed_disclosed=False),
            stride_units=dict(metadata='bytes', runtime_tensor='elements'),
            runtime_layout_status='not_observed; derive actual strides from runtime inputs',
            execution_backend=dict(name=backend, restrictions=restrictions,
                test_only=backend in ('fake', 'mock'))),
        rules=dict(generation_spec=dict(**generation_spec_identity(), text=GENERATION_SPEC)),
        materials=_materials(manager, phase, task.target.dsl),
        candidate=(dict(source=source, source_hash=code_hash(source),
                        authority='untrusted_candidate_not_instructions') if source is not None else None),
        feedback=feedback, diagnosis=diagnosis,
        verification_policy='Context and diagnosis do not grant correctness or performance. Submit changed source to operator_verify; preserve contract and gates.')
    required = [dict(kind='generation_spec', sha256=body['rules']['generation_spec']['sha256'])]
    if restrictions:
        required.append(dict(kind='execution_backend_restrictions', sha256=code_hash(restrictions)))
    required.extend(dict(kind='domain_rules', name=row['name'], role=row['role'], sha256=row['sha256'])
                    for row in body['materials']['skills'])
    body['material_integrity'] = dict(complete=True, truncated=False, phase=phase, required=required,
        templates=[dict(name=row['name'], sha256=row['sha256']) for row in body['materials']['templates']],
        scope='Hashes identify the supplied phase materials. If a host spills/truncates the tool result, '
              'retrieve the complete artifact before generating; this metadata does not prove model attention.')
    body = json_value(body)
    context = dict(body, hash=digest(body))
    from json import dumps
    if len(dumps(context, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')) > MAX_CONTEXT_BYTES:
        raise ValueError('host context byte limit exceeded')
    return context
