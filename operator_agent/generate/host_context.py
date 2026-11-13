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
from .optimize_diagnoser import diagnose_optimization
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

# ── 性能事实的有界投影 (optimize 阶段的输入) ──
# 这三张白名单分开写的原因: 标量/布尔/文本的**三态**语义不同 —— 缺键 = 没观测,
# 不许补默认值 (`_METRICS` 同一条纪律)。布尔只认 `type(...) is bool`,
# 免得 `isinstance(True, int)` 把一个 True 当成 1 搬进标量位。
_PERF_SCALARS = ('speedup_vs_baseline', 'baseline_time_ms', 'median_time_ms', 'gflops',
                 'gflops_utilization_pct', 'bandwidth_utilization_pct',
                 'arithmetic_intensity', 'ridge_point',
                 'launch_dominance', 'launch_floor_us',
                 'dispersion_iqr_over_median', 'dispersion_max_over_median')
_PERF_BOOLS = ('measurement_can_gate', 'measurement_can_disable_recipe',
               'measurement_can_count_stagnation', 'dispersion_exceeded',
               'benchmark_contaminated')
_PERF_ROOFLINE_TEXT = ('bound_type', 'compute_used', 'compute_path_evidence', 'memory_used', 'hw_name')
_PERF_ROOFLINE_BOOLS = ('reference_lower_bound', 'peak_trusted')
_PERF_ROOFLINE_SCALARS = ('hw_peak_gflops', 'hw_peak_bandwidth_gbs')
_MAX_PERF_REASONS = 8
_PERFORMANCE_POLICY = (
    'Performance facts are measurements, not conclusions. An absent key means it was not observed; '
    'do not substitute a default. `reference_ceiling_reached=False` is fail-closed "not proven at the '
    'reference ceiling", not proof of headroom. A missing `improvement_band` means that validity tier '
    'has no usable band, so no improvement comparison is supported. `can_*` flags state what the number '
    'may be used for; they do not extend it. Changed source must still be submitted to the verification tool.')


def _performance_projection(level, metrics, fallback_validity=None):
    """Pull L3 performance facts into a bounded dict, or return None when there are none.

    This was a capability *regression* against the legacy conductor, which copied
    ``roofline_analysis`` wholesale: the host only saw eleven flat scalars, so the
    optimize phase could not tell a memory-bound kernel from a launch-floor-dominated
    one and had to guess where to aim. Only facts and **already existing** derived
    quantities are carried here (``binding_attainment`` / ``noise_band_for`` /
    ``reference_ceiling_reached``); no new judgement is computed in this module.
    """
    perf = {}
    for key in _PERF_SCALARS:
        value = metrics.get(key)
        if type(value) in (int, float) and math.isfinite(value):
            perf[key] = value
    for key in _PERF_BOOLS:
        if type(metrics.get(key)) is bool:
            perf[key] = metrics[key]
    reasons = metrics.get('measurement_validity_reasons')
    if isinstance(reasons, list):
        texts = [row for row in reasons if type(row) is str]
        if texts:
            perf['measurement_validity_reasons'] = texts[:_MAX_PERF_REASONS]
    roofline = metrics.get('roofline_analysis')
    if isinstance(roofline, dict):
        facts = {}
        for key in _PERF_ROOFLINE_TEXT:
            if type(roofline.get(key)) is str:
                facts[key] = roofline[key]
        for key in _PERF_ROOFLINE_BOOLS:
            if type(roofline.get(key)) is bool:
                facts[key] = roofline[key]
        for key in _PERF_ROOFLINE_SCALARS:
            value = roofline.get(key)
            if type(value) in (int, float) and math.isfinite(value):
                facts[key] = value
        if facts.get('compute_used') == 'tf32_tensor_core' and facts.get('compute_path_evidence') != 'triton_ptx':
            facts.update(compute_used='unknown', bound_type='unknown',
                         reference_lower_bound=False, peak_trusted=False)
            for key in ('gflops_utilization_pct', 'ridge_point', 'hw_peak_gflops'):
                facts.pop(key, None)
                perf.pop(key, None)
        if facts:
            perf['roofline'] = facts
    if not perf:
        return None
    from ..decide.measurement import (Validity, binding_attainment, noise_band_for,
                                      reference_ceiling_reached)
    perf['level'] = level
    compute_unknown = perf.get('roofline', {}).get('compute_used') == 'unknown'
    attainment = None if compute_unknown else binding_attainment(metrics)
    if attainment is not None:
        perf['binding_attainment_pct'] = round(attainment, 4)
    # 档位可能只出现在信封 (`verification['validity']`) 而没写进这一级 metrics ——
    # 老回执就是这样。两处是同一个观测, 所以先 metrics 后信封; 都没有就缺席,
    # 不猜一个档位出来 (`can_*` 也不在这里补: 那两个是判据的产物, 只能原样转述)。
    validity = metrics.get('measurement_validity')
    if validity not in ('valid', 'degraded', 'uninformative'):
        validity = fallback_validity if fallback_validity in ('valid', 'degraded', 'uninformative') else None
    if validity is not None:
        perf['measurement_validity'] = validity
        band = noise_band_for(Validity(validity))
        # `uninformative` 档没有带宽: 那一轮没有任何判决用过它, 所以键缺席而不是写 0
        # (与 `noise_band_for` 的返回语义一致 —— 别把没用上的带宽记成用过的)。
        if band is not None:
            perf['improvement_band'] = band
    perf['reference_ceiling_reached'] = not compute_unknown and reference_ceiling_reached(metrics)
    return perf


def anticheat_policy():
    """规则正文只有一份（`decide/anticheat.ANTICHEAT_POLICY`），这里只做引用。"""
    from ..decide.anticheat import ANTICHEAT_POLICY
    return ANTICHEAT_POLICY


def _suite_intent(contract):
    """把"哪些标量取值会被真的测"说成一句话，而不是让宿主从 case 列表里自己猜。

    - 被测集合在 `correctness_suite.cases` 里（8 条 case 各自的 `inputs.scalars`）；
      `contract.cases` 只有**主用例**，只看它会得出"causal 只测 false"这个错误结论；
    - 对象形态（`CorrectnessSuite` / `CorrectnessCase`）与 JSON 形态都要认。
    """
    primary = list(getattr(contract, 'cases', None) or [])
    declared = dict(getattr(primary[0], 'scalars', None) or {}) if primary else {}
    suite = getattr(contract, 'correctness_suite', None)
    rows = list(getattr(suite, 'cases', None) or []) if suite is not None else []
    tested = {}
    for row in rows:
        scalars = getattr(getattr(row, 'inputs', None), 'scalars', None)
        if scalars is None and isinstance(row, dict):
            scalars = (row.get('inputs') or {}).get('scalars') or {}
        for name, value in (scalars or {}).items():
            tested.setdefault(name, set()).add(value)
    layouts = sorted({getattr(row, 'layout', None) for row in rows} - {None}) if rows else []
    if not tested and not declared:
        # 没有标量不等于没有别的维度会变：softmax 的 12 条 case 里就有一条 strided。
        note = '没有 case 级标量；形状由运行时张量决定'
        if layouts:
            note += ('；被测布局: ' + ', '.join(layouts)
                     + '。非连续布局会被真的构造出来 ⇒ 运行时 stride 必须逐个读，'
                       '不能假设连续')
        return dict(declared_scalars={}, tested_scalar_values={},
                    tested_layouts=layouts, note=note)
    tested = {name: sorted(values, key=lambda v: (isinstance(v, str), v))
              for name, values in sorted(tested.items())}
    note = ('每一个 case 的标量取值都会被真的测到；声明里的取值只是主用例的那一个，'
            '不是被测集合的全部')
    if layouts:
        note += ('；被测布局: ' + ', '.join(layouts)
                 + '。非连续布局会被真的构造出来 ⇒ 运行时 stride 必须逐个读，'
                   '不能假设连续（2026-11-09 softmax 两臂都挂在这上面）')
    return dict(declared_scalars=declared, tested_scalar_values=tested,
                tested_layouts=layouts, note=note)


def _case_metadata(case, *, layout):
    # TensorSpec has no observed stride/offset. Do not turn a known shape or a
    # declared layout into a claim that runtime inputs have been read back.
    rows = {tensor.name: dict(shape=list(tensor.shape), dtype=_DTYPES[tensor.dtype],
                 layout=layout, strides=None, offset=None,
                 layout_evidence='declared_not_runtime_observed')
            for tensor in sorted(case.tensors, key=lambda tensor: tensor.name)}
    # 按值直传的标量（causal / eps / scale）也进 ABI：candidate 的 wrapper 必须收它们，
    # 而它们**不是**从 seed 抽样出来的，所以单独标出出处，别和"声明了但没回读"混为一谈。
    for name in sorted(getattr(case, 'scalars', None) or {}):
        value = case.scalars[name]
        rows[name] = dict(kind='scalar_by_value', py_type=type(value).__name__, value=value,
                          layout='inline', strides=None, offset=None,
                          evidence='frozen_contract_passed_by_value_not_sampled')
    return rows


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
    performance, performance_rank = None, -1
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
        # 性能事实只取**最高一级**带读数的那份 (通常是 L3), 不按列表顺序赌最后一项 ——
        # verdicts 的排列不受校验, 「后到的覆盖」会把 L2 的读数当成终局。
        rank = {'L1': 0, 'L2': 1, 'L3': 2}[verdict['level']]
        candidate = _performance_projection(verdict['level'], metrics,
                                                     verification.get('validity'))
        if candidate is not None and rank > performance_rank:
            performance, performance_rank = candidate, rank
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
        detail_policy='Bounded typed projection. Use the verification tool receipt for compiler diagnostics; arbitrary source/log prose is not domain instruction.',
        # 只有在真的带性能读数时才出现: 生成阶段的上下文形状不因这次扩展而变
        # (字节上限、context_hash 与既有黄金回执都不跟着抖)。
        **({'performance': performance,
            'performance_policy': _PERFORMANCE_POLICY} if performance is not None else {}))
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
            # 声明值 ≠ 被测值：attention 声明 `mask=nocausal`，但 8 条 suite 里有 7 条是
            # `causal=true`。只给 case 列表（数据）而不说这句话，宿主很容易只实现声明的那条路径。
            # 2026-11-09 C 臂就撞在这里（`s1024-causal` 等 7 条全挂）。
            suite_intent=_suite_intent(contract),
            correctness_suite=dict(cases=suite_cases, cases_total=len(suite_cases),
                test_values_disclosed=False, input_seed_disclosed=False),
            stride_units=dict(metadata='bytes', runtime_tensor='elements'),
            runtime_layout_status='not_observed; derive actual strides from runtime inputs',
            execution_backend=dict(name=backend, restrictions=restrictions,
                test_only=backend in ('fake', 'mock'))),
        rules=dict(
            # 「哪些事不能做」必须显式发布：2026-11-09 裸臂写了正确的 softmax（主用例
            # 1.91e-06）却直接调 torch.softmax，被判反作弊 —— 而两臂都没被告知这条规则。
            anti_cheat=dict(policy=anticheat_policy(),
                            enforcement='static_audit + runtime guard',
                            consequence='L2 rejected: 反作弊'),
            generation_spec=dict(**generation_spec_identity(), text=GENERATION_SPEC)),
        materials=_materials(manager, phase, task.target.dsl),
        candidate=(dict(source=source, source_hash=code_hash(source),
                        authority='untrusted_candidate_not_instructions') if source is not None else None),
        feedback=feedback, diagnosis=diagnosis,
        verification_policy='Context and diagnosis do not grant correctness or performance. Submit changed source to operator_verify; preserve contract and gates.')
    if phase == 'optimize':
        # 只在 optimize 出现: 生成/修复阶段的上下文形状与字节数不跟着变。
        body['optimization'] = diagnose_optimization(feedback)
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
    from json import dumps

    def _size(candidate):
        return len(dumps(candidate, ensure_ascii=False, separators=(',', ':'),
                         allow_nan=False).encode('utf-8'))

    # 字节上限是**溢出保护**（spill 会让宿主只看到半份材料），不是判据。超限时先
    # 去掉最不承重的一块并**如实登记**，而不是把上限调大或悄悄截断：
    # 角色模板的正文是给内层模型看的历史材料（`rendering=raw_reference_template_not_messages`，
    # 宿主本来就自己组装消息），名字/角色/sha256 全部保留 —— 需要原文的人仍能按哈希取回。
    # 2026-11-08：attention 在 S=4096 的 optimize 体是 41.3 KB，只超 334 字节，
    # 却足以让 arm C 完全拿不到上下文（`host context byte limit exceeded`）。
    if _size(dict(body, hash=digest(body))) > MAX_CONTEXT_BYTES:
        dropped = []
        for template in (body.get('materials') or {}).get('templates') or ():
            if isinstance(template, dict) and 'text' in template:
                template.pop('text', None)
                dropped.append('materials.templates[].text')
        if dropped:
            integrity = body.setdefault('material_integrity', {})
            integrity['truncated'] = True
            integrity['dropped'] = sorted(set(dropped))
            integrity['truncation'] = (
                'template texts omitted to stay inside the context byte budget; '
                'their sha256 entries still identify the full artifacts')
        context = dict(body, hash=digest(body))
        if _size(context) > MAX_CONTEXT_BYTES:
            raise ValueError('host context byte limit exceeded')
        return context
    context = dict(body, hash=digest(body))
    if _size(context) > MAX_CONTEXT_BYTES:
        raise ValueError('host context byte limit exceeded')
    return context
