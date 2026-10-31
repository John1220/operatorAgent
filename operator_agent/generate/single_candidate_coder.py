"""One model-authored Triton candidate, with immutable caller-owned context.

No Planner, samples, template substitution, normalization, repair or retry.
Local checks never import/execute candidate code. Passing them is not proof of
GPU correctness: the ordinary runtime, Verifier and admission gates still apply.
"""
from __future__ import annotations

import ast
import copy
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType

import numpy as np

from ..decide.anticheat import static_audit
from ..core.contract import freeze_from_task
from ..core.interface_contract import check_kernel_interface
from ..generate.generation_spec import GENERATION_SPEC
from ..generate.generation_context import (build_generation_context, generation_request,
    input_metadata, wrapper_calling_convention, make_candidate_record)
from ..execute.llm_budget import LLMCallRefused, strict_object, validated_usage
from ..execute.llm_client import LLMClient, parse_usage
from ..core.run_contract import atomic_json, code_hash, digest, json_value


PROMPT = ('Write exactly one complete Triton operator implementation. Return ONLY a strict '
          'JSON object with exactly code (unfenced Python source) and rationale (nonempty text). '
          'The caller-owned contract, reference, dtype semantics, workload, tolerances and '
          'interface are immutable. Input values are deliberately not provided; implement '
          'general semantics, not a shape/value lookup. No dependencies, shell/file/network '
          'operations, benchmark code, builtin operator fallback, eval, exec or dynamic imports. '
          'Only torch allocation and Triton computation are permitted. '
          'No retries or alternative candidates. '
          'A local precheck is not correctness or performance acceptance.\n\n' + GENERATION_SPEC)


# Diagnostics contain only these caller-owned literals, not model/exception text.
# Hashing this module versions the vocabulary together with the static gates.
PRECHECK_REASONS = MappingProxyType({
    'source_syntax_invalid': 'invalid source syntax',
    'kernel_entry_count': 'one explicit kernel entry required',
    'kernel_entry_decorated': 'kernel entry must be an undecorated Python wrapper',
    'kernel_entry_rebound': 'kernel entry must not be rebound',
    'jit_helper_missing': 'Triton jit implementation required',
    'kernel_entry_launch': 'launch the JIT helper, not the kernel entry',
    'undeclared_decorator': 'undeclared decorator/autotuning',
    'executable_default': 'executable function default',
    'executable_assignment': 'executable module assignment',
    'executable_statement': 'executable module statement',
    'undeclared_import': 'undeclared/aliased import',
    'undeclared_import_from': 'undeclared import-from',
    'dunder_access': 'dunder access refused',
    'torch_not_allocation': 'torch is allocation-only',
    'undeclared_triton_host_operation': 'undeclared Triton host operation/autotuning',
    'dynamic_host_operation': 'dynamic/host operation refused',
    'static_anticheat_refused': 'static anticheat gate refused candidate',
    'kernel_interface_refused': 'kernel interface refused candidate',
    'kernel_return_missing': 'kernel entry must explicitly return an output value',
    'repair_unchanged_source': 'repair must change the failed source',
    'repair_no_structural_change': 'repair must change more than formatting/comments',
})
FAILURE_REASONS = MappingProxyType({
    'model_call': frozenset({'model_call_failed'}),
    'envelope': frozenset({'invalid_code_envelope'}),
    'precheck': frozenset(PRECHECK_REASONS) | {'static_precheck_failed'},
    'context': frozenset({'generation_context_drift'}),
})


class PrecheckRefused(ValueError):
    """Fixed reason code; no arbitrary syntax, audit or exception diagnostics."""
    def __init__(self, reason):
        if type(reason) is not str or reason not in PRECHECK_REASONS:
            raise ValueError('unknown precheck reason')
        self.reason = reason
        super().__init__(PRECHECK_REASONS[reason])


def generation_failure(lineage):
    """Read a bounded failure receipt, never source or a positive credential."""
    if type(lineage) is not dict:
        raise ValueError('invalid generation failure receipt')
    receipt = lineage.get('failure')
    if (lineage.get('status') != 'failed' or type(receipt) is not dict
            or set(receipt) != {'stage', 'reason'}
            or type(receipt['stage']) is not str or type(receipt['reason']) is not str
            or receipt['stage'] not in FAILURE_REASONS
            or receipt['reason'] not in FAILURE_REASONS[receipt['stage']]
            or len(json.dumps(receipt, ensure_ascii=False, allow_nan=False).encode()) > 256):
        raise ValueError('invalid generation failure receipt')
    return dict(receipt)


@dataclass(frozen=True)
class SingleCandidatePolicy:
    max_output_tokens: int = 4096
    max_source_bytes: int = 32768
    max_rationale_chars: int = 1000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(name + ' must be a positive integer')


def _has_value_return(nodes):
    """Syntactic presence only; nested functions do not return from the wrapper."""
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if (isinstance(node, ast.Return) and node.value is not None
                and not (isinstance(node.value, ast.Constant) and node.value.value is None)):
            return True
        if _has_value_return(ast.iter_child_nodes(node)):
            return True
    return False


def _allocation_memory_formats(tree):
    """Admit format constants only in their allocation keyword position.

    These are values, not another set of permitted torch operations. Keeping
    the AST context explicit refuses aliases, calls and writes to the newly
    admitted constants; runtime shape/format compatibility is still verified.
    """
    formats = {'contiguous_format', 'channels_last', 'channels_last_3d', 'preserve_format'}
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    admitted = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == 'torch' and node.attr in formats):
            continue
        keyword_node = parents.get(node)
        call = parents.get(keyword_node)
        if not (isinstance(node.ctx, ast.Load)
                and isinstance(keyword_node, ast.keyword)
                and keyword_node.arg == 'memory_format' and keyword_node.value is node
                and isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == 'torch'
                and call.func.attr in {'empty', 'empty_like'}
                and (node.attr != 'preserve_format' or call.func.attr == 'empty_like')):
            raise PrecheckRefused('torch_not_allocation')
        admitted.add(node)
    if not admitted:
        return admitted
    # The direct spelling must still name the imported module. Do not try to
    # infer alias or rebinding semantics to justify a newly permitted value.
    if not any(isinstance(node, ast.Import) and any(
            name.name == 'torch' and name.asname is None for name in node.names)
            for node in tree.body):
        raise PrecheckRefused('torch_not_allocation')
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name) and root.id == 'torch':
                raise PrecheckRefused('torch_not_allocation')
        if isinstance(node, ast.Name) and node.id == 'torch':
            parent = parents.get(node)
            if not (isinstance(node.ctx, ast.Load) and isinstance(parent, ast.Attribute)
                    and parent.value is node and isinstance(parent.ctx, ast.Load)):
                raise PrecheckRefused('torch_not_allocation')
        if ((isinstance(node, ast.arg) and node.arg == 'torch')
                or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name == 'torch')
                or (isinstance(node, ast.ExceptHandler) and node.name == 'torch')
                or (isinstance(node, ast.ImportFrom) and any(
                    (name.asname or name.name) == 'torch' for name in node.names))):
            raise PrecheckRefused('torch_not_allocation')
    return admitted


def precheck(code, metadata):
    """Narrow this first mode to a static, allocation-only Triton wrapper.

    This is conservative pruning, NOT a hostile-code sandbox. Production
    resource/exec guards and real reference verification remain mandatory.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        raise PrecheckRefused('source_syntax_invalid') from None
    kernels = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel']
    if len(kernels) != 1:
        raise PrecheckRefused('kernel_entry_count')
    if kernels[0].decorator_list:
        raise PrecheckRefused('kernel_entry_decorated')
    if not any(isinstance(n, ast.FunctionDef) and any(
            isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name)
            and d.value.id == 'triton' and d.attr == 'jit' for d in n.decorator_list)
            for n in tree.body):
        raise PrecheckRefused('jit_helper_missing')
    allowed_imports = {'torch', 'triton', 'triton.language'}
    forbidden_calls = {'exec', 'eval', 'compile', 'open', '__import__', 'getattr', 'setattr',
                       'delattr', 'globals', 'locals', 'vars', 'input', 'breakpoint'}
    torch_attrs = {'empty', 'empty_like', 'float16', 'float32', 'float64', 'bfloat16',
                   'int8', 'int16', 'int32', 'int64', 'uint8', 'bool'}
    memory_formats = _allocation_memory_formats(tree)
    # Do not execute model-chosen calls/decorators while importing the module.
    # Triton jit is the one required, trusted decorator. No autotuning: that
    # would hide a new search campaign inside this single source candidate.
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.FunctionDef):
            if any(not (isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name)
                        and d.value.id == 'triton' and d.attr == 'jit') for d in node.decorator_list):
                raise PrecheckRefused('undeclared_decorator')
            if any(isinstance(n, ast.Call) for default in node.args.defaults +
                   [d for d in node.args.kw_defaults if d is not None] for n in ast.walk(default)):
                raise PrecheckRefused('executable_default')
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Assign):
            if any(isinstance(n, ast.Name) and n.id == 'kernel'
                   for target in node.targets for n in ast.walk(target)):
                raise PrecheckRefused('kernel_entry_rebound')
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                raise PrecheckRefused('executable_assignment') from None
            continue
        raise PrecheckRefused('executable_statement')
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(n.name not in allowed_imports or
                (n.asname and not (n.name == 'triton.language' and n.asname == 'tl')) for n in node.names):
            raise PrecheckRefused('undeclared_import')
        if isinstance(node, ast.ImportFrom) and (node.level or node.module != 'triton.language'
                or any(n.name == '*' for n in node.names)):
            raise PrecheckRefused('undeclared_import_from')
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('__'):
                raise PrecheckRefused('dunder_access')
            if (isinstance(node.value, ast.Name) and node.value.id == 'torch'
                    and node.attr not in torch_attrs and node not in memory_formats):
                raise PrecheckRefused('torch_not_allocation')
            if isinstance(node.value, ast.Name) and node.value.id == 'triton' and node.attr not in {'jit', 'cdiv', 'next_power_of_2'}:
                raise PrecheckRefused('undeclared_triton_host_operation')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
            raise PrecheckRefused('dynamic_host_operation')
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'kernel'):
            raise PrecheckRefused('kernel_entry_launch')
    audit = static_audit(code, 'triton')
    if audit.applicable is not True or audit.violations:
        raise PrecheckRefused('static_anticheat_refused')
    for row in metadata:
        interface = check_kernel_interface(code, row.keys(), 'triton')
        if interface.applicable is not True or interface.problem:
            raise PrecheckRefused('kernel_interface_refused')
    args = kernels[0].args
    expected = wrapper_calling_convention(metadata)
    actual_names = [arg.arg for arg in args.args + args.kwonlyargs]
    if (args.posonlyargs or args.vararg or args.kwarg
            or set(actual_names) != set(expected['parameter_names'])):
        raise PrecheckRefused('kernel_interface_refused')
    if not _has_value_return(kernels[0].body):
        raise PrecheckRefused('kernel_return_missing')
    try:
        compile(code, '<single-candidate-precheck>', 'exec')  # Syntax only; never exec.
    except SyntaxError:
        raise PrecheckRefused('source_syntax_invalid') from None


def generation_mode(context):
    """Normal generation stays byte-compatible; repair is explicitly declared."""
    if 'repair' not in context:
        if 'repair_diagnosis' in context:
            raise ValueError('diagnosis requires explicit repair context')
        return 'single-candidate-coder-v1', PROMPT, 'SingleCandidateCoder', 0
    from ..generate.bounded_repair import REPAIR_INSTRUCTION, validate_repair
    validate_repair(context['repair'], context['contract'], context['input_metadata'])
    if 'repair_diagnosis' in context:
        from ..generate.repair_diagnoser import DIAGNOSED_REPAIR_INSTRUCTION, validate_diagnosis
        diagnosis = validate_diagnosis(context['repair_diagnosis'], context['repair'],
                                       context['contract'], context['input_metadata'])
        if diagnosis['state'] != 'supported_hypothesis':
            raise ValueError('no supported repair diagnosis')
        return ('single-candidate-repair-v2', PROMPT + REPAIR_INSTRUCTION + DIAGNOSED_REPAIR_INSTRUCTION,
                'SingleCandidateRepair', 1)
    return 'single-candidate-repair-v1', PROMPT + REPAIR_INSTRUCTION, 'SingleCandidateRepair', 1


def generation_declaration(context, policy):
    schema, prompt, caller, repairs = generation_mode(context)
    return json_value(dict(schema=schema, context=context, context_hash=digest(context),
        policy=asdict(policy), prompt=prompt, prompt_hash=code_hash(prompt), max_candidates=1,
        repairs=repairs, planner=False, parallel_samples=False, template_fallback=False,
        mechanical_rewrite=False, source_author='model'))


def audit_generation(declaration, budget, lineage, code):
    """Read-only, model response → exact source proof; never restore eligibility."""
    context = declaration['context']
    schema, prompt, caller, repairs = generation_mode(context)
    if (declaration['schema'] != schema
            or declaration['context_hash'] != digest(context)
            or declaration['prompt'] != prompt or declaration['prompt_hash'] != code_hash(prompt)
            or declaration['source_author'] != 'model' or declaration['max_candidates'] != 1
            or type(declaration['repairs']) is not int or declaration['repairs'] != repairs or declaration['planner']
            or declaration['parallel_samples'] or declaration['template_fallback']
            or declaration['mechanical_rewrite']):
        raise ValueError('generation declaration inconsistent')
    if context.get('calling_convention') != wrapper_calling_convention(context['input_metadata']):
        raise ValueError('generation calling convention inconsistent')
    limits = budget['limits']
    if (budget['schema'] != 'llm-call-budget-v1' or budget['aborted'] is not None
            or budget['reserved_calls'] != 1 or len(budget['attempts']) != 1
            or limits['max_calls'] != 1 or list(limits['allowed_callers']) != [caller]
            or budget['retries'] or budget['fallback'] or budget['redirects']):
        raise ValueError('not a successful single-call campaign')
    call = budget['attempts'][0]
    request = call['request']
    body = json.dumps(request, ensure_ascii=False, allow_nan=False)
    if (call['call'] != 1 or call['status'] != 'answered' or call['caller'] != caller
            or call['request_sha256'] != code_hash(body) or call['request_bytes'] != len(body.encode())
            or len(body.encode()) > limits['max_request_bytes']
            or request['model'] != budget['identity']['model'] or request['temperature'] != .1
            or request['stream'] is not False
            or request['chat_template_kwargs'] != {'enable_thinking': budget['identity']['enable_thinking']}
            or request['max_tokens'] != declaration['policy']['max_output_tokens']
            or call['output_tokens_reserved'] != request['max_tokens']
            or budget['output_tokens_reserved'] != call['output_tokens_reserved']
            or not 0 < call['output_tokens_reserved'] <= min(limits['max_output_tokens'], limits['total_output_token_reservations'])
            or not 0 <= call['wall_seconds'] <= call['timeout_seconds'] <= limits['call_seconds']
            or request['messages'] != [{'role': 'system', 'content': prompt},
                {'role': 'user', 'content': json.dumps(context, ensure_ascii=False)}]):
        raise ValueError('model request lineage inconsistent')
    raw = call['response_text']
    if (code_hash(raw) != call['response_sha256'] or len(raw.encode()) != call['response_bytes']
            or len(raw.encode()) > limits['max_response_bytes']):
        raise ValueError('model response lineage inconsistent')
    response = strict_object(raw)
    choice = response['choices'][0]
    if len(response['choices']) != 1 or response['model'] != request['model'] or choice['finish_reason'] != 'stop':
        raise ValueError('model identity/finish reason inconsistent')
    model = strict_object(choice['message']['content'])
    usage, usage_source = validated_usage(response.get('usage'))
    norm = parse_usage(usage)
    if (call['usage_source'] != usage_source
            or norm['prompt_tokens'] < 0 or not 0 <= norm['completion_tokens'] <= request['max_tokens']):
        raise ValueError('model usage inconsistent')
    if (set(model) != {'code', 'rationale'} or model['code'] != code
            or lineage['code'] != code or lineage['candidate_hash'] != code_hash(code)
            or lineage['rationale'] != model['rationale'] or lineage['model_call'] != 1
            or lineage['context_hash'] != declaration['context_hash']
            or lineage['status'] != 'prechecked' or lineage['source_author'] != 'model'
            or 'failure' in lineage):
        raise ValueError('model source lineage inconsistent')
    policy = declaration['policy']
    if (type(model['rationale']) is not str or not model['rationale'].strip()
            or len(model['rationale']) > policy['max_rationale_chars']
            or type(code) is not str or not code.strip() or '```' in code
            or len(code.encode()) > policy['max_source_bytes']):
        raise ValueError('model source envelope inconsistent')
    precheck(code, context['input_metadata'])
    if repairs:
        from ..generate.bounded_repair import repair_lineage, source_change
        change = source_change(context['repair']['source'], code)
        if (lineage.get('repair') != repair_lineage(context['repair'])
                or change['change'] != 'changed_ast'
                or ('repair_change' in lineage and lineage['repair_change'] != change)):
            raise ValueError('repair source/parent lineage inconsistent')
        if 'repair_diagnosis' in context:
            if (lineage.get('repair_diagnosis_hash') != context['repair_diagnosis']['hash']
                    or lineage.get('repair_change') != change):
                raise ValueError('repair diagnosis lineage inconsistent')
        elif 'repair_diagnosis_hash' in lineage:
            raise ValueError('undeclared repair diagnosis lineage')
    elif 'repair' in lineage:
        raise ValueError('normal generation cannot carry repair lineage')
    return dict(completed=True, candidate_hash=code_hash(code), model_calls=1,
                model_authored_source=True, precheck_only=True, gpu_measurements=0,
                restored_eligibility=False)


class SingleCandidateCoder:
    last_finalists = ()  # No multi-finalist selection or extra L1/L2 dispatch.

    def __init__(self, llm, task, sketch, test_inputs, policy=None, *, repair=None, diagnosis=None):
        if not isinstance(llm, LLMClient) or llm.call_budget is None:
            raise ValueError('production bounded LLMClient required')
        self.llm = llm
        self.policy = policy or SingleCandidatePolicy()
        self._repair = copy.deepcopy(repair)
        self._diagnosis = copy.deepcopy(diagnosis)
        self._context = self._describe(task, sketch, test_inputs)
        _, prompt, caller, _ = generation_mode(self._context)
        limits = llm.call_budget.limits
        if (limits.max_calls != 1 or limits.allowed_callers != (caller,)
                or self.policy.max_output_tokens > limits.max_output_tokens
                or self.policy.max_output_tokens > limits.total_output_token_reservations
                or llm.call_budget.summary()['reserved_calls']):
            raise ValueError('fresh, single-caller, one-call campaign required')
        if task.target.dsl != 'triton' or sketch.target_dsl != 'triton' or task.source.code_path:
            raise ValueError('only Triton with caller-supplied reference text is supported')
        if any((llm.call_budget.root / name).exists() or (llm.call_budget.root / name).is_symlink()
               for name in ('generation-context.json', 'generation-declaration.json', 'candidate.json',
                            'candidate-record.json')):
            raise ValueError('generation already frozen; create a fresh campaign')
        self._sketch = copy.deepcopy(sketch)
        self._hash = digest(self._context)
        self._policy = asdict(self.policy)
        self._prompt_hash = code_hash(prompt)
        self._attempted = False
        self.last_generation = None
        self.candidate_records = []
        self.last_candidate_record = None
        atomic_json(llm.call_budget.root / 'generation-context.json', self._context, readonly=True)
        atomic_json(llm.call_budget.root / 'generation-declaration.json', self.declaration(), readonly=True)

    def _describe(self, task, sketch, inputs):
        if task.target.dsl != 'triton' or sketch.target_dsl != 'triton' or task.source.code_path:
            raise ValueError('frozen Triton/reference declaration drift')
        shared = self._shared_context(task, sketch, inputs)
        metadata = shared['input_metadata']
        # Preserve the existing one-call disclosure and audit format exactly.
        context = json_value(dict(task_id=task.task_id, hardware=shared['task']['hardware'],
            dsl=shared['task']['dsl'], contract=shared['contract'],
            reference_source=task.source.reference_source, sketch=shared['sketch'],
            input_metadata=metadata, calling_convention=shared['interface']['wrapper']))
        if self._repair is not None:
            context['repair'] = copy.deepcopy(self._repair)
            precheck(context['repair']['source'], metadata)
        if self._diagnosis is not None:
            context['repair_diagnosis'] = copy.deepcopy(self._diagnosis)
        generation_mode(context)
        return context

    def _shared_context(self, task, sketch, inputs):
        return build_generation_context(task, sketch, inputs,
            producer='SingleCandidateCoder',
            strategy='bounded-repair' if self._repair is not None else 'bounded-single',
            policy=asdict(self.policy), strict=True,
            parent_source=self._repair['source'] if self._repair is not None else None)

    def declaration(self):
        return generation_declaration(self._context, self.policy)

    def _check_frozen(self):
        disk = json.loads((self.llm.call_budget.root / 'generation-context.json').read_text())
        _, prompt, _, _ = generation_mode(self._context)
        if (disk != self._context or digest(self._context) != self._hash
                or asdict(self.policy) != self._policy or code_hash(prompt) != self._prompt_hash
                or self._context.get('calling_convention') != wrapper_calling_convention(
                    self._context['input_metadata'])):
            raise ValueError('frozen generation declaration drift')
        disk_declaration = json.loads((self.llm.call_budget.root / 'generation-declaration.json').read_text())
        if disk_declaration != self.declaration():
            raise ValueError('frozen generation declaration drift')

    def design(self, task, hardware_info):
        # A fixed caller-owned sketch, not an extra model proposal. Recheck
        # task/contract at run() too, using the actual inputs at dispatch time.
        self._check_frozen()
        if (task.task_id != self._context['task_id']
                or json_value(freeze_from_task(task).as_dict()) != self._context['contract']):
            raise ValueError('frozen generation contract drift')
        return copy.deepcopy(self._sketch)

    def validate(self, task, inputs):
        self._check_frozen()
        if self._describe(task, self._sketch, inputs) != self._context:
            raise ValueError('frozen generation context drift before execution')

    def check_candidate(self, code):
        self._check_frozen()
        return audit_generation(self.declaration(), self.llm.call_budget.summary(),
                                self.last_generation, code)

    def run(self, sketch, task, optimization_recipes=None, test_inputs=None, **kwargs):
        self._check_frozen()
        if self._describe(task, sketch, test_inputs) != self._context:
            raise ValueError('frozen generation context drift before model dispatch')
        if self._attempted:
            raise LLMCallRefused('single candidate already attempted; no retry')
        self._attempted = True
        _, prompt, caller, repairs = generation_mode(self._context)
        messages = [{'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': json.dumps(self._context, ensure_ascii=False)}]
        shared = self._shared_context(task, sketch, test_inputs)
        request = generation_request(self.llm, messages, caller=caller,
                                     temperature=0.1, max_tokens=self.policy.max_output_tokens)
        outcome = dict(schema='model-code-lineage-v1', context_hash=self._hash,
                       source_author='model', status='failed')
        if repairs:
            from ..generate.bounded_repair import repair_lineage
            outcome['repair'] = repair_lineage(self._context['repair'])
            if 'repair_diagnosis' in self._context:
                outcome['repair_diagnosis_hash'] = self._context['repair_diagnosis']['hash']
        stage, reason = 'model_call', 'model_call_failed'
        try:
            result = self.llm.chat_json(messages, temperature=0.1,
                max_tokens=self.policy.max_output_tokens, caller_tag=caller)
            stage, reason = 'envelope', 'invalid_code_envelope'
            if (set(result) != {'code', 'rationale'} or type(result['code']) is not str
                    or not result['code'].strip() or '```' in result['code']
                    or len(result['code'].encode()) > self.policy.max_source_bytes
                    or type(result['rationale']) is not str or not result['rationale'].strip()
                    or len(result['rationale']) > self.policy.max_rationale_chars):
                raise ValueError('invalid single-candidate code envelope')
            code = result['code']  # EXACT decoded model source; never fix/normalize.
            outcome.update(candidate_hash=code_hash(code), code=code, rationale=result['rationale'])
            stage, reason = 'precheck', 'static_precheck_failed'
            precheck(code, self._context['input_metadata'])
            if repairs:
                from ..generate.bounded_repair import source_change
                change = source_change(self._context['repair']['source'], code)
                outcome['repair_change'] = change
                if change['change'] == 'identical_source':
                    raise PrecheckRefused('repair_unchanged_source')
                if change['change'] == 'same_ast':
                    raise PrecheckRefused('repair_no_structural_change')
            # The model does not own task/sketch: check them after the call, too.
            stage, reason = 'context', 'generation_context_drift'
            self._check_frozen()
            if self._describe(task, sketch, test_inputs) != self._context:
                raise ValueError('generation context drift after model response')
            outcome.update(status='prechecked', model_call=1)
            return code
        except Exception as exc:
            if (stage == 'precheck' and type(exc) is PrecheckRefused
                    and type(exc.reason) is str and exc.reason in PRECHECK_REASONS):
                reason = exc.reason
            outcome['failure'] = generation_failure(dict(status='failed', failure=dict(stage=stage, reason=reason)))
            self.llm.call_budget.abort('candidate_generation_failed')
            raise LLMCallRefused('single candidate failed; no repair/retry/template') from None
        finally:
            self.last_generation = copy.deepcopy(outcome)
            atomic_json(self.llm.call_budget.root / 'candidate.json', outcome, readonly=True)
            status = ('prechecked' if outcome['status'] == 'prechecked' else
                      'precheck_failed' if stage == 'precheck' else
                      'model_failed' if stage == 'model_call' else 'generation_failed')
            record = make_candidate_record(shared, outcome.get('code'), request=request,
                status=status, reason=outcome.get('failure'))
            self.candidate_records.append(record)
            self.last_candidate_record = record
            atomic_json(self.llm.call_budget.root / 'candidate-record.json', record, readonly=True)
