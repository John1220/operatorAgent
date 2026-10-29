"""Frozen, budgeted quality experiments; planning never calls a model or device.

Replay is an integration check against CPU execution, not evidence of model or
GPU quality. Real runs require an explicit frozen plan and SSH target. All task
attempts, including generation errors and missing results, remain in the report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import contextlib
import io
import json
import math
import os
from pathlib import Path
import time

from . import quality_suite
from .contract import freeze_from_task
from .fixer import FixerAgent
from .generation_spec import GENERATION_SPEC, generation_spec_identity
from .llm_budget import LLMCallBudget, LLMCallLimits, LLMCallRefused, strict_object
from .llm_client import CostTracker, LLMClient
from .repair_loop import _full_l2_passed, repair
from .run_contract import atomic_json, audit_value, code_hash, digest, protocol_identity
from .schemas import VerificationLevel
from .single_candidate_coder import input_metadata, precheck, wrapper_calling_convention
from .verifier import Verifier


@dataclass(frozen=True)
class QualityBudget:
    """Per-task call/candidate limits; no exact billing or GPU-time guarantee."""

    max_calls: int = 2
    max_output_tokens: int = 4096
    max_verifications: int = 2
    max_request_bytes: int = 65536
    max_response_bytes: int = 131072
    call_seconds: float = 60
    campaign_seconds: float = 600

    def __post_init__(self):
        for name in ('max_calls', 'max_output_tokens', 'max_verifications',
                     'max_request_bytes', 'max_response_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('quality budget requires positive integer ' + name)
        for name in ('call_seconds', 'campaign_seconds'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('quality budget requires finite positive ' + name)


def evaluator_identity():
    """Version the experiment, shared guidance and production verification code."""
    root = Path(__file__).parent
    extra = {name: code_hash((root / name).read_text()) for name in
             ('quality_evaluation.py', 'quality_suite.py', 'quality_pairing.py', 'quality_portfolio.py',
              'quality_preflight.py', 'run_layout.py', 'generation_spec.py',
              'fixer.py', 'repair_loop.py', 'prompt_manager.py')}
    return digest(dict(production=protocol_identity(), experiment=extra))


def make_plan(*, model='offline-replay', endpoint='', backend='mock', host='',
              strategy='diagnosis', case_ids=None, budget=None):
    """Return a serializable plan containing no credentials or concrete inputs."""
    if backend not in ('mock', 'ssh') or strategy not in ('baseline', 'diagnosis'):
        raise ValueError('unsupported quality backend/strategy')
    if backend == 'ssh' and not host:
        raise ValueError('SSH quality plan needs an explicit host alias')
    if not isinstance(model, str) or not model:
        raise ValueError('quality plan needs a model name')
    limits = budget or QualityBudget()
    if not isinstance(limits, QualityBudget):
        raise ValueError('quality budget required')
    controls = dict(model=dict(name=model, endpoint_sha256=digest(endpoint),
                               thinking_parameter=os.environ.get('LLM_THINKING_PARAMETER', 'chat_template_kwargs'),
                               enable_thinking=False),
                    strategy=strategy, budget=asdict(limits), backend=backend,
                    hardware=dict(host_sha256=digest(host), real_device_required=backend == 'ssh'),
                    evaluator=evaluator_identity(), generation_spec=generation_spec_identity(),
                    scope='full_L2_only_no_performance_or_deployment_claim')
    body = dict(schema='operator-quality-plan-v1',
                manifest=quality_suite.suite_manifest(case_ids), controls=controls,
                disclosures=dict(model_payload='task semantics, input metadata, failed candidate and diagnostic feedback',
                    candidate_execution='CPU subprocess for replay; remote same-UID runner for SSH',
                    exact_provider_bill_cap=False, hard_GPU_time_cap=False,
                    independent_device_isolation=False))
    return dict(body, hash=digest(body))


def validate_plan(plan):
    """Reject drift before creating a client, backend, or run directory."""
    if not isinstance(plan, dict) or plan.get('schema') != 'operator-quality-plan-v1':
        raise ValueError('invalid quality plan')
    body = {key: value for key, value in plan.items() if key != 'hash'}
    if digest(body) != plan.get('hash'):
        raise ValueError('quality plan hash drift')
    controls = plan['controls']
    expected_controls = {'model', 'strategy', 'budget', 'backend', 'hardware',
                         'evaluator', 'generation_spec', 'scope'}
    if (set(plan) != {'schema', 'manifest', 'controls', 'disclosures', 'hash'}
            or not isinstance(controls, dict) or set(controls) != expected_controls):
        raise ValueError('quality plan control drift')
    if controls.get('strategy') not in ('baseline', 'diagnosis') or controls.get('backend') not in ('mock', 'ssh'):
        raise ValueError('quality plan control drift')
    QualityBudget(**controls['budget'])
    model, hardware = controls['model'], controls['hardware']
    if (not isinstance(model, dict)
            or set(model) != {'name', 'endpoint_sha256', 'thinking_parameter', 'enable_thinking'}
            or not isinstance(model['name'], str) or not model['name']
            or model['thinking_parameter'] not in ('chat_template_kwargs', 'enable_thinking')
            or model['enable_thinking'] is not False
            or not isinstance(hardware, dict)
            or set(hardware) != {'host_sha256', 'real_device_required'}
            or hardware['real_device_required'] is not (controls['backend'] == 'ssh')
            or controls['scope'] != 'full_L2_only_no_performance_or_deployment_claim'):
        raise ValueError('quality plan control drift')
    ids = [row['case_id'] for row in plan['manifest']['cases']]
    if plan['manifest'] != quality_suite.suite_manifest(ids):
        raise ValueError('quality task manifest drift')
    if (controls['evaluator'] != evaluator_identity()
            or controls['generation_spec'] != generation_spec_identity()):
        raise ValueError('quality evaluator/source drift; freeze a new plan')
    return plan


def generation_request(case_id, task):
    """Project semantic inputs while withholding acceptance seeds and values."""
    inputs = quality_suite.inputs_for(case_id)
    metadata = input_metadata([inputs])
    contract = freeze_from_task(task)
    return dict(case_id=case_id, kernel_type=task.source.kernel_type,
                dsl=task.target.dsl, precision=task.constraints.precision.value,
                contract_hash=contract.hash,
                dtype_semantics=asdict(contract.dtype_semantics),
                inputs=metadata, calling_convention=wrapper_calling_convention(metadata),
                semantics=dict(reduction_axis=-1, layernorm_epsilon=1e-5,
                               layernorm_affine=False, layernorm_variance='population'),
                acceptance='All declared layouts and finite value patterns; independent seeds are withheld.')


def generation_messages(case_id, task):
    instruction = ('Return ONLY a strict JSON object with exactly code and rationale. '
                   'code is complete unfenced source; rationale is nonempty text. '
                   'Use Triton computation for triton tasks; NumPy for python replay tasks. '
                   'No benchmarks, filesystem/network access, fallback library operators, '
                   'autotuning, retries, or extra candidates. Do not alter the frozen reference or gates.')
    return [dict(role='system', content=GENERATION_SPEC + '\n' + instruction),
            dict(role='user', content=json.dumps(generation_request(case_id, task), ensure_ascii=False))]


class ReplayClient:
    """Consume explicitly supplied model messages without network access."""

    def __init__(self, replies, budget):
        self.replies = iter(replies)
        self.call_budget = budget
        self.cost_tracker = CostTracker()
        self.model = 'offline-replay'
        self.identity = dict(model=self.model, execution='replay')
        budget.bind(self.identity)

    def chat(self, messages, temperature=None, max_tokens=None, raw=False, caller_tag='Unknown'):
        body = json.dumps(dict(messages=messages, temperature=temperature,
                               max_tokens=max_tokens), ensure_ascii=False)
        number, _ = self.call_budget.reserve(self.identity, body, caller_tag, 60)
        try:
            reply = next(self.replies)
            if not isinstance(reply, str) or len(reply.encode()) > self.call_budget.limits.max_response_bytes:
                raise ValueError('invalid replay response')
        except (StopIteration, ValueError):
            self.call_budget.finish(number, outcome='replay_response_missing')
            raise LLMCallRefused('replay response missing; no implicit requests') from None
        self.call_budget.finish(number, response=reply, outcome='answered', usage_source='replay_fixture')
        self.cost_tracker.record(model=self.model, reported_model=self.model,
            prompt_tokens=0, completion_tokens=0, usage_source='replay_fixture', caller_tag=caller_tag,
            temperature=temperature, max_tokens=max_tokens, enable_thinking=False)
        return reply


def _usage(client):
    summary = client.cost_tracker.summary()
    calls = client.call_budget.summary()['reserved_calls']
    # A timeout may have reached the provider without returning usage. Keep its
    # call reservation, but do not turn missing billing evidence into zero tokens.
    return calls, summary['total_tokens'], summary['total_calls'] == calls


def _usage_sources(client):
    return sorted({row.usage_source for row in client.cost_tracker.records})


class RecordingVerifier:
    """Keep one record per actual candidate; verify all independent inputs at L2."""

    def __init__(self, verifier, client, task_root, case_id, budget, execution_kind, started=None):
        self.verifier, self.client = verifier, client
        self.root, self.case_id = task_root, case_id
        self.budget, self.execution_kind = budget, execution_kind
        self.records = []
        self.last_calls = self.last_tokens = 0
        self.last_time = time.monotonic() if started is None else started
        self.last_results = []

    def verify(self, code, *args, **kwargs):
        if len(self.records) >= self.budget.max_verifications:
            raise LLMCallRefused('quality verification budget exhausted')
        number = len(self.records) + 1
        (self.root / ('candidate-%d.py' % number)).write_text(code)
        results = self.verifier.verify(code, *args, **kwargs)
        self.last_results = results
        full = _full_l2_passed(results)
        calls, tokens, usage_complete = _usage(self.client)
        now = time.monotonic()
        record = dict(case_id=self.case_id, attempt=number,
            status='passed' if full is True else ('unknown' if full is None else 'failed'),
            full_l2_passed=full, performance_target_passed=None, measurement_validity=None,
            execution_kind=self.execution_kind, model_calls=calls-self.last_calls,
            tokens=tokens-self.last_tokens if usage_complete else None,
            token_usage_sources=_usage_sources(self.client), wall_seconds=now-self.last_time,
            source_hash=code_hash(code),
            levels={r.level.value: bool(r.passed) for r in results},
            verification_scope='primary_plus_independent_seeds_and_full_profile')
        self.records.append(record)
        self.last_calls, self.last_tokens, self.last_time = calls, tokens, now
        atomic_json(self.root / ('verification-%d.json' % number),
                    audit_value(dict(record=record, results=[dict(level=r.level.value, passed=r.passed,
                         metrics=r.metrics, message=r.message) for r in results])))
        return results


def _validate_execution(plan, replay, host, *, check_model=True):
    validate_plan(plan)
    controls = plan['controls']
    if replay:
        if controls['backend'] != 'mock':
            raise ValueError('replay requires a mock plan')
    else:
        if controls['backend'] != 'ssh' or digest(host) != controls['hardware']['host_sha256']:
            raise ValueError('real execution requires the frozen SSH target')
        if check_model and (os.environ.get('LLM_MODEL') != controls['model']['name']
                or digest(os.environ.get('LLM_BASE_URL', '')) != controls['model']['endpoint_sha256']
                or os.environ.get('LLM_THINKING_PARAMETER', 'chat_template_kwargs') != controls['model']['thinking_parameter']):
            raise ValueError('model configuration differs from the frozen plan')


def _make_backend(case_id, replay, host, factory):
    if factory is not None:
        return factory(case_id)
    from .hitl_tools import _backend
    return _backend('mock' if replay else 'ssh', host=host or None)


def _preflight(backend, root):
    from .quality_preflight import preflight_backend
    result = preflight_backend(backend)
    atomic_json(root / 'preflight.json', audit_value(result), readonly=True)
    if result.get('status') != 'passed':
        raise LLMCallRefused('quality backend preflight did not pass; no model request')
    return result


def _close_backend(backend, root):
    if backend is None:
        return
    try:
        result = backend.close()
        atomic_json(root / 'cleanup.json', audit_value(result))
    except Exception as error:
        atomic_json(root / 'cleanup.json', dict(error_type=type(error).__name__, quiescent=None))


def _limits(budget, *, max_calls=None, allowed_callers=('QualityGenerate', 'Fixer')):
    count = budget.max_calls if max_calls is None else max_calls
    return LLMCallLimits(max_calls=count, max_output_tokens=budget.max_output_tokens,
        total_output_token_reservations=count * budget.max_output_tokens,
        max_request_bytes=budget.max_request_bytes, max_response_bytes=budget.max_response_bytes,
        call_seconds=budget.call_seconds, campaign_seconds=budget.campaign_seconds,
        allowed_callers=allowed_callers)


def _new_client(case_id, ledger, budget, replay, replies, factory=None):
    if factory is not None:
        return factory(case_id, ledger)
    if replay:
        return ReplayClient(replies, ledger)
    return LLMClient(max_tokens=budget.max_output_tokens, temperature=.1,
                     timeout=budget.call_seconds, call_budget=ledger)


def _generate(case_id, task, client, root, budget, replay, *, messages=None, check_source=True):
    messages = generation_messages(case_id, task) if messages is None else messages
    atomic_json(root / 'generation-request.json', messages, readonly=True)
    raw = client.chat(messages, temperature=.1, max_tokens=budget.max_output_tokens,
                      raw=True, caller_tag='QualityGenerate')
    response = strict_object(raw)
    if (set(response) != {'code', 'rationale'} or not isinstance(response['code'], str)
            or not response['code'].strip() or len(response['code'].encode()) > 32768
            or not isinstance(response['rationale'], str) or not response['rationale'].strip()):
        raise ValueError('invalid quality source envelope')
    if not replay and check_source:
        precheck(response['code'], input_metadata(list(quality_suite.evaluation_inputs_for(case_id))))
    return response['code']


def _repair_candidate(code, task, inputs, recorder, strategy, budget):
    fixer = FixerAgent(llm=recorder.client)
    fixer.FIX_MAX_TOKENS = min(fixer.FIX_MAX_TOKENS, budget.max_output_tokens)
    result = repair(code, task, inputs, recorder, fixer,
        max_rounds=budget.max_verifications, stop_after=VerificationLevel.L2_FUNCTION,
        use_diagnosis=strategy == 'diagnosis')
    for record, round_result in zip(recorder.records, result['rounds']):
        record['repair_action'] = round_result.get('action')
        record['diagnosis_state'] = (round_result.get('diagnosis') or {}).get('state')
        record['repair_basis'] = round_result.get('repair_basis')
        record['repair_conflict'] = (round_result.get('conflict') or {}).get('present')
    atomic_json(recorder.root / 'repair.json', audit_value(
        {k: v for k, v in result.items() if k not in ('results', 'log')}))
    return result


def _finish_records(recorder, client, ledger, case_id, started, phase, replay):
    rows = recorder.records if recorder else []
    reserved = ledger.summary()['reserved_calls'] if ledger else 0
    calls, tokens, usage_complete = _usage(client) if client else (reserved, 0, reserved == 0)
    accounted_calls = sum(r['model_calls'] for r in rows)
    accounted_tokens = sum(r['tokens'] for r in rows if r['tokens'] is not None)
    if not rows or calls > accounted_calls:
        rows.append(dict(case_id=case_id, attempt=len(rows)+1,
            status=phase+'_failed' if not rows else 'repair_failed',
            full_l2_passed=None, performance_target_passed=None,
            measurement_validity=None, execution_kind='replay' if replay else 'real_device',
            model_calls=max(0, calls-accounted_calls),
            tokens=max(0, tokens-accounted_tokens) if usage_complete else None,
            token_usage_sources=_usage_sources(client) if client else [],
            wall_seconds=max(0, time.monotonic()-started-sum(r['wall_seconds'] for r in rows))))
    return rows


def run_preflight(plan, output_dir, *, host='', backend_factory=None):
    """Check an existing SSH environment without constructing any model client."""
    _validate_execution(plan, False, host, check_model=False)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    atomic_json(root / 'plan.json', plan, readonly=True)
    rows = []
    for declared in plan['manifest']['cases']:
        case_id = declared['case_id']
        task_root = root / case_id
        task_root.mkdir()
        backend, log = None, io.StringIO()
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                backend = _make_backend(case_id, False, host, backend_factory)
                result = _preflight(backend, task_root)
            row = dict(case_id=case_id, status='passed', preflight=result)
        except Exception as error:
            row = dict(case_id=case_id, status='failed', error_type=type(error).__name__)
        finally:
            _close_backend(backend, task_root)
            (task_root / 'execution.log').write_text(log.getvalue())
        rows.append(row)
    report = dict(schema='operator-quality-preflight-run-v1', plan_hash=plan['hash'],
                  cases=rows, passed=all(row['status'] == 'passed' for row in rows),
                  model_calls=0, candidate_verification=False)
    atomic_json(root / 'report.json', report)
    return report


def run_plan(plan, output_dir, *, responses=None, host='', client_factory=None, backend_factory=None):
    """Run a new experiment. Replay uses CPU; real execution requires a frozen SSH plan."""
    replay = responses is not None
    _validate_execution(plan, replay, host)
    controls = plan['controls']
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    atomic_json(root / 'plan.json', plan, readonly=True)
    records = []
    budget_spec = QualityBudget(**controls['budget'])
    for declared in plan['manifest']['cases']:
        case_id = declared['case_id']
        task_root = root / case_id
        task_root.mkdir()
        task = quality_suite.task_for(case_id, hardware='mock_gpu' if replay else 'nvidia_gpu',
                                      dsl='python' if replay else 'triton')
        task_inputs = list(quality_suite.evaluation_inputs_for(case_id))
        client = recorder = ledger = backend = None
        started = time.monotonic()
        log = io.StringIO()
        phase = 'generation' if replay else 'preflight'
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                backend = _make_backend(case_id, replay, host, backend_factory)
                if not replay:
                    _preflight(backend, task_root)
                phase = 'generation'
                ledger = LLMCallBudget(_limits(budget_spec), task_root / 'model-journal')
                client = _new_client(case_id, ledger, budget_spec, replay,
                                     responses.get(case_id, []) if replay else [], client_factory)
                code = _generate(case_id, task, client, task_root, budget_spec, replay)
                recorder = RecordingVerifier(Verifier(backend), client, task_root, case_id,
                    budget_spec, 'replay' if replay else 'real_device', started=started)
                phase = 'verification'
                _repair_candidate(code, task, task_inputs, recorder, controls['strategy'], budget_spec)
        except Exception as error:
            # Preserve failed tasks without serializing arbitrary provider or source exceptions.
            atomic_json(task_root / 'failure.json', dict(error_type=type(error).__name__))
        finally:
            records.extend(_finish_records(recorder, client, ledger, case_id, started, phase, replay))
            if client:
                atomic_json(task_root / 'usage.json', client.cost_tracker.summary())
            _close_backend(backend, task_root)
            (task_root / 'execution.log').write_text(log.getvalue())
            atomic_json(root / 'records.json', records)
    report = quality_suite.summarize(records, manifest=plan['manifest'], controls=controls)
    atomic_json(root / 'report.json', report)
    return report
