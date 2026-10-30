"""One production model-authored Triton candidate; LOCAL PLAN ONLY by default.

Execution requires a separately authorized, exact prepared plan and its hash.
Preparing a plan is not authorization. No HTTP/SSH/input allocation by default;
no implicit repair, template, resume, install, retries or timing-window selection.
An explicit --repair-parent-run prepares a separate, once-per-parent repair plan.
The remote runner is same-UID, NOT a hostile-code/VRAM/network sandbox.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.drivers import workload_decision_h20 as wd
from benchmarks import _numerical_repair as nr
from benchmarks._preparation import PreparationJournal, POLICY as PREPARATION_POLICY, gpu_snapshot
from benchmarks.drivers.input_fidelity_h20 import gpu_evidence
from operator_agent import Conductor, KernelDatabase
from operator_agent.artifact_store import ArtifactStore
from operator_agent.budget import BudgetPlan
from operator_agent.contract import case_key, freeze_from_task, load_dict
from operator_agent.kernel_db import FailureDatabase
from operator_agent.llm_budget import LLMCallBudget, LLMCallLimits, strict_object, validated_usage
from operator_agent.llm_client import LLMClient, CostTracker, parse_usage, _estimate_tokens_from_text
from operator_agent.reference_contract import registered_reference, validate_receipt
from operator_agent.run_contract import (atomic_json, audit_value, code_hash, digest,
                                        json_value, load_checkpoint, protocol_identity)
from operator_agent.schemas import UnifiedSketch, VerificationLevel as Level
from operator_agent.search_objective import reading, results_dict
from operator_agent.single_candidate_coder import (PROMPT, SingleCandidateCoder,
    SingleCandidatePolicy, audit_generation, generation_failure, wrapper_calling_convention,
    generation_declaration as describe_generation, generation_mode)

LIMITS = LLMCallLimits(max_calls=1, max_output_tokens=4096, total_output_token_reservations=4096)
POLICY = SingleCandidatePolicy()


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--approved-plan', help='prepared plan path; never an implicit approval')
    p.add_argument('--plan-hash', help='exact hash separately authorized by the human')
    p.add_argument('--host', default='algorithm-h20')
    p.add_argument('--gpu', type=int, choices=range(8))
    p.add_argument('--cuda-ruler-profile')
    p.add_argument('--model-url', help='explicit single endpoint, no credentials in URL')
    p.add_argument('--model', help='exact requested/reported model identity')
    p.add_argument('--repair-parent-run', help='original L1-pass/primary-L2-fail run; separate approval required')
    p.add_argument('--output', default=str(ROOT / 'output' / 'workload_codegen_h20'))
    p.set_defaults(cuda_calibration=None)
    return p


def model_identity(args):
    if not args.model_url and not args.model:
        return None
    if not args.model_url or not args.model:
        raise ValueError('both --model-url AND --model required')
    url = args.model_url.rstrip('/')
    try:
        parsed = urllib.parse.urlsplit(url)
        parsed.port
        valid = (parsed.scheme in ('http', 'https') and parsed.hostname
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('unsafe explicit model endpoint')
    return dict(endpoint_sha256=code_hash(url), model=args.model,
        credential_present=bool(os.environ.get('LLM_API_KEY', '')), enable_thinking=False,
        timeout=60, default_temperature=.1, default_max_tokens=4096)


def sketch():
    # Trusted semantic hint ONLY: no template source or model-chosen launch plan.
    return UnifiedSketch(algorithm='layernorm', target_dsl='triton',
        optimization_hints=[
            'Frozen native LayerNorm reference; FP32 accumulation; preserve input layout semantics.',
            'Normalize each logical row along the last axis, using population variance over '
            'the logical column count and the frozen reference epsilon; preserve the declared output dtype.',
            'Exclude padded lanes from both the mean and centered-square variance reductions, '
            'including after subtracting the mean; cover nonzero constant and near-constant rows.',
            'Use actual input row/column strides and independent output strides in element units; '
            'support the declared contiguous and strided layouts even when primary inputs are contiguous.',
        ])


def sources():
    result = wd.evaluator_sources()
    result['benchmarks/drivers/workload_codegen_h20.py'] = code_hash(Path(__file__).read_text())
    result['benchmarks/_preparation.py'] = code_hash((ROOT/'benchmarks/_preparation.py').read_text())
    result['benchmarks/_numerical_repair.py'] = code_hash((ROOT/'benchmarks/_numerical_repair.py').read_text())
    return result


def generation_declaration(task, repair=None, diagnosis=None):
    # Pure description: do not allocate 432 MiB, fake arrays or contact a model.
    metadata = [{'X': dict(shape=list(s), dtype='float16', strides=[s[1]*2, 2])} for s in wd.LARGE]
    context = json_value(dict(task_id=task.task_id, hardware=task.target.hardware,
        dsl=task.target.dsl, contract=freeze_from_task(task).as_dict(),
        reference_source=task.source.reference_source, sketch=sketch().to_dict(),
        input_metadata=metadata, calling_convention=wrapper_calling_convention(metadata)))
    if repair is not None:
        context['repair'] = copy.deepcopy(repair)
    if diagnosis is not None:
        context['repair_diagnosis'] = copy.deepcopy(diagnosis)
    return describe_generation(context, POLICY)


def request_body(generation, identity):
    return dict(model=identity['model'], messages=[dict(role='system', content=generation['prompt']),
        dict(role='user', content=json.dumps(generation['context'], ensure_ascii=False))],
        temperature=.1, max_tokens=4096, stream=False, chat_template_kwargs={'enable_thinking': False})


def declaration(args):
    task = wd.task('workload-single-codegen-h20')
    task.constraints.min_speedup = 10.  # Frozen wiring target, NOT a deployment promise.
    profile, calibration = wd.ruler(args)
    identity = model_identity(args)
    repair, parent = (nr.load_parent(args.repair_parent_run, task)
                      if getattr(args, 'repair_parent_run', None) else (None, None))
    if parent: nr.require_available(parent)
    gen = generation_declaration(task, repair)
    limits = LLMCallLimits(**dict(asdict(LIMITS), allowed_callers=(generation_mode(gen['context'])[2],)))
    body = dict(schema='workload-single-codegen-h20-v1', contract=gen['context']['contract'],
        supplied_cases=list(task.constraints.required_cases), generation=gen, model=identity,
        model_limits=json_value(asdict(limits)), measurement_protocol=protocol_identity(),
        preparation_policy=json_value(PREPARATION_POLICY),
        evaluator_sources=sources(),
        budget_accounting=dict(unit='candidate_verification_dispatch_per_input_group',
            reservation_policy='pre_dispatch_non_refundable',
            calibration_policy='observe_only_may_exceed_total',
            fixed_dispatch_sequence_unchanged=True),
        controls=dict(host=args.host, gpu=args.gpu, max_rounds=1, heartbeat_interval=1,
            cuda_ruler_profile=profile, cuda_calibration=calibration,
            wall_clock_s=1800, rpc_timeout=wd.RPC_TIMEOUT, max_rpcs=wd.MAX_RPCS,
            max_commands=wd.MAX_COMMANDS, cleanup_seconds=wd.CLEANUP_SECONDS,
            max_main_windows=2, max_confirmation_windows=2,
            # Existing 10% split needs total=20 to reserve two input groups for
            # confirmation. Model, candidate, window and RPC caps stay fixed.
            budget=dict(oracle_total=20, wall_clock_s=1800, llm_tokens=40000),
            warmup=inspect.signature(wd.ProfileSSH.benchmark_with_baselines).parameters['warmup'].default,
            repeat=inspect.signature(wd.ProfileSSH.benchmark_with_baselines).parameters['repeat'].default,
            baseline_priority=['pytorch_eager']),
        input_bounds=[dict(shape=list(s), storage_bytes=s[0]*s[1]*2,
                           packet_storage_plus_scalars_bytes=s[0]*s[1]*2+2) for s in wd.LARGE],
        input_storage_limit=wd.MAX_STORAGE_BYTES, sequential_cases=True, retry_windows=False,
        runtime_boundary=dict(owned_scratch_only=True, remote_same_uid=True,
            hostile_code_sandbox=False, network_isolated=False, filesystem_allowlist=False,
            rss_cap=False, vram_cap=False, gpu_watchdog=False, exclusive_device_lease=False,
            static_precheck='conservative pruning, not isolation', installs=False),
        disclosure=dict(tensor_values=False, scalar_values=False, repository_files=False,
            credentials_in_body=False, template_source=False, previous_campaign_feedback=False,
            content='exact frozen contract/registered reference, trusted sketch, caller-derived wrapper calling convention and input metadata only'),
        performance_scope='one model-authored source on two declared cases; not all-shape H20 acceptance')
    if parent:
        body['repair_parent'] = parent
        body['disclosure'].update(previous_campaign_feedback=True, failed_model_source=True,
            content='frozen generation context plus exact failed model source, parent identity hashes and allowlisted numerical feedback; no raw logs, output samples or templates')
    if identity:
        req = request_body(gen, identity)
        raw = json.dumps(req, ensure_ascii=False, allow_nan=False)
        if len(raw.encode()) > LIMITS.max_request_bytes:
            raise ValueError('frozen full request exceeds byte cap')
        body['request'] = dict(body=req, sha256=code_hash(raw), bytes=len(raw.encode()))
    return task, json_value(dict(body, hash=digest(body)))


def read_object(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
        raise ValueError('missing/symlink evidence: ' + path.name)
    return strict_object(path.read_text())


def check_plan(path, plan, task, *, frozen_sources=False):
    if (digest(read_object(path)) != digest(plan)
            or plan['hash'] != digest({k:v for k,v in plan.items() if k != 'hash'})
            or sources() != plan['evaluator_sources']
            or protocol_identity() != plan['measurement_protocol']
            or json_value(freeze_from_task(task).as_dict()) != plan['contract']
            or generation_declaration(task, plan['generation']['context'].get('repair')) != plan['generation']):
        raise ValueError('frozen codegen plan/context/evaluator drift')
    parent = plan.get('repair_parent')
    if parent:
        if digest(plan['generation']['context']['repair']['parent']) != parent['identity_hash']:
            raise ValueError('repair parent identity drift')
        nr.check_files(parent['root'], parent['files'])
        if frozen_sources:
            nr.check_files(Path(path).parent/'frozen-parent', parent['files'])
    if frozen_sources:
        for name, expected in plan['evaluator_sources'].items():
            p = Path(path).parent / 'frozen-sources' / name
            # Reject path traversal and all symlink components before reads.
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('unsafe source snapshot path')
            if p.is_symlink() or any(q.is_symlink() for q in p.parents) or code_hash(p.read_text()) != expected:
                raise ValueError('frozen source snapshot drift')


def freeze(root, plan):
    atomic_json(root / 'plan.json', plan, readonly=True)
    for name, expected in plan['evaluator_sources'].items():
        data = (ROOT / name).read_bytes()
        if code_hash(data.decode()) != expected:
            raise ValueError('source changed during freeze')
        p = root / 'frozen-sources' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
    if plan.get('repair_parent'):
        parent = plan['repair_parent']
        nr.check_files(parent['root'], parent['files'])
        for name, expected in parent['files'].items():
            data = nr.safe_bytes(parent['root'], name)
            if code_hash(data.decode()) != expected: raise ValueError('parent changed during freeze')
            p = root/'frozen-parent'/name
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('xb') as f:
                f.write(data); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(),0o444)


def claim_execution(path, root, plan):
    # Consume BEFORE external I/O. A crash/failed attempt does not buy a retry.
    marker = Path(path).parent / 'execution-claim.json'
    data = json.dumps(dict(plan_hash=plan['hash'], execution_root=str(root), resumable=False)).encode()
    fd, tmp = tempfile.mkstemp(dir=str(marker.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
        try:
            os.link(tmp, marker)  # Exclusive, including dangling symlinks.
        except FileExistsError:
            raise ValueError('prepared plan already consumed; no retry/resume') from None
        d = os.open(str(marker.parent), os.O_RDONLY)
        try: os.fsync(d)
        finally: os.close(d)
    finally:
        os.unlink(tmp)


class PlanLLMClient(LLMClient):
    """Final production request is compared to the approved bytes BEFORE reserve."""
    def _call_bounded_model(self, base_url, api_key, model, body, raw, caller_tag):
        self.plan_guard()
        if (self._bounded_identity() != self.frozen_plan['model']
                or json.dumps(body, ensure_ascii=False, allow_nan=False) !=
                    json.dumps(self.frozen_plan['request']['body'], ensure_ascii=False, allow_nan=False)):
            raise ValueError('approved model request/identity drift before dispatch')
        return super()._call_bounded_model(base_url, api_key, model, body, raw, caller_tag)


class CodegenSSH(wd.BoundedProfileSSH):
    """Plan guard on all external transport; cleanup bypasses a broken plan."""
    def __init__(self, *, plan_guard, candidate_guard, plan, preparation_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.plan_guard = plan_guard; self.candidate_guard = candidate_guard
        self.plan = plan; self.bench_dispatches = []; self.compile_dispatches = 0
        self.verify_dispatches = 0; self.heartbeat_dispatches = 0
        self.preparation = (PreparationJournal(preparation_dir, plan['controls']['gpu'])
                            if preparation_dir is not None else None)

    def prepare(self, requested_gpu=None):
        if (self.preparation is None or type(requested_gpu) is not int
                or requested_gpu != self.plan['controls']['gpu']):
            raise ValueError('fresh preparation journal and frozen GPU required')
        self.plan_guard()
        return super().prepare(requested_gpu)

    def _prepare_stage(self, name, operation):
        def checked_operation():
            self.plan_guard()
            return operation()
        def details(value):
            if name == 'runtime_probe': return dict(cuda_available=value['cuda'] is True)
            if name == 'gpu_query': return dict(devices=gpu_snapshot(value))
            if name == 'gpu_select':
                if value['index'] != self.plan['controls']['gpu'] or 'H20' not in value['name']:
                    raise ValueError('wrong physical GPU or non-H20')
                return dict(selected_index=value['index'], is_h20=True)
            if name == 'scratch_create': return dict(private_mode=value['mode'] == 0o700)
            if name == 'runner_readback': return dict(cuda=True, selected_index=self._gpu_pin)
            return {}
        return self.preparation.step(name, checked_operation, lambda: self.counts, details)

    def guard(self, kind):
        if not self.cleaning:
            self.plan_guard()
        super().guard(kind)

    def _ssh_exec_json(self, payload):
        if not self.cleaning and payload.get('action') == 'cuda_calibration_probe':
            if self.heartbeat_dispatches >= 2 or payload.get('mode') != 'heartbeat':
                raise ValueError('extra/recalibration heartbeat refused')
            self.heartbeat_dispatches += 1
        if not self.cleaning and payload.get('action') in {'compile', 'verify', 'benchmark_with_baselines'}:
            self.plan_guard(); self.candidate_guard(payload['code'])
            if payload['action'] == 'compile':
                if self.compile_dispatches:
                    raise ValueError('second candidate compile refused')
                self.compile_dispatches += 1
            else:
                contract = self.plan['contract']
                if (payload.get('kernel_type') != 'layernorm' or payload.get('precision') != contract['dtype_semantics']['dtype']
                        or payload.get('reference_contract') != registered_reference().content_dict()
                        or payload.get('reference_source') != (self.plan['generation']['context']['reference_source'] or '')):
                    raise ValueError('frozen remote precision/reference drift')
            if payload['action'] == 'verify':
                cap = 2 + 2*len(wd.as_suite(self.plan['contract']['correctness_suite']).cases)
                if self.verify_dispatches >= cap or payload.get('max_error') != self.plan['contract']['max_error']:
                    raise ValueError('extra correctness dispatch or tolerance drift')
                self.verify_dispatches += 1
            if payload['action'] == 'benchmark_with_baselines':
                controls = self.plan['controls']
                if any(payload.get(k) != controls[k] for k in ('warmup','repeat','baseline_priority')):
                    raise ValueError('frozen benchmark parameters drift')
                tensors = payload['inputs']['tensors']
                key = '|'.join(k+'='+'x'.join(map(str,v['shape'])) for k,v in sorted(tensors.items()))
                i = len(self.bench_dispatches)
                if i >= 4 or key != self.plan['supplied_cases'][i % 2]:
                    raise ValueError('undeclared/reordered/retried case window refused')
                self.bench_dispatches.append(key)  # Failures never refund dispatches.
        return super()._ssh_exec_json(payload)


def classify_cleanup(cleanup, backend, preparation):
    """No-op needs proof of no creation attempt, not merely a missing path."""
    state = 'unconfirmed'; errors = []
    cleanup = cleanup if isinstance(cleanup, dict) else {}
    if cleanup.get('quiescent') is True and cleanup.get('removed') is True:
        state = 'removed'
    elif (cleanup.get('quiescent') is True and cleanup.get('scratch_created') is False
            and 'removed' not in cleanup and preparation.get('evidence_consistent') is True
            and preparation.get('scratch_creation_attempted') is False
            and getattr(backend, 'scratch', 'unknown') is None
            and getattr(backend, '_session', 'unknown') is None
            and getattr(backend, '_pending_session', 'unknown') is None
            and getattr(backend, '_deployed', True) is False
            and getattr(backend, 'work_counts', {}).get('rpc') == 0
            and getattr(backend, 'compile_dispatches', None) == 0
            and getattr(backend, 'verify_dispatches', None) == 0
            and getattr(backend, 'bench_dispatches', None) == []):
        state = 'not_needed'
    if state == 'unconfirmed': errors.append('owned runner/scratch cleanup not confirmed')
    if getattr(backend, 'forbidden_actions', []): errors.append('forbidden remote actions')
    return dict(state=state, confirmed=state != 'unconfirmed', errors=errors)


def audit_no_model_attempt(root, plan):
    """Zero calls are not a successful generation, nor missing-candidate drift."""
    root = Path(root); journal = root/'model-journal'
    proof = dict(completed=False, status='not_started', evidence_consistent=False,
                 errors=[], read_only=True, gpu_measurements=0, restored_eligibility=False)
    try:
        if not journal.exists():
            if journal.is_symlink() or any((root/n).exists() or (root/n).is_symlink()
                    for n in ('model-budget.json', 'llm-usage.json')):
                raise ValueError('unexpected zero-call evidence')
        else:
            allowed = {'limits.json', 'identity.json'}
            names = {p.name for p in journal.iterdir()}
            if 'generation-context.json' in names or 'generation-declaration.json' in names:
                allowed |= {'generation-context.json', 'generation-declaration.json'}
            if names != allowed: raise ValueError('unexpected zero-call journal entries')
            paths = [journal/n for n in sorted(names)] + [root/'model-budget.json', root/'llm-usage.json']
            before = {str(p.relative_to(root)):code_hash(p.read_text()) for p in paths}
            values = {n:read_object(journal/n) for n in names}
            budget = read_object(root/'model-budget.json'); usage = read_object(root/'llm-usage.json')
            if (values['limits.json'] != plan['model_limits'] or values['identity.json'] != plan['model']
                    or budget['identity'] != plan['model'] or budget['limits'] != plan['model_limits']
                    or any(type(budget.get(k)) is not int or budget[k] != 0
                           for k in ('reserved_calls', 'output_tokens_reserved'))
                    or budget['attempts'] != [] or usage != CostTracker().summary()):
                raise ValueError('zero-call identity/usage/reservation drift')
            if 'generation-context.json' in values:
                if (values['generation-context.json'] != plan['generation']['context']
                        or values['generation-declaration.json'] != plan['generation']):
                    raise ValueError('zero-call context drift')
                proof['status'] = 'refused_before_dispatch'
            if before != {str(p.relative_to(root)):code_hash(p.read_text()) for p in paths}:
                raise ValueError('changing zero-call evidence')
            proof['file_hashes'] = before
        proof['evidence_consistent'] = True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        proof['errors'].append('zero-call evidence audit: '+type(exc).__name__)
    return proof


def audit_model(root, plan):
    """Audit retained production journal and exact source, without a live client."""
    errors = []; proof = dict(completed=False, errors=errors, read_only=True, gpu_measurements=0,
                              restored_eligibility=False)
    try:
        root = Path(root); journal = root / 'model-journal'
        names = ['limits.json', 'identity.json', 'generation-context.json',
                 'generation-declaration.json', 'call-001.json', 'candidate.json']
        paths = [journal/n for n in names] + [root/'model-budget.json', root/'llm-usage.json']
        before = {str(p.relative_to(root)): code_hash(p.read_text()) for p in paths if p.is_file() and not p.is_symlink()}
        values = [read_object(p) for p in paths]
        limits, identity, context, decl, call, lineage, budget, usage = values
        if (decl != plan['generation'] or context != decl['context'] or limits != plan['model_limits']
                or identity != plan['model'] or budget['identity'] != identity or budget['limits'] != limits
                or budget['attempts'] != [call] or call['request'] != plan['request']['body']
                or call['request_sha256'] != plan['request']['sha256']
                or call['request_bytes'] != plan['request']['bytes']
                or sorted(p.name for p in journal.glob('call-*.json')) != ['call-001.json']):
            raise ValueError('journal/approved request/declaration drift')
        if lineage.get('status') == 'failed':
            # Fixed vocabulary only, not exception text or model-owned rationale.
            # Retaining a diagnostic cannot turn an aborted campaign into a pass.
            proof['failure'] = generation_failure(lineage)
        proof.update(audit_generation(decl, budget, lineage, lineage['code']))
        response = strict_object(call['response_text']); content = response['choices'][0]['message']['content']
        raw_usage, source = validated_usage(response.get('usage')); norm = parse_usage(raw_usage)
        tracker = CostTracker()
        tracker.record(model=identity['model'], caller_tag=generation_mode(decl['context'])[2], usage_source=source,
            prompt_tokens=norm['prompt_tokens'] if source == 'api_usage' else _estimate_tokens_from_text(json.dumps(call['request'], ensure_ascii=False, allow_nan=False)),
            completion_tokens=norm['completion_tokens'] if source == 'api_usage' else _estimate_tokens_from_text(content),
            cached_prompt_tokens=norm['cached_prompt_tokens'], cache_semantics=norm['cache_semantics'],
            reported_model=response.get('model'), temperature=.1, max_tokens=4096, enable_thinking=False)
        if usage != tracker.summary():
            raise ValueError('raw response usage does not reconstruct saved cost summary')
        if before != {str(p.relative_to(root)): code_hash(p.read_text()) for p in paths}:
            raise ValueError('model audit observed changing evidence')
        proof.update(file_hashes=before, usage=usage, source=lineage['code'], declaration=decl,
                     lineage=lineage, budget=budget, completed=True)
    except (OSError, ValueError, KeyError, TypeError, IndexError, SyntaxError) as exc:
        proof['completed'] = False; errors.append('model evidence audit: ' + type(exc).__name__ + ': ' + str(exc))
    return proof


def observation_status(results):
    levels = {r.level:r.passed for r in results}
    if levels.get(Level.L1_COMPILE) is False or levels.get(Level.L2_FUNCTION) is False:
        return 'correctness_failed'
    l3 = next((r for r in results if r.level is Level.L3_PERFORMANCE), None)
    if levels.get(Level.L1_COMPILE) is not True or levels.get(Level.L2_FUNCTION) is not True or l3 is None:
        return 'incomplete'
    tier = l3.metrics.get('measurement_validity')
    if tier != 'valid':
        return tier if tier in ('degraded', 'uninformative') else 'incomplete'
    return 'valid_target_passed' if l3.passed else 'valid_below_target'


def audit_durable(artifact, rounds, plan, model, *, task, input_sets, controls):
    errors = []; proof = dict(completed=False, errors=errors, read_only=True, gpu_measurements=0, resumed=False)
    try:
        directory = Path(artifact.saved_dir)
        required = ['result.json', 'sketch.json', 'verification.json', 'execution_snapshot.json',
                    'contract.json', 'run_manifest.json', 'report.md', 'code.py']
        if artifact.benchmark_data: required.append('benchmark.json')
        for name in ('confirmation.json', 'admission.json'):
            if (directory/name).exists() or (directory/name).is_symlink():
                read_object(directory/name); required.append(name)
        paths = [directory/n for n in required]
        for p in paths:
            if p.is_symlink() or any(q.is_symlink() for q in p.parents) or not p.is_file():
                raise ValueError('missing/symlink durable evidence')
        before = {p.name: code_hash(p.read_text()) for p in paths}
        raw = read_object(directory/'result.json')
        loaded = ArtifactStore(str(directory)).load(str(directory))
        snap, binding = load_checkpoint(directory/'execution_snapshot.json', task, input_sets, controls)
        if len(rounds) != 1 or not model['completed']:
            raise ValueError('exactly one model-proven raw candidate required')
        code, results = rounds[0]
        generation = dict(declaration=model['declaration'], lineage=model['lineage'])
        if (code != model['source'] or loaded.code != code or artifact.code != code
                or loaded.sketch.to_dict() != plan['generation']['context']['sketch']
                or loaded.sketch.to_dict() != artifact.sketch.to_dict()
                or snap['current_code'] != code or snap['round_done'] != 1
                or snap['pending_fix'] is not None or snap['best_gflops'] != 0.
                or raw.get('code_generation') != generation or loaded.metadata.get('code_generation') != generation
                or raw.get('llm_call_budget') != model['budget'] or loaded.metadata.get('llm_call_budget') != model['budget']
                or binding['manifest']['controls']['code_generation'] != plan['generation']
                or raw.get('iteration_history') != snap['iteration_history']
                or loaded.iteration_history != json_value(audit_value(artifact.iteration_history))
                or loaded.metadata.get('stop') != artifact.metadata.get('stop')
                or loaded.all_passed != artifact.all_passed):
            raise ValueError('durable source/context/journal/checkpoint/history drift')
        cost = {k:v for k,v in raw.get('llm_cost', {}).items() if k not in ('scope', 'process_total_calls')}
        if (cost != model['usage'] or raw['llm_cost'].get('scope') != 'this_run'
                or raw['llm_cost'].get('process_total_calls') != 1
                or loaded.metadata.get('llm_cost') != raw['llm_cost']):
            raise ValueError('durable model usage missing or changed')
        derived = {'claim_level', 'objective', 'implementation_decision', 'case_key', 'case_source', 'case_declared', 'case_match'}
        def core(rows):
            return json_value(audit_value([dict(level=r.level.value, passed=r.passed, message=r.message,
                metrics={k:v for k,v in r.metrics.items() if k not in derived}) for r in rows]))
        if core(loaded.verification_results) != core(results):
            raise ValueError('durable verification not its own raw candidate evidence')
        if loaded.benchmark_data != json_value(audit_value(artifact.benchmark_data)):
            raise ValueError('durable benchmark drift')
        status = observation_status(results)
        l3 = next((r for r in results if r.level is Level.L3_PERFORMANCE), None)
        if l3:
            if core([type(l3)(l3.level, l3.passed, l3.message, metrics=loaded.benchmark_data)]) != core([l3]):
                raise ValueError('benchmark not its own verification ledger')
            problems = wd.binding_errors(loaded, l3.metrics, load_dict(plan['contract']),
                                         dict(plan, candidate=dict(hash=code_hash(code))))
            if problems: raise ValueError('; '.join(problems))
        selection = loaded.metadata.get('candidate_selection') or {}
        search = loaded.metadata.get('search_objective')
        if search != snap['search_objective'] or selection != artifact.metadata.get('candidate_selection'):
            raise ValueError('durable selection/search drift')
        point = reading(l3.metrics, load_dict(plan['contract'])) if l3 else dict(can_select=False)
        expected_round = 1 if point['can_select'] else None
        if selection.get('selected_round') != expected_round or selection.get('fallback_to_best'):
            raise ValueError('single candidate selection fabricated a winner')
        if not artifact.all_passed:
            if (loaded.metadata['stop']['reason'] != ('l1_l2_failed' if status == 'correctness_failed' else 'rounds_exhausted')
                    or (loaded.metadata.get('corpus_admission') or {}).get('kept') is True
                    or (l3 is not None and (loaded.metadata.get('corpus_admission') or {}).get('kept') is not False)
                    or any((directory/n).exists() or (directory/n).is_symlink() for n in ('confirmation.json', 'admission.json'))):
                raise ValueError('failed single candidate granted confirmation/admission')
        from operator_agent.report import build_report
        report = (directory/'report.md').read_text()
        regenerated = build_report(artifact=loaded, task=task, generated_at='audit-only')
        start = '### 声明 workload 的搜索选优'; end = '## 2. 证据链 (逐轮)'
        section = regenerated.split(start, 1)[1].split(end, 1)[0]
        chain = regenerated.split(end, 1)[1].split('## 3.', 1)[0]
        if start+section not in report or end+chain not in report:
            raise ValueError('durable report disagrees with loaded evidence')
        if before != {p.name:code_hash(p.read_text()) for p in paths}:
            raise ValueError('durable audit observed changing evidence')
        proof.update(completed=True, status=status, file_hashes=before, checkpoint_verified=True,
                     candidate_hash=code_hash(code), restored_eligibility=False)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        errors.append('durable codegen audit: ' + type(exc).__name__ + ': ' + str(exc))
    return proof


def audit_raw(artifact, rounds, trace, plan, *, task, corpus):
    """Raw CUDA evidence audit; honest failures are not acceptance successes."""
    errors = []; proof = dict(completed=False, accepted=False, deployment_accepted=False,
                              errors=errors, status='incomplete', read_only=True, gpu_measurements=0)
    try:
        if len(rounds) != 1 or artifact.code != rounds[0][0]:
            raise ValueError('one fresh candidate observation required')
        code, results = rounds[0]; status = observation_status(results); proof['status'] = status
        contract = load_dict(plan['contract']); controls = plan['controls']; supplied = plan['supplied_cases']
        for r in results:
            b = r.metrics.get('run_binding') or {}
            if b.get('candidate_hash') != code_hash(code) or b.get('contract_hash') != contract.hash:
                errors.append('raw candidate/contract binding drift')
        if len(trace) > controls['max_rpcs']:
            errors.append('remote RPC cap exceeded')
        candidate_rows = [r for r in trace if r['request']['action'] in {'compile','verify','benchmark_with_baselines'}]
        if any(r['request'].get('code') != code for r in candidate_rows):
            errors.append('undeclared candidate in remote trace')
        compiles = [r for r in candidate_rows if r['request']['action']=='compile']
        if len(compiles) != 1:
            errors.append('missing/extra compile dispatch')
        l1 = next((r for r in results if r.level is Level.L1_COMPILE), None)
        if l1 and l1.passed and (len(compiles) != 1 or compiles[0].get('response', {}).get('success') is not True):
            errors.append('passed L1 lacks actual successful remote compile')
        errors += wd.receipt_errors(registered_reference(), trace) + wd.transfer_errors(trace)
        for rpc in trace:
            req = rpc['request']
            if req['action']=='benchmark_with_baselines' and any(req.get(k) != controls[k]
                    for k in ('warmup','repeat','baseline_priority')):
                errors.append('raw benchmark policy drift')
        l3 = next((r for r in results if r.level is Level.L3_PERFORMANCE), None)
        if status == 'correctness_failed':
            if l3 or any(r['request']['action']=='benchmark_with_baselines' for r in trace):
                errors.append('failed correctness reached performance')
        elif l3:
            l2 = next(r for r in results if r.level is Level.L2_FUNCTION)
            c = l2.metrics.get('correctness') or {}; suite = wd.as_suite(contract.correctness_suite)
            rows = c.get('cases') or []
            verifies = [r for r in trace if r['request']['action']=='verify']
            if (c.get('verdict') != 'passed' or c.get('suite_hash') != suite.digest()
                    or c.get('coverage_lint', {}).get('status') != 'passed' or not gpu_evidence(l2.metrics)
                    or [r.get('case_id') for r in rows] != [s.case_id for s in suite.cases]
                    or len(verifies) != len(supplied) + 2*len(suite.cases)):
                errors.append('incomplete primary/native suite/repeat CUDA evidence')
            for key, rpc in zip(supplied, verifies[:len(supplied)]):
                req, res = rpc['request'], rpc.get('response') or {}
                tensors = req['inputs']['tensors']
                observed = '|'.join(k+'='+'x'.join(map(str,v['shape'])) for k,v in sorted(tensors.items()))
                if observed != key or res.get('success') is not True or res.get('device') != 'cuda':
                    errors.append('primary lacks matching successful CUDA RPC: '+key)
            for rpc in verifies:
                req = rpc['request']
                if (req.get('precision') != contract.dtype_semantics.dtype or req.get('max_error') != contract.max_error
                        or req.get('reference_contract') != registered_reference().content_dict()):
                    errors.append('raw verify frozen precision/tolerance/reference drift')
            for row, pair in zip(rows, [verifies[i:i+2] for i in range(len(supplied), len(verifies), 2)]):
                if row.get('verdict') != 'passed' or row.get('determinism') != 'passed':
                    errors.append('suite case/determinism did not pass: '+row['case_id'])
                for m, rpc in zip((row.get('metrics') or {}, row.get('repeat_metrics') or {}), pair):
                    res = rpc.get('response') or {}
                    tensors = rpc['request']['inputs']['tensors']
                    key = '|'.join(k+'='+'x'.join(map(str,v['shape'])) for k,v in sorted(tensors.items()))
                    if (not gpu_evidence(m) or not m.get('output_digest') or res.get('success') is not True
                            or res.get('device') != 'cuda' or key != row['shape_key']
                            or any(res.get(k) != m.get(k) for k in ('output_digest','max_error','max_rel_error',
                                'matched_ratio','bad_elements','atol','rtol','required_matched_ratio','reference_receipt'))):
                        errors.append('suite metrics lack matching actual CUDA RPC: '+row['case_id'])
                if row.get('metrics', {}).get('output_digest') != row.get('repeat_metrics', {}).get('output_digest'):
                    errors.append('suite repeat digest mismatch: '+row['case_id'])
            metrics = [l3.metrics]
            errors += wd.metrics_errors(l3.metrics, contract, supplied)
            if artifact.all_passed:
                m2, problems = wd.confirmation_errors(artifact, l3.metrics, contract, supplied)
                errors += problems
                if m2: metrics.append(m2)
                if (artifact.metadata.get('corpus_admission') or {}).get('kept') is True:
                    admission, problems = wd.corpus_readback(corpus, b'', artifact, task, l3.metrics)
                    proof['admission'] = admission; errors += problems
                    proof['deployment_accepted'] = admission['verified'] and not errors
            errors += wd.trace_metrics_errors(metrics, trace, supplied, code, controls['gpu'])
            errors += wd.ruler_errors(metrics, trace, controls['cuda_calibration'], controls['cuda_ruler_profile'], controls['gpu'])
            windows = [m['cases'][k]['arm_windows']['candidate'] for m in metrics for k in supplied]
            windows += [r['response']['paired']['arm_windows']['candidate'] for r in trace if r['request']['action']=='cuda_calibration_probe']
            if len(windows) != len(set(windows)): errors.append('reused main/confirmation/heartbeat windows')
            proof['windows'] = windows
            proof['objective'] = reading(l3.metrics, contract)
            proof['accepted'] = status.startswith('valid_') and proof['objective']['can_select'] and not errors
        if not (artifact.metadata.get('corpus_admission') or {}).get('kept'):
            if Path(corpus).exists() and Path(corpus).read_bytes(): errors.append('unconfirmed candidate appended to corpus')
        proof['completed'] = not errors and status != 'incomplete'
        proof['accepted'] = proof['accepted'] and not errors
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, StopIteration) as exc:
        errors.append('raw codegen audit: '+type(exc).__name__+': '+str(exc))
        proof.update(completed=False, accepted=False, deployment_accepted=False)
    return proof


def run(args):
    if args.execute and (args.gpu is None or not args.cuda_ruler_profile or not args.model_url or not args.model
                         or not args.approved_plan or not args.plan_hash):
        raise ValueError('--execute requires explicit GPU/profile/model AND approved plan/hash')
    if not args.execute and (args.approved_plan or args.plan_hash):
        raise ValueError('approved plan/hash only valid with explicit --execute')
    task, plan = declaration(args)
    budget_plan = BudgetPlan(**plan['controls']['budget'])
    if args.execute:
        prepared = read_object(args.approved_plan)
        if prepared['hash'] != args.plan_hash or digest(prepared) != digest(plan):
            raise ValueError('execution differs from separately authorized exact plan')
        check_plan(args.approved_plan, plan, task, frozen_sources=True)
        if plan['controls']['cuda_ruler_profile'] is None:
            raise ValueError('native ruler profile required')
    parent = Path(args.output); parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='workload-codegen-', dir=parent))
    freeze(root, plan)
    def guard():
        check_plan(root/'plan.json', plan, task, frozen_sources=True)
        if args.execute and plan.get('repair_parent'):
            nr.check_claim(plan['repair_parent'], plan['hash'], root)
    summary = dict(completed=False, accepted=False, deployment_accepted=False, status='plan_only',
        mode='execute' if args.execute else 'plan-only', run_root=str(root), plan_hash=plan['hash'],
        errors=[], cleanup=None, model_attempts=0, verifier_dispatches=0, performance_scope=plan['performance_scope'])
    print('PLAN '+str(root/'plan.json')+' HASH '+plan['hash'], flush=True)
    if not args.execute:
        atomic_json(root/'summary.json', summary, readonly=True); return 0
    backend = None; client = None; coder = None; art = None; rounds = []; dispatched = []
    corpus = root/'corpus'/'kernel_db.jsonl'
    previous = {k:os.environ.get(k) for k in ('OPERATOR_AGENT_LOG_DIR','OPERATOR_AGENT_LOG_KEEP')}
    try:
        claim_execution(args.approved_plan, root, plan)
        if plan.get('repair_parent'):
            nr.claim(plan['repair_parent'], plan['hash'], root)
        os.environ.update(OPERATOR_AGENT_LOG_DIR=str(root/'logs'), OPERATOR_AGENT_LOG_KEEP='10000')
        client = PlanLLMClient(base_url=args.model_url, model=args.model, api_key=os.environ.get('LLM_API_KEY',''),
            temperature=.1, max_tokens=4096, timeout=60, enable_thinking=False,
            call_budget=LLMCallBudget(LLMCallLimits(**dict(plan['model_limits'],
                allowed_callers=tuple(plan['model_limits']['allowed_callers']))), root/'model-journal'))
        client.plan_guard = guard; client.frozen_plan = plan
        if client.call_budget.summary()['identity'] != plan['model']: raise ValueError('model identity drift')
        def candidate_guard(code):
            if coder is None: raise ValueError('model source not generated yet')
            coder.check_candidate(code)
        backend = CodegenSSH(plan_guard=guard, candidate_guard=candidate_guard, plan=plan,
                             preparation_dir=root/'preparation-journal',
                             host=args.host, timeout=wd.RPC_TIMEOUT)
        environment = backend.prepare(args.gpu); wd.validate_scratch(backend.scratch)
        if (environment.get('selected', {}).get('index') != args.gpu
                or 'H20' not in environment.get('selected', {}).get('name', '')):
            raise ValueError('wrong physical GPU or non-H20')
        reference = backend.reference_identity(registered_reference().content_dict())
        validate_receipt(registered_reference(), reference.get('reference_receipt'))
        atomic_json(root/'environment.json', dict(environment=environment, reference=reference), readonly=True)
        guard()
        xs = [wd.primary(s) for s in wd.LARGE]
        if [case_key(x) for x in xs] != plan['supplied_cases']: raise ValueError('actual workload drift')
        coder = SingleCandidateCoder(client, task, sketch(), xs, POLICY,
            repair=plan['generation']['context'].get('repair'))
        if coder.declaration() != plan['generation']: raise ValueError('actual metadata/generation context drift')
        c = Conductor(backend=backend, llm=client, single_candidate_coder=coder,
            kernel_db=KernelDatabase(str(corpus), persist_retrievals=False),
            failure_db=FailureDatabase(str(root/'failures'/'failure_db.jsonl')), use_coder_pipeline=False,
            max_optimize_rounds=1, heartbeat_interval=1, budget=budget_plan,
            cuda_ruler_profile_path=args.cuda_ruler_profile,
            cuda_calibration_identity_hash=plan['controls']['cuda_calibration']['identity_hash'])
        c.store.output_dir = root/'artifacts'
        verify = c.verifier.verify
        def capture(*a, **k):
            guard(); candidate_guard(k['kernel_code'])
            if dispatched: raise ValueError('second Verifier candidate refused')
            dispatched.append(code_hash(k['kernel_code']))
            # Conductor attaches the authoritative run binding immediately
            # AFTER Verifier returns. Retain these objects through that step;
            # copying them here would lose that binding in every raw result.
            results = verify(*a, **k); rounds.append((k['kernel_code'], results)); return results
        c.verifier.verify = capture
        art = c.execute(task, xs, target_gflops=0.)
        guard()
        summary['saved_dir'] = getattr(art, 'saved_dir', None)
        # Audits run AFTER the final journal is retained, outside live execution.
    except (Exception, KeyboardInterrupt) as exc:
        # Do not persist unsanitized arbitrary HTTP/candidate exception text.
        summary['errors'].append('execution stopped: '+type(exc).__name__)
    finally:
        if backend is not None:
            try:
                summary['cleanup'] = backend.cleanup()
            except (Exception, KeyboardInterrupt):
                summary['cleanup'] = dict(quiescent=False, removed=False, error='owned cleanup failed')
                summary['errors'].append('owned cleanup unconfirmed')
            try:
                journal = getattr(backend, 'preparation', None)
                summary['preparation'] = journal.snapshot() if journal else dict(evidence_consistent=False)
            except (Exception, KeyboardInterrupt):
                summary['preparation'] = dict(evidence_consistent=False)
                summary['errors'].append('preparation evidence audit failed')
            summary['cleanup_status'] = classify_cleanup(summary['cleanup'], backend, summary['preparation'])
            summary['errors'] += summary['cleanup_status']['errors']
        try:
            if client:
                snapshot = client.call_budget.summary(); summary['model_attempts'] = snapshot['reserved_calls']
                atomic_json(root/'model-budget.json', snapshot, readonly=True)
                atomic_json(root/'llm-usage.json', client.cost_tracker.summary(), readonly=True)
            summary['verifier_dispatches'] = len(dispatched)
            atomic_json(root/'evidence.json', json_value(audit_value(dict(
                rounds=[dict(code=code, results=results_dict(rs)) for code,rs in rounds], dispatched=dispatched,
                trace=wd.compact_trace(backend.trace) if backend else [],
                preparation=summary.get('preparation'),
                work_counts=getattr(backend,'work_counts',{}),
                forbidden_actions=getattr(backend,'forbidden_actions',[])))), readonly=True)
        except (Exception, KeyboardInterrupt) as exc:
            summary['errors'].append('evidence retention failed: '+type(exc).__name__)
        finally:
            for k, old in previous.items():
                if old is None: os.environ.pop(k, None)
                else: os.environ[k] = old
    model = audit_model(root, plan) if summary['model_attempts'] else audit_no_model_attempt(root, plan)
    summary['model'] = {k:v for k,v in model.items() if k not in ('source','declaration','lineage','budget')}
    if not model['completed']:
        summary['status'] = ('generation_failed' if summary['model_attempts']
            or model.get('status') == 'refused_before_dispatch' else 'preparation_failed')
        summary['errors'] += model['errors']
    elif art is not None and getattr(art, 'saved_dir', None):
        try:
            retained = read_object(root/'evidence.json')
            expected = json_value(audit_value(dict(rounds=[dict(code=code,results=results_dict(rs)) for code,rs in rounds],
                dispatched=dispatched, trace=wd.compact_trace(backend.trace),
                preparation=backend.preparation.snapshot(),
                work_counts=getattr(backend,'work_counts',{}), forbidden_actions=backend.forbidden_actions)))
            if digest(retained) != digest(expected): raise ValueError('retained raw evidence drift')
            if summary['preparation'].get('completed') is not True:
                raise ValueError('incomplete preparation evidence')
            counts = retained['work_counts']
            if any(type(counts.get(k)) is not int or not 0 <= counts[k] <= plan['controls'][limit]
                   for k,limit in (('rpc','max_rpcs'),('command','max_commands'))):
                raise ValueError('missing/exceeded transport work counts')
            raw = audit_raw(art, rounds, backend.trace, plan, task=task, corpus=corpus)
            durable = audit_durable(art, rounds, plan, model, task=task, input_sets=xs, controls=c._run_controls)
            summary.update(raw=raw, durable=durable, status=raw['status'])
            summary['errors'] += raw['errors'] + durable['errors']
            summary['accepted'] = raw['accepted'] and durable['completed'] and not summary['errors']
            summary['deployment_accepted'] = raw['deployment_accepted'] and summary['accepted']
        except (Exception, KeyboardInterrupt) as exc:
            summary['status'] = 'incomplete'
            summary['errors'].append('final evidence audit failed: '+type(exc).__name__)
    else:
        summary['status'] = 'incomplete'
    summary['completed'] = bool(summary['accepted'] and summary.get('cleanup_status', {}).get('state') == 'removed')
    atomic_json(root/'summary.json', json_value(audit_value(summary)), readonly=True)
    print(wd.dumps(summary), flush=True)
    return 0 if summary['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))
