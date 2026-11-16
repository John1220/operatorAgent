"""Run one P2 pilot pair from the accepted LayerNorm source.

Each arm gets one search round, one Coder run and an independent final
evaluation of the resulting source. The pilot does not grant formal credit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from subprocess import TimeoutExpired
import uuid

from benchmarks.experiments.phase1 import sampling
from benchmarks.experiments.phase1.plan import write_once
from operator_agent.core.kernel_db import KernelDatabase
from operator_agent.core.contract import freeze_from_task
from operator_agent.core.run_contract import code_hash
from operator_agent.core.schemas import UnifiedSketch
from operator_agent.execute.llm_budget import ProductionLLMCallBudget, LLMCallRefused
from operator_agent.execute.llm_client import LLMClient
from operator_agent.execute.task_budget import TaskBudgetConfig
from operator_agent.generate.agents import CoderAgent
from operator_agent.generate.prompt_manager import PromptManager

from . import p2_optimize, p2_plan


class SeedCoderPrompt(PromptManager):
    def __init__(self, seed_code):
        super().__init__()
        self.seed_code = seed_code

    def render(self, name, **values):
        if name == 'coder_generate_triton':
            values['ref_code'] = self.seed_code
        return super().render(name, **values)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def generate_arm(*, root, arm, task, sketch, seed_code, seed_metrics, clients):
    """Persist the arm record even when the provider yields no usable source."""
    arm_dir = root / arm
    arm_dir.mkdir()
    with clients(arm_dir) as (llm, usage):
        coder = CoderAgent(KernelDatabase(jsonl_path=str(arm_dir / 'references.jsonl'),
                                         persist_retrievals=False), llm=llm,
                           prompt_manager=SeedCoderPrompt(seed_code))

        result = dict(arm=arm)
        try:
            updated, suggestion, record = p2_optimize.search_round(
                p2_optimize.make_searcher(llm, arm), UnifiedSketch.from_dict(sketch),
                p2_optimize.build_perf_data(seed_metrics, min_speedup=task.constraints.min_speedup,
                                            history=[]))
            result.update(search=record, suggestion=suggestion)
            code = coder.run(updated, task, optimization_recipes=suggestion)
            (arm_dir / 'candidate.py').write_text(code, encoding='utf-8')
            result['source_hash'] = code_hash(code)
        except (LLMCallRefused, RuntimeError, ValueError) as exc:
            code = None
            result['generation_error'] = type(exc).__name__
        finally:
            write_once(arm_dir / 'model-usage.json', usage())
    write_once(arm_dir / 'generation.json', result)
    return code, result


def run_pilot(*, root, seed_code, sketch, task, preflight, evaluate, clients,
              arm_order=('off', 'on'), binding=None):
    """Run a fresh pair. Injectable boundaries exercise the same path offline."""
    root = Path(root)
    p2_plan.validate_seed(seed_code)
    if freeze_from_task(task).hash != p2_plan.CONTRACT_HASH:
        raise ValueError('P2 task contract drift')
    if tuple(sorted(arm_order)) != ('off', 'on'):
        raise ValueError('P2 requires both arms exactly once')
    root.mkdir(parents=True)
    write_once(root / 'bindings.json', dict(binding or {}, contract_hash=p2_plan.CONTRACT_HASH,
        seed_source_hash=code_hash(seed_code), arm_order=list(arm_order), rounds_per_arm=1))

    def conclude(**facts):
        results = dict(decidable=False)
        results.update(facts, formal_gain_claim=False, arm_order=list(arm_order))
        write_once(root / 'pilot.json', results)
        return results

    def check_window(name):
        try:
            gate = preflight(root / name)
        except (RuntimeError, OSError, TimeoutExpired) as exc:
            raise ValueError(f'P2 preflight did not produce a receipt ({type(exc).__name__})') from exc
        # The CLI writes its own receipt. Injected preflights return one here.
        if not (root / name).exists():
            write_once(root / name, gate)
        p2_plan.gate_band(gate)
        return gate

    try:
        check_window('preflight-start.json')
        seed_receipt = evaluate('seed', seed_code)
        seed_metrics = p2_plan.final_metrics(seed_receipt, source_hash=code_hash(seed_code))
    except ValueError as exc:
        return conclude(verdict='held', stage='seed', reason=str(exc))

    candidates = {}
    for arm in arm_order:
        candidates[arm] = generate_arm(root=root, arm=arm, task=task, sketch=sketch,
                                       seed_code=seed_code, seed_metrics=seed_metrics,
                                       clients=clients)

    # Model calls can outlast the starting window. Probe again immediately
    # before timing the two end products on the same card.
    try:
        gate = check_window('preflight-final.json')
    except ValueError as exc:
        return conclude(verdict='held', stage='final_window', reason=str(exc))
    receipts, hashes, failures = {}, {}, {}
    for arm in arm_order:
        code, record = candidates[arm]
        if code:
            receipts[arm] = evaluate(arm, code)
            hashes[arm] = code_hash(code)
        if not code or (record.get('search') or {}).get('path') != 'llm':
            failures[arm] = 'search_model_path_incomplete'
    if failures:
        return conclude(verdict='unknown', stage='optimization', failures=failures)
    return conclude(**p2_plan.judge_pair(gate=gate, seed_receipt=seed_receipt,
                                          receipts=receipts, source_hashes=hashes))


def live_evaluation(root, args):
    from operator_agent.app import hitl_tools
    from operator_agent.execute.backend_ownership import owned_worker_scope

    def evaluate(name, code):
        run = root / name / 'evaluation'
        hitl_tools.prepare(p2_plan.DECLARATION, run, host_budget=dict(
            wall_clock_s=2 * args.verify_timeout + 60, max_verifications=2,
            max_model_calls=1, max_output_tokens_per_call=8192,
            total_output_token_reservations=8192))

        def verify(purpose, level):
            worker_dir = run / (purpose + '-worker')
            worker_dir.mkdir()
            with owned_worker_scope(worker_dir, uuid.uuid4().hex):
                return hitl_tools.verify(run, code, backend='ssh', host=args.host,
                    work_dir=args.work_dir, timeout=args.verify_timeout, persistent=True,
                    stop_after=level, verification_purpose=purpose)

        try:
            explore = verify('explore', 'L2')
            if not (explore.get('levels') or {}).get('L2'):
                return {}
            final = verify('final', 'L3')
        except hitl_tools.ToolError as exc:
            write_once(run / 'evaluation-error.json', dict(code=exc.code))
            return {}
        host_task = final.get('host_task') or {}
        if host_task.get('phase') != 'final_complete':
            write_once(run / 'evaluation-error.json', dict(code=host_task.get('stop_reason')))
            return {}
        receipt = read_json(final['evidence'])
        return dict(receipt, verification_purpose=final['verification_purpose'])

    return evaluate


def live_preflight(args, gpu_env):
    from .p1_run import run_preflight

    def preflight(path):
        return run_preflight(
            dict(command='python -m operator_agent preflight --backend ssh --host <HOST> '
                         '--op layernorm --shape 32768,6144 --dtype fp16 --reps 5 --repeat 8 '
                         '--planned-effect-pct 2.5'),
            host=args.host, work_dir=args.work_dir, gpu_env=gpu_env, out_path=path)

    return preflight


def wire_policy():
    return dict(sampling.policy(), reasoning_effort='medium')


@contextmanager
def model_client(arm_dir, args):
    wire = wire_policy()
    model_run = arm_dir / 'model'
    config = dict(run_dir=str(model_run), sampling=wire, model=args.model)
    write_once(arm_dir / 'cell-config.json', config)
    task_budget = TaskBudgetConfig(wall_clock_s=1800, max_llm_calls=4, oracle_total=2)
    limits = replace(task_budget.llm_limits(call_seconds=1000), max_output_tokens=wire['max_tokens'],
                     total_output_token_reservations=4 * wire['max_tokens'],
                     allowed_callers=('Search', 'Coder'))
    budget = ProductionLLMCallBudget(limits, model_run / 'journal')
    llm = LLMClient(base_url=args.base_url, api_key=os.environ.get('LLM_API_KEY') or 'EMPTY',
                    model=args.model, timeout=1000, call_budget=budget)
    with sampling.bounded(config):
        yield llm, budget.summary


def main(argv=None):
    from operator_agent.app.hitl_tools import build_task

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed-json', type=Path, required=True)
    parser.add_argument('--host', default='h20')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='Qwen3.8-27B')
    parser.add_argument('--verify-timeout', type=int, default=600)
    parser.add_argument('--formal-plan', type=Path)
    parser.add_argument('--pair-id')
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    if bool(args.formal_plan) != bool(args.pair_id):
        parser.error('--formal-plan and --pair-id must be supplied together')
    seed = read_json(args.seed_json)
    source = Path(seed['source'])
    seed_code = source.read_text(encoding='utf-8')
    if code_hash(seed_code) != seed['source_hash']:
        raise ValueError('P2 seed manifest and source differ')
    gpu_env = dict(OPERATOR_AGENT_REQUIRED_GPU=str(args.gpu),
                   OPERATOR_AGENT_REQUIRED_GPU_UUID=args.gpu_uuid,
                   OPERATOR_AGENT_REQUIRED_GPU_NAME='NVIDIA H20')
    binding = dict(model=args.model, sampling=wire_policy(),
                   gpu=dict(index=args.gpu, uuid=args.gpu_uuid, name='NVIDIA H20'))
    arm_order = ('off', 'on')
    if args.formal_plan:
        plan = read_json(args.formal_plan)
        problems = p2_plan.validate(plan)
        if problems:
            parser.error('invalid formal plan: ' + '; '.join(problems))
        if args.model != plan['model'] or wire_policy() != plan['sampling'] or binding['gpu'] != plan['gpu']:
            parser.error('runtime model, sampling or GPU differs from the formal plan')
        endpoint = hashlib.sha256(args.base_url.rstrip('/').encode()).hexdigest()
        if endpoint != plan['model_endpoint_sha256']:
            parser.error('runtime model endpoint differs from the formal plan')
        matches = [pair for pair in plan['pairs'] if pair['pair_id'] == args.pair_id]
        if len(matches) != 1:
            parser.error('pair is absent from the formal plan')
        arm_order = tuple(matches[0]['arm_order'])
        binding.update(formal_plan_hash=plan['hash'], formal_pair_id=args.pair_id)
    os.environ.update(gpu_env)
    result = run_pilot(
        root=args.root, seed_code=seed_code, sketch=read_json(source.with_name('sketch.json')),
        task=build_task(p2_plan.DECLARATION), preflight=live_preflight(args, gpu_env),
        evaluate=live_evaluation(args.root, args), clients=lambda arm: model_client(arm, args),
        arm_order=arm_order, binding=binding)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['decidable'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
