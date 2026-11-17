"""D2 首轮执行器：16 点确定性扫描 → 择优 → best/起点独立终评 → Q1 判读。

零模型调用。每点一次 explore 验证（stop_after=L3，完整 L2 + 时延），degraded 或
L2 挂的点记录后出局；择优只认 valid。主张读数来自 best 与起点各自 purpose=final 的
独立终评，同窗噪声门。量具只落判读，formal_gain_claim 恒 False，主张留给人。
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from benchmarks.experiments.phase1.plan import write_once
from operator_agent.core.run_contract import code_hash

from . import d2_plan, d2_rewrite


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def point_label(params):
    return f"bs{params['block_size']}-w{params['num_warps']}"


def run_sweep(*, root, plan, seed_code, preflight, evaluate, binding=None):
    """注入式边界（preflight/evaluate）让同一条路径离线可验。"""
    issues = d2_plan.validate(plan)
    if issues:
        raise ValueError(' / '.join(issues))
    if code_hash(seed_code) != d2_plan.SEED_SOURCE_HASH:
        raise ValueError('D2 seed source differs from the frozen hash')
    root = Path(root)
    root.mkdir(parents=True)
    write_once(root / 'bindings.json', dict(binding or {}, sweep_plan_hash=plan['hash'],
               contract_hash=d2_plan.CONTRACT_HASH, seed_source_hash=code_hash(seed_code),
               gpu=plan['gpu']))
    started = time.monotonic()

    def conclude(**facts):
        results = dict(decidable=False, formal_gain_claim=False)
        results.update(facts)
        write_once(root / 'sweep.json', results)
        return results

    def check_window(name):
        gate = preflight(root / name)
        if not (root / name).exists():
            write_once(root / name, gate)
        d2_plan.gate_band(gate)
        return gate

    def over_deadline():
        return time.monotonic() - started > plan['budget']['campaign_seconds']

    try:
        check_window('preflight-start.json')
    except ValueError as exc:
        return conclude(verdict='held', stage='start_window', reason=str(exc))

    records = []
    for point in plan['points']:
        if over_deadline():
            return conclude(verdict='held', stage='campaign_deadline',
                            completed_points=len(records))
        params = point['params']
        directory = root / 'points' / point_label(params)
        directory.mkdir(parents=True)
        code = d2_rewrite.rewrite(seed_code, **params)
        if code_hash(code) != point['source_hash']:
            raise ValueError(f'{point_label(params)}: rewrite drifted from the frozen grid hash')
        (directory / 'candidate.py').write_text(code, encoding='utf-8')
        receipt = evaluate(point_label(params), code, 'explore')
        record = dict(params=params, source_hash=point['source_hash'], eligible=False,
                      median_ms=None, validity=None)
        if receipt:
            write_once(directory / 'explore.json', receipt)
            try:
                metrics = d2_plan.explore_metrics(receipt, source_hash=point['source_hash'])
                record.update(median_ms=metrics['median_time_ms'],
                              validity=metrics.get('measurement_validity'),
                              eligible=metrics.get('measurement_validity') == 'valid')
                if not record['eligible']:
                    record['reason'] = 'measurement_validity not valid'
            except ValueError as exc:
                record['reason'] = str(exc)
        else:
            record['reason'] = 'explore evaluation produced no receipt'
        write_once(directory / 'point.json', record)
        records.append(record)

    best = d2_plan.select_best(records)
    if best is None:
        return conclude(verdict='unknown', stage='sweep', selected_point=None,
                        eligible_points=0, reason='no eligible sweep point')
    try:
        gate = check_window('preflight-final.json')
    except ValueError as exc:
        return conclude(verdict='held', stage='final_window', reason=str(exc),
                        selected_point=best['params'])
    if over_deadline():
        return conclude(verdict='held', stage='campaign_deadline', selected_point=best['params'])
    best_code = d2_rewrite.rewrite(seed_code, **best['params'])
    best_receipt = evaluate('best', best_code, 'final')
    seed_receipt = evaluate('seed', seed_code, 'final')
    try:
        d2_plan.final_metrics(best_receipt, source_hash=best['source_hash'])
        d2_plan.final_metrics(seed_receipt, source_hash=d2_plan.SEED_SOURCE_HASH)
    except ValueError as exc:
        return conclude(verdict='unknown', stage='final_evaluation', reason=str(exc),
                        selected_point=best['params'])
    judgement = d2_plan.judge_sweep(gate=gate, seed_receipt=seed_receipt,
                                    best_receipt=best_receipt,
                                    best_source_hash=best['source_hash'], plan=plan)
    return conclude(**judgement, selected_point=best['params'],
                    eligible_points=sum(record['eligible'] for record in records))


def live_evaluation(root, args):
    from operator_agent.app import hitl_tools
    from operator_agent.execute.backend_ownership import owned_worker_scope

    def evaluate(name, code, purpose):
        run = root / name / 'evaluation'
        # host 预算是派发上限而非必须用满：max_verifications>=2（explore 槽 + 1 final
        # 保留）、max_model_calls>=1 是 HostTaskLimits 的硬约束。D2 零模型靠「不派发」
        # 保证，不靠把上限设 0；终评内部需 explore(L2)+final(L3) 两次派发（host 契约）。
        hitl_tools.prepare(d2_plan.DECLARATION, run, host_budget=dict(
            wall_clock_s=2 * args.verify_timeout + 60, max_verifications=2,
            max_model_calls=1, max_output_tokens_per_call=8192,
            total_output_token_reservations=8192))

        def verify(p, level):
            worker_dir = run / (p + '-worker')
            worker_dir.mkdir()
            with owned_worker_scope(worker_dir, uuid.uuid4().hex):
                return hitl_tools.verify(run, code, backend='ssh', host=args.host,
                    work_dir=args.work_dir, timeout=args.verify_timeout, persistent=True,
                    stop_after=level, verification_purpose=p)

        try:
            if purpose == 'explore':
                # explore 点单次 explore(L3) 取时延；host task 停在 awaiting_final，
                # final_complete 是终评相位，不在此要求。直接读 explore 证据。
                result = verify('explore', 'L3')
                receipt = read_json(result['evidence'])
                return dict(receipt, verification_purpose=result['verification_purpose'])
            # 终评：host 契约要求同一 task 内先有该源码的 explore L2，再独立 final。
            explore = verify('explore', 'L2')
            if not (explore.get('levels') or {}).get('L2'):
                return {}
            result = verify('final', 'L3')
        except hitl_tools.ToolError as exc:
            write_once(run / 'evaluation-error.json', dict(code=exc.code))
            return {}
        host_task = result.get('host_task') or {}
        if host_task.get('phase') != 'final_complete':
            write_once(run / 'evaluation-error.json', dict(code=host_task.get('stop_reason')))
            return {}
        receipt = read_json(result['evidence'])
        return dict(receipt, verification_purpose=result['verification_purpose'])

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest='command', required=True)
    freeze_parser = subcommands.add_parser('freeze')
    freeze_parser.add_argument('--seed-json', type=Path, required=True)
    freeze_parser.add_argument('--output', type=Path, required=True)
    freeze_parser.add_argument('--gpu', type=int, default=d2_plan.GPU['index'])
    freeze_parser.add_argument('--gpu-uuid', default=d2_plan.GPU['uuid'])
    run_parser = subcommands.add_parser('run')
    run_parser.add_argument('--plan', type=Path, required=True)
    run_parser.add_argument('--seed-json', type=Path, required=True)
    run_parser.add_argument('--root', type=Path, required=True)
    run_parser.add_argument('--host', default='h20')
    run_parser.add_argument('--work-dir', required=True)
    run_parser.add_argument('--gpu', type=int, required=True)
    run_parser.add_argument('--gpu-uuid', required=True)
    run_parser.add_argument('--verify-timeout', type=int, default=600)
    args = parser.parse_args(argv)

    if args.command == 'freeze':
        seed = read_json(args.seed_json)
        seed_code = Path(seed['source']).read_text(encoding='utf-8')
        plan = d2_plan.freeze(seed_code, args.output,
                              gpu=dict(index=args.gpu, uuid=args.gpu_uuid, name='NVIDIA H20'))
        print(plan['hash'])
        return 0

    from .p2_seed_probe import _establish_reference, _set_gpu_env

    plan = read_json(args.plan)
    issues = d2_plan.validate(plan)
    if issues:
        parser.error('invalid sweep plan: ' + '; '.join(issues))
    seed = read_json(args.seed_json)
    seed_code = Path(seed['source']).read_text(encoding='utf-8')
    if code_hash(seed_code) != seed['source_hash']:
        parser.error('seed manifest and source differ')
    gpu = dict(index=args.gpu, uuid=args.gpu_uuid, name='NVIDIA H20')
    if gpu != plan['gpu']:
        parser.error('runtime GPU differs from the frozen plan')
    args.root = args.root.resolve()
    _set_gpu_env(args.gpu, args.gpu_uuid, gpu['name'])
    # 参考标定的 persistent backend 必须被 owned scope 回收：否则其常驻 runner_server
    # 在测量期间作为同卡 harness peer，把每个点判 degraded（自污染，I68）。worker 目录
    # 放在 sweep root 之外，避免 run_sweep 的 root.mkdir 撞已存在目录。
    from operator_agent.execute.backend_ownership import owned_worker_scope
    reference_worker = args.root.with_name(args.root.name + '-reference-worker')
    reference_worker.mkdir(parents=True, exist_ok=True)
    with owned_worker_scope(reference_worker, uuid.uuid4().hex):
        _establish_reference(d2_plan.DECLARATION, args.host, args.work_dir,
                             args.gpu, args.gpu_uuid, gpu['name'])
    gpu_env = dict(OPERATOR_AGENT_REQUIRED_GPU=str(args.gpu),
                   OPERATOR_AGENT_REQUIRED_GPU_UUID=args.gpu_uuid,
                   OPERATOR_AGENT_REQUIRED_GPU_NAME=gpu['name'])
    result = run_sweep(root=args.root, plan=plan, seed_code=seed_code,
                       preflight=live_preflight(args, gpu_env),
                       evaluate=live_evaluation(args.root, args), binding=dict(gpu=gpu))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['decidable'] else 1


if __name__ == '__main__':
    raise SystemExit(main())