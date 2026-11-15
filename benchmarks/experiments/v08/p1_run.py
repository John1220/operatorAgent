"""P1 measurement runner: execute the frozen parity plan on a real backend.

No generation, no model calls. For each registered pair, re-run an independent L3
(`purpose=final`, `stop_after=L3`) on each arm's **bound delivered source** under the same
frozen contract, then judge parity with `p1_plan.judge_pair` / `judge_series`. Probes are
re-run in this window (never inherited). Readings are taken here, not invented.

Evidence layout under --root:
  gate-<task>.json          one preflight receipt per task (this window)
  <pair_id>/{A,C}/          independent final-evaluation run dirs per arm
  <pair_id>/pair.json       bound hashes, both L3 readings, judge_pair verdict
  series.json               judge_series verdict + plan hash + GPU identity
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from . import p1_plan

AC_DIR_NAME = 'v08-ac-20261108'


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def resolve_source(pair, arm, ac_root):
    """Read the arm's delivered source from the evidence tree; verify the bound hash.

    The cell receipt stores an absolute source path; we re-anchor it under --ac-root by
    the experiment directory name, so no machine-specific absolute path is hardcoded.
    """
    ac_root = Path(ac_root)
    cell = json.loads((ac_root / f'{pair["round"]}-{pair["task_id"]}-{arm}.json').read_text())
    source = cell.get('source')
    if not source or AC_DIR_NAME not in source:
        raise ValueError(f'pair {pair["pair_id"]} arm {arm}: no resolvable delivered source')
    local = ac_root / source.split(AC_DIR_NAME + '/', 1)[1]
    data = local.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    bound = pair[f'{arm}_source_sha256']
    if actual != bound:
        raise ValueError(f'pair {pair["pair_id"]} arm {arm}: source hash {actual[:12]} '
                         f'!= bound {bound[:12]} (refusing to measure a different source)')
    return data.decode('utf-8'), actual


def run_preflight(probe, *, host, work_dir, gpu_env, out_path):
    """Re-run this task's probe in the current window via the operator CLI."""
    argv = shlex.split(probe['command'])
    argv[0] = sys.executable                       # 'python' -> this interpreter
    argv = [host if tok == '<HOST>' else tok for tok in argv]
    argv += ['--work-dir', work_dir, '--json-out', str(out_path)]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **gpu_env)
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=1200)
    gate = json.loads(Path(out_path).read_text()) if Path(out_path).is_file() else {}
    if not gate:
        raise RuntimeError(f'preflight produced no receipt: {proc.stderr[-800:]}')
    return gate


def _l2_status(receipt):
    """L2 正确性裁决：passed/failed/unknown/None。unknown 多为传输设施抖动，不是候选错。"""
    for verdict in (receipt or {}).get('verdicts') or []:
        if isinstance(verdict, dict) and verdict.get('level') == 'L2':
            correctness = (verdict.get('metrics') or {}).get('correctness') or {}
            return correctness.get('verdict')
    return None


def measure_arm(pair, arm, task, code, *, host, work_dir, run_dir, backend, host_budget,
                attempts=3):
    """Independent final L3 on one arm's bound source; returns the final receipt.

    终评的 L2 若判 unknown（传输设施抖动，如「传输结果未知」），换新 run_dir 重试，
    最多 attempts 次；passed/failed 是决定性裁决，立即返回。候选是已交付核，
    复测里的 unknown 按设施失败重试，不记成臂的损失（与 A/C 轮同一纪律）。
    """
    from .ac_driver import final_evaluation
    last = {}
    for index in range(attempts):
        rd = Path(run_dir) if index == 0 else Path(f'{run_dir}-retry{index}')
        result = final_evaluation(rd, code, declaration=task['declaration'],
                                  host_budget=host_budget, backend=backend, stop_after='L3',
                                  host=host, work_dir=work_dir)
        final = result.get('final') or {}
        # 契约必须等于交付那份（否则测的不是同一判据）。
        contract = json.loads((rd / 'contract.json').read_text()).get('hash')
        if contract != pair['contract_hash']:
            raise ValueError(f'pair {pair["pair_id"]} arm {arm}: re-frozen contract {contract} '
                             f'!= delivered {pair["contract_hash"]}')
        last = final
        if _l2_status(final) in ('passed', 'failed'):
            return final
        print(f'[p1] {pair["pair_id"]}/{arm} attempt {index + 1}: L2 unknown '
              f'(transport), retrying', flush=True)
    return last


def extract_l3(receipt):
    """(median_ms, validity, speedup) from a final receipt; missing => (None, None, None)."""
    validity = receipt.get('validity')
    for verdict in receipt.get('verdicts') or []:
        if isinstance(verdict, dict) and verdict.get('level') == 'L3':
            metrics = verdict.get('metrics') or {}
            return (metrics.get('median_time_ms'),
                    validity or metrics.get('measurement_validity'),
                    metrics.get('speedup_vs_baseline'))
    return None, validity, None


def _arm_budget(wall_clock_s):
    # 每臂一次探索 L3 + 一次独立终评；P1 不调模型，但 prepare 仍需完整的五项预算形状。
    return dict(wall_clock_s=wall_clock_s, max_verifications=2, max_model_calls=1,
                max_output_tokens_per_call=8192, total_output_token_reservations=8192)


def run(plan, *, host, work_dir, ac_root, root, backend='ssh', gpu_env=None):
    problems = p1_plan.validate(plan)
    if problems:
        raise ValueError('plan not executable: ' + '; '.join(problems))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    gpu_env = dict(gpu_env or {})
    tasks = {row['task_id']: row for row in plan['tasks']}
    wall = int(plan['budget']['wall_clock_s'])

    gates = {}
    for task_id, probe in plan['probes'].items():
        gate_path = root / f'gate-{task_id}.json'
        gate = run_preflight(probe, host=host, work_dir=work_dir, gpu_env=gpu_env,
                             out_path=gate_path)
        gates[task_id] = gate
        print(f'[p1] gate {task_id}: action={gate.get("action")} '
              f'band={gate.get("effective_band_pct")}', flush=True)
        if gate.get('action') != 'start':
            # 窗口不安静：保留回执，整轮记 unknown，不开测（不耗预算）。
            series = p1_plan.judge_series([])
            series.update(plan_id=plan['plan_id'], plan_hash=plan.get('hash'),
                          held_task=task_id, gate_action=gate.get('action'))
            _write_json(root / 'series.json', series)
            return series

    pair_verdicts = []
    for pair in plan['pairs']:
        task = tasks[pair['task_id']]
        band = float(plan['probes'][pair['task_id']]['observed_band_pct'])
        readings = {}
        for arm in ('A', 'C'):
            code, source_hash = resolve_source(pair, arm, ac_root)
            run_dir = root / pair['pair_id'] / arm
            final = measure_arm(pair, arm, task, code, host=host, work_dir=work_dir,
                                run_dir=run_dir, backend=backend, host_budget=_arm_budget(wall))
            median, validity, speedup = extract_l3(final)
            readings[arm] = dict(source_hash=source_hash, median_time_ms=median,
                                 validity=validity, speedup_vs_baseline=speedup,
                                 full_l2_passed=bool((final.get('levels') or {}).get('L2')),
                                 evidence=final.get('evidence'))
            print(f'[p1] {pair["pair_id"]}/{arm}: median={median} validity={validity} '
                  f'l2={readings[arm]["full_l2_passed"]}', flush=True)
        verdict = p1_plan.judge_pair(
            a_median_ms=readings['A']['median_time_ms'], c_median_ms=readings['C']['median_time_ms'],
            a_valid=readings['A']['validity'] == 'valid', c_valid=readings['C']['validity'] == 'valid',
            band_pct=band)
        pair_record = dict(pair_id=pair['pair_id'], task_id=pair['task_id'], round=pair['round'],
                           contract_hash=pair['contract_hash'], band_pct=band,
                           arms=readings, judgment=verdict)
        _write_json(root / pair['pair_id'] / 'pair.json', pair_record)
        print(f'[p1] {pair["pair_id"]}: {verdict["verdict"]}', flush=True)
        pair_verdicts.append(verdict)

    series = p1_plan.judge_series(pair_verdicts)
    series.update(plan_id=plan['plan_id'], plan_hash=plan.get('hash'),
                  gpu_identity=gpu_env or None,
                  gates={t: dict(action=g.get('action'),
                                 effective_band_pct=g.get('effective_band_pct'),
                                 channel=g.get('channel')) for t, g in gates.items()})
    _write_json(root / 'series.json', series)
    return series


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p1_run', description=__doc__.splitlines()[0])
    parser.add_argument('--plan', required=True, help='冻结的 P1 计划 JSON')
    parser.add_argument('--host', default='h20')
    parser.add_argument('--work-dir', required=True, help='远端私有工作目录')
    parser.add_argument('--ac-root', default=f'output/{AC_DIR_NAME}',
                        help='v6 交付证据目录（含各格回执与源码）')
    parser.add_argument('--root', required=True, help='本轮证据输出目录')
    parser.add_argument('--backend', default='ssh', choices=['ssh', 'mock', 'local', 'fake'])
    parser.add_argument('--gpu', default=os.environ.get('OPERATOR_AGENT_REQUIRED_GPU', '3'))
    parser.add_argument('--gpu-uuid', default=os.environ.get('OPERATOR_AGENT_REQUIRED_GPU_UUID'))
    parser.add_argument('--gpu-name', default=os.environ.get('OPERATOR_AGENT_REQUIRED_GPU_NAME', 'NVIDIA H20'))
    args = parser.parse_args(argv)

    plan = json.loads(Path(args.plan).read_text())
    gpu_env = {'OPERATOR_AGENT_REQUIRED_GPU': str(args.gpu)}
    if args.gpu_uuid:
        gpu_env['OPERATOR_AGENT_REQUIRED_GPU_UUID'] = args.gpu_uuid
    if args.gpu_name:
        gpu_env['OPERATOR_AGENT_REQUIRED_GPU_NAME'] = args.gpu_name
    os.environ.update(gpu_env)
    series = run(plan, host=args.host, work_dir=args.work_dir, ac_root=args.ac_root,
                 root=args.root, backend=args.backend, gpu_env=gpu_env)
    print(json.dumps(series, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())