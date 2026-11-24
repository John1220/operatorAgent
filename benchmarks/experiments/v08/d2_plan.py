"""D2 首轮（Q1）冻结计划与判读：确定性参数扫描单独能否把冻结起点核改快。

身份与终评校验复用 P2 冻结件（同一颗起点核、同一份 32768×6144 契约）：
CONTRACT_HASH / SEED_SOURCE_HASH / DECLARATION / PROBE_CHANNEL / GPU / 噪声门 / final_metrics。
D2 自己冻的是：16 点扫描格、每点源码哈希、零模型预算、Q1 单比较判读规则。
Q2（拆分因果：sweep-only vs sweep+语义）明确不在本计划。
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from operator_agent.core.run_contract import code_hash, digest

from . import d2_rewrite
from .p2_plan import (CONTRACT_HASH, DECLARATION, EFFECT_THRESHOLD_PCT, GPU, PROBE_CHANNEL,
                      SEED_SOURCE_HASH, SHAPE, final_metrics, gate_band, required_case_ids)

SCHEMA = 'v08-d2-sweep-plan-v1'
BLOCK_SIZES = (256, 512, 1024, 2048)
NUM_WARPS = (2, 4, 8, 16)
SEED_POINT = dict(block_size=512, num_warps=4)
BUDGET = dict(max_points=16, max_verifications_per_point=1, final_evaluations=2,
              max_model_calls=0, verify_timeout_s=600, campaign_seconds=5400)


def grid():
    return d2_rewrite.grid_points(BLOCK_SIZES, NUM_WARPS)


def _gpu_issues(gpu):
    """卡身份是 freeze 时绑定的运行期参数，不再钉死 P2 那张卡；只校验格式自洽。"""
    if (not isinstance(gpu, dict) or type(gpu.get('index')) is not int or gpu['index'] < 0
            or not isinstance(gpu.get('uuid'), str) or not gpu['uuid']
            or not isinstance(gpu.get('name'), str) or not gpu['name']):
        return ['sweep plan requires a well-formed GPU identity (index/uuid/name)']
    return []


def freeze(seed_code, output, *, gpu=GPU):
    """本地冻结：16 点机械改写 + 哈希。不触 GPU、不调模型。

    标定不变量：种子点 (512,4) 的改写必须逐字还原起点源码——改写器若连原值都
    还原不了，其余 15 点的差分不可信。gpu 默认沿用 P2 的测量卡，可显式换卡；
    换卡只改 D2 自己的绑定，P2 的冻结身份不受影响。
    """
    from benchmarks.experiments.phase1.plan import write_once

    gpu_issues = _gpu_issues(gpu)
    if gpu_issues:
        raise ValueError(' / '.join(gpu_issues))
    gpu = dict(gpu)
    if code_hash(seed_code) != SEED_SOURCE_HASH:
        raise ValueError('D2 starting source does not match its frozen hash')
    required_case_ids()
    points = []
    for params in grid():
        source = d2_rewrite.rewrite(seed_code, **params)
        points.append(dict(params=params, source_hash=code_hash(source)))
    if len({point['source_hash'] for point in points}) != len(points):
        raise ValueError('D2 grid points must have distinct source hashes')
    seed_point = next(point for point in points if point['params'] == SEED_POINT)
    if seed_point['source_hash'] != SEED_SOURCE_HASH:
        raise ValueError('D2 seed-point rewrite must reproduce the frozen source byte-for-byte')
    if d2_rewrite.rewrite(seed_code, **SEED_POINT) != seed_code:
        raise ValueError('D2 seed-point rewrite is not byte-identical')
    plan = dict(schema=SCHEMA, status='frozen', declaration=DECLARATION,
                contract_hash=CONTRACT_HASH, seed_source_hash=SEED_SOURCE_HASH,
                probe_channel=PROBE_CHANNEL, effect_threshold_pct=EFFECT_THRESHOLD_PCT,
                gpu=gpu, budget=BUDGET, seed_point=SEED_POINT, points=points,
                formal_gain_claim=False,
                frozen_at=datetime.now(timezone.utc).timestamp())
    plan = deepcopy(plan)
    plan['hash'] = digest(plan)
    write_once(Path(output), plan)
    return plan


def validate(plan):
    if not isinstance(plan, dict) or plan.get('schema') != SCHEMA:
        return ['expected a D2 sweep plan']
    fixed = dict(status='frozen', declaration=DECLARATION, contract_hash=CONTRACT_HASH,
                 seed_source_hash=SEED_SOURCE_HASH, probe_channel=PROBE_CHANNEL,
                 effect_threshold_pct=EFFECT_THRESHOLD_PCT, budget=BUDGET,
                 seed_point=SEED_POINT, formal_gain_claim=False)
    issues = [f'sweep plan field differs: {key}' for key, value in fixed.items()
              if plan.get(key) != value]
    issues.extend(_gpu_issues(plan.get('gpu')))
    points = plan.get('points')
    if not isinstance(points, list) or len(points) != BUDGET['max_points']:
        issues.append('sweep plan must carry exactly 16 points')
    else:
        if [point.get('params') for point in points] != grid():
            issues.append('sweep grid order differs from the registered enumeration')
        hashes = [point.get('source_hash') for point in points]
        if len(set(hashes)) != len(hashes) or any(
                not isinstance(h, str) or len(h) != 64
                or any(ch not in '0123456789abcdef' for ch in h) for h in hashes):
            issues.append('sweep point hashes must be distinct sha256 strings')
        seed_point = next((point for point in points if point.get('params') == SEED_POINT), None)
        if not seed_point or seed_point.get('source_hash') != SEED_SOURCE_HASH:
            issues.append('seed point must carry the frozen seed hash')
    if not isinstance(plan.get('frozen_at'), (int, float)) or plan['frozen_at'] <= 0:
        issues.append('sweep plan requires a freeze time')
    if plan.get('hash') != digest({k: v for k, v in plan.items() if k != 'hash'}):
        issues.append('sweep plan hash differs')
    return issues


def explore_metrics(receipt, *, source_hash):
    """扫描点的 explore 级读数：完整 L2 + L3 时延。有效性单独报，不在此拒。

    与终评不同：explore 允许 degraded（记录后出局），但 L2 必须全过、中位数必须可用。
    """
    if (not isinstance(receipt, dict) or receipt.get('ok') is not True
            or receipt.get('verification_purpose') != 'explore' or receipt.get('stop_after') != 'L3'
            or receipt.get('contract_hash') != CONTRACT_HASH
            or receipt.get('source_hash') != source_hash):
        raise ValueError('D2 explore evidence differs from the frozen contract or point source')
    levels = receipt.get('levels') or {}
    verdicts = receipt.get('verdicts')
    if not isinstance(verdicts, list) or levels.get('L1') is not True or levels.get('L2') is not True:
        raise ValueError('D2 explore evidence must pass L1 and the full L2 suite')
    l2 = [row for row in verdicts if isinstance(row, dict) and row.get('level') == 'L2']
    if len(l2) != 1 or l2[0].get('passed') is not True:
        raise ValueError('D2 explore evidence must carry one passed L2 verdict')
    correctness = l2[0].get('metrics', {}).get('correctness') or {}
    cases = correctness.get('cases')
    if (correctness.get('verdict') != 'passed' or not isinstance(cases, list)
            or not all(isinstance(case, dict) and case.get('verdict') == 'passed' for case in cases)
            or tuple(sorted(case.get('case_id') for case in cases)) != required_case_ids()):
        raise ValueError('D2 explore evidence did not pass the full bound L2 suite')
    l3 = [row for row in verdicts if isinstance(row, dict) and row.get('level') == 'L3']
    if len(l3) != 1 or not isinstance(l3[0].get('metrics'), dict):
        raise ValueError('D2 explore evidence must carry one L3 verdict with metrics')
    metrics = l3[0]['metrics']
    median = metrics.get('median_time_ms')
    if (type(median) not in (int, float) or not math.isfinite(median) or median <= 0
            or metrics.get('baseline_type') != 'pytorch_eager'):
        raise ValueError('D2 explore evidence requires a finite positive median against pytorch_eager')
    return metrics


def select_best(point_records):
    """合格点 = 完整 L2 + measurement_validity=valid；按中位时延择优，打平取枚举序先。"""
    eligible = [record for record in point_records
                if record.get('eligible') and isinstance(record.get('median_ms'), (int, float))]
    if not eligible:
        return None
    return min(eligible, key=lambda record: record['median_ms'])


def judge_sweep(*, gate, seed_receipt, best_receipt, best_source_hash, plan):
    """Q1 单比较：best 点与起点核各自独立终评，同窗配对。"""
    result = dict(decidable=False, verdict='not_decidable', difference_pct=None, band_pct=None,
                  effect_threshold_pct=EFFECT_THRESHOLD_PCT, formal_gain_claim=False, reasons=[])
    try:
        result['band_pct'] = gate_band(gate)
    except ValueError as error:
        result['reasons'].append(str(error))
        return result
    grid_hashes = {point['source_hash'] for point in plan['points']}
    if best_source_hash not in grid_hashes:
        result['reasons'].append('D2 best source is not a registered grid point')
        return result
    readings = {}
    for name, receipt, expected in (('seed', seed_receipt, SEED_SOURCE_HASH),
                                    ('best', best_receipt, best_source_hash)):
        try:
            readings[name] = final_metrics(receipt, source_hash=expected)['median_time_ms']
        except ValueError as error:
            result['reasons'].append(f'{name}: {error}')
    if result['reasons']:
        return result
    if seed_receipt['evidence'] == best_receipt['evidence']:
        result['reasons'].append('D2 seed and best must carry distinct final evaluations')
        return result
    seed = Decimal(str(readings['seed']))
    decrease = (seed - Decimal(str(readings['best']))) * 100
    difference = float(decrease / seed)
    if not math.isfinite(difference):
        result['reasons'].append('D2 measured difference is nonfinite')
        return result
    band = seed * Decimal(str(result['band_pct']))
    effect = seed * Decimal(str(EFFECT_THRESHOLD_PCT))
    verdict = 'no_increment'
    if decrease >= effect and decrease > band:
        verdict = 'sweep_helps'
    elif decrease < -band:
        verdict = 'sweep_hurts'
    result.update(decidable=True, verdict=verdict, difference_pct=difference,
                  seed_median_ms=readings['seed'], best_median_ms=readings['best'],
                  best_is_seed_point=best_source_hash == SEED_SOURCE_HASH)
    return result


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def _budget_records(evaluation):
    """预算记录**只从任务账本读**：最后一条带 `budget.host_task` 的事件就是现值。"""
    from operator_agent.core import task_journal
    state = task_journal.fold(task_journal.read_events(evaluation))['budget'].get('host_task') or {}
    return state.get('records') or []

def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def replay_sweep(path, plan):
    """从落盘证据复算 Q1 判读：逐点哈希/选择/终评绑定/零模型调用全部重验。"""
    issues = validate(plan)
    if issues:
        raise ValueError(' / '.join(issues))
    root = Path(path).parent.resolve()
    summary = _read(path)
    bindings = _read(root / 'bindings.json')
    expected = dict(sweep_plan_hash=plan['hash'], contract_hash=CONTRACT_HASH,
                    seed_source_hash=SEED_SOURCE_HASH, gpu=plan['gpu'])
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise ValueError('D2 sweep bindings differ from the frozen plan')
    if summary.get('formal_gain_claim') is not False:
        raise ValueError('D2 sweep record must not grant formal credit')
    if list(root.glob('points/*/model')) or list(root.glob('*/model')):
        raise ValueError('D2 is a zero-model-call instrument; model dispatch evidence found')

    records = []
    hashes = {'bindings.json': _sha(root / 'bindings.json')}
    for point in plan['points']:
        label = f"bs{point['params']['block_size']}-w{point['params']['num_warps']}"
        directory = root / 'points' / label
        candidate = directory / 'candidate.py'
        if _sha(candidate) != point['source_hash']:
            raise ValueError(f'{label}: candidate source differs from the frozen grid hash')
        record = _read(directory / 'point.json')
        if (record.get('params') != point['params']
                or record.get('source_hash') != point['source_hash']):
            raise ValueError(f'{label}: point record differs from the frozen grid')
        receipt_path = directory / 'explore.json'
        if receipt_path.exists():
            receipt = _read(receipt_path)
            if (receipt.get('source_hash') != point['source_hash']
                    or receipt.get('contract_hash') != CONTRACT_HASH):
                raise ValueError(f'{label}: explore receipt binding differs')
            hashes[str(receipt_path.relative_to(root))] = _sha(receipt_path)
        records.append(record)
        hashes[str(candidate.relative_to(root))] = _sha(candidate)
        hashes[str((directory / 'point.json').relative_to(root))] = _sha(directory / 'point.json')
    best = select_best(records)
    if summary.get('decidable') is not True:
        return dict(decidable=False, verdict=summary.get('verdict', 'unknown'),
                    reported_verdict=summary.get('verdict'), artifact_sha256=hashes)
    if best is None or summary.get('selected_point') != best['params']:
        raise ValueError('D2 saved selection differs from recomputed eligible argmin')

    gate_path = root / 'preflight-final.json'
    gate = _read(gate_path)
    gate_band(gate)
    gate_time = datetime.fromisoformat(gate['created_at']).timestamp()
    if gate_time <= plan['frozen_at']:
        raise ValueError('D2 final gate predates the freeze')
    hashes['preflight-final.json'] = _sha(gate_path)
    finals = {}
    for name, expected_hash in (('best', best['source_hash']), ('seed', SEED_SOURCE_HASH)):
        evaluation = root / name / 'evaluation'
        budget_records = _budget_records(evaluation)
        # host 契约：终评 task 内 explore(L2) 先行（verify-1）+ 独立 final(L3)（verify-2）。
        explores = [r for r in budget_records if r.get('purpose') == 'explore']
        final_records = [r for r in budget_records if r.get('purpose') == 'final']
        if len(budget_records) != 2 or len(explores) != 1 or len(final_records) != 1:
            raise ValueError(f'{name}: final host task must be explore(L2)+final(L3)')
        explore_rec, final_rec = explores[0], final_records[0]
        final_path = evaluation / 'verify-2' / 'result.json'
        candidate_path = final_path.with_name('candidate.py')
        manifest = _read(final_path.with_name('complete.json'))
        receipt = _read(final_path)
        source_hash = _sha(candidate_path)
        if (source_hash != expected_hash
                or explore_rec.get('status') != 'completed'
                or explore_rec.get('full_l2_passed') is not True
                or explore_rec.get('source_hash') != source_hash
                or final_rec.get('status') != 'completed'
                or final_rec.get('full_l2_passed') is not True
                or final_rec.get('finished_after_deadline') is not False
                or final_rec.get('evidence') != str(final_path)
                or final_rec.get('source_hash') != source_hash
                or manifest.get('evidence_complete') is not True
                or manifest.get('binding') != receipt.get('binding')
                or manifest['files_sha256'].get('result.json') != _sha(final_path)
                or manifest['files_sha256'].get('candidate.py') != source_hash):
            raise ValueError(f'{name}: independent final evidence binding differs')
        receipt['verification_purpose'] = 'final'
        metrics = final_metrics(receipt, source_hash=source_hash)
        if (metrics.get('selected_gpu') != plan['gpu']['index'] or receipt.get('backend') != 'ssh'
                or not isinstance(receipt.get('created_at'), (int, float))
                or receipt['created_at'] <= gate_time):
            raise ValueError(f'{name}: final evaluation GPU/backend/window differs')
        finals[name] = receipt
        for member in (evaluation / 'journal.jsonl', candidate_path, final_path,
                       final_path.with_name('complete.json')):
            hashes[str(member.relative_to(root))] = _sha(member)
    judgement = judge_sweep(gate=gate, seed_receipt=finals['seed'], best_receipt=finals['best'],
                            best_source_hash=best['source_hash'], plan=plan)
    if not judgement['decidable'] or any(summary.get(key) != judgement[key]
                                         for key in ('verdict', 'band_pct', 'seed_median_ms',
                                                     'best_median_ms', 'difference_pct')):
        raise ValueError('D2 saved judgment differs from replay')
    return dict(judgement, artifact_sha256=hashes)


def main(argv=None):
    import argparse
    from benchmarks.experiments.phase1.plan import write_once

    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    freeze_parser = subcommands.add_parser('freeze')
    freeze_parser.add_argument('--seed-json', type=Path, required=True)
    freeze_parser.add_argument('--output', type=Path, required=True)
    freeze_parser.add_argument('--gpu', type=int, default=GPU['index'])
    freeze_parser.add_argument('--gpu-uuid', default=GPU['uuid'])
    judge_parser = subcommands.add_parser('judge')
    judge_parser.add_argument('--plan', type=Path, required=True)
    judge_parser.add_argument('--sweep', type=Path, required=True)
    judge_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        seed = _read(args.seed_json)
        seed_code = Path(seed['source']).read_text(encoding='utf-8')
        plan = freeze(seed_code, args.output,
                      gpu=dict(index=args.gpu, uuid=args.gpu_uuid, name='NVIDIA H20'))
        print(plan['hash'])
    else:
        result = replay_sweep(args.sweep, _read(args.plan))
        write_once(args.output, result)
        print(result.get('verdict'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())