"""Frozen identities and the P2 pilot's paired-latency judge."""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from operator_agent.core.run_contract import code_hash, digest
from operator_agent.decide.noise_preflight import PROTOCOL

SHAPE = (32768, 6144)
DECLARATION = dict(operator='layernorm', request='fp16 32768x6144 layernorm', shapes=[list(SHAPE)],
                   precision='fp16', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0)
CONTRACT_HASH = 'cc9122d1dd0f926e8c5060f03b0f4ea0cdff20d2347db658a03285e8bc6a22b0'
SEED_SOURCE_HASH = 'cb851b67bc11530fe4da0b7043f5aaf2fb296c95dedd4b529409bff60d65ee96'
PROBE_CHANNEL = 'layernorm-32768x6144-fp16-cuda-event'
EFFECT_THRESHOLD_PCT = 2.5
MODEL = 'Qwen3.8-27B'
SAMPLING = dict(schema='phase1-wire-policy-v1', temperature=0.1, max_tokens=32768,
                n=1, enable_thinking=True, reasoning_effort='medium')
BUDGET = dict(max_calls=4, max_output_tokens=32768, total_output_token_reservations=131072,
              call_seconds=1000, campaign_seconds=1800, rounds_per_arm=1,
              max_verifications_per_candidate=2)
GPU = dict(index=3, uuid='GPU-634bbc76-9ab6-220a-48a8-27247d7acbe2', name='NVIDIA H20')
FORMAL_SCHEMA = 'v08-p2-formal-plan-v1'
PAIRS = [{'pair_id': 'pair-1', 'arm_order': ['off', 'on']},
         {'pair_id': 'pair-2', 'arm_order': ['on', 'off']}]


@lru_cache(maxsize=1)
def required_case_ids():
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task

    contract = freeze_from_task(build_task(DECLARATION))
    if contract.hash != CONTRACT_HASH:
        raise ValueError('P2 declaration no longer reproduces the registered contract')
    return tuple(sorted(case.case_id for case in contract.correctness_suite.cases))


def validate_seed(code: str):
    """Verify the original source under the re-frozen larger-shape contract."""
    required_case_ids()
    if code_hash(code) != SEED_SOURCE_HASH:
        raise ValueError('P2 starting source does not match its frozen hash')


def _positive_number(value):
    if type(value) not in (float, int) or value <= 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def gate_band(gate):
    """Read the effective noise band from a passing GPU-window gate."""
    if not isinstance(gate, dict) or gate.get('schema') != PROTOCOL:
        raise ValueError('P2 requires a noise-preflight receipt')
    identity = gate.get('identity') or {}
    if (gate.get('channel') != PROBE_CHANNEL or not isinstance(identity, dict)
            or identity.get('shape') != list(SHAPE) or identity.get('backend') != 'ssh'
            or gate.get('test_only') is True):
        raise ValueError('P2 requires the 32768x6144 LayerNorm fp16 CUDA-event probe')
    if gate.get('ok') is not True or gate.get('action') != 'start':
        raise ValueError('P2 noise preflight did not permit starting this window')
    band = gate.get('effective_band_pct')
    if (gate.get('planned_effect_pct') != EFFECT_THRESHOLD_PCT
            or not _positive_number(band) or band >= EFFECT_THRESHOLD_PCT):
        raise ValueError('P2 gate band must be finite, positive and below the registered 2.5% effect')
    return float(band)


def _verdict(receipt, level):
    verdicts = receipt.get('verdicts')
    if not isinstance(verdicts, list):
        raise ValueError('P2 final evidence must carry a list of verdicts')
    matches = [row for row in verdicts
               if isinstance(row, dict) and row.get('level') == level]
    if len(matches) != 1 or not isinstance(matches[0].get('metrics'), dict):
        raise ValueError(f'P2 evaluation must carry one {level} verdict with metrics')
    return matches[0]


def final_metrics(receipt, *, source_hash):
    """Require independent final evidence for exactly the submitted source."""
    if (not isinstance(receipt, dict) or receipt.get('ok') is not True
            or receipt.get('verification_purpose') != 'final' or receipt.get('stop_after') != 'L3'
            or receipt.get('evidence_complete') is not True or receipt.get('cached_from')
            or not isinstance(receipt.get('evidence'), str) or not receipt['evidence']):
        raise ValueError('P2 requires completed independent final L3 evidence')
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(char not in '0123456789abcdef' for char in source_hash)
            or receipt.get('contract_hash') != CONTRACT_HASH or receipt.get('source_hash') != source_hash):
        raise ValueError('P2 final evidence differs from the frozen contract or candidate source')
    levels = receipt.get('levels') or {}
    l2 = _verdict(receipt, 'L2')
    correctness = l2['metrics'].get('correctness') or {}
    cases = correctness.get('cases') if isinstance(correctness, dict) else None
    if (not isinstance(levels, dict) or levels.get('L1') is not True or levels.get('L2') is not True
            or l2.get('passed') is not True or not isinstance(cases, list)
            or correctness.get('verdict') != 'passed'
            or not all(isinstance(case, dict) and isinstance(case.get('case_id'), str)
                       and case.get('verdict') == 'passed' for case in cases)
            or tuple(sorted(case['case_id'] for case in cases)) != required_case_ids()):
        raise ValueError('P2 final evidence did not pass the full bound L2 suite')
    metrics = _verdict(receipt, 'L3')['metrics']
    if (metrics.get('measurement_validity') != 'valid'
            or receipt.get('validity') != 'valid' or metrics.get('baseline_type') != 'pytorch_eager'
            or not _positive_number(metrics.get('median_time_ms'))):
        raise ValueError('P2 requires valid L3 with a finite positive candidate median against pytorch_eager')
    return metrics


def judge_pair(*, gate, seed_receipt, receipts, source_hashes):
    """Judge a pilot pair using the effective band from the paired window."""
    result = dict(
        decidable=False, verdict='not_decidable', difference_pct=None, band_pct=None,
        effect_threshold_pct=EFFECT_THRESHOLD_PCT, formal_gain_claim=False, reasons=[])
    try:
        result['band_pct'] = gate_band(gate)
    except ValueError as error:
        result['reasons'].append(str(error))
        return result
    if (not isinstance(receipts, dict) or set(receipts) != {'on', 'off'}
            or not isinstance(source_hashes, dict) or set(source_hashes) != {'on', 'off'}):
        result['reasons'].append('P2 requires exactly one on and one off arm')
        return result

    readings = {}
    all_receipts = dict(seed=seed_receipt, **receipts)
    all_hashes = dict(seed=SEED_SOURCE_HASH, **source_hashes)
    for arm, receipt in all_receipts.items():
        try:
            readings[arm] = final_metrics(receipt, source_hash=all_hashes[arm])['median_time_ms']
        except ValueError as error:
            result['reasons'].append(f'{arm}: {error}')
    if result['reasons']:
        return result
    if len({receipt['evidence'] for receipt in all_receipts.values()}) != len(all_receipts):
        result['reasons'].append('P2 seed and arms must carry distinct final evaluations')
        return result

    off = Decimal(str(readings['off']))
    # Keep the recorded decimal medians exact for inclusive threshold and strict band checks.
    decrease = (off - Decimal(str(readings['on']))) * 100
    difference = float(decrease / off)
    if not math.isfinite(difference):
        result['reasons'].append('P2 measured pair difference is nonfinite')
        return result
    band = off * Decimal(str(result['band_pct']))
    effect = off * Decimal(str(EFFECT_THRESHOLD_PCT))
    verdict = 'no_increment'
    if decrease >= effect and decrease > band:
        verdict = 'optimize_helps'
    elif decrease < -band:
        verdict = 'optimize_hurts'
    seed = Decimal(str(readings['seed']))
    seed_deltas = {arm: float((seed - Decimal(str(readings[arm]))) / seed * 100)
                   for arm in ('on', 'off')}
    if not all(math.isfinite(value) for value in seed_deltas.values()):
        result['reasons'].append('P2 measured seed difference is nonfinite')
        return result
    result.update(decidable=True, verdict=verdict, difference_pct=difference,
                  on_median_ms=readings['on'], off_median_ms=readings['off'],
                  seed_median_ms=readings['seed'], improvement_vs_seed_pct=seed_deltas)
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


def _replay_pair(path, *, gpu_index, after=None, endpoint=None):
    """Recompute a pair from its evidence directory, including actual dispatches."""
    root = path.parent
    gate_path = root / 'preflight-final.json'
    gate = _read(gate_path)
    gate_band(gate)
    gate_time = datetime.fromisoformat(gate['created_at']).timestamp()
    if after is not None and gate_time <= after:
        raise ValueError('P2 formal gate predates the freeze')
    hashes = {path.name: _sha(path), gate_path.name: _sha(gate_path)}
    finals, sources, endpoints = {}, {}, set()
    for arm in ('seed', 'off', 'on'):
        evaluation = root / arm / 'evaluation'
        final_path = evaluation / 'verify-2' / 'result.json'
        candidate_path = final_path.with_name('candidate.py')
        manifest = _read(final_path.with_name('complete.json'))
        receipt = _read(final_path)
        source_hash = _sha(candidate_path)
        result_hash = _sha(final_path)
        records = _budget_records(evaluation)
        if len(records) != BUDGET['max_verifications_per_candidate']:
            raise ValueError(f'{arm}: requires one explore and one final evaluation')
        terminal = records[-1]
        if (records[0].get('purpose') != 'explore'
                or terminal.get('purpose') != 'final' or terminal.get('status') != 'completed'
                or terminal.get('finished_after_deadline') is not False
                or terminal.get('full_l2_passed') is not True
                or terminal.get('evidence') != receipt.get('evidence')
                or terminal.get('source_hash') != source_hash
                or manifest.get('evidence_complete') is not True
                or manifest.get('binding') != receipt.get('binding')
                or manifest['files_sha256'].get('result.json') != result_hash
                or manifest['files_sha256'].get('candidate.py') != source_hash):
            raise ValueError(f'{arm}: independent final evidence binding differs')
        receipt['verification_purpose'] = 'final'
        metrics = final_metrics(receipt, source_hash=source_hash)
        if metrics.get('selected_gpu') != gpu_index or receipt.get('backend') != 'ssh':
            raise ValueError(f'{arm}: final evaluation used a different GPU/backend')
        if after is not None and (not _positive_number(receipt.get('created_at'))
                                  or receipt['created_at'] <= after):
            raise ValueError(f'{arm}: formal evaluation predates the freeze')
        if arm != 'seed' and receipt.get('created_at', 0) < gate_time:
            raise ValueError(f'{arm}: final evaluation predates the pair window gate')
        finals[arm] = receipt
        sources[arm] = source_hash
        for member in (evaluation / 'journal.jsonl', candidate_path, final_path,
                       final_path.with_name('complete.json')):
            hashes[str(member.relative_to(root))] = _sha(member)
        if arm == 'seed':
            continue

        directory = root / arm
        generation = _read(directory / 'generation.json')
        if (generation.get('source_hash') != source_hash
                or (generation.get('search') or {}).get('path') != 'llm'
                or _sha(directory / 'candidate.py') != source_hash):
            raise ValueError(f'{arm}: generated source and Search dispatch are required')
        sampling_path = directory / 'model' / 'sampling-inner.jsonl'
        wire = {k: v for k, v in SAMPLING.items() if k != 'schema'}
        wire.update(model=MODEL, stream=False)
        requests = [json.loads(line) for line in sampling_path.read_text(encoding='utf-8').splitlines()]
        journal = directory / 'model' / 'journal'
        limits = _read(journal / 'limits.json')
        identity = _read(journal / 'identity.json')
        call_paths = sorted(journal.glob('call-*.json'))
        dispatches = [_read(item) for item in call_paths]
        callers = [entry.get('caller') for entry in dispatches]
        if (not 2 <= len(callers) <= BUDGET['max_calls'] or callers[0] != 'Search'
                or callers[1:] != ['Coder'] * (len(callers) - 1)
                or dispatches[0].get('status') != 'answered'
                or dispatches[-1].get('status') != 'answered'
                or not all(entry.get('dispatched_at_utc') and
                           entry.get('output_tokens_reserved') == SAMPLING['max_tokens']
                           for entry in dispatches)
                or len(requests) != len(dispatches)
                or any(row.get('caller') != entry.get('caller') or row.get('effective') != wire
                       or row.get('request_sha256') != entry.get('request_sha256')
                       for row, entry in zip(requests, dispatches))
                or any(limits.get(key) != BUDGET[key] for key in (
                    'max_calls', 'max_output_tokens', 'total_output_token_reservations',
                    'call_seconds', 'campaign_seconds'))
                or identity.get('model') != MODEL or not _hex_hash(identity.get('endpoint_sha256'))):
            raise ValueError(f'{arm}: dispatch budget or sampling differs from the registered pilot')
        endpoints.add(identity['endpoint_sha256'])
        for member in (directory / 'generation.json', sampling_path, journal / 'identity.json',
                       journal / 'limits.json', *call_paths):
            hashes[str(member.relative_to(root))] = _sha(member)
    if len(endpoints) != 1 or (endpoint is not None and endpoints != {endpoint}):
        raise ValueError('P2 model endpoint differs')
    seed_receipt = finals.pop('seed')
    sources.pop('seed')
    judgement = judge_pair(gate=gate, seed_receipt=seed_receipt,
                           receipts=finals, source_hashes=sources)
    return judgement, hashes, endpoints.pop()


def _hex_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(ch in '0123456789abcdef' for ch in value))


def freeze(pilot_path, output):
    """Seal the two new formal pairs after replaying a valid pilot."""
    from benchmarks.experiments.phase1.plan import write_once

    path = Path(pilot_path)
    summary = _read(path)
    judgement, hashes, endpoint = _replay_pair(path, gpu_index=GPU['index'])
    if (summary.get('decidable') is not True or summary.get('formal_gain_claim') is not False
            or not judgement['decidable'] or any(summary.get(key) != judgement[key]
                    for key in ('verdict', 'band_pct', 'on_median_ms', 'off_median_ms'))):
        raise ValueError('P2 formal freeze requires a reproducible valid pilot')
    plan = dict(schema=FORMAL_SCHEMA, status='frozen', declaration=DECLARATION,
                contract_hash=CONTRACT_HASH, seed_source_hash=SEED_SOURCE_HASH,
                probe_channel=PROBE_CHANNEL, effect_threshold_pct=EFFECT_THRESHOLD_PCT,
                model=MODEL, model_endpoint_sha256=endpoint, sampling=SAMPLING,
                gpu=GPU, budget=BUDGET, pairs=PAIRS, formal_gain_claim=False,
                frozen_at=datetime.now(timezone.utc).timestamp(),
                pilot=dict(artifact_sha256=hashes, judgement=judgement))
    plan = deepcopy(plan)
    plan['hash'] = digest(plan)
    write_once(Path(output), plan)
    return plan


def validate(plan):
    """Return validation issues for a formal plan before running a pair."""
    if not isinstance(plan, dict) or plan.get('schema') != FORMAL_SCHEMA:
        return ['expected a P2 formal plan']
    fixed = dict(status='frozen', declaration=DECLARATION, contract_hash=CONTRACT_HASH,
                 seed_source_hash=SEED_SOURCE_HASH, probe_channel=PROBE_CHANNEL,
                 effect_threshold_pct=EFFECT_THRESHOLD_PCT, model=MODEL, sampling=SAMPLING,
                 gpu=GPU, budget=BUDGET, pairs=PAIRS, formal_gain_claim=False)
    issues = [f'formal plan field differs: {key}' for key, value in fixed.items()
              if plan.get(key) != value]
    pilot = plan.get('pilot') or {}
    if (not _hex_hash(plan.get('model_endpoint_sha256')) or not _positive_number(plan.get('frozen_at'))
            or not _hex_hash((pilot.get('artifact_sha256') or {}).get('pilot.json'))
            or (pilot.get('judgement') or {}).get('decidable') is not True):
        issues.append('formal plan requires validated pilot provenance and a freeze time')
    if plan.get('hash') != digest({k: v for k, v in plan.items() if k != 'hash'}):
        issues.append('formal plan hash differs')
    return issues


def judge_series(plan, pair_paths):
    """Judge exactly two new write-once pair directories under one formal plan."""
    issues = validate(plan)
    if issues:
        raise ValueError(' / '.join(issues))
    paths = [Path(path).resolve() for path in pair_paths]
    if len(paths) != len(PAIRS) or len({path.parent for path in paths}) != len(PAIRS):
        raise ValueError('P2 requires exactly two distinct new pair records')
    pilot_hashes = set(plan['pilot']['artifact_sha256'].values())
    seen_evidence = set()
    verdicts = []
    for pair, path in zip(plan['pairs'], paths):
        expected = dict(formal_plan_hash=plan['hash'], formal_pair_id=pair['pair_id'],
                        model=plan['model'], sampling=plan['sampling'], gpu=plan['gpu'],
                        contract_hash=plan['contract_hash'], seed_source_hash=plan['seed_source_hash'],
                        arm_order=pair['arm_order'], rounds_per_arm=1)
        bindings = _read(path.with_name('bindings.json'))
        summary = _read(path)
        if any(bindings.get(key) != value for key, value in expected.items()):
            raise ValueError(f'{pair["pair_id"]}: formal run bindings differ')
        if (summary.get('formal_gain_claim') is not False
                or summary.get('arm_order') != pair['arm_order']):
            raise ValueError(f'{pair["pair_id"]}: formal pair record differs')
        if summary.get('decidable') is not True:
            verdicts.append(dict(pair_id=pair['pair_id'], decidable=False, verdict='unknown',
                                 record_sha256=_sha(path), reported_verdict=summary.get('verdict')))
            continue
        judgement, hashes, _ = _replay_pair(path, gpu_index=plan['gpu']['index'],
                                            after=plan['frozen_at'],
                                            endpoint=plan['model_endpoint_sha256'])
        finals = {sha for name, sha in hashes.items() if name.endswith('/verify-2/result.json')}
        evidence = finals | {hashes['preflight-final.json']}
        if not judgement['decidable'] or len(finals) != 3 or evidence & (pilot_hashes | seen_evidence):
            raise ValueError(f'{pair["pair_id"]}: independent new gate and final receipts are required')
        if any(summary.get(key) != judgement[key]
               for key in ('verdict', 'band_pct', 'on_median_ms', 'off_median_ms')):
            raise ValueError(f'{pair["pair_id"]}: saved pair judgment differs from replay')
        seen_evidence.update(evidence)
        verdicts.append(dict(judgement, pair_id=pair['pair_id'], artifact_sha256=hashes))
    comparable = sum(pair['decidable'] for pair in verdicts)
    helps = sum(pair['verdict'] == 'optimize_helps' for pair in verdicts)
    hurts = sum(pair['verdict'] == 'optimize_hurts' for pair in verdicts)
    gain = helps >= 2 and not hurts
    verdict = ('increment_established' if gain else
               'no_increment_observed' if comparable == len(PAIRS) else 'unknown')
    return dict(schema='v08-p2-formal-series-v1', plan_hash=plan['hash'], verdict=verdict,
                formal_gain_claim=gain, decidable_pairs=comparable, helps_pairs=helps,
                hurts_pairs=hurts, pairs=verdicts)


def main(argv=None):
    import argparse
    from benchmarks.experiments.phase1.plan import write_once

    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    freeze_parser = subcommands.add_parser('freeze')
    freeze_parser.add_argument('--pilot', type=Path, required=True)
    freeze_parser.add_argument('--output', type=Path, required=True)
    series_parser = subcommands.add_parser('judge-series')
    series_parser.add_argument('--plan', type=Path, required=True)
    series_parser.add_argument('--pairs', type=Path, nargs=2, required=True)
    series_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        plan = freeze(args.pilot, args.output)
        print(plan['hash'])
    else:
        result = judge_series(_read(args.plan), args.pairs)
        write_once(args.output, result)
        print(result['verdict'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
