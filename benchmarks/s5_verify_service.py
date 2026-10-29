"""One verification entry shared by all three S5 preflight arms.

The supervisor — not the tested agent — calls this module with the arm's final
answer text. The module binds the submission to the frozen plan (task identity,
contract and manifest hashes), records the candidate before evaluating it, and
never feeds the verdict back into the same arm run. The only wired backend is
the trusted-local mock L2 prototype; a real GPU backend is reported as unknown
instead of being simulated.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks import s5_l2_evaluator
from operator_agent.run_contract import code_hash, digest

SCHEMA = 's5-preflight-verification-v1'
MAX_SOURCE_BYTES = 65536
STATES = ('passed', 'failed', 'unknown')


class VerificationRefused(RuntimeError):
    """The submission was rejected before any evaluation happened."""


def _write_once(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def _identity(plan):
    """Extract the fields a submission must match, or refuse."""
    if not isinstance(plan, dict):
        raise VerificationRefused('plan object required')
    required = ('hash', 'case_id', 'manifest_hash', 'contract_hash', 'suite_hash')
    if any(not isinstance(plan.get(key), str) for key in required):
        raise VerificationRefused('plan identity incomplete')
    if not isinstance(plan.get('arms'), list) or not all(
            isinstance(arm, str) for arm in plan['arms']):
        raise VerificationRefused('plan identity incomplete')
    if plan.get('formal_s5_credit') is not False or plan.get('preflight') is not True:
        raise VerificationRefused('verification entry only accepts an excluded preflight plan')
    return dict(plan_hash=plan['hash'], case_id=plan['case_id'],
                manifest_hash=plan['manifest_hash'], contract_hash=plan['contract_hash'],
                suite_hash=plan['suite_hash'], arms=tuple(plan['arms']))


def _unknown(ident, *, reason, backend, source_sha256=None):
    return dict(schema=SCHEMA, status='unknown', full_l2_passed=None, reason=reason,
                backend=backend, case_id=ident['case_id'], contract_hash=ident['contract_hash'],
                suite_hash=ident['suite_hash'], source_sha256=source_sha256,
                mock_only=backend == 'mock', remote_backend_executed=False,
                formal_s5_credit=False, plan_hash=ident['plan_hash'])


def submit(plan, *, arm, candidate_text, run_dir, backend=None):
    """Record the candidate, then evaluate it once through the shared entry."""
    ident = _identity(plan)
    if arm not in ident['arms']:
        raise VerificationRefused('unplanned arm')
    if not isinstance(candidate_text, str) or not candidate_text.strip():
        raise VerificationRefused('empty candidate')
    if len(candidate_text.encode('utf-8')) > MAX_SOURCE_BYTES:
        raise VerificationRefused('candidate too large')
    backend = backend or plan.get('verify', {}).get('backend', 'mock')
    if backend not in ('mock', 'h20'):
        raise VerificationRefused('unknown verification backend')
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (target / 'submission.json').exists() or (target / 'result.json').exists():
        raise VerificationRefused('verification already recorded for this arm')
    source_sha256 = code_hash(candidate_text)
    submission = dict(schema=SCHEMA + '-submission', arm=arm, case_id=ident['case_id'],
                      candidate_sha256=source_sha256,
                      candidate_bytes=len(candidate_text.encode('utf-8')),
                      contract_hash=ident['contract_hash'], suite_hash=ident['suite_hash'],
                      manifest_hash=ident['manifest_hash'], plan_hash=ident['plan_hash'],
                      backend=backend)
    _write_once(target / 'submission.json', submission)
    if backend != 'mock':
        result = _unknown(ident, reason='real device adapter not wired; outcome unknown',
                          backend=backend, source_sha256=source_sha256)
    else:
        raw = s5_l2_evaluator.evaluate_mock_fixture(
            candidate_text, ident['case_id'], arm,
            expected_manifest_hash=ident['manifest_hash'])
        result = _from_mock(ident, arm, raw, source_sha256)
    _write_once(target / 'result.json', result)
    return result


def _from_mock(ident, arm, raw, source_sha256):
    """Re-check the mock evaluator's identity claims before trusting its verdict."""
    problems = []
    if not isinstance(raw, dict) or raw.get('schema') != 's5-mock-final-l2-v1':
        problems.append('unexpected evaluator output')
    else:
        if raw.get('case_id') != ident['case_id'] or raw.get('arm') != arm:
            problems.append('task or arm identity mismatch')
        if raw.get('source_hash') != source_sha256:
            problems.append('candidate hash mismatch')
        if raw.get('draft_manifest_hash') != ident['manifest_hash']:
            problems.append('manifest identity drift')
        if raw.get('draft_contract_hash') != ident['contract_hash']:
            problems.append('contract identity drift')
        if raw.get('suite_hash') != ident['suite_hash']:
            problems.append('suite identity drift')
        if raw.get('formal_s5_credit') is not False:
            problems.append('evaluator claimed formal credit')
    if problems:
        return _unknown(ident, reason='; '.join(problems), backend='mock',
                        source_sha256=source_sha256)
    state = raw.get('status')
    if state not in STATES:
        return _unknown(ident, reason='unmapped evaluator status', backend='mock',
                        source_sha256=source_sha256)
    return dict(schema=SCHEMA, status=state,
                full_l2_passed=(True if state == 'passed' else False if state == 'failed' else None),
                reason=raw.get('reason') or ('complete L2 suite passed' if state == 'passed' else None),
                backend='mock', case_id=ident['case_id'], contract_hash=ident['contract_hash'],
                suite_hash=ident['suite_hash'], source_sha256=source_sha256,
                contract_hash_mock=raw.get('mock_contract_hash'),
                execution_target=raw.get('execution_target'),
                mock_only=True, remote_backend_executed=False,
                formal_s5_credit=False, plan_hash=ident['plan_hash'])


def load_result(path):
    """Read a stored verdict and re-derive the fact a summary row needs."""
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise VerificationRefused('verification result missing')
    result = json.loads(target.read_text(encoding='utf-8'))
    if not isinstance(result, dict) or result.get('schema') != SCHEMA:
        raise VerificationRefused('not a preflight verification result')
    state = result.get('status')
    if state not in STATES:
        raise VerificationRefused('invalid verification state')
    expected = True if state == 'passed' else False if state == 'failed' else None
    if result.get('full_l2_passed') is not expected:
        raise VerificationRefused('verdict and state disagree')
    return result


def final_l2(plan, result):
    """The identity-bound triple a summary row may copy; never an artifact proof."""
    return dict(full_l2_passed=result.get('full_l2_passed'),
                source_sha256=result.get('source_sha256'),
                contract_hash=result.get('contract_hash'),
                verdict_sha256=digest({k: result.get(k) for k in
                                       ('status', 'source_sha256', 'contract_hash', 'suite_hash')}))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('A', 'B', 'C'))
    parser.add_argument('--candidate', required=True, type=Path,
                        help='file containing the final answer text to submit')
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    from benchmarks.s5_preflight_plan import load
    plan = load(args.plan)
    result = submit(plan, arm=args.arm,
                    candidate_text=args.candidate.read_text(encoding='utf-8'),
                    run_dir=args.run_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())