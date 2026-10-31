"""Budgeted mock validator for a single offline S5 vertical fixture.

Only the exact, trusted Python fixture strings below can be executed. This is
not a sandbox for generated code, a GPU service, or evidence of S5 quality.
Other inputs consume an attempt and return unknown without executing them.
The caller is responsible for persisting the returned events outside the
untrusted workspace; this in-process counter is not an isolation boundary.
"""
from __future__ import annotations

import contextlib
import io
import threading

from benchmarks.experiments.s5 import s5_l2_evaluator, s5_task_manifest
from operator_agent.core.run_contract import code_hash

CASE_ID = 'layernorm-tail'
MAX_SOURCE_BYTES = 65536
GOOD_FIXTURE = ('import numpy as np\ndef kernel(X):\n'
    '    x = X.astype(np.float32)\n'
    '    z = x - x.mean(axis=-1, keepdims=True)\n'
    '    return (z / np.sqrt((z*z).mean(axis=-1, keepdims=True) + 1e-5)).astype(X.dtype)\n')
BAD_FIXTURE = 'def kernel(X):\n    return X * 0\n'
_TRUSTED = frozenset((GOOD_FIXTURE, BAD_FIXTURE))


class VerticalValidator:
    """One in-process fixture's verification counter; no external effects."""
    def __init__(self, *, max_attempts=1):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 12:
            raise ValueError('verification attempt limit must be between 1 and 12')
        self.max_attempts = max_attempts
        self._attempts = 0
        self._lock = threading.Lock()
        manifest = s5_task_manifest.manifest()
        row = next(row for row in manifest['tasks'] if row['case_id'] == CASE_ID)
        self.manifest_hash = manifest['hash']
        self.contract_hash = row['contract_hash']
        self.suite_hash = row['correctness_suite_hash']

    def validate(self, candidate_code, *, arm='A'):
        """Evaluate trusted fixture or return a nonexecuting unknown result.

        The arm is only a caller-supplied label; it does not establish origin.
        Each accepted attempt is spent even if the result is unknown or the
        evaluator fails. Calls beyond the limit do not execute anything.
        """
        with self._lock:
            if self._attempts >= self.max_attempts:
                return self._result(arm, None, 'budget_exhausted', None,
                                    'verification budget exhausted before evaluation')
            self._attempts += 1
            if arm not in s5_l2_evaluator.ARMS:
                return self._result(arm, None, 'unknown', None, 'invalid arm label')
            if not isinstance(candidate_code, str):
                return self._result(arm, None, 'unknown', None, 'candidate source missing')
            try:
                size = len(candidate_code.encode('utf-8'))
            except UnicodeEncodeError:
                return self._result(arm, None, 'unknown', None, 'candidate source encoding invalid')
            if size > MAX_SOURCE_BYTES:
                return self._result(arm, None, 'unknown', None, 'candidate source exceeds limit')
            source_hash = code_hash(candidate_code)
            if candidate_code not in _TRUSTED:
                return self._result(arm, source_hash, 'unknown', None,
                                    'candidate is not an exact trusted fixture; not executed')
            try:
                # The existing mock verifier emits progress to stdout. Suppress
                # it to keep a caller's structured result stream clean.
                with contextlib.redirect_stdout(io.StringIO()):
                    evidence = s5_l2_evaluator.evaluate_mock_fixture(
                        candidate_code, CASE_ID, arm,
                        expected_manifest_hash=self.manifest_hash)
            except Exception as exc:
                return self._result(arm, source_hash, 'unknown', None,
                                    'mock evaluator unavailable: ' + type(exc).__name__)
            if (not isinstance(evidence, dict) or evidence.get('source_hash') != source_hash
                    or evidence.get('draft_contract_hash') != self.contract_hash
                    or evidence.get('suite_hash') != self.suite_hash
                    or evidence.get('status') not in ('passed', 'failed', 'unknown')):
                return self._result(arm, source_hash, 'unknown', None, 'mock evidence identity unavailable')
            return self._result(arm, source_hash, evidence['status'], evidence,
                                'trusted fixture evaluated by mock L2')

    def _result(self, arm, source_hash, status, evidence, reason):
        return dict(schema='s5-vertical-mock-validation-v1', case_id=CASE_ID,
            arm=arm, arm_origin_verified=False, candidate_sha256=source_hash,
            draft_manifest_hash=self.manifest_hash, contract_hash=self.contract_hash,
            suite_hash=self.suite_hash, status=status, reason=reason, mock_evidence=evidence,
            verification_attempts=self._attempts, max_verification_attempts=self.max_attempts,
            mock_only=True, formal_s5_credit=False, gpu_used=False,
            model_provider_used=False, untrusted_source_executed=False)
