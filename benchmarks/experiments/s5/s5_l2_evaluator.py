"""Offline-only, mock-backend S5 final-L2 evaluator prototype.

This evaluates *trusted local Python test fixtures*, not arbitrary generated
code: the mock execution guard is not a hostile-code sandbox. It never creates
an SSH backend, invokes a model, runs L3, or grants formal/GPU S5 credit. Each
primary/holdout is checked separately through the existing complete L2 suite.
"""
from __future__ import annotations

import ast
import math

from benchmarks.experiments.s5 import s5_task_manifest as draft
from operator_agent.contract import freeze_from_task
from operator_agent.correctness import CorrectnessCase, as_suite
from operator_agent.correctness_profiles import with_default_correctness
from operator_agent.input_fidelity import describe_inputs
from operator_agent.measurement import TRANSPORT_KEY, TRANSPORT_UNKNOWN
from operator_agent.quality_suite import get_case, input_cases_for, task_for
from operator_agent.repair_loop import _full_l2_passed
from operator_agent.run_contract import audit_value, code_hash, digest
from operator_agent.schemas import ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskSchema, VerificationLevel
from operator_agent.tool_evidence import verification_data

# Deliberately conservative for a local mock prototype: never allocate a new
# large candidate workload just to prove that its output remains unknown.
MAX_INPUT_BYTES = 1024 * 1024
ARMS = ('A', 'B', 'C')


def _trusted_python_fixture(code):
    """Constrain accidental GPU/provider imports, not a hostile-code sandbox."""
    try:
        tree = ast.parse(code)
    except (ValueError, SyntaxError, RecursionError):
        return False
    blocked = {'torch', 'triton', 'cuda', 'exec', 'eval', 'open', 'compile',
               '__import__', 'getattr', 'setattr', 'globals', 'locals', 'vars'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name != 'numpy' for alias in node.names):
            return False
        if isinstance(node, ast.ImportFrom):
            return False
        if isinstance(node, ast.Name) and node.id in blocked:
            return False
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in blocked):
            return False
    return True


def _new_case(spec):
    name, operator, shape, precision, seed = spec
    primary = draft._case(name, operator, shape, precision, seed)
    task = TaskSchema(task_id=name, created_at=0, source=SourceSpec(kernel_type=operator),
        target=TargetSpec(hardware='mock_gpu', dsl='python'),
        constraints=ConstraintSpec(precision=Precision(precision), cases=(primary,)))
    task = with_default_correctness(task)
    cases = [CorrectnessCase(name+'-primary', primary, 'random', 'contiguous')]
    cases.extend(CorrectnessCase(name+'-holdout-'+str(n),
        draft._case(name, operator, shape, precision, n), 'random', 'contiguous')
        for n in draft.HOLDOUT_SEEDS)
    return task, tuple(cases)


def _task_and_inputs(case_id):
    try:
        get_case(case_id)
    except ValueError:
        spec = next((row for row in draft.NEW_CASES if row[0] == case_id), None)
        if spec is None:
            raise ValueError('unsupported task')
        return _new_case(spec)
    return task_for(case_id, hardware='mock_gpu', dsl='python'), input_cases_for(case_id)


def _storage_bound(case):
    factor = 4 if case.layout in ('strided', 'broadcast') else 1
    widths = {'fp16': 2, 'fp32': 4}
    total = 0
    for tensor in case.inputs.tensors:
        if tensor.dtype not in widths or any(type(n) is not int or n <= 0 for n in tensor.shape):
            raise ValueError('input size unknown')
        total += math.prod(tensor.shape) * widths[tensor.dtype] * factor
    return total


def _unknown_result(case_id, arm, source_hash, reason):
    return dict(schema='s5-mock-final-l2-v1', case_id=case_id, arm=arm,
        source_hash=source_hash, status='unknown', full_l2_passed=None,
        reason=reason, inputs=[], mock_only=True, formal_s5_credit=False,
        scope='local mock-only prototype; no real device, model or performance claim')


def _qualified(results, expected_suite, expected_inputs):
    """Classify trustworthy complete L2 rather than treating any pass as proof."""
    if not isinstance(results, (list, tuple)):
        return None, 'verification results missing'
    state = _full_l2_passed(results)
    l1 = [r for r in results if r.level == VerificationLevel.L1_COMPILE]
    l2 = [r for r in results if r.level == VerificationLevel.L2_FUNCTION]
    # A verified interface mismatch is a local, deterministic rejection, and
    # the verifier intentionally does not run L2 afterwards. Other L1 failures
    # can indicate guard, resource or transport failure and remain unknown here.
    if len(l1) == 1 and not l1[0].passed and not l2:
        metrics = l1[0].metrics if isinstance(l1[0].metrics, dict) else {}
        if (metrics.get('compile_status') == 'interface_mismatch'
                and metrics.get(TRANSPORT_KEY) != TRANSPORT_UNKNOWN):
            return False, 'verified local interface mismatch'
    if len(l1) != 1 or len(l2) != 1:
        return None, 'missing or duplicate L1/L2 result'
    metrics = l2[0].metrics if isinstance(l2[0].metrics, dict) else {}
    c = metrics.get('correctness') if isinstance(metrics.get('correctness'), dict) else {}
    if (metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN or metrics.get('reference_identity_unknown')
            or metrics.get('correctness_primary_verdict') == 'unknown' or c.get('verdict') == 'unknown'):
        return None, 'input, transport or reference result unknown'
    if state is False:
        return False, 'existing correctness gate rejected candidate'
    if state is not True:
        return None, 'full L2 evidence unavailable'
    if (metrics.get('num_input_sets') != 1 or c.get('suite_hash') != expected_suite.digest()
            or c.get('declared_cases') != len(expected_suite.cases)):
        return None, 'input count or suite identity is incomplete'
    rows = c.get('cases')
    wanted = {case.case_id for case in expected_suite.cases}
    if (not isinstance(rows, list) or len(rows) != len(wanted)
            or any(not isinstance(row, dict) for row in rows)
            or {row.get('case_id') for row in rows} != wanted
            or any(row.get('verdict') != 'passed' for row in rows)):
        return None, 'declared suite is missing or incomplete'
    if expected_suite.deterministic and any(row.get('determinism') != 'passed' for row in rows):
        return None, 'suite determinism not established'
    by_id = {case.case_id: case for case in expected_suite.cases}
    for row in rows:
        expected_case = describe_inputs(by_id[row['case_id']].materialize())
        keys = ('metrics', 'repeat_metrics') if expected_suite.deterministic else ('metrics',)
        for key in keys:
            observed = row.get(key)
            if (not isinstance(observed, dict) or observed.get('input_fidelity_verified') is not True
                    or observed.get('input_fidelity_expected') != expected_case):
                return None, 'suite input fidelity or identity missing'
    fidelity = metrics.get('input_fidelity_readings')
    if (not isinstance(fidelity, list) or len(fidelity) != 1
            or fidelity[0].get('input_fidelity_verified') is not True
            or fidelity[0].get('input_fidelity_expected') != expected_inputs):
        return None, 'primary input fidelity or identity missing'
    return True, 'primary input and complete suite passed'


def evaluate_mock_fixture(code, case_id, arm, *, expected_manifest_hash=None):
    """Run only trusted Python fixture code locally against mock, never SSH.

    Three input checks share one mock verifier and the same task. A missing,
    malformed or unsupported check is unknown. The evaluator does not infer
    provenance from the arm label or use this result for real S5 credit.
    """
    source_hash = code_hash(code) if isinstance(code, str) and code else None
    if arm not in ARMS or not isinstance(case_id, str) or not source_hash:
        return _unknown_result(case_id, arm, source_hash, 'invalid arm, task or source')
    if not _trusted_python_fixture(code):
        return _unknown_result(case_id, arm, source_hash, 'prototype permits trusted Python mock fixtures only')
    manifest = draft.manifest()
    if expected_manifest_hash is not None and expected_manifest_hash != manifest['hash']:
        return _unknown_result(case_id, arm, source_hash, 'draft manifest identity changed')
    row = next((r for r in manifest['tasks'] if r['case_id'] == case_id), None)
    if row is None:
        return _unknown_result(case_id, arm, source_hash, 'unsupported task')
    try:
        task, cases = _task_and_inputs(case_id)
        suite = as_suite(task.constraints.correctness_suite)
        if suite.digest() != row['correctness_suite_hash']:
            raise ValueError('suite identity drift')
        if [c.content_dict() for c in cases] != row['evaluation_inputs']:
            raise ValueError('challenge/holdout identity drift')
        if len(cases) != 3 or any(_storage_bound(case) > MAX_INPUT_BYTES for case in cases):
            raise ValueError('mock input allocation exceeds prototype limit')
    except (ValueError, TypeError, OverflowError) as exc:
        return _unknown_result(case_id, arm, source_hash, str(exc))

    # Semantic contracts can coincide across mock/python and nvidia/triton;
    # record the actual target separately and never promote a mock result.
    from operator_agent.mock_gpu import MockGPU
    from operator_agent.verifier import Verifier
    verifier = Verifier(MockGPU())
    evidence=[]
    for case in cases:
        results = []
        try:
            inputs=case.materialize()
            expected=describe_inputs(inputs)
            results=verifier.verify(code,task,inputs,stop_after=VerificationLevel.L2_FUNCTION)
            state,reason=_qualified(results,suite,expected)
        except Exception as exc:
            state,reason=None,'mock verification unavailable: '+type(exc).__name__
        evidence.append(dict(case_id=case.case_id, input_spec_hash=digest(case.content_dict()),
            input_spec=case.content_dict(),
            source_hash=source_hash, contract_hash=freeze_from_task(task).hash,
            suite_hash=suite.digest(), full_l2_passed=state, reason=reason,
            verification=audit_value(verification_data(results))))
        if state is not True:
            break
    overall=(False if any(row['full_l2_passed'] is False for row in evidence) else
             True if len(evidence)==3 and all(row['full_l2_passed'] is True for row in evidence)
             else None)
    return dict(schema='s5-mock-final-l2-v1',case_id=case_id,arm=arm,source_hash=source_hash,
        draft_manifest_hash=manifest['hash'], draft_contract_hash=row['contract_hash'],
        mock_contract_hash=freeze_from_task(task).hash, suite_hash=suite.digest(),
        execution_target=dict(backend='mock', hardware='mock_gpu', dsl='python'),
        status='passed' if overall is True else ('failed' if overall is False else 'unknown'),
        full_l2_passed=overall, inputs=evidence, mock_only=True, formal_s5_credit=False,
        scope='local mock-only prototype; no real device, model or performance claim')
