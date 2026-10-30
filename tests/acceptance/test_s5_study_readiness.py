"""Current formal study must remain blocked before spending resources."""
import hashlib
import json

import pytest

from benchmarks.experiments.s5 import s5_study_readiness as gate
from operator_agent.run_contract import digest


def reference(tmp_path, report):
    path = tmp_path / 'formal.json'
    report = dict(report)
    report['hash'] = digest(report)
    path.write_text(json.dumps(report))
    return dict(report_path=str(path), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def claimed_pass(tmp_path):
    implementation = {'launcher': 'source-bound-test'}
    report = dict(schema=gate.REPORT_SCHEMA, implementation=implementation,
                  checks={key: True for key in gate.REQUIRED_CAPABILITIES})
    return dict(mode='study', implementation=implementation,
                formal_readiness=reference(tmp_path, report))


def test_missing_report_is_not_formal_permission():
    with pytest.raises(gate.FormalReadinessRefused, match='report missing'):
        gate.require_formal_readiness(dict(mode='study', formal_readiness=None))


def test_manually_claiming_all_passes_cannot_supply_missing_execution_boundaries(tmp_path):
    with pytest.raises(gate.FormalReadinessRefused, match='current launcher does not implement'):
        gate.require_formal_readiness(claimed_pass(tmp_path))


def test_formal_report_cannot_be_changed_after_freeze(tmp_path):
    plan = claimed_pass(tmp_path)
    (tmp_path / 'formal.json').write_text('{}')
    with pytest.raises(gate.FormalReadinessRefused, match='hash mismatch'):
        gate.require_formal_readiness(plan)


def test_formal_report_with_another_implementation_is_refused(tmp_path):
    plan = claimed_pass(tmp_path)
    plan['implementation'] = {'launcher': 'other'}
    with pytest.raises(gate.FormalReadinessRefused, match='implementation mismatch'):
        gate.require_formal_readiness(plan)


def test_current_audit_is_not_a_pass_report(tmp_path):
    plan = dict(mode='study', implementation={})
    plan['formal_readiness'] = reference(tmp_path, dict(schema=gate.REPORT_SCHEMA,
        implementation={}, checks={key: key == 'participant_file_boundary'
                                  for key in gate.REQUIRED_CAPABILITIES}))
    with pytest.raises(gate.FormalReadinessRefused, match='not all verified'):
        gate.require_formal_readiness(plan)


@pytest.mark.parametrize('reference', [{}, {'report_path': 'relative.json', 'report_sha256': 'a'*64},
    {'report_path': '/missing/report.json', 'report_sha256': 'a'*64}, 'pass'])
def test_missing_or_malformed_references_do_not_enable_dispatch(reference):
    with pytest.raises(gate.FormalReadinessRefused):
        gate.require_formal_readiness(dict(mode='study', formal_readiness=reference))
