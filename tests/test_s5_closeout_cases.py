"""Sharing a historical case must not leak free text or erase a failed attempt."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.s5_closeout_cases import build, load, project_verification, sha, verify


@pytest.fixture
def bundle(tmp_path):
    sources = Path(__file__).resolve().parents[1]/'output/plugin_completion'
    if not (sources/'20261029-s5-plugin-entry-acceptance-v1').is_dir():
        pytest.skip('retained private historical cases unavailable')
    dest = tmp_path/'cases'
    build(sources, dest)
    return dest


def rewrite(root, relative, change):
    path = root/relative
    value = load(path)
    change(value)
    path.write_text(json.dumps(value), encoding='utf-8')
    manifest_path = root/'manifest.json'
    manifest = load(manifest_path)
    manifest['files'][relative] = sha(path)
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')


def test_bundle_runs_outside_repo_without_site_packages_or_candidate_imports(bundle, tmp_path):
    result = subprocess.run([sys.executable, '-I', '-S', str(bundle/'verify.py')],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['uses_gpu'] is False
    assert json.loads(result.stdout)['authenticity_attested'] is False
    assert verify(bundle)['cases'] == 2


def test_original_failure_cannot_be_relabelled_success_even_with_updated_manifest(bundle):
    rewrite(bundle, 'repair/original-final.json', lambda d: d.update(status='passed', accepted=True))
    with pytest.raises(ValueError, match='original failure overwritten'):
        verify(bundle)


def test_summary_cannot_claim_comparative_gain_with_updated_manifest(bundle):
    rewrite(bundle, 'summary.json', lambda d: d.update(inference_allowed=True))
    with pytest.raises(ValueError, match='summary contradicts'):
        verify(bundle)


def test_not_run_cases_cannot_be_counted_as_failed_executions(bundle):
    rewrite(bundle, 'repair/initial.json',
            lambda d: [r.update(verdict='failed') for r in d['cases']])
    with pytest.raises(ValueError, match='unexecuted suite misreported'):
        verify(bundle)


def test_repair_feedback_must_precede_second_plugin_call(bundle):
    rewrite(bundle, 'repair/initial.json',
            lambda d: d['feedback'].update(call_event_index=90, result_event_index=91))
    with pytest.raises(ValueError, match='repair feedback order'):
        verify(bundle)


def test_candidate_mutation_rejected_even_if_manifest_is_updated(bundle):
    path = bundle/'entry/final.py'
    path.write_text(path.read_text()+'\n# changed\n')
    manifest = load(bundle/'manifest.json')
    manifest['files']['entry/final.py'] = sha(path)
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='historical candidate mismatch'):
        verify(bundle)


def test_sensitive_string_rejected_even_if_manifest_is_updated(bundle):
    rewrite(bundle, 'summary.json', lambda d: d.update(notes='https://private.invalid/credential'))
    with pytest.raises(ValueError, match='sensitive text'):
        verify(bundle)


def test_projection_omits_untrusted_free_text(tmp_path):
    root = tmp_path/'evaluation'
    (root/'task/verify-1').mkdir(parents=True)
    candidate = tmp_path/'candidate.py'
    candidate.write_text('pass\n')
    from benchmarks.s5_closeout_cases import CONTRACT
    service = dict(source_sha256=sha(candidate), contract_hash=CONTRACT,
                   status='failed', levels={'L1': True, 'L2': False}, full_l2_passed=False,
                   remote_backend_executed=True, mock_only=False,
                   private_url='https://secret.invalid/api')
    raw = dict(source_hash=sha(candidate), contract_hash=CONTRACT, implementation_stable=True,
               private_path='/Users/person/private', verdicts=[dict(level='L2',
               message='secret text', metrics=dict(input_fidelity_verified=True,
               correctness=dict(suite_hash='a'*64, verdict='failed', cases=[dict(
                   case_id='case-1', verdict='not_run', reason='Bearer secret-value')])) )])
    (root/'result.json').write_text(json.dumps(service))
    (root/'task/verify-1/result.json').write_text(json.dumps(raw))
    row = project_verification(root, candidate)
    rendered = json.dumps(row)
    assert 'secret' not in rendered and '/Users/' not in rendered and 'Bearer' not in rendered
    assert row['origin_result_sha256'] == sha(root/'task/verify-1/result.json')


def test_export_never_overwrites_existing_bundle(bundle):
    with pytest.raises(ValueError, match='output already exists'):
        build(bundle.parent, bundle)
