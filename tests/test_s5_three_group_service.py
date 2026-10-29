"""Three-arm preflight whose final answers are judged by the shared service."""
import json
import shutil
from pathlib import Path

import pytest

from benchmarks import (s5_preflight_plan, s5_three_group_service_run as service_run,
                        s5_verify_remote)
from benchmarks.s5_vertical_headless import DSH_ROOT
from benchmarks.s5_vertical_validation import GOOD_FIXTURE

DSH_READY = bool(shutil.which('node')) and (DSH_ROOT / 'dsh' / 'lib' / 'bin.js').is_file()
DECLARATION = dict(request='layernorm fp32', shapes=[[7, 33]], hardware='mock_gpu',
                   dsl='python', precision='fp32')


def _plans(tmp_path, *, declaration=None):
    plans = tmp_path / 'plans'
    plans.mkdir()
    preflight = s5_preflight_plan.build()
    preflight_path = plans / 'preflight.json'
    s5_preflight_plan.write(preflight, preflight_path)
    sha = s5_verify_remote.freeze(plans / 'service', declaration=declaration or DECLARATION,
                                  backend='mock')
    return preflight_path, plans / 'service' / 'plan.json', sha


@pytest.mark.skipif(not DSH_READY, reason='installed DSH headless unavailable')
def test_every_arm_is_judged_by_the_shared_service(tmp_path):
    preflight_path, service_path, sha = _plans(tmp_path)
    output = tmp_path / 'run'
    result = service_run.run(preflight_path, output, service_path, sha)
    assert result['counts'] == dict(passed=3, failed=0, unknown=0)
    assert result['wiring_complete'] is True
    assert result['formal_s5_credit'] is False and result['counts_toward_denominator'] is False
    assert result['service']['service_backend'] == 'mock'
    verdicts = set()
    for arm in ('A', 'B', 'C'):
        record = json.loads((output / 'records' / ('arm-' + arm + '.json')).read_text())
        assert record['verification']['service_plan_sha256'] == sha
        assert record['verification']['backend'] == 'mock'
        assert (output / 'arms' / arm / 'verification' / 'submission.json').is_file()
        assert (output / 'arms' / arm / 'verification' / 'result.json').is_file()
        assert (output / 'arms' / arm / 'verification' / 'task').is_dir()
        assert record['final_l2']['contract_hash']
        verdicts.add(record['verification']['verdict_sha256'])
    assert len(verdicts) == 1
    binding = json.loads((output / 'binding.json').read_text())
    assert binding['service_plan_sha256'] == sha
    assert binding['preflight_contract_hash'] != binding['service_contract_hash']
    assert 'different evaluation regime' in binding['regime_note']
    assert binding['task_shape'] == [[7, 33]] and binding['task_precision'] == 'fp32'


def test_a_service_prepared_for_another_task_is_refused(tmp_path):
    wrong = dict(DECLARATION, shapes=[[8, 32]])
    preflight_path, service_path, sha = _plans(tmp_path, declaration=wrong)
    output = tmp_path / 'run'
    with pytest.raises(ValueError):
        service_run.run(preflight_path, output, service_path, sha)
    assert not (output / 'arms').exists()


def test_service_plan_hash_is_required(tmp_path):
    preflight_path, service_path, _ = _plans(tmp_path)
    with pytest.raises(ValueError):
        service_run.run(preflight_path, tmp_path / 'run', service_path, 'f' * 64)
    assert not (tmp_path / 'run' / 'arms').exists()


def test_submission_hook_binds_the_service_verdict(tmp_path):
    _, service_path, sha = _plans(tmp_path)
    plan = s5_verify_remote.load_plan(service_path, sha)
    hook = service_run.submitter(plan, sha)
    verdict = hook({'case_id': 'layernorm-tail'}, 'A', GOOD_FIXTURE, tmp_path / 'arm')
    assert verdict['status'] == 'passed' and verdict['final_l2']['full_l2_passed'] is True
    assert verdict['service_plan_sha256'] == sha
    assert verdict['final_l2']['contract_hash'] == plan['contract_hash']
    result = json.loads((tmp_path / 'arm' / 'verification' / 'result.json').read_text())
    assert result['contract_hash'] == plan['contract_hash']
    assert result['remote_backend_executed'] is False