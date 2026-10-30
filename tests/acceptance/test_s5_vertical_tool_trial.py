"""Full local headless tool-feedback cycle; no real model or GPU."""
import json
import shutil

import pytest
from benchmarks.experiments.s5 import s5_vertical_tool_trial as trial
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT


@pytest.mark.skipif(not shutil.which('node') or not (DSH_ROOT/'dsh/lib/bin.js').is_file(),
                    reason='installed DSH Desktop unavailable')
def test_real_headless_tool_feedback_and_shared_budget(tmp_path):
    output=tmp_path/'run'; result=trial.run(output)
    assert result['status']=='passed' and result['fake_requests']==2
    assert result['mock_verifications']==1 and result['tool_call_id']=='call_s5_mock_01'
    assert not result['real_model_called'] and not result['gpu_used']
    assert not result['operator_plugin_tested'] and not result['isolation_verified']
    evidence=json.loads((output/'evidence-check.json').read_text())
    assert evidence['evidence_status']=='complete'
    observation=json.loads((output/'provider-observations.json').read_text())
    assert observation['validation_requests']==1
    assert observation['requests'][1]['tool_feedback'][0]['content_sha256']==result['tool_reply_sha256']
    with pytest.raises(FileExistsError): trial.run(output)
