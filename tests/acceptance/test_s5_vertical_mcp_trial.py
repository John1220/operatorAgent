"""Real headless → repository MCP → mock verification vertical fixture."""
import json
import shutil

import pytest
from benchmarks import s5_vertical_mcp_trial as trial
from benchmarks.s5_vertical_headless import DSH_ROOT


@pytest.mark.skipif(not shutil.which('node') or not (DSH_ROOT/'dsh/lib/bin.js').is_file()
                    or not (DSH_ROOT/'dsh-mcp-client/lib/index.js').is_file(),
                    reason='installed DSH Desktop or MCP client unavailable')
def test_headless_calls_real_mcp_and_evidence_recomputes(tmp_path):
    out=tmp_path/'run'
    result=trial.run(out)
    assert result['status']=='passed' and result['fake_requests']==2
    assert result['mock_verifications']==1 and result['tool_call_id']=='call_operator_verify_01'
    assert not result['formal_s5_credit'] and not result['real_model_called']
    assert not result['gpu_used'] and not result['linux_host_verified']
    check=json.loads((out/'evidence-check.json').read_text())
    assert check['evidence_status']=='complete'
    source=(out/'operator-run'/'verify-1'/'candidate.py').read_text()
    assert source and result['candidate_sha256']==check['candidate_sha256']
    with pytest.raises(FileExistsError): trial.run(out)
