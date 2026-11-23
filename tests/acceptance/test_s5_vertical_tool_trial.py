"""Full local headless tool-feedback cycle; no real model or GPU."""
import json
import shutil

import pytest
from benchmarks.experiments.s5 import s5_vertical_tool_trial as trial
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT

#: 这一组驱动的是 s5 的验证管线，它对**归档依赖树**有硬依赖：目录在 ≠ 当前解释器能用
#: （归档那份 numpy 是 cpython-39，换 3.12 跑会走到「不要从源码目录导入」）。
#: 所以按**探针**跳过，而不是让 20 条用例各自以「输出不是 JSON」的形式红。
import pytest as _pytest  # noqa: E402
from benchmarks.experiments.s5 import s5_runtime as _s5_runtime  # noqa: E402
if not _s5_runtime.available():
    pytestmark = _pytest.mark.skipif(
        True, reason='s5 runtime deps 不可用: ' + '; '.join(_s5_runtime.deps_problems())
        + '。冻结的那棵树是按 Python 3.9 建的，用当前解释器重建一次即可：'
        + 'python -m benchmarks.experiments.s5.s5_rebuild_deps')


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
