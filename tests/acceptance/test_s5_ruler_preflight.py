"""Offline fail-closed checks for the two-task S5 measurement preflight."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from benchmarks.experiments.s5 import s5_ruler_preflight as ruler

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


def test_freeze_and_validate_preserves_exact_declared_scope(tmp_path):
    out=tmp_path/'plan'
    ruler.freeze(out)
    plan=ruler.validate(out,ruler.sha(out/'plan.json'))
    assert [t['name'] for t in plan['tasks']]==['matmul-2048','matmul-4096']
    assert plan['model_calls']==0 and plan['max_verifications']==2
    assert all(t['declaration']['required_caps']==['real_device'] for t in plan['tasks'])
    with pytest.raises(FileExistsError): ruler.freeze(out)


def test_plan_candidate_or_implementation_drift_fails_before_dispatch(tmp_path):
    out=tmp_path/'plan'; ruler.freeze(out); digest=ruler.sha(out/'plan.json')
    with pytest.raises(ValueError,match='plan sha'): ruler.validate(out,'0'*64)
    (out/'candidate.py').write_text('def kernel(): pass')
    with pytest.raises(ValueError,match='candidate drift'): ruler.validate(out,digest)


def test_execution_is_one_shot_before_any_remote_backend(tmp_path):
    out=tmp_path/'plan'; ruler.freeze(out); digest=ruler.sha(out/'plan.json')
    (out/'execution-started.json').write_text('{}')
    frozen=json.loads((out/'plan.json').read_text())['remote_files']
    with patch.object(ruler,'remote_files',return_value=frozen), patch('operator_agent.execute.remote_executor.SSHExecutor') as backend:
        with pytest.raises(FileExistsError): ruler.execute(out,digest)
        backend.assert_not_called()


def test_failed_execution_restores_remote_and_operator_environment(tmp_path, monkeypatch):
    out=tmp_path/'plan'; ruler.freeze(out); digest=ruler.sha(out/'plan.json')
    (out/'execution-started.json').write_text('{}')
    frozen=json.loads((out/'plan.json').read_text())['remote_files']
    monkeypatch.setenv('OPERATOR_AGENT_REQUIRED_GPU','original-test-value')
    monkeypatch.setenv('REMOTE_TEST_ONLY','preserve-me')
    before={key:value for key,value in os.environ.items()
            if key.startswith(('REMOTE_', 'OPERATOR_AGENT_'))}
    with patch.object(ruler,'remote_files',return_value=frozen):
        with pytest.raises(FileExistsError): ruler.execute(out,digest)
    after={key:value for key,value in os.environ.items()
           if key.startswith(('REMOTE_', 'OPERATOR_AGENT_'))}
    assert after == before


def test_changed_remote_source_rejected_before_network(tmp_path):
    out=tmp_path/'plan'; ruler.freeze(out); digest=ruler.sha(out/'plan.json')
    with patch.object(ruler,'remote_files',return_value={}):
        with pytest.raises(ValueError,match='remote file identity drift'):
            ruler.validate(out,digest)


def test_snapshot_refuses_occupied_or_malformed_gpu_query():
    plan=dict(host='host',gpu_uuid=ruler.UUID,gpu_index=6,gpu_name='NVIDIA H20',min_free_mib=4096,max_util_pct=5)
    class P:
        def __init__(self,text): self.stdout=text
    good=f'6, {ruler.UUID}, NVIDIA H20, 9000, 0\n'
    with patch.object(ruler.subprocess,'run',side_effect=[P(good),P(f'{ruler.UUID}, 123\n')]):
        with pytest.raises(ValueError,match='another compute'): ruler.device_snapshot(plan,{})
    with patch.object(ruler.subprocess,'run',side_effect=[P(good),P('bad\n')]):
        with pytest.raises(ValueError,match='incomplete'): ruler.device_snapshot(plan,{})
    with patch.object(ruler.subprocess,'run',side_effect=[P(good),P('')]):
        assert ruler.device_snapshot(plan,{})['other_compute_processes']==0
