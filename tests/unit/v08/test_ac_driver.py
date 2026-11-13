"""P0-4 的两个半环：臂面核对（①）与 supervisor 独立终评（②）。

臂面那两条用记下来的 `--dump-config` 文本做输入，因为真正的 dump 需要装了 DSH 的机器；
真机那一条带 skipif，跑得到就核对、跑不到就跳过（不假装通过）。
"""
import json
from pathlib import Path

import pytest

from benchmarks.experiments.v08.ac_driver import (CELL_FIELDS, SurfaceError, cell,
                                                  final_evaluation, surface_check)

A_SURFACE = """# == @deepseek-ai/dsh-base
- id: tool-bash
  name: '@deepseek-ai/dsh-tool-bash'
- id: tool-fs
  name: '@deepseek-ai/dsh-tool-fs'
- id: tool-fs-search
  name: '@deepseek-ai/dsh-tool-fs-search'
"""

C_SURFACE = A_SURFACE + """- id: mcp-operator
  name: '/x/dsh-mcp-client/lib/index.js'
- id: operator-tools
  name: '/x/integrations/dsh-operator/index.mjs'
"""


def _builder(text):
    def build(arm, **kwargs):
        return dict(patch=[], home='/tmp/ac-home', work='/tmp/ac-work',
                    env={}, patch_sha256='0' * 64, surface=dict(arm=arm))
    return build


def test_a_healthy_a_arm_passes(monkeypatch):
    import benchmarks.experiments.v08.ac_driver as driver
    monkeypatch.setattr(driver, '_dump', lambda *a, **k: A_SURFACE)
    facts = surface_check('A', runtime={}, builder=_builder(A_SURFACE), node='node', entry='bin.js')
    assert facts['tools'] == {'tool-bash': True, 'tool-fs': True, 'tool-fs-search': True}
    assert facts['operator_entries'] == []


def test_an_a_arm_with_a_disabled_shell_is_refused(monkeypatch):
    """归档里出过的事故：bundle 静默 skip，而 prompt 还在按"你有插件"写。"""
    broken = A_SURFACE.replace("- id: tool-bash\n  name: '@deepseek-ai/dsh-tool-bash'",
                               "- id: tool-bash\n  disabled: true")
    import benchmarks.experiments.v08.ac_driver as driver
    monkeypatch.setattr(driver, '_dump', lambda *a, **k: broken)
    with pytest.raises(SurfaceError) as caught:
        surface_check('A', runtime={}, builder=_builder(broken), node='node', entry='bin.js')
    assert 'tool-bash' in str(caught.value)


def test_an_a_arm_with_plugin_entries_is_refused(monkeypatch):
    import benchmarks.experiments.v08.ac_driver as driver
    monkeypatch.setattr(driver, '_dump', lambda *a, **k: C_SURFACE)
    with pytest.raises(SurfaceError) as caught:
        surface_check('A', runtime={}, builder=_builder(C_SURFACE), node='node', entry='bin.js')
    assert 'mcp-operator' in str(caught.value)


def test_a_c_arm_without_its_plugin_is_refused(monkeypatch):
    import benchmarks.experiments.v08.ac_driver as driver
    monkeypatch.setattr(driver, '_dump', lambda *a, **k: A_SURFACE)
    with pytest.raises(SurfaceError) as caught:
        surface_check('C', runtime={}, builder=_builder(A_SURFACE), node='node', entry='bin.js',
                      relay=type('R', (), {'client_config': lambda self: {}})(),
                      development=dict(url='u', token='t', contract='c', backend='mock', timeout=1))
    assert 'mcp-operator' in str(caught.value)


def test_a_c_arm_with_its_plugin_passes(monkeypatch):
    import benchmarks.experiments.v08.ac_driver as driver
    monkeypatch.setattr(driver, '_dump', lambda *a, **k: C_SURFACE)
    facts = surface_check('C', runtime={}, builder=_builder(C_SURFACE), node='node', entry='bin.js',
                          relay=type('R', (), {'client_config': lambda self: {}})(),
                          development=dict(url='u', token='t', contract='c', backend='mock', timeout=1))
    assert facts['operator_entries'] == ['mcp-operator', 'operator-tools']


# ── ② supervisor 独立终评：final 的前置是同源码的一次完整 L2 ──────────────────

DECLARATION = {"request": "fp16 relu 256x256", "shapes": [[256, 256]],
               "hardware": "mock_gpu", "dsl": "python", "precision": "fp16",
               "min_speedup": 1.0}
BUDGET = dict(wall_clock_s=600, max_verifications=3, max_model_calls=3,
              max_output_tokens_per_call=1024, total_output_token_reservations=3072)
GOOD = "import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n"


def test_the_supervisor_final_evaluation_runs_prepare_then_final(tmp_path):
    result = final_evaluation(tmp_path / 'run', GOOD, declaration=DECLARATION,
                              host_budget=BUDGET, backend='fake')
    assert result['delivered'] is True
    final = result['final']
    assert final['verification_purpose'] == 'final'
    assert final['levels']['L2'] is True
    # 终评预留被用掉：这一格以后不能再拿一份"独立终评"
    assert final['host_task']['remaining_final'] == 0


def test_a_source_that_fails_l2_never_reaches_final(tmp_path):
    """`host_final_source` 会拒绝没有 L2 通过的源码 —— 所以顺序不是风格问题。"""
    broken = "import numpy as np\ndef kernel(X):\n    return X + 1\n"
    result = final_evaluation(tmp_path / 'run', broken, declaration=DECLARATION,
                              host_budget=BUDGET, backend='fake')
    assert result['delivered'] is False and result['final'] is None
    assert 'host_final_source' in result['reason']


def test_a_cell_speaks_the_comparator_language():
    body = cell('attention-prefill-4096', 'A', state='completed',
                final={'verification_purpose': 'final'}, source='/tmp/x.py',
                model_dispatches=17, verifications=3)
    assert set(body) <= set(CELL_FIELDS)
    assert body['state'] == 'completed' and body['source'] == '/tmp/x.py'
    with pytest.raises(SurfaceError):
        cell('t', 'A', state='not_a_state')
    with pytest.raises(SurfaceError):
        cell('t', 'A', state='completed', nonsense=1)

def test_the_l2_message_carries_the_shape_of_the_error_not_just_its_size():
    """`matched_ratio=0.017` + `first_bad@0` 指向"整体规模/布局错"；只报"误差 3.99"会被读成精度问题。"""
    from operator_agent.decide.verifier import Verifier
    import inspect

    source = inspect.getsource(Verifier)
    assert "matched_ratio={_ratio:.4f}" in source
    assert "first_bad@{_first}" in source
