"""判定缓存：同身份回放、环境漂移拒配、L3/unknown/不完整/final 永不缓存。"""
import json
from pathlib import Path

import pytest

from operator_agent.app import hitl_tools as ht
from operator_agent.core.run_contract import code_hash, protocol_identity
from operator_agent.decide import verdict_cache

DECL = dict(request='relu fp32', shapes=[[4, 8]], precision='fp32', dsl='python')
CORRECT = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'
WRONG = 'import numpy as np\ndef kernel(X):\n    return X.copy()\n'


def _prepared(tmp_path):
    run = tmp_path / 'task'
    ht.prepare(DECL, run)
    return run


def test_identical_resubmission_replays_with_new_evidence(tmp_path):
    run = _prepared(tmp_path)
    first = ht.verify(run, CORRECT, backend='mock', stop_after='L2')
    assert first['levels']['L2'] is True
    assert not first.get('cache_hit')
    assert first['environment']

    second = ht.verify(run, CORRECT, backend='mock', stop_after='L2',
                       candidate_origin='inner_coder')
    assert second['cache_hit'] is True
    assert second['candidate_origin'] == 'inner_coder'
    assert second['cached_from']['round'] == first['round']
    assert second['cached_from']['evidence'] == first['evidence']
    assert second['levels'] == first['levels']
    assert second['passed'] is True
    assert second['evidence'] != first['evidence']
    assert second['round'] == first['round'] + 1
    assert second['source_hash'] == code_hash(CORRECT)
    # 回放目录自身完整：manifest、报告、源码都在，预算门能独立复核。
    directory = Path(second['evidence']).parent
    complete = json.loads((directory / 'complete.json').read_text())
    assert complete['evidence_complete'] is True
    assert complete['binding']['cached_from']['round'] == first['round']
    assert (directory / 'candidate.py').read_text() == CORRECT
    assert Path(second['report']).is_file()
    # 回放报告记本次提交的归因，不照抄旧报告的来源。
    assert 'inner_coder' in Path(second['report']).read_text()
    full = json.loads(Path(second['evidence']).read_text())
    assert full['cache_hit'] is True
    assert full['verdicts'] == json.loads(Path(first['evidence']).read_text())['verdicts']


def test_decisive_failure_is_also_cached(tmp_path):
    run = _prepared(tmp_path)
    first = ht.verify(run, WRONG, backend='mock', stop_after='L2')
    assert first['levels']['L2'] is False
    second = ht.verify(run, WRONG, backend='mock', stop_after='L2')
    assert second['cache_hit'] is True
    assert second['passed'] is False
    assert second['levels'] == first['levels']


def test_different_source_does_not_hit(tmp_path):
    run = _prepared(tmp_path)
    ht.verify(run, CORRECT, backend='mock', stop_after='L2')
    other = ht.verify(run, WRONG, backend='mock', stop_after='L2')
    assert not other.get('cache_hit')


def test_l3_is_neither_cached_nor_served_from_cache(tmp_path):
    run = _prepared(tmp_path)
    first = ht.verify(run, CORRECT, backend='mock', stop_after='L3')
    assert not first.get('cache_hit')
    second = ht.verify(run, CORRECT, backend='mock', stop_after='L3')
    assert not second.get('cache_hit')
    # 已有的 L1/L2 判定也不许喂给 L3 请求（深度不同）。
    third = ht.verify(run, CORRECT, backend='mock', stop_after='L2')
    assert not third.get('cache_hit')


def test_host_budget_counts_dispatches_and_final_never_replays(tmp_path):
    run = tmp_path / 'host-task'
    ht.prepare(DECL, run, host_budget=dict(max_verifications=4, wall_clock_s=60))
    first = ht.verify(run, CORRECT, backend='mock', stop_after='L2')
    second = ht.verify(run, CORRECT, backend='mock', stop_after='L2')
    assert second['cache_hit'] is True
    # 缓存省 GPU，不省派发：两次探索都登记，终评预留不动。
    assert second['host_task']['remaining_exploration'] == 1
    assert second['host_task']['records'][-1]['full_l2_passed'] is True
    final = ht.verify(run, CORRECT, backend='mock', stop_after='L2',
                      verification_purpose='final')
    assert not final.get('cache_hit')
    assert final['evidence'] != second['evidence']
    assert final['host_task']['remaining_final'] == 0


# ── lookup 单元层：手工证据目录，逐条拒配规则 ──────────────────────────────

ENV = {'name': 'H20', 'device': 'cuda', 'compute_capability': '9.0',
       'torch_version': '2.14.0+cu130', 'triton_version': '3.5.0',
       'backend': 'SSHExecutor', 'driver_version': '580.105.08'}
IMPL = protocol_identity()


def _evidence(run, index, *, contract='c' * 64, source='s' * 64, stop_after='L2',
              environment=None, implementation=None, ok=True, passed=True,
              complete=True):
    directory = run / f'verify-{index}'
    directory.mkdir(parents=True)
    binding = dict(tool_version='t', contract_hash=contract, source_hash=source,
                   stop_after=stop_after,
                   environment=ENV if environment is None else environment)
    full = dict(ok=ok, evidence_complete=True, binding=binding,
                implementation=IMPL if implementation is None else implementation,
                levels={'L1': True, 'L2': passed},
                verdicts=[dict(level='L1', passed=True, message='', metrics={}),
                          dict(level='L2', passed=passed, message='', metrics={})],
                source_hash=source, contract_hash=contract, created_at=1.0)
    (directory / 'result.json').write_text(json.dumps(full))
    if complete:
        (directory / 'complete.json').write_text(json.dumps(
            dict(evidence_complete=True, binding=binding)))
    return directory


def _lookup(run, **overrides):
    kwargs = dict(contract_hash='c' * 64, source_hash='s' * 64, stop_after='L2',
                  environment=ENV, implementation=IMPL)
    kwargs.update(overrides)
    return verdict_cache.lookup(run, **kwargs)


def test_lookup_returns_latest_match(tmp_path):
    _evidence(tmp_path, 1)
    _evidence(tmp_path, 2)
    hit = _lookup(tmp_path)
    assert hit is not None and hit['round'] == 2


def test_lookup_skips_newer_nonmatch(tmp_path):
    _evidence(tmp_path, 1)
    _evidence(tmp_path, 2, source='x' * 64)
    hit = _lookup(tmp_path)
    assert hit is not None and hit['round'] == 1


def test_lookup_refuses_environment_drift(tmp_path):
    _evidence(tmp_path, 1, environment=dict(ENV, driver_version='590.1'))
    assert _lookup(tmp_path) is None


def test_lookup_refuses_missing_environment(tmp_path):
    _evidence(tmp_path, 1, environment={})
    assert _lookup(tmp_path) is None
    _evidence(tmp_path, 2)
    assert _lookup(tmp_path, environment={}) is None


def test_lookup_refuses_implementation_drift(tmp_path):
    _evidence(tmp_path, 1, implementation=dict(IMPL, extra='0' * 64))
    assert _lookup(tmp_path) is None


def test_lookup_refuses_l3_and_unknown_and_incomplete(tmp_path):
    _evidence(tmp_path, 1, stop_after='L3')
    assert _lookup(tmp_path) is None
    assert _lookup(tmp_path, stop_after='L3') is None
    _evidence(tmp_path, 2, passed=None)
    assert _lookup(tmp_path) is None
    _evidence(tmp_path, 3, ok=False)
    assert _lookup(tmp_path) is None
    _evidence(tmp_path, 4, complete=False)
    assert _lookup(tmp_path) is None


def test_lookup_refuses_binding_mismatch_between_result_and_complete(tmp_path):
    directory = _evidence(tmp_path, 1)
    complete = json.loads((directory / 'complete.json').read_text())
    complete['binding'] = dict(complete['binding'], extra='changed')
    (directory / 'complete.json').write_text(json.dumps(complete))
    assert _lookup(tmp_path) is None


def test_lookup_refuses_l3_verdict_inside_l2_binding(tmp_path):
    directory = _evidence(tmp_path, 1)
    full = json.loads((directory / 'result.json').read_text())
    full['verdicts'].append(dict(level='L3', passed=True, message='', metrics={}))
    (directory / 'result.json').write_text(json.dumps(full))
    assert _lookup(tmp_path) is None


def test_unknown_l2_is_not_cached_and_retry_reexecutes(tmp_path, monkeypatch):
    """L2 回执为 unknown 时，同源码重试必须重新执行验证，不许回放。

    复现真实形状：`correctness_preflight` 在 profile/参考身份 unknown 时返回单个 L2
    结果，`passed=False`（bool）但 `metrics.correctness.verdict='unknown'`（设施缺口，
    不是候选失败）。旧 `_decisive` 只查 passed 是 bool，把这种 unknown 当成决定性失败
    缓存 ⇒ 重试命中缓存、回放那次 unknown，而不重跑。
    """
    from operator_agent.core.schemas import VerificationResult, VerificationLevel
    from operator_agent.decide.verifier import Verifier

    unknown = VerificationResult(
        VerificationLevel.L2_FUNCTION, False, message='默认正确性配置无法确立',
        metrics={'correctness': {'verdict': 'unknown', 'reason': 'fixture', 'cases': []}})
    monkeypatch.setattr(Verifier, 'correctness_preflight',
                        staticmethod(lambda task, test_inputs=None: unknown))
    run = _prepared(tmp_path)
    first = ht.verify(run, CORRECT, backend='mock', stop_after='L2')
    assert first['levels']['L2'] is False
    second = ht.verify(run, CORRECT, backend='mock', stop_after='L2')
    assert not second.get('cache_hit'), 'unknown L2 不该被缓存，同源码重试必须重新执行'


def test_lookup_refuses_backend_only_environment(tmp_path):
    """环境探测只返回 backend 键（hw_info 空）时，应 fail-closed 判未命中。

    复现真实形状：`fingerprint_of({}, label)` 只产出 `{backend: label}`。旧 lookup 的
    `not environment` 守卫只挡真空字典，挡不住「仅 backend」⇒ 探测失败也被当成有效身份
    去匹配。设计（environment_identity docstring）要求此时关闭缓存。
    """
    backend_only = {'backend': 'SSHExecutor@h20(nvidia)'}
    _evidence(tmp_path, 1, environment=backend_only)
    assert _lookup(tmp_path, environment=backend_only) is None

def test_reference_identity_is_queried_once_per_verification_not_per_case():
    """32 条 case 不该各查一次远端身份（2026-11-19 实测冷 L2：68 次 gpu_info / 16.0 s）。

    缓存的**作用域**是一次 verify：同一次之内复用，跨 verify（以及「生成前 vs 生成后」
    这类跨阶段检查）仍各查一次 —— 后者由 `test_identity_hold_after_generation_...` 钉住。
    """
    from unittest.mock import Mock

    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.backend_caps import Cap
    from operator_agent.decide.reference_contract import (PROFILE_FUSED_LN_RESIDUAL,
                                                          check_observed, registered_reference)
    from operator_agent.decide.verifier import Verifier

    spec = registered_reference(PROFILE_FUSED_LN_RESIDUAL)
    receipt = check_observed(spec, spec.execution.content_dict())
    backend = Mock()
    backend.reference_identity = Mock(
        return_value=dict(success=True, reference_receipt=receipt))
    backend.capabilities = frozenset({Cap.REAL_DEVICE, Cap.REMOTE_VERIFY,
                                      Cap.SAME_DEVICE_BASELINE, Cap.REFERENCE_IDENTITY,
                                      Cap.GPU_PINNING})
    verifier = Verifier(backend)
    task = build_task(dict(request='fp16 32768x6144 residual add layernorm',
                           shapes=[[32768, 6144]], precision='fp16', dsl='triton',
                           hardware='nvidia_gpu', min_speedup=1.5))
    assert verifier.reference_preflight(task) is None
    assert verifier.reference_preflight(task) is None      # 同一次 L2：命中缓存
    assert backend.reference_identity.call_count == 1
    verifier._reference_identity_cache.clear()             # 新的一次 verify
    verifier.reference_preflight(task)
    assert backend.reference_identity.call_count == 2


def test_a_failed_reference_preflight_is_never_cached():
    """校验失败不能进缓存，否则下一次就失去了重试机会。"""
    from unittest.mock import Mock

    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.backend_caps import Cap
    from operator_agent.decide.verifier import Verifier

    backend = Mock()
    backend.reference_identity = Mock(return_value=dict(success=False,
                                                        reference_error='runtime drift'))
    backend.capabilities = frozenset({Cap.REAL_DEVICE, Cap.REMOTE_VERIFY,
                                      Cap.SAME_DEVICE_BASELINE, Cap.REFERENCE_IDENTITY,
                                      Cap.GPU_PINNING})
    verifier = Verifier(backend)
    task = build_task(dict(request='fp16 32768x6144 residual add layernorm',
                           shapes=[[32768, 6144]], precision='fp16', dsl='triton',
                           hardware='nvidia_gpu', min_speedup=1.5))
    first = verifier.reference_preflight(task)
    assert first is not None                                  # 判 unknown
    second = verifier.reference_preflight(task)
    assert second is not None
    assert backend.reference_identity.call_count == 2         # 没有被缓存吞掉
