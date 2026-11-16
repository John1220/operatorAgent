"""P2 起点核验收顺序的本地回归（mock backend，不烧 GPU/模型）。

钉住用户定的验收顺序：同一份代码**先过完整 L2（explore）→ 再独立终评（purpose=final）→
终评通过才冻 seed.json**。复用 ac_driver.final_evaluation，host_final_source 拒绝没有前置
完整 L2 的源码，所以失败路径**绝不发起终评**。成功/失败两条路径都用模拟候选覆盖。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.experiments.v08 import p2_seed

DECL = dict(request='relu fp32', shapes=[[8, 16]], precision='fp32', dsl='python', min_speedup=1.0)
BUDGET = dict(wall_clock_s=300, max_llm_calls=1, oracle_total=2)
GOOD = "import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n"
BAD = "import numpy as np\ndef kernel(X):\n    return np.minimum(X, 0)\n"


def _accept(tmp_path, code):
    return p2_seed.accept_seed(code, declaration=DECL, root=tmp_path, budget=BUDGET,
                               backend='mock')


def test_success_freezes_after_full_l2_then_independent_final(tmp_path):
    acc = _accept(tmp_path, GOOD)
    assert acc['explore_l2'] is True          # 先过完整 L2
    assert acc['l2_passed'] is True           # 独立终评 L2 也过
    assert acc['correctness'] == 'passed'
    assert acc['delivered'] is True
    assert acc['frozen'] is True
    assert acc['final_evidence']              # 终评有独立证据
    # 契约由声明现算，重冻结一致
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task
    assert acc['contract_hash'] == freeze_from_task(build_task(DECL)).hash
    assert (tmp_path / 'independent-l2' / 'contract.json').is_file()


def test_failure_does_not_attempt_final_and_does_not_freeze(tmp_path):
    acc = _accept(tmp_path, BAD)
    assert acc['explore_l2'] is False         # 完整 L2 没过
    assert acc['delivered'] is False
    assert acc['frozen'] is False
    # 关键：失败路径**不发起终评**（host_final_source 拒绝无前置 L2 的源码）
    assert acc['final_evidence'] is None
    assert acc['l2_passed'] is False
    assert 'prior full L2 did not pass' in (acc['reason'] or '')


def test_acceptance_order_explore_before_final(tmp_path, monkeypatch):
    """直接钉顺序：explore（无 purpose）必须在 final（purpose=final）之前；
    且只有 explore L2 通过时才调 final。"""
    from operator_agent.app import hitl_tools as ht
    calls = []
    real_verify = ht.verify

    def spy(run_dir, code, **kwargs):
        receipt = real_verify(run_dir, code, **kwargs)
        calls.append(dict(purpose=kwargs.get('verification_purpose', 'explore'),
                          stop_after=kwargs.get('stop_after'),
                          l2=bool((receipt.get('levels') or {}).get('L2'))))
        return receipt

    monkeypatch.setattr(ht, 'verify', spy)
    acc = p2_seed.accept_seed(GOOD, declaration=DECL, root=tmp_path, budget=BUDGET,
                              backend='mock')
    assert acc['frozen'] is True
    assert [c['purpose'] for c in calls] == ['explore', 'final']
    assert calls[0]['l2'] is True             # explore 先过
    assert calls[1]['purpose'] == 'final'     # 再独立终评


def test_failure_path_calls_only_explore(tmp_path, monkeypatch):
    from operator_agent.app import hitl_tools as ht
    calls = []
    real_verify = ht.verify

    def spy(run_dir, code, **kwargs):
        receipt = real_verify(run_dir, code, **kwargs)
        calls.append(kwargs.get('verification_purpose', 'explore'))
        return receipt

    monkeypatch.setattr(ht, 'verify', spy)
    acc = p2_seed.accept_seed(BAD, declaration=DECL, root=tmp_path, budget=BUDGET,
                              backend='mock')
    assert acc['frozen'] is False
    assert calls == ['explore']               # 失败路径只调 explore，不调 final