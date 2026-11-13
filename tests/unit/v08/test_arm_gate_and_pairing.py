"""A 臂接线：预算硬门 + 与 B/C 同一个读取器的 arm log；配对比较器的规则。

两条历史缺口来自 `usage-reconcile` 的 blocker 列表：
  * "A 臂没有同等的调用硬门" —— 于是"同预算"只是计划里的一句话；
  * "`pair.json` 的比较器还没写成代码" —— 于是判据靠人读表。
这里把它们各自钉成代码，并且钉住"坏输入会停止，而不是被判成不可比"。
"""
import hashlib
import json
import time

import pytest

from benchmarks.experiments.v08 import r2_plan
from benchmarks.experiments.v08.arm_gate import ArmBudgetError, ArmGate, in_flight
from benchmarks.experiments.v08.r2_compare import (PairInputError, arm_summary, compare,
                                                   series_state)
from operator_agent.decide.arm_metering import normalize_arm_document


BUDGET = dict(max_calls=3, max_output_tokens_per_call=1000,
              total_output_tokens_reservations=2500, wall_clock_s=600)


def gate(tmp_path, **overrides):
    clock = {'now': 0.0}
    instance = ArmGate(tmp_path / 'arm-A.json', budget=dict(BUDGET, **overrides),
                       clock=lambda: clock['now'])
    instance.clock = clock
    return instance


# ── 预算硬门 ────────────────────────────────────────────────────────────────

def test_the_call_cap_is_enforced_not_reported(tmp_path):
    g = gate(tmp_path)
    for _ in range(3):
        rid = g.reserve(provider='aliyun', model='m', max_output_tokens=100)
        g.finish(rid, usage=dict(inputTokens=1, outputTokens=2, totalTokens=3))
    with pytest.raises(ArmBudgetError) as caught:
        g.reserve(provider='aliyun', model='m', max_output_tokens=100)
    assert caught.value.code == 'arm_call_budget'
    # 被拒的那一次**不许**留下一条"已预留"的派发
    assert len(json.loads((tmp_path / 'arm-A.json').read_text())['dispatches']) == 3


def test_the_token_and_per_call_caps_are_enforced(tmp_path):
    g = gate(tmp_path)
    with pytest.raises(ArmBudgetError) as caught:
        g.reserve(provider='p', model='m', max_output_tokens=1001)
    assert caught.value.code == 'arm_output_cap'

    rid = g.reserve(provider='p', model='m', max_output_tokens=1000)
    g.finish(rid)
    rid = g.reserve(provider='p', model='m', max_output_tokens=1000)
    g.finish(rid)
    with pytest.raises(ArmBudgetError) as caught:      # 只剩 500
        g.reserve(provider='p', model='m', max_output_tokens=1000)
    assert caught.value.code == 'arm_token_budget'


def test_the_wall_clock_cap_is_enforced(tmp_path):
    g = gate(tmp_path)
    g.clock['now'] = 601.0
    with pytest.raises(ArmBudgetError) as caught:
        g.reserve(provider='p', model='m', max_output_tokens=10)
    assert caught.value.code == 'arm_deadline'


def test_the_log_is_the_document_usage_reconcile_reads(tmp_path):
    g = gate(tmp_path)
    rid = g.reserve(provider='aliyun', model='deepseek-v4.1-flash-1017',
                    max_output_tokens=100,
                    sampling=dict(temperature=0.7, n=1, enable_thinking=True))
    g.finish(rid, usage=dict(inputTokens=11, outputTokens=7, totalTokens=18))
    rid = g.reserve(provider='aliyun', model='deepseek-v4.1-flash-1017', max_output_tokens=100)
    g.finish(rid, status='failed', usage=None)          # provider 没给计数 = 未知，不是 0

    document = json.loads((tmp_path / 'arm-A.json').read_text())
    normal = normalize_arm_document(document)           # 与 B/C 同一个读取器
    assert normal['dispatches'] == 2 and normal['usage_unknown'] == 1
    assert normal['budget']['max_model_calls'] == 3
    # 任一次未知 ⇒ token 合计是 null（半截求和长得像总量）
    assert normal['output_tokens'] is None
    assert in_flight(document) == 0
    # 采样字段「能记就记」：第二次没给就真的没有，不会被填成相等
    assert document['dispatches'][0]['temperature'] == 0.7
    assert 'temperature' not in document['dispatches'][1]


def test_an_arm_log_refuses_to_reopen(tmp_path):
    gate(tmp_path)
    with pytest.raises(ArmBudgetError) as caught:
        gate(tmp_path)
    assert caught.value.code == 'arm_gate_exists'


def test_a_reservation_can_only_be_settled_once(tmp_path):
    g = gate(tmp_path)
    rid = g.reserve(provider='p', model='m', max_output_tokens=10)
    g.finish(rid)
    with pytest.raises(ArmBudgetError) as caught:
        g.finish(rid)
    assert caught.value.code == 'arm_gate_settled'


# ── 配对比较器 ──────────────────────────────────────────────────────────────

def _plan(tmp_path, task_id='attention-prefill-4096'):
    return r2_plan.freeze(dict(pairs=2, max_model_calls=30, max_output_tokens_per_call=8192,
                               total_output_token_reservations=245760,
                               max_verifications=8, wall_clock_s=2400, gpu_minutes=45),
                          tmp_path / 'plan.json', task_id=task_id)


def _receipt(tmp_path, name, *, speedup=1.0, validity='valid', contract='c0ffee',
             purpose='final', l2=True, source_text=None):
    source = tmp_path / f'{name}.py'
    source.write_text(source_text or f'def kernel():\n    return {name!r}\n', encoding='utf-8')
    body = dict(verification_purpose=purpose, contract_hash=contract,
                source=str(source), source_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
                levels={'L1': True, 'L2': l2, 'L3': validity == 'valid'},
                validity=validity,
                verdicts=[{'level': 'L3', 'passed': validity == 'valid',
                           'metrics': {'speedup_vs_baseline': speedup,
                                       'measurement_validity': validity,
                                       'median_time_ms': 1.0, 'baseline_time_ms': 1.0}}])
    return body


def test_a_comparable_pair_reports_the_paired_difference(tmp_path):
    plan = _plan(tmp_path)
    pair = compare(plan, {'B': _receipt(tmp_path, 'b', speedup=1.00),
                          'C': _receipt(tmp_path, 'c', speedup=1.03)})
    comparison = pair['comparison']
    assert comparison['comparable'] is True and comparison['reasons'] == []
    assert comparison['speedup_difference_C_minus_B'] == pytest.approx(0.03)
    assert comparison['relative_speedup_difference_pct'] == pytest.approx(3.0)
    assert comparison['clears_registered_positive_effect'] is True
    assert comparison['formal_gain_claim'] is False
    assert pair['plan_hash_verified'] is True
    assert pair['series'] if 'series' in pair else True


def test_a_degraded_arm_makes_the_pair_non_decidable_instead_of_a_smaller_number(tmp_path):
    plan = _plan(tmp_path)
    pair = compare(plan, {'B': _receipt(tmp_path, 'b', speedup=1.00),
                          'C': _receipt(tmp_path, 'c', speedup=1.03, validity='degraded')})
    comparison = pair['comparison']
    assert comparison['comparable'] is False
    assert any('degraded' in reason for reason in comparison['reasons'])
    assert comparison['speedup_difference_C_minus_B'] is None
    assert comparison['clears_registered_positive_effect'] is False


def test_two_contract_hashes_are_not_a_pair(tmp_path):
    plan = _plan(tmp_path)
    pair = compare(plan, {'B': _receipt(tmp_path, 'b', contract='aaa'),
                          'C': _receipt(tmp_path, 'c', contract='bbb')})
    assert pair['comparison']['comparable'] is False
    assert any('契约哈希不同' in reason for reason in pair['comparison']['reasons'])


def test_a_drifted_source_hash_stops_the_comparator(tmp_path):
    plan = _plan(tmp_path)
    receipt = _receipt(tmp_path, 'b')
    (tmp_path / 'b.py').write_text('def kernel():\n    return "edited"\n', encoding='utf-8')
    with pytest.raises(PairInputError) as caught:
        compare(plan, {'B': receipt, 'C': _receipt(tmp_path, 'c')})
    assert '绑定的源码哈希与实际文件不一致' in str(caught.value)


def test_an_explore_receipt_is_not_a_final_evaluation(tmp_path):
    plan = _plan(tmp_path)
    with pytest.raises(PairInputError) as caught:
        compare(plan, {'B': _receipt(tmp_path, 'b', purpose='explore'),
                       'C': _receipt(tmp_path, 'c')})
    assert '独立终评' in str(caught.value)


def test_a_draft_or_withdrawn_plan_cannot_produce_a_pair(tmp_path):
    draft = r2_plan.design('attention-prefill-4096')
    with pytest.raises(PairInputError) as caught:
        compare(draft, {'B': _receipt(tmp_path, 'b'), 'C': _receipt(tmp_path, 'c')})
    assert '冻结' in str(caught.value)

    withdrawn = r2_plan.design('attention-prefill-1024')
    with pytest.raises(PairInputError) as caught:
        compare(withdrawn, {'B': _receipt(tmp_path, 'b'), 'C': _receipt(tmp_path, 'c')})
    assert 'was withdrawn' in str(caught.value)


def test_the_series_stops_after_two_non_decidable_pairs(tmp_path):
    plan = _plan(tmp_path)
    dead = {'comparison': {'comparable': False}}
    assert series_state(plan, [dead])['stop'] is False
    state = series_state(plan, [dead, dead])
    assert state['consecutive_nondecidable'] == 2 and state['stop'] is True
    assert series_state(plan, [dead, {'comparison': {'comparable': True}}])['stop'] is False

def test_the_cli_writes_one_pair_file_with_a_series_state(tmp_path):
    from benchmarks.experiments.v08.r2_compare import main as cli_main

    plan = _plan(tmp_path)
    plan_path = tmp_path / 'plan.json'
    b = _receipt(tmp_path, 'b', speedup=1.00)
    c = _receipt(tmp_path, 'c', speedup=1.03)
    for name, body in (('b', b), ('c', c)):
        (tmp_path / f'{name}.json').write_text(json.dumps(body), encoding='utf-8')
    out = tmp_path / 'pair-1.json'
    assert cli_main(['--plan', str(plan_path),
                     '--arm', f'B={tmp_path / "b.json"}',
                     '--arm', f'C={tmp_path / "c.json"}',
                     '--json-out', str(out)]) == 0
    pair = json.loads(out.read_text())
    assert pair['comparison']['comparable'] is True
    assert pair['series']['stop'] is False and pair['series']['pairs_attempted'] == 1
    assert pair['hash'] and pair['plan_id'] == plan['plan_id']


# ---------------------------------------------------------------------------
# P0-3：预算硬门接进 A 臂派发路径（第 N+1 次被拒，日志仍是 N 行）
# ---------------------------------------------------------------------------

def test_the_wired_dispatch_refuses_the_call_after_the_budget(tmp_path):
    from benchmarks.experiments.v08.arm_gate import wrap

    g = ArmGate(tmp_path / 'arm-A.json',
                budget=dict(max_calls=30, max_output_tokens_per_call=8192,
                            total_output_tokens_reservations=245760, wall_clock_s=2400))
    sent = []

    def dispatch(request):
        sent.append(request)
        return 'ok'

    call = wrap(dispatch, g, provider='upstream', model='m', default_output_tokens=8192)
    for _ in range(30):
        assert call({'max_tokens': 8192}) == 'ok'
    with pytest.raises(ArmBudgetError) as caught:
        call({'max_tokens': 8192})
    assert caught.value.code == 'arm_call_budget'
    assert len(sent) == 30                       # 被拒的那次**没有**发到上游
    document = json.loads((tmp_path / 'arm-A.json').read_text())
    assert len(document['dispatches']) == 30
    assert in_flight(document) == 0
    assert normalize_arm_document(document)['dispatches'] == 30


def test_an_upstream_failure_settles_the_reservation_instead_of_leaking_it(tmp_path):
    from benchmarks.experiments.v08.arm_gate import wrap

    g = ArmGate(tmp_path / 'arm-A.json', budget=dict(max_calls=2, wall_clock_s=60))
    def boom(request):
        raise RuntimeError('upstream broke')
    call = wrap(boom, g, provider='upstream', model='m', default_output_tokens=10)
    with pytest.raises(RuntimeError):
        call({'max_tokens': 10})
    document = json.loads((tmp_path / 'arm-A.json').read_text())
    assert [row['status'] for row in document['dispatches']] == ['failed']
    assert in_flight(document) == 0               # 失败也要结算，否则永远"在飞"


def test_the_gate_cli_reports_the_refusal_and_the_log_it_wrote(tmp_path):
    from benchmarks.experiments.v08.arm_gate import main as gate_main

    out = tmp_path / 'gate.json'
    assert gate_main(['--budget', json.dumps(dict(max_calls=3, wall_clock_s=60)),
                      '--per-call', '10', '--probe', '4',
                      '--path', str(tmp_path / 'arm-A.json'),
                      '--json-out', str(out)]) == 0
    summary = json.loads(out.read_text())
    assert summary['dispatches'] == 3 and summary['settled'] == 3
    assert summary['refusal']['code'] == 'arm_call_budget'
    assert summary['refusal']['at_attempt'] == 4
    assert summary['parsed'] is True and summary['accepted_as_designed'] is True
