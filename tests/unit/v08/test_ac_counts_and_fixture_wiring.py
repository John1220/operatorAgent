"""A/C fixture wiring: per-arm dispatch counting and v7 pair aggregation."""
import json

import pytest

from benchmarks.experiments.v08 import ac_fixture, ac_plan, fused_ac_aggregate as fa


def _receipt(path, created_at=100.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(created_at=created_at, levels={'L2': True})),
                    encoding='utf-8')


def test_a_arm_session_dispatch_is_counted_once(tmp_path):
    """A 的一次会话提交会同时留 result-N.json 与 verify-N.json；两处都数是翻倍。"""
    run = tmp_path / 'run'
    _receipt(run / 'native-verification' / 'result-0001.json')
    _receipt(run / 'verify-1.json')
    _receipt(tmp_path / 'final-run' / 'verify-1.json')
    _receipt(tmp_path / 'final-run' / 'verify-2.json')
    # A: 1 session dispatch + 2 final-run dispatches (result-N and verify-N are one dispatch)
    assert ac_fixture._counts(tmp_path, arm='A')['verifications'] == 3
    assert ac_fixture._counts(tmp_path, arm='C')['verifications'] == 3


def test_c_arm_session_dispatch_is_counted_once(tmp_path):
    run = tmp_path / 'run'
    _receipt(run / 'verify-1.json')
    _receipt(tmp_path / 'final-run' / 'verify-1.json')
    _receipt(tmp_path / 'final-run' / 'verify-2.json')
    assert ac_fixture._counts(tmp_path, arm='C')['verifications'] == 3


def test_fixture_cells_aggregate_into_a_pair_verdict(tmp_path):
    plan = ac_plan.freeze(dict(cells=4), tmp_path / 'v8.json', version='v8')
    task_id = ac_plan.FUSED_TASK['task_id']

    def cell(arm, delivery_class, speedup):
        return dict(task_id=task_id, arm=arm, state='completed',
                    final_receipt={'verification_purpose': 'final'},
                    delivery_class=delivery_class, qualified=delivery_class == 'qualified',
                    final_speedup=speedup, final_validity='valid',
                    fusion_static={'passed': True}, model_dispatches=2, verifications=3,
                    wall_clock_total_s=900, scripted_fixture=True)

    pair = fa.pair_verdict(plan, [cell('A', 'qualified', 2.2392),
                                  cell('C', 'qualified', 2.2352)])
    assert pair['paired_difference_C_minus_A'] == 0
    assert pair['subtype'] == 'both_qualified'
    assert pair['round_complete'] is True
    # fixture 格必须能被识别成 test_only，不能混进正式读数
    assert pair['per_arm']['A']['delivery_class'] == 'qualified'


def test_a_stream_timeout_is_an_instrument_failure_not_a_candidate_failure():
    """2026-11-18：C 臂两次 `Stream ended without finish_reason` 被记成 failed。"""
    from benchmarks.experiments.v08.ac_live import session_instrument_failure

    timed_out = dict(returncode=1, events=[],
                     stderr='dsh: TRANSPORT: Stream ended without finish_reason')
    assert session_instrument_failure(timed_out) == 'Stream ended without finish_reason'
    idle = dict(returncode=1, events=[dict(type='status', phase='turn_end', reason=dict(
        kind='error', error=dict(message='stream idle timeout after 60000ms')))], stderr='')
    assert session_instrument_failure(idle) == 'stream idle timeout'
    clean = dict(returncode=0, events=[dict(type='final', text='done')], stderr='')
    assert session_instrument_failure(clean) is None


def test_repair_prompt_points_at_the_seed_source_and_the_named_failure():
    """修复轮不能再用 phase="generate"；要指向修前源码、1x33 回执与验收条件。"""
    from benchmarks.experiments.v08.ac_live import repair_workflow

    receipt = ('/Users/zh/repos/operatorAgent/output/ac-v08f-pair2-A/run/'
               'native-verification/result-0001.json')
    for arm in ('A', 'C'):
        text = repair_workflow(arm, run_dir='/tmp/run', ssh=dict(host='h', port=1,
                                                                user='u', key='/k'),
                               budget=dict(max_model_calls=20), seed_receipt=receipt)
        assert 'start_kernel.py' in text
        assert '完整 L2' in text and '独立终评' in text
        assert 'fused-ln-residual-1x33-random-contiguous' in text
        assert '9.96e-02' in text          # 回执里这条用例的误差原文
        assert 'phase="generate"' not in text
    c_text = repair_workflow('C', run_dir='/tmp/run',
                             ssh=dict(host='h', port=1, user='u', key='/k'),
                             budget=dict(max_model_calls=20), seed_receipt=receipt)
    assert 'phase="repair"' in c_text and 'verification_round=1' in c_text
