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

    receipt = ('output/ac-v08f-pair2-A/run/'
               'native-verification/result-0001.json')
    for arm in ('A', 'C'):
        text = repair_workflow(arm, run_dir='/tmp/run', ssh=dict(host='h', port=1,
                                                                user='u', key='/k'),
                               budget=dict(max_model_calls=20), seed_receipt=receipt,
                               declaration=ac_plan.FUSED_TASK['declaration'])
        assert 'start_kernel.py' in text
        assert '完整 L2' in text and '独立终评' in text
        assert 'fused-ln-residual-1x33-random-contiguous' in text
        assert '9.96e-02' in text          # 回执里这条用例的误差原文
        assert 'phase="generate"' not in text
    c_text = repair_workflow('C', run_dir='/tmp/run',
                             ssh=dict(host='h', port=1, user='u', key='/k'),
                             budget=dict(max_model_calls=20), seed_receipt=receipt,
                             declaration=ac_plan.FUSED_TASK['declaration'])
    assert 'phase="repair"' in c_text and 'verification_round=1' in c_text


def test_the_native_service_refuses_to_start_without_the_feedback_fields():
    """缺 `execution_context` 时不能起服务：否则 check.py 只能说 transport_failed，
    模型整轮看不到 L2 结果却还在花预算（2026-11-19 A 臂实测）。"""
    import pytest

    from benchmarks.experiments.phase1.native import VerificationService

    base = dict(run_dir='/tmp/x', backend='mock', declaration={}, budget={},
                deadline_monotonic=1.0, execution_context={'cell': 't'})
    for missing in ('execution_context', 'deadline_monotonic', 'budget'):
        broken = {k: v for k, v in base.items() if k != missing}
        with pytest.raises(ValueError) as error:
            VerificationService(broken, 'a' * 64)
        assert missing in str(error.value)


def test_the_feedback_reply_carries_the_L2_verdicts(tmp_path):
    """回执里必须有 levels/verdicts：模型就是靠这两个字段知道哪条用例挂了。"""
    from operator_agent.execute.task_clock import monotonic
    from operator_agent.app import hitl_tools as ht
    from benchmarks.experiments.phase1.native import VerificationService

    decl = dict(request='fp16 64x64 residual add layernorm', shapes=[[64, 64]],
                precision='fp16', dsl='python', hardware='mock_gpu', min_speedup=0.0)
    run = tmp_path / 'run'
    prepared = ht.prepare(decl, run, host_budget=dict(
        wall_clock_s=120, max_model_calls=2, max_output_tokens_per_call=1024,
        total_output_token_reservations=2048, max_verifications=2, final_reserve=1))
    service = VerificationService(dict(
        run_dir=str(run), backend='mock', connection={}, declaration=decl,
        execution_context=dict(cell='t'),
        budget=dict(wall_clock_s=120, max_llm_calls=2, oracle_total=2),
        deadline_monotonic=monotonic() + 120), prepared['contract_hash'])
    with service:
        reply = service.develop("def kernel(X, R):\n    return X\n",
                                prepared['contract_hash'])
    assert reply['ok'] is True
    assert reply['execution_context'] == {'cell': 't'}
    assert reply['levels'], '回执必须带 levels'
    assert [v['level'] for v in reply['verdicts']] == ['L2']


def test_the_case_count_in_the_prompt_comes_from_the_contract():
    """融合任务 32 条、matmul 15 条：写死会让另一族的 prompt 报错数。"""
    from benchmarks.experiments.v08.ac_live import repair_workflow

    fused = dict(request='fp16 32768x6144 residual add layernorm', shapes=[[32768, 6144]],
                 precision='fp16', dsl='triton', hardware='nvidia_gpu', min_speedup=1.5)
    matmul = dict(request='fp16 4096x4096x4096 matmul, triton', shapes=[[4096, 4096, 4096]],
                  precision='fp16', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0)
    ssh = dict(host='h', port=1, user='u', key='/k')
    for declaration, expected in ((fused, 32), (matmul, 15)):
        text = repair_workflow('A', run_dir='/tmp/run', ssh=ssh,
                               budget=dict(max_model_calls=20), seed_receipt=None,
                               declaration=declaration)
        assert f'{expected} 条套件' in text, text[:200]


def test_repair_finish_counts_each_dispatch_once_and_skips_L3(tmp_path, monkeypatch):
    """A4 曾记 4 次验证，实际 3 次：A 的会话提交同时留 result-N 与 verify-N。"""
    from benchmarks.experiments.v08 import ac_driver
    from benchmarks.experiments.v08.ac_fixture import _counts, _finish_cell

    run = tmp_path / 'run'
    (run / 'native-verification').mkdir(parents=True)
    for name in ('run/native-verification/result-0001.json', 'run/verify-1.json',
                 'final-run/verify-1.json', 'final-run/verify-2.json'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(created_at=100.0, levels={'L2': True})),
                        encoding='utf-8')
    # 同一份盘面：按臂去重后 A 是 3（1 次会话 + 终评 2 次）
    assert _counts(tmp_path, arm='A')['verifications'] == 3
    assert _counts(tmp_path, arm='C')['verifications'] == 3
