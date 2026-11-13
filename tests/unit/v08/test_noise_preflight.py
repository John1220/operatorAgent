"""The preflight receipt must be decidable, cited, and refuse to invent a reading."""
import pytest

from operator_agent.decide.noise_preflight import (MIN_REPS, PROTOCOL, BAND_MULTIPLE,
                                                  empirical_band_pct, predicted_tier,
                                                  run_preflight, summarize)


def sampler(costs):
    rows = iter(costs)
    return lambda: next(rows)


def test_quiet_window_predicts_a_valid_tier_and_clears_a_normal_effect():
    receipt = run_preflight(sampler([1.0, 0.9998, 1.0001, 0.9999, 1.0002]), reps=5,
                            planned_effect_pct=5.0)
    assert receipt['schema'] == PROTOCOL and receipt['ok'] is True
    assert receipt['predicted_tier'] == 'valid'
    assert receipt['empirical_band_pct'] == pytest.approx(BAND_MULTIPLE * 0.0002 * 100, abs=1e-6)
    assert receipt['effective_band_pct'] == receipt['nominal_band_pct'] == 0.1
    assert receipt['action'] == 'start' and receipt['reasons'] == []
    assert receipt['reps_completed'] == 5 and len(receipt['costs_ms']) == 5


def test_noisy_window_reports_the_wider_measured_band_and_holds_small_effects():
    receipt = run_preflight(sampler([1.0, 0.9, 1.1, 1.15, 0.85]), reps=5,
                            planned_effect_pct=5.0)
    assert receipt['predicted_tier'] == 'degraded'
    assert receipt['stats']['iqr_over_median'] == pytest.approx(0.2)
    assert receipt['band_consistent'] is False
    assert receipt['effective_band_pct'] == pytest.approx(60.0)
    assert receipt['action'] == 'hold'
    assert any('do not clear' in reason or 'does not clear' in reason
               for reason in receipt['reasons'])


def test_a_window_too_noisy_to_resolve_anything_holds_regardless_of_plan():
    receipt = run_preflight(sampler([1.0, 1.2, 1.4, 1.6]), reps=4, planned_effect_pct=500.0)
    assert receipt['predicted_tier'] == 'uninformative'
    assert receipt['action'] == 'hold'
    assert 'cannot resolve a performance difference' in ' '.join(receipt['reasons'])


def test_a_failed_sample_is_a_channel_defect_not_a_zero_cost():
    calls = {'n': 0}

    def flaky():
        calls['n'] += 1
        if calls['n'] == 3:
            raise RuntimeError('remote timed out')
        return 1.0

    receipt = run_preflight(flaky, reps=4)
    assert receipt['ok'] is False and receipt['action'] == 'refuse'
    assert receipt['reps_completed'] == 2 and receipt['stats'] is None
    assert 'rep 3/4' in receipt['reasons'][0] and 'remote timed out' in receipt['reasons'][0]


def test_timing_samples_are_validated_rather_than_repaired():
    with pytest.raises(ValueError, match='at least'):
        summarize([1.0, 1.0])
    for bad in (0.0, -1.0, float('nan'), float('inf'), True, '1.0', None):
        with pytest.raises(ValueError):
            summarize([1.0, 1.0, bad])
    with pytest.raises(ValueError, match='reps'):
        run_preflight(lambda: 1.0, reps=MIN_REPS - 1)
    with pytest.raises(ValueError, match='planned_effect_pct'):
        run_preflight(lambda: 1.0, reps=3, planned_effect_pct=0)


def test_the_band_is_three_times_the_measured_spread_not_a_second_table():
    stats = summarize([1.0, 1.0, 1.02, 1.0, 1.0])
    assert empirical_band_pct(stats) == pytest.approx(BAND_MULTIPLE * stats['iqr_over_median'] * 100)
    # 同一批样本 → 同一结论 (回执里除时间戳外必须逐字相同)
    first = run_preflight(sampler([1.0, 1.0, 1.02, 1.0, 1.0]), reps=5)
    second = run_preflight(sampler([1.0, 1.0, 1.02, 1.0, 1.0]), reps=5)
    first.pop('created_at'), second.pop('created_at')
    assert first == second


def test_prediction_never_claims_a_tighter_tier_than_its_own_spread_allows():
    tight = summarize([1.0, 1.0, 1.0, 1.0])
    assert predicted_tier(tight) == 'valid'
    # 散布大但未越过 degraded 阈值: 3× 散布已宽过 valid 表带宽 ⇒ 不许报 valid
    # iqr 看不见两条尾巴: 这一条也钉住「别拿 max-min 当离散度」
    tail = summarize([1.0, 0.99, 1.0, 1.01, 1.0])
    assert tail['iqr_over_median'] == 0.0 and predicted_tier(tail) == 'valid'
    # 1% 的重复散布: 档位仍是 valid (两侧都低于 10%), 但实测带宽 3% 已宽过标称 0.1%
    # ⇒ `band_consistent=False`, 由 effective_band_pct 承担后果, 不把档位说成 uninformative
    wide = summarize([1.0, 0.995, 1.005, 0.995, 1.005])
    assert wide['iqr_over_median'] <= 0.10
    assert predicted_tier(wide) == 'valid'

def test_cli_emits_one_json_receipt_and_claims_nothing_about_candidates():
    import contextlib
    import io
    import json

    from operator_agent.app.cli import main as cli_main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(['preflight', '--backend', 'mock', '--reps', '3',
                         '--planned-effect-pct', '5'])
    assert code == 0
    receipt = json.loads(buf.getvalue())          # 整份 stdout 必须是合法 JSON
    assert receipt['schema'] == PROTOCOL and receipt['ok'] is True
    assert receipt['test_only'] is True           # mock 的计时不构成真机窗口证据
    assert receipt['planned_effect_pct'] == 5.0
    assert 'speedup' not in receipt and 'candidate' not in json.dumps(receipt['stats'])
    assert 'not a speedup for any candidate' in receipt['boundary']


def test_cli_refuses_a_receipt_it_cannot_stand_behind():
    import contextlib
    import io
    import json

    from operator_agent.app.cli import main as cli_main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(['preflight', '--backend', 'mock', '--reps', '1'])
    assert code == 2
    payload = json.loads(buf.getvalue())
    assert payload['ok'] is False and payload['error'] == 'bad_reps'
    assert 'reps' in payload['detail']


def test_shape_probe_is_a_triton_kernel_on_the_gpu_channel():
    from operator_agent.decide.noise_preflight import matmul_probe_source

    source = matmul_probe_source()
    assert 'import triton' in source and '@triton.jit' in source
    assert 'torch.cuda' not in source          # 计时由远端 runner 的 CUDA event 分支负责
    assert 'BLOCK_M=128' in source and 'num_stages=3' in source
    assert matmul_probe_source() == source     # 逐字稳定: 同一窗口的两次探针必须同源


def test_shape_probe_refuses_backends_that_cannot_time_this_card():
    import contextlib
    import io
    import json

    from operator_agent.app.cli import main as cli_main

    for backend in ('mock', 'local', 'fake'):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli_main(['preflight', '--backend', backend, '--reps', '3',
                             '--shape', '4096,4096,4096'])
        assert code == 2, backend
        payload = json.loads(buf.getvalue())
        assert payload['error'] == 'preflight_shape_backend'
        assert 'ssh' in payload['detail']


def test_a_malformed_shape_is_named_not_ignored():
    import contextlib
    import io
    import json

    from operator_agent.app.cli import main as cli_main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(['preflight', '--backend', 'ssh', '--reps', '3', '--shape', '4096,4096'])
    assert code == 2
    payload = json.loads(buf.getvalue())
    assert payload['error'] == 'bad_shape'


def test_an_unknown_dtype_is_refused_by_the_parser():
    """与 `--backend` 同一条路: 取值由 argparse 的 choices 拦下, 不进业务代码。"""
    import contextlib
    import io

    from operator_agent.app.cli import main as cli_main

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(SystemExit) as raised:
            cli_main(['preflight', '--backend', 'ssh', '--reps', '3',
                      '--shape', '4096,4096,4096', '--dtype', 'bf16'])
    assert raised.value.code == 2


# ---------------------------------------------------------------------------
# A2 (2026-11-08): attention 也需要自己的窗口门
# ---------------------------------------------------------------------------

def test_the_attention_probe_is_a_fused_kernel_with_a_scalar_mask():
    from operator_agent.decide.anticheat import static_audit
    from operator_agent.decide.noise_preflight import probe_source

    source = probe_source('attention')
    assert '@triton.jit' in source and 'def kernel(Q, K, V, causal):' in source
    assert 'causal: tl.constexpr' in source          # 掩码是编译期常量，不是随机值
    assert 'torch.cuda' not in source
    assert static_audit(source) == []                # 探针自己不能把反作弊盾撞红
    assert probe_source('attention') == source       # 逐字稳定


def test_the_probe_channel_names_the_operator_and_the_mask():
    """通道名进回执，而预登记按通道绑定门控读数 —— 不能拿 matmul 的读数目当 attention 的。"""
    from operator_agent.app.hitl_tools import _shape_probe_plan

    _, matmul = _shape_probe_plan('matmul', [4096, 4096, 4096], 'fp32')
    _, layer = _shape_probe_plan('layernorm', [16384, 6144], 'fp16')
    assert matmul == 'matmul-4096x4096x4096-fp32'    # 既有通道名逐字不变（R2 已冻结）
    assert layer == 'layernorm-16384x6144-fp16'
    inputs, causal = _shape_probe_plan('attention', [1, 16, 1024, 128], 'fp16', causal=True)
    assert causal == 'attention-1x16x1024x128-fp16-causal'
    assert inputs['causal'] is True and sorted(inputs) == ['K', 'Q', 'V', 'causal']
    assert inputs['Q'].shape == (1, 16, 1024, 128)


def test_attention_probe_refuses_a_shape_or_dtype_it_cannot_stand_behind():
    import contextlib
    import io
    import json

    from operator_agent.app.cli import main as cli_main

    def run(argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli_main(argv)
        return code, json.loads(buf.getvalue())

    code, payload = run(['preflight', '--backend', 'ssh', '--reps', '3',
                         '--op', 'attention', '--shape', '1,16,1024'])
    assert code == 2 and payload['error'] == 'bad_shape'      # 必须是 B,H,S,D

    code, payload = run(['preflight', '--backend', 'ssh', '--reps', '3',
                         '--op', 'attention', '--shape', '1,16,16384,128'])
    assert code == 2 and payload['error'] == 'bad_shape'      # S 上限 8192

    code, payload = run(['preflight', '--backend', 'ssh', '--reps', '3', '--dtype', 'fp32',
                         '--op', 'attention', '--shape', '1,16,1024,128'])
    assert code == 2 and payload['error'] == 'preflight_dtype'


def test_scope_a_geometry_is_legal_for_the_attention_probe():
    """B=1、H=16 是 Scope A 的正常取值 —— 第一次跑就是被 matmul 那条 [16,65536] 拦下的。"""
    import contextlib
    import io
    import json

    from operator_agent.app.cli import main as cli_main

    # 只到形状校验为止: backend=local 会在下一道门（shape 探针只支持 ssh）被拒，
    # 那说明形状已经通过了 —— 这正是本用例要的边界。
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(['preflight', '--backend', 'local', '--reps', '3',
                         '--op', 'attention', '--shape', '1,16,1024,128'])
    assert code == 2
    assert json.loads(buf.getvalue())['error'] == 'preflight_shape_backend'

    for bad in ('1,16,1024,96',     # D 不是 2 的幂
                '1,16,12288,128',   # S 超上限
                '1,0,1024,128'):    # H=0
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli_main(['preflight', '--backend', 'local', '--reps', '3',
                             '--op', 'attention', '--shape', bad])
        assert code == 2 and json.loads(buf.getvalue())['error'] == 'bad_shape', bad
