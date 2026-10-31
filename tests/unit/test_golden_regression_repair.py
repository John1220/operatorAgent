"""Golden 的真实故障证人：数组真值、参考缺失、L3 混量及零 GFLOPS。

不靠刷新 registry 或放宽 min_speedup 把回归变绿；旧失败基线仍需 REVIEW。
"""
import contextlib
import io
import json
import sys
from unittest import mock

import numpy as np
import pytest

from operator_agent.execute.mock_gpu import ExecutionResult
from operator_agent.decide.verifier import Verifier
from tests import golden_regression as gr
from tests.backend_fakes import fake_backend


def task_for(case_id):
    return next(c for c in gr.DEFAULT_CASES if c.case_id == case_id).build_task()


@pytest.mark.parametrize("case_id", ["layernorm_mock_py", "relu_mock_py"])
@pytest.mark.parametrize("input_key", ["X", "input"])
@pytest.mark.parametrize("without_torch", [False, True])
def test_array_inputs_have_real_reference(case_id, input_key, without_torch):
    v = Verifier(fake_backend())
    xs = np.arange(-6, 6, dtype=np.float32).reshape(3, 4)
    if case_id.startswith("relu"):
        expected = np.maximum(xs, 0)
    else:
        expected = (xs - xs.mean(-1, keepdims=True)) / np.sqrt(
            xs.var(-1, keepdims=True) + 1e-5)
    ctx = (mock.patch.dict(sys.modules, {"torch": None}) if without_torch
           else contextlib.nullcontext())
    with ctx:
        fn, kind = v._build_reference_fn(task_for(case_id), {input_key: xs})
        assert fn is not None, "不能以仅验证可执行性代替正确性"
        np.testing.assert_allclose(np.asarray(fn()), expected, atol=1e-5)
        if without_torch:
            assert kind == "numpy_reference"


@pytest.mark.parametrize("inputs", [{}, {"X": None}, {"input": None}])
def test_missing_array_is_not_converted_to_scalar_nan(inputs):
    assert Verifier(fake_backend())._build_reference_fn(
        task_for("relu_mock_py"), inputs) == (None, None)


def test_zero_array_does_not_fall_through_to_input():
    fn, _ = Verifier(fake_backend())._build_reference_fn(
        task_for("relu_mock_py"), {"X": np.zeros((2, 3), dtype=np.float32),
                                  "input": np.ones((2, 3), dtype=np.float32)})
    np.testing.assert_array_equal(np.asarray(fn()), np.zeros((2, 3)))


def test_relu_fixture_names_the_operator_instead_of_generic_elementwise():
    assert task_for("relu_mock_py").source.kernel_type == "relu"


@pytest.mark.parametrize("throughput,bandwidth", [(420.0, 21.0), (0.0, 0.0)])
def test_l3_uses_benchmark_metrics_even_if_zero(throughput, bandwidth):
    execute = mock.Mock(side_effect=AssertionError("不能用冷执行覆盖 benchmark"))
    bench = {"avg_time_ms": 0.1, "min_time_ms": 0.1,
             "gflops": throughput, "memory_bandwidth_gbs": bandwidth}
    v = Verifier(fake_backend(benchmark=lambda *a, **kw: dict(bench), execute=execute))
    case = gr.DEFAULT_CASES[0]
    with mock.patch.object(v, "_measure_baseline", return_value={
            "baseline_time_ms": 0.1, "baseline_type": "test_reference"}):
        result = v._verify_performance(case.expected_code, case.build_inputs(),
                                       1e-6, case.build_task())
    assert result.passed, result.message
    assert result.metrics["gflops"] == throughput
    assert result.metrics["memory_bandwidth_gbs"] == bandwidth
    assert result.metrics["avg_time_ms"] == bench["avg_time_ms"]
    execute.assert_not_called()


def test_legacy_backend_without_throughput_keeps_compatibility_fallback():
    execute = mock.Mock(return_value=ExecutionResult(gflops=420.0, memory_bandwidth_gbs=21.0))
    v = Verifier(fake_backend(benchmark=lambda *a, **kw: {
        "avg_time_ms": 0.1, "min_time_ms": 0.1}, execute=execute))
    case = gr.DEFAULT_CASES[0]
    with mock.patch.object(v, "_measure_baseline", return_value={
            "baseline_time_ms": 0.1, "baseline_type": "test_reference"}):
        result = v._verify_performance(case.expected_code, case.build_inputs(),
                                       1e-6, case.build_task())
    assert result.metrics["gflops"] == 420.0
    execute.assert_called_once()


def rec(latency, passed=True):
    return gr.BaselineRecord(case_id="memory", description="test", timestamp=0,
                             all_passed=passed, latency_ms=latency)


@pytest.mark.parametrize("current,expected", [(1.0, "PASS"), (1.1, "PASS"),
                                            (2.0, "FAIL"), (0.5, "REVIEW")])
def test_latency_comparison_preserves_throughput_direction_and_threshold(current, expected):
    status, facts, reason, delta = gr._compare_one(rec(1.0), rec(current), 0.2, True)
    assert status == expected, (facts, reason)
    assert delta == pytest.approx(1.0 / current - 1.0)


def test_zero_gflops_legacy_baseline_is_not_silently_filled_from_current():
    status, facts, _, _ = gr._compare_one(rec(None), rec(1.0), 0.2, True)
    assert status == "REVIEW"
    assert "no_perf_reference" in facts


def test_fixed_kernel_with_old_failing_baseline_still_requires_review():
    status, facts, _, _ = gr._compare_one(rec(1.0, False), rec(1.0), 0.2, True)
    assert status == "REVIEW"
    assert "fixed_since_baseline" in facts


def test_latency_survives_registry_roundtrip_and_old_format_loads(tmp_path):
    h = gr.GoldenRegressionHarness(registry_path=str(tmp_path / "registry.json"))
    h.save_registry({"memory": rec(0.125)})
    assert h.load_registry()["memory"].latency_ms == 0.125
    payload = json.loads((tmp_path / "registry.json").read_text())
    del payload["baselines"]["memory"]["latency_ms"]
    (tmp_path / "registry.json").write_text(json.dumps(payload))
    assert h.load_registry()["memory"].latency_ms is None


@pytest.mark.parametrize("case", gr.DEFAULT_CASES, ids=lambda c: c.case_id)
def test_full_conductor_has_correctness_and_latency_for_each_golden(case, tmp_path):
    # 真跑编译/运行/L2，门槛仍是 CLI 的 0.5x。这里只断言正确性与性能
    # 读数存在，不把受 CPU 负载影响的速度当作固定的单测期望。
    from pathlib import Path
    before = Path(gr.REGISTRY_PATH).read_bytes()
    with contextlib.redirect_stdout(io.StringIO()):
        result = gr.GoldenRegressionHarness(
            registry_path=str(tmp_path / "not-written.json"), cases=[case]).run_case(case)
    assert result.l1_passed and result.l2_passed, result
    assert result.latency_ms is not None and result.latency_ms > 0
    assert Path(gr.REGISTRY_PATH).read_bytes() == before
