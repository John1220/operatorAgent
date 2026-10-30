from benchmarks.s5_perf_evaluator import qualified_timing


PIN = dict(gpu_index=2, gpu_name='NVIDIA H20')
ARMS = {
    name: dict(cuda_visible_devices='2', count=1, name='NVIDIA H20')
    for name in ('candidate', 'baseline', 'after')
}


def qualified_metrics():
    return dict(
        measurement_validity='valid', measurement_can_gate=True,
        eligibility=dict(can_report=True, can_verdict=True),
        backend_real_device=True, correctness_verified=True,
        gpu_other_procs=7, gpu_shared_with_others=True,
        benchmark_contaminated=False,
        measurement_scope='same_remote_device', measurement_window_scope='same_window',
        paired_samples=20, candidate_median_time_ms=1.0,
        baseline_time_ms=1.1, speedup_vs_baseline=1.1,
    )


def test_idle_known_shared_processes_do_not_override_production_measurement_gate():
    assert qualified_timing(qualified_metrics(), ARMS, PIN)


def test_shared_processes_still_require_uncontaminated_measurement_gate():
    metrics = qualified_metrics()
    metrics['benchmark_contaminated'] = True
    assert not qualified_timing(metrics, ARMS, PIN)

    metrics = qualified_metrics()
    metrics['measurement_can_gate'] = False
    assert not qualified_timing(metrics, ARMS, PIN)


def test_shared_process_count_and_gpu_identity_must_be_consistent():
    metrics = qualified_metrics()
    metrics['gpu_shared_with_others'] = False
    assert not qualified_timing(metrics, ARMS, PIN)

    arms = {**ARMS, 'candidate': dict(cuda_visible_devices='7', count=1, name='NVIDIA H20')}
    assert not qualified_timing(qualified_metrics(), arms, PIN)
