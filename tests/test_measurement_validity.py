"""测量有效性三态的回归测试 (docs/research/00 缺口 1, 设计见 docs/research/01 第五节)

钉住三件事:

1. `measurement.assess()` 的分档数学与两个信号 (launch 地板支配 / 基准污染) 的组合
   规则, 含「地板未知」时不得默认放行。
2. `Verifier` 的三条 L3 返回路径都要把三态写进 metrics —— 缺口 1 原来的失效方式
   不是「没算」而是「算出来没人读」(`benchmark_contaminated` 早在 metrics 里,
   conductor/fixer/agents 三个文件 grep 零命中)。
3. `Conductor` 真的按三态改变行为: 测量不可信时既不禁用配方也不累计停滞轮次。
   第 3 条是唯一的端到端断言, 也是这次改造的全部意义所在。

纯本地测试, 不需要 GPU / LLM / 远端。

运行: python3 -m unittest tests.test_measurement_validity -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.measurement import (
    CONTAMINATION_UTIL_PCT, LAUNCH_DOMINANCE_UNINFORMATIVE, LAUNCH_DOMINANCE_VALID,
    MeasurementAssessment, Validity, assess, launch_dominance, net_of_floor,
    net_of_floor_speedup,
)
from operator_agent.remote_executor import RemoteExecutor
from operator_agent.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType,
)
from operator_agent.verifier import Verifier

# H20 实测地板 (docs/research/01 第一节): median=20.24 min=18.69 µs
H20_FLOOR_US = 20.24
H20_FLOOR_MIN_US = 18.69
H20_HW_INFO = {"name": "NVIDIA H20", "device": "cuda", "compute_capability": "9.0"}

PYTHON_MATMUL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"


# ─── 一、分档数学 ────────────────────────────────────────────

class TestLaunchDominance(unittest.TestCase):

    def test_layernorm_measured_case(self):
        """docs/research/01 的 layernorm 1024×768: 候选 median 0.0316 ms"""
        self.assertAlmostEqual(launch_dominance(H20_FLOOR_US, 0.0316), 0.6405, places=4)

    def test_matmul_measured_case(self):
        """docs/research/01 的 matmul 4096³: 候选 median 1.040 ms → 地板占比 1.9%"""
        self.assertAlmostEqual(launch_dominance(H20_FLOOR_US, 1.040), 0.0195, places=4)

    def test_none_when_floor_unknown(self):
        self.assertIsNone(launch_dominance(None, 1.0))

    def test_none_when_time_missing_or_nonpositive(self):
        self.assertIsNone(launch_dominance(H20_FLOOR_US, None))
        self.assertIsNone(launch_dominance(H20_FLOOR_US, 0.0))
        self.assertIsNone(launch_dominance(0.0, 1.0))


class TestAssessTiers(unittest.TestCase):
    """阈值边界。0.10 / 0.30 是 docs/research/01 的经验分界, 不是推导值"""

    def _dom_for(self, dominance):
        """反解出能产生指定 dominance 的 median 时间 (ms)"""
        return (H20_FLOOR_US / 1000.0) / dominance

    def _assess_dom(self, dominance, **kw):
        return assess(median_time_ms=self._dom_for(dominance),
                      launch_floor_us=H20_FLOOR_US, **kw)

    def test_valid_below_lower_threshold(self):
        a = self._assess_dom(LAUNCH_DOMINANCE_VALID * 0.99)
        self.assertIs(a.validity, Validity.VALID)
        self.assertTrue(a.can_disable_recipe)

    def test_degraded_between_thresholds(self):
        a = self._assess_dom(0.20)
        self.assertIs(a.validity, Validity.DEGRADED)
        self.assertTrue(a.can_gate, "degraded 仍可用于通过/不通过判定")
        self.assertFalse(a.can_disable_recipe)
        self.assertFalse(a.can_count_stagnation)

    def test_uninformative_above_upper_threshold(self):
        a = self._assess_dom(LAUNCH_DOMINANCE_UNINFORMATIVE * 1.01)
        self.assertIs(a.validity, Validity.UNINFORMATIVE)
        self.assertFalse(a.can_gate)
        self.assertFalse(a.can_disable_recipe)

    def test_contamination_alone_is_degraded_not_uninformative(self):
        """同机同卡依次测候选与基线, 比值仍保留部分意义, 但不足以支撑禁用配方"""
        a = assess(median_time_ms=1.0, launch_floor_us=H20_FLOOR_US,
                   gpu_util_pct=CONTAMINATION_UTIL_PCT + 1)
        self.assertIs(a.validity, Validity.DEGRADED)
        self.assertTrue(a.contaminated)
        self.assertTrue(a.can_gate)
        self.assertFalse(a.can_disable_recipe)

    def test_contamination_cannot_upgrade_launch_verdict(self):
        a = self._assess_dom(0.9, gpu_util_pct=99)
        self.assertIs(a.validity, Validity.UNINFORMATIVE)

    def test_util_at_threshold_is_not_contamination(self):
        """沿用既有语义: `> 5` 才算, 等于 5 不算"""
        a = assess(median_time_ms=1.0, launch_floor_us=H20_FLOOR_US,
                   gpu_util_pct=CONTAMINATION_UTIL_PCT)
        self.assertFalse(a.contaminated)
        self.assertIs(a.validity, Validity.VALID)


class TestUnknownFloorIsNotSilentlyValid(unittest.TestCase):
    """地板未知 (hardware_caps 里没这台设备) 时不得默默当成有效

    缺证据不等于无效, 但必须显式说出来 —— 否则「没检查」会被下游读成「检查过了」。
    """

    def test_reason_recorded(self):
        a = assess(median_time_ms=0.001, launch_floor_us=None)
        self.assertIs(a.validity, Validity.VALID)
        self.assertTrue(any("未做地板支配检查" in r for r in a.reasons), a.reasons)
        self.assertIsNone(a.launch_dominance)

    def test_metrics_omit_dominance_but_keep_reasons(self):
        m = assess(median_time_ms=0.001, launch_floor_us=None).as_metrics()
        self.assertNotIn("launch_dominance", m)
        self.assertTrue(m["measurement_validity_reasons"])


class TestNetOfFloor(unittest.TestCase):

    def test_subtracts_floor(self):
        # dominance = 20.24/1000 / 0.0316 = 0.6405 → net = 0.0316 * 0.3595
        self.assertAlmostEqual(net_of_floor(H20_FLOOR_US, 0.0316), 0.011361, places=5)

    def test_clamped_at_zero(self):
        self.assertEqual(net_of_floor(H20_FLOOR_US, 0.001), 0.0)

    def test_none_when_floor_unknown(self):
        self.assertIsNone(net_of_floor(None, 1.0))

    def test_net_speedup_is_higher_than_measured_when_candidate_is_faster(self):
        """扣掉一个共同的加性开销后, 快的一侧优势被放大 → 净值比高于实测比

        这正是 launch 地板把加速比压向 1.0 的机制 (docs/research/01)。
        两侧都必须明显大于地板, 否则见下条。
        """
        candidate_ms, baseline_ms = 0.100, 0.150
        measured = baseline_ms / candidate_ms
        net = net_of_floor_speedup(H20_FLOOR_US, candidate_ms, baseline_ms)
        self.assertAlmostEqual(measured, 1.5, places=4)
        self.assertAlmostEqual(net, 1.627, places=3)
        self.assertGreater(net, measured)

    def test_net_speedup_refused_when_either_side_is_floor_dominated(self):
        """layernorm 那一档: 基线 20.6 µs vs 地板 20.24 µs, 净剩 0.36 µs, 低于该
        shape 0.86 µs 的 roofline 下界 —— 差值全是噪声, 相除出来的数字没有意义。
        """
        self.assertIsNone(net_of_floor_speedup(H20_FLOOR_US, 0.0316, 0.0206))
        self.assertIsNone(net_of_floor_speedup(H20_FLOOR_US, 1.000, 0.0206))

    def test_net_speedup_none_when_candidate_swallowed_by_floor(self):
        self.assertIsNone(net_of_floor_speedup(H20_FLOOR_US, 0.001, 0.030))


class TestDocumentedVerdictsAreStableUnderFloorChoice(unittest.TestCase):
    """用 median(20.24) 还是 min(18.69) 当地板, 不得翻转已记录的两个实测结论"""

    def test_layernorm_uninformative_either_way(self):
        for floor in (H20_FLOOR_US, H20_FLOOR_MIN_US):
            with self.subTest(floor=floor):
                self.assertIs(assess(median_time_ms=0.0316, launch_floor_us=floor).validity,
                              Validity.UNINFORMATIVE)

    def test_matmul_valid_either_way(self):
        for floor in (H20_FLOOR_US, H20_FLOOR_MIN_US):
            with self.subTest(floor=floor):
                self.assertIs(assess(median_time_ms=1.040, launch_floor_us=floor).validity,
                              Validity.VALID)


# ─── 二、Verifier 把三态写进 metrics ─────────────────────────

def _remote_verifier(candidate_ms, baseline_ms, util=None, hw_info=H20_HW_INFO):
    """搭一个远程形状的 L3: benchmark_with_baselines 返回同机同卡的候选与各基线"""
    backend = RemoteExecutor(mode="local")
    bench = {
        "success": True, "device": "cuda:0", "gflops": 1000.0,
        "memory_bandwidth_gbs": 500.0,
        "candidate": {"min_time_ms": candidate_ms, "median_time_ms": candidate_ms,
                      "avg_time_ms": candidate_ms, "max_time_ms": candidate_ms, "runs": 30},
        "baselines": {
            "pytorch_eager": {"median_time_ms": baseline_ms, "min_time_ms": baseline_ms},
        },
    }
    if util is not None:
        bench["gpu_util_pct"] = util
        bench["gpu_other_procs"] = 7
    backend.benchmark_with_baselines = lambda *a, **kw: bench
    backend.get_hardware_info = lambda: dict(hw_info)
    return Verifier(backend)


def _task():
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(framework="pytorch", kernel_type="matmul", fused_ops=[],
                          complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="python", compute_units=78, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=1.0, max_error=1e-2),
    )


def _matmul_inputs(n=64):
    rng = np.random.default_rng(0)
    return {"A": rng.standard_normal((n, n)).astype(np.float32),
            "B": rng.standard_normal((n, n)).astype(np.float32)}


class TestVerifierEmitsValidity(unittest.TestCase):

    def _l3(self, verifier):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = verifier._verify_performance(PYTHON_MATMUL, _matmul_inputs(), 0.0, _task())
        return r, buf.getvalue()

    def test_h20_floor_is_picked_up_from_hardware_caps(self):
        r, _ = self._l3(_remote_verifier(candidate_ms=1.040, baseline_ms=1.10))
        self.assertEqual(r.metrics["launch_floor_us"], H20_FLOOR_US)

    def test_big_shape_is_valid(self):
        r, _ = self._l3(_remote_verifier(candidate_ms=1.040, baseline_ms=1.10))
        self.assertEqual(r.metrics["measurement_validity"], Validity.VALID.value)
        self.assertTrue(r.metrics["measurement_can_disable_recipe"])

    def test_tiny_shape_is_uninformative_and_says_so_in_message(self):
        """layernorm 1024×768 那一档: 数字描述的是驱动派发, 不是核质量"""
        r, out = self._l3(_remote_verifier(candidate_ms=0.0316, baseline_ms=0.0206))
        self.assertEqual(r.metrics["measurement_validity"], Validity.UNINFORMATIVE.value)
        self.assertFalse(r.metrics["measurement_can_gate"])
        # message 是 Fixer / SearchAgent 的 prompt 素材, 是唯一保证被下游看到的通道
        self.assertIn("测量有效性=uninformative", r.message)
        self.assertIn("launch 地板", r.message)

    def test_contamination_marks_degraded(self):
        """docs/research/00 记录的实测: matmul 那轮测量前 util=14%"""
        r, _ = self._l3(_remote_verifier(candidate_ms=1.040, baseline_ms=1.10, util=14))
        self.assertTrue(r.metrics["benchmark_contaminated"])
        self.assertEqual(r.metrics["measurement_validity"], Validity.DEGRADED.value)
        self.assertFalse(r.metrics["measurement_can_disable_recipe"])
        self.assertIn("污染", r.message)

    def test_unknown_device_records_unchecked(self):
        """hardware_caps 里没有的设备: 地板未知, 显式记录「未做检查」而不是放行"""
        r, _ = self._l3(_remote_verifier(candidate_ms=0.0001, baseline_ms=0.0002,
                                         hw_info={"name": "Some Unknown Accelerator",
                                                  "device": "cuda", "compute_capability": ""}))
        self.assertIsNone(r.metrics.get("launch_floor_us"))
        self.assertTrue(any("未做地板支配检查" in x
                            for x in r.metrics["measurement_validity_reasons"]))

    def test_net_of_floor_speedup_reported_when_both_sides_clear_the_floor(self):
        r, _ = self._l3(_remote_verifier(candidate_ms=0.100, baseline_ms=0.150))
        self.assertIn("speedup_net_of_floor", r.metrics)
        self.assertGreater(r.metrics["speedup_net_of_floor"],
                           r.metrics["speedup_vs_baseline"])

    def test_net_of_floor_speedup_omitted_when_floor_dominated(self):
        """layernorm 那一档不报第二指标: 净值由噪声相除得来, 报出来比不报更糟"""
        r, _ = self._l3(_remote_verifier(candidate_ms=0.0316, baseline_ms=0.0206))
        self.assertNotIn("speedup_net_of_floor", r.metrics)


# ─── 三、Conductor 真的按三态改变行为 ─────────────────────────

SLOW_MATMUL = """import numpy as np

def kernel(A, B):
    for _ in range(20):
        C = A @ B
    return C
"""

SKETCH = {
    "algorithm": "matmul",
    "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
    "memory_plan": {}, "pipeline": [], "optimization_hints": [],
}


class ScriptedLLM:
    def chat(self, messages, **kw):
        return SLOW_MATMUL

    def chat_json(self, messages, **kw):
        return dict(SKETCH)


class LaunchDominatedGPU(MockGPU):
    """报成 H20 (取得实测地板), 并把候选计时与 gflops 都脚本化

    - 计时: 非远程后端下 L3 的 `kernel_min_ms` 取自 `benchmark()` 的
      median/min/avg, 改这几个键就能把 launch_dominance 推到指定档位。
    - gflops: 非远程后端下 L3 从 `execute(...).gflops` 取值。MockGPU 的 execute
      是真计时, 逐轮会抖 (实测同一核 16.03 / 16.07), 而 `current > best` 的判定
      对这种抖动毫无抵抗 —— 第 3 轮会被读成「改善」并重置停滞计数, 让下面那条
      反向保护用例失效。抖动正是本模块要隔离的东西, 不能拿它当测试激励, 所以钉成
      常量: 两个用例之间唯一的变量就只剩测量有效性。
    """

    def __init__(self, candidate_ms=0.0316, gflops=16.0):
        super().__init__()
        self.name = "NVIDIA H20"
        self._candidate_ms = candidate_ms
        self._gflops = gflops

    def get_hardware_info(self) -> dict:
        info = super().get_hardware_info()
        info.update(H20_HW_INFO)
        return info

    def benchmark(self, kernel_code, inputs, warmup=3, repeat=10):
        result = super().benchmark(kernel_code, inputs, warmup, repeat)
        for key in ("min_time_ms", "median_time_ms", "avg_time_ms", "max_time_ms"):
            result[key] = self._candidate_ms
        return result

    def execute(self, kernel_code, inputs):
        result = super().execute(kernel_code, inputs)
        result.gflops = self._gflops
        return result


class TestConductorIgnoresUninformativeMeasurement(unittest.TestCase):
    """端到端: 不可信的测量不得被当成「性能未改善」

    `min_speedup=5.0` 保证每轮 L3 都 FAILED, 于是每轮都会走到停滞判定; gflops
    钉成常量保证每轮都「未改善」。两个用例唯一的差别是候选计时落在地板的哪一侧。
    """

    maxDiff = 600

    def _run(self, candidate_ms, max_rounds=3):
        backend = LaunchDominatedGPU(candidate_ms)
        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(framework="pytorch", kernel_type="matmul", fused_ops=[],
                              complexity=TaskComplexity.MEDIUM),
            target=TargetSpec(hardware="mock_gpu", dsl="python", compute_units=78, warp_size=32),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=5.0, max_error=1e-5),
        )
        inputs = _matmul_inputs()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            Conductor(backend=backend, kernel_db=KernelDatabase(), llm=ScriptedLLM(),
                      max_optimize_rounds=max_rounds).execute(task, inputs, target_gflops=0.0)
        return buf.getvalue()

    def test_uninformative_rounds_run_full_budget(self):
        out = self._run(0.0316, max_rounds=3)
        self.assertEqual(sum(1 for r in (1, 2, 3) if f"轮次 {r}/3" in out), 3)
        self.assertNotIn("提前终止", out)

    def test_uninformative_rounds_do_not_disable_recipes(self):
        """断言后果而不是日志: 配方被禁用的可观测后果是后续轮次 Planner 无方可选"""
        out = self._run(0.0316, max_rounds=3)
        self.assertNotIn("性能未改善，禁用配方", out)
        self.assertNotIn("本轮无新增优化配方", out)

    def test_uninformative_rounds_are_announced(self):
        out = self._run(0.0316, max_rounds=3)
        self.assertIn("测量有效性=uninformative", out)
        self.assertIn("3 轮的 L3 测量被判为不可信", out)

    def test_valid_measurement_still_terminates_on_real_stagnation(self):
        """反向保护: 不得把改造做成「干脆不做停滞判定」

        候选计时 1.040 ms (matmul 4096³ 那一档) → launch_dominance 1.9% → VALID,
        于是停滞判定照常生效, max_stagnant_rounds=2 下第 3 轮不再执行。
        """
        out = self._run(1.040, max_rounds=3)
        self.assertNotIn("测量有效性=", out)
        self.assertIn("提前终止", out)


class TestMeasurementDrivesLoopDirectly(unittest.TestCase):
    """_measurement_drives_loop 的契约: 缺键时沿用改动前行为"""

    def _l3(self, **metrics):
        from operator_agent.schemas import VerificationLevel, VerificationResult
        return VerificationResult(level=VerificationLevel.L3_PERFORMANCE, passed=False,
                                  message="x", metrics=metrics)

    def test_missing_keys_defaults_to_driving_the_loop(self):
        """旧快照与不产 L3 的路径没有有效性键, 不做追溯性重判"""
        self.assertTrue(Conductor._measurement_drives_loop(self._l3(gflops=1.0), 1))

    def test_assessment_object_flags_are_consistent(self):
        a = MeasurementAssessment(validity=Validity.DEGRADED)
        self.assertTrue(a.can_gate)
        self.assertFalse(a.can_disable_recipe)
        self.assertFalse(a.can_count_stagnation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
