"""测量有效性三态的回归测试 (docs/research/00 缺口 1, 设计见 docs/research/01 第五节)

钉住三件事:

1. `measurement.assess()` 的分档数学与四个信号 (launch 地板支配 / 测量前 util 污染 /
   样本主体离散度 / 窗口内抢占) 的组合规则, 含「地板未知」与「分位数未上报」时
   不得默认放行。
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
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.measurement import (
    CONTAMINATION_UTIL_PCT, DISPERSION_DEGRADED, DISPERSION_UNINFORMATIVE,
    INTRUDER_MAX_OVER_MEDIAN, LAUNCH_DOMINANCE_UNINFORMATIVE, LAUNCH_DOMINANCE_VALID,
    NOISE_BAND, MeasurementAssessment, Validity, Verdict, assess, binding_attainment,
    launch_dominance, roofline_converged, verdict,
)
from operator_agent.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType,
)
from operator_agent.verifier import Verifier, baseline_plan
from tests.backend_fakes import fake_backend

# H20 launch 开销 (空 Triton 核, CUDA event 计时, 不冲刷 L2):
#   docs/research/01 实测 median=20.24 min=18.69 µs
#   benchmarks/calibrate_l2_flush_effect.py 复测 median=20.03 min=18.30 µs
# 两次相差 ~1%, 取先标定的 20.24。
#
# ⚠️ 这个数的含义被复测修正过: 同一个空核在**冲刷 L2 后**只测得 4.64 µs, 所以
# 20.24 µs 不是恒定的执行开销, 而是「空闲 GPU 上 CUDA event 窗口吃进的 launch
# 延迟」。runner 现在按 Triton do_bench 口径冲刷 L2, 报的是净执行时间, 因此
# launch_dominance 的分母必须是 floor+net 而不是 net。详见 docs/research/32。
H20_FLOOR_US = 20.24
H20_FLOOR_MIN_US = 18.69
H20_HW_INFO = {"name": "NVIDIA H20", "device": "cuda", "compute_capability": "9.0"}

# 冲刷 L2 后的真机净执行时间 (benchmarks/calibrate_l2_flush_effect.py, H20)
NET_MS_LAYERNORM_SMALL = 0.00739      # 1024×768, 不冲刷时是 0.02192
NET_MS_LAYERNORM_LARGE = 0.1637       # 16384×6144
NET_MS_MATMUL_4096 = 0.97992          # 4096³ fp16, 不冲刷时是 0.99304

PYTHON_MATMUL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"


# ─── 一、分档数学 ────────────────────────────────────────────

class TestLaunchDominance(unittest.TestCase):

    def test_layernorm_measured_case(self):
        """真机净时间 7.39 µs → launch 占端到端的 73.3%"""
        self.assertAlmostEqual(
            launch_dominance(H20_FLOOR_US, NET_MS_LAYERNORM_SMALL), 0.7325, places=4)

    def test_matmul_measured_case(self):
        """真机净时间 979.9 µs → launch 只占 2.0%"""
        self.assertAlmostEqual(
            launch_dominance(H20_FLOOR_US, NET_MS_MATMUL_4096), 0.0202, places=4)

    def test_never_exceeds_one(self):
        """floor/(floor+net) 恒 < 1 —— 这正是修正前 floor/net 做不到的

        修正前拿「不冲刷口径的地板」除「冲刷口径的净时间」, layernorm 1024×768
        实测算出 2.81。一个 >1 的"占比"是口径混用的信号。
        """
        for net in (1e-9, 1e-6, 0.001, 1.0, 1000.0):
            with self.subTest(net=net):
                self.assertLess(launch_dominance(H20_FLOOR_US, net), 1.0)

    def test_none_when_floor_unknown(self):
        self.assertIsNone(launch_dominance(None, 1.0))

    def test_none_when_time_missing_or_nonpositive(self):
        self.assertIsNone(launch_dominance(H20_FLOOR_US, None))
        self.assertIsNone(launch_dominance(H20_FLOOR_US, 0.0))
        self.assertIsNone(launch_dominance(0.0, 1.0))


class TestAssessTiers(unittest.TestCase):
    """阈值边界。0.10 / 0.30 是 docs/research/01 的经验分界, 不是推导值"""

    def _dom_for(self, dominance):
        """反解出能产生指定 dominance 的净时间 (ms): d = f/(f+net) → net = f(1-d)/d"""
        f = H20_FLOOR_US / 1000.0
        return f * (1.0 - dominance) / dominance

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


class TestNoNetOfFloorMetric(unittest.TestCase):
    """「扣除地板后的净值/净加速比」这个指标已被删除, 别再把它加回来

    它建立在「runner 报的 median 含 launch 开销」这个前提上。真机标定
    (benchmarks/calibrate_l2_flush_effect.py) 推翻了前提: runner 现在按 Triton
    do_bench 口径每次计时前冲刷 256 MB L2, 冲刷把 GPU 填满, launch 延迟被藏在
    事件窗口之外, 所以报出来的**已经是净执行时间**。再减一次地板是重复扣除。

    而且它在地板支配区会产出低于物理下界的噪声: layernorm 1024×768 的 Eager
    基线不冲刷时 median=19.62 µs 而地板 20.24 µs, 扣完是负的。
    """

    def test_module_exposes_no_net_of_floor(self):
        import operator_agent.measurement as mod
        self.assertFalse([n for n in dir(mod) if "net_of_floor" in n],
                         "net_of_floor* 已删除: 冲刷后的 median 本身就是净值")

    def test_metrics_carry_no_net_of_floor_keys(self):
        m = assess(median_time_ms=0.1, launch_floor_us=H20_FLOOR_US).as_metrics()
        self.assertFalse([k for k in m if "net_of_floor" in k], m)


class TestDocumentedVerdictsAreStableUnderFloorChoice(unittest.TestCase):
    """用 median(20.24) 还是 min(18.69) 当地板, 不得翻转已记录的两个实测结论"""

    def test_layernorm_uninformative_either_way(self):
        for floor in (H20_FLOOR_US, H20_FLOOR_MIN_US):
            with self.subTest(floor=floor):
                self.assertIs(
                    assess(median_time_ms=NET_MS_LAYERNORM_SMALL,
                           launch_floor_us=floor).validity,
                    Validity.UNINFORMATIVE)

    def test_large_layernorm_degraded_either_way(self):
        for floor in (H20_FLOOR_US, H20_FLOOR_MIN_US):
            with self.subTest(floor=floor):
                self.assertIs(
                    assess(median_time_ms=NET_MS_LAYERNORM_LARGE,
                           launch_floor_us=floor).validity,
                    Validity.DEGRADED)

    def test_matmul_valid_either_way(self):
        for floor in (H20_FLOOR_US, H20_FLOOR_MIN_US):
            with self.subTest(floor=floor):
                self.assertIs(
                    assess(median_time_ms=NET_MS_MATMUL_4096,
                           launch_floor_us=floor).validity,
                    Validity.VALID)


# ─── 一.5、基线偏好名单 (D3: 策略在控制端, 可用性在远端) ─────

class TestBaselinePlan(unittest.TestCase):
    """`baseline_plan` 是**唯一**的基线公平性策略, 三档都不能飘。

    飘了会怎样: 远端拿到一个它不认识的名单就退回自己的默认顺序, 于是「fp16 候选比
    cuBLAS fp32」这种不公平的比较会在没人报错的情况下回来。
    """

    def test_fp16_matmul_prefers_cublas_fp16(self):
        plan = baseline_plan("matmul", "fp16")
        self.assertEqual("pytorch_eager_fp16", plan[0])
        self.assertEqual("torch_compile", plan[-1])

    def test_fp32_matmul_prefers_tf32(self):
        plan = baseline_plan("matmul", "fp32")
        self.assertEqual("pytorch_eager_tf32", plan[0])
        self.assertNotIn("pytorch_eager_fp16", plan,
                         "fp32 任务里出现 fp16 基线 = 拿候选的精度不当挡箭牌")

    def test_gemm_counts_as_matmul(self):
        self.assertEqual(baseline_plan("matmul", "fp32"), baseline_plan("gemm", "fp32"))

    def test_non_matmul_has_two_ranks(self):
        self.assertEqual(["pytorch_eager", "torch_compile"],
                         baseline_plan("layernorm", "fp16"))

    def test_every_name_exists_in_the_runner_spec_table(self):
        """跨文件守卫: 名单里的名字必须是远端 `_baseline_specs` 真会建出来的。

        名字对不上时远端会静默跳过, 策略退化成「按 specs 的书写顺序」—— 典型的
        「 producer 与 consumer 词汇漂移」, 光看单文件测不出来。
        """
        src = (Path(__file__).resolve().parents[1] / "remote_scripts"
               / "run_kernel.py").read_text(encoding="utf-8")
        specs = set(re.findall(r'specs\["([a-z0-9_]+)"\]', src))
        self.assertTrue(specs, f"没从 runner 里认出任何基线名: {specs}")
        for kernel_type, prec in (("matmul", "fp16"), ("matmul", "fp32"),
                                  ("layernorm", "fp16"), ("softmax", "bf16")):
            plan = baseline_plan(kernel_type, prec)
            unknown = [n for n in plan if n not in specs]
            self.assertEqual([], unknown,
                             f"{kernel_type}/{prec} 的名单里有远端建不出的基线: {unknown}")


# ─── 二、Verifier 把三态写进 metrics ─────────────────────────

def _remote_verifier(candidate_ms, baseline_ms, util=None, hw_info=H20_HW_INFO,
                     dispersion=None, p25=None, p75=None, max_over_median=None,
                     cand_peak=None, base_peak=None, block_medians=None):
    """搭一个远程形状的 L3: benchmark_with_baselines 返回同机同卡的候选 + 一个基线

    D3 之后远端只测**一个**基线并回报 `baseline_selected`, 所以这里的 payload 也只带
    一份基线 —— 写两份会让测试继续默认「全都测过」这个已经被换掉的假设。
    """
    candidate = {"min_time_ms": candidate_ms, "median_time_ms": candidate_ms,
                 "avg_time_ms": candidate_ms, "max_time_ms": candidate_ms, "runs": 30}
    if dispersion is not None:
        candidate["dispersion_iqr_over_median"] = dispersion
    if max_over_median is not None:
        candidate["dispersion_max_over_median"] = max_over_median
    if p25 is not None:
        candidate["p25_time_ms"] = p25
    if p75 is not None:
        candidate["p75_time_ms"] = p75
    bench = {
        "success": True, "device": "cuda:0", "gflops": 1000.0,
        "memory_bandwidth_gbs": 500.0,
        "candidate": candidate,
        "baselines": {
            "pytorch_eager": {"median_time_ms": baseline_ms, "min_time_ms": baseline_ms},
        },
        # 远端现在只测一个基线并回报选了哪个 (D3); verifier 不再本地重排
        "baseline_selected": "pytorch_eager",
        "paired": {"speedup_paired_median": baseline_ms / candidate_ms,
                   "speedup_paired_iqr_over_median": 0.02,
                   "paired_runs": 30, "paired_block_samples": 10},
    }
    # 峰值显存 (KernelBench-Verified 那三条里的「28% 的核增加峰值显存」) 只有 runner
    # 报了才搬; 没报时留 None, 让「旧 runner」与「量具坏了」在日志里可分辨。
    if cand_peak is not None:
        bench["paired"]["candidate_peak_bytes_extra"] = cand_peak
    if base_peak is not None:
        bench["paired"]["baseline_peak_bytes_extra"] = base_peak
    if cand_peak is not None and base_peak:
        bench["paired"]["peak_memory_ratio"] = round(cand_peak / base_peak, 3)
    if block_medians is not None:
        bench["paired"]["block_ratio_medians"] = block_medians
    if util is not None:
        bench["gpu_util_pct"] = util
        bench["gpu_other_procs"] = 7
    backend = fake_backend(benchmark_with_baselines=lambda *a, **kw: bench,
                           get_hardware_info=lambda: dict(hw_info))
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
        r, _ = self._l3(_remote_verifier(candidate_ms=NET_MS_MATMUL_4096, baseline_ms=1.10))
        self.assertEqual(r.metrics["launch_floor_us"], H20_FLOOR_US)

    def test_big_shape_is_valid(self):
        r, _ = self._l3(_remote_verifier(candidate_ms=NET_MS_MATMUL_4096, baseline_ms=1.10))
        self.assertEqual(r.metrics["measurement_validity"], Validity.VALID.value)
        self.assertTrue(r.metrics["measurement_can_disable_recipe"])

    def test_tiny_shape_is_uninformative_and_says_so_in_message(self):
        """layernorm 1024×768 那一档: 净时间 7.39 µs, launch 占端到端 73%"""
        r, out = self._l3(_remote_verifier(candidate_ms=NET_MS_LAYERNORM_SMALL,
                                           baseline_ms=0.00995))
        self.assertEqual(r.metrics["measurement_validity"], Validity.UNINFORMATIVE.value)
        self.assertFalse(r.metrics["measurement_can_gate"])
        # message 是 Fixer / SearchAgent 的 prompt 素材, 是唯一保证被下游看到的通道
        self.assertIn("测量有效性=uninformative", r.message)
        self.assertIn("launch", r.message)

    def test_contamination_marks_degraded(self):
        """docs/research/00 记录的实测: matmul 那轮测量前 util=14%"""
        r, _ = self._l3(_remote_verifier(candidate_ms=NET_MS_MATMUL_4096,
                                         baseline_ms=1.10, util=14))
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

class TestPeakMemoryObservation(unittest.TestCase):
    """峰值显存必须从 runner 一路走到 metrics 与 message (KernelBench-Verified 第三条)

    外部依据 (✅ 一手核过 abstract 原句): "On the memory front, 28% of GPU kernels
    generated by the best model increase peak GPU memory usage." 本项目的 L3 此前只量
    时间, 所以这个失效模式整个看不见 —— 而 `Conductor` 现在会把 speedup 写进跨运行语料
    (docs/research/43), 「快但吃显存」的核因此会被固化成下一次的示范。
    这里只**观测**不判定 (本机还没有 ratio 的分布), 阈值条件见 docs/research/44。
    """

    def _l3(self, verifier):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = verifier._verify_performance(PYTHON_MATMUL, _matmul_inputs(), 0.0, _task())
        return r, buf.getvalue()

    def test_bytes_and_ratio_reach_metrics_and_message(self):
        mib = 1048576
        r, _ = self._l3(_remote_verifier(candidate_ms=NET_MS_MATMUL_4096, baseline_ms=1.10,
                                         cand_peak=6 * mib, base_peak=2 * mib))
        m = r.metrics
        self.assertEqual(6 * mib, m["candidate_peak_memory_bytes"])
        self.assertEqual(2 * mib, m["baseline_peak_memory_bytes"])
        self.assertEqual(3.0, m["peak_memory_ratio"])
        # message 是唯一保证被 Fixer/SearchAgent 看到的通道, 绝对值也要在里面
        self.assertIn("6.0 MiB", r.message)
        self.assertIn("2.0 MiB", r.message)

    def test_old_runner_still_works_and_reports_none(self):
        """runner 没报（旧版本 / 非 CUDA）时必须是 None, 不能折成 0 ——
        0 会被读成「这个核不吃显存」, 而真实含义是「没量到」。"""
        r, _ = self._l3(_remote_verifier(candidate_ms=NET_MS_MATMUL_4096, baseline_ms=1.10))
        self.assertIsNone(r.metrics["candidate_peak_memory_bytes"])
        self.assertIsNone(r.metrics["peak_memory_ratio"])
        self.assertNotIn("峰值显存", r.message)


class TestBlockSeries(unittest.TestCase):
    """逐块比值序列 (D5 的唯一可判输入, docs/research/46)

    真机已经量过: 一段 1.5 ms 的 host 侧抢占打在 20% 的块上, 那一块的比值掉到 0.0054
    (=候选看起来慢 186 倍), 而**同一次 run 的全局 median 一分没动**, iqr/median 只从
    0.0309 走到 0.0388。所以聚合口径对成段污染是聋的, 块序列不聋 —— 前提是这个数
    真的被搬进 metrics 并且留在那儿能被复盘。
    """

    def _l3(self, v):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = v._verify_performance(PYTHON_MATMUL, _matmul_inputs(), 0.0, _task())
        return r, buf.getvalue()

    def test_block_series_reaches_metrics(self):
        series = [1.001, 0.998, 0.005, 1.002]
        r, _ = self._l3(_remote_verifier(candidate_ms=NET_MS_MATMUL_4096, baseline_ms=1.10,
                                         block_medians=series))
        self.assertEqual(series, r.metrics["block_ratio_medians"],
                         "块序列没搬进 metrics —— D5 的可判输入到不了下游")

    def test_absent_series_is_none_not_empty(self):
        """旧 runner 不报这个键时必须是 None: 空列表会被将来的停止规则读成「零块 = 无证据」
        而不是「没有证据」"""
        r, _ = self._l3(_remote_verifier(candidate_ms=NET_MS_MATMUL_4096, baseline_ms=1.10))
        self.assertIsNone(r.metrics["block_ratio_medians"])


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

    def __init__(self, candidate_ms=NET_MS_LAYERNORM_SMALL, gflops=16.0):
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
        out = self._run(NET_MS_LAYERNORM_SMALL, max_rounds=3)
        self.assertEqual(sum(1 for r in (1, 2, 3) if f"轮次 {r}/3" in out), 3)
        self.assertNotIn("提前终止", out)

    def test_uninformative_rounds_do_not_disable_recipes(self):
        """断言后果而不是日志: 配方被禁用的可观测后果是后续轮次 Planner 无方可选"""
        out = self._run(NET_MS_LAYERNORM_SMALL, max_rounds=3)
        self.assertNotIn("性能未改善，禁用配方", out)
        self.assertNotIn("本轮无新增优化配方", out)

    def test_uninformative_rounds_are_announced(self):
        out = self._run(NET_MS_LAYERNORM_SMALL, max_rounds=3)
        self.assertIn("测量有效性=uninformative", out)
        self.assertIn("3 轮的 L3 测量被判为不可信", out)

    def test_valid_measurement_still_terminates_on_real_stagnation(self):
        """反向保护: 不得把改造做成「干脆不做停滞判定」

        候选净时间 979.9 µs (matmul 4096³ 那一档) → launch_dominance 2.0% → VALID,
        于是停滞判定照常生效, max_stagnant_rounds=2 下第 3 轮不再执行。
        """
        out = self._run(NET_MS_MATMUL_4096, max_rounds=3)
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


class TestDispersionGate(unittest.TestCase):
    """第三个信号: 样本自身在抖 (docs/research/20 §3.1 D1)

    runner 本来就跑 n 个样本, 多报 p25/p75 **不花任何额外 GPU 时间**, 于是下游
    能区分「核真的慢」和「这台机器此刻在抖」。以前只报 min/median/avg/max,
    从不看它们散不散 —— 30 个未收敛样本给出的 median 看起来权威, 实则是噪声,
    而 Conductor 拿它去禁用配方、累计停滞轮次。
    """

    def test_tight_samples_stay_valid(self):
        a = assess(median_time_ms=NET_MS_MATMUL_4096, launch_floor_us=H20_FLOOR_US,
                   dispersion=0.01)
        self.assertIs(a.validity, Validity.VALID)
        self.assertEqual(a.dispersion, 0.01)
        self.assertEqual([], a.reasons)

    def test_loose_samples_are_degraded_and_say_what_to_do(self):
        a = assess(median_time_ms=NET_MS_MATMUL_4096, launch_floor_us=H20_FLOOR_US,
                   dispersion=0.15)
        self.assertIs(a.validity, Validity.DEGRADED)
        self.assertTrue(a.can_gate, "DEGRADED 仍可参与通过/不通过判定")
        self.assertFalse(a.can_disable_recipe)
        self.assertFalse(a.can_count_stagnation)
        # reason 是 Fixer / SearchAgent 的 prompt 素材, 必须带处置动作
        self.assertTrue(any("离散度" in r and ("禁用配方" in r or "停滞" in r)
                            for r in a.reasons), a.reasons)

    def test_very_loose_samples_are_uninformative(self):
        a = assess(median_time_ms=NET_MS_MATMUL_4096, launch_floor_us=H20_FLOOR_US,
                   dispersion=0.45)
        self.assertIs(a.validity, Validity.UNINFORMATIVE)
        self.assertFalse(a.can_gate)
        self.assertTrue(any("重测" in r for r in a.reasons), a.reasons)

    def test_thresholds_match_the_documented_anchors(self):
        """0.10 锚在 PyTorch _IQR_WARN_THRESHOLD; 0.30 无外部锚点, 是经验值。

        把常量钉在这里, 改动时必须同时改文档, 不能悄悄漂。
        """
        self.assertEqual(0.10, DISPERSION_DEGRADED)
        self.assertEqual(0.30, DISPERSION_UNINFORMATIVE)

    def test_boundary_is_inclusive_at_zero_and_exclusive_at_threshold(self):
        self.assertIs(assess(median_time_ms=1.0, launch_floor_us=H20_FLOOR_US,
                             dispersion=0.0).validity, Validity.VALID)
        self.assertIs(assess(median_time_ms=1.0, launch_floor_us=H20_FLOOR_US,
                             dispersion=DISPERSION_DEGRADED).validity, Validity.VALID)

    def test_absent_dispersion_does_not_change_the_verdict(self):
        """旧 runner / MockGPU 不上报分位数: 不得因此把有效测量判成不可信。"""
        with_d = assess(median_time_ms=NET_MS_MATMUL_4096, launch_floor_us=H20_FLOOR_US)
        self.assertIs(with_d.validity, Validity.VALID)
        self.assertIsNone(with_d.dispersion)
        self.assertNotIn("dispersion_iqr_over_median", with_d.as_metrics())

    def test_worse_signal_wins_when_two_fire_at_once(self):
        """离散度只到 DEGRADED, 但地板支配到 UNINFORMATIVE → 取更差的。"""
        a = assess(median_time_ms=NET_MS_LAYERNORM_SMALL, launch_floor_us=H20_FLOOR_US,
                   dispersion=0.15)
        self.assertIs(a.validity, Validity.UNINFORMATIVE)
        self.assertEqual(2, len([r for r in a.reasons if "离散度" in r or "launch" in r]))

    def test_contamination_and_dispersion_both_recorded(self):
        a = assess(median_time_ms=NET_MS_MATMUL_4096, launch_floor_us=H20_FLOOR_US,
                   gpu_util_pct=14.0, dispersion=0.15)
        self.assertTrue(a.contaminated)
        self.assertIs(a.validity, Validity.DEGRADED)
        self.assertEqual(2, len(a.reasons))


class TestWindowIntruderSignal(unittest.TestCase):
    """max/median 是**诊断信号, 不改变判定** —— 这个性质必须钉住。

    它本来是被设计成判据的, 真机验证把它推翻了 (docs/research/33): runner 按毫秒
    预算自动加采样, 而 max 是顺序统计量, 样本越多越靠尾部。两者同向 —— 能通过
    「核足够长」这个前提的区间恰好是样本数最大的区间。实测 layernorm 16384×6144 在
    util=0%、完全没有邻居插队的情况下就报出 max/median = 1.86, 逼近阈值 2.0。
    拿它当判据会系统性地把干净测量判成污染, 也就是把闭环静默关掉 —— 正是缺口 1
    最初那个病。

    所以这里断言的不是「它能拦住什么」, 而是「它拦不住什么」。将来若换成对 n 稳健
    的统计量 (如「> 2×median 的样本占比」) 并重新赋予判据地位, 这些用例要一起改。
    """

    def test_never_changes_validity_even_at_absurd_ratios(self):
        for ratio in (2.1, 19.17, 500.0):
            a = assess(median_time_ms=NET_MS_MATMUL_4096, launch_floor_us=H20_FLOOR_US,
                       dispersion=0.001, max_over_median=ratio)
            self.assertIs(a.validity, Validity.VALID,
                          f"max/median={ratio} 不该改变档位 (原因项: {a.reasons})")
            self.assertTrue(a.can_disable_recipe,
                            f"max/median={ratio} 不该拿走禁用配方的权限")
            self.assertTrue(a.can_count_stagnation)

    def test_still_recorded_as_an_observation(self):
        """不改变判定 ≠ 不吭声。reasons 会进 Fixer / SearchAgent 的 prompt。"""
        a = assess(median_time_ms=NET_MS_MATMUL_4096, launch_floor_us=H20_FLOOR_US,
                   dispersion=0.001, max_over_median=19.17)
        self.assertTrue(a.intruder_suspected)
        self.assertTrue(a.as_metrics()["measurement_intruder_suspected"])
        self.assertEqual(19.17, a.as_metrics()["dispersion_max_over_median"])
        self.assertTrue(any("仅诊断" in r for r in a.reasons),
                        f"观察项必须自我声明不影响判定: {a.reasons}")

    def test_below_threshold_is_silent(self):
        a = assess(median_time_ms=NET_MS_MATMUL_4096, launch_floor_us=H20_FLOOR_US,
                   dispersion=0.001, max_over_median=1.86)
        self.assertFalse(a.intruder_suspected)
        self.assertEqual([], a.reasons)

    def test_absent_signal_is_silent(self):
        a = assess(median_time_ms=NET_MS_MATMUL_4096, launch_floor_us=H20_FLOOR_US)
        self.assertFalse(a.intruder_suspected)
        self.assertIsNone(a.max_over_median)
        self.assertIn("measurement_intruder_suspected", a.as_metrics())

    def test_real_machine_clean_measurement_stays_valid(self):
        """回归锚点: 真机那一行「干净但比值高」的数据, 判出来必须是 VALID。

        layernorm 16384×6144, 实测 median 0.1636 ms / 239 样本 / util=0% /
        max_over_median 1.86 / iqr_over_median 0.0105 / dominance 0.11。
        """
        a = assess(median_time_ms=0.1636, launch_floor_us=H20_FLOOR_US,
                   gpu_util_pct=0.0, dispersion=0.0105, max_over_median=1.86)
        self.assertIs(a.validity, Validity.DEGRADED,
                      "DEGRADED 只应来自 dominance=0.11 越过 0.10, 与抢占无关")
        self.assertEqual(1, len([r for r in a.reasons if "地板" in r]), a.reasons)
        self.assertFalse(any("抢占" in r for r in a.reasons), a.reasons)

    def test_threshold_constant_is_two(self):
        self.assertEqual(2.0, INTRUDER_MAX_OVER_MEDIAN)


class TestRunnerReportsDispersion(unittest.TestCase):
    """runner 侧: _percentile 与 numpy 同口径, _sample_stats 三条路径共用。

    直接 import remote_scripts/run_kernel.py —— 它只依赖 numpy, torch 是函数内
    延迟导入, 所以本地无 GPU 也能测这两个纯函数。
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "remote_scripts" / "run_kernel.py"
        spec = importlib.util.spec_from_file_location("_rk_under_test", path)
        cls.rk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.rk)

    def test_percentile_matches_numpy(self):
        rng = np.random.default_rng(7)
        for n in (1, 2, 5, 30, 31, 200):
            samples = sorted(rng.lognormal(0.0, 0.3, n).tolist())
            for q in (0, 25, 50, 75, 100):
                self.assertAlmostEqual(
                    float(np.percentile(samples, q)),
                    self.rk._percentile(samples, q), places=9,
                    msg=f"n={n} q={q} 与 numpy 口径不一致")

    def test_iqr_is_blind_to_a_single_outlier_and_max_over_median_is_not(self):
        """29 个稳定样本 + 1 个被抢占的样本。

        这是本机的真实污染形态, 也是这次吸纳最重要的发现: **iqr/median 完全看不
        见它** (p25=p75=1.0 → iqr=0), 因为分位数本来就对少量离群点稳健 —— 当决策
        统计量是 median 时这是优点。但 gpu_util 只在测量前采一次快照, 窗口中途插
        进来的抢占它同样看不见, 于是「测量期间发生过抢占」以前没有任何证据通道。
        max/median 补的就是这个洞, 两个统计量互补, 不可互相替代。
        """
        samples = [1.0] * 29 + [50.0]
        stats = self.rk._sample_stats(samples)
        self.assertEqual(1.0, stats["median_time_ms"])
        self.assertEqual(0.0, stats["dispersion_iqr_over_median"],
                         "iqr 应当看不见单个离群点 —— 它若看见了说明口径变了")
        self.assertEqual(50.0, stats["dispersion_max_over_median"])
        self.assertEqual(30, stats["runs"])

    def test_bulk_noise_moves_iqr_but_not_max_over_median_alone(self):
        """反过来: 主体在抖时 iqr 必须报警, 不能只靠 max/median。"""
        rng = np.random.default_rng(3)
        samples = (1.0 + rng.normal(0.0, 0.25, 40)).tolist()
        stats = self.rk._sample_stats(samples)
        self.assertGreater(stats["dispersion_iqr_over_median"], 0.2)

    def test_stats_survive_degenerate_input(self):
        self.assertEqual(0.0, self.rk._sample_stats([0.0, 0.0, 0.0])["median_time_ms"])
        # median 为 0 时不能抛 ZeroDivisionError
        degenerate = self.rk._sample_stats([0.0, 0.0])
        self.assertIsNone(degenerate["dispersion_iqr_over_median"])
        self.assertIsNone(degenerate["dispersion_max_over_median"])
        self.assertEqual(1, self.rk._sample_stats([2.5])["runs"])

    def test_selection_stops_at_the_first_usable_baseline(self):
        """远端按名单试到第一个能跑的就必须停 (省 GPU 时间那一半)

        守卫的是控制流而不是字面量: 找那个遍历偏好名单的 for, 要求它在选中后 break。
        没有 break 就等于回到「2~3 个基线全测一遍, 只用一个」。
        """
        import ast
        src = (Path(__file__).resolve().parents[1] / "remote_scripts"
               / "run_kernel.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "run_benchmark_with_baselines"), None)
        self.assertIsNotNone(fn, "找不到 run_benchmark_with_baselines")
        loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)
                 and any(isinstance(x, ast.Assign)
                         and any(getattr(y, "id", "") == "chosen"
                                 for y in x.targets)
                         for x in ast.walk(n))
                 and any(isinstance(x, ast.Break) for x in ast.walk(n))]
        self.assertEqual(1, len(loops),
                         "偏好名单的遍历里没有「选中即 break」—— 基线又被全测一遍了")

    def test_paired_function_actually_alternates_both_arms(self):
        """交错函数里必须**两臂都出现** _timed_runs, 否则「交错」只是名字

        退化成「候选测完再测基线」在文本上完全可以保持函数名与返回结构不变, 只有
        结构断言抓得住 —— 这正是 D3 唯一真金白银的那一半。
        """
        import ast
        src = (Path(__file__).resolve().parents[1] / "remote_scripts"
               / "run_kernel.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "_bench_pair_interleaved"), None)
        self.assertIsNotNone(fn, "交错配对函数不见了")
        arms = set()
        for c in ast.walk(fn):
            if (isinstance(c, ast.Call)
                    and getattr(c.func, "id", None) == "_timed_runs" and c.args):
                name = getattr(c.args[0], "id", "")
                if name.endswith("_fn"):
                    arms.add(name)
        self.assertEqual({"cand_fn", "base_fn"}, arms,
                         f"交错里只计时的臂: {arms} —— 另一臂没在同一个函数里被采样")
        # 且必须是**同一个循环体**内交替, 而不是两个先后出现的 for
        loops = [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While))]
        both = [n for n in loops
                if {"cand_fn", "base_fn"} <= {getattr(c.args[0], "id", "")
                                              for c in ast.walk(n)
                                              if isinstance(c, ast.Call)
                                              and getattr(c.func, "id", None) == "_timed_runs"}]
        self.assertTrue(both, "两臂的计时不在同一个循环里 —— 那不是交错")
        # 定义了不等于在用: L3 入口必须真的调它。变异 N8 实测过这个洞 —— 退回旧的
        # 「候选测完再测基线」时, 配对函数的定义、形参、返回结构都可以原样留着。
        caller = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                       and n.name == "run_benchmark_with_baselines"), None)
        self.assertIsNotNone(caller, "L3 计时入口 run_benchmark_with_baselines 不见了")
        called = {getattr(c.func, "id", "") for c in ast.walk(caller) if isinstance(c, ast.Call)}
        self.assertIn("_bench_pair_interleaved", called,
                      "L3 入口没有调用交错配对函数 —— D3 被回退了")

    def test_microsecond_median_is_not_quantized(self):
        """3 µs 量级的中位数不许被 round() 压成 0.1 µs 格点

        后果不是「数字不好看」: 五个重复各自量化到同一个值, iqr 就成了 0,
        离散度门在这样的 shape 上彻底失明。
        """
        # 真实相对散布 ~0.08%, 全部落在 4 位小数的同一个格点 (0.0031) 里
        close = [0.0031234, 0.0031238, 0.0031245, 0.0031251, 0.0031259]
        shifted = [x * 1.0005 for x in close]          # 只挪 0.05%
        a = self.rk._sample_stats(close)
        b = self.rk._sample_stats(shifted)
        self.assertNotEqual(a["median_time_ms"], b["median_time_ms"],
                            "相差 0.05% 的两组样本被 round 成了同一个中位数")
        self.assertGreater(a["dispersion_iqr_over_median"], 0.0,
                           "四分位距被量化压成 0 —— 离散度门在这种 shape 上会失明")
        self.assertGreater(a["max_time_ms"], a["min_time_ms"],
                           "max/min 撞进同一个格点, 抢占信号也就没了")

    def test_all_three_benchmark_paths_report_the_same_keys(self):
        """三条路径 (含基线 / triton 单测 / torch-numpy 单测) 必须口径一致。

        只有 _bench_callable 报分位数的话, 走另两条路径时离散度门会静默失效 ——
        与「算了没人读」是同一类缺陷。
        """
        import ast
        src = (Path(__file__).resolve().parents[1] / "remote_scripts"
               / "run_kernel.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        # 数「调用点个数」是脆的 (交错路径天然要调两次), 要数的是**路径**:
        # 每个 _bench* 计时助手都必须把样本交给 _sample_stats 汇总。
        helpers = {n.name for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name.startswith("_bench")}
        callers = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and any(isinstance(c, ast.Call)
                           and getattr(c.func, "id", None) == "_sample_stats"
                           for c in ast.walk(n))}
        self.assertTrue(helpers, "没认出任何 _bench* 计时助手, 这条断言形同虚设")
        self.assertEqual(helpers, helpers & callers,
                         f"这些计时路径没走 _sample_stats: {sorted(helpers - callers)}")
        self.assertIn("_bench_pair_interleaved", helpers,
                      "交错配对路径不见了 —— D3 被回退了?")
        # 旧的逐字段拼装不允许残留 (它会漏掉分位数)
        self.assertNotIn('"min_time_ms": round(min(times)', src)
        # gflops / bandwidth 必须一律由 median 推出, 不许有路径用 avg
        self.assertNotIn("_compute_metrics(real_inputs, avg_time)", src)


class TestVerifierPropagatesDispersion(unittest.TestCase):

    def _l3(self, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = _remote_verifier(**kw)._verify_performance(
                PYTHON_MATMUL, _matmul_inputs(), 0.0, _task())
        return r

    def test_dispersion_lands_in_metrics_and_message(self):
        r = self._l3(candidate_ms=1.040, baseline_ms=1.10, dispersion=0.42,
                     p25=0.9, p75=1.3)
        self.assertEqual(0.42, r.metrics["dispersion_iqr_over_median"])
        self.assertEqual(0.9, r.metrics["p25_time_ms"])
        self.assertEqual(1.3, r.metrics["p75_time_ms"])
        self.assertEqual(Validity.UNINFORMATIVE.value, r.metrics["measurement_validity"])
        self.assertIn("离散度", r.message)
        self.assertFalse(r.metrics["measurement_can_count_stagnation"])

    def test_quiet_machine_stays_valid(self):
        r = self._l3(candidate_ms=1.040, baseline_ms=1.10, dispersion=0.02)
        self.assertEqual(Validity.VALID.value, r.metrics["measurement_validity"])
        self.assertTrue(r.metrics["measurement_can_disable_recipe"])

    def test_runner_without_dispersion_is_not_penalised(self):
        r = self._l3(candidate_ms=1.040, baseline_ms=1.10)
        self.assertEqual(Validity.VALID.value, r.metrics["measurement_validity"])
        self.assertNotIn("dispersion_iqr_over_median", r.metrics)

    def test_window_intruder_reaches_metrics_and_message(self):
        """util 快照干净、iqr 干净, 只有 max/median 报警 —— 仍要写进 message。

        message 是 Fixer / SearchAgent 的 prompt 素材, 是唯一保证被下游看到的通道
        (缺口 1 原来的失效方式就是「算了没人读」)。但注意它**不改变档位**: 这个
        统计量被顺序统计量效应污染, 详见 TestWindowIntruderSignal。
        """
        r = self._l3(candidate_ms=1.040, baseline_ms=1.10, util=0,
                     dispersion=0.0, max_over_median=18.0)
        self.assertEqual(18.0, r.metrics["dispersion_max_over_median"])
        self.assertTrue(r.metrics["measurement_intruder_suspected"])
        self.assertEqual(Validity.VALID.value, r.metrics["measurement_validity"])
        self.assertTrue(r.metrics["measurement_can_disable_recipe"])
        self.assertIn("抢占", r.message)
        self.assertIn("仅诊断", r.message)


    def test_every_sample_stat_survives_the_remote_path(self):
        """runner 的 _sample_stats 吐出来的每个键, 都必须能在 L3 metrics 里找到。

        真机跑出来的第一个版本这里是 `runs=None / max_time_ms=None` —— runner 特意
        报出实际采样轮数 (快核会被自动加采样) 就是为了让下游知道这个数字有多少样本
        撑着, 结果 verifier 的搬运循环只在 bench 顶层查, 而远程路径上这些字段在
        bench["candidate"] 里。缺口 1 的失效方式「算了没人读」换了个形态重演了一次。

        这个用例直接拿 _sample_stats 的真实返回值当输入, 所以以后它加字段、verifier
        忘了搬, 就会在这里变红。
        """
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "remote_scripts" / "run_kernel.py"
        spec = importlib.util.spec_from_file_location("_rk_for_stats", path)
        rk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rk)
        stats = rk._sample_stats([1.0, 1.05, 1.1, 1.2, 50.0])

        # metrics 键名应与统计量键名一致
        bench = {"success": True, "device": "cuda:0", "gflops": 1000.0,
                 "memory_bandwidth_gbs": 500.0,
                 "candidate": dict(stats, runs_requested=30),
                 "baselines": {"pytorch_eager": {"median_time_ms": 1.0,
                                                 "min_time_ms": 1.0}},
                 "baseline_selected": "pytorch_eager"}
        backend = fake_backend(benchmark_with_baselines=lambda *a, **kw: bench,
                               get_hardware_info=lambda: dict(H20_HW_INFO))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = Verifier(backend)._verify_performance(
                PYTHON_MATMUL, _matmul_inputs(), 0.0, _task())

        missing = [k for k in stats if k not in r.metrics]
        self.assertEqual([], missing,
                         f"runner 报了但没搬进 metrics 的字段: {missing}")
        self.assertEqual(stats["runs"], r.metrics["runs"],
                         "runs 必须是实际轮数, 不是请求轮数")
        self.assertEqual(30, r.metrics["runs_requested"])
        # metrics["min_time_ms"] 从建立起装的就是 median, 是历史包袱, 不许被真 min
        # 覆盖; 要拿真正的 median 请读 median_time_ms。
        self.assertEqual(stats["median_time_ms"], r.metrics["min_time_ms"])
        self.assertEqual(stats["median_time_ms"], r.metrics["median_time_ms"])
        self.assertNotEqual(stats["min_time_ms"], r.metrics["min_time_ms"])

    def test_raw_util_samples_survive_to_metrics(self):
        """两次原始读数必须活着到 metrics —— doc 34 的全部结论都靠它

        runner 只报 min 的话, 「min 排掉的是自己的残留还是邻居的真占用」在生产里
        永远查不了; 这次能量出「残留 <0.3 s 就衰减完、而生产 L3 采样根本撞不上那个
        窗口」(docs/research/34), 就是因为 first/second 被搬了出来。
        """
        bench = {"success": True, "device": "cuda:0", "gflops": 1000.0,
                 "memory_bandwidth_gbs": 500.0,
                 "candidate": {"median_time_ms": 1.0, "min_time_ms": 1.0},
                 "baselines": {"pytorch_eager": {"median_time_ms": 1.0,
                                                 "min_time_ms": 1.0}},
                 "baseline_selected": "pytorch_eager",
                 "gpu_util_pct": 0, "gpu_util_samples": [34, 0], "gpu_other_procs": 7}
        backend = fake_backend(benchmark_with_baselines=lambda *a, **kw: bench,
                               get_hardware_info=lambda: dict(H20_HW_INFO))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = Verifier(backend)._verify_performance(
                PYTHON_MATMUL, _matmul_inputs(), 0.0, _task())
        self.assertEqual([34, 0], r.metrics["gpu_util_samples"])
        # min 才是判污染用的那个数, first 不许越权
        self.assertEqual(0, r.metrics["gpu_util_pct"])
        self.assertFalse(r.metrics.get("benchmark_contaminated"))


class TestFourStateVerdict(unittest.TestCase):
    """D4: 打平与退步必须是两态 (docs/research/20 第三节 D4、34 第四节)"""

    def test_band_is_defined_for_every_validity_tier(self):
        """新增档位却没有带宽 → 这里红, 而不是运行时 KeyError"""
        self.assertEqual({v.value for v in Validity},
                         {k.value for k in NOISE_BAND})

    def test_bands_are_ordered_by_tier_and_anchored_on_measurement(self):
        """带宽随档位变宽, 且宽度来自 H20 实测的跨次散布 (doc 34 第四节)"""
        self.assertLess(NOISE_BAND[Validity.VALID], NOISE_BAND[Validity.DEGRADED])
        self.assertLess(NOISE_BAND[Validity.DEGRADED], NOISE_BAND[Validity.UNINFORMATIVE])
        self.assertAlmostEqual(0.001, NOISE_BAND[Validity.VALID], places=6)
        self.assertAlmostEqual(0.004, NOISE_BAND[Validity.DEGRADED], places=6)

    def test_gain_loss_and_tie_split_into_three_states(self):
        cases = [(10.05, Verdict.FAST), (10.0, Verdict.SAME), (10.0005, Verdict.SAME),
                 (9.95, Verdict.SLOW)]
        for cur, want in cases:
            with self.subTest(current=cur):
                v, why = verdict(cur, 10.0, Validity.VALID)
                self.assertIs(want, v, why)

    def test_same_delta_is_same_when_degraded_but_fast_when_valid(self):
        """DEGRADED 的带宽更宽 (0.4% vs 0.1%): 同一个 +0.2% 在两档下判决不同"""
        self.assertIs(Verdict.SAME, verdict(10.02, 10.0, Validity.DEGRADED)[0])
        self.assertIs(Verdict.FAST, verdict(10.02, 10.0, Validity.VALID)[0])

    def test_uninformative_and_missing_values_are_ambiguous(self):
        self.assertIs(Verdict.AMBIGUOUS, verdict(20.0, 10.0, Validity.UNINFORMATIVE)[0])
        for cur, prev in ((None, 10.0), (10.0, None), (0.0, 10.0), (10.0, 0.0)):
            with self.subTest(cur=cur, prev=prev):
                self.assertIs(Verdict.AMBIGUOUS, verdict(cur, prev, Validity.VALID)[0])

    def test_binding_attainment_takes_the_tighter_axis(self):
        """取 max 而不是按 bound 分支选一个: 分支在两条轴接近时会选错"""
        self.assertAlmostEqual(93.0, binding_attainment(
            {"gflops_utilization_pct": 12.0, "bandwidth_utilization_pct": 93.0}))
        self.assertIsNone(binding_attainment({}))
        self.assertIsNone(binding_attainment({"gflops_utilization_pct": None}))

    def test_roofline_converged_needs_a_trustworthy_measurement(self):
        hot = {"gflops_utilization_pct": 101.0, "bandwidth_utilization_pct": 5.6}
        self.assertTrue(roofline_converged(dict(hot)))
        self.assertFalse(roofline_converged(dict(hot, benchmark_contaminated=True)),
                        "不可信的测量没说得出「已经到顶」")
        self.assertFalse(roofline_converged(dict(hot, measurement_can_gate=False)))
        self.assertFalse(roofline_converged({"gflops_utilization_pct": 40.0}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
