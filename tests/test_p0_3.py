"""P0-3 相对 baseline 性能评估的单元测试

覆盖:
- L3 首选加速比判定: speedup_vs_baseline >= min_speedup (替代绝对 GFLOPS)
- 慢 kernel 判失败 / 同实现判通过 / min_speedup 语义
- bound 类型分类 (compute/memory/unknown)
- baseline 测量缓存
- 无参考实现算子回退 legacy 绝对目标 (achieve_ratio)
- Search prompt 加速比导向

运行: python3 tests/test_p0_3.py  或  python3 -m unittest tests.test_p0_3 -v
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent.agents import SearchAgent
from operator_agent.verifier import Verifier, baseline_plan
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec, UnifiedSketch,
    TaskType, TaskComplexity, Precision, VerificationLevel,
)
from tests.backend_fakes import fake_backend


PYTHON_MATMUL = """import numpy as np

def kernel(A, B):
    return A @ B
"""

# 结果正确但做 50 倍冗余计算的慢 kernel
SLOW_MATMUL = """import numpy as np

def kernel(A, B):
    for _ in range(50):
        C = A @ B
    return C
"""

PYTHON_SOFTMAX = """import numpy as np

def kernel(X):
    mx = np.max(X, axis=-1, keepdims=True)
    ex = np.exp(X - mx)
    return ex / np.sum(ex, axis=-1, keepdims=True)
"""

IDENTITY_KERNEL = """import numpy as np

def kernel(input):
    return input * 2
"""


def _make_task(kernel="matmul", min_speedup=0.5) -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel, complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python",
                          compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32,
                                   min_speedup=min_speedup, max_error=1e-5),
    )


def _matmul_inputs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"A": (rng.standard_normal((16, 8)) * 0.1).astype(np.float32),
            "B": (rng.standard_normal((8, 16)) * 0.1).astype(np.float32)}


def _l3(verifier, code, inputs, task, target_gflops=0.0):
    return verifier._verify_performance(code, inputs, target_gflops, task)


# ─── 加速比判定 ──────────────────────────────────────────────

class TestSpeedupVerdict(unittest.TestCase):

    def setUp(self):
        self.verifier = Verifier(MockGPU())

    def test_equal_impl_passes_with_speedup_metrics(self):
        r = _l3(self.verifier, PYTHON_MATMUL, _matmul_inputs(), _make_task())
        # 不断言 `r.passed`: 同实现的真实比值本来就随负载在 0.3–3.0 之间漂
        # (`test_speedup_sanity_for_equal_impl` 守这个区间), 而邻居进程一挤, 判据就不该
        # 只凭「通过了」变红 —— 那是机器的状态, 不是代码的回归 (doc 49 §7.7 / #50)。
        # 「判定读的是这个比值」由下面 `TestSpeedupGateIsAFunctionOfTheRatio` 用固定计时确定性地测;
        # 这里只要求**通过了就不许自相矛盾**。
        if r.passed:
            self.assertGreaterEqual(
                r.metrics["speedup_vs_baseline"], r.metrics["min_speedup_required"],
                f"通过了, 但它自己报出来的比值不支持这个结论: {r.message}")
        self.assertIn("speedup_vs_baseline", r.metrics)
        self.assertIn("baseline_time_ms", r.metrics)
        self.assertIn(r.metrics["baseline_type"], ("pytorch_eager", "numpy_reference"))
        self.assertEqual(r.metrics["bound_type"], "compute")
        self.assertEqual(r.metrics["primary_metric"], "speedup_vs_baseline")
        self.assertIn("加速比", r.message)

    def test_slow_kernel_fails(self):
        r = _l3(self.verifier, SLOW_MATMUL, _matmul_inputs(),
                _make_task(min_speedup=1.0))
        self.assertFalse(r.passed)
        self.assertLess(r.metrics["speedup_vs_baseline"], 1.0)

    def test_min_speedup_semantics(self):
        """min_speedup 即'相对 baseline 的最低加速比'，阈值放低则通过"""
        r = _l3(self.verifier, SLOW_MATMUL, _matmul_inputs(),
                _make_task(min_speedup=0.001))
        self.assertTrue(r.passed, r.message)

    def test_speedup_sanity_for_equal_impl(self):
        """同一实现 kernel vs baseline 加速比应在 1.0 附近的合理区间"""
        r = _l3(self.verifier, PYTHON_MATMUL, _matmul_inputs(), _make_task())
        speedup = r.metrics["speedup_vs_baseline"]
        self.assertGreater(speedup, 0.3)
        self.assertLess(speedup, 3.0)

    def test_no_misleading_speedup_ratio(self):
        """旧的误导性 speedup_ratio (gflops/目标) 已移除"""
        r = _l3(self.verifier, PYTHON_MATMUL, _matmul_inputs(), _make_task())
        self.assertNotIn("speedup_ratio", r.metrics)


# ─── bound 分类 ──────────────────────────────────────────────

class TestBoundClassification(unittest.TestCase):

    def test_compute_bound(self):
        self.assertEqual(Verifier._bound_type("matmul"), "compute")
        self.assertEqual(Verifier._bound_type("flash_attention"), "compute")

    def test_memory_bound(self):
        self.assertEqual(Verifier._bound_type("softmax"), "memory")
        self.assertEqual(Verifier._bound_type("layernorm"), "memory")

    def test_unknown(self):
        self.assertEqual(Verifier._bound_type("exotic_op"), "unknown")

    def test_softmax_uses_memory_bound_metrics(self):
        """访存型算子走 memory 轴, 判定读的是加速比 —— **但不断言这台机器有多快**

        原先这里断言 `r.passed`, 而 passed 要求实测加速比 ≥ `min_speedup=0.5`:
        32×32 / 1024 元素 / warmup=2 repeat=5 是在**控制机 CPU**上量的, 邻居进程一挤就掉到
        0.40 ⇒ 红的时候分不清是这个核变慢了还是这台机器在忙 (doc 49 §7.7 记的定量结果:
        并发时约 3/24 次多出一条红, 机器安静时两边各 12 次全绿)。
        判定本身由下面那条**固定计时**的用例来断言 —— 那才是这条用例真正要看的东西。
        """
        verifier = Verifier(MockGPU())
        inputs = {"X": np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32)}
        r = _l3(verifier, PYTHON_SOFTMAX, inputs, _make_task("softmax"))
        self.assertEqual(r.metrics["bound_type"], "memory")
        self.assertIn("speedup_vs_baseline", r.metrics)   # 访存型也用加速比判定
        self.assertGreater(r.metrics["speedup_vs_baseline"], 0.0,
                           f"没量出加速比: {r.message}")
        # 阈值确实传到了判定处 (而不是被某处悄悄换成绝对目标)
        self.assertEqual(0.5, r.metrics["min_speedup_required"])


class _FixedTiming(MockGPU):
    """把「量到几毫秒」变成已知答案 —— 被测的是**判定读哪个数**, 不是这台机器的速度"""

    def __init__(self, median_ms: float):
        super().__init__()
        self._median_ms = float(median_ms)

    def benchmark(self, kernel_code, inputs, warmup=3, repeat=10):
        bench = dict(super().benchmark(kernel_code, inputs, warmup=warmup, repeat=repeat))
        for key in ("median_time_ms", "min_time_ms", "avg_time_ms"):
            bench[key] = self._median_ms
        return bench


class TestSpeedupGateIsAFunctionOfTheRatio(unittest.TestCase):
    """#50: 「墙钟加速比 ≥ 阈值」这条判据要可判定, 就得和机器当前忙不忙解耦

    候选固定 2.0 ms、基线固定 3.0 ms ⇒ 加速比是**算出来的已知量 1.50x**, 于是阈值放在
    它两侧各跑一次都必须给出确定的答案。这也是量具自身的标定: 报出来的比值必须等于
    我注入的幅度, 否则「判定读的是这个数」这句话就没有可验的内容。
    """

    CANDIDATE_MS = 2.0
    BASELINE_MS = 3.0

    def _run(self, min_speedup):
        verifier = Verifier(_FixedTiming(self.CANDIDATE_MS))
        task = _make_task("softmax", min_speedup=min_speedup)
        inputs = {"X": np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32)}
        with mock.patch.object(verifier, "_measure_baseline",
                              return_value={"baseline_type": "numpy_eager",
                                            "baseline_time_ms": self.BASELINE_MS}):
            return _l3(verifier, PYTHON_SOFTMAX, inputs, task)

    def test_the_ratio_is_the_one_that_was_injected(self):
        """先标定量具本身: 报出来的加速比必须等于注入的 3.0/2.0"""
        r = self._run(min_speedup=0.5)
        self.assertAlmostEqual(1.5, r.metrics["speedup_vs_baseline"], places=4)

    def test_passes_when_the_threshold_is_below_the_ratio(self):
        self.assertTrue(self._run(min_speedup=1.4).passed)

    def test_refuses_when_the_threshold_is_above_the_ratio(self):
        """阈值只挪一位小数, 结论就必须翻 —— 判定确实读的是这个数"""
        r = self._run(min_speedup=1.6)
        self.assertFalse(r.passed)
        self.assertIn("加速比", r.message, f"拒了但没说是加速比不够: {r.message}")
        self.assertIn("1.50x", r.message, "报出来的比值不是我注入的那个")


# ─── baseline 缓存 ───────────────────────────────────────────

class TestBaselineCache(unittest.TestCase):

    def test_baseline_cached_by_input_signature(self):
        verifier = Verifier(MockGPU())
        task = _make_task()
        inputs = _matmul_inputs()
        b1 = verifier._measure_baseline(task, inputs)
        b2 = verifier._measure_baseline(task, inputs)
        self.assertIs(b1, b2)  # 第二次命中缓存 (同一对象)

    def test_no_reference_returns_none(self):
        verifier = Verifier(MockGPU())
        task = _make_task("exotic_op")
        self.assertIsNone(verifier._measure_baseline(task, {"input": np.zeros(4)}))


class TestSameDeviceRemoteBenchmark(unittest.TestCase):

    def test_remote_uses_returned_same_device_baseline(self):
        payload = {
            "success": True, "device": "cuda:0", "gflops": 1000.0,
            "memory_bandwidth_gbs": 500.0,
            "candidate": {"min_time_ms": 1.0, "median_time_ms": 2.0,
                          "avg_time_ms": 2.1, "max_time_ms": 2.3, "runs": 30},
            # D3 之后远端只测一个基线并回报选了哪个; 这里模拟「远端按偏好
            # 试到 torch_compile 才成功」的情形 (eager 也测过但不参与判定)
            "baselines": {
                "torch_compile": {"median_time_ms": 2.5, "min_time_ms": 2.4},
            },
            "baseline_selected": "torch_compile",
        }
        backend = fake_backend(benchmark_with_baselines=lambda *args, **kwargs: payload)
        verifier = Verifier(backend)
        verifier._measure_baseline = lambda *_: self.fail("控制端 baseline 不得参与远程评测")
        r = _l3(verifier, PYTHON_MATMUL, _matmul_inputs(), _make_task(min_speedup=1.2))
        self.assertTrue(r.passed, r.message)
        self.assertEqual(r.metrics["baseline_type"], "torch_compile")
        self.assertEqual(r.metrics["measurement_scope"], "same_remote_device")
        self.assertEqual(r.metrics["speedup_vs_baseline"], 1.25)

    def test_control_side_trusts_the_remote_selection(self):
        """远端说选了谁就是谁 —— 控制端不许按偏好名单重排

        重排等于假设「所有基线都被测过」, 而 D3 之后远端只测一个。这个 payload 故意
        同时带上 tf32 与 eager 两份耗时、且 selected 指向名单里**靠后**的那个: 一旦有人
        把重排逻辑加回来, 这里会立刻选到 tf32 并给出 4.5x 的假加速比。
        """
        seen = {}

        def _bench(*a, **kw):
            seen.update(kw)
            return {
            "success": True, "device": "cuda:0", "gflops": 1000.0,
            "memory_bandwidth_gbs": 500.0,
            "candidate": {"median_time_ms": 2.0, "min_time_ms": 1.9},
            # tf32 在名单里排在 eager **前面**, 且耗时被故意写成 9.0 —— 一旦有人把
            # 本地重排加回来, 就会选中它并给出 4.5x 的假加速比
            "baselines": {
                "pytorch_eager_tf32": {"median_time_ms": 9.0, "min_time_ms": 9.0},
                "pytorch_eager": {"median_time_ms": 3.0, "min_time_ms": 2.8},
            },
            "baseline_selected": "pytorch_eager",
            "paired": {"speedup_paired_median": 1.47, "speedup_paired_iqr_over_median": 0.03,
                       "paired_runs": 240, "paired_block_samples": 8},
            }
        backend = fake_backend(benchmark_with_baselines=_bench)
        v = Verifier(backend)
        r = _l3(v, PYTHON_MATMUL, _matmul_inputs(),
                _make_task(kernel="matmul", min_speedup=1.2))
        self.assertTrue(r.passed, r.message)
        self.assertEqual("pytorch_eager", r.metrics["baseline_type"])
        self.assertEqual(1.5, r.metrics["speedup_vs_baseline"])
        # 交错配对的旁证必须能被下游看到, 否则「测了但没人读」又回来一次
        self.assertEqual(1.47, r.metrics["paired_speedup_median"])
        self.assertEqual(0.03, r.metrics["paired_speedup_dispersion"])
        self.assertEqual(240, r.metrics["paired_samples"])
        self.assertEqual(8, r.metrics["paired_block_samples"])
        # 策略必须真的被**下发**: 只在控制端排好但不发过去, 远端就退回自己的书写顺序
        self.assertEqual(baseline_plan("matmul", "fp32"), seen["baseline_priority"])


# ─── legacy 回退 ─────────────────────────────────────────────

class TestLegacyFallback(unittest.TestCase):

    def setUp(self):
        self.verifier = Verifier(MockGPU())
        self.inputs = {"input": np.ones((8, 8), dtype=np.float32)}

    def test_no_baseline_zero_target_passes(self):
        r = _l3(self.verifier, IDENTITY_KERNEL, self.inputs,
                _make_task("exotic_op"), target_gflops=0.0)
        self.assertTrue(r.passed, r.message)
        self.assertIn("achieve_ratio", r.metrics)
        self.assertIn("无 baseline", r.message)

    def test_no_baseline_absolute_target_enforced(self):
        r = _l3(self.verifier, IDENTITY_KERNEL, self.inputs,
                _make_task("exotic_op"), target_gflops=100.0)
        self.assertFalse(r.passed)  # GFLOPS=0 (非 A/B 输入) 达不到 100 GFLOPS


# ─── Search prompt 加速比导向 ────────────────────────────────

class MockLLMJson:
    def __init__(self):
        self.calls = []

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        self.calls.append(messages)
        return {"suggestion": "增大 tile", "new_tiling": {"block_m": 128},
                "new_hints": ["vectorized load"]}


class TestSearchPrompt(unittest.TestCase):

    def test_search_prompt_speedup_oriented(self):
        llm = MockLLMJson()
        search = SearchAgent(llm=llm)
        sketch = UnifiedSketch(algorithm="matmul", version=1, target_dsl="triton")
        _, suggestion = search.run(sketch, {
            "gflops": 12.3, "speedup": 0.62, "target_speedup": 1.0,
            "bound_type": "compute", "avg_time_ms": 0.05,
            "memory_bandwidth_gbs": 10.0,
        })
        prompt = llm.calls[0][-1]["content"]
        self.assertIn("0.62x", prompt)          # 当前加速比
        self.assertIn("1.00x", prompt)          # 要求加速比
        self.assertIn("compute", prompt)        # bound 类型
        self.assertIn("加速比", prompt)
        self.assertNotIn("目标 GFLOPS", prompt)  # 旧绝对目标表述已移除


if __name__ == "__main__":
    unittest.main(verbosity=2)
