"""Conductor L3 停滞判定的回归测试

钉住的是 `best_gflops` 的记账 bug: 初值 0.0, 而修复前唯一的赋值点被
`best_gflops > 0` 守卫 —— 自锁死锁, 从 0.0 出发永远变不成非零。后果是

  1. 首轮明明拿到了可用的 gflops, 却掉进"性能指标不完整"分支, 被报成缺失;
  2. 该分支无条件禁用本轮配方 (首轮 plan 是在 roofline=None 下按算子类型选的,
     无从判断好坏), 拿噪声做永久决策;
  3. `max_stagnant_rounds=2` 下, 一次正在起效的优化在第 2 轮就被提前终止。

所以这里不只测"首轮建立基线", 还要测"改善趋势能跑满预算"与"真停滞仍能提前
终止"两侧 —— 只测前者会把 bug 修成"干脆不做停滞判定"。

gflops 用脚本化后端注入 (MockGPU.benchmark 是真计时, 逐轮改善无法稳定复现);
kernel 代码保持真执行, L3 因 min_speedup=5.0 每轮都 FAILED, 确保进入停滞分支。

D4 之后这里还覆盖四态判决的后果 (`TestRealStagnationStillTerminates`) 与 roofline
收敛出口 (`TestRooflineConvergedTerminatesSearch`) —— 后者是变异测试逼出来的:
判据本身有单测, 但「循环里到底有没有读它」没有任何断言, 把出口整个删掉照样全绿。

运行: python3 -m unittest tests.test_conductor_stagnation -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent import hardware_caps as hc
from operator_agent.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity,
    TaskSchema, TaskType,
)


SLOW_MATMUL = """import numpy as np

def kernel(A, B):
    for _ in range(20):
        C = A @ B
    return C
"""

WRONG_MATMUL = """import numpy as np

def kernel(A, B):
    return (A @ B) * 3
"""

SKETCH = {
    "algorithm": "matmul",
    "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
    "memory_plan": {}, "pipeline": [], "optimization_hints": [],
}


class ScriptedLLM:
    def __init__(self, code=SLOW_MATMUL):
        self._code = code

    def chat(self, messages, **kw):
        return self._code

    def chat_json(self, messages, **kw):
        return dict(SKETCH)


class ScriptedGflopsGPU(MockGPU):
    """按轮次注入 gflops (可选再注入带宽), 其余行为 (compile/run/计时) 全部沿用 MockGPU

    注入点是 `execute()` 而不是 `benchmark()`: 非远程后端下 Verifier 的 L3 从
    `backend.execute(...).gflops` 与 `.memory_bandwidth_gbs` 取值 (verifier.py:444-446),
    `benchmark()` 返回的 gflops 只用于远程分支。
    每轮的调用序是 L2 execute → L3 benchmark → L3 execute, 所以用 benchmark
    的调用次数当轮次指针: 只有 L3 那次 execute 会读到本轮的脚本值 (L2 不消费
    gflops, 读到上一轮的无妨)。
    """

    def __init__(self, gflops_seq, bandwidth_seq=None, dispersion=None):
        super().__init__()
        self._seq = list(gflops_seq)
        self._bw_seq = None if bandwidth_seq is None else list(bandwidth_seq)
        self._dispersion = dispersion
        self._round = 0
        self.observed = []

    def benchmark(self, kernel_code, inputs, warmup=3, repeat=10):
        result = super().benchmark(kernel_code, inputs, warmup, repeat)
        if self._dispersion is not None:
            # MockGPU 不报离散度, 而 Verifier 会把它搬进 metrics 并据此定档
            result["dispersion_iqr_over_median"] = self._dispersion
        self._round = len(self.observed)
        self.observed.append(self._gflops_for_round())
        return result

    def execute(self, kernel_code, inputs):
        result = super().execute(kernel_code, inputs)
        result.gflops = self._gflops_for_round()
        if self._bw_seq is not None:
            result.memory_bandwidth_gbs = self._seq_at(self._bw_seq)
        return result

    def _gflops_for_round(self):
        return self._seq_at(self._seq)

    def _seq_at(self, seq):
        return seq[min(self._round, len(seq) - 1)]


class StagnationCase(unittest.TestCase):

    maxDiff = 600   # Conductor 的 stdout 很长, 失败时不要刷屏

    def _run(self, gflops_seq, max_rounds, bandwidth_seq=None, peak=None,
             dispersion=None, code=None):
        backend = ScriptedGflopsGPU(gflops_seq, bandwidth_seq, dispersion)
        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(framework="pytorch", kernel_type="matmul",
                              fused_ops=[], complexity=TaskComplexity.MEDIUM),
            target=TargetSpec(hardware="mock_gpu", dsl="python",
                              compute_units=64, warp_size=32),
            constraints=ConstraintSpec(precision=Precision.FP32,
                                       min_speedup=5.0, max_error=1e-5),
        )
        rng = np.random.default_rng(0)
        inputs = {"A": rng.standard_normal((64, 64)).astype(np.float32),
                  "B": rng.standard_normal((64, 64)).astype(np.float32)}

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            conductor = Conductor(backend=backend, kernel_db=KernelDatabase(),
                                  llm=ScriptedLLM(SLOW_MATMUL if code is None else code),
                                  max_optimize_rounds=max_rounds)
            if peak is not None:
                # `_theoretical_peak` 是 Verifier 的延迟探测缓存 (verifier.py:147),
                # 预置它等于对外声明「这台设备的理论峰值是 X」, 不必碰探测逻辑。
                # MockGPU 探测不到峰值 (hw_peak_gflops=0 → 利用率恒为 0), 所以
                # roofline 相关的用例只能从这里注入。
                conductor.verifier._theoretical_peak = dict(peak)
                # ⚠️ §13.5: 「实测过」那一半的**真源是存放处**, 缓存里只有表那一半是
                # 静态的 (verifier 每次调用都从存放处重读实测值)。所以钉「收敛出口开了」
                # 必须走生产路径把值放进存放处 —— 手改 dict 里那两个字段会在下一跳 refresh
                # 时被覆盖 (第一版就是这么写的, 用例红在「出口没开」, 而机制其实是对的)。
                hc.forget_measured_peaks()
                self.addCleanup(hc.forget_measured_peaks)
                key = hc.device_key_of({"name": peak.get("name", ""),
                                        "compute_capability": peak.get("compute_capability", "")})
                for prec, value in (peak.get("measured_peak_tflops") or {}).items():
                    hc.remember_measured_peak(key, prec, value, "用例注入的实测值")
            artifact = conductor.execute(task, inputs, target_gflops=0.0)
        return artifact, buf.getvalue(), backend

    def _rounds_run(self, out, max_rounds):
        return sum(1 for r in range(1, max_rounds + 1) if f"轮次 {r}/{max_rounds}" in out)


class TestFirstObservationEstablishesBaseline(StagnationCase):
    """首个有效观测必须被认作基线, 而不是被报成「指标不完整」"""

    def test_round_one_reports_baseline_not_incomplete(self):
        _, out, _ = self._run([10.0, 20.0, 40.0], max_rounds=3)
        self.assertIn("建立性能基线", out)
        self.assertNotIn("性能指标不完整", out,
                         "gflops 可用时不该走「指标不完整」分支 (修复前的死锁症状)")

    def test_round_one_does_not_disable_recipes(self):
        """首轮无「之前」可比, 禁用配方等于拿噪声做永久决策

        断言的是后果而不是日志: 修复前首轮走 else 分支, 那里 `disabled_recipes.add()`
        是静默的 (没有 print), 只看 stdout 会漏判。配方被禁用的可观测后果是后续
        轮次 Planner 无方可选。
        """
        _, out, _ = self._run([10.0, 20.0, 40.0], max_rounds=3)
        self.assertNotIn("本轮无新增优化配方", out)
        self.assertEqual(out.count("[Planner] 选中"), 3, "三轮都该拿到配方")

    def test_gflops_reaches_artifact(self):
        """best_gflops 能离开 0.0, 才有资格写进产物/知识库"""
        artifact, _, backend = self._run([10.0, 20.0, 40.0], max_rounds=3)
        self.assertEqual(backend.observed, [10.0, 20.0, 40.0])
        self.assertGreater(artifact.benchmark_data.get("gflops", 0), 0)


class TestImprovingTrendRunsFullBudget(StagnationCase):
    """正在起效的优化不得被提前终止 —— 这是死锁最直接的损失"""

    def test_all_rounds_run_when_gflops_improves(self):
        _, out, _ = self._run([10.0, 20.0, 40.0], max_rounds=3)
        self.assertEqual(self._rounds_run(out, 3), 3)
        self.assertNotIn("提前终止", out)

    def test_improvement_resets_stagnation_counter(self):
        _, out, _ = self._run([10.0, 20.0, 40.0], max_rounds=3)
        self.assertNotIn("性能未改善", out)


class TestRealStagnationStillTerminates(StagnationCase):
    """反向保护: 修死锁不能顺手把停滞判定废掉

    ⚠️ §10.4 的第二半（CHECKLOG I64）改了这里的**语义**：停滞 = 「K 轮 valid 无改善」**且**
    「有希望的区域已穷尽」，只满足前半句**不许**停。MockGPU 是代码型路径 —— 候选由 LLM/规则
    按需产生，覆盖**无法确立** ⇒ 停滞条件不成立（照旧跑满轮次），而这件事必须留痕。
    「真停滞仍能终止」那条保护改到**可判**的那条路径上（`TestExhaustedRegionStillTerminates`，
    模板型：候选全集可枚举）。这一改不是放宽：旧的断言把「只因轮数就宣布停滞」当成了正确行为，
    而设计点名那种读法是错的。
    """

    def test_flat_gflops_does_not_declare_stagnation_when_coverage_is_unknown(self):
        _, out, _ = self._run([10.0, 10.0, 10.0, 10.0], max_rounds=4)
        self.assertIn("性能未改善", out)
        self.assertIn("覆盖**无法确立**", out, "无法确立必须写明, 不许静默继续")
        self.assertNotIn("提前终止", out,
                         "代码型路径的候选不可枚举 ⇒ 「已穷尽」无法确立 ⇒ 不许只因轮数停")
        self.assertEqual(self._rounds_run(out, 4), 4, "轮次上限才是收口的那一条")

    def test_the_declined_stop_is_recorded_in_the_artifact(self):
        """「没停」与「判据没人读」在产物里必须分得开"""
        artifact, out, _ = self._run([10.0, 10.0, 10.0, 10.0], max_rounds=4)
        cov = artifact.metadata["coverage"]
        self.assertEqual("unknown", cov["last"]["state"])
        self.assertEqual(True, cov["last"]["exhausted"] is None,
                         "无法确立的三态值必须是 None (不是 False)")
        self.assertGreaterEqual(cov["declined"].get("unknown", 0), 2,
                                "达到阈值却没停的轮次要被记下来")
        rounds = [h["coverage"] for h in artifact.iteration_history if h["coverage"]]
        self.assertTrue(rounds, "逐轮的覆盖判定要进血缘")

    def test_flat_gflops_does_not_disable_recipes(self):
        """打平不再是「禁用配方」的证据 (D4 四态判决)

        原来这里断言的是 tie → 禁用: 判据是一个无带宽的 `current > best`, 于是
        「完全没变」与「真的更慢」被压成同一件事, 而禁用是**永久**决策。四态之后
        平局落 SAME —— 累计停滞 (上面那条用例仍在守), 但不拉黑配方。
        """
        _, out, _ = self._run([10.0, 10.0, 10.0, 10.0], max_rounds=4)
        after_first = out.split("轮次 2/4", 1)[-1]
        self.assertNotIn("禁用配方", after_first)
        self.assertIn("性能未改善", after_first)

    def test_regression_beyond_band_disables_recipes(self):
        """SLOW 仍然是可以禁用配方那一态 —— 修语义不能把停止恶化能力一起废掉"""
        _, out, _ = self._run([20.0, 10.0, 10.0, 10.0], max_rounds=4)
        self.assertIn("性能退步", out)
        self.assertIn("禁用配方", out)

    def test_within_band_gain_is_not_an_improvement(self):
        """带宽内的小涨不改 best、不重置停滞计数器

        这条是四态判决的代价, 写明白: 每步都小于噪声的缓慢爬升会被判停滞。真机上
        单轮的可测下限是 0.1% (doc 34 第四节), 这样的爬升本来也不该被当成证据。
        """
        _, out, _ = self._run([10.0, 10.002, 10.0], max_rounds=3)
        self.assertNotIn("[Conductor] 性能改善 (", out)
        self.assertIn("落在带宽", out)


class TestMissingGflopsFirstRound(StagnationCase):
    """gflops 不可用 (=0) 时首轮同样不该禁用配方"""

    def test_zero_gflops_round_one_keeps_recipes(self):
        _, out, _ = self._run([0.0, 0.0, 0.0], max_rounds=3)
        first_round = out.split("轮次 2/3")[0]
        self.assertIn("性能指标不完整", first_round)
        # 同上: else 分支的禁用是静默的, 只能看 Planner 后续还有没有方可选
        self.assertNotIn("本轮无新增优化配方", out)
        self.assertGreaterEqual(out.count("[Planner] 选中"), 2)


class TestRooflineConvergedTerminatesSearch(StagnationCase):
    """缺口 4 的正解要落在循环的调用点上

    `roofline_converged()` 在 `test_measurement_validity` 里有单测, 但变异 P8
    (把 conductor 里这段出口整个删掉) 逃逸了 —— 那就是缺口 1 记过的那类失效:
    判据算出来了, 下游没人读。所以这里断言的是**轮次确实停在这里**。

    四个方向都要钉住: 两条轴各自能触发 (取 max 而不是按 `bound_type` 分支选指标,
    才能在名字说 compute-bound、而带宽先贴上界时认出收敛)、不够高时不能停、
    测量不可信/没有 L3 时不能停。
    """

    # 峰值故意设得很小, 让脚本注入的 gflops / 带宽能落在利用率尺度的两端:
    # fp32 峰值 20 GFLOPS, 带宽峰值 1000 GB/s。
    # ⚠️ 这个 fixture 必须跟着真 producer (`Verifier._detect_theoretical_peak`) 的返回
    # 形状走: 它现在多带了 peak_source/peak_trusted/peak_table_key。少了 `peak_trusted`
    # 时「分母口径未核 → 不开出口」这道门会默认关掉, 于是本类的用例全都不停 —— 看着像
    # 「机制生效」, 实际是 fixture 落后了一格。
    # 这一份是**验尸台上**的分母: 两个位都真 (口径核过 + 实测过), 才允许收敛出口点火。
    # `peak_measured=True` 是刻意的 —— 今天没有 `probe_peak`, 真机发布出去的是 False,
    # 于是出口在所有真机上都是关的 (见下面 test_unverified_peak_* 那一族)。这里造的是
    # 「探测已经落地之后」的形状, 用来钉出口本身的机制。
    PEAK = {"fp32_tflops": 0.02, "tf32_tflops": 0.0,
            "fp16_tflops": 0.0, "bf16_tflops": 0.0,
            "memory_bandwidth_gbs": 1000.0,
            "launch_floor_us": None, "compute_capability": "",
            "device": "mock_gpu", "name": "MockGPU",
            "peak_table_key": "mock", "peak_source": "mock (口径已核)",
            "peak_trusted": True, "peak_measured": True,
            # §13.5 起实测峰值**按精度**发布 (`{"fp32": 0.02}`)。收敛出口读的是
            # `_roofline_analysis` 里按这一轮分母的精度键取出来的那一位 —— fixture 少了
            # 这一格就会「看着像机制生效, 其实是 fixture 落后了一格」(见上面那段告诫)。
            "measured_peak_tflops": {"fp32": 0.02}}

    def test_compute_axis_converged_stops_early(self):
        """50% → 95% GFLOPS 利用率: 第 2 轮已达上界, 不该再跑第 3、4 轮"""
        _, out, _ = self._run([10.0, 19.0, 40.0, 80.0], max_rounds=4, peak=self.PEAK)
        self.assertIn("判定已收敛", out)
        self.assertEqual(self._rounds_run(out, 4), 2)
        self.assertIn("95.0%", out, "报出来的利用率要能对上注入值/峰值")
        self.assertIn("[Conductor]   分母:", out,
                      "判停那一行要把它依赖的分母出处一起打出来")
        self.assertIn("口径已核", out)
        self.assertNotIn("提前终止", out, "停因是收敛而不是停滞 —— 两者是不同的判据")

    def test_unverified_peak_denominator_does_not_open_the_exit(self):
        """分母口径未核 (peak_trusted=False) 时, 利用率再高也不能宣布「已到上界」

        理由是不对称的代价: 出口不开只是多烧几轮 (停滞出口会兜底), 而分母偏小 (这张
        表里 5 个 ⚠️ 行的口径就没核过, 混过 2:4 sparse 就是 2 倍) 会把还能优化的核
        提前判成到顶。所以「这个数字凭什么」不能只写在源码注释里。
        """
        peak = dict(self.PEAK, peak_trusted=False,
                    peak_source="⚠️口径未核 (疑含 2:4 sparse)")
        _, out, _ = self._run([10.0, 19.0, 40.0, 80.0], max_rounds=4, peak=peak)
        self.assertNotIn("判定已收敛", out)
        self.assertEqual(self._rounds_run(out, 4), 4)
        self.assertNotIn("[Conductor]   分母", out,
                         "没判停就不该打印「判停所依赖的分母」那一行 "
                         "(Verifier 每轮的 Roofline 行里也带分母, 别拿它当判停证据)")

    def test_unmeasured_peak_does_not_open_the_exit_either(self):
        """分母**口径核过但没实测** ⇒ 同样不能宣布「已到上界」(设计 §13.5)

        与上面那条成对: 「口径核过」(dense/sparse 没混) 与「实测过」(probe_peak 打过一次)
        是两个问题, 而收敛要说的是「这张卡这一刻贴着上界了」。真机实测抓到的混同是:
        H20 那行的出处自己写着「无公开官方页 + 算术推导」, 却因为不带 ⚠️ 把出口开着。
        """
        peak = dict(self.PEAK, peak_measured=False, measured_peak_tflops={})
        _, out, _ = self._run([10.0, 19.0, 40.0, 80.0], max_rounds=4, peak=peak)
        self.assertNotIn("判定已收敛", out)
        self.assertEqual(self._rounds_run(out, 4), 4)

    def test_memory_axis_converged_stops_early(self):
        """matmul 的名字是 compute-bound, 但先贴上界的是带宽轴 —— 仍该停"""
        _, out, _ = self._run([1.0, 1.0, 1.0, 1.0], max_rounds=4,
                              bandwidth_seq=[100.0, 950.0, 950.0, 950.0],
                              peak=self.PEAK)
        self.assertIn("判定已收敛", out)
        self.assertEqual(self._rounds_run(out, 4), 2)
        self.assertIn("95.0%", out)

    def test_below_threshold_runs_full_budget(self):
        """反向保护: 利用率不够就不能停, 否则把「收敛」写成了「恒定终止」"""
        _, out, _ = self._run([10.0, 15.0, 16.0, 16.5], max_rounds=4, peak=self.PEAK)
        self.assertNotIn("判定已收敛", out)
        self.assertEqual(self._rounds_run(out, 4), 4)
        # 每次运行都要报一句语料现状 (缺口 2: 「RAG 命中 N 条」以前不说明那 N 条从哪来)
        self.assertIn("[KernelDB]", out)

    def test_no_l3_result_is_never_called_converged(self):
        """L2 挂掉的轮次压根没有 L3 结果: 既不能崩, 更不能宣布「已到上界」

        数值不过关时性能数字毫无意义, 而这里拿的是 `l3_result is not None` 这一道
        守卫 —— 去掉它, 收敛判定会在 None 上取 .metrics (变异 P9)。
        """
        _, out, _ = self._run([19.0, 19.0, 19.0, 19.0], max_rounds=4,
                              peak=self.PEAK, code=WRONG_MATMUL)
        self.assertIn("[Verifier] L2 FAILED", out, "前提: 数值门禁确实没过")
        self.assertNotIn("[Verifier] L3", out, "前提: 没有 L3 结果")
        self.assertNotIn("判定已收敛", out)
        self.assertEqual(self._rounds_run(out, 4), 4)

    def test_uninformative_measurement_is_not_called_converged(self):
        """档位 uninformative 的数字不能说「已经到顶」: 那是拿不可信的测量结束搜索

        这一条走的是 `roofline_converged` 里的 can_gate 守卫。它必须单独测: 停滞
        那条出口被 `measurement_ok` 挡住时就不再执行, 而收敛出口是独立的 if ——
        少一道守卫就会在脏数据上直接终止整个搜索。
        """
        _, out, _ = self._run([19.0, 19.0, 19.0, 19.0], max_rounds=4,
                              peak=self.PEAK, dispersion=0.5)
        self.assertNotIn("判定已收敛", out)
        self.assertEqual(self._rounds_run(out, 4), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
