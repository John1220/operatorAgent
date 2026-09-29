"""分母的原料: 每轮 roofline 的轴与输入形状必须穿过落盘边界 (缺口 3 的「先补分母」)

为什么单独有这一组: doc 42 量出「可达上界是按 shape 索引的一条曲线」, doc 49 §八 第 14 条
把顺序钉死成「先补分母再动排序」。而分母要能被离线复盘, 前提是血缘里有
`(shape, 算术强度) -> 本轮达到的速率` 这些点。Verifier 每轮其实都算过算术强度、脊点、
两侧达成率、峰值出处与 `peak_trusted` —— 它们只是从没离开过内存: `result.json` 里既没有
`benchmark_data`, 也没有 `verification_results` 的 metrics。

三条设计纪律各自对应一个已经犯过的错:

* **不挑键**: 整块复制 `roofline_analysis`。doc 48 那次给落盘键做了白名单 (只挑 7 个),
  于是 Conductor 算好的血缘键在落盘前被丢掉, 那一批真机数据的重复基率永久没了、只能重跑。
  白名单的问题不是漏了哪几个键, 而是「每加一个键都要记得来这里加一次, 而忘记的代价是静默的」。
  ⇒ 下面有一条用例专门喂一个 `roofline_analysis` 里**本文件从没听说过的键**进去。
* **缺键 ≠ 空块**: 这一轮没走到 L3 就不写 `roofline` 键。写成空块等于把「没测过」说成
  「测了, 值是空」(doc 47 / doc 49 §7.9 同一条纪律)。
* **绝对锚点**: `算术强度` 的两个数不是从代码里读出来的, 是**外部独立算出来**的 ——
  matmul 1024×1024×4096 声明 FP16 时 AI = 1365.3333 (doc 48 §6.5 那次 2 倍口径错误的
  正确侧), layernorm 1024×512 FP32 时 AI = 0.875 (doc 50 §7.2 真机产物里读回来的)。
  只配差分守卫是不够的: 两侧可以一起漂 (trap #13)。

运行: PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest tests/test_lineage_axes.py -q
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from operator_agent.conductor import Conductor, _round_axes
from operator_agent.kernel_db import KernelDatabase
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType,
                                    VerificationLevel, VerificationResult)
from operator_agent.verifier import Verifier

AI_OF = Verifier._compute_arithmetic_intensity


def _task(kernel_type, precision, hardware="mock_gpu", dsl="python"):
    return TaskSchema(
        task_id="axes-case",
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware=hardware, dsl=dsl),
        constraints=ConstraintSpec(precision=precision, min_speedup=5.0, max_error=1e-3),
    )


def _l3(metrics):
    return VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                              passed=bool(metrics), metrics=metrics)


class AxesAreCarriedWholesale(unittest.TestCase):
    """整块搬运 + 三态: 没测过就不写这个键。"""

    def test_whole_block_flows_including_keys_this_file_has_never_heard_of(self):
        # `brand_new_axis` 是故意编的: 它模拟「以后 verifier 新加了一个轴」。白名单的做法是
        # 这里要记得补一行, 而漏掉的表现不是报错而是那个键永远不在产物里。
        axes = {"arithmetic_intensity": 3.5, "brand_new_axis": [1, 2, 3]}
        out = _round_axes(_l3({"roofline_analysis": axes}), {})
        self.assertEqual(axes, out["roofline"])
        self.assertIsNot(out["roofline"], axes, "要复制一份, 不许把内部字典的引用交出去")

    def test_a_round_that_never_reached_l3_writes_no_roofline_key(self):
        for 没有l3 in (None, _l3({}), _l3({"roofline_analysis": {}})):
            out = _round_axes(没有l3, {"X": np.zeros((4, 4), np.float32)})
            self.assertNotIn("roofline", out,
                             f"{没有l3!r} 时写空块会把「没测过」说成「测了, 值是空」")

    def test_shapes_and_dtypes_are_recorded_only_for_arrays(self):
        inputs = {"X": np.zeros((1024, 512), np.float32),
                  "weight": np.ones(512, np.float32),
                  "eps": 1e-5}
        out = _round_axes(None, inputs)
        self.assertEqual({"X": [1024, 512], "weight": [512]}, out["input_shapes"])
        self.assertEqual({"X": "float32", "weight": "float32"}, out["input_dtypes"])
        self.assertNotIn("eps", out["input_shapes"], "标量没有 shape, 不该被塞成一个假形状")

    def test_no_inputs_at_all_writes_no_shape_keys(self):
        self.assertEqual({}, _round_axes(None, None))
        self.assertEqual({}, _round_axes(None, {}))

    def test_declared_dtype_not_numpy_dtype(self):
        """AI 走**声明精度**而不是 numpy 那份的 itemsize —— doc 48 §6.5 那个 2 倍口径错误。

        这条用例是差分 + 绝对两侧: 绝对值 1365.3333 是独立算出来的 (2·M·N·K / ((M·K+K·N+M·N)·2)),
        差分那侧直接调 verifier 自己那个函数, 保证血缘里的轴与判决用的轴同源。
        """
        task = _task("matmul", Precision.FP16)
        inputs = {"A": np.zeros((4096, 4096), np.float32),   # 故意是 float32
                  "B": np.zeros((4096, 4096), np.float32)}   # 4096³: AI = 1365.3333
        axes = {"arithmetic_intensity": AI_OF(task, inputs)}
        got = _round_axes(_l3({"roofline_analysis": axes}), inputs)["roofline"]
        self.assertAlmostEqual(1365.3333333, got["arithmetic_intensity"], places=4)
        self.assertEqual(axes["arithmetic_intensity"], got["arithmetic_intensity"])
        self.assertEqual({"A": [4096, 4096], "B": [4096, 4096]},
                         _round_axes(_l3({"roofline_analysis": axes}), inputs)["input_shapes"],
                         "dtype 走声明精度, 但形状仍然是真实那份 —— 两个都要留才查得出口径")


class AxesCrossThePersistenceBoundary(unittest.TestCase):
    """从产物里读回来: 单测读内存对象看不见落盘边界 (doc 49 §7.9 那条教训)。"""

    PYTHON_MATMUL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
    SKETCH = {"algorithm": "matmul",
              "tiling": {"block_m": 32, "block_n": 32, "block_k": 16},
              "memory_plan": {}, "pipeline": [], "optimization_hints": []}

    class _Llm:
        def __init__(self, outer):
            self.outer = outer
            self.calls = 0

        def chat(self, messages, **kw):
            self.calls += 1
            return self.outer.PYTHON_MATMUL

        def chat_json(self, messages, **kw):
            self.calls += 1
            return dict(self.outer.SKETCH)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _conductor(self):
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                      llm=AxesCrossThePersistenceBoundary._Llm(self),
                      max_optimize_rounds=1)
        from operator_agent.conductor import Conductor as C
        c.store.output_dir = Path(self.tmp.name)
        return c

    def test_shapes_that_reach_result_json_are_the_ones_we_passed(self):
        c = self._conductor()
        inputs = {"A": np.zeros((48, 64), np.float32), "B": np.zeros((64, 32), np.float32)}
        task = _task("matmul", Precision.FP32)
        art = c.execute(task, inputs, None, target_gflops=1.0)
        result = json.loads((Path(art.saved_dir) / "result.json").read_text(encoding="utf-8"))
        round1 = result["iteration_history"][0]
        self.assertEqual({"A": [48, 64], "B": [64, 32]}, round1["input_shapes"],
                         "血缘里的形状必须是**这次真的喂进去的那份**, 不是事后从 task 猜的")
        self.assertEqual({"A": "float32", "B": "float32"}, round1["input_dtypes"])
        # MockGPU 确实会产这块轴, 但峰值是 0 / `peak_trusted=False` / `roofline_bound=unknown`
        # ⇒ 一条**没标定过的曲线点**必须带着这个事实进产物, 否则离线复盘会把它当分母用。
        # (顺带抓出一处不一致: 同一个块里 `roofline_bound` 是 unknown 而 `name_bound`/`bound_type`
        #  是 compute —— 那是 doc 49 §八 第 10 条那类「两个口径住在同一个字典里」的形状, 记成账。)
        self.assertIn("roofline", round1)
        self.assertFalse(round1["roofline"]["peak_trusted"])
        self.assertEqual("unknown", round1["roofline"]["roofline_bound"])
        self.assertEqual(0, round1["roofline"]["hw_peak_bandwidth_gbs"])

    def test_snapshot_carries_the_same_axes_as_result_json(self):
        """快照的 `iteration_history` 与 result.json 的是同一份血缘 —— 否则续跑会丢掉轴。"""
        c = self._conductor()
        inputs = {"A": np.ones((16, 16), np.float32), "B": np.ones((16, 16), np.float32)}
        art = c.execute(_task("matmul", Precision.FP32), inputs, None, target_gflops=1.0)
        snap = json.loads((Path(art.saved_dir) / "execution_snapshot.json")
                          .read_text(encoding="utf-8"))
        res = json.loads((Path(art.saved_dir) / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(res["iteration_history"], snap["iteration_history"])


class TheHelpersAreNotVacuous(unittest.TestCase):
    """守卫自己也要能被撞一次。"""

    def test_ai_is_actually_computed_for_layernorm(self):
        """0.875 来自 doc 50 §7.2 的真机产物, 不是从代码里读出来的。

        如果哪天 `_compute_arithmetic_intensity` 的 layernorm 分支漂了, 这条会红,
        而 `AxesAreCarriedWholesale` 那批差分用例不会 —— 它们两侧一起漂。
        """
        inputs = {"X": np.zeros((1024, 512), np.float32),
                  "weight": np.ones(512, np.float32), "bias": np.zeros(512, np.float32)}
        self.assertAlmostEqual(0.875, AI_OF(_task("layernorm", Precision.FP32), inputs),
                               places=6)

    def test_round_axes_is_not_returning_a_hardcoded_dict(self):
        out_a = _round_axes(_l3({"roofline_analysis": {"arithmetic_intensity": 1.0}}),
                            {"X": np.zeros((2, 2), np.float32)})
        out_b = _round_axes(_l3({"roofline_analysis": {"arithmetic_intensity": 9.0}}),
                            {"Y": np.zeros((3, 4), np.float32)})
        self.assertNotEqual(out_a, out_b, "换了输入还返回同一份东西, 这个函数就是在空转")


if __name__ == "__main__":
    unittest.main(verbosity=2)
