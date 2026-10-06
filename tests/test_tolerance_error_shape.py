"""误差形态 (设计 §8.2): 分类规则 + **两个生产者的一致性**

「第一个坏点」回答**哪错了** (位置); 「误差形态」回答**该怎么修** (类别) ——
后者才是"修 vs 重写"最直接的依据 (设计 §8.2 / §21.2)。

这个文件钉三件事:

1. **规则本身**: 五种形态各自的判据与优先级, 以及"判不了就给 None"的那一态 ——
   一个"看着像但不成立"的标签会把修复引到错的地方, 比没有标签更坏;
2. **两个生产者不漂移**: 规则只有一份 (`tolerance.classify_error_shape`), 但归约在
   各自的数组后端上做 (本地 numpy / 真机 torch) ⇒ 同一份掩码喂给两边必须得到**同一个
   标签**。少了这条, 真机与本地会各自演化出一套"形态";
3. **接线**: 形态要真的走到读者手里 —— 远端键进白名单 (`_REMOTE_L2_METRIC_KEYS`)、
   进 L2 的 message (Fixer 与失败语料读它)、进 `failure_route` 的修复提示。
   形态**不改路由决策** (它是证据不是判据), 这一点也钉在这里。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent import tolerance
from operator_agent.tolerance import (ErrorShapeInputs, classify_error_shape,
                                      error_shape_from_axis_flags,
                                      summarize_bad_axes, ERROR_SHAPES,
                                      ERROR_SHAPE_BROADCAST_AXIS,
                                      ERROR_SHAPE_NONFINITE,
                                      ERROR_SHAPE_TAIL_BLOCK,
                                      ERROR_SHAPE_WHOLE_ROW,
                                      ERROR_SHAPE_WHOLE_TENSOR)

try:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "remote_scripts"))
    import run_kernel
except Exception as _exc:  # pragma: no cover - 环境缺 torch 时跳过真机侧的一致性
    torch = None
    run_kernel = None
    _RUNNER_IMPORT_ERROR = _exc


def _inputs(**kw):
    """分类器输入的默认值: 8×8 全坏, 再按需覆盖"""
    base = dict(n_bad=64, total=64, rows=8, cols=8, bad_rows=8, bad_cols=8,
                rows_all_bad=8, cols_all_bad=8, bad_row_prefix=8, bad_row_suffix=8,
                bad_col_prefix=8, bad_col_suffix=8)
    base.update(kw)
    return ErrorShapeInputs(**base)


def _mask_from(shape, bad_rows=(), bad_cols=(), bad_cells=()):
    """造一个坏点掩码 —— 测试里只说"哪几行/哪几列/哪几格坏", 不手算统计量"""
    m = np.zeros(shape, dtype=bool)
    for r in bad_rows:
        m[r, :] = True
    for c in bad_cols:
        m[:, c] = True
    for cell in bad_cells:
        m[cell] = True
    return m


class TestSummarizeAxes(unittest.TestCase):

    def test_counts_and_indices(self):
        per_row = [False, True, True, False]
        per_col = [True, False, False, True]
        s = summarize_bad_axes(per_row, per_col)
        self.assertEqual(s["rows"], 4)
        self.assertEqual(s["cols"], 4)
        self.assertEqual(s["bad_rows"], 2)
        self.assertEqual(s["bad_cols"], 2)
        self.assertEqual(s["first_bad_row"], 1)
        self.assertEqual(s["last_bad_row"], 2)
        self.assertEqual(s["first_bad_col"], 0)
        self.assertEqual(s["last_bad_col"], 3)

    def test_prefix_and_suffix(self):
        s = summarize_bad_axes([True, True, False, True], [True, False, False, True])
        self.assertEqual(s["bad_row_prefix"], 2)
        self.assertEqual(s["bad_row_suffix"], 1)
        self.assertEqual(s["bad_col_prefix"], 1)
        self.assertEqual(s["bad_col_suffix"], 1)

    def test_all_clean(self):
        s = summarize_bad_axes([False, False], [False, False])
        self.assertEqual(s["bad_rows"], 0)
        self.assertIsNone(s["first_bad_row"])
        self.assertEqual(s["bad_row_suffix"], 0)


class TestClassifierRules(unittest.TestCase):

    def test_nonfinite_wins_over_everything(self):
        """NaN/Inf 压倒一切: 别的形态都是从"哪几个元素错"推出来的, 而那些位置统计
        在含非有限值时本身就没有意义 (NaN 的比较恒假 ⇒ 坏点计数还是 0)"""
        self.assertEqual(classify_error_shape(
            _inputs(has_nonfinite=True, n_bad=0, bad_rows=0, bad_cols=0, rows_all_bad=0,
                    cols_all_bad=0, bad_row_prefix=0, bad_row_suffix=0,
                    bad_col_prefix=0, bad_col_suffix=0)), ERROR_SHAPE_NONFINITE)
        # 连形状都没有 (空张量) 时也要说得出来
        self.assertEqual(classify_error_shape(ErrorShapeInputs(
            n_bad=0, total=0, rows=0, cols=0, bad_rows=0, bad_cols=0, rows_all_bad=0,
            cols_all_bad=0, bad_row_prefix=0, bad_row_suffix=0, bad_col_prefix=0,
            bad_col_suffix=0, has_nonfinite=True)), ERROR_SHAPE_NONFINITE)

    def test_no_bad_points_means_no_shape(self):
        self.assertIsNone(classify_error_shape(_inputs(
            n_bad=0, bad_rows=0, bad_cols=0, rows_all_bad=0, cols_all_bad=0,
            bad_row_prefix=0, bad_row_suffix=0, bad_col_prefix=0, bad_col_suffix=0)))

    def test_whole_tensor_at_and_above_the_threshold(self):
        self.assertEqual(classify_error_shape(_inputs(n_bad=99, total=100)),
                         ERROR_SHAPE_WHOLE_TENSOR)
        # 差一个点就不算整张量 —— 阈值是判据的一部分, 不许被"差不多"读掉
        self.assertNotEqual(classify_error_shape(_inputs(
            n_bad=98, total=100, bad_rows=8, rows_all_bad=0, bad_row_suffix=0,
            bad_col_suffix=0, bad_col_prefix=0)), ERROR_SHAPE_WHOLE_TENSOR)

    def test_tail_block_is_suffix_and_small(self):
        """尾块 = 行方向的**末尾连续一小段** (非对齐尾块 / 最后一行)"""
        self.assertEqual(classify_error_shape(_inputs(
            n_bad=8, total=64, bad_rows=1, rows_all_bad=1, bad_row_prefix=0,
            bad_row_suffix=1, bad_cols=8, cols_all_bad=0, bad_col_prefix=8,
            bad_col_suffix=8)), ERROR_SHAPE_TAIL_BLOCK)

    def test_middle_full_row_is_not_a_tail_block(self):
        """中间整行错不是尾块 —— 要修的地方完全不同 (这是尾块判据里"连续"那半"""
        self.assertEqual(classify_error_shape(_inputs(
            n_bad=8, total=64, bad_rows=1, rows_all_bad=1, bad_row_prefix=0,
            bad_row_suffix=0, bad_cols=8, cols_all_bad=0, bad_col_prefix=8,
            bad_col_suffix=8)), ERROR_SHAPE_WHOLE_ROW)

    def test_middle_full_column_is_broadcast_axis(self):
        self.assertEqual(classify_error_shape(_inputs(
            n_bad=8, total=64, bad_cols=1, cols_all_bad=1, bad_col_prefix=0,
            bad_col_suffix=0, bad_rows=8, rows_all_bad=0, bad_row_prefix=8,
            bad_row_suffix=8)), ERROR_SHAPE_BROADCAST_AXIS)

    def test_partial_row_in_the_middle_gives_none(self):
        """一行里只坏了一半 ⇒ 三态里的"无法确立", 不许硬套成 whole_row"""
        self.assertIsNone(classify_error_shape(_inputs(
            n_bad=4, total=64, bad_rows=1, rows_all_bad=0, bad_row_prefix=0,
            bad_row_suffix=0, bad_cols=4, cols_all_bad=0, bad_col_prefix=0,
            bad_col_suffix=0)))

    def test_whole_row_requires_exactly_whole_rows(self):
        """整行错的定义: 坏点恰好是若干**完整行** (否则是别的形态)"""
        # 两整行 = 16 个坏点 ⇒ whole_row
        self.assertEqual(classify_error_shape(_inputs(
            n_bad=16, total=64, bad_rows=2, rows_all_bad=2, bad_row_prefix=0,
            bad_row_suffix=0, bad_cols=8, cols_all_bad=0, bad_col_prefix=8,
            bad_col_suffix=8)), ERROR_SHAPE_WHOLE_ROW)
        # 同样集中在两行, 但整行之外还有零散坏点 (n_bad 对不上) ⇒ 不是 whole_row
        self.assertNotEqual(classify_error_shape(_inputs(
            n_bad=17, total=64, bad_rows=2, rows_all_bad=2, bad_row_prefix=0,
            bad_row_suffix=0, bad_cols=8, cols_all_bad=0, bad_col_prefix=8,
            bad_col_suffix=8)), ERROR_SHAPE_WHOLE_ROW)

    def test_every_label_is_declared(self):
        """标签集合与常量表一致 —— 免得加了一个标签却忘了进 ERROR_SHAPES"""
        self.assertEqual(set(ERROR_SHAPES), {
            ERROR_SHAPE_NONFINITE, ERROR_SHAPE_WHOLE_TENSOR, ERROR_SHAPE_TAIL_BLOCK,
            ERROR_SHAPE_WHOLE_ROW, ERROR_SHAPE_BROADCAST_AXIS})
        self.assertNotIn("precision_accumulation", ERROR_SHAPES,
                         "设计 §8.2 的第六条还没有量具 (误差随 k 增长需要另一种证据): "
                         "它必须留在'判不了'那一态, 不许先占个名字")


@unittest.skipIf(run_kernel is None, "远端 runner 或其依赖 (torch) 不可用")
class TestTwoProducersAgree(unittest.TestCase):
    """同一份坏点掩码, 本地 (numpy) 与真机 (torch) 必须给出**同一个标签**

    规则只有一份, 但归约写了两遍 (各自的数组后端)。这条用例就是防那两遍漂移的 ——
    形态判错不会让错核通过 (它是证据不是判据), 但会让 Fixer 先看错地方。
    """

    def _both(self, mask, *, has_nonfinite=False):
        n_bad = int(mask.sum())
        total = int(mask.size)
        local = self._local(mask, n_bad, total, has_nonfinite=has_nonfinite)
        remote = run_kernel._error_shape_torch(
            torch.from_numpy(np.ascontiguousarray(mask)), n_bad=n_bad, total=total,
            has_nonfinite=has_nonfinite)
        return local, remote

    @staticmethod
    def _local(mask, n_bad, total, *, has_nonfinite=False):
        from operator_agent.verifier import Verifier
        return Verifier._error_shape_of(mask, n_bad=n_bad, total=total,
                                        has_nonfinite=has_nonfinite)

    def test_agreement_over_a_grid_of_scenarios(self):
        cases = {
            "整张量": _mask_from((8, 8), bad_rows=range(8)),
            "尾块": _mask_from((8, 8), bad_rows=[7]),
            "末尾两行": _mask_from((8, 8), bad_rows=[6, 7]),
            "中间整行": _mask_from((8, 8), bad_rows=[3]),
            "中间整列": _mask_from((8, 8), bad_cols=[3]),
            "末列": _mask_from((8, 8), bad_cols=[7]),
            "散点": _mask_from((8, 8), bad_cells=[(2, 2), (5, 6)]),
            "半行": _mask_from((8, 8), bad_cells=[(3, c) for c in range(4)]),
            "一维全坏": np.ones(16, dtype=bool),
            "一维尾段": np.array([False] * 14 + [True, True]),
            "四维尾部": np.zeros((2, 2, 4, 4), dtype=bool),
        }
        cases["四维尾部"][:, :, -1, :] = True
        for name, mask in cases.items():
            with self.subTest(case=name):
                local, remote = self._both(mask)
                self.assertEqual(local, remote,
                                 f"{name}: 本地={local!r} 真机={remote!r} —— 两处归约漂了")

    def test_agreement_on_nonfinite(self):
        mask = np.zeros((4, 4), dtype=bool)          # NaN 时坏点计数是 0
        local, remote = self._both(mask, has_nonfinite=True)
        self.assertEqual(local, ERROR_SHAPE_NONFINITE)
        self.assertEqual(remote, ERROR_SHAPE_NONFINITE)


class TestWiring(unittest.TestCase):
    """形态要真的走到读者手里 (算了没人读 = 没算)"""

    def _tail_result(self):
        from operator_agent import MockGPU, Verifier
        from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                            TargetSpec, TaskComplexity, TaskSchema,
                                            TaskType)
        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
            target=TargetSpec(hardware="mock_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0,
                                       max_error=1e-5))
        rng = np.random.default_rng(0)
        inputs = {"A": rng.standard_normal((64, 64)).astype(np.float32),
                  "B": rng.standard_normal((64, 64)).astype(np.float32)}
        code = ("import numpy as np\n\ndef kernel(A, B):\n    C = A @ B\n"
                "    C[-1, :] = 0.0\n    return C\n")
        return Verifier(MockGPU())._verify_function(code, inputs, None, task)

    def test_shape_is_reported_in_the_message(self):
        """message 是唯一保证被下游看到 (Fixer / 失败语料) 的通道"""
        r = self._tail_result()
        self.assertEqual(r.metrics["error_shape"], ERROR_SHAPE_TAIL_BLOCK)
        self.assertIn("形态=tail_block", r.message)

    def test_route_carries_the_shape_and_a_repair_hint(self):
        """形态进路由的**理由与字段**, 但**不改路由决策** (证据不是判据)"""
        from operator_agent.failure_route import (ROUTE_DEBUG_FIX, classify)
        r = self._tail_result()
        route = classify([r])
        self.assertEqual(route.route, ROUTE_DEBUG_FIX,
                         "形态不该改变路由决策 —— 它只影响提示")
        self.assertTrue(route.fix_not_rewrite)
        self.assertEqual(route.error_shape, ERROR_SHAPE_TAIL_BLOCK)
        self.assertIn("尾块", route.reason)
        self.assertEqual(route.as_dict()["error_shape"], ERROR_SHAPE_TAIL_BLOCK)

    def test_route_says_so_when_the_shape_is_unestablished(self):
        """判不了也要说出来: 沉默会被读成"形态没问题" (空章节必须带依据同一条纪律)"""
        from operator_agent import MockGPU, Verifier
        from operator_agent.failure_route import classify
        from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                            TargetSpec, TaskComplexity, TaskSchema,
                                            TaskType)
        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
            target=TargetSpec(hardware="mock_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0,
                                       max_error=1e-5))
        rng = np.random.default_rng(0)
        inputs = {"A": rng.standard_normal((64, 64)).astype(np.float32),
                  "B": rng.standard_normal((64, 64)).astype(np.float32)}
        code = ("import numpy as np\n\ndef kernel(A, B):\n    C = A @ B\n"
                "    C[3, 4] += 5.0\n    C[9, 9] += 5.0\n    return C\n")
        r = Verifier(MockGPU())._verify_function(code, inputs, None, task)
        self.assertIsNone(r.metrics["error_shape"])
        route = classify([r])
        self.assertIsNone(route.error_shape)
        self.assertIn("未确立", route.reason)

    def test_hint_table_covers_every_declared_label(self):
        from operator_agent.failure_route import ERROR_SHAPE_HINTS
        for label in ERROR_SHAPES:
            self.assertIn(label, ERROR_SHAPE_HINTS, f"{label} 没有对应的修复方向")


    def test_shape_suffix_does_not_hijack_the_failure_signature(self):
        """形态文本进的是**同一条 message**, 而失败类与签名是从它里面按**关键词**认出来的
        (`kernel_db._FAILURE_CLASS_MARKERS` / `failure_signature`)。

        加一句"形态=…"若撞上某个 marker, 这条失败就会换一个类 —— 复发门于是数到别的桶里,
        而那是**静默的**: 形态看起来只是多了个标签。`broadcast_axis` 是最像 marker 的一个。
        """
        from operator_agent.kernel_db import failure_signature
        r = self._tail_result()
        suffix = ", 形态=tail_block"
        self.assertIn(suffix, r.message)
        self.assertEqual(failure_signature("L2", r.message),
                         failure_signature("L2", r.message.replace(suffix, "")),
                         "形态那句改变了失败签名 —— 复发门会数到别的桶里")
        for label in ERROR_SHAPES:
            with self.subTest(label=label):
                self.assertEqual(
                    failure_signature("L2", f"功能验证失败: 1/2 个可比元素同时超 ……, 形态={label}"),
                    failure_signature("L2", "功能验证失败: 1/2 个可比元素同时超 ……"),
                    f"{label} 撞上了失败类关键词 —— 它会把这条失败归到别的桶")


if __name__ == "__main__":
    unittest.main(verbosity=2)
