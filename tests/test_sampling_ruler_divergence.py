"""采样窗口那条 L2 用的是另一把尺 —— 把它量出来 (#80 第 1 片，只观测)

诊断 (缺口 1 的这条尾巴，读代码核实过，不是推断)：L2 有**三条**比对路，只有两条走同一把尺。

| 路 | 判据 | 报哪些尺键 |
|---|---|---|
| 全量 (`Cap.FULL_OUTPUT_TENSOR` 在场) | `tolerance.effective_atol` + AND 形式 + ρ | `atol`/`rtol`/`required_matched_ratio` |
| 真机 (`remote_scripts/run_kernel.py`) | 同一个 `tolerance` 模块 | 同上三键 (逐键透传) |
| **采样窗口** (`FULL_OUTPUT_TENSOR` 不在) | `max_error <= task.constraints.max_error` | **一个都不报** |

两个缺陷叠在一起：

1. **这把尺只会同松或更松**：`effective_atol` 取的是 `min(表, 任务自带值)`，所以 AND 那条的
   绝对阈 **≤** 采样这条的；采样这条还没有相对项、没有 ρ。一个整体缩放 0.997 的候选在
   `max_error=1e-2` 的 fp32 任务上能从采样窗口放过去，而全量那条会拦 —— 正是 doc 12 立
   `tolerance.py` 时要拦的那一类 (「允许整体缩小 14.93%」的孪生)。
2. **它一个尺键都不报** ⇒ `single_ruler`（轮级）与 `l2_ruler_reading` / `cross_round_ruler`
   （运行级）在混用两条路时只能读成「这一轮没报尺」。`RULER_UNREPORTED` 与「这一轮的尺与
   别轮不同」是两件事，前者的修法在查部署，后者的修法在查口径 —— 把它们读成同一件，#87 那条
   守卫就等于没生效。

这一片**不改判决**（换尺要带真机 A/B，挂号在下面），只做观测：如实发布这一窗真正用过的
`atol`，把同一条 AND 判据套在同一窗上的第二个读数发布成 `and_form_*`，并在逐组与聚合两层
message 里说出一致/分歧。判定式仍只有 `tolerance` 那一份 —— 这里不调形状，只多问它一次。

实跑读数 (`benchmarks/probe_l2_sampling_gate.py` 臂 4, 2026-09-30 控制机): 一份常数偏移
3e-3 的候选配大幅值参考 —— 纯绝对那条 `atol=5.00e-01` 判 **PASS**, AND 那条 `atol=1.00e-04`
判 **FAIL**。同一个探针的臂 2 在参考幅值 0.1 与 0.01 两档上给的是同一个方向。所以「这条尺只会
同松或更松」不是推论, 是探针今天就能印出来的两列。

变异标定 (每条只改一处, 跑完立刻逐字节复原并断言。2026-09-30 实跑,
**行为分母 10/10 + 结构分母 1/1 全拦下**；表在 `benchmarks/mut_specs/sampling_ruler_divergence.py`):
    S1  采样那条不发布自己用过的 atol (退回本片之前)     → 4 条红
    S2  发布的 atol 用表值而不是这一窗真正用的值          → 2 条红
    S3  把缺席的 rtol 补成值 (造证据)                    → 2 条红
    S4  第二个读数整个不算                               → 17 条红
    S5  [结构] 判定式自己写、不问那个模块                 → 1 条红 (ρ=1.0 时行为不可区分)
    S6  tolerance 的 min() 翻成 max() (任务能放宽判据)    → 14 条红
    S7  聚合里点名用了哪几组时把数写成死字 0              → **2 条红 (第一版逃逸)**
    S8  分歧收集条件反过来                               → 3 条红
    S9  一致时也照样印分歧                               → 2 条红
    S10 顺手把判决换成那把紧尺 (本片明说没做的事)          → 15 条红
    S11 聚合不再发布 `absolute_only_sets`                → 3 条红

⚠️ **S7 第一版是逃逸的** (0 红)，而且它撞的东西与本文件上面那段诊断是同一族的第三种形状:
那条用例写的是 `assertIn("纯绝对", agg.message)` —— 词在哪都会印, 而**数字**才是读数。改成
`assertIn("⚠️ 1/1 组比的是采样窗口那把", …)` 之后 S7 才有牙齿。同一次还把「日志那句」与
「metrics 那两个名单」拆成两个方法, 否则 S7 与 S8 的红名单逐条相同 (标定报的粒度是方法名)。

挂号 (这一片刻意不做的事)：① 把采样那条换成同一把尺 —— 只会**收紧**，但会改真机通过率，
要 A/B；② 反作弊侧对「采样窗口」这一路该用更严的 ρ 而不是 1.0（窗口小，一个坏点就占 1/N），
那与 #84 同一片地；③ 容差表尚未在 H20 上用已知正确的核标定过 (`tolerance.py` 自己写着)。

运行: python3 -m pytest tests/test_sampling_ruler_divergence.py -q
"""
from __future__ import annotations

import ast
import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from operator_agent import Verifier  # noqa: E402
from operator_agent.measurement import (  # noqa: E402
    LINEAGE_RULER_KEY, RULER_REPORTED, RULER_UNREPORTED, cross_round_ruler, l2_ruler_reading,
)
from operator_agent.backend_caps import Cap  # noqa: E402
from operator_agent.mock_gpu import ExecutionResult  # noqa: E402
from operator_agent.remote_executor import RemoteExecutionResult  # noqa: E402
from operator_agent.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType,
)
from tests.backend_fakes import DeclaredFakeBackend  # noqa: E402

#: 8×8、幅值刻意取 1.0 附近, 这样「整体缩放 k」的绝对误差就是 (1-k) 而相对误差也是 (1-k)
REF = np.full((8, 8), 1.0, dtype=np.float32)
CODE = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"


def _task(max_error=1e-2, precision=Precision.FP32):
    return TaskSchema(task_id="t-samp-ruler", task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
                      target=TargetSpec(hardware="mock_gpu", dsl="python"),
                      constraints=ConstraintSpec(precision=precision, min_speedup=0.0,
                                                 max_error=max_error))


def _sampling_verifier(out):
    """一个只回样本的回显后端 ⇒ 走的就是 `Cap.FULL_OUTPUT_TENSOR` 不在的那一条"""

    def execute(kernel_code, inputs):
        return RemoteExecutionResult(success=True, output_shape=list(np.asarray(out).shape),
                                     output_sample=list(np.asarray(out).reshape(-1)))

    return Verifier(DeclaredFakeBackend(caps=(), execute=execute))


def _l2_sampling(out, max_error=1e-2, precision=Precision.FP32):
    with contextlib.redirect_stdout(io.StringIO()):
        return _sampling_verifier(out)._verify_single_set(
            CODE, {"A": np.ones((8, 8), np.float32)}, REF,
            _task(max_error, precision))


def _l2_sampling_multi(out, max_error=1e-2, precision=Precision.FP32):
    """走真实的 `_verify_function` 聚合那一条 (单组也会经过它) —— 聚合的 message 是重算的,
    逐组那句会被丢掉, 所以运行级那两句必须由聚合自己说"""
    v = _sampling_verifier(out)
    with contextlib.redirect_stdout(io.StringIO()):
        return v._verify_function(CODE, [{"A": np.ones((8, 8), np.float32)}], REF,
                                  _task(max_error, precision))


class TestTheSamplingPathPublishesItsOwnRuler(unittest.TestCase):
    """缺陷 2: 这条路以前一个尺键都不报, 三个已有的读者读成「没测」"""

    def test_the_fixture_really_is_on_the_sampling_path(self):
        r = _l2_sampling(REF)
        self.assertTrue(r.passed, r.message)
        self.assertIn("采样比对通过", r.message)
        self.assertIn("元素可比", r.message)

    def test_atol_is_the_tasks_own_value_and_the_other_two_stay_absent(self):
        """`atol` = 这一窗**真正**用过的绝对阈; `rtol` / ρ 缺席是因为这把尺根本没有那两项

        把缺席补成 0 或 1.0 就是造证据 (#53/#81 那一族), 所以这里断言的是「键不在」。
        """
        r = _l2_sampling(REF, max_error=1e-2)
        self.assertEqual(1e-2, r.metrics["atol"])
        self.assertNotIn("rtol", r.metrics)
        self.assertNotIn("required_matched_ratio", r.metrics)

    def test_the_ruler_reading_flips_from_unreported_to_reported(self):
        reading = l2_ruler_reading(_l2_sampling(REF).metrics)
        self.assertEqual(RULER_REPORTED, reading["state"],
                         f"聚合器仍然读成「这一轮没报尺」: {reading}")
        self.assertEqual({"state", "atol"}, set(reading))

    def test_a_full_tensor_round_is_still_unreported_when_it_has_no_gate_keys(self):
        """反向对照: `reported` 不是因为「任何 metrics 都算报了尺」"""
        self.assertEqual(RULER_UNREPORTED, l2_ruler_reading({"max_error": 1e-9}).get("state"),
                         "没有尺键时不许读成 reported")

    def test_two_rounds_from_the_two_paths_now_show_a_changed_ruler(self):
        """这一条是「三个读者第一次看得见」的现场: 采样组 atol=1e-2, 全量组 atol=1e-4 (表)

        改之前采样那一组不报 `atol` ⇒ `atol` 上只有一轮有数 ⇒ `ruler_same` 是 None
        (「测不到」)。现在两轮都在场, 于是那把松尺第一次被当成**证据**报出来。
        """
        sampling = l2_ruler_reading(_l2_sampling(REF, max_error=1e-2).metrics)
        full = l2_ruler_reading(_full_round_metrics())
        agg = cross_round_ruler([{LINEAGE_RULER_KEY: sampling, "round": 1},
                                 {LINEAGE_RULER_KEY: full, "round": 2}])
        self.assertFalse(agg["ruler_same"], f"两把尺没被看出来: {agg}")
        self.assertIn("atol", agg["ruler_changed"])
        self.assertEqual({1: 1e-2, 2: _full_round_atol()},
                         agg["ruler_changed"]["atol"])


class TestTheTwoPathsAreStillTellableApart(unittest.TestCase):
    """给键当路径标记的那只手也在这条线上 —— 它只在本片改完之后对了 half 一次

    `benchmarks/probe_l2_sampling_gate.branch_of` 第二版是 `if "atol" in metrics: AND 形式`,
    而本片给采样那条新增了 `atol` ⇒ 那个判别键当场失效 (表里每一行都印成「认不出哪条路」,
    臂 0b 于是把「这条路没算的量」误判成「AND 那条缺了那些量」)。它现在认的是两条路各自独有的
    那个键。这条用例就是让那个失效**下次必须红**, 而不是靠有人去重跑探针。
    """

    def setUp(self):
        from benchmarks.probe_l2_sampling_gate import branch_of
        self.branch_of = branch_of

    def test_the_sampling_path_is_recognised_per_group_and_in_aggregate(self):
        self.assertEqual("纯绝对", self.branch_of(_l2_sampling(REF)))
        self.assertEqual("纯绝对", self.branch_of(_l2_sampling_multi(REF)),
                         "聚合那层不带逐组的 `and_form_*`, 判别键要用 `absolute_only_sets`")

    def test_the_and_path_is_not_mislabelled(self):
        self.assertEqual("AND 形式", self.branch_of(_full_path_result()))
        self.assertEqual("AND 形式", self.branch_of(_full_path_result(multi=True)))

    def test_atol_is_no_longer_a_path_marker(self):
        """把这条单独钉住: 两条路**都**报 `atol` 了, 谁再拿它当标记就会把一半的行印错"""
        self.assertIn("atol", _l2_sampling(REF).metrics)
        self.assertIn("atol", _full_path_result().metrics)


class TestTheSecondReadingUsesTheSharedRule(unittest.TestCase):
    """缺陷 1 的测量半边: 同一条 AND 判据套在同一窗上, 规则仍只有 `tolerance` 那一份"""

    def test_table_wins_when_it_is_tighter_than_the_task(self):
        r = _l2_sampling(REF, max_error=1e-2, precision=Precision.FP32)
        self.assertEqual(1e-4, r.metrics["and_form_atol"], "min(表 1e-4, 任务 1e-2)")
        self.assertEqual(1e-4, r.metrics["and_form_rtol"])
        self.assertEqual(1.0, r.metrics["and_form_required_matched_ratio"])

    def test_task_wins_when_it_is_tighter_than_the_table(self):
        """另一侧也取一次样: 只测 min() 成立的那一边, 测的是代码在不在, 不是条件对不对"""
        r = _l2_sampling(REF, max_error=1e-6, precision=Precision.FP32)
        self.assertEqual(1e-6, r.metrics["and_form_atol"])
        self.assertEqual(1e-4, r.metrics["and_form_rtol"], "rtol 不走 min()")

    def test_the_and_form_thresholds_are_not_a_second_copy_of_the_rule(self):
        """第二个读数必须**问**那个模块, 不许在 verifier 里再写一遍比较式"""
        src = (_ROOT / "operator_agent" / "verifier.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_verify_single_set")
        block = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                     and "FULL_OUTPUT_TENSOR" in ast.get_source_segment(src, n.test))
        body = ast.get_source_segment(src, block)
        self.assertGreaterEqual(body.count("tolerance.effective_atol("), 1,
                                "采样那条没调用容差合并规则 ⇒ 这两个读数不是同一条规则给的")
        self.assertIn("tolerance.evaluate(", body)
        self.assertNotIn("matched_ratio >=", body,
                         "自己写判定就是把规则抄第二遍 (#87 那一族的形状)")


class TestTheDivergenceIsVisible(unittest.TestCase):
    """最危险的那个方向 (纯绝对放行、AND 会拦) 必须在读数与日志里都看得到"""

    SCALED = REF * 0.997          # 绝对误差 3e-3 ≤ 任务的 1e-2, 但 同时超 atol 与 rtol

    def test_a_scaled_candidate_passes_here_and_would_be_rejected_there(self):
        r = _l2_sampling(self.SCALED, max_error=1e-2)
        self.assertTrue(r.passed, "这一片不改判决 —— 它仍然是通过")
        self.assertFalse(r.metrics["and_form_passed"],
                         "同一条 AND 判据应当拦下它 (doc 12 立 `tolerance` 时要拦的那一类)")
        self.assertEqual(64, r.metrics["and_form_bad_elements"])
        self.assertIn("⚠️ 分歧", r.message)
        self.assertIn("纯绝对", r.message)

    def test_the_aggregate_names_the_groups_in_the_log(self):
        """断言打在**带数字的位置**上, 不是打在「纯绝对」这个词上

        S7 第一次是逃逸的: 它把那句报警里的 `{len(abs_only_sets)}` 写成死字 0, 而这条用例只查
        「句子里有没有『纯绝对』」⇒ 全体成立。词在哪都会印, 数字才是读数 (I32 的下一种形状)。
        """
        agg = _l2_sampling_multi(self.SCALED, max_error=1e-2)
        self.assertTrue(agg.passed, agg.message)
        self.assertIn("⚠️ 1/1 组比的是采样窗口那把", agg.message)
        self.assertIn("⚠️ 其中 1 组按那把尺", agg.message)
        self.assertIn("组 [1]", agg.message)

    def test_the_aggregate_publishes_the_group_lists_in_metrics(self):
        """日志与 metrics 拆成两条: S7 (改日志那句) 与 S8 (改收集条件) 的红才不会挤在同一个方法里

        上一片刚把这条规矩写进 doc 57 §十 —— 标定报的粒度是**方法名**, 一行变异要有它自己的红。
        """
        agg = _l2_sampling_multi(self.SCALED, max_error=1e-2)
        self.assertEqual([1], agg.metrics["absolute_only_sets"])
        self.assertEqual([1], agg.metrics["and_form_would_reject_sets"])

    def test_an_exact_candidate_reports_agreement_and_publishes_no_reject_list(self):
        """三态的另一侧: 没有分歧时那个键**缺席**, 而不是 [0] 也不是空表冒充查过"""
        agg = _l2_sampling_multi(REF, max_error=1e-2)
        self.assertTrue(agg.passed, agg.message)
        self.assertNotIn("and_form_would_reject_sets", agg.metrics)
        self.assertNotIn("⚠️ 其中", agg.message)
        self.assertIn("⚠️ 1/1 组比的是采样窗口那把", agg.message)
        self.assertEqual([1], agg.metrics["absolute_only_sets"])

    def test_a_clearly_failing_candidate_agrees_on_reject(self):
        r = _l2_sampling(REF * 0.5, max_error=1e-2)
        self.assertFalse(r.passed)
        self.assertTrue(r.metrics["and_form_passed"] is False)
        self.assertNotIn("⚠️ 分歧", r.message, "两把尺都判不过时不许报警")

    def test_the_note_carries_the_thresholds_it_compared_against(self):
        r = _l2_sampling(self.SCALED, max_error=1e-2)
        self.assertIn("atol=1.0e-02", r.message)
        self.assertIn("atol=1.0e-04", r.message)

    def test_the_verdict_of_this_slice_is_observation_only(self):
        """把「判决未换尺」钉成一条会红的用例: 换成同一把尺那天它必须红, 逼那次改动重写它"""
        self.assertTrue(_l2_sampling(self.SCALED, max_error=1e-2).passed)
        self.assertTrue(_l2_sampling(self.SCALED, max_error=1e-2).metrics["and_form_passed"]
                        is False)


def _full_path_result(multi=False):
    """AND 那一条的一份真结果 (逐组或聚合) —— 判别键的对照侧由产它的那条路给出"""
    def execute(kernel_code, inputs):
        return ExecutionResult(success=True, output=REF.copy())

    v = Verifier(DeclaredFakeBackend(caps=(Cap.FULL_OUTPUT_TENSOR,), execute=execute))
    if multi:
        with contextlib.redirect_stdout(io.StringIO()):
            return v._verify_function(CODE, [{"A": np.ones((8, 8), np.float32)}], REF,
                                      _task(1e-2))
    with contextlib.redirect_stdout(io.StringIO()):
        return v._verify_single_set(CODE, {"A": np.ones((8, 8), np.float32)}, REF, _task(1e-2))


def _full_round_metrics():
    """全量那条的一份真实读数 —— 对照用的另一把尺**由产它的那条路给出**, 不是抄的

    这里不用 MockGPU: 它以位置参数调 `kernel(A)`, 那份核要两个输入会直接死在执行侧
    (`missing 1 required positional argument: 'B'`), 于是拿到的是执行失败的 **空 metrics**,
    而断言会退化成「在比两份都没数的东西」。换成一个声明了 `FULL_OUTPUT_TENSOR`、
    回显正确张量的后端假件。
    """
    def execute(kernel_code, inputs):
        return ExecutionResult(success=True, output=REF.copy())

    v = Verifier(DeclaredFakeBackend(caps=(Cap.FULL_OUTPUT_TENSOR,), execute=execute))
    with contextlib.redirect_stdout(io.StringIO()):
        r = v._verify_single_set(CODE, {"A": np.ones((8, 8), np.float32)}, REF, _task(1e-2))
    assert r.passed, r.message
    return r.metrics


def _full_round_atol():
    return _full_round_metrics()["atol"]


if __name__ == "__main__":
    unittest.main()
