"""L2 聚合的「同一把尺」守卫 —— AReaL-TIK 行 150 那条 𝒮 immutable 落进 Verifier 的第一片

外部一手 (arXiv:2609.35140v1 行 150, 逐字):
    "𝒮 is the **immutable** evaluation specification ... It keeps correctness and
     performance criteria fixed across attempts."

改之前的形状 (`verifier.py` 聚合处):

    gate_params = {}
    ...
    if not gate_params:
        gate_params = {k: m[k] for k in ("atol", "rtol", "required_matched_ratio") if k in m}

也就是**第一个报了值的输入组代表全部组**, 而它下面聚合的是跨组的量 (`max_error` 取 max、
`matched_ratio` 取 min、`bad_elements` 求和)。于是产物里印着「atol=…」, 而那个聚合值其实可能
是几把不同的尺各自判过之后再混起来的 —— `benchmarks/probes/probe_l2_sampling_gate.py` 臂 0b 正是要拿
这几个键核「绝对+相对」那条门的, 读的人会把一个不存在的统一口径当成前提。

⚠️ 一条要说清楚的实话: **今天从内置的两条比对路径走, 这个冲突构造不出来** —— 一次
`_verify_function` 只有一个 `task`, 而 AND 那条的 `atol/rtol/rho` 全来自
`tolerance.effective_atol(precision, max_allowed)`, 组与组之间必然同值。所以这一片今天的
行为与改之前完全相同 (探针输出逐字未变)。它钉的是一条**不变量**, 不是一只在场的 bug。

那它凭什么不是空转? 因为「某一组报了、某一组没报」这一侧**今天真能发生**, 而且不需要谁出错:
远端那路的载荷是按 `_REMOTE_L2_METRIC_KEYS` **逐键过滤** (`if k in remote_verify`) 透传上来的,
所以只要部署在远端的 `run_kernel.py` 是旧版本、或者某组走了缺字段的分支, 就会出现「A 组报了
`required_matched_ratio`、B 组没报」。旧写法会从 A 组那一份把这三个键全部 publish 出去,
仿佛 B 组也在同一把尺下判过; 新写法那个键**直接不发布**, 并把每组的实际取值摊进
`metrics.gate_params_conflict`。这与 #64「分母不可信时 `bound_type` 不许替它判」同形,
也与本仓「缺证据不是通过」那条老纪律同族。

跨轮/跨 attempt 的那一半 (同一份 task 在第 1 轮与第 5 轮用的尺是否还是同一把) 不在本片,
记在 #87 的第二片 —— 那一半的现场是部署漂移, 与 #89 的 runtime drift gate 是同一条。

变异标定 (每条只改一处, 跑完立刻逐字节复原并断言 sha256 与改前一致。2026-09-30 实跑,
基线 11 passed, **8/8 全部拦住, 无逃逸**):
    M1 退回「第一组赢」(调用点直接取 `gates[0]`)             → 2 条红
    M2 缺组当同值 (`len(vals) == len(gates)` → `> 0`)        → 2 条红  ← 本片真正的牙齿
    M3 冲突时挑一个值发布 (去掉「同值」那半句)                 → 2 条红
    M4 冲突时不写进 message (只留 metrics)                    → 1 条红
    M5 空输入也造一把默认尺 (`return {"atol": 1e-5}, {}`)      → 1 条红
    M6 判据改成全有或全无 (任一键冲突则三键都不发布)             → 3 条红
    M7 去掉「谁都没报过」那一支 (`if not vals:` → `if False`)  → 3 条红  ← 实现时自己撞出来的
    M8 守卫总是报警 (`if <判据>:` → `if False:`)               → 8 条红
M8 是唯一能杀掉 `test_real_mockgpu_multiset_publishes_the_ruler_it_used` 的那条: 正向对照在
M1~M7 下全绿是**应该的** (它测的是守卫不许过度报警, 不是不许报警), 但一条任何变异都杀不掉的
用例等于没写, 所以要补一条「反向的变异」来给它派活 —— 这与 I19/I23「两侧取样」是同一句话的
两种用法: 判据要有两侧, 用例也要有能杀掉它的那个变异。
纯函数那 6 条不依赖任何变异就能核住规则本身; 集成侧的用例走真实的 `_verify_function`
聚合, 只有「每组报什么」是脚本发放的 (发放理由见上面那段实话)。
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent import MockGPU, Verifier
from operator_agent.core.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity,
    TaskSchema, TaskType, VerificationLevel, VerificationResult,
)
from operator_agent.decide.measurement import GATE_PARAM_KEYS, single_ruler

CORRECT_KERNEL = """import numpy as np

def kernel(A, B):
    return A @ B
"""

RULER_KEYS = set(GATE_PARAM_KEYS)


def _task(max_error=1e-5, precision=Precision.FP32):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=precision, min_speedup=0.0,
                                   max_error=max_error),
    )


def _inputs(scale=1.0, n=8):
    rng = np.random.default_rng(11)
    return {"A": (rng.standard_normal((n, n)) * scale).astype(np.float32),
            "B": (rng.standard_normal((n, n)) * scale).astype(np.float32)}


def _passing_set(**gate):
    """脚本发放的「某一组的 L2 结果」: 通过, 且只带调用方点名的那几把尺

    ⚠️ 为什么这里要发放而不是用真实后端: 内置的 AND 路径对同一份 task 的每一组必然报同一把尺
    (见文件头那段实话), 所以「各组的尺不同 / 某一组缺一个键」这一侧在真实路径上构造不出来。
    发放只替换**每组的读数**, 循环、聚合、`single_ruler`、message 拼接全部走真代码。
    """
    m = {"max_error": 1e-7, "matched_ratio": 1.0, "bad_elements": 0,
         "compared_elements": 64, "coverage": 1.0}
    m.update(gate)
    return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=True,
                              message="脚本发放的通过", metrics=m)


def _scripted_aggregate(verifier, per_set_gates):
    """跑真实的 `_verify_function` 聚合, 只把「每组报什么尺」换成脚本发放的那份

    组数与 `per_set_gates` 一一对应 —— 发放多于/少于真实组数都会让这条用例核的不是聚合。
    """
    results = [_passing_set(**g) for g in per_set_gates]
    orig = verifier._verify_single_set
    verifier._verify_single_set = lambda *a, **k: results.pop(0)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            r = verifier._verify_function(
                CORRECT_KERNEL, [_inputs() for _ in per_set_gates], None, _task())
    finally:
        verifier._verify_single_set = orig
    return r, buf.getvalue()


class TestSingleRulerRule(unittest.TestCase):
    """规则本身 (纯函数): 什么条件下才配发布一个单值"""

    def test_no_comparisons_publish_no_ruler(self):
        """M5: 一组都没比对时不许造一把默认的尺

        「没有证据」与「证据说尺是 1e-5」是两件事 —— 后者是 `m.get(k, 默认)` 造证据,
        本仓 #81 那条老毛病正是这个形状。
        """
        single, conflict = single_ruler([])
        self.assertEqual({}, single)
        self.assertEqual({}, conflict,
                         "没有任何组时也没有哪一组互相矛盾, 不该报冲突")

    def test_all_sets_agree_publishes_single_values(self):
        gates = [{"atol": 1e-5, "rtol": 1e-4, "required_matched_ratio": 0.99},
                 {"atol": 1e-5, "rtol": 1e-4, "required_matched_ratio": 0.99}]
        single, conflict = single_ruler(gates)
        self.assertEqual({"atol": 1e-5, "rtol": 1e-4, "required_matched_ratio": 0.99},
                         single)
        self.assertEqual({}, conflict)

    def test_one_set_publishes_itself(self):
        """缺席侧的一半: 只有一组时它就是全部, 不许因为「凑不齐多组」而判冲突"""
        single, conflict = single_ruler([{"atol": 2e-5, "rtol": 1e-4}])
        self.assertEqual({"atol": 2e-5, "rtol": 1e-4}, single)
        self.assertEqual({}, conflict)

    def test_differing_value_is_not_published_and_is_reported(self):
        gates = [{"atol": 1e-5, "rtol": 1e-4}, {"atol": 5e-2, "rtol": 1e-4}]
        single, conflict = single_ruler(gates)
        self.assertNotIn("atol", single, "两把不同的尺折成了一个单值")
        self.assertIn("atol", conflict)
        self.assertEqual({0: 1e-5, 1: 5e-2}, conflict["atol"],
                         "冲突要摊到「哪一组用的哪一把」, 读的人不必重跑")
        self.assertEqual({"rtol": 1e-4}, single,
                         "M6: 判据是逐键的, 不能一个键打架就让别的键陪着不发布")

    def test_a_set_that_omitted_the_key_counts_as_conflict(self):
        """本片真正的牙齿 (M2): 「某一组没报」不等于「某一组同值」

        远端那路是按 `_REMOTE_L2_METRIC_KEYS` 逐键过滤透传载荷的, 所以缺键今天就能发生
        (远端跑的是旧版脚本、或某组走了缺字段分支)。旧写法会从报了的那一组手里把三个键
        全部 publish, 仿佛没报的那一组也在同一把尺下判过。
        """
        gates = [{"atol": 1e-5, "rtol": 1e-4, "required_matched_ratio": 0.99},
                 {"atol": 1e-5, "rtol": 1e-4}]
        single, conflict = single_ruler(gates)
        self.assertNotIn("required_matched_ratio", single,
                         "有一组根本没报这个键, 凭什么说整场用的是同一把尺")
        self.assertEqual({0: 0.99}, conflict["required_matched_ratio"],
                         "缺席也要在冲突表里留下是哪一组缺的")
        self.assertEqual({"atol": 1e-5, "rtol": 1e-4}, single)

    def test_non_hashable_gate_values_still_compare(self):
        """`repr` 而不是 `set(vals.values())`: 容差键今天是标量, 明天可能是 tuple/dict

        用不可哈希值时 `set()` 会直接 TypeError —— 那会让整个 L2 聚合炸在一个观测性
        判据上, 而这条判据没有权利炸掉主流程。
        """
        gates = [{"atol": (1e-5, 1e-6)}, {"atol": (1e-5, 1e-6)}]
        single, conflict = single_ruler(gates)
        self.assertEqual({"atol": (1e-5, 1e-6)}, single)
        self.assertEqual({}, conflict)

    def test_a_key_nobody_reported_is_neither_published_nor_flagged(self):
        """**这一条是写实现时自己撞出来的**: 第一版逐键循环没有 `if not vals: continue`

        于是「谁都没报过的键」会带着一个空证人 (`{"rtol": {}}`) 进冲突表, 而 message
        照着念「这几组用的不是同一把尺 ['rtol']」。那是拿缺证据当反证 —— 报警的内容
        恰好是它自己那条判据说不能说的话。
        """
        single, conflict = single_ruler([{"atol": 1e-5}, {"atol": 1e-5}])
        self.assertEqual({"atol": 1e-5}, single)
        self.assertEqual({}, conflict,
                         f"谁都没报过的键不该出现在冲突表里: {conflict}")
        self.assertNotIn("rtol", conflict)
        self.assertNotIn("required_matched_ratio", conflict)


class TestAggregateThroughVerifier(unittest.TestCase):
    """走真实聚合: 发布/不发布之外, 还要「读的人看得见为什么」"""

    def setUp(self):
        self.verifier = Verifier(MockGPU())

    def test_real_mockgpu_multiset_publishes_the_ruler_it_used(self):
        """正向对照 (**不发放**, 走 MockGPU 真实的 AND 路径): 同值时三个键照旧全部发布

        删掉守卫时这一条也会红 —— 它测的是「守卫不许把今天的主路径判成冲突」,
        与上面那几条一起构成 I19/I23 要求的两侧取样。
        """
        with contextlib.redirect_stdout(io.StringIO()):
            r = self.verifier._verify_function(
                CORRECT_KERNEL, [_inputs(), _inputs(scale=8.0)], None, _task())
        self.assertTrue(r.passed, r.message)
        for key in GATE_PARAM_KEYS:
            self.assertIn(key, r.metrics,
                          f"聚合把 {key} 弄丢了: 内置 AND 路径每组都报了同一把尺")
        self.assertNotIn("gate_params_conflict", r.metrics)
        self.assertNotIn("不是同一把尺", r.message)
        self.assertEqual(2, r.metrics["num_input_sets"],
                         "构造前提: 这确实是一次多组输入的聚合, 不是单组")

    def test_conflicting_rulers_publish_no_single_value(self):
        r, log = _scripted_aggregate(
            self.verifier,
            [{"atol": 1e-5, "rtol": 1e-4, "required_matched_ratio": 0.99},
             {"atol": 5e-2, "rtol": 1e-4, "required_matched_ratio": 0.99}])
        self.assertNotIn("atol", r.metrics, "两组用 1e-5 与 5e-2, 聚合却发了一把握")
        self.assertEqual({0: 1e-5, 1: 5e-2}, r.metrics["gate_params_conflict"]["atol"])
        self.assertIn("不是同一把尺", r.message, "M4: 冲突只写在 metrics 里, 读日志的人看不到")
        self.assertIn("atol", r.message)
        self.assertIn("L2 PASSED", log)

    def test_partially_reported_ruler_publishes_no_single_value(self):
        """远端逐键过滤那个现场 (缺键侧), 经过真实聚合"""
        r, _ = _scripted_aggregate(
            self.verifier,
            [{"atol": 1e-5, "rtol": 1e-4, "required_matched_ratio": 0.99},
             {"atol": 1e-5, "rtol": 1e-4}])
        self.assertIn("required_matched_ratio", r.metrics["gate_params_conflict"])
        self.assertNotIn("required_matched_ratio", r.metrics)
        self.assertEqual(1e-5, r.metrics["atol"],
                         "这一组两组的 atol 确实相同, 该说的还是要说")

    def test_aggregate_still_aggregates_across_sets(self):
        """守卫不许顺手改掉跨组聚合本身: max_error 取 max、compared 求和照旧

        这一条是给「重构成 single_ruler 时手滑改动了循环」准备的证人。
        """
        r, _ = _scripted_aggregate(self.verifier, [{}, {}, {}])
        self.assertEqual(3, r.metrics["num_input_sets"])
        self.assertEqual(192, r.metrics["compared_elements"])
        self.assertEqual(1e-7, r.metrics["max_error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
