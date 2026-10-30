"""通过那一侧的 L2 聚合不许用默认值造证据 (#81)

现场 (doc 57 §三 第 3 条，`benchmarks/probes/probe_l2_sampling_gate.py` 臂 0b 打出来的读数)：

    LocalExecutor (空 Cap 集): atol=— 没这个键  matched_ratio=1.00e+00  max_rel=0.00e+00  bad=0

`matched_ratio=1.0` 与 `bad_elements=0` 长得完全合理，而这条路**从没算过它们** —— 它判的是
`max_error <= task.constraints.max_error`（纯绝对、不查表、没有相对项）。三个量是聚合处兜出来的：

    max_rel_error_overall = max(max_rel_error_overall, m.get("max_rel_error", 0.0))
    total_bad += m.get("bad_elements", 0)
    min_matched_ratio = min(min_matched_ratio, m.get("matched_ratio", 1.0))

三处细节让它比「一个多余的键」贵：
  ① 只发生在**通过**那一侧（失败时提前 `return` 原始 metrics）⇒ 被污染的正好是通过率的分母；
  ② 兜出来的值与真值同形，从 `result.json` 读不出两条比对路的区别；
  ③ 就算每组都报了，产物里也没写「这个 min 底下是几组」—— 第二层压平。

⚠️ 一句探针自己的学费（CHECKLOG I21）：探针打表最初也用 `.get(k, 默认)`，于是同一个
`matched=1.00e+00` 分不清是被测代码算的、`verifier` 兜的还是**探针自己**兜的 —— 差点把账记错一层。
改成缺键印 `— 没这个键` 之后才归因到 `verifier`。这一族用例因此两侧都要有：既断「没测的键不出现」，
也断「测过的键必须出现」（否则「不发布」这条规则会退化成「什么都不发布」而全绿）。

与 #87 片 1（`single_ruler`）是同一条纪律的两半：那半问「几组用的是不是同一把尺」，这半问
「聚合值底下是几组」。规则三条，各对应一种已经犯过的读法，写在
`measurement.aggregate_l2_readings` 的 docstring 里。**这片不改判决**：`passed` 仍由每组自己
决定，聚合只决定产物里印什么。

变异标定 (每条只改一处，跑完立刻逐字节复原并断言 sha256 与改前一致。2026-09-30 实跑，
基线 11 passed，**11/11 全部拦住、无逃逸**；A 组打在 `measurement.py`，V 组打在 `verifier.py`)：
    A1 把没测当 0 (`vals` 那行换成 `s.get(key, 0)`)             → 6 红
    A2 把没测当完美 (同一行换成 `s.get(key, 1.0)`)               → 6 红  ← #81 的旧形状本身
    A3 出处谎报成「全都报过」(`[len(vals), total]` → `[total, total]`) → 5 红
    A4 一个组都没报也发布值 (`if vals:` → `if True:`)             → 7 红
    A5 note 永不报警 (`if not missing and not partial:` → `if True:`) → 5 红
    A6 note 总是报警 (同一处 → `if False:`)                      → 3 红  ← I27 的反向变异
    A7 空输入不特判 (`if not total:` → `if False:`，于是印出 0/0 组) → 1 红
    V1 出处不进 metrics (verifier 那侧 `if agg_note:` → `if False:`) → 3 红
    V2 出处总是进 metrics (同一处 → `if True:`，全都报过时也在刷字)   → 1 红
    V3 message 无条件念 matched_ratio (`in agg_values` → `if True:`) → 3 红
    V4 收集逐组读数那行不干活 (`sets_with_reference.append(m)` → `pass`) → 6 红
A6 与 V2 是刻意配的两条**反向变异**：能杀掉它们的只有
`test_real_mockgpu_multiset_publishes_everything_it_computed` 与
`test_full_reporters_aggregate_by_each_operator` 那两条正向对照 —— 少了它们，「把守卫改成总是
报警」这种退化就没有证人 (与 #87 片 1 的 M8 同一条道理：判据要有两侧，用例也要有能杀掉它的变异)。

⚠️ 一处**没有读者**的东西要写清楚，不留白：`metrics["aggregated_over"]` 今天只有 message 一个读者
(打印 + 留在 `VerificationResult` 对象里)，它**没有**穿过落盘边界 —— `result.json` 里既没有
`verification_results` 的 metrics，血缘那一份也只从 L2 取尺的三个键
(`measurement.l2_ruler_reading`)。接读者与 #87 剩下的「`ref_oracle` 没穿过落盘边界」是同一片地，
记在 doc 57 §六。

运行: PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest tests/unit/test_l2_aggregate_evidence.py -q
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent import MockGPU, Verifier
from operator_agent.backend_caps import Cap
from operator_agent.measurement import (L2_AGGREGATIONS, aggregate_l2_readings,
                                        aggregate_note)
from operator_agent.remote_executor import LocalExecutor
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType,
                                    VerificationLevel)

CORRECT = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"

#: 只有 AND 那条比对路会算的三个量 (纯绝对那条一个都不算)
AND_ONLY = ("matched_ratio", "bad_elements", "max_rel_error")
#: 两条路都算、因此不该出现在「没测过」名单里的两个量
BOTH_PATHS = ("max_error", "compared_elements", "coverage")


def _set(**keys) -> dict:
    """一份「某一组的 L2 读数」: 只带调用方点名的那些键

    基线键 (`max_error` 等) 由调用方自己点名，这里不兜底 —— 兜底就是这片要抓的那个错。
    """
    return dict(keys)


def _task(max_error=1e-4):
    return TaskSchema(task_id="agg-evidence", task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
                      target=TargetSpec(hardware="mock_gpu", dsl="python"),
                      constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0,
                                                 max_error=max_error))


def _inputs(scale=1.0, n=8, seed=11):
    rng = np.random.default_rng(seed)
    return {"A": (rng.standard_normal((n, n)) * scale).astype(np.float32),
            "B": (rng.standard_normal((n, n)) * scale).astype(np.float32)}


class TestAggregationRule(unittest.TestCase):
    """规则本身 (纯函数): 什么条件下才配发布一个聚合值"""

    def test_no_sets_at_all_publish_nothing_and_complain_nothing(self):
        """M7: 一组都没进来时不写「0/0 组算过」这种噪音

        「没有组」与「有组但没报这个键」是两件事 —— 后者才是这片要报警的对象。
        """
        values, origin = aggregate_l2_readings([])
        self.assertEqual({}, values)
        self.assertEqual({}, origin)
        self.assertEqual("", aggregate_note(origin))

    def test_full_reporters_aggregate_by_each_operator(self):
        sets = [_set(max_error=1e-6, max_rel_error=2e-3, matched_ratio=0.999,
                     bad_elements=3, compared_elements=100, coverage=1.0),
                _set(max_error=5e-5, max_rel_error=1e-4, matched_ratio=0.995,
                     bad_elements=7, compared_elements=200, coverage=0.5)]
        values, origin = aggregate_l2_readings(sets)
        self.assertEqual(5e-5, values["max_error"], "误差取最差的一组")
        self.assertEqual(2e-3, values["max_rel_error"])
        self.assertEqual(0.995, values["matched_ratio"], "比率取最差的一组")
        self.assertEqual(10, values["bad_elements"], "坏点求和")
        self.assertEqual(300, values["compared_elements"])
        self.assertEqual(0.5, values["coverage"])
        for key, _op in L2_AGGREGATIONS:
            self.assertEqual([2, 2], origin[key], f"{key} 的出处应当是 2/2 组")
        self.assertEqual("", aggregate_note(origin),
                         "全都报过时不携带信息, 不该报警")

    def test_a_key_nobody_computed_is_absent_rather_than_zero(self):
        """**这一条是 #81 的正身**: 纯绝对那条走了三组, 聚合里仍不该有 matched_ratio"""
        sets = [_set(max_error=1e-6, compared_elements=64, coverage=0.25)] * 3
        values, origin = aggregate_l2_readings(sets)
        for key in AND_ONLY:
            self.assertNotIn(key, values, f"{key} 一条都没算过, 却被发布了出来")
        self.assertEqual([0, 3], origin["matched_ratio"])
        self.assertEqual(192, values["compared_elements"],
                         "这条路真算过的量不许被顺手丢掉")
        note = aggregate_note(origin)
        self.assertIn("没有任何一组算过", note)
        self.assertIn("matched_ratio", note)
        for sentence in ("matched_ratio=1.0", "bad_elements=0"):
            self.assertNotIn(sentence, note,
                             f"报警的内容自己造了证据 ({sentence}) —— 缺证据不等于零")

    def test_partial_reporters_fold_only_over_reporters_and_say_so(self):
        """2/3 组报过: 值是那 2 组的 min, 而「2/3」必须一起出产物

        旧写法把第三组当 1.0 参与 min —— 于是「全部对上了」被算进一个根本没比过的组里。
        """
        sets = [_set(matched_ratio=0.999), _set(matched_ratio=0.98), _set(max_error=1e-6)]
        values, origin = aggregate_l2_readings(sets)
        self.assertEqual(0.98, values["matched_ratio"],
                         "min 只许在那 2 组上取; 兜底会让它变成 0.98 或 1.0 里更诱的那个")
        self.assertEqual([2, 3], origin["matched_ratio"])
        note = aggregate_note(origin)
        self.assertIn("部分组", note)
        self.assertIn("matched_ratio 2/3 组", note)

    def test_origin_uses_lists_because_it_crosses_the_json_boundary(self):
        """出处要进 `result.json`: tuple 会被 json 悄悄变成 list, 读侧两种形状都要防 (I29 同族)"""
        _values, origin = aggregate_l2_readings([_set(max_error=1e-6)])
        self.assertTrue(all(isinstance(v, list) for v in origin.values()))
        self.assertEqual(origin, json.loads(json.dumps(origin)),
                         "过一遍 json 之后读数不该变形状")


class _ScriptedRemote:
    """一个「远端旧版 run_kernel.py 逐键少报」的假后端: 走真实的 REMOTE_VERIFY 那一路

    ⚠️ 为什么不 monkeypatch `_verify_single_set`: 混合覆盖这一侧**今天真能发生**而不需要谁出错
    (载荷按 `_REMOTE_L2_METRIC_KEYS` 的 `if k in remote_verify` 逐键过滤), 所以这里只发放
    「远端回什么载荷」, 分组、过滤、聚合、message 拼接全部走真代码。
    """

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.name = "ScriptedRemote"
        self._impl = MockGPU()

    @property
    def capabilities(self):
        return frozenset({Cap.REMOTE_VERIFY})

    @property
    def label(self):
        return self.name

    def __getattr__(self, item):
        return getattr(self._impl, item)

    def verify_function(self, kernel_code, inputs, kernel_type, max_error,
                        precision="", reference_source=""):
        return self._payloads.pop(0)


class TestThroughRealVerifier(unittest.TestCase):
    """走真实的 `_verify_function`: 发布/不发布之外, 还要「读的人看得见为什么」"""

    def _aggregate(self, backend, per_set):
        v = Verifier(backend)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            r = v._verify_function(CORRECT, [_inputs() for _ in per_set], None, _task())
        return r, buf.getvalue()

    def test_real_mockgpu_multiset_publishes_everything_it_computed(self):
        """正向对照 (**不发放**, 真 AND 路径): 全都算过时照旧全发布, 且不发布出处

        删掉「逐键只 fold 报过的组」这条规则时这一条也会红 —— 它测的是守卫不许把主路径判成缺证据。
        """
        r, log = self._aggregate(MockGPU(), [1, 2])
        self.assertTrue(r.passed, r.message)
        for key in AND_ONLY + BOTH_PATHS:
            self.assertIn(key, r.metrics, f"聚合把算过的 {key} 弄丢了")
        self.assertNotIn("aggregated_over", r.metrics,
                         "全都报过还发布出处, 那一栏只是在刷字数")
        self.assertNotIn("没有任何一组算过", r.message)
        self.assertIn("matched_ratio=1.000000", r.message)
        self.assertIn("L2 PASSED", log)
        self.assertEqual(2, r.metrics["num_input_sets"])

    def test_real_local_executor_run_publishes_none_of_the_three(self):
        """#81 的正身走真路径: `LocalExecutor` 声明空 Cap 集 ⇒ 那条路从没算过这三个量"""
        r, log = self._aggregate(LocalExecutor(), [1, 2])
        self.assertTrue(r.passed, r.message)
        for key in AND_ONLY:
            self.assertNotIn(key, r.metrics,
                             f"{key} 是聚合兜出来的: 这条路只判「绝对误差 <= 任务那个数」")
        self.assertEqual([0, 2], r.metrics["aggregated_over"]["matched_ratio"])
        self.assertIn("没有任何一组算过", r.message)
        self.assertIn("matched_ratio", r.message)
        self.assertIn("L2 PASSED", log)

    def test_older_remote_payload_missing_one_key_reads_as_partial(self):
        """混合覆盖走真实载荷过滤: 一组报了、一组没报 ⇒ 值是报了那组的, 出处写 1/2"""
        full = {"success": True, "max_error": 1e-6, "max_rel_error": 2e-3,
                "matched_ratio": 0.999, "bad_elements": 1,
                "compared_elements": 64, "coverage": 1.0,
                "atol": 1e-4, "rtol": 1e-4, "required_matched_ratio": 0.99}
        older = {"success": True, "max_error": 4e-6, "compared_elements": 64,
                 "coverage": 1.0, "atol": 1e-4, "rtol": 1e-4,
                 "required_matched_ratio": 0.99}  # 旧版 run_kernel.py 少报三个量
        r, _log = self._aggregate(_ScriptedRemote([full, older]), [1, 2])
        self.assertTrue(r.passed, r.message)
        self.assertEqual(0.999, r.metrics["matched_ratio"],
                         "min 应当只落在报了的那一组, 而不是被 1.0 兜底改写成别的")
        self.assertEqual([1, 2], r.metrics["aggregated_over"]["matched_ratio"])
        self.assertEqual([1, 2], r.metrics["aggregated_over"]["max_rel_error"])
        self.assertEqual(2e-3, r.metrics["max_rel_error"],
                         "这个量只有 `full` 那一组算了 ⇒ 聚合就是它那一个数, 不许再兜底")
        self.assertIn("只从部分组聚出来", r.message)
        self.assertIn("matched_ratio 1/2 组", r.message)

    def test_set_without_a_reference_still_counts_in_num_input_sets(self):
        """「比了 3 组」与「3 组都有参考可比」是两件事 —— 两个数都要在产物里

        旧版 `run_kernel.py` 连 `max_error` 都不回时, 那一组连聚合都不参与; 以前这件事
        与「聚合值」混在一起看不出来, 现在 `aggregated_over` 的分母是**参与聚合的组数**,
        而 `num_input_sets` 仍是全部组数。
        """
        with_ref = {"success": True, "max_error": 1e-6, "matched_ratio": 0.99,
                    "bad_elements": 0, "compared_elements": 64, "coverage": 1.0}
        no_ref = {"success": True, "compared_elements": 0, "coverage": 1.0}
        r, _log = self._aggregate(_ScriptedRemote([with_ref, no_ref]), [1, 2])
        self.assertTrue(r.passed, r.message)
        self.assertEqual(2, r.metrics["num_input_sets"])
        self.assertEqual(1, r.metrics["aggregated_over"]["matched_ratio"][1],
                         "出处里的分母必须是参与聚合的组数, 不是全部组数")
        self.assertEqual(0.99, r.metrics["matched_ratio"])

    def test_publication_does_not_move_the_verdict(self):
        """观测先于判定: 三个量发不发布, 判决一位不动"""
        for backend in (MockGPU(), LocalExecutor()):
            r, _log = self._aggregate(backend, [1, 2])
            self.assertTrue(r.passed,
                            f"{backend.label} 那条路应当仍然通过: {r.message}")
            self.assertEqual(VerificationLevel.L2_FUNCTION, r.level)

    def test_the_invented_perfect_value_is_gone_from_both_exits(self):
        """两个出口都不许留那个 1.0: metrics 与 message

        `assertNotIn("matched_ratio=1.000000", message)` 是这片最直接的读者侧证 —— 旧写法在
        纯绝对那条上每次都印它, 而它今天仍然印在 AND 那条上, 所以只能拿「同一条路」来对照。
        """
        r, _log = self._aggregate(LocalExecutor(), [1])
        self.assertNotIn("matched_ratio=1.000000", r.message)
        self.assertNotIn("matched_ratio", r.metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
