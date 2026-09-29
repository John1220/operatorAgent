"""成功率量具的聚合逻辑离线标定

这一组不碰 GPU、不碰网络, 只用假记录验 `aggregate()` / `format_report()`。
理由是本项目反复踩的那一条 (doc 46): **判据方向写反时, 量具明明看见了信号却报「看不见」**,
而一份坏掉的报表会打印出很像样的百分数。真机跑一轮很贵, 所以聚合逻辑必须能离线验。

每条用例都尽量做成**方向检查**: 断言「两种相反的输入给出两种相反的输出」,
而不是只断言某一个具体数字 —— 后者在判据整体失效时也可能碰巧成立。
"""
import ast
import json
import unittest
from pathlib import Path

from benchmarks.harness_pass_rate import (MIN_DENOM, aggregate, cost_from_metadata,
                                          device_record, first_pass_round, format_report,
                                          record_rounds)
from operator_agent.backend_caps import Cap
from tests.backend_fakes import DeclaredFakeBackend

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID, DEGRADED, UNINFO = "valid", "degraded", "uninformative"


def _round(n, l1=True, l2=None, l3=None, validity=None, speedup=None, action="optimize",
           signature=None, duplicate_of=None, recipes=()):
    """造一轮血缘记录。l2/l3 为 None 表示「没走到这一级」(键不存在, 不是 False)

    键集合必须与 Conductor 真写进 `iteration_history` 的一致 ——
    `TestLineageIsNotWhitelisted.test_the_fixtures_cover_every_real_key`
    会去 `conductor.py` 的 AST 里核对, 所以这里少写一个键就会红。
    """
    lv = {"L1": l1}
    if l2 is not None:
        lv["L2"] = l2
    if l3 is not None:
        lv["L3"] = l3
    return {"round": n, "action": action, "all_passed": all(lv.values()),
            "levels_passed": lv, "measurement_validity": validity,
            "gflops": None, "speedup": speedup, "recipes_applied": list(recipes),
            "candidate_signature": signature, "duplicate_of": duplicate_of}


def _run(rounds, final=None, admitted=False, reason="", tokens=100, usd=0.01,
         wall=10.0, ok=True, error=None, calls=7):
    if final is None:
        final = bool(rounds) and any(r["all_passed"] for r in rounds)
    rtp, via = first_pass_round(rounds) if final else (None, None)
    return {"case": "c", "kernel_type": "matmul", "shape": [8, 8, 8],
            "ok": ok, "error": error, "rounds": rounds,
            "final_all_passed": final, "rounds_to_pass": rtp, "passed_via": via,
            "admitted": admitted, "admit_reason": reason,
            "tokens": tokens, "cost_usd": usd, "llm_calls": calls, "wall_s": wall}


def _n_passing(n):
    """n 次全部一轮通过、读数全 valid、加速比 1.5x、且都入库"""
    return [_run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.5)],
                 admitted=True) for _ in range(n)]


class TestDirectionIsRight(unittest.TestCase):
    """最要命的一类错: 报表对「全过」和「全挂」给出同一句话"""

    def test_all_pass_and_all_fail_report_opposite_rates(self):
        good = aggregate(_n_passing(MIN_DENOM))
        bad = aggregate([_run([_round(1, l1=False)], final=False) for _ in range(MIN_DENOM)])
        self.assertEqual(100.0, good["task_success_pct"])
        self.assertEqual(0.0, bad["task_success_pct"])
        self.assertNotEqual(good["task_success_pct"], bad["task_success_pct"],
                            "两种相反的输入给出同一个成功率 —— 量具是瞎的")
        self.assertEqual(100.0, good["level_pcts"]["L1"])
        self.assertEqual(0.0, bad["level_pcts"]["L1"])

    def test_report_text_differs_between_pass_and_fail(self):
        good = format_report(aggregate(_n_passing(MIN_DENOM)))
        bad = format_report(aggregate([_run([_round(1, l1=False)], final=False)
                                       for _ in range(MIN_DENOM)]))
        self.assertIn("100.0%", good)
        self.assertIn("0.0%", bad)
        self.assertNotEqual(good, bad)


class TestLevelRateDenominator(unittest.TestCase):
    """分母必须是「走到该级的轮数」, 否则 L1 的失败率会污染 L2/L3"""

    def test_rounds_that_died_at_l1_do_not_count_against_l2(self):
        rounds = [_round(1, l1=False),                      # 死在 L1
                  _round(1, l2=True, l3=True, validity=VALID),
                  _round(1, l2=False),                      # 死在 L2
                  _round(1, l2=True, l3=True, validity=VALID)]
        agg = aggregate([_run([r]) for r in rounds])
        self.assertEqual((3, 4), agg["level_rates"]["L1"], "4 轮都走到 L1, 其中 3 轮过")
        self.assertEqual((2, 3), agg["level_rates"]["L2"],
                         "L2 的分母该是「走到 L2 的 3 轮」, 不是 4 轮")
        self.assertEqual((2, 2), agg["level_rates"]["L3"],
                         "L3 的分母该是「走到 L3 的 2 轮」")

    def test_missing_level_key_is_not_read_as_failure(self):
        """键不存在 ≠ 值为 False。混在一起会把「没走到」算成「没通过」"""
        runs = [_run([_round(1, l1=False)]) for _ in range(MIN_DENOM)]
        agg = aggregate(runs)
        self.assertEqual(0, agg["level_rates"]["L2"][1], "L2 分母应为 0 (一轮都没走到)")
        self.assertIsNone(agg["level_pcts"]["L2"], "分母为 0 时不该编一个百分数出来")


class TestSmallDenominatorRefusesPercentage(unittest.TestCase):
    def test_below_min_denom_does_not_print_a_rate(self):
        agg = aggregate(_n_passing(MIN_DENOM - 1))
        self.assertFalse(agg["denominator_ok"])
        text = format_report(agg)
        self.assertIn("分母不足", text)
        self.assertNotIn("任务级成功】0/", text)
        self.assertNotRegex(text, r"任务级成功】\d+/\d+ = \d")

    def test_at_min_denom_does_print(self):
        agg = aggregate(_n_passing(MIN_DENOM))
        self.assertTrue(agg["denominator_ok"])
        self.assertIn("任务级成功", format_report(agg))


class TestDecisionGradeIsSeparateFromPassRate(unittest.TestCase):
    """通过率与「读数可不可信」必须分开报 —— 这是这份量具存在的理由之一"""

    def test_tiers_are_counted_and_valid_only_is_decision_grade(self):
        runs = [
            _run([_round(1, l2=True, l3=True, validity=VALID, speedup=2.0)]),
            _run([_round(1, l2=True, l3=True, validity=DEGRADED, speedup=1.2)]),
            _run([_round(1, l2=True, l3=True, validity=UNINFO, speedup=0.4)]),
            _run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.8)]),
        ]
        agg = aggregate(runs)
        self.assertEqual({VALID: 2, DEGRADED: 1, UNINFO: 1}, agg["validity_tiers"])
        self.assertEqual((2, 4), agg["decision_grade"])
        self.assertEqual(50.0, agg["decision_grade_pct"])
        # 全部 4 轮都「通过」了, 但只有一半的读数有资格驱动闭环
        self.assertEqual(100.0, agg["task_success_pct"])
        self.assertIn("一起读", format_report(agg),
                      "报表必须提醒这两个数不能分开引用")

    def test_speedup_is_split_by_grade_not_merged(self):
        runs = [
            _run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.1)]),
            _run([_round(1, l2=True, l3=True, validity=UNINFO, speedup=9.9)]),
            _run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.3)]),
            _run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.2)]),
        ]
        agg = aggregate(runs)
        self.assertEqual(4, agg["speedup_all"]["n"])
        self.assertEqual(3, agg["speedup_decision_grade"]["n"],
                         "uninformative 那轮的 9.9x 不该混进成绩口径")
        self.assertLess(agg["speedup_decision_grade"]["median"], 2.0)
        self.assertGreater(agg["speedup_all"]["max"], 9.0)


class TestCrashesAndVacuousTruth(unittest.TestCase):
    def test_crashed_runs_are_counted_but_not_in_the_success_denominator(self):
        runs = _n_passing(2) + [_run([], ok=False, error="Timeout: x")] * 2
        agg = aggregate(runs)
        self.assertEqual(4, agg["runs"])
        self.assertEqual(2, agg["runs_ok"])
        self.assertEqual(2, agg["runs_crashed"])
        self.assertEqual((2, 2), agg["task_success"],
                         "崩溃样本不该混进成功率的分母, 但要单独计数")
        self.assertIn("Timeout: x", agg["crash_reasons"])

    def test_a_run_with_no_rounds_is_not_a_success(self):
        """`all([])` 是 True: 一份没有验证结果的产物会被 all_passed 判成「全过」。
        量具自己也得守住这一条, 不能只依赖 Conductor 的早退路径填了合成失败。"""
        agg = aggregate([_run([], final=False) for _ in range(MIN_DENOM)])
        self.assertEqual(0.0, agg["task_success_pct"])
        self.assertEqual(0, agg["rounds"])


class TestRoundsToPassDoesNotDropSamples(unittest.TestCase):
    """真机第 4 次运行抓到的: 通过的那一轮是 `fix` 轮时, rounds_to_pass 曾是 None

    于是这个样本被 rounds-to-pass 的分位数静默丢掉 —— 而它是**成功**样本,
    丢的正好是最该计入的那一类。指标悄悄少算样本比算错更难发现。
    """

    def test_a_passing_fix_round_still_counts(self):
        rounds = [_round(1, l2=False, action="generate"),
                  _round(2, l2=True, l3=True, validity=VALID, action="fix")]
        self.assertEqual((2, "fix"), first_pass_round(rounds))

    def test_first_pass_wins_over_later_rounds(self):
        rounds = [_round(1, l2=False), _round(2, l2=True, l3=True), _round(3, l2=True, l3=True)]
        self.assertEqual((2, "optimize"), first_pass_round(rounds))

    def test_never_passing_gives_none(self):
        self.assertEqual((None, None), first_pass_round([_round(1, l1=False), _round(2, l1=False)]))
        self.assertEqual((None, None), first_pass_round([]))

    def test_aggregate_keeps_the_fix_round_sample_in_the_quantiles(self):
        """端到端那一步: 修好的样本必须出现在 rounds_to_pass 的分位数里"""
        runs = [_run([_round(1, l2=False), _round(2, l2=True, l3=True, validity=VALID)])
                for _ in range(MIN_DENOM)]
        for r in runs:                       # 模拟真机那条: 通过发生在第 2 轮
            r["rounds_to_pass"], _ = first_pass_round(r["rounds"])
        agg = aggregate(runs)
        self.assertIsNotNone(agg["rounds_to_pass"], "成功样本全被丢掉了")
        self.assertEqual(MIN_DENOM, agg["rounds_to_pass"]["n"])
        self.assertEqual(2, agg["rounds_to_pass"]["median"])


class TestCostKeysMatchTheRealTracker(unittest.TestCase):
    """成本那一行曾经安静地全报 $0.0: 读的是 `cost_usd`, 真键名是 `total_cost_usd`

    540114 tokens 配 0 美元, 看起来像「自建端点不要钱」, 实际上是键名读错。
    这类错不会让任何断言变红 —— 所以守卫直接对着**真的** `CostTracker.summary()` 查键。
    """

    def test_the_keys_we_read_exist_in_the_real_summary(self):
        from operator_agent.llm_client import CostTracker
        keys = set(CostTracker().summary())
        for k in ("total_tokens", "total_cost_usd", "total_calls"):
            self.assertIn(k, keys, f"账单摘要里没有 {k} —— 量具读到的会是默认值 0")

    def test_cost_is_extracted_not_defaulted_to_zero(self):
        meta = {"llm_cost": {"total_calls": 7, "total_tokens": 1234,
                             "total_cost_usd": 0.5678}}
        self.assertEqual((1234, 0.5678, 7), cost_from_metadata(meta))

    def test_missing_metadata_is_zero_but_not_a_crash(self):
        self.assertEqual((0, 0.0, 0), cost_from_metadata(None))
        self.assertEqual((0, 0.0, 0), cost_from_metadata({}))


class TestCostIsPerRunNotCumulative(unittest.TestCase):
    """第一版把**累计**账单当成了每次运行的成本, 高估 3.25 倍

    `CostTracker` 挂在 LLMClient 上累加, 而 `main()` 只建了一个 client 传给所有运行,
    于是 `metadata["llm_cost"]` 是「从头到现在」的总量: 六次运行的 token 数成了
    15638 → 34457 → 78801 → 101156 → 143717 → 166345 一条单调递增曲线,
    报表再把这些累计值求和得到 540114, 而真实总量是 166345。
    **每一条单独看都像合理的数**, 所以只有跨运行看单调性才暴露。

    修法是 run_one 内部自建 client。这条守卫钉的就是「不许再把 client 从外面传进来」——
    参数一旦回来, 累加就会跟着回来。
    """

    def test_run_one_does_not_accept_a_shared_llm(self):
        import inspect
        from benchmarks.harness_pass_rate import run_one
        params = list(inspect.signature(run_one).parameters)
        self.assertNotIn("llm", params,
                         f"run_one 又接收共享 client 了, 账单会跨运行累加: {params}")

    def test_cumulative_cost_would_show_up_as_monotonic_tokens(self):
        """反向标定: 让聚合看见「单调递增的 token」这种形状, 确认它不是我们想要的

        这不是测 aggregate 的正确性, 是把「累计」这个失效模式写成一个可执行的样子,
        免得下次有人重新引入时看不出区别。
        """
        runs = _n_passing(MIN_DENOM)
        cumulative = [15638, 34457, 78801, 101156]
        for r, t in zip(runs, cumulative):
            r["tokens"] = t
        agg = aggregate(runs)
        self.assertEqual(sum(cumulative), agg["cost"]["tokens"],
                         "聚合只做求和, 所以喂累计值就会得到「累计值之和」—— 这正是第一版的错")
        self.assertNotEqual(cumulative[-1], agg["cost"]["tokens"])


class TestAdmissionAndCost(unittest.TestCase):
    def test_admission_rate_is_over_passing_runs_only(self):
        runs = [_run([_round(1, l2=True, l3=True, validity=VALID)], admitted=True),
                _run([_round(1, l2=True, l3=True, validity=UNINFO)], admitted=False,
                     reason="测量不可信 (can_gate=False)"),
                _run([_round(1, l2=True, l3=True, validity=VALID)], admitted=True),
                _run([_round(1, l2=True, l3=True, validity=VALID)], admitted=True)]
        agg = aggregate(runs)
        self.assertEqual((3, 4), agg["admitted"])
        self.assertEqual(75.0, agg["admitted_pct"])
        self.assertIn("测量不可信", agg["admit_rejections"][0])

    def test_cost_is_summed(self):
        agg = aggregate(_n_passing(MIN_DENOM))
        self.assertEqual(100 * MIN_DENOM, agg["cost"]["tokens"])
        self.assertEqual(round(0.01 * MIN_DENOM, 4), agg["cost"]["usd"])

    def test_llm_calls_are_summed_too(self):
        """`llm_calls` 一度只落盘不进报表 —— 与 doc 37 的 `llm_cost` 同类「算了没人读」"""
        agg = aggregate(_n_passing(MIN_DENOM))
        self.assertEqual(7 * MIN_DENOM, agg["cost"]["calls"])
        self.assertIn(str(7 * MIN_DENOM), format_report(agg))


def _dict_keys_in(source, required):
    """在 source 的 AST 里找出「键集合包含 required 的那个字典字面量」, 返回它的键集合们

    用来把测试里的假记录**钉在真代码上**: 手抄一遍键名, 抄的那一刻就准, 之后就开始漂。
    """
    hits = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if required <= keys:
            hits.append(keys)
    return hits


def _real_lineage_keys():
    src = (REPO_ROOT / "operator_agent" / "conductor.py").read_text(encoding="utf-8")
    hits = _dict_keys_in(src, {"round", "action", "all_passed"})
    if len(hits) != 1:
        raise AssertionError(f"conductor 里认出 {len(hits)} 处血缘字典, 这条守卫的前提变了")
    return hits[0]


class TestLineageIsNotWhitelisted(unittest.TestCase):
    """量具的第四个缺陷: 报表把血缘键**白名单**掉了

    第一版 `run_one` 只挑 7 个键落盘, 于是 Conductor 算好的 `candidate_signature` /
    `duplicate_of` / `recipes_applied` 在落盘前被丢掉 —— 而 doc 48 第七节第 2 条正承诺
    「跑完就能离线算重复率」。真机 12 轮跑完才发现, 那一批的重复基率**永久没了**
    (只剩终端里一行 `[Novelty]` 汇总, 那不是可复核的记录)。与 doc 33 的 `runs`/
    `max_time_ms`、doc 37 的 `llm_cost` 同一族: 值算出来了, 死在搬运路上。

    白名单的真正问题不是漏了哪三个键, 而是**血缘每加一个键都要记得来这里加一次,
    而忘记的代价是静默的**。所以修法是取消白名单, 并让守卫去读真代码的键集合。
    """

    def test_record_rounds_keeps_every_key_the_conductor_writes(self):
        keys = _real_lineage_keys()
        self.assertIn("candidate_signature", keys, "血缘里没有签名 —— 先去看 doc 47/48")
        hist = [{k: f"v-{k}" for k in keys}]
        hist[0]["all_passed"] = 1          # 规整成 bool
        hist[0]["levels_passed"] = None    # 规整成 {}
        out = record_rounds(hist)
        self.assertEqual(keys, set(out[0]),
                         f"键在搬运中被丢或被加: {sorted(keys ^ set(out[0]))}")
        self.assertIs(True, out[0]["all_passed"])
        self.assertEqual({}, out[0]["levels_passed"])
        json.dumps(out)  # 落盘那一步不许崩

    def test_a_key_added_later_survives(self):
        """白名单要是回来, 这一条会红: 一个它没听说过的键必须原样穿过"""
        out = record_rounds([{"round": 1, "some_future_knob": {"a": 1}}])
        self.assertEqual({"a": 1}, out[0]["some_future_knob"])

    def test_the_fixtures_cover_every_real_key(self):
        """假记录少一个键, 聚合逻辑就在那个键上从没被验过 —— 白名单正是这样隐形的"""
        bench = (REPO_ROOT / "benchmarks" / "harness_pass_rate.py").read_text(encoding="utf-8")
        record = _dict_keys_in(bench, {"case", "ok", "rounds"})
        self.assertEqual(1, len(record), f"run_one 的记录字典认出 {len(record)} 处")
        self.assertEqual(set(), _real_lineage_keys() - set(_round(1)),
                         "测试的假血缘比 Conductor 真写的少键")
        self.assertEqual(set(), record[0] - set(_run([])),
                         "测试的假运行记录比 run_one 真写的少键")


class TestDuplicateRateDenominator(unittest.TestCase):
    """doc 47 没量到的基率, 从此每份报表都带上; 分母与「测不到」都要对"""

    def test_first_rounds_are_not_in_the_denominator(self):
        """第一轮没有可比对象 —— 算进分母就是把「不可能重复」记成「没重复」"""
        runs = [_run([_round(1, signature="a"), _round(2, signature="b")])
                for _ in range(MIN_DENOM)]
        agg = aggregate(runs)
        self.assertEqual((0, MIN_DENOM), agg["duplicate_rounds"])
        self.assertTrue(agg["duplicate_measurable"])
        self.assertEqual(0.0, agg["duplicate_pct"])

    def test_a_flagged_repeat_is_counted(self):
        runs = [_run([_round(1, signature="a"), _round(2, signature="a", duplicate_of=1)])
                for _ in range(MIN_DENOM)]
        agg = aggregate(runs)
        self.assertEqual((MIN_DENOM, MIN_DENOM), agg["duplicate_rounds"])
        self.assertEqual(100.0, agg["duplicate_pct"])
        self.assertIn("100.0%", format_report(agg))

    def test_missing_signature_reports_unmeasurable_not_zero(self):
        """这就是 doc 48 那一批 12 轮的处境: 0% 与「测不到」必须是两行不同的字"""
        agg = aggregate([_run([_round(1), _round(2)]) for _ in range(MIN_DENOM)])
        self.assertFalse(agg["duplicate_measurable"])
        self.assertIsNone(agg["duplicate_pct"])
        text = format_report(agg)
        self.assertIn("测不到", text)
        self.assertNotIn("0.0%", text.split("【候选重复率】")[1].split("【")[0])


class TestPassedViaIsReported(unittest.TestCase):
    def test_fix_rescues_show_up_in_the_report(self):
        """doc 48 读数 ④ (33% 首轮死在 L1、被 Fixer 救回) 从此是报表输出而不是手算"""
        runs = [_run([_round(1, l1=False),
                      _round(2, l2=True, l3=True, action="fix", validity=VALID,
                             speedup=1.5)])
                for _ in range(MIN_DENOM)]
        agg = aggregate(runs)
        self.assertEqual({"fix": MIN_DENOM}, agg["passed_via"])
        self.assertIn("fix", format_report(agg))

    def test_first_pass_wins_over_later_optimize_rounds(self):
        runs = [_run([_round(1, l2=True, l3=True, action="generate", validity=VALID,
                             speedup=1.5),
                      _round(2, l2=True, l3=True, validity=VALID, speedup=1.6)])
                for _ in range(MIN_DENOM)]
        self.assertEqual({"generate": MIN_DENOM}, aggregate(runs)["passed_via"])


class TestOldPayloadStillRenders(unittest.TestCase):
    def test_a_report_replayed_from_an_older_run_does_not_crash(self):
        """`output/harness/20260928_174223/pass_rate.json` 是修缺陷之前落的盘

        它没有 `passed_via` / `llm_calls`, 轮记录里也没有 `candidate_signature`。
        「落原始记录以便离线重算」这个设计只有在**旧记录仍能被新报表读**的时候才算数,
        所以缺键必须降级成看得见的「(未记)」与「测不到」, 而不是 KeyError。
        """
        old = [{"case": "layernorm#2", "ok": True, "error": None,
                "rounds": [{"round": 1, "action": "generate", "all_passed": False,
                            "levels_passed": {"L1": False},
                            "measurement_validity": None, "gflops": None,
                            "speedup": None},
                           {"round": 2, "action": "fix", "all_passed": True,
                            "levels_passed": {"L1": True, "L2": True, "L3": True},
                            "measurement_validity": VALID, "gflops": 900.0,
                            "speedup": 1.0711}],
                # 缺陷 1 的现场: 任务过了, 但旧口径排除 `fix` 轮 ⇒ rounds_to_pass 是 None
                "final_all_passed": True, "rounds_to_pass": None,
                "admitted": True, "admit_reason": "",
                "tokens": 22355, "cost_usd": 0.0, "wall_s": 104.5}
               for _ in range(MIN_DENOM)]
        agg = aggregate(old)
        text = format_report(agg)
        self.assertIn("(未记)", text)
        self.assertIn("测不到", text)
        self.assertIn(str(22355 * MIN_DENOM), text)   # 成本仍按次求和
        self.assertIn("0 次调用", text)               # 旧记录没有 llm_calls ⇒ 0, 不是崩溃
        # 缺陷 1 的形状: 4 次运行都成功, 而 rounds-to-pass 的样本数是 0
        self.assertEqual((MIN_DENOM, MIN_DENOM), agg["task_success"])
        self.assertFalse(agg["rounds_to_pass"])


if __name__ == "__main__":
    unittest.main()


class TestAdmissionDenominatorsMatch(unittest.TestCase):
    """入库率的分母是「通过的运行」, 而拒收名单以前取自**所有未崩溃的运行** —— 两个口径
    不同源 (2026-09-29 那份真机报表: 「入库 3/3 = 100%」紧跟三条拒收, 而三条全部来自
    **未通过**的运行)。`_save_to_db()` 只在 `artifact.all_passed` 那一支被调
    (`conductor.py:1112` 明写), 所以那些运行在生产里**从没被这道门问过**。
    后果不是数字算错, 是故事讲反: 读者会以为入库门拒了 3 个好核。
    """

    def _mixed(self):
        """今天那份数据的样子: 3 次通过且入库 + 3 次未通过但都被量具问了这道门"""
        passed = [_run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.07)],
                       admitted=True, reason="speedup=1.070x, validity=valid")
                  for _ in range(3)]
        failed = [_run([_round(1, l2=False, l3=False, validity=VALID, speedup=None)],
                       final=False, admitted=False,
                       reason="相对同机基线没有加速 (speedup=0.99)")
                  for _ in range(3)]
        return passed + failed

    def test_unpassed_runs_are_not_counted_as_rejections(self):
        agg = aggregate(self._mixed())
        self.assertEqual([], agg["admit_rejections"],
                         "未通过的运行被算进拒收名单了 —— 分子分母又不同源")
        self.assertEqual(1, len(agg["counterfactual_rejections"]),
                         "名单按理由去重: 三次同一个理由只该有一种")
        self.assertEqual(3, agg["counterfactual_runs"],
                         "但运行数不能跟着被去重掉, 否则「另 3 条」到底是 3 次还是 3 种说不清")

    def test_rejection_kinds_and_runs_are_both_reported(self):
        """拒收名单是按理由去重的, 所以「几条」必须同时给运行数与理由数。
        4 条而不是 3 条: 量具自己「分母不足不出百分数」那条纪律 (§3.2) 会短路整份报表。
        """
        reasons = ("0.99", "0.9688", "None", "0.97")
        runs = [_run([_round(1, l2=True, l3=True, validity=VALID, speedup=1.07)],
                     final=True, admitted=False,
                     reason=f"相对同机基线没有加速 (speedup={sp})") for sp in reasons]
        text = format_report(aggregate(runs))
        self.assertIn("4 次运行, 4 种不同的理由", text)
        runs[1]["admit_reason"] = runs[0]["admit_reason"]
        text = format_report(aggregate(runs))
        self.assertIn("4 次运行, 3 种不同的理由", text)

    def test_report_says_the_gate_rejected_nothing(self):
        text = format_report(aggregate(self._mixed()))
        self.assertIn("没有拒收任何一次通过的运行", text)
        self.assertNotIn("  拒收: 相对同机基线没有加速", text)
        self.assertIn("假想判定", text)

    def test_a_real_rejection_still_shows_up(self):
        """方向对照: 修复不能把拒收名单清空成「门从没拒过任何东西」"""
        runs = self._mixed()
        runs[0]["admitted"] = False
        runs[0]["admit_reason"] = "窗口被抢占污染 (util 超阈值)"
        agg = aggregate(runs)
        self.assertEqual(["窗口被抢占污染 (util 超阈值)"], agg["admit_rejections"])
        self.assertIn("  拒收: 窗口被抢占污染", format_report(agg))

    def test_a_passed_run_with_no_verdict_is_flagged_not_silent(self):
        """门没被问 ≠ 门放过了 —— 这两种情况在报表里必须长得不一样"""
        runs = self._mixed()
        runs[0]["admitted"] = False
        runs[0]["admit_reason"] = ""
        agg = aggregate(runs)
        self.assertIn("没有入库判定记录", format_report(agg))


def _backend(caps=(), gpu=None, override=True):
    """一个能控制「实际用哪张卡」的假件

    `override=False` 走基类那个 `selected_gpu` (返回 None), 用来验「后端压根不固定卡」
    这一态 —— 它必须由**基类**回答, 不是由假件忘记覆盖来回答。
    """
    if not override:
        return DeclaredFakeBackend(caps=caps, label="FakeBackend")

    class _Pinned(DeclaredFakeBackend):
        @property
        def selected_gpu(self):
            return gpu

    return _Pinned(caps=caps, label="FakeBackend")


class TestReportedDeviceComesFromBackend(unittest.TestCase):
    """#52: 报表里的「跑在哪张卡」必须来自执行侧, 不是本地环境变量

    旧写法是 `"gpu": os.environ.get("OPERATOR_AGENT_GPU", "(未固定)")` —— 读的是用户**想**的卡,
    而 `SSHExecutor` 从不读那个变量 (它在本进程自动挑一张最闲的卡, 再把选中的号下发给远端)。
    第一份基线里那句「GPU 2 起跑」因此无法确认是报表值还是 runner 日志值 (实际是 GPU 7)。
    """

    def test_the_backend_wins_over_the_environment(self):
        """已知会被看见的信号: 两个来源给两个答案, 记录里必须是执行侧那个"""
        rec = device_record(_backend(caps={Cap.GPU_PINNING}, gpu=7),
                            env={"OPERATOR_AGENT_GPU": "2"})
        self.assertEqual(7, rec["gpu"])
        self.assertEqual("GPU 7", rec["state"])
        self.assertEqual("2", rec["requested_env"])
        self.assertTrue(rec["conflicts"], "用户想 2、实际 7 —— 这必须响, 不能静默换卡")

    def test_direction_is_right_not_just_a_value(self):
        """反向: 反过来设 (env=7、实卡=2) 时记录跟着变成 2, 而不是仍写 7"""
        rec = device_record(_backend(caps={Cap.GPU_PINNING}, gpu=2),
                            env={"OPERATOR_AGENT_GPU": "7"})
        self.assertEqual(2, rec["gpu"])
        self.assertTrue(rec["conflicts"])

    def test_matching_values_are_not_a_conflict_but_still_come_from_the_backend(self):
        rec = device_record(_backend(caps={Cap.GPU_PINNING}, gpu=7),
                            env={"OPERATOR_AGENT_GPU": "7"})
        self.assertFalse(rec["conflicts"])
        self.assertEqual("backend.selected_gpu", rec["source"])

    def test_no_env_var_means_no_conflict(self):
        rec = device_record(_backend(caps={Cap.GPU_PINNING}, gpu=7), env={})
        self.assertFalse(rec["conflicts"])
        self.assertIsNone(rec["requested_env"])

    def test_a_backend_without_pinning_is_not_reported_as_unpinned(self):
        """「没有固定卡号这件事」与「本来该固定但这轮没解析出来」不许共用一个字符串"""
        rec = device_record(_backend(caps=(), gpu=None, override=False), env={})
        self.assertIsNone(rec["gpu"])
        self.assertIn("不具备", rec["state"])
        self.assertNotIn("未固定", rec["state"])

    def test_declared_pinning_but_unresolved_is_its_own_state(self):
        rec = device_record(_backend(caps={Cap.GPU_PINNING}, gpu=None), env={})
        self.assertIn("远端自动选择", rec["state"])

    def test_the_three_states_are_mutually_distinguishable(self):
        states = {
            device_record(_backend(caps={Cap.GPU_PINNING}, gpu=7), env={})["state"],
            device_record(_backend(caps={Cap.GPU_PINNING}, gpu=None), env={})["state"],
            device_record(_backend(caps=(), override=False), env={})["state"],
        }
        self.assertEqual(3, len(states), f"三态塌成了 {states}")

    def test_record_is_json_serialisable(self):
        """这条数据要落进 `pass_rate.json` —— 不可序列化会在落盘那一步毁掉整份报表"""
        json.dumps(device_record(_backend(caps={Cap.GPU_PINNING}, gpu=3),
                                env={"OPERATOR_AGENT_GPU": "3"}))


def _env_lookup_sites(source: str) -> list:
    """哪个函数在读 `OPERATOR_AGENT_GPU` 这个环境变量

    ⚠️ 判据必须比「字面上的 `os.environ.get(...)`」宽一层: 真代码里 `device_record` 收的是
    `env=None → os.environ` 这个默认参数, 所以它写的是 `env.get(...)`。只认前一种写法的扫描器
    会对**真代码返回空集**, 于是那条守卫就变成恒真的绿 —— 这是「守卫空转」的镜像形态:
    不是解析区域找错, 而是解析的写法比被测代码窄。这里认任何 `.get("OPERATOR_AGENT_GPU")`
    (接收者是 `Name` 或 `os.environ`), 两种写法都漏不掉。
    """
    out = []
    for fn in [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and isinstance(node.func.value, (ast.Name, ast.Attribute))
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "OPERATOR_AGENT_GPU"):
                out.append(fn.name)
    return out


class TestReportNeverReadsTheEnvForTheDevice(unittest.TestCase):
    """源码级守卫: 那个环境变量的读取只能活在 `device_record` 里当**对照**, 不能当来源"""

    def test_only_device_record_reads_it(self):
        source = (REPO_ROOT / "benchmarks" / "harness_pass_rate.py").read_text(
            encoding="utf-8")
        self.assertEqual(["device_record"], _env_lookup_sites(source),
                         "又有一处把本地环境变量当成了实际用卡")

    def test_the_scanner_is_not_vacuous(self):
        """两种写法都要认: 直接读 `os.environ`, 以及经过一个指向它的局部别名"""
        synthetic = (
            "import os\n"
            "def device_record(b, env=None):\n"
            "    env = os.environ if env is None else env\n"
            "    return env.get(\"OPERATOR_AGENT_GPU\", \"\")\n"
            "def main():\n"
            "    return {\"gpu\": os.environ.get(\"OPERATOR_AGENT_GPU\", \"(未固定)\")}\n"
        )
        self.assertEqual(["device_record", "main"], _env_lookup_sites(synthetic))
