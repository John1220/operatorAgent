"""§21.3 对照实验的守卫: 两臂只差一个开关 / 留痕随开关变 / 结论不许越过 N

`benchmarks/evidence_ablation_h20.py` 要跑真机烧 token, 所以它的**接线**必须离线可验
(零 GPU / 零 LLM / 不联网): 接线错了却真机跑一遍, 读数会「看起来像结论」。这里钉四件事:

1. **两臂只差 `structured_evidence` 一个键** —— 键差集现算, 且两臂的值必须是**原始值**
   (混进 `KernelDatabase` 这类实例, diff 就会把「两个不同的 off 库」读成一处差异,
   守卫自己先失效);
2. **三个场景确实按设计失败** —— 坏码与保底实现逐字不同 (定点改动没命中就是静默失效:
   那一格会变成「第一轮就过」, 没有 Fixer, 对照价值在无声中归零);
3. **留痕随开关变** —— 开臂 `fix_evidence.injected=True` 且有 `chars`, 关臂没有; 且两臂
   的 Fixer prompt 砍掉证据段之后**除空白外逐字相同**、历史修复案例段两臂都是「无」
   (失败库串味会让「只差一个开关」这句话不成立);
4. **结论带着自己的量程** —— 结论句里必须有 N、有 `min_detectable_delta` 算出来的分辨下限,
   并且**不许**出现「有收益 / 证明了收益」这类越界断言; 两臂读数相同时必须说出「分辨不出」。

运行: python3 -m pytest tests/test_evidence_ablation.py -q
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import evidence_ablation_h20 as ab            # noqa: E402
from benchmarks.harness_pass_rate import min_detectable_delta  # noqa: E402
from operator_agent.mock_gpu import MockGPU                    # noqa: E402

#: 越界断言的形状 (不包括结论里那句「不许声称收益, 也不许声称无收益」—— 那句是**边界**本身,
#: 它不含「有收益」这三个连续字符, 所以不会被这张名单误伤)
FORBIDDEN_CLAIMS = ("有收益", "收益明显", "证明了收益", "收益为正", "建议默认开启",
                    "开关有效", "显著提升")

_RUNS = None


def _dry_runs() -> list:
    """离线把三个场景 × 两臂跑一遍 (MockGPU + 零 token 替身), 缓存起来给所有用例共用"""
    global _RUNS
    if _RUNS is None:
        runs = []
        scenarios = ab._scenarios("python", "mock_gpu", (64, 64))
        for scenario in scenarios:
            for on in (True, False):
                rec = ab.run_one(scenario, structured_evidence=on,
                                 llm=ab.DryLlm(scenario),
                                 backend=ab.RecordingBackend(MockGPU()),
                                 task=ab._task("python", "mock_gpu"),
                                 inputs=ab._inputs((64, 64)))
                rec["rep"] = 1
                runs.append(rec)
        _RUNS = runs
    return _RUNS


def _pairs() -> list:
    return ab.pair_up(_dry_runs())


class TestTheArmsDifferOnOneKeyOnly(unittest.TestCase):

    def test_the_key_diff_is_exactly_the_switch(self):
        """键差集是**现算**的: 谁往两臂里塞了第二个差异, 这条就红"""
        self.assertEqual({"structured_evidence"}, ab.arm_diff())

    def test_both_arms_carry_the_same_keys(self):
        self.assertEqual(set(ab.arm_kwargs(True)), set(ab.arm_kwargs(False)))

    def test_the_arm_values_are_primitives(self):
        """混进实例 ⇒ diff 会把「两个不同的实例」读成差异 (或反之), 守卫自己先失效"""
        for arm in (True, False):
            for key, value in ab.arm_kwargs(arm).items():
                self.assertIsInstance(value, (bool, int, float, str),
                                      f"{key} 不是原始值: {type(value)}")

    def test_the_switch_actually_flips(self):
        self.assertTrue(ab.arm_kwargs(True)["structured_evidence"])
        self.assertFalse(ab.arm_kwargs(False)["structured_evidence"])


class TestTheScenariosFailAsDesigned(unittest.TestCase):

    def test_every_broken_source_differs_from_the_good_one(self):
        """静默失效的形态: 定点改动没命中 ⇒ 坏码==好码 ⇒ 那一格第一轮就过, 没有 Fixer"""
        for dsl, hardware in (("python", "mock_gpu"), ("triton", "nvidia_gpu")):
            rows = ab._scenarios(dsl, hardware, (64, 64))
            self.assertEqual(3, len(rows), f"{dsl}: 场景数变了")
            for row in rows:
                self.assertNotEqual(row["broken"], row["good"],
                                    f"{dsl}/{row['id']}: 坏码与好码逐字相同")
                # 坏码必须是**好码的一处改动** (同源), 不是另一份手写的东西
                self.assertGreater(len(row["good"]), 0)

    def test_the_perturbation_refuses_a_missed_anchor(self):
        """锚点没命中要**抛错**, 不许安静地返回原文"""
        with self.assertRaises(ValueError):
            ab._perturb("def kernel():\n    return 1\n", "根本不存在的锚点", "x")

    def test_the_first_round_dies_where_the_scenario_says(self):
        """语法/接口都死在本地 L1, 数值那一格必须死在 L2 —— 读数里的「首轮挂」要能对上"""
        got = {}
        for rec in _dry_runs():
            if rec["arm"] == "on":
                got[rec["scenario"]] = rec["round1_failed_levels"]
        self.assertEqual(["L1"], got.get("syntax"), got)
        self.assertEqual(["L1"], got.get("interface"), got)
        self.assertEqual(["L2"], got.get("numeric"), got)

    def test_every_scenario_reaches_the_fixer(self):
        """两臂都要有 Fixer 调用 —— 没有失败就没有对照物"""
        for rec in _dry_runs():
            self.assertGreater(rec["fix_calls"], 0,
                               f"{rec['scenario']}/{rec['arm']} 没有 Fixer 调用")


class TestTheLineageFollowsTheSwitch(unittest.TestCase):

    def test_the_on_arm_records_injected_evidence(self):
        for rec in _dry_runs():
            if rec["arm"] != "on":
                continue
            self.assertIn(True, rec["evidence_injected"],
                          f"{rec['scenario']}: 开臂没有注入留痕")
            self.assertTrue(all(c > 0 for c in rec["evidence_chars"]),
                            f"{rec['scenario']}: 注入留痕的 chars 不是正数")

    def test_the_off_arm_records_no_evidence(self):
        for rec in _dry_runs():
            if rec["arm"] != "off":
                continue
            self.assertNotIn(True, rec["evidence_injected"],
                             f"{rec['scenario']}: 关臂在产物里记着注入过")
            self.assertEqual([], rec["evidence_chars"])
            self.assertNotIn(ab.EVIDENCE_HEADING, rec["fix_prompt_first"],
                             f"{rec['scenario']}: 关臂的 prompt 里还有证据段")

    def test_the_prompt_fingerprint_is_recorded(self):
        """「同一份 prompt 出了不同修复」只能靠指纹事后查证 —— 长度不够

        真机那一跑就撞上了: 三次开臂的 prompt 长度逐字相同 (13516)、证据段也同 (685),
        但一次带 `input_precision="ieee"`、两次不带。当时只存了长度, 于是只能推断
        「prompt 应该一样」而不能证明。指纹落盘之后, 这一问句下次是可直接回答的。
        """
        for rec in _dry_runs():
            self.assertEqual(len(rec["fix_prompt_chars"]), len(rec["fix_prompt_sha256"]))
            for digest in rec["fix_prompt_sha256"]:
                self.assertEqual(64, len(digest))
        # 离线替身是确定性的: 同一格同一臂跑两次, prompt 指纹必须只有一个
        for arm in ("on", "off"):
            for sid in {r["scenario"] for r in _dry_runs()}:
                seen = {tuple(r["fix_prompt_sha256"]) for r in _dry_runs()
                        if r["arm"] == arm and r["scenario"] == sid}
                self.assertEqual(1, len(seen), f"{sid}/{arm}: 同一份输入出了多个 prompt 指纹")

    def test_the_prompts_differ_only_on_that_block(self):
        """「只差一个开关」的实证: 砍掉证据段后, 非空白残差必须是 0"""
        pairs = _pairs()
        self.assertTrue(pairs, "没有配出任何一对")
        for pair in pairs:
            self.assertIn("prompt_diff", pair, f"{pair['scenario']} 没有成对读数")
            diff = pair["prompt_diff"]
            self.assertEqual(0, diff["squeezed_residual_chars"],
                             f"{pair['scenario']}: 两臂除了那一段还有别的差")
            self.assertGreater(diff["block_chars"], 0)
            # 字节级那一点差**只能**是模板行的空白 (今天恰好 -2 字符), 不能是内容
            self.assertLessEqual(abs(diff["seam_chars"]), 4,
                                 f"{pair['scenario']}: 空白残差异常大 {diff['seam_chars']}")

    def test_the_history_section_is_empty_in_both_arms(self):
        """干净两臂的第二条: 失败库串味会让第二臂的 prompt 多出第一臂写的案例"""
        for pair in _pairs():
            self.assertTrue(pair["history_clean"], f"{pair['scenario']}: 历史案例段不为「无」")
        for rec in _dry_runs():
            self.assertTrue(ab.history_cases_are_empty(
                ab.strip_evidence_block(rec["fix_prompt_first"]) or rec["fix_prompt_first"]))

    def test_the_evidence_size_matches_the_prompt_growth(self):
        """留痕里的 `chars` 与 prompt 的增量必须对得上 (血缘不是另写一个数)

        两个量差的是模板那一行的空白: `block_chars` 是 prompt 里那一段(含边界空白),
        `evidence_chars` 是 `ctx.render()` 本身的长度。两个都得对上, 差值只许是空白。
        """
        for pair in _pairs():
            on, off = pair["on"], pair["off"]
            diff = pair["prompt_diff"]
            grew = on["fix_prompt_chars"][0] - off["fix_prompt_chars"][0]
            self.assertGreater(grew, 0, f"{pair['scenario']}: 开臂的 prompt 反而没变长")
            self.assertEqual(diff["block_chars"] + diff["seam_chars"], grew,
                             f"{pair['scenario']}: prompt 增量与「那一段」对不上")
            self.assertLessEqual(abs(on["evidence_chars"][0] - diff["block_chars"]), 4,
                                 f"{pair['scenario']}: 血缘 chars 与 prompt 里那一段差太多")


class TestTheConclusionCarriesItsOwnBounds(unittest.TestCase):

    def test_the_conclusion_has_n_the_range_and_no_claim(self):
        pairs = _pairs()
        text = ab.conclusion(_dry_runs(), pairs)
        n = len([p for p in pairs if p["usable"]])
        self.assertGreater(n, 0, "这一跑没有可用的对 —— 结论句无从检验")
        self.assertIn(f"N={n}", text)
        self.assertIn("不许声称", text)
        # 量程必须是从仓内那一份算出来的数, 不是抄来的
        self.assertIn(str(round(100.0 * min_detectable_delta(n), 1)), text)
        for bad in FORBIDDEN_CLAIMS:
            self.assertNotIn(bad, text, f"结论越界了: {bad}")

    def test_the_conclusion_names_the_gate_that_killed_each_arm(self):
        """两臂都红时, 必须说清**红在哪一道门** —— 否则「都没通过」读不出任何东西

        真机那一跑就是这么用的: 三格两臂全红, 而「红在哪」那行把它拆成
        `l1_l2_failed` (L2 容差, 与证据无关) 与 `rounds_exhausted`, 于是读者能看出
        结果侧没有信息, 而不是把 0/3 当成「证据没用」。
        """
        pairs = [p for p in _pairs() if p["usable"]]
        self.assertTrue(pairs)
        text = ab.conclusion(_dry_runs(), _pairs())
        self.assertIn("红在哪", text)
        for pair in pairs:
            for arm in ("on", "off"):
                self.assertIn(str(pair[arm]["stop"]), text,
                              f"{pair['scenario']}/{arm} 的收场理由没进结论")

    def test_the_l3_gate_is_a_knob_and_gets_reported(self):
        """门槛必须可传且进读数: 保底模板 0.47x, 门槛 0.5 时结果变量被「模板性能」占满"""
        low = ab._task("triton", "nvidia_gpu", min_speedup=0.3)
        high = ab._task("triton", "nvidia_gpu", min_speedup=0.5)
        self.assertEqual(0.3, low.constraints.min_speedup)
        self.assertEqual(0.5, high.constraints.min_speedup)
        self.assertIn("--min-speedup",
                      (ROOT / "benchmarks" / "evidence_ablation_h20.py").read_text("utf-8"))

    def test_identical_arms_must_say_the_ruler_cannot_tell(self):
        """两臂读数相同时, 结论句必须说出「分辨不出」, 不许沉默地留一个 0.0 pp"""
        pairs = _pairs()
        same = all(p["on"]["final_passed"] == p["off"]["final_passed"] for p in pairs)
        text = ab.conclusion(_dry_runs(), pairs)
        if same:
            self.assertIn("分辨不出", text)

    def test_without_a_usable_pair_it_proves_nothing(self):
        text = ab.conclusion(_dry_runs(), [{"scenario": "x", "usable": False,
                                            "on": {}, "off": {}}])
        self.assertIn("什么也证明不了", text)

    def test_a_pair_that_differs_beyond_the_block_is_marked_unusable(self):
        """两臂不止差那一段 ⇒ 这一对的对照读数**作废**, 不是降级使用"""
        good = _pairs()[0]
        on, off = dict(good["on"]), dict(good["off"])
        on["scenario"] = off["scenario"] = "dirty"
        off["fix_prompt_first"] = off["fix_prompt_first"] + "\n额外的一行差异\n"
        pairs = ab.pair_up([on, off])
        self.assertEqual("dirty", pairs[0]["scenario"])
        self.assertFalse(pairs[0]["usable"], "两臂多了一处差异却仍被当成干净的一对")
        self.assertNotEqual(0, pairs[0]["prompt_diff"]["squeezed_residual_chars"])

    def test_a_pair_without_a_fixer_call_is_dropped(self):
        on = dict(_pairs()[0]["on"])
        off = dict(_pairs()[0]["off"])
        on["scenario"] = off["scenario"] = "nofix"
        on["fix_calls"] = 0
        pairs = ab.pair_up([on, off])
        self.assertFalse(pairs[0]["usable"])
        self.assertNotIn("prompt_diff", pairs[0])


class TestTheRunOrderIsRotated(unittest.TestCase):
    """两臂顺序在场景之间轮换 —— 这条只在源码层可判 (行为上两种顺序都会跑完)

    它防的是「把端点漂移读成开关效应」: 同一场景的两臂背靠背跑、顺序在场景间翻一次,
    于是端点变慢/变快对两臂的伤害是对称的。没有这条守卫, 把轮换改成固定顺序不会有任何
    用例变红, 而读数里会多出一个**看不见的**混淆变量。
    """

    def test_the_arm_order_alternates_between_scenarios(self):
        src = (ROOT / "benchmarks" / "evidence_ablation_h20.py").read_text(encoding="utf-8")
        self.assertIn("arms = [True, False] if (si + rep) % 2 else [False, True]", src)
        # 轮换必须真的被用上 (`arms` 造出来却不用 = 摆设)
        self.assertIn("for on in arms:", src)


class TestABrokenTreeIsReportedNotSwallowed(unittest.TestCase):
    """崩掉的一臂不许被当成「那一臂没通过」

    这一条是**真撞出来的**: 跑真机那次, 别人的 `conductor.py` 正改到中间态
    (`NameError: PATCH_KINDS`), 第二轮直接炸。驱动若把异常抛出去, 读数全丢, 且事后
    分不清「树坏了」与「实验没做」; 若把崩溃记成 `final_passed=False`, 结论句就会把
    「一臂崩了」读成「开了证据也没通过」—— 那是**最坏**的一种谎。
    """

    def test_a_crashing_run_returns_a_shaped_record(self):
        class _ExplodingBackend(MockGPU):
            @property
            def capabilities(self):
                raise RuntimeError("树被改到中间态")

        scenario = ab._scenarios("python", "mock_gpu", (64, 64))[0]
        rec = ab.run_one(scenario, structured_evidence=True, llm=ab.DryLlm(scenario),
                         backend=_ExplodingBackend(), task=ab._task("python", "mock_gpu"),
                         inputs=ab._inputs((64, 64)))
        self.assertIn("RuntimeError", rec["error"])
        self.assertEqual(0, rec["fix_calls"])
        # 形状完整: 缺键会被下游读成「没记」
        for key in ("evidence_chars", "evidence_injected", "round1_levels",
                    "fix_prompt_chars", "rounds_recs"):
            self.assertIn(key, rec)

    def test_a_crashed_pair_is_unusable_and_named_in_the_conclusion(self):
        scenario = ab._scenarios("python", "mock_gpu", (64, 64))[0]
        crashed = ab._empty_run(scenario, "on", 0.1, 0, "NameError: PATCH_KINDS")
        good = dict(_pairs()[0]["off"])
        good["scenario"] = crashed["scenario"] = "crash"
        crashed["error_excerpt"] = good["error_excerpt"] = ""
        pairs = ab.pair_up([crashed, good])
        self.assertFalse(pairs[0]["usable"], "崩掉的一臂被当成了干净的一对")
        text = ab.conclusion([crashed, good], pairs)
        self.assertIn("崩了", text)
        self.assertIn("PATCH_KINDS", text)
        # 崩掉的那一臂**不许**被算进通过率的分母 (0/1 vs 1/1 会读成「关掉更好」)
        self.assertIn("什么也证明不了", text)


class TestTheHelpersReadTheTemplateHonestly(unittest.TestCase):

    def test_strip_refuses_a_prompt_without_the_next_heading(self):
        with self.assertRaises(ValueError):
            ab.strip_evidence_block("## 结构化证据\n只有这一段, 后面没有标题\n")

    def test_strip_is_a_noop_when_there_is_no_block(self):
        text = "## 历史修复案例\n\n无\n"
        self.assertEqual(text, ab.strip_evidence_block(text))

    def test_history_cases_check_reads_the_first_content_line(self):
        self.assertTrue(ab.history_cases_are_empty(
            "## 历史修复案例 (同类算子的失败-修复经验)\n\n无\n\n## 错误信息\nx\n"))
        self.assertFalse(ab.history_cases_are_empty(
            "## 历史修复案例 (同类算子的失败-修复经验)\n\n上次有个案例: ...\n"))

    def test_error_excerpt_skips_the_boilerplate_line(self):
        prompt = ("## 错误信息\n\n以下代码未通过验证。本轮报失败的阶段: L1\n\n"
                  "第 4 行: invalid syntax\n")
        excerpt = ab.error_excerpt(prompt)
        self.assertNotIn("以下代码未通过验证", excerpt)
        self.assertIn("invalid syntax", excerpt)
        # 标题那一行也必须丢掉 (2026-10-06 变异标定 M8 逃逸后补的): 只断言样板句不在,
        # 「取回来的首行是标题」就杀不掉 —— 而「场景自检」印的正是首行。
        self.assertNotIn("## 错误信息", excerpt)
        self.assertEqual("第 4 行: invalid syntax", excerpt.splitlines()[0])


class TestTheOfflineSelfCheckRuns(unittest.TestCase):

    def test_dry_run_prints_the_table_and_the_conclusion(self):
        proc = subprocess.run(
            [sys.executable, "-B", "benchmarks/evidence_ablation_h20.py",
             "--dry-run", "--scenarios", "syntax", "--reps", "1"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(0, proc.returncode, proc.stdout[-2000:] + proc.stderr[-2000:])
        self.assertIn("键差集 = ['structured_evidence']", proc.stdout)
        self.assertIn("结论:", proc.stdout)
        self.assertIn("不许声称", proc.stdout)
        self.assertIn("读数已落盘", proc.stdout)
        for bad in FORBIDDEN_CLAIMS:
            self.assertNotIn(bad, proc.stdout)

    def test_the_run_cap_refuses_an_over_budget_plan(self):
        """token 不是免费的: 计划超过上限就在**开跑之前**拒绝"""
        proc = subprocess.run(
            [sys.executable, "-B", "benchmarks/evidence_ablation_h20.py",
             "--dry-run", "--reps", "3", "--max-runs", "2"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("上限", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()