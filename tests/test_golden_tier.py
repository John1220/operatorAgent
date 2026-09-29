"""golden 层的 `PASS` 不再盖住四件不同的事 (#73, 吸纳 DSH `docs/testing.md`)。

一手出处（本地 vendored 树 `~/repos/github/deepseek-harness`，HEAD `47f943859b`）：

* `docs/defensive-patterns.md:9` **"Report orthogonal outcomes independently"** —
  *"A result can be several things at once … Surface each independent fact … on its own;
  never nest one flag's report inside another's branch, or a caller reads a cut-short run
  as a clean success."*
* `docs/testing.md:29` **"Verify the world, not the self-report"** —
  *"An e2e assertion re-runs the command or re-reads the file externally …
  **Assert untouched files are byte-identical**."*
* `docs/testing.md:12` 那句 *"review every JSONL and expected-output diff"* —— 期望值
  变了必须有人签认，不许被一次自动刷新吃掉。

对本仓的失效不是抽象的：`python3 tests/golden_regression.py compare` 在这台机上
0.38 秒跑完并报 **`PASS 4 / FAIL 0`**，而四个 case 里只有一个真是仪器 ——
`layernorm`/`relu` 的 baseline 本身就是 FAIL（一致地坏，检测不了任何回归），
`softmax` 的 `gflops=0`（没有性能读数，而旧文案还谎称是 `strict_l3=False` 关掉的），
`matmul` 的期望值比当前读数低 **4.9 倍**（112.6 vs 666.7）而容忍窗写成「±20%」
却只罚下行。四件事都叫 PASS。

⚠️ 这一组用例**刻意不钉任何 GFLOPS 数值**：`mock_gpu.py:391` 的 gflops 是
`flops / time.perf_counter() 差` 得来的，同一段代码两次跑出 666.72 与 708.86（实测差 6%），
拿它当断言对象就是 doc 49 §7.16 那条「非确定性不许写死」。判据逻辑没有边界，
直接调 `_compare_one` 就是它的真实入口路径（DSH `testing.md:21` 那句
「prefer the real implementation」在这里不需要 mock）。

变异标定（9 条，0 逃逸，期望跑之前按「哪条断言的前提消失」推）：
  M1 死仪器不再记事实      → NE,GATE,DEAD,FVOC,REACH   M2 上漂检查拆掉 → NE,GATE,UP,REACH
  M3 REVIEW 塌回 FAIL      → GATE,UP,DEAD,NPR,SVOC     M4 lenient 分支不可达 → GATE
  M5 汇总不报独立事实      → SUM                       M6 有事实的例计入 PASS → SUM,PRC
  M7 portable 退化成恒等   → REL                       M8 compare 改写 registry → BYTE
  M9 FACTS 里混进状态词    → DISJ,FVOC,REACH
两处标定本身教的事：
① 词表的**静态**差集守卫（FVOC）看不见「写了但走不到」—— M2 把 `elif` 换成 `elif False:`
   时它是绿的，因为字面量还在源码里。所以补了 REACH（用输入表把每个词真的产一遍）。
   差集只保证「有写者」，可达性要另一张网。
② 中途加一条守卫（REACH）之后，**所有**受影响的期望必须重推 —— 这一轮我漏了 M1 与 M3 两次，
   两次都是「新增守卫的期望没回头补」，不是守卫错。

运行: python3 -B -m unittest tests.test_golden_tier -v
"""

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import golden_regression as gr  # noqa: E402

SRC = (ROOT / "tests" / "golden_regression.py").read_text(encoding="utf-8")
REGISTRY = ROOT / "tests" / "golden_registry.json"


def _rec(all_passed: bool, gflops: float = 0.0) -> gr.BaselineRecord:
    return gr.BaselineRecord(case_id="c", description="d", timestamp=0.0,
                             all_passed=all_passed, gflops=gflops)


def _cmp(bl_pass, bl_gflops, cur_pass, cur_gflops, strict_l3=True, tolerance=0.20):
    return gr._compare_one(_rec(bl_pass, bl_gflops), _rec(cur_pass, cur_gflops),
                           tolerance=tolerance, strict_l3=strict_l3)


class TestFactsDriveStatus(unittest.TestCase):
    """`:9` 的那一半：状态由**独立事实**推出，而不是一个词盖住多种情况。"""

    def test_a_nonempty_fact_set_is_never_reported_as_pass(self):
        cases = [
            _cmp(True, 100.0, False, 0.0),                 # 功能回归
            _cmp(True, 100.0, True, 40.0),                 # 性能回归
            _cmp(True, 100.0, True, 900.0),                # 性能上漂
            _cmp(False, 0.0, False, 0.0),                  # 两边都失败 = 死仪器
            _cmp(False, 0.0, True, 0.0),                   # baseline 坏、当前好
            _cmp(True, 0.0, True, 0.0),                    # 没有可用分母
        ]
        for status, facts, reason, _dr in cases:
            self.assertNotEqual("PASS", status,
                                f"有事实要上报却仍报 PASS: facts={facts} reason={reason}")
            self.assertTrue(facts, f"status={status} 却没有事实可报 —— 那这个词没有对象")

    def test_regression_facts_gate_fail_while_staleness_gates_review(self):
        """FAIL 与 REVIEW 要求的动作不同：修代码 vs 签认期望。合成一个码就叠回去了。

        每条都是**真实的输入对 → 代码给出的状态**，不在测试里重算一遍生产侧的规则
        （那样写的断言永远为真，是「假件当被测对象」的另一个形态）。
        """
        for bl, cur, strict, expect, why in (
                ((_rec(True, 100.0), _rec(False, 0.0), True, "FAIL", "功能回归")),
                ((_rec(True, 100.0), _rec(True, 40.0), True, "FAIL", "性能掉出窗口")),
                ((_rec(True, 100.0), _rec(True, 900.0), True, "REVIEW", "性能上漂")),
                ((_rec(False, 0.0), _rec(False, 0.0), True, "REVIEW", "一致地坏")),
                ((_rec(False, 0.0), _rec(True, 10.0), True, "REVIEW", "修好了没签认")),
                ((_rec(True, 0.0), _rec(True, 5.0), True, "REVIEW", "没有可用分母")),
                ((_rec(True, 100.0), _rec(True, 105.0), True, "PASS", "窗口内")),
                ((_rec(True, 100.0), _rec(True, 0.0), False, "REVIEW", "lenient 不判性能")),
        ):
            status, facts, _r, _dr = gr._compare_one(bl, cur, tolerance=0.2, strict_l3=strict)
            self.assertEqual(expect, status, f"{why}: bl={bl} cur={cur} ⇒ {status}/{facts}")

    def test_upward_drift_is_review_rather_than_regression(self):
        """快 5 倍不是回归，但那个期望值已经不代表任何被测物（`:12` 要人 review 每一个 diff）。"""
        status, facts, reason, dr = _cmp(True, 100.0, True, 900.0)
        self.assertEqual("REVIEW", status, f"上漂 8 倍却判成 {status}: {reason}")
        self.assertIn("perf_drift_up", facts)
        self.assertGreater(dr, 1.0)

    def test_a_consistently_failing_case_is_a_dead_instrument_not_a_pass(self):
        """旧版这一条返回 `("PASS", "baseline 本就是 FAIL, 当前也 FAIL (无功能回归)")`。"""
        status, facts, reason, _ = _cmp(False, 0.0, False, 0.0)
        self.assertEqual("REVIEW", status)
        self.assertIn("baseline_was_failing", facts)
        self.assertIn("死仪器", reason, "理由里要写明它检测不了回归，而不是「无功能回归」")

    def test_the_missing_perf_reference_is_named_instead_of_blamed_on_a_flag(self):
        """旧文案在 `strict_l3=True` 时也印「strict_l3=False, 不判性能」—— 谎报开关状态。"""
        status, facts, reason, _ = _cmp(True, 0.0, True, 5.0, strict_l3=True)
        self.assertEqual("REVIEW", status)
        self.assertIn("no_perf_reference", facts)
        self.assertNotIn("strict_l3=False", reason,
                         f"这一维是开着的, 不许把缺分母说成关开关: {reason}")


class TestVocabularyHasProducers(unittest.TestCase):
    """与 #72 同一条纪律：一个没有产生者的词就是「算了没人读」的又一个副本。"""

    def _facts_appended(self):
        fn = next(n for n in ast.walk(ast.parse(SRC))
                  if isinstance(n, ast.FunctionDef) and n.name == "_compare_one")
        return {c.args[0].value for c in ast.walk(fn)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "append" and isinstance(c.func.value, ast.Name)
                and c.func.value.id == "facts"
                and c.args and isinstance(c.args[0], ast.Constant)}

    def _statuses_produced(self):
        tree = ast.parse(SRC)
        out = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant):
                if any(isinstance(t, ast.Name) and t.id == "status" for t in n.targets):
                    out.add(n.value.value)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "CompareReport"):
                for kw in n.keywords:
                    if kw.arg == "status" and isinstance(kw.value, ast.Constant):
                        out.add(kw.value.value)
        return out

    def test_every_fact_word_has_a_producer(self):
        self.assertEqual(set(gr.FACTS), self._facts_appended(),
                         "FACTS 与 `_compare_one` 实际 append 的词对不上")

    def test_every_fact_word_is_actually_reachable(self):
        """上一条是**静态**的：它拦「词表里有、代码里从没写过」，拦不住「写了但走不到」
        （标定 M2 把 `elif dr > tolerance:` 换成 `elif False:` 时它是绿的 —— 字面量还在，
        分支已经永远不成立）。这一条用一张输入表把每个词**真的产出来**一遍，
        两张网才凑齐「有写者 + 写者可达」。
        """
        produced = set()
        for bl, cur, strict in (
                ((_rec(True, 100.0), _rec(False, 0.0), True)),
                ((_rec(True, 100.0), _rec(True, 40.0), True)),
                ((_rec(True, 100.0), _rec(True, 900.0), True)),
                ((_rec(False, 0.0), _rec(False, 0.0), True)),
                ((_rec(False, 0.0), _rec(True, 10.0), True)),
                ((_rec(True, 0.0), _rec(True, 5.0), True)),
                ((_rec(True, 100.0), _rec(True, 0.0), False)),
        ):
            _status, facts, _r, _dr = gr._compare_one(bl, cur, tolerance=0.2, strict_l3=strict)
            produced |= set(facts)
        self.assertEqual(set(gr.FACTS), produced,
                         f"这些事实没有任何输入能产出来: {set(gr.FACTS) ^ produced}")

    def test_every_status_word_has_a_producer(self):
        self.assertEqual(set(gr.STATUSES), self._statuses_produced(),
                         f"STATUSES 与代码产出的状态对不上: 代码 {self._statuses_produced()}")

    def test_fact_vocabulary_does_not_reuse_status_words(self):
        """两套词若重叠, 读报表的人会分不清哪个是门、哪个是事实。"""
        self.assertFalse(set(gr.FACTS) & set(gr.STATUSES))


class TestSummaryDoesNotCollapse(unittest.TestCase):
    """汇总行是本片的靶子本身：它以前只有一个 PASS 数。"""

    def _render(self, reports):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gr.GoldenRegressionHarness._print_report(reports)
        return buf.getvalue()

    def test_summary_counts_each_fact_separately(self):
        reports = [
            gr.CompareReport(case_id="healthy", status="PASS"),
            gr.CompareReport(case_id="dead", status="REVIEW",
                             facts=("baseline_was_failing",)),
            gr.CompareReport(case_id="stale", status="REVIEW",
                             facts=("perf_drift_up", "no_perf_reference")),
        ]
        out = self._render(reports)
        self.assertIn("独立事实", out, "汇总里必须逐条报独立事实, 不许只有一个 PASS 数")
        for w in ("baseline_was_failing", "perf_drift_up", "no_perf_reference"):
            self.assertIn(w, out)
        self.assertIn("REVIEW 2", out)
        self.assertIn("死仪器", out)

    def test_a_case_with_facts_never_adds_to_the_pass_count(self):
        out = self._render([gr.CompareReport(case_id="dead", status="REVIEW",
                                             facts=("baseline_was_failing",))])
        self.assertIn("PASS 0", out, f"有事实的例被算进了 PASS: {out}")


class TestGradedRunDoesNotTouchItsOwnExpectations(unittest.TestCase):
    """DSH `testing.md:29`：断言**没被动过的文件字节不变**。

    被评分的一方不许改写评分标准 —— `compare` 若哪天顺手把 baseline 刷新成当前值，
    「期望」就会永远追不上现实，而这条是唯一能看见那件事的哨兵。
    """

    def test_compare_leaves_the_registry_byte_identical(self):
        before = REGISTRY.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            harness = gr.GoldenRegressionHarness()
            reports = harness.compare()
        self.assertEqual(before, REGISTRY.read_bytes(),
                         "compare 改写了 registry ⇒ 被评分的一方在定评分标准")
        self.assertEqual(len(gr.GoldenRegressionHarness().cases), len(reports),
                         "每个注册 case 都要有一条结果, 少跑一例就是少一个读者")

    def test_the_committed_registry_carries_no_machine_local_path(self):
        """能提交的前提：期望文件里不许有本机绝对路径（旧版因此只能活在一台机上）。"""
        self.assertTrue(REGISTRY.exists(), "registry 不在 —— 它应当被提交而不是被 ignore")
        text = REGISTRY.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", text)
        self.assertNotIn("/home/", text)
        for rec in json.loads(text)["baselines"].values():
            d = rec.get("artifact_dir") or ""
            self.assertFalse(os.path.isabs(d), f"绝对路径又回来了: {d}")
            if d:
                self.assertTrue(d.startswith("output" + os.sep),
                                f"出处指针应当指向 repo 内的相对路径: {d}")

    def test_registry_is_version_controlled(self):
        """整套期望曾经只存在这台机上：`.gitignore:30` 排除它，`git log --all` 证明从未被跟踪。"""
        r = subprocess.run(["git", "ls-files", "--error-unmatch",
                            "tests/golden_registry.json"],
                           cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(0, r.returncode,
                         f"registry 没有入库, 别人 clone 到的是一套空期望: {r.stderr.strip()}")

    def test_the_untold_check_actually_detects_an_untracked_file(self):
        """反空转：上面那条判据**用的机制**要能分辨跟踪与否，否则它只是一句口号。
        拿一个确定没入库的文件走同一条命令，必须非零退出。
        """
        r = subprocess.run(["git", "ls-files", "--error-unmatch",
                            "tests/definitely_not_tracked_73.py"],
                           cwd=str(ROOT), capture_output=True, text=True)
        self.assertNotEqual(0, r.returncode, "`--error-unmatch` 对未跟踪文件也返回 0 ⇒ 上一条是恒真的")


class TestPortableProvenance(unittest.TestCase):
    """`portable()` 是「期望文件可以入库」这件事的前提，所以它自己要被测。"""

    def test_a_repo_internal_path_becomes_relative(self):
        p = os.path.join(str(ROOT), "output", "matmul", "mock_gpu", "x")
        got = gr.portable(p)
        self.assertEqual(os.path.join("output", "matmul", "mock_gpu", "x"), got)
        self.assertFalse(os.path.isabs(got))

    def test_a_path_outside_the_repo_stays_absolute_rather_than_escaping(self):
        """`../` 形式的「相对」会把出处指针变成会随 cwd 漂移的东西 —— 不如留绝对并让人看见。"""
        p = os.path.abspath(os.path.join(os.path.dirname(str(ROOT)), "elsewhere", "y"))
        got = gr.portable(p)
        self.assertTrue(os.path.isabs(got), f"repo 外的路径被改成了相对: {got}")
        self.assertNotIn("..", got)

    def test_empty_stays_empty(self):
        """缺出处不许造一个 `.` 或 `/` 出来（与「缺值不写键」同族）。"""
        self.assertEqual("", gr.portable(""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
