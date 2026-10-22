"""演示脚本的守卫: `benchmarks/demo_hitl_chain.py` 说得对, 而且今天仍然跑得起来

为什么给**演示**配守卫（本仓「每句 ✅ 都要有可运行的守卫」的同一条）: 演示里每一行读数
都是一句判断（「L2 拦住了错核」「同一条读数在两条契约下后果不同」「本机没有执行设备就
不谎报 L2」）。判据改名、`measurement.assess` 的规则变化、资格推导被改 —— 这些都不会让
脚本报错, 只会让它印出一张读起来仍然通顺、但已经不成立的图。

这个演示回答的是 `docs/dsh-plugin-evaluation.md` §9.5 第 1 条那个工程问题:
**判定层能不能不经过 Conductor 被串起来**。所以守卫也只问这件事:

1. 四个候选/契约的判定**逐条**出现（A 止步 L2、B 三级全过、B' 拒评、C 只到 L1）;
2. B 与 B' 的差别**只在资格掩码**（同一条读数, 五位从全开变全关）;
3. 报告产物真的写出来了, 且「我无法确立的事」里点名了缺的那个能力;
4. 落点在临时目录 —— 演示不许往仓库的 `output/`/`logs/` 里写;
5. 屏幕上**没有** `!!`（脚本里每一处 `!!` 都是「本不该发生的事发生了」）。

运行: python3 -m unittest tests.test_demo_hitl_chain -v
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import demo_hitl_chain as demo  # noqa: E402

_ENV_KEYS = ("OPERATOR_AGENT_OUTPUT_DIR", "OPERATOR_AGENT_LOG_DIR",
             "OPERATOR_AGENT_KERNEL_DB", "OPERATOR_AGENT_FAILURE_DB")


class TestTheHitlChainDemo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cls.rc = demo.main()
        cls.out = buf.getvalue()

    @classmethod
    def tearDownClass(cls):
        for key, value in cls._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_it_returns_zero_and_never_prints_a_violation_marker(self):
        self.assertEqual(self.rc, 0)
        self.assertNotIn("!!", self.out,
                         "演示里出现了 `!!` —— 那几行正是「本不该发生的事发生了」")

    # ── 场景 1: 契约与输入 ────────────────────────────────────────────────
    def test_the_contract_owns_a_reproducible_input(self):
        self.assertIn("cases: ['matmul-64x64x64']", self.out)
        self.assertIn("required_cases=1 条", self.out)
        self.assertIn("同一声明两次 materialize 逐元素相同: True", self.out)
        self.assertIn("契约哈希:", self.out)

    def test_the_rule_parser_needs_no_llm(self):
        self.assertIn("规则解析: kernel_type=matmul", self.out)
        self.assertIn("profile 来自 IntentParser 的规则分支", self.out)

    # ── 场景 2: 错核止步 L2, 处方是"修"不是"重写" ─────────────────────────
    def test_the_wrong_kernel_is_stopped_at_l2_with_a_fix_prescription(self):
        self.assertIn("候选 A（外层写错：结果整体小 10%）", self.out)
        self.assertIn("判定: {'L1': True, 'L2': False}", self.out)
        self.assertIn("route=debug_fix", self.out)
        self.assertIn("fix_not_rewrite=True", self.out)
        self.assertIn("误差形态: whole_tensor", self.out)

    # ── 场景 3/4: 同一条读数, 两条契约 → 两种后果 ──────────────────────────
    def test_the_correct_kernel_passes_all_three_levels(self):
        self.assertIn("判定: {'L1': True, 'L2': True, 'L3': True}", self.out)
        self.assertIn("validity=valid", self.out)
        self.assertIn("资格掩码 V T P R C: ✓✓✓✓✓", self.out)

    def test_requiring_a_real_device_flips_every_consequence_bit(self):
        self.assertIn("判定: {'L1': True, 'L2': True, 'L3': False}", self.out)
        self.assertIn("validity=uninformative", self.out)
        self.assertIn("l3_pass_refused=True", self.out)
        self.assertIn("资格掩码 V T P R C: ·····", self.out)
        self.assertIn("只有契约多了 required_caps", self.out)

    def test_the_route_vs_eligibility_seam_is_shown_not_hidden(self):
        """本轮实测撞出来的接缝: 处方会说「去搜索」, 而资格说这一读数不能驱动后果。

        这条**不是**本演示的结论, 而是它发现的接缝; 演示必须把它印出来 ——
        否则 HITL 工具照处方走, 就把一个 uninformative 读数喂进了搜索。
        """
        self.assertIn("接缝: 处方说", self.out)
        self.assertIn("必须与资格掩码并列", self.out)

    # ── 场景 5: Triton 只到 L1 ────────────────────────────────────────────
    def test_triton_source_stops_at_l1_here(self):
        self.assertIn("候选 C（Triton 源，stop_after=L1）", self.out)
        self.assertIn("判定: {'L1': True}\n", self.out,
                      "Triton 那一段不该出现 L2/L3 的判定")

    # ── 场景 6: 报告产物 ──────────────────────────────────────────────────
    def test_the_report_is_written_and_names_the_missing_capability(self):
        report = demo.RUN_ROOT / "run" / "report.md"
        self.assertTrue(report.is_file(), f"报告没写出来: {report}")
        text = report.read_text(encoding="utf-8")
        self.assertIn("# 自动跑报告", text)
        self.assertIn("我无法确立的事", text)
        self.assertIn("real_device", text,
                      "报告没点名缺的那个能力 —— 那「拒评」就不可归因了")
        self.assertIn("uninformative", text)

    # ── 落点: 不许污染仓库 ────────────────────────────────────────────────
    def test_every_artifact_lands_under_a_temp_dir(self):
        run_root = str(demo.RUN_ROOT)
        self.assertTrue(run_root.startswith(tempfile.gettempdir()), run_root)
        for name in ("run/report.md", "candidate-A.log", "candidate-B.log",
                     "candidate-B-refused.log", "candidate-C.log"):
            path = demo.RUN_ROOT / name
            self.assertTrue(path.is_file(), f"{name} 不在演示的临时根里: {path}")
        # 四个落点变量**不许指向仓库**（指向哪儿无所谓：演示用 setdefault，测试环境已重定向）
        for key in _ENV_KEYS:
            value = os.environ.get(key) or ""
            self.assertFalse(value.startswith(str(ROOT)),
                             f"{key} 指进了仓库: {value}")

    def test_the_three_process_logs_are_kept(self):
        for name in ("candidate-A.log", "candidate-B.log",
                     "candidate-B-refused.log", "candidate-C.log"):
            path = demo.RUN_ROOT / name
            self.assertTrue(path.is_file(), f"{name} 没留档")
            self.assertGreater(len(path.read_text(encoding="utf-8")), 0, name)


if __name__ == "__main__":
    unittest.main()
