"""可达上界聚合器 (doc 51 §五 A 片) —— 分母只能从这台卡自己的读数里长出来

三条判据各自钉一条纪律, 每条都有绝对锚点 (只有差分/存在性断言时两侧可以一起漂):

1. 一个曲线点 = `gflops / arithmetic_intensity` (AI = FLOPs/Bytes ⇒ Byte/s)。锚点是
   手算得出来的 875 GFLOPS ÷ 0.875 = **1000.0 GB/s**, 不是从模块里读出来的。
2. **只有 `valid` 档能定义天花板** —— doc 50 §7.4 那批全是 `uninformative` (launch 地板
   占 70~72%), 拿它们当上限会把派发开销当成「这台卡的可达速率」。被排除的条数照样要报:
   「拒了 0 条」与「压根没被问」不许长得一样 (doc 48 的入库名单同源那条)。
3. **测不到 ≠ 0**。样本不够或格子为空时打印「测不到」, 不打印一个看着像数的 0.0。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from operator_agent.core.achievable import (MIN_SAMPLES_FOR_PCT, Cell, Key, collect,
                                      points_from_rounds, report)


def axes(**kw):
    base = {"arithmetic_intensity": 0.875, "compute_used": "fp32_cuda_core",
            "hw_compute_capability": "9.0", "peak_trusted": True}
    base.update(kw)
    return base


def rnd(gflops=875.0, validity="valid", roofline=None, kernel_type="layernorm"):
    r = {"round": 1, "gflops": gflops, "measurement_validity": validity,
         "kernel_type": kernel_type}
    if roofline is not None:
        r["roofline"] = roofline
    return r


def write_artifact(root: Path, rounds, task_kernel="layernorm"):
    d = root / task_kernel / "nvidia_gpu" / f"2026{name_suffix()}"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(
        {"task_id": "t", "task": {"source": {"kernel_type": task_kernel}},
         "iteration_history": list(rounds)}), encoding="utf-8")


_suffix = [0]


def name_suffix():
    _suffix[0] += 1
    return f"dir{_suffix[0]}"


class OnePointIsRateOverIntensity(unittest.TestCase):
    def test_known_answer_bandwidth_is_gflops_over_ai(self):
        pts = points_from_rounds([rnd(gflops=875.0, roofline=axes())])
        self.assertEqual(1, len(pts))
        key, bw, gf, validity = pts[0]
        self.assertAlmostEqual(1000.0, bw, places=6)   # 875 / 0.875, 手算
        self.assertEqual(875.0, gf)
        self.assertEqual("valid", validity)
        self.assertEqual(0.875, key.ai)
        self.assertEqual("9.0", key.hardware)
        self.assertEqual("layernorm", key.kernel_type)

    def test_a_round_without_axes_is_not_a_zero_rate_point(self):
        """缺键 ≠ 零值: 没走到 L3 的轮不该产出一个速率为 0 的曲线点。"""
        self.assertEqual([], points_from_rounds([rnd(roofline=None)]))
        self.assertEqual([], points_from_rounds([rnd(roofline={})]))
        # AI 缺失/为 0 也不能产点 —— 除法分母为零时会造出一个 inf 或 0 的「上界」
        self.assertEqual([], points_from_rounds([rnd(roofline=axes(arithmetic_intensity=0))]))
        self.assertEqual([], points_from_rounds([rnd(gflops=None, roofline=axes())]))

    def test_parser_is_not_vacuous(self):
        """解析型代码要自证不空转: 一边是空, 一边必须真的认出了东西。"""
        self.assertEqual([], points_from_rounds([]))
        pts = points_from_rounds([rnd(roofline=axes()), rnd(gflops=1750.0, roofline=axes())])
        self.assertGreaterEqual(len(pts), 2)
        self.assertAlmostEqual(2000.0, pts[1][1], places=6)   # 1750 / 0.875


class ValidityGateIsNotSilent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        _suffix[0] = 0

    def test_uninformative_points_are_excluded_but_counted(self):
        write_artifact(self.root, [rnd(roofline=axes(), validity="uninformative"),
                                   rnd(roofline=axes(), validity="degraded"),
                                   rnd(roofline=axes(), validity="valid")])
        cells, stats = collect(self.root)
        cell = next(iter(cells.values()))
        self.assertEqual(1, cell.n, "只有 valid 那一条能进天花板")
        self.assertAlmostEqual(1000.0, cell.peak_bandwidth_gbs(), places=3)
        self.assertEqual({"uninformative": 1, "degraded": 1}, cell.excluded)
        self.assertEqual(2, stats["invalid_points"], "被排除的条数必须报出来")

    def test_no_trusted_sample_says_cannot_measure_not_zero(self):
        """这是这一片最重要的输出形态: doc 50 那种全 uninformative 的机时, 答案必须是「测不到」。"""
        write_artifact(self.root, [rnd(roofline=axes(), validity="uninformative")])
        cells, stats = collect(self.root)
        text = "\n".join(report(cells, stats))
        self.assertIn("测不到", text)
        self.assertNotIn("峰值有效带宽", text)
        self.assertNotIn("0.0 GB/s", text)

    def test_artifacts_without_axes_are_counted_as_missing_not_read_as_zero(self):
        write_artifact(self.root, [rnd(roofline=None), rnd(roofline=None)])
        cells, stats = collect(self.root)
        self.assertEqual(2, stats["no_axes"])
        self.assertEqual({}, cells)
        self.assertIn("一个格子都没有", "\n".join(report(cells, stats)))

    def test_small_sample_refuses_to_print_any_percentage(self):
        rounds = [rnd(gflops=875.0, roofline=axes()), rnd(gflops=900.0, roofline=axes()),
                  rnd(gflops=950.0, roofline=axes())]
        self.assertEqual(3, len(rounds))
        self.assertLess(len(rounds), MIN_SAMPLES_FOR_PCT)
        write_artifact(self.root, rounds)
        cells, stats = collect(self.root)
        text = "\n".join(report(cells, stats))
        self.assertIn(f"样本 3 < {MIN_SAMPLES_FOR_PCT}", text)
        self.assertNotIn("散布", text, "没到门槛就不许有百分数 —— 门槛本身要能被移动测出来")

    def test_report_line_carries_the_known_amplitude(self):
        """M7 第一次逃逸暴露的洞: 门槛之上的那条**数字本身**从没被断言过。

        变异「把打印的带宽写死成 0.0」在 14 条用例下全绿 —— 因为前面几条只断言了
        「散布出现/不出现」这种**形态**, 没断言**幅度**。这正 trap #1 的同族:
        量具能看见格子存在, 却看不见格子里的数是错的。
        """
        rounds = [rnd(gflops=g, roofline=axes()) for g in (875.0, 900.0, 950.0, 1000.0)]
        write_artifact(self.root, rounds)
        cells, stats = collect(self.root)
        text = "\n".join(report(cells, stats))
        # 1000 GFLOPS / 0.875 = 1142.857 GB/s —— 峰值取四条里最大的那条
        self.assertIn("1142.9 GB/s", text)
        self.assertIn("n=4", text)

    def test_crossing_the_threshold_unlocks_the_spread(self):
        """门槛的反向对照: 加一条样本, 百分数就该出现。少了这条, 门槛写死成 0 也没人会发现。"""
        rounds = [rnd(gflops=g, roofline=axes()) for g in (875.0, 900.0, 950.0, 1000.0)]
        write_artifact(self.root, rounds)
        cells, stats = collect(self.root)
        text = "\n".join(report(cells, stats))
        self.assertIn("散布", text)
        self.assertNotIn("拒绝打印", text)
        self.assertEqual(1, len(cells))


class TheL2BandIsHonestAboutWhatIsMissing(unittest.TestCase):
    def test_today_the_band_is_unknown_because_the_axes_lack_bytes(self):
        """如实标注: 现有 `roofline` 块里没有 `bytes_transferred` / `l2_cache_size_kb`。

        这条用例现在绿是**因为**它承认缺; 一旦 verifier 加上那两个键, 下一条用例会自动接管。
        """
        pts = points_from_rounds([rnd(roofline=axes())])
        self.assertEqual("unknown", pts[0][0].l2_band)

    def test_new_keys_flow_from_persistence_without_touching_the_aggregator(self):
        """不挑键那条决定的**回报** (doc 48 白名单的反面)。

        `_round_axes()` 整块复制 `roofline_analysis`, 所以提供方将来在同一个块里加
        `bytes_transferred` / `l2_cache_size_kb`, 这里就立刻能分档 —— 不需要再动落盘层。
        """
        fits = points_from_rounds([rnd(roofline=axes(bytes_transferred=1024,
                                                    l2_cache_size_kb=61440))])[0][0]
        exceeds = points_from_rounds([rnd(roofline=axes(bytes_transferred=61440 * 1024 * 3,
                                                        l2_cache_size_kb=61440))])[0][0]
        self.assertEqual("fits_l2", fits.l2_band)
        self.assertEqual("exceeds_l2", exceeds.l2_band)

    def test_missing_one_side_is_not_guessed_as_fits(self):
        from operator_agent.core.achievable import _l2_band
        self.assertEqual("unknown", _l2_band(None, 61440))
        self.assertEqual("unknown", _l2_band(1024, None))
        self.assertEqual("unknown", _l2_band(0, 0), "0 不是读数, 不许据此定档")


class CellArithmeticIsThreeValued(unittest.TestCase):
    def test_spread_is_none_not_zero_when_there_is_one_sample(self):
        c = Cell(key=Key("9.0", "fp32", "layernorm", 0.875, "unknown"), bandwidth_gbs=[1000.0])
        self.assertIsNone(c.spread_pct())
        self.assertEqual(1000.0, c.peak_bandwidth_gbs())

    def test_spread_of_known_amplitude(self):
        c = Cell(key=Key("9.0", "fp32", "layernorm", 0.875, "unknown"),
                 bandwidth_gbs=[1000.0, 1000.0, 1000.0, 1200.0])
        # 中位取排序后的中间位 (第 3 个) = 1000; (1200-1000)/1000 = 20%
        self.assertAlmostEqual(20.0, c.spread_pct(), places=6)

    def test_unreadable_result_json_is_counted_not_silently_skipped(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        d = root / "layernorm" / "nvidia_gpu" / "x"
        d.mkdir(parents=True)
        (d / "result.json").write_text("{这不是 JSON", encoding="utf-8")
        cells, stats = collect(root)
        self.assertEqual(1, stats["unreadable_files"])
        self.assertEqual(0, stats["result_files"])
        self.assertIn("读不动的文件 1", "\n".join(report(cells, stats)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
