"""配方条件「写得出就必须解析得出」的防腐守卫 (#66)。

起因是数出来的，不是审出来的：做 doc 49 §八 第 17 条那片时机械扫配方库，发现有 3 条在
`applies_when` 里写了利用率阈值（`gflops_util < 80%` / `< 50%` / `< 70%`），而
`_parse_conditions` 的入口守卫当时要的是字面量 `"utilization"` ⇒ 后面那条正则明明认得
`gflops_util < 80%`，却被守卫挡在外面 ⇒ 解析出阈值的配方数 = **0**，这 3 条今天只被
`compute_bound` 挡着。失效方式与「恒假」那批同族，但更坏：**条件不存在，于是连判都不判**，
而日志与报表里完全看不出来。

所以这里放的是**守卫而不是清单**（doc 49 §7.17 那条教训：清单会过期，守卫会在有人写新
拼法时变红）：

* 正向 —— 只要 `applies_when` 里写了 `util…< 数字`，那条配方的 `_conditions` 里就必须有阈值；
* 反向 —— 不许有「文本里没写却解析出阈值」的配方（那是解析器在替配方编条件）。

另含每个阈值的**两侧**取样（阈值 80：79.9 必须入选、80.0 必须被拒）——只在成立的一侧取样
时，把整个条件删掉也不会被看见，那正是这片自己被教过一次的事。

运行: python3 -B -m unittest tests.test_recipe_condition_parsing -v
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from operator_agent.hardware_caps import _KNOWN_HARDWARE, detect_hardware_caps  # noqa: E402
from operator_agent.optimization_planner import (  # noqa: E402
    OptimizationPlanner, _axis_field,
)

#: 文本里写出来的 util/bandwidth 阈值。按**整个标识符**匹配再分派, 不靠 lookbehind ——
#: 第一版写了 `(?<!\w)util\w*`，而配方写的是 `gflops_util`（前面是下划线，属于 `\w`），
#: 于是这条守卫**一个都没匹配上**，正向那条断言就变成恒真的空转 (doc 49 §八 那条
#: 「守卫的解析面窄于被测代码」的第四次出现)。
_TOKEN_THRESHOLD = re.compile(r"([a-z_]*util[a-z_]*)\s*<\s*(\d+)")


def _written_thresholds(cond_text: str) -> tuple:
    """从 `applies_when` 文本里数出 (util 阈值, bandwidth 阈值)，没写就是 0。

    ⚠️ 取**第一个**：生产那两条 `re.search` 也只取第一个，所以一条配方里写两个
    利用率阈值时，第二个是不作数的。今天没有这种写法；真有人写了，这条守卫会报
    「文本与解析不一致」，那时要做的决定是「允不允许一条配方有两个 util 阈值」，
    而不是把守卫改成跟着取第一个（那等于让守卫去适应实现）。
    """
    util = bw = 0
    for name, num in _TOKEN_THRESHOLD.findall((cond_text or "").lower()):
        if "bandwidth" in name:
            if not bw:
                bw = int(num)
        elif not util:
            util = int(num)
    return util, bw


def _library():
    p = OptimizationPlanner()
    return p, p._all_recipes


class TestWrittenConditionsAreParsed(unittest.TestCase):
    """防腐守卫的正反两面。"""

    def test_every_written_util_threshold_is_parsed(self):
        _p, recipes = _library()
        missed, checked = [], 0
        for r in recipes:
            want, _bw = _written_thresholds(r.applies_when)
            if not want:
                continue
            checked += 1
            if not r._conditions.get("low_utilization"):
                missed.append((r.id, r.applies_when, want))
        # 这条是防「守卫自己空转」的: 匹配不到任何写出来的阈值时, 上面那个断言恒真。
        self.assertGreaterEqual(checked, 3,
                                f"解析器只认出 {checked} 条写出来的阈值 —— 少于 3 条说明"
                                f"这条守卫的输入面已经窄过配方了")
        self.assertEqual([], missed,
                         f"这些配方写了利用率阈值却没被解析出来 (条件不存在 ⇒ 连判都不判): {missed}")

    def test_no_recipe_gains_a_threshold_it_did_not_write(self):
        """反向: 解析器不许替配方编条件。"""
        _p, recipes = _library()
        invented = []
        for r in recipes:
            want, want_bw = _written_thresholds(r.applies_when)
            got = r._conditions.get("low_utilization") or 0
            got_bw = r._conditions.get("low_bandwidth_util") or 0
            if got != want or got_bw != want_bw:
                invented.append((r.id, r.applies_when, (want, want_bw), (got, got_bw)))
        self.assertEqual([], invented,
                         f"文本与解析出来的阈值不一致 (写 X 得 Y 就是解析器在替配方编条件): {invented}")

    def test_the_guard_is_not_vacuous(self):
        """被这条守卫救起来的配方必须**确实存在** —— 若整个库一条阈值都没有，
        上面两例就只是在自我确认（与「守卫的解析面窄于被测代码」那一族同源）。"""
        _p, recipes = _library()
        with_util = [r for r in recipes if r._conditions.get("low_utilization")]
        self.assertGreaterEqual(len(with_util), 3,
                                f"今天真实库里带利用率阈值的配方少于 3 条: {[r.id for r in with_util]}")
        # 绝对锚点: 这三个数字是从配方 markdown 里读出来的，写死在这里是为了让
        # 「解析器抓错数字」这种失效有东西接住（差分守卫两侧可以一起漂）。
        self.assertEqual({"R-MATMUL-04": 80, "R-MATMUL-07": 50, "R-MATMUL-14": 70},
                         {r.id: r._conditions["low_utilization"] for r in with_util})

    def test_spelling_variants_all_reach_the_same_condition(self):
        parse = OptimizationPlanner._parse_conditions
        for text, want in (("gflops_util < 80%", 80),
                           ("utilization < 50", 50),
                           ("bandwidth_utilization_pct < 30", 0),   # 那一族不算 util 这条
                           ("memory_bound", 0),
                           ("l2_cache_size_kb > 0", 0)):
            self.assertEqual(want, parse(text)["low_utilization"],
                             f"{text!r} 的 low_utilization 应该是 {want}")
        # 带宽那一族必须被接住, 且不能串到 util 上 (否定先行断言)
        bw = parse("bandwidth_utilization < 30")
        self.assertEqual(30, bw["low_bandwidth_util"])
        self.assertEqual(0, bw["low_utilization"])


class TestThresholdFiltersOnBothSides(unittest.TestCase):
    """每个阈值取**两侧**：阈值下面一点必须入选，达到阈值必须被拒。

    这片自己就被这一点坑过一次：只喂 30%（所有阈值以下）的话，把 `_check_conditions`
    里那两个 `low_utilization` 判断整个删掉也不会有任何用例变红。
    """

    #: 一条可信的 compute-bound 轴: 门认它, 于是 bound 条件与 util 条件都真的生效。
    AXIS = {"bound_type": "compute", "roofline_bound": "compute", "peak_trusted": True}

    def _selected(self, hw_name, gflops_util):
        caps = _KNOWN_HARDWARE[hw_name]
        p = OptimizationPlanner()
        plan = p.plan(kernel_type="matmul", dsl="triton", hw_caps=caps,
                      roofline=dict(self.AXIS, gflops_utilization_pct=gflops_util),
                      precision="fp16", round_num=3)
        return {r.id for r in plan.selected_recipes}, {
            r.id: why for r, why in plan.rejected_recipes}

    def test_boundary_of_each_written_threshold(self):
        cases = [("R-MATMUL-04", 80), ("R-MATMUL-07", 50), ("R-MATMUL-14", 70)]
        for rid, thresh in cases:
            below, _ = self._selected("h20", thresh - 0.1)
            at, rej = self._selected("h20", float(thresh))
            self.assertIn(rid, below, f"{rid} 在 util={thresh - 0.1} (<{thresh}) 应该入选")
            self.assertNotIn(rid, at, f"{rid} 在 util={thresh} (>= {thresh}) 应该被拒")
            self.assertIn("conditions not met", rej[rid],
                          f"拒绝理由要留得下来, 否则下次没人知道是谁把它挡的: {rej.get(rid)}")

    def test_the_high_util_case_actually_loses_recipes(self):
        """整库层面的效应: util 打满时, 带阈值的 3 条都不该在计划里。
        这条是「修完之后确实少选了配方」的证据 —— 修之前它也是绿的 (因为那 3 条没有阈值),
        所以它单独不鉴别, 必须与上面那条边界用例**一起**读。"""
        _low, _ = self._selected("h20", 10.0)
        high, _ = self._selected("h20", 99.0)
        for rid in ("R-MATMUL-04", "R-MATMUL-07", "R-MATMUL-14"):
            self.assertNotIn(rid, high)
        self.assertTrue(high, "util 打满时一个配方都不剩 —— 那不是筛配方, 那是清空计划")

    def test_unverifiable_axis_voids_the_util_condition_too(self):
        """与 doc 49 §八 第 17 条那一片接得上: 分母没核实时, 这些刚生效的阈值条件
        **也不许**替配方做否定判断 —— 否则那片只挡住了 bound 条件, 利用率这条门照旧跑。"""
        caps = _KNOWN_HARDWARE["h20"]
        p = OptimizationPlanner()
        guess = {"bound_type": "compute", "roofline_bound": "unknown",
                 "peak_trusted": False, "gflops_utilization_pct": 99.0}
        plan = p.plan(kernel_type="matmul", dsl="triton", hw_caps=caps,
                      roofline=guess, precision="fp16", round_num=3)
        sel = {r.id for r in plan.selected_recipes}
        for rid in ("R-MATMUL-04", "R-MATMUL-07", "R-MATMUL-14"):
            self.assertIn(rid, sel, f"轴不可用时 {rid} 还在被一个未核分母除出来的百分比拒掉")

    def test_round_one_still_gets_the_util_unknown_treatment(self):
        """首轮没有块: `_axis_field(...)` 读不到任何百分比 ⇒ 视为 0 ⇒ 这些条件不参与。
        这条钉的是「缺读数」与「读数等于 0」在代码里是同一回事, 与那片「0 就是没有这个
        读数」的注释对齐 —— 有人把默认值改成 None 时这里会红 (比较会抛 TypeError)。"""
        caps = _KNOWN_HARDWARE["h20"]
        p = OptimizationPlanner()
        plan = p.plan(kernel_type="matmul", dsl="triton", hw_caps=caps,
                      roofline=None, precision="fp16", round_num=1)
        self.assertIsNone(_axis_field(plan.roofline, "gflops_utilization_pct"))
        self.assertFalse(plan.axis["usable"])
        sel = {r.id for r in plan.selected_recipes}
        self.assertIn("R-MATMUL-04", sel)

    def test_the_hardware_in_these_cases_is_the_production_path(self):
        """用例里的硬件必须与生产同一条路径认出来, 不是测试自己造的第三张表
        (doc 49 §八 第 5 条: 七张硬件表还没合并, 再造一张就等于给分叉添一刀)。"""
        probed = detect_hardware_caps({"name": "NVIDIA H20", "vendor": "nvidia",
                                       "device": "cuda", "compute_capability": "9.0"})
        table = _KNOWN_HARDWARE["h20"]
        self.assertEqual(table.global_mem_bandwidth_gbs, probed.global_mem_bandwidth_gbs)
        self.assertEqual(table.sm_count, probed.sm_count)
        self.assertTrue(probed.has_tensor_core,
                        "认成 H20 却没 Tensor Core 能力 ⇒ 这些 compute_bound 配方本来就不该选上")


if __name__ == "__main__":
    unittest.main(verbosity=2)
