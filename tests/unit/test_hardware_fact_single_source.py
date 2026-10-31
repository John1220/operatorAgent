"""硬件事实的单一真源与对照守卫。

起因（两个真实形态，不是假想）：

1) **一份数字抄在两处，靠手抄维持一致**。同一张卡的 (FP32/TF32/FP16/BF16, 带宽) 曾同时
   写在 hardware_caps._KNOWN_HARDWARE 与 verifier._PEAK_BY_NAME 里，5 张卡 × 5 个数
   逐位相同 —— 而 hardware_caps 自己的注释写着「与 verifier._PEAK_BY_NAME["h20"] 同一口径」，
   一句**没有守卫的口头约定**。只改一处，同一台机器的 roofline 分母与 caps 带宽就悄悄
   不同源，而没有任何东西会红。现在共有的行由 hardware_caps.spec_row() 派生。
   ⚠️ 注意：这不是「合并两张表」。它们的**成员**刻意不同（见下面两组集合），因为补一条
   hardware_caps 条目要连 launch 地板 / smem / 能力位一起补齐，那是实测活，不臆造。

2) **「出处就是可信度」这套机制此前没有完备性守卫**。verifier 用 _PEAK_SOURCE 的字符串
   自己承担每行可信度（以 ⚠️ 开头 ⇒ 口径未核 ⇒ 不许拿它的达成率判停），可是「某一行
   没写出处」这件事没人管 —— 新加一行忘了写，它就会静默拿到 _PEAK_SOURCE_DEFAULT 而不
   判停，或者被当成可信行。本文件把「每行都得出处」变成机械断言。

运行: python -m pytest tests/unit/test_hardware_fact_single_source.py -q
"""
import ast
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from operator_agent.core import hardware_caps as hc
from operator_agent.decide import verifier as v
from operator_agent.core.hardware_caps import _KNOWN_HARDWARE, HardwareCaps

VERIFIER_SRC = REPO / "operator_agent" / "decide" / "verifier.py"

#: name 行里必须由 spec_row 派生的型号（两张表都有的卡）
NAME_DERIVED = ("a100", "h100", "h20", "v100", "3090")
#: cc 行里能 1:1 映射到某个型号的（cc 键 -> 型号）
CC_DERIVED = {"8.0": "a100", "9.0": "h100", "7.5": "3090", "7.0": "v100"}
#: cc 行里**没有**对应型号、只能留在 verifier 本地的（8.6 = A100 40GB/A40, 8.9 = Ada/L40S）
CC_LOCAL = {"8.6", "8.9"}
#: 只有 hardware_caps 知道的型号：都不是 NVIDIA 卡（或本就是假件），roofline 表按规格页
#: 组织，本来就不收录。加卡时这里必须一起动 —— 逼一次有意的决定。
CAPS_ONLY = {"ascend910b", "mi250", "mock"}
#: 只有 verifier 知道的型号：hardware_caps 里没有条目。补上它们要连 launch_floor_us /
#: shared_mem_per_sm_kb / 能力位一起补齐，缺实测就是臆造（docs/research/49 §八 第 5 条）。
PEAK_ONLY = {"4090", "a5000", "a6000", "h200", "t4"}


def _spec_of(caps: HardwareCaps):
    """hardware_caps 条目 -> verifier 的 roofline 分母形状"""
    return (caps.fp32_tflops, caps.tf32_tflops, caps.fp16_tflops,
            caps.bf16_tflops, caps.global_mem_bandwidth_gbs)


def _divergences(peak, caps) -> list:
    """两张表共有的卡里，五元组不相等的那些（返回可读描述）。允许传入替身以便反向标定。"""
    out = []
    for model in sorted(set(peak) & set(caps)):
        spec = caps[model] if isinstance(caps[model], tuple) else _spec_of(caps[model])
        if peak[model] != spec:
            out.append("%s: peak=%s caps=%s" % (model, peak[model], spec))
    return out


def _table_rows(source: str, table: str) -> dict:
    """从源码里取出 table 参数指定的那个 dict 字面量的 {键: 值节点}（取不到返回 {}）"""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            target, value = node.targets[0].id, node.value
        else:
            continue
        if target == table and isinstance(value, ast.Dict):
            return {k.value: val for k, val in zip(value.keys, value.values)
                    if isinstance(k, ast.Constant)}
    return {}


def _derivation_errors(source: str) -> list:
    """返回「本该派生却写了字面量」的行描述。

    判据是 **AST 形状**而不是数值相等：一字不差地抄回来也会被这里抓住 —— 数值相等
    恰恰是最难分辨的那种漂移（改一处之前，两份就是相等的）。
    """
    bad = []
    for table, expected in (("_PEAK_BY_NAME",
                             {m: m for m in NAME_DERIVED}),
                            ("_PEAK_BY_CC", dict(CC_DERIVED))):
        rows = _table_rows(source, table)
        for key, model in expected.items():
            node = rows.get(key)
            if node is None:
                bad.append("%s[%r] 不存在了（守卫要跟着更新）" % (table, key))
                continue
            is_call = (isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name)
                       and node.func.id == "spec_row"
                       and len(node.args) == 1
                       and isinstance(node.args[0], ast.Constant)
                       and node.args[0].value == model)
            if not is_call:
                bad.append("%s[%r] 没有走 spec_row(%r) —— 又抄了一份字面量"
                           % (table, key, model))
    return bad


class TestTheNumbersHaveOneSource(unittest.TestCase):

    def test_shared_cards_agree_with_the_caps_table(self):
        self.assertEqual([], _divergences(v._PEAK_BY_NAME, _KNOWN_HARDWARE))
        self.assertEqual([], _divergences(v._PEAK_BY_CC,
                                          {k: _spec_of(_KNOWN_HARDWARE[m])
                                           for k, m in CC_DERIVED.items()}))

    def test_shared_rows_are_derived_not_copied(self):
        """数值相等不算数：必须是 spec_row 调用，抄回来同样红"""
        self.assertEqual([], _derivation_errors(VERIFIER_SRC.read_text(encoding="utf-8")))

    def test_spec_row_refuses_a_model_the_table_does_not_know(self):
        """「不知道」不许折成一个能当分母的数"""
        with self.assertRaises(KeyError):
            hc.spec_row("h200")
        with self.assertRaises(KeyError):
            hc.spec_row("no_such_card")


class TestTheCoverageAsymmetryIsPinned(unittest.TestCase):
    """成员不对称是**刻意的**；谁来加卡，就必须来这里改一行"""

    def test_caps_only_and_peak_only_sets_are_unchanged(self):
        caps_keys, peak_keys = set(_KNOWN_HARDWARE), set(v._PEAK_BY_NAME)
        self.assertEqual(CAPS_ONLY, caps_keys - peak_keys,
                         "hardware_caps 多认/少认了卡 —— 请连同本文件的清单一起改")
        self.assertEqual(PEAK_ONLY, peak_keys - caps_keys,
                         "verifier 多认/少认了卡 —— 补一条 caps 条目要实测, 别顺手编数")

    def test_local_cc_rows_are_only_the_ones_without_a_model(self):
        self.assertEqual(CC_LOCAL, set(v._PEAK_BY_CC) - set(CC_DERIVED),
                         "cc 表多出/少了没有对应型号的行 —— 守卫要跟着更新")


class TestEveryRowDeclaresItsSource(unittest.TestCase):
    """_PEAK_SOURCE 的字符串自己承担可信度（⚠️ 开头 ⇒ 口径未核 ⇒ 不许判停）,
    所以「某行没写出处」必须是机械可见的，而不是等人去发现。"""

    def test_no_peak_row_is_missing_a_source(self):
        self.assertEqual([], sorted(set(v._PEAK_BY_NAME) - set(v._PEAK_SOURCE)))
        self.assertEqual([], sorted(set(v._PEAK_BY_CC) - set(v._PEAK_CC_SOURCE)))

    def test_no_source_is_left_over_from_a_deleted_row(self):
        self.assertEqual([], sorted(set(v._PEAK_SOURCE) - set(v._PEAK_BY_NAME)))
        self.assertEqual([], sorted(set(v._PEAK_CC_SOURCE) - set(v._PEAK_BY_CC)))

    def test_the_unverified_marker_is_where_we_say_it_is(self):
        """以 ⚠️ 开头的行 = 口径未核。这份名单是「别拿它判停」的机械载体。"""
        unverified = {k for k, s in v._PEAK_SOURCE.items() if s.startswith("⚠️")}
        self.assertEqual({"3090", "4090", "a6000", "a5000", "t4"}, unverified)


class TestTheGuardIsNotANoOp(unittest.TestCase):
    """反向标定：每条守卫都要能变红"""

    def test_divergence_check_catches_a_drifted_card(self):
        """改动一处而不改另一处 —— 这正是守卫要抓的形态。

        ⚠️ 这里**不许写死具体的带宽数字**。第一版写死了 4000/4100, 于是标定
        「把 caps 的 h20 带宽改成 4100」时这条反向标定**自己先红了** —— 反向标定依赖
        它本要去扰动的那份数据, 就等于没有标定。2026-10-07 由变异标定抓到并改掉。
        """
        original = _spec_of(_KNOWN_HARDWARE["h20"])
        caps = dict(_KNOWN_HARDWARE)
        caps["h20"] = original[:4] + (original[4] + 100.0,)
        divs = _divergences(v._PEAK_BY_NAME, caps)
        self.assertEqual(1, len(divs), "共有的卡里应当正好有 1 处不一致: %s" % divs)
        self.assertTrue(divs[0].startswith("h20:"), divs[0])
        self.assertIn(str(original[4] + 100.0), divs[0])

    def test_derivation_check_rejects_a_retyped_literal(self):
        """把派生改回字面量（数值一模一样）也必须红 —— 逐位相等躲得过数值比较"""
        good = VERIFIER_SRC.read_text(encoding="utf-8")
        self.assertEqual([], _derivation_errors(good))
        bad = good.replace('"h20":  spec_row("h20")',
                           '"h20":  (39.5,  74.0,  148.0, 148.0, 4000.0)')
        self.assertNotEqual(good, bad, "替换没生效 —— 反向标定要跟着更新")
        errors = _derivation_errors(bad)
        self.assertTrue(any("h20" in e and "spec_row" in e for e in errors),
                        "抄回字面量没被抓到 ⇒ 这条守卫是永真的: %s" % errors)

    def test_source_completeness_check_catches_a_missing_row_source(self):
        """模拟「新加一行忘了写出处」"""
        peak = dict(v._PEAK_BY_NAME)
        peak["brand_new_card"] = (1.0, 1.0, 1.0, 1.0, 1.0)
        self.assertEqual(["brand_new_card"],
                         sorted(set(peak) - set(v._PEAK_SOURCE)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
