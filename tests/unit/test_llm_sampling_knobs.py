"""报表 knobs 少记的那一维 (#98) —— 记的必须是**真发出去的那一份**，不是构造参数

外部一手 (arXiv:2609.10226v1 行 306 / 319-320): Φ-Bench 把「单卡固定 + 禁用其他卡型 +
reasoning effort 统一 max」整条钉死写进实验设置，理由是 **effort 与分数不单调**、低档能让
一个模型掉 45% 的分。对照我方：`benchmarks/drivers/harness_pass_rate.py` 落盘的 knobs 六项里
**没有一项是模型侧的**，而请求体里带着 `temperature` 与 `chat_template_kwargs.enable_thinking`
—— 所有通过率数字都是在关掉思考的配置下测出来的，这个决定既不进 `git_head` 也不进 knobs。
报表自己写着「引用这些数字时必须带上 knobs」：它保护了配方侧，没保护模型侧。⇒ 这不是
「外部有一条好做法我们没抄」，是我方一条**声明与它记录的内容不一致**（doc 59 §四）。

核账时撞出来的第二件事决定了这一片的形状：**逐调用覆盖在这里是常态** —— Designer 0.3/2048、
Coder 0.1/40960、Fixer 0.1/1024、planner 0.2/2048。所以「把 `client.temperature` 印进 knobs」
会造出一个新的 #52（记录与事实不同源，而且印出来的那个数看着更权威）。这一片因此把读数
挂在**账本**上：`_call_model` 里一次性算出的局部值同时喂给 body、错误文案与 `record(...)`，
报表再从账本聚合。

变异标定 (每条只改一处，跑完立刻逐字节复原并断言 sha256 与改前一致。表的可执行形态在仓里:
`benchmarks/mut_specs/llm_sampling_knobs.py`，跑法是
`python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/llm_sampling_knobs.py`。
2026-09-30 实跑，基线 17 passed，**行为分母 12/12 拦住、结构分母 0/0**；
L 组打在 `execute/llm_client.py`，H 组打在 `harness_pass_rate.py`):
    L1 body 里的 temperature 不看逐调用覆盖            → 2 红 (红在 msg=「发出去」那一侧)
    L2 账本里的 temperature 另读一遍构造参数            → 2 红 (红在 msg=「记下来」那一侧)

⚠️ L1 与 L2 的**红名单逐条相同**，harness 因此报「复核归属」。核过：两行改的是同一对读者的两端
—— L1 改发出去的那个值，L2 改账本里的那个值；第一次跑时两条失败文本都是裸的 `0.1 != 0.3`，
归属读不出来，所以给断言补了 msg。归属应当能从读数看出来，不是从注释信它。
    L3 `enable_thinking` 写死 False (属性没人读)       → 1 红
    L4 聚合把 max_tokens 那一轴并掉了                  → 5 红
    L5 没记采样的条目折进组合 (而不是单独计数)             → 1 红
    L6 `summary()` 不发布 `sampling` 这个键            → 5 红，但其中**只有 2 条是证人**:
                                                        另 3 条是 `pretty()` 里 `s["sampling"]`
                                                        抛 KeyError 的连带 (另外两个文件各有
                                                        一条印账单的用例撞在同一处)
    L7 `pretty()` 不印采样那一段                       → 1 红
    L8 截断错误文案另算一遍 max_tokens                  → 1 红
    H1 `sampling_from_metadata` 缺账本时返回 `{}`      → 1 红
    H2 跨运行合并不累加 calls                          → 1 红
    H3 「一侧没记」的格子印成空串                       → 1 红  ⚠️ 第一次跑是**逃逸**, 见下
    H4 `format_comparison` 不印 knobs (算了没人读)     → 4 红

⚠️ H3 的第一次读数是一条**假绿**，而它本身就是这个文件要防的那类错：我的报警文案里写着
「(一侧『— 没记』也算不一致)」，而那条用例断言的是 `assertIn("— 没记", text)` ——
needle 出现在被测代码**自己的解释文案**里，于是格子印成空串时断言仍然被文案满足。修法两半：
文案里不再出现那个串，断言改打在**位置上** (`"→  B — 没记"`)。登记 CHECKLOG I32。
同一轮还有两条同类清理：`assertNotIn("= min_speedup: 5.0 ⚠️")` 那种 needle 在任何版本里都
不存在（永真断言，看着像覆盖），换成「逐键判定不连坐」这一真实性质。
"""
from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.drivers.harness_pass_rate import (  # noqa: E402
    MIN_DENOM, compare_runs, format_comparison, sampling_from_metadata,
    sampling_from_runs)
from operator_agent.execute import llm_client as lc  # noqa: E402
from operator_agent.execute.llm_client import (  # noqa: E402
    CostTracker, LLMClient, format_sampling_configs)

SRC = Path(__file__).resolve().parents[2] / "operator_agent" / "execute" / "llm_client.py"


# ───────────────────────── 假端点：只记录真发出去的 body ─────────────────────────

class _Resp:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ok_payload(model="Fake-M"):
    return {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model": model}


def _truncated_payload(model="Fake-M"):
    return {"choices": [{"message": {"content": None}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model": model}


class _Capturing:
    """假 `urlopen`: 把每一次真发出去的请求体记下来"""

    def __init__(self, payload=None):
        self.bodies = []
        self.payload = payload or _ok_payload()

    def __call__(self, req, timeout=None):
        self.bodies.append(json.loads(req.data.decode("utf-8")))
        return _Resp(self.payload)


def _client(payload=None, **kwargs):
    kwargs.setdefault("base_url", "http://fake.invalid/v1")
    kwargs.setdefault("model", "Fake-M")
    kwargs.setdefault("api_key", "")
    return LLMClient(**kwargs), _Capturing(payload)


class TestTheLedgerRecordsWhatWasSent(unittest.TestCase):
    """账本里那三个字段必须等于**这一次请求体里**的值"""

    def test_per_call_override_is_what_the_ledger_keeps(self):
        client, fake = _client(temperature=0.3, max_tokens=4096)
        with mock.patch.object(lc.urllib.request, "urlopen", fake):
            client.chat([{"role": "user", "content": "hi"}],
                        temperature=0.1, max_tokens=1024, caller_tag="Fixer")
        body = fake.bodies[0]["chat_template_kwargs"]
        rec = client.cost_tracker.records[0]
        self.assertEqual(0.1, fake.bodies[0]["temperature"],
                         msg="「发出去」那一侧: 请求体里的 temperature 不是逐调用的那个")
        self.assertEqual(1024, fake.bodies[0]["max_tokens"],
                         msg="「发出去」那一侧: 请求体里的 max_tokens 不是逐调用的那个")
        self.assertEqual(fake.bodies[0]["temperature"], rec.temperature,
                         msg="「记下来」那一侧: 账本里的 temperature 与请求体不同源")
        self.assertEqual(fake.bodies[0]["max_tokens"], rec.max_tokens,
                         msg="「记下来」那一侧: 账本里的 max_tokens 与请求体不同源")
        self.assertEqual(body["enable_thinking"], rec.enable_thinking)
        # 反向对照：账本里若印的是 client 的默认值，上面两条就都成了「恒等式自证」。
        # 这一条钉住的是「0.3 这个默认值确实没被记进来」。
        self.assertNotEqual(client.temperature, rec.temperature)

    def test_the_error_text_and_the_ledger_agree_on_max_tokens(self):
        """被截断时那句错误文案也报 max_tokens —— 它必须是**同一个**值，不是另算一遍"""
        client, fake = _client(payload=_truncated_payload(), max_tokens=4096)
        with mock.patch.object(lc.urllib.request, "urlopen", fake):
            with self.assertRaises(RuntimeError) as ctx:
                client.chat([{"role": "user", "content": "hi"}], max_tokens=64)
        self.assertIn("max_tokens=64", str(ctx.exception))
        self.assertEqual(64, client.cost_tracker.records[0].max_tokens)

    def test_enable_thinking_flows_from_the_constructor_into_both(self):
        client, fake = _client(enable_thinking=True)
        with mock.patch.object(lc.urllib.request, "urlopen", fake):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertIs(True, fake.bodies[0]["chat_template_kwargs"]["enable_thinking"])
        self.assertIs(True, client.cost_tracker.records[0].enable_thinking)

    def test_the_default_is_thinking_off_and_is_recorded_as_such(self):
        """反向对照：`False` 是**记下来的取值**，不是「这一项没记」。

        如果默认路径写成「不记」，报表就会把「我们关了思考」印成「不知道关没关」。
        """
        client, fake = _client()
        with mock.patch.object(lc.urllib.request, "urlopen", fake):
            client.chat([{"role": "user", "content": "hi"}])
        rec = client.cost_tracker.records[0]
        self.assertIs(False, rec.enable_thinking)
        self.assertIsNotNone(rec.enable_thinking)

    def test_body_and_ledger_read_the_same_local(self):
        """AST 守卫：body 与 record 不许各算一遍 `temperature or self.temperature`

        两处各写一遍就是两条会各自漂移的路径 (本仓那一族「值算出来死在搬运路上」的反面
        写法：值活着，但有两个副本)。行为用例核的是同一批请求的两端，改法只要**两端一起改**
        就看不见 —— 这条守卫补的正是那个缝。
        """
        tree = ast.parse(SRC.read_text(encoding="utf-8"))
        fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "_call_model"), None)
        self.assertIsNotNone(fn, "找不到 `_call_model` —— 这条守卫的前提已变")

        def body_value_for(key):
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                    names = [k.value if isinstance(k, ast.Constant) else None
                             for k in node.value.keys]
                    if "chat_template_kwargs" in names and key in names:
                        return node.value.values[names.index(key)]
            self.fail(f"body 里没有 `{key}` 这一项 —— 前提已变")

        def record_value_for(key):
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if kw.arg == key:
                            return kw.value
            self.fail(f"`record(...)` 没有传 `{key}=` ⇒ 账本里那一列永远是 None")

        for key, expect in (("temperature", "sent_temperature"),
                            ("max_tokens", "sent_max_tokens")):
            b, r = body_value_for(key), record_value_for(key)
            for node in (b, r):
                self.assertIsInstance(node, ast.Name,
                                      f"body/账本里的 `{key}` 不是同一个局部变量 "
                                      f"(写 {expect} 的那个)")
            self.assertEqual(expect, b.id)
            self.assertEqual(b.id, r.id, f"`{key}` 在 body 与账本读的不是同一个局部变量")


class TestTheAggregationRule(unittest.TestCase):
    """报表要的是「哪几种配置各跑了多少次」，不是一个标量"""

    def _rec(self, caller, temperature=None, max_tokens=None, thinking=False):
        self.tracker.record(model="Fake-M", prompt_tokens=1, completion_tokens=1,
                            usage_source="api_usage", caller_tag=caller,
                            temperature=temperature, max_tokens=max_tokens,
                            enable_thinking=thinking)

    def setUp(self):
        self.tracker = CostTracker()
        # 真发出去的三种配置：Coder 两种 (同 temp 不同 max_tokens)、Designer 一种
        for _ in range(3):
            self._rec("Coder", 0.1, 40960)
        for _ in range(2):
            self._rec("Coder", 0.1, 1024)
        self._rec("Designer", 0.3, 2048)

    def test_combos_not_cartesian_products(self):
        s = self.tracker.sampling_configs()
        coder = s["by_caller"]["Coder"]
        self.assertEqual(2, len(coder), "Coder 只有两种真实配置 —— 各轴取并集会配出四种")
        self.assertEqual([40960, 1024], [c["max_tokens"] for c in coder],
                         "排序得让主力配置在前 (calls 多者先)")
        self.assertEqual([3, 2], [c["calls"] for c in coder])
        self.assertEqual(6, s["n_records_with_sampling"])

    def test_records_without_sampling_are_counted_not_defaulted(self):
        """「没记」单独计数，既不折进组合也不当成「温度为 0」的那一档"""
        self.tracker.record(model="Fake-M", prompt_tokens=1, completion_tokens=1,
                            usage_source="api_usage", caller_tag="Legacy")
        s = self.tracker.sampling_configs()
        self.assertEqual(1, s["n_records_without_sampling"])
        self.assertNotIn("Legacy", s["by_caller"])
        self.assertEqual(6, s["n_records_with_sampling"])
        printed = "\n".join(format_sampling_configs(s))
        self.assertIn("另有 1 条调用没记采样配置", printed)

    def test_a_partially_missing_axis_prints_a_dash_not_none(self):
        self.tracker.record(model="Fake-M", prompt_tokens=1, completion_tokens=1,
                            usage_source="api_usage", caller_tag="Half",
                            temperature=0.7)
        lines = "\n".join(format_sampling_configs(self.tracker.sampling_configs()))
        self.assertIn("max_tokens=—", lines)
        self.assertNotIn("max_tokens=None", lines)

    def test_the_summary_publishes_the_key_the_harness_reads(self):
        """键名守卫：`sampling_from_metadata` 读的是 `sampling`，漂移了就静默变 None"""
        self.assertIn("sampling", self.tracker.summary())

    def test_pretty_actually_prints_it(self):
        """记了但没人印 = 「算了没人读」；一次普通 run 的读者只有这份日志"""
        text = self.tracker.pretty()
        self.assertIn("采样配置", text)
        self.assertIn("Coder: model=Fake-M temperature=0.1", text)


class TestHarnessKnobs(unittest.TestCase):
    """落盘与 A/B 那两侧：None 与空表是两件事，两侧的 knobs 要并排看见"""

    @staticmethod
    def _sampling(pairs):
        """造一份 `sampling_configs` 形状的东西 (pairs = [(caller, temp, max_tokens, 次数)])"""
        return {
            "n_records_with_sampling": sum(n for *_, n in pairs),
            "n_records_without_sampling": 0,
            "by_caller": {c: [{"model": "Fake-M", "temperature": t, "max_tokens": m,
                               "enable_thinking": False, "calls": n}
                              for (cc, t, m, n) in pairs if cc == c]
                          for c in {c for c, *_ in pairs}},
        }

    def test_absent_ledger_is_none_not_an_empty_table(self):
        self.assertIsNone(sampling_from_metadata(None))
        self.assertIsNone(sampling_from_metadata({}))
        self.assertIsNone(sampling_from_metadata({"llm_cost": {}}))
        got = sampling_from_metadata({"llm_cost": {"sampling": self._sampling([("Coder", 0.1, 40960, 3)])}})
        self.assertIn("Coder", got["by_caller"])

    def test_merge_sums_calls_across_runs(self):
        runs = [{"llm_sampling": self._sampling([("Coder", 0.1, 40960, 3)])},
                {"llm_sampling": self._sampling([("Coder", 0.1, 40960, 2)])},
                {"llm_sampling": self._sampling([("Designer", 0.3, 2048, 1)])}]
        merged = sampling_from_runs(runs)
        self.assertEqual(3, merged["n_runs_with_ledger"])
        self.assertEqual(0, merged["n_runs_without_ledger"])
        coder = merged["by_caller"]["Coder"]
        self.assertEqual(1, len(coder), "同一组合必须合并成一条, 每次运行各写一条会看着像两种配置")
        self.assertEqual(5, coder[0]["calls"])
        self.assertEqual(1, len(merged["by_caller"]["Designer"]))

    def test_runs_without_a_ledger_are_counted_separately(self):
        runs = [{"llm_sampling": self._sampling([("Coder", 0.1, 40960, 3)])},
                {"llm_sampling": None}]
        merged = sampling_from_runs(runs)
        self.assertEqual(1, merged["n_runs_with_ledger"])
        self.assertEqual(1, merged["n_runs_without_ledger"])

    def _cmp(self, knobs_a, knobs_b):
        """走**真的** `compare_runs` 再印: 手搓一份 cmp 字典会在 meta 形状漂移时仍然全绿"""
        runs = [{"case": "c", "ok": True, "final_all_passed": True,
                 "rounds": [{"round": 1, "levels_passed": {"L3": True}}]}
                for _ in range(MIN_DENOM)]
        cmp = compare_runs(runs, list(runs),
                           meta_a={"git_head": "A", "knobs": knobs_a},
                           meta_b={"git_head": "B", "knobs": knobs_b})
        return format_comparison(cmp)

    def test_the_ab_path_prints_both_knobs_and_flags_the_difference(self):
        """A/B 现场必须看见读数条件：报表那句警告此前没有读者"""
        text = self._cmp({"min_speedup": 5.0, "llm": self._sampling([("Coder", 0.1, 40960, 3)])},
                         {"min_speedup": 5.0, "llm": self._sampling([("Coder", 0.9, 40960, 3)])})
        self.assertIn("读数条件 (knobs)", text)
        self.assertIn("⚠️ llm:", text)
        self.assertIn("1 项读数条件不一致", text)
        self.assertIn("不能只当机制差读", text)
        # 逐键判定不连坐: 相同的 min_speedup 不许被 llm 那一项的差异拖进报警
        self.assertIn("= min_speedup: 5.0", text)

    def test_identical_knobs_are_marked_equal_not_warned(self):
        """反向对照：只在会报警那一侧取样，测不出判据方向"""
        knobs = {"min_speedup": 5.0, "repeats": 3}
        text = self._cmp(knobs, dict(knobs))
        self.assertIn("= min_speedup: 5.0", text)
        self.assertNotIn("⚠️", text.split("判决:")[0])

    def test_one_side_recording_nothing_is_not_read_as_agreement(self):
        """老报表产自 #98 之前 ⇒ 那一侧「没记」，不能算「两边一致」

        断言打在**格子的位置上** (`→ B …`)，不是打在解释文案里：变异 H3 第一次就是这么
        逃掉的 —— 文案写着「一侧『— 没记』也算不一致」，于是 `assertIn("— 没记")` 被
        文案自己满足，格子印成空串时全绿 (CHECKLOG I21 那一族)。
        """
        text = self._cmp({"min_speedup": 5.0, "llm": self._sampling([("Coder", 0.1, 40960, 3)])},
                         {"min_speedup": 5.0})
        self.assertIn("→  B — 没记", text)
        self.assertIn("⚠️ llm:", text)
        self.assertIn("= min_speedup: 5.0", text)

    def test_both_sides_without_knobs_say_so_instead_of_claiming_equality(self):
        text = self._cmp(None, None)
        self.assertIn("两份报表都没有", text)
        # 一份 knobs 都没有时不许冒出「逐项都一致」那种计数行 —— 那是把「没记」读成「相同」
        self.assertNotIn("项读数条件不一致", text)


if __name__ == "__main__":
    unittest.main()
