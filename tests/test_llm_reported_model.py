"""端点自报的模型标识与请求的那个并排记 —— #89 的「先观测」那半边

外部一手 (arXiv:2609.11987v1 行 89 的 runtime drift gate): 每条腿开跑前核验「服务端**实际**
模型标识（带日期快照归一）」，并对模型自报的训练截止做 soft-check；它观察到自报值在几次探测里
变过 (2023-07 / 2023-10 / 2024-07) ⇒ 那句结论是「**自报只作参考**」。后来还出过一次事故：
门探测的是路由器的默认线路，而实验跑在另一条固定线路上 ⇒ 探测被修订成「测这次跑的确切固定后端」。

我方 `backend_caps`（#44/#45）管的是「后端声明**能力**」，没有任何东西核验「LLM 端点今天答的是
哪个模型标识」—— 而端点漂移会直接污染 prompt 对照实验（同一份 prompt 今天打到的不是同一个模型）。

本片只做观测，不做判定：账本条目带 `reported_model`，账单摘要与 `pretty()` 把「请求 X → 自报 Y」
逐族报出来。**刻意不做同族归一**（外部那篇按日期快照归一是它端点上的规则，本仓不抄没核过的映射表），
也刻意不把「响应里没有这个字段」当成「一致」—— 那是「不知道」。

变异标定 (2026-09-30 实跑, 6 条全拦; 每条改一处、跑完立刻逐字节复原并断言):
    M1 按日期快照归一 (抄外部那条规则)              → 1 红 (test_no_normalization_...)
    M2 把「响应里没这个字段」当一致                  → 1 红 (test_absent_report_...)
    M3 pretty 不印漂移那一行                        → 1 红 (test_a_different_self_reported_...)
    M4 接线断掉 (`reported_model=None` 写死)        → 1 红 (那条 AST 接线守卫)
    M5 `summary_since` 不看范围 (继承上一次)         → 1 红 (test_the_scoped_reading_...)
    M6 逐族计数的键丢了「请求」那一侧                → 1 红 (与 M3 **同一个读者**)
    ⚠️ M3 与 M6 红名单逐条相同 —— 复核过归属: 两行改的是同一条 pretty() 输出的两种失效形态
       (整段不印 vs 印出来但少了「请求的那一侧」)，读者本来就是那一条同时断言两个子串的用例；
       两行都保留是因为缺陷各在一层，不是为了凑数。
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.llm_client import CostTracker  # noqa: E402


def _rec(tracker: CostTracker, model: str, reported, calls: int = 1) -> CostTracker:
    for _ in range(calls):
        tracker.record(model=model, prompt_tokens=100, completion_tokens=10,
                       usage_source="api_usage", caller_tag="Coder",
                       reported_model=reported)
    return tracker


class TestTheLedgerKeepsBothNames(unittest.TestCase):
    def test_agreement_is_silent(self):
        """反向对照: 只在会报警那一侧取样, 测的是代码在不在, 不是条件对不对"""
        s = _rec(CostTracker(), "deepseek-chat", "deepseek-chat").summary()
        self.assertEqual({}, s["model_mismatches"])
        self.assertEqual(0, s["n_model_mismatch_calls"])
        self.assertEqual(0, s["n_reported_model_absent"])

    def test_a_different_self_reported_name_is_counted_per_family(self):
        t = _rec(CostTracker(), "deepseek-chat", "deepseek-chat-v4", calls=2)
        s = t.summary()
        self.assertEqual({"deepseek-chat -> deepseek-chat-v4": 2}, s["model_mismatches"])
        self.assertEqual(2, s["n_model_mismatch_calls"])
        pretty = t.pretty()
        self.assertIn("deepseek-chat -> deepseek-chat-v4: 2 次", pretty)
        self.assertIn("自报只作参考", pretty)

    def test_no_normalization_of_dated_snapshots(self):
        """`model` 与 `model-2026-08-01` 这里算**不同**。

        这不是遗漏: 外部那篇做了日期快照归一, 但那是它自己端点上的规则。少一条没核过的映射表
        换来的是「多报一些不一致」—— 方向是安全的 (观测阶段不据此判定任何事)。
        """
        s = _rec(CostTracker(), "gpt-5.5", "gpt-5.5-2026-08-01").summary()
        self.assertEqual(1, s["n_model_mismatch_calls"])

    def test_absent_report_is_not_read_as_agreement(self):
        s = _rec(CostTracker(), "deepseek-chat", None).summary()
        self.assertEqual(1, s["n_reported_model_absent"])
        self.assertEqual(0, s["n_model_mismatch_calls"], "不知道 ≠ 对上了")
        s2 = _rec(CostTracker(), "deepseek-chat", "").summary()
        self.assertEqual(1, s2["n_reported_model_absent"], "空串也是「没有」, 不是「与请求同名」")

    def test_the_scoped_reading_does_not_inherit_an_earlier_run(self):
        """`summary_since` 是「这一次运行」的口径 —— 漂移计数也必须跟着那个范围走

        否则第二次运行会替第一次的运行报出漂移 (账本挂在 client 上、`reset()` 全仓没人调,
        那是 #70 记过的同一次事故)。
        """
        t = CostTracker()
        _rec(t, "a-model", "other-model")          # 上一次运行
        mark = t.watermark()
        _rec(t, "b-model", "b-model")              # 这一次运行: 一致的
        s = t.summary_since(mark)
        self.assertEqual(0, s["n_model_mismatch_calls"],
                         "本次运行没漂移, 但读数继承了上一次的")
        self.assertEqual(1, s["total_calls"])


class TestTheWiringIsLive(unittest.TestCase):
    """记账函数写好了不等于有人调它 —— 这两条钉的是接线, 不是逻辑"""

    SRC = Path("operator_agent/llm_client.py")

    def _call_model_body(self):
        tree = ast.parse(self.SRC.read_text(encoding="utf-8"))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                if fn.name == "_call_model":
                    return fn
        self.fail("找不到 `_call_model` —— 这条守卫的前提已变")

    def test_the_client_passes_the_self_reported_name_into_the_ledger(self):
        body = self._call_model_body()
        for call in (n for n in ast.walk(body) if isinstance(n, ast.Call)):
            kws = {k.arg for k in call.keywords}
            if "reported_model" in kws:
                value = next(k.value for k in call.keywords if k.arg == "reported_model")
                src = ast.dump(value)
                self.assertIn("model", src,
                              f"`reported_model=` 传进来的不是响应里那个字段: {src}")
                self.assertNotIsInstance(value, ast.Constant,
                                         "写成常量就不是观测了, 是装饰")
                return
        self.fail("`_call_model` 没有把 `reported_model=` 传给账本 ⇒ 这一列永远是 None")

    def test_the_reader_of_the_column_exists(self):
        """`summary()` 里没有 `model_mismatches` 这一列时, 记进去的字符串没有任何读者"""
        text = self.SRC.read_text(encoding="utf-8")
        self.assertIn('"model_mismatches"', text)
        self.assertIn("n_reported_model_absent", text)
        # 而 pretty() 必须真读它 —— 「记了但没人印」是本仓反复反对的那一种
        tree = ast.parse(text)
        pretty = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "pretty")
        self.assertIn("n_model_mismatch_calls", ast.dump(pretty),
                      "pretty() 不读这一列 ⇒ 观测落不到任何人眼前")


if __name__ == "__main__":
    unittest.main()
