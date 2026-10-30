"""LLM usage 的两种缓存约定 —— 归一器不许把其中一种算成两倍 (arXiv:2609.11987v1)

外部一手 (行 78, 逐字):
    "input-token fields are cache-inclusive on some SDKs and cache-exclusive on others,
     and a normalizer that assumes one convention doubles the other."
行 1108-1111 给出三家: Anthropic Messages 的 `input_tokens` 是**排除**缓存读写的;
OpenAI Responses 的 `usage.input_tokens_details.cached_tokens` 是**包含**的;
LangChain 的 `UsageMetadata.input_token_details` 也是包含的。

为什么我方现在管这件事: #71 的第四个出口（预算触顶）要拿这份账单当货币。而那篇的失效形状
恰是「**钱算错了，判定一个都没受影响**」(§5.1 与其修订记录: 八月版的全部成本结论与 stop-loss
用的货币都被自家归一器污染) —— 一个只影响「花多少」不影响「过没过」的错，可以整篇存活。

判据都写成**能算出理论值**的绝对锚点（不是「看起来合理」）:
  · inclusive: 1000 prompt 里 800 是缓存 ⇒ 输入总量仍是 1000。**加成 1800 就是那句 doubles**。
  · exclusive: 100 + 800 读 + 50 写 ⇒ 950。
  · 钱: deepseek-chat 表价 0.14/0.28 每 M ⇒ 1000 in + 200 out = $0.000196 (逐位核过)。
  · 缓存那部分**不给折扣价**（本仓不抄没核过的数），改成显式标 `cost_basis`，让读者知道
    这是按全价算的、不是账单。

运行: python3 -m pytest tests/unit/test_llm_usage_semantics.py -q
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent.llm_client import CostTracker, parse_usage  # noqa: E402

INCLUSIVE = {"prompt_tokens": 1000, "completion_tokens": 200,
             "prompt_tokens_details": {"cached_tokens": 800}}
RESPONSES = {"input_tokens": 1000, "output_tokens": 200,
             "input_tokens_details": {"cached_tokens": 800}}
EXCLUSIVE = {"input_tokens": 100, "output_tokens": 20,
             "cache_read_input_tokens": 800, "cache_creation_input_tokens": 50}


class TestUsageSemantics(unittest.TestCase):
    def test_inclusive_shape_does_not_double_the_cached_span(self):
        """这条是整件事的 kill witness: 把 cached 再叠进 prompt 就变成 1800"""
        n = parse_usage(INCLUSIVE)
        self.assertEqual(1000, n["prompt_tokens"], n)
        self.assertEqual(800, n["cached_prompt_tokens"])
        self.assertEqual("inclusive", n["cache_semantics"])
        self.assertNotEqual(1800, n["prompt_tokens"])

    def test_responses_key_is_the_same_convention(self):
        n = parse_usage(RESPONSES)
        self.assertEqual(1000, n["prompt_tokens"], n)
        self.assertEqual(800, n["cached_prompt_tokens"])

    def test_exclusive_shape_adds_cache_reads_and_writes(self):
        n = parse_usage(EXCLUSIVE)
        self.assertEqual(950, n["prompt_tokens"], n)
        self.assertEqual(800, n["cached_prompt_tokens"])
        self.assertEqual("exclusive", n["cache_semantics"])

    def test_no_usage_is_absent_not_zero_silently(self):
        """取不到约定就明写 absent, 别让「没有缓存」与「没读出来」长得一样"""
        n = parse_usage({})
        self.assertEqual({"prompt_tokens": 0, "completion_tokens": 0,
                          "cached_prompt_tokens": 0, "cache_semantics": "absent"}, n)

    def test_cached_tokens_never_exceed_the_reported_total(self):
        """坏 payload (cached > prompt) 不许把总量撑大 —— 上界钳在这里"""
        n = parse_usage({"prompt_tokens": 500, "completion_tokens": 10,
                         "prompt_tokens_details": {"cached_tokens": 900}})
        self.assertEqual(500, n["prompt_tokens"], n)
        self.assertEqual(500, n["cached_prompt_tokens"])


class TestLedgerCarriesTheBasis(unittest.TestCase):
    def test_cost_is_the_table_price_of_the_whole_input_and_is_flagged(self):
        tracker = CostTracker()
        tracker.record(model="deepseek-chat", prompt_tokens=1000, completion_tokens=200,
                       usage_source="api_usage", cached_prompt_tokens=800,
                       cache_semantics="inclusive")
        s = tracker.summary()
        # 绝对锚点: 1000×0.14/M + 200×0.28/M
        self.assertAlmostEqual(0.000196, s["total_cost_usd"], places=9)
        self.assertEqual(800, s["total_cached_prompt_tokens"])
        self.assertEqual(1, s["n_cache_unpriced_calls"])
        self.assertEqual({"inclusive": 1}, s["cache_semantics"])

    def test_no_cache_means_no_flag(self):
        tracker = CostTracker()
        tracker.record(model="deepseek-chat", prompt_tokens=100, completion_tokens=20,
                       usage_source="api_usage")
        s = tracker.summary()
        self.assertEqual(0, s["n_cache_unpriced_calls"])
        self.assertEqual({"absent": 1}, s["cache_semantics"])
        self.assertEqual(0, s["total_cached_prompt_tokens"])

    def test_the_human_report_says_the_money_is_not_a_bill(self):
        """报表里必须有一行说明「这是按全价算的」—— 否则 #71 拿它当账单就是拿自述当证据"""
        tracker = CostTracker()
        tracker.record(model="deepseek-chat", prompt_tokens=1000, completion_tokens=200,
                       usage_source="api_usage", cached_prompt_tokens=800,
                       cache_semantics="inclusive")
        text = tracker.pretty()
        self.assertIn("按全价算的", text, text)
        self.assertIn("800", text)

    def test_the_report_stays_quiet_when_there_is_no_cache(self):
        """反向对照: 没有缓存命中时不许凭空多出那句免责声明 (只在条件成立一侧取样=没测条件)"""
        tracker = CostTracker()
        tracker.record(model="deepseek-chat", prompt_tokens=100, completion_tokens=20,
                       usage_source="api_usage")
        self.assertNotIn("按全价算的", tracker.pretty())


class TestTheWiringIsLive(unittest.TestCase):
    """两条「算了有人读」的接线守卫 —— 归一器写在文件里但没人调用，就等于没有"""

    def test_the_client_reads_usage_through_the_normalizer(self):
        src = (Path(__file__).resolve().parents[2] / "operator_agent" / "llm_client.py"
               ).read_text(encoding="utf-8")
        body = src[src.index("def _call_model"):]
        self.assertIn("parse_usage(", body[:body.index("\n    def ") if "\n    def " in body else 4000],
                      "调用点还在直接读 usage['prompt_tokens'] ⇒ 归一器是死代码")

    def test_the_persistence_path_copies_the_whole_cost_dict(self):
        """`result.json` 里的 `llm_cost` 必须是整包导出。

        本仓记过一次: 「只挑这几个键」的白名单把算好的值丢在落盘前 (doc 49 §八)，
        所以这条守卫盯的是**没有键过滤**，而不是「键在不在名单里」—— 后者从抄下那刻就开始漂，
        新加的 `cache_semantics` / `total_cached_prompt_tokens` 就是会被漂掉的那种。
        """
        src = (Path(__file__).resolve().parents[2] / "operator_agent" / "artifact_store.py"
               ).read_text(encoding="utf-8")
        self.assertIn('result_data["llm_cost"] = _llm_cost', src)
        self.assertNotIn('result_data["llm_cost"] = {', src, "又变成挑键导出了")


if __name__ == "__main__":
    unittest.main(verbosity=2)
