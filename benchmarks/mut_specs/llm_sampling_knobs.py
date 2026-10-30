"""`tests/unit/test_llm_sampling_knobs.py` 那张变异表的**可执行**形态 (每行带 kill witness)

为什么要在仓里有一份 (而不是一段 /tmp 脚本): 同一个理由在 doc 41 与 CHECKLOG I29 各撞过一次 ——
写在文档里的「N/N 拦住」如果产它的锚点不在仓里，那个数就不可复现，而下一个人只能重猜。
这一片的 /tmp 版本还多撞出一件事：H3 **第一次是逃逸的**，逃逸的原因是断言的 needle
出现在被测代码自己的解释文案里 (登记 I32)。那一行的 witness 写的是它第二次被拦住的位置。

跑:
    python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/llm_sampling_knobs.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {
    "llm": "operator_agent/llm_client.py",
    "hrn": "benchmarks/harness_pass_rate.py",
}

TARGETS = ["tests/unit/test_llm_sampling_knobs.py", "tests/unit/test_harness_pass_rate.py",
           "tests/unit/test_llm_reported_model.py", "tests/unit/test_llm_usage_semantics.py"]

ROWS = [
    # ── L 组: 账本必须记「真发出去的那一份」 ──
    Row("L1 body 里的 temperature 不看逐调用覆盖", "llm",
        '            "temperature": sent_temperature,',
        '            "temperature": self.temperature,',
        "同一份 client 上 `chat(temperature=0.1)` ⇒ 假 urlopen 抓到的 body 印 0.3；"
        "AST 那条也红 (body 那格不再是 Name)"),

    Row("L2 账本里的 temperature 另读一遍构造参数", "llm",
        "            temperature=sent_temperature,",
        "            temperature=self.temperature,",
        "`chat(temperature=0.1)` 之后 `records[0].temperature` 变 0.3 ⇒ 与 body 不一致"),

    Row("L3 enable_thinking 写死 False (属性没人读)", "llm",
        '"chat_template_kwargs": {"enable_thinking": self.enable_thinking},',
        '"chat_template_kwargs": {"enable_thinking": False},',
        "`LLMClient(enable_thinking=True)` 发一次 ⇒ body 与账本从 True 翻成 False"),

    Row("L4 聚合把 max_tokens 那一轴并掉了", "llm",
        "            key = (r.caller_tag, r.model, r.temperature, r.max_tokens, r.enable_thinking)",
        "            key = (r.caller_tag, r.model, r.temperature)",
        "Coder 真有两种配置 (0.1/40960 ×3 与 0.1/1024 ×2) ⇒ 并轴后只剩一种、calls 折成 5"),

    Row("L5 没记采样的条目折进组合 (而不是单独计数)", "llm",
        "            if r.temperature is None and r.max_tokens is None and r.enable_thinking is None:",
        "            if False:",
        "一条 `record(...)` 不带那三个字段 ⇒ 它进 by_caller，`n_records_without_sampling` 归零"),

    Row("L6 summary() 不发布 sampling 这个键", "llm",
        '            "sampling": self.sampling_configs(records),',
        "            # (变异掉: 不发布这一列)",
        "`summary()` 的键集合里没有 `sampling` ⇒ 键名守卫红。⚠️ 这一行的 5 条红里**只有 2 条是"
        "证人** (键名守卫 + 本文件的 pretty 用例)，另 3 条是 `pretty()` 里 `s[\"sampling\"]` 抛 "
        "KeyError 打穿了另外两个文件的账单用例 —— 连带不是覆盖，数进分母就会把「一处缺陷」报成"
        "「五处都盯着这列」"),

    Row("L7 pretty() 不印采样那一段", "llm",
        '        sampling_lines = format_sampling_configs(s["sampling"])\n        if sampling_lines:',
        '        sampling_lines = format_sampling_configs(s["sampling"])\n        if False:',
        "账本里有 6 条时 `pretty()` 不再出现「采样配置」那一行 (「算了没人读」的那一侧)"),

    Row("L8 截断错误文案另算一遍 max_tokens", "llm",
        'f"思考过程耗尽 max_tokens={sent_max_tokens}，"',
        'f"思考过程耗尽 max_tokens={self.max_tokens}，"',
        "`chat(max_tokens=64)` 且响应 finish_reason=length ⇒ 异常文本从 64 变 4096"),

    # ── H 组: 报表侧的三态与「两侧并排」 ──
    Row("H1 缺账本时返回空表而不是 None", "hrn",
        '    return c.get("sampling") or None',
        '    return c.get("sampling") or {}',
        "`sampling_from_metadata({})` 从 None 变 `{}` ⇒ 「这次没账本」与「账本里空」在报表同形"),

    Row("H2 跨运行合并不累加 calls", "hrn",
        "                bucket[key] = bucket.get(key, 0) + int(c.get(\"calls\") or 0)",
        "                bucket[key] = int(c.get(\"calls\") or 0)",
        "两次运行各有 Coder(0.1/40960) 3 次与 2 次 ⇒ 合并后 5 变 2 (后写覆盖前写)"),

    Row("H3 「一侧没记」的格子印成空串", "hrn",
        '        return "— 没记"',
        '        return ""',
        "A 侧有 llm 那一维、B 侧没有 ⇒ B 的格子从 `— 没记` 变成空白，读起来像「两边都没值」。"
        "⚠️ 这行**第一次跑是逃逸的**：报警文案里原样引着「— 没记」，而用例断言的是整段子串，"
        "于是文案替格子满足了断言 (I32)。改法两半：文案不再出现那个串，断言打在位置上 "
        "(`\"→  B — 没记\"`) —— 现在的 witness 是改过之后的那条断言"),

    Row("H4 format_comparison 不印 knobs (算了没人读)", "hrn",
        "    lines.extend(format_knobs_pair(ma, mb))",
        "    pass  # (变异掉: A/B 现场不看读数条件)",
        "两份 knobs 不同的报表走 `--compare` ⇒ 输出里没有「读数条件 (knobs)」那一段，"
        "而落盘那句「必须带上 knobs」的警告至此仍没有读者"),
]
