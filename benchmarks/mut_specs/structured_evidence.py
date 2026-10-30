"""§21.3 生产者 + §21.4 依赖基线那张变异表的**可执行**形态

靶子是 `tests/unit/test_structured_evidence.py` 与两张既有用例表 (wiring / agent_runtime)。
为什么把计数与产它的锚点放在一起: 文档里的「N/N 拦住」不可复现就等于没有 (I15/I11 那一族)。

witness 一栏按 doc 57 §二 第 1 条分两类:
  · behavioral=True —— 有**具体输入或条件**让这次改动能被观察到 (写清那个输入是什么);
  · behavioral=False —— 今天在行为层与原文不可区分, 拦住它的只可能是结构守卫。

本片里只有 M6 是 structural: `COMPILE_FACT_KEYS` 换成手写子集, 在**今天这台 MockGPU**
上不改变任何一次裁决 (L3 的 metrics 里本来就没这些键), 拦住它的是
`test_the_profiler_whitelist_has_one_source` 那句源码断言 —— 它守的是「名单只有一份」,
不是某一次行为。

⚠️ 本表里有一条是**真机抓到的洞** (M11/M12, CHECKLOG I78): 「新增依赖」的基线若取
patch 的父代, 而父代可能是**上一轮被拒**的那份候选 ⇒ 同一份违规候选第二次提交就过门,
真的上了远端编译 (`benchmarks/drivers/agent_action_h20.py` 第一版实测: 第 3 轮 torch 候选
`L1 PASSED`)。修法与守卫都在表里, 不许再退回去。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/structured_evidence.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "ar": "operator_agent/agent_runtime.py",
    "cond": "operator_agent/conductor.py",
    "fixer": "operator_agent/fixer.py",
    "verifier": "operator_agent/verifier.py",
    "tmpl": "operator_agent/prompts/fixer_repair.md",
}

TARGETS = [
    "tests/unit/test_structured_evidence.py",
    "tests/unit/test_agent_runtime.py",
    "tests/unit/test_agent_runtime_wiring.py",
    "tests/unit/test_harness_pass_rate.py",
]

ROWS = [
    # ── §21.3 的生产链路: 开关 / 拼装点 / 留痕 / 模板 ──
    Row("M1 开关恒关 (证据永远不注入)", "cond",
        "        if self.structured_evidence:",
        "        if False:",
        "**具体输入**: 默认构造的 Conductor 跑一轮修复 ⇒ Fixer 的 prompt 里没有那八段 "
        "⇒ `TestTheEvidenceReachesThePrompt::test_the_eight_items_are_in_the_fixers_prompt` 红; "
        "「开关是干净的两臂」也随之退化成两臂相同"),
    Row("M2 留痕写回局部变量 (后填的值进不了产物)", "cond",
        "                            _fix_evidence_record.update(_rec)",
        "                            _fix_evidence_record = _rec",
        "**具体输入**: 同一轮修复 ⇒ 血缘条目在修复**之前**就 append 了, 后填的绑定只改"
        "局部变量 ⇒ `test_the_round_facts_are_in_it_not_boilerplate` 红 "
        "(`rec[\"injected\"]` 恒 False 而 prompt 里有证据 —— I77 的原形)"),
    Row("M3 diff 基线取「正在被修的那一份」", "cond",
        "                                parent_code=_code_before_this_round,",
        "                                parent_code=repair_base,",
        "**具体输入**: 两轮都失败 (第 2 轮末尾那次修复) ⇒ 基线与 `current_code` 同一份 "
        "⇒ diff 恒空而标题仍在 ⇒ "
        "`TestTheDiffBaselineIsThePreviousCandidate` 红 (I77 的第二处)"),
    Row("M4 Fixer 丢掉那一段", "fixer",
        "            structured_evidence=structured_evidence,",
        "            structured_evidence=\"\",",
        "**具体输入**: 修复轮 ⇒ prompt 里没有那八段 (模板占位符被替换成空) ⇒ "
        "`test_the_eight_items_are_in_the_fixers_prompt` 红; "
        "这一行是「拼好的证据真的交给了 LLM」的唯一通道"),
    Row("M5 模板里的占位符被删掉", "tmpl",
        "{{structured_evidence}}\n\n",
        "",
        "**具体输入**: 修复轮 ⇒ 证据段根本没地方放 (Conductor 拼了也进不去) ⇒ "
        "`test_the_eight_items_are_in_the_fixers_prompt` 红 —— "
        "证明那个占位符是**承重**的, 不是装饰; "
        "⚠️ 与 M4 的红名单相同 (4 条) —— 同一根通道的两端 (Fixer 传没传 / 模板放不放), "
        "归属已复核: 两条都是「证据到不了 LLM」这一件事"),
    Row("M6 profiler 名单分成两份", "verifier",
        "for k in COMPILE_FACT_KEYS if k in bench",
        "for k in (\"num_warps\", \"num_stages\") if k in bench",
        "**结构守卫**: 今天这台 MockGPU 的 metrics 里没有这些键 ⇒ 行为层不可区分; "
        "`test_the_profiler_whitelist_has_one_source` 那句源码断言红 "
        "(名单两份必然漂: L3 写进 metrics 的与喂给模型的是两套)",
        behavioral=False),
    Row("M7 契约摘要丢掉短哈希", "ar",
        '        f"契约短哈希={stamp[:12] or \'(算不出)\'}"',
        '        f"契约短哈希=(算不出)"',
        "**具体输入**: 任一修复轮 ⇒ prompt 里那句「契约短哈希」变成「(算不出)」⇒ "
        "模型看不出这一轮是哪一份契约在管它, 复盘也无法把 prompt 与判据对上是同一份 ⇒ "
        "`test_the_digest_carries_the_contract_facts` 与 "
        "`test_the_round_facts_are_in_it_not_boilerplate` 双红"),
    Row("M8 首个错误取了「通过」的那一条", "cond",
        "            if not getattr(r, \"passed\", True):",
        "            if getattr(r, \"passed\", True):",
        "**具体输入**: 一轮里同时有通过/失败结果时, 取到的是通过那条 ⇒ 「首个错误」整段为空 "
        "而标题仍在 ⇒ `test_the_round_facts_are_in_it_not_boilerplate` 红 "
        "(断言「形态=whole_tensor」那句原文)"),
    Row("M9 profiler 一格不筛 (把 metrics 全喂进去)", "cond",
        "            for k in COMPILE_FACT_KEYS:",
        "            for k in list((getattr(r, \"metrics\", None) or {})):",
        "**具体输入**: 一轮的 metrics 里带 `max_error` 等非编译事实 ⇒ 它们被当成 profiler "
        "读数喂给模型 (`§21.3` 第 5 项是「相关 profiler 指标」, 不是「所有指标」) ⇒ "
        "`TestTheProfilerFactsComeFromTheWhitelist::test_only_compile_facts_are_fed` 红"),
    Row("M10 证据段不进血缘", "cond",
        '                "fix_evidence": _fix_evidence_record,\n',
        "",
        "**具体输入**: 任一跑 ⇒ 产物侧无法区分「注入了」与「没注入」(A/B 两臂从产物上"
        "分不开) ⇒ `test_the_round_facts_are_in_it_not_boilerplate` 红 + "
        "`test_harness_pass_rate::test_the_fixtures_cover_every_real_key` 红 "
        "(真血缘的键集合少一个)"),
    # ── §21.4 依赖基线的那个洞 (真机抓到) ──
    Row("M11 被拒的候选也进依赖基线", "cond",
        "            if _action_verdict.accepted:",
        "            if True:",
        "**具体输入**: 修复产物**一直**夹带 `import torch` (rounds=3) ⇒ 第 3 轮的基线"
        "变成第 2 轮那份**被拒的**候选 ⇒ `torch` 不再是「新增」⇒ 门放行, 那份源码进了后端 "
        "⇒ `test_the_same_violating_candidate_is_refused_every_time` 红 "
        "(真机第一版就是这么烧掉一次远端编译的, I78)"),
    Row("M12 依赖判据不看调用方给的基线", "ar",
        "    baseline = ctx.dep_baseline_source or patch.parent_source",
        "    baseline = patch.parent_source",
        "**具体输入**: 同一份违规候选 + 父代是被拒的那份 ⇒ 判据回落到 patch 的父代 ⇒ "
        "`test_new_dependency_prefers_the_admitted_baseline` 红 (判据那一侧的最后一道门)"),
]