"""§21 Agent Runtime 那张变异表的**可执行**形态 (`tests/test_agent_runtime*.py` 的靶子)

为什么把计数与产它的锚点放在一起 (与另几张表同一条纪律): 文档里的「N/N 拦住」不可复现
就等于没有 (doc 41 的「13 行 13/13」有两行锚点查不到 —— CHECKLOG I15/I11 那一族)。

witness 一栏按 doc 57 §二 第 1 条 (arXiv:2609.22220v1 行 246-251) 分两类:
  · behavioral=True —— 有一个**具体输入或条件**让这次改动能被观察到 (写清那个输入是什么);
  · behavioral=False —— 今天在行为层与原文不可区分, 拦住它的只可能是结构守卫。
    本片只有 M16 一行是这一类: `DESIGN_CHECKS` 今天只喂给报告/覆盖表的 `design_items`,
    删掉一行不会改变任何一次裁决 —— 所以它**不进行为分母**, 而「七条一条都不许丢」
    由 `test_agent_runtime.py::TestTheSevenDesignChecksAreFullyMapped` 以结构方式钉住。

⚠️ 这张表的靶子是**「判据接上了没有」**, 不是「候选变好了」。后者要有真机对照实验,
不在这一片。也正因为靶子在装配上, 有一条**必须**在表里: M20「门的裁决结果不看」
—— 判据算得再对, 没人读就是本仓最贵的失效方式 (CHECKLOG I46/I47/I49/I50/I52 同一族)。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/agent_action_gate.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "ar": "operator_agent/generate/agent_runtime.py",
    "cond": "operator_agent/app/conductor.py",
}

TARGETS = [
    "tests/unit/test_agent_runtime.py",
    "tests/unit/test_agent_runtime_wiring.py",
    "tests/unit/test_budget_gate.py",
    "tests/unit/test_harness_pass_rate.py",
    "tests/unit/test_escalation.py",
]

ROWS = [
    # ── 三态纪律: 只有违约才拒绝 (M1/M4) ──
    Row("M1 unknown 被当成拒绝", "ar",
        "        return not self.violations",
        "        return all(o.state == STATE_OK for o in self.outcomes)",
        "「没有授权根」这一格 (unknown) 变成拒绝 ⇒ 生成路径停摆: "
        "`TestOnlyViolationsRefuse::test_unknown_coverage_does_not_refuse` 红"),
    Row("M2 schema 那道不跑", "ar",
        "        schema = schema_problems(action)",
        "        schema = ()",
        "空 action_id / 表外种类 / 缺 patch 的动作全部放行 ⇒ `TestSchemaChecks` 全体红"),
    Row("M3 受保护目录少一个", "ar",
        '    "contract", "harness", "baseline", "measurements", "report",',
        '    "contract", "harness", "baseline", "report",',
        "声明要写 `candidate/measurements/…` 的 patch 被放行 ⇒ "
        "`test_every_protected_dir_is_refused` 红 (设计 §22.1 点名的五个之一)"),
    Row("M4 没有授权根 = 在范围内", "ar",
        '            "write_scope", STATE_UNKNOWN,',
        '            "write_scope", STATE_OK,',
        "判不了被写成「在范围内」 ⇒ 同一条 unknown 用例红 (fail-open 的那一半)"),
    Row("M5 标准库也算新依赖", "ar",
        "    added = [p for p in DependencyAnalyzer.to_pip_names(sorted(new_imports - old_imports))\n"
        "             if p not in DependencyAnalyzer.to_pip_names(sorted(old_imports))]",
        "    added = sorted(new_imports - old_imports)",
        "父代加一行 `import math` 就被判「新增外部依赖」⇒ "
        "`test_new_dependency_rejects_only_new_external_modules` 红 (误拦截那一半)"),
    Row("M6 ABI 判据不看了", "ar",
        "    problem = next((v.problem for v in verdicts if v.problem), None)\n    if problem:",
        "    problem = next((v.problem for v in verdicts if v.problem), None)\n    if False:",
        "签名与调用契约对不上也放行 ⇒ `test_abi_uses_the_same_judge_as_l1` 红; "
        "这一条正是**真跑必然 TypeError** 的那类候选"),
    Row("M7 生成轮重复也拒", "ar",
        '    if ctx.route_kind in ("", "performance") or not ctx.parent_verified:',
        "    if False:",
        "一字未改的**生成**候选被拒 ⇒ `test_duplicate_of_parent_refuses_only_a_repair` 红; "
        "doc 47 的口径是「无基率前只观测」, 这一刀会把更好的核丢掉"),
    Row("M8 没有实现物的工具放行", "ar",
        '                problems.append(f"请求的工具 {tool!r} 今天没有实现物 (注册表里是 None)")',
        "                pass",
        "`request_profiler` 被当成可用 ⇒ "
        "`test_the_two_never_implemented_tools_are_refused_with_a_reason` 红 "
        "(今天 profiler / shape_sweep 都没有实现物)"),
    Row("M9 没有量具的瓶颈放行", "ar",
        "        if action.expected_bottleneck not in gauged:",
        "        if False:",
        "声明 `spill-bound` / `bank-conflict` 的动作被放行 ⇒ "
        "`test_capability_refuses_a_bottleneck_without_a_gauge` 红 "
        "(§10.3 四类里只有两类有量具)"),
    Row("M10 ABI 破坏状态被忽略", "ar",
        "    if ctx.abi_broken and action.kind not in SIDE_EFFECT_FREE_KINDS:",
        "    if False:",
        "上一轮已判「接口/调用契约不符」时仍接受新候选 ⇒ "
        "`test_a_broken_abi_refuses_every_candidate_changing_action` 红 "
        "(§21.2 最后一行: 拒绝候选、回退父代)"),
    Row("M11 数值修复不要求误差形态", "ar",
        "    if action.kind == KIND_REPAIR_CORRECTNESS and ctx.error_shape:",
        "    if False:",
        "形态已知而理由里不带它 ⇒ `test_numeric_repair_must_carry_the_error_shape` 红 "
        "(丢了形态的修复与「随便改改」在产物里长得一样)"),
    Row("M12 修复不点名父代", "ar",
        '    if action.kind in REPAIR_KINDS and not str(action.parent_candidate or "").strip():',
        "    if False:",
        "修复动作可以不带 parent_candidate ⇒ 同一个用例红; 在真链路上还会让 "
        "`test_the_fix_round_names_the_parent_it_repairs` 失去回滚点"),
    Row("M13 性能路由无条件放行改源码", "ar",
        "        allow = base + (KIND_PROPOSE_TEMPLATE,)",
        "        allow = base + (KIND_PROPOSE_TEMPLATE,) + STRUCTURAL_KINDS",
        "「性能不达标」不再走搜索、直接改代码 ⇒ "
        "`test_performance_route_does_not_license_a_rewrite_by_default` 红 (§21.2)"),
    Row("M14 预算用尽也接受", "ar",
        "    if left <= 0:",
        "    if False:",
        "`tokens_left=0` 时仍接受要花 token 的动作 ⇒ "
        "`test_an_exhausted_budget_refuses_an_action_that_has_not_spent_yet` 红"),
    Row("M15 花过的钱不记账 (照拒)", "ar",
        "        deferred = tuple(budget) if (budget and ctx.spend_committed) else ()",
        "        deferred = ()",
        "具**体输入**: `BudgetPlan(llm_tokens=100)` 而本次已用 150 ⇒ 每一轮候选都被拒, "
        "`[Budget] … 退化成纯数值搜索` 不再出现、`llm_degraded_rounds` 归零 ⇒ "
        "`test_a_committed_spend_is_reported_but_not_refused` 与 "
        "`test_budget_gate::test_over_budget_never_asks_the_llm_for_a_structural_change` 双红"),
    Row("M16 设计七条少一条", "ar",
        '    ("是否引用不存在的能力", "capability"),\n',
        "",
        "今天行为不可区分: `DESIGN_CHECKS` 只喂给报告/覆盖表的 `design_items`, "
        "删一行不改变任何一次裁决 —— 拦住它的是结构守卫 "
        "`TestTheSevenDesignChecksAreFullyMapped` (七条原文一条都不许丢)",
        behavioral=False),
    Row("M17 非标量指标也渲染", "ar",
        "            if isinstance(v, bool) or isinstance(v, (int, float, str)):",
        "            if True:",
        "具**体输入**: `profiler_metrics={\"raw_ncu_dump\": {...}}` ⇒ 一个 dict 进了 "
        "prompt ⇒ `test_raw_metric_blobs_are_not_rendered` 红 (上下文里塞全量指标是负收益)"),
    Row("M18 截断不留痕", "ar",
        '        return (text[:limit] +\n'
        '                f"\\n… (已截断: 原文 {len(text)} 字符, 只给了前 {limit} 字符)")',
        "        return text[:limit]",
        "具**体输入**: 三倍上限的错误日志 ⇒ 产物里看不出「模型只看到开头」⇒ "
        "`test_a_huge_error_log_is_clipped_with_a_visible_marker` 红"),
    Row("M19 工具不可用不说为什么", "ar",
        '                out[name] = "注册表里没有实现物 (设计词汇里有, 兑现它的东西今天不存在)"',
        '                out[name] = ""',
        "「没有」而不说为什么 ⇒ `test_unavailable_tools_say_why` 红 "
        "(报告里一个空的理由与「没实现」是两件事)"),
    # ── 装配: 判据接上了没有 (M20 是这一片的核心) ──
    Row("M20 门的裁决结果不看", "cond",
        "            if not _action_verdict.accepted:\n"
        '                print(f"\\n[§21 Runtime] 动作被拒 (零 GPU 秒): "',
        "            if False:\n"
        '                print(f"\\n[§21 Runtime] 动作被拒 (零 GPU 秒): "',
        "门变成装饰: 夹带 `import torch` 的修复候选被送去编译、本轮产出一条真的 L1 结果 ⇒ "
        "`test_a_candidate_that_breaks_a21_rule_never_reaches_the_backend` 红 "
        "(判据算了却没人读, 是本仓最贵的失效方式)"),
    Row("M21 数值路当成编译修", "cond",
        '            if route is not None and getattr(route, "kind", "") == "numeric":',
        "            if False:",
        "§19.3 的「调试式修、不重写」在这一格上失配 ⇒ "
        "`test_compile_and_numeric_repairs_are_told_apart` 与 "
        "`test_the_fix_round_names_the_parent_it_repairs` 红"),
    Row("M22 修复轮没有父代", "cond",
        '                parent_code=((_code_before_this_round or verified_parent or "")\n'
        '                             if action == "fix" else (verified_parent or "")),',
        '                parent_code=(verified_parent or ""),',
        "修复轮的 diff 基线退回「已验证父代」, 而一份从未通过正确性的候选没有它 ⇒ "
        "`parent_candidate` 为空、被权限判据拦下 (拒绝理由从「新增依赖」变成「没点名父代」) ⇒ "
        "`test_a_candidate_that_breaks_a21_rule_never_reaches_the_backend` 红"),
    Row("M23 声明的预算是拍的数", "cond",
        "        return int(FixerAgent.FIX_MAX_TOKENS)",
        "        return 8192",
        "声明 8192 而那次调用真花 4096 ⇒ `test_the_declared_token_budget_is_the_limit_the_call_will_use` 红 "
        "(声明与实际不同源, 这道门看起来就永远能过)"),
    Row("M24 拒绝结果当成通过", "cond",
        "            level=VerificationLevel.L1_COMPILE, passed=False,\n"
        '            message="动作被拒 (未进编译): "',
        "            level=VerificationLevel.L1_COMPILE, passed=True,\n"
        '            message="动作被拒 (未进编译): "',
        "被拒的一轮报成「L1 通过」⇒ `TestTheRefusalResult` 与接线用例红 "
        "(拒绝必须是一条**失败**结果, 否则 §8 的路由与 Fixer 都不会动)"),
    Row("M25 被拒的候选照样跑依赖分析", "cond",
        '                    if l1_result and not (l1_result.metrics or {}).get("agent_action_refused") \\\n'
        "                            and Cap.DEP_ANALYSIS in self.backend.capabilities:",
        "                    if l1_result and Cap.DEP_ANALYSIS in self.backend.capabilities:",
        "被拒的候选触发一次远端依赖分析 (SSH 往返) ⇒ "
        "`test_a_refused_candidate_skips_the_remote_dependency_probe` 红"),
    Row("M26 记账不按次运行清零", "cond",
        "        self.agent_runtime.reset_run()",
        "        pass",
        "具**体输入**: 同一个 Conductor 跑两个任务 ⇒ 第二个任务的产物里带着第一个任务拦下的动作 "
        "(计数 2 而非 1) ⇒ `test_the_runtime_is_reset_between_runs_on_one_conductor` 红"),
    Row("M27 裁决不进产物", "cond",
        '                "agent_action": _action_verdict.as_dict(),\n',
        "",
        "「拒了」不进血缘 ⇒ 复盘时无法区分「判据拦住了坏候选」与「上下文装配错了」; "
        "`test_every_round_records_the_action_the_verdict_and_the_context` 与 "
        "`test_harness_pass_rate::test_the_fixtures_cover_every_real_key` 双红 "
        "(键集合只能从真代码读, 少一个就红)"),
]