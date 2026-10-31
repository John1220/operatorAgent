"""绑定轴的可信度门 (缺口 4 的另一半, doc 49 §八 第 17 条)。

钉住的行为: 一条 roofline 块里的 `bound_type` 只有在**脊点算出来了**且**分母口径核过**
时才有资格替配方筛选做「这条不是 memory-bound」那样的否定判断; 否则轴向整体不参与,
并且 prompt 里那句「当前状态: bound=…」换成一句明说的「本轮不作判定」。

为什么要盯 prompt: 对今天的真实配方库 (28 条, `memory_bound` 条件 0 条) 而言, 改筛选
那半的影响是**潜在的**; 而 `to_prompt()` 一直在把「算子名启发式猜出来的 compute」当
实测结论发给模型 —— 那一半是 live 的。两组用例分别盯这两面, 不要只盯容易的那面。

运行: python3 -B -m unittest tests.test_bound_axis_gate -v
"""

import ast
import io
import os
import re
import sys
import tokenize
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from operator_agent.core.hardware_caps import unknown_caps  # noqa: E402
from operator_agent.generate.optimization_planner import (  # noqa: E402
    AXIS_VALUES, OptimizationPlan, OptimizationPlanner, Recipe, _axis_field,
    axis_evidence,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 一份可核的轴: 脊点算出来了 (roofline_bound 不是 unknown), 分母口径核过了。
MEMORY = {"bound_type": "memory", "roofline_bound": "memory", "peak_trusted": True,
          "gflops_utilization_pct": 32.5, "bandwidth_utilization_pct": 12.7}
COMPUTE = dict(MEMORY, bound_type="compute", roofline_bound="compute")

# doc 49 §八 第 17 条实测到的那个形状: 同一个块里 `roofline_bound="unknown"`
# (脊点没算出来) 与 `bound_type="compute"` (Verifier 兜到算子名启发式) 并存。
NAME_GUESS = {"bound_type": "compute", "roofline_bound": "unknown",
              "peak_trusted": False, "gflops_utilization_pct": 0.0,
              "bandwidth_utilization_pct": 0.0}


def _synth(rid: str, when: str, prio: int) -> Recipe:
    return Recipe(id=rid, name=rid, applies_to=["all"], applies_when=when,
                  priority=prio, description=f"合成配方, 只为覆盖 `{when}` 这一侧",
                  _conditions=OptimizationPlanner._parse_conditions(when))


def _plan(axis_src) -> OptimizationPlan:
    """用一组合成配方跑一次 plan() —— 真实库里 `memory_bound` 条件今天 0 条,
    只拿真实库测的话, 「unknown 不再拒绝访存向配方」这条改动根本不会被执行到。"""
    p = OptimizationPlanner()
    p._all_recipes = [_synth("SYN-COMPUTE", "compute_bound", 1),
                      _synth("SYN-MEMORY", "memory_bound", 2),
                      _synth("SYN-ALWAYS", "always", 3)]
    return p.plan(kernel_type="matmul", dsl="triton", hw_caps=unknown_caps("test-gpu"),
                  roofline=axis_src, precision="fp16", round_num=2)


def _ids(plan: OptimizationPlan) -> set:
    return {r.id for r in plan.selected_recipes}


class TestAxisEvidence(unittest.TestCase):
    """`axis_evidence()` 的每一条门单独钉 —— 门与门共用一个返回类型, 少一条会静默放行。"""

    def test_a_full_verifiable_axis_is_evidence(self):
        self.assertTrue(axis_evidence(COMPUTE)[0])
        self.assertTrue(axis_evidence(MEMORY)[0])

    def test_round_one_is_not_evidence(self):
        ok, note = axis_evidence({})
        self.assertFalse(ok)
        self.assertIn("首轮", note)

    def test_name_heuristic_fallback_is_not_evidence(self):
        """§八 第 17 条的那个块: bound_type 有值但 roofline_bound=unknown。"""
        ok, note = axis_evidence(NAME_GUESS)
        self.assertFalse(ok, "脊点没算出来时 bound_type 是名字猜的, 不该承重")
        self.assertIn("算子名启发式", note)

    def test_unverified_peak_denominator_is_not_evidence(self):
        """分母来自 ⚠️ 未核口径的表行 (3090/4090/a6000/a5000/t4) 时, 比较两边不同源。"""
        ok, note = axis_evidence(dict(COMPUTE, peak_trusted=False))
        self.assertFalse(ok)
        self.assertIn("分母口径未核", note)

    def test_missing_peak_trusted_is_not_evidence(self):
        """缺键 = 出处不可考 ⇒ 不承重。方向是刻意选的: 拿一个无法核实的分母去**拒绝**
        配方会在 rejected_recipes 里留下一个看起来很合理的理由, 而没人会去查它凭什么。"""
        src = {k: v for k, v in COMPUTE.items() if k != "peak_trusted"}
        self.assertFalse(axis_evidence(src)[0])
        self.assertIn("peak_trusted=None", axis_evidence(src)[1])

    def test_vocabulary_drift_is_not_evidence(self):
        """`memory_bound` / `mixed` 这类拼法漂移以前会一路落到「不是 memory」的分支,
        于是所有 compute_bound 配方放行。现在它连证据都不是, 两个方向都不参与。"""
        for bad in ("memory_bound", "compute_bound", "mixed", "MEMORY", "", None):
            ok, _ = axis_evidence({"bound_type": bad, "roofline_bound": "memory",
                                   "peak_trusted": True})
            self.assertFalse(ok, f"轴向词汇外的值 {bad!r} 不该被当成证据")

    def test_nested_shape_is_read_too(self):
        """`plan()` 的 docstring 说的是 roofline 块本身, 生产调用方传的是 L3 metrics 顶层。
        只认一种形状时, 「我没接对」会被读成「没有这个数据」—— 两种都得能认出证据。"""
        nested = {"roofline_analysis": dict(COMPUTE)}
        self.assertTrue(axis_evidence(nested)[0],
                        "只带 roofline_analysis 块的形状应该可用")
        self.assertEqual(COMPUTE["gflops_utilization_pct"],
                         _axis_field(nested, "gflops_utilization_pct"))

    def test_axis_values_match_what_verifier_emits(self):
        """轴向词表不能自己漂移: 从 verifier 的 `_roofline_analysis` 里现取它写入
        `bound_type` 的字面量, 逐个确认要么在 `AXIS_VALUES` 里要么被明确当作非证据。"""
        path = os.path.join(REPO_ROOT, "operator_agent", "decide", "verifier.py")
        with io.open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "_roofline_analysis"), None)
        self.assertIsNotNone(fn, "verifier._roofline_analysis 不见了, 这条守卫失去锚点")
        emitted = {n.value for n in ast.walk(fn)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and re.fullmatch(r"(memory|compute|unknown|mixed)", n.value)}
        self.assertTrue(emitted)
        # 门要的键必须还在产出端 —— 少一个, 门会永远读到 None 而把轴向**永久**关掉,
        # 那种红不会出现在任何报错里, 只会让所有轮次都不按轴筛配方。
        keys = {k.value for n in ast.walk(fn) if isinstance(n, ast.Dict)
                for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        for required in ("bound_type", "roofline_bound", "peak_trusted"):
            self.assertIn(required, keys,
                          f"产出端不再写 {required}, 门会静默永久关闭")
        self.assertEqual(set(AXIS_VALUES), emitted - {"unknown", "mixed"},
                         f"产出端能写出 {sorted(emitted)}, 而门只认 {sorted(AXIS_VALUES)} —— "
                         f"两边必须对齐, 且「没有判决」的那个值不能被收进词表")
        self.assertNotIn("unknown", AXIS_VALUES,
                         "把 unknown 收进轴向词表就等于把「不知道」恢复成「按某个真轴筛」")


class TestSelectionDoesNotFilterOnAnUnverifiableAxis(unittest.TestCase):
    """筛选面: 轴不可用时 bound 条件**整体不参与**, 而不是「默认视为 compute-bound」。"""

    def test_both_bound_conditions_survive_when_axis_unverifiable(self):
        sel = _ids(_plan(NAME_GUESS))
        self.assertIn("SYN-MEMORY", sel,
                      "不可信的轴还在拒绝访存向配方 —— §八 第 17 条没修掉")
        self.assertIn("SYN-COMPUTE", sel)
        self.assertIn("SYN-ALWAYS", sel)

    def test_memory_axis_still_rejects_compute_recipe(self):
        """反向对照: 轴可核时拒绝必须**照旧发生**。没有这一条, 上面那例把门整个删掉
        也能全绿 (那才是真的空转)。"""
        plan = _plan(MEMORY)
        self.assertNotIn("SYN-COMPUTE", _ids(plan))
        self.assertIn("conditions not met",
                      {r.id: w for r, w in plan.rejected_recipes}["SYN-COMPUTE"])

    def test_compute_axis_still_rejects_memory_recipe(self):
        self.assertNotIn("SYN-MEMORY", _ids(_plan(COMPUTE)))

    def test_differential_is_attributable_to_the_gate_alone(self):
        """同一组合成配方, 只换 provenance 键: 差异必须**恰好**落在被门放行/拒绝的那
        一条上。两个输入之间差两个键 (roofline_bound + peak_trusted), 所以这里断言的是
        「对称差 = 一个元素」而不是「集合不同」—— 后者在门连带影响别的条件时也会绿。"""
        sel_gate_off = _ids(_plan(NAME_GUESS))
        sel_gate_on = _ids(_plan(dict(COMPUTE, roofline_bound="memory",
                                      bound_type="memory")))
        self.assertEqual({"SYN-COMPUTE"}, sel_gate_off ^ sel_gate_on,
                         f"门的效应不止一条配方: off={sel_gate_off} on={sel_gate_on}")

    def test_untrusted_utilization_percentages_stop_rejecting(self):
        """`low_utilization` / `low_bandwidth_util` 是从同一个分母除出来的百分比,
        分母不核实时它们一起作废 (否则就是「一个块里两个口径」)。

        三例一起看才有意义: 只写「轴不可用时该通过」的话, 把利用率原样带过去也能通过
        (32.5 < 80 本来就成立) —— 必须补一个**高于阈值**的读数, 那才是只有作废利用率
        才会通过的形状。
        """
        p = OptimizationPlanner()
        p._all_recipes = [_synth("SYN-LOWUTIL", "compute_bound AND utilization < 80", 1)]
        kwargs = dict(kernel_type="matmul", dsl="triton", hw_caps=unknown_caps("t"),
                      precision="fp16", round_num=2)
        # (a) 轴不可用 + 利用率 91%: 那个百分比是从未核的分母除出来的, 不能据此拒绝
        off = p.plan(roofline=dict(NAME_GUESS, gflops_utilization_pct=91.0), **kwargs)
        self.assertIn("SYN-LOWUTIL", _ids(off),
                      "分母未核实时, 利用率百分比还在替配方做否定判断")
        # (b) 轴可用 + 32.5% < 80: 条件成立, 通过
        low = p.plan(roofline=dict(COMPUTE, gflops_utilization_pct=32.5), **kwargs)
        self.assertIn("SYN-LOWUTIL", _ids(low))
        # (c) 轴可用 + 91% >= 80: 拒绝必须照旧发生 (绝对锚点, 否则 (a)(b) 删掉条件也全绿)
        high = p.plan(roofline=dict(COMPUTE, gflops_utilization_pct=91.0), **kwargs)
        self.assertNotIn("SYN-LOWUTIL", _ids(high))


class TestPromptDoesNotAssertAnAxis(unittest.TestCase):
    """prompt 面 (live): 不可信的轴向不能再以「当前状态: bound=…」的形式发给模型。"""

    def test_unverifiable_axis_is_reported_as_no_verdict(self):
        plan = _plan(NAME_GUESS)
        text = plan.to_prompt()
        self.assertIn("瓶颈轴", text)
        self.assertIn("不作判定", text)
        self.assertNotIn("当前状态: bound=", text)
        self.assertNotIn("bound=compute", text)

    def test_the_reason_printed_names_the_gate_that_fired(self):
        """原因要可见, 而且要指对是哪一条门。

        两条门对 `NAME_GUESS` **都**适用, 于是断言它印的是哪句话其实挂在代码里那两条
        检查的先后顺序上 —— 所以这里各用一个只触发一条门的块。挂在顺序上的一次改序
        就会让这条用例红得毫无道理, 而挂在上面上才是真缺陷。
        """
        only_name_guess = dict(COMPUTE, roofline_bound="unknown")
        text = _plan(only_name_guess).to_prompt()
        self.assertIn("算子名启发式", text)
        self.assertNotIn("当前状态: bound=", text)

        only_untrusted_peak = dict(COMPUTE, peak_trusted=False)
        text2 = _plan(only_untrusted_peak).to_prompt()
        self.assertIn("分母口径未核", text2)
        self.assertNotIn("当前状态: bound=", text2)

    def test_verifiable_axis_still_prints_the_state_with_anchors(self):
        """幅度锚点: 两个百分比必须出现在文本里。只断言「当前状态」在不在, 删掉数字
        格式化也照样绿 (M7 那次就是这么漏的)。"""
        text = _plan(MEMORY).to_prompt()
        self.assertIn("当前状态: bound=memory", text)
        self.assertIn("32.5%", text)
        self.assertIn("12.7%", text)
        self.assertNotIn("瓶颈轴", text)

    def test_round_one_prompt_is_unchanged(self):
        """首轮既没有块也没有猜测: 保持**什么都不印**, 与改动前逐字节一致。
        给一个「轴向不可用」的说明会把「还没测过」和「测过但不可信」弄成同一句话。"""
        text = _plan(None).to_prompt()
        self.assertNotIn("瓶颈轴", text)
        self.assertNotIn("当前状态:", text)

    def test_directly_constructed_plan_does_not_invent_a_verdict(self):
        """`axis` 缺省 (没经过 plan() 直接构造) 时必须按「不可用」处理, 否则
        「没有裁决」会被读成「有一个可以断言的 bound」。"""
        plan = OptimizationPlan(kernel_type="matmul", dsl="triton", round_num=2,
                                selected_recipes=[_synth("SYN-ALWAYS", "always", 1)],
                                rejected_recipes=[], roofline=dict(NAME_GUESS),
                                precision="fp16", hw_caps=unknown_caps("t"))
        self.assertEqual({}, plan.axis)
        text = plan.to_prompt()
        self.assertIn("plan() 未给出轴向裁决", text)
        self.assertNotIn("当前状态: bound=", text)


class TestTheVerdictIsVisibleInLogs(unittest.TestCase):
    def test_summary_carries_the_axis_decision(self):
        self.assertIn("axis=[on:memory]", _plan(MEMORY).summary())
        off = _plan(NAME_GUESS).summary()
        self.assertIn("axis=[off:", off)
        # 上面那例不检查具体是哪条门 (NAME_GUESS 两条门都适用), 这里只钉「off 也要出现在
        # 日志里」; 具体原因的归属由 prompt 那组用例负责。
        self.assertNotIn("axis=[off:]", off, "off 却没有原因 = 又变成一个没人读的标记")

    def test_plan_axis_is_a_record_not_a_rederived_guess(self):
        """`plan.axis` 必须真的带 bound_type —— 否则 prompt/summary 各自再读一遍
        roofline 块, 两处就会分叉 (一个说 memory 一个说 compute)。"""
        plan = _plan(MEMORY)
        self.assertTrue(plan.axis["usable"])
        self.assertEqual("memory", plan.axis["bound_type"])
        self.assertIn(plan.axis["bound_type"], plan.to_prompt())


def _code_text(path):
    """去掉注释和裸字符串 (docstring) 之后的源码文本, **保留原始行内布局**。

    两件事都要照顾到:
      * 必须去注释 —— `kernel_db.py` 在注释里写着「`KernelRecord` 里没有 `bound_type`
        的写者」, 按整行匹配会把这条登记成一个读者, 于是清单里出现一个根本不读这个值的
        文件, 而这份清单的用处恰恰是「它说读的地方都在」。
      * 不能把 token 拼成一串 —— 那样 `.get("bound_type", "unknown")` 会变成
        `.get ( "bound_type" , "unknown"` , 下面那条正则**永远匹配不到任何东西**,
        测试就退化成恒真 (与 `test_recipe_caps_gate` 里「断言要盯策略文本而不是 bound
        这个词」是同一类防空转)。
    """
    with io.open(path, encoding="utf-8") as fh:
        src = fh.read()
    lines = src.splitlines()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return src            # 读不出 token 就别剔注释, 让上层测试按原样匹配
    doc_lines, cuts = set(), {}
    for i, t in enumerate(toks):
        if t.type == tokenize.COMMENT:
            prev = cuts.get(t.start[0])
            cuts[t.start[0]] = t.start[1] if prev is None else min(prev, t.start[1])
        elif (t.type == tokenize.STRING and i + 1 < len(toks)
              and toks[i + 1].type == tokenize.NEWLINE
              # ⚠️ 「后面跟着 NEWLINE」不足以认出 docstring: `BOUND_AXIS_KEY = "bound_type"`
              # 也满足这一条, 而它是**真在读这个键**的一处代码。#87 第 3 片把键名收成常量之后
              # 就正好撞上这个形状 —— 只按后一个 token 判, 全仓就会对「键名被集中到常量里」
              # 的那些读者永久失明 (登记 CHECKLOG I35)。所以再加一条: 这个字符串必须是
              # **它所在那一行的第一个非空白 token**。
              and not lines[t.start[0] - 1][:t.start[1]].strip()):
            doc_lines.update(range(t.start[0], t.end[0] + 1))
    out = []
    for n, ln in enumerate(lines, 1):
        if n in doc_lines:
            continue
        out.append(ln[:cuts[n]] if n in cuts else ln)
    return "\n".join(out)


class TestBoundReadersAreRegistered(unittest.TestCase):
    """读者登记表: 谁在读这三个键, 必须逐个登记并写明它拿这个值做什么。

    缺口 3/4 反复出现的失效方式是「值算出来了, 用它的地方没人登记」, 于是改产出端时
    没人知道会波及谁。登记表**只锁文件集合**, 不锁每个文件的出现次数: 次数会随旁边的
    注释与无关改动漂移, 那种红只会让人把测试数字改成当前数字, 学不到任何东西。
    """

    REGISTERED = {
        # 产出端: 写 bound_type / roofline_bound / name_bound 三兄弟的地方。
        "decide/verifier.py": "产出端 (_roofline_analysis 综合算子名与脊点)",
        # 消费端 4 处, 每一处的用途都不同, 所以必须分别确认它认不认识 "unknown":
        "generate/optimization_planner.py": "配方筛选 + prompt (本次的门, 只有它经 axis_evidence)",
        "generate/agents.py": "search_optimize prompt 与 _rule_optimize 的 tiling 选择 (缺省 unknown)",
        "generate/triton_opt_rules.py": "normalize_bound_type + 按轴推荐 tiling (unknown 落 mixed)",
        "app/conductor.py": "血缘与 result.json 的落盘搬运",
        # #87 第 3 片新增的第 5 个消费端: 「这一轮的尺说的是哪条轴」这件事实。
        "decide/measurement.py": "只当**尺的身份**比较, 不据此选轴; 认得 unknown —— "
                          "`l3_ruler_reading` 明确把 `unknown` 剔出读数 (两轮都 unknown "
                          "不许被读成「同一把尺」, 见那里的 docstring)",
    }

    def test_reader_set_matches_the_registry(self):
        pat = re.compile(r"\b(bound_type|roofline_bound|name_bound)\b")
        found = {}
        # 2026-10-31: operator_agent/ 分五层，平铺 listdir 扫不到模块 ⇒ 递归
        pkg = os.path.join(REPO_ROOT, "operator_agent")
        for dirpath, _dirs, names in os.walk(pkg):
            for name in sorted(names):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                n = len(pat.findall(_code_text(path)))
                if n:
                    found[os.path.relpath(path, pkg).replace(os.sep, "/")] = n
        self.assertEqual(set(self.REGISTERED), set(found),
                         f"读者清单与代码不一致。新增读者: {set(found) - set(self.REGISTERED)}; "
                         f"已消失: {set(self.REGISTERED) - set(found)}。"
                         "新增的那个必须在 REGISTERED 里写明「拿这个值做什么」, "
                         "并确认它认识 \"unknown\" 这个第三态。")

    def test_the_stripper_sees_an_assigned_key_name_but_not_a_docstring(self):
        """`_code_text` 那条「去掉裸字符串」的规则, 两个方向都要有证人 (I35)。

        只查「后一个 token 是不是 NEWLINE」时, `BOUND_AXIS_KEY = "bound_type"` 会被当成
        docstring 整行丢掉 ⇒ 清单对「键名被收成常量」的那些读者永久失明。这一条就是那次
        失明的最小复现; 反向那一半 (docstring 里提一句不算读者) 是它原本的用处, 也要钉住,
        否则把规则放宽成「什么都不剔」同样全绿 (I19: 互斥判据要双向)。
        """
        import tempfile
        assigned = ('"""doc 里提一下 bound_type, 这句话不是代码。"""\n'
                    'BOUND_AXIS_KEY = "bound_type"\n')
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.py")
            with io.open(p, "w", encoding="utf-8") as fh:
                fh.write(assigned)
            stripped = _code_text(p)
        self.assertIn("bound_type", stripped,
                      "键名被收成常量 ⇒ 这个文件是读者, 不能被当成 docstring 剔掉")
        self.assertNotIn("这句话不是代码", stripped,
                         "docstring 那一行必须被剔掉, 不然清单会收进根本不读它的文件")

    def test_no_consumer_defaults_to_a_real_axis(self):
        """§八 第 17 条那个非对称默认今天**只存在于 planner 一处**, 这条断言就是那句结论
        的证据: planner 之外的每个消费者, 读这三个键时的缺省都不许是 "memory"/"compute"。

        generate/agents.py 与 generate/triton_opt_rules.py 都以 "unknown" 为缺省, 所以那两处不需要本次的
        门; 有人把一个缺省改成真轴时, 这条会红, 而那时「还有谁拿 bound 做决定」这句
        话就得重写 —— 这正是登记表存在的原因。
        """
        read_any = re.compile(
            r"""\.get\(\s*["'](bound_type|roofline_bound|name_bound)["']\s*,\s*"""
            r"""["']([^"']*)["']""")
        offenders = []
        for name in self.REGISTERED:
            if name == "generate/optimization_planner.py":
                continue      # 门在这里, 它自己就要求值 ∈ AXIS_VALUES
            for _, default in read_any.findall(_code_text(
                    os.path.join(REPO_ROOT, "operator_agent", name))):
                if default in AXIS_VALUES:
                    offenders.append((name, default))
        self.assertEqual([], offenders,
                         f"这些消费者把 bound 的缺省兜成了真轴, 与 planner 修掉的那条同形: "
                         f"{offenders}")

    def test_the_planner_gate_is_the_only_bound_consumer_that_gates(self):
        planner_src = _code_text(os.path.join(REPO_ROOT, "operator_agent",
                                             "generate/optimization_planner.py"))
        self.assertIn("AXIS_VALUES", planner_src,
                      "planner 的 bound 判断不再经过词表常量 —— 门可能被绕过")
        self.assertIn("axis_evidence(roofline)", planner_src,
                      "plan() 不再调用 axis_evidence, 那条 off 分支就是死代码")


class TestTheRealLibraryToday(unittest.TestCase):
    """反空转: 真实配方库今天在这条门上的形状, 说清楚哪些是「潜在」哪些是「live」。"""

    def test_real_library_has_compute_recipes_and_no_memory_recipes(self):
        p = OptimizationPlanner()
        compute = [r for r in p._all_recipes if r._conditions.get("compute_bound")]
        memory = [r for r in p._all_recipes if r._conditions.get("memory_bound")]
        self.assertTrue(compute, "真实库里一条 compute_bound 配方都没有, 那这组测试在测空气")
        self.assertEqual([], memory,
                         "真实库出现了 memory_bound 配方 —— 那么「unknown 拒绝它」这半边"
                         "就不再是潜在缺陷, 上面的合成配方用例应该换成真实配方")

    def test_gate_does_not_change_real_selection_between_round_one_and_the_guess(self):
        """真实库上的差分: 首轮 (roofline=None) 与「名字猜的 compute」在今天的配方库上
        选出**同一组**配方 —— 因为 memory_bound 条件 0 条。这句话就是「筛选面是潜在的、
        prompt 面才是 live 的」这个判断的证据, 不是顺手加的。"""
        real = OptimizationPlanner()
        kwargs = dict(kernel_type="matmul", dsl="triton", hw_caps=unknown_caps("t"),
                      precision="fp16", round_num=2)
        from_real = _ids(real.plan(roofline=None, **kwargs))
        from_guess = _ids(real.plan(roofline=NAME_GUESS, **kwargs))
        self.assertEqual(from_real, from_guess)
        # 但 prompt 不一样 —— 这才是那半边 live 的证据
        self.assertIn("瓶颈轴", real.plan(roofline=NAME_GUESS, **kwargs).to_prompt())


if __name__ == "__main__":
    unittest.main(verbosity=2)
