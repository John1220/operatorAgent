"""§21 Agent Runtime 的行为守卫（协议 / 权限 / 预算 / 七条独立检查 / 结构化上下文）

这份文件守四件事，每一件都对应设计 §21 的一节：

1. **§21.1 只接受结构化的动作**：schema / 权限 / 预算三道各自能拒，而拒绝理由
   必须是**出处的**（说得出哪一条、为什么），不是一句「不合格」；
2. **§21.2 结构性变更与局部修复的边界**：路由 → 允许的动作种类；
   「性能不达标不默认改代码」与「ABI 破坏 ⇒ 拒绝候选」是这条的两处硬边；
3. **§21.4 进编译前的独立检查**：七条设计原文 → 六条判据的映射完整；
   **只有 `violation` 才拒绝**，`unknown` / `not_applicable` 是结论而不是拒绝理由
   （把它们读成拒绝会让生成路径停摆，读成通过就是 fail-open —— 两个方向都得有守卫）；
4. **§21.3 上下文只有结构化的那八项**：没有第九个注入通道，截断必须留痕。

⚠️ 边界（必须一并读）：本模块**不是沙箱**。路径判据的对象是**声明的**角色路径
（§22.1 的 `candidate/`），强制是 §22 的事；`capability` 判据只覆盖「动作声明的能力」，
源码级幻觉 API 今天**判不了**（仓里没有任何语言的 API 面清单），它会出现在
`AuditCoverage.uncovered_rules` 里 —— 那一条也在下面有用例钉住。

运行: python3 -m unittest tests.test_agent_runtime -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent import agent_runtime as ar
from operator_agent.backend_caps import Cap

PARENT = "def kernel(A, B):\n    return A + B\n"
CHILD = "def kernel(A, B):\n    return A - B\n"


def _patch(new=CHILD, parent=PARENT, path="candidate/code.py"):
    return ar.Patch(path=path, new_source=new, parent_source=parent)


def _action(kind=ar.KIND_PROPOSE_SOURCE_PATCH, **kw):
    kw.setdefault("action_id", "r1-generate")
    kw.setdefault("rationale", "第 1 轮 generate")
    kw.setdefault("token_budget", 8192)
    if kind in ar.PATCH_KINDS:
        kw.setdefault("patch", _patch())
    return ar.AgentAction(kind=kind, **kw)


def _ctx(**kw):
    kw.setdefault("authorized_roots", (ar.CANDIDATE_DIR,))
    kw.setdefault("input_key_sets", (("A", "B"),))
    kw.setdefault("language", "python")
    return ar.ActionContext(**kw)


class TestTheSevenDesignChecksAreFullyMapped(unittest.TestCase):
    """设计 §21.4 的七条原文一条都不许丢 —— 丢一条就是「少了一道检查」而没人知道"""

    #: 设计 §21.4 的七条原文（照抄，包括标点）
    DESIGN = ("文件写入范围", "patch 是否只修改授权路径",
              "contract / harness / baseline 是否被修改", "依赖是否新增",
              "ABI 是否变化", "候选是否与父代重复", "是否引用不存在的能力")

    def test_every_design_item_has_a_rule(self):
        self.assertEqual(list(self.DESIGN), [item for item, _ in ar.DESIGN_CHECKS])
        for item, rule in ar.DESIGN_CHECKS:
            self.assertIn(rule, ar.CHECK_RULES, f"{item} 指向了一个不存在的判据")

    def test_every_rule_has_at_least_one_design_item(self):
        used = {rule for _, rule in ar.DESIGN_CHECKS}
        self.assertEqual(set(ar.CHECK_RULES), used,
                         "有判据没有任何设计条文指着它 —— 那就是自己发明的检查")

    def test_the_two_path_items_share_one_rule_on_purpose(self):
        """第 1、2 条是同一件事的两种说法（单文件候选上是同一个包含关系）"""
        m = dict((item, rule) for item, rule in ar.DESIGN_CHECKS)
        self.assertEqual(m["文件写入范围"], m["patch 是否只修改授权路径"])


class TestOnlyViolationsRefuse(unittest.TestCase):
    """三态纪律：`unknown` / `not_applicable` **不**拒绝，但也不许被读成通过"""

    def setUp(self):
        self.rt = ar.AgentRuntime(capabilities=[])

    def test_unknown_coverage_does_not_refuse(self):
        """没有授权根 ⇒ 判不了「在不在范围内」⇒ 放行，但覆盖率里明写没查成"""
        v = self.rt.admit(_action(), ar.ActionContext())
        self.assertTrue(v.accepted, v.refusal_note())
        self.assertEqual(ar.STATE_UNKNOWN, v.precompile.state_of("write_scope"))
        self.assertIn("write_scope", v.precompile.coverage.uncovered_rules)
        self.assertNotIn("write_scope", v.precompile.coverage.covered_rules)

    def test_a_violation_does_refuse(self):
        v = self.rt.admit(_action(patch=_patch(path="contract/contract.json")),
                          _ctx())
        self.assertFalse(v.accepted)
        self.assertEqual(ar.STATE_VIOLATION, v.precompile.state_of("protected_paths"))

    def test_check_states_are_a_closed_vocabulary(self):
        with self.assertRaises(ValueError):
            ar.CheckOutcome("write_scope", "passed")      # 表外结论
        with self.assertRaises(ValueError):
            ar.CheckOutcome("vibes", ar.STATE_OK)         # 表外判据

    def test_a_refused_action_has_no_precompile_when_schema_fails(self):
        """`precompile=None` 的含义是「没跑」，不是「跑过了没问题」"""
        v = self.rt.admit(_action(kind=ar.KIND_PROPOSE_SOURCE_PATCH, patch=None), _ctx())
        self.assertFalse(v.accepted)
        self.assertIsNone(v.precompile, "schema 没过却跑了独立检查")


class TestSchemaChecks(unittest.TestCase):

    def setUp(self):
        self.rt = ar.AgentRuntime()

    def test_every_problem_has_a_reason(self):
        cases = {
            "空 action_id": _action(action_id=""),
            "空 rationale": _action(rationale="   "),
            "表外的种类": ar.AgentAction("a", "rewrite_everything", "试试"),
            "非正预算": _action(token_budget=0),
            "表外工具": _action(requested_tools=("ncu",)),
            "缺 patch": _action(patch=None),
            "空源码": _action(patch=ar.Patch(path="candidate/code.py", new_source="  ")),
            "空路径": _action(patch=ar.Patch(path="", new_source=CHILD)),
            "解释却带 patch": _action(kind=ar.KIND_EXPLAIN_FAILURE, patch=_patch()),
            "请求却带 patch": _action(kind=ar.KIND_REQUEST_NEW_SHAPE,
                                      patch=_patch()),
        }
        for name, action in cases.items():
            v = self.rt.admit(action, _ctx())
            self.assertFalse(v.accepted, f"{name} 居然通过了 schema")
            self.assertTrue(v.schema_problems, f"{name} 没有给出理由")
            self.assertTrue(all(len(p) > 10 for p in v.schema_problems),
                            f"{name} 的理由太短, 不像能照着修: {v.schema_problems}")

    def test_a_well_formed_side_effect_free_action_passes_schema(self):
        v = self.rt.admit(_action(kind=ar.KIND_EXPLAIN_FAILURE), _ctx())
        self.assertEqual((), v.schema_problems)
        self.assertTrue(v.accepted)


class TestPermissionFollowsTheRoute(unittest.TestCase):
    """§21.2 的表：路由决定允许的动作种类"""

    def setUp(self):
        self.rt = ar.AgentRuntime(capabilities=[])

    def _v(self, *, kind=ar.KIND_PROPOSE_SOURCE_PATCH, action_kw=None, **ctx):
        """裁决一次。`action_kw` 里的东西属于**动作**（它是被裁决的对象）,
        其余关键字属于**上下文**（这一轮的环境）—— 两者不许混。"""
        return self.rt.admit(_action(kind=kind, **(action_kw or {})), _ctx(**ctx))

    def _accepted(self, kind, action_kw=None, **ctx):
        return self._v(kind=kind, action_kw=action_kw, **ctx).accepted

    def test_compile_error_licenses_local_repair_only(self):
        repair = {"parent_candidate": "src:abc"}
        self.assertTrue(self._accepted(ar.KIND_REPAIR_COMPILE, repair,
                                       route_kind="compile"))
        self.assertFalse(self._accepted(ar.KIND_REPAIR_CORRECTNESS, repair,
                                        route_kind="compile"))
        self.assertFalse(self._accepted(ar.KIND_PROPOSE_SOURCE_PATCH,
                                        route_kind="compile"))

    def test_numeric_error_licenses_correctness_repair_only(self):
        kwargs = {"parent_candidate": "src:abc",
                  "rationale": "第 2 轮 fix; 误差形态=tail_block"}
        self.assertTrue(self._accepted(ar.KIND_REPAIR_CORRECTNESS, kwargs,
                                       route_kind="numeric", error_shape="tail_block"))
        self.assertFalse(self._accepted(ar.KIND_REPAIR_COMPILE, kwargs,
                                        route_kind="numeric", error_shape="tail_block"))

    def test_a_repair_must_name_the_candidate_it_repairs(self):
        v = self._v(kind=ar.KIND_REPAIR_COMPILE, route_kind="compile")
        self.assertFalse(v.accepted)
        self.assertTrue(any("parent_candidate" in p for p in v.permission_problems),
                        v.permission_problems)

    def test_numeric_repair_must_carry_the_error_shape(self):
        """§21.2 的「必须带误差形态」—— 形态已知时必须出现在理由里"""
        bad = self._v(kind=ar.KIND_REPAIR_CORRECTNESS,
                      action_kw={"parent_candidate": "src:abc",
                                 "rationale": "随便改改"},
                      route_kind="numeric", error_shape="tail_block")
        self.assertFalse(bad.accepted)
        self.assertIn("tail_block", bad.refusal_note())
        good = self._v(kind=ar.KIND_REPAIR_CORRECTNESS,
                       action_kw={"parent_candidate": "src:abc",
                                  "rationale": "按形态 tail_block 修边界守卫"},
                       route_kind="numeric", error_shape="tail_block")
        self.assertTrue(good.accepted, good.refusal_note())

    def test_an_unknown_error_shape_is_not_a_refusal(self):
        """形态判不了(三态) 时不许因此拒绝 —— 那会把「判不了」读成「有错」

        与 `failure_route` 的口径同源: 形态是**证据不是判据**。
        """
        v = self._v(kind=ar.KIND_REPAIR_CORRECTNESS,
                    action_kw={"parent_candidate": "src:abc",
                               "rationale": "按首坏点修"},
                    route_kind="numeric", error_shape=None)
        self.assertTrue(v.accepted, v.refusal_note())

    def test_performance_route_does_not_license_a_rewrite_by_default(self):
        """「性能不达标 ⇒ 进入搜索, 不默认改代码」"""
        v = self._v(route_kind="performance",
                    escalation_refusals=("参数搜索已平台化",))
        self.assertFalse(v.accepted)
        self.assertIn("结构升级门未放行", v.refusal_note())
        self.assertIn("参数搜索已平台化", v.refusal_note())

    def test_performance_route_licenses_a_rewrite_with_the_escalation_gate(self):
        v = self._v(route_kind="performance", escalation_allowed=True)
        self.assertTrue(v.accepted, v.refusal_note())
        self.assertTrue(v.structural_licensed)

    def test_a_template_point_is_the_default_search_action(self):
        self.assertTrue(self._accepted(ar.KIND_PROPOSE_TEMPLATE,
                                       route_kind="performance"))

    def test_a_broken_abi_refuses_every_candidate_changing_action(self):
        for kind in ar.PATCH_KINDS:
            kw = {"parent_candidate": "src:abc"} if kind in ar.REPAIR_KINDS else {}
            v = self._v(kind=kind, action_kw=kw, abi_broken=True)
            self.assertFalse(v.accepted, f"{kind} 在 ABI 破坏时还被放行")
        # 不改候选的动作不受这条影响（拒绝候选 ≠ 不许解释）
        self.assertTrue(self._v(kind=ar.KIND_EXPLAIN_FAILURE,
                                abi_broken=True).accepted)

    def test_the_route_free_first_round_is_a_proposal_not_a_repair(self):
        """首轮没有失败在驱动它 ⇒ 允许提议，不允许声称在「修」"""
        self.assertTrue(self._accepted(ar.KIND_PROPOSE_SOURCE_PATCH))
        self.assertFalse(self._accepted(ar.KIND_REPAIR_COMPILE))

    def test_infra_and_silent_routes_do_not_trim_permissions(self):
        """§21.2 的表**没有**这两行 ⇒ 不据此裁剪（那会与 §8 的路由自己打架）"""
        for kind in ar.ACTION_KINDS:
            allow = set(ar._allowed_kinds(_ctx(route_kind="infra")))
            self.assertIn(kind, allow, f"infra 路由把 {kind} 裁掉了")
            allow_hold = set(ar._allowed_kinds(_ctx(route_kind="silent")))
            self.assertIn(kind, allow_hold, f"silent 路由把 {kind} 裁掉了")


class TestBudgetChecks(unittest.TestCase):

    def setUp(self):
        self.rt = ar.AgentRuntime()

    def test_no_cap_means_no_refusal_and_no_pretence(self):
        v = self.rt.admit(_action(token_budget=10 ** 9), _ctx(tokens_left=None))
        self.assertTrue(v.accepted)
        self.assertEqual((), v.budget_problems)

    def test_an_over_ask_is_refused(self):
        v = self.rt.admit(_action(token_budget=8192), _ctx(tokens_left=100))
        self.assertFalse(v.accepted)
        self.assertIn("8192", v.budget_problems[0])

    def test_an_exhausted_budget_refuses_an_action_that_has_not_spent_yet(self):
        v = self.rt.admit(_action(token_budget=10), _ctx(tokens_left=0))
        self.assertFalse(v.accepted)
        self.assertIn("用尽", v.budget_problems[0])

    def test_a_committed_spend_is_reported_but_not_refused(self):
        """钱花过了 ⇒ 预算问题照记, 但不据此拒绝 (拒绝省不下 token, 只会毁掉这份候选)

        这是 §10.2「超了退化成纯数值搜索」与 §21.1「只接受通过预算校验的动作」的分界：
        预算校验的作用是**事前**拦住花费, 而本仓的候选是「先生成、后提交」。
        """
        v = self.rt.admit(_action(token_budget=8192),
                          _ctx(tokens_left=-50, spend_committed=True))
        self.assertTrue(v.accepted, v.refusal_note())
        self.assertTrue(v.budget_problems, "预算问题没有记账")
        self.assertIn("退化成纯数值搜索", v.budget_deferred_note)
        self.assertIn("剩 -50", v.budget_deferred_note)   # 记账的是**真口径**, 不是声明值


class TestPrecompileChecks(unittest.TestCase):
    """§21.4 的六条判据逐条：能拒的拒，判不了的**说判不了**"""

    def setUp(self):
        self.rt = ar.AgentRuntime()

    def _v(self, *, kind=ar.KIND_PROPOSE_SOURCE_PATCH, action_kw=None, **ctx):
        return self.rt.admit(_action(kind=kind, **(action_kw or {})), _ctx(**ctx))

    def _one(self, rule, *, kind=ar.KIND_PROPOSE_SOURCE_PATCH, action_kw=None, **ctx):
        """裁决一次并取出某一条判据的结论（`action_kw` 属于动作, 其余属于上下文）"""
        v = self._v(kind=kind, action_kw=action_kw, **ctx)
        out = next(o for o in v.precompile.outcomes if o.rule == rule)
        return v, out

    def test_write_scope_contains_relative_and_rejects_escape(self):
        _, ok = self._one("write_scope")
        self.assertEqual(ar.STATE_OK, ok.state)
        _, bad = self._one("write_scope", action_kw={"patch": _patch(path="../contract/x.py")})
        self.assertEqual(ar.STATE_VIOLATION, bad.state)
        _, abs_bad = self._one("write_scope", action_kw={"patch": _patch(path="/etc/code.py")})
        self.assertEqual(ar.STATE_VIOLATION, abs_bad.state)

    def test_every_protected_dir_is_refused(self):
        for name in ar.PROTECTED_DIR_NAMES:
            _, out = self._one("protected_paths",
                               action_kw={"patch": _patch(path=f"candidate/{name}/x.py")})
            self.assertEqual(ar.STATE_VIOLATION, out.state, f"{name} 没被拦住")
        # 设计 §22.1 点名的五个, 少一个就是少一道门
        self.assertEqual(("contract", "harness", "baseline", "measurements", "report"),
                         ar.PROTECTED_DIR_NAMES)

    def test_new_dependency_rejects_only_new_external_modules(self):
        _, builtin = self._one("new_dependency",
                               action_kw={"patch": _patch(new="import math\n" + CHILD)})
        self.assertEqual(ar.STATE_OK, builtin.state, "标准库被当成新依赖了")
        _, external = self._one("new_dependency",
                                action_kw={"patch": _patch(new="import torch\n" + CHILD)})
        self.assertEqual(ar.STATE_VIOLATION, external.state)
        self.assertIn("torch", external.detail)

    def test_new_dependency_has_no_baseline_on_a_first_proposal(self):
        _, out = self._one("new_dependency", action_kw={"patch": _patch(parent="")})
        self.assertEqual(ar.STATE_NOT_APPLICABLE, out.state)
        self.assertIn("可比基线", out.detail)

    def test_new_dependency_prefers_the_admitted_baseline(self):
        """I78: 基线优先取「上一次**过门**的候选」, 不是 patch 的父代

        真机一手 (见 CHECKLOG I78 / `benchmarks/agent_action_h20.py`): 修复产物夹带
        `import torch`, 第 2 轮被拒; 第 3 轮 Fixer 又交同一份 —— 那时 patch 的父代是
        **上一轮那份被拒的候选**, 于是 `torch` 不再是「新增」, 门放行、真机编译真的发生。
        这里把两种基线摆在一起判: 同一份候选在「父代=被拒的候选」下必须仍然违约。
        """
        accepted_parent = "import numpy as np\n" + CHILD
        refused_parent = "import numpy as np\nimport torch\n" + CHILD.replace(
            "return A + B", "return torch.matmul(A, B)")
        new = "import numpy as np\nimport torch\n" + CHILD.replace(
            "return A + B", "return torch.matmul(A, B)")
        # 没有基线字段时的老口径: 对着**被拒的**父代 ⇒ 看不出新增 (这就是那个洞)
        _, old_reading = self._one("new_dependency", action_kw={
            "patch": _patch(new=new, parent=refused_parent)})
        self.assertEqual(ar.STATE_OK, old_reading.state)
        # 给了「过门的候选」当基线 ⇒ 同一份候选违约
        _, fixed = self._one("new_dependency",
                             action_kw={"patch": _patch(new=new, parent=refused_parent)},
                             dep_baseline_source=accepted_parent)
        self.assertEqual(ar.STATE_VIOLATION, fixed.state)
        self.assertIn("torch", fixed.detail)
        # 基线存在时, patch 连父代都没有也判得了 (否则「首次提议」那一格会把洞留着)
        _, no_parent = self._one("new_dependency",
                                 action_kw={"patch": _patch(new=new, parent="")},
                                 dep_baseline_source=accepted_parent)
        self.assertEqual(ar.STATE_VIOLATION, no_parent.state)

    def test_abi_uses_the_same_judge_as_l1(self):
        _, ok = self._one("abi")
        self.assertEqual(ar.STATE_OK, ok.state)
        _, bad = self._one("abi", action_kw={
            "patch": _patch(new="def kernel(A):\n    return A\n")})
        self.assertEqual(ar.STATE_VIOLATION, bad.state)
        self.assertIn("TypeError", bad.detail)

    def test_abi_without_input_names_is_not_applicable(self):
        _, out = self._one("abi", input_key_sets=())
        self.assertEqual(ar.STATE_NOT_APPLICABLE, out.state)
        self.assertIn("没有可判的对象", out.detail)

    def test_duplicate_of_parent_refuses_only_a_repair(self):
        """doc 47 的口径：生成类重复只观测；**一字未改的修复**才拒"""
        # 结构相同、**文本不同**（多一行注释）—— α 等价的 AST 哈希才认得出这种重复;
        # 签名与父代一致, 否则先撞上 ABI 那条, 这条用例就测不到重复判据了。
        same = _patch(new="def kernel(A, B):\n    # 只是加了行注释\n    return A + B\n",
                      parent=PARENT)
        v_gen, out_gen = self._one("duplicate_of_parent", action_kw={"patch": same})
        self.assertEqual(ar.STATE_UNKNOWN, out_gen.state)
        self.assertTrue(v_gen.accepted, "生成类重复被拒绝了 (doc 47 的口径是只观测)")
        v_fix, out_fix = self._one(
            "duplicate_of_parent", kind=ar.KIND_REPAIR_COMPILE,
            action_kw={"parent_candidate": "src:abc", "patch": same},
            route_kind="compile", parent_verified=True)
        self.assertEqual(ar.STATE_VIOLATION, out_fix.state)
        self.assertFalse(v_fix.accepted)

    def test_capability_refuses_a_bottleneck_without_a_gauge(self):
        for key in ("spill-bound", "bank-conflict"):
            v, out = self._one("capability",
                               action_kw={"expected_bottleneck": key})
            self.assertEqual(ar.STATE_VIOLATION, out.state, f"{key} 被当成了有量具的瓶颈")
            self.assertFalse(v.accepted)
        # 有量具的那两个放行（与 `escalation.GAUGES` 同源）
        gauged = ar.escalation.gauged_bottlenecks()
        self.assertEqual(("smem-bound", "occupancy-bound"), gauged)
        # 有量具的瓶颈不再是违约 —— 这一条的结论落在「源码级那一半判不了」(not_applicable)
        _, ok = self._one("capability",
                          action_kw={"expected_bottleneck": gauged[0]})
        self.assertNotEqual(ar.STATE_VIOLATION, ok.state, ok.detail)

    def test_capability_refuses_a_tool_the_backend_does_not_have(self):
        v, out = self._one("capability",
                           action_kw={"requested_tools": (ar.TOOL_VERIFY_NUMERICS,)})
        self.assertEqual(ar.STATE_VIOLATION, out.state)
        self.assertIn("remote_verify", out.detail)
        # 后端声明了能力就放行：判据读的是**声明**，不是猜的
        rt = ar.AgentRuntime(capabilities=[Cap.REMOTE_VERIFY])
        v2 = rt.admit(_action(requested_tools=(ar.TOOL_VERIFY_NUMERICS,)), _ctx())
        self.assertTrue(v2.accepted, v2.refusal_note())
        self.assertNotEqual(ar.STATE_VIOLATION,
                            v2.precompile.state_of("capability"))

    def test_the_two_never_implemented_tools_are_refused_with_a_reason(self):
        """`profiler` / `shape_sweep` 今天没有实现物 —— 拒绝理由必须说得出这件事"""
        for kind, tool in ((ar.KIND_REQUEST_PROFILER, ar.TOOL_PROFILER),
                           (ar.KIND_REQUEST_NEW_SHAPE, ar.TOOL_SHAPE_SWEEP)):
            self.assertIsNone(ar.TOOLS[tool], f"{tool} 今天有实现了, 状态文档要同步改")
            v = self.rt.admit(_action(kind=kind, requested_tools=(tool,)), _ctx())
            self.assertFalse(v.accepted)
            self.assertIn("没有实现物", v.refusal_note())

    def test_the_source_level_hallucinated_api_half_is_reported_uncovered(self):
        """「查不了」必须出现在覆盖表里，而不是被读成「没有幻觉 API」"""
        v = self.rt.admit(_action(), _ctx())
        self.assertEqual(ar.STATE_UNKNOWN, v.precompile.state_of("capability"))
        self.assertIn("capability", v.precompile.coverage.uncovered_rules)
        self.assertIn("API 面清单", next(o.detail for o in v.precompile.outcomes
                                         if o.rule == "capability"))

    def test_a_rule_without_an_object_is_not_a_coverage_gap(self):
        """`not_applicable` 不进覆盖表: 「这一份候选没有这个对象」不是「这一层查不了」(I90)

        没有父代 ⇒ `new_dependency` / `duplicate_of_parent` 两条**没有对象**。它们的名字
        只出现在 `outcomes` 里，覆盖表两端都不收 —— 否则「七条里四条没覆盖」会被读成
        「这一层的静态检查很弱」，而其中两条只是「这一轮不适用」。"""
        v = self.rt.admit(_action(patch=_patch(parent="")), _ctx())
        for rule in ("new_dependency", "duplicate_of_parent"):
            self.assertEqual(ar.STATE_NOT_APPLICABLE, v.precompile.state_of(rule), rule)
            self.assertNotIn(rule, v.precompile.coverage.uncovered_rules, rule)
            self.assertNotIn(rule, v.precompile.coverage.covered_rules, rule)
        self.assertTrue(v.accepted, "`not_applicable` 不是拒绝理由")

    def test_a_patchless_action_still_reports_every_rule(self):
        """没有 patch 的动作也要给全六条结论 —— 缺行会被读成「都通过了」"""
        v = self.rt.admit(_action(kind=ar.KIND_EXPLAIN_FAILURE), _ctx())
        self.assertEqual(set(ar.CHECK_RULES),
                         {o.rule for o in v.precompile.outcomes})

    def test_coverage_modes_come_from_the_language_verdict(self):
        v = self.rt.admit(_action(), _ctx(language="cuda_c"))
        self.assertEqual("regex_only", v.precompile.coverage.mode)
        v_none = self.rt.admit(_action(), _ctx(language="verilog"))
        self.assertEqual("none", v_none.precompile.coverage.mode)
        # 不适用的语言里, 依赖与 ABI 都判不了 —— 不是「没问题」
        self.assertEqual(ar.STATE_NOT_APPLICABLE,
                         v_none.precompile.state_of("new_dependency"))
        self.assertEqual(ar.STATE_NOT_APPLICABLE,
                         v_none.precompile.state_of("abi"))


class TestProposalContext(unittest.TestCase):
    """§21.3：上下文只有那八项，截断留痕，工具/预算从真源来"""

    def setUp(self):
        self.rt = ar.AgentRuntime(capabilities=[Cap.DEP_ANALYSIS],
                                  authorized_roots=("candidate",))

    def test_the_rendered_context_has_exactly_the_design_sections(self):
        ctx = self.rt.context_for(contract_summary="契约: matmul fp32",
                                  patch=_patch(), first_error="第 3 行越界")
        text = ctx.render()
        heads = [ln for ln in text.splitlines() if ln.startswith("### ")]
        self.assertEqual(["### 当前契约摘要", "### 父候选 diff", "### 最近一次失败分类",
                          "### 首个错误与误差形态", "### 相关 profiler 指标",
                          "### 可用工具列表", "### 剩余预算", "### 允许修改的文件"], heads)
        self.assertIn("契约: matmul fp32", text)
        self.assertIn("analyze_dependencies", text)
        self.assertIn("candidate", text)

    def test_an_unknown_error_shape_is_rendered_as_unknown_not_as_a_guess(self):
        text = self.rt.context_for().render()
        self.assertIn("误差形态未确立", text)

    def test_a_huge_error_log_is_clipped_with_a_visible_marker(self):
        ctx = self.rt.context_for(first_error="X" * (ar.MAX_ERROR_CHARS * 3))
        text = ctx.render()
        self.assertIn("已截断", text)
        self.assertIn("first_error", ctx.truncations())
        self.assertIn("truncated_sections", ctx.as_dict(), "截断没有进产物")

    def test_raw_metric_blobs_are_not_rendered(self):
        """全量 NCU 指标的天真做法就是往上下文里塞 dict —— 这里结构上做不到"""
        ctx = self.rt.context_for(profiler_metrics={
            "gflops": 12.5, "smem_kb": 96,
            "raw_ncu_dump": {"a": 1, "b": [1, 2]},
        })
        text = ctx.render()
        self.assertIn("gflops: 12.5", text)
        self.assertNotIn("raw_ncu_dump", text, "一个 dict 指标被渲染进上下文了")
        self.assertIn("raw_ncu_dump", ctx.as_dict()["profiler_metrics"],
                      "产物里也不该丢（读者是复盘, 不是 prompt）")

    def test_too_many_metrics_are_counted_not_silently_dropped(self):
        metrics = {f"m{i}": i for i in range(ar.MAX_PROFILER_KEYS + 3)}
        ctx = self.rt.context_for(profiler_metrics=metrics)
        self.assertIn(f"(+3 项未渲染", ctx.render())
        self.assertIn("profiler_metrics", ctx.truncations())

    def test_the_context_dict_matches_the_rendered_fields(self):
        ctx = self.rt.context_for(contract_summary="c", first_error="e")
        d = ctx.as_dict()
        self.assertEqual({"contract_summary", "parent_diff", "last_failure_kind",
                          "first_error", "error_shape", "profiler_metrics",
                          "available_tools", "budget_left", "writable_paths",
                          "truncated_sections"}, set(d))


class TestPatchIsAuditableAndRollbackable(unittest.TestCase):

    def test_diff_and_line_counts_track_the_change(self):
        p = _patch()
        self.assertIn("--- a/candidate/code.py", p.diff())
        self.assertIn("+++ b/candidate/code.py", p.diff())
        self.assertEqual((1, 1), p.changed_lines())
        self.assertEqual(PARENT, p.rollback_source())
        self.assertEqual(p.parent_hash(), ar.Patch(path="candidate/code.py",
                                                   new_source=PARENT).source_hash())

    def test_no_parent_means_no_rollback_point_and_no_diff(self):
        p = ar.Patch(path="candidate/code.py", new_source=CHILD)
        self.assertEqual("", p.diff())
        self.assertEqual((0, 0), p.changed_lines())
        self.assertIsNone(p.rollback_source(), "没有父代被读成了「回滚到空源码」")
        self.assertFalse(p.has_parent)

    def test_the_action_dict_carries_both_hashes(self):
        d = _action().as_dict()
        self.assertEqual({"path", "source_hash", "parent_hash", "has_parent",
                          "added_lines", "removed_lines"}, set(d["patch"]))
        json.dumps(d)   # 落进 iteration_history 那一步不许崩


class TestRuntimeBookkeeping(unittest.TestCase):

    def test_rejections_are_counted_per_kind_and_reset_per_run(self):
        rt = ar.AgentRuntime()
        rt.admit(_action(patch=_patch(path="contract/x.py")), _ctx())
        rt.admit(_action(kind=ar.KIND_REPAIR_COMPILE), _ctx(route_kind="compile"))
        rt.admit(_action(), _ctx())
        self.assertEqual(1, rt.admitted)
        self.assertEqual({ar.KIND_PROPOSE_SOURCE_PATCH: 1, ar.KIND_REPAIR_COMPILE: 1},
                         rt.rejected)
        rt.reset_run()
        self.assertEqual((0, {}), (rt.admitted, rt.rejected))

    def test_unavailable_tools_say_why(self):
        rt = ar.AgentRuntime(capabilities=[])
        why = rt.unavailable_tools()
        self.assertEqual(set(ar.TOOLS), set(why))
        for name, reason in why.items():
            self.assertTrue(reason, f"{name} 只说了「没有」, 没说为什么")
        self.assertEqual((), rt.available_tools())

    def test_the_runtime_dict_is_json_serializable(self):
        rt = ar.AgentRuntime(capabilities=[Cap.DEP_ANALYSIS], authorized_roots=("candidate",))
        json.dumps(rt.as_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)