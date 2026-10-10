"""RunState 的字段清单是单一真源 —— 用断言钉住它的五条外部契约。

Stage1 把 _execute_inner 的收尾阶段从 24 个关键字形参改成 state 一个对象, Stage2 又把
1128 行的轮次循环体搬成 _run_round。这两步真正的风险不是「搬错了一行」(那是
tests/test_execute_inner_behavior.py 的事), 而是**字段清单悄悄漂移**:

  (a) 快照恢复清单多一项、RunState 忘了跟 -> resume 之后少恢复一个状态;
  (b) RunState 加了字段但没人读           -> 多一份能互相跑偏的副本
                                             (本仓明文的「新增字段必须同时新增读者」);
  (c) 构造点与字段清单脱节                -> 少初始化一个字段, 或传了个不是字段的名字;
  (d) _finalize_run 的接口又长回一串形参  -> 这次拆分就白做了;
  (e) _run_round 里出现「可能未绑定就读取」的名字, 而它在循环之前本来有值
                                         -> 不再是崩溃, 而是静默拿到上一轮/初值。

(e) 不是假想, Stage2 搬循环体时真踩到了两次。第一次是 verified_parent: 同一轮里先读后写,
第一轮就读到自己还没赋的值 -> UnboundLocalError。第二次更隐蔽: last_tunable_perf 赋值在
「if _can_tune」那一支、读在「elif」那一支 —— 光比行号顺序看不出来 (读的行号在写的后面),
得按**可能未绑定就读取**的数据流分析才现形。这些名字必须进 state。它们不在快照里
(进程重启后回到 setup 初值, 这是既有行为)。

判据的两半缺一不可: 「可能未绑定就读取」(数据流) 与「循环之前本来有值」(作用域)。只有两半
都成立才是这次搬运引入的静默旧值 —— 循环之前没有的名字若可能未绑定就读取, 那它在搬运之前
也会崩, 与本次改动无关。实测: 单看数据流会多报 verify_results / _outcome / fix_note /
suggestion 四个 (它们只在 try 体里赋值, 分析保守地不认), 加上作用域那半就正好归零。
五条都写成机械断言, 每条都配一个反向标定 (在内存里造出那个漂移, 断言判据变红),
免得判据只是看着像判据。
"""
import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONDUCTOR = Path(__file__).resolve().parents[1] / "operator_agent" / "conductor.py"
NL = chr(10)

#: _restore_state_from_snapshot 的恢复清单 -> RunState 字段名。两处故意不同名:
#:   vr_list           -> all_verification_results (那个方法里的局部别名)
#:   iteration_history -> None: 循环里对它的唯一用法是 artifact.iteration_history,
#:     它始终是同一个对象上的同一个列表。收进 RunState 只会平白多出一份能互相跑偏的副本。
#: 注意这张表只覆盖**快照**那一部分 —— 上面 (e) 点名的 4 个跨轮记忆不在快照里。
RESTORE_TO_FIELD = {
    "start_round": "start_round",
    "current_sketch": "current_sketch",
    "current_code": "current_code",
    "pending_fix": "pending_fix",
    "iteration_history": None,
    "best_gflops": "best_gflops",
    "rounds_without_improvement": "rounds_without_improvement",
    "optimization_history": "optimization_history",
    "applied_recipes": "applied_recipes",
    "disabled_recipes": "disabled_recipes",
    "vr_list": "all_verification_results",
}


def _tree(src):
    return ast.parse(src)


def _func(tree, name):
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _run_state_fields(src):
    rs = next(c for c in _tree(src).body
              if isinstance(c, ast.ClassDef) and c.name == "RunState")
    return [n.target.id for n in rs.body if isinstance(n, ast.AnnAssign)]


def _round_loop(tree):
    fn = _func(tree, "_execute_inner")
    return next(n for n in fn.body if isinstance(n, ast.For)
                and "state.start_round" in ast.unparse(n.iter))


def _constructor_keywords(src):
    fn = _func(_tree(src), "_execute_inner")
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "RunState")
    return [k.arg for k in call.keywords]


def _restored_names(src):
    rs = _func(_tree(src), "_restore_state_from_snapshot")
    ret = next(n for n in ast.walk(rs) if isinstance(n, ast.Return))
    return [ast.unparse(e) for e in ret.value.elts]


#: 轮次的组成部分 —— state.<名字> 的读者要在这些方法里找。Stage3 从 _run_round 里又
#: 搬出了三段 (轮首门 / _snap / 收尾), 所以这份清单跟着长: 少一个方法, 它读的字段就会被
#: 误报成「新增字段没有读者」。
_ROUND_PARTS = ("_run_round", "_round_enter_or_stop", "_snap", "_round_close_out",
                "_finalize_run")


def _state_reads(src):
    """轮次的各个部分 (循环 + 搬出去的阶段方法 + 收尾) 里 state.<名字> 的字段集合。"""
    tree = _tree(src)
    out = set()
    for node in [_round_loop(tree)] + [_func(tree, name) for name in _ROUND_PARTS]:
        for x in ast.walk(node):
            if (isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)
                    and x.value.id == "state"):
                out.add(x.attr)
    return out


#: Stage3 拆出来的三个阶段方法, 以及它们对外的接口 (依赖清单)。接口就是契约: 多一个形参
#: 不是「顺手」, 是这个阶段又多赖了一层外面的状态。
_PHASE_PARTS = (("_round_enter_or_stop", ["self", "state", "round_num"]),
                ("_snap", ["self", "state", "round_num"]),
                ("_round_close_out", ["self", "state", "round_num", "_esc_decision",
                                      "_measure_s", "_space", "bandit_arm", "l3_result"]))


def _phase_problems(src):
    """Stage3 的三个阶段还在不在、接口有没有变、_run_round 里有没有长出嵌套函数。

    这些漂移**都是行为等价的** (把门内联回去还是那两道门; 闭包读的 3 个名字恰好都在
    外层), 所以只有形状判据能拦 —— 全量 2572 条与行为摘要都不会说话。
    """
    tree = _tree(src)
    out = []
    rr = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == "_run_round"), None)
    if rr is None:
        return ["_run_round 不见了"]
    gate = [s for s in rr.body if isinstance(s, ast.If)
            and "_round_enter_or_stop" in ast.unparse(s.test)
            and "_ROUND_STOP" in ast.unparse(s.test)]
    if len(gate) != 1:
        out.append("轮首门没有以「委派」的形式出现在 _run_round 里 (被内联回来或写法变了)")
    if not (isinstance(rr.body[-1], ast.Return)
            and "_round_close_out" in ast.unparse(rr.body[-1])):
        out.append("_run_round 的最后一步不是收尾阶段 _round_close_out")
    nested = [n.name for n in ast.walk(rr) if isinstance(n, ast.FunctionDef) and n is not rr]
    if nested:
        out.append("_run_round 里长出了嵌套函数 (闭包会让依赖清单消失): %s" % nested)
    for name, params in _PHASE_PARTS:
        fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == name), None)
        if fn is None:
            out.append("阶段方法 %s 不见了" % name)
            continue
        got = [a.arg for a in fn.args.posonlyargs + fn.args.args]
        got += [a.arg for a in fn.args.kwonlyargs]
        if got != params:
            out.append("%s 的接口变了: %s != %s" % (name, got, params))
    return out


def _round_level_control(src):
    """_run_round 里绑到**这个方法自己**的 break/continue (不含内层循环与嵌套函数的)。"""
    out = []

    def walk(node, in_inner):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(ch, (ast.Break, ast.Continue)) and not in_inner:
                out.append((type(ch).__name__, ch.lineno))
                continue
            walk(ch, in_inner or isinstance(ch, (ast.For, ast.While)))

    walk(_func(_tree(src), "_run_round"), False)
    return out


def _outerly_bound(src):
    """_execute_inner 进入轮次循环之前就有值的名字 (含它自己的形参)。

    判据的另一半。一个名字若在循环之前就有值, 那它在 _run_round 里「可能未绑定就读取」
    就不再是崩溃, 而是**静默拿到上一轮留下的值 / setup 初值** —— Stage2 那 6 个字段
    (verified_parent / verified_parent_round / snapshot_target_dir / search_alternatives /
    plan / last_tunable_perf) 正是这么来的。
    """
    fn = _func(_tree(src), '_execute_inner')
    out = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}

    def own(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                out.add(ch.name)
                continue          # 嵌套函数自己的赋值是它自己的局部
            if isinstance(ch, ast.Name) and isinstance(ch.ctx, (ast.Store, ast.Del)):
                out.add(ch.id)
            elif isinstance(ch, ast.ExceptHandler) and ch.name:
                out.add(ch.name)
            own(ch)

    for st in fn.body:
        if isinstance(st, ast.For) and 'state.start_round' in ast.unparse(st.iter):
            break
        own(st)
    return out


def _unbound_local_reads(src):
    """_run_round 里「可能未绑定就读取」且「循环之前本来有值」的本层局部。

    保守的**定值分析**, 按语句顺序推进: if 两支都赋才算赋; 循环体与 try 体里的赋值都不算
    (可能没跑到); with ... as / except ... as / 嵌套函数的读照算。方向是「可能有」而不是
    「一定有」—— 对判据来说这个方向正好: 宁可多报一个, 也不放过一个静默的旧值。

    嵌套函数体里的**读**是对本方法局部的闭包读, 所以在 def 那一行保守判一次。
    """
    rr = _func(_tree(src), '_run_round')
    params = {a.arg for a in rr.args.posonlyargs + rr.args.args + rr.args.kwonlyargs}
    outer, own, bad = _outerly_bound(src), set(), []

    def no_nested(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            yield ch
            yield from no_nested(ch)

    for st in rr.body:
        for n in no_nested(st):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                own.add(n.id)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                own.add(n.name)
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            own.add(st.name)

    def bind(target, bound):
        if isinstance(target, ast.Name):
            bound.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for e in target.elts:
                bind(e, bound)
        elif isinstance(target, ast.Starred):
            bind(target.value, bound)

    def check(node, bound):
        # no_nested 走的是子节点, 所以 node 自己是 Name 时要补上 —— 否则
        # 「x = plan」这种直接拿裸名当右值的读会整段漏掉 (写本判据时真漏过一次)。
        for n in [node] + list(no_nested(node)):
            if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    and n.id in own and n.id not in bound
                    and n.id not in params and n.id in outer):
                bad.append((n.lineno, n.id))

    def flow(stmts, bound):
        b = set(bound)
        for st in stmts:
            b = one(st, b)
        return b

    def one(st, b):
        if isinstance(st, ast.Assign):
            check(st.value, b)
            for t in st.targets:
                bind(t, b)
        elif isinstance(st, ast.AugAssign):
            check(st.target, b); bind(st.target, b); check(st.value, b)
        elif isinstance(st, ast.AnnAssign):
            if st.value:
                check(st.value, b)
            bind(st.target, b)
        elif isinstance(st, ast.If):
            check(st.test, b)
            a = flow(st.body, b)
            c = flow(st.orelse, b) if st.orelse else b
            b = (a & c) if st.orelse else b      # 只有两支都赋才算赋
        elif isinstance(st, (ast.For, ast.AsyncFor)):
            check(st.iter, b)
            inner = set(b); bind(st.target, inner)
            flow(st.body, inner); flow(st.orelse, inner)
        elif isinstance(st, ast.While):
            check(st.test, b); flow(st.body, set(b)); flow(st.orelse, set(b))
        elif isinstance(st, ast.Try):
            flow(st.body, set(b))                # try 体可能中途就跳走了
            for h in st.handlers:
                hb = set(b)
                if h.name:
                    hb.add(h.name)
                if h.type:
                    check(h.type, b)
                flow(h.body, hb)
            flow(st.orelse, set(b)); flow(st.finalbody, set(b))
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            for it in st.items:
                check(it.context_expr, b)
                if it.optional_vars:
                    bind(it.optional_vars, b)
            b = flow(st.body, b)
        elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            b.add(st.name)
            check(ast.Module(body=st.body, type_ignores=[]), b).__class__  # 闭包读在 def 处判一次
        elif isinstance(st, ast.ClassDef):
            b.add(st.name)
        elif isinstance(st, (ast.Import, ast.ImportFrom)):
            for a in st.names:
                b.add((a.asname or a.name).split('.')[0])
        elif isinstance(st, ast.Delete):
            for t in st.targets:
                if isinstance(t, ast.Name):
                    b.discard(t.id)
        else:
            check(st, b)
        return b

    flow(rr.body, set(params))
    seen, out = set(), []
    for line, name in bad:
        if name not in seen:
            seen.add(name); out.append((line, name))
    return out

def _problems(src):
    """把五条不变量检查成一份问题清单 (空 = 全过)。"""
    bad = []
    fields = _run_state_fields(src)
    restored = _restored_names(src)
    if set(restored) != set(RESTORE_TO_FIELD):
        bad.append("恢复清单与 RESTORE_TO_FIELD 不一致: %s"
                   % sorted(set(restored) ^ set(RESTORE_TO_FIELD)))
    for item, field in RESTORE_TO_FIELD.items():
        if field is not None and field not in fields:
            bad.append("恢复项 %s 映射到的字段 %s 不在 RunState 里" % (item, field))
    kw = _constructor_keywords(src)
    if sorted(kw) != sorted(fields):
        bad.append("构造点与字段清单不一致: 缺 %s / 多 %s"
                   % (sorted(set(fields) - set(kw)), sorted(set(kw) - set(fields))))
    reads = _state_reads(src)
    for f in fields:
        if f not in reads:
            bad.append("字段 %s 没有任何 state. 读者 (新增字段必须同时新增读者)" % f)
    fin = _func(_tree(src), "_finalize_run")
    args = [a.arg for a in fin.args.posonlyargs + fin.args.args + fin.args.kwonlyargs]
    if args != ["self", "state"] or fin.args.vararg or fin.args.kwarg:
        bad.append("_finalize_run 的接口变了: %s" % args)
    left = _round_level_control(src)
    if left:
        bad.append("_run_round 里还有绑到自己的 break/continue (该用返回值): %s" % left)
    rbw = _unbound_local_reads(src)
    if rbw:
        bad.append("_run_round 里有会静默读到旧值的名字 (该进 state): %s" % rbw)
    return bad


class TestTheFieldListHasOneSource(unittest.TestCase):

    def setUp(self):
        self.src = CONDUCTOR.read_text(encoding="utf-8")

    def test_the_real_source_satisfies_every_invariant(self):
        bad = _problems(self.src)
        self.assertEqual([], bad, "RunState 契约被破坏: " + " | ".join(bad))

    def test_the_field_count_is_pinned(self):
        """数目写死: 少一个 (丢状态) 或多一个 (多副本) 都要有人来解释"""
        # search_objective retains the frozen metric and its session-local
        # candidate receipt; live state is deliberately NOT restored on resume.
        self.assertEqual(33, len(_run_state_fields(self.src)))

    def test_finalize_takes_exactly_one_state_object(self):
        fin = _func(_tree(self.src), "_finalize_run")
        self.assertEqual(["state"], [a.arg for a in fin.args.kwonlyargs])
        self.assertEqual(["self"], [a.arg for a in fin.args.args])


class TestTheRoundIsOneNamedMethod(unittest.TestCase):
    """Stage2 的契约: 轮次循环体只有一个出口机制 (返回值), 循环本身只剩驱动。"""

    def setUp(self):
        self.src = CONDUCTOR.read_text(encoding="utf-8")

    def test_the_driver_loop_only_dispatches(self):
        loop = _round_loop(_tree(self.src))
        self.assertEqual(1, len(loop.body), ast.unparse(loop))
        self.assertIsInstance(loop.body[0], ast.If)

    def test_run_round_takes_state_and_round_num_plus_readonly_context(self):
        rr = _func(_tree(self.src), "_run_round")
        self.assertEqual(["self", "state", "round_num"],
                         [a.arg for a in rr.args.posonlyargs + rr.args.args])
        self.assertEqual(["expected_output", "hw_info", "_mem_caps",
                          "max_stagnant_rounds", "verify_drift"],
                         [a.arg for a in rr.args.kwonlyargs])

    def test_run_round_has_no_break_or_continue(self):
        self.assertEqual([], _round_level_control(self.src))

    def test_the_driver_stops_exactly_when_the_round_says_stop(self):
        """驱动条件取反是**静默**的: 单轮跑 (max_optimize_rounds=1) 时那一轮返回
        _ROUND_CONTINUE, 取反之后照样 break -> 观察不出差别。实测: 取反后两条快判据
        16 条全绿。所以要有一条判据直接读这个条件 (绝对锚点, doc 49 §7.7)。"""
        cond = ast.unparse(_round_loop(_tree(self.src)).body[0].test)
        self.assertIn("is _ROUND_STOP", cond)
        self.assertNotIn("is not", cond)

    def test_the_round_is_three_named_phases(self):
        """Stage3 之后一轮读起来是三步: 轮首门 -> 主流水线 -> 收尾。

        这条钉的是**结构本身** (绝对锚点, doc 49 §7.7)。把这三段再内联回 _run_round 是
        「拆分白做」, 而纯搬运的回退不会有任何行为判据变红 —— 实测: 搬运前后 2571 条
        全绿、行为摘要也相同。所以形状只能写死在这里。

        另外钉住「_run_round 里没有嵌套函数」: _snap 原来是闭包, 闭包的读会被判据漏掉
        (它读的 3 个名字恰好都在外层), 现在它是方法, 参数就是它的依赖清单。
        """
        bad = _phase_problems(self.src)
        self.assertEqual([], bad, "阶段结构被破坏: " + " | ".join(bad))


class TestTheGuardIsNotANoOp(unittest.TestCase):
    """反向标定: 在内存里造出每一种漂移, 判据必须变红 —— 没红就说明这条判据是摆设。"""

    def setUp(self):
        self.src = CONDUCTOR.read_text(encoding="utf-8")

    def _drift(self, old, new):
        drifted = self.src.replace(old, new, 1)
        self.assertNotEqual(drifted, self.src, "标定没接上: 锚点没匹配")
        return drifted

    def test_catches_a_field_with_no_reader(self):
        drifted = self._drift(
            "    round_signatures: list" + NL,
            "    round_signatures: list" + NL + "    zzz_nobody_reads_me: int" + NL)
        self.assertTrue(any("zzz_nobody_reads_me" in b for b in _problems(drifted)))

    def test_catches_the_round_phases_being_inlined_back(self):
        """把轮首门的委派换掉 -> 「一轮是三步」这条判据必须变红

        这是纯搬运式回退: 行为一个字不变 (内联回去还是那两道门), 所以只有形状判据能拦。
        """
        drifted = self._drift(
            "        if self._round_enter_or_stop(state, round_num) is _ROUND_STOP:" + NL
            + "            return _ROUND_STOP" + NL,
            '        if self._budget.oracle_may("explore", 1):' + NL + '            pass' + NL)
        self.assertTrue(any("轮首门" in b for b in _phase_problems(drifted)))

    def test_catches_a_phase_growing_an_interface(self):
        """收尾阶段多吃一个形参 -> 依赖清单变了"""
        drifted = self._drift(
            "                         _esc_decision, _measure_s, _space, bandit_arm, l3_result) -> bool:" + NL,
            "                         _esc_decision, _measure_s, _space, bandit_arm, l3_result, zzz) -> bool:" + NL)
        self.assertTrue(any("_round_close_out 的接口变了" in b for b in _phase_problems(drifted)))

    def test_catches_a_nested_function_growing_in_run_round(self):
        """_run_round 里又长出嵌套函数 -> 闭包的依赖会被判据漏掉"""
        head = ("    def _run_round(self, state, round_num, *, expected_output, hw_info, _mem_caps,"
                + NL + "                   max_stagnant_rounds, verify_drift) -> bool:" + NL)
        drifted = self.src.replace(
            head, head + "        def _zzz() -> None:" + NL + "            return None" + NL, 1)
        self.assertNotEqual(drifted, self.src, "标定没接上: 锚点没匹配")
        self.assertTrue(any("嵌套函数" in b for b in _phase_problems(drifted)))

    def test_catches_a_constructor_that_forgot_a_field(self):
        drifted = self._drift("            artifact=artifact," + NL, "")
        self.assertTrue(any("构造点与字段清单不一致" in b for b in _problems(drifted)))

    def test_catches_a_new_item_in_the_restore_tuple(self):
        drifted = self._drift("            vr_list," + NL, "            vr_list, zzz_new," + NL)
        self.assertTrue(any("恢复清单" in b for b in _problems(drifted)))

    def test_catches_finalize_growing_a_parameter_again(self):
        drifted = self._drift("    def _finalize_run(self, *, state):" + NL,
                              "    def _finalize_run(self, *, state, extra=1):" + NL)
        self.assertTrue(any("接口变了" in b for b in _problems(drifted)))

    def test_catches_a_break_left_in_run_round(self):
        """把 _run_round 里那处委派出口改回 break -> 出口机制又变成藏在循环体里的跳转

        锚点必须落在 _run_round **里面**: 现在全仓有 14 处 `return _ROUND_STOP`,
        第一处住在 _round_enter_or_stop 里 —— 用「第一个 12 空格缩进的」去接会接到别人
        身上, 标定就会「没红」。那是量具坏了, 不是判据松了。
        """
        anchor = ("        if self._round_enter_or_stop(state, round_num) is _ROUND_STOP:" + NL
                  + "            return _ROUND_STOP" + NL)
        self.assertIn(anchor, self.src, "标定没接上: 锚点不在")
        drifted = self.src.replace(anchor, anchor.replace("return _ROUND_STOP", "break"), 1)
        self.assertNotEqual(drifted, self.src, "标定没接上: 锚点没匹配")
        self.assertTrue(_round_level_control(drifted))

    def test_catches_a_conditional_assignment_read_on_the_other_branch(self):
        """正是 last_tunable_perf 的形状: 赋在 if 这一支, 读在 else/elif 那一支

        读的行号在写的后面, 所以「先读后写」那种按行号比的判据抓不到它。
        """
        head = ("    def _run_round(self, state, round_num, *, expected_output, hw_info, _mem_caps,"
                + NL + "                   max_stagnant_rounds, verify_drift) -> bool:" + NL)
        drifted = self._drift(head, head
                              + "        if _zzz_gate:" + NL
                              + "            plan = None" + NL
                              + "        _zzz_probe = plan" + NL)
        self.assertIn("plan", [n for _, n in _unbound_local_reads(drifted)])

    def test_the_pair_of_conditions_is_both_needed(self):
        """只有「循环之前有值」的名字才算: 否则搬运之前它也会崩, 与本次改动无关"""
        outer = _outerly_bound(self.src)
        self.assertIn("plan", outer)
        self.assertNotIn("verify_results", outer)

if __name__ == "__main__":
    unittest.main(verbosity=2)
