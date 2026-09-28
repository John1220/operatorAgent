"""两个远端 runner 的**契约对照**守卫 (docs/research/49 §3.6 与 §五第 4 片)

守的是同一件事的三个方向, 形状抄 DSH `scripts/gen-doc-graphs.ts:297-298` 那种
「两个方向都抛」的构建失败:

  1. 控制端发了、runner 没读  → 「声明了没人用」。这一条能抓住 `baseline_priority`
     与 `precision` 被 run_kernel_xpu.py 丢掉 (doc 49 §3.6 标 ⚠️ 的那条)。
  2. 两个 runner 的 action 集合不一致 → 接下一个 GPU 平台时 fork 出 runner 再手工
     对齐 action 名, 对齐漏了也无人报警。
  3. 两个 runner 对同名 action 读的键集合不一致 → 同一个 action 名在两个平台上
     不是一回事。

三条都**从源码里解析**, 不抄期望值: 抄的那一刻是准的, 之后就开始漂 (doc 48 §6.4
第 4 条就是这个形状 —— 报表落盘时给血缘键做了白名单)。

解析对象是 runner 的分派现场, 两种写法都认:
  - `run_kernel.py` 的 `if action == ... / elif action in (...)` 链;
  - `run_kernel_xpu.py` 的 `dispatch = {...}` 字典 (靠 `dispatch.get(action)` 认出表名)。

⚠️ 本文件只读源码、不 import runner: 它们会拉起 torch/triton, 本机没有。

运行: PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest tests/test_runner_contract.py -q
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNERS = (
    ("run_kernel.py", REPO_ROOT / "remote_scripts" / "run_kernel.py"),
    ("run_kernel_xpu.py", REPO_ROOT / "remote_scripts" / "run_kernel_xpu.py"),
)
CONTROL_DIR = REPO_ROOT / "operator_agent"


# ─── 解析: 从 runner 源码里取出 (action -> 它读的 payload 键) ──────────────

def _str_of(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_action_read(node, payload_vars):
    """`<payload>.get("action" ...)`"""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and isinstance(node.func.value, ast.Name)
            and node.func.value.id in payload_vars
            and node.args and _str_of(node.args[0]) == "action")


def _payload_vars(tree):
    """payload 变量名 = 被 `.get("action")` 读过的名字 + 被整体赋成它的别名

    run_kernel_xpu.py 是 `payload = json.loads(raw)` 之后 `args = payload`, 分派字典里
    读的是 `args.get(...)`; 不跟这个别名的话, 那个 runner 会看起来一个键都不读。
    """
    names = {n.func.value.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
             and n.args and _str_of(n.args[0]) == "action"}
    while True:
        grown = {t.id for n in ast.walk(tree)
                 if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name)
                 and n.value.id in names
                 for t in n.targets if isinstance(t, ast.Name)} - names
        if not grown:
            return names
        names |= grown


def _action_vars(tree):
    """装 action 名的变量: `action = <payload>.get("action")` 的左值"""
    pvars = _payload_vars(tree)
    return {t.id for n in ast.walk(tree)
            if isinstance(n, ast.Assign) and _is_action_read(n.value, pvars)
            for t in n.targets if isinstance(t, ast.Name)}


def _table_vars(region, action_vars):
    """分派表变量名: `<表>.get(action)` / `<表>[action]` 里那个表

    靠这个认表, 而不是靠「值全是字符串键的字典字面量」: 后者会把 handler 里的
    `result = {"success": ..., "error": ...}` 也当成分派表, 于是 `success`/`error`
    会变成两个假 action。
    """
    out = set()
    for n in _walk_all(region):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
                and n.args and isinstance(n.args[0], ast.Name)
                and n.args[0].id in action_vars):
            out.add(n.func.value.id)
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and isinstance(n.slice, ast.Name) and n.slice.id in action_vars):
            out.add(n.value.id)
    return out


def _walk_all(stmts):
    for st in stmts:
        yield st
        yield from ast.walk(st)


def _is_dunder_main(test):
    """`if __name__ == "__main__":` 的测试式 (两个方向都认)"""
    if not isinstance(test, ast.Compare) or not any(isinstance(o, ast.Eq) for o in test.ops):
        return False
    sides = [getattr(test.left, "id", None)] + [_str_of(c) for c in test.comparators]
    return "__name__" in sides and "__main__" in sides


def _region(tree):
    """分派现场: 含 action 读取的最内层 Module / 函数体 / `if __name__` 块

    取最内层是为了不把整个模块的辅助函数也算进来 (它们的参数名可能与 payload 变量
    撞车), 同时又必须跨过 run_kernel_xpu.py 里 `if len(sys.argv) > 1:` 那一层 ——
    只认 Module / def / `if __name__` 三种作用域正好落在这两者之间。
    """
    pvars = _payload_vars(tree)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def depth(node):
        d = 0
        while node in parents:
            node, d = parents[node], d + 1
        return d

    scopes = [(tree, tree.body)]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append((node, node.body))
        elif isinstance(node, ast.If) and _is_dunder_main(node.test):
            scopes.append((node, node.body))
    hits = [(depth(scope), body) for scope, body in scopes
            if any(_is_action_read(n, pvars) for n in ast.walk(scope))]
    assert hits, "没在任何作用域里找到 `<payload>.get(\"action\")`, 解析前提不成立"
    return max(hits, key=lambda x: x[0])[1]


def _test_actions(test, action_vars):
    """`action == "x"` / `action in ("x", "y")` -> {"x"} / {"x","y"}; 不是动作测试则 None"""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.left, ast.Name) and test.left.id in action_vars):
        return None
    op, comp = test.ops[0], test.comparators[0]
    if isinstance(op, ast.Eq):
        name = _str_of(comp)
        return {name} if name else None
    if isinstance(op, ast.In) and isinstance(comp, (ast.Tuple, ast.List, ast.Set)):
        names = {_str_of(e) for e in comp.elts}
        return names if names and None not in names else None
    return None


def _split_chain(node, action_vars):
    """把 if/elif/.../else 拆成 [(动作集, 语句)] ; else 的动作集是 None。非纯动作分派返回 None"""
    out, cur = [], node
    while True:
        acts = _test_actions(cur.test, action_vars)
        if acts is None:
            return None
        out.append((acts, cur.body))
        if not cur.orelse:
            return out
        if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
            cur = cur.orelse[0]
            continue
        out.append((None, cur.orelse))
        return out


_SUBBODIES = {ast.If: ("body", "orelse"), ast.For: ("body", "orelse"),
              ast.While: ("body",), ast.With: ("body",),
              ast.Try: ("body", "orelse", "finalbody")}


def _read_keys(node, payload_vars):
    """节点子树里对 payload 的键读取: `<payload>.get("k")` 与 `<payload>["k"]`"""
    keys = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
                and n.func.value.id in payload_vars and n.args):
            k = _str_of(n.args[0])
            if k:
                keys.add(k)
        elif (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
              and n.value.id in payload_vars):
            # 3.8 及更早把下标包在 ast.Index 里, 3.9+ 直接是表达式
            k = _str_of(getattr(n.slice, "value", n.slice))
            if k:
                keys.add(k)
    return keys


def _harvest(node, serving, out, payload_vars):
    if not serving:
        return
    for k in _read_keys(node, payload_vars):
        for a in serving:
            out[a].add(k)


class _Contract:
    """一个 runner 的 (action 集合, action -> 消费的 payload 键集)"""

    def __init__(self, label, path):
        self.label = label
        self.path = path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        self.pvars = _payload_vars(tree)
        self.avars = _action_vars(tree)
        body = _region(tree)
        self.tables = _table_vars(body, self.avars)
        self.actions = self._find_actions(body)
        self.keys = {a: set() for a in self.actions}
        self._collect(body, set(self.actions))
        assert self.actions, f"{label}: 一个 action 都没解析出来, 守卫会空转"
        assert all(v for v in self.keys.values()), \
            f"{label}: 有 action 一个键都不读, 解析漏了: " \
            f"{sorted(a for a, v in self.keys.items() if not v)}"

    def _dispatch_dict(self, st):
        return (isinstance(st, ast.Assign) and isinstance(st.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id in self.tables
                        for t in st.targets))

    def _find_actions(self, body):
        acts = set()
        for st in _walk_all(body):
            if self._dispatch_dict(st):
                acts |= {k for k in (_str_of(x) for x in st.value.keys) if k}
            elif isinstance(st, ast.If):
                chain = _split_chain(st, self.avars)
                if chain:
                    acts |= {a for pair, _ in chain if pair for a in pair}
        return acts

    def _collect(self, body, serving):
        if not serving:
            return
        for st in body:
            if self._dispatch_dict(st):
                for k, v in zip(st.value.keys, st.value.values):
                    name = _str_of(k)
                    if name:
                        self._harvest(v, serving & {name})
                continue
            if isinstance(st, ast.If):
                chain = _split_chain(st, self.avars)
                if chain is not None:
                    matched = set()
                    for acts, stmts in chain:
                        if acts is None:
                            self._collect(stmts, serving - matched)
                        else:
                            self._collect(stmts, serving & acts)
                            matched |= acts
                    continue
            self._collect_generic(st, serving)

    def _collect_generic(self, st, serving):
        fields = _SUBBODIES.get(type(st))
        if fields is None:
            self._harvest(st, serving)
            return
        for attr in ("test", "iter"):
            head = getattr(st, attr, None)
            if head is not None:
                self._harvest(head, serving)
        for item in getattr(st, "items", None) or []:
            self._harvest(item, serving)
        for f in fields:
            self._collect(getattr(st, f, None) or [], serving)
        for h in getattr(st, "handlers", None) or []:
            self._collect(h.body, serving)

    def _harvest(self, node, serving):
        _harvest(node, serving, self.keys, self.pvars)


# ─── 解析: 控制端每个 action 实际发出的键 ──────────────────────────────────

def _control_side_payloads():
    """`operator_agent/` 里每个 `{"action": "<字面量>", ...}` -> (action, 键集, 出处)

    只认值为字符串字面量的: conductor 往 iteration_history 里写的 `"action": action`
    是血缘字段, 与远端协议无关。
    """
    out = []
    for py in sorted(CONTROL_DIR.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            action = None
            for k, v in zip(node.keys, node.values):
                if _str_of(k) == "action":
                    action = _str_of(v)
            if not action:
                continue
            keys = {k for k in (_str_of(x) for x in node.keys) if k}
            out.append((action, keys, f"{py.relative_to(REPO_ROOT)}:{node.lineno}"))
    return out


CONTRACTS = {label: _Contract(label, path) for label, path in RUNNERS}
SENT = _control_side_payloads()


# ─── 守卫 ────────────────────────────────────────────────────────────────

class TestRunnerContractParity(unittest.TestCase):

    def test_control_side_payloads_were_found(self):
        self.assertGreaterEqual(len(SENT), 6,
                                f"只认出 {len(SENT)} 条控制端 payload, 解析前提不成立")
        self.assertGreaterEqual(len(CONTRACTS), 2, "runner 少于两个, 无从对照")

    def test_both_runners_dispatch_the_same_action_set(self):
        a, b = (CONTRACTS[label] for label, _ in RUNNERS)
        self.assertEqual(
            a.actions, b.actions,
            f"action 集合不一致\n"
            f"  只有 {a.label} 派发: {sorted(a.actions - b.actions)}\n"
            f"  只有 {b.label} 派发: {sorted(b.actions - a.actions)}")

    def test_same_action_consumes_the_same_payload_keys(self):
        a, b = (CONTRACTS[label] for label, _ in RUNNERS)
        diffs = []
        for action in sorted(a.actions & b.actions):
            only_a, only_b = a.keys[action] - b.keys[action], b.keys[action] - a.keys[action]
            if only_a or only_b:
                halves = [f"{lbl} 独占 {sorted(ks)}"
                          for lbl, ks in ((a.label, only_a), (b.label, only_b)) if ks]
                diffs.append(f"  {action}: " + "; ".join(halves))
        self.assertEqual([], diffs,
                         "同名 action 在两个 runner 上读的 payload 键不一致 —— 控制端发出"
                         "而某一侧不读的键会被静默丢掉 (不报错):\n" + "\n".join(diffs))

    def test_every_sent_action_is_dispatched_by_every_runner(self):
        """用了没声明: 控制端发得出、runner 不认的 action 会在远端变成 Unknown action"""
        bad = [f"{action} ({where}) 不被 {c.label} 派发"
               for action, _, where in SENT for c in CONTRACTS.values()
               if action not in c.actions]
        self.assertEqual([], bad, "控制端在发没人派发的 action:\n  " + "\n".join(bad))

    def test_every_dispatched_action_is_sent_by_the_control_side(self):
        """声明了没人用: runner 里挂着一个控制端从不发的 action (抄 gen-doc-graphs.ts:563)"""
        sent_actions = {a for a, _, _ in SENT}
        bad = [f"{c.label} 派发了 {action}"
               for c in CONTRACTS.values() for action in sorted(c.actions - sent_actions)]
        self.assertEqual([], bad, "runner 里挂着没人发的 action (死分派, 或者控制端漏发):\n  "
                                  + "\n".join(bad))

    def test_every_sent_key_is_consumed_by_every_runner(self):
        """声明了没人用 (键这一层): 控制端发了、runner 从不读的键 = 静默降级

        这一条是 `baseline_priority` / `precision` 被 run_kernel_xpu.py 丢掉时唯一会响的
        守卫: 两侧都「成功」返回, 只是口径悄悄变了。
        """
        bad = []
        for action, keys, where in SENT:
            for c in CONTRACTS.values():
                if action not in c.actions:
                    continue          # 由 test_every_sent_action_is_dispatched 负责
                dropped = keys - c.keys[action]
                if dropped:
                    bad.append(f"{where} 发 {action} 的 {sorted(dropped)}, "
                               f"{c.label} 不读")
        self.assertEqual([], bad, "控制端发出的键有 runner 不消费:\n  " + "\n".join(bad))


class TestExtractorIsNotVacuous(unittest.TestCase):
    """守卫自己也要防「空转」: 两种分派写法都要真解析出东西

    没有这一条, 一个把 region 找错、于是两边都返回空集的解析器会让上面全部相等断言
    永真地绿着 —— 而那个「绿」正是全部安全感的来源。
    """

    DICT_FORM = '''
import sys, json

def main():
    args = {}
    action = None
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        payload = json.loads(sys.stdin.read())
        action = payload.get("action")
        args = payload
    dispatch = {
        "gpu_info": lambda: info(),
        "compile": lambda: comp(args.get("code", "")),
        "benchmark_with_baselines": lambda: bench(
            args.get("code", ""), args.get("kernel_type", ""),
            precision=args.get("precision", "")),
    }
    fn = dispatch.get(action)
    print(fn())
'''

    CHAIN_FORM = '''
import sys, json

if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())
    action = payload.get("action", "execute")
    if action == "gpu_info":
        result = info()
    elif action == "compile":
        result = comp(payload.get("code", ""))
    elif action in ("benchmark", "benchmark_with_baselines"):
        idle = sample()
        if action == "benchmark":
            result = bench(payload.get("code", ""), payload.get("warmup", 3))
        else:
            result = bench(payload.get("code", ""), payload.get("kernel_type", ""),
                           precision=payload.get("precision", ""))
        result["gpu_util_pct"] = idle
    else:
        result = {"success": False, "error": "Unknown action"}
    print(result)
'''

    def _parse(self, src, label):
        path = Path(self.tmp) / f"{label}.py"
        path.write_text(src, encoding="utf-8")
        return _Contract(label, path)

    def setUp(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = self._dir.name
        self.addCleanup(self._dir.cleanup)

    def test_dict_form_is_parsed_through_the_alias(self):
        c = self._parse(self.DICT_FORM, "dict_form")
        self.assertEqual({"gpu_info", "compile", "benchmark_with_baselines"}, c.actions)
        # `args = payload` 这个别名必须被跟到, 否则每个 action 都只剩 {action}
        self.assertEqual({"action"}, c.keys["gpu_info"])
        self.assertEqual({"action", "code"}, c.keys["compile"])
        self.assertEqual({"action", "code", "kernel_type", "precision"},
                         c.keys["benchmark_with_baselines"])

    def test_chain_form_attributes_nested_branches_to_one_action(self):
        c = self._parse(self.CHAIN_FORM, "chain_form")
        self.assertEqual({"gpu_info", "compile", "benchmark",
                          "benchmark_with_baselines"}, c.actions)
        # 外层 clause 服务两个 action, 内层 if/else 必须把键分给各自那一个
        self.assertEqual({"action", "code", "warmup"}, c.keys["benchmark"])
        self.assertEqual({"action", "code", "kernel_type", "precision"},
                         c.keys["benchmark_with_baselines"])
        # `result["gpu_util_pct"] = ...` 不是 payload 读取, 不许混进来
        self.assertNotIn("gpu_util_pct", c.keys["benchmark"])
        self.assertNotIn("gpu_util_pct", c.keys["benchmark_with_baselines"])

    def test_result_dict_in_a_handler_is_not_mistaken_for_a_dispatch_table(self):
        c = self._parse(self.CHAIN_FORM, "chain_form")
        for bogus in ("success", "error"):
            self.assertNotIn(bogus, c.actions,
                             "把 handler 里的返回字典当成了分派表")

    def test_a_dropped_key_shows_up_as_a_difference(self):
        """变异方向标定: 从字典那一侧删掉 precision, 两侧键集就必须不等"""
        full = self._parse(self.DICT_FORM, "full")
        mutated = self._parse(self.DICT_FORM.replace('precision=args.get("precision", "")',
                                                     'unused=1'), "mutated")
        self.assertIn("precision", full.keys["benchmark_with_baselines"])
        self.assertNotIn("precision", mutated.keys["benchmark_with_baselines"])
        self.assertNotEqual(full.keys["benchmark_with_baselines"],
                            mutated.keys["benchmark_with_baselines"])

    def test_real_runners_are_parsed_non_vacuously(self):
        for c in CONTRACTS.values():
            self.assertGreaterEqual(len(c.actions), 6, f"{c.label} 只认出 {c.actions}")
            self.assertIn("action", set().union(*c.keys.values()),
                          f"{c.label} 连 action 本身都没解析到")
            self.assertTrue(c.pvars, f"{c.label} 没认出 payload 变量")


if __name__ == "__main__":
    unittest.main(verbosity=2)
