"""Verifier 的结构守卫: 「Oracle 唯一收口点」必须是**具名方法**, 且接口是钉死的那几个.

守什么 (以及为什么必须是机械断言):
  * 代码里反复写着「唯一收口点」(spine-review P0-1 / P0-2, 设计 §13.4), 而它此前只是
    `_assess_measurement` 里一个 98 行的 `try` —— 「唯一」这件事只有注释在说。
  * 它的接口不是随意的: 一切读数从 `metrics` 取 (已经发布进产物的那一份),
    **不回头问调用方的局部量** —— 因为 `_assess_measurement` 看不到
    `_verify_performance` 的局部量, 用局部量在这里根本看不见 (会被
    `observation_error` 吞成一行日志)。所以「多一个形参」不是风格问题, 而是
    **有人在往回伸手**。
  * 调用点的实参值也钉住: `cv_exceeded=None` 这种「形参还在、值换了」的漂移,
    只查关键字名字是看不见的。

反向标定 (三条, 都在下面跑): 把那段 try **原样搬回**调用点 (真正的逆操作) / 给方法
加一个形参 / 把调用点某个实参换成别的值 —— 三种漂移都必须让这里变红。
"""
import ast
import unittest
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[2] / "operator_agent" / "decide" / "verifier.py"

_CLOSURE = "_publish_oracle_observation"
_CALLER = "_assess_measurement"
_PARAMS = ("self", "metrics", "assessment", "task", "cv_exceeded",
           "heartbeat", "heartbeat_expected", "required")
_KWARGS = ("cv_exceeded", "heartbeat", "heartbeat_expected", "required")
_POSITIONAL = ("metrics", "assessment", "task")
_CALL = '''        self._publish_oracle_observation(
            metrics, assessment, task, cv_exceeded=cv_exceeded,
            heartbeat=heartbeat, heartbeat_expected=heartbeat_expected,
            required=required)
'''


def _method(tree, name):
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "Verifier"), None)
    if cls is None:
        return None
    return next((x for x in cls.body
                 if isinstance(x, ast.FunctionDef) and x.name == name), None)


def _params(fn):
    a = fn.args
    return tuple([x.arg for x in list(a.posonlyargs) + list(a.args)]
                 + [x.arg for x in a.kwonlyargs])


def _problems(src):
    """返回这门纪律的问题清单 (空 = 通过)。不抛异常, 便于反向标定复用。"""
    tree = ast.parse(src)
    out = []
    fn = _method(tree, _CLOSURE)
    if fn is None:
        return ["没有 %s —— 「唯一收口点」又变回匿名的一段代码" % _CLOSURE]
    if _params(fn) != _PARAMS:
        out.append("接口变了: %s != %s" % (_params(fn), _PARAMS))
    body = [s for s in fn.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], ast.Try):
        out.append("方法体不再只有那一个 try —— 收口点被改写成别的东西")
    caller = _method(tree, _CALLER)
    if caller is None:
        return out + ["没有 %s" % _CALLER]
    if any(isinstance(s, ast.Try) for s in caller.body):
        out.append("%s 里又有 try —— 收口点被内联回去了" % _CALLER)
    calls = [n for n in ast.walk(caller)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == _CLOSURE]
    if len(calls) != 1:
        out.append("调用点有 %d 处 (应恰好 1 处)" % len(calls))
    else:
        c = calls[0]
        if tuple(k.arg for k in c.keywords) != _KWARGS:
            out.append("调用点关键字变了: %s != %s"
                       % (tuple(k.arg for k in c.keywords), _KWARGS))
        if tuple(ast.unparse(a) for a in c.args) != _POSITIONAL:
            out.append("调用点位置实参变了: %s != %s"
                       % (tuple(ast.unparse(a) for a in c.args), _POSITIONAL))
        for k in c.keywords:
            # 形参名字对、值换了 (例如 cv_exceeded=None) 也要拦住
            if ast.unparse(k.value) != k.arg:
                out.append("调用点 %s 的实参不是同名局部量: %s" % (k.arg, ast.unparse(k.value)))
    return out


class TestTheOracleClosurePoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = VERIFIER.read_text(encoding="utf-8")

    def test_the_closure_point_is_one_named_method_with_a_pinned_interface(self):
        self.assertEqual([], _problems(self.src),
                         "Oracle 唯一收口点的形状或接口漂了")

    def test_the_call_site_is_unique_in_the_whole_file(self):
        """「唯一收口点」的「唯一」要数得出来: 一个定义 + 一个调用点"""
        self.assertEqual(1, self.src.count("def " + _CLOSURE + "("))
        self.assertEqual(1, self.src.count("self." + _CLOSURE + "("))


class TestTheGuardIsNotANoOp(unittest.TestCase):
    """每条守卫都要有一个会变红的输入, 否则它可能是空跑"""

    @classmethod
    def setUpClass(cls):
        cls.src = VERIFIER.read_text(encoding="utf-8")

    def test_catches_the_closure_point_being_inlined_back(self):
        """真正的逆操作: 把方法体里那段 try 原样搬回调用点"""
        lines = self.src.splitlines(keepends=True)
        tri = next(s for s in _method(ast.parse(self.src), _CLOSURE).body
                   if isinstance(s, ast.Try))
        block = "".join(lines[tri.lineno - 1: tri.end_lineno])
        self.assertTrue(_CALL in self.src, "调用点文本变了, 这条标定要跟着改")
        mutated = self.src.replace(_CALL, block, 1)
        self.assertNotEqual([], _problems(mutated), "内联回去却没红")

    def test_catches_a_parameter_growing_on_the_closure_point(self):
        mutated = self.src.replace(
            "def _publish_oracle_observation(self, metrics, assessment, task, *,",
            "def _publish_oracle_observation(self, metrics, assessment, task, extra=1, *,", 1)
        self.assertIn("task, extra=1, *,", mutated, "标定锚点失效了")
        self.assertNotEqual([], _problems(mutated), "多一个形参却没红")

    def test_catches_a_call_site_value_being_swapped(self):
        mutated = self.src.replace(
            "metrics, assessment, task, cv_exceeded=cv_exceeded,",
            "metrics, assessment, task, cv_exceeded=None,", 1)
        self.assertIn("cv_exceeded=None,", mutated, "标定锚点失效了")
        self.assertNotEqual([], _problems(mutated), "实参换值却没红")


if __name__ == "__main__":
    unittest.main()
