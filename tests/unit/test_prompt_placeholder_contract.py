"""模板占位符 ↔ render 传参 的两侧契约 (#67)。

`PromptManager.render()` 两个方向都是**静默**的 (读它的 `replacer` 就看得见)：

* 模板里有 `{{x}}` 而调用方没传 `x` ⇒ 那一处**原样留在发给模型的文本里**
  (`if key not in kwargs: return match.group(0)`) —— 模型看到的不是「缺这块信息」，
  而是一串 `{{x}}` 字面量。
* 调用方算了并传了 `x` 而模板里没有 `{{x}}` ⇒ 值被丢掉，**没有任何一处会报错**；
  这就是本仓反复出现的「算了没人读」，只不过这次发生在 prompt 装配环节。

所以这里两条守卫一起放，而且**第二条走生产代码**（真的调一次 `render`，把所有传参喂成
哨兵值，断言输出里没有残留的 `{{`）—— 集合算术的差集只在「两侧的名字写法一致」时才等价于
「渲染出来的东西是对的」，而写法本身正是会漂的那个东西。

运行: python3 -B -m unittest tests.test_prompt_placeholder_contract -v
"""

import ast
import io
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from operator_agent.generate.prompt_manager import PromptManager  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(REPO_ROOT, "operator_agent")
PROMPTS = os.path.join(PKG, "prompts")

_PLACE = re.compile(r"\{\{(.+?)\}\}")
#: render() 自己消费掉的、不该出现在模板里的两个控制参数
CONTROL = {"_agent_role", "_skill_name"}


def _template_placeholders(text: str) -> set:
    """模板要的变量名。`{{skill}}` / `{{skill:3}}` 由 PromptManager 自己填，不算传参。"""
    out = set()
    for m in _PLACE.finditer(text):
        key = m.group(1).strip()
        if key == "skill" or key.startswith("skill:"):
            continue
        out.add(key)
    return out


def _string_values_assigned_to(fn_node, name: str) -> set:
    """在所属函数里找 `name = "某个模板名"` 的字面量集合。

    `agents.py` 的那处按 DSL 分派模板 (triton / cuda_c / 通用)，第一个实参是个变量。
    不处理这一形态的话，枚举就只覆盖常量名那几处 ⇒ 守卫的解析面窄于被测代码
    (doc 49 §八 那条陷阱的第 ⑤ 次出现)。
    """
    found = set()
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name \
                        and isinstance(node.value, ast.Constant) \
                        and isinstance(node.value.value, str):
                    found.add(node.value.value)
    return found


def _render_sites():
    """枚举 `operator_agent/**/*.py` 里所有 `pm.render(<模板>, **kwargs)` 调用点。

    返回 `[(文件, 行号, 模板名, {传出去的变量名})]` —— 动态模板名会展开成多条。
    2026-10-31: 包分五层后必须递归枚举；平铺 `listdir` 只会看到 `__init__.py`，
    于是这份守卫会「绿在什么都没看上」。
    """
    sites = []
    for dirpath, _dirs, names in os.walk(PKG):
        for fname in sorted(names):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            rel = os.path.relpath(path, PKG).replace(os.sep, "/")
            tree = ast.parse(io.open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "render" and node.args):
                    continue
                kwargs = {kw.arg for kw in node.keywords if kw.arg} - CONTROL
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names_ = {first.value}
                elif isinstance(first, ast.Name):
                    owner = next((n for n in ast.walk(tree)
                                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                  and any(a is node for a in ast.walk(n))), None)
                    names_ = _string_values_assigned_to(owner, first.id) if owner else set()
                    if not names_:
                        raise AssertionError(
                            f"{path}:{node.lineno} render 的模板名是变量 {first.id} 但找不到"
                            f"它的字面量赋值 —— 枚举会漏掉这一处, 守卫必须报错而不是静默少覆盖")
                else:
                    raise AssertionError(f"{path}:{node.lineno} render 的模板名形态不认识: "
                                         f"{type(first).__name__}")
                for tpl in sorted(names_):
                    sites.append((f"operator_agent/{rel}", node.lineno, tpl, kwargs))
    return sites


_SITES = _render_sites()


class TestTheEnumerationSeesEverything(unittest.TestCase):
    """守卫自己的前提: 它真的覆盖了全部模板与全部调用点。没有这一组, 下面两条
    有可能红在错的地方, 或者更坏 —— 绿在什么都没看上。"""

    def test_every_template_on_disk_is_rendered_somewhere(self):
        on_disk = {f[:-3] for f in os.listdir(PROMPTS) if f.endswith(".md")}
        covered = {tpl for _f, _l, tpl, _k in _SITES}
        self.assertEqual(set(), on_disk - covered,
                         f"这些模板没人渲染 = 死文件 (改它不会有任何后果): {on_disk - covered}")
        self.assertEqual(set(), covered - on_disk,
                         f"这些 render 指向不存在的模板: {covered - on_disk}")

    def test_enumeration_covers_the_dynamic_template_name(self):
        """`agents.py` 按 DSL 选模板那一处必须被展开成 3 条。
        漏了它, 这三个模板就在守卫的盲区里。"""
        by_tpl = {tpl for _f, _l, tpl, _k in _SITES}
        for tpl in ("coder_generate", "coder_generate_triton", "coder_generate_cuda_c"):
            self.assertIn(tpl, by_tpl, f"动态模板名没被解析出来: {tpl}")
        self.assertGreaterEqual(len(_SITES), 9, f"只枚举出 {len(_SITES)} 个调用点")

    def test_a_missing_value_really_does_reach_the_model(self):
        """把失效形态本身钉一遍: 模板要 `{{x}}` 而调用方没传时, 生产 render 会把
        `{{x}}` **原样留在输出里** —— 这条不是多余的, 它是「为什么这个契约值得守」的证据。"""
        pm = PromptManager(prompts_dir=PROMPTS)
        out = pm.render("search_optimize", _agent_role="search", _skill_name=None)
        self.assertIn("{{arithmetic_intensity}}", out,
                      "预期渲染后还剩字面量占位符 —— render 的失效形态变了, "
                      "那这条用例和上面两条守卫都得重读")


class TestNoValueIsComputedThenDropped(unittest.TestCase):
    """传了而模板里没有 ⇒ 静默丢弃 (#67 的那两处就是这个形状)。

    ⚠️ 判据按**调用点**算，把该点可能选中的**全部**模板的占位符取并集当一个集合。
    `agents.py` 那处按 DSL 分派三个模板，`dsl` 只在通用模板里被渲染 —— 按「每个模板
    各自比一次」写的话，`dsl` 会被报成两处「没人读」，而那不是缺陷（走 triton 那条路
    时模板名本身就说明了 DSL）。并集是这里的**语义正确**，不是为了让守卫闭嘴：
    一个值只要在**某一条**会被选中的模板里被渲染，它就被读了。
    """

    def test_every_passed_name_appears_in_the_template(self):
        by_site: dict = {}
        for path, line, tpl, kwargs in _SITES:
            with io.open(os.path.join(PROMPTS, tpl + ".md"), encoding="utf-8") as fh:
                text = fh.read()
            slot = by_site.setdefault((path, line), [{"placeholders": set(), "passed": set()}])
            slot[0]["placeholders"] |= _template_placeholders(text)
            slot[0]["passed"] |= kwargs
        offenders = []
        for (path, line), (slot,) in sorted(by_site.items()):
            extra = sorted(slot["passed"] - slot["placeholders"])
            if extra:
                offenders.append((f"{path}:{line}", extra))
        self.assertEqual([], offenders,
                         f"这些值算了并传给了 render, 该调用点的所有模板都不渲染它: {offenders}")


class TestProductionRenderLeavesNoLiteralPlaceholder(unittest.TestCase):
    """走一次**生产** render: 所有传参喂哨兵值, 输出里不许残留 `{{`。

    这条与上面那条集合差集不是重复: 差集只保证「名字两侧都在」，这条保证
    「渲染出来的文本里没有没被填上的坑」—— 比如 `{{ x }}` 这种带空格的写法、
    或者模板里出现了枚举没认出来的形态。
    """

    SENTINEL = "<<哨兵值>>"

    def test_no_literal_placeholder_survives(self):
        pm = PromptManager(prompts_dir=PROMPTS)
        problems = []
        for path, line, tpl, kwargs in _SITES:
            text = pm.render(tpl, _agent_role="coder", _skill_name="triton_skill",
                             **{k: self.SENTINEL for k in kwargs})
            left = sorted({m.group(0) for m in _PLACE.finditer(text)})
            if left:
                problems.append((f"{path}:{line}", tpl, left))
        self.assertEqual([], problems,
                         f"这些占位符会以字面量形式出现在发给模型的 prompt 里: {problems}")

    def test_the_sentinel_actually_reaches_the_output(self):
        """反空转: 哨兵必须真的出现在渲染结果里。喂错了键名 (或模板被清空) 时,
        上面那条「没有残留 `{{`」会在一份**空 prompt** 上恒真地绿。"""
        pm = PromptManager(prompts_dir=PROMPTS)
        site = [s for s in _SITES if s[2] == "designer_sketch"]
        self.assertTrue(site, "枚举没找到 designer_sketch 这个调用点")
        kwargs = site[0][3]
        self.assertGreaterEqual(len(kwargs), 8, f"这处只传出 {sorted(kwargs)} 个值")
        text = pm.render("designer_sketch", _agent_role="designer",
                         _skill_name="triton_skill",
                         **{k: self.SENTINEL for k in kwargs})
        self.assertIn(self.SENTINEL, text, "哨兵没进输出 ⇒ 上面那条守卫在看一份空文本")
        self.assertIn("SKILL", text, "triton_skill 应该被注入 (没有 ⇒ 注入面也漂了)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
