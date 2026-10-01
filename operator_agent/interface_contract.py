"""核函数接口契约的静态检查 (架构缺口 6)

调用方与生成核之间有一条从未写下来的约定: 远端固定以

    module.kernel(**inputs)

**全关键字**方式调用 (`remote_scripts/run_kernel.py:_execute_triton`)。生成代码只要少
一个参数、多一个参数、或把参数声明成 positional-only，运行时就是一句
`TypeError: kernel() got an unexpected keyword argument 'causal'` —— 而这句话要等到
一次完整的 SSH 往返 + 远程 import + Triton JIT 之后才会出现，于是一个纯静态可见的
错误要花掉一整轮 Fixer (真实历史: attention 任务连烧 3 轮, 见 docs/research/00 缺口 6)。

为什么用 AST 而不是 `inspect.signature`
---------------------------------------
两条都是实测/一手事实:
1. **不能在控制端 import 生成代码**。`@triton.jit` 要求代码真的落在文件里
   (triton 3.8.0 在 `python -c` 下直接 `ValueError: @jit functions should be defined
   in a Python file`)，而控制端也没有 torch。项目本来就把「引用实现只在远端构建」写成
   了规则 (`verifier._reference_torch` 的 docstring)。
2. **`inspect.signature` 对 Triton 的 JITFunction 拿不到真签名**。真机实测 (triton
   3.8.0) 它返回 `(*args: 'P.args', **kwargs: 'P.kwargs')` —— 于是「有 **kwargs 就放行」
   的规则会**永远放行**，一条看起来正确、实际永不触发的死守卫。真参数表在 `.fn` 与
   `.params` 上。AST 路线没有这个问题: 装饰器不影响 `def` 的参数列表。

保守原则
--------
只在「运行时必然 TypeError」时判失败；任何看不准的形状 (`kernel = something`、条件定义、
函数内定义) 一律放行，让 L2 去撞。误放行的代价是一轮 L2，误拦截的代价是把一份能跑的核
打回去重画 —— 后者更糟。
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from typing import Optional

from .source_lang import LangVerdict, classify


def _find_kernel_def(tree: ast.Module) -> Optional[ast.FunctionDef]:
    """只在模块顶层找 `def kernel(...)`; 看不准就返回 None (交调用方放行)"""
    found = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "kernel":
            found = node                      # 顶层重复定义时后一个生效, 与 Python 一致
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "kernel":
                    return None               # `kernel = ...` 别名: 静态看不出来
        if isinstance(node, (ast.If, ast.Try)):
            # 条件/异常分支里的定义意味着"可能没有", 也可能有别的形态 —— 不猜
            for sub in ast.walk(node):
                if isinstance(sub, ast.FunctionDef) and sub.name == "kernel":
                    return None
    return found


def _arg_names(a: ast.arguments) -> dict:
    """{'posonly': [...], 'poskw': [...], 'kwonly': [...], 'vararg': bool,
        'kwarg': bool, 'no_default': set(名字)}"""
    kw = a.kwonlyargs
    pos_only = list(a.posonlyargs)
    regular = list(a.args)
    # defaults 只覆盖"尾部"参数: posonly+regular 合并后的最后 len(defaults) 个
    positional = pos_only + regular
    # defaults 覆盖的是**尾部** len(defaults) 个参数, 所以必填的是前 n_no_default 个。
    # 写成 positional[n_no_default:] 会把「必填」算成空集 —— 两个用例就是这么抓出它的。
    n_no_default = len(positional) - len(a.defaults)
    # ⚠️ 这里必须取 .arg 名字: 直接放 ast.arg 对象的话, 后面 `名字 in no_default` 恒为
    # False, 于是「必填参数没传」「positional-only 挡道」两条判断一起变成永不触发。
    no_default = {x.arg for x in positional[:max(0, n_no_default)]}
    for arg, d in zip(kw, a.kw_defaults):
        if d is None:
            no_default.add(arg.arg)
    return {
        "posonly": [x.arg for x in pos_only],
        "poskw": [x.arg for x in regular],
        "kwonly": [x.arg for x in kw],
        "vararg": a.vararg is not None,
        "kwarg": a.kwarg is not None,
        "no_default": no_default,
    }


def input_key_sets(test_inputs) -> list:
    """把调用方可能传的输入形态摊成「若干组关键字名」

    `Verifier.verify(test_inputs)` 同时接受单个 dict 与**多组输入套件** (list[dict],
    见 L2 的 num_input_sets)。契约要对每一组单独成立 —— 某一组多带一个键, 那一组就会
    在运行时抛 TypeError, 所以不能只看第一组, 也不能取并集后只判一次。
    形态不认识就返回空列表 (= 不检查), 与整模块的保守原则一致。
    """
    if isinstance(test_inputs, dict):
        return [list(test_inputs.keys())]
    if isinstance(test_inputs, (list, tuple)) and test_inputs:
        sets = [list(x.keys()) for x in test_inputs if isinstance(x, dict)]
        return sets
    return []


_CONTRACT_HEADER = "## 调用契约 (由 harness 广播, 不是你的推测)"

_UNKNOWN = (
    "调用方固定以 `kernel(**输入)` 的**全关键字**方式调用你的入口函数。\n"
    "本轮 harness **没有**拿到输入名 (任务没带 test_inputs) ⇒ 请以参考实现/任务描述里出现的"
    "输入名逐一作为参数名, 不要新增, 也不要省略。"
)


def broadcast_calling_convention(test_inputs) -> str:
    """生成一段可以直接进 prompt 的调用契约广播。永远返回非空文字。

    为什么由 harness 来说而不是让模型猜 (CUDA-Harness arXiv:2609.00058v1 §4.1.2 一手):
    那边的 scaffold 把 calling convention 在**编译期**打出来, 目的是
    "the agent remains unaware of the details of the test inputs when implementing the
    kernel and cannot tailor an overfitted implementation to them" —— 契约是**测出来的事实**,
    不是 prompt 里的一句话。我方的 `planner.py:54,84` 恰好相反: 它按 `kernel_type` 猜一个
    `def kernel(A, B)` 写进计划 JSON, 而那份 JSON 会整体注入 coder 的 prompt ⇒
    layernorm 这类输入名不是 A/B 的任务, prompt 里就同时存在一条**错的**契约和一个
    只能等 TypeError 才出现的纠正。这段文字取代那个猜测的位置, 并注明以本节为准。
    """
    sets = [sorted({str(k) for k in s}) for s in (input_key_sets(test_inputs) or [])]
    # 空组不算契约: `{}` 摊出来是 `[[]]`, 直接走「单组」分支会印出一句
    # 「会传入的输入名: (什么都没有)」—— 那是把「不知道」说成一个假的空集。
    distinct = {tuple(s) for s in sets if s}
    if not distinct:
        return _CONTRACT_HEADER + "\n\n" + _UNKNOWN

    if len(distinct) == 1:
        names = ", ".join(f"`{n}`" for n in sorted(distinct.pop()))
        tail = f"  (共 {len(sets)} 组输入, 名字完全相同)" if len(sets) > 1 else ""
        body = f"本次任务实际会传入的输入名: {names}{tail}"
    else:
        listing = "\n".join(f"  - 第 {i} 组会传入的输入名: "
                            + ", ".join(f"`{n}`" for n in s)
                            for i, s in enumerate(sets, 1))
        body = ("本次任务有多组输入, 各组的输入名**不同**:\n" + listing +
                "\n⇒ 你的入口必须接得住上面每一组。")

    return (_CONTRACT_HEADER + "\n\n"
            "调用方固定以 `kernel(**输入)` 的**全关键字**方式调用你的入口函数。\n"
            + body + "\n"
            "`def kernel(...)` 的参数名必须与上面**逐一相等** —— 多一个或少一个都会在运行时"
            "抛 `TypeError`。\n"
            "计划 JSON 里的 `kernel_signature` 只是 Planner 的建议, **以本节为准**。")


@dataclass(frozen=True)
class InterfaceVerdict:
    """接口契约的判定 = 问题描述 + **适用性**（CHECKLOG I65）

    三个值原来压在一个 `Optional[str]` 里: `None` 既是「签名对得上」也是「看不出来」。
    现在 `problem` 只表示真的对不上; 「判不了」由 `lang.applicable` 与 `reason` 说。
    """

    problem: Optional[str]
    lang: "LangVerdict"

    @property
    def applicable(self):
        return self.lang.applicable

    def as_metrics(self) -> dict:
        return {"static_guard_lang": self.lang.as_dict(),
                "static_guard_applicable": self.lang.applicable,
                "static_guard_reason": self.lang.reason}


def check_kernel_interface(source: str, input_keys,
                           language: str = "") -> InterfaceVerdict:
    """签名对不上就给出处方式说明; 对得上放 `problem=None`; **判不了**会说判不了

    input_keys: 调用方这次会作为关键字传进去的名字 (即 L2 用的那份输入的键)。
    `language`: 这份源码声明的语言/DSL (空 = 没声明 ⇒ 只做一条推断, 见 `source_lang`)。
    """
    lang = classify(source, language)
    if lang.applicable is False:
        return InterfaceVerdict(problem=None, lang=lang)
    if lang.coverage != "ast":
        # 有覆盖 (文本模式) 但**这道**守卫要的是 `def kernel` (Python AST) ⇒ 对这份源码
        # 明确不适用。与「签名对得上」必须分得开 (CHECKLOG I65)。
        return InterfaceVerdict(problem=None, lang=replace(
            lang, applicable=False,
            reason=f"接口契约要 `def kernel` (Python AST), 而 "
                   f"{lang.language or '未声明的语言'} 的覆盖强度是 {lang.coverage} ⇒ "
                   f"这道守卫不适用 (不是「签名对得上」)"))
    keys = {str(k) for k in (input_keys or ())}
    if not keys:
        return InterfaceVerdict(problem=None, lang=lang)   # 没有输入信息 = 没有可判的对象
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # 语法错由 L1 的预检负责, 这里不抢话 —— 但「解析不了」这件事要说出来:
        # 语言声明是 Python 却解析不了 ⇒ 适用性从「适用」降成「无法确立」。
        return InterfaceVerdict(problem=None, lang=replace(
            lang, applicable=None, family="unknown", coverage="none",
            reason="声明/推断为 Python, 但源码解析不了 (语法错由 L1 预检负责) ⇒ "
                   "接口契约这一道判不了"))
    fn = _find_kernel_def(tree)
    if fn is None:
        return InterfaceVerdict(problem=None, lang=replace(
            lang, applicable=None, family="unknown", coverage="none",
            reason="找不到 `def kernel` 入口 ⇒ 这道契约检查对本轮候选判不了 "
                   "(不是「签名对得上」)"))
    a = _arg_names(fn.args)
    named = set(a["posonly"]) | set(a["poskw"]) | set(a["kwonly"])

    problems = []
    if a["kwarg"]:
        return InterfaceVerdict(problem=None, lang=lang)   # **kwargs: 传什么都接得住
    # positional-only 的参数**无法**用关键字传 → 必填的那些必然撞上
    blocked = [n for n in a["posonly"] if n in a["no_default"]]
    if blocked:
        problems.append(
            f"positional-only 参数无法用关键字传入: {', '.join(sorted(blocked))}")
    # 只有**必填**参数 (无默认值) 缺了才会抛 TypeError; 调用方不传一个有默认值的参数
    # 是完全合法的, 把它算进问题就等于把能跑的核打回去重画。
    required_missing = sorted(n for n in (set(a["poskw"]) | set(a["kwonly"]))
                              if n in a["no_default"] and n not in keys)
    extra = sorted(keys - named)
    if required_missing:
        problems.append(f"kernel() 必填但调用方不会传的参数: {', '.join(required_missing)}")
    if extra:
        problems.append(f"调用方会传、但 kernel() 没收的参数: {', '.join(extra)}")
    if not problems:
        return InterfaceVerdict(problem=None, lang=lang)

    declared = ", ".join(a["posonly"] + a["poskw"] + a["kwonly"]) or "(无参数)"
    return InterfaceVerdict(problem=(
        "根因: 调用方固定以 `kernel(**输入)` 的**全关键字**方式调用核函数, 而这份代码的 "
        "kernel() 签名与之对不上 —— 真跑起来必然抛 "
        "`TypeError: kernel() got an unexpected keyword argument ...`。\n"
        f"  本次会传入的输入名: {', '.join(sorted(keys))}\n"
        f"  kernel() 声明的参数: ({declared})\n"
        + "".join(f"  不匹配: {p}\n" for p in problems)
        + "改法: 让 `def kernel(...)` 的参数名**逐一等于**上面「会传入的输入名」; "
          "多余的那个若确实用不到, 也照样写进签名并在函数体里忽略 "
          "(或整体写成 `def kernel(*args, **kwargs)` 再自己取)。"
          "注意不要改输入名 —— 那是任务侧给的。\n"
          f"位置: 第 {fn.lineno} 行的 `def kernel`。"
    ), lang=lang)
