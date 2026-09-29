"""修复取向按「哪一级报的失败 + 库里怎么归类」分支 —— CUDA-Harness §4.3.1 的两路
加 Fabrica 行 521-522 的第三路 interface repair (docs/research/55 §3.3, 本文件量具见下)

外部一手 (arXiv:2609.00058v1, 逐字核过):
    "The refinement branches based on which validation stage reports the failure.
     ... When the kernel fails functional validation, the agent is instructed to
     retreat to a more conservative implementation, prioritizing correctness
     above all else."
    "When optimization is built on a kernel that looks high-performance but is
     functionally flawed, subsequent refinement typically remains trapped in the
     same faulty pattern."

我方的先决事实 —— **这一段被自己纠正过两次, 两次都错在「只核机制、没核入口」**。
① 原本写「`conductor.py:780` 交进来的是 `[r for r in verify_results if not r.passed]`,
   所以 L2 数值失败会**顺带**和 L3『还不够快』同框」。读回 `verifier.verify()`
   (`operator_agent/verifier.py:233-249`) 是: 每一级失败就 `return results` ——
   **fail-fast**, 一份 `verify_results` 里最多只有一个未通过的级别 ⇒ 同轮 L2+L3 不可达。
② 我改口成「跨轮经 `history_cases` 并存, 因为 L3 会被 `conductor.py:948-951` 记进库」。
   这句也错: Fixer 那条支路被 `conductor.py:763` 的 `l1_l2_failed` 门挡着, 循环外那处
   兜底入库 `:952-955` 明写 `if any(level in (L1, L2))` ⇒ **今天的代码不产 L3 记录**。

结论: `_L2_RETREAT` 末句那句「若错误信息里同时出现性能未达标」今天是**明标的保险**,
不是现行路径的描述。两条通道各由本文件一条结构性守卫钉住
(`TestThePerformanceTextIsDeadProseForNow`), 任一松掉就得重读它。
方向没变 (两条通道都可能带性能文本), **通道变了** —— 这就是 doc 49 §八 第 26 条。

变异标定 (每条只改一处, 逐轮复原 + 复原后逐字节断言。2026-09-29 把**两张表合成一张**跑完:
21/21 拦住, 0 逃逸。脚本不进仓库, 所以每行的**锚点**就是括号里那句改动 —— 否则数字不可复现):
    M1 `stage_directive` 恒返回 `_L1_INPLACE`      → 7 条红
    M2 恒返回 `_L2_RETREAT`                        → 6 条红
    M3 分支条件 `"L2" in levels` 改成 `"L3"`        → 6 条红 (让性能失败决定取向正是原文
       说的那个陷阱; 注意红名单里**也**有活通道那条 —— 它同样依赖取向由 L2 触发)
    M4 模板删掉 `{{stage_directive}}`               → 9 条红 (含 #67 的装配契约: 值算了没人读)
    M5 模板把它换成死文字                           → 9 条红
    M6 模板恢复「在编译 (L1) 或功能 (L2) 验证中失败」 → 2 条红 (装配契约 + 那句被合写的旧话)
    M7 调用处把级别写死成 `{"L1"}`                  → 4 条红
    M8 删掉 `_L2_RETREAT` 的末句「本轮**不要**为此优化」
                                                  → 1 条红 (那条防御性的 L2+L3 优先级用例)
    M9a 上下文检索里滤掉 L3                        → 2 条红
    M9b 合并处滤掉 L3 (两条检索通道一起)             → 同样 2 条红
    M11 把 `conductor` 那道 `l1_l2_failed` 门放宽到 L3
                                                  → 1 条红 (钉那句保险可达性的那条门守卫)
    ── 以下 10 行是本片 (第三路 interface repair) 的整条链, 一层一条 ──
    M12 `failure_class` 返回**整条签名** (本片的原始 bug)
                                                  → 3 条红
    M13 调用处漏传第二个参数 ⇒ 第三路成死代码         → 1 条红
    M14 分流条件恒假                               → 2 条红
    M15 分流条件恒真 (一切失败都算接口)              → 12 条红
    M16 取向自己正则判文案、不走分类器 (两套口径)      → 1 条红 (那条 AST 结构守卫)
    M17 `failure_class_of` 取错段 (取层级而非类名)    → 3 条红
    M18 归类词表里那条 marker 删掉                   → 5 条红
    M19 门文案改一个字 (词表与文案脱钩)              → 7 条红
    M20 删掉第三路那句改动范围锁                     → 2 条红 (第一版**逃逸**, 见下)
    M21 只改桶名常量的值 (改名)                      → 1 条红 (第一版被我当作「应全绿的对照」)

⚠️ 这张表**第一版不是 10/10**: M9a 当时全绿。原因不在本文件, 而在
`FailureDatabase.search_by_error` 末尾的兜底 —— 零词元重叠/取不出词元时它
`return candidates[:limit]`, 把不相关的历史记录带进 prompt, 而 `fixer.py:423` 按
「在不在这份返回列表里」给它们打上 `[错误相似度命中]`。也就是说: 一条声称「按 Jaccard>0
匹配」的检索, 在没匹配到时**照样返回**, 于是 M9a 滤掉上下文通道后记录从「假命中」那条路
回来了。修掉兜底 (kernel_db.py, 用例在 `tests/test_failure_memory.py`) 之后 M9a 才归位。
**这条「逃逸」不是用例写松了, 是被测代码里一只真的「把没找到说成找到了」** —— doc 49 §八
第 26/27 条、CHECKLOG I16/I17。

第一轮跑这张表时**归属是错的**: 脚本只在最后统一复原文件, 于是改模板的 M4/M5/M6
一直留在盘上, 后面两条 fixer 变异的红其实是 M6 的红 (那句把两个占位符一起删掉,
装配契约必然红)。加了逐轮复原 + 复原后逐字节自检才是上面这份: M6 从 4 条红变 2 条,
M7/M8 不再挂着与它们无关的 `test_every_passed_name_appears_in_the_template`。
教训不是「全绿骗人」这一种, 而是**串味道的红会把账记到错的守卫头上** —— 照着它去
加固, 加固的是那条本来不需要加固的 (docs/research/55 §六, 陷阱族第 ㉒ 次)。

本片跑出的两条不在「分流有没有效」那一层, 都在量具自己身上:

- **M20 是这次唯一一条逃逸**。那条用例只朝一个方向查 (「别人的段落里没有我这句取向」), 于是
  把第三路**自己**那句范围锁删掉时, 所有缺席断言全体成立。⇒ 互斥/差集这类判据必须**在场 +
  缺席两个方向都查**; 只写单向时, 「内容被删」正好落在盲区里。这是「判据方向写反」那只的孪生
  (CHECKLOG I18)。
- **M21 我原本设计成「应当全绿的对照组」**, 理由是分流键已经收进 `kernel_db` 那一个常量, 改值
  不该有后果。它红了一条: `tests/test_failure_memory.py:113` 把整串 `"L1|interface_mismatch|-"`
  钉成期望值 —— 桶名是**已入库记录 id 的一部分** (`cx-{task_id}:{signature}`), 改名会让历史
  记录与新归类分家。⇒ 「收成单点」只覆盖**产它的那一半代码**, 名字一旦被写进别的事实就不再是
  自由重构。我这轮又一次**只数了产者、没数读者**, 与 §八 第 26 条同一个错 (CHECKLOG I19)。

⚠️ 上一版在 doc 41 里把这张表写成「13 行、13/13」, 其中 **M10a/M10b 两行的锚点在任何文件里
都查不到** (既不在本文件的表里, 也不在脚本留在文档里的描述里) ⇒ 那个数字不可复现。上面这份是
一次跑出来的, 每行的锚点写在括号里, 下次能照着重建脚本。

运行: python3 -m pytest tests/test_fixer_stage_branch.py -q
"""

import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXER_SRC = os.path.join(_ROOT, "operator_agent", "fixer.py")
with open(os.path.join(_ROOT, "operator_agent", "verifier.py"), encoding="utf-8") as _fh:
    _VERIFIER_SRC = _fh.read()


def _early_returns(src: str) -> list:
    """找出 `Verifier.verify()` 里「某级失败就 return」的早退点。

    只认这个形状: `if not <x>.passed: … return …`。它服务的断言是「同一次 verify() 的
    结果列表里最多只有一个未通过的级别」, 所以匹配器自己也要有反空转用例 (见
    `test_that_guard_is_not_a_no_op`)。
    """
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "verify"
               and any(a.arg == "kernel_code" for a in n.args.args)), None)
    if fn is None:
        raise AssertionError("找不到带 kernel_code 参数的 Verifier.verify —— 解析面对不上被测代码")
    return [n for n in ast.walk(fn)
            if isinstance(n, ast.If) and isinstance(n.test, ast.UnaryOp)
            and isinstance(n.test.operand, ast.Attribute)
            and n.test.operand.attr == "passed"
            and any(isinstance(s, ast.Return) for s in n.body)]

from operator_agent.fixer import (  # noqa: E402
    FixerAgent, stage_directive, failure_class,
    _L1_INPLACE, _L1_INTERFACE, _L2_RETREAT,
)
from operator_agent.kernel_db import (  # noqa: E402
    FAILURE_CLASS_INTERFACE, failure_class_of, failure_signature,
)
from operator_agent.schemas import (  # noqa: E402
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision,
    VerificationResult, VerificationLevel,
)

GOOD = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
BROKEN = "import numpy as np\n\ndef kernel(A, B:\n    return A @ B\n"


class _LLM:
    """记录每一次 chat 的 prompt; 每次都回一份能通过预检的代码。"""

    def __init__(self):
        self.prompts: list[str] = []

    def chat(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return f"```python\n{GOOD}```"

    def chat_json(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return {}


def _task():
    return TaskSchema(
        task_id="t-stage",
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, max_error=1e-5),
    )


def _res(level, msg):
    return VerificationResult(level=level, passed=False, message=msg, error_log=msg)


def _prompt_for(results):
    llm = _LLM()
    FixerAgent(llm=llm).run(BROKEN, _task(), results)
    return llm.prompts[0]


# 门自己产的那份文本长什么样, 不由本文件规定 —— 夹具必须**调用那道门**。
# 这条核函数与下面那份输入集是 doc 41 §六 里真实拦下的形状 (多一个必填的 BLOCK、
# 少一个调用方会传的 bias)。
_GATE_SRC = ("import triton\nimport triton.language as tl\n\n"
             "@triton.jit\ndef kernel(X, Y, BLOCK: tl.constexpr):\n    return\n")
_GATE_INPUTS = ["X", "Y", "bias"]   # 与 `input_key_sets` 给门的那份同形 (键名列表, 不是 dict)


def _interface_gate_text() -> str:
    from operator_agent.interface_contract import check_kernel_interface

    text = check_kernel_interface(_GATE_SRC, _GATE_INPUTS)
    # 夹具自己先过一道: 门哪天不再拦这个形状, 后面所有用例就都在测一份假的错误原文
    assert text is not None, "夹具那份代码不该被门放过 —— 否则下面的用例失去被测对象"
    assert "签名与之对不上" in text, "门的话改了 ⇒ 归类词表与第三路取向都哑了"
    return text


def _gate_result():
    """**逐字照 `verifier.py:285-292` 的形状**造一条接口契约失败 (message 只有一行,
    完整文本在 error_log; 取向读的是 error_log, 与库里归类同一个口径)。
    """
    text = _interface_gate_text()
    return VerificationResult(
        level=VerificationLevel.L1_COMPILE, passed=False,
        message="编译失败: " + text.split("\n")[0], error_log=text,
        metrics={"compile_status": "interface_mismatch", "backend": "fake"},
    )


class TestThirdRouteInterfaceRepair(unittest.TestCase):
    """第三路 (Fabrica 行 521-522 的 interface repair) —— 分流键是**归类**, 不是文案

    外部一手 (arXiv:2608.25124v1, 逐字核过):
        "The reviewer then routes architectural, implementation, and host/layout
         failures to redesign, code repair, or interface repair, respectively."

    这一路是这片里唯一**差点写了走不到**的地方: 第一版把分流键写成
    `signature == "interface_mismatch"`, 而 `failure_signature()` 返回的是三段
    `层级|类名|最内层帧` (真值 `"L1|interface_mismatch|-"`) —— 那条比较**永远为假**,
    取向照样非空、照样有绿用例 (它们只断言「取向永不为空」), 于是第三路是一段谁也
    到不了的死文字。`test_the_whole_signature_is_not_the_class` 就是把这只 bug 钉住的
    那条 (缺陷族: 按具体值列的门换环境恒绿 / 算了没人读)。

    基率 (doc 48 §六B, 16 轮真机): 4 轮死在这道门上, **且今天的 L1 失败全部来自它**
    ⇒ 这一路不是边角, 是 L1 的主路径。
    """

    def test_the_interface_gate_routes_to_the_third_directive(self):
        p = _prompt_for([_gate_result()])
        self.assertIn("入口契约门", p)
        self.assertIn("本轮只改 `def kernel` 那一行的参数名", p)   # 范围锁要真到了模型眼前
        # 三条取向互斥: 走第三路时既不许是「修一处 API 误用、保持策略」那条通用 L1,
        # 也不许是数值退让 (并存等于没分流)
        self.assertNotIn("保持既有的优化策略不变", p)
        self.assertNotIn("correctness 优先于性能", p)

    def test_the_route_is_reachable_through_run_not_only_the_helper(self):
        # 上面那条已经从 run() 进去 (I14: 只从被调方进的用例看不见「中间那层忘了转发」)。
        # 这条钉住另一头: 归类确实算进去了, 而不是调用处漏传第二个参数。
        self.assertEqual(_L1_INTERFACE, stage_directive({"L1"}, FAILURE_CLASS_INTERFACE))
        self.assertEqual(_L1_INPLACE, stage_directive({"L1"}))

    def test_the_whole_signature_is_not_the_class(self):
        """分流键取的是三段签名里那**一段** —— 拿整条比类名永远不等 (本片的原始 bug)"""
        res = _gate_result()
        levels = "+".join(sorted({r.level.value for r in [res]}))
        summary = " | ".join(FixerAgent._truncate_error(r.error_log or r.message, 300)
                             for r in [res])
        sig = failure_signature(levels, summary)
        self.assertEqual([levels, FAILURE_CLASS_INTERFACE, "-"], sig.split("|", 2))
        self.assertNotEqual(sig, FAILURE_CLASS_INTERFACE,
                            "签名与类名相等 ⇒ 三段形状已变, `failure_class_of` 的取法要重读")
        self.assertEqual(FAILURE_CLASS_INTERFACE, failure_class_of(levels, summary))

    def test_the_routing_key_and_the_corpus_bucket_are_the_same_number(self):
        """取向用的归类必须与 `conductor._record_failure` 入库那一条同一口径

        否则会出现「prompt 说是接口问题、库里记成 TypeError」的分裂: 复发门、检索和本轮
        取向读的本来是同一个桶。这条把两边的**输入构造**也一起比了 (排序级别 + 截断拼接)。
        """
        res = _gate_result()
        levels = "+".join(sorted({r.level.value for r in [res]}))
        summary = " | ".join(FixerAgent._truncate_error(r.error_log or r.message, 300)
                             for r in [res])
        self.assertEqual(failure_class_of(levels, summary), failure_class([res]))
        self.assertEqual(FAILURE_CLASS_INTERFACE, failure_class([res]))

    def test_the_list_the_directive_points_at_is_actually_in_the_prompt(self):
        """取向让模型「见错误原文里『本次会传入的输入名』」⇒ 那句必须在同一份 prompt 里

        指向一份不存在清单的取向就是伪造证据 (模型只能猜)。这条把取向与 `_format_failures`
        的呈现绑在一起, 谁改了展示口径 (比如只留 message 那一行) 就会红。
        """
        p = _prompt_for([_gate_result()])
        self.assertIn("本次会传入的输入名", p)
        self.assertIn("bias", p)          # 缺的那个参数名要真看得见, 不是让模型推

    def test_an_ordinary_compile_error_does_not_take_the_third_route(self):
        """反空转: 门不放行的失败必须仍走原来那条「就地修」"""
        p = _prompt_for([_res(VerificationLevel.L1_COMPILE,
                              "NameError: name 'tl' is not defined")])
        self.assertIn("保持既有的优化策略不变", p)
        self.assertNotIn("入口契约门", p)

    def test_a_numeric_failure_does_not_take_the_third_route(self):
        self.assertEqual(_L2_RETREAT,
                         stage_directive({"L2"}, failure_class(
                             [_res(VerificationLevel.L2_FUNCTION,
                                   "最大绝对误差 1.2e-1 超出容差")])))

    def test_the_marker_that_opens_the_route_has_exactly_one_producer(self):
        """归类词表里那个关键词只有一个产者 (那道门自己) —— 多一个或断了都该红

        词表与门文案是两个文件里的两份字面量; 门改措辞 ⇒ 第三路哑掉, 别人新写一处相同
        措辞 ⇒ 不相干的失败被领进这条路。全仓扫一遍把这两件事都变成红。
        """
        holders = []
        for name in sorted(os.listdir(os.path.join(_ROOT, "operator_agent"))):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(_ROOT, "operator_agent", name), encoding="utf-8") as fh:
                if "签名与之对不上" in fh.read():
                    holders.append(name)
        self.assertEqual(["interface_contract.py", "kernel_db.py"], holders,
                         "产这条错误原文的只有 interface_contract, 认它的只有归类词表")

    def test_three_directives_do_not_share_their_scope_sentences(self):
        """三路各自那句「本轮改什么」要**只在自己那段里** —— 而且双向都要查

        单向 (只查「别人的段落里没有这句」) 看不见内容被删: 标定时的 M20 把第三路那句
        范围锁删掉, 缺席断言于是全体成立, 整张表只有这一条逃逸。在场 + 缺席合起来才是
        「分流分干净」的读数条件 (缺陷族: 判据方向写反的孪生 —— 判据只朝一个方向)。
        """
        uniq = {
            "interface": ("入口契约门", "本轮只改 `def kernel` 那一行的参数名"),
            "numeric": ("数值检查器", "correctness 优先于性能"),
            "compile": ("编译器 / 预检", "保持既有的优化策略不变"),
        }
        texts = {"interface": _L1_INTERFACE, "numeric": _L2_RETREAT, "compile": _L1_INPLACE}
        for own, phrases in uniq.items():
            for phrase in phrases:
                self.assertIn(phrase, texts[own], f"{own} 那段丢了它自己的取向句")
                for other, body in texts.items():
                    if other == own:
                        continue
                    self.assertNotIn(phrase, body, f"{phrase} 同时出现在 {own} 与 {other}")

    def test_an_unknown_class_falls_back_to_the_level_route(self):
        # 不认识的归类 (真异常名 / unknown) 一律退到级别那条老路径, 取向不能为空
        self.assertEqual(_L1_INPLACE, stage_directive({"L1"}, "TypeError"))
        self.assertEqual(_L2_RETREAT, stage_directive({"L2"}, "unknown"))
        self.assertEqual(_L2_RETREAT, stage_directive({"L1", "L2"}, ""))

    def test_fixer_holds_no_copy_of_the_bucket_name(self):
        """分流键在 fixer 里**只能是 import 进来的那一个常量**, 不许有第二份字面量

        这条钉的是「两套口径」这种形状: 取向若自己去正则匹配错误原文 (或在 fixer 里再写
        一份 `"interface_mismatch"`), 今天两条通道的结论碰巧一致, 用例照样绿; 而词表一改
        就变成「prompt 说是接口问题、库里记成 TypeError」的分裂 —— 复发门、检索与取向本来
        读的是同一个桶。所以断言的是**出处数量**, 不是当次结论。
        """
        src = open(_FIXER_SRC, encoding="utf-8").read()
        tree = ast.parse(src)
        # 只看**代码里的**字符串: docstring/注释里出现桶名是有意的 (它要解释那只 bug)
        doc_consts = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                first = (n.body or [None])[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    doc_consts.add(id(first.value))
        copies = [n.value for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and isinstance(n.value, str)
                  and "interface_mismatch" in n.value and id(n) not in doc_consts]
        self.assertEqual([], copies,
                         "fixer 里出现了桶名的第二份字面量副本 ⇒ 分流键与库归类脱钩")
        fn = next((n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "failure_class"), None)
        self.assertIsNotNone(fn, "找不到 failure_class ⇒ 解析面对不上被测代码")
        called = {n.func.id for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("failure_class_of", called,
                      "failure_class 没走库里那一个分类器 ⇒ 取向与归类是两套口径")
        imported = {a.name for n in tree.body if isinstance(n, ast.ImportFrom)
                    and n.module == "kernel_db" for a in n.names}
        self.assertIn("FAILURE_CLASS_INTERFACE", imported,
                      "分流键常量不是从 kernel_db import 的那一个 ⇒ 它另有一份出处")


class TestStageBranch(unittest.TestCase):

    def test_a_numeric_failure_permits_retreating_to_a_conservative_form(self):
        p = _prompt_for([_res(VerificationLevel.L2_FUNCTION, "最大绝对误差 1.2e-1 超出容差")])
        self.assertIn("数值检查器", p)
        self.assertIn("correctness 优先于性能", p)
        # 编译分支那段不许同时出现 —— 两句并存等于没分支 (缺陷族: 一个状态词盖两件事)
        self.assertNotIn("编译器 / 预检", p)
        self.assertNotIn("保持既有的优化策略不变", p)

    def test_a_compile_failure_tells_the_model_to_keep_the_strategy(self):
        p = _prompt_for([_res(VerificationLevel.L1_COMPILE, "SyntaxError: invalid syntax")])
        self.assertIn("编译器 / 预检", p)
        self.assertIn("保持既有的优化策略不变", p)
        self.assertNotIn("correctness 优先于性能", p)
        self.assertNotIn("撤销可疑的激进改动", p)

    def test_a_numeric_failure_outranks_a_performance_one(self):
        # ⚠️ 这个形状生产里**到不了** (`verifier.verify()` 每级失败就 return, 见下面
        # TestWhereThePerformanceTextActuallyComesFrom)。留着的价值是防御性的: 钉住
        # 分支条件是 L2 而不是 L3 —— 万一哪天 verify() 改成收集全部级别, 这条会替我们
        # 提醒「取向不能让性能失败决定」。它**不是**「今天会发生」的证据 (那句以前写错了)。
        p = _prompt_for([
            _res(VerificationLevel.L2_FUNCTION, "最大绝对误差 1.2e-1 超出容差"),
            _res(VerificationLevel.L3_PERFORMANCE, "加速比 0.98 未达标"),
        ])
        self.assertIn("correctness 优先于性能", p)
        self.assertIn("本轮**不要**为此优化", p)
        self.assertNotIn("保持既有的优化策略不变", p)

    def test_the_stage_that_failed_is_reported_rather_than_a_fixed_sentence(self):
        p = _prompt_for([
            _res(VerificationLevel.L1_COMPILE, "SyntaxError"),
            _res(VerificationLevel.L3_PERFORMANCE, "加速比未达标"),
        ])
        self.assertIn("L1, L3", p)
        # 模板里那句被合写的旧话不许以任何形式回来
        self.assertNotIn("以下代码在编译 (L1) 或功能 (L2) 验证中失败", p)
        # 退让只由**数值**失败触发: L1+L3 里没有 L2, 就不该许可退回保守写法
        # (这条同时钉住「分支的条件是 L2 而不是 L3」—— 让性能失败决定取向正是
        #  原文说的那个陷阱的反方向)
        self.assertIn("保持既有的优化策略不变", p)
        self.assertNotIn("correctness 优先于性能", p)

    def test_the_directive_survives_the_retry_feedback(self):
        # 预检重试的 feedback 是**追加**在 base_prompt 后面的, 取向不能被它冲掉
        class _BadThenGood(_LLM):
            def chat(self, messages, **kw):
                self.prompts.append(messages[-1]["content"])
                if len(self.prompts) == 1:
                    return "这次没有代码块"
                return f"```python\n{GOOD}```"

        llm = _BadThenGood()
        FixerAgent(llm=llm).run(BROKEN, _task(), [
            _res(VerificationLevel.L2_FUNCTION, "最大绝对误差超出容差")])
        self.assertGreaterEqual(len(llm.prompts), 2)
        self.assertIn("correctness 优先于性能", llm.prompts[1])

    def test_no_orientation_is_never_an_option(self):
        # 空列表 / 只给了没预料到的级别: 取向必须仍然是一段实文字
        self.assertEqual({_L2_RETREAT, _L1_INPLACE},
                         {stage_directive(set()), stage_directive({"L1"}),
                          stage_directive({"L2"}), stage_directive({"L3"}),
                          stage_directive({"L2", "L3"}), stage_directive({"L1", "L3"})})
        for lv in (set(), {"L1"}, {"L2"}, {"L3"}):
            self.assertTrue(stage_directive(lv).strip(), f"levels={lv} 取向为空")

    def test_the_rendered_prompt_has_no_unsubstituted_placeholder(self):
        p = _prompt_for([_res(VerificationLevel.L2_FUNCTION, "误差超标")])
        self.assertNotIn("{{", p)
        self.assertNotIn("未知", p)   # failure_stage 的兜底值不该出现在真实路径里


class TestThePerformanceTextIsDeadProseForNow(unittest.TestCase):
    """`_L2_RETREAT` 末句「若错误信息里同时出现性能未达标」**今天没有活对象** —— 两条通道都堵着

    这一处我连着纠正错两次, 两次的错都是同一个形状 (只核了机制、没核入口):
      ① 原话「`conductor.py:780` 把所有未通过结果交给 Fixer ⇒ 同轮并存」—— 读回
         `verifier.verify():233-249` 是**每级失败就 return** (fail-fast), 同轮最多一个
         未通过级别。
      ② 我改成「跨轮经 `history_cases` 并存」并说「L3 失败会被 `conductor.py:948-951`
         记进库」—— 这句也是错的: Fixer 那条支路被 `conductor.py:763` 的 `l1_l2_failed`
         门挡着, 循环外那处兜底入库 `:952-955` 明写 `if any(level in (L1, L2))` ⇒
         **今天的代码一条 L3 记录都不产**。(仓库现存 `output/failure_db.jsonl` 里那 2 条
         L3 没有生产写者, 全仓 grep 不到任何代码写出那句 error_summary。)

    所以末句是一条**明标的保险**: 它不替任何未通过的判定说话, 只在两条结构约束任一松掉时
    才有意义。下面两条守卫分别钉住那两条约束 —— 它们变红就是「这句该重读」的信号, 而不是
    「测试坏了」。第一条用例钉的是**检索侧**性质 (库里若真有 L3 记录, 它经上下文检索进得到
    数值那一轮的 prompt), 这条在今天的生产路径上到不了, 但它守着 #79 那个假标签的对照面。
    """

    def _fixer_with_history(self):
        from operator_agent.kernel_db import FailureDatabase, FailureRecord

        # `jsonl_path="off"` 不是随手写的: 默认构造会重放 `<repo>/output/failure_db.jsonl`,
        # 于是这条用例会读到**仓库自己的真语料** —— M9 第一次标定就是这么被污染成绿的
        # (过滤掉本文件加的那条 L3 之后, 盘上重放回来的 L3 记录仍把「加速比」带进 prompt)。
        db = FailureDatabase(jsonl_path="off")
        self.assertIsNone(db._jsonl, "持久化没关掉 ⇒ 用例会被 output/ 下的真语料污染")
        db.add(FailureRecord(failure_id="h-l3", kernel_type="matmul", hardware="mock_gpu",
                             dsl="python", level="L3",
                             error_summary="加速比 0.97x 未达标 (需要 >= 1.00x)",
                             fix_summary="加大 BLOCK_K", fixed=True))
        llm = _LLM()
        return llm, FixerAgent(llm=llm, failure_db=db)

    def test_a_stored_performance_failure_reaches_a_numeric_round(self):
        """检索侧性质: 库里有 L3 记录时, 它经**上下文**检索进得到数值那一轮的 prompt。
        (今天没有代码去产这条记录 —— 见本类的说明。)"""
        llm, fixer = self._fixer_with_history()
        fixer.run(BROKEN, _task(), [_res(VerificationLevel.L2_FUNCTION, "最大绝对误差超出容差")])
        p = llm.prompts[0]
        self.assertIn("加速比 0.97x 未达标", p,
                      "历史栏里没有那条 L3 ⇒ 末句即便哪天活了也仍没有对象")
        self.assertIn("[上下文命中]", p, "中文报错取不出词元, 这条只能经上下文检索来")
        self.assertNotIn("[错误相似度命中]", p, "零相似不得标成相似度命中 (#79)")
        self.assertIn("correctness 优先于性能", p)   # 本轮仍是数值退让

    def test_conductor_gates_both_the_fixer_and_the_recorder_on_l1_l2(self):
        """第二条通道的门: Fixer 支路与循环外兜底入库都只认 L1/L2 ⇒ 没有 L3 记录被产出来"""
        src = open(os.path.join(_ROOT, "operator_agent", "conductor.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        gate = next((n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "l1_l2_failed"
                             for t in n.targets)), None)
        self.assertIsNotNone(gate, "找不到 `l1_l2_failed` 这道门 —— Fixer 的入口条件变了")
        attrs = {a.attr for a in ast.walk(gate) if isinstance(a, ast.Attribute)}
        self.assertIn("L1_COMPILE", attrs)
        self.assertIn("L2_FUNCTION", attrs)
        self.assertNotIn("L3_PERFORMANCE", attrs,
                         "这道门把 L3 也放进来的那天, `_L2_RETREAT` 末句就活了 —— "
                         "届时本类的说明与 doc 49 §八 第 26 条都要重读")

    def test_verify_is_fail_fast_so_two_levels_cannot_co_occur(self):
        """钉住「同轮 L2+L3 不可达」这件事本身 —— 上面那条纠正全靠它"""
        early = _early_returns(_VERIFIER_SRC)
        self.assertGreaterEqual(len(early), 2,
                                f"verify() 里的『失败即 return』早退从 2 处变成 {len(early)} 处 —— "
                                "若改成收集全部级别, 本文件那条『防御性优先级』用例就变成现行路径, "
                                "而 doc 55 §六 与 doc 49 §八 第 26 条的纠正要重读")

    def test_that_guard_is_not_a_no_op(self):
        """反空转: 把早退写成「记下来继续往下跑」时, 上面那条匹配必须数到 0 ——
        否则它对真源码返回 0 也会 assert 成功, 那条守卫就是在空转。"""
        mutated = '''
class V:
    def verify(self, kernel_code, task, test_inputs):
        results = []
        l1 = self._verify_compile(kernel_code)
        results.append(l1)
        l2 = self._verify_function(kernel_code, test_inputs)
        results.append(l2)
        results.append(self._verify_performance(kernel_code))
        return results
'''
        self.assertEqual([], _early_returns(mutated),
                         "匹配器在一份**没有**早退的源码上也数到了早退 ⇒ 它在真源码上的绿没有意义")
        # 反向对照: 把 L3 加进门里, 上面那条门守卫必须能看见 (否则它也是空转)
        widened = ('l1_l2_failed = any(\n'
                   '    not r.passed and r.level in (VerificationLevel.L1_COMPILE,\n'
                   '    VerificationLevel.L2_FUNCTION, VerificationLevel.L3_PERFORMANCE)\n'
                   '    for r in verify_results)')
        attrs = {a.attr for a in ast.walk(ast.parse(widened)) if isinstance(a, ast.Attribute)}
        self.assertIn("L3_PERFORMANCE", attrs,
                      "对照组本身取不到 L3 ⇒ 上面那条断言是在空转")


if __name__ == "__main__":
    unittest.main(verbosity=2)
