"""缺口 5 第 1 片的守卫: 能力是**声明**出来的, 缺失必须响 (doc 49 §五 第 1 片)

这一片修的是一整类缺陷: 调用方用 `isinstance(backend, RemoteExecutor)` 或
`hasattr(backend, "deploy")` 猜能力, 猜错时**静默走另一条路**而不是报错。修法来自
Triton (`python/triton/backends/driver.py:149-153`: 可选方法在基类上存在, 函数体就是
`raise NotImplementedError`) 与 DSH (`docs/subsystems/sandbox.md:22`: 完备度是**被上报
的事实**; postmortem 0004: 适配器必须保留下层拥有的结构化失败)。

七条守卫, 每条都在 doc 49 §五 里配了「注入这个回归、看着它变红」的对照:

  ① 登记表 ↔ `Backend.__subclasses__()` **双向**完备 (缺一边就红)
  ② 每个后端声明的能力 == 登记表里写死的那一份 (改声明必须是一次有意识的编辑)
  ③ 声明了可选能力 ⇒ 解析到的方法**不是**基类那个一调就抛的存根
  ④ 门面对每个可选方法都**只转发** (AST: 不许再出现 getattr/hasattr 垫片)
  ⑤ `operator_agent/` 里不许再出现 `isinstance(x, <后端类>)` / `hasattr(backend, …)`
  ⑥ `return`/`raise` 之后不许有语句 —— 见 `test_no_unreachable_statements` 的由来
  ⑦ `require()` 必须有真实消费方, 否则它就是「算了没人读」
"""
import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.backend_caps import (
    OPTIONAL_CAP_METHODS, Backend, CapabilityMissing, Cap, concrete_backends,
)
from operator_agent.mock_gpu import MockGPU, _has_cuda
from operator_agent.remote_executor import (
    HTTPExecutor, LocalExecutor, RemoteExecutor, SSHExecutor,
)

REPO = Path(__file__).resolve().parents[1]
PRODUCTION = sorted((REPO / "operator_agent").glob("*.py"))

#: SSHExecutor(nvidia) 的完整声明。xpu 不声明 GPU_PINNING 或 CUDA REFERENCE_IDENTITY。
#: `_pinned_gpu()` 依赖
#: nvidia-smi, 对非 nvidia 平台第一行就 return None, 所以「跨轮固定同一张卡」在那边
#: 根本不发生; 不声明就是不说谎。
_SSH_CAPS = {Cap.REAL_DEVICE, Cap.REMOTE_VERIFY, Cap.SAME_DEVICE_BASELINE,
             Cap.ENV_PROVISION, Cap.DEPLOY, Cap.DEP_ANALYSIS, Cap.GPU_PINNING, Cap.REFERENCE_IDENTITY}

#: MockGPU 的 `get_hardware_info()` 返回写死的规格 (doc 49 §3.3 S8), 所以两种情形下
#: 都**不该**出现「硬件信息是真探测的」这类能力 —— 词汇表里现在压根没有这一项。
_MOCK_CAPS = ({Cap.REAL_DEVICE} if _has_cuda()
              else {Cap.EMULATED, Cap.SIMULATED_PERF}) | {Cap.FULL_OUTPUT_TENSOR}


def _facade(inner: Backend) -> RemoteExecutor:
    """门面 + 手工塞好的内层

    不能用 `RemoteExecutor(mode="ssh").capabilities`: 读它会触发 `_init_backend()`,
    而那会**真去连 SSH**。这里要验的是「能力派生自解析结果」, 不是连不连得上。
    """
    ex = RemoteExecutor(mode="ssh")
    ex._backend = inner
    return ex


#: 用**真实签名**调每个可选方法。门面的转发方法签名是写死的 (不是 `*args`),
#: 所以拿四个占位位置参数去调 `deploy()` 会先炸 TypeError, 而不是抛出我们要的
#: `CapabilityMissing` —— 那样守卫测到的是签名, 不是能力。
STUB_CALLS = {
    "reference_identity": lambda b: b.reference_identity({}),
    "verify_function": lambda b: b.verify_function("code", {}, "matmul", 1e-3),
    "benchmark_with_baselines": lambda b: b.benchmark_with_baselines("code", {}, "matmul"),
    "ensure_environment": lambda b: b.ensure_environment(),
    "deploy": lambda b: b.deploy(),
    "analyze_dependencies": lambda b: b.analyze_dependencies("code"),
}


#: 登记表: 条目名 → (类名, 不联网的构造器, 期望声明)
REGISTRY = {
    "MockGPU": ("MockGPU", lambda: MockGPU(), _MOCK_CAPS),
    "SSHExecutor(nvidia)": ("SSHExecutor", lambda: SSHExecutor(host="h", user="u"), _SSH_CAPS),
    "SSHExecutor(xpu)": ("SSHExecutor",
                         lambda: SSHExecutor(host="h", user="u", platform="xpu"),
                         _SSH_CAPS - {Cap.GPU_PINNING, Cap.REFERENCE_IDENTITY}),
    # 空集是**故意**的: `get_hardware_info()` 无条件编造一整套 A100 量级数字
    # (doc 49 §3.3 S2), 所以连「对面是不是加速卡」都不知道。等 S2 修掉再声明。
    "HTTPExecutor": ("HTTPExecutor", lambda: HTTPExecutor(base_url=""), frozenset()),
    "LocalExecutor": ("LocalExecutor", lambda: LocalExecutor(), frozenset()),
    "RemoteExecutor→SSH": ("RemoteExecutor", lambda: _facade(SSHExecutor(host="h", user="u")),
                           _SSH_CAPS),
    # 降级到 CPU 之后能力是空集 ⇒ 「降级了」这件事在能力上看得出来, 不必再靠
    # `"LocalExecutor" in remote.name` 这种子串匹配 (doc 49 §3.1 A3)
    "RemoteExecutor→Local": ("RemoteExecutor", lambda: RemoteExecutor(mode="local"),
                             frozenset()),
}


def _names(*caps) -> str:
    return "/".join(sorted(c.value for c in caps))


# ─── ① 登记表 ↔ 子类树, 双向 ────────────────────────────────

class TestRegistryIsCompleteBothWays(unittest.TestCase):
    """抄 DSH `scripts/gen-doc-graphs.ts:297-298`: `missing` 与 `stale` 都要抛

    只查一边是不够的: 只查「登记了但类不存在」会漏掉新增后端忘了登记 (那正是
    新能力静默缺席的入口); 只查「类存在但没登记」会漏掉登记表里的僵尸条目。
    """

    def _live(self):
        return {c.__name__ for c in concrete_backends()
                if c.__module__.startswith("operator_agent")}

    def test_every_live_backend_is_registered(self):
        registered = {cls for cls, _, _ in REGISTRY.values()}
        self.assertEqual(registered, self._live(),
                         "新增/删除后端时, 登记表必须同步改 (双向完备)")

    def test_registry_has_no_stale_entry(self):
        live = self._live()
        stale = sorted({cls for cls, _, _ in REGISTRY.values()} - live)
        self.assertEqual([], stale, f"登记表里有已经不存在的后端: {stale}")

    def test_registered_instances_are_really_that_class(self):
        for entry, (cls_name, factory, _) in REGISTRY.items():
            self.assertIsInstance(factory(), Backend, f"{entry} 不是 Backend 子类")
            self.assertEqual(cls_name, type(factory()).__name__,
                             f"{entry} 的登记表类名与构造出来的类不一致")


# ─── ②③ 声明的内容 ─────────────────────────────────────────

class TestDeclaredCapabilities(unittest.TestCase):

    def test_declarations_match_the_registry(self):
        for entry, (_, factory, expected) in REGISTRY.items():
            self.assertEqual(frozenset(expected), frozenset(factory().capabilities),
                             f"{entry} 的能力声明变了 —— 改声明必须同时改登记表, "
                             "因为声明是**契约**, 不是随手写的集合")

    def test_declared_optional_cap_is_implemented_not_inherited_stub(self):
        """声明了却不实现 = doc 49 §3.3 那一类静默降级, 只是换了个位置

        基类的存根一调就抛, 所以「声明多了」不会安静地返回 None —— 但它会在
        运行到那一行时才炸, 而且炸的地方看起来像调用方的错。这条守卫把它挪到测试期。
        """
        stubs = {name: Backend.__dict__[name] for name in OPTIONAL_CAP_METHODS.values()}
        for entry, (_, factory, expected) in REGISTRY.items():
            inst = factory()
            for cap in expected:
                name = OPTIONAL_CAP_METHODS.get(cap)
                if name is None:
                    continue
                resolved = getattr(inst, name).__func__
                self.assertIsNot(stubs[name], resolved,
                                 f"{entry} 声明了 {cap.value} 却没实现 {name}")

    def test_optional_cap_absent_means_the_stub_raises(self):
        """反向对照: 没声明的能力, 调它必须抛, 而不是安静地返回 None/True"""
        ex = RemoteExecutor(mode="local")
        for cap, name in OPTIONAL_CAP_METHODS.items():
            self.assertNotIn(cap, ex.capabilities)
            with self.assertRaises(CapabilityMissing, msg=f"调用未声明的 {name} 没抛"):
                STUB_CALLS[name](ex)

    def test_exception_carries_the_inner_backend_identity(self):
        """postmortem 0004: 保留下层拥有的结构化失败, 不换成自己最近的泛型类别"""
        ex = RemoteExecutor(mode="local")
        with self.assertRaises(CapabilityMissing) as ctx:
            ex.deploy()
        self.assertEqual("LocalExecutor (CPU)", ctx.exception.backend_label,
                         "门面的 label 是内层的 label; 异常里必须看得出是谁不支持")
        self.assertEqual({Cap.DEPLOY}, ctx.exception.caps)
        self.assertIn("deploy", str(ctx.exception))


# ─── ④⑤⑥⑦ AST 守卫 ───────────────────────────────────────

def _tree(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _code_of(fn) -> str:
    """函数体的**可执行部分** (去掉 docstring)

    必须去掉: 那几个转发方法的 docstring 里恰恰引用了旧写法
    `getattr(self.backend, "verify_function", None)` 来解释为什么不能再那么写。
    不去掉的话, 守卫会被自己的说明文字喂出假阳性 —— 量具被散文骗了。
    """
    body = fn.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return "\n".join(ast.unparse(s) for s in body)


def _class_def(tree, name):
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == name)


class TestFacadeOnlyForwards(unittest.TestCase):
    """④ 门面不许再用 `getattr(内层, "方法", None)` 垫片

    旧写法把「内层不支持」与「内层支持但返回 None」压成同一个值, 而 verifier 把 None
    读成「走兼容路径」(doc 49 §3.2 B3/B4)。现在: 每个可选方法都必须**显式转发**, 让
    内层那个一调就抛的存根自己说话。
    """

    def setUp(self):
        self.tree = _tree(REPO / "operator_agent" / "remote_executor.py")
        self.cls = _class_def(self.tree, "RemoteExecutor")
        self.methods = {n.name: n for n in self.cls.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def test_every_optional_method_is_forwarded(self):
        missing = sorted(set(OPTIONAL_CAP_METHODS.values()) - set(self.methods))
        self.assertEqual([], missing,
                         f"门面缺转发方法: {missing} —— 基类存根会抛, 但异常里说的"
                         "是门面不支持, 而不是内层不支持")

    def test_forwarding_contains_no_getattr_shim(self):
        for name, fn in self.methods.items():
            if name not in OPTIONAL_CAP_METHODS.values():
                continue
            src = _code_of(fn)
            for banned in ("getattr(", "hasattr("):
                self.assertNotIn(banned, src,
                                 f"RemoteExecutor.{name} 又用 {banned}… 垫片判能力了")
            self.assertIn(f"self.backend.{name}(", src,
                          f"RemoteExecutor.{name} 没有落到内层的同名方法上")

    def test_the_shim_guard_is_not_fooled_by_the_docstrings(self):
        """已知答案对照: 这五个方法的 docstring 里**就写着** `getattr(self.backend, …)`

        守卫若不去掉 docstring, 这条会立刻假阳性 —— 而假阳性的样子是「测试变红」,
        看起来像真的抓到了什么, 于是下一个人会去改说明文字而不是改判据。
        """
        for name in OPTIONAL_CAP_METHODS.values():
            raw = ast.unparse(self.methods[name])
            self.assertIn(name, raw)
        self.assertTrue(any("getattr(" in ast.unparse(self.methods[n])
                            for n in ("verify_function", "analyze_dependencies")),
                        "前提变了: 说明文字里已经不再引用旧写法, 这条对照失去意义")


class TestNoCapabilitySniffing(unittest.TestCase):
    """⑤ `operator_agent/` 里不许再出现按类型或按方法名猜能力的写法"""

    BACKEND_CLASSES = {"Backend"} | {c.__name__ for c in concrete_backends()}

    def _sniff_hits(self, trees=None):
        """判据本体。`trees` 可换成一段已知答案的代码 —— 对照必须走**同一条**判据,
        否则标定的是判据的副本, 不是判据"""
        hits = []
        for label, tree in (trees if trees is not None
                            else [(p.name, _tree(p)) for p in PRODUCTION]):
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id == "isinstance" and len(node.args) == 2:
                    for name in self._type_names(node.args[1]):
                        if name in self.BACKEND_CLASSES:
                            hits.append(f"{label}:{node.lineno} isinstance(…, {name})")
                if (isinstance(fn, ast.Name) and fn.id in ("hasattr", "getattr")
                        and node.args and self._is_backend_expr(node.args[0])):
                    hits.append(f"{label}:{node.lineno} {fn.id}(backend, …)")
        return hits

    @staticmethod
    def _type_names(node):
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            return [node.attr]
        if isinstance(node, ast.Tuple):
            return [n.id for n in node.elts if isinstance(n, ast.Name)]
        return []

    @staticmethod
    def _is_backend_expr(node) -> bool:
        """`self.backend` / `backend` / `x.backend` —— 但不包括 `backend_caps` 这类模块名"""
        if isinstance(node, ast.Name):
            return node.id == "backend"
        if isinstance(node, ast.Attribute):
            return node.attr == "backend"
        return False

    def test_no_isinstance_or_hasattr_dispatch_on_backends(self):
        self.assertEqual([], self._sniff_hits(),
                         "能力判断只许走 `Cap.X in backend.capabilities` (doc 49 §4.1 第 3 条)")

    def test_the_guard_would_have_caught_the_original_defect(self):
        """已知答案对照: 把 conductor 那三处旧写法喂进来, 守卫必须认出来

        没有这条, 上面那个 `assertEqual([], …)` 在守卫自己写错时会安静地通过 ——
        「会自我批评的错误判据比缺失的判据更难发现」。
        """
        probe = ast.parse(
            "class C:\n"
            "    def f(self, backend):\n"
            "        if hasattr(backend, 'deploy'):\n"
            "            backend.deploy()\n"
            "        if isinstance(self.backend, RemoteExecutor):\n"
            "            pass\n"
            "        fn = getattr(self.backend, 'verify_function', None)\n"
        )
        hits = self._sniff_hits([("probe", probe)])
        self.assertEqual(["probe:3 hasattr(backend, …)",
                          "probe:5 isinstance(…, RemoteExecutor)",
                          "probe:7 getattr(backend, …)"], hits)

    def test_a_benign_backend_reference_is_not_flagged(self):
        """反向对照: 不能写成「什么都拦」—— 那比不拦更糟

        `backend.capabilities` / `backend.require(…)` / `getattr(cfg, "backend")`
        都是新模型**要求**的写法, 被拦下来的话守卫就会逼人把能力判断改回类型判断。
        """
        benign = ast.parse(
            "def f(self, backend, cfg):\n"
            "    if Cap.DEPLOY in backend.capabilities:\n"
            "        backend.deploy()\n"
            "    backend.require(Cap.REAL_DEVICE)\n"
            "    name = getattr(cfg, 'backend')\n"
            "    if isinstance(name, str):\n"
            "        pass\n"
        )
        self.assertEqual([], self._sniff_hits([("benign", benign)]))


class TestNoUnreachableStatements(unittest.TestCase):
    """⑥ `return`/`raise` 之后不许还有语句

    由来 (2026-09-28, 本片的开发过程中真踩了一次): 往 `SSHExecutor.__init__` 的**尾部**
    插入 `@property capabilities` 时, 属性体的 `return frozenset(caps)` 后面跟了原本
    `__init__` 的最后一行 `self._uv_path = "…"`。Python 不报错、不警告, 那行就此不执行,
    而 `ensure_environment` 会在真机上 AttributeError —— 一个只在真机路径上出现的坑。
    """

    TERMINAL = (ast.Return, ast.Raise, ast.Break, ast.Continue)
    HOLDERS = (ast.If, ast.For, ast.While, ast.Try, ast.With,
               ast.FunctionDef, ast.AsyncFunctionDef, ast.ExceptHandler)

    def _blocks(self, node):
        for field in ("body", "orelse", "finalbody"):
            seq = getattr(node, field, None)
            if isinstance(seq, list) and seq:
                yield seq

    def _unreachable(self, trees):
        hits = []
        for label, tree in trees:
            for node in ast.walk(tree):
                if not isinstance(node, self.HOLDERS):
                    continue
                for seq in self._blocks(node):
                    for i, stmt in enumerate(seq[:-1]):
                        if isinstance(stmt, self.TERMINAL):
                            hits.append(f"{label}:{seq[i + 1].lineno} "
                                        f"(在 {type(stmt).__name__}@{stmt.lineno} 之后)")
        return hits

    def test_no_unreachable_statements(self):
        hits = self._unreachable([(p.name, _tree(p)) for p in PRODUCTION])
        self.assertEqual([], hits, f"这些语句永远执行不到: {hits}")

    def test_the_original_bug_is_flagged(self):
        """已知答案对照: 就是本片开发时真踩的那一个

        `@property` 插进 `__init__` 尾部之后, 属性体的 `return` 后面跟着原本
        `__init__` 的最后一行赋值。Python 接受它, 所以只有这条守卫能看见。
        """
        probe = ast.parse(
            "class SSHExecutor:\n"
            "    def __init__(self):\n"
            "        self._gpu_pin = None\n"
            "    @property\n"
            "    def capabilities(self):\n"
            "        return frozenset()\n"
            "        self._uv_path = '~/.local/bin/uv'\n"
        )
        self.assertEqual(["probe:7 (在 Return@6 之后)"], self._unreachable([("probe", probe)]))

    def test_a_terminator_in_its_own_block_is_fine(self):
        """反向对照: 终结语句只在**自己那一层**才算数

        `if x: return 1` 后面还有 `return 2` 是完全合法的写法; 循环里的 `continue`
        也一样。守卫要是把这些也报了, 它就会逼人把早退改成嵌套 if —— 比死代码更糟。
        """
        benign = ast.parse(
            "def f(x):\n"
            "    if x:\n"
            "        return 1\n"
            "    for i in range(3):\n"
            "        if i:\n"
            "            continue\n"
            "        break\n"
            "    try:\n"
            "        return 2\n"
            "    except ValueError:\n"
            "        raise\n"
            "    finally:\n"
            "        pass\n"
        )
        self.assertEqual([], self._unreachable([("benign", benign)]))


class TestRequireHasAConsumer(unittest.TestCase):
    """⑦ `require()` 不许只是「定义了但没人调」

    本仓反复出现的失效模式叫「算了没人读」(doc 48 §6.4 第 4 个缺陷、doc 49 §3.5)。
    一个没人调用的 `require` 比没有更糟: 它让读者以为消费方已经声明过需求了。
    """

    ROOTS = [REPO / "operator_agent", REPO / "benchmarks"]

    def _call_sites(self):
        sites = []
        for root in self.ROOTS:
            for path in sorted(root.rglob("*.py")):
                for node in ast.walk(_tree(path)):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "require" and node.args):
                        sites.append(f"{path.relative_to(REPO)}:{node.lineno}")
        return sites

    def test_require_is_called_outside_its_own_module(self):
        sites = [s for s in self._call_sites()
                 if not s.startswith("operator_agent/backend_caps.py")]
        self.assertTrue(sites, "require() 没有任何真实消费方 —— 它就是死词汇")

    def test_require_raises_with_every_missing_cap(self):
        ex = RemoteExecutor(mode="local")
        with self.assertRaises(CapabilityMissing) as ctx:
            ex.require(Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE)
        self.assertEqual({Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE}, ctx.exception.caps,
                         "缺两项就要报两项, 只报第一项会让人修一半")

    def test_require_is_silent_when_satisfied(self):
        ex = _facade(SSHExecutor(host="h", user="u"))
        self.assertIsNone(ex.require(Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE))


class TestBaseStubsExistOnEveryBackend(unittest.TestCase):
    """① 的另一半: 五个可选方法必须在**基类**上存在, 这样 hasattr 才恒为真

    Triton `driver.py:149-153` 的做法: 可选方法不是 `@abstractmethod`, 函数体就是
    `raise NotImplementedError`。于是「用方法存不存在判断能力」这种写法**不可表达**。
    """

    def test_stubs_are_defined_on_backend_and_raise(self):
        for cap, name in OPTIONAL_CAP_METHODS.items():
            self.assertIn(name, Backend.__dict__, f"{name} 必须是 Backend 自己的方法")
            inst = LocalExecutor()
            with self.assertRaises(CapabilityMissing) as ctx:
                getattr(inst, name)()
            self.assertEqual({cap}, ctx.exception.caps)

    def test_missing_required_method_makes_the_class_uninstantiable(self):
        """`@abstractmethod` 的作用: 少一个必需方法就**构造不出来**"""
        class Incomplete(Backend):
            @property
            def capabilities(self):
                return frozenset()

        with self.assertRaises(TypeError):
            Incomplete()
        self.assertNotIn(Incomplete, concrete_backends(),
                         "抽象类不该被当成具体后端")


if __name__ == "__main__":
    unittest.main(verbosity=2)
