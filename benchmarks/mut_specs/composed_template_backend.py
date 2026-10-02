"""`TritonTemplateSSHExecutor`（d66c430, 真执行+真模板的组合层）的变异矩阵

d66c430 提交说明自己明写「本片还没有变异矩阵 (下一轮补)」—— 这就是那一轮。
目标只有一个类: 组合层。每一行对应一种**组合会静默失效**的写法, 不是抽象的
「代码被改了」; 证人全部是 `tests/test_triton_template_backend.py` 里的具体输入:

    M1 param_space 拿执行那一半的答案      M5 label 丢了模板身份
    M2 instantiate 拿执行那一半的答案      M6 capabilities 拿模板那一半的空集
    M3 转发 fail-open (缺方法返 None)      M7 template_backend 访问器指错那一半
    M4 转发整个断掉 (执行半不存在)         M8 组合时丢了 smem 上界

跑: python3 -B -m benchmarks.mutation_calibrate \
      --spec benchmarks/mut_specs/composed_template_backend.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {"tmt": "operator_agent/triton_matmul_template.py"}

TARGETS = ["tests/test_triton_template_backend.py"]

ROWS = [
    Row("M1 param_space 拿执行那一半的答案 (嗅探形状回归)", "tmt",
        "    def param_space(self, contract: object = None) -> ParamSpace:\n"
        "        return self._template.param_space(contract)",
        "    def param_space(self, contract: object = None) -> ParamSpace:\n"
        "        return getattr(self._executor, \"param_space\",\n"
        "                       lambda contract=None: None)(contract)",
        "⇒ `test_both_halves_come_from_the_right_place` 红: `_FakeExec` 没有 param_space ⇒ "
        "组合后端返 None ⇒ `.template_id` 上 AttributeError。真实失效: 代码型执行后端的 "
        "None 答案顶掉模板空间 ⇒ §12 开关把组合后端读成**代码型**, 轮首永远不取模板候选 "
        "—— 而且这正是类 docstring 点名禁的 `getattr(backend, \"param_space\", None)` 嗅探形状。"),

    Row("M2 instantiate 拿执行那一半的答案 (半声明: 有空间、没实例化)", "tmt",
        "        return self._template.instantiate(template_id, params, contract)",
        "        return getattr(self._executor, \"instantiate\",\n"
        "                       lambda *a, **k: None)(template_id, params, contract)",
        "⇒ 同一条用例红在 `.instantiator_ver` (None 上取属性)。真实失效: 类 docstring 的"
        "「一个后端要么两半都有、要么都没有」被破坏 —— 代码型执行侧没有 instantiate, "
        "模板轮在第一个点上就炸, 而 `param_space()` 又非 None, 控制器已经切进了模板模式。"),

    Row("M3 转发 fail-open: 缺方法返 None 而不是 raise", "tmt",
        "        return getattr(self._executor, name)",
        "        return getattr(self._executor, name, None)",
        "⇒ `test_a_missing_attribute_still_raises` 红: `be.not_a_method_at_all` 返 None。"
        "真实失效: 下游把 None 读成「不支持」, 「少了一个方法」被吞 —— 与能力嗅探同族, "
        "本仓 AST 守卫 (TestNoCapabilitySniffing) 拦的就是这个形状的**写法**, 这里拦它的**后果**。"),

    Row("M4 转发整个断掉 (执行那一半不存在)", "tmt",
        "        return getattr(self._executor, name)",
        "        raise AttributeError(name)",
        "⇒ `test_both_halves_come_from_the_right_place` 红在 `be.compile(\"src\")` "
        "(AttributeError)。真实失效: Verifier 拿到组合后端, 第一个执行调用就炸 —— "
        "「合成」没合成, 这一层的存在意义整个落空。⚠️ 与 M3 同锚点 (各自独立生效), "
        "锚点在文件里恰好一次。"),

    Row("M5 label 丢了模板身份 (产物里看不出组合)", "tmt",
        "        return f\"{self._executor.label}+{TEMPLATE_ID}@${INSTANTIATOR_VER}\"",
        "        return self._executor.label",
        "⇒ 同一条用例红在 `assertIn(\"triton_matmul_v1\", be.label)`。真实失效: 产物/报表里"
        "「跑在哪个后端」退回纯执行侧, 模板模式不可见 —— 与 #52 (报表读的是用户**想要**的卡) "
        "同族: 读数与事实脱钩。"),

    Row("M6 capabilities 拿了模板那一半的空集", "tmt",
        "    @property\n"
        "    def capabilities(self) -> frozenset:\n"
        "        return self._executor.capabilities",
        "    @property\n"
        "    def capabilities(self) -> frozenset:\n"
        "        return self._template.capabilities",
        "⇒ 同一条用例红在 `assertEqual({\"real_device\"}, be.capabilities)` (模板提供者的 "
        "capabilities 是空集)。真实失效: 组合后端对 `Cap.REAL_DEVICE` 之类的能力要求全部 "
        "fail-closed 拒绝 —— 它谁都服务不了, 而且**静默** (没有报错, 只有拒绝)。"),

    Row("M7 template_backend 访问器指错那一半", "tmt",
        "    @property\n"
        "    def template_backend(self) -> TritonMatmulTemplateBackend:\n"
        "        return self._template",
        "    @property\n"
        "    def template_backend(self) -> TritonMatmulTemplateBackend:\n"
        "        return self._executor",
        "⇒ 红在 `be.template_backend.param_space().template_id` (`_FakeExec` 没有 param_space)。"
        "⚠️ 这行矩阵**先于证人存在**: 访问器此前零测试, 矩阵设计时它必逃逸 —— 证人是为这行"
        "补的 (与 doc 74 M8 同族: 变异照出覆盖洞, 补喂法而不是删变异)。"),

    Row("M8 组合时丢了 smem 上界 (空间按默认格枚举)", "tmt",
        "        self._template = TritonMatmulTemplateBackend(smem_limit_bytes, elem_bytes)",
        "        self._template = TritonMatmulTemplateBackend(None, elem_bytes)",
        "⇒ `test_the_smem_limit_reaches_the_template_half` 红: 丢了上界 ⇒ 空间回到默认格, "
        "`assertLess(默认, 默认)` 不成立。真实失效: 剪枝才摘点 —— 这张卡上放不下的 tile "
        "全被实例化+测量一遍, 白烧机时 (I66 修的就是这个浪费的另一面: 剪枝口径)。"
        "证人是本片矩阵新补的, 此前上界传递零观察。"),
]
