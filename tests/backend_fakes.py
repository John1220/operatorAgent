"""测试用的后端假件: 能力是**声明出来的**, 不是 monkeypatch 上去的

为什么需要这个文件 (doc 49 §4.1 第 1 条的直接后果):
`Backend` 把五个可选方法做成了「基类上存在、一调就抛」的存根, 于是消费方改用
`Cap.X in backend.capabilities` 判断能力。旧写法

    backend = RemoteExecutor(mode="local")
    backend.benchmark_with_baselines = lambda *a, **kw: payload

在新模型下**静默失效**: 门面的能力派生自内层 `LocalExecutor` (空集), 消费方那道
`Cap.SAME_DEVICE_BASELINE in caps` 恒为 False, 打上去的方法根本不会被调用 ——
用例会以「走了另一条路」的方式通过或失败, 而看不出是声明缺了。

所以假件把这条规则做成构造期检查, 而且是**双向**的:
  ① 替换了某个可选方法 ⇒ 自动声明对应能力 (声明跟着实现走);
  ② 声明了某项可选能力却没给实现 ⇒ 构造时就 AssertionError, 不等用例跑一半。
②就是 `backend_caps.concrete_backends()` 那条完备性守卫在单个假件上的缩影。
"""
from __future__ import annotations

from operator_agent.backend_caps import OPTIONAL_CAP_METHODS, Backend, Cap
from operator_agent.remote_executor import LocalExecutor

#: 允许被替换的方法名。`label` 不是方法, 单独走构造参数。
OVERRIDABLE = frozenset(
    {"get_hardware_info", "compile", "execute", "benchmark"}
    | set(OPTIONAL_CAP_METHODS.values())
)


class DeclaredFakeBackend(Backend):
    """一个「具备的能力 == 声明的能力」的后端假件

    未被替换的必需方法委托给 `LocalExecutor` (在控制机上真跑 numpy), 与以前
    `RemoteExecutor(mode="local")` 的行为一致; 未被替换的**可选**方法走基类存根,
    也就是抛 `CapabilityMissing` —— 抛出来的 `backend_label` 是本假件, 不是 LocalExecutor。
    """

    def __init__(self, caps=(), label: str = "DeclaredFakeBackend", **overrides):
        unknown = sorted(set(overrides) - OVERRIDABLE)
        if unknown:
            raise AssertionError(
                f"假件不认识这些方法名: {unknown} (可替换: {sorted(OVERRIDABLE)})。"
                "拼错的方法名以前只是安静地不被调用, 现在必须响。")
        declared = set(caps)
        for cap, method in OPTIONAL_CAP_METHODS.items():
            if method in overrides:
                declared.add(cap)
        missing_impl = sorted(OPTIONAL_CAP_METHODS[c] for c in declared
                              if c in OPTIONAL_CAP_METHODS
                              and OPTIONAL_CAP_METHODS[c] not in overrides)
        if missing_impl:
            raise AssertionError(
                f"声明了可选能力却没提供实现: {missing_impl}。声明得比实际多就是静默降级 "
                "(doc 49 §3.3); 要么把实现传进来, 要么别声明这项能力。")
        self._overrides = overrides
        self._impl = LocalExecutor()
        self._caps = frozenset(declared)
        self.name = label

    @property
    def capabilities(self) -> frozenset:
        return self._caps

    @property
    def label(self) -> str:
        return self.name

    # ── 必需方法 ─────────────────────────────────────────────
    def get_hardware_info(self) -> dict:
        return self._call("get_hardware_info")

    def compile(self, kernel_code: str) -> bool:
        return self._call("compile", kernel_code)

    def execute(self, kernel_code: str, inputs: dict):
        return self._call("execute", kernel_code, inputs)

    def benchmark(self, kernel_code: str, inputs: dict, warmup: int = 3, repeat: int = 10) -> dict:
        return self._call("benchmark", kernel_code, inputs, warmup, repeat)

    # ── 可选方法: 有实现就用, 没有就落到基类那个一调就抛的存根 ──
    def verify_function(self, kernel_code, inputs, kernel_type, max_error, precision="",
                        reference_source=""):
        return self._optional("verify_function", kernel_code, inputs, kernel_type,
                              max_error, precision=precision,
                              reference_source=reference_source)

    def benchmark_with_baselines(self, kernel_code, inputs, kernel_type, warmup=10,
                                 repeat=30, precision="", baseline_priority=None,
                                 reference_source="", input_seed=None):
        # This in-process fake receives the actual inputs and does not implement
        # the remote seed transport. Accept its optional production argument,
        # without turning it into fabricated input-fidelity evidence.
        return self._optional("benchmark_with_baselines", kernel_code, inputs, kernel_type,
                              warmup, repeat, precision=precision,
                              baseline_priority=baseline_priority,
                              reference_source=reference_source)

    def ensure_environment(self, required_pips=None):
        return self._optional("ensure_environment", required_pips=required_pips)

    def calibration_probe(self, mode="heartbeat"):
        return self._optional("calibration_probe", mode=mode)

    def deploy(self):
        return self._optional("deploy")

    def analyze_dependencies(self, kernel_code):
        return self._optional("analyze_dependencies", kernel_code)

    def _call(self, name, *args, **kwargs):
        fn = self._overrides.get(name)
        return fn(*args, **kwargs) if fn else getattr(self._impl, name)(*args, **kwargs)

    def _optional(self, name, *args, **kwargs):
        fn = self._overrides.get(name)
        if fn is not None:
            return fn(*args, **kwargs)
        return getattr(Backend, name)(self, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.name}(caps={sorted(c.value for c in self._caps)})"


def fake_backend(**overrides) -> DeclaredFakeBackend:
    """最常见的用法: 只替换要观察的那几个方法, 能力自动跟着声明

        backend = fake_backend(benchmark_with_baselines=lambda *a, **kw: payload,
                               get_hardware_info=lambda: dict(H20_HW_INFO))
    """
    return DeclaredFakeBackend(**overrides)


def gpu_fake(**overrides) -> DeclaredFakeBackend:
    """带 `REAL_DEVICE` 的假件: 给「量具不许静默记录 CPU 数字」那类用例用

    `require(Cap.REAL_DEVICE, …)` 的消费方 (benchmarks/harness_pass_rate.py) 靠它做
    正向对照 —— 没有真卡的控制机上, 这条声明是**假件说出来的**, 而不是探测出来的,
    所以它只能用于用例, 不能用于生产路径。
    """
    return DeclaredFakeBackend(caps={Cap.REAL_DEVICE}, **overrides)
