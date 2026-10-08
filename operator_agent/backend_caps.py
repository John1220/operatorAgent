"""后端能力声明与「缺失必须响」(doc 49 §4.1 的第 1/2/4 条)

三条设计, 每条都有外部一手出处:

1. **可选方法在基类上存在、一调就 raise** —— Triton `python/triton/backends/driver.py:149-153`
   的 `allocate_default_profile_scratch` **不是** `@abstractmethod`, 函数体就是
   `raise NotImplementedError`。于是 `hasattr()` 恒为真、`getattr(..., None)` 恒不为 None,
   「静默 no-op」在这个设计下**不可表达**。本仓那一整类缺陷就是这么活下来的:
   `hasattr` 返回 False 时, 调用方把它读成「不需要」而不是「没有」——
   `conductor.py` 的 `analyze_dependencies` 分支因此在生产里死了很久 (doc 49 §3.2 B6)。
2. **能力是提供方声明的闭合词汇**, 不是调用方猜的类型 (Triton 用类属性
   `supports_native_tensor_specialization` 声明差异、无人 `isinstance`;
   DSH `docs/subsystems/sandbox.md:22` 的说法是「完备度是**被上报的事实**」)。
   用 Enum 而不是字符串: 拼错的名字在写下的那一刻就炸, 而不是被 `.get(...)` 读成「不支持」。
3. **缺失时抛出的异常必须携带下层的原因** —— DSH postmortem 0004 的 Lessons:
   适配器要保留下层拥有的结构化失败, 不能换成自己最近的泛型类别。
   所以门面对可选方法只做**转发**: 内层没有时, 抛出来的是内层那一层的 `CapabilityMissing`。

⚠️ 这里的 capability 是**传输/提供方能力**, 与 `hardware_caps.py` 的**器件能力**
(`has_tma` / `supported_dsls` / `launch_floor_us`) 是两条正交的轴 —— DSH 在
2026-06-13 那篇笔记 `:38` 专门点名了这个同名陷阱。证明它们独立的例子: 指向**同一台 H20** 的
`SSHExecutor` 与 `HTTPExecutor`, 器件能力完全相同、后端能力完全不同。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

if TYPE_CHECKING:                      # 只给注解用: 运行时不导入, 免得与 candidates 成环
    from .candidates import ParamSpace


class Cap(str, Enum):
    """闭合词汇。新增成员要同时给出「谁声明它」与「谁读它」, 否则它就是死词汇

    (DSH `scripts/gen-doc-graphs.ts:563` 对「声明了却没人派发的事件」直接构建失败;
     本仓对应的失败模式叫「算了没人读」。)
    """

    #: 真加速卡上执行 —— 不是模拟、不是 CPU 回退
    REAL_DEVICE = "real_device"
    #: 提供方是模拟器 (SYCL 的 `aspect::emulated` 那个意思: 不为性能而生)
    EMULATED = "emulated"
    #: 性能数字是按硬件规格**估**的, 不是测的 (`mock_gpu` 的 `"mode": "simulated"`)
    SIMULATED_PERF = "simulated_perf"
    #: `verify_function`: 候选与参考在**同一台设备**上比对, 只回摘要
    REMOTE_VERIFY = "remote_verify"
    REFERENCE_IDENTITY = "reference_identity"
    #: `benchmark_with_baselines`: 候选与基线在同一张卡上交错计时
    SAME_DEVICE_BASELINE = "same_device_baseline"
    #: `ensure_environment`: 能自检并 provision 自己的运行环境
    ENV_PROVISION = "env_provision"
    #: `deploy`: 能把 runner 及其依赖同步到执行侧
    DEPLOY = "deploy"
    #: `analyze_dependencies`: 能从源码里解析出缺失的 import
    DEP_ANALYSIS = "dep_analysis"
    #: 跨轮固定同一张卡 (依赖 nvidia-smi; 非 NVIDIA 平台上 `_pinned_gpu` 返回 None)
    GPU_PINNING = "gpu_pinning"
    #: `execute()` 回传**完整输出张量** (`ExecutionResult.output`), 而不是采样
    #: (`RemoteExecutionResult.output_sample`)。L2 的比对方式由它决定: 全量比对走
    #: `tolerance` 的 AND 形式并给出 matched_ratio, 采样比对只能覆盖前缀、
    #: 并且必须把 `coverage` 一起报出去。
    FULL_OUTPUT_TENSOR = "full_output_tensor"


#: 可选能力 → 实现它的方法名。基类为每个方法都准备了「一调就 raise」的存根,
#: 所以这张表同时是**声明**与**实现**的对照表 (守卫会拿它核两边是否一致)。
OPTIONAL_CAP_METHODS = {
    Cap.REMOTE_VERIFY: "verify_function",
    Cap.REFERENCE_IDENTITY: "reference_identity",
    Cap.SAME_DEVICE_BASELINE: "benchmark_with_baselines",
    Cap.ENV_PROVISION: "ensure_environment",
    Cap.DEPLOY: "deploy",
    Cap.DEP_ANALYSIS: "analyze_dependencies",
}


class CapabilityMissing(RuntimeError):
    """能力缺失。**必须**带着下层给出的原因, 不换成自己最近的泛型类别

    `reason` 存在的理由是 DSH postmortem 0004: 那个 bash 后端把下层的
    `SandboxUnavailableError` 一律换成了自己的 `SEARCH_FAILED`, 于是「沙箱不可用」
    在日志里长得像「目录扫描失败」。本仓的对应版本是 `getattr(..., None)` 返回 None,
    然后调用方把 None 读成「走兼容路径」——两个意思共用一个值。
    """

    def __init__(self, caps, backend, reason: str = ""):
        self.caps = frozenset(caps if isinstance(caps, (set, frozenset, list, tuple)) else [caps])
        self.backend_label = backend.label
        self.reason = reason
        names = "/".join(sorted(c.value for c in self.caps))
        msg = f"后端 {self.backend_label} 不具备能力: {names}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


@dataclass(frozen=True)
class Instantiation:
    """模板候选**实例化**的产物 (§12): 一份可以交给评测器的源码 + 它的出处

    为什么实例化器要报 `instantiator_ver`: 参数空间的身份是
    `(template_id, instantiator_ver, 参数)` (`candidates.dedup_key`) —— 同一个点位由
    不同版本的实例化器产出的是**不同的构建**。若实例化器悄悄换了版本却不说, 去重键会
    把两个不同的构建读成同一个候选; 所以这里要求它**自报家门**, 由调用方比对。
    `""` = 没自报 (三态里的「不知道」)—— 调用方必须显式决定怎么处理, 不许当成匹配。

    `evidence` 是人读的一行出处 (模板名/参数/编译器版本之类), 落进产物便于复核。
    """

    source: str
    instantiator_ver: str = ""
    evidence: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "instantiator_ver": self.instantiator_ver,
                "evidence": self.evidence}


class Backend(ABC):
    """执行后端的 Service Definition (DSH 三角色里的第一个)

    必需方法用 `@abstractmethod`: 少一个就**构造不出来** (Triton 把同类校验写在
    `BaseBackend.__init__` 的 `assert self.supports_target(target)` 里, 意思一样:
    拿错后端不该有机会跑起来)。可选方法**不是** abstract, 而是基类上「一调就 raise」
    的存根 —— 见模块 docstring 第 1 条。
    """

    @property
    @abstractmethod
    def capabilities(self) -> frozenset:
        """本后端**真实**具备的 `Cap` 集合。声明得比实际多, 就是 3.3 那类静默降级"""

    def correctness_input_support(self, kernel_code: str, layout: str) -> tuple:
        """Whether L2 preserves suite values/layout in candidate execution.

        Default unknown: FULL_OUTPUT_TENSOR does not prove input fidelity. SSH
        regenerates values; JSON/pickle may erase strides. Providers must declare
        support for the actual source execution path, not just their hardware.
        """
        return False, "后端未声明正确性输入的值/布局保真传输"

    @property
    def label(self) -> str:
        """日志与异常里用的名字

        刻意不叫 `name`: 各提供方早就有自己的 `name` (有的是实例属性、有的是 property),
        在基类上再定义一个同名 property 会让 `self.name = ...` 直接 AttributeError。
        """
        return type(self).__name__

    @property
    def fingerprint_label(self) -> str:
        """进**环境指纹**的那个身份 (`environment.fingerprint_of` 的 backend 键)。

        默认 = `label`: 「同一台机器、换了个执行后端」是最要紧的漂移 (门面连不上时
        静默解析到 LocalExecutor, 器件名与版本一项都不变, 只有这个键会变 —— doc 49
        §3.1 A1)。**组合后端覆写它**返回被包住的执行后端的 label: 模板那一半是
        **候选来源**, 不是执行环境 —— 同机/同卡/同 runner 换组合壳被判「环境漂移」
        的实测教训 (CHECKLOG I72: 每个新模板后端首跑必 degraded, can_persist=False)。
        立在基类上而不是让调用方 `hasattr` 嗅探, 与 `param_space` 同一条理由。
        """
        return self.label

    @property
    def selected_gpu(self) -> Optional[int]:
        """**实际**用来跑核的那张卡的编号; `None` = 这里没有编号可报。

        存在的理由 (#52): 量具报表以前把 `knobs.gpu` 写成 `os.environ["OPERATOR_AGENT_GPU"]`,
        读的是用户**想**的那张卡, 而 `SSHExecutor` 从来不读那个变量 —— 它在本进程里自动挑
        一张最闲的卡。于是报表能写「跑在 GPU 2」而实际跑在 GPU 7: 记录的字段与发生的事
        不同源, 这正是本项目反复出现的那一族。

        三个状态要**合着 `capabilities` 一起读**, 因为单看这个值分不清后两个:
          * `Cap.GPU_PINNING` 不在能力里 → 这个后端没有「固定一张卡」这件事 (CPU / 模拟器 / HTTP)。
          * 在能力里但值为 `None` → 本来该固定, 可这一轮**没解析出卡号** (选卡失败, 远端自动选)。
          * 在能力里且是整数 → 就是这个数。
        基类默认 `None` 而不是抛: 这是一个**读值**, 不是一次能力调用; 而且「不固定」对
        CPU 后端是诚实答案。提供方要覆写它, 不许在这里加哨兵值 (如 `-1`) —— 那会让
        「没有卡」与「第 -1 号卡」共用一个数。
        """
        return None

    def param_space(self, contract: object = None) -> "Optional[ParamSpace]":
        """§12 的参数空间: **返回 None 的后端是代码型** —— 基类默认就是它

        与上面那些「一调就 raise」的存根**刻意不同**, 因为 None 在这里是一个**有意义
        的答案**, 不是「缺能力」: 设计 §12 原话「`Backend.param_space()` 返回 `None`
        的后端 = **代码型后端**，控制器自动切换成 "LLM / 模板产出候选、bandit 只做局部
        精修"」。基类若 raise, 今天每一个后端 (全是代码型) 都会在第一次搜索时炸。

        模板型后端覆写它并返回 `candidates.ParamSpace`; 那一刻控制器会 **fail-closed
        拒绝** (`candidates.require_supported_space`) —— 模板候选的实例化还没实现,
        声明的参数空间不许被无声丢掉。方法先立在基类上 (而不是让调用方 `hasattr` 嗅探)
        是同一个模块开头第 1 条的理由: 「静默 no-op」在这个设计下不可表达。
        """
        return None

    def instantiate(self, template_id: str, params: "Mapping[str, Any]",
                    contract: object = None) -> Instantiation:
        """把一个**参数点**实例化成源码 (§12 的 `TemplateCandidate` 那一半)

        基类默认 **raise**: 「声明了参数空间、却没有实例化器」不许被读成「实例化出一个
        空的候选」。这与 `param_space` 默认返回 `None` 是**刻意相反**的两个默认 ——
        `None` 在那里是有意义的答案 (代码型), 而这里没有「默认实例化」这种东西。

        模板型后端覆写它, 返回 `Instantiation(source=…, instantiator_ver=…, …)`。
        调用方 `candidates.instantiate_candidate` 会比对 `instantiator_ver` 与那个点
        在参数空间里的版本: 不一致 ⇒ 拒绝 (去重键会因此指称另一个构建)。
        """
        raise NotImplementedError(
            f"{self.label!r} 没有实现模板实例化 (Backend.instantiate): "
            f"模板 {template_id!r} 的参数 {dict(params)!r} 无法变成源码。"
            "声明了参数空间的后端必须自己实现它 —— §12 要挡的正是「把候选悄悄当成"
            "源码字符串」。")

    def lock_identity(self) -> str:
        """**同一个执行侧**的后端必须返回同一个串 —— 进程内串行锁按它取键

        默认用 `label`, 但指向同一台机器的两个后端可以有不同 label
        (`SSHExecutor@host(nvidia)` 与 `SSHExecutor@host(xpu)`), 那种情况下提供方要覆写:
        锁保护的是**那张卡**, 不是那份配置。
        门面以前按**请求的** mode 取键 (doc 49 §3.3 S6): `mode="auto"` 降级到本地之后
        键是 `auto:<host>`, 而 `if self._mode == "local"` 永远不触发 —— 两个都跑在本机
        CPU 上的后端拿到两把不同的锁, 于是它们可以并发。
        """
        return self.label

    def require(self, *caps: Cap) -> None:
        """消费方声明自己需要什么; 缺就抛, **不返回默认值**

        DSH `2026-08-19-session-projection-mandatory-seam.md:23,24,30` 把「给缺失状态
        一个默认值」列为被否决的方案: "configuration errors must fail explicitly"、
        "Explicit first-access failure … without allowing silent degradation"、
        "it never degrades to a default value"。
        """
        missing = [c for c in caps if c not in self.capabilities]
        if missing:
            raise CapabilityMissing(missing, self)

    # ── 必需能力 ──────────────────────────────────────────────
    @abstractmethod
    def get_hardware_info(self) -> dict:
        """返回硬件信息。**要么真探测、要么抛** —— 不许编一个像样的字典 (doc 49 §3.3 S1/S2)"""

    @abstractmethod
    def compile(self, kernel_code: str) -> bool:
        ...

    @abstractmethod
    def execute(self, kernel_code: str, inputs: dict):
        ...

    @abstractmethod
    def benchmark(self, kernel_code: str, inputs: dict, warmup: int = 3, repeat: int = 10) -> dict:
        ...

    # ── 可选能力: 基类上存在, 一调就抛 ─────────────────────────
    # 可选存根保证 `hasattr(backend, "deploy")`
    # 恒为真, 于是「用方法存不存在来判断能力」这种写法**再也得不到 False**。
    def verify_function(self, *args, **kwargs):
        raise CapabilityMissing(Cap.REMOTE_VERIFY, self)

    def reference_identity(self, *args, **kwargs):
        raise CapabilityMissing(Cap.REFERENCE_IDENTITY, self)

    def benchmark_with_baselines(self, *args, **kwargs):
        raise CapabilityMissing(Cap.SAME_DEVICE_BASELINE, self)

    def ensure_environment(self, *args, **kwargs):
        raise CapabilityMissing(Cap.ENV_PROVISION, self)

    def deploy(self, *args, **kwargs):
        raise CapabilityMissing(Cap.DEPLOY, self)

    def analyze_dependencies(self, *args, **kwargs):
        raise CapabilityMissing(Cap.DEP_ANALYSIS, self)


def concrete_backends() -> set:
    """`Backend` 的全部具体子类 (递归)。完备性守卫拿它与登记表双向对照"""
    out, stack = set(), list(Backend.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if not getattr(cls, "__abstractmethods__", None):
            out.add(cls)
    return out
