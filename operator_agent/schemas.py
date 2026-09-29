"""
核心数据结构定义
Defines all schemas and data structures used across the operator agent system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import uuid
import time

from .backend_caps import Cap


class TaskType(Enum):
    """任务类型"""
    OPERATOR_GENERATION = "operator_generation"
    MODEL_ADAPTATION = "model_adaptation"
    CODE_TRANSLATION = "code_translation"


class DSLEnum(Enum):
    """支持的 DSL (Domain Specific Language) 类型"""
    PYTHON = "python"       # 纯 Python / NumPy (CPU)
    TRITON = "triton"       # Triton (GPU, 需要 CUDA)
    CUDA_C = "cuda_c"       # CUDA C/C++
    OPENCL = "opencl"       # OpenCL
    SYCL = "sycl"           # SYCL (oneAPI)

    @classmethod
    def is_gpu_dsl(cls, dsl: str) -> bool:
        """判断是否为 GPU 类型的 DSL"""
        gpu_dsls = {cls.TRITON.value, cls.CUDA_C.value, cls.OPENCL.value, cls.SYCL.value}
        return dsl.lower() in gpu_dsls

    @classmethod
    def is_triton(cls, dsl: str) -> bool:
        """判断是否为 Triton DSL"""
        return dsl.lower() == cls.TRITON.value


class TaskComplexity(Enum):
    """算子复杂度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Precision(Enum):
    """精度类型"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"

    @property
    def itemsize(self) -> int:
        """一个元素占几字节

        roofline 的 Byte 轴要用**任务声明的精度**, 不能用调用方传进来的数组 dtype:
        runner 会按声明精度转换输入 (`remote_scripts/run_kernel.py`: precision == "fp16"
        时 `.to(torch.float16)`), 所以控制端拿到的 float32 数组是核**根本见不到**的形态。
        用错会让算数强度整体偏小 2 倍 (matmul 4096³: 682.67 而不是 1365.33)。
        TF32 不是独立精度而是 fp32 的一种计算档位, 存储仍是 4 字节。
        """
        return {Precision.FP32: 4, Precision.FP16: 2,
                Precision.BF16: 2, Precision.INT8: 1}[self]


class TaskState(Enum):
    """任务状态机"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    ROUTED = "routed"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    OPTIMIZING = "optimizing"
    DONE = "done"
    FAILED = "failed"


class RouteType(Enum):
    """适配路线

    只有两个值是**可到达**的: `Router.route()` 的全部出口就是这两个。原来还有第三个
    `CODE_TRANSLATION`, 而它没有任何生产者 (tests/test_p1 那条断言因此长期红), 于是
    「有三条路线」这件事是声明出来的而不是实现的 —— 已删。真正的问题不是少一个枚举值,
    而是这个决策**没有任何消费者** (只被 print 了一次), 见 doc 49 §7.14。
    """
    COMPILER_BACKEND = "compiler_backend"
    AGENT_GENERATION = "agent_generation"


class VerificationLevel(Enum):
    """验证级别"""
    L1_COMPILE = "L1"       # 编译验证
    L2_FUNCTION = "L2"      # 功能验证
    L3_PERFORMANCE = "L3"   # 性能验证


@dataclass
class TritonCompileConfig:
    """Triton 编译配置 - 控制 Triton JIT 编译行为"""
    num_warps: int = 4           # 每个 program 的 warp 数量
    num_stages: int = 2          # 软件流水线阶段数
    enable_warp_specialization: bool = False  # warp 特化 (Triton 2.1+)
    enable_fp_fusion: bool = True # FP 融合 (FMA)
    max_num_imprecise_acc: int = 0  # 不精确累加最大次数 (None=全部精确)

    def to_dict(self) -> dict:
        return {
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "enable_warp_specialization": self.enable_warp_specialization,
            "enable_fp_fusion": self.enable_fp_fusion,
            "max_num_imprecise_acc": self.max_num_imprecise_acc,
        }

    def validate(self) -> list[str]:
        """校验 Triton 编译配置, 返回错误列表"""
        errors = []
        if not isinstance(self.num_warps, int) or self.num_warps <= 0:
            errors.append(f"TritonCompileConfig.num_warps 必须是正 int, 实际 {self.num_warps!r}")
        elif self.num_warps & (self.num_warps - 1):
            errors.append(
                f"TritonCompileConfig.num_warps={self.num_warps} 必须是 2 的幂 "
                "(否则 warp 调度会产生未定义行为)"
            )
        if not isinstance(self.num_stages, int) or self.num_stages <= 0:
            errors.append(f"TritonCompileConfig.num_stages 必须是正 int, 实际 {self.num_stages!r}")
        elif self.num_stages > 8:
            errors.append(
                f"[WARNING] TritonCompileConfig.num_stages={self.num_stages} 超过 8, "
                "可能超出共享内存容量 (A100 常用 3~5)"
            )
        if not isinstance(self.enable_warp_specialization, bool):
            errors.append("TritonCompileConfig.enable_warp_specialization 必须是 bool")
        if not isinstance(self.enable_fp_fusion, bool):
            errors.append("TritonCompileConfig.enable_fp_fusion 必须是 bool")
        if not isinstance(self.max_num_imprecise_acc, int) or self.max_num_imprecise_acc < 0:
            errors.append(f"TritonCompileConfig.max_num_imprecise_acc 必须是非负 int, 实际 {self.max_num_imprecise_acc!r}")
        return errors

    @classmethod
    def from_dict(cls, d: dict) -> "TritonCompileConfig":
        return cls(
            num_warps=d.get("num_warps", 4),
            num_stages=d.get("num_stages", 2),
            enable_warp_specialization=d.get("enable_warp_specialization", False),
            enable_fp_fusion=d.get("enable_fp_fusion", True),
            max_num_imprecise_acc=d.get("max_num_imprecise_acc", 0),
        )


@dataclass
class SourceSpec:
    """源算子描述"""
    framework: str = "pytorch"
    code_path: Optional[str] = None
    kernel_type: str = ""
    fused_ops: list = field(default_factory=list)
    complexity: TaskComplexity = TaskComplexity.MEDIUM

    # 已在项目中实现 / 被 KernelDB 收录的算子类型, 用于 Designer/Verifier 中做合法值范围校验
    _KNOWN_KERNEL_TYPES = frozenset({
        "matmul", "gemm",
        "softmax", "log_softmax",
        "layernorm", "rmsnorm",
        "attention", "flash_attention", "flash_attention_v2",
        "add", "relu", "gelu", "silu",
        "conv2d",
    })

    def validate(self) -> list[str]:
        errors = []
        if not isinstance(self.kernel_type, str) or not self.kernel_type.strip():
            errors.append("source.kernel_type 不能为空字符串")
        elif self.kernel_type.lower() not in self._KNOWN_KERNEL_TYPES:
            # 非已知类型只 Warning (允许探索新算子), 不阻塞
            errors.append(
                f"[WARNING] source.kernel_type='{self.kernel_type}' 不在已知集合 "
                f"{sorted(self._KNOWN_KERNEL_TYPES)} 内, 请确认拼写"
            )
        if not isinstance(self.fused_ops, list):
            errors.append(f"source.fused_ops 必须是 list, 实际 {type(self.fused_ops).__name__}")
        if not isinstance(self.complexity, TaskComplexity):
            errors.append(f"source.complexity 必须是 TaskComplexity 枚举")
        return errors


@dataclass
class TargetSpec:
    """目标硬件描述"""
    hardware: str = "mock_gpu"
    dsl: str = "python"          # python | triton | cuda_c | opencl | sycl
    compute_units: int = 64
    warp_size: int = 32
    vector_width: int = 128
    triton_config: Optional[TritonCompileConfig] = None  # Triton 专用编译配置
    #: 这次运行**要求执行侧具备**的 `Cap`。消费方声明自己需要什么 (与 `Backend.require()`
    #: 同一套词汇), 缺了不是「降级继续」而是「这份读数没有资格判定 L3」——
    #: 见 `measurement.provenance_conflicts` 与 doc 49 §3.5 第 2/3 处。
    #:
    #: 默认空集是有意的: 本地 mock 运行本来就没有真卡, 无条件要求 `REAL_DEVICE`
    #: 等于把整条离线链路判死。要真卡的是**量具与真机驱动**, 由它们显式声明
    #: (`benchmarks/harness_pass_rate.py` 就是这么声明的)。
    required_caps: tuple = ()

    # 已知硬件 + 已知 DSL (允许 Router 的 _DSL_CAPABILITY 之外的值, 只做 warning)
    _KNOWN_HARDWARE = frozenset({
        "nvidia_gpu", "mock_gpu",
        "ascend", "npu",
        "amd_gpu", "rocm",
    })
    _KNOWN_DSL = frozenset({e.value for e in DSLEnum})

    def __post_init__(self):
        """当 dsl 为 triton 但未提供 config 时，自动创建默认配置"""
        if DSLEnum.is_triton(self.dsl) and self.triton_config is None:
            self.triton_config = TritonCompileConfig()
        # `required_caps` 一律归一成 `Cap` 成员的 frozenset。走 `Cap(v)` 而不是原样收
        # 字符串: 闭合词汇的意义就是「拼错的名字在写下的那一刻炸」。收字符串的话,
        # 一个拼错的要求会被下游的集合差读成「不要求任何事」, 门就此静默敞开
        # (doc 49 §4.1 第 2 条)。
        self.required_caps = frozenset(Cap(v) for v in (self.required_caps or ()))

    def validate(self) -> list[str]:
        errors = []
        if not isinstance(self.hardware, str) or not self.hardware.strip():
            errors.append("target.hardware 不能为空")
        elif self.hardware.lower() not in self._KNOWN_HARDWARE:
            errors.append(
                f"[WARNING] target.hardware='{self.hardware}' 不在已知集合 "
                f"{sorted(self._KNOWN_HARDWARE)} 内"
            )
        if not isinstance(self.dsl, str) or not self.dsl.strip():
            errors.append("target.dsl 不能为空")
        elif self.dsl.lower() not in self._KNOWN_DSL:
            errors.append(
                f"target.dsl='{self.dsl}' 不在允许集合 {sorted(self._KNOWN_DSL)} 内"
            )
        for name in ("compute_units", "warp_size", "vector_width"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"target.{name} 必须是正 int, 实际 {value!r}")
        if DSLEnum.is_triton(self.dsl) and self.triton_config is not None:
            errors.extend(self.triton_config.validate())
        return errors


@dataclass
class ConstraintSpec:
    """约束条件"""
    precision: Precision = Precision.FP32
    min_speedup: float = 1.0
    max_error: float = 1e-5

    def validate(self) -> list[str]:
        errors = []
        if not isinstance(self.precision, Precision):
            errors.append("constraints.precision 必须是 Precision 枚举")
        if not isinstance(self.min_speedup, (int, float)) or isinstance(self.min_speedup, bool):
            errors.append(f"constraints.min_speedup 必须是数值, 实际 {type(self.min_speedup).__name__}")
        elif self.min_speedup < 0:
            errors.append(f"constraints.min_speedup 不能为负, 实际 {self.min_speedup}")
        if not isinstance(self.max_error, (int, float)) or isinstance(self.max_error, bool):
            errors.append(f"constraints.max_error 必须是数值, 实际 {type(self.max_error).__name__}")
        elif self.max_error < 0:
            errors.append(f"constraints.max_error 不能为负, 实际 {self.max_error}")
        return errors


@dataclass
class TaskSchema:
    """标准任务描述 - 所有下游模块的统一契约"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: TaskType = TaskType.OPERATOR_GENERATION
    source: SourceSpec = field(default_factory=SourceSpec)
    target: TargetSpec = field(default_factory=TargetSpec)
    constraints: ConstraintSpec = field(default_factory=ConstraintSpec)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        target_dict = {
            "hardware": self.target.hardware,
            "dsl": self.target.dsl,
            "compute_units": self.target.compute_units,
            "warp_size": self.target.warp_size,
        }
        if self.target.triton_config:
            target_dict["triton_config"] = self.target.triton_config.to_dict()
        # 只在非空时写: 产物里一个恒为 [] 的键会被读成「任务声明过、要求为空」,
        # 而绝大多数任务确实没声明 —— 这两种情况该在产物里长得不一样。
        if self.target.required_caps:
            target_dict["required_caps"] = sorted(
                c.value for c in self.target.required_caps)
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "source": {
                "framework": self.source.framework,
                "kernel_type": self.source.kernel_type,
                "fused_ops": self.source.fused_ops,
                "complexity": self.source.complexity.value,
            },
            "target": target_dict,
            "constraints": {
                "precision": self.constraints.precision.value,
                "min_speedup": self.constraints.min_speedup,
                "max_error": self.constraints.max_error,
            },
        }

    def validate(self) -> list[str]:
        """校验任务描述契约, 统一收集所有子字段的错误; 返回空列表=全部通过"""
        errors = []
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            errors.append("task.task_id 不能为空")
        if not isinstance(self.task_type, TaskType):
            errors.append("task.task_type 必须是 TaskType 枚举")
        # 子结构校验: 合并各组件的错误, 加上所属前缀便于定位
        for prefix, obj in (
            ("task.source", self.source),
            ("task.target", self.target),
            ("task.constraints", self.constraints),
        ):
            sub_errs = obj.validate()
            for e in sub_errs:
                # WARNING 不加前缀, 保持可读性; 其他错误标明来源
                if e.startswith("[WARNING]"):
                    errors.append(e)
                else:
                    errors.append(f"{prefix}: {e}")
        return errors


@dataclass
class TilingStrategy:
    """Tiling策略"""
    block_m: int = 64
    block_n: int = 64
    block_k: int = 32

    # 允许的 tile 取值 (2 的幂, 覆盖小/中/大矩阵常见配置)
    _ALLOWED_BLOCK_M_N = frozenset({16, 32, 64, 128, 256, 512})
    _ALLOWED_BLOCK_K = frozenset({16, 32, 64, 128})

    def validate(self) -> list[str]:
        """校验 tile 参数合法性, 返回错误列表 (空=通过)"""
        errors = []
        for name, value, allowed in (
            ("block_m", self.block_m, self._ALLOWED_BLOCK_M_N),
            ("block_n", self.block_n, self._ALLOWED_BLOCK_M_N),
            ("block_k", self.block_k, self._ALLOWED_BLOCK_K),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"tiling.{name} 必须是 int, 实际 {type(value).__name__}")
                continue
            if value <= 0:
                errors.append(f"tiling.{name} 必须为正整数, 实际 {value}")
            elif value not in allowed:
                errors.append(
                    f"tiling.{name}={value} 不在允许集合 {sorted(allowed)} 内; "
                    "请选 2 的幂的常见 tile (避免非对齐 tile 导致编译/性能异常)"
                )
        return errors


@dataclass
class MemoryPlan:
    """内存层次规划"""
    global_memory: list = field(default_factory=list)
    local_memory: list = field(default_factory=list)
    register_memory: list = field(default_factory=list)


@dataclass
class PipelineStage:
    """流水线阶段"""
    stage: str = ""
    source: str = ""
    destination: str = ""
    hint: str = ""


@dataclass
class UnifiedSketch:
    """Designer Agent 输出的统一设计草图"""
    algorithm: str = ""
    version: int = 1
    tiling: TilingStrategy = field(default_factory=TilingStrategy)
    memory_plan: MemoryPlan = field(default_factory=MemoryPlan)
    pipeline: list = field(default_factory=list)
    target_dsl: str = "python"
    optimization_hints: list = field(default_factory=list)
    # C2 热/冷两层 (docs/research/20 第三节)。`optimization_hints` 是**累计日志**
    # (冷层, 只进快照与 SearchAgent); `hints_this_version` 是本轮 SearchAgent 新提的
    # 那条 (热层, 唯一进 Coder prompt 的)。
    # 这里刻意不做摘要/压缩器: 一手证据是递归摘要会毁掉事实与操作性约束, 而我方
    # 的历史负担是「逐轮堆积」—— 不注入就行, 不需要压。
    hints_this_version: list = field(default_factory=list)

    def hot_hints(self) -> list:
        """给 Coder 看的那一份: 本轮新 hint; 第一轮还没有「本轮」, 退回全量日志"""
        return self.hints_this_version or self.optimization_hints

    def to_prompt_dict(self) -> dict:
        """渲染进 prompt 的 sketch 结构: **不带任何 hint**

        hint 在 prompt 里只留一个通道 —— 模板的 {{optimization_hints}} 块 (渲染的是
        hot_hints())。此前 sketch JSON 与该块是同一份内容的两个注入点, 实测每条
        hint 在 Coder prompt 里出现两次。
        """
        d = self.to_dict()
        d.pop("optimization_hints", None)
        d.pop("hints_this_version", None)
        return d

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "version": self.version,
            "tiling": {
                "block_m": self.tiling.block_m,
                "block_n": self.tiling.block_n,
                "block_k": self.tiling.block_k,
            },
            "memory_plan": {
                "global": self.memory_plan.global_memory,
                "local": self.memory_plan.local_memory,
                "register": self.memory_plan.register_memory,
            },
            "pipeline": [
                {"stage": p.stage, "from": p.source, "to": p.destination, "hint": p.hint}
                for p in self.pipeline
            ],
            "target_dsl": self.target_dsl,
            "optimization_hints": self.optimization_hints,
            "hints_this_version": self.hints_this_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UnifiedSketch":
        """从 to_dict() 产物反序列化, 缺失字段用默认值填充 (向前兼容)"""
        tiling_d = d.get("tiling") or {}
        mem_d = d.get("memory_plan") or {}
        pipe_d = d.get("pipeline") or []
        return cls(
            algorithm=d.get("algorithm", "matmul"),
            version=int(d.get("version") or 1),
            tiling=TilingStrategy(
                block_m=int(tiling_d.get("block_m") or 64),
                block_n=int(tiling_d.get("block_n") or 64),
                block_k=int(tiling_d.get("block_k") or 32),
            ),
            memory_plan=MemoryPlan(
                global_memory=mem_d.get("global", ""),
                local_memory=mem_d.get("local", ""),
                register_memory=mem_d.get("register", ""),
            ),
            pipeline=[
                PipelineStage(
                    stage=p.get("stage"),
                    source=p.get("from"),
                    destination=p.get("to"),
                    hint=p.get("hint"),
                )
                for p in pipe_d
                if isinstance(p, dict)
            ],
            target_dsl=d.get("target_dsl", "python"),
            optimization_hints=list(d.get("optimization_hints") or []),
            hints_this_version=list(d.get("hints_this_version") or []),
        )

    _KNOWN_ALGORITHMS = frozenset({
        "tiled_gemm", "tiled_matmul", "gemm", "matmul",
        "online_softmax", "softmax",
        "layernorm", "rmsnorm",
        "flash_attention_v1", "flash_attention_v2", "flash_attention",
        "elementwise",
    })

    def validate(self) -> list[str]:
        """校验 Sketch 合法性校验, 返回错误列表 (空=通过); 对应 Hermes Function-Calling 的 function schema 校验

        该方法拦截 LLM 在 Designer 输出的典型错误(乱码)."""
        errors = []
        if not isinstance(self.algorithm, str) or not self.algorithm.strip():
            errors.append("sketch.algorithm 不能为空 (Designer 必须给出明确算法名)")
        elif self.algorithm.lower() not in self._KNOWN_ALGORITHMS:
            errors.append(
                f"[WARNING] sketch.algorithm='{self.algorithm}' 不在已知集合 "
                f"{sorted(self._KNOWN_ALGORITHMS)} 内, 请确认; 仅当确实在探索新算子时才允许"
            )
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version <= 0:
            errors.append(f"sketch.version 必须是正 int, 实际 {self.version!r}")

        # Tiling
        tiling_errs = self.tiling.validate()
        errors.extend(f"sketch: {e}" for e in tiling_errs)

        # pipeline 中每个 stage 的结构校验
        if not isinstance(self.pipeline, list):
            errors.append(f"sketch.pipeline 必须是 list, 实际 {type(self.pipeline).__name__}")
        else:
            for i, stage in enumerate(self.pipeline):
                if not isinstance(stage, PipelineStage):
                    errors.append(
                        f"sketch.pipeline[{i}] 必须是 PipelineStage 实例, 实际 {type(stage).__name__}")

        # target_dsl 校验
        if not isinstance(self.target_dsl, str) or not self.target_dsl.strip():
            errors.append("sketch.target_dsl 不能为空")
        elif self.target_dsl.lower() not in {e.value for e in DSLEnum}:
            errors.append(
                f"sketch.target_dsl='{self.target_dsl}' 不在允许集合 {sorted(e.value for e in DSLEnum)} 内"
            )

        # optimization_hints 类型校验
        if not isinstance(self.optimization_hints, list):
            errors.append("sketch.optimization_hints 必须是 list")
        elif any(not isinstance(h, str) for h in self.optimization_hints):
            errors.append("sketch.optimization_hints 所有元素必须是 str")

        # memory_plan 类型
        mp = self.memory_plan
        for name in ("global_memory", "local_memory", "register_memory"):
            val = getattr(mp, name)
            if not isinstance(val, list):
                errors.append(f"sketch.memory_plan.{name} 必须是 list, 实际 {type(val).__name__}")
        return errors


@dataclass
class VerificationResult:
    """验证结果"""
    level: VerificationLevel
    passed: bool
    message: str = ""
    metrics: dict = field(default_factory=dict)
    error_log: Optional[str] = None


@dataclass
class KernelArtifact:
    """最终输出的 Kernel 产物"""
    kernel_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str = ""
    code: str = ""
    language: str = "python"           # python | triton | cuda_c
    sketch: Optional[UnifiedSketch] = None
    verification_results: list = field(default_factory=list)
    benchmark_data: dict = field(default_factory=dict)
    compile_info: dict = field(default_factory=dict)  # 编译元信息 (Triton: num_warps, registers, etc.)
    iteration_history: list = field(default_factory=list)  # 血缘: 每轮 {round, action, all_passed, gflops}
    #: 本次运行的环境指纹 (`environment.fingerprint_of` 的产物)。它以前只在
    #: `conductor` 里被 `print` 一次就丢掉 (doc 49 §3.5 第 1 处), 于是「环境漂移自检」
    #: 这句注释描述的是一个不存在的机制。落盘的读者是**下一次运行**: 没有它, 下一次
    #: 就无从比较。
    environment_fingerprint: dict = field(default_factory=dict)
    #: 与上一次运行的指纹比较结果 (`environment.EnvironmentDrift.as_dict()`)。
    #: None = 本次没做比较 (无从比较时也写一个 comparable=False 的字典, 不写 None)。
    environment_drift: Optional[dict] = None
    metadata: Optional[dict] = None    # 存放成本监控账单 / 执行快照路径 等外挂字段 (可 JSON 化)
    saved_dir: str = ""                # ArtifactStore.save 落盘目录 (由 Conductor 赋值, 便于引用)
    created_at: float = field(default_factory=time.time)

    @property
    def all_passed(self) -> bool:
        return all(v.passed for v in self.verification_results)

    @property
    def is_triton(self) -> bool:
        """判断是否为 Triton kernel"""
        return DSLEnum.is_triton(self.language)

    def to_summary(self) -> dict:
        """生成摘要信息"""
        return {
            "kernel_id": self.kernel_id,
            "task_id": self.task_id,
            "language": self.language,
            "code_length": len(self.code),
            "sketch_version": self.sketch.version if self.sketch else None,
            "all_passed": self.all_passed,
            "benchmark": self.benchmark_data,
            "compile_info": self.compile_info,
        }
