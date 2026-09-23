"""Hardware Capability Descriptor
硬件能力描述 — 跨平台 GPU 硬件特征的统一抽象层

设计原则:
  1. 不假设任何特定厂商 (NVIDIA/AMD/昇腾/寒武纪)
  2. 所有字段都有合理默认值 (对应"最小能力"硬件)
  3. 从 backend.get_hardware_info() 自动推断，也可手动配置
  4. OptimizationPlanner 根据这些能力选择配方，而非硬编码硬件名

能力分层:
  - Compute: 计算单元类型 (tensor_core 版本、向量宽度)
  - Memory: 内存层次 (共享内存/L2/带宽)
  - ISA: 指令特性 (缓存提示、原子操作、特殊函数)
  - Execution: 执行模型 (warp/sub_group 大小、最大并发)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class HardwareCaps:
    """硬件能力描述符 — 所有字段均有保守默认值"""

    # ---- 厂商与标识 ----
    vendor: str = "unknown"         # nvidia, amd, ascend, cambricon, ...
    device_name: str = "unknown"
    compute_capability: str = ""    # 如 "8.0" (NVIDIA), "gfx90a" (AMD)

    # ---- 计算能力 ----
    # 峰值算力 (GFLOPS), 0 表示未知
    fp32_tflops: float = 0.0
    tf32_tflops: float = 0.0        # Ampere+ Tensor Core
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0
    int8_tops: float = 0.0

    has_tensor_core: bool = False   # 是否有矩阵乘加速单元
    tensor_core_version: int = 0    # 1=Volta, 2=Turing, 3=Ampere, 4=Hopper
    tensor_core_m_k_n: tuple = (16, 16, 16)  # Tensor Core 基础形状

    # ---- 向量/SIMD 能力 ----
    vector_width_bytes: int = 16    # 原生向量加载宽度 (float4=16)
    sub_group_size: int = 32        # warp/wavefront/sub_group 大小

    # ---- 内存层次 ----
    global_mem_bandwidth_gbs: float = 0.0  # HBM/GDDR 带宽
    l2_cache_size_kb: int = 0       # L2 cache 大小 (KB)
    shared_mem_per_sm_kb: int = 0   # 每 SM shared memory (KB)
    shared_mem_bank_count: int = 32 # bank 数量 (NVIDIA=32, AMD=64)
    registers_per_thread: int = 255 # 每线程寄存器数

    # ---- 执行模型 ----
    sm_count: int = 0               # SM/CU/计算单元数量
    max_threads_per_sm: int = 2048  # 每 SM 最大线程数
    max_blocks_per_sm: int = 32     # 每 SM 最大 block 数
    max_warps_per_sm: int = 64      # 每 SM 最大 warp 数

    # ---- ISA / 编程模型特性 ----
    has_l2_cache_hint: bool = False      # 是否支持 L2 缓存提示 (eviction_policy)
    has_software_pipelining: bool = False  # 是否支持 async copy / software pipeline
    has_double_buffering: bool = False  # shared memory double buffering
    has_block_reduce: bool = True       # 是否支持 block 级 reduction
    has_warp_shuffle: bool = True       # 是否支持 warp shuffle
    has_warp_primitives: bool = True    # 是否支持 warp 级原语 (warp shuffle, tl.sum 等)
    has_shared_memory: bool = True      # 是否支持 shared memory (几乎所有 GPU 都支持)
    has_atomic_fadd: bool = True        # 是否支持 float 原子加
    has_fast_math: bool = True          # 是否支持 fast math (approx transcendental)

    # ---- 编程模型/DSL 支持 ----
    supported_dsls: set = field(default_factory=lambda: {"cuda_c", "python"})
    # triton, cuda_c, hip, ascend-c, bisheng (寒武纪), opencl

    def to_dict(self) -> dict:
        d = asdict(self)
        d["supported_dsls"] = sorted(self.supported_dsls)
        return d

    def supports_dsl(self, dsl: str) -> bool:
        return dsl in self.supported_dsls

    def effective_fp_compute_tflops(self, precision: str = "fp32",
                                    use_tensor_core: bool = True) -> float:
        """根据精度获取有效峰值 TFLOPS"""
        if precision == "fp16" and use_tensor_core and self.has_tensor_core:
            return self.fp16_tflops
        if precision == "bf16" and use_tensor_core and self.has_tensor_core:
            return self.bf16_tflops
        if precision == "fp32" and use_tensor_core and self.tf32_tflops > 0:
            return self.tf32_tflops
        return self.fp32_tflops


# ---- 已知硬件的预定义能力 ----
# 新硬件只需在此添加一条记录，Planner 自动适配

_KNOWN_HARDWARE: dict[str, HardwareCaps] = {
    # NVIDIA A100
    "a100": HardwareCaps(
        vendor="nvidia", device_name="A100",
        compute_capability="8.0",
        fp32_tflops=19.5, tf32_tflops=156.0, fp16_tflops=312.0, bf16_tflops=312.0,
        has_tensor_core=True, tensor_core_version=3, tensor_core_m_k_n=(16, 8, 16),
        vector_width_bytes=16, sub_group_size=32,
        global_mem_bandwidth_gbs=2039.0, l2_cache_size_kb=40960,
        shared_mem_per_sm_kb=164, shared_mem_bank_count=32,
        registers_per_thread=255,
        sm_count=108, max_threads_per_sm=2048, max_blocks_per_sm=32,
        has_l2_cache_hint=True, has_software_pipelining=True,
        has_double_buffering=True, has_block_reduce=True,
        has_warp_shuffle=True, has_atomic_fadd=True,
        supported_dsls={"triton", "cuda_c", "python"},
    ),
    # NVIDIA H100
    "h100": HardwareCaps(
        vendor="nvidia", device_name="H100",
        compute_capability="9.0",
        fp32_tflops=67.0, tf32_tflops=989.0, fp16_tflops=1979.0, bf16_tflops=989.0,
        int8_tops=3958.0,
        has_tensor_core=True, tensor_core_version=4, tensor_core_m_k_n=(16, 16, 16),
        vector_width_bytes=16, sub_group_size=32,
        global_mem_bandwidth_gbs=3350.0, l2_cache_size_kb=51200,
        shared_mem_per_sm_kb=228, shared_mem_bank_count=32,
        registers_per_thread=255,
        sm_count=132, max_threads_per_sm=2048,
        has_l2_cache_hint=True, has_software_pipelining=True,
        has_double_buffering=True, has_warp_shuffle=True,
        supported_dsls={"triton", "cuda_c", "python"},
    ),
    # NVIDIA H20 (Hopper 削减版) - 2026-09-16 于 algorithm-h20 实测 (cuBLAS 8192³ + triad)
    # 与 H100 同为 cc 9.0, 但算力砍到 138 TFLOPS 而带宽保留 3.68 TB/s
    # → ridge point ≈ 37.5 FLOP/B (H100 ≈ 295), 优化取向相反, 勿套用 H100 配方
    "h20": HardwareCaps(
        vendor="nvidia", device_name="H20",
        compute_capability="9.0",
        fp32_tflops=31.6, tf32_tflops=71.6, fp16_tflops=138.2, bf16_tflops=138.3,
        has_tensor_core=True, tensor_core_version=4, tensor_core_m_k_n=(16, 16, 16),
        vector_width_bytes=16, sub_group_size=32,
        global_mem_bandwidth_gbs=3680.0, l2_cache_size_kb=61440,
        shared_mem_per_sm_kb=228, shared_mem_bank_count=32,
        registers_per_thread=255,
        sm_count=78, max_threads_per_sm=2048,
        has_l2_cache_hint=True, has_software_pipelining=True,
        has_double_buffering=True, has_block_reduce=True,
        has_warp_shuffle=True, has_atomic_fadd=True,
        supported_dsls={"triton", "cuda_c", "python"},
    ),
    # NVIDIA V100
    "v100": HardwareCaps(
        vendor="nvidia", device_name="V100",
        compute_capability="7.0",
        fp32_tflops=15.7, fp16_tflops=125.0, bf16_tflops=125.0,
        has_tensor_core=True, tensor_core_version=1,
        global_mem_bandwidth_gbs=900.0, l2_cache_size_kb=6144,
        shared_mem_per_sm_kb=96,
        sm_count=80,
        has_l2_cache_hint=False, has_software_pipelining=False,
        supported_dsls={"triton", "cuda_c", "python"},
    ),
    # RTX 3090 / Ampere consumer
    "3090": HardwareCaps(
        vendor="nvidia", device_name="RTX 3090",
        compute_capability="8.6",
        fp32_tflops=16.3, tf32_tflops=130.6, fp16_tflops=326.0, bf16_tflops=326.0,
        has_tensor_core=True, tensor_core_version=3,
        global_mem_bandwidth_gbs=936.0, l2_cache_size_kb=6144,
        shared_mem_per_sm_kb=100,
        sm_count=82,
        has_l2_cache_hint=True, has_software_pipelining=True,
        supported_dsls={"triton", "cuda_c", "python"},
    ),
    # AMD MI250 (gfx90a)
    "mi250": HardwareCaps(
        vendor="amd", device_name="MI250",
        compute_capability="gfx90a",
        fp32_tflops=47.9, fp16_tflops=383.0, bf16_tflops=383.0,
        has_tensor_core=True, vector_width_bytes=16, sub_group_size=64,
        global_mem_bandwidth_gbs=3277.0, l2_cache_size_kb=8192,
        shared_mem_per_sm_kb=64, shared_mem_bank_count=64,
        sm_count=220,
        has_l2_cache_hint=True, has_software_pipelining=True,
        supported_dsls={"hip", "triton", "python"},
    ),
    # 华为昇腾 910B (示例配置)
    "ascend910b": HardwareCaps(
        vendor="ascend", device_name="Ascend 910B",
        compute_capability="davinci_m2",
        fp32_tflops=60.0, fp16_tflops=320.0, bf16_tflops=320.0,
        has_tensor_core=True, vector_width_bytes=32, sub_group_size=32,
        global_mem_bandwidth_gbs=1600.0,
        shared_mem_per_sm_kb=256,
        sm_count=64,
        has_l2_cache_hint=False, has_software_pipelining=True,
        supported_dsls={"ascend-c", "bisheng", "python"},
    ),
    # Mock / CPU fallback
    "mock": HardwareCaps(
        vendor="mock", device_name="MockGPU/CPU",
        fp32_tflops=0.1, global_mem_bandwidth_gbs=50.0,
        shared_mem_per_sm_kb=48, sm_count=2,
        has_tensor_core=False, has_l2_cache_hint=False,
        has_software_pipelining=False, has_double_buffering=False,
        has_block_reduce=True, has_warp_shuffle=False,
        supported_dsls={"python", "triton"},
    ),
}


def detect_hardware_caps(hw_info: dict) -> HardwareCaps:
    """
    从 backend.get_hardware_info() 返回的 dict 自动推断硬件能力。

    策略:
      1. 如果有已知 vendor + name，直接返回预定义配置
      2. 否则根据字段逐个推断，未知字段用保守默认值
    """
    name = (hw_info.get("name", "") or "").lower()
    vendor = (hw_info.get("vendor", "") or "").lower()
    cc = str(hw_info.get("compute_capability", "") or "")
    device = str(hw_info.get("device", "") or "")

    # 尝试匹配已知硬件 (键名长者优先, 避免 "h20" 抢在 "h200" 之前命中)
    for key in sorted(_KNOWN_HARDWARE, key=len, reverse=True):
        if key in name or key in vendor or key in device:
            caps = _KNOWN_HARDWARE[key]
            # 用 hw_info 中的动态数据覆盖预定义值
            caps.sm_count = hw_info.get("multiprocessor_count", caps.sm_count) or caps.sm_count
            return caps

    # 未知硬件：根据字段尽力推断
    caps = HardwareCaps(
        vendor=vendor or "unknown",
        device_name=name or "unknown",
        compute_capability=cc,
        sm_count=hw_info.get("multiprocessor_count", 0) or 0,
    )

    # 根据 device 类型推断
    if "cuda" in device or vendor == "nvidia":
        caps.vendor = "nvidia"
        caps.supported_dsls = {"triton", "cuda_c", "python"}
        caps.has_l2_cache_hint = cc.startswith("8.") or cc.startswith("9.")
        caps.has_software_pipelining = cc.startswith("8.") or cc.startswith("9.")
        caps.has_warp_shuffle = True
        caps.sub_group_size = 32
        caps.vector_width_bytes = 16
        caps.has_tensor_core = not cc.startswith("6.") and not cc.startswith("5.")
    elif "cpu" in device:
        caps.vendor = "cpu"
        caps.has_tensor_core = False
        caps.has_l2_cache_hint = False
        caps.has_software_pipelining = False
        caps.has_warp_shuffle = False
        caps.supported_dsls = {"python"}

    return caps


def get_caps_for_target(target_hardware: str, dsl: str = "triton") -> HardwareCaps:
    """根据 target.hardware 字符串获取硬件能力 (用于 Planner 初始化时)"""
    th = (target_hardware or "").lower()
    for key, caps in _KNOWN_HARDWARE.items():
        if key in th:
            return caps
    # 通用 nvidia_gpu 回退到 A100 级别
    if "nvidia" in th:
        return _KNOWN_HARDWARE["a100"]
    return _KNOWN_HARDWARE["mock"]
