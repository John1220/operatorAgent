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

import copy
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


class UnknownHardware(KeyError):
    """查表未命中的目标硬件。

    继承 KeyError 是因为它本来就是查表未命中; 带 target 与 known_keys 是因为
    「认不出」这个事实要能被调用方原样转述出来, 而不是被某个具体型号的规格顶掉。
    """

    def __init__(self, target: str, known_keys):
        self.target = target
        self.known_keys = sorted(known_keys)
        super().__init__(
            f"认不出的目标硬件 {target!r}; 已知键名: {self.known_keys}")


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

    # ---- 测量口径 ----
    # **空闲 GPU 上 CUDA event 窗口吃进的 launch 延迟**, 用于判断 L3 数字是否被
    # launch 支配 (缺口 1)。空 Triton 核, 25 预热 + 200 次取 median。
    #
    # ⚠️ 这不是「launch 的固有开销」, 两者差 4 倍: 真机标定发现同一个空核在
    # GPU 被填满时 (Triton do_bench 式的 256MB L2 冲刷排在 start.record() 之前)
    # 只测得 4.64 µs, 而 GPU 空闲时测得 20.03 µs —— 差的那 15 µs 是 GPU 空等
    # 命令到达的时间, 被事件窗口计了进去。见 docs/research/32 第二节。
    #
    # runner 现在按冲刷口径报**净执行时间**, 所以 measurement.launch_dominance
    # 的分母必须是 floor+net 而不是 net, 否则会算出 >1 的「占比」。
    #
    # None = 该设备未实测。measurement.assess() 拿到 None 会跳过这项检查并在
    # metrics 里写明「未做检查」, 不会默认放行 —— 缺证据不等于有效。
    launch_floor_us: Optional[float] = None

    # **实测峰值** —— `probe_peak` 拿一个已知好 GEMM 打到真实峰值的结果 (设计 §13.5)。
    #
    # None = 该设备**从未探测过**, 于是 `roofline_converged` 那个收敛出口一律不开:
    # 收敛要说的是「这张卡**这一刻**贴着上界了」, 而分母只能来自实测。表里的规格行
    # (哪怕口径核过) 不是实测值 —— 它是 datasheet 推算, 与「这次测量的上限」之间隔了
    # 时钟/温度/占用率。设计 §13.5 的原话是「`probe_peak` 必须实测 … 探测失败 →
    # roofline 收敛出口**永久关闭**」。
    #
    # 为什么另立一个字段而不是把表里的数字调小: 「口径核过」与「实测过」是**两个**问题
    # (见 `verifier._PEAK_SOURCE` 上方那段)。混成一个标志就会出现实测抓到的那种情形 ——
    # H20 那行的出处自己写着「无公开官方页 + 算术推导」, 却被当成可信分母**开了**收敛出口。
    measured_peak_tflops: Optional[float] = None

    # ---- 执行模型 ----
    sm_count: int = 0               # SM/CU/计算单元数量
    max_threads_per_sm: int = 2048  # 每 SM 最大线程数
    max_blocks_per_sm: int = 32     # 每 SM 最大 block 数
    max_warps_per_sm: int = 64      # 每 SM 最大 warp 数
    # 每 SM 的寄存器文件总量 (不是每线程上限, 那是 registers_per_thread)。
    # persistent kernel 按占用率算 grid 大小时要用: occupancy =
    # registers_per_sm // (n_regs * sub_group_size * num_warps)。
    # NVIDIA Volta 以后一律 64K 个 32-bit 寄存器/SM (cudaDevAttrMaxRegistersPerMultiprocessor)。
    registers_per_sm: int = 65536

    # ---- Hopper+ 数据搬运与调度特性 ----
    # 这三个字段是 R-MATMUL-13 / R-MATMUL-14 / R-UNIV-05 的 requires_caps 判据。
    # 不设字段时 getattr(caps, "has_tma", False) 恒为 False, 配方永远选不出来 ——
    # 与 R-MATMUL-07 曾经那个恒假的 registers_per_thread>=255 是同一类死条件。
    has_tma: bool = False               # Tensor Memory Accelerator (cc >= 9.0)
    has_warp_specialization: bool = False  # Hopper autoWS / Blackwell WS (cc >= 9.0)
    has_pdl: bool = False               # Programmatic Dependent Launch (cc >= 9.0)
    has_fp8_tensor_core: bool = False   # e4m3/e5m2 Tensor Core (cc >= 8.9)

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
    # ⚠️ 全部为 **dense** 规格。原先这行抄的是 NVIDIA 规格页的大字号数字, 那几个是
    # 2:4 sparse 的 (fp16 1979 / tf32 989 / int8 3958), 页脚小字才写 "With sparsity"。
    # dense = 其一半。这是 docs/research/00 缺口 3 里同一个错误在第二处的副本 ——
    # 它会让 fp16 核的 roofline 达成率被砍半, 「已到上界」的出口在这类卡上永不打开。
    "h100": HardwareCaps(
        vendor="nvidia", device_name="H100",
        compute_capability="9.0",
        fp32_tflops=67.0, tf32_tflops=494.7, fp16_tflops=989.4, bf16_tflops=989.4,
        int8_tops=1979.0,
        has_tensor_core=True, tensor_core_version=4, tensor_core_m_k_n=(16, 16, 16),
        vector_width_bytes=16, sub_group_size=32,
        global_mem_bandwidth_gbs=3350.0, l2_cache_size_kb=51200,
        shared_mem_per_sm_kb=228, shared_mem_bank_count=32,
        registers_per_thread=255,
        sm_count=132, max_threads_per_sm=2048,
        has_l2_cache_hint=True, has_software_pipelining=True,
        has_double_buffering=True, has_warp_shuffle=True,
        has_tma=True, has_warp_specialization=True, has_pdl=True,
        has_fp8_tensor_core=True,
        supported_dsls={"triton", "cuda_c", "python"},
    ),
    # NVIDIA H20 (Hopper 削减版) - dense 规格, 与 verifier._PEAK_BY_NAME["h20"] 同一口径
    # 与 H100 同为 cc 9.0, 但算力砍到 fp16 148 TFLOPS 而带宽保留 4.0 TB/s
    # → ridge point ≈ 37.0 FLOP/B (H100 ≈ 295), 优化取向相反, 勿套用 H100 配方
    #
    # 这一行以前填的是 2026-09-16 在 algorithm-h20 上 cuBLAS 8192³ + triad **实测**的
    # (31.6, 71.6, 138.2, 138.3, 3680)。测得没错, **用途错了**: 一次库实现的读数是
    # 「已达成的成绩」, 而 HardwareCaps 的字段名读起来像规格 —— 下游一旦拿它做分母就会
    # 出现「达成率 >100%」(真机 4096³ 跑到 140.0 TFLOPS)。实测阶梯 (fp16 跨 shape
    # 116.3~140.3、D2D 带宽随 footprint 1196~3357 GB/s) 见 benchmarks/
    # calibrate_achievable_peak.py 与 docs/research/42, 那是「这台卡达得到什么」,
    # 不是「这台卡的上界在哪里」。
    "h20": HardwareCaps(
        vendor="nvidia", device_name="H20",
        compute_capability="9.0",
        fp32_tflops=39.5, tf32_tflops=74.0, fp16_tflops=148.0, bf16_tflops=148.0,
        has_tensor_core=True, tensor_core_version=4, tensor_core_m_k_n=(16, 16, 16),
        vector_width_bytes=16, sub_group_size=32,
        global_mem_bandwidth_gbs=4000.0, l2_cache_size_kb=61440,
        # 2026-09-24 标定 (空 Triton 核, CUDA event, 25 预热 + 200 取中位):
        # median=20.24 min=18.69。2026-09-25 复测 median=20.03 min=18.30, 差 ~1%。
        # ⚠️ 同一个空核在 L2 冲刷口径下只有 4.64 µs —— 见字段注释与 docs/research/32。
        launch_floor_us=20.24,
        shared_mem_per_sm_kb=228, shared_mem_bank_count=32,
        registers_per_thread=255,
        sm_count=78, max_threads_per_sm=2048,
        has_l2_cache_hint=True, has_software_pipelining=True,
        has_double_buffering=True, has_block_reduce=True,
        has_warp_shuffle=True, has_atomic_fadd=True,
        # Hopper cc9.0: TMA / autoWS / PDL / FP8 全部具备。
        # ⚠️ autoWS 有极窄的静默失效门槛, 见 R-MATMUL-14。
        has_tma=True, has_warp_specialization=True, has_pdl=True,
        has_fp8_tensor_core=True,
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


def _nvidia_cc_major(cc: str) -> int:
    """把 compute_capability 字符串解析成主版本号, 解析不出来返回 0。

    cc 在本模块里是字符串 ("9.0" / "gfx90a"), 直接拿去做数值比较会抛 TypeError,
    所以任何按架构代次推断能力的地方都先过这个函数。
    """
    head = (cc or "").split(".")[0]
    return int(head) if head.isdigit() else 0


def _match_known_key(*texts: str) -> Optional[str]:
    """在若干**已小写化**的串里找 _KNOWN_HARDWARE 的键, 返回命中的键名。

    detect_hardware_caps 与 get_caps_for_target 共用这一套策略 (此前一个按键名长度
    降序、一个按字典插入顺序)。两条规则各自挡一类事故:

      * 长者优先 —— 短键不许靠插入顺序抢在包含它的长键前面;
      * 键的前后都不许紧跟数字 —— 表里没有 h200 条目, 纯子串匹配会让 "nvidia h200"
        命中 "h20", 于是 H200 拿着 H20 的 4000 GB/s / 148 TFLOPS 当自己的规格,
        roofline 分母偏小 6.7 倍, 正是会提前判停的那个方向。型号尾部再接数字就是
        另一个型号; 尾部接字母 (MI250X / V100S / MockGPU) 仍按同族的能力表处理,
        那是本表一贯的粗粒度。
    """
    needles = [t for t in texts if t]
    if not needles:
        return None
    for key in sorted(_KNOWN_HARDWARE, key=len, reverse=True):
        token = re.compile(r"(?<![0-9])" + re.escape(key) + r"(?![0-9])")
        if any(token.search(t) for t in needles):
            return key
    return None


def unknown_caps(device_name: str = "") -> HardwareCaps:
    """显式标注为「认不出」的能力描述: 所有 has_* 为 False, 算力/带宽/SM 数为 0。

    给「拿不到硬件事实但流程要继续」的调用方用。dataclass 上默认为 True 的那几个
    has_* 在这里一并置 False, supported_dsls 置空: 未知设备不该拿到任何以硬件特性
    为前提的配方, 也不该宣称支持某套 DSL。
    """
    return HardwareCaps(
        vendor="unknown",
        device_name=device_name or "unknown",
        has_block_reduce=False,
        has_warp_shuffle=False,
        has_warp_primitives=False,
        has_shared_memory=False,
        has_atomic_fadd=False,
        has_fast_math=False,
        supported_dsls=set(),
    )


#: 「这台卡有多少个计算单元」在仓库里存在的**两种拼法**, 顺序即优先级。
#:
#: §3.4 第 4 处的形状: 远端 runner 的载荷用 torch 的字段名 `multiprocessor_count`,
#: 而 `Backend.get_hardware_info()` 的字典用聚合名 `compute_units`。这两个名字从来没有
#: 在一个地方登记过, 于是 `detect_hardware_caps` 只认前者 —— 走 `get_hardware_info()`
#: 这条路 (verifier 的 launch 地板、planner 的配方筛选) 时**探测到的 SM 数被静默丢掉**,
#: 沿用表里的静态值。H20 恰好 78=78 所以从没暴露; 一台与表不一致的卡会拿着错的 SM 数选配方。
SM_COUNT_KEYS = ("multiprocessor_count", "compute_units")


def sm_count_from(hw_info: dict) -> Optional[int]:
    """从两种已知拼法里取**探测到的**计算单元数; 取不到返回 None (不猜、不给默认值)"""
    for key in SM_COUNT_KEYS:
        value = hw_info.get(key)
        # `bool` 显式排除: `isinstance(True, int)` 为真, 一个写成 True 的 SM 数会当成 1
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def detect_hardware_caps(hw_info: dict) -> HardwareCaps:
    """
    从 backend.get_hardware_info() 返回的 dict 自动推断硬件能力。

    策略:
      1. 命中 `_KNOWN_HARDWARE` → 返回该条目的**副本**, 只把探测到的 SM 数覆盖上去
      2. 没命中 → 从 `unknown_caps()` 起步 (什么都不承诺), 只往上加**能从中立事实
         推出来**的字段; 两个分支都走不到时原样返回那个「不知道」

    ⚠️ 本函数**不碰** dataclass 的默认值。表里 8 个条目里有 5 个从没显式写过
    `has_warp_primitives`/`has_shared_memory`/`has_fast_math`, 它们对已知卡的承诺
    就是从默认值继承来的; 把默认改成 False 会静默改掉包括 H20 在内每一张卡的口径
    (哪些配方还选得上), 那不是本函数的改动范围。
    """
    name = (hw_info.get("name", "") or "").lower()
    vendor = (hw_info.get("vendor", "") or "").lower()
    cc = str(hw_info.get("compute_capability", "") or "")
    device = str(hw_info.get("device", "") or "")

    key = _match_known_key(name, vendor, device)
    if key is not None:
        # 必须是副本: 表里存的是模块级共享实例, 就地写 sm_count 等于把这一次探测到的
        # 数字留给之后所有调用方 (跨任务、跨设备都读到同一份污染值)。deepcopy 而不是
        # dataclasses.replace —— supported_dsls 是可变 set, 浅拷贝仍然共享同一个对象。
        caps = copy.deepcopy(_KNOWN_HARDWARE[key])
        # 用 hw_info 中的动态数据覆盖预定义值
        probed = sm_count_from(hw_info)
        if probed is not None:
            caps.sm_count = probed
        return caps

    # 未知硬件：从「什么都不承诺」起步, 往上只加能从手头的中立事实推出来的东西。
    #
    # 这里原本直接是 `HardwareCaps(...)`, 也就是吃 dataclass 的宽松默认: 一张
    # vendor / device / cc **三者皆空**的卡会拿到 has_block_reduce / has_warp_shuffle /
    # has_warp_primitives / has_shared_memory / has_atomic_fadd / has_fast_math 全 True,
    # 外加 supported_dsls={"cuda_c","python"} —— 于是「认不出这台卡」被下游读成
    # 「这是一台支持 CUDA C、有共享内存和 warp shuffle 的类 NVIDIA 设备」。同一个模块里的
    # `unknown_caps()` 对这六个字段给的是 False、DSL 集合给空: **两个「不知道」的口径相反**,
    # 而且推断路径这个是乐观的那一个 —— 与 §3.3 那一族同型 (缺证据被当成肯定证据)。
    # 方向上的代价也不对称: 取 False 只是少一次优化机会, 取 True 是把配方发到一台
    # 可能根本没有共享内存的机器上, 失效方式是核编译/启动直接失败。
    caps = unknown_caps(device_name=name or "unknown")
    caps.vendor = vendor or "unknown"
    caps.compute_capability = cc
    caps.sm_count = sm_count_from(hw_info) or 0

    # 根据 device 类型推断
    if "cuda" in device or vendor == "nvidia":
        caps.vendor = "nvidia"
        # 这一行的**依据**是「已知这是 CUDA 设备」, 不是默认值: 下面这些是 PTX ISA 在
        # 整个受支持架构区间都保证的东西 (共享内存、warp 级原语与 shuffle、块内同步
        # 归约、float 原子加、MUFU 近似超越函数), 所以可以声明; 而上一段里那六个
        # 「认不出」的字段一个都不声明。两者是不同的判断, 不是同一份默认值的两种写法。
        caps.supported_dsls = {"triton", "cuda_c", "python"}
        caps.has_block_reduce = True
        caps.has_warp_shuffle = True
        caps.has_warp_primitives = True
        caps.has_shared_memory = True
        caps.has_atomic_fadd = True
        caps.has_fast_math = True
        major = _nvidia_cc_major(cc)
        caps.has_l2_cache_hint = major >= 8
        caps.has_software_pipelining = major >= 8
        caps.sub_group_size = 32
        caps.vector_width_bytes = 16
        caps.has_tensor_core = major >= 7
        # 推断路径一律取保守值: 误报 True 会把配方塞给不具备该能力的硬件 (核直接
        # 编译/启动失败), 误报 False 只是少一次优化机会。所以 Hopper 特性按
        # major >= 9 一刀切, Ada(8.9)/Blackwell(10.x) 的 FP8 要靠 _KNOWN_HARDWARE
        # 显式建条目, 不在这里猜。
        caps.has_tma = caps.has_warp_specialization = caps.has_pdl = major >= 9
        caps.has_fp8_tensor_core = major >= 9
    elif "cpu" in device:
        caps.vendor = "cpu"
        caps.has_tensor_core = False
        caps.has_l2_cache_hint = False
        caps.has_software_pipelining = False
        caps.has_warp_shuffle = False
        # 那六个 has_* 保持 unknown_caps() 给的 False: CPU 没有 GPU 意义上的共享内存、
        # warp 与块归约。旧写法在这里**只**显式否认了 warp shuffle, 其余全靠默认值替它
        # 答应 —— 于是同一个分支一边说「这不是 GPU」一边宣称 has_shared_memory=True。
        caps.supported_dsls = {"python"}

    return caps


def get_caps_for_target(target_hardware: str, dsl: str = "triton") -> HardwareCaps:
    """根据 target.hardware 字符串获取硬件能力 (用于 Planner 初始化时)

    认不出来就抛 UnknownHardware。这里曾经按字样兜底: 含 "nvidia" 的一律返回 A100 的
    完整能力表 (156 TF32 / 312 FP16 / 2039 GB/s / 108 SM), 其余一律返回 2 个 SM 的
    mock —— 两个兜底都把「不知道这台卡是什么」翻译成了一个像样的规格, 调用方无从
    分辨。宁可让调用方拿到一个显式的未知 (unknown_caps), 也不给它一个能当分母用的数。
    """
    th = (target_hardware or "").lower()
    key = _match_known_key(th)
    if key is None:
        raise UnknownHardware(target_hardware, _KNOWN_HARDWARE)
    return copy.deepcopy(_KNOWN_HARDWARE[key])
