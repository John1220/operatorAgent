"""
验证层 - 三级验证体系
L1: 编译验证 (代码可编译、语法正确)
L2: 功能验证 (输出正确，误差在容许范围内)
L3: 性能验证 (性能达到预期加速比) + Roofline 分析

本模块是**消费方**: 只认 `backend_caps.Backend` 与它声明的 `Cap`, 不 import 任何
具体提供方、也不做 `isinstance` 分派 (doc 49 §4.1 第 3 条)。
"""

import time
import dataclasses
import copy
import hashlib
import re
import numpy as np
from typing import Mapping, Optional
from ..core.schemas import (
    VerificationResult, VerificationLevel, TaskSchema, Precision,
)
from ..decide.anticheat import static_audit
from ..core.contract import case_key
from ..decide.correctness import as_suite, verdict_of
from ..decide.correctness_boundaries import has_boundary_policy, lint_coverage
from ..decide.correctness_profiles import profile_check
from ..decide.reference_contract import as_reference, validate_receipt
from ..decide.bench_cases import BenchCaseError
from ..decide.input_fidelity import (InputFidelityError, MAX_SEED_STORAGE_BYTES,
                             describe_inputs, check_readback)
from ..core.source_lang import classify
from ..decide.compile_evidence import validate_compile_failure


def _reference_array(value, dtype):
    """Built-in references consume logical values, not storage-indexing tricks.

    torch.from_numpy cannot represent negative strides and warns about readonly
    views. Copy ONLY on the reference side; candidate inputs stay untouched.
    """
    array = np.asarray(value, dtype=dtype)
    if not array.flags.writeable or any(s < 0 for s in array.strides):
        array = array.copy(order="K")
    return array


def _l1_guard_metrics(lang) -> dict:
    """把静态守卫的**适用性**搬进 metrics (CHECKLOG I65)。

    这一段是**搬运**不是判定 (判据在 `source_lang.classify`): 产物里要能读到「两道静态
    守卫对这份源码适用吗」, 而不是只有一个空的违规列表。三态照记 —— 「无法确立」与
    「不适用」与「适用」在产物里必须分得开 (资格层读的就是这个值)。
    """
    if lang is None:
        return {}
    return {"static_guards": lang.applicable,
            "static_guards_lang": lang.as_dict()}
from ..core.interface_contract import check_kernel_interface, input_key_sets
from ..core import tolerance
from ..decide import measurement
from ..decide.measurement import (GATE_PARAM_KEYS, PERF_CRITERION_ABSOLUTE_GFLOPS,
                          PERF_CRITERION_SPEEDUP, TRANSPORT_KEY, TRANSPORT_UNKNOWN, TRANSPORT_ANSWERED, aggregate_l2_readings,
                          aggregate_note, single_ruler, transport_outcome_of_result)
from ..core.hardware_caps import (
    detect_hardware_caps, device_key_of, measured_peak_record,
    measured_peaks_for, remember_reference_lower_bound, spec_row,
)
from ..decide.peak_probe import precision_key_of, probe_peak, sample_from_l3_metrics
from ..execute.placement import lease_evidence, observed_lease
from ..execute.placement import (observed_pairing, observed_windows,
                        SCOPE_SAME, SCOPE_CROSS, SCOPE_UNOBSERVED,
                        WINDOW_DIFFERENT)
from ..core.backend_caps import Backend, Cap, CapabilityMissing
from ..execute.exec_guard import GuardRefusal


#: 编译产物的**事实**（远端 runner 报了才带上，缺失 ≠ 0 ≠ 没有上限）。
#: 名单抽成模块级常量有两个读者, 各写一遍就会各自漂:
#:   * 本模块 L3 判据要的那几个（下面 metrics 的组装，§10.3 条件 2 的真源）；
#:   * §21.3 的「相关 profiler 指标」（`agent_runtime` 的上下文里给模型看的那一段）。
COMPILE_FACT_KEYS: tuple = ("num_warps", "num_stages", "n_regs", "n_spills",
                            "n_max_threads", "shared_bytes_per_cta")


#: 「在这台**执行设备**上, torch 高层 API 是激活算力的正当路径」—— 反作弊静态审计
#: (`anticheat.static_audit`) 的白名单输入。为什么是一个常量而不是读任务字段:
#: 设计 §3.5 要求安全判据「不可被 Agent 或任务参数改写」, 而这里以前传的是
#: `task.target.hardware` —— 一个任务只要声明 `hardware="Ascend910"` 就能让静态审计
#: 零违规 (CHECKLOG I80 硬伤一; 白名单当时还是子串匹配, 含 `dc/ls/ft/rv` 这类两字母 token)。
#: ⚠️ 今天**没有生产者**: 没有任何后端声明这件事 (加一条 `Cap` 必须同时给出声明者与读者,
#: 而真正的声明者还要把运行时屏蔽那一半一起接通 —— 真机 runner 的 `shield_torch()` 今天
#: 全是无参的) ⇒ 恒 False ⇒ 静态与运行时**两眼一致地拒绝**。厂商卡上「torch 合法」那条路
#: 要开通必须两半一起做, 记在 CHECKLOG I81。**不许**把它接到任务字段上。
VENDOR_TORCH_OK: bool = False


# 远端 L2 返回的指标里需要透传到控制端的键。matched_ratio / bad_elements / atol /
# rtol 是数值门改成「绝对+相对」之后新增的: 只留 max_error 的话, 控制端分不清
# 「差一点点」和「整体缩放了 3%」, 而后者正是 docs/research/12 要拦的那类。
#
# ⚠️ 这个 `if k in remote_verify` 的**逐键过滤**就是「某一组报了、某一组没报」那一侧的现场:
# 远端跑的是旧版 `run_kernel.py` 时缺哪个键, 控制端只能如实缺着。数值门口径的三键名与
# 「同值才配发布单值」那条规则都不在这里, 它们在 `measurement.GATE_PARAM_KEYS` /
# `single_ruler` —— 轮级 (这里) 与运行级 (`cross_round_ruler`) 两个读者必须读同一个集合。
_REMOTE_L2_METRIC_KEYS = (
    "max_error", "max_rel_error", "matched_ratio", "bad_elements",
    "atol", "rtol", "required_matched_ratio",
    "compared_elements", "coverage", "reference_type", "device",
    # 第一个坏元素的展平下标 (LeetCUDA book/tests 的 book_check() 同款)。
    # Fixer 拿它定位坏 tile, 比只有 max_error 有用一个量级 ——
    # 「哪错了」和「错了多少」是两个问题。
    "first_bad_index", "first_bad_actual", "first_bad_expected",
    # 误差**形态** (设计 §8.2): 「哪错了」的下一问是「该怎么修」。它由远端用**同一份**
    # `tolerance` 规则算出来 (与本地路径对称), 控制端只是把它带过来 —— 忘了放进这张
    # 白名单, 真机路径上它就会静默消失 (本地用例照样全绿, 那正是最坏的一种缺法)。
    "error_shape", "reference_nonfinite", "reference_error", "output_digest",
    "reference_receipt", "output_dtype", "candidate_semantics_error", "reference_identity_unknown",
    "input_transfer",
)

# 常见 GPU 硬件 **dense 规格峰值** (FP32/TF32/FP16/BF16 Tensor Core + 显存带宽)
# 格式: (FP32_TFLOPS, TF32_TFLOPS, FP16_TFLOPS, BF16_TFLOPS, memory_bandwidth_GBs)
# 匹配顺序: 先设备名 (更具体), 未命中再回退 compute_capability。
# 不可反过来 —— cc 9.0 同时对应 H100/H200/H20, 三者算力带宽差异极大。
#
# ⚠️ **这一张表回答的是「离理论上限还有多远」, 所以只能放 dense 规格, 不能放实测值。**
# 原先 h20 那行写的是 138.2/138.3/71.6/31.6/3680, 注释老实标着「实测 (cuBLAS 8192³)」——
# 出处没错, **用途错了**: 一次库实现的读数是「已达成的成绩」, 拿它当上界就会出现
# 「达成率 101.5%」(真机 4096³ 跑到 140.0 TFLOPS), 而 D4 的天花板出口正是拿达成率判停的。
# 实测值仍然有用, 但它不进这张表、也不进 roofline 分母 —— 理由见 _PEAK_BY_CC 之前那段。
# 每一行的口径记在 _PEAK_SOURCE 里, 并随 Roofline 日志一起打出来 —— 未经核实的行
# 会带着「口径不明」出现在每一轮验证的输出里, 而不是安静地假装是权威。
_PEAK_BY_NAME: dict[str, tuple[float, float, float, float, float]] = {
    # 与 hardware_caps 共有的卡: **数值从那边派生 (spec_row), 这里不再写第二遍**。
    # 此前两边各写一遍 (5 张卡 × 5 个数逐位相同, 全靠手抄维持一致), 而 hardware_caps
    # 自己的注释就写着「与 verifier._PEAK_BY_NAME["h20"] 同一口径」—— 一句没有守卫的
    # 口头约定: 只改一处, 同一台机器的 roofline 分母与 caps 带宽就悄悄不同源了。
    # ⚠️ 口径与出处仍归这边: 每行照样要在 _PEAK_SOURCE 里写明凭什么, 派生不改变这件事。
    # NVIDIA 官方规格页 (dense; 大字号那几个是 2:4 sparse, 差整 2 倍) + 与
    # flash-attention benchmarks/benchmark_attn.py 的 BF16 dense 表交叉核对
    "a100": spec_row("a100"),
    "h100": spec_row("h100"),
    # H20 没有公开的 NVIDIA 规格页 (两个官方 URL 均 404)。fp16/bf16/带宽取
    # flash-attention 硬编码的 dense 表 (148e12 / 4.0e12), 它自称来自官方规格并用
    # SM×tensor_core_clock×ops_per_cycle 交叉核对过; tf32 = bf16/2 是 Hopper 惯例
    # (H100 989.4 vs 494.7 就是这个比例); fp32 由 78 SM × 128 FMA × 2 × 1.98 GHz 推得。
    # 第三方规格页给 fp32 写 44 TFLOPS, 与上面的算术不自洽 (要 2.2 GHz), 故不用。
    # (这五个数现在落在 hardware_caps 的 h20 条目里, 由上面那个 spec_row 取回来。)
    "h20":  spec_row("h20"),
    "v100": spec_row("v100"),
    # ⚠️ 以下这些行**未核实口径** (很可能混了 2:4 sparse / FP16-with-FP32-accumulate)。
    # 保留数值不臆改, 但 _PEAK_SOURCE 会标「未核」并打进日志, 别拿它们的达成率做判停决定。
    # (原文写「以下四行」而实际列了五行, 顺手改成与事实相符的说法。)
    "3090": spec_row("3090"),
    # hardware_caps 里**没有**下面这几张卡的条目: 补一条要连 launch 地板 / smem / 能力位
    # 一起补齐, 那是实测活, 不臆造 (docs/research/49 §八 第 5 条)。所以数值只能留在这里。
    "h200": (67.0,  494.7, 989.4, 989.4, 4800.0),
    "4090": (82.6,  82.6,  330.0, 330.0, 1008.0),  # 4090 FP32=82.6 (no TF32 boost for Ada? actually Ada has TF32)
    "a6000":(38.7,  310.0, 77.4,  77.4,  768.0),
    "a5000":(27.8,  222.0, 55.5,  55.5,  768.0),
    "t4":   (8.1,   65.0,  130.0, 130.0, 320.0),
}

# 逐行可信度**不另立一张表**, 而是由下面的出处字符串自己承担: 以 ⚠️ 开头的行视为
# 「口径未核」, `measurement.reference_ceiling_reached` 据此拒绝开「已到 roofline」这个出口。
# 为什么默认关: 不对称在于分母偏大只是让出口不开 (白烧几轮, 还有停滞出口兜底),
# 分母偏小会把还能优化的核**提前**判成到顶。所以新加行时不写出处 = 不配拿来判停,
# 这个默认值就是要逼着写。
#
# 每行的出处。日志里跟着 peak 一起打印, 让「这个数字凭什么」在使用点就可见。
_PEAK_SOURCE: dict[str, str] = {
    "a100": "NVIDIA 规格 (dense)",
    "h100": "NVIDIA 规格 (dense; 原行用了 sparse 的 1979)",
    "h200": "NVIDIA 规格 (dense), 带宽 4.8 TB/s",
    "h20": "flash-attention dense 表 148/4.0 + 算术推导; 无公开官方页",
    "v100": "NVIDIA 规格 (dense)",
    "3090": "⚠️口径未核 (疑含 2:4 sparse)",
    "4090": "⚠️口径未核 (疑含 2:4 sparse)",
    "a6000": "⚠️口径未核",
    "a5000": "⚠️口径未核",
    "t4": "⚠️口径未核",
}
_PEAK_SOURCE_DEFAULT = "⚠️出处未记录"

# 「这台机器实际达得到什么」不进这张表, 也不立第二张表。
# 因为它不是一个标量: 2026-09-26 用 benchmarks/calibration/calibrate_achievable_peak.py 在 H20 GPU2
# 上量到的阶梯是 fp16 从 2048² 的 116.3 TFLOPS 爬到 6144² 的 140.3 (跨 20.7%), D2D 带宽
# 从 8 MB 的 1196 GB/s 爬到 4 GB 的 3357 GB/s (跨 2.8 倍)。按 shape 桶查它需要知道当前
# 张量落在哪个桶 —— 那件事的正确归宿是持久化的每-shape 记录 (缺口 2), 而不是又一堆
# 硬编码常量。数据本身留在 benchmarks/ 与 docs/research/42, 不在这里留没人读的副本。
_PEAK_BY_CC: dict[str, tuple[float, float, float, float, float]] = {
    # 回退表: 只有设备名没命中 _PEAK_BY_NAME 时才会用到。
    # 原先 "8.9"/"9.0" 两行都填着 H100 的数字, 而 fp16 那一列用的是 **2:4 sparse** 的
    # 1979 (被删掉的那行注释自己写着 "FP8=1979") —— 表里其余每行都满足 fp16 == bf16,
    # 只有它们差 2.0 倍, 列错位不需要外部资料也能看出来。后果是有后果的: fp16 核的
    # 达成率被这个分母砍半, D4 的天花板出口在这类卡上永不打开。
    # 两个死键 "9.0a"/"8.9-bw" 已删 (匹配只用 `cc in 表` 与 `cc.startswith(键)`, 而 cc
    # 来自 f"{major}.{minor}", 永远不会以 "…a" 结尾)。
    #
    # ⚠️ 分母偏大 vs 偏小的不对称: 偏大只是让判停出口不开 (白烧轮次), 偏小会**提前停**
    # (把还能优化的核判成已到顶)。所以拿不准时一律取较大的那个口径。
    "8.0":  spec_row("a100"),                      # A100 (80GB, HBM2e)
    "8.6":  (19.5,  156.0, 312.0, 312.0, 1555.0),  # A100 (40GB), A40
    # 8.9 是 Ada 而不是 H100 (hardware_caps 自己写着 h100 的 cc 是 9.0)。按 L40S 填:
    # 362 是 **FP16 累加** 的 dense 率; 用 FP32 累加时上限其实是 181。取 362 是因为
    # 上面那条不对称 —— 偏大只是不出口, 偏小会提前停。
    "8.9":  (91.3,  181.0, 362.0, 362.0, 864.0),
    "9.0":  spec_row("h100"),                      # H100 SXM dense (cc 9.0 也含 H20/H200)
    "7.5":  spec_row("3090"),                      # RTX 3090 / A40 (⚠️口径未核)
    "7.0":  spec_row("v100"),                      # V100 (Volta, no TF32)
}

_PEAK_CC_SOURCE: dict[str, str] = {
    "8.0": "A100 80GB NVIDIA 规格 (dense)",
    "8.6": "A100 40GB NVIDIA 规格 (dense); 同 cc 的 A40 会偏高 (真机 149/696), 只会让出口不开",
    "8.9": "Ada/L40S dense (362 是 FP16 累加口径; 同 cc 的 4090/A10 会更偏高)",
    "9.0": "H100 SXM dense; 只有设备名能把它和 H20/H200 分开",
    "7.0": "V100 NVIDIA 规格 (Volta 无 TF32)",
    "7.5": "⚠️口径未核",
}


def syntax_error_report(source: str, err: SyntaxError, context: int = 2) -> str:
    """把 SyntaxError 摊成「哪一行 / 编译器看到什么 / 它在哪一列停下」。

    `str(SyntaxError)` 只有 `invalid syntax (<file>, line N)` 一行, **不含出错源码与
    插入符** —— 那是 traceback 模块额外加上去的。Verifier 以前存的就是 str(e), 于是
    Fixer 拿到「第 4 行语法错」却从来看不到第 4 行长什么样, 只能凭猜改。

    LLM 修复回路里最贵的不是信息不准, 而是信息不全: 一次猜错就是一次完整的
    生成 + 远程编译往返。
    """
    lines = (source or "").splitlines()
    n = err.lineno or 0
    start = max(1, n - context)
    end = min(len(lines), n + context)
    out = [f"{type(err).__name__}: {err.msg}",
           f"位置: 第 {n} 行" + (f", 第 {err.offset} 列" if err.offset else "")]
    if n:
        out.append("  编译器看到的源码 (>>> 标出出错那一行):")
        for i in range(start, end + 1):
            marker = ">>>" if i == n else "   "
            out.append(f"  {marker} {i:4d} | {lines[i - 1]}")
        if err.offset:
            # 插入符要对齐到文本列: 前缀 "       i | " 的长度必须一并补偿
            gutter = len(f"   {n:4d} | ")
            out.append("        " + " " * (gutter + max(0, err.offset - 1)) + "^")
    if err.text and err.text.strip() and not n:
        out.append(f"  出错行内容: {err.text.rstrip()}")
    end_lineno = getattr(err, "end_lineno", None)
    if end_lineno and n and end_lineno != n:
        out.append(f"  该语法结构从第 {n} 行一直延伸到第 {end_lineno} 行 "
                   f"(通常意味着中间少了闭合符号, 而不是这一行本身写错)")
    return "\n".join(out)


def failure_pair(err) -> tuple[str, str]:
    """把执行侧错误摊成 (单行现象, 全文日志)。

    message 只留首行是因为 `_format_failures` 会把 message 与 error_log 都写进 Fixer
    prompt: 全文放两份 = 双份上下文, 而 `_truncate_error` 只压得住 error_log 那一份。
    远程 Triton `CompilationError` 会把整个核的源码回显一遍, 一份就上千字符。
    """
    text = str(err or "")
    lines = text.strip().splitlines()
    return (lines[0] if lines else "无错误详情"), text


def baseline_plan(kernel_type: str, precision: str) -> list:
    """下发给远端的基线**有序偏好**。

    为什么以名单而不是结论的形式下发: 选谁当基线是公平性策略 (fp16 候选就该比 cuBLAS
    fp16, 否则是拿 fp16 比 fp32), 这个判断控制端做得; 但某个基线在这台机器上跑不跑得
    起来 (`torch.compile` 失败、显存不够、非 CUDA 设备没有 TF32) 只有远端知道。所以
    控制端只排序, 远端取第一个能跑的并回报 `baseline_selected`。

    名单里出现远端没有的项是安全的 —— 远端按 `name in specs` 过滤。
    """
    k = (kernel_type or "").lower()
    is_matmul = "matmul" in k or "gemm" in k
    is_fp16 = (precision or "").lower() == "fp16"
    if is_matmul and is_fp16:
        return ["pytorch_eager_fp16", "pytorch_eager_tf32", "pytorch_eager", "torch_compile"]
    if is_matmul:
        return ["pytorch_eager_tf32", "pytorch_eager", "torch_compile"]
    return ["pytorch_eager", "torch_compile"]


def _bad_mask_as_2d(mask):
    """把坏点掩码看成二维 (除最后一维外全部折叠) ⇒ `(二维掩码, rows, cols)`; 判不了 ⇒ None

    为什么要折叠: 设计 §8.2 的形态判据问的是"坏点是不是集中在某几行/某几列", 而输出可能
    是 (M,N) 也可能是 (B,H,S,D) —— 折叠成 (rows, cols) 之后同一套规则管所有 rank。
    一维输出视作 `(1, N)` (整体当一行): 它没有"行"可谈, "整行坏"自然落不到它头上。
    """
    a = np.asarray(mask)
    if a.ndim == 0 or a.size == 0:
        return None
    if a.ndim == 1:
        return a.reshape(1, -1), 1, int(a.size)
    cols = int(a.shape[-1])
    if cols <= 0:
        return None
    return a.reshape(-1, cols), int(a.size // cols), cols


#: 一个 case 因「传输结果未知」被重发的次数上限。它只管**已确认未知**之后的补发，
#: 不改变「在飞请求不重试」那条规矩；上限 1 是为了不让一次抖动变成无限退避循环。
_TRANSPORT_RETRIES = 1


class Verifier:
    """
    验证器 - 执行三级验证
    L1 Compile -> L2 Function -> L3 Performance + Roofline
    """

    def __init__(self, backend: Backend, *, cuda_ruler_profile=None):
        self.backend = backend
        from ..decide.cuda_ruler_profile import validate_profile
        self._cuda_ruler_profile = (validate_profile(cuda_ruler_profile)
                                   if cuda_ruler_profile is not None else None)
        self.name = "Verifier"
        # 能力**快照**: 门面/提供方的 `capabilities` 可能是延迟解析的 property,
        # 每次分支都去读它会让「同一轮验证内判据变了」这种事在原理上可能发生。
        self._caps = frozenset(backend.capabilities)
        # 编译缓存: sha256(code) -> 通过标记, 相同代码重复验证时跳过编译
        self._compile_cache: dict[str, bool] = {}
        # 输入「能不能按 seed 物化」的判定缓存: case 身份 -> spec | None
        # (2026-10-20: L2 的输入不再整份传字节; 判据见 `_seed_spec_for`)
        self._seed_spec_cache: dict = {}
        # baseline 缓存: (算子, 输入签名) -> 耗时, 避免多轮验证重复测量
        self._baseline_cache: dict = {}
        # 硬件理论峰值缓存 (延迟探测)
        self._theoretical_peak: Optional[dict] = None
        self._reference_bindings = {}
        self.on_level_complete = None

    def _tag_evidence(self, result):
        if Cap.TEST_ONLY in self._caps:
            result.metrics = dict(result.metrics or {}, test_only=True,
                                  evidence_kind='test_only', gpu_acceptance=False,
                                  performance_evidence=False, backend=self.backend.label)
        return result

    def _timed_level(self, fn, source, audit_task, previous, *args, **kwargs):
        if self.on_level_complete is None:
            return self._tag_evidence(fn(*args, **kwargs))
        from ..execute.task_clock import monotonic
        started_utc, started = time.time(), monotonic()
        result = self._tag_evidence(fn(*args, **kwargs))
        timing = dict(started_at=started_utc, completed_at=time.time(),
                      started_monotonic=started, completed_monotonic=monotonic())
        result.metrics = dict(result.metrics or {}, execution_timing=timing)
        # An audit write error stops the audited call. It must not leave an
        # unaudited success that a campaign could count as complete.
        self.on_level_complete(source, audit_task, result, [*previous, result], timing)
        return result

    def verify(
        self,
        kernel_code: str,
        task: TaskSchema,
        test_inputs: dict,
        expected_output=None,
        target_gflops: float = 10.0,
        environment_drift: Optional[list] = None,
        stop_after: Optional["VerificationLevel"] = None,
        heartbeat=None,
        heartbeat_expected: bool = False,
    ) -> list[VerificationResult]:
        """执行完整三级验证

        `environment_drift`: 本次运行的环境指纹与上一次相比变了哪些项
        (`environment.compare_fingerprints` 的输出)。它是 L3 有效性判定的一个输入,
        所以必须由**拿着产物库的那一层** (Conductor) 传进来 —— Verifier 自己不知道
        「上一次」是谁。None = 没做比较, 与「比过了没变」(空列表) 是两件事。

        `stop_after`: 在该级**通过**后停止, 不再往下跑 (P1-1)。读者是 Conductor 的
        候选选择: 对多个决赛候选只跑到 L2 (正确性), 用真机结果定胜负; L3 只跑
        胜者, 机时不随候选数翻倍。None = 完整三级 (默认, 旧行为不变)。

        `heartbeat` / `heartbeat_expected`: 标定心跳 (设计 §10.5-1)。前者是**本窗口**
        的一对已知参照读数, 后者说「这次到期了」—— 到期而 `heartbeat is None` 时收口点
        按 RULER_BROKEN 处理 (spine-review P1-5: 「没做心跳」不许读成「心跳通过」)。
        两个都由**拿着测量计数的那一层** (Conductor) 传进来: Verifier 不知道这是第几次
        真值测量。
        """
        results = []

        preflight = self.correctness_preflight(task, test_inputs)
        if preflight is not None:
            return [self._tag_evidence(preflight)]  # Harness gap, not a candidate failure.
        preflight = self.reference_preflight(task)
        if preflight is not None:
            return [self._tag_evidence(preflight)]

        l1 = self._timed_level(self._verify_compile, kernel_code, task, results,
                              kernel_code, task.target.dsl, input_key_sets(test_inputs))
        results.append(l1)
        if not l1.passed:
            return results
        if stop_after is VerificationLevel.L1_COMPILE:
            return results

        l2 = self._timed_level(self._verify_function, kernel_code, task, results,
                              kernel_code, test_inputs, expected_output, task)
        results.append(l2)
        if not l2.passed:
            return results
        if stop_after is VerificationLevel.L2_FUNCTION:
            return results

        l3 = self._timed_level(self._verify_performance, kernel_code, task, results,
                                      kernel_code, test_inputs, target_gflops, task,
                                      environment_drift=environment_drift,
                                      heartbeat=heartbeat,
                                      heartbeat_expected=heartbeat_expected,
                                      static_guards=l1.metrics.get("static_guards"),
                                      # 走到这里 L2 必已通过 (上面 `if not l2.passed: return`)。
                                      # 仍然显式传: 「正确性是性能观测的前置条件」必须是**数据**,
                                      # 不是这条控制流的隐含性质 (§20.4)。
                                      correctness_verified=l2.passed)
        results.append(l3)
        return results

    def verify_l3_only(self, kernel_code: str, test_inputs, target_gflops: float,
                       task: TaskSchema, environment_drift=None,
                       heartbeat=None, heartbeat_expected: bool = False,
                       static_guards: Optional[bool] = None,
                       correctness_verified: Optional[bool] = None,
                       ) -> list[VerificationResult]:
        """只跑 L3 (P1-1): 胜出候选已带 L1+L2 结果时的续接入口

        读者是 Conductor 的候选选择: 决赛候选在 `verify(stop_after=L2)` 里真机
        定过胜负后, 胜者用本方法补跑 L3 —— 真机 L2 不重复执行。返回单元素列表,
        由调用方与已有的 L1+L2 结果拼接成完整三级。

        `static_guards`: 胜者那份源码的静态守卫适用性, 从它的 L1 结果里读出来
        (`metrics["static_guards"]`)。为什么必须由调用方传: 它属于**那一份源码**, 而
        Verifier 一个实例会连着验多个候选 —— 存成实例字段就是「记录说 A、判定按 B」
        那个坑 (CHECKLOG I65)。None = 没观测 ⇒ 不改判。

        `correctness_verified`: 这份候选的**正确性凭据** (设计 §14-10 / §20.4)。本方法是
        **唯一能绕开 L2 产出 L3 观测的入口** —— 一个没验过正确性的候选在这里同样能拿到
        gflops 与资格位, 而那正是"错误候选不得产生可用于晋级的性能观测"的缺口。
        凭据必须由调用方按**它手里那份 L2 结果**给出 (`Conductor._correctness_credential`),
        不由本方法假设、也不看实例状态。三态: `None` = 没带凭据 (回放/脚本) ⇒ 不写键、
        不追溯改判; **`False` ⇒ 该读数全资格 False** (规则 3 那一族: 这个数没有指称对象)。
        """
        if task.constraints.correctness_profile is not None:
            preflight = self.correctness_preflight(task, test_inputs)
            if preflight is not None:
                return [self._tag_evidence(preflight)]
        preflight = self.reference_preflight(task)
        if preflight is not None:
            return [self._tag_evidence(preflight)]
        if task.constraints.reference_contract is not None and correctness_verified is not True:
            return [self._tag_evidence(self._reference_unknown(
                "v2 L3 必须携带已通过的 L2 凭据", VerificationLevel.L3_PERFORMANCE))]
        l3 = self._timed_level(self._verify_performance, kernel_code, task, [],
                                      kernel_code, test_inputs, target_gflops, task,
                                      environment_drift=environment_drift,
                                      heartbeat=heartbeat,
                                      heartbeat_expected=heartbeat_expected,
                                      static_guards=static_guards,
                                      correctness_verified=correctness_verified)
        return [l3]

    def _verify_compile(self, kernel_code: str, dsl: str = "python",
                        input_key_sets=None) -> VerificationResult:
        print("  [Verifier] L1: 编译验证...")
        # 静态守卫的适用性: 两道守卫 (接口契约 / 反作弊) 都是 Python 形状的, 对一份 C++/CUDA
        # 模板源它们「判不了」—— 而旧的返回值形状只有一个「有/没有问题」的通道, 于是
        # 「查不了」被读成「没问题」(fail-open)。判据只有一处 (`source_lang.classify`),
        # 结论随 L1 的 metrics 走, 由资格层收口 (CHECKLOG I65)。
        _guard_lang = classify(kernel_code, dsl)
        _guard_metrics = _l1_guard_metrics(_guard_lang)
        if _guard_lang.applicable is not True:
            print(f"  [Verifier] L1 静态守卫**不适用/无法确立**: {_guard_lang.reason}")
        # 本地语法预检: 截断/缩进/未终结字面量等 SyntaxError 直接判失败,
        # 避免把明知无效的代码发去远程编译 (省 SSH 往返, 错误更清晰)
        # cuda_c (PyTorch C++ Extension): 代码内含 C++ 源码字符串, Python compile() 可能
        # 因字符串内 C++ 特殊字符误判; C++ 编译由远程 load_inline 完成, 跳过本地预检
        if (dsl or "").lower() != "cuda_c":
            try:
                compile(kernel_code, "<l1_precheck>", "exec")
            except SyntaxError as e:
                msg = f"本地语法预检失败: {e.msg} (第 {e.lineno} 行)"
                print(f"  [Verifier] L1 FAILED: {msg}")
                return VerificationResult(
                    level=VerificationLevel.L1_COMPILE, passed=False,
                    message=f"编译失败: {msg}",
                    # ⚠️ str(SyntaxError) 只有 "invalid syntax (<..., line 4)" 这一行,
                    # **不含出错的源码内容与插入符** (那是 traceback 才加的)。此前
                    # error_log=str(e) 让 Fixer 拿到「第 4 行语法错」却从来看不到第 4
                    # 行长什么样, 只能猜。这里把 e.text/e.offset 连同上下文一起带上。
                    error_log=syntax_error_report(kernel_code, e),
                )
        # 接口契约静态检查 (缺口 6): 远端固定以 `kernel(**输入)` 全关键字方式调用, 签名
        # 对不上就是必然的 TypeError —— 但那要等一次 SSH 往返 + 远程 import + Triton JIT
        # 之后才现形, 于是一个纯静态可见的错误要花掉一整轮 Fixer (真实历史: attention
        # 连烧 3 轮在 `kernel() got an unexpected keyword argument 'causal'`)。
        # 放在缓存之前: 这是本地 AST 遍历, 微秒级, 而缓存键只看代码哈希 —— 同一份代码
        # 配不同的输入集, 契约结论可以不同, 不能让缓存把它吞掉。
        verdicts = [check_kernel_interface(kernel_code, keys, language=dsl)
                    for keys in (input_key_sets or [])]
        mismatch = next((v.problem for v in verdicts if v.problem), None)
        if mismatch:
            print("  [Verifier] L1 FAILED: 核函数接口与调用契约不符 (未动 GPU)")
            return VerificationResult(
                level=VerificationLevel.L1_COMPILE, passed=False,
                message="编译失败: " + mismatch.split("\n")[0],
                error_log=mismatch,
                metrics={"compile_status": "interface_mismatch",
                         "backend": str(self.backend),
                         **_guard_metrics},
            )
        # Source shape/dtype is not physical shared storage. Preserve an explicit
        # unknown report, then let the normal compiler decide; never turn a
        # skipped proxy into a compile pass.
        try:
            from ..decide.resource_check import check_resource_feasibility
            caps = getattr(self, "_l1_caps", "unset")
            if caps == "unset":
                try:
                    caps = detect_hardware_caps(self.backend.get_hardware_info())
                except Exception:  # noqa: BLE001
                    caps = None
                self._l1_caps = caps
            rep = check_resource_feasibility(kernel_code, caps, "")
            _guard_metrics['resource_precheck'] = rep.as_dict()
            if not rep.feasible:
                msg = rep.as_message()
                print(f"  [Verifier] L1 FAILED: {msg}")
                return VerificationResult(
                    level=VerificationLevel.L1_COMPILE, passed=False,
                    message=f"编译失败: {msg}", error_log=msg,
                    metrics={"compile_status": "resource_infeasible",
                             "backend": str(self.backend),
                             "estimated_smem_bytes": rep.estimated_smem_bytes,
                             "smem_limit_bytes": rep.limit_bytes, **_guard_metrics})
        except Exception as exc:  # noqa: BLE001 — fail unknown, never prove fit
            _guard_metrics['resource_precheck'] = dict(assessment='unknown',
                estimated_smem_bytes=None, evidence_kind=None,
                notes=[f'Resource precheck unavailable: {type(exc).__name__}'])
        import hashlib
        cache_key = hashlib.sha256(kernel_code.encode("utf-8")).hexdigest()
        if cache_key in self._compile_cache:
            print("  [Verifier] L1 PASSED: 编译缓存命中")
            return VerificationResult(level=VerificationLevel.L1_COMPILE, passed=True, message="编译验证通过 (缓存命中)", metrics={"compile_status": "cached", "backend": str(self.backend),
                             **_guard_metrics})
        try:
            self.backend.compile(kernel_code)
            self._compile_cache[cache_key] = True
            print("  [Verifier] L1 PASSED: 编译成功")
            return VerificationResult(level=VerificationLevel.L1_COMPILE, passed=True,
                                      message="编译验证通过",
                                      metrics={"compile_status": "success",
                                               "backend": str(self.backend),
                                               **_guard_metrics})
        except Exception as e:
            print(f"  [Verifier] L1 FAILED: {e}")
            # 远程编译错误全文可达上千字符 (gcc/Triton 会回显源码), 只把首行放进 message,
            # 否则 Fixer prompt 里同一份全文会出现两遍
            head, full = failure_pair(e)
            metrics = dict(_guard_metrics)
            if getattr(e, "session_state", None):
                metrics.update({TRANSPORT_KEY: TRANSPORT_UNKNOWN,
                                "session_state": e.session_state,
                                "session_phase": getattr(e, "session_phase", ""),
                                "compile_status": "transport_unknown"})
            # §22.2 的执行器**拦下**候选（编译超时 / 撞上限 / 盾命中 / 起不来）与「候选
            # 自己编译失败」必须在产物里分得开 —— 前者是本层新出现的一档（进程内 exec
            # 既没有墙钟也没有上限，它根本不会有），不写出来就成了「无名编译失败」。
            # 取值单独一族（`compile_*`）：`compile_status` 的既有消费者做的是**相等判据**
            # （`interface_mismatch` / `resource_infeasible` / `cached` / `success`），
            # 加一个新取值不会让它们误判；`benchmarks/verify/verify_corpus_honesty.py` 的
            # `origin_of()` 已同步认这一族是**本地**失败。
            if isinstance(e, GuardRefusal):
                metrics["compile_status"] = f"compile_{e.status}"
                metrics["exec_guard_status"] = e.status
                if e.kind:
                    metrics["exec_guard_kind"] = e.kind
                metrics["exec_guard_executor"] = e.executor
                if e.not_enforced:
                    metrics["exec_guard_not_enforced"] = list(e.not_enforced)
            return VerificationResult(level=VerificationLevel.L1_COMPILE, passed=False,
                                      message=f"编译失败: {head}", error_log=full,
                                      metrics=metrics)

    @staticmethod
    def correctness_preflight(task: TaskSchema, test_inputs=None) -> Optional[VerificationResult]:
        """Declared boundary gaps stop before compile, GPU probe or candidate exec.

        Legacy suites have no policy and keep their old gate/hash. In particular,
        not declaring a policy must not be reported as passing coverage lint.
        """
        profile = profile_check(task, test_inputs)
        if profile["status"] == "unknown":
            reason = "默认正确性配置无法确立: " + profile["reason"]
            return VerificationResult(VerificationLevel.L2_FUNCTION, False, message=reason,
                metrics={**({"reference_identity_unknown": True, "reference_error": reason}
                            if task.constraints.reference_contract is not None else {}),
                    "correctness": {"verdict": "unknown", "reason": reason,
                    "profile": profile, "scope": profile["scope"], "cases": []}})
        raw = task.constraints.correctness_suite
        if not has_boundary_policy(raw):
            return None
        suite = None
        try:
            suite = as_suite(raw)
            coverage = lint_coverage(suite)
        except (ValueError, TypeError) as exc:
            coverage = {"status": "invalid", "missing": [], "errors": [str(exc)],
                        "scope": "生成/声明未确立，未执行候选"}
        if coverage["status"] == "passed":
            return None
        reason = "边界生成/覆盖 lint 无法确立: " + "; ".join(coverage["errors"] + coverage["missing"])
        correctness = {"verdict": "unknown", "reason": reason, "coverage_lint": coverage,
                       "scope": coverage["scope"], "cases": []}
        if suite is not None:
            correctness.update(suite_hash=suite.digest(), declared_cases=len(suite.cases),
                cases=[{"case_id": c.case_id, "verdict": "not_run", "reason": "覆盖预检查未通过"}
                       for c in suite.cases])
        return VerificationResult(VerificationLevel.L2_FUNCTION, False,
                                  message=reason, metrics={"correctness": correctness})

    @staticmethod
    def _reference_unknown(reason, level=VerificationLevel.L2_FUNCTION, payload=None):
        metrics = dict(reference_identity_unknown=True, reference_error=reason,
                       correctness={"verdict": "unknown", "reason": reason, "cases": [],
                                    "scope": "参考身份/精度契约未确立，不执行/不重试候选"})
        if isinstance(payload, dict) and payload.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN:
            metrics.update({k: payload[k] for k in
                (TRANSPORT_KEY, "session_state", "session_phase", "request_id", "bytes_sent") if k in payload})
        if level == VerificationLevel.L3_PERFORMANCE:
            metrics.update(measurement_validity="uninformative", correctness_verified=False,
                           eligibility={k: False for k in ("can_verdict", "can_tune", "can_persist", "can_report", "can_count_progress")})
        return VerificationResult(level, False, message="参考身份无法确立: " + reason, metrics=metrics)

    def reference_preflight(self, task, *, start_run=False):
        raw = task.constraints.reference_contract
        bound = self._reference_bindings.get(task.task_id)
        if raw is None and bound is None:
            return None
        response = None
        try:
            from ..core.contract import freeze_from_task
            from ..decide.reference_contract import validate_receipt
            if raw is None:
                raise ValueError("冻结后的参考声明被移除，不退回 legacy")
            contract = freeze_from_task(task)
            if bound is not None and not start_run and bound != contract.hash:
                raise ValueError("生成前冻结的契约哈希已漂移")
            self._reference_bindings[task.task_id] = contract.hash
            self.backend.require(Cap.REAL_DEVICE, Cap.REMOTE_VERIFY, Cap.SAME_DEVICE_BASELINE,
                                 Cap.REFERENCE_IDENTITY, Cap.GPU_PINNING)
            response = self.backend.reference_identity(contract.reference_contract.content_dict())
            if not isinstance(response, dict) or response.get("success") is not True:
                raise ValueError(response.get("reference_error", "执行侧参考身份回执缺失")
                                 if isinstance(response, dict) else "执行侧参考身份回执缺失")
            validate_receipt(raw, response.get("reference_receipt"))
        except (ValueError, TypeError, RuntimeError, OSError, CapabilityMissing) as exc:
            return self._reference_unknown(str(exc), payload=response)
        return None

    def _verify_function(self, kernel_code: str, test_inputs, expected_output, task: TaskSchema) -> VerificationResult:
        """Suite checks are part of L2, before candidate selection or L3.

        Caller inputs cannot replace declared checks. Suite arrays stay local to
        this method, so performance ledgers/weights retain their original cases.
        """
        raw = task.constraints.correctness_suite
        preflight = self.correctness_preflight(task, test_inputs)
        if preflight is not None:
            return preflight
        preflight = self.reference_preflight(task)
        if preflight is not None:
            return preflight
        if raw is None:
            return self._verify_primary_function(kernel_code, test_inputs, expected_output, task)
        try:
            suite = as_suite(raw)
        except (ValueError, TypeError) as exc:
            return VerificationResult(
                level=VerificationLevel.L2_FUNCTION, passed=False,
                message=f"CorrectnessSuite 无法确立: 声明无效: {exc}",
                metrics={"correctness": {"verdict": "unknown", "reason": str(exc)}})

        try:
            primary = self._verify_primary_function(kernel_code, test_inputs, expected_output, task)
        except (ValueError, CapabilityMissing) as exc:
            primary = VerificationResult(
                level=VerificationLevel.L2_FUNCTION, passed=False,
                message=f"正确性检查无法执行: {exc}",
                metrics={"correctness_primary_verdict": "unknown"})
        rows = []
        primary_state = primary.metrics.get("correctness_primary_verdict")
        if primary_state is None:
            primary_state = verdict_of(primary)[0]
        if primary_state != "passed":
            rows = [{"case_id": c.case_id, "verdict": "not_run",
                     "reason": "调用方输入/静态守卫未通过，未运行套件"} for c in suite.cases]
            return self._suite_result(primary, suite, rows, primary_state, task)

        ordered_cases = sorted(suite.cases, key=lambda c: c.case_id)
        for index, case in enumerate(ordered_cases):
            row = {"case_id": case.case_id, "shape_key": case.inputs.key(),
                   "pattern": case.pattern, "layout": case.layout,
                   "dtypes": {t.name: t.dtype for t in case.inputs.tensors}}
            supported, reason = self.backend.correctness_input_support(kernel_code, case.layout)
            if not supported:
                rows.append(dict(row, verdict="unknown", reason=reason))
                continue
            dtypes = {t.dtype for t in case.inputs.tensors}
            if len(dtypes) != 1 or not dtypes <= {"fp16", "fp32"}:
                rows.append(dict(row, verdict="unknown",
                                 reason="当前执行协议仅确立同精度 fp16/fp32 输入；不降级 dtype"))
                continue
            case_task = copy.deepcopy(task)
            case_task.constraints.precision = Precision(next(iter(dtypes)))
            try:
                inputs = case.materialize()
                r = self._verify_single_set(kernel_code, inputs, None, case_task)
                state, reason = verdict_of(r)
                retried = 0
                while (state == "unknown"
                       and r.metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN
                       and retried < _TRANSPORT_RETRIES):
                    # 「不重试在飞请求」仍然成立：结果是**已确认的未知**之后才走到这里，
                    # 再发一次是**新请求**（新的预算预留、新的 request_id），不是重放在飞的那个。
                    # 没有这一步，一个 case 的瞬时抖动就让整份套件 unknown ⇒ 整格 hold
                    # （2026-11-04 matmul-4096-B：15 个 case 里 1 个 unknown，整格作废）。
                    retried += 1
                    r = self._verify_single_set(kernel_code, case.materialize(), None, case_task)
                    state, reason = verdict_of(r)
                    row["transport_retry"] = retried
                row.update(verdict=state, reason=reason, metrics=dict(r.metrics))
                if state == "passed" and suite.deterministic:
                    # Re-materialize: a candidate must not mutate its first input
                    # and thereby change the object used in the repeat check.
                    repeat = self._verify_single_set(kernel_code, case.materialize(), None, case_task)
                    repeated_state, repeated_reason = verdict_of(repeat)
                    first_hash = r.metrics.get("output_digest")
                    second_hash = repeat.metrics.get("output_digest")
                    row["repeat_metrics"] = dict(repeat.metrics)
                    if repeated_state != "passed":
                        row.update(verdict=repeated_state, reason=f"重复检查: {repeated_reason}")
                    elif not first_hash or not second_hash:
                        row.update(verdict="unknown", reason="未返回完整输出指纹，确定性无法确立")
                    elif first_hash != second_hash:
                        row.update(verdict="failed", reason="同输入两次执行输出不同（确定性失败）")
                    else:
                        row["determinism"] = "passed"
            except (BenchCaseError, ValueError, CapabilityMissing) as exc:
                row.update(verdict="unknown", reason=f"检查无法执行: {exc}")
            rows.append(row)
            unknown_transport = next((m for m in (row.get("metrics", {}), row.get("repeat_metrics", {}))
                                      if m.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN or m.get("reference_identity_unknown")), None)
            if unknown_transport is not None:
                primary.metrics.update({k: v for k, v in unknown_transport.items()
                    if k in (TRANSPORT_KEY, "session_state", "session_phase", "request_id", "bytes_sent", "reference_identity_unknown", "reference_error")})
                rows.extend({"case_id": c.case_id, "verdict": "not_run",
                             "reason": "传输/参考身份未知，未继续发送套件请求"} for c in ordered_cases[index + 1:])
                break
        rows.extend({"case_id": f"check:{check}", "verdict": "unknown",
                     "reason": f"{check} 没有执行协议/证据生产者，不能当成通过"}
                    for check in sorted(set(suite.required_checks)))
        state = ("failed" if any(r["verdict"] == "failed" for r in rows) else
                 "unknown" if any(r["verdict"] != "passed" for r in rows) else "passed")
        if primary.metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN or primary.metrics.get("reference_identity_unknown"):
            state = "unknown"
        return self._suite_result(primary, suite, rows, state, task)

    @staticmethod
    def _suite_result(primary, suite, rows, state, task=None):
        primary_passed = primary.passed
        primary.passed = state == "passed"
        primary.metrics["correctness"] = {
            "verdict": state, "suite_hash": suite.digest(),
            "declared_cases": len(suite.cases), "deterministic": suite.deterministic,
            "cases": rows,
            "primary_inputs": primary.metrics.get("input_fidelity_readings", []),
            "scope": "仅证明声明的套件；未声明的形状/语义不算覆盖",
        }
        if suite.coverage_policy is not None:
            primary.metrics["correctness"]["coverage_lint"] = lint_coverage(suite)
        if task is not None and task.constraints.correctness_profile is not None:
            primary.metrics["correctness"]["profile"] = profile_check(task)
        problems = [f"{r['case_id']}: {r['reason']}" for r in rows if r["verdict"] != "passed"]
        if state != "passed":
            primary.metrics["correctness"]["reason"] = (
                "; ".join(problems) if primary_passed else primary.message)
        primary.message += (f"; CorrectnessSuite {state} ({len(suite.cases)} case)"
                            + (" —— " + "; ".join(problems) if problems else ""))
        print(f"  [Verifier] CorrectnessSuite {state}: {primary.message}")
        return primary

    def _verify_primary_function(self, kernel_code: str, test_inputs, expected_output, task: TaskSchema) -> VerificationResult:
        print("  [Verifier] L2: 功能验证...")

        # 反作弊静态审计: 执行前先检查是否存在原生算子回退调用。
        # ⚠️ 白名单的输入是**执行侧声明** (`VENDOR_TORCH_OK`), **不是**任务字段 ——
        # 以前这里传 `task.target.hardware`, 于是一个任务声明 `hardware="Ascend910"`
        # 就能让静态审计整段放行 (CHECKLOG I80 硬伤一)。今天没有声明者 ⇒ 恒 False。
        audit = static_audit(kernel_code, task.target.dsl,
                             vendor_torch_ok=VENDOR_TORCH_OK)
        violations = audit.violations
        if violations:
            print(f"  [Verifier] L2 FAILED: 反作弊静态审计未通过 ({len(violations)} 处违规)")
            from ..decide.anticheat import lib_fallback_metrics
            return VerificationResult(
                level=VerificationLevel.L2_FUNCTION, passed=False,
                message=f"反作弊: 检测到禁用的原生算子回退调用 ({len(violations)} 处)",
                error_log="\n".join(violations),
                # lib_fallback 生产者 (spine-review P1-4): 静态审计命中也是库回退
                metrics=dict(lib_fallback_metrics(violations=violations),
                             **_l1_guard_metrics(audit.lang)),
            )
        if audit.applicable is not True:
            # 「查不了」必须说出来 (CHECKLOG I65): 空的违规列表在这里**不表示**没有违规。
            # 不拦这一轮 (守卫不是判据, 判据是它的适用性 —— 由资格层收口), 但不许静默。
            print(f"  [Verifier] L2 反作弊静态审计**不适用**: {audit.lang.reason}")
            if task.constraints.correctness_suite is not None:
                return VerificationResult(
                    level=VerificationLevel.L2_FUNCTION, passed=False,
                    message=f"静态守卫不适用，正确性无法确立: {audit.lang.reason}",
                    metrics={**_l1_guard_metrics(audit.lang),
                             "correctness_primary_verdict": "unknown"})

        # 支持单组输入 (dict) 或多组输入套件 (list[dict])
        input_sets = test_inputs if isinstance(test_inputs, list) else [test_inputs]
        # 报了 `max_error` 的那些组的原始 metrics。聚合不在这里逐键做, 交给
        # `measurement.aggregate_l2_readings` 一次做完 —— 旧写法在这里备了六个标量累加器并用
        # `m.get(键, 默认)` 逐键兜底, 于是纯绝对那条路**从没算过**的 matched_ratio /
        # bad_elements / max_rel_error 被补成「全部匹配、零坏点、无相对误差」(#81,
        # 现场读数见 doc 57 §三 第 3 条与探针臂 0b)。
        sets_with_reference: list[dict] = []
        # 每组输入各自上报它**实际用的那把尺**; 是否合成一个单值, 由循环后的
        # `measurement.single_ruler` 判 (见那里的注释 —— 旧写法是「第一组赢」)
        gates: list[dict] = []
        # 逐组的高精度读数; 聚合规则见 `_aggregate_oracles` (那里记了第一版漏搬的现场)
        oracles: list[dict] = []
        # 逐组报上来的「参考自己非有限」的个数。聚合规则与上面那份同一条: **没人报过就不写这个键**,
        # 写 0 会被读成「查过了、干净的」, 而远程那一路今天还不报这个数 (见 #94 第 2 片)。
        ref_nonfinite: list[int] = []
        # 逐组的「这一窗按纯绝对判过, 而同一条 AND 判据判不过」的组号。聚合那支的 message 是
        # **重算**的 (每组自己那句被丢掉), 所以这份分歧必须在轮级这里接上, 否则最危险的那个
        # 方向 (放行) 永远只有逐组那一行读得到 —— 与 `_oracle_note` 那次同一处教训。
        alt_rejected: list[int] = []
        # 走「纯绝对」那把尺的组号 (认它的标志是 `and_form_atol` 在场 —— 只有那条路会算第二个读数)
        abs_only_sets: list[int] = []
        has_reference = False
        reference_types = set()
        execution_devices = set()
        input_fidelity_readings = []
        reference_receipts = []

        for idx, inputs in enumerate(input_sets):
            # 外部提供的 expected_output 仅对单组输入有效
            single_expected = expected_output if len(input_sets) == 1 else None
            r = self._verify_single_set(kernel_code, inputs, single_expected, task)
            if task.constraints.correctness_suite is not None:
                state, reason = verdict_of(r)
                if state != "passed":
                    r.passed = False
                    r.message = reason
                    r.metrics["correctness_primary_verdict"] = state
            if not r.passed:
                r.message = f"[输入集 {idx + 1}/{len(input_sets)}] {r.message}"
                print(f"  [Verifier] L2 FAILED: {r.message}")
                return r
            m = r.metrics
            if "reference_receipt" in m:
                reference_receipts.append(m["reference_receipt"])
            if "input_fidelity_verified" in m:
                input_fidelity_readings.append({k: v for k, v in m.items()
                                                if k.startswith("input_fidelity")})
            if "max_error" in m:
                has_reference = True
                sets_with_reference.append(m)
                g = {k: m[k] for k in GATE_PARAM_KEYS if k in m}
                if g:
                    gates.append(g)
                oracles.append({k: v for k, v in m.items()
                                if k == "ref_oracle" or k.startswith("oracle_")})
                if "reference_nonfinite" in m:
                    ref_nonfinite.append(int(m["reference_nonfinite"]))
                if "and_form_atol" in m:
                    abs_only_sets.append(idx + 1)
                # 走到这里说明这一组按本轮那把尺**判过了**; `and_form_passed is False` 就是
                # 「同一条 AND 判据会拦下它」。用 `is False` 而不是 falsy —— 全量那条根本不报
                # 这个键, 缺席是「没做这个观测」, 与「观测说不会拦」是两件事。
                if m.get("and_form_passed") is False:
                    alt_rejected.append(idx + 1)
                if m.get("reference_type"):
                    reference_types.add(m["reference_type"])
                if m.get("device"):
                    execution_devices.add(m["device"])

        gate_params, gate_conflicts = single_ruler(gates)
        oracle_agg = self._aggregate_oracles(oracles)
        agg_values, agg_origin = aggregate_l2_readings(sets_with_reference)
        agg_note = aggregate_note(agg_origin)
        if has_reference:
            msg = (f"功能验证通过 ({len(input_sets)} 组输入), "
                   f"最大误差={agg_values['max_error']:.2e}")
            # 只有真有组报过才念 —— 旧写法在这里无条件念 `matched_ratio=1.000000`,
            # 而那个 1.0 是 `m.get(..., 1.0)` 兜出来的 (#81)
            if "matched_ratio" in agg_values:
                msg += f", matched_ratio={agg_values['matched_ratio']:.6f}"
            if "coverage" in agg_values and agg_values["coverage"] < 1.0:
                msg += f", 覆盖率={agg_values['coverage'] * 100:.1f}%"
            if ref_nonfinite and sum(ref_nonfinite):
                # 「覆盖率」说的是比了多大比例, 这一句说的是**为什么**没比上 —— 两个都要有,
                # 否则读者只能去猜是分片取样的锅还是参考自己炸了
                msg += f" (参考另有 {sum(ref_nonfinite)} 个非有限值, 已从分母剔除)"
            msg += agg_note
            if abs_only_sets:
                # 「这一轮用的是哪把尺」今天必须说出来: 采样那条只报 `atol`, 而全量那条报
                # `atol`+`rtol`+`ρ` —— 少报那两个键在 `single_ruler` 眼里与「同值」不同,
                # 但读聚合结果的人看不出少了什么。
                msg += (f" ⚠️ {len(abs_only_sets)}/{len(sets_with_reference)} 组比的是采样窗口那把"
                        f"**纯绝对**尺 (无相对项、无 ρ), 组 {abs_only_sets}")
            if alt_rejected:
                msg += (f" ⚠️ 其中 {len(alt_rejected)} 组按那把尺**判过**、而同一条 AND 判据会拦下, "
                        f"组 {alt_rejected} (判决未换尺, 这一句是观测)")
            if gate_conflicts:
                msg += (f" ⚠️ 这几组用的**不是同一把尺** {sorted(gate_conflicts)}, "
                        f"聚合值因此不发布单值 (见 metrics.gate_params_conflict)")
            # 聚合这一支的 message 是**重算**的, 每组自己那句被丢掉了 ⇒ 高精度那把尺的话
            # 必须在这里接上, 否则「通过」那一侧永远读不到它 (与 _aggregate_oracles 同一条教训)
            msg += self._oracle_note(oracle_agg)
            metrics = {
                "num_input_sets": len(input_sets),
                **agg_values,
                **gate_params,
                **oracle_agg,
            }
            if agg_note:
                # 与 `aggregate_note` 同一个条件: 全都齐时不写这个键 (它就不携带信息),
                # 一旦有键缺报或部分缺报, 它说清「这个 min/sum 底下是几组」
                metrics["aggregated_over"] = agg_origin
            if gate_conflicts:
                # 摊开成 {键: {组下标: 那一组实际用的值}} —— 读的人不必重跑就能看出
                # 是哪几组、哪一把尺不一致
                metrics["gate_params_conflict"] = gate_conflicts
            if abs_only_sets:
                # 两个键都是**组号列表**而不是一个计数: 读数要能回到那一组去核
                metrics["absolute_only_sets"] = abs_only_sets
            if alt_rejected:
                metrics["and_form_would_reject_sets"] = alt_rejected
            if ref_nonfinite:
                # 有人报过才写: 一组都没报 (全走远程那条旧载荷) 时这个键**缺席**, 而不是 0 ——
                # 0 会被读成「查过了, 参考是干净的」(本仓那一族: 把没检查当检查过)
                metrics["reference_nonfinite"] = sum(ref_nonfinite)
            if reference_types:
                metrics["reference_type"] = ",".join(sorted(reference_types))
            if execution_devices:
                metrics["device"] = ",".join(sorted(execution_devices))
            if reference_receipts:
                metrics["reference_receipts"] = reference_receipts
        else:
            msg = f"功能验证通过 ({len(input_sets)} 组输入, 无参考输出, 仅验证可执行性)"
            metrics = {"num_input_sets": len(input_sets)}
        if input_fidelity_readings:
            metrics["input_fidelity_readings"] = input_fidelity_readings
        print(f"  [Verifier] L2 PASSED: {msg}")
        return VerificationResult(
            level=VerificationLevel.L2_FUNCTION, passed=True, message=msg, metrics=metrics,
        )

    @staticmethod
    def _error_shape_of(bad, *, n_bad: int, total: int, has_nonfinite: bool):
        """坏点掩码 → 误差形态标签 (numpy 侧的归约; 规则在 `tolerance`)

        归约与规则分开是有意的: 规则只有一份 (`tolerance.classify_error_shape`),
        而"在哪个后端上归约"是每个生产者自己的事 (这里是 numpy, 真机那边是 torch)。
        二维视图拿不到时 (空张量/标量) 仍然问一次规则 —— `nonfinite` 不依赖任何形状。
        """
        shape2d = _bad_mask_as_2d(bad)
        if shape2d is None:
            return tolerance.classify_error_shape(tolerance.ErrorShapeInputs(
                n_bad=n_bad, total=total, rows=0, cols=0, bad_rows=0, bad_cols=0,
                rows_all_bad=0, cols_all_bad=0, bad_row_prefix=0, bad_row_suffix=0,
                bad_col_prefix=0, bad_col_suffix=0, has_nonfinite=has_nonfinite))
        mask, n_rows, n_cols = shape2d
        return tolerance.error_shape_from_axis_flags(
            per_row=(mask.sum(axis=1) > 0).tolist(),
            per_col=(mask.sum(axis=0) > 0).tolist(),
            n_bad=n_bad, total=total,
            rows_all_bad=int((mask.sum(axis=1) == n_cols).sum()),
            cols_all_bad=int((mask.sum(axis=0) == n_rows).sum()),
            has_nonfinite=has_nonfinite)

    def _seed_spec_for(self, task: TaskSchema, inputs) -> Optional[dict]:
        """这组输入是「契约里某条 case 物化出来的那份」吗？是就返回 seed spec。

        为什么要有这一层：L2 的输入原先整份 base64 过线（fp16 4096³ 两个张量 ≈ 90 MB，
        实测链路 ~0.28 MB/s ⇒ 一次 L2 光传输 570 s）。契约本来就拥有 seed/shape/dtype，
        所以可以只传描述 + digest，远端按同一规则重新物化。

        **判据不因此变松**：这里在控制端先物化一遍、逐字节比 digest；只有完全一致才
        允许替换。远端还会再核一次 digest，任意一侧不一致就退回「传字节」（见
        `SSHExecutor.verify_function` 的 `input_seed_mismatch` 分支）。

        带标量/非 ndarray 的输入一律返回 None —— spec 表示不了它们，别硬塞。
        """
        if not isinstance(inputs, dict) or not inputs:
            return None
        if any(not isinstance(v, np.ndarray) for v in inputs.values()):
            return None
        try:
            key = case_key(inputs)
        except Exception:                       # noqa: BLE001 — 判不了就不走这条
            return None
        if key in self._seed_spec_cache:
            return self._seed_spec_cache[key]
        spec = None
        try:
            from ..decide.bench_cases import BenchmarkCase, case_spec, cases_from_shapes
            from ..decide.remote_input_fidelity import seed_input_digest
            shapes = tuple(getattr(task.constraints, "shapes", ()) or ())
            dtype = getattr(task.constraints.precision, "value", None) or "fp32"
            entries = tuple((tuple(int(d) for d in s), 1.0) for s in shapes)
            cases = cases_from_shapes((task.source.kernel_type or "").strip(), entries,
                                      dtype) or ()
            for case in cases:
                names = {t.name for t in case.tensors}
                if case.key() == key:
                    expected = case.materialize()
                    if seed_input_digest(expected) == seed_input_digest(inputs):
                        spec = case_spec(case)
                        break
                    continue
                # rows-case（layernorm/softmax/...）声明 X/weight/bias，但无 affine 的
                # 内建 profile 只物化 X。集合不相等会让这条大输入回落成整份 JSON
                # （2026-10-30 实测 249 MB 传 5.5 min 仍未完）。只在「缺的恰好是
                # weight/bias」时按**该 case 的子集**重建，并照旧核 key 与 digest；
                # 任何数值不一致仍返回 None（回退传字节）。
                extra = names - set(inputs)
                if extra and extra <= {"weight", "bias"} and set(inputs) <= names:
                    subset = BenchmarkCase(
                        name=case.name, seed=case.seed,
                        tensors=tuple(t for t in case.tensors if t.name in inputs))
                    if subset.key() != key:
                        continue
                    expected = subset.materialize()
                    if seed_input_digest(expected) == seed_input_digest(inputs):
                        spec = case_spec(subset)
                        break
        except Exception as exc:                # noqa: BLE001 — 出任何事都退回传字节
            spec = None
        self._seed_spec_cache[key] = spec
        return spec

    def _verify_single_set(self, kernel_code: str, inputs: dict, expected_output, task: TaskSchema) -> VerificationResult:
        """对单组输入执行功能验证
        metrics 中含 max_error 表示完成了参考比对，否则仅为可执行性验证"""
        max_allowed = task.constraints.max_error
        if task.constraints.reference_contract is not None or task.task_id in self._reference_bindings:
            preflight = self.reference_preflight(task)
            if preflight is not None:
                return preflight
        expected_inputs = None
        fidelity_metrics = {}
        # Seed spec 要在 describe 之前算：能走 seed 的输入不再整份过 JSON/字节，
        # 因此可放宽单 tensor 上限（JSON 路径仍是 MAX_STORAGE_BYTES）。
        seed_spec = (self._seed_spec_for(task, inputs)
                     if Cap.REMOTE_VERIFY in self._caps else None)
        if task.constraints.correctness_suite is not None:
            try:
                expected_inputs = describe_inputs(
                    inputs, limit=(MAX_SEED_STORAGE_BYTES if seed_spec is not None else None))
            except InputFidelityError as exc:
                return VerificationResult(
                    VerificationLevel.L2_FUNCTION, False,
                    message=f"输入保真无法确立: {exc}",
                    metrics={"input_fidelity_verified": None,
                             "input_fidelity_reason": str(exc)})

        if Cap.REMOTE_VERIFY in self._caps:
            # 候选与 reference 均在远端执行设备上比较，仅返回摘要，
            # 避免把完整张量通过 SSH 回传到控制端。
            remote_verify = self.backend.verify_function(
                kernel_code, inputs, task.source.kernel_type, max_allowed,
                precision=task.constraints.precision.value,
                reference_source=task.source.reference_source or "",
                **({"input_seed": seed_spec} if seed_spec is not None else {}),
                **({"reference_contract": as_reference(task.constraints.reference_contract).content_dict()}
                   if task.constraints.reference_contract is not None else {}),
            )
            if task.constraints.reference_contract is not None:
                try:
                    if not isinstance(remote_verify, dict):
                        raise ValueError("远端参考回执缺失")
                    if remote_verify.get("reference_error"):
                        raise ValueError(remote_verify["reference_error"])
                    validate_receipt(task.constraints.reference_contract, remote_verify.get("reference_receipt"))
                    if remote_verify.get("success") and remote_verify.get("output_dtype") != "torch.float16":
                        raise ValueError("fp16 输出精度回读证据缺失")
                    if "max_error" in remote_verify:
                        spec = as_reference(task.constraints.reference_contract)
                        if (remote_verify.get("atol"), remote_verify.get("rtol"),
                                remote_verify.get("required_matched_ratio")) != (max_allowed, spec.rtol, spec.required_matched_ratio):
                            raise ValueError("执行侧数值门与冻结契约不一致")
                    if remote_verify.get("success") is True:
                        if ("max_error" not in remote_verify or remote_verify.get("coverage") != 1.0
                                or type(remote_verify.get("compared_elements")) is not int
                                or remote_verify["compared_elements"] != inputs["X"].size
                                or remote_verify.get("reference_nonfinite") != 0
                                or remote_verify.get("matched_ratio") != 1.0
                                or remote_verify.get("bad_elements") != 0
                                or any(type(remote_verify.get(k)) not in (int, float)
                                       or not np.isfinite(remote_verify[k]) or remote_verify[k] < 0
                                       for k in ("max_error", "max_rel_error"))
                                or not isinstance(remote_verify.get("output_digest"), str)
                                or len(remote_verify["output_digest"]) != 64):
                            raise ValueError("v2 完整数值比对/有限参考/输出摘要证据缺失")
                except (ValueError, TypeError) as exc:
                    return self._reference_unknown(str(exc), payload=remote_verify)
            # 声明了 REMOTE_VERIFY 就必须给出载荷。「内层没这个方法」已经由 Backend
            # 基类那个一调就抛的存根拦掉了, 走到这里的 None 只能是「声明了却没数据」——
            # 把它读成「换条路再试试」正是 doc 49 §3.2 B3: 两个意思共用一个值。
            if remote_verify is None:
                raise CapabilityMissing(Cap.REMOTE_VERIFY, self.backend,
                                        "verify_function 声明可用但返回了 None")
            if remote_verify.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN:
                return VerificationResult(VerificationLevel.L2_FUNCTION, False,
                    message="远程传输结果未知: " + remote_verify.get("error", ""),
                    metrics={k: remote_verify[k] for k in
                             (TRANSPORT_KEY, "session_state", "session_phase", "request_id", "bytes_sent")
                             if k in remote_verify})
            if expected_inputs is not None:
                fidelity_metrics = check_readback(expected_inputs, remote_verify.get("input_fidelity"))
                if fidelity_metrics["input_fidelity_verified"] is not True:
                    if remote_verify.get("input_fidelity_error"):
                        fidelity_metrics["input_fidelity_reason"] = remote_verify["input_fidelity_error"]
                    return VerificationResult(
                        VerificationLevel.L2_FUNCTION, False,
                        message=fidelity_metrics["input_fidelity_reason"], metrics=fidelity_metrics)
                if Cap.REAL_DEVICE in self._caps:
                    # A real-device capability claim cannot turn a CPU fallback
                    # into accelerator correctness. Device facts come from the
                    # same actual tensors that supplied layout/value readback.
                    devices = remote_verify["input_fidelity"].get("tensor_devices")
                    names = {k for k, v in expected_inputs.items() if v.get("kind") != "scalar"}
                    accelerator_types = {"cuda", "xpu", "npu", "mps", "hpu", "privateuseone"}
                    if (not names or not isinstance(devices, dict) or set(devices) != names
                            or any(type(d) is not str or d.split(":", 1)[0] not in accelerator_types
                                   for d in devices.values()) or len(set(devices.values())) != 1):
                        reason = "后端声明真实设备，但实际输入的同一加速设备回读证据未确立（不接受 CPU 回退）"
                        fidelity_metrics["correctness_execution_error"] = reason
                        return VerificationResult(VerificationLevel.L2_FUNCTION, False,
                                                  message=reason, metrics=fidelity_metrics)
            if not remote_verify.get("success", False):
                if remote_verify.get("candidate_semantics_error"):
                    return VerificationResult(VerificationLevel.L2_FUNCTION, False,
                        message=remote_verify["candidate_semantics_error"], metrics={**fidelity_metrics,
                            **{k: remote_verify[k] for k in _REMOTE_L2_METRIC_KEYS if k in remote_verify}})
                # ``success=False`` 也可能是数值误差超标；保留远端的精确诊断。
                if "max_error" in remote_verify:
                    # 形态与本地的 message 一样要进 message (Fixer / 失败语料读它),
                    # 远端老版本不带这个键 ⇒ 如实说"未能确立", 不假装没有形态这回事。
                    _shape = remote_verify.get("error_shape")
                    return VerificationResult(
                        level=VerificationLevel.L2_FUNCTION, passed=False,
                        message=(f"远程功能验证失败: 误差 {remote_verify['max_error']:.2e}"
                                 + (f", 形态={_shape}" if _shape else
                                    ", 形态未能确立")),
                        error_log=remote_verify.get("error", ""),
                        metrics={**{k: remote_verify[k] for k in _REMOTE_L2_METRIC_KEYS
                                    if k in remote_verify}, **fidelity_metrics},
                    )
                if remote_verify.get("output_shape") and remote_verify.get("expected_shape"):
                    return VerificationResult(
                        level=VerificationLevel.L2_FUNCTION, passed=False,
                        message=(f"输出形状不匹配: {remote_verify['output_shape']} vs "
                                 f"{remote_verify['expected_shape']}"),
                        error_log=remote_verify.get("error", ""),
                        metrics=fidelity_metrics,
                    )
                # message 只带首行, 全文放 error_log (见 failure_pair)
                head, full = failure_pair(remote_verify.get("error", ""))
                # 运行时盾的证据必须落地。盾跑在**远端进程**里 ⇒ 本地那个
                # `take_lib_fallback()` 永远是空的, 不认错误文本的话这条库回退只在日志里
                # 活一次, 而资格规则 4 恰恰是给「静态审计漏了、盾拦住了」这条路径准备的
                # (真机 §14-3: 拼名字的 `torch.matmul` 被远端盾拦下, lib_fallback 却是 None)。
                from ..decide.anticheat import lib_fallback_metrics
                fail_metrics = ({TRANSPORT_KEY: remote_verify[TRANSPORT_KEY]}
                                if TRANSPORT_KEY in remote_verify else {})
                fail_metrics.update(lib_fallback_metrics(remote_error=full or head))
                fail_metrics.update(fidelity_metrics)
                if remote_verify.get("reference_error"):
                    fail_metrics["reference_error"] = remote_verify["reference_error"]
                # Typed compiler provenance is useful only after actual input
                # readback established a known candidate failure. Exception
                # text, reference errors and transport uncertainty cannot
                # manufacture a compiler diagnosis.
                if (remote_verify.get("success") is False
                        and fidelity_metrics.get("input_fidelity_verified") is True
                        and remote_verify.get("device") == "cuda"
                        and (TRANSPORT_KEY not in remote_verify
                             or remote_verify[TRANSPORT_KEY] == TRANSPORT_ANSWERED)
                        and not remote_verify.get("reference_error")
                        and not remote_verify.get("reference_identity_unknown")):
                    readback = remote_verify.get("input_fidelity", {})
                    devices = readback.get("tensor_devices")
                    names = {name for name, spec in expected_inputs.items()
                             if spec.get("kind") != "scalar"}
                    cuda_readback = (readback.get("representation") == "torch-storage-v1"
                        and names and isinstance(devices, dict) and set(devices) == names
                        and all(type(value) is str and re.fullmatch(r"cuda:[0-9]+", value)
                                for value in devices.values())
                        and len(set(devices.values())) == 1)
                    compile_failure = (validate_compile_failure(
                        remote_verify.get("candidate_compile_failure"), kernel_code)
                        if cuda_readback else None)
                    if compile_failure is not None:
                        fail_metrics["candidate_compile_failure"] = compile_failure
                        fail_metrics["device"] = "cuda"
                return VerificationResult(
                    level=VerificationLevel.L2_FUNCTION, passed=False,
                    message=f"远程执行失败: {head}",
                    error_log=full,
                    # 「没听到回答」与「听到了失败」要分开带着走 (#102 第 1 片) —— 只在载荷
                    # 真报了这一格时发布, 缺的键不填默认值 (与 aggregated_over 同一纪律)
                    metrics=fail_metrics)
            return VerificationResult(
                level=VerificationLevel.L2_FUNCTION, passed=True,
                message=(f"远程全量比对通过 ({remote_verify.get('compared_elements', 0)} 元素), "
                         f"最大误差={remote_verify.get('max_error', 0):.2e}"),
                metrics={**{k: remote_verify[k] for k in _REMOTE_L2_METRIC_KEYS
                            if k in remote_verify}, **fidelity_metrics},
            )

        # 没有远端紧凑比对协议: 在执行侧跑一遍, 由 Verifier 自己比对参考输出。
        result = self.backend.execute(kernel_code, inputs)
        if not result.success and transport_outcome_of_result(result) == TRANSPORT_UNKNOWN:
            return VerificationResult(VerificationLevel.L2_FUNCTION, False,
                message="远程传输结果未知: " + result.error,
                metrics={TRANSPORT_KEY: TRANSPORT_UNKNOWN})
        if expected_inputs is not None:
            fidelity_metrics = check_readback(
                expected_inputs, (getattr(result, "metrics", None) or {}).get("input_fidelity"))
            if fidelity_metrics["input_fidelity_verified"] is not True:
                if (getattr(result, "metrics", None) or {}).get("input_fidelity_error"):
                    fidelity_metrics["input_fidelity_reason"] = result.metrics["input_fidelity_error"]
                return VerificationResult(
                    VerificationLevel.L2_FUNCTION, False,
                    message=fidelity_metrics["input_fidelity_reason"], metrics=fidelity_metrics)
        if not result.success:
            head, full = failure_pair(result.error)
            fail_metrics = {TRANSPORT_KEY: transport_outcome_of_result(result), **fidelity_metrics}
            # lib_fallback 生产者 (spine-review P1-4): 运行时 shield 拦到的库回退
            # 必须搬进 metrics, 否则 eligibility 规则 4 恒 False。
            # 优先抄执行结果里已有的 metrics (mock_gpu 已 harvest 过), 避免二次 take。
            src = getattr(result, "metrics", None) or {}
            if src.get("lib_fallback") or getattr(result, "lib_fallback", False):
                fail_metrics.update({k: src[k] for k in
                                     ("lib_fallback", "lib_fallback_evidence")
                                     if k in src})
                fail_metrics["lib_fallback"] = True
            else:
                from ..decide.anticheat import lib_fallback_metrics
                fail_metrics.update(lib_fallback_metrics())
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False,
                                      message=f"执行失败: {head}", error_log=full,
                                      metrics=fail_metrics)
        if expected_output is None:
            expected_output = self._generate_reference(task, inputs)
        if expected_output is None:
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=True, message="无参考输出，仅验证可执行性", metrics={"execution_time_ms": result.execution_time_ms, **fidelity_metrics})

        if Cap.FULL_OUTPUT_TENSOR not in self._caps:
            # execute() 只回样本: 与参考的前缀比对, 并把 coverage 一起报出去 ——
            # 覆盖不足是**数据**, 不能因为「比过的那部分都对」就读成通过。
            ref = np.asarray(expected_output, dtype=np.float32)
            # 形状检查 (此前远程模式完全不做形状校验)
            if result.output_shape and list(result.output_shape) != list(ref.shape):
                return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"输出形状不匹配: {result.output_shape} vs {list(ref.shape)}")

            ref_flat = ref.flatten()
            sample = np.array(result.output_sample[:ref_flat.size], dtype=np.float32) if result.output_sample else np.array([])
            if sample.size == 0:
                return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message="执行侧未返回输出样本，无法比对")

            compared = int(sample.size)
            # 同一条纪律的第二处 (全量那条在 `effective_atol` 那一段): 「比得上」要显式算,
            # 不能靠 NaN 的比较性质。这条路上候选自己产 NaN/Inf 时**照旧判死** —— 那是候选的
            # 错, 与参考侧非有限是两件事, 归属不能混 (`cand_bad`)。
            ref_window = ref_flat[:compared]
            ref_ok = np.isfinite(ref_window)
            n_compared = int(ref_ok.sum())
            cand_bad = int((~np.isfinite(sample)).sum())
            max_error = float(np.abs(sample[ref_ok] - ref_window[ref_ok]).max()) if n_compared else 0.0
            coverage = n_compared / ref_flat.size if ref_flat.size else 1.0
            metrics = {"max_error": max_error, "compared_elements": n_compared,
                       **fidelity_metrics,
                       "coverage": coverage,
                       "reference_nonfinite": compared - n_compared}
            if n_compared == 0:
                return VerificationResult(
                    level=VerificationLevel.L2_FUNCTION, passed=False,
                    message=(f"采样窗口内参考没有可比的元素 ({compared} 个全是 NaN/Inf) —— "
                             f"本级判不出, 这不是候选错; 该查 reference_source / 内建参考"),
                    metrics=metrics)
            if cand_bad:
                return VerificationResult(
                    level=VerificationLevel.L2_FUNCTION, passed=False,
                    message=f"执行失败: 候选输出含 {cand_bad} 个非有限值 (NaN/Inf)",
                    metrics=metrics)
            # #80 的观测半边 —— 这一条路用的**不是**全量那条与真机那条的尺: 它是纯绝对
            # (`max_error <= task.constraints.max_error`), 不查容差表、没有相对项、也没有 ρ。
            # 因为 `effective_atol` 取的是 `min(表, 任务自带值)`, 这把尺只会**同松或更松**,
            # 所以「整体缩放一点点」那一类候选 (doc 12 / tolerance.py docstring 立的案) 能从
            # 这里放过去, 而 AND 那条会拦。更糟的是它以前一个尺键都不报 ⇒ 混用两条路时
            # `single_ruler` (轮级) 与 `l2_ruler_reading` / `cross_round_ruler` (运行级)
            # 只能读成「这一轮没报尺」, 看不见「这几组用的不是同一把尺」。
            # 这一片**不改判决**, 只做两件事: ① 如实发布这一窗真正用过的绝对阈 (`atol`),
            # 让那三个已有的读者第一次能报警; ② 把同一条 AND 判据套在同一窗上的第二个读数
            # 发布成 `and_form_*` 旁证键 —— 判定规则仍只有 `tolerance` 那一份, 这里不调形状。
            passed_abs = max_error <= max_allowed
            alt_atol, alt_rtol, alt_rho = tolerance.effective_atol(
                task.constraints.precision.value, max_allowed)
            alt_diff = np.abs(sample[ref_ok] - ref_window[ref_ok])
            alt_bad = int(((alt_diff > alt_atol) &
                           (alt_diff / (np.abs(ref_window[ref_ok]) + tolerance.EPS) > alt_rtol)).sum())
            alt_passed, alt_matched = tolerance.evaluate(alt_bad, n_compared, False, alt_rho)
            metrics.update({
                "atol": max_allowed,
                "and_form_atol": alt_atol, "and_form_rtol": alt_rtol,
                "and_form_required_matched_ratio": alt_rho,
                "and_form_bad_elements": alt_bad,
                "and_form_matched_ratio": alt_matched,
                "and_form_passed": alt_passed,
            })
            both = ("两把尺判得一致" if alt_passed == passed_abs else
                    f"⚠️ 分歧: 纯绝对判{'过' if passed_abs else '不过'} 而 AND 判"
                    f"{'过' if alt_passed else '不过'} ({alt_bad}/{n_compared} 个可比元素同时超 "
                    f"atol={alt_atol:.1e} 与 rtol={alt_rtol:.1e})")
            note = (f" —— 这一窗用的尺是纯绝对 atol={max_allowed:.1e} (无相对项、无 ρ); {both}"
                    f"。判决没换尺, 这一句是观测")
            if passed_abs:
                extra = (f", 另有 {compared - n_compared} 个元素参考非有限已剔除"
                         if compared - n_compared else "")
                return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=True,
                                          message=(f"采样比对通过 ({n_compared}/{ref_flat.size} 元素可比), "
                                                   f"最大误差={max_error:.2e}" + extra + note),
                                          metrics=metrics)
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False,
                                      message=f"功能验证失败: 误差 {max_error:.2e}" + note,
                                      metrics=metrics)

        actual = result.output
        expected = np.asarray(expected_output, dtype=np.float32)
        if actual.shape != expected.shape:
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"输出形状不匹配: {actual.shape} vs {expected.shape}", metrics=fidelity_metrics)

        # 数值门禁与真机路径 (remote_scripts/run_kernel.py) 共用 tolerance.py 里的
        # 同一张容差表和同一条判定式: 绝对 + 相对 的 AND 形式, 见该模块 docstring。
        atol, rtol, rho = tolerance.effective_atol(
            task.constraints.precision.value, max_allowed)
        actual_f = np.asarray(actual, dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            # 非有限的那几行会在这里发出 "invalid value encountered in divide" 的 RuntimeWarning。
            # 屏蔽它是安全的、也是有条件的: 它们随后就被 `comparable` 掩码挡在分子与分母之外。
            # 不屏蔽的话这条信号会成倍刷屏 (每个坏元素一次), 而它要说的事已经由
            # `reference_nonfinite` 那个**计数**说清了 —— 一次说清, 而不是刷 256 行。
            abs_error = np.abs(actual_f - expected)
            rel_error = abs_error / (np.abs(expected) + tolerance.EPS)
        # 参考自己也可能不是数。外部一手 (arXiv:2609.33074v1 行 147, 逐字):
        #     "Any sample that produces invalid values such as NaN or Inf is filtered out."
        # 他们在**任务侧**做这一步, 我方原来只在候选侧查 (`has_nonfinite` 看的是 `actual_f`)。
        # 漏掉的后果不是「错核被放行」那么轻: 参考含 NaN 时 `NaN > atol` 为假, 含 Inf 时
        # `inf/inf = NaN` 让相对项为假 —— AND 形式下**任一侧出非有限值就一票否决坏点**, 于是
        # 那 16 个元素既不算坏点又留在分母里, `compared_elements` 报的是「比了 256 个」而实际
        # 只比上了 240 个。所以这里把「比得上」显式成一侧掩码, 而不是继续靠 NaN 的比较性质。
        comparable = np.isfinite(expected)
        n_compared = int(comparable.sum())
        n_ref_nonfinite = int(expected.size) - n_compared
        # ⚠️ 分子上那句 `& comparable` **今天是个恒等操作**, 所以变异 R1 撞不红任何用例:
        # AND 要置坏点需两条同时为真, 而参考非有限时 `abs_error` 或 `rel_error` 必有一个是
        # NaN、比较恒假 —— 那些元素在加掩码之前就已经不是坏点了。牙齿在分母与上报那三处
        # (变异 R2/R3/R5/R6 全部拦下)。留着它的理由: 判据形状一改 (AND→任一侧超标) 它立刻
        # 变成承重墙, 而那一步今天由 `tests/unit/test_l2_tolerance.py` 的行为用例把守。
        bad = (abs_error > atol) & (rel_error > rtol) & comparable
        has_nonfinite = bool((~np.isfinite(actual_f)).any())
        passed, matched_ratio = tolerance.evaluate(
            int(bad.sum()), n_compared, has_nonfinite, rho)

        size = int(expected.size)
        metrics = {
            **fidelity_metrics,
            "max_error": float(abs_error[comparable].max()) if n_compared else 0.0,
            "max_rel_error": float(rel_error[comparable].max()) if n_compared else 0.0,
            "matched_ratio": matched_ratio,
            "bad_elements": int(bad.sum()),
            "atol": atol, "rtol": rtol, "required_matched_ratio": rho,
            # `compared_elements` 从此只数**比得上的**那些, `coverage` 跟着它一起变小 ——
            # 与采样那条路的 `coverage` 同一个意思 (比了多大比例), 不是「形状对不对」。
            "compared_elements": n_compared,
            "coverage": (n_compared / size) if size else 1.0,
            "reference_nonfinite": n_ref_nonfinite,
        }
        if task.constraints.correctness_suite is not None:
            # Hash the actual dtype/shape/bytes before float32 comparison casting.
            arr = np.asarray(actual)
            digest = hashlib.sha256()
            digest.update(str((arr.dtype.str, arr.shape)).encode())
            digest.update(arr.tobytes(order="C"))
            metrics["output_digest"] = digest.hexdigest()
        # 高精度参考 (float64) 那把尺的读数。**不参与判定**，只跟着上报 —— 理由与阈值来源
        # 写在 `tolerance.error_ratio` 的注释里 (一句话: 那三个阈是别人语料上的数)。
        metrics.update(self._oracle_reading(task, inputs, actual_f, expected))
        # first_bad_*: 第一个坏元素的展平下标与两侧取值 (LeetCUDA book_check 同款)。
        # 展平下标够 Fixer 定位坏 tile; 顺带带上 actual/expected 才分得清
        # 「整体 scale 错」(符号同、比值恒定) 与「局部炸」(个别点)。
        if size and int(bad.sum()) > 0:
            fb = int(np.argmax(bad))
            metrics["first_bad_index"] = fb
            metrics["first_bad_actual"] = float(actual_f.reshape(-1)[fb])
            metrics["first_bad_expected"] = float(expected.reshape(-1)[fb])

        # 误差**形态** (设计 §8.2): 位置回答"哪错了", 形态回答"该怎么修"。
        # ⚠️ 触发条件是**这一组判失败**, 不是"有坏点": 含 NaN/Inf 时坏点计数是 0
        # (NaN 的比较恒假), 而那恰恰是最该说出口的一种形态 —— 挂在坏点计数上的话,
        # 它会是唯一永远报不出来的那一类。
        # 归约在这里 (numpy), 分类规则在 `tolerance` (与真机路径共用一份); 把张量看成
        # 二维 (除最后一维外全部折叠), 于是 matmul 与 attention 同一套规则。
        if not passed:
            metrics["error_shape"] = self._error_shape_of(
                bad, n_bad=int(bad.sum()), total=int(n_compared),
                has_nonfinite=has_nonfinite)
        if passed:
            extra = (f" (参考另有 {n_ref_nonfinite} 个非有限值, 已从分母剔除)"
                     if n_ref_nonfinite else "")
            return VerificationResult(
                level=VerificationLevel.L2_FUNCTION, passed=True,
                message=(f"全量比对通过 ({n_compared}/{size} 元素可比), "
                         f"最大误差={metrics['max_error']:.2e}, "
                         f"matched_ratio={matched_ratio:.6f}" + extra
                         + self._oracle_note(metrics)),
                metrics=metrics)
        # 归属要说清: 一个可比元素都没有时, 「没过」不是候选的错, 而是本级**没有对象**。
        # 这句话是给 Fixer 与失败语料看的 (它们都按「哪一级报的失败」分支, `fixer.py:99`),
        # 而把坏参考记成坏候选会发出一份指向错误对象的处方 —— 那是 #85/#76① 那一族在避免的。
        if n_compared == 0:
            reason = (f"参考自身没有可比的元素 ({size} 个全是 NaN/Inf) —— "
                      f"本级判不出, 这不是候选错; 该查的是 reference_source / 内建参考的算法")
        else:
            reason = ("输出含 NaN/Inf" if has_nonfinite else
                      f"{metrics['bad_elements']}/{n_compared} 个可比元素同时超 "
                      f"atol={atol:.1e} 与 rtol={rtol:.1e}")
            if n_ref_nonfinite:
                reason += f" (另有 {n_ref_nonfinite} 个元素参考非有限, 已从分母剔除)"
        if "first_bad_index" in metrics:
            reason += (f", first_bad@{metrics['first_bad_index']}"
                       f" (got {metrics['first_bad_actual']:.6g},"
                       f" want {metrics['first_bad_expected']:.6g})")
        # 形态进 message 是**有意的**: message 会进 Fixer 与失败语料 (它们是唯一保证被
        # 下游看到的那条通道), 而"哪一类错"比"错了多少"更能决定该改哪里 (设计 §8.2)。
        # `None` (判不了) 也要说出来 —— 沉默会被读成"形态没问题"。
        reason += (f", 形态={metrics['error_shape']}" if metrics.get("error_shape")
                   else ", 形态未能确立 (坏点分布不落在已知的几类上)")
        return VerificationResult(
            level=VerificationLevel.L2_FUNCTION, passed=False,
            message=(f"功能验证失败: {reason}"
                     + (f", 最大误差={metrics['max_error']:.2e}" if n_compared else "")),
            metrics=metrics)

    def _oracle_reading(self, task: TaskSchema, inputs: dict,
                        actual, expected) -> dict:
        """高精度参考 (同一份参考、float64 再算一遍) 那把尺的读数 —— **只观测, 不判定**

        外部一手: KernelOPT `2609.30059v1` 行 372-389 与行 1344-1362 (规则抄在
        `tolerance.error_ratio` 那里, 包括「这三个阈是别人语料上的数」那句警告)。

        与 AND 那条的差别值得再说一遍: AND 比「候选 vs 参考」, 条带是绝对 atol ⇒ 参考幅值一小
        就看不见任何东西 (`benchmarks/probes/probe_l2_sampling_gate.py` 臂 1/2 拍到的全零核 PASS)。
        这把尺的分母是**参考自己到真值的距离**, 条带自动随幅值缩放。

        三种拿不到读数的情况都**不造一个数** (缺证据不是通过, 也不是 0):
          · 没有 float64 参考 (外挂 `reference_source` / int8 任务) ⇒ `ref_oracle="absent"`
          · 参考自己算炸了 ⇒ `"error:<异常类型名>"` (远程错误原文不进仓库)
          · 参考输出的形状与 float32 那份不一致 ⇒ `"shape_mismatch"`
        """
        try:
            oracle = self._generate_reference(task, inputs, high_precision=True)
        except Exception as exc:  # noqa: BLE001
            return {"ref_oracle": f"error:{type(exc).__name__}"}
        if oracle is None:
            return {"ref_oracle": "absent"}
        oracle = np.asarray(oracle, dtype=np.float64)
        if oracle.shape != expected.shape:
            return {"ref_oracle": "shape_mismatch"}
        r = tolerance.error_ratio(
            d_opt=float(np.abs(actual - oracle).max()),
            d_ref=float(np.abs(expected.astype(np.float64) - oracle).max()),
            ref_amplitude=float(np.abs(expected).max()),
        )
        out = {"ref_oracle": "measured", "oracle_branch": r["branch"],
               "oracle_d_ref": r["d_ref"], "oracle_d_opt": r["d_opt"],
               "oracle_scale_relative": r["scale_relative_error"],
               "oracle_abs_bound": r["abs_bound"],
               "oracle_meets_ratio": r["meets_ratio"],
               "oracle_meets_scale_relative": r["meets_scale_relative"],
               "oracle_meets_abs_bound": r["meets_abs_bound"]}
        # **按这一组该走的那一支**折成一个「那把尺反不反对」: ratio 支看 ρ, exact 支看幅值缩放。
        # 不这么折的话, 走 exact 支的那一组的 `meets_ratio` 恒 False, 跨组再 AND 一次就成了
        # 「分支不同 ⇒ 一律反对」—— 两个意思共用一个值 (doc 49 §3.2 B3 那一族)。
        out["oracle_objects"] = (r["meets_ratio"] if r["branch"] == "ratio"
                                 else r["meets_scale_relative"]) is False
        # 退化分支没有 ρ 这个数 —— 缺就**不写这个键**, 写成 None 会被下游的
        # `metrics.get(k, 0)` 读成「ρ=0 = 完美」(#53 那一族)。
        if r["rho"] is not None:
            out["oracle_rho"] = r["rho"]
        return out

    @staticmethod
    def _aggregate_oracles(oracles: list) -> dict:
        """把逐组的高精度读数搬进聚合 metrics

        ⚠️ 这一整块为什么存在: 读数是逐组算出来的, 而聚合处**手写**了一份新 metrics ——
        第一版就死在这条搬运路上 (值算出来了，而「通过」那一侧没有它)。本仓同一个形状撞过
        三次: doc 33 的 `runs`/`max_time_ms`、doc 37 的 `llm_cost`、doc 48 的 `candidate_signature`。
        「通过」正是唯一会被读的那一侧, 所以这条搬运必须有一条用例当证人。
        """
        if not oracles:
            return {}
        measured = [o for o in oracles if o.get("ref_oracle") == "measured"]
        if not measured:
            # 一组都没测到 ⇒ 什么都不上报 (缺席既不是 True 也不是 0)
            return {"ref_oracle": oracles[0].get("ref_oracle", "absent")}
        out = {"ref_oracle": ("measured" if len(measured) == len(oracles)
                              else "partially_measured"),
               "oracle_sets_measured": len(measured),
               "oracle_sets_total": len(oracles),
               "oracle_objects": any(o["oracle_objects"] for o in measured),
               "oracle_branches": ",".join(sorted({o["oracle_branch"] for o in measured}))}
        rhos = [o["oracle_rho"] for o in measured if "oracle_rho" in o]
        if rhos:
            out["oracle_rho_max"] = max(rhos)
        scales = [o["oracle_scale_relative"] for o in measured
                  if "oracle_scale_relative" in o]
        if scales:
            # 取 max 而不是 mean: 聚合要说的从来不是「平均差多少」，而是「最差那一组差多少」
            # (与上面 min_matched_ratio / max_error_overall 同一套折叠方向)
            out["oracle_scale_relative_max"] = max(scales)
        return out

    @staticmethod
    def _oracle_note(metrics: dict) -> str:
        """判决与高精度那把尺不一致时, 在 message 里说一句 (仍然不改判决)

        这一句是给**读日志的人**的: 「本级判通过，而按参考自身误差缩放的那把尺它不该过」
        是 #82 那条盲区的现场。今天只报不判，因为阈是抄来的。

        ⚠️ 两个形状都要读得出来: 聚合那一份的键名带 `_max`/`_es` 后缀 (`oracle_branches`),
        逐组那一份没有 (`oracle_branch`)。第一版只读聚合那份, 于是**逐组那一路印出来的是
        「（None 支）」而且幅值那半句整个消失** —— 一个印了却什么都没说的句子。今天那条路径
        只有直接调 `_verify_single_set` 的探针/单测会看到 (聚合处会重算 message), 但 #94 正要在
        同一个函数上动参考侧的检查, 与其留一个「读不到就不说话」的 formatter, 不如现在就接上。
        """
        if metrics.get("ref_oracle") not in ("measured", "partially_measured"):
            return ""
        if not metrics.get("oracle_objects"):
            return ""
        # 先认「手里拿的是哪一份」: 聚合那份的键名带 `_max`/`_es`，逐组那份没有。两个形状共用
        # 一个 formatter 时**口径也要跟着分** —— 「最差一组」这个限定只在聚合那份上成立，
        # 加到逐组那一路就读成「还有更差的组」。
        agg = any(k in metrics for k in ("oracle_rho_max", "oracle_scale_relative_max",
                                         "oracle_branches"))
        branch = (metrics.get("oracle_branches") if agg else metrics.get("oracle_branch"))
        rho = metrics.get("oracle_rho_max" if agg else "oracle_rho")
        scale = metrics.get("oracle_scale_relative_max" if agg else "oracle_scale_relative")
        where = ("（" + branch + " 支" if branch else "（分支未知")
        note = (f" ⚠️ 高精度参考那把尺反对这个判决{where}；"
                f"阈取自 KernelOPT、我方未标定，仅观测）")
        lead = "，最差一组" if agg else "，"
        if isinstance(rho, (int, float)):
            return note + lead + f"ρ={rho:.3g}"
        if isinstance(scale, (int, float)):
            # exact 支没有 ρ (分母不可信时不许除) —— 说「参考幅值的多少倍」是同一件事的另一种说法
            return note + lead + f"误差是参考幅值的 {scale:.3g} 倍"
        return note + "，而这把尺没给出可比的数"

    def _bench_case(self, kernel_code: str, inputs: dict, task: TaskSchema):
        """量**一组**输入的「候选 + 基线」—— 主 case 与逐 case 账本共用这一处

        返回 `(bench, remote_bench)`：远端优先（`benchmark_with_baselines`，同机基准
        协议），否则本机 `benchmark`。协议参数（`repeat=5` / `baseline_priority`）只
        在这里写一次 —— 主 case 与账本里其余 case 的读数会被放在同一张表里比，两处
        协议一漂就会比出两种数（这是「一个事实一处生产」在本函数上的样子）。
        """
        remote_bench = None
        preflight = self.reference_preflight(task)
        if preflight is not None:
            held = dict(success=False, error=preflight.message, **preflight.metrics)
            return held, held
        if Cap.TEST_ONLY in self._caps:
            # The fixture owns both arms. Never invoke the local Torch baseline,
            # which can probe/select CUDA independently of the fake backend.
            fixture = self.backend.benchmark(kernel_code, inputs, warmup=2, repeat=5)
            return fixture, fixture
        if Cap.SAME_DEVICE_BASELINE in self._caps:
            seed_spec = self._seed_spec_for(task, inputs)
            remote_bench = self.backend.benchmark_with_baselines(
                kernel_code, inputs, task.source.kernel_type,
                precision=task.constraints.precision.value,
                baseline_priority=(["pytorch_eager"] if task.constraints.reference_contract is not None else
                                   baseline_plan(task.source.kernel_type, task.constraints.precision.value)),
                reference_source=task.source.reference_source or "",
                **({"input_seed": seed_spec} if seed_spec is not None else {}),
                **({"reference_contract": as_reference(task.constraints.reference_contract).content_dict()}
                   if task.constraints.reference_contract is not None else {}),
            )
            if task.constraints.reference_contract is not None:
                try:
                    if not isinstance(remote_bench, dict):
                        raise ValueError("远端基线参考身份回执缺失")
                    if remote_bench.get("reference_error"):
                        raise ValueError(remote_bench["reference_error"])
                    validate_receipt(task.constraints.reference_contract, remote_bench.get("reference_receipt"))
                    if remote_bench.get("success") and remote_bench.get("baseline_selected") != "pytorch_eager":
                        raise ValueError("v2 baseline changed; fallback forbidden")
                    if remote_bench.get("success"):
                        fidelity = check_readback(describe_inputs(inputs), remote_bench.get("input_fidelity"))
                        if fidelity.get("input_fidelity_verified") is not True:
                            raise ValueError("v2 L3 精确输入回读未确立")
                except (ValueError, TypeError) as exc:
                    remote_bench = dict(success=False, error=str(exc), reference_error=str(exc), reference_identity_unknown=True,
                        **{k: remote_bench[k] for k in (TRANSPORT_KEY, "session_state", "session_phase", "request_id", "bytes_sent")
                           if isinstance(remote_bench, dict) and k in remote_bench})
        bench = (self.backend.benchmark(kernel_code, inputs, warmup=2, repeat=5)
                 if remote_bench is None else remote_bench)
        return bench, remote_bench

    def _measure_case_speedup(self, kernel_code: str, inputs: dict, task: TaskSchema,
                              **context) -> dict:
        """Non-primary cases use the exact same L3/Oracle policy as the primary.

        Only recursive ledger publication and the *global* objective verdict
        are skipped. A failed/unknown case keeps diagnostics, never a made-up 0.
        """
        try:
            result = self._verify_performance(kernel_code, inputs, 0.0, task,
                                              _case_measurement=True, **context)
            row = dict(result.metrics)
            if "speedup_vs_baseline" not in row:
                row.update(speedup_vs_baseline=None, reason=result.message)
            return row
        except Exception as exc:
            return {"speedup_vs_baseline": None, "reason": f"逐 case 测量抛异常: {exc}"}

    def _publish_case_ledger(self, metrics: dict, kernel_code: str, test_inputs,
                             task: TaskSchema, **context) -> Optional[str]:
        """把**逐 case** 的加速比账本写进这次 L3 的 metrics（`metrics["cases"]`）

        §6.2 条款 2/3 的证据面要求「本次评了哪几条 case、各自多少」。在此之前 L3 明
        说「多组输入套件时，性能基准使用第一组输入」（§3.4 的欠账）—— 于是「全部形状」
        只剩 L2 的正确性，加速比只有第一条，而 §16 反模式点的正是「用单形状的加速比
        作为成功标准」。现在: 主 case 用**已经算出来的那个数**（同一件事不量两遍），
        其余每组各量一次。

        ⚠️ 只在**有主 case 读数**的两条出口调用（远端 / 本机有参考实现）: 没有参考
        实现时这条路判的是绝对 GFLOPS，逐 case 的加速比在那里没有指称对象 —— 硬造
        一份全是 `None` 的账本只会让「没量到」看起来像「量过了」。

        ⚠️ case 身份 = **输入形状签名**（`contract.case_key`）⇒ 两组**同形状**的输入
        是**同一条 case**：账本按 case 去重（后一组不重复测）；而 L2 的逐组正确性照旧
        逐组做（正确性按组、身份按签名，两件事本来就不是一个粒度）。
        """
        input_sets = test_inputs if isinstance(test_inputs, list) else [test_inputs]
        # 返回值 = **主 case 的身份**（单组输入时不写账本，但调用方要拿它去目标函数里找权重）
        first = case_key(input_sets[0])
        primary = str(first) if first else None
        if len(input_sets) < 2:
            from ..core.contract import freeze_from_task
            from ..core.workload import apply_qualification
            try:
                contract = freeze_from_task(task)
                if contract.required_cases or contract.case_weights or contract.max_regression:
                    metrics["cases"] = {primary: copy.deepcopy(metrics)}
                    apply_qualification(metrics, contract, primary)
            except Exception as exc:
                metrics["workload_qualification_error"] = str(exc)
                metrics["measurement_can_gate"] = False
            return primary
        ledger = {}
        if first:
            ledger[first] = copy.deepcopy(metrics)
            ledger[first]["basis"] = "主 case: 本次 L3 那个读数（未重复测）"
        for index, inputs in enumerate(input_sets[1:], start=1):
            key = case_key(inputs)
            if not key or key in ledger:
                continue
            ledger[key] = self._measure_case_speedup(kernel_code, inputs, task, **context)
            if ledger[key].get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN:
                metrics.update({k: v for k, v in ledger[key].items() if k in
                                (TRANSPORT_KEY, "session_state", "session_phase", "request_id", "bytes_sent")})
                for remaining in input_sets[index + 1:]:
                    missing = case_key(remaining)
                    if missing and missing not in ledger:
                        ledger[missing] = {"speedup_vs_baseline": None, "reason": "传输结果未知，未继续测量"}
                break
            if ledger[key].get("reference_identity_unknown"):
                metrics.update(reference_identity_unknown=True, reference_error=ledger[key]["reason"])
                for remaining in input_sets[index + 1:]:
                    missing = case_key(remaining)
                    if missing and missing not in ledger:
                        ledger[missing] = dict(speedup_vs_baseline=None, reason="参考身份未知，未继续测量")
                break
        if len(ledger) > 1:
            metrics["cases"] = ledger
        from ..core.contract import freeze_from_task
        from ..core.workload import apply_qualification
        try:
            contract = freeze_from_task(task)
            apply_qualification(metrics, contract, primary)
        except Exception as exc:
            # An unfreezable declared workload must not inherit the primary mask.
            metrics["workload_qualification_error"] = str(exc)
            from ..decide.eligibility import ALL_FALSE
            metrics["eligibility"] = ALL_FALSE.as_dict()
            metrics["measurement_can_gate"] = False
        return primary

    @staticmethod
    def _unknown_transport_result(level, payload, metrics=None):
        diagnostic = {k: payload[k] for k in
                      (TRANSPORT_KEY, "session_state", "session_phase", "request_id", "bytes_sent") if k in payload}
        return VerificationResult(level, False, message="传输结果未知: " + payload.get("error", ""),
                                  metrics={**(metrics or {}), **diagnostic})

    def _verify_performance(self, kernel_code: str, test_inputs, target_gflops: float,
                            task: TaskSchema,
                            environment_drift: Optional[list] = None,
                            heartbeat=None, heartbeat_expected: bool = False,
                            static_guards: Optional[bool] = None,
                            correctness_verified: Optional[bool] = None,
                            _case_measurement: bool = False,
                            ) -> VerificationResult:
        min_required = task.constraints.min_speedup
        print(f"  [Verifier] L3: 性能验证 (要求加速比 >= {min_required:.2f}x vs baseline)...")
        # 多组输入套件时，性能基准使用第一组输入
        bench_inputs = test_inputs[0] if isinstance(test_inputs, list) else test_inputs
        # 一次 case 的「候选 + 基线」测量只有这一处（`_bench_case`）—— 逐 case 账本
        # （`_publish_case_ledger` / `_measure_case_speedup`）也走同一条路: 两处协议
        # 一漂（repeat / 远端优先 / 基线选择）就会在同一张表里比出两种数。
        bench, remote_bench = self._bench_case(kernel_code, bench_inputs, task)

        if bench.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN:
            return self._unknown_transport_result(VerificationLevel.L3_PERFORMANCE, bench)

        if remote_bench is not None and not remote_bench.get("success", False):
            if remote_bench.get("reference_identity_unknown"):
                return self._reference_unknown(remote_bench.get("reference_error", "unknown"),
                    VerificationLevel.L3_PERFORMANCE, remote_bench)
            return VerificationResult(
                level=VerificationLevel.L3_PERFORMANCE, passed=False,
                message=f"远程性能测试失败: {remote_bench.get('error', '')}",
                error_log=remote_bench.get("error", ""),
            )

        # 按**拿到的载荷**分派, 而不是按后端类型: `candidate` 那一层只在同机基准协议
        # 的返回里存在 (doc 49 §3.2 B4 —— 旧代码按 `_is_remote` 分派, 于是内层降级成
        # LocalExecutor 时仍走 remote 分支, 读一个形状不对的字典)。
        if remote_bench is not None:
            candidate = bench.get("candidate", {}) if remote_bench is not None else bench
            actual_gflops = bench.get("gflops", 0)
            exec_time = candidate.get("avg_time_ms", bench.get("avg_time_ms", 0))
            bandwidth = bench.get("memory_bandwidth_gbs", 0)
        else:
            # 本地 benchmark 也拥有这一批预热后的吞吐/带宽读数。不能拿另一次
            # 冷执行（尤其是受控子进程的首次 BLAS 调用）去覆盖它们，否则报告的
            # 延迟与 GFLOPS 属于两次不同实验，Golden 会把冷启动误报成性能回归。
            # 老后端没发布这些字段时才保留 execute 的兼容回退；0 是有效读数。
            exec_result = None
            if "gflops" not in bench or "memory_bandwidth_gbs" not in bench:
                exec_result = self.backend.execute(kernel_code, bench_inputs)
            actual_gflops = (bench["gflops"] if "gflops" in bench
                             else exec_result.gflops)
            exec_time = bench["avg_time_ms"]
            bandwidth = (bench["memory_bandwidth_gbs"] if "memory_bandwidth_gbs" in bench
                         else exec_result.memory_bandwidth_gbs)

        candidate_metrics = bench.get("candidate", {}) if remote_bench is not None else bench
        kernel_min_ms = candidate_metrics.get("median_time_ms") or candidate_metrics.get("min_time_ms") or exec_time

        # ---- Roofline 分析: 基于实测数据判断真实瓶颈 ----
        roofline = self._roofline_analysis(
            task=task, bench_inputs=bench_inputs,
            actual_gflops=actual_gflops, actual_bandwidth_gbs=bandwidth,
        )
        bound = roofline["bound_type"]  # 替代原有的名字匹配 bound

        metrics = {"avg_time_ms": exec_time, "min_time_ms": kernel_min_ms,
                   **({"input_transfer": bench["input_transfer"]} if "input_transfer" in bench else {}),
                   "gflops": round(actual_gflops, 4),
                   # 编译产物的事实 (§10.3 条件 2 的真源): **runner 报了才带上** ——
                   # 三态, 不许在这里补默认值 (缺失 ≠ 0 ≠ 没有上限)。控制侧此前是手写
                   # 白名单, 于是 runner 报的新键会被静默丢掉 (实测: 远端已报、L3 无键)。
                   # 只放与我们判据有关的那几个, 不把 runner 的整个信封灌进 metrics。
                   **{k: bench[k] for k in COMPILE_FACT_KEYS if k in bench},
                   "memory_bandwidth_gbs": round(bandwidth, 4),
                   "bound_type": bound, "primary_metric": "speedup_vs_baseline",
                   "backend": str(self.backend)}
        if "reference_receipt" in bench:
            metrics["reference_receipt"] = bench["reference_receipt"]
            metrics["reference_contract_hash"] = as_reference(task.constraints.reference_contract).digest()
            metrics["frozen_contract_hash"] = self._reference_bindings[task.task_id]
        if "timing_receipt" in bench:
            metrics["timing_receipt"] = bench["timing_receipt"]
        # 从 benchmark 结果传递额外字段。
        #
        # ⚠️ 必须 bench 优先、candidate 兜底, 两个都要查。远程路径上这两层的分工不同:
        # `mode/device/gpu_util_pct/selected_gpu` 在 bench 顶层, 而
        # `runs/max_time_ms/p25/p75/dispersion_*` 在 bench["candidate"] 里。原先只有
        # 一个循环、只查 bench, 于是真机跑出来是 `runs=None` —— runner 特意报出实际
        # 采样轮数 (快核会被自动加采样) 就是为了让下游知道这个数字有多少样本撑着,
        # 结果搬运环节把它丢了。这正是缺口 1 那一类「算了没人读」的失效方式,
        # 由 docs/research/33 的真机验证抓出来。
        for extra_key in ("mode", "language", "device", "runs", "runs_requested",
                          "max_time_ms",
                          "p25_time_ms", "p75_time_ms",
                          "dispersion_iqr_over_median", "dispersion_max_over_median",
                          "gpu_util_pct", "gpu_util_samples", "gpu_other_procs",
                          "gpu_harness_peers", "gpu_harness_peer_scans",
                          "selected_gpu",
                          # D2: 窗口边界上的 SM/显存时钟与 throttle 位图。搬运是必须的
                          # —— 缺口 1 的教训就是「runner 报了但控制端没人搬, 于是没人读」。
                          "sm_clock_mhz_before", "sm_clock_mhz_after",
                          "sm_clock_max_mhz_before", "sm_clock_max_mhz_after",
                          "mem_clock_mhz_before", "mem_clock_mhz_after",
                          "throttle_bits_before", "throttle_bits_after",
                          # spine-review P0-2: 记录**携带**的契约哈希必须能搬进来,
                          # 否则 oracle 那边只剩「现算一个」这一条路, 校验恒真。
                          # 载荷带头哈希 = 这条读数出自某次已在别处烙过契约的测量,
                          # 与当前契约比对才有意义 (不匹配 → 拒评)。
                          "contract_hash"):
            if extra_key in bench:
                metrics[extra_key] = bench[extra_key]
            elif extra_key in candidate_metrics:
                metrics[extra_key] = candidate_metrics[extra_key]
        # ⚠️ 故意**不**搬 min_time_ms: metrics["min_time_ms"] 从建表那一刻起装的就是
        # kernel_min_ms (= median 优先), 搬进来会把真最小值盖掉, 让这个数字的含义在
        # 远程/非远程两条路径上分叉。这里补一个名字说清楚的 median 键, 让「加速比
        # 建立在哪个统计量上」无歧义 —— 改 metrics["min_time_ms"] 的含义会波及
        # kernel_db 里的历史快照与既有下游。
        metrics["median_time_ms"] = kernel_min_ms

        # ---- §11-4: 这一轮实际用的那张卡的**观测租约** ----
        #
        # 设计原话: 「租不到独占卡时, 记录 external_load 并在 evidence 中显式标注; 观测降为
        # DEGRADED, 而不是失败」。前半句(记录)在这里落地, 后半句(降级)是
        # `_assess_measurement` -> `measurement.assess` 的职责 —— 它读**同一个**
        # `gpu_util_pct`。租约只给证据, 不给结论: 两处各自判一次, 阈值迟早分叉 (>0 vs >5%)。
        lease = observed_lease(metrics, source="L3 载荷 (bench 顶层 + candidate)")
        lease_text, lease_state = lease_evidence(lease)
        metrics["device_lease"] = dict(lease.as_dict(), evidence=lease_text,
                                       competition=lease_state)
        metrics["device_lease_text"] = lease_text

        # 静态守卫的适用性**发布** (CHECKLOG I65): `_assess_measurement` 是另一个方法,
        # 它拿不到这里的局部量 —— 所以判定要读的那一份必须先发布进 metrics (与
        # `arm_device_labels`/`arm_windows` 同一写法)。三态: `None` (没观测) **不写键**,
        # 「观测了但没有值」与「没观测」在产物里必须分得开。
        if static_guards is not None:
            metrics["static_guards"] = bool(static_guards)

        # 正确性凭据 (设计 §14-10 / §20.4): 同上 —— 发布事实, 判据在资格层
        # (`eligibility.derive_eligibility` 规则 3 那一族: 没有指称对象的数全资格 False)。
        # 为什么必须由调用方传: L3 观测可以由两条路产生, 而 `verify_l3_only` **绕开了 L2** ——
        # 「这个核是对的」是调用方手里的事实, 不是本方法能假设的。`None` = 没带凭据 ⇒ 不写键
        # (老产物/回放脚本不因此被追溯改判), 只有明确的 `False` 才拒。
        if correctness_verified is not None:
            metrics["correctness_verified"] = bool(correctness_verified)

        # ---- 注入 Roofline 指标, 供 SearchAgent 做结构化优化决策 ----
        metrics.update({
            "roofline_analysis": roofline,
            "hw_peak_gflops": roofline["hw_peak_gflops"],
            "hw_peak_bandwidth_gbs": roofline["hw_peak_bandwidth_gbs"],
            "gflops_utilization_pct": roofline["gflops_utilization_pct"],
            "bandwidth_utilization_pct": roofline["bandwidth_utilization_pct"],
            "arithmetic_intensity": roofline["arithmetic_intensity"],
            "ridge_point": roofline["ridge_point"],
            "roofline_bound": roofline["roofline_bound"],
            "hw_name": roofline["hw_name"],
            "hw_compute_capability": roofline["hw_compute_capability"],
            # 代理剪枝的上界 (0 = 不知道)。跟着 roofline 一起发布, 因为它们的来源
            # 是同一个 caps; 分成两条路发布迟早会出现「峰值有、上界没有」的读数。
            "shared_mem_per_sm_kb": roofline.get("shared_mem_per_sm_kb", 0),
        })

        if roofline["hw_peak_gflops"] > 0 and roofline["hw_peak_bandwidth_gbs"] > 0:
            compute_label = roofline.get("compute_used", "fp32_cuda_core")
            # 分母的出处必须出现在分母被读到的那一行: 判停出口 (REFERENCE_CEILING_PCT)
            # 直接拿达成率当证据, 而「这个 peak 凭什么」以前只在源码注释里, 日志里看不到。
            print(f"  [Verifier] Roofline: {roofline['roofline_bound']} "
                  f"(AI={roofline['arithmetic_intensity']:.2f} FLOP/Byte, "
                  f"peak={roofline['hw_peak_gflops']/1000:.1f} TFLOPS ({compute_label}|"
                  f"分母[{roofline['peak_table_key']}] {roofline['peak_source']}"
                  f"{'' if roofline.get('reference_lower_bound') else '; ⚠️未实测⇒天花板出口关闭'}), "
                  f"GFLOPS利用率={roofline['gflops_utilization_pct']:.1f}%, "
                  f"带宽利用率={roofline['bandwidth_utilization_pct']:.1f}%)")

        # 远程模式必须使用同机/同卡 baseline；禁止退回控制端计时。
        # 选哪个基线的策略在 baseline_plan() (随请求下发), 这里只认远端回报的
        # `baseline_selected` —— 远端现在**只测那一个**, 本地再重排等于假设其余基线
        # 也被测过, 而那正是 D3 要省的 GPU 时间。
        if remote_bench is not None:
            baselines = bench.get("baselines", {})
            paired = bench.get("paired", {}) or {}
            baseline_name = bench.get("baseline_selected")
            baseline = baselines.get(baseline_name) if baseline_name else None
            if baseline:
                baseline_ms = baseline.get("median_time_ms", baseline.get("min_time_ms", 0))
                speedup = baseline_ms / kernel_min_ms if kernel_min_ms > 0 else 0.0
                # 美化显示名 (仅 matmul 用 cuBLAS, 其他算子用 PyTorch Eager)
                is_matmul = "matmul" in task.source.kernel_type.lower() or "gemm" in task.source.kernel_type.lower()
                if baseline_name == "pytorch_eager_fp16" and is_matmul:
                    display_name = "cuBLAS(FP16)"
                elif baseline_name == "pytorch_eager_tf32" and is_matmul:
                    display_name = "cuBLAS(TF32)"
                elif baseline_name == "pytorch_eager" and is_matmul and bench.get("device", "").startswith("cuda"):
                    display_name = "cuBLAS(FP32)"
                elif baseline_name == "pytorch_eager" and bench.get("device", "").startswith("cuda"):
                    display_name = "PyTorch Eager(CUDA)"
                else:
                    display_name = baseline_name
                # 逐臂身份 → 配对口径 (§11-2 / CHECKLOG I47)。`arm_identity` 原样落盘
                # (可审计), 标签那对与 scope 是同一个函数算出来的。
                _ident = paired.get("arm_identity") or {}
                _cand_dev, _base_dev, _pair_scope = observed_pairing(
                    _ident.get("candidate"), _ident.get("baseline"))
                # 逐臂**测量窗口** (§11-2 的「同一时间窗」那一半, CHECKLOG I47 的窗口半格)。
                # runner 在一次 `benchmark_with_baselines` 里按块交错量两臂 ⇒ 每臂的样本
                # 各自带着窗口 id 回来; 哪天某臂被复用 (缓存/短路) 带回来的就是旧窗。
                _wins = paired.get("arm_windows") or {}
                _cand_win, _base_win, _win_scope = observed_windows(
                    _wins.get("candidate"), _wins.get("baseline"))
                metrics.update({
                    "baseline_type": baseline_name,
                    "baseline_display": display_name,
                    "baseline_time_ms": round(baseline_ms, 6),
                    "speedup_vs_baseline": round(speedup, 4),
                    # 三元组原文, 落盘可直接比 (LeetCUDA bench 的 "%.1f/%.1f (%.2fx)")
                    "ratio_triple": f"{kernel_min_ms:.4g}/{baseline_ms:.4g} ({speedup:.2f}x)",
                    "min_speedup_required": min_required,
                    # 这一轮**判的是哪个量** (#87 第 3 片)。三条比对路共用 `min_speedup`
                    # 这一个旋钮, 但前两条拿它当「快多少倍」的地板、第三条拿它乘在绝对目标上
                    # 当系数 —— 只比数值会把一次换了量纲的漂移读成「尺没变」。
                    "perf_criterion": PERF_CRITERION_SPEEDUP,
                    # 配对口径由**逐臂观测**算出 (CHECKLOG I47)。此前这行写死
                    # `"same_remote_device"` —— 键名正是 §11-2 要的那件事, 而当时没有
                    # 任何观测在背后 (全仓只有一个测试在读它)。现在三种取值都写得出:
                    # 有观测且一致 / 观测到不同卡 / 没观测 (老 payload 或非远端路径)。
                    "measurement_scope": _pair_scope,
                    # 窗口与设备**分开两个键**: 一个是「在哪张卡上」, 一个是「在哪扇窗里」,
                    # 混成一个 scope 串会让「同卡但跨窗」这种情形两种解释都说得通。
                    "arm_windows": _wins,
                    "measurement_window_scope": _win_scope,
                    "candidate_median_time_ms": candidate_metrics.get("median_time_ms"),
                    "baseline_metrics": baselines,
                    # 交错配对的信号: 逐样本比值的 median 与其散布。判定仍走上面那个
                    # ratio-of-medians (不静默换口径), 这两个先作为旁证记录。
                    "paired_speedup_median": paired.get("speedup_paired_median"),
                    "paired_speedup_dispersion": paired.get("speedup_paired_iqr_over_median"),
                    "paired_samples": paired.get("paired_runs"),
                    "paired_block_samples": paired.get("paired_block_samples"),
                    # 逐块比值序列: D5「什么时候可以停」唯一读得动的粒度。本轮只搬运不判定
                    # (false-stop 率还没量出来, 见 docs/research/46)。
                    "block_ratio_medians": paired.get("block_ratio_medians"),
                    # 峰值显存: 与加速比**并列**的第二项观测, 不参与判定 (本机还没有
                    # 「我们的核会增加多少」的分布, 先攒数 —— 与 D2 的 SM 时钟同一处置)。
                    # 外部依据: KernelBench-Verified 报最好模型 28% 的核增加峰值显存。
                    "candidate_peak_memory_bytes": paired.get("candidate_peak_bytes_extra"),
                    "baseline_peak_memory_bytes": paired.get("baseline_peak_bytes_extra"),
                    "peak_memory_ratio": paired.get("peak_memory_ratio"),
                    "baseline_attempts": bench.get("baseline_attempts") or [],
                    # 远端逐臂计时当时的卡身份 (原样保留: 报告/复盘要能看见「哪张卡」,
                    # 而不是只看见一个结论位), 以及喂给 oracle 的那对标签。
                    "arm_identity": _ident,
                    "arm_device_labels": {"candidate": _cand_dev,
                                          "baseline": _base_dev},
                })
                # §16 收口（CHECKLOG I95）: 性能面**判的是什么**不再在这里定 —— 这里只立
                # 「前置条件（窗口/配对/污染）都没否掉」，判决在返回前那一处算
                # （`_apply_performance_verdict`: 声明了目标函数就判目标函数，没声明退回
                # 主 case 并把「§16 未收口」写进产物）。
                passed = True
                if _win_scope == WINDOW_DIFFERENT:
                    # 跨窗与跨卡同一族: 「同一时间窗」是 §11-2 给加速比立的**指称前提**
                    # (热/时钟状态可比), 观测到两臂不在同一扇窗里 ⇒ 这个数没有指称对象,
                    # 「通过」也不许立。判据是**观测到的**不同窗, 不是「没观测到同窗」
                    # —— 后者 (`unobserved`) 不拒评, 与设备那半同一边界。
                    passed = False
                if _pair_scope == SCOPE_CROSS:
                    # 跨卡加速比**没有指称对象** (§11-2 原句: 「A 卡基线 vs B 卡候选的
                    # 『加速比』直接作废」) —— 与 §4.3 规则 3 同一族, 所以「通过」这个
                    # 结论也不许立: passed=False。
                    # 资格位那件事仍**只**由 oracle 判 (它算的是「这个数被允许造成什么
                    # 后果」); 这里判的是另一件事 ——「passed 有没有对象」(与
                    # `l3_pass_is_refused` 刻意不折进 `can_gate` 是同一处分)。两处读的是
                    # 同一对标签 (`observed_pairing`), 所以不可能分叉。
                    passed = False
                # 三元组 (LeetCUDA bench 同款): 候选/基线 (倍率)。只报一个 1.04x
                # 时读者看不见两边的绝对量级 —— 「0.01/0.0096 (1.04x)」与
                # 「10.4/10.0 (1.04x)」是两回事, 前者可能整个被 launch 地板吞掉。
                ratio_triple = (f"{kernel_min_ms:.4g}/{baseline_ms:.4g} ({speedup:.2f}x)")
                # 这句话以前写死「同机同卡」—— 而 `measurement_scope` 现在可能是
                # 「未观测」或「跨卡」。message 是唯一保证被下游 (Fixer/Search prompt、
                # 报告、人眼) 看到的通道, 所以它**不许比证据多说一个字**。
                _scope_note = {
                    SCOPE_SAME: "同机同卡",
                    SCOPE_UNOBSERVED: "配对未观测",
                    SCOPE_CROSS: "配对=跨卡⇒作废",
                }.get(_pair_scope, "配对未观测")     # 未知取值按最弱的口径写
                msg = (f"加速比 {ratio_triple} vs {display_name} "
                       f"({_scope_note} median, 要求 >= {min_required:.2f}x, {bound}-bound)")
                _cp = metrics["candidate_peak_memory_bytes"]
                _bp = metrics["baseline_peak_memory_bytes"]
                if isinstance(_cp, int) and isinstance(_bp, int):
                    # 报绝对值而不是只报倍数: 小 shape 上「×4」可能只是 0.2→0.8 MiB,
                    # 而大 shape 上 ×1.2 才是真吃掉几个 GB。
                    msg += (f"; 每次调用额外峰值显存 候选 {_cp / 1048576:.1f} MiB / "
                            f"基线 {_bp / 1048576:.1f} MiB "
                            f"(×{metrics['peak_memory_ratio']})")

                # 共享机器上基准可信度的两个信号, 强度不同, 分开记录:
                #   util_before > 5%  → 硬污染, 测量时确实在争用, 加速比不可比
                #   other_procs > 0   → 仅提示卡被共享 (这台机器上恒为真), 不单独判污染
                # 若把"有进程"也算污染, 每次运行都会被标记, 信号就失效了。
                util = bench.get("gpu_util_pct")
                others = bench.get("gpu_other_procs")
                metrics["gpu_shared_with_others"] = bool(others)
                msg, assessment = self._assess_measurement(
                    metrics, msg, kernel_min_ms, gpu_util_pct=util, task=task,
                    environment_drift=environment_drift,
                    heartbeat=heartbeat, heartbeat_expected=heartbeat_expected)
                passed = self._refuse_l3_pass(passed, assessment)
                if not metrics.get("benchmark_contaminated") and others:
                    msg += f" (注: 卡与 {others} 个其他进程共享, 瞬时 util={util}%)"

                # 账本先发布（性能判决要读它），判决只算一次、只在这一处（两条路同一份）。
                _primary = (case_key(bench_inputs) if _case_measurement else
                    self._publish_case_ledger(metrics, kernel_code, test_inputs, task,
                        environment_drift=environment_drift, heartbeat=heartbeat,
                        heartbeat_expected=heartbeat_expected, static_guards=static_guards,
                        correctness_verified=correctness_verified))
                if metrics.get("reference_identity_unknown"):
                    return self._reference_unknown(metrics["reference_error"], VerificationLevel.L3_PERFORMANCE, metrics)
                if metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN:
                    return self._unknown_transport_result(VerificationLevel.L3_PERFORMANCE, metrics, metrics)
                if not _case_measurement:
                    passed = self._apply_performance_verdict(passed, metrics, task, speedup,
                                                             primary_key=_primary)
                print(f"  [Verifier] L3 {'PASSED' if passed else 'FAILED'}: {msg}")
                return VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                                          passed=passed, message=msg, metrics=metrics)

            head, full = failure_pair(bench.get("error", ""))
            return VerificationResult(
                level=VerificationLevel.L3_PERFORMANCE, passed=False,
                message=("远程性能测试未返回可用 baseline: "
                         f"{head or '远端没有报 error, 也没报 baseline_selected'}"),
                error_log=(f"{full}\n远端按偏好顺序尝试过: {bench.get('baseline_attempts')}"
                           "\n若远端确实跑了基线却没报 baseline_selected, 说明部署的 "
                           "run_kernel.py 与 verifier 不同步 (deploy 该按 hash 覆盖)。"),
            )

        # 非远程后端使用本机 reference 计时。
        baseline = self._measure_baseline(task, bench_inputs)
        if baseline is not None:
            speedup = baseline["baseline_time_ms"] / kernel_min_ms if kernel_min_ms > 0 else 0.0
            metrics.update({
                "baseline_type": baseline["baseline_type"],
                "baseline_time_ms": round(baseline["baseline_time_ms"], 4),
                "speedup_vs_baseline": round(speedup, 4),
                "min_speedup_required": min_required,
                # 跨窗复用的**观测** (不是判决): 控制端参照臂的缓存命中。非远程路径
                # 的配对窗口没有设备含义 (§11-2 那句话说的是设备租约臂), 所以它不进
                # `pairing_conflict`; 但它今天是**看得见**的, 不再是一个沉默的捷径。
                "baseline_reused": bool(baseline.get("baseline_reused")),
                "perf_criterion": PERF_CRITERION_SPEEDUP,
            })
            # 同远程那条路: 判的是什么在返回前那一处算（§16 收口, I95）。
            passed = True
            base_ms = baseline["baseline_time_ms"]
            ratio_triple = f"{kernel_min_ms:.4g}/{base_ms:.4g} ({speedup:.2f}x)"
            # 基线是**这次量的**还是**缓存里那条**: 复用时消息里必须写出来 (§11-2 的
            # 「同一时间窗」在控制端参照臂上不成立 ⇒ 读者有权知道比较是跨窗的)。
            _reused = bool(baseline.get("baseline_reused"))
            msg = (f"加速比 {ratio_triple} vs {baseline['baseline_type']} "
                   f"(要求 >= {min_required:.2f}x, {bound}-bound)"
                   + ("  ⏳ 基线复用本轮之前的测量 (跨窗, §11-2 的窗口前提不成立)"
                      if _reused else ""))
            msg, assessment = self._assess_measurement(
                metrics, msg, kernel_min_ms,
                # ⚠️ 非远程这条路**原先不传 util**, 于是同一个 `gpu_util_pct` 有两条命运:
                # 远程路径上它把档位打到 DEGRADED, 本地路径上它只被搬进产物、**没人读**
                # (而本地路径恰恰可能是真卡: `RemoteExecutor(mode="local")` 就在这台机器上
                # 跑)。于是「这张卡当时在被别人用」这件事在两条路上的后果不同 ——
                # 这正是「一个键两种含义」那一族。判据只有一处 (`measurement.assess`),
                # 两条路都必须把**产物里记的那一份**喂给它。
                gpu_util_pct=metrics.get("gpu_util_pct"), task=task,
                environment_drift=environment_drift,
                heartbeat=heartbeat, heartbeat_expected=heartbeat_expected)
            passed = self._refuse_l3_pass(passed, assessment)
            _primary = (case_key(bench_inputs) if _case_measurement else
                    self._publish_case_ledger(metrics, kernel_code, test_inputs, task,
                        environment_drift=environment_drift, heartbeat=heartbeat,
                        heartbeat_expected=heartbeat_expected, static_guards=static_guards,
                        correctness_verified=correctness_verified))
            if metrics.get("reference_identity_unknown"):
                return self._reference_unknown(metrics["reference_error"], VerificationLevel.L3_PERFORMANCE, metrics)
            if metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN:
                return self._unknown_transport_result(VerificationLevel.L3_PERFORMANCE, metrics, metrics)
            if not _case_measurement:
                passed = self._apply_performance_verdict(passed, metrics, task, speedup,
                                                         primary_key=_primary)
            print(f"  [Verifier] L3 {'PASSED' if passed else 'FAILED'}: {msg}")
            return VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                                      passed=passed, message=msg, metrics=metrics)

        # 回退判定: 无参考实现的算子使用绝对 GFLOPS 目标 (legacy)
        achieve = actual_gflops / target_gflops if target_gflops > 0 else 0.0
        metrics.update({"target_gflops": target_gflops, "achieve_ratio": round(achieve, 4),
                        # 第三条比对路。这里也发布 `min_speedup_required` —— 那个旋钮在这一条
                        # 路上**确实被用了**, 不报它就是「把没报当没用到」。但它在这里的角色是
                        # **乘在绝对目标上的系数**, 不是前两条路那个「快多少倍的地板」; 两者数值
                        # 相同而含义不同, 所以必须有 `perf_criterion` 一起在场, 否则跨轮读数会
                        # 把「从比加速比改成比 GFLOPS」读成「尺没变」(#87 第 3 片)。
                        "min_speedup_required": min_required,
                        "perf_criterion": PERF_CRITERION_ABSOLUTE_GFLOPS})
        passed = target_gflops <= 0 or actual_gflops >= target_gflops * min_required
        msg = (f"无 baseline 参考, 按绝对目标判定: {actual_gflops:.2f} GFLOPS "
               f"(目标 {target_gflops * min_required:.2f})")
        msg, assessment = self._assess_measurement(
            metrics, msg, kernel_min_ms, task=task,
            environment_drift=environment_drift,
            heartbeat=heartbeat, heartbeat_expected=heartbeat_expected)
        passed = self._refuse_l3_pass(passed, assessment)
        if not _case_measurement and (task.constraints.case_weights or task.constraints.aggregate
                                     or task.constraints.max_regression or task.constraints.required_cases):
            # Declaring a speedup workload cannot silently turn into absolute
            # GFLOPS success merely because its baseline is unavailable.
            passed = self._apply_performance_verdict(False, metrics, task, None,
                                                     primary_key=case_key(bench_inputs))
        print(f"  [Verifier] L3 {'PASSED' if passed else 'FAILED'}: {msg}")
        return VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                                  passed=passed, message=msg, metrics=metrics)

    def _apply_performance_verdict(self, prior: bool, metrics: dict, task,
                                   primary_speedup, primary_key=None) -> bool:
        """§16 收口: 性能面的**成功标准**只在这一处算（CHECKLOG I95）

        `prior` = 前置条件那几条拒绝（窗口/配对/污染/测量档位）的结论 —— 它们与「判的是
        哪个量」无关，照旧在调用点算; 本方法只算**判的是什么**，并把结论发布进
        `metrics["performance_verdict"]`（含依据；`basis` 是这一片唯一的新不变量）。

        判据面在 `eligibility.performance_verdict`（纯函数，一个事实一处生产）: 契约声明了
        目标函数 ⇒ 判目标函数（未确立 ⇒ **不通过**，fail-closed）; 没声明 ⇒ 退回主 case 的
        加速比，并把「§16 没对它收口」写进依据。

        契约冻不住时读**任务里的原始声明**（只为了不把「声明了目标函数」看成「没声明」——
        那会把一次 fail-closed 变成一次静默放水），并在依据里登记；`contract_frozen` 那一格
        让产物自己说得清判据是从哪来的。
        """
        from ..core.contract import freeze_from_task
        from ..decide.eligibility import (case_speedups, performance_verdict,
                                  regression_over_limit)

        min_required = float(getattr(getattr(task, "constraints", None),
                                    "min_speedup", 0.0) or 0.0)
        contract = None
        if task is not None:
            try:
                contract = freeze_from_task(task)
            except Exception:  # noqa: BLE001 —— 契约不可判定 ⇒ 读任务里的原始声明
                contract = None
        if contract is not None:
            weights, aggregate = tuple(contract.case_weights), str(contract.aggregate or "")
            limits = tuple(contract.max_regression)
        else:
            cons = getattr(task, "constraints", None)
            weights = getattr(cons, "case_weights", ()) or ()
            weights = tuple(weights.items()) if isinstance(weights, Mapping) else tuple(weights)
            aggregate = str(getattr(cons, "aggregate", "") or "")
            limits = getattr(cons, "max_regression", ()) or ()
            limits = tuple(limits.items()) if isinstance(limits, Mapping) else tuple(limits)
        speedups = case_speedups(metrics, case_key_hint=primary_key,
                                speedup_hint=primary_speedup)
        # §6.2 条款 3 的判据也在判据层（`regression_over_limit`）—— 这里只传结论。
        rol, rol_why = regression_over_limit(limits, speedups)
        passed, basis, why, value = performance_verdict(
            min_required, aggregate=aggregate, case_weights=weights,
            case_speedups=speedups, primary_speedup=primary_speedup,
            max_regression=limits, regression_over_limit_value=rol)
        if contract is None and task is not None:
            why += "（契约不可判定 ⇒ 判据面读的是任务里的原始声明）"
        metrics["performance_verdict"] = {
            "basis": basis, "passed": bool(passed), "value": value,
            "min_speedup": min_required, "why": why,
            "cases": len(speedups), "regression_over_limit": rol,
            "regression_why": list(rol_why), "contract_frozen": contract is not None,
            "aggregate": aggregate, "case_weights": dict(weights),
            "max_regression": dict(limits),
            "required_cases": list(contract.required_cases) if contract is not None else
                              list(getattr(getattr(task, "constraints", None), "required_cases", ()) or ()),
        }
        verdict_text = "过" if passed else "不过"
        print(f"  [Verifier] 性能结论 (§16): {basis} ⇒ {verdict_text} —— {why}")
        cons = getattr(task, "constraints", None)
        declared_scope = any(getattr(cons, k, None) for k in
                             ("required_cases", "case_weights", "aggregate", "max_regression", "cases", "shapes"))
        return (bool(prior) and bool(passed)
                and (not declared_scope
                     or metrics.get("measurement_can_gate") is True))

    @staticmethod
    def _refuse_l3_pass(passed: bool, assessment) -> bool:
        """出处不对的 L3 不许记成通过 (doc 49 §3.5 第 2/3 处 + §五 第 4 片验收)

        与 `can_gate` 分开: 档位 `uninformative` 只说明「这个数字撑不住结论」, 历史上
        它并不把 `passed` 拉成 False (launch 地板支配那一档的核可能确实是对的)。
        而**任务声明了要真卡、执行侧给不出**是另一回事 —— 那份加速比不是测得不准,
        是压根没测, 「通过」这个结论没有对象。
        """
        return False if (passed and assessment.l3_pass_is_refused) else passed

    def _measurement_launch_floor(self, metrics, heartbeat):
        if self._cuda_ruler_profile is None:
            return self._detect_theoretical_peak().get("launch_floor_us")
        from ..decide.cuda_ruler_profile import measurement_floor, validate_profile
        try:
            profile = validate_profile(self._cuda_ruler_profile)
            floor = measurement_floor(profile, heartbeat, metrics.get("timing_receipt"))
            metrics["cuda_ruler_profile"] = dict(protocol=profile["protocol"],
                hash=profile["hash"], identity_hash=profile["identity_hash"],
                calibration_hash=profile["calibration_hash"], channel=profile["channel"],
                policy=profile["policy"], launch_floor_us=floor,
                expected_good_ms=profile["expected_good_ms"])
            metrics["launch_floor_source"] = profile["protocol"]
            return floor
        except Exception as exc:
            metrics["ruler_identity_error"] = str(exc)
            metrics["launch_floor_source"] = "calibrated-ruler-refused-no-fallback"
            return None

    def _assess_measurement(self, metrics: dict, msg: str, candidate_ms,
                            gpu_util_pct=None, task: Optional[TaskSchema] = None,
                            environment_drift: Optional[list] = None,
                            heartbeat=None, heartbeat_expected: bool = False):
        """写入测量有效性三态判定, 返回 `(追加了原因说明的 msg, assessment)`

        缺口 1 的失效方式不是「没算」而是「算了没人读」: `benchmark_contaminated`
        早就写进 metrics, 但 conductor / fixer / agents 三个文件里 grep 零命中。
        所以这里做两件事:
          - metrics 里写三个 can_* 布尔, 供 Conductor 决定「这个数字能不能拿去做
            这个决定」—— 暴露用途而不是档位, 免得调用方各自重新解释一遍分档表;
          - 把结论写进 message。message 会进 Fixer 与 SearchAgent 的 prompt, 是
            唯一保证被下游看到的通道。

        出处那两条信号 (doc 49 §3.5 第 2/3 处) 读的是**已经搬进 metrics 的**
        `mode`/`device`, 而不是回头再问一次载荷: 判定看的必须就是产物里记的那份,
        否则「记录说 A、判定按 B」这种分叉永远查不出来。能力则一律走
        `self._caps` 的声明, 不从 `device`/`name` 的字符串去猜 (doc 49 §3.1)。
        """
        required = frozenset(task.target.required_caps) if task else frozenset()
        assessment = measurement.assess(
            median_time_ms=candidate_ms,
            launch_floor_us=self._measurement_launch_floor(metrics, heartbeat),
            ruler_identity_error=metrics.get("ruler_identity_error", ""),
            gpu_util_pct=gpu_util_pct,
            # 正确性凭据 (§14-10 / §20.4): 读**已经发布进 metrics 的那一份** (与
            # static_guards / arm_device_labels 同一写法) —— `_verify_performance` 是
            # 另一个方法, 它的局部量在这里根本看不见, 而 `observation_error` 会把那种
            # 漏接吞成一行日志。缺键 = 没带凭据 ⇒ 三态里的"未观测", 不改判。
            correctness_verified=metrics.get("correctness_verified"),
            # I68: harness 同卡声明从**已搬进 metrics 的**那份读 —— 判定看的必须
            # 就是产物里记的 (与 mode/device 同一条规矩), 不回头再问载荷。
            gpu_harness_peers=metrics.get("gpu_harness_peers"),
            dispersion=metrics.get("dispersion_iqr_over_median"),
            max_over_median=metrics.get("dispersion_max_over_median"),
            backend_caps=self._caps,
            required_caps=required,
            reported_mode=metrics.get("mode"),
            reported_device=metrics.get("device"),
            environment_drift=environment_drift,
        )
        metrics.update(assessment.as_metrics())
        metrics["benchmark_contaminated"] = assessment.contaminated
        # §13.5: 拿**这一次**已经付过钱的基线臂做一次参考下界标定 (零额外 GPU 秒,
        # 也不占 §10.2 那 5% 的标定额度 —— 那个额度留给心跳)。产物里永远留下裁决
        # (包括拒绝与拒绝的理由): 「没标定」与「没接线」必须分得开。
        self._publish_peak_probe(metrics)
        # 离散度判据的**唯一**来源 (spine-review P1-6)。原先两处调用点各自写
        # `assessment.dispersion is not None and validity is not VALID` —— 那个式子
        # 从不与任何 CV 阈值比较, 只是把「档位不是 VALID」重编码了一遍, 于是污染 /
        # 出处矛盾 / 漂移造成的降级全被贴上「离散度超标」标签 (名不副实),
        # 而且同一次调用里传了两遍。真正判过阈值的是 `measurement.assess`,
        # 结论现在以 `dispersion_exceeded` 暴露, 这里只搬运不重推。
        cv_exceeded = bool(assessment.dispersion_exceeded)
        # 五位资格 (设计 §4.3): 与三个 can_* 并列写进 metrics, 让 Conductor 不必自己
        # 重新解释分档表 —— 「资格决定这个数被允许造成什么后果」。
        #
        # ⚠️ **唯一来源是收口点** (下面的 oracle 调用), 这里只写 fail-closed 的 all-False:
        # 收口点没跑成, 就不许有任何一位自称有资格 (**fail-closed**, spine-review P0-3:
        # 原先 `except Exception: pass` 让守卫出 bug 时「守卫消失并放行」)。
        #
        # 为什么**不再**在这里自己推一份 (2026-10-01 真机 §14-8 抓到): 那次推导拿不到
        # 心跳窗口, 于是同一个 metrics 里 `ruler_broken=True` 与 `can_persist=True`
        # 并排出现 —— 而入库判据读的正是 `metrics["eligibility"]`, 于是「关掉心跳该窗口
        # 不写长期记忆」在真机路径上**没有生效** (那次没入库只是因为加速比 0.42x 被另一道
        # 门偶然挡住了)。两份推导的差别只在心跳, 所以收口点的答案**严格更知情**: 同一组
        # 输入 + 心跳, 且用的就是同一个 `derive_eligibility`。
        from ..decide.eligibility import ALL_FALSE
        metrics["eligibility"] = ALL_FALSE.as_dict()

        # ── Oracle 唯一收口点 (spine-review P0-1 / P0-2 / 设计 §13.4) ──
        # 每个 L3 读数都产一条 Observation: 资格决定它被允许造成什么后果。
        self._publish_oracle_observation(
            metrics, assessment, task, cv_exceeded=cv_exceeded,
            heartbeat=heartbeat, heartbeat_expected=heartbeat_expected,
            required=required)
        if assessment.contaminated:
            metrics["contamination_reason"] = f"测量前 util={gpu_util_pct}%"
        if assessment.reasons:
            msg += (f"  ⚠️ 测量有效性={assessment.validity.value}: "
                    + "; ".join(assessment.reasons))
        if assessment.l3_pass_is_refused:
            # 成因由 `assessment` 自己点名 (与判据同处一个类, 按同一分支顺序) ——
            # 这里只拼接, 不再写一遍判据。原先那句写死了"能力没拿到", 而凭据这条
            # 新增成因会印出「缺 —」: 判定对了、说的是另一件事 (I46 那一族)。
            _refusal_causes = "; ".join(assessment.l3_pass_refusal_causes)
            msg += ("  ⛔ 本轮 L3 不记为通过: "
                    + (_refusal_causes or "「通过」没有指称对象"))
        return msg, assessment

    # 算子类型 -> bound 分类 (决定优化时关注的参考指标)
    _COMPUTE_BOUND_OPS = ("matmul", "gemm", "conv", "attention", "einsum")
    _MEMORY_BOUND_OPS = ("softmax", "layernorm", "layer_norm", "batchnorm",
                         "batch_norm", "relu", "gelu", "sigmoid", "add",
                         "dropout", "norm", "elementwise")

    def _publish_oracle_observation(self, metrics, assessment, task, *,
                                    cv_exceeded, heartbeat, heartbeat_expected,
                                    required) -> None:
        """§13.4 的 Oracle **唯一收口点** —— 每个 L3 读数都产一条 Observation。

        为什么值得独立成方法 (而不是继续留在 `_assess_measurement` 里): 这一段是
        spine-review P0-1 / P0-2 那条「唯一收口点」纪律的落点, 而它的**接口恰好就是
        那句纪律本身** —— 一切读数从 `metrics` 里取 (已经发布进产物的那一份), 不回头
        问调用方的局部量: `_assess_measurement` 是**另一个**方法, 拿不到
        `verify_l3_only` 的参数, 用局部量在这里根本看不见 (而 `observation_error`
        会把它吞成一行日志 —— 那条注释就写在下面那段实参里)。所以传进来的只有两件事:
        「这次判定」(`assessment` 与它上游的 `cv_exceeded`) 与「心跳窗」
        (`heartbeat` / `heartbeat_expected` / `required`)。

        契约: 写 `metrics["observation"]` 与 `metrics["eligibility"]`; 收口点
        **没跑成**时保留调用方刚写下的 fail-closed 全 False —— 发布的档位必须是收口点的
        答案, 不是 `assess()` 自己算的那一份 (真机 §14-8 就是两份推导并存的后果)。
        """
        try:
            from ..core.contract import freeze_from_task
            from ..decide.oracle import evaluate as oracle_evaluate
            contract = freeze_from_task(task) if task is not None else None
            if contract is not None:
                stamp = contract.hash or contract.compute_hash()
                # **记录携带**的哈希 (spine-review P0-2): 只有当载荷真带了旧记录哈希
                # 时才有得校。原来这里写的是
                #     metrics.setdefault("contract_hash", stamp)
                # 再把它当 `record_hash` 回喂 —— 两侧都由这一行自己造出来, 于是
                # `assert_hash` 恒真, 「唯一收口点」名存实亡。现在:
                #   * 带旧哈希 → 真比对 (不匹配 → oracle 拒评, 且下面记 mismatch 键);
                #   * 没带     → 明说这是**新观测** (`fresh=True`), 只盖溯源戳,
                #                observation 里记 `hash_check="fresh_unverified"`,
                #                绝不声称校验过 —— 「没得校」与「校验过」可区分。
                carried = metrics.get("contract_hash")
                metrics["contract_hash"] = stamp   # 溯源: 这条读数在哪个契约下取得
                # (载荷带的那个值会留在 observation.provenance.record_hash 里,
                #  oracle 自己记, 这里不必再开一个键。)
                obs = oracle_evaluate(
                    contract,
                    lambda: {
                        "value": metrics.get("speedup_vs_baseline"),
                        "validity": assessment.validity.value,
                        "lib_fallback": bool(metrics.get("lib_fallback")),
                        "real_device": assessment.backend_real_device,
                        "l3_pass_is_refused": assessment.l3_pass_is_refused,
                        "ruler_identity_error": metrics.get("ruler_identity_error", ""),
                        "cuda_ruler_profile": metrics.get("cuda_ruler_profile"),
                        "cv_exceeded": cv_exceeded,
                        "launch_dominance_exceeded": (
                            assessment.launch_dominance is not None
                            and assessment.launch_dominance > 0.5),
                        "partition_drift": bool(assessment.environment_drift),
                        # 静态守卫的适用性 (CHECKLOG I65): 读**已经发布进 metrics 的
                        # 那一份** —— 判定看的必须就是产物里记的 (与 arm_device_labels /
                        # arm_windows 同一个写法)。它由 `_verify_performance` 从 L1 结果
                        # 搬进来, 不是 Verifier 实例字段 (一个实例验多个候选)。
                        "static_guards": metrics.get("static_guards"),
                        # 正确性凭据 (§14-10 / §20.4): 同上 —— 读**已经发布进 metrics 的
                        # 那一份**, 不回头问调用方的局部量 (`_assess_measurement` 是另一个
                        # 方法, 拿不到 `verify_l3_only` 的参数)。缺键 = 没带凭据 = 不改判。
                        "correctness_verified": metrics.get("correctness_verified"),
                    },
                    record_hash=carried or None,
                    fresh=not carried,
                    require_real_device=Cap.REAL_DEVICE.value in required,
                    metric="speedup_vs_baseline", unit="x",
                    # 心跳: 只对**它自己量过的那扇窗**作数 (§13.5) —— Conductor 是为
                    # 本轮这条读数跑的心跳, 所以本读数记在同一个 window 里。
                    heartbeat=heartbeat,
                    heartbeat_expected=heartbeat_expected,
                    candidate_window=(heartbeat.window_id if heartbeat is not None else None),
                    # 同卡配对 (§11-2, CHECKLOG I47): 喂的是 `arm_device_labels` ——
                    # 上面刚发布进 metrics 的那一份, 不是现在重算一份。
                    candidate_device=(metrics.get("arm_device_labels") or {}).get("candidate"),
                    baseline_device=(metrics.get("arm_device_labels") or {}).get("baseline"),
                    # 窗口那一半**单列一组参数**: `candidate_window` 是心跳窗口 id, 两者
                    # 不是一回事 (把心跳窗当测量窗会让心跳那一段判成「别窗读数」)。
                    # 窗口也从**产物里记的那一份**读 (`_assess_measurement` 是另一个
                    # 方法, 拿不到 `_verify_performance` 的局部量 —— 用局部量在这里
                    # 根本看不见, 而 `observation_error` 会把它吞成一行日志)。
                    pair_candidate_window=(metrics.get("arm_windows") or {}).get("candidate"),
                    pair_baseline_window=(metrics.get("arm_windows") or {}).get("baseline"),
                )
                metrics["observation"] = obs.as_dict()
                # 收口点的答案就是**发布值**: 它读的是同一组输入加心跳窗口, 是严格更
                # 知情的那一份。发布别的推导 = 让「记录里说 A、判定按 B」这种分叉在
                # 生产上长期存在 (真机 §14-8 就是它)。
                metrics["eligibility"] = obs.eligibility.as_dict()
                if obs.refused:
                    # 拒评 = 本次**没有**有效观测 ⇒ 发布的档位必须是收口点的答案,
                    # 而不是上面 `assess()` 算的那一份。否则又回到 I46 的形状:
                    # `measurement_validity=valid` 与 `eligibility` 全 False 并排 ——
                    # 「记录里说 A、判定按 B」正是那次真机抓到的病。
                    # 三个用途位用 `replace` 重新摊平而不是手写: 档位→用途的映射只有
                    # 一处 (`MeasurementAssessment` 的三个 property), 这里不许再写第二遍。
                    metrics.update(dataclasses.replace(
                        assessment,
                        validity=measurement.Validity(obs.validity)).as_metrics())
                # 让「契约被改写」这条结论真的有人读: `confirm.py` 的 UnableReport
                # 一直在读 `contract_hash_mismatch`, 而此前**没有任何生产写入者**
                # —— 那条「本次评测作废」的条目永远不会出现在报告里 (死读键)。
                if obs.hash_check == "mismatch":
                    metrics["contract_hash_mismatch"] = True
                # 同一族的另一个死读键, 这份是尺子: `confirm.py` 的「该窗口内所有数字
                # 不可用」与 `harness_pass_rate` 都在读 `ruler_broken`/`ruler_reason`,
                # 而此前同样没有生产写入者。只在**真有心跳可谈**时写:
                #   有心跳且通过 → False  (「检查过, 是好的」)
                #   失败 / 到期没跑 → True + 理由
                #   本次与心跳无关 → 不写 (「没检查」不许伪装成「检查过且好」)
                if obs.provenance.get("heartbeat_checked") or \
                        obs.provenance.get("heartbeat_expected"):
                    metrics["heartbeat"] = obs.provenance
                    metrics["ruler_broken"] = bool(obs.ruler_broken)
                    if obs.ruler_broken:
                        metrics["ruler_reason"] = "; ".join(
                            obs.provenance.get("heartbeat_reasons") or [])
        except Exception as exc:  # noqa: BLE001 — 观测失败不挡主流程, 但要留痕
            metrics["observation_error"] = f"{type(exc).__name__}: {exc}"

    @classmethod
    def _bound_type(cls, kernel_type: str) -> str:
        """算子 bound 分类: compute (关注 GFLOPS/Tensor Core) / memory (关注有效带宽)"""
        kt = (kernel_type or "").lower()
        if any(op in kt for op in cls._COMPUTE_BOUND_OPS):
            return "compute"
        if any(op in kt for op in cls._MEMORY_BOUND_OPS):
            return "memory"
        return "unknown"

    # ---- Roofline 分析 ----

    def _refresh_measured_peak(self) -> None:
        """把存放处里的参考下界重新合并进已缓存的 peak 字典 (零成本, 不探硬件)

        为什么需要: `_detect_theoretical_peak` 是缓存的, 而 `probe_peak` 的发布发生在
        同一次运行的**第 1 轮** L3 之后 —— 不重读的话第 2 轮读到的还是「没实测」, 于是
        标定永远只在下一个进程里生效, 而存放处明确不跨进程 (见 `hardware_caps` 那段)。
        实测抓到过: 用例里第一轮 publish 成功, 第二轮的出口仍然关着。
        """
        peak = self._theoretical_peak or {}
        key = device_key_of({"name": peak.get("name", ""),
                             "compute_capability": peak.get("compute_capability", "")})
        measured = measured_peaks_for(key)
        peak["reference_lower_bound_tflops"] = dict(measured)
        peak["reference_lower_bound"] = any(
            isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0
            for x in measured.values())

    def _detect_theoretical_peak(self) -> dict:
        """探测硬件理论峰值 (GFLOPS, 带宽)，优先按设备名匹配，回退 compute_capability

        ⚠️ 缓存的是**表那一半**; 参考下界那一半会变 (§13.5: 同一次运行里第 1 轮的 L3 标定
        完, 第 2 轮就该读到), 所以每次调用都从存放处重读。这里不重新探测硬件 ——
        `get_hardware_info()` 在 SSH 路径上是一次真实往返, 而设备身份没变。
        """
        if self._theoretical_peak is not None:
            self._refresh_measured_peak()
            return self._theoretical_peak

        hw = {}
        try:
            hw = self.backend.get_hardware_info()
        except Exception:
            pass

        cc = str(hw.get("compute_capability", "")).strip()
        device = str(hw.get("device", "")).lower()
        name = str(hw.get("name", "")).lower()

        # 1. 设备名优先: cc 9.0 同时是 H100/H200/H20, 只有设备名能区分
        #    键名长者先匹配, 避免 "h20" 抢在 "h200" 之前命中
        vals = None
        matched_key = None
        matched_from = ""
        for key in sorted(_PEAK_BY_NAME, key=len, reverse=True):
            if key in name or key in device:
                vals = _PEAK_BY_NAME[key]
                matched_key, matched_from = key, "name"
                break

        # 2. 设备名未命中时回退 compute_capability (精确 → 前缀)
        if vals is None and cc:
            if cc in _PEAK_BY_CC:
                vals = _PEAK_BY_CC[cc]
                matched_key, matched_from = cc, "cc"
            else:
                for key in sorted(_PEAK_BY_CC, key=len, reverse=True):
                    if cc.startswith(key):
                        vals = _PEAK_BY_CC[key]
                        matched_key, matched_from = key, "cc-prefix"
                        break

        fp32, tf32, fp16, bf16, bw = vals if vals else (0, 0, 0, 0, 0)
        # 出处跟着数字走: 日志里每一行 peak 都要能回答「这个分母凭什么」。
        # 未核口径的行会带着 ⚠️ 出现在每次验证输出里 —— 缺口 3 的病根不是数值错,
        # 而是数值在使用点看不出它有多可信。
        if matched_key is None:
            peak_source = "未命中任何表 (peak=0, 达成率不可用)"
        elif matched_from == "name":
            peak_source = _PEAK_SOURCE.get(matched_key, _PEAK_SOURCE_DEFAULT)
        else:
            peak_source = ("回退按 cc 匹配: "
                           + _PEAK_CC_SOURCE.get(matched_key, _PEAK_SOURCE_DEFAULT))
        # 出处字符串本身就是可信度的载体 (见 _PEAK_SOURCE 上方的说明): 以 ⚠️ 开头、
        # 或者根本没命中表的行, 都**不配**用来开「已到 roofline」这个判停出口。
        peak_trusted = matched_key is not None and not peak_source.startswith("⚠️")

        # launch 地板与**参考下界**都走 hardware_caps 而不是上面这两张表: 它们是实测值
        # 而非 datasheet 推算值, 且按缺口 3 的方向, 硬件知识应该收敛到 HardwareCaps 一处。
        # 取不到 (未知设备 / get_hardware_info 抛错) 时为 None, 下游会显式记录
        # 「未做地板支配检查」而不是默认放行。
        caps = None
        try:
            caps = detect_hardware_caps(hw)
        except Exception:
            caps = None
        launch_floor_us = getattr(caps, "launch_floor_us", None)
        # 分母**实测过**没有 (设计 §13.5)。与 `peak_trusted` 刻意分开: 后者问的是「表行
        # 口径核过没有」(dense/sparse 没混、算术自洽), 前者问的是「有没有一次真探测」。
        # H20 今天两问的答案是「核过」+「从没探测」, 而天花板出口只认后者。
        # §13.5 起 `reference_lower_bound_tflops` 是**按精度**的字典 (`{"tf32": 68.1}`), 由
        # `peak_probe` 判过、`hardware_caps` 存放处发布。这里的标量 `reference_lower_bound`
        # 只回答「这张卡**任何一个**精度有没有实测过」(透明用); 让天花板出口开的那一位是
        # `_roofline_analysis` 里**按这一轮实际分母的精度键**算出来的那个 —— 两者混用
        # 会拿 tf32 的实测值去开 fp32 那一轮的出口, 数值差近一倍而日志里长得一样。
        _mp = getattr(caps, "reference_lower_bound_tflops", None)
        reference_lower_bound = bool(_mp) and any(
            isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0
            for x in _mp.values()) if isinstance(_mp, dict) else False

        self._theoretical_peak = {
            "fp32_tflops": fp32, "tf32_tflops": tf32,
            "fp16_tflops": fp16, "bf16_tflops": bf16,
            "memory_bandwidth_gbs": bw,
            "peak_source": peak_source,
            "peak_trusted": peak_trusted,
            # 天花板出口判停用的**唯一**那个位 (measurement.reference_ceiling_reached)。
            # `probe_peak` 落地时: 按精度把实测值写进 HardwareCaps.reference_lower_bound_tflops,
            # 这个位才会翻真 —— 在那之前出口是关的, 这是设计要的行为, 不是缺陷。
            "reference_lower_bound": reference_lower_bound,
            "reference_lower_bound_tflops": dict(_mp) if isinstance(_mp, dict) else {},
            "peak_table_key": matched_key or "",
            "launch_floor_us": launch_floor_us,
            # 每 SM 的 smem 上界: §12 的**便宜代理剪枝**要它 (不花 GPU 秒就能摘掉
            # 「这套 tile 根本放不下」的点)。走 caps 这一处 —— 与 L1 资源预检同一个
            # 真源, 不在这里另抄一个数字; 取不到时 0 (= 不知道), 下游必须显式处理。
            "shared_mem_per_sm_kb": int(getattr(caps, "shared_mem_per_sm_kb", 0) or 0),
            # 每 SM 的线程上界: §10.3 条件 2 的第二把量具 (占用率) 要它。与 smem 同一个
            # 真源、同一条纪律: 取不到时 0 (= 不知道), 由判据显式处理 —— 缺键既不等于
            # 「没有上限」, 也不等于「有硬限制」。
            "max_threads_per_sm": int(getattr(caps, "max_threads_per_sm", 0) or 0),
            "compute_capability": cc, "device": device, "name": name,
        }
        return self._theoretical_peak

    @staticmethod
    def _compute_arithmetic_intensity(task: TaskSchema, inputs: dict) -> float:
        """算术强度 = `_ai_numer_denom` 的比值。**唯一**实现仍在上面那一处分支逻辑里。

        为什么不把比值也算在那个函数里返回: 探针只要分子/分母, 而比值有三个调用者
        (roofline/血缘/用例)。两边各取所需, 分支逻辑只有一份 —— 比「返回一个三元组、
        每个调用者自己拆」少一层出错的机会。
        """
        flops, bytes_transferred = Verifier._ai_numer_denom(task, inputs)
        if flops is None or not bytes_transferred or bytes_transferred <= 0:
            return 0.0
        return flops / bytes_transferred

    @staticmethod
    def _ai_numer_denom(task: TaskSchema, inputs: dict):
        """算术强度的**分子与分母**(flops, bytes); 认不出的算子返回 `(None, None)`。

        §13.5 的 `probe_peak` 要拿「基线臂的实测可达值 = flops / baseline_ms」当分母,
        而 flops 只在这里算过。于是把**这一处**的输入露出来给探针用 —— 另写一个
        「算 flops」的函数就是第二个真源, 而漂移的症状恰好是「探针的分母与 roofline 的
        分子不是同一个量」, 那种错在日志里看不出来 (两个数都长得像 FLOPS)。
        """
        """计算算子的算数强度 (FLOPs / Bytes transferred)
        返回 FLOP/Byte 或 0 (无法计算时)

        ⚠️ Byte 轴用的是**任务声明精度**的字节数, 不是 `inputs` 里数组自己的 dtype。
        runner 会按声明精度转换输入 (`remote_scripts/run_kernel.py`: precision == "fp16"
        时 `.to(torch.float16)`), 所以控制端手里那份 float32 数组是核**根本见不到**的形态。
        第一版就是这么写的: matmul 4096³ 声明 FP16 而输入是 float32, AI 算成 682.67
        而不是 1365.33 —— 正好差 2 倍 (doc 48 §6.5)。达成率与计时对比都不受影响,
        但 AI 决定脊点两侧的分类, 而分类决定天花板出口与 SearchAgent 走哪条建议。
        """
        kt = (task.source.kernel_type or "").lower()
        nbytes = task.constraints.precision.itemsize

        if "matmul" in kt or "gemm" in kt:
            # GEMM: 2*M*N*K FLOPs, 数据量 = (M*K + K*N + M*N) * dtype_bytes
            if "A" in inputs and "B" in inputs:
                a = np.asarray(inputs["A"])
                b = np.asarray(inputs["B"])
                M, K = a.shape[-2], a.shape[-1]
                K2, N = b.shape[-2], b.shape[-1]
                if K != K2:
                    return None, None
                flops = 2.0 * M * N * K
                bytes_transferred = (M * K + K * N + M * N) * nbytes
                return flops, bytes_transferred

        if "softmax" in kt:
            # Softmax: ~5*N FLOPs per element (exp, sum, div), memory: N * dtype
            if "X" in inputs:
                x = np.asarray(inputs["X"])
                n = int(np.prod(x.shape))
                flops = 5.0 * n
                bytes_transferred = 2.0 * n * nbytes  # read + write
                return flops, bytes_transferred

        if any(op in kt for op in ("relu", "gelu", "sigmoid", "add", "elementwise")):
            # Element-wise: ~1 FLOP per element, memory: 2-3 * N * dtype
            for key in ("X", "input", "A"):
                if key in inputs:
                    x = np.asarray(inputs[key])
                    n = int(np.prod(x.shape))
                    flops = 1.0 * n
                    bytes_transferred = 2.0 * n * nbytes
                    return flops, bytes_transferred

        if "layernorm" in kt or "layer_norm" in kt:
            # LayerNorm: ~7*N FLOPs (mean, var, norm, scale, shift), memory: ~2*N
            # (只算 X 的读写, 不含 weight/bias 那 O(N) 项; 远端 runner 的公式含它们,
            #  两边在 M≫1 时差 <0.1%, 但确实不是同一个公式)
            for key in ("X", "input"):
                if key in inputs:
                    x = np.asarray(inputs[key])
                    n = int(np.prod(x.shape))
                    flops = 7.0 * n
                    bytes_transferred = 2.0 * n * nbytes
                    return flops, bytes_transferred

        return None, None

    def _roofline_analysis(self, task: TaskSchema, bench_inputs: dict,
                           actual_gflops: float, actual_bandwidth_gbs: float) -> dict:
        """
        Roofline 模型分析:
        - 计算算数强度 (AI = FLOPs / Bytes)
        - 获取硬件理论峰值
        - 判断真实瓶颈 (compute-bound vs memory-bound)
        - 计算利用率
        """
        peak = self._detect_theoretical_peak()
        hw_peak_gflops = peak.get("fp32_tflops", 0) * 1000  # TFLOPS -> GFLOPS
        hw_peak_bandwidth = peak.get("memory_bandwidth_gbs", 0)

        # 根据精度 + 硬件选择计算峰值:
        # - FP16/BF16: Tensor Core 峰值
        # - FP32 + CUDA matmul + Ampere+: Triton tl.dot 默认用 TF32 Tensor Core
        # - 其他: FP32 CUDA Core 峰值
        precision = task.constraints.precision
        compute_used = "fp32_cuda_core"
        if precision in (Precision.FP16,):
            hw_peak_gflops = peak.get("fp16_tflops", hw_peak_gflops / 1000) * 1000
            compute_used = "fp16_tensor_core"
        elif precision in (Precision.BF16,):
            hw_peak_gflops = peak.get("bf16_tflops", hw_peak_gflops / 1000) * 1000
            compute_used = "bf16_tensor_core"
        else:
            # FP32: 检查是否为 CUDA matmul 且硬件支持 TF32 (Ampere+)
            kt = (task.source.kernel_type or "").lower()
            is_matmul = "matmul" in kt or "gemm" in kt
            device = str(peak.get("device", "")).lower()
            cc = str(peak.get("compute_capability", ""))
            is_cuda = "cuda" in device
            has_tf32 = (is_cuda and cc and cc.startswith(("8.", "9.")) and
                        peak.get("tf32_tflops", 0) > 0)
            if is_matmul and is_cuda and has_tf32:
                hw_peak_gflops = peak.get("tf32_tflops", hw_peak_gflops / 1000) * 1000
                compute_used = "tf32_tensor_core"

        # §13.5: 这个精度的分母**实测过**没有 —— 只有这一位能让天花板出口开
        # (`measurement.reference_ceiling_reached` 只认 `reference_lower_bound`)。键必须与上面选分母
        # 用的那个精度**同一个** (`compute_used`), 取不到就当作未实测: 不许借用别的
        # 精度 (tf32 vs fp32 差近一倍, 借用会把还能优化的核提前判成到顶)。
        peak_key = precision_key_of(compute_used)
        _measured = peak.get("reference_lower_bound_tflops") or {}
        _measured_tflops = _measured.get(peak_key) if peak_key else None
        reference_lower_bound = bool(isinstance(_measured_tflops, (int, float))
                             and not isinstance(_measured_tflops, bool)
                             and _measured_tflops > 0)
        if reference_lower_bound:
            hw_peak_gflops = float(_measured_tflops) * 1000.0
        # 设备身份取 `_detect_theoretical_peak` 发布的那两个字段 (name/cc) ——
        # 它是 `detect_hardware_caps` 认卡用的同一批字段, 也是存放处的分组键。
        _dev_key = device_key_of({"name": peak.get("name", ""),
                                  "compute_capability": peak.get("compute_capability", "")})
        _measured_rec = (measured_peak_record(_dev_key, peak_key) if reference_lower_bound
                         else {"method": "", "evidence": {}})

        ai = self._compute_arithmetic_intensity(task, bench_inputs)
        # 分子/分母走**同一个**静态方法 (纯函数, 再算一遍不会有第二种答案) —— 探针要的
        # 那一份分子就是 roofline 用的那一份, 而不是另算一份。
        _flops, _bytes = self._ai_numer_denom(task, bench_inputs)
        _ai_inputs = {"flops": _flops, "bytes": _bytes}

        # Roofline 判定: AI * peak_bandwidth >= peak_gflops -> compute-bound
        # 即当 AI > peak_gflops / peak_bandwidth 时，性能受限于计算而非带宽
        ridge_point = (hw_peak_gflops / hw_peak_bandwidth
                       if hw_peak_gflops > 0 and hw_peak_bandwidth > 0 else 0.0)
        # 没有可信硬件峰值（如 MockGPU）时不能假装得到 roofline 结论；
        # 回退到算子类别，避免 ``0 / 0 -> memory`` 的偶然误判。
        roofline_bound = ("compute" if ai >= ridge_point else "memory") if ridge_point > 0 else "unknown"

        gflops_util = (actual_gflops / hw_peak_gflops * 100) if hw_peak_gflops > 0 else 0.0
        bw_util = (actual_bandwidth_gbs / hw_peak_bandwidth * 100) if hw_peak_bandwidth > 0 else 0.0

        # 综合 bound 类型: 结合算子名启发式 + Roofline 计算结果
        name_bound = self._bound_type(task.source.kernel_type)
        if roofline_bound == "unknown":
            bound_type = name_bound
        elif name_bound == "compute" and roofline_bound == "memory":
            # 名字说是 compute-bound 但 Roofline 显示 memory-bound: 小矩阵场景
            bound_type = "memory"  # 信任 Roofline
        elif name_bound == "memory" and roofline_bound == "compute":
            bound_type = "compute"  # 信任 Roofline
        elif name_bound == "unknown":
            bound_type = roofline_bound if ai > 0 else "unknown"
        else:
            bound_type = name_bound  # 一致时沿用

        return {
            "hw_peak_gflops": round(hw_peak_gflops, 1),
            # 分子也要发布: 「基线臂的实测可达值」是拿 `flops / baseline_ms` 算的
            # (peak_probe 的装配点), 而 flops 只在 `_compute_arithmetic_intensity` 里
            # 算过。这里记的是**它用过的那一份**, 不是重算的 (重算=第二个真源)。
            "flops": (round(_ai_inputs["flops"], 1) if _ai_inputs.get("flops") else None),
            "hw_peak_bandwidth_gbs": round(hw_peak_bandwidth, 1),
            "compute_used": compute_used,
            "arithmetic_intensity": round(ai, 4),
            "ridge_point": round(ridge_point, 2),
            "roofline_bound": roofline_bound,
            "name_bound": name_bound,  # 算子名启发式分类 (对比用)
            "bound_type": bound_type,  # 综合判定 (roofline + 启发式)
            "gflops_utilization_pct": round(gflops_util, 1),
            "bandwidth_utilization_pct": round(bw_util, 1),
            "hw_name": peak.get("name", "unknown"),
            "hw_compute_capability": peak.get("compute_capability", "unknown"),
            "shared_mem_per_sm_kb": peak.get("shared_mem_per_sm_kb", 0),
            # 分母的出处跟着分母走。缺口 3 的病根不是某个数字错, 而是**用这个数字的人
            # 看不出它有多可信**: 138.2 和 1979 在日志里长得一模一样。
            "peak_source": peak.get("peak_source", ""),
            "peak_trusted": peak.get("peak_trusted", False),
            # 天花板出口判停**只认这一个位** (设计 §13.5)。发布出去是必要的: 它不在这里,
            # 下游拿到的是一份「核过口径」的记录, 而「实测过没有」这件事实就丢了 ——
            # 与 I45/I46/I48 同族的「算了一处、判在另一处」。
            # **按精度**的那一位: 天花板出口读的就是它 (不是上面那个「任何精度都算」的标量)。
            "reference_lower_bound": reference_lower_bound,
            "peak_key": peak_key or "",
            "reference_lower_bound_source": _measured_rec.get("method", ""),
            "reference_lower_bound_evidence": _measured_rec.get("evidence", {}),
            "peak_table_key": peak.get("peak_table_key", ""),
        }

    def _publish_peak_probe(self, metrics: dict) -> None:
        """§13.5: 参考下界标定 —— 判据在 `peak_probe.probe_peak`, 这里只装配与发布

        样本来自同一次 L3 调用里的**基线臂**(同机、同形状、同精度的已知好参考核),
        即「这台机器上这个精度**至少**能跑这么快」。它是下界 ⇒ 拿它当分母, 利用率偏大、
        天花板出口偏松; 这条残余风险明写在 `peak_probe` 的模块 docstring 里(约束是
        「样本自己必须 compute-bound 且 valid」两条), 这里不再复述一遍。

        发布进 `hardware_caps` 的**进程级**存放处 ⇒ 下一次 `detect_hardware_caps` 才读得到,
        于是天花板出口从下一轮开始有分母可用。**不现场改这一轮的分母**: 这一轮的 roofline
        已经用表算了, 事后替换会让同一份产物里的分子分母来自两个时刻。
        """
        roofline = metrics.get("roofline_analysis") or {}
        peak_key = precision_key_of(roofline.get("compute_used"))
        table = self._theoretical_peak or {}
        spec = table.get(f"{peak_key}_tflops") if peak_key else None
        sample = sample_from_l3_metrics(
            metrics, real_device=(Cap.REAL_DEVICE in self._caps),
            spec_tflops=spec,
            method=("L3 基线臂 (同机/同形状/同精度的已知好参考核): flops / baseline_ms"),
            flops=roofline.get("flops"),
            baseline_ms=metrics.get("baseline_time_ms"),
        )
        decision = probe_peak(sample)
        metrics["peak_probe"] = decision.as_dict()
        if not decision.measured:
            print(f"  [Verifier] 参考下界标定未成立 ⇒ 天花板出口继续关闭: "
                  f"{decision.refusal_note()}")
            return
        profile = decision.profile
        key = device_key_of({"name": table.get("name", ""),
                             "compute_capability": table.get("compute_capability", "")})
        remember_reference_lower_bound(key, profile.precision, profile.tflops,
                               profile.method, profile.evidence)
        print(f"  [Verifier] 参考下界标定: {profile.precision} = "
              f"{profile.tflops:.2f} TFLOPS ({profile.method}) ⇒ 下一轮的天花板出口有分母了")

    def _build_reference_fn(self, task: TaskSchema, inputs: dict,
                            high_precision: bool = False):
        """
        构建 PyTorch 参考实现 (可调用); torch 不可用时回退 NumPy
        返回 (callable, baseline_type) 或 (None, None)

        **外挂参考优先**: `source.reference_source` 非空时不再查内建表。
        新架构算子没有内建参考, 没有这一层时 L3 会「无可用基线」、L2 会
        「无参考仅验证可执行性」—— 整条闭环不存在 (见 reference.py)。

        `high_precision=True` 要的是**同一份参考、换 float64 再算一遍** (高精度真值 oracle,
        给 `tolerance.error_ratio` 用), 不是另一套规则 —— 所以这里只换 dtype 取值那两行,
        下面的整条 if-链原样复用 (另写一份 fp64 版参考实现就是第二个真源, 会漂)。
        ⚠️ 两种情况下如实返回 None 而不是降级凑一个数:
        · 外挂 `reference_source` —— 不保证它对 float64 输入是多态的, 喂进去可能悄悄换个算法;
        · 整数任务 —— 没有「更高精度的整数」可算。
        """
        if task.constraints.reference_contract is not None:
            raise ValueError("v2 native CUDA reference cannot be replaced by a local/CPU oracle")
        from operator_agent.decide.reference import ReferenceSourceError, resolve_reference
        try:
            ext = resolve_reference(task)
        except ReferenceSourceError as exc:
            # 加载失败是任务定义错误, 不是「没有参考」—— 后者会静默放行 L2
            raise ValueError(f"外挂 reference_source 不可用: {exc}") from exc
        if ext is not None:
            if high_precision:
                return None, None
            # 与内建参考同一约定: 闭包捕获当前 inputs, 零参可调
            return (lambda fn=ext, xs=inputs: fn(**xs)), "reference_source"

        kernel_type = task.source.kernel_type.lower()
        precision = task.constraints.precision
        if high_precision and precision is Precision.INT8:
            return None, None
        dtype_np = {Precision.FP32: np.float32, Precision.FP16: np.float16,
                    Precision.BF16: np.float32, Precision.INT8: np.int8}.get(precision, np.float32)

        # 尝试 PyTorch (优先, 与真实 eager baseline 对齐)
        try:
            import torch
            torch_available = True
            dtype_torch = {Precision.FP32: torch.float32, Precision.FP16: torch.float16,
                           Precision.BF16: torch.bfloat16, Precision.INT8: torch.int8}.get(precision, torch.float32)
        except ImportError:
            torch_available = False

        if high_precision:
            dtype_np = np.float64
            if torch_available:
                dtype_torch = torch.float64

        if ("matmul" in kernel_type or "gemm" in kernel_type) and "A" in inputs and "B" in inputs:
            if torch_available:
                a = torch.from_numpy(_reference_array(inputs["A"], dtype_np)).to(dtype_torch)
                b = torch.from_numpy(_reference_array(inputs["B"], dtype_np)).to(dtype_torch)
                return lambda: (a @ b), "pytorch_eager"
            a = np.asarray(inputs["A"], dtype=dtype_np)
            b = np.asarray(inputs["B"], dtype=dtype_np)
            return lambda: (a @ b), "numpy_reference"

        if "softmax" in kernel_type and "X" in inputs:
            if torch_available:
                import torch.nn.functional as F
                x = torch.from_numpy(_reference_array(inputs["X"], dtype_np)).to(dtype_torch)
                return lambda: F.softmax(x, dim=-1), "pytorch_eager"
            x = np.asarray(inputs["X"], dtype=dtype_np)

            def _softmax():
                mx = np.max(x, axis=-1, keepdims=True)
                ex = np.exp(x - mx)
                return ex / np.sum(ex, axis=-1, keepdims=True)
            return _softmax, "numpy_reference"

        # 不对 Tensor/ndarray 求真值；缺输入先判 None，不能把 np.asarray(None)
        # 得到的标量 NaN 当作有效输入。X 显式缺失时才回退 input。
        x_value = inputs.get("X")
        if x_value is None:
            x_value = inputs.get("input")
        if x_value is None:
            return None, None
        x_np = _reference_array(x_value, dtype_np)

        # 更多算子: 优先 torch 原生 (与 eager 对齐)
        if torch_available:
            x = torch.from_numpy(x_np).to(dtype_torch)
            if "relu" in kernel_type:
                return lambda: torch.relu(x), "pytorch_eager"
            if "sigmoid" in kernel_type:
                return lambda: torch.sigmoid(x), "pytorch_eager"
            if "gelu" in kernel_type:
                return lambda: torch.nn.functional.gelu(x), "pytorch_eager"
            if "layernorm" in kernel_type or "layer_norm" in kernel_type:
                w_np = _reference_array(inputs.get("weight"), dtype_np) if inputs.get("weight") is not None else None
                b_np = _reference_array(inputs.get("bias"), dtype_np) if inputs.get("bias") is not None else None
                if w_np is not None and b_np is not None:
                    w = torch.from_numpy(w_np).to(dtype_torch)
                    b = torch.from_numpy(b_np).to(dtype_torch)
                    return lambda: torch.nn.functional.layer_norm(x, x.shape[-1:], w, b), "pytorch_eager"
                return lambda: torch.nn.functional.layer_norm(x, x.shape[-1:]), "pytorch_eager"
            if "batchnorm" in kernel_type or "batch_norm" in kernel_type:
                return lambda: torch.nn.functional.batch_norm(
                    x, torch.ones(x.shape[-1]), torch.zeros(x.shape[-1]),
                    training=True), "pytorch_eager"
        # 离线 Golden 只要求 NumPy；torch 未安装时也必须真正校验这两种
        # 算子的结果，而不是以「无参考，仅可执行」代替正确性。
        if not torch_available:
            if "relu" in kernel_type:
                return lambda: np.maximum(x_np, 0), "numpy_reference"
            if "layernorm" in kernel_type or "layer_norm" in kernel_type:
                def _layernorm():
                    mu = np.mean(x_np, axis=-1, keepdims=True)
                    var = np.var(x_np, axis=-1, keepdims=True)
                    result = (x_np - mu) / np.sqrt(var + 1e-5)
                    if inputs.get("weight") is not None:
                        result = result * np.asarray(inputs["weight"], dtype=dtype_np)
                    if inputs.get("bias") is not None:
                        result = result + np.asarray(inputs["bias"], dtype=dtype_np)
                    return result
                return _layernorm, "numpy_reference"
        return None, None

    def _generate_reference(self, task: TaskSchema, inputs: dict,
                            high_precision: bool = False):
        """生成参考输出 (numpy 数组); torch 张量自动转 numpy

        `high_precision=True` 时返回 float64 的那一份 (高精度真值 oracle) —— 这时**不许**再走
        `_to_numpy` 的默认 float32 转换, 否则 oracle 自己就被压成候选同一个精度, ρ 的分母归零。
        """
        fn, _ = self._build_reference_fn(task, inputs, high_precision=high_precision)
        if fn is None:
            return None
        out = fn()
        return self._to_numpy(out, dtype=np.float64 if high_precision else np.float32)

    @staticmethod
    def _to_numpy(out, dtype=np.float32):
        """torch tensor / numpy array 统一转 numpy (默认 float32, 传 dtype 可要 float64)"""
        try:
            import torch
            if isinstance(out, torch.Tensor):
                return out.detach().cpu().numpy().astype(dtype)
        except ImportError:
            pass
        return np.asarray(out, dtype=dtype)

    def _measure_baseline(self, task: TaskSchema, inputs: dict,
                          warmup: int = 3, repeat: int = 10):
        """
        测量 PyTorch eager baseline 耗时 (min-of-repeats，抗调度噪声)
        torch 不可用时回退 NumPy。按 (算子, 输入形状/dtype 签名) 缓存。
        无参考实现时返回 None
        """
        fn, baseline_type = self._build_reference_fn(task, inputs)
        if fn is None:
            return None
        cache_key = self._baseline_cache_key(task, inputs)
        if cache_key in self._baseline_cache:
            # ⚠️ 这是**跨窗复用**: 命中缓存 = 这条基线是**本轮之前**量的 (§11-2 的
            # 「同一时间窗」在这个控制端参照臂上并不成立)。今天不作判决 —— 这条臂不是
            # 设备租约臂, 复用是**声明的**优化 (省一次 torch eager 的墙钟); 但它从此
            # **看得见**: 产物里 `baseline_reused=True`, 消息里也会写出来。
            return dict(self._baseline_cache[cache_key], baseline_reused=True)

        # torch 可用时做同步 (CUDA stream 等待), 确保计时准确
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False

        def _sync():
            if has_torch and torch.cuda.is_available():
                torch.cuda.synchronize()

        for _ in range(warmup):
            fn()
            _sync()
        times = []
        for _ in range(repeat):
            _sync()
            start = time.perf_counter()
            fn()
            _sync()
            times.append(time.perf_counter() - start)
        result = {
            "baseline_time_ms": min(times) * 1000,
            "baseline_avg_ms": sum(times) / len(times) * 1000,
            "baseline_type": baseline_type,
        }
        result["baseline_reused"] = False       # 同一次调用里刚量的 ⇒ 不是复用
        self._baseline_cache[cache_key] = result
        return result

    @staticmethod
    def _baseline_cache_key(task: TaskSchema, inputs: dict) -> tuple:
        sig = tuple(sorted(
            (k, tuple(np.asarray(v).shape), str(np.asarray(v).dtype))
            for k, v in inputs.items()
        ))
        return (task.source.kernel_type.lower(), sig)
