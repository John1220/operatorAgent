"""
验证层 - 三级验证体系
L1: 编译验证 (代码可编译、语法正确)
L2: 功能验证 (输出正确，误差在容许范围内)
L3: 性能验证 (性能达到预期加速比) + Roofline 分析

本模块是**消费方**: 只认 `backend_caps.Backend` 与它声明的 `Cap`, 不 import 任何
具体提供方、也不做 `isinstance` 分派 (doc 49 §4.1 第 3 条)。
"""

import time
import numpy as np
from typing import Optional
from .schemas import (
    VerificationResult, VerificationLevel, TaskSchema, Precision,
)
from .anticheat import static_audit
from .interface_contract import check_kernel_interface, input_key_sets
from . import tolerance
from . import measurement
from .measurement import GATE_PARAM_KEYS, single_ruler
from .hardware_caps import detect_hardware_caps
from .backend_caps import Backend, Cap, CapabilityMissing


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
)

# 常见 GPU 硬件 **dense 规格峰值** (FP32/TF32/FP16/BF16 Tensor Core + 显存带宽)
# 格式: (FP32_TFLOPS, TF32_TFLOPS, FP16_TFLOPS, BF16_TFLOPS, memory_bandwidth_GBs)
# 匹配顺序: 先设备名 (更具体), 未命中再回退 compute_capability。
# 不可反过来 —— cc 9.0 同时对应 H100/H200/H20, 三者算力带宽差异极大。
#
# ⚠️ **这一张表回答的是「离理论上限还有多远」, 所以只能放 dense 规格, 不能放实测值。**
# 原先 h20 那行写的是 138.2/138.3/71.6/31.6/3680, 注释老实标着「实测 (cuBLAS 8192³)」——
# 出处没错, **用途错了**: 一次库实现的读数是「已达成的成绩」, 拿它当上界就会出现
# 「达成率 101.5%」(真机 4096³ 跑到 140.0 TFLOPS), 而 D4 的收敛出口正是拿达成率判停的。
# 实测值仍然有用, 但它不进这张表、也不进 roofline 分母 —— 理由见 _PEAK_BY_CC 之前那段。
# 每一行的口径记在 _PEAK_SOURCE 里, 并随 Roofline 日志一起打出来 —— 未经核实的行
# 会带着「口径不明」出现在每一轮验证的输出里, 而不是安静地假装是权威。
_PEAK_BY_NAME: dict[str, tuple[float, float, float, float, float]] = {
    # NVIDIA 官方规格页 (dense; 大字号那几个是 2:4 sparse, 差整 2 倍) + 与
    # flash-attention benchmarks/benchmark_attn.py 的 BF16 dense 表交叉核对
    "a100": (19.5,  156.0, 312.0, 312.0, 2039.0),
    "h100": (67.0,  494.7, 989.4, 989.4, 3350.0),
    "h200": (67.0,  494.7, 989.4, 989.4, 4800.0),
    # H20 没有公开的 NVIDIA 规格页 (两个官方 URL 均 404)。fp16/bf16/带宽取
    # flash-attention 硬编码的 dense 表 (148e12 / 4.0e12), 它自称来自官方规格并用
    # SM×tensor_core_clock×ops_per_cycle 交叉核对过; tf32 = bf16/2 是 Hopper 惯例
    # (H100 989.4 vs 494.7 就是这个比例); fp32 由 78 SM × 128 FMA × 2 × 1.98 GHz 推得。
    # 第三方规格页给 fp32 写 44 TFLOPS, 与上面的算术不自洽 (要 2.2 GHz), 故不用。
    "h20":  (39.5,  74.0,  148.0, 148.0, 4000.0),
    "v100": (15.7,  0.0,   125.0, 125.0, 900.0),
    # ⚠️ 以下四行**未核实口径** (很可能混了 2:4 sparse / FP16-with-FP32-accumulate)。
    # 保留数值不臆改, 但 _PEAK_SOURCE 会标「未核」并打进日志, 别拿它们的达成率做判停决定。
    "3090": (16.3,  130.6, 326.0, 326.0, 936.0),
    "4090": (82.6,  82.6,  330.0, 330.0, 1008.0),  # 4090 FP32=82.6 (no TF32 boost for Ada? actually Ada has TF32)
    "a6000":(38.7,  310.0, 77.4,  77.4,  768.0),
    "a5000":(27.8,  222.0, 55.5,  55.5,  768.0),
    "t4":   (8.1,   65.0,  130.0, 130.0, 320.0),
}

# 逐行可信度**不另立一张表**, 而是由下面的出处字符串自己承担: 以 ⚠️ 开头的行视为
# 「口径未核」, `measurement.roofline_converged` 据此拒绝开「已到 roofline」这个出口。
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
# 因为它不是一个标量: 2026-09-26 用 benchmarks/calibrate_achievable_peak.py 在 H20 GPU2
# 上量到的阶梯是 fp16 从 2048² 的 116.3 TFLOPS 爬到 6144² 的 140.3 (跨 20.7%), D2D 带宽
# 从 8 MB 的 1196 GB/s 爬到 4 GB 的 3357 GB/s (跨 2.8 倍)。按 shape 桶查它需要知道当前
# 张量落在哪个桶 —— 那件事的正确归宿是持久化的每-shape 记录 (缺口 2), 而不是又一堆
# 硬编码常量。数据本身留在 benchmarks/ 与 docs/research/42, 不在这里留没人读的副本。
_PEAK_BY_CC: dict[str, tuple[float, float, float, float, float]] = {
    # 回退表: 只有设备名没命中 _PEAK_BY_NAME 时才会用到。
    # 原先 "8.9"/"9.0" 两行都填着 H100 的数字, 而 fp16 那一列用的是 **2:4 sparse** 的
    # 1979 (被删掉的那行注释自己写着 "FP8=1979") —— 表里其余每行都满足 fp16 == bf16,
    # 只有它们差 2.0 倍, 列错位不需要外部资料也能看出来。后果是有后果的: fp16 核的
    # 达成率被这个分母砍半, D4 的收敛出口在这类卡上永不打开。
    # 两个死键 "9.0a"/"8.9-bw" 已删 (匹配只用 `cc in 表` 与 `cc.startswith(键)`, 而 cc
    # 来自 f"{major}.{minor}", 永远不会以 "…a" 结尾)。
    #
    # ⚠️ 分母偏大 vs 偏小的不对称: 偏大只是让判停出口不开 (白烧轮次), 偏小会**提前停**
    # (把还能优化的核判成已到顶)。所以拿不准时一律取较大的那个口径。
    "8.0":  (19.5,  156.0, 312.0, 312.0, 2039.0),  # A100 (80GB, HBM2e)
    "8.6":  (19.5,  156.0, 312.0, 312.0, 1555.0),  # A100 (40GB), A40
    # 8.9 是 Ada 而不是 H100 (hardware_caps 自己写着 h100 的 cc 是 9.0)。按 L40S 填:
    # 362 是 **FP16 累加** 的 dense 率; 用 FP32 累加时上限其实是 181。取 362 是因为
    # 上面那条不对称 —— 偏大只是不出口, 偏小会提前停。
    "8.9":  (91.3,  181.0, 362.0, 362.0, 864.0),
    "9.0":  (67.0,  494.7, 989.4, 989.4, 3350.0),  # H100 SXM dense (cc 9.0 也含 H20/H200)
    "7.5":  (16.3,  130.6, 326.0, 326.0, 936.0),   # RTX 3090 / A40 (⚠️口径未核)
    "7.0":  (15.7,  0.0,   125.0, 125.0, 900.0),   # V100 (Volta, no TF32)
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


class Verifier:
    """
    验证器 - 执行三级验证
    L1 Compile -> L2 Function -> L3 Performance + Roofline
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.name = "Verifier"
        # 能力**快照**: 门面/提供方的 `capabilities` 可能是延迟解析的 property,
        # 每次分支都去读它会让「同一轮验证内判据变了」这种事在原理上可能发生。
        self._caps = frozenset(backend.capabilities)
        # 编译缓存: sha256(code) -> 通过标记, 相同代码重复验证时跳过编译
        self._compile_cache: dict[str, bool] = {}
        # baseline 缓存: (算子, 输入签名) -> 耗时, 避免多轮验证重复测量
        self._baseline_cache: dict = {}
        # 硬件理论峰值缓存 (延迟探测)
        self._theoretical_peak: Optional[dict] = None

    def verify(
        self,
        kernel_code: str,
        task: TaskSchema,
        test_inputs: dict,
        expected_output=None,
        target_gflops: float = 10.0,
        environment_drift: Optional[list] = None,
    ) -> list[VerificationResult]:
        """执行完整三级验证

        `environment_drift`: 本次运行的环境指纹与上一次相比变了哪些项
        (`environment.compare_fingerprints` 的输出)。它是 L3 有效性判定的一个输入,
        所以必须由**拿着产物库的那一层** (Conductor) 传进来 —— Verifier 自己不知道
        「上一次」是谁。None = 没做比较, 与「比过了没变」(空列表) 是两件事。
        """
        results = []

        l1 = self._verify_compile(kernel_code, task.target.dsl,
                                  input_key_sets(test_inputs))
        results.append(l1)
        if not l1.passed:
            return results

        l2 = self._verify_function(kernel_code, test_inputs, expected_output, task)
        results.append(l2)
        if not l2.passed:
            return results

        l3 = self._verify_performance(kernel_code, test_inputs, target_gflops, task,
                                      environment_drift=environment_drift)
        results.append(l3)
        return results

    def _verify_compile(self, kernel_code: str, dsl: str = "python",
                        input_key_sets=None) -> VerificationResult:
        print("  [Verifier] L1: 编译验证...")
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
        mismatch = None
        for keys in (input_key_sets or []):
            mismatch = check_kernel_interface(kernel_code, keys)
            if mismatch:
                break
        if mismatch:
            print("  [Verifier] L1 FAILED: 核函数接口与调用契约不符 (未动 GPU)")
            return VerificationResult(
                level=VerificationLevel.L1_COMPILE, passed=False,
                message="编译失败: " + mismatch.split("\n")[0],
                error_log=mismatch,
                metrics={"compile_status": "interface_mismatch",
                         "backend": str(self.backend)},
            )
        # 资源可行性预检 (LeetCUDA bench/utils.cu 的 smem 预检同款):
        # 这套 BLOCK 放不下时, 真启动只会给一句 `out of resource`, 看不出是哪个
        # BLOCK_*; 一次 SSH 往返 ~2.6s, 而这里只是正则。宁可漏拦 (超限 20% 内放行)
        # 也不误拦 —— 但拦下来的反馈必须带上 BLOCK 形状。
        try:
            from .resource_check import check_resource_feasibility
            caps = getattr(self, "_l1_caps", "unset")
            if caps == "unset":
                try:
                    caps = detect_hardware_caps(self.backend.get_hardware_info())
                except Exception:  # noqa: BLE001
                    caps = None
                self._l1_caps = caps
            if caps is not None:
                rep = check_resource_feasibility(kernel_code, caps, "")
                if not rep.feasible:
                    msg = rep.as_message()
                    print(f"  [Verifier] L1 FAILED: {msg}")
                    return VerificationResult(
                        level=VerificationLevel.L1_COMPILE, passed=False,
                        message=f"编译失败: {msg}",
                        error_log=msg,
                        metrics={"compile_status": "resource_infeasible",
                                 "backend": str(self.backend),
                                 "estimated_smem_bytes": rep.estimated_smem_bytes,
                                 "smem_limit_bytes": rep.limit_bytes},
                    )
        except Exception:  # noqa: BLE001 — 预检自身出错不挡主流程
            pass
        import hashlib
        cache_key = hashlib.sha256(kernel_code.encode("utf-8")).hexdigest()
        if cache_key in self._compile_cache:
            print("  [Verifier] L1 PASSED: 编译缓存命中")
            return VerificationResult(level=VerificationLevel.L1_COMPILE, passed=True, message="编译验证通过 (缓存命中)", metrics={"compile_status": "cached", "backend": str(self.backend)})
        try:
            self.backend.compile(kernel_code)
            self._compile_cache[cache_key] = True
            print("  [Verifier] L1 PASSED: 编译成功")
            return VerificationResult(level=VerificationLevel.L1_COMPILE, passed=True, message="编译验证通过", metrics={"compile_status": "success", "backend": str(self.backend)})
        except Exception as e:
            print(f"  [Verifier] L1 FAILED: {e}")
            # 远程编译错误全文可达上千字符 (gcc/Triton 会回显源码), 只把首行放进 message,
            # 否则 Fixer prompt 里同一份全文会出现两遍
            head, full = failure_pair(e)
            return VerificationResult(level=VerificationLevel.L1_COMPILE, passed=False,
                                      message=f"编译失败: {head}", error_log=full)

    def _verify_function(self, kernel_code: str, test_inputs, expected_output, task: TaskSchema) -> VerificationResult:
        print("  [Verifier] L2: 功能验证...")

        # 反作弊静态审计: 执行前先检查是否存在原生算子回退调用
        #   注意: 昇腾等自研加速卡白名单放开 torch.matmul 等 (因为它们的 Tensor Core
        #   只能通过厂商的 torch 后端下放激活)
        violations = static_audit(kernel_code, task.target.dsl, hardware=task.target.hardware)
        if violations:
            print(f"  [Verifier] L2 FAILED: 反作弊静态审计未通过 ({len(violations)} 处违规)")
            return VerificationResult(
                level=VerificationLevel.L2_FUNCTION, passed=False,
                message=f"反作弊: 检测到禁用的原生算子回退调用 ({len(violations)} 处)",
                error_log="\n".join(violations),
            )

        # 支持单组输入 (dict) 或多组输入套件 (list[dict])
        input_sets = test_inputs if isinstance(test_inputs, list) else [test_inputs]
        max_error_overall = 0.0
        max_rel_error_overall = 0.0
        total_compared = 0
        total_bad = 0
        min_matched_ratio = 1.0
        min_coverage = 1.0
        # 每组输入各自上报它**实际用的那把尺**; 是否合成一个单值, 由循环后的
        # `measurement.single_ruler` 判 (见那里的注释 —— 旧写法是「第一组赢」)
        gates: list[dict] = []
        # 逐组的高精度读数; 聚合规则见 `_aggregate_oracles` (那里记了第一版漏搬的现场)
        oracles: list[dict] = []
        has_reference = False
        reference_types = set()
        execution_devices = set()

        for idx, inputs in enumerate(input_sets):
            # 外部提供的 expected_output 仅对单组输入有效
            single_expected = expected_output if len(input_sets) == 1 else None
            r = self._verify_single_set(kernel_code, inputs, single_expected, task)
            if not r.passed:
                r.message = f"[输入集 {idx + 1}/{len(input_sets)}] {r.message}"
                print(f"  [Verifier] L2 FAILED: {r.message}")
                return r
            m = r.metrics
            if "max_error" in m:
                has_reference = True
                max_error_overall = max(max_error_overall, m["max_error"])
                max_rel_error_overall = max(max_rel_error_overall, m.get("max_rel_error", 0.0))
                total_compared += m.get("compared_elements", 0)
                total_bad += m.get("bad_elements", 0)
                # 多组输入取最差的一组: 任一组掉到 rho 以下就不该算通过
                min_matched_ratio = min(min_matched_ratio, m.get("matched_ratio", 1.0))
                min_coverage = min(min_coverage, m.get("coverage", 1.0))
                g = {k: m[k] for k in GATE_PARAM_KEYS if k in m}
                if g:
                    gates.append(g)
                oracles.append({k: v for k, v in m.items()
                                if k == "ref_oracle" or k.startswith("oracle_")})
                if m.get("reference_type"):
                    reference_types.add(m["reference_type"])
                if m.get("device"):
                    execution_devices.add(m["device"])

        gate_params, gate_conflicts = single_ruler(gates)
        oracle_agg = self._aggregate_oracles(oracles)
        if has_reference:
            msg = (f"功能验证通过 ({len(input_sets)} 组输入), 最大误差={max_error_overall:.2e}, "
                   f"matched_ratio={min_matched_ratio:.6f}")
            if min_coverage < 1.0:
                msg += f", 覆盖率={min_coverage * 100:.1f}%"
            if gate_conflicts:
                msg += (f" ⚠️ 这几组用的**不是同一把尺** {sorted(gate_conflicts)}, "
                        f"聚合值因此不发布单值 (见 metrics.gate_params_conflict)")
            # 聚合这一支的 message 是**重算**的, 每组自己那句被丢掉了 ⇒ 高精度那把尺的话
            # 必须在这里接上, 否则「通过」那一侧永远读不到它 (与 _aggregate_oracles 同一条教训)
            msg += self._oracle_note(oracle_agg)
            metrics = {
                "max_error": max_error_overall,
                "max_rel_error": max_rel_error_overall,
                "matched_ratio": min_matched_ratio,
                "bad_elements": total_bad,
                "num_input_sets": len(input_sets),
                "compared_elements": total_compared,
                "coverage": min_coverage,
                **gate_params,
                **oracle_agg,
            }
            if gate_conflicts:
                # 摊开成 {键: {组下标: 那一组实际用的值}} —— 读的人不必重跑就能看出
                # 是哪几组、哪一把尺不一致
                metrics["gate_params_conflict"] = gate_conflicts
            if reference_types:
                metrics["reference_type"] = ",".join(sorted(reference_types))
            if execution_devices:
                metrics["device"] = ",".join(sorted(execution_devices))
        else:
            msg = f"功能验证通过 ({len(input_sets)} 组输入, 无参考输出, 仅验证可执行性)"
            metrics = {"num_input_sets": len(input_sets)}
        print(f"  [Verifier] L2 PASSED: {msg}")
        return VerificationResult(
            level=VerificationLevel.L2_FUNCTION, passed=True, message=msg, metrics=metrics,
        )

    def _verify_single_set(self, kernel_code: str, inputs: dict, expected_output, task: TaskSchema) -> VerificationResult:
        """对单组输入执行功能验证
        metrics 中含 max_error 表示完成了参考比对，否则仅为可执行性验证"""
        max_allowed = task.constraints.max_error

        if Cap.REMOTE_VERIFY in self._caps:
            # 候选与 reference 均在远端执行设备上比较，仅返回摘要，
            # 避免把完整张量通过 SSH 回传到控制端。
            remote_verify = self.backend.verify_function(
                kernel_code, inputs, task.source.kernel_type, max_allowed,
                precision=task.constraints.precision.value,
                reference_source=task.source.reference_source or "",
            )
            # 声明了 REMOTE_VERIFY 就必须给出载荷。「内层没这个方法」已经由 Backend
            # 基类那个一调就抛的存根拦掉了, 走到这里的 None 只能是「声明了却没数据」——
            # 把它读成「换条路再试试」正是 doc 49 §3.2 B3: 两个意思共用一个值。
            if remote_verify is None:
                raise CapabilityMissing(Cap.REMOTE_VERIFY, self.backend,
                                        "verify_function 声明可用但返回了 None")
            if not remote_verify.get("success", False):
                # ``success=False`` 也可能是数值误差超标；保留远端的精确诊断。
                if "max_error" in remote_verify:
                    return VerificationResult(
                        level=VerificationLevel.L2_FUNCTION, passed=False,
                        message=f"远程功能验证失败: 误差 {remote_verify['max_error']:.2e}",
                        error_log=remote_verify.get("error", ""),
                        metrics={k: remote_verify[k] for k in _REMOTE_L2_METRIC_KEYS
                                 if k in remote_verify},
                    )
                if remote_verify.get("output_shape") and remote_verify.get("expected_shape"):
                    return VerificationResult(
                        level=VerificationLevel.L2_FUNCTION, passed=False,
                        message=(f"输出形状不匹配: {remote_verify['output_shape']} vs "
                                 f"{remote_verify['expected_shape']}"),
                        error_log=remote_verify.get("error", ""),
                    )
                # message 只带首行, 全文放 error_log (见 failure_pair)
                head, full = failure_pair(remote_verify.get("error", ""))
                return VerificationResult(
                    level=VerificationLevel.L2_FUNCTION, passed=False,
                    message=f"远程执行失败: {head}",
                    error_log=full)
            return VerificationResult(
                level=VerificationLevel.L2_FUNCTION, passed=True,
                message=(f"远程全量比对通过 ({remote_verify.get('compared_elements', 0)} 元素), "
                         f"最大误差={remote_verify.get('max_error', 0):.2e}"),
                metrics={k: remote_verify[k] for k in _REMOTE_L2_METRIC_KEYS
                         if k in remote_verify},
            )

        # 没有远端紧凑比对协议: 在执行侧跑一遍, 由 Verifier 自己比对参考输出。
        result = self.backend.execute(kernel_code, inputs)
        if not result.success:
            head, full = failure_pair(result.error)
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"执行失败: {head}", error_log=full)
        if expected_output is None:
            expected_output = self._generate_reference(task, inputs)
        if expected_output is None:
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=True, message="无参考输出，仅验证可执行性", metrics={"execution_time_ms": result.execution_time_ms})

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
            coverage = compared / ref_flat.size if ref_flat.size else 1.0
            max_error = float(np.max(np.abs(sample - ref_flat[:compared])))
            metrics = {"max_error": max_error, "compared_elements": compared, "coverage": coverage}
            if max_error <= max_allowed:
                return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=True, message=f"采样比对通过 ({compared}/{ref_flat.size} 元素), 最大误差={max_error:.2e}", metrics=metrics)
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"功能验证失败: 误差 {max_error:.2e}", metrics=metrics)

        actual = result.output
        expected = np.asarray(expected_output, dtype=np.float32)
        if actual.shape != expected.shape:
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"输出形状不匹配: {actual.shape} vs {expected.shape}")

        # 数值门禁与真机路径 (remote_scripts/run_kernel.py) 共用 tolerance.py 里的
        # 同一张容差表和同一条判定式: 绝对 + 相对 的 AND 形式, 见该模块 docstring。
        atol, rtol, rho = tolerance.effective_atol(
            task.constraints.precision.value, max_allowed)
        actual_f = np.asarray(actual, dtype=np.float32)
        abs_error = np.abs(actual_f - expected)
        rel_error = abs_error / (np.abs(expected) + tolerance.EPS)
        bad = (abs_error > atol) & (rel_error > rtol)
        has_nonfinite = bool((~np.isfinite(actual_f)).any())
        passed, matched_ratio = tolerance.evaluate(
            int(bad.sum()), int(abs_error.size), has_nonfinite, rho)

        size = int(expected.size)
        metrics = {
            "max_error": float(abs_error.max()) if size else 0.0,
            "max_rel_error": float(rel_error.max()) if size else 0.0,
            "matched_ratio": matched_ratio,
            "bad_elements": int(bad.sum()),
            "atol": atol, "rtol": rtol, "required_matched_ratio": rho,
            "compared_elements": size, "coverage": 1.0,
        }
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
        if passed:
            return VerificationResult(
                level=VerificationLevel.L2_FUNCTION, passed=True,
                message=(f"全量比对通过 ({size} 元素), 最大误差={metrics['max_error']:.2e}, "
                         f"matched_ratio={matched_ratio:.6f}" + self._oracle_note(metrics)),
                metrics=metrics)
        reason = ("输出含 NaN/Inf" if has_nonfinite else
                  f"{metrics['bad_elements']}/{size} 个元素同时超 atol={atol:.1e} 与 rtol={rtol:.1e}")
        if "first_bad_index" in metrics:
            reason += (f", first_bad@{metrics['first_bad_index']}"
                       f" (got {metrics['first_bad_actual']:.6g},"
                       f" want {metrics['first_bad_expected']:.6g})")
        return VerificationResult(
            level=VerificationLevel.L2_FUNCTION, passed=False,
            message=f"功能验证失败: {reason}, 最大误差={metrics['max_error']:.2e}",
            metrics=metrics)

    def _oracle_reading(self, task: TaskSchema, inputs: dict,
                        actual, expected) -> dict:
        """高精度参考 (同一份参考、float64 再算一遍) 那把尺的读数 —— **只观测, 不判定**

        外部一手: KernelOPT `2609.30059v1` 行 372-389 与行 1344-1362 (规则抄在
        `tolerance.error_ratio` 那里, 包括「这三个阈是别人语料上的数」那句警告)。

        与 AND 那条的差别值得再说一遍: AND 比「候选 vs 参考」, 条带是绝对 atol ⇒ 参考幅值一小
        就看不见任何东西 (`benchmarks/probe_l2_sampling_gate.py` 臂 1/2 拍到的全零核 PASS)。
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

    def _verify_performance(self, kernel_code: str, test_inputs, target_gflops: float,
                            task: TaskSchema,
                            environment_drift: Optional[list] = None) -> VerificationResult:
        min_required = task.constraints.min_speedup
        print(f"  [Verifier] L3: 性能验证 (要求加速比 >= {min_required:.2f}x vs baseline)...")
        # 多组输入套件时，性能基准使用第一组输入
        bench_inputs = test_inputs[0] if isinstance(test_inputs, list) else test_inputs
        remote_bench = None
        if Cap.SAME_DEVICE_BASELINE in self._caps:
            remote_bench = self.backend.benchmark_with_baselines(
                kernel_code, bench_inputs, task.source.kernel_type,
                precision=task.constraints.precision.value,
                baseline_priority=baseline_plan(task.source.kernel_type,
                                                task.constraints.precision.value),
                reference_source=task.source.reference_source or "",
            )
        bench = self.backend.benchmark(kernel_code, bench_inputs, warmup=2, repeat=5) if remote_bench is None else remote_bench

        if remote_bench is not None and not remote_bench.get("success", False):
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
            exec_result = self.backend.execute(kernel_code, bench_inputs)
            actual_gflops = exec_result.gflops
            exec_time = bench["avg_time_ms"]
            bandwidth = exec_result.memory_bandwidth_gbs

        candidate_metrics = bench.get("candidate", {}) if remote_bench is not None else bench
        kernel_min_ms = candidate_metrics.get("median_time_ms") or candidate_metrics.get("min_time_ms") or exec_time

        # ---- Roofline 分析: 基于实测数据判断真实瓶颈 ----
        roofline = self._roofline_analysis(
            task=task, bench_inputs=bench_inputs,
            actual_gflops=actual_gflops, actual_bandwidth_gbs=bandwidth,
        )
        bound = roofline["bound_type"]  # 替代原有的名字匹配 bound

        metrics = {"avg_time_ms": exec_time, "min_time_ms": kernel_min_ms,
                   "gflops": round(actual_gflops, 4),
                   "memory_bandwidth_gbs": round(bandwidth, 4),
                   "bound_type": bound, "primary_metric": "speedup_vs_baseline",
                   "backend": str(self.backend)}
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
                          "selected_gpu",
                          # D2: 窗口边界上的 SM/显存时钟与 throttle 位图。搬运是必须的
                          # —— 缺口 1 的教训就是「runner 报了但控制端没人搬, 于是没人读」。
                          "sm_clock_mhz_before", "sm_clock_mhz_after",
                          "sm_clock_max_mhz_before", "sm_clock_max_mhz_after",
                          "mem_clock_mhz_before", "mem_clock_mhz_after",
                          "throttle_bits_before", "throttle_bits_after"):
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
        })

        if roofline["hw_peak_gflops"] > 0 and roofline["hw_peak_bandwidth_gbs"] > 0:
            compute_label = roofline.get("compute_used", "fp32_cuda_core")
            # 分母的出处必须出现在分母被读到的那一行: 判停出口 (ROOFLINE_CONVERGED_PCT)
            # 直接拿达成率当证据, 而「这个 peak 凭什么」以前只在源码注释里, 日志里看不到。
            print(f"  [Verifier] Roofline: {roofline['roofline_bound']} "
                  f"(AI={roofline['arithmetic_intensity']:.2f} FLOP/Byte, "
                  f"peak={roofline['hw_peak_gflops']/1000:.1f} TFLOPS ({compute_label}|"
                  f"分母[{roofline['peak_table_key']}] {roofline['peak_source']}), "
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
                metrics.update({
                    "baseline_type": baseline_name,
                    "baseline_display": display_name,
                    "baseline_time_ms": round(baseline_ms, 6),
                    "speedup_vs_baseline": round(speedup, 4),
                    # 三元组原文, 落盘可直接比 (LeetCUDA bench 的 "%.1f/%.1f (%.2fx)")
                    "ratio_triple": f"{kernel_min_ms:.4g}/{baseline_ms:.4g} ({speedup:.2f}x)",
                    "min_speedup_required": min_required,
                    "measurement_scope": "same_remote_device",
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
                })
                passed = speedup >= min_required
                # 三元组 (LeetCUDA bench 同款): 候选/基线 (倍率)。只报一个 1.04x
                # 时读者看不见两边的绝对量级 —— 「0.01/0.0096 (1.04x)」与
                # 「10.4/10.0 (1.04x)」是两回事, 前者可能整个被 launch 地板吞掉。
                ratio_triple = (f"{kernel_min_ms:.4g}/{baseline_ms:.4g} ({speedup:.2f}x)")
                msg = (f"加速比 {ratio_triple} vs {display_name} "
                       f"(同机同卡 median, 要求 >= {min_required:.2f}x, {bound}-bound)")
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
                    environment_drift=environment_drift)
                passed = self._refuse_l3_pass(passed, assessment)
                if not metrics.get("benchmark_contaminated") and others:
                    msg += f" (注: 卡与 {others} 个其他进程共享, 瞬时 util={util}%)"

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
            })
            passed = speedup >= min_required
            base_ms = baseline["baseline_time_ms"]
            ratio_triple = f"{kernel_min_ms:.4g}/{base_ms:.4g} ({speedup:.2f}x)"
            msg = (f"加速比 {ratio_triple} vs {baseline['baseline_type']} "
                   f"(要求 >= {min_required:.2f}x, {bound}-bound)")
            msg, assessment = self._assess_measurement(
                metrics, msg, kernel_min_ms, task=task,
                environment_drift=environment_drift)
            passed = self._refuse_l3_pass(passed, assessment)
            print(f"  [Verifier] L3 {'PASSED' if passed else 'FAILED'}: {msg}")
            return VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                                      passed=passed, message=msg, metrics=metrics)

        # 回退判定: 无参考实现的算子使用绝对 GFLOPS 目标 (legacy)
        achieve = actual_gflops / target_gflops if target_gflops > 0 else 0.0
        metrics.update({"target_gflops": target_gflops, "achieve_ratio": round(achieve, 4)})
        passed = target_gflops <= 0 or actual_gflops >= target_gflops * min_required
        msg = (f"无 baseline 参考, 按绝对目标判定: {actual_gflops:.2f} GFLOPS "
               f"(目标 {target_gflops * min_required:.2f})")
        msg, assessment = self._assess_measurement(
            metrics, msg, kernel_min_ms, task=task,
            environment_drift=environment_drift)
        passed = self._refuse_l3_pass(passed, assessment)
        print(f"  [Verifier] L3 {'PASSED' if passed else 'FAILED'}: {msg}")
        return VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                                  passed=passed, message=msg, metrics=metrics)

    @staticmethod
    def _refuse_l3_pass(passed: bool, assessment) -> bool:
        """出处不对的 L3 不许记成通过 (doc 49 §3.5 第 2/3 处 + §五 第 4 片验收)

        与 `can_gate` 分开: 档位 `uninformative` 只说明「这个数字撑不住结论」, 历史上
        它并不把 `passed` 拉成 False (launch 地板支配那一档的核可能确实是对的)。
        而**任务声明了要真卡、执行侧给不出**是另一回事 —— 那份加速比不是测得不准,
        是压根没测, 「通过」这个结论没有对象。
        """
        return False if (passed and assessment.l3_pass_is_refused) else passed

    def _assess_measurement(self, metrics: dict, msg: str, candidate_ms,
                            gpu_util_pct=None, task: Optional[TaskSchema] = None,
                            environment_drift: Optional[list] = None):
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
            launch_floor_us=self._detect_theoretical_peak().get("launch_floor_us"),
            gpu_util_pct=gpu_util_pct,
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
        if assessment.contaminated:
            metrics["contamination_reason"] = f"测量前 util={gpu_util_pct}%"
        if assessment.reasons:
            msg += (f"  ⚠️ 测量有效性={assessment.validity.value}: "
                    + "; ".join(assessment.reasons))
        if assessment.l3_pass_is_refused:
            msg += ("  ⛔ 本轮 L3 不记为通过: 任务声明要的执行侧能力没拿到 "
                    f"(缺 {', '.join(assessment.capability_shortfall) or '—'}"
                    f"{'; 且 ' + '; '.join(assessment.provenance_conflict) if assessment.provenance_conflict else ''})")
        return msg, assessment

    # 算子类型 -> bound 分类 (决定优化时关注的参考指标)
    _COMPUTE_BOUND_OPS = ("matmul", "gemm", "conv", "attention", "einsum")
    _MEMORY_BOUND_OPS = ("softmax", "layernorm", "layer_norm", "batchnorm",
                         "batch_norm", "relu", "gelu", "sigmoid", "add",
                         "dropout", "norm", "elementwise")

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

    def _detect_theoretical_peak(self) -> dict:
        """探测硬件理论峰值 (GFLOPS, 带宽)，优先按设备名匹配，回退 compute_capability"""
        if self._theoretical_peak is not None:
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

        # launch 地板走 hardware_caps 而不是上面这两张表: 它是实测值而非 datasheet
        # 推算值, 且按缺口 3 的方向, 硬件知识应该收敛到 HardwareCaps 一处。
        # 取不到 (未知设备 / get_hardware_info 抛错) 时为 None, 下游会显式记录
        # 「未做地板支配检查」而不是默认放行。
        launch_floor_us = None
        try:
            launch_floor_us = detect_hardware_caps(hw).launch_floor_us
        except Exception:
            pass

        self._theoretical_peak = {
            "fp32_tflops": fp32, "tf32_tflops": tf32,
            "fp16_tflops": fp16, "bf16_tflops": bf16,
            "memory_bandwidth_gbs": bw,
            "peak_source": peak_source,
            "peak_trusted": peak_trusted,
            "peak_table_key": matched_key or "",
            "launch_floor_us": launch_floor_us,
            "compute_capability": cc, "device": device, "name": name,
        }
        return self._theoretical_peak

    @staticmethod
    def _compute_arithmetic_intensity(task: TaskSchema, inputs: dict) -> float:
        """计算算子的算数强度 (FLOPs / Bytes transferred)
        返回 FLOP/Byte 或 0 (无法计算时)

        ⚠️ Byte 轴用的是**任务声明精度**的字节数, 不是 `inputs` 里数组自己的 dtype。
        runner 会按声明精度转换输入 (`remote_scripts/run_kernel.py`: precision == "fp16"
        时 `.to(torch.float16)`), 所以控制端手里那份 float32 数组是核**根本见不到**的形态。
        第一版就是这么写的: matmul 4096³ 声明 FP16 而输入是 float32, AI 算成 682.67
        而不是 1365.33 —— 正好差 2 倍 (doc 48 §6.5)。达成率与计时对比都不受影响,
        但 AI 决定脊点两侧的分类, 而分类决定收敛出口与 SearchAgent 走哪条建议。
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
                    return 0.0
                flops = 2.0 * M * N * K
                bytes_transferred = (M * K + K * N + M * N) * nbytes
                return flops / bytes_transferred if bytes_transferred > 0 else 0.0

        if "softmax" in kt:
            # Softmax: ~5*N FLOPs per element (exp, sum, div), memory: N * dtype
            if "X" in inputs:
                x = np.asarray(inputs["X"])
                n = int(np.prod(x.shape))
                flops = 5.0 * n
                bytes_transferred = 2.0 * n * nbytes  # read + write
                return flops / bytes_transferred if bytes_transferred > 0 else 0.0

        if any(op in kt for op in ("relu", "gelu", "sigmoid", "add", "elementwise")):
            # Element-wise: ~1 FLOP per element, memory: 2-3 * N * dtype
            for key in ("X", "input", "A"):
                if key in inputs:
                    x = np.asarray(inputs[key])
                    n = int(np.prod(x.shape))
                    flops = 1.0 * n
                    bytes_transferred = 2.0 * n * nbytes
                    return flops / bytes_transferred if bytes_transferred > 0 else 0.0

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
                    return flops / bytes_transferred if bytes_transferred > 0 else 0.0

        return 0.0

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

        ai = self._compute_arithmetic_intensity(task, bench_inputs)

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
            # 分母的出处跟着分母走。缺口 3 的病根不是某个数字错, 而是**用这个数字的人
            # 看不出它有多可信**: 138.2 和 1979 在日志里长得一模一样。
            "peak_source": peak.get("peak_source", ""),
            "peak_trusted": peak.get("peak_trusted", False),
            "peak_table_key": peak.get("peak_table_key", ""),
        }

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
        from operator_agent.reference import ReferenceSourceError, resolve_reference
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
                a = torch.from_numpy(np.asarray(inputs["A"], dtype=dtype_np)).to(dtype_torch)
                b = torch.from_numpy(np.asarray(inputs["B"], dtype=dtype_np)).to(dtype_torch)
                return lambda: (a @ b), "pytorch_eager"
            a = np.asarray(inputs["A"], dtype=dtype_np)
            b = np.asarray(inputs["B"], dtype=dtype_np)
            return lambda: (a @ b), "numpy_reference"

        if "softmax" in kernel_type and "X" in inputs:
            if torch_available:
                import torch.nn.functional as F
                x = torch.from_numpy(np.asarray(inputs["X"], dtype=dtype_np)).to(dtype_torch)
                return lambda: F.softmax(x, dim=-1), "pytorch_eager"
            x = np.asarray(inputs["X"], dtype=dtype_np)

            def _softmax():
                mx = np.max(x, axis=-1, keepdims=True)
                ex = np.exp(x - mx)
                return ex / np.sum(ex, axis=-1, keepdims=True)
            return _softmax, "numpy_reference"

        # 更多算子: 优先 torch 原生 (与 eager 对齐)
        if torch_available:
            x_np = np.asarray(inputs.get("X") or inputs.get("input"), dtype=dtype_np)
            if x_np is not None:
                x = torch.from_numpy(x_np).to(dtype_torch)
                if "relu" in kernel_type:
                    return lambda: torch.relu(x), "pytorch_eager"
                if "sigmoid" in kernel_type:
                    return lambda: torch.sigmoid(x), "pytorch_eager"
                if "gelu" in kernel_type:
                    return lambda: torch.nn.functional.gelu(x), "pytorch_eager"
                if "layernorm" in kernel_type or "layer_norm" in kernel_type:
                    w_np = np.asarray(inputs.get("weight"), dtype=dtype_np) if inputs.get("weight") is not None else None
                    b_np = np.asarray(inputs.get("bias"), dtype=dtype_np) if inputs.get("bias") is not None else None
                    if w_np is not None and b_np is not None:
                        w = torch.from_numpy(w_np).to(dtype_torch)
                        b = torch.from_numpy(b_np).to(dtype_torch)
                        return lambda: torch.nn.functional.layer_norm(x, x.shape[-1:], w, b), "pytorch_eager"
                    return lambda: torch.nn.functional.layer_norm(x, x.shape[-1:]), "pytorch_eager"
                if "batchnorm" in kernel_type or "batch_norm" in kernel_type:
                    return lambda: torch.nn.functional.batch_norm(
                        x, torch.ones(x.shape[-1]), torch.zeros(x.shape[-1]),
                        training=True), "pytorch_eager"

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
            return self._baseline_cache[cache_key]

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
        self._baseline_cache[cache_key] = result
        return result

    @staticmethod
    def _baseline_cache_key(task: TaskSchema, inputs: dict) -> tuple:
        sig = tuple(sorted(
            (k, tuple(np.asarray(v).shape), str(np.asarray(v).dtype))
            for k, v in inputs.items()
        ))
        return (task.source.kernel_type.lower(), sig)
