"""
验证层 - 三级验证体系
L1: 编译验证 (代码可编译、语法正确)
L2: 功能验证 (输出正确，误差在容许范围内)
L3: 性能验证 (性能达到预期加速比) + Roofline 分析

后端可以是 MockGPU 或 RemoteExecutor，接口统一
"""

import time
import numpy as np
from typing import Union, Optional
from .schemas import (
    VerificationResult, VerificationLevel, TaskSchema, Precision,
)
from .anticheat import static_audit
from . import tolerance
from . import measurement
from .hardware_caps import detect_hardware_caps
from .mock_gpu import MockGPU
from .remote_executor import RemoteExecutor, RemoteExecutionResult


# 远端 L2 返回的指标里需要透传到控制端的键。matched_ratio / bad_elements / atol /
# rtol 是数值门改成「绝对+相对」之后新增的: 只留 max_error 的话, 控制端分不清
# 「差一点点」和「整体缩放了 3%」, 而后者正是 docs/research/12 要拦的那类。
_REMOTE_L2_METRIC_KEYS = (
    "max_error", "max_rel_error", "matched_ratio", "bad_elements",
    "atol", "rtol", "required_matched_ratio",
    "compared_elements", "coverage", "reference_type", "device",
)


# 常见 GPU 硬件理论峰值 (FP32/TF32/FP16/BF16 Tensor Core + 显存带宽)
# 格式: (FP32_TFLOPS, TF32_TFLOPS, FP16_TFLOPS, BF16_TFLOPS, memory_bandwidth_GBs)
# 匹配顺序: 先设备名 (更具体), 未命中再回退 compute_capability。
# 不可反过来 —— cc 9.0 同时对应 H100/H200/H20, 三者算力带宽差异极大。
_PEAK_BY_NAME: dict[str, tuple[float, float, float, float, float]] = {
    "a100": (19.5,  156.0, 312.0, 312.0, 2039.0),
    "h100": (67.0,  989.0, 1979.0, 989.0, 3350.0),
    "h200": (67.0,  989.0, 1979.0, 989.0, 3350.0),
    # H20: 2026-09-16 于 algorithm-h20 实测 (cuBLAS 8192³ / triad), 非 datasheet
    # 算力砍到 138 TFLOPS 但带宽保留 3.68 TB/s → ridge ≈ 37.5 (H100 ≈ 295)
    "h20":  (31.6,  71.6,  138.2, 138.3, 3680.0),
    "v100": (15.7,  0.0,   125.0, 125.0, 900.0),
    "3090": (16.3,  130.6, 326.0, 326.0, 936.0),
    "4090": (82.6,  82.6,  330.0, 330.0, 1008.0),  # 4090 FP32=82.6 (no TF32 boost for Ada? actually Ada has TF32)
    "a6000":(38.7,  310.0, 77.4,  77.4,  768.0),
    "a5000":(27.8,  222.0, 55.5,  55.5,  768.0),
    "t4":   (8.1,   65.0,  130.0, 130.0, 320.0),
}

_PEAK_BY_CC: dict[str, tuple[float, float, float, float, float]] = {
    # ⚠️ 这张回退表里有两处**结构性**问题, 这里只删掉可证明永不再匹配的键, 数值本身
    # 待外部调研给出可追溯的出处后再改 (docs/research/00 缺口 3 有全部证据):
    #   1. fp16 那一列在 h100/h200/8.9/9.0 上写着 1979.0 —— 那是 **FP8** 的数值
    #      (被删掉的那行注释自己写着 "FP8=1979")。这条表里其余每一行都满足
    #      fp16 == bf16 (a100 312/312、v100 125/125、3090 326/326、h20 138.2/138.3),
    #      只有 Hopper 这两行差 2.0 倍 —— 不需要外部资料就能看出列错位。
    #      后果是有后果的: fp16 核的达成率被这个分母砍半, D4 的收敛出口在这类卡上永不打开。
    #   2. 键 "8.9" 标着 H100, 但 `hardware_caps` 自己写着 h100 的 cc 是 9.0 —— 8.9 是
    #      Ada。所以一张名字没命中 `_PEAK_BY_NAME` 的 Ada 卡 (如 L40S) 会拿到 H100 的
    #      算力与 HBM3 带宽, 于是「离上界还有多远」整个失真。这条**没删**, 因为我们没有
    #      Ada 机器可验, 删掉会让 roofline 分类直接失效 (0 峰值), 那是更大的行为改动。
    # 已删的两个键 "9.0a" / "8.9-bw" 是死键: 匹配只用 `cc in 表` 与 `cc.startswith(键)`,
    # 而 cc 的真实取值是 "8.9"/"9.0"/"10.0" 这类, 永远不会以 "…a" 或 "8.9-bw" 开头。
    "8.0":  (19.5,  156.0, 312.0, 312.0, 2039.0),  # A100 (80GB, HBM2e)
    "8.6":  (19.5,  156.0, 312.0, 312.0, 1555.0),  # A100 (40GB), A40
    "8.9":  (67.0,  989.0, 1979.0, 989.0, 3350.0),  # 标着 H100, 但 8.9 是 Ada —— 见上
    "9.0":  (67.0,  989.0, 1979.0, 989.0, 3350.0),  # H200 (est.)
    "7.5":  (16.3,  130.6, 326.0, 326.0, 936.0),   # RTX 3090 / A40 (GA102, Ampere has TF32)
    "7.0":  (15.7,  0.0,   125.0, 125.0, 900.0),   # V100 (Volta, no TF32)
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
    后端支持 MockGPU 和 RemoteExecutor
    """

    def __init__(self, backend: Union[MockGPU, RemoteExecutor]):
        self.backend = backend
        self.name = "Verifier"
        self._is_remote = isinstance(backend, RemoteExecutor)
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
    ) -> list[VerificationResult]:
        """执行完整三级验证"""
        results = []

        l1 = self._verify_compile(kernel_code, task.target.dsl)
        results.append(l1)
        if not l1.passed:
            return results

        l2 = self._verify_function(kernel_code, test_inputs, expected_output, task)
        results.append(l2)
        if not l2.passed:
            return results

        l3 = self._verify_performance(kernel_code, test_inputs, target_gflops, task)
        results.append(l3)
        return results

    def _verify_compile(self, kernel_code: str, dsl: str = "python") -> VerificationResult:
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
        gate_params = {}
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
                if not gate_params:
                    gate_params = {k: m[k] for k in
                                   ("atol", "rtol", "required_matched_ratio") if k in m}
                if m.get("reference_type"):
                    reference_types.add(m["reference_type"])
                if m.get("device"):
                    execution_devices.add(m["device"])

        if has_reference:
            msg = (f"功能验证通过 ({len(input_sets)} 组输入), 最大误差={max_error_overall:.2e}, "
                   f"matched_ratio={min_matched_ratio:.6f}")
            if min_coverage < 1.0:
                msg += f", 覆盖率={min_coverage * 100:.1f}%"
            metrics = {
                "max_error": max_error_overall,
                "max_rel_error": max_rel_error_overall,
                "matched_ratio": min_matched_ratio,
                "bad_elements": total_bad,
                "num_input_sets": len(input_sets),
                "compared_elements": total_compared,
                "coverage": min_coverage,
                **gate_params,
            }
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
        """对单组输入执行功能验证 (远程或本地 Mock 后端)
        metrics 中含 max_error 表示完成了参考比对，否则仅为可执行性验证"""
        max_allowed = task.constraints.max_error

        if self._is_remote:
            # 新协议: 候选与 reference 均在远端执行设备上比较，仅返回摘要，
            # 避免把完整张量通过 SSH 回传到控制端。
            remote_verify = self.backend.verify_function(
                kernel_code, inputs, task.source.kernel_type, max_allowed,
                precision=task.constraints.precision.value,
            )
            if remote_verify is not None:
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

            # 旧执行器兼容路径。新 SSHExecutor 不会进入这里。
            result = self.backend.execute(kernel_code, inputs)
            if not result.success:
                head, full = failure_pair(result.error)
                return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"执行失败: {head}", error_log=full)
            if expected_output is None:
                expected_output = self._generate_reference(task, inputs)
            if expected_output is None:
                return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=True, message="远程执行成功 (无参考输出)", metrics={"execution_time_ms": result.execution_time_ms})

            ref = np.asarray(expected_output, dtype=np.float32)
            # 形状检查 (此前远程模式完全不做形状校验)
            if result.output_shape and list(result.output_shape) != list(ref.shape):
                return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"输出形状不匹配: {result.output_shape} vs {list(ref.shape)}")

            ref_flat = ref.flatten()
            sample = np.array(result.output_sample[:ref_flat.size], dtype=np.float32) if result.output_sample else np.array([])
            if sample.size == 0:
                return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message="远程未返回输出样本，无法比对")

            compared = int(sample.size)
            coverage = compared / ref_flat.size if ref_flat.size else 1.0
            max_error = float(np.max(np.abs(sample - ref_flat[:compared])))
            metrics = {"max_error": max_error, "compared_elements": compared, "coverage": coverage}
            if max_error <= max_allowed:
                return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=True, message=f"远程比对通过 ({compared}/{ref_flat.size} 元素), 最大误差={max_error:.2e}", metrics=metrics)
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"功能验证失败: 误差 {max_error:.2e}", metrics=metrics)

        from .mock_gpu import ExecutionResult
        result: ExecutionResult = self.backend.execute(kernel_code, inputs)
        if not result.success:
            head, full = failure_pair(result.error)
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"执行失败: {head}", error_log=full)
        if expected_output is None:
            expected_output = self._generate_reference(task, inputs)
        if expected_output is None:
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=True, message="无参考输出，仅验证可执行性", metrics={"execution_time_ms": result.execution_time_ms})

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
        if passed:
            return VerificationResult(
                level=VerificationLevel.L2_FUNCTION, passed=True,
                message=(f"全量比对通过 ({size} 元素), 最大误差={metrics['max_error']:.2e}, "
                         f"matched_ratio={matched_ratio:.6f}"),
                metrics=metrics)
        reason = ("输出含 NaN/Inf" if has_nonfinite else
                  f"{metrics['bad_elements']}/{size} 个元素同时超 atol={atol:.1e} 与 rtol={rtol:.1e}")
        return VerificationResult(
            level=VerificationLevel.L2_FUNCTION, passed=False,
            message=f"功能验证失败: {reason}, 最大误差={metrics['max_error']:.2e}",
            metrics=metrics)

    def _verify_performance(self, kernel_code: str, test_inputs, target_gflops: float, task: TaskSchema) -> VerificationResult:
        min_required = task.constraints.min_speedup
        print(f"  [Verifier] L3: 性能验证 (要求加速比 >= {min_required:.2f}x vs baseline)...")
        # 多组输入套件时，性能基准使用第一组输入
        bench_inputs = test_inputs[0] if isinstance(test_inputs, list) else test_inputs
        remote_bench = None
        if self._is_remote:
            remote_bench = self.backend.benchmark_with_baselines(
                kernel_code, bench_inputs, task.source.kernel_type,
                precision=task.constraints.precision.value,
                baseline_priority=baseline_plan(task.source.kernel_type,
                                                task.constraints.precision.value),
            )
        bench = self.backend.benchmark(kernel_code, bench_inputs, warmup=2, repeat=5) if remote_bench is None else remote_bench

        if remote_bench is not None and not remote_bench.get("success", False):
            return VerificationResult(
                level=VerificationLevel.L3_PERFORMANCE, passed=False,
                message=f"远程性能测试失败: {remote_bench.get('error', '')}",
                error_log=remote_bench.get("error", ""),
            )

        if self._is_remote:
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
            print(f"  [Verifier] Roofline: {roofline['roofline_bound']} "
                  f"(AI={roofline['arithmetic_intensity']:.2f} FLOP/Byte, "
                  f"peak={roofline['hw_peak_gflops']/1000:.1f} TFLOPS ({compute_label}), "
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
                    "baseline_attempts": bench.get("baseline_attempts") or [],
                })
                passed = speedup >= min_required
                msg = (f"加速比 {speedup:.2f}x vs {display_name} "
                       f"(同机同卡 median, 要求 >= {min_required:.2f}x, {bound}-bound)")

                # 共享机器上基准可信度的两个信号, 强度不同, 分开记录:
                #   util_before > 5%  → 硬污染, 测量时确实在争用, 加速比不可比
                #   other_procs > 0   → 仅提示卡被共享 (这台机器上恒为真), 不单独判污染
                # 若把"有进程"也算污染, 每次运行都会被标记, 信号就失效了。
                util = bench.get("gpu_util_pct")
                others = bench.get("gpu_other_procs")
                metrics["gpu_shared_with_others"] = bool(others)
                msg = self._assess_measurement(metrics, msg, kernel_min_ms,
                                               gpu_util_pct=util)
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
            msg = (f"加速比 {speedup:.2f}x vs {baseline['baseline_type']} "
                   f"(要求 >= {min_required:.2f}x, {bound}-bound)")
            msg = self._assess_measurement(metrics, msg, kernel_min_ms)
            print(f"  [Verifier] L3 {'PASSED' if passed else 'FAILED'}: {msg}")
            return VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                                      passed=passed, message=msg, metrics=metrics)

        # 回退判定: 无参考实现的算子使用绝对 GFLOPS 目标 (legacy)
        achieve = actual_gflops / target_gflops if target_gflops > 0 else 0.0
        metrics.update({"target_gflops": target_gflops, "achieve_ratio": round(achieve, 4)})
        passed = target_gflops <= 0 or actual_gflops >= target_gflops * min_required
        msg = (f"无 baseline 参考, 按绝对目标判定: {actual_gflops:.2f} GFLOPS "
               f"(目标 {target_gflops * min_required:.2f})")
        msg = self._assess_measurement(metrics, msg, kernel_min_ms)
        print(f"  [Verifier] L3 {'PASSED' if passed else 'FAILED'}: {msg}")
        return VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                                  passed=passed, message=msg, metrics=metrics)

    def _assess_measurement(self, metrics: dict, msg: str, candidate_ms,
                            gpu_util_pct=None) -> str:
        """写入测量有效性三态判定, 返回追加了原因说明的 msg

        缺口 1 的失效方式不是「没算」而是「算了没人读」: `benchmark_contaminated`
        早就写进 metrics, 但 conductor / fixer / agents 三个文件里 grep 零命中。
        所以这里做两件事:
          - metrics 里写三个 can_* 布尔, 供 Conductor 决定「这个数字能不能拿去做
            这个决定」—— 暴露用途而不是档位, 免得调用方各自重新解释一遍分档表;
          - 把结论写进 message。message 会进 Fixer 与 SearchAgent 的 prompt, 是
            唯一保证被下游看到的通道。
        """
        assessment = measurement.assess(
            median_time_ms=candidate_ms,
            launch_floor_us=self._detect_theoretical_peak().get("launch_floor_us"),
            gpu_util_pct=gpu_util_pct,
            dispersion=metrics.get("dispersion_iqr_over_median"),
            max_over_median=metrics.get("dispersion_max_over_median"),
        )
        metrics.update(assessment.as_metrics())
        metrics["benchmark_contaminated"] = assessment.contaminated
        if assessment.contaminated:
            metrics["contamination_reason"] = f"测量前 util={gpu_util_pct}%"
        if assessment.reasons:
            msg += (f"  ⚠️ 测量有效性={assessment.validity.value}: "
                    + "; ".join(assessment.reasons))
        return msg

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
        for key in sorted(_PEAK_BY_NAME, key=len, reverse=True):
            if key in name or key in device:
                vals = _PEAK_BY_NAME[key]
                break

        # 2. 设备名未命中时回退 compute_capability (精确 → 前缀)
        if vals is None and cc:
            if cc in _PEAK_BY_CC:
                vals = _PEAK_BY_CC[cc]
            else:
                for key in sorted(_PEAK_BY_CC, key=len, reverse=True):
                    if cc.startswith(key):
                        vals = _PEAK_BY_CC[key]
                        break

        fp32, tf32, fp16, bf16, bw = vals if vals else (0, 0, 0, 0, 0)

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
            "launch_floor_us": launch_floor_us,
            "compute_capability": cc, "device": device, "name": name,
        }
        return self._theoretical_peak

    @staticmethod
    def _compute_arithmetic_intensity(task: TaskSchema, inputs: dict) -> float:
        """计算算子的算数强度 (FLOPs / Bytes transferred)
        返回 FLOP/Byte 或 0 (无法计算时)"""
        kt = (task.source.kernel_type or "").lower()

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
                bytes_transferred = (M * K + K * N + M * N) * a.itemsize
                return flops / bytes_transferred if bytes_transferred > 0 else 0.0

        if "softmax" in kt:
            # Softmax: ~5*N FLOPs per element (exp, sum, div), memory: N * dtype
            if "X" in inputs:
                x = np.asarray(inputs["X"])
                n = int(np.prod(x.shape))
                flops = 5.0 * n
                bytes_transferred = 2.0 * n * x.itemsize  # read + write
                return flops / bytes_transferred if bytes_transferred > 0 else 0.0

        if any(op in kt for op in ("relu", "gelu", "sigmoid", "add", "elementwise")):
            # Element-wise: ~1 FLOP per element, memory: 2-3 * N * dtype
            for key in ("X", "input", "A"):
                if key in inputs:
                    x = np.asarray(inputs[key])
                    n = int(np.prod(x.shape))
                    flops = 1.0 * n
                    bytes_transferred = 2.0 * n * x.itemsize
                    return flops / bytes_transferred if bytes_transferred > 0 else 0.0

        if "layernorm" in kt or "layer_norm" in kt:
            # LayerNorm: ~7*N FLOPs (mean, var, norm, scale, shift), memory: ~2*N
            for key in ("X", "input"):
                if key in inputs:
                    x = np.asarray(inputs[key])
                    n = int(np.prod(x.shape))
                    flops = 7.0 * n
                    bytes_transferred = 2.0 * n * x.itemsize
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
        }

    def _build_reference_fn(self, task: TaskSchema, inputs: dict):
        """
        构建 PyTorch 参考实现 (可调用); torch 不可用时回退 NumPy
        返回 (callable, baseline_type) 或 (None, None)
        """
        kernel_type = task.source.kernel_type.lower()
        precision = task.constraints.precision
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

    def _generate_reference(self, task: TaskSchema, inputs: dict):
        """生成参考输出 (numpy 数组); torch 张量自动转 numpy"""
        fn, _ = self._build_reference_fn(task, inputs)
        if fn is None:
            return None
        out = fn()
        return self._to_numpy(out)

    @staticmethod
    def _to_numpy(out):
        """torch tensor / numpy array 统一转 numpy float32"""
        try:
            import torch
            if isinstance(out, torch.Tensor):
                return out.detach().cpu().numpy().astype(np.float32)
        except ImportError:
            pass
        return np.asarray(out, dtype=np.float32)

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
