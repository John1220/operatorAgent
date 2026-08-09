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
from .mock_gpu import MockGPU
from .remote_executor import RemoteExecutor, RemoteExecutionResult


# 常见 GPU 硬件理论峰值 (FP32/TF32/FP16/BF16 Tensor Core + 显存带宽)
# 格式: (FP32_TFLOPS, TF32_TFLOPS, FP16_TFLOPS, BF16_TFLOPS, memory_bandwidth_GBs)
_HARDWARE_THEORETICAL_PEAK: dict[str, tuple[float, float, float, float, float]] = {
    # compute_capability 前缀匹配
    "8.0":  (19.5,  156.0, 312.0, 312.0, 2039.0),  # A100 (80GB, HBM2e)
    "8.6":  (19.5,  156.0, 312.0, 312.0, 1555.0),  # A100 (40GB), A40
    "8.9":  (67.0,  989.0, 1979.0, 989.0, 3350.0),  # H100 (HBM3, FP8=1979)
    "9.0":  (67.0,  989.0, 1979.0, 989.0, 3350.0),  # H200 (est.)
    "9.0a": (82.6,  1228.0, 2457.0, 1228.0, 4800.0),  # B200
    "7.5":  (16.3,  130.6, 326.0, 326.0, 936.0),   # RTX 3090 / A40 (GA102, Ampere has TF32)
    "7.0":  (15.7,  0.0,   125.0, 125.0, 900.0),   # V100 (Volta, no TF32)
    "8.9-bw":(67.0, 989.0, 1979.0, 989.0, 1800.0),  # H100 (PCIe)
    # 设备名关键字匹配 (fallback 当 compute_capability 不可用时)
    "a100": (19.5,  156.0, 312.0, 312.0, 1555.0),
    "h100": (67.0,  989.0, 1979.0, 989.0, 3350.0),
    "h200": (67.0,  989.0, 1979.0, 989.0, 3350.0),
    "v100": (15.7,  0.0,   125.0, 125.0, 900.0),
    "3090": (16.3,  130.6, 326.0, 326.0, 936.0),
    "4090": (82.6,  82.6,  330.0, 330.0, 1008.0),  # 4090 FP32=82.6 (no TF32 boost for Ada? actually Ada has TF32)
    "a6000":(38.7,  310.0, 77.4,  77.4,  768.0),
    "a5000":(27.8,  222.0, 55.5,  55.5,  768.0),
    "t4":   (8.1,   65.0,  130.0, 130.0, 320.0),
}


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
                    message=f"编译失败: {msg}", error_log=str(e),
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
            return VerificationResult(level=VerificationLevel.L1_COMPILE, passed=False, message=f"编译失败: {e}", error_log=str(e))

    def _verify_function(self, kernel_code: str, test_inputs, expected_output, task: TaskSchema) -> VerificationResult:
        print("  [Verifier] L2: 功能验证...")

        # 反作弊静态审计: 执行前先检查是否存在原生算子回退调用
        #   注意: 昆仑芯/昇腾等自研加速卡白名单放开 torch.matmul 等 (因为它们的 Tensor Core
        #   只能通过 torch.matmul 下放 xblas 激活)
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
        total_compared = 0
        min_coverage = 1.0
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
                total_compared += m.get("compared_elements", 0)
                min_coverage = min(min_coverage, m.get("coverage", 1.0))
                if m.get("reference_type"):
                    reference_types.add(m["reference_type"])
                if m.get("device"):
                    execution_devices.add(m["device"])

        if has_reference:
            msg = f"功能验证通过 ({len(input_sets)} 组输入), 最大误差={max_error_overall:.2e}"
            if min_coverage < 1.0:
                msg += f", 覆盖率={min_coverage * 100:.1f}%"
            metrics = {
                "max_error": max_error_overall,
                "num_input_sets": len(input_sets),
                "compared_elements": total_compared,
                "coverage": min_coverage,
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
                            metrics={k: remote_verify[k] for k in (
                                "max_error", "max_rel_error", "compared_elements", "coverage",
                                "reference_type", "device") if k in remote_verify},
                        )
                    if remote_verify.get("output_shape") and remote_verify.get("expected_shape"):
                        return VerificationResult(
                            level=VerificationLevel.L2_FUNCTION, passed=False,
                            message=(f"输出形状不匹配: {remote_verify['output_shape']} vs "
                                     f"{remote_verify['expected_shape']}"),
                            error_log=remote_verify.get("error", ""),
                        )
                    return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False,
                                              message=f"远程执行失败: {remote_verify.get('error', '')}",
                                              error_log=remote_verify.get("error", ""))
                return VerificationResult(
                    level=VerificationLevel.L2_FUNCTION, passed=True,
                    message=(f"远程全量比对通过 ({remote_verify.get('compared_elements', 0)} 元素), "
                             f"最大误差={remote_verify.get('max_error', 0):.2e}"),
                    metrics={k: remote_verify[k] for k in (
                        "max_error", "max_rel_error", "compared_elements", "coverage",
                        "reference_type", "device") if k in remote_verify},
                )

            # 旧执行器兼容路径。新 SSHExecutor 不会进入这里。
            result = self.backend.execute(kernel_code, inputs)
            if not result.success:
                return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"执行失败: {result.error}", error_log=result.error)
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
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"执行失败: {result.error}", error_log=result.error)
        if expected_output is None:
            expected_output = self._generate_reference(task, inputs)
        if expected_output is None:
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=True, message="无参考输出，仅验证可执行性", metrics={"execution_time_ms": result.execution_time_ms})

        actual = result.output
        expected = np.asarray(expected_output, dtype=np.float32)
        if actual.shape != expected.shape:
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"输出形状不匹配: {actual.shape} vs {expected.shape}")

        size = int(expected.size)
        max_error = float(np.max(np.abs(actual - expected)))
        metrics = {"max_error": max_error, "compared_elements": size, "coverage": 1.0}
        if max_error <= max_allowed:
            return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=True, message=f"全量比对通过 ({size} 元素), 最大误差={max_error:.2e}", metrics=metrics)
        return VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False, message=f"功能验证失败: 误差 {max_error:.2e}", metrics=metrics)

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
        # 从 benchmark 结果传递额外字段 (mode, language, device 等)
        for extra_key in ("mode", "language", "device", "runs", "max_time_ms"):
            if extra_key in bench:
                metrics[extra_key] = bench[extra_key]

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
        # baseline 选择策略 (按优先级):
        #   FP16 matmul: 1. pytorch_eager_fp16 (cuBLAS FP16) — 最公平
        #               2. pytorch_eager_tf32 (cuBLAS + TF32)
        #               3. pytorch_eager (cuBLAS FP32)
        #   其他:       1. pytorch_eager_tf32 / pytorch_eager
        if remote_bench is not None:
            baselines = bench.get("baselines", {})
            is_cuda_matmul = (bench.get("device", "").startswith("cuda") and
                              "matmul" in task.source.kernel_type.lower())
            is_fp16 = (task.constraints.precision == Precision.FP16)
            if is_cuda_matmul and is_fp16:
                baseline_priority = ["pytorch_eager_fp16", "pytorch_eager_tf32", "pytorch_eager", "torch_compile"]
            elif is_cuda_matmul:
                baseline_priority = ["pytorch_eager_tf32", "pytorch_eager", "torch_compile"]
            else:
                baseline_priority = ["pytorch_eager", "torch_compile"]
            baseline_name = None
            for name in baseline_priority:
                if name in baselines and baselines[name]:
                    baseline_name = name
                    break
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
                    "baseline_time_ms": round(baseline_ms, 4),
                    "speedup_vs_baseline": round(speedup, 4),
                    "min_speedup_required": min_required,
                    "measurement_scope": "same_remote_device",
                    "candidate_median_time_ms": candidate_metrics.get("median_time_ms"),
                    "baseline_metrics": baselines,
                })
                passed = speedup >= min_required
                msg = (f"加速比 {speedup:.2f}x vs {display_name} "
                       f"(同机同卡 median, 要求 >= {min_required:.2f}x, {bound}-bound)")
                print(f"  [Verifier] L3 {'PASSED' if passed else 'FAILED'}: {msg}")
                return VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                                          passed=passed, message=msg, metrics=metrics)

            return VerificationResult(level=VerificationLevel.L3_PERFORMANCE, passed=False,
                                      message="远程性能测试未返回可用 baseline")

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
            print(f"  [Verifier] L3 {'PASSED' if passed else 'FAILED'}: {msg}")
            return VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                                      passed=passed, message=msg, metrics=metrics)

        # 回退判定: 无参考实现的算子使用绝对 GFLOPS 目标 (legacy)
        achieve = actual_gflops / target_gflops if target_gflops > 0 else 0.0
        metrics.update({"target_gflops": target_gflops, "achieve_ratio": round(achieve, 4)})
        passed = target_gflops <= 0 or actual_gflops >= target_gflops * min_required
        msg = (f"无 baseline 参考, 按绝对目标判定: {actual_gflops:.2f} GFLOPS "
               f"(目标 {target_gflops * min_required:.2f})")
        print(f"  [Verifier] L3 {'PASSED' if passed else 'FAILED'}: {msg}")
        return VerificationResult(level=VerificationLevel.L3_PERFORMANCE,
                                  passed=passed, message=msg, metrics=metrics)

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
        """探测硬件理论峰值 (GFLOPS, 带宽)，优先从 compute_capability 匹配，回退设备名"""
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

        # 优先按 compute_capability 精确匹配
        if cc:
            if cc in _HARDWARE_THEORETICAL_PEAK:
                fp32, tf32, fp16, bf16, bw = _HARDWARE_THEORETICAL_PEAK[cc]
            else:
                # 前缀匹配: "8.0" 匹配 "8.0", "8.6" 匹配 "8.6"
                matched = None
                for key in sorted(_HARDWARE_THEORETICAL_PEAK.keys(), key=len, reverse=True):
                    if cc.startswith(key) and key.replace(".", "").replace("-", "").isdigit():
                        matched = key
                        break
                if matched:
                    fp32, tf32, fp16, bf16, bw = _HARDWARE_THEORETICAL_PEAK[matched]
                else:
                    fp32, tf32, fp16, bf16, bw = (0, 0, 0, 0, 0)
        else:
            # 回退设备名关键字匹配
            fp32, tf32, fp16, bf16, bw = (0, 0, 0, 0, 0)
            for key, vals in _HARDWARE_THEORETICAL_PEAK.items():
                if key in name or key in device:
                    fp32, tf32, fp16, bf16, bw = vals
                    break

        self._theoretical_peak = {
            "fp32_tflops": fp32, "tf32_tflops": tf32,
            "fp16_tflops": fp16, "bf16_tflops": bf16,
            "memory_bandwidth_gbs": bw,
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
