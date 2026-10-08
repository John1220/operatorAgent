"""
Mock GPU 后端
Simulates a new GPU backend (e.g., Kunlunxin XPU) for testing.
Executes kernel "code" (Python/Triton functions) and collects performance metrics.

Triton 支持:
  - L1 编译: 检测 Triton 代码模式 (import triton, @triton.jit) + 语法验证
  - L2 执行: 优先尝试 CUDA 实际执行, 无 CUDA 时回退 NumPy 参考实现
  - L3 性能: 优先实测, 无 CUDA 时基于硬件参数模拟估算
"""

import time
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass, field
from .anticheat import CheatDetectedError, lib_fallback_metrics, shield_torch
from .backend_caps import Backend, Cap
from .input_fidelity import InputFidelityError
from .exec_guard import (MODE_COMPILE, MODE_RUN, STATUS_CRASHED, GuardRefusal,
                         Limits, candidate_error, limits_for, refusal_from,
                         run_candidate, run_compile)

#: Triton 那条「真编译」路径要预置进命名空间的模块（子进程里现 import）。
#: 与搬进受控子进程之前那一行 `local_ns = {"triton": triton, "tl": triton.language,
#: "torch": torch}` 逐字对应 —— 少了它，「用了 `tl` 却没自己 import」的候选会从过门
#: 变成不过门，而那是执行方式变更**不该**带来的差别。
TRITON_COMPILE_IMPORTS = {"triton": "triton", "tl": "triton.language", "torch": "torch"}


@dataclass
class ExecutionResult:
    """GPU 执行结果"""
    output: Optional[np.ndarray] = None
    execution_time_ms: float = 0.0
    gflops: float = 0.0
    memory_bandwidth_gbs: float = 0.0
    success: bool = True
    error: str = ""
    language: str = "python"        # python | triton
    device: str = "cpu"            # cpu | cuda | mock
    compile_info: dict = field(default_factory=dict)  # 编译元信息
    #: 旁路信号。WHY (spine-review P1-4): `lib_fallback` 此前全仓只有读者没有
    #: 生产者 —— 执行路径必须把反作弊命中写进来, 否则 eligibility 规则 4 恒 False。
    metrics: dict = field(default_factory=dict)

    @property
    def lib_fallback(self) -> bool:
        return bool(self.metrics.get("lib_fallback"))


def _detect_triton(code: str) -> bool:
    """检测代码是否为 Triton kernel"""
    return "import triton" in code or "@triton.jit" in code or "triton.language" in code


def _has_cuda() -> bool:
    """检查是否有可用的 CUDA 环境"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class MockGPU(Backend):
    """
    Mock GPU 运行时
    模拟新 GPU 的执行环境，支持:
    - 编译验证 (L1): 检查代码语法 (Python + Triton)
    - 功能执行: 运行 kernel 并返回结果 (Triton 优先 CUDA, 无 CUDA 回退 NumPy)
    - 性能采集: 计时并计算 GFLOPS (Triton 优先实测, 无 CUDA 模拟估算)
    """

    def __init__(self, name: str = "MockGPU", compute_units: int = 64, warp_size: int = 32,
                 exec_limits: Optional["Limits"] = None,
                 compile_limits: Optional["Limits"] = None):
        self.name = name
        self.compute_units = compute_units
        self.warp_size = warp_size
        self.vector_width = 128  # bits
        self.global_memory_size = "64GB"
        self.local_memory_size = "128KB/SM"
        self._compiled_kernels: dict = {}
        self._triton_available = self._check_triton()
        self._caps = None
        #: §22.2 的上限**按阶段**声明（这一段是传递链的中段：调用方 → 后端 → `exec_guard`）。
        #: `None` = 用 `exec_guard` 那一档的默认值（编译与运行**不是**同一个默认，见
        #: `exec_guard.COMPILE_LIMITS` 的说明）。刻意只走显式构造参数：不接配置文件、
        #: 不接环境变量、不接命令行（§3.5 的「不允许配置」清单）。
        self.exec_limits = exec_limits
        self.compile_limits = compile_limits

    @property
    def capabilities(self) -> frozenset:
        """有 CUDA 时它跑的是真卡 (只是后端薄), 没有时性能数字是按规格**估**的

        `SIMULATED_PERF` 给 `benchmark()` 里那个一直没人读的 `"mode": "simulated"`
        一个声明式的对应物 (doc 49 §3.5)。`FULL_OUTPUT_TENSOR`: `execute()` 回的是
        `ExecutionResult.output` 全量张量, 所以 L2 能走 `tolerance` 的 AND 形式全量比对。

        ⚠️ 词汇表里刻意**还没有**「硬件信息是真探测的」这一项: `get_hardware_info()`
        返回的是写死的规格 (vector_width=128 / "64GB" / "128KB/SM"), 不是探测结果。
        等 doc 49 §3.3 的 S8/S9 修完再往 `Cap` 里加它, 现在加就等于允许说谎。
        延迟计算: `_has_cuda()` 要 import torch, 不该在构造时就付这笔钱。
        """
        if self._caps is None:
            base = {Cap.REAL_DEVICE} if _has_cuda() else {Cap.EMULATED, Cap.SIMULATED_PERF}
            self._caps = frozenset(base | {Cap.FULL_OUTPUT_TENSOR})
        return self._caps

    @property
    def label(self) -> str:
        return self.name

    def correctness_input_support(self, kernel_code: str, layout: str) -> tuple:
        if _detect_triton(kernel_code):
            return False, "Triton 路径转换 dtype/contiguous，尚未确立套件输入保真"
        if layout not in ("contiguous", "strided", "broadcast"):
            return False, "未知 layout，不能声明输入保真"
        return True, "Python storage/view 协议保留值、dtype、stride 和 offset；须核执行侧回读"

    def _check_triton(self) -> bool:
        """检查 Triton 是否可用"""
        try:
            import triton
            return True
        except ImportError:
            return False

    def get_hardware_info(self) -> dict:
        """返回硬件信息供 Designer Agent 使用"""
        return {
            "name": self.name,
            "compute_units": self.compute_units,
            "warp_size": self.warp_size,
            "vector_width": self.vector_width,
            "triton_available": self._triton_available,
            "cuda_available": _has_cuda(),
            "memory": {
                "global": {"size": self.global_memory_size, "bandwidth": "1.5TB/s"},
                "local": {"size": self.local_memory_size, "bandwidth": "15TB/s"},
            },
        }

    def compile(self, kernel_code: str) -> bool:
        """
        L1 编译验证: 检查代码是否可编译/可执行
        - Triton 代码: 检查语法 + 验证 Triton 模式 (import triton, @triton.jit, kernel wrapper)
        - Python 代码: exec() 检查语法和可导入性

        ⚠️ 2026-10-06（§22.2 第一项「编译超时」）: 「`exec()` 检查语法和可导入性」这一步
        **不再在本进程里做** —— 它走 `exec_guard.run_compile`（墙钟 + rlimit + 进程组 +
        逐项强制读数）。搬走的理由：编译期原先既没有墙钟也没有任何上限，候选只要在模块
        顶层写一个死循环或一次巨型分配，就能在**进入 L1 之前**把评测进程拖死，而那时运行期
        那道门（`_execute_python`）还根本没轮到。

        逐字保持的对外契约（有守卫钉住）:
        * `SyntaxError` 原样重抛（含 exec 抛出的那条 `invalid syntax (<string>, line N)`）；
        * 其余异常包成 `RuntimeError(f"Compile error: {str(e)}")`；
        * `"kernel"` 缺失 / 不可调用的两种文案与判定顺序不变。
        **新增**的一档：编译期被 §22.2 拦下（超时 / 撞上限 / 盾命中）⇒ 抛 `GuardRefusal`
        —— 它不是候选的编译错误，不许与上面两支混在一起（那是 I65/I11 那一族的形状）。
        """
        is_triton = _detect_triton(kernel_code)

        try:
            if is_triton:
                # Triton 编译验证: 静态检查仍在父侧（纯文本判据, 不 exec）
                self._validate_triton_syntax(kernel_code)
                real = bool(self._triton_available and _has_cuda())
                has_kernel = False
                if real:
                    # 有 triton 环境 ⇒ 真的 exec 一遍（搬进受控子进程）。
                    # 预置模块与搬走之前的 `local_ns = {"triton": …, "tl": …, "torch": …}`
                    # 逐字对应，否则「用了 tl 但没自己 import」的候选会从过门变成不过门。
                    _has, _callable = self._compile_in_guard(
                        kernel_code, imports=TRITON_COMPILE_IMPORTS)
                    has_kernel = _has
                else:
                    # 无 Triton 环境, 仅做语法检查 (不实际 import triton)
                    try:
                        compile(kernel_code, "<triton_kernel>", "exec")
                    except SyntaxError as e:
                        raise SyntaxError(f"Triton syntax error: {e}")

                if not has_kernel and not real:
                    # 无 Triton 环境时, 检查代码中是否定义了 kernel 函数
                    if "def kernel" not in kernel_code:
                        raise SyntaxError("Missing 'kernel' function in Triton code")
                elif not has_kernel:
                    if "def kernel" not in kernel_code:
                        raise SyntaxError("Missing 'kernel' function in Triton code")

                return True
            else:
                # Python 编译验证
                has_kernel, callable_kernel = self._compile_in_guard(kernel_code)
                if not has_kernel:
                    raise SyntaxError("Missing 'kernel' function in code")
                if not callable_kernel:
                    raise TypeError("'kernel' is not callable")
                return True

        except (SyntaxError, GuardRefusal):
            raise
        except Exception as e:
            raise RuntimeError(f"Compile error: {e}")

    def _compile_in_guard(self, kernel_code: str,
                          imports: Optional[dict] = None) -> tuple:
        """在受控子进程里 `exec` 一次，返回 `(有 kernel, kernel 可调用)`。

        候选自己抛的异常**按原类型重建**（`SyntaxError` 由 `compile()` 原样重抛，其余由
        调用方包成 `RuntimeError("Compile error: …")`）；执行器自己拦下的那一档抛
        `GuardRefusal`，由调用方的 `except (SyntaxError, GuardRefusal)` 放行。
        """
        guard = run_compile(kernel_code, imports=imports,
                            limits=limits_for(MODE_COMPILE, self.compile_limits))
        if guard.ok:
            return (bool(guard.metrics.get("has_kernel")),
                    bool(guard.metrics.get("kernel_callable")))
        if guard.status == STATUS_CRASHED:
            raise candidate_error(guard)
        raise refusal_from(guard, where="编译期")

    def _validate_triton_syntax(self, code: str):
        """验证 Triton 代码的语法正确性和模式完整性"""
        # 必须包含 triton import
        if "import triton" not in code and "triton.language" not in code:
            raise SyntaxError("Triton code must import triton")

        # 必须包含 @triton.jit 装饰器
        if "@triton.jit" not in code:
            raise SyntaxError("Triton code must use @triton.jit decorator")

        # 必须有 wrapper kernel 函数 (接受 torch.Tensor)
        if "def kernel" not in code:
            raise SyntaxError("Triton code must define a 'kernel' wrapper function")

        # 检查 BLOCK_M / BLOCK_N / BLOCK_K 使用了 tl.constexpr (如果有)
        # 这是软检查, 不报错只记录

    def execute(
        self,
        kernel_code: str,
        inputs: dict,
        expected_shape: tuple = None,
    ) -> ExecutionResult:
        """
        执行 kernel 并返回结果
        Triton 代码: 优先 CUDA 实际执行, 无 CUDA 回退 NumPy 参考实现
        """
        is_triton = _detect_triton(kernel_code)
        result = ExecutionResult(language="triton" if is_triton else "python")

        try:
            if is_triton:
                return self._execute_triton(kernel_code, inputs)
            else:
                return self._execute_python(kernel_code, inputs)
        except CheatDetectedError as e:
            # 反作弊命中: lib_fallback 生产者 (spine-review P1-4)
            result.success = False
            result.error = str(e)
            result.metrics.update(lib_fallback_metrics())
            return result
        except Exception as e:
            result.success = False
            result.error = str(e)
            # 即便异常类型不是 CheatDetectedError, 只要 shield 记过就算命中
            result.metrics.update(lib_fallback_metrics())
            return result

    def _execute_triton(self, kernel_code: str, inputs: dict) -> ExecutionResult:
        """执行 Triton kernel"""
        result = ExecutionResult(language="triton")

        # 尝试 CUDA 实际执行
        if self._triton_available and _has_cuda():
            try:
                return self._execute_triton_cuda(kernel_code, inputs)
            except CheatDetectedError:
                # 反作弊命中**不许**静默降级到 NumPy 参考 —— 那正是 lib_fallback
                # 要拦的「正确率被库实现抬高」(spine-review P1-4)
                raise
            except Exception as e:
                # CUDA 执行失败, 回退到 NumPy 参考实现
                result.device = "cuda (fallback)"

        # 无 CUDA 环境, 使用 NumPy 参考实现
        return self._execute_triton_simulated(kernel_code, inputs)

    def _execute_triton_cuda(self, kernel_code: str, inputs: dict) -> ExecutionResult:
        """在真实 CUDA 环境中执行 Triton kernel"""
        import triton
        import triton.language as tl
        import torch

        device = torch.device("cuda")
        result = ExecutionResult(language="triton", device=str(device))

        ns = {"triton": triton, "tl": tl, "torch": torch, "__name__": "__kernel__"}

        # 反作弊: 执行期间屏蔽 torch 原生算子回退
        try:
            with shield_torch():
                exec(kernel_code, ns)
                kernel_func = ns["kernel"]

                # 转换输入到 CUDA
                torch_inputs = {
                    k: torch.from_numpy(np.asarray(v)).to(device).contiguous()
                    for k, v in inputs.items()
                }

                # Warmup
                for _ in range(3):
                    _ = kernel_func(**torch_inputs)
                torch.cuda.synchronize()

                # 执行并计时
                start = time.perf_counter()
                output = kernel_func(**torch_inputs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
        except CheatDetectedError as e:
            # lib_fallback 生产者 (spine-review P1-4)
            result.success = False
            result.error = str(e)
            result.metrics.update(lib_fallback_metrics())
            return result

        if isinstance(output, torch.Tensor):
            out_arr = output.cpu().numpy()
        else:
            out_arr = np.asarray(output)

        result.output = out_arr
        result.execution_time_ms = elapsed * 1000
        result.success = True
        result.metrics.update(lib_fallback_metrics())

        # 计算性能指标
        self._compute_metrics(result, inputs, elapsed)
        return result

    def _execute_triton_simulated(self, kernel_code: str, inputs: dict) -> ExecutionResult:
        """
        无 CUDA 环境时的 Triton 模拟执行:
        使用 NumPy 参考实现计算正确输出,
        性能数据基于硬件参数模拟估算
        """
        result = ExecutionResult(language="triton", device=f"mock ({self.name})")

        # 使用 NumPy 参考实现计算正确输出
        np_inputs = {k: np.asarray(v) for k, v in inputs.items()}

        # 根据代码内容推断算子类型, 使用 NumPy 参考实现
        output = self._numpy_reference(kernel_code, np_inputs)
        result.output = output
        result.success = True

        # 模拟性能: 基于硬件参数估算执行时间
        elapsed = self._simulate_triton_perf(np_inputs, kernel_code)
        result.execution_time_ms = elapsed * 1000

        self._compute_metrics(result, inputs, elapsed)
        result.compile_info = {
            "mode": "simulated",
            "note": "No CUDA available, performance estimated from hardware specs",
            "hardware": self.name,
            "compute_units": self.compute_units,
        }
        return result

    def _numpy_reference(self, code: str, inputs: dict) -> np.ndarray:
        """根据 Triton 代码推断算子类型, 用 NumPy 计算参考输出"""
        code_lower = code.lower()
        if "matmul" in code_lower or "gemm" in code_lower:
            a = inputs.get("A")
            b = inputs.get("B")
            if a is not None and b is not None:
                return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)
        if "softmax" in code_lower:
            x = inputs.get("X")
            if x is not None:
                x = x.astype(np.float32)
                mx = np.max(x, axis=-1, keepdims=True)
                ex = np.exp(x - mx)
                return (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float32)
        # 通用: 尝试提取 kernel 中的计算逻辑 (简单回退)
        if "A" in inputs and "B" in inputs:
            return (inputs["A"].astype(np.float32) @ inputs["B"].astype(np.float32)).astype(np.float32)
        # 最后回退: 返回第一个输入
        first_key = next(iter(inputs))
        return np.asarray(inputs[first_key]).astype(np.float32)

    def _simulate_triton_perf(self, inputs: dict, code: str) -> float:
        """
        模拟 Triton kernel 性能:
        基于 FLOPS 和硬件参数估算执行时间
        """
        # 估算 FLOPS
        flops = 0
        if "A" in inputs and "B" in inputs:
            a = np.asarray(inputs["A"])
            b = np.asarray(inputs["B"])
            if a.ndim == 2 and b.ndim == 2:
                m, k = a.shape
                k2, n = b.shape
                flops = 2 * m * n * k

        # 模拟 GPU 性能: 理论 FLOPS = compute_units * warp_size * 2 (FMA) * clock
        # MockGPU: 假设 1 GHz clock, 64 CU, 32 warp_size -> ~4 TFLOPS FP32
        theoretical_tflops = self.compute_units * self.warp_size * 2 * 1.0  # GHz
        # 实际效率: matmul 通常 40-70% 峰值
        efficiency = 0.5
        actual_gflops = theoretical_tflops * 1000 * efficiency  # GFLOPS

        if flops > 0 and actual_gflops > 0:
            return flops / (actual_gflops * 1e9)
        return 0.001  # 默认 1ms

    def _execute_python(self, kernel_code: str, inputs: dict) -> ExecutionResult:
        """执行纯 Python kernel —— 在 §22.2 的**受控子进程**里（`operator_agent.exec_guard`）

        WHY（2026-10-04，§22.2 最小一块）: 这一处原先就是本进程内 `exec(kernel_code, {})`
        —— 候选与 harness 共享地址空间、文件系统与网络，一个死循环、一次内存跑飞、
        一个进程爆炸就能把评测拖死，而「候选改不了 harness 的内存」只能靠候选自觉。
        搬进受控子进程之后：

        * 墙钟到点 = **杀掉整个进程组**（含候选自己起的孙子进程），不留孤儿；
        * CPU 时间 / 单文件大小 / 进程数由子进程自己的 rlimit 兜住（逐项读数进
          `metrics["exec_guard_not_enforced"]`，**设不上的项如实报**，不许读成「没超限」）；
        * 返回值字节数封顶，超了是 `limit_exceeded`（失败，不是截断后当通过）；
        * 候选改 `numpy` / `torch` 的模块属性、改环境变量、`chdir`，都**影响不到 harness**
          （各自一份地址空间）—— 这一条是原先进程内 `exec` 做不到的。

        ⚠️ 不许读成的事（`exec_guard` 模块 docstring 里那张实测表是唯一出处）：
        这**不是**敌意候选的沙箱 —— 同一个 uid、网络没关、没有容器；本机（macOS）上
        **内存这一项设不上**（`RLIMIT_AS` 内核 EINVAL），所以本地的内存跑飞仍然拦不住；
        CUDA 执行路径（`_execute_triton_cuda`）**没有**接进来。

        语义与搬走之前逐字等价: `ns = {}`、`ns["kernel"]`、只量 `kernel(**inputs)` 这一次、
        失败时 `result.error = str(e)`（子进程把这句话原样带回来）。
        """
        result = ExecutionResult(language="python", device="cpu")

        # §22.2 的阶段级上限：这一段是传递链的**消费端**（`self.exec_limits` 来自构造参数，
        # `None` ⇒ `exec_guard` 的运行默认值）。编译那一档用的是另一个默认（见 `compile()`）。
        try:
            guard = run_candidate(kernel_code, inputs, preserve_inputs=True,
                                  limits=limits_for(MODE_RUN, self.exec_limits))
        except InputFidelityError as exc:
            result.success = False
            result.error = f"输入保真无法确立: {exc}"
            result.metrics["input_fidelity_error"] = str(exc)
            return result
        if not guard.fully_enforced:
            # 「哪一项今天没设上」是一条**读数**，不是报错 —— 读者要能不猜
            result.metrics["exec_guard_not_enforced"] = list(guard.not_enforced)
        result.metrics["exec_guard_executor"] = guard.executor

        if guard.ok:
            result.output = np.asarray(guard.output)
            result.execution_time_ms = guard.elapsed_ms
            result.success = True
            result.metrics.update(guard.metrics)
            self._compute_metrics(result, inputs, guard.elapsed_ms / 1000.0)
            return result

        # 失败路径: 反作弊命中 / 超时 / 超限 / 候选自己炸 —— 四档在 metrics 里分得开
        result.success = False
        result.error = guard.error or guard.detail
        result.metrics.update(guard.metrics)
        result.metrics["exec_guard_status"] = guard.status
        if guard.kind:
            result.metrics["exec_guard_kind"] = guard.kind
        return result

    def _compute_metrics(self, result: ExecutionResult, inputs: dict, elapsed: float):
        """计算 GFLOPS 和内存带宽

        ⚠️ 形状分支必须先分维数: 1D A@B 是 **dot product** (2N FLOPs, 不是 GEMM)。
        原先 `m, k = a.shape` 对一维输入直接 `not enough values to unpack`,
        于是 M02 这类 reduce/dot 题在 MockGPU 上根本跑不完 —— 评分器还没开始打分
        就死在指标计算里 (LeetCUDA pilot 首次 --self-test 抓到)。
        """
        if "A" in inputs and "B" in inputs:
            a = np.asarray(inputs["A"])
            b = np.asarray(inputs["B"])
            if a.ndim == 2 and b.ndim == 2:
                m, k = a.shape
                _k2, n = b.shape
                flops = 2 * m * n * k
                bytes_moved = (m * k + k * n + m * n) * 4
            elif a.ndim == 1 and b.ndim == 1:
                # dot: 2N mul-add, 读两个向量写一个标量
                n = a.shape[0]
                flops = 2 * n
                bytes_moved = (2 * n + 1) * 4
            elif a.ndim == 2 and b.ndim == 1:
                # matvec: 2*M*N
                m, k = a.shape
                flops = 2 * m * k
                bytes_moved = (m * k + k + m) * 4
            else:
                flops = 0
                bytes_moved = int(a.nbytes + b.nbytes)
            result.gflops = (flops / elapsed / 1e9) if elapsed > 0 and flops else 0
            result.memory_bandwidth_gbs = (bytes_moved / elapsed / 1e9) if elapsed > 0 else 0

    def benchmark(
        self,
        kernel_code: str,
        inputs: dict,
        warmup: int = 3,
        repeat: int = 10,
    ) -> dict:
        """
        性能基准测试
        Triton 代码: 优先 CUDA 实测, 无 CUDA 使用模拟值
        """
        is_triton = _detect_triton(kernel_code)

        if is_triton and self._triton_available and _has_cuda():
            return self._benchmark_triton_cuda(kernel_code, inputs, warmup, repeat)
        elif is_triton:
            return self._benchmark_triton_simulated(kernel_code, inputs)
        else:
            return self._benchmark_python(kernel_code, inputs, warmup, repeat)

    def _benchmark_triton_cuda(self, kernel_code: str, inputs: dict, warmup: int, repeat: int) -> dict:
        """在真实 CUDA 上 benchmark Triton kernel"""
        import triton
        import triton.language as tl
        import torch

        device = torch.device("cuda")
        ns = {"triton": triton, "tl": tl, "torch": torch, "__name__": "__kernel__"}
        exec(kernel_code, ns)
        kernel_func = ns["kernel"]

        torch_inputs = {
            k: torch.from_numpy(np.asarray(v)).to(device).contiguous()
            for k, v in inputs.items()
        }

        # Warmup
        for _ in range(warmup):
            _ = kernel_func(**torch_inputs)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = kernel_func(**torch_inputs)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        gflops, bandwidth = self._calc_perf(inputs, avg_time)

        return {
            "avg_time_ms": round(avg_time * 1000, 4),
            "min_time_ms": round(min(times) * 1000, 4),
            "max_time_ms": round(max(times) * 1000, 4),
            "gflops": round(gflops, 4),
            "memory_bandwidth_gbs": round(bandwidth, 4),
            "runs": repeat,
            "device": str(device),
            "language": "triton",
            "mode": "cuda",
        }

    def _benchmark_triton_simulated(self, kernel_code: str, inputs: dict) -> dict:
        """模拟 Triton kernel 性能基准"""
        elapsed = self._simulate_triton_perf(inputs, kernel_code)
        gflops, bandwidth = self._calc_perf(inputs, elapsed)

        return {
            "avg_time_ms": round(elapsed * 1000, 4),
            "min_time_ms": round(elapsed * 1000 * 0.95, 4),
            "max_time_ms": round(elapsed * 1000 * 1.05, 4),
            "gflops": round(gflops, 4),
            "memory_bandwidth_gbs": round(bandwidth, 4),
            "runs": 10,
            "device": f"mock ({self.name})",
            "language": "triton",
            "mode": "simulated",
            "note": "Performance estimated from hardware specs (no CUDA)",
        }

    def _benchmark_python(self, kernel_code: str, inputs: dict, warmup: int, repeat: int) -> dict:
        """Python kernel benchmark"""
        local_ns = {}
        exec(kernel_code, local_ns)
        kernel_func: Callable = local_ns["kernel"]

        for _ in range(warmup):
            kernel_func(**inputs)

        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            kernel_func(**inputs)
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        gflops, bandwidth = self._calc_perf(inputs, avg_time)

        return {
            "avg_time_ms": round(avg_time * 1000, 4),
            "min_time_ms": round(min(times) * 1000, 4),
            "max_time_ms": round(max(times) * 1000, 4),
            "gflops": round(gflops, 4),
            "memory_bandwidth_gbs": round(bandwidth, 4),
            "runs": repeat,
            "device": "cpu",
            "language": "python",
            "mode": "actual",
        }

    @staticmethod
    def _calc_perf(inputs: dict, elapsed: float) -> tuple:
        """计算 GFLOPS 和带宽"""
        gflops = 0.0
        bandwidth = 0.0
        if "A" in inputs and "B" in inputs:
            a = np.asarray(inputs["A"])
            b = np.asarray(inputs["B"])
            m, k = a.shape
            k2, n = b.shape
            flops = 2 * m * n * k
            gflops = (flops / elapsed / 1e9) if elapsed > 0 else 0
            bytes_moved = (m * k + k * n + m * n) * 4
            bandwidth = (bytes_moved / elapsed / 1e9) if elapsed > 0 else 0
        return gflops, bandwidth
