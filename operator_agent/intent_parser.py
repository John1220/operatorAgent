"""
用户接口层 - Intent Parser
将自然语言请求解析为结构化 TaskSchema
优先 LLM 意图理解，失败时回退到关键词匹配
prompt 模板从 prompts/ 目录加载
"""

import re
import json
from typing import Optional
from .schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision, DSLEnum,
)
from .prompt_manager import PromptManager


class IntentParser:
    """
    意图解析器 - 解析自然语言 -> TaskSchema
    优先 LLM，回退到规则匹配
    """

    OPERATOR_KEYWORDS = {
        "matmul": ["matmul", "mat_mul", "矩阵乘法", "矩阵乘", "gemm", "mm"],
        "attention": ["attention", "flash_attention", "flash attention", "注意力"],
        "softmax": ["softmax", "soft max"],
        "layernorm": ["layernorm", "layer_norm", "层归一化"],
        "relu": ["relu", "激活函数"],
        "conv2d": ["conv2d", "卷积", "convolution"],
        "batchnorm": ["batchnorm", "batch_norm", "批归一化"],
        "add": ["add", "elementwise_add", "逐元素加法", "加法"],
    }

    PRECISION_KEYWORDS = {
        "fp32": ["fp32", "float32", "单精度"],
        "fp16": ["fp16", "float16", "半精度"],
        "bf16": ["bf16", "bfloat16"],
        "int8": ["int8", "int8量化"],
    }

    HARDWARE_KEYWORDS = {
        "nvidia_gpu": ["nvidia", "cuda", "gpu", "英伟达"],
        "mock_gpu": ["mock", "mock_gpu", "模拟gpu"],
        "kunlunxin": ["kunlunxin", "kunlun", "昆仑芯", "xpu"],
        "iluvatar": ["iluvatar", "天数", "corex"],
        "cambricon": ["cambricon", "寒武纪", "mlu"],
        "ascend": ["ascend", "昇腾", "npu"],
    }

    DSL_KEYWORDS = {
        "triton": ["triton", "tl.load", "tl.store", "triton.jit", "@triton"],
        "cuda_c": ["cuda c", "cuda_c", "cuda kernel", "cu文件", ".cu"],
        "python": ["python", "numpy", "torch", "pytorch"],
    }

    def __init__(self, llm=None, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm
        self.pm = prompt_manager or PromptManager()

    def parse(self, request: str) -> TaskSchema:
        """解析自然语言请求为 TaskSchema"""
        if self.llm:
            try:
                task = self._llm_parse(request)
                print(f"[IntentParser] LLM 解析完成:")
                print(f"  算子类型: {task.source.kernel_type}")
                print(f"  目标硬件: {task.target.hardware}")
                print(f"  DSL: {task.target.dsl}")
                print(f"  精度: {task.constraints.precision.value}")
                print(f"  复杂度: {task.source.complexity.value}")
                return task
            except Exception as e:
                print(f"[IntentParser] LLM 解析失败 ({e})，回退到规则匹配")

        task = self._rule_parse(request)
        print(f"[IntentParser] 规则解析完成:")
        print(f"  算子类型: {task.source.kernel_type}")
        print(f"  目标硬件: {task.target.hardware}")
        print(f"  DSL: {task.target.dsl}")
        print(f"  精度: {task.constraints.precision.value}")
        print(f"  复杂度: {task.source.complexity.value}")
        return task

    def _llm_parse(self, request: str) -> TaskSchema:
        """调用 LLM 解析意图"""
        prompt = self.pm.render("intent_parse", user_request=request)

        result = self.llm.chat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
            caller_tag="IntentParser",
        )

        dsl = result.get("dsl", "python")
        if dsl not in [e.value for e in DSLEnum]:
            dsl = "python"

        hardware = result.get("hardware", "nvidia_gpu")

        return TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(
                kernel_type=result.get("kernel_type", "unknown"),
                complexity=TaskComplexity(result.get("complexity", "medium")),
            ),
            target=TargetSpec(
                hardware=hardware,
                dsl=dsl,
                compute_units=self._hardware_cu(hardware),
                warp_size=self._hardware_warp(hardware),
            ),
            constraints=ConstraintSpec(
                precision=Precision(result.get("precision", "fp32")),
                min_speedup=result.get("min_speedup", 1.0),
                max_error=1e-5,
            ),
        )

    def _rule_parse(self, request: str) -> TaskSchema:
        """规则匹配解析 (fallback)"""
        text_lower = request.lower()
        kernel_type = self._extract_operator(text_lower)
        hardware = self._extract_hardware(text_lower)
        precision = self._extract_precision(text_lower)
        complexity = self._extract_complexity(text_lower, kernel_type)
        min_speedup = self._extract_speedup(text_lower)
        matrix_dim = self._extract_matrix_dim(text_lower)
        dsl = self._extract_dsl(text_lower, hardware)

        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type=kernel_type, complexity=complexity),
            target=TargetSpec(
                hardware=hardware, dsl=dsl,
                compute_units=self._hardware_cu(hardware),
                warp_size=self._hardware_warp(hardware),
            ),
            constraints=ConstraintSpec(precision=precision, min_speedup=min_speedup, max_error=1e-5),
        )
        if matrix_dim:
            print(f"  矩阵维度: {matrix_dim}")
        return task

    def _extract_dsl(self, text: str, hardware: str) -> str:
        """从请求中提取目标 DSL，默认根据硬件推断"""
        for dsl_name, keywords in self.DSL_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return dsl_name
        # 根据硬件推断默认 DSL
        if hardware in ("nvidia_gpu",):
            return "triton"  # NVIDIA GPU 默认使用 Triton
        return "python"

    def _extract_operator(self, text: str) -> str:
        for op_name, keywords in self.OPERATOR_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return op_name
        return "unknown"

    def _extract_hardware(self, text: str) -> str:
        for hw_name, keywords in self.HARDWARE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return hw_name
        return "nvidia_gpu"

    def _extract_precision(self, text: str) -> Precision:
        for prec_name, keywords in self.PRECISION_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return Precision(prec_name)
        return Precision.FP32

    def _extract_complexity(self, text: str, kernel_type: str) -> TaskComplexity:
        if "融合" in text or "fused" in text:
            return TaskComplexity.HIGH
        if kernel_type in ("attention", "conv2d"):
            return TaskComplexity.HIGH
        if kernel_type in ("matmul", "layernorm", "batchnorm"):
            return TaskComplexity.MEDIUM
        return TaskComplexity.LOW

    def _extract_speedup(self, text: str) -> float:
        for p in [r"加速比\s*(\d+(?:\.\d+)?)\s*[x倍]", r"(\d+(?:\.\d+)?)\s*[x倍]\s*加速"]:
            m = re.search(p, text)
            if m:
                return float(m.group(1))
        return 1.0

    def _extract_matrix_dim(self, text: str) -> Optional[tuple]:
        for p in [r"(\d+)\s*[x*]\s*(\d+)\s*[x*]\s*(\d+)", r"(\d+)\s*[x*]\s*(\d+)"]:
            m = re.search(p, text)
            if m:
                return tuple(int(x) for x in m.groups())
        return None

    @staticmethod
    def _hardware_cu(hw: str) -> int:
        cu_map = {"nvidia_gpu": 128, "mock_gpu": 64, "ascend": 64}
        return cu_map.get(hw, 64)

    @staticmethod
    def _hardware_warp(hw: str) -> int:
        warp_map = {"nvidia_gpu": 32, "mock_gpu": 32, "ascend": 16}
        return warp_map.get(hw, 32)


class TestInputGenerator:
    """根据任务描述生成测试输入 (种子可控，支持多组输入套件)"""

    @staticmethod
    def generate(task: TaskSchema, matrix_dim: tuple = None, seed: int = None) -> dict:
        """
        生成单组测试输入

        Args:
            task: 任务描述
            matrix_dim: 矩阵维度 (可选)
            seed: 随机种子。None = 不可预测 (默认, 防止生成结果过拟合固定输入)；
                  传入整数可复现 (用于回归测试)
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        kernel_type = task.source.kernel_type.lower()
        precision = task.constraints.precision
        dtype_map = {Precision.FP32: np.float32, Precision.FP16: np.float16, Precision.BF16: np.float32, Precision.INT8: np.int8}
        dtype = dtype_map.get(precision, np.float32)

        if "matmul" in kernel_type or "gemm" in kernel_type:
            if matrix_dim and len(matrix_dim) >= 3:
                M, N, K = matrix_dim[0], matrix_dim[1], matrix_dim[2]
            elif matrix_dim and len(matrix_dim) == 2:
                M, K = matrix_dim; N = K
            else:
                M, N, K = 256, 256, 128
            return {"A": (rng.standard_normal((M, K)) * 0.1).astype(dtype),
                    "B": (rng.standard_normal((K, N)) * 0.1).astype(dtype)}

        if "softmax" in kernel_type:
            dim = matrix_dim[0] if matrix_dim else 256
            return {"X": rng.standard_normal((dim, dim)).astype(dtype)}

        if "attention" in kernel_type:
            return {"Q": (rng.standard_normal((2, 64, 8, 64)) * 0.1).astype(dtype),
                    "K": (rng.standard_normal((2, 64, 8, 64)) * 0.1).astype(dtype),
                    "V": (rng.standard_normal((2, 64, 8, 64)) * 0.1).astype(dtype)}

        dim = matrix_dim[0] if matrix_dim else 128
        return {"input": rng.standard_normal((dim, dim)).astype(dtype)}

    @staticmethod
    def generate_suite(
        task: TaskSchema,
        matrix_dim: tuple = None,
        num_sets: int = 5,
        base_seed: int = None,
    ) -> list:
        """
        生成多组不同随机种子的输入套件，用于强化 L2 功能验证
        (对应验证端 Verifier 的 list[dict] 输入支持)

        Args:
            num_sets: 输入组数 (默认 5，对齐 CUDA-Agent 的 5 输入检查)
            base_seed: 起始种子。None = 随机选取；传入整数则套件整体可复现
        """
        import numpy as np
        if base_seed is None:
            base_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        return [
            TestInputGenerator.generate(task, matrix_dim, seed=base_seed + i)
            for i in range(num_sets)
        ]
