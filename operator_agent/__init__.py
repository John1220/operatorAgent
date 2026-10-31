"""算子开发Agent v0.6.1
支持 Triton DSL: 矩阵乘法/Softmax/Attention 等 GPU 算子
模块结构:
  schemas.py            - 核心数据结构 (含 DSLEnum, TritonCompileConfig)
  kernel_db.py          - 算子知识库 (RAG) - 含 Triton 种子数据
  mock_gpu.py           - Mock GPU 后端 (支持 Triton 编译/模拟执行)
  remote_executor.py    - 远程代码执行器 (SSH/HTTP/Local) - 支持 Triton
  llm_client.py         - LLM 客户端
  prompt_manager.py     - Prompt 模板管理器
  agents.py             - Agent (Designer + Coder + Search) - 支持 Triton 代码生成
  verifier.py           - 验证层 (L1/L2/L3) + Roofline 分析
  optimization_planner.py - 优化配方选择器 (确定性规则引擎)
  conductor.py          - Conductor 调度层 (集成 Planner)
  intent_parser.py      - 用户接口层 - 支持 DSL 解析
  artifact_store.py     - 产物持久化存储
  skills/               - SKILL 规范 + 优化配方库
  prompts/              - Prompt 模板目录 (含 Triton 模板)
  output/               - 生成的算子产物 (自动创建)"""

__version__ = "0.6.1"

from .core.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskState, TaskComplexity, RouteType,
    DSLEnum, TritonCompileConfig,
    Precision, VerificationLevel, VerificationResult,
    UnifiedSketch, TilingStrategy, MemoryPlan, PipelineStage,
    KernelArtifact,
)
from .core.kernel_db import KernelDatabase, KernelRecord
from .execute.mock_gpu import MockGPU, ExecutionResult
from .execute.remote_executor import RemoteExecutor, RemoteExecutionResult, SSHExecutor
from .execute.llm_client import LLMClient
from .generate.prompt_manager import PromptManager
from .core.artifact_store import ArtifactStore
from .generate.agent_logger import capture_execution, get_logger, get_latest_logs, read_log
from .generate.agents import DesignerAgent, CoderAgent, SearchAgent
from .decide.verifier import Verifier
from .decide.correctness import CorrectnessCase, CorrectnessSuite
from .decide.correctness_boundaries import (BoundaryAxis, TensorAxes, BoundaryPolicy,
                                     generate_correctness_suite, lint_coverage)
from .decide.correctness_profiles import with_default_correctness
from .app.conductor import Conductor, Router
from .app.intent_parser import IntentParser, TestInputGenerator
