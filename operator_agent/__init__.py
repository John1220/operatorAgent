"""算子开发Agent v0.6.4
支持 Triton DSL: 矩阵乘法/Softmax/Attention 等 GPU 算子
模块结构（2026-10-31 分五层，全量归属见 docs/design/repo-slimming-20261031.md §5）:
  core/       词汇与产物  - schemas / contract / run_contract / candidates / kernel_db /
                            artifact_store / run_layout / report / human_ledger
  decide/     判定        - verifier (L1/L2/L3 + Roofline) / measurement / eligibility /
                            correctness* / anticheat / reference* / failure_route / quality_*
  execute/    执行与预算  - remote_executor / exec_guard / backend_* / mock_gpu /
                            task_budget / task_cancellation / llm_*
  generate/   生成与修复  - planner / coder_pipeline / single_candidate_coder / fixer /
                            repair_* / prompt_manager / optimization_planner / triton_*
  app/        入口与编排  - conductor / cli / mcp_server / hitl_tools / autonomous /
                            intent_parser / tool_calls
  ops/ prompts/ skills/   内嵌子包与资源目录（SKILL 规范 + Prompt 模板）
外部用法保持不变: `from operator_agent import Conductor` 一类再导出仍走包根 __init__.py,
`python -m operator_agent` 由包根 __main__.py 委托。"""

__version__ = "0.6.4"

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
