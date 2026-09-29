"""Conductor 调度层
任务编排核心 - 管理状态机、路由分发、Agent 协作
支持 LLM 驱动 + RemoteExecutor 远程验证 (SSH/HTTP/Local)
所有执行过程自动输出到日志文件

=== 可复现性 Harness (Execution Snapshot) ===
每轮优化循环结束后写 execution_snapshot.json 到当前 artifact 目录。
若 execute(resume_from=artifact_dir) 被调用, 则从快照恢复:
  - 已完成的轮次 (round_num 起始)
  - best_gflops / 优化历史 / 禁用&已应用配方
  - sketch / 当前代码 / pending_fix / fix_attempts
任何可 JSON 化的字段都能恢复; 无法序列化的 (LLM / Backend 对象) 由调用方重新注入。
"""

import json
import os
from pathlib import Path
from typing import Optional, Union
from .schemas import (
    TaskSchema, RouteType, VerificationLevel, VerificationResult,
    UnifiedSketch, KernelArtifact,
)
from .agents import DesignerAgent, CoderAgent, SearchAgent
from .fixer import FixerAgent
from .verifier import Verifier
from .kernel_db import KernelDatabase, FailureDatabase, FailureRecord, failure_signature
from .novelty import signature_of
from .backend_caps import Cap
from .mock_gpu import MockGPU
from .remote_executor import RemoteExecutor
from .llm_client import LLMClient
from .prompt_manager import PromptManager
from .artifact_store import ArtifactStore
from .environment import EnvironmentDrift, compare_fingerprints, fingerprint_of
from .measurement import (ROOFLINE_CONVERGED_PCT, Validity, Verdict,
                          binding_attainment, roofline_converged, verdict)
from .agent_logger import capture_execution
from .optimization_planner import OptimizationPlanner


class Router:
    """
    路由器 - 多特征选择适配路线
    特征: 复杂度 / 融合算子 / 硬件-DSL 能力矩阵 / 知识库历史命中
    """

    # 各硬件原生支持的 DSL 集合 (能力矩阵)
    _DSL_CAPABILITY = {
        "nvidia_gpu": {"triton", "cuda_c", "python"},
        "mock_gpu": {"python", "triton"},
        "ascend": {"python"},
    }

    def route(self, task: TaskSchema, kernel_db=None) -> RouteType:
        from .schemas import TaskComplexity
        complexity = task.source.complexity

        # 融合算子 / 高复杂度 -> Agent 生成
        if task.source.fused_ops or complexity == TaskComplexity.HIGH:
            return RouteType.AGENT_GENERATION

        # ── 昇腾等非 NVIDIA 新硬件：一律走 AGENT_GENERATION ──
        # 原因:
        #   - kernel_db 里目前没有适配这些硬件的 RAG 案例
        #   - Compiler backend 会复用 nvidia 的优化配方 (R-MATMUL-*) 对这些硬件无效
        #   - 必须让 Agent 从头到尾生成"纯 Python/torch.matmul 风格"的适配代码
        if task.target.hardware and any(
                tok in task.target.hardware.lower()
                for tok in ("ascend", "npu")):
            return RouteType.AGENT_GENERATION

        # 目标硬件不支持该 DSL -> Agent 生成探路
        capable = self._DSL_CAPABILITY.get(task.target.hardware, {"python"})
        if task.target.dsl not in capable:
            return RouteType.AGENT_GENERATION

        if complexity == TaskComplexity.LOW:
            # LOW 复杂度也走编译器后端 (需要 RAG 辅助和优化配方)
            if kernel_db is not None:
                hits = kernel_db.search(
                    algorithm=task.source.kernel_type,
                    hardware=task.target.hardware,
                    dsl=task.target.dsl,
                    limit=1,
                )
                if hits:
                    return RouteType.COMPILER_BACKEND
            # 无 RAG 命中则走 Agent 生成 (让 LLM 从头设计)
            return RouteType.AGENT_GENERATION

        # MEDIUM: 知识库有同类成功案例 -> 编译器后端复用; 否则 Agent 生成
        if kernel_db is not None:
            hits = kernel_db.search(
                algorithm=task.source.kernel_type,
                hardware=task.target.hardware,
                dsl=task.target.dsl,
                limit=1,
            )
            return RouteType.COMPILER_BACKEND if hits else RouteType.AGENT_GENERATION
        return RouteType.COMPILER_BACKEND


def _validity_of(metrics: dict) -> Validity:
    """把 verifier 写进 metrics 的档位字符串还原成枚举

    缺键时按 VALID 处理: 那是「没有测量有效性信息」的老路径 (MockGPU / legacy),
    不该因为缺键就把一次可用测量判成不可信。
    """
    try:
        return Validity(metrics["measurement_validity"])
    except (KeyError, ValueError):
        return Validity.VALID


def _round_axes(l3_result, test_inputs) -> dict:
    """本轮 roofline 的轴 + 输入形状身份, 原样搬进血缘。

    为什么要搬: 可达上界是**按 shape / 算术强度索引的一条曲线** (doc 42 实测), 而曲线要
    能被复盘, 就得有 `(shape, AI) -> 本轮达到的速率` 这些点。Verifier 每轮其实都算过
    (算术强度、脊点、两侧达成率、峰值出处与 `peak_trusted`), 只是从没离开内存。

    两条纪律:
    * **不挑键** —— 整块复制 `roofline_analysis`。给落盘键做白名单的那次 (doc 48), 代价是
      那一批真机数据的血缘永久没了, 而且忘加一个键的失效方式是静默的。
    * **没有就不写** —— 这一轮没走到 L3 (L1/L2 就死了) 时返回空 dict, 于是 `result.json`
      里根本没有 `roofline` 这个键。缺键 ≠ 空块: 后者会把「没测过」写成「测了, 值是空」。
    """
    out: dict = {}
    metrics = getattr(l3_result, "metrics", None) or {}
    axes = metrics.get("roofline_analysis")
    if isinstance(axes, dict) and axes:
        out["roofline"] = dict(axes)
    shapes, dtypes = {}, {}
    for name, arr in (test_inputs or {}).items():
        shape = getattr(arr, "shape", None)
        if shape is None:
            continue
        shapes[name] = list(shape)
        dtypes[name] = str(getattr(arr, "dtype", ""))
    if shapes:
        out["input_shapes"] = shapes
        out["input_dtypes"] = dtypes
    return out


class Conductor:
    """
    Conductor - 任务编排器
    管理 Designer -> Coder -> Verifier 完整流水线
    支持 Search Agent 的多轮优化循环
    后端支持 MockGPU / RemoteExecutor (SSH/HTTP/Local)
    Agent 基于 LLM 驱动生成
    Prompt 模板从 prompts/ 目录统一管理
    """

    def __init__(
        self,
        backend: Optional[Union[MockGPU, RemoteExecutor]] = None,
        kernel_db: Optional[KernelDatabase] = None,
        llm: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        max_optimize_rounds: int = 3,
        failure_db: Optional[FailureDatabase] = None,
        use_coder_pipeline: bool = False,
    ):
        # `is None` 而不是 `or`: `FailureDatabase` 定义了 `__len__`, 于是**一条记录都没有的
        # 注入实例是假的**, `failure_db or FailureDatabase()` 会把它丢掉换成一个默认实例 ——
        # 调用方拿不到自己注入的那个库, 而失败记录被写进了默认路径。真机探针 (doc 49 §7.17)
        # 就是这么把 7 条探针记录写进 `output/failure_db.jsonl` 的: 它给的是临时目录。
        self.backend = backend if backend is not None else MockGPU()
        self.kernel_db = kernel_db if kernel_db is not None else KernelDatabase()
        self.failure_db = failure_db if failure_db is not None else FailureDatabase()
        self.llm = llm
        self.pm = prompt_manager or PromptManager()
        self.store = ArtifactStore()
        self.router = Router()
        self.max_optimize_rounds = max_optimize_rounds
        #: 本次运行开始时账本的长度, 每次进 `execute` 重新取。
        #: 写侧用它把口径从「这个进程到现在」换成「这一次」。
        self._cost_watermark = 0

        # 初始化 Agent (注入 LLM + PromptManager)
        self.designer = DesignerAgent(self.kernel_db, llm=llm, prompt_manager=self.pm)
        # 可选启用 CoderPipeline (Plan→并行采样→筛选→修复→回退)
        coder_pipeline = None
        if use_coder_pipeline and llm is not None:
            from .coder_pipeline import CoderPipeline
            coder_pipeline = CoderPipeline(
                llm=llm, kernel_db=self.kernel_db, prompt_manager=self.pm,
            )
        self.coder = CoderAgent(self.kernel_db, llm=llm, prompt_manager=self.pm,
                                pipeline=coder_pipeline)
        self.verifier = Verifier(self.backend)
        self.search = SearchAgent(llm=llm, prompt_manager=self.pm)
        self.fixer = FixerAgent(llm=llm, prompt_manager=self.pm, failure_db=self.failure_db)
        # 优化配方选择器 (确定性规则，不依赖 LLM)
        self.planner = OptimizationPlanner()
        # 本次运行的环境指纹与漂移比较结果 (由 `_execute_inner` 填)。初值给空指纹:
        # `_flush_snapshot` 也可以在 `execute()` 之外被调到 (单测直接调它), 不该 AttributeError。
        self._environment: dict = {"fingerprint": {}, "drift": None}

    #: 循环出口的词汇表。**每个词都要有写者** —— 多一个没写者的词就是「算了没人读」的副本,
    #: `tests/test_stop_reason.py` 按差集钉这条。
    STOP_REASONS = ("passed", "stagnant", "converged", "fix_error", "l1_l2_failed",
                    "search_error", "rounds_exhausted")

    def _record_stop(self, artifact, round_num: int, reason: str, detail: str = "") -> None:
        """把「循环为什么停」记成数据, 而不是只留一行 print。

        形状借自 DSH 的 `agent/turn-stopping` (doc 53 §7.1): 停止是一个有名字的边界,
        不是散在 `break` 里的副作用。读者是量具与事后复盘 —— rounds-to-pass 归因需要
        区分「没练够」(`rounds_exhausted`) 与「练到顶了」(`converged`), 今天在产物里
        这两者与「修坏了」长得完全一样 (`all_passed=False` + 一个轮次数)。
        """
        if reason not in self.STOP_REASONS:
            raise ValueError(f"出口词汇外的 reason: {reason!r} (词汇表 {self.STOP_REASONS})")
        if artifact.metadata is None:
            artifact.metadata = {}
        artifact.metadata["stop"] = {"reason": reason, "round": int(round_num),
                                     "detail": detail}

    def execute(
        self,
        task: TaskSchema,
        test_inputs: dict,
        expected_output=None,
        target_gflops: float = 10.0,
        resume_from: Optional[str] = None,
    ) -> KernelArtifact:
        """
        执行完整任务流水线:
        1. 路由决策
        2. Designer -> 生成 Sketch
        3. Coder -> 生成代码
        4. Verifier -> L1/L2/L3 验证
        5. 若 L3 未通过 -> Search 优化 -> 回到 step 3

        所有执行过程自动写入日志文件

        Args:
            resume_from: 断点续跑: artifact 目录 (含 execution_snapshot.json) 或
                指向该目录的 artifact_id。若提供则恢复已完成轮次的 sketch/code/
                配方集合/best_gflops/优化历史, 从下一轮继续跑, 不用从头生成。
        """
        operator_name = task.source.kernel_type or "operator"
        hardware_name = task.target.hardware or "unknown"
        # 账本水位线: `CostTracker` 挂在 client 上**累加**, 而 `reset()` 全仓没人调
        # (`harness_pass_rate.run_one` 的 docstring 记过一次真事故: 六次运行的 token 被当成
        # 六份独立样本求和, 总量高估 3.25 倍)。一个 client 跑多个任务时, 后一个任务的
        # `summary()` 含着前一个任务的量 —— 落盘的必须是**本次**那份, 所以在这里记下水位线。
        self._cost_watermark = (
            self.llm.cost_tracker.watermark()
            if self.llm is not None and hasattr(self.llm, "cost_tracker") else 0)
        with capture_execution(
            task_id=task.task_id,
            operator=operator_name,
            hardware=hardware_name,
        ) as self._log_path:
            return self._execute_inner(
                task, test_inputs, expected_output, target_gflops,
                resume_from=resume_from,
            )

    def _execute_inner(
        self,
        task: TaskSchema,
        test_inputs: dict,
        expected_output=None,
        target_gflops: float = 10.0,
        resume_from: Optional[str] = None,
    ) -> KernelArtifact:
        """实际执行逻辑 (被 execute 的日志上下文包裹)"""
        backend_name = self.backend.label
        llm_info = f" (LLM: {self.llm})" if self.llm else ""

        print(f"\n{'='*60}")
        print(f"Conductor: 开始执行任务 {task.task_id}")
        print(f"  算子类型: {task.source.kernel_type}")
        print(f"  目标硬件: {task.target.hardware}")
        print(f"  DSL: {task.target.dsl}")
        print(f"  精度: {task.constraints.precision.value}")
        print(f"  验证后端: {backend_name}")
        print(f"  生成模式: {llm_info}")
        if resume_from:
            print(f"  续跑模式: resume_from={resume_from}")
        print(f"{'='*60}")

        # Step 0: TaskSchema 入口契约校验 (Hermes Function-Call Schema 风格)
        # 拦截非法 kernel_type / 负 tile / 错误 DSL 枚举等, 避免下游 Agent 因输入异常跑偏
        task_errors = task.validate()
        hard_task_errors = [e for e in task_errors if not e.startswith("[WARNING]")]
        for w in [e for e in task_errors if e.startswith("[WARNING]")]:
            print(f"[Conductor] {w}")
        if hard_task_errors:
            print(f"\n{'!'*60}")
            print("TaskSchema 入口校验未通过, 终止任务:")
            for e in hard_task_errors:
                print(f"  - {e}")
            print(f"{'!'*60}")
            from .schemas import VerificationResult, UnifiedSketch
            empty_sketch = UnifiedSketch(algorithm=task.source.kernel_type, version=0,
                                         target_dsl=task.target.dsl)
            artifact = KernelArtifact(
                kernel_id=task.task_id, code="", sketch=empty_sketch,
                language=task.target.dsl,
                verification_results=[VerificationResult(
                    level=VerificationLevel.L1_COMPILE, passed=False,
                    message="TaskSchema 入口校验失败",
                    error_log="\n".join(hard_task_errors),
                )],
            )
            return artifact

        # Step 1: 路由决策 (多特征: 复杂度/融合算子/DSL 能力/历史命中)
        route = self.router.route(task, self.kernel_db)
        print(f"\n[Router] 路由决策: {route.value}")

        # Step 1.5: 远程环境前置自检 (缺依赖自动配置/列出清单)
        # 能力判断而不是 `hasattr`: 可选方法在 Backend 基类上恒存在 (一调就抛),
        # 于是 hasattr 恒为真 —— 那正是 doc 49 §3.2 那一类静默 no-op 的温床。
        if Cap.ENV_PROVISION in self.backend.capabilities:
            print(f"\n[Conductor] 远程环境前置自检...")
            passed, manifest = self.backend.ensure_environment()
            if not passed:
                print(f"\n{'!'*60}")
                print(f"环境自检未通过, 无法继续执行任务")
                print(manifest)
                print(f"{'!'*60}")
                from .schemas import VerificationResult, UnifiedSketch
                empty_sketch = UnifiedSketch(algorithm=task.source.kernel_type, version=0,
                                             target_dsl=task.target.dsl)
                artifact = KernelArtifact(
                    kernel_id=task.task_id, code="", sketch=empty_sketch,
                    language=task.target.dsl,
                    verification_results=[VerificationResult(
                        level=VerificationLevel.L1_COMPILE, passed=False,
                        message=f"环境自检未通过",
                        error_log=manifest,
                    )],
                )
                return artifact

        # 获取硬件信息 + 环境指纹。
        #
        # 指纹以前只在这里被 `print` 一次就丢掉 (doc 49 §3.5 第 1 处: 全仓 3 次命中
        # 全在这 6 行里), 旁边那行注释却写着「用于环境漂移自检」—— 描述的是一个不存在
        # 的机制。现在它有三个读者: 落进 result.json / execution_snapshot.json (下一次
        # 运行据此比较)、打进日志、以及作为 L3 测量有效性的一个输入 (漂移 ⇒ 降到
        # DEGRADED, 不禁用配方也不计停滞, 但**不**据此判核失败)。
        hw_info = self.backend.get_hardware_info()
        fingerprint = fingerprint_of(hw_info, self.backend.label)
        # 挂在实例上而不是加到 `_flush_snapshot` 的 12 个形参里: 它在一次运行内是常量,
        # 而 `_flush_snapshot` 有 7 个调用点 (与 `capture_execution` 写 `self._log_path`
        # 同一种「本次运行的上下文」)。`execute()` 不是可重入的。
        self._environment = {"fingerprint": fingerprint, "drift": None}
        if fingerprint:
            print(f"  环境指纹: {fingerprint}")

        # ── 可复现性 Harness: 初始化优化循环的可变状态 ──
        # Step 2: Designer 生成 Sketch (非续跑时才调用 LLM)
        artifact = KernelArtifact(
            task_id=task.task_id,
            language=task.target.dsl,  # python | triton | ...
        )
        all_verification_results = []
        current_sketch = None
        current_code = ""
        pending_fix: Optional[str] = None  # Fixer 修复后的代码，下一轮优先于重新生成
        fix_attempts: list[tuple] = []     # (失败结果, 修复说明, 那条失败的 Reflexion 记录或 None)
        # Reflexion 记录创建那一刻**还不知道**这次修复成没成 (下一轮验证才有答案), 所以先按
        # fixed=False 落盘 (崩溃也不丢教训), 任务结束时若产物通过再补写一次。这里存的是
        # 「欠着一次补写」的那几条。
        reflexion_pending: list = []
        best_gflops: float = 0.0           # 历史最优 GFLOPS
        rounds_without_improvement: int = 0  # 连续无改善轮数
        max_stagnant_rounds: int = 2       # 连续无改善上限，超过则提前终止
        optimization_history: list[dict] = []  # 优化历史记录，每轮追加
        applied_recipes: set[str] = set()      # 已成功应用的配方 ID
        disabled_recipes: set[str] = set()     # 被禁用的配方 ID (应用后性能下降)
        current_recipes_text: str = ""         # 当前轮次注入 Coder 的配方文本
        plan = None                            # 当前轮次的优化计划
        start_round: int = 1                   # 从第 1 轮开始 (续跑时会调整)
        snapshot_target_dir: Optional[str] = None  # 快照写入目录 (由 ArtifactStore 确定, 或 resume_from)

        # ── 可复现性 Harness: 若提供 resume_from 则加载快照恢复状态 ──
        snap_path, snap_scan = self._resolve_snapshot_path(resume_from)
        snap_fingerprint: Optional[dict] = None
        # 续跑的结局要是一条**能落盘的账**, 不是一行 print。「请求了续跑但没续上」与
        # 「压根没请求」在 stdout 消失之后长得一样, 而代价差着好几轮 LLM 与 GPU。
        # (2026-09-29 实测: 一份写到一半崩的快照让一次续跑从 1 次 LLM 调用变成 4 次。)
        resume_status = {"requested": bool(resume_from), "status": "not_requested",
                         "root": snap_scan["root"], "candidates": snap_scan["candidates"],
                         "unreadable": snap_scan["unreadable"]}
        if resume_from:
            if not snap_path:
                # 「找不到」与「找到但读不动」分开写: 前者是缺证据, 后者是有一条坏记录在那儿,
                # 处置方式完全不同 (前者可以问用户要目录, 后者该去查是谁把文件写坏的)。
                resume_status["status"] = "no_snapshot"
                print(f"\n[Conductor] ⚠️ 请求续跑 (resume_from={resume_from}), 但在 {snap_scan['root']} "
                      f"下没有 task_id 匹配的完整快照 (候选 {snap_scan['candidates']} 个, "
                      f"其中 {snap_scan['unreadable']} 个读不动) —— 本次从头开始")
            else:
                resume_status["path"] = os.path.basename(snap_path)
                try:
                    snap = self._load_snapshot(snap_path)
                except Exception as e:
                    # 只存异常类名与文件大小: `str(e)` 可能带路径, 而 result.json 会被搬走
                    resume_status.update(status="unreadable_snapshot",
                                         reason=type(e).__name__,
                                         size_bytes=os.path.getsize(snap_path)
                                         if os.path.exists(snap_path) else None)
                    print(f"\n[Conductor] ⚠️ 快照加载失败，改为从头开始: "
                          f"{type(e).__name__}: {e}")
                    snap = None
                if snap:
                    # 基本一致性校验 (task id 必须匹配)
                    if snap.get("task_id") and snap["task_id"] != task.task_id:
                        resume_status["status"] = "task_mismatch"
                        resume_status["snapshot_task_id"] = snap["task_id"]
                        print(f"[Conductor] ⚠️ 快照 task_id={snap['task_id']} 与当前 task.task_id={task.task_id} 不一致，"
                              f"视为快照无效，从头开始。")
                    else:
                        (start_round, current_sketch, current_code,
                         pending_fix, artifact.iteration_history,
                         best_gflops, rounds_without_improvement,
                         optimization_history, applied_recipes, disabled_recipes,
                         all_verification_results) = self._restore_state_from_snapshot(snap, task)
                        # 快照目录就是续跑写入目录
                        snapshot_target_dir = os.path.dirname(snap_path)
                        snap_fingerprint = snap.get("environment_fingerprint") or None
                        resume_status.update(status="resumed",
                                             round_restored=start_round - 1,
                                             dir_name=os.path.basename(
                                                 os.path.dirname(snap_path)))
                        print(f"\n[Conductor] 🔄 断点续跑: 已恢复到完成 {start_round-1} 轮, "
                              f"best_gflops={best_gflops:.2f}, 当前从轮次 {start_round} 开始")
                        if pending_fix:
                            print(f"  有 pending_fix ({len(pending_fix)} chars), 下一轮直接用修复代码")
                        if current_sketch:
                            artifact.sketch = current_sketch
        artifact.resume_status = resume_status

        # ── 环境漂移自检 (doc 49 §3.5 第 1 处) ────────────────────────────────
        # 「上一次」优先取续跑快照: 那条路径上 `best_gflops` / `iteration_history` /
        # 已禁用配方**全都是从上一次继承来的**, 环境一变, 继承下来的那个数就不再与本
        # 轮可比 —— 这是漂移唯一会直接污染判决的场合, 所以它排在产物库前面。
        # 没有续跑时才回落到产物库里最近一次同 (算子, 硬件) 的运行。
        #
        # 刻意**不**每轮重探一次 `get_hardware_info()`: 那会给每一轮加一次远端往返,
        # 而「这一轮跑在哪张卡上」已经随 L3 载荷回来了 (`selected_gpu` / `device` /
        # `sm_clock_*`)。轮内漂移用那些键复盘, 跨运行漂移用指纹。
        if snap_fingerprint:
            previous_fp, previous_from = snap_fingerprint, "断点续跑的快照"
        else:
            # 必须在本次 `store.save()` 之前问: `previous_environment` 不知道「本次」
            # 是谁, 落盘之后再问就会把自己当成上一次。
            prev = self.store.previous_environment(
                task.source.kernel_type or "unknown",
                task.target.hardware or "unknown",
            )
            previous_fp = (prev or {}).get("fingerprint")
            previous_from = (f"上一次运行的产物 ({os.path.basename(prev['artifact_dir'])})"
                             if prev else "")
        drift = compare_fingerprints(fingerprint, previous_fp, previous_from)
        for line in drift.report_lines():
            print(line)
        artifact.environment_fingerprint = fingerprint
        artifact.environment_drift = drift.as_dict()
        self._environment["drift"] = artifact.environment_drift
        # 传给 Verifier 的只有「变了哪些项」; 无从比较时传 None (与「比过了没变」分开,
        # 见 `measurement.assess` 里那条「没提供 ≠ 检查过」的处理)。
        verify_drift = list(drift.changed) if drift.drifted else None

        if current_sketch is None:
            # 非续跑: 正常跑 Designer
            print(f"\n[Designer] 生成设计草图...")
            current_sketch = self.designer.run(task, hw_info)
            artifact.sketch = current_sketch
        else:
            print(f"\n[Designer] 已从快照恢复 sketch (v{current_sketch.version}), 跳过 Designer 调用")

        # 没指定续跑目录且还没有 snapshot_target_dir: 先用 store 拼一个前缀, 每轮 save 时写到 store.save 前的 "预创建" 目录
        # 为了避免目录名和 ArtifactStore.save 出的 timestamp_id 错开, 我们用 store.save 的目录作为权威写入源,
        # 在每次 store.save 之前写 snapshot 到同一个 artifact_dir (通过 store.save 返回的路径更新).

        # 本次运行内累计的「测量无效」轮数。不进快照: 它描述的是本次会话的测量
        # 环境, 续跑时机器负载早变了, 恢复旧值没有意义。
        invalid_measurement_rounds = 0
        # 出处不对而拒绝通过的那些轮 (doc 49 §7.9)。与上面那个计数器**分开**数是因为
        # 两者的含义相反: `invalid` 是「测了但撑不住结论」, `refused` 是「压根没测过
        # 一个可判定的东西」—— 后者的轮次烧了 LLM 与远程编译, 却没有产出任何关于这台
        # 机器的证据, 混在一个数里就看不出搜索其实在空转。
        refused_rounds = 0

        # 本次运行内每轮候选的结构签名 (doc 47)。同上, 不进快照 —— 它只在「这一次搜索
        # 有没有原地打转」这个尺度上有意义。
        round_signatures: list = []
        duplicate_rounds = 0

        for round_num in range(start_round, self.max_optimize_rounds + 1):
            print(f"\n{'─'*40}")
            print(f"轮次 {round_num}/{self.max_optimize_rounds}")
            print(f"{'─'*40}")

            # Step 3: 获取代码
            # - 首轮 / Search 优化后: Coder 根据 Sketch + OptimizationPlanner 配方重新生成
            # - Fixer 修复后: 直接使用修复版代码重新验证
            if pending_fix is not None:
                current_code = pending_fix
                pending_fix = None
                action = "fix"
                print(f"\n[Conductor] 使用 Fixer 修复后的代码重新验证...")
            else:
                action = "generate" if round_num == 1 else "optimize"

                # ---- OptimizationPlanner: 根据 Roofline 选择本轮要应用的配方 ----
                prev_l3_metrics = {}
                if all_verification_results:
                    prev_l3 = next((r for r in all_verification_results
                                    if r.level == VerificationLevel.L3_PERFORMANCE), None)
                    if prev_l3:
                        prev_l3_metrics = prev_l3.metrics

                plan = self.planner.plan(
                    kernel_type=task.source.kernel_type,
                    dsl=current_sketch.target_dsl if current_sketch else task.target.dsl,
                    roofline=prev_l3_metrics if round_num > 1 else None,
                    hardware=hw_info,
                    precision=task.constraints.precision.value,
                    round_num=round_num,
                    disabled_recipes=disabled_recipes,
                    active_recipes=applied_recipes,
                )
                current_recipes_text = plan.to_prompt()
                if plan.selected_recipes:
                    recipe_names = ", ".join(
                        f"{r.id}({r.name})" for r in plan.selected_recipes
                    )
                    print(f"\n[Planner] 选中 {len(plan.selected_recipes)} 个优化配方: {recipe_names}")
                    print(f"  {plan.summary()}")
                else:
                    print(f"\n[Planner] 本轮无新增优化配方")

                print(f"\n[Coder] 根据 Sketch + 优化配方生成代码 (v{current_sketch.version})...")
                current_code = self.coder.run(
                    current_sketch, task,
                    optimization_recipes=current_recipes_text,
                )

            # ---- 候选新颖度观测 (doc 47) ----
            # 只做两件事: 把结构签名写进 iteration_history (它会被序列化进 result.json,
            # 于是「模型重新生成时有多少轮是结构重复」第一次有一个能复盘的数), 以及在
            # 当场打印出来。**不跳过验证** —— 假阳的代价是丢掉一个真的更好的核, 在没有
            # 基率之前拿它省 GPU 时间就是拿判决换一个还没量的东西。
            sig = signature_of(current_code)
            dup_of = next((r for (r, s) in round_signatures if s and s == sig), None)
            if sig:
                round_signatures.append((round_num, sig))
            if dup_of is not None and action == "optimize":
                duplicate_rounds += 1
                print(f"\n[Novelty] 第 {round_num} 轮与第 {dup_of} 轮结构相同 (签名 {sig}) "
                      f"—— 照常验证, 本轮只在量基率")

            # Step 4: Verifier 验证
            print(f"\n[Verifier] 执行三级验证...")
            verify_results = self.verifier.verify(
                kernel_code=current_code,
                task=task,
                test_inputs=test_inputs,
                expected_output=expected_output,
                target_gflops=target_gflops,
                environment_drift=verify_drift,
            )
            all_verification_results = verify_results

            all_passed = all(r.passed for r in verify_results)
            l3_result = next(
                (r for r in verify_results if r.level == VerificationLevel.L3_PERFORMANCE), None,
            )

            # 血缘记录: 每轮动作与结果
            artifact.iteration_history.append({
                "round": round_num,
                "action": action,
                "all_passed": all_passed,
                # 分级结果必须进血缘: `all_verification_results` 每轮被覆盖
                # (见上面 `all_verification_results = verify_results`), 所以出了循环
                # 就只剩最后一轮的分级 —— 「第 2 轮卡在 L1」与「第 2 轮卡在 L3」在产物里
                # 无法区分, 而 rounds-to-pass 这类统计要的正是它。与 doc 33 抓到的
                # `runs`/`max_time_ms` 从未进过 metrics 是同一类搬运缺陷。
                "levels_passed": {r.level.value: r.passed for r in verify_results},
                "gflops": l3_result.metrics.get("gflops") if l3_result else None,
                # 加速比同理: `benchmark_data` 只有最后一轮的, 而「逐轮到底有没有在变快」
                # 这个问句需要每一轮的读数
                "speedup": (l3_result.metrics.get("speedup_vs_baseline")
                            if l3_result else None),
                # 档位进血缘是为了让「有多少轮的读数有资格驱动闭环」可以离线统计,
                # 而不必重跑一遍才知道那次循环是不是全程都在盲跑
                "measurement_validity": (l3_result.metrics.get("measurement_validity")
                                         if l3_result else None),
                "recipes_applied": list(applied_recipes),
                # 重复率的原料: 签名进 result.json 之后, 「模型重新生成时有多少轮是同一份
                # 结构」这个问题第一次可以被离线复盘, 而不是只能靠当场打印的那一行
                "candidate_signature": sig,
                "duplicate_of": dup_of,
                # 分母的原料 (缺口 3 的「先补分母」那半, doc 49 §八 第 14 条): Verifier 每轮**已经算出** roofline 的全部轴
                # (算术强度、脊点、两侧达成率、峰值出处与可信位), 但它们只活到本次判决结束,
                # 从没进过血缘 —— 于是「可达上界是按 shape/AI 索引的曲线」这件已经量出来的
                # 事实, 任何离线复盘都用不上。整块搬, **不挑键**: doc 48 那次给落盘键做
                # 白名单, 代价是那一批真机数据的血缘永久没了。缺的那块不填默认值。
                **_round_axes(l3_result, test_inputs),
            })

            def _snap() -> str:
                """把当前循环状态写进 execution_snapshot.json

                循环内的落盘点都走这里 (通过 / 停滞 / 已到 roofline / L1L2 耗尽轮次 /
                走到轮次末尾)。少数例外是参数不同的那几处 —— 例如 Fixer 抛异常时
                `pending_fix` 要写成 None, 它们仍直接调 `_flush_snapshot`。

                ⚠️ 变异测试的一个空跑: 终止出口处这次落盘写在 break 之前、save 之前,
                而 `execute()` 末尾一定会再补写一次进权威的 `saved_dir`。所以删掉出口
                那次不会有任何断言变红 —— 它只在「进程在 save 之前就没」时才有读者。
                保留是为了与其余出口动作一致。
                """
                return self._flush_snapshot(
                    task=task,
                    artifact_dir=snapshot_target_dir,
                    round_done=round_num,
                    current_sketch=current_sketch,
                    current_code=current_code,
                    pending_fix=pending_fix,
                    iteration_history=artifact.iteration_history,
                    best_gflops=best_gflops,
                    rounds_without_improvement=rounds_without_improvement,
                    optimization_history=optimization_history,
                    applied_recipes=applied_recipes,
                    disabled_recipes=disabled_recipes,
                    all_verification_results=all_verification_results,
                )

            if all_passed:
                # 标记本轮配方为成功应用
                if plan and plan.selected_recipes:
                    for r in plan.selected_recipes:
                        applied_recipes.add(r.id)
                print(f"\n[Conductor] 所有验证通过!")
                # ── 可复现性 Harness: 通过前写快照, 方便后续 resume 时直接复用最终轮 ──
                snapshot_target_dir = _snap()
                self._record_stop(artifact, round_num, "passed", "所有验证通过 (L1/L2/L3)")

                break

            # 缺口 1: 「数字不好看」不等于「性能未改善」—— 测量本身可能不可信。
            # 污染与 launch 地板支配压低的是数字, 不是核的质量; 拿它禁用配方会
            # **永久**拉黑实际有效的配方, 拿它累计停滞会在 max_stagnant_rounds=2 下
            # 把一次正在起效的优化提前终止。L3 通过时同样要看: 那一路会把本轮配方
            # 记进 applied_recipes, 依据不可信的测量给配方记功是同一个错误的镜像。
            measurement_ok = True
            if l3_result is not None:
                measurement_ok = self._measurement_drives_loop(l3_result, round_num)
                if not measurement_ok:
                    invalid_measurement_rounds += 1
                # 读的是 metrics 里那个布尔而不是重算一遍判据: 键没了就说明提供方
                # 没做过出处检查, 此时如实计 0, 不假装数过了。
                if (l3_result.metrics or {}).get("measurement_l3_pass_refused"):
                    refused_rounds += 1

            # L3 性能改善追踪: 连续无改善时提前终止，避免浪费 LLM token
            if l3_result and l3_result.passed is False and measurement_ok:
                current_gflops = l3_result.metrics.get("gflops", 0)
                # 当 gflops 为 0 (指标计算不可用) 时，用速度比判断
                if current_gflops > 0 and best_gflops <= 0:
                    # 首个有效观测只能用来建立基线: 没有"之前"可比, 因此既不算改善
                    # 也不算停滞, 更不能据此禁用本轮配方 (首轮 plan 是在 roofline=None
                    # 下按算子类型选的, 无从判断其好坏)。
                    # 缺少这一支时 best_gflops 的初值 0.0 会让首轮掉进下面的"指标不
                    # 完整"分支: 把可用的 gflops 报成缺失, 无条件禁用配方, 并在
                    # max_stagnant_rounds=2 下把一次正在起效的优化提前终止。
                    best_gflops = current_gflops
                    print(f"\n[Conductor] 建立性能基线: {current_gflops:.2f} GFLOPS "
                          f"(speedup={l3_result.metrics.get('speedup_vs_baseline', 0):.4f})")
                elif current_gflops > 0 and best_gflops > 0:
                    # D4 四态判决。原来这里是无带宽的 `current > best`: 于是「完全
                    # 打平」和「差 0.0001%」都会去禁用配方 —— 而禁用是**永久**决策,
                    # 需要的证据比「这次没测出更好」强。现在只有 SLOW 才禁用。
                    v, why = verdict(current_gflops, best_gflops,
                                     _validity_of(l3_result.metrics or {}))
                    if v is Verdict.FAST:
                        best_gflops = current_gflops
                        rounds_without_improvement = 0
                        if plan and plan.selected_recipes:
                            for r in plan.selected_recipes:
                                applied_recipes.add(r.id)
                        print(f"\n[Conductor] 性能改善 ({current_gflops:.2f} GFLOPS, {why})")
                    else:
                        if v is Verdict.SLOW:
                            # 只有 SLOW 才禁用配方 —— SAME 与 AMBIGUOUS 都是「没有证据」,
                            # 而禁用是永久决策。进到这里时 measurement_ok 已经成立
                            # (它要求 can_count_stagnation, 也就是 VALID), 所以不必再
                            # 检查 can_disable_recipe: 那会是永不生效的条件。
                            new_recipes = [r for r in plan.selected_recipes
                                           if r.id not in applied_recipes] if plan else []
                            for r in new_recipes:
                                disabled_recipes.add(r.id)
                            if new_recipes:
                                disabled_names = ", ".join(r.id for r in new_recipes)
                                print(f"  [Planner] 性能退步，禁用配方: {disabled_names}")
                            print(f"\n[Conductor] 性能退步 ({current_gflops:.2f} < {best_gflops:.2f} "
                                  f"GFLOPS, {why}), 连续 {rounds_without_improvement + 1}/{max_stagnant_rounds} 轮")
                        else:
                            print(f"\n[Conductor] 性能未改善 ({current_gflops:.2f} vs {best_gflops:.2f} "
                                  f"GFLOPS, {why}), 连续 {rounds_without_improvement + 1}/{max_stagnant_rounds} 轮")
                        rounds_without_improvement += 1
                else:
                    # gflops 不可用: 用速度比或时间判断
                    speedup = l3_result.metrics.get("speedup_vs_baseline", 0)
                    # 首轮同样无基线可比, 不禁用配方 (与上面 gflops 分支同理)
                    if round_num < 2 or (speedup > 0 and speedup < 1.0):
                        # 无 gflops 参考时禁用配方等于拿噪声做永久决策，跳过避免死锁
                        pass
                    elif plan and plan.selected_recipes:
                        new_recipes = [r for r in plan.selected_recipes if r.id not in applied_recipes]
                        for r in new_recipes:
                            disabled_recipes.add(r.id)
                    rounds_without_improvement += 1
                    print(f"\n[Conductor] 性能指标不完整 (gflops={current_gflops:.2f}, speedup={speedup:.4f}), "
                          f"连续 {rounds_without_improvement}/{max_stagnant_rounds} 轮")
                if rounds_without_improvement >= max_stagnant_rounds:
                    print(f"\n[Conductor] 连续 {max_stagnant_rounds} 轮无性能改善，提前终止优化")
                    snapshot_target_dir = _snap()
                    self._record_stop(artifact, round_num, "stagnant", "连续无改善达上限")

                    break

            # 缺口 4 的正解: 改善判据该看「离 roofline 上界还有多远」, 而不是继续拿
            # 一个标量比大小。绑定轴已经贴着上界的核, 换任何配方都不可能有可测收益 ——
            # 再跑一轮只是白烧 GPU 与 LLM token。
            #
            # 只判「本轮有没有 L3 结果」: 再往前 `all_passed` 已经 break, 而
            # `Verifier.verify()` 在 L1/L2 失败时直接 return (verifier.py:158-175),
            # 所以拿得到 L3 就意味着 L1/L2 过了、且 L3 没过。在这里再写
            # `not l3_result.passed` 或 `not l1_l2_failed` 都是永不生效的条件
            # (同一文件里刚为 `can_disable_recipe` 删掉过一个)。
            if l3_result is not None and roofline_converged(l3_result.metrics or {}):
                _att = binding_attainment(l3_result.metrics or {})
                _rl = (l3_result.metrics or {}).get("roofline_analysis") or {}
                print(f"\n[Conductor] 绑定轴已达 roofline {_att:.1f}% "
                      f"(>= {ROOFLINE_CONVERGED_PCT:.0f}%), 判定已收敛, 停止后续优化轮")
                # 这条 break 是整个循环里最贵的一次判定 (它决定不再花轮次), 所以把它
                # 依赖的分母连同出处一起落进日志: 以后复查时不需要回头猜 90% 是相对
                # 哪个数算出来的、那个数凭什么。
                print(f"[Conductor]   分母: [{_rl.get('peak_table_key', '?')}] "
                      f"peak={_rl.get('hw_peak_gflops', 0) / 1000:.1f} TFLOPS / "
                      f"bw={_rl.get('hw_peak_bandwidth_gbs', 0):.0f} GB/s "
                      f"({_rl.get('compute_used', '?')}) — {_rl.get('peak_source', '出处未知')}")
                snapshot_target_dir = _snap()
                self._record_stop(artifact, round_num, "converged", "绑定轴已达 roofline 上界")

                break

            l1_l2_failed = any(
                not r.passed and r.level in (VerificationLevel.L1_COMPILE, VerificationLevel.L2_FUNCTION)
                for r in verify_results
            )
            if l1_l2_failed:
                # L1(编译)/L2(功能) 失败: 携带错误信息进入 Fixer 修复闭环
                if round_num < self.max_optimize_rounds:
                    # L1 编译失败可能是缺依赖: 分析代码 import 并按需补装
                    l1_result = next((r for r in verify_results
                                      if r.level == VerificationLevel.L1_COMPILE and not r.passed), None)
                    if l1_result and Cap.DEP_ANALYSIS in self.backend.capabilities:
                        deps = self.backend.analyze_dependencies(current_code)
                        if deps:
                            print(f"\n[Conductor] 代码依赖: {deps}, 尝试按需补装...")
                            if Cap.ENV_PROVISION in self.backend.capabilities:
                                self.backend.ensure_environment(required_pips=deps)

                    failed_results = [r for r in verify_results if not r.passed]
                    print(f"\n[Fixer] L1/L2 验证失败，启动错误修复...")
                    try:
                        fixed_code, fix_note = self.fixer.run(
                            current_code, task, failed_results,
                        )
                    except Exception as e:
                        print(f"\n[Conductor] Fixer 修复失败 ({e})，停止迭代")
                        snapshot_target_dir = self._flush_snapshot(
                            task=task,
                            artifact_dir=snapshot_target_dir,
                            round_done=round_num,
                            current_sketch=current_sketch,
                            current_code=current_code,
                            pending_fix=None,
                            iteration_history=artifact.iteration_history,
                            best_gflops=best_gflops,
                            rounds_without_improvement=rounds_without_improvement,
                            optimization_history=optimization_history,
                            applied_recipes=applied_recipes,
                            disabled_recipes=disabled_recipes,
                            all_verification_results=all_verification_results,
                        )
                        self._record_stop(artifact, round_num, "fix_error", "Fixer 修复抛异常, 停止迭代")

                        break
                    print(f"  修复说明: {fix_note}")
                    pending_fix = fixed_code

                    # ---- Reflexion: 修复后让 Fixer 产出结构化教训并沉淀 ----
                    reflexion = None
                    try:
                        reflexion = self.fixer.produce_reflexion()
                        if reflexion is not None:
                            self.failure_db.add(reflexion)
                            reflexion_pending.append(reflexion)
                            lessons_cnt = len(reflexion.lessons)
                            cause = reflexion.root_cause or "N/A"
                            print(f"  [Reflexion] 已入库: 根因={cause[:60]}, lessons={lessons_cnt}条")
                    except Exception as _rx_err:
                        # Reflexion 失败不影响主流程, 只记录不中断
                        reflexion = None
                        print(f"  [Reflexion] 跳过沉淀: {_rx_err}")

                    # 第三个位置带的是**那条记录本身或 None**, 不是「已经沉淀过了」这个
                    # 声称。原先写死 True, 而唯一的循环内沉淀是 produce_reflexion() ——
                    # 它只在修好时返回记录, 于是「修不好的那类失败」被 815-821 的兜底
                    # 当成「已经记过了」跳过, 永远进不了负样本库 (doc 49 §7.14 / #56)。
                    fix_attempts.append((failed_results, fix_note, reflexion))

                    # ── 可复现性 Harness: Fixer 修复后走 continue 前也要落快照 (否则 crash 就丢了修复代码) ──
                    snapshot_target_dir = self._flush_snapshot(
                        task=task,
                        artifact_dir=snapshot_target_dir,
                        round_done=round_num,
                        current_sketch=current_sketch,
                        current_code=current_code,
                        pending_fix=pending_fix,
                        iteration_history=artifact.iteration_history,
                        best_gflops=best_gflops,
                        rounds_without_improvement=rounds_without_improvement,
                        optimization_history=optimization_history,
                        applied_recipes=applied_recipes,
                        disabled_recipes=disabled_recipes,
                        all_verification_results=all_verification_results,
                    )
                    continue
                print(f"\n[Conductor] L1/L2 验证失败且已达最大轮次，停止迭代")
                snapshot_target_dir = _snap()
                self._record_stop(artifact, round_num, "l1_l2_failed", "L1/L2 失败且已达最大轮次")

                break

            if l3_result and not l3_result.passed and round_num < self.max_optimize_rounds:
                print(f"\n[Search] L3 性能未达标，启动优化...")
                perf_data = {
                    "gflops": l3_result.metrics.get("gflops", 0),
                    "speedup": l3_result.metrics.get("speedup_vs_baseline", 0),
                    "target_speedup": task.constraints.min_speedup,
                    "bound_type": l3_result.metrics.get("bound_type", "unknown"),
                    "avg_time_ms": l3_result.metrics.get("avg_time_ms", 0),
                    "memory_bandwidth_gbs": l3_result.metrics.get("memory_bandwidth_gbs", 0),
                    # ---- Roofline 数据 (Verifier 注入) ----
                    "hw_peak_gflops": l3_result.metrics.get("hw_peak_gflops", 0),
                    "hw_peak_bandwidth_gbs": l3_result.metrics.get("hw_peak_bandwidth_gbs", 0),
                    "gflops_utilization_pct": l3_result.metrics.get("gflops_utilization_pct", 0),
                    "bandwidth_utilization_pct": l3_result.metrics.get("bandwidth_utilization_pct", 0),
                    "arithmetic_intensity": l3_result.metrics.get("arithmetic_intensity", 0),
                    "ridge_point": l3_result.metrics.get("ridge_point", 0),
                    "roofline_bound": l3_result.metrics.get("roofline_bound", "unknown"),
                    "name_bound": l3_result.metrics.get("name_bound", "unknown"),
                    "hw_name": l3_result.metrics.get("hw_name", "unknown"),
                    "hw_compute_capability": l3_result.metrics.get("hw_compute_capability", "unknown"),
                    # ---- 优化历史 ----
                    "optimization_history": optimization_history,
                }
                try:
                    current_sketch, suggestion = self.search.run(current_sketch, perf_data)
                except Exception as e:
                    # 规则优化也失败时, 终止优化但不崩整个任务
                    print(f"\n[Conductor] Search 优化异常 ({e}), 终止优化循环")
                    snapshot_target_dir = self._flush_snapshot(
                        task=task,
                        artifact_dir=snapshot_target_dir,
                        round_done=round_num,
                        current_sketch=current_sketch,
                        current_code=current_code,
                        pending_fix=pending_fix,
                        iteration_history=artifact.iteration_history,
                        best_gflops=best_gflops,
                        rounds_without_improvement=rounds_without_improvement,
                        optimization_history=optimization_history,
                        applied_recipes=applied_recipes,
                        disabled_recipes=disabled_recipes,
                        all_verification_results=all_verification_results,
                    )
                    self._record_stop(artifact, round_num, "search_error", "Search 优化抛异常, 终止优化循环")

                    break
                print(f"  优化建议: {suggestion}")
                print(f"  新 Sketch 版本: v{current_sketch.version}")

                # ---- 记录本轮优化历史 (供后续轮次参考) ----
                optimization_history.append({
                    "tiling": f"{current_sketch.tiling.block_m}x{current_sketch.tiling.block_n}x{current_sketch.tiling.block_k}",
                    "gflops": l3_result.metrics.get("gflops", 0),
                    "stage": "algorithm",  # 由 SearchAgent 返回实际 stage
                    "outcome": f"GFLOPS={l3_result.metrics.get('gflops', 0):.2f}, "
                               f"加速比={l3_result.metrics.get('speedup_vs_baseline', 0):.2f}x",
                })

            # ── 可复现性 Harness: 循环走到末尾 (没走 break/continue) 时统一 flush 一次快照 ──
            snapshot_target_dir = _snap()
            if not (l3_result and not l3_result.passed and round_num < self.max_optimize_rounds):
                print(f"\n[Conductor] 已达最大优化轮次，停止迭代")

        # 循环正常走到末尾 (没被任何 break 收走) —— 这一档以前只在 stdout 上有一行 print,
        # 产物里与「修坏了」「到顶了」完全同形。补记成数据。
        if not (artifact.metadata or {}).get("stop"):
            self._record_stop(artifact, self.max_optimize_rounds, "rounds_exhausted",
                              "轮次用完仍未通过, 也没有触发任何提前出口")

        # 构建最终产物
        artifact.code = current_code
        artifact.sketch = current_sketch
        artifact.verification_results = all_verification_results

        if all_verification_results:
            l3 = next((r for r in all_verification_results if r.level == VerificationLevel.L3_PERFORMANCE), None)
            if l3:
                artifact.benchmark_data = l3.metrics

            # 提取编译元信息 (Triton: num_warps, num_stages 等)
            l1 = next((r for r in all_verification_results if r.level == VerificationLevel.L1_COMPILE), None)
            if l1 and l1.metrics:
                artifact.compile_info = {
                    k: v for k, v in l1.metrics.items()
                    if k not in ("compile_status", "backend")
                }
            # 从 task 配置补充 Triton 编译参数
            if task.target.triton_config:
                artifact.compile_info.update(task.target.triton_config.to_dict())

        # 失败案例入库: 修复记录与最终失败均沉淀为负样本 (供后续 Fixer 检索)
        # 循环内只沉淀 Reflexion 那一条 (带 lessons 结构化教训), 且当时 `fixed` 只能是
        # 「还没验证过」; 这里补齐两件事: 没有 Reflexion 记录的修复走兜底入库,
        # 有 Reflexion 记录的在结果已知后补写一次。
        if fix_attempts:
            for failed_results, fix_note, reflexion in fix_attempts:
                if reflexion is not None:
                    continue      # 这条失败已经有它自己的记录 (下面那次补写会更新它)
                self._record_failure(task, failed_results, fix_note, fixed=artifact.all_passed)
        elif all_verification_results and not artifact.all_passed:
            failed = [r for r in all_verification_results if not r.passed]
            if any(r.level in (VerificationLevel.L1_COMPILE, VerificationLevel.L2_FUNCTION)
                   for r in failed):
                self._record_failure(task, failed, "", fixed=False)

        if artifact.all_passed:
            # `fixed` 的字段注释是「该失败**最终**是否被修复」, 而 produce_reflexion() 被调用的
            # 那一刻下一轮验证还没跑 —— 原先它无条件写 True, 于是每条反思都在结果未知时替结果
            # 写了答案, 且修的到底是哪一次也没人核对过。现在创建时留默认 False, 到这里才补写。
            # 追加式日志里同一条 id 的 fixed=False → fixed=True 演进是**设计支持的**
            # (kernel_db.py:806-812: 重放按后写覆盖先写), 所以这不是新增机制, 是用它本来的用法。
            for reflexion in reflexion_pending:
                reflexion.fixed = True
                self.failure_db.add(reflexion)
            if reflexion_pending:
                print(f"\n[Conductor] Reflexion 记录补写 fixed=True ({len(reflexion_pending)} 条)")

        # ── 成本监控 Harness: 汇总 LLM 账单, 写入 artifact.metadata ──
        if self.llm and hasattr(self.llm, 'cost_tracker'):
            tracker = self.llm.cost_tracker
            cost_summary = tracker.summary_since(self._cost_watermark)
            # 口径随行: 这份是**本次运行**的量, 不是这个进程到现在的累计。落盘与报表将来
            # 都要按它判事, 没有这行标注, 累计值与本次值在产物里长得一模一样。
            # `if cost_summary` 不是防御空账本 —— 假件/无账本的 LLM 返回 `{}`，给它补上
            # `scope` 会把「没有账本」伪装成「有一份只写了口径的账单」(把像样的值塞进失败里)。
            if cost_summary:
                cost_summary["scope"] = "this_run"
                cost_summary["process_total_calls"] = tracker.total_calls
                if artifact.metadata is None:
                    artifact.metadata = {}
                artifact.metadata["llm_cost"] = cost_summary

        # 保存产物到文件系统
        saved_dir = self.store.save(artifact, task)
        # Artifact 上记录一下落盘目录, 方便下游 (golden harness / 日志) 直接引用
        try:
            artifact.saved_dir = str(saved_dir)
        except Exception:
            pass
        # 保存后用权威 artifact 目录刷新 snapshot_target_dir, 并补写一次最终快照
        # (这样 resume_from=saved_dir 就能读到 execution_snapshot.json)
        snapshot_target_dir = saved_dir
        self._flush_snapshot(
            task=task,
            artifact_dir=snapshot_target_dir,
            round_done=max(
                (h.get("round") or 0) for h in artifact.iteration_history
            ) if artifact.iteration_history else 0,
            current_sketch=current_sketch,
            current_code=current_code,
            pending_fix=None,  # 循环结束就没有 pending 了
            iteration_history=artifact.iteration_history,
            best_gflops=best_gflops,
            rounds_without_improvement=rounds_without_improvement,
            optimization_history=optimization_history,
            applied_recipes=applied_recipes,
            disabled_recipes=disabled_recipes,
            all_verification_results=all_verification_results,
        )
        print(f"\n[Conductor] 产物已保存到: {saved_dir}")

        # ── 成本监控 Harness: 打印账本摘要 (save 之后打, 不影响持久化时序) ──
        if self.llm and hasattr(self.llm, 'cost_tracker') and self.llm.cost_tracker.total_calls > 0:
            print(f"\n[Conductor] {self.llm.cost_tracker.pretty()}")

        if artifact.all_passed:
            self._save_to_db(artifact, task)
        # 每次运行都在日志里报一句语料现状。这条的存在理由是缺口 2 的失效方式本身:
        # 「RAG 命中 N 个相似案例」以前只说明语料里有 N 条, 不说明它们是不是真的
        # 从上一次运行继承下来的 (答案曾经是「不是」)。
        print(f"[KernelDB] {self.kernel_db.summary()} "
              f"本次取用 {self.kernel_db.retrievals_this_run} 次命中")
        # 负样本侧同样每轮报一句。这条的读者是**复发门到底挡了多少**: 语料在长、门在不在
        # 起作用, 不看这一句就只能靠读代码相信它存在 (doc 45 第二节的失效方式)。
        print(f"[FailureDB] {self.failure_db.summary()} "
              f"本次被复发门挡住 {len(self.failure_db.suppressed_by_gate)} 条"
              f"·被类别门挡住 {len(self.failure_db.suppressed_by_kind)} 条")
        # 候选新颖度: 现在只有「重复了几轮」这一个数, 它的读者是 doc 47 里那条基率 ——
        # 在拿签名去省 GPU 时间之前, 先得知道重复到底常不常见。
        n_rounds = len(artifact.iteration_history)
        print(f"[Novelty] {n_rounds} 轮候选, {len({s for _, s in round_signatures})} 个不同结构, "
              f"其中 {duplicate_rounds} 轮与更早的轮次结构相同")

        if invalid_measurement_rounds:
            # 括号里以前写死「污染 / launch 地板支配」两项, 而判据已经有五项 (还有
            # 样本离散度、数字的出处、环境漂移)。列举会过期, 逐轮的原因本来就在上面
            # 打过一遍, 这里只报数。
            print(f"\n[Conductor] ⚠️ 本次 {invalid_measurement_rounds} 轮的 L3 测量被判为不可信 "
                  f"(逐轮原因见上), 这些轮次未参与配方禁用与停滞计数")

        if refused_rounds:
            print(f"\n[Conductor] ⛔ 本次 {refused_rounds} 轮的 L3 因**出处不对**而不记为通过 "
                  f"(任务声明了执行侧没有的能力) —— 这些轮不是「核太慢」, 是压根没测过"
                  f"一个可判定的东西; 它们烧了 LLM 与远程编译, 没产出关于这台机器的证据")

        self._print_summary(artifact, task, saved_dir)

        # 执行结束后打印实际使用的 LLM 模型 (可能已降级)
        if self.llm and hasattr(self.llm, '_fallback_triggered') and self.llm._fallback_triggered:
            print(f"\n[Conductor] ⚠️ 本次执行已降级: 主模型 [{self.llm.model}] 失败, "
                  f"实际使用 [{self.llm._active_model}] @ {self.llm._active_base_url}")
        return artifact

    @staticmethod
    def _measurement_drives_loop(l3_result: VerificationResult, round_num: int) -> bool:
        """本轮 L3 数字能否驱动闭环决策 (配方禁用 / 停滞计数 / 记功)

        缺口 1 的原失效方式不是「没算出来」而是「算出来了没人读」: Verifier 早就把
        `benchmark_contaminated` 写进了 metrics, 但 conductor / fixer / agents 三个
        文件里 grep 零命中, `passed` 照旧。于是污染压低 gflops → 判「未改善」→
        `disabled_recipes.add()`, 而 disabled_recipes 在后续轮次不会恢复。

        metrics 里没有有效性键时返回 True: 旧的 execution_snapshot 与不带 L3 的
        路径都不产这几个键, 对它们沿用改动前的行为, 不做追溯性重判。
        """
        m = l3_result.metrics or {}
        if m.get("measurement_can_count_stagnation", True):
            return True
        print(f"\n[Conductor] 第 {round_num} 轮测量有效性="
              f"{m.get('measurement_validity', 'unknown')}, "
              f"不禁用配方 / 不计入停滞轮次")
        for reason in m.get("measurement_validity_reasons") or []:
            print(f"  [Conductor]   - {reason}")
        return False

    # ─── 可复现性 Harness: 快照读写辅助 ────────────────────────────────────────────

    SNAPSHOT_NAME = "execution_snapshot.json"

    def _resolve_snapshot_path(self, resume_from: Optional[str]) -> tuple:
        """把 `resume_from` (快照文件 / artifact 目录 / task_id) 解析成快照路径。

        返回 `(path, scan)` —— `scan` 是给人和机器一起看的账: 在哪个根目录下找、看到几个
        同 task_id 的候选、其中几个读不动。**以前不返回这些**，于是「按 task_id 续跑」这件事
        的失效形态只剩一行 print (doc 50)。

        两处口径是这轮改的:
        ① 搜索根取 `self.store.output_dir` 而不是写死的 `"output"` —— 后者不认
           `OPERATOR_AGENT_OUTPUT_DIR`，设了重定向的运行会在 CWD 找一个根本没写过的目录，
           结果要么找不到 (静默从头跑)，要么捡到**别的配置**留下的快照。
        ② 多个候选时取 `(round_done, mtime)` 最大的**可解析**那个 —— 以前是 `os.walk` 的
           第一个命中，而 walk 顺序既不是时间序也没人规定过；实测两个候选 (round 1 与 3)
           永远选中 round 1，即续跑会把已完成的轮次扔掉重来。
        """
        scan = {"root": str(self.store.output_dir), "candidates": 0, "unreadable": 0}
        if not resume_from:
            return None, scan
        # 1) 直接就是快照文件
        if os.path.isfile(resume_from):
            return resume_from, scan
        # 2) artifact 目录
        candidate = os.path.join(resume_from, self.SNAPSHOT_NAME)
        if os.path.isfile(candidate):
            return candidate, scan
        # 3) 也许是 task_id: 在**本次配置真正写产物**的那棵树下按 id 找
        root = Path(self.store.output_dir)
        if not root.is_dir():
            return None, scan
        best_key, best_path = None, None
        for path in sorted(root.rglob(self.SNAPSHOT_NAME)):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                scan["unreadable"] += 1
                continue
            if not isinstance(data, dict) or data.get("task_id") != resume_from:
                continue
            scan["candidates"] += 1
            key = (int(data.get("round_done") or 0), path.stat().st_mtime)
            if best_key is None or key > best_key:
                best_key, best_path = key, str(path)
        return best_path, scan

    @staticmethod
    def _build_snapshot_dict(*, task: TaskSchema, round_done: int,
                             current_sketch: Optional[UnifiedSketch],
                             current_code: str,
                             pending_fix: Optional[str],
                             iteration_history: list[dict],
                             best_gflops: float,
                             rounds_without_improvement: int,
                             optimization_history: list[dict],
                             applied_recipes: set,
                             disabled_recipes: set,
                             all_verification_results: list,
                             environment: Optional[dict] = None,
                             ) -> dict:
        """把运行时状态打包成可 JSON 化的 dict"""
        def vr_to_dict(vr: VerificationResult) -> dict:
            return {
                "level": vr.level.value,
                "passed": vr.passed,
                "message": vr.message,
                "error_log": vr.error_log,
                "metrics": vr.metrics or {},
            }
        env = environment or {}
        return {
            "schema_version": 1,
            "created_at": __import__("time").time(),
            "task_id": task.task_id,
            "kernel_type": task.source.kernel_type,
            "hardware": task.target.hardware,
            "dsl": task.target.dsl,
            "round_done": round_done,
            "sketch_dict": current_sketch.to_dict() if current_sketch else None,
            "current_code": current_code,
            "pending_fix": pending_fix,
            "iteration_history": list(iteration_history or []),
            "best_gflops": float(best_gflops or 0.0),
            "rounds_without_improvement": int(rounds_without_improvement or 0),
            "optimization_history": list(optimization_history or []),
            "applied_recipes": sorted(applied_recipes or set()),
            "disabled_recipes": sorted(disabled_recipes or set()),
            "verification_results": [vr_to_dict(v) for v in (all_verification_results or [])],
            # 指纹进快照是为了续跑: `resume_from` 会把 `best_gflops` 与已禁用配方一并
            # 继承过来, 环境一变那些就不再可比 —— 那是漂移唯一直接污染判决的场合
            # (doc 49 §3.5 第 1 处)。旧快照没这个键, 读出来是 None, 比较时会如实报
            # 「没有可比的上一次」而不是「没漂移」。
            "environment_fingerprint": env.get("fingerprint") or None,
            "environment_drift": env.get("drift"),
        }

    def _flush_snapshot(self, *, task: TaskSchema, artifact_dir: Optional[str],
                        round_done: int, current_sketch: Optional[UnifiedSketch],
                        current_code: str, pending_fix: Optional[str],
                        iteration_history: list, best_gflops: float,
                        rounds_without_improvement: int,
                        optimization_history: list,
                        applied_recipes: set, disabled_recipes: set,
                        all_verification_results: list) -> Optional[str]:
        """写 execution_snapshot.json 到 artifact_dir; artifact_dir 不存在时用 output/<kernel>/<hw>/snap_<task_id>/ 预创建。

        返回实际写入的目录 (用于把 snapshot_target_dir 指向该预创建目录, 避免后续轮次错位)。
        """
        snap = self._build_snapshot_dict(
            task=task, round_done=round_done,
            current_sketch=current_sketch, current_code=current_code,
            pending_fix=pending_fix,
            iteration_history=iteration_history,
            best_gflops=best_gflops,
            rounds_without_improvement=rounds_without_improvement,
            optimization_history=optimization_history,
            applied_recipes=applied_recipes,
            disabled_recipes=disabled_recipes,
            all_verification_results=all_verification_results,
            environment=self._environment,
        )
        target_dir = artifact_dir
        # 若没有 artifact_dir (第 1 轮还没走 save), 就预创建一个 "staging" 快照目录。
        # 根目录取 `self.store.output_dir` 而不是写死 "output": 后者绕开
        # `OPERATOR_AGENT_OUTPUT_DIR`, 于是设了重定向的运行会把快照落进仓库的 output/ ——
        # 那个环境变量存在的全部理由就是别让旁路产物混进真实语料 (doc 43 第三节)。
        if not target_dir:
            import time as _t
            ts = _t.strftime("%Y%m%d_%H%M%S")
            target_dir = str(self.store.output_dir /
                             (task.source.kernel_type or "operator") /
                             (task.target.hardware or "unknown") /
                             f"snap_{ts}_{task.task_id}")
        try:
            os.makedirs(target_dir, exist_ok=True)
            path = os.path.join(target_dir, "execution_snapshot.json")
            # **先写临时文件再 `os.replace`**。原先是 `open(path,"w")` 直接覆写: 那个 open
            # 当场把文件截成 0 字节, 于是一次崩溃/磁盘满就留下一份截断 JSON, 而下一次的
            # 读取侧把「读不出」当成「没有快照」从头重跑 (真机量测: 一次续跑从 1 次 LLM
            # 调用变成 4 次)。同目录内的 rename 是原子的, 所以读者要么看到旧的完整版本,
            # 要么看到新的完整版本, 永远不会看到半个。
            tmp = path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(snap, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            return target_dir
        except Exception as e:
            # 快照写入失败绝不能打断主任务 — Harness 是旁路
            print(f"  [Snapshot] ⚠️ 快照写入失败, 忽略: {type(e).__name__}: {e}")
            return artifact_dir

    @staticmethod
    def _load_snapshot(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema_version") != 1:
            raise ValueError(f"不支持的 snapshot schema_version={data.get('schema_version')}")
        return data

    def _restore_state_from_snapshot(self, snap: dict, task: TaskSchema):
        """从 snap dict 还原 _execute_inner 的运行时变量, 返回 tuple 按顺序供解包:

        (start_round, current_sketch, current_code, pending_fix,
         iteration_history, best_gflops, rounds_without_improvement,
         optimization_history, applied_recipes, disabled_recipes,
         all_verification_results)
        """
        from .schemas import VerificationLevel as _VL
        sketch_dict = snap.get("sketch_dict")
        current_sketch = UnifiedSketch.from_dict(sketch_dict) if sketch_dict else None
        if current_sketch is None:
            current_sketch = UnifiedSketch(
                algorithm=task.source.kernel_type, version=1, target_dsl=task.target.dsl,
            )

        start_round = int(snap.get("round_done") or 0) + 1
        current_code = snap.get("current_code") or ""
        pending_fix = snap.get("pending_fix") or None
        iteration_history = list(snap.get("iteration_history") or [])
        best_gflops = float(snap.get("best_gflops") or 0.0)
        rounds_without_improvement = int(snap.get("rounds_without_improvement") or 0)
        optimization_history = list(snap.get("optimization_history") or [])
        applied_recipes = set(snap.get("applied_recipes") or [])
        disabled_recipes = set(snap.get("disabled_recipes") or [])

        vr_list = []
        for d in snap.get("verification_results") or []:
            try:
                lv = _VL(d.get("level") or "L1")
            except Exception:
                lv = _VL.L1_COMPILE
            vr_list.append(VerificationResult(
                level=lv,
                passed=bool(d.get("passed", False)),
                message=str(d.get("message") or ""),
                error_log=d.get("error_log"),
                metrics=d.get("metrics") or {},
            ))

        return (
            start_round, current_sketch, current_code,
            pending_fix, iteration_history,
            best_gflops, rounds_without_improvement,
            optimization_history, applied_recipes, disabled_recipes,
            vr_list,
        )

    def _save_to_db(self, artifact: KernelArtifact, task: TaskSchema):
        """把这次的核写进跨运行经验库 (缺口 2)。门不过就只进内存, 不落盘。

        刻意放在 `artifact.all_passed` 那一支里、但**再**过一道 `persist_decision`:
        all_passed 只说明「三级验证都点了通过」, 而缺口 1 的真机数据证明那个通过可以是
        被 launch 地板或邻居抢占造出来的 (doc 31/32/33)。持久化会把一次假信号变成
        下一次检索的样本, 所以这里比 Voyager 的 `if success` 多两条。
        """
        from .kernel_db import KernelRecord
        from .measurement import persist_decision

        metrics = artifact.benchmark_data or {}
        keep, reason = persist_decision(metrics)
        record = KernelRecord(
            kernel_id=artifact.kernel_id,
            algorithm=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=task.target.dsl,
            code=artifact.code,
            sketch=artifact.sketch.to_dict() if artifact.sketch else None,
            performance=metrics.get("gflops", 0),
            tags=[task.source.kernel_type, "verified", f"v{artifact.sketch.version if artifact.sketch else 1}"],
            compile_info=artifact.compile_info,
            score=float(metrics.get("speedup_vs_baseline") or 0.0),
            measurement_validity=str(metrics.get("measurement_validity") or ""),
        )
        if not keep:
            print(f"[KernelDB] 未入库 (只留在本次进程): {reason}")
            self.kernel_db.add(record)
            return False
        self.kernel_db.promote(record)
        print(f"[KernelDB] 已入库并落盘: {reason}")
        return True

    def _record_failure(self, task: TaskSchema, failed_results: list[VerificationResult],
                        fix_note: str, fixed: bool):
        """将验证失败/修复记录写入失败案例库 (负样本)"""
        # 层级必须**排序去重**后再拼: 原先的 "+".join(r.level.value for r in ...) 跟着
        # failed_results 的顺序走, 同一份失败可以写成 "L1+L2" 也可以写成 "L2+L1",
        # 于是签名碎成两个桶 —— 而复发门要数的就是同一个桶里的任务数 (fixer 那边
        # 一直是 sorted 的, 两边不一致)。
        levels = "+".join(sorted({r.level.value for r in failed_results}))
        error_summary = " | ".join(
            FixerAgent._truncate_error(r.error_log or r.message, 300)
            for r in failed_results
        )
        signature = failure_signature(levels, error_summary)
        self.failure_db.add(FailureRecord(
            # id 由内容决定, 不再用 `task_id_{库里现有条数}`: 后者是**插入顺序**的函数,
            # 一旦语料跨运行重放, 同一个任务的同一类失败每次都会拿到新 id, 于是
            # 「这算是老问题还是新问题」在库里根本没有答案。
            failure_id=f"cx-{task.task_id}:{signature}",
            kernel_type=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=task.target.dsl,
            level=levels,
            error_summary=error_summary,
            fix_summary=fix_note,
            fixed=fixed,
            signature=signature,
            task_id=task.task_id,
        ))
        print(f"[Conductor] 失败案例已入库 (fixed={fixed}, level={levels}, 签名={signature})")

    def _print_summary(self, artifact: KernelArtifact, task: TaskSchema, saved_dir=None):
        print(f"\n{'='*60}")
        print(f"Conductor: 任务完成 {task.task_id}")
        print(f"{'='*60}")
        print(f"  Kernel ID:     {artifact.kernel_id}")
        print(f"  代码长度:       {len(artifact.code)} chars")
        print(f"  Sketch 版本:   v{artifact.sketch.version if artifact.sketch else 'N/A'}")
        print(f"  验证结果:")
        for r in artifact.verification_results:
            status = "PASS" if r.passed else "FAIL"
            print(f"    {r.level.value} [{status}] {r.message}")
        if artifact.benchmark_data:
            print(f"  性能数据:")
            for k, v in artifact.benchmark_data.items():
                print(f"    {k}: {v}")
        print(f"  总体验证: {'ALL PASSED' if artifact.all_passed else 'FAILED'}")
        if saved_dir:
            print(f"  产物目录: {saved_dir}")
        print(f"{'='*60}\n")
