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
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union
from .eligibility import decision_speedup
from .schemas import (
    TaskSchema, RouteType, VerificationLevel, VerificationResult,
    UnifiedSketch, KernelArtifact,
)
from .agents import DesignerAgent, CoderAgent, SearchAgent
from .fixer import FixerAgent, failure_class
from .verifier import COMPILE_FACT_KEYS, Verifier
from .kernel_db import (
    KernelDatabase, FailureDatabase, FailureRecord, FAILURE_CLASS_REFERENCE,
    failure_signature,
)
from .novelty import signature_of
from .backend_caps import Cap
from .mock_gpu import MockGPU
from .remote_executor import RemoteExecutor
from .llm_client import LLMClient
from .prompt_manager import PromptManager
from .artifact_store import ArtifactStore
from .environment import EnvironmentDrift, compare_fingerprints, fingerprint_of
from .bandit import Bandit
from .exhaustion import (FROM_LLM_ON_DEMAND, FROM_TEMPLATE_SPACE,
                          CoverageLedger, assess_coverage, undeclared)
from .selection import DEFAULT_EPSILON, choose, explore_draw, phase_of
from .source_lang import flag_from_metrics
from .escalation import (
    EscalationInputs, decide, occupancy_bound_evidence, smem_bound_evidence,
    untried_directions,
)
from .hardware_caps import detect_hardware_caps
from .resource_check import caps_from_metrics, check_resource_feasibility
from .candidates import (param_space_of, point_label, require_supported_space,
                        select_template_candidate, template_arm, template_points)
from .budget import BudgetLedger, BudgetPlan
from .measurement import (BAND_SPEC_DRIFT, BAND_UNEXPLAINED, LINEAGE_BAND_KEY,
                          LINEAGE_PERF_RULER_KEY, LINEAGE_RULER_KEY,
                          PERF_GATE_PARAM_KEYS, REFERENCE_CEILING_PCT,
                          RULER_LABEL_CORRECTNESS,
                          RULER_LABEL_PERFORMANCE, TRANSPORT_KEY, TRANSPORT_UNKNOWN, Verdict,
                          Validity, band_note, binding_attainment,
                          cross_round_band, cross_round_ruler,
                          correctness_verified_levels, l2_ruler_reading,
                          l3_ruler_reading, reference_ceiling_reached,
                          ruler_note, verdict)
from .agent_logger import capture_execution
from .optimization_planner import OptimizationPlanner
from .run_layout import ROLE_CANDIDATE, RunLayout, run_layout
from .exec_guard import Limits
from .agent_runtime import (
    KIND_PROPOSE_SOURCE_PATCH, KIND_PROPOSE_TEMPLATE, KIND_REPAIR_COMPILE,
    KIND_REPAIR_CORRECTNESS, KIND_EXPLAIN_FAILURE, PATCH_KINDS,
    ACTION_SOURCE_HARNESS, ACTION_SOURCE_LLM, ENVELOPE_ABSENT,
    ActionContext, ActionEnvelope, AgentAction, AgentRuntime, Patch,
    ask_unavailable_reason, candidate_path,
)
from .failure_route import classify as classify_failure
from ._harness_controls import enabled as harness_controls_enabled
from . import run_contract


# 生产运行必须在有限步长内重新校准量具；超大值只允许测试/benchmark
# harness 通过进程内控制打开，不能由任务字段或环境变量关闭心跳。
MAX_HEARTBEAT_INTERVAL = 100


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


def correctness_verified(results: list) -> bool:
    """L1 与 L2 **都到场且都通过**才算「正确性已验证」

    外部一手 (arXiv:2609.35140v1 行 214, 逐字): "If an offspring fails, revisions must start
    from a retained **verified** parent, not that offspring." 我方要落地它, 第一件事不是改
    Fixer, 而是把「这一版被验证过」写成一条能判的谓词 —— 而这里的「verified」是**正确性**
    (L1+L2), 不含 L3 的性能门: 一个「正确但不够快」的核完全可以当修订起点, 一个「编不过」的
    不行。

    「verified 是正确性 (L1+L2), 不含 L3 的性能门」与「两级都要在场」这两条**不在这里定义**,
    它们在 `measurement.CORRECTNESS_LEVELS` / `correctness_verified_levels` —— 端到端报表的
    「摘掉真机那一栏」(#77) 读的是同一个集合。所以下面这层只是把 `VerificationResult` 对象
    映射成那个 dict 形状。
    """
    return correctness_verified_levels({r.level.value: r.passed for r in results})


def verified_parent_from(iteration_history: list[dict],
                         current_code: str) -> Optional[str]:
    """续跑时从血缘里重建「最近一份已验证的实现」, 重建不出来就返回 None

    判据取**最后一条**血缘: 快照里的 `current_code` 就是那一轮被验证的那份代码, 所以只有
    最后一轮过了 L1/L2 时两者才对得上。更早的已验证版本不在快照里 (血缘只存了每轮的
    `levels_passed` 与签名, 没存代码文本), 于是那种情况下这里**如实返回 None** ⇒ Fixer
    退回今天的形状 (在失败那一版上继续修), 而不是猜一个祖先。

    这一条「续跑后只能重建一步」的限制是有意接受的, 写在 doc 58 §三 的 #85 落地段里。
    """
    if not iteration_history or not current_code:
        return None
    last = iteration_history[-1]
    if correctness_verified_levels(last.get("levels_passed") or {}):
        return current_code
    return None


def _correctness_credential(results) -> Optional[bool]:
    """这份候选的**正确性凭据**, 从它自己的 L2 结果里读 (三态)。

    设计 §14-10 / §20.4: 性能观测的晋级要求"这个核算得对"是**已验证的事实**,
    而 `Verifier.verify_l3_only` 是唯一能绕开 L2 产出 L3 观测的入口 —— 所以凭据必须由
    调用方按**它手里那份结果**给出, 不由 Verifier 假设、也不看实例状态 (与
    `static_guards` 同一条纪律: 事实属于那一份候选, 不属于那个 Verifier 实例)。

    三态, 与 `source_lang.flag_from_metrics` 同边界:
      * 有 L2 结果 → 它过没过 (`all(...)`: 多组之间取最保守, 一组没过就是没过);
      * 没有 L2 结果 → `None` = 没凭据。**不写成 False** —— "这一格空着"与
        "验过且不合格"是两件事, 混成一个值会让"没检查"伪装成"检查过且坏"。
    """
    l2 = [r for r in (results or [])
          if getattr(getattr(r, "level", None), "name", "") == "L2_FUNCTION"]
    if not l2:
        return None
    from .correctness import credential
    return credential(l2)


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


def _round_ruler(verify_results: list) -> dict:
    """本轮 L2 报的那把尺, 收成能进血缘的那一份读数 (#87 第 2 片)

    三态判据与键名都在 `measurement.l2_ruler_reading` / `GATE_PARAM_KEYS` —— 这里只负责从
    `VerificationResult` 列表里把 L2 那一格挑出来。挑不出来 (L1 就死了, `verify()` 是
    fail-fast) 时得到的是 `unreported`, 而**不是**「这一轮的尺和上一轮一样」。

    为什么要单独存一份而不是让读者去翻 `levels_passed`: 尺的**值**只有在 L2 的 metrics 里,
    而那份在循环结束时会被最后一轮的盖掉 (与 #63 那批「轴只活到本次判决结束」同形)。
    血缘是唯一穿过落盘边界的地方, 于是「第 1 轮与第 5 轮用的是不是同一把尺」这个问题
    今天第一次可以离线复盘, 而不必重跑。
    """
    l2 = next((r for r in verify_results
               if getattr(r, "level", None) == VerificationLevel.L2_FUNCTION), None)
    return l2_ruler_reading(getattr(l2, "metrics", None) if l2 else None)


def _round_perf_ruler(verify_results: list) -> dict:
    """本轮 L3 报的那把性能尺 (#87 第 3 片) —— 与上面那条同形, 只是取的是另一半

    这一侧 `unreported` 的**发生率比 L2 高得多, 而且大多数时候不是缺陷**: 层级门是
    fail-fast (L1/L2 没过就根本没有 L3 那一格), 而重复候选根本不烧 L3 (#41 那条能力)。
    所以「10 轮里只有 3 轮报了性能尺」今天是一句正常的话 —— 正因为这样, 这一格绝不能
    被读者折成「其余 7 轮的尺与第 3 轮相同」: 那才是本仓那一族 (把没检查当检查过)。
    """
    l3 = next((r for r in verify_results
               if getattr(r, "level", None) == VerificationLevel.L3_PERFORMANCE), None)
    return l3_ruler_reading(getattr(l3, "metrics", None) if l3 else None)


def _round_transport(verify_results: list):
    """本轮 L2 有没有「听到回答」—— 没有则这一轮的执行结果**不能**被读成「核跑错了」

    依据 Limbo `2609.29095v1` 行 226-232: 一次超时在 absent / committed(回执丢了) / late
    (还在飞) 三种世界里对客户端同形, 所以恢复策略不能直接按结果态分支 —— 它要么把这一格
    单独记下来, 要么选一个在三种世界里都安全的动作。我方此前两样都没做 (doc 60 §四)。

    返回 `None` 表示**这一轮没报** (L1 就死了、或本地执行没走远程那条 except) —— 「没报」与
    「报了 answered」在报表里是两栏, 不许折叠 (缺的键不填默认值, 与 `aggregated_over` 同一纪律)。
    """
    l2 = next((r for r in verify_results
               if getattr(r, "level", None) == VerificationLevel.L2_FUNCTION), None)
    metrics = getattr(l2, "metrics", None) if l2 else None
    return (metrics or {}).get(TRANSPORT_KEY)


_PERF_ADVICE = ("门槛或判据换了去查部署; 只有绑定轴换了通常是候选自己变了形状")


#: 连续无改善上限 (轮)。它同时是 §10.3 条件 1 里的那个 K。
#:
#: 为什么必须是**同一个数**: 两个独立的数会长成两种谁也没设计过的行为 ——
#: K 比它大 ⇒ 循环先停了, 升级门**永远开不了** (第一版 K=3 对着它=2, 实测撞到);
#: K 比它小 ⇒ 在同一段平台上反复放行。取「相等」时语义是清楚的: **循环准备放弃的
#: 那一轮**, 正是唯一一次问 LLM 要不要做结构性跳跃的机会 —— 而升级买到的是 K 轮
#: (停滞计数清零, 因为那段平坦区是对**旧结构**的陈述)。
MAX_STAGNANT_ROUNDS = 2


@dataclass
class RunState:
    """一次运行的状态 —— 轮次循环与收尾阶段共同读写的那一份东西。

    **为什么需要它**: `_execute_inner` 的 1128 行轮次循环与它后面的收尾, 靠二十多个
    同名局部变量互相通信。这些名字在收进来之前只是「函数里碰巧还在作用域内的东西」:
    谁能改它、改完谁看得见, 只能靠通读 1600 行来回答。收成一个对象之后, 这个问题才有
    有限答案 —— 「哪些值跨轮活下来」是一份**字段清单**, 而不是作用域的副产品。

    **字段清单不是随手定的**, 它由三条**已经存在**的契约交叉确定:
      * `_restore_state_from_snapshot` 恢复的那 10 项 = 快照里跨轮活下来的状态;
      * `_finalize_run` 收尾要读的全部东西;
      * 轮次循环自己攒的跨轮记忆 (`verified_parent` / `verified_parent_round` /
        `snapshot_target_dir` / `search_alternatives` / `plan` / `last_tunable_perf`)。这一组是 Stage2 把循环体搬成
        `_run_round` 时才现形的: 它们在**同一轮里先读后写**, 所以必须有值从上一轮传进来 ——
        在原作用域里这是隐形的, 搬成方法后第一轮就是 UnboundLocalError。后两位的形状更隐蔽:
        赋值在 if 这一支、读在 elif 的另一支 —— 光比行号顺序看不出来, 要按「可能未绑定就读取」
        的数据流分析才现形 (所以判据也照数据流写, 见 tests/test_run_state_single_source.py)。它们**不在快照里**:
        进程重启后回到 setup 初值 (既有行为, 此处只让它可见)。
    `tests/test_run_state_single_source.py` 把这三条都钉成了断言 —— 少一个字段,
    或加一个没人读的字段, 都会变红。

    只读上下文那一组: 绑定在整次运行里不变, 但值本身可能是可变的累积器
    (`bandit` / `coverage_ledger` 内部一直在攒东西) —— 变的是它们内部, 不是这个字段。
    """

    # ---- 只读上下文: 整次运行不变的绑定 ----
    task: TaskSchema
    target_gflops: float
    test_inputs: Mapping
    input_sets: list
    declared_cases: set
    case_source: str
    #: 选型种子 (字符串, 由 task_id + 后端标签拼成 —— 不是随机数)
    _selection_seed: str
    bandit: Bandit
    coverage_ledger: CoverageLedger

    # ---- 产物 ----
    artifact: KernelArtifact

    # ---- 跨轮活下来的状态 (与 _restore_state_from_snapshot 的恢复清单逐一对应) ----
    start_round: int
    current_sketch: Optional[UnifiedSketch]
    current_code: Optional[str]
    pending_fix: Any
    best_gflops: float
    rounds_without_improvement: int
    optimization_history: list
    applied_recipes: set
    disabled_recipes: set
    all_verification_results: list

    # ---- 也只跨轮活下来, 但**不在快照契约里** (见类文档) ----
    verified_parent: Optional[str]
    verified_parent_round: int
    snapshot_target_dir: Any
    search_alternatives: list
    plan: Any
    last_tunable_perf: Any

    # ---- 只在收尾读的累积量 ----
    fix_attempts: list
    reflexion_pending: list
    duplicate_rounds: int
    refused_rounds: int
    invalid_measurement_rounds: int
    round_signatures: list


#: _run_round 的返回值: 轮次循环接下来该继续还是该停。
#: 用具名常量而不是裸 True/False —— 那 14 处 return 是从 break/continue 换过来的,
#: 读的人需要一个能自查的名字 (而不是回头数这个布尔是什么极性)。
_ROUND_CONTINUE = True
_ROUND_STOP = False


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
        # P1-1 (generation-side-review): pipeline 是默认路径 —— Planner→3 路并行
        # 采样→compile-gate 剪枝→修复→模板兜底, 把「留哪一个」的选择权交给验证侧
        # (决赛候选逐个跑 L1+L2, 见 _select_finalist)。单体路径保留为调试逃生门。
        use_coder_pipeline: bool = True,
        # 标定心跳 (设计 §10.5-1): 每 N 次**真值测量**前塞一对已知参照核。
        # 为什么是 N 而不是每次都塞: 心跳自身要花机时, 而它要抓的是「尺子随时间的
        # 缓慢漂移 / 悄悄坏掉」, 不需要逐次; 短跑根本到不了期, 于是默认不会给已有
        # 链路添任何行为变化。
        heartbeat_interval: int = 5,
        # Optional independently audited identity, only narrows acceptance.
        cuda_calibration_identity_hash: str = "",
        cuda_ruler_profile_path: Optional[str] = None,
        # ⚠️ 这里**曾经**有一个 `require_double_confirm: bool = False` 的构造参数。
        # 设计 §3.5 把「长期记忆的独立确认」列在**不允许配置**那一栏 (无开关、无环境
        # 变量、不可被任务参数改写), 而那个默认 `False` 的参数正好是「默认关的安全装置
        # 等于不存在」那条纪律的反例 —— 默认配置下单次读数就能进语料。现在它是
        # **无条件**的: `all_passed` 的产物先补一次独立重测 (`_produce_remeasure`),
        # 再过 `admit_to_corpus` 的两次确认那道门。消融 (§14-9 要的「关掉资格审计的后果」
        # 那一臂) 只能在 harness 侧 patch, 见 `tests/test_spine_wiring.py` 里那条用例
        # —— §3.5 规则 3: 能一键关掉的尺子会先被用在无人值守的夜里。
        # 设计 §10.2 的三种预算 (`operator_agent/budget.py`)。**默认全关**: 没设上限
        # 就不拦, 也不假装拦过 —— 「本轮要不要花这次钱」是任务级的决定, 不是模块能替
        # 调用方拍的。
        budget: Optional[BudgetPlan] = None,
        # §10.3 的 ε (探索概率)。设计没给具体数 ⇒ `selection.DEFAULT_EPSILON` (偏探索)。
        # 做成构造参数是为了 §14-9 的对照实验: ε=0 就是「纯利用」那条基线 —— 而把它
        # 放在 `execute()` 的局部状态里 (像 `max_stagnant_rounds` 那样) 就没法做对照。
        selection_epsilon: float = DEFAULT_EPSILON,
        # §21 Agent Runtime (结构化动作 + 进编译前的独立检查)。默认**自己建一个**:
        # 一个「默认没有 Runtime」的控制器会让 §21 退化成「测试里才存在的一层」，
        # 而那正是本仓最贵的缺陷类型（判据算了但没人调）。
        # 注入的用途是替身与替身级对照（`tests/test_agent_runtime_wiring.py`）。
        agent_runtime: Optional[AgentRuntime] = None,
        # §22.1 的运行布局根。默认 `""` ⇒ **不建布局**，候选的角色路径是
        # `candidate/code.<ext>`（§21 的动作协议活在「声明」那一档）。传了真实
        # `run_root` ⇒ 每次 `execute` 建/核一次布局（七个角色目录 + 权限位），
        # 并且 `Patch.path` 变成**真实绝对路径**（同一个判据，输入端换了一档）。
        # ⚠️ 仓内今天**没有一处生产调用点传它** —— 传了才建，所以当前形态是
        # 「声明 + 可运行守卫」，不是「线上已经在拦」。
        run_root: str = "",
        # §21.3 的「结构化证据」进不进**修复提示**。默认开：设计 §21.3 是强制句
        # （「每次 LLM 调用**只**提供」那八项），而这一段是它的生产者。
        # 做成构造参数的理由与 `selection_epsilon` 同一个：**对照实验需要关掉它的那一臂**
        # （「筛过的证据 vs 全量错误日志」到今天还没有我方实测数字）。
        # ⚠️ 收益**未实测** ⇒ 不许声称。差集口径（不是「关掉 == 接线前」）：两臂
        # prompt 砍掉证据那一段后除空白外逐字相同，其余段（契约短哈希、误差形态、
        # 动作信封协议）两臂**同值** ⇒ 不进差集。
        structured_evidence: bool = True,
        # §22.2 的**阶段级**执行上限（编译一档、运行一档）。默认 `None` ⇒ 各自由
        # `exec_guard` 那一档的默认值给（编译与运行**不是**同一个默认）。
        # 传递链是「调用方 → Conductor → 后端 → `exec_guard`」，只走显式构造参数：
        # 不接配置文件、不接环境变量、不接命令行（§3.5 的「不允许配置」清单）。
        # ⚠️ 只在**本类自己建后端**（`backend is None`）时生效：注入后端时策略属于那个
        # 后端自己的构造参数，这里再塞一份就会出现「两个地方都能改同一个数」。
        execution_limits: Optional["Limits"] = None,
        compile_limits: Optional["Limits"] = None,
    ):
        # `is None` 而不是 `or`: `FailureDatabase` 定义了 `__len__`, 于是**一条记录都没有的
        # 注入实例是假的**, `failure_db or FailureDatabase()` 会把它丢掉换成一个默认实例 ——
        # 调用方拿不到自己注入的那个库, 而失败记录被写进了默认路径。真机探针 (doc 49 §7.17)
        # 就是这么把 7 条探针记录写进 `output/failure_db.jsonl` 的: 它给的是临时目录。
        self.backend = backend if backend is not None else MockGPU(
            exec_limits=execution_limits, compile_limits=compile_limits)
        self.kernel_db = kernel_db if kernel_db is not None else KernelDatabase()
        self.failure_db = failure_db if failure_db is not None else FailureDatabase()
        self.llm = llm
        self.pm = prompt_manager or PromptManager()
        self.store = ArtifactStore()
        self.router = Router()
        self.max_optimize_rounds = max_optimize_rounds
        #: §10.2 的预算计划 (不含账本 —— 账本每次运行新建, 见 `_execute_inner`)。
        self.budget_plan = budget if budget is not None else BudgetPlan()
        #: 本次运行开始时账本的长度, 每次进 `execute` 重新取。
        #: 写侧用它把口径从「这个进程到现在」换成「这一次」。
        self._cost_watermark = 0
        #: 本次运行里「本级判不出」的轮数 —— 那些轮既不入库也不产 Reflexion,
        #: 计数只在日志里活一次 (见 `_record_failure` 与结尾那行 `[FailureDB]`)。
        #: 不打印就等于没生效: 门挡掉的东西必须能被读者看见 (缺口 2 的失效方式)。
        self._reference_hold_rounds = 0

        # 标定心跳的状态 (每次 `execute` 重置 —— 「每 N 次真值测量」的单位是**一次
        # 运行**: 一个 Conductor 实例可以跑多个任务, 计数不许跨任务累计)。
        # 非法 interval 在这里就响: `heartbeat_due` 会抛 ValueError, 与其在循环深处
        # 炸掉一次真机运行, 不如构造时就拒绝。
        if not isinstance(heartbeat_interval, int) or isinstance(heartbeat_interval, bool) \
                or heartbeat_interval <= 0:
            raise ValueError(f"heartbeat_interval 必须为正整数: {heartbeat_interval!r}")
        if heartbeat_interval > MAX_HEARTBEAT_INTERVAL and not harness_controls_enabled():
            raise ValueError(
                f"heartbeat_interval 不能超过 {MAX_HEARTBEAT_INTERVAL}: "
                f"{heartbeat_interval!r}（仅 harness 可显式放宽）")
        self.heartbeat_interval = heartbeat_interval
        if (type(cuda_calibration_identity_hash) is not str
                or (cuda_calibration_identity_hash and
                    (len(cuda_calibration_identity_hash) != 64 or
                     any(c not in "0123456789abcdef" for c in cuda_calibration_identity_hash)))):
            raise ValueError("cuda_calibration_identity_hash must be an exact SHA256")
        self.cuda_calibration_identity_hash = cuda_calibration_identity_hash
        self.cuda_ruler_profile = None
        if cuda_ruler_profile_path is not None:
            from .cuda_ruler_profile import read_profile
            self.cuda_ruler_profile = read_profile(cuda_ruler_profile_path)
            if heartbeat_interval != 1:
                raise ValueError("calibrated CUDA ruler requires heartbeat_interval=1")
            identity = self.cuda_ruler_profile["identity_hash"]
            if cuda_calibration_identity_hash and cuda_calibration_identity_hash != identity:
                raise ValueError("CUDA ruler/calibration identity mismatch")
            self.cuda_calibration_identity_hash = identity
        #: §10.3 的 ε。抽签值本身是 (seed, 轮次) 的纯函数, 不用随机数发生器 —— 于是
        #: ε-greedy 与「跑完能离线复算」不冲突 (见 `selection` 模块注释)。
        self.selection_epsilon = float(selection_epsilon)
        #: §21 的动作入口。能力取自**后端自己声明的**那一份（不是猜的），授权根只有
        #: §22.1 的 `candidate/` 一个 —— 「候选只能写候选」这条权限今天就有判据面。
        self.run_root = str(run_root or "")
        #: §22.1 的布局句柄（`execute` 里建，跨任务不保留 —— 布局是「这一次运行」的单位）。
        self.layout: Optional[RunLayout] = None
        self.structured_evidence = bool(structured_evidence)
        #: §21.1 **生产者侧**的那一格: 修复轮里模型交上来的动作信封, 由下一轮的判据点消费
        #: （就在 `current_code = pending_fix` 那两行旁边）。它与 `pending_fix` 同生同灭 ——
        #: 单独一个变量会留下「谁清它」这个问句, 而本仓 I77/I78 两个缺陷的形状正是
        #: 「一个写进去、没人负责清的留痕」。
        self._pending_action_envelope: Optional[ActionEnvelope] = None
        #: 上一次**过了 §21 门**的候选源码 —— 「新增依赖」的基线 (I78)。
        #: ⚠️ 被拒的候选**不**更新它: 它从没进过流程, 拿它当基线就等于把同一份违规
        #: 候选的第二次提交放行 (真机实测过一次, 见 ActionContext.dep_baseline_source)。
        self._admitted_baseline: str = ""
        self.agent_runtime = agent_runtime if agent_runtime is not None else AgentRuntime(
            capabilities=self.backend.capabilities,
            authorized_roots=(candidate_path(self.run_root).rsplit("/", 1)[0],),
        )
        #: 本次运行里**真值测量**(L3 读数)的累计条数 —— 心跳按它的倍数触发。
        self._true_measurements = 0
        #: §10.2 的账本。构造时就建一个 (与 `_true_measurements` 同一个理由: 只走
        #: 收口点那条路之外的调用 —— 比如直接调心跳 —— 不许因为没有账本而炸),
        #: 每次 `execute` 再新建一个, 使「这次运行花了多少」不跨任务累计。
        self._budget = BudgetLedger(self.budget_plan)
        self._run_started = time.monotonic()
        # §10.3 升级门的运行态: 判决词尾巴 (只进 valid 轮) 与逐轮裁决日志
        self._verdict_tail = []
        self._escalation_log = []
        self._heartbeat_epoch = 0
        #: 已经为「第几次测量」跑过心跳了。防的是这一支: 计数停在第 5 次而 L3 一直没
        #: 跑成 (L1/L2 挂了), 于是每一轮都重跑一次心跳。到位只跑一次, 之后的轮次复用
        #: 同一次心跳 (同一扇窗)。
        self._heartbeat_ran_for = 0
        #: 最近一次心跳。只对它自己的 window 作数。
        self._last_heartbeat = None

        #: 永久结论**必须**量两遍 (设计 §10.5-4 / §3.5 的不变量, 不是开关)。
        #: 生产者是 `_produce_remeasure`, 消费者是入库判据 (`oracle.admit_to_corpus`
        #: 的两次确认那道门, 生产路径恒传 `requires_remeasure=True`)。两者成对存在, 所以
        #: 「一律要求」不再是「一律拒」: 每一份要进语料的结论都会**先**补一次独立重测。
        #: 曾经这里是一个默认 `False` 的构造参数 ⇒ 默认配置下单次读数即可入库 (I79)。

        # 初始化 Agent (注入 LLM + PromptManager)
        self.designer = DesignerAgent(self.kernel_db, llm=llm, prompt_manager=self.pm)
        # CoderPipeline 默认启用 (P1-1): Plan→并行采样→剪枝→修复→回退
        coder_pipeline = None
        if use_coder_pipeline and llm is not None:
            from .coder_pipeline import CoderPipeline
            coder_pipeline = CoderPipeline(
                llm=llm, kernel_db=self.kernel_db, prompt_manager=self.pm,
            )
        # I75-G1: 留引用给 execute() —— hw_info 到手后把 smem 上界注入
        # compile-gate (不可行 tile 本地剪掉, 不烧 SSH)。None = 单体路径/假件。
        self._coder_pipeline = coder_pipeline
        self.coder = CoderAgent(self.kernel_db, llm=llm, prompt_manager=self.pm,
                                pipeline=coder_pipeline)
        self.verifier = Verifier(self.backend, cuda_ruler_profile=self.cuda_ruler_profile)
        self.search = SearchAgent(llm=llm, prompt_manager=self.pm)
        self.fixer = FixerAgent(llm=llm, prompt_manager=self.pm, failure_db=self.failure_db)
        # 优化配方选择器 (确定性规则，不依赖 LLM)
        self.planner = OptimizationPlanner()
        # 本次运行的环境指纹与漂移比较结果 (由 `_execute_inner` 填)。初值给空指纹:
        # `_flush_snapshot` 也可以在 `execute()` 之外被调到 (单测直接调它), 不该 AttributeError。
        self._environment: dict = {"fingerprint": {}, "drift": None}
        self._run_binding = None
        self._run_binding_dir = None
        self._resume_candidate_once = False

    #: 循环出口的词汇表。**每个词都要有写者** —— 多一个没写者的词就是「算了没人读」的副本,
    #: `tests/test_stop_reason.py` 按差集钉这条。
    STOP_REASONS = ("passed", "stagnant", "converged", "fix_error", "l1_l2_failed",
                    "search_error", "rounds_exhausted", "infra_retry", "transport_unknown_hold", "reference_identity_hold",
                    # §10.2 的两条预算出口。与 `rounds_exhausted` 分开数: 那个是
                    # 「轮次练够了」, 这两个是「钱花完了」—— 复盘时一个要调轮次上限,
                    # 一个要调预算或缩小任务。
                    "budget_exhausted", "wall_clock_exhausted",
                    # §12/Stage 3: 模板型后端的参数空间里**一个点都建不出来** (实例化器
                    # 拒了全部点)。与 `search_error` 分开: 那个是搜索本身抛异常, 这个
                    # 是「候选源枯了」—— 复盘时一个看代码, 一个看实例化器的报错。
                    "template_no_candidate",
                    # §21.1 生产者侧: 模型连续两次交的都是「不改候选」的动作信封
                    # (解释 / 请求观测), 响应过之后仍然没有补丁 ⇒ 本轮没有候选可验证。
                    # 与 `fix_error` 分开: 那个是修复**抛了异常**, 这个是模型**明说不改**。
                    "ask_turns_exhausted")

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

    @staticmethod
    def _denied_escalation():
        """没有证据时的裁决: **拒绝**（三条件一条都不成立）。

        不用 `None` 表示「没判过」: 下游要的是「这一轮能不能升级」这个答案,
        而 `None` 会逼每个读者各自决定怎么解释它 (必然有人解释成放行)。
        """
        return decide(EscalationInputs(k_valid_rounds=MAX_STAGNANT_ROUNDS))

    def _escalation_decision(self, plan, applied_recipes, disabled_recipes,
                             metrics, current_code, task=None):
        """§10.3 三条件的输入装配 (判据本体在 `operator_agent/escalation.py`)

        三条输入各自的出处:
          * 条件 1: `self._verdict_tail` —— 由 `verdict()` 那一个点写入, 只含 valid 轮;
          * 条件 2: `smem_bound_evidence` —— 今天唯一有量具的硬限制。估的**不是 sketch 的
            tiling 而是那一轮被测量的代码**: 走 `check_resource_feasibility` (与 L1 预检
            **同一个估算器**、同一份产物), 否则「证据」与「预检」会是两套算法, 而 sketch
            与 LLM 真正出的码今天本来就允许不一致;
          * 条件 3: planner 选中 ∩ 未 applied/未 disabled 的配方 = 还剩的结构方向。
        """
        caps_kb = metrics.get("shared_mem_per_sm_kb") or 0
        limits = ()
        if caps_kb and current_code:
            # `check_resource_feasibility` 只读 caps 的这一个字段 ⇒ shim 就够,
            # 不必为了归因再去解析一份 HardwareCaps (那会引入第二个真源)。
            class _CapsKB:
                shared_mem_per_sm_kb = int(caps_kb)

            rep = check_resource_feasibility(
                current_code, _CapsKB(),
                getattr(getattr(task, "source", None), "kernel_type", "") or "")
            limits = smem_bound_evidence(rep.estimated_smem_bytes, rep.limit_bytes)
            # §10.3 条件 2 的第二把量具 (占用率上限)。输入全部取自**已测指标**,
            # 缺任一 ⇒ 该条不产出 (三态), 绝不拿默认值假装测过。
            limits = limits + occupancy_bound_evidence(
                rep.estimated_smem_bytes, metrics.get("num_warps"),
                shared_mem_per_sm_kb=caps_kb,
                max_threads_per_sm=metrics.get("max_threads_per_sm"))
        dirs = untried_directions(
            [r.id for r in (plan.selected_recipes if plan else [])],
            sorted(applied_recipes), sorted(disabled_recipes))
        return decide(EscalationInputs(
            verdicts=tuple(self._verdict_tail), k_valid_rounds=MAX_STAGNANT_ROUNDS,
            hard_limits=limits, untried_directions=dirs))

    # ─── §21 Agent Runtime：候选进编译之前的那一次裁决 ──────────────

    @staticmethod
    def _round_action_kind(round_action: str, route) -> str:
        """本轮的动作种类（§21.1 的词汇）× 本轮在做什么（Conductor 的 `action` 词）

        | 本轮的 `action` | 动作种类 | 依据 |
        |---|---|---|
        | `generate` / `optimize` | `propose_source_patch` | LLM 出的是一份实现（首轮是提议，之后是搜索驱动的重生成） |
        | `template` | `propose_template` | 源码来自实例化器，不经 LLM（§12） |
        | `fix` | `repair_correctness` / `repair_compile` | §8 的路由判的是「数值错还是编译错」 |

        `fix` 而没有路由（或路由说不清是数值还是编译）时按 `repair_compile` 记：
        §8 的五路里只有这两路会走到 Fixer，而「局部修复」这一侧两只都符合；
        写成第三种词会造出一个 §21.1 词汇表里没有的种类。
        """
        if round_action == "template":
            return KIND_PROPOSE_TEMPLATE
        if round_action == "fix":
            if route is not None and getattr(route, "kind", "") == "numeric":
                return KIND_REPAIR_CORRECTNESS
            return KIND_REPAIR_COMPILE
        return KIND_PROPOSE_SOURCE_PATCH

    @staticmethod
    def _candidate_ext(task: TaskSchema) -> str:
        """候选文件的后缀（`candidate/code.<ext>` 那一格）。

        只有一处：§21 的 `Patch.path`（`candidate_path(run_root, ext)`）与 §22.1 落盘时写的
        相对路径必须指向**同一个文件**，两处各写一份映射迟早对不上。今天三种 DSL 都是 `py`
        （`cuda_c` 也走 Python 入口，见 §12 的模板后端），表外的值给 `py`。
        """
        return {"triton": "py", "cuda_c": "py", "python": "py"}.get(
            (task.target.dsl or "python").lower(), "py")

    def _materialize_candidate(self, code: str, task: TaskSchema) -> Optional[str]:
        """把这一轮**过门**的候选写进 §22.1 的 `candidate/` 那一格（唯一写入口）

        WHY（2026-10-06，切片 C）: `run_root` 有生产调用点之后，§21 的 `Patch.path` 变成
        **真实绝对路径** —— 但那个路径上如果没有文件，§22.1 的权限位保护的就是零个产物，
        `write_for` 也仍然零生产调用者（I82 的边界说的正是这一格：「声明 + 可运行守卫」）。
        这一步把「判据指着的路径」与「盘上真实存在的产物」接起来。

        三条刻意划下的线:
        * **只写候选源码这一件**：`report` / `measurements` / `kernel_db` 那些既有产物今天
          **不搬**（搬它们要连「谁生成/清理根」一起定，见状态文档 §22 行）；
        * **只有过门的候选才落盘**：被 §21 拒掉的候选一个字符都不留（与「零远端载荷」同一条
          纪律，见 I78）；
        * **没有 `run_root` 就什么都不做**：那是今天的另一档形态（判据活在「声明」那一档），
          不是「默认建一个相对 CWD 的布局」。

        每轮的源码哈希与 patch 都在 `iteration_history` 里，所以同一个路径**每轮覆盖**是
        被接受的语义（这个文件回答的是「这一轮的候选在哪」，不是历史版本库）。

        Returns:
            写入的绝对路径；没有布局时返回 `None`。
        """
        if not self.run_root or self.layout is None:
            return None
        rel = candidate_path("", self._candidate_ext(task))   # 角色路径 = 布局内相对路径
        return self.layout.write_for(ROLE_CANDIDATE, rel, code)

    def _round_agent_action(self, *, round_num: int, round_action: str, route,
                            current_code: str, parent_code: str, task: TaskSchema,
                            error_shape: Optional[str],
                            envelope: Optional[ActionEnvelope] = None) -> AgentAction:
        """把这一轮的产物包成 §21.1 的结构化动作（`Patch` 的审计与回滚点在这里成型）

        **动作是谁写的**（§21.1 的生产者侧）：`envelope` 是模型自己交上来的意图时，
        种类 / 理由 / 预期瓶颈 / 想用的工具**取模型写的**；两样东西仍然由 harness 写 ——
        `patch.path`（§22.1 的布局，唯一产者 `candidate_path()`）与 `token_budget`
        （判它的数必须与真正花掉的那个上限同源；交给模型填就是让被判的人填判据里的数，
        而它只要写 1 就能绕过 `budget_problems`）。

        `rationale` 不再无条件由 harness 拼装（只在**交不出信封**时拼）—— 这是刻意的：
        §21.2「`repair_correctness` 必须带误差形态」原先靠 harness 拼装成为**构造保证**，
        于是那条判据永远不会红。现在模型写的理由要自己点名形态，不点名就被拒（零 GPU 秒），
        而拒绝理由会作为下一轮的错误信息回到模型眼前 —— 判据第一次遇到一个真正不可靠的
        生产者。harness 那几条事实（轮次 / 路由 / 误差形态 / ±行数）**没有**混进模型的
        理由里（混进去上面那条判据又变成永不失败）：它们本来就各自在血缘里
        （`round` / `action` / `agent_action.context` / `patch.added_lines`）。
        """
        kind = self._round_action_kind(round_action, route)
        source = ACTION_SOURCE_HARNESS
        rationale = ""
        bottleneck: Optional[str] = None
        tools: Tuple[str, ...] = ()
        if envelope is not None and envelope.usable:
            source = ACTION_SOURCE_LLM
            kind = envelope.kind
            rationale = envelope.rationale
            bottleneck = envelope.expected_bottleneck
            tools = envelope.requested_tools
        ext = self._candidate_ext(task)
        patch: Optional[Patch] = Patch(path=candidate_path(self.run_root, ext),
                                       new_source=current_code,
                                       parent_source=parent_code or "")
        if source == ACTION_SOURCE_LLM and kind not in PATCH_KINDS:
            # 模型说它不是「改候选」的那一类 ⇒ harness **不替它**造一个 patch。
            # 造了就是 harness 自己编语义（「借解释之名改代码」正是 §21.1 要挡的形状，
            # 而那一半由 `schema_problems` 的 `SIDE_EFFECT_FREE_KINDS` 判）。
            patch = None
        bits = [f"第 {round_num} 轮 {round_action}"]
        if route is not None:
            bits.append(f"路由 kind={getattr(route, 'kind', '')} route={getattr(route, 'route', '')}")
        if error_shape:
            bits.append(f"误差形态={error_shape}")
        if patch is not None and patch.has_parent:
            added, removed = patch.changed_lines()
            bits.append(f"相对父代 +{added}/-{removed} 行")
        return AgentAction(
            action_id=f"r{round_num}-{round_action}",
            kind=kind,
            # 模型没写理由 ⇒ 退回 harness 拼装（接线前的行为）。写了就用它的：
            # 于是「理由是谁写的」在产物里分得开（`action_envelope` 那一格）。
            rationale=(rationale or "; ".join(bits)),
            # 父代的**身份**取它的内容哈希, 不取任务 id: 判据与回滚点必须指向同一份源码
            # (`Patch.rollback_source()`), 而任务 id 对「改的是哪一份」一个字都没说。
            parent_candidate=(
                f"src:{patch.parent_hash()}"
                if patch is not None and patch.has_parent else None),
            patch=patch,
            expected_bottleneck=bottleneck,
            requested_tools=tools,
            token_budget=self._llm_tokens_budget_claim(round_action),
        )

    @staticmethod
    def _llm_tokens_budget_claim(round_action: str) -> int:
        """本动作**声明**要花多少 token（§21.1 的 `token_budget`）

        取的是生成侧真正用的那个上限（`CoderPipeline.GEN_MAX_TOKENS` / Fixer 的
        `max_tokens`），不是拍一个数：预算校验的两侧必须同源，否则「声明 1000、实际花
        8192」会让这道门看起来永远能过。模板轮不花 token ⇒ 1（正整数下限）。
        """
        from .coder_pipeline import CoderPipeline
        if round_action == "template":
            return 1
        if round_action == "generate" or round_action == "optimize":
            return int(CoderPipeline.GEN_MAX_TOKENS)
        return int(FixerAgent.FIX_MAX_TOKENS)

    def _answer_ask_and_retry(self, outcome: "FixOutcome", *, repair_base: str,
                              task: TaskSchema, failed_results, route,
                              evidence: str, record: dict) -> "FixOutcome":
        """响应一次「不改候选」的动作，并**有界**地再要一次补丁（§21.1 生产者侧）

        三类询问各有各的答法，但只有两种结局 —— **答得出**与**答不出**，而且必须说清
        是哪一种（答不出的那两类要原样告诉模型，否则它会以为「观测结果是空」）：

        * `explain_failure`：答得出（它不需要任何工具）。模型的话记进本轮失败记录的
          `explanation`，然后要补丁。
        * `request_profiler` / `request_new_shape`：今天**答不出** —— 判据来自
          `agent_runtime.ask_unavailable_reason()`，与 §21.4 第 7 条读**同一个**注册表
          （那两档工具在 `TOOLS` 里是 `None`）。原因原样进 prompt。
        * 上界：第二次**还是**提问 ⇒ 返回 `code` 为空的结果，调用方据此停下并记
          `ask_turns_exhausted`。不设这个上界，一条「模型只会提问」的环会白烧 token；
          而设成「无限次」等于把 §21.1 的预算账本交给模型来花。

        `ask_answer` 走**声明过的**模板段（`fixer_repair.md` 的 `{{ask_answer}}`），
        不是拼进 §21.3 那八段证据里 —— 后者是「只有那八项」的那个拼装点，把答复塞进去
        就是第二个注入通道，而且两臂对照也会被污染。
        """
        env = outcome.envelope
        reason = ask_unavailable_reason(env.kind)
        if env.kind == KIND_EXPLAIN_FAILURE:
            answer = ("你的解释已记入本轮失败记录。现在请给出**补丁**: 完整修复代码，"
                      "必要时在代码块之前带一行 `根因:`。")
            record["ask_answer_kind"] = "accepted_explanation"
            if env.rationale:
                record["explanation"] = env.rationale
        elif reason:
            answer = (f"这条请求**答不出来**: {reason}。请在不依赖它的前提下给出补丁；"
                      f"如果确实改不动，改用 `explain_failure` 说清卡在哪。")
            record["ask_answer_kind"] = "refused_no_gauge"
            record["ask_refusal"] = reason
        else:
            # 工具在注册表里有实现物、但今天没有任何一条路把它**跑起来**再把事实喂回去
            # （producer 不在岗）—— 所以这里也不能说「观测结果是空」。
            answer = ("已收到你的请求；本轮能给你的只有已有的那段证据。现在请给出**补丁**。")
            record["ask_answer_kind"] = "accepted_without_running_it"
        record["ask_field"] = env.kind
        record["ask_turns"] = 1
        print(f"  [Conductor] 响应动作信封 {env.kind} → {record['ask_answer_kind']}")
        second = self.fixer.run_with_action(
            repair_base, task, failed_results,
            fix_not_rewrite=route.fix_not_rewrite,
            structured_evidence=evidence,
            error_shape=(getattr(route, "error_shape", "") or ""),
            ask_answer=answer,
        )
        record["second_envelope"] = second.envelope.as_dict()
        if second.asked:
            record["ask_turns"] = 2
            record["ask_outcome"] = "ask_turns_exhausted"
        else:
            record["ask_outcome"] = "patch_after_ask"
        return second

    def _round_action_context(self, *, route, current_code: str, test_inputs,
                              task: TaskSchema, verified_parent,
                              previous_results=(), spend_committed: bool = False,
                              dep_baseline: str = "") -> ActionContext:
        """§21 裁决的输入装配。**每个字段都有出处**，缺就是「这一条判不了」

        * `escalation_allowed` 取**最近一次**门裁决（`self._escalation_log[-1]`）：
          门是在每一轮的判决点算的，而它放行的是**下一轮**的结构改动 —— 读「本轮算的
          那一份」等于用还没发生的事给自己发许可证。
        * `abi_broken` 读上一轮 L1 发布的 `compile_status`（`interface_mismatch` 是
          `verifier._verify_compile` 那条出口写的字），不重算判据。
        * `input_key_sets` 走 `interface_contract.input_key_sets`：逐组判，不取并集。
        * `tokens_left` 与 §10.2 的退化出口**同一个口径**（本次运行的用量）——
          两处各算一遍会让「还剩多少」有两个答案。
        """
        from .interface_contract import input_key_sets
        last_esc = self._escalation_log[-1] if self._escalation_log else None
        abi_broken = False
        for r in (previous_results or ()):
            metrics = getattr(r, "metrics", None) or {}
            if str(metrics.get("compile_status") or "") == "interface_mismatch" \
                    and not getattr(r, "passed", True):
                abi_broken = True
        key_sets = tuple(tuple(sorted(str(k) for k in keys))
                         for keys in (input_key_sets(test_inputs) or []) if keys)
        used = self._llm_tokens_used()
        return ActionContext(
            route_kind=(getattr(route, "kind", "") if route is not None else ""),
            error_shape=(getattr(route, "error_shape", None) if route is not None else None),
            fix_not_rewrite=bool(getattr(route, "fix_not_rewrite", False)) if route else False,
            escalation_allowed=bool(last_esc and last_esc.get("allowed")),
            escalation_refusals=tuple(last_esc.get("refusals", ())) if last_esc else (),
            abi_broken=abi_broken,
            authorized_roots=self.agent_runtime.authorized_roots,
            input_key_sets=key_sets,
            language=task.target.dsl or "",
            parent_verified=bool(verified_parent),
            tokens_left=self._budget.tokens_left(used),
            spend_committed=bool(spend_committed),
            # 「新增依赖」的基线 = 上一次**过门**的候选 (I78)。空 = 用 patch 的父代。
            dep_baseline_source=str(dep_baseline or ""),
        )

    def _fix_evidence(self, *, task, route, verify_results, current_code,
                      parent_code, gate_ctx):
        """§21.3 的生产者: 把这一轮**手里已有的结构化事实**渲染成修复提示里那一段

        为什么由这里拼、而不是 Fixer 自己拼: `AgentRuntime.context_for(...).render()`
        是唯一的拼装点 (`ActionContext` 的注释里写着同一条纪律)。本方法只负责**采集**
        八个输入, 不做措辞 —— 措辞与上限都在 `ProposalContext` 那一侧, 于是「模型看到
        的证据」与「判据看到的证据」不会各写一套。

        三件**刻意不采**的东西 (采了就是自欺):
          * profiler 指标只取 `verifier.COMPILE_FACT_KEYS`（编译产物的事实, 与 L3 写进
            metrics 的是同一份名单）。本轮没有这些键就传空 dict ⇒ 上下文里那一段会说
            「本轮没有编译产物事实」, 而不是拿错误计数冒充 profiler 读数;
          * 失败分类/误差形态取**同一个** `route` 对象（gate 那一侧也用它）—— 再分类一次
            就是第三个口径;
          * 预算取 gate 那一份 `tokens_left`（同一个 `_llm_tokens_used`）。
        """
        from .agent_runtime import contract_digest
        from .contract import freeze_from_task

        # 父候选 diff 的基线是**本轮的上一份**(`_code_before_this_round`), 不是
        # 「正在被修的那一份」: 后者与 `current_code` 常常同一份 ⇒ diff 恒空, 而
        # 「这一轮的改动干了什么」恰好是修复时最该看到的证据。首轮没有上一份 ⇒ 空串,
        # `Patch` 会说「没有父代」而不是拿空文件冒充。
        patch = Patch(candidate_path(self.run_root), current_code, parent_code or "")
        contract = freeze_from_task(task) if task is not None else None
        first_error = ""
        for r in (verify_results or ()):
            if not getattr(r, "passed", True):
                first_error = str(getattr(r, "error_log", "") or
                                  getattr(r, "message", "") or "")
                break
        facts = {}
        for r in (verify_results or ()):
            for k in COMPILE_FACT_KEYS:
                v = (getattr(r, "metrics", None) or {}).get(k)
                if v is not None and not isinstance(v, bool):
                    facts[k] = v
        ctx = self.agent_runtime.context_for(
            contract_summary=contract_digest(contract),
            patch=patch,
            route=route,
            first_error=first_error,
            profiler_metrics=facts,
            budget_left=(gate_ctx.tokens_left if gate_ctx is not None else None),
        )
        record = {"injected": True,
                  "chars": len(ctx.render()),
                  "truncations": list(ctx.truncations()),
                  "profiler_facts": sorted(facts),
                  "has_parent_diff": bool(patch.has_parent),
                  # diff 的规模 (新增/删除行数): 0 与「有父代但一字未改」在当时
                  # 长得一样, 而这两种情形对「这一轮的改动干了什么」是相反的结论。
                  "parent_diff_lines": list(patch.changed_lines()),
                  "has_contract": bool(contract)}
        return ctx.render(), record

    @staticmethod
    def _refusal_result(verdict) -> VerificationResult:
        """被拒的动作 → 一条 **L1 失败**结果（不进编译、零 GPU 秒）

        为什么是「合成一条 L1 失败」而不是另加一条控制流：本轮的候选确实没通过
        「进编译之前的检查」，而下游（§8 路由 → Fixer → 负样本）对「候选不通过」只有
        一条已存在的通道。合成本地失败结果 = 复用那条通道，并让拒绝理由**逐字**进
        Fixer 的处方（Wave 3 的退出标准「编译失败和正确性失败都有结构化修复路径」）。
        `metrics` 上的两个键是给读者用的：`agent_action_refused` 让
        「这是被 §21 拦下的、根本没编译」与「远端编译真的失败了」分得开，
        也用于跳过下面那条根本没意义的依赖分析（省一次 SSH 往返）。
        """
        note = verdict.refusal_note()
        return VerificationResult(
            level=VerificationLevel.L1_COMPILE, passed=False,
            message="动作被拒 (未进编译): " + (note.split(";")[0][:200] or "无理由"),
            error_log=("§21 动作裁决未通过, 这份候选**没有**被送去编译。\n"
                       f"动作: {verdict.action.kind} ({verdict.action.action_id})\n"
                       f"理由: {note}\n"
                       "修法: 让下一份候选不再命中上面任何一条 —— 这些检查全部是"
                       "零 GPU 时间的静态判据, 不修就会被同样拦下。"),
            metrics={"compile_status": "agent_action_refused",
                     "agent_action_refused": True,
                     "agent_action": verdict.as_dict()},
        )

    def _llm_tokens_used(self) -> int:
        """**本次运行**的 LLM token 用量。两个读者共用这一个口径（不是两次实现）：
        §10.2 的退化出口（`note_llm_usage`）与 §21.1 的预算校验。"""
        if self.llm is None or not hasattr(self.llm, "cost_tracker"):
            return 0
        return int(self.llm.cost_tracker.summary_since(self._cost_watermark)
                   .get("total_tokens", 0) or 0)

    def _heartbeat_for_next_measurement(self):
        """下一次真值测量是否该伴随标定心跳 —— 该跑就**现在**跑掉 (同一个窗口)。

        返回 `(HeartbeatResult|None, expected)`。`expected=True` 而结果为 None 表示
        「到期了但没跑成」, 由收口点按 RULER_BROKEN 处理 —— 不在这里吞掉, 否则
        「忘了跑」就是一条静默放行通道 (spine-review P1-5)。

        为什么在 L1/L2 之前跑: 心跳必须落在**与读数同一个窗口**里, 而窗口由这一轮的
        测量流决定。等知道 L3 会不会跑就晚了 (收口点那时已经算完)。代价是 L1/L2 挂掉
        的那一轮白跑一次心跳 —— `_heartbeat_ran_for` 保证「第 N 次测量」只白跑一次,
        不会因为计数不前进而每轮重跑。
        """
        from .calibration import (backend_bench_fn, heartbeat as run_heartbeat,
                                  heartbeat_due)
        nxt = self._true_measurements + 1
        if not heartbeat_due(nxt, self.heartbeat_interval):
            return None, False
        if self._heartbeat_ran_for == nxt:
            # 已经为这一次测量跑过: 复用那一次 (同窗), 不重跑也不换窗。
            return self._last_heartbeat, True
        self._heartbeat_epoch += 1
        epoch = self._heartbeat_epoch
        window = f"hb-{epoch}"
        self._heartbeat_ran_for = nxt
        try:
            # CUDA 测量不能被远端 CPU 小 matmul 心跳认证。缺原生能力时失败关闭。
            if (self.cuda_ruler_profile is not None or
                    (self._environment or {}).get("fingerprint", {}).get("device") == "cuda"):
                from .cuda_calibration import native_heartbeat
                result = native_heartbeat(self.backend, epoch=epoch, window_id=window,
                    expected_identity_hash=self.cuda_calibration_identity_hash,
                    ruler_profile=self.cuda_ruler_profile)
            else:
                result = run_heartbeat(
                    backend_bench_fn(self.backend), epoch=epoch, window_id=window,
                )
        except Exception as exc:  # noqa: BLE001 — 心跳炸了不挡主流程, 但必须留痕
            # 清掉上一次的结果: 否则「没跑成」之后的轮次会复用**上一扇窗**的心跳,
            # 而它恰好是 ok=True —— 那就把「心跳没跑成」洗成了「尺子是好的」。
            self._last_heartbeat = None
            print(f"[Heartbeat] 第 {nxt} 次测量到期, 心跳没跑成: "
                  f"{type(exc).__name__}: {exc} —— 本次按 RULER_BROKEN 处理")
            return None, True
        self._last_heartbeat = result
        # §10.2 的 5% 预检标定: 心跳就是那个「预检」, 花掉的是一次机时预算。
        # **只记账、不被拦** —— 让预算饿死标定, 等于把「尺子有没有坏」变成一件可以因为
        # 缺钱而跳过的事 (§10.5-5 的同族病)。超支由 `calibration_over_plan` 留痕。
        self._budget.note_calibration_run()
        verdict = "OK" if result.ok else "RULER_BROKEN"
        print(f"[Heartbeat] 第 {nxt} 次测量前标定 ({window}): {verdict} "
              f"好核比值={result.known_good_ratio:.3g} "
              f"慢核比值={result.known_slow_ratio:.3g}"
              + (f" {list(result.reasons)}" if result.reasons else ""))
        return result, True

    def _select_finalist(self, current_code: str, finalists: list,
                         task, input_sets, expected_output) -> tuple:
        """P1-1 (generation-side-review): 决赛候选逐个跑 L1+L2, 用真机正确性定胜负

        选择权在验证侧: compile-gate 只决定谁能进决赛 (≤ CoderPipeline.L2_FINALISTS)。
        规则:
          - 按 temperature 序 (0.1→0.4) 逐个 `verify(stop_after=L2)`, 首个全过者胜出;
          - 胜者 != 默认候选时换码 (低温候选贴 plan 但可能错, 高温候选救场);
          - 全败则保持默认候选 —— 选择不是门: 「没有候选过 L2」是 Fixer 的输入,
            不是任务失败。

        返回 `(选中的代码, 已跑完的 L1+L2 结果 或 None, 结果是否已是最终态, 真机 L2 次数)`:
          - 胜出候选通过 L1+L2 → `(胜者, [l1, l2], False, n)`, 调用方只补 L3
            (真机 L2 不重复; L1 也不必复检 —— 它已经在选择循环里跑过);
          - 决赛全败 → `(默认候选, 默认候选的 [l1, l2], True, n)`, 调用方**直接采用**:
            L2 已经失败, 本来就不该跑 L3; 而这份失败结果正是喂 Fixer 的证据,
            上一版丢掉它、让调用方重跑一遍 L1+L2 —— 全败轮白烧一次真机 L2;
          - 名单里找不到默认候选 (自定义 coder 替身) → `(默认候选, None, False, n)`,
            退回完整三级验证的老路径, 不复用来源不明的结果。
        `n` 是选择循环里**真实执行过的 L2 次数** (L1 就挂的候选不计), 由调用方写进
        本轮血缘 —— P1-1 翻转把每轮生成从 1 个候选变成最多 2 个决赛候选, 「机时涨了
        多少」这个问句的原料只能在这里落下。
        """
        from .verifier import VerificationLevel
        default_results = None
        l2_runs = 0
        for i, cand in enumerate(finalists):
            if self._run_binding is not None:
                self._check_run_binding(task, input_sets)
            sel = self.verifier.verify(
                cand, task=task, test_inputs=input_sets,
                expected_output=expected_output,
                stop_after=VerificationLevel.L2_FUNCTION,
            )
            if self._run_binding is not None:
                self._check_run_binding(task, input_sets)
            l2_runs += sum(1 for r in sel
                           if r.level is VerificationLevel.L2_FUNCTION)
            if any(not r.passed and (r.metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN
                                    or r.metrics.get("reference_identity_unknown")) for r in sel):
                # The in-flight candidate is unknown, not a loser. Do not try
                # another finalist / L3 on the same possibly active runner.
                return cand, sel, True, l2_runs
            if cand == current_code:
                # 默认候选的结果留一份给「全败」出口复用 (就是喂 Fixer 的失败证据)
                default_results = sel
            if sel and all(r.passed for r in sel):
                if cand != current_code:
                    print(f"\n[Conductor] 候选选择 (P1-1): 决赛第 {i+1}/{len(finalists)} 名 "
                          f"首个通过 L1+L2, 取代默认候选")
                return cand, sel, False, l2_runs
        if default_results is not None:
            print(f"\n[Conductor] 候选选择 (P1-1): 决赛候选无一通过 L1+L2, "
                  f"保持默认候选 (复用它的 L1+L2 失败证据, 不重跑真机)")
            return current_code, default_results, True, l2_runs
        print(f"\n[Conductor] 候选选择 (P1-1): 决赛候选无一通过 L1+L2, "
              f"保持默认候选交完整验证")
        return current_code, None, False, l2_runs

    def execute(
        self,
        task: TaskSchema,
        test_inputs: Union[dict, list],
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
        # 与水位线同一个理由: 一个 Conductor 实例可以跑多个任务, 落盘与打印的必须是
        # **本次**那一份, 不是「这个进程到现在为止」。
        self._reference_hold_rounds = 0
        self._run_binding = None
        self._run_binding_dir = None
        self._resume_candidate_once = False
        self._resume_status = None
        self._environment = {"fingerprint": {}, "drift": None}
        # §10.2 的账本同理: 「这次运行花了多少次真值测量」跨任务累计会让第二个任务
        # 少花钱。起始时刻一起在这里取 —— 墙钟预算量的是**这一次运行**。
        self._budget = BudgetLedger(self.budget_plan)
        self._run_started = time.monotonic()
        # §10.3 升级门的运行态: 判决词尾巴 (只进 valid 轮) 与逐轮裁决日志
        self._verdict_tail = []
        self._escalation_log = []
        # §21 的记账同理 (与上面三条同一个理由): 「这一次运行拦下过几个动作」不许
        # 跨任务累计 —— 第二个任务的产物里出现第一个任务拦下的动作, 复盘时会去找一个
        # 不存在的现场。
        self.agent_runtime.reset_run()
        # 基线与记账同一条纪律: 单位是**一次运行** (同一个 Conductor 可以跑多个任务)
        self._admitted_baseline = ""
        # 心跳计数同理: 「每 N 次真值测量」是**这一次运行**内的事, 跨任务累计会让
        # 第二个任务在错误的位置插入参照对。
        self._true_measurements = 0
        self._heartbeat_epoch = 0
        self._heartbeat_ran_for = 0
        self._last_heartbeat = None
        # §22.1: 布局在**写任何 artifact 之前**建/核一次，单位与上面那一串运行态相同
        # （同一个 Conductor 跑多个任务时，上一轮的目录不许被下一轮接着用）。
        # `run_root` 为空时**不建**：空串不是「默认布局」，而是「这一档还没开」。
        self.layout = run_layout(self.run_root) if self.run_root.strip() else None
        with capture_execution(
            task_id=task.task_id,
            operator=operator_name,
            hardware=hardware_name,
        ) as self._log_path:
            try:
                return self._execute_inner(
                    task, test_inputs, expected_output, target_gflops,
                    resume_from=resume_from,
                )
            except run_contract.RunBindingError as exc:
                return self._run_identity_hold(task, str(exc))

    def _run_identity_hold(self, task, reason):
        """Fail closed without generation, Fixer, remeasure or corpus writes."""
        print(f"[Conductor] 冻结运行身份 unknown/HOLD: {reason}; 不从头重跑")
        result = VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False,
            message="冻结运行身份 unknown/HOLD: " + reason,
            metrics={"run_identity_unknown": True, "correctness": {"verdict": "unknown"},
                     "eligibility": {k: False for k in
                         ("can_verdict", "can_tune", "can_persist", "can_report", "can_count_progress")}})
        artifact = KernelArtifact(task_id=task.task_id, language=task.target.dsl,
                                  verification_results=[result])
        artifact.resume_status = self._resume_status
        artifact.environment_fingerprint = self._environment.get("fingerprint")
        artifact.environment_drift = self._environment.get("drift")
        # A HOLD report is not a resumable checkpoint and never overwrites its
        # source. Failure to write the report is explicit, not a restart.
        try:
            artifact.saved_dir = str(self.store.save(artifact, task))
        except (OSError, ValueError, TypeError) as exc:
            artifact.metadata = {"hold_report_error": type(exc).__name__}
        return artifact

    def _execute_inner(
        self,
        task: TaskSchema,
        test_inputs: Union[dict, list],
        expected_output=None,
        target_gflops: float = 10.0,
        resume_from: Optional[str] = None,
    ) -> KernelArtifact:
        """实际执行逻辑 (被 execute 的日志上下文包裹)"""
        # The trusted harness must have a complete declared boundary suite before
        # any routing, LLM generation, environment provisioning or GPU work. Keep
        # this as L2 unknown, not a candidate bug eligible for Fixer/corpus lessons.
        # Pure trusted declaration check, not a new optional method required of
        # injected/scripted verifiers (legacy users need not implement it).
        preflight = Verifier.correctness_preflight(task, test_inputs)
        if preflight is not None:
            return KernelArtifact(task_id=task.task_id, language=task.target.dsl,
                                  verification_results=[preflight])
        # §3.4 起**契约也拥有输入**（I94）: 调用方传数组（`dict` 或 `list[dict]`），或者
        # **什么都不传** —— 那时输入由契约的 `cases` 逐个 `materialize()` 造出来。
        # §6.2 条款 2「候选须在**全部**形状上通过」: 调用方可以传**多组**输入
        # （`list[dict]`，每组 = 一条 case）。主 case = 第一组 —— **轮级的所有既有逻辑
        # 一个字节都不动**（轮轴 / 提示词 / 动作上下文 / 停因都按主 case 走）;
        # 而**验证链**拿的是整份列表: L2 逐组判正确性（条款 2 的正确性那半）、L3 逐 case
        # 量加速比（`Verifier._publish_case_ledger` 的账本）。
        # 代价是**声明的**: 传几组就量几组（线性增长）; 只传一组 = 与 I92 之前一字不差。
        input_sets, case_source, declared_cases = self._resolve_input_sets(task, test_inputs)
        preflight = Verifier.correctness_preflight(task, input_sets)
        if preflight is not None:
            return KernelArtifact(task_id=task.task_id, language=task.target.dsl,
                                  verification_results=[preflight])
        self._run_inputs = input_sets
        try:
            self._run_controls = {
                "target_gflops": float(target_gflops),
                "heartbeat_interval": self.heartbeat_interval,
                "cuda_calibration_identity_hash": self.cuda_calibration_identity_hash,
                "cuda_ruler_profile_hash": (self.cuda_ruler_profile["hash"]
                                            if self.cuda_ruler_profile is not None else ""),
                "expected_output": (run_contract.describe_inputs({"expected": expected_output})
                                    if expected_output is not None else None),
            }
        except (ValueError, TypeError) as exc:
            raise run_contract.RunBindingError("unbound expected output/control: " + type(exc).__name__) from exc
        snap_path, snap_scan = self._resolve_snapshot_path(resume_from)
        resume_status = {"requested": bool(resume_from), "status": "not_requested",
                         "root": snap_scan["root"], "candidates": snap_scan["candidates"],
                         "unreadable": snap_scan["unreadable"]}
        self._resume_status = resume_status
        snap = None
        if resume_from:
            if snap_scan["unreadable"] and not os.path.exists(resume_from):
                resume_status["status"] = "ambiguous_lookup"
                raise run_contract.RunBindingError("task_id 检索含不可读记录，请指定明确的快照路径")
            if not snap_path:
                resume_status["status"] = "no_snapshot"
                raise run_contract.RunBindingError("没有 task_id 匹配的完整快照")
            resume_status["path"] = os.path.basename(snap_path)
            try:
                # Keep parse failures distinguishable from integrity failures.
                raw = self._load_snapshot(snap_path)
            except (OSError, ValueError, TypeError) as exc:
                resume_status.update(status="unreadable_snapshot", reason=type(exc).__name__,
                    size_bytes=os.path.getsize(snap_path) if os.path.exists(snap_path) else None)
                raise run_contract.RunBindingError("快照加载失败: " + type(exc).__name__) from exc
            if raw.get("task_id") != task.task_id:
                resume_status.update(status="task_mismatch", snapshot_task_id=raw.get("task_id"))
                raise run_contract.RunBindingError("快照 task_id 与当前任务不一致")
            try:
                snap, self._run_binding = run_contract.load_checkpoint(
                    snap_path, task, input_sets, self._run_controls)
            except run_contract.RunBindingError:
                resume_status["status"] = "identity_mismatch"
                raise
            if snap["round_done"] >= self.max_optimize_rounds:
                resume_status["status"] = "no_remaining_rounds"
                raise run_contract.RunBindingError("没有剩余轮次；旧通过结果不能重新晋级")
            self._run_binding_dir = str(Path(snap_path).parent)
        if task.constraints.reference_contract is not None:
            # Freeze/bind BEFORE Router, provisioning, Designer or Coder. A
            # missing execution identity is a harness hold, never an LLM fix.
            preflight = self.verifier.reference_preflight(task, start_run=True)
            if preflight is not None:
                return KernelArtifact(task_id=task.task_id, language=task.target.dsl,
                                      verification_results=[preflight])
        test_inputs = input_sets[0]
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
        # §10.3 的 ε 抽签种子: 一次运行一个。抽签值是它的纯函数, 而它自己落进产物 ⇒
        # 「第 3 轮为什么走了探索那一支」跑完可以逐位复算 (不必记下每个随机数)。
        _selection_seed = f"{task.task_id}|{self.backend.label}"
        # 指纹的 backend 键取 `fingerprint_label` 而不是 `label` (I72): 组合后端的
        # label 带模板后缀, 而换模板壳是**候选来源**变化, 不是执行环境变化 —— 拿 label
        # 进指纹会让每个新组合后端首跑都被判「环境漂移」(degraded, can_persist=False)。
        fingerprint = fingerprint_of(hw_info, self.backend.fingerprint_label)
        for key in ("selected_gpu", "driver_version", "cuda_version", "gpu_uuid"):
            if key in hw_info:
                fingerprint[key] = hw_info[key]
        # 挂在实例上而不是加到 `_flush_snapshot` 的 12 个形参里: 它在一次运行内是常量,
        # 而 `_flush_snapshot` 有 7 个调用点 (与 `capture_execution` 写 `self._log_path`
        # 同一种「本次运行的上下文」)。`execute()` 不是可重入的。
        self._environment = {"fingerprint": fingerprint, "drift": None}
        if fingerprint:
            print(f"  环境指纹: {fingerprint}")

        try:
            if self._run_binding is None:
                self._run_binding = run_contract.create(task, input_sets, fingerprint,
                                                       self._run_controls)
                # UUID avoids task_id / seconds collisions; contract is durable
                # before routing, retrieval persistence or any agent generation.
                self._run_binding_dir = str(self.store.output_dir /
                    (task.source.kernel_type or "operator") / (task.target.hardware or "unknown") /
                    ("snap_" + self._run_binding["manifest"]["run_id"]))
            else:
                drift = compare_fingerprints(fingerprint,
                    self._run_binding["manifest"]["environment"], "断点续跑的快照")
                self._environment["drift"] = drift.as_dict()
            run_contract.check_current(self._run_binding, task, input_sets, fingerprint,
                                       self._run_controls)
            run_contract.write(self._run_binding_dir, self._run_binding)
        except (OSError, ValueError, TypeError) as exc:
            if resume_from:
                resume_status["status"] = "identity_mismatch"
            raise run_contract.RunBindingError(str(exc)) from exc

        route = self.router.route(task, self.kernel_db)
        print(f"\n[Router] 路由决策: {route.value}")

        # ── 可复现性 Harness: 初始化优化循环的可变状态 ──
        # Step 2: Designer 生成 Sketch (非续跑时才调用 LLM)
        artifact = KernelArtifact(
            task_id=task.task_id,
            language=task.target.dsl,  # python | triton | ...
        )
        artifact.metadata = {"run_binding": self._run_binding}
        all_verification_results = []
        current_sketch = None
        current_code = ""
        pending_fix: Optional[str] = None  # Fixer 修复后的代码，下一轮优先于重新生成
        # 「最近一份 L1+L2 已验证的实现」(arXiv:2609.35140v1 行 214 的 retained verified
        # parent)。它存在的理由是一句话: 修订要从未验过的后代身上退回那个被验过的祖先,
        # 否则修环就是在坏的那一版上继续叠 —— 而今天的代码正是那样 (`fixer.run(current_code, ...)`)。
        # None = 「本次运行还没有任何一版过了正确性验证」, 那时退回哪里都不是这件事的语义。
        verified_parent: Optional[str] = None
        verified_parent_round: int = 0
        fix_attempts: list[tuple] = []     # (失败结果, 修复说明, 那条失败的 Reflexion 记录或 None)
        # Reflexion 记录创建那一刻**还不知道**这次修复成没成 (下一轮验证才有答案), 所以先按
        # fixed=False 落盘 (崩溃也不丢教训), 任务结束时若产物通过再补写一次。这里存的是
        # 「欠着一次补写」的那几条。
        reflexion_pending: list = []
        best_gflops: float = 0.0           # 历史最优 GFLOPS
        rounds_without_improvement: int = 0  # 连续无改善轮数
        # 连续无改善上限，超过则提前终止。**§10.3 条件 1 的 K 就是它** ——
        # 见 `MAX_STAGNANT_ROUNDS` 的说明。
        max_stagnant_rounds: int = MAX_STAGNANT_ROUNDS
        optimization_history: list[dict] = []  # 优化历史记录，每轮追加
        applied_recipes: set[str] = set()      # 已成功应用的配方 ID
        disabled_recipes: set[str] = set()     # 被禁用的配方 ID (应用后性能下降)
        current_recipes_text: str = ""         # 当前轮次注入 Coder 的配方文本
        plan = None                            # 当前轮次的优化计划
        start_round: int = 1                   # 从第 1 轮开始 (续跑时会调整)
        snapshot_target_dir: Optional[str] = self._run_binding_dir

        # File consistency has already been checked before any backend work.
        snap_fingerprint = None
        if snap is not None:
            (start_round, current_sketch, current_code,
             pending_fix, old_history, old_best, old_stagnation,
             old_optimization, old_applied, old_disabled,
             old_results) = self._restore_state_from_snapshot(snap, task)
            if not (pending_fix or current_code):
                resume_status["status"] = "identity_mismatch"
                raise run_contract.RunBindingError("恢复状态没有候选，不允许自动重新生成")
            # Old observations are audit data, not today's tuning/promote
            # credentials. Fresh L1/L2/L3 is mandatory even for the same source.
            artifact.metadata["resume_audit"] = {
                "round_done": snap["round_done"], "best_gflops": old_best,
                "iteration_history": old_history,
                "verification_results": snap["verification_results"],
                "evidence_scope": "previous_process_audit_only",
            }
            pending_fix = pending_fix or current_code
            self._resume_candidate_once = True
            snap_fingerprint = snap.get("environment_fingerprint")
            resume_status.update(status="resumed", round_restored=start_round - 1,
                dir_name=os.path.basename(os.path.dirname(snap_path)),
                evidence="fresh_verification_required")
            print(f"[Conductor] 断点续跑: 恢复候选，轮次 {start_round}; 旧验证仅作审计")
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

        # P1-4 (generation-side-review): Search 排好序的邻域候选队列 —— LLM 建议
        # 降为先验后, 下轮 L3 再败时按序轮换, 不重新问 LLM。真机 L3 是裁判:
        # 改进即停 (stagnation 逻辑照旧)。不进快照: 断点续跑时机器状态已变,
        # 旧排序作废, 重新让 Search 出一轮队列。
        search_alternatives: list = []
        # §10.4 的「有希望的区域已穷尽」：逐轮的覆盖判定收成一份运行级记录。它回答的是
        # 「K 轮无改善」之外的另一半 —— 没有它, 停滞出口就只是「轮数够了」。
        coverage_ledger = CoverageLedger()
        # §5 bandit: 让队列顺序吃本轮累积的证据, 而不是纯 FIFO。与上面那条注释同一个
        # 理由 —— 证据只覆盖本次运行, 机器状态一变就作废, 所以同样不进快照。
        bandit = Bandit()
        # smem 上界的**载体** (§12 便宜代理剪枝的输入, §9 的「便宜代理先剪枝」)。
        # 与 Verifier 发布进 L3 的那一格**同一个真源** (`detect_hardware_caps`), 不是
        # 在这里另查一遍硬件。取不到 ⇒ MemCaps(0) ⇒ 上界未知 ⇒ 剪枝不生效并明说。
        _mem_caps = caps_from_metrics(detect_hardware_caps(hw_info))
        # I75-G1: 同一份 smem 上界也交给 compile-gate —— 不可行 tile 在采样处
        # 剪掉, 不烧 SSH (首跑教训: 六次 L1 全是 256×256 超限, 每次白烧一轮往返)。
        # 单体路径/测试假件 (pipeline=None) 时这里自然跳过。
        if self._coder_pipeline is not None:
            self._coder_pipeline.caps = _mem_caps
        #: 最近一次**有调参资格**的读数构成的信念 (设计 §10.4 伪码里的 `belief`)。
        #: `can_tune=False` 的轮次不许覆盖它, 只许沿用它。
        last_tunable_perf: Optional[dict] = None

        # 初始化运行状态: 从这里往下, 轮次循环与收尾读写的是同一个对象。
        # 上面那段「准备」仍然用局部名 —— 它们只在这一次构造里被读一次, 收进来没有意义。
        state = RunState(
            artifact=artifact,
            task=task,
            target_gflops=target_gflops,
            test_inputs=test_inputs,
            input_sets=input_sets,
            declared_cases=declared_cases,
            case_source=case_source,
            _selection_seed=_selection_seed,
            bandit=bandit,
            coverage_ledger=coverage_ledger,
            current_sketch=current_sketch,
            current_code=current_code,
            pending_fix=pending_fix,
            best_gflops=best_gflops,
            rounds_without_improvement=rounds_without_improvement,
            optimization_history=optimization_history,
            applied_recipes=applied_recipes,
            disabled_recipes=disabled_recipes,
            all_verification_results=all_verification_results,
            start_round=start_round,
            verified_parent=verified_parent,
            verified_parent_round=verified_parent_round,
            snapshot_target_dir=snapshot_target_dir,
            search_alternatives=search_alternatives,
            plan=plan,
            last_tunable_perf=last_tunable_perf,
            fix_attempts=fix_attempts,
            reflexion_pending=reflexion_pending,
            duplicate_rounds=duplicate_rounds,
            refused_rounds=refused_rounds,
            invalid_measurement_rounds=invalid_measurement_rounds,
            round_signatures=round_signatures,
        )

        for round_num in range(state.start_round, self.max_optimize_rounds + 1):
            # 一轮的全部流程在 _run_round 里 (循环体那 1128 行搬过去了)。返回
            # _ROUND_STOP = 这一轮命中了某个退出出口（含传输 unknown HOLD）。
            if self._run_round(state, round_num, expected_output=expected_output,
                               hw_info=hw_info, _mem_caps=_mem_caps,
                               max_stagnant_rounds=max_stagnant_rounds,
                               verify_drift=verify_drift) is _ROUND_STOP:
                break

        # 收尾: 产物成型 / 报告 / 返回值。循环到此结束, 状态整体交出去。
        return self._finalize_run(state=state)


    def _snap(self, state, round_num) -> str:
        """把当前循环状态写进 execution_snapshot.json

        循环内的落盘点都走这里 (通过 / 停滞 / 已到 roofline / L1L2 耗尽轮次 /
        走到轮次末尾)。少数例外是参数不同的那几处 —— 例如 Fixer 抛异常时
        `pending_fix` 要写成 None, 它们仍直接调 `_flush_snapshot`。

        ⚠️ 变异测试的一个空跑: 终止出口处这次落盘写在 return _ROUND_STOP 之前、save 之前,
        而 `execute()` 末尾一定会再补写一次进权威的 `saved_dir`。所以删掉出口
        那次不会有任何断言变红 —— 它只在「进程在 save 之前就没」时才有读者。
        保留是为了与其余出口动作一致。
        """
        return self._flush_snapshot(
            task=state.task,
            artifact_dir=state.snapshot_target_dir,
            round_done=round_num,
            current_sketch=state.current_sketch,
            current_code=state.current_code,
            pending_fix=state.pending_fix,
            iteration_history=state.artifact.iteration_history,
            best_gflops=state.best_gflops,
            rounds_without_improvement=state.rounds_without_improvement,
            optimization_history=state.optimization_history,
            applied_recipes=state.applied_recipes,
            disabled_recipes=state.disabled_recipes,
            all_verification_results=state.all_verification_results,
        )


    def _round_enter_or_stop(self, state, round_num) -> bool:
        """轮次**最前面**那两道预算门; 返回 _ROUND_STOP 表示这一轮不该开 (§10.2)。

        位置是判据的一部分: 放在 Coder/Planner 之后等于「先花完 LLM 与写代码的钱,
        再说这次测量买不起」。两道门都是 fail-closed: 买不起就停, 不停在半路上。

        为什么 oracle 这一格用 "explore" 而不是按 explore/refine 分: 本仓今天
        **没有**「参数搜索平台化」这条判据 (§10.3 的升级门, CHECKLOG I53), 分不出来
        就如实只花一格; 预留 (`confirm`) 那一格一律不许碰 —— 那才是这条预算的意义。

        搬出来是纯搬运: 函数体逐字未改; 依赖只有 self/state/round_num —— 定值分析实测
        _why/_plan/_elapsed 都是现写现读的临时变量, 不出这一段。
        """
        if not self._budget.within_wall_clock(time.monotonic() - self._run_started):
            _elapsed = time.monotonic() - self._run_started
            _why = (f"墙钟预算用尽 ({_elapsed:.1f}s >= "
                    f"{self._budget.plan.wall_clock_s}s) ⇒ 不再开新的一轮")
            print(f"\n[Budget] {_why}")
            self._budget.oracle_refuse("explore")
            self._record_stop(state.artifact, round_num, "wall_clock_exhausted", _why)
            return _ROUND_STOP
        if not self._budget.oracle_may("explore", 1):
            _plan = self._budget.plan
            _why = (f"真值测量预算用尽 (已花 {self._budget.spent_all}/"
                    f"{_plan.oracle_total}, 其中 {self._budget.caps['confirm']} 次"
                    f"留给最终确认不许挪用) ⇒ 不再开新的一轮")
            print(f"\n[Budget] {_why}")
            self._budget.oracle_refuse("explore")
            self._record_stop(state.artifact, round_num, "budget_exhausted", _why)
            return _ROUND_STOP


    def _round_close_out(self, state, round_num, *,
                         _esc_decision, _measure_s, _space, bandit_arm, l3_result) -> bool:
        """这一轮的收尾: 参考天花板 / 无改善停滞 / 调参资格门 / 出口判定 (§10.4)。

        入参这 5 个不是设计出来的接口, 是**实测**出来的依赖 —— 用「段内可能未先
        赋值就读」的定值分析算的 (粗糙的「前面写过、后面读过」会多报: _why/_plan
        这类每处都是现写现读的临时变量, 实际不出段)。

        搬出来是纯搬运: 形参名与原来的局部名逐字相同, 所以函数体一个字符都不用改。
        """
        if l3_result and not l3_result.passed and round_num < self.max_optimize_rounds:
            # ── 资格门: `can_tune` 管的是 **belief**, 不是「停循环」 ──
            # 设计 §10.4 的伪码就是这条:
            #     obs = oracle.evaluate(c, contract)
            #     if obs.eligibility.can_tune: belief.update(c, obs)
            #     if obs.eligibility.can_count_progress: stagnation.update(belief)
            # 也就是说没资格的那个数**不进信念**, 于是 `propose()` 只能从**上一次
            # 有资格的读数**构成的信念出发 —— 循环照常跑满预算 (那条策略另有真机
            # 用例守着: `uninformative` 轮次必须跑满、不许提前终止)。
            #
            # 此前这里**没有任何门** (CHECKLOG I52): 一个档位 `uninformative`、五位
            # 全 False 的读数照样被拼成 `perf_data` (gflops/speedup/`roofline_bound`
            # 全在里面), 喂给邻居排序 (`agents.py:680` 读 roofline_bound 决定往大
            # BLOCK_K 还是小 BLOCK_M/N 排) 与 LLM 先验 (`agents.py:708` 读 speedup)。
            # §10.2 第三行: LLM token 独立管, 超了退化成纯数值搜索。口径是
            # **本次运行**的用量 (`summary_since(水位线)`), 不是这个进程到现在的累计。
            # 口径只有一处 (`_llm_tokens_used`): §21.1 的预算校验也读它。两处各算
            # 一遍同一件事, 迟早会出现两个「还剩多少」。
            _tokens_used = self._llm_tokens_used()
            _llm_broke = self._budget.note_llm_usage(_tokens_used)
            if _llm_broke:
                print(f"\n[Budget] LLM token 预算用尽 ({_tokens_used}/"
                      f"{self._budget.plan.llm_tokens}) ⇒ 本轮退化成纯数值搜索 "
                      f"(不问 LLM 要结构改动)")
            _elig = l3_result.metrics.get("eligibility") or {}
            _can_tune = _elig.get("can_tune") is True
            if not _can_tune:
                print(f"\n[Search] 本轮读数无调参资格 "
                      f"(can_tune=False; 档位="
                      f"{l3_result.metrics.get('measurement_validity')!r})"
                      f" ⇒ 不进信念")
            print(f"\n[Search] L3 性能未达标，启动优化...")
            # ── bandit 记账 (§5 / §10.5-3) ──
            # 回报口径 = 相对**本轮之前**最好 gflops 的改进 —— 与 `best_gflops` /
            # 停滞判定同一个量, 不另立一个口径。资格位决定这次回报算不算数:
            # `can_count_progress=False` 的那一轮只记「试过」, 回报不进均值
            # (「试过但没数」既不是「没试过」, 也不是「试过且不好」)。
            _cur = l3_result.metrics.get("gflops")
            _reward = None
            if (isinstance(_cur, (int, float)) and not isinstance(_cur, bool)
                    and isinstance(state.best_gflops, (int, float)) and state.best_gflops > 0):
                _reward = (float(_cur) - float(state.best_gflops)) / float(state.best_gflops)
            state.bandit.update(bandit_arm, _reward,
                          eligible=_elig.get("can_count_progress") is True,
                          # §10.3 的分母原料: 这一轮把秒换成数字花了多少。只有真跑出
                          # L3 读数才算 (L1/L2 就挂掉的轮次没有测量秒可言 ⇒ None)。
                          cost_s=_measure_s if l3_result is not None else None)
            perf_data = {
                "gflops": l3_result.metrics.get("gflops", 0),
                "speedup": (decision_speedup(l3_result.metrics) or 0),
                "target_speedup": state.task.constraints.min_speedup,
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
                # ---- §12 便宜代理剪枝的输入 (不花 GPU 秒) ----
                # 上界由 Verifier 从 caps 发布 (0 = 不知道); elem_bytes 取**任务声明
                # 精度**的字节数 —— 与 smem 估算、AI 分母同一个口径 (核根本见不到
                # 控制端手里那份 float32 数组, 见 verifier._compute_arithmetic_intensity)。
                "shared_mem_per_sm_kb": l3_result.metrics.get("shared_mem_per_sm_kb", 0),
                "elem_bytes": state.task.constraints.precision.itemsize,
                # ---- 优化历史 ----
                "optimization_history": state.optimization_history,
            }
            if _llm_broke:
                # 与 `tuning_refused` 走同一条通道 (perf_data), 不新增签名 ——
                # SearchAgent 见到它就跳过 `_llm_optimize` 走规则路径。
                perf_data["llm_budget_exhausted"] = True
            # §10.3: 把本轮裁决**发布**给 SearchAgent (判据在判决点之后已经算过,
            # 这里只发布 —— 一个判定一处实现)。不通过 ⇒ 这一轮只做参数搜索。
            perf_data["structural_escalation_denied"] = list(_esc_decision.refusals)
            if _can_tune:
                # 有资格 ⇒ 这份读数进信念, 供后面的轮次沿用
                state.last_tunable_perf = perf_data
            elif state.last_tunable_perf is not None:
                # 没资格 ⇒ 沿用**上一次有资格读数**构成的信念 (伪码里的 `belief`)
                perf_data = state.last_tunable_perf
            else:
                # 本次运行还没有任何一次有资格的读数: 给一个**不含任何数字**的
                # 结构先验。`tuning_refused` 是给 SearchAgent 的信号 —— 它必须
                # 把「没有读数」如实写进 prompt, 不许用 `.get(k, 0)` 编一个 0 出来
                # (`_llm_optimize` 原来正是那样: 缺键 ⇒ 0.00 印进 prompt)。
                perf_data = {
                    "target_speedup": state.task.constraints.min_speedup,
                    "tuning_refused": "can_tune=False",
                    "optimization_history": [],
                }
            if _space is not None:
                # §12: 模板型后端的候选来自参数空间 (下一轮轮首再取一次, 那时
                # bandit 已经吃过本轮的回报 ⇒ 顺位会跟着证据走)。这条 LLM/邻域搜索
                # 路径**整条不适用**: 让 SearchAgent 改 tiling 会把「模板点」与
                # 「LLM 结构」混进同一个候选队列 —— 而它们连身份口径都不同。
                print(f"\n[Search] 模板型后端: 候选来自参数空间, 本轮不走 LLM 搜索")
                # 模板轮**没有**新 sketch 也没有 LLM 建议: 那两行原样打出来会
                # 读成「这一轮改了 sketch」—— 那是代码型那边的语义。
                #
                # `optimization_history` 在这里**一个字都不写**: 它唯一的读者是
                # SearchAgent 的 prompt (`agents._llm_optimize`), 而模板轮根本不叫
                # SearchAgent —— 写一条进去就是「只有调试者会读到」的键 (与
                # CHECKLOG I36 那类「算了没人读」同族)。这一轮的候选身份已经在
                # `iteration_history.template_selection` 里, 那是有读者的那一格。
            else:
                if state.search_alternatives:
                    # P1-4: 上一轮排好的邻域候选按序轮换 —— 系统搜索代替「每轮
                    # 重新问一次 LLM 直觉」。轮换候选已带结构 (hints/memory_plan),
                    # 只换了 tiling, 不需要再过 Search。
                    #
                    # 选**哪一个**交给 §10.3 的选择器 (`choose`): ε 那一支去未探索的
                    # 区域, 1-ε 那一支去分数最高的; 计划格子 (分相预算) 约束模式。
                    _nsel = choose(state.search_alternatives, state.bandit, self._arm_of,
                                   epsilon=self.selection_epsilon,
                                   draw=explore_draw(state._selection_seed, round_num),
                                   phases=self._plan_phases())
                    if _nsel is None:
                        _room = self._budget.as_dict()["phase_room"]
                        _why = (f"探索/精修两格的计划额度都用完了 (剩余 {_room}) 且"
                                "邻域队列里没有符合模式的候选 ⇒ 不再开新的一轮")
                        self._budget.oracle_refuse("explore")
                        self._record_stop(state.artifact, round_num, "budget_exhausted", _why)
                        return _ROUND_STOP
                    state.current_sketch = _nsel.chosen
                    state.search_alternatives = [x for x in _nsel.ranking
                                           if x is not _nsel.chosen]
                    _round_selection = dict(_nsel.as_dict(), applies_to="next_round",
                                            chosen_label=self._arm_of(_nsel.chosen))
                    print(f"\n[Selection] {_nsel.note}")
                    t = state.current_sketch.tiling
                    suggestion = "P1-4 邻域候选轮换"
                    print(f"\n[Search] 轮换邻域候选: tiling={t.block_m}x{t.block_n}x{t.block_k} "
                          f"(队列剩 {len(state.search_alternatives)} 个)")
                else:
                    # ── §12 的开关: 后端声明了参数空间 ⇒ 模板型 ──
                    # 控制器「自动切换成 LLM/模板产出候选、bandit 只做局部精修」这件事,
                    # 今天只有代码型那一半有实现 (bandit 精修 = P1-4 邻域)。模板型那一半
                    # (由参数空间枚举候选并实例化) 还没落地 ⇒ 在这里 fail-closed, 而不是
                    # 把它降级成代码型继续跑: §12 点名的风险正是「把候选悄悄当成源码字符串」。
                    # 放在真值测量**之前** (这一处是每轮搜索的入口), 因为拒绝要发生在花钱之前。
                    try:
                        state.current_sketch, suggestion = self.search.run(state.current_sketch, perf_data)
                    except Exception as e:
                        # 规则优化也失败时, 终止优化但不崩整个任务
                        print(f"\n[Conductor] Search 优化异常 ({e}), 终止优化循环")
                        state.snapshot_target_dir = self._flush_snapshot(
                            task=state.task,
                            artifact_dir=state.snapshot_target_dir,
                            round_done=round_num,
                            current_sketch=state.current_sketch,
                            current_code=state.current_code,
                            pending_fix=state.pending_fix,
                            iteration_history=state.artifact.iteration_history,
                            best_gflops=state.best_gflops,
                            rounds_without_improvement=state.rounds_without_improvement,
                            optimization_history=state.optimization_history,
                            applied_recipes=state.applied_recipes,
                            disabled_recipes=state.disabled_recipes,
                            all_verification_results=state.all_verification_results,
                        )
                        self._record_stop(state.artifact, round_num, "search_error", "Search 优化抛异常, 终止优化循环")

                        return _ROUND_STOP
                    # P1-4: 排队下一轮的轮换候选 (SearchAgent 只在 triton 路径产出)
                    # §5: 顺序交给 bandit (本轮累积的证据优先, 无观测的保持先验顺序),
                    # 而不是纯 FIFO —— 后者对脏数据免疫但也无知: 哪个候选上次变快了,
                    # 队列不知道。资格门在这层之外 (没资格的数根本没进来)。
                    # 队列在这里**只入队不排序** (保持先验顺序): 排序与「取哪一个」
                    # 是同一个判断, 它发生在**消费那一轮** (`choose`) —— 那时证据最新,
                    # 而且一次判断只有一个地方做。这里再排一遍会让同一个选择器对同一批
                    # 候选跑两次 (真实 bandit 是幂等的, 但那是巧合, 不是设计)。
                    state.search_alternatives = list(
                        getattr(self.search, "last_alternatives", []))
                    if state.bandit.total_scored or state.bandit.unscored_updates:
                        print(f"  [Bandit] 队列按证据重排: 已采纳 {state.bandit.total_scored} 个回报, "
                              f"没采纳 {state.bandit.unscored_updates} 个")
                print(f"  优化建议: {suggestion}")
                print(f"  新 Sketch 版本: v{state.current_sketch.version}")

                # ---- 记录本轮优化历史 (供后续轮次参考) ----
                # 只在有资格时累加: 这份 history 会印进下一轮的 prompt, 也就是**信念**的
                # 一部分。把没有调参资格的那一轮数字写进去, 等于换个通道让它指挥搜索。
                if _can_tune:
                    state.optimization_history.append({
                        "tiling": f"{state.current_sketch.tiling.block_m}x{state.current_sketch.tiling.block_n}x{state.current_sketch.tiling.block_k}",
                        "gflops": l3_result.metrics.get("gflops", 0),
                        "stage": "algorithm",  # 由 SearchAgent 返回实际 stage
                        "outcome": f"GFLOPS={l3_result.metrics.get('gflops', 0):.2f}, "
                                   f"加速比={l3_result.metrics.get('speedup_vs_baseline', 0):.2f}x",
                    })

        # ── 可复现性 Harness: 循环走到末尾 (没走 break/continue) 时统一 flush 一次快照 ──
        state.snapshot_target_dir = self._snap(state, round_num)
        if not (l3_result and not l3_result.passed and round_num < self.max_optimize_rounds):
            print(f"\n[Conductor] 已达最大优化轮次，停止迭代")


    def _run_round(self, state, round_num, *, expected_output, hw_info, _mem_caps,
                   max_stagnant_rounds, verify_drift) -> bool:
        """跑完一轮; 返回 _ROUND_STOP 表示这一轮命中了退出出口。

        这是从 _execute_inner 的轮次循环体**整体**搬出来的一步。搬运方式是逐字的:
        除下面这 14 处控制流改写之外, 方法体与原来的循环体逐 AST 节点相同 (写盘前按
        AST 逐节点比对过; 搬完之后那 6 个跨轮记忆名字的判据在
        tests/test_run_state_single_source.py):
          13 处 break    -> return _ROUND_STOP
           1 处 continue -> return _ROUND_CONTINUE
        改过之后, 这个方法内部再没有能跳出调用者的语句 —— 「这一轮要不要终止整个
        优化循环」只由返回值表达, 不再靠 break 从 1128 行深处跳出去。

        Stage3 又把轮次的两头拆成具名阶段 (同样是纯搬运, 写盘前逐 AST 节点比对过):
          * _round_enter_or_stop —— 轮首两道预算门 (§10.2), 零依赖;
          * _round_close_out     —— 收尾的参考天花板/停滞/资格门/出口判定 (§10.4),
                                    吃 5 个**实测**出来的依赖;
          * _snap                —— 快照落盘, 原来在这里是个闭包 (只读 self/state/round_num)。
        中段没有再拆: 每个内部切分点的真实入向是 12-25 个值 (用「可能未先赋值就读」的
        定值分析算的), 没有概念上的接口可言 —— 硬拆就得造一个 12-25 字段的中转记录,
        那是把「依赖清单」这张契约换成一堆凑数的字段。形状由 _phase_problems 钉住。

        接口为什么是 state + round_num + 5 个关键字:
          * state 装的是**跨轮活下来**的那些值 —— 既包括收尾也要读的, 也包括循环
            自己攒的记忆 (见 RunState 类文档)。字段清单由三条既有契约钉住
            (tests/test_run_state_single_source.py)。
          * 另外 5 个是**只有这一轮才需要、而且从不被写**的只读上下文 (期望输出 /
            硬件信息 / 显存上限 / 停滞阈值 / 漂移判定)。它们刻意不进 state: 那样
            字段清单就退化成「凡是这一轮读到的都算」, 也就不再是契约了。
        """

        print(f"\n{'─'*40}")
        print(f"轮次 {round_num}/{self.max_optimize_rounds}")
        print(f"{'─'*40}")

        # ── §10.2 的两道预算门在最前面 (位置是判据的一部分), 见 _round_enter_or_stop ──
        if self._round_enter_or_stop(state, round_num) is _ROUND_STOP:
            return _ROUND_STOP
        if self._run_binding is not None:
            self._check_run_binding(state.task, state.input_sets)

        # ── §12 模板型后端: 本轮的候选来自**参数空间**, 不走 LLM/Coder ──
        # 位置与预算门同级 (轮次最前面): 候选的**来源**决定这一轮怎么花钱, 所以
        # 要在花钱之前定下来。开关只有一处判据 (`require_supported_space`): 后端没
        # 实现实例化器 ⇒ 那里就炸; 这里只管「建不建得出来、先试哪个点」。
        _space = require_supported_space(
            param_space_of(self.backend, state.task), self.backend)
        template_choice = None
        _template_sel = None
        # §21.3 这一段进没进**本轮修复提示**的留痕 (读者: A/B 对照与复盘)。
        # 初值必须在轮首: 上一轮修过的话, 这一轮的血缘会带着上一轮的记录 ——
        # 「这一轮的提示里有没有结构化证据」这个问题就答错了 (`selection` 那一格
        # 初值放晚了是同一种错误, 被用例抓过一次)。
        # ⚠️ 必须是**会被后填的字典** (与 `_coverage_record` 同一处置): 血缘条目在
        # 修复之前就 append 了, 直接填一个 None/字符串的话后填的值进不了产物 ——
        # 第一版就是这么写的, 产物里每一轮都是 `null`, 用例当场抓到。
        _fix_evidence_record: dict = {"injected": False}
        # §21.1 生产者侧的动作留痕 (与上面同一种写法: 血缘在轮首就 append, 后填的值
        # 靠引用进产物)。它记的是「动作是谁写的」以及询问类信封被怎么响应 ——
        # 没有这一格，「模型写的理由」与「harness 拼的理由」在产物里长得一样。
        _fix_action_record: dict = {"asked": False}
        # 本轮**可枚举**的候选全集 (§10.4 的覆盖判据要它): 模板型 = 剪枝后的点,
        # 代码型 = None (候选由 LLM/规则按需产生 ⇒ 覆盖无法确立)。初值必须在轮首 ——
        # 否则代码型轮会读到**上一轮**留下的一批点 (那会让覆盖判定拿旧全集说新话)。
        _universe = None
        # 这一轮的覆盖判定 (§10.4 的另一半)。血缘条目在**轮首**就 append 了, 而覆盖
        # 判定要等本轮的观测进账 (这一轮的候选试过没有) —— 所以这里放一个**会被后填的
        # 字典**, 与 `permanent_effects` 同一处置 (血缘里存的是同一个对象的引用)。
        # 第一版直接 `_coverage = None` 再在下面赋值 ⇒ 每一轮血缘里都是 `null`。
        _coverage_record: dict = {}
        # §10.3 这一轮的选择 (模式/抽签/分数/顺序)。模板型 = 本轮用的候选就是它选的;
        # 代码型 = 它在轮末选的是**下一轮**的候选 (`applies_to` 说明是哪一种)。
        # ⚠️ 初值必须在**选择之前**: 第一版把它放在模板选择之后, 于是刚选出来的那份
        # 记录被擦成 None (产物里每一轮都是 `selection: null`, 用例红了才发现)。
        _round_selection = None
        _draw = explore_draw(state._selection_seed, round_num)
        if _space is not None and not self._resume_candidate_once:
            # §10.3 的选择: 候选 = 参数空间的点。`allowed` 是「还有计划格子的相位」
            # (分相预算) —— 它**约束**这一轮是探索还是精修, 于是 55%/30% 那个计划
            # 真的体现在花的钱上, 而不只是一行注释。
            # 候选 = **剪枝后**的点 (与队列同一批): 否则选择器会把剪掉的点排进顺位,
            # 队列再过滤掉 ⇒ 「想选谁」与「用了谁」分叉, 连花哪一格预算都跟着错。
            _universe = template_points(
                _space, _mem_caps, state.task.constraints.precision.itemsize)
            _sel = choose(_universe.kept, state.bandit, template_arm,
                          epsilon=self.selection_epsilon, draw=_draw,
                          phases=self._plan_phases())
            if _sel is None:
                _room = self._budget.as_dict()["phase_room"]
                _why = (f"探索与精修两格的计划额度都用完了 (剩余 {_room}) ⇒ 不再开"
                        "新的一轮")
                print(f"\n[Template] {_why}")
                self._budget.oracle_refuse("explore")
                self._record_stop(state.artifact, round_num, "budget_exhausted", _why)
                return _ROUND_STOP
            # `chosen_label` 是「想选谁」的人读名字 (臂是去重键,留给 bandit 的统计);
            # 它与队列最终用的那个点可能不同 (顺位回退), 所以两者都要在产物里。
            _round_selection = dict(_sel.as_dict(), applies_to="this_round",
                                    chosen_label=point_label(_sel.chosen))
            print(f"\n[Selection] {_sel.note}")
            _template_sel = select_template_candidate(
                _space, self.backend, caps=_mem_caps,
                elem_bytes=state.task.constraints.precision.itemsize,
                order=_sel.ranking)
            template_choice = _template_sel.chosen
            print(f"\n[Template] {_template_sel.note()}")
            if template_choice is None:
                # 空间非空 (ParamSpace 校验过), 所以这个 None 只剩一个含义: **所有点
                # 都建不出来**。停在这里并说明, 不退回代码型 —— §12 点名的风险正是
                # 「把声明的参数空间悄悄丢掉后继续跑」。
                _why = (f"参数空间 {_space.template_id!r} 的 {len(_template_sel.ranked)} "
                        f"个点一个都没建成 (实例化器拒绝 {len(_template_sel.refusals)} 条)"
                        " ⇒ 停止, 不退回代码型生成路径")
                print(f"\n[Template] {_why}")
                self._record_stop(state.artifact, round_num, "template_no_candidate", _why)
                return _ROUND_STOP

        # 本轮的 bandit 臂 = **本轮候选的身份** (§5 的排序器吃的就是它): 模板轮 =
        # 那个点的去重键, 代码型轮 = tiling 三元组。刻意与候选身份同源 —— 一个候选
        # 只有一个臂, 否则「这个点上次变快了没有」会随调用方各自解释。
        #
        # 位置: 必须在血缘 append **之前** (那是一轮里更早的位置), 也必须在本轮
        # 搜索**之前** —— 搜索那一步改的是下一轮的 sketch 与队列, 都在它之后, 所以
        # 这里算出来的与搜索块用掉的是同一个值。
        bandit_arm = (template_arm(template_choice.candidate)
                      if template_choice is not None
                      else self._arm_of(state.current_sketch))
        # 这一轮的候选值哪一相预算 (§10.3 的分相): 没观测过 ⇒ 探索, 观测过 ⇒ 精修。
        # 它与 `Selection.mode` 是同一件事的两半 (探索那一支只取未观测的候选), 所以
        # 「记的相位 = 当时选的模式」是一条可被测试钉住的不变量。
        round_phase = phase_of(state.bandit, bandit_arm)

        # P1-1 (generation-side-review): 本轮若走了候选选择, 已跑完的 L1+L2
        # 结果缓存在这里 —— Step 4 复用它, 不重复真机执行。
        # None = 本轮没有多候选 (单体路径 / 修复轮 / 名单里没有默认候选)。
        # l1_l2_is_final = 这份结果已是最终态 (决赛全败: L2 已失败, 不该再跑 L3)。
        selected_l1_l2 = None
        l1_l2_is_final = False
        # P1-1 成本记账: 本轮 compile-gate 放行了几个决赛候选, 以及真机 L2 跑了几次。
        # 翻转默认路径后每轮生成从 1 个候选变成最多 2 个决赛候选 + 最多 2 次 L2,
        # 而「这次翻转让机时涨了多少」在产物里此前没有任何原料。
        round_candidates = 0
        round_l2_runs = None      # None = 由 verify_results 现算 (非选择路径)
        # (臂在下面「模板选择」之后立刻算 —— 血缘条目在那一轮更早的位置 append,
        #  所以臂必须在 append 之前就有值; 见那一处的说明。)

        # Step 3: 获取代码
        # - 首轮 / Search 优化后: Coder 根据 Sketch + OptimizationPlanner 配方重新生成
        # - Fixer 修复后: 直接使用修复版代码重新验证
        # §21 的回滚点与 diff 基线: 本轮开始**之前**手里的那份代码 (= 上一轮的成品,
        # 也就是本轮 Fixer 被要求去修的那一份)。必须在下面三支改写 `current_code`
        # 之前取 —— 取晚了就成了「自己与自己比」。
        _code_before_this_round = state.current_code
        # 本轮候选的**来路**（§21.1 生产者侧）。轮首置空 —— 只有「本轮用的是 Fixer
        # 交上来的那份」这一支会武装它, 于是生成轮/模板轮恒为 `absent`。
        _model_envelope: Optional[ActionEnvelope] = None
        if state.pending_fix is not None:
            state.current_code = state.pending_fix
            state.pending_fix = None
            # 信封与 `pending_fix` **同生同灭**（唯一的消费点就在这两行）: 拿出来的
            # 是「这份候选是谁怎么提出来的」。它进不了别的轮次 —— 上面那一行已经把它
            # 清空, 而下一轮的候选要么由 Coder 生成（没有信封）, 要么由新的修复轮武装。
            _model_envelope = self._pending_action_envelope
            self._pending_action_envelope = None
            action = "fix"
            if self._resume_candidate_once:
                action = "resume"
                self._resume_candidate_once = False
            print(f"\n[Conductor] 使用 Fixer 修复后的代码重新验证...")
        elif template_choice is not None:
            # §12: 模板候选的源码**来自实例化器**, 不经 LLM/Coder (§12 的
            # 「控制器自动切换成『LLM / 模板产出候选』」那一半)。action 单列一个值,
            # 于是「这一轮是模板产出的」在血缘里可分辨, 不必从源码去猜。
            state.current_code = template_choice.source
            action = "template"
            print(f"\n[Conductor] 模板候选 (跳过 Coder): "
                  f"{point_label(template_choice.candidate)}")
        else:
            action = "generate" if round_num == 1 else "optimize"

            # ---- OptimizationPlanner: 根据 Roofline 选择本轮要应用的配方 ----
            prev_l3_metrics = {}
            if state.all_verification_results:
                prev_l3 = next((r for r in state.all_verification_results
                                if r.level == VerificationLevel.L3_PERFORMANCE), None)
                if prev_l3:
                    prev_l3_metrics = prev_l3.metrics

            state.plan = self.planner.plan(
                kernel_type=state.task.source.kernel_type,
                dsl=state.current_sketch.target_dsl if state.current_sketch else state.task.target.dsl,
                roofline=prev_l3_metrics if round_num > 1 else None,
                hardware=hw_info,
                precision=state.task.constraints.precision.value,
                round_num=round_num,
                disabled_recipes=state.disabled_recipes,
                active_recipes=state.applied_recipes,
            )
            current_recipes_text = state.plan.to_prompt()
            if state.plan.selected_recipes:
                recipe_names = ", ".join(
                    f"{r.id}({r.name})" for r in state.plan.selected_recipes
                )
                print(f"\n[Planner] 选中 {len(state.plan.selected_recipes)} 个优化配方: {recipe_names}")
                print(f"  {state.plan.summary()}")
            else:
                print(f"\n[Planner] 本轮无新增优化配方")

            print(f"\n[Coder] 根据 Sketch + 优化配方生成代码 (v{state.current_sketch.version})...")
            state.current_code = self.coder.run(
                state.current_sketch, state.task,
                optimization_recipes=current_recipes_text,
                test_inputs=state.test_inputs,
            )
            # Multi-finalist selection runs real L1/L2 before the main
            # verification block; it must not bypass this generation gate.
            if self._run_binding is not None:
                self._check_run_binding(state.task, state.input_sets)

            # ---- P1-1 候选选择: compile-gate 只定了决赛名单, 真机 L1+L2 定胜负 ----
            # 必须紧跟本轮流水的 coder.run 读名单: `last_finalists` 跨轮残留,
            # 修复轮 (pending_fix) 不该拿上一轮的名单重新选。
            # getattr: 测试常用自定义 coder 替身 (无此属性 = 单候选, 跳过选择)。
            finalists = getattr(self.coder, "last_finalists", [])
            round_candidates = len(finalists)
            if len(finalists) > 1:
                (state.current_code, selected_l1_l2, l1_l2_is_final,
                 round_l2_runs) = self._select_finalist(
                    state.current_code, finalists, state.task, state.input_sets, expected_output,
                )

        # ---- 候选新颖度观测 (doc 47) ----
        # 只做两件事: 把结构签名写进 iteration_history (它会被序列化进 result.json,
        # 于是「模型重新生成时有多少轮是结构重复」第一次有一个能复盘的数), 以及在
        # 当场打印出来。**不跳过验证** —— 假阳的代价是丢掉一个真的更好的核, 在没有
        # 基率之前拿它省 GPU 时间就是拿判决换一个还没量的东西。
        sig = signature_of(state.current_code)
        dup_of = next((r for (r, s) in state.round_signatures if s and s == sig), None)
        if sig:
            state.round_signatures.append((round_num, sig))
        if dup_of is not None and action == "optimize":
            state.duplicate_rounds += 1
            print(f"\n[Novelty] 第 {round_num} 轮与第 {dup_of} 轮结构相同 (签名 {sig}) "
                  f"—— 照常验证, 本轮只在量基率")

        # ── §21 Agent Runtime: 候选进编译之前的独立检查 (§21.4) ──────────────
        # 位置是判据的一部分: 它在**所有**候选来源（Coder / 模板实例化器 / Fixer
        # 修复版 / 决赛选择）汇合之后、在**任何** GPU 与 SSH 之前。放到 producer
        # 那一侧就漏掉三条路（模板、修复、决赛选择），放到 Verifier 里面则会把
        # 「动作协议」塞进评测器。
        #
        # 被拒的后果是**定义好的**：本轮不进 L1/L2/L3，合成一条 L1 失败结果
        # （`agent_action_refused`），于是 §8 的路由与 Fixer 的处方照旧工作 ——
        # 拒绝不是「静默丢掉这个候选」。
        _prev_route = (classify_failure(state.all_verification_results)
                       if action == "fix" and state.all_verification_results else None)
        # §21.1 生产者侧: 这一轮的候选如果是 Fixer 交上来的, 它可能同时交了**动作信封**
        # （种类 / 理由 / 预期瓶颈 / 想用的工具）。信封只在这一轮有效 —— 消费点就在上面
        # `current_code = pending_fix` 那一处（与 `pending_fix` 同生同灭），于是它不可能
        # 泄漏到别的轮次。没有信封（生成轮 / 模板轮 / 快照续跑）⇒ `absent`，
        # 动作由 harness 按轮次与路由拼（接线前的行为）。
        _action_envelope_record: dict = {
            "status": ENVELOPE_ABSENT,
            "reason": ("这一轮不是修复轮 —— 只有 Fixer 会交信封"
                       if action != "fix" else "这一次修复没有交出信封 (或快照续跑丢了它)"),
            "used_for_action": False,
            "source": ACTION_SOURCE_HARNESS,
        }
        if _model_envelope is not None:
            _action_envelope_record = dict(_model_envelope.as_dict())
            _action_envelope_record["used_for_action"] = bool(_model_envelope.usable)
            _action_envelope_record["source"] = (
                ACTION_SOURCE_LLM if _model_envelope.usable else ACTION_SOURCE_HARNESS)
        _used_envelope = _model_envelope
        _model_envelope = None       # 一轮只用一次: 下一轮重新由修复轮武装
        _agent_action = self._round_agent_action(
            round_num=round_num, round_action=action, route=_prev_route,
            current_code=state.current_code,
            # 回滚点/ diff 基线: 修复轮是「本轮开始前的那一份」(Fixer 修的就是它,
            # 而它不一定是**已验证**父代 —— 一份从未通过正确性的候选同样可以被修);
            # 其余轮次用已验证父代 (它就是这份新候选要替换掉的那一份)。
            # 两者都没有 ⇒ 空, 于是 `Patch.rollback_source()` 返回 None 而不是空源码。
            parent_code=((_code_before_this_round or state.verified_parent or "")
                         if action == "fix" else (state.verified_parent or "")),
            task=state.task,
            error_shape=(getattr(_prev_route, "error_shape", None)
                         if _prev_route is not None else None),
            envelope=_used_envelope,
        )
        _action_ctx = self._round_action_context(
            route=_prev_route, current_code=state.current_code, test_inputs=state.test_inputs,
            task=state.task, verified_parent=state.verified_parent,
            previous_results=state.all_verification_results or (),
            dep_baseline=self._admitted_baseline,
            # 候选在这个位置**已经生成** ⇒ 钱花过了。预算问题照记, 但不据此拒绝
            # (见 `ActionContext.spend_committed`)。
            spend_committed=True,
        )
        _action_verdict = self.agent_runtime.admit(_agent_action, _action_ctx)
        if _action_verdict.accepted:
            # 过了门才配当下一轮的「新增依赖」基线 (I78): 被拒的那份从没进过流程,
            # 它声明的 import 也就没进过环境 —— 拿它当基线等于把同一份违规候选的
            # 第二次提交放行 (真机实测: 第 3 轮那份夹带 torch 的候选过了门, 真的上了
            # 远端编译, 只有 L2 的反作弊静态审计把它拦住)。
            self._admitted_baseline = state.current_code
            # §22.1 的落盘点：过门的候选**走唯一写入口**写进 `candidate/` 那一格
            # （没有 `run_root` 时是空操作）。它让「§21 判据指着的那个路径」与
            # 「盘上真实存在的产物」是同一个文件 —— 否则权限位保护的是零个产物。
            _materialized = self._materialize_candidate(state.current_code, state.task)
            if _materialized:
                print(f"  [§22.1] 候选已落盘: {_materialized}")
        if not _action_verdict.accepted:
            print(f"\n[§21 Runtime] 动作被拒 (零 GPU 秒): "
                  f"{_action_verdict.refusal_note()}")
            if _action_verdict.fallback_available:
                print("  [§21 Runtime] 回退父代是一条可选出路 (§21.2) —— "
                      "本轮的修复提示会把这一点带上")
            verify_results = [self._refusal_result(_action_verdict)]

        # Step 4: Verifier 验证
        # P1-1 拼接: 候选选择里已经跑过 L1+L2, 这里只补缺的那一段 ——
        # 胜者补 L3; **全败**时那份 L1+L2 本身就是最终结果 (L2 已失败, 不跑 L3),
        # 同时也是喂 Fixer 的失败证据, 不必重算 (上一版丢掉它、重跑一遍真机 L2;
        # 旧注释写的「胜者的 L1 复检走 SHA256 编译缓存」其实不存在 —— 拼接路径
        # 根本不进 L1)。
        # 标定心跳 (设计 §10.5-1): 到期就在**本窗口内**塞一对已知参照核。读数与心跳
        # 同窗, 于是尺子坏了这条读数当场作废, 而不是等下一轮才发现。
        # 放在「已知不跑 L3」的那一支**之后**: 那一支没有真值测量, 塞心跳纯属烧机时。
        # §10.3 分数(c) 的分母: 「预期 GPU 秒」的口径 = **控制端观测到的测量墙钟**。
        # 与 §10.2 的墙钟预算同一个货币 (同一个进程、同一个 `time.monotonic()`),
        # 于是「这一轮把秒换成数字花了多少」与预算拦人的口径是同一个。它不是卡上时间。
        _meas_started = time.monotonic()
        if self._run_binding is not None:
            self._check_run_binding(state.task, state.input_sets)
        if not _action_verdict.accepted:
            # 上一段已经把 `verify_results` 设成那条拒绝结果 —— 这里**不许**再碰
            # 后端 (心跳也不跑: 这一轮没有任何「把时间换成数字」的动作)。
            pass
        elif selected_l1_l2 is not None and l1_l2_is_final:
            print(f"\n[Verifier] 复用候选选择中的 L1+L2 失败结果 (不重跑真机)...")
            verify_results = selected_l1_l2
        else:
            heartbeat, heartbeat_expected = self._heartbeat_for_next_measurement()
            if selected_l1_l2 is not None:
                print(f"\n[Verifier] 执行 L3 验证 (L1+L2 已在候选选择中通过)...")
                verify_results = selected_l1_l2 + self.verifier.verify_l3_only(
                    kernel_code=state.current_code,
                    test_inputs=state.input_sets,
                    target_gflops=state.target_gflops,
                    task=state.task,
                    environment_drift=verify_drift,
                    heartbeat=heartbeat,
                    heartbeat_expected=heartbeat_expected,
                    # 静态守卫的适用性来自**这次胜者**的 L1/L2 结果, 不是 Verifier
                    # 的实例状态 (一个实例连验多个候选, 存字段就是「记录说 A、
                    # 判定按 B」)—— CHECKLOG I65。
                    static_guards=flag_from_metrics(
                        *(r.metrics for r in selected_l1_l2 or [])),
                    # 正确性凭据 (§14-10 / §20.4): 从**这次胜者**的 L2 结果里读 ——
                    # L3 只跑胜者, 所以"胜者是靠正确性选出来的"这件事必须作为数据
                    # 随读数走, 而不是靠"我们没走到这里就说明它对了"的隐含推理。
                    correctness_verified=_correctness_credential(selected_l1_l2),
                )
            else:
                print(f"\n[Verifier] 执行三级验证...")
                verify_results = self.verifier.verify(
                    kernel_code=state.current_code,
                    task=state.task,
                    test_inputs=state.input_sets,
                    expected_output=expected_output,
                    target_gflops=state.target_gflops,
                    environment_drift=verify_drift,
                    heartbeat=heartbeat,
                    heartbeat_expected=heartbeat_expected,
                )
        state.all_verification_results = verify_results
        if self._run_binding is not None:
            self._check_run_binding(state.task, state.input_sets)
            self._bind_results(verify_results, state.current_code)

        # P1-1 成本记账 (续): 非选择路径的 L2 次数直接从这一轮的验证结果里数 ——
        # L2 挂了就早退没有 L3, 所以「结果里有没有 L2」就是「L2 跑没跑」。
        if round_l2_runs is None:
            round_l2_runs = sum(
                1 for r in verify_results
                if r.level is VerificationLevel.L2_FUNCTION)

        all_passed = all(r.passed for r in verify_results)
        # 「这一版被验证过」必须在这一刻、由这一刻的读数记下 —— 等下一轮再回头找就没了:
        # `current_code` 会被 `pending_fix` 覆盖掉 (见上面 `if pending_fix:` 那一支)。
        # 注意这里**不看** L3: 正确性已验证但不够快的核正是要留在手里的那个起点。
        if correctness_verified(verify_results):
            state.verified_parent = state.current_code
            state.verified_parent_round = round_num
        _measure_s = time.monotonic() - _meas_started
        l3_result = next(
            (r for r in verify_results if r.level == VerificationLevel.L3_PERFORMANCE), None,
        )
        # 「真值测量」= 这一轮真的产生了一条 L3 读数。只数它, 不数 L1/L2 —— 心跳
        # 要盯的是「把时间换成数字」的那条通道; 而 L1/L2 失败的轮次不推进计数,
        # 于是同一扇窗的参照对不会被反复重跑 (见 `_heartbeat_ran_for`)。
        if l3_result is not None:
            self._true_measurements += 1
            # §10.2 的记账与这个计数**同点**发生 (一个地方决定「什么算一次真值测量」)。
            # 放在这里而不是放在门那边: 门是「该不该花」, 这是「花了」—— 两者分开才
            # 使得同一次测量不可能出现「记了两次」或「花了没记」。
            # 花的是**这一相**那一格 (§10.3 的分相): 相位由这一轮的候选决定 ——
            # 未观测的候选花「探索」, 已观测的花「精修」。`oracle_spend` 只记账不拦,
            # 计划格子被突破时账本里 `over_plan` 会涨 (与标定超支同一处置)。
            self._budget.oracle_spend(round_phase)

        # 这一轮造成的**永久后果** —— 供设计 §14-9 的资格审计对照实验离线统计。
        #
        # 为什么必须单独记: `measurement_validity` 进血缘的理由是「有多少轮的读数有资格
        # 驱动闭环可以离线统计」, 而**后果**从来没被记过。于是设计 §4.3 那句「资格决定
        # 这个数被允许造成什么后果」在产物侧无法验证 —— 判据算了、后果发生了, 但
        # 「一个不可信的读数有没有真的永久改掉配置 / 覆盖 best」, 出了循环只剩一个最终
        # 的 `disabled_recipes` 集合, 归因不到轮。
        #
        # 值先建 (全 False), 判决分支里改的就是血缘里这一份 —— 没走到判决的轮次因此读作
        # 「没有永久后果」, 而不是缺键 (缺键会被读成「没记」)。
        _perm = {"best_updated": False, "recipes_disabled": [], "stagnation_counted": False}
        # 血缘记录: 每轮动作与结果
        state.artifact.iteration_history.append({
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
            "speedup": (decision_speedup(l3_result.metrics)
                        if l3_result else None),
            # 档位进血缘是为了让「有多少轮的读数有资格驱动闭环」可以离线统计,
            # 而不必重跑一遍才知道那次循环是不是全程都在盲跑
            "measurement_validity": (l3_result.metrics.get("measurement_validity")
                                     if l3_result else None),
            # 资格位 (五位) 也进血缘 —— 上面那段为 `measurement_validity` 写的理由逐字
            # 适用于它, 而且更精确: 档位说「这个数可不可信」, 资格位说「它被允许造成
            # 什么后果」。缺了它, 「有多少轮真的推进了信念 / 有资格写进记忆」这个问句
            # 只能靠重跑估。读者: 设计 §14-9 的资格审计对照实验 (ablation_eligibility)。
            "eligibility": (l3_result.metrics.get("eligibility")
                            if l3_result else None),
            "recipes_applied": list(state.applied_recipes),
            # 永久后果的记账 (上面刚建好那个 dict 的引用; §14-9 的读者)
            "permanent_effects": _perm,
            # P1-1 成本记账: CoderPipeline 翻为默认后, 一个「生成轮」的机时开销变成
            # 「最多 2 个决赛候选各跑一次 L1+L2, 胜者再跑一次 L3」。这两个数不进血缘,
            # 「这次翻转让机时涨了多少」就只能靠重跑一遍估 —— 与上面 `levels_passed`
            # 记的是同一类搬运缺陷 (原料在循环里有效, 出了循环只剩最后一轮)。
            # `generation_candidates` 三态: 0 = 本轮没走 pipeline 生成 (修复轮);
            # 1 = 单候选生成 (没触发选择); 2 = 多候选上了真机 L1+L2 选择。
            "generation_candidates": round_candidates,
            "real_l2_runs": round_l2_runs,
            # 本轮数值门用的哪把尺 (#87 第 2 片)。三态: 报了 / 自己就说不一致 / 没报。
            # 「跨轮的尺是不是同一把」这个问句的原料只能在这里落下 —— 循环之外只剩
            # 最后一轮的 metrics。
            LINEAGE_RULER_KEY: _round_ruler(verify_results),
            # 性能那一半的兄弟格 (#87 第 3 片)。外部那句 𝒮 说的是 "correctness **and**
            # performance criteria fixed across attempts", 只落 L2 那一格等于只问了一半。
            LINEAGE_PERF_RULER_KEY: _round_perf_ruler(verify_results),
            # 执行侧那一格 (#102 第 1 片, **只观测不改判决**): 这一轮的 L2 失败到底是
            # 「远端答了失败」还是「没听到回答」。后者在血缘里第一次有了名字, 于是
            # 「这次远端可能已经跑过一遍」这个问题离线可查, 而不是只活在终端那一行。
            TRANSPORT_KEY: _round_transport(verify_results),
            # 重复率的原料: 签名进 result.json 之后, 「模型重新生成时有多少轮是同一份
            # 结构」这个问题第一次可以被离线复盘, 而不是只能靠当场打印的那一行
            "candidate_signature": sig,
            "duplicate_of": dup_of,
            # §21 的动作与裁决 (**审计与可回滚**的落点): 动作者、理由、patch 的
            # 双向哈希、改了几行, 以及七条独立检查逐条的结论 + 覆盖率。
            # 不进血缘的话, 「Agent 产生的每个 patch 都可审计、可回滚」(Wave 3 的
            # 退出标准) 只能靠重跑一次才知道 —— 而拒绝过的那一轮连代码都不存在。
            "agent_action": _action_verdict.as_dict(),
            # §21.1 **生产者侧**: 这一轮的动作是哪来的（模型交的信封 / harness 拼的）、
            # 信封读没读出来（`parsed` / `absent` / `malformed` + 原因）、模型声明的
            # token 预算（**不进判据** —— 判据读的是 harness 那一份）、以及询问类信封
            # 被怎么响应。缺了它，「动作是谁写的」与「模型自己声明了多少预算」这两件
            # 事在产物里都无法区分 (本仓 I77 那一族)。
            "action_envelope": _action_envelope_record,
            "fix_action": _fix_action_record,
            # §21.3 的**注入留痕**: 这一轮的修复提示里有没有那一段结构化证据、
            # 它有多长、哪几段被截断了、profiler 那一格喂的是哪些键。没有这一格,
            # 「筛过的证据 vs 全量错误日志」这个对照的产物侧无法区分两臂
            # (`injected` 只由构造参数与这一轮有没有走修复共同决定)。
            "fix_evidence": _fix_evidence_record,
            # §12: 这一轮的候选如果是**模板点**产出的, 它的身份 (template_id /
            # instantiator_ver / 参数) 与「怎么选的」(顺位/剪枝/被拒的点) 一起进血缘。
            # 没有这一格, 「这个 tiling 更好」与「这个模板点更好」在产物里长得一样。
            "template_selection": (_template_sel.as_dict()
                                   if _template_sel is not None else None),
            # §10.3 这一轮的选择: 模式 / 抽签值 / 每个候选的分数 / 顺序 / 被计划格子
            # 逼成什么样。模板型轮 `applies_to="this_round"`, 代码型轮是给下一轮选的。
            "selection": _round_selection,
            # §10.4 的另一半: 这一轮的候选空间覆盖到哪了 (无法确立也照记)。引用同一个
            # 会被后填的字典 —— 与上面 `permanent_effects` 同一种写法: 血缘在轮首就
            # append 了, 而这个判定要等本轮的观测进账才算得出来。
            "coverage": _coverage_record,
            # 这一轮的候选花的是哪一格预算 (§10.3 的分相)。与 `selection.mode` 是同一个
            # 事实的两处写法 —— 测试钉住它们相等, 于是「记的相位」不会漂。
            "phase": round_phase,
            # §10.3 分数(c) 分母的原料: 控制端观测到的这一轮测量墙钟 (秒)。
            # `None` = 这一轮没有真值测量 (L1/L2 就挂了), 不是 0。
            "measurement_wall_s": (round(_measure_s, 3) if l3_result is not None else None),
            # 本轮候选属于哪个臂 (§5 的排序器吃的就是它)。注意这一格说的是
            # 「候选是谁」, 不是「这一轮有没有更新统计」—— 后者的读法是
            # `metadata["bandit"].arms[arm]`: `pulls` 涨了才算这一轮的回报进过账,
            # 而 L3 直接通过的那些轮不会进搜索 (所以那种轮里这一格一样有值)。
            "bandit_arm": bandit_arm,
            # 分母的原料 (缺口 3 的「先补分母」那半, doc 49 §八 第 14 条): Verifier 每轮**已经算出** roofline 的全部轴
            # (算术强度、脊点、两侧达成率、峰值出处与可信位), 但它们只活到本次判决结束,
            # 从没进过血缘 —— 于是「可达上界是按 shape/AI 索引的曲线」这件已经量出来的
            # 事实, 任何离线复盘都用不上。整块搬, **不挑键**: doc 48 那次给落盘键做
            # 白名单, 代价是那一批真机数据的血缘永久没了。缺的那块不填默认值。
            **_round_axes(l3_result, state.test_inputs),
        })
        # 这一轮那条记录的引用: 带宽要等**判决发生的那一刻**才知道用没用、用了多少
        # (见下面 `verdict(...)` 那一支), 而这一轮的 dict 已经落在这里 —— 回头按 `-1`
        # 取会依赖「中间没有第二次 append」, 那个前提将来被人破坏时不会响, 只会把
        # 带宽记到别人的轮上。
        _round_rec = state.artifact.iteration_history[-1]

        # Applies to L1/L2 AND L3: unknown benchmark execution is not a slow
        # candidate. Stop before tuning, stagnation, another finalist or learning.
        if any(not r.passed and r.metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN for r in verify_results):
            self._record_stop(state.artifact, round_num, "transport_unknown_hold",
                              "传输结果未知: 不修候选、不重试在飞请求; 先确认 owned runner 静默")
            return _ROUND_STOP
        if any(not r.passed and r.metrics.get("reference_identity_unknown") for r in verify_results):
            self._record_stop(state.artifact, round_num, "reference_identity_hold",
                              "冻结参考身份未确立/已漂移: 不修候选、不搜索、不计进展、不写失败语料")
            return _ROUND_STOP

        # 每轮都算一遍「这一场里各轮的尺是不是同一把」, 但**只在它有话说的时候**打印
        # (与上面 [Novelty] 同一个姿势)。一致时不占日志: 「各轮同尺」是这片要守的
        # 默认前提, 不是成果; 要露出来的只有「变了」与「某一轮自己说不清用了哪把」。
        # 这一句不改任何判决 —— 跨轮换尺今天没有阈可判, 处置与环境漂移同一条
        # (观测先于判定, 见 `measurement.ruler_note`)。
        #
        # 两半各一行, 而且**两句的修法不同** ⇒ 建议也各一条: 数值尺换了几乎一定是部署
        # 中途换过版本, 而 `bound_type` 换了通常只是候选自己换了形状 (那是事实, 不是
        # 待查的漂移)。标签必须带上 —— 不带的话日志里会有两行「各轮的尺一致」而读者
        # 分不清说的是哪一把 (本仓「同一句话有多个副本」那一族的反面: 这里是两个副本
        # 各说不同的事, 却长得一模一样)。
        _ruler_agg = cross_round_ruler(state.artifact.iteration_history)
        if (_ruler_agg["ruler_same"] is False
                or _ruler_agg["ruler_rounds_self_conflict"]):
            print(f"\n[Ruler] {ruler_note(_ruler_agg, RULER_LABEL_CORRECTNESS)}")
        _perf_agg = cross_round_ruler(state.artifact.iteration_history,
                                      keys=PERF_GATE_PARAM_KEYS,
                                      lineage_key=LINEAGE_PERF_RULER_KEY)
        if _perf_agg["ruler_same"] is False:
            print(f"\n[Ruler] {ruler_note(_perf_agg, RULER_LABEL_PERFORMANCE, _PERF_ADVICE)}")

        if all_passed:
            # 标记本轮配方为成功应用
            if state.plan and state.plan.selected_recipes:
                for r in state.plan.selected_recipes:
                    state.applied_recipes.add(r.id)
            print(f"\n[Conductor] 所有验证通过!")
            # ── 可复现性 Harness: 通过前写快照, 方便后续 resume 时直接复用最终轮 ──
            state.snapshot_target_dir = self._snap(state, round_num)
            self._record_stop(state.artifact, round_num, "passed", "所有验证通过 (L1/L2/L3)")

            return _ROUND_STOP

        # 缺口 1: 「数字不好看」不等于「性能未改善」—— 测量本身可能不可信。
        # 污染与 launch 地板支配压低的是数字, 不是核的质量; 拿它禁用配方会
        # **永久**拉黑实际有效的配方, 拿它累计停滞会在 max_stagnant_rounds=2 下
        # 把一次正在起效的优化提前终止。L3 通过时同样要看: 那一路会把本轮配方
        # 记进 applied_recipes, 依据不可信的测量给配方记功是同一个错误的镜像。
        # §10.3 升级门: 本轮的裁决。缺省是**拒绝**(三条件一条都不成立) —— 于是
        # 「这一轮没走到重算」(例如测量不可信、进了别的分支) 与「判过、没通过」
        # 对下游是同一个答案, 而不会因为漏算一次就放行。真正的重算在判决之后。
        _esc_decision = self._denied_escalation()
        # 每一轮门都被问过, **答了什么都要留痕** —— 包括「没走到重算、按缺省拒绝」
        # 的那一轮。不留这一条的话, 读数不可信的轮次在产物里与「门没接上」长得一样
        # (实测: 那几轮原本一条裁决都没有, 于是缺省裁决成了没人读的值)。
        self._escalation_log.append(dict(_esc_decision.as_dict(), round=round_num))
        measurement_ok = True
        if l3_result is not None:
            measurement_ok = self._measurement_drives_loop(l3_result, round_num)
            if not measurement_ok:
                state.invalid_measurement_rounds += 1
            # 读的是 metrics 里那个布尔而不是重算一遍判据: 键没了就说明提供方
            # 没做过出处检查, 此时如实计 0, 不假装数过了。
            if (l3_result.metrics or {}).get("measurement_l3_pass_refused"):
                state.refused_rounds += 1

        # L3 性能改善追踪: 连续无改善时提前终止，避免浪费 LLM token
        if l3_result and l3_result.passed is False and measurement_ok:
            current_gflops = l3_result.metrics.get("gflops", 0)
            # 当 gflops 为 0 (指标计算不可用) 时，用速度比判断
            if current_gflops > 0 and state.best_gflops <= 0:
                # 首个有效观测只能用来建立基线: 没有"之前"可比, 因此既不算改善
                # 也不算停滞, 更不能据此禁用本轮配方 (首轮 plan 是在 roofline=None
                # 下按算子类型选的, 无从判断其好坏)。
                # 缺少这一支时 best_gflops 的初值 0.0 会让首轮掉进下面的"指标不
                # 完整"分支: 把可用的 gflops 报成缺失, 无条件禁用配方, 并在
                # max_stagnant_rounds=2 下把一次正在起效的优化提前终止。
                state.best_gflops = current_gflops
                _perm["best_updated"] = True
                print(f"\n[Conductor] 建立性能基线: {current_gflops:.2f} GFLOPS "
                      f"(speedup={l3_result.metrics.get('speedup_vs_baseline', 0):.4f})")
            elif current_gflops > 0 and state.best_gflops > 0:
                # D4 四态判决。原来这里是无带宽的 `current > best`: 于是「完全
                # 打平」和「差 0.0001%」都会去禁用配方 —— 而禁用是**永久**决策,
                # 需要的证据比「这次没测出更好」强。现在只有 SLOW 才禁用。
                _validity = _validity_of(l3_result.metrics or {})
                v, why, used_band = verdict(current_gflops, state.best_gflops, _validity)
                # §10.3 条件 1 的输入就在**这一个点**产出: 这个分支只在
                # `measurement_ok` 成立时进入 (它要求 can_count_stagnation = VALID),
                # 所以尾巴里天然只有 valid 轮 —— 「K 轮 valid」不需要第二次筛
                # (再筛一次 = 同一个判定两处实现)。
                self._verdict_tail.append(v.name)
                # 这一轮的判决**真正用过**的那条带宽进血缘 (#87 第 4 片, **只观测**)。
                # 取的是 `verdict` 自己返回的那个值, 不是这里再查一次表: 分两次查同一张
                # 表, 中间表被换过就会记下一个「判决没用过的数」, 而这一格要回答的题正是
                # 「判决用过什么」。AMBIGUOUS 那两支没 consult 表 ⇒ 返回 `None`, 键**缺席**
                # 才是「这一轮没到判决」, 两者在 `cross_round_band` 里分开数。
                # 为什么这条要落盘: 带宽换了同时是两件事的读数 (档位降级按表换宽 vs
                # 表本身被改过), 折成一句「尺变了」就把前者报成后者 —— 归因在
                # `measurement.cross_round_band` 里分开做。
                _round_rec[LINEAGE_BAND_KEY] = used_band
                _band_agg = cross_round_band(state.artifact.iteration_history)
                if _band_agg["band_state"] in (BAND_SPEC_DRIFT, BAND_UNEXPLAINED):
                    print(f"\n[Band] {band_note(_band_agg)}")
                if v is Verdict.FAST:
                    state.best_gflops = current_gflops
                    _perm["best_updated"] = True
                    state.rounds_without_improvement = 0
                    if state.plan and state.plan.selected_recipes:
                        for r in state.plan.selected_recipes:
                            state.applied_recipes.add(r.id)
                    print(f"\n[Conductor] 性能改善 ({current_gflops:.2f} GFLOPS, {why})")
                else:
                    if v is Verdict.SLOW:
                        # 只有 SLOW 才禁用配方 —— SAME 与 AMBIGUOUS 都是「没有证据」,
                        # 而禁用是永久决策。进到这里时 measurement_ok 已经成立
                        # (它要求 can_count_stagnation, 也就是 VALID), 所以不必再
                        # 检查 can_disable_recipe: 那会是永不生效的条件。
                        new_recipes = [r for r in state.plan.selected_recipes
                                       if r.id not in state.applied_recipes] if state.plan else []
                        for r in new_recipes:
                            state.disabled_recipes.add(r.id)
                            _perm["recipes_disabled"].append(r.id)
                        if new_recipes:
                            disabled_names = ", ".join(r.id for r in new_recipes)
                            print(f"  [Planner] 性能退步，禁用配方: {disabled_names}")
                        print(f"\n[Conductor] 性能退步 ({current_gflops:.2f} < {state.best_gflops:.2f} "
                              f"GFLOPS, {why}), 连续 {state.rounds_without_improvement + 1}/{max_stagnant_rounds} 轮")
                    else:
                        print(f"\n[Conductor] 性能未改善 ({current_gflops:.2f} vs {state.best_gflops:.2f} "
                              f"GFLOPS, {why}), 连续 {state.rounds_without_improvement + 1}/{max_stagnant_rounds} 轮")
                    state.rounds_without_improvement += 1
                    _perm["stagnation_counted"] = True
            else:
                # gflops 不可用: 用速度比或时间判断
                speedup = (decision_speedup(l3_result.metrics) or 0)
                # 首轮同样无基线可比, 不禁用配方 (与上面 gflops 分支同理)
                if round_num < 2 or (speedup > 0 and speedup < 1.0):
                    # 无 gflops 参考时禁用配方等于拿噪声做永久决策，跳过避免死锁
                    pass
                elif state.plan and state.plan.selected_recipes:
                    new_recipes = [r for r in state.plan.selected_recipes if r.id not in state.applied_recipes]
                    for r in new_recipes:
                        state.disabled_recipes.add(r.id)
                        _perm["recipes_disabled"].append(r.id)
                state.rounds_without_improvement += 1
                _perm["stagnation_counted"] = True
                print(f"\n[Conductor] 性能指标不完整 (gflops={current_gflops:.2f}, speedup={speedup:.4f}), "
                      f"连续 {state.rounds_without_improvement}/{max_stagnant_rounds} 轮")
            # ── §10.3 的三条件升级门 (判据的输入刚刚齐: 本轮判决已进尾巴) ──
            # 位置是判据的一部分: 必须在**停滞出口之前**。第一版把它放在优化分支里,
            # 于是「记满 K 轮平坦」的那一轮先被停滞出口 break 掉, 门永远开不了
            # (实测: K=2 时门能看到的尾巴最多只有 1 条)。
            _esc_decision = self._escalation_decision(
                state.plan, state.applied_recipes, state.disabled_recipes,
                (l3_result.metrics or {}) if l3_result else {}, state.current_code, state.task)
            # **替换**而不是追加: 这一轮的缺省裁决已经记过一条, 同一轮记两条会让
            # 「判过几次」变成「轮数 × 分支数」, 而读者要的是每轮一个答案。
            self._escalation_log[-1] = dict(_esc_decision.as_dict(), round=round_num)
            if not _esc_decision.allowed:
                print(f"\n[Escalation] 结构升级门未通过 ⇒ 本轮只做参数搜索: "
                      f"{_esc_decision.refusal_note()}")
            elif state.rounds_without_improvement >= max_stagnant_rounds:
                # 升级**买到的是 K 轮**: 平台化是对**旧结构**的陈述, 换成新结构之后
                # 那段平坦区不再是关于它的证据 ⇒ 停滞计数清零。有界性靠
                # `max_optimize_rounds`(总轮次) 与「清零后要重新攒够 K 轮 valid」
                # 两条一起保证, 不会因为门反复放行而无限跑。
                print("\n[Escalation] 结构升级门通过 ⇒ 停滞计数清零, "
                      "本轮把结构改动交给 LLM")
                state.rounds_without_improvement = 0
            # ── §10.4 可靠停滞 = 两个条件**同时**成立 ──
            # 前半句 (K 轮 valid 无改善) 上面刚记完; 后半句是「有希望的区域已穷尽」。
            # 位置: 停滞出口之前 —— 判据的输入是本轮候选全集与 bandit 的观测, 两者都
            # 已经定下来了。
            _coverage = (assess_coverage(_universe.kept, state.bandit, template_arm,
                                         label=point_label, universe=FROM_TEMPLATE_SPACE)
                         if _universe is not None
                         else undeclared(
                             FROM_LLM_ON_DEMAND,
                             "代码型路径的候选由 LLM/规则按需产生, 没有可列举的全集"))
            _threshold = state.rounds_without_improvement >= max_stagnant_rounds
            _coverage_record.update(_coverage.as_dict(),
                                    threshold_reached=_threshold)
            state.coverage_ledger.record(_coverage, threshold_reached=_threshold)
            if _threshold and not _coverage.exhausted:
                # 「只因轮数」不许停 (设计 §10.4 的原话)。停不成的原因要留痕: 没有这一格,
                # 「因为没穷尽所以继续跑」与「停滞判据根本没人读」在产物里长得一样。
                print(f"\n[Conductor] 连续 {max_stagnant_rounds} 轮无性能改善, 但"
                      f"{_coverage.note()}")
            if _threshold and _coverage.exhausted:
                print(f"\n[Conductor] 连续 {max_stagnant_rounds} 轮无性能改善，"
                      f"{_coverage.note()} ⇒ 提前终止优化")
                state.snapshot_target_dir = self._snap(state, round_num)
                # 停机理由要**点名两半**都成立 (只有前一半的话, 那正是设计点名的
                # 错读法): 「连续无改善」+「候选区域已穷尽 (哪个集合, 几个点)」。
                _why = (f"连续无改善达上限且候选区域已穷尽 "
                        f"({_coverage.universe} {_coverage.tried}/{_coverage.total})")
                self._record_stop(state.artifact, round_num, "stagnant", _why)

                return _ROUND_STOP

        # 缺口 4 的正解: 改善判据该看「离 参考天花板还有多远」, 而不是继续拿
        # 一个标量比大小。绑定轴已经贴着上界的核, 换任何配方都不可能有可测收益 ——
        # 再跑一轮只是白烧 GPU 与 LLM token。
        #
        # 只判「本轮有没有 L3 结果」: 再往前 `all_passed` 已经 break, 而
        # `Verifier.verify()` 在 L1/L2 失败时直接 return (verifier.py:158-175),
        # 所以拿得到 L3 就意味着 L1/L2 过了、且 L3 没过。在这里再写
        # `not l3_result.passed` 或 `not l1_l2_failed` 都是永不生效的条件
        # (同一文件里刚为 `can_disable_recipe` 删掉过一个)。
        if l3_result is not None and reference_ceiling_reached(l3_result.metrics or {}):
            _att = binding_attainment(l3_result.metrics or {})
            _rl = (l3_result.metrics or {}).get("roofline_analysis") or {}
            print(f"\n[Conductor] 绑定轴已达 roofline {_att:.1f}% "
                  f"(>= {REFERENCE_CEILING_PCT:.0f}%), 判定已收敛, 停止后续优化轮")
            # 这条 break 是整个循环里最贵的一次判定 (它决定不再花轮次), 所以把它
            # 依赖的分母连同出处一起落进日志: 以后复查时不需要回头猜 90% 是相对
            # 哪个数算出来的、那个数凭什么。
            print(f"[Conductor]   分母: [{_rl.get('peak_table_key', '?')}] "
                  f"peak={_rl.get('hw_peak_gflops', 0) / 1000:.1f} TFLOPS / "
                  f"bw={_rl.get('hw_peak_bandwidth_gbs', 0):.0f} GB/s "
                  f"({_rl.get('compute_used', '?')}) — {_rl.get('peak_source', '出处未知')}")
            state.snapshot_target_dir = self._snap(state, round_num)
            self._record_stop(state.artifact, round_num, "converged", "绑定轴已达 参考天花板")

            return _ROUND_STOP

        l1_l2_failed = any(
            not r.passed and r.level in (VerificationLevel.L1_COMPILE, VerificationLevel.L2_FUNCTION)
            for r in verify_results
        )
        if l1_l2_failed:
            # §8 路由: 先分类再动作。性能不达标≠bug (走搜索), 基础设施≠进度,
            # 编译过、仅数值错 → **修不重写** (§19.3, 外部实测省 92.8% token)。
            from .failure_route import ROUTE_DEBUG_FIX, ROUTE_HOLD, ROUTE_RETRY, classify
            route = classify(verify_results)
            print(f"  [Route] §8 分类: kind={route.kind} route={route.route}"
                  f"{' · fix_not_rewrite' if route.fix_not_rewrite else ''}"
                  f" — {route.reason}")
            if route.route == ROUTE_RETRY:
                # 基础设施失败: 退避重试, 不计进度、不进负样本库 (§8 表)
                self._record_stop(state.artifact, round_num, "infra_retry",
                                  route.reason)
                return _ROUND_STOP
            # ROUTE_HOLD 仍走 Fixer (既有控制流: 「那一轮仍然走 Fixer」),
            # 只是下面的写入侧按 reference_hold 跳过沉淀 —— 不在这里 continue。
            if round_num < self.max_optimize_rounds:
                # L1 编译失败可能是缺依赖: 分析代码 import 并按需补装
                l1_result = next((r for r in verify_results
                                  if r.level == VerificationLevel.L1_COMPILE and not r.passed), None)
                # §21 拦下的候选**根本没被送进编译**, 为它跑一次远端依赖分析
                # (SSH 往返) 毫无意义 —— 那正是「不进编译」想省的机时。
                if l1_result and not (l1_result.metrics or {}).get("agent_action_refused") \
                        and Cap.DEP_ANALYSIS in self.backend.capabilities:
                    deps = self.backend.analyze_dependencies(state.current_code)
                    if deps:
                        print(f"\n[Conductor] 代码依赖: {deps}, 尝试按需补装...")
                        if Cap.ENV_PROVISION in self.backend.capabilities:
                            self.backend.ensure_environment(required_pips=deps)

                failed_results = [r for r in verify_results if not r.passed]
                # 「本级判不出」的那一轮对这张核没有作出任何判定 (参考自己产 NaN/Inf)。
                # 它仍然走 Fixer —— 控制流这一片不动, 省一轮 LLM 是另一次要带 A/B 的改动
                # (doc 58 §4.2 剩下的那半)。这里先接的是**写入侧**: 那一轮的 Reflexion
                # 会写出一份指向错误对象的教训, 而那份教训在同任务的后续轮次里被检索到。
                reference_hold = failure_class(failed_results) == FAILURE_CLASS_REFERENCE
                if reference_hold:
                    print("\n[Conductor] 本轮失败归为「本级判不出」(参考侧没有可比对象): "
                          "不沉淀教训、不进负样本库")
                print(f"\n[Fixer] L1/L2 验证失败，启动错误修复...")
                # 修订起点 (arXiv:2609.35140v1 行 214)。有已验证的祖先就不从这一版出发:
                # 它没通过正确性验证, 在它上面继续叠就是「在坏的后代上盖一层」—— 那正是
                # 这句规则要挡的漂移。但**错误信息仍然取这一版的**: 那是「上一次改动坏在哪」
                # 的唯一证据, 换成祖先的错误信息就什么都修不了。
                # `!= current_code` 那半句管的是同一份文本这次没过 (验证非确定: 共享卡、
                # 容差边界) 的场合 —— 那时起点没变, 少了它就印一句「退回」的谎话。
                # 它的证人是 `tests/test_verified_parent_retreat.py` 里用假 Verifier 造的
                # 那个现场; 没有那一条时这个条件构造不出反例, 就会被当成死文字删掉 (M4)。
                repair_base = state.current_code
                if state.verified_parent is not None and state.verified_parent != state.current_code:
                    repair_base = state.verified_parent
                    print(f"  [Conductor] 修订起点退回第 {state.verified_parent_round} 轮那份 "
                          f"L1/L2 已验证的实现 (本轮这一版没有通过正确性验证)")
                try:
                    # §21.3: 结构化证据那一段 (**唯一拼装点** = `_fix_evidence`),
                    # 默认开。关掉时传空串 ⇒ 只少那一段（其余段两臂同值，见构造参数注释）。
                    _evidence = ""
                    if self.structured_evidence:
                        _evidence, _rec = self._fix_evidence(
                            task=state.task, route=route, verify_results=failed_results,
                            current_code=state.current_code,
                            parent_code=_code_before_this_round,
                            gate_ctx=_action_ctx,
                        )
                        _fix_evidence_record.update(_rec)
                    # §21.1 生产者侧: 修复这一步现在**同时**取回模型交上来的动作信封。
                    # `run` 保留给不关心信封的调用方, 而这里用的是带信封的那一个 ——
                    # 于是「动作是谁写的」在内侧就分得开。
                    _outcome = self.fixer.run_with_action(
                        repair_base, state.task, failed_results,
                        fix_not_rewrite=route.fix_not_rewrite,
                        structured_evidence=_evidence,
                        error_shape=(getattr(route, "error_shape", "") or ""),
                    )
                    if _outcome.asked:
                        # 模型交的是**不改候选**的意图 (解释 / 请求观测): 这一轮没有候选。
                        # 响应它一次 (答或拒), 再要一次补丁 —— 有界: 第二次还是提问就停。
                        # 「一直问」不设上限就是一条白烧 token 的环 (§21.1 的预算账)。
                        _outcome = self._answer_ask_and_retry(
                            _outcome, repair_base=repair_base, task=state.task,
                            failed_results=failed_results, route=route,
                            evidence=_evidence,
                            record=_fix_action_record,
                        )
                    fixed_code, fix_note = _outcome.code, _outcome.summary
                    # 信封交给**下一轮**的判据点 (它就是这份候选的来路)。只有改候选的
                    # 那几类值得传（提问类在上面已经被响应掉了）。
                    self._pending_action_envelope = (
                        _outcome.envelope if _outcome.envelope.usable else None)
                    _fix_action_record.update(_outcome.as_dict())
                except Exception as e:
                    print(f"\n[Conductor] Fixer 修复失败 ({e})，停止迭代")
                    state.snapshot_target_dir = self._flush_snapshot(
                        task=state.task,
                        artifact_dir=state.snapshot_target_dir,
                        round_done=round_num,
                        current_sketch=state.current_sketch,
                        current_code=state.current_code,
                        pending_fix=None,
                        iteration_history=state.artifact.iteration_history,
                        best_gflops=state.best_gflops,
                        rounds_without_improvement=state.rounds_without_improvement,
                        optimization_history=state.optimization_history,
                        applied_recipes=state.applied_recipes,
                        disabled_recipes=state.disabled_recipes,
                        all_verification_results=state.all_verification_results,
                    )
                    self._record_stop(state.artifact, round_num, "fix_error", "Fixer 修复抛异常, 停止迭代")

                    return _ROUND_STOP
                if not _outcome.code:
                    # 响应过、又要过一次补丁，第二次**还是**提问 ⇒ 这一轮没有候选。
                    # 停（不是「拿上一次的代码继续跑」）：模型已经明说它不改候选，
                    # 再拿旧候选上去验证就是 harness 替它改了它的声明。
                    print(f"\n[Conductor] 连续两次收到「不改候选」的动作信封 "
                          f"({_outcome.envelope.kind})，本轮没有候选 —— 停止迭代")
                    self._pending_action_envelope = None
                    # 「紧邻 break 的三行内必须有 _record_stop」是 `test_stop_reason`
                    # 的计数守卫 —— 详情写多行会把它顶出窗口, 于是那条守卫就漏了这一处。
                    self._record_stop(state.artifact, round_num, "ask_turns_exhausted",
                                      f"连续两次收到不改候选的意图 "
                                      f"({_outcome.envelope.kind})，响应后仍无补丁")
                    return _ROUND_STOP
                print(f"  修复说明: {fix_note}")
                state.pending_fix = fixed_code

                # ---- Reflexion: 修复后让 Fixer 产出结构化教训并沉淀 ----
                reflexion = None
                try:
                    # 判不出那一轮不产 Reflexion: 那一步是让模型回答「这次失败的根因
                    # 是什么、下次怎么避免」, 而这一轮根本没有失败可归因 —— 让它回答就是
                    # 让它编一个候选侧的根因, 再把那份编出来的教训 `add()` 进语料。
                    if not reference_hold:
                        reflexion = self.fixer.produce_reflexion()
                    if reflexion is not None:
                        self.failure_db.add(reflexion)
                        state.reflexion_pending.append(reflexion)
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
                state.fix_attempts.append((failed_results, fix_note, reflexion))

                # ── 可复现性 Harness: Fixer 修复后走 continue 前也要落快照 (否则 crash 就丢了修复代码) ──
                state.snapshot_target_dir = self._flush_snapshot(
                    task=state.task,
                    artifact_dir=state.snapshot_target_dir,
                    round_done=round_num,
                    current_sketch=state.current_sketch,
                    current_code=state.current_code,
                    pending_fix=state.pending_fix,
                    iteration_history=state.artifact.iteration_history,
                    best_gflops=state.best_gflops,
                    rounds_without_improvement=state.rounds_without_improvement,
                    optimization_history=state.optimization_history,
                    applied_recipes=state.applied_recipes,
                    disabled_recipes=state.disabled_recipes,
                    all_verification_results=state.all_verification_results,
                )
                return _ROUND_CONTINUE
            print(f"\n[Conductor] L1/L2 验证失败且已达最大轮次，停止迭代")
            state.snapshot_target_dir = self._snap(state, round_num)
            self._record_stop(state.artifact, round_num, "l1_l2_failed", "L1/L2 失败且已达最大轮次")

            return _ROUND_STOP

        # ── 收尾: 参考天花板 / 停滞 / 资格门 / 出口判定 (§10.4), 见 _round_close_out ──
        return self._round_close_out(
            state, round_num,
            _esc_decision=_esc_decision, _measure_s=_measure_s, _space=_space,
            bandit_arm=bandit_arm, l3_result=l3_result)

    def _finalize_run(self, *, state):
        """收尾: 把跑完的循环状态落成最终产物、写报告、给出返回值。

        从 `_execute_inner` 拆出来的第一段 (1620 -> 1448 行)。**为什么先拆这一段**:
        它有一个单出口 (`return artifact`)、不写任何 `self.` 属性、也不依赖外层循环的
        控制流 (区内的 `continue` 绑的是区内自己的循环) —— 所以能用「逐字搬 + 显式
        接线」完成, 循环内部一行没动, 风险被限制在尾部。

        ⚠️ 接口宽 (24 个关键字) 是这里**真实的**耦合, 不是随手摊开的参数表: 收尾确实要
        读这次运行攒下的全部东西, `_restore_state_from_snapshot` 那 11 个「快照状态」
        只是它的一个子集。正确的收敛方向是把它收成一个运行状态对象, 但那要求 1128 行的
        轮次循环也改成从那个对象读 —— 那是下一步, 不在这一版里。这一版刻意**不**动循环,
        为的是让「行为是否变了」只由 `tests/test_execute_inner_behavior.py` 一个判据回答。
        """

        # 循环正常走到末尾 (没被任何 break 收走) —— 这一档以前只在 stdout 上有一行 print,
        # 产物里与「修坏了」「到顶了」完全同形。补记成数据。
        if not (state.artifact.metadata or {}).get("stop"):
            self._record_stop(state.artifact, self.max_optimize_rounds, "rounds_exhausted",
                              "轮次用完仍未通过, 也没有触发任何提前出口")

        # 构建最终产物
        state.artifact.code = state.current_code
        state.artifact.sketch = state.current_sketch
        state.artifact.verification_results = state.all_verification_results

        if state.all_verification_results:
            l3 = next((r for r in state.all_verification_results if r.level == VerificationLevel.L3_PERFORMANCE), None)
            if l3:
                state.artifact.benchmark_data = l3.metrics
                from .contract import case_key

                # §6.2 条款 2/4 的第一件证据：**本次评的是哪一条 case**（输入形状签名，
                # `contract.case_key`）。写在这里的理由：这是全仓唯一一处「本轮的 L3 读数」
                # 与「本轮的输入」同时在手的地方 —— 判据侧（`_save_to_db` 的两条证据）
                # 读的就是这一格。没有输入 ⇒ 空串 = **没有 case**（判据按未确立走），
                # 不拿别的键冒充（见 I92）。
                state.artifact.benchmark_data["case_key"] = case_key(state.test_inputs)
                # §3.4（I94）: **输入从哪来** + 「契约声明的 case」与「本次评的那几组」对不对
                # 得上。三态: `case_source` 空串 = 契约没声明具体输入（老契约，输入只能由
                # 调用方给）; `case_match` 只在契约声明了 case 时才有这一格（没声明 ⇒ 不写，
                # 免得「没得对账」被读成「对上了」）。`case_declared` = 契约手里有几组具体输入。
                state.artifact.benchmark_data["case_source"] = state.case_source
                if state.declared_cases:
                    evaluated = {case_key(item) for item in state.input_sets}
                    state.artifact.benchmark_data["case_declared"] = len(state.declared_cases)
                    state.artifact.benchmark_data["case_match"] = bool(
                        evaluated and evaluated <= set(state.declared_cases))

            # 提取编译元信息 (Triton: num_warps, num_stages 等)
            l1 = next((r for r in state.all_verification_results if r.level == VerificationLevel.L1_COMPILE), None)
            if l1 and l1.metrics:
                state.artifact.compile_info = {
                    k: v for k, v in l1.metrics.items()
                    if k not in ("compile_status", "backend")
                }
            # 从 task 配置补充 Triton 编译参数
            if state.task.target.triton_config:
                state.artifact.compile_info.update(state.task.target.triton_config.to_dict())

        # 失败案例入库: 修复记录与最终失败均沉淀为负样本 (供后续 Fixer 检索)
        # 循环内只沉淀 Reflexion 那一条 (带 lessons 结构化教训), 且当时 `fixed` 只能是
        # 「还没验证过」; 这里补齐两件事: 没有 Reflexion 记录的修复走兜底入库,
        # 有 Reflexion 记录的在结果已知后补写一次。
        if state.fix_attempts:
            for failed_results, fix_note, reflexion in state.fix_attempts:
                if reflexion is not None:
                    continue      # 这条失败已经有它自己的记录 (下面那次补写会更新它)
                self._record_failure(state.task, failed_results, fix_note, fixed=state.artifact.all_passed)
        elif state.all_verification_results and not state.artifact.all_passed:
            failed = [r for r in state.all_verification_results if not r.passed]
            if any(r.level in (VerificationLevel.L1_COMPILE, VerificationLevel.L2_FUNCTION)
                   for r in failed):
                self._record_failure(state.task, failed, "", fixed=False)

        if state.artifact.all_passed:
            # `fixed` 的字段注释是「该失败**最终**是否被修复」, 而 produce_reflexion() 被调用的
            # 那一刻下一轮验证还没跑 —— 原先它无条件写 True, 于是每条反思都在结果未知时替结果
            # 写了答案, 且修的到底是哪一次也没人核对过。现在创建时留默认 False, 到这里才补写。
            # 追加式日志里同一条 id 的 fixed=False → fixed=True 演进是**设计支持的**
            # (kernel_db.py:806-812: 重放按后写覆盖先写), 所以这不是新增机制, 是用它本来的用法。
            for reflexion in state.reflexion_pending:
                reflexion.fixed = True
                self.failure_db.add(reflexion)
            if state.reflexion_pending:
                print(f"\n[Conductor] Reflexion 记录补写 fixed=True ({len(state.reflexion_pending)} 条)")

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
                if state.artifact.metadata is None:
                    state.artifact.metadata = {}
                state.artifact.metadata["llm_cost"] = cost_summary

        # 保存产物到文件系统
        saved_dir = self.store.save(state.artifact, state.task)
        # Artifact 上记录一下落盘目录, 方便下游 (golden harness / 日志) 直接引用
        try:
            state.artifact.saved_dir = str(saved_dir)
        except Exception:
            pass
        # 保存后用权威 artifact 目录刷新 snapshot_target_dir, 并补写一次最终快照
        # (这样 resume_from=saved_dir 就能读到 execution_snapshot.json)
        snapshot_target_dir = saved_dir
        self._flush_snapshot(
            task=state.task,
            artifact_dir=snapshot_target_dir,
            round_done=max(
                (h.get("round") or 0) for h in state.artifact.iteration_history
            ) if state.artifact.iteration_history else 0,
            current_sketch=state.current_sketch,
            current_code=state.current_code,
            pending_fix=state.pending_fix,  # Last-round repair is still resumable.
            iteration_history=state.artifact.iteration_history,
            best_gflops=state.best_gflops,
            rounds_without_improvement=state.rounds_without_improvement,
            optimization_history=state.optimization_history,
            applied_recipes=state.applied_recipes,
            disabled_recipes=state.disabled_recipes,
            all_verification_results=state.all_verification_results,
        )
        print(f"\n[Conductor] 产物已保存到: {saved_dir}")

        # ── 成本监控 Harness: 打印账本摘要 (save 之后打, 不影响持久化时序) ──
        if self.llm and hasattr(self.llm, 'cost_tracker') and self.llm.cost_tracker.total_calls > 0:
            print(f"\n[Conductor] {self.llm.cost_tracker.pretty()}")

        if state.artifact.all_passed:
            # 永久结论量两遍 (设计 §10.5-4 / §3.5 的**不变量**, 不是开关): 只有真会进语料
            # 的这一支做重测, 而这一支**一定**做 —— 曾经它挂在一个默认 `False` 的构造参数
            # 后面, 于是默认配置下单次读数就能入库 (I79: 「默认关的安全装置等于不存在」)。
            # 生产者与消费者 (下面 `_save_to_db` 里那道门) 成对存在, 所以无条件要求不是
            # 「一律拒」: 先补一次独立重测, 再由判据读它。
            self._produce_remeasure(state.artifact, state.task, state.input_sets, state.target_gflops)
            if self._run_binding is not None:
                self._check_run_binding(state.task, state.input_sets)
            self._save_to_db(state.artifact, state.task)
        elif state.artifact.benchmark_data:
            self._publish_decision(state.artifact, state.task)
            from .corpus_admission import publish
            state.artifact.metadata["corpus_admission"] = dict(kept=False, status="not_qualified",
                reason="L3 workload/correctness gate did not pass")
            publish(state.artifact.saved_dir, state.artifact.metadata["corpus_admission"],
                    state.artifact.benchmark_data["claim_level"], state.artifact.benchmark_data)

        # ── 报告产物 (设计 §17 第 3 条) ──
        # 位置是刻意的: **必须在 `store.save()` 与 `_save_to_db()` 之后**。前者给出产物
        # 目录, 后者落进「这条结论有没有进长期记忆 + 为什么」—— 那是报告里最要紧的一句
        # 之一, 而它此刻才存在。报告写失败不让这次实验失败 (write_report 内部把异常写进
        # 那个文件), 所以这里不需要再兜一层。
        # bandit 状态落盘: 「有几轮的数进了统计」这件事必须能在产物里读到, 否则
        # 排序就成了一个没有出处的黑箱 (与 `selection_rejections`/`stop` 同一处置)。
        if state.artifact.metadata is None:
            state.artifact.metadata = {}
        state.artifact.metadata["bandit"] = state.bandit.as_dict()
        # §10.3 的选择策略: ε 与抽签种子落盘, 于是逐轮那个抽签值可以**离线复算**
        # (`selection.explore_draw(seed, round)`), 而不必把每个随机数都记下来。
        # §10.4: 「有希望的区域已穷尽」的逐轮判定 (含被拒绝的停滞 —— 那正是它的观察面)
        state.artifact.metadata["coverage"] = state.coverage_ledger.as_dict()
        state.artifact.metadata["selection_policy"] = {
            "epsilon": self.selection_epsilon, "seed": state._selection_seed,
            "rule": "score(c) = P(beats_best) × 预期增益 / 预期测量秒",
            "seconds_source": "控制端测量墙钟 (§10.2 墙钟预算同一个货币, 不是卡上时间)",
        }
        # §10.2 的账本同样落盘: 「计划是多少 / 花了多少 / 谁被挡住了」必须能被读到,
        # 否则预算就是一个只在 `print` 里活过的装置。
        state.artifact.metadata["budget"] = self._budget.as_dict()
        # §10.3 的逐轮裁决: 门是**判据**, 判过什么必须留痕 (否则「没升级」与
        # 「没接门」在下游看起来一样)。
        state.artifact.metadata["escalation"] = self._escalation_log

        from .report import write_report
        _admission = (state.artifact.metadata or {}).get("corpus_admission") or {}
        _refusals = ([_admission.get("reason")]
                     if _admission.get("kept") is False and _admission.get("reason")
                     else None)
        write_report(saved_dir, artifact=state.artifact, task=state.task, refusals=_refusals)
        print(f"[Conductor] 报告已写出: {saved_dir}/report.md")

        # 每次运行都在日志里报一句语料现状。这条的存在理由是缺口 2 的失效方式本身:
        # 「RAG 命中 N 个相似案例」以前只说明语料里有 N 条, 不说明它们是不是真的
        # 从上一次运行继承下来的 (答案曾经是「不是」)。
        print(f"[KernelDB] {self.kernel_db.summary()} "
              f"本次取用 {self.kernel_db.retrievals_this_run} 次命中")
        # 负样本侧同样每轮报一句。这条的读者是**复发门到底挡了多少**: 语料在长、门在不在
        # 起作用, 不看这一句就只能靠读代码相信它存在 (doc 45 第二节的失效方式)。
        print(f"[FailureDB] {self.failure_db.summary()} "
              f"本次被复发门挡住 {len(self.failure_db.suppressed_by_gate)} 条"
              f"·被类别门挡住 {len(self.failure_db.suppressed_by_kind)} 条"
              f"·参考侧判不出 {self._reference_hold_rounds} 轮 (没入库)")
        # 候选新颖度: 现在只有「重复了几轮」这一个数, 它的读者是 doc 47 里那条基率 ——
        # 在拿签名去省 GPU 时间之前, 先得知道重复到底常不常见。
        n_rounds = len(state.artifact.iteration_history)
        print(f"[Novelty] {n_rounds} 轮候选, {len({s for _, s in state.round_signatures})} 个不同结构, "
              f"其中 {state.duplicate_rounds} 轮与更早的轮次结构相同")

        if state.invalid_measurement_rounds:
            # 括号里以前写死「污染 / launch 地板支配」两项, 而判据已经有五项 (还有
            # 样本离散度、数字的出处、环境漂移)。列举会过期, 逐轮的原因本来就在上面
            # 打过一遍, 这里只报数。
            print(f"\n[Conductor] ⚠️ 本次 {state.invalid_measurement_rounds} 轮的 L3 测量被判为不可信 "
                  f"(逐轮原因见上), 这些轮次未参与配方禁用与停滞计数")

        if state.refused_rounds:
            print(f"\n[Conductor] ⛔ 本次 {state.refused_rounds} 轮的 L3 因**出处不对**而不记为通过 "
                  f"(任务声明了执行侧没有的能力) —— 这些轮不是「核太慢」, 是压根没测过"
                  f"一个可判定的东西; 它们烧了 LLM 与远程编译, 没产出关于这台机器的证据")

        self._print_summary(state.artifact, state.task, saved_dir)

        # 执行结束后打印实际使用的 LLM 模型 (可能已降级)
        if self.llm and hasattr(self.llm, '_fallback_triggered') and self.llm._fallback_triggered:
            print(f"\n[Conductor] ⚠️ 本次执行已降级: 主模型 [{self.llm.model}] 失败, "
                  f"实际使用 [{self.llm._active_model}] @ {self.llm._active_base_url}")
        return state.artifact


    def _plan_phases(self):
        """§10.3 的「计划约束选择」: 调 `selection.choose` 时该传什么**预算相位**。

        两件事在这里定死, 免得两个调用点各解释一遍:

        * 还有计划格子 ⇒ 传 `("explore", ...)` 那一族,**计划就真的限制选择**;
        * 计划格子**一个都没有** ⇒ 传 `None` = **不拿计划当约束**, 而不是传空元组。
          两者不同: 空元组在 `choose` 里意为「没有可选的相位 ⇒ 不许选」, 而这里要说的是
          「这份计划没有可分配的空间」(总数太小, 5%/55%/30% 被 `int()` 抹平成 0)。那时硬
          边界只剩总数与预留那一条, 超计划由账本 `over_plan` 记账 —— 不假装有计划。
        """
        return self._budget.phases_with_room() or None

    @staticmethod
    def _arm_of(sketch) -> str:
        """bandit 的臂 = tiling 三元组, 写法与 `optimization_history`/血缘里那个一致。

        为什么用 tiling 而不是候选签名: 搜索真正要选的**就是** tiling (其余结构由 LLM
        先验与配方决定), 拿签名当臂会把「同一个 tiling 被不同先验包装」拆成两个臂 ——
        而我们要问的问题是「这个 tiling 上次有没有变快」。
        """
        t = getattr(sketch, "tiling", None)
        if t is None:
            return "unknown"
        return f"{t.block_m}x{t.block_n}x{t.block_k}"

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

    def _check_run_binding(self, task, inputs):
        try:
            run_contract.check_current(self._run_binding, task, inputs,
                                       self._environment["fingerprint"], self._run_controls)
            run_contract.write(self._run_binding_dir, self._run_binding)
        except (OSError, ValueError, TypeError) as exc:
            raise run_contract.RunBindingError(str(exc)) from exc

    def _bind_results(self, results, code):
        m = self._run_binding["manifest"]
        import uuid
        verification_id = uuid.uuid4().hex
        for result in results:
            result.metrics["run_binding"] = {"run_id": m["run_id"],
                "manifest_hash": m["hash"], "contract_hash": m["contract_hash"],
                "workload_hash": m["workload_hash"],
                "protocol_hash": run_contract.digest(m["measurement_protocol"]),
                "candidate_hash": run_contract.code_hash(code),
                "verification_id": verification_id}

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
            try:
                key = (int(data.get("round_done") or 0), path.stat().st_mtime)
            except (OSError, ValueError, TypeError):
                scan["unreadable"] += 1
                continue
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
            if self._run_binding is not None:
                self._check_run_binding(task, self._run_inputs)
                run_contract.write(target_dir, self._run_binding)
                snap = run_contract.audit_value(snap)
                run_contract.seal(snap, self._run_binding)
                run_contract.atomic_json(path, snap)
                return target_dir
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
            if self._run_binding is not None:
                raise run_contract.RunBindingError("checkpoint write failed: " + type(e).__name__) from e
            # 快照写入失败绝不能打断主任务 — Harness 是旁路
            print(f"  [Snapshot] ⚠️ 快照写入失败, 忽略: {type(e).__name__}: {e}")
            return artifact_dir

    @staticmethod
    def _load_snapshot(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("snapshot must be a JSON object")
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

    def _produce_remeasure(self, artifact, task, input_sets, target_gflops: float) -> None:
        """给「即将进长期记忆」的结论补一次**独立重测** (设计 §10.2 末 10% 预算 / §10.5-4)。

        为什么在这里、而不是每轮都测: 只有 `all_passed` 的产物会进语料, 而确认的对象
        正是「进语料」这个动作。对每一轮都重测等于机时翻倍, 而绝大多数轮次根本不会被
        记住 —— 「最后 10% 预算」说的就是这个取舍。

        独立性的**诚实边界**: 这里给的是「另跑一次 L3」—— 新的计时窗口 (新的 runner
        调用、新的一段安静时间), 同一台机器、同一份输入、同一张卡。它抓得住「第一次
        是侥幸」(单次快读数), 抓不住「这台机器的尺子系统性偏了」—— 后者是标定心跳的活。
        `independent=True` 声称的只是前者。
        """
        from .confirm import ConfirmRecord
        if self._run_binding is not None:
            self._check_run_binding(task, input_sets)
            for result in artifact.verification_results:
                receipt = result.metrics.get("run_binding") or {}
                if (receipt.get("candidate_hash") != run_contract.code_hash(artifact.code)
                        or receipt.get("manifest_hash") != self._run_binding["manifest"]["hash"]):
                    raise run_contract.RunBindingError("confirmation candidate/evidence mismatch")
        metrics_1 = artifact.benchmark_data or {}
        # 重测也是一条 L3 读数, 所以同样过尺子检查 —— 否则「决定永久记忆的那一次测量」
        # 恰好是唯一没被检查过的那一次。到期就用本层的判据 (不额外计数: 循环已结束,
        # 计数的作用是循环内的节奏)。
        heartbeat, heartbeat_expected = self._heartbeat_for_next_measurement()
        try:
            results = self.verifier.verify_l3_only(
                kernel_code=artifact.code, test_inputs=input_sets,
                target_gflops=target_gflops, task=task,
                heartbeat=heartbeat, heartbeat_expected=heartbeat_expected,
                # 重测也是一次资格判定: 适用性从**产物里记的那几级结果**里读
                # (`flag_from_metrics` 取最保守的一档; 一条都没有 ⇒ None = 不改判)。
                static_guards=flag_from_metrics(
                    *(r.metrics for r in (artifact.verification_results or []))),
                # 正确性凭据同上 (§14-10 / §20.4): 重测是**晋级层**的那一次测量 ——
                # 它比任何别的读数都更需要说清"这份核验过正确性", 而不是靠它当初
                # 为什么会被记下来这个上下文。
                correctness_verified=_correctness_credential(artifact.verification_results))
            if self._run_binding is not None:
                self._check_run_binding(task, input_sets)
                self._bind_results(results, artifact.code)
            # §10.2 的「最后 10% 不可挪用」: 这一次重测花的是**预留**那一格。记账放在
            # 调用之后 (它已经花了机时), 而且花的是 confirm —— 非 confirm 的开销碰不到它。
            self._budget.oracle_spend("confirm")
            second = results[0] if results else None
        except run_contract.RunBindingError:
            raise
        except Exception as exc:  # noqa: BLE001 — 重测没跑成也是一种结论, 不许静默补 True
            second = None
            print(f"[Confirm] 重测没跑成: {type(exc).__name__}: {exc}")
        m2 = ((getattr(second, "metrics", None) or {}) if second is not None else {})
        obs2 = m2.get("observation") or {}
        if self._run_binding is not None and second is not None:
            artifact.metadata["remeasure_binding"] = m2.get("run_binding")
        # `ok` 的含义是「这次重测本身是一条可用读数」, 不是「它赢了基线」—— 后者由
        # `confirm_persist` 的 speedup 门槛判, 两件事混在一个布尔里就无从诊断。
        # 尺子坏了的窗口里的读数**不能当确认**: 与 `admit_to_corpus` 对「该窗口内所有
        # 数字不可用」的处理同一条道理, 只是这里是第二次确认那一侧。
        usable = bool(second is not None and not obs2.get("refused")
                      and str(m2.get("measurement_validity") or "") != "uninformative"
                      and not m2.get("ruler_broken"))
        if second is None:
            reason = "重测没有结果 (没跑成)"
        elif not usable:
            bits = [f"refused={obs2.get('refused') or '无'}",
                    f"validity={m2.get('measurement_validity') or '未报'}"]
            if m2.get("ruler_broken"):
                bits.append(f"ruler_broken: {m2.get('ruler_reason') or ''}".strip())
            reason = "重测不可用: " + ", ".join(bits)
        else:
            reason = str(getattr(second, "message", "") or "")
        from .workload import confirmation_evidence
        artifact.remeasure = ConfirmRecord(
            ok=usable,
            speedup=decision_speedup(m2),
            performance_basis=(m2.get("performance_verdict") or {}).get("basis", ""),
            workload_evidence=confirmation_evidence(m2),
            validity=str(m2.get("measurement_validity") or ""),
            reason=reason,
            independent=True,
            # **第二次**读数的那条契约, 不是第一次的: 判据在两次之间被改写时,
            # 这个字段就是能证明它的东西 (见 `confirm_persist` 的同一性判据)。
            contract_hash=str(m2.get("contract_hash") or ""),
        )
        if self._run_binding is not None:
            from dataclasses import asdict
            try:
                run_contract.atomic_json(Path(artifact.saved_dir) / "confirmation.json",
                    run_contract.audit_value({"first_binding": metrics_1.get("run_binding"),
                        "second_binding": m2.get("run_binding"),
                        "confirmation": asdict(artifact.remeasure), "second_metrics": m2}))
            except (OSError, ValueError, TypeError) as exc:
                raise run_contract.RunBindingError("confirmation evidence write failed: " + type(exc).__name__) from exc
        first = decision_speedup(metrics_1)
        print(f"[Confirm] 独立重测: "
              f"首次 {first if first is not None else 'n/a'}x → "
              f"重测 {artifact.remeasure.speedup if artifact.remeasure.speedup is not None else 'n/a'}x "
              f"({artifact.remeasure.validity or '未报档位'})")

    @staticmethod
    def _l2_verdict(artifact: KernelArtifact) -> Optional[bool]:
        """L2 的结论（`True` / `False`）；没有 L2 结果 ⇒ `None`（缺证据不是证据）。

        §4.4 的 `correctness_verified` 与 §6.2 条款 2 的「**这条 case** 正确吗」读的是
        **同一份 L2 结果** —— 两处各挑一次就会漂（本仓反复收过的那类账）。
        """
        l2 = [r for r in (getattr(artifact, "verification_results", None) or [])
              if str(getattr(getattr(r, "level", None), "value",
                             getattr(r, "level", ""))).lower() == "l2"]
        from .correctness import credential
        return credential(l2)

    def _resolve_input_sets(self, task, test_inputs):
        """本次要评**哪些 case**、输入从哪来 —— 契约拥有输入 or 调用方给（I94）

        返回 `(input_sets, case_source, declared_cases)`：

        * 调用方给了数组 ⇒ 就是它（`case_source="caller"`）。契约声明了 case 时，产物里的
          `case_match` 会说清「本次评的是不是契约里声明的那几条」;
        * 调用方**没给** ⇒ 从契约的 `cases` 逐个 `materialize()`（`case_source="contract"`）。
          契约拥有输入时，评测的输入就是契约里写的那一组 —— 挡住「针对调用方挑的 shape
          特化」;
        * 契约也没声明 case ⇒ **拒绝开工**：没有 case 可评，不许拿别的输入凑。`None` 在过去
          本来就崩在更深处（一个 `TypeError`），这里把它变成一句说得清的拒绝。
        """
        contract = self._contract_for(task)
        declared_cases = tuple(c.key() for c in contract.cases) if contract is not None else ()
        if isinstance(test_inputs, list):
            sets = list(test_inputs)
        elif test_inputs is not None:
            sets = [test_inputs]
        else:
            sets = []
        if sets:
            return sets, ("caller" if declared_cases else ""), declared_cases
        if not declared_cases:
            raise ValueError(
                "没有输入，契约也没声明 cases ⇒ 没有 case 可评（不拿别的输入凑）: "
                "要么显式传 test_inputs，要么在任务里声明 shapes/cases")
        return [c.materialize() for c in contract.cases], "contract", declared_cases

    @staticmethod
    def _contract_for(task):
        """从任务冻结契约；拿不到 ⇒ `None`（**三态**，不是「空契约」）

        冻结失败**不在这里**抛：调用方给了输入时，契约只是判据面的另一条路（`_save_to_db`
        会再算一次并比对 hash）。只有**依赖契约取输入**时才非它不可 —— 那条路由
        `_resolve_input_sets` 自己拒绝（说得清为什么）。
        """
        if task is None:
            return None
        try:
            from .contract import freeze_from_task
            return freeze_from_task(task)
        except Exception:  # noqa: BLE001 —— 契约不可判定 = 没有契约（不是「空判据」）
            return None

    def _claim_level_for(self, artifact: KernelArtifact, task: TaskSchema,
                         metrics: dict, persisted: bool = False,
                         independent_confirmation: Optional[bool] = None):
        """§4.4 的声明等级：把这次读数手上的证据摊给 `eligibility.claim_level`。

        这里**不算等级**（阶梯与覆盖上限都在那一个纯函数里），只摊平三件证据：
          * `eligibility` —— 产物里的五位掩码（缺一位 ⇒ 按「没有资格位」处理，不许补齐）；
          * `correctness_verified` —— L2 的结论（没有 L2 结果 ⇒ `None`，缺证据不升 validated）；
          * `coverage_mode` —— `source_lang.classify(候选源码, 声明的 dsl)`，与 §21.4 的
            `PrecompileVerdict.coverage.mode` **同源**（守卫钉住这条等式，见
            `tests/test_claim_level.py`）。
        """
        from .eligibility import claim_level
        from .source_lang import classify

        lang = classify(str(getattr(artifact, "code", "") or ""),
                        str(getattr(getattr(task, "target", None), "dsl", "") or ""))
        return claim_level(
            eligibility=metrics.get("eligibility"),
            validity=metrics.get("measurement_validity"),
            correctness_verified=self._l2_verdict(artifact),
            coverage_mode=lang.coverage,
            persisted=persisted,
            independent_confirmation=independent_confirmation,
        )

    def _save_to_db(self, artifact: KernelArtifact, task: TaskSchema):
        """把这次的核写进跨运行经验库 (缺口 2)。门不过就只进内存, 不落盘。

        判据**只有一处** (`oracle.admit_to_corpus`, 设计 §10.5-4): 四道门 (读数够不够格 /
        资格位 can_persist / 声明等级上限 §22.3 / 两次独立确认) 的顺序与理由文本都在那里。
        这里以前自己又写了一遍这三道门 —— 2026-10-01 核实: 那个判据函数全仓没有生产调用者, 而两份实现的口径
        并不一样 (`admit_to_corpus` 永远要求重测, 这里默认不要求), 用例测的是前者。同一个
        决定两个实现, 就是「判据不在链路上」的同族 (spine-review P0-1)。

        刻意放在 `artifact.all_passed` 那一支里、但**再**过门: all_passed 只说明
        「三级验证都点了通过」, 而缺口 1 的真机数据证明那个通过可以是被 launch 地板
        或邻居抢占造出来的 (doc 31/32/33)。一次侥幸读数不能变成永久真理。
        """
        from .kernel_db import KernelRecord
        from .oracle import admit_to_corpus
        from . import corpus_admission as admissions
        from .corpus_journal import CommitResult

        metrics = artifact.benchmark_data or {}
        # §4.4 的声明等级：**推导只有一处**（`eligibility.claim_level`），这里只负责把这次
        # 读数手上的证据摊进去。覆盖那一格与 §22.3 的 `AuditCoverage.mode` 读的是**同一个
        # 函数**（`source_lang.classify`）在同一个来源上的结论，于是「上游发布在血缘里的那个
        # mode」与「入库门拿来判的那个 mode」不会漂（守卫钉住这条等式）。
        claim = self._claim_level_for(artifact, task, metrics)
        # 判据的输入与产物是同一份（与 eligibility / observation 同一条纪律）：
        # `admit_to_corpus` 的覆盖门读的就是这里写进去的那个上限。
        metrics["claim_level"] = claim.as_dict()
        keep, reason = admit_to_corpus(
            metrics,
            getattr(artifact, "remeasure", None),
            min_speedup=float(task.constraints.min_speedup or 1.0),
            # 第 3 道门**恒开** —— 它是设计 §3.5「不允许配置」那一栏里的一项
            # (「长期记忆的独立确认」), 所以这里写死 True 而不是读某个开关:
            # 一个能一键关掉的尺子会先被用在无人值守的夜里 (§3.5 规则 3)。
            # 消融那一臂只能在 harness 侧 patch (见 `tests/test_spine_wiring.py`)。
            requires_remeasure=True,
        )
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
            score=float(decision_speedup(metrics) or 0.0),
            measurement_validity=str(metrics.get("measurement_validity") or ""),
        )
        # 「有没有进语料 + 为什么」以前只 print 一次就没了, 于是同一份产物里
        # 「没入库」与「入库了」在 result.json 上长得一样 (写入时机见下面那段: 落盘与
        # 快照都发生在这个判定**之前**, 所以它只能靠报告这一件产物带走)。
        if artifact.metadata is None:
            artifact.metadata = {}
        commit = None
        request = None
        if keep:
            try:
                # A bound production observation must prepare durable intent
                # before its side effect. Missing evidence cannot fall back to
                # the compatibility legacy-record API.
                if (metrics.get("run_binding") is not None
                        or record.compile_info.get("run_binding") is not None):
                    request = admissions.prepare(artifact.saved_dir, self.kernel_db,
                        record, metrics, artifact.remeasure, reason)
                    commit = self.kernel_db.commit(record, request["key"])
                else:
                    commit = self.kernel_db.commit(record)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                commit = CommitResult("failed", reason=f"{type(exc).__name__}: {exc}")
        persisted = commit is not None and commit.persisted
        artifact.metadata["corpus_admission"] = admissions.admission_metadata(keep, reason, commit)
        # 入库本身就是 `promotable → persisted` 那一级上升（§4.4），所以入没入库之后
        # 要重算一次等级：产物里记的是**最后**的那个地位，不是入库之前那个。
        # 覆盖上限（`ceiling`）不随 `persisted` 变，所以把同一个键改写成终态不会动到
        # 刚刚那道门读过的数 —— 门已经跑完了。
        if keep:
            claim = self._claim_level_for(artifact, task, metrics, persisted=persisted,
                                         independent_confirmation=True)
        final = claim.as_dict()
        artifact.metadata["claim_level"] = final
        metrics["claim_level"] = final
        self._publish_decision(artifact, task, claim)
        if artifact.saved_dir:
            try:
                if request is not None:
                    admissions.finish(artifact.saved_dir, request, self.kernel_db, commit, final, metrics)
                else:
                    admissions.publish(artifact.saved_dir, artifact.metadata["corpus_admission"], final, metrics)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                # The journal remains authoritative. A crash/publication error
                # cannot undo its commit; explicit recovery reconciles files.
                artifact.metadata["corpus_admission"]["publication_error"] = f"{type(exc).__name__}: {exc}"
        if not keep:
            print(f"[KernelDB] 未入库 (只留在本次进程): {reason}")
            self.kernel_db.add(record)
            return False
        if not persisted:
            print(f"[KernelDB] 未确认落盘: {artifact.metadata['corpus_admission']['reason']}")
            return False
        print(f"[KernelDB] 已入库并落盘 ({commit.status}): {reason}")
        return True

    def _publish_decision(self, artifact, task, claim=None):
        from .contract import freeze_from_task
        from .eligibility import (aggregate_speedup, case_speedups, implementation_decision,
                                  regression_over_limit, required_cases_ok)
        metrics = artifact.benchmark_data or {}
        if not metrics:
            return
        claim = claim or self._claim_level_for(artifact, task, metrics)
        if "claim_level" not in metrics:
            metrics["claim_level"] = claim.as_dict()
            artifact.metadata["claim_level"] = claim.as_dict()
        # §6.2 条款 2/3 的两条证据：**逐 run 可判**（I92 之前恒 `None` —— 契约里既没有
        # `required_cases` 也没有 `max_regression`，也没有「本次评了哪一条 case」的账）。
        # 契约从**同一个任务**重新冻结（哈希不一致就不拿新判据去判旧读数），两条证据由
        # `eligibility` 的两个纯函数推导（三态 + 逐 run 理由）。口径与产物同一条纪律：
        # 写进 `metrics` 的就是判据读的那一份。
        contract = freeze_from_task(task)
        declared = str(metrics.get("contract_hash") or "")
        case = str(metrics.get("case_key") or "")
        if declared and (contract.hash or contract.compute_hash()) != declared:
            rco: Optional[bool] = None
            rol: Optional[bool] = None
            notes = (f"本次评测用的契约哈希 {declared[:12]}… 与当前任务重算的 "
                     f"{(contract.hash or contract.compute_hash())[:12]}… 不一致 "
                     f"⇒ 不拿新判据去判旧读数 ⇒ 两条证据都未确立",)
        else:
            # 「本次评过**哪些** case、各自多少」= 逐 case 账本（多组输入时由
            # `Verifier._publish_case_ledger` 发布 `metrics["cases"]`），单组输入退回 I92
            # 那一格 —— 两种形状在 `eligibility.case_speedups` 里收口成同一份账（I95 搬家:
            speedups = case_speedups(metrics)
            # 正确性**按组**、身份**按签名**（见 verifier 那段注释）: 账本里的每一条都过了
            # L2（任一组不过就回不到这里），所以逐 case 的正确性读的是**同一份 L2 结果**
            # （`_l2_verdict`）—— 与 §4.4 的 `correctness_verified` 同一处, 看一遍不看两遍。
            l2_verdict = self._l2_verdict(artifact)
            evaluated = {c: l2_verdict for c in speedups}
            rco, rco_why = required_cases_ok(contract.required_cases, evaluated)
            rol, rol_why = regression_over_limit(contract.max_regression, speedups,
                                                 target_metric=contract.target_metric)
            agg, agg_why = aggregate_speedup(contract.case_weights, speedups,
                                             contract.aggregate,
                                             target_metric=contract.target_metric)
            # §6.2 的**目标函数**在这里发布成数据（不是门槛 —— §16 那条「用单形状的
            # 加速比作为成功标准」还没收口, 见 CHECKLOG I93）。发布它的理由: 「这一次跑
            # 的到底是什么目标」必须可查, 否则权重/口径进了 hash 也没人能核对。
            metrics["objective"] = {
                "aggregate": contract.aggregate or None,
                "weighted_cases": len(contract.case_weights),
                "evaluated_cases": len(speedups),
                "speedup": None if agg is None else round(float(agg), 4),
                "reason": tuple(agg_why),
            }
            notes = tuple(rco_why) + tuple(rol_why)
        # §17.1 的实现裁决：推导只有一处（`eligibility.implementation_decision`），这里只把
        # 这次运行手上**有**的证据摊进去（见 CHECKLOG I91 / I92）。
        verdict = metrics.get("performance_verdict") or {}
        performance_ok = verdict.get("passed") is True if verdict else None
        if contract.case_weights or contract.aggregate:
            from .confirm import confirm_metrics
            confirmed, confirmation_reason = confirm_metrics(metrics, artifact.remeasure,
                min_speedup=float(task.constraints.min_speedup if task.constraints.min_speedup is not None else 1.0))
            performance_ok = ((metrics.get("performance_verdict") or {}).get("passed") is True
                              and confirmed)
            if not confirmed:
                notes += (confirmation_reason,)
        metrics["implementation_decision"] = implementation_decision(
            eligibility=metrics.get("eligibility"), claim_record=claim,
            required_cases_ok=rco, regression_over_limit=rol,
            evidence_notes=notes, performance_ok=performance_ok).as_dict()

    def _record_failure(self, task: TaskSchema, failed_results: list[VerificationResult],
                        fix_note: str, fixed: bool):
        """将验证失败/修复记录写入失败案例库 (负样本)

        「本级判不出」那一类在这里直接返回, 不写 —— 理由与那段注释都在函数体里
        (第 2 片: doc 58 §4.2 的判决侧)。
        """
        if any(r.metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN or r.metrics.get("reference_identity_unknown")
               for r in failed_results):
            print("[Conductor] 传输/参考身份未知，未入负样本库")
            return
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
        # 「本级判不出」不入库。`is_promotable_failure` 那道类别门只管**从文件重放回来**的
        # 记录 (复发门同理, `usable()` 只挡 `_replayed`), 本次进程内新产生的一条一定放行 ——
        # 所以「不入库」必须由写的一侧自己做, 否则同一句话在两个读者那里得到相反的处理。
        if failure_class(failed_results) == FAILURE_CLASS_REFERENCE:
            self._reference_hold_rounds += 1
            print(f"[Conductor] 参考侧判不出, 未入负样本库 "
                  f"(本次运行累计 {self._reference_hold_rounds} 轮, 签名={signature})")
            return
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
