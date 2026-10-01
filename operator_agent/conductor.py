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
from .fixer import FixerAgent, failure_class
from .verifier import Verifier
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
from .measurement import (BAND_SPEC_DRIFT, BAND_UNEXPLAINED, LINEAGE_BAND_KEY,
                          LINEAGE_PERF_RULER_KEY, LINEAGE_RULER_KEY,
                          PERF_GATE_PARAM_KEYS, ROOFLINE_CONVERGED_PCT,
                          RULER_LABEL_CORRECTNESS,
                          RULER_LABEL_PERFORMANCE, TRANSPORT_KEY, Verdict,
                          Validity, band_note, binding_attainment,
                          cross_round_band, cross_round_ruler,
                          correctness_verified_levels, l2_ruler_reading,
                          l3_ruler_reading, roofline_converged,
                          ruler_note, verdict)
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
        # 打开后: 进语料前**由本进程**补一次独立重测 (设计 §10.2 最后 10% 预算 /
        # §10.5-4)。默认关, 因为它是**要花机时**的承诺: 每一份要记住的结论都要量两遍。
        # 关键区别在于它不再是「开着就一律拒」—— 生产者 (`_produce_remeasure`) 与
        # 消费者 (`_save_to_db` → `oracle.admit_to_corpus` 的 `requires_remeasure`)
        # 成对存在, 打开这个开关是可用状态而不是自锁状态。
        require_double_confirm: bool = False,
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
        self.heartbeat_interval = heartbeat_interval
        #: 本次运行里**真值测量**(L3 读数)的累计条数 —— 心跳按它的倍数触发。
        self._true_measurements = 0
        self._heartbeat_epoch = 0
        #: 已经为「第几次测量」跑过心跳了。防的是这一支: 计数停在第 5 次而 L3 一直没
        #: 跑成 (L1/L2 挂了), 于是每一轮都重跑一次心跳。到位只跑一次, 之后的轮次复用
        #: 同一次心跳 (同一扇窗)。
        self._heartbeat_ran_for = 0
        #: 最近一次心跳。只对它自己的 window 作数。
        self._last_heartbeat = None

        #: 永久结论要不要量两遍 (设计 §10.5-4)。生产者是 `_produce_remeasure`, 消费者是
        #: 入库判据 (`oracle.admit_to_corpus` 的 `requires_remeasure`) —— 两边必须同时
        #: 存在, 否则这个开关只有「一律拒」一档。默认 `False` = **不花这份机时**,
        #: 于是默认配置下**单次读数即可入库** (判据的允许理由里会写明这一点)。
        self.require_double_confirm = bool(require_double_confirm)

        # 初始化 Agent (注入 LLM + PromptManager)
        self.designer = DesignerAgent(self.kernel_db, llm=llm, prompt_manager=self.pm)
        # CoderPipeline 默认启用 (P1-1): Plan→并行采样→剪枝→修复→回退
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
                    "search_error", "rounds_exhausted", "infra_retry")

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
        verdict = "OK" if result.ok else "RULER_BROKEN"
        print(f"[Heartbeat] 第 {nxt} 次测量前标定 ({window}): {verdict} "
              f"好核比值={result.known_good_ratio:.3g} "
              f"慢核比值={result.known_slow_ratio:.3g}"
              + (f" {list(result.reasons)}" if result.reasons else ""))
        return result, True

    def _select_finalist(self, current_code: str, finalists: list,
                         task, test_inputs, expected_output) -> tuple:
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
            sel = self.verifier.verify(
                cand, task=task, test_inputs=test_inputs,
                expected_output=expected_output,
                stop_after=VerificationLevel.L2_FUNCTION,
            )
            l2_runs += sum(1 for r in sel
                           if r.level is VerificationLevel.L2_FUNCTION)
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
        # 与水位线同一个理由: 一个 Conductor 实例可以跑多个任务, 落盘与打印的必须是
        # **本次**那一份, 不是「这个进程到现在为止」。
        self._reference_hold_rounds = 0
        # 心跳计数同理: 「每 N 次真值测量」是**这一次运行**内的事, 跨任务累计会让
        # 第二个任务在错误的位置插入参照对。
        self._true_measurements = 0
        self._heartbeat_epoch = 0
        self._heartbeat_ran_for = 0
        self._last_heartbeat = None
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
                        # 已验证祖先不在快照的字段里, 但**可以从血缘的最后一条重建**
                        # (`levels_passed` 每轮都存了, 而快照的 `current_code` 就是那一轮那份)。
                        # 重建不出来时是 None ⇒ 续跑后的第一次修复退回今天的形状, 不猜祖先。
                        verified_parent = verified_parent_from(
                            artifact.iteration_history, current_code)
                        if verified_parent is not None:
                            verified_parent_round = int(
                                (artifact.iteration_history[-1] or {}).get("round") or 0)
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

        # P1-4 (generation-side-review): Search 排好序的邻域候选队列 —— LLM 建议
        # 降为先验后, 下轮 L3 再败时按序轮换, 不重新问 LLM。真机 L3 是裁判:
        # 改进即停 (stagnation 逻辑照旧)。不进快照: 断点续跑时机器状态已变,
        # 旧排序作废, 重新让 Search 出一轮队列。
        search_alternatives: list = []

        for round_num in range(start_round, self.max_optimize_rounds + 1):
            print(f"\n{'─'*40}")
            print(f"轮次 {round_num}/{self.max_optimize_rounds}")
            print(f"{'─'*40}")

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
                    test_inputs=test_inputs,
                )

                # ---- P1-1 候选选择: compile-gate 只定了决赛名单, 真机 L1+L2 定胜负 ----
                # 必须紧跟本轮流水的 coder.run 读名单: `last_finalists` 跨轮残留,
                # 修复轮 (pending_fix) 不该拿上一轮的名单重新选。
                # getattr: 测试常用自定义 coder 替身 (无此属性 = 单候选, 跳过选择)。
                finalists = getattr(self.coder, "last_finalists", [])
                round_candidates = len(finalists)
                if len(finalists) > 1:
                    (current_code, selected_l1_l2, l1_l2_is_final,
                     round_l2_runs) = self._select_finalist(
                        current_code, finalists, task, test_inputs, expected_output,
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
            # P1-1 拼接: 候选选择里已经跑过 L1+L2, 这里只补缺的那一段 ——
            # 胜者补 L3; **全败**时那份 L1+L2 本身就是最终结果 (L2 已失败, 不跑 L3),
            # 同时也是喂 Fixer 的失败证据, 不必重算 (上一版丢掉它、重跑一遍真机 L2;
            # 旧注释写的「胜者的 L1 复检走 SHA256 编译缓存」其实不存在 —— 拼接路径
            # 根本不进 L1)。
            # 标定心跳 (设计 §10.5-1): 到期就在**本窗口内**塞一对已知参照核。读数与心跳
            # 同窗, 于是尺子坏了这条读数当场作废, 而不是等下一轮才发现。
            # 放在「已知不跑 L3」的那一支**之后**: 那一支没有真值测量, 塞心跳纯属烧机时。
            if selected_l1_l2 is not None and l1_l2_is_final:
                print(f"\n[Verifier] 复用候选选择中的 L1+L2 失败结果 (不重跑真机)...")
                verify_results = selected_l1_l2
            else:
                heartbeat, heartbeat_expected = self._heartbeat_for_next_measurement()
                if selected_l1_l2 is not None:
                    print(f"\n[Verifier] 执行 L3 验证 (L1+L2 已在候选选择中通过)...")
                    verify_results = selected_l1_l2 + self.verifier.verify_l3_only(
                        kernel_code=current_code,
                        test_inputs=test_inputs,
                        target_gflops=target_gflops,
                        task=task,
                        environment_drift=verify_drift,
                        heartbeat=heartbeat,
                        heartbeat_expected=heartbeat_expected,
                    )
                else:
                    print(f"\n[Verifier] 执行三级验证...")
                    verify_results = self.verifier.verify(
                        kernel_code=current_code,
                        task=task,
                        test_inputs=test_inputs,
                        expected_output=expected_output,
                        target_gflops=target_gflops,
                        environment_drift=verify_drift,
                        heartbeat=heartbeat,
                        heartbeat_expected=heartbeat_expected,
                    )
            all_verification_results = verify_results

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
                verified_parent = current_code
                verified_parent_round = round_num
            l3_result = next(
                (r for r in verify_results if r.level == VerificationLevel.L3_PERFORMANCE), None,
            )
            # 「真值测量」= 这一轮真的产生了一条 L3 读数。只数它, 不数 L1/L2 —— 心跳
            # 要盯的是「把时间换成数字」的那条通道; 而 L1/L2 失败的轮次不推进计数,
            # 于是同一扇窗的参照对不会被反复重跑 (见 `_heartbeat_ran_for`)。
            if l3_result is not None:
                self._true_measurements += 1

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
                # 资格位 (五位) 也进血缘 —— 上面那段为 `measurement_validity` 写的理由逐字
                # 适用于它, 而且更精确: 档位说「这个数可不可信」, 资格位说「它被允许造成
                # 什么后果」。缺了它, 「有多少轮真的推进了信念 / 有资格写进记忆」这个问句
                # 只能靠重跑估。读者: 设计 §14-9 的资格审计对照实验 (ablation_eligibility)。
                "eligibility": (l3_result.metrics.get("eligibility")
                                if l3_result else None),
                "recipes_applied": list(applied_recipes),
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
                # 分母的原料 (缺口 3 的「先补分母」那半, doc 49 §八 第 14 条): Verifier 每轮**已经算出** roofline 的全部轴
                # (算术强度、脊点、两侧达成率、峰值出处与可信位), 但它们只活到本次判决结束,
                # 从没进过血缘 —— 于是「可达上界是按 shape/AI 索引的曲线」这件已经量出来的
                # 事实, 任何离线复盘都用不上。整块搬, **不挑键**: doc 48 那次给落盘键做
                # 白名单, 代价是那一批真机数据的血缘永久没了。缺的那块不填默认值。
                **_round_axes(l3_result, test_inputs),
            })
            # 这一轮那条记录的引用: 带宽要等**判决发生的那一刻**才知道用没用、用了多少
            # (见下面 `verdict(...)` 那一支), 而这一轮的 dict 已经落在这里 —— 回头按 `-1`
            # 取会依赖「中间没有第二次 append」, 那个前提将来被人破坏时不会响, 只会把
            # 带宽记到别人的轮上。
            _round_rec = artifact.iteration_history[-1]

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
            _ruler_agg = cross_round_ruler(artifact.iteration_history)
            if (_ruler_agg["ruler_same"] is False
                    or _ruler_agg["ruler_rounds_self_conflict"]):
                print(f"\n[Ruler] {ruler_note(_ruler_agg, RULER_LABEL_CORRECTNESS)}")
            _perf_agg = cross_round_ruler(artifact.iteration_history,
                                          keys=PERF_GATE_PARAM_KEYS,
                                          lineage_key=LINEAGE_PERF_RULER_KEY)
            if _perf_agg["ruler_same"] is False:
                print(f"\n[Ruler] {ruler_note(_perf_agg, RULER_LABEL_PERFORMANCE, _PERF_ADVICE)}")

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
                    _perm["best_updated"] = True
                    print(f"\n[Conductor] 建立性能基线: {current_gflops:.2f} GFLOPS "
                          f"(speedup={l3_result.metrics.get('speedup_vs_baseline', 0):.4f})")
                elif current_gflops > 0 and best_gflops > 0:
                    # D4 四态判决。原来这里是无带宽的 `current > best`: 于是「完全
                    # 打平」和「差 0.0001%」都会去禁用配方 —— 而禁用是**永久**决策,
                    # 需要的证据比「这次没测出更好」强。现在只有 SLOW 才禁用。
                    _validity = _validity_of(l3_result.metrics or {})
                    v, why, used_band = verdict(current_gflops, best_gflops, _validity)
                    # 这一轮的判决**真正用过**的那条带宽进血缘 (#87 第 4 片, **只观测**)。
                    # 取的是 `verdict` 自己返回的那个值, 不是这里再查一次表: 分两次查同一张
                    # 表, 中间表被换过就会记下一个「判决没用过的数」, 而这一格要回答的题正是
                    # 「判决用过什么」。AMBIGUOUS 那两支没 consult 表 ⇒ 返回 `None`, 键**缺席**
                    # 才是「这一轮没到判决」, 两者在 `cross_round_band` 里分开数。
                    # 为什么这条要落盘: 带宽换了同时是两件事的读数 (档位降级按表换宽 vs
                    # 表本身被改过), 折成一句「尺变了」就把前者报成后者 —— 归因在
                    # `measurement.cross_round_band` 里分开做。
                    _round_rec[LINEAGE_BAND_KEY] = used_band
                    _band_agg = cross_round_band(artifact.iteration_history)
                    if _band_agg["band_state"] in (BAND_SPEC_DRIFT, BAND_UNEXPLAINED):
                        print(f"\n[Band] {band_note(_band_agg)}")
                    if v is Verdict.FAST:
                        best_gflops = current_gflops
                        _perm["best_updated"] = True
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
                                _perm["recipes_disabled"].append(r.id)
                            if new_recipes:
                                disabled_names = ", ".join(r.id for r in new_recipes)
                                print(f"  [Planner] 性能退步，禁用配方: {disabled_names}")
                            print(f"\n[Conductor] 性能退步 ({current_gflops:.2f} < {best_gflops:.2f} "
                                  f"GFLOPS, {why}), 连续 {rounds_without_improvement + 1}/{max_stagnant_rounds} 轮")
                        else:
                            print(f"\n[Conductor] 性能未改善 ({current_gflops:.2f} vs {best_gflops:.2f} "
                                  f"GFLOPS, {why}), 连续 {rounds_without_improvement + 1}/{max_stagnant_rounds} 轮")
                        rounds_without_improvement += 1
                        _perm["stagnation_counted"] = True
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
                            _perm["recipes_disabled"].append(r.id)
                    rounds_without_improvement += 1
                    _perm["stagnation_counted"] = True
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
                # §8 路由: 先分类再动作。性能不达标≠bug (走搜索), 基础设施≠进度,
                # 编译过、仅数值错 → **修不重写** (§19.3, 外部实测省 92.8% token)。
                from .failure_route import ROUTE_DEBUG_FIX, ROUTE_HOLD, ROUTE_RETRY, classify
                route = classify(verify_results)
                print(f"  [Route] §8 分类: kind={route.kind} route={route.route}"
                      f"{' · fix_not_rewrite' if route.fix_not_rewrite else ''}"
                      f" — {route.reason}")
                if route.route == ROUTE_RETRY:
                    # 基础设施失败: 退避重试, 不计进度、不进负样本库 (§8 表)
                    self._record_stop(artifact, round_num, "infra_retry",
                                      route.reason)
                    break
                # ROUTE_HOLD 仍走 Fixer (既有控制流: 「那一轮仍然走 Fixer」),
                # 只是下面的写入侧按 reference_hold 跳过沉淀 —— 不在这里 continue。
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
                    repair_base = current_code
                    if verified_parent is not None and verified_parent != current_code:
                        repair_base = verified_parent
                        print(f"  [Conductor] 修订起点退回第 {verified_parent_round} 轮那份 "
                              f"L1/L2 已验证的实现 (本轮这一版没有通过正确性验证)")
                    try:
                        fixed_code, fix_note = self.fixer.run(
                            repair_base, task, failed_results,
                            fix_not_rewrite=route.fix_not_rewrite,
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
                        # 判不出那一轮不产 Reflexion: 那一步是让模型回答「这次失败的根因
                        # 是什么、下次怎么避免」, 而这一轮根本没有失败可归因 —— 让它回答就是
                        # 让它编一个候选侧的根因, 再把那份编出来的教训 `add()` 进语料。
                        if not reference_hold:
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
                if search_alternatives:
                    # P1-4: 上一轮排好的邻域候选按序轮换 —— 系统搜索代替「每轮
                    # 重新问一次 LLM 直觉」。轮换候选已带结构 (hints/memory_plan),
                    # 只换了 tiling, 不需要再过 Search。
                    current_sketch = search_alternatives.pop(0)
                    t = current_sketch.tiling
                    suggestion = "P1-4 邻域候选轮换"
                    print(f"\n[Search] 轮换邻域候选: tiling={t.block_m}x{t.block_n}x{t.block_k} "
                          f"(队列剩 {len(search_alternatives)} 个)")
                else:
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
                    # P1-4: 排队下一轮的轮换候选 (SearchAgent 只在 triton 路径产出)
                    search_alternatives = list(getattr(self.search, "last_alternatives", []))
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
            # 永久结论量两遍 (设计 §10.5-4): 只在**真会进语料**的这一支做重测。生产者与
            # 消费者 (下面 `_save_to_db` 里那道门) 成对存在, 所以打开
            # `require_double_confirm` 是可用状态, 不是「入库率归零」的自锁状态。
            if self.require_double_confirm:
                self._produce_remeasure(artifact, task, test_inputs, target_gflops)
            self._save_to_db(artifact, task)

        # ── 报告产物 (设计 §17 第 3 条) ──
        # 位置是刻意的: **必须在 `store.save()` 与 `_save_to_db()` 之后**。前者给出产物
        # 目录, 后者落进「这条结论有没有进长期记忆 + 为什么」—— 那是报告里最要紧的一句
        # 之一, 而它此刻才存在。报告写失败不让这次实验失败 (write_report 内部把异常写进
        # 那个文件), 所以这里不需要再兜一层。
        from .report import write_report
        _admission = (artifact.metadata or {}).get("corpus_admission") or {}
        _refusals = ([_admission.get("reason")]
                     if _admission.get("kept") is False and _admission.get("reason")
                     else None)
        write_report(saved_dir, artifact=artifact, task=task, refusals=_refusals)
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

    def _produce_remeasure(self, artifact, task, test_inputs, target_gflops: float) -> None:
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
        metrics_1 = artifact.benchmark_data or {}
        # 重测也是一条 L3 读数, 所以同样过尺子检查 —— 否则「决定永久记忆的那一次测量」
        # 恰好是唯一没被检查过的那一次。到期就用本层的判据 (不额外计数: 循环已结束,
        # 计数的作用是循环内的节奏)。
        heartbeat, heartbeat_expected = self._heartbeat_for_next_measurement()
        try:
            results = self.verifier.verify_l3_only(
                kernel_code=artifact.code, test_inputs=test_inputs,
                target_gflops=target_gflops, task=task,
                heartbeat=heartbeat, heartbeat_expected=heartbeat_expected)
            second = results[0] if results else None
        except Exception as exc:  # noqa: BLE001 — 重测没跑成也是一种结论, 不许静默补 True
            second = None
            print(f"[Confirm] 重测没跑成: {type(exc).__name__}: {exc}")
        m2 = ((getattr(second, "metrics", None) or {}) if second is not None else {})
        obs2 = m2.get("observation") or {}
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
        artifact.remeasure = ConfirmRecord(
            ok=usable,
            speedup=m2.get("speedup_vs_baseline"),
            validity=str(m2.get("measurement_validity") or ""),
            reason=reason,
            independent=True,
            # **第二次**读数的那条契约, 不是第一次的: 判据在两次之间被改写时,
            # 这个字段就是能证明它的东西 (见 `confirm_persist` 的同一性判据)。
            contract_hash=str(m2.get("contract_hash") or ""),
        )
        first = metrics_1.get("speedup_vs_baseline")
        print(f"[Confirm] 独立重测: "
              f"首次 {first if first is not None else 'n/a'}x → "
              f"重测 {artifact.remeasure.speedup if artifact.remeasure.speedup is not None else 'n/a'}x "
              f"({artifact.remeasure.validity or '未报档位'})")

    def _save_to_db(self, artifact: KernelArtifact, task: TaskSchema):
        """把这次的核写进跨运行经验库 (缺口 2)。门不过就只进内存, 不落盘。

        判据**只有一处** (`oracle.admit_to_corpus`, 设计 §10.5-4): 三道门 (读数够不够格 /
        资格位 can_persist / 两次独立确认) 的顺序与理由文本都在那里。这里以前自己又写了
        一遍这三道门 —— 2026-10-01 核实: 那个判据函数全仓没有生产调用者, 而两份实现的口径
        并不一样 (`admit_to_corpus` 永远要求重测, 这里默认不要求), 用例测的是前者。同一个
        决定两个实现, 就是「判据不在链路上」的同族 (spine-review P0-1)。

        刻意放在 `artifact.all_passed` 那一支里、但**再**过门: all_passed 只说明
        「三级验证都点了通过」, 而缺口 1 的真机数据证明那个通过可以是被 launch 地板
        或邻居抢占造出来的 (doc 31/32/33)。一次侥幸读数不能变成永久真理。
        """
        from .kernel_db import KernelRecord
        from .oracle import admit_to_corpus

        metrics = artifact.benchmark_data or {}
        keep, reason = admit_to_corpus(
            metrics,
            getattr(artifact, "remeasure", None),
            min_speedup=float(task.constraints.min_speedup or 1.0),
            # 第 3 道门是不是强制, 是**机时承诺**而不是判据细节: 见 `admit_to_corpus`
            # 的 docstring 与 `require_double_confirm` 的声明处。
            requires_remeasure=bool(getattr(self, "require_double_confirm", False)),
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
            score=float(metrics.get("speedup_vs_baseline") or 0.0),
            measurement_validity=str(metrics.get("measurement_validity") or ""),
        )
        # 「有没有进语料 + 为什么」以前只 print 一次就没了, 于是同一份产物里
        # 「没入库」与「入库了」在 result.json 上长得一样 (写入时机见下面那段: 落盘与
        # 快照都发生在这个判定**之前**, 所以它只能靠报告这一件产物带走)。
        if artifact.metadata is None:
            artifact.metadata = {}
        artifact.metadata["corpus_admission"] = {"kept": bool(keep), "reason": str(reason)}
        if not keep:
            print(f"[KernelDB] 未入库 (只留在本次进程): {reason}")
            self.kernel_db.add(record)
            return False
        self.kernel_db.promote(record)
        print(f"[KernelDB] 已入库并落盘: {reason}")
        return True

    def _record_failure(self, task: TaskSchema, failed_results: list[VerificationResult],
                        fix_note: str, fixed: bool):
        """将验证失败/修复记录写入失败案例库 (负样本)

        「本级判不出」那一类在这里直接返回, 不写 —— 理由与那段注释都在函数体里
        (第 2 片: doc 58 §4.2 的判决侧)。
        """
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
