"""端到端成功率量具: 这条流水线现在到底能不能产出「又快又对」的核

为什么单独有这个驱动: 67 个提交里改了一大堆判定与口径, 但**「生成的核有多少比例能过 L2、
中位加速比多少、平均烧几轮」这三个数, 从头到尾没有被量过**。`tests/golden_regression.py`
跑在 MockGPU 上、用的是「伪 GFLOPS」, 结构上答不了这个问题; `benchmarks/` 里其余 30 个驱动
全是单点探针 (一次量一个机制), 没有一个产出聚合通过率。

于是四件事同时悬着, 且都卡在这一个量具上:
  · C1 处方案式错误反馈的真实效应 (doc 35 的 A/B 打平, 而那个设计看不见效应);
  · doc 47 没量出来的候选重复基率;
  · `peak_memory_ratio` 该定什么阈值 (doc 44 只有 4 个点);
  · C3 最后一跳 (复发计数喂进 Planner) 接上之后到底有没有用。

跑法 (需要 GPU 机的 ssh 别名 + config.yaml 里的 LLM 端点):
    python3 benchmarks/harness_pass_rate.py            # 全量
    python3 benchmarks/harness_pass_rate.py --repeats 1 # 先冒烟

⚠️ 三条读数限制, 都在报表里显式打印, 不要脱离它们引用数字:
  1. **语料只有两个算子** (matmul fp16 / layernorm fp32)。加算子的前提是输入契约已经
     验证过 —— doc 41 记录过一次真实事故: attention 因为多传一个 `causal` 连烧三轮。
     凭空造输入会让「通过率」量到契约错误而不是模型能力。
  2. **分母小**。每次运行通常第 1 轮就通过 (生产门槛是 1.0x), 所以轮数≈运行数。
     分母不足 MIN_DENOM 时本驱动**拒绝打印百分数** —— 「0/1」和「0/12」都写成 0.0%
     是这份量具第一版真实犯过的错。
  3. **通过率与档位必须一起读**。一个在 `uninformative` 档上「通过」的核, 它的加速比
     没有资格驱动任何闭环决策 (缺口 1)。所以报表把 decision-grade 占比与通过率并列,
     单看通过率会把「盲跑但碰巧过了」读成成功。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from operator_agent.backend_caps import Cap  # noqa: E402
from operator_agent.conductor import Conductor  # noqa: E402
from operator_agent.llm_client import LLMClient  # noqa: E402
from operator_agent.measurement import Validity, persist_decision  # noqa: E402
from operator_agent.remote_executor import RemoteExecutor  # noqa: E402
from operator_agent.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema, TaskType,
)

# 分母下限: 低于它不出百分数。理由见文件头第 2 条。
MIN_DENOM = 4

# 语料。shape 选的是 doc 31/44 真机标定过的两档: matmul 4096³ 落 `valid`
# (launch 地板占比 0.020), layernorm 16384×6144 落 `degraded` (0.111)。
# 故意一档一个: 报表要能看出「通过率」与「读数可不可信」是两件事。
CASES = [
    ("matmul", Precision.FP16, (4096, 4096, 4096)),
    ("layernorm", Precision.FP32, (16384, 6144)),
]
# 生产门槛, 不是 doc 47 那个「故意达不到」的 10x。这里量的是「达不达标」,
# 所以必须用真实门槛, 否则量到的是一个永远失败的循环。
MIN_SPEEDUP = 1.0
MAX_ERROR = 1e-2


def llm_from_config() -> LLMClient:
    """从 config.yaml 读自建端点。值只进环境变量, **不打印**。"""
    import yaml
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        llm = (cfg.get("llm") or {})
        os.environ.setdefault("LLM_BASE_URL", str(llm.get("base_url") or ""))
        os.environ.setdefault("LLM_MODEL", str(llm.get("model") or ""))
        os.environ.setdefault("LLM_API_KEY", str(llm.get("api_key") or ""))
    client = LLMClient(temperature=0.3, max_tokens=4096)
    # 判别实验的输入必须是**同一个模型**: 混进付费降级模型的话, 量的就是两个模型的差,
    # 而不是机制的差 (doc 47 的驱动同一条纪律)
    if getattr(client, "fallback_model", None):
        raise SystemExit("❌ 本次测量不允许走降级模型 (会污染通过率的口径)")
    return client


def _task(kernel_type: str, precision: Precision) -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, complexity=TaskComplexity.MEDIUM),
        # 这份量具的表头写着「真机」, 所以任务就把「要真机」声明出来。声明的读者是
        # `Verifier._verify_performance`: 执行侧给不出 `REAL_DEVICE` 时 L3 不再记为通过
        # (doc 49 §3.5 第 2/3 处)。`main()` 里那道 `backend.require(...)` 是**开工前**的
        # 硬门, 挡的是「门面静默降级成 LocalExecutor 而报表照旧写着真机」; 这一条是
        # **每一轮读数**上的门, 挡的是「某一路载荷其实是在 CPU 上测的」。两道不重复。
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton",
                          required_caps=(Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE)),
        constraints=ConstraintSpec(precision=precision, min_speedup=MIN_SPEEDUP,
                                   max_error=MAX_ERROR),
    )


def inputs_for(kernel_type: str, shape):
    """与 `measure_candidate_duplicates.py` 同一套输入 —— 已验证能端到端跑通的契约"""
    if kernel_type == "matmul":
        m, n, k = shape
        return {"A": np.random.randn(m, k).astype(np.float32),
                "B": np.random.randn(k, n).astype(np.float32)}
    m, n = shape
    return {"X": np.random.randn(m, n).astype(np.float32),
            "weight": np.ones(n, dtype=np.float32),
            "bias": np.zeros(n, dtype=np.float32)}


# ─── 聚合: 纯函数, 可单测 ─────────────────────────────────────
# 放在这里而不是内联在 main 里, 是为了让「报表算法」本身可以被假数据验一遍。
# 这个项目的教训 (doc 46): 判据方向写反时, 量具明明看见了信号却报「看不见」——
# 会自我批评的错误判据比缺失的更难发现。所以聚合逻辑必须能离线跑。

def _level_rate(rounds, level):
    """某一级的通过率。**分母是「走到了这一级的轮数」**, 不是总轮数。

    L1 就挂掉的轮次根本没有 L2/L3 的条目; 拿总轮数当分母会把「没走到」算成「没通过」,
    于是 L2/L3 的通过率被 L1 的失败率污染 —— 这正是 doc 47 第一版把 `generate` 轮
    算进 optimize 分母的同一个错。
    """
    reached = [r for r in rounds if level in (r.get("levels_passed") or {})]
    passed = [r for r in reached if (r["levels_passed"] or {}).get(level)]
    return len(passed), len(reached)


def _pct(num, den):
    return round(num / den * 100, 1) if den else None


def _duplicate_rate(ok_runs):
    """候选重复率 —— doc 47 没能量到的那个基率, 顺手在每份报表里带上。

    **分母是「本轮之前在同一任务里已经有过候选」的轮数**: 第一轮没有可比对象,
    算进分母就是把「不可能重复」记成「没重复」, 与 `_level_rate` 那条「没走到 ≠ 没通过」
    是同一个错的镜像 (doc 47 第一版把 `generate` 轮算进 optimize 分母也是它)。

    返回 `(重复轮数, 可比分母, 血缘里有没有签名)`。第三项是必须的: 签名缺失时要报
    「测不到」而**不是 0%** —— 这正是 doc 48 那一批 12 轮的处境, 键在血缘里算好了,
    却被报表自己的键白名单丢在落盘之前。
    """
    dup = at_risk = signed = 0
    for r in ok_runs:
        for i, h in enumerate(r.get("rounds") or []):
            if h.get("candidate_signature") is not None:
                signed += 1
            if i == 0:
                continue
            at_risk += 1
            if h.get("duplicate_of") is not None:
                dup += 1
    return dup, at_risk, signed > 0


def cost_from_metadata(metadata):
    """从 `artifact.metadata["llm_cost"]` 取出 (tokens, usd, calls)

    键名必须与 `LLMUsageTracker.summary()` 一致: 是 **`total_cost_usd`** 而不是
    `cost_usd`。第一版读错了键, 于是整场测量的成本一律报成 $0.0 —— 540114 tokens
    配一个 0 美元, 安静得像是对的。`tests/test_harness_pass_rate.py` 里有一条机械守卫
    直接对着真的 `summary()` 查这几个键在不在, 键名再漂移就会红。
    """
    c = (metadata or {}).get("llm_cost") or {}
    return (int(c.get("total_tokens") or 0),
            float(c.get("total_cost_usd") or 0.0),
            int(c.get("total_calls") or 0))


def first_pass_round(rounds):
    """第一次 `all_passed` 的那一轮 → (轮号, 该轮的 action); 从没通过则 (None, None)

    **不管那轮的 action 是什么**。第一版把 `fix` 轮排除在外 (理由是「修好的不算一次
    干净的通过」), 真机第 4 次运行立刻证伪: layernorm#2 第 1 轮失败、Fixer 修完第 2 轮
    通过, 于是 `最终通过=True` 而 `rounds_to_pass=None` —— 这个样本被 rounds-to-pass 的
    分位数**静默丢掉**。指标悄悄少算样本比指标算错更难发现。
    「本来就对」还是「修好的」这个区分保留在返回的 action 里, 不去动分母。
    """
    for r in rounds or ():
        if r.get("all_passed"):
            return r.get("round"), r.get("action")
    return None, None


def _quantiles(values):
    if not values:
        return None
    s = sorted(values)

    def q(p):
        i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return round(s[i], 4)

    return {"n": len(s), "min": q(0.0), "p25": q(0.25), "median": q(0.5),
            "p75": q(0.75), "max": q(1.0)}


def aggregate(runs):
    """把逐次运行的原始记录聚合成报表。返回的每个百分数都**带分母**。"""
    ok_runs = [r for r in runs if r.get("ok")]
    crashed = [r for r in runs if not r.get("ok")]
    rounds = [x for r in ok_runs for x in (r.get("rounds") or [])]

    l1 = _level_rate(rounds, "L1")
    l2 = _level_rate(rounds, "L2")
    l3 = _level_rate(rounds, "L3")

    passed_runs = [r for r in ok_runs if r.get("final_all_passed")]
    rtp = [r["rounds_to_pass"] for r in passed_runs if r.get("rounds_to_pass")]

    tiers = {}
    for r in rounds:
        v = r.get("measurement_validity")
        if v is not None:
            tiers[v] = tiers.get(v, 0) + 1
    graded = [r for r in rounds if r.get("measurement_validity") is not None]
    decision_grade = [r for r in graded
                      if r.get("measurement_validity") == Validity.VALID.value]

    # 加速比分两套口径给, 不合并: 混在一起就是把「不可信的读数」当成成绩
    sp_all = [r["speedup"] for r in rounds if isinstance(r.get("speedup"), (int, float))]
    sp_grade = [r["speedup"] for r in decision_grade
                if isinstance(r.get("speedup"), (int, float))]

    # 入库率与拒收名单必须同分母。`_save_to_db()` 只在 `artifact.all_passed` 那一支被调
    # (`conductor.py:1112` 明写), 所以**未通过的运行在生产里根本不会被问这道门** —— 把它们
    # 的判定混进「拒收」名单, 就是让读者以为入库门拒了一些好核, 而它其实一条都没拒
    # (2026-09-29 那份报表印了「入库 3/3」紧跟三条拒收, 三条全来自未通过的运行)。
    admitted = [r for r in passed_runs if r.get("admitted")]
    # 假想事件单独放: 它有用的场合是「如果这些运行当时通过了, 门会不会放它进去」,
    # 那是一个关于门的问题, 不是一个关于本轮成果的问题。两者必须分栏。
    counterfactual = sorted({r["admit_reason"] for r in ok_runs
                             if r.get("admit_reason") and not r.get("admitted")
                             and not r.get("final_all_passed")})
    # 名单是**去重后的理由**, 不是运行数 —— 三次同样的拒收只印一行。所以两个数都要给,
    # 否则「另 3 条」会被读成「还有 3 次被拒」, 而它可能只是 1 种理由的 3 次重复。
    counterfactual_runs = sum(1 for r in ok_runs
                              if r.get("admit_reason") and not r.get("admitted")
                              and not r.get("final_all_passed"))

    dup, at_risk, has_sig = _duplicate_rate(ok_runs)
    via = {}
    for r in passed_runs:
        k = r.get("passed_via") or "(未记)"
        via[k] = via.get(k, 0) + 1

    return {
        "runs": len(runs),
        "runs_ok": len(ok_runs),
        "runs_crashed": len(crashed),
        "crash_reasons": [r.get("error") for r in crashed],
        "rounds": len(rounds),
        "level_rates": {"L1": l1, "L2": l2, "L3": l3},
        "level_pcts": {"L1": _pct(*l1), "L2": _pct(*l2), "L3": _pct(*l3)},
        "task_success": (len(passed_runs), len(ok_runs)),
        "task_success_pct": _pct(len(passed_runs), len(ok_runs)),
        "rounds_to_pass": _quantiles(rtp),
        "passed_via": via,
        "validity_tiers": tiers,
        "decision_grade": (len(decision_grade), len(graded)),
        "decision_grade_pct": _pct(len(decision_grade), len(graded)),
        "speedup_all": _quantiles(sp_all),
        "speedup_decision_grade": _quantiles(sp_grade),
        "duplicate_rounds": (dup, at_risk),
        "duplicate_measurable": has_sig,
        "duplicate_pct": _pct(dup, at_risk) if has_sig else None,
        "admitted": (len(admitted), len(passed_runs)),
        "admitted_pct": _pct(len(admitted), len(passed_runs)),
        "admit_rejections": sorted({r["admit_reason"] for r in passed_runs
                                    if r.get("admit_reason") and not r.get("admitted")}),
        "gate_never_asked": sorted({r["admit_reason"] for r in passed_runs
                                    if not r.get("admit_reason")}),
        "counterfactual_rejections": counterfactual,
        "counterfactual_runs": counterfactual_runs,
        "admit_rejection_runs": sum(1 for r in passed_runs
                                    if r.get("admit_reason") and not r.get("admitted")),
        "cost": {
            "tokens": sum(int(r.get("tokens") or 0) for r in ok_runs),
            "calls": sum(int(r.get("llm_calls") or 0) for r in ok_runs),
            "usd": round(sum(float(r.get("cost_usd") or 0.0) for r in ok_runs), 4),
            "wall_s": round(sum(float(r.get("wall_s") or 0.0) for r in ok_runs), 1),
        },
        "denominator_ok": min(len(ok_runs), len(rounds)) >= MIN_DENOM,
    }


def device_record(backend, env=None) -> dict:
    """报表里的「跑在哪张卡」必须来自**执行侧**, 不是本地环境变量 (#52)

    旧写法是 `"gpu": os.environ.get("OPERATOR_AGENT_GPU", "(未固定)")` —— 读的是用户
    **想**的那张卡。而 `SSHExecutor` 从不读那个变量: 它在本进程里自动挑一张最闲的卡
    (`_pinned_gpu()`), 再把**自己选中的那个号**作为前缀下发给远端。于是第一份基线里
    「GPU 2 起跑」可能是报表值而不是事实值 (实际是 GPU 7) —— 记录的字段与发生的事不同源,
    是本项目那一族缺陷的「读了, 但读错了对象」变体。

    三态要**分开表达**, 不能都塌成 `(未固定)`:
      * 后端不具备 `GPU_PINNING` (CPU / 模拟器 / HTTP) → 没有「固定一张卡」这件事;
      * 具备但没解析出卡号 → 本来该固定, 这一轮其实是远端自动选的;
      * 具备且有号 → 就是它。
    `conflicts` 只在本地设了那个变量时才可能为真: 设了就说明用户想钉一张卡, 而这条路
    今天做不到, 所以必须响一行, 不能静默换卡。
    """
    env = os.environ if env is None else env
    requested = (env.get("OPERATOR_AGENT_GPU") or "").strip()
    actual = backend.selected_gpu
    can_pin = Cap.GPU_PINNING in backend.capabilities
    if not can_pin:
        state = f"({backend.label} 不具备 {Cap.GPU_PINNING.value}, 没有固定卡号这件事)"
    elif actual is None:
        state = "(声明可固定, 但这一轮没解析出卡号 ⇒ 由远端自动选择)"
    else:
        state = f"GPU {actual}"
    conflicts = bool(requested) and (str(actual) if actual is not None else "") != requested
    return {"state": state, "gpu": actual, "can_pin": can_pin,
            "requested_env": requested or None, "conflicts": conflicts,
            "source": "backend.selected_gpu"}


def format_report(agg):
    lines = ["", "=" * 62, "端到端成功率报表", "=" * 62]
    lines.append(f"运行 {agg['runs_ok']}/{agg['runs']} 次完成"
                 + (f" (崩 {agg['runs_crashed']})" if agg["runs_crashed"] else "")
                 + f" · 共 {agg['rounds']} 轮")
    if not agg["denominator_ok"]:
        lines.append(f"\n❌ 分母不足 (运行 {agg['runs_ok']} / 轮 {agg['rounds']}, "
                     f"下限 {MIN_DENOM}) —— 不出百分数。")
        lines.append("   0/1 与 0/12 都会打印成 0.0%, 那不是测量值。先让循环真跑起来。")
        return "\n".join(lines)

    lines.append("\n【分级通过率】分母 = 走到该级的轮数 (不是总轮数)")
    for lv in ("L1", "L2", "L3"):
        p, d = agg["level_rates"][lv]
        lines.append(f"  {lv}: {p}/{d}" + (f" = {agg['level_pcts'][lv]}%" if d else " (无)"))

    ts, td = agg["task_success"]
    lines.append(f"\n【任务级成功】{ts}/{td} = {agg['task_success_pct']}%  (最终 all_passed)")
    if agg["rounds_to_pass"]:
        q = agg["rounds_to_pass"]
        lines.append(f"【rounds-to-pass】median {q['median']} · p25 {q['p25']} · "
                     f"p75 {q['p75']} · max {q['max']} (n={q['n']})")
    if agg.get("passed_via"):
        # 「靠哪一轮过的」不是花边: doc 48 的读数 ④ (33% 的运行首轮死在 L1、被 Fixer
        # 一轮救回) 只能从这里读出来, 手算的话下一次就没人算了。
        lines.append(f"  通过方式: {agg['passed_via']}")

    d, dt = agg["duplicate_rounds"]
    lines.append("\n【候选重复率】分母 = 本轮之前同一任务里已有候选的轮数")
    if not agg["duplicate_measurable"]:
        lines.append(f"  测不到: 血缘里没有 candidate_signature (可比轮 {dt}) —— 这不是 0%")
    else:
        lines.append(f"  {d}/{dt}" + (f" = {agg['duplicate_pct']}%" if dt else " (无可比轮)"))

    dg, dgt = agg["decision_grade"]
    lines.append(f"\n【读数可信度】decision-grade (valid 档) {dg}/{dgt}"
                 + (f" = {agg['decision_grade_pct']}%" if dgt else ""))
    lines.append(f"  档位分布: {agg['validity_tiers'] or '无 L3 读数'}")
    lines.append("  ⚠️ 通过率必须与这一行一起读: 在 uninformative 档上「通过」的核, "
                 "它的加速比没有资格驱动任何闭环决策。")

    for label, key in (("全部轮次", "speedup_all"), ("仅 decision-grade", "speedup_decision_grade")):
        q = agg[key]
        if q:
            lines.append(f"\n【加速比 · {label}】median {q['median']}x · p25 {q['p25']} · "
                         f"p75 {q['p75']} · min {q['min']} · max {q['max']} (n={q['n']})")
        else:
            lines.append(f"\n【加速比 · {label}】无数据")

    a, at = agg["admitted"]
    lines.append(f"\n【入库】通过的运行里 {a}/{at}"
                 + (f" = {agg['admitted_pct']}%" if at else "") + " 被写进跨运行经验库")
    for reason in agg["admit_rejections"]:
        lines.append(f"  拒收: {reason}")
    if agg["admit_rejections"]:
        lines.append(f"  （{agg['admit_rejection_runs']} 次运行, "
                     f"{len(agg['admit_rejections'])} 种不同的理由——名单按理由去重）")
    if at and not agg["admit_rejections"]:
        # 空名单也要说一句: 不印的话「门拒了 0 条」与「门没被问过」在报表里长得一样
        lines.append("  入库门本轮没有拒收任何一次通过的运行")
    if agg.get("gate_never_asked"):
        lines.append(f"  ⚠️ 有 {len(agg['gate_never_asked'])} 次通过的运行**没有入库判定记录**"
                     f" —— 门没被问, 不等于门放过了")
    if agg.get("counterfactual_rejections"):
        lines.append(f"  (另 {agg['counterfactual_runs']} 次**未通过**的运行给出了 "
                     f"{len(agg['counterfactual_rejections'])} 种假想判定 —— 生产里 "
                     f"`_save_to_db` 只在 all_passed 那一支问这道门, 所以它们不算拒收)")

    c = agg["cost"]
    lines.append(f"\n【成本】{c['tokens']} tokens / {c.get('calls', 0)} 次调用 ≈ ${c['usd']}"
                 f" · 墙钟 {c['wall_s']} s")
    return "\n".join(lines)


# ─── 真机执行 ─────────────────────────────────────────────────

def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "?"
    except Exception:  # noqa: BLE001
        return "?"


def record_rounds(history):
    """把 `iteration_history` 原样搬进报表记录, **不做键白名单**

    第一版是白名单 (`round`/`action`/`all_passed`/`levels_passed`/
    `measurement_validity`/`gflops`/`speedup`), 于是 Conductor 写的
    `candidate_signature`/`duplicate_of`/`recipes_applied` 三键在这一步被悄悄丢掉 ——
    而 doc 48 第七节第 2 条正承诺「跑完就能离线算重复率」。真机跑完才发现落盘 JSON 里
    根本没有签名, 那一批 12 轮的重复基率**永久丢了** (只剩终端里那行 `[Novelty]` 汇总)。
    与 doc 33 的 `runs`/`max_time_ms`、doc 37 的 `llm_cost` 同一族: 值算出来了,
    死在搬运路上。**白名单的问题不是漏了哪几个键, 而是「血缘以后每加一个键都要记得来
    这里加一次」—— 而忘记的代价是静默的。**
    """
    out = []
    for h in history or ():
        rec = dict(h)
        rec["all_passed"] = bool(h.get("all_passed"))
        rec["levels_passed"] = h.get("levels_passed") or {}
        out.append(rec)
    return out


def run_one(label, kernel_type, precision, shape, backend, rounds):
    """跑一次完整任务, 返回一份原始记录 (不聚合, 便于落盘复盘)

    **每次运行新建一个 LLMClient**: `CostTracker` 是挂在 client 上累加的, 共用一个
    client 会让 `metadata["llm_cost"]` 变成**从头到现在的累计值**而不是本次的值。
    第一版就是这么写的, 于是六次运行的 token 数成了 15638 → 34457 → 78801 → 101156
    → 143717 → 166345 这样一条单调递增曲线, 报表把「累计值」再求和得到 540114,
    而真实总量是 166345 —— **高估 3.25 倍**, 且每一条单独看都像合理的数。
    顺带也让每次运行成为独立样本 (不共享任何跨运行状态)。
    """
    llm = llm_from_config()
    rec = {"case": label, "kernel_type": kernel_type, "shape": list(shape),
           "ok": False, "error": None, "rounds": [], "final_all_passed": False,
           "rounds_to_pass": None, "passed_via": None, "admitted": False,
           "admit_reason": "", "tokens": 0, "cost_usd": 0.0, "llm_calls": 0,
           "wall_s": 0.0}
    t0 = time.time()
    try:
        conductor = Conductor(backend=backend, llm=llm, max_optimize_rounds=rounds)
        art = conductor.execute(_task(kernel_type, precision),
                                inputs_for(kernel_type, shape), target_gflops=0.0)
    except Exception as exc:  # noqa: BLE001
        # 一次 ssh 抖动不该毁掉整场测量: 记成崩溃样本, 继续下一个
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        rec["wall_s"] = round(time.time() - t0, 1)
        return rec

    rec["ok"] = True
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["rounds"] = record_rounds(art.iteration_history)
    # `KernelArtifact.all_passed` 是 `all(v.passed for v in verification_results)`,
    # 而 **all([]) 是 True** —— 一份没有任何验证结果的产物会被它判成「全过」。
    # 两条早退路径 (入口校验失败 / 环境自检未通过) 都填了合成的 L1 失败, 所以现在踩不到;
    # 但「跑出了候选」本来就该是任务成功的定义之一, 所以这里显式要求至少有一轮。
    rec["final_all_passed"] = bool(rec["rounds"]) and bool(getattr(art, "all_passed", False))
    # rounds_to_pass 的口径见 first_pass_round 的 docstring (含被真机证伪的那一版)
    rec["rounds_to_pass"], rec["passed_via"] = first_pass_round(rec["rounds"])
    # 入库决定在 Conductor 里只打印、不落产物, 所以这里用同一个纯函数复算一遍。
    # 复算而不是解析 stdout: 后者会把报表绑在日志文案上。
    keep, reason = persist_decision(art.benchmark_data or {})
    rec["admitted"], rec["admit_reason"] = bool(keep), reason
    rec["tokens"], rec["cost_usd"], rec["llm_calls"] = cost_from_metadata(art.metadata)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3, help="每个用例重复几次 (默认 3)")
    ap.add_argument("--rounds", type=int, default=3, help="max_optimize_rounds (默认 3)")
    ap.add_argument("--out", default="", help="报表落盘目录 (默认 output/harness/<时间戳>)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "output" / "harness"
        / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 不在这里建 LLMClient: run_one 每次自建一个, 否则 CostTracker 会跨运行累加
    # (见 run_one 的 docstring —— 第一版因此把成本高估了 3.25 倍)。
    # 这里只先建一次做**口径自检**: 判别实验不允许混进付费降级模型。
    probe = llm_from_config()
    print(f"[harness] 模型口径自检通过: {getattr(probe, 'model', '?')} (无降级模型)")
    backend = RemoteExecutor(mode="ssh", timeout=900)
    # `mode="ssh"` 现在是**承诺**: 连不上就在上一行抛 BackendUnavailable, 不会静默降级到
    # LocalExecutor 让这份报表装满 CPU 数字而表头照旧写着真机 (doc 49 §7.8)。
    # 这道 require 因此是**第二**防线, 但它还管着 SAME_DEVICE_BASELINE 那半边 ——
    # 缺能力就在开工前响, 而不是产出一份看起来正常的报表。
    backend.require(Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE)
    print(f"[harness] 执行侧能力自检通过: {backend.label}")

    runs = []
    for rep in range(args.repeats):
        for kernel_type, precision, shape in CASES:
            label = f"{kernel_type}/{precision.value}/{'x'.join(map(str, shape))}#{rep + 1}"
            print(f"\n{'=' * 62}\n[{label}] 真跑 (门槛 {MIN_SPEEDUP}x, 最多 {args.rounds} 轮)\n{'=' * 62}")
            rec = run_one(label, kernel_type, precision, shape, backend, args.rounds)
            runs.append(rec)
            print(f"[{label}] ok={rec['ok']} 轮数={len(rec['rounds'])} "
                  f"最终通过={rec['final_all_passed']} rounds_to_pass={rec['rounds_to_pass']} "
                  f"入库={rec['admitted']}")
            if rec["error"]:
                print(f"  ❌ {rec['error']}")

    agg = aggregate(runs)
    report = format_report(agg)
    print(report)

    device = device_record(backend)
    print(f"\n【实际用卡】{device['state']}  (来源: {device['source']})")
    if device["conflicts"]:
        print(f"  ⚠️ 本地设了 OPERATOR_AGENT_GPU={device['requested_env']}, 但 "
              f"`SSHExecutor` **从不读它** —— 它在本进程自动挑最闲的卡, 再把选中的号"
              f"下发给远端。这份报表记录的卡号以 {device['state']} 为准; "
              f"要真的钉住某张卡得改 `_pinned_gpu()` 的选卡策略 (那是行为改动, 另一次对照)。")

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
        "knobs": {"min_speedup": MIN_SPEEDUP, "max_error": MAX_ERROR,
                  "max_optimize_rounds": args.rounds, "repeats": args.repeats,
                  "min_denom": MIN_DENOM, "cases": [c[0] for c in CASES]},
        "device": device,
        "aggregate": agg,
        "runs": runs,
    }
    path = out_dir / "pass_rate.json"
    # record_rounds 不再挑键, 所以血缘里万一出现一个不可序列化的值, 退化成它的 repr
    # 也比让整份报表在落盘这一步崩掉好 —— 崩在这里会连带丢掉已经付过钱的 12 轮真机数据。
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    print(f"\n报表已落盘: {path}")
    print("⚠️ 引用这些数字时必须带上 git_head 与 knobs —— 换一次配方/判据/prompt, "
          "上一份报表就只是历史, 不是基线。")
    return 0 if agg["denominator_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
