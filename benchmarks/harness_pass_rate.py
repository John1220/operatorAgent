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
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
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

    admitted = [r for r in ok_runs if r.get("admitted")]

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
        "validity_tiers": tiers,
        "decision_grade": (len(decision_grade), len(graded)),
        "decision_grade_pct": _pct(len(decision_grade), len(graded)),
        "speedup_all": _quantiles(sp_all),
        "speedup_decision_grade": _quantiles(sp_grade),
        "admitted": (len(admitted), len(passed_runs)),
        "admitted_pct": _pct(len(admitted), len(passed_runs)),
        "admit_rejections": sorted({r["admit_reason"] for r in ok_runs
                                    if r.get("admit_reason") and not r.get("admitted")}),
        "cost": {
            "tokens": sum(int(r.get("tokens") or 0) for r in ok_runs),
            "usd": round(sum(float(r.get("cost_usd") or 0.0) for r in ok_runs), 4),
            "wall_s": round(sum(float(r.get("wall_s") or 0.0) for r in ok_runs), 1),
        },
        "denominator_ok": min(len(ok_runs), len(rounds)) >= MIN_DENOM,
    }


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

    c = agg["cost"]
    lines.append(f"\n【成本】{c['tokens']} tokens ≈ ${c['usd']} · 墙钟 {c['wall_s']} s")
    return "\n".join(lines)


# ─── 真机执行 ─────────────────────────────────────────────────

def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "?"
    except Exception:  # noqa: BLE001
        return "?"


def run_one(label, kernel_type, precision, shape, backend, llm, rounds):
    """跑一次完整任务, 返回一份原始记录 (不聚合, 便于落盘复盘)"""
    rec = {"case": label, "kernel_type": kernel_type, "shape": list(shape),
           "ok": False, "error": None, "rounds": [], "final_all_passed": False,
           "rounds_to_pass": None, "admitted": False, "admit_reason": "",
           "tokens": 0, "cost_usd": 0.0, "wall_s": 0.0}
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
    rec["rounds"] = [
        {"round": h.get("round"), "action": h.get("action"),
         "all_passed": bool(h.get("all_passed")),
         "levels_passed": h.get("levels_passed") or {},
         "measurement_validity": h.get("measurement_validity"),
         "gflops": h.get("gflops"), "speedup": h.get("speedup")}
        for h in (art.iteration_history or [])
    ]
    # `KernelArtifact.all_passed` 是 `all(v.passed for v in verification_results)`,
    # 而 **all([]) 是 True** —— 一份没有任何验证结果的产物会被它判成「全过」。
    # 两条早退路径 (入口校验失败 / 环境自检未通过) 都填了合成的 L1 失败, 所以现在踩不到;
    # 但「跑出了候选」本来就该是任务成功的定义之一, 所以这里显式要求至少有一轮。
    rec["final_all_passed"] = bool(rec["rounds"]) and bool(getattr(art, "all_passed", False))
    rec["rounds_to_pass"] = next((r["round"] for r in rec["rounds"]
                                  if r["all_passed"] and r["action"] != "fix"), None)
    # 入库决定在 Conductor 里只打印、不落产物, 所以这里用同一个纯函数复算一遍。
    # 复算而不是解析 stdout: 后者会把报表绑在日志文案上。
    keep, reason = persist_decision(art.benchmark_data or {})
    rec["admitted"], rec["admit_reason"] = bool(keep), reason
    cost = (art.metadata or {}).get("llm_cost") or {}
    rec["tokens"] = cost.get("total_tokens", 0)
    rec["cost_usd"] = cost.get("cost_usd", 0.0)
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

    llm = llm_from_config()
    backend = RemoteExecutor(mode="ssh", timeout=900)

    runs = []
    for rep in range(args.repeats):
        for kernel_type, precision, shape in CASES:
            label = f"{kernel_type}/{precision.value}/{'x'.join(map(str, shape))}#{rep + 1}"
            print(f"\n{'=' * 62}\n[{label}] 真跑 (门槛 {MIN_SPEEDUP}x, 最多 {args.rounds} 轮)\n{'=' * 62}")
            rec = run_one(label, kernel_type, precision, shape, backend, llm, args.rounds)
            runs.append(rec)
            print(f"[{label}] ok={rec['ok']} 轮数={len(rec['rounds'])} "
                  f"最终通过={rec['final_all_passed']} rounds_to_pass={rec['rounds_to_pass']} "
                  f"入库={rec['admitted']}")
            if rec["error"]:
                print(f"  ❌ {rec['error']}")

    agg = aggregate(runs)
    report = format_report(agg)
    print(report)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
        "knobs": {"min_speedup": MIN_SPEEDUP, "max_error": MAX_ERROR,
                  "max_optimize_rounds": args.rounds, "repeats": args.repeats,
                  "min_denom": MIN_DENOM, "cases": [c[0] for c in CASES]},
        "gpu": os.environ.get("OPERATOR_AGENT_GPU", "(未固定)"),
        "aggregate": agg,
        "runs": runs,
    }
    path = out_dir / "pass_rate.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报表已落盘: {path}")
    print("⚠️ 引用这些数字时必须带上 git_head 与 knobs —— 换一次配方/判据/prompt, "
          "上一份报表就只是历史, 不是基线。")
    return 0 if agg["denominator_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
