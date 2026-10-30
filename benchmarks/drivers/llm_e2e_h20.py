"""Stage 4 第一步: 代码型路径的 LLM 端到端 —— 真 LLM + 真机, spine 首次

spine 至今全程 LLM-free (I72/I74 的模板路径刻意不经 LLM; doc 79 复跑亦然)。
这一跑把**生成侧**第一次点亮: Designer → CoderPipeline (Planner→3 路采样→
compile-gate 剪枝→修复) → 真机 L1/L2/L3 → (败则) Fixer / Search —— 全部走
真 LLM (config.yaml 的 llm 段, algorithm-h20 上的 vLLM 免 key 实例) 与真机。

判据 (每条独立报; 「有判定」= 过或败都算, 生成质量是另一个问题):
  C1 端点预检通过 (10 token 试探, 报时延)
  C2 Designer 有花费 (chat_json 成功 ⇒ sketch 产出; 失败会在重试后整跑失败)
  C3 CoderPipeline 真的走了 (血缘 generation_candidates >= 1)
  C4 L1 有判定 (真机编译; 全灭则 C4 ✗ 并如实报 Fixer 足迹)
  C5 L2 有判定
  C6 L3 有判定 (读数一并报: speedup / 有效性档位)
  C7 llm_cost 按 caller_tag 落盘 (Designer/Coder/Planner/Fixer/Search 分列)

口径: matmul 4096³ fp32 / triton / min_speedup=0.5 (搜索臂口径: LLM 初版
triton 大概率过不了 0.5x ⇒ 正好观测 Search/停滞/升级门的真机首秀; 若过了
也是读数) / max_rounds=3 / 心跳保持默认 (第 5 次真值测量若到, 心跳的真机
观测是白捡的 —— I74 的边界顺带补)。

用法: python3 benchmarks/drivers/llm_e2e_h20.py [--rounds 3] [--min-speedup 0.5]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from operator_agent._harness_controls import enable_for_harness
enable_for_harness()
os.environ.setdefault("OPERATOR_AGENT_KERNEL_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_FAILURE_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_LOG_KEEP", "100")

import numpy as np  # noqa: E402
import yaml  # noqa: E402  (本地依赖, 只为读 config.yaml; 仓内只有 example)

from operator_agent import Conductor, LLMClient, RemoteExecutor  # noqa: E402
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,  # noqa: E402
                                    TaskSchema, TaskComplexity, TaskType, TargetSpec)


def _llm_from_config():
    """端点只活在本地 config.yaml (未跟踪), 仓内只有 config.example.yaml。"""
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        sys.exit(f"缺 {cfg_path}: 请从 config.example.yaml 复制并填 llm.base_url")
    llm_cfg = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("llm") or {}
    if not llm_cfg.get("base_url"):
        sys.exit("config.yaml 的 llm.base_url 为空")
    return LLMClient(base_url=llm_cfg["base_url"],
                     model=llm_cfg.get("model", "Qwen3.8-27B"),
                     temperature=llm_cfg.get("temperature", 0.3),
                     max_tokens=llm_cfg.get("max_tokens", 4096),
                     timeout=llm_cfg.get("timeout", 120))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--min-speedup", type=float, default=0.5)
    args = ap.parse_args()

    llm = _llm_from_config()
    t0 = time.monotonic()
    pong = llm.chat([{"role": "user", "content": "回复两个字: 在线"}],
                    max_tokens=8, caller_tag="Preflight")
    preflight_s = round(time.monotonic() - t0, 1)
    print(f"[预检] {llm.model} @ {llm.base_url} → {pong.strip()!r} ({preflight_s}s)",
          flush=True)

    task = TaskSchema(task_id="llm-e2e-matmul",
                      task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(kernel_type="matmul",
                                        complexity=TaskComplexity.MEDIUM),
                      target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
                      constraints=ConstraintSpec(precision=Precision.FP32,
                                                 min_speedup=args.min_speedup,
                                                 max_error=1e-2))
    rng = np.random.default_rng(11)     # 与 replay/e2e/exhaustion 同一种子
    inputs = {"A": rng.standard_normal((4096, 4096)).astype(np.float32),
              "B": rng.standard_normal((4096, 4096)).astype(np.float32)}

    conductor = Conductor(backend=RemoteExecutor(mode="ssh"), llm=llm,
                          max_optimize_rounds=args.rounds)
    t1 = time.monotonic()
    art = conductor.execute(task, inputs, target_gflops=0.0)
    wall = time.monotonic() - t1

    meta = art.metadata or {}
    hist = art.iteration_history
    rounds = [{
        "round": h.get("round", i + 1),
        "action": h.get("action"),
        "candidates": h.get("generation_candidates"),
        "levels": h.get("levels_passed"),
        "gflops": h.get("gflops"),
        "l2_runs": h.get("real_l2_runs"),
    } for i, h in enumerate(hist)]
    # llm_cost: result.json 落在**顶层** (ArtifactStore 把 metadata.llm_cost 提升),
    # 进程内则在 artifact.metadata —— 两个口径都看; 分解的键形 (扁平/嵌套) 不猜,
    # 原样进 RESULT, 判据只用「有量」这种结构事实。
    cost = (getattr(art, "metadata", None) or {}).get("llm_cost") or {}
    if not cost:
        top = getattr(art, "llm_cost", None)
        cost = top if isinstance(top, dict) else {}
    any_level = lambda lv: any(lv in (r["levels"] or {}) for r in rounds)
    deepest = ("L3" if any_level("L3") else "L2" if any_level("L2")
               else "L1" if any_level("L1") else "无")
    out = {"wall_s": round(wall, 1), "preflight_s": preflight_s,
           "stop": meta.get("stop"), "llm_cost": cost, "rounds": rounds}
    print("RESULT " + json.dumps(out, ensure_ascii=False, default=str), flush=True)

    def _calls(*tags):
        # 兼容两种分解形状: {tag: {...}} 或 {by_caller: {tag: {...}}}
        by_caller = cost.get("by_caller") if isinstance(cost.get("by_caller"), dict) else cost
        return sum((by_caller.get(t) or {}).get("calls", 0) for t in tags)

    total_calls = cost.get("total_calls") or sum(
        (v or {}).get("calls", 0) for v in
        (cost.get("by_caller") or cost).values() if isinstance(v, dict))

    checks = {
        "C1 端点预检通过": bool(pong.strip()),
        "C2 生成侧有 LLM 花费 (Designer/Coder/Planner)": total_calls >= 1,
        "C3 CoderPipeline 走了 (candidates>=1)": any(
            isinstance(r["candidates"], int) and r["candidates"] >= 1 for r in rounds),
        "C4 L1 有判定 (真机编译)": any_level("L1"),
        "C5 L2 有判定": any_level("L2"),
        "C6 L3 有判定": any_level("L3"),
        "C7 llm_cost 按 caller_tag 落盘": isinstance(cost, dict) and len(cost) >= 2,
    }
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}", flush=True)
    print(f"  (读数) 走到的最深层级: {deepest or '无'}; stop={meta.get('stop')}; "
          f"LLM 各环节: {json.dumps({t: (cost.get(t) or {}).get('calls', 0) for t in ('Designer', 'Planner', 'Coder', 'Fixer', 'Search')}, ensure_ascii=False)}",
          flush=True)
    ok = all(checks.values())
    print(f"VERDICT {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
