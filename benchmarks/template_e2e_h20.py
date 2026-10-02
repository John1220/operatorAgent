"""模板路径的真机端到端（§12: Conductor + TritonTemplateSSHExecutor → H20）

回答的问题（批次审查 No.2 的收口 / I70 边界的另一半）: Stage 3 的六片机制
(候选队列 → ε-greedy 选择 → bandit 回报 → 停止条件) 此前只在**假后端**上过过
Conductor 主循环, 「点 → 源码 → 评测器」也只在**回放脚本**里过过真机 —— 两条
从没合在一起。这一跑把真组合后端 (`TritonTemplateSSHExecutor`, I71 修后其在
全新进程可构造) 交进 Conductor 主循环, 于是第一次能同时观察到:

  C1  模板轮真的跳过 Coder/Fixer/Search (LLM 的 chat 一旦被调就是读数, 不是噪音)
  C2  每轮 action=template 且臂 = 参数点的去重键 (bandit_arm 与候选身份同源)
  C3  达标臂 (min_speedup=0.1): 第 1 轮 L1/L2/L3 全过 ⇒ stop=passed
  C4  搜索臂 (min_speedup=0.5, 实测点位 0.32~0.37x ⇒ 必不过): 连续多轮**换点**试
      (队列/bandit 真的在动, 不是同一个点跑到死)
  C5  血缘里 template_selection 落盘 (顺位/选中点可复盘)

刻意不经 LLM (与 doc 79 同一理由: 不引入 LLM_API_KEY 依赖, 读数才可重建) ——
Designer 那一次 chat_json 由桩供给; chat 被调 ⇒ Coder/Fixer/Search 混进了模板
路径, 让它炸出来而不是悄悄容许。

用法:
    python3 benchmarks/template_e2e_h20.py                # 两臂都跑
    python3 benchmarks/template_e2e_h20.py --arm pass
    python3 benchmarks/template_e2e_h20.py --arm search
    python3 benchmarks/template_e2e_h20.py --rounds 3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 驱动跑的现场不进语料 (与 template_matmul_replay 同款理由: 语料是「跨运行的记忆」,
# 驱动现场不是要沉淀的实验); 日志保留数抬高 (conftest 记过的事故: keep=10 会把真实
# 实验现场轮掉 —— 这里是真机跑, 现场要留着)。
os.environ.setdefault("OPERATOR_AGENT_KERNEL_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_FAILURE_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_LOG_KEEP", "100")

import numpy as np  # noqa: E402

from operator_agent.conductor import Conductor  # noqa: E402
from operator_agent.remote_executor import RemoteExecutor  # noqa: E402
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,  # noqa: E402
                                    TaskSchema, TaskComplexity, TaskType, TargetSpec)
from operator_agent.triton_matmul_template import TritonTemplateSSHExecutor  # noqa: E402


class _DesignerOnlyLLM:
    """模板端到端的 LLM 桩: 只供 Designer 那一次 sketch (循环结构需要它), chat 一律炸。

    chat 被调 = Coder/Fixer/Search 进了模板路径 —— 按 §12 那不该发生, 所以用异常
    把它变成当场可见的读数 (C1 的观察面), 而不是让桩悄悄答点什么把路径走完。
    """

    def __init__(self):
        self.chat_calls = 0
        self.chat_json_calls = 0

    def chat(self, *a, **k):
        self.chat_calls += 1
        raise AssertionError("模板端到端不许调 LLM chat (Coder/Fixer/Search 未被跳过)")

    def chat_json(self, *a, **k):
        self.chat_json_calls += 1
        # sketch 对模板轮只是循环结构的一环 (源码来自实例化器), 值本身不进判决;
        # 取与队列测试同一个形状, 免得给 Designer 的解析器喂一个没见过的样子。
        return {"algorithm": "matmul",
                "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                "memory_plan": {}, "pipeline": [], "optimization_hints": []}


def _task(min_speedup: float) -> TaskSchema:
    return TaskSchema(
        task_id=f"tmpl-e2e-ms{min_speedup}",
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
        constraints=ConstraintSpec(precision=Precision.FP32,
                                   min_speedup=min_speedup, max_error=1e-2),
    )


def _inputs():
    # 与 template_matmul_replay 同一种子与形状 ⇒ 两边的读数可以直接对表 (I70 的
    # 两个点: 128x128x32s2w8 = 6.108ms / 64x64x64s3w4 = 5.209ms)。
    rng = np.random.default_rng(11)
    return {"A": rng.standard_normal((4096, 4096)).astype(np.float32),
            "B": rng.standard_normal((4096, 4096)).astype(np.float32)}


def _backend() -> TritonTemplateSSHExecutor:
    """组合后端的第一个生产消费者 (I70 边界: Conductor 的模板接线此前只在假后端上)。

    smem 上界取 H20 的 228 KiB/SM —— 空间缩格与剪枝用的是同一张真实规格,
    不另造一个「驱动专属」的数。
    """
    return TritonTemplateSSHExecutor(RemoteExecutor(mode="ssh"),
                                     smem_limit_bytes=228 * 1024, elem_bytes=4)


def run_arm(name: str, min_speedup: float, rounds: int):
    print(f"\n{'=' * 62}\n臂 {name}: min_speedup={min_speedup}, max_rounds={rounds}\n{'=' * 62}",
          flush=True)
    llm = _DesignerOnlyLLM()
    conductor = Conductor(backend=_backend(), llm=llm, max_optimize_rounds=rounds)
    t0 = time.monotonic()
    art = conductor.execute(_task(min_speedup), _inputs(), target_gflops=0.0)
    wall = time.monotonic() - t0

    hist = art.iteration_history
    stop = (art.metadata or {}).get("stop") or {}
    rounds_seen = [{
        "round": h.get("round", i + 1),
        "action": h.get("action"),
        "point": (h.get("template_selection") or {}).get("chosen"),
        "arm": h.get("bandit_arm"),
        "phase": h.get("phase"),
        "levels": h.get("levels_passed"),
        "gflops": h.get("gflops"),
        "measure_s": h.get("measurement_wall_s"),
    } for i, h in enumerate(hist)]
    out = {"arm": name, "min_speedup": min_speedup, "wall_s": round(wall, 1),
           "stop": stop, "llm": {"chat": llm.chat_calls,
                                 "chat_json": llm.chat_json_calls},
           "rounds": rounds_seen}
    print("RESULT " + json.dumps(out, ensure_ascii=False, default=str), flush=True)

    # ---- 判据: 每条独立报, 不折叠成一个数 (一个红了要知道是哪个) ----
    checks = {
        "C1 chat 从未被调 (Coder/Fixer/Search 被跳过)": llm.chat_calls == 0,
        "C2 每轮 action=template 且臂非空": bool(rounds_seen) and all(
            r["action"] == "template" and r["arm"] for r in rounds_seen),
        "C5 template_selection 落进血缘": bool(rounds_seen) and all(
            r["point"] for r in rounds_seen),
    }
    if name == "pass":
        checks["C3 第 1 轮 L1/L2/L3 全过 ⇒ stop=passed"] = (
            stop.get("reason") == "passed" and bool(rounds_seen)
            and all(rounds_seen[0]["levels"].get(k) for k in ("L1", "L2", "L3")))
    else:
        arms = {json.dumps(r["arm"], default=str) for r in rounds_seen}
        checks["C4 >=2 轮且换了点 (队列/bandit 在动)"] = (
            len(rounds_seen) >= 2 and len(arms) >= 2)
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}", flush=True)
    ok = all(checks.values())
    print(f"VERDICT {name} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["pass", "search"], default=None,
                    help="只跑一臂 (缺省两臂都跑)")
    ap.add_argument("--rounds", type=int, default=3,
                    help="max_optimize_rounds (搜索臂的多轮行为靠它)")
    args = ap.parse_args()
    arms = [("pass", 0.1), ("search", 0.5)] if args.arm is None else [
        ("pass", 0.1) if args.arm == "pass" else ("search", 0.5)]
    sys.exit(0 if all(run_arm(n, ms, args.rounds) for n, ms in arms) else 1)


if __name__ == "__main__":
    main()
