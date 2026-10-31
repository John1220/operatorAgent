"""LeetCUDA pilot · agent 真跑一轮 (L2-only, 不需要 GPU)

    python3 -m benchmarks.leetcuda_bench.run_agent
    python3 -m benchmarks.leetcuda_bench.run_agent --only E01,E02 --out /tmp/r.json

链路: TaskSchema → CoderAgent(LLM) → `def kernel(**inputs)` → 本地 L2 对拍。
DSL 刻意用 **python**: MockGPU 能真执行, 评到的是生成能力而不是 Triton 模拟
(`_numpy_reference` 对新算子是瞎猜的)。L3/真机是下一轮。

反污染: 只跑 split=test 的题当 holdout 分数; `--include-dev` 才跑 dev
(dev 分数可以靠 few-shot 灌出来, 不代表泛化)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo 根 (本文件在 benchmarks/leetcuda_bench/)
sys.path.insert(0, str(ROOT))

from benchmarks.leetcuda_bench.evaluate import aggregate, evaluate_one, format_curve  # noqa: E402
from benchmarks.leetcuda_bench.problems import PROBLEMS, Problem, make_inputs  # noqa: E402
from operator_agent.core.kernel_db import KernelDatabase  # noqa: E402
from operator_agent.execute.llm_client import LLMClient  # noqa: E402
from operator_agent.execute.mock_gpu import MockGPU  # noqa: E402
from operator_agent.core.schemas import (ConstraintSpec, DSLEnum, Precision, SourceSpec,  # noqa: E402
                                    TargetSpec, TaskComplexity, TaskSchema, TaskType,
                                    TilingStrategy, UnifiedSketch)
from operator_agent.decide.verifier import Verifier  # noqa: E402


def llm_from_config() -> LLMClient:
    import yaml
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        llm = (cfg.get("llm") or {})
        os.environ.setdefault("LLM_BASE_URL", str(llm.get("base_url") or ""))
        os.environ.setdefault("LLM_MODEL", str(llm.get("model") or ""))
        os.environ.setdefault("LLM_API_KEY", str(llm.get("api_key") or ""))
    client = LLMClient(temperature=0.3, max_tokens=4096)
    if getattr(client, "fallback_model", None):
        raise SystemExit("❌ 不允许走降级模型 (会污染通过率口径)")
    return client


def task_for(p: Problem) -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=p.kernel_type, complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=1.0,
                                   max_error=p.max_error),
    )


def sketch_for(p: Problem, pin_signature: bool = False) -> UnifiedSketch:
    names = ", ".join(p.input_spec)
    hints = [
        f"实现算子: {p.title} (LeetCUDA pilot {p.pid})",
        f"输入参数名必须**逐一**是: {names}  (调用约定 kernel(**inputs))",
        f"输入形状/dtype: {p.input_spec}",
        "函数名必须是 kernel; 用 **kwargs 或显式同名参数接住上述输入",
        "输出 numpy.ndarray (0-d 可, 如 reduce/dot 的标量)",
        "只用 numpy, 不要 import torch",
    ]
    if pin_signature:
        hints.insert(0, f"第一行必须是: def kernel({names}):")
        hints.append("禁止写成 def kernel(A, B) 或 def kernel(x) —— 参数名不一致即判失败")
    return UnifiedSketch(
        algorithm=p.kernel_type,
        target_dsl="python",
        tiling=TilingStrategy(),
        optimization_hints=hints,
        hints_this_version=[
            f"算子语义: {p.title}; 签名 def kernel({names}): ...",
        ],
    )


def generate_one(p: Problem, coder, llm, pin_signature: bool = False) -> dict:
    """一次生成 + 本地 L2。返回评分记录 (含 code 与 token 代价)。"""
    t0 = time.time()
    rec = {"pid": p.pid, "title": p.title, "difficulty": p.difficulty,
           "family": p.family, "split": p.split, "passed": False,
           "max_error": None, "first_bad_index": None, "message": "",
           "code_chars": 0, "llm_calls": 0, "wall_s": 0.0}
    try:
        code = coder.run(sketch_for(p, pin_signature=pin_signature), task_for(p))
        rec["code_chars"] = len(code or "")
        rec["def_line"] = next((ln.strip() for ln in (code or "").splitlines()
                                if ln.strip().startswith("def ")), "")
        v = Verifier(MockGPU())
        scored = evaluate_one(p, code, v)
        rec.update({k: scored[k] for k in
                    ("passed", "max_error", "first_bad_index", "message", "fail_kind")})
    except Exception as exc:  # noqa: BLE001
        rec["message"] = f"{type(exc).__name__}: {exc}"[:200]
        from benchmarks.leetcuda_bench.evaluate import classify_failure
        rec["fail_kind"] = classify_failure(rec)
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔 pid (默认全部 test 题)")
    ap.add_argument("--include-dev", action="store_true",
                    help="连 dev 题一起跑 (dev 分数不代表泛化)")
    ap.add_argument("--pin-signature", action="store_true",
                    help="提示词里钉死 def kernel(参数名) —— 接口 A/B 的第二臂")
    ap.add_argument("--out", default="", help="结果 JSON 落盘")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    pool = [p for p in PROBLEMS if (not only or p.pid in only)
            and (args.include_dev or p.split == "test" or only)]
    if not pool:
        print("没有可跑的题")
        return 2

    llm = llm_from_config()
    print(f"[leetcuda] model={getattr(llm, 'model', '?')}  题数={len(pool)}")
    from operator_agent.generate.agents import CoderAgent
    coder = CoderAgent(kernel_db=KernelDatabase(), llm=llm)

    runs = []
    for i, p in enumerate(pool, 1):
        print(f"\n{'=' * 62}\n[{i}/{len(pool)}] {p.pid} {p.title} "
              f"({p.difficulty} / {p.split})\n{'=' * 62}")
        rec = generate_one(p, coder, llm, pin_signature=args.pin_signature)
        runs.append(rec)
        mark = "✅" if rec["passed"] else "❌"
        print(f"  {mark} {rec['message'][:140]}")
        print(f"  chars={rec['code_chars']}  wall={rec['wall_s']}s")

    agg = aggregate(runs)
    print(format_curve(agg))

    out = args.out or str(ROOT / "output" / "leetcuda_bench"
                          / time.strftime("%Y%m%d_%H%M%S") / "results.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"runs": runs, "aggregate": agg},
                                    ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    print(f"\n结果已落盘: {out}")
    print(f"离线复现曲线: python3 -m benchmarks.leetcuda_bench --from-json {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
