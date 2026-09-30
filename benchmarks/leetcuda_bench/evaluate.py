"""LeetCUDA pilot 评分器: 对一题跑 L2, 聚合成难度曲线

用法:
  python3 -m benchmarks.leetcuda_bench --self-test
      # 用每题的参考当「oracle 核」跑一遍 —— 验证评分器本身不是空转

  python3 -m benchmarks.leetcuda_bench --from-json results.json
      # 读 agent 产出的逐题结果, 打难度曲线 (L2 通过率 by difficulty / family)

  python3 -m benchmarks.leetcuda_bench --check-fixture broken.json
      # 用故意写错的核验「评分会挂」(方向检查)

结果 JSON 形状 (agent 侧产出):
  {"runs": [{"pid": "E01", "passed": true, "max_error": 1e-7,
             "first_bad_index": null, "message": "...", "code": "..."}]}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from benchmarks.leetcuda_bench.problems import (PROBLEMS, Problem, by_id,  # noqa: E402
                                                make_inputs)
from operator_agent.mock_gpu import MockGPU  # noqa: E402
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,  # noqa: E402
                                    TargetSpec, TaskComplexity, TaskSchema, TaskType)
from operator_agent.verifier import Verifier  # noqa: E402


def task_for(p: Problem) -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=p.kernel_type, complexity=TaskComplexity.MEDIUM,
                          reference_source=_reference_source_for(p)),
        target=TargetSpec(hardware="mock_gpu", dsl=p.dsl),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=1.0,
                                   max_error=p.max_error),
    )


def _reference_source_for(p: Problem) -> str:
    """把题目的 NumPy 参考包成可传输的 `def reference(**inputs)`。

    评分器走**外挂参考**路径 (P1), 不经过 kernel_type 内建表 —— 这样
    `rope`/`hardshrink`/`embedding` 这些表里没有的算子也能 L2。
    """
    # 参考函数是闭包, 不能直接当源码传; 这里用「题目 id → 在评分器侧调 p.reference」
    # 的桩: 真正的比对在 evaluate_one 里做 (本地直接调 p.reference),
    # reference_source 只为打通 Verifier 的外挂参考通道。
    return (
        "def reference(**inputs):\n"
        "    import numpy as np\n"
        f"    # {p.pid} {p.title}\n"
        "    raise RuntimeError('评分器本地直调 reference, 不走这条远端通道')\n"
    )


def oracle_code(p: Problem) -> str:
    """把 p.reference 包成 `def kernel(**inputs)` —— 评分器的已知答案。"""
    return (
        "import numpy as np\n"
        "def kernel(**inputs):\n"
        f"    from benchmarks.leetcuda_bench.problems import by_id\n"
        f"    return by_id({p.pid!r}).reference(**inputs)\n"
    )


def broken_code(p: Problem) -> str:
    """故意写错的核: 输出乘 0.5 —— L2 必须挂。"""
    return (
        "import numpy as np\n"
        "def kernel(**inputs):\n"
        f"    from benchmarks.leetcuda_bench.problems import by_id\n"
        f"    out = by_id({p.pid!r}).reference(**inputs)\n"
        "    return np.asarray(out, dtype=np.float32) * 0.5\n"
    )


def evaluate_one(p: Problem, code: str, verifier: Verifier | None = None) -> dict:
    """本地 L2: 用 p.reference 直接比, 不依赖远端。返回可落盘的记录。

    输入介质: 生产路径 (`run_kernel.py`) 喂的是 **torch.Tensor**; 本评分器先用
    numpy (MockGPU 原生), 若错误像是「torch API 打在 numpy 上」(`.to` / `clamp` /
    `mean(dim=`) 就换成 torch 重试 —— 否则评到的是「框架选错」而不是算子能力。
    """
    v = verifier or Verifier(MockGPU())
    rec = {
        "pid": p.pid, "title": p.title, "difficulty": p.difficulty,
        "family": p.family, "split": p.split, "passed": False,
        "max_error": None, "first_bad_index": None, "message": "",
        "wall_s": 0.0,
    }
    t0 = time.time()
    xs = make_inputs(p, seed=0)
    expected = p.reference(**xs)
    try:
        r = v._verify_function(code, xs, expected, task_for(p))
        rec["message"] = r.message
        if (not r.passed) and _looks_like_torch_on_numpy(r.message) and _torch_ok():
            r = v._verify_function(code, _to_torch(xs), expected, task_for(p))
            rec["message"] = r.message + "  [arm=torch inputs]"
        rec["passed"] = bool(r.passed)
        rec["max_error"] = r.metrics.get("max_error")
        rec["first_bad_index"] = r.metrics.get("first_bad_index")
    except Exception as exc:  # noqa: BLE001
        rec["message"] = f"{type(exc).__name__}: {exc}"
    rec["fail_kind"] = classify_failure(rec)
    rec["wall_s"] = round(time.time() - t0, 4)
    return rec


def _looks_like_torch_on_numpy(msg: str) -> bool:
    return any(s in (msg or "") for s in (
        "'numpy.ndarray' object has no attribute 'to'",
        "clamp() received an invalid combination",
        "mean() received an invalid combination",
        "numpy.ndarray' object has no attribute",
        "must be Tensor, not numpy.ndarray",
        "must be Tensor, not numpy",
    ))


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _to_torch(xs: dict) -> dict:
    import torch
    out = {}
    for k, v in xs.items():
        if np.asarray(v).dtype.kind in "iu":
            out[k] = torch.from_numpy(np.asarray(v).astype(np.int64))
        else:
            out[k] = torch.from_numpy(np.asarray(v, dtype=np.float32))
    return out


def classify_failure(rec: dict) -> str | None:
    """失败分类: interface (签名/契约) vs numeric (算错) vs crash (崩了) vs None (过了)。

    LeetCUDA pilot 首跑抓到的形态: GEMM (A,B) 全过, 单输入 `X` 的题 100% 挂在
    `kernel() got an unexpected keyword argument 'X'` —— 那不是算术问题是**契约**
    问题, 混在一条曲线里会把「easy 0%」读成「连 relu 都不会」。
    """
    if rec.get("passed"):
        return None
    msg = rec.get("message") or ""
    if ("unexpected keyword" in msg or "missing" in msg and "argument" in msg
            or "interface" in msg.lower() or "契约" in msg):
        return "interface"
    if "执行失败" in msg or "Error" in msg or "Exception" in msg or "SyntaxError" in msg:
        return "crash"
    if "误差" in msg or "max_error" in msg or "个元素" in msg:
        return "numeric"
    return "other"


def run_suite(codes: dict[str, str], problems=None) -> list:
    """codes: {pid: kernel_source}。缺的题记 passed=False, message='未提供'。"""
    v = Verifier(MockGPU())
    problems = problems or PROBLEMS
    out = []
    for p in problems:
        code = codes.get(p.pid)
        if code is None:
            out.append({"pid": p.pid, "title": p.title, "difficulty": p.difficulty,
                        "family": p.family, "split": p.split, "passed": False,
                        "max_error": None, "first_bad_index": None,
                        "message": "未提供候选核", "wall_s": 0.0})
        else:
            out.append(evaluate_one(p, code, v))
    return out


# ─── 聚合: 难度曲线 ───────────────────────────────────────────

def aggregate(runs: list) -> dict:
    by_diff = defaultdict(lambda: [0, 0])
    by_family = defaultdict(lambda: [0, 0])
    by_split = defaultdict(lambda: [0, 0])
    for r in runs:
        for key, table in ((r.get("difficulty", "?"), by_diff),
                           (r.get("family", "?"), by_family),
                           (r.get("split", "?"), by_split)):
            table[key][1] += 1
            if r.get("passed"):
                table[key][0] += 1

    def _rows(table):
        return [{"key": k, "passed": v[0], "total": v[1],
                 "pct": round(100.0 * v[0] / v[1], 1) if v[1] else None}
                for k, v in sorted(table.items())]

    passed = sum(1 for r in runs if r.get("passed"))
    by_kind = defaultdict(lambda: [0, 0])
    for r in runs:
        k = r.get("fail_kind") or "pass"
        by_kind[k][1] += 1
        if r.get("passed"):
            by_kind[k][0] += 1
    return {
        "n": len(runs),
        "passed": passed,
        "pct": round(100.0 * passed / len(runs), 1) if runs else None,
        "by_difficulty": _rows(by_diff),
        "by_family": _rows(by_family),
        "by_split": _rows(by_split),
        "by_fail_kind": {k: v[1] for k, v in sorted(by_kind.items())},
        "failures": [{"pid": r.get("pid"), "difficulty": r.get("difficulty"),
                      "fail_kind": r.get("fail_kind"),
                      "message": (r.get("message") or "")[:120],
                      "first_bad_index": r.get("first_bad_index")}
                     for r in runs if not r.get("passed")],
    }


def format_curve(agg: dict) -> str:
    lines = ["", "=" * 62, "LeetCUDA pilot · 难度曲线", "=" * 62]
    lines.append(f"总分: {agg['passed']}/{agg['n']}"
                 + (f" = {agg['pct']}%" if agg["pct"] is not None else ""))
    lines.append("\n【按难度】")
    for row in agg["by_difficulty"]:
        lines.append(f"  {row['key']:8s}  {row['passed']}/{row['total']}"
                     + (f" = {row['pct']}%" if row["pct"] is not None else ""))
    lines.append("\n【按家族】(anti-contamination: 同族不许一半 train 一半 test)")
    for row in agg["by_family"]:
        lines.append(f"  {row['key']:12s}  {row['passed']}/{row['total']}")
    lines.append("\n【按 split】")
    for row in agg["by_split"]:
        lines.append(f"  {row['key']:8s}  {row['passed']}/{row['total']}"
                     + (f" = {row['pct']}%" if row["pct"] is not None else ""))
    if agg.get("by_fail_kind"):
        lines.append("\n【失败分类】interface=签名/契约  numeric=算错  crash=崩  other")
        for k, n in agg["by_fail_kind"].items():
            lines.append(f"  {k:10s}  {n}")
    if agg["failures"]:
        lines.append("\n【失败】")
        for f in agg["failures"][:12]:
            fb = f["first_bad_index"]
            lines.append(f"  {f['pid']} [{f['difficulty']}/{f.get('fail_kind')}] "
                         f"{f['message']}"
                         + (f"  first_bad@{fb}" if fb is not None else ""))
    lines.append("\n⚠️ holdout (split=test) 的分数才是泛化能力; "
                 "dev 分数可以靠 few-shot 灌出来。")
    return "\n".join(lines)


# ─── CLI ───────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="LeetCUDA pilot 评分器")
    ap.add_argument("--self-test", action="store_true",
                    help="用每题的参考当 oracle 核跑一遍 (评分器非空转检查)")
    ap.add_argument("--check-fixture", action="store_true",
                    help="用 ×0.5 的坏核跑一遍, 断言全部 L2 挂")
    ap.add_argument("--from-json", default="", help="读 agent 逐题结果, 打曲线")
    ap.add_argument("--out", default="", help="结果落盘路径")
    args = ap.parse_args()

    if args.self_test:
        codes = {p.pid: oracle_code(p) for p in PROBLEMS}
        runs = run_suite(codes)
        agg = aggregate(runs)
        print(format_curve(agg))
        # oracle 必须全过 —— 否则评分器/参考有 bug
        bad = [r for r in runs if not r["passed"]]
        if bad:
            print(f"\n❌ oracle 有 {len(bad)} 题没过 —— 评分器或参考有 bug")
            return 1
        print("\n✅ oracle 全过: 评分器与参考一致, 曲线可信")
        if args.out:
            Path(args.out).write_text(json.dumps({"runs": runs}, ensure_ascii=False,
                                                 indent=2, default=str),
                                      encoding="utf-8")
            print(f"结果已落盘: {args.out}")
        return 0

    if args.check_fixture:
        codes = {p.pid: broken_code(p) for p in PROBLEMS}
        runs = run_suite(codes)
        n_pass = sum(1 for r in runs if r["passed"])
        print(f"坏核 (×0.5) 通过 {n_pass}/{len(runs)} —— 期望 0")
        if n_pass != 0:
            print("❌ 评分器空转: 坏核也过了")
            return 1
        print("✅ 方向检查: 坏核全挂, first_bad 被填上")
        sample = next(r for r in runs if r.get("first_bad_index") is not None)
        print(f"   样例 {sample['pid']}: first_bad@{sample['first_bad_index']} "
              f"max_err={sample['max_error']:.3g}")
        return 0

    if args.from_json:
        data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        agg = aggregate(data.get("runs") or [])
        print(format_curve(agg))
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
