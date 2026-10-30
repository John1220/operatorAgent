"""本地零依赖演示 —— 亲眼看一下脊柱的判定 (设计 §4.3 / §10.5 / §13.4)

不需要 GPU、不需要 LLM、不联网:

    python3 benchmarks/demo_spine_local.py

五个场景回答的是**同一个问题**: 这个数字被允许造成什么后果。

  1. 同一条读数, 出处不同 → 后果不同 (五位资格掩码)
  2. 尺子自己坏了 (标定心跳) → 该窗口的读数全部失效, 但数仍被记录
  3. 判据被改写 → 拒评, 并且进「我无法确立的事」报告
  4. 永久结论量两遍 → 两次确认; 判据在两次之间变了 → 拒
  5. 报告产物 (设计 §17 第 3 件): 空章节必须带依据, 差一位就是「没能确立已到顶」

真机版 (要 GPU + `~/.ssh/config` 里的 algorithm-h20):

    python3 benchmarks/smoke_spine_h20.py

可执行文档 (与本文一一对应): tests/unit/test_oracle.py / tests/unit/test_l3_provenance.py /
tests/unit/test_spine_wiring.py / tests/unit/test_calibration.py
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from operator_agent.calibration import HeartbeatResult  # noqa: E402
from operator_agent.confirm import UnableToEstablish  # noqa: E402
from operator_agent.contract import (BaselineDef, DTypeSemantics, ShapeDist,  # noqa: E402
                                     freeze)
from operator_agent.oracle import evaluate  # noqa: E402


def contract(**over):
    kw = dict(kernel_type="matmul", dtype_semantics=DTypeSemantics(dtype="fp16"),
              baseline=BaselineDef(name="cuBLAS_fp16"),
              shapes=ShapeDist(entries=(((256, 256, 256), 1.0),)),
              max_error=1e-2, min_speedup=1.0)
    kw.update(over)
    return freeze(**kw)


def measure(value=1.8, **over):
    d = {"value": value, "validity": "valid"}
    d.update(over)
    return d


def bits(elig) -> str:
    """五位掩码, 顺序即设计 §13.3 的叙述顺序"""
    return "".join("✓" if b else "·"
                   for b in (elig.can_verdict, elig.can_tune, elig.can_persist,
                             elig.can_report, elig.can_count_progress))


def pad(text: str, width: int) -> str:
    """按**显示宽度**补齐 (CJK 占两格) —— 表格错位会让这张表读不成表"""
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - w)


def head(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ── 场景 1: 同一条读数, 出处不同 → 后果不同 ───────────────────────────────
def scene_1():
    head("场景 1  同一条读数 1.8x, 出处不同 → 后果不同 (设计 §4.3 资格推导)")
    c = contract()
    print(f"  契约哈希: {c.hash[:16]}…   (同一条契约, 下面每一行只换「出处」)\n")
    print("  ✓ = 允许   · = 不允许")
    print(f"  {pad('出处 (同一条 1.8x)', 36)}{pad('V T P R C', 12)}Oracle 记的档位")
    print("  " + " " * 36 + "│ │ │ │ └ 计一次有效迭代")
    print("  " + " " * 36 + "│ │ │ └── 对外报数")
    print("  " + " " * 36 + "│ │ └──── 写长期记忆 (永久)")
    print("  " + " " * 36 + "│ └────── 改配置/参数")
    print("  " + " " * 36 + "└──────── 判对错\n")

    cases = [
        ("干净读数", {}, "valid", "全开: 只有它能永久拉黑配方"),
        ("cv 离散度超标", {"cv_exceeded": True}, "degraded",
         "数还能用, 但**噪声不许变成永久真理**, 也不许算收敛证据"),
        ("lib_fallback (测的是库)", {"lib_fallback": True}, "valid",
         "它在测别的东西 —— 五格全关"),
        ("real_device=False 但契约要求真机", {"real_device": False}, "valid",
         "这个数**没有指称对象**: 连 passed 都不算数"),
        ("value 超过实测峰值 ×1.05", {}, "uninformative",
         "物理上不可能 —— 分母/单位错了, 五格全关"),
    ]
    rows = []
    for name, over, validity, why in cases:
        m = {"value": 1.8, "validity": validity}
        m.update(over)
        obs = evaluate(
            c, lambda m=m: m, record_hash=c.hash,
            peak=5.0 if "峰值" in name else None,
        )
        rows.append((name, bits(obs.eligibility), obs.validity,
                     obs.eligibility.can_persist, why))
    for name, b, validity, persist, why in rows:
        print(f"  {pad(name, 36)}{pad(b, 12)}{validity}")
        print(f"       └─ {why}")
    print("\n  ⚠️ 注意第一行与第二行: 数字一模一样, 后果不一样。")
    print("     「资格决定这个数被允许造成什么后果」, 而不只是「过没过」。")
    print("     这一层是**产品**, LLM 只是可替换的提议者。")


# ── 场景 2: 尺子自己坏了 ───────────────────────────────────────────────────
def scene_2():
    head("场景 2  尺子自己坏了: 标定心跳 (设计 §10.5-1 / §4.3 规则 2)")
    print("  做法: 每 N 次真值测量前, 量一对参照 (一个已知好核 + 一个故意慢 8 倍的核),")
    print("        走**远端那条路** (部署 → runner → exec → 计时 → JSON 回传)。")
    print("        两臂分不出快慢 = 尺子坏了。")
    print("        ⚠️ 边界: 参照核走 runner 的 Python/perf_counter 分支, 覆盖不到 triton")
    print("           候选的 CUDA event 分支 (要那个得按 DSL 塞一对 triton 参照核)。\n")
    from operator_agent.conductor import Conductor
    from operator_agent.kernel_db import KernelDatabase
    from operator_agent.mock_gpu import MockGPU
    from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                        TargetSpec, TaskComplexity, TaskSchema,
                                        TaskType)

    class PinnedTimer(MockGPU):
        """计时通道坏掉的假件: 快核慢核报同一个数"""

        def benchmark(self, kernel_code, inputs, warmup=3, repeat=10):
            r = super().benchmark(kernel_code, inputs, warmup, repeat)
            for k in ("min_time_ms", "median_time_ms", "avg_time_ms", "max_time_ms"):
                r[k] = 1.0
            return r

    code = ("import numpy as np\n\n\ndef kernel(A, B):\n"
            "    C = A\n    for _ in range(20):\n        C = A @ B\n    return C\n")
    sketch = {"algorithm": "matmul",
              "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
              "memory_plan": {}, "pipeline": [], "optimization_hints": []}

    class FakeLLM:
        def chat(self, messages, **kw):
            return code

        def chat_json(self, messages, **kw):
            return dict(sketch)

    task = TaskSchema(task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(framework="pytorch", kernel_type="matmul",
                                        fused_ops=[], complexity=TaskComplexity.MEDIUM),
                      target=TargetSpec(hardware="mock_gpu", dsl="python"),
                      constraints=ConstraintSpec(precision=Precision.FP32,
                                                 min_speedup=1.0, max_error=1e-4))
    import numpy as np
    rng = np.random.default_rng(0)
    inputs = {"A": rng.standard_normal((64, 64)).astype(np.float32),
              "B": rng.standard_normal((64, 64)).astype(np.float32)}

    for label, backend, expect in (("诚实计时器", MockGPU(), "尺子可信 → 读数照常可用"),
                                   ("坏计时器 (两臂同速)", PinnedTimer(),
                                    "RULER_BROKEN → 本窗读数全部失效")):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cond = Conductor(backend=backend, kernel_db=KernelDatabase(), llm=FakeLLM(),
                             max_optimize_rounds=1, heartbeat_interval=1)
            art = cond.execute(task, inputs, target_gflops=0.0)
        log = buf.getvalue()
        hb = [l.strip() for l in log.splitlines() if "[Heartbeat]" in l]
        m = art.benchmark_data or {}
        elig = (m.get("observation") or {}).get("eligibility") or {}
        print(f"  ── {label} ──  {expect}")
        for line in hb:
            print(f"     {line}")
        print(f"     产物指标: ruler_broken={m.get('ruler_broken')}"
              f"  validity={m.get('measurement_validity')}")
        if elig:
            mask = "".join("✓" if elig.get(k) else "·" for k in (
                "can_verdict", "can_tune", "can_persist", "can_report",
                "can_count_progress"))
            print(f"     资格掩码: {mask}   (数仍在, 只是不让它造成后果)")
        print()
    print("  ⚠️ 第二条里最关键的不是「拒了」, 而是**它发现了自己的尺子坏了** ——")
    print("     无人值守下这是唯一能发现「度量器悄悄坏掉」的机制。")


# ── 场景 3: 判据被改写 → 拒评 ─────────────────────────────────────────────
def scene_3():
    head("场景 3  判据被改写: 记录携带的哈希对不上 (设计 §13.4 / §10.5-5)")
    c1 = contract()
    c2 = contract(max_error=1e-1)          # 同一算子, 容差被放宽 —— 判据变了
    print(f"  记录上说: 我在 {c1.hash[:16]}… 这条契约下量的")
    print(f"  现在手上的契约是: {c2.hash[:16]}…   (max_error 1e-2 → 1e-1)")
    obs = evaluate(c2, lambda: measure(2.0), record_hash=c1.hash)
    print(f"\n  → refused={obs.refused}   hash_check={obs.hash_check!r}")
    print("  数一个都没算 (measure 根本没被调用) —— 拒评, 不是降级。\n")
    u = UnableToEstablish()
    u.extend_metrics({"contract_hash_mismatch": obs.hash_check == "mismatch"})
    print("  " + u.render().replace("\n", "\n  "))


# ── 场景 4: 永久结论量两遍 ────────────────────────────────────────────────
def scene_4():
    head("场景 4  永久结论量两遍: 两次确认 + 判据同一性 (设计 §10.5-4 / §10.2)")
    print("  §10.2 把「最后 10% 预算」留给最终确认: 任何要写进长期记忆的结论,")
    print("  必须留有**一次独立重测**的预算；而 §3.5 把「长期记忆的独立确认」列在")
    print("  **不允许配置**那一栏 —— 所以它今天没有开关, 消融只能在 harness 侧 patch。\n")
    from operator_agent.conductor import Conductor
    from operator_agent.kernel_db import KernelDatabase
    from operator_agent.mock_gpu import MockGPU
    from operator_agent.schemas import (ConstraintSpec, KernelArtifact, Precision,
                                        SourceSpec, TargetSpec, TaskSchema, TaskType)
    import numpy as np

    task = TaskSchema(task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(kernel_type="matmul"),
                      target=TargetSpec(hardware="mock_gpu", dsl="python"),
                      constraints=ConstraintSpec(precision=Precision.FP32,
                                                 min_speedup=1.0, max_error=1e-4))
    metrics = {"speedup_vs_baseline": 1.25, "measurement_validity": "valid",
               "measurement_can_gate": True, "backend_real_device": True,
               "gflops": 10.0}
    rng = np.random.default_rng(0)
    inputs = {"A": rng.standard_normal((64, 64)).astype(np.float32),
              "B": rng.standard_normal((64, 64)).astype(np.float32)}

    cond = Conductor(backend=MockGPU(), kernel_db=KernelDatabase())
    print(f"  开关: {'有' if hasattr(cond, 'require_double_confirm') else '**没有**'}"
          f"  (曾经是一个默认 False 的构造参数 ⇒ 默认配置下单次读数即可入库; I79)")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        art = KernelArtifact(kernel_id="k-demo",
                             code="def kernel(A, B):\n    return A @ B\n",
                             verification_results=[], benchmark_data=dict(metrics))
        cond._produce_remeasure(art, task, inputs, 0.0)
    rm = art.remeasure
    print(f"\n  ① 没有重测记录时 → 拒: ", end="")
    cond2 = Conductor(backend=MockGPU(), kernel_db=KernelDatabase())
    io2 = io.StringIO()
    with contextlib.redirect_stdout(io2):
        bare = KernelArtifact(kernel_id="k-bare", code="def kernel(A, B):\n    return A @ B\n",
                              verification_results=[], benchmark_data=dict(metrics))
        keep_bare = cond2._save_to_db(bare, task)
    print(f"{keep_bare}   (「没有重测就当过了」不是本仓的做法)")
    print("     " + [l for l in io2.getvalue().splitlines() if "未入库" in l or "重测" in l][-1].strip())

    print(f"\n  ② 独立重测: 首次 {metrics['speedup_vs_baseline']}x → "
          f"重测 {rm.speedup if rm.speedup is None else round(rm.speedup, 4)}x"
          f"  ({rm.validity or '未报档位'})")
    print(f"     independent={rm.independent}  ok={rm.ok}")
    print(f"     契约戳(重测那次)= {rm.contract_hash[:16]}…")
    print("     ⚠️ 诚实边界: 新的计时窗口, 同一台机器/同一份输入/同一张卡 ——")
    print("        抓得住「第一次是侥幸」, 抓不住「这把尺子系统性偏了」(那是场景 2 的活)。")

    art.benchmark_data["contract_hash"] = rm.contract_hash
    # ③④ 验的是**判据同一性**这一条门, 所以把机时钉成过关值: 64×64 上候选与基线都落在
    # ~2µs, 比值本身就是噪声 (实测在 0.96x~1.04x 之间跳), 不该让演示的结论跟着抖。
    # 真实的机时见 ①/② 与 `benchmarks/measurement_*`。
    import dataclasses
    art.remeasure = dataclasses.replace(art.remeasure, ok=True, speedup=1.4,
                                        validity="valid")
    print("     (③④ 把重测机时钉成 1.4x —— 这两步验的是判据同一性, 不验机时)")
    io3 = io.StringIO()
    with contextlib.redirect_stdout(io3):
        keep_same = cond._save_to_db(art, task)
    print(f"\n  ③ 两次同一条契约 → 入库: {keep_same}")

    art.benchmark_data["contract_hash"] = "f" * 64
    io4 = io.StringIO()
    with contextlib.redirect_stdout(io4):
        keep_other = cond._save_to_db(art, task)
    reason = [l for l in io4.getvalue().splitlines() if "未入库" in l]
    print(f"  ④ 两次之间判据被改写 → 入库: {keep_other}")
    if reason:
        print("     " + reason[-1].strip())


# ── 场景 5: 报告产物 (设计 §17 第 3 条) ──────────────────────────────────
def scene_5():
    head("场景 5  报告产物: best-known-good + 证据链 + 「我无法确立的事」(设计 §17)")
    print("  §17 的第 3 件交付物是「报告 (含『我无法确立的事』)」, 设计把它写成")
    print("  「自动跑系统的信誉所在」。看两份读数怎么变成两份不同的报告。\n")
    from operator_agent.report import build_report
    from operator_agent.schemas import KernelArtifact

    def _art(**m):
        a = KernelArtifact(kernel_id="demo0001", code="def kernel(A, B):\n    return A @ B\n")
        a.benchmark_data = dict({
            "measurement_can_gate": True, "measurement_validity": "valid",
            "speedup_vs_baseline": 1.31, "gflops": 210.0, "ruler_broken": False,
            "benchmark_contaminated": False,
            "roofline_analysis": {"reference_lower_bound": True, "peak_source": "实测探测"},
            "eligibility": {"can_verdict": True, "can_tune": True, "can_persist": True,
                            "can_report": True, "can_count_progress": True},
        }, **m)
        a.environment_drift = {"comparable": True, "drifted": False, "changed": []}
        a.iteration_history = [{"round": 1, "action": "generate",
                                "levels_passed": {"L1": True, "L2": True, "L3": True},
                                "speedup": 1.31, "gflops": 210.0,
                                "measurement_validity": "valid", "recipes_applied": []}]
        return a

    def _section3(art):
        txt = build_report(artifact=art, generated_at="2026-01-01 00:00:00")
        body = txt.split("## 3. 我无法确立的事")[1].split("## 4.")[0]
        return "\n".join("  " + l for l in body.strip().strip("`").strip().splitlines()
                          if l.strip() and l.strip() != "```")

    print("  ── ① 全项核过的读数 ──")
    print(_section3(_art()))
    print("\n  ── ② 同一条 1.31x, 但峰值分母从没实测过 (今天所有真机的状态) ──")
    print(_section3(_art(roofline_analysis={"reference_lower_bound": False,
                                            "peak_source": "规格表推导"})))
    print("\n  差别只有一位。这一位开着 ⇒ 收敛出口开着; 关着 ⇒ 除了「没测出上限」")
    print("  之外, 报告还得明说「这一轮没能确立『已经到顶』」。\n")


def main():
    print(__doc__.strip().split("\n\n")[0])
    scene_1()
    scene_2()
    scene_3()
    scene_4()
    scene_5()
    print(f"\n{'=' * 78}")
    print("  一句话: 上面每一个「拒 / 失效」都不是为了保守, 而是为了让")
    print("  写进长期记忆的那句话**只能是同一把尺子量出来的**。")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()