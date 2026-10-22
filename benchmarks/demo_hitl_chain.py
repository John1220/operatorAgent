"""冒烟：判定层能不能**不经过 Conductor** 被外层驱动串起来（零 LLM / 零 GPU / 不联网）

    python3 benchmarks/demo_hitl_chain.py

为什么有这个演示（`docs/dsh-plugin-evaluation.md` §9.5 第 1 条）：

本仓的 `Conductor` 是一个**确定性外部编排器** —— LLM 只是工位上的工人。要把「前期
HITL（人看得见每一步、能干预）」做成「外层每步一个模型回合」，先得回答一个工程问题：

> 判定层（L1/L2/L3 + 测量有效性 + 资格 + 反作弊 + 契约哈希）能不能被一条**不经过
> Conductor** 的链串起来？能，HITL 形态就不是重写；不能，一切免谈。

这个脚本把那条链用最小依赖跑一遍：

    IntentParser(NL)  →  with_default_correctness  →  声明 shapes  →  freeze_from_task
    →  契约 materialize 出输入  →  候选源码（外层写的）
    →  Verifier.verify(L1/L2/L3)  →  failure_route.classify(处方)  →  report.write_report

四个候选/契约回答四个不同的问题：

    A  错的核（数值小 10%）      → L2 拦住，处方 `debug_fix`（**不是重写**），带误差形态
    B  对的核（契约没要求真机）  → L1/L2 过；L3 是**本机 CPU 上真测的 numpy 读数**
                                  （`mode=actual`，后端 emulated）⇒ 档位 valid、五位全开
    B' 同一个核（契约要求真机）  → 同一条读数，**判定变成 uninformative + 拒评**：
                                  五位全关，理由指名「real_device 执行侧没有」
    C  Triton 源码（只到 L1）    → 本机没有可执行设备 ⇒ 不谎报 L2/L3，停在 L1

B 与 B' 是同一个核、同一份输入、同一次计时，**只换了契约里的一条声明** —— 这就是
「资格决定这个数被允许造成什么后果」，不需要任何开关。

边界（写清，避免读大）：
* 候选是**手写的**，不是 LLM 写的 —— 本演示验的是「链能不能串」，不是「模型写得好不好」。
  外层写核质量必须真机实测（§9.5 第 2 条）。
* 演示跑在 `MockGPU` 上，所以把任务的 `hardware`/`dsl` 钉成 `mock_gpu`/`python`；真实场景
  是 `triton` + H20。**判定链与 DSL 无关**，换 DSL 只换候选源码与执行后端。
* B 的 `mode=actual` 不等于「真卡实测」：`MockGPU` 声明的是 `EMULATED`，它真测的是控制机
  的 numpy。机器读数是真的，**它不代表加速卡的性能** —— 这正是 B' 要说明的事。
* 产物落在临时目录，不写 `output/` 与 `logs/` —— 与 `tests/conftest.py` 同一个理由。
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 先隔离落点，再 import —— 与 README「跑测试不要污染仓库」同一条纪律。
#
# ⚠️ 用 `setdefault` 而不是赋值：`tests/conftest.py` 有一套 session 级断言钉住
# 「这四个变量必须还是它设的那份」—— 演示若在这里覆盖，单测会在 setup 阶段就红
# （`_assert_side_effects_are_redirectable`），而那正是它该红的地方。
# 演示自己的产物（报告 + 四份过程日志）仍然落在 RUN_ROOT。
RUN_ROOT = Path(tempfile.mkdtemp(prefix="oa-hitl-chain-"))
for _key, _value in (("OPERATOR_AGENT_OUTPUT_DIR", str(RUN_ROOT / "output")),
                     ("OPERATOR_AGENT_LOG_DIR", str(RUN_ROOT / "log")),
                     ("OPERATOR_AGENT_KERNEL_DB", "off"),
                     ("OPERATOR_AGENT_FAILURE_DB", "off")):
    os.environ.setdefault(_key, _value)

from operator_agent.contract import freeze_from_task  # noqa: E402
from operator_agent.failure_route import classify  # noqa: E402
from operator_agent.intent_parser import IntentParser  # noqa: E402
from operator_agent.mock_gpu import MockGPU  # noqa: E402
from operator_agent.report import write_report  # noqa: E402
from operator_agent.schemas import KernelArtifact, VerificationLevel  # noqa: E402
from operator_agent.verifier import Verifier  # noqa: E402


NL_REQUEST = "把 64x64x64 的 fp32 矩阵乘法写成算子"

# 外层（人或模型）写的候选。第一个故意错 10%。
WRONG = """def kernel(A, B):
    return (A @ B) * 0.9
"""

CORRECT = """def kernel(A, B):
    return A @ B
"""

TRITON = """import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK + tl.arange(0, BLOCK)
    rn = pid_n * BLOCK + tl.arange(0, BLOCK)
    rk = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        a = tl.load(a_ptr + rm[:, None] * stride_am + (k0 + rk)[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k0 + rk)[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def kernel(A, B):
    M, K = A.shape
    K2, N = B.shape
    C = A @ B
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    _matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1),
                         B.stride(0), B.stride(1), C.stride(0), C.stride(1), 32)
    return C
"""


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def quiet(fn, *a, **kw):
    """跑一次判定，把 Verifier 的过程输出收进缓冲区（演示只印结论）。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        value = fn(*a, **kw)
    return value, buf.getvalue()


def levels(results) -> dict:
    return {r.level.value: r.passed for r in results}


def mask(metrics: dict) -> str:
    elig = (metrics.get("observation") or {}).get("eligibility") or metrics.get("eligibility") or {}
    keys = ("can_verdict", "can_tune", "can_persist", "can_report", "can_count_progress")
    return "".join("✓" if elig.get(k) else "·" for k in keys)


def make_task(require_real_device: bool):
    """NL → TaskSchema（规则分支，无 LLM）→ 外层钉住目标与形状。"""
    parsed, _ = quiet(IntentParser().parse, NL_REQUEST)
    return replace(
        parsed,
        target=replace(parsed.target, hardware="mock_gpu", dsl="python",
                       required_caps=(("real_device",) if require_real_device else ())),
        # 门槛刻意设成 0.5x：64³ 在本机 CPU 上的候选/基线比值在 0.91x~1.28x 之间跳，
        # 那是噪声。这一段验的是**资格**，不是机时 —— 与 demo_spine 场景 4 同一个理由。
        constraints=replace(parsed.constraints, shapes=((64, 64, 64),),
                            min_speedup=0.5),
    )


# ── 场景 1: 自然语言 → 冻结契约 → 契约自己造输入 ────────────────────────────
def scene_1():
    head("场景 1  NL → TaskSchema → 冻结契约 → 契约 materialize 出输入（零 LLM）")
    parsed, _ = quiet(IntentParser().parse, NL_REQUEST)   # llm=None ⇒ 规则回退
    print(f"  规则解析: kernel_type={parsed.source.kernel_type}"
          f"  precision={parsed.constraints.precision.value}"
          f"  hardware={parsed.target.hardware}  dsl={parsed.target.dsl}")
    print(f"  正确性 profile: {parsed.constraints.correctness_profile}"
          f"  suite={'有' if parsed.constraints.correctness_suite else '无'}")
    print("  （profile 来自 IntentParser 的规则分支，不经过任何 LLM）")

    task = make_task(require_real_device=False)
    print("\n  ── 外层钉住目标（本机无 GPU；判定链与 DSL 无关）──")
    print(f"     hardware={task.target.hardware}  dsl={task.target.dsl}"
          f"  shapes={task.constraints.shapes}  required_caps={task.target.required_caps}")

    contract = freeze_from_task(task)
    print(f"\n  契约哈希: {contract.hash[:32]}…")
    print(f"  cases: {[c.name for c in contract.cases]}"
          f"  required_cases={len(contract.required_cases)} 条")
    print(f"  判据进哈希: max_error={contract.max_error}"
          f"  min_speedup={contract.min_speedup}"
          f"  dtype={contract.dtype_semantics.dtype}")

    first = contract.cases[0].materialize()
    again = contract.cases[0].materialize()
    same = all((first[k] == again[k]).all() for k in first)
    print(f"  契约造输入: {sorted(first)}  {first['A'].shape} x {first['A'].dtype}")
    print(f"  同一声明两次 materialize 逐元素相同: {same}"
          f"   (契约拥有输入 ⇒ 可复算，不是「谁跑谁造」)")
    return task, contract, first


# ── 判定 + 报告一行 ────────────────────────────────────────────────────────
def scene_candidate(label, code, task, inputs, verifier, stop_after=None, note=""):
    results, log = quiet(verifier.verify, code, task, inputs, None, 0.0,
                         None, stop_after)
    route = classify(results)
    print(f"\n  ── {label} ──")
    if note:
        print(f"     {note}")
    print(f"     判定: {levels(results)}")
    for r in results:
        m = r.metrics or {}
        detail = []
        if "max_error" in m:
            detail.append(f"max_error={m['max_error']:.3g}")
        if "first_bad_index" in m:
            detail.append(f"first_bad_index={m['first_bad_index']}")
        if m.get("mode"):
            detail.append(f"mode={m['mode']}")
        if m.get("measurement_validity"):
            detail.append(f"validity={m['measurement_validity']}")
        if m.get("measurement_l3_pass_refused"):
            detail.append("l3_pass_refused=True")
        if m.get("speedup_vs_baseline") is not None:
            detail.append(f"speedup={m['speedup_vs_baseline']:.3f}x")
        if detail:
            print(f"       {r.level.value}: " + "  ".join(detail))
        if not r.passed and r.message:
            print(f"       message: {r.message.splitlines()[0][:130]}")
            for line in r.message.splitlines()[1:]:
                if "⚠️" in line or "有效" in line:
                    print(f"                {line.strip()[:130]}")
    print(f"     处方: route={route.route}  kind={route.kind}"
          f"  counts_as_iteration={route.counts_as_iteration}"
          f"  fix_not_rewrite={route.fix_not_rewrite}")
    if route.error_shape:
        print(f"     误差形态: {route.error_shape}")
    print(f"     理由: {route.reason.splitlines()[0][:150]}")
    l3 = [r for r in results if r.level is VerificationLevel.L3_PERFORMANCE]
    if l3:
        m = l3[0].metrics or {}
        print(f"     资格掩码 V T P R C: {mask(m)}"
              f"   (判对错/可调参/可入库/可报数/算进展)")
        reasons = m.get("measurement_validity_reasons") or []
        if reasons and m.get("measurement_validity") != "valid":
            print(f"     档位理由: {reasons[0][:150]}")
        # ⚠️ 本轮实测撞出来的接缝：`classify` 只按「哪一级失败 + 文本标记」定路，
        # **不读资格位**。于是一个刚被判 uninformative 的读数，处方仍会说
        # 「性能不达标 ⇒ 去搜索」，而且 counts_as_iteration=True。
        # Conductor 那边另有 can_tune 门兜着；但 HITL 形态里**外层的下一步是读处方的人/模型**，
        # 所以工具必须把 route 与资格一起给 —— 这条写进 §9.5 的设计要求。
        if m.get("measurement_validity") not in (None, "valid") and route.counts_as_iteration:
            print("     ⚠️ 接缝: 处方说「可计入一轮/去搜索」，而资格说这一读数不能驱动任何后果")
            print("        ⇒ HITL 工具**不许**只给 route，必须与资格掩码并列（见 §9.5）")
    return results, log


def main() -> int:
    print(__doc__.strip().split("\n\n")[0])
    print(f"  产物流向: {RUN_ROOT}")

    task, contract, inputs = scene_1()

    backend = MockGPU()
    verifier = Verifier(backend)
    print(f"\n  后端: {backend.name}  capabilities="
          f"{sorted(c.value for c in backend.capabilities)}")

    head("场景 2  错的核：L2 拦住，处方是 debug_fix（不是重写）")
    wrong_results, wrong_log = scene_candidate(
        "候选 A（外层写错：结果整体小 10%）", WRONG, task, inputs, verifier,
        stop_after=VerificationLevel.L2_FUNCTION)

    head("场景 3  对的核：L1/L2 过；L3 是本机 CPU 上真测的 numpy 读数（后端 emulated）")
    ok_results, ok_log = scene_candidate(
        "候选 B（正确，契约**没有**要求真机）", CORRECT, task, inputs, verifier)

    head("场景 4  同一个核、同一条读数，契约要求真机 ⇒ 拒评（五位全关）")
    print("  ⚠️ 门槛被设成 0.5x：64³ 在本机 CPU 上的比值在 0.91x~1.28x 之间跳，是噪声。")
    print("     这一步验的是**资格**（哪一条判据把数关掉），不是机时。")
    strict_task = make_task(require_real_device=True)
    strict_contract = freeze_from_task(strict_task)
    print(f"  契约哈希变了: {strict_contract.hash[:32]}…"
          f"  required_caps={strict_contract.required_caps}")
    print("  ⚠️ 这一步**没有重跑核**——判定读的是同一条读数，换的是判据。")
    refused_results, refused_log = scene_candidate(
        "候选 B'（同一个核，契约声明 required_caps=('real_device',)）",
        CORRECT, strict_task, inputs, verifier,
        note="（同一份源码、同一份输入；只有契约多了 required_caps）")

    head("场景 5  Triton 源码：只到 L1（本机无执行设备，不谎报 L2/L3）")
    triton_results, triton_log = scene_candidate(
        "候选 C（Triton 源，stop_after=L1）", TRITON, task, inputs, verifier,
        stop_after=VerificationLevel.L1_COMPILE,
        note="（外层可以写 Triton；判定侧只做它能诚实做的那些级）")

    # ── 场景 6: 处方 + 报告（外层把一次回合收成产物）──────────────────────
    head("场景 6  一次回合的产物：报告 +「我无法确立的事」")
    l3_metrics = next((r.metrics for r in refused_results
                       if r.level is VerificationLevel.L3_PERFORMANCE), {})
    artifact = KernelArtifact(
        kernel_id="hitl-demo",
        task_id=getattr(task, "task_id", "") or "hitl-demo-task",
        code=CORRECT,
        language="python",
        verification_results=list(refused_results),
        benchmark_data=dict(l3_metrics),
        iteration_history=[
            {"round": 1, "action": "generate", "levels_passed": levels(wrong_results),
             "speedup": None, "gflops": None, "note": "候选 A 数值错 10%，止步 L2"},
            {"round": 2, "action": "generate", "levels_passed": levels(ok_results),
             "speedup": next((r.metrics.get("speedup_vs_baseline") for r in ok_results
                              if r.level is VerificationLevel.L3_PERFORMANCE), None),
             "gflops": None, "note": "候选 B 正确；L3 是 emulated 后端上的读数"},
            {"round": 3, "action": "generate", "levels_passed": levels(refused_results),
             "speedup": l3_metrics.get("speedup_vs_baseline"),
             "gflops": l3_metrics.get("gflops"),
             "note": "候选 B'（同核，契约要求真机）⇒ 拒评"},
        ],
    )
    report_dir = RUN_ROOT / "run"
    # 落点由调用方负责：`write_report` 只写文件，不造目录（Conductor 那边是 run_layout 的活）。
    report_dir.mkdir(parents=True, exist_ok=True)
    path = write_report(report_dir, artifact=artifact, task=strict_task)
    text = path.read_text(encoding="utf-8")
    print(f"  报告: {path}  ({len(text)} B)")
    seen = set()
    for line in text.splitlines():
        if line.startswith("- ") and any(
                k in line for k in ("轮数", "claim", "档位", "kernel_id")) and line not in seen:
            seen.add(line)
            print(f"    {line}")
    section3 = text.split("## 3. 我无法确立的事")[1].split("## 4.")[0]
    body = [l for l in section3.strip().strip("`").splitlines()
            if l.strip() and l.strip() != "```"]
    print("  ── §3 我无法确立的事（截断）──")
    for line in body[:8]:
        print(f"    {line}")
    if len(body) > 8:
        print(f"    …（共 {len(body)} 行，完整见报告）")

    # 四份过程日志留档（不满足于「看起来跑过了」）
    for name, log in (("candidate-A.log", wrong_log), ("candidate-B.log", ok_log),
                      ("candidate-B-refused.log", refused_log),
                      ("candidate-C.log", triton_log)):
        (RUN_ROOT / name).write_text(log, encoding="utf-8")

    print(f"\n{'=' * 78}")
    print("  一句话：判定层不经过 Conductor 也能被串起来 ——")
    print("  链是 NL → 契约 → 候选 → L1/L2/L3 → 处方 → 报告，每一步都留产物。")
    print("  B 与 B' 是同一个核、同一条读数：只改了契约里的一格 required_caps，")
    print("  后果就从「五位全开」变成「五位全关」—— 这就是资格层的用处，且它没有开关。")
    print("  ⚠️ 本演示**没有**证明外层模型写核的质量（那要真机实测，§9.5 第 2 条），")
    print("     只证明了「这条链在工程上成立」。")
    print(f"{'=' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
