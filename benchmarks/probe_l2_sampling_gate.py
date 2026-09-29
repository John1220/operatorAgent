"""L2 数值门的**充分性**探针: 一个全零候选能不能被判通过? (doc 57, 外部一手 2609.22220)

外部一手 (arXiv:2609.22220v1《Measuring the Checker: Mutation Analysis for GPU-Kernel
Benchmark Oracles》, 行号取自本文抽出的那份带行号文本):
  行 197   "A kernel that returns all zeros passes every official trial. The blind band is
           not an edge case; it widens systematically with reduction size"
  行 193-196  官方 `torch.rand` 输入下参考输出每元素均 2.5e-6, 而检查接受 `atol=1e-2` 内的
           任何输出 —— 「四千倍于信号」
  行 249-251  只有带 **kill witness** (一个具体、经有效性门验证的输入, 在其上它确实失败) 的
           变异才进分母; 不可杀的行走隔离, 因为「拿不可杀的行去评分协议, 会把每个协议
           一样地压低, 而对任何协议都不提供任何信息」

我方 L2 有**两条**比对路径, 而 `tolerance.py` 的 docstring 自称「判定式只有一份」:
  · `_verify_single_set` 里 `Cap.FULL_OUTPUT_TENSOR in caps` 那路 (本探针用 MockGPU 走它)
    走 `effective_atol` = 容差表与任务 max_error **取 min**, 再加相对项的 AND 形式;
  · 没有那个 Cap 的那路 —— `LocalExecutor` 声明的是**空集** —— 判的是
    `max_error <= task.constraints.max_error`: 纯绝对、不查表、没有相对项。

⚠️ 这里**不写 `verifier.py:NNN`**: 上一版有八处这样的行号 (分布在七行: 两条路各一处、判别键说明
三处、两条返回值各一处、臂 0b 标题一处), 而 `verifier.py` 是并发会话会改的文件 —— 引入本探针的
那个提交上 `:509` 确实落在 `effective_atol` 那一段里, 今天同一句已经在 572 行 (+64), 而 `:509`
落到两条路分叉**之前**的公共段。指错的坐标比没有坐标更糟 —— 读的人会把「采样那条」当成 AND 那条
去改。锚点用符号名 (`_verify_single_set` / `Cap.FULL_OUTPUT_TENSOR`), 它们改名字时会响,
挪位置时不会 (doc 58 §4.2 / CHECKLOG I30)。

姿势 (先标定再读数):
  臂 0  两条已知答案的对照 (正确核必须 PASS、0.97 缩放必须 FAIL) + 前提守卫
        (两个在册后端确实各走一条路; 前提哪天消失就退出, 不印一张看不出自己失效理由的表)
  臂 0b 默认值审计: 通过那一侧的聚合 metrics 里, 哪些键是**算出来的**、哪些是
        `m.get(k, 默认)` 造出来的 (行 380/382/384)
  臂 1  全零候选在两路各自读数
  臂 2  参考幅值扫描: 两路开始「看不见」的临界, 连「带宽/信号」比一起打
  臂 3  反向对照: 收紧 max_error 之后同一个候选必须被拦 —— 否则「通过」不是幅值造成的
  臂 4  常数偏移候选: 相对项在不在, 差别就在这一臂
  臂 5  高精度参考 (float64 同一份参考) 那把尺与本级判决对照: 印出「L2 判通过而那把尺反对」的处数
        (读数规则来自 KernelOPT `2609.30059v1` 行 372-389, 我方只观测不判定, 阈未标定 ⇒ #82)
  臂 6  坏的是**参考**而不是候选: 同一份候选在「参考干净 / 参考含 NaN / 含 Inf / 全 NaN」四种
        参考下的 compared / reference_nonfinite / max_error / coverage 读数
        (起因是 KernelZero `2609.33074v1` 行 143-148 那句任务侧的数值稳定预检 ⇒ #94)

运行: python3 -B benchmarks/probe_l2_sampling_gate.py
"""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from operator_agent.mock_gpu import MockGPU
from operator_agent.remote_executor import LocalExecutor
from operator_agent.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType,
)
from operator_agent.verifier import Verifier

CORRECT = "import numpy as np\n\ndef kernel(A=None, B=None, **kw):\n    return A @ B\n"
SCALED = "import numpy as np\n\ndef kernel(A=None, B=None, **kw):\n    return 0.97 * (A @ B)\n"
ZEROS = "import numpy as np\n\ndef kernel(A=None, B=None, **kw):\n    return np.zeros_like(A @ B)\n"
SHIFT = "import numpy as np\n\ndef kernel(A=None, B=None, **kw):\n    return (A @ B) + 3e-3\n"

K = 64
# 只有 AND 那条路会算这几个键; 纯绝对那条一个都不算 (它只报 max_error/coverage)。
AND_ONLY_KEYS = ("matched_ratio", "bad_elements", "atol", "rtol", "required_matched_ratio")


def make_task(max_error: float, precision: Precision = Precision.FP32) -> TaskSchema:
    return TaskSchema(
        task_id="probe-l2-adequacy",
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="probe", dsl="python"),
        constraints=ConstraintSpec(precision=precision, max_error=max_error),
    )


def make_inputs(scale: float) -> dict:
    rng = np.random.default_rng(7)
    a = (rng.standard_normal((K, K)) * scale).astype(np.float32)
    b = (rng.standard_normal((K, K)) * scale).astype(np.float32)
    return {"A": a, "B": b}


def branch_of(res) -> str:
    """走的是哪条比对路径。

    ⚠️ 判别键**不能用 `coverage`**: 两条路都上报它 (采样那条与 AND 那条各写各的, 后者在参考
    干净时恒 1.0), 而通过那一侧的聚合还会用 `m.get("coverage", 1.0)` 再兜一次。第一版就是拿它
    当标记, 于是表里一半的行被印成错的那条路。`atol` 只有 AND 那条会算 (它才是查了容差表的那条),
    所以用它。
    """
    if "atol" in res.metrics:
        return "AND 形式"
    if "max_error" in res.metrics:
        return "纯绝对"
    return "❓认不出哪条路"


def fmt(res, key):
    val = res.metrics.get(key)
    if val is None:
        return "— 没这个键"
    if isinstance(val, float):
        return f"{val:.2e}"
    return str(val)


def verdict(backend, code, scale, max_error):
    v = Verifier(backend=backend)
    t = make_task(max_error)
    inp = make_inputs(scale)
    r = v._verify_function(code, inp, None, t)
    ref = v._generate_reference(t, inp)
    return r, float(np.max(np.abs(ref)))


def main() -> int:
    backends = [("LocalExecutor (空 Cap 集)", LocalExecutor()),
                ("MockGPU (声明 FULL_OUTPUT_TENSOR)", MockGPU())]

    print("== 臂 0 · 已知答案的对照 ==")
    branches, bad = [], 0
    for label, be in backends:
        r, _ = verdict(be, CORRECT, 1.0, 5e-1)
        r2, _ = verdict(be, SCALED, 1.0, 5e-1)
        branches.append(branch_of(r))
        ok = r.passed and not r2.passed
        bad += 0 if ok else 1
        print(f"  {label}: 走 {branch_of(r)} | 正确核={'PASS' if r.passed else '❌没通过'} / "
              f"0.97 缩放={'FAIL(拦住)' if not r2.passed else '❌放行'}")
    if bad:
        print("❌ 对照臂不符合预期 ⇒ 后面的读数没有意义, 先修探针")
        return 2
    if len(set(branches)) < 2:
        print(f"❌ 前提变了: 两个在册后端走的是同一条路 ({branches[0]}) ⇒ 本探针的分叉读数过期")
        return 2

    print("\n== 臂 0b · 默认值审计 (`_verify_function` 通过侧聚合里那句 `m.get(..., 默认)`) ==")
    for (label, be), br in zip(backends, branches):
        r, _ = verdict(be, CORRECT, 1.0, 5e-1)
        invented = [k for k in AND_ONLY_KEYS if k in r.metrics and br.startswith("纯绝对")]
        print(f"  {label}: atol={fmt(r, 'atol')} matched_ratio={fmt(r, 'matched_ratio')} "
              f"max_rel={fmt(r, 'max_rel_error')} bad={fmt(r, 'bad_elements')}")
        if invented:
            print(f"    ❗这条路自己**不算** matched_ratio/bad_elements/rtol, 但聚合结果里出现了它们 "
                  f"({', '.join(invented)})")
            print("      ⇒ 来源是 `m.get('matched_ratio', 1.0)` / `m.get('max_rel_error', 0.0)` / "
                  "`m.get('bad_elements', 0)` —— 缺证据被写成了「全部元素匹配、零坏点、无相对误差」。"
                  "从 `result.json` 读不出这两条路的区别")

    print("\n== 臂 1 · 全零候选 (max_error=5e-1, 即三个 driver 历史上一直传的那个值) ==")
    for label, be in backends:
        r, ref_max = verdict(be, ZEROS, 1.0, 5e-1)
        print(f"  {label}: 走 {branch_of(r)} | passed={r.passed}  ref_max={ref_max:.2e}  "
              f"max_error={fmt(r, 'max_error')}  coverage={fmt(r, 'coverage')}  "
              f"matched={fmt(r, 'matched_ratio')}")

    print("\n== 臂 2 · 参考幅值扫描: 「带宽 / 信号」比与全零候选的读数 ==")
    print("  scale  ref_max   band/signal   LocalExecutor(纯绝对)      MockGPU(AND)")
    for scale in (1.0, 1e-1, 1e-2, 1e-3, 1e-4):
        cells, ref_max = [], None
        for _, be in backends:
            r, ref_max = verdict(be, ZEROS, scale, 5e-1)
            cells.append(("PASS ❌" if r.passed else "FAIL ✔") + f" [{branch_of(r)}]")
        print(f"  {scale:<6g} {ref_max:<9.2e} {5e-1 / max(ref_max, 1e-30):<13.3g} "
              f"{cells[0]:<24} {cells[1]}")

    print("\n== 臂 3 · 反向对照 (max_error=1e-5, 同一份 scale=1e-3 的输入) ==")
    for label, be in backends:
        r, ref_max = verdict(be, ZEROS, 1e-3, 1e-5)
        print(f"  {label}: passed={r.passed}  ref_max={ref_max:.2e}  "
              f"max_error={fmt(r, 'max_error')}  atol={fmt(r, 'atol')}")

    print("\n== 臂 4 · 常数偏移 3e-3 的候选 (max_error=5e-1, 大幅值参考) ==")
    for label, be in backends:
        r, ref_max = verdict(be, SHIFT, 1.0, 5e-1)
        print(f"  {label}: 走 {branch_of(r)} | passed={r.passed}  "
              f"max_error={fmt(r, 'max_error')}  max_rel={fmt(r, 'max_rel_error')}  "
              f"matched={fmt(r, 'matched_ratio')}  atol={fmt(r, 'atol')}")

    # 臂 5: 高精度参考那把尺 (KernelOPT `2609.30059v1` 行 372-389) 在这同一批读数上看见什么。
    # 它**不参与判定** (阈是别人语料上的数), 所以这一臂印的是两栏: L2 判了什么 / 那把尺反不反对。
    # 只有两栏不一致的行才是信息 —— 一致的行说明这把尺在重复 AND 已经做的事。
    print("\n== 臂 5 · 高精度参考 (float64 同一份参考) 那把尺 vs 本级判决 ==")
    # ⚠️ 这一臂**不用** 上面几臂的 `PASS ❌ / FAIL ✔` 记号: 那套记号的含义是「这个候选是错的」,
    # 而这里 correct 那一行判通过是**对的**, 跟着印 ❌ 会把读者指向反的结论。
    print("  scale  候选          AND 判    那把尺     量级")
    disagreements = []
    for scale in (1.0, 1e-2, 1e-3, 1e-4, 1e-6):
        for cname, code in (("correct", CORRECT), ("all-zeros", ZEROS), ("scaled .97", SCALED)):
            r_local = verdict(LocalExecutor(), code, scale, 5e-1)[0]
            r = verdict(MockGPU(), code, scale, 5e-1)[0]
            m = r.metrics
            state = m.get("ref_oracle")
            called = "放行" if r.passed else "拦住"
            if state not in ("measured", "partially_measured"):
                print(f"  {scale:<6g} {cname:<11} {called:8s} 没测 ({state})")
                continue
            objects = bool(m.get("oracle_objects"))
            rho = m.get("oracle_rho_max")
            scale_rel = m.get("oracle_scale_relative_max")
            mag = (f"ρ={rho:.3g}" if isinstance(rho, (int, float))
                   else (f"幅值的 {scale_rel:.3g} 倍" if isinstance(scale_rel, (int, float)) else "—"))
            print(f"  {scale:<6g} {cname:<11} {called:8s} "
                  f"{'反对' if objects else '不反对':6s} {mag:16s} "
                  f"(纯绝对那条: {'放行' if r_local.passed else '拦住'})")
            if r.passed and objects:
                disagreements.append((scale, cname))
    print(f"\n  ⇒ L2 判通过而那把尺反对的读数: {len(disagreements)} 处 {disagreements}")
    print("     这些正是臂 1/2 拍到的盲区 (AND 那条的 atol 是绝对条带); 那把尺的分母是参考自身")
    print("     到 float64 真值的距离, 因此条带随幅值缩放。定阈要我方自己的标定 ⇒ #82。")

    # 臂 6: 坏的是**参考**而不是候选 (arXiv:2609.33074v1 行 143-148 —— KernelZero 在**任务侧**
    # 先验数值稳定性, "Any sample that produces invalid values such as NaN or Inf is filtered
    # out"；我方原来只在候选侧查)。这一臂比的是「同一个候选, 参考坏 1/64 vs 参考干净」,
    # 要看的是**分母与 max_error 会不会撒谎**, 而不是候选对不对。
    print("\n== 臂 6 · 参考自己产 NaN/Inf: 分母与误差读数 (#94) ==")
    v6 = Verifier(backend=MockGPU())
    t6 = make_task(5e-1)
    in6 = make_inputs(1.0)
    truth6 = np.asarray(v6._generate_reference(t6, in6))
    row0_6 = np.indices(truth6.shape)[0] == 0
    refs6 = [("干净", truth6),
             ("含 NaN", np.where(row0_6, np.float32(np.nan), truth6)),
             ("含 Inf", np.where(row0_6, np.float32(np.inf), truth6)),
             ("全 NaN", np.full_like(truth6, np.nan))]
    print(f"  参考      候选        判    compared  ref_nonfinite  max_error  matched   coverage   走的路")
    for rname, ref in refs6:
        for cname, code in (("正确核", CORRECT), ("全零核", ZEROS)):
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                r = v6._verify_single_set(code, in6, ref, t6)
            m = r.metrics
            print(f"  {rname:<7} {cname:<9} {'过' if r.passed else '不过':4s}  "
                  f"{fmt(r, 'compared_elements'):>9}  {fmt(r, 'reference_nonfinite'):>12}  "
                  f"{fmt(r, 'max_error'):>9}  {fmt(r, 'matched_ratio'):>8}  "
                  f"{fmt(r, 'coverage'):>9}   {branch_of(r)}")
    print("     ⇒ 要看的三件事: (1) 参考坏 64 个时 compared 应当是 4032 而不是 4096,")
    print("       max_error 应当是个数而不是 nan;  (2) 全 NaN 参考 ⇒ compared 0 且**判不过** ——")
    print("       而那句理由说的是「本级判不出、不是候选错」(归属), 不是「候选超容差」;")
    print("       (3) 正确核在坏参考下今天**仍然过**: 判决侧要等 #94 第 2 片 (失败归属路由)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
