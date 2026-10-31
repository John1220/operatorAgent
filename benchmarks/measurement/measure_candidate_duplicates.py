"""真机基率: 让模型连续优化三轮, 有多少轮的候选在结构上是**同一份东西**

这条测量是「重复轮要不要跳过 L3」这个决定的**前置** (docs/research/45/46 同一套纪律:
先量, 再判)。跳过验证省的是真机时间, 代价是**假阳时丢掉一个真的更好的核** ——
所以现在只统计, 不跳过。

跑法 (需要 GPU 机的 ssh 别名 + config.yaml 里的 LLM 端点):
    python3 benchmarks/measurement/measure_candidate_duplicates.py

⚠️ 关于「基率」这件事的解读限制: 每轮的候选是在**同一个 prompt 家族**里生成的
(配方 + 上一轮的 L3 指标), 所以「结构相同」既可能来自模型原地打转, 也可能来自
**它确实认为没有第二种写法**。这两者对这个决定的含义相反 (前者该拦, 后者拦了也没用),
本驱动分不开, 只能把签名与配方名单一起带回来给人看。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operator_agent.app._harness_controls import enable_for_harness
enable_for_harness()

# 语料与日志都不该被一次测量写进仓库
os.environ.setdefault("OPERATOR_AGENT_KERNEL_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_FAILURE_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_LOG_DIR", "off")

import numpy as np  # noqa: E402

from operator_agent.app.conductor import Conductor  # noqa: E402
from operator_agent.execute.llm_client import LLMClient  # noqa: E402
from operator_agent.execute.remote_executor import RemoteExecutor  # noqa: E402
from operator_agent.core.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType,
)


def llm_from_config() -> LLMClient:
    """从 config.yaml 读出自建端点。值只进环境变量, **不打印**。"""
    import yaml
    cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        llm = (cfg.get("llm") or {})
        os.environ.setdefault("LLM_BASE_URL", str(llm.get("base_url") or ""))
        os.environ.setdefault("LLM_MODEL", str(llm.get("model") or ""))
        os.environ.setdefault("LLM_API_KEY", str(llm.get("api_key") or ""))
    client = LLMClient(temperature=0.3, max_tokens=4096)
    # 判别实验的输入必须是**同一个模型**: 混进付费降级模型的话, 重复率量的就是两个模型
    if getattr(client, "fallback_model", None):
        raise SystemExit("❌ 本次测量不允许走降级模型 (会污染基率的口径)")
    return client


# min_speedup 定成一个**达不到**的值: Conductor 一通过 L3 就 break, 而默认门槛
# (1.0x vs cuBLAS) 第一轮就能过 —— 上一版跑完 4 个用例只拿到 1 轮 optimize, 因为
# 另外 3 个都在第 1 轮通过后停了。没有第 2、3 轮就没有「重复」可测。
# ⚠️ 这是一个**已知混淆**, 不是中性的实验设置: 它让 prompt 一直带着「还差 10 倍」的
# 压力, 而真实生产里第 2 轮只发生在第 1 轮**失败**的任务上 (编译错 / 数值超差 / 加速比
# 不够)。所以这里量到的重复率是「失败任务上的重复率」的下界口径, 不能当成全局基率;
# 它回答的问题是「当循环真的转起来时, 模型有多大概率重新端出同一份结构」。
MIN_SPEEDUP = 10.0


def _task(kernel_type: str, precision: Precision) -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
        constraints=ConstraintSpec(precision=precision, min_speedup=MIN_SPEEDUP,
                                   max_error=1e-2),
    )


CASES = [
    ("matmul", Precision.FP16, (1024, 1024, 1024)),
    ("layernorm", Precision.FP32, (4096, 1024)),
]
# 基率的分母是「轮」, 不是「用例」。两个用例 × 一轮 = 最多 6 轮, 拿去支撑一个
# 「要不要跳过真机验证」的决定太薄; 重复同一用例是**同一个分布**的第二次抽样,
# 不引入新的输入契约风险, 就能把分母翻倍。
REPEATS = 3


def inputs_for(kernel_type: str, shape):
    if kernel_type == "matmul":
        m, n, k = shape
        return {"A": np.random.randn(m, k).astype(np.float32),
                "B": np.random.randn(k, n).astype(np.float32)}
    m, n = shape
    return {"X": np.random.randn(m, n).astype(np.float32),
            "weight": np.ones(n, dtype=np.float32),
            "bias": np.zeros(n, dtype=np.float32)}


def main() -> int:
    llm = llm_from_config()
    backend = RemoteExecutor(mode="ssh", timeout=600)

    total_rounds = 0
    total_dups = 0
    per_case = []
    crashed = []
    for (kernel_type, precision, shape), rep in (
            (c, i) for i in range(REPEATS) for c in CASES):
        label = f"{kernel_type}#{rep + 1}"
        print(f"\n{'='*60}\n[{label} / {precision.value}] {shape} — 真跑三轮优化\n{'='*60}")
        conductor = Conductor(backend=backend, llm=llm, max_optimize_rounds=3)
        # 一次 LLM 502 / ssh 抖动会让 conductor 提前收尾。整轮测量不能因为
        # 「跑第一个用例时断了」就把已经拿到的数字全丢掉 —— 分用例兜住, 最后一起判。
        try:
            art = conductor.execute(_task(kernel_type, precision),
                                    inputs_for(kernel_type, shape), target_gflops=0.0)
        except Exception as exc:  # noqa: BLE001
            crashed.append((label, f"{type(exc).__name__}: {str(exc)[:120]}"))
            print(f"\n[{label}] ❌ 用例中断: {type(exc).__name__} (不计入基率)")
            continue
        hist = art.iteration_history
        sigs = [(h.get("round"), h.get("action"), h.get("candidate_signature") or "",
                 h.get("duplicate_of"), list(h.get("recipes_applied") or []))
                for h in hist]
        # 只看 optimize 轮: fix 轮本来就该长得像上一轮 (修的是同一个缺陷),
        # 把它们算进来就是拿「模型在原地打转」去量「模型在打转」
        opt = [s for s in sigs if s[1] == "optimize"]
        dups = sum(1 for s in opt if s[3] is not None)
        total_rounds += len(opt)
        total_dups += dups
        per_case.append((label, sigs))
        others = sorted({x[1] for x in sigs if x[1] != "optimize"})
        print(f"\n[{label}] {len(opt)} 轮 optimize, {len({s[2] for s in opt if s[2]})} 个不同结构, "
              f"{dups} 轮重复 (非 optimize 轮 {len(sigs) - len(opt)} 轮: {others or '无'})")
        for r, act, sg, d, recipes in sigs:
            print(f"  轮 {r} [{act}]: {sg[:12] or '(无签名)'} "
                  f"{'重复于轮 ' + str(d) if d is not None else '新结构'}  配方={recipes}")

    if crashed:
        print(f"\n⚠️ 中断的用例: {crashed}")
    n_runs = len(per_case)
    multi = sum(1 for _, sg in per_case if len([x for x in sg if x[1] == "optimize"]) > 0)
    print(f"\n=== 基率 ===")
    print(f"  {n_runs} 次运行里有 {multi} 次跑进了第 2 轮 (循环转不起来就没有重复可测)")
    if total_rounds >= 4:
        print(f"  {total_rounds} 轮 optimize 候选里 {total_dups} 轮结构重复 "
              f"= {total_dups / total_rounds * 100:.1f}%")
    else:
        # 分母只有 1~3 时打印一个百分数, 读起来像测量值而其实是单次抽样
        print(f"  ❌ 分母只有 {total_rounds} 轮, 不出百分数 (0/1 和 0/12 都写成 0.0%)")
        print("  → 循环没转起来: 检查是不是 L1/L2 一轮就过、或 LLM/ssh 中断")
        return 1
    print("""
判读 (doc 47 的改判条件):
  · 基率 ≈ 0        → 「重复轮烧掉一次 L3」这个问题在本项目不存在, 该把 novelty 模块退掉
                      或只留描述符当诊断;
  · 基率可观 (>15%) → 才有资格谈「重复轮跳过 L3」, 且要先测假阳率: 被人跳过的历史轮次里
                      有多少轮其实换了配置 (对比 descriptor 与 structural 的粒度差);
  · 两者之间        → 先只做一件事: 把重复这件事告诉模型 (在 prompt 里点名「第 N 轮与第 M 轮
                      结构相同」), 因为它零成本、且不替模型决定什么是无用的。
""")
    for kernel_type, sigs in per_case:
        print(f"\n[{kernel_type}] 结构相同但配方不同的轮次 (描述符能区分, 结构键不能):")
        by_struct = {}
        for r, act, sg, d, recipes in sigs:
            if sg:
                by_struct.setdefault(sg, []).append((r, recipes))
        for sg, rows in by_struct.items():
            if len(rows) > 1 and len({tuple(x[1]) for x in rows}) > 1:
                print(f"  签名 {sg[:12]}: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
