"""量一遍「逐轮堆积」到底占多少上下文 —— C2 热/冷两层的前后对照工具

在控制端跑, 不需要 GPU 也不需要 LLM:

    python3 benchmarks/measurement/measure_prompt_layers.py

为什么不打真机: 要看的是 prompt **构造**阶段的规模, 而构造全在 Python 里,
LLM 只是接收方。用真实的 SearchAgent 规则回退路径 (`_rule_optimize`) 推进 sketch
版本, 再用真实的 Coder prompt 渲染路径 (`_llm_generate_code`) 出 prompt, 每一环
都是生产代码本身 —— 打真机只会多花 GPU 和 tokens, 不会让这个曲线更准。

三件事分别打印:
  1. 每轮 prompt 字符数 (以及其中 hints 块占多少);
  2. 同一批 hint 文本在 prompt 里出现几次 —— sketch.to_dict() 与
     {{optimization_hints}} 是两个独立的注入点, 飘了会各留一份;
  3. 修完之后期望: 曲线应当**平**, 而不是随版本数线性抬。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from operator_agent.agents import CoderAgent, SearchAgent          # noqa: E402
from operator_agent.schemas import (ConstraintSpec, MemoryPlan,     # noqa: E402
                                    PipelineStage, Precision,
                                    SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema,
                                    TaskType, TilingStrategy,
                                    UnifiedSketch)

ROUNDS = 6


class CapturingLLM:
    """把送进 LLM 的 prompt 原样收下, 回一份能过预检的最小代码 / 一条新 hint"""

    def __init__(self):
        self.prompts = []
        self.n = 0

    def chat(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return ("```python\nimport torch\nimport triton\n"
                "import triton.language as tl\n\n\n"
                "@triton.jit\ndef _k(a):\n    return a\n\n\n"
                "def kernel(A, B):\n    return A @ B\n```\n")

    def chat_json(self, messages, **kw):
        """SearchAgent._llm_optimize 的返回: 每轮给一条**互不相同**的 hint。

        与规则回退路径的区别要分开看 —— 规则路径每轮生成同文案的 hint (列表里全是
        重复串), LLM 路径才是真实的「逐轮新增不同内容」。两条曲线的斜率不一样。
        """
        self.prompts.append(messages[-1]["content"])
        self.n += 1
        return {"new_tiling": {"block_m": 64, "block_n": 64 + self.n * 16, "block_k": 32},
                "new_hints": [f"第 {self.n} 轮: 把 BN 提到 {64 + self.n * 16} 以增大"
                              f"每线程复用, 同时保持 SMEM 在 48KB 以内 (来自 SearchAgent)"],
                "suggestion": f"round {self.n}"}


class EmptyDB:
    def search(self, **kw):
        return []


def task():
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton",
                          compute_units=78, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32,
                                   min_speedup=1.0, max_error=1e-2))


def initial_sketch() -> UnifiedSketch:
    return UnifiedSketch(
        algorithm="matmul_blocked",
        version=1,
        tiling=TilingStrategy(block_m=64, block_n=64, block_k=32),
        memory_plan=MemoryPlan(global_memory="A/B/C 三个 fp32 矩阵",
                               local_memory="无", register_memory="累加器"),
        pipeline=[PipelineStage(stage="load", source="global", destination="register",
                                hint="coalesced load")],
        target_dsl="triton",
        optimization_hints=["初始设计: 64x64x32 分块"],
    )


def perf_data(gflops: float) -> dict:
    return {"bound_type": "compute", "roofline_bound": "compute",
            "hw_name": "nvidia h20", "hw_compute_capability": "9.0",
            "gflops": gflops, "target_gflops": 120.0,
            "optimization_history": []}


def curve(label: str, step):
    llm = CapturingLLM()
    coder = CoderAgent(kernel_db=EmptyDB(), llm=llm)
    t = task()
    sketch = initial_sketch()
    print(f"\n----- {label} -----")
    print(f"{'版本':<6}{'冷层日志':<10}{'热层条数':<10}{'prompt 字符':<14}"
          f"{'热块字符':<12}{'每条热 hint 出现次数':<22}")
    rows = []
    counts_seen = []
    for _ in range(ROUNDS):
        llm.prompts.clear()
        coder._llm_generate_code(sketch, t)
        prompt = llm.prompts[0]
        hot = sketch.hot_hints()
        block = "\n".join(f"- {h}" for h in hot)
        counts = sorted({prompt.count(h) for h in hot})
        counts_seen.append(counts)
        rows.append((sketch.version, len(sketch.optimization_hints), len(prompt),
                     len(block)))
        print(f"v{sketch.version:<5}{len(sketch.optimization_hints):<10}"
              f"{len(hot):<10}{len(prompt):<14}{len(block):<12}{str(counts):<22}")
        sketch = step(llm, sketch)
    _, _, p0, _ = rows[0]
    _, cold, pn, nb = rows[-1]
    print(f"  → {ROUNDS} 轮: prompt {p0} → {pn} 字符 ({(pn - p0) / p0:+.1%}), "
          f"冷层累计 {cold} 条 (不进 prompt), 热块 {nb} 字符")

    # 两条不变量: 只允许一个 hint 通道; 规模不随版本继续爬 (后半段每轮增幅 < 30 字符,
    # 那个额度留给 tiling 数字本身的变化)
    bad = [r for r in counts_seen if r != [1]]
    assert not bad, f"这些轮里热 hint 在 prompt 中出现次数不是 1: {bad}"
    tail_growth = (rows[-1][2] - rows[len(rows) // 2][2]) / (len(rows) - len(rows) // 2)
    assert tail_growth < 30, f"后半段每轮仍涨 {tail_growth:.0f} 字符 —— 冷层又漏进 prompt 了"
    return rows


def main():
    def llm_step(llm, sketch):
        searcher = SearchAgent(llm=llm)
        new, _ = searcher._llm_optimize(sketch, perf_data(30.0))
        return new

    def rule_step(llm, sketch):
        searcher = SearchAgent(llm=llm)
        new, _ = searcher._rule_optimize(sketch, perf_data(30.0))
        return new

    curve("SearchAgent LLM 路径 (每轮不同的新 hint)", llm_step)
    curve("SearchAgent 规则回退路径 (每轮同文案 hint)", rule_step)


if __name__ == "__main__":
    main()
