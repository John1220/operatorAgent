"""§21 Agent Runtime 本地零依赖演示 —— 亲眼看一下「候选进编译之前」那道门

不需要 GPU、不需要 LLM、不联网:

    python3 benchmarks/demo_agent_runtime.py

六个场景回答的是**同一个问题**: 这一轮把机时花出去之前, 谁有权说「不」。

  1. 一次**真循环**的两种命运 —— 同一条主循环, 只换 Fixer 的产物:
     正常候选过门; 夹带 `import torch` 的修复候选在**零 GPU 秒**被判死,
     后端一个字符都没看到它 (被拒的那一轮仍然走 §8 路由 → Fixer, 不是静默丢弃)
  2. §21.2 那张**动作权限表** —— 逐个「路由 × 动作种类」, 允许 / 拒绝
  3. §21.3 的结构化证据上下文 —— 八段白名单 + 截断留痕 + 非标量指标进不去
  4. §21.4 的**覆盖率三态** —— 「查过并通过」与「今天没有对象」必须分得开
  5. §21.4 的依赖基线 —— 同一个违规候选重复提交也每轮被拒 (真机抓到的洞)
  6. §21.1 的**生产者侧** —— 动作不再由 harness 自己包出来, 而是**模型交上来的**:
     模型写意图 (种类/理由/瓶颈/工具), harness 写数字与路径 (`patch.path` 来自 §22.1
     的布局, `token_budget` 是真的上限); 写不出理由就退回 harness 那一份;
     「解释 / 请求观测」这三类终于有生产者 —— 答得出就答, 答不出就**说答不出**

可执行文档 (与本文一一对应): `tests/test_agent_runtime.py`(49 条) /
`tests/test_agent_runtime_wiring.py`(11 条) / `tests/test_llm_action_producer.py`(24 条);
变异表 27 行 (`agent_action_gate.py`) + 16 行 (`llm_action_producer.py`):

    python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/agent_action_gate.py

⚠️ 边界 (与状态文档同一句话): 这不是沙箱 —— 判的是**声明的**路径, 强制是 §22 的事;
源码级「幻觉 API」判不了 (进 `uncovered_rules`); token 预算那一轴只能在**事后**裁决,
省不下钱 (见 CHECKLOG I76)。
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from operator_agent import agent_runtime as ar  # noqa: E402
from operator_agent.agent_runtime import ENVELOPE_FENCE  # noqa: E402
from operator_agent.conductor import Conductor  # noqa: E402
from operator_agent.failure_route import classify  # noqa: E402
from operator_agent.kernel_db import KernelDatabase  # noqa: E402
from operator_agent.mock_gpu import MockGPU  # noqa: E402
from operator_agent.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType, VerificationLevel, VerificationResult,
)

#: L1 过、L2 挂 ⇒ 把循环推进「数值修复」那一格
WRONG_NUMBERS = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B + 1.0\n"
#: Fixer 的产物: 比父代**多一个外部依赖** ⇒ §21.4 第 4 条, 零 GPU 秒判死
FIXER_ADDS_DEP = ("import numpy as np\nimport torch\n\n"
                  "def kernel(A, B):\n    return torch.matmul(A, B)\n")
GOOD_KERNEL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
SKETCH = {"algorithm": "matmul",
          "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
          "memory_plan": {}, "pipeline": [], "optimization_hints": []}


def pad(text: str, width: int) -> str:
    """按**显示宽度**补齐 (CJK 占两格) —— 表格错位会让这张表读不成表"""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - w)


def head(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def clip(text: str, limit: int = 76) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def mark(code: str) -> str:
    """源码的短身份: 内容哈希 (与 §21.1 的 `ParentCandidate` 同一个口径)"""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:8]


# ── 场景 1 的两个小替身 (真循环, 假后端/假 LLM) ───────────────────────────
class RecordingGPU(MockGPU):
    """记下**每一份**被送进编译/执行的源码 —— 判据是内容级的, 不是次数级的

    次数会被生成侧的编译剪枝搅进来 (流水线自己会编译几个候选), 而
    「这份源码有没有到过后端」不会被搅。
    """

    def __init__(self):
        super().__init__()
        self.seen = []

    def compile(self, kernel_code: str) -> bool:
        self.seen.append(kernel_code)
        return super().compile(kernel_code)

    def execute(self, kernel_code: str, inputs):
        self.seen.append(kernel_code)
        return super().execute(kernel_code, inputs)

    def benchmark(self, kernel_code: str, inputs, warmup=3, repeat=10):
        self.seen.append(kernel_code)
        return super().benchmark(kernel_code, inputs, warmup, repeat)


class ScriptedLlm:
    """按 `caller_tag` 分派 —— 按调用次序分派会让演示随采样配置一起悄悄变形"""

    def __init__(self, generate, fix):
        self.generate, self.fix = generate, fix

    def chat(self, messages, **kw):
        tag = kw.get("caller_tag", "?")
        return self.fix if tag.startswith("Fixer") else self.generate

    def chat_json(self, messages, **kw):
        return dict(SKETCH)


def task(min_speedup=0.5):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(framework="pytorch", kernel_type="matmul", fused_ops=[],
                          complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="mock_gpu", dsl="python", compute_units=78,
                          warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=min_speedup,
                                   max_error=1e-5),
    )


def inputs():
    rng = np.random.default_rng(0)
    return {"A": rng.standard_normal((64, 64)).astype(np.float32),
            "B": rng.standard_normal((64, 64)).astype(np.float32)}


def run_loop(label, generate, fix, rounds=2):
    backend = RecordingGPU()
    c = Conductor(backend=backend, kernel_db=KernelDatabase(jsonl_path=None),
                  llm=ScriptedLlm(generate, fix), max_optimize_rounds=rounds,
                  heartbeat_interval=1000)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        art = c.execute(task(), inputs())
    print(f"\n  ── 臂 {label} ──")
    for h in art.iteration_history:
        verdict = h.get("agent_action")
        if verdict is None:
            continue
        ctx = verdict["context"]
        state = "过门" if verdict["accepted"] else "**被拒**"
        print(f"  第 {h['round']} 轮  {state}  动作={verdict['action']['kind']}"
              f"  路由={ctx['route_kind'] or '(无失败驱动)'}"
              f"  已花钱={ctx['spend_committed']}")
        for o in (verdict["precompile"] or {}).get("outcomes", []):
            if o["state"] != "ok":
                print(f"        ├ {pad(o['rule'], 22)} {o['state']}: {clip(o['detail'])}")
    # 提交上去的源码**不等于** LLM 交出来的那份: 生成侧还有一次确定性机械改写
    # (`normalize_candidate`: 钉 BLOCK_* 与 ieee)。所以这里按**内容特征**判, 而不是
    # 拿声明的那份字符串去比哈希 —— 后者会把「被机械改写过的」误报成「没到过后端」。
    distinct = []
    for code in backend.seen:
        if not any(code.strip() == d.strip() for d in distinct):
            distinct.append(code)
    print(f"  后端一共见过 {len(backend.seen)} 次提交, 去重后 {len(distinct)} 份:")
    for code in distinct:
        print(f"      · {mark(code)}  {len(code.splitlines()):>3} 行   含 'torch': "
              f"{'是' if 'torch' in code else '否'}")
    hit_torch = [c for c in distinct if "torch" in c]
    print(f"  声明的那份修复候选 {mark(fix)}: "
          f"{'到过后端' if any(fix.strip() == c.strip() for c in distinct) else '**没到过后端**'}"
          f"; 到过后端的提交里含 'torch' 的共 {len(hit_torch)} 份")
    # 演示**自带断言**: 屏幕上那句读数不能只是「打印得像」—— 声明里夹带 torch 的候选
    # 若真的到了后端, 这里当场炸, 而不是让人读着一张好看的图 (与全仓「✅ 要有守卫」同一条)。
    if [c for c in (generate, fix) if "torch" in c]:
        assert not hit_torch, ("演示的前提被破坏了: 声明里夹带 torch 的候选到了后端 "
                               f"({[mark(c) for c in hit_torch]}) —— 门的接线不在了")
    l1 = [r for r in art.verification_results if r.level == VerificationLevel.L1_COMPILE]
    if l1:
        print(f"  末条 L1: passed={l1[-1].passed} "
              f"compile_status={l1[-1].metrics.get('compile_status')}")
    print(f"  停止原因: {(art.metadata.get('stop') or {}).get('reason')}")
    print(f"  Runtime 记账: 放行 {c.agent_runtime.admitted}, 拒 {dict(c.agent_runtime.rejected)}")
    return backend, art, buf.getvalue()


def scene_1():
    head("场景 1  一次真循环的两种命运 (设计 §21.4: 零 GPU 秒的独立检查)")
    print("  同一条主循环、同一份假后端 (MockGPU)、同一个脚本化 LLM —— 只换 Fixer 的产物。")
    run_loop("A 正常修复", WRONG_NUMBERS, GOOD_KERNEL)
    print("\n  臂 A 的读法: 修复候选是**改过的**代码, 所以门放行; 它照常进 L1/L2。")
    backend, art, out = run_loop("B 修复产物夹带 `import torch`", WRONG_NUMBERS, FIXER_ADDS_DEP)
    print("\n  臂 B 的读法 (三件一起看):")
    print("   ① 那份带 torch 的源码**一次都没进后端** —— 拒绝发生在任何 GPU/SSH 之前;")
    print("   ② 被拒的一轮不是静默丢弃: 它变成一条 L1 失败结果, §8 的路由与 Fixer 照旧工作;")
    print(f"   ③ 拒绝理由逐字进了修复提示: "
          f"{'新增了外部依赖' in out and '✓ 在日志里' or '✗ 没进日志'}")
    print("  ⚠️ 边界: 拦的是**新增外部依赖**这一条; 判断本身依赖父代源码, 首轮无从判起")
    print("     (首轮那两条以父代为基线的判据是 `not_applicable`, 不是「通过」)。")


# ── 场景 2: §21.2 的动作权限表 ────────────────────────────────────────────
def _action(kind):
    """给每一种动作配一个**自身合法**的动作 (于是表里只剩路由这一条原因)"""
    patch = ar.Patch("candidate/code.py", GOOD_KERNEL,
                     "import numpy as np\n\ndef kernel(A, B):\n    return A @ B * 1.0\n")
    if kind in ar.PATCH_KINDS:
        return ar.AgentAction("r-fix", kind, "修: whole_tensor", 
                              parent_candidate="src:aaaa1111", patch=patch,
                              token_budget=4096)
    return ar.AgentAction("r-obs", kind, "请给我一次 profiler 读数", token_budget=1)


ROUTES = [
    ("编译错误", dict(route_kind="compile", input_key_sets=(("A", "B"),))),
    ("数值错误", dict(route_kind="numeric", error_shape="whole_tensor",
                  input_key_sets=(("A", "B"),))),
    ("性能不达标 (无许可证)", dict(route_kind="performance",
                          input_key_sets=(("A", "B"),))),
    ("性能不达标 (§10.3 放行)", dict(route_kind="performance", escalation_allowed=True,
                            input_key_sets=(("A", "B"),))),
    ("首轮生成 (无失败驱动)", dict(route_kind="", input_key_sets=(("A", "B"),))),
    ("ABI 已破坏 (横切规则)", dict(route_kind="compile", abi_broken=True,
                          input_key_sets=(("A", "B"),))),
]
KINDS = [(ar.KIND_REPAIR_COMPILE, "修编译"), (ar.KIND_REPAIR_CORRECTNESS, "修数值"),
         (ar.KIND_PROPOSE_TEMPLATE, "模板点"), (ar.KIND_PROPOSE_SOURCE_PATCH, "改源码"),
         (ar.KIND_EXPLAIN_FAILURE, "解释")]


def scene_2():
    head("场景 2  §21.2 的动作权限表: 路由 → 允许的动作种类")
    print("  ✓ = 允许   · = 拒绝 (§21.1 第二道校验)。判据函数: "
          "agent_runtime.permission_problems")
    print("  列名: " + " / ".join(f"{short}={kind}" for kind, short in KINDS))
    print()
    print("  " + pad("路由 \\ 动作种类", 26) + "".join(pad(short, 9)
                                                      for _, short in KINDS))
    for label, kw in ROUTES:
        ctx = ar.ActionContext(authorized_roots=("candidate",), **kw)
        row = "  " + pad(label, 26)
        for kind, _ in KINDS:
            problems = ar.permission_problems(_action(kind), ctx)
            row += pad("✓" if not problems else "·", 9)
        print(row)
    print("\n  两条横切规则不在表里, 但在同一道校验里 (下面是它们各自的原话):")
    ctx = ar.ActionContext(route_kind="numeric", error_shape="whole_tensor",
                           input_key_sets=(("A", "B"),))
    good = _action(ar.KIND_REPAIR_CORRECTNESS)
    # 横切 2: 修复类不点名父代
    nameless = ar.AgentAction("r1", ar.KIND_REPAIR_CORRECTNESS, "修: whole_tensor",
                              patch=good.patch, token_budget=4096)
    # 横切 3: 数值修复丢了误差形态
    shapeless = ar.AgentAction("r1", ar.KIND_REPAIR_CORRECTNESS, "重写一遍试试",
                               parent_candidate="src:x", patch=good.patch,
                               token_budget=4096)
    for a in (nameless, shapeless):
        for p in ar.permission_problems(a, ctx):
            print(f"      · {clip(p, 100)}")
    print("  「必须带误差形态」为什么算权限: 形态是这一次的**处方方向**, 丢掉它,")
    print("  「按形态修」与「随便改改」在产物里长得一模一样 (§8.2)。")


# ── 场景 3: §21.3 的结构化证据上下文 ──────────────────────────────────────
def scene_3():
    head("场景 3  §21.3 的结构化证据: 八段白名单之外没有第二个注入通道")
    long_err = "SyntaxError: 在第 40 行…\n" * 60
    ctx = ar.AgentRuntime(capabilities=(), authorized_roots=("candidate",)).context_for(
        contract_summary="matmul fp32 4096^3, max_error=1e-5, min_speedup=0.5",
        patch=ar.Patch("candidate/code.py", GOOD_KERNEL,
                       "import numpy as np\n\ndef kernel(A, B):\n    return A @ B + 1.0\n"),
        route=classify([VerificationResult(level=VerificationLevel.L2_FUNCTION,
                                           passed=False, message="最大误差=1.0")]),
        first_error=long_err,
        profiler_metrics={"dram_throughput_pct": 71.0, "raw_ncu_dump": {"blob": [1] * 99}},
        budget_left=None,
    )
    print("  上下文里被塞进的原始事实: 一个 1000+ 字符的错误日志、一个 dict 形态的原始")
    print("  NCU 转储、一个没设上限的预算。渲染出来的是这些:\n")
    for line in ctx.render().splitlines():
        if line.startswith("###"):
            print("     " + line)
    print(f"\n  截断留痕 (truncations): {list(ctx.truncations())} —— 模型「只看到开头」这件事")
    print("  是产物的一部分, 不是后台日志。")
    rendered = ctx.render()
    print(f"  非标量指标 `raw_ncu_dump` 进没进上下文: "
          f"{'进了 (缺陷!)' if 'raw_ncu_dump' in rendered else '**没进** (结构上就进不去)'}")
    print(f"  预算没设上限时被写成什么: {ctx.budget_left!r} —— 不许被读成「用完了」")
    print("  ⚠️ 边界 (诚实记): 这个上下文今天**没有生产调用者** —— 白名单与截断是结构")
    print("     性质, 但「每次 LLM 调用只提供这八项」在链路上尚未成立 (状态文档 §21 行)。")


# ── 场景 4: §21.4 的覆盖率三态 ────────────────────────────────────────────
def scene_4():
    head("场景 4  §21.4 的独立检查: 七条设计原文 → 六条判据, 三态各自有名有姓")
    print("  设计原文那七条, 到判据这一层是六条 (映射见 agent_runtime.DESIGN_CHECKS):")
    for design, rule in ar.DESIGN_CHECKS:
        print(f"      · {pad(design, 30)} → {rule}")
    print("\n  一份**首轮**候选 (没有父代) 跑一遍 —— 注意 `Patch(path, new_source)`:"
          " 第三格才是父代源码\n")
    action = ar.AgentAction("r1", ar.KIND_PROPOSE_SOURCE_PATCH, "首轮提议",
                            patch=ar.Patch("candidate/code.py", GOOD_KERNEL),
                            token_budget=4096)
    ctx = ar.ActionContext(authorized_roots=("candidate",), input_key_sets=(("A", "B"),),
                           language="python")
    verdict = ar.precompile_checks(action, ctx, capabilities=frozenset())
    print("  " + pad("判据", 24) + pad("结论", 16) + "依据")
    for o in verdict.outcomes:
        print("  " + pad(o.rule, 24) + pad(o.state, 16) + clip(o.detail))
    cov = verdict.coverage
    print(f"\n  覆盖率: 真查过的 {list(cov.covered_rules)}")
    print(f"          今天没有对象的 {list(cov.uncovered_rules)}")
    print(f"  裁决: accepted={verdict.accepted} —— 只有 `violation` 才拒绝; "
          f"`unknown` / `not_applicable` 是**结论**, 不是「没问题」也不是「有问题」。")
    print("\n  同一条判据, 只换目标路径 ⇒ 违约 (这是唯一会拒绝的一档):")
    bad = ar.AgentAction("r2", ar.KIND_PROPOSE_SOURCE_PATCH, "把测量记录写进产物",
                         patch=ar.Patch("candidate/measurements/round1.json", GOOD_KERNEL),
                         token_budget=4096)
    bad_verdict = ar.precompile_checks(bad, ctx, capabilities=frozenset())
    for o in bad_verdict.violations:
        print(f"      · {pad(o.rule, 22)} {o.state}: {clip(o.detail, 90)}")
    print(f"  accepted={bad_verdict.accepted} ⇒ 这一轮不会花任何 GPU 秒 "
          f"(而且**不是**沙箱拦的: 强制是 §22 的事, 今天只有声明)。")
    print("  ⚠️ 两个诚实的「判不了」: ① `capability` 里「源码引用不存在的 API」那一半")
    print("     今天没有 API 面清单 ⇒ 恒不可测 (只算「声明了没有量具的瓶颈」那一半);")
    print("     ② 没有授权根时路径判据报 `unknown`, 不许读成「在范围内」(fail-open)。")


# ── 场景 5: §21.4 的依赖基线 (真机抓到的那个洞) ────────────────────────────
def scene_5():
    head("场景 5  §21.4 的依赖基线: 同一个违规候选**重复提交**也每轮都被拒")
    print("  为什么单独一个场景: 「新增依赖」是相对**基线**判的, 而基线选错一个对象, 门")
    print("  就被自己走过的路绕开。真机一手 (CHECKLOG I78, algorithm-h20 + 真 LLM):")
    print("    · 第一版基线 = patch 的父代 ⇒ 第 2 轮拒了, 但第 3 轮 Fixer 又交同一份,")
    print("      父代成了**上一轮那份被拒的候选** ⇒ `torch` 不再算「新增」⇒ 门放行,")
    print("      日志 `[Verifier] L1 PASSED: 编译成功` —— 那份源码真的上了远端编译,")
    print("      最后靠 L2 的反作弊静态审计拦下 (proxy 读数: torch 载荷 1 份);")
    print("    · 现在基线 = 上一次**过门**的候选 (`ActionContext.dep_baseline_source` +")
    print("      `Conductor._admitted_baseline`) ⇒ 第 2、3 轮都被拒, torch 载荷 0 份")
    print("      (真机复跑 35.3s: 后端全跑只收到 1 份载荷 —— 第一份坏码的依赖探针)。")
    backend, art, _ = run_loop("C 同一份违规候选提交三次",
                               WRONG_NUMBERS, FIXER_ADDS_DEP, rounds=3)
    refused = [h["round"] for h in art.iteration_history
               if (h.get("agent_action") or {}).get("accepted") is False]
    # 演示**自带断言**: 「每一轮都被拒」这句话必须在屏幕上当场成立 (与场景 1 同一条纪律)
    assert refused == [2, 3], f"重复提交的轮次形状变了: {refused}"
    assert not [c for c in backend.seen if "torch" in c], "违规候选又到后端了 (I78 的洞回来了)"
    print(f"\n  读法: 被拒的轮 = {refused}; 后端收到的载荷里含 'torch' 的 = "
          f"{len([c for c in backend.seen if 'torch' in c])} 份 —— "
          f"「被拒 = 零 GPU 秒」每一轮都成立, 而不是只在第一次成立。")


# ── 场景 6: §21.1 的生产者侧 (动作由模型交上来) ─────────────────────────
def envelope_text(**payload):
    """模型交上来的那一份: 一段围栏 + (要改候选时) 一份代码"""
    import json as _json
    body = _json.dumps(payload, ensure_ascii=False)
    return f"```{ENVELOPE_FENCE}\n{body}\n```\n\n```python\n{GOOD_KERNEL}\n```\n"


class FixSequenceLlm:
    """Fixer 的每一次调用按**脚本**发回复 (生成侧恒给「结果错」的码)

    按 `caller_tag` 分派, 不按调用次序: 生成侧那几个采样候选与这里要测的东西无关。
    """

    def __init__(self, replies, generate=WRONG_NUMBERS):
        self.replies, self.generate, self.calls, self.sent = list(replies), generate, 0, []

    def chat(self, messages, **kw):
        tag = kw.get("caller_tag", "?")
        self.sent.append((tag, "".join(str(m.get("content", "")) for m in messages)))
        if tag.startswith("Fixer"):
            i = min(self.calls, len(self.replies) - 1)
            self.calls += 1
            return self.replies[i]
        return self.generate

    def chat_json(self, messages, **kw):
        return dict(SKETCH)


def run_envelope_loop(label, replies, rounds=3):
    backend = RecordingGPU()
    llm = FixSequenceLlm(replies)
    c = Conductor(backend=backend, kernel_db=KernelDatabase(jsonl_path=None), llm=llm,
                  max_optimize_rounds=rounds, heartbeat_interval=1000)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        art = c.execute(task(), inputs())
    print(f"\n  ── {label} ──")
    for h in art.iteration_history:
        verdict = h.get("agent_action")
        if verdict is None:
            continue
        env = h.get("action_envelope") or {}
        state = "过门" if verdict["accepted"] else "**被拒**"
        print(f"  第 {h['round']} 轮  {state}  动作={verdict['action']['kind']}"
              f"  动作是谁写的={env.get('source')}  信封={env.get('status')}")
        print(f"        理由: {clip(verdict['action']['rationale'] or '(空)', 68)}")
        print(f"        预算: 声明 {verdict['action']['token_budget']}"
              f" (模型在信封里写的是 {env.get('declared_token_budget')})")
        if env.get("unknown_keys"):
            print(f"        信封里那些越权的键 (只留痕, 不采纳): {list(env['unknown_keys'])}")
        for o in (verdict["precompile"] or {}).get("outcomes", []):
            if o["state"] == "violation":
                print(f"        ├ 违约 {pad(o['rule'], 18)} {clip(o['detail'], 76)}")
    fix = next((h["fix_action"] for h in art.iteration_history if h.get("fix_action")), {})
    if fix.get("asked") or fix.get("ask_turns"):
        print(f"  询问: 种类={fix.get('ask_field')} 轮次={fix.get('ask_turns')} "
              f"结论={fix.get('ask_answer_kind') or fix.get('ask_outcome')}")
        if fix.get("ask_refusal"):
            print(f"        拒的理由: {clip(fix['ask_refusal'], 80)}")
        # 屏幕上要看到**给模型看的那一句**, 而不只是产物里的分类: 「答不出来」与
        # 「观测结果是空」长得一样这件事, 只有在读者看见那句话时才算说清。
        said = next((l.strip() for l in llm.sent[-1][1].splitlines()
                     if "答不出来" in l or "已记入本轮失败记录" in l), "")
        if said:
            print(f"        给模型的那句: {clip(said, 84)}")
    print(f"  停止原因: {(art.metadata.get('stop') or {}).get('reason')}")
    return backend, art, llm, buf.getvalue()


def scene_6():
    head("场景 6  §21.1 的生产者侧: 动作不再是 harness 自己包的, 是**模型交上来的**")
    print("  接线前: `AgentAction` 的 kind / rationale / bottleneck / tools 全由 harness 填,")
    print("  于是七个种类里 `request_profiler` / `request_new_shape` / `explain_failure`")
    print("  **没有生产者** (设计里写着、代码里没人发)。现在模型在回复里夹一段信封:")
    print(f"      ```{ENVELOPE_FENCE}  {{\"kind\": ..., \"rationale\": ...}}  ```")
    print("  解析是三态的: 没有那一段 = `absent` (不是错, 退回 harness 拼装); 多于一段 /")
    print("  JSON 坏 / 类型不对 = `malformed`; 只有一段且读得出 = `parsed`。")
    print("  表外的**种类**算 `parsed` —— 模型说了它想干什么, 让判据去拒并留下理由。")
    print("\n  一: 模型按协议交信封 (种类与理由都是它写的)")
    backend, art, llm, _ = run_envelope_loop("A 模型自己写动作", [envelope_text(
        kind="repair_correctness", rationale="误差形态=whole_tensor: 累加器 dtype 写错了",
        requested_tools=[], path="contract/frozen.json", token_budget=1)])
    env = next(h["action_envelope"] for h in art.iteration_history
               if h.get("action_envelope", {}).get("source") == "llm")
    act = next(h["agent_action"]["action"] for h in art.iteration_history
               if (h.get("action_envelope") or {}).get("source") == "llm")
    assert "累加器 dtype 写错了" in act["rationale"], "理由不是模型写的那份"
    assert act["token_budget"] == 4096, "模型声明的预算顶掉了 harness 的上限"
    assert act["patch"]["path"].endswith("candidate/code.py"), act["patch"]["path"]
    print("  读法: ① 上面那句理由**逐字**是模型写的 (没有 harness 补的「第 N 轮 fix」碎片);")
    print("        ② 模型在信封里写 `token_budget: 1` 与一个 `path` —— 前者只留档、后")
    print("           者进 `unknown_keys`; 真正进判据的预算与路径仍是 harness 的 "
          f"({act['token_budget']} / {act['patch']['path'].split('/')[-1]})。")

    print("\n  二: 模型写不出误差形态 ⇒ §21.2 那半条**真的会红** (零 GPU 秒)")
    backend, art, _, _ = run_envelope_loop("B 理由里没有误差形态", [envelope_text(
        kind="repair_correctness", rationale="我把索引顺序改了一下")])
    assert not [c for c in backend.seen if c.strip() == GOOD_KERNEL], "被拒的候选上了后端"
    print("  读法: 接线前理由由 harness 拼装 ⇒ 那条判据**永不失败**; 现在它是对真实输入的")
    print("        判断, 而拒绝理由会作为下一轮的错误信息回到模型眼前 (自纠的环)。")

    print("\n  三: 三个「不改候选」的种类终于有响应 (答得出就答, 答不出就说答不出)")
    ask = "```" + ENVELOPE_FENCE + '\n{"kind": "request_profiler", "rationale": "想看带宽"}\n```\n'
    backend, art, llm, out = run_envelope_loop(
        "C 先请求 profiler, 得到答复后再交补丁",
        [ask, f"```python\n{GOOD_KERNEL}\n```"])
    fix = next(h["fix_action"] for h in art.iteration_history if h.get("fix_action"))
    assert fix["ask_answer_kind"] == "refused_no_gauge", fix
    assert "这条请求**答不出来**" in llm.sent[-1][1] or "答不出来" in llm.sent[-1][1], \
        "答复没到模型眼前"
    print(f"  读法: profiler 那一档在注册表里是 `None` ⇒ 响应用的是"
          f"**{fix['ask_answer_kind']}** (「观测结果是空」与「没有量具」在产物里长得")
    print("        一样 —— I65 那条纪律); 答复走的是模板里**声明过的**那段, 不是塞进 §21.3")
    print("        的八段证据 (第二个注入通道会污染两臂对照)。")
    print("  ⚠️ 边界: 这条链只保证「模型看得到答复」; 七个种类里真正被 harness **跑起来**")
    print("     的仍然只有那些有实现物的 —— 「请求了 profiler」不会变出 profiler。")


def main():
    print(__doc__.strip().split("\n\n")[0])
    scene_1()
    scene_2()
    scene_3()
    scene_4()
    scene_5()
    scene_6()
    print(f"\n{'=' * 78}")
    print("  一句话: 上面每一次「拒」都发生在**花钱之前**, 而每一次「判不了」都被写成")
    print("  一个有名有姓的结论 —— 读者能分清「查过并通过」与「今天没有量具」。")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()