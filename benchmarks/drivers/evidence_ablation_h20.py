"""§21.3 的两臂对照: 「**筛过的证据** vs **全量错误日志**」到底差多少 (真机 + 真 LLM)

## 这个实验要回答的问题

设计 §21.3 是强制句: 每次 LLM 调用**只**提供那八项结构化证据。生产者已经落地
(`Conductor.structured_evidence` 默认开, 每次调 Fixer 前由唯一拼装点
`AgentRuntime.context_for(...).render()` 拼出来, 经 `fixer.run(structured_evidence=...)`
进 `fixer_repair.md`)。但这个开关**从没跑过两臂对照** ⇒ 状态文档里那一行写着
「收益未实测, 不许声称」。这一跑就是去把它变成实测: 同一个场景、同一个替身产物、
同一个后端, 只把 `structured_evidence` 关掉/打开, 看读数差在哪里。

## 判据 / 开关 / 结论三者的关系 (读结论之前必须先读这一段)

  * **能证的是「差在哪」**: 两臂的 prompt 逐字留档 (替身 LLM 是唯一证人) + 产物侧
    `fix_evidence` 血缘 (`injected`/`chars`/`truncations`/`profiler_facts`/`parent_diff_lines`),
    于是「证据段进了模型的眼睛」与「它占了多少字符 / token」是**可核对的数**, 不是声明。
  * **不能证的是「收益」**: N 这么小时通过率量具分辨不出东西。量程用仓内既有的那一份
    算 (`benchmarks.drivers.harness_pass_rate.min_detectable_delta`, 不另写一套统计): N=3 时
    每臂只能分辨 ≈60pp 的通过率差 —— 两臂都绿、都红、还是一绿一红, **都不构成**
    「有用/无用」的证据。所以结论句里必须带 N 与量程, 只许说方向与差值。
  * **关掉的那一臂不是「没有信息」**: `fixer_repair.md` 里 `error_details`(全量错误日志,
    含 `syntax_error_report` 的行号+插入符) 与 `history_cases`(历史修复案例) 两段**恒在**,
    证据段是**追加**在它们**之前** (`fixer_repair.md:26`)。所以这个对照的正确名字是
    「筛过的证据 **vs** 全量错误日志」, 不是「有证据 vs 没信息」—— 把它读成后者会
    凭空放大效应。
  * **两臂的干净性是被实证的, 不是被声明的**: 除了 `structured_evidence` 这一个键,
    构造参数/任务/输入/种子/替身产物逐字相同 (键差集由 `arm_diff()` 现算并打印),
    并且脚本会把两臂**第一份** Fixer prompt 逐字对比 —— 去掉证据段之后必须相同。
    两臂不止差这一段时, 这一跑的所有对照读数**作废** (脚本会这么打印)。
  * **`history_cases` 恒为「无」**是这条干净性的一部分: 失败库与长期记忆都关掉
    (`jsonl_path="off"`), 否则第一臂写进去的失败案例会出现在第二臂的 prompt 里 ——
    那样两臂就不止差一个开关了。脚本会把这一条也打成读数 (从留档的 prompt 里读)。

## 场景与结果变量

三个场景, 每个都有**确定性的第一份坏码** (由脚本给定, 不让 27B 随机决定), 三种不同的
失败阶段: 语法 / 接口契约 / 数值。判据面之外的部分 (Fixer 的修复) 走**真 LLM** ——
那正是证据段要去的地方。

结果变量: 每一轮 `all_passed` 与 `levels_passed`、轮数、远端调用次数、Fixer 被调用次数;
成本变量: prompt 字符、prompt token、总 token、墙钟。

⚠️ 三条读数限制, 不许脱离它们引用数字:
  1. **N 小**: 见上面的量程。这一跑给的是「差值」, 不是「效应」。
  2. **两臂都未通过 ≠ 开关无用**: 那可能只是任务对 27B 太难; 反之两臂都通过也**不**说明
     证据段没用 —— 关掉的那一臂手里有全量错误日志, 三个场景对它都不算难。
  3. **采样噪声是真的**: Fixer 的 temperature 是 0.1, 同一臂重跑也会不一样。所以场景之间
     **轮换两臂顺序** (同一场景的两臂背靠背跑), 免得把端点漂移读成开关效应。

用法:
    python3 benchmarks/drivers/evidence_ablation_h20.py --dry-run          # 离线自检: 装配 + 留痕, 零 token
    python3 benchmarks/drivers/evidence_ablation_h20.py                    # 真机: 3 场景 × 2 臂
    python3 benchmarks/drivers/evidence_ablation_h20.py --reps 2 --scenarios numeric
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from operator_agent._harness_controls import enable_for_harness
enable_for_harness()
# 「跑测试/探针不得改动仓库状态」的三条逃生门 (§3.5 硬伤二记过它们的账): 这一跑要的是
# **干净两臂**, 而失败库与长期记忆会跨臂串味 —— 第一臂写进去的案例会出现在第二臂的
# prompt 里, 那就不是「只差一个开关」了。显式传 `jsonl_path="off"` 是主保险, 这两条 env
# 是第二层 (与 `benchmarks/drivers/agent_action_h20.py` 同一写法)。
os.environ.setdefault("OPERATOR_AGENT_KERNEL_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_FAILURE_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_LOG_KEEP", "100")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from operator_agent.rule_templates import RuleTemplateCoder  # noqa: E402
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,  # noqa: E402
                                    TaskComplexity, TaskSchema, TaskType,
                                    TargetSpec, VerificationLevel)

#: 数据库一律关掉 —— 干净两臂的前提 (见文件头最后一条)
DB_OFF = "off"
#: 每个场景最多迭代几轮 (预算保护: 不让一次运行无上限地烧真机时间)
ROUNDS = 2
#: 标定心跳: 这一跑不量心跳, 调大以免它在轮次之间插进来 (与 tests/test_structured_evidence 同)
HEARTBEAT_INTERVAL = 1000
#: 单个场景最多跑几次 (两臂 × reps) —— 真机 token 不是免费的, 这里只用于**报账**
SKETCH = {"algorithm": "matmul",
          "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
          "memory_plan": {}, "pipeline": [], "optimization_hints": []}
EVIDENCE_HEADING = "## 结构化证据"
NEXT_HEADING = "## 历史修复案例"


# ─────────────────────────── 场景 ───────────────────────────

def _perturb(good: str, old: str, new: str) -> str:
    """在**仓内自己的保底实现**上做一处定点改动 —— 坏码必须与好码同源, 否则量到的是别的

    定点改动没命中就是**静默失效**: 场景会变成「第一轮就过」, 于是这一格压根没有 Fixer,
    两臂无从对照。所以这里对「一字未改」直接抛错, 而不是让它安静地跑下去。
    """
    if good.count(old) != 1:
        raise ValueError(f"定点改动锚点出现 {good.count(old)} 次 (要恰好 1): {old!r}")
    return good.replace(old, new)


def _scenarios(dsl: str, hardware: str, shape) -> list:
    """三个场景。坏码**派生**自 `RuleTemplateCoder` 的保底实现 (同源, 不是手写另一份)"""
    task = _task(dsl=dsl, hardware=hardware)

    def good() -> str:
        return RuleTemplateCoder().generate(task)

    good_src = good()
    if not good_src:
        raise RuntimeError(f"保底实现为空 ({dsl}/{hardware}) —— 场景无法派生")
    if dsl == "python":
        numeric = _perturb(good_src, "    return (A @ B).astype(A.dtype)",
                           "    return (A @ B).astype(A.dtype) * 1.05")
        # 语法错: 表达式写了一半 (`@` 后面没有右操作数) ⇒ `SyntaxError`
        syntax = _perturb(good_src, "    return (A @ B).astype(A.dtype)",
                          "    return (A @ B) @")
    else:
        numeric = _perturb(good_src, "    return C\n", "    return C * 1.05\n")
        syntax = _perturb(good_src, "    return C", "    return C @")
    rows = [
        # ① 语法错: 本地 L1 预检就拦下 (不花远端时间) ⇒ route=compile
        {"id": "syntax", "broken": syntax,
         "expect": "本地语法预检", "shape": shape, "good": good_src, "dsl": dsl,
         "hardware": hardware},
        # ② 接口契约不符: 远端固定以 `kernel(**输入)` 调用, 签名多一个必填参数 ⇒ 必然
        #    `TypeError`, 但那是本地 AST 就能看见的 (缺口 6) ⇒ route=compile
        {"id": "interface", "broken": _perturb(good_src, "def kernel(A, B):",
                                               "def kernel(A, B, C):"),
         "expect": "本地接口契约预检", "shape": shape, "good": good_src, "dsl": dsl,
         "hardware": hardware},
        # ③ 数值错: 编译得过、跑得动, 只是答案差 5% ⇒ L2 数值门拦下, route=numeric
        #    (这一格会真的走远端编译 + 执行, 是唯一花钱的场景)
        {"id": "numeric", "broken": numeric, "expect": "L2 数值门", "shape": shape,
         "good": good_src, "dsl": dsl, "hardware": hardware},
    ]
    # 三个坏码都必须与好码**不同源文本** —— 否则那一格第一轮就过, 静默失去对照价值
    for row in rows:
        if row["broken"] == row["good"]:
            raise RuntimeError(f"场景 {row['id']} 的坏码与好码逐字相同 (定点改动没命中)")
    return rows


#: L3 性能门的默认值. ⚠️ `0.5` 是从 `benchmarks/drivers/agent_action_h20.py` 抄来的 —— 在那个
#: 驱动里它**从未生效过** (那一跑走不到 L3)。真机实测 (2026-10-05): 本仓保底 triton 模板
#: 在 1024³ fp32 上是 **0.47x** cuBLAS(TF32) ⇒ 门槛定 0.5 时, 修好的候选也过不了 L3,
#: 结果变量会被「模板性能」占满, 与本题 (修复是否成功) 无关。所以门槛可传, 且每次读数
#: 都记下当时用的是哪一档。
DEFAULT_MIN_SPEEDUP = 0.5


def _task(dsl: str, hardware: str, min_speedup: float = DEFAULT_MIN_SPEEDUP) -> TaskSchema:
    return TaskSchema(task_id=f"evidence-ablation-{dsl}",
                      task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(kernel_type="matmul",
                                        complexity=TaskComplexity.MEDIUM),
                      target=TargetSpec(hardware=hardware, dsl=dsl),
                      constraints=ConstraintSpec(precision=Precision.FP32,
                                                 # ⚠️ 1e-2 而不是更严的 1e-4: 真机 triton 的
                                                 # `tl.dot` 默认走 TF32 (误差 ~1e-3), 容差钉到 1e-4
                                                 # 时**修好的**模板自己也会挂在数值门上 —— 那时
                                                 # 量到的是「模板 vs 容差」, 不是证据段的效应。
                                                 # 5% 的坏码照样离这个门很远 (`numeric` 场景)。
                                                 min_speedup=min_speedup,
                                                 max_error=1e-2))


def _inputs(shape) -> dict:
    rng = np.random.default_rng(11)      # 与 replay/e2e/exhaustion 同一种子
    return {"A": rng.standard_normal(shape).astype(np.float32),
            "B": rng.standard_normal(shape).astype(np.float32)}


# ─────────────────── 两臂唯一允许不同的地方 ───────────────────

def arm_kwargs(structured_evidence: bool) -> dict:
    """两臂的构造参数。**只有 `structured_evidence` 允许不同** —— 由 `arm_diff()` 守着。

    这里只放**原始值** (不放 `KernelDatabase` 这类实例): diff 是按键值比出来的, 混进实例
    就会把「两个不同的 off 库」读成一处差异, 于是守卫自己先失效。
    """
    return {"structured_evidence": bool(structured_evidence),
            "max_optimize_rounds": ROUNDS,
            "heartbeat_interval": HEARTBEAT_INTERVAL,
            "use_coder_pipeline": False}


def arm_diff() -> set:
    """两臂的键差集 —— 现算, 不是注释里写死的「只差一个键」"""
    on, off = arm_kwargs(True), arm_kwargs(False)
    keys = set(on) | set(off)
    return {k for k in keys if on.get(k, "<缺>") != off.get(k, "<缺>")}


# ─────────────────────── 留档: prompt 与替身 ───────────────────────

class ScriptedLlm:
    """真 LLM 端点的**选择性替身**: 生成侧由脚本给定, 修复侧转发 (并逐字留档)

    为什么生成侧要替: 要一个**确定会失败**的第一份候选, 就不能让 27B 随机决定 ——
    否则两臂的起点都不同, 对照从第一步就不干净。修复侧**必须**是真 LLM: 证据段就是进
    那里的, 替掉它这个实验就没了对象。
    """

    def __init__(self, inner, scenario):
        self.inner = inner
        self.scenario = scenario
        self.sent = []                   # [(caller_tag, prompt)]

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def chat(self, messages, **kw):
        tag = str(kw.get("caller_tag") or "")
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        self.sent.append((tag, prompt))
        if tag == "Coder":
            return self.scenario["broken"]
        return self.inner.chat(messages, **kw)

    def chat_json(self, messages, **kw):
        tag = str(kw.get("caller_tag") or "")
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        self.sent.append((tag, prompt))
        if tag in ("Designer", "Planner", "IntentParser"):
            return dict(SKETCH)
        return self.inner.chat_json(messages, **kw)

    def prompts(self, tag, since=0):
        return [p for t, p in self.sent[since:] if t == tag]


class DryLlm:
    """零 token 的替身: 生成侧给坏码, 修复侧**一口价**给好码 (只用来验装配与留痕)"""

    def __init__(self, scenario):
        self.scenario = scenario
        self.sent = []

    def chat(self, messages, **kw):
        tag = str(kw.get("caller_tag") or "")
        self.sent.append((tag, "\n".join(str(m.get("content", "")) for m in messages)))
        if tag == "Coder":
            return self.scenario["broken"]
        return self.scenario["good"]

    def chat_json(self, messages, **kw):
        self.sent.append((str(kw.get("caller_tag") or "?"), ""))
        return dict(SKETCH)

    def prompts(self, tag, since=0):
        return [p for t, p in self.sent[since:] if t == tag]


class RecordingBackend:
    """只记账的外壳: 每一次**可能花 GPU 秒**的调用都记下方法名 (次数=成本代理)"""

    RECORDED = ("compile", "execute", "benchmark", "benchmark_with_baselines",
                "analyze_dependencies")

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _rec(self, method, *a, **kw):
        self.calls.append(method)
        return getattr(self.inner, method)(*a, **kw)

    def compile(self, *a, **kw):
        return self._rec("compile", *a, **kw)

    def execute(self, *a, **kw):
        return self._rec("execute", *a, **kw)

    def benchmark(self, *a, **kw):
        return self._rec("benchmark", *a, **kw)

    def benchmark_with_baselines(self, *a, **kw):
        return self._rec("benchmark_with_baselines", *a, **kw)

    def analyze_dependencies(self, *a, **kw):
        return self._rec("analyze_dependencies", *a, **kw)


# ─────────────────────── 两臂只差那一段 (实证) ───────────────────────

def strip_evidence_block(prompt: str) -> str:
    """切掉证据段 (从它的标题到下一个 `## ` 标题之前)"""
    i = prompt.find(EVIDENCE_HEADING)
    if i < 0:
        return prompt
    j = prompt.find(NEXT_HEADING, i)
    if j < 0:
        raise ValueError("证据段后面找不到下一个标题 —— prompt 形状变了")
    return prompt[:i] + prompt[j:]


def _squeeze(text: str) -> str:
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def two_arms_differ_only_on_the_block(on_prompt: str, off_prompt: str) -> dict:
    """两臂 prompt 的差是不是**恰好**那一段

    两档读数一起报, 因为字节级那一档今天**必然**差一点点: 模板里 `{{structured_evidence}}`
    自己占一行, 空串替换后留下的是两个换行 ⇒ 砍掉证据段之后还差 2 个空白字符 (`seam_chars`)。
    真正该问的是「除了空白, 还有没有别的残差」, 那就是 `squeezed_residual_chars` —— 它不为 0
    才说明两臂不止差那一段。不写成 `assert` 就完事: 这个函数返回的是**读数**, 判定在调用方
    (`pair_up` 把不干净的格子标成作废, 而不是悄悄降级)。
    """
    stripped = strip_evidence_block(on_prompt)
    return {"strict": stripped == off_prompt,
            "normalized": _squeeze(stripped) == _squeeze(off_prompt),
            "seam_chars": len(stripped) - len(off_prompt),
            "squeezed_residual_chars": len(_squeeze(stripped)) - len(_squeeze(off_prompt)),
            "block_chars": len(on_prompt) - len(stripped)}


def error_excerpt(prompt: str, limit: int = 300) -> str:
    """从留档的 prompt 里读**第 1 轮的真实错误原话** —— 「这个场景真的按设计失败了」的证人

    `artifact.verification_results` 每轮被覆盖 (`conductor.py:1496`/`:2239`), 于是出了循环
    只剩最后一轮的分级。要证「这一格确实死在语法/接口/数值门上」, 手边唯一没被覆盖的原料
    就是这份 prompt (两臂都有, 因为 `error_details` 与开关无关)。
    """
    i = prompt.find("## 错误信息")
    if i < 0:
        return ""
    j = prompt.find("\n## ", i + 4)
    body = prompt[i:j if j > 0 else len(prompt)]
    lines = [ln for ln in body.splitlines()[1:] if ln.strip()]   # 丢掉标题那一行
    # 模板自己那句固定话 (`以下代码未通过验证。本轮报失败的阶段: …`) 不是错误原话: 留着它,
    # 「场景自检」印出来的首行对三个场景完全一样 —— 那时它就没在作证
    lines = [ln for ln in lines if not ln.startswith("以下代码未通过验证")]
    return "\n".join(lines)[:limit]


def history_cases_are_empty(prompt: str) -> bool:
    """干净两臂的第二条: 历史修复案例那一段必须是「无」(失败库/长期记忆不许串味)

    标题那一行自己带着括注 (`## 历史修复案例 (同类算子的失败-修复经验)`), 所以先丢掉标题行,
    再看第一行非空内容 —— 它必须逐字是「无」。
    """
    i = prompt.find(NEXT_HEADING)
    if i < 0:
        return False
    body = [ln.strip() for ln in prompt[i:].splitlines()[1:]]
    lines = [ln for ln in body if ln]
    return bool(lines) and lines[0] == "无"


# ─────────────────────────── 一次运行 ───────────────────────────

def _empty_run(scenario, arm, wall_s, remote_calls, error) -> dict:
    """崩掉的那一次也要留下一条**形状完整**的记录 —— 缺键会被读成「没记」"""
    return {"scenario": scenario["id"], "arm": arm, "wall_s": wall_s,
            "rounds": 0, "final_passed": False, "repaired": False, "rounds_recs": [],
            "round1_levels": {}, "round1_failed_levels": [], "last_l1_status": None,
            "error_excerpt": "", "fix_calls": 0, "fix_prompt_chars": [],
            "fix_prompt_first": None, "n_fix_prompt_blocks": 0,
            "fix_prompt_sha256": [], "evidence_chars": [],
            "evidence_injected": [], "profiler_facts": [], "parent_diff_lines": [],
            "tokens_total": 0, "tokens_prompt": 0, "tokens_completion": 0,
            "llm_calls": 0, "remote_calls": remote_calls, "stop": None,
            "over_wall_budget": False, "error": error}


def run_one(scenario, *, structured_evidence, llm, backend, task, inputs,
            wall_budget_s=240.0) -> dict:
    """跑一臂。**只有 `structured_evidence` 一个键**会不同 (由 `arm_kwargs` 保证)

    ⚠️ 崩了**不往外抛**: 真机跑一次要烧 token, 而这一跑已经真撞上过两次工作区中间态
    (`NameError: ActionEnvelope` / `PATCH_KINDS` —— 别人正在改 `conductor.py`)。
    一次异常若把整跑带走, 读数全丢, 且事后分不清「树坏了」与「实验没跑」。
    所以这里记一条带 `error` 的记录: `pair_up` 会因为 `fix_calls == 0` 把它判成不可用,
    结论句里再逐条点名 —— 崩掉的臂**不许**被当成「那一臂没通过」。
    """
    from operator_agent import Conductor
    from operator_agent.kernel_db import FailureDatabase, KernelDatabase

    arm = "on" if structured_evidence else "off"
    sent_before = len(getattr(llm, "sent", []))
    calls_before = len(getattr(backend, "calls", []))
    t0 = time.monotonic()
    try:
        conductor = Conductor(backend=backend, llm=llm,
                              kernel_db=KernelDatabase(jsonl_path=DB_OFF),
                              failure_db=FailureDatabase(jsonl_path=DB_OFF),
                              **arm_kwargs(structured_evidence))
        art = conductor.execute(task, inputs, target_gflops=0.0)
    except Exception as exc:                      # noqa: BLE001 — 见 docstring: 崩了也要留读数
        return _empty_run(scenario, arm, round(time.monotonic() - t0, 1),
                          len(getattr(backend, "calls", [])) - calls_before,
                          f"{type(exc).__name__}: {exc}")
    wall = round(time.monotonic() - t0, 1)

    hist = list(art.iteration_history or [])
    rounds_recs = [{"round": h.get("round"), "action": h.get("action"),
                    "all_passed": bool(h.get("all_passed")),
                    "levels": h.get("levels_passed") or {}, "gflops": h.get("gflops"),
                    "wall_s": h.get("measurement_wall_s"),
                    "fix_evidence": h.get("fix_evidence") or {}}
                   for h in hist]
    # 逐轮分级只能从血缘读: `artifact.verification_results` 每轮被覆盖
    # (`conductor.py:1496`/`:2239`), 出了循环只剩最后一轮 —— 「第 1 轮卡在哪一级」
    # 这个问句在 `verification_results` 上会得到错误答案 (第一版就是这么读的, 全绿)。
    first_levels = rounds_recs[0]["levels"] if rounds_recs else {}
    failed_levels = [lv for lv, ok in first_levels.items() if not ok]
    fix_prompts = llm.prompts("Fixer", since=sent_before)
    tracker = getattr(llm, "cost_tracker", None)
    usage = (tracker.summary_since(conductor._cost_watermark) if tracker is not None
             else {})
    evidence = [r["fix_evidence"] for r in rounds_recs]
    return {
        "scenario": scenario["id"], "arm": arm,
        "wall_s": wall, "rounds": len(rounds_recs),
        "final_passed": bool(rounds_recs and rounds_recs[-1]["all_passed"]),
        "repaired": bool(any(r["all_passed"] for r in rounds_recs[1:])),
        "rounds_recs": rounds_recs,
        "round1_levels": first_levels,
        "round1_failed_levels": failed_levels,
        "last_l1_status": next(((v.metrics or {}).get("compile_status")
                                for v in reversed(art.verification_results or [])
                                if v.level == VerificationLevel.L1_COMPILE), None),
        "error_excerpt": error_excerpt(fix_prompts[0]) if fix_prompts else "",
        "fix_calls": len(fix_prompts),
        "fix_prompt_chars": [len(p) for p in fix_prompts],
        # prompt 的指纹: 「同一份 prompt 出了不同修复」这件事只能靠指纹事后查证 ——
        # 2026-10-05 那一跑就撞上了 (同长 13516、证据段同 685, 一次带 `input_precision="ieee"`
        # 一次不带), 而当时只存了长度 ⇒ 只能推断、不能证明。
        "fix_prompt_sha256": [hashlib.sha256(p.encode("utf-8")).hexdigest()
                              for p in fix_prompts],
        "fix_prompt_first": fix_prompts[0] if fix_prompts else None,
        "n_fix_prompt_blocks": sum(1 for p in fix_prompts
                                   if EVIDENCE_HEADING in p),
        "evidence_chars": [e.get("chars") for e in evidence if e.get("injected")],
        "evidence_injected": [bool(e.get("injected")) for e in evidence],
        "profiler_facts": [sorted(e.get("profiler_facts") or []) for e in evidence],
        "parent_diff_lines": [list(e.get("parent_diff_lines") or []) for e in evidence],
        "tokens_total": int(usage.get("total_tokens", 0) or 0),
        "tokens_prompt": int(usage.get("total_prompt_tokens", 0) or 0),
        "tokens_completion": int(usage.get("total_completion_tokens", 0) or 0),
        "llm_calls": int(usage.get("total_calls", 0) or 0),
        "remote_calls": len(getattr(backend, "calls", [])) - calls_before,
        "stop": ((art.metadata or {}).get("stop") or {}).get("reason"),
        "over_wall_budget": wall > wall_budget_s,
    }


# ─────────────────────────── 报表 ───────────────────────────

def pair_up(runs: list) -> list:
    """按场景把两臂配成对 (顺序无关): 只配有 Fixer 的格子

    `usable` 三条同时成立才算干净的一对 (否则读数作废, 不是「降级使用」):
      ① 两臂都有 Fixer 调用 (没有失败就没有对照物);
      ② 砍掉证据段后两臂 prompt 除空白外**逐字相同** (`squeezed_residual_chars == 0`);
      ③ 历史修复案例那一段两臂都是「无」(失败库没串味)。
    """
    pairs = []
    for sid in dict.fromkeys(r["scenario"] for r in runs):
        on = [r for r in runs if r["scenario"] == sid and r["arm"] == "on"]
        off = [r for r in runs if r["scenario"] == sid and r["arm"] == "off"]
        for a, b in zip(on, off):
            pair = {"scenario": sid, "on": a, "off": b, "usable": False}
            if min(a["fix_calls"], b["fix_calls"]) > 0:
                pair["prompt_diff"] = two_arms_differ_only_on_the_block(
                    a["fix_prompt_first"], b["fix_prompt_first"])
                pair["history_clean"] = (history_cases_are_empty(a["fix_prompt_first"])
                                         and history_cases_are_empty(b["fix_prompt_first"]))
                pair["usable"] = bool(pair["prompt_diff"]["normalized"]
                                      and pair["prompt_diff"]["squeezed_residual_chars"] == 0
                                      and pair["history_clean"])
            pairs.append(pair)
    return pairs


def _error_lines(runs: list) -> list:
    """崩掉的运行逐条点名 —— 「那一臂没通过」与「那一臂根本没跑完」不许混成一句话"""
    errors = [f"{r['scenario']}/{r['arm']}: {r['error']}" for r in runs if r.get("error")]
    if not errors:
        return []
    return ["  · ❌ 这些运行**直接崩了** (工作区被改到中间态 / 后端不可用), 已从上面所有比例里"
            "排除 (不许读成「那一臂没通过」): " + "; ".join(errors)]


def conclusion(runs: list, pairs: list) -> str:
    """结论句: 必须带 N、差值、量程, 且**不许**出现「有收益/无收益」"""
    from benchmarks.drivers.harness_pass_rate import min_detectable_delta

    usable = [p for p in pairs if p["usable"]]
    n = len(usable)
    errs = _error_lines(runs)
    if not n:
        # 崩了也要点名: 早退分支若只说「没有一格跑出 Fixer」, 「树坏了」就被读成
        # 「实验没做」—— 而这两件事对读者要做的事完全不同 (前者重跑, 后者改场景)
        return "\n".join(["结论: 没有一格跑出 Fixer (第一轮就过 / 根本没失败 / 整跑崩了) ⇒ "
                           "**这一跑什么也证明不了**, 一个数字都不许引用"] + errs)
    on_pass = sum(1 for p in usable if p["on"]["final_passed"])
    off_pass = sum(1 for p in usable if p["off"]["final_passed"])
    delta_pp = round(100.0 * (on_pass - off_pass) / n, 1)
    detectable = round(100.0 * min_detectable_delta(n), 1)
    same = all(p["on"]["final_passed"] == p["off"]["final_passed"] for p in usable)
    blocks = [p["prompt_diff"]["block_chars"] for p in usable]
    residual = {p["prompt_diff"]["squeezed_residual_chars"] for p in usable}
    seams = {p["prompt_diff"]["seam_chars"] for p in usable}
    strict = all(p["prompt_diff"]["strict"] for p in usable)
    tok = [p["on"]["tokens_prompt"] - p["off"]["tokens_prompt"] for p in usable]
    rounds_on = [p["on"]["rounds"] for p in usable]
    rounds_off = [p["off"]["rounds"] for p in usable]
    lines = [
        f"结论: N={n} 个场景 × 2 臂 (每格两臂只差 `structured_evidence`: 键差集 {sorted(arm_diff())});",
        f"  · 成本 (可分辨的一侧): 证据段 {min(blocks)}~{max(blocks)} chars, "
        f"砍段后非空白残差 {sorted(residual)} chars (纯空白残差 {sorted(seams)} chars), "
        f"两臂 prompt {'字节级相同' if strict else '字节级不同(见残差)'}; "
        f"prompt token 增量 {tok};",
        "  · 红在哪: " + "; ".join(
            f"{p['scenario']}: 开={p['on']['stop']} 关={p['off']['stop']}"
            for p in usable),
        f"  · 结果 (分辨不出的一侧): 通过 {on_pass}/{n} (开) vs {off_pass}/{n} (关), "
        f"差值 {delta_pp:+} pp, 轮数 (开)={rounds_on} (关)={rounds_off}"
        + ("; 两臂读数**完全相同** ⇒ 这个量具在 N 这么小时分辨不出" if same else "") + ";",
        f"  · 量程: N={n} 时通过率量具最多只能分辨 ≈{detectable} pp 的差 "
        f"(`harness_pass_rate.min_detectable_delta`) ⇒ **N 太小, 不许声称收益, 也不许声称无收益**。",
    ]
    bad = [p["scenario"] for p in pairs if not p["usable"]]
    if bad:
        lines.append(f"  · ⚠️ 这些场景没有干净的一对 (没 Fixer 调用 / 两臂不止差那一段) ⇒ "
                     f"对照读数作废, 已排除: {bad}")
    lines.extend(errs)
    return "\n".join(lines)


def _file_hashes() -> dict:
    """这一跑读的是**哪一份**代码 —— 真机读数必须自带它的条件

    这不是形式主义: 跑之前刚发生过一次「别人正在改 `conductor.py`」的中间态 (导入直接
    NameError)。读数落盘时若不带文件指纹, 事后无从判断那次读数量的是哪一版流水线。
    """
    import hashlib
    keys = ("operator_agent/conductor.py", "operator_agent/agent_runtime.py",
            "operator_agent/fixer.py", "operator_agent/prompts/fixer_repair.md",
            "benchmarks/drivers/evidence_ablation_h20.py")
    out = {}
    for key in keys:
        path = ROOT / key
        if path.exists():
            out[key] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return out


def _table(runs: list) -> str:
    head = (f"{'场景':<10} {'臂':<4} {'通过':<5} {'轮':<3} {'首轮挂':<16} {'Fixer':<6} "
            f"{'证据chars':<10} {'prompt':<14} {'tok(in)':<8} {'tok(all)':<9} "
            f"{'远端':<5} {'墙钟':<7}")
    rows = [head, "-" * len(head)]
    for r in runs:
        rows.append(f"{r['scenario']:<10} {r['arm']:<4} "
                    f"{('崩' if r.get('error') else ('绿' if r['final_passed'] else '红')):<5} "
                    f"{r['rounds']:<3} "
                    f"{','.join(r['round1_failed_levels'])[:15]:<16} "
                    f"{r['fix_calls']:<6} {str(r['evidence_chars']):<10} "
                    f"{str(r['fix_prompt_chars']):<14} {r['tokens_prompt']:<8} "
                    f"{r['tokens_total']:<9} {r['remote_calls']:<5} {r['wall_s']:<7}")
    return "\n".join(rows)


# ─────────────────────────── 入口 ───────────────────────────

def _llm_from_config():
    from operator_agent import LLMClient
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        sys.exit(f"缺 {cfg_path}: 请从 config.example.yaml 复制并填 llm.base_url")
    cfg = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("llm") or {}
    if not cfg.get("base_url"):
        sys.exit("config.yaml 的 llm.base_url 为空")
    return LLMClient(base_url=cfg["base_url"], model=cfg.get("model", "Qwen3.8-27B"),
                     temperature=cfg.get("temperature", 0.3),
                     max_tokens=cfg.get("max_tokens", 4096),
                     timeout=cfg.get("timeout", 120))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="离线自检: MockGPU + 零 token 替身, 只验装配与留痕")
    ap.add_argument("--reps", type=int, default=1, help="每臂重复几次 (预算见 --max-runs)")
    ap.add_argument("--scenarios", default="", help="逗号分隔, 只跑这几格")
    ap.add_argument("--max-runs", type=int, default=12, help="总运行次数上限 (硬约束)")
    ap.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP,
                    help="L3 性能门 (默认 0.5; 保底模板实测 0.47x ⇒ 想量「修复是否成功」"
                         "就调低到 0.3, 读数里会记下这一档)")
    ap.add_argument("--wall-budget-s", type=float, default=240.0,
                    help="单次运行的墙钟上限, 超了就停下并如实报告")
    ap.add_argument("--out", default="", help="读数落盘路径 (默认 output/ 下带时间戳)")
    args = ap.parse_args()

    shape = (64, 64) if args.dry_run else (1024, 1024)
    dsl, hardware = ("python", "mock_gpu") if args.dry_run else ("triton", "nvidia_gpu")
    scenarios = _scenarios(dsl, hardware, shape)
    if args.scenarios:
        want = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s["id"] in want]
        if not scenarios:
            sys.exit(f"--scenarios 没匹配到任何场景: {args.scenarios}")

    plan = len(scenarios) * 2 * args.reps
    if plan > args.max_runs:
        sys.exit(f"计划 {plan} 次 > 上限 {args.max_runs}: 减场景或减 reps (token 不是免费的)")

    real_llm = None
    if not args.dry_run:
        real_llm = _llm_from_config()
        t0 = time.monotonic()
        pong = real_llm.chat([{"role": "user", "content": "回复两个字: 在线"}],
                             max_tokens=8, caller_tag="Preflight")
        print(f"[预检] {real_llm.model} @ {real_llm.base_url} → {pong.strip()!r} "
              f"({round(time.monotonic() - t0, 1)}s)", flush=True)

    from benchmarks.experiments.misc import provenance
    tree = {"stamp": provenance.stamp(str(ROOT)), "files": _file_hashes()}
    print(f"[代码] {tree['stamp']}", flush=True)
    print(f"[代码] 指纹 {json.dumps(tree['files'], ensure_ascii=False)}", flush=True)
    print(f"[两臂] 键差集 = {sorted(arm_diff())} (必须只有一个键); "
          f"L3 门 min_speedup={args.min_speedup}", flush=True)
    if arm_diff() != {"structured_evidence"}:
        sys.exit("两臂不止差一个开关 ⇒ 这不是对照实验, 停下")

    from operator_agent.kernel_db import FailureDatabase, KernelDatabase
    from operator_agent.mock_gpu import MockGPU

    task = _task(dsl=dsl, hardware=hardware, min_speedup=args.min_speedup)
    inputs = _inputs(shape)
    runs, aborted = [], ""
    order = 0
    for rep in range(1, args.reps + 1):
        for si, scenario in enumerate(scenarios):
            # 场景之间轮换两臂顺序: 同一场景的两臂背靠背, 免得把端点漂移读成开关效应
            arms = [True, False] if (si + rep) % 2 else [False, True]
            for on in arms:
                if args.dry_run:
                    llm = DryLlm(scenario)
                    backend = RecordingBackend(MockGPU())
                else:
                    from operator_agent import RemoteExecutor
                    llm = ScriptedLlm(real_llm, scenario)
                    backend = RecordingBackend(RemoteExecutor(mode="ssh"))
                order += 1
                print(f"\n[{order}/{plan}] 场景={scenario['id']} 臂="
                      f"{'开' if on else '关'} rep={rep} "
                      f"(预期拦在: {scenario['expect']})", flush=True)
                rec = run_one(scenario, structured_evidence=on, llm=llm,
                              backend=backend, task=task, inputs=inputs,
                              wall_budget_s=args.wall_budget_s)
                rec["rep"] = rep
                rec["order"] = order
                runs.append(rec)
                if rec.get("error"):
                    # 崩了就**停下整跑**: 树被改到中间态时, 后面每一格都会先烧一次 Fixer
                    # 再死在同一处 —— 那是拿 token 换一串同样的异常。
                    print(f"  → ❌ 崩了: {rec['error']}", flush=True)
                    aborted = (f"场景 {scenario['id']} 那一格崩了 ({rec['error']}) ⇒ "
                               f"停下, 剩余计划不跑 (工作区可能正被改到中间态)")
                    break
                print(f"  → {'绿' if rec['final_passed'] else '红'} "
                      f"轮数={rec['rounds']} Fixer={rec['fix_calls']} "
                      f"证据={rec['evidence_chars']} prompt={rec['fix_prompt_chars']} "
                      f"tok={rec['tokens_prompt']}in/{rec['tokens_total']}all "
                      f"远端={rec['remote_calls']} 墙钟={rec['wall_s']}s "
                      f"stop={rec['stop']}", flush=True)
                if rec["over_wall_budget"]:
                    aborted = (f"场景 {scenario['id']} 单次 {rec['wall_s']}s 超过 "
                               f"{args.wall_budget_s}s ⇒ 停下, 剩余计划不跑")
                    print(f"  ⚠️ {aborted}", flush=True)
                    break
            if aborted:
                break
        if aborted:
            break

    pairs = pair_up(runs)
    print("\n" + _table(runs), flush=True)
    print("\n[逐轮] " + json.dumps(
        [{k: r[k] for k in ("scenario", "arm", "round1_levels", "evidence_injected",
                            "evidence_chars", "profiler_facts", "parent_diff_lines")}
         for r in runs], ensure_ascii=False), flush=True)
    # 场景自检: 第一轮到底死在哪一级 + 那一轮的真实错误原话 (只从留档 prompt 里读得到)
    for sid in dict.fromkeys(r["scenario"] for r in runs):
        r = next(x for x in runs if x["scenario"] == sid and x["fix_calls"] > 0)
        print(f"[场景自检] {sid}: 首轮挂={r['round1_failed_levels']} "
              f"错误原话首行={r['error_excerpt'].splitlines()[0][:90] if r['error_excerpt'] else '(无)'}",
              flush=True)
    for p in pairs:
        if "prompt_diff" in p:
            print(f"[两臂 prompt] 场景={p['scenario']} "
                  f"非空白残差={p['prompt_diff']['squeezed_residual_chars']} chars "
                  f"纯空白残差={p['prompt_diff']['seam_chars']} chars "
                  f"证据段={p['prompt_diff']['block_chars']} chars "
                  f"历史案例段为空={p['history_clean']} 干净={p['usable']}", flush=True)
        else:
            print(f"[两臂 prompt] 场景={p['scenario']} 没有 Fixer 调用 ⇒ 无从对照", flush=True)
    text = conclusion(runs, pairs)
    print("\n" + text, flush=True)
    if aborted:
        print(f"⚠️ 这一跑**被截断**: {aborted}", flush=True)

    out = Path(args.out) if args.out else (
        ROOT / "output" / f"evidence_ablation_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "tree": tree,
               "mode": "dry-run" if args.dry_run else "real",
               "model": (real_llm.model if real_llm is not None else "DryLlm"),
               "endpoint": (real_llm.base_url if real_llm is not None else "离线"),
               "arm_diff": sorted(arm_diff()), "plan": plan, "aborted": aborted,
               "min_speedup": args.min_speedup,
               "scenarios": [{k: s[k] for k in ("id", "expect", "dsl", "hardware")}
                             for s in scenarios],
               "runs": [{k: v for k, v in r.items() if k != "fix_prompt_first"}
                        for r in runs],
               "pairs": [{k: v for k, v in p.items()
                          if k not in ("on", "off")} |
                         {"on_final_passed": p["on"]["final_passed"],
                          "off_final_passed": p["off"]["final_passed"],
                          "on_tokens_prompt": p["on"]["tokens_prompt"],
                          "off_tokens_prompt": p["off"]["tokens_prompt"]}
                         for p in pairs],
               "conclusion": text}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\n读数已落盘: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())