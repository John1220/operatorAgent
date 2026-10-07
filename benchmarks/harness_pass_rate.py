"""端到端成功率量具: 这条流水线现在到底能不能产出「又快又对」的核

为什么单独有这个驱动: 67 个提交里改了一大堆判定与口径, 但**「生成的核有多少比例能过 L2、
中位加速比多少、平均烧几轮」这三个数, 从头到尾没有被量过**。`tests/golden_regression.py`
跑在 MockGPU 上、用的是「伪 GFLOPS」, 结构上答不了这个问题; `benchmarks/` 里其余 30 个驱动
全是单点探针 (一次量一个机制), 没有一个产出聚合通过率。

于是四件事同时悬着, 且都卡在这一个量具上:
  · C1 处方案式错误反馈的真实效应 (doc 35 的 A/B 打平, 而那个设计看不见效应);
  · doc 47 没量出来的候选重复基率;
  · `peak_memory_ratio` 该定什么阈值 (doc 44 只有 4 个点);
  · C3 最后一跳 (复发计数喂进 Planner) 接上之后到底有没有用。

跑法 (需要 GPU 机的 ssh 别名 + config.yaml 里的 LLM 端点):
    python3 benchmarks/harness_pass_rate.py            # 全量
    python3 benchmarks/harness_pass_rate.py --repeats 1 # 先冒烟

    # 改了一处配方/prompt 之后, 拿两份落盘报表判「有没有用」(不需要 GPU):
    python3 benchmarks/harness_pass_rate.py --compare A/pass_rate.json B/pass_rate.json

「能判变化」不是把两次的百分数并排看一眼: 5/6 → 3/6 时 Fisher 精确双侧 p=0.545
(doc 48 §六B 亲算过), n=6 这个量级**判不出** 30 个百分点的回归。所以报表必须同时
说出「当前样本能分辨多大的差」, 而「没测出差异」只有在**本来测得出**的时候才允许说。

⚠️ 四条读数限制, 都在报表里显式打印, 不要脱离它们引用数字:
  1. **语料只有两个算子** (matmul fp16 / layernorm fp32)。加算子的前提是输入契约已经
     验证过 —— doc 41 记录过一次真实事故: attention 因为多传一个 `causal` 连烧三轮。
     凭空造输入会让「通过率」量到契约错误而不是模型能力。
  2. **分母小**。每次运行通常第 1 轮就通过 (生产门槛是 1.0x), 所以轮数≈运行数。
     分母不足 MIN_DENOM 时本驱动**拒绝打印百分数** —— 「0/1」和「0/12」都写成 0.0%
     是这份量具第一版真实犯过的错。
  3. **通过率与档位必须一起读**。一个在 `uninformative` 档上「通过」的核, 它的加速比
     没有资格驱动任何闭环决策 (缺口 1)。所以报表把 decision-grade 占比与通过率并列,
     单看通过率会把「盲跑但碰巧过了」读成成功。
  4. **单一通过率是门槛位置的函数** (doc 48 §6.2 ①)。把用例选在「中位加速比≈门槛」处,
     83.3% 这个数换一套 shape 就会变。报表因此附一条门槛扫描曲线 (默认 0.9/1.0/1.1x),
     并且曲线是**对同一批已生成候选的重新计分** —— 不是「把门槛改低再跑一遍」:
     搜索行为本身是被当时的门槛塑形的, 改低门槛重跑会生成另一批候选。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from operator_agent.backend_caps import Cap  # noqa: E402
from operator_agent.conductor import Conductor  # noqa: E402
from operator_agent.bench_cases import case_from_inputs, case_key  # noqa: E402
from operator_agent.run_layout import new_run_root  # noqa: E402
from operator_agent.llm_client import (  # noqa: E402
    LLMClient, format_sampling_configs)
from operator_agent.measurement import (BAND_SAME, BAND_SPEC_DRIFT,  # noqa: E402
                                        BAND_TIER_EXPLAINED, BAND_UNEXPLAINED,
                                        BAND_UNMEASURABLE, LINEAGE_BAND_KEY,
                                        LINEAGE_PERF_RULER_KEY, LINEAGE_RULER_KEY,
                                        PERF_GATE_PARAM_KEYS, TRANSPORT_ANSWERED,
                                        TRANSPORT_KEY, TRANSPORT_UNKNOWN, Validity,
                                        band_note, attributable_evidence,
                                        correctness_verified_levels, cross_round_band,
                                        cross_round_ruler, persist_decision)
from operator_agent.remote_executor import RemoteExecutor  # noqa: E402
from operator_agent.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema, TaskType,
)

# 分母下限: 低于它不出百分数。理由见文件头第 2 条。
MIN_DENOM = 4

# 语料。shape 选的是 doc 31/44 真机标定过的两档: matmul 4096³ 落 `valid`
# (launch 地板占比 0.020), layernorm 16384×6144 落 `degraded` (0.111)。
# 故意一档一个: 报表要能看出「通过率」与「读数可不可信」是两件事。
CASES = [
    ("matmul", Precision.FP16, (4096, 4096, 4096)),
    ("layernorm", Precision.FP32, (16384, 6144)),
]
# 生产门槛, 不是 doc 47 那个「故意达不到」的 10x。这里量的是「达不达标」,
# 所以必须用真实门槛, 否则量到的是一个永远失败的循环。
MIN_SPEEDUP = 1.0
MAX_ERROR = 1e-2
#: §6.2 条款 3 的**操作者声明**（不是机器学出来的常量）：关键 shape 上允许的回退比例。
#: 判据要的是「有一个被声明过的上限」，值本身是人的判断；它只影响 §17.1 的实现裁决
#: （「自己实现 vs 直接用库」那一条建议），**不**参与入库门（§4.4 的阶梯与品种门都不读它）。
MAX_REGRESSION = 0.05

#: §6.2 **聚合口径**：声明出「这次跑的目标函数是什么」。单 case 的权重就是 1（今天这个
#: 驱动一条 case 一条 case 地跑），所以 `weighted_geomean(1 条 case)` = 那一条的加速比
#: —— 这个数的用途是让目标在产物里**可查**，不是门槛（准入仍看主 case 的加速比，见
#: CHECKLOG I93 的仍欠）。多 case 的权重是**声明的**，不是机器学出来的。
SINGLE_CASE_WEIGHT = 1.0
CASE_AGGREGATE = "weighted_geomean"


def llm_from_config() -> LLMClient:
    """从 config.yaml 读自建端点。值只进环境变量, **不打印**。"""
    import yaml
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        llm = (cfg.get("llm") or {})
        os.environ.setdefault("LLM_BASE_URL", str(llm.get("base_url") or ""))
        os.environ.setdefault("LLM_MODEL", str(llm.get("model") or ""))
        os.environ.setdefault("LLM_API_KEY", str(llm.get("api_key") or ""))
    client = LLMClient(temperature=0.3, max_tokens=4096)
    # 判别实验的输入必须是**同一个模型**: 混进付费降级模型的话, 量的就是两个模型的差,
    # 而不是机制的差 (doc 47 的驱动同一条纪律)
    if getattr(client, "fallback_model", None):
        raise SystemExit("❌ 本次测量不允许走降级模型 (会污染通过率的口径)")
    return client


def _task(kernel_type: str, precision: Precision, *, required_cases=(),
          max_regression=None, case_weights=None, cases=()) -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, complexity=TaskComplexity.MEDIUM),
        # 这份量具的表头写着「真机」, 所以任务就把「要真机」声明出来。声明的读者是
        # `Verifier._verify_performance`: 执行侧给不出 `REAL_DEVICE` 时 L3 不再记为通过
        # (doc 49 §3.5 第 2/3 处)。`main()` 里那道 `backend.require(...)` 是**开工前**的
        # 硬门, 挡的是「门面静默降级成 LocalExecutor 而报表照旧写着真机」; 这一条是
        # **每一轮读数**上的门, 挡的是「某一路载荷其实是在 CPU 上测的」。两道不重复。
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton",
                          required_caps=(Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE)),
        constraints=ConstraintSpec(precision=precision, min_speedup=MIN_SPEEDUP,
                                   max_error=MAX_ERROR,
                                   required_cases=tuple(required_cases or ()),
                                   max_regression=dict(max_regression or {}),
                                   case_weights=dict(case_weights or {}),
                                   aggregate=CASE_AGGREGATE,
                                   # §3.4（I94）: 让**契约拥有输入** —— 声明由这组输入派生
                                   cases=tuple(cases or ())),
    )


def inputs_for(kernel_type: str, shape):
    """与 `measure_candidate_duplicates.py` 同一套输入 —— 已验证能端到端跑通的契约"""
    if kernel_type == "matmul":
        m, n, k = shape
        return {"A": np.random.randn(m, k).astype(np.float32),
                "B": np.random.randn(k, n).astype(np.float32)}
    m, n = shape
    return {"X": np.random.randn(m, n).astype(np.float32),
            "weight": np.ones(n, dtype=np.float32),
            "bias": np.zeros(n, dtype=np.float32)}


# ─── 聚合: 纯函数, 可单测 ─────────────────────────────────────
# 放在这里而不是内联在 main 里, 是为了让「报表算法」本身可以被假数据验一遍。
# 这个项目的教训 (doc 46): 判据方向写反时, 量具明明看见了信号却报「看不见」——
# 会自我批评的错误判据比缺失的更难发现。所以聚合逻辑必须能离线跑。

def _level_rate(rounds, level):
    """某一级的通过率。**分母是「走到了这一级的轮数」**, 不是总轮数。

    L1 就挂掉的轮次根本没有 L2/L3 的条目; 拿总轮数当分母会把「没走到」算成「没通过」,
    于是 L2/L3 的通过率被 L1 的失败率污染 —— 这正是 doc 47 第一版把 `generate` 轮
    算进 optimize 分母的同一个错。
    """
    reached = [r for r in rounds if level in (r.get("levels_passed") or {})]
    passed = [r for r in reached if (r["levels_passed"] or {}).get(level)]
    return len(passed), len(reached)


def _pct(num, den):
    return round(num / den * 100, 1) if den else None


def last_round_correctness_passed(rec) -> bool:
    """「摘掉真机那一栏」的判定: **最后一轮**那份代码的 L1 与 L2 都通过

    外部一手 (arXiv:2609.26777v1 行 99-101, 逐字): "45.9% under the verifier versus
    69.4% with E2E tests excluded from scoring" —— 同一批运行、两套记分口径并排报,
    那个差就是「本地过 ≠ 生产过」的量 (#77)。

    取**最后一轮**而不是「曾经有哪一轮过」: 两个口径必须落在同一份代码上。
    `final_all_passed` 说的是最终交付那一版, 而「第 2 轮对了、第 3 轮改坏了」的运行
    按「曾经过」会被算成本地通过 —— 那是一个关于历史的问题, 不是关于这份代码的问题。
    谓词本身在 `measurement.correctness_verified_levels`: 修订起点 (#85) 与这一栏读同
    一个集合, 两处各写一遍就是两套口径。
    """
    rounds = rec.get("rounds") or []
    if not rounds:
        return False
    return correctness_verified_levels(rounds[-1].get("levels_passed") or {})


def _duplicate_rate(ok_runs):
    """候选重复率 —— doc 47 没能量到的那个基率, 顺手在每份报表里带上。

    **分母是「本轮之前在同一任务里已经有过候选」的轮数**: 第一轮没有可比对象,
    算进分母就是把「不可能重复」记成「没重复」, 与 `_level_rate` 那条「没走到 ≠ 没通过」
    是同一个错的镜像 (doc 47 第一版把 `generate` 轮算进 optimize 分母也是它)。

    返回 `(重复轮数, 可比分母, 血缘里有没有签名)`。第三项是必须的: 签名缺失时要报
    「测不到」而**不是 0%** —— 这正是 doc 48 那一批 12 轮的处境, 键在血缘里算好了,
    却被报表自己的键白名单丢在落盘之前。
    """
    dup = at_risk = signed = 0
    for r in ok_runs:
        for i, h in enumerate(r.get("rounds") or []):
            if h.get("candidate_signature") is not None:
                signed += 1
            if i == 0:
                continue
            at_risk += 1
            if h.get("duplicate_of") is not None:
                dup += 1
    return dup, at_risk, signed > 0


def _ruler_stats(ok_runs, prefix: str, *, keys=None, lineage_key: str) -> dict:
    """一份「跨轮的尺」的报表统计, 按 `prefix` 出键名 (#87 第 2/3 片共一个实现)。

    判据全在 `measurement.cross_round_ruler` 那一个函数里, 这里只负责数运行。两半 (correctness
    的 `atol/rtol/ρ` 与 performance 的 `min_speedup/判据/绑定轴`) 各自调一次 ⇒ 五栏的算法
    只有一份。写成两份就是本仓 I22/H4 那一族: 「报表里那两个数」会慢慢长成两套口径。

    ⚠️ 分母直接取谓词自己的三态, 不在报表里另推「几轮才算可比」—— 另推一次就是第二套口径。
    `ruler_same is None` (报了尺但凑不出可比对) 的那些运行**两边都不进**: 既不记成「一致」
    也不记成「变过」 (`#53` 那一族: 缺证据不是证据)。
    """
    kw = {"keys": keys} if keys is not None else {}
    per_run = {str(r.get("case")): cross_round_ruler(r.get("rounds") or [],
                                                    lineage_key=lineage_key, **kw)
               for r in ok_runs}
    comparable = sorted(c for c, a in per_run.items() if a["ruler_same"] is not None)
    drift = sorted(c for c in comparable if per_run[c]["ruler_same"] is False)
    uncomparable = sorted(c for c, a in per_run.items()
                          if a["ruler_same"] is None and a["ruler_rounds_recorded"])
    return {
        f"{prefix}_measurable": any(a["ruler_rounds_recorded"] for a in per_run.values()),
        f"{prefix}_drift_runs": (len(drift), len(comparable)),
        f"{prefix}_drift_cases": [f"{c} → {dict(per_run[c]['ruler_changed'])}" for c in drift],
        f"{prefix}_uncomparable_runs": len(uncomparable),
        f"{prefix}_conflict_rounds": sum(a["ruler_rounds_self_conflict"]
                                         for a in per_run.values()),
        f"{prefix}_measurable_lineage_key": lineage_key,
    }


def _band_stats(ok_runs) -> dict:
    """「改善判据那条带宽跨轮是不是同一把」的报表统计 (#87 第 4 片, 只观测)。

    判据与归因全在 `measurement.cross_round_band` 一个函数里, 这里只数运行。**这一栏存在的
    理由**是那条门的归因有两种、而把它们折成一句「尺变了」会指错地方:
      · `tier_explained` —— 带宽随**测量档位**降级而变宽 (另一格 `measurement_validity`
        已经记着这件事, 这里只是别再把它报成规格漂移)
      · `spec_drift` —— **同一档位**下带宽换了值 ⇒ 表本身被换过 (现场: 续跑跨过了一次代码改动)
      · `unexplained` —— 缺档位或对不上表 ⇒ 两种都不说 (缺证据既不当通过也不当反证)
    分母 = 凑得出比较的运行 (≥2 轮用过带宽), `unmeasurable` 的那些运行两边都不进 ——
    与 `_ruler_stats` 同一姿势。
    """
    per_run = {str(r.get("case")): cross_round_band(r.get("rounds") or []) for r in ok_runs}
    comparable = sorted(c for c, a in per_run.items() if a["band_state"] != BAND_UNMEASURABLE)

    def _cases(state):
        # 每一句都走 `band_note` 那一个实现: 报表与 conductor 现场打印的那句必须同措辞,
        # 否则「同一件事在两个地方长得不一样」会让读者以为是两件事 (I22 那一族)。
        return sorted(f"{c} → {band_note(a)}" for c, a in per_run.items()
                      if a["band_state"] == state)

    return {
        "band_runs_with_band": sum(1 for a in per_run.values() if a["rounds_with_band"]),
        "band_comparable_runs": len(comparable),
        "band_unmeasurable_runs": len(per_run) - len(comparable),
        "band_same_runs": len(_cases(BAND_SAME)),
        "band_tier_explained_cases": _cases(BAND_TIER_EXPLAINED),
        "band_spec_drift_cases": _cases(BAND_SPEC_DRIFT),
        "band_unexplained_cases": _cases(BAND_UNEXPLAINED),
        "band_rounds_no_band": sum(a["rounds_no_band"] for a in per_run.values()),
        "band_rounds_no_key": sum(a["rounds_no_key"] for a in per_run.values()),
    }


def cost_from_metadata(metadata):
    """从 `artifact.metadata["llm_cost"]` 取出 (tokens, usd, calls)

    键名必须与 `LLMUsageTracker.summary()` 一致: 是 **`total_cost_usd`** 而不是
    `cost_usd`。第一版读错了键, 于是整场测量的成本一律报成 $0.0 —— 540114 tokens
    配一个 0 美元, 安静得像是对的。`tests/test_harness_pass_rate.py` 里有一条机械守卫
    直接对着真的 `summary()` 查这几个键在不在, 键名再漂移就会红。
    """
    c = (metadata or {}).get("llm_cost") or {}
    return (int(c.get("total_tokens") or 0),
            float(c.get("total_cost_usd") or 0.0),
            int(c.get("total_calls") or 0))


def sampling_from_metadata(metadata):
    """取本次运行「真发出去的采样配置」; 没有账本时返回 **None** 而不是空表 (#98)

    空表和 None 在两处读者眼里必须不同: 空表读起来像「这次跑没发任何请求」,
    None 读起来才是「这份报表的产出者没记这件事」。键名 `sampling` 的机械守卫在
    `tests/test_llm_sampling_knobs.py` —— 与上面那条 `total_cost_usd` 键名守卫同族。
    """
    c = (metadata or {}).get("llm_cost") or {}
    return c.get("sampling") or None


def sampling_from_runs(runs):
    """把各次运行账本里的采样配置并成一份读数条件 (#98 / doc 59 §四)

    三个入口分开计数, 折成一个数就会造出一张「看起来是全的」表:
      * `n_runs_without_ledger` —— 这次运行没产出账本 (崩溃 / 早退), 它的配置**无从知道**;
      * `n_records_without_sampling` —— 账本里有调用条目但没带那三个字段 (老账本);
      * 「没记」既不是「用了默认值」也不是「配置为 None 的一次取值」(#53 那一族)。
    合并按 (caller, model, temperature, max_tokens, enable_thinking) 组合相加 —— 不是
    各轴取并集, 那样会把「0.1 配 1024」与「0.3 配 40960」配出四种不存在的配置。
    """
    buckets: dict = {}
    with_ledger = 0
    without_ledger = 0
    records_without = 0
    for rec in runs:
        sampling = rec.get("llm_sampling")
        if not sampling:
            without_ledger += 1
            continue
        with_ledger += 1
        records_without += int(sampling.get("n_records_without_sampling") or 0)
        for caller, configs in (sampling.get("by_caller") or {}).items():
            bucket = buckets.setdefault(caller, {})
            for c in configs:
                key = (c.get("model"), c.get("temperature"),
                       c.get("max_tokens"), c.get("enable_thinking"))
                bucket[key] = bucket.get(key, 0) + int(c.get("calls") or 0)
    by_caller: dict = {}
    for caller, bucket in sorted(buckets.items()):
        rows = [{"model": k[0], "temperature": k[1], "max_tokens": k[2],
                 "enable_thinking": k[3], "calls": n} for k, n in bucket.items()]
        rows.sort(key=lambda c: (-c["calls"], str(c["temperature"]), str(c["max_tokens"])))
        by_caller[caller] = rows
    return {"n_runs_with_ledger": with_ledger, "n_runs_without_ledger": without_ledger,
            "n_records_without_sampling": records_without, "by_caller": by_caller}


def first_pass_round(rounds):
    """第一次 `all_passed` 的那一轮 → (轮号, 该轮的 action); 从没通过则 (None, None)

    **不管那轮的 action 是什么**。第一版把 `fix` 轮排除在外 (理由是「修好的不算一次
    干净的通过」), 真机第 4 次运行立刻证伪: layernorm#2 第 1 轮失败、Fixer 修完第 2 轮
    通过, 于是 `最终通过=True` 而 `rounds_to_pass=None` —— 这个样本被 rounds-to-pass 的
    分位数**静默丢掉**。指标悄悄少算样本比指标算错更难发现。
    「本来就对」还是「修好的」这个区分保留在返回的 action 里, 不去动分母。
    """
    for r in rounds or ():
        if r.get("all_passed"):
            return r.get("round"), r.get("action")
    return None, None


def _quantiles(values):
    if not values:
        return None
    s = sorted(values)

    def q(p):
        i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return round(s[i], 4)

    return {"n": len(s), "min": q(0.0), "p25": q(0.25), "median": q(0.5),
            "p75": q(0.75), "max": q(1.0)}


# ─── 判变化: 样本量 / Fisher / 门槛扫描 ─────────────────────────
# 这一层存在的理由是 doc 48 §六B 那次亲自算错预期的教训: 任务级 5/6 → 3/6
# 看起来「掉了一半」, 而 Fisher 精确双侧 p = 0.545 —— n=6 这个量级什么都判不出。
# 报表若只印两个百分数, 读者会自己做这个错误判决, 而错误判决长得和正确的一模一样。

def fisher_exact(a_pass, a_n, b_pass, b_n):
    """2×2 Fisher 精确双侧 p 值。纯 Python, 不依赖 scipy。

    表:
             通过   未通过
      臂 A   a     a_n-a
      臂 B   b     b_n-b

    双侧 p = 「所有出现概率 ≤ 观测表概率」的表的概率之和。
    doc 48 §六B 的已知答案: 5/6 vs 3/6 ⇒ p ≈ 0.545454… —— 测试里钉了这一对,
    公式方向写反或边界漏表都会撞红它。
    """
    if not (0 <= a_pass <= a_n and 0 <= b_pass <= b_n) or a_n <= 0 or b_n <= 0:
        raise ValueError(f"非法 2×2 表: ({a_pass}/{a_n}, {b_pass}/{b_n})")
    from math import comb

    total_n = a_n + b_n
    total_pass = a_pass + b_pass
    # A 臂通过数 k 的超几何边缘分布; k 的合法范围受两边样本数同时夹住
    lo = max(0, total_pass - b_n)
    hi = min(a_n, total_pass)

    def _p(k):
        # C(a_n, k) * C(b_n, total_pass-k) / C(total_n, total_pass)
        return (comb(a_n, k) * comb(b_n, total_pass - k)) / comb(total_n, total_pass)

    p_obs = _p(a_pass)
    # 浮点容差: 概率本身很小, 用相对容差把「同样极端」的表收进来, 否则 5/6 vs 3/6
    # 会因二进制表示把对称表丢掉半边, p 变成 0.27 而不是 0.545
    eps = max(1e-12, p_obs * 1e-9)
    return min(1.0, sum(_p(k) for k in range(lo, hi + 1) if _p(k) <= p_obs + eps))


def _z(p):
    """标准正态分位数。Python 3.8+ 自带, 不引入 scipy。"""
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def required_runs_per_arm(p_a, delta, alpha=0.05, power=0.8):
    """两比例检验: 要分辨 |p_b - p_a| ≥ delta, 每臂至少几次运行。

    正态近似 (与 doc 48「粗算 n≈20 每臂」同一族的估算, 只是把假设摊开):
        n ≈ (z_{1-α/2} + z_{power})² · (p_a(1-p_a) + p_b(1-p_b)) / delta²
    p_b 取 p_a + delta (改善侧) 与 p_a - delta (恶化侧) 里方差更大的那个 ——
    分辨「变差」和分辨「变好」需要的样本不一样多时, 用多的那个。

    返回整数 (向上取整)。给不出合法输入就 raise, **不猜一个数**。
    """
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError(f"alpha/power 必须在 (0,1): alpha={alpha} power={power}")
    if not 0 < p_a < 1:
        raise ValueError(f"p_a 必须在 (0,1): {p_a}")
    if delta <= 0:
        raise ValueError(f"delta 必须 > 0: {delta}")
    if p_a + delta > 1 and p_a - delta < 0:
        raise ValueError(f"delta={delta} 相对 p_a={p_a} 连一侧都够不着")
    z_sum = _z(1 - alpha / 2.0) + _z(power)
    candidates = []
    for p_b in (p_a + delta, p_a - delta):
        if 0 < p_b < 1:
            candidates.append(p_a * (1 - p_a) + p_b * (1 - p_b))
    if not candidates:
        raise ValueError(f"p_a={p_a}, delta={delta} 没有合法的 p_b")
    var = max(candidates)
    return int(np.ceil(z_sum * z_sum * var / (delta * delta)))


def min_detectable_delta(n_per_arm, p_a=0.5, alpha=0.05, power=0.8):
    """当前每臂样本量「最多只能分辨」多大的比例差 (最坏方差 p_a=0.5 时的下限)。

    从 `required_runs_per_arm` 的公式反解 Δ (取 p_b = p_a + Δ, 且 p_a=0.5 时
    p_a(1-p_a)+p_b(1-p_b) = 0.5 - Δ²):
        Δ = z_sum · sqrt(0.5 / (n + z_sum²))
    n 越小这个数越大。n=6 时约 53pp —— 正好解释了为什么 5/6→3/6 判不出东西。
    """
    if n_per_arm <= 0:
        raise ValueError(f"n_per_arm 必须 > 0: {n_per_arm}")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError(f"alpha/power 必须在 (0,1): alpha={alpha} power={power}")
    # 仅在 p_a=0.5 (最大方差) 处有闭式解; 其它 p_a 下需要迭代, 那会把「量程」
    # 伪装成一个依赖未知真比例的数。所以这里**只报最坏情况**, 它是乐观界限:
    # 真 p 离 0.5 越远, 同样 n 能分辨的 Δ 越小, 而我们事先并不知道真 p。
    z_sum = _z(1 - alpha / 2.0) + _z(power)
    return z_sum * np.sqrt(0.5 / (n_per_arm + z_sum * z_sum))


def _collect_unable(rounds, runs) -> list:
    """从逐轮 metrics 里捞「说不清」的信号 (设计 §17)。只搬事实, 不猜数字。"""
    from operator_agent.confirm import UnableToEstablish
    u = UnableToEstablish()
    for r in rounds:
        m = r if isinstance(r, dict) else {}
        # 轮级 metrics 在 record_rounds 原样搬进 rounds 的各键里
        u.extend_metrics({
            "measurement_can_gate": m.get("measurement_can_gate"),
            "measurement_validity": m.get("measurement_validity"),
            "measurement_validity_reasons": m.get("measurement_validity_reasons") or [],
            "measurement_l3_pass_refused": m.get("measurement_l3_pass_refused"),
            "ruler_broken": m.get("ruler_broken"),
            "ruler_reason": m.get("ruler_reason"),
            "contract_hash_mismatch": m.get("contract_hash_mismatch"),
            "benchmark_contaminated": m.get("benchmark_contaminated"),
        })
    # 崩溃运行: 没有任何读数, 也是「无法确立」的一种
    for r in runs:
        if r.get("error"):
            u.add(f"运行 {r.get('case', '?')} 崩溃, 无任何读数",
                  str(r.get("error"))[:120], kind="measurement")
    return [{"claim": i.claim, "reason": i.reason, "kind": i.kind,
             "evidence": i.evidence} for i in u.items]


def threshold_curve(runs, thresholds=(0.9, 1.0, 1.1)):
    """同一批候选在不同 `min_speedup` 门槛下的重新计分 (doc 48 §6.2 ①)。

    两层分开给, 分母不同不合并:
      · 轮级: 有 speedup 读数的轮里, speedup ≥ t 的比例 (功能过的轮才有读数可言,
        但这里不过滤 L2 —— 过滤会让「差 2~3%」那批消失, 而那正是曲线要看的);
      · 任务级: 完成的运行里, 「存在至少一轮 speedup ≥ t 且该轮 L2 过」的比例。
        这是**假设那个门槛当时就生效**时, 这批候选能不能交差。

    ⚠️ 这不是「把门槛改低再跑一遍」: 搜索是被当时的门槛塑形的, 换门槛会生成另一批
    候选。曲线只能回答「这批货在别的门槛下卖不卖得掉」, 不能回答「改低门槛流水线
    会变强吗」。这句话必须跟曲线一起出现在报表里。
    """
    thr = tuple(float(t) for t in thresholds)
    ok_runs = [r for r in runs if r.get("ok")]
    speedups = []
    for r in ok_runs:
        for h in r.get("rounds") or []:
            sp = h.get("speedup")
            if isinstance(sp, (int, float)) and not isinstance(sp, bool):
                speedups.append(float(sp))
    round_rows = []
    task_rows = []
    for t in thr:
        r_num = sum(1 for s in speedups if s >= t)
        r_den = len(speedups)
        t_num = 0
        for r in ok_runs:
            for h in r.get("rounds") or []:
                sp = h.get("speedup")
                lv = h.get("levels_passed") or {}
                if (isinstance(sp, (int, float)) and not isinstance(sp, bool)
                        and sp >= t and lv.get("L2")):
                    t_num += 1
                    break
        t_den = len(ok_runs)
        round_rows.append({"threshold": t, "passed": r_num, "denom": r_den,
                           "pct": _pct(r_num, r_den)})
        task_rows.append({"threshold": t, "passed": t_num, "denom": t_den,
                          "pct": _pct(t_num, t_den)})
    return {"thresholds": list(thr), "rounds": round_rows, "tasks": task_rows,
            "caveat": "同一批已生成候选的重新计分, 不是改门槛重跑 —— 搜索行为被当时的门槛塑形"}


def pair_cases(runs_a, runs_b):
    """按 `case` 名把两批运行配成 `(名字, A 侧结果, B 侧结果)`, 配不上的都是 `None`。

    三种「配不上」分开记 (`reasons`), 因为报表要说得出**为什么**这条没有结论:
      · `absent` —— 这一侧的批次里根本没有这条 case (换过 case 清单)
      · `crashed` —— 这一侧跑了但 `ok=False` (崩溃的那次**没有**任务级结果)
      · `duplicate` —— 同一批里这个名字出现两次 (不猜该用哪一次; 真机重跑同名多次是
        `--repeats` 之外的形状, 出现了就说明记录方式与 `case` 名的约定打架)

    ⚠️ 三种都不许折算成 `False`: 那正是「缺证据当反证」(#53 那一族)。也不许折算成
    「两侧都一样」: 那会静默缩小分母。
    """
    def _index(runs):
        seen = {}
        for r in runs or []:
            name = str(r.get("case"))
            if name in seen:
                seen[name] = "duplicate"
            elif not r.get("ok"):
                seen[name] = None
            else:
                seen[name] = bool(r.get("final_all_passed"))
        return seen

    ia, ib = _index(runs_a), _index(runs_b)
    reasons = {}
    cases = []
    for name in sorted(set(ia) | set(ib)):
        va = ia.get(name, "absent")
        vb = ib.get(name, "absent")
        for side, val, idx in (("A", va, ia), ("B", vb, ib)):
            if val == "absent":
                reasons.setdefault(name, []).append(f"{side} 侧没有这条 case")
            elif val == "duplicate":
                reasons.setdefault(name, []).append(f"{side} 侧同名出现多次")
            elif val is None:
                reasons.setdefault(name, []).append(f"{side} 侧那次运行崩溃了")
        clean_a = va if isinstance(va, bool) else None
        clean_b = vb if isinstance(vb, bool) else None
        cases.append((name, clean_a, clean_b))
    return cases, reasons


def compare_runs(runs_a, runs_b, meta_a=None, meta_b=None, target_delta=0.30):
    """两份运行记录的 A/B 判决: 这次改动「测得出变化」还是「本来测不出」。

    判决词只有四个, 而且语义是互斥的:
      · `improved` / `regressed` —— Fisher 双侧 p < α, 且方向由成功率差的符号定;
      · `no_detected_difference` —— p ≥ α **且** 当前样本本来分辨得出 target_delta;
      · `underpowered` —— p ≥ α 且样本量连 target_delta 都分辨不出。这时正确的读法是
        「不知道」, 不是「没变化」。doc 48 §六B 那次 5/6→3/6 正是这一档。

    主判决看**任务级成功率** (运行是独立样本)。L3 轮级作为辅助一起算, 但注意
    轮嵌在运行里、并不独立 —— 报表只把它当地图, 不当第二张判决书。
    """
    ok_a = [r for r in runs_a if r.get("ok")]
    ok_b = [r for r in runs_b if r.get("ok")]
    pass_a = sum(1 for r in ok_a if r.get("final_all_passed"))
    pass_b = sum(1 for r in ok_b if r.get("final_all_passed"))
    n_a, n_b = len(ok_a), len(ok_b)

    def _l3(runs):
        num = den = 0
        for r in runs:
            for h in r.get("rounds") or []:
                lv = h.get("levels_passed") or {}
                if "L3" in lv:
                    den += 1
                    if lv.get("L3"):
                        num += 1
        return num, den

    l3_a, l3_na = _l3(ok_a)
    l3_b, l3_nb = _l3(ok_b)

    # 样本不足时 Fisher 仍算 (它是对的), 但**判决词必须让路**
    denom_ok = n_a >= MIN_DENOM and n_b >= MIN_DENOM
    p_task = fisher_exact(pass_a, n_a, pass_b, n_b) if n_a and n_b else None
    p_l3 = fisher_exact(l3_a, l3_na, l3_b, l3_nb) if l3_na and l3_nb else None

    rate_a = pass_a / n_a if n_a else None
    rate_b = pass_b / n_b if n_b else None
    delta_obs = (rate_b - rate_a) if (rate_a is not None and rate_b is not None) else None

    # 量程: 这批样本最多分辨多大的差。p_a 取基线侧的观测率 (夹到 (0,1)),
    # 没有基线时用 0.5 —— 最坏方差, 于是量程偏保守 (说「能分辨的差」偏大 ⇒
    # 更容易把 `no_detected_difference` 降级成 `underpowered`, 方向是诚实的)。
    p_scale = rate_a if (rate_a is not None and 0 < rate_a < 1) else 0.5
    n_min = min(n_a, n_b) if (n_a and n_b) else 0
    detectable = min_detectable_delta(n_min, p_a=p_scale) if n_min else None
    required = None
    try:
        required = required_runs_per_arm(max(0.05, min(0.95, p_scale)), target_delta)
    except ValueError:
        required = None

    alpha = 0.05
    if not denom_ok:
        verdict = "underpowered"
        reason = (f"每臂完成的运行不足 MIN_DENOM={MIN_DENOM} "
                  f"(A={n_a}, B={n_b}) —— 百分数都不该出, 更谈不上判变化")
    elif p_task is None:
        verdict = "underpowered"
        reason = "有一臂没有可用的成功率分母"
    elif p_task < alpha:
        if delta_obs is not None and delta_obs > 0:
            verdict = "improved"
        elif delta_obs is not None and delta_obs < 0:
            verdict = "regressed"
        else:
            verdict = "no_detected_difference"
        reason = f"任务级 Fisher 双侧 p={p_task:.4g} < {alpha}, Δ={delta_obs:+.3f}"
    else:
        if detectable is not None and target_delta is not None and detectable <= target_delta:
            verdict = "no_detected_difference"
            reason = (f"p={p_task:.4g} ≥ {alpha}, 而样本量分辨得出 "
                      f"{target_delta:.0%} 量级的差 (可分辨下限 ≈{detectable:.1%})")
        else:
            verdict = "underpowered"
            reason = (f"p={p_task:.4g} ≥ {alpha}, 且样本量连 {target_delta:.0%} 的差都"
                      f"分辨不出 (可分辨下限 ≈{(detectable or 0):.1%}) —— "
                      f"正确的读法是「不知道」, 不是「没变化」")

    # 逐 case 那一层 (#103 第 1 片)。**算在同一份存量证据上、什么都不重新执行** —— 外部那句
    # "Nothing is re-executed: both rules are applied to the same stored evidence" (行 1050)
    # 是这片的做法: 两条规则读同一批 `runs`, 于是分歧是精确的而不是两次测量的差。
    # 它不参与上面那个判决: 聚合门 (Fisher) 仍是唯一的判决书, 这一栏只补「谁变坏了」这个名字。
    cases, case_reasons = pair_cases(runs_a, runs_b)
    attributable = attributable_evidence(cases)
    attributable["unmatched_reasons"] = case_reasons

    return {
        "verdict": verdict,
        "attributable": attributable,
        "reason": reason,
        "alpha": alpha,
        "target_delta": target_delta,
        "task_success": {
            "a": (pass_a, n_a), "b": (pass_b, n_b),
            "rate_a": rate_a, "rate_b": rate_b, "delta": delta_obs,
            "p_value": p_task,
        },
        "l3": {
            "a": (l3_a, l3_na), "b": (l3_b, l3_nb),
            "p_value": p_l3,
            "note": "轮嵌在运行里, 不独立 —— 只作地图, 不作第二张判决书",
        },
        "power": {
            "n_a": n_a, "n_b": n_b, "n_min": n_min,
            "detectable_delta": detectable,
            "required_per_arm": required,
            "target_delta": target_delta,
        },
        "speedup": {
            "a": _quantiles([h["speedup"] for r in ok_a for h in (r.get("rounds") or [])
                             if isinstance(h.get("speedup"), (int, float))
                             and not isinstance(h.get("speedup"), bool)]),
            "b": _quantiles([h["speedup"] for r in ok_b for h in (r.get("rounds") or [])
                             if isinstance(h.get("speedup"), (int, float))
                             and not isinstance(h.get("speedup"), bool)]),
            "note": "描述性分布, 不做检验: 轮内相关, 不满足独立样本假设",
        },
        "meta": {"a": meta_a or {}, "b": meta_b or {}},
    }


def _knobs_cell(value):
    """knobs 的一格印成什么: 没记 ⇒ `— 没记`, 而不是空串 (空串读起来像「两边都没值」)"""
    if value is None:
        return "— 没记"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def format_knobs_pair(meta_a, meta_b):
    """把两份报表的 knobs 并排印, 不同的那几项标 ⚠️ (#98 / doc 59 §四)

    这一段的存在理由不是「好看」。报表落盘时自己写着「引用这些数字时必须带上 `git_head` 与
    `knobs`」, 而 `--compare` 这条路径此前只印 `git_head` —— 那句警告在 A/B 现场**没有读者**,
    是「算了没人读」的一个实例。两边读数条件不一致时, 那个 Δ 就不是同一次尺下的差。
    """
    ka = (meta_a or {}).get("knobs") or {}
    kb = (meta_b or {}).get("knobs") or {}
    if not ka and not kb:
        return ["  读数条件 (knobs): 两份报表都没有 —— 这份对比不判「条件相同」"]
    lines = ["  读数条件 (knobs):"]
    n_diff = 0
    for key in sorted(set(ka) | set(kb)):
        va, vb = _knobs_cell(ka.get(key)), _knobs_cell(kb.get(key))
        if va == vb:
            lines.append(f"     = {key}: {va}")
        else:
            n_diff += 1
            lines.append(f"    ⚠️ {key}: A {va}  →  B {vb}")
    if n_diff:
        # ⚠️ 这句里**不许**出现 `— 没记` 那个串本身: 变异标定 H3 第一次就是这么逃掉的 ——
        # 断言 `assertIn("— 没记", ...)` 被解释文案自己满足了, 于是「格子印成空串」这个缺陷
        # 在测试面前完全隐形 (CHECKLOG I21 那一族: 文案与判据共用一个串, 读者就分不清是谁)。
        lines.append(f"    ⚠️ {n_diff} 项读数条件不一致 ⇒ 上面那个 Δ 混进了配置差, "
                     f"不能只当机制差读 (一侧没记也算不一致)")
    return lines


def format_comparison(cmp):
    lines = ["", "=" * 62, "A/B 判决 (能不能判出变化)", "=" * 62]
    ma, mb = cmp.get("meta", {}).get("a") or {}, cmp.get("meta", {}).get("b") or {}
    if ma.get("git_head") or mb.get("git_head"):
        lines.append(f"  A: {ma.get('git_head', '?')}  →  B: {mb.get('git_head', '?')}")
    lines.extend(format_knobs_pair(ma, mb))
    ts = cmp["task_success"]
    a_ok, a_n = ts["a"]
    b_ok, b_n = ts["b"]

    def _rate(n, d):
        return f"{n}/{d}" + (f" = {n / d * 100:.1f}%" if d else "")

    lines.append(f"  任务级成功: A {_rate(a_ok, a_n)}  →  B {_rate(b_ok, b_n)}")
    if ts["p_value"] is not None:
        lines.append(f"  Fisher 精确双侧 p = {ts['p_value']:.4g}"
                     + (f"   Δ = {ts['delta']:+.3f}" if ts["delta"] is not None else ""))
    l3 = cmp["l3"]
    lines.append(f"  L3 轮级 (地图): A {_rate(*l3['a'])}  →  B {_rate(*l3['b'])}"
                 + (f"  p = {l3['p_value']:.4g}" if l3["p_value"] is not None else ""))
    pw = cmp["power"]
    if pw.get("detectable_delta") is not None:
        lines.append(f"  量程: 每臂 n≈{pw['n_min']} 时最多分辨 ≈{pw['detectable_delta']:.1%} 的差"
                     + (f" (要分辨 {pw['target_delta']:.0%} 约需 n={pw['required_per_arm']}/臂)"
                        if pw.get("required_per_arm") else ""))
    sa, sb = cmp["speedup"]["a"], cmp["speedup"]["b"]
    if sa and sb:
        lines.append(f"  加速比中位: A {sa['median']}x (n={sa['n']})  →  "
                     f"B {sb['median']}x (n={sb['n']})   [描述性, 未检验]")
    label = {
        "improved": "✅ 测出了改善",
        "regressed": "❌ 测出了回归",
        "no_detected_difference": "➖ 没测出差异 (样本量本来分辨得出目标差幅)",
        "underpowered": "⚠️  判不出 (样本量不够) —— 不是「没变化」",
    }[cmp["verdict"]]
    lines.append(f"\n判决: {label}")
    lines.append(f"  {cmp['reason']}")

    # 【逐 case 的得与失】(#103 第 1 片, 只观测)。外面那行判决读的是**一个聚合数**,
    # 而 HarnessLens 行 99 点名的病正是 "aggregate scores to obscure specific regressions"。
    # 我方那份存量对照比那句还糟一档: 5/6 → 3/6 那一对比, 统计门印「判不出 —— 不知道」,
    # 而逐 case 那一层其实是 1 条修好 + 3 条变坏 (doc 60 §六)。缺的不是更严的阈, 是**点名**。
    att = cmp.get("attributable") or {}
    cnt = att.get("counts") or {}
    unmatched = att.get("unmatched_cases") or []
    lines.append("\n【逐 case 的得与失】(两条规则算在同一份存量证据上, 什么都不重新执行)")
    lines.append(f"  配上的 case {att.get('matched', 0)} · 修好 {cnt.get('recovered', 0)} · "
                 f"变坏 {cnt.get('regressed', 0)} · 两侧都过 {cnt.get('stable_pass', 0)} · "
                 f"两侧都不过 {cnt.get('stable_fail', 0)}"
                 + (f" · 没有可用读数 {len(unmatched)}" if unmatched else ""))
    if not att.get("matched"):
        lines.append("  测不到: 一条都没配上 ⇒ 下面三条结论都是 `None`, "
                     "**这不是「没有回归」也不是「都变坏了」**")
    else:
        lines.append(f"  配对口径的通过数: {att['pass_a']} → {att['pass_b']} "
                     f"(Δ={att['delta']:+d}, 只在配上的 case 上数)")
        if att["regressed_cases"]:
            lines.append("  ❌ 变坏: " + ", ".join(att["regressed_cases"]))
        if att["recovered_cases"]:
            lines.append("  ✅ 修好: " + ", ".join(att["recovered_cases"]))
    for name in unmatched:
        why = "; ".join((att.get("unmatched_reasons") or {}).get(name, []))
        lines.append(f"  ⚪ 无结论: {name} —— {why or '这一侧没有可用读数'}")
    if att.get("matched"):
        def _word(v):
            return "接受" if v is True else ("不接受" if v is False else "无从判定")
        _dis = att["disagreement"]
        _ev = {"recovery": f"修好了 {len(att['recovered_cases'])} 条 (从不过到过)"}.get(
            att.get("positive_evidence"))
        lines.append(f"  两条规则: metric-only (Δ>0 就接受) {_word(att['accepted_metric_only'])}"
                     f" · 可归因门 (要有正证据且零回归) {_word(att['accepted_gate'])}"
                     f" · 分歧 {'有' if _dis else ('无' if _dis is False else '无从判定')}"
                     + (" ← 聚合数与具体回归打架的那一次" if _dis is True else ""))
        _no_ev = "没有 (只有 preservation 不算正证据 —— 行 880 那句 preservation alone is insufficient)"
        lines.append(f"  正证据出处: {_ev or _no_ev}")
    lines.append("  ⚠️ 只观测不改判决: 上面那行「判决」仍然只由 Fisher 那道门给, 这一栏不参与"
                 "任何一次决定 (#103 第 1 片, 依据 arXiv:2608.27311v1 行 878-880)")
    return "\n".join(lines)


def aggregate(runs):
    """把逐次运行的原始记录聚合成报表。返回的每个百分数都**带分母**。"""
    ok_runs = [r for r in runs if r.get("ok")]
    crashed = [r for r in runs if not r.get("ok")]
    rounds = [x for r in ok_runs for x in (r.get("rounds") or [])]

    l1 = _level_rate(rounds, "L1")
    l2 = _level_rate(rounds, "L2")
    l3 = _level_rate(rounds, "L3")

    passed_runs = [r for r in ok_runs if r.get("final_all_passed")]
    # 双记分口径 (#77): 同一批运行、同一个分母, 只差最后那道真机门。
    local_runs = [r for r in ok_runs if last_round_correctness_passed(r)]
    # 形状上 `passed_runs` 应当是 `local_runs` 的子集 (`final_all_passed` 含 L1/L2)。
    # 反过来出现「真机说过了、本地口径说没过」= 血缘与判决不同源, 那种运行**不能**被
    # 折进一个百分数里 —— 单独列出来, 让读报表的人去查那一份 result.json。
    dual_scoring_conflicts = sorted(
        str(r.get("case")) for r in passed_runs if not last_round_correctness_passed(r))
    l3_only_blockers = sorted({str(r.get("case")) for r in local_runs
                               if not r.get("final_all_passed")})
    rtp = [r["rounds_to_pass"] for r in passed_runs if r.get("rounds_to_pass")]

    tiers = {}
    for r in rounds:
        v = r.get("measurement_validity")
        if v is not None:
            tiers[v] = tiers.get(v, 0) + 1
    graded = [r for r in rounds if r.get("measurement_validity") is not None]
    decision_grade = [r for r in graded
                      if r.get("measurement_validity") == Validity.VALID.value]

    # 执行侧「没听到回答」那一格 (#102 第 1 片, **只观测**): 逐格数, 不折成一个数 ——
    # 「报了 answered」「报了 outcome_unknown」「这一轮压根没报」是三件事, 折起来就又是
    # 把没检查当检查过 (本仓那一族)。
    transport_counts = {}
    for r in rounds:
        t = r.get(TRANSPORT_KEY)
        if t is not None:
            transport_counts[t] = transport_counts.get(t, 0) + 1

    # 加速比分两套口径给, 不合并: 混在一起就是把「不可信的读数」当成成绩
    sp_all = [r["speedup"] for r in rounds if isinstance(r.get("speedup"), (int, float))]
    sp_grade = [r["speedup"] for r in decision_grade
                if isinstance(r.get("speedup"), (int, float))]

    # 入库率与拒收名单必须同分母。`_save_to_db()` 只在 `artifact.all_passed` 那一支被调
    # (`conductor.py:1112` 明写), 所以**未通过的运行在生产里根本不会被问这道门** —— 把它们
    # 的判定混进「拒收」名单, 就是让读者以为入库门拒了一些好核, 而它其实一条都没拒
    # (2026-09-29 那份报表印了「入库 3/3」紧跟三条拒收, 三条全来自未通过的运行)。
    admitted = [r for r in passed_runs if r.get("admitted")]
    # 假想事件单独放: 它有用的场合是「如果这些运行当时通过了, 门会不会放它进去」,
    # 那是一个关于门的问题, 不是一个关于本轮成果的问题。两者必须分栏。
    counterfactual = sorted({r["admit_reason"] for r in ok_runs
                             if r.get("admit_reason") and not r.get("admitted")
                             and not r.get("final_all_passed")})
    # 名单是**去重后的理由**, 不是运行数 —— 三次同样的拒收只印一行。所以两个数都要给,
    # 否则「另 3 条」会被读成「还有 3 次被拒」, 而它可能只是 1 种理由的 3 次重复。
    counterfactual_runs = sum(1 for r in ok_runs
                              if r.get("admit_reason") and not r.get("admitted")
                              and not r.get("final_all_passed"))

    dup, at_risk, has_sig = _duplicate_rate(ok_runs)
    # 跨轮的尺是不是同一把 (#87 第 2 片 = correctness 那一半, 第 3 片 = performance 那一半)。
    # 判据与三态都在 `measurement.cross_round_ruler`, 这两栏是它的**第二、第三个读者** ——
    # 前两处是 conductor 每轮那两句 `[Ruler]`。同一个 `_ruler_stats` 出两半的数, 所以
    # 「当场打印的那一行」与「报表里的这个数」不会长成两套口径 (I22/H4 那一族)。
    ruler_stats = _ruler_stats(ok_runs, "ruler", lineage_key=LINEAGE_RULER_KEY)
    perf_ruler_stats = _ruler_stats(ok_runs, "perf_ruler",
                                    keys=PERF_GATE_PARAM_KEYS,
                                    lineage_key=LINEAGE_PERF_RULER_KEY)
    # 改善判据那条带宽 (#87 第 4 片)。它的**第二个**读者 (第一个是 conductor 那句 `[Band]`,
    # 而那句只在归因得出「规格漂移」或「无从归因」时才印) —— 判据与归因都在
    # `measurement.cross_round_band` 一个实现里, 这里只数运行。
    band_stats = _band_stats(ok_runs)
    via = {}
    for r in passed_runs:
        k = r.get("passed_via") or "(未记)"
        via[k] = via.get(k, 0) + 1

    return {
        "runs": len(runs),
        "runs_ok": len(ok_runs),
        "runs_crashed": len(crashed),
        "crash_reasons": [r.get("error") for r in crashed],
        "rounds": len(rounds),
        "transport_counts": dict(transport_counts),
        "transport_reported": sum(transport_counts.values()),
        "level_rates": {"L1": l1, "L2": l2, "L3": l3},
        "level_pcts": {"L1": _pct(*l1), "L2": _pct(*l2), "L3": _pct(*l3)},
        "task_success": (len(passed_runs), len(ok_runs)),
        "task_success_pct": _pct(len(passed_runs), len(ok_runs)),
        "task_success_local": (len(local_runs), len(ok_runs)),
        "task_success_local_pct": _pct(len(local_runs), len(ok_runs)),
        "l3_only_blockers": l3_only_blockers,
        "dual_scoring_conflicts": dual_scoring_conflicts,
        "rounds_to_pass": _quantiles(rtp),
        "passed_via": via,
        "validity_tiers": tiers,
        "decision_grade": (len(decision_grade), len(graded)),
        "decision_grade_pct": _pct(len(decision_grade), len(graded)),
        "speedup_all": _quantiles(sp_all),
        "speedup_decision_grade": _quantiles(sp_grade),
        "duplicate_rounds": (dup, at_risk),
        "duplicate_measurable": has_sig,
        "duplicate_pct": _pct(dup, at_risk) if has_sig else None,
        # ── 跨轮的尺 (#87 第 2 片 correctness / 第 3 片 performance) ──
        # 两半各五栏, 都由 `_ruler_stats` 那一个实现产出。L2 那半的键名与本片之前**逐字
        # 相同** —— 报表消费方 (与钉它们的用例) 读的是这些名字, 改名字不是自由重构 (I20)。
        **ruler_stats,
        **perf_ruler_stats,
        **band_stats,
        "admitted": (len(admitted), len(passed_runs)),
        "admitted_pct": _pct(len(admitted), len(passed_runs)),
        "admit_rejections": sorted({r["admit_reason"] for r in passed_runs
                                    if r.get("admit_reason") and not r.get("admitted")}),
        "gate_never_asked": sorted({r["admit_reason"] for r in passed_runs
                                    if not r.get("admit_reason")}),
        "counterfactual_rejections": counterfactual,
        "counterfactual_runs": counterfactual_runs,
        "admit_rejection_runs": sum(1 for r in passed_runs
                                    if r.get("admit_reason") and not r.get("admitted")),
        "cost": {
            "tokens": sum(int(r.get("tokens") or 0) for r in ok_runs),
            "calls": sum(int(r.get("llm_calls") or 0) for r in ok_runs),
            "usd": round(sum(float(r.get("cost_usd") or 0.0) for r in ok_runs), 4),
            "wall_s": round(sum(float(r.get("wall_s") or 0.0) for r in ok_runs), 1),
        },
        "threshold_curve": threshold_curve(ok_runs),
        "unable_to_establish": _collect_unable(rounds, runs),
        "power": {
            "n_a": len(ok_runs),
            "n_min": len(ok_runs),
            "detectable_delta": (
                min_detectable_delta(len(ok_runs)) if ok_runs else None),
            "required_per_arm": required_runs_per_arm(0.5, 0.30),
            "target_delta": 0.30,
            "note": "量程按最坏方差 p=0.5 报, 是乐观界限: 真 p 离 0.5 越远越容易分辨",
        },
        "denominator_ok": min(len(ok_runs), len(rounds)) >= MIN_DENOM,
    }


def device_record(backend, env=None) -> dict:
    """报表里的「跑在哪张卡」必须来自**执行侧**, 不是本地环境变量 (#52)

    旧写法是 `"gpu": os.environ.get("OPERATOR_AGENT_GPU", "(未固定)")` —— 读的是用户
    **想**的那张卡。而 `SSHExecutor` 从不读那个变量: 它在本进程里自动挑一张最闲的卡
    (`_pinned_gpu()`), 再把**自己选中的那个号**作为前缀下发给远端。于是第一份基线里
    「GPU 2 起跑」可能是报表值而不是事实值 (实际是 GPU 7) —— 记录的字段与发生的事不同源,
    是本项目那一族缺陷的「读了, 但读错了对象」变体。

    三态要**分开表达**, 不能都塌成 `(未固定)`:
      * 后端不具备 `GPU_PINNING` (CPU / 模拟器 / HTTP) → 没有「固定一张卡」这件事;
      * 具备但没解析出卡号 → 本来该固定, 这一轮其实是远端自动选的;
      * 具备且有号 → 就是它。
    `conflicts` 只在本地设了那个变量时才可能为真: 设了就说明用户想钉一张卡, 而这条路
    今天做不到, 所以必须响一行, 不能静默换卡。
    """
    env = os.environ if env is None else env
    requested = (env.get("OPERATOR_AGENT_GPU") or "").strip()
    actual = backend.selected_gpu
    can_pin = Cap.GPU_PINNING in backend.capabilities
    if not can_pin:
        state = f"({backend.label} 不具备 {Cap.GPU_PINNING.value}, 没有固定卡号这件事)"
    elif actual is None:
        state = "(声明可固定, 但这一轮没解析出卡号 ⇒ 由远端自动选择)"
    else:
        state = f"GPU {actual}"
    conflicts = bool(requested) and (str(actual) if actual is not None else "") != requested
    return {"state": state, "gpu": actual, "can_pin": can_pin,
            "requested_env": requested or None, "conflicts": conflicts,
            "source": "backend.selected_gpu"}


def format_report(agg):
    lines = ["", "=" * 62, "端到端成功率报表", "=" * 62]
    lines.append(f"运行 {agg['runs_ok']}/{agg['runs']} 次完成"
                 + (f" (崩 {agg['runs_crashed']})" if agg["runs_crashed"] else "")
                 + f" · 共 {agg['rounds']} 轮")
    if not agg["denominator_ok"]:
        lines.append(f"\n❌ 分母不足 (运行 {agg['runs_ok']} / 轮 {agg['rounds']}, "
                     f"下限 {MIN_DENOM}) —— 不出百分数。")
        lines.append("   0/1 与 0/12 都会打印成 0.0%, 那不是测量值。先让循环真跑起来。")
        return "\n".join(lines)

    lines.append("\n【分级通过率】分母 = 走到该级的轮数 (不是总轮数)")
    for lv in ("L1", "L2", "L3"):
        p, d = agg["level_rates"][lv]
        lines.append(f"  {lv}: {p}/{d}" + (f" = {agg['level_pcts'][lv]}%" if d else " (无)"))

    ts, td = agg["task_success"]
    lines.append(f"\n【任务级成功】{ts}/{td} = {agg['task_success_pct']}%  (最终 all_passed)")
    # 双记分口径 (#77, 外部一手 arXiv:2609.26777v1 行 99-101 的形状): 同一批运行、同一个分母,
    # 只差最后那道真机门。只报上面那个数, 「本地做不到」与「做到了但过不了性能门」就分不开。
    tl, tld = agg["task_success_local"]
    lines.append(f"【同一批, 摘掉真机 L3】{tl}/{tld} = {agg['task_success_local_pct']}%"
                 f"  ← 差 {tl - ts} 次是「本地已经过了、真机那道门判死」")
    if agg["l3_only_blockers"]:
        lines.append("  只被 L3 拦住的运行: " + ", ".join(agg["l3_only_blockers"]))
    if agg["dual_scoring_conflicts"]:
        lines.append("  ❌ 两套口径互相矛盾 (all_passed 为真而最后一轮的 L1/L2 读数不支持): "
                     + ", ".join(agg["dual_scoring_conflicts"])
                     + " —— 那是血缘与判决不同源, 别把这两个数当成绩读")
    if agg["rounds_to_pass"]:
        q = agg["rounds_to_pass"]
        lines.append(f"【rounds-to-pass】median {q['median']} · p25 {q['p25']} · "
                     f"p75 {q['p75']} · max {q['max']} (n={q['n']})")
    if agg.get("passed_via"):
        # 「靠哪一轮过的」不是花边: doc 48 的读数 ④ (33% 的运行首轮死在 L1、被 Fixer
        # 一轮救回) 只能从这里读出来, 手算的话下一次就没人算了。
        lines.append(f"  通过方式: {agg['passed_via']}")

    d, dt = agg["duplicate_rounds"]
    lines.append("\n【候选重复率】分母 = 本轮之前同一任务里已有候选的轮数")
    if not agg["duplicate_measurable"]:
        lines.append(f"  测不到: 血缘里没有 candidate_signature (可比轮 {dt}) —— 这不是 0%")
    else:
        lines.append(f"  {d}/{dt}" + (f" = {agg['duplicate_pct']}%" if dt else " (无可比轮)"))

    dg, dgt = agg["decision_grade"]
    lines.append(f"\n【读数可信度】decision-grade (valid 档) {dg}/{dgt}"
                 + (f" = {agg['decision_grade_pct']}%" if dgt else ""))
    lines.append(f"  档位分布: {agg['validity_tiers'] or '无 L3 读数'}")
    lines.append("  ⚠️ 通过率必须与这一行一起读: 在 uninformative 档上「通过」的核, "
                 "它的加速比没有资格驱动任何闭环决策。")

    # 两半各一栏 (#87 第 2 片 = correctness, 第 3 片 = performance)。**一个循环出两栏**:
    # 两半的算法逐字相同 (五栏都来自 `_ruler_stats` 那一个实现), 抄两份就会长成两套口径。
    # 标题与那句解释按半各给 —— 它们说的是不同的键, 混成一句就没人知道这一栏在问什么。
    for _title, _prefix, _conflict_note in (
            ("各轮 L2 报的 atol/rtol/required_matched_ratio 是不是同一把", "ruler",
             "  (= metrics.gate_params_conflict 的程序侧读者, #87 第 1 片那条)"),
            ("各轮 L3 报的 min_speedup/判的是哪个量/绑定轴 是不是同一把", "perf_ruler",
             "  (性能那半结构上到不了这一态: L3 只拿第一组输入做基准, 没有跨组可打架)")):
        _rk = f"{_prefix}_drift_runs"
        dnum, dden = agg[_rk]
        lines.append(f"\n【跨轮的尺 · {_title}】"
                     " (arXiv:2609.35140v1 行 150 那条 𝒮 immutable —— 那句说的是 correctness "
                     "**和** performance 两半)")
        if not agg[f"{_prefix}_measurable"]:
            # 「测不到」而不是 0 —— 与上面【候选重复率】那一栏同一个位置、同一条纪律
            lines.append(f"  测不到: 血缘里没有 {agg[f'{_prefix}_measurable_lineage_key']} 这项"
                         f" (共 {agg['rounds']} 轮)"
                         " —— 那多半是这批运行跑在写下这项之前, **不是**「查过了没漂移」")
            continue
        lines.append(f"  可比运行 (≥2 轮报了尺) {dden} · 尺变过 {dnum}"
                     + (f" ← 明细: {', '.join(agg[f'{_prefix}_drift_cases'])}"
                        if agg[f"{_prefix}_drift_cases"] else ""))
        if agg[f"{_prefix}_uncomparable_runs"]:
            lines.append(f"  另有 {agg[f'{_prefix}_uncomparable_runs']} 次运行报了尺但凑不出可比对"
                         " —— 这一栏既不记成「一致」也不记成「变过」")
        lines.append(f"  某一轮内部就不是同一把尺的轮数: {agg[f'{_prefix}_conflict_rounds']}"
                     + _conflict_note)
        lines.append("  ⚠️ 只观测不判决: 跨轮换尺今天没有阈可判, 现场通常是远端脚本中途换过版本")

    # 执行侧「没听到回答」那一格 (#102 第 1 片)。**三格并排、分母各带各的** —— 把「报了 answered」
    # 与「这一轮压根没报」折成一格, 就是把没检查当检查过 (本仓那一族, 与【候选重复率】同形)。
    _tc = agg["transport_counts"]
    _unknown = _tc.get(TRANSPORT_UNKNOWN, 0)
    _answered = _tc.get(TRANSPORT_ANSWERED, 0)
    _reported = agg["transport_reported"]
    _silent = agg["rounds"] - _reported
    lines.append(f"\n【执行侧「没听到回答」】outcome_unknown {_unknown} · answered {_answered}"
                 f" · 报了这一格的轮 {_reported}/{agg['rounds']}")
    if _silent:
        lines.append(f"  另有 {_silent} 轮这一格是空的 —— 那不是「听到回答了」, 是这一级没走到"
                     "远程执行 (L1 就死了 / 本地执行 / 重复候选不烧远端)")
    lines.append("  ⚠️ 只观测不改判决: 这些轮今天照样判 L2 失败。它的用处是让「这次远端可能已经"
                 "跑过一遍」在血缘里第一次有名字 (doc 60 §四 / #102 第 1 片)")

    # 改善判据那条带宽 (#87 第 4 片)。**这一栏的整件事是「归因分两种」**: 把「带宽跨轮换过」
    # 直接报成规格漂移, 会把测量档位降级这件事也报成漂移 (而另一格 `measurement_validity`
    # 早就记着它); 反过来只认「档位一致就算尺一致」, 那就永远看不见表被换过。所以两种分开数、
    # 分开点名, 归因不出来的第三态也不许并进任何一边。
    _bcom = agg["band_comparable_runs"]
    if agg["band_runs_with_band"] == 0:
        lines.append(f"\n【改善判据那条带宽】测不到: 没有任何一轮用过判决带宽 "
                     f"(血缘里没有 {LINEAGE_BAND_KEY} 这一项 —— 多半是这批运行跑在写下它之前, "
                     "**不是**「各轮同一条带宽」)")
    else:
        lines.append(f"\n【改善判据那条带宽】可比的运行 {_bcom} 个 · "
                     f"同一条带宽 {agg['band_same_runs']} · 档位降级换宽 "
                     f"{len(agg['band_tier_explained_cases'])} · 同档位下换了值 (规格漂移) "
                     f"{len(agg['band_spec_drift_cases'])} · 归因不了 "
                     f"{len(agg['band_unexplained_cases'])}"
                     f" · 凑不出比较的 {agg['band_unmeasurable_runs']} 个")
        for _key, _lab in (("band_tier_explained_cases", "降级换宽"),
                           ("band_spec_drift_cases", "规格漂移"),
                           ("band_unexplained_cases", "无从归因")):
            for entry in agg[_key]:
                lines.append(f"  {_lab}: {entry}")
        lines.append(f"  另有 {agg['band_rounds_no_band']} 轮走到了判决但没用带宽 "
                     "(AMBIGUOUS 那两支: 缺可比性能值 / 档位 uninformative), "
                     f"{agg['band_rounds_no_key']} 轮压根没这一项 —— 两种都不是「同一条带宽」")
        lines.append("  ⚠️ 只观测不改判决: 四态判决照旧用当轮的档位查表, 这一栏只回答"
                     "「那把尺跨轮是不是同一把」(AReaL-TIK 行 150 那句 𝒮 immutable 的"
                     " performance 那一半; #87 第 4 片)")

    for label, key in (("全部轮次", "speedup_all"), ("仅 decision-grade", "speedup_decision_grade")):
        q = agg[key]
        if q:
            lines.append(f"\n【加速比 · {label}】median {q['median']}x · p25 {q['p25']} · "
                         f"p75 {q['p75']} · min {q['min']} · max {q['max']} (n={q['n']})")
        else:
            lines.append(f"\n【加速比 · {label}】无数据")

    a, at = agg["admitted"]
    lines.append(f"\n【入库】通过的运行里 {a}/{at}"
                 + (f" = {agg['admitted_pct']}%" if at else "") + " 被写进跨运行经验库")
    for reason in agg["admit_rejections"]:
        lines.append(f"  拒收: {reason}")
    if agg["admit_rejections"]:
        lines.append(f"  （{agg['admit_rejection_runs']} 次运行, "
                     f"{len(agg['admit_rejections'])} 种不同的理由——名单按理由去重）")
    if at and not agg["admit_rejections"]:
        # 空名单也要说一句: 不印的话「门拒了 0 条」与「门没被问过」在报表里长得一样
        lines.append("  入库门本轮没有拒收任何一次通过的运行")
    if agg.get("gate_never_asked"):
        lines.append(f"  ⚠️ 有 {len(agg['gate_never_asked'])} 次通过的运行**没有入库判定记录**"
                     f" —— 门没被问, 不等于门放过了")
    if agg.get("counterfactual_rejections"):
        lines.append(f"  (另 {agg['counterfactual_runs']} 次**未通过**的运行给出了 "
                     f"{len(agg['counterfactual_rejections'])} 种假想判定 —— 生产里 "
                     f"`_save_to_db` 只在 all_passed 那一支问这道门, 所以它们不算拒收)")

    c = agg["cost"]
    lines.append(f"\n【成本】{c['tokens']} tokens / {c.get('calls', 0)} 次调用 ≈ ${c['usd']}"
                 f" · 墙钟 {c['wall_s']} s")

    curve = agg.get("threshold_curve") or {}
    if curve.get("tasks"):
        lines.append("\n【门槛扫描】" + curve.get(
            "caveat", "同一批已生成候选的重新计分"))
        for tr, rr in zip(curve["tasks"], curve.get("rounds") or []):
            tp, td = tr["passed"], tr["denom"]
            rp, rd = rr["passed"], rr["denom"]
            lines.append(f"  ≥{tr['threshold']}x: 任务 {tp}/{td}"
                         + (f" = {tr['pct']}%" if tr["pct"] is not None else "")
                         + f" · 轮 {rp}/{rd}"
                         + (f" = {rr['pct']}%" if rr["pct"] is not None else ""))

    pw = agg.get("power") or {}
    det = pw.get("detectable_delta")
    if det is not None:
        lines.append(f"\n【判变化的量程】当前 n≈{pw.get('n_min')} 时最多分辨 "
                     f"≈{det:.1%} 的比例差 (α=0.05, power=0.8, 最坏方差)")
        if pw.get("required_per_arm"):
            lines.append(f"  要分辨 {pw.get('target_delta', 0.30):.0%} 量级的差, "
                         f"约需 n={pw['required_per_arm']}/臂。"
                         f"「没测出差异」只有在这个量程内说才有意义。")

    # 「我无法确立的事」(设计 §17) —— 从不说测不准的系统, 所有数字都不可信。
    # 空章节也要在: 「本轮没有无法确立的事」本身是一个断言, 不是省略。
    un = agg.get("unable_to_establish") or []
    lines.append("\n【我无法确立的事】(设计 §17 —— 一等公民, 不是附注)")
    if not un:
        lines.append("  (无) —— 本轮所有读数都够格支撑其结论")
    else:
        for i, it in enumerate(un, 1):
            lines.append(f"  {i}. [{it.get('kind', 'other')}] {it.get('claim', '')}")
            lines.append(f"     原因: {it.get('reason', '')}")
        lines.append("  ⚠️ 以上条目**不许**被读成否定结论 (「没测出」≠「没有」)。")
    return "\n".join(lines)


# ─── 真机执行 ─────────────────────────────────────────────────

def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "?"
    except Exception:  # noqa: BLE001
        return "?"


def record_rounds(history):
    """把 `iteration_history` 原样搬进报表记录, **不做键白名单**

    第一版是白名单 (`round`/`action`/`all_passed`/`levels_passed`/
    `measurement_validity`/`gflops`/`speedup`), 于是 Conductor 写的
    `candidate_signature`/`duplicate_of`/`recipes_applied` 三键在这一步被悄悄丢掉 ——
    而 doc 48 第七节第 2 条正承诺「跑完就能离线算重复率」。真机跑完才发现落盘 JSON 里
    根本没有签名, 那一批 12 轮的重复基率**永久丢了** (只剩终端里那行 `[Novelty]` 汇总)。
    与 doc 33 的 `runs`/`max_time_ms`、doc 37 的 `llm_cost` 同一族: 值算出来了,
    死在搬运路上。**白名单的问题不是漏了哪几个键, 而是「血缘以后每加一个键都要记得来
    这里加一次」—— 而忘记的代价是静默的。**
    """
    out = []
    for h in history or ():
        rec = dict(h)
        rec["all_passed"] = bool(h.get("all_passed"))
        rec["levels_passed"] = h.get("levels_passed") or {}
        out.append(rec)
    return out


def run_one(label, kernel_type, precision, shape, backend, rounds, run_root=""):
    """跑一次完整任务, 返回一份原始记录 (不聚合, 便于落盘复盘)

    `run_root`（§22.1）: 一次运行一个布局根，由 `main` 用 `new_run_root(out_dir, label)`
    算好传进来 —— 于是「哪一个目录是这一次运行的根」只有一处命名，而 `base_dir`（`out_dir`）
    的清理/保留仍是本驱动的既有策略。**不传**时 `Conductor` 的形态回到「声明」（候选的角色
    路径是 `candidate/code.<ext>`，不落盘）。

    **每次运行新建一个 LLMClient**: `CostTracker` 是挂在 client 上累加的, 共用一个
    client 会让 `metadata["llm_cost"]` 变成**从头到现在的累计值**而不是本次的值。
    第一版就是这么写的, 于是六次运行的 token 数成了 15638 → 34457 → 78801 → 101156
    → 143717 → 166345 这样一条单调递增曲线, 报表把「累计值」再求和得到 540114,
    而真实总量是 166345 —— **高估 3.25 倍**, 且每一条单独看都像合理的数。
    顺带也让每次运行成为独立样本 (不共享任何跨运行状态)。
    """
    llm = llm_from_config()
    rec = {"case": label, "kernel_type": kernel_type, "shape": list(shape),
           "ok": False, "error": None, "rounds": [], "final_all_passed": False,
           "rounds_to_pass": None, "passed_via": None, "admitted": False,
           "admit_reason": "", "tokens": 0, "cost_usd": 0.0, "llm_calls": 0,
           "llm_sampling": None, "wall_s": 0.0}
    t0 = time.time()
    try:
        conductor = Conductor(backend=backend, llm=llm, max_optimize_rounds=rounds,
                              run_root=run_root)
        # §6.2 条款 2/3 的声明：**harness 是唯一一处「契约与输入同时在手」的地方**
        # （task 与 inputs 都在这里造）。必判 case 的清单与它的硬回退上限从任务进契约
        # ⇒ 进 hash ⇒ 判据不可改（CHECKLOG I92）。case 身份 = 输入形状签名，与评测侧
        # 发布的那一格**同一个函数**（`contract.case_key`）—— 两处各写一个格式就会漂。
        inputs = inputs_for(kernel_type, shape)
        _case = case_key(inputs)
        # §3.4（I94）: 契约**拥有输入** —— 声明从这组输入自己派生（`case_from_inputs` 读形状
        # 与数值语义），所以「契约说评什么」与「本次评了什么」不可能对不上。未知布局的算子
        # 也能这样声明（不依赖那张布局表）。
        _declared = case_from_inputs(inputs, name=f"{kernel_type}-{shape}")
        art = conductor.execute(
            _task(kernel_type, precision, required_cases=(_case,),
                  max_regression={_case: MAX_REGRESSION},
                  # 权重与必判清单是**同一条 case**（同一个 `_case` 变量）: 声明与输入
                  # 来自同一次构造，不许各写一个字符串。
                  case_weights={_case: SINGLE_CASE_WEIGHT}, cases=(_declared,)),
            inputs, target_gflops=0.0)
    except Exception as exc:  # noqa: BLE001
        # 一次 ssh 抖动不该毁掉整场测量: 记成崩溃样本, 继续下一个
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        rec["wall_s"] = round(time.time() - t0, 1)
        # 崩溃的这次也可能已经发过几个请求。它们那侧的配置同样是读数条件, 而产物没有
        # `metadata` 可读 ⇒ 直接问账本 (`client` 是本次独有的, 不会串到别的运行)。
        # 读不到就照旧留 None ⇒ 报表会把它算进「无从知道」那一栏，不假装是全的。
        try:
            rec["llm_sampling"] = llm.cost_tracker.sampling_configs()
        except Exception:  # noqa: BLE001
            pass
        return rec

    rec["ok"] = True
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["rounds"] = record_rounds(art.iteration_history)
    # `KernelArtifact.all_passed` 是 `all(v.passed for v in verification_results)`,
    # 而 **all([]) 是 True** —— 一份没有任何验证结果的产物会被它判成「全过」。
    # 两条早退路径 (入口校验失败 / 环境自检未通过) 都填了合成的 L1 失败, 所以现在踩不到;
    # 但「跑出了候选」本来就该是任务成功的定义之一, 所以这里显式要求至少有一轮。
    rec["final_all_passed"] = bool(rec["rounds"]) and bool(getattr(art, "all_passed", False))
    # rounds_to_pass 的口径见 first_pass_round 的 docstring (含被真机证伪的那一版)
    rec["rounds_to_pass"], rec["passed_via"] = first_pass_round(rec["rounds"])
    # 入库决定在 Conductor 里只打印、不落产物, 所以这里用同一个纯函数复算一遍。
    # 复算而不是解析 stdout: 后者会把报表绑在日志文案上。
    keep, reason = persist_decision(art.benchmark_data or {})
    rec["admitted"], rec["admit_reason"] = bool(keep), reason
    rec["tokens"], rec["cost_usd"], rec["llm_calls"] = cost_from_metadata(art.metadata)
    return rec


def _load_payload(path: str) -> dict:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if "runs" not in data:
        raise SystemExit(f"❌ {p} 不是 pass_rate.json (没有 runs 键)")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3, help="每个用例重复几次 (默认 3)")
    ap.add_argument("--rounds", type=int, default=3, help="max_optimize_rounds (默认 3)")
    ap.add_argument("--out", default="", help="报表落盘目录 (默认 output/harness/<时间戳>)")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), default=None,
                    help="离线判决两份已落盘的 pass_rate.json (不需要 GPU/LLM)")
    ap.add_argument("--target-delta", type=float, default=0.30,
                    help="判变化时关心的最小差幅 (默认 0.30 = 30 个百分点)")
    args = ap.parse_args()

    if args.compare:
        # 纯离线路径: 不建 LLMClient、不碰 RemoteExecutor —— 这是「改完之后判有没有用」
        # 的日常入口, 不该每次都要 GPU 机在线。
        payload_a = _load_payload(args.compare[0])
        payload_b = _load_payload(args.compare[1])
        meta_a = {"git_head": payload_a.get("git_head"), "knobs": payload_a.get("knobs"),
                  "path": args.compare[0]}
        meta_b = {"git_head": payload_b.get("git_head"), "knobs": payload_b.get("knobs"),
                  "path": args.compare[1]}
        cmp = compare_runs(payload_a.get("runs") or [], payload_b.get("runs") or [],
                           meta_a=meta_a, meta_b=meta_b, target_delta=args.target_delta)
        print(format_comparison(cmp))
        # 退出码反映判决本身, 便于脚本化: 0=测出改善 / 1=测出回归 / 2=没测出或判不出
        return {"improved": 0, "regressed": 1,
                "no_detected_difference": 2, "underpowered": 3}[cmp["verdict"]]

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "output" / "harness"
        / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 不在这里建 LLMClient: run_one 每次自建一个, 否则 CostTracker 会跨运行累加
    # (见 run_one 的 docstring —— 第一版因此把成本高估了 3.25 倍)。
    # 这里只先建一次做**口径自检**: 判别实验不允许混进付费降级模型。
    probe = llm_from_config()
    print(f"[harness] 模型口径自检通过: {getattr(probe, 'model', '?')} (无降级模型)")
    backend = RemoteExecutor(mode="ssh", timeout=900)
    # `mode="ssh"` 现在是**承诺**: 连不上就在上一行抛 BackendUnavailable, 不会静默降级到
    # LocalExecutor 让这份报表装满 CPU 数字而表头照旧写着真机 (doc 49 §7.8)。
    # 这道 require 因此是**第二**防线, 但它还管着 SAME_DEVICE_BASELINE 那半边 ——
    # 缺能力就在开工前响, 而不是产出一份看起来正常的报表。
    backend.require(Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE)
    print(f"[harness] 执行侧能力自检通过: {backend.label}")

    runs = []
    for rep in range(args.repeats):
        for kernel_type, precision, shape in CASES:
            label = f"{kernel_type}/{precision.value}/{'x'.join(map(str, shape))}#{rep + 1}"
            print(f"\n{'=' * 62}\n[{label}] 真跑 (门槛 {MIN_SPEEDUP}x, 最多 {args.rounds} 轮)\n{'=' * 62}")
            rec = run_one(label, kernel_type, precision, shape, backend, args.rounds,
                          run_root=new_run_root(str(out_dir), label))
            runs.append(rec)
            print(f"[{label}] ok={rec['ok']} 轮数={len(rec['rounds'])} "
                  f"最终通过={rec['final_all_passed']} rounds_to_pass={rec['rounds_to_pass']} "
                  f"入库={rec['admitted']}")
            if rec["error"]:
                print(f"  ❌ {rec['error']}")

    agg = aggregate(runs)
    report = format_report(agg)
    print(report)

    device = device_record(backend)
    print(f"\n【实际用卡】{device['state']}  (来源: {device['source']})")
    if device["conflicts"]:
        print(f"  ⚠️ 本地设了 OPERATOR_AGENT_GPU={device['requested_env']}, 但 "
              f"`SSHExecutor` **从不读它** —— 它在本进程自动挑最闲的卡, 再把选中的号"
              f"下发给远端。这份报表记录的卡号以 {device['state']} 为准; "
              f"要真的钉住某张卡得改 `_pinned_gpu()` 的选卡策略 (那是行为改动, 另一次对照)。")

    llm_side = sampling_from_runs(runs)
    print("\n【实际请求的模型侧配置】(逐调用覆盖之后真发出去的那些, 出处=账本 —— #98)")
    sampling_lines = format_sampling_configs(llm_side)
    if not sampling_lines:
        print("  ⚠️ 没有任何一条调用带上了采样配置 ⇒ 这份报表的模型侧条件**无从知道**。"
              "「无从知道」不是「用了默认值」。")
    for x in sampling_lines:
        print(f"  - {x}")
    if llm_side["n_runs_without_ledger"]:
        print(f"  ⚠️ 另有 {llm_side['n_runs_without_ledger']}/{len(runs)} 次运行没有账本"
              f" (崩溃且连 client 也问不到) ⇒ 上面这张表不覆盖它们")

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
        "knobs": {"min_speedup": MIN_SPEEDUP, "max_error": MAX_ERROR,
                  "max_optimize_rounds": args.rounds, "repeats": args.repeats,
                  "min_denom": MIN_DENOM, "cases": [c[0] for c in CASES],
                  # #98: 这一维之前是空的 —— knobs 里六项没有一项是模型侧的, 而下面这份是
                  # 逐调用覆盖**之后**真发出去的配置 (来源=账本, 不是构造参数)。
                  "llm": llm_side},
        "device": device,
        "aggregate": agg,
        "runs": runs,
    }
    path = out_dir / "pass_rate.json"
    # record_rounds 不再挑键, 所以血缘里万一出现一个不可序列化的值, 退化成它的 repr
    # 也比让整份报表在落盘这一步崩掉好 —— 崩在这里会连带丢掉已经付过钱的 12 轮真机数据。
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    print(f"\n报表已落盘: {path}")
    print("⚠️ 引用这些数字时必须带上 git_head 与 knobs —— 换一次配方/判据/prompt, "
          "上一份报表就只是历史, 不是基线。")
    # 这句话在 #98 之前是**多报了一口**: knobs 当时六项没有一项是模型侧的, 而所有数字都是在
    # `enable_thinking=False`、逐调用覆盖过的 temperature 下发出来的。现在 `knobs.llm` 记的是
    # 账本里真发出去的那些 —— 但「读数的全部条件」仍然是一句**够不到**的声明, 它只覆盖
    # 判据/采样这两侧, 不覆盖 prompt 装配的每一次改动 (那要靠 git_head 去看代码)。
    print("   knobs 现在含 `llm` 这一维 (model/temperature/max_tokens/enable_thinking, "
          "出处=账本而非构造参数, #98)。")
    return 0 if agg["denominator_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
