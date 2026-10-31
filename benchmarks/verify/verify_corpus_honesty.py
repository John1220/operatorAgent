"""真机核对「负样本记账」与「排序轴」 (doc 49 §7.14 / §7.15 / §八 第 13 条)

跑法 (会触发 deploy(), 需要能连到 GPU 机; **不消耗任何 LLM token**):
    python3 benchmarks/verify/verify_corpus_honesty.py

要还的债: §7.14 (失败语料的三处记账) 与 §7.15 (排序轴的先验数) 落地时**只有离线用例**,
而离线用例喂的是我自己写的错误文本。签名 `level|cls|frame` 的三个成分里, `cls` 走关键词表
还是走「最后一个异常名」、`frame` 在远端回传的错误原文里到底是不是恒为 `-`、真实文本能不能
在 JSONL 里逐字往返 —— 都取决于远端那份载荷长什么样, 离线用例回答不了。doc 45 §八 的三条
改判条件 (签名碎成 1:1 / `unknown` 桶占比 >1/3 / 栈帧成分从不发火) 要的是**真实文本的分布**,
而截至本次只有 3 条。

臂之间只差「失败从哪来」—— 六个各有独立根因的真实失败, 全部经**生产函数**
`Conductor._record_failure` / `_save_to_db` 入库, 不是另写一份记账代码:

| 臂 | 注入 | 期望的来源 |
|---|---|---|
| R1 | 模块级 `__import__("…absent…")` | 远端 ModuleNotFoundError |
| R2 | 模块级读一个未定义名 | 远端 NameError |
| R3 | 模块级 `1 // 0` | 远端 ZeroDivisionError |
| R4 | 模块级 `"x" + 7` | 远端 TypeError |
| L1 | 未终结的函数定义 (语法错) | 本地语法预检 |
| L2 | `def kernel(X, weight)` 少一个入参 | 本地接口契约检查 |

为什么「来源」是硬门而不是读数: 一条被本地预检拦掉的失败, 它的错误文本是**我们自己写的**,
拿它算出来的签名分布不代表语料将来收到的是什么样文本; 反过来若一个设计成远端的臂其实在
本地就红了, 说明预检变强、远端那类文本进不了语料。两种都会让读数失去意义, 所以先门后读数。

判据 (方向都口算过, 见文末 `main()` 里的 checks):
  · **注入的两个语料实例就是被写的那两个, 且落点在临时目录** —— 这条是第一版探针缺的,
    它因此把 7 条探针记录写进了仓库语料而自己全绿: `FailureDatabase` 有 `__len__`, 空实例
    是 falsy 的, 而 `Conductor.__init__` 写的是 `failure_db or FailureDatabase()` (已修,
    `is None` 判断 + `tests/unit/test_kernel_db_persistence.py::TestInjectedCorpusIsTheOneWritten`)
  · 每条真实失败落库时 `fixed is False`, id 是 `cx-{task_id}:{签名}` (内容派生, 不是插入序号)
  · 签名里不含 `/` 或 `\\` —— 它是跨运行语料的键, 带路径就等于把那台机器写进仓库
  · 同一条失败以两个 task_id 各记一次 ⇒ `recurrence()` 对该签名数到 **2** (复发门的前提)
  · 一次真实 L3 读数经 `_save_to_db` ⇒ `score` 就是测得的比值; 这条真实记录把 `_perf_weight`
    从 0 抬起, 且严格高于 `KernelDatabase` **自己的** 那批 seed 的权重上界 (口径取模块自身的
    数据算, 不抄数字 —— 以后给 seed 补了实测值, 这条会跟着动)
  · 六个臂里没有两个不同根因被合并成同一个签名 —— 碎分 (签名≈记录) 与过并 (记录>签名)
    是同一道门的两个失效方向, 只盯一个会漏掉另一个
  · `speedup_vs_baseline` 与 `baseline_time_ms / min_time_ms` 一致 (量具自检)
  · 真实错误文本在临时 JSONL 里往返后逐字相等, 每个 id 只回一条
  · `FixerAgent.produce_reflexion()` 在无 LLM 时返回 None —— F1 那个「已沉淀」分支的前提
  · 检索方向: 相似度高的**未修复**那条, 必须排在无关但 `fixed=True` 那条之前 (F3)

读数 (不判对错, 只打分布): 记录数 vs 不同签名数、`unknown` 桶占比、栈帧非 `-` 的条数、
`is_promotable_failure` 的放行率、六个臂里几个到了远端。

⚠️ 只打印派生值 (层级、类名、布尔、计数)。远程错误原文带那台机器的工作目录与用户名,
不进 stdout、不进仓库; 语料写到 `tempfile` 目录, 跑完即弃 —— 本驱动不该在
`output/failure_db.jsonl` 里留下任何一行 (这条也是硬门, 见 J10)。
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operator_agent.core.backend_caps import Cap  # noqa: E402
from operator_agent.app.conductor import Conductor  # noqa: E402
from operator_agent.core.kernel_db import (  # noqa: E402
    FailureDatabase, FailureRecord, KernelDatabase, _perf_weight, is_promotable_failure)
from operator_agent.execute.remote_executor import RemoteExecutor  # noqa: E402
from operator_agent.core.schemas import (  # noqa: E402
    ConstraintSpec, KernelArtifact, Precision, SourceSpec, TargetSpec, TaskComplexity,
    TaskSchema, TaskType)
from operator_agent.decide.verifier import Verifier  # noqa: E402

BASE = (Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE)
ROWS, COLS = 16384, 6144
KEYS = {"X", "weight", "bias"}

LN_BODY = '''
@triton.jit
def _ln(x_ptr, w_ptr, b_ptr, y_ptr, n_rows, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols,
             (xc * rstd * w + b).to(y_ptr.dtype.element_ty), mask=mask)


def kernel(X, weight, bias):
    n_rows, n_cols = X.shape
    Y = torch.empty_like(X)
    _ln[(n_rows,)](X, weight, bias, Y, n_rows, n_cols, 1e-5,
                   BLOCK=triton.next_power_of_2(n_cols))
    return Y
'''

LN_HEAD = '''
import torch, triton, triton.language as tl

'''

# (标签, 毒在哪一行, 期望来源: "remote" | "local")
FAILING_ARMS = (
    ("R1 模块级导入一个不存在的模块", 'MISSING = __import__("probe_absent_module_x")\n', "remote"),
    ("R2 模块级读未定义名", 'SCALE = UNBOUND_PROBE_NAME + 1\n', "remote"),
    ("R3 模块级除零", 'DENOM = 1 // 0\n', "remote"),
    ("R4 模块级 str + int", 'STAMP = "x" + 7\n', "remote"),
    ("L1 未终结的函数定义 (语法错)", "", "local"),
    ("L2 接口契约少一个入参", "", "local"),
)


def arm_code(label, poison):
    if label.startswith("L1"):
        # 语法错: 截断 kernel() 的定义 —— 本地 compile() 必红, 不该发去远端
        return LN_HEAD + poison + LN_BODY.split("def kernel")[0] + "def kernel(X, weight, bias):\n"
    if label.startswith("L2"):
        body = LN_BODY.replace("def kernel(X, weight, bias):", "def kernel(X, weight):")
        return LN_HEAD + body
    return LN_HEAD + poison + LN_BODY


def make_task(task_id, min_speedup=0.0):
    return TaskSchema(
        task_id=task_id,
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="layernorm", complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton", compute_units=78,
                          warp_size=32, required_caps=BASE),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=min_speedup,
                                   max_error=1e-1),
    )


def inputs():
    rng = np.random.default_rng(29)
    return {"X": rng.standard_normal((ROWS, COLS)).astype(np.float32),
            "weight": np.ones(COLS, dtype=np.float32),
            "bias": np.zeros(COLS, dtype=np.float32)}


def origin_of(result):
    """这条 L1 失败是本地拦的还是远端报的 —— 只从结果里能读到的信号判, 不靠臂名"""
    metrics = result.metrics or {}
    if metrics.get("compile_status") == "interface_mismatch":
        return "local"
    # §22.2 的执行器拦下候选（编译超时 / 撞上限）是**本地**发生的事：`compile_status`
    # 取 `compile_<status>` 那一族（生产者 `verifier._verify_compile`）。漏了这一支，
    # 一次本地超时会被这条探针读成「远端报的」—— 结论方向正好相反。
    if str(metrics.get("compile_status") or "").startswith("compile_"):
        return "local"
    if "本地语法预检失败" in (result.message or ""):
        return "local"
    return "remote"


def repo_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="probe_corpus_")
    failure_jsonl = str(Path(workdir) / "failure_db.jsonl")
    kernel_jsonl = str(Path(workdir) / "kernel_db.jsonl")
    repo_failure_db = (Path(__file__).resolve().parents[2] / "output" / "failure_db.jsonl")
    repo_lines_before = repo_lines(repo_failure_db)

    backend = RemoteExecutor(mode="ssh", timeout=900)
    backend.require(*BASE)
    print(f"执行侧: {backend.label}  (degraded={backend.resolution.degraded})")
    if backend.resolution.degraded:
        print("❌ 门面降级了 —— 那这些失败文本就不是远端的")
        return 1
    if not backend.deploy():
        print("❌ 部署失败")
        return 1

    my_failure_db = FailureDatabase(jsonl_path=failure_jsonl)
    my_kernel_db = KernelDatabase(jsonl_path=kernel_jsonl)
    conductor = Conductor(backend=backend, kernel_db=my_kernel_db,
                          failure_db=my_failure_db, llm=None)
    # J0 前提门, 必须在任何一次写之前。上一版探针把语料注进 Conductor 就以为落点在临时目录,
    # 而 `failure_db or FailureDatabase()` 因为空语料的 `__len__` 是 0 而**把它换成默认实例**
    # —— 探针往仓库语料里写了 7 条, 自己的记账判据却全绿 (它读的就是那个被换上去的实例)。
    # 观察面写两路: 身份 (是不是我给的那个对象) 与落点 (它要写的路径), 缺一路就会漏掉
    # 「对象对了但路径不对」或「路径对了但写进了另一个实例」。
    if (conductor.failure_db is not my_failure_db
            or conductor.kernel_db is not my_kernel_db
            or str(conductor.failure_db._jsonl) != failure_jsonl
            or str(conductor.kernel_db._jsonl) != kernel_jsonl):
        print("❌ 注入的语料不是实际被写的那一个 —— 本驱动会污染仓库语料, 停")
        return 1
    verifier = conductor.verifier
    data = inputs()

    print(f"\n{'=' * 66}\n[前提] 已知能编译的核必须在远端编译成功 —— 否则下面的读数没有意义\n{'=' * 66}")
    control = verifier._verify_compile(LN_HEAD + LN_BODY, dsl="triton",
                                       input_key_sets=[KEYS])
    print(f"  control passed={control.passed}  来源={origin_of(control)}")
    if not control.passed:
        print("❌ 已知能编译的核在远端编译失败: 远端环境/部署不同步")
        return 1

    # ── 六个真实失败各经生产函数入库 ────────────────────────────────
    print(f"\n{'=' * 66}\n[臂] 六条真实失败 → Conductor._record_failure\n{'=' * 66}")
    task = make_task("probe-task-1")
    per_arm = []          # (标签, 期望来源, 实得来源, 签名, promotable)
    seen_ids = set(conductor.failure_db._records)
    over_merged = []
    last_real_failure = None
    for label, poison, expect_origin in FAILING_ARMS:
        code = arm_code(label, poison)
        r = verifier._verify_compile(code, dsl="triton", input_key_sets=[KEYS])
        got = origin_of(r)
        if r.passed:
            print(f"  ❌ {label}: 这条臂**通过**了 —— 它不是已知会被看见的信号")
            per_arm.append((label, expect_origin, "passed", "", False))
            continue
        conductor._record_failure(task, [r], "探针: 未修复, 只测记账", fixed=False)
        # 取回刚写的那条靠**键集合差**, 不靠重算 `cx-{task_id}:{failure_signature(...)}`:
        # 后者要在探针里复刻生产的取文本口径 (`error_log or message` 再截断再拼 level),
        # 复刻错一个字段就拿不到记录, 于是「取不到」会被读成「记账坏了」。
        fresh = [k for k in conductor.failure_db._records if k not in seen_ids]
        seen_ids |= set(fresh)
        if not fresh:
            # 两个不同根因拿到同一个内容派生 id ⇒ 签名把不该合并的合并了 (碎分的反面)
            over_merged.append(label)
            print(f"  ⚠️ {label}: 复写了先前某条的 id —— 两个根因同一签名")
            per_arm.append((label, expect_origin, got, "(复写)", False))
            continue
        rec = conductor.failure_db._records[fresh[-1]]
        sig = rec.signature
        if got == "remote" and last_real_failure is None:
            last_real_failure = rec
        per_arm.append((label, expect_origin, got, sig, is_promotable_failure(rec)))

    records = list(conductor.failure_db._records.values())
    sigs = sorted({r.signature for r in records})
    unknown = sum(1 for r in records if r.signature.split("|")[1] == "unknown")
    framed = sum(1 for r in records if r.signature.split("|")[-1] != "-")
    promotable = sum(1 for r in records if is_promotable_failure(r))
    reached_remote = sum(1 for _, _, got, _, _ in per_arm if got == "remote")
    print(f"\n  读数: {len(records)} 条记录 / {len(sigs)} 个不同签名"
          f" ({reached_remote}/{len(per_arm)} 个臂真的到了远端)")
    for label, expect, got, sig, ok in per_arm:
        leak = ("/" in sig) or ("\\" in sig)
        print(f"    {label}: 期望来源={expect} 实得={got}  "
              f"签名={'<含路径,不打印>' if leak else sig}  可复用={ok}")
    print(f"  读数: unknown 桶 {unknown} 条, 栈帧非 '-' {framed} 条, "
          f"可被跨运行复用的 {promotable} 条")

    # 落点确认放在烧 GPU 时间之前: 语料要是写到了仓库, 越往后跑得越久、污染越多
    polluted = repo_lines(repo_failure_db) - repo_lines_before
    if polluted:
        print(f"  ❌ 仓库的 output/failure_db.jsonl 被写动了 {polluted} 行 —— "
              f"落点不是临时目录, 后面不必再跑 (J14 就是为这一条加的)")
        return 1

    # ── 复发计数: 同一条真实失败在两个任务下各记一次 ────────────────
    print(f"\n{'=' * 66}\n[复发] 同一条远端失败以两个 task_id 各记一次\n{'=' * 66}")
    dup_task = make_task("probe-task-2")
    r_dup = verifier._verify_compile(arm_code(FAILING_ARMS[1][0], FAILING_ARMS[1][1]),
                                     dsl="triton", input_key_sets=[KEYS])
    conductor._record_failure(dup_task, [r_dup], "探针: 换个任务再记一次", fixed=False)
    dup_sig = next((r.signature for r in conductor.failure_db._records.values()
                    if r.task_id == "probe-task-2"), "")
    rec_count = conductor.failure_db.recurrence().get(dup_sig, 0)
    print(f"  读数: 该签名的不同任务数 = {rec_count} (门槛 = {conductor.failure_db._gate})")

    # ── 正样本: 一次真实 L3 读数经生产函数入库 ──────────────────────
    print(f"\n{'=' * 66}\n[正样本] 真实 L3 读数 → Conductor._save_to_db\n{'=' * 66}")
    perf_task = make_task("probe-task-3")
    l3 = verifier._verify_performance(LN_HEAD + LN_BODY, data, 0.0, perf_task)
    artifact = KernelArtifact(kernel_id="probe-real-0001", task_id=perf_task.task_id,
                              code=LN_HEAD + LN_BODY, language="triton",
                              verification_results=[l3])
    artifact.benchmark_data = l3.metrics          # 与 conductor.py:811 同一句
    admitted = conductor._save_to_db(artifact, perf_task)
    m = l3.metrics or {}
    rec = conductor.kernel_db.get(artifact.kernel_id)
    speedup = m.get("speedup_vs_baseline")
    base_ms, cand_ms = m.get("baseline_time_ms"), m.get("min_time_ms")
    seeded = KernelDatabase(jsonl_path=str(Path(workdir) / "seeds_only.jsonl"))
    seed_w = max((_perf_weight(r) for r in seeded._records.values()), default=0.0)
    print(f"  读数: passed={l3.passed}  档位={m.get('measurement_validity')}  "
          f"can_gate={m.get('measurement_can_gate')}  入库={admitted}")
    print(f"  读数: speedup={speedup!r}  baseline_ms={base_ms!r}  candidate_ms={cand_ms!r}")
    if rec is not None:
        print(f"  落库: score={rec.score!r}  权重={_perf_weight(rec)!r}  "
              f"(seed 权重上界 {seed_w!r})")
    else:
        print("  ❌ 落库后取不到这条记录")

    # ── 往返: 真实文本过一遍 JSONL ─────────────────────────────────
    print(f"\n{'=' * 66}\n[往返] 重新打开临时语料, 逐字比对真实错误文本\n{'=' * 66}")
    reloaded = FailureDatabase(jsonl_path=failure_jsonl)
    same = (last_real_failure is not None
            and reloaded._records.get(last_real_failure.failure_id) is not None
            and reloaded._records[last_real_failure.failure_id].error_summary
            == last_real_failure.error_summary)
    n_lines = repo_lines(Path(failure_jsonl))
    print(f"  读数: 磁盘 {n_lines} 行 → 重放 {reloaded.loaded_from_disk} 条, "
          f"坏行 {reloaded.disk_skipped_lines}")
    if n_lines == 0:
        print("  ❌ 临时语料是空的: 写根本没落到这个文件 (与 J14 同源)")

    # ── 检索方向 (F3) + Reflexion 前提 ─────────────────────────────
    print(f"\n{'=' * 66}\n[检索] 真实文本做查询时, 相似度是否仍排在 fixed 之前\n{'=' * 66}")
    # `jsonl_path="none"` 而不是 `None`: `_resolve_jsonl` 对 falsy 值一律回落到仓库默认
    # 路径, 传 None 会把**真实历史语料**读进来, 这两条探针记录就和历史记录混排了。
    probe_db = FailureDatabase(jsonl_path="none")
    if last_real_failure is not None:
        probe_db.add(FailureRecord(
            failure_id="unrelated-fixed", kernel_type="layernorm", hardware="nvidia_gpu",
            dsl="triton", level="L1", error_summary="completely unrelated message zzz",
            fixed=True, signature="L1|SyntaxError|-", task_id="other"))
        probe_db.add(FailureRecord(**{**last_real_failure.to_dict(),
                                      "failure_id": "same-error-unfixed",
                                      "fixed": False}))
        hits = probe_db.search_by_error(last_real_failure.error_summary,
                                         kernel_type="layernorm", dsl="triton", limit=2)
        order = [h.failure_id for h in hits]
        print(f"  读数: search_by_error → {order}")
        print(f"  读数: search_by_context → "
              f"{[h.failure_id for h in probe_db.search_by_context(kernel_type='layernorm', dsl='triton', limit=2)]}")
        retrieval_ok = order and order[0] == "same-error-unfixed"
        context_ok = [h.failure_id for h in
                      probe_db.search_by_context(kernel_type="layernorm", dsl="triton",
                                                 limit=2)] == ["unrelated-fixed",
                                                               "same-error-unfixed"]
    else:
        print("  ❌ 没有可用的远端失败样本, 这一臂没做成")
        order, retrieval_ok, context_ok = [], False, False

    reflexion_none = conductor.fixer.produce_reflexion() is None
    all_records = list(conductor.failure_db._records.values())

    repo_lines_after = repo_lines(repo_failure_db)

    checks = [
        ("J14 注入的两个语料实例就是被写的那两个, 且落点在临时目录",
         conductor.failure_db is my_failure_db and conductor.kernel_db is my_kernel_db
         and str(conductor.failure_db._jsonl) == failure_jsonl),
        ("J1 六条臂的来源与臂名一致 (远端的确实到了远端, 本地的被本地拦住)",
         all(got == exp for _, exp, got, _, _ in per_arm)),
        ("J1b 没有两个不同根因被合并成同一个签名 (碎分的反面也要看得见)",
         not over_merged),
        ("J2 每条真实失败落库时 fixed=False",
         all(not r.fixed for r in all_records)),
        ("J3 签名里不含路径分隔符 (跨运行键不得带那台机器的目录)",
         all("/" not in r.signature and "\\" not in r.signature for r in all_records)),
        ("J4 id 全部是 cx-{task_id}:{签名} 形态",
         all(r.failure_id.startswith(f"cx-{r.task_id}:") for r in all_records)),
        ("J5 同一条失败换任务再记一次 → 复发计数 =2 (门槛的前提)", rec_count == 2),
        ("J6 真实读数落库后 score 就是测得的 speedup",
         rec is not None and rec.score == float(speedup or 0.0)),
        ("J7 真实读数让排序轴离开 0, 且严格高于 seed 的权重上界 (§7.15 的全部主张)",
         rec is not None and _perf_weight(rec) > 0.0 and _perf_weight(rec) > seed_w),
        ("J8 量具自检: speedup == baseline_ms / candidate_ms (同一份 metrics)",
         isinstance(speedup, (int, float)) and isinstance(base_ms, (int, float))
         and isinstance(cand_ms, (int, float)) and cand_ms > 0
         and abs(speedup - round(base_ms / cand_ms, 4)) < 5e-4),
        ("J9 真实错误文本在 JSONL 里逐字往返", same),
        ("J10 仓库里的 output/failure_db.jsonl 一行都没多 (探针不污染语料)",
         repo_lines_after == repo_lines_before),
        ("J11 produce_reflexion() 在无 LLM 时返回 None (F1 分支的前提)", reflexion_none),
        ("J12 真实文本查询: 相似但未修复的那条排在无关但 fixed 之前", retrieval_ok),
        ("J13 对照臂: 不带文本的 search_by_context 仍按 fixed 优先 (F3 没把排序改成常量)",
         context_ok),
    ]
    print(f"\n{'=' * 66}\n标定判据\n{'=' * 66}")
    bad = 0
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")
        bad += 0 if ok else 1
    print(f"\n未判定 (只攒数): 签名碎分比 {len(sigs)}/{len(records)}、"
          f"unknown 占比 {unknown}/{len(records)}、栈帧发火 {framed}/{len(records)}、"
          f"可复用 {promotable}/{len(records)} —— 门槛在 doc 45 §八, 样本还不够")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
