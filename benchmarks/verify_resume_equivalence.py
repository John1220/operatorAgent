#!/usr/bin/env python3
"""真机: 续跑到底续的是什么 —— 恢复出来的第 1 轮必须是被量过的那一轮

doc 50 §六 记的债: #59 那 9 例全跑在 MockGPU + 脚本化 LLM 上, 于是
「恢复的状态与不中断跑完等价」这句话从没在真机上验过。这里验三件事:

① **续跑不是重跑**: 恢复臂的第 1 轮条目必须与被中断那次**逐字段相同**。真机上两次
   独立测量得到同一个 `speedup` 的概率极低, 所以「数值完全相等」在这里是有鉴别力的
   证据 —— 而签名相等还不够 (同一份结构被重测一次, 签名相同、数值不同)。
② **少烧一轮**: 恢复臂的 LLM 调用次数必须显著低于「从头跑同样轮数」的对照臂。
   每臂各建一个 `LLMClient` —— 账单挂在 client 上累加, 共用一个 client 再去求和会
   高估 (doc 48 的 3.25 倍事故), 而差分读数还会把「谁也没调用」读成「调用了一次」。
③ **真实的 SIGKILL**: 不是「干净地停在第 1 轮之后」, 而是 round 2 进行中被打死 ——
   没有 atexit、没有收尾、`result.json` 根本没写过。这验的是循环内那个落盘点真的
   落了盘, 而不只是末次落盘。

外加一条反向对照: 半个 JSON 的快照必须报 `unreadable_snapshot`(带 `JSONDecodeError`),
**不许**被当成「没有快照」而静默从头跑。

跑法 (需要 GPU 机的 ssh 别名 + config.yaml 里的 LLM 端点):
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B benchmarks/verify_resume_equivalence.py
子进程模式 (被自身 SIGKILL 那一臂用, 不要手工调):
    ... verify_resume_equivalence.py --child <task_id> <output_root>
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from operator_agent.conductor import Conductor  # noqa: E402
from operator_agent.kernel_db import FailureDatabase, KernelDatabase  # noqa: E402
from operator_agent.llm_client import LLMClient  # noqa: E402
from operator_agent.remote_executor import RemoteExecutor  # noqa: E402
from operator_agent.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType,
)

# 跑动窗口内的代码指纹 (doc 50 §7.5: 那次五臂跑完才发现窗口里有人改了同一条路径的 5 个文件,
# 而一次真机跑没法「重跑一遍就好」)。依赖名单是**算出来的**, 不是手抄的。
# 按模块限定调用 (`provenance.stamp(...)`) 而不是 import 名字: `main()` 里那个局部变量就叫 `stamp`。
from benchmarks import provenance  # noqa: E402

REPO_ROOT = str(Path(__file__).resolve().parents[1])
DRIVER_REL = "benchmarks/verify_resume_equivalence.py"

SNAP = "execution_snapshot.json"
# 达不到的门槛: 循环必须真的转到第 2 轮, 否则「恢复第 1 轮」这件事根本没有第 2 轮可续
MIN_SPEEDUP = 10.0
def spec() -> dict:
    """真机臂 = layernorm on H20; dry 臂 = matmul on MockGPU + 脚本化 LLM。

    dry 只用来标定量具自己 (键名 / 找快照 / 状态字符串 / 子进程姿势), **不产出真机读数**;
    MockGPU 的计时是确定性的, 所以 B2 那条「逐字段相同」在 dry 下没有鉴别力。
    """
    if DRY:
        return dict(kernel_type="matmul", precision=Precision.FP32, shape=(32, 32),
                    hardware="mock_gpu", dsl="python")
    return dict(kernel_type="layernorm", precision=Precision.FP32, shape=(1024, 512),
                hardware="nvidia_gpu", dsl="triton")


# ── 判据清单: 每条都要能指出「修复改回旧写法时它会红」 ──────────────────
J = []          # (编号, 说明, 通过? (None = 本轮没有鉴别力), 读数)
KIDS = []       # 被本脚本拉起的子进程, 退出前必须收干净
DRY = bool(os.environ.get("RESUME_PROBE_DRY"))  # --dry: MockGPU + 无 LLM, 只标定量具自己 (不产生任何真机读数)


def check(idx: str, what: str, ok, reading: str) -> None:
    J.append((idx, what, ok, reading))
    mark = "➖" if ok is None else ("✅" if ok else "❌")
    print(f"  {idx} [{mark}] {what}  —— {reading}")


def make_backend():
    if DRY:
        from operator_agent.mock_gpu import MockGPU
        return MockGPU()
    return RemoteExecutor(mode="ssh", timeout=600)


def llm_from_config():
    """从 config.yaml 读自建端点。值只进环境变量, **不打印**。"""
    if DRY:
        # 脚本化 LLM 只用于标定量具; 它**不是**被测对象 (doc 48 §6.5 那条纪律)
        from tests.unit.test_resume_integrity import _ScriptedLLM
        return _ScriptedLLM()
    import yaml
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        llm = (cfg.get("llm") or {})
        os.environ.setdefault("LLM_BASE_URL", str(llm.get("base_url") or ""))
        os.environ.setdefault("LLM_MODEL", str(llm.get("model") or ""))
        os.environ.setdefault("LLM_API_KEY", str(llm.get("api_key") or ""))
    client = LLMClient(temperature=0.3, max_tokens=4096)
    if getattr(client, "fallback_model", None):
        raise SystemExit("❌ 本次验证不允许走降级模型 (会污染等价性的口径)")
    return client


def make_task(task_id: str) -> TaskSchema:
    s = spec()
    return TaskSchema(
        task_id=task_id,
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=s["kernel_type"], complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware=s["hardware"], dsl=s["dsl"]),
        constraints=ConstraintSpec(precision=s["precision"],
                                   min_speedup=MIN_SPEEDUP, max_error=1e-2),
    )


def inputs_for() -> dict:
    """**固定 seed**: 各臂必须看到同一份输入, 否则「第 1 轮读数逐字段相同」这条判据
    会被输入差异冒充成失败 (或更糟: 被确定性掩盖)。"""
    s = spec()
    rng = np.random.default_rng(7)
    if s["kernel_type"] == "matmul":
        m, k = s["shape"]
        return {"A": rng.standard_normal((m, k)).astype(np.float32),
                "B": rng.standard_normal((k, m)).astype(np.float32)}
    m, n = s["shape"]
    return {"X": rng.standard_normal((m, n)).astype(np.float32),
            "weight": np.ones(n, dtype=np.float32),
            "bias": np.zeros(n, dtype=np.float32)}


def arm_calls(llm):
    """每臂各建一个 LLMClient: 账单挂在 client 上累加, 共用一个再去求和会高估 (doc 48)。"""
    tracker = getattr(llm, "cost_tracker", None)
    if tracker is not None:
        return tracker.total_calls
    return getattr(llm, "calls", None)


def build(backend, rounds: int, out_root: Path, work: Path) -> Conductor:
    """一臂一个新 LLMClient + 一份临时语料。

    临时语料是**必须**的: doc 49 §7.17 那次探针把 7 条记录写进了仓库的
    `output/failure_db.jsonl`, 因为它注入的库被 `x or Default()` 换掉了。
    """
    c = Conductor(backend=backend, llm=llm_from_config(),
                  kernel_db=KernelDatabase(jsonl_path=str(work / f"kdb-{uuid.uuid4().hex[:6]}.jsonl")),
                  failure_db=FailureDatabase(jsonl_path=str(work / f"fdb-{uuid.uuid4().hex[:6]}.jsonl")),
                  max_optimize_rounds=rounds)
    c.store.output_dir = out_root
    return c


def find_snapshot(out_root: Path, task_id: str) -> Path | None:
    """按 task_id 找 round_done 最大的那份快照 (与 `_resolve_snapshot_path` 同口径)。"""
    best_key, best = None, None
    for p in sorted(out_root.rglob(SNAP)):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("task_id") != task_id:
            continue
        key = (int(data.get("round_done") or 0), p.stat().st_mtime)
        if best_key is None or key > best_key:
            best_key, best = key, p
    return best


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def result_of(out_root: Path, task_id: str, since: float | None = None):
    """按 task_id 数有几份产物 —— 只用来把「同一个 task_id 有多个候选」这件事说出来。

    ⚠️ 续跑臂与被续的那次**共用 task_id** (不然就是 task_mismatch), 所以「按 task_id
    找第一个命中」会读到上一臂的产物 —— dry 标定就是这么给出两个假绿的: B2 拿 A 的
    快照去比 A 自己的 result.json, 当然逐字段相同。这里用「本次运行开始之后落盘」做
    下界, 并把候选个数一起报出来, 让「读到的是谁」这件事可见。
    """
    cands = []
    for path in out_root.rglob("result.json"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if since is not None and mtime < since:
            continue
        data = read_json(path) or {}
        if data.get("task_id") == task_id:
            cands.append((mtime, path, data))
    if not cands:
        return None, None, 0
    _mtime, path, data = max(cands, key=lambda t: t[0])
    return path, data, len(cands)


def run_arm(name: str, backend, out_root: Path, work: Path, rounds: int,
            task_id: str, resume_from: str | None = None) -> dict:
    t0 = time.time()
    c = build(backend, rounds, out_root, work)
    try:
        art = c.execute(make_task(task_id), inputs_for(), None,
                        target_gflops=10.0, resume_from=resume_from)
    except Exception as e:  # noqa: BLE001
        # 只报类名: 远端/SSH 的错误原文里有路径, 不进 stdout 之外的任何地方
        print(f"\n  [{name}] ❌ 这一臂崩了: {type(e).__name__} —— "
              f"依赖它的判据会显式失败, 不会被当成「测到了」")
        return {"name": name, "artifact": None, "error": type(e).__name__,
                "calls": arm_calls(c.llm), "tokens": None, "snapshot": None,
                "snapshot_data": None, "result_path": None, "result": None,
                "conductor": c}

    calls = arm_calls(c.llm)
    tracker = getattr(c.llm, "cost_tracker", None)
    tokens = tracker.total_tokens if tracker is not None else None
    # 权威落点 = 本次运行的 artifact.saved_dir (conductor.py:895)。以前这里是
    # 「按 task_id 在整棵 output 树下找第一个命中」, 而**续跑臂与被续的那次共用
    # task_id** ⇒ 读到的是上一臂的产物。dry 标定就是这么给出两个假绿的:
    # B2 拿 A 的快照比 A 自己的 result.json, 当然逐字段相同。
    saved = Path(art.saved_dir) if getattr(art, "saved_dir", "") else None
    res_path = (saved / "result.json") if saved else None
    res = read_json(res_path) if res_path else None
    _p, _d, n_res = result_of(out_root, task_id)
    snap = (saved / SNAP) if saved and (saved / SNAP).is_file() else find_snapshot(out_root, task_id)
    print(f"\n  [{name}] 用时 {time.time() - t0:.0f}s · LLM 调用 {calls} 次 · "
          f"tokens {tokens} · 快照 round_done="
          f"{(read_json(snap) or {}).get('round_done') if snap else '无'} · "
          f"result.json {'有' if res else '无'} · 落点={saved.name if saved else '无'} · "
          f"该 task_id 在 output 树下共 {n_res} 份产物")
    return {"name": name, "artifact": art, "calls": calls, "tokens": tokens,
            "snapshot": snap, "snapshot_data": read_json(snap) if snap else None,
            "result_path": res_path, "result": res, "conductor": c}


# ── 被 SIGKILL 的那一臂: 子进程只做「跑到第 2 轮进行中」这件事 ────────────
def child_main(task_id: str, out_root: str) -> int:
    os.environ["OPERATOR_AGENT_OUTPUT_DIR"] = out_root
    backend = make_backend()
    # ⚠️ 这里必须和 `build()` 一样注入临时语料。真机那一跑 (doc 50 §7.7) 就是因为子进程
    # 走的是默认路径, 把 3 条 `retrieved` 事件写进了仓库的 `output/kernel_db.jsonl` ——
    # 而那些事件有读者 (检索排序里的「曾被检索 N 次」), 所以这不是无害的噪音。
    # `build()` 注入了、`child_main()` 没注入, 正是「同一个不变量只在一条路径上成立」的形状。
    work = Path(out_root).parent
    c = Conductor(backend=backend, llm=llm_from_config(),
                  kernel_db=KernelDatabase(jsonl_path=str(work / f"kdb-child-{task_id}.jsonl")),
                  failure_db=FailureDatabase(jsonl_path=str(work / f"fdb-child-{task_id}.jsonl")),
                  max_optimize_rounds=2)
    c.store.output_dir = Path(out_root)
    c.execute(make_task(task_id), inputs_for(), None, target_gflops=10.0)
    return 0


def kill_during_round2(out_root: Path, work: Path, backend, task_id: str) -> dict:
    """轮询到第 1 轮快照落盘后立刻 SIGKILL —— 此刻进程在 round 2 的 LLM 调用里。"""
    log = open(work / "child.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-B", __file__, "--child", task_id, str(out_root)],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    KIDS.append(proc)
    deadline = time.time() + 420
    snap = None
    while time.time() < deadline:
        snap = find_snapshot(out_root, task_id)
        if snap is not None:
            break
        if proc.poll() is not None:
            break
        time.sleep(2.0)
    got_snapshot = snap is not None
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=30)
    log.close()
    print(f"\n  [kill] 快照出现={got_snapshot} 子进程退出码={proc.returncode} "
          f"(被信号杀则为负数) result.json="
          f"{'有' if result_of(out_root, task_id)[1] else '没有(被杀在 save 之前)'}")
    if got_snapshot:
        # 把残骸也报出来: `.tmp` 留着不丢人, 把它当快照才丢人
        leftovers = sorted(p.name for p in snap.parent.glob(SNAP + ".tmp*"))
        print(f"  [kill] 该目录下的 .tmp 残骸: {leftovers or '无'}")
    data = read_json(snap) if snap else None
    return {"snapshot": snap, "snapshot_data": data, "returncode": proc.returncode,
            "leftovers": sorted(str(p) for p in snap.parent.glob(SNAP + ".tmp*")) if snap else []}


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        return child_main(sys.argv[2], sys.argv[3])
    if "--dry" in sys.argv:
        global DRY
        DRY = True
        os.environ["RESUME_PROBE_DRY"] = "1"   # 子进程 (--child 那一臂) 必须同源

    stamp = time.strftime("%Y%m%d-%H%M%S")
    root_name = "probe-resume-dry" if DRY else "probe-resume"
    work = Path(__file__).resolve().parents[1] / "output" / f"{root_name}-{stamp}"
    out_root = work / "artifacts"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"探针工作目录: {work}" + ("   [DRY: MockGPU + 无 LLM, 只标定量具]" if DRY else ""))
    # 读数条件 + 开跑前的代码指纹。指纹由**子进程**读盘算 (doc 56: "the remedy is another
    # writer, not richer logging"), 取不到就返回 None —— 后面据此拒绝打印结论行。
    print("  " + provenance.stamp(REPO_ROOT))
    deps = provenance.closure(DRIVER_REL, REPO_ROOT,
                              extra_entries=["remote_scripts/run_kernel.py"])
    prov_before = provenance.snapshot_via_subprocess(deps, REPO_ROOT)

    backend = make_backend()
    caps = getattr(backend, "capabilities", frozenset())
    from operator_agent.backend_caps import Cap
    # ⚠️ 这里曾经写过 `backend.device_record()` —— **提供方上没有这个方法** (它住在
    # `benchmarks/harness_pass_rate.py`, 是量具自己的函数), 而外面套了 `except Exception`
    # ⇒ 属性错误被吞成一个空字典, 表头打出「实际器件=None 卡号=None」, 看起来像「这台机
    # 没告诉我们它是谁」, 真相是我的量具发明了一个不存在的 API。这是 doc 49 §7.7 那条
    # 「假件不能被当被测对象」的对偶:**异常兜底不能把「我没接对」伪装成「没有这个数据」**。
    # 卡号改从提供方真有的那条取 (#52 的 `selected_gpu`), 器件名交给日志里的环境指纹。
    print(f"  后端: {backend.label}  REAL_DEVICE={Cap.REAL_DEVICE in caps}  "
          f"卡号={getattr(backend, 'selected_gpu', None)}")

    tid_a = f"rsA-{stamp}"
    tid_b = f"rsB-{stamp}"
    tid_k = f"rsK-{stamp}"

    # ── A: 干净地跑 1 轮, 留下可用的第 1 轮快照 ───────────────────────
    print(f"\n{'=' * 66}\n[A] 只跑 1 轮 (制造一个可续的断点)\n{'=' * 66}")
    a = run_arm("A", backend, out_root, work, 1, tid_a)
    hist_a = (a["snapshot_data"] or {}).get("iteration_history") or []
    check("A1", "A 留下了可解析的快照且 round_done==1",
          (a["snapshot_data"] or {}).get("round_done") == 1 and len(hist_a) == 1,
          f"round_done={(a['snapshot_data'] or {}).get('round_done')} "
          f"history={len(hist_a)} 条")
    check("A2", "A 没请求续跑时 resume 状态是 not_requested (不是缺键)",
          ((a["result"] or {}).get("resume") or {}).get("status") == "not_requested",
          f"result.json 的 resume={json.dumps((a['result'] or {}).get('resume'), ensure_ascii=False)}")

    # ── B: 从 A 的目录续跑到 2 轮 ────────────────────────────────────
    print(f"\n{'=' * 66}\n[B] resume_from=A 的产物目录, max_rounds=2\n{'=' * 66}")
    b = run_arm("B", backend, out_root, work, 2, tid_a,
                resume_from=str(a["snapshot"].parent))
    res_b = b["result"] or {}
    st_b = res_b.get("resume") or {}
    hist_b = res_b.get("iteration_history") or []
    r1_b = next((h for h in hist_b if h.get("round") == 1), None)
    r1_a = hist_a[0] if hist_a else None

    def _cmp_keys(h):
        return {k: (h or {}).get(k) for k in
                ("action", "all_passed", "levels_passed", "gflops", "speedup",
                 "measurement_validity", "recipes_applied", "candidate_signature")}

    check("B1", "result.json 里 resume.status == resumed 且 round_restored == 1",
          st_b.get("status") == "resumed" and st_b.get("round_restored") == 1,
          f"status={st_b.get('status')} round_restored={st_b.get('round_restored')} "
          f"dir={st_b.get('dir_name')}")
    check("B2", "续跑的第 1 轮与被中断那次**逐字段相同** (含真机测出来的 speedup)",
          r1_b is not None and _cmp_keys(r1_b) == _cmp_keys(r1_a),
          f"相同={_cmp_keys(r1_b) == _cmp_keys(r1_a)} · A 的第 1 轮 speedup="
          f"{(r1_a or {}).get('speedup')} 签名={(r1_a or {}).get('candidate_signature')} · "
          f"B 的 speedup={(r1_b or {}).get('speedup')} 签名={(r1_b or {}).get('candidate_signature')}")
    check("B3", "B 真的转到了第 2 轮 (不是只有恢复出来的第 1 轮)",
          any((h or {}).get("round") == 2 for h in hist_b),
          f"轮号={[h.get('round') for h in hist_b]}")

    # ── C: 从零跑 2 轮, 作为「重跑要花多少」的对照 ────────────────────
    print(f"\n{'=' * 66}\n[C] 不续跑, 从零跑 2 轮 (对照成本)\n{'=' * 66}")
    c_arm = run_arm("C", backend, out_root, work, 2, tid_b)
    if None in (b["calls"], c_arm["calls"]):
        check("C1", "续跑臂的 LLM 调用少于从零跑同样轮数", None, "读不到账单, 这条没有被问")
    else:
        check("C1", "续跑臂的 LLM 调用少于从零跑同样轮数 (B 少一轮 = 没重烧第 1 轮)",
              0 < b["calls"] < c_arm["calls"],
              f"B={b['calls']} 次 / C={c_arm['calls']} 次 · tokens B={b['tokens']} "
              f"C={c_arm['tokens']}")

    # ── K: 真实 SIGKILL 后仍能续 ─────────────────────────────────────
    print(f"\n{'=' * 66}\n[K] 第 2 轮进行中被 SIGKILL, 然后续跑\n{'=' * 66}")
    k = kill_during_round2(out_root, work, backend, tid_k)
    check("K0", "前提: 子进程真的是被信号打死的, 不是自己跑完的",
          k["returncode"] is not None and k["returncode"] < 0,
          f"returncode={k['returncode']} (负数=被信号杀; 0 意味着它自己跑完了, 下面几条就没有「崩溃」可验)")
    check("K1", "被杀死时磁盘上已有一份可解析的快照 (循环内落盘点真的落了盘)",
          isinstance(k["snapshot_data"], dict)
          and int((k["snapshot_data"] or {}).get("round_done") or 0) >= 1,
          f"round_done={(k['snapshot_data'] or {}).get('round_done')} "
          f"history={len((k['snapshot_data'] or {}).get('iteration_history') or [])} 条")
    if k["snapshot"] is None:
        check("K2", "写一半的 .tmp 没被当成快照", None, "本轮没留下 .tmp 残骸, 这条没有被问")
    else:
        check("K2", "写一半的 .tmp 没被当成快照",
              None if not k["leftovers"] else Path(k["snapshot"]).name == SNAP,
              f"残骸={[Path(p).name for p in k['leftovers']] or '无 (kill 没撞上写盘那一刻)'} · "
              f"选中的={Path(k['snapshot']).name}")
    if k["snapshot"] is not None:
        k_rounds = int((k["snapshot_data"] or {}).get("round_done") or 0)
        k_hist = (k["snapshot_data"] or {}).get("iteration_history") or []
        k2 = run_arm("K-resume", backend, out_root, work, k_rounds + 1, tid_k,
                     resume_from=str(k["snapshot"].parent))
        st_k = (k2["result"] or {}).get("resume") or {}
        hist_k = (k2["result"] or {}).get("iteration_history") or []
        first_restored = hist_k[0] if hist_k else None
        src_first = k_hist[0] if k_hist else None
        check("K3", "真崩溃之后续上: status=resumed 且 round_restored 与快照一致",
              st_k.get("status") == "resumed" and st_k.get("round_restored") == k_rounds,
              f"status={st_k.get('status')} round_restored={st_k.get('round_restored')} "
              f"快照={k_rounds}")
        check("K4", "被杀那次的第 1 轮读数原样恢复 (签名与 speedup 都对得上)",
              first_restored is not None and _cmp_keys(first_restored) == _cmp_keys(src_first),
              f"相同={_cmp_keys(first_restored) == _cmp_keys(src_first)} · "
              f"快照第 1 轮 speedup={(src_first or {}).get('speedup')} → "
              f"恢复后 {(first_restored or {}).get('speedup')}")
    else:
        check("K3", "真崩溃之后续上 (子进程没能撑到第一次落盘, 这一判据本轮没有鉴别力)",
              None, "跳过: 没拿到快照 —— 见 K1 的读数")
        check("K4", "被杀那次的第 1 轮读数原样恢复 (同上, 本轮没有鉴别力)",
              None, "跳过: 没拿到快照")

    # ── D: 反向对照 —— 半个 JSON 不许被当成「没有快照」 ────────────────
    print(f"\n{'=' * 66}\n[D] 把 A 的快照截成半个 JSON 后续跑 (反向对照)\n{'=' * 66}")
    torn_dir = work / "torn"
    torn_dir.mkdir(parents=True, exist_ok=True)
    full = a["snapshot"].read_text(encoding="utf-8") if a["snapshot"] else ""
    (torn_dir / SNAP).write_text(full[: len(full) // 2], encoding="utf-8")
    d = run_arm("D", backend, out_root, work, 1, tid_a, resume_from=str(torn_dir))
    st_d = (d["result"] or {}).get("resume") or {}
    check("D1", "坏快照报 unreadable_snapshot 而不是 no_snapshot / resumed",
          st_d.get("status") == "unreadable_snapshot",
          f"status={st_d.get('status')} reason={st_d.get('reason')} "
          f"size_bytes={st_d.get('size_bytes')}")
    check("D2", "归因带上了异常类名 (远端错误原文不进产物, 但类名必须有)",
          st_d.get("reason") == "JSONDecodeError",
          f"reason={st_d.get('reason')!r}")

    # ── 归属: 跑动窗口内这些文件变过吗? 变过就**不印结论行** ─────────────────
    prov_after = provenance.snapshot_via_subprocess(
        provenance.closure(DRIVER_REL, REPO_ROOT,
                           extra_entries=["remote_scripts/run_kernel.py"]), REPO_ROOT)
    prov_ok = provenance.report(prov_before, prov_after, DRIVER_REL, REPO_ROOT)

    for proc in KIDS:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    if not prov_ok:
        print(f"\n  ❌ 拒绝打印汇总: 上面这些文件在本次跑的窗口内被改 ⇒ 五臂不是同一份代码, "
              f"读数不能归给任何 commit (产物仍在 {work}, 可离线复盘但不能当证据)")
        return 3

    # ── 汇总 ────────────────────────────────────────────────────────
    bad = [j for j in J if j[2] is False]
    na = [j for j in J if j[2] is None]
    print(f"\n{'=' * 66}\n汇总: {len(J) - len(bad) - len(na)} 通过 / {len(bad)} 失败 / "
          f"{len(na)} 没有被问 (n/a)\n{'=' * 66}")
    for idx, what, ok, reading in J:
        mark = "➖" if ok is None else ("✅" if ok else "❌")
        print(f"  {mark} {idx} {what}")
    if na:
        print(f"  ⚠️ n/a 的有: {[j[0] for j in na]} —— 「没被问」不等于「通过」"
              f" (与「拒了 0 条」和「压根没被问」长得一样是同一条纪律)")
    print(f"\n  成本: A/B/C/D 四轮臂的 LLM 调用 = "
          f"{a['calls']}/{b['calls']}/{c_arm['calls']}/{d['calls']} 次")
    print(f"  ⚠️ 这些数字各臂独立取自各自的 LLMClient, 不许相加当成总量")
    print(f"  ⚠️ 有鉴别力的判据: B2/C1/K3/K4/D1 (把 #59 的修复改回旧写法会红); "
          f"A1/A2/B1/B3/K1/K2/D2 里有些在旧代码上同样绿, 它们是前提与反向对照")
    print(f"\n  产物留在 {work} (可离线复盘)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
