"""真机 §14 冒烟 —— 在 H20 上把「标定标定器本身」那一半跑出来

设计 §14 八条的 MockGPU 等价物已经绿了 (`tests/test_stage1_acceptance.py`); 这份脚本补
**真机**那一半。与上一版最大的不同有两条, 都是被实测打出来的:

1. **走生产链路**。上一版自己拼 `oracle.evaluate(...)`, 亲手写 `validity="valid"` ——
   那测的是库, 而真正会失效的是**接线**: 入库判据 `admit_to_corpus` 就这么「有用例、
   零生产调用者」了好几轮 (spine-review ①, 2026-10-01)。现在每一条读数都从
   `Verifier._verify_performance` 出来 (真基线选择 + 交错配对 + 评估 + 资格 + oracle 收口)。
2. **参照核是 triton, 不是 numpy**。上一版三个核全被 `_detect_language` 判成 `numpy`
   ⇒ 量在**远端 CPU** 上 (doc 61 §二②), 于是 §14-1/§14-2 里"关于 GPU"的那两问它答不了;
   占卡进程也是 numpy (`CUDA_VISIBLE_DEVICES` 对 numpy 无效) ⇒ 占的是 CPU 核而不是卡。
   现在参照对用 triton (doc 61 实测 0.1057 ms vs 0.8193 ms = 7.75×), 占卡进程用 torch。

跑法 (需 ~/.ssh/config 里的 algorithm-h20):
    python3 benchmarks/smoke_spine_h20.py
    python3 benchmarks/smoke_spine_h20.py --skip-contention     # 不占卡, 只测 1/2/3/8
    python3 benchmarks/smoke_spine_h20.py --n 1024 --runs 2

覆盖与**边界** (末尾会按同一张表复述一遍, 免得读者把「跑了」当成「八条都验了」):

    §14-1 已知好核      ✅ 真机: 真基线下的 VALID + 跨次 cv
    §14-2 故意写慢      ✅ 真机: SLOW + 慢/好 分离度
    §14-3 库回退假候选  ✅ 真机: 静态审计拦下 + 运行时盾在**远端进程里**拦下
                            (「资格全 False」那一步仍是离线判据, 见段内说明)
    §14-4 后台占卡      ✅ 真机: 真占卡 → DEGRADED, 且不是错误通过
    §14-5 kill 续跑     ⛔ 本脚本不覆盖: 续跑要的是「真跑一次再被杀」, 而那要 LLM
                            (本机没有 LLM_API_KEY ⇒ `Conductor.execute` 起不来)。
                            离线那一半 (`Acceptance5`) 已绿, 真机那一半缺席。
    §14-6 改契约文件    ➖ 纯离线检测器, 真机无额外内容 (`Acceptance6` 已涵盖)
    §14-7 跨卡配对      ⚠️ 见段内: 真机上「一次调用 = 一张卡」是**结构保证**, 而运行时
                            检查 (`pairing_conflict` 进 oracle) 在**生产路径上没人喂**
                            —— 这条今天只有离线检测器。
    §14-8 关掉心跳      ✅ 真机: 到期没跑的窗口 → 资格全 False **且不进语料**
                            (对照臂 = 同一条读数 + 真心跳 → 进语料; 两臂只差心跳)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from operator_agent.anticheat import lib_fallback_metrics, static_audit  # noqa: E402
from operator_agent.backend_caps import Cap  # noqa: E402
from operator_agent.calibration import backend_bench_fn, heartbeat  # noqa: E402
from operator_agent.conductor import Conductor  # noqa: E402
from operator_agent.contract import freeze_from_task  # noqa: E402
from operator_agent.kernel_db import KernelDatabase  # noqa: E402
from operator_agent.measurement import Validity, Verdict, verdict  # noqa: E402
from operator_agent.oracle import admit_to_corpus, evaluate  # noqa: E402
from operator_agent.remote_executor import RemoteExecutor  # noqa: E402
from operator_agent.schemas import (ConstraintSpec, KernelArtifact, Precision,  # noqa: E402
                                    SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
                                    TaskType, VerificationLevel)
from operator_agent.verifier import Verifier  # noqa: E402

# 共享机的落点与拆除纪律 (#69 / test_shared_host_scratch_paths): 随机名 + 读回静默
from _scratch import remote_scratch_path, stop_interferer  # noqa: E402

# 真机参照对只有一处实现 (doc 61 量过 7.75×), 这里**引用**而不是抄一份 —— 抄一份就会
# 在下次修参照对时留下第二个真源。`_Ref` 同样: 它只提供 `__source__` 与形参表。
from measure_heartbeat_real import TRITON_GOOD_SRC, TRITON_SLOW_SRC, _Ref  # noqa: E402

# ── 假候选 (故意回退到库) ────────────────────────────────────────
# A 版: 直白调用 —— 静态审计该拦下 (L2 的第一层)。
LIB_PLAIN = "import torch\n\ndef kernel(A, B):\n    return torch.matmul(A, B)\n"
# B 版: 把名字拼出来 —— 静态审计看不出来, 只有**远端进程里的运行时盾**能拦。
# 这一版才是"真机"那一半: 静态审计在控制端就能跑, 盾要在真卡上跑起来才生效。
LIB_OBFUSCATED = (
    "import torch\n\ndef kernel(A, B):\n"
    '    f = getattr(torch, "mat" + "mul")\n'
    "    return f(A, B)\n"
)

# 结果台账: (条目, 结论, 说明)
LEDGER = []


def check(item: str, ok: bool, detail: str = "") -> bool:
    LEDGER.append((item, "✅" if ok else "❌", detail))
    print(f"  {'✅' if ok else '❌'} {item}: {detail}")
    return ok


def note(item: str, mark: str, detail: str) -> None:
    """只记录结论, 不参与成败 (⚠️/➖/⛔ 那几条)。"""
    LEDGER.append((item, mark, detail))
    print(f"  {mark} {item}: {detail}")


def _task(ref_source: str = "", dsl: str = "triton",
          kernel_type: str = "matmul") -> TaskSchema:
    """真机任务的判据: 必须声明 REAL_DEVICE, 否则 §14-3/§14-8 的资格规则整条不生效
    (`_assess_measurement` 读的正是 `task.target.required_caps`)。"""
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, complexity=TaskComplexity.MEDIUM,
                          reference_source=ref_source),
        target=TargetSpec(hardware="nvidia_gpu", dsl=dsl,
                          required_caps=(Cap.REAL_DEVICE.value,)),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=1.0,
                                   max_error=1e-2),
    )


def _inputs(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {"A": rng.standard_normal((n, n)).astype(np.float32),
            "B": rng.standard_normal((n, n)).astype(np.float32)}


def l3(verifier, code, xs, task, label, heartbeat_result=None, expected=False):
    """**生产** L3 入口。返回 (result, metrics, 墙钟)。

    这就是真机上「一次测量」的全部: 远端选基线 → 交错配对 → 评估三态 → 资格 →
    oracle 收口。下面所有断言读的都是它写出来的键, 不是我们自己算的。
    """
    t0 = time.perf_counter()
    r = verifier._verify_performance(code, xs, 0.0, task,
                                     heartbeat=heartbeat_result,
                                     heartbeat_expected=expected)
    dt = time.perf_counter() - t0
    m = r.metrics or {}
    print(f"\n[{label}] passed={r.passed}  墙钟 {dt:.2f}s")
    if not m:
        # 没有 metrics 的唯一线索就在 message 里 (L3 早期返回, 例如远端建不出基线)
        print(f"  ⚠️ 这一轮没产 metrics: {str(r.message)[:220]}")
        print(f"     error_log={str(getattr(r, 'error_log', ''))[:220]!r}")
    print(f"  candidate={m.get('candidate_median_time_ms')} ms  "
          f"baseline={m.get('baseline_time_ms')} ms ({m.get('baseline_type')})")
    print(f"  speedup={m.get('speedup_vs_baseline')}x  "
          f"validity={m.get('measurement_validity')}  "
          f"device={m.get('device')!r}  language={m.get('language')!r}  "
          f"gpu={m.get('selected_gpu')}")
    print(f"  dispersion(iqr/med)={m.get('dispersion_iqr_over_median')}  "
          f"launch_dominance={m.get('launch_dominance')}  "
          f"util={m.get('gpu_util_pct')}  runs={m.get('runs')}")
    print(f"  eligibility={m.get('eligibility')}")
    if m.get("ruler_broken") is not None:
        print(f"  ruler_broken={m.get('ruler_broken')}  "
              f"reason={m.get('ruler_reason')!r}")
    return r, m, dt


def _burner_command(gpu: int, log_path: str, seconds: int) -> str:
    """远端占卡命令 —— **真的占卡** (torch, 不是 numpy)。

    两条都是实测打出来的:

    * 上一版用 numpy: `CUDA_VISIBLE_DEVICES` 对 numpy 无效 ⇒ 它占的是 CPU 核, 不是卡,
      于是 §14-4 那一臂"造出的竞争"是假的 (util 没涨), 而脚本只会打印一句"这一臂不算数"。
    * 上一版的循环写成 `exec("while ...:\n  x = a @ a")` —— 那个 `\n` 经 ssh → bash
      单引号 → `python -c` 之后变成了**字面换行**, 而 Python 不许单行字符串字面量里有真
      换行 ⇒ 远端 `SyntaxError`, 占位一启动就死 (真机实测: util 序列 23 个 0, 日志第一
      行就是那个 SyntaxError)。所以这里直接用**多行 `-c` 脚本**, 让缩进本身表达循环体,
      并且有一条离线棘轮 (`test_shared_host_scratch_paths.py`) 拿 `ast.parse` 解析它。

    落点仍来自 `remote_scratch_path` (随机名, 不在世界可写区留固定名), argv 里带
    `MARK=<落点本体>`; 拆除的 pkill/pgrep/rm 三个动作全靠这一个不可预测路径定位。
    自带截止时间 (_scratch 的纪律: 远端进程必须有自我期限)。
    """
    body = (
        f'MARK="{log_path}";import time, torch\n'
        f'print("cuda_devices", torch.cuda.device_count(), flush=True)\n'
        f'a = torch.randn(4096, 4096, device="cuda")\n'
        f't0 = time.time()\n'
        f'n = 0\n'
        f'while time.time() - t0 < {seconds}:\n'
        f'    x = a @ a\n'
        f'    n += 1\n'
        f'print(n, flush=True)'
    )
    # 第三处实测: `nohup ... &` 还不够 —— ssh 通道仍被那个后台进程占着, 启动命令**卡住**
    # 30 秒直到 `_run_remote` 超时 (而超时会把整个脚本掀翻)。要真脱离: `setsid` 另起会话,
    # stdin 归 `/dev/null`, 整条放进子 shell 里后台化, 最后回一句 `launched` 让调用方有回执。
    # 第四处实测: `CUDA_VISIBLE_DEVICES=N cd X && ... python` 里那个赋值是**前缀赋值**,
    # 只作用于 `cd` 这一条命令 —— `cd` 又是内建命令, 于是变量**根本没进 shell 环境**,
    # python 落在 **GPU 0** 上: 真机现场看到的是「卡 2 的 util 序列 23 个 0」(那时占位
    # 正在卡 0 上跑), 而拆除又报 `quiescent` —— 一切自洽, 就是没测到东西。改 `env` 显式
    # 传环境, 并让占位自己把 `device_count()` 打进日志 (钉住了就是 1, 没钉住是 8)。
    return (f"cd ~/operator_agent && "
            f"(setsid env CUDA_VISIBLE_DEVICES={gpu} .venv/bin/python -c '{body}' "
            f"> {log_path} 2>&1 < /dev/null &) ; echo launched")


def _gpu_util(alias: str, gpu: int) -> Optional[int]:
    """那张卡当前的 util% (读不到返回 None)。"""
    out = _run_remote(f"nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits "
                      f"-i {gpu}", alias=alias, timeout=20)
    for tok in out.split():
        if tok.strip().rstrip("%").isdigit():
            return int(tok.strip().rstrip("%"))
    return None


def _wait_for_contention(alias: str, gpu: int, log_path: str, timeout: float = 60.0) -> list:
    """等到那张卡**真的忙起来**再测 —— 否则测到的是「占位进程还在 import torch」。

    上一版起完占位只 `sleep(2)` 就测, 真机实测 (2026-10-01) 拿到 util=0 与
    `validity=valid`: torch 的 import + CUDA 上下文 + 4096³ 的 randn 常常超过 2 秒,
    于是竞争臂量的是安静窗口 —— 而脚本只会打印一句「这一臂不算数」, 与「占位崩了」
    分不开。这里读回 util 序列当证据 (也顺带把占位进程的异常暴露出来)。
    """
    seq = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        u = _gpu_util(alias, gpu)
        seq.append(u)
        if u is not None and u > 20:
            break
        time.sleep(2.0)
    print(f"  竞争窗口 util 序列: {seq}")
    if not seq or seq[-1] is None or seq[-1] <= 20:
        # 占位没忙起来: 把它的日志读回来 (崩了的话第一行就是 traceback)
        tail = _run_remote(f"head -c 400 {log_path}; echo; wc -l < {log_path}",
                           alias=alias, timeout=20)
        print(f"  ⚠️ 占位没让卡忙起来, 远端日志: {tail.strip()[:300]!r}")
        if "cuda_devices" in tail and "cuda_devices 1" not in tail:
            print("     ↑ `cuda_devices` 不是 1 ⇒ CUDA_VISIBLE_DEVICES **没钉住**, "
                  "占位跑在别的卡上 (这一臂量的是安静窗口)")
    return seq


def _run_remote(cmd: str, alias: str = "algorithm-h20", timeout: int = 30) -> str:
    """远端命令**永远不许掀翻脚本**: 超时回一个显式标记, 让现场诊断继续往下走。

    为什么 (2026-10-01 实测): 占位启动那条命令把 ssh 通道占住 30 秒, `TimeoutExpired`
    直接把整个冒烟脚本掀翻 —— 而它已经跑完的 §14-1/§14-2/§14-3 读数与后面的诊断全部作废。
    一次远端卡住是**要记的事实**, 不是要命的错误 (真错误由后面的断言报)。
    """
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", alias, cmd],
                           check=False, capture_output=True, timeout=timeout, text=True)
    except subprocess.TimeoutExpired:
        print(f"  ⚠️ 远端命令 {timeout}s 未返回 (通道被后台进程占住?): {cmd[:80]}…")
        return "__REMOTE_TIMEOUT__"
    return (r.stdout or "") + (r.stderr or "")


def _repo_corpus_lines() -> int:
    p = ROOT / "output" / "kernel_db.jsonl"
    return len(p.read_text(encoding="utf-8").splitlines()) if p.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048, help="方阵边长 (默认 2048)")
    ap.add_argument("--runs", type=int, default=3, help="§14-1 重复次数 (跨次 cv)")
    ap.add_argument("--skip-contention", action="store_true", help="不跑 §14-4 (不占卡)")
    ap.add_argument("--burn-seconds", type=int, default=90, help="占卡进程的自我期限")
    args = ap.parse_args()

    print("[smoke] 连接 algorithm-h20 …")
    backend = RemoteExecutor(mode="ssh", timeout=600)
    print(f"[smoke] backend={backend.label}  gpu={getattr(backend, 'selected_gpu', '?')}")
    # 前提门: 没有这两个能力, 下面的读数不是「同卡同窗」的, 整轮无意义
    for cap in (Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE):
        if cap not in backend.capabilities:
            print(f"❌ 后端不具备 {cap.value} —— 真机冒烟的前提不成立")
            return 1
    if not backend.deploy():
        print("❌ 部署失败")
        return 1

    verifier = Verifier(backend)
    task = _task()
    xs = _inputs(args.n)
    contract = freeze_from_task(task)

    print(f"\n判据: {contract.hash[:16]}…  n={args.n}³  fp32  "
          f"REAL_DEVICE 必需  same_device={contract.baseline.same_device}")

    # ── §14-0 前提: 已知能编译的核必须在远端编译成功 ────────────────
    print("\n" + "=" * 66 + "\n§14-0 前提: 已知好核在远端编译\n" + "=" * 66)
    ctrl = verifier._verify_compile(TRITON_GOOD_SRC, dsl="triton",
                                    input_key_sets=[set(xs.keys())])
    if not check("§14-0 前提 (远端编译)", bool(ctrl.passed), f"passed={ctrl.passed}"):
        print("  已知能编译的核编不过 ⇒ 远端环境/部署不同步, 后面的读数都不算数")
        return 1

    # ── §14-1 已知好核 → VALID + 跨次 cv ───────────────────────────
    print("\n" + "=" * 66 + f"\n§14-1 已知好核 (triton matmul @{args.n}³, {args.runs} 次)\n"
          + "=" * 66)
    good = []
    for i in range(args.runs):
        r, m, _ = l3(verifier, TRITON_GOOD_SRC, xs, task, f"§14-1 run{i + 1}")
        good.append((r, m))
    validities = [m.get("measurement_validity") for _, m in good]
    check("§14-1 档位 VALID", all(v == "valid" for v in validities),
          f"validity={validities}")
    meds = [m.get("candidate_median_time_ms") for _, m in good
            if m.get("candidate_median_time_ms")]
    if len(meds) >= 2:
        med = float(np.median(meds))
        mad = float(np.median(np.abs(np.array(meds) - med)))
        cv = mad / med if med else float("inf")
        print(f"  跨次读数: {['%.4f' % x for x in meds]} ms  median={med:.4f}  "
              f"mad/med={cv:.4f}")
        check("§14-1 跨次不散 (mad/med < 0.05)", cv < 0.05, f"mad/med={cv:.4f}")
    else:
        check("§14-1 跨次不散", False, "拿不到两次以上候选读数")
    # oracle 收口真的产了观测 (链路上有东西, 不是没接线)
    obs1 = (good[0][1].get("observation") or {})
    check("§14-1 收口点产了 Observation", bool(obs1),
          f"validity={obs1.get('validity')} hash_check={obs1.get('hash_check')!r}")
    check("§14-1 观测可判决 (can_verdict)",
          bool((obs1.get("eligibility") or {}).get("can_verdict")),
          f"{obs1.get('eligibility')}")

    # ── §14-2 故意写慢 → SLOW ─────────────────────────────────────
    print("\n" + "=" * 66 + "\n§14-2 故意写慢的核 (同一 launcher 连跑 8 遍)\n" + "=" * 66)
    r_slow, m_slow, _ = l3(verifier, TRITON_SLOW_SRC, xs, task, "§14-2 slow")
    slow_ms = m_slow.get("candidate_median_time_ms")
    good_ms = good[0][1].get("candidate_median_time_ms")
    sp_slow = m_slow.get("speedup_vs_baseline")
    if slow_ms and good_ms:
        ratio = slow_ms / good_ms
        print(f"  慢/好 = {ratio:.2f}×  (期望 ≥2; 加速比 {sp_slow}x)")
        check("§14-2 慢核确实慢 (≥2×)", ratio >= 2.0, f"{ratio:.2f}×")
        # verdict 收的是「越大越好」的轴 (docstring: 同轴, 越大越好) ⇒ 换成吞吐 1/ms
        thr_slow, thr_good = 1000.0 / slow_ms, 1000.0 / good_ms
        v, why, _ = verdict(current=thr_slow, previous=thr_good, validity=Validity.VALID)
        print(f"  verdict(slow vs good) = {v.value}  ({why[:90]})")
        check("§14-2 判决 SLOW", v is Verdict.SLOW, f"verdict={v.value}")
        check("§14-2 没被记成通过", not r_slow.passed,
              f"passed={r_slow.passed} (加速比 {sp_slow}x < 1.0)")
    else:
        check("§14-2 慢核确实慢", False, "拿不到读数")

    # ── §14-3 库回退假候选 ────────────────────────────────────────
    print("\n" + "=" * 66 + "\n§14-3 回退到库的假候选 (两层反作弊)\n" + "=" * 66)
    stat = static_audit(LIB_PLAIN, dsl="triton")
    check("§14-3 静态审计看得见直白调用", bool(stat), f"{stat[:2]}")
    stat_obf = static_audit(LIB_OBFUSCATED, dsl="triton")
    print(f"  拼名字那版的静态审计结果: {stat_obf or '（空 —— 这正是需要运行时盾的原因）'}")
    for label, code, expect_static in (("直白", LIB_PLAIN, True),
                                       ("拼名字", LIB_OBFUSCATED, False)):
        res = verifier.verify(code, task, xs, target_gflops=0.0,
                              stop_after=VerificationLevel.L2_FUNCTION)
        passed = all(r.passed for r in res)
        last = res[-1]
        mm = last.metrics or {}
        print(f"  [{label}] 生产 L1+L2: passed={passed}  {last.message[:100]}")
        if mm.get("lib_fallback"):
            print(f"     lib_fallback={mm.get('lib_fallback')}  "
                  f"evidence={mm.get('lib_fallback_evidence')!r}")
        check(f"§14-3 {label}版被拦下", not passed, f"passed={passed}")
        if not expect_static:
            # 「拼名字」这版静态审计看不见 ⇒ 拦下它的只能是**远端进程里**的运行时盾,
            # 这是本条目唯一真机独有的那一半。
            check("§14-3 拦下它的是运行时盾 (非静态)",
                  bool(mm.get("lib_fallback"))
                  and "runtime:" in str(mm.get("lib_fallback_evidence", "")),
                  f"evidence={mm.get('lib_fallback_evidence')!r}")
    # 「资格全 False」那一步: 生产里假候选在 L2 就被拦, 根本到不了 L3 ⇒ 这一步没有
    # 真机形态, 只有判据形态。用生产同一个生产者 (`lib_fallback_metrics`) 造输入。
    fb = lib_fallback_metrics(stat)
    obs_fb = evaluate(contract, lambda: {"value": 5.0, "validity": "valid",
                                         "real_device": True, **fb},
                      record_hash=contract.hash, require_real_device=True)
    ok_fb, why_fb = admit_to_corpus(
        {"speedup_vs_baseline": 5.0, "measurement_validity": "valid",
         "measurement_can_gate": True, "backend_real_device": True,
         "eligibility": obs_fb.eligibility.as_dict(),
         "observation": obs_fb.as_dict()},
        None, requires_remeasure=False)
    note("§14-3 资格全 False / 拒入库", "➖",
         f"离线判据 (真机到不了这里): can_persist="
         f"{obs_fb.eligibility.can_persist} admit={ok_fb} · {why_fb[:60]}")

    # ── §14-4 后台占卡 → DEGRADED ─────────────────────────────────
    if args.skip_contention:
        note("§14-4 后台占卡", "➖", "--skip-contention: 未跑 (不占卡)")
    else:
        print("\n" + "=" * 66 + "\n§14-4 后台真占卡 → DEGRADED (不是错误通过)\n"
              + "=" * 66)
        gpu = getattr(backend, "selected_gpu", None) or 0
        log_path = remote_scratch_path("burner.log")
        print(f"  起远端占卡: GPU {gpu} 上 torch 4096³ matmul ×{args.burn_seconds}s "
              f"(日志 {log_path})")
        try:
            started = _run_remote(_burner_command(gpu, log_path, args.burn_seconds))
            if started.strip():
                print(f"  占位命令输出: {started.strip()[:200]!r}")
            _wait_for_contention("algorithm-h20", gpu, log_path)
            r4, m4, _ = l3(verifier, TRITON_GOOD_SRC, xs, task, "§14-4 contention")
            util = m4.get("gpu_util_pct")
            others = m4.get("gpu_other_procs")
            contaminated = m4.get("benchmark_contaminated")
            print(f"  util={util}  other_procs={others}  "
                  f"contaminated={contaminated}")
            check("§14-4 真的造出了竞争 (util > 5)", bool(util and util > 5),
                  f"util={util}")
            check("§14-4 降为 DEGRADED",
                  m4.get("measurement_validity") == "degraded",
                  f"validity={m4.get('measurement_validity')}")
            check("§14-4 不是错误通过 (can_persist=False)",
                  not (m4.get("eligibility") or {}).get("can_persist", True),
                  f"eligibility={m4.get('eligibility')}")
        finally:
            # 拆除要读到静默 (#69 / DSH :19): pkill→pgrep→rm 对着同一个不可预测路径
            ev = stop_interferer(None, _run_remote, log_path)
            print(f"  拆除: {ev}")

    # ── §14-7 跨卡配对: 事实与缺口 ────────────────────────────────
    print("\n" + "=" * 66 + "\n§14-7 跨卡配对\n" + "=" * 66)
    sel = good[0][1].get("selected_gpu")
    print(f"  真机事实: 同机基准协议**一次调用**里跑完候选与基线, 载荷只报一个 "
          f"selected_gpu={sel}")
    print(f"            (benchmark_with_baselines: candidate/baselines 同在一个返回里, "
          f"两臂没有各自的卡号键)")
    print("  运行时检查侧: `pairing_conflict` 在 oracle 里 (设计 §11-2), 输入是"
          " `candidate_device`/`baseline_device`")
    obs_x = evaluate(contract, lambda: {"value": 1.5, "validity": "valid",
                                        "real_device": True},
                     record_hash=contract.hash,
                     candidate_device="gpu:0", baseline_device="gpu:1")
    check("§14-7 跨卡观测被拒 (检测器)", bool(obs_x.refused), f"{obs_x.refused[:80]}")
    # 缺口: 生产收口点**不传**这两个参数 ⇒ 这条检查在生产上从不执行。
    # 判据是**整个 verifier 模块**: 那两个名字只可能出现在喂给点上, 一处都没有就是没有。
    import inspect
    import operator_agent.verifier as _vf
    src = inspect.getsource(_vf)
    fed = "candidate_device" in src or "baseline_device" in src
    note("§14-7 运行时检查的生产喂给者", "⚠️",
         f"verifier 的 oracle 调用{'传了' if fed else '**没传**'}设备名 ⇒ "
         f"真机上「两臂同卡」是**结构保证**(一次调用一张卡), 不是这条检查抓到的; "
         f"检查今天只有离线检测器")

    # ── §14-8 关掉心跳 → 该窗口观测不可用 + 不写记忆 ───────────────
    print("\n" + "=" * 66 + "\n§14-8 到期没跑心跳 → 资格全 False + 不进语料\n" + "=" * 66)
    workdir = tempfile.mkdtemp(prefix="smoke14_kdb_")
    kernel_jsonl = str(Path(workdir) / "kernel_db.jsonl")
    my_db = KernelDatabase(jsonl_path=kernel_jsonl)
    cond = Conductor(backend=backend, kernel_db=my_db, llm=None)
    # J0 前提门 (同 `verify_corpus_honesty`): 注入的语料必须就是被写的那一个, 否则
    # 这一臂会污染仓库语料而自己的记账判据全绿 (那个探针真踩过)。
    if cond.kernel_db is not my_db or str(cond.kernel_db._jsonl) != kernel_jsonl:
        check("§14-8 J0 前提 (语料落点)", False, "注入的库不是被写的库 —— 停")
        return 1
    check("§14-8 J0 前提 (语料落点)", True, f"隔离语料 {kernel_jsonl}")
    repo_before = _repo_corpus_lines()

    # 两臂**只差心跳**。基线尽量换成慢参照核 (任务声称一个没有内建基线的算子 +
    # 自带 reference_source): 远端建不出内建基线时, 它才拿参照核当基线 —— 这样加速比
    # 才有机会 >1.0x, 让 `persist_decision` 那道门不成为混淆因素。真机实测 (2026-10-01)
    # 对 `matmul` 是**做不到**的: 基线偏好名单永远非空 (`baseline_plan` 对未知算子也回
    # `pytorch_eager`), 而 triton fp32 打不过 TF32 cuBLAS (0.416x) ⇒ 语料级的对照要
    # 条件化 (见下), 判据级的那条断言才是本条目真正的判据。
    # 任务就按最普通的那条走。**试过并放弃**的写法: 声称一个没有内建基线的算子 +
    # `reference_source=慢参照核`, 想让候选跑赢 —— 真机实测远端直接建不出基线, L3 早期
    # 返回、metrics 为空 (§14-8 两臂都只剩一句「远程性能测试失败」)。空 metrics 还会让
    # 「发布值 == 收口点」那条断言**空转绿**, 所以下面每条断言都先要求观测真的存在。
    task_ref = _task()

    # ── 负臂: 到期没跑心跳 ────────────────────────────────────────
    r_broken, m_broken, _ = l3(verifier, TRITON_GOOD_SRC, xs, task_ref,
                               "§14-8 无心跳 (到期没跑)", heartbeat_result=None,
                               expected=True)
    check("§14-8 记录里标了尺子坏", bool(m_broken.get("ruler_broken")),
          f"ruler_broken={m_broken.get('ruler_broken')} "
          f"reason={m_broken.get('ruler_reason')!r}")
    check("§14-8 该窗口资格全 False",
          not any((m_broken.get("eligibility") or {}).values()),
          f"{m_broken.get('eligibility')}")
    # 反空转: 没有观测时这一条会以 None == None 通过 —— 那正是「守卫静默通过」的样本
    obs_broken = m_broken.get("observation") or {}
    check("§14-8 收口点真的产了观测", bool(obs_broken),
          f"observation={'有' if obs_broken else '缺'} message={str(r_broken.message)[:80]!r}")
    check("§14-8 发布的资格位 == 收口点算的那份",
          bool(obs_broken) and m_broken.get("eligibility") == obs_broken.get("eligibility"),
          f"发布={m_broken.get('eligibility')} 收口点={obs_broken.get('eligibility')}")
    keep_broken, why_broken = admit_to_corpus(m_broken, None, requires_remeasure=False)
    art_broken = KernelArtifact(kernel_id="k-broken-window", code=TRITON_GOOD_SRC,
                                verification_results=[r_broken], benchmark_data=m_broken)
    saved_broken = cond._save_to_db(art_broken, task_ref)
    n_broken = len(my_db._records)
    check("§14-8 入库判据拒了它", not keep_broken and not saved_broken,
          f"admit={keep_broken} _save_to_db={saved_broken} 语料={n_broken}")
    # 边界 (明写, 不假装成绿): 入库判据里**速度门先于资格门**, 而那道门把 1.0x 写死在
    # `measurement.persist_decision` (`speedup <= 1.0` ⇒ 拒, 不吃调用方给的 `min_speedup`)
    # ⇒ 这台机器上 (triton fp32 0.42x vs TF32 cuBLAS) **拒因永远先落在速度门那一条**。
    # 「拒因点名尺子」这一形状在这台机器上不可达; 要看到它, 需要一条真机跑得赢的读数。
    note("§14-8 拒因为何不点名尺子", "➖",
         f"速度门 (写死 >1.0x) 先拒: {why_broken[:58]!r} —— 坏窗口的资格位确实全 False "
         f"(上面已断言), 但入库判据在它之前先被速度门挡住, 所以这一条**归因不唯一**")

    # ── 对照臂: 同一条读数 + **真**心跳 (doc 61 量过: 真机 ok=True) ──
    hb = heartbeat(backend_bench_fn(backend), epoch=1, window_id="smoke-ok",
                   good_kernel=_Ref("known_good_kernel", TRITON_GOOD_SRC),
                   slow_kernel=_Ref("known_slow_kernel", TRITON_SLOW_SRC))
    print(f"  真心跳: ok={hb.ok} 好={hb.known_good_ratio} 慢/好={hb.known_slow_ratio} "
          f"reasons={list(hb.reasons)}")
    check("§14-8 对照臂: 真心跳可用", bool(hb.ok), f"ok={hb.ok}")
    r_ok, m_ok, _ = l3(verifier, TRITON_GOOD_SRC, xs, task_ref,
                       "§14-8 有心跳 (对照)", heartbeat_result=hb, expected=True)
    elig_ok = m_ok.get("eligibility") or {}
    print(f"  对照臂 eligibility={elig_ok}  baseline={m_ok.get('baseline_type')}")
    check("§14-8 对照臂: 该窗口资格位不再被尺子清空",
          bool(m_ok.get("observation")) and bool(elig_ok.get("can_persist")),
          f"can_persist={elig_ok.get('can_persist')} "
          f"ruler_broken={m_ok.get('ruler_broken')} 观测={'有' if m_ok.get('observation') else '缺'}")
    # 两臂的**唯一**差别是心跳: 负臂被资格位挡住, 对照臂没有被资格位挡住。
    keep_ok, why_ok = admit_to_corpus(m_ok, None, requires_remeasure=False)
    art_ok = KernelArtifact(kernel_id="k-good-window", code=TRITON_GOOD_SRC,
                            verification_results=[r_ok], benchmark_data=m_ok)
    saved_ok = cond._save_to_db(art_ok, task_ref)
    n_ok = len(my_db._records)
    print(f"  对照臂: admit={keep_ok} _save_to_db={saved_ok} 语料={n_ok} "
          f"理由={why_ok[:90]!r}")
    # §14-8 能证的那一半: **判据读的那一位** (发布出去的 eligibility) 两臂相反 —— 两臂只差
    # 心跳, 同一台机器同一张卡同一条链。语料级的后果 (坏窗口那条不进语料) 在这台机器上
    # 归因不唯一 (速度门先拒), 已明写在上面那条 note 里。
    check("§14-8 只换心跳: 判据读的那一位两臂相反",
          (not (m_broken.get("eligibility") or {}).get("can_persist"))
          and bool(elig_ok.get("can_persist")),
          f"坏窗口 can_persist={(m_broken.get('eligibility') or {}).get('can_persist')} | "
          f"好窗口 can_persist={elig_ok.get('can_persist')}")
    if not (m_ok.get("speedup_vs_baseline") or 0) > 1.0:
        note("§14-8 语料级对照 (>1.0x 才可判)", "➖",
             f"真机候选打不过这条基线 (speedup={m_ok.get('speedup_vs_baseline')}x) "
             f"⇒ 门 1 先拒, 两臂都不会进语料, 这一条**分不出**是尺子的功劳; "
             f"判据级那条 (上面) 才是本条目真正的判据")
    else:
        check("§14-8 语料级对照: 合格读数进语料", bool(saved_ok) and n_ok == 1,
              f"入库={saved_ok} 语料={n_ok}")
        check("§14-8 语料级对照: 坏窗口那条没进", n_broken == 0, f"语料={n_broken}")
    check("§14-8 仓库语料没被碰", _repo_corpus_lines() == repo_before,
          f"{repo_before} → {_repo_corpus_lines()}")

    # ── 台账 ──────────────────────────────────────────────────────
    print("\n" + "=" * 66 + "\n§14 真机冒烟台账\n" + "=" * 66)
    for item, mark, detail in LEDGER:
        print(f"  {mark} {item:<34} {detail}")
    fails = [i for i, m, _ in LEDGER if m == "❌"]
    print(f"\n  失败 {len(fails)} 条" + (f": {fails}" if fails else ""))
    print("  边界: §14-5 kill 续跑未覆盖 (需 LLM 起一次真跑); §14-6 纯离线;")
    print("        §14-7 的运行时检查在生产上没有喂给者 (见上)。")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())