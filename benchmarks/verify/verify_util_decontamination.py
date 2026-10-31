"""真机复核: util 快照去污染到底挡住了什么 (docs/research/34)

在控制端运行 (会触发 deploy, 需要能连到 GPU 机):
    python3 benchmarks/verify/verify_util_decontamination.py
    python3 benchmarks/verify/verify_util_decontamination.py --reps 20

背景 (docs/research/33 第三节): `nvidia-smi` 的 utilization.gpu 是过去约 1 秒窗口的
占比, 而 L1/L2/L3 是三次独立的 ssh 调用 —— 在「测量前」采样挡得住本进程的核, 挡不住
**上一个 runner 进程**的尾巴。实测在一台真空闲的卡上紧挨着自己一轮 benchmark 之后采,
能读到 10-34%。这个假阳性直接把加速比判成不可比, Conductor 因此不计停滞、不禁用配方,
闭环静默失效。

修法是把采样改成「隔 1.2 s 采两次取 min」。但只留 min 的话, 生产里没人能判断这次
到底是「自己的残留被排掉了」还是「邻居真的在跑、被 min 掩盖了」—— 所以 runner 现在
把两次原始读数一起回报 (`gpu_util_samples`)。

这个脚本要的就是那份可审计性, 三件事全部用真机数据判:

  1. **触发率**: 连跑 N 次 L3, `benchmark_contaminated` 现在多久响一次?
  2. **反事实**: 同一批读数下, 旧的「单次快照」规则会响多少次?
     (旧规则 = 取 first; 若 first 明显高于 second 的轮次就是自污染, 两者之差就是
      这次修复省下的假阳性数)
  3. **配对散布分布**: D3 新增的 `paired_speedup_dispersion` 在本机的真实分布 ——
     D4 (四态判决) 的阈值要按它标, 不能沿用锁频独占机上的 0.5%/2%。
  4. **跨轮 SM 时钟** (D2 的第一步, 只观测不判定): runner 现在在计时窗口前后各读一次
     `clocks.sm` / `clocks_throttle_reasons.active`。要看的是**跨轮**时钟散布, 因为
     Conductor 的改善判据比的是隔几分钟的两轮 gflops, 而加速比是同窗口内之比
     (时钟涨落对后者几乎无影响, 两边同涨同落)。关键产出是一个可判的数:
     「gflops 跨轮 iqr/median」对比「gflops/时钟 的 iqr/median」—— 归一化收得下多少
     散布, 决定 D2 是做「阈值降级」还是做「除法归一化」。

形状选两个: matmul 4096³ 是「valid」档 (单样本 ms 级, 自污染窗口相对短),
layernorm 1024×768 是 launch 主导档 —— 自动加采样会连着跑几千个微秒级 kernel,
是自己的尾巴最容易盖住读数的那一类。
"""
import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from operator_agent.decide.measurement import CONTAMINATION_UTIL_PCT      # noqa: E402
from operator_agent.execute.remote_executor import RemoteExecutor          # noqa: E402
from operator_agent.decide.verifier import Verifier                       # noqa: E402

TRITON_LN = '''
import torch, triton, triton.language as tl

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

TRITON_MM = '''
import torch, triton, triton.language as tl

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
        sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BM + tl.arange(0, BM)
    off_n = pid_n * BN + tl.arange(0, BN)
    a_ptrs = a_ptr + off_m[:, None] * sam + tl.arange(0, BK)[None, :] * sak
    b_ptrs = b_ptr + tl.arange(0, BK)[:, None] * sbk + off_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    tl.store(c_ptr + off_m[:, None] * scm + off_n[None, :] * scn, acc.to(tl.float16))

def kernel(A, B):
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    BM, BN, BK = 64, 64, 32
    _mm[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), BM=BM, BN=BN, BK=BK)
    return C
'''


def ln_inputs(rows, cols):
    rng = np.random.default_rng(11)
    return {"X": rng.standard_normal((rows, cols)).astype(np.float16),
            "weight": np.ones(cols, dtype=np.float16),
            "bias": np.zeros(cols, dtype=np.float16)}


def mm_inputs(n):
    rng = np.random.default_rng(7)
    return {"A": (rng.standard_normal((n, n)) * 0.1).astype(np.float16),
            "B": (rng.standard_normal((n, n)) * 0.1).astype(np.float16)}


def make_task(kernel_type):
    from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec,
                                        TargetSpec, TaskComplexity, TaskSchema,
                                        TaskType)
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton",
                          compute_units=78, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP16, min_speedup=0.0,
                                   max_error=1e-1))


def dist(xs):
    a = np.asarray(sorted(xs), dtype=np.float64)
    q1, q3 = np.percentile(a, [25, 75])
    return (float(np.median(a)), float(q3 - q1), float(a[0]), float(a[-1]))


def report(label, runs):
    n = len(runs)
    print(f"\n===== {label}  (n={n}) =====")
    print(f"{'#':<4}{'两次读数':<14}{'min':<6}{'others':<8}{'污染?':<7}"
          f"{'旧规则会响?':<14}{'加速比':<10}{'配对散布':<10}{'档':<14}{'SM钟前/后':<14}")
    for i, r in enumerate(runs):
        old = "是" if r["first"] is not None and r["first"] > CONTAMINATION_UTIL_PCT else "否"
        now = "是" if r["contaminated"] else "否"
        paired = "-" if r["paired"] is None else f"{r['paired']:.4f}"
        clk = f"{r.get('sm_b')}/{r.get('sm_a')}"
        print(f"{i:<4}{str(r['samples']):<14}{str(r['min']):<6}{str(r['others']):<8}"
              f"{now:<7}{old:<14}{r['speedup']:<10.4f}{paired:<10}{r['validity']:<14}{clk:<14}")

    contam = sum(1 for r in runs if r["contaminated"])
    old_hit = sum(1 for r in runs
                  if r["first"] is not None and r["first"] > CONTAMINATION_UTIL_PCT)
    resid = sum(1 for r in runs if r["first"] is not None and r["second"] is not None
                and r["first"] > r["second"])
    print(f"\n  判污染轮数: 现规则(min) {contam}/{n} = {contam / n:.0%}   "
          f"旧规则(单次快照 first) {old_hit}/{n} = {old_hit / n:.0%}")
    print(f"  first > second 的轮数 (自污染签名): {resid}/{n} = {resid / n:.0%}")
    deltas = [r["first"] - r["second"] for r in runs
              if r["first"] is not None and r["second"] is not None]
    if deltas:
        print(f"  first-second 中位数 {statistics.median(deltas):+.1f} 个百分点, "
              f"最大 {max(deltas):+d}")
    sp = [r["speedup"] for r in runs if r["speedup"]]
    pd = [r["paired"] for r in runs if r["paired"] is not None]
    if sp:
        m, iqr, lo, hi = dist(sp)
        print(f"  加速比: median={m:.4f}  iqr={iqr:.4f} ({iqr / m:.2%})  范围 {lo:.4f}~{hi:.4f}")
    if pd:
        m, iqr, lo, hi = dist(pd)
        print(f"  paired_speedup_dispersion: median={m:.4f}  iqr={iqr:.4f}  "
              f"范围 {lo:.4f}~{hi:.4f}")
        print(f"    → D4 阈值该按这条分布定, 而不是独占锁频机的 0.005/0.02")
    report_clocks(runs)


def report_clocks(runs):
    """D2 的观测段: SM 时钟到底跨不跨轮漂, 以及归一化能不能把散布收下来

    为什么这是 D2 的关键量而不是「窗口内稳不稳」: Conductor 的改善判据
    (`best_gflops` vs 本轮 gflops) 比的是**隔了几分钟的两轮**测量, 而加速比是
    同一窗口内候选/基线之比 —— 时钟涨落对后者几乎无影响 (两边同涨同落),
    对前者是直接的系统误差。所以要看的是跨轮时钟散布, 以及 gflops/时钟 是不是
    比 gflops 本身稳。
    """
    sm_b = [r["sm_b"] for r in runs if r.get("sm_b")]
    if not sm_b:
        print("  SM 时钟: runner 没回报 (nvidia-smi 查询失败或字段不支持) —— D2 无从判断")
        return
    sm_a = [r["sm_a"] for r in runs if r.get("sm_a")]
    mx = [r["sm_max"] for r in runs if r.get("sm_max")]
    m, iqr, lo, hi = dist(sm_b)
    print(f"\n  SM 时钟 (窗口起点读数): median={m:.0f} MHz  范围 {lo:.0f}~{hi:.0f}  "
          f"iqr/median={iqr / m:.2%}")
    if mx:
        print(f"    标称上限 clocks.max.sm = {statistics.median(mx):.0f} MHz  "
              f"→ 实际/上限 = {m / statistics.median(mx):.1%}")
    inside = [a - b for a, b in zip(sm_a, sm_b)] if len(sm_a) == len(sm_b) else []
    if inside:
        print(f"    窗口内漂移 (after − before): median={statistics.median(inside):+.0f} MHz  "
              f"最大 |Δ|={max(abs(x) for x in inside)} MHz")
    # ⚠️ 不能把 None 折成 0 再数「非零」: 「throttle 位全 0」和「throttle 位没读到」
    # 会输出同一行字。probe_clock_dvfs.py 控制 0 已确认本驱动这个字段可解析, 但那只是
    # 换机器时的假设 —— 这里按「读到/没读到」分开报。
    got = [r.get("thr_b") for r in runs] + [r.get("thr_a") for r in runs]
    have = [b for b in got if b is not None]
    nz = sorted({b for b in have if b})
    print(f"    throttle 位图: 读到 {len(have)}/{len(got)} 个读数  "
          f"非零 {sum(1 for b in have if b)}/{len(have)}  出现过的位={nz}  "
          f"(0x1=GPU idle, 通常无害)")
    if len(have) < len(got):
        print(f"      ⚠️ 有 {len(got) - len(have)} 次没读到 → 「全 0」不能当成「没有被限频」")

    pairs = [(r["sm_b"], r["gflops"]) for r in runs
             if r.get("sm_b") and r.get("gflops")]
    if len(pairs) >= 4:
        # 按窗口起点时钟分高低两组: 若 gflops 随时钟走, 跨轮比 gflops 就是拿两个口径比
        pairs.sort(key=lambda p: p[0])
        half = len(pairs) // 2
        lo_g = pairs[:half]
        hi_g = pairs[half:]
        lo_clk = statistics.median([p[0] for p in lo_g])
        hi_clk = statistics.median([p[0] for p in hi_g])
        lo_gf = statistics.median([p[1] for p in lo_g])
        hi_gf = statistics.median([p[1] for p in hi_g])
        print(f"    低钟组 {lo_clk:.0f}MHz → {lo_gf:.0f} GFLOPS   "
              f"高钟组 {hi_clk:.0f}MHz → {hi_gf:.0f} GFLOPS   "
              f"比值 {hi_gf / lo_gf:.4f} vs 时钟比 {hi_clk / lo_clk:.4f}")
        raw = [g for _, g in pairs]
        norm = [g / c for c, g in pairs]
        _, iqr_raw, _, _ = dist(raw)
        _, iqr_norm, _, _ = dist(norm)
        rel_raw = iqr_raw / statistics.median(raw)
        rel_norm = iqr_norm / statistics.median(norm)
        print(f"    跨轮散布: gflops iqr/median={rel_raw:.2%}   "
              f"gflops/时钟={rel_norm:.2%}")
        if rel_raw > 0:
            print(f"      → 按时钟归一化收下 {(1 - rel_norm / rel_raw) * 100:.0f}% 的跨轮散布"
                  "  (这才是 D2 该不该做的判据)")
        else:
            print("      → gflops 跨轮散布为 0, 归一化无从比较")


PROBE = '''
import subprocess, sys, time
import torch

gpu = int(sys.argv[1])
thr = float(sys.argv[2])

def read():
    r = subprocess.run(["nvidia-smi", "--id=%d" % gpu,
                        "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return int(r.stdout.strip().splitlines()[0])

x = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
for _ in range(20):
    x @ x
torch.cuda.synchronize()
time.sleep(6)
print("idle_before_burst: %d%%" % read(), flush=True)

for burst_s, label in ((0.3, "burst0.3s"), (2.0, "burst2.0s")):
    t0 = time.time()
    while time.time() - t0 < burst_s:
        x @ x
    torch.cuda.synchronize()
    t_end = time.time()
    rows = []
    for delay in (0.0, 0.3, 0.6, 1.2, 2.4, 4.0):
        while time.time() - t_end < delay:
            pass
        rows.append((delay, read()))
    first = dict(rows)[0.0]
    second = dict(rows)[1.2]
    print(label + ": " + "  ".join("t+%gs=%d%%" % (d, v) for d, v in rows), flush=True)
    print("   -> 现规则取 min(t+0, t+1.2) = %d%%   旧单次快照 = %d%%   阈值 %g%%"
          % (min(first, second), first, thr), flush=True)
'''


def probe_only(ssh, gpu):
    """进程内探针: 自己跑完一轮核之后, 立刻采 util 到底会读到多少

    这是 doc 33 那个「10-34% 自污染」结论的机制检验。生产 L3 路径测不到它 —— 每次
    远端调用都是新进程 (ssh + uv + import torch 要几秒), 到自己的核排空之后才采样。
    所以「连跑 N 次 L3, 污染触发率 0%」这个零结果本身没有判别力, 必须把量具挪到
    能看见的地方: 同一个进程里, 核刚停就采。
    """
    import shlex
    import subprocess
    import sys
    from pathlib import Path

    from _scratch import private_dir, remote_scratch_path, write_private

    alias = ssh._ssh_target()
    # 固定名 (`/tmp/_util_probe.py`) 换成随机名: 那台机是多用户的, 而这个名字既被我们
    # 写又被我们执行 —— 详见 `benchmarks/_scratch.py` 的模块 docstring。
    local = private_dir("utilprobe-")
    remote = remote_scratch_path("_util_probe.py")
    local_file = write_private(local, "_util_probe.py", PROBE)
    subprocess.run(["scp", "-q", str(local_file), f"{alias}:{remote}"], check=True)
    subprocess.run(["ssh", alias, f"chmod 600 {shlex.quote(remote)}"], check=True)
    cmd = (f"cd {ssh.work_dir} && CUDA_VISIBLE_DEVICES={gpu} "
           f"~/.local/bin/uv run --no-project python {shlex.quote(remote)} {gpu} "
           f"{CONTAMINATION_UTIL_PCT}")
    print("=== 远端进程内 util 采样探针 ===")
    try:
        p = subprocess.run(["ssh", alias, cmd], capture_output=True, text=True)
        print(p.stdout.strip() or p.stderr.strip()[:600])
    finally:
        # 那份可执行文件是**在共享机上**的, 不收了它就等于留一个公开路径给别人。
        subprocess.run(["ssh", alias, f"rm -f {shlex.quote(remote)}"], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--probe", action="store_true",
                    help="只跑远端进程内的采样探针 (不跑 L3), 用来判「自污染残留」这个机制在不在")
    args = ap.parse_args()

    backend = RemoteExecutor(mode="ssh")
    print("=== deploy ===")
    if not backend.deploy():
        sys.exit(1)
    ssh = backend._backend
    gpu = ssh._pinned_gpu()

    if args.probe:
        probe_only(ssh, gpu)
        return

    v = Verifier(backend)
    print(f"控制端固定的 GPU = {gpu}  污染阈值 util > {CONTAMINATION_UTIL_PCT}%")

    cases = [
        ("layernorm 1024×768 fp16 (launch 主导, 自动加采样到几千个微秒核)",
         TRITON_LN, "layernorm", ln_inputs(1024, 768)),
        ("layernorm 16384×6144 fp16 (degraded 档)",
         TRITON_LN, "layernorm", ln_inputs(16384, 6144)),
        ("matmul 4096³ fp16 (valid 档)",
         TRITON_MM, "matmul", mm_inputs(4096)),
    ]
    for label, code, ktype, inputs in cases:
        runs = []
        for _ in range(args.reps):
            r = v._verify_performance(code, inputs, 0.0, make_task(ktype))
            m = r.metrics
            samples = m.get("gpu_util_samples") or []
            runs.append({
                "samples": samples,
                "first": samples[0] if samples else None,
                "second": samples[1] if len(samples) > 1 else None,
                "min": m.get("gpu_util_pct"),
                "others": m.get("gpu_other_procs"),
                "contaminated": bool(m.get("benchmark_contaminated")),
                "speedup": m.get("speedup_vs_baseline"),
                "paired": m.get("paired_speedup_dispersion"),
                "validity": m.get("measurement_validity"),
                # D2: 窗口边界上的 SM 时钟与 throttle 位图 (只观测, 不判定)
                "sm_b": m.get("sm_clock_mhz_before"),
                "sm_a": m.get("sm_clock_mhz_after"),
                "sm_max": m.get("sm_clock_max_mhz_before"),
                "thr_b": m.get("throttle_bits_before"),
                "thr_a": m.get("throttle_bits_after"),
                "gflops": m.get("gflops"),
            })
        report(label, runs)


if __name__ == "__main__":
    main()
