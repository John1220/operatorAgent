"""真机验证 D3: 交错配对计时 + 只测一个基线, 在 H20 上到底改了多少

在控制端运行 (会触发 deploy, 需要能连到 GPU 机):
    python3 benchmarks/verify_interleaved_ab.py
    python3 benchmarks/verify_interleaved_ab.py --interfere   # 另起一条持续抢占卡

对照的是同一个远端、同一份 payload、同一张卡上的两种 runner:
  新  = 工作区里的 run_kernel.py (按块交错 + 只测一个基线)
  旧  = HEAD 里的 run_kernel.py  (顺序测候选, 再逐个测 2~3 个基线)
旧版用 `git show` 取出后 scp 成 run_kernel_prev.py, 调用时只是把 executor 的
`remote_runner_path` 临时指过去 —— GPU 固定、输入序列化、deploy 状态全部共用,
两条路的唯一差别就是那个脚本。旧脚本会忽略它不认识的 `baseline_priority` 键,
所以两臂吃的确实是同一份请求。

要看三件事:
  1. 端到端跑通, 且 `baselines` 只剩一个键 (省 GPU 时间那一半是否落地);
  2. 单次 L3 的墙钟时间 (省的到底是多少);
  3. 加速比在**有邻居抢卡时**离安静值有多远 —— 交错那一半的全部意义。
"""
import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from operator_agent.remote_executor import RemoteExecutor
from operator_agent.verifier import baseline_plan

REPS = 5
SHAPES = ((16384, 6144), (1024, 768))

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

# 干扰源: 一条不停跑大 matmul 的流, 模拟「邻居在抢同一张卡」
CONTENDER = '''
import sys, time, torch
d = torch.device("cuda")
x = torch.randn(4096, 4096, device=d, dtype=torch.float16)
end = float(sys.argv[1])
while time.time() < end:
    x @ x
    torch.cuda.synchronize()
'''


def ln_inputs(rows, cols):
    rng = np.random.default_rng(11)
    return {"X": rng.standard_normal((rows, cols)).astype(np.float16),
            "weight": np.ones(cols, dtype=np.float16),
            "bias": np.zeros(cols, dtype=np.float16)}


def rel_spread(xs):
    a = np.asarray(sorted(xs), dtype=np.float64)
    med = float(np.median(a))
    q1, q3 = np.percentile(a, [25, 75])
    return med, (float((q3 - q1) / med) if med else 0.0)


def pick_baseline_ms(b):
    """从返回里取那个基线的耗时: 新脚本只有一份并明说选了谁, 旧脚本要按名单挑"""
    bl = b.get("baselines", {})
    name = b.get("baseline_selected")
    if name and name in bl:
        return bl[name]["median_time_ms"], name
    for cand in baseline_plan("layernorm", "fp16"):
        if isinstance(bl.get(cand), dict):
            return bl[cand]["median_time_ms"], cand
    raise AssertionError(f"返回里没有可用基线: {list(bl)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interfere", action="store_true",
                    help="测量期间另起一条持续抢占 GPU 的 matmul 流")
    args = ap.parse_args()

    backend = RemoteExecutor(mode="ssh")
    print("=== deploy (新 runner) ===")
    if not backend.deploy():
        sys.exit(1)
    ssh = backend._backend
    alias, work = ssh._ssh_target(), ssh.work_dir
    runner_dir = ssh.remote_runner_path.rsplit("/", 1)[0]

    old_src = subprocess.run(["git", "-C", str(ROOT), "show",
                              "HEAD:remote_scripts/run_kernel.py"],
                             capture_output=True, text=True).stdout
    assert len(old_src) > 1000, "git show 取不到 HEAD 版 runner"
    Path("/tmp/run_kernel_prev.py").write_text(old_src, encoding="utf-8")
    subprocess.run(["scp", "-q", "/tmp/run_kernel_prev.py",
                    f"{alias}:{runner_dir}/run_kernel_prev.py"], check=True)
    Path("/tmp/_contender.py").write_text(CONTENDER, encoding="utf-8")
    subprocess.run(["scp", "-q", "/tmp/_contender.py", f"{alias}:/tmp/_contender.py"],
                   check=True)
    print(f"对照 runner: {runner_dir}/run_kernel_prev.py "
          f"({len(old_src.splitlines())} 行, 取自 HEAD)")

    # 干扰流必须落在**同一张卡**上, 否则抢的不是同一份资源, 测出来的是空气。
    # _pinned_gpu() 是懒选定的, 这里先逼它定下来。
    gpu = ssh._pinned_gpu()
    print(f"控制端固定的 GPU = {gpu}")
    pin = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
    interferer = None
    if args.interfere:
        cmd = (f"cd {work} && {pin}~/.local/bin/uv run --no-project python "
               f"/tmp/_contender.py {time.time() + 300}")
        interferer = subprocess.Popen(["ssh", alias, cmd],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        time.sleep(10)
        print(f"干扰流 pid={interferer.pid} (300s)")

    plan = baseline_plan("layernorm", "fp16")
    print(f"\n=== 基线偏好名单 (下发给远端): {plan}")
    print(f"=== 每个 shape {REPS} 次, 新旧 runner 逐次交替 ===")
    summary = {}
    for rows, cols in SHAPES:
        inputs = ln_inputs(rows, cols)
        sp = {"new": [], "prev": []}
        wall = {"new": [], "prev": []}
        nbl = {"new": set(), "prev": set()}
        for _ in range(REPS):
            for tag in ("new", "prev"):
                if tag == "prev":
                    ssh.remote_runner_path = f"{runner_dir}/run_kernel_prev.py"
                else:
                    ssh.remote_runner_path = f"{runner_dir}/run_kernel.py"
                t0 = time.perf_counter()
                b = ssh.benchmark_with_baselines(TRITON_LN, inputs, "layernorm",
                                                 precision="fp16",
                                                 baseline_priority=plan)
                wall[tag].append(time.perf_counter() - t0)
                if not b.get("success"):
                    print(f"  ❌ {tag} 失败: {str(b.get('error'))[:200]}")
                    continue
                base_ms, name = pick_baseline_ms(b)
                sp[tag].append(base_ms / b["candidate"]["median_time_ms"])
                nbl[tag].add(len(b.get("baselines", {})))
        print(f"\nlayernorm {rows}×{cols} fp16")
        for tag, label in (("new", "交错+单基线"), ("prev", "顺序+多基线")):
            if not sp[tag]:
                continue
            med, rel = rel_spread(sp[tag])
            print(f"  {label}  加速比 median={med:.4f}  跨次 iqr/median={rel:.4f}  "
                  f"每次测得基线数={sorted(nbl[tag])}  墙钟={statistics.mean(wall[tag]):.2f}s")
        if sp["new"] and sp["prev"]:
            mn, _ = rel_spread(sp["new"])
            mp, rp = rel_spread(sp["prev"])
            mn2, rn = rel_spread(sp["new"])
            print(f"  → 中位数之差 {abs(mn - mp):.4f} ({abs(mn - mp) / mp:.1%})   "
                  f"跨次散布 旧/新 = {rp / rn if rn else float('inf'):.2f}×")
            summary[(rows, cols)] = (mn, mp, rn, rp)

    if interferer:
        interferer.terminate()
        print("\n(干扰流已终止)")
    print("\n注: 墙钟里含一次 SSH 往返 + 依赖检查, 不只是 GPU 时间;"
          " 真正的对比口径是「每次测得基线数」。")


if __name__ == "__main__":
    main()
