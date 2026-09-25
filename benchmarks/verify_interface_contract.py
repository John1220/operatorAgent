"""真机复核: 接口契约静态检查到底拦不拦得住、会不会误拦 (缺口 6, docs/research/41)

在控制端运行 (会触发 deploy, 需要能连到 GPU 机):
    python3 benchmarks/verify_interface_contract.py

要判的三件事, 每件都得用真机数据判 —— 单元测试只能证明逻辑, 证明不了「生产路径上真的
省钱」和「对真 Triton 核不误伤」:

  A. **拦得住**: 复现仓库自己的历史现场 —— `test_nvidia_attention.py:190` 就往
     `test_inputs` 里塞了一个 `causal`, 而生成核十有八九不接这个参数。这条以前会走完
     一次 SSH 往返 + 远程 import + Triton JIT 之后抛
     `TypeError: kernel() got an unexpected keyword argument 'causal'`，烧掉一整轮 Fixer。
     现在期望: L1 直接失败, `results` 只有 1 条 (没有 L2/L3), **墙钟接近 0**。
  B. **不误伤**: 拿一份真在 H20 上跑得动的 Triton matmul (与
     `verify_util_decontamination.py` 同一份源码) 走完整三级验证。
     期望: L1 通过、L2/L3 真的执行并给出加速比。这条是防「检查写成无条件拦截」——
     那种实现比没有检查更糟。
  C. **多组输入也拦得住**: 同一个合法核, 输入套件的第 2 组多带一个键。
     期望: L1 失败。单测里这条是靠 Verifier 的循环实现的, 生产路径要再确认一次。

墙钟是这里的关键读数: A/C 应当在亚秒级, B 是几十秒 (真机 import + JIT + 三级验证)。
差多少个数量级, 就是每次命中省下多少 GPU 与 SSH 时间。
"""
import sys
import time

sys.path.insert(0, ".")

from benchmarks.verify_util_decontamination import TRITON_MM, make_task, mm_inputs
from operator_agent.remote_executor import RemoteExecutor
from operator_agent.verifier import Verifier

# 复现历史现场的形态: 注意力 wrapper 只声明 Q/K/V, 而调用方会多传一个 causal
ATTENTION_STUB = '''
import torch, triton, triton.language as tl

@triton.jit
def _attn(q_ptr, k_ptr, v_ptr, o_ptr, sm_scale, BLOCK: tl.constexpr):
    pass

def kernel(Q, K, V):
    return Q @ K.transpose(-1, -2) @ V
'''

ATTENTION_FIXED = ATTENTION_STUB.replace("def kernel(Q, K, V):",
                                         "def kernel(Q, K, V, causal=False):")


def attention_inputs():
    import numpy as np
    rng = np.random.default_rng(3)
    b, h, n, d = 2, 4, 128, 64
    qkv = lambda: rng.standard_normal((b, h, n, d)).astype(np.float16)
    # 与 test_nvidia_attention.py 同形: 三个张量 + 一个布尔标量
    return {"Q": qkv(), "K": qkv(), "V": qkv(), "causal": False}


def summarize(label, results, elapsed):
    l1 = results[0]
    status = l1.metrics.get("compile_status") if l1.metrics else None
    print(f"\n===== {label} =====")
    print(f"  跑了 {len(results)} 级验证  "
          f"levels={[r.level.name for r in results]}")
    print(f"  L1 passed={l1.passed}  compile_status={status}  墙钟={elapsed:.2f}s")
    if not l1.passed:
        head = l1.error_log.split("\n")
        for line in head[:6]:
            print(f"    | {line}")
    return l1, status


def main():
    print("=== deploy ===")
    backend = RemoteExecutor(mode="ssh")
    if not backend.deploy():
        sys.exit(1)
    v = Verifier(backend)

    # ---- A: 该拦的拦得住, 且不碰 GPU ----
    t0 = time.perf_counter()
    res_a = v.verify(ATTENTION_STUB, make_task("attention"), attention_inputs(),
                     target_gflops=0.0)
    ea = time.perf_counter() - t0
    l1a, sta = summarize("A 生成核不接 causal (历史现场)", res_a, ea)
    ok_a = (not l1a.passed and sta == "interface_mismatch" and len(res_a) == 1)
    print(f"  → {'✅' if ok_a else '❌'} 期望: L1 失败 + 没有 L2/L3 + 墙钟亚秒级")

    # ---- B: 真 Triton 核不受影响 ----
    t0 = time.perf_counter()
    res_b = v.verify(TRITON_MM, make_task("matmul"), mm_inputs(2048),
                     target_gflops=0.0)
    eb = time.perf_counter() - t0
    l1b, stb = summarize("B 真能跑的 Triton matmul (防误伤)", res_b, eb)
    ok_b = l1b.passed and len(res_b) == 3 and res_b[1].passed
    print(f"  → {'✅' if ok_b else '❌'} 期望: L1 通过 + 三级全跑完 + L2 数值通过")

    # ---- C: 多组输入里第 2 组才不匹配 ----
    ins = mm_inputs(2048)
    sets = [ins, dict(ins, scale=1.0)]
    t0 = time.perf_counter()
    res_c = v.verify(TRITON_MM, make_task("matmul"), sets, target_gflops=0.0)
    ec = time.perf_counter() - t0
    l1c, stc = summarize("C 输入套件第 2 组多带一个键", res_c, ec)
    ok_c = not l1c.passed and stc == "interface_mismatch"
    print(f"  → {'✅' if ok_c else '❌'} 期望: L1 失败 (只看第 1 组就会漏掉)")

    # ---- D: 修好签名的同一份核应当放行 ----
    t0 = time.perf_counter()
    res_d = v.verify(ATTENTION_FIXED, make_task("attention"), attention_inputs(),
                     target_gflops=0.0)
    ed = time.perf_counter() - t0
    l1d, std = summarize("D 同一份核补上 causal 参数之后", res_d, ed)
    ok_d = l1d.passed
    print(f"  → {'✅' if ok_d else '❌'} 期望: 放行 (证明报错信息指对了改法)")
    print(f"     注: 这条会真的去远端执行, 因为它只是**签名**合法; 数值不合法由 L2 判")

    print("\n===== 汇总 =====")
    for name, ok in (("A 拦得住历史现场", ok_a), ("B 不误伤真核", ok_b),
                     ("C 多组输入逐组判", ok_c), ("D 补对签名即放行", ok_d)):
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  墙钟读数: 被拦下 {ea:.2f}s / {ec:.2f}s, "
          f"完整三级验证 {eb:.1f}s, 单是一次远程编译就要 {ed:.1f}s。")
    print("  每次命中省下的是「SSH 往返 + 远程 import torch + Triton JIT + 一轮 Fixer」,")
    print("  而检查本身只是本地 AST 遍历 —— 两个数量级的差就是这项吸纳的全部理由。")


if __name__ == "__main__":
    main()
