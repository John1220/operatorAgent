"""真机探针: Triton 编译产物里有哪些**关于被测量那份码**的事实 (§10.3 条件 2 的真源)

用法 (探针要跑在 GPU 机上, 本机无卡):
    ssh <host> "cat > <远端临时文件>" < benchmarks/probes/probe_remote_jit_metadata.py
    ssh <host> "cd ~/operator_agent && ~/.local/bin/uv run python <远端临时文件>"

    (刻意的: 不写死任何本机或远端固定路径 —— 共享主机上写固定 scratch 路径会被
     `tests/unit/test_shared_host_scratch_paths.py` 的棘轮判红, 那是 doc 52 的落点纪律。)


为什么要有它: 控制端的 L1 只有 4 个键 (backend/compile_status/static_guards/
static_guards_lang)、L3 的 87 个键里**没有一个**关于线程数/寄存器/溢出 —— 也就是说
`num_warps`/`n_regs`/`n_spills` 今天根本没人发布, §10.3 条件 2 的第二把量具
(`escalation.occupancy_bound_evidence`) 因此在真机上不可能产出证据。
在动 runner 之前, 先用这个探针确认**真源到底露在哪** —— 不拿「应该有」当「确实有」。

2026-10-03 实测 (H20, triton 3.8.0, torch 2.14.0+cu130):
  * `_k.device_caches[dev][0]` 是一个 dict[key → CompiledKernel]  (`.cache` 已不存在)
  * `CompiledKernel.n_regs` / `.n_spills` / `.n_max_threads` 都在
  * `CompiledKernel.metadata` 是 `KernelMetadata`, 带 num_warps / num_stages / shared
    (每 CTA 的 smem **字节**, 实测) / warp_size / arch / hash / maxnreg
  ⇒ 条件 2 需要的两个输入 (num_warps、每 CTA smem) 与 spill-bound 的 n_spills
    **全都有真源**, 且都比控制端的估算强 (实测 vs 估算)。
  ⚠️ 版本相关: `.cache` 路线在 3.8 已消失, 所以 runner 取元数据必须按
    device_caches 走, 并且取不到时**不许编造**(三态: 缺失 ⇒ 不发布该键)。
"""
import torch, triton, triton.language as tl
print("triton", triton.__version__, "torch", torch.__version__)

@triton.jit
def _k(x_ptr, y_ptr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + o, tl.load(x_ptr + o) * 2.0)

x = torch.ones(1024, device="cuda"); y = torch.empty_like(x)
_k[(1024 // 128,)](x, y, BLOCK=128, num_warps=8, num_stages=3)
torch.cuda.synchronize()
print("launch ok; 现在找编译产物:")
print("  device_caches:", hasattr(_k, "device_caches"), "cache:", hasattr(_k, "cache"))
ck = None
try:
    dc = getattr(_k, "device_caches", None)
    if dc:
        for dev, entry in dc.items():
            cache = entry[0] if isinstance(entry, tuple) else entry
            print("  dev", dev, "缓存项", len(cache))
            ck = list(cache.values())[0]
except Exception as e:
    print("  device_caches 路线失败:", type(e).__name__, e)
if ck is None:
    try:
        ck = list(getattr(_k, "cache", {})[0].values())[0]
    except Exception as e:
        print("  cache 路线失败:", type(e).__name__, e)
if ck is not None:
    print("  编译产物类型:", type(ck).__name__)
    for attr in ("n_regs", "n_spills", "n_max_threads", "metadata", "num_warps"):
        print(f"   {attr} =", getattr(ck, attr, "<缺席>"))
    md = getattr(ck, "metadata", None)
    if md is not None:
        print("   metadata 字段:", [f for f in dir(md) if not f.startswith("_")])
