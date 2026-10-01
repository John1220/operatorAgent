"""资源可行性预检 —— 在动 GPU/SSH 之前把「这套 tile 根本放不下」拦下来

形制借自 LeetCUDA `kernels/interview/bench/utils.cu:27-33` 的 smem 预检与
`book/tests/common_test.h:31-37` 的 `book_require_sm()`：资源不满足是
**SKIP/拒绝**，不是让 kernel 启动后再炸。

为什么放在这里: 一次 L1 远程编译 ~2.6 s，失败的 launch 更贵。而且「smem 超限」
的报错在 Triton/CUDA 侧往往只是 `out of resource` / 私有地址空间——看不出
「BLOCK 多大」才是原因。静态估一下，把 BLOCK_* 直接写进反馈。

⚠️ 估的是**数量级**，不是精确占用:
  · matmul 形态: (BM*BK + BK*BN + BM*BN) * elem_bytes  (A/B/C 三块 SMEM)
  · 其它形态: 只报 BLOCK_* 乘积 * elem_bytes 的下界, 只在**明显超限**时才拒
宁可漏拦 (交给真启动去报) 也不误拦 —— 与接口契约门同一条保守原则
(doc 41: 误拦比误放便宜, 但这里误拦会挡住一个本来能跑的核, 所以阈值放宽到
「超限 20% 以上」才拒, 边界附近交给远端)。
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional

# 保守放行的倍数: 估出 1.2× 上限以内都不拦
_SLACK = 1.2

#: 同上, **公开名** —— §12 的便宜代理剪枝 (`candidates.prune_by_smem`) 用同一个常数。
#: 为什么要共用而不是各写一个: 两套阈值会长成「预检放行、剪枝拒绝」这种自相矛盾的
#: 行为, 而两处的估算用的是同一个公式 ⇒ 没有理由有两个系数。
SMEM_SLACK = _SLACK


def smem_limit_bytes(caps) -> Optional[int]:
    """每 SM 的 smem 上界 (字节)。`caps` 缺这个字段 / 为 0 ⇒ **None (缺 ≠ 无上限)**

    None 与 0 必须分开: 0 会被读成「一点都放不下」(把所有点都拒掉), 而事实是
    「这个设备的 smem 上界我们不知道」。调用方 (预检 / 剪枝) 各自显式处理 None。
    """
    limit_kb = getattr(caps, "shared_mem_per_sm_kb", 0) or 0
    if limit_kb <= 0:
        return None
    # 取整口径与重构前**逐字节一致**: int(kb * 1024), 不是 int(kb) * 1024
    # —— 小数 KB 时两者差一整 KB, 而既有用例正是钉在边界上的。
    return int(limit_kb * 1024)


@dataclass
class ResourceReport:
    feasible: bool = True
    reason: str = ""
    estimated_smem_bytes: Optional[int] = None
    limit_bytes: Optional[int] = None
    block_shapes: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def as_message(self) -> str:
        if self.feasible:
            return ""
        parts = [self.reason]
        if self.estimated_smem_bytes is not None and self.limit_bytes is not None:
            parts.append(f"估算 smem {self.estimated_smem_bytes / 1024:.1f} KiB "
                         f"> 上限 {self.limit_bytes / 1024:.1f} KiB")
        if self.block_shapes:
            parts.append(f"BLOCK 形状 {self.block_shapes}")
        parts.append("请调小 BLOCK_* 或拆 tile (资源预检, 未动 GPU)")
        return "; ".join(parts)


def _elem_bytes_from_code(code: str) -> int:
    """从源码猜元素宽度; 认不出就 2 (fp16 常见) —— 估大比估小安全。"""
    if re.search(r"float32|tl\.float32|torch\.float32|np\.float32|dtype=np\.float32", code):
        return 4
    if re.search(r"float16|tl\.float16|torch\.float16|fp16|bfloat16|tl\.bfloat16", code):
        return 2
    return 2


def _extract_block_sizes(code: str) -> list[tuple[str, int]]:
    """从 Triton 源码里抠出 `BLOCK_*: tl.constexpr = N` 或 autotune 配置里的字面量。"""
    found: dict[str, int] = {}

    # 1) `BLOCK_M: tl.constexpr = 64` / `BLOCK_M = 64` / `BLOCK_M=64`
    for m in re.finditer(r"\b(BLOCK_[A-Z0-9_]+)\s*(?::\s*tl\.constexpr\s*)?=\s*(\d+)", code):
        name, val = m.group(1), int(m.group(2))
        found[name] = max(found.get(name, 0), val)

    # 2) autotune 的 dict 字面量: `{'BLOCK_M': 256}` / `{"BLOCK_N": 128}`
    #    —— 不认这一种就漏掉最坏配置 (真机上表现为「预检说能跑、启动 out of resource」)
    for m in re.finditer(r"['\"](BLOCK_[A-Z0-9_]+)['\"]\s*:\s*(\d+)", code):
        name, val = m.group(1), int(m.group(2))
        found[name] = max(found.get(name, 0), val)

    return sorted(found.items())


def estimate_matmul_smem_bytes(block_sizes: dict[str, int], elem_bytes: int) -> Optional[int]:
    """(BM*BK + BK*BN + BM*BN) * elem。缺任一维就 None。"""
    bm = block_sizes.get("BLOCK_M")
    bn = block_sizes.get("BLOCK_N")
    bk = block_sizes.get("BLOCK_K")
    if not (bm and bn and bk):
        return None
    return (bm * bk + bk * bn + bm * bn) * elem_bytes


def estimate_generic_smem_bytes(block_sizes: dict[str, int], elem_bytes: int) -> Optional[int]:
    """非 matmul: 所有 BLOCK_* 乘积 * elem 当下界 (会低估, 所以只用于「明显超限」)。"""
    nums = [v for v in block_sizes.values() if v > 0]
    if not nums:
        return None
    prod = 1
    for v in nums:
        prod *= v
    return prod * elem_bytes


def check_resource_feasibility(code: str, caps, kernel_type: str = "") -> ResourceReport:
    """静态资源预检。`caps` 是 HardwareCaps (或任何有 shared_mem_per_sm_kb 的对象)。

    返回 ResourceReport; `feasible=False` 时 `as_message()` 是给 L1 用的整句反馈。
    """
    rep = ResourceReport()
    if not code or not code.strip():
        return rep

    limit = smem_limit_bytes(caps)
    if limit is None:
        rep.notes.append("caps 没有 shared_mem_per_sm_kb, 跳过 smem 预检")
        return rep
    # 每 block 的 optin 上限通常 ≤ 每 SM 的总量; 用 SM 总量当上界是**宽松**的
    # (Triton 还会留静态 smem), 这里故意不更严 —— 宁可漏拦。
    rep.limit_bytes = limit

    pairs = _extract_block_sizes(code)
    if not pairs:
        rep.notes.append("未解析到 BLOCK_* 字面量, 跳过 smem 预检")
        return rep
    block_sizes = dict(pairs)
    rep.block_shapes = [f"{k}={v}" for k, v in pairs]
    elem = _elem_bytes_from_code(code)

    kind = (kernel_type or "").lower()
    is_matmul = any(s in kind for s in ("matmul", "gemm", "linear")) or (
        "tl.dot" in code or "BLOCK_K" in block_sizes
    )
    est = (estimate_matmul_smem_bytes(block_sizes, elem) if is_matmul
           else estimate_generic_smem_bytes(block_sizes, elem))
    rep.estimated_smem_bytes = est
    if est is None:
        rep.notes.append("smem 估算缺维 (BLOCK_M/N/K 不全), 交给远端")
        return rep

    if est > rep.limit_bytes * SMEM_SLACK:
        rep.feasible = False
        rep.reason = "资源预检: 共享内存放不下这套 tile"
    return rep


def check_shape_divisibility(code: str, inputs: dict) -> ResourceReport:
    """BLOCK_* 对输入 shape 的整除性 —— 只在**硬要求**的写法下才拒。

    Triton 一般会自动 mask, 所以 `M % BM != 0` 不是错误而是性能问题;
    LeetCUDA 对 WGMMA 类硬要求才 SKIP。这里认两种硬要求:
      1. 源码里出现 `assert` / `static_assert` 含 BLOCK 与 shape;
      2. 显式 `# hard-require: BLOCK_M | M` 注释 (给 Designer 的标记)。
    认不出就放行。
    """
    rep = ResourceReport()
    if not inputs:
        return rep
    hard = bool(re.search(r"hard-require|static_assert", code))
    if not hard:
        return rep
    pairs = dict(_extract_block_sizes(code))
    if not pairs:
        return rep
    # 取第一个二维 shape (M, N) 与 BLOCK_M/N 对照; 对不上才拒
    shape = None
    for v in inputs.values():
        try:
            import numpy as np
            arr = np.asarray(v)
            if arr.ndim >= 2:
                shape = arr.shape
                break
        except Exception:  # noqa: BLE001
            continue
    if shape is None:
        return rep
    problems = []
    for dim_name, block_name in (("M", "BLOCK_M"), ("N", "BLOCK_N")):
        b = pairs.get(block_name)
        if b and shape and shape[0] % b != 0 and dim_name == "M":
            problems.append(f"{dim_name}={shape[0]} % {block_name}={b} != 0")
        if b and shape and len(shape) > 1 and shape[-1] % b != 0 and dim_name == "N":
            problems.append(f"{dim_name}={shape[-1]} % {block_name}={b} != 0")
    if problems:
        rep.feasible = False
        rep.reason = "资源预检: 形状不满足硬整除要求 (" + ", ".join(problems) + ")"
    return rep
