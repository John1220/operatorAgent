"""DeepSeek-V4 CSA 压缩条目 (Compressed Sparse Attention · Compressed KV Entries)

公式出处: DeepSeek-V4 (arXiv 2606.19348) §2.3.1 式 (11)–(12)。摘要:

    C = H·W^{KV} ∈ R^{n×c},  Z = H·W^{Z} ∈ R^{n×c}

    对第 i 个压缩条目 (i = 0 … n/m − 1), 取**相邻两窗共 2m 行** (当前窗 a + 前一窗 b):

        [S^a ; S^b] = Softmax_row([Z^a + B^a ; Z^b + B^b])
        C^Comp_i    = Σ_j S^a_j ⊙ C^a_j  +  Σ_j S^b_j ⊙ C^b_j

    · Softmax_row 在 2m 行上按列归一 (每个通道独立);
    · ⊙ 是 Hadamard;
    · i = 0 时前一窗的 Z^b 填 −∞、C^b 填 0 (论文原句);
    · 相邻压缩窗的 C^b 与 C^a **重叠**, 故序列长压成 1/m 而不是 1/(2m)。

为什么先打这一个: 投影 (H·W) 是普通 GEMM, **压缩本身**才是没有库对手的新算子。
对手是 PyTorch eager 组合而不是 cuBLAS —— L3 的加速比语义在这里才成立。

接口契约 (doc 41 的 `kernel(**inputs)` 约定, 参数名必须逐一等于输入名):
    C, Z : (n, c) float
    B_a, B_b : (m, c) float   # m = B_a.shape[0] 即压缩比
    n % m == 0                # v1 契约; 尾窗行为论文未写死, 不在这里猜
    return C_comp : (n // m, c)
"""
from __future__ import annotations

import numpy as np

KERNEL_TYPE = "csa_compress"
COMPRESSION_NAME = "csa_kv_compress"

# 与 REFERENCE_SOURCE 保持同一实现。这里用 NumPy, 便于在没有 torch 的控制端单测;
# 传给远端的是 REFERENCE_SOURCE (torch 版), 两边用同一组已知答案对拍。
REFERENCE_SOURCE = '''
def reference(C=None, Z=None, B_a=None, B_b=None, **_extra):
    """DeepSeek-V4 CSA 压缩条目 (arXiv 2606.19348 式 11-12)。

    C, Z: (n, c); B_a, B_b: (m, c); n % m == 0 → (n//m, c)
    """
    import torch
    if C is None or Z is None or B_a is None or B_b is None:
        raise ValueError("csa_compress 需要 C, Z, B_a, B_b 四个输入")
    C = torch.as_tensor(C)
    Z = torch.as_tensor(Z)
    B_a = torch.as_tensor(B_a, dtype=Z.dtype, device=Z.device)
    B_b = torch.as_tensor(B_b, dtype=Z.dtype, device=Z.device)
    if C.shape != Z.shape:
        raise ValueError(f"C 与 Z 形状不一致: {tuple(C.shape)} vs {tuple(Z.shape)}")
    if B_a.shape != B_b.shape or B_a.dim() != 2 or B_a.shape[1] != C.shape[1]:
        raise ValueError(f"B_a/B_b 形状必须同为 (m, {C.shape[1]}): "
                         f"{tuple(B_a.shape)} / {tuple(B_b.shape)}")
    n, c = C.shape
    m = B_a.shape[0]
    if m <= 0 or n % m != 0:
        raise ValueError(f"要求 n % m == 0 且 m > 0, 实际 n={n}, m={m}")
    n_out = n // m
    neg = torch.full_like(Z[:m], float("-inf"))
    zeros = torch.zeros_like(C[:m])
    outs = []
    for i in range(n_out):
        ca = C[i * m:(i + 1) * m]
        za = Z[i * m:(i + 1) * m] + B_a
        if i == 0:
            cb, zb = zeros, neg
        else:
            cb = C[(i - 1) * m:i * m]
            zb = Z[(i - 1) * m:i * m] + B_b
        # Softmax_row: 在 2m 行上按列归一
        s = torch.softmax(torch.cat([za, zb], dim=0), dim=0)
        sa, sb = s[:m], s[m:]
        outs.append((sa * ca).sum(dim=0) + (sb * cb).sum(dim=0))
    return torch.stack(outs, dim=0)
'''


def reference(C, Z, B_a, B_b):
    """与 REFERENCE_SOURCE 同义的本地实现 (NumPy, 无 torch 也能跑)。"""
    C = np.asarray(C, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    B_a = np.asarray(B_a, dtype=np.float64)
    B_b = np.asarray(B_b, dtype=np.float64)
    if C.shape != Z.shape:
        raise ValueError(f"C 与 Z 形状不一致: {C.shape} vs {Z.shape}")
    if B_a.shape != B_b.shape or B_a.ndim != 2 or B_a.shape[1] != C.shape[1]:
        raise ValueError(f"B_a/B_b 形状必须同为 (m, {C.shape[1]}): {B_a.shape} / {B_b.shape}")
    n, c = C.shape
    m = B_a.shape[0]
    if m <= 0 or n % m != 0:
        raise ValueError(f"要求 n % m == 0 且 m > 0, 实际 n={n}, m={m}")
    n_out = n // m
    neg = np.full((m, c), -np.inf, dtype=np.float64)
    zeros = np.zeros((m, c), dtype=np.float64)
    outs = np.empty((n_out, c), dtype=np.float64)
    for i in range(n_out):
        ca = C[i * m:(i + 1) * m]
        za = Z[i * m:(i + 1) * m] + B_a
        if i == 0:
            cb, zb = zeros, neg
        else:
            cb = C[(i - 1) * m:i * m]
            zb = Z[(i - 1) * m:i * m] + B_b
        logits = np.concatenate([za, zb], axis=0)
        # 与 torch.softmax(dim=0) 同语义: 按列稳定 softmax
        logits = logits - logits.max(axis=0, keepdims=True)
        # −∞ 行: exp 后为 0; max 也可能是 −∞ (整列全 −∞ 时), 显式处理
        logits = np.where(np.isfinite(logits), logits, -np.inf)
        ex = np.exp(logits)
        s = ex / np.sum(ex, axis=0, keepdims=True)
        sa, sb = s[:m], s[m:]
        outs[i] = (sa * ca).sum(axis=0) + (sb * cb).sum(axis=0)
    return outs.astype(np.float32)


def make_inputs(n: int = 4096, c: int = 512, m: int = 4, seed: int = 0) -> dict:
    """造一套可复现输入。n 必须是 m 的整数倍 (见模块 docstring 的契约)。"""
    if n <= 0 or c <= 0 or m <= 0 or n % m != 0:
        raise ValueError(f"非法 shape: n={n}, c={c}, m={m} (要求 n % m == 0)")
    rng = np.random.default_rng(seed)
    return {
        "C": rng.standard_normal((n, c)).astype(np.float32),
        "Z": rng.standard_normal((n, c)).astype(np.float32),
        "B_a": (rng.standard_normal((m, c)) * 0.1).astype(np.float32),
        "B_b": (rng.standard_normal((m, c)) * 0.1).astype(np.float32),
    }
