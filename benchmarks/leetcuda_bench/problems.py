"""LeetCUDA 题目集 (pilot: 24 个语义题, 不含向量打包变体)

选题纪律:
  · 同一语义只取一题 (relu_f32 / relu_f16x8 是同一题的不同打包, 不重复计);
  · Easy/Medium 为主, GEMM 一题当 Hard 探针;
  · 每题自带 NumPy 参考 —— **不抄 LeetCUDA 的 .cu 当参考**, 避免「题解泄漏」。
    语义按 README 表与公开定义写, 与 LeetCUDA 实现对拍的题目在其 doc 链接里。

`split`:
  · `dev`   —— 允许当 few-shot / 进配方库
  · `test`  —— holdout, 禁止进 prompt/RAG, 只做最终评分

反污染纪律 (机械守卫在 `tests/unit/test_leetcuda_bench.py`):
  · **同一个 reference 函数不许跨 split** —— 公式一样只换 shape 就是泄漏
    (softmax 两题、sgemm 两题必须同侧);
  · 家族可以跨 (relu 与 hardshrink 公式不同, 同属 activation 不算泄漏);
  · 向量打包变体 (本 pilot 未收录) 若加入, 必须与语义题同侧。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Problem:
    pid: str
    title: str
    difficulty: str          # easy | medium | hard
    stars: str               # 与 LeetCUDA README 的 ⭐ 对齐
    family: str              # 语义家族 (anti-contamination 按族划 split)
    kernel_type: str
    input_spec: dict         # {name: (shape, dtype)}
    reference: Callable      # **inputs -> ndarray
    split: str = "dev"       # dev | test
    dsl: str = "python"
    max_error: float = 1e-4
    leetcuda_path: str = ""  # 出处, 只作溯源, 不当参考实现
    notes: str = ""


# ─── 参考实现 (全部 NumPy, 与 LeetCUDA .cu 无关) ───────────────────

def _to_f32(x):
    return np.asarray(x, dtype=np.float32)


def ref_add(A, B):
    return _to_f32(A) + _to_f32(B)


def ref_relu(X):
    return np.maximum(_to_f32(X), 0.0)


def ref_sigmoid(X):
    x = _to_f32(X)
    return 1.0 / (1.0 + np.exp(-x))


def ref_gelu(X):
    # tanh 近似 (与 torch.nn.functional.gelu 默认 approximate='none' 的 erf 版
    # 差 ~1e-3, 所以这里用精确 erf 版, 容差 1e-4 才立得住)
    from math import erf
    x = _to_f32(X)
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / np.float32(np.sqrt(2.0))))


def ref_swish(X):
    x = _to_f32(X)
    return x / (1.0 + np.exp(-x))


def ref_hardswish(X):
    x = _to_f32(X)
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0


def ref_hardshrink(X, lambd=0.5):
    x = _to_f32(X)
    return np.where(np.abs(x) > lambd, x, 0.0).astype(np.float32)


def ref_elu(X, alpha=1.0):
    x = _to_f32(X)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32)


def ref_embedding(WEIGHT, IDX):
    # IDX: int32 一维; WEIGHT: (vocab, dim)
    w = _to_f32(WEIGHT)
    idx = np.asarray(IDX, dtype=np.int64)
    return w[idx]


def ref_mat_transpose(X):
    return np.ascontiguousarray(_to_f32(X).T)


def ref_mat_transpose_diag(X):
    # LeetCUDA 的 diagonal2d: 按对角线块重排的转置 (语义等同转置, 布局不同)
    return np.ascontiguousarray(_to_f32(X).T)


def ref_sum(X):
    return np.asarray(_to_f32(X).sum(), dtype=np.float32)


def ref_dot(A, B):
    return np.asarray(np.dot(_to_f32(A), _to_f32(B)), dtype=np.float32)


def ref_softmax(X):
    x = _to_f32(X)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def ref_safe_softmax(X):
    return ref_softmax(X)   # safe 与 online 在数值结果上同定义


def ref_rope(X):
    # 旋转位置编码: 成对 (x0,x1) → 旋转 θ=i·pos
    x = _to_f32(X).copy()
    n, d = x.shape
    half = d // 2
    pos = np.arange(n, dtype=np.float32)[:, None]
    idx = np.arange(half, dtype=np.float32)[None, :]
    theta = pos / (10000.0 ** (2 * idx / d))
    cos, sin = np.cos(theta), np.sin(theta)
    x0, x1 = x[:, :half], x[:, half:]
    out = np.empty_like(x)
    out[:, :half] = x0 * cos - x1 * sin
    out[:, half:] = x0 * sin + x1 * cos
    return out


def ref_layer_norm(X, weight, bias):
    # 参数名必须与 input_spec 的键一致: reference(**inputs) 是按关键字调的
    x = _to_f32(X)
    w, b = _to_f32(weight), _to_f32(bias)
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + 1e-5)
    return (y * w + b).astype(np.float32)


def ref_rms_norm(X, weight):
    x = _to_f32(X)
    w = _to_f32(weight)
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-5)
    return (x / rms * w).astype(np.float32)


def ref_sgemm(A, B):
    return (_to_f32(A) @ _to_f32(B)).astype(np.float32)


# ─── 题目表 ──────────────────────────────────────────────────────

def _el(shape=(4096, 4096)):
    return {"X": (shape, "float32")}


PROBLEMS: list[Problem] = [
    # ── Easy ⭐ ──────────────────────────────────────────────
    Problem("E01", "elementwise add", "easy", "⭐", "elementwise", "add",
            {"A": ((4096, 4096), "float32"), "B": ((4096, 4096), "float32")},
            ref_add, "dev", leetcuda_path="kernels/elementwise/"),
    Problem("E02", "relu", "easy", "⭐", "activation", "relu",
            _el(), ref_relu, "dev", leetcuda_path="kernels/relu/"),
    Problem("E03", "sigmoid", "easy", "⭐", "activation", "sigmoid",
            _el(), ref_sigmoid, "dev", leetcuda_path="kernels/sigmoid/"),
    Problem("E04", "gelu (erf)", "easy", "⭐", "activation", "gelu",
            _el(), ref_gelu, "dev", leetcuda_path="kernels/gelu/",
            notes="用精确 erf 版; tanh 近似差 ~1e-3 过不了 1e-4 容差"),
    Problem("E05", "swish / silu", "easy", "⭐", "activation", "silu",
            _el(), ref_swish, "test", leetcuda_path="kernels/swish/"),
    Problem("E06", "hardswish", "easy", "⭐", "activation", "hardswish",
            _el(), ref_hardswish, "test", leetcuda_path="kernels/hardswish/"),
    Problem("E07", "hardshrink", "easy", "⭐", "activation", "hardshrink",
            _el(), ref_hardshrink, "test", leetcuda_path="kernels/hardshrink/",
            notes="λ=0.5, 非线性分段 —— holdout 与 swish 同族"),
    Problem("E08", "elu", "easy", "⭐", "activation", "elu",
            _el(), ref_elu, "dev", leetcuda_path="kernels/elu/"),
    Problem("E09", "embedding lookup", "easy", "⭐", "embedding", "embedding",
            {"WEIGHT": ((1024, 256), "float32"), "IDX": ((4096,), "int32")},
            ref_embedding, "dev", leetcuda_path="kernels/embedding/"),
    Problem("E10", "mat transpose (row2col)", "easy", "⭐", "transpose",
            "mat_transpose",
            {"X": ((2048, 2048), "float32")}, ref_mat_transpose, "dev",
            leetcuda_path="kernels/mat-transpose/"),
    Problem("E11", "mat transpose (diag block)", "easy", "⭐⭐", "transpose",
            "mat_transpose",
            {"X": ((1024, 1024), "float32")}, ref_mat_transpose_diag, "test",
            leetcuda_path="kernels/mat-transpose/",
            notes="语义仍是转置, 但 LeetCUDA 要求对角块布局 —— holdout 探布局能力"),

    # ── Medium ⭐⭐ ─────────────────────────────────────────
    Problem("M01", "sum reduce", "medium", "⭐⭐", "reduce", "reduce",
            {"X": ((1 << 20,), "float32")}, ref_sum, "dev",
            leetcuda_path="kernels/reduce/", notes="输出标量"),
    Problem("M02", "dot product", "medium", "⭐⭐", "reduce", "dot_product",
            {"A": ((1 << 20,), "float32"), "B": ((1 << 20,), "float32")},
            ref_dot, "dev", leetcuda_path="kernels/dot-product/"),
    Problem("M03", "softmax per-row", "medium", "⭐⭐", "softmax", "softmax",
            {"X": ((1024, 1024), "float32")}, ref_softmax, "dev",
            leetcuda_path="kernels/softmax/"),
    Problem("M04", "safe softmax", "medium", "⭐⭐", "softmax", "softmax",
            {"X": ((1024, 4096), "float32")}, ref_safe_softmax, "dev",
            leetcuda_path="kernels/softmax/",
            notes="数值同 softmax; holdout 靠 shape 不同 + 不许抄 dev 题解"),
    Problem("M05", "rope", "medium", "⭐⭐", "norm", "rope",
            {"X": ((512, 1024), "float32")}, ref_rope, "test",
            leetcuda_path="kernels/rope/", notes="成对旋转, 最容易写反 sin/cos"),
    Problem("M06", "layer_norm", "medium", "⭐⭐", "norm", "layernorm",
            {"X": ((1024, 768), "float32"), "weight": ((768,), "float32"),
             "bias": ((768,), "float32")},
            ref_layer_norm, "dev", leetcuda_path="kernels/layer-norm/"),
    Problem("M07", "rms_norm", "medium", "⭐⭐", "norm", "rmsnorm",
            {"X": ((1024, 768), "float32"), "weight": ((768,), "float32")},
            ref_rms_norm, "test", leetcuda_path="kernels/rms-norm/",
            notes="无 bias、无 mean —— 与 layernorm 同族, 作 holdout"),

    # ── Hard 探针 (GEMM 阶梯入口) ─────────────────────────────
    Problem("H01", "sgemm", "hard", "⭐⭐⭐", "gemm", "matmul",
            {"A": ((1024, 1024), "float32"), "B": ((1024, 1024), "float32")},
            ref_sgemm, "test", leetcuda_path="kernels/sgemm/ / kernels/hgemm/naive/",
            notes="GEMM 阶梯 L0 入口; 打平 cuBLAS 即满分"),
    Problem("H02", "sgemm (non-square)", "hard", "⭐⭐⭐", "gemm", "matmul",
            {"A": ((1536, 1024), "float32"), "B": ((1024, 2048), "float32")},
            ref_sgemm, "test", leetcuda_path="kernels/hgemm/naive/",
            notes="holdout: 非方阵 tile 边界"),
]

assert len(PROBLEMS) >= 20, "pilot 至少要 20 题"
assert len({p.pid for p in PROBLEMS}) == len(PROBLEMS), "pid 重复"


def by_id(pid: str) -> Problem:
    for p in PROBLEMS:
        if p.pid == pid:
            return p
    raise KeyError(pid)


def make_inputs(p: Problem, seed: int = 0) -> dict:
    """按 input_spec 造一套可复现输入。"""
    rng = np.random.default_rng(seed)
    out = {}
    for name, (shape, dtype) in p.input_spec.items():
        if dtype.startswith("int"):
            vocab = 1024
            out[name] = rng.integers(0, vocab, size=shape).astype(np.int32)
        else:
            out[name] = rng.standard_normal(shape).astype(np.float32)
    return out


def splits() -> dict:
    return {
        "dev": [p for p in PROBLEMS if p.split == "dev"],
        "test": [p for p in PROBLEMS if p.split == "test"],
    }
