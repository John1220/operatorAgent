"""L2 数值门的容差与判定规则

单一事实来源: 容差表与判定式都定义在这里, 本地路径 (verifier.py, numpy) 与真机
路径 (remote_scripts/run_kernel.py, torch) 共用。runner 是独立部署到远端的单文件,
所以本模块由 SSHExecutor.deploy() 连同 run_kernel.py 一起推过去, 由 runner 以
同级模块的方式 import。

为什么要有这个模块 (docs/research/12 第一、二节):
  原判定是 `max_abs_error <= max_error` —— 纯绝对容差, 没有相对项。实测在
  4096³ fp16 matmul 上, driver 传的 max_error=5e-1 相当于允许把结果整体缩小
  14.93%; 而 max_rel_error 虽然被算出来并上报, 却从不参与判定。

  但「直接把 max_rel_error <= rtol 加进门禁」是错的: 一个合法的 fp16 累加核,
  其 max_rel 实测高达 2.1e4 —— 因为参考输出里有接近 0 的元素, abs/|ref| 在这些
  点上爆炸 (原先的 clamp_min(1e-12) 只会更糟)。相对项必须用能容忍近零参考的形式。

判定式采用 FlashInfer-Bench 的 AND 形式
(flashinfer_bench/bench/utils.py:113-118, 一手核实):

    abs_err = |actual - expected|                 # 均在 fp32 下比较
    rel_err = abs_err / (|expected| + EPS)        # EPS=1e-8, 不是 clamp_min(1e-12)
    bad     = (abs_err > atol) & (rel_err > rtol) # AND: 只有同时超两个阈值才算坏点
    matched_ratio = 1 - bad.sum() / bad.numel()
    passed  = 全部有限 and matched_ratio >= rho

AND 而非 OR 是关键: 近零参考处 rel_err 会爆, 但只要 abs_err 在 atol 内就不算坏点,
于是合法的 fp16 累加核不会被误杀; 而整体缩放 0.97 这种错误答案, 其 rel_err 处处
等于 0.03, 只要 rtol <= 1e-2 就每一个元素都是坏点, matched_ratio = 0, 必挂。

实测三门禁对照 (benchmarks/probe_gates.py):

    候选                 max_abs      纯atol(0.5)  KernelBench加性  本模块AND形式
    correct_fp32acc    0.000e+00      PASS         PASS             PASS
    correct_fp16acc    1.217e-02      PASS         PASS             PASS
    scaled_0.97        1.005e-01      PASS(漏)     fail             fail
    scaled_0.999       3.350e-03      PASS         PASS             PASS
"""

from typing import Tuple

# 参考输出接近 0 时 rel_err 的分母保护。取 1e-8 与 FlashInfer-Bench 一致;
# 原 runner 用的 clamp_min(1e-12) 太小, 会让近零点上的 rel_err 再放大 4 个数量级。
EPS = 1e-8

# 容差表。起点取 KernelBench v1 (src/kernelbench/eval.py:93-99, 一手核实):
#   fp32 1e-4 / fp16 1e-2 / bf16 1e-2, 且同一个值同时当 atol 和 rtol。
#   该表本身引用 torchbench 的 env_check.py。
#
# ⚠️ 这些数值尚未在 H20 上用已知正确的核标定过。风险点: 我们的参考是 cuBLAS,
# 而 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction 默认为
# True 时, fp16 GEMM 的中间归约可能就在 fp16 里做 —— 那么「候选与参考之差」里
# 有一部分是参考自身的误差。真机标定见 docs/research 的 3x 验证报告。
#
# (atol, rtol, rho)
TOLERANCE_BY_PRECISION = {
    "fp32": (1e-4, 1e-4, 1.0),
    "tf32": (1e-3, 1e-3, 1.0),   # TF32 尾数 10 bit, 与 fp16 同量级
    "fp16": (1e-2, 1e-2, 1.0),
    "bf16": (1e-2, 1e-2, 1.0),
    # 整数输出应当逐元素相等; 若将来出现量化核需要容差, 在这里改而不是在调用处放宽
    "int8": (0.0, 0.0, 1.0),
}

# precision 缺失或不认识时的兜底。取最严的 fp32 档: 宁可让一个本该通过的任务
# 报出数值失败(可诊断), 也不要因为字段没传就悄悄放行一个错核。
DEFAULT_PRECISION = "fp32"


def tolerance_for(precision: str) -> Tuple[float, float, float]:
    """按精度取 (atol, rtol, rho)"""
    return TOLERANCE_BY_PRECISION.get((precision or "").lower(),
                                      TOLERANCE_BY_PRECISION[DEFAULT_PRECISION])


def effective_atol(precision: str, task_max_error) -> Tuple[float, float, float]:
    """容差表与任务自带的 max_error 合并: 任务只能收紧, 不能放宽

    三个 driver 历史上都传 max_error=5e-1, 注释写的是"FP16 精度约 3.3 位有效
    数字, 4096 次累加误差较大, 放宽到 0.5"。那不是疏忽 —— 在只有绝对项的门禁下,
    输出幅值到 3.35 时确实需要一个很大的 atol 才能让正确核通过。加上相对项之后,
    实测合法 fp16 累加核的 max_abs 只有 1.2e-2, 0.5 这个量级就不再需要了, 而它
    恰好等于「允许整体缩小 14.93%」。

    所以这里取 min: driver 传的宽松值变成 no-op 而不是继续放行错核, 而某个任务
    若确实需要更严的阈值 (传 1e-5) 仍然生效。
    """
    atol, rtol, rho = tolerance_for(precision)
    if isinstance(task_max_error, (int, float)) and not isinstance(task_max_error, bool):
        atol = min(atol, float(task_max_error))
    return atol, rtol, rho


def evaluate(n_bad: int, total: int, has_nonfinite: bool,
             rho: float) -> Tuple[bool, float]:
    """由坏点计数得出判定与 matched_ratio

    元素级的 abs/rel/AND 由调用方在自己的数组后端上做 (numpy 或 torch), 这里只
    收标量 —— 这样判定规则仍只有一份, 又不必把整个输出张量从显存搬回主机
    (attention 类算子的输出可以是 GB 量级)。

    Returns:
        (passed, matched_ratio)
    """
    if has_nonfinite:
        # NaN/Inf 不进容差比较: NaN 与任何值比较都是 False, 会让坏点统计静默
        # 漏掉它。照 FlashInfer-Bench 的做法直接判死。
        return False, 0.0
    if total <= 0:
        return False, 0.0
    matched_ratio = 1.0 - (float(n_bad) / float(total))
    if matched_ratio < 0.0:
        matched_ratio = 0.0
    elif matched_ratio > 1.0:
        matched_ratio = 1.0
    return matched_ratio >= rho, matched_ratio
