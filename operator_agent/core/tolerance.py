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

实测三门禁对照 (benchmarks/probes/probe_gates.py):

    候选                 max_abs      纯atol(0.5)  KernelBench加性  本模块AND形式
    correct_fp32acc    0.000e+00      PASS         PASS             PASS
    correct_fp16acc    1.217e-02      PASS         PASS             PASS
    scaled_0.97        1.005e-01      PASS(漏)     fail             fail
    scaled_0.999       3.350e-03      PASS         PASS             PASS
"""

from dataclasses import dataclass
from typing import Optional, Tuple

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


# ---- 误差形态 (设计 §8.2) ----------------------------------------------------
#
# 「第一个坏点」回答**哪错了** (位置); 「误差形态」回答**该怎么修** (类别) ——
# 形态才是「修 vs 重写」最直接的依据 (设计 §8.2 / §21.2)。
#
# 与 `evaluate` 同一个约定: 元素级归约由调用方在自己的数组后端上做 (numpy / torch),
# 这里只收**两个小布尔向量** (每行 / 每列有没有坏点) 与几个计数 —— 于是分类规则只有
# 一份, 两个生产者 (`verifier.py` 的本地路径与 `remote_scripts/run_kernel.py` 的真机
# 路径) 不会各写一套, 而"首/末/尾部连续"这类最容易写歪的逻辑也只有一个实现。
#
# ⚠️ 形态是**证据**, 不是判据: 它进 reason 与 Fixer 的 prompt, **不改路由决策**。
# 判歪的代价是"提示指歪了", 不是"候选被误杀" —— 这条边界是刻意的 (设计 §8.2)。

ERROR_SHAPE_NONFINITE = "nonfinite"
ERROR_SHAPE_WHOLE_TENSOR = "whole_tensor"
ERROR_SHAPE_TAIL_BLOCK = "tail_block"
ERROR_SHAPE_WHOLE_ROW = "whole_row"
ERROR_SHAPE_BROADCAST_AXIS = "broadcast_axis"

#: 今天**有量具**的五种形态。设计 §8.2 还列了 `precision_accumulation`(误差随 k 增长):
#: 它要的是"误差对归约维度的依赖"这类证据, 今天没有量具 ⇒ 分类器返回 `None`(无法确立),
#: 而不是拿一个看着像的标签去顶替 (与 §4.3 规则 8、"未观测不改判" 同一条纪律)。
ERROR_SHAPES = (ERROR_SHAPE_NONFINITE, ERROR_SHAPE_WHOLE_TENSOR,
                ERROR_SHAPE_TAIL_BLOCK, ERROR_SHAPE_WHOLE_ROW,
                ERROR_SHAPE_BROADCAST_AXIS)

#: 「整张量都错」的坏点占比门 (整体缩放 / 累加精度错都落在这一格)
WHOLE_TENSOR_MIN_BAD_RATIO = 0.99
#: 尾块判据的宽度分母: 坏行/坏列只占不到 1/8 才叫"尾块", 否则是整片错
TAIL_MAX_FRACTION_DENOM = 8


@dataclass(frozen=True)
class ErrorShapeInputs:
    """分类器的输入 —— 全部是整数 / 布尔, 由 `summarize_bad_axes` 从两个小向量算出。

    `rows`/`cols` 是**把张量看成二维**之后的形状 (除最后一维外全部折叠): 于是 matmul 的
    (M,N)、attention 的 (B,H,S,D)→(B*H*S,D) 用同一套规则。
    """

    n_bad: int
    total: int
    rows: int
    cols: int
    bad_rows: int                 # 至少有一个坏点的行数
    bad_cols: int
    rows_all_bad: int             # **整行**全坏的行数
    cols_all_bad: int
    bad_row_prefix: int           # 行方向**开头**连续"有坏点"的行数
    bad_row_suffix: int           # 行方向**末尾**连续"有坏点"的行数
    bad_col_prefix: int
    bad_col_suffix: int
    first_bad_row: Optional[int] = None
    last_bad_row: Optional[int] = None
    first_bad_col: Optional[int] = None
    last_bad_col: Optional[int] = None
    has_nonfinite: bool = False


def _prefix_len(flags) -> int:
    n = 0
    for v in flags:
        if not v:
            break
        n += 1
    return n


def _suffix_len(flags) -> int:
    n = 0
    for v in reversed(list(flags)):
        if not v:
            break
        n += 1
    return n


def summarize_bad_axes(per_row, per_col) -> dict:
    """从"每行/每列有没有坏点"两个小布尔向量算出分类器要的计数 (纯 stdlib)

    为什么只收这两个向量: 全量掩码可能上 GB, 而它们的长度分别只有 rows / cols ——
    归约在调用方自己的后端上做 (torch 在设备上, numpy 直接), 这里只做**纯整数**统计。
    """
    row = [bool(v) for v in per_row]
    col = [bool(v) for v in per_col]
    row_idx = [i for i, v in enumerate(row) if v]
    col_idx = [i for i, v in enumerate(col) if v]
    return {
        "rows": len(row),
        "cols": len(col),
        "bad_rows": len(row_idx),
        "bad_cols": len(col_idx),
        "first_bad_row": row_idx[0] if row_idx else None,
        "last_bad_row": row_idx[-1] if row_idx else None,
        "first_bad_col": col_idx[0] if col_idx else None,
        "last_bad_col": col_idx[-1] if col_idx else None,
        "bad_row_prefix": _prefix_len(row),
        "bad_row_suffix": _suffix_len(row),
        "bad_col_prefix": _prefix_len(col),
        "bad_col_suffix": _suffix_len(col),
    }


def classify_error_shape(inp: ErrorShapeInputs) -> Optional[str]:
    """按设计 §8.2 判定误差形态; **判不了就返回 `None`**(无法确立), 不猜

    规则顺序即优先级, 理由写在各条上:

    1. `nonfinite` —— NaN/Inf 压倒一切: 别的形态都是从"哪几个元素错"推出来的, 而
       含非有限值时那些位置统计本身就没有意义。
    2. 坏点占比 ≥ 99% ⇒ `whole_tensor`: 整体缩放 (rel_err 处处相同) 与累加精度错
       都落在这一格。这是**形状**的描述, 不是成因的断言 —— 例如"丢了输入 B"(A@A)
       同样整张量错, 它也是 `whole_tensor`, 而"是什么错"由理由文本承担。
    3. `tail_block`: 坏点集中在行/列方向的**末尾连续一小段** (非对齐尾块 / 最后一行 /
       最后一列)。判据同时要求"连续"与"够小": 只满足"在一行里"而落在中间的, 是下一格的
       `whole_row`, 不是尾块 —— 这两者要修的地方完全不同。
    4. `whole_row` / 5. `broadcast_axis`: 坏点恰好是若干**完整行** / **完整列**
       (广播轴选错、索引少一维的典型形状)。两者对称, 分开命名是为了让提示能直接说
       "检查行方向的广播"还是"检查列方向"。
    6. 其余 ⇒ `None`。**这是三态里的一态, 不是失败**: 一个"看着像但不成立"的标签会把
       修复引到错的地方, 比没有标签更坏。
    """
    rows, cols = int(inp.rows), int(inp.cols)
    total, n_bad = int(inp.total), int(inp.n_bad)
    if inp.has_nonfinite:
        return ERROR_SHAPE_NONFINITE
    if n_bad <= 0 or total <= 0 or rows <= 0 or cols <= 0:
        return None                      # 没有失败 ⇒ 没有形态可谈
    if float(n_bad) / float(total) >= WHOLE_TENSOR_MIN_BAD_RATIO:
        return ERROR_SHAPE_WHOLE_TENSOR
    tiny_r = max(1, rows // TAIL_MAX_FRACTION_DENOM)
    tiny_c = max(1, cols // TAIL_MAX_FRACTION_DENOM)
    if 0 < inp.bad_row_suffix <= tiny_r and inp.bad_rows <= tiny_r:
        return ERROR_SHAPE_TAIL_BLOCK
    if 0 < inp.bad_col_suffix <= tiny_c and inp.bad_cols <= tiny_c:
        return ERROR_SHAPE_TAIL_BLOCK
    if inp.bad_rows and inp.rows_all_bad == inp.bad_rows and inp.bad_rows * cols == n_bad:
        return ERROR_SHAPE_WHOLE_ROW
    if inp.bad_cols and inp.cols_all_bad == inp.bad_cols and inp.bad_cols * rows == n_bad:
        return ERROR_SHAPE_BROADCAST_AXIS
    return None


def error_shape_from_axis_flags(*, per_row, per_col,
                                n_bad: int, total: int,
                                rows_all_bad: int = 0, cols_all_bad: int = 0,
                                has_nonfinite: bool = False) -> Optional[str]:
    """两个生产者唯一的入口: 交两个小布尔向量 + 三个计数, 拿一个形态标签

    `rows_all_bad` / `cols_all_bad` 是"整行/整列全坏"的计数 (各一次归约), 其余统计由
    `summarize_bad_axes` 做。收成一个入口是为了让"两个生产者传的东西必须一样"这件事
    在签名上就成立 —— 少传一个计数, 类型上就看得出。
    """
    summ = summarize_bad_axes(per_row, per_col)
    return classify_error_shape(ErrorShapeInputs(
        n_bad=int(n_bad), total=int(total),
        rows_all_bad=int(rows_all_bad), cols_all_bad=int(cols_all_bad),
        has_nonfinite=bool(has_nonfinite), **summ))


# ---- 高精度参考那把尺 (KernelOPT `2609.30059v1` 行 372-389 与行 1344-1362) ----------------
#
# 外部一手 (行 375-381 逐字, 公式里的 ‖·‖∞ 在 HTML 里丢了竖线):
#     "Let dref = rf32 - rf64 and dopt = of32 - rf64. When dref >= 1e-8 (the baseline itself
#      has FP32 rounding error), the error ratio is: rho = dopt / dref. If rho <= 10, the
#      error is within the range expected from different FP32 accumulation orders; correct
#      TF32 kernels yield rho in [1,3] while algorithmically incorrect kernels yield rho > 100."
#
# 它与我方 AND 那条的**根本差别**: AND 比的是「候选 vs 参考」, 误差条带是**绝对**的 atol,
# 于是参考幅值一小就看不见任何东西 (`benchmarks/probes/probe_l2_sampling_gate.py` 臂 1/2 拍到的
# 全零核 PASS 就是这个)。这把尺比的是「候选到真值的距离 / 参考到真值的距离」—— 分母是**参考
# 自己的误差**, 所以条带自动随幅值缩放, 全零核在任意幅值下 ρ 都是天文数字。
#
# ⚠️ 今天**只观测不判定** (与 #89 的观测半边同形): 这三个阈值是**他们语料上**的数
# (ρ∈[1,3] 对 ρ>100), 我方在 H20 上的 ρ 分布还没测过 ⇒ 拿它当门就是把抄来的数当成自己的标定。
ORACLE_EXACT_DENOM = 1e-8     # d_ref 小于此 ⇒ 参考本身是精确运算 (max/argmax/sort 那一类)
ORACLE_RHO_MAX = 10.0         # 他们的比值阈
ORACLE_SCALE_REL_MAX = 1e-4   # 退化分支: d_opt / max(‖r_f32‖∞, 1e-12) <= 1e-4
ORACLE_SCALE_EPS = 1e-12
ORACLE_ABS_FLOOR = 5e-3       # 第三条兜底: d_opt <= max(5e-3, 1e-3 * ‖r_f32‖∞)
ORACLE_ABS_PER_AMPLITUDE = 1e-3


def error_ratio(d_opt: float, d_ref: float, ref_amplitude: float) -> dict:
    """按「候选到高精度真值的距离 / 参考到它的距离」给一份读数 —— 只收标量

    与 `evaluate` 同一个约定: 元素级的减法与 ∞-范数由调用方在自己的数组后端上做
    (numpy 或 torch), 这里只把规则集中成一份。本模块会被 `SSHExecutor.deploy()` 推到远端,
    所以真机路径将来要用同一个规则时不必再抄一遍。

    **三个判据分开返回, 不折成一个布尔** (`meets_ratio` / `meets_scale_relative` /
    `meets_abs_bound`): 原文是「ratio 不过再试 scale-relative, 再不过还有一个绝对兜底」,
    折成一个 `passed` 就把「差一点点」与「靠兜底过的」两种完全不同的事压成一层
    (doc 49 §3.2 B3 那一族)。`branch` 说的是**原文会走哪一支**, 不是「我方判它过不过」。
    """
    d_opt = float(d_opt)
    d_ref = float(d_ref)
    amplitude = float(ref_amplitude)
    abs_bound = max(ORACLE_ABS_FLOOR, ORACLE_ABS_PER_AMPLITUDE * abs(amplitude))
    out = {"d_opt": d_opt, "d_ref": d_ref, "ref_amplitude": amplitude,
           "meets_ratio": False, "abs_bound": abs_bound,
           "scale_relative_error": d_opt / max(amplitude, ORACLE_SCALE_EPS)}
    out["meets_scale_relative"] = out["scale_relative_error"] <= ORACLE_SCALE_REL_MAX
    out["meets_abs_bound"] = d_opt <= abs_bound
    if d_ref >= ORACLE_EXACT_DENOM:
        out["branch"] = "ratio"
        out["rho"] = d_opt / d_ref
        out["meets_ratio"] = out["rho"] <= ORACLE_RHO_MAX
    else:
        # 参考自身没有可测的舍入误差 ⇒ 除以它就是除以噪声。这里**不返回一个 ρ**,
        # 因为「分母不可信时不许替它判」在 #64 已经付过学费了。
        out["branch"] = "exact"
        out["rho"] = None
    return out

