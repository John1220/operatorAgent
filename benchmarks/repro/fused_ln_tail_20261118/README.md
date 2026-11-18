# 失败回放：fused residual+LayerNorm 的尾块数值超差（27B/medium，A 臂）

**一句话**：同一份候选在**大形状与单列行上都对，唯独尾块（N=33）超差 9.96e-02**。
根因是补齐 lane 参与了方差计算；同一份源码还漏了内层 stride，所以 strided 的 `8x33` 错得更狠。
修好后 32/32 通过，独立终评（`purpose=final`）L2 也是 32/32。

- 采集时间：2026-11-18T11:33:00+08:00
- 来源：`output/ac-v08f-pair2-A`（A 臂正式格，计划 `operator-v08-workflow-ac-v8`，pair-2）
- 原始回执：[`result-0001.json`](../../../output/ac-v08f-pair2-A/run/native-verification/result-0001.json)
- 原始候选：本目录 [`candidate.py`](candidate.py)，sha256 `9e760eb0…`（与回执逐字一致）
- 契约：`093339587d8c462c…`，套件 `652f65b1…`（32 case）

## 一条命令重跑

```bash
# 1) 复现同一处数值错误（核对源码 + 契约 + 具体失败用例）
python -m benchmarks.experiments.v08.failure_replay \
  --case benchmarks/repro/fused_ln_tail_20261118 \
  --source candidate.py --host h20 --work-dir /root/operator-p2-seed-20261116

# 2) 验证补丁（含独立终评 L2 回执）
python -m benchmarks.experiments.v08.failure_replay \
  --case benchmarks/repro/fused_ln_tail_20261118 \
  --source candidate_fixed.py --expect pass --final \
  --host h20 --work-dir /root/operator-p2-seed-20261116
```

第 1 条不只看 `L1/L2` 两格：**契约哈希与修前源码哈希都要对得上**，再核对 `case.json` 里点名的用例——
`fused-ln-residual-1x1-random-contiguous` 必须**通过**、
`fused-ln-residual-1x33-random-contiguous` 必须**失败**且 `max_error=0.099609375`、
`matched_ratio=0.1515`（容差内相等）。任一项对不上就判回放不成立。

## 失败形态

```
fused-ln-residual-1x1-random-contiguous                     passed  (0.00e+00)
fused-ln-residual-1x33-random-contiguous                    failed  (9.96e-02, matched 0.1515, first_bad@0)
fused-ln-residual-8x1-*（全部 8 条）                          passed  (0.00e+00)
```

**形状特异性**是这份样本的价值：主用例（32768×6144）与单列行都对，只有 N=33 这种
“末块不满”的宽度才炸。这类错误在只看声明形状的验收里会整块漏掉。

## 缺陷 1：补齐 lane 计入方差（数值）

`BLOCK = next_power_of_2(n)`，N=33 时 `BLOCK=64`，其余 31 个 lane 被 load 的 mask 置 0。
但 kernel 里：

```python
s = r + x            # pad lane: 0
mean = tl.sum(s) / n
d = s - mean         # pad lane: -mean   ← 错
var = tl.sum(d * d) / n
```

pad lane 把 `mean²` 加进方差，方差被抬高、输出被整体压扁。N=1 或 N=2 的幂时没有 pad lane，
所以 `1x1` 与 `8x1` 全过；N=33 必挂。

## 缺陷 2：内层 stride 被忽略（布局）

```python
r = tl.load(r_ptr + row * r_stride + idx, ...)     # 少了 idx * r_inner
```

ABI 明确要求 wrapper“独立读取输入与输出的 stride”。contiguous 输入下 `inner=1` 看不出来，
`8x33-*-strided` 因此错到 `7.29e-01`。冻结套件里正好有 strided 用例，所以它没能蒙混过去。

## 补丁

见 [`patch.diff`](patch.diff)（新增 3 个内层 stride 形参、mask 中心化、按内层 stride 存取）：

| 位置 | 改动 |
|---|---|
| kernel 形参 | 新增 `r_inner, x_inner, o_inner` |
| load / store | 地址改为 `idx * inner` |
| 中心化 | `d = tl.where(mask, s - mean, 0.0)` |
| wrapper | 传 `stride(-1)` 三个内层步长 |

## 回执

| 文件 | 内容 |
|---|---|
| [`replay-candidate.json`](replay-candidate.json) | 复现：`L1=true / L2=false`，点名的两条用例逐项对上（`passed=true`） |
| [`replay-candidate_fixed.json`](replay-candidate_fixed.json) | 补丁：`L2=true`，32/32；`final_levels.L2=true`（独立终评） |

## 范围与边界

- 只覆盖这一份候选、这一条冻结契约与 32 条套件；不证明其它形状/实现。
- 两个缺陷**都是候选自身的**，不是仪器问题：同一回执里主用例通过、套件失败，说明通道正常。
- 本回放不跑 L3，不构成性能主张。
