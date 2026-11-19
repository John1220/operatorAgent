# 失败回放：FP16 MatMul 的非连续布局故障（A=32×4 / B=4×32）

**一句话**：`4096³` 主用例与全部连续用例都对，**只有 `strided` 那条**（A=32×4、B=4×32）
误差 **9.69140625**；核按连续布局写死地址（`rows*K`、`ks*N`），没读运行时 stride。
历史补丁补上四个 stride 参数后 **15/15** 通过。

- 采集：2026-11-07（历史真机回执，S3 复现回归夹具）
- 冻结契约：`6e337b0f1be60591…`（2026-11-19 复算一致，15 条 case）
- 修前源码：[`candidate.py`](candidate.py)，sha256 `93eba1db…`（与回执逐字一致）
- 历史补丁：[`historical_fix.py`](historical_fix.py)（**只给验收方核对，不给任何一臂**）
- 原始回执：[`verify-1`](../../../output/v08-real-receipt-20261107/run/verify-1/result.json)（修前失败）·
  [`verify-2`](../../../output/v08-real-receipt-20261107/run/verify-2/result.json)（补丁 15/15）

## 零模型回放（2026-11-19，H20）

```bash
# 1) 原核仍在同一条用例上失败
python -m benchmarks.experiments.v08.failure_replay \
  --case benchmarks/repro/matmul_strided_20261107 \
  --source candidate.py --host h20 --work-dir /root/operator-p2-seed-20261116

# 2) 历史补丁仍过完整 L2
python -m benchmarks.experiments.v08.failure_replay \
  --case benchmarks/repro/matmul_strided_20261107 \
  --source historical_fix.py --expect pass \
  --host h20 --work-dir /root/operator-p2-seed-20261116
```

两条都在当前 H20 上通过：

| 回放 | 结果 |
|---|---|
| 修前 | `passed: true` —— 契约哈希、源码哈希、`L1=true/L2=false`、`auto-k4-m32-n32-fp16-random-strided` 失败且 `max_error=9.69140625`、`matched_ratio=0.0029296875`，逐项对上 |
| 历史补丁 | `passed: true` —— `L2=True cases=15/15` |

回执：[`replay-candidate.json`](replay-candidate.json) · [`replay-historical_fix.json`](replay-historical_fix.json)

## 失败形态

```
auto-k4-m32-n32-fp16-random-contiguous      passed   0.0
auto-k4-m32-n32-fp16-random-strided        failed   9.69140625   (1021/1024 元素超差)
auto-k4-m32-n32-fp16-{constant,large,small,zeros}-contiguous  passed
auto-k4-m31-*, auto-k4-m32-n31/n33, auto-k4-m33-*, auto-k1/k3/k5-*  passed
```

价值在于**布局特异性**：所有形状边界、值模式、K 变化都过，只有非连续布局挂。
只看声明形状（4096³）的验收会整块漏掉它。

## 根因与补丁

```python
# 修前：假设连续
a_ptrs = a_ptr + rows[:, None] * K + ks[None, :]
b_ptrs = b_ptr + ks[:, None] * N + cols[None, :]

# 历史补丁：把 stride 传进来
a_ptrs = a_ptr + rows[:, None] * sa_m + ks[None, :] * sa_k
b_ptrs = b_ptr + ks[:, None] * sb_k + cols[None, :] * sb_n
```

## 范围

- 只覆盖这份修前源码、这条契约与 15 条 case；不证明其它形状/实现。
- 本回放不跑 L3，不构成性能主张。
- `historical_fix.py` 不进任何一臂的工作目录：修复对照的主判据就是"能否自己修出来"。
