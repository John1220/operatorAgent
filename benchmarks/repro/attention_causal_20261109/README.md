# 失败回放：FP16 prefill Attention 的因果掩码错误（1×16×4096×128）

**一句话**：非因果用例通过，**五条 causal 用例全部失败**（最大误差 1.95e-03，匹配率
仍显示 0.9999 —— 单看匹配率会漏掉它）。根因是掩码乘在 softmax **之后**，
被屏蔽的位置仍贡献了归一化概率。历史补丁把掩码改到 softmax 之前（-inf 加法），**8/8** 通过。

- 采集：2026-11-09（A/C 工作流轮，C 臂第 5 次提交失败、第 6 次修好）
- 冻结契约：`e9f75d60cf1c2b35…`（8 条显式 case）
- 修前源码：[`candidate.py`](candidate.py)，sha256 `e1b711b7…`
- 历史补丁：[`historical_fix.py`](historical_fix.py)（**只给验收方核对，不进任何一臂**）
- 原始回执：[`verify-5`](../../../output/v08-ac-20261108/cell-attention-prefill-4096-C/run/verify-5/result.json) ·
  [`verify-6`](../../../output/v08-ac-20261108/cell-attention-prefill-4096-C/run/verify-6/result.json)

## 零模型回放（2026-11-19，H20）

```bash
python -m benchmarks.experiments.v08.failure_replay \
  --case benchmarks/repro/attention_causal_20261109 \
  --source candidate.py --host h20 --work-dir /root/operator-p2-seed-20261116

python -m benchmarks.experiments.v08.failure_replay \
  --case benchmarks/repro/attention_causal_20261109 \
  --source historical_fix.py --expect pass \
  --host h20 --work-dir /root/operator-p2-seed-20261116
```

| 回放 | 结果 |
|---|---|
| 修前 | `passed: true` —— 契约、源码哈希、`L1=true/L2=false`、五条 causal 用例与误差逐项对上 |
| 历史补丁 | `passed: true` —— `L2=True cases=8/8` |

## 失败形态

| case | 结果 |
|---|---|
| `s1024-nocausal` | **passed** |
| `s1000-causal` / `s1024-causal` / `s2048-causal` | failed，1.953125e-03 |
| `s4096-causal` / `s4096-causal-strided` | failed，9.765625e-04 |

**这份样本的价值**：误差只有 2e-3，而 attention 的标定 atol 是 3e-4 —— 它落在
"匹配率 0.9999 但逐元素判否"的区间。只看匹配率或只看最大误差绝对值都容易放过它，
必须按 `atol + rtol·|ref|` 的 AND 口径判。

## 范围

- 只覆盖这份修前源码、这条契约与 8 条 case；不证明其它 S/D/mask。
- 不跑 L3，不构成性能主张。
