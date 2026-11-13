# v0.8 支持矩阵（截至 2026-11-08）

这张表的用途是**说清支持什么、排除什么**，让调用方不必从代码里猜。
机器可读版本在 `operator_agent capabilities` 的 `correctness_profiles.families`；
`correctness_profiles.operators` 是**二维 axis 生成器的词表**，不是支持清单。

## 1. 受支持的族

| 族 | profile | dtype | 形状 / 布局 | 判定范围 | 窗口门 |
|---|---|---|---|---|---|
| matmul / gemm | `builtin-l2-v1`（axis 生成器） | fp16 / fp32 | `(M,N,K)`；contiguous + strided 派生 | 二维算子有限 L2 子域 | `preflight --shape M,N,K [--dtype]` |
| relu / softmax / layernorm 等逐元素与归约 | `builtin-l2-v1`（axis 生成器） | fp16 / fp32 | `(M,N)`；contiguous + strided 派生 | 同上 | `preflight --op layernorm --shape M,N` |
| layernorm v2（受信任参考） | `builtin-layernorm-fp16-cuda-v2` | fp16 | `(M,N)`，末轴归一化，`atol=1e-5` | 32 组标定套件 | 同上 |
| **attention（prefill）** | `builtin-attention-fp16-cuda-v1` | **fp16 only** | `(B,H,S,D)` = **B=1, H_q=H_kv=16, D=128, S ∈ {1024, 2048, 4096}**；主用例 contiguous，派生 case 含 strided | 8 条显式 case（含 `S=1000` 尾块、`strided`、`large` 极值）；`atol=3e-4`、`rtol=1e-2`、匹配率 1.0 | `preflight --op attention --shape B,H,S,D [--causal]` |

attention 的语义（冻结）：`O = softmax(QKᵀ/√D + mask)·V`，fp16 输入输出、fp32 累加，
无 affine / 无 dropout / 无 bias；`causal` 只做 `j ≤ i`，按**标量**直传（不是张量）。

## 2. 明确排除（不是"以后再说"，是当前判不了）

| 项 | 原因 |
|---|---|
| decode（`S_q=1`） | 核只有几微秒，`launch_dominance` 会过 0.5，测量判 `uninformative`——结构上不可测 |
| GQA / MQA（`H_kv ≠ H_q`） | 参考语义、套件与容差都只按 MHA 标定过 |
| `D ≠ 128` | 同上；L2 的 `_attention_case` 只有 D=128 的布局 |
| bf16 / fp8 | numpy 侧造不出 bf16 输入；fp8 没有参考与容差 |
| sliding window / additive bias / paged KV | 不在 Scope A，套件里没有对应 case |
| 非 2 的幂的 S 作为**声明**形状 | 只在派生 case 里出现（`s1000-causal`）；声明它会被 `attention_scope` 拒绝 |
| attention + fp32 | 没有 profile：`prepare` 报"默认正确性配置未确立（不猜算子语义）" |

## 3. 环境要求

| 项 | 值 | 怎么核对 |
|---|---|---|
| torch | `2.14.0+cu130`（受信构建） | 部署自动钉版本并回读；不等于 pin 就拒绝把环境标记为就绪 |
| CUDA / 驱动 | 13.0 / `580.105.08` | 参考身份回执里的 `execution` 段 |
| GPU | NVIDIA H20（`compute_capability 9.0`） | 同上；可 `OPERATOR_AGENT_REQUIRED_GPU` 固定卡 |
| triton | `3.8.0`（torch 依赖） | 远端 venv |

受信任参考任务（layernorm v2、attention）在身份不一致时 **L2 前 fail closed**，这是设计行为，
不是误报——2026-11-08 之前默认 runner 漂到 2.14.1 时，两条链一起停在这里。

## 4. 自检三步

```bash
operator-agent capabilities                       # 支持清单与后端接线（不联网）
operator-agent preflight --backend ssh --host h20 \
    --op attention --shape 1,16,1024,128 --dtype fp16 \
    --reps 5 --repeat 8 --planned-effect-pct 2.5  # 这个窗口能不能判 2.5% 的效应
operator-agent prepare --request "fp16 attention prefill 1x16x1024x128" \
    --shapes 1,16,1024,128 --hardware nvidia_gpu --dsl triton \
    --precision fp16 --run-dir <RUN>              # 冻结契约（越界在这里被拒）
```

## 5. 这张矩阵不证明什么

- **8 条 case 是有限域，不是全域证明**：`S/D/H/mask` 之外的组合没有被验证过。
- attention 的达成率来自**粗估模型**（`attention_coarse_v1`，忽略 softmax 与中间搬运），
  `reference_lower_bound=false`，天花板出口关闭。
- L3 **只测主用例**（nocausal），8 条 case 是正确性范围；没有声明 `case_weights` / `max_regression`。
- `degraded` 读数可用于诊断，**不进**任何达标判定；`uninformative` 连比较都不做。

## 6. 证据入口

[C5 端到端](../status/v08/c5-attention-e2e-20261108.md) ·
[A1/A2 部署与窗口门](../status/v08/a1-a2-deploy-pin-and-attention-gate-20261108.md) ·
[A3 R2 预登记](../status/v08/a3-attention-r2-preregistration-20261108.md) ·
[attention 设计](../design/v08/attention-family.md) ·
[候选余量实测](../status/v08/candidate-headroom-20261107.md)