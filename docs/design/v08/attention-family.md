# attention 族：Scope A 设计（v0.8 提案，未实现）

2026-11-07 · 设计文档 · 状态：**待评审，未动代码**

本文件回答一个问题：把 attention 变成**可被本仓判定**的算子族，需要定义什么、验证什么、边界在哪。
它不承诺工期，也不承诺 attention 会带来性能收益——那要等 C1–C5 走完并由 R2 的规则去判。

## 1. 为什么要它

现有受支持的四个算子（matmul / relu / softmax / layernorm）实测都贴着峰值（fp16 matmul 余量 6–8%），
先把候选做对、再做快这条路径**测不出插件增量**（见 [候选余量实测](../../status/v08/candidate-headroom-20261107.md)）。
attention 是唯一同时满足三件事的热点：

1. 同时吃算力与带宽 → **绑定轴不显然**，靠猜容易选错；
2. 对 mask、block 大小、S 是否为 2 的幂敏感 → 诊断有东西可说；
3. 是 LLM 每层都有的真实路径 → 结果对用户有意义。

## 2. 范围（Scope A）

| 项 | 纳入 | 明确排除 |
|---|---|---|
| 形态 | **prefill**：S_q = S_kv = S | decode（S_q=1，被 launch 地板吃掉，测不了）|
| 头 | MHA：H_q = H_kv = 16，B=1 | GQA / MQA |
| 头维 | D = 128 固定 | D=64、其它 |
| 序列 | S ∈ {1024, 2048, 4096} | 非 2 的幂（作为**派生 case** 单独加，见 §6）|
| dtype | fp16 输入输出，fp32 累加 | fp8、bf16、fp32 |
| mask | 无 mask（全注意力）、**纯 causal**（j ≤ i）两种 | 加性 bias、dropout、滑窗、paged |
| 布局 | contiguous 主用例 + strided 派生用例 | 任意非对齐 offset |

**为什么排除 decode**：S_q=1 时 kernel 只有几微秒，`launch_dominance` 会到 0.5 以上，
measurement 判 `uninformative`，跑出来的数字不能做任何闭环决策。这不是"以后再说"，是**结构上不可测**。

## 3. 语义（必须写死，否则"评的是什么"会漂移）

对每个 (b, h)：

```
S = Q·Kᵀ / sqrt(D)            # 形状 [S, S]，fp32 累加
P = softmax(S + mask, axis=-1) # mask: 全 0 或 causal(-inf for j > i)
O = P·V                        # 形状 [S, D]
```

冻结项：

- **缩放因子** `1/sqrt(D)`，D 是 head 维，不是序列长度；
- `softmax` 沿最后一维、数值稳定实现（减行最大），**不要求**候选与参考的归一化顺序一致；
- **无 affine、无 dropout、无 bias**；
- 输入 fp16、中间 fp32 累加、输出 fp16；
- `causal` 通过**标量参数按值直传**（`SSHExecutor._serialize_inputs` 的 `scalars` 通道），
  绝不允许走"shape+seed 重新生成"那条路——那会把 `True` 变成随机布尔（2026-10-01 真机踩过）。

## 4. 布局与容量

张量四个：`Q`、`K`、`V`（输入）与 `O`（输出），均为 `(B, H, S, D)` fp16。

| S | 单张量字节 | 四张量合计 | 256 MiB 上限 |
|---|---|---|---|
| 1024 | 4.2 MB | 16.8 MB | ✅ |
| 2048 | 16.8 MB | 67 MB | ✅ |
| 4096 | 67 MB | 268 MB | ✅（单张量 67 MB < 256 MiB；**注意合计不受限，单张量才受限**）|

派生用例的 strided 变体由现有 `bench_cases` 机制生成，但必须**真机回读** shape/stride/dtype，
并把回读结果写进 case 证据——这是 C1 的验收条件，不是可选项。

## 5. 参考实现与容差

### 5.1 参考

参考侧用 `torch.nn.functional.scaled_dot_product_attention`（**参考侧可用原生算子**；反作弊只约束候选）。
但参考必须被登记为**受信任定义**——当前 `as_reference()` 只认唯一一个注册项（layernorm），
所以 attention 需要新增一项受信任定义，含：

| 字段 | 取值 |
|---|---|
| `definition` | `native-pytorch-sdpa-mha-prefill-fp16-v1` |
| `input_dtype` / `output_dtype` | fp16 / fp16 |
| `candidate_accum` / `reference_accum` | fp32 / fp32 |
| `variance` | population（softmax 分母是全部 S 项，causal 时为 j ≤ i 的项）|
| `epsilon` | 不适用（softmax 内部稳定化不暴露为参数）→ 记 `0.0` 并在定义里说明 |
| `fp_fusion` | `forbid`（候选不得调用原生 SDPA）|
| `comparison` | 逐元素，按下面的容差 |

### 5.2 容差（**待真机标定，不许先填数**）

fp16 attention 的误差来源与现有四算子不同，至少三条叠加：

1. `P·V` 沿 S 累加（S=4096 时长行）；
2. softmax 的指数与归一化在 fp16 输入下本身有量化；
3. causal 时每行有效长度不同（第 0 行只有 1 项，最后一行 4096 项），**误差随行号单调变化**。

因此容差必须按"行号桶"标定，而不是一个全局 `rtol`。C2 的产出应包含一张
`(S, D, causal) → (max_rel_error 分布 p50/p95/p99, matched_ratio)` 的实测表，
再由它反推 `rtol` 与 `required_matched_ratio`。

**在这张表出来之前，任何写进代码的容差数字都是猜测**，本仓不允许（`default_policy` 对未知算子直接抛错，
正是为了不让"猜的语义"静默生效）。

## 6. 正确性套件（派生 case 清单）

| # | case | 判据 | 为什么必须有 |
|---|---|---|---|
| 1 | `S=1024, causal=False, contiguous` | 全量比对 | 主用例 |
| 2 | `S=1024, causal=True, contiguous` | 全量比对 | mask 语义 |
| 3 | `S=2048, causal=True` | 全量比对 | 规模变化 |
| 4 | `S=4096, causal=True` | 全量比对 | 长行累加误差 |
| 5 | `S=1000`（非 2 的幂） | 全量比对 | 尾部 block 的 mask |
| 6 | `S=4096, causal=True, strided` | 全量比对 | 布局假设（八格里 `relu-small` 就死在这）|
| 7 | 全零 / 常数 Q,K,V | 全量比对 | 除零与 NaN 传播（`1/sqrt(D)` 不涉及 0，但 softmax 全等行会）|
| 8 | 极值输入（large） | 全量比对 | 溢出与稳定化 |

case 1–8 全部通过才算"完整 L2 通过"；任何一条失败，L3 不得作为性能结论使用。

## 7. 反作弊

已覆盖（`decide/anticheat.py`）：

- `torch.scaled_dot_product_attention`、`F.scaled_dot_product_attention`、`.scaled_dot_product_attention(`
- `bmm` / `matmul` / `einsum` / `softmax` / `log_softmax`
- `tl.dot` 不被误杀（`_KNOWN_MODULE_ROOTS` 让 triton 路径走白名单规则）

C3 需要补的：

| 待补 | 原因 |
|---|---|
| `torch.nn.attention.*`、`torch._scaled_dot_product_*` | 内部 API 绕道 |
| `flash_attn`、`xformers` | 第三方包装 |
| `torch.compile` 包裹原生 SDPA | 静态审计看不穿编译图——需在 runner 侧加运行期检查或直接禁 `torch.compile` |

## 8. roofline 口径（三选一，需你定）

attention 是"两个 GEMM 夹一个 softmax"，现有 `_roofline_analysis` 按 A/B 两矩阵算算术强度，
套上去会给出错误结论。三个选项：

| 选项 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A 报 `unknown`** | 不建模型，`bound_type='unknown'` | 最诚实，零成本 | optimize 诊断失去轴（这次的主角缺席）|
| **B 粗估模型（建议）** | `FLOPs = 2·B·H·2·S²·D`（causal 乘 0.5），`bytes = 4·B·H·S·D·2`，AI = FLOPs/bytes，据此判 compute/memory | 能给出轴与达成率 | 忽略 softmax 与中间搬运，**必须标注"粗估"** |
| C 精确 kernel 级模型 | 分块 roofline + 每次 tile 的读写 | 最准 | 成本高，且依赖候选实现细节，会奖励特定写法 |

建议 **B**，并把 `roofline_analysis.reference_lower_bound` 保持为 `false`、
在诊断里走 `audited_peak` 口径（余量算得出来、但注明是规格峰值）——与 2026-11-07 真机那次一致。

## 9. 检查点与估时

| # | 产出 | 验收 | 估时 |
|---|---|---|---|
| C1 | `_attention_case` 布局 + 真机回读 | 远端读回 Q/K/V 的 shape/stride/dtype 与本端逐项一致 | 0.5 天 |
| C2 | 受信任参考 + 容差标定表 | 同环境两次运行参考逐位一致；跨环境身份可核；§5.2 那张表 | 1–1.5 天 |
| C3 | 套件 case 1–8 + 反作弊补充 | 每条 case 有判据与失败归因；作弊写法被拦 | 1 天 |
| C4 | roofline 选项 B + 标注 | 与手算一致；`reference_lower_bound=false` | 0.5 天 |
| C5 | 端到端一个任务跑通 L1/L2/L3 | 拿到第一份 L3 与绑定轴，且 validity 不是 `uninformative` | 0.5 天 |

合计 **3.5–4 天**；C2 是经验活，实际可能到 5 天。

## 10. 边界：这份设计不证明什么

- 不证明任意 S/D/H/mask 组合正确；causal 只覆盖 `j ≤ i`，不含滑动窗与加性 bias。
- 不覆盖 decode、GQA/MQA、fp8、paged KV cache。
- 不证明 attention 能带来收益；它只把 attention 变成**可判定**的对象。
- 粗估 roofline 给出的"达成率"是估算，不是实测峰值口径（与 `peak_measured` 的区分见
  `measurement.py` 的同名注释）。

## 11. 已决定（2026-11-07）

| 项 | 决定 | 影响 |
|---|---|---|
| 真机 | **可用**，容差标定按 C2 留窗口 | C2 的产出是实测表，不是估值 |
| roofline | **选 B（粗估模型）** | `bound_type` 由 `FLOPs/bytes` 与脊点比值得出；`reference_lower_bound=false`，诊断走 `audited_peak` |
| causal | **只做 `j ≤ i`** | 滑窗/加性 bias 不进 Scope A |

## 12. C1 的第一个发现：`causal` 不能当张量传

设计文档原本打算把 `causal` 作为 0 维张量塞进 case。核对实现后确认**这条路会静默出错**：

- `BenchmarkCase.materialize()` 委托给 `materialize_seed_inputs()`，按 `seed` 造值；
  浮点抽 `standard_normal`、**整数抽 `integers(1, 8)`**；
- 所以一个 `shape=()`、`dtype=int64` 的 `causal` 会被物化成 1..7 之间的随机整数，
  真值 `0/1` 丢失 —— 这正是 2026-10-01 真机踩过的"标量被随机值顶掉"那一族；
- 而远端执行器**已经**有标量按值直传的通道（`_serialize_inputs` 的 `scalars`），
  缺的是控制端把标量放进 case 与契约的途径。

因此 C1 的正确做法是给 `BenchmarkCase` 增加一个**有类型的 `scalars` 字段**，触达点：

| 位置 | 改什么 |
|---|---|
| `BenchmarkCase` | 新增 `scalars: dict`；只允许 `bool/int/float/str`，拒绝数组与容器 |
| `content_dict()` / `to_dict()` / `load_dict()` | 标量进契约哈希与落盘往返（**不进哈希就等于没声明**）|
| `seed_spec()` → `materialize_seed_inputs` | 标量原样透传，不参与抽样 |
| `wrapper_calling_convention` | 把标量渲染成普通参数 |
| `input_fidelity` 回读 | 标量进 digest，否则保真检查漏掉它 |
| 契约与既有 case | 老 case 没有该字段时行为不变（三态：缺失 ≠ 空）|

这是一处 schema 变更，影响契约哈希；但它是 attention 的**前置**，而且顺带把
"带标量参数的算子"（causal、eps、scale）整类打开。

C1 的验收因此改写为两条：

1. `Q/K/V` 三张量的 shape/stride/dtype 在远端回读与本端一致；
2. `causal` 作为标量在远端**取到的是 `0/1`**，而不是随机整数。