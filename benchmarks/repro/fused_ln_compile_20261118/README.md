# 失败回放：fused residual+LayerNorm 的 Triton 编译失败（27B/medium，A 臂）

**一句话**：裸 DSH 臂用 27B/medium 交付的融合核在 L1 通过、L2 死在启动期编译
（`tl.arange` 的 bound 不是 2 的幂）；修掉编译错后又暴露出第二个缺陷（补齐 lane 被算进方差）。
两个缺陷都修好后，同一份冻结契约下 32/32 通过，独立终评（`purpose=final`）L2 也是 32/32。

- 采集时间：2026-11-18T00:15:49+08:00
- 来源：`output/ac-fused-smoke-A4`（A 臂正式试跑格，计划 `operator-v08-workflow-ac-v8`）
- 原始回执：[`result-0001.json`](../../../output/ac-fused-smoke-A4/run/native-verification/result-0001.json)
- 原始候选：本目录 [`candidate.py`](candidate.py)，sha256 `55110ad2c532…`（与回执逐字一致）
- 契约：`093339587d8c462c…`，套件 `652f65b1…`（32 case），profile
  `builtin-fused-layernorm-residual-fp16-cuda-v1`

## 一条命令重跑

```bash
# 1) 复现失败（应输出 passed=true：与冻结的预期一致）
python -m benchmarks.experiments.v08.failure_replay \
  --case benchmarks/repro/fused_ln_compile_20261118 \
  --source candidate.py --host h20 --work-dir /root/operator-p2-seed-20261116

# 2) 验证补丁（含独立终评 L2 回执）
python -m benchmarks.experiments.v08.failure_replay \
  --case benchmarks/repro/fused_ln_compile_20261118 \
  --source candidate_fixed.py --expect pass --final \
  --host h20 --work-dir /root/operator-p2-seed-20261116
```

两条命令都用生产同一套入口（`operator_agent.app.hitl_tools.prepare/verify`），不是第二套判据。
回放每次新建 run 目录（host task 的验证预算是 write-once 的）。

## 缺陷 1：`tl.arange` bound 不是 2 的幂（编译期）

`candidate_compile_failure`（L2 回执里原样带着）：

```json
{"producer": "triton.compiler.errors.CompilationError",
 "phase": "candidate_launch", "line": 30, "column": 11,
 "helper_name": "_fused_ln_residual_kernel"}
```

第 30 行是 `cols = tl.arange(0, NBLK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]`。
wrapper 在 `N=6144` 时给出 `BLOCK=1024, NBLK=cdiv(6144,1024)=6` —— **6 不是 2 的幂**，
Triton 拒绝编译。L1 只做导入/语法检查，所以这一条只能等到 L2 启动期才炸。

## 缺陷 2：补齐 lane 被算进方差（数值）

只修缺陷 1 之后，32 条 case 里 7 条挂（全部是 `N=33` 这种尾块）：

```
fused-ln-residual-8x33-near_constant-contiguous  max 4.79e-01  matched 0.0000  形态=whole_tensor
fused-ln-residual-1x33-random-contiguous         max 9.96e-02  matched 0.1515  形态=tail_block
...
```

原因：`s = r + x` 在 pad lane 上是 0，`diff = s - mean` 于是给每个 pad lane 一个 `-mean`，
`var = tl.sum(diff*diff)/N` 把这些 pad 贡献也加进去，方差被抬高、输出被压扁。
修法是中心化后再按 `cols < N` 归零。

## 补丁

见 [`patch.diff`](patch.diff)（两处、共 6 行）：

| 位置 | 改动 |
|---|---|
| kernel 内 `diff` | `diff = tl.where(cols < N, s - mean, 0.0)` |
| wrapper 内 `NBLK` | `triton.next_power_of_2(triton.cdiv(N, BLOCK))`，并把 `EXACT` 收窄为「无 padding 且 NBLK 恰好等于 cdiv」 |

## 回执

| 文件 | 内容 |
|---|---|
| [`replay-candidate.json`](replay-candidate.json) | 原候选复现：`L1=true / L2=false`，`CompilationError` line 30 col 11，`passed=true`（与冻结预期一致） |
| [`replay-candidate_fixed.json`](replay-candidate_fixed.json) | 补丁：`L2=true`，32/32；`final_levels.L2=true`（独立终评） |

## 范围与边界

- 只覆盖这一份候选、这一条冻结契约与 32 条套件；不证明其它形状/实现。
- 回放判据包含身份：契约哈希、修前源码哈希、级别与具名用例，四项全对才算复现成功。

回放证明「同样的输入与候选仍然得到同样的失败」，以及「补丁在同一判据下通过」；
  它不构成性能主张（L3 未在本回放里跑）。
- 缺陷 2 是在修掉缺陷 1 之后才暴露的：**修完一个错才会看到下一个**，这也是为什么
  回放要绑定契约与套件，而不是只比对一行报错。
