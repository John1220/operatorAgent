# HITL `bench` / `sweep`：把「调」这一步还给判定层

> 状态：**M1 已实现**（`operator_bench`，2026-10-25）；M2（`sweep`）/ M3（stages 轴）待做。
> 触发事实来自一次真实使用（见 §0），不是设想出来的需求。

## 0. 问题：插件被当成裁判，不是教练

一次真实会话（79 次工具调用）的分布：

| 工具 | 次数 | 干了什么 |
|---|---|---|
| `bash` | 45 | scp 上传、扫 tile、量 cuBLAS 基线、三组隔离实验 |
| `write`/`edit`/`read` | 16/4/4 | 手写 12 个 Triton 变体与实验脚本 |
| `operator_verify` | 3 | 拿官方裁决 |
| `operator_capabilities` / `operator_prepare` | 1 / 1 | 体检、冻契约 |
| `operator_run` / `repair` / `status` | **0** | 未使用 |

最终成绩：手搓路线 **1.0568x**（我用同一份契约 `ee6734fa…` 独立复核确认，L1/L2/L3 全过、
L2 误差 0、`validity=valid`）；而插件自己的 `operator_run`（2 轮、15 个配方）只有
**1.0036x**。差 5.3 个百分点**全部来自搜索**，判据没问题。

根因不是"插件判得不准"，而是**唯一能出数的调用太重**：

- `verify` 是**裁决级**调用：deploy + 环境自检 + 重测基线 + 15 条 L2 case ≈ 40–60 s，
  且要求一个新 run_dir 一个回合；
- 调优要几十到几百次测量。10 秒一轮的手搓脚本 vs 40 秒一次的 `verify`，agent 必然绕开。

**缺的是一层「便宜、可编程、但不可裁决」的测量接口。**

## 1. 不变量（不可动）

1. **只有 `verify` 能产出可裁决读数**；`bench` 的读数在**结构上**不可裁决。
2. **正确性凭据只能来自 L2**。`verify_l3_only(correctness_verified=…)` 已经把这条写死
   （「错误候选不得产生可用于晋级的性能观测」），`bench` 不得绕开它。
3. **基线与候选同一测量窗口**（§11-2 / CHECKLOG I47 的 `arm_windows`）。跨窗口**不复用
   基线数字**——那是这次设计最容易犯的错（见 §8）。
4. 判据不可由调用方放宽：`FORBIDDEN_KEYS`（`atol`/`max_error`/…）不变。

## 2. `operator_bench` —— 批量测量（探索级）

```python
operator_bench(
    run_dir,                                  # 必填：复用已 prepare 的契约
    candidates=[{"name": "c1", "code_file": "..."}, ...],   # 1..N，code 或 code_file
    backend="ssh", host="algorithm-h20",
    target_gflops=0.0,
    persistent=True,                          # 复用 runner_server（省每次 ~4 s）
    deadline_s=120,
)
```

**语义**：一次调用 = **一个测量窗口** = 1 次基线 + N 臂交错采样。沿用
`benchmark_with_baselines` 的配对与窗口 id，只是把「2 臂」推广成「N 臂」。

**回包**（注意：**没有** `levels` / `route` / `eligibility` / `passed`）：

```python
{
  "ok": True,
  "grade": "exploration",          # ← 结构上的分级标记
  "window_id": "...",              # 同一批 N 臂共享
  "baseline": {"type": "pytorch_eager_fp16", "median_ms": 1.037, "samples": 30},
  "results": [
     {"name": "c1", "median_ms": 0.981, "speedup": 1.0568, "gflops": 140060,
      "validity": "valid", "validity_reasons": []}
  ],
  "best": "c3",
  "note": "探索级读数：不可用于裁决/入库/晋级；裁决请用 operator_verify"
}
```

**为什么不能给 `eligibility`**：资格位是「这个数能不能当依据」的闸门。一旦 `bench` 能出，
agent 就会拿它替代 `verify`——这正是本次会话的动机。所以 `bench` 的回包**连字段都没有**，
不是"有但为 false"。

## 3. `operator_sweep` —— 声明式网格，产出**可 verify 的源码**

```python
operator_sweep(
    run_dir,
    template="tile_matmul",       # 或 code/code_file：含 {BLOCK_M} 等占位符的源码
    space={"BLOCK_M": [64,128], "BLOCK_N": [64,128], "BLOCK_K": [32,64],
           "num_warps": [4,8], "num_stages": [1,2,3,4]},
    max_candidates=48,            # 去重/剪枝后的上限
    top_k=3,                      # 留几个进 survivors
    prune={"smem_bytes": 227*1024},
    backend="ssh", host="algorithm-h20", persistent=True,
)
```

**回包**：`tried`（实际测了几组）、`ranked`（每臂读数，复用 `bench` 的字段）、
`survivors`（**自包含源码**，可直接 `operator_verify(code=…)`）、`dropped`（被剪掉的原因）。

两条实现路，**选 ①**：

1. 层内生成 N 份源码 → 交给 `bench` 测。计时由本层控制，读数口径与 `verify` 一致。
2. 生成一个 `@triton.autotune` 版本、跑一次。更快，但计时是 Triton 自己的
   `do_bench`，**不受本层有效性判据约束**——只可作为 `sweep` 的粗筛，不能回填读数。

**为什么要输出源码而不是只输出参数**：`verify` 需要一个自包含候选，接口约定是
`kernel(A, B) -> C`（`run_kernel.py:340`）。本次会话为此多花了一轮——它得先发现这个约定，
再手写包装。`sweep` 直接产出可 verify 的源码，这一步摩擦就消掉了。

## 4. 执行侧改动（四处，均为增量）

| 文件 | 改动 |
|---|---|
| `remote_scripts/run_kernel.py` | 新增 action `bench_multi`：N 臂 + 1 基线、块间交错、回报 `arm_windows`；`run_action` 已有单一分派，按现有「每一支只读自己用到的键」的约定加一支 |
| `operator_agent/remote_executor.py` | `bench_multi(candidates, inputs, …)`，走 `persistent` 会话；`benchmark_with_baselines` 保留不动 |
| `operator_agent/hitl_tools.py` | `bench()` / `sweep()`；`sweep` 内部调 `bench` |
| `operator_agent/mcp_server.py` | 两个工具定义 + 说明里写死「探索级，不可裁决」 |

## 5. 四条防滥用守卫

1. **结构守卫**：`bench` 回包无 `levels`/`route`/`eligibility`/`passed`（契约测试钉住）。
2. **凭据守卫**：`verify` 不接受 `bench` 读数当 L2 凭据（`verify_l3_only` 已有，测试补一条）。
3. **落盘守卫**：`bench-N.json` 与 `verify-N.json` 分开，`claim`/`manifest`/`kernel_db` 只认后者。
4. **预算守卫**：`max_candidates` + `deadline_s`；超时返回 `partial: true`，截断的网格
   **不许**当完整结果（回包里明写 `tried < requested`）。

## 6. 三条测试

1. N 臂共享同一个 `window_id`，且 `bench` 回包**不含**裁决字段；
2. `sweep` 的 `survivors` 直接喂 `operator_verify` 能过（端到端，假 CLI + 真契约）；
3. 同一 run_dir 先 `bench` 后 `verify`：`manifest` 只记 `verify` 那一轮，`claim_level` 不变。

## 7. 收益（用本次实测代入）

本次赢的配置是 `64×64×64 / w4 / s4`；模板空间 `DEFAULT_STAGES=(1,2,3)` 上限是 3，
**结构上搜不到**。

- 网格规模：BM/BN ∈ {64,128}、BK ∈ {32,64}、warps ∈ {4,8}、stages ∈ {1,2,3,4}
  ⇒ 2×2×2×2×4 = **64 组**；按 smem 上限与 tile 合法性剪枝后仍在 40 组以上
  （`128×128×64` 单 stage 32 KB、4 stages 128 KB，H20 的 228 KB smem 放得下，
  所以剪枝主要来自小 tile 的低效组合，不是显存）；
- `persistent` 下单臂 ~0.3–0.5 s，基线一次 ~2 s ⇒ **20–30 s** 覆盖整网格；
- 对照：本次手搓用了 45 次 bash（每次 scp+ssh ~2–5 s）≈ 2–4 分钟，外加 12 个脚本。

差别不只是快：**它能覆盖 stages=4 这一轴**，所以这次有机会自己搜到 1.05x。

## 8. 明确不做

- **不缓存基线数字**。只缓存部署与 GPU 绑定；基线每个 batch 重测一次。
  跨 batch 复用基线会破坏「同窗口」不变量，是比慢更难查的错。
- **不让 `bench` 产出 claim / 入库**（见 §5 守卫 2、3）。
- **不替代 `verify`**：`bench` 只回答"哪个方向更好"，不回答"能不能交付"。
- **不并发抢卡**：同一窗口内交错采样，不开多进程并行——那会把读数变成不可比。

## 9. 落地顺序

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M1** | `bench`（N=1 也有用）+ 四条守卫 + 测试 1、3 | 单次测量从 40 s 降到 ~1 s，且裁决路径不变 |
| **M2** | `sweep` 网格 + `survivors` → `verify` 闭环 + 测试 2 | 一次 sweep 覆盖 40 组，survivors 直接被 verify 接受 |
| **M3** | 模板空间补 `stages=4/5`（依赖 M2 验证） | 4096³ 上 `operator_run` 能自己达到 ≥1.05x |
## 10. M1 实现与实测（2026-10-25）

实现选择：**没有新增 runner action**。`Verifier.verify_l3_only()` 已经是"只跑 L3"的入口，
而 `benchmark_with_baselines` 内部就是同窗口交错配对 —— 所以 `bench` =
`_make_backend(persistent=True)` **一次** + 逐臂 `verify_l3_only`。好处是：

- 执行侧与反作弊的结构性守卫（`_bench_pair_interleaved` 的 `shield_torch` 字面区间、
  形参位置）**一行没动**；
- 每臂拿到的是与 `verify` 同一把尺子量出来的 `measurement_validity`，不是第二套口径。

实测（H20 / GPU 5 / 4096³ fp16 / 契约 `ee6734fa…`，一次 `operator_bench` 调用）：

| 臂 | median | speedup | GFLOPS | validity |
|---|---|---|---|---|
| 手写 `64×64×64 / w4 / s4` | 0.981776 ms | **1.0575x** | 139 990 | valid |
| 插件生成 `128×128×64` | 1.030240 ms | **1.0045x** | 133 405 | valid |

`best` 选中手写那版；回包 `grade=exploration`、无 `levels`/`route`/`eligibility`/`passed`。
另两次独立 `operator_verify` 分别是 1.0568x / 1.0036x —— 三把尺子互相吻合。

**成本**：单臂 17.7 s（其中部署 + 环境自检占大头），对照 `verify` 的 40–60 s 且还要跑
15 条 L2。批次摊薄后每臂增量主要是测量本身。

**还没做的**（M2/M3）：`sweep` 网格入口、`stages≥4` 进入模板参数空间。所以这次 A/B 的
两个候选仍是**人给的两份源码**，不是插件自己搜出来的。
