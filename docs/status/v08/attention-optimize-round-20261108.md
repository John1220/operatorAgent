# attention 的 optimize 轮：真机诊断 → 三次实验 → 选中（2026-11-08）

| 项 | 值 |
|---|---|
| 机器 | `h20` GPU 3（与 1 个其他进程共享），隔离 runner `torch 2.14.0+cu130` |
| 任务 | `fp16 attention prefill 1x16x4096x128`，契约 `e9f75d60cf1c2b35…` |
| 门控 | `attention-1x16x4096x128-fp16-nocausal-cuda-event`，有效带宽 `0.1%`，`action=start` |
| 结论 | 诊断给的轴（compute_bound）成立，但**tiling 两个方向都更慢**；`num_stages 2→3` 两次测得 **−2.14% / −1.86%** |
| 证据 | `output/v08-attention-20261108/c5/optimize-round-20261108/`（门控 2 份、六次回执、四份源码） |

这是 attention 第一次跑 `operator_context(phase=optimize)`，也是第一次在同窗口里做
"诊断 → 实验 → 选择"的闭环。

## 0. 先修一个真缺陷：optimize 上下文建不出来

第一次调用直接失败：`host context byte limit exceeded`。体是 **41,294 字节**，上限 40,960 ——
只超 334 字节，但足以让 arm C 完全拿不到上下文（attention 的 8 条 case 让 `authoritative` 变大）。
分项：materials 18.8 KB、authoritative 8.1 KB、rules 5.2 KB、candidate 2.7 KB、feedback 2.4 KB。

修法：超限时**裁最不承重的一块并如实登记**，不是把上限调大。裁的是角色模板正文
（`rendering=raw_reference_template_not_messages`，宿主本来就自己组装消息），
名字/角色/sha256 全留着——需要原文的人仍能按哈希取回。`material_integrity` 里
`truncated=true` + `dropped=['materials.templates[].text']`。裁完还超限就继续抛错
（不许悄悄扔掉领域材料）。有 3 条测试钉住：裁哪块、身份不丢、裁不动时 fail closed。

## 1. 诊断说了什么

```json
{"bottleneck": {"binding": "compute_bound", "attainment_key": "gflops_utilization_pct",
                "attainment_pct": 76.6},
 "decision_band_pct": 0.1,
 "headroom": {"basis": "audited_peak", "remaining_pct": 23.4,
              "note": "against a spec-audited peak, not a measured one"},
 "experiments": [
   {"axis": "tiling", "change": "raise per-program work (BM/BN/BK) until SMEM or occupancy stops the growth"},
   {"axis": "pipelining", "change": "increase num_stages/num_warps to overlap loads with the MMA pipeline"}]}
```

两条实验轴，三次单变量实验（每次只改一个参数，起点在同一窗口内复测）：

| 变更 | 中位 | 相对起点 | 加速比 vs SDPA | L2 / 有效性 |
|---|---:|---:|---:|---|
| **起点** 64/64, warps=4, stages=2 | 1.212160 ms | — | 0.8610× | 通过 / `valid` |
| `BLOCK_M 64→128` | 1.631952 ms | **+34.63%** | 0.6395× | 通过 / `valid` |
| `BLOCK_N 64→128` | 1.343488 ms | **+10.83%** | 0.7788× | 通过 / `valid` |
| **`num_stages 2→3`** | 1.186208 ms | **−2.14%** | 0.8805× | 通过 / `valid` |
| `num_stages 2→3`（复核） | 1.189568 ms | **−1.86%** | 0.8796× | 通过 / `valid` |

起点在本次窗口复测为 1.212160 ms，与两小时前那次 `1.212096 ms` 相差 0.005% —— 同卡同窗，可比。

## 2. 为什么 tiling 反而更慢

`D=128` 时累加器是 `BLOCK_M×128` 的 fp32：`BLOCK_M` 从 64 到 128 直接把每个 program 的
累加器翻倍（8 KB → 16 KB/列组），4 warps 下寄存器压力与 occupancy 一起变差，实测 +34.6%；
`BLOCK_N` 翻倍虽然不动累加器，但 K/V 分块与掩码的工作集变大，也 +10.8%。
**"提高每 program 的工作量"这条假设在这个核上不成立** —— 它默认寄存器/共享内存还够用。

`num_stages` 加一级则只是在同一份 tile 上多做一层软件流水，−2.0%（两次一致）。
方向对了，收益只有 2%，离"减少 23.4% 余量"还很远。

## 3. 这次能得出的结论与不能得出的

**能**：在同一窗口、同一张卡上，`num_stages=3` 比起点快约 2%，两次测量方向与量级一致，
且两次都通过完整 L2、L3 都 `valid`。这是"诊断提出了轴，实验选出了一个配置"的完整证据链。

**不能**：
- −2% 超过判决带宽（0.1%）20 倍，但**低于 R2 登记的 2.5% 配对效应**。它对"诊断是否让模型
  更容易选到好配置"这个问题不作回答——那要配对。
- 达成率的分母是规格峰值（`audited_peak`），23.4% 余量是粗估模型算出来的，不是实测峰值口径。
- 三次实验都在共享卡上（1 个其他进程常驻），窗口读写自同一份门控回执；换窗口要重做。
- 没有跑 `verification_purpose=final`：那需要 host task 的终评预留，本轮的 repo 是 CLI 直跑。

## 4. 顺带记一条计划层面的缺陷

R2 的 attention 计划按 "B = v0.7.2 / C = v0.8" 登记，但 **v0.7.2 根本准备不了 attention 任务**
（没有这个 profile）。所以这对比较按字面跑不起来，可行的形式是**同一份 v0.8、开关 optimize 诊断**
（即 `diagnosis-ab-ieee-20261108` 那种 `OPERATOR_R2_DIAGNOSIS=off/on`）。这一条要在开跑前重新登记，
不能直接拿现有计划开。
## 5. 措辞收口：实验表是维度菜单，不是方向指令

这一轮同时暴露了一个**措辞层**的问题。插件给的两条实验里，`tiling` 那句原文是：

> raise per-program work (BM/BN/BK) until SMEM or occupancy stops the growth

而三次实验里有两次正是被这句话带着走、并且更慢。方向不是测量能回答的——它取决于累加器大小、
head 维与占用率，那要读源码。所以把 `_EXPERIMENTS` 的 `change` 全部改成**维度 + 观测量**：

| 轴 | 改前 | 改后 |
|---|---|---|
| tiling | raise per-program work (BM/BN/BK) until SMEM or occupancy stops the growth | tile shape (BM / BN / BK) at fixed arithmetic — either direction is a candidate |
| pipelining | increase num_stages/num_warps to overlap loads… | software pipeline depth (num_stages) and warp count around the current point |
| coalescing | make the innermost per-thread access unit-stride… | access order and index arithmetic on the innermost lane |
| vectorized_access | use 128-bit vector loads/stores… | transaction width of each load/store and base alignment |
| dtype_width | narrow the loaded dtype where the tolerance allows… | width of the dtype actually loaded |
| work_per_launch | increase elements per program and remove tiny grids… | elements per program (grid size against the dispatch floor) |

`POLICY` 同时补一句：实验表是 **MENU OF DIMENSIONS with expected signals, not instructions**，
每个旋钮往哪转取决于源码。原有边界（caller-owned hypothesis / 不发信用 / 对
`decision_band_pct` 判）一字未动。

**边界**：改的是宿主看到的那段文字，不是判据。副作用是 `context['hash']` 变了
（attention S=4096 从 `6930012e4c61c46d…` 变成 `e7775cbe12d1af52…`），所以旧回执的
`context_hash` 不再复现——这是必然，回执里本来就记着它。
