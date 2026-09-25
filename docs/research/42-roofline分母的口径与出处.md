# 42 — roofline 的分母：把「实测成绩」换成「dense 规格」，并让出处跟着数字走

对应缺口 3（docs/research/00 第三节）。日期：2026-09-26。真机：8×H20 的 GPU 2。

## 一、问题陈述在做的过程中被改写了

缺口 3 原来的写法是「硬件知识是静态查表、且散落在多处」。这条没错，但它不是要紧的那半。
要紧的半是 D4 之后才出现的：`measurement.roofline_converged()` 开始拿
「绑定轴达成率 ≥ 90%」当**提前停止优化循环**的出口，于是那张表里的 peak 从「注释里的
背景数字」变成了**判决的直接输入**。一个数字一旦有后果，它凭什么是真的就不再是洁癖问题。

所以这一轮的落点不是「把散落的知识收敛到一处」（那是架构整理，可以慢慢做），而是三件
更小的事：

1. 分母必须是**规格上界**，不能是某次库实现的读数；
2. 分母的**出处要出现在使用点**（日志），而不是只活在源码注释里；
3. 出处没核过的行，**不配开那个出口**。

## 二、触发点：一个不可能出现的 101.5%

doc 34/38 的真机读数里 matmul 4096³ fp16 跑到 140.0 TFLOPS，而当时表里 H20 的 fp16 写着
**138.2** —— 达成率 101.5%，比它自己声称的「上界」还高 2.4%。138.2 那行注释老实标着
「2026-09-16 于 algorithm-h20 实测 (cuBLAS 8192³)」。出处没有错，**用途错了**：一次库实现
的读数是「已达成的成绩」，成绩是可以被超过的，上界不行。

## 三、阶梯实测：「achievable peak」不是一个标量

`benchmarks/calibrate_achievable_peak.py`（提交 2468adc）在与 runner 同口径的计时下
（256 MB L2 冲刷排在 `start.record()` 之前，30 样本取 median）量了两条阶梯。2026-09-26
跑了两次，下面是第二次（GPU 2，torch 2.14.0+cu130，L2=63 MB，SM=78）：

| n (方阵) | fp16 TFLOPS | A+B+C | 装进 L2? | tf32 | fp32(TF32关) | D2D 拷贝 GB/s |
|---|---|---|---|---|---|---|
| 2048 | 116.37 | 25 MB | 是 | 64.32 | 24.66 | 8 MB → 1225 |
| 3072 | 136.57 | 57 MB | 是 | 69.43 | 25.79 | 64 MB → 2866 |
| 4096 | 132.41 | 101 MB | 否 | 71.00 | 30.41 | 256 MB → 3217 |
| 6144 | 140.16 | 226 MB | 否 | 71.41 | 25.63 | 1024 MB → 3324 |
| 8192 | 137.73 | 403 MB | 否 | 71.72 | 31.58 | 4096 MB → 3357 |

四条读法：

- **跨 shape 相差 20% 以上**（fp16 116.4→140.2 = +20.4%，bf16 +20.8%，fp32 +28.1%），
  而 `NOISE_BAND[DEGRADED]` 是 0.4%。差两个数量级 —— 所以「一个 dtype 一个峰值标量」这个
  口径本身就是错的，达成率会随 shape 系统性偏高或偏低。
- **带宽随 footprint 爬 2.8 倍**（8 MB 的 1225 → 4 GB 的 3357 GB/s）。所谓「H20 带宽
  3.68/4.0 TB/s」是在很大的拷贝上才成立的读法。
- **L2 常驻假设再次被数据推翻**：原本推测「3072² 装得下 63 MB L2，所以 138.2 这个
  8192³ 上的读数在 4096³ 偏低」。实测装得下的 2048/3072 是**最低**的两档（116/137），
  装不下的 6144 最高（140）。方向反了，这条假设不成立。
- **两次连续运行的复现性**：fp16 各档相差 ≤0.2%，带宽除 8 MB 档（1196 vs 1225，差 2.4%）
  外 ≤0.5%。阶梯是真的，不是噪声。

第一次跑的数（同日，_PEAK_ATTAINABLE 曾经抄过一遍）：fp16 116.3/136.7/132.3/140.3/137.6，
带宽 1196/2853/3221/3324/3357。

## 四、外部锚点（本轮一手核过）

**① flash-attention `benchmarks/benchmark_attn.py`。** 逐字读到的 dense 表：

```python
_PEAK_BF16_FLOPS = {
    "A100": 312e12, "A6000": 309.7e12, "L40S": 362e12,
    "H100 SXM": 989e12, "H100 NVL": 835e12, "H100 PCIe": 756e12,
    "H200": 989e12, "H20": 148e12,
    "GB200": 2.5e15, "GB300": 2.5e15, "B300": 2.25e15, "B200": 2.25e15,
}
# get_peak_bandwidth: "A100": 2.0e12, "L40S": 0.864e12, "H100 SXM": 3.35e12,
#                     "H200": 4.8e12, "H20": 4.0e12, ...
```

以及它自己写明为什么硬编码：

> `# BF16 dense peak FLOPS from official NVIDIA spec sheets.`
> `# Checked against: num_SMs * tensor_core_clock * ops_per_SM_per_cycle.`
> `# The tensor core clock is NOT queryable (it differs from the max boost SM`
> `# clock reported by nvidia-smi), so we hardcode the spec-sheet values.`

三条有用信息：(a) **H20 = 148e12 / 4.0e12**，一个成熟项目用的就是 dense 规格而非实测；
(b) 它按「长键先匹配」处理设备名（`H100 SXM` 先于 `H100`），我们的 `_PEAK_BY_NAME` 已经是
同一个写法，不是巧合而是同一个约束（cc 9.0 分不开 H100/H200/H20）；(c) **张量核频率查不到**
—— 这条正好接上 doc 40：nvidia-smi 报的 1980 MHz 是 SM boost 上限，不能拿去推张量核峰值。
我们的 fp32=39.5 用 SM×FMA×1.98 GHz 推是合法的（FP32 走 CUDA core），fp16 那格**不能**这么推，
所以取 148 的规格值。

**② NVIDIA 规格页的 `* With sparsity` 脚注。** H100 页上大号字体的 1979 TFLOPS (FP16)、
989 (TF32)、3958 TOPS (INT8) 全是 2:4 稀疏值，dense 是一半。旧代码里 `h100` 行抄的正是大号
那几个字 —— 这解释了为什么 `_PEAK_BY_CC` 里 fp16 列会写着 1979，而那行注释自己标的是
`FP8=1979`（FP8 dense 才等于 FP16 sparse）。

**③ KernelBench-Hard 的一句量级描述**：人写的 Hopper 核常常只到峰值的 **60~85%**。
这条对**阈值**的含义有影响（见第六节），不是装饰性引用。

另外三条本轮顺带确认、但不作为改动依据的观察：MLPerf 的推理/训练榜单不以「距 peak 百分比」
为口径（它排绝对吞吐，且锁频）；LBNL 的 ERT 一类工作把「measured ceiling」当成要测的量而不是
查表的常数；Nsight Compute 的 `% of sustained rate` 实测可以超过 100%（因为 sustained 是另一套
口径）。共同点：**没人拿一次库实现的成绩当上界用**。

## 五、落地了什么

`operator_agent/verifier.py`：

- `_PEAK_BY_NAME` / `_PEAK_BY_CC` 全部改成 **dense 规格**，H20 → `(39.5, 74.0, 148.0,
  148.0, 4000.0)`，H100/H200 → `989.4 + 494.7(tf32)`、H200 带宽 4800，回退表的 `"8.9"`
  从「标着 H100 的 Hopper 数字」改成 Ada/L40S（8.9 是 Ada；同 cc 的 4090/A10 会偏高，
  偏高的方向只是让出口不开，见下）。
- 新增 `_PEAK_SOURCE` / `_PEAK_CC_SOURCE`：每行一句出处。**可信度不再另立一张表** ——
  出处字符串以 `⚠️` 开头就表示「口径未核」，`_detect_theoretical_peak` 据此算出
  `peak_trusted`。这样加新行时不写出处 = 默认不配拿来判停，这个默认值是故意设计的。
- `_detect_theoretical_peak` 的返回带上 `peak_source`/`peak_table_key`/`peak_trusted`，
  经 `_roofline_analysis` 进 metrics，并打印在每次 L3 的 `Roofline:` 行里。
- `conductor.py`：触发「已到 roofline，停止优化轮」那条 break 时，多打一行它依赖的分母
  （哪张表哪一行、fp16/bw 各是多少、出处一句话）。这是全循环里最贵的一次判定，以前它
  在日志里是一句没有分母的百分比。
- `measurement.roofline_converged()`：`peak_trusted is False` 时**不开**出口。用
  `is False` 而不是 `not`，老记录里没这个键时保持原行为。
- `hardware_caps.py`：`h100` / `h20` 两行的算力与带宽同步改成 dense 规格。这两处此前与
  verifier 的表**互相矛盾**：`remote_executor.py` 注释写着「H20 实为 4.0TB/s」，而它读的
  `hardware_caps` 里是 3680。
- 配方文本 `optimization_recipes.md`：`tf32x3` 那张表两列都用错了口径（H100 的 tf32 抄了
  sparse 的 989，H20 抄了实测的 71.6），改成 dense 的 494.7 / 74.0。**结论没翻转**
  （H100 上 tf32x3 从「4.9× 更快」变「2.5× 更快」，H20 从 0.76× 变 0.63×，方向一致），
  但 H100 那侧的余量被砍掉一半，配方里补了这句话。TMA 那条的「H20 算力 138.2/带宽 3.68」
  同步改数，**ridge point 两种口径都是 ≈37 FLOP/B**，所以「几乎所有算子 memory-bound」
  这条判断不变 —— 这很重要：改分母改的是「离上界多远」，不是「它属于哪一类」。

删掉的东西：`_PEAK_BY_CC` 的两个死键 `"9.0a"`/`"8.9-bw"`（提交 9c96d11，匹配逻辑永不命中）
和这一轮一度写进来的 `_PEAK_ATTAINABLE`。**后者是「算了没人读」的现行犯**：它是按
shape/footprint 查的阶梯，硬编码成第二张表就需要一个 shape 桶查询器，而那个东西的正确归宿是
缺口 2 的持久化每-shape 记录。表留在 `benchmarks/` 和本文档，不留副本。

## 六、真机验证（2026-09-26，GPU 2，`benchmarks/verify_measurement_validity.py`）

```
  name='nvidia h20' cc='9.0' launch_floor_us=20.24
  分母行 [h20] fp16=148.0 TFLOPS / bw=4000.0 GB/s — flash-attention dense 表 148/4.0 + 算术推导; 无公开官方页 (trusted=True)
  [Verifier] Roofline: compute (AI=1365.33 FLOP/Byte, peak=148.0 TFLOPS (fp16_tensor_core|分母[h20] ...), GFLOPS利用率=94.7%, 带宽利用率=2.6%)
  matmul 4096³: 绑定轴达成率 = 94.7%   roofline_converged = True (阈值 90%)
  layernorm 16384×6144: 达成率 61.4%   roofline_converged = False
```

不可能的 101.5% 消失了，同一个核现在是 94.7%（140.2/148），出口仍然打得开。

**但这次改动付了一个代价，而且它在 memory 轴上：**本机 D2D 拷贝实测最好 3357 GB/s，
除以新的分母 4000 = **84%**，低于 90 —— 也就是说在这台卡上，**memory-bound 核的收敛出口
永远打不开**，任何 layernorm/softmax 类任务都会把优化轮次跑满（由停滞出口兜底）。改之前
分母是 3680，同一个读数是 91.2%，是能出口的。

这是**故意接受的**：不对称在于分母偏小会把还能优化的核**提前**判成到顶（不可恢复的错误），
分母偏大只是多烧几轮（可恢复，且有 `max_optimize_rounds` 与停滞出口兜底）。正解不是把阈值
调回 84 去迁就一个更好的分母，而是按 shape 查这台卡在这个 footprint 上的可达上界 —— 那需要
缺口 2 的持久化记录，所以缺口 2 现在顺带挡住了一件具体的事。

`ROOFLINE_CONVERGED_PCT` 维持 90 不动，但注释里改写了它的含义，并写明两条轴的可达性不对称
（旧注释给的理由就是那个 101.5%，已经随分母一起作废）。

离线侧新增两个用例（`tests/test_conductor_stagnation.py`）：未核分母 → 跑满 4 轮不判停；
已核分母 → 第 2 轮判停且日志带出分母出处。两条都做过变异验证（删掉 `measurement` 里的门
或删掉 `conductor` 里那行 print，各自红一次）。

## 七、明确不采纳 / 不做的

| 项 | 决定 | 理由 |
|---|---|---|
| 把 `_PEAK_ATTAINABLE` 做成第二张硬编码表 | ❌ 删 | 按 shape 查的阶梯需要 shape 桶；没有查询者的表就是死数据（本轮先写后又删，正是这条） |
| `ROOFLINE_CONVERGED_PCT` 按轴分设（compute 90 / memory 84） | ❌ 暂缓 | 84 是本机 4 GB 拷贝的读数，换个 footprint 就不是它；这等于把实测成绩重新请回分母的位置 |
| 把 peak 出处注入 Designer/Fixer 的 prompt | ❌ 暂缓 | 出口判停已经有日志；prompt 侧要动模板与 token 预算，等缺口 2 有真记录时一起做 |
| 「用同期 cuBLAS 基线做分母」（doc 00 原方案的一半） | ❌ 不采 | 基线读数与候选同 shape，确实免掉标量假设，但它回答的是「比库快多少」，不是「离上界多远」；我们已经拿它算 speedup 了，再拿它当 roofline 会把两件事混成一个数 |
| 修 `_PEAK_BY_NAME` 里 5 个 ⚠️ 行 | ⏭ 不修 | 本机用不到；且它们现在**不会**造成后果（出口对它们关闭），修数字不如让「没核过」可见 |
| 删掉标着 H100 的 `"8.9"` 回退行 | ⏭ 改数不删 | 删了 roofline 分类在 Ada 卡上直接失效（peak=0），是更大的行为改动；本轮只把它从 Hopper 数字改成 Ada 规格并记上出处 |

## 八、还没解决的（记在这里，别当成已经好了）

1. 硬件知识仍然**散在三处**（`verifier._PEAK_*`、`hardware_caps._KNOWN_HARDWARE`、配方
   markdown 正文）。本轮让三处**口径一致**了，没有合并成一处。缺口 3 的后半句仍未做。
2. `_PEAK_BY_NAME["a6000"]` 写着 fp16 77.4，flash-attention 给 A6000 的是 **309.7e12**，
   差 4 倍 —— 这类行就是「⚠️未核」存在的原因。没有机器可验，先不臆改，但已记下具体差值。
3. fp32 那一列（H20 39.5）是算术推导值，第三方规格页写 44；两者需要 2.2 GHz 才能同时成立，
   与 H100 用 1.98 GHz 推 67 TFLOPS 的口径不自洽。取推导值，但它是**推的**不是查的。
4. 阶梯只覆盖了 fp16/bf16/tf32/fp32 的方阵与 D2D 拷贝，没有 attention/conv 类形状。
