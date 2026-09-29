# 正确性验证与 reward hacking 防护：一个缩小 14.93% 的核会被判「功能验证通过」

> 状态：✅ 2026-09-25　对应缺口 1（测量有效性）、缺口 6（核函数接口契约）
> 关联：[`03-反作弊与正确性防线实测.md`](03-反作弊与正确性防线实测.md)（洞 3 就是本文的量化工版）
>
> **标注约定**：✅ = 本人一手核实（读过源码 / 本机实测出数字）；⚠️ = 调研转述，本人未一手核对。
> 本文所有 ✅ 数字均可用 `/tmp/probe_l2_gate.py`、`/tmp/probe_gates.py` 复现。

---

## 一、本项目的 L2 数值门是纯绝对容差 ✅

`remote_scripts/run_kernel.py:415-429`：

```python
abs_error = (actual_f - expected_f).abs()
max_abs_error = float(abs_error.max().item())
denom = expected_f.abs().clamp_min(1e-12)
max_rel_error = float((abs_error / denom).max().item())
return {
    "success": max_abs_error <= max_error,      # ← 门禁只看绝对误差
    ...
    "max_rel_error": max_rel_error,             # ← 算了、报了，但从不参与判定
}
```

**相对误差被完整计算并上报，却不在判定式里**（`:420` 算、`:428` 报、`:422` 判）。

阈值来自 driver，三个都传 `5e-1`：

| 位置 | 值 | 注释原文 |
|---|---|---|
| `test_nvidia_matmul.py:114` | `5e-1` | "FP16 精度约 3.3 位有效数字, 4096 次累加误差较大, 放宽到 0.5" |
| `test_nvidia_layernorm.py:115` | `5e-1` | "FP16 精度放宽" |
| `test_nvidia_attention.py:123` | `5e-1` | "Flash Attention online softmax 精度稍低" |

而 `schemas.py:223` 的默认值是 `1e-5`——**相差 5 万倍**。默认值从未在真机路径上生效过。

### 实测：这个门 admits 什么 ✅

配方严格照抄 runner（输入 `_deserialize_inputs:261` 的 `standard_normal × 0.1 → fp16`，参考 `_reference_torch:296` 的 `A @ B`，比对在 float32 下做），shape 用 driver 真实的 4096³：

```
参考输出 |ref|: max=3.3499  mean=0.5108  p99=1.6492
→ 门禁 5e-1 相当于允许 14.9% 的最大幅值偏差

缩放因子   max_abs_err       判定     等价 rtol
    1.00    0.0000e+00     PASS      0.0000   ← 正确核
    0.99    3.3499e-02     PASS      0.0100
    0.97    1.0050e-01     PASS      0.0300
    0.95    1.6750e-01     PASS      0.0500
    0.90    3.3500e-01     PASS      0.1000
    0.80    6.6999e-01     fail      0.2000
    0.00    3.3499e+00     fail      1.0000   ← 全零核

二分下界：仍能通过门禁的最小缩放因子 = 0.850744
```

**一个把结果整体缩小 14.93% 的核，会被判「功能验证通过」。**

这比 doc 03 那个 `return A @ B` 更隐蔽：`A @ B` 需要 LLM 主动作弊，而 14.93% 的偏差**不需要任何恶意**——一个累加顺序不同、或用了 fp16 累加器的 kernel 就能自然滑进去，然后带着一个漂亮的 L3 加速比入库、污染 RAG。

---

## 二、为什么 `max_rel_error` 当初没接进门禁 ✅

不是疏忽。实测一个**合法但精度差**的 fp16 累加 kernel：

```
候选                   max_abs    max_rel   p99.9_abs
correct_fp32acc    0.000e+00  0.000e+00  0.000e+00
correct_fp16acc    1.217e-02  2.119e+04  4.043e-03   ← 正确核，max_rel 两万倍
```

`max_rel = 2.119e+04`。原因是参考输出里有接近 0 的元素，`abs/|ref|` 在这些点上爆炸——`clamp_min(1e-12)` 只会让它更炸。

**所以「把 `max_rel_error <= rtol` 加进门禁」这个最直觉的修法是错的**，它会把每一个正确核都判死。相对项必须用能容忍近零参考的形式。这也解释了原作者为什么算了却不用：他大概率撞过这个坑。

---

## 三、外部门禁在同一份数据上的判定 ✅

两种形式都读了一手源码：

**KernelBench v1**（`eval.py:83-101` 容差表 + `:799-805` 判定）✅

```python
PRECISION_TOLERANCES = {
    torch.float32: 1e-4,     # "By default for fp32, 1e-4 is used according to torchbench."
    torch.float16: 1e-2,     # "torchbench states for bf16 and fp16, use 1e-3 ... and 1e-2 if it's too strict"
    torch.bfloat16: 1e-2,
}
# 同一个值同时当 atol 和 rtol:
tolerance = get_tolerance_for_precision(precision)
if not torch.allclose(output, output_new, atol=tolerance, rtol=tolerance
```

即加性形式 `|a-e| ≤ t + t·|e|`，**逐元素全过**才算过。`num_correct_trials: int = 1`（`:398`）是默认值，语义为"correctness pass only if all trials pass"（`:420`、`:845`）。

**FlashInfer-Bench**（`bench/utils.py:96-123`）✅

```python
eps = 1e-8
abs_error = torch.abs(x - y)
rel_error = abs_error / (torch.abs(y) + eps)
exceeds_tol_mask = (abs_error > cfg.atol) & (rel_error > cfg.rtol)   # AND，不是 OR
matched_ratio = 1.0 - (exceeds_count / total_elements)
exceeds_tol = matched_ratio < required_matched_ratio
```

**只有同时超绝对阈值和相对阈值才算坏点**，再要求坏点占比不超过 `1-ρ`。注意 `eps=1e-8` 而非我们的 `clamp_min(1e-12)`。

三种门禁跑同一份数据（KB 与 FIB 都取 fp16 的 1e-2，FIB 取 ρ=1.0）：

| 候选 | max_abs | OURS `≤0.5` | KB `allclose(1e-2,1e-2)` | FIB `AND, ρ=1.0` | 应当 |
|---|---|---|---|---|---|
| `correct_fp32acc` | 0.000e+00 | PASS | PASS | PASS | PASS |
| `correct_fp16acc` | 1.217e-02 | PASS | PASS | PASS | PASS |
| `scaled_0.97` | 1.005e-01 | **PASS** | fail | fail | fail |
| `scaled_0.999` | 3.350e-03 | PASS | PASS | PASS | PASS |

两点值得注意：

1. **两个外部门禁都拦下了 0.97，都没误杀合法 fp16 累加核。** `correct_fp16acc` 的 max_abs=1.217e-2 其实已经**超过** KB 的 atol=1e-2，是靠加性项里的 `+rtol·|e|` 才过的；FIB 则是靠 AND 形式（近零点 rel 爆但 abs 小，不计坏点）。**两者的容错来自不同机制，但都必需。**
2. `scaled_0.999`（0.1% 偏差）三家都放行，这是对的——该量级本就在 fp16 噪声内，不算洞。

---

## 四、我们已经做对的一件事 ✅（更正调研输入）

调研转述 Kevin-32B 附录 A.1 指出 KernelBench 的漏洞：参考实现先跑、候选后跑，候选可以「回收参考刚释放的显存」，于是一个只算部分输出的核也能过；对策是**改成候选先跑**。

**我们的 runner 已经是候选先跑**（`run_kernel.py:408-409`）：

```python
with torch.no_grad():
    actual = module.kernel(**candidate_inputs)   # 候选先
    expected = reference()                       # 参考后
```

参考此刻还没算出来，候选无从回收。本地路径 `verifier.py:261` 同样是 `execute()` 在前、`_generate_reference()` 在后，且参考在控制端用 numpy 算，根本不共享分配器。

**这一项不需要改。** 记录在此是为了避免照着外部清单做无用功——顺便说明「候选先跑」这个保护是隐式的、没有注释也没有测试守着，将来重排这两行不会有任何东西报警（属缺口 6：接口契约隐式）。

仍然缺的是另外两件事：调用之间没有 `empty_cache()` / `synchronize()`，以及没有「同输入双跑候选」。

---

## 五、外部共识与本项目现状对照

| 维度 | 外部主流 | 本项目 | 差距 |
|---|---|---|---|
| 判定形式 | 绝对 + 相对（加性或 AND） | **纯绝对** | ★ 核心缺陷 |
| 容差来源 | 按 precision 查表 | driver 手传标量 `5e-1` | ★ 且与 schema 默认 `1e-5` 差 5 万倍 |
| 输入组数 | KB 默认 1；robust-kbench ⚠️ 5；FIB ⚠️ 3 | 1（`verifier.py` 支持 list 套件但 driver 只传一组） | 中 |
| 执行顺序 | 候选先跑 | ✅ 已满足 | 无 |
| 调用间清理 | ⚠️ `empty_cache`+`synchronize` | 无 | 中 |
| NaN/Inf | ⚠️ FIB 显式判死 | 未显式处理 | 中 |
| 静态反作弊 | ⚠️ Kevin reward-0 规则（`try/except`、`pass`、`torch.nn`） | 正则 `static_audit`，5 种写法可绕（doc 03） | 大 |

### ⚠️ 调研转述、本人未一手核对的部分

以下来自调研 agent 的一手来源检索，我未逐个复核原文，**采信前需自查**：

- torchbench `env_check.py` 的 fp32 1e-4 / fp16 1e-3 / bf16 1e-2 与 `REQUIRE_HIGHER_BF16_TOLERANCE` 名单（KernelBench 注释引用了它，但具体数值我没读原文件）
- METR：`atol = 0.01` 纯绝对、剔除 45/250 任务、17 个无同步的 CUDA stream 解、`torch.empty` 复用参考显存"extracting the answer for free"、推荐 `triton.testing.do_bench`
- Sakana：初版继承 KB 的 1e-2 → 作弊核 50–120x 假加速；robust-kbench 重建后 rtol=atol=**1e-5**、`num_correct_trials=5`、`graceful_eval_cleanup`、同输入双跑；剔除污染任务后平均加速 **3.13x → 1.49x**
- Kevin-32B：`S = 0.3·1{correct} + (T_base/T_kernel)·1{correct}`；reward-0 规则；RL 中 hacked 核因 advantage normalization 被过度强化
- PyTorch：`cudnn.allow_tf32` 默认 True 而 `matmul.allow_tf32` 默认 False（**不对称**）；`allow_fp16_reduced_precision_reduction` 默认 True → fp16 GEMM 的中间归约可能就在 fp16 里做
- CyberTron：**未找到任何一手论文或仓库**，无法给数据

最后一条 PyTorch 的不对称如果成立，对本项目影响很直接：**我们的参考 `A @ B` 走 cuBLAS，它自己可能就是降精度归约的**，那么"候选与参考的差"里有一部分是参考的误差而非候选的。我尝试抓 PyTorch 官方文档核实，返回的是重定向页没拿到正文——**这一条必须在 H20 上直接 `print(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)` 核实**，它决定容差表能不能照抄 KernelBench 的 1e-2。

---

## 六、吸纳决策

### 采纳

1. **L2 门禁改成「绝对 + 相对」的 AND 形式 + matched_ratio**（照 FlashInfer `utils.py:113-118`）。
   理由：实测证明 AND 形式既能拦 0.97 又不误杀合法 fp16 累加核；`matched_ratio` 提供 ρ 旋钮，用于吸收参考自身降精度带来的少量离群点。
   同时把 `eps` 从 `clamp_min(1e-12)` 改成 `+1e-8`，与 FlashInfer 一致。
2. **容差按 precision 查表，不再由 driver 手传标量**。起点用 KernelBench 的 fp32 1e-4 / fp16 1e-2 / bf16 1e-2，但**数值必须在 H20 上用一个已知正确的核标定后再锁**（见上节末尾的参考精度问题）。
3. **NaN/Inf 直接判死**，不进容差比较。
4. **候选先跑这条隐式契约补上注释与测试**（缺口 6）——它现在是对的，但没有任何东西阻止将来被改错。

### 改造后采纳

5. **多 trial 多 seed**：`verifier.py` 已支持 `list[dict]` 套件，缺的是 driver 只传一组。改成默认 ≥3 组，而不是 robust-kbench 的 5 组——真机每组都要跑一遍 L2+L3，5 组把单任务时延推到不可接受；先上 3 组，看假阳性率再调。
6. **调用间 `empty_cache()` + `synchronize()`**：只在 L2 的多次 trial 之间做，不在 L3 计时循环里做（`empty_cache` 会显著抬高首次分配开销，污染计时）。

### 暂不采纳

7. **子进程 spawn 隔离**（FlashInfer `isolated_runner.py`）：我们的候选已经在远端独立进程里跑（SSH + `run_kernel.py`），隔离级别已足够；再套一层 spawn 只增加时延。
8. **robust-kbench 的 rtol=atol=1e-5**：那是为 KernelBench 的 fp32 任务设的。我们的 fp16 路径上实测合法核的 max_abs 就有 1.2e-2，照抄会把所有 fp16 任务判死。
9. **LLM 静态验证器 + clang-tidy + NCU 三层**（robust-kbench）：与本项目的 `static_audit` 定位重叠，且引入 LLM 判定会让验证结果不可复现。反作弊这块走 AST 而不是再加一个 LLM（见 doc 03 修复方案）。

---

## 七、与缺口的关系

- **缺口 1（测量有效性无三态）**：本文的 `matched_ratio` 天然是个连续量，可以直接作为三态判据的输入之一——`ρ=1.0` 通过为 valid，`0.99≤ρ<1.0` 为 degraded，更低为 uninformative。与 doc 01 的 launch 占比三态是**正交的两个维度**，按 DSH「正交结果独立上报」应当分别上报而不是合成一个标志位。
- **缺口 6（核函数接口契约隐式）**：第四节的「候选先跑」是本缺口的一个具体实例——一个正确性攸关的顺序约束，既无注释也无测试。

## 八、2026-09-30 回访：第一节那条「纯绝对」并没有被换干净

第一节立的那条缺陷（L2 是纯绝对容差）在 2026-09 被 AND 形式替换掉了 —— 但替换只覆盖了
**三条比对路里的两条**。今天再读一遍（`operator_agent/verifier.py` 的 `_verify_single_set`）：

| 路 | 什么时候走 | 判据 | 这之前报的尺键 |
|---|---|---|---|
| 全量 | 后端声明 `FULL_OUTPUT_TENSOR`（本地 numpy / MockGPU） | `effective_atol` + AND + ρ | `atol` `rtol` `required_matched_ratio` |
| 真机 | 后端声明 `REMOTE_VERIFY`（SSH 上的 `run_kernel.py`） | 同一个 `tolerance` 模块 | 同上三键（`_REMOTE_L2_METRIC_KEYS` 逐键透传） |
| **采样窗口** | 两个能力都没声明、执行侧只回 `output_sample` | **`max_error <= 任务自带的 max_error`** | **一个都不报** |

两件事叠在一起才是要害：

1. **这把尺只会同松或更松**。`effective_atol` 取 `min(表, 任务值)`，所以 AND 那条的绝对项
   **≤** 采样这条的；采样这条还没有相对项、没有 ρ。第一节举的那个「整体缩小 14.93%」在
   这里换个形状又回来了。可重建的读数（`benchmarks/probe_l2_sampling_gate.py`，控制机零机时）：
   * 臂 4（常数偏移 3e-3 配大幅值参考，`max_error=5e-1`）：纯绝对 `atol=5.00e-01` → **PASS**；
     AND 那条 `atol=1.00e-04` → **FAIL**；
   * 臂 2（参考幅值扫描）：0.1 与 0.01 两档同样是「纯绝对 PASS / AND FAIL」。
2. **它一个尺键都不报，于是那把松尺在读数里不存在**。`measurement.single_ruler`（轮级）与
   `l2_ruler_reading` / `cross_round_ruler`（运行级）拿的是 `GATE_PARAM_KEYS`，采样那条缺席 ⇒
   读成 `RULER_UNREPORTED`（「这一轮没测尺」）。而「没测」与「测了、用的是另一把尺」在
   #87 那条守卫里是**两种不同的处置**（前者查部署，后者查口径），压成一个就是把缺陷读成前提。

**这一片（#80 第 1 片）只观测，不改判决**：采样那条现在如实发布自己真正用过的 `atol`，
并把同一条 AND 判据套在同一窗上的第二个读数发布成 `and_form_*`（判定式仍只有 `tolerance`
那一份 —— 这里只多问它一次，不抄第二遍）；逐组 message 与运行级聚合 message 各说一句
「两把尺一致 / ⚠️ 分歧」，聚合的 `metrics` 里是**组号名单** `absolute_only_sets` 与
`and_form_would_reject_sets`（没人报过就不写这个键，不写空表也不写 0）。18 例 +
`benchmarks/mut_specs/sampling_ruler_divergence.py` 11 行（行为 10/10 + 结构 1/1）。

**换尺那一步（第 2 片）留着没做**，理由不是难：把 `passed_abs` 换成 AND 只可能**收紧**，
但它会改 L2 通过率，而通过率的任何改动都要一次前后对照（#48/#77 那片地）。`S10` 那行变异
就是钉这件事的 —— 谁顺手换了尺，日志与用例必须当场响。

顺手记下两处自己撞出来的：
* 那个探针的 `branch_of` 判别键**第二次腐烂**：第二版是「`atol` 只有 AND 那条会算」，本片给
  采样那条新增了 `atol` ⇒ 表里每一行都印成「认不出哪条路」。改成认两条路各自独有的键，而
  聚合层只认 `absolute_only_sets`（逐组的 `and_form_atol` 不穿过聚合）。⇒ 登记 **I33**：
  拿「某个键只有一条路会报」当分类依据时，那个性质是**当前产物的观察结果**，不是契约。
* 变异 **S7 第一版逃逸**：用例写 `assertIn("纯绝对", message)`，而那条报警改成印死字 `0` 时
  这句话照样成立 —— 词在哪都会印，**数字**才是读数。⇒ I32 的又一种形状。


## 复现

```bash
python3 benchmarks/probe_l2_gate.py    # 第一节：5e-1 admits 14.93% 缩放
python3 benchmarks/probe_gates.py      # 第二、三节：三门禁对照（含 4096³ fp16 matmul，约 1 分钟）
```

两个脚本纯 numpy，不需要 GPU / torch / LLM / 远端；它们复现的是 runner 的判定配方，
不是 runner 本身，因此结论对「门禁形式」有效，对真机上 cuBLAS 参考的具体数值仍需
在 H20 上复核（见第五节末尾）。

外部源码（本文引用处均已读过原文）：

- [flashinfer-bench `bench/utils.py:96-123`](https://github.com/flashinfer-ai/flashinfer-bench/blob/main/flashinfer_bench/bench/utils.py)
- [KernelBench `src/kernelbench/eval.py:83-101, 398, 799-805`](https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/eval.py)
- [Kevin-32B arXiv:2507.11948](https://arxiv.org/abs/2507.11948)　[robust-kbench arXiv:2509.14279](https://arxiv.org/abs/2509.14279)　[FlashInfer-Bench arXiv:2601.00227](https://arxiv.org/abs/2601.00227)
- [METR: Measuring Automated Kernel Engineering](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/)　[Sakana 事后复盘](https://sakana.ai/ai-cuda-engineer-post-mortem/)　[续报 3.13x→1.49x](https://sakana.ai/ai-cuda-engineer-update/)
