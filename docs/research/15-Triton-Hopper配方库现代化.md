# 15 · Triton 3.8 / Hopper 配方库现代化

> 日期: 2026-09-25 · 分支: `research/agent-design-2026`
> 关联: [doc 00 缺口 3/4](00-架构基线与缺口核实.md) · [doc 14 搜索策略](14-搜索策略与硬件感知目标函数.md) · [doc 32 计时口径](32-吸纳Triton-do_bench计时口径.md)
> 证据标注: ✅ = 本人打开源码/官方文档一手核对 · ⚠️ = 二手转述或推断，未核实

---

## 一、为什么先动配方库

调研外部 agent 设计时得到的第一条硬结论不是「缺新配方」，而是**现有配方库里有几处会让配方静默失效的缺陷**。配方是 Planner 注入 Coder prompt 的唯一领域知识通道，它失效不会报错，只会让某一整类优化永久缺席，而 Coder 每轮都收到一堆用不上的建议。

本次动手前先做了一遍逐条核实，结果发现三类问题：

1. **悬空引用** —— `skills/coder_generate_triton.md:37` 点名推荐 `R-MATMUL-12`，配方库里没有这一条。也就是说 Coder prompt 每轮都出现一个标着「关键性能优化」的 ID，而 Planner 永远选不出它。
2. **死条件** —— `R-MATMUL-07` 的 `applies_when` 写的是 `registers_per_thread >= 255`，而 `HardwareCaps.registers_per_thread` 的默认值恰好就是 255，条件恒假。
3. **过期 API** —— `R-MATMUL-03` 教 Coder 用 `allow_tf32`，这个参数在 Triton 3.8 已废弃。

这三类的共同点是**没有任何运行期信号**：不报错、不告警、日志里看不出异常。所以修完之后必须留下结构性守卫，否则下一次改配方还会重演。

---

## 二、一手核实的 Triton 3.8 API 事实

全部在本地 `triton/` 检出（3.8.0）里逐条打开源码确认。

### 2.1 `tl.dot` 的精度控制换了名字

✅ `python/triton/language/core.py` —— `tl.dot(..., input_precision=...)`，取值 `"tf32" | "tf32x3" | "ieee"`；`allow_tf32` 仍在签名里但走 `DeprecationWarning`。

✅ **默认值是 `"tf32"`，不是 `"ieee"`**。这一条对本项目后果很直接：`Precision.FP32` 的任务在 Triton 后端上实际跑的是 TF32 张量核，而 L2 数值门是按 fp32 容差判的。doc 30 第六节记录的「同一核 L2 按 fp32 判精度、L3 按 TF32 给算力」的不一致，根子就在这里。

实测数据（H20，4096³ matmul，来自 doc 30）：

| 组合 | max_abs | 相对误差 | L2 门 |
|---|---|---|---|
| Triton 默认（tf32） | 1.22e-03 | 0.3163 | **FAIL** |
| `input_precision="ieee"` | 0.0 | 0.0 | PASS |

### 2.2 block pointer 已被删除

✅ `core.py:2434-2438` —— `tl.make_block_ptr` / `tl.advance` 现在直接：

```python
raise NotImplementedError("Block pointers have been removed in favor of the tensor descriptor API")
```

网上大量 Triton 教程仍以 block pointer 为主线。配方库里不能出现任何 block pointer 写法，否则 Coder 照抄即崩。

### 2.3 TMA 描述符的硬约束

✅ `core.py:2467-2478`、`_utils.py:63-74`（`validate_block_shape`）：

- `base` 必须 **16 字节对齐**
- 除最后一维外所有 stride × 元素大小必须是 16 字节倍数；**最后一维必须 contiguous**
- 只支持 **2–5 维**
- `block_shape` **每一维都必须是 2 的幂**，总 numel ≤ `TRITON_MAX_TENSOR_NUMEL`
- device 侧描述符必须先 `triton.set_allocator(...)`（✅ `triton/__init__.py:24` 顶层导出），漏了启动即失败
- `padding_option` 默认 `"zero"`

### 2.4 `supports_tma()` / `supports_pdl()` 不是公开 API

✅ 两者都是 **tutorial 里的本地辅助函数**，判据分别是 `capability[0] >= 9`：

- `tutorials/08-grouped-gemm.py:42` → `supports_tma()`
- PDL tutorial → `supports_pdl()`

生成的核不能 `import` 它们。正确做法是让 Planner 用 `requires_caps: has_tma=True` / `has_pdl=True` 把关，而不是在核里自己探测。

### 2.5 Hopper autoWS 的静默失效门槛

✅ `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp`：

| 条件 | 不满足时的行为 |
|---|---|
| `num_warps` **必须恰好 == 4**（`if (numWarps != 4) return;`） | **静默不做 warp spec**，不报错不警告 |
| 该 loop 的 `num_stages > 1` | 静默跳过（loop 不被收集） |
| 函数内**任何** `scf.if` 都不能带 else 分支 | **静默整体放弃**（源码注释写明是 FIXME） |
| 任务分区失败 | `signalPassFailure()` → 硬编译错误 |

「静默不做」是最坏的一种失败：代码照跑、结果照对、性能没变。只能靠真机计时对比确认生效。

✅ 另外 `third_party/nvidia/backend/compiler.py` 里 Hopper 走的是独立 pass `nvidia.passes.hopper.add_hopper_warpspec`（分支条件 `capability // 10 in [8, 9]`），Blackwell 才走通用 `add_warp_specialize`。该 pass 于 2025-06-23（PR #7136）打通，即 **Triton v3.4.0（2025-07-30）起可用** —— 所以 tutorial 09 那句 "FIXME: This only works on Blackwell right now" **是过时的**。

⚠️ 冲突点：`R-MATMUL-02` 的 autotune 配置表里 12 组有 8 组是 `num_warps=8`，与 autoWS 的 `num_warps==4` 硬门槛直接互斥。两条配方同时选中时，autoWS 会静默失效。这一条已写进 R-MATMUL-14 的正文，但**配置表本身还没改**（见第六节待办）。

⚠️ 收益期望：可查到的数字全部来自别的卡 —— PyTorch 官方博客的 1.5–2× 是 **B200 上的 FlashAttention fwd**（且基于 Helion autotuning + ptxas 高级配置）；Tawa 论文（arXiv:2510.14719）在 **H100** 上是 GEMM ≤1.1× cuBLAS、attention 1.2× over Triton。**H20 上没有任何公开数据**，而 warp specialization 解决的是「计算与搬运重叠」，H20 算力已非瓶颈，收益应低于 H100 —— 这是推断。

### 2.6 persistent kernel 的占用率公式

✅ tutorial `02-fused-softmax.py`：

```python
kernel._init_handles()          # 必须先 init, 否则 n_regs / metadata 还没填
n_regs    = kernel.n_regs
size_smem = kernel.metadata.shared
occupancy = min(NUM_REGS // (n_regs * WARP_SIZE * num_warps),
                SIZE_SMEM  // size_smem)
num_programs = min(NUM_SM * occupancy, n_rows)
```

⚠️ `_init_handles()` / `n_regs` / `metadata.shared` 都是**运行时私有 API**（下划线开头），不是稳定契约。配方里已要求包 try/except 并给保守回退。

### 2.7 PDL 的三个动作

✅ `launch_pdl=True` 要在**装饰器和启动点两处**都写；核内 `tl.extra.cuda.gdc_wait()`（读上游数据前）与 `gdc_launch_dependents()`（本核写完后）。

⚠️ `gdc_wait()` 位置错是**正确性问题**不是性能问题：PDL 让后继核提前跑起来，不在读输入前 wait 就会读到未写完的 buffer。这类 bug 在小 shape 上常常碰巧不出现，换大 shape 才炸。

⚠️ **与 L3 测量的冲突**（本次新识别）：runner 的计时循环背靠背重复调用同一个核，开了 PDL 后相邻两次重复会互相重叠，测出的是吞吐而不是单次延迟，数字会明显偏快，而生产里这个核只被调用一次。若候选核使用 `launch_pdl=True`，其 L3 数字不可与未用 PDL 的基线直接比较。已写进 R-UNIV-05 正文。

### 2.8 `tl.cache` 不存在

✅ 全量扫 `triton/python/triton/language/*.py`，无任何 `cache` 相关的顶层导出。真实 API 是 `tl.load/tl.store` 的关键字参数（✅ `core.py:2333,1177`）：

- `cache_modifier` ∈ `{"", ".ca", ".cg", ".cv"}`
- `eviction_policy` ∈ `{"", "evict_first", "evict_last"}`

但 `triton_opt_rules.generate_optimization_hints` 一直在给 SearchAgent 输出「使用 `tl.cache` 优化数据缓存」。这条提示被 Coder 采纳就是 NameError。

---

## 三、顺带挖出来的两个更严重的缺陷

配方核实过程中撞见了两个不属于配方库、但影响面更大的问题。

### 3.1 `bound_type` 有两套拼法在流通，bound 感知从未生效

这是本次最有价值的发现。

✅ `verifier.py:703-720` 产出的是**裸词**：`"memory"` / `"compute"` / `"unknown"`。
✅ `optimization_planner.py:467,470` 按裸词比较 —— **与产出端一致，正确**。
✅ `triton_opt_rules.py:68,71,185,192` 按 `"compute_bound"` / `"memory_bound"` 比较 —— **永远不匹配**。

后果：`recommend_tiling` 和 `generate_optimization_hints` 的 compute / memory 两个分支是**死代码**，一路落到 `else`。也就是说 SearchAgent 的规则回退路径从上线起就没做过 bound 感知：

- H20 上明明测出 memory-bound（ridge ≈ 37.5，几乎所有算子都是），给出的建议是「mixed-bound: 均衡调整 tiling」，而不是「增大 BLOCK_K 提升数据复用率」
- `agents.py:597` 是唯一调用点，传进来的正是 verifier 的裸词

修法：在消费端加 `normalize_bound_type()`，两套拼法都归一到 `compute|memory|unknown|mixed`。**没有改产出端** —— 那个字符串已经进了 metrics / 日志 / kernel_db 快照，改它会连带改动下游既有断言。

这属于 doc 00 缺口 4（改善判据与 bound 类型解耦）的一个具体表现，也解释了为什么缺口 3（硬件知识散落）值得优先处理：**同一个语义值在三个模块里有两种写法，没有任何一处是权威定义**。

### 3.2 hints 里有两条会直接坑掉一轮的建议

- 「使用 float16 累加器提升吞吐量」—— fp16 累加会丢失大 K 求和的高位，是**精度 bug 不是优化**，且与配方库 R-UNIV-03（累加器始终 fp32）直接冲突。已改成「输入 cast 到 fp16/bf16 触发 tensor core，累加器保持 fp32」。
- 「使用 `tl.cache` 优化数据缓存」—— API 不存在（见 2.8）。已改成真实的 `eviction_policy` / `cache_modifier` 写法。

这两条都只在 bound 分支里，而 bound 分支因为 3.1 从来没执行过 —— **所以它们一直没被发现**。修好 3.1 之后它们才会真正进入 prompt，必须先一起修掉。

---

## 四、落地内容

### 4.1 配方库（`operator_agent/skills/optimization_recipes.md`）

| 配方 | 动作 | 依据 |
|---|---|---|
| `R-MATMUL-03` | 重写：`allow_tf32` → `input_precision`；补「默认就是 tf32」警告与 H20 L2 实测表 | 2.1 ✅ |
| `R-MATMUL-07` | 死条件 `registers_per_thread >= 255` → `compute_bound AND gflops_util < 50%` | 第一节 |
| `R-MATMUL-12` | **新增** Divisibility-Based Mask Elimination，补上 `coder_generate_triton.md` 的悬空引用 | 第一节 |
| `R-MATMUL-13` | **新增** TMA Tensor Descriptor，含全部硬约束与源码行号；block pointer 已删除的警告 | 2.2/2.3/2.4 ✅ |
| `R-MATMUL-14` | **新增** Hopper autoWS，含静默失效门槛表、与 R-MATMUL-02 的 `num_warps` 冲突 | 2.5 ✅ |
| `R-UNIV-04` | **新增** persistent kernel + 按占用率定 grid | 2.6 ✅ |
| `R-UNIV-05` | **新增** PDL，含「L3 量不出来甚至会带偏测量」的警告 | 2.7 ✅ |

### 4.2 硬件能力表（`operator_agent/hardware_caps.py`）

新增 5 个字段，默认值一律保守（False / 65536）：

```python
registers_per_sm: int = 65536          # persistent kernel 算占用率要用
has_tma: bool = False                  # cc >= 9.0
has_warp_specialization: bool = False  # cc >= 9.0
has_pdl: bool = False                  # cc >= 9.0
has_fp8_tensor_core: bool = False      # cc >= 8.9
```

H20 / H100 条目按 Hopper cc9.0 置 True，A100 / V100 / 3090 保持 False。

**这里差点重演死条件**：R-MATMUL-13/14 先写了 `requires_caps: has_tma=True`，而字段还不存在 —— `getattr(caps, "has_tma", False)` 恒为 False，配方在任何硬件上都选不出来。与 R-MATMUL-07 是同一类 bug，只是方向相反（一个是恒假选不出，一个是字段缺失选不出）。

未知硬件的推断路径改成按主版本号判定，并显式取保守值：

```python
caps.has_tma = caps.has_warp_specialization = caps.has_pdl = major >= 9
caps.has_fp8_tensor_core = major >= 9
```

误报 True 会把配方塞给不具备该能力的硬件（核直接编译/启动失败），误报 False 只是少一次优化机会，所以推断一律偏保守。Ada(8.9)/Blackwell(10.x) 的 FP8 不在推断里猜，要靠 `_KNOWN_HARDWARE` 显式建条目。

顺带修掉旧的 `cc.startswith("8.")` 判定 —— 它认不出 cc `"10.0"`（Blackwell）。

### 4.3 Planner 崩溃防护（`operator_agent/optimization_planner.py`）

✅ 实测过的崩溃：配方写 `requires_caps: compute_capability >= 90`，而 `compute_capability` 是字符串 `"9.0"`，`"9.0" >= 90` 抛 `TypeError`，**打断整轮 `plan()`**。

两处修复：

1. 数值比较前判类型，非数值一律判不满足（不抛）。
2. 新增 `_caps_failure_note()`，把「硬件不够」和「配方写错」在拒绝原因里区分开：

```
compute_capability >= 90   [compute_capability='9.0' 非数值, 不能做数值比较]
no_such_field >= 1         [HardwareCaps 无 no_such_field 字段, 条件写错]
```

不写这行说明的话，配方被拒的表现和「硬件确实不支持」完全一样，没人会去修那行配方。

另外把裸名 `has_xxx` 的判定从**硬编码枚举表**改成前缀匹配 `re.fullmatch(r"has_\w+", req)`。旧枚举表漏掉任何新字段，该配方就会落到函数末尾的「无法解析则默认通过」，于是被塞给不具备该能力的硬件 —— 与 4.2 那个 bug 是同一方向，但后果更糟（不是选不出，是乱选）。

### 4.4 结构性守卫（`tests/test_recipe_caps_gate.py`，26 个用例）

修完缺陷必须留下守卫，否则下一次改配方还会重演。这个文件守的是**配方库与能力表之间的一致性**，五类：

1. 配方引用的每个能力字段都必须是 `HardwareCaps` 上的真实字段（防死条件）
2. 任何已声明条件 × 任何已知硬件，`_check_caps_requirement` 都不许抛异常（防崩溃）
3. 仓库里出现的每个 `R-XXX-NN` 都必须在配方库里存在（防悬空引用）
4. `verifier` 实际产出的 `bound_type` 取值集合，必须被 `triton_opt_rules` 认识（防拼法漂移）
5. 配方文件的 7 个 `## ` 分节标题、每条配方的 5 个必填字段（防编辑时吃掉标题 —— 本次真实发生过一次）

第 4 条的取值集合是**用 AST 从 `verifier._roofline_analysis` 源码里现取的**，不写死在测试里。将来改了产出端，测试会跟着变，从而逼着消费端一起改。

#### 变异测试

守卫本身也要验证能不能失败。做了 10 个变异，逐个确认被捕获：

| # | 变异 | 捕获它的用例 |
|---|---|---|
| A | `recommend_tiling` 改回 `== "compute_bound"` | `test_opt_rules_recognizes_every_emitted_value` + `..._mixed_branch_is_reachable_only_for_mixed` |
| B | 删掉 `has_tma` 字段 | 收集期即报错（`_KNOWN_HARDWARE["h20"]` 构造失败） |
| C | `R-MATMUL-13` 改名 → 悬空引用 | `test_no_dangling_recipe_id_anywhere_in_repo` + `..._selected_on_h20...` |
| D | 吃掉 `## Softmax 优化配方` 标题 | `test_all_top_level_sections_present_and_ordered` |
| E | `requires_caps` 指向不存在的字段 | `test_every_referenced_attr_is_a_real_field` |
| F | planner 去掉数值类型守卫 | `test_string_valued_numeric_comparison_is_refused_not_fatal` |
| G | hints 恢复 `tl.cache` | `test_hints_never_recommend_a_nonexistent_triton_api` |
| H | `generate_optimization_hints` 改回死拼法 | `test_hints_are_bound_specific` |
| I | hints 恢复「fp16 累加器」 | `test_hints_never_recommend_a_nonexistent_triton_api` |
| J | planner 裸名判定改回硬编码枚举表 | `test_bare_has_prefix_is_not_silently_passed` |

A 和 H 是**第一版测试漏掉的**，两处都因为同一个原因：断言只检查了「文本里出现了 bound 这个词」，而兜底分支恰好会把传入的 bound 插进 reason 字符串里，于是走错分支在字面上看不出来。改成断言**策略关键词**（compute → `BLOCK_M/BLOCK_N`，memory → `BLOCK_K`）并额外断言兜底句不出现，才真正锁住。

这条教训值得记下来：**兜底分支如果会回显输入，就不能用「输出里有没有输入」当断言。**

---

## 五、遗留缺陷（本次未修，已记录）

1. **H200 会被误认成 H20。** `detect_hardware_caps` 按「键名长者优先」匹配 `_KNOWN_HARDWARE`，但表里没有 `h200` 条目，于是 `"nvidia h200"` 命中 `"h20"`。`sm_count` 会被 `hw_info` 覆盖成 132，但带宽仍用 H20 的 3680 GB/s（H200 实际 4800），roofline 会算错。当前环境用不到 H200，未修。
2. **`R-MATMUL-02` 的 autotune 配置表与 autoWS 冲突未解。** 12 组 config 里 8 组是 `num_warps=8`，会让 R-MATMUL-14 静默失效。`benchmarks/triton_autotune_matmul.py` 是同一份表的拷贝，要同步改。
3. **5 个测试是长期红的，且早于本次所有工作**（在 `a5ad7e0` 上同样失败，已用 worktree 核实）：
   - `test_fixer.py::TestConductorRepairLoop::test_budget_exhausted_stops_with_failure`（LLM 调用数 9 ≠ 5）
   - `test_fixer.py::TestConductorRepairLoop::test_fixer_exception_breaks_gracefully`
   - `test_p1.py::TestConductorFailureRecording::test_budget_exhausted_recorded_as_unfixed`（`records` 为空）
   - `test_p1.py::TestRouter::test_low_complexity_translation`（路由到 `compiler_backend` 而非 `code_translation`）
   - `test_p0_3.py::TestSameDeviceRemoteBenchmark::test_remote_uses_returned_same_device_baseline`（`baseline_type` 拿到 `pytorch_eager` 而非 `torch_compile`）

   前三个像是同一根因（修复循环的轮次/记账），后两个各自独立。按用户「代码问题差不多就行，后续有时间再细致优化」的指示暂缓，但**不能把它们误读成本次改动引入的回归** —— 本次改动后全量测试 244 passed / 5 failed，改动前 217 passed / 6 failed（第 6 个是 `test_anticheat_bypass`，本次一并修好）。

---

## 六、下一步

配方库这一批到此为止。接下来按用户指示把重心放在外部技术吸纳上，两份调研已到手：

- **harness 工程**（agent 设计侧）：处方式错误消息 + 外部归因、热/冷两层上下文禁止递归摘要、跨任务复发才升级为配方、verifier 隔离、三值判决 + 行为指纹回归。证据强度最高的一条是 SPARC —— 它的四个骨干模型是 Llama-3-8B / Qwen2.5-7B / DeepSeek-R1-Distill-7B / QwQ-32B，**正是本项目自建 Qwen 27B 的档位**。
- **性能测量方法论**（算法开发侧）：离散度门（零额外 GPU 时间）、每样本 SM 频率 + 丢弃退避、交错 A/B、NVBench 四态判决替换点估计门限、噪声导向停止。其中「离散度门」成本为零（runner 已有 30 个样本），应最先做。

两者的吸纳决策记录进 doc 20（`20-吸纳决策.md`，待写）。

---

## 附：本次未采信的外部说法

- ⚠️ warp specialization 的 1.5–2× 收益 —— 来自 B200 + Helion，不可外推到 H20
- ⚠️ TMA 的性能期望 —— 官方 tutorial 只把 "Triton + TMA" 与 "Triton" 并列展示，没给可引用的倍数；配方里写的是「按 H20 上比手工指针算术更可能拿到带宽采纳，期望值需真机标定后再写死」
- ⚠️ persistent kernel 的收益倍数 —— tutorial 02 的数字是行归约类算子的，对 matmul 这类每 tile 工作量已经很大的算子不成立
