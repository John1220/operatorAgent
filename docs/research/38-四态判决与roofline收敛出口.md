# 38 · 吸纳 D4：四态判决、按档带宽、以及 roofline 收敛出口

对应缺口 4（改善判据与 bound 类型解耦）。决策记录在 [doc 20 第三节 D4 行](20-吸纳决策与阶段总结.md)，
本机阈值标定的出处在 [doc 34 第四节](34-util快照去污染复核.md)。

落地时间：2026-09-26。提交：`feat(measurement)` + 本文档。

---

## 一、起点：一条被缓过两次的判据

循环里判断「这一轮有没有变好」的代码，一直到本提交前是一行无带宽的标量比较：

```python
if current_gflops > best_gflops:   # conductor.py, D4 之前
```

它把三件不同的事压成一件：

1. **真的更慢** —— 有证据说明本轮配方有害；
2. **完全打平** —— 没有任何证据，只是这次没测出更好；
3. **差 0.0001%** —— 比这台机器的读数精度还细。

而 `else` 分支的惩罚是**永久**的：`disabled_recipes.add(r.id)` 会把本轮新配方拉黑，
后续所有轮次都不再考虑它。doc 00 记下缺口 4 时用的措辞是「判据与 bound 类型解耦」，
指向的是另一半问题：一个 memory-bound 的核，用 GFLOPS 标量比大小本身就比错了轴。

D4 在 doc 20 里被**暂缓过两次**，理由不是不重要，而是判决带宽没有本机的数：外部
（SOL-ExecBench 一系）用的是 0.5% / 2%，那是锁频专用机上的值。照抄的后果被 doc 34
量化过 —— 0.5% 在 valid 档是真实噪声底的 25 倍（会把真的小改善判成「没变」），
在 uninformative 档又比实测散布范围 2.1% 还窄（一切都落 SAME/AMBIGUOUS）。
doc 34 补齐了本机三档的跨次散布，这次才有资格动它。

## 二、动手前先被数字纠正的两件事

### 2.1 「按 bound 选指标」这一半其实是空转的

缺口 4 的原表述容易读成：写一个 `if bound == "memory": 用带宽轴 else: 用 GFLOPS 轴`
的分支。真查下来这个分支解决不了任何问题 —— **同一个 task 的 shape 在整个优化循环里
不变**，两条轴的利用率来自同一次测量（`verifier.py:808-835`），换配方不会让「该看哪条轴」
这个事实翻转。加那个分支只是把同一个数字换个名字读。

真正有后果的是另一半：**判据该看离上界还有多远，而不是继续拿一个标量比大小**。
所以现在落地的是 `roofline_converged()`（绑定轴达成率 ≥ 90% 就停），而不是 bound 分支。
分支选指标反而是要防的东西 —— 两条轴接近时它会选错，因此 `binding_attainment()`
取两者的 **max**（roofline 的语义本就是「离上界多远由紧的那条轴决定」；而且
`roofline_bound` 这个字符串还有过 memory/compute 与 name/bound 两套拼法飘掉的历史，
见 doc 15 第五节，把它当分支键是在依赖一个不可靠的字符串）。

### 2.2 判决带宽不能拿 `paired_speedup_dispersion`

doc 34 第四节明确记了这个坑：配对散布在三个档分别是 valid 0.0016 / degraded 0.0114 /
uninformative 0.0248，**系统性比真实跨次散布大 3~8 倍**（它量的是单样本的散布，而决策
统计量是 median）。拿它当带宽等于把整个闭环静默关掉 —— 那正是缺口 1 最初的病。

所以带宽存表，锚在实测的**跨次**散布上，取 3 倍：

| 档位 | 实测跨次散布 (doc 34 第四节) | `NOISE_BAND` |
|---|---|---|
| valid | 0.02% | **0.10%** |
| degraded | 0.13% | **0.40%** |
| uninformative | 0.66%（宽到 2.1%） | **2.00%** |

⚠️ 0.30 那一档之外，`uninformative` 的 2% 其实**大于**它自己的实测散布上限 2.1%。
这不是 bug：那一档本来就不参与判决（`verdict()` 见到 `UNINFORMATIVE` 直接返回
`AMBIGUOUS`），存一个值只是为了 `NOISE_BAND` 对每个档位都有定义，别让运行时 KeyError
替设计说话。这条有测试钉住（`test_band_is_defined_for_every_validity_tier`）。

## 三、落地清单

`operator_agent/measurement.py`：

- `Verdict` 四态枚举 `FAST / SAME / SLOW / AMBIGUOUS`；
- `NOISE_BAND: {Validity → float}`；
- `verdict(current, previous, validity) → (Verdict, 人类可读理由)`；
- `binding_attainment(metrics)` / `roofline_converged(metrics)` —— 后者带可信性守卫：
  `measurement_can_gate is False` 或 `benchmark_contaminated` 时一律 False（不可信的
  测量没有资格宣布「已经到顶」）。

`operator_agent/conductor.py`：

- L3 改善追踪改走 `verdict()`：**只有 `SLOW` 才禁用配方**；`SAME` 与 `AMBIGUOUS` 只累计
  停滞轮次。日志相应拆成「性能退步」与「性能未改善」两句，不再共用「未改善」。
- 新增收敛出口：本轮 L3 未通过但绑定轴已达 roofline → 打印达成率并 `break`。
- 顺手把三处 16 行的 `_flush_snapshot(...)` 收进 `_snap()` 闭包（新出口要写快照，
  复制粘贴第三份之前先合并）。

### 为什么有两处「看起来该加但没加」的守卫

两次的理由相同：**它们永不生效**。

1. `SLOW` 分支里没有再查 `measurement_can_disable_recipe`。能走到这里的前提是
   `measurement_ok`，而它要求 `can_count_stagnation`（= `validity is VALID`），
   与 `can_disable_recipe` 是同一个条件。
2. 收敛出口里没有 `not l1_l2_failed`，也没有 `not l3_result.passed`。
   `Verifier.verify()` 在 L1/L2 任一失败时直接 `return`（`verifier.py:158-175`），
   所以「拿得到 L3 结果」就意味着编译与数值都已经过了；而再往前 `all_passed` 已经
   `break`，能走到这一行的 L3 必然没过。

这两个都是本项目反复记过的账（缺口 1 的「算了没人读」、doc 35 的「死规则守卫」）。
判据的正确性和守卫的可存活性是两件事，写永不生效的条件会让下一个读这段的人以为
存在某种保护。两处都留了注释说明为什么**不**写。

## 四、真机读数：判据第一次吃真数

`benchmarks/verify_measurement_validity.py` 现在除了三态判定，还把同一份真机 metrics
喂给 `binding_attainment / roofline_converged / verdict`（不需要 LLM，只走 SSH + 真卡）。
H20 GPU 2、2026-09-26、util=0%、7 个其他进程：

| case | 档位 | 绑定轴达成率 | converged | 三组 verdict（cur vs 持平 / 差3倍 / 好1.5倍） |
|---|---|---|---|---|
| layernorm 1024×768 | uninformative | 11.7%（带宽轴） | False | ambiguous ×3 |
| layernorm 16384×6144 | degraded | 66.8%（带宽轴） | False | same / faster / slower |
| matmul 4096³ fp16 | valid | **101.5%（GFLOPS 轴）** | **True** | same / faster / slower |

三条都值得单独说：

- **matmul 4096³ 的 101.5% 是真实读数，不是 bug**：实测 140.3 TFLOPS，峰值表给的
  H20 fp16 是 138.2 TFLOPS，同一次测量里 cuBLAS 基线是 133 TFLOPS（加速比 1.0566x）。
  含义很直接：这个核已经在 roofline 上，**换任何配方都不可能补出 1.2x 的差距**。
  旧循环在这种情况下会连着烧三轮 GPU + LLM token 才因停滞停下（或者根本停不下来），
  新循环第一轮就停，并把「为什么停」说清楚。这一条就是缺口 4 想要的全部东西。
- **layernorm 16384×6144 的 66.8% 判 False 是对的**：带宽轴还剩 1.5 倍空间，
  该继续换配方。收敛出口不能见谁都停，这条是它的反向保护。
- **layernorm 1024×768 三组全 ambiguous 也是对的**：这一档以前报的是「0.65x →
  未改善 → 禁用配方」，数字描述的是驱动派发开销。现在判决和收敛都不读它。

⚠️ **未修的风险**：达成率能超过 100%，说明分母（峰值表）本身有 ±5% 量级的误差，
而 `ROOFLINE_CONVERGED_PCT = 90` 是**除在这个分母上**的。也就是说「已到顶」这句话的
可信度上限就是峰值表的精度。这属于缺口 3（硬件知识散落且静态）的活，不在这里顺手改。
同一张表上另一处已知的薄点，而且它是**两张表互相矛盾**：一台 "nvidia h200" 在
`_PEAK_BY_NAME` 里命中 `"h200"`（按键长优先匹配），拿到的是一份照抄 H100 的峰值
（`verifier.py:40-41`，带宽 3350 GB/s）；而在 `hardware_caps._KNOWN_HARDWARE` 里
根本没有 `h200` 键（`hardware_caps.py:127-247`，尽管 272 行的注释以为有），
于是 `"h20" in "nvidia h200"` 命中 → 拿的是 H20 的 launch 地板与 3680 GB/s。
同一次测量里分子分母来自两台不同的机器。两条都记在缺口 3 名下（doc 20 §六第 2 条）。

## 五、变异测试：判据有单测 ≠ 循环读了它

定点变异 10 条（一次性脚本，跑完即删）：

| 变异 | 结果 |
|---|---|
| P1 带宽不参与判决（退回严格比较） | 红 ✅ |
| P2 `SAME` 也禁用配方（旧语义） | 红 ✅ |
| P3 `roofline_converged` 不看污染/不可信 | 红 ✅ |
| P4 `binding_attainment` 只看 GFLOPS 轴 | 红 ✅ |
| P5 `uninformative` 不再返回 AMBIGUOUS | 红 ✅ |
| P6 conductor 不再调用 `verdict` | 红 ✅ |
| P7 valid 档带宽被压成 0 | 红 ✅ |
| **P8 收敛出口整个删掉** | **第一版逃逸 ❌ → 补测试后红 ✅** |
| **P9 收敛出口不看 `l3_result is None`** | **第一版逃逸 ❌ → 补测试后红 ✅** |
| P10 收敛出口不写快照 | **逃逸 ❌（已知，见下）** |

P8 是最值得记的一条：`roofline_converged()` 与 `binding_attainment()` 在
`test_measurement_validity` 里各有单测，全绿；把 conductor 里那段出口删掉，
**没有任何断言变红**。这跟缺口 1 是
同一个失效方式（判据算出来了，下游没人读），只不过这次是我自己的新代码。补的是
`TestRooflineConvergedTerminatesSearch`，断言的对象是**轮次真的停在这里**
（`_rounds_run(out, 4) == 2`）而不是某个日志字符串。P9 同理：要有一轮 L2 失败、
从而根本没有 L3 的轮次，`l3_result is None` 这道守卫才是可判的。

**P10 的逃逸不是漏测，是一个真实的空转**：终止出口写的那次快照落在 save 之前的目录，
而 `execute()` 末尾一定会再补写一次进权威的 `saved_dir`（`conductor.py:743`），
resume 读的是后者。删掉出口那次不影响任何可观测结果。曾想过直接把它删掉，最终选择
保留（与其余终止出口动作一致，且它是「进程在 save 之前就没」时唯一的读数），
但把这件事写进 `_snap()` 的 docstring，避免下一个人以为它有读者。**不做无法判绿的设计。**

## 六、语义变更（会影响既有跑法的地方）

1. 打平不再禁用配方。原来 `[10, 10, 10]` 会连禁用带终止；现在只累计停滞。
2. 带宽内的小涨不再算改善：`+0.02%` 在 valid 档落 `SAME`，不改 `best_gflops`、
   不重置停滞计数。代价写明白：**每步都小于 0.1% 的缓慢爬升会被判停滞**。可接受的
   理由是 valid 档的跨次散布本身就是 0.02%，那种爬升在这个仪器上本来就不是证据。
3. 日志新增三种说法：「性能改善 (+x% > 带宽)」「性能退步 (−x% < −带宽, 禁用配方)」
   「绑定轴已达 roofline N% (>= 90%), 判定已收敛」。原有「性能未改善」保留但含义收窄
   到 `SAME`。任何按日志文本做统计的脚本要跟着改。
4. `metrics` 无新增键（三态那批键已有），四态判决本身不写进 metrics —— 它只在
   Conductor 的局部，理由见 §八。

## 七、测试账

- `tests/test_measurement_validity.py::TestFourStateVerdict`：7 条（四态切分、档位
  全覆盖、uninformative→AMBIGUOUS、按档的同一 Δ 得不同结论、达成率取紧轴、收敛需要
  可信测量、阈值方向性）。
- `tests/test_conductor_stagnation.py::TestRooflineConvergedTerminatesSearch`：5 条
  （GFLOPS 轴收敛、带宽轴收敛、未达阈值不停、uninformative 不停、无 L3 不停）。
  这一族的夹具有改动：`ScriptedGflopsGPU` 现在可选注入带宽与离散度，`_run` 支持
  `peak=` —— 预置 `Verifier._theoretical_peak`（那是它的延迟探测缓存），因为 MockGPU
  探测不到峰值（`hw_peak_gflops=0` → 利用率恒为 0），roofline 类用例只能从这里进。
- 全量：`python3 -m pytest tests/ -q` → **339 passed / 4 failed / 3 skipped**。
  4 条红是本分支既有的遗留红（`test_fixer.py` 修复循环计数 ×2、`test_p1.py`
  路由与失败记账 ×2），与 D4 无关，属「代码问题差不多就行」那一批。
- 真机驱动 `benchmarks/verify_measurement_validity.py` 扩了 §四那一段，不新增文件。

## 八、没做的事

- **四态判决没有落进 `metrics`/快照**。目前只有 Conductor 当场消费。若以后要离线分析
  「哪一轮被判成什么」，得把 `verdict` 与所用带宽一起写进 `metrics` —— 那需要先把
  `measurement_validity` 那批键的搬运约定理清（doc 33 抓过一次搬运丢字段）。
- **缺口 3（峰值表收敛到 `HardwareCases` 一处 + 用实测替代 datasheet）**：§四末尾那条
  风险的正解在这里，也是 D4 判据可信度的天花板。优先级因此往上提一格。
- **D5 噪声导向停止**：D4 处理的是「这一轮的读数算不算证据」，D5 处理的是「样本够不够
  支撑这个读数」。地基（每轮回报对的散布）已由 D3 就位，仍未做。
- **C6 三值判决 + 行为指纹**：与 D4 同源，四态已覆盖判决部分，行为指纹回归另开。
