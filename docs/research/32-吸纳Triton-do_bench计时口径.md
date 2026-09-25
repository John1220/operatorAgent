# 32 - 吸纳 Triton do_bench 计时口径：一次改变了结论的方法论升级

> 外部技术吸纳记录。来源是**仓库内已 checkout 的 Triton 3.8.0 源码**
> （`triton/`，与 H20 上部署的版本完全一致），不是二手博客。
> 这次吸纳推翻了 [`31`](31-测量有效性三态真机验证.md) 里两处刚写下的结论，
> 更正写在本文第四节与 [`31`](31-测量有效性三态真机验证.md) 的更正块里。
>
> 状态：✅ 已实现并在 H20 上标定（2026-09-25）

---

## 一、吸纳了什么

`remote_scripts/run_kernel.py` 的计时循环原来是「固定次数 + 不冲刷 L2」。
Triton 官方的 `do_bench` 两件事都不这么做，✅ 一手读源码
（`triton/python/triton/testing.py:267-337`）：

| | 我们原来 | Triton `do_bench` | 依据 |
|---|---|---|---|
| 重复次数 | 固定 `repeat`（默认 10/30 **次**） | `warmup=25, rep=100` 的单位是**毫秒**，`n_repeat = max(1, int(rep / estimate_ms))` | `testing.py:273-275, 313-314` |
| L2 状态 | 不冲刷，数据常驻 L2 | 每次计时前 `clear_cache(cache)`，一个 **256 MB** 的 buffer `zero_()` | `testing.py:299,329` + `third_party/nvidia/backend/driver.py:398-408` |
| 冲刷位置 | — | 在 `start_event.record()` **之前**，同流有序，所以不进事件窗口 | `testing.py:329-333` |
| 聚合 | median | 默认 `mean`，可选 min/max/median/quantiles | `testing.py:284-287` |

两条都采纳了，第三条（聚合量）保留 median 不动 —— median 抗离群，在共享机器上比
mean 更合适，这是我们比 `do_bench` 默认值更有理由的地方。

**为什么 L2 冲刷对我们特别要紧**：H20 的 L2 是 **60 MiB**，比 A100 的 40 MiB 大
一半。layernorm 1024×768 读写共 3.15 MB —— 整份数据第一轮之后就常驻 L2，此后每次
「测量」量的都是 L2 命中。而 `verifier` 拿这个时间去和 **3.68 TB/s 的 HBM 峰值**比，
算出的「带宽利用率」因此系统性偏高。matmul 4096³ 的 100 MB 流量超出 L2 只有部分
命中，偏差方向相同但幅度不同 —— **候选与基线被偏袒的程度不一样，连加速比本身都是脏的。**

---

## 二、真机标定：冲刷改变了什么

✅ `benchmarks/calibrate_l2_flush_effect.py`，H20 GPU 2（测量前 util=0%），
torch 2.14.0+cu130 / triton 3.8.0。同一个核，冲刷与不冲刷各测一次，CUDA event
计时、每轮 synchronize、取 median —— 与 runner 同口径。

先量冲刷本身的代价：256 MB `zero_()` median **74.21 µs**，折算 3.62 TB/s
（≈ H20 的 3.68 TB/s 实测带宽，说明这一步就是纯带宽操作，没有额外损耗）。

| 被测对象 | 不冲刷 median | 冲刷 median | 差 |
|---|---|---|---|
| **空 Triton 核**（只有 launch，零计算） | **20.03 µs** | **4.64 µs** | −15.4 µs |
| layernorm 1024×768 fp16 | 21.92 µs | 7.39 µs | −14.5 µs |
| layernorm 1024×768 的 PyTorch Eager 基线 | 19.62 µs | 9.95 µs | −9.7 µs |
| matmul 4096³ fp16 | 993.04 µs | 979.92 µs | −13.1 µs（−1.3%） |

### 这张表最重要的两行

**第一行推翻了「launch 地板 = 20.24 µs」这个说法。** 同一个空核，GPU 空闲时测得
20.03 µs，GPU 被冲刷填满时测得 4.64 µs。所以那 20 µs 不是「launch 的固有开销」，
而是**空闲 GPU 上 CUDA event 窗口吃进的 launch 延迟** —— `start.record()` 之后 GPU
无事可做，要等命令到达，这段等待被计进了窗口。`do_bench` 的冲刷把 GPU 填满，
下一条 kernel 的命令早已入队，等待被藏在窗口之外，于是只剩 4.64 µs 的真实派发+执行。

[`01`](01-launch开销与shape充分性实测.md) 第一节把这个数称为「launch 地板」并据此
建立了整套 shape 充分性论证。**数值本身没错**（复测 20.03 vs 标定 20.24，差 1%），
错的是对它的解释，以及「它是恒定的、可以从任何测量里减掉」这个隐含假设。

**第四行说明大 shape 不受影响。** matmul 4096³ 只差 1.3%，因为 993 µs 里那 13 µs
的 launch 延迟本来就微不足道。所以这次改动**不影响项目此前唯一被判定为可信的那个
数字**（[`00`](00-架构基线与缺口核实.md) 的 matmul 0.91x）。

---

## 三、结论翻转：layernorm 从「比 Eager 慢 11%」变成「快 35%」

用上表第一组数字直接算加速比：

| 口径 | 候选 | Eager 基线 | 加速比 |
|---|---|---|---|
| 不冲刷（改动前） | 21.92 µs | 19.62 µs | **0.895x** |
| 冲刷（改动后） | 7.39 µs | 9.95 µs | **1.346x** |

**同一个核、同一台机器、同一天，结论从「比 PyTorch 慢」翻转成「比 PyTorch 快 35%」。**

机制很清楚：不冲刷时两边都被 ~20 µs 的 launch 延迟垫高，
`19.62/21.92` 里分子分母各含一个几乎相同的大常数，比值被压向 1.0，
真正的核质量差异（7.39 vs 9.95 µs，候选快 26%）被淹没。
这正是 [`01`](01-launch开销与shape充分性实测.md) 说的「launch 地板把加速比压向
1.0」，但当时**只推断出「数字无意义」，没有推断出「数字还会反向」** ——
垫高一个常数不只压缩比值，它会**保留符号地压缩**，于是候选比基线多出的那 2.3 µs
真实差距，被放大成「慢 11%」的结论。

端到端复核（✅ `benchmarks/verify_measurement_validity.py`，走完整 Verifier 路径）
与手算一致：layernorm 1024×768 报 `speedup=1.3425`。

**这条对搜索闭环的含义比数字本身更重要**：改动前 Conductor 看到 0.895x，会判
「这个核不行」并去禁用本轮配方；而配方其实是有效的。这和
[`00`](00-架构基线与缺口核实.md) 缺口 1 记录的污染误判是**同一个失效模式的另一个
入口** —— 不是噪声，是系统性偏差，而且方向固定（总是把小 shape 上的好核判成差核）。

---

## 四、被迫更正：`launch_dominance` 的口径

[`31`](31-测量有效性三态真机验证.md) 里 `launch_dominance = floor / candidate_median`。
加上冲刷之后，这个式子对 layernorm 1024×768 算出 **2.81** —— 一个「占比」大于 1，
物理上不可能。

原因是分子分母来自两个口径：分子 20.24 µs 是**不冲刷**标定的 launch 延迟，
分母 7.39 µs 是**冲刷后**的净执行时间。混用。

改成

```
launch_dominance = floor / (floor + net)
```

即「用户真正付出的端到端时间（launch 开销 + 净执行）里，launch 占多少」。
这个式子恒 < 1，且恰好就是 [`01`](01-launch开销与shape充分性实测.md) **第四节**
那张表用的口径 —— 而 [`31`](31-测量有效性三态真机验证.md) 第 3.1 节当时判定
第四节与第五节「口径不一致」并选了第五节。**选错了**：在冲刷口径下，第四节的
`floor/(floor+下界)` 才是对的，第五节的 `floor/median` 只在「median 本身含 launch」
时才成立，而冲刷恰好把那一段拿掉了。

用真机净时间复算，三档结论与 [`31`](31-测量有效性三态真机验证.md) 完全一致：

| 用例 | 净时间 | 旧式 floor/net | 新式 floor/(floor+net) | 档位 |
|---|---|---|---|---|
| layernorm 1024×768 | 7.39 µs | 2.81（**非法**） | **0.7349** | uninformative |
| layernorm 16384×6144 | 163.6 µs | 0.1237 | **0.1101** | degraded |
| matmul 4096³ | 979.9 µs | 0.0207 | **0.0203** | valid |

注意中间两行：**大 shape 上新旧两式几乎相同**（0.1237 vs 0.1101、0.0207 vs 0.0203），
因为 net ≫ floor 时 `floor/(floor+net) ≈ floor/net`。两者只在 net 与 floor 同量级时
分道扬镳 —— 而那正好是唯一需要判定的区间。这也解释了为什么这个 bug 在
[`31`](31-测量有效性三态真机验证.md) 的真机验证里没被发现：当时 runner 还没冲刷，
两个口径意外地是同一个。

### 顺带删掉一个指标

`net_of_floor_speedup`（扣除地板后的加速比，[`31`](31-测量有效性三态真机验证.md)
第 3.2 节）**整个删除**。它建立在「median 含 launch 开销」这个前提上，而冲刷后
median 已经是净值，再减一次是重复扣除。它在地板支配区还会产出低于物理下界的
噪声（Eager 基线不冲刷 19.62 µs，减 20.24 µs 地板是负数）。
`tests/test_measurement_validity.py::TestNoNetOfFloor` 钉住这个删除，防止被加回来。

---

## 五、成本

冲刷不是免费的：每轮多 74.21 µs 的 GPU 时间，且次数由时间预算决定。

| 用例 | 净时间 | `n_repeat`（50 ms 预算） | 冲刷总开销 | 单次 L3 五组配置合计 |
|---|---|---|---|---|
| layernorm 1024×768 | 7.39 µs | 5000（**触及 MAX_REPEATS 上限**） | 371 ms | ~1.9 s |
| layernorm 16384×6144 | 163.6 µs | 305 | 22.6 ms | ~0.2 s |
| matmul 4096³ | 979.9 µs | 51 | 3.8 ms | ~0.5 s |

小 shape 是成本大头：核越快，为凑满 50 ms 预算要跑的次数越多，而每次都要付
74 µs 的冲刷。`MAX_REPEATS = 5000` 就是为此设的上限。
预算取 25/50 ms 而不是 `do_bench` 的 25/100 ms，理由同
[`31`](31-测量有效性三态真机验证.md)：一次 L3 要测五组，且这台机器共享、
每次评测都在 LLM 循环的关键路径上。

**这笔账划算**：1.9 秒的额外 GPU 时间，换来的是第三节那个方向性错误的消除。
改动前 Conductor 会基于 0.895x 去禁用一个实际有效的配方，而 `disabled_recipes`
在后续轮次不会恢复。

---

## 六、仍未做

1. **`hardware_caps.launch_floor_us` 的语义要改名。** 它现在存的是「空闲 GPU 上
   事件窗口吃进的 launch 延迟」（20.24 µs），而不是「launch 的固有开销」
   （冲刷后是 4.64 µs）。字段名与 docstring 已在本次改正，但**两个量都该有位置**：
   前者用于 `launch_dominance`，后者才是核的不可压缩下界。目前只存了前者。
2. **冲刷与 `@triton.autotune` 的交互。** autotune 的首次搜索发生在第一次调用内部，
   而 `_resolve_counts` 的估时也在那之后 —— 若 warmup 不足，搜索时间会被算进 L3
   成绩。本次未测。
3. **基线是否也该按时间预算采样。** 现在候选与四条基线都走 `_bench_callable`，
   口径一致。但 `run_benchmark_with_baselines` 里 fp16/tf32 基线各自还有一次
   warmup 调用在 `_bench_callable` 之外，未纳入预算。影响很小，未改。

---

## 七、复现

```bash
# 标定（在远端 GPU 机器上跑；@triton.jit 需要真实源文件, 不能用 stdin 管道）
scp benchmarks/calibrate_l2_flush_effect.py <别名>:/tmp/ && \
  ssh <别名> 'cd ~/operator_agent && CUDA_VISIBLE_DEVICES=<空闲卡> \
    ~/.local/bin/uv run python /tmp/calib_flush.py'

# 端到端（在控制端跑，会触发 deploy()）
python3 benchmarks/verify_measurement_validity.py
```
