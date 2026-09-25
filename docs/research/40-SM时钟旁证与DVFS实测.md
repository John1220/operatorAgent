# 40 — SM 时钟旁证与 DVFS 实测：一个用三道控制撑起来的零结果

对应吸纳决策 [`20`](20-吸纳决策与阶段总结.md) 第三节 **D2**（每样本 SM 频率 +
丢弃退避，源自 ✅ NVBench `measure_cold.cu`）。
落地代码：`remote_scripts/run_kernel.py:read_gpu_state()`、`operator_agent/verifier.py`
的 metrics 搬运表。判别实验：`benchmarks/probe_clock_dvfs.py`（远端自包含）。

结论先放：

| 项 | 结果 |
|---|---|
| 观测通路 | ✅ 已落地并真机复现两次（合并查询前后各一次），每次 L3 只多付 **1 次** nvidia-smi 调用 |
| 本机 SM 时钟随负载变化 | ❌ **不动**：空载 1980 MHz、持续满载 40 s 仍 1980 MHz，`pstate` 恒 P0，throttle 位图 0/34 非零 |
| 判定（阈值 / 归一化） | ❌ **都不加**。加了就是一条永不触发的死判据 |
| 改判条件 | 已写进探针末尾，换平台时照那三条复核 |

---

## 一、要判的问题

这台卡 8 租户共享。D2 的原始担忧是：邻居的负载会通过功耗/温度预算把**我们的** SM
频率压下来（DVFS）。若真如此，「这一轮比上一轮慢」描述的就可能是「这一轮跑在
1.4 GHz、上一轮跑在 1.8 GHz」，而不是核的质量 —— 与缺口 1 处理过的两种情形同构
（launch 地板支配读数、邻居抢时间片）。

在写判据之前先要一个数：**本机上频率到底变不变**。若恒定，任何「时钟低于上限就
降级」的阈值都是一条写出来就不会响的警报 —— 这正是本项目已经栽过几次的死条件。

所以 D2 分成两步走，这一步只做观测：runner 在计时窗口前后各读一次
`clocks.sm` / `clocks.mem` / `clocks.max.sm` / `clocks_throttle_reasons.active`，
搬到 `metrics` 里存着，不做任何判定。

## 二、三道控制：为什么「时钟没变」必须自己证明自己没瞎

`docs/research/34/35/36/39` 已经四次犯同一类错误 —— **拿看不见被测量级的量具报零
结果**。这一节差点犯第五次，而且是写在报表里的那种。

第一版的报表行是这样的：

```python
bits = [r.get("thr_b") or 0 for r in runs] + [r.get("thr_a") or 0 for r in runs]
nz = sorted({b for b in bits if b})
print(f"throttle 位图非零的读数: {sum(1 for b in bits if b)}/{len(bits)}  出现过的位={nz}")
```

`or 0` 把 `None`（字段没读到）折成 `0`（读到了，值是全 0）。于是**「驱动不报这个
字段」和「这个字段一直是 0」输出同一行字**，而且看起来都像好消息。第一版打出的
「throttle 位图非零的读数: 0/16」完全可能是 16 次 `None`。修法是先分开计数：

```python
got = [r.get("thr_b") for r in runs] + [r.get("thr_a") for r in runs]
have = [b for b in got if b is not None]
...
print(f"throttle 位图: 读到 {len(have)}/{len(got)} 个读数  非零 {sum(1 for b in have if b)}/{len(have)}")
```

有了这一行还不够 —— 它只把「没读到」暴露出来，仍不能证明「读到了 0」的那 0 是真 0。
所以另写一个探针 `benchmarks/probe_clock_dvfs.py`，从源头逐字段问一遍，并且拿一个
**已知会动的量**去校准：

- **控制 0 字段可得性**：14 个候选字段逐个单独查询。⚠️ 必须分开问 ——
  `--query-gpu` 只要有一个字段不被支持就**整条查询**返回码非 0，合并问会把「这台
  驱动没这个字段」伪装成「查询失败」，看起来像 NVML 坏了。
- **控制 1 采样链路对负载有反应**：同一条查询里的 `utilization.gpu` 是已知会随负载
  从 0% 走到 100% 的量。它动了 → 采样活着；它不动 → 是探针坏，时钟的零结果不作数。
- **控制 2 功耗/温度确实被拉上去**：这一条区分两种含义相反的「不动」——
  「离上限很远所以无事可降」（→ D2 在这类机器上永远不会触发）与
  「已顶到功耗/温度上限而读数仍不变」（→ 驱动上报口径本身不可信，连观测都不该信）。

控制 0 的实测输出（节选）：

```
  ✅       生产用到 utilization.gpu                  原始='0'
  ✅       生产用到 clocks.sm                        原始='1980'
  ✅       生产用到 clocks.mem                       原始='2619'
  ✅       生产用到 clocks.max.sm                    原始='1980'
  ✅       生产用到 clocks.max.mem                   原始='2619'
  ✅       生产用到 clocks_throttle_reasons.active   原始='0x0000000000000000'
  ✅ 文本      补充   pstate                           原始='P0'
  ❌ 不支持   clocks.applications.sm           Field "clocks.applications.sm" is not a valid field to query
  ❌ 不支持   clocks.max.boost                 Field "clocks.max.boost" is not a valid field to query.

  可查 12/14   数值可用 11   有字段但非数值: ['pstate']   值为 N/A: 无
  ✅ 生产 6 个字段全部可解析 → 「时钟没变」不是量具瞎了
```

顺带修掉探针自己的一个分类错误：第一版把 `pstate` 归进「值为 N/A」，因为 `parse()`
只认数字，而它的值是字符串 `P0`。字段完全可用，**而 P0 恰恰是「性能状态在最高档」
的证据** —— 误分类会把它从报表里丢掉。「非数值」和「N/A」从此分开。

## 三、空载 / 满载实测

`8192² fp16 matmul` 连续循环 40 s —— 这是 tensor-core 满载、也就是对 DVFS 压力最大的
一类负载（探针里那点道理：要造出降频就该造最坏情形，而不是造一个「像负载」的负载）。

| 字段 | 空载 12 s | 满载 40 s | 恢复 12 s |
|---|---|---|---|
| `utilization.gpu` | median 0%（范围 0~0） | **median 100%** | median 0% |
| `clocks.sm` | **1980 MHz**（范围 1980~1980） | **1980 MHz**（范围 1980~1980） | 1980 MHz |
| `clocks.mem` | 2619 MHz | 2619 MHz | 2619 MHz |
| `power.draw` | 126 W（126~127） | **327 W**（126~329） | 127 W |
| `temperature.gpu` | 47 °C | **56 °C**（47~58） | 51 °C |
| `pstate` | `['P0']` | `['P0']` | `['P0']` |
| `clocks_throttle_reasons.active` | 11/11 读到，非零位无 | 34/34 读到，非零位无 | 11/11，无 |
| `clocks_event_reasons.active` | 同上 | 同上 | 同上 |

判定行由探针自己算出来，不靠人读表：

```
  控制 1  采样链路: util 空载 0% → 满载 100.0%   ✅ 活着
  控制 2  满载 SM 时钟 1980 → 1980 MHz  降幅 +0.00%   满载功耗 326.69 W / 上限 500.0 W = 65%
  → 本机 SM 时钟在 0%~100% 占用之间**不动**: 离上限很远, 没有降频压力 (这类机器上 D2 永不触发)
```

这张表的结构值得多看一眼，因为它是「零结果可信」的全部理由：
**同一批传感器里 util 走了 0→100%、功耗走了 126→327 W（2.6 倍）、温度走了 47→58 °C，
只有时钟一格没动。** 也就是说这不是「什么都没读到」，而是「读到了别的东西在动，
时钟没跟着动」。控制 2 进一步给出机制：满载功耗只到上限的 65%，温度 58 °C，
**DVFS 没有任何下调的理由** —— 这块卡跑在功率与温度的舒适区里。

### 一个必须收回的说法

`clocks.applications.sm` 在本驱动上**不是合法字段**（见控制 0）。所以「这块卡被
application clock 锁在 1980」这句话**没有证据**，它是我在拿到这张表之前的推测，
差点顺着写进文档。诚实的版本是：`clocks.sm` 恒等于 `clocks.max.sm`，至于这是锁频
还是「无压力所以不降」，**这台机器上的 nvidia-smi 无法区分**。P0 + 65% 功耗余量
两条更支持后者。

## 四、代价实测：字段数是免费的，调用次数才是钱

第一版实现是给时钟**单独开一次** `nvidia-smi` —— 那意味着每次 L3 多两三个进程。
实测把这笔账算清楚后合并成了一条查询：

```
6 字段合并查询  median=135.0 ms  p90=163.2 ms  max=271.4 ms
单字段(旧 util) median=138.2 ms  p90=182.7 ms  max=244.7 ms
→ 加 5 个字段的边际代价 = -3.2 ms
```

**边际代价是负的**（即落在噪声里）。成本全在进程启动 + NVML 握手，与字段数无关。
这条数字决定了实现形状：能塞进已有查询的旁证就塞进去，值得单独开口的只有「窗口
结束后再读一次」这一处 —— 于是整个 L3 只多付 1 次调用 ≈ 135 ms。

顺带一个可用的结论（对任何要往 runner 加旁证的后续工作都成立）：**在 nvidia-smi
上多问几个字段不要钱，多开一次调用很贵。**

## 五、落地了什么，以及为什么没人读它也没关系

- `read_gpu_state()`（`remote_scripts/run_kernel.py`）一次调用返回
  `util_pct / sm_clock_mhz / mem_clock_mhz / sm_clock_max_mhz / mem_clock_max_mhz / throttle_bits`；
  `[N/A]` 一律解析成 `None` 而不是 0，hex 位图按 16 进制解析。
- `_sample_gpu_idle()` 改为复用 `read_gpu_state()`，返回值多一个「第二次采样的完整
  状态」，**没有为此多付一次进程**（原本就要采两次 util）。
- benchmark 出口多一次 `read_gpu_state()`，把 8 个 `*_before` / `*_after` 键并进结果。
- `operator_agent/verifier.py` 的搬运表显式列出这 8 个键。⚠️ 这一步不能省 ——
  缺口 1 的教训就是「runner 报了但控制端没人搬，于是没人读」。

这些键目前**没有任何判据消费者**，这是刻意的：判据要等时钟真的动。与 `doc 20` 里
「算了没人读」那几处不同，这里的「没人读」是记录在案的设计状态，而不是遗漏。
窗口前的读数取自 util 第二次采样（warmup 前 ~130 ms 那一刻），不是 launch 那一瞬 ——
这条限制写在 `read_gpu_state()` 的 docstring 里，随数据一起交给下游。

## 六、改判条件（什么时候回头做 D2 的判定）

1. **换平台 / 换云实例**：只要满载段 `clocks.sm` 相对空载段降幅 > 一个测量档位
   （`measurement.NOISE_BAND` 的 degraded 档 = 0.4%），跨轮比 gflops 就得先按时钟
   归一化。判据用**降幅**而不是「时钟 < 上限」这种绝对值：本机的降幅恰好为 0，
   说明绝对值阈值在锁频/无压力机器上永不触发。
2. **throttle 位图出现非零位**（`0x4` SW power cap、`0x8` HW slowdown 一类）：
   那时「慢」有了外部归因，走 `doc 20` C#1 的外部归因路径，而不是让 Conductor 把
   邻居造成的退步算成核的质量问题。
3. **想做归一化而不是加阈值**：得先有跨轮时钟散布。除以常数等于什么都没除 ——
   `verify_util_decontamination.py` 的报表行「按时钟归一化收下 X% 的跨轮散布」
   本机三次实测 X = **0%**（时钟 iqr/median = 0.00%，范围 1980~1980）。

复现命令：

```
scp benchmarks/probe_clock_dvfs.py <ssh别名>:/tmp/ && \
  ssh <ssh别名> "cd ~/operator_agent && CUDA_VISIBLE_DEVICES=2 \
    ~/.local/bin/uv run python /tmp/probe_clock_dvfs.py"
```

## 七、顺带的读数

生产 L3 路径连跑 8 次 × 3 个形状（合并查询前后各跑了一轮，两轮结论一致）：

| 形状 | SM 钟（窗口前） | 窗口内漂移 | gflops 跨轮 iqr/median | 归一化收益 | `paired_speedup_dispersion` |
|---|---|---|---|---|---|
| layernorm 1024×768（launch 主导） | 1980（1980~1980） | +0 MHz，max \|Δ\|=0 | 0.66% → 1.43% | 0% | 0.0234 → 0.0251 |
| layernorm 16384×6144（degraded） | 1980 | +0 MHz | 0.12% → 0.07% | 0% | 0.0109 → 0.0109 |
| matmul 4096³（valid） | 1980 | +0 MHz | 0.22% → 0.10% | 0% | 0.0016 → 0.0016 |

- 「旧单次快照规则」与「现 min 规则」的污染触发率 8 轮都是 0/8 = **0%**，
  与 [`34`](34-util快照去污染复核.md) 的结论一致（生产节奏撞不上那个窗口，
  这层是保险不是警报器）。
- `paired_speedup_dispersion` 三档单调跟随有效性档位、量级与 doc 34 复现
  （valid 千分之 1.6 / degraded 百分之一 / uninformative 百分之 2.5）。
  它仍然**没有任何判据消费者**（D4 用的是跨次散布，不是它）。
- 最有信息量的一行是**第一行的 0.66% → 1.43%**：launch 主导档的跨轮散布在两次
  连续运行之间自己就漂了一倍，而 `NOISE_BAND[UNINFORMATIVE] = 2.0%` 只留了 1.4 倍
  余量。这一档的带宽定得紧不紧，值得在积累更多样本后复核一次（D5 的素材）。
