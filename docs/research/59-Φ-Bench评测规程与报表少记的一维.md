# 59 · Φ-Bench：一份把「参考解自己不合格就作废任务」写进规程的评测，与我方报表少记的那一维

第 58 篇之后的第一轮扫描。选中这一篇的理由不是它新（`2609.10226v1`，2026-09-10），而是它把我方
**已经撞过的那几件事**写成了外部规程：评测规格不可变（#87）、层级门先于 reward（#76②/#91）、
配对计时（D3/#32）、反作弊要查「绕过检查」而不是只查禁用的 API（#84），以及最要紧的一条 ——
**参考解自己不合格时，作废的是任务，不是候选（#94 第 2 片）**。

---

## 一 · 取数（一手，含产行号的那把尺）

| 论文 | `arXiv id` | 字节 | 行数 | 读到哪 |
|---|---|---|---|---|
| Φ-Bench: Can Large Language Models Engineer the Infrastructure of AI? | `2609.10226v1` | 168,484 | 617 | 摘要、§1、§2（任务合成三法 + Quality Control 行 165-175）、§3 Evaluation Metrics（行 176-193，含配对公式与作弊检测全文）、§4 实验设置与主表读数（行 306-333）、§5 迭代/预算分析（行 317-320） |

取数姿势与前四篇相同：`https://export.arxiv.org/html/2609.10226v1` 直连 → `curl` 落盘
**168,484 B** → `python3 -B benchmarks/arxiv_html_to_lines.py` 抽出 **617 行**。上面那张表是唯一
真源，重跑与核引用都在仓里：

```
python3 -B benchmarks/recheck_arxiv_anchors.py            # 默认把 58/59 两份表并起来核
python3 -B benchmarks/recheck_arxiv_anchors.py --doc docs/research/59-*.md   # 只核这一份
```

⚠️ 这一轮顺手撞出这条检查自己的一个假红形状：它原先只读**一份**文档的表（默认 doc 58），
而 `CITATIONS` 是一份扁平清单 —— 第一次指到 doc 59 跑，报的是 **13/32**，19 条 ❌ 全是 doc 58
的引文（它们没被解析到表里，于是「锚点对不上」）。⇒ `--doc` 改成可重复、默认并读
`DEFAULT_DOCS = ("docs/research/58-*.md", "docs/research/59-*.md")`；**新开一份调研文档要往那
两行里加一条**，漏掉的表现不是报错而是那份表从来没被重跑过（I28 那一族）。改完真跑：
**36/36 对上、退出码 0**（24 条来自 doc 58、12 条来自本篇）。

本文档逐字引的十二句全部登记进 `CITATIONS`（行 78 / 103 / 172×2 / 173 / 177 / 181 / 183 / 193×2 /
306 / 319），每条都先在抽出的文本里 `grep -F` 核过命中数为 1 才写进表（I28 的规矩：needle 不许凭
印象拼）。

---

## 二 · 它的三档任务形态，对照我方只有一档

85 题 = **KFC 55**（单文件内的核函数补全/优化，接口与输入输出语义给定）+ **LHI 20**（仓库尺度、
跨文件的长时程实现）+ **E2EO 10**（端到端系统优化，开放式、假设驱动）。行 103 那句批评恰好是
照着我方这类系统说的（逐字）：

> "Although several benchmarks have evaluated LLMs on GPU kernel implementation … their evaluation
> settings are typically restricted to individual functions or small collections of isolated
> components."（行 103）

摘要那一句是同一意思的另一版措辞（"existing benchmarks mainly focus on isolated kernels, predefined
operators, or pre-specified optimization targets"，行 78）—— 两处我都核过命中行，**别把措辞相同的
两句当成同一句**：`citation_in_range` 是按声明区间找 needle 的，写错行号会红（I28 的那条范围限制）。

⇒ 我方 harness 今天的 `CASES` 全在 KFC 那一档（单算子生成 + 固定 shape）。这不是「要去做 LHI」
的意思 —— 我方是编排现有算子生成流程的 agent，不是仓库级工程师 —— 但它决定了**我方通过率的
可比范围**：拿 `#48` 那套量具第二轮去对照外部数字时，只能对 KFC 那一档，且必须写明档位
（与 #90「按 DSL 分层」是同一句话的另一维：分层要连**任务形态**一起分）。登记 ⇒ #95。

---

## 三 · 度量规程四条（逐字，带行号）

### 3.1 任务准入是一道双向门（行 172-173）

> "For a task evaluated with the performance metric, the reference solution must achieve a stable
> performance of at least 1.15 in the official execution environment, whereas the starter or no-op
> solution must not outperform the baseline. For a task evaluated with the implementation metric,
> the reference solution must pass all test cases in the official environment, while the starter
> solution must fail at least one test."（行 172）

两侧各挡一种失效：**参考解跑不赢基线 ⇒ 这题没有可达的改善**（挡掉「做出来也不知道有没有用」的题），
**no-op 却超过基线 ⇒ 计分口径本身在奖励不动手**（挡掉「作弊即满分」的题）。对照我方：

| 它的门 | 我方今天 | 差距 |
|---|---|---|
| 参考解在官方环境里稳定 ≥1.15 才准入 | 任务的 `min_speedup=5.0` 是**候选侧**判据；没有任何一条检查说「参考实现自己能不能跑赢基线」 | 我方 `benchmarks/harness_pass_rate.py` 的 `CASES` 只有 baseline，没有「参考解也要过一遍」这道 —— 于是「题目不可达」与「模型不行」在通过率里混成一件事 |
| starter/no-op 不许超过 baseline；实现类题 starter 必须至少失败一个 | 我方有「全零核通过每次试验」的实测（doc 57 臂 1-4、#82），但那是**候选**层的读数，不是**任务准入**层 | 同一件事的两个位置：我方把它当判据缺陷记（#82 需机时定阈），它把它当题库质量门 |

⇒ 登记 **#95**（任务准入门：参考解自证可达 + no-op 不许超基线；与 #61/#65 的实测分母、#48 的难度
选择互相衔接 —— 三者是同一道题的三个面，不必都做，但做 #48 那轮扫描时这两条数要顺手采下来）。

### 3.2 参考坏了作废的是任务，不是候选（行 173）—— #94 第 2 片的外部一手

> "A task whose reference solution fails these requirements is considered invalid and is revised or
> excluded rather than assigned a score."（行 173）

这句话正是 #94 那一片停在门口没迈出去的那一步。第 1 片把测量洗诚实了（`compared_elements` 只数
真比上的那些、新报 `reference_nonfinite`、全非有限参考 ⇒ 判不过并把归因写对），但**判决侧仍然
把「本级判不出」当成一次候选失败**去走 Fixer 那条路 —— 而 Fixer 的处方表按「哪一级报的失败 +
错误文本」分支，一句「参考另有 4096 个非有限值」在它眼里就是数值失败，开出来的是一张指向候选的
处方。它的处置与我方推理一致：**证据坏了就撤证据本身**（revise or exclude the task），
不 assign a score。⇒ #94 第 2 片的依据从「我方觉得应该这样」升级成「外部把它写成了规程」，
这一条现在有了出处。挂号内容不变（归属路由 + 远程 `run_kernel.py` 同一个形状）。

### 3.3 配对聚合：它判 median-of-ratios，我方判 ratio-of-medians —— 而**两条读数我方都已上报**（行 181-186）

> "For a valid candidate, we measure its performance relative to the unmodified baseline using an
> AB-BA paired measurement protocol with at least five measurement pairs." （行 181）
> 公式：`s = median_i(t_i^base / t_i^cand)`（行 182）
> "A candidate receives a reward of zero if the reference solution does not yield a valid
> performance, i.e., s_ref ≤ 1."（行 183）

核过我方代码（`verifier.py` 的 L3 那一段）：判定走的是 **ratio-of-medians**（`median_time_ms`
优先、基线同理），而 **median-of-ratios** 作为旁证一起上报，键名是 `paired_speedup_median`，
旁边写明了「不静默换口径」这个决定。⇒ 结论：**我方不是漏了这条，而是声明过选了另一条**。
真正缺的是第三件事——这两条口径**什么时候会反号**（同批数据上一个说快一个说慢），那需要真机
分布，今天没有数。登记 ⇒ #96（拿 #34 采过的配对散布核一次两口径的分歧率；与 #82/#93 同批机时）。
顺带一条它比我方多的：`s_ref ≤ 1 ⇒ reward 0` 是**任务级**的失效，与我方 3.1 那条同源。

它的 reward 刻度也值得记一句：`min(1, ln(s/s_ref)/ln s_ref)` —— 达到参考的平方倍记满分，
分子分母都是对数。我方 L3 是线性阈（`min_speedup`、`NOISE_BAND`、达成率百分比），
这是**第三种**「怎么把加速比折成分数」的做法；不吸纳（我方不做 reward，做判决），但登记为
#93 的口径参照之一：晋升时换尺必须显式换并重测，而「换到哪一把」这件事现在有了第二个外部样本。

### 3.4 反作弊两条通道：轨迹级规则 + 一个 proctor agent（行 193）

> "The rule-based monitor scans the complete solution trajectory and final code changes for
> predefined prohibited behaviors, such as searching GitHub for the original implementation,
> retrieving the corresponding patch or commit diff and recovering upstream code from published
> Python packages. A dedicated proctor agent additionally inspects trajectories and submissions for
> detecting more complex hacking behaviors, such as hard-coding expected outputs, skipping or
> bypassing correctness checks, and introducing branches that artificially inflate the measured
> performance. If either proctoring mechanism detects a prohibited behavior, the corresponding task
> attempt receives a reward of zero."（行 193）

我方 `anticheat.py` 今天查的是**一条静态签名**（禁用的原生算子回退调用）。它列的六项里我方一样
都没有：前三项是**轨迹级**（搜原实现 / 取 patch / 从已发布包里捞上游代码 —— 我方 prompt 里塞的
SKILL 与配方确实让模型「抄现成的」在通路内可行），后三项是**语义级**（硬编码期望输出、绕过正确性
检查、为抬高性能而生的特殊分支）。⇒ 登记 **#97**：把这三条语义签名做成可跑的检查（至少
「候选里出现被期望输出常量表」与「测量路径里出现按 shape 跳过的分支」两条能在控制机上构造用例），
轨迹级那三条要动 harness 的日志层，先记不排期。**#84**（失败按 gamed/overfit/benign 分档）的
分档判据因此有了具体的三类可观察形态，不再是抽象标签。

---

## 四 · 本轮当场撞出的一处我方账：报表的 knobs 少记了模型侧那一维

实验设置那一句（行 306）把他们为什么要钉住它写得很清楚：

> "We allocate 8 CPU cores, 32 GiB of memory, and a single NVIDIA H20 GPU to every task, while
> disabling all other GPU types to ensure a consistent evaluation environment."（行 306）

同一行还钉了 "the models were evaluated with reasoning effort uniformly set to max"，而行 319-320
给出为什么要钉：**effort 与分数不单调**（"performance does not increase consistently with
reasoning budget"），而低档可以让一个模型掉 **45%** 的分。

我方这边的现状（本轮逐条核过，不是推断）：

* `benchmarks/harness_pass_rate.py` 落盘的 `knobs` = `{min_speedup, max_error, max_optimize_rounds,
  repeats, min_denom, cases}` —— 六项里**没有一项是模型侧的**；
* `operator_agent/llm_client.py` 的请求体里带 `"temperature": temperature or self.temperature`
  （默认 0.3）与 `"chat_template_kwargs": {"enable_thinking": False}` —— 也就是说我方所有通过率
  数字都是在**关掉思考**的配置下测出来的，而这个决定既不进 `git_head` 也不进 `knobs`；
* 报表自己那句警告「引用这些数字时必须带上 `git_head` 与 `knobs` —— 换一次配方/判据/prompt,
  上一份报表就只是历史，不是基线」—— 它保护了配方侧，**没保护模型侧**。

⇒ 这不是「外部有一条我们没有的好做法」，而是**我方一条声明与它记录的内容不一致**：报表说 knobs
是读数的全部条件，而 knobs 少了一维，且那一维按外部证据值 45%。⇒ 登记 **#98**（零机时、可落码）：
`knobs` 补 `model` / `temperature` / `enable_thinking` / `max_tokens` 四项，`compare_runs` 两侧
并排印，与 #89（端点自报模型标识）合成同一条读数链 —— 那条记录的是「服务端实际用的模型」，
这条记录的是「我方请求的配置」。

---

## 五 · 追读目标（未读，不下结论）

* 参考文献行 468：**SOL-execbench — speed-of-light benchmarking for real-world gpu kernels against
  hardware limits**。标题与我方 #61/#65（把「可达上界」从厂商标称峰值换成实测定律）直接同题。
  本轮只读到标题与出处，没有读方法 ⇒ 不写任何判断（#92 那条纪律的形状：未读 ⇒ 只登记）。
* Φ-Bench 的附录（85 题清单、每题的 editable region 定义）没读 —— 3.1 那条「starter 不许超基线」
  的**具体判法**（几个 pair、什么阈）很可能在附录里，届时再核。

---

## 六 · 决策与挂号

| 外部机制 | 决策 | 落点 |
|---|---|---|
| 任务准入双向门（参考解 ≥1.15、no-op 不许超基线、starter 必须至少失败一个，行 172） | **吸纳（判据）** | **#95**（新）：与 #61/#65 的分母、#48 的难度选择同批采数 |
| 参考解不合格 ⇒ 作废任务而不是给候选打分（行 173） | **依据加固**（判据我方已推到同一点） | **#94 第 2 片**：归属路由 + 远程 `run_kernel.py` 同一个形状，这片现在有了外部规程做依据 |
| AB-BA 配对 ≥5 对、判定用 median-of-ratios、`s_ref ≤ 1 ⇒ 0`（行 181-183） | **不抄口径，但补一次对照** | **#96**（新）：我方判定走 ratio-of-medians 是**声明过的**选择（`paired_speedup_median` 已在旁边上报），缺的是两口径分歧率的实测 ⇒ 需机时，与 #82/#93 同批 |
| 对数刻度 reward（达参考平方倍记满分，行 184-186） | **登记为参照**，不吸纳 | #93（晋升换尺要显式换并重测）多了一个外部样本 |
| 反作弊双通道：轨迹级规则 + proctor agent 查「硬编码期望输出 / 绕过检查 / 抬高性能的空分支」（行 193） | **改造后吸纳** | **#97**（新）：三条语义签名做成可跑检查；轨迹级三条要动 harness 日志层，先记不排期。#84 的分档判据因此具体化 |
| 单卡固定 + 禁用其他卡型 + best-of-16 + effort 统一 max（行 306）；effort 不单调、低档掉 45%（行 319-320） | **我方报表少记一维**（本轮核出） | **#98**（新，零机时可落）：`knobs` 补 model/temperature/enable_thinking/max_tokens，与 #89 合成一条读数链 |
| 三档任务形态 KFC/LHI/E2EO（行 107/150-151） | **吸纳（口径标注）** | 并入 #95/#90：通过率报表要标**任务形态**这一维，与外部数字对照时档位必须写明 |
| 层级门先于任何 reward（行 177："Both metrics are correctness-gated"） | **已吸纳过**（#76②/#91 那次） | 无新增；这一句是我方层级门设计的第三个外部同形 |

**没做的事**：仍然**零机时、零真机活、零 LLM 调用** —— 本篇全部是我方代码与报表的静态核对。
`#95/#96/#97/#98` 四条里只有 #98 是我方能立刻落码的（其余分别要机时或要先动 harness 的日志层）。
Φ-Bench 的 §4 主表我只读了 KFC/LHI/E2EO 三档的总分行（行 330）与迭代曲线那段（行 317），
逐题分数、baseline 定义、以及 3.1 那条门槛的具体判法都没读 —— 记在 §五 而不是假装读完。
