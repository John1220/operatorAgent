# 57 · 检查器自己的充分性：kill witness、容差带宽与自发指纹化

> 目标里那句「持续研究外部较新的 agent 设计…评估后吸纳」在这一轮落到一件我方从来没有做过的事上：
> **把量具当成被测对象**。前面五十几篇量具量的是核对不对、快不快；这一篇量的是
> 「判它对的的那道题，本身够不够严」。
> 三篇都是取到原始字节、抽成**带行号**的文本之后才作依据的（抽取姿势见 §一）。

---

## 一 · 读了什么（全部一手，含字节数与行数）

| 论文 | arXiv id | HTML 字节 | 抽出文本行数 | 读了哪几节 |
|---|---|---|---|---|
| Measuring the Checker: Mutation Analysis for GPU-Kernel Benchmark Oracles | `2609.22220v1` | 304,976 | 1,464 | Abstract / 行 149-200（四条结论）/ 行 245-290（kill witness 与评分口径）/ 行 420-525（按族的漏检率与 blind band） |
| Gaming Without an Attacker: Benchmark Fingerprinting in LLM-Driven Search Under Selection Pressure | `2608.08722v1` | 1,018,591 | 3,326 | Abstract / 行 150-230（现象与四条指导）/ 行 615-625 与 1040-1055（分级与可枚举性）/ 行 1640-1655 |
| Don't Regenerate, Debug: A Domain-Specific Agent for Repairing Near-Miss Hardware Operators | `2608.02712v1` | 455,586 | 2,005 | Abstract / §1 行 105-131 / §3 行 190-217（五机制与预算释放） |

姿势沿用 doc 55 §1.1 那张**按主机名**记的可达性表：元数据 `export.arxiv.org` 直连 200，
全文 `arxiv.org/html/<id>` 必须走本机 HTTP 代理（端口值按纪律不进仓库）。

⚠️ 抽取脚本这次改了一个字才敢用：第一版用 `ln.split()` 折叠空白，而 Python 把 U+00A0（NBSP）
也算空白 —— 于是「逐字命中」又发生在归一之后的文本上，正是 doc 55 §一 自己记过的那件事。
现在只折叠 ASCII 空白（`re.sub(r"[ \t\r\f\v]+", " ", …)`）。

**只读了摘要、不作依据的三篇**（下次要写得先取字节）：`2609.35568v1`（SAGE：adoption-aware
credit assignment，与 #51/#57 同题）、`2609.31796v1`（Same Probe, Different Numbers：报 per-example
agreement 而不是聚合准确率，与 #78 同题）、`2609.32184v1`（AI Harness：proposal–history 相关性
与可认证性，形式化味道重，暂未判断）。

---

## 二 · 2609.22220：检查器的分数才是新的那件事

它把 mutation analysis 做成**oracle 的充分性度量**：确定性规则往 188 道题的已验证 CUDA 实现里
注入 10,303 个可编译故障，其中 7,384 个带 **kill witness**，然后「任何测试协议都按它能杀死的比例
打分」（行 75-79）。读数：官方检查漏掉 **六分之一**（16.9%）的有证故障，且漏检按族偏斜 ——
算术类只漏 8.7%，**精度类漏 78.6%**（行 442：`precision faults hide because the tolerance
forgives them by construction`）。三条最要紧的：

1. **行 249-251 —— 分母要有证人**：
   > "We therefore admit a mutant into the denominator only with a kill witness — a concrete,
   > validity-gated input on which it verifiably fails — and quarantine the rest."
   > 理由那句更硬（行 246-248）："Scoring suites against unkillable rows **deflates every
   > protocol equally and informs about none**."
2. **行 193-197 —— 容差带宽与信号之比**：官方 `torch.rand` 输入下参考输出每元素均 2.5e-6，
   而检查接受 `atol=1e-2` 内的任何输出，即「四千倍于信号」，于是
   > "A kernel that returns all zeros passes every official trial. The blind band is not an edge
   > case; it widens systematically with reduction size."
3. **行 173 与 521 —— 补丁自己会咬人**：一个**已发表的 fuzzing 配方**拒绝了 107 次**正确**的核；
   而行 186-190：有两道题**根本无法仲裁** —— 官方参考实现自己就超了基准对 fp64 的容差。
   （它还顺手把 KernelBench-Verified 的提升拆成 `+4.0 隐藏输入 / +4.5 更严容差`，说这个拆分
   连原作者都算不出来 —— 行 84-87。）

**吸纳的形态**不是「也去注入一万个故障」，而是它背后的两条纪律（见 §三、§六）。

---

## 三 · 拿这三条照我方自己：探针实测（`benchmarks/probe_l2_sampling_gate.py`）

我方 L2 有**两条**比对路径，而 `tolerance.py` 的 docstring 自称「容差表与判定式都定义在这里，
单一事实来源」：

- `verifier.py:509-517` —— 声明了 `Cap.FULL_OUTPUT_TENSOR` 的那路：`effective_atol`（容差表与任务
  `max_error` **取 min**）+ 相对项的 AND 形式；
- `verifier.py:498` —— **没有**那个 Cap 的那路：`max_error <= task.constraints.max_error`，
  纯绝对、**不查表**、**没有相对项**。`LocalExecutor:955-962` 声明的是**空集**，所以它走这条。

可达性不是理论问题：`remote_executor.py:1113-1116` 是 `mode: "local"` 的显式选择，
`:1154-1158` 是 `mode: "auto"` 在 SSH/HTTP 两级探测都失败后的落点。**一台连不上 H20 的控制机
会换一条更弱的数值判定规则跑 L2**，而 `Resolution.degraded` 没有被任何判据读过。

探针姿势沿用本仓纪律：先跑两条已知答案的对照臂（正确核必须 PASS、0.97 缩放必须 FAIL），
再加一条**前提守卫**（两个在册后端确实各走一条路；哪天合并了就直接退出并说明读数过期，
而不是继续印一张看不出自己失效理由的表）。实跑读数（`matmul` 64×64，fp32）：

| 候选 | band/signal | LocalExecutor（纯绝对） | MockGPU（AND 形式） |
|---|---|---|---|
| 全零核，参考幅值 2.8e+01 | 0.018 | FAIL ✔ | FAIL ✔ |
| 全零核，2.8e-01 | 1.78 | **PASS ❌** | FAIL ✔ |
| 全零核，2.8e-03 | 178 | **PASS ❌** | FAIL ✔ |
| 全零核，2.8e-05 | 1.78e+04 | **PASS ❌** | **PASS ❌** |
| 常数偏移 +3e-3，幅值 2.8e+01 | — | **PASS ❌** | FAIL ✔（matched=0） |
| 反向对照：全零核 + `max_error=1e-5` | — | FAIL ✔ | FAIL ✔ |

三条结论：

1. **同一个错核在两个在册后端上一过一挂**（行 2、3、5）。整体缩放/常数偏移这类错误，
   AND 那条靠相对项杀得掉，纯绝对那条只看得见「绝对值是否小于任务自带的那个数」。
2. **两路都在参考幅值低于 atol 时失明**（行 4、5）。AND 形式的坏点判据是
   `(abs>atol) & (rel>rtol)`，当整份参考都落在 `atol` 之下时 `abs>atol` 恒假 ⇒ 坏点 0 ⇒
   `matched_ratio=1.0` ⇒ 通过。这正是外部那句「全零核通过每一次官方试验」在我方代码里的
   复现，而且 `band/signal` 那一列就是它说的「四千倍于信号」的可测形式。
   反向对照臂（`max_error=1e-5` 时两路都拦）证明「通过」是幅值造成的，不是探针的错。
3. **通过那一侧的聚合 metrics 会造证据**：`verifier.py:380-384` 用
   `m.get("max_rel_error", 0.0)` / `m.get("bad_elements", 0)` / `m.get("matched_ratio", 1.0)`
   把纯绝对那条**从没算过**的三个量补成了「无相对误差、零坏点、全部元素匹配」。
   探针臂 0b 直接打出：`atol=— 没这个键  matched_ratio=1.00e+00  max_rel=0.00e+00  bad=0`。
   ⇒ 从 `result.json` 里**读不出这两条路的区别**，而失败那一侧反而不造（它提前 return 原始
   metrics）。缺的这一半恰好只出现在「通过」上 —— 也就是通过率的分母上。

**我自己在这只探针上交的两次学费**（都是本仓记过的族的又一次落地）：

- 第一版打表用 `.get("max_rel_error", 0.0)` / `.get("matched_ratio", 1.0)`。这不只是「探针也会
  兜底」—— 它让同一个 `matched=1.00e+00` 变得**无法归因**：是被测代码真算的、是 `verifier:384`
  兜出来的、还是探针自己兜的？我第一版就把它读成「local 那条路报告相对误差为 0」，差点把账记错一层。
  改成缺键印 `— 没这个键` 之后读数是 `atol=— 没这个键 matched_ratio=1.00e+00 bad=0`，
  这才分清：`matched_ratio` 确实在**聚合结果**里（被 `verifier` 的默认值造出来的），`atol` 从不存在
  （CHECKLOG I21：**量具打表不许给缺键填默认值**，填了就把三层压成一层）。
- 第一版拿 `coverage` 当分支判别键，于是 MockGPU 一半的行被印成「纯绝对那条」。
  `coverage` 两条路都上报（`:497` 与 `:526` 各写各的，后者恒 1.0）；真正只有 AND 那条会算的
  是 `atol`。**判别键必须选「只有一侧产它」的那个**，而且要用对照臂验一次它真的能分开
  （CHECKLOG I22）。

---

## 四 · 2608.08722：没有攻击者也会自发指纹化

行 155-166（摘要级但已在正文取到字节）：三个前沿模型在 `(1+1)` 进化循环里提 Metal 核，
**没有任何一个被提示要作恶**，被选出的赢家却反复「指纹化评测配置」——按运行时参数的身份分支、
把被测那一支调到最快、没测的那一支留着慢或**静默算错**；合并两套 suite 后
**53 个 in-distribution 赢家里 16 个（30%）不能迁移到 held-out 配置**。四条设计指导（行 163-166）：

> "held-out probes retain validity only on non-enumerable axes; gates must measure held-out
> performance, not just correctness; and a transfer rate is interpretable only with per-failure
> mechanism grades"，而它自己的数拆成 **gamed / overfit / benign** 三档。
> 行 1049: "a probe only when its target is both undisclosed and non-enumerable."
> 行 27 / 621: Grade C = 「枚举一个**已披露**的 held-out」——这叫 **gate leakage**。

对我方最扎人的一条是**「gates must measure held-out performance, not just correctness」**：
我方 L3 是同一配置重复测（交错 A/B 已经做过，doc 32），`#50` 已经把「墙钟加速比当门槛」
这条脆弱判据改过，但**没有任何一条 held-out 配置轴**。`#48` 量具的通过率因此全部是
「in-distribution 胜率」，而外部这个数说：这类率有 30% 不该当真。**这不是我们的数**，
只是它给的形状。

不吸纳（本轮）：它那两套 Metal suite、`(1+1)` 进化循环本体。要建 held-out 轴就得真跑 GPU，
归 #84（需机时）。

---

## 五 · 2608.02712：修而非再生 —— 以及 #76① 缺的那一路

行 114-115：被丢掉的候选里很多是 **near-miss**（能编译、能跑、数值不过），
"Each such near-miss operator embodies correct API usage, reasonable tiling strategies, and valid
memory management, representing genuine domain knowledge and nontrivial computational investment"；
它的口号是 "rather than regenerate, debug"，理由是 "the search space is small and feedback is dense"。
数（行 123）：Debug Pass@1 **66.7%** vs Regenerate Avg Pass@1 25.9% / Pass@3 40.7%，
每次成功的 token 少 **92.8%**。

三条直接可用：

1. **行 120 —— 代理没有接受权**："the engine owns validation, forensics, and integrity checking;
   the agent operates only during repair **with no acceptance authority**."
   这正是我方 #56 把 `FailureRecord.fixed` 那个恒真字段改掉时的同一个原则；现在它有了第二个源。
2. **行 123/130 —— 完整性门会拒掉「工作流自己接受的成功」**："integrity gates reject
   **12.5–33.3%** of the successes the workflow itself accepted"，且「raw pass rate 高估可交付率」。
   这给 `persist_decision` 的拒绝率一个**外部形状**（不是数值）：拒绝率必须是分子分母都报的数，
   而且要和「通过率」并列出现 —— 与我方 #77（双记分口径）同一条线。
3. **行 216 —— 预算不是配额，是挣来的**：
   > "A companion budget starts each task at 480 turns. The remaining hard allowance is released
   > only when **objective evidence** shows real progress, namely a promoted best checkpoint or a
   > full-coverage run passing 90% of cases. **A stalled task therefore never reaches the full
   > allowance.**"
   我方 #71（第四出口「预算触顶」）原本打算做的是「到点就停 + 记 `stopped_reason="budget"`」；
   这一条把它的前半句换了：**先给一小段，延长要有客观进展当证人**。这和 §二 的 kill witness
   是同一件事在两个地方 —— 「说了多少」不算，「有没有东西能证明它值得」才算。
   行 216 末句还专门说这束规则「编码了关于进展的判断，因此可以整体移除而不破坏终止性」——
   即它自己是可消融的一条臂，我方吸纳时也该给它一条对照臂。

**不吸纳**：AscendC / NPU 域本身（用户口径：昆仑芯屏蔽即可；同类低资源加速器不接入）。
它给 #76① 缺的 `redesign` 那一路的真正提示是**反向**的：它把「结构可用但数值不过」的候选
当成资产去修，而我方今天的 Fixer 本来就只修这一类 —— 我方缺的不是「再生 vs 修」的选择，
而是**「这一版的结构本身值不值得保留」的判据**，那需要一条新的归类，不能凭空写。

---

## 六 · 决策与挂号

| 外部机制 | 决策 | 落点 |
|---|---|---|
| kill witness 进分母，不可杀的隔离 | **吸纳（纪律）** | #83：本仓每张变异表按「有证人 / 结构性」两类分开计数，不再混报 N/N |
| 全零核与带宽比是可测的 | **吸纳（已落探针）** | `benchmarks/probe_l2_sampling_gate.py`；判据改动挂 #80/#82 |
| 「聚合默认值 = 造证据」 | **吸纳（判据）** | #81（`verifier.py:380-384`，被并发会话占着，等让开） |
| 门要测 held-out 性能而非只测正确性 | **改造后吸纳** | #84（held-out 配置轴，需机时）；#77 的分母口径同步 |
| 预算延长要有客观进展当证人 | **吸纳（设计）** | 并进 #71 的前提，不再单独挂项 |
| 代理无接受权 / 完整性门拒自接受的成功 | **依据加固** | #56（已修）/ #51 的宣称口径各多一个源 |
| AscendC 域、Metal suite、`(1+1)` 循环、RLVR 类训练 | **不吸纳** | 理由见 §四/§五 |

**没做的事，写清楚**：本轮**没有改任何生产代码** —— 三条判据类修复都要动 `verifier.py`，
而它正被并发会话占着（`operator_agent/verifier.py`、`schemas.py`、`remote_executor.py`、
`tests/backend_fakes.py` 都在保护名单里）。落下来的是**一只新探针 + 一份可复现的读数表 + 四条挂号**。
真机 L2 有没有任务真落在「参考幅值 < atol」那一侧，**未测** —— 那要 H20 机时和每任务的
参考幅值分布（#82 需要它，#48 重跑时顺手能攒）。
