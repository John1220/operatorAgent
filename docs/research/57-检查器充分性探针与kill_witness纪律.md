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

- `_verify_single_set` 里 `Cap.FULL_OUTPUT_TENSOR in caps` 那路：`effective_atol`（容差表与任务
  `max_error` **取 min**）+ 相对项的 AND 形式；
- **没有**那个 Cap 的那路（同一个函数的另一支）：`max_error <= task.constraints.max_error`，
  纯绝对、**不查表**、**没有相对项**。`LocalExecutor.capabilities` 声明的是**空集**，所以它走这条。

⚠️ 这一节原先写的是 `verifier.py:509-517` / `:498` 这类行号，而 `verifier.py` 是并发会话会改的
文件 —— 行号早就指错地方了（同一个坐标在探针的输出表里也被重播，见 doc 58 §4.2 与 CHECKLOG I30）。
锚点一律用符号名：改名字会响，挪位置不会。

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
3. **通过那一侧的聚合 metrics 会造证据**：旧写法用
   `m.get("max_rel_error", 0.0)` / `m.get("bad_elements", 0)` / `m.get("matched_ratio", 1.0)`
   把纯绝对那条**从没算过**的三个量补成了「无相对误差、零坏点、全部元素匹配」。
   探针臂 0b 直接打出：`atol=— 没这个键  matched_ratio=1.00e+00  max_rel=0.00e+00  bad=0`。
   ⇒ 从 `result.json` 里**读不出这两条路的区别**，而失败那一侧反而不造（它提前 return 原始
   metrics）。缺的这一半恰好只出现在「通过」上 —— 也就是通过率的分母上。
   **【#81 已于 2026-09-30 落码，见 §八；同一臂的改后读数是
   `atol=— 没这个键 matched_ratio=— 没这个键 max_rel=— 没这个键 bad=— 没这个键`】**

**我自己在这只探针上交的两次学费**（都是本仓记过的族的又一次落地）：

- 第一版打表用 `.get("max_rel_error", 0.0)` / `.get("matched_ratio", 1.0)`。这不只是「探针也会
  兜底」—— 它让同一个 `matched=1.00e+00` 变得**无法归因**：是被测代码真算的、是 `_verify_function`
  兜出来的、还是探针自己兜的？我第一版就把它读成「local 那条路报告相对误差为 0」，差点把账记错一层。
  改成缺键印 `— 没这个键` 之后读数是 `atol=— 没这个键 matched_ratio=1.00e+00 bad=0`，
  这才分清：`matched_ratio` 确实在**聚合结果**里（被 `verifier` 的默认值造出来的），`atol` 从不存在
  （CHECKLOG I21：**量具打表不许给缺键填默认值**，填了就把三层压成一层）。
- 第一版拿 `coverage` 当分支判别键，于是 MockGPU 一半的行被印成「纯绝对那条」。
  `coverage` 两条路都上报（`_verify_single_set` 两个分支各写各的，AND 那条在参考干净时恒 1.0）；
  真正只有 AND 那条会算的
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
| kill witness 进分母，不可杀的隔离 | **吸纳（纪律）→ 已落，见 §七** | #83：`benchmarks/mutation_calibrate.py` + `benchmarks/mut_specs/fixer_stage_branch.py`，本仓那张表现在报两个分母 |
| 全零核与带宽比是可测的 | **吸纳（已落探针）** | `benchmarks/probe_l2_sampling_gate.py`；判据改动挂 #80/#82 |
| 「聚合默认值 = 造证据」 | **吸纳（判据）→ 已落，见 §八** | #81：`measurement.aggregate_l2_readings` + `_verify_function` 通过侧，11 例 / 11 条变异 11/11 |
| 门要测 held-out 性能而非只测正确性 | **改造后吸纳** | #84（held-out 配置轴，需机时）；#77 的分母口径同步 |
| 预算延长要有客观进展当证人 | **吸纳（设计）** | 并进 #71 的前提，不再单独挂项 |
| 代理无接受权 / 完整性门拒自接受的成功 | **依据加固** | #56（已修）/ #51 的宣称口径各多一个源 |
| AscendC 域、Metal suite、`(1+1)` 循环、RLVR 类训练 | **不吸纳** | 理由见 §四/§五 |

**没做的事，写清楚**：本轮**没有改任何生产代码** —— 三条判据类修复都要动 `verifier.py`，
而它正被并发会话占着（`operator_agent/verifier.py`、`schemas.py`、`remote_executor.py`、
`tests/backend_fakes.py` 都在保护名单里）。落下来的是**一只新探针 + 一份可复现的读数表 + 四条挂号**。
真机 L2 有没有任务真落在「参考幅值 < atol」那一侧，**未测** —— 那要 H20 机时和每任务的
参考幅值分布（#82 需要它，#48 重跑时顺手能攒）。

---

## 七 · #83 落了（2026-09-30）：那条纪律变成仓里一只可跑的 harness

上一节把 kill witness 记成「吸纳（纪律）」，这一节是它落下来的样子。**零生产代码改动、零机时。**

三个新文件（不写行数：那正是「刚写完就已经过期」的那种数字，本文档 §七 写的这一小节自己
就把它改过一次）：`benchmarks/mutation_calibrate.py` 是 harness，
`benchmarks/mut_specs/fixer_stage_branch.py` 是那 21 行的可执行规格，
`tests/test_mutation_calibration.py` 是标定 harness 自己的 12 条用例：

| 件 | 作用 | 为什么长这样 |
|---|---|---|
| `Row(name, file_key, old, new, witness, behavioral=True)` | 一行变异 = 一处锚点 + 一个证人 | `witness` 必填；`behavioral=False` 的行必须把「为什么不可杀」写进 witness 开头（有一条用例专门盯这个前缀） |
| `check_shape()` | 锚点在目标文件里**恰好出现 1 次** | 不唯一 ⇒ 判 `ANCHOR` 而不是 `KILLED`。上次 M10a/M10b 就是「锚点查不到」的账，不能反过来「锚点撞两处」还照跑 |
| `run_suite()` | 跑目标测试、解析汇总行 | **解析不出汇总行就 `raise RuntimeError`** —— 第一版 `/tmp` 脚本正是把「两条路径当成一条传」导致零条测试运行、被读成「全绿」，即 CHECKLOG I15 那一族 |
| `calibrate()` | 逐条：改 → 跑 → **立刻复原并逐字节断言** | 归属靠这条（doc 55 §六 我自己那次错）。红名单与上一行完全相同 ⇒ 打印 `⚠️ 复核归属`，不静默 |
| `summarize()` | 恒印两栏 | 结构行不进行为分母；只报一个数就是这次要修的那个错 |
| `stamp()`（现在住在 `benchmarks/provenance.py`，两份报表共用一个定义）| 报表第一行印 `python/torch/HEAD/未提交改动` | 这台控制机上有两套解释器，`python3` 解析到哪一套取决于当时的 `PATH` —— 不带条件的「N/N」下一轮分不出是**条件差**还是**回归**（见下面那小节；落地过程与它抓到的三处空转见 doc 50 §7.8 / CHECKLOG I25）|

### 先标定量具，再让它报数（这一段是我自己的学费）

harness 自己的 12 条用例分两组：A 组 7 条拿**假仓库**标定它，B 组 5 条核真规格不腐烂。
（这一节初稿写的是 13：A 组 8 条，第 8 条是 `stamp()` 的读数条件断言。`benchmarks/provenance.py`
被两份报表共用之后那条用例搬进了 `tests/test_run_provenance.py`，所以这里改的是**归属**不是删掉 ——
「13 条」现在仍然成立，只是不再全在这个文件里；数一条用例的归属要连着它住在哪个文件一起数，
否则搬一次家就变成一次无人察觉的覆盖流失。）
A 组那份假生产文件是 `pick(x)`（返 `"a"` 或 `"b"`），配两种测试 —— 「盲」的那份只钉常量
（`assertEqual(1, 1)`，所以**用例数不是零、汇总行照样印 `1 passed`**，而这正是最容易被读成
「覆盖了」的形态），「覆盖」的那份钉 `pick` 两侧的取值。
**第一版这份「覆盖」夹具只喂了 `pick(True)`** ⇒ 「把分支条件删掉、恒返 `"a"`」这条变异在这个
夹具上**真的逃逸**，而我的对照用例声称它可杀。同一个形状的已知先例是 CHECKLOG I19（那条「三路
互斥」用例只朝一个方向查缺席，删掉第三路的范围锁时全体成立）；我不写「第 N 次」这种没数过的账，
只写判据：**只在条件成立的那一侧取样，测的是代码在不在，不是条件对不对。**
补成 `"a"/"b"` 两侧都断言之后，同一条变异变成 KILLED（这条纠正写在
`tests/test_mutation_calibration.py:91-98` 那条用例的 docstring 里，不只在本文）。

另两处是给 harness 自己下的规矩，不是给被测代码的：
- **不许用 `isinstance` 核规格里的行类型。** `python3 -m benchmarks.mutation_calibrate` 时
  `benchmarks.mutation_calibrate` 会以 `__main__` 和模块名**各装载一份**，于是 `Row` 是两个不同的类，
  `load_spec` 里那句 isinstance 会把一份完全合法的规格判成「不是 Row」。改成按**字段结构**核
  （五个必填字段 + `behavioral` 是 bool），并在代码里写明原因（`benchmarks/mutation_calibrate.py:72-74`）
  —— 否则下一个人会把它「修回去」。
- **子进程环境继承 `os.environ` 再叠 `PYTHONDONTWRITEBYTECODE=1`（配 `-B`）。** 第一版是手写一份
  `env={"PATH": …, "HOME": …}` —— 那等于把「运行环境」当被测对象重新发明了一遍，读数与手工跑
  不可比；而字节码缓存那条陷阱（单字符变异同尺寸、同一秒还原 ⇒ 仍执行被变异的字节码）必须双堵。

### 读数（`59c6acf` 之后的工作树，2026-09-30 00:38 起算）

```
读数条件: python=3.9.12 @ ~/opt/miniconda3/bin/python3 torch=有 HEAD=00a88ed 未提交改动=有
行为分母 (带 kill witness): 18/18 拦住
结构分母 (今天行为不可区分, 只有结构守卫杀得掉): 3/3 拦住
```
`--check-only` 先过：「锚点与证人核对: 21 行全部唯一且有 witness」。跑完 `git status -uall` 只剩
本轮自己要动的四个文件（一份测试 docstring + 那三个新文件），五个被变异的**生产**文件全部逐字节
复原 ⇒ 归属成立（这条不是推测：`calibrate()` 每行跑完就 `assert` 逐字节相等，断言不过直接抛）。

两条 `⚠️` 归属提示被复核后**判定为合法**，并把结论写进测试文件的 docstring（提示本身留着，
因为它是判据的产出而不是噪声）：
- **M4/M5 红名单逐条相同（9 条）** —— 两行改的是同一个占位符的两种失效形态（删掉 vs 换成死文字），
  读者本来就是同一批断言；
- **M9a/M9b 逐条相同（2 条）** —— M9b 是 M9a 的**超集**（在合并处把两条检索通道一起堵掉），
  两行保留是因为失效点各在一层，不是为了凑数。

判别的办法不是「看像不像」，而是能指出**每行改的是哪个文件哪一行、哪条用例会因此执行到那里**；
指不出来的那才是串了味道的账。

### 这台控制机上有两套 python：读数条件那一行是撞出来的，不是设计出来的

跑完全套去归因「本片新增 13 条用例」时，手上同时有三个读数：**974/1 skipped**（`HEAD~1`，miniconda）、
**986/1 skipped**（`HEAD`，miniconda）、以及本仓文档里一路写下来的 **969 passed / 1 failed /
5 skipped** 那种「带一条红」的形状。第一反应是「并发会话把 torch 那一族的红修了」—— 把两套
解释器各跑一遍才看清：

| 解释器 | torch | 同一份 HEAD `00a88ed` 的干净 worktree |
|---|---|---|
| `~/opt/miniconda3/bin/python3` (3.9.12) | 有 | **986 passed / 1 skipped**（总 987） |
| `/usr/bin/python3` (3.9.6) | 无 | **981 passed / 1 failed / 5 skipped**（总 987） |

两边**总数都是 987**，差的 5 条全是 CHECKLOG I5 那一族（用例判定随本机装不装 torch 翻转），
没有一行代码不同 —— 所以这不是修复，是**换了跑它的解释器**。而 Bash 工具每次调用都是新 shell，
`python3` 解析到哪一套取决于当时的 `PATH`，**同一会话内的两次「本机读数」可以来自两套解释器**。
这张 21 行表的红名单我也在两套下各跑一遍：**逐行相同**（18/18 + 3/3，每行条数一致），
所以这条不是「数会变」的证据，是「条件要跟着数一起印」的证据 —— 只印一个数，下一轮就做不出
「条件差还是回归」这个判断（doc 20 那条计数行早就写过「读数条件要跟数字同处」，这次的新形态是
**同一台机器上也有两套解释器**，而同一个 `python3` 名字解析到哪一套取决于当时的 `PATH`）。

落码：harness 报表第一行恒印 `读数条件: python=… @ … torch=… HEAD=… 未提交改动=…`（实现现在住在 `benchmarks/provenance.py::stamp`，变异 harness 与真机驱动共用同一个定义 —— 一句话留两个副本就会漂），并配一条反空转用例 —— **`HEAD=?` 不许悄悄印过去**（我第一版把
`git rev-parse` 写成了 `git -rev-parse`，取不到就兜成 `?`，看起来完全像一行正常的 stamp，
正是 I21 那个「兜底值冒充读数」的形状）；另一处自我纠正是：这条 stamp 的注释第一版拿
**986 与 974** 当解释器之差，那其实是**两个不同 commit** 的读数 —— 又一次把并排的两个数
当同源（doc 49 §八 第 9 条那族）。

### 旧账同步改掉（一句描述有多个副本，先枚举再改）

`grep -rn "21/21"` 逐处读下去，**把它当现行宣称**的只有 3 处：doc 55 §六、doc 41 §七、
`docs/research/README.md` 那条 doc 55 索引行；其余命中都是「把它当被纠正的对象」的文字
（本轮新写的 spec / harness / 两份测试的 docstring）。三处都改成「**行为 18/18 + 结构 3/3**，
两个数恒并排印」，并各自保留那句「当天报的是 21/21」的叙述 —— 要改的不是一个数字，
是**那句话宣称的覆盖强度**：「21/21 拦住」听起来像 21 行都有行为证据，而真相是三行今天
谁都杀不掉、只能靠结构断言预警。

### 没做的事

- **只有这一张表迁成了 spec。** `grep -rn "0 逃逸" tests/` 数出来还有 **4 张**表各报一个单分母数：
  `test_cost_persistence.py:29`（6 条）、`test_golden_tier.py:28`（9 条）、
  `test_stop_reason.py:15`（9 条）、`test_calling_convention_broadcast.py:27`（8 条），
  合计 32 行没有 witness 字段。它们里面大概也有「今天行为不可区分」的行 —— 但**迁一次要重跑一次
  全套**，把老数搬进新格式却没重跑正是这次要反对的事，所以留到各自被重跑时再迁。
  （我第一版这里写的是「doc 47 / 53 那几处」—— 那是没数过的猜测，`grep` 一跑就看见名单其实
  由**测试文件**而不是文档定义。）
- **harness 不会自动证明「每条红归到唯一注入」**，它只报「与上一行相同」让人去核。想做成自动的
  得跑单次注入的归因矩阵（N 倍成本），今天不值。
- **结构那三行的隔离理由没有机械退休条件。** 如果哪天它们变成可杀（比如给 `run()` 加一条能造出
  「L3 单独失败还进了 Fixer」的活路径），得**有人**把它们移回行为分母 —— 目前没有任何东西会红。
  这是 §五 那条「算了没人读」的自愿版，登记在此而不是假装解决。

---

## 八 · #81 落了（2026-09-30）：通过侧的聚合只 fold「报过的组」

§三 第 3 条撞出的那条缺陷，这一节是它落下来的样子。**零机时、零生产行为改变** —— 判决仍然由
每组自己决定，这片动的只有「产物与日志里印什么」。

**为什么排在这一步而不是先去改两条路的容差口径（#80）**：#80 改的是判据，会把「同一份核在两个
在册后端上一过一挂」这个已经量到的事实改掉，动它需要机时对照才说得清代价；而 #81 是纯上报侧的
诚实性，与 §六 那条「先补分母再动排序」（doc 49 §八 第 14 条）同一条顺序纪律 —— 先把读数洗诚实，
再去动会改变判决的东西。

三个落点，每个都对应一个已经犯过的错：

1. **规则上收成单点**：`measurement.aggregate_l2_readings(sets)` → `(值, 出处)`。
   逐键只 fold 报了这个键的那些组；一个组都没报的键**不发布**；出处是
   `{键: [报过的组数, 参与聚合的组数]}`。放进 `measurement` 而不是留在 `verifier` 里，是因为
   它与 #87 那族（`single_ruler` / `l2_ruler_reading` / `cross_round_ruler`）读的是同一批键，
   两处各写一遍就是两套口径。
2. **两个出口都要有话**：`metrics` 里少掉不发布的键、`metrics["aggregated_over"]` 带上出处，
   同时 message 里那句 `matched_ratio=…` 改成**只在真有组报过时才念**。理由与 §三 那条探针学费
   互为镜像：只在 metrics 里改而 message 照旧念 1.0，读日志的人拿到的还是同一句假话（#87 片 1 的
   那条 M4 变异就是专门打这个的）。
3. **探针那一臂从「只读不判」升成两侧都判**：臂 0b 原先只在缺证据被发布时喊一句 ❗，修完之后
   那条报警永远不触发，于是它退化成的样子与「这一臂根本没在检查」在屏幕上完全一样。现在它两侧
   都判并打 ✔：纯绝对那条**不许**出现那三个键、且必须带出处；AND 那条**必须**出现、且不许带
   出处（此时那一栏不携带信息）。任何一侧不符 ⇒ 探针退出码 2。

**读数**（同一张表，臂 0b，改前 → 改后）：

```
LocalExecutor (空 Cap 集): atol=— 没这个键  matched_ratio=1.00e+00  max_rel=0.00e+00  bad=0
                 →  atol=— 没这个键  matched_ratio=— 没这个键  max_rel=— 没这个键  bad=— 没这个键
                    aggregated_over={'max_error': [1, 1], 'max_rel_error': [0, 1], 'matched_ratio': [0, 1],
                                     'bad_elements': [0, 1], 'compared_elements': [1, 1], 'coverage': [1, 1]}
MockGPU (AND 那条): 四个数全部照旧, 且**没有** aggregated_over 那一栏
```

**「某一组报了、某一组没报」这一侧不需要构造**：远端载荷是按 `_REMOTE_L2_METRIC_KEYS` 的
`if k in remote_verify` 逐键过滤透传上来的，所以部署在远端的 `run_kernel.py` 少报哪个键，控制端
就只能如实缺着 —— 这一条用例（`test_older_remote_payload_missing_one_key_reads_as_partial`）走的是
真实分支（`Cap.REMOTE_VERIFY` + 发放「远端回什么载荷」），不是 monkeypatch 聚合本身。它与 #87
片 1 用的是同一个现场，两片各核一件事：那半核「各组的尺是否同一把」，这半核「聚合值底下是几组」。

**变异标定 11 条, 11/11 全拦、无逃逸**（表在 `tests/test_l2_aggregate_evidence.py` 文件头）。
两条是刻意配的**反向变异**（A6「note 总是报警」、V2「出处总是发布」），能杀掉它们的只有那两条
正向对照 —— 又一次落地 I27「用例也要有能杀掉它的那个变异」。

### 没做的事（登记而不是假装解决）

- **`aggregated_over` 目前只有 message 一个读者**：它没穿过落盘边界 —— `result.json` 里既没有
  `verification_results` 的 metrics，血缘那一份也只从 L2 取尺的三个键
  （`measurement.l2_ruler_reading`）。给它接第二个读者与 #87 剩下的「`ref_oracle` 没穿过落盘
  边界」是同一片地，两处一起接才不必动两次血缘形状。这与 §五 那条「算了没人读」同族，只是这次
  是自愿留的、且写在了键的旁边（`measurement.aggregate_l2_readings` 的 docstring 与用例文件头）。
- **失败那一侧不聚合**（它提前 return 原始 metrics），所以这片没动它；「失败侧的 metrics 是否会
  被下游按聚合口径读」还没核。
- **#80 仍未动**：两条比对路的容差口径不同源（纯绝对那条不查表、没有相对项）这件事今天还在，
  探针臂 1-4 那批读数仍然有效。改它是判据改动，要配机时对照。

## 九 · 检查器自己给错建议（2026-09-30）：「取数失败」被写成「你的表错了」

§一 那张锚点表的可执行形态是 `benchmarks/recheck_arxiv_anchors.py`。它在 #98 那一轮里
**先给了自己一条错建议**，而那条建议如果照做，会毁掉它本来要保护的东西。

**现象**：同一份代码，`--doc` 指到 doc 59 时先后读出过 **36/36** 与 **22/36**。掉下来的那 14 条
不是锚点变了 —— 是表里五个 id 中有一个**瞬时取回失败**（`arXiv` 那头），于是它的表行全部记成
`no_bytes`、引文全部记成 `citation_absent`。退出码是对的（非零），但收尾那句是无条件的：

> `⇒ 上面这些数要改的是**文档里那张表**, 不是这里 (表是唯一真源)`

**为什么这句比沉默更坏**：把网络抖动读成「表错了」的人，会去改一份本来对的表 —— 而且改完
自带依据（「是检查器让我改的」）。这跟 §二 那条「判据方向写反时，量具明明看见了信号却报
『看不见』」是同一族，只是这次错的不是判据而是**处置建议**。

**改法**（`closing_advice(rows, fetch_failed)`，纯函数，两条相反的方向都有用例）：

| 红名单的形状 | 收尾那句 |
|---|---|
| 全是 `ok`/`cited` | 不说任何话（反向对照：以前没有这一条，只在报警侧取样） |
| 有 id 没取回字节 | 「那是**取数失败**，不是表错 ⇒ 先重跑，仍失败记成『未核』」 |
| 字节都取回了而对不上 | 「要改的是文档里那张表」（原句保留，因为那种情况它就是对的） |

**标定**：`benchmarks/mut_specs/recheck_advice_triage.py`，5 条变异 **5/5 拦住**。
其中 **R5（`except` 里不记 id）第一次也是逃逸的** —— 我原先断言的是「输出里出现这个 arXiv id」，
而每个失败 id 都会单独印一行 `❌ 取 <id> 失败`，于是那行替断言满足了条件。改成断言**位置**
（`没取回字节**（2609.` 那个短语后面紧跟 id）才杀掉它。⇒ 这是 **CHECKLOG I32** 在同一次会话里
**第二次**以同一机制出现（第一次是 #98 的 H3），所以 I32 的措辞按「会重复犯的」而不是
「一次疏忽」来写。

**没做的事**：这一轮仍然没有把 `recheck` 接进任何 CI/定时任务 —— 它今天仍然只有「我手动跑一次」
这一个触发点，而它给的那句建议也就只在我记得重跑时才起作用。#95/#48 那批机时项里可以顺手
把「每次调研落档后跑一次并把读数贴进文档」写成固定动作，但那要先有读者。

---

## 十 · I32 的第 3、4 次（2026-09-30）：这一次不是标定撞出来的，是反向对照撞出来的

`#94 第 2 片`（`tests/test_reference_uninformative_route.py` 22 例 +
`benchmarks/mut_specs/reference_uninformative_route.py` 14 行，行为分母 **12/12** + 结构分母 **2/2**）
把 I32 又续了两次，而且这两次各带一条**新形状**，不是同一件事再抄一遍：

**第 3 次 · needle 出现在「每次都会印」的那行账里 ⇒ 断言永真。**
那条断言写的是 `assertIn("参考侧判不出", log)`，而被测代码结尾那行 `[FailureDB] … ·参考侧判不出 K 轮 (没入库)`
**每次运行都印这四个字**，印的还是 `K=0`。前两次（H3、R5）都是「变异跑完 0 红」才看见的，
这一次是**配套的反向对照第一跑就红** —— 我给数值那一侧写的是 `assertNotIn("参考侧没有沉淀…")`
形态的那半句，于是同一次运行里两条断言互相拆穿了对方。⇒ 规矩要从「断言要打在位置上」升级成
**「每条在场断言都配一条反向对照，并且两条要各打各的位置」**：配了反向对照的永真断言活不过第一次跑，
而不配的那条要等一轮变异标定。

**第 4 次 · 标定报的粒度是测试方法名，不是断言。**
M6/M7（前言抑制 / 处方抑制）与 M12/M13（排除集合 / 桶名改名）各自的红名单**逐条相同**，
因为两对断言各挤在同一个方法里。把每对拆成两个方法之后，M6 从 1 条红变 2 条、M7 仍 1 条，
归属才从读数里看得出来。⇒ 这不是「红名单相同」需要解释，而是**用例的粒度要与变异的粒度对齐**：
一行变异对应一条它独有的红，才谈得上 kill witness。

两条结构行按 #83 那格分母分开的老规矩各写了隔离理由：M13（只改桶名的值）今天行为不可区分，
杀它的是把 `L2|reference_uninformative|-` **整串**钉成期望值那条 —— 桶名会被写进已入库记录 id
（`cx-{task_id}:{signature}`），这与 `fixer_stage_branch` 的 M21 同族（CHECKLOG I19/I20 那两次
「只数产者、没数读者」）；M14（写库侧换成自己正则看文案）也是同一天花板，拦住它的是那条 AST 守卫。

**顺带一条流程规矩**：本片刻意动了既有那张 21 行表的靶子（`stage_directive` 加第四路 + 那条
「取向互斥」守卫改名并把表从三条扩到四条），所以**整张旧表重跑**，仍是 行为 18/18 + 结构 3/3；
它报的两处「红名单与上一行相同」（M4/M5、M9a/M9b）都是那张表文件头里早就记过的两条，不是新增。
改了别人的靶子却只跑自己的表，就等于把别人的覆盖悄悄改窄而没人复核 —— 这与 #83 那条
「代码动了 ≠ 覆盖变宽」是同一件事的反方向。
