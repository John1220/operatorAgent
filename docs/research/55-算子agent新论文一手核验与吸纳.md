# 55 · 五篇算子 agent 新论文：一手核验与本轮吸纳

- **任务**：#75（调研 2026-08/09 的算子开发 agent 新论文并吸纳）
- **日期**：2026-09-29
- **一手材料**：五篇 arXiv HTML 版（前两批三篇 + 本会话后补两篇），原始字节与抽取行数都记在下面；
  本文所有引文给**行号**，可按 §1.2 那份脚本重跑复核
- **本轮落码**：修复取向按报失败的阶段分支（§六），外部依据是 CUDA-Harness §4.3.1

| 论文 | arXiv | HTML 字节 | 抽取后行数 | 读到哪 |
|---|---|---|---|---|
| CUDA-Harness | `2609.00058v1` [cs.CL] 30 Aug 2026 | 196,908 | 871 | §3–§4 全文 + §5 目录 |
| Fabrica-Bench / Fabrica Agent | `2608.25124v1` [cs.CE] 25 Aug 2026 | 191,484 | 1,078 | II-B / III-B / III-D / V-C / V-D / V-E |
| MaxKernel | `2609.04523v1` [cs.AI] 03 Sep 2026 | 155,158 | 706 | §2 (RAG / 子 agent / 路径管理) + Table 2 讨论 |
| AMDKernelVault | `2609.12471v1` [cs.CL] 11 Sep 2026 | 477,623 | 1,621 | Abstract / §1 / §2 / §3 语料口径 / §4 reward / §7 / Appendix B 开头 |
| SWE-Serve | `2609.26777v1` [cs.AI] 22 Sep 2026 | 630,480 | 2,907 | Abstract / §2 任务与完整性控制 / §3.1-3.2 |

---

## 一 · 抽取方式决定我能声称什么（本轮先纠正自己的工具）

上一轮我把三篇 HTML 抽成了 `.txt`，那份东西**一行都没有**（`wc -l` = 0）：所有换行被并成空格。
我用它做过「逐字命中」检查 —— 那种检查只在**子串**这一层成立，两处损失是静默的：

1. 非断行空格与软连字符在归一时被吃掉，所以「命中」的其实是归一后的文本，不是原文；
2. **给不出行号**，而 §七 与 doc 53 的复核姿势都要行号。

所以本轮重新抽了一遍（块级标签 → 换行，其余标签剥掉），得到上面那张行数表。
「转述里的闭式与节号要自己取原始字节」这条纪律是 doc 53 给 AutoKernel / KernelBand
那批公式做逐字核验时立起来的；本轮给它加一条推论：**取字节的方式也要核**，
否则「我读过原文」这句话的强度低于它听起来的高度。

### 1.1 取原文的姿势（本轮逐条实测，别再写「直连就行」）

| 主机 / 路径 | 直连 | 走本机 HTTP 代理 |
|---|---|---|
| `https://export.arxiv.org/api/query?id_list=…` | ✅ 200（3,292 B） | — |
| `http://export.arxiv.org/…` | 301（**要升到 https 才有正文**） | — |
| `http(s)://arxiv.org/abs/…` | ❌ 15 s 超时（`http=000`） | ✅ 200（42,942 B） |
| `http://arxiv.org/html/2609.00058v1`（本文全文来源） | ❌ 25 s 超时（`http=000`） | ✅ 200（**196,908 B**，与本文表格里那个字节数逐字相符） |

```bash
curl -s  "https://export.arxiv.org/api/query?id_list=2609.00058"      # 元数据, 直连
curl -sL -x "$PROXY" http://arxiv.org/html/2609.00058v1 -o /tmp/x.html  # 全文, 代理
```

上一轮我写下的「纠正」是「arXiv 无需代理、直连可用（301 要 `-L`）」—— 那句把
**一个主机名**（`export.arxiv.org`）的成功读成了**整个站点**的成功，而 `arxiv.org` 的
`/abs` 与 `/html` 两条路径今天直连都不通；那条 `-L` 看到的 301 恰恰是「http 正文取不到、
要换 https」的信号，不是「直连可用」的证据。**写进文档的取数命令必须自己跑过一遍**，
否则它的作用是让下一个人重新踩（CHECKLOG I12：按主机名而不是按站点记可达性）。
代理端口的值按纪律不进仓库，见本机环境。


### 1.2 本文行号指的是**这份抽取**的行号

`/tmp/*.lines.txt` 会随机器消失，所以把抽取姿势连代码一起放在这里 —— 下面这份重跑出来的三份文本
与本文表格里的 871/1078/706 行**逐字节相同**（本轮 `cmp` 验过三篇），否则「按行复核」
只是一句装饰：

```python
# /usr/bin/python3 extract.py <in.html> <out.txt>
import re, sys, html
raw = open(sys.argv[1], encoding="utf-8", errors="replace").read()
raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
raw = re.sub(r"</?(p|div|section|article|h[1-6]|li|ul|ol|table|tr|td|th|figure|"
             r"figcaption|br|math)\b[^>]*>", "\n", raw, flags=re.I)
raw = re.sub(r"<[^>]+>", " ", raw)
raw = html.unescape(raw).replace("\u00a0", " ").replace("\u2011", "-").replace("\u2212", "-")
lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.split("\n")]
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(x for x in lines if x) + "\n")
```

---

## 二 · CUDA-Harness：四条机制 —— 两条我方已有，一条改判潜伏，一条本轮落码

### 2.1 §4.2.1 解耦测试数据合成 —— 后半句才是重点（行 201-204）

> "If test data synthesis shares context with kernel generation, the agent may adapt test
> inputs and reference outputs to the kernel implementation rather than to the intended
> algorithm."

我方核对：`test_inputs` 由任务与 harness 供给（`conductor.py:225` 的入参，不是模型产出），
参考输出走 `verifier.py:1172 _generate_reference` —— 内建表或 `source.reference_source`
外挂（`verifier.py:1096-1108`）。**没有一条路径让写核的上下文去产出输入或参考**。
⇒ 前半句 by construction 已满足。

### 2.2 Eq. 1 与「小幅值输入让 atol 支配」（行 212-232）

> "when the test inputs concentrate on small-magnitude values, the output magnitude |y| also
> tends to be small, causing ε_rtol|y| to diminish and ε_atol to dominate the criterion.
> In such cases, a numerically incorrect kernel may still pass the validation"

我方核对到两件被我上一轮写错的事：

- **判定式早已不是纯绝对**。`operator_agent/tolerance.py` 是容差与判定的单一真源，
  AND 形式（`abs>atol` **且** `rel>rtol` 才算坏点）+ `matched_ratio`，被两条路径共用：
  `verifier.py:507-516`（本地 numpy）与 `remote_scripts/run_kernel.py:577-612`（远端 torch）。
  它自己的 docstring 记着为什么是 AND、以及近零参考上 `rel` 会爆到 2.1e4。
- **「本地是纯绝对、与远程不同源」这句是活的缺陷吗？不是。** 纯绝对那段在
  `verifier.py:481-500` 的**采样比对分支**里，条件是「后端声明了执行能力但没声明
  `FULL_OUTPUT_TENSOR`」。在册的两个后端都进不去：`mock_gpu.py:83` 声明
  `FULL_OUTPUT_TENSOR`（所以走 AND 形式），`remote_executor.py:224` 声明
  `REMOTE_VERIFY`（所以在远端比，根本不走这段）。⇒ **潜伏缺陷**，新后端一进来就活。
  已登记为 #76 的第 ②项，不当现行缺陷修。

真缺口是**输入幅值**：我方对任务给的输入没有任何范围要求，而 AND 形式对
「近零参考 + 绝对误差在 atol 内」是**刻意放行**的（`tolerance.py` 自己写了这个取舍）。
所以「测数据要覆盖宽幅值」这条我方没有 —— 它需要先用已知对/错的核标定，属于机时活，
归 #48 / #65，不在本轮凭空定阈值。

### 2.3 §4.2.2 fail-fast + 工具集（行 236-266）—— 已有，且我们多一层

compile → correctness → performance 的失败即停就是我方 L1/L2/L3；MCP 只是传输外壳，
我们的对等物是 `backend_caps.py` 的能力声明 + `require()`（缺口 5 那几片）。
「工具的具体执行绑定在部署环境上，所以硬件可扩展」（行 240 那句）与
`Cap.SAME_DEVICE_BASELINE` 的动机同句。⇒ 不吸纳新代码。

### 2.4 §4.1.2 calling convention 在编译期广播（行 193-195, 265）

> "the scaffold prohibits the use of dummy inputs and provides helper functions for reading
> the well-prepared test inputs. The test input paths are passed as command-line arguments,
> and the calling convention is emitted at compile time."

我方是反的：入口签名写在 prompt 里由模型遵守（`planner.py:54,84`
`result.setdefault("kernel_signature", "def kernel(A, B)")`，`interface_contract.py:38`
只在模块顶层找 `def kernel(...)`），签名不符时靠**失败信息**回吐（#58 那批处方文案）。
形态差异是真的（我们没有 C++ 编译期可挂钩的 `#pragma message`），但**方向**可借：
把「契约」从"prompt 里的一句话"改成"执行侧实测后回报的事实"。这一条我方今天就能做的是
让 L1 失败信息带上「runner 期望的入口形状」—— 已归 #76 第 ③项，不在本轮空改契约层。

> ⚠️ 上面这句「今天就能做的是…」是**当时没核现状**留下的错，两处都不对：
> ① L1 失败信息带输入名这件事**第 1 片就已经做了**（`check_kernel_interface` 的处方正文里
> 就有一行「本次会传入的输入名: …」，doc 41 §四），照它做等于什么都不做；
> ② 更便宜的一侧不在**失败之后**而在**生成之前**——那边行 194 说的是「agent 写第一版时不知道」，
> 而我方从没有任何时刻把真实输入名告诉过生成方。
> 这一片因此按 ② 落，落点与两条改判的理由见 [doc 41 §七](41-核函数接口契约静态检查.md)；
> 它同时把 §2.4 的判定从「我方是反的」推进到「同向，但时机比它早一轮」。

---

## 三 · Fabrica：最有用的是一条**负结果**

### 3.1 II-B preserve / discard / build（行 196-213）

跨架构翻译的三种义务：preserve（依赖与 I/O 形状）、discard（源架构的专属选择，
"cuda block size means nothing on a fixed PE mesh"）、build（源里根本没有的东西 ——
通信要**凭空造**）。它给的理由（行 208-210）：

> "translation demands working knowledge of both architectures at once, and why a model
> fluent in only one of them produces CSL that looks plausible and does not run."

我方对应物是缺口 3 那三处散落（`hardware_caps.py` / `achievable.py` / `triton_opt_rules.py`）。
这个三分法对我有直接用处：接第二个 GPU 平台时，**「散落」不是同一件事** ——
标称峰值属 preserve、`block=256` 那类属 discard、而 H20 的可达速率曲线属 build
（doc 51 与 #65 正在做的正是它）。#61/#65 的口径按这三档重写一遍，避免把 discard 类
知识当事实搬过去（这正是昆仑芯那轮踩过的形状）。

### 3.2 结构记录到底买什么：它自己测了，答案是「不买通过率」（行 508-518 + 666-669）

> "These records are prompt guidance, not compiler IR: parsing is best-effort and no lowering
> pass constructs CSL, so the feedback ablation in Section V-C measures whether the extra
> structure helps the model rather than crediting it with compiler guarantees."

测出来的数（行 667-669，逐字）：加设计记录把 unguided 保留通过率从 **21/40 变成 20/40**、
guided 的翻译阶段成功率从 **33/40 变成 32/40**。

这条对我们是**防身的**：我方 `optimization_planner.py` 的配方记录、`UnifiedSketch`
那些结构化字段和 `interface_contract.py` 同样是 prompt guidance。若我们把「知识结构化」
的收益写成"提升通过率"，就是他们明确拒绝的那种记账。⇒ 缺口 3 / #51 的宣称口径改成
**可检视性 + 归因**，通过率增益当作待测假设（不是已知收益）。

### 3.3 三路分流（行 519-522）—— 本轮落码只是它的第 1 片

> "The reviewer then routes architectural, implementation, and host/layout failures to
> redesign, code repair, or interface repair, respectively."

我方今天是一条 Fixer 全接（`conductor.py:779-784`：`failed_results = [r for r in
verify_results if not r.passed]`）。本轮把「取向」按阶段分支（§六）是这三路里最省钱的一片；
完整的「redesign / code repair / interface repair」归 #76 第 ①项。

**【2026-09-29 第 2 片：第三路 interface repair 已接上】** 分流键不再是「只看级别」：
`fixer.failure_class(failed_results)` 用**负样本库那一个分类器**给本轮失败归类（入参口径与
`conductor._record_failure` 逐字一致，所以 prompt 里的取向、复发门、检索读的是同一个桶），
归类为 `interface_mismatch` 时走 `_L1_INTERFACE`。这一路的增量不是「改哪一行」—— 那道门自己的
文案（`interface_contract.py:184-197`）已经把缺哪几个、多哪几个和「改 `def kernel` 的参数名」
讲清楚了；它补的是门讲不了的那一半：**本轮的改动范围**（函数体、`BLOCK_*` 取值、已应用配方
一律不动）与不动的理由（这一版的写法还没被验证过一次，整段重写等于用一次已知签名错误换一轮
未知失败）。级别在前还是归类在前是有讲究的：接口契约那一类必然落在 L1 里，只看级别就会把它
并进「就地修一处 API 误用」那条通用取向，而那条恰恰不限制范围。

**【2026-09-30 第 4 路：Fabrica 那三路里没有的那一路】** Fabrica 的 reviewer 假设被审的那一次
运行是**一次观测**；我方 #94 第 1 片之后 L2 会报「本级判不出」（参考自己没有可比元素），那一条
不是候选的症状，而是一次**没有发生的判定**。它排在归类的最前（比 `interface_mismatch` 还前），
拿到的取向是「本轮不改你的核」。所以 `stage_directive` 现在是四路 = Fabrica 三路 + 这一路，
落法与它自己撞出的两处假绿记在 doc 58「§4.2 之第 2 片」。

⚠️ 这一片落地时**自己就是死的**：第一版写成 `signature == "interface_mismatch"`，而
`failure_signature()` 返回三段 `层级|类名|最内层帧`（实测 `"L1|interface_mismatch|-"`），
那条比较永远为假。取向照样非空、模板照样有读者、现有用例照样全绿 —— 是变异标定
（M12/M17）把它撞出来的。doc 49 §八 第 28 条 / CHECKLOG I18。

**还差的两路**：`redesign`（结构性失败退回重画）与「按 bound 类型分流」都未做，仍在 #76①。

> ⚠️ 这句「今天就会发生」**错了两次**，留在这里当反面：① `verifier.verify()` 是
> fail-fast（`:233-249` 每级失败就 `return`），一份 `verify_results` 里最多只有一个未通过
> 级别 ⇒ 同轮不可能并存；② 我随后改口说「L3 会经 `history_cases` 跨轮进来」，而那也没有
> 生产者 —— `conductor.py:763` 的 `l1_l2_failed` 门只放行 L1/L2，兜底入库 `:952-955` 也按
> 级别过滤 ⇒ 今天的代码不产 L3 记录。详见 doc 49 §八 第 26 条 / CHECKLOG I16。

### 3.4 每条改动的接受率（V-E，行 730-733）

> "correctness and cycle gates accept 110/601 proposed edits (18%). Descriptor-offset
> chaining, task simplification, and buffer cleanup are accepted at 27/97, 20/65, and 20/83;
> bulk FMAC is accepted in 12 of 131 proposals."

我方已有 `applied_recipes` / `disabled_recipes`（`conductor.py:378-380`，`:601` 落盘），
所以「哪个配方最终被留下」这一半是有的；缺的是**分母**（被提出次数）与跨运行汇总 ⇒
不能报「命中率」，只能报「留下过哪些」。这正是 #51 里挂着的那半句
「达成率口径待可信分母」。他们的 `12/131` 那种写法是可直接照搬的报表形状。

### 3.5 不吸纳：Eq. 3 的逐轮信用

`r_τ(a_i) = 1 / 0 / −1` 那个式子是给 **post-training 数据打标**用的（III-E），
而我方 `FailureDatabase.recurrence()` 的产出是**下一次 prompt 注入的负样本**。
用途不同 ⇒ 不构成缺陷（上一轮判过，这轮复核 `kernel_db.py:788-801` 仍无 `fixed` 过滤，
仍判不适用）。要改的是别的东西：见 §四的 provenance。

---

## 四 · MaxKernel：两条独立来源把我方的两个既有决定钉死了

1. **检索语料要排除人写好的核**（行 94，逐字）：
   > "we explicitly exclude hand-tuned kernel code from the retrieval corpus; a primary
   > objective of this study is to evaluate the extent to which agents can generalize and
   > discover TPU-specific optimizations without relying on human-engineered baselines."

   与 Fabrica 的 reference-isolation firewall（行 375-376："withholds the validated compute
   CSL and the held-out inputs before anything reaches a model"）是**两条独立来源、同一个方向**。
   ⇒ #57（把检索到的核当起始候选）的前置条件写死：实现时必须给「种子来源」记一个值，
   且能力口径按它分组 —— 否则我们把复用当能力。本轮已把这句写进 #57 的任务描述。

2. **根 agent 规定绝对路径**（行 121）：
   > "the root agent explicitly dictates absolute file paths for all artifacts at
   > initialization. These rigid constraints are injected into the sub-agent prompts and
   > file-system tools"

   与 DSH `defensive-patterns.md:27`（#69 已吸纳的「可预测路径」）同一条。
   ⇒ 同一纪律有外部两源；#69 的落地理由从「一家之言」升级，`portable()` 那条不再单薄。

3. **数值不吸纳**：beam 3 / depth 3 / 2 branches、inner loop「strictly capped at 2 iterations
   per node」（行 173）—— 单位与我们的轮次不同构，照抄就是把别人的常数当我们的判据（doc 53 的纪律）。

4. **一条对我们报表口径不利的真话**（行 239）：Table 2 的几何均值（2.32× / 1.78× vs 人写 2.02×）
   底下，8 个工作负载有 1 个是**人 4.65× vs agent 1.42×**（Ragged Paged Attention）。
   ⇒ 我方 `harness_pass_rate` / golden 层的报表要按**逐 case 分布**给，geomean 只当摘要；
   这条与 #48（重复基率与散布）一起做，不需要新任务。

---

## 五 · 后补的两篇（同一天取到字节，结论照同一条规矩）

上一节这行原本写的是「没取字节 ⇒ 不许作依据」。后来用 §1.1 那条代理路径把两篇 HTML
都取到了（477,623 / 630,480 字节），所以下面才是有资格写的话。

### 5.1 SWE-Serve `2609.26777v1`：一条可照搬的报表形状，和一条我们**已经有了**的控制

它的完整性控制（行 131-132，逐字）：

> "We qualify each task by checking that its hidden tests **reject the unchanged repository**
> (the no-op control) and **accept the oracle solution**, followed by agent-assisted adversarial
> probing and human adjudication."

我方核对：这一对我方在**数值门**上已经有等价物 —— `tests/test_l2_tolerance.py:141`
(`test_scaled_kernel_rejected`，注入 0.97 缩放的错核必须被拦) 与 `:148`
(`test_correct_kernel_accepted`，合法核必须放行)，再加 `:154` 那条「近零参考上 rel 爆到
2.1e4 仍须放行」。所以「门只对回归失败变红」这一层不是缺口。

**真正的吸纳点是他们把这个 idea 做成了一个数字**（行 99-101，逐字）：

> "On 19 tasks with end-to-end coverage, model-serving E2E tests reject roughly one-third of
> patches that pass every other test (**45.9% under the verifier versus 69.4% with E2E tests
> excluded from scoring**)"

也就是**同一批运行、两套记分口径各报一次**（含真机端到端 / 不含），于是「本地过 ≠ 生产过」
那个差距是一个可追踪的数而不是一句故事。我方现状：`harness_pass_rate` 逐任务报
L1/L2/L3，原料在，但**没有「把 L3 从记分里摘掉再报一次通过率」这一栏** ——
这正是我方 doc 33/34/35 一路在讲的那个差，缺的只是把它印成两个数。
⇒ 落成 **#77**。
**【2026-09-30 #77 已落，并拿两份既有真机报表回读出结论】**第二栏「同一批运行、摘掉 L3」进了 `aggregate`/`format_report`，谓词与 #85 的修订起点共用 `measurement.CORRECTNESS_LEVELS` 那**一个**集合（两处各写一遍就是两套口径），10 例 / 9 条变异全拦。**零机时的回读**（复算已落盘的那两份 H20 报表）已经把这条口径用上了：5/6 vs 3/6 那个 headline 差，摘掉 L3 之后是 6/6 vs 5/6 —— 两次净差里只有一次归生成/编译质量，另两次「本地全过只差性能门」的读数是 0.990 与 0.9688 且**全是 `valid`**。读数与推论写在 [doc 48](48-端到端成功率量具.md) §六B 的新那一段，本条只记落点。
顺带它给「为什么 #50 把墙钟门槛修掉」提供了同族的表述（行 297：calibrated performance gates
只在**需要的**任务上用，全基准只有 3 个任务带性能门）—— 性能判据是例外而不是默认。

### 5.2 AMDKernelVault `2609.12471v1`：一条给我们既有纪律的第三方独立佐证，和一个可登记的外部资源

它的语料口径（行 251，逐字）：

> "Since rejection logs were not aggregated uniformly across source pipelines, we report
> retained counts and subset-specific validation **rather than a single corpus-wide failure rate**."

这与 CHECKLOG H4 那条（「入库率分母与拒收名单不同源，把『门没问』印成『门拒了』」）
是同一件事：**不同来源的分母不能折成一个率**。我方是撞出来的，他们是明写的 ⇒
给 H4 补第三个独立来源（前两个是 doc 48 §六B 与 DSH `testing.md:29`）。

其余核对结论（都是「已有 / 形式不对应」，不产生代码）：
- generate–evaluate–reflect + 在 ROCm 下编译 + 跨多组测试输入验数值 + 在 AMD 硬件上测延迟
  （行 163）= 我方 L1/L2(多组输入 `num_input_sets`)/L3 同形，MCP 那层换成 SSH。
- 它的 reward 三项（行 308）里有一句对我们 #71 有用：speedup 项是 **bounded**，
  理由是 "without allowing timing outliers to dominate learning" —— 与 doc 53 那条
  「墙钟不能当预算口径」同一动机（离群读数不该决定去留），但我方是判据侧不是奖励侧，
  所以对应物是已有的离散度门 / 三态判决，不是新代码。
- 「SFT 是 RL 之前必需的初始化」「multi-turn RL +13.1%」属训练配方，我方不做训练 ⇒ 不对应。
- 诚实的一面（行 487）："does not uniformly lead compilation or speed" —— 它自己的 8B
  在 correctness 上第一、编译与速度上并非第一 ⇒ 别把「单一指标最好」读成「全面最好」，
  与我方 #48「按 case 分布而非单一 geomean」同调。
- **可登记的外部资源**（因为项目可能接入非 NVIDIA 平台）：公开的 HIP/Triton
  execution-verified 语料（62,153 + 39,893 条）与 `github.com/AMD-AGI/hip_kernel_llm_lab`。
  只登记不接入 —— 按 §四 那条 provenance 纪律，接外部人写好的核当种子必须先有来源字段。

---

## 六 · 本轮落码：修复取向按「哪一级报的失败」分支

**外部依据**（CUDA-Harness §4.3.1，行 274-277，逐字）：

> "The refinement branches based on which validation stage reports the failure.
> If the generated kernel fails to compile, the agent is steered towards a specific repair
> based on the error details provided by the compilation tool.
> When the kernel fails functional validation, the agent is instructed to retreat to a more
> conservative implementation, prioritizing correctness above all else."

它的理由句（行 271-272）才是可迁移的机制："When optimization is built on a kernel that
looks high-performance but is functionally flawed, subsequent refinement typically remains
trapped in the same faulty pattern."

**我方的先决事实**（读来的，但**读错了一处，已在下面第二次纠正**）：`fixer_repair.md` 在改之前
只有一句被合写的话「以下代码在编译 (L1) 或功能 (L2) 验证中失败」，取向永远是一条「修好它」，
从不许可退让 —— 这半句到今天就还是真的。当时紧跟的那半句「`conductor.py:779` 交给 Fixer 的是
**所有**未通过项，所以 L2 那一轮常同时带着 L3 的『还不够快』」**不成立**：`verify()` 每级失败就
`return`（同轮最多一个未通过级别），而 Fixer 的入口被 `conductor.py:763` 的 `l1_l2_failed` 门
挡着、兜底入库 `:952-955` 也按级别过滤（⇒ 今天没有 L3 记录可检索）。
所以 `_L2_RETREAT` 末句那句「若同时出现性能未达标」今天是**明标的保险**而非现行描述，
两条结构约束各有一条守卫钉住（`tests/test_fixer_stage_branch.py`，含 M11 变异）。
**分支本身不受这个纠正影响**：取向按阶段分是对的，因为原文给的理由（优化叠在数值缺陷上
会困在同一个错误模式）针对的是「本轮谁优先」，而「本轮只有 L2 在报失败」恰恰让这条更必要。

**改了什么**（3 个文件，纯加法，判定路径零改动）：

| 位置 | 改动 |
|---|---|
| `operator_agent/fixer.py` | `_L2_RETREAT` / `_L1_INPLACE` 两段取向 + `stage_directive(levels)`（**只有 L2** 触发退让；L1+L3 不触发） |
| `operator_agent/fixer.py` | `run()` 先算 `levels`/`level_set`，向模板传 `failure_stage` 与 `stage_directive` |
| `operator_agent/prompts/fixer_repair.md` | 合写那句换成 `本轮报失败的阶段: {{failure_stage}}` + `{{stage_directive}}` |
| `tests/test_fixer_stage_branch.py` | 22 条用例 + 21 行变异表（第 1 片 7 条 / 8 行，#76③ 之后 11 条 / 13 行，#76① 第 2 片 +11 例 / +10 行） |

**变异标定：⚠️ 这句原来写的「13/13 拦住、0 逃逸」不可复现，2026-09-29 的一次合并重跑把它改成
21/21，而那个 21/21 在 2026-09-30 又被自己拆开了**（第一次的错：那 13 行里 M10a/M10b 两行的
**锚点在任何文件里都查不到** —— 表只在 doc 里数过、没把「改哪一处」落到能重建脚本的地方，
这正是我自己记过的那族：计数要连着产它的锚点一起写。第二次的错：21 行里有 3 行
（M11/M16/M21）**今天没有任何输入能让它与原文在行为上分家**，拦住它们的是 AST / 契约那类
结构守卫，把它们算进同一个分母等于宣称「21 行都有行为证据」）。
现在的读数：**行为分母 18/18 拦住、结构分母 3/3 拦住**，两个数恒并排印。
表的可执行形态已进仓库 —— 规格 `benchmarks/mut_specs/fixer_stage_branch.py`（每行一条
`Row(名字, 文件, 锚点, 变异, kill witness)`）+ harness `benchmarks/mutation_calibrate.py`，
命令 `python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/fixer_stage_branch.py`；
`tests/test_fixer_stage_branch.py` 的 docstring 只留人读的表，数字由脚本产。详见 doc 57 §六/§七（#83）。
其中三条值得单独点出：

- **M12/M17 抓的是本片自己那只 bug**（分流拿整条签名去比裸类名 ⇒ 第三路恒假），见 §3.3；
- **M20 是重跑前唯一一条真逃逸**：那条「三路互斥」用例只朝一个方向查缺席，删掉第三路自己
  那句范围锁时全体成立。补成在场 + 缺席双向才红 —— CHECKLOG I19；
- **M21 我原本设计成「应当全绿的对照组」，它红了**：桶名已经写进已入库记录的 id
  （`cx-{task_id}:{signature}`），改名不是自由重构 ⇒ 「预期全绿的变异」这个设计从这张表里
  删掉了，没有预期全绿的变异，只有没数到的读者 —— CHECKLOG I20。

其中 M8（删掉「本轮**不要**为此
优化」那句）只有 1 条用例红，而那条是**明标的防御性优先级**用例 —— 它钉的是「分支条件不许是
L3」，不是「今天会发生 L2+L3 同框」。上面那句「这与 §六 的先决事实对齐」原本是错的：
§六 那条先决事实后来被**纠正了两次**（同轮被 `verify()` 的 fail-fast 排除；跨轮那条通道也没有
生产者，因为 Fixer 的入口只放行 L1/L2），详见 §3.3 的更正块与 doc 49 §八 第 26 条。
这张表另外撞出一条真缺陷（`search_by_error` 的零相似兜底把不相关历史标成「错误相似度命中」，
#79 / CHECKLOG I17），那是 M9a 第一次「逃逸」的去向 —— 记录在 doc 41 §七。

**没做的事，写清楚**：真机 A/B 没跑（本轮无机时）。落下来的是**机制**，
效果断言为 0。要判它有没有用，姿势是 doc 35 那套：同一批失败、两臂、看
rounds-to-compilable / rounds-to-correct 而不是单次通过率 —— 归 #76。

**我自己这轮的四个错**：

1. **变异脚本的归属串了味道**：第一版只在循环结束后统一复原文件，于是改模板的 M4/M5/M6
   一直留在盘上，后面两条 `fixer.py` 变异的红其实还是 M6 的红。逐轮复原 + 复原后逐字节
   自检之后 M6 从 4 条红变 2 条、M7/M8 不再挂着与它们无关的装配契约用例。
   ⇒ **红要能归到唯一原因，否则「拦住」这个数字本身没有读数条件**
   （doc 49 §八 第 22 条 / CHECKLOG I11）。
2. **引用早于文档存在**（本会话第 3 次）：`fixer.py` 与测试里先写了 `docs/research/55`，
   被 `test_repo_hygiene::test_every_reference_points_at_an_existing_doc` 抓红。守卫是对的，
   我错在把链条当装饰 —— 每次都以为这次不会忘。
3. 上一轮那条「本地 L2 纯绝对是现行缺陷」的结论**没核 reachability**（§2.2）。
4. 把「一个主机名直连可用」写成了「整个站点直连可用」（§1.1）：本轮把文档里那条
   `curl` 真跑了一遍，`arxiv.org/abs` 与 `/html` 两条路径直连都超时，全文其实一直是
   经本机代理取的 —— 196,908 那个字节数只有走代理才复现得出。写成表格后同一件事
   就有了可核对的形状（CHECKLOG I12）。

---

## 七 · 状态

| 项 | 结论 |
|---|---|
| CUDA-Harness §4.1/§4.2/§4.3 | 4 条里 2 条已有、1 条潜伏（#76②）、1 条落码（§六）；**§4.1.2 那第 3 条已落第 1 片**（#76③ → doc 41 §七，生成前广播） |
| Fabrica II-B / III-D / V-C / V-D / V-E | preserve/discard/build 进 #61/#65 的口径；「记录不买通过率」进 #51 的宣称口径；三路分流 **V-C 已落第 2 片**（interface repair 接上，`ce11d4e`），剩 redesign 一路在 #76①；acceptance 分母进 #51 |
| MaxKernel | provenance 进 #57 前置；路径纪律升级 #69 的依据；数值不吸纳 |
| 不吸纳（含 SWE-Serve 的训练/基准构造与我们不对应的部分、AMDKernelVault 的 SFT+RL 配方） | FABRICA Eq. 3（用途不同）、RLVR Eq. 4（无可更新参数，我方是 prompt 回路）、MCP 外壳、bandit/beam 常数 |
| 未验 | §六 的真机效果；#76③ 第 1 片（doc 41 §七）的真机效果 —— 对照基线已有：**16 轮里契约门拦下 4 轮**（doc 48 §六B），广播能不能把这 4 轮压下去由 #48 重跑回答；#76① 第 2 片（第三路取向）**效果断言为 0**，它改的是那 4 轮之后 Fixer 拿到的一句话，同一批 4 轮就是它的对照数 |
| 后补 | 两篇取到字节后的结论（§五）：SWE-Serve ⇒ #77（双记分口径），AMDKernelVault ⇒ CHECKLOG H4 第三源 |
| 下一步（本文这**五篇**都已一手核过，下一轮不要重新调研） | #76 的三项（#77 已落）；#48/#65/#71/#72/#74 的机时活 |
