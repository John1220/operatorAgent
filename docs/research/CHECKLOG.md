# CHECKLOG — 「文档/注释说的」与「实测/代码做的」矛盾台账

> 形制借自 LeetCUDA `kernels/interview/book/CHECKLOG.md`：分类 + 证据 + 状态，
> **冻结源不回写**（原文留着，错在这里挂号）。
>
> 为什么要这份：本仓已被「论文说 vs 仓库做」坑过至少四次（Ecdysis / AdaExplore /
> OpenEvolve / Sakana），也被自己的过期注释坑过（util 窗口、判据方向、量具键名）。
> 正样本语料会把这些错误陈述**学进去**——台账是给 Designer/Coder/Search/Fixer
> 查的负知识，不是给人看的 changelog。
>
> 立条纪律（缺一不收）：
> 1. **原话**（或极短摘录）+ 出处（文件:行 / URL / commit）
> 2. **实证**（我们测到的、读到的源码、公式推导）+ 出处
> 3. **后果**（不改会怎样；已修的写修在哪）
> 4. **类别**：`C` claim 宣传/论文 · `I` implementation 代码与文档不符 ·
>    `M` math 公式/口径 · `P` performance 性能数字 · `S` stale 过期注释 ·
>    `H` harness 量具自己

---

## C — 宣传/论文 vs 实际行为

### C1 · Ecdysis「优先复用复发失败」只是 prompt 形容词
- **原话**：复发案例应被优先注入 / 优先检索（论文叙述）
- **实证**：`evolution.py:168-192` 分组键刻意不含任务；「优先」无 `>=2` 过滤、无阈值、无消融
- **出处**：doc 45 §外部一手；doc 20 L115
- **后果**：照抄会把「优先」写成排序键，而排序键在原系统里不存在
- **类别**：C · **状态**：⚠️ 已知，设计时按「负证据」处理（我们的复发门自己写了阈值）

### C2 · AdaExplore 经验注入出厂是静默 no-op
- **原话**：跨任务经验注入（memory）
- **实证**：出厂 `memory_update: false`，且默认记忆文件不在仓库树；干净 checkout 上整条链路不跑
- **出处**：doc 45 L54-57；doc 20 L115-116
- **后果**：「有记忆」≠「记忆在用」；引用其机制前必须先核开关
- **类别**：C/I · **状态**：⚠️ 已知

### C3 · OpenEvolve embedding 去重出厂关闭
- **原话**：`_is_novel` 用 embedding 余弦 + LLM judge 去重
- **实证**：`embedding_model=None` 时函数第一行 `return True`，过滤静默 no-op；且去重发生在 `add()`，评估早已付掉
- **出处**：doc 47 L60-61；doc 43
- **后果**：宣传中的「执行前跳过」实际不存在；只影响入库不影响评估成本
- **类别**：C/I · **状态**：⚠️ 已知

### C4 · Sakana AI CUDA Engineer 加速比不可当检索语料
- **原话**：归档 kernel 的加速比可作示范
- **实证**：已被独立复核证伪（见 doc 13 §三）
- **出处**：doc 13 L61-67
- **后果**：不得进 KernelDatabase 正样本
- **类别**：C/P · **状态**：🚫 禁用

### C5 · doc 11 把 `NVlabs/SOLAR` 当成 SOL-ExecBench 的官方实现
- **原话**：`11-kernel生成系统全景.md` 两处 —— L19 可信度列「方法论可信（NVIDIA 官方 + NVlabs/SOLAR）」、
  L93 出处列「2603.19173 §4.3 · NVlabs/SOLAR」（原话已由本人**就地标注更正**，理由见下）
- **实证**：`api.github.com/repos/NVlabs/SOLAR` 真实描述为
  **"Speed of Light Analysis for ML Model Runtime"**，是 einsum/timeloop 方向的解析建模；
  SOL-ExecBench 的正主是 `NVIDIA/SOL-ExecBench`（Apache-2.0，★301，push 2026-07-15）。✅ 本人本轮核到
- **出处**：doc 17 §六
- **后果**：照 doc 11 的指针去找「SOL Score 的实现」会找到一棵建模树、拿不到计时/评测口径；
  我方吸纳 SOL Score（doc 11 §四 首条）的出处因此是错的
- **对「冻结源不回写」的一次偏离**：本条纪律的原意是**保留外部宣称的原文**以便审计。
  这里改的是**自家文档里的一个错 URL** —— 它不是一条值得冻结的宣称，是一个断掉的指针。
  故采取「原文措辞保留 + 就地标注更正 + 此处挂号」三者并存，不是静默覆盖。用户 2026-09-29 已认可该改法。
- **类别**：C/I · **状态**：✅ 已更正（就地标注 + 本条挂号）

---

## I — 实现与文档/注释不符（本仓）

### I1 · `failure_id` 注释写「固定 id」，代码拼 uuid
- **原话**：注释声称稳定/固定 id
- **实证**：实现 `uuid4()`；同文件两处语义冲突
- **出处**：doc 45（读代码抓出的 3 个既有缺陷之一）
- **后果**：跨运行对不上同一条记录
- **类别**：I · **状态**：✅ 已修（签名改为内容哈希口径）

### I2 · 层级拼接顺序不定导致 failure_signature 碎成两桶
- **原话**：签名 = `层级|失败类名|最内层栈帧`（应与任务无关）
- **实证**：拼接顺序依赖 set/dict 遍历 ⇒ 同一错误落进两个桶（doc 45 读代码抓出）
- **出处**：doc 45
- **后果**：复发门数不出「同一错误」
- **类别**：I · **状态**：✅ 已修（顺序写死）（顺序写死）

### I3 · 远端 L1 编译错误没有异常类名
- **原话**：同文件其余错误都是 `{type(e).__name__}: {e}`
- **实证**：编译路径写 `f"Compile error: {e}"`；真机 3/3 落 `unknown` 签名
- **出处**：doc 45
- **后果**：负样本签名无信息
- **类别**：I · **状态**：✅ 已修

### I4 · 文本子串判据把「名字还在」当成「还在用」
- **原话**：`tests/test_shared_host_scratch_paths.py` 一条守卫写作
  `assertIn("stop_interferer", text)`，注释说它盯住「驱动不许退回裸 terminate()」
- **实证**：变异标定 M12 把调用整段删成 `interferer.terminate()` 后 **15 条用例全绿** ——
  那行 `from _scratch import … stop_interferer …` 本身就带着这个词，文本判据恒真
- **出处**：doc 52 §三（16 条变异表、首轮 1 条逃逸）；`_called_names()` 是修法
- **后果**：这类守卫删掉生产调用后不响，且**看起来正在保护**——比没有守卫更糟。
  通则：**清点「有没有在用」必须走 AST 的调用位，不能走文本**；
  同理注释里提一句 `chmod 600` 不算做过（M15），helper 还在 import 但绕开它自己干也不算
  （M16）
- **类别**：I · **状态**：✅ 已修（改 AST 判据后 M12/M15/M16 三条都响）

### I5 · 一条已提交用例的判定随控制机装不装 torch 翻转
- **原话**：`test_absent_external_reference_still_uses_builtin` 断言没有外挂参考时
  参考侧是 `pytorch_eager`
- **实证**：`operator_agent/verifier.py:1099` 在 `import torch` 成功时返回 `pytorch_eager`，
  `:1102` 的 except 分支返回 `numpy_reference`；这台控制机
  `importlib.util.find_spec('torch') is None` ⇒ 本机**稳定红**，装了 torch 的机器稳定绿
- **出处**：doc 52 §五；doc 49 §八 第 20 条；commit `ac2f497`
- **后果**：两层。① 这条用例绿不绿说明的是这台机器的依赖表，不是代码对不对 ——
  与「非确定性期望值不许写死」同族，新形态是依赖 availability 而非随机数，
  所以它**不会偶发**，会在一类机器上恒红、另一类上恒绿；
  ② 它照出的生产语义更该担心：L2 的参考在没有 torch 的机器上**静默换成 NumPy**
  （没有 TF32/fp16 那条路径），「L2 通过」的对比对象跟着控制机变 ——
  缺口 5 那四道门全在管后端与分母，**参考侧没有门**
- **类别**：I/S · **状态**：⚠️ 已知，未修（`verifier.py` 等是并发会话的在途文件）
---
### I6 · 「没人读」是按手里那个变量名数出来的
- **原话**：doc 53 第一版与提交说明写「全仓 `cost_summary` 只有两行 ⇒ 读者数 0，成本算了没人读」
- **实证**：按**接口上的键名** `llm_cost` 重数，读者有两处 —— `harness_pass_rate.py:176` 的
  `cost_from_metadata()` 与 `golden_regression.py:357`（后者把两列抄进
  `tests/golden_registry.json`）。真实缺陷是「每次运行的 `result.json` 里没有成本」，
  缺的是**跨运行**那一半（`artifact_store.py:71-74`：`metadata` 整个不进 `result.json`）
- **出处**：doc 53 §五.1（已改口）；落码 commit `aaa2632`
- **后果**：两种说法要修的东西不同 —— 「没人读」会让人去动那两个本来就工作的读者，
  而正确的动作只是补一个顶层键。**夸大的缺陷描述比没有描述更贵**
- **类别**：I · **状态**：✅ 已改口（同一会话自查发现；通则写进 `tests/test_cost_persistence.py` 的 docstring）


### I7 · 「成本进了 result.json」是真，「本次运行的成本」当时是假的
- **原话**：#70 的提交说明与 doc 53 写「每次运行的成本进了 `result.json`」
- **实证**：#70 接的是 `cost_tracker.summary()`，而账本挂在 client 上**累加**、`reset()`
  全仓无人调 ⇒ 一个 client 跑第二个任务时落盘的是累计值。同一失效已由
  `harness_pass_rate.run_one` 的 docstring 量化过一次：六次运行的 token 排成
  15638→…→166345 的累计曲线被当独立样本求和，总量高估 **3.25 倍**（540114 vs 166345）
- **出处**：修在 commit `2de05dc`（`summary_since(watermark())` + `scope="this_run"` +
  `process_total_calls`）；用例 `tests/test_iteration_lineage.py::TestCostScopeIsPerRun`
- **后果**：按累计值判停 = 第二个任务一开局就背着第一个任务的账单被提前判穷 ——
  而 #71 的第四出口正打算按这个数判事
- **类别**：I/H · **状态**：✅ 已修（6 条变异 0 逃逸）
- **通则**：**值过了落盘边界 ≠ 口径是对的**。补「谁在读这个值」时，同一轮就要问
  「它统计的是哪一段时间」；否则接好的读者会安静地把累计当单次用。

### I8 · 注释拿「没验证的外挂」当理由，顺带藏了一条不可区分的用例
- **原话**：我在 `artifact_store.py:91-92` 写「`metadata` 里还放着『执行快照路径』这类
  含真实路径的外挂」，用它作为「只搬白名单键」的理由
- **实证**：清点 `artifact.metadata` 的写者，`operator_agent/` 下只有两处 ——
  `conductor.py:219`（`stop`）与 `conductor.py:982`（`llm_cost`），两个都是登记键；
  执行快照是**另一个文件** `execution_snapshot.json`（`conductor.py:7`），从来不在
  `metadata` 里。那句话是把 `schemas.py:572` 的一句注释（「存放…执行快照路径 等外挂」）
  当成了实测。同一处还让变异标定 M6（整包倒 `metadata`）打不红
  `test_an_artifact_without_a_stop_block_writes_no_key`：那次运行的 `metadata` 只有
  `stop` 一个键，pop 之后整包是空的 ⇒ 两种写法**不可区分**
- **出处**：注释与 `tests/test_cost_persistence.py:122` 的 docstring 已改口（门不动）；
  标定改用例 `test_the_rest_of_metadata_is_not_dumped` 作 M6 的读者，并补 M8/M9 两条
  变异 ⇒ 9 条 0 逃逸
- **后果**：举一个不存在的例子当理由，下一个读代码的人会在 `metadata` 里找那个键，
  找不到就判定白名单是 cargo cult 而拆掉它 —— 而那道门守的恰恰是**将来**写进
  `metadata` 的东西。另一半：一条变异打不红某条用例时，先怀疑 fixture 里那个键是不是
  只有一个（不可区分），再怀疑用例是死守卫 —— 顺序反了会误删有用的断言
- **类别**：I · **状态**：✅ 注释已改口（同一会话变异标定撞出，代码的门与判据不动）

## M — 公式 / 口径

### M1 · Roofline 峰值误用 sparse 规格或库实测值当 dense 上界
- **原话**：NVIDIA 页大号 TFLOPS；某次 cuBLAS 8192³ 读数当 peak
- **实证**：大号字是 2:4 sparse（差整 2 倍）；库实测是「已达成成绩」不是上界，
  造成「达成率 101.5%」；改 dense 后 matmul 4096³ 101.5%→94.7%
- **出处**：doc 42；`verifier._PEAK_BY_NAME` 注释
- **后果**：D4 收敛出口提前判停
- **类别**：M/P · **状态**：✅ 已修（只收 dense + 出处随日志打印）

### M2 · Arithmetic intensity 用调用方 numpy 的 itemsize
- **原话**：AI = FLOPs / Bytes，字节按「调用方张量」的 dtype 算
- **实证**：fp16 任务按 fp32 算字节 ⇒ AI 偏小 2 倍 ⇒ roofline 误判 memory-bound
  （doc 48 §6.5；改在 `Precision.itemsize`，不再开第四张硬件表）
- **出处**：doc 48 §6.5
- **后果**：绑定轴选错、配方选错
- **类别**：M · **状态**：✅ 已修

### M3 · Triton tutorial 09「warp specialize 只在 Blackwell」过时
- **原话**：`FIXME: This only works on Blackwell right now`
- **实证**：Hopper 专用 pass `add_hopper_warpspec` 自 Triton v3.4.0 起可用
  （PR #7136, 2025-06-23）；`capability//10 in [8,9]` 走 Hopper 分支
- **出处**：doc 15 L83
- **后果**：不敢在 H20 上用 autoWS
- **类别**：S · **状态**：✅ 已改配方表述

---

## P — 性能数字

### P1 · 效用比「训练 30% fewer steps」引的是 CodeAct 论文不是自测
- **原话**：smolagents 宣称相对基线少 30% 步数
- **实证**：引文指向 CodeAct (arXiv 2402.01030) 的数字，不是 smolagents 自测
  （doc 16 §七清单；上游 README 表述）
- **出处**：doc 16 §七
- **后果**：不能当该框架的实测收益引用
- **类别**：C/P · **状态**：⚠️ 不引用

### P2 · 同核同参三次运行的采样次数不同 (2171/2021/3084)
- **原话**：固定 `n_repeat` 可复现
- **实证**：do_bench 式自适应计数，样本数本身是随机变量（doc 46 实测三连跑）
- **出处**：doc 46
- **后果**：拿样本数当常数的统计前提不成立
- **类别**：P/H · **状态**：⚠️ 已知，块序列按块分析

---

## H — 量具自己

### H1 · 判据方向写反：量具看见 ×186 却报「看不见」
- **原话**：污染时长占 20%、幅度 ×186 的人造抢占，量具应报「看不见」
- **实证**：比值是 基线/候选，候选变慢该变小；第一版算 `max(dirty)/median(clean)`
  ⇒ 方向反了，信号被抹平（doc 46）
- **出处**：doc 46
- **后果**：**会自我批评的错误判据比缺失更难发现**
- **类别**：H · **状态**：✅ 已修 + 方向检查进单测

### H2 · 量具键白名单丢掉血缘，重复基率永久丢失
- **原话**：跑完就能从血缘离线算候选重复率
- **实证**：`run_one` 只挑 7 键落盘，`candidate_signature` 在落盘前被丢
  （doc 48 §6.4 第 4 条；那一批 12 轮的重复基率永久没了）
- **出处**：doc 48 §6.4 第 4 条
- **后果**：「跑完就能离线算重复率」在写下时是假的
- **类别**：H · **状态**：✅ 已修（取消白名单）

### H3 · 成本键名 `cost_usd` vs 真键 `total_cost_usd` ⇒ 全程 $0.0
- **原话**：量具从 `metadata["llm_cost"]` 读 `cost_usd`
- **实证**：`LLMUsageTracker.summary()` 的真键是 `total_cost_usd`（doc 48 §6.4 第 2 条）；
  540114 tokens 配 $0.0，安静得像「自建端点不要钱」
- **出处**：doc 48 §6.4 第 2 条
- **后果**：成本读数全错
- **类别**：H · **状态**：✅ 已修 + 对着真 `CostTracker().summary()` 的机械守卫

### H4 · 入库率分母与拒收名单不同源，把「门没问」印成「门拒了」
- **原话**：入库率 3/3 = 100%，另有 3 条拒收
- **实证**：`admit_rejections` 取自 `ok_runs`，而 `_save_to_db` 只在 `all_passed` 那一支
  被调 —— 未通过运行根本不会被这道门问，它们的判定是假想事件（doc 48 §六B ①）
- **出处**：doc 48 §六B ①
- **后果**：故事讲反（门其实一次都没拒过）
- **类别**：H · **状态**：✅ 已修（分栏 + 空名单明说）

---

## 使用约定

1. **检索侧**（Designer/Coder/Search/Fixer）注入 prompt 时，只取 `状态=⚠️` 且与当前
   kernel_type/硬件相关的条目；`🚫` 与 `✅` 不进 prompt（已闭环的不是教训是历史）。
2. **新立条**必须四件套齐全；证据栏不写「据某某说」。
3. **不回写冻结源**：论文/上游文档保持原样，错在这里挂号并给出我们自己的口径。
4. 与 `FailureDatabase` 的分工：那边记**我们跑出来的失败**，这边记**别人文档里
   的假话与我们自己的过期注释**。

## 索引

| ID | 一句话 | 类 | 状态 |
|---|---|---|---|
| C1 | Ecdysis 优先复发 = prompt 形容词 | C | ⚠️ |
| C2 | AdaExplore 记忆出厂 no-op | C/I | ⚠️ |
| C3 | OpenEvolve 去重出厂关闭 | C/I | ⚠️ |
| C4 | Sakana 加速比不可用 | C/P | 🚫 |
| C5 | doc 11 的 SOLAR 指错仓 | C/I | ✅ |
| I1 | 固定 id 注释 vs uuid | I | ✅ |
| I2 | 签名拼接顺序不稳 | I | ✅ |
| I3 | 远端 L1 无类名 | I | ✅ |
| I4 | 文本判据把「名字还在」当「还在用」 | I | ✅ |
| I5 | 用例判定随本机有无 torch 翻转 | I/S | ⚠️ |
| I6 | 「没人读」按手里那个变量名数出来 | I | ✅ |
| I7 | 落盘了但口径是累计 | I/H | ✅ |
| I8 | 注释拿没验证的外挂当理由，顺带藏了一条不可区分的用例 | I | ✅ |
| M1 | peak 用 sparse/实测当上界 | M/P | ✅ |
| M2 | AI 用错 itemsize | M | ✅ |
| M3 | Triton WS「仅 Blackwell」过时 | S | ✅ |
| P1 | 30% fewer steps 是引文 | C/P | ⚠️ |
| P2 | 采样次数不是常数 | P/H | ⚠️ |
| H1 | 判据方向写反 | H | ✅ |
| H2 | 量具键白名单丢血缘 | H | ✅ |
| H3 | 成本键名错 | H | ✅ |
| H4 | 入库率分母不同源 | H | ✅ |
