# spine 实现审查 —— `design/autonomous-spine` 8 个提交

| 项 | 值 |
|---|---|
| 审查对象 | 分支 `design/autonomous-spine`，提交范围 `3a5d25c..24d0e53`（spine 本体 8 个提交） |
| 被审代码 | `calibration.py` / `contract.py` / `eligibility.py` / `confirm.py` / `oracle.py` + `conductor.py` `_save_to_db` + `verifier.py` `_assess_measurement` |
| 读数条件 | 2026-09-30，`24d0e53`，净工作树 |
| 测试状态 | `113 passed in 1.46s`（7 个 spine 测试文件） |
| 设计出处 | `docs/design/autonomous-operator-agent-design.md` §3 / §4 / §10.5 / §13 / §14 |
| 审查结论 | **§14 验收未通过** —— 实现是一个库，不是一条链路 |
| 落地状态 | **2026-09-30 已落地**：P0-1 / P0-3 / P1-4 由 spine 修复批次落地（`2d37625` / `4b427a1` / `aa9b5bc`）；P0-2 / P1-6（`57dd970`）与 **P1-5 / P2-10 + P3-11/12/13**（本轮）随后落地，各节末尾有「落地」注。10% 预算的**账本**也已在 2026-10-02 落地（`operator_agent/budget.py`，见 P2-10 节末与 `docs/research/67`；仍未做的是它按 `explore`/`refine` 分相与按 GPU 秒计费）。**2026-10-01 补真机证据**：P1-5 的参照对在 `algorithm-h20` 上首跑（doc 61），当场抓出两个 bug 并修掉。**2026-10-01 入库判据收口**：P2-8 与 P0-1 的同族问题 —— `oracle.admit_to_corpus` **全仓零生产调用者**，而 `_save_to_db` 自己又写了一遍那三道门（两份口径还不同：前者永远要求重测，后者默认不要求）⇒ 同一决定两个实现、用例测的是没人调的那个。已收成一处（判据吃生产路径真正持有的序列化读数），并补 4 条**接线**守卫；同时核实 P2-7 / P3-15 / P3-16 / P3-18 也已修（下表与各节补注） |

## 结论摘要

| 级别 | 条目 | 一句话 |
|---|---|---|
| **P0** | 1 | `oracle.py` / `contract.py` **零生产调用者** → 收口点不在链路上 |
| **P0** | 2 | `evaluate(contract, contract_hash, …)` 比较两侧都由调用方给 → **恒真** |
| **P0** | 3 | `verifier.py` 的 `except Exception: pass` → 守卫出错时**静默放行** |
| **P1** | 4 | `lib_fallback` **全仓无生产者** → 规则 4 恒为 False |
| **P1** | 5 | 资格推导默认值全绿；`heartbeat=None` 被读成心跳通过 |
| **P1** | 6 | `cv_exceeded` 从不与任何 CV 阈值比较 |
| **P2** | 7 | 所有 `ALL_FALSE` 压成 `validity="uninformative"` → 规则 3/6 被合并 → **已按「有理由的合并」处理**（成因进 `reasons`/`rule_inputs`） |
| **P2** | 8 | `admit_to_corpus` 与 `confirm_persist` 对"重测可 DEGRADED"互相矛盾 → **已修**（改委托），且 2026-10-01 收掉了它零调用者这件事 |
| **P2** | 9 | `exceeds_measured_peak` docstring 与代码不一致（潜伏陷阱）→ **已修（2026-10-02）**：常数改为余量比例、oracle 那处字面量收成同一真源 |
| **P2** | 10 | `require_double_confirm` 默认关；打开则**入库率归零** |
| **P3** | 11–18 | 死代码、只存在于测试里的被测函数、`load_dict` 不校验载荷哈希 |

---

## P0

### P0-1 「唯一收口点」不在链路上

**证据**

```bash
$ grep -rn "from .oracle\|oracle\.evaluate\|import oracle" --include=*.py . | grep -v "^./tests/"
（无输出）

$ grep -rn "from .contract\|contract\.freeze\|FrozenContract\|assert_hash" --include=*.py . | grep -v "^./tests/"
./operator_agent/conductor.py:1492:  三道门, 缺一不可 (设计 §10.5-4 / oracle.admit_to_corpus):   ← 只是注释
```

`Oracle.evaluate` 与整个 `contract.py` 没有任何生产调用者。

**后果**

- 契约哈希校验在生产里**从不运行**
- 标定心跳在生产里**从不运行**
- `Observation` / `provenance` 在生产里**从不产生**
- `contract.py` 没有任何落盘 I/O → 设计 §10.5-5「写入只读文件并哈希；评测器每轮校验哈希」**只存在于注释里**

**唯一真实接线**是 `verifier.py:1144` 的 `derive_eligibility` 调用（写 `metrics["eligibility"]`）
与 `conductor.py:1526` 的读取。

**为什么算 P0**：按设计 §14，Stage 1 的八条验收里 #3/#4/#6 都是**链路级**行为
（注入假候选、起后台负载、改契约文件）。现在测的是库级替代品，所以严格说 Stage 1 还没过。
这 8 个提交对生产行为的影响 ≈ 0。

### P0-2 哈希收口是恒真的

`oracle.py:58`

```python
def evaluate(contract: FrozenContract,
             contract_hash: str,          # ← 调用方自己给
             measure: Callable[[], Mapping[str, Any]],
             *, ...)
```

比较的两侧都由调用方提供。测试里就是这么调的（`tests/test_stage1_acceptance.py:266`）：

```python
obs = evaluate(c, c.hash, lambda: {"value": 2.0, "validity": "valid"}, heartbeat=r)
```

设计 §13.4 要的是 `rec.contract_hash != contract.hash` —— **记录携带生成时的哈希**，
与当前契约比对。现在的形状下，"物理收口"的强度完全取决于调用方自觉，
而这恰是收口本应消灭的东西。

**落地（2026-09-30）**：签名侧早在 P0-1 批次就换成了 `record_hash=…`，本轮把**唯一的生产调用方**接上 —— `verifier.py` 不再 `setdefault` 一个从当前契约现算的哈希回喂，而是把载荷**携带**的旧哈希搬进来比对（`contract_hash` 进了 extra_key 白名单，这是它此前根本到不了 oracle 的原因）；载荷没带就显式声明 `fresh=True`，结论记 `hash_check="fresh_unverified"`。三种形态由调用方明说，含糊一律拒评：

| 调用形态 | 结论 | 含义 |
|---|---|---|
| `record_hash="…"` 匹配 | `verified` | 真的校验过 |
| `record_hash="…"` 不符 | `mismatch` + 拒评 | 判据被改写 |
| `fresh=True` | `fresh_unverified` | 新观测，**没得校**（不声称校验过） |
| 两者都不给 / 同时给 | `missing` + 拒评 | 含糊 = 失败关闭 |

顺带补上一个**死读键**：`confirm.py:104` 一直读 `metrics["contract_hash_mismatch"]` 来出「契约被改写，本次评测作废」那条结论，而全仓**没有任何生产写入者** —— 于是它永远不会出现在报告里。现在 `hash_check == "mismatch"` 时由 verifier 写入该键，并有端到端用例钉住「verifier 写的键正是报告在读的键」。

**残留（本轮未动，已单列）**：这条校验真正该守的是**复用**路径 —— `confirm.ConfirmRecord` 不带契约哈希，`admit_to_corpus` 只信 `remeasure.speedup`；而 `artifact.remeasure` 的生产路径无人写入（`schemas.py:583` 已自记）。所以「契约被改写」在**跨轮复用**上仍无守卫，缺的是生产端而不是判据。

### P0-3 守卫会静默消失（fail-open）

`verifier.py:1144-1162`

```python
try:
    from .eligibility import derive_eligibility
    elig = derive_eligibility(...)
    metrics["eligibility"] = elig.as_dict()
except Exception:  # noqa: BLE001 — 资格位是增强, 不挡主流程
    pass
```

链路：推导抛错 → `eligibility` 键缺失 → `conductor.py:1526` 的
`elig = metrics.get("eligibility") or {}` → `if elig and ...` 条件不成立 →
**入库门整条跳过，无痕**。

注释把资格位称作"增强"，但设计里它是**护栏底线**。这段代码的实际效果是：
**守卫出 bug 时守卫消失，并且放行。**

**建议**：fail-closed —— 异常时写入 all-False 的资格位并把异常文案记进 `reasons`，
让"没算出来"和"算出来不合格"在下游同样可见。

---

## P1

### P1-4 `lib_fallback` 全仓没有生产者

```bash
$ grep -rn "lib_fallback" . --include=*.py | grep -v docs/ | grep -v tests/
operator_agent/oracle.py:70      ← docstring
operator_agent/oracle.py:108     ← 读 raw.get()
operator_agent/verifier.py:1150  ← 读 metrics.get()
operator_agent/eligibility.py:107/151  ← 形参与判定
```

**没有任何一处写入。** 规则 4 在生产里恒为 `False`。

这条恰是 `§19.3` 认定"最有价值"的那条（Atrex 实测「正确率被 PyTorch fallback 抬高」）。
本仓反复记录过的**「声明了没人派发」**，这次披着资格位的外衣。

**建议**：给 anticheat / 运行时屏蔽的结果接一个真实生产者；否则**删掉这一位**——
恒 False 的护栏比没有护栏更危险，因为报表上它会显示为"已检查"。

### P1-5 资格推导的默认值是全绿

`eligibility.py:100-113`：无参调用 `derive_eligibility()` 返回 `ALL_TRUE`。
`oracle.py:89-93`：

```python
hb_ok = True
if heartbeat is not None:      # ← 不传心跳 = 尺子健康
    hb_ok = bool(heartbeat.ok)
```

**「没做心跳」被读成「心跳通过」。** 无人值守下 `heartbeat_ok` 应当必填或默认 `False`。

**保留意见**：`real_device=None` 的默认**是对的**（`eligibility.py:131-134` 的理由成立：
缺键 ≠ 检查过且不合格，否则诚实的本地 mock 会被判死）。本文只反对 `heartbeat_ok` 的默认。

**补证（2026-09-30）**：`oracle.py` 那时引了
`from .calibration import HeartbeatResult, heartbeat_due`，而 `heartbeat_due`
**全仓只有这一处 import、零调用**。也就是说「到了该塞参照对的时刻」这道判据，
本该写在收口点里却从未写成；同时 `ruler_broken_window(None, …)` **显式**返回
`False`（`heartbeat_result is None` → 不算坏）。两者叠加 = **一次心跳都没做过的
运行，尺子被当成健康的**。

**落地（2026-09-30）**：按上文那句「先建生产者」的顺序做了，判据没有再动 `oracle`
的默认值，而是让调用方**明说**这一次是否到期：

- **生产者**：`Conductor` 维护「本次运行的真值测量条数」，`_heartbeat_for_next_measurement()`
  在 `heartbeat_due(n, interval)` 成立时**当场**跑一对已知参照（`heartbeat_interval`
  默认 5，非法值构造时就拒）。到位只跑一次（`_heartbeat_ran_for`）—— 否则 L1/L2 挂掉、
  计数不前进的那些轮次会每轮重跑一次心跳。心跳读数走 `calibration.backend_bench_fn`
  → `backend.benchmark`，即**远端那条路**（部署 → runner → exec → 计时 → JSON 回传）；
  只在控制机 `perf_counter` 上计时的那种 bench 测不出远端尺子坏了，只留给离线单测。
  ⚠️ 但它**不等于**「与候选完全同一条计时通道」，边界见本节末。
- **判据**：`oracle.evaluate` 收 `heartbeat_expected`。到期而 `heartbeat is None` →
  RULER_BROKEN（全资格 False）；没到期且没传 → 不作尺子声明。`ruler_broken_window`
  终于被派发（同一扇窗才牵连，§13.5）。
- **可分辨**：「检查过且好」与「没检查」不共用一个 False —— `Observation.ruler_broken`
  之外，provenance 里另有 `heartbeat_checked` / `heartbeat_expected` / `heartbeat_reasons`
  三格。verifier 据此写 `metrics["ruler_broken"]`/`ruler_reason`（**这两个键此前同样是
  零生产者的死读键**，`confirm.py` 的「该窗口内所有数字不可用」一直在读）、只有真有心跳
  可谈时才写。
- **只有本窗才算数**（§13.5 的两面）：别窗坏了不牵连本次，别窗**好了也不给本次背书** ——
  「检查过」若按「收到过一张心跳」记，递一张陈旧心跳就能把没被测过的窗口洗成「已检查」。
  到期却只递来别窗读数 → 本窗仍按「到期没跑」判坏。生产路径上 `Conductor` 总是传当窗
  那张，这条只在调用方递错时才响。
- **`heartbeat_due` 的死 import 已删**：它现在活在真正需要它的那一层（Conductor 拿得到
  累计计数，oracle 拿不到）。

**真正的边界（三条，都写在代码注释与用例里，不藏着）**：

1. **不是绝对锚点**。默认参照对是纯 Python 小 matmul 与重复 8 遍的慢核，所以它证明的是
   「计时通道还能分辨快慢 + 后端量得出参照物」，没有 `expected_good` 时只做分离度检查。
   真机要连绝对偏置一起抓，得由调用方注入硬件的 `expected_good`（注入点一直留着）。
2. **只覆盖 runner 的 Python 计时分支**（勘误，2026-10-01 读 `run_kernel.py` 时发现）：
   参照核不带 triton/torch import → `_detect_language` 判成 `numpy` → 走
   `_benchmark_torch_numpy` 的 else 分支 → `_timed_runs(..., device_is_cuda=False)`
   → **远端 `perf_counter`**，回传 `"device": "cpu"`。triton/torch 候选走的是 CUDA event
   那条分支。所以它抓得住部署/回传/exec/时钟/顺序坏掉，以及**在 Python 分支上**计时已经
   分不出快慢；**抓不住 CUDA event 分支坏掉**。本轮提交信息与初版注释说的「与候选同一条
   计时通道」是过头话，已就地改掉。要覆盖后者得按任务 DSL 塞一对 triton 参照核 —— **没做**。
3. **不许给它加「自报 `device=cpu` 就判坏」的检查**：参照核本来就该在 CPU 上跑，加了会变成
   每次心跳都误报 RULER_BROKEN（把「本就不该是 GPU 数」读成「出处不对」）。

**真机证据（2026-10-01，doc 61）**：`backend_bench_fn` + `heartbeat` 第一次放到
`algorithm-h20`（GPU 2）上跑。修复后连跑三次 **`ok=True`，慢/好 = 8.09× / 8.07× / 7.87×**
（好核 2.1096 ms、慢核 16.7163 ms，`language='numpy'`、`device='cpu'` —— 边界第 2 条在这台机上
实测对上）。**第一次跑它是判了自己 RULER_BROKEN 的**：当时慢核的遍数是形参、走 `inputs` 传，
而 `SSHExecutor._serialize_inputs` 会把标量也「按 shape+dtype+seed 重新生成」⇒ 远端收到
`repeats=np.int64(0)`，慢核只跑 1 遍，比值 1.0。修法是把遍数**烘进源码**（`SLOW_REPEATS`），
参照对只依赖数组输入；同一条线上标量被顶掉是个**独立缺陷**（`causal=False` 到远端成
`np.True_`，挂号 CHECKLOG I44），已在 `_serialize_inputs`/`_deserialize_inputs` 两侧修掉并
在真机复验。**成本**：一次心跳 ≈ **8.6 s 墙钟**，而两条臂的测量时间合计只有 18.8 ms
（99.8% 是 SSH/uv 往返）；参照对走 CPU 分支，**不占卡**。按默认 `heartbeat_interval=5` 算，
每 5 次真值测量多 2 次往返 = **+40% 往返、+0 GPU 机时**。边界第 2 条要的那对 triton 参照核
也在同一台机上预演过（`matmul @1024³`：0.1057 ms vs 0.8193 ms = **7.75×**，CUDA event 分支），
**结论是可行但还没接进 `calibration`**（要按 `target.dsl` 选对，且参照核得自己 `A.float()`）。

### P1-6 `cv_exceeded` 名不副实

`verifier.py:1151-1152`

```python
cv_exceeded=(assessment.dispersion is not None
             and assessment.validity is not measurement.Validity.VALID),
```

**它从不与任何 CV 阈值比较**，只是把「validity 不是 VALID」重编码了一遍，
还与同一次调用传入的 `validity` 参数重复。

真实来源就在手边：`measurement.py:1164-1174` 已用
`DISPERSION_DEGRADED / DISPERSION_UNINFORMATIVE` 判过，结论在 `assessment.reasons` 里。

后果：任何原因导致的降级都会被额外贴上"离散度超标"标签，`reasons` 被污染。

**落地（2026-09-30）**：判据搬到了**判过阈值的那一处** —— `MeasurementAssessment`
新增 `dispersion_exceeded`（`> DISPERSION_DEGRADED`，含 UNINFORMATIVE 那档；
`None` = 没上报样本，与 `False` = 判过且没超是两件事），verifier 两次重编码合并成
一处 `cv_exceeded = bool(assessment.dispersion_exceeded)`。

两点如实说明：

1. **本轮它的行为是惰性的** —— 规则 6 与规则 7（`tier == degraded`）落同一个位型，
   所以旧写法算出的资格位与新写法一致。修的是**名字与输入**：`cv_exceeded=True`
   现在真的意味着「离散度越过了阈值」，而不是「档位不是 VALID」。
2. 正因为惰性，只测资格位是**测不出**这条修复的。所以资格推导的输入被收成
   `rule_inputs` 写进 `Observation.provenance` —— 五个布尔回推不出是哪个输入判的，
   而「上游传了个错的值、下游无从分辨」正是这一族的失效方式。回归用例就钉在
   `rule_inputs.cv_exceeded` 上（换回旧表达式即红，已实测）。

---

## P2

### P2-7 规则 3 与规则 6 被合并

`oracle.py:137`

```python
validity="uninformative" if not elig.can_verdict else validity,
```

所有 `ALL_FALSE` 都压成 `uninformative`，且 `refused` 仍为 `""`。
但设计 §4.3 第二条设计意图**明确要求区分**：

- 降级 = 「数不可信」（规则 6）
- 溯源缺失 = 「数不存在」（规则 3）

现在下游只读 `validity` 分不出这两者，必须回头重新解释 eligibility。
**这正是设计文档警告过的合并。**

**落地（2026-10-01，按「有理由的合并」处理，不是照原样留着）**：档位仍然只表示
「这个数能支撑到什么程度」，规则 1~5 命中时合并成 `uninformative` 是**刻意的** ——
否则会出现 `can_verdict=False` 配 `validity="valid"` 的自相矛盾记录，而只看档位的读者
（`confirm_persist` 那类）会被那个 `valid` 骗过去。**成因没有被丢掉**：`reasons` 里逐条
写着是规则 1/2/3/4/5 里的哪一条，`Observation.provenance.rule_inputs` 留着**判据的原料**
（五个布尔与 `real_device` 等），`rejected`/`refused` 文案也在。于是「降级（数不可信）」
与「溯源缺失（数不存在）」在**记录里**仍然可分，只是不必从档位这一个字段上分 ——
原条目担心的「必须回头重新解释 eligibility」正是现在这条设计。

### P2-8 两次确认的互相矛盾

`admit_to_corpus`（`oracle.py:145-162`）要求**重测那次**也 `can_persist`；
而 `confirm.py:38-39` 明确允许重测是 DEGRADED（"DEGRADED 允许更宽的噪声"）。

但 DEGRADED 的位型是 `can_persist=False`（`eligibility.py:68`）——
所以 `confirm_persist` 那条宽容在 `admit_to_corpus` 路径上**永远走不到**。
两者必有一个是错的。

**落地（2026-10-01）**：两条一起改。

- **矛盾本身**：`admit_to_corpus` 不再自己复述「重测也要 `can_persist`」，而是整条委托
  `confirm_persist`（DEGRADED 走更宽噪声带的规则就在那里）。于是「重测可以 DEGRADED」
  与「入库判据」不再互相打架 —— 只有一处判。
- **更要紧的那件事（本轮核实出来的）**：`admit_to_corpus` 当时**全仓零生产调用者** ——
  `Conductor._save_to_db` 自己又写了一遍三道门，而且两份实现口径不同（判据函数**永远**
  要求重测；内联那份默认**不**要求）。用例测的是判据函数，生产走的是内联那份，所以
  「两次确认」在生产里既不是它测的样子、也没有任何用例钉住。这是 P0-1「判据不在链路上」
  的同族，只是换了个门。**修法**：判据的输入换成生产路径真正持有的东西 —— 序列化的
  `metrics` + `ConfirmRecord`（原先要一个 `Observation` 对象，而 `VerificationResult.metrics`
  只能搬 JSON 化的东西，这正是调用方被迫重写一遍的**原因**）；三道门的顺序、短路与理由
  文本都收在 `oracle.admit_to_corpus` 一处，`_save_to_db` 只留一次调用。
- **4 条接线守卫**（`tests/test_spine_wiring.py::TestTheGateHasExactlyOneImplementation`）：
  判据说拒就必须不入库、判据说行就必须入库（两个方向都验，免得只测「调了」）、
  `requires_remeasure` 必须跟着 `require_double_confirm` 走（机时承诺不许在接线处写死）、
  判据拿到的必须是产物里那份 metrics（不许重新拼一份）。**对旧实现 4/4 变红**（实测）。
- **口径仍然是一个显式参数**：`requires_remeasure` 默认 `True`（设计 §10.5-4 要的是
  「永久结论量两遍」），生产路径把它接到 `require_double_confirm`（默认 `False`，机时承诺）。
  于是**默认配置下单次读数即可入库** —— 这一点现在写在判据的 docstring 里、也写进允许入库
  时的那句理由（「本次**未要求**两次确认……这不是『两次确认已通过』」），免得日后把
  「入库了」读成「两次确认通过了」。

### P2-9 `exceeds_measured_peak` 的 docstring 与代码不一致

`eligibility.py:73-86`

```python
"""`value > 实测峰值 × (1+slack)`? ..."""     # ← 说 (1+slack)
...
return value > measured_peak * slack          # ← 做 ×slack
```

默认 `slack=1.05` 时两者数值巧合相等，所以测试全绿。
第二个调用方若按 docstring 传 `slack=0.05`（意为"5% 余量"），阈值变成 `峰值 × 0.05`
→ **几乎所有读数都触发**。

**附**：该 helper 与 `PEAK_SLACK` 都**没有生产读者**，
`oracle.py:111` 把 `1.05` 字面量又抄了一遍 → 同一阈值两个真源，会漂。

**已修（2026-10-02）**：常数语义改成**余量比例**（`PEAK_SLACK = 0.05`，阈值
`峰值 × (1 + PEAK_SLACK)` = 设计那句 `× 1.05`），判据与 docstring 一致；
`oracle.py` 那句手写 `* 1.05` 删掉、改走 `exceeds_measured_peak` ⇒ 阈值只剩**一个真源**。
三条守卫：① 常数两句话都钉住（余量 `0.05` **与** `1 + 余量 == 1.05`）；② P2-9 点名的
「第二调用方」直接成用例（`slack=0.05` 时 `50.0` 不许被判成超上界 —— 旧代码在这里是
一条假警报发生器）；③ 机械守卫：`oracle.py` 的**代码**（注释不算）里不许再出现 `1.05`。
三处变异全红（判据回退成 `×slack` / 常数回退成乘数 / oracle 再抄一遍阈值）。

### P2-10 `require_double_confirm`：默认关，打开就全拒

`conductor.py:1531-1533`

```python
elif getattr(self, "require_double_confirm", False):
    remeasure = getattr(artifact, "remeasure", None)
```

- 全仓除测试外**无人设为 True** → 门 3 默认不生效
- 更糟：`KernelArtifact` **没有 `remeasure` 字段**（`schemas.py` 无此字段）→
  一旦打开该开关，`remeasure` 恒为 `None` → `confirm_persist` 恒拒 →
  **入库率 100% 归零**

测试只靠 monkeypatch 造 `remeasure`（`tests/test_spine_wiring.py:63-84`），
所以测不出这个。这个开关目前是「要么没用，要么是自杀开关」。

**落地（2026-09-30）**：两条都改了。

- 字段早在 P2-10 那批就补上了（`schemas.py` 的 `remeasure` 有注释记着来由），本轮补的是
  **写者**：`Conductor._produce_remeasure()` 在**真会进语料**的那一支（`artifact.all_passed`）
  里再跑一次 L3，把结果写成 `ConfirmRecord`。放在这里而不是每轮都跑：确认的对象是
  「进语料」这个动作，对每一轮重测等于机时翻倍，而绝大多数轮次不会被记住 —— 这正是设计
  §10.2「最后 10% 预算」的取舍。开关 `require_double_confirm` 现在是构造参数的一部分
  （默认 `False`，仍是机时承诺），不再是「有本事你自己挂上去」的隐藏属性。
- **独立性的诚实边界**（写进方法 docstring）：给的是「另跑一次 L3」—— 新的计时窗口，同一台
  机器、同一份输入、同一张卡。它抓得住「第一次是侥幸」，抓不住「这台机器的尺子系统性偏了」
  —— 后者是标定心跳的活。`independent=True` 声称的只是前者。
- 重测**也过尺子检查**（到期就跑心跳）：否则「决定永久记忆的那一次测量」恰好会成为唯一
  没被检查过的那一次。落在坏窗口里的重测 `ok=False`，理由里带 `ruler_broken`（实测摘掉
  这条判据，用例变红）。
- **顺带补上 P0-2 那个残留**（本文档 P0-2 节末列的）：`ConfirmRecord` 现在带
  `contract_hash`（取**第二次**读数那条契约的戳，不是第一次的），`confirm_persist` 在首次
  那头给了哈希时判同一性 —— 不符回「重测在另一条契约下取得」，**没带**回「无从证明两次
  量的是同一条判据」。判据在两次测量之间被改写时，那两条读数就不再是同一条断言的两次
  确认，这个字段是唯一能证明它的东西。
- 端到端用例：`require_double_confirm=True` + 生产者 → `_save_to_db` **放行**（不再是归零）；
  换掉首次那条契约的哈希 → 拒，且断言拒的理由**正是**契约那一条（否则摘掉该判据用例仍会
  因别的门而拒，等于空转 —— 实测摘掉后 3 条用例变红）。

⚠️ 一条**未做**的：10% 预算仍然只有原则、没有账本。「留了多少预算给确认」与「实际花了
多少」在产物里都读不出来，所以 `_produce_remeasure` 无法按预算决定要不要跑 —— 它按
`require_double_confirm` 跑。要做成设计里那张预算表，得先有跨轮的成本账本（现有的
`CostTracker` 只记 LLM token，不记 GPU 机时）。

---

## P3 死代码与门面

| # | 位置 | 问题 |
|---|---|---|
| ~~11~~ | ~~`oracle.py:23`~~ | ~~导入 `heartbeat_due` 但从未使用~~ **已修**：死 import 删除；`heartbeat_due` 现在活在拿得到累计计数的那一层（`Conductor._heartbeat_for_next_measurement`）|
| ~~12~~ | ~~`calibration.py:218`~~ | ~~`heartbeat_due` / `ruler_broken_window` 除测试外无读者~~ **已修**：两者都有了生产读者（计数与到位判定在 `Conductor`，窗口判据在 `oracle.evaluate`）。仍**未做**：§10.6 那个 `abort` —— 心跳失败目前只作废那一扇窗的读数，不终止整轮运行 |
| ~~13~~ | ~~`oracle.py:65`~~ | ~~`remeasure` 参数声明了、从不使用~~ **已修**：死形参删除（全仓无人传；确认逻辑本来就在 `admit_to_corpus`）。`oracle.evaluate` 的 docstring 不再声称它做确认 |
| 14 | `contract.py:31` | `_DTYPE_KEYS` 定义了、从未使用 |
| ~~15~~ | ~~`contract.py:194-214`~~ | ~~**`load_dict()` 忽略载荷里的 `hash`**~~ **已修（2026-10-01 核实）**：现在重算后与载荷自带的 `hash` 比对，不符抛 `ContractError`（「判据被改写过」）|
| ~~16~~ | ~~`tests/test_stage1_acceptance.py:204`~~ | ~~`pairing_conflict()` 只存在于测试文件里~~ **已修（2026-10-01 核实）**：搬进 `operator_agent/placement.py`，`oracle.admit_to_corpus` 路径在读它，测试改为从库里 import |
| 17 | `contract.py:47` | `accum` 允许 `"tf32"`——TF32 是输入格式，不是累加精度；`dtype` 是不校验的自由字符串，与"显式钉死数值语义"的意图有落差 |
| ~~18~~ | ~~`contract.py:83`~~ | ~~遇畸形项抛裸 `ValueError`~~ **已修（2026-10-01 核实）**：`load_dict` 把 `KeyError/TypeError/ValueError` 统一包成 `ContractError`；`ShapeDist.validate()` 改为收集错误而不是就地抛 |

---

## 做对的部分（不是客套）

三处**比设计文档更好**：

1. **心跳的成对双比值**（`calibration.py:9-18`）。文档只写了"已知好 + 故意慢"，
   实现补上了两个盲区：只有好核看不出整体偏置；只有比值则在 harness 被整体缩放 10× 时
   仍然"看起来对"。`known_good_ratio` 抓绝对偏置、`known_slow_ratio` 抓分辨力。
   这是真补强。
2. **`ruler_broken_window` 在缺 `window_id` 时失败关闭**（`calibration.py:243-244`），
   且明确写了"跨窗不牵连"的理由。边界取对了。
3. **`heartbeat_due` 拒绝非正整数 `interval`**；`test_end_to_end_detects_field_edits`
   这类"改一个字段哈希必须变"的方向也对。

八条规则顺序、DEGRADED 位型 `(1,1,0,1,0)`、`ALL_FALSE` 共享冻结实例 ——
与设计文档一致，无问题。113 个测试全绿。

---

## 修复顺序建议

| 优先级 | 动作 | 理由 |
|---|---|---|
| **P0** | 把 `oracle.evaluate` 接进 Conductor 的 L3 出口 | 不接，这 8 个提交对生产行为影响 ≈ 0 |
| **P0** | `except Exception: pass` → fail-closed + 异常进 `reasons` | 现在守卫出错会静默放行 |
| **P0** | `lib_fallback` 接真实生产者，否则删掉这一位 | 恒 False 的护栏比没有更危险 |
| **P1** | `heartbeat_ok` 去掉默认 `True`；`evaluate` 要求显式心跳 | 「没做心跳」不能读成「心跳通过」 |
| **P1** | 哈希改为**记录携带**，不让调用方给两侧 | 现在的形状可恒真 |
| **P1** | `KernelArtifact.remeasure` 补上，或删掉该开关 | 现在打开即全拒 |
| **P2** | `cv_exceeded` 改用 `DISPERSION_*` 的真实结论 | 名不副实 + 污染 reasons |
| **P2** | `exceeds_measured_peak` 对齐 docstring，或被 oracle 复用 | 双真源 + 潜伏陷阱 |
| **P2** | `pairing_conflict` 移进库里，测试改为测库 | 验收不该测测试自己的函数 |
| **P3** | 死代码清理；`load_dict` 校验载荷 hash | 卫生 |

---

## 未决问题（会改变接口形状）

**Oracle 接在哪个位置？**

| 位置 | `Observation` 的含义 | 后果 |
|---|---|---|
| L3 出口（每次测量都过） | 「每个读数的记录」 | 契约哈希与心跳校验每轮都跑；但要给每个候选造记录 |
| `_save_to_db` 入口（只在入库时过） | 「入库申请单」 | 便宜；但降级读数不会留下观测记录，`can_count_progress` 无处落地 |

**建议**：L3 出口。因为设计 §10.4 的"可靠停滞"判据依赖 `can_count_progress`，
而它只有在每个读数都产生 `Observation` 时才累得起来。若只在入库时过，
停滞计数永远停在 0，停止条件形同虚设。

---

## 附录：本文每条结论的复核命令

```bash
# P0-1 零生产调用者
grep -rn "from .oracle\|oracle\.evaluate" --include=*.py . | grep -v "^./tests/"
grep -rn "from .contract\|contract\.freeze\|FrozenContract\|assert_hash" --include=*.py . | grep -v "^./tests/"

# P1-4 lib_fallback 无生产者
grep -rn "lib_fallback" . --include=*.py | grep -v docs/ | grep -v tests/

# P2-10 KernelArtifact 无 remeasure 字段
grep -n "remeasure" operator_agent/schemas.py
grep -rn "require_double_confirm" --include=*.py .

# P3 死读者
grep -rn "heartbeat_due\|exceeds_measured_peak\|PEAK_SLACK\|ruler_broken_window" \
     --include=*.py . | grep -v "^./operator_agent/calibration.py\|^./operator_agent/eligibility.py"

# 测试（全绿但只覆盖库）
python -m pytest tests/test_stage1_acceptance.py tests/test_oracle.py \
      tests/test_eligibility.py tests/test_calibration.py \
      tests/test_contract_freeze.py tests/test_confirm.py \
      tests/test_spine_wiring.py -q
```