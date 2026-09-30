# spine 实现审查 —— `design/autonomous-spine` 8 个提交

| 项 | 值 |
|---|---|
| 审查对象 | 分支 `design/autonomous-spine`，提交范围 `3a5d25c..24d0e53`（spine 本体 8 个提交） |
| 被审代码 | `calibration.py` / `contract.py` / `eligibility.py` / `confirm.py` / `oracle.py` + `conductor.py` `_save_to_db` + `verifier.py` `_assess_measurement` |
| 读数条件 | 2026-09-30，`24d0e53`，净工作树 |
| 测试状态 | `113 passed in 1.46s`（7 个 spine 测试文件） |
| 设计出处 | `docs/design/autonomous-operator-agent-design.md` §3 / §4 / §10.5 / §13 / §14 |
| 审查结论 | **§14 验收未通过** —— 实现是一个库，不是一条链路 |

## 结论摘要

| 级别 | 条目 | 一句话 |
|---|---|---|
| **P0** | 1 | `oracle.py` / `contract.py` **零生产调用者** → 收口点不在链路上 |
| **P0** | 2 | `evaluate(contract, contract_hash, …)` 比较两侧都由调用方给 → **恒真** |
| **P0** | 3 | `verifier.py` 的 `except Exception: pass` → 守卫出错时**静默放行** |
| **P1** | 4 | `lib_fallback` **全仓无生产者** → 规则 4 恒为 False |
| **P1** | 5 | 资格推导默认值全绿；`heartbeat=None` 被读成心跳通过 |
| **P1** | 6 | `cv_exceeded` 从不与任何 CV 阈值比较 |
| **P2** | 7 | 所有 `ALL_FALSE` 压成 `validity="uninformative"` → 规则 3/6 被合并 |
| **P2** | 8 | `admit_to_corpus` 与 `confirm_persist` 对"重测可 DEGRADED"互相矛盾 |
| **P2** | 9 | `exceeds_measured_peak` docstring 与代码不一致（潜伏陷阱） |
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

### P2-8 两次确认的互相矛盾

`admit_to_corpus`（`oracle.py:145-162`）要求**重测那次**也 `can_persist`；
而 `confirm.py:38-39` 明确允许重测是 DEGRADED（"DEGRADED 允许更宽的噪声"）。

但 DEGRADED 的位型是 `can_persist=False`（`eligibility.py:68`）——
所以 `confirm_persist` 那条宽容在 `admit_to_corpus` 路径上**永远走不到**。
两者必有一个是错的。

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

---

## P3 死代码与门面

| # | 位置 | 问题 |
|---|---|---|
| 11 | `oracle.py:23` | 导入 `heartbeat_due` 但从未使用 |
| 12 | `calibration.py:218` | `heartbeat_due` / `ruler_broken_window` 除测试外无读者 → 设计 §10.6 的 `if heartbeat_due(): if not calibrate(): abort` 主循环**不存在** |
| 13 | `oracle.py:65` | `remeasure` 参数声明了、从不使用（commit message 却称 oracle 做"可选重测确认"） |
| 14 | `contract.py:31` | `_DTYPE_KEYS` 定义了、从未使用 |
| 15 | `contract.py:194-214` | **`load_dict()` 忽略载荷里的 `hash`**，重新 `freeze()` 算一个新哈希 → 被改过的载荷得到合法新哈希；续跑若拿内嵌 hash 当期望值，改写检测不出来 |
| 16 | `tests/test_stage1_acceptance.py:204` | **`pairing_conflict()` 只存在于测试文件里**，库里没有 → §14-7「跨卡配对拒绝」测的是一个不会发布的函数 |
| 17 | `contract.py:47` | `accum` 允许 `"tf32"`——TF32 是输入格式，不是累加精度；`dtype` 是不校验的自由字符串，与"显式钉死数值语义"的意图有落差 |
| 18 | `contract.py:83` | `for shape, w in self.entries` 遇畸形项抛**裸 `ValueError`**，不是 `ContractError`（它是子类，捕获 `ContractError` 的调用方抓不到） |

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