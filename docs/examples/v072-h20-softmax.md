# v0.7.2 完整执行示例：H20 FP16 Softmax

这是 **2026-11-06 实际执行**的候选包示例：DSH 外层真实模型自主选择工具、编写源码，在 H20 上完成探索验证及同源码独立终评。本文按日志整理，没有重新调用模型或 GPU，没有改写原始日志。

版本：Python `0.7.2rc1` / 插件 `0.2.2-rc.4` / 工具契约 `12`，尚未正式封版。模型为 DSH 已配置的 `aliyun-native / deepseek-v4.1-flash-1017`。GPU、运行时与 15 个 runner 文件在执行前后核对一致。

## 先看哪些文件

| 阅读目的 | 实际文件 |
|---|---|
| 原始用户任务与执行约束 | [prompt.txt](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/prompt.txt) |
| DSH 对外输出、工具参数和回包 | [gpu.stdout.jsonl](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/gpu.stdout.jsonl) |
| 模型请求、usage、阶段完成与材料送达的时间线 | [events.jsonl](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/events.jsonl) |
| 每个工具的完整参数与未经展示裁剪的原始回包 | [tool-results.jsonl](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/tool-results.jsonl) |
| 实际交付的领域材料 | [context JSON](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/task/contexts/f7e6fc6ef0094f6bb1f58957d0fcc15c.json) |
| 第一次探索验证的执行日志 | [verify-1/execution.log](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/task/verify-1/execution.log) |
| 第二次独立终评的执行日志 | [verify-2/execution.log](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/task/verify-2/execution.log) |
| 逐 case 的终评证据与成员哈希 | [verify-2/result.json](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/task/verify-2/result.json)、[complete.json](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/task/verify-2/complete.json) |
| 最终源码与报告 | [code.py](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/export/code.py)、[report.md](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/export/report.md) |
| 验证预算和模型最终账本 | [host-task.json](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/task/host-task.json)、[host-model-budget.json](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/task/host-model-budget.json) |
| 验收摘要 | [summary.json](/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1/summary.json) |

原始记录目前在 `/private/tmp/operator-v072-closeout.rVQIRa/native-gpu-rc1`，正式归档属于 F6 封版收尾，尚未完成。本页不链接私有 profile 或凭据。`gpu.stdout.jsonl` 是面向 CLI 的展示流，部分工具展示会标 truncated；材料是否实际完整到达模型，以 `events.jsonl` 的 `context_delivery` 和完整工具回包为准。本次核对 exact=true、spill=false。

## 整体执行过程

```text
用户声明 Softmax FP16 4096×4096
  → 模型 #1 决定 prepare，插件冻结任务和预算
  → 模型 #2 决定取 generate context
  → 模型 #3 结合领域材料生成 Triton 源码，提交 explore verify
  → Python 执行 L1 → 完整 L2 → L3，保存证据
  → 模型 #4 提交相同源码的独立 final verify
  → Python 新执行完整验证，保存独立终评证据并关闭开发
  → 模型 #5 汇总；调用完成后的模型账本包含这次总结
```

实际工具顺序只有 4 次：`prepare → context → verify(explore) → verify(final)`。本例首次候选完整 L2 即通过，没有修复轮；修复和中断恢复的补充实例在文末。

### 1. 冻结任务

模型提交的声明为 `softmax fp16`、shape `[4096, 4096]`、DSL `triton`，性能门槛 `min_speedup=1`。本例显式冻结的预算为墙钟 600 秒、验证 4 次（其中一次 final 固定保留）、绑定后主模型 12 次、每次输出上限 8192、累计输出预留 98304。4 次验证是本例参数，产品默认是 6 次。

任务 ID 为 `46e5f44f1e43427ca04f3883bead834f`。准备返回之后，JS 适配器绑定原任务并开始计模型预算；第一次准备请求本身尚未绑定，单列而不计入这份任务账本。

### 2. 取材与生成

模型调用 `operator_context(phase=generate, backend=ssh)`，取得冻结任务、ABI、生成规约、技能等领域材料，再由 DSH 组装最终模型消息。observer 核对这份工具正文完整进入模型 #3 的请求，无 spill。

模型生成的接口是 `kernel(X)`。实现按行启动 Triton program，使用 mask 处理尾部、减最大值稳定化、fp32 归约并输出 fp16；运行时读取 strides，包装层会把非连续输入转为连续。完整源码见上方 `code.py`，不能把“通过 strided 测试”误写成“零拷贝处理 strided 输入”。

### 3. 探索验证与独立终评

两次验证均使用原 run_dir、契约和完全相同源码。第一次为 `explore`，第二次明确传 `verification_purpose=final`；第二次实际执行新验证，不复用第一次结果。终评后任务阶段为 `final_complete`，这表示独立正确性终评完成，并不代表 L3 通过。

最终执行日志中的关键原文：

```text
[Verifier] L1 PASSED: 编译成功
[Verifier] L2 PASSED: 功能验证通过 (1 组输入), 最大误差=1.91e-06, matched_ratio=1.000000
[Verifier] CorrectnessSuite passed: ... CorrectnessSuite passed (12 case)
[Verifier] L3 FAILED: ... 测量有效性=uninformative: launch 地板占实测时间 42.1% (> 30%) ...
```

上面 CorrectnessSuite 行中的 `...` 是本页节选省略，完整原文在 execution.log。最终判定 **L1=true、L2=true、L3=false、validity=uninformative**；主用例比较 16,777,216 元素。实测约 3.40× 仅为无资格的观测数字，不建立性能收益。工具返回的 `route=search` 必须连同 eligibility/conflict 阅读，不能绕过无效测量约束继续调参。

### 4. 最终账本与原始模型总结的区别

全程观察到模型请求 **5 次**，成功绑定后的模型账本 **4 次**（包含总结），输出预留 **32768 token**；Python 内层模型调用 **0 次**。验证实际使用 **2 次**，final 保留已用完；已完成的终评关闭开发，剩余计数不重新开放开发资格。

原始模型总结引用了 final 工具回包中的旧快照，写成“3 次 / 24576 token”；它没有计入正在生成该总结的第 5 次请求。最终结算应读 `host-model-budget.json` 的 **4 次 / 32768**，不按模型自然语言结算。原总结中的“~3.4x faster”措辞也不能替代 `L3=false / uninformative` 的判定。原始输出保留，本文明确校正解释。

## 带时间戳的日志索引

以下为 2026-11-06 **Asia/Shanghai** 时间，从首次原生模型请求开始计相对时间；覆盖记录窗口约 49.114 秒，不是性能收益或端到端服务 SLA。

| 本地时间 | 相对时间 | 实际事件 | 原始日志行 |
|---|---:|---|---|
| 10:03:50.403 | +0.000s | 模型请求 #1，输出上限 8192 token | events.jsonl:1 |
| 10:03:52.509 | +2.106s | 模型回答结束：tool-calls | events.jsonl:3 |
| 10:03:52.657 | +2.254s | operator_prepare 返回，ok=True | events.jsonl:5 |
| 10:03:52.708 | +2.305s | 模型请求 #2，输出上限 8192 token | events.jsonl:6 |
| 10:03:53.985 | +3.582s | 模型回答结束：tool-calls | events.jsonl:8 |
| 10:03:54.033 | +3.630s | operator_context 返回，ok=True | events.jsonl:10 |
| 10:03:54.057 | +3.654s | 核对材料进入下一次模型请求：完整=True，spill=False，31548 个 JS UTF-16 单元 | events.jsonl:11 |
| 10:03:54.058 | +3.655s | 模型请求 #3，输出上限 8192 token | events.jsonl:12 |
| 10:04:00.778 | +10.375s | 模型回答结束：tool-calls | events.jsonl:14 |
| 10:04:13.798 | +23.395s | operator_verify 返回，ok=True；第 1 轮 explore，L1/L2/L3=True/True/False | events.jsonl:16 |
| 10:04:13.832 | +23.429s | 模型请求 #4，输出上限 8192 token | events.jsonl:17 |
| 10:04:20.483 | +30.080s | 模型回答结束：tool-calls | events.jsonl:19 |
| 10:04:32.649 | +42.246s | operator_verify 返回，ok=True；第 2 轮 final，L1/L2/L3=True/True/False | events.jsonl:21 |
| 10:04:32.682 | +42.279s | 模型请求 #5，输出上限 8192 token | events.jsonl:22 |
| 10:04:39.500 | +49.097s | 模型回答结束：stop | events.jsonl:24 |

## 身份与核对方式

- 契约 SHA256：`8fd7e3ed178667c3e6ade373b032c98edef82454de6393edcde72bdda6b8413c`。
- 候选 SHA256：`0698da97ab61741b7adc8a16035d6c5015ee58fcadb189f05c5aab40be41531a`；两轮源码逐字相同，导出源码也相同。
- `callId` 可串起 DSH `tool_call`、`tool_result`、完整工具回包及材料送达事件。
- `verify-2/complete.json` 绑定终评证据成员哈希；模型账本持有其自身 state hash 和冻结 policy 身份。
- wheel/tgz 哈希与候选安装身份见[本轮验收记录](../status/v072-budget-resume-acceptance-20261106.md)。

## 另外两条完整实例

- **错误候选 → 绑定失败反馈 → 外层修复 → 同源码独立终评：** [完整工具记录](/private/tmp/operator-v072-closeout.rVQIRa/native-repair-rc1/tool-results.jsonl)、[DSH 原始过程](/private/tmp/operator-v072-closeout.rVQIRa/native-repair-rc1/fake-repair.stdout.jsonl)、[独立证据复核](/private/tmp/operator-v072-closeout.rVQIRa/native-repair-rc1/repair-evidence-review.json)。该驱动因模型省略错误 seed 的末尾 LF 而判失败，原 summary=false 保留；独立复核确认仅末尾 LF 差异、AST 相同、实际失败证据与修复/终评闭环成立。
- **用户暂停 → 正常退出 → 同 session 重启 → 追加源码要求 → 独立终评：** [暂停过程](/private/tmp/operator-v072-closeout.rVQIRa/native-resume-rc1/pause.stdout.jsonl)、[恢复过程](/private/tmp/operator-v072-closeout.rVQIRa/native-resume-rc1/resume.stdout.jsonl)、[完整工具记录](/private/tmp/operator-v072-closeout.rVQIRa/native-resume-rc1/tool-results.jsonl)、[验收摘要](/private/tmp/operator-v072-closeout.rVQIRa/native-resume-rc1/summary.json)。实际 8 次模型请求，绑定后 7 次，保持原任务、契约、截止与历史记录，新增用户注释进入真实请求和实际源码。
