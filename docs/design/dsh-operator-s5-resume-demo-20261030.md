# 算子开发插件：三分钟演示与简历口径（2026-10-30）

状态：**S6 演示材料已准备；按用户决定不录制视频。** 本文、[脱敏案例 ZIP](../../output/plugin_completion/20261030-s5-closeout-v1/shareable/s5-plugin-cases.zip)和[安装验收记录](../status/dsh-operator-s6-acceptance-20261030.md)构成材料包。演示使用历史日志和候选回放，不运行模型/GPU，不把回放显示为当前执行。当前安装工件的新增真机验收另行标注，以 [S5 结项报告](../status/dsh-operator-s5-closeout-20261030.md)及最终交付清单为准。

**2026-11-10 更新**：v0.8 A/C 工作流轮完成预登记两轮复现，判读器在限定范围内判 `reproduced`，见[对照报告](../status/v08/ac-workflow-round-report-20261110.md)与[复现判读回执](../../output/v08-ac-20261108/reproduction-verdict.json)。本文的简历表述与追问表已按该证据改写，演示收尾段口述同步更新；其余演示段仍基于 S5/S6 历史材料。各历史文档里「插件整体优于原生 DSH 未建立」是当日事实，保留不改——新轮是新计划新身份，不补旧 S5 分母、不是翻案。

对外演示优先解压 [脱敏案例 ZIP](../../output/plugin_completion/20261030-s5-closeout-v1/shareable/s5-plugin-cases.zip)，先运行 `python3 -I -S verify.py`。第一段对应 `entry/development.json`、`entry/final.py`、`entry/final.json`；修复段对应 `repair/initial.py`、`repair/initial.json`、`repair/revised.py`、`repair/revised.json`、`repair/original-final.json` 和 `repair/recovery.json`。下文原始链接供私下追查，不直接公开投屏或发送。

## 0:00–0:25：说明解决的问题

口述：“DSH 可以写 Triton 源码。这个插件把编译、真实 GPU 正确性验证和结构化反馈接到开发循环里，每次判定都有对应源码和契约；失败、测量不合格和执行未知分别留存。”

展示插件调用链：DSH → `operator_verify` MCP → 监督验证服务 → H20 → 反馈 → 最终源码独立验证。说明这里演示的是有限 L2 正确性，不承诺性能加速。

## 0:25–1:05：一条真实插件闭环

打开 [LayerNorm 入口验收 audit](../../output/plugin_completion/20261029-s5-plugin-entry-acceptance-v1/execution/acceptance-audit.json)：

1. 指出 `operator_verify_submitted`、`l2_feedback_returned_to_dsh`、`final_candidate_passed` 三项为 true；这是实际 DSH 的工具 call/result 审计，不是监督器终评冒充插件调用。
2. 打开 [最终候选](../../output/plugin_completion/20261029-s5-plugin-entry-acceptance-v1/execution/runs/layernorm-tail/C/supervised-verification/002-final/candidate.py)与 audit，指出候选哈希一致、H20 GPU2、`correctness_cases_total=12`、`full_l2_passed=true`。
3. 口述：“这次会话完成五次模型调用，证明该历史版本的单任务入口真实可用。12 个 case 是生产有限套件，不是对全部输入的正确性证明。”

## 1:05–2:05：错误如何被发现、如何修复

并排打开受控修复的 [初始候选](../../output/plugin_completion/20261029-s5-plugin-repair-acceptance-v1/execution/runs/layernorm-tail/C/supervised-verification/001-development/candidate.py)和 [修复候选](../../output/plugin_completion/20261029-s5-plugin-repair-acceptance-v1/execution/runs/layernorm-tail/C/supervised-verification/002-development/candidate.py)。

- 明确起点是预先注入的数值缺陷：方差求和遗漏除以列数，不是随机抽样的自然模型错误。
- 打开 [第一次反馈](../../output/plugin_completion/20261029-s5-plugin-repair-acceptance-v1/execution/runs/layernorm-tail/C/supervised-verification/001-development/record.json)：L1 通过，L2 数值失败。失败的首个输入已足以拒绝，不能说所有 12 case 均执行且失败。
- 打开 [第二次反馈](../../output/plugin_completion/20261029-s5-plugin-repair-acceptance-v1/execution/runs/layernorm-tail/C/supervised-verification/002-development/record.json)：DSH 收到反馈后修改了源码，关键修改是 `var = tl.sum(xc * xc, axis=0) / n_cols`，复验完整 suite 通过。
- 最后打开 [原始首跑汇总](../../output/plugin_completion/20261029-s5-plugin-repair-acceptance-v1/execution/aggregate.json)和 [恢复终评 summary](../../output/plugin_completion/20261029-s5-plugin-repair-acceptance-v1/final-recovery/summary.json)：首跑因说明文字被旧提取器当作源码而交付失败；修正提取后明确追加一次独立验证，12/12 通过，新增模型调用 0。原首跑没有被改成成功。

口述：“这个例子证明错误判定、真实反馈和修复提交链路成立；它不证明插件比原生 DSH 更容易修好，也不证明反馈的因果增益。”

## 2:05–2:40：性能数字与证据边界

打开 [大形状 LayerNorm 的真实 L3](../../output/plugin_completion/20261030-gpu-freeze/layernorm16384x6144-v4/verify-1/result.json)：H20 GPU7，16384×6144 fp32，`valid`、`measurement_can_gate=true`，相对 PyTorch Eager 观测 1.4135×。注明这是手写候选经 **CLI/core** 的单形状测量，不是插件相对原生 DSH 的提升，也未完成独立性能确认。完整环境与中间修正见 [GPU 冻结报告](../status/dsh-operator-gpu-freeze-20261030.md)。

打开 [S4 LayerNorm 的性能投影](../../output/plugin_completion/20261027-s4/portable-case-bundle-v2/cases/layernorm-tail/final-l3.json)：`can_gate=false`，launch 地板约占 78.2%，保留为性能未知。注明这来自历史 **CLI/core** 测量，不是前述 MCP 插件案例的性能结果。

如面试关注可靠性，可用 [S5 v2 结论](../status/dsh-operator-s5-cooperative-study-v2-20261030.md)替换性能片段：运行期间验证实现哈希变动，底层 L1/L2 通过仍不足以接受整次验证；终评 unknown 后停止派发，23 格未执行。

口述：“数值失败、测量不合格和设施未知有不同含义。报告保留这些状态，不能把没有合格读数当加速，也不能把设施中断当作模型数值错误。”

## 2:40–3:00：收尾与能力边界

展示 [结项报告的各批次表](../status/dsh-operator-s5-closeout-20261030.md)：各计划独立保留分母。口述：“S5 的探索性对照当时未建立整体增益，批次分母原样保留。2026-11-10 的 v0.8 A/C 轮按预登记规则完成两轮复现：在弱模型宿主（Qwen3.8-27B/xhigh）、一次生成可交付的任务上，插件臂每轮多交付 +1 任务，判 reproduced；强模型下两臂打平。性能与开发人时不在结论范围内。”随后打开[对照报告](../status/v08/ac-workflow-round-report-20261110.md)与[复现判读回执](../../output/v08-ac-20261108/reproduction-verdict.json)，指出 scope 与 boundaries 字段是结论的一部分，引用时不得省略。

展示最终交付清单中冻结工件与脱敏案例包的位置；若当前工件 P1 尚未通过，说明真实案例来自历史版本。分享使用脱敏包，私有原始目录不直接投屏或发送。

## 简历表述

可直接使用的头条（2026-11-10 起，依据[复现判读回执](../../output/v08-ac-20261108/reproduction-verdict.json)）：

> 设计并执行裸 Agent vs 插件的预登记对照实验：同沙箱、同预算、同冻结契约，弱模型宿主（Qwen3.8-27B/xhigh）下两轮复现均成立（每轮多交付 +1 任务，判读器判 reproduced）；插件臂交付用 3–9 次模型调用，对照臂每格用满 20 次预算。强模型下两臂打平。结论范围不含性能与开发人时。

功能主体（仍然有效）：

> 开发 DSH 算子开发插件，打通 MCP、远端 H20 编译/正确性验证与结构化反馈；完成 LayerNorm 真实插件闭环及最终 12-case L2 验收，验证受控数值缺陷的修复流程。

可按版面补充的工程贡献：

> 实现候选/契约哈希绑定、逐轮证据留存、调用预算与任务取消；区分数值失败、无效测量和设施未知，支持离线复核判定与运行成本。

可按版面补充的实验工程：

> 搭建双臂对照 harness：沙箱隔离（仓库不可读）、模型调用预算硬门、采样身份六字段逐字核对、预登记冻结计划与复现判读器；定位并修复 7 个会伪装成能力差异的仪器缺陷；单测 6400+。

若版面希望突出性能判定，可另写：

> 在 H20 上验证 compute-bound Matmul 与 memory-bound LayerNorm 的有效测量链路；16384×6144 fp32 LayerNorm 相对 PyTorch Eager 单形状观测 1.4135×，保留完整正确性、基线与测量资格记录。

最后一条引用历史 CLI/core 实测，不归因于插件接入或模型生成；答辩时应主动说明单形状、单设备且无独立性能确认。

“12-case”指上述单个任务的有限套件，不能改写为“12 个算子”或“正确率 100%”；attention 的 8 条 case 同理。不写“显著优于 DSH”“提高成功率 X%”“模型调用下降 X%”“性能全面达标”，也不将 S4 CLI 四项成功写成四次真实 MCP 插件验收。2026-11-10 起允许的唯一增益表述是复现判读回执的**带范围原样**（弱宿主 Qwen3.8-27B/xhigh、一次生成可交付的任务、两轮每轮 +1）：不得省略 scope 写成无条件增益，不得换算成百分比；调用数只写实测原始值（3–9 次 vs 20 次预算上限），不写“下降 X%”。

## 面试追问准备

| 追问 | 可据实回答 |
|---|---|
| 为什么插件值得做？ | 它将验证条件、真实 GPU 执行、结构化反馈和证据留存接入模型已有开发入口；2026-11-10 起另有预登记两轮复现的限定范围增益（弱宿主、一次生成可交付的任务）。 |
| 模型会不会自行宣布成功？ | 最终通过来自验证器；候选、契约、回执和完整 suite 需要相符。模型输出文本不能直接改变工具结论。 |
| 有没有证明优于原生 DSH？ | 有，但只在复现判读写死的范围内：弱模型宿主（Qwen3.8-27B/xhigh）、一次生成可交付的任务，两轮每轮 +1 任务。强模型（deepseek-v4.1-flash）同模型三格打平。这不是 S5 翻案：旧 24 格分母保留不补，新轮是新计划新身份。 |
| 增益是怎么验证的？ | 计划先冻结再跑格；每轮判读要求 round_complete 且采样身份六字段逐字一致；聚合规则「每轮 ≥+1，任一轮掉 0 判 null 且不追加轮次」写在冻结计划里。过程中抓出并修复 7 个仪器缺陷（预算口径、臂分派写死、ABI 缺标量、流式截止、MCP 超时、两臂采样身份不一致、网关预留护栏）。 |
| 为什么强模型上没有差距？ | 机制是对照可解释的：插件把 ABI、数值契约、被测布局、规则一次性给全；强模型本来就能补齐这些事实，弱模型猜错概率高、试错预算少，差距被放大。 |
| 每个任务都稳定复现吗？ | 不稳定。复现轮里 attention 两臂都是 ✗→✓，只有 softmax 稳定（C 2/2 交付、A 0/2 失败）。结论是轮级交付差，不是单点稳定性；墙钟方向也逐任务不一致。 |
| 修复是否一次成功？ | 开发复验通过，原始首跑最终交付因提取格式失败。修正提取器后独立追加一次 GPU 终评通过；两条记录分别保留。 |
| 有没有性能提升？ | 插件相对 DSH 的性能/效率提升未建立。已有 CLI/core 单形状 valid L3：Matmul 4096³/8192³ 分别 1.0569×/1.0239×，LayerNorm 16384×6144 为 1.4135×，均针对其记录的基线，未作独立性能确认。S4 四项测量仍不可裁决。 |
| 如何复现与分享？ | 冻结交付工件与环境，分享脱敏源码/判定投影及离线核验器；GPU 数值复演仍需对应硬件/依赖。哈希证明内部一致性，不是历史真实性的数字签名。 |
