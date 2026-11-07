# R2 预登记：v0.7.2 对 v0.8 的增量对照

2026-11-07 · v0.8 · 计划编号 `operator-v08-increment-r2`

本文件是**跑之前**冻结的判定规则，配合 [r2_plan.py](../../benchmarks/experiments/v08/r2_plan.py) 使用。
R2 回答的是 [v08-preparation §2](../design/v08-preparation.md#L23) 表格第二行那个问题：
同样的冻结预算下，加上"瓶颈诊断 + 候选比较"能不能让**终稿性能**更好。

它**不是**"插件比普通 DSH 强"的证明（那是 R1，已降级为副产物）。

## 1. 冻结了什么

| 项 | 冻结值 | 依据 |
|---|---|---|
| 两臂 | B = v0.7.2 封版工件；C = 同宿主 + 启用 `operator_context(phase=optimize)` | 只差 optimize 阶段，比的是增量不是工作流 |
| 任务 | `matmul-4096`：fp16 4096×4096×4096，triton | 唯一在本窗口可判的 shape（3× 带宽 0.203%） |
| 噪声门 | `preflight --shape 4096,4096,4096 --reps 5 --repeat 8 --planned-effect-pct 2.5`，`action != "start"` 就不启动 | 上一轮 74 分钟、195 次调用换来零个 valid 终评 |
| 最小可检效应 | **2.5%** | 0.203% 的 12.3 倍；容忍窗口恶化 10 倍仍可判 |
| 主指标 | 终稿加速比的**配对差值**，仅限终评 L3 为 `valid` 的配对 | `degraded` 读数只作观测 |
| 系列停止 | 连续 **2** 对没有 `valid` ⇒ 记 `unknown`，不启动第三对 | 两对都不可判是系统性窗口问题 |
| 终评预留 | 每对 1 次独立终评，同源码哈希 | 不允许把终评预算挪去探索 |
| 预算 | **未填**（`pairs` / 调用 / 输出 / 验证 / 墙钟 / GPU 分钟） | 必须由人填；填不全不许冻结 |

## 2. 两个容易混的数字

| 名字 | 是什么 | 用在哪 |
|---|---|---|
| `min_speedup`（契约里，如 1.0） | 任务**要求**达到多少 | 判定候选达没达标 |
| 最小可检效应（预登记里，2.5%） | 这次实验**能看出**多大差别 | 判定这次配对值不值得跑 |

写这份文件时我第一版填的是 2.0%，被 `validate()` 直接拦下——因为
`0.203% × 10 = 2.03%`，2.0 不够。**规则先于数字生效，这是好事**，所以最终登记 2.5%。

## 3. 怎么冻结

```bash
python -c "
from benchmarks.experiments.v08 import r2_plan
r2_plan.freeze(dict(pairs=2, max_model_calls=12, max_output_tokens_per_call=8192,
                    total_output_token_reservations=98304, max_verifications=6,
                    wall_clock_s=2400, gpu_minutes=60), '/path/outside/repo/preregister.json')
"
```

写盘用 `open('x')`：**不允许覆盖**已冻结的计划。要改判据只能新开计划、新编号，
和 [v0.7.2 的"不得原位改判据"](../../docs/status/v072-todo.md) 同一条纪律。

## 4. 它禁止什么

`validate()` 对以下情况直接拒绝：

- `formal_gain_claim` 或 `formal_comparison_credit` 为真（预登记不能自带结论）；
- 计划里出现 `outcome` 字段（跑完才能有结果）；
- 冻结时预算有空项；
- 最小可检效应没有越过 `窗口带宽 × 10`；
- 系列停止规则不是"连续 2 对无 valid"；
- 内容与 `hash` 不符（改一个数就会被发现）。

## 5. 每对怎么跑

1. **门**：`preflight` 落 `hold` ⇒ 这一对不启动，不计失败、不消耗模型/验证预算，等下一个窗口。
2. **两臂**：同宿主、同模型、同冻结预算；顺序交替（B→C、C→B），避免把漂移算到某一臂头上。
3. **终评**：每对跑一次独立终评，绑定提交源码的哈希；L2 不全过或 L3 非 `valid`，这一对记为不可判。
4. **停**：连续 2 对不可判 ⇒ 记 `unknown` 收工。

## 6. 这条系列不能得出的结论

写在计划里，也写在这里：

- 两臂都用插件，**不能**据此说"插件优于普通 Agent"。
- `sampling_comparability` 仍是 `not_established`（A 臂无同等硬门、账本未记
  `temperature/n/enable_thinking`），所以只报**配对观测**，不报绝对收益。
- 任务没拿到 `valid` 终评时，那是关于**窗口**的结论，不是关于候选的结论。

## 7. 两臂的执行环境（2026-11-07 核实）

两臂必须用**各自独立安装**，否则来回切换工作树会让身份漂移：

| | 臂 B | 臂 C |
|---|---|---|
| 源码 | `git worktree` @ tag `v0.7.2`（`/Users/zh/op-agent-delivery/v072-arm-b`） | 当前分支 `codex/v0.8` 工作树 |
| Python | `/Users/zh/op-agent-delivery/v072-venv/bin/operator-agent*` | `/Users/zh/op-agent-delivery/v08-venv/bin/operator-agent*` |
| `TOOL_VERSION` | 12 | **12**（工具面相同，这是两臂可比的前提） |
| `decide.noise_preflight` | 无 | 有 |
| `generate.optimize_diagnoser` | 无 | 有 |

⚠️ **当前 desktop profile 的 `mcp-operator` 跑的是 `v07-venv` 里 site-packages 的 `0.7.0rc1`**
（见 `~/.dsh/profiles/desktop/cordis.patch.yml:130`），既不是 B 也不是 C。R2 必须在
分别指向上面两个 venv 的会话里跑，不能在当前会话里跑。

## 8. 现状

- 计划模板与校验器已完成（[r2_plan.py](../../benchmarks/experiments/v08/r2_plan.py)），7 个用例钉住上述规则。
- **尚缺**：预算数字（需人填）、实验宿主（`--host`）、以及 R2 的第一对被真实执行。
- 噪声门的实现见 [v08-noise-preflight.md](v08-noise-preflight.md)；本窗口实测见
  [真机回执记录](../status/v08/real-receipt-20261107.md)。