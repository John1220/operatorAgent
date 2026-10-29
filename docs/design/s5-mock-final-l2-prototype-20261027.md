# S5 共用 L2 终评：本地 mock 原型

状态：**仅用于本地可信 Python 测试夹具；不用于 S5 正式裁决**。本原型不启动 DSH、SSH、模型或 L3，也不把 mock 结果记成真实 GPU 正确性/性能。

`benchmarks/s5_l2_evaluator.py` 按同一入口接收 A/B/C 标签、源码和任务 ID。标签只是声明，不证明实际生成路径；终评只判给定源码，不修改或修复候选。十项历史任务的 primary、两个 holdout 与既有 `quality_suite` 的声明完全一致；四项大形状候选沿用 S5 草案生成的 primary 与两个 holdout。任务及 suite 仍是候选草案，不能称为正式冻结或独立未见。

每个输入组分别物化并调用既有 `Verifier.verify(stop_after=L2)`：一次 primary 或 holdout 必须经过 L1、该输入的 L2 与完整可信 suite。验证器复核 primary 的输入保真、全部 suite case 和确定性重复的输入身份，绑定源码 SHA256、语义契约 SHA256、suite SHA256 和输入声明摘要；返回完整原始 stage metrics 供调用方保存。三组检查全部通过才能给该 mock 任务 `full_l2_passed=true`；失败返回 false，缺阶段、缺 case、suite/输入身份不符、缺失回执或不可支持返回 unknown。首次明确失败可停止，不能把未跑的 holdout 算作通过。

语义契约不一定包含设备与 DSL；本原型单独记录 `mock_gpu` / `python`，并固定 `formal_s5_credit=false`，不因语义契约哈希相同就冒充 `nvidia_gpu` / `triton`。每个主输入组限制小于 1 MiB 的保守存储估计，四项大形状默认在物化前返回 unknown；该限制仅用于安全的本地原型，不是正式任务的能力或性能结论。

**信任边界**：调用者只能传入可信本地测试夹具。简易 AST 限制避免误用常见 GPU/provider 导入，但执行 guard 不是恶意代码沙箱，不应直接运行未经隔离的模型生成源码。证据仅从当前进程返回，不提供正式的不可变写入、外部签名、独立 GPU 执行或来源认证；正式 S5 还需独立进程/有界监督、真实设备及回执、候选隔离、每组一致的输入/语义以及最终 L3 和独立确认。

本地验证：`python -m pytest tests/test_s5_l2_evaluator.py tests/test_s5_task_manifest.py -q` **15 通过**；覆盖实际 mock 完整 L2、失败候选、大形状在物化前终止、外部/GPU 导入拒绝、未知任务与草案漂移、缺 suite case、缺 holdout、缺失/错绑输入保真。没有进行真实模型或 GPU 调用。
