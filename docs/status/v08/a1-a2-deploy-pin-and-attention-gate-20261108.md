# A1 部署钉版本 + A2 attention 窗口门（2026-11-08）

| 项 | 值 |
|---|---|
| 分支 | `codex/v0.8`（工作树，未提交） |
| 真机 | `h20`，GPU 3（`GPU-634bbc76…`），隔离 runner 与默认 runner 两条路都验了 |
| 证据 | `output/v08-attention-20261108/c5/`（契约、两次运行、gate 回执）；`output/v08-attention-20261108/a1-default-verify.json` |

这两条是 attention 进 R2 之前的前置：**跑得起来**（部署不再漂）与**判得了**（有自己的窗口门）。

## A1：部署模板钉 torch 版本

### 原来的问题

`remote_executor._setup_uv_env` 跑的是 `uv add numpy torch triton`，**没有版本约束**。
2026-11-08 第一次跑 C5 时，它把工作目录里的 2.14.0 升成 2.14.1，参考身份门随即
fail closed（`native_libraries, torch_config_sha256, torch_git, torch_version`），
attention 与 layernorm v2 一起停在 L2 之前。这正是 [C5 卡点 §2](c5-identity-blocker-20261107.md) 记的根因，
那天是靠 `OPERATOR_AGENT_EXISTING_REMOTE_ONLY=1` 绕过去的。

### 改了什么

| 位置 | 改动 |
|---|---|
| `core/runtime_pin.py`（新） | `TORCH_VERSION = "2.14.0+cu130"` + `torch_requirement()`（发行版本 `torch==2.14.0`）+ `matches()` |
| `execute/remote_executor.py` | `uv add numpy "torch==2.14.0" triton`；装完**回读** `torch.__version__`，不等于 pin 就 `return False`，不把环境标记成就绪 |
| `decide/reference_contract.py` | `PINNED_RUNTIME` 仍写字面量，**不** import：远端把本文件当**顶层模块**平铺加载 |

版本号为什么有两份字面量：`execute` 不能 import `decide`（分层规矩），而
`reference_contract.py` 在远端是裸文件（`scp` 到工作目录根）。两份由
`tests/unit/v08/test_deploy_runtime_pin.py::test_the_two_literals_are_one_value` 钉住；
另有一条 `test_the_reference_module_still_imports_flat` 把"平铺可导入"这件事变成测试——
**这一条是当场踩出来的**：第一版我用 `from ..core.runtime_pin import TORCH_VERSION`，
本机全绿、远端正解（`ImportError`），身份回执变空，报的是"执行侧参考身份回执缺失"。

### 真机验收

| 场景 | 结果 |
|---|---|
| 全新工作目录 `~/operator-o8-a1-161201`（普通部署，无任何绕过开关） | `pyproject` 落 `torch==2.14.0`，venv 报 `2.14.0+cu130`；L1/L2 通过，**8 条 case 全过** |
| **默认工作目录 `~/operator_agent`（就是被升坏的那个）** | 下一次部署自动降回 `torch==2.14.0`；L1/L2 通过，8 条 case 全过 |

也就是说：**不需要人工干预，下一个部署会自己把漂掉的环境修回来，修不回来就拒绝放行。**

边界：pin 只钉 torch（身份门只看它）。numpy/triton 仍是浮动下限（`numpy>=2.2.6`、`triton>=3.8.0`）。

## A2：attention 的噪声门

### 改了什么

| 位置 | 改动 |
|---|---|
| `decide/noise_preflight.py` | `ATTENTION_PROBE`：flash-attention 式分块核（在线 softmax、fp32 累加、`causal: tl.constexpr`），静态审计 0 违规 |
| `app/hitl_tools.py` | `_shape_probe_plan()` 单一决定「探哪条通道」；attention 逐位有界；`--dtype fp32` 直接拒绝 |
| `app/cli.py` | `--op attention`、`--causal` |

边界（拒绝而不是默认）：`fp16` only、`1<=B<=4096`、`1<=H<=4096`、`16<=S<=8192`、D 为 `[16,256]` 内的 2 的幂。
第一版用 matmul 那条 `[16, 65536]` 校验，把 **B=1 判成了非法**——真机第一次就撞上，已改。

### 真机回执（2026-11-08 16:19 CST）

```bash
operator-agent preflight --backend ssh --host h20 --op attention \
    --shape 1,16,1024,128 --causal --dtype fp16 \
    --reps 5 --repeat 8 --planned-effect-pct 2.5
```

| 字段 | 值 |
|---|---|
| `channel` | `attention-1x16x1024x128-fp16-causal-cuda-event` |
| 5 个 episode 中位 | `0.097472 / 0.097280 / 0.097280 / 0.097248 / 0.097280 ms` |
| `empirical_band_pct` / `effective_band_pct` | `0.0` / `0.1` |
| `planned_effect_pct` | `2.5` ⇒ `action=start`，`predicted_tier=valid` |

**交叉核对**：同一窗口的 L3 候选中位是 `0.09648 ms`，探针 `0.09728 ms`，差 0.8%。
"探针读数必须能被同窗口的 L3 复现"这条判据成立——这是探针没量错的唯一证据。

## 还没做

1. R2 计划里登记 attention 任务（A3）：新编号 + 这条 channel + 2.5% 门槛；本轮只把门建好。
2. `capabilities()` / 支持清单仍只报 4 个算子（A4），设计文档头部还写"未实现"。
3. Scope A 几何仍不校验（A5）：实测 `S=3072`、`B=8/H=8/D=64` 都能静默冻结契约。