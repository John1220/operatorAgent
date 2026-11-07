# C5 卡点：受信任参考的执行身份与远端环境不一致（2026-11-07）

| 项目 | 值 |
|---|---|
| 结论 | C5 **未完成**：L2 在"执行身份"这一步 fail closed，候选核没被执行 |
| 候选 | `output/v08-attention-20261107/attention_kernel.py`（triton，分块 + 在线 softmax + causal mask，fp32 累加）|
| 证据 | `output/v08-attention-20261107/`（`run-c5/verify-1`、`verify-2`、两份 CLI 回包、指纹探针）|

## 1. 实测的指纹

| 项 | 受信任 pin | 远端 runner（`~/operator_agent/.venv`） | 系统 `python3` |
|---|---|---|---|
| torch | **2.14.0+cu130** | 2.14.1+cu130 | 2.13.0+cu130 |
| `torch.version.git_version` | `08187d9e0fba…` | `5c4886908584…` | `cf30153c4c13…` |
| `torch.__config__` sha256 | `41c1a5f5…` | `195c121a…` | `e5799578…` |
| `libtorch_cpu.so` sha256 | `3b720eed…` | `3ed5d130…` | `657a070f…` |
| `libtorch_cuda.so` sha256 | `9ed4b912…` | `20748ec5…` | `181dfc4f…` |
| `F.layer_norm` 源码 sha256 | `b1034c38…` | **`b1034c38…`（一致）** | `b1034c38…`（一致）|

**语义那一项是相同的，漂的是构建**（patch 版本 + 库哈希 + config）。受信任注册表要求整份身份逐字段相等，所以按设计 fail closed —— 这不是误报。

## 2. 根因：部署模板把环境写成了浮动依赖

```python
# operator_agent/execute/remote_executor.py:1082
uv add numpy torch triton        # 没有版本约束
```

于是每次新环境都会装到当时最新的 patch（今天拿到 2.14.1），而 `reference_contract.PINNED_RUNTIME`
是**某次隔离审计**时记下的 2.14.0。两者迟早会分家 —— 今天分了。

## 3. 影响面：不止 attention

这条链任何"受信任参考"任务都会在 L2 前被拦下，包括 **layernorm v2**。
v0.7.2 那次 H20 layernorm 的 `CorrectnessSuite=unknown / execution identity mismatch`
是同一个原因（当时归因为"layernorm 需要自己的校准包"，现在看是环境漂移）。

## 4. 两条修法（需选一条，不能自动选）

| | 做法 | 代价 | 影响 |
|---|---|---|---|
| **A（建议）把环境 pin 回受信版本** | 部署时写成 `uv add numpy "torch==<attested>" triton`，版本号从 `PINNED_RUNTIME` 取（单一真源）| 远端重装 torch（`uv pip install --dry-run` 已验证可解析，只换 1 个包）| 现有 layernorm v2 与 attention v1 两份 profile **继续有效**；可复现 |
| B 重新审计并重 pin | 在当前 2.14.1 上重做隔离审计，更新 `PINNED_RUNTIME`，两份 profile 版本号都 +1 | 需一次完整的 read-only 审计 | 已归档回执的"同一身份"主张要重新表述 |

`reference_contract.py` 文件头写着：身份"由隔离只读审计填充，**从不**在数值失败后自动更新"。
所以这件事不能由我替它决定。

## 5. 已经完成的部分（不受这个卡点影响）

`prepare` 已能冻结 attention 契约（profile、Q/K/V + causal、8 条 case 的套件引用）。
核本身已写好并通过静态审计（`tl.dot`/`tl.max`/`tl.exp` 都在白名单内），只是**没被执行**。

C1–C4 的产出（布局、标量、参考语义、容差、套件、反作弊、roofline）都不依赖这次执行。