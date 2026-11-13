# C5 卡点：受信任参考的执行身份与远端环境不一致（2026-11-07）

| 项目 | 值 |
|---|---|
| 初次结果 | 默认 runner 在 L2 参考身份检查处停止。隔离 runner 的后续结果见第 6 节 |
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

## 6. 隔离 runner 复测

在 H20 单独创建 runner，用 PyTorch 官方索引的 SHA256 核对 wheel 后安装 `2.14.0+cu130`，
复用现有 NumPy 2.2.6 和 Triton 3.8.0。GPU 3 上的完整参考指纹通过，参考哈希为
`145c6cce7c6a3192…`。四轮都使用同一冻结契约 `33a95e6c538722eb…`。

| 候选 | 主用例结果 |
|---|---|
| 原版 `424d015cea52` | L1 通过，L2 执行时在 Triton 的 causal 分支编译失败 |
| `causal: tl.constexpr`，`90f83ca50d2e` | 编译通过；2,097,152 个元素中 99.9608% 匹配，最大绝对误差 `1.22e-4` |
| 概率拆成两项累加，`4e392b8f86c5` | L2 未通过，最大绝对误差 `2.44e-4` |
| 反向遍历 K/V 分块，`17b40e88412e` | L2 未通过，最大绝对误差 `2.44e-4` |

冻结主用例使用绝对阈值 `1e-5`、相对阈值 `1e-2` 和匹配率 100%。`3e-4` 是注册表的
校准容差上限，当前冻结契约使用了更严的默认值。复测保留冻结判据，8 条派生用例和 L3
因主用例失败而未派发。证据归档在仓外 `output/v08-attention-20261107/c5-attested-20261107/`。

那个"更严的默认值"不是谁选的——它是 layernorm 的标定值漏过来的，见 §7。

## 7. 容差错绑：定位与修复（2026-11-08）

### 7.1 那个 `1e-5` 是从哪来的

它不是为 attention 定的判据，是**全局默认**：

| 环 | 位置 | 行为 |
|---|---|---|
| 1 | `app/intent_parser.py:137` / `:160` | 两条解析路径都硬编码 `max_error=1e-5`，不看 kernel_type |
| 2 | `core/schemas.py:284` | `ConstraintSpec.max_error` 默认同样是 `1e-5` |
| 3 | `decide/correctness_profiles.py` | 挂 attention 的 profile + 参考 + 8 条套件，**唯独不碰 max_error** |
| 4 | `decide/reference_contract.py` | `validate_gate` 只查 `0 < max_error <= ceiling` |
| 5 | `remote_scripts/run_kernel.py:721` | `effective_atol("fp16", 1e-5)` = `min(1e-2, 1e-5)` = **1e-5** 进元素级门 |

`1e-5` 恰好是 **layernorm** 的标定值，而它的 ceiling 也是 `1e-5` ⇒ 默认正好坐在天花板上，
"只查上限"碰巧够用。attention 标定的是 `3e-4`，比默认宽 **30 倍**，门却分不出这两个值：
实测 `1e-5` 与 `3e-4` **都被接受**。注册表里那个标定数全仓只有一个读者，且只当上限用。

**调用方无法纠正**：`max_error` / `atol` / `rtol` 都在 `hitl_tools.FORBIDDEN_KEYS` 里
（"容差是判据，不许由调用方放宽"）。系统不绑就没人能绑。

另外还有第二处错绑：L3 与 dispatch 两条路径把 `1e-5` **写死**在 `validate_gate` 调用里
（它们不做数值比较，只是身份前置门，却没有任务容差可传）。

### 7.2 修法（A：按 profile 相等绑定）

- `MAX_ATOL` → `REGISTERED_ATOL`，语义从"上限"改成"该绑定的值"；
- `validate_gate` 从 `<= ceiling` 收紧为**相等**，并给可归因的拒绝消息；
- 挂受信任参考的那一步（`with_default_correctness`）一并写入该 profile 的标定 atol；
- 新增 `registered_atol(contract)`，供 L3/dispatch 这类不做数值比较的调用点填 atol 槽，
  替换掉两处写死的 `1e-5`；未知 profile fail closed，不兜默认。

**没有**给 `ReferenceContract` 加 `atol` 字段——那会改 `digest()`，等于作废所有已冻结的
layernorm 回执（文件头明确禁止顺手这么做）。

### 7.3 身份卫生：哈希差异只来自容差

同一份归档 `declaration.json` 复算：

| 口径 | `contract_hash` | `max_error` |
|---|---|---|
| 归档那次（C5 四轮用的） | `33a95e6c538722eb…` | `1e-5` |
| 把绑定与门都退回旧语义后复算 | `33a95e6c538722eb…` | `1e-5` |
| 修复后复算 | `984834bd3448882a…` | `3e-4` |

退回旧语义**精确复现**归档哈希 ⇒ 差异只来自容差绑定，其它字段零漂移。
layernorm 侧同样复算过：契约哈希前后**逐字相同**（`f3d65814c726ead2…`），
因为它的标定值就等于旧默认，绑定对它是 no-op ⇒ **已冻结的 layernorm 回执一份都没作废**。

### 7.4 这算不算"数值失败后改判据"

不算，但区分要讲清楚：`3e-4` 是 **2026-11-07 标定并写进注册表的**（见
[attention 设计 §14](../../design/v08/attention-family.md)），**早于**这次复测。
缺陷是绑定没把已标定的值应用上，不是看到失败才去放宽。§4 那条纪律
（"从不在数值失败后自动更新"）针对的是**身份指纹**，本次也没动 `PINNED_RUNTIME`。

标定网格同时证明 `3e-4` 仍拦得住语义错误：错缩放 `1/sqrt(S)` → 0.0134 拒绝、
causal 差一位 → 0.4904 拒绝。"不做 max 平移"在 `3e-4` 下确实拦不住，那是套件第 8 条
`large` 的职责，不是容差的。

### 7.5 由此可推导的结果（尚未上机验证）

远端用 AND 形式：`bad = (abs_error > atol) & (rel_error > rtol)`。
**只要 max_abs < atol，坏点数必为 0 ⇒ matched_ratio = 1.0 ⇒ 通过**。三个候选的实测
max_abs 全部低于 `3e-4`：

| 候选 | max_abs | 旧 `atol=1e-5` | 标定 `atol=3e-4` |
|---|---|---|---|
| `causal: tl.constexpr` | `1.22e-4` | 99.9608% → 挂 | **1.0000 → 过** |
| 概率拆两项累加 | `2.44e-4` | 挂 | **1.0000 → 过** |
| 反向遍历 K/V 分块 | `2.44e-4` | 挂 | **1.0000 → 过** |

三个都是语义正确、只差累加顺序的实现，全部通过符合预期。**这是推导，不是实测**：
主用例过了之后才会派发那 8 条派生用例与 L3，其中 `large` 那条才是真正有判别力的一格。

### 7.6 还剩什么

1. 用新契约（`984834bd3448882a…`）在 H20 隔离 runner 上重跑 C5——需要一次机时；
2. 旧契约 `33a95e6c538722eb…` 归档保留不改，但它已被门拒绝（`max_error` 不等于标定值），
   不再是可执行契约；
3. 8 条派生用例与 L3 这次才第一次真正派发，结果未知。
