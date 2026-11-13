# C5：attention 端到端跑通（2026-11-08）

| 项 | 值 |
|---|---|
| 机器 | `h20`，GPU 3（UUID `GPU-634bbc76…`，与 1 个其他进程共享） |
| 工作目录 | `/root/operator-v08-c5.eJCQQO`（隔离 runner，torch 2.14.0+cu130） |
| 契约 | `556b76336accdb41…`（`builtin-attention-fp16-cuda-v1`，`max_error=3e-4`） |
| 候选 | `attention-constexpr.py`，源码哈希 `90f83ca50d2e…`（与 11-07 那次 1.22e-4 同一份） |
| 结果 | **L1 通过 / L2 通过（8 条 case 全过）/ L3 `degraded`** |
| 证据 | 仓外 `output/v08-attention-20261108/c5/`（两份契约、两次运行目录、三份回执与日志） |

## 1. 回执

| 层 | 结果 |
|---|---|
| L1 | 编译通过 |
| L2 | 主用例 max_abs `1.22e-04`（`atol=3e-4`、`rtol=1e-2`、匹配率 1.0）；**8 条派生 case 首次派发并全部通过** |
| L3 | 候选中位 `0.09648 ms` / 同卡 SDPA 基线 `0.08515 ms` ⇒ **0.8826×**；绑定轴 `compute`（粗估模型 `attention_coarse_v1`，AI=512 vs 脊点 37）；fp16 tensor-core 规格峰值 148 TFLOPS，达成率 **60.2%**；带宽利用率 4.3% |
| 测量有效性 | **`degraded`**：launch 地板占 17.3%（地板 20.24 µs）、最慢样本 2.5× 中位；`can_gate/can_tune/can_report/can_verdict=true`，但**不得禁用配方或累计停滞** |

这就是 C5 的验收条件（"拿到第一份 L3 与绑定轴，且 validity 不是 `uninformative`"），已满足。
**它不是性能结论**：0.8826× 是这一次窗口的观测，且 `degraded` 读数不进入任何达标判定。

## 2. 这一轮改了三处，都是"身份门之后的缺陷"

C5 之前一直停在身份门，所以下面三处一次都没被真机走到。

### 2.1 隔离 runner 被部署模板升级（环境侧）

按档案用 `~/op-agent-delivery/v08-venv` 发起第一次 verify 时，`SSHExecutor.deploy` →
`uv add numpy torch triton` 把工作目录里的 torch 从 **2.14.0 升到 2.14.1**，身份门随即 fail closed
（`native_libraries,torch_config_sha256,torch_git,torch_version`）——这就是
[C5 卡点 §2](c5-identity-blocker-20261107.md) 记的根因，第一次被**主动复现**。

处置：用工作目录里留着的 `torch-2.14.0+cu130` wheel 复原，逐项核对 `git_version=08187d9e0fba…`
与三个 `.so` 哈希；之后所有验证都带 `OPERATOR_AGENT_EXISTING_REMOTE_ONLY=1`（只读既有环境、
不 mkdir/不装包/不覆盖远端文件，不一致就硬停）。**修法 A/B 仍未选**，这条只是绕过。

### 2.2 v2 回执校验里第二处 `inputs["X"]`（控制端）

[review-fixes #1](review-fixes-20261107.md) 修的是**输出设备检查**那一处；同一个分支里还有
`remote_verify["compared_elements"] != inputs["X"].size`。attention 走到"成功"分支时没有 `X` 键 ⇒
`KeyError` 逃出 worker，回执退化成 `worker_failed`（不是"候选错了"，是一行代码假设了输入名）。

修法：该检查的语义是"**参考输出**有多少元素"，不是"第一个输入叫什么"。有本地参考输出就用它；
没有（生产路径 `expected_output` 恒为 `None`，比对全在远端做）则问 profile：
`correctness_profiles.expected_output_elements(spec, inputs)` —— layernorm 取 `X`、attention 取 `Q`，
未知 profile 返回 `None` 由调用方判 unknown，不兜默认。

### 2.3 case 身份两侧不是同一个串（契约与账本）

`BenchmarkCase.key()` 只把**张量**喂给 `case_key`，运行时那侧拿到的是 `materialize()` 的结果
（**含**按值直传的 `causal`）⇒ 契约的 `required_cases` 是
`K=…|Q=…|V=…`，L3 账本的 key 是 `…|causal=?`。`apply_qualification` 按 scope 取不到这一格，
整次 L3 被判 `uninformative`（第三轮就是死在这，数字其实都在）。

修法：`key()` 把标量一起喂进**同一个** `case_key`，渲染规则不另写一份（没有 `.shape` 的值仍写 `?`，
这是既有约定并有测试钉着）。于是两侧逐字相同：`…|causal=?`。

> 边界：`causal=True/False` 在**形状签名**上不加区分（都渲染成 `?`）。正确性套件按 `case_id` 区分，
> 不受影响；workload 账本里"同形状不同标量"仍算同一条 case。要不要把标量值编进身份，是下一步的设计选择，
> 本轮没改。

## 3. 落地的测试

| 测试 | 钉住什么 |
|---|---|
| `tests/unit/v08/test_attention_reference.py::test_the_v2_receipt_gate_accepts_qkv_inputs_without_an_x_tensor` | 没有 `X` 的输入集能过 v2 回执校验（旧写法 KeyError） |
| 同上 `::test_expected_output_elements_is_owned_by_the_profile_not_by_a_key_name` | 输出元素数由 profile 拥有；问不出来返回 `None` |
| `tests/unit/v08/test_case_scalars.py::test_the_case_identity_is_the_same_string_on_both_paths` | `key()` 与 `case_key(materialize())` 逐字相同 |

## 4. 还没做

1. 拿一份 `valid`（而不是 `degraded`）的 L3：地板 17.3% 是 96 µs 核 + 20 µs 地板的结构性结果，
   换 S=4096 主用例或更快的核才会掉到 10% 以下——那是 R2 的取数问题，不是 C5 的。
2. 标量是否应进 case 身份（见 §2.3 边界）。
3. 修法 A/B（部署时 pin `torch==<attested>` vs 重做隔离审计）仍待决定；本轮只做了绕行。