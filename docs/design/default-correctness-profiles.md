# 首批算子的可信默认正确性配置

日期：2026-10-08。对应 §20 的入口采用与真机链路验收，不宣称完整算子语义/§20 已完成。
实现：`correctness_profiles.py`；有限生成规则仍见
[自动边界生成](correctness-boundary-generation.md)。

## 1. 新入口默认采用，旧入口不强制迁移

正式自然语言入口 `IntentParser.parse()` 的规则与 LLM 分支，统一经过
`with_default_correctness(task)`。这份可信注册表不接受模型返回的 coverage policy、
case 列表、预算或关闭开关；LLM 只解释任务标签，不替 harness 定义量尺。

```python
from operator_agent import IntentParser, TestInputGenerator, Conductor

task = IntentParser().parse("实现 fp32 matmul，使用 Triton")
# constraints.correctness_profile = "builtin-l2-v1"
# correctness_suite = {"coverage_policy": ...}，由既有入口确定性展开。
inputs = TestInputGenerator.generate(task, matrix_dim=(128, 128, 64), seed=11)
# Conductor(backend=...).execute(task, inputs)
```

程序化新入口可显式调用 `with_default_correctness`。函数返回副本，不修改原任务，
不覆盖调用方已有的手工 suite。**直接构造 TaskSchema 的旧调用方仍保留原行为**；
`correctness_profile=None` 不写入旧任务/契约内容，旧 hash 不变。

注册表按精确算子标签选择，不从候选源码、BLOCK 或性能输入猜新域。
自然语言规则中的 GEMM 别名会解析成 matmul；程序化 `gemm` 标签本版没有 profile。
英文关键词改为词边界匹配，不能把 `custom_softmax` 当作普通 softmax。

## 2. 第一版明确支持的语义

共同要求：PyTorch 内建参考；正整数二维输入；所有输入与任务声明 dtype 一致；
不含融合、外挂参考、源文件或自定义语义参数。

| 算子 | 输入/参数语义 | dtype | L2 采样子域 | case / 覆盖规则 |
|---|---|---|---|---|
| ReLU | X[rows,cols]，逐元素 max(x,0) | fp16/fp32 | rows=[1,9]，anchor=8，alignment=8；cols=[1,33]，anchor=32，alignment=32 | 每精度 12 / 21 |
| Matmul | A[M,K] @ B[K,N]；共享 K | fp16/fp32 | M/N=[1,33]，anchor=32，alignment=32；K=[1,5]，anchor=4，alignment=4 | 每精度 15 / 27 |
| Softmax | X[rows,cols]，末轴归一化 | fp16/fp32 | 同 ReLU | 每精度 12 / 21 |
| LayerNorm | X[rows,cols]，末轴、无 affine、eps=1e-5 | **仅 fp32** | 同 ReLU | 12 / 21 |

每个 profile 只测**目标精度**，不让一个仅承诺 fp32 的核自动承担 fp16。
五种值模式 random/zeros/constant/large/small、连续/真实 stride、确定性要求均保留。
Matmul 的大值=100、anchor K=4 得到 40,000，在 fp16 中有限；这是明确登记的小采样
子域，不是缩小调用方性能 workload，也不声称长 K/全部归约长度已覆盖。

主输入独立验证，不受采样盒的上界限制：允许更大的合法二维输入，但必须满足输入名、
rank、dtype、共享轴关系，不能把不一致 K、weight/bias、自定义 eps 等参数悄悄忽略。
`TestInputGenerator` 对新 profile 的三个单输入算子统一生成 `X`，支持矩形主输入；
旧任务的 `input` 命名规则不改。L3 仍使用原主输入，自动 suite 不进入性能 case 账本。

### 不支持就 unknown，不退回无 suite 的旧路径

- fp16 LayerNorm：本机 Torch 2.8 CPU 的单列参考对一个数学上应为零的输出给出约
  `-2.38419e-5` 残差，超出本任务 1e-5 绝对项。2026-10-08 已完成
  [独立 CPU/H20 标定](fp16-layernorm-calibration.md)：同一 Torch 2.14 中仍有 6 组
  CPU/CUDA 差异触发原门限；关闭浮点融合的固定 Triton 候选在 H20 32 组通过，
  但参考身份/精度语义尚未冻结，v1 仍明确拒绝该组合，不自动启用；
  **未放宽容差、删除单列样本、替换参考或把失败改成通过**。
- bf16/int8、attention/conv、自定义名称、融合、外挂参考、其他框架/源文件：没有可信
  内建配置就保留 requirement marker，前置返回 L2 unknown。
- `SourceSpec.semantic_parameters` 留存未解决参数。入口保守检测明确的 eps/axis/dim/
  affine/转置/融合等请求，不能因为 LLM 没输出该字段而把它们当作普通变体；
  `log_softmax` 也不能当作 softmax。该检测**不是完整自然语言语义解析器**，
  验证仍针对 TaskSchema 声明的语义，不证明任意自然语言请求被完整理解。

已有可信手工套件/外挂语义应走显式 task + suite 路径，不借内建 profile 冒充支持。

## 3. 冻结、执行和报告

`ConstraintSpec.correctness_profile` 与完整展开 suite/policy 同时进 FrozenContract hash。
冻结与载荷重建检查 profile 版本、算子、dtype、固定 policy 身份和覆盖，未知版本不自动升级。
已声明 marker 时移除 suite、删模式/布局/dtype、修改 policy 或关闭确定性均拒绝。

预检查由 Verifier 普通入口、私有 L2 入口、L3-only 续接入口及 Conductor 消费。
主输入检查在 Conductor 的输入解析前后都保留；契约自造输入也不能绕过。
缺口在编译/路由/LLM/provisioning 前变成 unknown/HOLD，不调用 GPU，不沉淀候选教训。
前置拒绝仍直接返回无候选 artifact，不走正常候选保存；调用者可渲染报告。

报告同时显示 profile 的**声明检查**与实际 L2 verdict/回读，二者不能混读。
实际 L2 数值、原值回读、确定性、L3 资格和独立重测/入库门均保持原判据。

## 4. H20：真实生产链路正反验收

入口：`python3 benchmarks/correctness_profiles_h20.py --gpu 2`。
不调用 LLM API；只替换 Designer/Coder 的候选生产，**不替换 Verifier、Conductor 主循环、
性能测量、确认、晋级或报告**。使用既有运行时、随机 0700 runner、共享源码 SHA 校验，
拒绝环境安装/sync/execute 等操作；数据库与日志定向到私有运行目录，不污染共享语料。

2026-10-08，H20 GPU 2 / 实际 cuda:0，Torch 2.14.0+cu130 / Triton 3.8.0：

- **26/26 符合预期**：7 passed、16 failed、3 unknown；不是 26 个核均通过。
- 7 个好核（四个 fp32 + 前三个 fp16）执行 **90 个自动 case**、逐 case 原值回读和两次
  完整输出指纹；全部 L2 passed，并进入正常 L3 与独立确认。
- 14 个错误 dense/tail 核 + 2 个 fp32 错误精度核：对齐、整数主输入通过，而有限套件
  在实际 CUDA 上判 failed；**无 L3 请求、无正向持久化**。负例所有访存仍有合法 mask。
- attention fp32、ReLU bf16、LayerNorm fp16 在编译前 unknown，无候选执行或 L3。
- **565 次 verify / 565 份实际 CUDA 回读、23 次 compile、14 次真实基线配对性能请求**。
  7 次确认均跑过；但小张量性能读数 `can_gate=False`，7 个好核的正常入库门全部拒绝。
  不能据本验收宣称速度收益、测量可信或正向语料 admission 已成功。
- 小主输入固定 ≤32 KiB；请求留痕保存 wire 长度/哈希，不重复保存大 base64 缓冲；
  逐试验释放 trace，最后确认 runner quiescent、隔离目录 removed。

成功运行编号：`correctness-profiles-h20-20261008T090335752990Z-65336-1`。
完整证据在 ignored `output/correctness_profiles_h20/<编号>/report/summary.json`。
不提交真实主机/用户路径。

**早先运行的有限失败也保留**：`correctness-profiles-h20-20261008T085319936767Z-60741-1`
完成前两项后，第三项的大主输入在常驻客户端 `stdin.write` 处长时间等待。
中断后自动 cleanup 当时未确立 quiescent，保留目录；随后核验**精确 PID、命令路径、属主、
目录 0700**，只终止本次 runner 并确认删除，独立读回记在 `report/cleanup-followup.json`。
最终运行使用小型链路对照，**没有改 profile、容差、参考或性能判据**。
这暴露了大载荷写入未纳入请求等待截止时间的边界；本轮不宣称完整大输入传输/超时问题已修。

**同日后续**：[常驻 SSH RPC 截止时间与 owned-runner 清理](ssh-rpc-deadline-and-cleanup.md)
已独立接通，并完成 H20 1 MiB / 16 MiB 原值、限时停读/迟到对照和精确清理 6/6 验收。
这补的是常驻 RPC 的发送/等待与恢复边界，不是任意大输入、完整网络故障或 GPU watchdog。

## 5. 本地守卫与后续范围

`tests/test_correctness_profiles.py`：规则/LLM 默认采用、模型不能选择尺子、显式域、
hash/载荷往返、旧载荷、语义/主输入不匹配、suite 移除/漂移、unknown/HOLD、
三个 Verifier 入口、Conductor 正常保存、性能 workload 不替换与报告。
`tests/test_correctness_profiles_h20_acceptance.py`：离线标定驱动，不冒充 GPU 证据；
不完整/CPU 回读、错 hash、缺 case/repeat、漏 L3、错误核计时/入库等必须拒绝。

本地验证（Python 3.9.12 / Torch 2.8.0 CPU）：

- 本轮新增 **81 项守卫**（profile 65、H20 驱动 16）。
- 全量 pytest：**2918 passed / 1 skipped / 2 warnings**。跳过项需本机 CUDA；两条 warning
  是上一阶段刻意非有限参考的 oracle 差值，仍验证 unknown，不放宽或删除该负例。
- 全量和变异完成后串行 Golden compare：**PASS 4 / REVIEW 0 / FAIL 0 / SKIP 0**；
  原有 MockGPU 回归基线、容忍窗口和计时代码均未修改。
- 在临时副本中逐项标定 **6/6 关键变异被捕获**：跳过正式入口默认采用、跳过 profile
  预检查、profile 不进 hash、放过 policy 漂移、跳过主输入语义检查、移除 L3-only 预检查。
  不是完整变异覆盖率，也不是 GPU 证据。

仍欠：全部旧入口迁移、完整自然语言语义落地、任意参数/合法域、bf16、fp16 LayerNorm
数值标定、NaN/Inf/梯度/aliasing/sanitizer、XPU/HTTP、GPU launch 探针、L3 原值保真、
正向入库真机对照。常驻 SSH 大载荷写入截止时间由上述后续方案补齐，完整 RPC 生命周期/
任意传输故障仍未覆盖。此轮是有限采用与链路验收，不是完整 §20 收口。
