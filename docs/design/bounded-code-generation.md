# 有界单候选代码生成：生产接线与验收边界

日期：2026-10-10。接续 [真实参数提案验收](workload-llm-search-acceptance.md)。

## 1. 本次实现与尚未证明的部分

此前模型只能选择 `num_warps`，源码来自可信模板。本次新增生产模式，让模型返回
**一份真实 Triton 源码**，但暂时只做本地接线与守卫验证，没有调用真实模型或 GPU。
不能把 mock 响应、静态预检、fake device 的资格读数称为真机正确性或性能验收。

本次不启用多路采样、自动修复或模板替代；不调整参考身份、精度、容差、正确性
suite、workload 权重、测量资格或独立确认政策。

## 2. 生产 LLM 调用前硬预算

`operator_agent/llm_budget.py` 提供 `LLMCallLimits` / `LLMCallBudget`；生产
`LLMClient(call_budget=...)` 接入它，而不是继续借用上一阶段的 benchmark 专用传输。

- 调用前检查冻结 endpoint/model/默认采样/凭据是否漂移、caller 白名单、调用次数、
  完整 UTF-8 request body 字节、单次输出预留、累计输出预留及剩余墙钟。
- 使用全新 owned journal；先 exclusive-create、fsync 预留，再发请求。旧目录不可
  resume，旧调用或 symlink 不可覆盖；预算声明、身份及已有调用记录漂移即拒绝。
- 失败也占调用格，未用完的输出 tokens 不返还。CostTracker.reset、换 caller、
  Conductor 新一轮执行都不会重置 campaign 额度。并行调用共用带锁账本。
- 有界调用绕开旧 fallback 分支；不重试、不给 reasoning_content 补位，不抽取或
  修补 JSON。网络失败、空/截断/错模型/非法 usage/格式错误、日志写入失败均关闭
  campaign，外层 agent 再试也不会产生下一笔请求。
- 独立子进程只执行一次 HTTP；禁代理与 redirect，限定响应读取字节。父进程超时
  会终止并回收该子进程，不能用慢速滴流不断延长 socket 操作来突破调用墙钟。
- endpoint 只记录 hash，传输凭据只经过子进程 stdin，不进入 argv、请求消息或
  正常账本。HTTP 错误正文和异常原文不进入日志。
- API 完整 usage 与字符估算区分；部分 usage 不冒充完整 token 计数。失败收到的
  响应也保留；没有响应/usage 不等于供应商没有计费。

**硬上限是调用数、字节、客户端墙钟和输出预留**。没有可信 tokenizer，不能宣称
输入 tokens 或供应商账单被精确封顶。`max_tokens` 是给服务器的生成上限请求；
供应商若违约，响应会被拒绝，但不能追溯取消已发生的远端计算或费用。杀客户端
子进程也不等于证明服务端已经取消生成。

父进程 deadline 强制约束的是 HTTP worker 的启动/传输/读取；返回后若解析处理已
越限也拒绝输出，但不宣称本地 journal 的每个 filesystem syscall 都有强制 deadline。

旧的无 `call_budget` 路径保留兼容，仍有 fallback；不能声称全项目默认已切换成
有界模式。另修正 temperature=0 的逐调用覆盖，避免被 `or` 吃掉。

## 3. 单候选 Coder 与 Conductor

`SingleCandidateCoder` 必须绑定全新、生产有界 LLMClient：最多一次调用，caller
仅 `SingleCandidateCoder`。默认输出预留 4096，源码 ≤32 KiB，rationale ≤1000 字符。

执行前冻结 caller 提供的 task/完整契约、参考文本、sketch、完整 workload 的输入
名/shape/dtype/stride，连同 prompt 和生成 policy 保留只读文件。数组元素、实际 scalar
值不进入输入元数据；契约里本来声明的语义常量仍属于可信任务定义。拒绝额外读取
`source.code_path` 文件、第二次冻结同一 campaign，以及调用前后契约/policy 漂移。

模型必须返回严格 `{code, rationale}`，源码按解码后的**原文**使用：

- 不去 fence、不补代码、不修语法、不按 sketch 机械改写、不用规则模板换掉响应。
- AST/语法、明确 kernel 接口、Triton jit、现有 anticheat 做本地剪枝，不 import/exec
  模型源码。第一版限制 import，torch 仅用于分配，拒绝明显动态/宿主操作以及
  autotune、顶层执行语句；防止一份源码藏进额外自动搜索。
- 这是保守静态剪枝，**不是完备的敌对代码沙箱证明**。真正执行还必须通过生产
  AgentRuntime、资源与 exec guard、真实 L1/L2、完整 workload L3、参考身份及精度门。
- 单候选没有性能“选优竞赛”：只能保留自身合格证据或报告失败/degraded，不能补
  候选找绿。静态 prechecked 也不能升级为 GPU correctness passed。

Conductor 显式注入 `single_candidate_coder`，要求匹配同一 LLM、关闭普通 Pipeline、
`max_optimize_rounds=1`。使用冻结 sketch，不调用 Designer 或 Planner；也不选择/记功
未发送给模型的启发式配方。Coder 获得**全部 input_sets** 的元数据，而非只有主 case。

该模式拒绝 resume 和模板参数空间替代，不自动 provision/install。测完后无论 L1/L2
失败、L3 不达标或 degraded，都不进入 Search/Fixer/依赖分析/隐式重试。已有普通
验证、确认预留、独立重测和 corpus admission 仍适用；达到目标也不能省略独立确认。
物理 GPU、RPC/commands、远端清理的上限仍属于执行 backend/验收驱动，**不是**
LLMCallBudget 能替代的机制。

生成失败会保留原始 model-call / candidate journal 并抛错，**不制造通过的 artifact**。
已经进入验证的候选则通过普通 ArtifactStore 保存失败、degraded 或合格的原始结果。

## 4. 生成血缘与普通重载

`audit_generation` 只读核对：冻结上下文/prompt → 真正 request → 原始 response →
严格 JSON 解码的源码 → candidate hash。检查预算、请求参数、reported model、
finish_reason、usage、字节/墙钟边界及 source_author；不发 API、不跑 GPU、不保存或
恢复历史资格。Conductor 在真实验证入口前也执行该血缘检查。

`llm_call_budget` 和 `code_generation` 是显式登记的 artifact 字段，普通重载保留它们
以及 `llm_cost`。后者以前写入 result.json，却未进入普通 load 的 metadata，本次补齐
这个搬运缺口。生成声明还进入 run controls / checkpoint seal。

生产 protocol identity 新增 LLM/client/budget/transport/coder 源码指纹；Conductor/
ArtifactStore 接线也改变了实现身份。因此**新协议 hash 必然不同于旧真机试验**。
旧源码快照和原始证据不改写、不重标、不以当前代码 resume 成新协议 live 资格。

## 5. 使用入口

下面仅是配置示例，不会自行批准任何外发或远端执行：

```python
from operator_agent.llm_budget import LLMCallLimits, LLMCallBudget
from operator_agent.llm_client import LLMClient
from operator_agent.single_candidate_coder import SingleCandidateCoder

budget = LLMCallBudget(LLMCallLimits(
    max_calls=1, max_output_tokens=4096,
    total_output_token_reservations=4096,
    max_request_bytes=65536, max_response_bytes=131072,
    call_seconds=60, campaign_seconds=1800,
), owned_run_root / "llm-calls")
llm = LLMClient(call_budget=budget)  # endpoint/key 来自环境，不写进仓库
coder = SingleCandidateCoder(llm, frozen_task, frozen_sketch, input_sets)
c = Conductor(
    backend=already_provisioned_bounded_backend, llm=llm,
    kernel_db=isolated_kernel_db, failure_db=isolated_failure_db,
    use_coder_pipeline=False, max_optimize_rounds=1,
    single_candidate_coder=coder, budget=oracle_budget,
)
# 真机验收还须由 owned driver 完成外层冻结、授权检查及 finally cleanup。
```

## 6. 验证与下一步

本地测试覆盖预留先于请求、失败不退款、无隐藏 fallback/重试、并行不超支、凭据
隔离、deadline、字节、部分 usage、旧 journal/漂移拒绝、严格代码信封、完整 workload、
失败/degraded 保存与普通重载、只读血缘审计、checkpoint，以及独立确认仍然生效。
fake compile/function/device 仅用于生产接线守卫，不是对示例源码数值正确性的背书。

最终源码的本地验证：

- 新增守卫 **104 passed**；含既有 Pipeline/预算/搜索/写入口/终止原因守卫的专项
  合计 **342 passed**。
- 全量 **3764 passed / 1 skipped / 2 warnings**，约 177.2 秒；两项 warning 仍是
  原有 nonfinite correctness fixture，没有新增 warning。
- 随后串行 Golden：**4 PASS / 0 REVIEW / 0 FAIL / 0 SKIP**；没有修改 registry 或
  默认容差。两个旧真机 run roots 的 **123 个原始文件**校验值未变。
- 最终生产 protocol hash：
  `9cd253e6d1e0fee3c7796ae5e488b7447a5138357a817c61c0ac71d85e9d4689`。
  原 ruler profile hash 保持
  `97dec1cd4d762682d713271146ac1fcd76b4c3b813842247083edd61d9348899`。
- **真实模型调用=0，远端 GPU 派发=0**；没有 commit/push，也没有入库或部署的真实
  验收结论。

日志位于 `output/bounded_codegen_validation/local-0jveylet/`：
`before-final.json`、`targeted-fixed-source.log`、`full-tests-final.log`、
`golden-final.log`、`final-summary.json`。首轮失败日志仍保留，不覆盖成通过记录。

首轮全量暴露两个结构约束冲突：journal 不应 chmod 既有文件打开写窗，新增终止
出口也必须紧邻 stop reason。实现已改为 0600 新 inode 的 exclusive link 发布，并
调整 stop/snapshot 顺序；终止出口绝对锚点从 15 明确登记为 16，同时保留逐出口的
精确 reason 映射。不移除守卫、不调整 Golden 期望或容差；失败日志保留供复盘。

下一步是新增**单候选真实代码生成的默认 plan-only H20 验收驱动**，冻结全部源码/
契约/原 ruler profile、单请求载荷和 GPU/RPC/时间/清理边界，再请求新的明确授权。
上一轮“最多两次参数提案”的批准已用完，不能用于新增自由代码生成请求或 GPU 试验。
真实验收须保留成功、失败、degraded 的原始数据，不为了找绿加模型请求、候选或补测。
一次修复仍是后续独立阶段，不在本轮偷偷启用。

### 后续：单候选 H20 驱动已完成本地接线

新增默认 plan-only 入口 `benchmarks/workload_codegen_h20.py`，具体冻结、授权绑定、
原始/持久化审计及未覆盖的远端隔离边界见
[单候选模型代码生成 H20 验收驱动](single-candidate-codegen-h20-acceptance.md)。
专项 284 passed，其中新驱动 93 项；不沿用上一轮参数提案授权，不外发模型请求或
执行模型生成代码。真实代码生成验收仍待单独批准，不能由本地 mock 结果补成通过。
项目全量 3857 passed / 1 skipped，串行 Golden 4 PASS；生产 protocol 与旧真机
123 个文件保持不变。本轮也没有 commit/push。

### 后续真实执行：准备失败，未进入模型生成

2026-10-10 用户单独批准计划 `20e78e8edc20` 后，执行目录
`output/workload_codegen_h20/workload-codegen-6ar1c1kx/` 在准备阶段停止：
`preparation_failed`，两条准备 command，模型调用/RPC/候选验证/测量窗口均为 0，
scratch 未创建。没有重试、安装、入库或 commit/push；失败证据与执行 claim 保留。
现有回执只留异常类型，无法确认是 GPU 查询失败还是指定卡不满足资格；下一步先补
有界脱敏的准备诊断，再声明新计划并重新批准，而不是续用本次批准找绿。
详情与清理回执的解释见
[首次人工批准执行记录](single-candidate-codegen-h20-acceptance.md#8-首次人工批准执行准备失败不重试)。

### 后续本地修复：准备回执与严格清理分类

新增固定七阶段、最多 15 个只读文件的脱敏 preparation journal；保存可取得的
返回码和结构化 GPU 资格信息，区分查询失败与指定卡不满足条件。
早于 scratch 创建的失败只有在回执一致、没有 runner/候选副作用时才可标记
`cleanup_status=not_needed`；创建已尝试但回执丢失仍为 unconfirmed，不伪造移除。
零调用模型状态单独审计为 not_started/refused_before_dispatch，不补出候选。
这些分类不会放宽正确性/性能/确认/入库门槛，也不会把早期准备失败改成验收成功。
实现和故障注入测试见
[准备诊断修复](single-candidate-codegen-h20-acceptance.md#9-准备阶段诊断与清理分类本地修复)。
最终专项 471 passed（新测试 71 项）、全量 3928 passed / 1 skipped / 2 原有 warnings，
随后串行 Golden 4 PASS；生产 protocol、原 ruler/Golden registry 和 304 个历史
文件不变。该本地修复阶段结束时，新计划 `3d71c64785d9` 仅本地冻结，未批准/未执行；真实请求与远端派发
均为 0，没有 commit/push。

### 后续真实执行：准备通过，生成源码静态预筛失败

2026-10-11 用户明确批准新计划 `3d71c64785d9` 后，真实执行目录
`output/workload_codegen_h20/workload-codegen-02rd6d4j/` 完成七阶段准备和参考读回，
发送且仅发送一次冻结的 11789-byte 请求。模型正常返回完整 JSON 源码，usage 为
3981 输入 + 413 输出 tokens，但 Triton JIT 与 Python 包装入口都定义成 `kernel`，
被唯一入口静态守卫拒绝；原文保留，budget aborted，不修复、不重试。

结果为 `generation_failed`，accepted=false；Verifier、候选编译/数值验证、主测/
确认/心跳窗口均为 0，因此不是 GPU 正确性或性能验收结论。4 个 RPC 均为元数据/
参考身份读取，work command 计数 18。owned runner/scratch 已确认 quiescent +
removed，没有共享进程信号、安装或入库。

事后仅本地读回并对原始源码复现同一静态守卫，不执行候选、不请求 API/GPU。
失败证据、304 个历史文件、生产 protocol、原 ruler 与 Golden registry 不变；
失败事实核对完成不等于正向验收完成。下一步先离线明确 helper/入口命名与派发规则，
补有界静态诊断和回归测试；不改名重测当前候选、不续用已消费批准。详情见
[新计划人工批准执行记录](single-candidate-codegen-h20-acceptance.md#10-新计划人工批准执行准备通过源码静态预筛失败)。

### 后续离线修复：入口/helper 命名与固定失败诊断

提示显式区分唯一、无装饰器的 Python `kernel` 入口与不同名的 Triton JIT helper，
要求调用 helper 的 launch 而不是包装入口；不提供实现模板、不改写模型源码。
静态门保留并补强入口装饰器、顶层重绑定和直接自 launch 的拒绝。失败诊断只保存
固定的 `stage/reason` 两字段（最多 256 bytes），不保存任意异常文本或类名；原始
响应/源码仍独立保留。诊断不能使 aborted campaign 通过，也不能恢复 eligibility。

新增 74 项离线测试，包括上一轮真实源码的逐字文本夹具；始终拒绝，不执行候选。
最终专项 427 passed，全量 4002 passed / 1 skipped / 2 原有 warnings，随后串行
Golden 4 PASS。671 个历史文件、原 ruler/Golden registry 不变；没有 commit/push。
生产 identity 因 coder 源码改变更新为 `8bc9ff3330790e63b656dda8bf9ed9a74007f24ddf5e5f710a2442b474544e7d`，
不把旧失败结果改签成新协议事实。数值、workload、目标、预算和远端隔离边界未放宽。

新计划 `4d460efbe3dd` 只在本地冻结：85 份源码、12077-byte 请求，尚未批准/执行，
真实模型请求/远端派发为 0。真实验收须再次单独批准，详情见
[离线命名和诊断修复](single-candidate-codegen-h20-acceptance.md#11-离线修复入口helper-命名与固定失败诊断)。
