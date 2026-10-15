# 单候选模型代码生成：H20 验收驱动

## 1. 本轮边界

入口：`benchmarks/workload_codegen_h20.py`。

这是上一阶段生产 `LLMCallBudget + SingleCandidateCoder + Conductor` 的验收驱动，
不是扩展成多候选搜索。模型只能生成**一份完整 Triton 源码**；保存解码后的原文，
不经过模板替换、机械改写、修复或第二次采样。以前两次 `num_warps` 提案的真实证据，
不能用来证明模型已经写过代码，也不能提供本次外发/执行授权。

**第 7 节记录本地实现和守卫测试；该阶段真实模型请求及远端 GPU 派发均为 0。**
测试中的模型响应、CUDA 标签、时间、RPC/参考回执有明确的 offline/mock 标记，
只能证明接线和审计器的行为，不能证明数值正确、真实 CUDA 身份或 H20 性能。
随后经人工批准的首次真实执行在准备阶段失败，事实与边界见第 8 节。
第 9 节记录随后完成的准备诊断修复；它不回填第 8 节的未知原因或失败结论。
第 10 节记录新计划获批后的真实执行：模型返回源码，但静态预筛失败，未进入
候选 CUDA 编译、数值验证或性能主测；仍无正向代码生成或部署验收结论。
第 11 节记录随后完成的离线命名/诊断修复，不覆盖第 10 节的候选或失败证据。

## 2. 默认 plan-only 与执行资格

默认执行仅创建新的 0700 本地目录、只读 `plan.json` 和完整源码快照：

```bash
python -B /Users/zh/repos/operatorAgent/benchmarks/workload_codegen_h20.py
```

不读取项目模型配置，不使用隐式模型/端点环境变量，不分配两个大数组，不创建
模型 journal，不访问 HTTP、SSH 或 GPU。没有显式模型时，`model=null`，不是“端点已核验”。

准备可供人工审阅的具体计划，可显式指定 `--model-url`、`--model`、`--host`、
`--gpu`、`--cuda-ruler-profile`，但仍不加 `--execute`。只有 key 是否存在进入计划；
key 仅允许从 `LLM_API_KEY` 获取，不进入 argv、载荷正文或产物。端点在证据中留 hash，
实际外发地址必须另外向用户明确披露，不能用 hash 代替授权说明。

执行必须同时提供 `--execute --approved-plan <path> --plan-hash <exact-hash>` 和上述
全部显式配置。驱动要求当前完整声明与被批准的计划逐字语义一致，验证计划 hash、
当前源码、只读源码副本、契约和 generation context 后，才允许外部动作。
**这些 CLI 参数只是绑定条件，不是工具自行授予的权限；仍需人类单独批准。**

执行前在准备目录 exclusive 发布 `execution-claim.json`。一个计划只准消费一次；
失败、取消、断电都不退回资格，不覆盖 dangling symlink，不把旧 journal 当可续额度。
执行另建新目录，不覆盖准备目录和旧真机产物。失败后若要重试，应重新声明、审阅和授权。

## 3. 冻结内容与预算

### 不改判据

- 保留已注册的 native LayerNorm fp16 CUDA v2 参考、精度声明、全量 32-case 正确性
  suite、布局/边界 lint、数值容差及独立确认机制。
- 两个主 workload 固定为 `(8192,15360)`、`(8192,12288)`，等权几何均值，逐 case
  5% 回退上限；不按模型生成代码、BLOCK 或第一次成绩调整形状、权重。
- `min_speedup=10` 在调用前冻结，表示本阶段的接线目标，不是部署承诺。不能为了
  入库把它改为 1.05；如果意外通过，仍走生产独立确认/入库门，不能跳过确认。
- 原 ruler profile 由真实只读文件校验，不接受新的 floor 标量，也不在线重标定。
- 固定 caller-owned sketch 仅有语义提示，不含可信候选模板；发送内容为冻结契约/
  已注册参考定义、suite 声明、sketch 及输入名字、形状、dtype、strides。
  不发送输入元素、标量值、其他项目文件、上一轮反馈或模板源码。

### 硬限制与记账必须分开

| 项目 | 本次计划 |
|---|---|
| 模型请求 | 最多 1，caller=`SingleCandidateCoder`，失败不退款 |
| 输出额度 | 预留 4096 tokens；无第二次请求、fallback、redirect、格式修复 |
| 完整请求 / 原始响应 | 64 KiB / 128 KiB |
| 生成源码 / rationale | 32 KiB / 1000 字符 |
| 模型 HTTP worker | 最多 60 秒；campaign dispatch deadline 1800 秒 |
| 候选与优化轮 | 1 份源代码、1 轮；Planner/Search/Fixer/多 finalist 禁用 |
| 编译 / 数值验证 | 至多 1 次远端编译；两个 primary 加原 suite 的 32×2 次验证 |
| 主性能 / 确认 | 每个 case 主测 1 窗；仅原生通过路径允许每个 case 确认 1 窗 |
| 心跳 | 最多 2 次 native heartbeat，无新 calibration |
| 远端传输 | 512 RPC / 512 commands；180 秒每 RPC，1800 秒 dispatch deadline |
| owned cleanup | 独立 60 秒 allowance；必须确认 quiescent 与 removed |
| Oracle 记账 | 总池 4：主测、确认与两次心跳的空间；确认预留仍不可挪用 |
| 旧 token 估算记账 | 40000，仅诊断，不是本次模型硬请求/账单上限 |

Oracle 校准份额仍按生产规则切分：心跳该跑就跑并记录 soft over-plan，不改成预算
饿死尺子。候选固定一轮，所以 Oracle 余量不能换成第二个候选。确认 RPC 不重新
跑 L1/L2，窗口不同不等于新候选。

输入单包 storage+scalars 严格小于 256 MiB，两个大数组分别 240/192 MiB；顺序传输，
缓存被逐出后的有证据重传是合法行为。**这不是控制端 RSS、远端 RAM 或 VRAM 总上限。**
硬 deadline 约束外部派发和 worker，不声称所有本地 fsync 都有严格墙钟界，也不证明
HTTP 超时后服务端已取消计费。

## 4. 请求与 GPU 派发前守卫

`PlanLLMClient` 继承真实生产客户端：最后构建出的完整 HTTP body 和 endpoint/model
身份必须与批准计划一致，检查发生在 reserve 和 HTTP 之前。随后仍使用生产 durable
reservation、隔离 HTTP child、严格原始 JSON/usage/finish_reason 解析。

`CodegenSSH` 继承既有 bounded owned-scratch 后端：每次 command/RPC 先检查冻结计划，
candidate compile/verify/benchmark 再核模型响应→原始源码的生产血缘。计数先于派发，
失败不退款；检查精度、参考、容差、性能协议参数、case 顺序、编译/数值验证/窗口上限。
cleanup 故意绕过已经损坏的计划守卫，避免“发现漂移→不能收拾自己启动的 runner”。

Verifier 只允许一个候选派发。结果需要保留到 Conductor 附加运行绑定之后，不能在
Verifier 返回时提前 deepcopy，制造一份缺 binding 的“原始结果”。最后读回已保存
`evidence.json` 与现场完整证据比较，缺失、撕裂、越界计数均不能通过。

## 5. 只读审计与结果分类

- `audit_model`：独立读 production journal、冻结请求/身份、原始响应、generation
  declaration、exact source lineage；从 raw usage 重建成本估算，与保存摘要比较。
  缺字段、额外调用、响应/source 置换、symlink、成本改写都拒绝。
- `audit_raw`：检查一份候选的真实 CUDA compile、完整 suite/repeat、primary、逐 case
  实际 RPC/时间比/参考回执/输入传输、native ruler/心跳、新鲜窗口；重算既有
  full-workload 资格与目标。主 case GFLOPS 不能替代完整 workload。
- `audit_durable`：普通 ArtifactStore reload + 生产 sealed checkpoint reader，核对
  源码、sketch、generation/journal/usage、验证与 benchmark、search selection/history、
  终止原因及报告。失败路径可能没有 corpus_admission 键；不得把缺键补成通过。
  ArtifactStore 原有普通重载不保留 error_log；本审计不假装恢复它，原始诊断保留在
  execution/evidence 中，匹配已有持久化的 level/passed/message/metrics。

审计函数不调用 API/GPU，不保存或修补被审计产物，不 resume，不恢复 eligibility。
`completed` 表示该项证据检查完成，`accepted` 才是本阶段正向验收，两者分开：

- **valid_below_target**：完整正确性及 VALID 可选 full-workload 证据，保存/重载一致，
  owned cleanup 完成，可证明单候选链路通过；不代表部署目标通过。
- **valid_target_passed**：除上述内容，必须核独立确认；实际入库还需 sealed corpus
  fresh replay，才允许独立的 `deployment_accepted`，仍仅限声明的两个 case。
- **degraded / uninformative / correctness_failed**：留存并审计失败事实，不接受、不
  重测、不修复、不放宽阈值。生成失败时保留 journal，不伪造 KernelArtifact。
- **preparation_failed / generation_failed / incomplete**：保留已发生动作和清理状态；
  missing receipt、raw drift 或未确认清理不会被汇总为成功。

全局 `completed` 必须包含模型/raw/durable 正向检查以及清理，不能用静态 precheck
或 exit code 的绿色替代真实数值、测量和副作用验收。

## 6. 尚未具备的安全能力

当前远端 runner 仍是同 UID import/exec，静态代码预筛不是敌意代码沙箱。
**没有**独立 UID/container/cgroup、网络硬关闭、文件系统 allowlist、RSS/VRAM quota、
GPU watchdog 或独占设备 lease。请求时限和 owned shutdown 也不能证明卡级故障已恢复。
因此不能把生产本地 `exec_guard` 的 rlimit/进程组能力说成远端已覆盖。

真实试验必须重新披露这个边界，获得对“模型生成源码在指定卡/owned scratch 执行”
的明确批准；上一轮可信模板参数试验的权限不能升级到任意代码试验。本阶段既不
自动执行真实试验，也不安装新沙箱、编辑共享 runtime 或杀共享进程。

后续若面向无人值守或不可信模型，优先补远端候选进程隔离与可观察的资源强制，
而不是直接开放修复、多路采样或多轮搜索。

## 7. 本地验证

新增 `tests/test_workload_codegen_driver.py`：plan-only 零外部动作、批准计划 hash/
目的地绑定、source snapshot 漂移、exclusive 消费、预请求/派发守卫、journal/usage/
产物篡改、失败/degraded、原始 suite/窗口重构、独立确认、准备失败和清理失败。

本轮验证日志与冻结快照：`output/workload_codegen_validation/local-dqq2s2af/`。
专项最终 **284 passed**（其中本驱动 **93** 项），包含生产 hard-budget/single-coder
及旧搜索验收驱动守卫；首轮失败日志保留，没有改期望、删守卫或调整 Golden 容差。
项目全量 **3857 passed / 1 skipped / 2 warnings**，185.21 秒；两个 warning 仍来自
原有 nonfinite correctness fixture。随后串行 Golden **4 PASS / 0 REVIEW / 0 FAIL /
0 SKIP**，未改 registry/容差。详情见本目录 `full-tests.log`、`golden-compare.log` 与
`final-summary.json`。
全量命令明确为 `python -B -m pytest -q -p no:cacheprovider tests`：首次未指定项目
`tests` 目录，误收集 vendored Triton 的 GPU/插件测试，产生 76 个 collection errors；
原日志保留为 `full-tests-including-vendor-collection.log`，没有为此安装组件或修改 vendor。

生产 measurement protocol 保持
`9cd253e6d1e0fee3c7796ae5e488b7447a5138357a817c61c0ac71d85e9d4689`；
本轮只增验收 driver/tests/docs，driver 本身另进 evaluator_sources 冻结身份。
旧真机 123 个文件、Golden registry 与原 ruler 不覆盖、不重新标记成新代码生成证据。

本地验证结束时，为原端点/模型和 GPU 2 准备了一份具体只读计划（当时未批准/未执行）：
`output/workload_codegen_h20/workload-codegen-rcxfg6ph/plan.json`，hash
`20e78e8edc209de43cde3d790d6e6ec01dae61a8ca851b9fa6993e19069401bd`。
其中源码快照 84 文件，完整请求 11789 bytes，credential_present=false，ruler hash
`97dec1cd4d762682d713271146ac1fcd76b4c3b813842247083edd61d9348899`。
`prepared-plan-context.json` 仅登记当时的 plan-only 状态，不是已取得许可的凭证；
该记录保留原样，后来的批准与执行另记如下。

## 8. 首次人工批准执行：准备失败，不重试

2026-10-10，用户明确批准计划 `20e78e8edc20` 的单候选有界验收，并接受已披露的
同 UID、无敌意代码沙箱/网络硬隔离/显存 quota 限制。此次批准不增加原计划的请求、
候选、窗口或时间额度。执行前重新核对完整计划、84 份源码快照、请求正文、原 ruler
与旧真机文件，均一致；在任何外部动作之前 exclusive 消费 execution claim。

真实执行目录：`output/workload_codegen_h20/workload-codegen-6ar1c1kx/`。
授权回执、执行日志与事后只读审计另存于
`output/workload_codegen_validation/authorized-lchvok3h/`，不覆盖本地验证的历史摘要。

结果为 **`preparation_failed`，exit code=1，accepted=false，deployment_accepted=false**：

| 实际动作 | 数量 / 状态 |
|---|---|
| 准备 command 派发计数 | 2 |
| 模型请求 / 输出 token 预留 | 0 / 0 |
| RPC / Verifier 候选派发 | 0 / 0 |
| 候选生成 / 编译 / 数值验证 | 均未发生 |
| 主测 / 独立确认窗口 | 0 / 0 |
| runner / scratch | 未启动 / 未创建 |
| 重试、fallback、安装、共享进程操作、入库 | 均未发生 |

原始错误仅保存 `execution stopped: RuntimeError`。根据冻结准备流程与两个 command
计数，只能推断停止于 runtime probe 之后的 GPU 查询或指定卡资格检查，早于第三条
scratch 创建命令；**未保留查询 stdout/stderr 或 GPU snapshot，无法认定具体原因**。
不能将它直接写成“GPU 2 正忙”，更不能伪造一份 CUDA 测量结果。

cleanup 原始回执是 `{quiescent: true, scratch_created: false}`。冻结的通用清理
审计器要求 `removed=true`，因此 summary 另报 `owned runner/scratch cleanup not
confirmed`。这并不证明有遗留 scratch 或进程：创建动作尚未派发，没有可移除对象。
此次仍按原驱动判失败，不回填 removed、不改 summary 或审计门槛。缺少 generation
context/candidate 的模型审计也没有通过，符合未开始生成的事实。

事后审计仅本地读回：计划/claim、实际计数、零调用 journal、冻结源码与所有失败
证据文件一致；旧真机 123 文件、Golden registry、生产 protocol 和原 ruler 不变。
`post-run-audit.json` 的 `evidence_check_completed=true` 仅表示这些失败事实已核对，
**不等于 campaign completed 或验收成功**。没有补发模型请求、再查远端或补测找绿，
也没有 commit/push。

下一步优先补**有界、脱敏的准备阶段回执**：区分 runtime probe、GPU 查询、指定卡
资格检查与 scratch 创建；保留有限的状态/返回码和结构化 GPU 资格信息，不能直接
落盘任意异常文本、凭据或模型源码。先做离线故障注入测试，再冻结新计划并单独请求
人工批准。此次 execution claim 不删除、不续用，即使模型额度实际未消耗也不退款。

## 9. 准备阶段诊断与清理分类：本地修复

新增 `benchmarks/_preparation.py` 的 `PreparationJournal`，并在共享 acceptance
后端加入默认无副作用的 `_prepare_stage` 接点。旧驱动仍使用原来的 transport、
选择算法和调用顺序；单候选驱动显式开启新 journal，不改生产 measurement protocol。

固定七个阶段：`runtime_probe → gpu_query → gpu_select → scratch_create →
source_upload → source_readback → runner_readback`。每阶段开始回执必须在操作之前
持久化，结束回执另建只读文件；阶段只能依序消费一次，失败、取消或结束回执丢失均
不能重试。派发仍计入原 RPC/command/deadline 上限，没有新增远端查询或探针。

### 9.1 有界且脱敏

- journal 使用新的 0700 目录；最多 15 文件（policy + 七组 start/finish），每文件
  最多 4096 bytes、发布后只读。读回拒绝缺失、额外、篡改、symlink 和超限文件。
- 只保留固定阶段/状态/原因码、前后 transport 计数和经过时间；不保存 argv、原始
  stdout/stderr、任意异常文本或自定义异常类名。
- 可由 SSH/SCP 异常取得的整数返回码才保存；否则为 `null`，**不猜测 rc=0**。
  timeout、interrupted、ssh_command_failed、gpu_ineligible 等原因使用固定词汇。
- GPU 查询回执至多解析 4096 bytes、8 行，拒绝越界/重复 ID、无效显存/利用率等；
  只保存 index、free_mb、utilization、is_h20，不保留远端提供的名称文本。
  GPU 选择仍要求原指定卡、利用率 ≤5%、空闲显存 ≥1024 MiB；非 H20 在创建
  scratch 前拒绝，不换卡、不放宽条件。

上述字节界限约束**诊断解析和持久化**，不是现有 SSH transport capture 的内存
强制上限，也不是新增 RSS/VRAM quota、敌意代码隔离或设备独占保证。

### 9.2 清理状态不等于验收资格

原始 cleanup 回执不改写，另加 `cleanup_status`：

- `removed`：quiescent=true 且 removed=true，确实完成 owned 清理。
- `not_needed`：只读 journal 一致且**未开始 scratch_create**，本地无 scratch/
  active 或 pending session/deployed runner，没有 RPC/compile/verify/bench 派发，
  原始回执为 quiescent=true、scratch_created=false。不是回填 removed=true。
- `unconfirmed`：其余情况一律不确认。尤其是 scratch_create 已开始但 SSH 回执
  丢失，可能远端已创建目录；本地 scratch=None 也不能当作“无需清理”。

`not_needed` 仅消除早期准备失败的清理误报，**不会使 campaign 成功**。
整体 completed 仍要求模型→真实 CUDA/raw→durable 正向验收和真实 removed 清理；
准备失败保持 accepted=false、exit=1，exclusive claim 也不退款。

零模型调用另做只读 `audit_no_model_attempt`：核零 reservation/usage、严格 journal
文件集和冻结身份；尚未开始时记录 `model.status=not_started`，上下文已冻结但未派发
时记录 `refused_before_dispatch`。这两种状态的 completed 均为 false，不伪造模型
源码或 qualification，也不将“从未创建 candidate”误报成 candidate 文件漂移。

### 9.3 离线故障注入

新增 `tests/test_preparation_receipts.py`，全部为 offline/mock：覆盖七阶段成功/失败、
GPU 查询失败与资格拒绝的区分、返回码/超时/取消、敏感文本、坏 CSV、字节/文件/
计数上限、先落盘后操作、回执读回、失败不重试、未知 scratch 副作用、严格 no-op
证明，以及零模型审计拒绝伪造。成功 fixture 的 CUDA 标签仅测试接线，不是真卡验收。

本轮日志与历史文件哈希位于
`output/workload_codegen_validation/preparation-local-lckpefow/`。
最终专项 **471 passed**，其中新测试 **71** 项；项目全量 **3928 passed / 1 skipped /
2 warnings**（183.04 秒，仍为原 nonfinite fixture warning），随后串行 Golden
**4 PASS / 0 REVIEW / 0 FAIL / 0 SKIP**。没有改 Golden registry、容差或生产门槛。
首轮专项/全量日志也保留；最后补强了“start 回执写入失败即锁住阶段”和并发不重复
派发守卫，再重跑上述最终验证。
不修改旧的 `20e78e8edc20` 计划、execution claim、summary、失败证据或授权回执；
也不利用新诊断倒推当时未留存的返回码或 GPU 数值。当前驱动与旧计划的源码身份已
不同，后续真实尝试须重新冻结完整计划并取得新的明确批准。

本地修复结束时，仅准备了新的、**当时未批准/未执行**计划（后续执行另记第 10 节）：
`output/workload_codegen_h20/workload-codegen-2h8onaia/plan.json`，hash
`3d71c64785d9a31c1eda7222ac93ed46335cc8b9106e9d4fd4be6ab525062790`。
冻结 85 个 evaluator 源码文件；请求正文仍为 11789 bytes，模型身份、原 ruler、
LayerNorm 契约、workload、所有调用/派发/清理额度和验收门槛均与上次保持一致。
新增 preparation policy 和源码身份使它成为独立的新计划，不能沿用原批准。
本轮真实模型请求/远端派发均为 0，旧真机与已失败/已消费计划等 **304 个历史文件**
哈希不变；没有 commit/push。完整只读结果见本轮目录的 `final-summary.json`。

## 10. 新计划人工批准执行：准备通过，源码静态预筛失败

2026-10-11，用户明确批准新计划 `3d71c64785d9` 的单候选有界验收，接受明文 HTTP
以及已披露的同 UID、无敌意代码沙箱/网络硬隔离/显存 quota 限制。执行前重新核对
完整计划、85 份源码快照、请求正文、原 ruler、Golden registry 与 304 个历史文件，
均一致。新计划在外部动作前 exclusive 消费；不续用原 `20e78e8edc20` 计划。

真实执行目录：`output/workload_codegen_h20/workload-codegen-02rd6d4j/`。
授权回执、执行日志和事后只读核对另存于
`output/workload_codegen_validation/authorized-preparation-oi0rmi2f/`。
旧失败证据和第 9 节 plan-only 历史记录保留原样，不倒填批准或成功状态。

结果为 **`generation_failed`，exit code=1；completed、accepted、deployment_accepted
均为 false**，并非 GPU 正确性不通过、性能不达标或 degraded：候选根本未进入测量。

| 实际动作 | 数量 / 状态 |
|---|---|
| 准备回执 | 七阶段全部 passed；15 个文件，每个不超过 4096 bytes |
| 指定 GPU | H20、物理 GPU 2；查询时空闲 39334 MiB、利用率 0% |
| work command / RPC 派发计数 | 18 / 4；4 个 RPC 均为 gpu_info，含 2 次参考身份回执 |
| 模型请求 / 输出 token 预留 | 1 / 4096；响应 answered、finish_reason=stop |
| 请求 / 原始响应 | 11789 / 2061 bytes；请求 hash 与批准计划一致 |
| API usage | 输入 3981、输出 413、合计 4394 tokens；调用耗时约 44.68 秒 |
| 模型源码 | 原文保存在 candidate.json；status=failed，没有改写或替换 |
| Verifier / 编译 / 数值验证派发 | 0 / 0 / 0 |
| 主测 / 确认 / 心跳窗口 | 0 / 0 / 0 |
| KernelArtifact / 入库 | 无候选结果产物 / 无入库 |
| owned cleanup | quiescent=true、removed=true；shutdown 已确认，owned PID absent、signals=[] |
| 重试、修复、fallback、安装、共享进程操作 | 均未发生 |

GPU snapshot 只描述查询时的资格，不是独占设备或后续一直空闲的证明。Conductor
预先保留了 `contract.json`、`run_manifest.json` 两个绑定文件；它们不是通过验证的
KernelArtifact，不包含编译、数值或性能结果。费用摘要中的约 `$0.001607` 是仓库
CostTracker 的估算，不是服务端账单或硬计费上限。

### 10.1 失败原因：两个同名 kernel，而非超时或 JSON 格式问题

响应通过严格 JSON 信封解析，返回的模型源码在第 6、19 行分别定义了：

- `@triton.jit def kernel(X_ptr, Y_ptr, N, eps, BLOCK: tl.constexpr)`；
- Python 包装入口 `def kernel(X, eps=1e-05)`，内部使用 `kernel[(M,)](...)` 派发。

冻结静态预筛要求模块顶层恰有一个 `def kernel`，因此首先拒绝为
`one explicit kernel entry required`。从 Python 名称绑定语义还能静态推断：后一个
包装函数会覆盖前一个 JIT 名称，包装函数内的 launch 也会引用包装函数，而非 JIT
函数。**这不是观察到的远端运行异常**；没有执行模型源码来验证这个推断。

生产 catch 只记录通用 `candidate_generation_failed`，budget 随即 aborted，
candidate 原文和单次请求/响应仍被保留。事后审计对照相同的冻结源码，仅对原始
候选做 AST 与同一 precheck 的只读复现，得到上述首个失败原因；未改源码、未 import/
exec 候选、未重新调用模型或 GPU。其他源码语义、布局、正确性、性能均未经验证，
不能假设改个函数名就能通过真实验收。

正向 `audit_model` 拒绝 aborted campaign，原 summary 保持失败；其
`not a successful single-call campaign` 是失败后的审计结论，不是新的模型请求错误。

### 10.2 事后核对与下一步边界

本地 `post-run-audit.py` / 只读 `post-run-audit.json` 核对了：冻结计划与两份 85 文件
快照、消费 claim、原始请求/响应 hash、failed source 血缘、七阶段回执、派发计数、
原始 usage 重建、参考回执与 owned cleanup。执行证据文件在审计前后 hash 一致；
304 个历史文件、生产 protocol `9cd253e6d1e0fee3c7796ae5e488b7447a5138357a817c61c0ac71d85e9d4689`、
原 ruler 和 Golden registry 均不变。没有 commit/push。

`evidence_check_completed=true` **仅表示失败事实已核对，不是模型/raw/durable 正向
验收通过**。本次未修改生产源码，也未重跑或冒充第 9 节的本地全量/Golden 测试。
当前成功准备回执也不能倒推出第 8 节失败时的 GPU 数值或具体原因。

下一步先做独立的**离线修复阶段**：在模型提示中显式区分唯一 Python `kernel`
入口与不同名的 Triton JIT helper，明确 launch 应引用 helper、输入关键字必须匹配
冻结契约；同时考虑保留有界、固定词汇的静态失败原因码，并加入回归测试。
不放宽唯一入口、原文血缘、正确性、性能或确认门槛，不把当前失败候选改名后重测。
后续真实尝试必须重新冻结完整计划并获得新的人工批准；本次 claim 不退款、不恢复。

## 11. 离线修复：入口/helper 命名与固定失败诊断

本阶段只修改提示、静态预筛、失败诊断接线与离线回归测试；真实模型请求、SSH/GPU
派发均为 0。不修改第 10 节返回的源码，不将它改名后重测，也不消费旧计划或批准。

### 11.1 先明确调用约定，不替模型修代码

系统提示现在显式要求：

- 恰好一个未加装饰器的顶层 Python `def kernel`，输入关键字与冻结声明大小写一致；
- `@triton.jit` 计算 helper 使用不同名字，不重定义或重新绑定 `kernel`；
- 用 `helper[grid](...)` 派发计算，不使用 `kernel[grid](...)`，不把包装入口加上 JIT。

提示仅描述接口/命名/派发规则，不提供 LayerNorm 实现、可信候选模板或上一轮反馈。
源码仍按解码原文保存；没有机械改名、修复、第二次请求、fallback 或替换候选。

保留原有唯一入口、JIT、导入/顶层执行、allocation-only、anticheat、接口和语法门。
另外显式拒绝给入口加装饰器、模块顶层 assignment 重新绑定入口、直接的
`kernel[grid](...)` 调用。这是更窄的静态模式，**不是放宽拒绝条件**。不声称能完整
解析所有动态别名、作用域或敌意代码行为；预筛仍不是沙箱或数值/性能证明。

### 11.2 只保存固定词汇，不保存任意异常文本

新增 `PrecheckRefused(ValueError)`：18 个不可变 reason code 对应固定消息，包括
`kernel_entry_count`、`kernel_entry_decorated`、`kernel_entry_rebound`、
`kernel_entry_launch`、`source_syntax_invalid`、`kernel_interface_refused` 等。
语法错误、anticheat 详情、接口 problem、任意异常文本或异常类名不进入诊断回执。

失败的 `candidate.json` 新增 `failure={stage, reason}`：严格两字段、固定组合、最多
256 UTF-8 bytes。四个 stage 为 `model_call / envelope / precheck / context`；无法
识别的静态异常使用 `static_precheck_failed`，其他阶段使用固定的粗粒度原因码。
诊断不包含源码或 rationale；已有 raw response / 原始候选源码证据仍按原规则保留，
这不是声明整个 journal 都不含源码。取消或诊断缺失也不能补造成功凭证。

`generation_failure` 只读验证词汇、字段和字节界限，拒绝任意文本、额外字段、坏
类型或跨 stage 的 reason。驱动 `audit_model` 可以将固定回执带入失败摘要，但
aborted campaign 仍然 completed=false。正向 `audit_generation` 拒绝任何带 failure
字段的 lineage；诊断不能代替模型/source 血缘、数值验证、性能窗口或恢复 eligibility。

### 11.3 离线回归与历史隔离

新增 `tests/test_single_candidate_diagnostics.py`，覆盖每个静态原因、parse/compile
语法拒绝、提示规则、有效 helper launch 的静态路径、异常文本/类名/坏 reason 脱敏、
只读/只调用一次/失败不退款、正向凭证拒绝 failure 字段、Conductor 不派发 Verifier/
Fixer/Search，以及驱动诊断读回和篡改拒绝。

`tests/fixtures/single_candidate_duplicate_kernel.txt` 仅作为文本读取，逐字保存第 10 节
原始源码，hash 仍为 `6857d1d975367ff010bda617e72acb7998f0ebd1746fc0b5f68c4ae30643bb73`。
它始终触发 `kernel_entry_count`；不 import/exec、不改名，不是真卡正确性 fixture。
正向 helper launch 的 mock 夹具同样不证明它是正确的 matmul 或真实 CUDA 实现。

本轮记录在 `output/workload_codegen_validation/naming-local-qcgkgum6/`。新测试 **74** 项，
最终专项 **427 passed**；项目全量 **4002 passed / 1 skipped / 2 warnings**（218.16 秒，
仍为原 nonfinite fixture warning），随后串行 Golden **4 PASS / 0 REVIEW / 0 FAIL /
0 SKIP**。Golden 仅为 CPU/mock 回归，不是 H20 性能证明；没有修改 registry 或容差。
首轮命名/诊断专项为 279 passed；补强入口重绑定、异常 reason 类型和词汇不可变性后
重跑最终专项及全量。一次扩展专项命令误填不存在的 `test_run_binding.py`，未运行
测试；原日志保留，改正为实际的 resume integrity 测试后重新运行，没有调整期望。

事前/事后核对的 **671 个历史文件**均不变，包含旧真机、已消费 claim、两次失败
执行、此前准备目录及本地验证记录。原 ruler 和 Golden registry 也不变；源码在
最终验证与新计划冻结期间保持一致，没有 commit/push。

### 11.4 新身份必须配新计划，不能恢复旧批准

本轮修改了生产 `single_candidate_coder.py`，因此整体 production protocol 身份
由旧 `9cd253e6d1e0fee3c7796ae5e488b7447a5138357a817c61c0ac71d85e9d4689` 更新为
`8bc9ff3330790e63b656dda8bf9ed9a74007f24ddf5e5f710a2442b474544e7d`。
protocol 文件集未变，其中仅该模块 hash 改变；验收 driver 的诊断接线另进 evaluator
源码快照。**这不是把旧失败改签成新协议结果**，旧文件和原始 hash 全部保留。

新计划仅在本地准备，**未批准/未执行**：
`output/workload_codegen_h20/workload-codegen-s4c21jrn/plan.json`，hash
`4d460efbe3dd93129b310c89a32d200c0cab98a77e6acc6e2bffb8232dfaeaa2`。
仍冻结 85 份源码；命名提示使完整请求从 11789 增至 **12077 bytes**，request hash
为 `6b7ad0441c9d88246de3f14f0383f6b223da41110d1cd33e011c9b93fdd30a65`。
`prepared-plan-context.json` 登记 credential_present=false、无 execution claim、
真实模型请求/远端派发均为 0；准备计划不是执行批准。

已逐项比较：generation context、LayerNorm 契约/精度/参考、suite、workload/权重/
阈值、模型身份、调用/派发/清理预算、同 UID 隔离限制和原 ruler 都保持不变。只有
提示及其 hash、请求及其 hash/字节数、源码/protocol 身份和计划 hash 变化。模型
输入仍不含上轮反馈、历史候选或模板源码；新提示不保证下一次模型一定生成合格源码。

下一步是在取得**针对该新计划的明确人工批准**后，执行一轮原额度的单候选真实
验收；失败仍停，不偷偷启用修复、多路采样或补测找绿。当前本地阶段的 completed
只表示离线工作完成，不表示真实代码生成、CUDA 正确性或部署验收已通过。
