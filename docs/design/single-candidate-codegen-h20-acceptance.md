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
第 12 节记录新计划的真实执行：静态预筛和编译通过，第二个主输入数值验证失败；
没有进入候选性能主测，owned 清理已确认，仍无入库或部署验收结论。
第 13 节记录针对 masked reduction 和 stride 语义的离线提示与回归补强。
第 14 节记录该新计划获批后的真实执行：模型扩展了包装入口的必填参数，静态接口
预筛拒绝，未进入候选 CUDA 编译或测量，owned 清理已确认。
第 15 节记录调用签名、包装函数职责和返回约定的离线结构化补强。
第 16 节记录该新计划的真实执行：包装接口与编译通过，第二个主输入再次因 padding
方差错误失败；owned 清理已确认，未进入性能主测。
第 17 节记录随后新增的显式一次数值修复模式；该阶段仅本地接线、回归和计划冻结，
不将新增历史反馈载荷视为已获旧计划授权。
第 18 节记录该修复计划明确获批后的真实执行：模型原样返回失败源码，未进入
候选验证，两个 claim 已消费，清理确认，无新增正确性或性能结论。
第 19 节记录之后的离线诊断层与伪修复守卫，不能视作再次验收或恢复修复额度。

## 2. 默认 plan-only 与执行资格

默认执行仅创建新的 0700 本地目录、只读 `plan.json` 和完整源码快照：

```bash
python -B benchmarks/workload_codegen_h20.py
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

本地修复结束时，新计划仅在本地准备，**当时未批准/未执行**（后续执行另记第 12 节）：
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

## 12. 新计划真实执行：编译通过，主输入数值验证失败

2026-10-15，用户对“先本地提交，再执行新计划 `4d460efbe3dd` 的单候选验收”的
建议回复“开动吧”。本次据此执行已披露的模型端点、物理 GPU 2 和原额度，不续用
第 10 节已消费的计划。实现先本地提交为 `9cb6b1a`（`feat: add bounded model code
generation and H20 acceptance`）；没有 push。

执行前重新核对完整计划、85 份源码快照、12077-byte 请求、原 ruler、Golden registry
和 825 个历史文件；计划 hash 仍为
`4d460efbe3dd93129b310c89a32d200c0cab98a77e6acc6e2bffb8232dfaeaa2`。
执行 claim 在外部动作前 exclusive 消费，不能退款或恢复。

- 真实执行目录：`output/workload_codegen_h20/workload-codegen-uwh99mow/`。
- KernelArtifact：该目录下的 `artifacts/layernorm/nvidia_gpu/20261015_140819_44a6b620/`。
- 授权、提交回执、执行日志、局部数学诊断与只读审计：
  `output/workload_codegen_validation/authorized-naming-u3z7b84d/`。

结果为 **`correctness_failed`，exit code=1；completed、accepted、deployment_accepted
均为 false**。这次模型请求、静态预筛和 CUDA 编译均成功；失败来自实际数值验证，
不是生成失败、性能不达标或 degraded。

| 实际动作 | 数量 / 状态 |
|---|---|
| 准备回执 | 七阶段全部 passed；15 个有界只读文件 |
| 指定 GPU | H20、物理 GPU 2；查询时空闲 14084 MiB、利用率 0% |
| work command / RPC | 18 / 12；gpu_info 8、native heartbeat 1、compile 1、verify 2 |
| 模型请求 / 输出 token 预留 | 1 / 4096，无重试或 fallback |
| 请求 / 原始响应 | 12077 / 1907 bytes；finish_reason=stop |
| API usage | 输入 4045、输出 389，合计 4434 tokens；调用约 34.18 秒 |
| 模型源码 | 761 bytes，原文保留；唯一 Python 入口与不同名 JIT helper，prechecked |
| L1 编译 | 1 次 RPC，通过；不等同于底层 JIT specialization 次数上限 |
| 主输入数值验证 | 2 次 RPC，第一个通过、第二个失败，均全元素比较 |
| 32-case correctness suite | 全部 not_run；主输入失败后未进入 suite |
| 候选主测 / 确认窗口 | 0 / 0；无候选性能成绩 |
| 原生校准心跳 | 1；独立冻结 ruler 锚点检查通过，不是候选性能测量 |
| owned cleanup | quiescent=true、removed=true，shutdown 已确认；owned PID absent、signals=[] |
| 入库、重试、修复、安装、共享进程操作 | 均未发生 |

GPU snapshot 不是独占租约。CostTracker 约 `$0.001602` 仅为估算，不是服务端账单。
模型源码 hash 为 `bdb1698cfa2e9a6a341ecf35e2987d8d4349384ed43b23d71661b4500853b2cf`；
没有修改源码、换成可信模板或补测找绿。

### 12.1 根因：padding 元素错误地进入中心化方差

源码用 `BLOCK=next_power_of_2(N)`；masked load 将越界 lane 填成 0，但随后的
`x_centered = x - mean` 把这些 lane 变成 `-mean`。方差归约没有再次排除它们，故
每个 padding lane 额外贡献 `mean²`：

```text
var_generated = var_true + (BLOCK - N) * mean² / N
```

冻结的可信主输入生成器按 `[-1, 0, 1, 2]` 重复填充；均值为 0.5，总体方差为 1.25。
两个 workload 的 BLOCK 都是 16384，但 padding 比例不同：

| 主输入 shape | padding 数 | 错误方差 | 实测最大绝对误差 | 匹配比例 / 结果 |
|---|---:|---:|---:|---|
| `(8192,15360)` | 1024 | 约 1.266667 | 0.0087890625 | 1.0 / 通过 |
| `(8192,12288)` | 4096 | 约 1.333333 | 0.04296875 | 0.0 / 失败 |

第二个输入的 **100663296 / 100663296** 个元素超出冻结容差；首个错误值为
`-1.298828125`，参考为 `-1.341796875`，无参考 nonfinite。第一个输入通过仅因
这次误差仍在容差内，不代表算法正确。容差保持 atol=1e-5、rtol=0.01、要求匹配比例 1。

事后只用四个输入值做局部数学演算，预测出的最大绝对误差和匹配比例与两次实测
完全一致；没有在本地 import/exec 模型源码，也没有新增 API/GPU 调用。
此外源码使用 `row*N` 连续地址假设，未处理输入 strides；这是待处理的布局语义
问题，**不是已观测到的 strided case 失败**，因为 suite 尚未运行。

### 12.2 失败证据审计与下一步

本地重新执行 model/raw/durable 只读审计，核对请求/响应、源码血缘、参考回执、
两次真实数值结果、checkpoint、准备与清理证据。三项审计均 completed=true、
errors=[]，raw/durable 均保持 `correctness_failed`；**审计成功只说明证据一致，
不等于候选验收通过**。审计自身的 gpu_measurements=0 不能解释为本轮没有真机动作。

初次事后审计脚本误把远端 probe 的 `absolute_anchor_checked` 要求为 true。
实际协议明确要求原始 probe 为 false，由控制器对独立冻结 ruler 执行锚点检查。
修正本地审计后，`anchor_reading` 只读重算通过（good ratio 约 0.997、slow ratio
约 7.1972）；原脚本和失败日志保留，不修改原始回执，也不新增测量。

最终审计前后：本轮执行证据、准备快照和 825 个历史文件均不变；生产 protocol
仍为 `8bc9ff3330790e63b656dda8bf9ed9a74007f24ddf5e5f710a2442b474544e7d`，原 ruler 和
Golden registry 不变。提交后的收尾仅更新验收文档；不冒充重新跑过第 11 节的
全量/Golden 测试，也不将之前的 CPU/mock 成绩当作当前模型源码的 CUDA 证明。

下一步先做独立的**离线语义修复阶段**：明确 masked reduction 中无效 lane 的贡献、
非二次幂长度与输入布局/stride 语义，补充相应回归；不放宽容差、不缩减 suite，
不针对当前四值数据特化，不修改这份失败候选后重测。真实再试需冻结新计划并获得
针对它的授权，本次 claim 已消费；本轮没有启动第二个候选。

## 13. 离线补强：masked reduction 与输入/输出 stride 语义

2026-10-15，本阶段更新生成要求和离线回归；真实模型请求、SSH/GPU 派发均为 0。
第 12 节的失败候选、原始数值结果、执行 claim 和清理回执保留原样。

### 13.1 把通用数值与布局要求写入冻结请求

生产 `SingleCandidateCoder.PROMPT` 增加以下要求：

- 覆盖声明中的形状、值模式和布局，包括非二次幂归约长度；
- 无效 lane 在所有变换后仍应对归约贡献单位元，不能只在 load 时填零；
- 归一化使用逻辑归约长度，不能使用补齐后的 block 大小；
- 以运行时张量的实际 stride 计算地址，输入和输出分别取值；明确序列化 NumPy
  metadata 的 stride 单位是 bytes，而运行时 Torch tensor stride 的单位是 elements；
- 主 workload 连续不意味着可以忽略 suite 声明的 strided 输入。

驱动的 caller-owned LayerNorm sketch 补充末轴总体方差、逻辑列数、冻结 epsilon、
输出 dtype、均值/中心化平方归约中的 padding 处理，以及非零常量/近常量行与二维
stride 要求。这些都是通用语义文字，不包含实现模板、上次源码、具体输入值、
历史误差或性能成绩；没有把某个 BLOCK、launch 参数或四值数据答案写进提示。

本阶段没有新增“靠 AST 证明数值正确”的守卫。源码仍原文保存，静态预筛只负责
原有结构/接口等限制；参考、全量 suite、容差、真实验证与入库门继续负责验收。
提示与 sketch 一并进入冻结声明、请求 hash 和执行前 drift 检查。

### 13.2 回归同时保留错误反例和通过对照

新增 `tests/test_single_candidate_semantics.py`，共 32 项离线测试，覆盖：

- 通过 mock 模型派发核对完整请求字节、mask/stride 要求、32-case 声明、输入
  metadata 单位以及仅传 metadata、不传模板/输入值的边界；
- 提示、mask hint、stride hint 漂移时，在预留请求额度前拒绝；
- 单独的 stride copy 文本夹具可通过静态门，仅说明允许读取实际 tensor stride；
- 可信 CPU 反例在逻辑列数 3/31/33、near_constant/large_varying、连续/strided
  布局中复现 padding 方差错误；连续地址假设也会被 strided case 拦住；
- 主输入通过后 suite 失败、主输入失败后 suite not_run，均不能进入 L3；正确的
  CPU 对照通过相同值模式/布局及确定性重复检查。

CPU 反例使用明确的 NumPy fp32 参考，只验证现有数值/suite 路由。首轮测试遇到
本机可选 PyTorch 的 near-constant CPU 参考差异，以及成功主输入不写 failure-only
状态字段的问题；测试随后改为显式 NumPy 参考和核对实际主输入读数，保留初轮日志。
没有为此修改生产参考、容差或 suite，也不把 CPU 对照当作 native fp16 CUDA 证明。

`tests/fixtures/single_candidate_unmasked_variance.txt` 逐字保存第 12 节的 761-byte
失败源码，hash 仍为 `bdb1698cfa2e9a6a341ecf35e2987d8d4349384ed43b23d71661b4500853b2cf`。
该文件始终只作为文本读取，不 import/exec；测试还明确证明它仍可通过静态预筛和
mock 模型血缘审计，但没有数值正确性凭证。不会因新增提示而把旧失败改判为成功。

本轮本地验证记录位于 `output/workload_codegen_validation/semantics-local-q_eslm5k/`。
全量初轮还发现既有 CLI 示例含本机绝对路径，触发仓库卫生守卫；已改为仓库相对
命令，不放宽扫描规则。原失败日志保留为 `full-tests.log`。
最终专项 **250 passed**，卫生与最终语义夹具复验 **48 passed**；修正后全量
**4034 passed / 1 skipped / 2 warnings**（158.18 秒，仍为原 nonfinite fixture
warnings），随后串行 Golden **4 PASS / 0 REVIEW / 0 FAIL / 0 SKIP**。两轮全量日志
分别保留，Golden 没有刷新 registry 或容差；这些结果均为本地测试。
事前/事后核对的 **1033 个历史文件**、原 ruler 和 Golden registry 均不变；最终
验证前后的四份源码/测试/文本夹具 hash 一致。本轮没有 commit/push。

### 13.3 新计划冻结了新提示，额度与验收判据保持原值

生产 protocol 仅因 coder 源码改变，从
`8bc9ff3330790e63b656dda8bf9ed9a74007f24ddf5e5f710a2442b474544e7d` 更新为
`910115a969881d1f5c0d7719c8599d5dde9ea7829f5a4bcdff90f7ebfc222f40`。
driver 的 sketch 改动另进入 evaluator 源码身份；仍冻结 85 份源码。

离线阶段结束时，新计划**仅本地准备，当时未批准/未执行**（后续执行另记第 14 节）：
`output/workload_codegen_h20/workload-codegen-3bp19hjl/plan.json`，hash
`169b9d0bfb9e61295ca8e04f282b00c1d47f7505c01c17e7ae89ed34d08ebbe6`。
完整请求为 **13181 bytes**，request hash
`658487de996da4655fae3b3cc0a6d9a1a185b81f0978256fd3614e1a7f2699ae`。

`prepared-plan-context.json` 已对照上个计划：仅提示/sketch、相关 context/request
hash、源码/protocol 身份和计划 hash 变化。LayerNorm 契约、32-case suite、参考、
两个主 workload、权重/门槛、原 ruler、模型身份、请求/派发/清理额度和远端隔离
边界均保持一致。credential_present=false，无 execution claim、无 model journal。
该离线阶段没有执行真实尝试；后续可审阅此具体计划，再进行新的一轮有界验收。

## 14. 新计划获批执行：包装入口接口不匹配，预筛拒绝

2026-10-15，最初自动审批在进程启动前拒绝了外部执行，此时请求/远端派发均为 0、
新计划未消费。用户随后对计划 `169b9d0bfb9e` 的具体目的地、13181-byte 载荷和执行
边界明确回复“明确批准新计划及上述数据外发，接受隔离限制”。再次核对 85 份源码、
冻结请求、此前验证过的六个工作区文件及 1187 个历史文件后，才执行唯一一次真实请求。

真实执行目录：`output/workload_codegen_h20/workload-codegen-n_jl29gn/`。
预检、两次授权语境回执、首次未启动记录、执行日志与事后审计均另存于
`output/workload_codegen_validation/authorized-semantics-jfykcsip/`。
原 `execution-not-started.json` 记录仍按当时事实保留，没有覆盖成成功执行凭证。
本阶段没有新增生产源码改动，没有 commit/push。

结果为 **`generation_failed`，exit code=1；completed、accepted、deployment_accepted
均为 false**。准备和模型响应成功，固定诊断为
`failure={stage: precheck, reason: kernel_interface_refused}`。

| 实际动作 | 数量 / 状态 |
|---|---|
| 准备回执 | 七阶段全部 passed，15 个有界只读文件 |
| 指定 GPU | H20、物理 GPU 2；查询时空闲 25312 MiB、利用率 0% |
| work command / RPC | 18 / 4；RPC 均为 gpu_info，含两次参考身份核对 |
| 模型请求 / 输出 token 预留 | 1 / 4096；answered，finish_reason=stop |
| 请求 / 原始响应 | 13181 / 2225 bytes；与批准计划一致 |
| API usage | 输入 4235、输出 446，合计 4681 tokens；调用约 24.16 秒 |
| 模型源码 | 924 bytes，原文保留；status=failed，budget aborted |
| Verifier / 编译 / 数值验证派发 | 0 / 0 / 0 |
| 32-case suite 执行 | 0；未创建 L2 结果，不能声称已有 suite 的数值 verdict |
| 主测 / 确认 / 心跳窗口 | 0 / 0 / 0 |
| 候选结果产物 / 入库 | 无 / 无；只保留预先绑定的 contract、run_manifest |
| owned cleanup | quiescent=true、removed=true，shutdown 已确认；owned PID absent、signals=[] |
| 重试、修复、fallback、安装、共享进程操作 | 均未发生 |

GPU snapshot 不是独占租约；CostTracker 约 `$0.001716` 为估算，不是服务端账单。
源码 hash 为 `25c204a4385ca5810759e9d90ee41469000be0ce179812249016ed283d3ac119`。

### 14.1 根因：把包装函数需要计算的量误当成调用方参数

冻结输入 metadata 的关键字集合只有 `X`，实际执行约定为 `kernel(X=...)`。模型
却将包装入口写成：

```python
def kernel(X, Y, N, eps, stride_xm, stride_xn, stride_ym, stride_yn):
```

额外七个参数全部必填，调用方不会提供；因此静态接口守卫按原规则拒绝。包装入口
应按冻结接口在函数体内分配输出、读取形状和实际 stride、使用冻结 epsilon，并
返回输出；不能要求调用方新增输入。本次源码的包装入口也没有 return 语句。

AST 读回还显示，模型虽然使用了 stride 参数，方差归约仍包含中心化后的 padding
lane。这里只能记为源码中的静态观察：候选没有执行，不能沿用第 12 节的误差数值，
也不能宣称本次 stride 或 masked reduction 已通过真机验证。模型 rationale 中的
“matching the reference semantics”同样不构成验证证据。

### 14.2 只读复核与下一步

`post-run-audit.py` 对原始 JSON 响应、请求/响应 hash、源码血缘、usage/费用重建、
准备回执、两份源码快照、固定失败原因、参考身份与 owned cleanup 做只读核对；
在原文上复现同一静态守卫，不 import/exec、不改写候选，不新增 API/GPU 调用。

`evidence_check_completed=true` 只说明失败事实一致。生产正向 `audit_model` 保持
completed=false，并保留 `kernel_interface_refused`；aborted campaign 不具备正向
凭证。没有可用于 raw/durable 候选审计的验证结果或 KernelArtifact，两个预绑定
文件不能补成已验证产物。

审计前后，本轮执行证据、准备快照、1187 个历史文件、原 ruler 与 Golden registry
均不变，production protocol 仍为
`910115a969881d1f5c0d7719c8599d5dde9ea7829f5a4bcdff90f7ebfc222f40`。
本轮只做真实单次验收和文档收尾，不把第 13 节的本地测试改记为本轮 CUDA 结果。

下一步先做独立离线阶段：将**实际调用签名、包装入口职责和返回约定**明确写入
冻结 generation context；输入参数只能来自 caller metadata，形状、stride、输出
分配和参考常量应由包装入口处理，并加入本次真实源码的文本回归。仍保留 mask/
stride 数值门，不能因为接口先失败就忽略已知语义问题。本次 claim 已消费，
不改写这份源码后重测，也不在本轮追加候选。

## 15. 离线补强：结构化包装入口调用与返回约定

本阶段在生成上下文新增 caller-derived `calling_convention`，由生产 coder 和默认
plan-only 驱动共用同一个纯函数构造。真实模型请求、SSH/GPU 派发均为 0；第 14 节
原始候选、失败结果与已消费 claim 保留原样。

### 15.1 冻结实际参数来源，而不是让模型猜入口

`wrapper_calling_convention(metadata)` 仅读取各组实际 caller keyword 名，生成
版本为 `keyword-wrapper-v1` 的结构化约定，包含：

- `entrypoint=kernel`、`invocation=kernel(**inputs)`；
- 实际 `keyword_sets`、允许的 `parameter_names` 和规范化 `signature`；
- `extra_parameters_allowed=false`，参数须显式且可用 keyword 调用；
- 返回值为由 wrapper 分配的 tensor，形状/dtype 遵守冻结算子语义；
- wrapper 内部分配输出、读取运行时形状/归约长度、分别读取输入和输出 stride、
  从冻结契约/注册参考取得语义常量，并构造 helper 所需的额外 launch 参数；
- metadata stride 单位为 bytes，运行时 tensor stride 单位为 elements。

本次 LayerNorm 的参数集合精确为 `['X']`，规范化签名为 `def kernel(X):`；输出
buffer、N、epsilon 与 stride 不会被编造成调用方输入。若其他受支持调用方显式
声明 scalar 参数，则保留其名字，但仍不发送 scalar 值。多组不同 keyword 集合
使用共同必填参数及其余参数的 `None` 默认值描述，不把每组输入扩展成并集实参。
无法表示为显式 Python 参数的输入名在生成前拒绝，不把它插入签名文本。

该结构只包含调用约定和职责文字，函数头属于接口声明，不包含实现函数体、可信
候选模板、历史源码或历史误差。原有 mask/stride 数值要求也继续保留在提示/sketch。
`context_hash`、完整请求、源码快照和计划身份全部覆盖新字段；执行前冻结检查和
正向 `audit_generation` 都会从 input metadata 重建它，拒绝“改字段再重算 hash”
伪造出来的调用约定。plan-only 只构造 metadata，不分配主 workload 数组。

### 15.2 静态门拒绝额外参数与明显缺少返回值

保留原 `check_kernel_interface` 的逐组参数兼容检查；仅在本单候选模式中进一步
要求显式参数集合等于 caller keyword 集合的并集，拒绝额外参数（包括带默认值的
额外参数）、`*args`、`**kwargs` 和 positional-only。与 keyword 调用兼容的
keyword-only 参数仍允许；没有改动其他执行模式共用的接口检查器。

新增固定原因 `kernel_return_missing`：wrapper 中至少需要一个带非 None 表达式
的 return；helper、嵌套函数/类中的 return 不算 wrapper 返回。它只证明语法存在，
不证明所有控制路径都返回、返回对象的 tensor 类型、输出分配或数值正确；这些
仍须运行时与原 Verifier 验证。失败诊断词汇由 18 扩为 19 个静态原因，仍保持原
两字段、固定词汇与 256-byte 界限。

接口检查先于新增返回检查，因此第 14 节源码仍以 `kernel_interface_refused`
拒绝，没有改写它的既有失败原因。第 12 节 padding 方差错误源码仍可通过静态门，
不能据此改判数值正确。

### 15.3 回归、历史隔离与新计划

新增 `tests/test_single_candidate_wrapper_contract.py`，45 项离线测试覆盖实际请求
字节与 context、metadata 推导、多输入集合/声明 scalar、非法参数名、无共享可变
状态、冻结漂移和重算 hash 后的伪造拒绝，以及额外/隐藏参数、缺少返回值、嵌套
return、keyword-only 正向静态路径、固定失败回执和单次请求不修复。

`tests/fixtures/single_candidate_extra_wrapper_args.txt` 逐字保存第 14 节的 924-byte
原文，hash 为 `25c204a4385ca5810759e9d90ee41469000be0ce179812249016ed283d3ac119`。
它只作为文本读取，不 import/exec；本阶段不把其参数删掉后重测。记录位于
`output/workload_codegen_validation/wrapper-local-fhog33k6/`。
最终专项 **295 passed**；全量 **4079 passed / 1 skipped / 2 warnings**（164.48 秒，
仍为原 nonfinite fixture warnings），随后串行 Golden **4 PASS / 0 REVIEW / 0 FAIL /
0 SKIP**。验证前后六份源码/测试/文本夹具 hash 一致，1314 个历史文件、原 ruler
和 Golden registry 不变；所有改动文件（含未跟踪文件）通过同一仓库卫生模式扫描。
这些结果只证明本地接线、静态守卫与回归，不代表新模型源码已通过 CUDA 验收。

离线阶段结束时，新计划**仅本地准备，当时未批准/未执行**（后续执行另记第 16 节）：
`output/workload_codegen_h20/workload-codegen-bvqb4x61/plan.json`，hash
`c601e21ad167f9ff5d0b7750d4ff734af8168c3c33273c44633a9f38c5fdf5f3`。
完整请求 **14570 bytes**，request hash
`90bf59bdf399a82c246b105d2e3af68dd6bbe4efbd7089ac85a0fca96dac1518`，仍冻结 85 份源码。
production protocol 更新为
`8088734fe5ac8e3360b28a3407d4fc915016ff014ab255f6247731c7a2c7c92b`。

已对照上一计划：除提示、新 calling_convention、disclosure 内容说明以及相应
context/request/源码/protocol/计划 hash 外，契约、注册参考、32-case suite、
sketch、workload/权重/门槛、原 ruler、模型身份、所有额度和远端隔离边界保持一致。
新计划没有 execution claim 或 model journal，credential_present=false。本阶段
没有 commit/push；后续真实验收须针对这个具体新计划获得授权。

## 16. 新计划获批执行：包装接口通过，padding 方差错误再次被数值门拦住

2026-10-15，用户对计划 `c601e21ad167` 的 14570-byte 具体载荷、模型目的地、GPU 2
单候选额度和隔离边界明确回复“明确批准新计划及上述数据外发，接受隔离限制”。
执行前重查 85 份源码快照、完整请求、八个已验证工作区文件和 1465 个历史文件；
核对一致后才 exclusive 消费本次 claim，不续用第 14 节的已消费计划。

- 真实执行目录：`output/workload_codegen_h20/workload-codegen-mn841e1w/`。
- KernelArtifact：该目录下 `artifacts/layernorm/nvidia_gpu/20261015_212243_490247e7/`。
- 预检、明确批准、执行日志、局部数学诊断和只读审计：
  `output/workload_codegen_validation/wrapper-acceptance-f_3n_ey8/`。

结果为 **`correctness_failed`，exit code=1；completed、accepted、deployment_accepted
均为 false，summary.errors=[]**。模型正常返回原文源码，包装入口、静态预筛与
H20 编译均通过；真实数值验证失败后停止，未修复、重试或替换候选。

| 实际动作 | 数量 / 状态 |
|---|---|
| 准备回执 | 七阶段全部 passed，15 个有界只读文件 |
| GPU snapshot | H20、物理 GPU 2；查询时空闲 25312 MiB、利用率 0%，不代表独占 |
| work command / RPC | 18 / 12；gpu_info 8、native heartbeat 1、compile 1、verify 2 |
| 模型请求 / 输出 token 预留 | 1 / 4096，answered，finish_reason=stop |
| 请求 / 原始响应 | 14570 / 2321 bytes；与批准计划一致 |
| API usage | 输入 4510、输出 485，合计 4995 tokens；调用约 19.75 秒 |
| 源码 | 955 bytes，status=prechecked；保持模型原文 |
| 包装入口 | `kernel(X)`，内部 `empty_like` 分配 Y，四个实际 input/output stride 查询，返回 Y |
| 编译 / 数值验证 RPC | 1 / 2；编译通过，两个主输入一过一败 |
| 32-case suite | 全部 not_run，未产生 suite 的数值测量 |
| 候选主测 / 确认窗口 | 0 / 0，无候选性能成绩 |
| 原生校准心跳 | 1，独立冻结 ruler 锚点通过，不是候选性能测量 |
| owned cleanup | quiescent=true、removed=true，shutdown 已确认；owned PID absent、signals=[] |
| 入库、修复、重试、fallback、安装、共享进程操作 | 均未发生 |

源码 hash 为 `b3f95ac94f85e6027b54df920ef48425f200185c466eaeec89b21b1a539b8c95`。
CostTracker 约 `$0.001838` 是估算，不是服务端账单。一个编译 RPC 也不能作为底层
JIT 专门化次数上限。

### 16.1 接口要求得到遵守，数值提示没有转化为正确实现

本次包装入口没有再要求调用方提供 Y、N、epsilon 或 stride，且明确返回输出。
helper 使用独立 input/output stride 参数，但 strided suite 未运行，因此这些源码
特征不能证明 strided 正确性。

方差部分仍为 `x_centered = x - mean` 后对整个 block 归约平方和。masked load 的
padding 先被置零，减去均值后变成 `-mean`，从而再次错误地贡献 `mean²`。本次实测：

| 主输入 | 全量比较元素 | 最大绝对误差 | 匹配比例 | 结果 |
|---|---:|---:|---:|---|
| `(8192,15360)` | 125829120 | 0.0087890625 | 1.0 | 通过 |
| `(8192,12288)` | 100663296 | 0.04296875 | 0.0 | 失败，全部元素超差 |

第二个输入首个错误值为 `-1.298828125`，参考为 `-1.341796875`；两次 coverage 均为
1、参考 nonfinite 均为 0。atol=1e-5、rtol=0.01、要求匹配比例 1，均未改变。

只读数学诊断基于冻结主输入的四值重复模式重新计算：
`var_generated = var_true + (BLOCK-N)*mean²/N`。均值为 0.5，真实方差为 1.25；
两个 workload 的错误方差分别约为 1.266667 和 1.333333。局部四值演算预测出的
最大绝对误差和匹配比例与本次真机结果一致，未在本地 import/exec 候选或新增 GPU
测量。数值与第 12 节相同，但属于本次独立的原始验证回执，不是复制旧成绩。

### 16.2 失败事实已核对，后续应转向有界反馈修复

事后重新执行 model/raw/durable 只读审计，三者均 completed=true、errors=[]；
raw/durable 保持 `correctness_failed`，checkpoint/source/contract/input 绑定一致。
审计完成只说明证据一致，不等于候选验收通过；审计自身 gpu_measurements=0 也
不代表本次未做真机验证。校准原始 probe 的 absolute_anchor_checked=false，由
控制器独立核对冻结 ruler；只读重算 good ratio 约 1.01846、slow ratio 约 6.87009，
锚点检查通过。

审计前后，本轮执行证据、准备快照、1465 个历史文件、原 ruler 与 Golden registry
均不变；production protocol 仍为
`8088734fe5ac8e3360b28a3407d4fc915016ff014ab255f6247731c7a2c7c92b`。
本轮未修改生产源码，未 commit/push，也未重跑或改记第 15 节的全量/Golden 成绩。

这轮说明接口约定对本次输出有效，但反复补充通用提示仍未纠正已知数值错误。
下一步建议设计独立的**带结构化验证反馈的一次有界数值修复**：冻结失败源码 hash、
原始验证回执和允许反馈的字段，明确最多一次修复请求和一份新候选，保留原文与
修复血缘，并沿用相同契约、参考、workload、容差与验证门。届时载荷会包含失败
反馈，必须在新的具体计划中如实披露并另获批准。

这是后续方向，本次 repair=0，claim 已消费；没有在本轮自动修复源码或追加候选，
也没有为了找绿调整形状、容差或验收门槛。

## 17. 离线新增：失败源码与结构化反馈的一次有界数值修复

新增 `--repair-parent-run`，普通单候选生成默认行为和本文件前述历史事实不变。
只接受原始单候选 L1 通过、primary 全量数值失败、suite 全部 not_run、清理确认的
父运行；核对模型响应、原文、checkpoint、manifest、原始 RPC 与历史 evaluator。
模型只收到旧模型源码和白名单数值指标，不收到原始日志、输出样本、输入数组、
模板或仓库文件。新源码必须重新经过原来的完整正确性、测量及确认/入库门。

一次性 plan claim 外，再由固定父身份 claim 防止换计划后追加修复；失败或准备
中断不退额度，不允许修复链、第二次请求或机械补代码。普通 caller 不可冒充修复
caller，血缘明确绑定父源码/身份、反馈和新源码。完整说明及新计划 hash 见
[一次有界数值修复](bounded-numerical-repair.md)。

本地阶段真实模型/SSH/GPU 派发均为 0；新计划尚未消费或批准，没有新增候选的
真机正确性或性能结论，也没有 commit/push。
专项 330 passed，全量 4114 passed / 1 skipped / 2 原有 warnings；随后串行 Golden
4 PASS，2943 份历史文件、原 ruler 和 registry 未变。未修改判据或覆盖旧失败。

## 18. 修复计划获批执行：源码完全未改，生成阶段拒绝

2026-10-16，用户明确批准计划 `c507d78d6545` 的 17906-byte 新载荷、失败模型源码/
数值反馈外发与远端同 UID 隔离限制。真实目录为
`output/workload_codegen_h20/workload-codegen-_conj61k/`。只调用一次模型，返回合法
JSON 与 955-byte 源码，但该源码与父候选逐字节一致；固定守卫
`repair_unchanged_source` 拒绝，结果 `generation_failed`、accepted=false、exit=1。

模型声称采用了更稳定的两阶段计算，但原代码本就先求均值再求中心化方差，实际
未改掉 padding 参与方差的问题。rationale 不作为正确性或源码改动证明。
Verifier、compile、数值验证、suite、主测、确认和原生心跳均为 0，无入库；实际
18 commands / 4 gpu_info RPC，准备 7 阶段/15 回执通过，owned 清理确认。

正向 generation audit 保持 completed=false；另行只读核对原始响应、精确源码/
父血缘、预算、usage、claim 和清理，未恢复资格。3208 份历史文件、原 ruler 与
registry 未变，生产源码未改，不回填旧测试成绩，无重试或 commit/push。
计划与父修复 claim 均消费，不能给同一父候选换计划重试。详见
[真实修复执行记录](bounded-numerical-repair.md#5-获批真实执行模型原样返回失败源码修复守卫拒绝)。

## 19. 随后离线实现：带源码证据的诊断，不再直接重试模型

新增真实响应 fixture、AST 伪修复检查与 Repair Diagnoser：保守绑定 masked load、
均值、中心化、未屏蔽平方和及归一化输出的直线 def-use 链。诊断附源码位置、独立
三个标量的数学反例和修复要求，unsupported 返回 unknown，不授予任何资格。

显式 repair v2 上下文增加诊断/hash 血缘，默认普通生成与 v1 声明不自动加字段。
离线 preview 不支持 execute，不访问模型、SSH、GPU、输入物化或 claim；当前已
耗尽的父修复额度保持不变，没有新可执行计划。详情见[离线 Repair Diagnoser](repair-diagnoser.md)。

新增 49 项离线回归；专项 379 passed，全量 4163 passed / 1 skipped / 2 原有
warnings，随后串行 Golden 4 PASS。3439 份历史文件、原 ruler、registry、已消费
claim 均不变；本阶段真实模型/SSH/GPU 派发为 0，无 commit/push。

### 后续离线：预览证据可只读重算，不产生验收资格

新增 preview audit，重建并核对 frozen inputs → diagnosis/source spans → generation
context 的绑定。v2 预览有只读文件清单/hash 和生产者源码快照，v1 不伪造缺失的
快照证明；两种格式的历史/新预览都已只读复核。completed 仅表示内部一致，不
代表源程序曾执行或已被人批准，报告不授予候选资格，不重置已消费的 claim。
本阶段仅离线实现/验证，未产生新模型/GPU 请求或真机执行计划。

专项 422 passed，全量 4206 passed / 1 skipped / 2 原有 warnings，随后串行
Golden 4 PASS；新增 43 项离线回归，3511 份历史文件与原 ruler/registry/claim
未变，生产 protocol 未改。本阶段未 commit/push。
