# 无人值守算子开发 Agent —— 实现状态

| 项 | 值 |
|---|---|
| 文档性质 | **状态**（不是规范，也不是研究日志） |
| 规范 | [`docs/design/autonomous-operator-agent-design.md`](../design/autonomous-operator-agent-design.md) |
| 研究日志 | [`docs/research/CHECKLOG.md`](../research/CHECKLOG.md)、`docs/research/*.md` |
| 记录口径 | 每条状态必须指向**可运行的守卫**（测试文件 / 产物 / 真机运行编号），不指向人写的段落 |
| 生效分支 | `design/autonomous-spine` |

> **为什么状态要独立成文**：规范与状态混在一个文件里，`git diff` 就不可读，行为变更会被状态更新淹没，
> 更正只能写进历史单元格（"上次误判已撤回"），而读者必须读完几百字才知道现在的口径。
> 规范只回答"应该是什么"，本文只回答"现在是什么、被什么守着、边界在哪"。

## 状态记号

| 记号 | 含义 |
|---|---|
| ✅ | 已落地，且**有守卫**（测试或产物） |
| ⏳ | 部分落地 / 有判据无输入端 / 有实现无真机证据 |
| ❌ | 未落地 |
| ⚠️ | 已落地但**边界必须一并读**，单看 ✅ 会读大 |

---

## 一、按设计节的状态

| 设计节 | 产物 | 状态 | 守卫 | 边界（必须一并读） |
|---|---|---|---|---|
| §3.1 契约冻结 + 哈希 | `operator_agent/contract.py`（`freeze_from_task` / `load_dict` 校验载荷 hash） | ✅ | `tests/test_contract_freeze.py` | — |
| §3.2 数值语义 | `contract.py` 的 `DTypeSemantics`；缺失即拒绝，不猜默认值 | ✅ | `tests/test_arithmetic_intensity_dtype.py` | — |
| §3.5 安全不变量 | 契约哈希 / 反作弊 / 接口契约 / 同卡配对 / 心跳 / 日志均为无条件生效 | ⚠️ | `tests/test_spine_wiring.py`（31 条，含"没有开关"与"消融只能 harness 侧 patch"）、`benchmarks/mut_specs/double_confirm_invariant.py`（**6 行全拦住**：5 行为 + 1 结构，0 逃逸） | 「长期记忆的独立确认」**2026-10-03 已从默认关的构造参数翻成不变量**（`all_passed` 分支无条件补一次重测 + 判据侧写死 `requires_remeasure=True`；传旧参数名抛 `TypeError`；见 [CHECKLOG I79](../research/CHECKLOG.md)）。⚠️ 同日的**逐项审计**（[CHECKLOG I80](../research/CHECKLOG.md)）结论：**没有第二处默认关的安全装置**（`operator_agent/` 内零 `argparse`、零配置文件读取），但有**三处可被非人化输入关掉**的门：①**静态反作弊白名单的输入是任务字段** `task.target.hardware` —— **2026-10-03 已收**（白名单只认执行侧声明 `VENDOR_TORCH_OK`，子串匹配器连同 `dc`/`ls`/`ft`/`rv` 名单删除；今天没有声明者 ⇒ 静态与运行时**两眼一致地拒绝**；6/6 变异，见 [CHECKLOG I81](../research/CHECKLOG.md)。⚠️ 代价要一起读：厂商卡上「torch 合法」那条路今天**两半都不通**（运行时盾本来就在所有生产调用点恒开），要开通必须两半一起接，仍欠）②五条环境变量关"写"（`LOG_DIR=off` / `KERNEL_DB=off` / `FAILURE_GATE=0` …，设计意图是测试逃生门）③`heartbeat_interval` 无上限、仓内 benchmark 正用 999 关掉心跳（§3.5 规则 3 不许的形状）⇒ **本行不许读成"全部无条件生效"** |
| §3.6 CONTRACT_LINT | 六项清单 | ✅ | `tests/test_contract_freeze.py` | 清单里"形状分布覆盖边缘形状"这一条此前**只对性能负责**，正确性形状集不是硬要求（§20 已补）。⚠️ 但**形状分布本身今天没有消费者** —— 见 §四 的 §3.4 行 |
| §4.3 五位资格 + 规则 8 | `operator_agent/eligibility.py`、`source_lang.py`（适用性三态）、`anticheat.AuditVerdict`、`interface_contract.InterfaceVerdict` | ✅ | `tests/test_eligibility.py`、`tests/test_guard_applicability.py`；18/18 变异红（CHECKLOG I65 / [doc 77](../research/77-静态守卫的适用性.md)） | 反作弊白名单的输入是**执行侧声明**（`verifier.VENDOR_TORCH_OK`，今天无生产者 ⇒ 恒 False；任务字段进不来，[CHECKLOG I81](../research/CHECKLOG.md)）。两道静态守卫是 **Python 形状**的：对 C++/CUDA 模板源明确判"不适用" ⇒ DEGRADED；"未观测"不改判。**今天没有任何后端能执行非 Python 源**，所以端到端不可观察；`cuda_c` 只有文本覆盖（看不穿宏与头文件） |
| §4.3 rule 3 vs 6 | `measurement.py`：`l3_pass_is_refused` 与降级分开 | ✅ | `tests/test_measurement_validity.py` | `l3_pass_is_refused` 只在契约真要求 `REAL_DEVICE` 时开火（裸矛盾只降级，否则诚实的本地 mock 会被判死） |
| §10.5-1 标定心跳 | `operator_agent/calibration.py`（成对参照双比值） | ✅ | `tests/test_calibration.py`、`tests/test_mutation_calibration.py` | 心跳**只记账、不被预算拦**（饿死标定 = 把"尺子有没有坏"变成缺钱就能跳过的事） |
| §10.5-4 两次确认 | `operator_agent/confirm.py` + 唯一判据 `oracle.admit_to_corpus` | ✅ | `tests/test_confirm.py`、`tests/test_spine_wiring.py`（31 条：默认配置下没有重测**不入库** / **没有开关**（传旧参数名抛 `TypeError`、实例上无可改布尔）/ 默认路径上生产者真的跑了 / 消融只能在 harness 侧 patch 且判据仍要求着） | **2026-10-03 起它是 §3.5 的不变量**（[CHECKLOG I79](../research/CHECKLOG.md)）：`all_passed` 的产物**无条件**先补一次独立重测（`_produce_remeasure`）再过第 3 道门（生产写死 `requires_remeasure=True`）；`admit_to_corpus` 的参数只服务 §14-9 的消融臂，docstring 明文"不许接到配置文件/环境变量/命令行/任务字段上"。历史：2026-10-01 入库判据收成一处（此前 `oracle.admit_to_corpus` 零生产调用者，`_save_to_db` 另写一遍同族三道门且口径不同）；"打开即入库率归零"的自锁已由生产者 + 字段 `artifact.remeasure` 配平（[spine-review](../design/spine-review.md) P2-10）。⚠️ **守卫覆盖缺口**：全仓没有"入库成功（`corpus_admission.kept=True`）"的正向端到端对照 —— 控制机上 `MockGPU` 只声明 `EMULATED`/`SIMULATED_PERF`，读数按设计被判 uninformative，入库在更前面那道门就该被拒 ⇒ 要验"入库成功"得有一台真卡 |
| §13.4 Oracle 收口 | `operator_agent/oracle.py`，接 L3 出口（`verifier._assess_measurement`） | ✅ | `tests/test_oracle.py`、`tests/test_l3_provenance.py` | 入库判据**只有一处**（此前零生产调用者的那一段是收口点不在链路上的 P0） |
| §8 失败分类路由 | `operator_agent/failure_route.py`（五路 + `fix_not_rewrite`） | ✅ | `tests/test_failure_route.py`、`tests/test_prescriptive_errors.py` | 五路动作由**内容**（而不是类别名）决定，`fix_not_rewrite` 直接给出、不靠调用方自觉 |
| §8.2 误差形态 | `tolerance.classify_error_shape`（规则）+ 两个生产者：`verifier._error_shape_of`（numpy）与 `remote_scripts/run_kernel._error_shape_torch`（torch）+ `failure_route.ERROR_SHAPE_HINTS` | ⚠️ | `tests/test_tolerance_error_shape.py`（19 条，含**两个生产者一致性**用例与"形态文本不许撞上失败类关键词"用例）、`tests/test_leetcuda_report_format.py`（远端白名单）、`tests/test_stage1_acceptance.py::Acceptance10` | 已实现五种：`nonfinite` / `whole_tensor` / `tail_block` / `whole_row` / `broadcast_axis`。**判不了 ⇒ `None`（三态），理由文本里也说**。⚠️ 两条边界：① 设计 §8.2 的第六种 `precision_accumulation`（误差随 k 增长）**没有量具** ⇒ 恒为未确立；② **昆仑芯那条 runner**（`remote_scripts/run_kernel_xpu.py`）的数值门只有绝对项、也没有形态 —— 它靠 §3.2 的精度拒绝把这件事挡在前面（只接受 fp32），不是靠形态。形态是**证据不是判据**：它进 message / Fixer 首行 / 路由理由，**不改路由决策**。触发条件是"判失败"而不是"有坏点"（NaN 时坏点计数为 0，否则 `nonfinite` 永远报不出来） |
| §9.3 双键（编译 / 测量缓存） | `placement.py` 的 `BuildKey`（可跨卡）与 `PartitionKey`（绝不可跨卡） | ✅ | `tests/test_placement.py` | `PartitionKey` 今天只有 device/arch/driver/runtime/backend_ver，**无 harness / warmup / 采样协议指纹**（§9.4 已补规范） |
| §10.2 三种预算 | `operator_agent/budget.py`（`BudgetPlan` + `BudgetLedger`）+ 轮首两道门 | ⚠️ | `tests/test_budget.py`、`tests/test_budget_gate.py` | **三种预算默认全关（opt-in）**；`explore`/`refine` 分相已落地（`phase_room`/`phases_with_room`，计划被突破只留痕不停运行）；**按 GPU 秒的预算未做**（今天只有次数）；心跳只记账不被拦 |
| §10.3 候选选择 | `operator_agent/selection.py`（`score_candidate` + `choose`）+ `bandit.py` + 轮首选择 | ✅ | `tests/test_selection.py`、`tests/test_cost_persistence.py`；19/19 变异红（I63 / [doc 75](../research/75-分数与ε-greedy与分相预算.md)） | ε 抽签是 `(seed, 轮次)` 的纯函数 ⇒ 逐轮值可离线复算；**分母 = 本臂平均测量秒（`bandit.mean_cost`），本臂缺失才退回全局均值**；`控制端测量墙钟`说的是**货币**（不是卡上时间），不是全局常数。参数轴今天只有 tiling 三维 |
| §10.3 三条件升级门 | `escalation.py` + conductor 判决点接线 + SearchAgent 发布通道 | ⚠️ | `tests/test_escalation.py`（含 `TestGaugeInventory`）、`tests/test_occupancy_instrument.py`；13/13 变异红（I53→I57 / [doc 69](../research/69-结构升级门的三条件与位置.md)） | 条件 2 今天**两把量具**：`smem-bound`（只在上界 ~×1.2 那条带里成立）与 `occupancy-bound`（2026-10-03 落地：判据 + `max_threads_per_sm` 生产者 + 远端 `num_warps` 编译元数据）。**`spill` / `bank conflict` 仍无量具** ⇒ 缺证据就是门关着。量具清单写 stale 过一次（docstring 说"占用率上限还没有量具"，而兑现它的函数就在同文件里），已修并由 `TestGaugeInventory` 钉住：已实现的那几类不许在任何一句话里被说成"还没有" |
| §10.4 停止条件 | `operator_agent/exhaustion.py`（覆盖判据三态 + 逐轮/运行级账本）+ conductor 停滞出口（两条件**同时**成立才停） | ✅ | `tests/test_exhaustion.py`、`tests/test_stop_reason.py`；14/14 变异红（I64 / [doc 76](../research/76-有希望的区域已穷尽.md)） | **代码型路径恒 `unknown`**（候选按需产生，没有可枚举全集）⇒ 收口靠轮次与 §10.2 预算；空集不算已穷尽。真机已观测（2026-10-03 I74：12 点空间 13 轮，4 次"阈值到但未穷尽 ⇒ 不许宣布停滞"）。⚠️ 小缺口：账本把停止轮也记进 `declined`；升级门逐轮裁决未落 RESULT |
| §10.6 信念门 | `Conductor` 的 `last_tunable_perf` 门；`optimization_history` 与 bandit 回报按 `can_tune` / `can_count_progress` 收口 | ✅ | `tests/test_search_belief_gate.py`、`tests/test_bandit.py` | `can_tune=False` 的轮次**照样跑满预算、不许提前终止**；prompt 不许把"没有读数"渲染成 `0.00` |
| §11 多卡放置 / 同卡配对 | `operator_agent/placement.py`（`DeviceKey` / 干扰域 / 同卡配对 / 观测租约） | ⚠️ | `tests/test_placement.py`、`tests/test_pairing_observation.py`、`tests/test_pairing_window.py`、`tests/test_device_lease.py` | 设备那半读远端逐臂身份；窗口那半读 runner 的 `paired["arm_windows"]`（三态，缺观测不拒评）。**真机上"同窗"由构造保证** ⇒ `cross_window_refused` 尚不可能出现，那道路径靠 fakes 用例与变异证明是活的。非远程路径的控制端参照臂命中 `_baseline_cache` 属于字面跨窗复用 —— **记录不判决**（`baseline_reused`）。多卡并行（`lease_waves` 无多卡消费者）、`interference_domain` 无可观测来源 |
| §11 竞争检测（同卡多 harness） | `remote_scripts/gpu_claim.py`（进程级占卡声明）+ `assess` 第 6 信号 + 租约 `harness_peers` | ✅ | `tests/test_gpu_claim.py`、`tests/test_shared_host_scratch_paths.py` | 修的是"串行化只覆盖进程内（`_EXEC_LOCKS` 是 threading 锁）⇒ 两进程静默共卡，实测 2.1× 膨胀而档位仍 `valid`"。真机验收 4/4（并发双臂 1.84× 且双双 `degraded+contended`）。**边界**：跨进程**避让/互斥**仍无机制（claim 治"看不见"不治"不发生"）；claim 目录在 `~` 只覆盖同用户 harness。**长期单任务串行**（2026-10-03 使用决策）⇒ 避让与并发部署竞争一并缓做 |
| §12 剪枝口径（含 `stages` 轴） | `operator_candidates` → `candidates.py`（`tile_smem_bytes(stages=)` + `stages_of` + `PruneResult.stages_seen`/`stages_unobserved`） | ✅ | `tests/test_stage_axis_prune.py`、`tests/test_proxy_prune.py`；9/9 变异红（I66 / [doc 78](../research/78-stages那一轴进剪枝.md)） | 三态"未观测 ≠ 1 份"（没这一轴或值不可用 ⇒ 按 1 份算并记『下界』）。此前 `stages=4` 按 1 份估算、放不下的点被放行。`warps`/向量宽度/layout/split-K 仍无产出者与剪枝口径 |
| §12 模板后端 / `param_space()` | `candidates.py` + `backend_caps.Backend.param_space`/`instantiate` + 便宜代理剪枝 | ✅ | `tests/test_candidates.py`、`tests/test_template_instantiation.py`、`tests/test_template_queue.py`、`tests/test_triton_template_backend.py` | Triton matmul 真模板后端已通，真机过**完整 Conductor 主循环**（I72 两臂；I73=部署竞争另挂号）。`Backend.instantiate` 默认 raise + 五道守卫 + 批量 `refusals`。L1 两道静态守卫对 C++/CUDA 模板源空转（见 §4.3 行）；参数轴只有 tiling 三维有产出者 |
| §13.5 参考下界（`probe_peak`） | `operator_agent/peak_probe.py` + `HardwareCaps.reference_lower_bound_tflops` 存放处 + `verifier._publish_peak_probe` / 按精度读取 | ⚠️ | `tests/test_peak_probe.py`、`tests/test_achievable_ceiling.py` | 判据七条（仿真读数 / 精度键 / 出处 / 非有限正数 / 探针自己的档位 / 离散度未知 / 没打到算力峰值 / 高于规格 5%），按精度发布，**零额外 GPU 秒**（复用同一次 L3 的基线臂）。**发布的是下界**（基线臂实测可达值，不是器件极限）⇒ 出口偏松；**只在进程内生效**（不跨运行）；`MockGPU` 一律拒发；真机已 fire（2026-10-02 I72 第 3 轮：valid 档 `tf32=70.80 TFLOPS`，refusals 空）。**命名已对齐**（`roofline_converged` → `reference_ceiling_reached`，`peak_measured` → `reference_lower_bound`，96 处含 tests/benchmarks）。⚠️ **器件峰值出口在代码里不存在**：`peak_measured`（独立 compute-bound microbenchmark）还没有量具 ⇒ 只有"贴着参考实现的天花板"这一个出口 |
| §14-10 错误候选门禁 | `tests/test_stage1_acceptance.py::Acceptance10_WrongKernelIsRefused` + `verify_l3_only(correctness_verified=)` 凭据 | ✅ 2026-10-03 | `Acceptance10` 6 条；`tests/test_spine_wiring.py::TestCorrectnessCredentialWiring`（结构守卫，变异已验红）；`tests/test_measurement_validity.py::TestCorrectnessCredential` / `TestL3PassRefusalCauses` | 数值错的候选**止步 L2、不产生 L3**（正常路径），理由指向 `first_bad_index`（尾块用例钉在 `(N-1)*N`）。**唯一绕开 L2 的入口**（`verify_l3_only`，也是晋级层那次重测的入口）现在必须带凭据：`False` ⇒ `passed=False` + 档位 `uninformative` + 五位资格全 False；缺键 ⇒ 不改判但留痕（三态）。凭据由 `Conductor._correctness_credential` 从**那一份候选自己的 L2 结果**读，两个生产调用点结构守卫钉住 |
| §14 验收 | `tests/test_stage1_acceptance.py`（离线）+ `benchmarks/smoke_spine_h20.py`（真机 H20） | ⚠️ | 该文件现有 **33 条用例**（`pytest --collect-only`） | 真机已跑（2026-10-01，8×H20）：0/1/2/3/7/8 绿、4 见下、5 未覆盖（需 LLM）、6 纯离线。**真机抓出两处"算了但没接上"**（I46 发布的资格位不是收口点算的那份 ⇒ §14-8 在生产上没生效；I48 远端运行时盾的证据不落地），都已修 + 守卫。**前 26 条全部落在度量/资格/契约哈希/配对/窗口/心跳上，没有一条喂数值错的候选** —— 第 10 条已补（见上一行） |
| §14-9 资格审计消融 | `benchmarks/ablation_eligibility.py` + `tests/test_ablation_eligibility.py` | ⏳ | 五态判据 + "不许有开关"政策守卫（该测试文件） | **比较器已落地、两臂未跑**：真机对照需要同卡两次完整循环（要 LLM）⇒ "审计有收益"仍不许声称。带宽取 `NOISE_BAND[uninformative]`。裁判组只能在 harness 侧 patch，命令行不留开关 |
| §14-4 竞争降级 | 见 §11 竞争检测行 | ⏳ | 同上 | 真机上"后台占卡"这一臂缺一次成功证据 |
| §17「无法确立」 | `confirm.UnableToEstablish` + `report.py` / `report.md` 产物 | ✅ | `tests/test_report_deliverable.py`、`tests/test_leetcuda_report_format.py` | 2026-10-01 补齐：第 3 件交付物此前**完全没有产物**（收集器只被两个 benchmark 调过）。空章节必须带依据（逐项核过的清单印出来）；`can_persist=False` 单列一行，不进"无法确立"章节；**2026-10-07（[I90](../research/CHECKLOG.md)）起这一节还收 §21.4/§22.3 的覆盖缺口**（有对象、没量具的判据，带逐条 detail；没判过则不说话 —— 三态） |
| §21 Agent Runtime | `operator_agent/agent_runtime.py`（`AgentAction` / `ActionContext` / `PrecompileVerdict` / `ActionVerdict` / `ProposalContext` / `AgentRuntime`）+ Conductor 接线（**Step 4 之前**、所有候选来源汇合之后；裁决进血缘 `agent_action`）。**离线可当场看**：`python3 benchmarks/demo_agent_runtime.py`（零 GPU / 零 LLM / 不联网，六个场景：含 I78 那个洞的离线复现与守卫、以及 §21.1 生产者侧「动作是谁写的」） | ⚠️ | `tests/test_agent_runtime.py`（51 条）、`tests/test_agent_runtime_wiring.py`（12 条，含"被拒的候选一个字符都没到后端"、"**同一个违规候选重复提交也每轮都被拒**"与"裁决进产物"）、`tests/test_demo_agent_runtime.py`（6 条，含"演示自带的断言没炸"与场景 5/6 的读数钉住）、`tests/test_llm_action_producer.py`（24 条：**把信封真的喂进一次 `Conductor.execute`** ⇒ `source == llm` 那一支必须被执行到；三态 / 切段 / 越权种类与工具被拒 / 预算与路径仍是 harness 的 / 三个 ask 种类各有响应与上界 / 续跑丢意图不丢候选）、`tests/test_structured_evidence.py`（15 条：那八段**真的到了 LLM 的 messages 里**、模板变量不漏、开关是干净的两臂、diff 基线取的是本轮的上一份、profiler 只喂白名单里那几项、契约摘要与名单各只有一个真源）；**真机一手证据**：`python3 benchmarks/agent_action_h20.py`（algorithm-h20 + 真 LLM 端点，一次跑 ≈35s：违规候选**零远端载荷**、被拒轮 [2,3]、后端全跑只收到 1 份载荷）→ [CHECKLOG I78](../research/CHECKLOG.md)；`benchmarks/mut_specs/agent_action_gate.py` **27 行全拦住**（26 行为 + 1 结构，0 逃逸，见 [CHECKLOG I76](../research/CHECKLOG.md)）、`benchmarks/mut_specs/structured_evidence.py` **12 行全拦住**（11 行为 + 1 结构，含 I77/I78 的原形：留痕写回局部变量 / diff 基线取错 / 被拒的候选也进依赖基线）、`benchmarks/mut_specs/coverage_states.py` **6 行全拦住**（6 行为，0 逃逸：状态借名 / 集合又并回 / 缺口丢掉 / 报告不读 / 读错轮次 / 没判过被读成有缺口，见 [I90](../research/CHECKLOG.md)） | 三条边界必须一并读：① **这不是沙箱** —— 判的是**声明的**角色路径（`candidate/code.py`，§22.1 的布局；2026-10-07 起过门的候选真的落在这个文件上，见 [I88](../research/CHECKLOG.md)），§22 的目录权限只落地 harness 侧一半、进程隔离未落地 ⇒ 强制是 §22 的事；② **源码级「幻觉 API」判不了**（仓里没有任何语言的 API 面清单）⇒ 进 `AuditCoverage.uncovered_rules`，不许读成"没有幻觉 API"；**2026-10-07（[I90](../research/CHECKLOG.md)）起这一格是 `unknown`（有对象、没量具）而不是 `not_applicable`**，且「我无法确立的事」里有一条专门说它（覆盖缺口第一次有人读）——⚠️ 只是**记对了地方**，量具仍不存在；③ **token 预算那一轴的拒绝只对事前动作生效**：候选是"先生成、后提交"，事后拒绝省不下 token，且会把 §10.2 的"超了退化成纯数值搜索"打成死循环（I76）⇒ 预算问题照记进产物（`budget_deferred_note`）但不据此拒绝。**④ 生产侧落地了一半**（这是本行最容易读大的一处）：**§21.3 的结构化证据今天真的进了修复提示** —— `Conductor.structured_evidence`（默认开）在每次调用 Fixer 前用**唯一拼装点** `AgentRuntime.context_for(...).render()` 生成那八段，经 `fixer.run(structured_evidence=...)` 进 `fixer_repair.md`；注入留痕进血缘 `fix_evidence`（`injected` / `chars` / `truncations` / `profiler_facts` / `parent_diff_lines`），于是 A/B 两臂在**产物上**分得开。三条**仍未落地**的：① 生成侧（`CoderPipeline` 的采样与 `coder_repair`）**不吃**这一段 —— 今天的理由是它会乘以采样份数而收益未对照；② **生产者侧已接线（2026-10-05，见 [CHECKLOG I84](../research/CHECKLOG.md)）**：`AgentAction` 不再由 harness 自己包 —— 模型在回复里夹一段 ```agent-action 信封（`ENVELOPE_FENCE`），解析是三态 `parsed` / `absent`（接线前就是这么回话的 ⇒ 退回 harness 拼装，行为逐字不变）/ `malformed`（多于一段 / JSON 坏 / 类型不对 ⇒ 整段不认），表外**种类**算 `parsed` 并交给判据去拒；`FixerAgent.run_with_action` 用信封替掉回复里那一段（`run` 是它的两格投影），`Conductor` 在**修复轮 → 下一轮**这一格消费（与 `pending_fix` 同生同灭；读点一处、写点四处，AST 计数守卫钉住）。**作者边界**：模型写意图（kind / rationale / bottleneck / tools），harness 写**数字与路径**（`patch.path` 来自 §22.1 的 `candidate_path`、`token_budget` 是真的上限；模型自己在信封里写的预算只留档、`path` 进 `unknown_keys`）。于是 §21.2「数值修复必须带误差形态」从**构造保证**变成**对真实输入的判断**（模型不点名形态 ⇒ 零 GPU 秒被拒，拒绝理由作为下一轮的错误信息回到模型眼前）；三个 ask 种类第一次有生产者：`explain_failure` 答并把解释记进本轮失败记录，`request_profiler` / `request_new_shape` **答不出来**（注册表里是 `None`，如实报 `refused_no_gauge`，不许读成「观测结果是空」），响应**有界**（答/拒一次，再问就停并记 `ask_turns_exhausted`），答复走**声明过的**模板段而不是塞进 §21.3 的八段证据。代价与边界：模型不守协议时白烧一次 Fixer 调用；跨**续跑**只恢复候选不恢复意图（信封不进快照 ⇒ 那一轮必然记 `absent`，宁可退回 harness 拼装也不凭空造一份模型理由）；③ 证据那一段是**追加**在既有「错误信息」与「历史修复案例」之外，**不是替代** —— "筛过的证据 vs 全量日志"的对照**今天已真机跑过**（2026-10-05，algorithm-h20 + 真 LLM，N=3 场景 × 2 臂，三对全部干净：砍段后非空白残差 0、历史案例段两臂皆「无」）：通过 **0/3（开）vs 0/3（关）**、轮数与停止原因两臂相同 ⇒ **可分辨的那一侧只有成本**（证据段 685~924 chars ⇒ 每次 Fixer 调用的 prompt token +334~+387），而 N=3 时通过率量具最多只能分辨 ≈60.1 pp ⇒ ⭐ **收益不许声称，也不许声称无收益**，见 [CHECKLOG I83](../research/CHECKLOG.md)。判据口径：「新增依赖」的基线是**上一次过了门的候选**（`ActionContext.dep_baseline_source` + `Conductor._admitted_baseline`），不是 `Patch` 的父代 —— 父代可能是**上一轮被拒**的那份，拿它当基线就等于把同一份违规候选的第二次提交放行（真机实测过一次：第 3 轮那份夹带 `import torch` 的候选过了门、真的上了远端编译，只有 L2 的反作弊静态审计兜住，见 I78）。只有 `violation` 才拒绝，`unknown` / `not_applicable` 是**结论**不是拒绝理由（与 §4.3 的三态同一条纪律）；7 条设计原文 → 6 条判据（第 1/2 条在单文件候选上是同一个包含关系，第 3 条的契约篡改那半由 `oracle.assert_hash` 承担）；被拒 = 一条**合成的 L1 失败**（`compile_status=agent_action_refused`，零 GPU 秒）⇒ 复用 §8 路由与 Fixer 的处方通道，不新增停止词。结构性变更的许可证仍读 §10.3 的 `_escalation_log[-1]`（这一层不自己判）。**变异标定（2026-10-06 串行跑）：16 行 → 行为 14/14 + 结构 2/2、0 逃逸**（第一版 M16 写的是「另一个名字的属性」⇒ 逃逸，改成同一属性名的多余写点后被 AST 计数守卫杀掉 —— 逃逸的行说明证人写错了，已写进规格） |
| §22.1 目录权限 | `operator_agent/run_layout.py`（七格布局：五个受保护格 `0500`、`candidate`/`build` `0700`；`write_for(role, rel, data)` **唯一写入口**，先判角色再碰文件系统，artifact 一律 `0400`；受保护格写窗 `finally` 复原；`ensure_dir` 走同一判据）+ `Conductor.execute` 建布局（`run_root` 为空则不建）+ `run_layout.new_run_root`（一次运行一个新根）+ `Conductor._materialize_candidate`（过门的候选落 `candidate/code.<ext>`）+ §21 判据同源（`PROTECTED_DIR_NAMES` 只有一份定义）。**离线可当场看**：`python3 benchmarks/demo_run_layout.py`（零 GPU / 零 LLM / 不联网，四个场景：七格与权限位 / 越权写 / 裸写 / 同源） | ⚠️ | `tests/test_run_layout.py`（33 条：形状 / 5×7 角色矩阵 / 裸写 / 逃逸 / 同源 / 单门 AST / 接线 / 可写格子目录回归）、`tests/test_demo_run_layout.py`（6 条）、`benchmarks/mut_specs/run_layout_doors.py`（**9 行全拦住**：8 行为 + 1 结构，0 逃逸，含落地当天自查出的 M9）、`tests/test_run_root_wiring.py`（10 条：根唯一性 / 落盘内容与权限位 / 被拒不留痕 / 唯一写入口有调用者 / 三个真机驱动在传 `run_root`）、`benchmarks/mut_specs/run_root_wiring.py`（**6 行全拦住**：5 行为 + 1 结构，0 逃逸） | **只落地了 harness 侧那一半**（[CHECKLOG I82](../research/CHECKLOG.md)）：越权写与裸写在写的那一刻失败（`PermissionError`/`CrossRoleWrite`），失败不留痕，且 `operator_agent/` 里只有一处开写窗（AST ratchet）。⚠️ **「候选被挡住」今天证不了** —— 候选与 harness **同一个 uid**（本地 `exec` / 远端 runner 落 `/tmp` 后 `exec_module`），属主可自己 `chmod` 回去；远端 runner 侧零覆盖；§22.2 的**本地半片**已落（运行期见 [I85](../research/CHECKLOG.md)、编译期见 [I87](../research/CHECKLOG.md)），仍缺的是**隔离本体**（独立 uid / 网络关闭 / 容器 / 文件系统白名单）与远端侧，厂商卡 runner `run_kernel_xpu.py` 连运行时盾与容差都没有；`run_root` **2026-10-07 起有生产调用点**（[I88](../research/CHECKLOG.md)）：三个真机驱动经 `new_run_root` 一次运行一个新根，**过门**的候选走唯一写入口真的落进 `candidate/code.<ext>`（被 §21 拒的一个字符都不留）⇒ 「§21 判据指着的路径」与「盘上真实产物」终于是同一个文件。⚠️ 但**只搬了候选源码这一件**（`report` / `measurements` / `kernel_db` 未进布局），且同 uid ⇒「候选被挡住」仍证不了、远端零覆盖 ⇒ 形态是「候选有权限位 + 驱动有调用点」，不是「线上已经在拦」。③**§22.3 的覆盖率数据现在有下游了（2026-10-07，[CHECKLOG I89](../research/CHECKLOG.md)）**：`AuditCoverage` 进 `verdict.as_dict()['coverage']`、拒绝理由与血缘（`tests/test_agent_runtime_wiring.py` 读它），并且「覆盖不足影响 `ClaimLevel` 与 promotion」这一半已接线 —— 等级上限（`ast` 不设限 / `regex_only` ⇒ `validated` / `none` 与**没判过** ⇒ `provisional`）由 `eligibility.claim_level` **唯一推导**、`Conductor._save_to_db` **一处写入**、`oracle.admit_to_corpus` 的**第 3 道门**读它（静态事实 ⇒ 排在花机时的两次确认**之前**），报告与「我无法确立的事」各有一处读者。⚠️ 两处边界必须一并读：`uncovered_rules` **不**参与压级（源码级幻觉 API 那一格对**每一份** patch 候选都是 `unknown` ⇒ 拿它压级会让入库率归零 —— 那是**量具缺失**不是候选的问题；2026-10-07 切片 E 起该集合**只收 `unknown`**，「没有对象」不再并进来（[I90](../research/CHECKLOG.md)）⇒ 它同时成了报告「我无法确立的事」里的一条），`promotable` 今天只作**门槛**（本仓「独立确认」与「入库」是同一步） |
| §22.2 执行隔离（本地半片） | `operator_agent/exec_guard.py`（两个公开入口：`run_candidate` 跑候选 / `run_compile` **只编译**）+ `operator_agent/mock_gpu.py`（`_execute_python` 与 `compile`→`_compile_in_guard`）+ `operator_agent/remote_executor.py::LocalExecutor.compile`（**编译那一档两个后端都接了**，语义逐字不变） | ⚠️ | `tests/test_exec_guard.py`（43 条，含 `TestCompileMode` 8 条 + 「只减不增」的进程内 exec 账本 2 条）+ `tests/test_exec_compile_guard.py`（14 条：契约 / 阶段级上限链 / `compile_status` 发布面 / 语料探针归属）+ `tests/test_demo_exec_guard.py`（5 条）+ `benchmarks/demo_exec_guard.py`（零 GPU / 零 LLM / 不联网，四个场景）+ `benchmarks/mut_specs/exec_guard.py`（14 行，自查 14/14 拦住、0 逃逸）+ `benchmarks/mut_specs/compile_guard.py`（**13 行：行为 11/11 + 结构 2/2，0 逃逸**，2026-10-06 串行跑） | **只落了「跑飞/吃满/失控」那一半，不是沙箱**（[CHECKLOG I85](../research/CHECKLOG.md)）：五档状态 `ok`/`timeout`/`limit_exceeded`/`crashed`/`cheat`，强制项逐项报 `enforced` / `not_enforced`（设不上就如实报 + 原始 errno，不许读成「没超限」）。**真拦得住的**：CPU（SIGXCPU rc=-24）/ 单文件大小（Errno 27，文件停在边界字节，**没有** SIGXFSZ）/ 进程数（`EAGAIN`，且是 **per-UID**，不许读成「这个进程可以有 N 个子进程」）/ 超时杀**整个进程组**（孙进程零残留，正常返回也收尾）。**设不上的**：`RLIMIT_AS`/`DATA`/`RSS` 在本机（macOS）内核 `EINVAL`/`EPERM`（libc 直调复现）⇒ 内存这一档只能报 `not_enforced`，本机做不了活体内存看门狗。**顺带修掉一条真缺陷**：产物经 pickle 文件回传，`RLIMIT_FSIZE` 同时管它 ⇒ 第一版 8MB/64MB 把 LeetCUDA 那些**合法**的 64MB 产物截断成 `UnpicklingError`（看着像候选坏了，实际是两个上限打架）⇒ 默认 512MB/256MB + 构造期不变量 `artifact_bytes <= file_size_bytes`。**2026-10-06 新增：编译那一档也进了受控子进程**（[CHECKLOG I87](../research/CHECKLOG.md)）—— 第 5 张词表 `MODES`、`run_compile`（只 `exec` 不跑核）、契约逐字不变、异常跨进程重建带 `exc_type` 与 `syntax_fields`（行号列号不丢）、第三档 `GuardRefusal` → `compile_status=compile_<status>`；§22.2 的**阶段级上限第一次有消费端**（`Conductor(execution_limits=, compile_limits=)` → 后端 → `exec_guard`，默认值唯一出处 `limits_for`）。**不许声称**：不是敌意候选的沙箱（同 uid、**网络没关**、无容器/cgroup、无文件系统白名单）；无 GPU watchdog；无设备租约回收；远端 runner 侧零覆盖；**L1 成功那一趟的逐项强制读数没有发布面**（后端 `compile` 只回 `bool`）⇒「设不上的项如实报」今天只在失败那一支成立；账本里仍有 6 处**候选侧**进程内 `exec`（`LocalExecutor.execute`/`benchmark`、`MockGPU._execute_triton_cuda`/`_benchmark_triton_cuda`/`_benchmark_python`、`reference.load_reference_source`，`tests/test_exec_guard.py` 的 `EXPECTED` 表逐处登记）；编译档默认值 30s/60s 是**暂定、真机未实测**。**变异标定**：`exec_guard.py` 14 行 → **14/14（0 逃逸）**，复跑时把 M12 从结构分母升为行为证人、并修掉 M2 证人的**余量**缺陷（只等 1.3s 而孙进程要「起解释器 + 1.0s」才落盘 ⇒ 机器忙时被读成逃逸）（[I85](../research/CHECKLOG.md)）；`compile_guard.py` 13 行 → 11/11 + 2/2（[I87](../research/CHECKLOG.md)） |
| §23 断点恢复 | 快照原子性 + 追加日志 | ⚠️ | `tests/test_stage1_acceptance.py::Acceptance5_*`、`tests/test_resume_integrity.py`、`tests/test_persistent_session.py` | 快照写一半不损坏、追加日志跨重启存活已验证；**无事件日志 / sequence / 幂等键**（§23 已补规范） |

## 二、Stage 实施顺序上的缺件

| 缺的东西 | 现在的真实状态 | 后果 |
|---|---|---|
| `probe_peak`（实测峰值） | ✅ 2026-10-02 已落地（判据 + 按精度发布 + 按精度读取，零额外 GPU 秒） | 收敛出口不再是"永远关"：同一轮 L3 里基线臂过判据 ⇒ 下一轮起有分母。**仍受的约束**：分母是**下界**（出口偏松）、不跨运行、真机未复跑 —— [doc 70](../research/70-实测峰值的判据与按精度发布.md) |
| `Placement.lease`（多卡放置） | ⏳ **半片**（2026-10-02）：`DeviceLease` 有生产者了（`observed_lease` 落 `metrics["device_lease"]`，三态）；`lease_waves` 给出 `lease(contract, n)` 的形状 | §11 的"异域并行省 wall clock"仍是**纸面承诺**；`interference_domain` 无可观测来源（`can_measure_parallel` 恒保守串行）。配对检查的**窗口**那半已有喂给者（[doc 72](../research/72-同一时间窗那一半的观测与拒评.md)） |

两件都排在 Stage 3 之前 —— 前者决定停止条件能不能用，后者决定多卡是不是真的多卡。

## 三、跨阶段补的洞（都在 CHECKLOG 里有编号）

| 洞 | 状态 | 守卫 |
|---|---|---|
| §4.3 规则 8：静态守卫对非 Python 源空转（fail-open） | ✅ 已收（I65 / doc 77） | `tests/test_guard_applicability.py`，18/18 变异红 |
| §10.2 `phase_caps` 曾是"计划"而不是"约束" | ✅ 已是选择的约束（I63 / doc 75） | `tests/test_selection.py` |
| §11 串行化只覆盖进程内 ⇒ 静默共卡 | ✅ 已修（I68） | `tests/test_gpu_claim.py`，真机验收 4/4 |
| §12 多缓冲未进剪枝口径 | ✅ 已收（I66 / doc 78） | `tests/test_stage_axis_prune.py`，9/9 |
| §10.3 条件 2 曾只有 `smem-bound` 一把量具 | ⏳ 已补 `occupancy-bound`（2026-10-03）；`spill` / `bank conflict` 仍缺。**量具清单的 stale docstring 已修 + 钉住** | `tests/test_escalation.py::TestGaugeInventory`（变异已验红）、`tests/test_occupancy_instrument.py` |
| 生成侧三缺口 G1–G3（I75，真 27B 全链首跑暴露） | ⏳ 登记中，修复是下一片 | `tests/test_coder_pipeline.py`、`tests/test_prompt_layers.py` |
| §22.2 第一项「编译超时」曾是**空的**（编译期在 harness 进程内 `exec`，上游于运行期那道门） | ⚠️ 已落本地两个后端（I87）：`run_compile` + 第 5 张词表 `MODES` + 第三档 `GuardRefusal` + `compile_status=compile_*` + 阶段级上限链；欠账见该行 | `tests/test_exec_compile_guard.py`（14 条）、`tests/test_exec_guard.py::TestCompileMode`、`benchmarks/mut_specs/compile_guard.py`（11/11 + 2/2） |

## 四、已知的文档/代码不一致（都属于"状态"范畴，不属于规范）

| 不一致 | 位置 |
|---|---|
| ~~`escalation.py` 模块 docstring 说"占用率上限还没有量具"~~ | `operator_agent/escalation.py` —— **已收**（清单跟着代码走，`TestGaugeInventory` 钉住） |
| ~~`placement.py` 的注释引用"设计 §15 的缺件表"，而该表已搬到本文~~ | `operator_agent/placement.py` —— **已改**指本文 + 设计 §11.3 |
| ~~代码注释引用"设计 §3.2"，而设计正文此前没有 §3.2 小节~~ | `operator_agent/contract.py` —— **已成立**（§3.2 = 「数值语义显式钉死」） |
| ~~设计 §13.5 的命名与代码不一致（`roofline_converged` / `peak_measured`）~~ | `operator_agent/measurement.py`、`peak_probe.py`、`hardware_caps.py` —— **已收**（96 处改名 + 中文术语对齐；两个生产者与报告措辞逐字同步） |
| ~~设计 §3.5 已把"长期记忆的独立确认"列为**不可配置不变量**；代码仍是构造参数 `require_double_confirm`（默认 `False`）~~ | `operator_agent/conductor.py`、`operator_agent/oracle.py` —— **已收**（2026-10-03 翻转成无条件，四条守卫，见 [CHECKLOG I79](../research/CHECKLOG.md)） |
| **§3.5 逐项审计剩的两处**（[CHECKLOG I80](../research/CHECKLOG.md)；①已于 2026-10-03 收口，见 [I81](../research/CHECKLOG.md)）：①~~静态反作弊白名单的输入是任务字段~~ **已收**；②五条环境变量关"写" ③`heartbeat_interval` 无上限（仓内 benchmark 正用它关掉心跳） | `operator_agent/agent_logger.py`、`kernel_db.py`、`conductor.py`；另有**厂商卡两半一起开通**这件事（静态白名单 + 运行时盾，今天两侧都是拒）也要接 |
| ~~设计 §4.4 `ClaimLevel` 尚无字段~~ **§4.4 已收**（2026-10-07，[CHECKLOG I89](../research/CHECKLOG.md)）：`CLAIM_LEVELS` 六级 + `claim_level` 唯一推导处 + 覆盖上限 + 入库门第 3 道门 + 报告/「无法确立」两处读者；**§17.1 的 `ImplementationDecision` 仍零命中**（它要的 `≥ VALIDATED` 前置今天有了，那个决定本身还没写） | `operator_agent/eligibility.py`、`operator_agent/oracle.py`、`operator_agent/conductor.py`、`operator_agent/report.py` |
| 设计 §6.2 要求判决用成对差值 + 置信区间并可离线复算；代码的 `verdict()` 仍是"比值 vs 按档位噪声带" | `operator_agent/measurement.py` |
| 设计 §13.5 的**器件峰值出口**在代码里不存在（`peak_measured` 需要独立 compute-bound 的 microbenchmark）；今天只有参考天花板出口 | `operator_agent/peak_probe.py`、`operator_agent/measurement.py` |
| **§3.4 形状分布/多 shape 目标**：契约里 `shapes` 字段**只被校验与哈希，全仓没有消费者**；L3 只测 `test_inputs[0]`（`verifier.py` 原话「多组输入套件时，性能基准使用第一组输入」）；「加权几何平均」全仓无 `geomean` 实现 ⇒ **今天的目标函数实际是单形状**，而这正是 §16 反模式里那条「用单形状的加速比作为成功标准」 | `operator_agent/contract.py`（无消费者）、`operator_agent/verifier.py` |
| 研究日志里仍写着旧词（"roofline 收敛"/"实测峰值"）：那批是**历史记录**，按 §25.2 不追改；当前口径以本文与规范为准 | `docs/research/`、`CHECKLOG.md` |
| ~~原 `CLAUDE.md` 说"`config.yaml` 是主配置文件"~~ | **假**：全仓无 `import yaml`，`config.yaml` 无消费者 —— 见 `docs/user_guide.md` §4.1（已按事实改写）与 `config.example.yaml` 顶部。原件归档在 [`docs/archive/CLAUDE-md-archived.md`](../archive/CLAUDE-md-archived.md) |
| ~~原 `CLAUDE.md` 说"per-host locks prevent concurrent benchmark pollution"~~ | **不准确**：那个锁只在**进程内**（`_EXEC_LOCKS` 是 threading 锁）⇒ 两个 harness 进程会静默共卡。已修（I68：`gpu_claim` + `assess` 第 6 信号 + `harness_peers`），见 §一「竞争检测」行 |

## 五、实现导览（机制 → 代码位置 → 守卫）

> 这一节收纳的是**别处没有、但仍成立**的实现事实。原 `CLAUDE.md` 把这批内容与规范、历史
> 混在一份文件里；归档后它们归这里（规范只管"应该是什么"，不复述实现）。
> 换机器/接手时按这张表找入口，不必去读归档件。

| 机制 | 今天的口径 | 代码位置 | 守卫 |
|---|---|---|---|
| **Router 路由决策** | 融合算子或 HIGH 复杂度 → `AGENT_GENERATION`；目标硬件不支持该 DSL → `AGENT_GENERATION`；LOW → `CODE_TRANSLATION`；MEDIUM + RAG 命中 → `COMPILER_BACKEND`，未命中 → `AGENT_GENERATION` | `operator_agent/conductor.py`（Router） | `tests/test_p1.py` |
| **调用约定广播** | 全 keyword 调用 `kernel(**inputs)` 由 harness **在生成前广播进 prompt**（不让模型猜），L1 再静态复核；判不了时倾向放行（`**kwargs` / 带默认值 / 无 `def kernel`） | `operator_agent/interface_contract.py` | `tests/test_interface_contract.py`、`tests/test_calling_convention_broadcast.py` |
| **CoderPipeline 终选口径** | 决赛候选由 Conductor 逐个跑真机 L1+L2，取**第一个完全正确的**；L3 只跑胜者（拼接时跳过重跑胜者的 L2）。**按正确性选，不按速度选** —— 多个候选都过 L2 时温度序第一个赢，更快的那个不会被计时。这是刻意的机时取舍，不是疏漏 | `operator_agent/coder_pipeline.py` + Conductor | `tests/test_coder_pipeline.py`、`tests/test_finalist_selection.py` |
| **SKILL 段 → agent 映射** | `skills/triton_skill.md` 按 `## N.` 切段，经 `{{skill}}` 注入对应 agent 的 prompt：§1-2,5-6 → Coder；§3 → Search；§4-5 → Designer | `operator_agent/prompt_manager.py` + `skills/triton_skill.md` | `tests/test_prompt_layers.py`、`tests/test_prompt_placeholder_contract.py` |
| **Search 的 LLM 建议只是先验** | tiling 邻域（每维 ±1，仅 triton DSL）由规则确定性排序，胜者成为下一份 sketch、其余进队列轮转；队列非空时不再问 LLM | `operator_agent/agents.py`（Search）+ Conductor | `tests/test_search_neighborhood.py` |
| **执行后端差异** | `SSHExecutor`：JSON over stdin/stdout，`uv` 管环境，探测失败**抛 `HardwareProbeFailed`**（不返回 spec 字典），无出处的字段报 `"unknown"`。`HTTPExecutor`：协议**没有硬件端点** ⇒ `get_hardware_info()` 抛异常、`capabilities` 是**空集**（因此无法被 `require()`）。`MockGPU`：无 CUDA 时回落 NumPy 参考实现 | `operator_agent/remote_executor.py`、`mock_gpu.py` | `tests/test_hardware_probe_honesty.py`、`tests/test_backend_resolution.py` |
| **反作弊两层** | 静态 `static_audit()`（AST，能拆 import 别名与 `getattr` 常量折叠）+ 运行时 `shield_torch()` 包裹**每一次**候选执行 —— 两层缺一不可（曾有一个 `return A @ B` 的假核通过三级验证） | `operator_agent/anticheat.py`、`remote_scripts/run_kernel.py` | `tests/test_anticheat_bypass.py` |
| **负样本学习** | 失败签名 = `level \| error_class \| innermost_frame`；两道门：复发门（默认 3 个不同 `task_id`，只对从盘上回放生效）与类别门（数值容差类与基础设施类**不进**学习库）。`FailureDatabase` 与 `KernelDatabase` **同在** `kernel_db.py` | `operator_agent/kernel_db.py`、`failure_route.py` | `tests/test_failure_memory.py`、`tests/test_failure_route.py` |
| **跨运行血缘** | `iteration_history` 每轮记 action / 过没过 / gflops / 应用的配方 / α-等价结构签名及其重复轮次 —— **只观测**，不据此跳过任何东西 | `operator_agent/novelty.py` + Conductor | `tests/test_iteration_lineage.py`、`tests/test_novelty_wiring.py` |
| **编译/测量缓存分区** | `BuildKey`（可跨同 arch 共享）与 `PartitionKey`（绝不可跨卡）**刻意分成两个键**；工具链与测量协议指纹见设计 §9.4 | `operator_agent/placement.py` | `tests/test_placement.py` |

## 六、规范章节的落地分布（哪一节现在有代码）

> 用途: 一句话回答"现在做到设计文档的哪一部分了"。判据是 **grep 命中**, 不是印象 ——
> 下面"未落地"那一组的每个概念在 `operator_agent/` 与 `remote_scripts/` 里都是**零命中**。

### 已有代码 + 有守卫（规范已生效）

```
§1.4/§3.5 安全不变量（8 条里 7 条）  §3.1 契约冻结+哈希      §3.2 数值语义
§3.6 CONTRACT_LINT                  §4.1 观测记录            §4.2 资格五类检查
§4.3 资格推导 9 条 + rule 3b        §8.1 五路失败路由        §8.2 误差形态（5/6 种）
§9.2 两级 oracle                    §9.3 双键分区            §10.1 状态机
§10.2 三种预算                      §10.3 选择器 + 三条件门  §10.4 停止条件（5 个出口）
§10.5 安全装置（4 条里 3 条）        §10.6 信念门             §11.1 DeviceKey
§11.2 同卡配对/窗口/竞争             §12 模板后端 + 剪枝       §13.1–§13.4 接口
§13.5 参考下界 + 天花板出口          §14 门禁 10 条 + 消融协议 §17 交付物 + 报告
§20 正确性门禁（§14-10 + 正确性凭据） §23 快照原子性 + 续跑     §25 文档治理（本仓已照做）
```

### 部分落地（有骨架，缺一件明确的东西）

| 节 | 缺的那件 |
|---|---|
| §3.4 | 形状分布**无消费者**、L3 只测第一组输入、无加权几何平均 ⇒ 目标函数实际是单形状 |
| §6.3 | 探索/精修相位已有，但"三层测量各带不同权限"没有形式化 |
| §9.4 | `PartitionKey` 已有；`ToolchainKey` / `MeasurementProtocolKey` 零命中 |
| §13.5 | 器件峰值出口不存在（`peak_measured` 无量具） |
| §14-9 | 比较器已落、**两臂真机未跑** ⇒ "审计有收益"仍不许声称 |
| §20 | 门禁与凭据已落；`CorrectnessSuite` 进契约、特化/空转探针（`no_op`/`perturb` 零命中）未落 |
| §22 | §22.1 的 **harness 侧一半已落**（七格 + 权限位 + 唯一写入口，见 I82），且 **2026-10-07 起有生产调用点**：三个真机驱动经 `new_run_root` 建根、过门的候选真的落进 `candidate/`（见 I88）—— 但只搬了候选源码这一件，`report`/`measurements`/`kernel_db` 未进布局；§22.2 的**本地半片已落**（运行期的受控执行器 + 逐项强制读数，见 I85；**编译期那一档**见 I87 —— 两个后端的 `compile` 都走 `run_compile`，阶段级上限链 `Conductor` → 后端 → `exec_guard` 第一次有消费端）—— 但它挡的是**跑飞/吃满/失控**，**不是隔离**（同 uid、网络没关、无容器/cgroup/白名单；内存本机设不上）；**候选侧那一半仍缺**（同一 uid ⇒ 权限位拦不住属主）；仍**零命中**的是：GPU watchdog / 设备租约回收 / 网络默认关闭 / 文件系统白名单 / 远端运行布局权限位，以及**候选侧**剩下的 6 处进程内 `exec`（`LocalExecutor.execute`/`benchmark`、`MockGPU._execute_triton_cuda`/`_benchmark_triton_cuda`/`_benchmark_python`、`reference.load_reference_source`；编译期已不在其列）；§22.3 的**两半已收**（2026-10-07，见 I89）：覆盖 `mode` → 声明等级上限 → 入库门第 3 道门 → 报告与「我无法确立的事」，守卫 43 条 + 变异 8 行 0 逃逸（`uncovered_rules` 只进留痕/报告、**不**压级；`promotable` 今天只作门槛） |
| §23 | 快照/续跑已有；**事件日志 / `sequence` / 幂等键**零命中 |

### 规范已写、代码零命中

```
§3.3 扩展后的 OperatorContract      §6.2 统计协议（成对差值+CI）
§11.3 多卡能力最小闭环               §17.1 ImplementationDecision    §20.1/§20.3 CorrectnessSuite / 探针
§22.2 进程与资源隔离的**其余几项**（GPU watchdog / 设备租约回收 / 网络默认关闭 / 文件系统白名单）  §23.1 append-only 事件日志
§24 记忆治理（MemoryEntry / TTL / 撤销）
```

**当前所在位置**：Wave 0「规范收口与安全默认值」—— 治理那一半（§25）已做完，
安全默认值那一半的**翻转 2026-10-03 已收**（§3.5 的「长期记忆的独立确认」不再可配置，
见 [CHECKLOG I79](../research/CHECKLOG.md)），同日逐项审计出**三处可被非人化输入关掉**的门，其中
硬伤一（静态反作弊白名单被任务字段打开）**当天已收口**（[CHECKLOG I81](../research/CHECKLOG.md)），剩两处未修（I80）；正确性轴（§20）已开了一个口子
（§14-10 + 正确性凭据）；**隔离轴（§22）开了第一个口子**：§22.1 的目录布局与权限位（harness 侧一半，见 I82）已落 —— 越权写在写的那一刻失败、§21 的路径判据与布局同源；但候选与 harness 同一个 uid，「候选被挡住」要 §22.2，远端的运行布局一个字都没说。
**运行时轴（§21）开的口子更大了**：动作协议 + §21.4 的独立检查
已在主循环的候选汇合点生效（27 行变异、0 逃逸），**而且动作不再由 harness 自己包** ——
模型在回复里交动作信封，harness 只写数字与路径（见 [CHECKLOG I84](../research/CHECKLOG.md)；16 行变异（14 行为 + 2 结构，**2026-10-06 串行标定 0 逃逸**）+ 24 条守卫 + 演示场景 6）。两条边界照旧要一并读：它的强制力是**声明的**路径，
沙箱只有 §22.2 的**本地半片**（挡跑飞/吃满，不是隔离；运行期 14 行 + 编译期 13 行变异串行标定 0 逃逸，见 [I87](../research/CHECKLOG.md)）；源码级幻觉 API 也判不了。
**「筛过的证据 vs 全量日志」已经有真机两臂读数了，但结论是「分辨不出」**：N=3、三对干净、0/3 vs 0/3、轮数与停止原因两臂相同，可分辨的只有成本（+334~+387 prompt token/次）⇒ 收益与无收益都不许声称（量程 ≈60.1 pp），见 [CHECKLOG I83](../research/CHECKLOG.md)。
