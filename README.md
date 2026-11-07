# Operator Agent

一个 LLM 驱动的 **GPU 算子开发 agent**：输入自然语言或结构化请求（matmul / softmax /
attention 等），由 LLM agent 设计、生成、修复 Triton / CUDA 核，经三级验证
（L1 编译 → L2 功能 → L3 性能 + Roofline）后产出**可审计**的结论。

- 版本：`0.7.2`，采用 [DSH 外层组装上下文与迭代](docs/design/dsh-host-context-workflow.md)。本地交付状态及工件身份见 [v0.7.2 封版记录](docs/status/v072-release-20261106.md)，能力与限制见 [支持清单](docs/testing/v072-support.md)。已封版的 `0.7.1` 与[安装回退记录](docs/testing/v07-delivery.md#12-v071-本地封版)保留。已发布：PyPI [`dsh-operator-agent==0.7.2`](https://pypi.org/project/dsh-operator-agent/) / npm `dsh-operator-agent@0.2.5`；GitHub `main` 与版本标签已推送。
- 当前开发与封版剩余事项见 [v0.7.2 封版待办](docs/status/v072-todo.md)。
- [v0.7.2 技术方案与图示](docs/design/dsh-host-context-workflow.md)包含总体架构图、开发流程图、恢复时序图和模块职责映射。
- [完整执行示例与日志](docs/examples/v072-h20-softmax.md)：真实 DSH 外层在 H20 上生成 Softmax、完成验证和独立终评，并附修复与同会话恢复实例。
- [v0.7.2 自助试跑指南](docs/testing/v072-quickstart.md)：本机配置、可复制的 ReLU 提示词、日志查看与取消/恢复。
- 固定 DSH 模型路由已完成[真实模型与 H20 验收](docs/status/v071-model-relay-acceptance-20261105.md)，转发仍需显式启用，性能提升未确立。历史 `0.7.0` 的[冻结支持清单](docs/testing/v07-support.md)保持原样。
- 运行模式：DSH 外层驱动上下文、生成与迭代；独立 CLI 保留无人值守流程
- DSH 默认配置为绑定后的主模型请求执行任务级调用与输出预留硬门；同会话恢复保留原任务和预算，`operator_resume` 返回候选与证据引用。任务结束后可讨论，用户在新一轮直接提出新任务，由 DSH 自动准备独立任务；`/op new` 保留为手动入口。绑定前准备和辅助调用不在该限额内，费用未知。
- DSL：Triton / CUDA 为主；后端接口预留模板型后端

## 30 秒开始

默认开发版本为 **Python 3.12**，由仓库根目录的 `.python-version` 固定；包的最低兼容版本仍为 3.9。
使用 uv 时可运行 `uv venv --python 3.12`，然后激活 `.venv`。

```bash
# 依赖极少，本仓库没有 requirements.txt
pip install numpy            # 必需：参考实现 / MockGPU
pip install torch triton     # 仅真机执行需要

# 全部单元测试（离线，不碰 GPU / LLM）
python -m pytest tests/ -q

# 零依赖演示：亲眼看一下「脊柱」的判定
# ⚠️ 它默认用 output/kernel_db.jsonl 那个 KernelDatabase（会往语料追加 retrieved 事件），
#    所以先把三个落点指到仓库外 —— 与跑测试同一个理由：
OPERATOR_AGENT_OUTPUT_DIR=/tmp/oa-demo/out \
OPERATOR_AGENT_LOG_DIR=/tmp/oa-demo/log \
OPERATOR_AGENT_KERNEL_DB=off \
python3 benchmarks/demos/demo_spine_local.py

# 真机端到端（需要 GPU + ~/.ssh/config 里的别名）
python test_nvidia_matmul.py
```

> ⚠️ 请始终用 `python -m pytest tests/`：`pyproject.toml` 已设 `testpaths`，裸跑也只收集
> `tests/`，不会扫到根目录那几个需要真机的 driver 或 vendored 的 `triton/` 检出。

日常循环不必跑全量（全量约 12 分钟，含真实 DSH/沙箱子进程）：`tests/acceptance` 整组和
LeetCUDA 评分器子进程自测标了 `slow`，`-m "not slow"` 可跳过；进程/截止/取消敏感的用例标了
`serial`，并行运行时须单独串行。常用组合：

```bash
python -m pytest -q -m "not slow and not serial" -n 4 --dist loadfile    # 日常，约 80 s
python -m pytest -q -m "slow"                                             # 验收组，约 4 min（普通终端）
python -m pytest -q -m "serial"                                           # 时序组，约 20 s
python -m pytest -q                                                        # 全量（封版前），约 12 min
```

> 需要 `pytest-xdist`（`uv pip install pytest-xdist`）。并发度别开到 `-n auto`：
> 本机 11 核实测会让几个依赖时序/共享状态的用例偶发失败（`-n 4` 连跑两遍全绿）。

## 先读哪份文档

本仓把同一个「事实」按**边界**拆成三份，读之前先记住这个分工（设计规范 §0.1）：

| 文档 | 只回答 | 不回答 |
|---|---|---|
| [设计规范](docs/design/autonomous-operator-agent-design.md) | 应该是什么：接口 / 不变量 / 状态机 / schema / 验收标准 | 现在做到了哪一步 |
| [实现状态](docs/status/autonomous-operator-agent-status.md) | 现在是什么、被哪条守卫守着、边界在哪 | 应该是什么 |
| [研究日志](docs/research/CHECKLOG.md) 与[研究索引](docs/research/README.md) | 为什么这么改、哪次实验推翻了什么 | 当前口径（历史结论**可被推翻**） |

除此之外：

- **长期产品路线**：[算子开发助手：可用版本迭代路线](docs/design/operator-development-product-roadmap.md) —— v0.7 基础闭环 → v0.8 优化增强 → v0.9 业务接入 → v1.0 稳定发布 → v1.1 团队复用 → v1.2 批量运行；每版的用户工作流、交付与验收条件，未来版本均为计划
- **方案评审（2026-10-26）**：[算子开发方案评审](docs/design/operator-development-review-20261026.md) —— 当前入口与设计的一致性、生成质量与成本证据、后续改进优先级；评审建议不替代设计规范
- **插件封版待执行计划**：[收尾顺序与验收清单](docs/design/dsh-operator-completion-plan-20261026.md) —— 证据留存、任务取消、资源预检、当前入口验收、整体对照及交付包
  ；[S0 固定起点验收](docs/status/dsh-operator-s0-20261026.md) —— 工作区快照、实际入口能力、历史证据清单与持久归档
  ；[S1 独立留存验收](docs/status/dsh-operator-s1-20261026.md) —— 每轮源码、完整判定、报告、修复/探索证据与回归记录
  ；[S2 取消与中断验收](docs/status/dsh-operator-s2-20261026.md) —— 按实例取消、MCP/CLI 中断、本机清理和预算保留
  ；[S3 资源预检修复](docs/status/dsh-operator-s3-20261026.md) —— 混合精度、配置作用域、明确存储证据与未知状态
  ；[S5 有限范围结项](docs/status/dsh-operator-s5-closeout-20261030.md)与[三分钟演示/简历口径](docs/design/dsh-operator-s5-resume-demo-20261030.md) —— 真实插件闭环、脱敏案例及独立批次结果；完整重复对照未完成，未建立插件整体增益
 ；[S6 安装与交付验收](docs/status/dsh-operator-s6-acceptance-20261030.md) —— 本地工件安装/卸载、离线证据包和演示材料；按用户决定不录制视频
- **操作指南**：[docs/user_guide.md](docs/user_guide.md) —— 安装、示例、`Conductor` 参数、环境变量全表
- **无 GPU 的功能检查**：[FakeBackend 使用说明](docs/testing/fake-backend.md) —— 脚本模型、故障场景、预算/取消与测试结果隔离；真实 GPU 验收另行执行
  ；[一期 A/B 自动执行与回放](docs/testing/phase1-rehearsal.md) —— 八格离线演练、统一预算核对、独立终评和带来源的事件汇总
  ；[模型 / SSH / DSH 入口接入](docs/testing/phase1-connected.md) —— 显式模型与 GPU 身份、共同预算、独立终评；正式同卡对照另行冻结
  ；[v0.7 候选安装与交付](docs/testing/v07-delivery.md) —— Python 3.12、wheel/tgz 干净安装、随包离线示例、导出与回退
- **生成与修复质量**：[共享规约、诊断与固定任务对照](docs/design/operator-quality-improvement.md) —— 插件修复闭环、离线回放和真实评测入口
  ；[预检与同候选配对](docs/design/operator-quality-paired-evaluation.md) —— 模型调用前检查环境，生成一次、独立比较两种修复策略
  ；[三候选完整 L2 筛选](docs/design/operator-quality-portfolio.md) —— 三种生成策略、首槽基线、完整费用与失败记录
  ；[十任务 H20 实测](docs/research/operator-quality-portfolio-20261024.md) —— 首槽 8/10、组合 9/10，区分数值失败与预检未知
  ；[边界修复验收](docs/research/operator-quality-boundary-fixes-20261024.md) —— 三项固定问题验证通过，新生成首槽 2/2、全部候选 3/6
  ；[编译与列 stride 诊断](docs/design/compile-repair-diagnosis.md) —— 绑定实际编译节点、可信单列输入与 rank=2 寻址证据
  ；[重复故障修复对照](docs/research/operator-quality-diagnosed-repair-20261024.md) —— 两组各修好 2/2，新增诊断覆盖 2/2，额外修复收益 0
- **DSH 插件（判定层工具面）**：[实现说明](docs/design/dsh-plugin-implementation.md)（契约/状态机/拒绝码/扩展点）
  与[方案评估](docs/dsh-plugin-evaluation.md)（为什么是这个形态、真机读数）；
  可安装壳在 [integrations/dsh-operator/](integrations/dsh-operator/README.md)，
  安装形态（profile / 组合包 / 手工条目的区别，为什么 GUI 看不见）见
  [DSH profile 与组合包](docs/design/dsh-profile-and-bundles.md)
  ；[内层/外层模型的分工边界](docs/design/hitl-generation-boundary.md) —— 写核与搜索为什么归内层、
  这条边界目前只是建议性的、以及启用生成工具的判据
  ；[H20 冻结验收（2026-10-30）](docs/status/dsh-operator-gpu-freeze-20261030.md) ——
  v0.6.0 的 4096³ `valid` L3、§14 冒烟与 launch/冲刷复标
  ；[GPU2 中断与后续任务验收](docs/status/gpu-interrupt-recovery-20261030.md) ——
  常驻 SSH 默认精确清理，真实执行中取消/超时后两个新任务均完整 L2 12/12 通过
- **阅读路线**：设计规范 §0.2 给了按依赖顺序的读法；**实现状态 §五** 是一张
  **机制 → 代码位置 → 守卫**的对照表，它才是本项目的目录
- 根目录**没有** `CLAUDE.md`：原件的规范 / 状态 / 历史混在一起，已按治理规则（§25）归档到
  [docs/archive/CLAUDE-md-archived.md](docs/archive/CLAUDE-md-archived.md)

### 磁盘上什么在仓里、什么不在

工作树约 1.0 GB，其中 905 MB 是 `triton/` 本地检出；1.0 GB 的审计证据（各批次冻结计划、
候选、判定回执、GPU 快照）**不在仓内**，仓里只有 `output` → `~/op-agent-evidence/output`
的符号链接。为什么这样：那些证据既不可 diff，也不该被 `git status`/`rg`/`find` 每次扫过，
而 `.gitignore` 让任何提交都不会提醒它正在膨胀。位置、重建方式与注意事项见
[docs/research/evidence-location.json](docs/research/evidence-location.json)；
布局由 [tests/hygiene/test_repo_layout.py](tests/hygiene/test_repo_layout.py) 守住（根目录白名单、
平铺目录条目预算、缓存与产物禁止入库、证据目录只能是链接）。

## 代码怎么读

规范 §1–§19 的编号是**稳定契约**：代码注释按号引用它（想复算：
`grep -rho '设计 §[0-9]' operator_agent/*.py | wc -l`，本轮 111 处）。所以读任何模块的第一步，
是读它 docstring 顶部的 §号，翻到规范那一节，再回来看代码。

逻辑依赖顺序是 **先可信 → 再可搜 → 最后可生成**。推荐这条路线：

包里按**层**分五个子包（2026-10-31 起；`operator_agent/` 根目录只剩 `__init__.py`，
公开再导出不变 —— `from operator_agent import Conductor` 照旧）：

1. `core/` —— 词汇与产物：[schemas.py](operator_agent/core/schemas.py)（`TaskSchema` /
   `SourceSpec` / `TargetSpec` / `ConstraintSpec`）、`contract.py`、`run_contract.py`、
   `artifact_store.py`、`kernel_db.py`
2. `decide/` —— 判定：`verifier.py` → `measurement.py` → `eligibility.py`，
   以及 `correctness*`、`anticheat`、`reference*`、`resource_check`
3. `execute/` —— 执行：[remote_executor.py](operator_agent/execute/remote_executor.py)、
   `exec_guard.py`、`task_budget/cancellation/clock`、
   [remote_scripts/run_kernel.py](remote_scripts/run_kernel.py)
4. `generate/` —— 生成与修复：[agents.py](operator_agent/generate/agents.py) →
   `coder_pipeline.py` → `fixer.py` / `repair_*` →
   [prompts/](operator_agent/prompts) + [skills/triton_skill.md](operator_agent/skills/triton_skill.md)
5. `app/` —— 入口与编排：[conductor.py](operator_agent/app/conductor.py)（`Router` 与主循环
   `_execute_inner()`。**别顺序读**，3000+ 行，用实现状态 §五 的表跳读）、`cli.py`、
   `mcp_server.py`、`hitl_tools.py`、`autonomous.py`

依赖方向是单向的：`app → generate → decide → execute → core`（由
[tests/hygiene/test_repo_layout.py](tests/hygiene/test_repo_layout.py) 与各层 docstring 守着）。

## 目录结构

```
operatorAgent/
├── operator_agent/          # 核心包（按层分组，根目录只有 __init__.py）
│   ├── core/                # 词汇/契约/产物：schemas contract run_contract artifact_store kernel_db …
│   ├── decide/              # 判定：verifier measurement eligibility correctness* anticheat reference* …
│   ├── execute/             # 执行：remote_executor exec_guard task_budget task_clock mock_gpu …
│   ├── generate/            # 生成与修复：agents coder_pipeline fixer repair_* prompt_manager …
│   ├── app/                 # 入口与编排：conductor cli mcp_server hitl_tools autonomous
│   ├── ops/  prompts/  skills/   # 资源与内嵌子包
│   └── __main__.py          # `python -m operator_agent` 的委托入口
├── remote_scripts/          # 远端执行载荷（run_kernel.py）
├── tests/                   # 按用途分三层（见下）
│   ├── conftest.py  fixtures/  golden_regression.py  # 共享夹具与 golden harness
│   ├── unit/                # 208 个单模块/单机制用例
│   ├── acceptance/          # 50 个批次验收（S4/S5/S6 协议、网关、隔离）
│   └── hygiene/             # 仓库守卫：打包、凭据扫描、布局预算、契约冻结
├── benchmarks/              # 按用途分目录（见下）
│   ├── probes/  verify/  calibration/  measurement/  demos/
│   ├── drivers/             # 真机驱动（*_h20、test_nvidia 同族）
│   └── experiments/s4 s5 misc/   # 批次实验与服务
│   └── _scratch.py _remote_bench.py …   # 多个驱动共用的私有辅助
├── test_nvidia_*.py         # 真机端到端 driver：唯一的 LLM 全流程入口
├── docs/                    # design / status / research / user_guide
├── output -> ~/op-agent-evidence/output   # 审计证据的**符号链接**，!不! 在仓内
├── logs/                    # 运行期日志（gitignored）
└── triton/                  # 本地 vendored 检出（gitignored）
```

## 配置

`config.yaml` **不是生效配置** —— 全仓没有 `import yaml`，它零消费者。真正生效的是三处：
`SSHExecutor.__init__` 的默认值、driver 里的 `build_llm()` / `ssh_config()`、以及环境变量。
[config.example.yaml](config.example.yaml) 保留为这些默认值的**可读索引**（改环境时请三处同步）。

常用环境变量（全表见 [user_guide](docs/user_guide.md) §4.2）：

| 变量 | 作用 |
|---|---|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | LLM 端点与凭据 |
| `REMOTE_SSH_HOST` / `_PORT` / `_USER` / `_KEY` | 真机连接（推荐用 `~/.ssh/config` 别名） |
| `OPERATOR_AGENT_OUTPUT_DIR` / `OPERATOR_AGENT_LOG_DIR` | 产物与日志落点（跑测试时指到仓库外） |
| `OPERATOR_AGENT_KERNEL_DB` / `OPERATOR_AGENT_FAILURE_DB` | 经验库路径；取 `off` 则关闭持久化 |

## 仓库纪律（改代码前请读）

- **凭据不进仓**：真实主机名 / IP / 用户名 / 密钥路径一律走 `~/.ssh/config` 与环境变量。
  [tests/hygiene/test_repo_hygiene.py](tests/hygiene/test_repo_hygiene.py) 会机械扫描所有已跟踪文件里的凭据形态
  （含家目录绝对路径与真实端口），命中即红。
- **`config.yaml` 不进仓**：它含真实值，已被 `.gitignore` 忽略；模板是 `config.example.yaml`。
- **根目录的 `_*.py` 是临时脚本**：由 `.gitignore` 的 `/_*.py` 忽略，**不要 `git add -f`**
  （历史上含过旧凭据）。
- **跑测试不要污染仓库**：`Conductor` 无条件落盘，见 [tests/conftest.py](tests/conftest.py)
  如何把 `output/` 与 `logs/` 重定向到 tmp。
- **改实现要留守卫**：本仓的验收标准是「一条能变红的守卫」，而不是一句描述 ——
  规范章节 → 守卫的对应关系见[实现状态 §五](docs/status/autonomous-operator-agent-status.md)。

## 回归与真机

```bash
python tests/golden_regression.py compare   # exit 0 无回归 / 1 有回归 / 2 期望过期需人签认
python3 benchmarks/demos/demo_spine_local.py      # 零依赖：看脊柱判定（会写 output/，见上文 ⚠️）
python3 benchmarks/demos/smoke_spine_h20.py       # 真机版（要 GPU + ~/.ssh/config 别名）
```

Golden 的 CPU 对照保留真实计时：matmul 比较 GFLOPS，零 GFLOPS 的内存算子比较平均延迟。
`compare` 默认将 mock 产物/日志写到临时目录并关闭经验库持久化，不污染真实运行现场；
旧失败基线或缺失的延迟仍返回 `REVIEW`（exit 2），不会自动刷新 registry 或放宽门槛。

## 版本与依赖

- Python >= 3.9
- **没有 `requirements.txt`**：`numpy` 必需；`torch` / `triton` 真机执行才需要；
  LLM 客户端只用标准库 `urllib`（零依赖）

## License

**Apache-2.0**，见 [LICENSE](LICENSE)。版权：`Copyright 2026 the operator-agent authors`（见 [NOTICE](NOTICE)）。
第三方组件（ECharts / Mermaid）保留各自许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
