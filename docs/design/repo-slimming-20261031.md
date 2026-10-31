# 仓库瘦身：已完成与未完成（2026-10-31）

状态：**Phase 0/1（体积出仓 + 布局守卫）、Phase 2（`operator_agent/` 分包）、Phase 3（`tests/`、`benchmarks/` 分目录）全部完成并通过全量回归。**

全量回归：`python -m pytest tests/ -q` → **6185 passed, 1 skipped**（含新增的层依赖棘轮两条）。插件侧 `npm test` **58/58**。

## 1. Phase 0/1：体积与守卫

| 项 | 前 | 后 |
|---|---:|---:|
| 工作树体积 | 2.0 GB | 约 1.0 GB（其中 905 MB 是 `triton/` 本地检出） |
| `output/`（审计证据） | 1.0 GB 与源码同级 | 移到 `~/op-agent-evidence/output`，仓内留符号链接 |

证据位置、重建方式与「单副本」警告写在 [evidence-location.json](../research/evidence-location.json)。
守卫在 [tests/hygiene/test_repo_layout.py](../../tests/hygiene/test_repo_layout.py)：根目录白名单、
单层条目预算、缓存/产物/证据禁止入库、`output/` 只能是符号链接、证据位置说明必须存在。
守卫能变红（往根目录扔 `_layout_probe.py` 即红），已实测。

**为什么用符号链接而不是搬走删掉**：`docs/**` 里有 94 处 `../../output/...` 链接，另有四个验收
测试按路径读证据。链接让这些引用全部照旧，同时工作树里不再有那 1.0 GB。

## 2. Phase 3：tests/ 与 benchmarks/ 分目录

**tests/**：264 个平铺文件 → 一层 4 个共享件 + 三个子目录。

| 目录 | 内容 | 数量 |
|---|---|---:|
| `tests/unit/` | 单模块、单机制用例 | 208 |
| `tests/acceptance/` | 批次验收（S4/S5/S6 协议、网关、隔离、工具面） | 50 |
| `tests/hygiene/` | 仓库守卫：打包、凭据扫描、布局预算、契约冻结 | 4 |

**benchmarks/**：142 个平铺 → 按用途分目录，根目录只留 `__init__` 与五个私有辅助。

| 目录 | 内容 |
|---|---|
| `benchmarks/experiments/s5/`（60）、`s4/`、`misc/` | 批次实验、服务、网关与会话驱动 |
| `benchmarks/drivers/`（22） | 真机驱动（`*_h20`、`test_attention/layernorm/matmul`、`harness_pass_rate`） |
| `benchmarks/probes/` `verify/` `calibration/` `measurement/` `demos/` | 探针、守卫型核验、标定、测量与零依赖 demo |

搬迁过程中修掉的三类真实坏点（都不是「文件放错」这么简单）：

1. **仓库根推导**：`parents[1]` / `parent.parent` / 双 `os.path.dirname` 是按老深度写死的，
   移动后指到 `tests/` 或 `benchmarks/`。一次性全局替换曾把 `operator_agent/exec_guard.py`
   的 `_REPO_ROOT` 顶到 `~/repos`，表现是子进程候选全部 `rc=1` crashed —— 已回退并留下记录。
2. **在仓内记录「哪个文件」的路径字符串**：变异规格、golden 登记、驱动清单里写的
   `tests/test_x.py` / `benchmarks/x.py` 是**数据**而不是 import，正则改 import 改不到它们。
3. **冻结身份清单**：`workload_decision_h20.evaluator_sources()` 与
   `workload_llm_search_h20.sources()` 把文件名变成哈希键。分目录后键值从
   `benchmarks/x.py` 变成 `benchmarks/drivers/x.py` ⇒ **这些 workload 的 evaluator 身份变了**。
   不是判据或数值改动，但引用旧身份的冻结记录不能与新身份互换。

顺带修了一处守卫本身：`test_shared_host_scratch_paths` 原先 `glob("*.py")` 平铺扫描
`benchmarks/`，分目录后会**静默漏扫**子目录；改成 `rglob`。同时把 docstring 排除出扫描
（它是说明文字，与注释同性质）—— 否则 `leetcuda_bench` 的用法示例 `--out /tmp/r.json`
会被当成运行时落点。

**历史文档不跟着改**：`docs/research/**` 与带日期的 `docs/status/**` 是当时快照，里面的
`tests/test_x.py` 保留原样。当前入口类文档（`README`、`user_guide`、实现状态 §五 的
机制→代码→守卫表）已随迁移更新。

## 3. Phase 2：`operator_agent/` 分包（未执行）

现状：**99 个 `.py` 平铺在一层**，是全仓最后一处大平铺。

建议分五层，边界按「词汇/判定/执行/生成/入口」：

| 层 | 收纳 |
|---|---|
| `core/` | `schemas` `contract` `interface_contract` `run_contract` `artifact_store` `run_layout` `kernel_db` `candidates` `environment` `tolerance` `source_lang` |
| `decide/` | `verifier` `measurement` `eligibility` `correctness*` `anticheat` `reference*` `input_fidelity` `resource_check` `oracle` `quality_*` |
| `execute/` | `remote_executor` `exec_guard` `backend_*` `mock_gpu` `task_*` `budget` `*_budget` `placement` `*_client` |
| `generate/` | `agents` `coder_pipeline` `single_candidate_coder` `fixer` `repair_*` `bounded_repair` `generation_*` `prompt_manager` `planner` `optimization_planner` `selection` `search_objective` `bandit` |
| `app/` | `conductor` `cli` `mcp_server` `hitl_tools` `autonomous` `intent_parser` `report` |

执行时的四条硬约束（这次 Phase 3 已经踩过一遍）：

1. **`run_contract.protocol_identity()` 的 73 文件清单按路径写死**，且被 178 处引用。
   分包必须同批改这份清单，并明确记录「协议身份变了」——冻结的验收记录不能与新身份互换。
2. **`__file__` 相对推导**：`prompt_manager` 的 `prompts/`/`skills/`、`exec_guard._REPO_ROOT`、
   `artifact_store.output_dir`、`kernel_db` 的默认库路径、`optimization_planner` 的技能文件，
   都要按新深度逐处核对，**不要**再跑一次全局正则。
3. **导入重写要分形态**：`from operator_agent.X import`、`from .X import`（包内相对）、
   `import operator_agent.X`、`"operator_agent.X.Y"` 字符串（`mock.patch` 目标与 `-m` 参数）
   是四种不同形态，上一轮漏掉后三种各造成过一批红。
4. **分步验证**：先 `--collect-only`（几秒）再跑全量（8 分钟）；每层搬完就提交一次，
   不要攒到最后。

建议顺序：`core` → `decide` → `execute` → `generate` → `app`，每步一次提交、一次全量。
`operator_agent/__init__.py` 的公开再导出保持不变（`from operator_agent import Conductor`
这类外部用法不能断）。

## 4. 复核

```bash
python -m pytest tests/ -q                      # 全量：6183 passed, 1 skipped
python -m pytest tests/hygiene -q               # 布局/打包/凭据/契约冻结
cd integrations/dsh-operator && npm test        # 插件宿主契约 58 条
shasum -a 256 output/... 2>/dev/null || true    # 证据在 ~/op-agent-evidence/output
```

## 5. Phase 2 完成记录（2026-10-31）

98 个平铺模块 → 五个层子包，包根只剩 `__init__.py`（公开再导出不变）与 `__main__.py`（`python -m operator_agent` 的委托入口）。

| 层 | 模块数 | 收纳 |
|---|---:|---|
| `core/` | 23 | `schemas` `contract` `interface_contract` `run_contract` `artifact_store` `run_layout` `report` `kernel_db` `corpus_*` `candidates` `confirm` `environment` `tolerance` `source_lang` `*_templates` `execution_audit` `*_caps` `workload` `achievable` `novelty` |
| `decide/` | 30 | `verifier` `measurement` `eligibility` `correctness*` `anticheat` `reference*` `input_fidelity` `session_inputs` `resource_check` `oracle` `compile_evidence` `tool_evidence` `bench_cases` `peak_probe` `calibration` `cuda_*` `quality_*` `failure_route` `exhaustion` `escalation` |
| `execute/` | 15 | `remote_executor` `exec_guard` `backend_*` `mock_gpu` `placement` `task_budget` `task_cancellation` `task_clock` `budget` `llm_*` `s5_*_budget` `s5_verify_client` |
| `generate/` | 21 | `agents` `coder_pipeline` `single_candidate_coder` `fixer` `repair_*` `bounded_repair` `generation_*` `prompt_manager` `triton_*` `planner` `optimization_planner` `selection` `search_objective` `bandit` `agent_logger` `agent_runtime` |
| `app/` | 9 | `conductor` `cli` `mcp_server` `hitl_tools` `autonomous` `intent_parser` `tool_calls` `__main__` `_harness_controls` |

### 分包过程中修掉的真坏点（都不是「文件放错」这么简单）

1. **身份清单的键变了**：`quality_evaluation.evaluator_identity()`、
   `workload_decision_h20.evaluator_sources()`、`s5_gateway_real_preflight._identities()`
   用具名文件当哈希键，分目录后键从 `x.py` 变成 `<层>/x.py` ⇒ **这些实验的 evaluator 身份变了**。
   不是判据或数值改动，但引用旧身份的冻结记录不能与新身份互换。
2. **入口路径**：`pyproject` 的 console scripts 指向 `operator_agent.app.cli/mcp_server`；
   `python -m operator_agent` 由包根 `__main__.py` 委托。插件宿主契约测试原来探
   `import operator_agent.mcp_server`，分目录后**静默 skip** —— 改成探公开入口 `import operator_agent`。
3. **远端部署契约**：`remote_executor._remote_files()` 的本地路径要带层、远端部署名必须保持裸名
   （runner 按裸模块名 import）；`run_kernel.py` / `runner_server.py` 的仓库内 fallback 改走包路径。
4. **平铺扫描静默变瞎**：多处守卫用 `glob("*.py")` 扫包，分包后只会看到 `__init__.py`。
   全部改 `rglob` 并把标签改成包内相对路径（`decide/verifier.py` 这种），否则「扫不到 ⇒ 绿」。

### 新增守卫：层依赖棘轮

`tests/hygiene/test_repo_layout.py::TestLayerDirection` 钉住方向 `app → generate → decide → execute → core`。
当前有 **23 处向上依赖**（几乎全是函数内延迟导入，为避免循环导入），全部逐条登记在
`UPWARD_IMPORTS`：新增一处即红，拆掉一处也要求同步删表（`test_the_ratchet_table_has_no_stale_rows`）。

### 预算下调

`operator_agent: 4`、`core: 30`、`decide: 35`、`execute: 20`、`generate: 26`、`app: 12`
（实际 1/23/30/15/21/9）。`tests`、`benchmarks` 的预算沿用 Phase 3 的值。

### 未处理的已知项

- 一次全量跑里出现过 1 条顺序相关的偶发（`test_hitl_tools::...gets_two_consequences`），
  单独跑、`tests/unit` 整目录跑、`acceptance + 该用例` 组合跑均绿，复跑全量未再现。
  未定位到污染源，记为待办而不是当作噪声。
- `operator_agent/ops/` 与 `prompts/`、`skills/` 仍在包根（资源目录与内嵌子包），预算里按目录计。
