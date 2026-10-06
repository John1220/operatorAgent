> ⚠️ **已归档（2026-10-03），不再维护，也不参与工作区指令加载。**
> 这是仓库根目录 `CLAUDE.md` 的原文。它被移走的原因不是内容错，而是**承载方式错了**：
> 它把规范、实现状态与历史变更混在同一份文件里（治理规则见设计规范 §25），
> 于是 `git diff` 不可读、更正只能写进正文，而且它的自述会与代码漂开。
>
> 当前口径以这些为准：
> - **规范**：`docs/design/autonomous-operator-agent-design.md`
> - **实现状态与守卫**：`docs/status/autonomous-operator-agent-status.md`
> - **研究日志**：`docs/research/CHECKLOG.md`
> - **操作指南**：`docs/user_guide.md`
>
> 已知的失实之处（例如"`config.yaml` 是主配置文件"、"per-host locks 防止并发污染"）
> 记在状态文档的「已知的文档/代码不一致」一节，**以那一节为准**。

---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Operator Agent (v0.5.0) is an LLM-driven GPU kernel code generation framework. It takes a natural-language or structured request for a GPU operator (matmul, softmax, attention, etc.), uses LLM agents to design and generate kernel code (Triton/CUDA/Python), then verifies it through a three-level pipeline (compile → functional → performance). Results are saved to `output/` and can be executed remotely on GPU servers via SSH.

## Build, Test, and Run

```bash
# Install dependencies (no requirements.txt; dependencies are minimal)
pip install numpy torch triton

# Run the end-to-end test (NVIDIA GPU matmul via Triton)
python test_nvidia_matmul.py

# Run all unit tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_p1.py -v
# or directly:
python tests/test_p1.py

# Run a single test case
python -m pytest tests/test_coder_pipeline.py::TestPlanStage::test_planner_outputs_structured_plan -v

# Quick smoke test — no GPU or LLM needed
python -c "
from operator_agent import MockGPU, KernelDatabase
db = KernelDatabase()
print(db.search(algorithm='matmul', hardware='nvidia_gpu', dsl='triton')[0].kernel_id)
"
```

There is no `requirements.txt`. Core dependencies: `numpy`, `torch`, `triton` (the last two only needed for real GPU execution). The LLM client uses only Python stdlib (`urllib`).

## Architecture

```
User Request (NL or structured)
  → IntentParser (LLM or rule-based → TaskSchema)
  → Conductor.execute()
      ├── Router: decides CODE_TRANSLATION / COMPILER_BACKEND / AGENT_GENERATION
      ├── DesignerAgent (LLM): TaskSchema + RAG → UnifiedSketch (tiling, memory plan, hints)
      ├── CoderAgent (LLM): Sketch → kernel code (with pre-validation + retry loop)
      │     └── CoderPipeline (default, P1-1): Planner → 3 parallel samples (T=0.1/0.4/0.7)
      │           → compile-gate as early-exit pruning (≤2 finalists) → stateful repair
      │           (≤2 rounds) → rule template fallback. Selection among finalists happens
      │           at the caller: the Conductor runs L1+L2 on each finalist and keeps the
      │           first fully-correct one; L3 runs only on the winner (the splice skips
      │           re-running the winner's L2; the all-fail path reuses the default
      │           candidate's L1+L2 as the Fixer evidence instead of re-running it).
      │           **Selection is by correctness only** — when several finalists pass L2
      │           the temperature-order first one wins and the faster one is never timed:
      │           a deliberate machine-time tradeoff, not an oversight, so a "first-correct
      │           beats fastest" reading of the winner is the correct one.
      │           `use_coder_pipeline=False` is the legacy
      │           single-call escape hatch.
      ├── Verifier: L1 (contract gate → syntax → compile) → L2 (functional + anti-cheat) → L3 (perf + Roofline)
      │     └── L3 output is tiered by measurement validity; the tier decides what the
      │         number may drive (see "Decision eligibility" below)
      ├── FixerAgent (LLM): on L1/L2 failure, repairs code with error logs
      └── SearchAgent (LLM + rule fallback): on L3 failure, adjusts tiling/hints
  → ArtifactStore.save() → output/<kernel_type>/<hardware>/<timestamp>_<id>/
  → KernelDatabase.add() only if persist_decision() passes (see below)
```

**Decision eligibility** (`operator_agent/measurement.py`). The recurring failure mode this
guards against is "a signal was computed but a downstream consumer treated it as trustworthy"
— e.g. a speedup inflated by launch overhead permanently blacklisting a working recipe.
Validity is tiered `valid / degraded / uninformative`, from five signals: three ask *is this
measurement accurate* (launch-floor dominance, sample dispersion, pre-measure GPU util) and
two ask *is this measurement even about this thing* (provenance — a `Cap` the task declared in
`TargetSpec.required_caps` that the backend does not declare, or a payload that self-reports
`mode="simulated"` / `device="cpu"` while the backend claims `REAL_DEVICE`; environment
fingerprint drifted from the previous run) — exposed as **uses**, not as a tier:

- `can_gate` — may this number gate a pass/fail verdict at all (`not uninformative`)
- `can_disable_recipe` — **only `valid`**; recipe disabling is permanent, so ties and
  no-evidence verdicts must not trigger it
- `can_count_stagnation` — **only `valid`**; a contaminated round must not end the loop
- `l3_pass_is_refused` — deliberately *not* folded into `can_gate`: a low tier means "this
  number cannot support a conclusion" and has never rewritten `passed`, whereas a provenance
  shortfall means the number was **never measured**, so "passed" has no referent. Only fires
  when the task actually required `REAL_DEVICE` (a bare contradiction degrades but does not
  refuse, otherwise honest local mock runs would be judged dead)

Related rules: round-to-round comparison is a four-state `verdict()` (`FAST/SAME/SLOW/
AMBIGUOUS`) with a per-tier noise band, and only `SLOW` disables recipes; a roofline
convergence exit exists but stays closed when the peak denominator is unverified;
`persist_decision()` refuses to write a passing-but-untrustworthy kernel into the corpus.

**Anti-cheat and interface contract**: `anticheat.py` statically blocks library calls in
generated code and `remote_scripts/run_kernel.py` wraps *every* candidate execution in a
runtime `shield_torch()` (both are required — a `return A @ B` fake kernel once passed all
three levels). `interface_contract.py` checks, at L1 and before any GPU time, that the
kernel's parameter names match the all-keyword call convention `kernel(**inputs)`; it errs
toward passing when it cannot tell (`**kwargs`, defaulted params, no `def kernel`). The same
module also **broadcasts** that convention into every generation prompt (before the model
writes a line) so the names come from the harness, not from the Planner's guess.

**Key data structures** (`operator_agent/schemas.py`):
- `TaskSchema` — the universal contract: `SourceSpec` (kernel type, complexity), `TargetSpec` (hardware, DSL), `ConstraintSpec` (precision, min speedup, max error)
- `UnifiedSketch` — Designer output: algorithm, `TilingStrategy` (block_m/n/k), `MemoryPlan`, pipeline stages, optimization hints
- `KernelArtifact` — final output: code, sketch, verification results, benchmark data, iteration lineage

**Agent roles** (each consumes SKILL sections from `skills/triton_skill.md`):
- **Designer** (sections 4, 5): produces tiling strategy and optimization hints from hardware info + RAG
- **Coder** (sections 1, 2, 5, 6): generates code from sketch, with pre-validation (truncation/bracket/syntax checks) and up to 3 retries with feedback
- **Search** (section 3): analyzes Roofline metrics; the LLM suggestion is a **prior**, not an adoption (P1-4) — the tiling lattice neighborhood (±1 step per dim, triton DSL only) is ranked deterministically (memory-bound → larger BLOCK_K first, compute-bound → smaller BLOCK_M/N first, unknown → closest-to-prior first), the winner becomes the next sketch and the rest queue for round-robin in the Conductor (real L3 runs are the judge; LLM is not re-asked while the queue is non-empty). Falls back to `TritonOptimizationRules` if LLM fails
- **Fixer**: repairs L1/L2 failures using error logs + `FailureDatabase` (negative samples)

**Execution backends** (`operator_agent/remote_executor.py`):
- `SSHExecutor` — deploys `remote_scripts/run_kernel.py` to a remote GPU server, sends JSON over SSH stdin, receives JSON on stdout. Uses `uv` for environment management. `get_gpu_info()` values are **derived from the probe or from `hardware_caps`**; if the probe fails it **raises `HardwareProbeFailed`** rather than returning a spec dict, and any field with no source in either is reported as `"unknown"`.
- `HTTPExecutor` — talks to a remote HTTP API. That protocol has **no hardware endpoint**, so `get_hardware_info()` raises and `capabilities` is the **empty set** (it cannot be `require()`d by real-device consumers).
- `LocalExecutor` — runs kernels locally on CPU
- `RemoteExecutor` — unified facade that auto-selects SSH → HTTP → Local
- `MockGPU` — local simulation backend (no GPU needed); falls back to NumPy reference implementations when CUDA is unavailable

**Verification** (`operator_agent/verifier.py`):
- L1: interface-contract check (static, no GPU) → syntax check → remote compilation (with SHA256 cache to skip re-compiles)
- L2: functional correctness — runs kernel, compares against PyTorch/NumPy reference, with an **absolute + relative AND** tolerance check tiered by precision dtype. Supports multi-input suites. Anti-cheat (`operator_agent/anticheat.py`) blocks `torch.matmul`/`F.softmax` etc. in generated code
- L3: benchmark vs. a baseline **selected on the remote side** (control trusts only `baseline_selected`; for fp32 matmul the first choice is TF32-enabled eager, since a TF32-off baseline inflates speedup ~1.9x). Candidate and baseline are timed **block-interleaved** with L2 flushes; outputs include peak-memory-per-call for both arms, per-block speedup medians, SM clock/throttle readings, and the measurement-validity tier
- Baseline note: `RemoteExecutor` must delegate `ensure_environment`/`deploy` — the P1-3 environment pre-check silently never ran on the real-hardware path before this was fixed

**Routing** (`conductor.py` Router):
- Fused ops or HIGH complexity → `AGENT_GENERATION`
- Target hardware doesn't support the DSL → `AGENT_GENERATION`
- LOW complexity → `CODE_TRANSLATION`
- MEDIUM + RAG hit → `COMPILER_BACKEND`; no hit → `AGENT_GENERATION`

**Knowledge bases** (`operator_agent/kernel_db.py`):
- `KernelDatabase` — seeded with Triton matmul/softmax/attention examples; score-based retrieval weighted by DSL match, token overlap, performance. Since v0.6.x it is an **append-only JSONL event log** (`output/kernel_db.jsonl`, override with `OPERATOR_AGENT_KERNEL_DB=off` to disable) with a usage counter, and writes go through `persist_decision()` — a kernel that passed but whose measurement can't gate, was preempted, or didn't beat the baseline is refused. Retrieval has readers (Designer, Coder, Fixer); **ranking does not yet prefer the historical best**
- `FailureDatabase` — negative samples, also append-only JSONL. Each failure carries a
  task-independent `failure_signature` = `level | error_class | innermost_frame`, and two gates
  decide whether a record may become a reusable lesson: a **recurrence gate** (default 3
  distinct `task_id`s, `OPERATOR_AGENT_FAILURE_GATE`; applies only to records replayed from
  disk) and a **class gate** that excludes numeric-tolerance and infrastructure failures —
  the only thing learnable from "error exceeded threshold" is "loosen the check"
- Both default paths live under `output/` (gitignored): the corpus is what *this* machine
  measured, so it is not portable

**Cross-run lineage**: `KernelArtifact.iteration_history` records per round the action
(`generate`/`optimize`/`fix`), pass/fail, gflops, applied recipes, and an α-equivalent
structural signature of the candidate (`operator_agent/novelty.py`) with the round it
duplicates — **observation only**, nothing is skipped on the basis of it.

**Prompt system** (`operator_agent/prompt_manager.py`):
- Templates in `operator_agent/prompts/*.md` with `{{variable}}` substitution
- SKILL spec in `operator_agent/skills/triton_skill.md` — parsed by `## N.` sections, each section auto-injected into the matching agent's prompt via `{{skill}}` placeholder
- Section-agent mapping: §1-2,5-6→Coder, §3→Search, §4-5→Designer

## Configuration

`config.yaml` is the primary config file. Key settings:
- `llm.base_url`, `llm.model`, `llm.temperature` (default 0.3), `llm.max_tokens` (4096)
- `remote.mode`: `ssh` / `http` / `local` / `auto`
- `optimization.max_rounds`: 3 (max LLM optimization cycles per task)
- `optimization.min_speedup`: 1.0 (minimum speedup vs. baseline for L3 pass)

Connection settings stay env-driven: `LLM_API_KEY`, `REMOTE_SSH_HOST/PORT/USER/KEY`,
`REMOTE_WORK_DIR`, `REMOTE_HTTP_URL` (see `config.example.yaml`; `~/.ssh/config` aliases are
the expected way to supply host details, so no real addresses ever land in the repo).

Runtime switches (env only, no code change): `OPERATOR_AGENT_GPU` (**honoured only by the remote
runner** — `SSHExecutor` never reads it: it auto-picks the least-busy card in-process and passes
*its own* choice down, so setting it locally does nothing; the harness report takes the actual card
from the backend via `selected_gpu`, see doc 49 §7.12), `OPERATOR_AGENT_KERNEL_DB` / `OPERATOR_AGENT_FAILURE_DB` (JSONL paths, or
`off`/`none`/`0`/`false` to disable persistence), `OPERATOR_AGENT_FAILURE_GATE` (recurrence
threshold, default 3 distinct tasks), `OPERATOR_AGENT_OUTPUT_DIR` and
`OPERATOR_AGENT_LOG_DIR` (`off` = never write to the repo) + `OPERATOR_AGENT_LOG_KEEP`
(default 10). The last group exists because `Conductor` writes unconditionally, so a test run
used to flood `output/` and rotate away real experiment logs — 5316 of 5326 stored snapshots
were unit-test mock sites. Full table: `docs/user_guide.md` §4.2; what each gate does: §6.

The LLM client (`operator_agent/llm_client.py`) is a zero-dependency OpenAI-compatible client using only `urllib`. It disables Qwen3 thinking mode (`chat_template_kwargs={"enable_thinking": False}`) to prevent JSON/code truncation, and has robust JSON extraction with auto-repair of common formatting errors.

## Key Patterns

- **LLM output is always cleaned**: code goes through `_clean_code()` / `_extract_code()` which strips thinking tags, markdown fences, dedents, and removes docstrings/comments before validation
- **Pre-validation before compilation**: `CoderAgent._validate_code()` checks for truncation (unbalanced brackets, trailing operators), missing `def kernel`, missing `@triton.jit` — avoids wasteful SSH round-trips for obviously broken code
- **Anti-cheat**: `static_audit()` scans for forbidden `torch.matmul`/`F.softmax` calls; `shield_torch()` runtime-monkeypatches them to raise errors during execution
- **Serialized remote execution**: per-host locks prevent concurrent benchmark pollution on shared GPUs
- **Roofline everywhere**: L3 injects hardware peak GFLOPS/bandwidth, arithmetic intensity, and utilization % into metrics for SearchAgent to make data-driven optimization decisions
- **Negative sample learning**: the Conductor feeds L1/L2 failures + Fixer results into `FailureDatabase`; the Fixer retrieves similar past cases to inform repairs
- **Pipeline mode is the default** (`use_coder_pipeline=True`, P1-1): CoderPipeline = Planner → 3 parallel candidates → compile-gate pruning (≤2 finalists) → repair loop → rule template fallback; finalist selection by real L1+L2 happens in the Conductor. `use_coder_pipeline=False` keeps the legacy single-call Coder path
