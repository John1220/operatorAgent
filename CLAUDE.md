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
      │     └── Optional CoderPipeline: Plan → 3 parallel samples (T=0.1/0.4/0.7)
      │           → compile-select first-pass → stateful repair (≤2 rounds) → rule template fallback
      ├── Verifier: L1 (compile) → L2 (functional + anti-cheat) → L3 (performance + Roofline)
      ├── FixerAgent (LLM): on L1/L2 failure, repairs code with error logs
      └── SearchAgent (LLM + rule fallback): on L3 failure, adjusts tiling/hints
  → ArtifactStore.save() → output/<kernel_type>/<hardware>/<timestamp>_<id>/
```

**Key data structures** (`operator_agent/schemas.py`):
- `TaskSchema` — the universal contract: `SourceSpec` (kernel type, complexity), `TargetSpec` (hardware, DSL), `ConstraintSpec` (precision, min speedup, max error)
- `UnifiedSketch` — Designer output: algorithm, `TilingStrategy` (block_m/n/k), `MemoryPlan`, pipeline stages, optimization hints
- `KernelArtifact` — final output: code, sketch, verification results, benchmark data, iteration lineage

**Agent roles** (each consumes SKILL sections from `skills/triton_skill.md`):
- **Designer** (sections 4, 5): produces tiling strategy and optimization hints from hardware info + RAG
- **Coder** (sections 1, 2, 5, 6): generates code from sketch, with pre-validation (truncation/bracket/syntax checks) and up to 3 retries with feedback
- **Search** (section 3): analyzes Roofline metrics, suggests new tiling; falls back to `TritonOptimizationRules` if LLM fails
- **Fixer**: repairs L1/L2 failures using error logs + `FailureDatabase` (negative samples)

**Execution backends** (`operator_agent/remote_executor.py`):
- `SSHExecutor` — deploys `remote_scripts/run_kernel.py` to a remote GPU server, sends JSON over SSH stdin, receives JSON on stdout. Uses `uv` for environment management.
- `HTTPExecutor` — talks to a remote HTTP API
- `LocalExecutor` — runs kernels locally on CPU
- `RemoteExecutor` — unified facade that auto-selects SSH → HTTP → Local
- `MockGPU` — local simulation backend (no GPU needed); falls back to NumPy reference implementations when CUDA is unavailable

**Verification** (`operator_agent/verifier.py`):
- L1: syntax check + remote compilation (with SHA256 cache to skip re-compiles)
- L2: functional correctness — runs kernel, compares against PyTorch/NumPy reference. Supports multi-input suites. Anti-cheat (`operator_agent/anticheat.py`) blocks `torch.matmul`/`F.softmax` etc. in generated code
- L3: benchmark vs. PyTorch eager baseline (same-device on remote), Roofline analysis (arithmetic intensity, ridge point, utilization %)

**Routing** (`conductor.py` Router):
- Fused ops or HIGH complexity → `AGENT_GENERATION`
- Target hardware doesn't support the DSL → `AGENT_GENERATION`
- LOW complexity → `CODE_TRANSLATION`
- MEDIUM + RAG hit → `COMPILER_BACKEND`; no hit → `AGENT_GENERATION`

**Knowledge bases** (`operator_agent/kernel_db.py`):
- `KernelDatabase` — seeded with Triton matmul/softmax/attention examples; score-based retrieval weighted by DSL match, token overlap, performance
- `FailureDatabase` — stores (error → fix) pairs; searched by FixerAgent to avoid repeating past mistakes

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

Environment variables: `LLM_API_KEY`, `REMOTE_SSH_HOST/PORT/USER/KEY`, `REMOTE_WORK_DIR`, `REMOTE_HTTP_URL`.

The LLM client (`operator_agent/llm_client.py`) is a zero-dependency OpenAI-compatible client using only `urllib`. It disables Qwen3 thinking mode (`chat_template_kwargs={"enable_thinking": False}`) to prevent JSON/code truncation, and has robust JSON extraction with auto-repair of common formatting errors.

## Key Patterns

- **LLM output is always cleaned**: code goes through `_clean_code()` / `_extract_code()` which strips thinking tags, markdown fences, dedents, and removes docstrings/comments before validation
- **Pre-validation before compilation**: `CoderAgent._validate_code()` checks for truncation (unbalanced brackets, trailing operators), missing `def kernel`, missing `@triton.jit` — avoids wasteful SSH round-trips for obviously broken code
- **Anti-cheat**: `static_audit()` scans for forbidden `torch.matmul`/`F.softmax` calls; `shield_torch()` runtime-monkeypatches them to raise errors during execution
- **Serialized remote execution**: per-host locks prevent concurrent benchmark pollution on shared GPUs
- **Roofline everywhere**: L3 injects hardware peak GFLOPS/bandwidth, arithmetic intensity, and utilization % into metrics for SearchAgent to make data-driven optimization decisions
- **Negative sample learning**: the Conductor feeds L1/L2 failures + Fixer results into `FailureDatabase`; the Fixer retrieves similar past cases to inform repairs
- **Pipeline mode** (`use_coder_pipeline=True` on Conductor) enables the CoderPipeline: Planner → 3 parallel candidates → compile-gate → repair loop → rule template fallback
