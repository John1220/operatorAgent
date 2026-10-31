"""§21 真机一手证据: **被拒的那一轮确实零 GPU 秒** (algorithm-h20 + 真 LLM 端点)

要证的一句话只有一个: §21 的拒绝发生在**送编译之前**, 于是那一轮**一次远端往返都没有**。
单看「零」是证不了的 —— 零也可能是「这段代码根本没接上」。所以这一跑同时收三条:

  * **活的对照**: 同一跑里别的轮**真的**在远端编译过/执行过 (proxy 记录逐次调用 +
    真的 L1 失败原话), 且依赖探针 (`analyze_dependencies`, 走 SSH) 真的为第一份坏码跑了
    —— 有这个基线, 「被拒的候选一次都没出现」才有内容;
  * **内容级否证**: 那份带 `import torch` 的候选 (由脚本给定的修复产物) 在**全部**后端调用
    载荷里搜不到一次。比次数强: 次数会被生成侧自己的编译剪枝搅进来, 而「这份源码到没到过
    后端」不会被搅 (与 `tests/unit/test_agent_runtime_wiring.py` 同一口径);
  * **产物事实**: 被拒那一轮末条 L1 是**合成的拒绝** (`compile_status=agent_action_refused`),
    该轮 `measurement_wall_s is None` (没有真值测量), 与「别的轮有真读数」形成对照。

⚠️ **这个驱动第一版抓到了一个真洞** (CHECKLOG I78): 修复产物一直夹带 `import torch` 时,
第 2 轮被拒, 但第 3 轮同一份候选**过了门并真的上了远端编译** (`L1 PASSED: 编译成功`,
最后靠 L2 的反作弊静态审计拦住) —— 因为「新增依赖」当时是对着 patch 的父代判的, 而父代
正是上一轮那份**被拒的**候选。修法: 基线 = 上一次**过门**的候选
(`ActionContext.dep_baseline_source` + `Conductor._admitted_baseline`)。这一跑同时是那条
修法的真机守卫: 每一轮重复提交的违规候选都必须零载荷。

顺带收两条 (白捡的真机证据, 都不另跑):
  * §21.3 的生产者在**真机链路**上生效: 拒绝理由逐字进了 Fixer 的处方, 且修复提示里
    有那八段结构化证据 (proxy 记下送进 LLM 的 prompt);
  * §21.2 的「被拒 → §8 路由 → Fixer」在真机上确实接上了 (Fixer 真的被调用过第二次)。

**口径与边界** (必须与结论一起读):
  * 修复产物与生成产物由**脚本给定** (要一个 §21 违规的候选, 不能让 27B 随机决定);
    Designer/其余环节走真 LLM 端点。所以这一跑证的是**判据与链路**, 不是模型质量。
  * 「零 GPU 秒」的口径 = **零远端调用载荷**: 控制端一次都没有把那份源码交给后端。
    这一跑**不**读 nvidia-smi 的累计秒数 (`algorithm-h20` 是多进程共享卡, 卡级读数分不出
    是谁的) —— 想要卡级读数就得独占卡, 那是另一回事, 这里不冒充。

用法: python3 benchmarks/drivers/agent_action_h20.py [--rounds 3]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from operator_agent.app._harness_controls import enable_for_harness
enable_for_harness()
os.environ.setdefault("OPERATOR_AGENT_KERNEL_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_FAILURE_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_LOG_KEEP", "100")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from operator_agent import Conductor, LLMClient, RemoteExecutor  # noqa: E402
from operator_agent.core.run_layout import new_run_root  # noqa: E402
from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec,  # noqa: E402
                                    TaskSchema, TaskComplexity, TaskType,
                                    TargetSpec, VerificationLevel)

#: 第一份候选: 语法错 ⇒ 远端编译真的失败 (依赖探针 / 路由 / Fixer 都被这一份点燃)
BROKEN_COMPILE = "import numpy as np\n\ndef kernel(A, B):\n    return A @\n"
#: 修复产物: 比父代多一个外部依赖 ⇒ §21.4「新增依赖」判死 (零 GPU 秒)。真机跑的是
#: triton DSL, 但违规判定发生在**送编译之前**, 判的是源码里多出的 import。
FIXER_ADDS_DEP = ("import numpy as np\nimport torch\n\n"
                  "def kernel(A, B):\n    return torch.matmul(A, B)\n")
SKETCH = {"algorithm": "matmul",
          "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
          "memory_plan": {}, "pipeline": [], "optimization_hints": []}


class RecordingBackend:
    """只记账的外壳: 每一次**可能花 GPU 秒**的调用都记下 `(方法名, 源码)`

    不做任何判断、不改任何返回 —— 判断在被包的 backend 里 (真机执行器), 这里只是
    把「控制端交出去过哪些源码」留成可搜索的记录 (`__getattr__` 兜住其余接口,
    包括 `capabilities` 这类属性 —— 门面后端的能力是**派生**的, 不能在这里拦)。
    """

    RECORDED = ("compile", "execute", "benchmark", "benchmark_with_baselines",
                "analyze_dependencies")

    def __init__(self, inner):
        self.inner = inner
        self.calls = []          # [(method, kernel_code|None)]

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _record(self, method, code):
        self.calls.append((method, code))

    def compile(self, kernel_code):
        self._record("compile", kernel_code)
        return self.inner.compile(kernel_code)

    def execute(self, kernel_code, inputs):
        self._record("execute", kernel_code)
        return self.inner.execute(kernel_code, inputs)

    def benchmark(self, kernel_code, inputs, **kw):
        self._record("benchmark", kernel_code)
        return self.inner.benchmark(kernel_code, inputs, **kw)

    def benchmark_with_baselines(self, kernel_code, inputs, **kw):
        self._record("benchmark_with_baselines", kernel_code)
        return self.inner.benchmark_with_baselines(kernel_code, inputs, **kw)

    def analyze_dependencies(self, kernel_code):
        self._record("analyze_dependencies", kernel_code)
        return self.inner.analyze_dependencies(kernel_code)

    # ---- 记录视图 (判据只读这三条) ----
    def payloads(self, needle=None):
        return [c for _, c in self.calls
                if c is not None and (needle is None or needle in c)]

    def methods(self):
        return sorted({m for m, _ in self.calls})


class ScriptedLlm:
    """真 LLM 端点的**选择性替身**: 只有生成/修复两处由脚本给定, 其余全部转发

    为什么必须替这两处: 要一个确定会踩 §21.4 的候选, 就不能让 27B 随机决定。
    转发的部分 (Designer 等) 走真端点 ⇒ 真机跑里仍有真 LLM 花费与真时延。
    同时把送进 LLM 的 prompt 逐字记下来 (§21.3 的注入证据只能在 prompt 里看)。
    """

    def __init__(self, inner):
        self.inner = inner
        self.sent = []           # [(caller_tag, prompt)]

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _note(self, kw, messages):
        self.sent.append((kw.get("caller_tag", "?"),
                          "\n".join(str(m.get("content", "")) for m in messages)))

    def chat(self, messages, **kw):
        tag = str(kw.get("caller_tag") or "")
        self._note(kw, messages)
        if tag == "Coder":
            return BROKEN_COMPILE
        if tag.startswith("Fixer") and "Reflexion" not in tag:
            return FIXER_ADDS_DEP
        return self.inner.chat(messages, **kw)

    def chat_json(self, messages, **kw):
        tag = str(kw.get("caller_tag") or "")
        self._note(kw, messages)
        if tag == "Coder":
            return dict(SKETCH)
        return self.inner.chat_json(messages, **kw)

    def prompts(self, tag):
        return [p for t, p in self.sent if t == tag]


def _llm_from_config():
    """端点只活在本地 config.yaml (未跟踪), 仓内只有 config.example.yaml。"""
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        sys.exit(f"缺 {cfg_path}: 请从 config.example.yaml 复制并填 llm.base_url")
    llm_cfg = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("llm") or {}
    if not llm_cfg.get("base_url"):
        sys.exit("config.yaml 的 llm.base_url 为空")
    return LLMClient(base_url=llm_cfg["base_url"],
                     model=llm_cfg.get("model", "Qwen3.8-27B"),
                     temperature=llm_cfg.get("temperature", 0.3),
                     max_tokens=llm_cfg.get("max_tokens", 4096),
                     timeout=llm_cfg.get("timeout", 120))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=3)
    # §22.1 的布局根落点（一次运行一个新根，`new_run_root` 保证唯一）。
    # 默认落在被 gitignore 的 `output/` 下：这一跑的目的是**留下可复核的候选文件**
    # （§21 的 `Patch.path` 与盘上产物指向同一个文件），所以不往 /tmp 里塞。
    ap.add_argument("--layout-dir", default=str(ROOT / "output" / "agent_action_h20"),
                    help="§22.1 运行布局的基目录（默认 output/agent_action_h20）")
    args = ap.parse_args()

    real_llm = _llm_from_config()
    t0 = time.monotonic()
    pong = real_llm.chat([{"role": "user", "content": "回复两个字: 在线"}],
                         max_tokens=8, caller_tag="Preflight")
    preflight_s = round(time.monotonic() - t0, 1)
    print(f"[预检] {real_llm.model} @ {real_llm.base_url} → {pong.strip()!r} "
          f"({preflight_s}s)", flush=True)

    llm = ScriptedLlm(real_llm)
    backend = RecordingBackend(RemoteExecutor(mode="ssh"))
    print(f"[后端] {backend.name} · 能力 {sorted(c.value for c in backend.capabilities)}",
          flush=True)

    task = TaskSchema(task_id="agent-action-refusal",
                      task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(kernel_type="matmul",
                                        complexity=TaskComplexity.MEDIUM),
                      target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
                      constraints=ConstraintSpec(precision=Precision.FP32,
                                                 min_speedup=0.5, max_error=1e-2))
    rng = np.random.default_rng(11)     # 与 replay/e2e/exhaustion 同一种子
    inputs = {"A": rng.standard_normal((1024, 1024)).astype(np.float32),
              "B": rng.standard_normal((1024, 1024)).astype(np.float32)}

    # 单路生成: 流水线的编译剪枝会把这份语法坏码换掉 (见 wiring 用例的注释)
    conductor = Conductor(backend=backend, llm=llm, max_optimize_rounds=args.rounds,
                          use_coder_pipeline=False,
                          run_root=new_run_root(args.layout_dir, "agent-action"))
    t1 = time.monotonic()
    art = conductor.execute(task, inputs, target_gflops=0.0)
    wall = round(time.monotonic() - t1, 1)

    meta = art.metadata or {}
    rounds = [{"round": h.get("round"), "action": h.get("action"),
               "levels": h.get("levels_passed"), "gflops": h.get("gflops"),
               "wall_s": h.get("measurement_wall_s"),
               "agent_action": (h.get("agent_action") or {}).get("accepted"),
               "fix_evidence": h.get("fix_evidence")}
              for h in art.iteration_history]
    l1 = [{"round": None, "passed": v.passed,
           "status": (v.metrics or {}).get("compile_status"),
           "refused": bool((v.metrics or {}).get("agent_action_refused")),
           "message": (v.message or "")[:120]}
          for v in art.verification_results
          if v.level == VerificationLevel.L1_COMPILE]
    refused_rounds = [r for r in rounds if r["agent_action"] is False]
    out = {"wall_s": wall, "preflight_s": preflight_s, "backend": getattr(backend, "name", "?"),
           "stop": (meta.get("stop") or {}).get("reason"),
           "rounds": rounds, "l1_results": l1,
           "backend_calls": {"n": len(backend.calls), "methods": backend.methods()},
           "deps_probe_payloads": len(backend.payloads("return A @")),
           "torch_payloads": len(backend.payloads("import torch")),
           "distinct_payloads": len({c for _, c in backend.calls if c})}
    print("RESULT " + json.dumps(out, ensure_ascii=False, default=str), flush=True)

    refused = refused_rounds[0] if refused_rounds else None
    violation = []
    if refused:
        for h in art.iteration_history:
            aa = h.get("agent_action") or {}
            if aa.get("accepted") is False:
                violation = [o for o in (aa.get("precompile") or {}).get("outcomes", [])
                             if o.get("state") == "violation"]
                break
    fix_prompts = llm.prompts("Fixer")
    real_rounds = [r for r in rounds if r["wall_s"] is not None]
    synthetic = [r for r in l1 if r["refused"]]

    checks = {
        "C1 端点预检通过 (真 LLM 在环)": bool(pong.strip()),
        # ⚠️ 口径: 这一跑**刻意**没有 L3 读数 —— 违规候选被拒、第一份候选死在本地语法
        # 预检, 所以「真机往返」的活性证人只有依赖探针与按需补装 (那条走 SSH, 真机日志里
        # 有 `[SSHExecutor] 依赖安装成功`)。不许把它写成「远端编译过了」。
        "C2 后端是真的远端执行器 (SSHExecutor@algorithm-h20)": (
            "SSHExecutor" in str(getattr(backend, "name", ""))
            and "algorithm-h20" in str(getattr(backend, "name", ""))),
        # ⚠️ 口径按真机事实写: 第一份坏码的 L1 是**本地**语法预检失败 (不花远端时间),
        # 它真正花掉的远端资源是依赖探针与按需补装 (`[SSHExecutor] 依赖安装成功`)。
        # 不许写成「远端编译失败」—— 那是另一种失败, 真机会走另一条路。
        "C3 第一份坏码真的走过远端 (依赖探针/补装), 且不是被 §21 拒的": (
            len(backend.payloads("return A @")) >= 1
            and not any(r["round"] == 1 and r["agent_action"] is False for r in rounds)),
        "C4 依赖探针是活的 (为坏码走过一次 SSH)": len(
            backend.payloads("return A @")) >= 1
            and "analyze_dependencies" in backend.methods(),
        "C5 每一轮的违规裁决都是 accepted=False 且违约是 new_dependency": bool(
            len(refused_rounds) >= 2
            and [o for o in violation if o.get("rule") == "new_dependency"]),
        "C6 带 torch 的候选**零载荷** (被拒 = 零 GPU 秒, 每一轮重复提交都是)": (
            out["torch_payloads"] == 0),
        "C7 被拒轮以合成拒绝收场且没有真值测量": bool(
            synthetic and refused and refused["wall_s"] is None),
        "C8 §21.3 那八段 + 拒绝理由逐字进了修复处方": bool(
            fix_prompts and any("结构化证据" in p for p in fix_prompts)
            and any("新增" in p or "依赖" in p for p in fix_prompts)),
    }
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}", flush=True)
    print(f"  (读数) 轮数={len(rounds)} 有真值测量的轮={len(real_rounds)} "
          f"被拒轮={[r['round'] for r in refused_rounds]} "
          f"后端调用={len(backend.calls)} 次 "
          f"(deps 探针 {out['deps_probe_payloads']} 次, torch 载荷 "
          f"{out['torch_payloads']} 次); stop={out['stop']}", flush=True)
    print(f"  (读数) 各轮墙钟: "
          f"{[(r['round'], r['wall_s']) for r in rounds]} (None = 该轮没有真值测量)",
          flush=True)
    # 逐条列出真的交出去的源码开头 —— 「零载荷」这句话的证据就是这个清单
    print("  (读数) 后端真的收到的载荷:", flush=True)
    for m, c in backend.calls:
        print(f"    - {m}: {c.splitlines()[0][:44]!r}… ({len(c)} chars)", flush=True)
    print(f"  (读数) 被拒的轮: {[(r['round'], r['action']) for r in refused_rounds]}",
          flush=True)
    ok = all(checks.values())
    print(f"\n{'PASS' if ok else 'FAIL'} §21 被拒的候选零远端载荷 (真机)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())