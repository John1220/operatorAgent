"""Fused residual+LayerNorm (registered family) tolerance calibration.

Runs three candidate variants through the real SSH L2 path at the currently
registered atol and reports per-case max/matched readings:

- good: one-pass fp32-statistics fused kernel (the intended headroom kernel);
- alt:  fp16 add then fp32 statistics (mirrors the eager reference's add rounding);
- bad:  drops R entirely (must be rejected).

Atol selection is *not* done here: this script only records what the registered
gate admits. Change REGISTERED_ATOL for the fused profile and rerun to measure a
different gate. No profile is claimed until the good/alt/bad separation is
recorded and the value is written into the registry.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

DECLARATION = dict(request="fp16 32768x6144 residual add layernorm",
                   shapes=[[32768, 6144]], precision="fp16", dsl="triton",
                   hardware="nvidia_gpu", min_speedup=1.0)

_HEAD = '''import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(x_ptr, r_ptr, y_ptr, N, eps,
            sxm, sxn, srm, srn, sym, syn, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * sxm + cols * sxn, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * srm + cols * srn, mask=mask, other=0.0).to(tl.float32)
'''

_TAIL = '''    s = {sum_expr}
    mean = tl.sum(s, axis=0) / N
    centred = tl.where(mask, s - mean, 0.0)
    var = tl.sum(centred * centred, axis=0) / {var_div}
    y = centred * tl.rsqrt(var + eps)
    tl.store(y_ptr + row * sym + cols * syn, y.to(y_ptr.dtype.element_ty), mask=mask)


def kernel(X, R):
    M, N = X.shape
    Y = torch.empty_like(X)
    _kernel[(M,)](X, R, Y, N, 1e-5,
                  X.stride(0), X.stride(1), R.stride(0), R.stride(1),
                  Y.stride(0), Y.stride(1),
                  BLOCK=triton.next_power_of_2(N), num_warps=8)
    return Y
'''

GOOD = _HEAD + _TAIL.format(sum_expr="x + r", var_div="N")
ALT = _HEAD + _TAIL.format(sum_expr="(x + r).to(tl.float16).to(tl.float32)", var_div="N")
BAD = _HEAD + _TAIL.format(sum_expr="x", var_div="N")

VARIANTS = {"good": GOOD, "alt": ALT, "bad": BAD}


def _l2_summary(receipt_path):
    result = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    for verdict in result.get("verdicts") or []:
        if verdict.get("level") != "L2":
            continue
        metrics = verdict.get("metrics") or {}
        cases = []
        for case in (metrics.get("correctness") or {}).get("cases") or []:
            m = case.get("metrics") or {}
            cases.append(dict(case_id=case.get("case_id"),
                              max_error=m.get("max_error"),
                              max_rel_error=m.get("max_rel_error"),
                              bad_elements=m.get("bad_elements"),
                              compared_elements=m.get("compared_elements"),
                              matched_ratio=m.get("matched_ratio")))
        return dict(passed=verdict.get("passed"), atol=metrics.get("atol"),
                    max_error=metrics.get("max_error"),
                    bad_elements=metrics.get("bad_elements"),
                    matched_ratio=metrics.get("matched_ratio"),
                    compared_elements=metrics.get("compared_elements"),
                    cases=cases)
    return {}


def run_variant(name, code, root, args):
    from operator_agent.app import hitl_tools
    from operator_agent.execute.backend_ownership import owned_worker_scope

    run = root / name
    hitl_tools.prepare(DECLARATION, run, host_budget=dict(
        wall_clock_s=1800, max_verifications=2, max_model_calls=1,
        max_output_tokens_per_call=8192, total_output_token_reservations=8192))
    worker = run / "explore-worker"
    worker.mkdir()
    with owned_worker_scope(worker, uuid.uuid4().hex):
        response = hitl_tools.verify(
            run, code, backend="ssh", host=args.host, work_dir=args.work_dir,
            timeout=args.timeout, persistent=True, stop_after="L2",
            verification_purpose="explore")
    evidence = response.get("evidence")
    receipt = _l2_summary(evidence) if evidence else {}
    record = dict(variant=name, evidence=evidence,
                  passed=response.get("passed"), levels=response.get("levels"),
                  l2=receipt)
    (run / "calibration.json").write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return record


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="h20")
    parser.add_argument("--work-dir", default="/root/operator-p2-seed-20261116")
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--gpu-uuid", default="GPU-52ca6155-d8bc-a416-847a-6f0c71db1552")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--variants", nargs="+", default=["good", "alt", "bad"])
    args = parser.parse_args(argv)
    os.environ.setdefault("OPERATOR_AGENT_REQUIRED_GPU", args.gpu)
    os.environ.setdefault("OPERATOR_AGENT_REQUIRED_GPU_UUID", args.gpu_uuid)
    os.environ.setdefault("OPERATOR_AGENT_REQUIRED_GPU_NAME", "NVIDIA H20")
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    results = [run_variant(name, VARIANTS[name], args.root, args) for name in args.variants]
    summary = [{k: r[k] for k in ("variant", "passed", "levels")}
               | {"l2": {k: (r.get("l2") or {}).get(k)
                         for k in ("atol", "max_error", "bad_elements",
                                   "matched_ratio", "compared_elements")}}
               for r in results]
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
