"""给一份**已有的**核，按处方自动修到过（或 N 轮）

为什么单独有这一层：`Conductor` 的路是「从零生成 → 判 → 修」，而真实使用里经常是
「我已经有一份核（人写的、别处拷来的、上一轮的遗留），让它过」。`operator_run` 不接受
种子候选，`operator_verify` 只判不修 —— 中间缺一格。

这一层只做三件事：**判 → 读处方 → 调 Fixer → 再判**，最多 N 轮。判据全部来自既有模块
（`Verifier` / `failure_route` / `Fixer`），这里不新增任何判据、不放宽任何阈值。

它同时是「自动修复能力」的量具：每个 bug 跑一遍，记下几轮改对、初次落在哪一级。
"""

from __future__ import annotations

import contextlib
import io
from typing import Optional

from .failure_route import classify
from .fixer import FixerAgent
from .schemas import VerificationLevel
from .verifier import Verifier


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        value = fn(*a, **kw)
    return value, buf.getvalue()


def repair(code: str, task, test_inputs, verifier: Verifier,
           fixer: Optional[FixerAgent] = None, *,
           max_rounds: int = 2, expected_output=None,
           target_gflops: float = 0.0,
           stop_after: Optional[VerificationLevel] = None) -> dict:
    """反复「判 → 修 → 再判」。返回每一轮的判定、处方与修复摘要。

    * `route.counts_as_iteration = False`（`hold`：静默退化 / 参考侧无对象）⇒ **不修**，
      直接停并说明理由 —— 那是「这一轮什么都没判」，拿它去修等于对着噪声开处方。
    * `Fixer` 只回一段说明、不给代码（模型在提问）⇒ 停，并把提问原文带出来。
    """
    if max_rounds < 1:
        raise ValueError("max_rounds 至少为 1")
    fixer = fixer or FixerAgent(llm=None)
    current = code
    rounds: list[dict] = []
    log_chunks: list[str] = []
    results = []
    passed = False

    for index in range(1, int(max_rounds) + 1):
        results, log = _quiet(verifier.verify, current, task, test_inputs,
                              expected_output, float(target_gflops), None, stop_after)
        log_chunks.append(log)
        passed = all(r.passed for r in results)
        route = classify(results)
        entry = {
            "round": index,
            "levels": {r.level.value: bool(r.passed) for r in results},
            "passed": passed,
            "route": route.as_dict(),
            "prescription": route.reason,
        }
        rounds.append(entry)
        if passed:
            entry["action"] = "done"
            break
        if not route.counts_as_iteration:
            entry["action"] = "stop"
            entry["why"] = "处方不可据以行动（hold：这一轮没有可判的对象），不调 Fixer"
            break
        if index == int(max_rounds):
            entry["action"] = "budget_exhausted"
            break
        failed = [r for r in results if not r.passed]
        try:
            outcome = _quiet(fixer.run_with_action,
                             current, task, failed,
                             fix_not_rewrite=bool(route.fix_not_rewrite),
                             error_shape=route.error_shape or "")[0]
        except Exception as exc:                      # noqa: BLE001 — 修复失败要可归因
            entry["action"] = "fixer_failed"
            entry["why"] = f"{type(exc).__name__}: {exc}"
            break
        if not outcome.code:
            entry["action"] = "ask"
            entry["why"] = outcome.summary or "模型只提问、未给代码"
            break
        entry["action"] = "fixed"
        entry["fix_summary"] = outcome.summary
        current = outcome.code

    return {
        "passed": bool(passed),
        "rounds": rounds,
        "code": current,
        "results": results,
        "log": "\n".join(log_chunks),
    }
