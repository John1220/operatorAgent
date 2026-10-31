"""Fixed operator tasks and task-level quality accounting, with no execution.

The formal correctness profile remains unchanged. Additional primary and holdout
inputs use the existing CorrectnessCase materializer and are declared in the
manifest. A driver must verify every evaluation input before setting
``full_l2_passed``. Replay and mock readings never establish GPU performance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Optional

from ..decide.bench_cases import BenchmarkCase, TensorSpec
from ..core.contract import freeze_from_task
from ..decide.correctness import CorrectnessCase, as_suite
from ..decide.correctness_profiles import with_default_correctness
from ..core.run_contract import digest, json_value
from ..core.schemas import ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskSchema


SUITE_VERSION = "operator-quality-v1"
REPORT_VERSION = "operator-quality-report-v1"


@dataclass(frozen=True)
class QualityCase:
    """One fixed task, including a challenge and two independent input seeds."""

    case_id: str
    kernel_type: str
    shape: tuple
    precision: str = "fp32"
    pattern: str = "random"
    layout: str = "contiguous"
    seed: int = 11
    holdout_seeds: tuple = (317, 619)

    def content_dict(self) -> dict:
        """Return a detached JSON declaration suitable for hashing."""
        return json_value(asdict(self))


_CASES = (
    QualityCase("matmul-aligned-fp16", "matmul", (64, 64, 64), "fp16"),
    QualityCase("matmul-tail-fp16", "matmul", (31, 47, 19), "fp16", seed=13),
    QualityCase("matmul-strided-fp32", "matmul", (17, 29, 13), layout="strided", seed=19),
    QualityCase("layernorm-aligned", "layernorm", (8, 64), seed=23),
    QualityCase("layernorm-tail", "layernorm", (7, 33), seed=29),
    QualityCase("layernorm-large-mean", "layernorm", (9, 63), pattern="large_varying", seed=31),
    QualityCase("layernorm-constant-strided", "layernorm", (9, 17),
                pattern="constant_nonbinary", layout="strided", seed=37),
    QualityCase("softmax-tail", "softmax", (7, 33), seed=41),
    QualityCase("softmax-extreme", "softmax", (5, 63), pattern="signed_large", seed=43),
    QualityCase("softmax-constant-strided", "softmax", (9, 17),
                pattern="constant", layout="strided", seed=47),
)


def cases() -> tuple:
    """Return the immutable, versioned default task set."""
    return _CASES


def get_case(case_id: str) -> QualityCase:
    """Resolve a declared id; unknown ids do not create new evaluation tasks."""
    for case in _CASES:
        if case.case_id == case_id:
            return case
    raise ValueError("unknown quality case id")


def _case(value) -> QualityCase:
    if isinstance(value, QualityCase):
        if value != get_case(value.case_id):
            raise ValueError("quality case declaration drift")
        return value
    return get_case(value)


def _benchmark(case: QualityCase, seed: int) -> BenchmarkCase:
    if case.kernel_type == "matmul":
        m, n, k = case.shape
        tensors = (TensorSpec("A", (m, k), case.precision),
                   TensorSpec("B", (k, n), case.precision))
    else:
        # Trusted profiles specify no-affine normalization and one X argument;
        # the legacy shapes layout table also has affine weight/bias inputs.
        tensors = (TensorSpec("X", case.shape, case.precision),)
    return BenchmarkCase(name=case.case_id, tensors=tensors, seed=seed)


def input_cases_for(case_or_id) -> tuple:
    """Return primary challenge plus independent random holdout declarations.

    These are evaluation inputs, not replacements for the trusted full suite.
    Constants intentionally remain a challenge; random holdouts prevent such a
    challenge from being the only numerical input a candidate sees.
    """
    case = _case(case_or_id)
    primary = CorrectnessCase(case.case_id + "-primary", _benchmark(case, case.seed),
                              case.pattern, case.layout)
    holdouts = tuple(CorrectnessCase(case.case_id + f"-holdout-{seed}",
                                    _benchmark(case, seed), "random", case.layout)
                     for seed in case.holdout_seeds)
    return (primary,) + holdouts


def inputs_for(case_or_id) -> dict:
    """Materialize only the task's primary input, without model or GPU calls."""
    return input_cases_for(case_or_id)[0].materialize()


def evaluation_inputs_for(case_or_id) -> tuple:
    """Materialize all declared inputs, whose L2 results must all be checked."""
    return tuple(case.materialize() for case in input_cases_for(case_or_id))


def task_for(case_or_id, *, hardware: str = "nvidia_gpu", dsl: str = "triton") -> TaskSchema:
    """Build a formal task while retaining the complete trusted profile.

    Explicit BenchmarkCase inputs avoid legacy affine layouts. The quality
    manifest owns the actual shape/pattern/layout domain; the legacy contract's
    representative ``shapes`` field is not used as that domain.
    """
    case = _case(case_or_id)
    task = TaskSchema(task_id=case.case_id, created_at=0,
                      source=SourceSpec(kernel_type=case.kernel_type),
                      target=TargetSpec(hardware=hardware, dsl=dsl),
                      constraints=ConstraintSpec(precision=Precision(case.precision),
                                                 cases=(_benchmark(case, case.seed),)))
    task = with_default_correctness(task)
    # Freeze now so unsupported reference/profile declarations fail before a
    # driver can dispatch a model call for this task.
    freeze_from_task(task)
    return task


def suite_manifest(case_ids: Optional[Iterable[str]] = None) -> dict:
    """Describe a fixed suite or explicit subset without materializing arrays."""
    selected = _CASES if case_ids is None else tuple(get_case(key) for key in case_ids)
    if not selected or len({case.case_id for case in selected}) != len(selected):
        raise ValueError("quality suite selection must be nonempty and unique")
    rows = []
    for case in selected:
        task = task_for(case)
        contract = freeze_from_task(task)
        rows.append(dict(case.content_dict(), contract_hash=contract.hash,
                         correctness_profile=task.constraints.correctness_profile,
                         correctness_suite_hash=as_suite(task.constraints.correctness_suite).digest(),
                         evaluation_inputs=[row.content_dict() for row in input_cases_for(case)]))
    body = dict(schema=SUITE_VERSION, cases=rows,
                scope="finite fixed tasks; no-affine last-axis LayerNorm eps=1e-5",
                acceptance="all primary/holdout L2 and the complete trusted correctness suite",
                performance_scope="primary input only; independent native measurement required",
                holdouts_are_secret=False)
    return json_value(dict(body, hash=digest(body)))


def _manifest(value) -> dict:
    manifest = suite_manifest() if value is None else json_value(value)
    if not isinstance(manifest, dict) or manifest.get("schema") != SUITE_VERSION:
        raise ValueError("invalid quality suite manifest")
    claimed = manifest.pop("hash", None)
    if claimed != digest(manifest):
        raise ValueError("quality suite manifest hash mismatch")
    rows = manifest.get("cases")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("invalid quality suite cases")
    ids = [row.get("case_id") for row in rows]
    if manifest != {key: val for key, val in suite_manifest(ids).items() if key != "hash"}:
        raise ValueError("quality suite definition drift")
    manifest["hash"] = claimed
    return manifest


def _count(value, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(field + " must be a nonnegative integer")
    return value


def _seconds(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("wall_seconds must be finite and nonnegative")
    return float(value)


def _rate(count: int, denominator: int) -> dict:
    return dict(count=count, denominator=denominator,
                rate=count / denominator if denominator else None)


def summarize(records: Iterable[dict], *, manifest=None, controls=None) -> dict:
    """Retain every candidate and score complete tasks, including missing ones.

    ``full_l2_passed`` is a driver-owned fact requiring actual complete suite and
    holdout results. This reducer does not manufacture it from L1 or exit codes.
    Failure, timeout, generation failure and unknown rows remain in the task
    denominator and all reported costs. No mock/replay record earns native
    performance credit, even if its producer marked a performance pass.
    """
    manifest = _manifest(manifest)
    expected = [row["case_id"] for row in manifest["cases"]]
    grouped = {key: [] for key in expected}
    copied = []
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("quality record must be an object")
        row = json_value(raw)
        key = row.get("case_id")
        if key not in grouped:
            raise ValueError("record has unknown quality case")
        attempt = row.get("attempt")
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if any(old["attempt"] == attempt for old in grouped[key]):
            raise ValueError("duplicate quality attempt")
        if not isinstance(row.get("status"), str) or not row["status"]:
            raise ValueError("quality record needs an explicit status")
        for name in ("full_l2_passed", "performance_target_passed"):
            if row.get(name) is not None and type(row[name]) is not bool:
                raise ValueError(name + " must be bool or null")
        if row.get("execution_kind") not in ("real_device", "mock", "replay"):
            raise ValueError("unknown quality execution kind")
        for name in ("model_calls", "tokens"):
            if row.get(name) is not None:
                _count(row[name], name)
        if row.get("wall_seconds") is not None:
            _seconds(row["wall_seconds"])
        grouped[key].append(row)
        copied.append(row)

    tasks = []
    for key in expected:
        attempts = sorted(grouped[key], key=lambda row: row["attempt"])
        if [row["attempt"] for row in attempts] != list(range(1, len(attempts) + 1)):
            raise ValueError("quality attempt history has gaps")
        first = attempts[0] if attempts else {}
        final = attempts[-1] if attempts else {}
        full = lambda row: row.get("full_l2_passed") is True
        native_performance = any(full(row) and row.get("status") == "passed"
                                 and row.get("performance_target_passed") is True
                                 and row.get("measurement_validity") == "valid"
                                 and row.get("execution_kind") == "real_device" for row in attempts)
        native_measured = any(full(row) and row.get("status") in ("passed", "failed")
                              and type(row.get("performance_target_passed")) is bool
                              and row.get("measurement_validity") == "valid"
                              and row.get("execution_kind") == "real_device" for row in attempts)
        tasks.append(dict(case_id=key, status=final.get("status", "unknown"),
                          attempts=len(attempts), first_full_l2_passed=full(first),
                          final_full_l2_passed=full(final),
                          within_budget_full_l2_passed=any(full(row) for row in attempts),
                          native_performance_target_passed=native_performance,
                          native_performance_measured=native_measured))

    metrics = {}
    for label, field in (("first_full_l2", "first_full_l2_passed"),
                         ("final_full_l2", "final_full_l2_passed"),
                         ("within_budget_full_l2", "within_budget_full_l2_passed"),
                         ("native_performance_target", "native_performance_target_passed")):
        metrics[label] = _rate(sum(row[field] for row in tasks), len(expected))
    measured = sum(row["native_performance_measured"] for row in tasks)
    metrics["native_performance_target"].update(
        measured_tasks=measured, state="measured" if measured else "not_measured")
    if not measured:
        metrics["native_performance_target"]["rate"] = None
    costs = {}
    successes = metrics["within_budget_full_l2"]["count"]
    for name in ("model_calls", "tokens", "wall_seconds"):
        missing = sum(row.get(name) is None for row in copied)
        known_total = sum(row[name] for row in copied if row.get(name) is not None)
        costs[name] = dict(known_total=known_total, missing_records=missing,
                           per_correct_task=known_total / successes if successes and not missing else None)
    controls = json_value({} if controls is None else controls)
    if not isinstance(controls, dict):
        raise ValueError("quality controls must be an object")
    fixed_controls = {key: value for key, value in controls.items() if key not in ("model", "strategy")}
    return dict(schema=REPORT_VERSION, suite_hash=manifest["hash"], manifest=manifest,
                controls=controls, controls_hash=digest(controls),
                comparison_identity=digest(fixed_controls),
                attempts=copied, tasks=tasks, metrics=metrics, costs=costs,
                missing_tasks=sum(row["attempts"] == 0 for row in tasks),
                native_performance_observed=bool(measured),
                evidence_scope="caller-supplied records; replay/mock do not attest GPU quality")


def compare_reports(left: dict, right: dict) -> dict:
    """Compare only identical tasks and fixed controls, allowing model/strategy changes.

    Reports are recomputed from their retained attempts before comparison. This
    rejects modified aggregates and does not infer significance from a small
    finite suite. Hardware, budget and verification/ruler controls must match.
    """
    checked = []
    for report in (left, right):
        if not isinstance(report, dict) or report.get("schema") != REPORT_VERSION:
            raise ValueError("invalid quality report")
        rebuilt = summarize(report.get("attempts", ()), manifest=report.get("manifest"),
                            controls=report.get("controls"))
        if json_value(report) != json_value(rebuilt):
            raise ValueError("quality report does not match retained evidence")
        checked.append(rebuilt)
    a, b = checked
    if a["suite_hash"] != b["suite_hash"] or a["comparison_identity"] != b["comparison_identity"]:
        raise ValueError("quality comparison requires identical suite and fixed controls")
    differences = {key: dict(left=a["controls"].get(key), right=b["controls"].get(key))
                   for key in ("model", "strategy")
                   if a["controls"].get(key) != b["controls"].get(key)}
    return dict(comparable=True, suite_hash=a["suite_hash"], varied_controls=differences,
                metrics={name: dict(left=a["metrics"][name], right=b["metrics"][name],
                                   rate_delta=(b["metrics"][name]["rate"] - a["metrics"][name]["rate"]
                                               if a["metrics"][name]["rate"] is not None
                                               and b["metrics"][name]["rate"] is not None else None))
                         for name in a["metrics"]},
                costs=dict(left=a["costs"], right=b["costs"]),
                scope="descriptive finite-suite comparison; no statistical significance claim")
