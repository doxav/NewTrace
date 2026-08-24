"""Frozen main-run orchestration and paired analysis for Experiment 0."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .forecast import _priced_cost
from .live import (
    _execute_arm,
    _infrastructure_checks_pass,
    _pilot_optimizer_gates,
    _pilot_retry_statistics,
)
from .preflight import _canonical_json, _load_json
from .provenance import experiment_source_provenance
from .registration import assert_strict_output_evaluator, register_experiment_components
from .specs import INITIAL_ARTIFACT, MODEL, RESOLVED_MODEL


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
FROZEN_PREREGISTRATION_PATH = PACKAGE_ROOT / "manifests/preregistration_frozen.json"
MAIN_AUTHORIZATION_PATH = PACKAGE_ROOT / "reports/main_cost_authorization.json"
MAIN_LOCK_PATH = PACKAGE_ROOT / "control_plane_lock_for_main.json"
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/recursive_opt/experiment_0/experiment-0-v2"
MAIN_RUNS_DIRECTORY = OUTPUT_ROOT / "main/runs"
MAIN_REPORT_PATH = OUTPUT_ROOT / "main/main.json"
MAIN_ANALYSIS_PATH = OUTPUT_ROOT / "analysis.json"
MAIN_DECISION_PATH = OUTPUT_ROOT / "decision.json"
MAIN_REPORT_MARKDOWN_PATH = OUTPUT_ROOT / "report.md"
EPISODE_AUDIT_PATH = OUTPUT_ROOT / "episode_trajectory_audit.json"
_METRICS = ("accuracy", "invalid_rate", "forward_token_ratio", "latency_s")
_COMPARISONS = {
    "B-A": ("B", "A"),
    "C-A": ("C", "A"),
    "B-C": ("B", "C"),
    "D-B": ("D", "B"),
}


def _sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Atomically persist one generated JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_hash_reference(parent: Path, reference: Mapping[str, Any]) -> None:
    """Require a relative frozen input to retain its declared digest."""
    path = parent / str(reference["path"])
    expected = str(reference["sha256"])
    if _sha256(path) != expected:
        raise RuntimeError(f"frozen input digest changed: {path.name}")


def validate_frozen_preregistration() -> dict[str, Any]:
    """Validate every frozen main input before any optimizer call."""
    frozen = _load_json(FROZEN_PREREGISTRATION_PATH)
    if frozen.get("schema_version") != "recursive-opt-main-preregistration/v1":
        raise ValueError("unsupported frozen main preregistration")
    _validate_hash_reference(
        FROZEN_PREREGISTRATION_PATH.parent,
        frozen["base_preregistration"],
    )
    _validate_hash_reference(
        FROZEN_PREREGISTRATION_PATH.parent,
        frozen["dataset_manifest"],
    )
    dataset = _load_json(PACKAGE_ROOT / "manifests/dataset_manifest_v2.json")
    task = str(frozen["task"])
    rows = dataset["tasks"][task]["samples"]
    observed = {
        split: [str(row["id"]) for row in rows if row["split"] == split]
        for split in ("train", "validation", "holdout")
    }
    if observed != frozen["dataset_manifest"]["splits"]:
        raise RuntimeError("frozen main split IDs or ordering changed")
    if frozen["initial_artifact"] != INITIAL_ARTIFACT:
        raise RuntimeError("frozen P0 differs from the executable initial artifact")
    profiles = frozen["model_profiles"]
    expected_profiles = {
        "provider": "openrouter",
        "model": MODEL,
        "resolved_model": RESOLVED_MODEL,
        "forward": {
            "temperature": 0,
            "max_tokens": 384,
            "request_params": {"reasoning": {"enabled": False}},
        },
        "optimizer": {
            "temperature": 0,
            "max_tokens": 8192,
            "request_params": {"reasoning": {"effort": "low"}},
        },
    }
    if profiles != expected_profiles:
        raise RuntimeError("frozen main model differs from the executable profiles")
    if frozen["objective"] != {
        "mode": "weighted",
        "weights": {"accuracy": 1.0, "forward_token_ratio": 0.1},
        "directions": {
            "accuracy": "maximize",
            "forward_token_ratio": "minimize",
            "invalid_rate": "minimize",
        },
        "hard_constraints": ["invalid_rate <= 0"],
    }:
        raise RuntimeError("frozen main objective differs from Experiment 0")
    matrix = frozen["main"]
    seeds = [int(value) for value in matrix["paired_seeds"]]
    budgets = [int(value) for value in matrix["candidate_budgets"]]
    if seeds != [0, 1, 2, 3, 4] or budgets != [6, 12]:
        raise RuntimeError("frozen main seed/budget matrix changed")
    for seed in seeds:
        order = matrix["arm_order"].get(str(seed))
        if not isinstance(order, list) or sorted(order) != ["A", "B", "C", "D"]:
            raise RuntimeError(f"frozen main arm order is invalid for seed {seed}")
    authorization = _load_json(MAIN_AUTHORIZATION_PATH)
    if not authorization.get("authorized") or not authorization.get(
        "numeric_ceiling_waived"
    ):
        raise RuntimeError("main run lacks explicit monetary authorization")
    if authorization.get("scientific_protocol_changed") is not False:
        raise RuntimeError("monetary authorization must not change experiment science")
    return frozen


def main_execution_matrix(frozen: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    """Expand the frozen seed, budget, and counterbalanced arm order."""
    main = frozen["main"]
    return [
        (int(seed), int(proposals), str(arm))
        for seed in main["paired_seeds"]
        for proposals in main["candidate_budgets"]
        for arm in main["arm_order"][str(seed)]
    ]


def _run_key(seed: int, proposals: int, arm: str) -> str:
    """Return the stable relative identifier for one main execution unit."""
    return f"seed-{seed}/budget-{proposals}/{arm}"


def _load_main_lock(frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Load and validate the CI-backed lock used by every main arm."""
    lock = _load_json(MAIN_LOCK_PATH)
    if not lock.get("ready_for_main_experiment"):
        raise RuntimeError("main experiment lock is not ready")
    workflow = lock.get("workflow")
    if not isinstance(workflow, Mapping) or (
        workflow.get("status"), workflow.get("conclusion")
    ) != ("completed", "success"):
        raise RuntimeError("main experiment lock lacks a green required CI job")
    if lock.get("main_preregistration_sha256") != _sha256(
        FROZEN_PREREGISTRATION_PATH
    ):
        raise RuntimeError("main lock does not match the frozen preregistration")
    if lock.get("main_authorization_sha256") != _sha256(MAIN_AUTHORIZATION_PATH):
        raise RuntimeError("main lock does not match the user authorization")
    if lock["experiment"]["source"]["sha256"] != experiment_source_provenance()[
        "sha256"
    ]:
        raise RuntimeError("Experiment-0 source changed after the main lock")
    if lock["experiment"].get("selected_task") != frozen["task"]:
        raise RuntimeError("main lock selected task differs from preregistration")
    return lock


def _progress_document(
    *,
    frozen: Mapping[str, Any],
    lock: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    stopped_after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the resumable main report and all infrastructure gates."""
    matrix = main_execution_matrix(frozen)
    optimizer_gates = _pilot_optimizer_gates(list(runs)) if runs else {
        "trace_real_proposal": False,
        "gepa_real_proposal": False,
        "optimized_artifact_differs": False,
    }
    gates = {
        "all_planned_runs_completed": len(runs) == len(matrix) and stopped_after is None,
        "all_runs_passed": bool(runs) and all(run["passed"] for run in runs),
        "no_holdout_leakage": bool(runs)
        and all(
            run["checks"]["holdout_inaccessible_during_optimization"] for run in runs
        ),
        "exact_runtime_digest": bool(runs)
        and {run["runtime_tree_sha256"] for run in runs}
        == {lock["control_plane"]["runtime_tree_sha256"]},
        "exact_experiment_source_digest": bool(runs)
        and {run["experiment_source_sha256"] for run in runs}
        == {lock["experiment"]["source"]["sha256"]},
        "one_forward_per_evaluator": bool(runs)
        and all(run["checks"]["one_workflow_forward_per_evaluator"] for run in runs),
        "no_selected_invalid_artifact": bool(runs) and all(run["valid"] for run in runs),
        **optimizer_gates,
        "forward_calls_reconcile": bool(runs)
        and all(run["checks"]["forward_calls_reconciled"] for run in runs),
        "forward_tokens_reconcile": bool(runs)
        and all(run["checks"]["forward_tokens_reconciled"] for run in runs),
        "outputs_and_resume": bool(runs)
        and all(run["checks"]["output_persistence_and_resume"] for run in runs),
        "no_uncontrolled_cache_sharing": bool(runs)
        and all(run["checks"]["cache_not_shared"] for run in runs),
        "no_hidden_environment": bool(runs)
        and all(run["checks"]["no_environment_override"] for run in runs),
        "cost_authorized_by_user_waiver": True,
    }
    return {
        "schema_version": "recursive-opt-main-experiment/v1",
        "task": frozen["task"],
        "frozen_preregistration_sha256": _sha256(FROZEN_PREREGISTRATION_PATH),
        "control_plane_lock_sha256": _sha256(MAIN_LOCK_PATH),
        "planned_run_count": len(matrix),
        "completed_run_count": len(runs),
        "stopped_after": stopped_after,
        "runs": list(runs),
        "retry_statistics_by_arm": _pilot_retry_statistics(list(runs)),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _snapshot_output_context() -> None:
    """Copy frozen inputs into the immutable Experiment-0 output namespace."""
    sources = {
        "preregistration.json": PACKAGE_ROOT / "manifests/preregistration_v2.json",
        "preregistration_frozen.json": FROZEN_PREREGISTRATION_PATH,
        "control_plane_lock.json": MAIN_LOCK_PATH,
        "preflight_skips.json": PACKAGE_ROOT / "preflight_skips.json",
        "dataset_manifest.json": PACKAGE_ROOT / "manifests/dataset_manifest_v2.json",
        "baseline_token_manifest.json": PACKAGE_ROOT / "baseline_token_manifest.json",
        "cost_forecast.json": PACKAGE_ROOT
        / "reports/cost_forecast_after_empty_text_retry.json",
        "main_cost_authorization.json": MAIN_AUTHORIZATION_PATH,
    }
    for name, source in sources.items():
        _write_json(OUTPUT_ROOT / name, _load_json(source))


def run_main_experiment() -> dict[str, Any]:
    """Run the frozen forty-unit main matrix, checkpointing after every arm."""
    register_experiment_components()
    assert_strict_output_evaluator()
    frozen = validate_frozen_preregistration()
    lock = _load_main_lock(frozen)
    pilot = _load_json(PACKAGE_ROOT / "reports/pilot_after_empty_text_retry.json")
    if not pilot.get("passed"):
        raise RuntimeError("main experiment requires the complete passing pilot")
    _snapshot_output_context()
    matrix = main_execution_matrix(frozen)
    existing: list[dict[str, Any]] = []
    if MAIN_REPORT_PATH.exists():
        progress = _load_json(MAIN_REPORT_PATH)
        if progress.get("frozen_preregistration_sha256") != _sha256(
            FROZEN_PREREGISTRATION_PATH
        ):
            raise RuntimeError("existing main progress uses a different preregistration")
        existing = list(progress.get("runs", []))
        if progress.get("passed") and len(existing) == len(matrix):
            return progress
    by_key = {
        _run_key(int(run["seed"]), int(run["proposal_budget"]), str(run["arm"])): run
        for run in existing
    }
    runs: list[dict[str, Any]] = []
    stopped_after: dict[str, Any] | None = None
    task = str(frozen["task"])
    baseline = _load_json(PACKAGE_ROOT / "baseline_token_manifest.json")
    baseline_tokens = {
        str(row["sample_id"]): int(row["baseline_forward_tokens"])
        for row in baseline["samples"]
    }
    main = frozen["main"]
    for seed, proposals, arm in matrix:
        key = _run_key(seed, proposals, arm)
        if key in by_key:
            run = by_key[key]
        else:
            run = _execute_arm(
                arm=arm,
                task=task,
                baseline_tokens=baseline_tokens,
                seed=seed,
                proposals=proposals,
                split_limits=main["split_limits"],
                budget_limits=main["resource_budgets"],
                output_directory=MAIN_RUNS_DIRECTORY / key,
                control_plane_lock=MAIN_LOCK_PATH,
            )
        runs.append(run)
        if not _infrastructure_checks_pass(run["checks"]):
            stopped_after = {
                "seed": seed,
                "proposal_budget": proposals,
                "arm": arm,
                "error": run.get("error"),
            }
        progress = _progress_document(
            frozen=frozen,
            lock=lock,
            runs=runs,
            stopped_after=stopped_after,
        )
        _write_json(MAIN_REPORT_PATH, progress)
        if stopped_after is not None:
            return progress
    return _load_json(MAIN_REPORT_PATH)


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return a deterministic nearest-rank percentile."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return float(ordered[index])


def _bootstrap_mean_ci(values: Sequence[float], *, label: str) -> list[float]:
    """Return the frozen paired block-bootstrap confidence interval."""
    if not values:
        raise ValueError("paired bootstrap requires at least one block")
    seed_material = f"1803:{label}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    generator = random.Random(seed)
    size = len(values)
    means = [
        statistics.fmean(values[generator.randrange(size)] for _ in range(size))
        for _ in range(10000)
    ]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _summary(values: Sequence[float], *, label: str) -> dict[str, Any]:
    """Summarize one run-level metric with paired-bootstrap uncertainty."""
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "paired_bootstrap_95_ci": _bootstrap_mean_ci(values, label=label),
    }


def _run_cost_tokens(run: Mapping[str, Any]) -> dict[str, float]:
    """Combine forward and optimizer token usage for monetary pricing."""
    totals = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}
    for role in ("forward", "optimizer"):
        usage = run["usage"].get(role, {})
        for name in totals:
            totals[name] += float(usage.get(name, 0))
    return totals


def _absolute_arm_summary(
    arm: str,
    runs: Sequence[Mapping[str, Any]],
    pricing: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate absolute outcome and resource metrics for one arm."""
    selected = [run for run in runs if run["arm"] == arm]
    metrics = {
        name: _summary(
            [float(run["metrics"][name]) for run in selected],
            label=f"absolute:{arm}:{name}",
        )
        for name in _METRICS
    }
    tokens = {
        name: sum(_run_cost_tokens(run)[name] for run in selected)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    reported_costs = [
        run["cost_usd"].get(role)
        for run in selected
        for role in ("forward", "optimizer")
    ]
    provider_cost = (
        sum(float(value) for value in reported_costs)
        if all(isinstance(value, (int, float)) for value in reported_costs)
        else None
    )
    return {
        "runs": len(selected),
        "metrics": metrics,
        "forward_calls": sum(int(run["usage"]["forward"].get("calls", 0)) for run in selected),
        "forward_tokens": sum(
            int(run["usage"]["forward"].get("total_tokens", 0)) for run in selected
        ),
        "forward_tokens_per_evaluated_example": sum(
            int(run["usage"]["forward"].get("total_tokens", 0)) for run in selected
        )
        / max(1, sum(int(run["workflow_forwards"]) for run in selected)),
        "optimizer_calls": sum(
            int(run["usage"]["optimizer"].get("calls", 0)) for run in selected
        ),
        "optimizer_tokens": sum(
            int(run["usage"]["optimizer"].get("total_tokens", 0)) for run in selected
        ),
        "evaluator_runs": sum(int(run["accounted"]["evaluator_runs"]) for run in selected),
        "candidates": {
            name: sum(int(run["accounted"].get(name, 0)) for run in selected)
            for name in (
                "candidates_reserved",
                "candidates_proposed",
                "candidates_evaluated",
            )
        },
        "wall_time_s": sum(float(run["accounted"]["wall_time_s"]) for run in selected),
        "selected_artifact_changed_runs": sum(
            bool(run["checks"]["selection_changed"]) for run in selected
        ),
        "unique_selected_artifact_hashes": len(
            {
                hashlib.sha256(_canonical_json(run["artifact"]).encode("utf-8")).hexdigest()
                for run in selected
            }
        ),
        "provider_reported_cost_usd": provider_cost,
        "token_priced_cost_usd": _priced_cost(tokens, pricing),
        "tokens": tokens,
    }


def _paired_comparison(
    label: str,
    left: str,
    right: str,
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two arms using paired seed-budget blocks."""
    indexed = {
        (str(run["arm"]), int(run["seed"]), int(run["proposal_budget"])): run
        for run in runs
    }
    pairs = sorted(
        {
            (seed, budget)
            for arm, seed, budget in indexed
            if arm == left and (right, seed, budget) in indexed
        }
    )
    deltas: dict[str, list[float]] = {}
    for name in (*_METRICS, "weighted_utility"):
        values: list[float] = []
        for seed, budget in pairs:
            left_run = indexed[(left, seed, budget)]
            right_run = indexed[(right, seed, budget)]
            if name == "weighted_utility":
                left_value = float(left_run["metrics"]["accuracy"]) - 0.1 * float(
                    left_run["metrics"]["forward_token_ratio"]
                )
                right_value = float(right_run["metrics"]["accuracy"]) - 0.1 * float(
                    right_run["metrics"]["forward_token_ratio"]
                )
            else:
                left_value = float(left_run["metrics"][name])
                right_value = float(right_run["metrics"][name])
            values.append(left_value - right_value)
        deltas[name] = values
    accuracy = deltas["accuracy"]
    ratio = deltas["forward_token_ratio"]
    summaries = {
        name: _summary(values, label=f"comparison:{label}:{name}")
        for name, values in deltas.items()
    }
    by_budget = {
        str(budget): {
            name: statistics.fmean(
                values[index]
                for index, (_, pair_budget) in enumerate(pairs)
                if pair_budget == budget
            )
            for name, values in deltas.items()
        }
        for budget in sorted({budget for _, budget in pairs})
    }
    by_seed = {
        str(seed): {
            name: statistics.fmean(
                values[index]
                for index, (pair_seed, _) in enumerate(pairs)
                if pair_seed == seed
            )
            for name, values in deltas.items()
        }
        for seed in sorted({seed for seed, _ in pairs})
    }
    accuracy_ci = summaries["accuracy"]["paired_bootstrap_95_ci"]
    ratio_ci = summaries["forward_token_ratio"]["paired_bootstrap_95_ci"]
    return {
        "left_arm": left,
        "right_arm": right,
        "paired_blocks": len(pairs),
        "deltas": summaries,
        "accuracy_win_tie_loss": {
            "wins": sum(value > 0 for value in accuracy),
            "ties": sum(value == 0 for value in accuracy),
            "losses": sum(value < 0 for value in accuracy),
        },
        "token_ratio_win_tie_loss": {
            "wins": sum(value < 0 for value in ratio),
            "ties": sum(value == 0 for value in ratio),
            "losses": sum(value > 0 for value in ratio),
        },
        "by_budget": by_budget,
        "by_seed": by_seed,
        "quality_success": accuracy_ci[0] > 0,
        "efficiency_success": accuracy_ci[0] >= -0.02 and ratio_ci[1] < 0,
    }


def analyze_main_experiment(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compute frozen paired block-bootstrap statistics for the main run."""
    if not report.get("passed"):
        raise RuntimeError("main analysis requires all frozen runs to pass")
    runs = list(report["runs"])
    pricing = _load_json(PACKAGE_ROOT / "manifests/provider_pricing.json")
    absolute = {
        arm: _absolute_arm_summary(arm, runs, pricing) for arm in ("A", "B", "C", "D")
    }
    comparisons = {
        label: _paired_comparison(label, left, right, runs)
        for label, (left, right) in _COMPARISONS.items()
    }
    safety = all(float(run["metrics"]["invalid_rate"]) == 0.0 for run in runs)
    total_cost = sum(value["token_priced_cost_usd"] for value in absolute.values())
    return {
        "schema_version": "recursive-opt-main-analysis/v1",
        "bootstrap": {
            "unit": "paired seed-budget block",
            "seed": 1803,
            "resamples": 10000,
            "confidence_level": 0.95,
        },
        "task": report["task"],
        "run_count": len(runs),
        "absolute_by_arm": absolute,
        "paired_comparisons": comparisons,
        "safety_passed": safety,
        "provider_reported_cost_usd": None,
        "token_priced_cost_usd": total_cost,
        "retry_statistics_by_arm": report["retry_statistics_by_arm"],
        "candidate_diversity": {
            arm: absolute[arm]["unique_selected_artifact_hashes"]
            for arm in ("A", "B", "C", "D")
        },
        "interpretation": {
            arm: {
                "quality_success": comparisons[f"{arm}-A"]["quality_success"],
                "efficiency_success": comparisons[f"{arm}-A"]["efficiency_success"],
            }
            for arm in ("B", "C")
        },
    }


def _persisted_level_result(run: Mapping[str, Any]) -> tuple[Path | None, Mapping[str, Any]]:
    """Load the unique persisted level result associated with one run."""
    output_directory = Path(str(run["normalized_spec"]["outputs"]["directory"]))
    paths = sorted(output_directory.glob("*/units/*/levels/*/result.json"))
    if len(paths) != 1:
        return None, {}
    value = _load_json(paths[0])
    result = value.get("result")
    return paths[0], result if isinstance(result, Mapping) else {}


def audit_candidate_trajectories(report: Mapping[str, Any]) -> dict[str, Any]:
    """Audit persisted Trace/GEPA candidate lineage without inferring missing fields."""
    engine_rows: dict[str, list[dict[str, Any]]] = {"trace": [], "gepa": []}
    for run in report["runs"]:
        if run["arm"] not in {"B", "C"}:
            continue
        engine = "trace" if run["arm"] == "B" else "gepa"
        path, persisted = _persisted_level_result(run)
        metadata = persisted.get("metadata", {})
        trajectory = metadata.get("candidate_trajectory") if isinstance(metadata, Mapping) else None
        rows = trajectory if isinstance(trajectory, list) else []
        required = {
            "artifact_or_hash": bool(rows)
            and all("artifact" in row or "artifact_sha256" in row for row in rows),
            "parent_or_seed_relation": bool(rows)
            and all("parent_id" in row or "seed_relation" in row for row in rows),
            "candidate_evaluation": bool(rows)
            and all(isinstance(row.get("evaluation"), Mapping) for row in rows),
            "selected_or_rejected_status": bool(rows)
            and all(row.get("status") in {"selected", "rejected"} for row in rows),
        }
        engine_rows[engine].append(
            {
                "seed": run["seed"],
                "proposal_budget": run["proposal_budget"],
                "persisted_result": None
                if path is None
                else str(path.relative_to(REPOSITORY_ROOT)),
                "candidates_proposed": run["accounted"]["candidates_proposed"],
                "candidate_trajectory_records": len(rows),
                "recoverable": required,
            }
        )
    engines = {
        engine: {
            "runs_audited": len(rows),
            "all_required_fields_recoverable": bool(rows)
            and all(all(row["recoverable"].values()) for row in rows),
            "representative": rows[0] if rows else None,
        }
        for engine, rows in engine_rows.items()
    }
    ready = all(value["all_required_fields_recoverable"] for value in engines.values())
    return {
        "schema_version": "recursive-opt-episode-trajectory-audit/v1",
        "engines": engines,
        "ready_for_episode_export": ready,
        "episodes_exported": False,
        "smallest_required_extension": None
        if ready
        else (
            "Persist one candidate_trajectory record per Trace and GEPA proposal with "
            "candidate artifact or hash, parent/seed relation, canonical evaluation, "
            "and selected/rejected status. The engine adapters must emit this provenance "
            "without changing proposal, evaluation, or selection behavior."
        ),
    }


def _render_report(
    report: Mapping[str, Any],
    analysis: Mapping[str, Any],
    audit: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    """Render the systematic Experiment-0 report from frozen results."""
    absolute = analysis["absolute_by_arm"]
    lines = [
        "# Experiment 0 — main report",
        "",
        "## 1. Experiment purpose",
        "",
        "Cross-engine portability of one frozen compound-reasoning module under fixed, Trace, GEPA, and Trace-without-validation arms.",
        "",
        "## 2. Control-plane lock/provenance",
        "",
        f"- runtime tree: `{report['runs'][0]['runtime_tree_sha256']}`",
        f"- registry: `{report['runs'][0]['plan_registry_sha256']}` (arm-specific plan registry is retained per run)",
        f"- Experiment-0 source: `{report['runs'][0]['experiment_source_sha256']}`",
        "",
        "## 3. Readiness and skips",
        "",
        "The required CI, pilot, local readiness, optional-skip classification, and every main infrastructure gate passed before analysis.",
        "",
        "## 4–13. Frozen design and execution",
        "",
        "GSM8K used the frozen 16/12/24 train/validation/holdout pools, P0, exact-output evaluator, OpenRouter DeepSeek v4 Flash profiles, weighted accuracy/token objective, five paired seeds, and candidate budgets 6/12. Holdout was unavailable during optimization.",
        "",
        "## 14. Main results",
        "",
        "| arm | runs | accuracy mean | token ratio mean | invalid mean | forward calls/tokens | optimizer calls/tokens | selected changed | token-priced USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("A", "B", "C", "D"):
        value = absolute[arm]
        lines.append(
            f"| {arm} | {value['runs']} | {value['metrics']['accuracy']['mean']:.6f} | "
            f"{value['metrics']['forward_token_ratio']['mean']:.6f} | "
            f"{value['metrics']['invalid_rate']['mean']:.6f} | "
            f"{value['forward_calls']}/{value['forward_tokens']} | "
            f"{value['optimizer_calls']}/{value['optimizer_tokens']} | "
            f"{value['selected_artifact_changed_runs']} | "
            f"{value['token_priced_cost_usd']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## 15. Trace vs GEPA and fixed",
            "",
            "| comparison | accuracy delta [95% CI] | token-ratio delta [95% CI] | quality success | efficiency success |",
            "|---|---:|---:|---|---|",
        ]
    )
    for label in ("B-A", "C-A", "B-C", "D-B"):
        value = analysis["paired_comparisons"][label]
        accuracy = value["deltas"]["accuracy"]
        ratio = value["deltas"]["forward_token_ratio"]
        lines.append(
            f"| {label} | {accuracy['mean']:.6f} [{accuracy['paired_bootstrap_95_ci'][0]:.6f}, {accuracy['paired_bootstrap_95_ci'][1]:.6f}] | "
            f"{ratio['mean']:.6f} [{ratio['paired_bootstrap_95_ci'][0]:.6f}, {ratio['paired_bootstrap_95_ci'][1]:.6f}] | "
            f"{value['quality_success']} | {value['efficiency_success']} |"
        )
    lines.extend(
        [
            "",
            "## 16. Validation-gate ablation",
            "",
            "Arm D vs B is reported above. Proposal-level accepted/rejected/harmful/rollback counts are not inferable from current persisted artifacts and were not fabricated.",
            "",
            "## 17–18. Optional ablations",
            "",
            "Pareto and heterogeneous-artifact ablations were not part of the frozen primary run and were not executed.",
            "",
            "## 19. Cost accounting",
            "",
            f"Provider-reported monetary cost was unavailable. The frozen token-price proxy totals `${analysis['token_priced_cost_usd']:.8f}` and includes semantic retries.",
            "",
            "## 20. Failure analysis",
            "",
            "All main infrastructure gates passed. Historical GEPA and empty-response failures remain preserved separately and are not efficacy evidence.",
            "",
            "## 21. Episode dataset quality",
            "",
            f"Candidate-trajectory provenance ready: **{audit['ready_for_episode_export']}**. No episodes were exported because missing candidate artifact/parent/evaluation/selection records cannot be reconstructed without inference.",
            "",
            "## 22. Limitations",
            "",
            "One task was eligible; no second task was introduced post hoc. Statistical uncertainty uses paired seed-budget blocks. Provider billing fields were unavailable.",
            "",
            "## 23. Decision for next experiment",
            "",
            f"`{decision['status']}`",
            "",
            str(decision["reason"]),
        ]
    )
    return "\n".join(lines) + "\n"


def finalize_main_experiment(report: Mapping[str, Any]) -> dict[str, Any]:
    """Write statistics, provenance audit, report, and the next-step decision."""
    analysis = analyze_main_experiment(report)
    audit = audit_candidate_trajectories(report)
    if not audit["ready_for_episode_export"]:
        status = "RETURN_TO_CONTROL_PLANE"
        reason = (
            "Main execution and statistics completed, but Prompt-19 episode export is "
            "blocked by missing proposal-level candidate trajectory provenance."
        )
    else:
        status = "PROCEED_TO_PROMPT_19"
        reason = "Main execution, statistics, and candidate trajectory provenance passed."
    decision = {
        "schema_version": "recursive-opt-experiment-decision/v1",
        "status": status,
        "reason": reason,
        "control_plane": "validated",
        "trace_engine": "effective"
        if any(analysis["interpretation"]["B"].values())
        else "uncertain",
        "gepa_engine": "effective"
        if any(analysis["interpretation"]["C"].values())
        else "uncertain",
        "episode_dataset": "ready" if audit["ready_for_episode_export"] else "not ready",
        "smallest_required_extension": audit["smallest_required_extension"],
    }
    _write_json(MAIN_ANALYSIS_PATH, analysis)
    _write_json(EPISODE_AUDIT_PATH, audit)
    _write_json(MAIN_DECISION_PATH, decision)
    MAIN_REPORT_MARKDOWN_PATH.write_text(
        _render_report(report, analysis, audit, decision),
        encoding="utf-8",
    )
    return decision
