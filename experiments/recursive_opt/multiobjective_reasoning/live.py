"""Live micro-smoke and paired pilot runners for Experiment 0 v2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from opto.features.recursive_opt import spec as control_plane

from .components import ARTIFACT_KEYS, FORWARD_EVENTS, clear_forward_events, validate_artifact
from .evaluator import EVALUATOR_EVENTS, clear_evaluator_events
from .preflight import _load_json, _per_example_forward_usage, reliable_cost_usd
from .provenance import experiment_source_provenance
from .registration import assert_strict_output_evaluator, register_experiment_components
from .specs import INITIAL_ARTIFACT, MODEL, build_spec


PACKAGE_ROOT = Path(__file__).resolve().parent
CONTROL_PLANE_LOCK = PACKAGE_ROOT / "control_plane_lock_after_empty_text_retry.json"
MICRO_REPORT_PATH = PACKAGE_ROOT / "reports/live_micro_smoke_after_empty_text_retry.json"
MICRO_RUNS_DIRECTORY = PACKAGE_ROOT / "reports/micro_smoke_runs_after_empty_text_retry"
PILOT_REPORT_PATH = PACKAGE_ROOT / "reports/pilot_after_empty_text_retry.json"
PILOT_RUNS_DIRECTORY = PACKAGE_ROOT / "reports/pilot_runs_after_empty_text_retry"
COST_FORECAST_PATH = PACKAGE_ROOT / "reports/cost_forecast_after_empty_text_retry.json"
_OVERRIDE_ENVIRONMENT = (
    "RECURSIVE_OPT_TRAINER",
    "RECURSIVE_OPT_OPTIMIZER",
    "RECURSIVE_OPT_ITERATIONS",
    "RECURSIVE_OPT_NUM_CANDIDATES",
    "RECURSIVE_OPT_OPTIMIZER_KWARGS",
    "RECURSIVE_OPT_TRAINER_KWARGS",
    "RECURSIVE_OPT_LLM_PROFILES",
    "RECURSIVE_OPT_MODEL",
    "TRACE_LITELLM_MODEL",
)
_RESUME_INFRASTRUCTURE_PREREQUISITES = (
    "run_succeeded",
    "one_workflow_forward_per_evaluator",
    "evaluator_received_exact_output",
    "forward_calls_reconciled",
    "forward_tokens_reconciled",
    "holdout_inaccessible_during_optimization",
    "source_digest_stable",
    "experiment_source_digest_stable",
    "artifact_reloadable",
)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _baseline_tokens() -> tuple[str, dict[str, int]]:
    manifest = _load_json(PACKAGE_ROOT / "baseline_token_manifest.json")
    values = {
        str(row["sample_id"]): int(row["baseline_forward_tokens"])
        for row in manifest["samples"]
    }
    if len(values) != int(manifest["sample_count"]):
        raise RuntimeError("baseline token manifest contains duplicate sample IDs")
    return str(manifest["selected_task"]), values


def _locked_runtime_tree_sha256(lock_path: Path | None = None) -> str:
    """Return the runtime digest from the selected Experiment-0 lock."""
    lock = _load_json(CONTROL_PLANE_LOCK if lock_path is None else lock_path)
    digest = lock.get("control_plane", {}).get("runtime_tree_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Experiment 0 control-plane lock lacks a runtime digest")
    return digest


def _locked_experiment_source_sha256(lock_path: Path | None = None) -> str:
    """Return the Experiment-0 source digest from the selected protocol lock."""
    lock = _load_json(CONTROL_PLANE_LOCK if lock_path is None else lock_path)
    digest = lock.get("experiment", {}).get("source", {}).get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Experiment 0 protocol lock lacks a source digest")
    return digest


def _arm_configuration(arm: str) -> tuple[str, bool]:
    values = {
        "A": ("fixed", True),
        "B": ("trace", True),
        "C": ("gepa_optimize_anything", True),
        "D": ("trace", False),
    }
    if arm not in values:
        raise ValueError(f"unknown Experiment 0 arm {arm!r}")
    return values[arm]


def _proposal_path_exercised(
    *,
    optimized: bool,
    optimizer_usage: Mapping[str, Any],
    accounted: Mapping[str, Any],
) -> bool:
    """Return whether an optimized arm executed and evaluated a real proposal."""
    return not optimized or (
        int(optimizer_usage.get("calls", 0)) >= 1
        and int(optimizer_usage.get("total_tokens", 0)) > 0
        and int(accounted.get("candidates_proposed", 0)) >= 1
        and int(accounted.get("candidates_evaluated", 0)) >= 1
    )


def _selection_changed(*, optimized: bool, artifact: Mapping[str, Any]) -> bool:
    """Return whether selection retained an optimized artifact instead of P0."""
    return optimized and artifact != INITIAL_ARTIFACT


def _resume_infrastructure_ready(checks: Mapping[str, bool]) -> bool:
    """Return whether the prerequisites for a persistence/resume probe hold."""
    return all(checks[name] for name in _RESUME_INFRASTRUCTURE_PREREQUISITES)


def _infrastructure_checks_pass(checks: Mapping[str, bool]) -> bool:
    """Evaluate run gates while retaining selection outcome as a diagnostic."""
    return all(
        value for name, value in checks.items() if name != "selection_changed"
    )


def _pilot_optimizer_gates(runs: list[Mapping[str, Any]]) -> dict[str, bool]:
    """Keep proposal execution and selected-artifact change as separate pilot gates."""
    optimized = [run for run in runs if run["arm"] in {"B", "C", "D"}]
    trace_runs = [run for run in runs if run["arm"] in {"B", "D"}]
    gepa_runs = [run for run in runs if run["arm"] == "C"]
    return {
        "trace_real_proposal": bool(trace_runs)
        and all(run["checks"]["proposal_path_exercised"] for run in trace_runs),
        "gepa_real_proposal": bool(gepa_runs)
        and all(run["checks"]["proposal_path_exercised"] for run in gepa_runs),
        "optimized_artifact_differs": bool(optimized)
        and any(run["checks"]["selection_changed"] for run in optimized),
    }


def _pilot_retry_statistics(runs: list[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Aggregate metered semantic-response diagnostics by frozen pilot arm."""
    names = (
        "empty_text_responses",
        "semantic_retries",
        "semantic_retry_prompt_tokens",
        "semantic_retry_completion_tokens",
        "semantic_retry_total_tokens",
        "semantic_retry_cost_usd",
    )
    totals: dict[str, dict[str, float | int]] = {}
    for run in runs:
        target = totals.setdefault(str(run["arm"]), {name: 0 for name in names})
        usage = run.get("usage", {}).get("optimizer", {})
        for name in names:
            target[name] += usage.get(name, 0)
    return totals


def _execute_arm(
    *,
    arm: str,
    task: str,
    baseline_tokens: Mapping[str, int],
    seed: int,
    proposals: int,
    split_limits: Mapping[str, int],
    budget_limits: Mapping[str, int],
    output_directory: Path,
    control_plane_lock: Path | None = None,
) -> dict[str, Any]:
    engine, validation_gate = _arm_configuration(arm)
    hidden_environment = [name for name in _OVERRIDE_ENVIRONMENT if os.getenv(name)]
    raw = build_spec(
        task=task,
        engine=engine,
        seed=seed,
        output_directory=output_directory,
        proposals=proposals,
        validation_gate=validation_gate,
        baseline_tokens=baseline_tokens,
        split_limits=split_limits,
    )
    raw["budget"].update({name: int(value) for name, value in budget_limits.items()})
    raw["runtime"]["resume"] = True
    plan = control_plane.compile_plan(raw)
    normalized = _jsonable(plan.spec)
    holdout_ids = {
        str(row["id"])
        for row in _load_json(PACKAGE_ROOT / "manifests/dataset_manifest_v2.json")[
            "tasks"
        ][task]["samples"]
        if row["split"] == "holdout"
    }
    clear_forward_events()
    clear_evaluator_events()
    result = control_plane.run_spec(raw)
    per_example = _per_example_forward_usage()
    forward_output_ids = {id(event["output"]) for event in FORWARD_EVENTS}
    provider_calls = [
        call for row in per_example for call in row["provider_calls"]
    ]
    actual_providers = sorted(
        {str(call.get("actual_provider", "")) for call in provider_calls}
    )
    actual_models = sorted({str(call.get("actual_model", "")) for call in provider_calls})
    forward_usage = _jsonable(result.usage.get("forward", {}))
    optimizer_usage = _jsonable(result.usage.get("optimizer", {}))
    local_forward_tokens = sum(
        int(row["usage"].get("total_tokens", 0)) for row in per_example
    )
    accounted = _jsonable(result.budget["accounted"])
    artifact = _jsonable(result.artifact)
    artifact_reloadable = False
    try:
        validate_artifact(artifact)
        artifact_reloadable = set(artifact) == ARTIFACT_KEYS
    except (TypeError, ValueError):
        artifact_reloadable = False
    optimized = arm in {"B", "C", "D"}
    request_parameters = {
        role: raw["llm_profiles"][profile]["request_params"]
        for role, profile in (
            ("forward", "forward_primary"),
            ("optimizer", "optimizer_primary"),
        )
    }
    checks = {
        "run_succeeded": result.status == "success" and result.valid,
        "exact_model_available": actual_providers == ["openrouter"]
        and actual_models == [MODEL],
        "reasoning_parameters_recorded": request_parameters
        == {
            "forward": {"reasoning": {"enabled": False}},
            "optimizer": {"reasoning": {"effort": "low"}},
        },
        "one_workflow_forward_per_evaluator": len(FORWARD_EVENTS)
        == len(EVALUATOR_EVENTS)
        and len(FORWARD_EVENTS) > 0,
        "evaluator_received_exact_output": all(
            event["output_identity"] in forward_output_ids for event in EVALUATOR_EVENTS
        ),
        "usage_populated": int(forward_usage.get("calls", 0)) > 0
        and int(forward_usage.get("total_tokens", 0)) > 0
        and (
            not optimized
            or (
                int(optimizer_usage.get("calls", 0)) > 0
                and int(optimizer_usage.get("total_tokens", 0)) > 0
            )
        ),
        "forward_calls_reconciled": int(forward_usage.get("calls", 0))
        == int(accounted.get("eval_llm_calls", -1)),
        "forward_tokens_reconciled": int(forward_usage.get("total_tokens", 0))
        == local_forward_tokens,
        "holdout_inaccessible_during_optimization": not any(
            event["sample_id"] in holdout_ids and event["phase"] != "final_evaluation"
            for event in EVALUATOR_EVENTS
        ),
        "source_digest_stable": plan.code_provenance["runtime_tree_sha256"]
        == (
            _locked_runtime_tree_sha256()
            if control_plane_lock is None
            else _locked_runtime_tree_sha256(control_plane_lock)
        ),
        "experiment_source_digest_stable": experiment_source_provenance()["sha256"]
        == (
            _locked_experiment_source_sha256()
            if control_plane_lock is None
            else _locked_experiment_source_sha256(control_plane_lock)
        ),
        "artifact_reloadable": artifact_reloadable,
        "proposal_path_exercised": _proposal_path_exercised(
            optimized=optimized,
            optimizer_usage=optimizer_usage,
            accounted=accounted,
        ),
        "selection_changed": _selection_changed(
            optimized=optimized,
            artifact=artifact,
        ),
        "no_environment_override": not hidden_environment,
        "cache_not_shared": not any(call.get("cache_hit") is True for call in provider_calls),
    }
    resume_succeeded = False
    if _resume_infrastructure_ready(checks):
        clear_forward_events()
        clear_evaluator_events()
        resumed = control_plane.run_spec(raw)
        resume_succeeded = (
            resumed.to_dict() == result.to_dict()
            and not FORWARD_EVENTS
            and not EVALUATOR_EVENTS
        )
    checks["output_persistence_and_resume"] = resume_succeeded
    return {
        "arm": arm,
        "engine": engine,
        "validation_gate": validation_gate,
        "seed": seed,
        "proposal_budget": proposals,
        "split_limits": dict(split_limits),
        "budget_limits": _jsonable(result.budget),
        "status": result.status,
        "valid": result.valid,
        "error": result.error,
        "artifact": artifact,
        "metrics": _jsonable(result.evaluation.metrics),
        "usage": _jsonable(result.usage),
        "cost_usd": {
            "forward": reliable_cost_usd(forward_usage),
            "optimizer": reliable_cost_usd(optimizer_usage),
        },
        "accounted": accounted,
        "workflow_forwards": len(per_example),
        "evaluator_invocations": len(EVALUATOR_EVENTS)
        if not resume_succeeded
        else int(accounted.get("evaluator_runs", 0)),
        "provider_calls": provider_calls,
        "actual_providers": actual_providers,
        "actual_models": actual_models,
        "selected_models": _jsonable(result.metadata.get("selected_models", {})),
        "request_params": request_parameters,
        "hidden_environment": hidden_environment,
        "spec_fingerprint": plan.fingerprint,
        "runtime_tree_sha256": plan.code_provenance["runtime_tree_sha256"],
        "experiment_source_sha256": experiment_source_provenance()["sha256"],
        "plan_registry_sha256": plan.code_provenance["registry_sha256"],
        "checks": checks,
        "passed": _infrastructure_checks_pass(checks),
        "normalized_spec": normalized,
    }


def run_micro_smoke() -> dict[str, Any]:
    """Run the live one-unit A/B/C smoke and stop on the first failed arm."""
    register_experiment_components()
    assert_strict_output_evaluator()
    preregistration = _load_json(PACKAGE_ROOT / "manifests/preregistration_v2.json")
    task, baseline_tokens = _baseline_tokens()
    split_limits = preregistration["dataset_pools"]["micro_smoke_subset"]
    budget_limits = preregistration["resource_budgets"]["micro_smoke"]
    arms: dict[str, Any] = {}
    for arm in ("A", "B", "C"):
        result = _execute_arm(
            arm=arm,
            task=task,
            baseline_tokens=baseline_tokens,
            seed=0,
            proposals=1,
            split_limits=split_limits,
            budget_limits=budget_limits,
            output_directory=MICRO_RUNS_DIRECTORY / arm,
        )
        arms[arm] = result
        if not result["passed"]:
            break
    return {
        "schema_version": "recursive-opt-live-micro-smoke/v3",
        "task": task,
        "seed": 0,
        "arms": arms,
        "completed_arms": list(arms),
        "passed": list(arms) == ["A", "B", "C"]
        and all(result["passed"] for result in arms.values()),
    }


def run_pilot() -> dict[str, Any]:
    """Run the frozen three-seed, two-budget A/B/C/D Experiment 0 pilot."""
    register_experiment_components()
    assert_strict_output_evaluator()
    preregistration = _load_json(PACKAGE_ROOT / "manifests/preregistration_v2.json")
    micro = _load_json(MICRO_REPORT_PATH)
    if not micro.get("passed"):
        raise RuntimeError("pilot requires a passing live micro-smoke")
    forecast = _load_json(COST_FORECAST_PATH)
    if not forecast.get("pilot_forecast_complete"):
        raise RuntimeError("pilot requires a complete post-smoke cost forecast")
    task, baseline_tokens = _baseline_tokens()
    split_limits = preregistration["dataset_pools"]["pilot_subset"]
    budget_limits = preregistration["resource_budgets"]["pilot"]
    runs: list[dict[str, Any]] = []
    stopped_after: dict[str, Any] | None = None
    for seed in preregistration["pilot"]["paired_seeds"]:
        for proposals in preregistration["pilot"]["candidate_budgets"]:
            for arm in preregistration["pilot"]["arm_order"][str(seed)]:
                run = _execute_arm(
                    arm=arm,
                    task=task,
                    baseline_tokens=baseline_tokens,
                    seed=int(seed),
                    proposals=int(proposals),
                    split_limits=split_limits,
                    budget_limits=budget_limits,
                    output_directory=(
                        PILOT_RUNS_DIRECTORY
                        / f"seed-{seed}"
                        / f"budget-{proposals}"
                        / arm
                    ),
                )
                runs.append(run)
                if not run["passed"]:
                    stopped_after = {
                        "seed": seed,
                        "proposal_budget": proposals,
                        "arm": arm,
                    }
                    break
            if stopped_after:
                break
        if stopped_after:
            break
    eligibility = _load_json(PACKAGE_ROOT / "reports/task_eligibility_v2.json")
    skips = _load_json(PACKAGE_ROOT / "preflight_skips.json")
    configured_by_budget: dict[int, set[str]] = {}
    for run in runs:
        limits = {
            key: run["budget_limits"][key]
            for key in ("eval_llm_calls", "evaluator_runs", "total_tokens", "wall_time_s")
        }
        configured_by_budget.setdefault(int(run["proposal_budget"]), set()).add(
            json.dumps(limits, sort_keys=True)
        )
    gates = {
        "all_planned_runs_completed": len(runs) == 24 and stopped_after is None,
        "no_holdout_leakage": all(
            run["checks"]["holdout_inaccessible_during_optimization"] for run in runs
        ),
        "exact_source_digests": len({run["runtime_tree_sha256"] for run in runs}) == 1,
        "exact_experiment_source_digests": len(
            {run["experiment_source_sha256"] for run in runs}
        )
        == 1,
        "one_forward_per_evaluator": all(
            run["checks"]["one_workflow_forward_per_evaluator"] for run in runs
        ),
        "no_selected_invalid_artifact": all(run["valid"] for run in runs),
        **_pilot_optimizer_gates(runs),
        "forward_calls_reconcile": all(
            run["checks"]["forward_calls_reconciled"] for run in runs
        ),
        "forward_tokens_reconcile": all(
            run["checks"]["forward_tokens_reconciled"] for run in runs
        ),
        "evaluator_budget_comparable": all(
            len(values) == 1 for values in configured_by_budget.values()
        ),
        "forward_call_budget_comparable": all(
            len(values) == 1 for values in configured_by_budget.values()
        ),
        "outputs_and_resume": all(
            run["checks"]["output_persistence_and_resume"] for run in runs
        ),
        "task_remains_nonflat": bool(
            eligibility["task_results"][task]["quality_signal"]
            or eligibility["task_results"][task]["efficiency_signal"]
        ),
        "no_uncontrolled_cache_sharing": all(
            run["checks"]["cache_not_shared"] for run in runs
        ),
        "no_hidden_environment": all(
            run["checks"]["no_environment_override"] for run in runs
        ),
        "no_unexplained_critical_skip": int(skips["blocking_skips"]) == 0,
    }
    return {
        "schema_version": "recursive-opt-pilot/v3",
        "task": task,
        "planned_run_count": 24,
        "completed_run_count": len(runs),
        "stopped_after": stopped_after,
        "runs": runs,
        "retry_statistics_by_arm": _pilot_retry_statistics(runs),
        "gates": gates,
        "passed": all(gates.values()),
    }
