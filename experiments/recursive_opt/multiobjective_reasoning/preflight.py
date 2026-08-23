"""Live fixed-artifact task eligibility preflight for Experiment 0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from opto.features.recursive_opt import spec as control_plane

from .components import FORWARD_EVENTS, clear_forward_events
from .evaluator import EVALUATOR_EVENTS, clear_evaluator_events
from .registration import assert_strict_output_evaluator, register_experiment_components
from .specs import build_spec


PACKAGE_ROOT = Path(__file__).resolve().parent


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path.name}")
    return value


def _provider_calls(trace: Any) -> list[dict[str, Any]]:
    traces = trace if isinstance(trace, list) else [trace]
    calls: list[dict[str, Any]] = []
    for item in traces:
        if isinstance(item, Mapping):
            calls.extend(
                dict(call)
                for call in item.get("provider_calls", [])
                if isinstance(call, Mapping)
            )
    return calls


def _per_example_forward_usage() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in FORWARD_EVENTS:
        output = getattr(event["output"], "data", event["output"])
        if not isinstance(output, Mapping):
            raise TypeError("forward event output must be a structured mapping")
        calls = [output["analysis"], output["answer_response"]]
        usage: dict[str, float | int] = {"calls": len(calls)}
        for call in calls:
            for name, amount in call.get("usage", {}).items():
                if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                    usage[name] = usage.get(name, 0) + amount
        rows.append(
            {
                "sample_id": str(output["sample_id"]),
                "usage": usage,
                "provider_calls": [dict(call["provider"]) for call in calls],
            }
        )
    return rows


def _run_probe_split(
    *,
    task: str,
    probe_name: str,
    artifact: Mapping[str, str],
    split: str,
    output_root: Path,
    limit: int | None = None,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    if limit is not None and sample_ids is not None:
        raise ValueError("probe split cannot combine limit and sample_ids")
    raw = build_spec(
        task=task,
        engine="fixed",
        seed=1803,
        output_directory=output_root / task / probe_name / split,
        split_limits=None if limit is None else {split: limit},
    )
    if sample_ids is not None:
        raw["levels"][0]["datasets"][split]["config"]["sample_ids"] = list(
            sample_ids
        )
    raw["levels"][0]["module"]["config"] = dict(artifact)
    raw["levels"][0]["module"]["artifact"] = dict(artifact)
    if split == "train":
        raw["levels"][0]["datasets"]["validation"] = []
    if split == "holdout":
        raw["levels"][0]["datasets"]["train"] = []
        raw["levels"][0]["datasets"]["validation"] = []
    else:
        raw["levels"][0]["datasets"]["holdout"] = []
    clear_forward_events()
    clear_evaluator_events()
    result = control_plane.run_spec(raw)
    if result.status == "error":
        raise RuntimeError(result.error or f"{task}/{probe_name}/{split} failed")
    if len(FORWARD_EVENTS) != len(EVALUATOR_EVENTS):
        raise RuntimeError("workflow/evaluator invocation count diverged")
    if any(
        event["output_identity"] not in {id(item["output"]) for item in FORWARD_EVENTS}
        for event in EVALUATOR_EVENTS
    ):
        raise RuntimeError("evaluator did not receive the exact traced output")
    usage = json.loads(json.dumps(result.usage))
    budget = json.loads(json.dumps(result.budget))
    provider_calls = _provider_calls(result.evaluation.trace)
    return {
        "status": result.status,
        "valid": result.valid,
        "sample_count": int(budget["accounted"]["evaluator_runs"]),
        "metrics": dict(result.evaluation.metrics),
        "usage": usage,
        "budget": budget,
        "provider_calls": provider_calls,
        "per_example_forward_usage": _per_example_forward_usage(),
        "cost_usd": reliable_cost_usd(usage["forward"]),
        "evaluator_records": json.loads(
            json.dumps(result.metadata.get("evaluator_records", []))
        ),
        "workflow_forwards": len(FORWARD_EVENTS),
        "evaluator_invocations": len(EVALUATOR_EVENTS),
    }


def reliable_cost_usd(usage: Mapping[str, Any]) -> float | None:
    """Return a provider-reported positive cost, or null when unavailable."""
    tokens = int(usage.get("total_tokens", 0))
    value = usage.get("cost_usd")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    cost = float(value)
    if cost < 0 or (tokens > 0 and cost == 0):
        return None
    return cost


def classify_near_eligible_tasks(
    v1_report: Mapping[str, Any], preregistration_v2: Mapping[str, Any]
) -> list[str]:
    """Mechanically identify v1 tasks warranting expanded v2 calibration."""
    mapping = preregistration_v2["v1_to_v2_task_names"]
    near: list[str] = []
    for v1_task, task in v1_report["task_results"].items():
        checks = task["checks"]
        prerequisite_names = (
            "baseline_accuracy_min",
            "baseline_accuracy_max",
            "invalid_rate",
            "token_use_variation",
            "nonempty_splits",
            "disjoint_splits",
        )
        failed_only_spread = all(checks[name] for name in prerequisite_names) and not checks[
            "probe_accuracy_spread"
        ]
        baseline = task["probes"]["P0"]
        feasible_efficiency = any(
            probe_id != "P0"
            and float(probe["invalid_rate"]) == 0.0
            and float(probe["accuracy"]) >= float(baseline["accuracy"]) - 0.02
            and float(probe["forward_tokens_per_example"])
            <= 0.80 * float(baseline["forward_tokens_per_example"])
            for probe_id, probe in task["probes"].items()
        )
        if failed_only_spread or feasible_efficiency:
            near.append(str(mapping[v1_task]))
    return near


def evaluate_v2_eligibility(
    probes: Mapping[str, Mapping[str, Any]],
    split_ids: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Apply the frozen validation-only v2 eligibility rule."""
    if "P0" not in probes:
        raise ValueError("v2 eligibility requires the frozen P0 baseline")

    def validation_metric(probe: Mapping[str, Any], name: str) -> float:
        return float(probe["splits"]["validation"]["metrics"][name])

    def validation_tokens(probe: Mapping[str, Any]) -> float:
        validation = probe["splits"]["validation"]
        count = int(validation["sample_count"])
        if count <= 0:
            raise ValueError("validation sample count must be positive")
        return float(validation["usage"]["forward"]["total_tokens"]) / count

    baseline = probes["P0"]
    baseline_accuracy = validation_metric(baseline, "accuracy")
    baseline_invalid = validation_metric(baseline, "invalid_rate")
    baseline_tokens = validation_tokens(baseline)
    quality_probe_ids: list[str] = []
    efficiency_probe_ids: list[str] = []
    feasible_ratios: list[float] = []
    for probe_id, probe in probes.items():
        if probe_id == "P0":
            continue
        accuracy = validation_metric(probe, "accuracy")
        invalid_rate = validation_metric(probe, "invalid_rate")
        token_ratio = validation_tokens(probe) / max(1.0, baseline_tokens)
        if invalid_rate == 0.0:
            feasible_ratios.append(token_ratio)
            if accuracy != baseline_accuracy:
                quality_probe_ids.append(probe_id)
            if accuracy >= baseline_accuracy - 0.02 and token_ratio <= 0.80:
                efficiency_probe_ids.append(probe_id)
    nonempty = all(split_ids.get(split) for split in ("train", "validation", "holdout"))
    disjoint = all(
        split_ids[left].isdisjoint(split_ids[right])
        for left, right in (
            ("train", "validation"),
            ("train", "holdout"),
            ("validation", "holdout"),
        )
    )
    checks = {
        "baseline_validation_accuracy_range": 0.20 <= baseline_accuracy <= 0.95,
        "baseline_validation_invalid_rate": baseline_invalid == 0.0,
        "nonempty_splits": nonempty,
        "disjoint_splits": disjoint,
        "informativeness": bool(quality_probe_ids or efficiency_probe_ids),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    validation_accuracies = [validation_metric(probe, "accuracy") for probe in probes.values()]
    token_means = [validation_tokens(probe) for probe in probes.values()]
    return {
        "baseline_validation_accuracy": baseline_accuracy,
        "baseline_validation_invalid_rate": baseline_invalid,
        "quality_signal": bool(quality_probe_ids),
        "quality_probe_ids": quality_probe_ids,
        "efficiency_signal": bool(efficiency_probe_ids),
        "efficiency_probe_ids": efficiency_probe_ids,
        "best_feasible_token_ratio": min(feasible_ratios) if feasible_ratios else None,
        "validation_accuracy_spread": max(validation_accuracies)
        - min(validation_accuracies),
        "token_spread": max(token_means) - min(token_means),
        "checks": checks,
        "eligible": not reasons,
        "exclusion_reasons": reasons,
    }


def select_eligible_task(
    task_results: Mapping[str, Mapping[str, Any]], candidate_order: list[str]
) -> tuple[str | None, str | None]:
    """Rank eligible tasks by comparable cost, then tokens, then frozen order."""
    eligible = [task for task in candidate_order if task_results.get(task, {}).get("eligible")]
    if not eligible:
        return None, None
    costs = [task_results[task].get("cost_usd") for task in eligible]
    comparable_cost = all(isinstance(value, (int, float)) for value in costs)
    if comparable_cost:
        return min(
            eligible,
            key=lambda task: (
                float(task_results[task]["cost_usd"]),
                candidate_order.index(task),
            ),
        ), "provider_cost_usd"
    return min(
        eligible,
        key=lambda task: (
            float(task_results[task]["forward_tokens_per_evaluated_example"]),
            candidate_order.index(task),
        ),
    ), "forward_tokens_per_evaluated_example"


def _weighted_metric(splits: Mapping[str, Mapping[str, Any]], name: str) -> float:
    count = sum(int(value["sample_count"]) for value in splits.values())
    if count <= 0:
        raise ValueError("preflight split sample count must be positive")
    return sum(
        float(value["metrics"][name]) * int(value["sample_count"])
        for value in splits.values()
    ) / count


def run_task_eligibility_preflight() -> dict[str, Any]:
    """Evaluate frozen manual probes and select the cheapest eligible task."""
    register_experiment_components()
    assert_strict_output_evaluator()
    preregistration = _load_json(PACKAGE_ROOT / "manifests/preregistration.json")
    dataset_manifest = _load_json(PACKAGE_ROOT / "manifests/dataset_manifest.json")
    probes = preregistration["manual_probes"]
    task_results: dict[str, Any] = {}
    output_root = PACKAGE_ROOT / "reports/task_preflight_runs"
    for task in preregistration["candidate_tasks"]:
        probe_results: dict[str, Any] = {}
        for probe_name, artifact in probes.items():
            splits = {
                split: _run_probe_split(
                    task=task,
                    probe_name=probe_name,
                    artifact=artifact,
                    split=split,
                    output_root=output_root,
                )
                for split in ("train", "validation")
            }
            provider_calls = [
                call
                for split_result in splits.values()
                for call in split_result["provider_calls"]
            ]
            probe_results[probe_name] = {
                "artifact_fingerprint": hashlib.sha256(
                    _canonical_json(artifact).encode("utf-8")
                ).hexdigest(),
                "accuracy": _weighted_metric(splits, "accuracy"),
                "invalid_rate": _weighted_metric(splits, "invalid_rate"),
                "forward_tokens_per_example": sum(
                    int(value["usage"]["forward"].get("total_tokens", 0))
                    for value in splits.values()
                )
                / sum(int(value["sample_count"]) for value in splits.values()),
                "latency_s": _weighted_metric(splits, "latency_s"),
                "sample_count": sum(
                    int(value["sample_count"]) for value in splits.values()
                ),
                "provider_cost_usd": sum(
                    float(value["usage"]["forward"].get("cost_usd", 0.0))
                    for value in splits.values()
                ),
                "provider_calls": provider_calls,
                "splits": splits,
            }
        accuracies = [float(value["accuracy"]) for value in probe_results.values()]
        token_means = [
            float(value["forward_tokens_per_example"])
            for value in probe_results.values()
        ]
        invalid_rates = [
            float(value["invalid_rate"]) for value in probe_results.values()
        ]
        source_samples = dataset_manifest["tasks"][task]["samples"]
        split_ids = {
            split: {
                sample["id"] for sample in source_samples if sample["split"] == split
            }
            for split in ("train", "validation", "holdout")
        }
        rule = preregistration["task_selection"]["rule"]
        checks = {
            "baseline_accuracy_min": accuracies[0]
            >= float(rule["baseline_accuracy_min"]),
            "baseline_accuracy_max": accuracies[0]
            <= float(rule["baseline_accuracy_max"]),
            "probe_accuracy_spread": max(accuracies) - min(accuracies)
            > float(rule["probe_accuracy_spread_strictly_greater_than"]),
            "token_use_variation": max(token_means) > min(token_means),
            "invalid_rate": max(invalid_rates)
            <= float(rule["maximum_invalid_rate"]),
            "nonempty_splits": all(split_ids.values()),
            "disjoint_splits": all(
                split_ids[left].isdisjoint(split_ids[right])
                for left, right in (
                    ("train", "validation"),
                    ("train", "holdout"),
                    ("validation", "holdout"),
                )
            ),
        }
        task_results[task] = {
            "eligible": all(checks.values()),
            "checks": checks,
            "baseline_accuracy": accuracies[0],
            "probe_accuracy_spread": max(accuracies) - min(accuracies),
            "invalid_rate_max": max(invalid_rates),
            "forward_token_spread": max(token_means) - min(token_means),
            "estimated_cost_usd": sum(
                float(value["provider_cost_usd"])
                for value in probe_results.values()
            ),
            "sample_count": sum(len(ids) for ids in split_ids.values()),
            "probes": probe_results,
        }
    eligible = [
        task
        for task in preregistration["candidate_tasks"]
        if task_results[task]["eligible"]
    ]
    selected = min(
        eligible,
        key=lambda task: (
            float(task_results[task]["estimated_cost_usd"]),
            preregistration["candidate_tasks"].index(task),
        ),
        default=None,
    )
    return {
        "schema_version": "recursive-opt-task-eligibility/v1",
        "selected_task": selected,
        "eligible_tasks": eligible,
        "holdout_used_for_eligibility": False,
        "task_results": task_results,
        "passed": selected is not None,
    }


def run_task_eligibility_preflight_v2() -> dict[str, Any]:
    """Run sequential expanded calibration only for mechanically near tasks."""
    register_experiment_components()
    assert_strict_output_evaluator()
    preregistration = _load_json(PACKAGE_ROOT / "manifests/preregistration_v2.json")
    dataset_manifest = _load_json(PACKAGE_ROOT / "manifests/dataset_manifest_v2.json")
    v1_report = _load_json(PACKAGE_ROOT / "reports/v1/task_eligibility.json")
    near_tasks = classify_near_eligible_tasks(v1_report, preregistration)
    subset = preregistration["dataset_pools"]["eligibility_subset"]
    if int(subset["holdout"]) != 0:
        raise RuntimeError("eligibility protocol must configure zero holdout evaluations")
    output_root = PACKAGE_ROOT / "reports/task_preflight_runs_v2"
    task_results: dict[str, Any] = {}
    for task in preregistration["candidate_tasks"]:
        if task not in near_tasks:
            task_results[task] = {
                "calibrated": False,
                "baseline_validation_accuracy": None,
                "baseline_validation_invalid_rate": None,
                "quality_signal": False,
                "quality_probe_ids": [],
                "efficiency_signal": False,
                "efficiency_probe_ids": [],
                "best_feasible_token_ratio": None,
                "validation_accuracy_spread": None,
                "token_spread": None,
                "cost_usd": None,
                "forward_tokens_per_evaluated_example": None,
                "eligible": False,
                "exclusion_reasons": ["not_near_eligible_under_preserved_v1_evidence"],
                "probes": {},
            }
            continue
        probe_results: dict[str, Any] = {}
        for probe_name, artifact in preregistration["manual_probes"].items():
            splits = {
                split: _run_probe_split(
                    task=task,
                    probe_name=probe_name,
                    artifact=artifact,
                    split=split,
                    output_root=output_root,
                    limit=int(subset[split]),
                )
                for split in ("train", "validation")
            }
            costs = [value["cost_usd"] for value in splits.values()]
            probe_results[probe_name] = {
                "artifact_fingerprint": hashlib.sha256(
                    _canonical_json(artifact).encode("utf-8")
                ).hexdigest(),
                "cost_usd": sum(float(value) for value in costs)
                if all(value is not None for value in costs)
                else None,
                "splits": splits,
            }
        source_samples = dataset_manifest["tasks"][task]["samples"]
        split_ids = {
            split: {
                str(sample["id"])
                for sample in source_samples
                if sample["split"] == split
            }
            for split in ("train", "validation", "holdout")
        }
        eligibility = evaluate_v2_eligibility(probe_results, split_ids)
        all_split_results = [
            split_result
            for probe in probe_results.values()
            for split_result in probe["splits"].values()
        ]
        costs = [value["cost_usd"] for value in all_split_results]
        total_examples = sum(int(value["sample_count"]) for value in all_split_results)
        total_forward_tokens = sum(
            int(value["usage"]["forward"].get("total_tokens", 0))
            for value in all_split_results
        )
        task_results[task] = {
            "calibrated": True,
            **eligibility,
            "cost_usd": sum(float(value) for value in costs)
            if all(value is not None for value in costs)
            else None,
            "forward_tokens_per_evaluated_example": total_forward_tokens
            / total_examples,
            "eligibility_subset_ids": {
                split: [
                    str(sample["id"])
                    for sample in source_samples
                    if sample["split"] == split
                ][: int(subset[split])]
                for split in ("train", "validation")
            },
            "probes": probe_results,
        }
    selected, ranking_basis = select_eligible_task(
        task_results, list(preregistration["candidate_tasks"])
    )
    eligible = [
        task
        for task in preregistration["candidate_tasks"]
        if task_results[task]["eligible"]
    ]
    return {
        "schema_version": "recursive-opt-task-eligibility/v2",
        "experiment_version": preregistration["experiment_version"],
        "v1_evidence_sha256": hashlib.sha256(
            (PACKAGE_ROOT / "reports/v1/task_eligibility.json").read_bytes()
        ).hexdigest(),
        "near_eligible_tasks_from_v1": near_tasks,
        "calibration_sequence": near_tasks,
        "eligibility_subset": dict(subset),
        "holdout_evaluator_invocations": 0,
        "holdout_used_for_eligibility": False,
        "selected_task": selected,
        "selection_ranking_basis": ranking_basis,
        "eligible_tasks": eligible,
        "cost_semantics": {
            "zero_with_nonzero_tokens_means_unavailable": True,
            "unavailable_cost_representation": None,
            "pilot_cost_forecast_requirement": "deferred to and mandatory after live micro-smoke",
        },
        "task_results": task_results,
        "passed": selected is not None,
    }
