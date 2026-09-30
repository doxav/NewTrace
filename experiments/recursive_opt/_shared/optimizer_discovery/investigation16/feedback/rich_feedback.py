"""Observation-only feedback for prospective EXP-16 mechanism ablations.

This module changes no frozen EXP-15 behavior. It sends neither host benchmark
metadata nor normalized scientific scores to the generator. Output serialization
has an explicit size gate and never truncates an observation/event or task panel.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from typing import Any

from opto.features.recursive_opt.optimizer_program import _validate_inputs

ANYTIME_OBJECTIVE = (
    "Optimize anytime performance: for each task, average the best-so-far "
    "objective value after every evaluation from 1 through the declared budget. "
    "Lower is better. Earlier improvements reduce this average across more "
    "evaluations. Task averages receive fixed positive task-specific scaling "
    "and fixed aggregation weights, shared by every candidate. Optimize this "
    "full-budget best-so-far criterion."
)

_STATUSES = frozenset(
    {
        "valid",
        "missing_source",
        "source_size",
        "syntax_error",
        "protocol_violation",
        "timeout",
        "exception",
        "import_error",
        "missing_propose",
        "signature_error",
        "process_error",
        "nondeterministic",
        "shape_error",
        "nonfinite",
        "out_of_bounds",
    }
)


def _source_hash(source: str) -> str:
    """Hash exact source text without parsing or repairing invalid candidate code."""
    if not isinstance(source, str):
        raise TypeError("source must be text")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _event(index: int, observation: dict[str, Any]) -> dict[str, Any]:
    """Copy the permitted raw observation and attach its one-based evaluation index."""
    return {
        "evaluation": index,
        "x": list(observation["x"]),
        "value": observation["value"],
    }


def summarize_trajectory(
    row: dict[str, Any], bounds: list[list[float]]
) -> dict[str, Any]:
    """Project a validated host row onto raw progress, validity and behavior fields."""
    if not isinstance(row, dict):
        raise TypeError("trajectory must be a mapping")
    budget = row.get("budget")
    observations = row.get("observations")
    valid = row.get("valid")
    status = row.get("status")
    if type(budget) is not int or budget <= 0:
        raise ValueError("trajectory budget must be a positive integer")
    if (
        type(valid) is not bool
        or not isinstance(status, str)
        or status not in _STATUSES
    ):
        raise ValueError("trajectory needs typed validity and a declared status")
    if (status == "valid") != valid:
        raise ValueError("trajectory status and validity disagree")
    _validate_inputs(observations, bounds, seed=0, timeout_s=1.0)
    if len(observations) > budget or (valid and len(observations) != budget):
        raise ValueError("trajectory observations disagree with the budget")
    if row.get("fallback_used", False):
        raise ValueError("training feedback cannot contain deployment fallback")

    initial = _event(1, observations[0]) if observations else None
    incumbent = initial
    improvements: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    nonimproving = 0
    longest_nonimproving = 0
    for index, observation in enumerate(observations, start=1):
        if index > 1:
            if observation["value"] < incumbent["value"]:
                incumbent = _event(index, observation)
                improvements.append(incumbent)
                nonimproving = 0
            else:
                nonimproving += 1
                longest_nonimproving = max(longest_nonimproving, nonimproving)
        curve.append({"evaluation": index, "value": incumbent["value"]})
    points = [tuple(observation["x"]) for observation in observations]
    unique_points = len(set(points))
    boundary_points = sum(
        any(coordinate in pair for coordinate, pair in zip(point, bounds))
        for point in points
    )
    extent = (
        [
            [
                min(point[dimension] for point in points),
                max(point[dimension] for point in points),
            ]
            for dimension in range(len(bounds))
        ]
        if points
        else None
    )
    return {
        "valid": valid,
        "status": status,
        "budget": budget,
        "bounds": [list(pair) for pair in bounds],
        "observed_evaluations": len(observations),
        "initial": initial,
        "incumbent": incumbent,
        "improvements": improvements,
        "best_so_far_curve": curve,
        "raw_anytime_mean": (
            statistics.mean(item["value"] for item in curve) if valid else None
        ),
        "diagnostics": {
            "unique_points": unique_points,
            "repeated_points": len(points) - unique_points,
            "points_on_boundary": boundary_points,
            "coordinate_extent": extent,
            "longest_nonimproving_run": longest_nonimproving,
            "final_nonimproving_run": nonimproving,
        },
    }


def _panel(
    source: str, rows: list[dict[str, Any]], bounds: list[list[list[float]]]
) -> dict[str, Any]:
    """Preserve every trajectory in order with exact source identity and no host IDs."""
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(bounds, list)
        or len(rows) != len(bounds)
    ):
        raise ValueError(
            "nonempty trajectory panel and bounds must have matching lengths"
        )
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("trajectory must be a mapping")
    digest = _source_hash(source)
    if any(row.get("source_sha256", digest) != digest for row in rows):
        raise ValueError("trajectory source hash differs from the declared source")
    tasks = [summarize_trajectory(row, limits) for row, limits in zip(rows, bounds)]
    return {
        "source_sha256": digest,
        "valid": all(task["valid"] for task in tasks),
        "tasks": tasks,
    }


def build_feedback(
    current_source: str,
    current_rows: list[dict[str, Any]],
    bounds_by_trajectory: list[list[list[float]]],
    *,
    previous_source: str | None = None,
    previous_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build deterministic raw progress feedback with an attributable previous attempt.

    The current source is identified by hash and is supplied separately in the
    generation prompt. A different previous attempt includes its exact source.
    Identical source and projected observations become an explicit reference.
    Serialize with ``serialize_feedback`` to enforce the registered output limit.
    """
    if (previous_source is None) != (previous_rows is None):
        raise ValueError("previous source and trajectories must be supplied together")
    current = _panel(current_source, current_rows, bounds_by_trajectory)
    result = {
        "schema": "investigation16.raw_trace_feedback.v1",
        "current": current,
    }
    if previous_source is not None and previous_rows is not None:
        previous = _panel(previous_source, previous_rows, bounds_by_trajectory)
        result["previous_attempt"] = (
            {"same_as_current": True, "source_sha256": current["source_sha256"]}
            if previous == current
            else {**previous, "source": previous_source}
        )
    return result


def serialize_feedback(payload: dict[str, Any], *, max_chars: int) -> str:
    """Encode finite JSON deterministically or reject an exceeded size limit intact."""
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("feedback size limit must be a positive integer")
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text) > max_chars:
        raise ValueError("feedback exceeds the registered size limit")
    return text
