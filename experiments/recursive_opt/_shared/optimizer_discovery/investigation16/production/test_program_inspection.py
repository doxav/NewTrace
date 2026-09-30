"""Synthetic checks for the read-only program inspection; no benchmark imports."""

from __future__ import annotations

import json

import pytest

from experiments.recursive_opt._shared.optimizer_discovery.investigation16.production import (
    program_inspection as P,
)


def test_lineage_rejects_future_parent_and_retains_duplicate_origins() -> None:
    """Source identities must refer to earlier proposals, including repeated sources."""
    rows = [
        {"index": 0, "parent_sha256": "seed", "source_sha256": "a"},
        {"index": 1, "parent_sha256": "seed", "source_sha256": "a"},
        {"index": 2, "parent_sha256": "a", "source_sha256": "b"},
    ]
    enriched = P.lineage(rows, "seed")
    assert enriched[2]["parent_origin_slots"] == [0, 1]
    assert enriched[2]["depth_range"] == [2, 2]
    with pytest.raises(ValueError, match="earlier"):
        P.lineage([dict(rows[0], parent_sha256="future")], "seed")


def test_feedback_projection_counts_improvement_coordinates_only() -> None:
    """The entire incumbent curve is distinct from an entire observation history."""
    row = {
        "valid": True,
        "status": "valid",
        "observations": [
            {"x": [0.0], "value": 10.0},
            {"x": [1.0], "value": 12.0},
            {"x": [2.0], "value": 5.0},
        ],
    }
    first = {"evaluation": 1, **row["observations"][0]}
    best = {"evaluation": 3, **row["observations"][2]}
    task = {
        "valid": True,
        "status": "valid",
        "observed_evaluations": 3,
        "initial": first,
        "incumbent": best,
        "improvements": [best],
        "best_so_far_curve": [
            {"evaluation": i, "value": value}
            for i, value in enumerate([10.0, 10.0, 5.0], 1)
        ],
    }
    assert P.verify_projection(task, row) == (3, 2)
    with pytest.raises(ValueError, match="curve"):
        P.verify_projection(dict(task, best_so_far_curve=[]), row)
    with pytest.raises(ValueError, match="undeclared"):
        P.verify_projection(dict(task, optimum=0.0), row)


def test_feedback_envelope_supports_native_invalid_without_execution() -> None:
    """Decode only literal containers and JSON, never evaluated candidate source."""
    payload = {"current": {"source_sha256": "s", "tasks": [], "valid": False}}
    text = json.dumps(payload)
    assert P.feedback_payload("ID [0]: " + text) == payload
    assert P.feedback_payload("ID [0]: " + repr([text])) == payload
    with pytest.raises((ValueError, SyntaxError)):
        P.feedback_payload("ID [0]: [__import__('os').getcwd()]")


def test_behavior_counts_repeated_and_boundary_points_without_source_execution() -> (
    None
):
    """Recompute descriptive geometry using saved observations only."""
    row = {
        "candidate_valid": True,
        "fallback_used": False,
        "observations": [
            {"x": [0.0, 0.0], "value": 1.0},
            {"x": [5.0, 0.0], "value": 2.0},
            {"x": [5.0, 0.0], "value": 2.0},
        ],
    }
    result = P.behavior([row])
    assert result["repeated_points"] == 1
    assert result["boundary_points"] == 2
    assert result["midpoint_initials"] == 1
