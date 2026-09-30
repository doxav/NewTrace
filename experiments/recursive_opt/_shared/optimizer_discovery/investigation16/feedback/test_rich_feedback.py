"""Test-first requirements for an observation-only, complete feedback projection."""

import copy
import json
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery.investigation16.feedback import rich_feedback as R

SOURCE = "def propose(history, bounds, seed):\n    return [0.0]\n"
OTHER_SOURCE = "def propose(history, bounds, seed):\n    return [1.0]\n"
BOUNDS = [[-10.0, 10.0]]


def row(values: list[float], *, valid: bool = True, budget: int = 32) -> dict[str, Any]:
    """Construct a trajectory with hostile host metadata that must never propagate."""
    return {
        "valid": valid,
        "status": "valid" if valid else "timeout",
        "budget": budget,
        "observations": [{"x": [value**0.5], "value": value} for value in values],
        "normalization": "HIDDEN_SCALE_SENTINEL",
        "optimum": "HIDDEN_OPTIMUM_SENTINEL",
        "task_identity": "HIDDEN_TASK_SENTINEL",
        "stratum": "HIDDEN_FAMILY_SENTINEL",
        "metrics": {"auc": "HIDDEN_METRIC_SENTINEL"},
        "stdout": "HIDDEN_OUTPUT_SENTINEL",
        "local_seed": "HIDDEN_SEED_SENTINEL",
    }


def test_anytime_collision_is_resolved_with_indexed_improvements() -> None:
    """The raw-only projection distinguishes early and late discovery of zero."""
    early = row([100.0, 100.0] + [0.0] * 30)
    late = row([100.0] * 30 + [0.0, 0.0])
    left = R.build_feedback(SOURCE, [early], [BOUNDS])
    right = R.build_feedback(SOURCE, [late], [BOUNDS])
    assert left != right
    a = left["current"]["tasks"][0]
    b = right["current"]["tasks"][0]
    assert a["improvements"][0]["evaluation"] == 3
    assert b["improvements"][0]["evaluation"] == 31
    assert a["raw_anytime_mean"] == 6.25
    assert b["raw_anytime_mean"] == 93.75
    assert a["initial"]["evaluation"] == 1
    assert a["incumbent"]["evaluation"] == 3
    assert a["diagnostics"]["final_nonimproving_run"] == 29
    assert a["diagnostics"]["unique_points"] == 2


@pytest.mark.parametrize("tasks", [6, 7, 12, 18])
def test_all_tasks_and_all_improvements_are_preserved(tasks: int) -> None:
    """Increasing the panel does not silently truncate tasks or improvement events."""
    trajectory = row([float(value) for value in range(32, 0, -1)])
    result = R.build_feedback(SOURCE, [trajectory] * tasks, [BOUNDS] * tasks)
    assert len(result["current"]["tasks"]) == tasks
    assert all(len(item["improvements"]) == 31 for item in result["current"]["tasks"])
    encoded = R.serialize_feedback(result, max_chars=200000)
    assert json.loads(encoded) == result


def test_hidden_fields_are_excluded_and_serialization_is_deterministic() -> None:
    """Only explicitly approved raw fields cross from host evaluation to feedback."""
    rows = [row([1.0] * 32)]
    original = copy.deepcopy(rows)
    first = R.build_feedback(SOURCE, rows, [BOUNDS])
    second = R.build_feedback(SOURCE, rows, [BOUNDS])
    encoded = R.serialize_feedback(first, max_chars=20000)
    assert encoded == R.serialize_feedback(second, max_chars=20000)
    assert "HIDDEN_" not in encoded
    assert rows == original
    assert first["current"]["tasks"][0]["diagnostics"]["repeated_points"] == 31


def test_previous_code_has_identity_and_redundant_panel_is_not_repeated() -> None:
    """The generator can attribute failed changes and avoids duplicate current data."""
    current = [row([1.0] * 32)]
    redundant = R.build_feedback(
        SOURCE, current, [BOUNDS], previous_source=SOURCE, previous_rows=current
    )
    assert redundant["previous_attempt"]["same_as_current"] is True
    assert "tasks" not in redundant["previous_attempt"]
    different = R.build_feedback(
        SOURCE,
        current,
        [BOUNDS],
        previous_source=OTHER_SOURCE,
        previous_rows=[row([4.0] * 32)],
    )
    assert different["previous_attempt"]["source"] == OTHER_SOURCE
    assert (
        different["previous_attempt"]["source_sha256"]
        != different["current"]["source_sha256"]
    )


def test_invalid_partial_trajectory_keeps_error_without_full_trajectory_score() -> None:
    """An invalid partial trajectory remains typed invalid and is never given an AUC."""
    result = R.build_feedback(SOURCE, [row([100.0, 4.0], valid=False)], [BOUNDS])
    task = result["current"]["tasks"][0]
    assert not task["valid"] and task["status"] == "timeout"
    assert task["raw_anytime_mean"] is None
    assert task["observed_evaluations"] == 2
    assert task["incumbent"]["value"] == 4.0
    empty = R.build_feedback("", [row([], valid=False)], [BOUNDS])
    assert empty["current"]["tasks"][0]["initial"] is None


@pytest.mark.parametrize("bad_value", [float("inf"), float("nan"), True])
def test_nonfinite_or_boolean_values_are_rejected(bad_value: Any) -> None:
    """Malformed numbers cannot enter a JSON request or a summary statistic."""
    trajectory = row([1.0] * 32)
    trajectory["observations"][4]["value"] = bad_value
    with pytest.raises(ValueError, match="history"):
        R.build_feedback(SOURCE, [trajectory], [BOUNDS])


def test_bad_bounds_lengths_status_and_budget_are_rejected() -> None:
    """The adapter fails explicitly when trajectory structure or protocol is wrong."""
    with pytest.raises(ValueError, match="bounds"):
        R.build_feedback(SOURCE, [row([1.0] * 32)], [])
    invalid_point = row([1.0] * 32)
    invalid_point["observations"][0]["x"] = [11.0]
    with pytest.raises(ValueError, match="history"):
        R.build_feedback(SOURCE, [invalid_point], [BOUNDS])
    with pytest.raises(ValueError, match="budget"):
        R.build_feedback(SOURCE, [row([1.0] * 31)], [BOUNDS])
    unknown = row([1.0] * 32)
    unknown["status"] = "HIDDEN_STATUS_SENTINEL"
    with pytest.raises(ValueError, match="status"):
        R.build_feedback(SOURCE, [unknown], [BOUNDS])


def test_payload_limit_fails_without_truncating_json_or_dropping_tasks() -> None:
    """Oversized feedback requires a protocol decision instead of silent truncation."""
    payload = R.build_feedback(SOURCE, [row([1.0] * 32)] * 7, [BOUNDS] * 7)
    with pytest.raises(ValueError, match="limit"):
        R.serialize_feedback(payload, max_chars=100)
    assert len(payload["current"]["tasks"]) == 7
    with pytest.raises(ValueError, match="limit"):
        R.serialize_feedback(payload, max_chars=0)


@pytest.mark.parametrize("status", ["shape_error", "nonfinite", "out_of_bounds"])
def test_real_proposal_failure_statuses_are_supported(status: str) -> None:
    """The summary accepts the status vocabulary emitted by the portable runner."""
    failed = row([], valid=False)
    failed["status"] = status
    payload = R.build_feedback(SOURCE, [failed], [BOUNDS])
    assert payload["current"]["tasks"][0]["status"] == status


def test_malformed_panels_and_previous_provenance_fail_explicitly() -> None:
    """Bad panel types and incomplete previous attempts never yield partial output."""
    with pytest.raises(ValueError, match="bounds"):
        R.build_feedback(SOURCE, [row([1.0] * 32)], None)
    with pytest.raises(TypeError, match="mapping"):
        R.build_feedback(SOURCE, [None], [BOUNDS])
    with pytest.raises(ValueError, match="together"):
        R.build_feedback(SOURCE, [row([1.0] * 32)], [BOUNDS], previous_source=SOURCE)
    mismatch = row([1.0] * 32)
    mismatch["source_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="hash"):
        R.build_feedback(SOURCE, [mismatch], [BOUNDS])


def test_common_objective_defines_anytime_without_hidden_constants() -> None:
    """The same invariant instruction is suitable for both generation arms."""
    assert "anytime" in R.ANYTIME_OBJECTIVE.lower()
    assert "best-so-far" in R.ANYTIME_OBJECTIVE
    assert "32" not in R.ANYTIME_OBJECTIVE
