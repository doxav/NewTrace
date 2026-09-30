"""Test-first specification of fixed-bank diagnostic accounting and selection."""

from pathlib import Path

import numpy as np
import pytest

from experiments.recursive_opt._shared.optimizer_discovery.investigation16.selection import run
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.selection.run import (
    checked_result,
    choose,
    subset_schedule,
    verify_complete,
)


def test_resume_verifies_completed_result_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed result resumes without evaluation but cannot change its identity."""
    monkeypatch.setattr(run, "ROOT", tmp_path)
    key = {
        "split": "train",
        "source_sha256": "source",
        "task_identity": "task",
        "local_seed": 3,
        "budget": 32,
    }
    job = {"id": "a", "key": key}
    result = {
        key_: key[key_]
        for key_ in ("source_sha256", "task_identity", "local_seed", "budget")
    }
    run.E.persist(run.trajectory_path("train", "a"), {"key": key, "result": result})
    assert checked_result(job)["result"] == result
    changed = {"id": "a", "key": {**key, "local_seed": 4}}
    with pytest.raises(RuntimeError, match="key"):
        checked_result(changed)


def test_audit_guard_precedes_any_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent global selection freeze fails before task/job construction."""
    monkeypatch.setattr(run, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="blocked"):
        run.run_split({}, "audit")


def test_schedule_is_nested_and_reproducible() -> None:
    """Each draw preserves every instance/local seed once and replays exactly."""
    rows = subset_schedule()
    assert rows == subset_schedule()
    assert len(rows) == 200
    assert all(
        sorted(indices) == list(range(4))
        for row in rows
        for indices in row["instances"]
    )
    assert all(sorted(row["locals"]) == list(range(4)) for row in rows)


def test_invalid_policy_is_not_assigned_a_poor_score() -> None:
    """An otherwise attractive incomplete policy cannot be selected."""
    scores = np.ones((3, 6, 4, 4))
    scores[1] = 0.5
    scores[2] = 0.1
    valid = np.ones_like(scores, dtype=bool)
    valid[2] = False
    result = choose(scores, valid, subset_schedule()[0], 1, 1)
    assert result["bank_index"] == 1
    assert result["ranking"][2] is None


def test_ties_use_bank_position_and_weights_are_equal() -> None:
    """Tied policies select the earlier bank entry with balanced aggregation."""
    scores = np.ones((2, 6, 4, 4))
    scores[0, 0] = 7.0
    scores[1] = 2.0
    result = choose(
        scores, np.ones_like(scores, dtype=bool), subset_schedule()[0], 4, 4
    )
    assert result["bank_index"] == 0
    assert result["score"] == 2.0
    assert result["ranking"] == [2.0, 2.0]


def test_incomplete_grid_cannot_open_audit() -> None:
    """A missing completed training trajectory blocks the selection freeze."""
    with pytest.raises(RuntimeError, match="incomplete"):
        verify_complete({"a", "b"}, {"a"})
    verify_complete({"a", "b"}, {"a", "b"})


def test_trusted_seed_failure_is_an_engineering_defect() -> None:
    """Seed invalidity cannot become ordinary selection noise."""
    values = np.ones((2, 6, 4, 4))
    valid = np.ones_like(values, dtype=bool)
    valid[0] = False
    with pytest.raises(RuntimeError, match="trusted seed"):
        choose(values, valid, subset_schedule()[0], 4, 4)
