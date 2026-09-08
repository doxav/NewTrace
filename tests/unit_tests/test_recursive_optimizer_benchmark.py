"""Scientific benchmark invariants, including real isolated program execution."""

import math
from typing import Any

import pytest

from artifacts.optimizer_discovery import benchmark as B


def test_tasks_are_deterministic_disjoint_and_optima_feasible() -> None:
    """Semantic task identity, rather than labels alone, separates frozen splits."""
    tasks = [
        t
        for phase in ("pilot", "confirmation")
        for split in ("train", "validation", "holdout")
        for t in B.make_tasks(phase, split)
    ]
    assert B.make_tasks("pilot", "train") == B.make_tasks("pilot", "train")
    assert len({B.task_identity(t) for t in tasks}) == len(tasks) == 48
    for task in tasks:
        assert B.objective(task, task["shift"]) == 0
        assert all(-5 <= x <= 5 for x in task["shift"])
        assert math.isfinite(B.normalization(task)) and B.normalization(task) > 0
    with pytest.raises(ValueError):
        B.make_tasks("unknown", "train")


def test_metric_equations_and_censoring() -> None:
    """Regret AUC retains values above one and never treats censoring as attainment."""
    result = B.metrics([8, 4, 6, 0], 2, 4)
    assert result["auc"] == 2 and result["final_regret"] == 0
    assert result["target_evaluations"] == 4 and result["attained"]
    result = B.metrics([4, 4], 2, 2)
    assert result["auc"] == 2 and result["target_evaluations"] is None
    assert result["capped_target_evaluations"] == 3
    with pytest.raises(ValueError):
        B.metrics([1], 0, 1)
    with pytest.raises(ValueError):
        B.metrics([-0.1], 1, 1)
    assert B.metrics([-1e-14], 1, 1)["auc"] == 0


@pytest.mark.parametrize(
    "source,status",
    [
        ("def :", "syntax_error"),
        (
            "import os\ndef propose(history,bounds,seed): return os.getcwd()",
            "protocol_violation",
        ),
        (
            "def propose(history,bounds,seed): return open('x').read()",
            "protocol_violation",
        ),
        (
            "def propose(history,bounds,seed): return getattr(history, '__class__')",
            "protocol_violation",
        ),
        (
            "def propose(history,bounds,seed): return history.__class__",
            "protocol_violation",
        ),
        (B.SEED_SOURCE, "valid"),
    ],
)
def test_source_protocol_screen(source: str, status: str) -> None:
    """The declared static screen rejects forbidden operations before execution."""
    assert B.source_status(source) == status


def test_real_trajectory_budget_determinism_and_hidden_payload() -> None:
    """Programs see only the public three-argument contract in a fresh process."""
    task = B.make_tasks("pilot", "train")[0]
    source = "def propose(history, bounds, seed):\n    assert all(set(r) == {'x', 'value'} for r in history)\n    return [0.0] * len(bounds)\n"
    first = B.evaluate(source, task, 42, budget=3)
    second = B.evaluate(source, task, 42, budget=3)
    assert first["valid"] and first["objective_calls"] == 3
    assert first["observations"] == second["observations"]
    assert first["subprocess_executions"] == 6
    assert B.local_seed("pilot", 701, task) == B.local_seed("pilot", 701, task)


def test_failure_is_typed_and_fallback_preserves_history_budget() -> None:
    """A failed selected program switches permanently without erasing real observations."""
    task = B.make_tasks("pilot", "train")[0]
    source = "def propose(history, bounds, seed):\n    if len(history) >= 2: raise ValueError('bad')\n    return [0.0] * len(bounds)\n"
    invalid = B.evaluate(source, task, 4, budget=4)
    assert not invalid["valid"] and invalid["metrics"] is None
    assert invalid["objective_calls"] == 2
    deployed = B.evaluate(source, task, 4, budget=4, deployment=True)
    assert deployed["valid"] and not deployed["candidate_valid"]
    assert deployed["objective_calls"] == 4 and deployed["fallback_used"]
    assert deployed["observations"][:2] == invalid["observations"]
    with pytest.raises(RuntimeError, match="seed"):
        B.evaluate(
            source,
            task,
            4,
            budget=4,
            deployment=True,
            seed_source="def propose(history,bounds,seed): return []",
        )


def test_timeout_and_equal_stratum_aggregation(monkeypatch: Any) -> None:
    """Timeout remains an execution failure; task weights respect family/dimension strata."""
    task = B.make_tasks("pilot", "train")[0]
    result = B.evaluate(
        "def propose(history,bounds,seed):\n while True: pass",
        task,
        1,
        budget=2,
        timeout_s=0.05,
    )
    assert not result["valid"] and result["status"] == "timeout"
    assert (
        B.aggregate(
            [
                {"stratum": "a", "metrics": {"auc": 1}},
                {"stratum": "a", "metrics": {"auc": 3}},
                {"stratum": "b", "metrics": {"auc": 8}},
            ],
            "auc",
        )
        == 5
    )
