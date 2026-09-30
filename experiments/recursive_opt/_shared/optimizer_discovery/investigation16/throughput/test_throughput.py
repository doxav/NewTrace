"""Test concurrency-condition coverage and strict behavior-preservation checks."""

from experiments.recursive_opt._shared.optimizer_discovery.investigation16.throughput.run_throughput import (
    conditions,
    scientific_projection,
)


def test_all_workers_have_two_prespecified_rounds() -> None:
    """The counterbalanced schedule includes all conditions before results exist."""
    rows = conditions()
    assert [(r["round"], r["workers"]) for r in rows] == [
        (0, 4),
        (0, 1),
        (0, 16),
        (0, 8),
        (1, 16),
        (1, 8),
        (1, 4),
        (1, 1),
    ]


def test_comparison_excludes_time_only_and_detects_behavior_change() -> None:
    """Different timing is acceptable; changed values or validity cannot be hidden."""
    left = {"execution_s": 1.0, "valid": True, "metrics": {"auc": 0.1}}
    right = {"execution_s": 7.0, "valid": True, "metrics": {"auc": 0.1}}
    assert scientific_projection(left) == scientific_projection(right)
    right["metrics"]["auc"] = 0.2
    assert scientific_projection(left) != scientific_projection(right)
    right["metrics"]["auc"] = 0.1
    right["valid"] = False
    assert scientific_projection(left) != scientific_projection(right)
