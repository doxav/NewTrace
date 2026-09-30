"""Test observed prefix selection and separate availability from seed retention."""

from experiments.recursive_opt._shared.optimizer_discovery.investigation16.budget.audit_budget import prefix


def test_eligible_but_worse_program_does_not_mean_empty_search() -> None:
    """Seed selection remains distinct from having no eligible generated program."""
    pool = [
        {"index": -1, "source_sha256": "seed", "eligible": True, "validation_auc": 0.1},
        {"index": 0, "source_sha256": "a", "eligible": True, "validation_auc": 0.3},
        {"index": 1, "source_sha256": "b", "eligible": False, "validation_auc": None},
    ]
    value = prefix(pool, 2)
    assert value["selected_index"] == -1
    assert value["eligible_generated"] == 1
    assert not value["no_eligible_generated"]


def test_prefix_keeps_invalid_slots_and_earliest_ties() -> None:
    """An invalid completed response consumes a slot without score imputation."""
    pool = [
        {"index": -1, "source_sha256": "seed", "eligible": True, "validation_auc": 0.2},
        {"index": 0, "source_sha256": "a", "eligible": False, "validation_auc": None},
        {"index": 1, "source_sha256": "b", "eligible": True, "validation_auc": 0.2},
    ]
    first = prefix(pool, 1)
    assert first["completed_slots"] == 1
    assert first["no_eligible_generated"]
    assert prefix(pool, 2)["selected_index"] == -1
