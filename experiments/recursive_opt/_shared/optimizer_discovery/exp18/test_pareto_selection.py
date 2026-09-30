"""The EXP-18 parent selector must exercise a complete TRAIN-only archive."""

import hashlib
from dataclasses import replace
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery.exp18.pareto_selection import (
    TrainingCandidate,
    select_pareto_parent,
)


def candidate(index: int, values: tuple[float, ...] | None) -> TrainingCandidate:
    """Make a distinct unit-test artifact without running generated source."""
    return TrainingCandidate(
        index, hashlib.sha256(str(index).encode()).hexdigest(), values
    )


def choose(archive: list[TrainingCandidate], seed: int = 37) -> dict[str, Any]:
    """Select the next allocated parent with a fixed isolated namespace."""
    return select_pareto_parent(
        archive, namespace="EXP18-UNIT", outer_seed=seed, next_slot=len(archive) - 1
    )


def test_frontier_excludes_dominated_and_invalid_candidates() -> None:
    """Tradeoff specialists survive while aggregate-poor dominated records do not."""
    archive = [
        candidate(-1, (4.0, 4.0)),
        candidate(0, (1.0, 3.0)),
        candidate(1, (3.0, 0.0)),
        candidate(2, None),
    ]
    result = choose(archive)
    assert result["frontier_indices"] == [0, 1]
    assert result["invalid_indices"] == [2]
    assert result["scalar_best_index"] == 1
    assert result["selected_index"] in (0, 1)
    assert result["archive_indices"] == [-1, 0, 1, 2]
    assert result["valid_distinct_sources"] == 3


def test_seed_only_and_exact_replay() -> None:
    """Initialization chooses the trusted seed and replay is byte-equivalent data."""
    archive = [candidate(-1, (0.1, 0.2))]
    first = choose(archive)
    assert first == choose(archive)
    assert first["selected_index"] == -1
    assert first["frontier_indices"] == [-1]


def test_frontier_sampling_really_can_choose_a_non_scalar_parent() -> None:
    """A real specialist can be explored despite losing the scalar comparison."""
    archive = [candidate(-1, (0.1, 0.2)), candidate(0, (0.0, 0.4))]
    selected = {choose(archive, seed)["selected_index"] for seed in range(64)}
    assert selected == {-1, 0}
    assert all(choose(archive, seed)["scalar_best_index"] == -1 for seed in range(64))


def test_duplicate_sources_and_equal_vectors_have_no_extra_sampling_weight() -> None:
    """Repeated responses or identical behavior vectors retain earliest identities."""
    seed = candidate(-1, (0.1, 0.2))
    archive = [seed, replace(seed, index=0), candidate(1, (0.1, 0.2))]
    result = choose(archive)
    assert result["selected_index"] == -1
    assert result["frontier_indices"] == [-1]
    assert result["duplicate_source_indices"] == [0]
    assert result["equivalent_vector_indices"] == [1]
    assert result["archive_indices"] == [-1, 0, 1]


def test_archive_order_cannot_change_parent_choice() -> None:
    """Index order canonicalizes receipt loading and production heap iteration."""
    archive = [candidate(-1, (1.0, 3.0)), candidate(0, (3.0, 1.0))]
    assert choose(archive) == choose(list(reversed(archive)))


@pytest.mark.parametrize(
    "archive,slot,message",
    [
        ([candidate(-1, (1.0,))], 1, "every completed slot"),
        ([candidate(-1, (1.0,)), candidate(1, (0.1,))], 1, "every completed slot"),
        ([candidate(-1, (1.0,)), candidate(-1, (0.1,))], 1, "every completed slot"),
        ([candidate(-1, None)], 0, "trusted seed"),
        ([candidate(-1, (1.0,)), candidate(0, (0.1, 0.2))], 1, "same instance order"),
    ],
)
def test_incomplete_future_or_inconsistent_archives_fail(
    archive: list[TrainingCandidate], slot: int, message: str
) -> None:
    """Missing slots, unknown future evidence and broken evaluations are defects."""
    with pytest.raises(ValueError, match=message):
        select_pareto_parent(archive, namespace="UNIT", outer_seed=1, next_slot=slot)


def test_same_source_cannot_have_conflicting_training_receipts() -> None:
    """A source, panel and optimizer seed identify one deterministic measurement."""
    seed = candidate(-1, (0.1, 0.2))
    with pytest.raises(ValueError, match="conflicting TRAIN"):
        choose([seed, replace(seed, index=0, auc_by_instance=(0.1, 0.3))])


@pytest.mark.parametrize(
    "values", [(), (float("nan"),), (float("inf"),), (-0.1,), (True,)]
)
def test_invalid_numeric_vectors_fail(values: tuple[float, ...]) -> None:
    """Only finite nonnegative AUC values can enter the Pareto relation."""
    with pytest.raises(ValueError, match="AUC"):
        candidate(0, values)


@pytest.mark.parametrize("index", [-2, True, 0.5])
def test_invalid_candidate_index_fails(index: int) -> None:
    """A seed is -1 and completed response slots are nonnegative integers."""
    with pytest.raises(ValueError, match="index"):
        candidate(index, (0.1,))


def test_malformed_hash_fails() -> None:
    """Every archive identity must address exact persisted source bytes."""
    with pytest.raises(ValueError, match="SHA256"):
        TrainingCandidate(0, "missing-source", (0.1,))


@pytest.mark.parametrize(
    "kwargs", [{"namespace": ""}, {"outer_seed": True}, {"next_slot": -1}]
)
def test_invalid_selection_context_fails(kwargs: dict[str, Any]) -> None:
    """Selection cannot silently accept an ambiguous seeded context."""
    settings = {"namespace": "UNIT", "outer_seed": 1, "next_slot": 0} | kwargs
    with pytest.raises(ValueError):
        select_pareto_parent([candidate(-1, (0.1,))], **settings)
