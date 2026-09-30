"""Pure deterministic TRAIN-frontier selection for the EXP-18 production adapter.

The caller supplies already evaluated TRAIN receipts in the frozen instance
order. This module does not evaluate candidates, inspect validation, or generate
proposals. Selection samples the frontier instead of scalarizing it again.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from opto.trainer.objectives import pareto_rank


@dataclass(frozen=True)
class TrainingCandidate:
    """One completed proposal's TRAIN vector; ``None`` preserves typed invalidity.

    ``auc_by_instance`` averages registered local replicates within each TRAIN
    instance, in the exact frozen instance order shared by every candidate.
    The seed has index -1. Candidate source bytes remain with the caller.
    """

    index: int
    source_sha256: str
    auc_by_instance: tuple[float, ...] | None

    def __post_init__(self) -> None:
        """Reject incomplete identities and numeric values before ranking."""
        if type(self.index) is not int or self.index < -1:
            raise ValueError("candidate index must be an integer at least -1")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_sha256
        ):
            raise ValueError("candidate source must have a lowercase SHA256 digest")
        values = self.auc_by_instance
        if values is not None and (
            not isinstance(values, tuple)
            or not values
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in values
            )
        ):
            raise ValueError(
                "TRAIN AUC must be a nonempty tuple of finite nonnegative values"
            )


def _digest(value: Any) -> str:
    """Hash an explicit JSON context without Python's randomized hash()."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def select_pareto_parent(
    archive: Sequence[TrainingCandidate],
    *,
    namespace: str,
    outer_seed: int,
    next_slot: int,
) -> dict[str, Any]:
    """Sample one nondominated TRAIN parent and return a complete audit decision.

    Include the trusted seed and every completed slot strictly before
    ``next_slot``, including invalid responses. Identical sources and exactly
    equal vectors get one sampling position, represented by the earliest index.
    Uniform sampling uses ``Random(SHA256(namespace, outer_seed, next_slot))``;
    it cannot depend on heap order, wall time, validation, or process state.
    """
    if (
        not isinstance(namespace, str)
        or not namespace.strip()
        or type(outer_seed) is not int
        or type(next_slot) is not int
        or next_slot < 0
    ):
        raise ValueError(
            "selection requires a namespace, integer seed and nonnegative slot"
        )
    if any(not isinstance(item, TrainingCandidate) for item in archive):
        raise ValueError("archive requires typed TRAIN candidate receipts")
    ordered = sorted(archive, key=lambda item: item.index)
    if [item.index for item in ordered] != list(range(-1, next_slot)):
        raise ValueError(
            "archive must include the seed and every completed slot exactly once"
        )
    if ordered[0].auc_by_instance is None:
        raise ValueError("trusted seed has an invalid TRAIN receipt")
    dimensions = len(ordered[0].auc_by_instance)
    if any(
        item.auc_by_instance is not None and len(item.auc_by_instance) != dimensions
        for item in ordered
    ):
        raise ValueError(
            "all TRAIN vectors must use the same instance order and length"
        )

    unique_sources: dict[str, TrainingCandidate] = {}
    duplicates: list[int] = []
    for item in ordered:
        previous = unique_sources.get(item.source_sha256)
        if previous is not None:
            if previous.auc_by_instance != item.auc_by_instance:
                raise ValueError("same source has conflicting TRAIN receipts")
            duplicates.append(item.index)
        else:
            unique_sources[item.source_sha256] = item
    valid = [
        item for item in unique_sources.values() if item.auc_by_instance is not None
    ]
    scalar_best = min(
        valid,
        key=lambda item: (math.fsum(item.auc_by_instance) / dimensions, item.index),
    )

    unique_vectors: dict[tuple[float, ...], TrainingCandidate] = {}
    equivalents: list[int] = []
    for item in valid:
        if item.auc_by_instance in unique_vectors:
            equivalents.append(item.index)
        else:
            unique_vectors[item.auc_by_instance] = item
    candidates = list(unique_vectors.values())
    # Reuse production dominance; negation converts minimization to its convention.
    scores = [
        {str(index): -value for index, value in enumerate(item.auc_by_instance)}
        for item in candidates
    ]
    ranks = pareto_rank(scores)
    frontier = [item for item, rank in zip(candidates, ranks) if rank == 0]
    seed_digest = _digest(["EXP18-PARETO-UNIFORM-V1", namespace, outer_seed, next_slot])
    selected = frontier[random.Random(int(seed_digest, 16)).randrange(len(frontier))]
    receipts = [
        {
            "index": item.index,
            "source_sha256": item.source_sha256,
            "auc_by_instance": item.auc_by_instance,
        }
        for item in ordered
    ]
    return {
        "schema": "EXP18-TRAIN-PARETO-DECISION-V1",
        "namespace": namespace,
        "outer_seed": outer_seed,
        "next_slot": next_slot,
        "selected_index": selected.index,
        "selected_source_sha256": selected.source_sha256,
        "scalar_best_index": scalar_best.index,
        "scalar_best_source_sha256": scalar_best.source_sha256,
        "frontier_indices": [item.index for item in frontier],
        "frontier_source_sha256": [item.source_sha256 for item in frontier],
        "archive_indices": [item.index for item in ordered],
        "invalid_indices": [
            item.index for item in ordered if item.auc_by_instance is None
        ],
        "duplicate_source_indices": duplicates,
        "equivalent_vector_indices": equivalents,
        "valid_distinct_sources": len(valid),
        "instance_count": dimensions,
        "archive_sha256": _digest(receipts),
        "sampling_seed_sha256": seed_digest,
    }
