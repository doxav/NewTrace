"""A single parent-selection hook over production PrioritySearch for EXP-18."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery.exp18.pareto_selection import (
    TrainingCandidate,
    select_pareto_parent,
)
from opto.trainer import algorithms
from opto.trainer.algorithms.priority_search import ModuleCandidate, PrioritySearch


def _candidate_source(candidate: ModuleCandidate) -> str:
    """Read the actual production candidate's single trainable source artifact."""
    values = list(candidate.update_dict.values())
    if len(values) != 1 or not isinstance(values[0], str):
        raise RuntimeError(
            "Pareto parent must have exactly one trainable source string"
        )
    return values[0]


def make_pareto_trainer(
    *,
    namespace: str,
    outer_seed: int,
    proposal_slots: int,
    next_slot: Callable[[], int],
    training_archive: Callable[[], Sequence[TrainingCandidate]],
    record_decision: Callable[[dict[str, Any]], None],
) -> type[PrioritySearch]:
    """Bind immutable TRAIN receipts to production's one-parent exploration hook.

    Register the returned class with ``registered_pareto_trainer`` and inject
    its yielded name as the canonical module engine's ``trainer`` resource.
    ``training_archive`` must read already persisted TRAIN receipts;
    it must not invoke evaluation or touch validation/holdout. ``next_slot`` is
    the completed update count. ``record_decision`` must preserve an identical
    replay and refuse a conflicting replacement of the same slot decision.

    The production propose/backward/step/validate/memory/exploit methods remain
    unchanged. Final validation selection belongs to the common external driver.
    """
    if (
        not isinstance(namespace, str)
        or not namespace.strip()
        or type(outer_seed) is not int
        or type(proposal_slots) is not int
        or proposal_slots < 1
        or not all(
            callable(value) for value in (next_slot, training_archive, record_decision)
        )
    ):
        raise ValueError(
            "Pareto trainer requires a frozen context and callable receipt hooks"
        )

    class TrainingParetoPrioritySearch(PrioritySearch):
        """Explore one real archived candidate chosen by TRAIN-vector dominance."""

        def explore(
            self, verbose: bool = False, **kwargs: Any
        ) -> tuple[list[ModuleCandidate], list[float | None], dict[str, Any]]:
            """Select without changing the heap, executing code, or adding responses."""
            if self.num_candidates != 1 or self.num_proposals != 1:
                raise RuntimeError(
                    "Pareto adapter requires one parent and one proposal"
                )
            slot = next_slot()
            if type(slot) is not int or not 0 <= slot <= proposal_slots:
                raise RuntimeError(
                    "Pareto selector reached an unallocated response slot"
                )
            archive = tuple(training_archive())
            decision = select_pareto_parent(
                archive, namespace=namespace, outer_seed=outer_seed, next_slot=slot
            )
            candidates: dict[str, ModuleCandidate] = {}
            for _, candidate in self.memory:
                source = _candidate_source(candidate)
                source_hash = hashlib.sha256(source.encode()).hexdigest()
                candidates.setdefault(source_hash, candidate)
            valid_sources = {
                item.source_sha256
                for item in archive
                if item.auc_by_instance is not None
            }
            if not valid_sources.issubset(candidates):
                raise RuntimeError(
                    "TRAIN-valid archive source is absent from production memory"
                )
            if set(candidates) - {item.source_sha256 for item in archive}:
                raise RuntimeError("production memory contains an unregistered source")
            chosen = candidates[decision["selected_source_sha256"]]
            decision.update(
                {
                    "will_generate": slot < proposal_slots,
                    "available_memory_source_count": len(candidates),
                    "trainer_hook": "PrioritySearch.explore",
                    "trainer_runtime_alias": type(self).__name__,
                }
            )
            record_decision(decision)
            priority = self.compute_exploration_priority(chosen)
            return (
                [chosen],
                [priority],
                {
                    "num_exploration_candidates": 1,
                    "exploration_candidates_mean_priority": priority,
                    "exploration_candidates_mean_score": chosen.mean_score(),
                    "exploration_candidates_average_num_rollouts": chosen.num_rollouts,
                    "pareto_frontier_size": len(decision["frontier_indices"]),
                    "pareto_selected_index": decision["selected_index"],
                    "pareto_selected_non_scalar_best": (
                        decision["selected_source_sha256"]
                        != decision["scalar_best_source_sha256"]
                    ),
                },
            )

    context = json.dumps([namespace, outer_seed, proposal_slots], separators=(",", ":"))
    TrainingParetoPrioritySearch.__name__ = (
        "EXP18Pareto_" + hashlib.sha256(context.encode()).hexdigest()[:24]
    )
    return TrainingParetoPrioritySearch


@contextmanager
def registered_pareto_trainer(trainer_type: type[PrioritySearch]) -> Iterator[str]:
    """Temporarily expose the adapter to the existing string-only trainer resolver.

    The name is deterministic across resume. A collision is an error, never an
    override. Registration is removed even if execution raises. No production
    source, existing trainer entry, or trainer method is modified.
    """
    if (
        not isinstance(trainer_type, type)
        or not issubclass(trainer_type, PrioritySearch)
        or not trainer_type.__name__.startswith("EXP18Pareto_")
    ):
        raise ValueError("registration requires an EXP18 Pareto trainer factory result")
    name = trainer_type.__name__
    if hasattr(algorithms, name):
        raise RuntimeError("EXP18 Pareto trainer name is already registered")
    setattr(algorithms, name, trainer_type)
    try:
        yield name
    finally:
        if getattr(algorithms, name, None) is not trainer_type:
            raise RuntimeError("EXP18 trainer registration changed during execution")
        delattr(algorithms, name)
