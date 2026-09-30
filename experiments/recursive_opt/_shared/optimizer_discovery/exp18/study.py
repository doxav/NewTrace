"""Compact TRAIN feedback, prior-attempt memory and Pareto hooks for EXP-18.

The experiment retains the existing Control Plane/Trace/Optimizer/PrioritySearch
loop and common evaluator. This owner only builds authenticated prompt contexts
and injects the declared single-parent exploration hook for P and PM.
"""

from __future__ import annotations

import ast
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import study as N
from experiments.recursive_opt._shared.optimizer_discovery.exp18 import memory_projection as M
from experiments.recursive_opt._shared.optimizer_discovery.exp18.pareto_selection import (
    TrainingCandidate,
    select_pareto_parent,
)
from experiments.recursive_opt._shared.optimizer_discovery.exp18.pareto_trainer import (
    make_pareto_trainer,
    registered_pareto_trainer,
)
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import search_experiment as S
from opto.features.recursive_opt import spec as control

_BINDING_LOCK = threading.Lock()
_CURRENT_SCHEMA = "exp18-current-training-v1"


def _decode_feedback(feedback: str | None) -> dict[str, Any]:
    """Decode the existing scalar and typed-invalid production feedback envelopes."""
    if not isinstance(feedback, str) or not feedback.startswith("ID [0]: "):
        raise RuntimeError("unexpected propagated TRAIN feedback envelope")
    text = feedback[len("ID [0]: ") :].strip()
    if text.startswith("["):
        invalid = ast.literal_eval(text)
        if (
            not isinstance(invalid, list)
            or len(invalid) != 1
            or not isinstance(invalid[0], str)
        ):
            raise RuntimeError("unexpected propagated invalid-feedback envelope")
        text = invalid[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("propagated feedback must contain one TRAIN projection")
    return value


class MechanismStudy(N.Study):
    """Own EXP-18 contexts while inheriting proposals, evaluator and final selection."""

    def __init__(self, root: Path, arm: str, *, client: Any = None) -> None:
        """Bind a registered EXP-18 arm and initialize replay-local update counting."""
        super().__init__(root, arm, client=client)
        if self.config["experiment"] != "EXP-18" or arm not in {"L", "M", "P", "PM"}:
            raise ValueError("mechanism owner requires a registered EXP-18 arm")
        self._next_slot = 0

    def _directory(self, outer: int) -> Path:
        """Locate only this arm's registered outer-seed evidence."""
        if outer not in self.seeds:
            raise ValueError("unregistered outer seed")
        return self.root / str(outer) / self.arm

    def _receipt(self, outer: int, index: int) -> dict[str, Any]:
        """Authenticate an existing TRAIN receipt without filling missing history."""
        directory = self._directory(outer)
        path = directory / (
            "seed_train_receipt.json"
            if index == -1
            else f"slot_{index:02d}/train_receipt.json"
        )
        if not E.exists(path):
            raise RuntimeError(
                "missing historical TRAIN receipt; archive cannot evaluate it"
            )
        return self.training_receipt(outer, index)

    def _parent_receipt(
        self, outer: int, parent: str, before_slot: int
    ) -> tuple[int, dict[str, Any], int]:
        """Resolve the earliest authenticated parent occurrence strictly before this slot."""
        digest = B.source_hash(parent)
        for index in range(-1, before_slot):
            receipt = self._receipt(outer, index)
            if receipt["source_sha256"] == digest:
                completed = (
                    0
                    if index == -1
                    else E.read(
                        self._directory(outer) / f"slot_{index:02d}/response.json"
                    )["completed_ns"]
                )
                return index, receipt, completed
        raise RuntimeError(
            "propagated parent has no earlier authenticated TRAIN receipt"
        )

    @staticmethod
    def _training_projection(receipt: dict[str, Any]) -> dict[str, Any]:
        """Exclude host instance vectors, row hashes and split parameters from prompts."""
        keys = (
            "split",
            "source_sha256",
            "observed_ns",
            "allocated_trajectories",
            "observed_trajectories",
            "valid_trajectories",
            "status_counts",
            "aggregate_auc",
        )
        return {
            **{key: receipt[key] for key in keys},
            "evidence_sha256": B.digest(receipt),
        }

    def _compact_feedback(
        self, source: str, outer: int, before_slot: int
    ) -> dict[str, Any]:
        """Project the current parent's complete TRAIN receipt into deterministic feedback."""
        _, receipt, completed = self._parent_receipt(outer, source, before_slot)
        summary = M.project_training(
            self._training_projection(receipt),
            source_sha256=B.source_hash(source),
            completed_ns=completed,
            snapshot_ns=receipt["observed_ns"] + 1,
        )
        return {
            "schema": _CURRENT_SCHEMA,
            "current": {"source_sha256": B.source_hash(source), "train": summary},
        }

    def feedback(self, source: str, outer: int) -> dict[str, Any]:
        """Give production Trace the same compact current-parent summary in every arm."""
        return self._compact_feedback(source, outer, self._next_slot)

    def _attempt(self, outer: int, slot: int) -> dict[str, Any]:
        """Authenticate a completed same-arm response and its already allocated TRAIN rows."""
        folder = self._directory(outer) / f"slot_{slot:02d}"
        request, response = E.read(folder / "request.json"), E.read(
            folder / "response.json"
        )
        S._verify_response(request, response)
        if (
            request["outer"] != outer
            or request["arm"] != self.arm
            or request["slot"] != slot
        ):
            raise RuntimeError(
                "historical request belongs to another arm, seed or slot"
            )
        receipt = self._receipt(outer, slot)
        return {
            "arm": self.arm,
            "outer": outer,
            "slot": slot,
            "slot_id": request["slot_id"],
            "response_sha256": B.digest(response),
            "parent_sha256": request["parent_sha256"],
            "source": response["source"],
            "source_sha256": response["source_sha256"],
            "source_status": response["source_status"],
            "completed_ns": response["completed_ns"],
            "train": self._training_projection(receipt),
        }

    def _archive(self, outer: int, before_slot: int) -> list[TrainingCandidate]:
        """Return complete prior-slot instance vectors solely from authenticated TRAIN receipts."""
        records = []
        for index in range(-1, before_slot):
            receipt = self._receipt(outer, index)
            vector = receipt["auc_by_instance"]
            records.append(
                TrainingCandidate(
                    index,
                    receipt["source_sha256"],
                    None if vector is None else tuple(vector),
                )
            )
        return records

    def messages(
        self, outer: int, slot: int, parent: str, feedback: str | None
    ) -> list[dict[str, str]]:
        """Freeze or exactly reconstruct a causal prompt context before the model request."""
        if type(slot) is not int or not 0 <= slot < self.slots:
            raise ValueError("unregistered proposal slot")
        expected = self._compact_feedback(parent, outer, slot)
        if _decode_feedback(feedback) != expected:
            raise RuntimeError(
                "propagated TRAIN feedback does not match its actual parent"
            )
        directory = self._directory(outer)
        folder = directory / f"slot_{slot:02d}"
        path = folder / "current_context.json"
        if E.exists(path):
            snapshot_ns = E.read(path)["snapshot_ns"]
        else:
            if E.exists(folder / "request.json") or E.exists(folder / "response.json"):
                raise RuntimeError(
                    "completed request is missing its historical context"
                )
            snapshot_ns = time.time_ns()
        index, receipt, completed = self._parent_receipt(outer, parent, slot)
        current = M.project_training(
            self._training_projection(receipt),
            source_sha256=B.source_hash(parent),
            completed_ns=completed,
            snapshot_ns=snapshot_ns,
        )
        memory = None
        if self.arm in {"M", "PM"}:
            memory = M.build_memory(
                [self._attempt(outer, previous) for previous in range(slot)],
                arm=self.arm,
                outer=outer,
                before_slot=slot,
                snapshot_ns=snapshot_ns,
                current_parent_sha256=B.source_hash(parent),
                max_sources=self.config["memory_max_sources"],
                max_chars=self.config["memory_max_chars"],
            )
        if self.arm in {"P", "PM"}:
            decision = E.read(directory / f"parent_decisions/slot_{slot:02d}.json")
            replayed = select_pareto_parent(
                self._archive(outer, slot),
                namespace=self.config["namespace"],
                outer_seed=outer,
                next_slot=slot,
            )
            if (
                any(decision.get(key) != value for key, value in replayed.items())
                or not decision["will_generate"]
                or decision["selected_source_sha256"] != B.source_hash(parent)
            ):
                raise RuntimeError(
                    "actual Trace parent differs from its frozen Pareto decision"
                )
        context = {
            "schema": "exp18-request-context-v1",
            "arm": self.arm,
            "outer": outer,
            "slot": slot,
            "snapshot_ns": snapshot_ns,
            "current_source_sha256": B.source_hash(parent),
            "current_index": index,
            "current_receipt_sha256": B.digest(receipt),
            "current_training": current,
            "trace_feedback_sha256": B.source_hash(feedback),
            "memory": memory,
        }
        I.persist(path, context)
        content = "Improve the current optimizer.\nCURRENT SOURCE:\n" + parent
        content += "\nTRAINING FEEDBACK:\n" + feedback
        if memory is not None:
            content += "\nPRIOR ATTEMPT MEMORY:\n" + memory["text"]
        return [
            {"role": "user", "content": S._invariant(self.config)},
            {"role": "user", "content": content},
        ]

    def proposal_from_trace(
        self, outer: int, slot: int, parent: str, feedback: str
    ) -> dict[str, Any]:
        """Consume real propagated compact feedback, then count one completed update."""
        if slot != self._next_slot:
            raise RuntimeError(
                "production update count differs from the allocated slot"
            )
        if _decode_feedback(feedback) != self._compact_feedback(parent, outer, slot):
            raise RuntimeError(
                "propagated TRAIN feedback does not match its actual parent"
            )
        I.persist(
            self._directory(outer) / f"slot_{slot:02d}/propagated_feedback.json",
            {"text": feedback},
        )
        response = self._proposal(outer, slot, parent, feedback)
        self._next_slot += 1
        return response

    @contextmanager
    def _pareto_engine_binding(self, outer: int) -> Iterator[None]:
        """Scope a registered trainer resource to the unchanged production schedule.

        The frozen EXP-16 schedule binds its engine by a module-level callback.
        Generation is explicitly serialized; the adapter restores both callback
        and registry entry in finally, including when an execution fails.
        """
        if not _BINDING_LOCK.acquire(blocking=False):
            raise RuntimeError("concurrent Pareto engine bindings are not permitted")
        old_runner = E._trace_engine
        old_entry = control._ENGINE_REGISTRY.get("exp16_trace_v1")
        trainer_type = make_pareto_trainer(
            namespace=self.config["namespace"],
            outer_seed=outer,
            proposal_slots=self.slots,
            next_slot=lambda: self._next_slot,
            training_archive=lambda: self._archive(outer, self._next_slot),
            record_decision=lambda value: I.persist(
                self._directory(outer)
                / f"parent_decisions/slot_{value['next_slot']:02d}.json",
                value,
            ),
        )

        def engine(unit: Any, level: Any, resources: Any) -> Any:
            """Inject the narrow trainer while retaining the actual slot optimizer binding."""
            binding = E._TRAIN_CONTEXTS[level.datasets["train"][0]["context_id"]]
            with registered_pareto_trainer(trainer_type) as trainer_name:
                return control._run_module_engine(
                    unit,
                    level,
                    {
                        **resources,
                        "optimizer": binding["optimizer"],
                        "trainer": trainer_name,
                    },
                    fit=True,
                )

        try:
            if old_entry is not None and old_entry.run is not old_runner:
                raise RuntimeError(
                    "existing Trace engine registry has an unexpected owner"
                )
            E._trace_engine = engine
            control._ENGINE_REGISTRY.pop("exp16_trace_v1", None)
            yield
        finally:
            E._trace_engine = old_runner
            control._ENGINE_REGISTRY.pop("exp16_trace_v1", None)
            if old_entry is not None:
                control._ENGINE_REGISTRY["exp16_trace_v1"] = old_entry
            _BINDING_LOCK.release()

    def generate(self, outer: int) -> None:
        """Run the inherited production schedule with the registered treatment hooks."""
        self._next_slot = 0
        if self.arm in {"P", "PM"}:
            with self._pareto_engine_binding(outer):
                super().generate(outer)
        else:
            super().generate(outer)
