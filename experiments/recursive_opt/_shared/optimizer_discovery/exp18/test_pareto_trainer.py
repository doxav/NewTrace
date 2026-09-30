"""EXP-18 changes the real production parent hook without adding search steps."""

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.exp18.pareto_selection import TrainingCandidate
from experiments.recursive_opt._shared.optimizer_discovery.exp18.pareto_trainer import (
    make_pareto_trainer,
    registered_pareto_trainer,
)
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import trace_schedule as T
from opto import trace
from opto.features.recursive_opt import spec as control
from opto.trainer import algorithms
from opto.trainer.algorithms.priority_search import HeapMemory, ModuleCandidate


@trace.model
class SourceModule:
    """Provide real trainable source nodes for the parent-selection unit tests."""

    def __init__(self, source: str) -> None:
        """Create exactly the one trainable source expected by the adapter."""
        self.source = trace.node(source, trainable=True)

    def forward(self, value: Any) -> Any:
        """Expose source through a real Trace node without executing it."""
        return self.source


def fixture_trainer(
    *, invalid_missing: bool = False, next_slot: int = 1, slots: int = 4
) -> tuple[Any, list[dict[str, Any]], ModuleCandidate, ModuleCandidate]:
    """Build a heap containing real candidate objects and two tradeoff vectors."""
    first, second = "seed source", "specialist source"
    sources = [first, second]
    vectors = [(0.1, 0.2), None if invalid_missing else (0.0, 0.4)]
    records = [
        TrainingCandidate(index - 1, B.source_hash(source), vector)
        for index, (source, vector) in enumerate(zip(sources, vectors))
    ]
    decisions: list[dict[str, Any]] = []
    trainer_type = make_pareto_trainer(
        namespace="EXP18-UNIT",
        outer_seed=4,
        proposal_slots=slots,
        next_slot=lambda: next_slot,
        training_archive=lambda: records,
        record_decision=decisions.append,
    )
    trainer = object.__new__(trainer_type)
    trainer.num_candidates = trainer.num_proposals = 1
    trainer.score_function = "mean"
    trainer.memory_update_frequency = 0
    trainer.long_term_memory = HeapMemory()
    seed = ModuleCandidate(SourceModule(first))
    specialist = ModuleCandidate(SourceModule(second))
    trainer.long_term_memory.push(-0.15, seed)
    if not invalid_missing:
        trainer.long_term_memory.push(-0.20, specialist)
    return trainer, decisions, seed, specialist


def test_selects_actual_archived_candidate_without_popping_or_evaluating() -> None:
    """The returned object carries the actual source and Trace candidate identity."""
    trainer, decisions, seed, specialist = fixture_trainer()
    before = list(trainer.memory)
    chosen, priorities, info = trainer.explore()
    expected = {
        B.source_hash("seed source"): seed,
        B.source_hash("specialist source"): specialist,
    }
    assert chosen[0] is expected[decisions[0]["selected_source_sha256"]]
    assert priorities == [None]  # These initial unit candidates have no rollouts.
    assert list(trainer.memory) == before
    assert info["num_exploration_candidates"] == 1
    assert decisions[0]["will_generate"] is True
    assert decisions[0]["available_memory_source_count"] == 2
    trainer.explore()
    assert decisions[0] == decisions[1]


def test_invalid_archived_source_may_be_absent_from_production_memory() -> None:
    """Typed-invalid slots remain auditable without becoming selectable parents."""
    trainer, decisions, seed, _ = fixture_trainer(invalid_missing=True)
    selected, _, _ = trainer.explore()
    assert selected == [seed]
    assert decisions[0]["invalid_indices"] == [0]


def test_missing_valid_production_candidate_is_an_infrastructure_error() -> None:
    """Do not create a replacement module or change source behind Trace."""
    trainer, _, _, _ = fixture_trainer()
    trainer.memory.pop()
    with pytest.raises(RuntimeError, match="TRAIN-valid archive"):
        trainer.explore()


def test_terminal_parent_decision_is_explicitly_unused() -> None:
    """The trainer's final explore callback must not imply an extra proposal slot."""
    trainer, decisions, _, _ = fixture_trainer(slots=1)
    trainer.explore()
    assert decisions[0]["next_slot"] == 1
    assert decisions[0]["will_generate"] is False


@pytest.mark.parametrize("field,value", [("num_candidates", 2), ("num_proposals", 2)])
def test_width_or_proposal_multiplication_is_rejected(field: str, value: int) -> None:
    """The selector adapter cannot silently turn into a wider search procedure."""
    trainer, _, _, _ = fixture_trainer()
    setattr(trainer, field, value)
    with pytest.raises(RuntimeError, match="one parent and one proposal"):
        trainer.explore()


def test_unallocated_callback_fails() -> None:
    """A callback beyond the terminal slot violates the frozen allocation."""
    trainer, _, _, _ = fixture_trainer(next_slot=2, slots=1)
    with pytest.raises(RuntimeError, match="unallocated"):
        trainer.explore()


def test_scoped_registration_rejects_collision_and_removes_alias_on_failure() -> None:
    """Compatibility registration never replaces another trainer or survives an error."""
    trainer, _, _, _ = fixture_trainer()
    trainer_type = type(trainer)
    with (
        pytest.raises(RuntimeError, match="unit failure"),
        registered_pareto_trainer(trainer_type) as name,
    ):
        assert getattr(algorithms, name) is trainer_type
        with (
            pytest.raises(RuntimeError, match="already registered"),
            registered_pareto_trainer(trainer_type),
        ):
            pytest.fail("a registry collision must fail before yielding")
        assert getattr(algorithms, name) is trainer_type
        raise RuntimeError("unit failure")
    assert not hasattr(algorithms, name)
    with registered_pareto_trainer(trainer_type) as resumed_name:
        assert resumed_name == name
    assert not hasattr(algorithms, name)


def test_real_control_plane_trace_path_preserves_slots_and_cached_train_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real production search retains each response, including a typed invalid one."""
    monkeypatch.setitem(B.MANIFEST, "inner_budget", 2)
    monkeypatch.setitem(B.MANIFEST, "pilot_proposal_slots", 4)
    monkeypatch.setattr(
        control,
        "_ENGINE_REGISTRY",
        {
            key: entry
            for key, entry in control._ENGINE_REGISTRY.items()
            if key != "exp16_trace_v1"
        },
    )
    calls: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    consumed_callbacks = [0]

    def client(**kwargs: Any) -> Any:
        """Return fixed unit candidates; this is not live experimental evidence."""
        calls.append(kwargs)
        source = (
            "unparsable unit response"
            if len(calls) == 2
            else f"def propose(history, bounds, seed):\n    return [{len(calls) / 10}] * len(bounds)\n"
        )
        return SimpleNamespace(
            id=f"unit-pareto-{len(calls)}",
            model=B.MANIFEST["model"]["model"],
            usage={},
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=source), finish_reason="stop"
                )
            ],
        )

    class ReplayOwner(E.Experiment):
        """Count production updates independently of fresh or replayed model responses."""

        def proposal(
            self, outer: int, arm: str, slot: int, parent: str
        ) -> dict[str, Any]:
            """Advance only after the immutable proposal slot has completed."""
            result = super().proposal(outer, arm, slot, parent)
            consumed_callbacks[0] = slot + 1
            return result

    owner = ReplayOwner(tmp_path, "pilot", client=client)
    directory = tmp_path / "701" / "A2"

    def archived() -> Sequence[TrainingCandidate]:
        """Read already persisted TRAIN cache rows; never call the evaluator here."""
        records = []
        sources = [B.SEED_SOURCE] + [
            E.read(directory / f"slot_{slot:02d}" / "response.json")["source"]
            for slot in range(consumed_callbacks[0])
        ]
        tasks = B.make_tasks("pilot", "train")
        for index, source in enumerate(sources):
            cached = [
                E.read(path)
                for path in (tmp_path / "cache").glob("*.json")
                if E.read(path)["key"]["source_sha256"] == B.source_hash(source)
            ]
            by_identity = {
                entry["key"]["task_identity"]: entry["result"] for entry in cached
            }
            assert set(by_identity) == {B.task_identity(task) for task in tasks}
            rows = [by_identity[B.task_identity(task)] for task in tasks]
            values = (
                tuple(row["metrics"]["auc"] for row in rows)
                if all(row["valid"] for row in rows)
                else None
            )
            records.append(TrainingCandidate(index - 1, B.source_hash(source), values))
        return records

    trainer_type = make_pareto_trainer(
        namespace="EXP18-PRODUCTION-UNIT",
        outer_seed=701,
        proposal_slots=owner.slots,
        next_slot=lambda: consumed_callbacks[0],
        training_archive=archived,
        record_decision=decisions.append,
    )

    def engine(unit: Any, level: Any, resources: Any) -> Any:
        """Inject only the trainer hook into the existing canonical module engine."""
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

    monkeypatch.setattr(E, "_trace_engine", engine)
    T.generate_recursive(owner, 701)
    assert len(calls) == 4
    assert len(decisions) == 5
    assert sum(row["will_generate"] for row in decisions) == 4
    assert decisions[-1]["invalid_indices"] == [1]
    saved = E.read(directory / "trace.json")
    assert saved["completed_update_callbacks"] == 4
    assert saved["result"]["level_results"][0]["metadata"]["trace_optimize_path"]
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    updates = [row for row in events if row["event"] == "trace_update"]
    assert [row["parent_sha256"] for row in updates] == [
        row["selected_source_sha256"] for row in decisions[:-1]
    ]
    rows = [E.read(path)["result"] for path in (tmp_path / "cache").glob("*.json")]
    assert len(rows) == 5 * 6
    assert sum(row["objective_calls"] for row in rows) == 4 * 6 * 2
    assert all(
        row["key"]["split"] == "train" for row in events if row["event"] == "evaluation"
    )
    before = len(list((tmp_path / "cache").glob("*.json")))
    T.generate_recursive(owner, 701)
    assert len(calls) == 4
    assert len(list((tmp_path / "cache").glob("*.json"))) == before
    # Simulate a crash after responses were saved but before final Trace export.
    # Deletion is restricted to this test's disposable evidence directory.
    (directory / "trace.json").unlink()
    (directory / "generation_complete.json").unlink()
    consumed_callbacks[0] = 0
    original_decisions = list(decisions)
    T.generate_recursive(owner, 701)
    assert len(calls) == 4
    assert decisions[5:] == original_decisions
    replay_events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ][len(events) :]
    assert sum(row["event"] == "trace_update_replay" for row in replay_events) == 4
    assert all(
        row["cache_hit"] for row in replay_events if row["event"] == "evaluation"
    )
