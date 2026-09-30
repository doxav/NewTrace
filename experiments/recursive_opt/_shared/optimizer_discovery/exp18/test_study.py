"""The four EXP-18 information/selection treatments use actual production Trace."""

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import study as N
from experiments.recursive_opt._shared.optimizer_discovery.exp18.study import MechanismStudy
from opto.features.recursive_opt import spec as control
from tests.unit_tests.test_investigation16_search_experiment import _client, _prepared


def prepared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    slots: int = 2,
    arms: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Reuse the established synthetic evaluator and freeze a separate unit namespace."""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _, state = _prepared(fixture, monkeypatch)
    config = N.configuration(
        experiment="EXP-18",
        namespace="EXP18-MECHANISMS-UNIT",
        arms=arms or ["L", "M", "P", "PM"],
        outer_seeds=[18011],
        slots=slots,
        budget=2,
        local_replicates=1,
        task_replicates={"train": 1, "validation": 1, "audit": 1},
        workers=2,
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("UNIT ONLY. No live model responses.\n")
    root = tmp_path / "run"
    N.prepare(root, config, protocol)
    return root, state


def test_all_four_arms_real_trace_equal_slots_and_hidden_information(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each real update consumes compact propagated feedback and exactly one response."""
    root, state = prepared(tmp_path, monkeypatch)
    state["invalid_at"] = 2  # First M response; it must appear in M's next memory.
    original_engine = E._trace_engine
    N.run_generation(root, client=_client(state))
    assert E._trace_engine is original_engine
    assert control._ENGINE_REGISTRY["exp16_trace_v1"].run is original_engine
    assert len(state["calls"]) == 8
    for arm in ("L", "M", "P", "PM"):
        directory = root / "raw" / "18011" / arm
        assert E.read(directory / "trace.json")["completed_update_callbacks"] == 2
        assert E.read(directory / "allocations_train.json")["candidate_slots"] == 3
        for slot in range(2):
            folder = directory / f"slot_{slot:02d}"
            request = E.read(folder / "request.json")
            context = E.read(folder / "current_context.json")
            feedback = E.read(folder / "propagated_feedback.json")["text"]
            text = "\n".join(message["content"] for message in request["messages"])
            assert feedback in text
            assert "exp18-current-training-v1" in text
            assert "aggregate_auc" in text
            assert all(
                forbidden not in text
                for forbidden in (
                    "HOST_ONLY_SENTINEL",
                    "auc_by_instance",
                    "row_hashes",
                    "snapshot_ns",
                )
            )
            assert (context["memory"] is not None) == (arm in ("M", "PM"))
            assert ("exp18-attempt-memory-v1" in text) == (arm in ("M", "PM"))
            if arm in ("P", "PM"):
                decision = E.read(directory / f"parent_decisions/slot_{slot:02d}.json")
                assert request["parent_sha256"] == decision["selected_source_sha256"]
                assert decision["will_generate"]
        if arm in ("P", "PM"):
            assert not E.read(directory / "parent_decisions/slot_02.json")[
                "will_generate"
            ]
    memory = E.read(root / "raw/18011/M/slot_01/current_context.json")["memory"]
    assert memory["counts"]["no_source_slots"] == 1
    assert json.loads(memory["text"])["attempts"][0]["train"]["aggregate_auc"] is None
    calls, evaluations = len(state["calls"]), len(state["evaluations"])
    N.run_generation(root, client=None)
    assert len(state["calls"]) == calls
    assert len(state["evaluations"]) == evaluations
    N.select_all(root)
    result = N.run_audit(root)
    assert set(result["per_seed"]["18011"]) == {"L", "M", "P", "PM", "A0", "B2"}
    assert N.verify_chronology(root)["completed_responses"] == 8


@pytest.mark.parametrize("arm", ["M", "P", "PM"])
def test_rebuilt_trace_replays_frozen_contexts_without_future_leak_or_new_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """Disk already holds later responses; replay still uses each original cutoff."""
    root, state = prepared(tmp_path, monkeypatch, slots=4, arms=[arm])
    owner = MechanismStudy(root, arm, client=_client(state))
    owner.generate(18011)
    directory = root / "raw" / "18011" / arm
    before = {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*.json")
        if path.name in {"current_context.json", "request.json", "response.json"}
        or path.parent.name == "parent_decisions"
    }
    calls, evaluations = len(state["calls"]), len(state["evaluations"])
    (directory / "trace.json").unlink()
    (directory / "generation_complete.json").unlink()
    MechanismStudy(root, arm, client=None).generate(18011)
    assert len(state["calls"]) == calls
    assert len(state["evaluations"]) == evaluations
    assert {
        relative: (directory / relative).read_bytes() for relative in before
    } == before
    first = E.read(directory / "slot_00/current_context.json")
    if arm in ("M", "PM"):
        assert first["memory"]["counts"]["prior_slots"] == 0


def test_missing_context_after_response_is_not_reconstructed_post_hoc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing historical snapshot is a defect rather than permission to use later data."""
    root, state = prepared(tmp_path, monkeypatch, arms=["M"])
    owner = MechanismStudy(root, "M", client=_client(state))
    owner.generate(18011)
    folder = root / "raw/18011/M/slot_00"
    request = E.read(folder / "request.json")
    feedback = E.read(folder / "propagated_feedback.json")["text"]
    (folder / "current_context.json").unlink()
    with pytest.raises(RuntimeError, match="historical context"):
        owner.messages(18011, 0, B.SEED_SOURCE, feedback)
    assert request["parent_sha256"] == B.source_hash(B.SEED_SOURCE)


def test_wrong_trace_parent_cannot_reach_the_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrupted parent feedback fails before model generation."""
    root, state = prepared(tmp_path, monkeypatch, arms=["L"])
    owner = MechanismStudy(root, "L", client=_client(state))
    owner.training_receipt(18011, -1)
    feedback = "ID [0]: " + json.dumps(owner.feedback(B.SEED_SOURCE, 18011))
    wrong = feedback.replace(B.source_hash(B.SEED_SOURCE), "0" * 64)
    with pytest.raises(RuntimeError, match="propagated"):
        owner.proposal_from_trace(18011, 0, B.SEED_SOURCE, wrong)
    assert not state["calls"]


def test_future_receipt_cannot_reconstruct_an_earlier_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt available only after the request is forbidden during replay."""
    root, state = prepared(tmp_path, monkeypatch, arms=["M"])
    owner = MechanismStudy(root, "M", client=_client(state))
    owner.generate(18011)
    directory = root / "raw/18011/M"
    context = E.read(directory / "slot_00/current_context.json")
    feedback = E.read(directory / "slot_00/propagated_feedback.json")["text"]
    receipt = E.read(directory / "seed_train_receipt.json")
    receipt["observed_ns"] = context["snapshot_ns"] + 1
    (directory / "seed_train_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    calls = len(state["calls"])
    with pytest.raises(ValueError, match="between response and snapshot"):
        owner.messages(18011, 0, B.SEED_SOURCE, feedback)
    assert len(state["calls"]) == calls


def test_historical_context_cannot_fill_a_missing_train_cache_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Context authentication is read-only even when an earlier cache row is missing."""
    root, state = prepared(tmp_path, monkeypatch, arms=["M"])
    owner = MechanismStudy(root, "M", client=_client(state))
    owner.generate(18011)
    folder = root / "raw/18011/M/slot_00"
    feedback = E.read(folder / "propagated_feedback.json")["text"]
    cached = next(
        path
        for path in (root / "cache").glob("*.json")
        if E.read(path)["key"]["source_sha256"] == B.source_hash(B.SEED_SOURCE)
    )
    cached.unlink()
    evaluations = len(state["evaluations"])
    with pytest.raises(RuntimeError, match="read-only cache miss"):
        owner.messages(18011, 0, B.SEED_SOURCE, feedback)
    assert len(state["evaluations"]) == evaluations
    assert not cached.exists()


def test_pareto_scoped_engine_cleanup_after_failure_and_reentry_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine compatibility binding cannot leak across later scalar or Pareto arms."""
    root, _ = prepared(tmp_path, monkeypatch, arms=["P"])
    owner = MechanismStudy(root, "P")
    original_runner = E._trace_engine
    original_entry = control._ENGINE_REGISTRY.get("exp16_trace_v1")
    with (
        pytest.raises(RuntimeError, match="unit engine failure"),
        owner._pareto_engine_binding(18011),
    ):
        assert E._trace_engine is not original_runner
        with (
            pytest.raises(RuntimeError, match="concurrent"),
            owner._pareto_engine_binding(18011),
        ):
            pytest.fail("nested binding must fail before execution")
        raise RuntimeError("unit engine failure")
    assert E._trace_engine is original_runner
    assert control._ENGINE_REGISTRY.get("exp16_trace_v1") == original_entry
    with owner._pareto_engine_binding(18011):
        assert E._trace_engine is not original_runner


def test_pareto_specialist_is_the_actual_non_scalar_parent_at_trace_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberately conflicting TRAIN fixture exercises the frontier rather than a label."""
    root, state = prepared(tmp_path, monkeypatch, slots=8, arms=["P"])
    original = B.evaluate

    def tradeoff(
        source: str, task: dict[str, Any], seed: int, **kwargs: Any
    ) -> dict[str, Any]:
        """Give generated specialists a better dimension-2 and worse dimension-4 score."""
        row = original(source, task, seed, **kwargs)
        if source and source != B.SEED_SOURCE:
            value = 0.0 if task["dimension"] == 2 else 3.0
            row["observations"] = [
                {"x": [0.0] * task["dimension"], "value": value}
            ] * kwargs["budget"]
            row["metrics"] = B.metrics(
                [value] * kwargs["budget"], 1.0, kwargs["budget"]
            )
        return row

    monkeypatch.setattr(B, "evaluate", tradeoff)
    owner = MechanismStudy(root, "P", client=_client(state))
    owner.generate(18011)
    directory = root / "raw/18011/P"
    decisions = [
        E.read(directory / f"parent_decisions/slot_{slot:02d}.json")
        for slot in range(8)
    ]
    assert any(len(row["frontier_indices"]) > 1 for row in decisions)
    assert any(
        row["selected_source_sha256"] != row["scalar_best_source_sha256"]
        for row in decisions
    )
    events = [E.read(path) for path in (root / "events").glob("*.json")]
    updates = sorted(
        (row for row in events if row["event"] == "trace_update"),
        key=lambda row: row["slot"],
    )
    assert [row["parent_sha256"] for row in updates] == [
        row["selected_source_sha256"] for row in decisions
    ]
    assert len(state["calls"]) == 8
