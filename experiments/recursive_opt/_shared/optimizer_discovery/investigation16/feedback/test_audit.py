"""Offline mechanism checks for the frozen EXP-15 feedback/search interface."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.feedback import audit
from opto.trainer.algorithms.priority_search import PrioritySearch


def trajectory(values: list[float]) -> dict[str, Any]:
    """Build valid observations on the same one-dimensional quadratic x squared."""
    return {
        "valid": True,
        "status": "valid",
        "observations": [{"x": [value**0.5], "value": value} for value in values],
    }


def raw_auc(values: list[float]) -> float:
    """Compute unnormalized best-so-far AUC for the diagnostic quadratic."""
    return sum(min(values[: index + 1]) for index in range(len(values))) / len(values)


def test_actual_feedback_loses_anytime_progress() -> None:
    """A fifteen-fold AUC difference can map to exactly the same actual feedback."""
    early = [100.0, 100.0] + [0.0] * 30
    late = [100.0] * 30 + [0.0, 0.0]
    early_owner = SimpleNamespace(panel=lambda *args: [trajectory(early)])
    late_owner = SimpleNamespace(panel=lambda *args: [trajectory(late)])
    early_feedback = E.Experiment.feedback(early_owner, "unused", 0)
    late_feedback = E.Experiment.feedback(late_owner, "unused", 0)
    assert early_feedback == late_feedback
    assert early_feedback == audit.feedback_from_rows([trajectory(early)])
    assert raw_auc(early) == 6.25
    assert raw_auc(late) == 93.75
    assert raw_auc(late) / raw_auc(early) == 15


def test_actual_feedback_retains_typed_failures_without_fake_values() -> None:
    """An empty failed trajectory stays invalid with no invented best objective."""
    failed = {"valid": False, "status": "timeout", "observations": []}
    owner = SimpleNamespace(panel=lambda *args: [failed])
    projected = E.Experiment.feedback(owner, "unused", 0)
    assert projected == audit.feedback_from_rows([failed])
    assert projected["tasks"][0]["best_observed_value"] is None
    assert projected["tasks"][0]["status"] == "timeout"


def test_propagated_trace_feedback_changes_do_not_reach_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The EXP-15 slot adapter rebuilds observations instead of consuming Trace text."""
    original_evaluator = E._training_evaluator
    injected_evaluations: list[str] = []
    requests: list[dict[str, Any]] = []

    def evaluator(output: Any, example: Any, context: Any) -> Any:
        """Inject a trace-only signal into the real versioned evaluator channel."""
        original = original_evaluator(output, example, context)
        injected_evaluations.append("TRACE_ONLY_SENTINEL")
        return replace(original, feedback="TRACE_ONLY_SENTINEL")

    def client(**kwargs: Any) -> Any:
        """Return unchanged code for an offline mechanism test, never live evidence."""
        requests.append(kwargs)
        return SimpleNamespace(
            id="offline-mechanism",
            model=B.MANIFEST["model"]["model"],
            usage={},
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=B.SEED_SOURCE), finish_reason="stop"
                )
            ],
        )

    monkeypatch.setattr(E, "_training_evaluator", evaluator)
    monkeypatch.setattr(
        E.control,
        "_EVALUATOR_REGISTRY",
        {
            reference: entry
            for reference, entry in E.control._EVALUATOR_REGISTRY.items()
            if reference != "recursive_opt.evaluator.exp15_training@1"
        },
    )
    monkeypatch.setattr(E, "_TRAIN_CONTEXTS", dict(E._TRAIN_CONTEXTS))
    monkeypatch.setitem(B.MANIFEST, "inner_budget", 2)
    experiment = E.Experiment(tmp_path, "pilot", client=client)
    experiment.generate(701, "A2")
    assert injected_evaluations and len(requests) == 2
    for request in requests:
        prompt = "\n".join(message["content"] for message in request["messages"])
        assert "TRACE_ONLY_SENTINEL" not in prompt
        assert "TRAINING FEEDBACK" in prompt
        assert "best_observed_value" in prompt


def candidate(score: float) -> Any:
    """Provide the minimal candidate protocol used by production exploration."""
    return SimpleNamespace(mean_score=lambda: score, num_rollouts=1)


@pytest.mark.parametrize("memory_score", [0.1, 100.0])
def test_single_incumbent_exploration_cannot_branch(memory_score: float) -> None:
    """With EXP-15 settings even changed archive priorities cannot change the parent."""
    incumbent = candidate(0.5)
    alternative = candidate(memory_score)
    owner = SimpleNamespace(
        memory=[(-memory_score, alternative)],
        num_candidates=1,
        _best_candidate=incumbent,
        _best_candidate_priority=0.5,
        use_best_candidate_to_explore=True,
    )
    parents, priorities, _ = PrioritySearch.explore(owner)
    assert parents == [incumbent] and priorities == [0.5]
    assert len(owner.memory) == 1


def test_two_candidate_exploration_can_branch_through_production_path() -> None:
    """Changing the declared exploration width enables an actual archived parent."""
    incumbent, alternative = candidate(0.5), candidate(0.4)
    owner = SimpleNamespace(
        memory=[(-0.4, alternative)],
        num_candidates=2,
        _best_candidate=incumbent,
        _best_candidate_priority=0.5,
        use_best_candidate_to_explore=True,
    )
    parents, _, _ = PrioritySearch.explore(owner)
    assert parents == [incumbent, alternative]
    assert len(owner.memory) == 0


def test_rank_correlation_handles_ties_and_constants() -> None:
    """Descriptive ranking statistics do not invent variation for duplicate programs."""
    assert audit.ranks([3.0, 1.0, 1.0]) == [3.0, 1.5, 1.5]
    assert audit.correlation([1.0, 1.0], [1.0, 2.0]) is None
    with pytest.raises(ValueError, match="matching"):
        audit.correlation([1.0], [])


def test_frozen_all_requests_are_reconstructible_without_truncation() -> None:
    """The retrospective audit covers all slots and verifies exact feedback bytes."""
    result = audit.analyze()
    assert result["summary"]["requests"] == 80
    assert result["summary"]["a2_feedback_truncated"] == 0
    assert result["summary"]["prompt_mentions_auc_anytime_or_regret"] == 0
    assert len(result["selection_comparisons"]) == 10
    a2 = [row for row in result["requests"] if row["arm"] == "A2"]
    assert all(row["current_tasks"] == 6 for row in a2)
    assert all(row["current_observations_available"] == 192 for row in a2)
    assert all(row["current_observations_exposed"] == 24 for row in a2)
