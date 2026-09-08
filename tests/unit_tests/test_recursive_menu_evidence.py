"""Automatic menu evidence must describe actual, comparable evaluations."""

from typing import Any

import pytest

from opto.features.recursive_opt.measurement import menu_evidence


def observation(
    code: str,
    behavior: Any = None,
    *,
    valid: bool = True,
    score: float = 1.0,
    example: str = "x",
    phase: str = "fit",
) -> dict[str, Any]:
    """Build one observed candidate evaluation, without source-based equivalence."""
    result = {
        "candidate": {"text": code},
        "example": example,
        "phase": phase,
        "valid": valid,
        "metrics": {"score": score} if valid else {},
    }
    if behavior is not None:
        result["behavior_signature"] = behavior
    return result


@pytest.mark.parametrize(
    "candidates",
    [("def a(): return 1", "def b(): return 1"), ("Answer directly.", "Be concise.")],
)
def test_equivalent_code_and_prose(candidates: tuple[str, str]) -> None:
    """Byte differences never establish distinct evaluated behavior."""
    result = menu_evidence(
        [observation(c, [0, 1]) for c in candidates], declared_menu_size=2
    )
    assert result["effective_menu_size"] == 1 and result["menu_collapsed"]
    assert result["basis_of_equivalence"] == "behavior_signature"


def test_ties_distinct_behavior_duplicates_invalid() -> None:
    """Score ties can hide distinct behavior; invalid programs have no score."""
    rows = [
        observation("a", [0, 1]),
        observation("a", [0, 1]),
        observation("b", [1, 0]),
        observation("broken", valid=False),
    ]
    result = menu_evidence(rows, declared_menu_size=4)
    assert result["declared_menu_size"] == 4
    assert result["evaluated_candidate_count"] == 3
    assert result["evaluation_observation_count"] == 4
    assert result["valid_candidate_count"] == 2
    assert result["effective_menu_size"] == 2
    assert result["menu_collapsed"] is False


def test_metric_only_is_explicit_and_holdout_is_excluded() -> None:
    """A scalar-only evaluator cannot certify behavioral headroom."""
    result = menu_evidence(
        [
            observation("a"),
            observation("b"),
            observation("b", [9], score=9, phase="final_evaluation"),
        ]
    )
    assert result["basis_of_equivalence"] == "metric_vector"
    assert result["effective_menu_size"] == 1
    assert result["behavior_equivalence_known"] is False
    assert result["evaluation_observation_count"] == 2


def test_incomparable_or_stochastic_evidence_is_unknown() -> None:
    """Do not compare different inputs or call random output differences headroom."""
    rows = [observation("a", [0], example="a"), observation("b", [1], example="b")]
    assert menu_evidence(rows)["effective_menu_size"] is None
    rows = [observation("a", [0]), observation("a", [1]), observation("b", [1])]
    result = menu_evidence(rows)
    assert result["effective_menu_size"] is None
    assert result["menu_collapsed"] is None
    assert result["basis_of_equivalence"] == "observed_stochasticity"


def test_all_invalid_and_empty() -> None:
    """Absence of valid candidates differs from absence of observations."""
    assert menu_evidence([observation("bad", valid=False)])["effective_menu_size"] == 0
    assert menu_evidence([])["effective_menu_size"] is None


def test_canonical_records_actual_candidates_and_persists(tmp_path: Any) -> None:
    """Real Trace proposals retain behavior evidence even when scalar scores tie."""
    from opto.features.recursive_opt import spec as S
    from opto.optimizers.optimizer import Optimizer
    from opto.trainer.objectives import EvaluationResult

    class Change(Optimizer):
        """Change the actual trainable prose through the real trainer."""

        def _step(self, *args: Any, **kwargs: Any) -> dict[Any, str]:
            """Return a distinct deterministic response."""
            return {p: "b" for p in self.parameters}

    def evaluator(output: Any, example: Any, context: Any) -> EvaluationResult:
        """A tied reward with distinct actual choices."""
        return EvaluationResult(
            valid=True,
            status="ok",
            metrics={"score": 1.0},
            artifacts={"behavior_signature": output.data["components"]["answer"]},
        )

    S.register_evaluator("phase0.menu_test@1", evaluator)
    raw = {
        "schema_version": S.SCHEMA_VERSION,
        "kind": S.SPEC_KIND,
        "runtime": {"offline": True, "test_mode": True},
        "outputs": {"directory": str(tmp_path)},
        "module": {
            "ref": "recursive_opt.module.reasoning_workflow@1",
            "config": {"components": {"answer": "a"}},
        },
        "surface": {"kind": "module", "targets": ["answer"]},
        "engine": {"name": "trace", "config": {"iterations": 2, "num_candidates": 1}},
        "objective": {
            "evaluator_ref": "phase0.menu_test@1",
            "metrics": {
                "score": {"direction": "maximize", "source": "evaluation.metrics.score"}
            },
            "selection": {"mode": "scalar", "score_key": "score"},
        },
        "datasets": {
            "train": ["shared"],
            "validation": ["shared"],
            "holdout": ["secret"],
        },
    }
    result = S.execute_plan(S.compile_plan(raw), {"optimizer": Change})[0]
    evidence = result.metadata["menu_evidence"]
    assert evidence["effective_menu_size"] == 2
    assert evidence["basis_of_equivalence"] == "behavior_signature"
    records = result.metadata["menu_observations"]
    assert {
        row["candidate"]["components"]["answer"]
        for row in records
        if row["phase"] == "fit"
    } == {"a", "b"}
    assert all(row["example"] != "secret" for row in records)
    raw["runtime"]["resume"] = True
    resumed = S.execute_plan(S.compile_plan(raw), {"optimizer": Change})[0]
    assert resumed.metadata["menu_evidence"] == evidence


def test_invalid_aggregation_has_no_numeric_metric() -> None:
    """Missing invalid metrics must not crash aggregation or become a penalty mean."""
    from opto.features.recursive_opt import spec as S
    from opto.trainer.objectives import EvaluationResult

    objective = S.compile_objective(
        {
            "metrics": {
                "score": {"direction": "maximize", "source": "evaluation.metrics.score"}
            },
            "selection": {"mode": "scalar", "score_key": "score"},
        },
        capabilities=S._engine_entry("fixed").capabilities,
    )
    result = S._aggregate_evaluations(
        [
            EvaluationResult(valid=True, status="ok", metrics={"score": 4}),
            EvaluationResult(valid=False, status="invalid", error="shape_error"),
        ],
        objective,
    )
    assert not result.valid and not result.metrics
    assert result.error == "shape_error"


def test_legacy_rejection_is_typed_and_not_counted_as_legal_floor() -> None:
    """Normalizer floor and final priors must not inflate a legacy candidate menu."""
    from types import SimpleNamespace

    from opto.features.recursive_opt import spec as S
    from opto.features.recursive_opt.levels import (
        LevelConfig,
        MetaLevel,
        invalid_result,
    )

    module = MetaLevel(LevelConfig(), lambda cfg, task: (1.0, "ok"))
    rejected = 'SCORE_NORMALIZATION_JSON={"invalid": true, "score": -1.0}'
    rollouts = [
        {
            "x": "shared",
            "target": {"score": -1, "feedback": rejected},
            "score": -1,
            "feedback": rejected,
        }
    ]
    trainer = SimpleNamespace(
        memory=SimpleNamespace(
            memory=[(0, SimpleNamespace(get_module=lambda: module, rollouts=rollouts))]
        )
    )
    rows = S._legacy_menu_observations(trainer, {"surface": "config"})
    assert rows[0]["valid"] is False and rows[0]["metrics"] == {}
    assert menu_evidence(rows)["effective_menu_size"] == 0
    assert invalid_result("bad config", floor=-1)["valid"] is False
