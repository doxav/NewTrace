from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt.multiobjective_reasoning import main_experiment


def _fake_run(
    arm: str,
    seed: int,
    budget: int,
    *,
    accuracy: float,
    token_ratio: float,
) -> dict[str, Any]:
    """Build one deterministic synthetic main result."""
    optimized = arm != "A"
    checks = {
        "run_succeeded": True,
        "exact_model_available": True,
        "reasoning_parameters_recorded": True,
        "one_workflow_forward_per_evaluator": True,
        "evaluator_received_exact_output": True,
        "usage_populated": True,
        "forward_calls_reconciled": True,
        "forward_tokens_reconciled": True,
        "holdout_inaccessible_during_optimization": True,
        "source_digest_stable": True,
        "experiment_source_digest_stable": True,
        "artifact_reloadable": True,
        "proposal_path_exercised": True,
        "selection_changed": optimized,
        "no_environment_override": True,
        "cache_not_shared": True,
        "output_persistence_and_resume": True,
    }
    return {
        "arm": arm,
        "seed": seed,
        "proposal_budget": budget,
        "passed": True,
        "valid": True,
        "error": None,
        "metrics": {
            "accuracy": accuracy,
            "invalid_rate": 0.0,
            "forward_token_ratio": token_ratio,
            "latency_s": 1.0,
        },
        "usage": {
            "forward": {
                "calls": 2,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "optimizer": {
                "calls": int(optimized),
                "prompt_tokens": 20 if optimized else 0,
                "completion_tokens": 10 if optimized else 0,
                "total_tokens": 30 if optimized else 0,
            },
        },
        "cost_usd": {"forward": None, "optimizer": None},
        "accounted": {
            "candidates_reserved": int(optimized),
            "candidates_proposed": int(optimized),
            "candidates_evaluated": int(optimized),
            "evaluator_runs": 1,
            "wall_time_s": 1.0,
        },
        "workflow_forwards": 1,
        "artifact": {
            "analysis_instruction": f"analysis-{arm}" if optimized else "analysis",
            "answer_instruction": "answer",
        },
        "runtime_tree_sha256": "r" * 64,
        "experiment_source_sha256": "e" * 64,
        "plan_registry_sha256": "p" * 64,
        "checks": checks,
        "normalized_spec": {"outputs": {"directory": "/does/not/exist"}},
    }


def _synthetic_main_runs() -> list[dict[str, Any]]:
    """Build a complete forty-unit synthetic matrix for analysis tests."""
    values = {
        "A": (0.8, 1.0),
        "B": (0.9, 0.9),
        "C": (0.8, 0.5),
        "D": (0.7, 0.8),
    }
    return [
        _fake_run(
            arm,
            seed,
            budget,
            accuracy=values[arm][0],
            token_ratio=values[arm][1],
        )
        for seed in range(5)
        for budget in (6, 12)
        for arm in "ABCD"
    ]


def test_frozen_main_matrix_is_complete_and_valid() -> None:
    """The frozen main design expands to forty unique paired units."""
    frozen = main_experiment.validate_frozen_preregistration()
    matrix = main_experiment.main_execution_matrix(frozen)

    assert len(matrix) == 40
    assert len(set(matrix)) == 40
    assert {seed for seed, _, _ in matrix} == set(range(5))
    assert {budget for _, budget, _ in matrix} == {6, 12}
    assert all(
        {arm for run_seed, run_budget, arm in matrix if (run_seed, run_budget) == pair}
        == set("ABCD")
        for pair in {(seed, budget) for seed, budget, _ in matrix}
    )


def test_frozen_main_rejects_missing_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A numeric-ceiling waiver must still be an explicit authorization."""
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "authorized": False,
                "numeric_ceiling_waived": False,
                "scientific_protocol_changed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main_experiment, "MAIN_AUTHORIZATION_PATH", authorization)

    with pytest.raises(RuntimeError, match="monetary authorization"):
        main_experiment.validate_frozen_preregistration()


def test_main_statistics_are_paired_deterministic_and_directional() -> None:
    """Paired bootstrap statistics preserve quality and efficiency direction."""
    report = {
        "passed": True,
        "task": "gsm8k",
        "runs": _synthetic_main_runs(),
        "retry_statistics_by_arm": {},
    }

    first = main_experiment.analyze_main_experiment(report)
    second = main_experiment.analyze_main_experiment(report)

    assert first == second
    assert first["paired_comparisons"]["B-A"]["paired_blocks"] == 10
    assert first["paired_comparisons"]["B-A"]["quality_success"] is True
    assert first["paired_comparisons"]["C-A"]["quality_success"] is False
    assert first["paired_comparisons"]["C-A"]["efficiency_success"] is True
    assert first["safety_passed"] is True


def test_episode_audit_rejects_missing_candidate_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing proposal lineage blocks episode export without inference."""
    runs = [run for run in _synthetic_main_runs() if run["arm"] in {"B", "C"}]
    monkeypatch.setattr(
        main_experiment,
        "_persisted_level_result",
        lambda run: (
            main_experiment.REPOSITORY_ROOT / "result.json",
            {"metadata": {"evaluator_records": []}},
        ),
    )

    audit = main_experiment.audit_candidate_trajectories({"runs": runs})

    assert audit["ready_for_episode_export"] is False
    assert audit["episodes_exported"] is False
    assert "candidate_trajectory" in audit["smallest_required_extension"]


def test_episode_audit_accepts_complete_candidate_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete proposal lineage satisfies the future episode gate."""
    runs = [run for run in _synthetic_main_runs() if run["arm"] in {"B", "C"}]
    trajectory = [
        {
            "artifact_sha256": "a" * 64,
            "seed_relation": "P0",
            "evaluation": {"valid": True},
            "status": "selected",
        }
    ]
    monkeypatch.setattr(
        main_experiment,
        "_persisted_level_result",
        lambda run: (
            main_experiment.REPOSITORY_ROOT / "result.json",
            {"metadata": {"candidate_trajectory": trajectory}},
        ),
    )

    audit = main_experiment.audit_candidate_trajectories({"runs": runs})

    assert audit["ready_for_episode_export"] is True
    assert audit["smallest_required_extension"] is None


def test_main_runner_checkpoints_all_frozen_units(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The main runner checkpoints every frozen matrix unit exactly once."""
    frozen = main_experiment.validate_frozen_preregistration()
    source_sha = "e" * 64
    runtime_sha = "r" * 64
    lock = {
        "control_plane": {"runtime_tree_sha256": runtime_sha},
        "experiment": {"source": {"sha256": source_sha}},
    }
    calls: list[tuple[int, int, str]] = []

    def execute(**kwargs: Any) -> dict[str, Any]:
        """Return a passing synthetic arm while retaining execution order."""
        calls.append((kwargs["seed"], kwargs["proposals"], kwargs["arm"]))
        return _fake_run(
            kwargs["arm"],
            kwargs["seed"],
            kwargs["proposals"],
            accuracy=0.8,
            token_ratio=0.8,
        )

    report_path = tmp_path / "main.json"
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    preregistration_path = tmp_path / "preregistration.json"
    preregistration_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main_experiment, "validate_frozen_preregistration", lambda: frozen)
    monkeypatch.setattr(main_experiment, "_load_main_lock", lambda value: lock)
    monkeypatch.setattr(main_experiment, "_snapshot_output_context", lambda: None)
    monkeypatch.setattr(main_experiment, "_execute_arm", execute)
    monkeypatch.setattr(main_experiment, "MAIN_REPORT_PATH", report_path)
    monkeypatch.setattr(main_experiment, "MAIN_RUNS_DIRECTORY", tmp_path / "runs")
    monkeypatch.setattr(main_experiment, "MAIN_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        main_experiment,
        "FROZEN_PREREGISTRATION_PATH",
        preregistration_path,
    )

    result = main_experiment.run_main_experiment()

    assert result["passed"] is True
    assert result["completed_run_count"] == 40
    assert calls == main_experiment.main_execution_matrix(frozen)
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
