from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt.multiobjective_reasoning import main_experiment
from experiments.recursive_opt.multiobjective_reasoning.offline_contract import (
    _FakeClient,
    _Factory,
)
from experiments.recursive_opt.multiobjective_reasoning.registration import (
    register_experiment_components,
)
from experiments.recursive_opt.multiobjective_reasoning.specs import build_spec
from opto.features.recursive_opt import spec as control_plane


def _fake_run(
    arm: str,
    seed: int,
    budget: int,
    *,
    accuracy: float,
    token_ratio: float,
    invalid_rate: float = 0.0,
    infrastructure_passed: bool = True,
) -> dict[str, Any]:
    """Build one deterministic synthetic main result."""
    optimized = arm != "A"
    checks = {
        "execution_completed": infrastructure_passed,
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
        "no_environment_override": True,
        "cache_not_shared": True,
        "output_persistence_and_resume": True,
    }
    return {
        "arm": arm,
        "seed": seed,
        "proposal_budget": budget,
        "passed": infrastructure_passed,
        "valid": invalid_rate == 0.0,
        "status": "success" if invalid_rate == 0.0 else "invalid",
        "error": None if infrastructure_passed else "synthetic infrastructure error",
        "metrics": {
            "accuracy": accuracy,
            "invalid_rate": invalid_rate,
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
        "execution_completed": infrastructure_passed,
        "scientific_feasible": invalid_rate == 0.0,
        "safety_passed": invalid_rate == 0.0,
        "selection_changed": optimized,
        "scientific_outcomes": {
            "scientific_feasible": invalid_rate == 0.0,
            "safety_passed": invalid_rate == 0.0,
            "selection_changed": optimized,
        },
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


def test_constraint_failure_is_scientific_not_infrastructure() -> None:
    """A canonical constraint failure completes but fails feasibility and safety."""
    run = _fake_run(
        "A",
        0,
        6,
        accuracy=0.95,
        token_ratio=1.0,
        invalid_rate=0.05,
    )

    assert run["execution_completed"] is True
    assert run["scientific_feasible"] is False
    assert run["safety_passed"] is False
    assert main_experiment._infrastructure_checks_pass(run["checks"])


def test_main_retains_unsafe_run_and_completes_all_statistics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One safety failure neither stops nor disappears from the paired matrix."""
    frozen = main_experiment.validate_frozen_preregistration()
    runs = _synthetic_main_runs()
    runs[0] = _fake_run(
        "A",
        0,
        6,
        accuracy=0.7,
        token_ratio=1.0,
        invalid_rate=0.05,
    )
    lock = {
        "control_plane": {"runtime_tree_sha256": "r" * 64},
        "experiment": {"source": {"sha256": "e" * 64}},
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main_experiment, "MAIN_LOCK_PATH", lock_path)

    progress = main_experiment._progress_document(
        frozen=frozen,
        lock=lock,
        runs=runs,
        stopped_after=None,
    )
    analysis = main_experiment.analyze_main_experiment(progress)

    assert progress["completed_run_count"] == 40
    assert progress["execution_complete"] is True
    assert progress["passed"] is True
    assert progress["scientific_outcomes"]["safety_failure_count"] == 1
    assert analysis["safety_passed"] is False
    assert analysis["absolute_by_arm"]["A"]["runs_with_invalid_output"] == 1
    assert analysis["paired_comparisons"]["B-A"]["paired_blocks"] == 10
    assert analysis["paired_comparisons"]["B-A"]["deltas"]["invalid_rate"]["n"] == 10


def test_actual_infrastructure_error_fails_gate() -> None:
    """An incomplete execution remains a blocking infrastructure error."""
    run = _fake_run(
        "A",
        0,
        6,
        accuracy=0.0,
        token_ratio=1.0,
        infrastructure_passed=False,
    )

    assert run["execution_completed"] is False
    assert main_experiment._infrastructure_checks_pass(run["checks"]) is False


def test_invalid_completed_core_result_resumes_without_provider_calls(
    tmp_path: Path,
) -> None:
    """The real control plane resumes a canonical invalid completed result."""

    class InvalidAnswerClient(_FakeClient):
        """Return non-extractable text only for the final forward answer."""

        def _forward_content(self, prompt: str) -> str:
            if "Analysis from the first stage:" in prompt:
                return "no parseable final answer"
            return super()._forward_content(prompt)

    class InvalidAnswerFactory(_Factory):
        """Create metered offline clients with invalid final answers."""

        def __call__(self, profile: dict[str, Any], role: str) -> _FakeClient:
            client = InvalidAnswerClient(role, str(profile["resolved_model"]))
            self.clients.append(client)
            return client

    register_experiment_components()
    factory = InvalidAnswerFactory()
    raw = build_spec(
        task="object_counting",
        engine="fixed",
        seed=1803,
        output_directory=tmp_path,
        offline=True,
        test_mode=True,
    )
    raw["runtime"]["resume"] = True

    first = control_plane.run_spec(raw, resources={"llm_factory": factory})
    calls_after_first = sum(len(client.requests) for client in factory.clients)
    second = control_plane.run_spec(raw, resources={"llm_factory": factory})

    assert first.status == "invalid"
    assert first.evaluation.status == "constraint_failed"
    assert first.evaluation.metrics["invalid_rate"] > 0.0
    assert second.to_dict() == first.to_dict()
    assert calls_after_first > 0
    assert sum(len(client.requests) for client in factory.clients) == calls_after_first


def test_frozen_protocol_hashes_profiles_and_constraints_are_unchanged() -> None:
    """The amendment leaves P0, parser inputs, pools, profiles, and constraints frozen."""
    package_root = Path(main_experiment.PACKAGE_ROOT)

    assert main_experiment._sha256(main_experiment.FROZEN_PREREGISTRATION_PATH) == (
        "e6954f457132518e9d62f9d0ee2dd0f7d73b49669f3deb905ae481a609c8b8ee"
    )
    assert main_experiment._sha256(
        package_root / "manifests/dataset_manifest_v2.json"
    ) == "3c2ca2924868dc4edd8bb717139e6bfaa7b20388f1fc773f595551e0936661ee"
    assert main_experiment._sha256(package_root / "evaluator.py") == (
        "96aba14935d026cc3a4771ac86df7043b27eefdf66175260589fb89401484eea"
    )
    assert main_experiment._sha256(package_root / "specs.py") == (
        "1088a7b77bc4c9518295a5d657e31f03110d796f8ea9daa3e7962312220939a8"
    )
    frozen = main_experiment.validate_frozen_preregistration()
    assert frozen["initial_artifact"] == main_experiment.INITIAL_ARTIFACT
    assert frozen["model_profiles"]["model"] == main_experiment.MODEL
    assert frozen["objective"]["hard_constraints"] == ["invalid_rate <= 0"]
    for arm, engine in {
        "A": "fixed",
        "B": "trace",
        "C": "gepa_optimize_anything",
        "D": "trace",
    }.items():
        raw = build_spec(task="gsm8k", engine=engine, seed=0, output_directory=None)
        level = raw["levels"][0]
        assert level["objective"]["hard_constraints"] == [
            {"metric": "invalid_rate", "op": "<=", "value": 0.0}
        ]
        assert level["module"]["artifact"] == main_experiment.INITIAL_ARTIFACT, arm


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
        """Return healthy executions, including one scientific safety failure."""
        calls.append((kwargs["seed"], kwargs["proposals"], kwargs["arm"]))
        return _fake_run(
            kwargs["arm"],
            kwargs["seed"],
            kwargs["proposals"],
            accuracy=0.8,
            token_ratio=0.8,
            invalid_rate=0.05 if len(calls) == 1 else 0.0,
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
    assert result["scientific_outcomes"]["safety_failure_count"] == 1
    assert calls == main_experiment.main_execution_matrix(frozen)
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True


def test_main_runner_stops_on_actual_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The unchanged first-infrastructure-failure rule still stops execution."""
    frozen = main_experiment.validate_frozen_preregistration()
    lock = {
        "control_plane": {"runtime_tree_sha256": "r" * 64},
        "experiment": {"source": {"sha256": "e" * 64}},
    }
    calls: list[tuple[int, int, str]] = []

    def execute(**kwargs: Any) -> dict[str, Any]:
        """Return one incomplete canonical run."""
        calls.append((kwargs["seed"], kwargs["proposals"], kwargs["arm"]))
        return _fake_run(
            kwargs["arm"],
            kwargs["seed"],
            kwargs["proposals"],
            accuracy=0.0,
            token_ratio=1.0,
            infrastructure_passed=False,
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

    assert len(calls) == 1
    assert result["completed_run_count"] == 1
    assert result["stopped_after"] is not None
    assert result["passed"] is False
