"""Deterministic tests for Experiment 0 v2 eligibility semantics."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.recursive_opt.multiobjective_reasoning import forecast, live
from experiments.recursive_opt.multiobjective_reasoning.datasets import (
    _resolve_v2,
    v2_pool_indices,
)
from experiments.recursive_opt.multiobjective_reasoning.offline_contract import (
    _FakeClient,
    run_offline_contract,
)
from experiments.recursive_opt.multiobjective_reasoning.preflight import (
    classify_near_eligible_tasks,
    evaluate_v2_eligibility,
    reliable_cost_usd,
    select_eligible_task,
)
from experiments.recursive_opt.multiobjective_reasoning.specs import build_spec


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPLIT_IDS = {
    "train": {"train-1"},
    "validation": {"validation-1"},
    "holdout": {"holdout-1"},
}


def _probe(
    accuracy: float, invalid_rate: float, tokens: int, *, holdout_accuracy: float = 0.0
) -> dict[str, Any]:
    validation = {
        "metrics": {"accuracy": accuracy, "invalid_rate": invalid_rate},
        "sample_count": 1,
        "usage": {"forward": {"total_tokens": tokens, "cost_usd": 0.0}},
    }
    return {
        "splits": {
            "train": copy.deepcopy(validation),
            "validation": validation,
            "holdout": {
                "metrics": {"accuracy": holdout_accuracy, "invalid_rate": 0.0},
                "sample_count": 1,
                "usage": {"forward": {"total_tokens": 999999}},
            },
        }
    }


def test_accuracy_sensitive_task_is_eligible() -> None:
    result = evaluate_v2_eligibility(
        {"P0": _probe(0.5, 0.0, 100), "P1": _probe(0.75, 0.0, 100)},
        SPLIT_IDS,
    )

    assert result["eligible"] is True
    assert result["quality_signal"] is True
    assert result["quality_probe_ids"] == ["P1"]


def test_equal_accuracy_with_half_tokens_is_eligible() -> None:
    result = evaluate_v2_eligibility(
        {"P0": _probe(0.5, 0.0, 100), "P1": _probe(0.5, 0.0, 50)},
        SPLIT_IDS,
    )

    assert result["eligible"] is True
    assert result["efficiency_signal"] is True
    assert result["efficiency_probe_ids"] == ["P1"]
    assert result["best_feasible_token_ratio"] == 0.5


def test_flat_quality_and_cost_is_ineligible() -> None:
    result = evaluate_v2_eligibility(
        {"P0": _probe(0.5, 0.0, 100), "P1": _probe(0.5, 0.0, 100)},
        SPLIT_IDS,
    )

    assert result["eligible"] is False
    assert result["exclusion_reasons"] == ["informativeness"]


def test_one_invalid_probe_does_not_exclude_informative_task() -> None:
    result = evaluate_v2_eligibility(
        {
            "P0": _probe(0.5, 0.0, 100),
            "P1": _probe(0.75, 0.0, 100),
            "P2": _probe(0.0, 1.0, 10),
        },
        SPLIT_IDS,
    )

    assert result["eligible"] is True
    assert result["quality_probe_ids"] == ["P1"]


def test_invalid_baseline_is_ineligible() -> None:
    result = evaluate_v2_eligibility(
        {"P0": _probe(0.5, 0.1, 100), "P1": _probe(0.75, 0.0, 100)},
        SPLIT_IDS,
    )

    assert result["eligible"] is False
    assert "baseline_validation_invalid_rate" in result["exclusion_reasons"]


def test_floor_baseline_without_qualifying_rule_is_ineligible() -> None:
    result = evaluate_v2_eligibility(
        {"P0": _probe(0.1, 0.0, 100), "P1": _probe(0.1, 0.0, 100)},
        SPLIT_IDS,
    )

    assert result["eligible"] is False
    assert "baseline_validation_accuracy_range" in result["exclusion_reasons"]


def test_unavailable_cost_is_null_and_token_proxy_ranks_tasks() -> None:
    assert reliable_cost_usd({"total_tokens": 100, "cost_usd": 0.0}) is None
    selected, basis = select_eligible_task(
        {
            "first": {
                "eligible": True,
                "cost_usd": None,
                "forward_tokens_per_evaluated_example": 100.0,
            },
            "second": {
                "eligible": True,
                "cost_usd": None,
                "forward_tokens_per_evaluated_example": 50.0,
            },
        },
        ["first", "second"],
    )

    assert selected == "second"
    assert basis == "forward_tokens_per_evaluated_example"


def test_holdout_metrics_never_enter_eligibility() -> None:
    probes = {"P0": _probe(0.5, 0.0, 100), "P1": _probe(0.5, 0.0, 50)}
    altered = copy.deepcopy(probes)
    altered["P0"]["splits"]["holdout"]["metrics"]["accuracy"] = 1.0
    altered["P1"]["splits"]["holdout"]["metrics"]["accuracy"] = 0.0

    assert evaluate_v2_eligibility(probes, SPLIT_IDS) == evaluate_v2_eligibility(
        altered, SPLIT_IDS
    )


def test_v1_artifacts_are_preserved_byte_for_byte() -> None:
    hashes = json.loads(
        (PACKAGE_ROOT / "manifests/v1/evidence_hashes.json").read_text(encoding="utf-8")
    )["files"]
    pairs = (
        ("preregistration.json", PACKAGE_ROOT / "manifests/preregistration.json"),
        ("dataset_manifest.json", PACKAGE_ROOT / "manifests/dataset_manifest.json"),
        ("task_eligibility.json", PACKAGE_ROOT / "reports/task_eligibility.json"),
    )
    for name, original in pairs:
        preserved_root = "manifests" if name != "task_eligibility.json" else "reports"
        preserved = PACKAGE_ROOT / preserved_root / "v1" / name
        assert hashlib.sha256(original.read_bytes()).hexdigest() == hashes[name]
        assert hashlib.sha256(preserved.read_bytes()).hexdigest() == hashes[name]


def test_v2_pool_generation_is_repeatable_and_disjoint() -> None:
    first = v2_pool_indices()
    second = v2_pool_indices()

    assert first == second
    for splits in first.values():
        ids = {
            split: {f"{source}:{index}" for source, index in values}
            for split, values in splits.items()
        }
        assert len(ids["train"]) >= 16
        assert len(ids["validation"]) >= 12
        assert len(ids["holdout"]) >= 24
        assert ids["train"].isdisjoint(ids["validation"])
        assert ids["train"].isdisjoint(ids["holdout"])
        assert ids["validation"].isdisjoint(ids["holdout"])


def test_v2_resolver_accepts_frozen_sample_id_tuples() -> None:
    source_split, index = v2_pool_indices()["gsm8k"]["train"][0]
    sample_id = f"gsm8k:{source_split}:{index}"

    rows = _resolve_v2("gsm8k", "train", {"sample_ids": (sample_id,)})

    assert [row["id"] for row in rows] == [sample_id]


def test_preserved_v1_mechanically_classifies_only_gsm8k_as_near() -> None:
    v1 = json.loads(
        (PACKAGE_ROOT / "reports/v1/task_eligibility.json").read_text(encoding="utf-8")
    )
    preregistration = json.loads(
        (PACKAGE_ROOT / "manifests/preregistration_v2.json").read_text(encoding="utf-8")
    )

    assert classify_near_eligible_tasks(v1, preregistration) == ["gsm8k"]


def test_optimizer_profile_pins_low_effort_and_nontrivial_output_budget() -> None:
    raw = build_spec(
        task="gsm8k",
        engine="trace",
        seed=0,
        output_directory=None,
        offline=True,
    )
    profile = raw["llm_profiles"]["optimizer_primary"]

    assert profile["request_params"] == {"reasoning": {"effort": "low"}}
    assert profile["max_tokens"] == 8192
    assert profile["max_tokens"] > 1024


def test_offline_provider_rejects_gepa_positional_protocol() -> None:
    client = _FakeClient("optimizer", "fake/strict-chat")

    with pytest.raises(TypeError, match="keyword chat messages"):
        client("foreign GEPA prompt")

    response = client(
        messages=[
            {
                "role": "user",
                "content": "## Optimization Goal\nImprove.\n## Current Component\nSeed.",
            }
        ]
    )

    assert response.choices[0].message.content == "```\nOFFLINE GEPA IMPROVED\n```"
    assert [request["kind"] for request in client.requests] == ["gepa_reflection"]


def test_complete_offline_contract_uses_no_external_network() -> None:
    """Run all offline assertions under the caller's pytest-socket policy."""
    result = run_offline_contract()

    assert result["passed"] is True
    assert len(result["assertions"]) == 20


def test_live_runner_reads_versioned_runtime_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "control_plane_lock.json"
    monkeypatch.setattr(live, "CONTROL_PLANE_LOCK", lock_path)
    digest = "a" * 64
    lock_path.write_text(
        json.dumps({"control_plane": {"runtime_tree_sha256": digest}}),
        encoding="utf-8",
    )

    assert live._locked_runtime_tree_sha256() == digest

    lock_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks a runtime digest"):
        live._locked_runtime_tree_sha256()


def test_rejected_candidate_exercises_proposal_without_changing_selection() -> None:
    accounted = {"candidates_proposed": 1, "candidates_evaluated": 1}
    optimizer_usage = {"calls": 1, "total_tokens": 10}

    assert live._proposal_path_exercised(
        optimized=True,
        optimizer_usage=optimizer_usage,
        accounted=accounted,
    )
    assert not live._selection_changed(
        optimized=True,
        artifact=live.INITIAL_ARTIFACT,
    )
    checks = {
        name: True for name in live._RESUME_INFRASTRUCTURE_PREREQUISITES
    }
    checks.update(
        proposal_path_exercised=True,
        output_persistence_and_resume=True,
    )
    assert live._infrastructure_checks_pass(checks)


def test_missing_optimizer_call_or_candidate_fails_proposal_path() -> None:
    assert not live._proposal_path_exercised(
        optimized=True,
        optimizer_usage={"calls": 0, "total_tokens": 0},
        accounted={"candidates_proposed": 0, "candidates_evaluated": 0},
    )


def test_accepted_candidate_exercises_proposal_and_changes_selection() -> None:
    accepted = {
        **live.INITIAL_ARTIFACT,
        "analysis_instruction": "Accepted optimized instruction.",
    }

    assert live._proposal_path_exercised(
        optimized=True,
        optimizer_usage={"calls": 1, "total_tokens": 10},
        accounted={"candidates_proposed": 1, "candidates_evaluated": 1},
    )
    assert live._selection_changed(optimized=True, artifact=accepted)


@pytest.mark.parametrize("status", ["error", "budget_exhausted"])
def test_infrastructure_status_does_not_count_as_completed(status: str) -> None:
    result = SimpleNamespace(
        status=status,
        evaluation=SimpleNamespace(status="error"),
    )

    assert live._execution_completed(result) is False
    assert live._execution_completed(SimpleNamespace(status="success", evaluation=None)) is False


def test_invalid_constraint_result_is_scientific_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = {
        "budget": {},
        "runtime": {},
        "llm_profiles": {
            "forward_primary": {
                "request_params": {"reasoning": {"enabled": False}}
            },
            "optimizer_primary": {
                "request_params": {"reasoning": {"effort": "low"}}
            },
        },
    }
    output = object()
    result = SimpleNamespace(
        status="invalid",
        valid=False,
        error=None,
        artifact=copy.deepcopy(live.INITIAL_ARTIFACT),
        evaluation=SimpleNamespace(
            status="constraint_failed",
            metrics={"accuracy": 0.95, "invalid_rate": 0.05},
        ),
        usage={
            "forward": {"calls": 1, "total_tokens": 10},
            "optimizer": {"calls": 1, "total_tokens": 5},
        },
        budget={
            "accounted": {
                "eval_llm_calls": 1,
                "evaluator_runs": 1,
                "candidates_proposed": 1,
                "candidates_evaluated": 1,
            }
        },
        metadata={"selected_models": {}},
        to_dict=lambda: {"result": "stable"},
    )
    plan = SimpleNamespace(
        spec=raw,
        fingerprint="fingerprint",
        code_provenance={
            "runtime_tree_sha256": "a" * 64,
            "registry_sha256": "b" * 64,
        },
    )
    calls = 0

    def fake_run_spec(spec: dict[str, Any]) -> Any:
        """Populate events only for the initial execution, not its resume."""
        nonlocal calls
        calls += 1
        if calls == 1:
            live.FORWARD_EVENTS.append({"output": output})
            live.EVALUATOR_EVENTS.append(
                {
                    "output_identity": id(output),
                    "sample_id": "train-1",
                    "phase": "proposal",
                }
            )
        return result

    monkeypatch.setattr(live, "build_spec", lambda **kwargs: copy.deepcopy(raw))
    monkeypatch.setattr(live.control_plane, "compile_plan", lambda spec: plan)
    monkeypatch.setattr(live.control_plane, "run_spec", fake_run_spec)
    monkeypatch.setattr(live, "_locked_runtime_tree_sha256", lambda: "a" * 64)
    monkeypatch.setattr(
        live,
        "_locked_experiment_source_sha256",
        lambda: "c" * 64,
    )
    monkeypatch.setattr(
        live,
        "experiment_source_provenance",
        lambda: {"sha256": "c" * 64},
    )
    monkeypatch.setattr(
        live,
        "_load_json",
        lambda path: {"tasks": {"gsm8k": {"samples": []}}},
    )
    monkeypatch.setattr(
        live,
        "_per_example_forward_usage",
        lambda: [
            {
                "usage": {"total_tokens": 10},
                "provider_calls": [
                    {
                        "actual_provider": "openrouter",
                        "actual_model": live.MODEL,
                        "cache_hit": False,
                    }
                ],
            }
        ],
    )
    for name in live._OVERRIDE_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    run = live._execute_arm(
        arm="C",
        task="gsm8k",
        baseline_tokens={},
        seed=0,
        proposals=1,
        split_limits={"train": 1, "validation": 1, "holdout": 1},
        budget_limits={},
        output_directory=tmp_path,
    )

    assert calls == 2
    assert run["checks"]["proposal_path_exercised"] is True
    assert run["checks"]["output_persistence_and_resume"] is True
    assert run["execution_completed"] is True
    assert run["scientific_feasible"] is False
    assert run["safety_passed"] is False
    assert run["scientific_outcomes"]["selection_changed"] is False
    assert run["passed"] is True


def test_pilot_keeps_real_proposals_and_selection_change_independent() -> None:
    runs = [
        {
            "arm": arm,
            "checks": {
                "proposal_path_exercised": True,
            },
            "scientific_outcomes": {"selection_changed": False},
        }
        for arm in ("B", "C", "D")
    ]

    gates = live._pilot_optimizer_gates(runs)

    assert gates == {
        "trace_real_proposal": True,
        "gepa_real_proposal": True,
        "optimized_artifact_differs": False,
    }


def test_pilot_retry_statistics_are_separate_and_metered_by_arm() -> None:
    runs = [
        {
            "arm": "B",
            "usage": {
                "optimizer": {
                    "empty_text_responses": 1,
                    "semantic_retries": 1,
                    "semantic_retry_prompt_tokens": 10,
                    "semantic_retry_completion_tokens": 2,
                    "semantic_retry_total_tokens": 12,
                    "semantic_retry_cost_usd": 0.01,
                }
            },
        },
        {
            "arm": "B",
            "usage": {
                "optimizer": {
                    "empty_text_responses": 2,
                    "semantic_retries": 1,
                    "semantic_retry_prompt_tokens": 20,
                    "semantic_retry_completion_tokens": 4,
                    "semantic_retry_total_tokens": 24,
                    "semantic_retry_cost_usd": 0.02,
                }
            },
        },
        {"arm": "C", "usage": {"optimizer": {}}},
    ]

    statistics = live._pilot_retry_statistics(runs)

    assert statistics["B"] == {
        "empty_text_responses": 3,
        "semantic_retries": 2,
        "semantic_retry_prompt_tokens": 30,
        "semantic_retry_completion_tokens": 6,
        "semantic_retry_total_tokens": 36,
        "semantic_retry_cost_usd": pytest.approx(0.03),
    }
    assert statistics["C"]["empty_text_responses"] == 0
    assert statistics["C"]["semantic_retries"] == 0


def test_completed_pilot_replaces_smoke_projection_and_includes_retry_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(forecast, "PILOT_REPORT_PATH", pilot_path)
    micro = {
        "passed": True,
        "arms": {
            arm: {
                "usage": {
                    "forward": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                    "optimizer": {},
                }
            }
            for arm in ("A", "B", "C")
        },
    }
    pilot = {
        "passed": True,
        "runs": [
            {
                "arm": arm,
                "usage": {
                    "forward": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    "optimizer": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                },
            }
            for arm in ("A", "B", "C", "D")
        ],
        "retry_statistics_by_arm": {
            "B": {
                "semantic_retry_prompt_tokens": 20,
                "semantic_retry_completion_tokens": 10,
            }
        },
    }
    preregistration = {
        "dataset_pools": {
            "micro_smoke_subset": {"train": 1, "validation": 1, "holdout": 1},
            "pilot_subset": {"train": 4, "validation": 4, "holdout": 8},
            "minimum_sizes": {"train": 16, "validation": 12, "holdout": 24},
        },
        "pilot": {"candidate_budgets": [4, 6]},
        "main_monetary_ceiling_usd": None,
    }
    pricing = {
        "input_usd_per_million_tokens": 1.0,
        "output_usd_per_million_tokens": 2.0,
    }

    def fake_load(path: Path) -> dict[str, Any]:
        """Return deterministic forecast inputs for each requested path."""
        if path == forecast.MICRO_REPORT_PATH:
            return micro
        if path == pilot_path:
            return pilot
        if path.name == "preregistration_v2.json":
            return preregistration
        if path.name == "provider_pricing.json":
            return pricing
        raise AssertionError(f"unexpected forecast input: {path}")

    monkeypatch.setattr(forecast, "_load_json", fake_load)

    result = forecast.build_cost_forecast()

    assert result["pilot"]["complete"] is True
    assert result["pilot"]["tokens"]["total_tokens"] == 720
    assert result["pilot"]["retry_cost_usd_by_arm"]["B"] == 0.00004
    assert result["main_full_v2_pool"]["projected_tokens"]["total_tokens"] > 720
    assert result["recommended_main_cost_ceiling_usd"] == pytest.approx(
        result["main_full_v2_pool"]["projected_cost_usd"] * 1.2
    )
    assert result["main_run_authorized"] is False
