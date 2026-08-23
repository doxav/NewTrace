"""Deterministic tests for Experiment 0 v2 eligibility semantics."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt.multiobjective_reasoning import live
from experiments.recursive_opt.multiobjective_reasoning.datasets import (
    _resolve_v2,
    v2_pool_indices,
)
from experiments.recursive_opt.multiobjective_reasoning.offline_contract import (
    _FakeClient,
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
