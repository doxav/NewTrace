"""Post-audit numeric-integrity tests with synthetic files and real deterministic objectives."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.production import (
    verify_numerics as V,
)


def _fixture(tmp_path: Path, monkeypatch: Any) -> tuple[dict[str, Any], dict[str, int]]:
    """Declare completed toy evidence; objective values and metrics are computed by the real benchmark."""
    tasks = {
        split: G.fresh_tasks("UNIT-NUMERIC-POSTAUDIT", split, 1)[:1]
        for split in V.SPLITS
    }
    frozen = {"config": {"budget": 2}, "tasks": tasks, "benchmark_manifest": B.MANIFEST}
    bundle: dict[str, Any] = {
        "freeze": frozen,
        "cache": {},
        "audit": {"completed_ns": 3},
    }
    for split, panel in tasks.items():
        task = panel[0]
        observations = [
            {
                "x": [x] * task["dimension"],
                "value": B.objective(task, [x] * task["dimension"]),
            }
            for x in (0.0, 1.0)
        ]
        valid = split != "validation"
        if not valid:
            observations = observations[:1]
        row = {
            "valid": valid,
            "candidate_valid": valid and split != "audit",
            "fallback_used": split == "audit",
            "budget": 2,
            "observations": observations,
            "objective_calls": len(observations),
            "unused_objective_allocation": 2 - len(observations),
            "source_sha256": B.source_hash("UNIT_SOURCE"),
            "task_identity": B.task_identity(task),
            "local_seed": 1,
            "metrics": (
                B.metrics(
                    [observation["value"] for observation in observations],
                    B.normalization(task),
                    2,
                )
                if valid
                else None
            ),
        }
        key = {
            "split": split,
            "task_identity": row["task_identity"],
            "source_sha256": row["source_sha256"],
            "local_seed": 1,
            "budget": 2,
        }
        bundle["cache"][B.digest(key)] = {
            "key": key,
            "row": row,
            "row_sha256": B.digest(row),
        }
    for name in V.REQUIRED_ARTIFACTS:
        (tmp_path / name).write_text("{}\n")
    counters = {"objective": 0, "read_bundle": 0, "summarize": 0}
    original_objective = B.objective

    def objective(task: dict[str, Any], point: list[float]) -> float:
        """Count actual numerical invocations, including reference-design reconstruction."""
        counters["objective"] += 1
        return original_objective(task, point)

    def read_bundle(root: Path) -> dict[str, Any]:
        """Supply the already-complete toy bundle without reading any experiment data."""
        assert root == tmp_path
        counters["read_bundle"] += 1
        return bundle

    def summarize(value: dict[str, Any]) -> dict[str, Any]:
        """Represent separately tested frozen structural analysis in this numerical unit fixture."""
        assert value is bundle
        counters["summarize"] += 1
        return {"unit_structural_checks": True}

    monkeypatch.setattr(B, "objective", objective)
    monkeypatch.setattr(V.A, "read_bundle", read_bundle)
    monkeypatch.setattr(V.A, "summarize", summarize)
    return bundle, counters


@pytest.mark.parametrize(
    "missing",
    ["audit_results.json", "selections_frozen.json", "generation_frozen.json"],
)
def test_missing_completion_guard_prevents_all_reads_and_numerics(
    tmp_path: Path, monkeypatch: Any, missing: str
) -> None:
    """An engineering-only run or incomplete search cannot trigger hidden split reconstruction."""
    _, counters = _fixture(tmp_path, monkeypatch)
    (tmp_path / missing).unlink()
    with pytest.raises(RuntimeError, match="completed"):
        V.verify(tmp_path)
    assert counters == {"objective": 0, "read_bundle": 0, "summarize": 0}


def test_rebuilds_references_and_preserves_partial_invalid_and_fallback_rows(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Recompute real values without executing policies or assigning scores to invalid partial rows."""
    bundle, counters = _fixture(tmp_path, monkeypatch)
    before = copy.deepcopy(bundle)
    report = V.verify(tmp_path)
    assert report["status"] == "PASS"
    assert report["cache_rows_verified"] == 3
    assert report["invalid_partial_rows_verified"] == 1
    assert report["fallback_rows_verified"] == 1
    assert report["integrity_objective_calls"] == {
        "observations": 5,
        "reference_design": 384,
        "total": 389,
    }
    assert report["reference_cache_misses"] == 3
    assert counters == {"objective": 389, "read_bundle": 1, "summarize": 1}
    assert bundle == before


def test_frozen_structural_guard_runs_before_cache_reset_or_objectives(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """File presence alone cannot authorize reconstruction when frozen chronology fails."""
    _, counters = _fixture(tmp_path, monkeypatch)
    cache_before = B._reference_scale.cache_info()

    def reject(root: Path) -> dict[str, Any]:
        """Represent the frozen reader rejecting incomplete or tampered evidence."""
        raise RuntimeError("unit frozen audit barrier rejected")

    monkeypatch.setattr(V.A, "read_bundle", reject)
    with pytest.raises(RuntimeError, match="barrier rejected"):
        V.verify(tmp_path)
    assert counters["objective"] == 0
    assert B._reference_scale.cache_info() == cache_before


def test_numeric_check_never_executes_a_candidate(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Objective recomputation must not invoke an optimizer evaluation or proposal subprocess."""
    _fixture(tmp_path, monkeypatch)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        """Fail loudly if a future verifier accidentally reruns a candidate."""
        raise AssertionError("candidate execution forbidden in numerical verification")

    monkeypatch.setattr(B, "evaluate", forbidden)
    monkeypatch.setattr(B, "propose_point", forbidden)
    assert V.verify(tmp_path)["status"] == "PASS"


@pytest.mark.parametrize(
    "mutation", ["observation", "metric", "invalid_metric", "task_identity"]
)
def test_numeric_and_identity_corruption_is_reported_without_repair(
    tmp_path: Path, monkeypatch: Any, mutation: str
) -> None:
    """Reject wrong numerical evidence even when its storage hash was recomputed consistently."""
    bundle, _ = _fixture(tmp_path, monkeypatch)
    entry = next(
        entry
        for entry in bundle["cache"].values()
        if entry["key"]["split"]
        == ("validation" if mutation == "invalid_metric" else "train")
    )
    if mutation == "observation":
        entry["row"]["observations"][0]["value"] += 0.125
    elif mutation == "metric":
        entry["row"]["metrics"]["auc"] += 0.125
    elif mutation == "invalid_metric":
        entry["row"]["metrics"] = {"auc": 0.0}
    else:
        entry["row"]["task_identity"] = "wrong"
    entry["row_sha256"] = B.digest(entry["row"])
    before = copy.deepcopy(bundle)
    report = V.verify(tmp_path)
    assert report["status"] == "FAIL"
    assert report["failures"]
    assert bundle == before
