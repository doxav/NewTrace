"""Post-audit successor integrity tests using actual deterministic benchmark math."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import analysis as A
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import test_analysis as T
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import verify_numerics as V
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I


def fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch, kind: str = "exp17"
) -> tuple[dict[str, Any], dict[str, int]]:
    """Replace synthetic scores with real values while keeping the tested study layout."""
    bundle = T.bundle(kind, seeds=[17101])
    frozen = bundle["freeze"]
    source_file = root / "frozen_source.py"
    source_file.write_text("# numerical test fixture\n")
    frozen["files"] = {str(source_file): B.source_hash(source_file.read_text())}
    frozen["files"][str(Path(B.__file__).resolve())] = B.source_hash(
        Path(B.__file__).read_text()
    )
    tasks = {
        (split, B.task_identity(task)): task
        for split, panel in frozen["tasks"].items()
        for task in panel
    }
    seed_hash = frozen["seed_sha256"]
    b2_hash = frozen["fixed_controls"]["B2"]["source_sha256"]
    for entry in bundle["cache"].values():
        key, row = entry["key"], entry["row"]
        key["evaluator_sha256"] = frozen["files"][str(Path(B.__file__).resolve())]
        task = tasks[key["split"], key["task_identity"]]
        if row["valid"]:
            point = (
                [5.0] * task["dimension"]
                if row["source_sha256"] == seed_hash
                else (
                    [0.0] * task["dimension"]
                    if row["source_sha256"] == b2_hash
                    else list(task["shift"])
                )
            )
            value = B.objective(task, point)
            row["observations"] = [
                {"x": point, "value": value} for _ in range(row["budget"])
            ]
            row["metrics"] = B.metrics(
                [value] * row["budget"], B.normalization(task), row["budget"]
            )
        entry["row_sha256"] = B.digest(row)
    key_map = {
        digest: B.digest(entry["key"]) for digest, entry in bundle["cache"].items()
    }
    bundle["cache"] = {
        key_map[digest]: entry for digest, entry in bundle["cache"].items()
    }
    for event in bundle["events"]:
        event["key"] = key_map[event["key"]]
    index = {
        (
            entry["key"]["outer"],
            entry["key"]["split"],
            entry["key"]["source_sha256"],
            entry["key"]["task_identity"],
            entry["key"]["local_seed"],
        ): entry["row"]
        for entry in bundle["cache"].values()
    }

    def panel(
        rows: list[dict[str, Any]], outer: int, split: str
    ) -> list[dict[str, Any]]:
        """Reuse the exact cache rows in every logical pool and deployment allocation."""
        return [
            copy.deepcopy(
                index[
                    outer,
                    split,
                    row["source_sha256"],
                    row["task_identity"],
                    row["local_seed"],
                ]
            )
            for row in rows
        ]

    for name, candidates in bundle["pools"].items():
        outer = int(name.split("/")[0])
        for candidate in candidates:
            for split in ("train", "validation"):
                candidate[split] = panel(candidate[split], outer, split)
            candidate["validation_auc"] = (
                B.aggregate(candidate["validation"], "auc")
                if candidate["eligible"]
                else None
            )
        bundle["selections"][name]["validation_auc"] = candidates[1]["validation_auc"]
    for outer, arms in bundle["audit"]["per_seed"].items():
        for result in arms.values():
            result["rows"] = panel(result["rows"], int(outer), "audit")
            for metric in ("auc", "final_regret"):
                result[metric] = B.aggregate(result["rows"], metric)
    for slot in bundle["slots"].values():
        slot["request"]["freeze_sha256"] = B.digest(frozen)
    A.summarize(bundle)
    for name in V.REQUIRED_ARTIFACTS:
        I.persist(root / name, bundle["audit"] if name == "audit_results.json" else {})
    I.persist(root / "freeze.json", frozen)
    for digest, entry in bundle["cache"].items():
        I.persist(root / "cache" / f"{digest}.json", entry)
    counters = {"read_bundle": 0, "objective": 0}
    original = B.objective

    def objective(task: dict[str, Any], point: list[float]) -> float:
        """Account every recomputed value, including normalization reference designs."""
        counters["objective"] += 1
        return original(task, point)

    def read_bundle(path: Path) -> dict[str, Any]:
        """Stand in for separately tested disk/chronology loading, preserving real analysis."""
        assert path == root
        counters["read_bundle"] += 1
        return bundle

    monkeypatch.setattr(B, "objective", objective)
    monkeypatch.setattr(V.A, "read_bundle", read_bundle)
    return bundle, counters


@pytest.mark.parametrize("kind", ["exp17", "exp18"])
def test_complete_dynamic_arms_and_b2_recompute_without_input_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    bundle, counters = fixture(tmp_path, monkeypatch, kind)
    before = copy.deepcopy(bundle)
    report = V.verify(tmp_path)
    assert report["status"] == "PASS"
    assert report["cache_rows_verified"] == len(bundle["cache"])
    assert report["invalid_partial_rows_verified"] > 0
    assert report["fixed_control_cache_rows_verified"]["B2"] > 0
    assert report["input_files_unchanged"] is True
    assert report["frozen_source_files_unchanged"] is True
    assert report["integrity_objective_calls"]["total"] == counters["objective"]
    assert (
        report["integrity_objective_calls"]["reference_design"]
        == 18 * B.MANIFEST["normalization_reference_size"]
    )
    assert report["candidate_subprocesses"] == report["model_calls"] == 0
    assert counters["read_bundle"] == 1
    assert bundle == before


@pytest.mark.parametrize(
    "missing",
    ["generation_frozen.json", "selections_frozen.json", "audit_results.json"],
)
def test_completion_presence_guard_precedes_all_reads_and_numerics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    _, counters = fixture(tmp_path, monkeypatch)
    (tmp_path / missing).unlink()
    before = B._reference_scale.cache_info()
    with pytest.raises(RuntimeError, match="completed"):
        V.verify(tmp_path)
    assert counters == {"read_bundle": 0, "objective": 0}
    assert B._reference_scale.cache_info() == before


def test_chronology_rejection_precedes_reference_cache_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, counters = fixture(tmp_path, monkeypatch)
    before = B._reference_scale.cache_info()

    def reject(path: Path) -> dict[str, Any]:
        """Model a failed lazy study preflight/chronology check inside the reader."""
        raise RuntimeError("frozen chronology rejected")

    monkeypatch.setattr(V.A, "read_bundle", reject)
    with pytest.raises(RuntimeError, match="chronology rejected"):
        V.verify(tmp_path)
    assert counters["objective"] == 0
    assert B._reference_scale.cache_info() == before


def test_no_candidate_execution_or_secret_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture(tmp_path, monkeypatch)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        """Catch any accidental attempt to rerun code or access a live credential."""
        raise AssertionError(
            "numeric verification must not execute candidates or load keys"
        )

    from experiments.recursive_opt._shared.optimizer_discovery import phase0

    monkeypatch.setattr(B, "evaluate", forbidden)
    monkeypatch.setattr(B, "propose_point", forbidden)
    monkeypatch.setattr(phase0, "_load_key", forbidden)
    assert V.verify(tmp_path)["status"] == "PASS"


@pytest.mark.parametrize("target", ["observation", "normalized_metric", "bounds"])
def test_self_consistent_storage_corruption_fails_numeric_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    bundle, _ = fixture(tmp_path, monkeypatch)
    entry = next(entry for entry in bundle["cache"].values() if entry["row"]["valid"])
    if target == "observation":
        entry["row"]["observations"][0]["value"] += 0.125
    elif target == "normalized_metric":
        entry["row"]["metrics"]["curve"] = [0.1, 0.1]
        entry["row"]["metrics"].update(auc=0.1, final_regret=0.1)
    else:
        entry["row"]["observations"][0]["x"][0] = 6.0
    entry["row_sha256"] = B.digest(entry["row"])
    # Numerical behavior is isolated from the separately tested logical-copy gate.
    monkeypatch.setattr(V.A, "summarize", lambda value: {})
    before = copy.deepcopy(bundle)
    report = V.verify(tmp_path)
    assert report["status"] == "FAIL"
    assert report["failures"]
    assert bundle == before


@pytest.mark.parametrize("change", ["frozen_source", "run_input", "new_input"])
def test_concurrent_input_mutation_cannot_receive_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    _, _ = fixture(tmp_path, monkeypatch)
    original = B.objective
    mutated = [False]

    def mutate(task: dict[str, Any], point: list[float]) -> float:
        """Alter one input during integrity math to exercise the before/after audit."""
        if not mutated[0]:
            mutated[0] = True
            filename = {
                "frozen_source": "frozen_source.py",
                "run_input": "generation_frozen.json",
                "new_input": "unexpected_input.json",
            }[change]
            (tmp_path / filename).write_text("changed input\n")
        return original(task, point)

    monkeypatch.setattr(B, "objective", mutate)
    report = V.verify(tmp_path)
    assert report["status"] == "FAIL"
    assert any(
        failure["kind"] == "input_files_changed" for failure in report["failures"]
    )


def test_numbered_reports_and_gzip_evidence_remain_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = fixture(tmp_path, monkeypatch)
    digest, entry = next(iter(bundle["cache"].items()))
    path = tmp_path / "cache" / f"{digest}.json"
    entry["padding"] = "x" * 500000
    path.unlink()
    I.persist(path, entry)
    assert path.with_suffix(".json.gz").exists()
    first, report = V.verify_and_persist(tmp_path)
    first_bytes = (
        first.read_bytes()
        if first.exists()
        else first.with_suffix(".json.gz").read_bytes()
    )
    second, again = V.verify_and_persist(tmp_path)
    assert first.name == "attempt_001.json"
    assert second.name == "attempt_002.json"
    assert report["status"] == again["status"] == "PASS"
    assert E.read(first) == report
    assert E.read(second) == again
    assert (
        first.read_bytes()
        if first.exists()
        else first.with_suffix(".json.gz").read_bytes()
    ) == first_bytes
    assert any(name.endswith(".json.gz") for name in report["input_files_sha256"])


def test_numeric_core_retains_nonempty_partial_failure_and_deployment_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolate arithmetic over typed partial/fallback rows, without simulating execution."""
    bundle, counters = fixture(tmp_path, monkeypatch)
    seed_hash = bundle["freeze"]["seed_sha256"]
    partial = next(
        entry
        for entry in bundle["cache"].values()
        if entry["key"]["split"] == "train"
        and entry["row"]["valid"]
        and entry["row"]["source_sha256"] != seed_hash
    )
    partial["row"].update(
        valid=False,
        candidate_valid=False,
        status="exception",
        observations=partial["row"]["observations"][:1],
        objective_calls=1,
        unused_objective_allocation=1,
        metrics=None,
    )
    partial["row_sha256"] = B.digest(partial["row"])
    fallback = next(
        entry
        for entry in bundle["cache"].values()
        if entry["key"]["split"] == "audit"
        and entry["row"]["source_sha256"] != seed_hash
    )
    fallback["row"].update(fallback_used=True, candidate_valid=False)
    fallback["row_sha256"] = B.digest(fallback["row"])
    before = copy.deepcopy(bundle)
    report = V._numerics(bundle)
    assert report["failures"] == []
    assert report["fallback_rows_verified"] == 1
    assert any(
        not row["valid"]
        and row["observations_recomputed"] == 1
        and row["recomputed_metrics"] is None
        for row in report["rows"]
    )
    assert report["integrity_objective_calls"]["total"] == counters["objective"]
    assert bundle == before


def test_failed_reference_design_does_not_claim_all_allocated_calls_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture(tmp_path, monkeypatch)
    original = B.objective
    failed = [False]

    def fail_once(task: dict[str, Any], point: list[float]) -> float:
        """Abort the first reference design after one actual numerical invocation."""
        value = original(task, point)
        if not failed[0]:
            failed[0] = True
            raise ValueError("synthetic normalization defect")
        return value

    monkeypatch.setattr(B, "objective", fail_once)
    report = V.verify(tmp_path)
    assert report["status"] == "FAIL"
    assert report["failed_reference_designs"] == 1
    assert report["integrity_objective_calls"]["reference_design"] is None
    assert report["integrity_objective_calls"]["total"] is None
    assert report["reference_design_calls_upper_bound"] == 18 * 128
