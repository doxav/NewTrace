"""Prospective fixed-baseline guards with synthetic rows only; no live calls."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.production import (
    baseline_control as C,
)


def synthetic_row(
    source: str, task: dict[str, Any], local: int, **kwargs: Any
) -> dict[str, Any]:
    """Provide an intentionally poor complete deployment and a visible fallback."""
    value, budget = 0.2, kwargs["budget"]
    fallback = local % 2 == 0
    return {
        "valid": True,
        "candidate_valid": not fallback,
        "status": "valid",
        "source_sha256": B.source_hash(source),
        "task_identity": B.task_identity(task),
        "stratum": f"{task['family']}/{task['dimension']}",
        "local_seed": local,
        "budget": budget,
        "observations": [{"x": [0.0] * task["dimension"], "value": value}] * budget,
        "proposal_attempts": (
            [{"evaluation_index": 0, "status": "exception", "policy": "candidate"}]
            if fallback
            else []
        )
        + [
            {
                "evaluation_index": i,
                "status": "valid",
                "policy": "fallback" if fallback else "candidate",
            }
            for i in range(budget)
        ],
        "fallback_used": fallback,
        "objective_calls": budget,
        "unused_objective_allocation": 0,
        "subprocess_executions": 2 * budget + int(fallback),
        "execution_s": 0.001,
        "metrics": B.metrics([value] * budget, 1.0, budget),
    }


@pytest.fixture
def prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Any]]:
    """Bind a synthetic primary manifest without opening real P1/E1/F1 evidence."""
    primary, control = tmp_path / "primary", tmp_path / "control"
    primary.mkdir()
    source = tmp_path / "optimizer.py"
    source.write_text(C.SOURCE_PATH.read_text())
    frozen = {
        "created_ns": 1,
        "config": {
            "namespace": "P1",
            "outer_seeds": list(C.OUTER_SEEDS),
            "slots": 8,
            "budget": 32,
            "local_replicates": 2,
            "workers": 8,
            "timeout_s": 2,
            "arms": ["I", "C", "R", "W"],
        },
        "tasks": {"audit": G.fresh_tasks("UNIT-B2-CONTROL", "audit", 2)},
        "seed_source": B.SEED_SOURCE,
        "seed_sha256": B.source_hash(B.SEED_SOURCE),
        "benchmark_manifest": B.MANIFEST,
    }
    I.persist(primary / "freeze.json", frozen)
    monkeypatch.setattr(C.S, "preflight", lambda path: E.read(path / "freeze.json"))
    state: dict[str, Any] = {
        "evaluations": [],
        "primary_analysis_reads": 0,
        "frozen": frozen,
    }

    def evaluate(
        source: str, task: dict[str, Any], local: int, **kwargs: Any
    ) -> dict[str, Any]:
        """Record only control evaluations and return synthetic complete metrics."""
        state["evaluations"].append(
            (B.source_hash(source), B.task_identity(task), local, kwargs)
        )
        return synthetic_row(source, task, local, **kwargs)

    def read_bundle(root: Path) -> dict[str, Any]:
        """Count comparator reads and emulate the already-tested complete primary loader."""
        state["primary_analysis_reads"] += 1
        audit = {"per_seed": {}}
        for outer in C.OUTER_SEEDS:
            rows = [
                synthetic_row(B.SEED_SOURCE, job["task"], job["local_seed"], budget=32)
                for job in C.jobs(frozen)
                if job["outer"] == outer
            ]
            audit["per_seed"][str(outer)] = {
                "A0": {
                    "auc": 0.1,
                    "final_regret": 0.1,
                    "source_sha256": frozen["seed_sha256"],
                    "rows": rows,
                },
                "R": {
                    "auc": 0.05,
                    "final_regret": 0.05,
                    "source_sha256": "registered-R",
                    "rows": copy.deepcopy(rows),
                },
            }
        return {"freeze": frozen, "audit": audit}

    monkeypatch.setattr(C.B, "evaluate", evaluate)
    monkeypatch.setattr(C.A, "read_bundle", read_bundle)
    monkeypatch.setattr(
        C.A,
        "summarize",
        lambda bundle: {
            "schema": "synthetic-verified",
            "outer_seeds": list(C.OUTER_SEEDS),
        },
    )
    C.prepare(primary, control, source_path=source)
    return primary, control, state


def complete_primary(primary: Path) -> None:
    """Create synthetic completion barriers after prospective control registration."""
    I.persist(primary / "generation_started.json", C.S.clock_snapshot())
    for name in ("generation_frozen", "selections_frozen", "audit_results"):
        I.persist(
            primary / (name + ".json"),
            {"completed_ns": C.S.clock_snapshot()["wall_ns"]},
        )


@pytest.mark.parametrize(
    "marker",
    [
        "generation_started.json",
        "raw/1/R/slot_00/started_1.json",
        "raw/1/R/slot_00/response.json",
    ],
)
def test_cannot_register_reference_after_primary_generation(
    prepared: Any, marker: str
) -> None:
    """A late extra baseline cannot be passed off as prospectively fixed."""
    primary, control, _ = prepared
    I.persist(primary / marker, {})
    with pytest.raises(RuntimeError, match="before primary generation"):
        C.prepare(primary, control.parent / "late")


@pytest.mark.parametrize(
    "missing", ["generation_frozen", "selections_frozen", "audit_results"]
)
def test_no_audit_read_or_execution_before_all_primary_barriers(
    prepared: Any, missing: str
) -> None:
    """Incomplete primary evidence cannot trigger comparator reads or control evaluation."""
    primary, control, state = prepared
    I.persist(primary / "generation_started.json", C.S.clock_snapshot())
    for name in ("generation_frozen", "selections_frozen", "audit_results"):
        if name != missing:
            I.persist(
                primary / (name + ".json"),
                {"completed_ns": C.S.clock_snapshot()["wall_ns"]},
            )
    with pytest.raises(RuntimeError, match="primary completion"):
        C.run(control)
    assert state["evaluations"] == []
    assert state["primary_analysis_reads"] == 0


def test_full_control_preserves_losses_fallback_and_primary_files(
    prepared: Any,
) -> None:
    """Exactly144 fixed policy trajectories leave primary evidence byte-identical."""
    primary, control, state = prepared
    complete_primary(primary)
    before = {
        str(path): path.read_bytes() for path in primary.rglob("*") if path.is_file()
    }
    result = C.run(control)
    assert len(state["evaluations"]) == 144
    assert result["resources"]["objective_calls"] == 4608
    assert result["resources"]["generative_calls"] == 0
    assert result["contrasts"]["B2-A0"]["mean"] == pytest.approx(0.1)
    assert result["contrasts"]["B2-A0"]["interpretation"] == "negative signal"
    assert result["contrasts"]["R-B2"]["mean"] == pytest.approx(-0.15)
    assert result["deployment"]["fallback_trajectories"] > 0
    assert set(result["per_seed"]) == {str(outer) for outer in C.OUTER_SEEDS}
    assert before == {
        str(path): path.read_bytes() for path in primary.rglob("*") if path.is_file()
    }
    assert all(
        record[0] == C.SOURCE_SHA256
        and record[3]["deployment"] is True
        and record[3]["seed_source"] == B.SEED_SOURCE
        for record in state["evaluations"]
    )
    state["evaluations"].clear()
    assert C.run(control) == result
    assert state["evaluations"] == []


def test_source_or_primary_freeze_drift_blocks_evaluation(prepared: Any) -> None:
    """The precommitted reference cannot change after seeing search evidence."""
    _primary, control, state = prepared
    freeze = E.read(control / "freeze.json")
    Path(freeze["source_path"]).write_text("changed reference")
    with pytest.raises(RuntimeError, match="source"):
        C.run(control)
    assert not state["evaluations"]


def test_missing_job_and_changed_row_are_never_dropped(prepared: Any) -> None:
    """Analysis requires the complete fixed schedule and verifies every stored row."""
    primary, control, _ = prepared
    complete_primary(primary)
    C.run(control)
    raw_path = next((control / "raw").rglob("*.json"))
    saved = raw_path.read_bytes()
    raw_path.unlink()
    with pytest.raises(RuntimeError, match="complete control"):
        C.analyze(control)
    raw_path.write_bytes(saved)
    value = E.read(raw_path)
    value["row"]["metrics"]["auc"] = 0.0
    raw_path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="integrity"):
        C.analyze(control)


def test_primary_integrity_failure_blocks_all_control_evaluations(
    prepared: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing file-existence barriers is insufficient if the primary analysis rejects evidence."""
    primary, control, state = prepared
    complete_primary(primary)

    def reject(bundle: Any) -> None:
        """Represent a failed primary S/A integrity check."""
        raise ValueError("primary integrity failure")

    monkeypatch.setattr(C.A, "summarize", reject)
    with pytest.raises(ValueError, match="integrity"):
        C.run(control)
    assert state["evaluations"] == []


def test_interrupted_control_resumes_only_unfinished_jobs(
    prepared: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded infrastructure exception preserves other completed rows and is not an outcome."""
    primary, control, state = prepared
    complete_primary(primary)
    job = E.read(control / "freeze.json")["jobs"][0]
    evaluate = C.B.evaluate
    fail_once = [True]

    def interrupt(
        source: str, task: dict[str, Any], local: int, **kwargs: Any
    ) -> dict[str, Any]:
        """Fail one uniquely identified synthetic job before producing a valid row."""
        if (
            fail_once[0]
            and local == job["local_seed"]
            and B.task_identity(task) == B.task_identity(job["task"])
        ):
            fail_once[0] = False
            raise RuntimeError("synthetic infrastructure interruption")
        return evaluate(source, task, local, **kwargs)

    monkeypatch.setattr(C.B, "evaluate", interrupt)
    with pytest.raises(RuntimeError, match="synthetic infrastructure"):
        C.run(control)
    completed = list((control / "raw").rglob("*.json"))
    assert 0 < len(completed) < 144
    assert not E.exists(control / "results.json")
    preserved = {path: path.read_bytes() for path in completed}
    state["evaluations"].clear()
    result = C.run(control)
    assert len(state["evaluations"]) == 144 - len(completed)
    assert len(result["resources"]["infrastructure_errors"]) == 1
    assert result["resources"]["objective_calls"] == 4608
    assert all(path.read_bytes() == payload for path, payload in preserved.items())


def test_uncheckpointed_attempt_is_reported_after_successful_resume(
    prepared: Any,
) -> None:
    """A lost attempt is retained as unknown work, never silently treated as zero cost."""
    primary, control, _ = prepared
    complete_primary(primary)
    job = E.read(control / "freeze.json")["jobs"][0]
    I.persist(
        control / "attempts/interrupted.started.json",
        {"job_id": job["id"], "clock": C.S.clock_snapshot()},
    )
    result = C.run(control)
    assert result["resources"]["uncheckpointed_attempt_ids"] == ["interrupted"]


def test_primary_binding_and_storage_boundaries(prepared: Any) -> None:
    """A new primary freeze or nested output tree cannot reuse the reference registration."""
    primary, control, state = prepared
    with pytest.raises(RuntimeError, match="separate"):
        C.prepare(primary, primary / "control")
    value = E.read(primary / "freeze.json")
    value["created_ns"] += 1
    (primary / "freeze.json").write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="primary freeze"):
        C.run(control)
    assert state["evaluations"] == []


def test_control_row_must_postdate_primary_audit(prepared: Any) -> None:
    """Verbatim metrics cannot legitimize a control trajectory evaluated before the barrier."""
    primary, control, _ = prepared
    complete_primary(primary)
    C.run(control)
    path = next((control / "raw").rglob("*.json"))
    value = E.read(path)
    value["started"]["wall_ns"] = 1
    journal = control / "attempts" / (value["attempt_id"] + ".started.json")
    journal.write_text(
        json.dumps({"job_id": value["job"]["id"], "clock": value["started"]})
    )
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="chronology"):
        C.analyze(control)


def test_completed_result_cannot_replace_deleted_raw(prepared: Any) -> None:
    """Completed scientific evidence requires recovery, not a fresh replacement evaluation."""
    primary, control, state = prepared
    complete_primary(primary)
    C.run(control)
    next((control / "raw").rglob("*.json")).unlink()
    state["evaluations"].clear()
    with pytest.raises(RuntimeError, match="complete control"):
        C.run(control)
    assert state["evaluations"] == []


def test_corrupted_partial_checkpoint_blocks_before_new_jobs(prepared: Any) -> None:
    """Already available rows are integrity-checked before any unfinished work resumes."""
    primary, control, state = prepared
    complete_primary(primary)
    C.run(control)
    (control / "results.json").unlink()
    paths = list((control / "raw").rglob("*.json"))
    value = E.read(paths[0])
    value["row_sha256"] = "corrupted"
    paths[0].write_text(json.dumps(value))
    paths[1].unlink()
    state["evaluations"].clear()
    with pytest.raises(RuntimeError, match="integrity"):
        C.run(control)
    assert state["evaluations"] == []


def test_completed_row_requires_matching_attempt_journal(prepared: Any) -> None:
    """A preserved outcome must retain its matching start record for provenance."""
    primary, control, state = prepared
    complete_primary(primary)
    C.run(control)
    row = E.read(next((control / "raw").rglob("*.json")))
    (control / "attempts" / (row["attempt_id"] + ".started.json")).unlink()
    state["evaluations"].clear()
    with pytest.raises(RuntimeError, match="attempt journal"):
        C.run(control)
    assert state["evaluations"] == []
