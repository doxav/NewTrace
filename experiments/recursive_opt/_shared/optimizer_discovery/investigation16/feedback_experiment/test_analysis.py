"""F1 final audit tests use unit namespaces and never execute generated programs."""

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import feedback_experiment as F
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G


def load_analysis() -> ModuleType:
    """Load a report file without shadowing the frozen feedback_experiment.py module."""
    spec = importlib.util.spec_from_file_location(
        "f1_final_analysis", F.ROOT / "analysis.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load_analysis()


def unit_row(
    source: str, task: dict[str, Any], block: int, deployment: bool
) -> dict[str, Any]:
    """Represent a synthetic full trajectory or initial failure with common fallback."""
    candidate_valid = bool(source)
    valid = candidate_valid or deployment
    value = B.objective(task, [0.0] * task["dimension"])
    attempts = (
        []
        if candidate_valid
        else [
            {"evaluation_index": 0, "status": "missing_source", "policy": "candidate"}
        ]
    )
    if valid:
        attempts.extend(
            {
                "evaluation_index": i,
                "status": "valid",
                "policy": "candidate" if candidate_valid else "fallback",
                "stdout": "",
                "stderr": "",
            }
            for i in range(32)
        )
    return {
        "source_sha256": B.source_hash(source),
        "task_identity": B.task_identity(task),
        "stratum": f'{task["family"]}/{task["dimension"]}',
        "local_seed": G.local_seed(F.TASK_NAMESPACE, block, task),
        "budget": 32,
        "valid": valid,
        "candidate_valid": candidate_valid,
        "status": "valid" if valid else "missing_source",
        "fallback_used": not candidate_valid and deployment,
        "observations": [{"x": [0.0] * task["dimension"], "value": value}]
        * (32 if valid else 0),
        "objective_calls": 32 if valid else 0,
        "unused_objective_allocation": 0 if valid else 32,
        "metrics": (
            B.metrics([value] * 32, B.normalization(task), 32) if valid else None
        ),
        "proposal_attempts": attempts,
        "execution_s": 0.001,
        "subprocess_executions": 64 if valid else 0,
    }


@pytest.fixture
def completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run the real freeze/summary code around synthetic panels and 24 explicit fake responses."""
    tasks = {
        split: G.fresh_tasks("F1_UNIT_ANALYSIS", split, 1)
        for split in ("train", "validation")
    }
    monkeypatch.setattr(F, "ROOT", tmp_path)
    monkeypatch.setattr(
        G,
        "fresh_tasks",
        lambda namespace, split, replicates: copy.deepcopy(tasks[split]),
    )
    monkeypatch.setattr(F, "environment", dict)
    monkeypatch.setattr(
        F,
        "_g1_result",
        lambda: {
            "caps": {
                "8000": {"length": 1, "eligible": 8},
                "32000": {"length": 0, "eligible": 10},
            }
        },
    )
    monkeypatch.setattr(
        F,
        "evaluate_panel",
        lambda source, tasks, block, deployment=False: [
            unit_row(source, task, block, deployment) for task in tasks
        ],
    )
    frozen = F.prepare()
    for index, request in enumerate(frozen["requests"]):
        directory = tmp_path / "raw" / request["slot_id"]
        source = "" if request["condition"] == "legacy_code" else B.SEED_SOURCE
        started = frozen["created_ns"] + index * 10000
        I.persist(directory / "request.json", request)
        I.persist(
            directory / "started_1.json", {"time_ns": started, "status": "in_flight"}
        )
        I.persist(
            directory / "attempt_1.json",
            {"status": "completed", "id": f"unit-{index}", "wall_s": 0.001},
        )
        I.persist(
            directory / "response.json",
            {
                "completed": True,
                "completed_ns": started + 100,
                "id": f"unit-{index}",
                "model": G.MODEL,
                "source": source,
                "source_sha256": B.source_hash(source),
                "content": source if source else None,
                "parse_status": "parsed" if source else "unparsable",
                "source_status": B.source_status(source),
                "finish_reason": "stop",
                "attempt": 1,
                "wall_s": 0.001,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        )
        I.persist(
            directory / "training.json",
            F.evaluate_panel(source, tasks["train"], request["block"]),
        )
    F.validate()
    F.summarize()
    return frozen


def test_complete_audit_keeps_every_invalid_response_and_fallback(
    completed_run: dict[str, Any],
) -> None:
    """One failed condition remains six full deployment comparisons, not omitted samples."""
    result = A.analyze()
    assert len(result["rows"]) == 24
    assert result["conditions"]["legacy_code"]["responses"] == 6
    assert result["conditions"]["legacy_code"]["training_eligible"] == 0
    assert result["conditions"]["legacy_code"]["fallback_trajectories"] == 36
    assert all(row["validation_auc"] is not None for row in result["rows"])
    assert result["receipt_coverage"]["present"] == 0
    assert result["receipt_coverage"]["reported_cost_sum"] is None
    assert result["usage"]["cost_usd"]["reported_sum"] is None
    assert all(len(value["deltas"]) == 6 for value in result["contrasts"].values())
    assert (
        result["total_allocation"]["candidate_train_validation_objective_calls"] == 9216
    )


@pytest.mark.parametrize(
    "missing", ["results.json", "evaluations_frozen.json", "validation_opened.json"]
)
def test_incomplete_final_state_blocks_all_efficacy_analysis(
    completed_run: dict[str, Any], missing: str
) -> None:
    """Existing partial outcomes are never enough to trigger final condition ranking."""
    (F.ROOT / missing).unlink()
    with pytest.raises(RuntimeError, match="complete"):
        A.analyze()


@pytest.mark.parametrize(
    "tamper", ["source", "request", "evaluation_seal", "summary", "chronology"]
)
def test_corrupted_evidence_or_result_summary_is_rejected(
    completed_run: dict[str, Any], tamper: str
) -> None:
    """No altered source, asymmetric request, omitted row or late response enters the report."""
    request = completed_run["requests"][0]
    directory = F.ROOT / "raw" / request["slot_id"]
    if tamper in {"source", "chronology"}:
        path = directory / "response.json"
        value = E.read(path)
        value["source"] += "# changed\n" if tamper == "source" else ""
        if tamper == "chronology":
            value["completed_ns"] = (
                E.read(F.ROOT / "validation_opened.json")["time_ns"] + 1
            )
    elif tamper == "request":
        path = directory / "request.json"
        value = E.read(path)
        value["settings"]["temperature"] = 1.0
    elif tamper == "evaluation_seal":
        path = F.ROOT / "evaluations_frozen.json"
        value = E.read(path)
        value.pop(next(iter(value)))
    else:
        path = F.ROOT / "results.json"
        value = E.read(path)
        value["rows"].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError):
        A.analyze()


def test_fallback_attempts_preserve_cursor_and_never_switch_back() -> None:
    """Failures consume no objective value and the seed receives the existing budget cursor."""
    task = G.fresh_tasks("F1_UNIT_ATTEMPTS", "train", 1)[0]
    row = unit_row("", task, 1, True)
    assert A.verify_attempts(row, deployment=True)["failures"] == 1
    row["proposal_attempts"][2]["policy"] = "candidate"
    with pytest.raises(RuntimeError, match="fallback|policy"):
        A.verify_attempts(row, deployment=True)
    row = unit_row("", task, 1, True)
    row["proposal_attempts"][1]["evaluation_index"] = 1
    with pytest.raises(RuntimeError, match="cursor|budget"):
        A.verify_attempts(row, deployment=True)


def test_receipt_must_belong_to_the_same_completed_response(
    completed_run: dict[str, Any],
) -> None:
    """Missing bills are explicit; a different generation's receipt is an integrity error."""
    request = completed_run["requests"][0]
    I.persist(
        F.ROOT / "raw" / request["slot_id"] / "provider_generation.json",
        {"id": "different"},
    )
    with pytest.raises(RuntimeError, match="receipt"):
        A.analyze()


def test_provider_token_inconsistency_is_flagged_without_scientific_invalidation() -> (
    None
):
    """Native accounting inconsistencies remain evidence, never a reason to replace a response."""
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 32000,
        "reasoning_tokens": 33818,
        "total_tokens": 32100,
    }
    original = copy.deepcopy(usage)
    issues = A.usage_issues(usage)
    assert "reasoning_exceeds_completion" in issues
    assert usage == original
    assert A.usage_issues(
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 123}
    ) == ["prompt_plus_completion_differs_from_total"]


def test_suspension_observation_preserves_recorded_latency(
    completed_run: dict[str, Any],
) -> None:
    """Independent clocks qualify elapsed time without changing the recorded request duration."""
    observation = {"additional_suspension_s": 5577.196, "source": "unit clock"}
    I.persist(F.ROOT / "clock_observation_02.json", observation)
    result = A.analyze()
    assert result["host_clock_observations"] == [observation]
    assert all(row["wall_s"] == 0.001 for row in result["rows"])


@pytest.mark.parametrize(
    "finish_reason,category",
    [
        ("error", "provider_error"),
        ("length", "token_limit"),
        ("stop", "normal_stop"),
        (None, "unreported"),
        ("tool_calls", "other_reported_finish"),
    ],
)
def test_termination_annotation_keeps_provider_error_distinct(
    finish_reason: str | None, category: str
) -> None:
    """Response termination is distinct from source parsing, syntax and execution validity."""
    assert A.termination_category(finish_reason) == category


def test_completed_provider_error_remains_one_slot_without_changing_metrics(
    completed_run: dict[str, Any],
) -> None:
    """An error finish is reported separately without dropping a completed null-source response."""
    before = A.analyze()
    request = next(
        row for row in completed_run["requests"] if row["condition"] == "legacy_code"
    )
    path = F.ROOT / "raw" / request["slot_id"] / "response.json"
    response = E.read(path)
    response["finish_reason"] = "error"
    path.write_text(json.dumps(response))
    barrier_path = F.ROOT / "validation_opened.json"
    barrier = E.read(barrier_path)
    barrier["responses"][request["slot_id"]] = B.digest(response)
    barrier_path.write_text(json.dumps(barrier))
    after = A.analyze()
    row = next(row for row in after["rows"] if row["slot_id"] == request["slot_id"])
    assert row["termination_category"] == "provider_error"
    assert row["source_status"] == "missing_source"
    assert row["parse_status"] == "unparsable"
    assert row["transport_attempts"] == 1
    assert len(after["rows"]) == 24
    assert after["contrasts"] == before["contrasts"]
    assert after["conditions"]["legacy_code"]["finish_reasons"] == {
        "error": 1,
        "stop": 5,
    }
    assert after["conditions"]["legacy_code"]["termination_categories"] == {
        "provider_error": 1,
        "normal_stop": 5,
    }
