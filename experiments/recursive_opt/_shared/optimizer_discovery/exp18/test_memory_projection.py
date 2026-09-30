"""Pure memory provenance and information-boundary tests; no model or evaluator."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery.exp18 import memory_projection as M


def digest(text: str) -> str:
    """Hash exact test text without invoking research or candidate code."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record(slot: int, source: str | None = None) -> dict[str, Any]:
    """Construct a completed response with an aligned available TRAIN receipt."""
    source = (
        f"def propose(history, bounds, seed):\n    return [{slot}.0]\n"
        if source is None
        else source
    )
    source_hash = digest(source)
    return {
        "arm": "M",
        "outer": 18111,
        "slot": slot,
        "slot_id": f"TEST_18111_M_{slot:02d}",
        "response_sha256": digest(f"response-{slot}"),
        "parent_sha256": digest("seed"),
        "source": source,
        "source_sha256": source_hash,
        "source_status": "valid" if source.strip() else "missing_source",
        "completed_ns": 100 + slot * 10,
        "train": {
            "split": "train",
            "source_sha256": source_hash,
            "observed_ns": 101 + slot * 10,
            "evidence_sha256": digest(f"receipt-{slot}"),
            "allocated_trajectories": 4,
            "observed_trajectories": 4,
            "valid_trajectories": 4,
            "status_counts": {"valid": 4},
            "aggregate_auc": 0.5 + slot,
        },
    }


def project(records: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Apply one fixed test snapshot with caller-selected budget or current parent."""
    arguments = {
        "arm": "M",
        "outer": 18111,
        "before_slot": len(records),
        "snapshot_ns": 1000,
        "current_parent_sha256": digest("seed"),
        "max_sources": 7,
        "max_chars": 32768,
    }
    arguments.update(kwargs)
    return M.build_memory(records, **arguments)


def test_empty_memory_is_explicit_and_deterministic() -> None:
    result = project([])
    prompt = json.loads(result["text"])
    assert prompt["attempts"] == []
    assert prompt["sources"] == []
    assert result["counts"]["prior_slots"] == 0
    assert result["text_sha256"] == digest(result["text"])
    assert result == project([])


def test_recency_deduplication_current_parent_and_source_limit() -> None:
    rows = [record(slot) for slot in range(10)]
    rows.append(record(10, rows[9]["source"]))
    original = copy.deepcopy(rows)
    result = project(rows, current_parent_sha256=rows[8]["source_sha256"])
    prompt = json.loads(result["text"])
    assert [item["slot"] for item in prompt["attempts"]] == list(range(10, -1, -1))
    assert [item["latest_slot"] for item in prompt["sources"]] == [10, 7, 6, 5, 4, 3, 2]
    assert prompt["attempts"][0]["source_display"] == "full"
    assert prompt["attempts"][1]["source_display"] == "full"
    assert prompt["attempts"][2]["source_display"] == "current_parent"
    assert prompt["attempts"][-1]["source_display"] == "omitted_source_limit"
    assert result["counts"]["duplicate_source_slots"] == 1
    assert result["counts"]["full_sources"] == 7
    assert rows == original
    assert (
        project(list(reversed(rows)), current_parent_sha256=rows[8]["source_sha256"])
        == result
    )


def test_invalid_and_empty_response_slots_are_not_dropped_or_scored() -> None:
    invalid = record(0, "def propose(:")
    invalid["source_status"] = "syntax_error"
    invalid["train"].update(
        valid_trajectories=0, status_counts={"syntax_error": 4}, aggregate_auc=None
    )
    missing = record(1, "")
    missing["train"].update(
        valid_trajectories=0, status_counts={"missing_source": 4}, aggregate_auc=None
    )
    result = project([invalid, missing])
    prompt = json.loads(result["text"])
    assert len(prompt["attempts"]) == 2
    assert prompt["attempts"][0]["source_display"] == "no_source"
    assert prompt["attempts"][1]["train"]["status_counts"] == {"syntax_error": 4}
    assert all(
        attempt["train"]["aggregate_auc"] is None for attempt in prompt["attempts"]
    )
    assert prompt["sources"][0]["source"] == invalid["source"]
    assert result["counts"]["full_sources"] == 1
    assert result["counts"]["no_source_slots"] == 1


def test_unavailable_and_partial_training_are_explicit() -> None:
    unavailable, partial = record(0), record(1)
    unavailable["train"] = None
    partial["train"].update(
        observed_trajectories=2,
        valid_trajectories=1,
        status_counts={"valid": 1, "exception": 1},
        aggregate_auc=None,
    )
    prompt = json.loads(project([unavailable, partial])["text"])
    assert prompt["attempts"][1]["train"] == {"state": "unavailable"}
    assert prompt["attempts"][0]["train"]["state"] == "partial"
    assert prompt["attempts"][0]["train"]["aggregate_auc"] is None


def test_complete_source_or_explicit_omission_under_character_budget() -> None:
    long_source = "# résumé\n" * 2000
    rows = [record(0, long_source), record(1, "short source")]
    result = project(rows, max_chars=5000)
    prompt = json.loads(result["text"])
    assert len(result["text"]) <= 5000
    assert prompt["sources"][0]["source"] == "short source"
    assert prompt["sources"][1]["source"] is None
    assert prompt["sources"][1]["display"] == "omitted_character_budget"
    assert prompt["attempts"][1]["source_display"] == "omitted_character_budget"
    assert result["counts"]["full_sources"] == 1
    assert result["counts"]["omitted_character_budget_sources"] == 1
    assert result["counts"]["source_bytes_included"] == len(b"short source")
    with pytest.raises(ValueError, match="metadata"):
        project(rows, max_chars=1)


def test_exact_fit_and_unicode_hash_are_counted_as_declared() -> None:
    rows = [record(0, "# été\n")]
    complete = project(rows)
    assert project(rows, max_chars=len(complete["text"]))["text"] == complete["text"]
    assert complete["counts"]["source_chars_included"] == 6
    assert complete["counts"]["source_bytes_included"] == 8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arm", "I"),
        ("outer", 18112),
        ("slot", 1),
        ("source_sha256", digest("another source")),
        ("parent_sha256", "bad"),
        ("response_sha256", "bad"),
        ("completed_ns", 1001),
        ("source_status", "rejected"),
    ],
)
def test_record_identity_mismatches_are_rejected(field: str, value: Any) -> None:
    row = record(0)
    row[field] = value
    with pytest.raises(ValueError):
        project([row])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "validation"),
        ("source_sha256", digest("another source")),
        ("observed_ns", 1001),
        ("observed_ns", 99),
        ("evidence_sha256", "bad"),
        ("allocated_trajectories", 0),
        ("observed_trajectories", 5),
        ("valid_trajectories", 3),
        ("status_counts", {"valid": 3}),
        ("status_counts", {"hidden_task_identity": 4}),
        ("aggregate_auc", float("nan")),
        ("aggregate_auc", -0.1),
        ("aggregate_auc", None),
    ],
)
def test_training_receipt_mismatches_are_rejected(field: str, value: Any) -> None:
    row = record(0)
    row["train"][field] = value
    with pytest.raises(ValueError):
        project([row])


def test_invalid_or_partial_panel_cannot_carry_numeric_metric() -> None:
    row = record(0)
    row["train"].update(
        valid_trajectories=3, status_counts={"valid": 3, "exception": 1}
    )
    with pytest.raises(ValueError, match="aggregate"):
        project([row])


@pytest.mark.parametrize("location", ["record", "train"])
def test_extra_fields_cannot_leak_hidden_information(location: str) -> None:
    row = record(0)
    target = row if location == "record" else row["train"]
    target["task_identity"] = "FORBIDDEN_TASK_MARKER"
    with pytest.raises(ValueError, match="fields"):
        project([row])


def test_all_prior_slots_and_unique_slot_identities_are_required() -> None:
    with pytest.raises(ValueError, match="prior slot"):
        project([record(0)], before_slot=2)
    with pytest.raises(ValueError, match="prior slot"):
        project([record(0), record(0)])
    rows = [record(0), record(1)]
    rows[1]["slot_id"] = rows[0]["slot_id"]
    with pytest.raises(ValueError, match="slot identities"):
        project(rows)


def test_host_provenance_is_hashed_without_entering_prompt() -> None:
    row = record(0)
    result = project([row])
    provenance = result["provenance"][0]
    assert provenance["slot_id"] == row["slot_id"]
    assert provenance["response_sha256"] == row["response_sha256"]
    assert provenance["train_evidence_sha256"] == row["train"]["evidence_sha256"]
    assert len(provenance["record_sha256"]) == 64
    assert "18111" not in result["text"]
    assert row["slot_id"] not in result["text"]
    assert row["train"]["evidence_sha256"] not in result["text"]
    assert "validation" not in result["text"]


def test_valid_source_execution_timeout_retains_failure_without_numeric_score() -> None:
    row = record(0)
    row["train"].update(
        valid_trajectories=0, status_counts={"timeout": 4}, aggregate_auc=None
    )
    attempt = json.loads(project([row])["text"])["attempts"][0]
    assert attempt["source_status"] == "valid"
    assert attempt["train"]["status_counts"] == {"timeout": 4}
    assert attempt["train"]["aggregate_auc"] is None


def test_zero_source_limit_keeps_every_attempt_and_no_complete_code() -> None:
    rows = [record(0), record(1)]
    result = project(rows, max_sources=0)
    prompt = json.loads(result["text"])
    assert len(prompt["attempts"]) == 2
    assert prompt["sources"] == []
    assert result["counts"]["omitted_source_limit_sources"] == 2
    assert result["counts"]["full_sources"] == 0


def test_current_seed_can_use_the_identical_compact_training_projection() -> None:
    row = record(0)
    summary = M.project_training(
        row["train"],
        source_sha256=row["source_sha256"],
        completed_ns=0,
        snapshot_ns=1000,
    )
    archived = json.loads(project([row])["text"])["attempts"][0]["train"]
    assert summary == archived


def test_same_snapshot_rebuild_is_independent_of_later_external_records() -> None:
    row = record(0)
    first = project([row])
    later = record(1)
    later["train"]["aggregate_auc"] = 1234.0
    assert project([row]) == first
    with pytest.raises(ValueError, match="prior slot"):
        project([row, later], before_slot=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_sources": 8},
        {"max_sources": True},
        {"max_chars": False},
        {"before_slot": -1},
        {"snapshot_ns": 0},
    ],
)
def test_invalid_projection_limits_fail_before_any_output(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        project([], **kwargs)
