"""Pure, bounded projection of previously available TRAIN attempt receipts.

The caller authenticates immutable response and evaluation files and freezes this
projection before requesting a proposal. This module checks their supplied
identities, chronology and closed schema; it neither reads files nor evaluates
candidates. Source text is preserved whole or explicitly omitted. A TRAIN scalar
is permitted only for a complete, valid panel. No per-task data enter the prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

SCHEMA = "exp18-attempt-memory-v1"
_SOURCE_STATUSES = frozenset(
    {"valid", "missing_source", "source_size", "syntax_error", "protocol_violation"}
)
_STATUSES = _SOURCE_STATUSES | {
    "timeout",
    "exception",
    "import_error",
    "missing_propose",
    "signature_error",
    "process_error",
    "nondeterministic",
    "shape_error",
    "nonfinite",
    "out_of_bounds",
}
_RECORD_FIELDS = {
    "arm",
    "outer",
    "slot",
    "slot_id",
    "response_sha256",
    "parent_sha256",
    "source",
    "source_sha256",
    "source_status",
    "completed_ns",
    "train",
}
_TRAIN_FIELDS = {
    "split",
    "source_sha256",
    "observed_ns",
    "evidence_sha256",
    "allocated_trajectories",
    "observed_trajectories",
    "valid_trajectories",
    "status_counts",
    "aggregate_auc",
}


def _serialize(value: Any) -> str:
    """Use one exact JSON encoding for prompt size and provenance digests."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _hash_text(value: str) -> str:
    """Hash exact UTF-8 text without parsing or altering generated source."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_hash(value: Any) -> None:
    """Require a canonical SHA-256 digest rather than an arbitrary identifier."""
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("provenance requires a canonical SHA-256 digest")


def _require_integer(value: Any, name: str, *, minimum: int = 0) -> None:
    """Exclude booleans, negative counts and ambiguous timestamp types."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def project_training(
    receipt: dict[str, Any] | None,
    *,
    source_sha256: str,
    completed_ns: int,
    snapshot_ns: int,
) -> dict[str, Any]:
    """Validate an available TRAIN receipt and return only permitted aggregate data.

    The observation timestamp refers to when this response's panel was available
    to its owner, including authenticated cache reuse. It must precede the frozen
    request snapshot. Missing receipts remain unavailable; no work is triggered.
    """
    _require_hash(source_sha256)
    _require_integer(completed_ns, "response completion")
    _require_integer(snapshot_ns, "snapshot timestamp", minimum=1)
    if completed_ns >= snapshot_ns:
        raise ValueError("response completion must precede the snapshot")
    if receipt is None:
        return {"state": "unavailable"}
    if not isinstance(receipt, dict) or set(receipt) != _TRAIN_FIELDS:
        raise ValueError("TRAIN receipt fields differ from the closed schema")
    if receipt["split"] != "train" or receipt["source_sha256"] != source_sha256:
        raise ValueError("TRAIN receipt split or source identity mismatch")
    _require_hash(receipt["evidence_sha256"])
    observed_ns = receipt["observed_ns"]
    _require_integer(observed_ns, "TRAIN availability timestamp", minimum=1)
    if not completed_ns <= observed_ns < snapshot_ns:
        raise ValueError(
            "TRAIN receipt was not available between response and snapshot"
        )
    allocated = receipt["allocated_trajectories"]
    observed = receipt["observed_trajectories"]
    valid = receipt["valid_trajectories"]
    _require_integer(allocated, "allocated trajectories", minimum=1)
    _require_integer(observed, "observed trajectories")
    _require_integer(valid, "valid trajectories")
    counts = receipt["status_counts"]
    if not isinstance(counts, dict) or any(
        status not in _STATUSES for status in counts
    ):
        raise ValueError("TRAIN status counts require declared typed statuses")
    for count in counts.values():
        _require_integer(count, "status count", minimum=1)
    if (
        not 0 <= valid <= observed <= allocated
        or sum(counts.values()) != observed
        or counts.get("valid", 0) != valid
    ):
        raise ValueError("TRAIN trajectory and status counts disagree")
    auc = receipt["aggregate_auc"]
    complete_valid = valid == observed == allocated
    if complete_valid:
        if type(auc) not in (int, float) or not math.isfinite(auc) or auc < 0:
            raise ValueError(
                "complete valid TRAIN aggregate must be finite and nonnegative"
            )
    elif auc is not None:
        raise ValueError(
            "invalid or partial TRAIN panel cannot have an aggregate score"
        )
    return {
        "state": "complete" if observed == allocated else "partial",
        "allocated_trajectories": allocated,
        "observed_trajectories": observed,
        "valid_trajectories": valid,
        "status_counts": dict(sorted(counts.items())),
        "aggregate_auc": auc,
    }


def _validate_record(
    record: dict[str, Any], *, arm: str, outer: int, before_slot: int, snapshot_ns: int
) -> dict[str, Any]:
    """Reject cross-arm, future, hidden-field or unaligned source/receipt records."""
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise ValueError("attempt record fields differ from the closed schema")
    if (
        record["arm"] != arm
        or type(record["outer"]) is not int
        or record["outer"] != outer
    ):
        raise ValueError("memory record arm or outer seed mismatch")
    _require_integer(record["slot"], "prior slot")
    if record["slot"] >= before_slot:
        raise ValueError("memory can contain only a prior slot")
    if not isinstance(record["slot_id"], str) or not record["slot_id"]:
        raise ValueError("memory requires a nonempty slot identity")
    for key in ("response_sha256", "parent_sha256", "source_sha256"):
        _require_hash(record[key])
    source = record["source"]
    if not isinstance(source, str) or _hash_text(source) != record["source_sha256"]:
        raise ValueError("exact candidate source and declared hash disagree")
    status = record["source_status"]
    if not isinstance(status, str) or status not in _SOURCE_STATUSES:
        raise ValueError("source status is outside the declared source screen")
    if (not source.strip()) != (status == "missing_source"):
        raise ValueError("missing-source status disagrees with exact source")
    summary = project_training(
        record["train"],
        source_sha256=record["source_sha256"],
        completed_ns=record["completed_ns"],
        snapshot_ns=snapshot_ns,
    )
    if status != "valid" and summary.get("valid_trajectories", 0):
        raise ValueError("invalid source cannot have valid TRAIN trajectories")
    return summary


def build_memory(
    records: list[dict[str, Any]],
    *,
    arm: str,
    outer: int,
    before_slot: int,
    snapshot_ns: int,
    current_parent_sha256: str,
    max_sources: int = 7,
    max_chars: int = 32768,
) -> dict[str, Any]:
    """Freeze compact prior-attempt memory without model, file or evaluator access.

    All completed slots below ``before_slot`` are mandatory. At most seven unique
    nonempty sources other than the displayed parent are considered, newest slot
    first. Metadata for every prior slot is always retained. Whole source texts
    are then included greedily in that order if their exact JSON fits the total
    character budget. Omissions and duplicates remain explicit. If metadata alone
    exceeds the budget, raise instead of silently dropping an attempted slot.

    ``provenance`` and ``snapshot`` are host-only evidence. Only ``text`` is meant
    for the generating model. Source screening and authentication of raw response
    or receipt files remain the caller's responsibility; no rejection decision is
    inferred from an attempt's source being different from the current parent.
    """
    if not isinstance(records, list) or not isinstance(arm, str) or not arm:
        raise ValueError("memory requires a record list and a nonempty arm")
    _require_integer(outer, "outer seed")
    _require_integer(before_slot, "prior slot cutoff")
    _require_integer(snapshot_ns, "snapshot timestamp", minimum=1)
    _require_integer(max_sources, "source limit")
    _require_integer(max_chars, "character budget", minimum=1)
    if max_sources > 7:
        raise ValueError("memory source limit cannot exceed seven")
    _require_hash(current_parent_sha256)
    summaries = {}
    for record in records:
        summary = _validate_record(
            record,
            arm=arm,
            outer=outer,
            before_slot=before_slot,
            snapshot_ns=snapshot_ns,
        )
        summaries[record["slot"]] = summary
    if sorted(record["slot"] for record in records) != list(range(before_slot)):
        raise ValueError("every prior slot must be present exactly once")
    if len({record["slot_id"] for record in records}) != len(records):
        raise ValueError("memory slot identities must be unique")
    ordered = sorted(records, key=lambda record: record["slot"], reverse=True)
    unique_sources: dict[str, dict[str, Any]] = {}
    for record in ordered:
        if (
            record["source"].strip()
            and record["source_sha256"] != current_parent_sha256
        ):
            unique_sources.setdefault(record["source_sha256"], record)
    selected = list(unique_sources)[:max_sources]
    displays = {
        digest: (
            "omitted_character_budget" if digest in selected else "omitted_source_limit"
        )
        for digest in unique_sources
    }
    source_entries = [
        {
            "source_sha256": digest,
            "latest_slot": unique_sources[digest]["slot"],
            "display": displays[digest],
            "source": None,
        }
        for digest in selected
    ]

    def payload() -> dict[str, Any]:
        """Rebuild the bounded JSON projection using current whole-source decisions."""
        return {
            "schema": SCHEMA,
            "meaning": "Prior attempt execution summaries; source differences do not imply rejection. Lower aggregate TRAIN AUC is better.",
            "current_parent_sha256": current_parent_sha256,
            "attempts": [
                {
                    "slot": record["slot"],
                    "source_sha256": record["source_sha256"],
                    "parent_sha256": record["parent_sha256"],
                    "source_status": record["source_status"],
                    "source_display": (
                        "no_source"
                        if not record["source"].strip()
                        else (
                            "current_parent"
                            if record["source_sha256"] == current_parent_sha256
                            else displays[record["source_sha256"]]
                        )
                    ),
                    "train": summaries[record["slot"]],
                }
                for record in ordered
            ],
            "sources": source_entries,
        }

    text = _serialize(payload())
    # For very short sources, including the whole text needs fewer characters
    # than the explicit omission labels. Reserve the smallest faithful metadata
    # representation so a whole-source exact fit is never rejected incorrectly.
    for entry in source_entries:
        digest = entry["source_sha256"]
        entry["source"] = unique_sources[digest]["source"]
        entry["display"] = displays[digest] = "full"
        candidate = _serialize(payload())
        if len(candidate) <= len(text):
            text = candidate
        else:
            entry["source"] = None
            entry["display"] = displays[digest] = "omitted_character_budget"
    if len(text) > max_chars:
        raise ValueError("all prior-slot metadata exceed the memory character budget")
    for entry in source_entries:
        if entry["display"] == "full":
            continue
        digest = entry["source_sha256"]
        entry["source"] = unique_sources[digest]["source"]
        entry["display"] = displays[digest] = "full"
        candidate = _serialize(payload())
        if len(candidate) <= max_chars:
            text = candidate
        else:
            entry["source"] = None
            entry["display"] = displays[digest] = "omitted_character_budget"
    # Failed inclusion attempts restore the prior state; serialize once more so
    # returned text cannot accidentally reflect a rejected oversized candidate.
    text = _serialize(payload())
    included = [
        entry["source"] for entry in source_entries if entry["display"] == "full"
    ]
    nonempty = [
        record["source_sha256"] for record in ordered if record["source"].strip()
    ]
    return {
        "schema": SCHEMA,
        "text": text,
        "text_sha256": _hash_text(text),
        "snapshot": {
            "arm": arm,
            "outer": outer,
            "before_slot": before_slot,
            "snapshot_ns": snapshot_ns,
            "current_parent_sha256": current_parent_sha256,
            "max_sources": max_sources,
            "max_chars": max_chars,
            "records_sha256": _hash_text(_serialize(ordered)),
        },
        "provenance": [
            {
                "slot": record["slot"],
                "slot_id": record["slot_id"],
                "response_sha256": record["response_sha256"],
                "source_sha256": record["source_sha256"],
                "parent_sha256": record["parent_sha256"],
                "completed_ns": record["completed_ns"],
                "train_observed_ns": (
                    record["train"]["observed_ns"]
                    if record["train"] is not None
                    else None
                ),
                "train_evidence_sha256": (
                    record["train"]["evidence_sha256"]
                    if record["train"] is not None
                    else None
                ),
                "record_sha256": _hash_text(_serialize(record)),
            }
            for record in ordered
        ],
        "counts": {
            "prior_slots": len(records),
            "distinct_nonparent_sources": len(unique_sources),
            "duplicate_source_slots": len(nonempty) - len(set(nonempty)),
            "current_parent_slots": sum(
                record["source_sha256"] == current_parent_sha256 for record in ordered
            ),
            "no_source_slots": sum(not record["source"].strip() for record in ordered),
            "full_sources": len(included),
            "omitted_source_limit_sources": sum(
                display == "omitted_source_limit" for display in displays.values()
            ),
            "omitted_character_budget_sources": sum(
                display == "omitted_character_budget" for display in displays.values()
            ),
            "source_chars_included": sum(len(source) for source in included),
            "source_bytes_included": sum(
                len(source.encode("utf-8")) for source in included
            ),
            "prompt_chars": len(text),
            "prompt_bytes": len(text.encode("utf-8")),
        },
    }
