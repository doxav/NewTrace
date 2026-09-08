"""Descriptive EXP-15 reporting repairs; never used by generation or selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from artifacts.optimizer_discovery.exp15 import read


def attempt_timing(directory: Path) -> dict[str, Any]:
    """Associate response latency with its actual attempt and separate total slot time."""
    response = read(directory / "response.json")
    attempt_id = response["attempt"]
    matching_start = read(directory / f"started_{attempt_id}.json")["time_ns"]
    starts = [read(p)["time_ns"] for p in directory.glob("started_*.json")]
    attempts = [read(p) for p in directory.glob("attempt_*.json")]
    if (
        not starts
        or not attempts
        or sum(a["status"] == "completed" for a in attempts) != 1
    ):
        raise ValueError(
            "timing requires exactly one completed response and its attempt receipts"
        )
    slot_wall = (response["completed_ns"] - min(starts)) / 1e9
    successful_wall = (response["completed_ns"] - matching_start) / 1e9
    measured = sum(a["wall_s"] for a in attempts)
    if successful_wall < 0 or slot_wall < 0:
        raise ValueError("response precedes its registered attempt")
    return {
        "completed_attempt": attempt_id,
        "transport_attempts": len(attempts),
        "transport_failures": sum(a["status"] == "transport_failure" for a in attempts),
        "slot_wall_s": slot_wall,
        "successful_attempt_wall_s": successful_wall,
        "successful_attempt_monotonic_s": response["wall_s"],
        "measured_attempts_s": measured,
        "unattributed_wall_gap_s": slot_wall - measured,
        "gap_interpretation": "May include system suspension, backoff, scheduling and resume gaps; do not attribute it solely to provider latency.",
    }
