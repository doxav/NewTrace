"""Preserve T1 and replace only its independently verified suspension-affected timing."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.throughput import run_throughput as T

ROOT = Path(__file__).resolve().parent


def calendar_gap(timing: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    """Compare recorded realtime trajectory span with monotonic condition elapsed time."""
    if not rows:
        raise ValueError("timing audit needs trajectory timestamp evidence")
    return (
        max(row["completed_ns"] for row in rows)
        - min(row["started_ns"] for row in rows)
    ) / 1e9 - timing["wall_s"]


def main() -> None:
    """Freeze and run one predetermined serial replacement with exact scientific replay."""
    original = T.preflight()
    old_results = E.read(ROOT / "results.json")
    audit = []
    for condition in original["conditions"]:
        root = T.directory(condition)
        rows = [E.read(root / f"{job['id']}.json") for job in original["jobs"]]
        timing = E.read(root / "timing.json")
        gap = calendar_gap(timing, rows)
        audit.append(
            {
                **condition,
                "calendar_minus_monotonic_s": gap,
                "suspension_or_clock_gap": abs(gap) > 1.0,
            }
        )
    E.persist(
        ROOT / "timing_audit.json",
        {
            "conditions": audit,
            "original_result_sha256": hashlib.sha256(
                (ROOT / "results.json").read_bytes()
            ).hexdigest(),
            "user_reported_machine_suspension": True,
        },
    )
    affected = [row for row in audit if row["suspension_or_clock_gap"]]
    if len(affected) != 1 or (affected[0]["round"], affected[0]["workers"]) != (0, 1):
        raise RuntimeError(
            "timing repair is authorized only for the identified serial condition"
        )
    repair_root = ROOT / "timing_repair_r1"
    frozen_path = repair_root / "freeze.json"
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), ROOT / "TIMING_REPAIR_R1.md")
    }
    frozen = {
        "stage": "T1-R1",
        "original_freeze_sha256": B.digest(original),
        "source_hashes": source_hashes,
        "condition": {"round": 0, "workers": 1},
        "trajectories": 24,
        "objective_calls": 768,
        "reference_calls": 768,
    }
    if E.exists(frozen_path):
        if E.read(frozen_path)["configuration"] != frozen:
            raise RuntimeError("timing repair freeze changed")
    else:
        E.persist(frozen_path, {"frozen_ns": time.time_ns(), "configuration": frozen})
    for task in original["tasks"]:
        B.normalization(task)
    old_root = T.ROOT
    try:
        T.ROOT = repair_root
        T.run_condition(original, frozen["condition"])
        replacement_dir = T.directory(frozen["condition"])
    finally:
        T.ROOT = old_root
    replacement = E.read(replacement_dir / "timing.json")
    rows = [E.read(replacement_dir / f"{job['id']}.json") for job in original["jobs"]]
    reference_dir = T.directory(frozen["condition"])
    mismatches = []
    for job, actual in zip(original["jobs"], rows):
        expected = E.read(reference_dir / f"{job['id']}.json")
        if (
            actual["job"] != job
            or actual["status"] != "completed"
            or T.scientific_projection(actual.get("result", {}))
            != T.scientific_projection(expected.get("result", {}))
        ):
            mismatches.append(job["id"])
    gap = calendar_gap(replacement, rows)
    healthy = not mismatches and abs(gap) <= 1 and replacement["timing_eligible"]
    corrected = [
        replacement if (row["round"], row["workers"]) == (0, 1) else row
        for row in old_results["timings"]
    ]
    summary = []
    baseline = statistics.mean(
        row["wall_s"] for row in corrected if row["workers"] == 1
    )
    for workers in (1, 4, 8, 16):
        times = [row["wall_s"] for row in corrected if row["workers"] == workers]
        mean = statistics.mean(times)
        summary.append(
            {
                "workers": workers,
                "wall_s": times,
                "mean_wall_s": mean,
                "mean_s_per_trajectory": mean / 24,
                "speedup_vs_healthy_serial": baseline / mean,
            }
        )
    result = {
        "status": (
            "HEALTHY_TIMING_REPLACEMENT"
            if healthy
            else "TIMING_REPLACEMENT_NOT_VALIDATED"
        ),
        "original_evidence_preserved": True,
        "replacement_calendar_minus_monotonic_s": gap,
        "scientific_mismatches": mismatches,
        "all_24_scientific_results_identical": not mismatches,
        "new_trajectories": 24,
        "objective_calls": sum(
            row["result"]["objective_calls"]
            for row in rows
            if row["status"] == "completed"
        ),
        "subprocess_executions": sum(
            row["result"]["subprocess_executions"]
            for row in rows
            if row["status"] == "completed"
        ),
        "reference_objective_calls": 768,
        "corrected_summaries": summary if healthy else None,
        "original_unadjusted_summaries": old_results["summaries"],
    }
    E.persist(repair_root / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
