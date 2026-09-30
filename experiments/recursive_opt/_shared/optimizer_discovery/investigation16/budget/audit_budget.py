"""Describe all observed EXP-15 proposal prefixes using train/validation only."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parents[1] / "exp15/raw"


def prefix(pool: list[dict[str, Any]], size: int) -> dict[str, Any]:
    """Select an observed prefix without dropping completed invalid response slots."""
    if size not in range(1, 9):
        raise ValueError("observed prefix size must be 1 through 8")
    observed = [row for row in pool if row["index"] < size]
    if {row["index"] for row in observed} != {-1, *range(size)}:
        raise ValueError("prefix is missing a completed registered response slot")
    eligible = [row for row in observed if row["eligible"]]
    if not eligible or not any(row["index"] == -1 for row in eligible):
        raise ValueError("trusted seed must remain eligible")
    if not all(math.isfinite(row["validation_auc"]) for row in eligible):
        raise ValueError("eligible source lacks a finite validation metric")
    selected = min(eligible, key=lambda row: (row["validation_auc"], row["index"]))
    generated = [row for row in eligible if row["index"] >= 0]
    seed = next(row for row in eligible if row["index"] == -1)
    return {
        "completed_slots": size,
        "eligible_generated": len(generated),
        "unique_generated_sources_excluding_seed": len(
            {row["source_sha256"] for row in generated} - {seed["source_sha256"]}
        ),
        "no_eligible_generated": not generated,
        "selected_index": selected["index"],
        "selected_source_sha256": selected["source_sha256"],
        "seed_source_selected": selected["source_sha256"] == seed["source_sha256"],
        "validation_auc": selected["validation_auc"],
    }


def behavior_hash(candidate: dict[str, Any]) -> str:
    """Identify exact observed training behavior, excluding source/runtime metadata."""
    observations = [row["observations"] for row in candidate["train"]]
    return hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest()


def main() -> None:
    """Persist all prefix rows and caveated availability/budget summaries."""
    rows = []
    inputs = {}
    saturation = []
    for outer in (11, 23, 37, 41, 53):
        for arm in ("A1", "A2"):
            path = RAW / str(outer) / arm / "pool.json.gz"
            payload = path.read_bytes()
            inputs[str(path.relative_to(RAW))] = hashlib.sha256(payload).hexdigest()
            pool = json.loads(gzip.decompress(payload))
            if {row["index"] for row in pool} != {-1, *range(8)}:
                raise RuntimeError(
                    "complete raw pool does not preserve all eight slots"
                )
            seed = next(row for row in pool if row["index"] == -1)
            seed_behavior = behavior_hash(seed)
            block = []
            for size in range(1, 9):
                row = {"outer_seed": outer, "arm": arm, **prefix(pool, size)}
                generated = [
                    candidate
                    for candidate in pool
                    if 0 <= candidate["index"] < size and candidate["eligible"]
                ]
                row["unique_generated_train_behaviors_excluding_seed"] = len(
                    {behavior_hash(candidate) for candidate in generated}
                    - {seed_behavior}
                )
                block.append(row)
            saturation.append(
                {
                    "outer_seed": outer,
                    "arm": arm,
                    "first_size_attaining_final_validation_minimum": next(
                        row["completed_slots"]
                        for row in block
                        if row["validation_auc"] == block[-1]["validation_auc"]
                    ),
                    "source_selection_changes_after_first_slot": sum(
                        left["selected_source_sha256"]
                        != right["selected_source_sha256"]
                        for left, right in pairwise(block)
                    ),
                }
            )
            rows.extend(block)
    summaries = []
    for arm in ("A1", "A2"):
        for size in range(1, 9):
            block = [
                row
                for row in rows
                if row["arm"] == arm and row["completed_slots"] == size
            ]
            summaries.append(
                {
                    "arm": arm,
                    "completed_slots": size,
                    "mean_validation_auc": statistics.mean(
                        row["validation_auc"] for row in block
                    ),
                    "mean_eligible_generated": statistics.mean(
                        row["eligible_generated"] for row in block
                    ),
                    "mean_unique_generated_sources_excluding_seed": statistics.mean(
                        row["unique_generated_sources_excluding_seed"] for row in block
                    ),
                    "mean_unique_generated_train_behaviors_excluding_seed": statistics.mean(
                        row["unique_generated_train_behaviors_excluding_seed"]
                        for row in block
                    ),
                    "no_eligible_generated_outer_seeds": sum(
                        row["no_eligible_generated"] for row in block
                    ),
                    "seed_source_selected_outer_seeds": sum(
                        row["seed_source_selected"] for row in block
                    ),
                    "seed_selected_despite_eligible_outer_seeds": sum(
                        row["seed_source_selected"] and not row["no_eligible_generated"]
                        for row in block
                    ),
                }
            )
    hypothetical = {}
    for arm in ("A1", "A2"):
        full = next(
            row
            for row in summaries
            if row["arm"] == arm and row["completed_slots"] == 8
        )
        invalid = 1 - full["mean_eligible_generated"] / 8
        hypothetical[arm] = {
            "observed_invalid_fraction": invalid,
            "hypothetical_iid_probability_all_eight_invalid": invalid**8,
            "observed_empty_pools": full["no_eligible_generated_outer_seeds"],
            "outer_seed_denominator": 5,
            "zero_of_five_one_sided_exact_95pct_upper_bound": 1 - 0.05**0.2,
        }
    output = {
        "status": "RETROSPECTIVE_EXPLORATORY_TRAIN_VALIDATION_ONLY",
        "new_model_calls": 0,
        "new_objective_calls": 0,
        "holdout_files_read": 0,
        "rows": rows,
        "summaries": summaries,
        "saturation": saturation,
        "hypothetical_availability": hypothetical,
        "input_sha256": inputs,
        "interpretation_caveat": "Validation improvement with nested candidate pools is monotone by construction. No N>8 benefit, holdout gain, response independence or algorithmic equivalence is inferred.",
    }
    (ROOT / "results.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
