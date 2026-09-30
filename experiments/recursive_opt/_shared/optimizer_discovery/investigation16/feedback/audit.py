"""Read-only EXP-15 feedback and search diagnostics, without holdout access."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("experiments/recursive_opt/_shared/optimizer_discovery/exp15/raw")
OUTPUT = Path(__file__).with_name("audit_results.json")
OUTER_SEEDS = (11, 23, 37, 41, 53)


def read_json(path: Path) -> Any:
    """Read the exact preserved JSON or its lossless compressed representation."""
    payload = (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
    )
    return json.loads(payload)


def feedback_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the frozen feedback projection for comparison to raw requests."""
    return {
        "valid": all(row["valid"] for row in rows),
        "tasks": [
            {
                "valid": row["valid"],
                "status": row["status"],
                "observations": row["observations"][:2] + row["observations"][-2:],
                "best_observed_value": min(
                    (item["value"] for item in row["observations"]), default=None
                ),
            }
            for row in rows
        ],
    }


def panel_metric(candidate: dict[str, Any], split: str, metric: str) -> float | None:
    """Average the six equally weighted family/dimension strata, retaining failures."""
    rows = candidate[split]
    if len(rows) != 6 or len({row["stratum"] for row in rows}) != 6:
        raise ValueError("Expected the frozen six-stratum panel")
    if not all(row["valid"] for row in rows):
        return None
    return statistics.mean(row["metrics"][metric] for row in rows)


def ranks(values: list[float]) -> list[float]:
    """Return average ranks with exact tie handling for descriptive correlations."""
    return [
        1
        + sum(other < value for other in values)
        + (sum(other == value for other in values) - 1) / 2
        for value in values
    ]


def correlation(left: list[float], right: list[float]) -> float | None:
    """Return Pearson correlation, or missing when either side is constant."""
    if len(left) != len(right) or not left:
        raise ValueError("Correlation requires matching nonempty vectors")
    a = [x - statistics.mean(left) for x in left]
    b = [x - statistics.mean(right) for x in right]
    scale = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / scale if scale else None


def analyze() -> dict[str, Any]:
    """Audit every request against train/validation evidence without reading holdout."""
    requests: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    source_files: dict[str, str] = {}
    for outer in OUTER_SEEDS:
        for arm in ("A1", "A2"):
            pool_path = ROOT / str(outer) / arm / "pool.json"
            candidates = read_json(pool_path)
            source_files[str(pool_path) + ".gz"] = hashlib.sha256(
                pool_path.with_suffix(".json.gz").read_bytes()
            ).hexdigest()
            eligible = [candidate for candidate in candidates if candidate["eligible"]]
            train_values = [panel_metric(item, "train", "auc") for item in eligible]
            validation_values = [item["validation_auc"] for item in eligible]
            final_values = [
                panel_metric(item, "train", "final_regret") for item in eligible
            ]
            if any(value is None for value in [*train_values, *final_values]):
                raise ValueError("Eligible candidate has invalid training")
            selected = min(
                eligible, key=lambda item: (item["validation_auc"], item["index"])
            )
            train_best = min(
                eligible,
                key=lambda item: (panel_metric(item, "train", "auc"), item["index"]),
            )
            inversions = sum(
                (train_values[i] - train_values[j])
                * (validation_values[i] - validation_values[j])
                < 0
                for i in range(len(eligible))
                for j in range(i)
            )
            final_auc_inversions = sum(
                (train_values[i] - train_values[j])
                * (final_values[i] - final_values[j])
                < 0
                for i in range(len(eligible))
                for j in range(i)
            )
            comparisons.append(
                {
                    "outer_seed": outer,
                    "arm": arm,
                    "eligible_pool_size_including_seed": len(eligible),
                    "train_validation_spearman": correlation(
                        ranks(train_values), ranks(validation_values)
                    ),
                    "train_validation_pairwise_inversions": inversions,
                    "pair_count_including_ties": len(eligible)
                    * (len(eligible) - 1)
                    // 2,
                    "train_auc_vs_final_regret_pairwise_inversions": final_auc_inversions,
                    "selected_index": selected["index"],
                    "train_best_index": train_best["index"],
                    "selected_train_auc": panel_metric(selected, "train", "auc"),
                    "train_best_train_auc": panel_metric(train_best, "train", "auc"),
                    "selected_train_rank": ranks(train_values)[
                        eligible.index(selected)
                    ],
                    "selected_validation_auc": selected["validation_auc"],
                    "candidates": [
                        {
                            "index": item["index"],
                            "source_sha256": item["source_sha256"],
                            "eligible": item["eligible"],
                            "train_auc": panel_metric(item, "train", "auc"),
                            "train_final_regret": panel_metric(
                                item, "train", "final_regret"
                            ),
                            "validation_auc": item["validation_auc"],
                        }
                        for item in candidates
                    ],
                }
            )
            by_hash = {item["source_sha256"]: item for item in candidates}
            seed_hash = candidates[0]["source_sha256"]
            depths = {seed_hash: 0}
            for slot, candidate in enumerate(candidates[1:]):
                path = ROOT / str(outer) / arm / f"slot_{slot:02d}" / "request.json"
                raw = read_json(path)
                source_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                row: dict[str, Any] = {"outer_seed": outer, "arm": arm, "slot": slot}
                joined = "\n".join(message["content"] for message in raw["messages"])
                row["prompt_characters"] = len(joined)
                row["prompt_mentions_auc_anytime_or_regret"] = any(
                    word in joined.lower() for word in ("auc", "anytime", "regret")
                )
                row["eligible"] = candidate["eligible"]
                if arm == "A2":
                    parent_hash = raw["parent_sha256"]
                    parent = by_hash[parent_hash]
                    text = raw["messages"][-1]["content"].split(
                        "\nTRAINING FEEDBACK:\n", 1
                    )[1]
                    reconstructed = {"current": feedback_from_rows(parent["train"])}
                    if slot:
                        reconstructed["previous_attempt"] = feedback_from_rows(
                            candidates[slot]["train"]
                        )
                    full_text = json.dumps(
                        reconstructed, sort_keys=True, allow_nan=False
                    )
                    if text != full_text[:12000]:
                        raise ValueError(
                            "Preserved request differs from reconstructed feedback"
                        )
                    feedback = json.loads(text)
                    row.update(
                        {
                            "parent_sha256": parent_hash,
                            "source_sha256": candidate["source_sha256"],
                            "parent_is_seed": parent_hash == seed_hash,
                            "parent_is_previous_source": bool(
                                slot
                                and parent_hash == candidates[slot]["source_sha256"]
                            ),
                            "feedback_characters": len(text),
                            "feedback_truncated": len(full_text) > len(text),
                            "current_tasks": len(feedback["current"]["tasks"]),
                            "current_observations_exposed": sum(
                                len(task["observations"])
                                for task in feedback["current"]["tasks"]
                            ),
                            "current_observations_available": sum(
                                len(task["observations"]) for task in parent["train"]
                            ),
                            "duplicated_current_previous_feedback": bool(
                                slot
                                and feedback["current"] == feedback["previous_attempt"]
                            ),
                            "previous_source_absent_when_different_from_parent": bool(
                                slot
                                and parent_hash != candidates[slot]["source_sha256"]
                                and candidates[slot]["source"] not in joined
                            ),
                            "parent_depth": depths[parent_hash],
                            "candidate_depth": (
                                0
                                if candidate["source_sha256"] == seed_hash
                                else depths[parent_hash] + 1
                            ),
                            "parent_train_auc": panel_metric(parent, "train", "auc"),
                            "candidate_train_auc": panel_metric(
                                candidate, "train", "auc"
                            ),
                        }
                    )
                    if candidate["source_sha256"] != seed_hash:
                        depths[candidate["source_sha256"]] = row["candidate_depth"]
                requests.append(row)
    a2 = [row for row in requests if row["arm"] == "A2"]
    return {
        "classification": "EXPLORATORY retrospective implementation diagnostics; no causal effect estimate",
        "reads_holdout": False,
        "outer_seeds": list(OUTER_SEEDS),
        "source_hashes": source_files,
        "requests": requests,
        "selection_comparisons": comparisons,
        "summary": {
            "requests": len(requests),
            "prompt_mentions_auc_anytime_or_regret": sum(
                row["prompt_mentions_auc_anytime_or_regret"] for row in requests
            ),
            "a2_feedback_truncated": sum(row["feedback_truncated"] for row in a2),
            "a2_feedback_character_range": [
                min(row["feedback_characters"] for row in a2),
                max(row["feedback_characters"] for row in a2),
            ],
            "a2_parent_is_seed": sum(row["parent_is_seed"] for row in a2),
            "a2_duplicated_current_previous_feedback": sum(
                row["duplicated_current_previous_feedback"] for row in a2
            ),
            "a2_previous_source_absent_when_different_from_parent": sum(
                row["previous_source_absent_when_different_from_parent"] for row in a2
            ),
            "a2_parent_depth_counts": dict(Counter(row["parent_depth"] for row in a2)),
            "a2_candidate_depth_counts": dict(
                Counter(row["candidate_depth"] for row in a2)
            ),
            "a2_eligible_candidate_depth_counts": dict(
                Counter(row["candidate_depth"] for row in a2 if row["eligible"])
            ),
            "selected_train_best_matches": sum(
                row["selected_index"] == row["train_best_index"] for row in comparisons
            ),
        },
    }


if __name__ == "__main__":
    result = analyze()
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
