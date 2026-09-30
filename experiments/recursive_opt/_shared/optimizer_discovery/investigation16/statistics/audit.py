"""Read-only, explicitly exploratory diagnostics of preserved EXP-15 evidence."""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist, correlation, mean, median, stdev
from typing import Any

RAW = Path(__file__).resolve().parents[2] / "exp15" / "raw"
OUT = Path(__file__).resolve().parent
SEEDS = (11, 23, 37, 41, 53)
ARMS = ("A0", "A1", "A2")
INPUT_HASHES: dict[str, str] = {}


def read(path: Path) -> Any:
    """Read preserved JSON, transparently handling gzip without rewriting it."""
    payload = path.read_bytes()
    INPUT_HASHES[str(path.relative_to(RAW))] = hashlib.sha256(payload).hexdigest()
    return json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)


def response_text_bytes(value: str | None) -> int:
    """Measure null response content as empty while preserving every response row."""
    return len((value or "").encode())


def ranks(values: list[float]) -> list[float]:
    """Return one-based average ranks, retaining exact ties."""
    return [
        1 + sum(other < value for other in values) + (values.count(value) - 1) / 2
        for value in values
    ]


def spearman(x: list[float], y: list[float]) -> float | None:
    """Compute descriptive rank correlation; constant vectors have no correlation."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Paired rank vectors require at least two equal-size samples")
    if len(set(x)) == 1 or len(set(y)) == 1:
        return None
    return correlation(ranks(x), ranks(y))


def signflip_p(values: list[float]) -> float:
    """Calculate the exact two-sided paired sign-flip descriptive p-value."""
    if not values or len(values) > 20 or not all(map(math.isfinite, values)):
        raise ValueError("Sign-flip enumeration requires 1–20 finite values")
    threshold = abs(mean(values)) - 1e-15
    null = [
        abs(mean(sign * value for sign, value in zip(signs, values)))
        for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    return sum(value >= threshold for value in null) / len(null)


def panel_auc(rows: list[dict[str, Any]]) -> float:
    """Aggregate an all-valid balanced panel, rejecting missing/invalid trajectories."""
    if not rows or not all(row["valid"] for row in rows):
        raise ValueError("Cannot aggregate an empty or invalid diagnostic panel")
    strata: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row["metrics"]["auc"])
    return mean(mean(values) for values in strata.values())


def selection_sensitivity(pool: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe six-task selection sensitivity; this is not a confidence interval."""
    eligible = [candidate for candidate in pool if candidate["eligible"]]
    selected = min(eligible, key=lambda c: (c["validation_auc"], c["index"]))
    train_best = min(eligible, key=lambda c: (panel_auc(c["train"]), c["index"]))
    train_scores = [panel_auc(c["train"]) for c in eligible]
    val_scores = [c["validation_auc"] for c in eligible]
    omitted: list[int] = []
    for missing in range(6):
        winner = min(
            eligible,
            key=lambda c: (
                mean(
                    row["metrics"]["auc"]
                    for j, row in enumerate(c["validation"])
                    if j != missing
                ),
                c["index"],
            ),
        )
        omitted.append(winner["index"])
    pairs = list(itertools.combinations(range(len(eligible)), 2))
    strict = [
        (train_scores[a] - train_scores[b]) * (val_scores[a] - val_scores[b])
        for a, b in pairs
        if train_scores[a] != train_scores[b] and val_scores[a] != val_scores[b]
    ]
    ordered = sorted(eligible, key=lambda c: (c["validation_auc"], c["index"]))
    return {
        "eligible_count_including_seed": len(eligible),
        "selected_index": selected["index"],
        "train_winner_index": train_best["index"],
        "selected_train_auc": panel_auc(selected["train"]),
        "selected_validation_auc": selected["validation_auc"],
        "selected_train_rank": ranks(train_scores)[eligible.index(selected)],
        "train_winner_validation_rank": ranks(val_scores)[eligible.index(train_best)],
        "train_validation_spearman": spearman(train_scores, val_scores),
        "pairwise_order_agreement": (
            mean(value > 0 for value in strict) if strict else None
        ),
        "validation_gap_to_runner_up": ordered[1]["validation_auc"]
        - ordered[0]["validation_auc"],
        "leave_one_stratum_out_winners": omitted,
        "leave_one_stratum_out_changes": sum(
            index != selected["index"] for index in omitted
        ),
        "candidates": [
            {
                "index": candidate["index"],
                "source_sha256": candidate["source_sha256"],
                "eligible": candidate["eligible"],
                "train_auc": (
                    panel_auc(candidate["train"])
                    if all(row["valid"] for row in candidate["train"])
                    else None
                ),
                "validation_auc": candidate["validation_auc"],
                "train_strata": {
                    row["stratum"]: row["metrics"]["auc"] if row["metrics"] else None
                    for row in candidate["train"]
                },
                "validation_strata": {
                    row["stratum"]: row["metrics"]["auc"] if row["metrics"] else None
                    for row in candidate["validation"]
                },
            }
            for candidate in pool
        ],
    }


def contrasts(result: dict[str, Any]) -> dict[str, Any]:
    """Compute conditional paired uncertainty and explicitly approximate planning quantities."""
    output = {}
    for control in ("A0", "A1"):
        values = [row["A2"]["auc"] - row[control]["auc"] for row in result["per_seed"]]
        sd = stdev(values)
        normal_factor = (NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)) ** 2
        output[f"A2-{control}"] = {
            "paired_values": values,
            "mean": mean(values),
            "median": median(values),
            "sample_sd": sd,
            "standard_error": sd / math.sqrt(len(values)),
            "registered_bootstrap_percentile_95_from_raw": result["contrasts"][
                f"A2-{control}"
            ]["paired_bootstrap_95"],
            "exploratory_signflip_two_sided_p": signflip_p(values),
            "leave_one_outer_seed_out_means": [
                mean(values[:i] + values[i + 1 :]) for i in range(len(values))
            ],
            "normal_approximate_n_80pct_power_two_sided_5pct": {
                str(effect): math.ceil(normal_factor * sd**2 / effect**2)
                for effect in (0.005, 0.01, 0.02, 0.03)
            },
            "planning_caveat": "Exploratory normal approximation from only five seeds; not a guaranteed power estimate or prospective effect prediction.",
        }
    return output


def main() -> None:
    """Write diagnostics derived solely from immutable EXP-15 raw evidence."""
    result = read(RAW / "results.json")
    selections = []
    responses = []
    feedback = []
    holdout = []
    lineage = []
    for outer in SEEDS:
        raw_holdout = read(RAW / str(outer) / "holdout.json.gz")
        for arm in ARMS:
            for row in raw_holdout[arm]:
                curve = row["metrics"]["curve"]
                holdout.append(
                    {
                        "outer_seed": outer,
                        "arm": arm,
                        "task_identity": row["task_identity"],
                        "stratum": row["stratum"],
                        "auc": row["metrics"]["auc"],
                        "final_regret": row["metrics"]["final_regret"],
                        "early_auc_contribution": {
                            str(n): sum(curve[:n]) / 32 for n in (1, 4, 8, 16)
                        },
                    }
                )
        for arm in ("A1", "A2"):
            pool = read(RAW / str(outer) / arm / "pool.json.gz")
            selections.append(
                {"outer_seed": outer, "arm": arm, **selection_sensitivity(pool)}
            )
            by_hash = {c["source_sha256"]: c for c in pool if c["eligible"]}
            depths = {pool[0]["source_sha256"]: 0}
            for slot in range(8):
                directory = RAW / str(outer) / arm / f"slot_{slot:02d}"
                response = read(directory / "response.json")
                request = read(directory / "request.json")
                provider = read(directory / "provider_generation.json")
                responses.append(
                    {
                        "outer_seed": outer,
                        "arm": arm,
                        "slot": slot,
                        "provider": provider["provider_name"],
                        "finish_reason": response["finish_reason"],
                        "parse_status": response["parse_status"],
                        "source_status": response["source_status"],
                        "eligible": pool[slot + 1]["eligible"],
                        "source_sha256": response["source_sha256"],
                        "content_bytes": response_text_bytes(response["content"]),
                        "source_bytes": response_text_bytes(response["source"]),
                        "completion_tokens": response["usage"]["completion_tokens"],
                        "prompt_tokens": response["usage"]["prompt_tokens"],
                        "reasoning_tokens": response["usage"]["reasoning_tokens"],
                        "visible_completion_tokens_estimate": response["usage"][
                            "completion_tokens"
                        ]
                        - response["usage"]["reasoning_tokens"],
                        "cost_usd": response["usage"]["cost_usd"],
                    }
                )
                if arm == "A2":
                    parent = request["parent_sha256"]
                    child = pool[slot + 1]
                    depth = depths.get(parent, 0) + 1
                    if child["source_sha256"] not in depths:
                        depths[child["source_sha256"]] = depth
                    lineage.append(
                        {
                            "outer_seed": outer,
                            "slot": slot,
                            "parent_sha256": parent,
                            "source_sha256": child["source_sha256"],
                            "depth_if_new": depth,
                            "parent_is_seed": parent == pool[0]["source_sha256"],
                            "child_is_seed": child["source_sha256"]
                            == pool[0]["source_sha256"],
                            "eligible": child["eligible"],
                            "train_auc_delta_from_parent": (
                                panel_auc(child["train"])
                                - panel_auc(by_hash[parent]["train"])
                                if child["eligible"]
                                else None
                            ),
                            "validation_auc_delta_from_parent": (
                                child["validation_auc"]
                                - by_hash[parent]["validation_auc"]
                                if child["eligible"]
                                else None
                            ),
                        }
                    )
                    text = request["messages"][1]["content"].split(
                        "TRAINING FEEDBACK:\n", 1
                    )[1]
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        feedback.append(
                            {
                                "outer_seed": outer,
                                "slot": slot,
                                "json_valid": False,
                                "characters": len(text),
                            }
                        )
                        continue
                    panels = {}
                    for label, panel in payload.items():
                        if panel is None:
                            continue
                        tasks = panel["tasks"]
                        valid = [task for task in tasks if task["valid"]]
                        panels[label] = {
                            "tasks": len(tasks),
                            "observations": sum(
                                len(task["observations"]) for task in tasks
                            ),
                            "valid_tasks": len(valid),
                            "actual_incumbent_missing_from_shown_observations": sum(
                                not any(
                                    obs["value"] == task["best_observed_value"]
                                    for obs in task["observations"]
                                )
                                for task in valid
                            ),
                            "raw_best_value_max_min_ratio": (
                                max(task["best_observed_value"] for task in valid)
                                / min(task["best_observed_value"] for task in valid)
                                if valid
                                and min(task["best_observed_value"] for task in valid)
                                > 0
                                else None
                            ),
                        }
                    feedback.append(
                        {
                            "outer_seed": outer,
                            "slot": slot,
                            "json_valid": True,
                            "characters": len(text),
                            "panels": panels,
                        }
                    )
    strata = {}
    for stratum in sorted({row["stratum"] for row in holdout}):
        strata[stratum] = {
            arm: mean(
                row["auc"]
                for row in holdout
                if row["arm"] == arm and row["stratum"] == stratum
            )
            for arm in ARMS
        }
        strata[stratum]["A2-A1"] = strata[stratum]["A2"] - strata[stratum]["A1"]
    provider_groups = {}
    for name in sorted({row["provider"] for row in responses}):
        rows = [row for row in responses if row["provider"] == name]
        provider_groups[name] = {
            "responses": len(rows),
            "eligible": sum(row["eligible"] for row in rows),
            "arm_counts": dict(Counter(row["arm"] for row in rows)),
            "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
        }
    local_variation = {}
    for split in ("train", "validation"):
        values = [
            next(c for c in item["candidates"] if c["index"] == -1)[f"{split}_auc"]
            for item in selections
            if item["arm"] == "A1"
        ]
        local_variation[split] = {
            "seed_policy_auc": values,
            "mean": mean(values),
            "sd": stdev(values),
            "range": max(values) - min(values),
        }
    early = {
        arm: {
            str(n): mean(
                row["early_auc_contribution"][str(n)]
                for row in holdout
                if row["arm"] == arm
            )
            for n in (1, 4, 8, 16)
        }
        for arm in ARMS
    }
    output = {
        "status": "EXPLORATORY_RETROSPECTIVE_ONLY",
        "new_model_calls": 0,
        "new_objective_calls": 0,
        "changed_exp15_artifacts": 0,
        "contrasts": contrasts(result),
        "selections": selections,
        "responses": responses,
        "feedback": feedback,
        "lineage": lineage,
        "holdout_rows": holdout,
        "stratum_means": strata,
        "provider_groups": provider_groups,
        "seed_policy_local_randomness_variation": local_variation,
        "early_auc_contributions": early,
        "input_sha256": INPUT_HASHES,
    }
    (OUT / "diagnostics.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
